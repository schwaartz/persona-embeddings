"""LoRA contrastive fine-tuning of Qwen3-Embedding-0.6B on the persona queries.

    python src/train.py --language nl --epochs 5 --seed 42
    python src/train.py --language nl --objective mixed --epochs 1 --seed 42

Objective "q2q" (default): the positive for a query is another query of the same
person; the queries of the other persons in the batch are negatives (symmetric InfoNCE).
Objective "mixed" adds the person's full profile as a positive for each query, with the
other persons' profiles as negatives. Long profiles do not fit in memory all at once,
so their gradients are computed in small chunks ("gradient caching").

After every epoch the adapter is scored on the validation persons; the best one is kept
and finally evaluated on the test persons. Everything is written to output/<name>/.
"""

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, get_linear_schedule_with_warmup

from dataset import balanced_batches, load_split
from encoder import PROFILE_MAX_LENGTH, Q2D_INSTRUCTION, Q2Q_INSTRUCTION, QUERY_MAX_LENGTH, QWEN_ID, Encoder
from evaluate import OUTPUT_DIR, evaluate, summarize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=("nl", "en"), required=True)
    parser.add_argument("--objective", choices=("q2q", "mixed"), default="q2q")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128, help="Distinct persons per batch.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, help="LoRA dropout (default: 0.05 for q2q, 0 for mixed).")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--profile-chunk", type=int, default=4, help="Profiles per gradient-caching chunk.")
    parser.add_argument("--name", help="Run name (default: qwen-lora[-mixed]-<language>-seed<seed>).")
    args = parser.parse_args()
    if args.dropout is None:
        # Gradient caching embeds each profile twice; dropout would make the two passes differ.
        args.dropout = 0.0 if args.objective == "mixed" else 0.05
    if args.name is None:
        args.name = f"qwen-lora{'-mixed' if args.objective == 'mixed' else ''}-{args.language}-seed{args.seed}"
    return args


def contrastive_loss(anchors, positives, temperature):
    """InfoNCE: each anchor must pick its own positive out of the batch."""
    labels = torch.arange(len(anchors), device=anchors.device)
    return F.cross_entropy(anchors @ positives.T / temperature, labels)


def main():
    args = parse_args()
    out_dir = OUTPUT_DIR / args.name
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = load_split(args.language)
    print({name: len(persons) for name, persons in splits.items()}, flush=True)
    mixed = args.objective == "mixed"

    base = AutoModel.from_pretrained(QWEN_ID, dtype=torch.bfloat16)
    # Gradient checkpointing keeps a 128-person batch (256+ sequences) within 10 GB of VRAM.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()  # LoRA weights in fp32, frozen base in bf16.
    model.cuda()
    model.print_trainable_parameters()
    encoder = Encoder(QWEN_ID, model=model)

    rng = random.Random(args.seed)
    steps_per_epoch = sum(1 for _ in balanced_batches(splits["train"], args.batch_size, random.Random(0)))
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    profile_of = {q: person["profile"] for person in splits["train"].values() for q in person["queries"]}

    def validation_score():
        """Mean validation MRR@10 over the trained task(s)."""
        model.eval()
        rows = evaluate(encoder, splits["validation"], with_profiles=mixed)
        model.train()
        scores = {f"val_{task}_mrr@10": summarize(r, bootstrap=1)["mrr@10"] for task, r in rows.items()}
        return {**scores, "val_score": sum(scores.values()) / len(scores)}

    history = [{"epoch": 0, **validation_score()}]
    print(history[-1], flush=True)
    best_score, best_epoch = history[0]["val_score"], 0
    model.save_pretrained(out_dir)  # Epoch 0 is the untouched base model, in case training never helps.
    started = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for step, (anchors, positives) in enumerate(balanced_batches(splits["train"], args.batch_size, rng)):
            if mixed:
                # Gradient caching, pass 1: embed the profiles without keeping the graph.
                profiles = [profile_of[a] for a in anchors]
                chunks = range(0, len(profiles), args.profile_chunk)
                with torch.no_grad():
                    cached = torch.cat([encoder.forward(profiles[i:i + args.profile_chunk], PROFILE_MAX_LENGTH)
                                        for i in chunks])
                cached.requires_grad_()

            a = encoder.forward(encoder.format(anchors, "query", Q2Q_INSTRUCTION), QUERY_MAX_LENGTH)
            p = encoder.forward(encoder.format(positives, "query", Q2Q_INSTRUCTION), QUERY_MAX_LENGTH)
            loss = (contrastive_loss(a, p, args.temperature) + contrastive_loss(p, a, args.temperature)) / 2
            if mixed:
                qa = encoder.forward(encoder.format(anchors, "query", Q2D_INSTRUCTION), QUERY_MAX_LENGTH)
                qp = encoder.forward(encoder.format(positives, "query", Q2D_INSTRUCTION), QUERY_MAX_LENGTH)
                loss = loss + (contrastive_loss(qa, cached, args.temperature)
                               + contrastive_loss(qp, cached, args.temperature)) / 2
            loss.backward()
            if mixed:
                # Pass 2: re-embed each chunk of profiles with a graph and push the cached gradient through it.
                for i in chunks:
                    encoder.forward(profiles[i:i + args.profile_chunk], PROFILE_MAX_LENGTH).backward(
                        cached.grad[i:i + args.profile_chunk])

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)

        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), **validation_score(),
                        "minutes": round((time.time() - started) / 60, 2)})
        print(history[-1], flush=True)
        if history[-1]["val_score"] > best_score:
            best_score, best_epoch = history[-1]["val_score"], epoch
            model.save_pretrained(out_dir)

    # Test the best adapter, loaded fresh from disk.
    del model, encoder, base
    torch.cuda.empty_cache()
    rows = evaluate(Encoder(QWEN_ID, adapter=out_dir), splits["test"])
    result = {"name": args.name, "model": QWEN_ID, "language": args.language, "objective": args.objective,
              "best_epoch": best_epoch, "config": vars(args), "history": history,
              "split_sizes": {name: len(persons) for name, persons in splits.items()},
              "test": {task: summarize(r, seed=args.seed) for task, r in rows.items()}}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({task: round(s["mrr@10"], 4) for task, s in result["test"].items()}), flush=True)


if __name__ == "__main__":
    main()
