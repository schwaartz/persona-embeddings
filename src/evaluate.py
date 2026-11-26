"""Retrieval metrics on the held-out test persons, and a table of all results.

    python src/evaluate.py --baselines   # evaluate untuned Qwen3-Embedding and E5-NL, then print the table
    python src/evaluate.py               # print the table of all results in output/

Two tasks are measured for every test query:
  q2q: find the person's other queries among all test queries.
  q2d: find the person's full profile among all test profiles.
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def ranking_metrics(scores, relevant_sets, k=10):
    """Per-query MRR@10, Acc@1, Recall@10 and nDCG@10 from a query x candidate score matrix."""
    rows = []
    k = min(k, scores.shape[1])
    for ranking, relevant in zip(scores.topk(k, dim=1).indices.tolist(), relevant_sets):
        hits = [candidate in relevant for candidate in ranking]
        dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(hits) if hit)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        rows.append({"mrr@10": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0),
                     "acc@1": float(hits[0]), "recall@10": sum(hits) / len(relevant), "ndcg@10": dcg / ideal})
    return rows


def evaluate(encoder, persons: dict[str, dict], with_profiles: bool = True) -> dict[str, list[dict]]:
    """Per-query metric rows for q2q (and q2d), each tagged with the person's UUID."""
    import torch
    from encoder import PROFILE_MAX_LENGTH, Q2D_INSTRUCTION, Q2Q_INSTRUCTION
    queries = [q for person in persons.values() for q in person["queries"]]
    owners = [uuid for uuid, person in persons.items() for _ in person["queries"]]
    result = {}

    # The task is symmetric, so both sides of a query-query comparison get the same instruction.
    embedded = encoder.encode(queries, role="query", instruction=Q2Q_INSTRUCTION)
    scores = embedded @ embedded.T
    scores.fill_diagonal_(-1e4)  # A query must not find itself.
    relevant = [{j for j, other in enumerate(owners) if other == owner and j != i} for i, owner in enumerate(owners)]
    result["q2q"] = [dict(row, uuid=owner) for row, owner in zip(ranking_metrics(scores, relevant), owners)]

    if with_profiles:
        embedded = encoder.encode(queries, role="query", instruction=Q2D_INSTRUCTION)
        profiles = encoder.encode([person["profile"] for person in persons.values()], role="document",
                                  max_length=PROFILE_MAX_LENGTH if encoder.kind == "qwen" else 512, batch_size=8)
        index = {uuid: i for i, uuid in enumerate(persons)}
        rows = ranking_metrics(embedded @ profiles.T, [{index[owner]} for owner in owners])
        result["q2d"] = [dict(row, uuid=owner) for row, owner in zip(rows, owners)]
    torch.cuda.empty_cache()
    return result


def summarize(rows: list[dict], bootstrap: int = 1000, seed: int = 0) -> dict:
    """Mean of every metric, with a 95% interval from resampling persons (not queries)."""
    by_person = {}
    for row in rows:
        by_person.setdefault(row["uuid"], []).append(row)
    metrics = [key for key in rows[0] if key != "uuid"]
    summary = {m: sum(row[m] for row in rows) / len(rows) for m in metrics}
    rng = random.Random(seed)
    samples = {m: [] for m in metrics}
    for _ in range(bootstrap):
        chosen = [row for person in rng.choices(list(by_person), k=len(by_person)) for row in by_person[person]]
        for m in metrics:
            samples[m].append(sum(row[m] for row in chosen) / len(chosen))
    for m in metrics:
        values = sorted(samples[m])
        summary[m + "_ci"] = [values[int(0.025 * bootstrap)], values[int(0.975 * bootstrap) - 1]]
    summary["queries"], summary["persons"] = len(rows), len(by_person)
    return summary


def evaluate_baselines():
    import torch
    from dataset import load_split
    from encoder import E5_NL_ID, QWEN_ID, Encoder
    for model_id, short_name in ((QWEN_ID, "qwen-base"), (E5_NL_ID, "e5-nl")):
        encoder = Encoder(model_id)
        for language in ("nl", "en"):
            rows = evaluate(encoder, load_split(language)["test"])
            result = {"name": f"{short_name}-{language}", "model": model_id, "language": language,
                      "test": {task: summarize(r) for task, r in rows.items()}}
            path = OUTPUT_DIR / result["name"] / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(result["name"], {task: round(s["mrr@10"], 4) for task, s in result["test"].items()}, flush=True)
        del encoder
        torch.cuda.empty_cache()


def print_table():
    """One row per model and language; runs that differ only in seed are averaged (mean ± SD)."""
    groups = {}
    for path in sorted(OUTPUT_DIR.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        model = result["name"].rsplit("-seed", 1)[0].removesuffix(f"-{result['language']}")
        groups.setdefault((result["language"], model), []).append(result)
    print("| Language | Model | Seeds | q2q MRR@10 | q2q Acc@1 | q2q Recall@10 | q2d MRR@10 | q2d Acc@1 | q2d Recall@10 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (language, model), runs in sorted(groups.items()):
        cells = []
        for task in ("q2q", "q2d"):
            for metric in ("mrr@10", "acc@1", "recall@10"):
                values = [run["test"][task][metric] for run in runs]
                if len(runs) == 1:
                    low, high = runs[0]["test"][task][metric + "_ci"]
                    cells.append(f"{values[0]:.3f} [{low:.3f}, {high:.3f}]")
                else:
                    cells.append(f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}")
        print(f"| {language} | {model} | {len(runs)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", action="store_true", help="Evaluate the untuned baseline models first.")
    if parser.parse_args().baselines:
        evaluate_baselines()
    print_table()
