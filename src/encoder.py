"""Text encoders: Qwen3-Embedding (optionally with a LoRA adapter) and the Dutch E5 baseline."""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

QWEN_ID = "Qwen/Qwen3-Embedding-0.6B"
E5_NL_ID = "clips/e5-large-trm-nl"

# Qwen3-Embedding expects an instruction in front of each query.
Q2Q_INSTRUCTION = ("Given a short description of a person's interests, work or background, "
                   "retrieve other descriptions of the same person")
Q2D_INSTRUCTION = "Given a short description of a person, retrieve the full profile of that person"
QUERY_MAX_LENGTH = 128
PROFILE_MAX_LENGTH = 2304  # Full profiles are ~1.2k (en) to ~2.2k (nl) Qwen tokens.


class Encoder:
    """Qwen3-Embedding uses last-token pooling, E5 mean pooling. Embeddings are L2-normalised."""

    def __init__(self, model_id: str = QWEN_ID, adapter: Path | None = None, model=None):
        self.kind = "e5" if "e5" in model_id.lower() else "qwen"
        self.dtype = torch.bfloat16 if self.kind == "qwen" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left" if self.kind == "qwen" else "right")
        if model is None:  # During training the model being trained is passed in instead.
            model = AutoModel.from_pretrained(model_id, dtype=self.dtype)
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter))
            model = model.cuda().eval()
        self.model = model

    def tokenize(self, texts: list[str], max_length: int) -> dict:
        if self.kind == "e5":
            batch = self.tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            return {k: v.cuda() for k, v in batch.items()}
        # The tokenizer ends every text with <|endoftext|>; the end-of-sequence token <|im_end|>
        # is appended after truncation, and its hidden state is the embedding. The published
        # Fishstick models were trained with this exact ending.
        eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token or "<|endoftext|>")
        ids = [x + [eos] for x in self.tokenizer(texts, truncation=True, max_length=max_length - 1)["input_ids"]]
        width = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        return {"input_ids": torch.tensor([[pad] * (width - len(x)) + x for x in ids]).cuda(),
                "attention_mask": torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids]).cuda()}

    def forward(self, texts: list[str], max_length: int) -> torch.Tensor:
        """Differentiable embeddings, one row per text."""
        batch = self.tokenize(texts, max_length)
        with torch.autocast("cuda", dtype=self.dtype):
            hidden = self.model(**batch).last_hidden_state
        if self.kind == "qwen":
            pooled = hidden[:, -1]  # Left padding: the last position is every text's final token.
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def format(self, texts: list[str], role: str, instruction: str | None = None) -> list[str]:
        """Add the prefix each model expects: an instruction (Qwen) or 'query: '/'passage: ' (E5)."""
        if self.kind == "e5":
            return [("query: " if role == "query" else "passage: ") + t for t in texts]
        if role == "query":
            return [f"Instruct: {instruction}\nQuery:{t}" for t in texts]
        return list(texts)

    def encode(self, texts: list[str], *, role: str, instruction: str | None = None,
               max_length: int = QUERY_MAX_LENGTH, batch_size: int = 64) -> torch.Tensor:
        """Embeddings without gradients, in the order of `texts`."""
        texts = self.format(texts, role, instruction)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # Similar lengths: less padding.
        out = [None] * len(texts)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                for i, vector in zip(indices, self.forward([texts[i] for i in indices], max_length)):
                    out[i] = vector
        return torch.stack(out)
