"""Download the persona query dataset and turn it into training batches.

    python src/dataset.py        # download the data to data/ and print the split sizes

The query dataset (EdwardHuy/persona-queries) holds three short queries per person.
The full profile each query should retrieve is rebuilt from the same person in
Nemotron-Personas-Belgium. People are split into train / validation / test by a hash
of their UUID, identically for both languages, so no person (or the translation of a
person) appears in two splits.
"""

import hashlib
import random
from pathlib import Path

DATASET_ID = "EdwardHuy/persona-queries"
NEMOTRON = "nvidia/Nemotron-Personas-Belgium"
DATA_DIR = Path(__file__).parent.parent / "data"
TEST_PERSONS, VALIDATION_PERSONS = 400, 200


def load_queries(language: str) -> dict[str, list[str]]:
    """{uuid: [query, ...]} for one language; downloaded to data/ on first use."""
    from datasets import load_dataset
    rows = load_dataset(DATASET_ID, split=f"{language}_BE", cache_dir=str(DATA_DIR / "cache"))
    return {row["uuid"]: [row[f"query_{i}"] for i in (1, 2, 3) if row[f"query_{i}"]] for row in rows}


def load_profiles(language: str, uuids: set[str]) -> dict[str, str]:
    """The full Nemotron persona of each UUID as labelled lines, as shown to the query generator."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(NEMOTRON, f"data/{language}_BE.parquet", repo_type="dataset",
                           local_dir=DATA_DIR / "nemotron")
    profiles = {}
    # Read in batches and keep only the needed rows: the full file would take about 6 GB of RAM.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        for p in batch.to_pylist():
            if p["uuid"] in uuids:
                profiles[p["uuid"]] = "\n".join(f"[{field}] {value}" for field, value in p.items() if field != "uuid")
    return profiles


def load_split(language: str) -> dict[str, dict[str, dict]]:
    """{"train" | "validation" | "test": {uuid: {"queries": [...], "profile": str}}}."""
    queries = {lang: load_queries(lang) for lang in ("nl", "en")}
    # Rank the UUIDs of both languages together, so a person lands in the same split in both.
    ranked = sorted(set(queries["nl"]) | set(queries["en"]),
                    key=lambda uuid: hashlib.sha256(f"embedding-split:{uuid}".encode()).digest())
    split_of = {uuid: "test" if i < TEST_PERSONS else "validation" if i < TEST_PERSONS + VALIDATION_PERSONS
                else "train" for i, uuid in enumerate(ranked)}
    profiles = load_profiles(language, set(queries[language]))
    splits = {"train": {}, "validation": {}, "test": {}}
    for uuid, person_queries in queries[language].items():
        splits[split_of[uuid]][uuid] = {"queries": person_queries, "profile": profiles[uuid]}
    return splits


def balanced_batches(persons: dict[str, dict], batch_size: int, rng: random.Random):
    """Yield (anchors, positives) batches for one epoch.

    Every person contributes each pair of their queries once per epoch ((q1,q2), (q1,q3),
    (q2,q3)), one pair per round, so all persons weigh equally. Within a round a person
    appears once, so a batch never holds two pairs of the same person: every in-batch
    negative belongs to someone else.
    """
    plans = {}
    for uuid, person in persons.items():
        q = person["queries"]
        pairs = [(q[i], q[j]) for i in range(len(q)) for j in range(i + 1, len(q))]
        rng.shuffle(pairs)
        plans[uuid] = [pair if rng.random() < 0.5 else pair[::-1] for pair in pairs]
    for round_index in range(max(len(plan) for plan in plans.values())):
        uuids = [uuid for uuid in plans if round_index < len(plans[uuid])]
        rng.shuffle(uuids)
        for start in range(0, len(uuids), batch_size):
            chunk = uuids[start:start + batch_size]
            if len(chunk) < batch_size // 4:
                continue  # A tiny final batch has too few negatives to be useful.
            yield [plans[u][round_index][0] for u in chunk], [plans[u][round_index][1] for u in chunk]


if __name__ == "__main__":
    for language in ("nl", "en"):
        print(language, {name: len(persons) for name, persons in load_split(language).items()})
