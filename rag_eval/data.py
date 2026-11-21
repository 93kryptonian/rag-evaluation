import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(name: str) -> list[dict]:
    with open(DATA / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_docs() -> list[dict]:
    return load_jsonl("docs.jsonl")


def load_golden_set() -> list[dict]:
    return load_jsonl("golden_set.jsonl")
