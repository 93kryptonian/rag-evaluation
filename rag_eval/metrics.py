"""Retrieval metrics. Each takes the retrieved ids (in rank order) and the relevant ids."""


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top k, else 0.0."""
    return float(any(doc in relevant for doc in retrieved[:k]))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of the relevant docs that appear in the top k."""
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / position of the first relevant doc (1.0 = ranked first, 0 = not found)."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
