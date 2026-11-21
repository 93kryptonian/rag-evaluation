"""Step 2 of the article: score three retrievers on the golden set. Runs locally, no API key.

    python scripts/eval_retrieval.py

Writes results/retrieval.json and prints a summary table.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.data import load_docs, load_golden_set  # noqa: E402
from rag_eval.metrics import hit_rate_at_k, mean, recall_at_k, reciprocal_rank  # noqa: E402
from rag_eval.retrievers import BM25Retriever, EmbeddingRetriever, HybridRetriever  # noqa: E402

KS = [1, 3, 5]


def evaluate(retriever, questions):
    rows = []
    for q in questions:
        retrieved = retriever.search(q["question"], k=10)
        row = {"id": q["id"], "type": q["type"], "retrieved": retrieved[:5],
               "rr": reciprocal_rank(retrieved, q["relevant_ids"])}
        for k in KS:
            row[f"hit@{k}"] = hit_rate_at_k(retrieved, q["relevant_ids"], k)
            row[f"recall@{k}"] = recall_at_k(retrieved, q["relevant_ids"], k)
        rows.append(row)
    return rows


def summarize(rows):
    out = {"n": len(rows), "mrr": mean(r["rr"] for r in rows)}
    for k in KS:
        out[f"hit@{k}"] = mean(r[f"hit@{k}"] for r in rows)
    return out


def main():
    docs = load_docs()
    # Retrieval metrics only make sense for questions that HAVE a relevant document.
    questions = [q for q in load_golden_set() if q["relevant_ids"]]

    bm25 = BM25Retriever(docs)
    emb = EmbeddingRetriever(docs)
    retrievers = [bm25, emb, HybridRetriever(bm25, emb)]

    results = {}
    for r in retrievers:
        rows = evaluate(r, questions)
        results[r.name] = {
            "overall": summarize(rows),
            "by_type": {t: summarize([x for x in rows if x["type"] == t]) for t in ("keyword", "paraphrase")},
            "rows": rows,
        }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "retrieval.json").write_text(json.dumps(results, indent=2))

    print(f"{len(questions)} answerable questions\n")
    print(f"{'retriever':24} {'subset':11} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}")
    for name, res in results.items():
        for subset, s in [("all", res["overall"]), *res["by_type"].items()]:
            print(f"{name:24} {subset:11} {s['hit@1']:6.2f} {s['hit@3']:6.2f} {s['hit@5']:6.2f} {s['mrr']:6.2f}")

    print("\nMisses at k=3 (question -> what we retrieved instead):")
    gold = {q["id"]: q for q in questions}
    for name, res in results.items():
        for row in res["rows"]:
            if not row["hit@3"]:
                q = gold[row["id"]]
                print(f"  [{name}] {q['question']!r} wanted {q['relevant_ids']} got {row['retrieved'][:3]}")


if __name__ == "__main__":
    main()
