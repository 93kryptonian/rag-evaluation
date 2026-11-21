"""Step 3 of the article: generate answers with an LLM and grade them with an LLM judge.

    python scripts/eval_generation.py --retriever embeddings   # default
    python scripts/eval_generation.py --retriever bm25         # see how bad retrieval hurts answers

Needs OPENAI_API_KEY in .env. About 80 API calls per run (40 answers + 40 judgments).
Writes results/generation_<retriever>.json and prints a summary.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.data import load_docs, load_golden_set  # noqa: E402
from rag_eval.llm import JUDGE_MODEL, GENERATOR_MODEL, generate_answer, judge  # noqa: E402
from rag_eval.metrics import mean  # noqa: E402
from rag_eval.retrievers import BM25Retriever, EmbeddingRetriever  # noqa: E402

TOP_K = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["embeddings", "bm25"], default="embeddings")
    args = ap.parse_args()

    docs = load_docs()
    by_id = {d["id"]: d for d in docs}
    golden = load_golden_set()
    retriever = EmbeddingRetriever(docs) if args.retriever == "embeddings" else BM25Retriever(docs)

    def run_one(q):
        ctx = [by_id[i] for i in retriever.search(q["question"], k=TOP_K)]
        answer = generate_answer(q["question"], ctx)
        verdict = judge(q["question"], ctx, answer, q["reference_answer"])
        return {**q, "retrieved": [d["id"] for d in ctx],
                "retrieval_hit": any(d["id"] in q["relevant_ids"] for d in ctx),
                "answer": answer, "verdict": verdict.model_dump() if verdict else None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_one, golden))

    judged = [r for r in rows if r["verdict"]]
    answerable = [r for r in judged if r["relevant_ids"]]
    unanswerable = [r for r in judged if not r["relevant_ids"]]
    summary = {
        "retriever": args.retriever, "generator": GENERATOR_MODEL, "judge": JUDGE_MODEL,
        "n": len(rows), "not_judged": len(rows) - len(judged),
        "faithfulness": mean(r["verdict"]["faithful"] for r in judged),
        "answer_relevance": mean(r["verdict"]["answers_question"] for r in judged),
        "correct": mean(r["verdict"]["correct"] == "correct" for r in answerable),
        "correct_when_retrieval_hit": mean(r["verdict"]["correct"] == "correct" for r in answerable if r["retrieval_hit"]),
        "correct_when_retrieval_miss": mean(r["verdict"]["correct"] == "correct" for r in answerable if not r["retrieval_hit"]),
        "retrieval_misses": sum(not r["retrieval_hit"] for r in answerable),
        "abstained_on_unanswerable": mean("don't know" in r["answer"].lower() for r in unanswerable),
    }

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / f"generation_{args.retriever}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nAnswers judged not faithful or incorrect:")
    for r in judged:
        v = r["verdict"]
        if not v["faithful"] or (r["relevant_ids"] and v["correct"] != "correct"):
            print(f"- {r['question']}\n  answer: {r['answer']}\n  judge: {v['correct']}, faithful={v['faithful']} — {v['reasoning'][:200]}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
