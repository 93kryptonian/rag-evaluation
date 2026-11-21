# RAG Evaluation in Practice

Companion code for the Medium article **[Stop Eyeballing Your RAG: A Practical Guide to Evaluating It](https://medium.com/@93Kryptonian/stop-eyeballing-your-rag-a-practical-guide-to-evaluating-it-b091a664ed41)**.

A small, self-contained project that shows how to evaluate a Retrieval-Augmented Generation (RAG) pipeline in two separate parts:

1. **Retrieval**: did we find the right document? (hit rate@k, recall@k, MRR). Runs fully locally, no API key.
2. **Generation**: is the answer faithful to the retrieved context, relevant and correct? (LLM-as-judge via the OpenAI API)

The knowledge base is a **synthetic help center for a fictional app, "Lumen Notes"** (24 articles, 40 test questions). No real customer or personal data is used.

## Project layout

```
data/
  build_data.py        # creates the synthetic docs + golden test set
  docs.jsonl           # 24 help-center articles
  golden_set.jsonl     # 40 questions: keyword-style, paraphrased, unanswerable
rag_eval/
  retrievers.py        # BM25 (keyword), embeddings (semantic), hybrid (RRF)
  metrics.py           # hit_rate_at_k, recall_at_k, reciprocal_rank
  llm.py               # answer generation + LLM judge (OpenAI Responses API, structured output)
scripts/
  eval_retrieval.py    # step 2: score the retrievers
  eval_generation.py   # step 3: generate answers and judge them
  make_figures.py      # figures used in the article
results/               # JSON outputs from the scripts
figures/               # generated charts and diagrams
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/build_data.py          # build the dataset
python scripts/eval_retrieval.py   # retrieval metrics (local, no key needed)
```

To run the generation evaluation, copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env               # then edit .env
python scripts/eval_generation.py --retriever embeddings
python scripts/eval_generation.py --retriever bm25   # compare: weak retrieval -> worse answers
```

Each generation run makes about 80 API calls (40 answers + 40 judgments). Models default to `gpt-5.4-mini` (generator) and `gpt-5.5` (judge) and can be changed with `GENERATOR_MODEL` / `JUDGE_MODEL` in `.env`. The approach works with any LLM provider; only `rag_eval/llm.py` needs to change.

## Retrieval results (from `scripts/eval_retrieval.py`)

36 answerable questions (18 keyword-style, 18 paraphrased), each doc treated as one chunk:

| Retriever | Hit@3 keyword | Hit@3 paraphrased | MRR (all) |
|---|---|---|---|
| BM25 (keyword) | 100% | 50% | 0.71 |
| Embeddings (`all-MiniLM-L6-v2`) | 100% | 94% | 0.92 |
| Hybrid (RRF of both) | 100% | 78% | 0.80 |

## Generation results (from `scripts/eval_generation.py`)

Generator `gpt-5.4-mini`, judge `gpt-5.5`, top-3 retrieved docs as context:

| Retriever | Faithfulness (40 q) | Correct (36 answerable) | Said "I don't know" on unanswerable (4) |
|---|---|---|---|
| BM25 | 100% | 67% | 4/4 |
| Embeddings | 98% | 89% | 4/4 |

Weak retrieval kept faithfulness perfect while correctness dropped: the model stayed faithful to the *wrong* context. Every BM25 retrieval miss produced an incorrect answer. Full per-question outputs and judge reasoning are in `results/`.

This is a tiny dataset, so treat the numbers as a demonstration of the method, not as a benchmark of these techniques.

## Adapting it to your own RAG

- Replace `data/docs.jsonl` with your chunks (`id`, `title`, `text`) and write a golden set from real (anonymized) user questions.
- Plug your own retriever into `scripts/eval_retrieval.py`. Any object with `search(query, k) -> list[doc_id]` works.
- Keep the judge prompt in `rag_eval/llm.py` strict and spot-check its verdicts against your own labels before trusting it.

## License

MIT
