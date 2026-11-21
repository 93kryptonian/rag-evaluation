"""Three retrievers to compare: keyword (BM25), semantic (embeddings) and hybrid (RRF)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("a an the is are am be to of in on for and or my me i it can do does how what why your you with from "
                "if will at as this that there so still get".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    """Classic keyword search. Small, dependency-free BM25 (Okapi) implementation."""

    name = "BM25 (keyword)"

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.ids = [d["id"] for d in docs]
        self.toks = [tokenize(d["title"] + " " + d["text"]) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in self.toks]

    def search(self, query: str, k: int = 5) -> list[str]:
        q = tokenize(query)
        scores = []
        for tf, doc in zip(self.tf, self.toks):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            scores.append(s)
        order = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in order]


class EmbeddingRetriever:
    """Semantic search with a small open-source embedding model (runs locally, no API key)."""

    name = "Embeddings (semantic)"

    def __init__(self, docs: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.ids = [d["id"] for d in docs]
        self.model = SentenceTransformer(model_name)
        self.doc_vecs = self.model.encode([d["title"] + ". " + d["text"] for d in docs], normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[str]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        order = np.argsort(self.doc_vecs @ q)[::-1][:k]
        return [self.ids[i] for i in order]


class HybridRetriever:
    """Combine rankings with Reciprocal Rank Fusion: score = sum(1 / (60 + rank))."""

    name = "Hybrid (RRF)"

    def __init__(self, *retrievers, c: int = 60, depth: int = 20):
        self.retrievers, self.c, self.depth = retrievers, c, depth

    def search(self, query: str, k: int = 5) -> list[str]:
        scores: Counter = Counter()
        for r in self.retrievers:
            for rank, doc_id in enumerate(r.search(query, self.depth), start=1):
                scores[doc_id] += 1 / (self.c + rank)
        return [doc_id for doc_id, _ in scores.most_common(k)]
