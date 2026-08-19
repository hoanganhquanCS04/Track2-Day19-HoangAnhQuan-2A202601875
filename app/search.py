"""Searcher — keyword (BM25) + semantic (vector) + hybrid (RRF) on the lab corpus.

Designed to work in both lite (Qdrant in-memory) and docker (Qdrant server) modes;
switch via env var QDRANT_MODE=memory|server (defaults to memory).

The hybrid mode uses Reciprocal Rank Fusion with k=60 — the same default used
by Vespa, Elasticsearch, and the hybrid RAG production stacks in the deck §3.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

Mode = Literal["keyword", "semantic", "hybrid"]
# Model + dimension now come from EMBEDDING_BACKEND (see app/embeddings.py).
# Defaults are unchanged: fastembed / BAAI/bge-small-en-v1.5 / 384-dim.
EMBED_MODEL = Embedder().model_name
EMBED_DIM = Embedder().dim
COLLECTION = "lab19_corpus"


@dataclass
class SearchHit:
    doc_id: str
    title: str
    text: str
    score: float

    def dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": self.score}


class Searcher:
    """Holds the BM25 index, Qdrant client, and document metadata.

    Construction is deliberately heavy (loading the embedding model + indexing
    the whole corpus once); callers should reuse a single instance.
    """

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.client: QdrantClient | None = None
        self.embedder: Embedder | None = None

    @property
    def size(self) -> int:
        return len(self.docs)

    @classmethod
    def from_corpus(cls, corpus_path: Path) -> "Searcher":
        # A student who opens NB1 before running setup otherwise gets a bare
        # FileNotFoundError pointing at a relative path, with no hint that the
        # corpus is generated rather than committed.
        if not Path(corpus_path).exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}.\n"
                "The corpus is generated, not committed. Run:\n"
                "    bash setup-lite.sh      # first time (venv + deps + data)\n"
                "    make seed               # if you only need to regenerate data"
            )
        s = cls()
        s._load_docs(corpus_path)
        s._build_bm25()
        s._build_vector_index()
        return s

    # ── ingestion ───────────────────────────────────────────────────────
    def _load_docs(self, corpus_path: Path) -> None:
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
                self.doc_ids.append(d["doc_id"])

    def _build_bm25(self) -> None:
        # Tokenise on whitespace — for VN+EN mixed text this is "good enough" baseline.
        # A real production system would use a proper VN tokenizer (underthesea / pyvi).
        # That choice is a "think hard" decision flagged in VIBE-CODING.md.
        tokenized = [self._tokenize(d["title"] + " " + d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def _build_vector_index(self) -> None:
        self.embedder = Embedder()

        mode = os.getenv("QDRANT_MODE", "memory")
        if mode == "server":
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(":memory:")

        # Recreate is OK in lite mode (it's in-memory); for server, only create if missing.
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing and mode == "server":
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            # dimension must follow the chosen model, not a module constant --
            # switching EMBEDDING_BACKEND changes it (384 -> 1024 -> 1536).
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

        # Embed in batches of 64 — fastembed is CPU-bound and that batch size is sweet spot.
        BATCH = 64
        points: list[PointStruct] = []
        for start in range(0, len(self.docs), BATCH):
            batch = self.docs[start:start + BATCH]
            texts = [d["title"] + " " + d["text"] for d in batch]
            # documents -> passage side of the asymmetric pair (see app/embeddings.py)
            vectors = list(self.embedder.embed_documents(texts))
            for i, (d, v) in enumerate(zip(batch, vectors)):
                points.append(PointStruct(
                    id=start + i,
                    vector=v.tolist(),
                    payload={"doc_id": d["doc_id"], "title": d["title"], "text": d["text"]},
                ))
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── retrieval ───────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        if mode == "keyword":
            return self._search_keyword(query, top_k)
        if mode == "semantic":
            return self._search_semantic(query, top_k)
        if mode == "hybrid":
            return self._search_hybrid(query, top_k, rrf_k)
        raise ValueError(f"unknown mode {mode!r}")

    def _search_keyword(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            SearchHit(
                doc_id=self.docs[i]["doc_id"],
                title=self.docs[i]["title"],
                text=self.docs[i]["text"],
                score=float(scores[i]),
            )
            for i in ranked
        ]

    def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.client is not None and self.embedder is not None
        # queries -> query side. Must match how the index was built.
        q_vec = next(self.embedder.embed_query([query])).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=top_k,
        )
        return [
            SearchHit(
                doc_id=p.payload["doc_id"],
                title=p.payload["title"],
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in result.points
        ]

    def _search_hybrid(self, query: str, top_k: int, rrf_k: int) -> list[SearchHit]:
        # Pull a deeper top-K from each retriever so RRF has signal beyond top-10.
        depth = max(top_k * 5, 50)
        kw_hits = self._search_keyword(query, depth)
        sem_hits = self._search_semantic(query, depth)

        # Reciprocal Rank Fusion — score(d) = sum over rankers of 1 / (k + rank_r(d))
        # rank_r is 1-based (first position is rank 1, not 0).
        rrf_scores: dict[str, float] = {}
        meta: dict[str, SearchHit] = {}
        for hits in (kw_hits, sem_hits):
            for rank, h in enumerate(hits, start=1):
                rrf_scores[h.doc_id] = rrf_scores.get(h.doc_id, 0.0) + 1.0 / (rrf_k + rank)
                meta.setdefault(h.doc_id, h)

        ordered = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            SearchHit(
                doc_id=doc_id,
                title=meta[doc_id].title,
                text=meta[doc_id].text,
                score=score,
            )
            for doc_id, score in ordered
        ]
