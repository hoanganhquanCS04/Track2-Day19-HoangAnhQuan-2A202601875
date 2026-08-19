"""Semantic cache over Qdrant (NB7).

Deck reference: "Semantic Cache: Hien Thuc Trong 12 Dong" + the security frame.

Three knobs, three distinct failure modes -- this module makes all three
measurable instead of theoretical:

  threshold  too low  -> FALSE HIT: you serve the answer to a different question
  ttl        missing  -> STALE HIT: a correct-in-March answer served in August
  namespace  missing  -> CROSS-TENANT LEAK: user A receives user B's answer.
                         That is a security incident, not a caching bug.

`namespaced=False` exists purely so students can *observe* the leak before
fixing it. Never ship that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient, models

CACHE_COLLECTION = "lab19_semantic_cache"


def _embed_query(embedder, text: str):
    """Question-side embedding (the cache matches questions to questions).

    `app.embeddings.Embedder` exposes the asymmetric pair; tests and older code
    may hand in a bare `TextEmbedding`, which only has `.embed`. Falling back
    keeps both working -- and for symmetric models the two are identical anyway.
    """
    fn = getattr(embedder, "embed_query", None) or embedder.embed
    return next(iter(fn([text])))



@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale_evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class CacheHit:
    answer: str
    question: str
    score: float
    tenant: str
    age_s: float


@dataclass
class SemanticCache:
    client: QdrantClient
    embedder: object
    dim: int = 384
    threshold: float = 0.75      # AWS ElastiCache benchmark used 0.75
    ttl_s: float | None = 3600.0
    namespaced: bool = True      # False = deliberately vulnerable, for NB7 step 3
    stats: CacheStats = field(default_factory=CacheStats)
    _clock: float = 0.0          # virtual clock: lets TTL be tested without sleeping
    _next_id: int = 0

    def __post_init__(self) -> None:
        if CACHE_COLLECTION in {c.name for c in self.client.get_collections().collections}:
            self.client.delete_collection(CACHE_COLLECTION)
        self.client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )

    # ── time ────────────────────────────────────────────────────────────
    def advance(self, seconds: float) -> None:
        """Move the virtual clock forward (so TTL is testable in a notebook)."""
        self._clock += seconds

    def _embed(self, text: str) -> list[float]:
        return np.asarray(_embed_query(self.embedder, text), dtype=np.float32).tolist()

    # ── api ─────────────────────────────────────────────────────────────
    def get(self, tenant: str, question: str) -> CacheHit | None:
        qf = None
        if self.namespaced:
            qf = models.Filter(must=[models.FieldCondition(
                key="tenant", match=models.MatchValue(value=tenant))])

        pts = self.client.query_points(
            collection_name=CACHE_COLLECTION,
            query=self._embed(question),
            query_filter=qf,
            limit=1,
        ).points
        if not pts:
            self.stats.misses += 1
            return None

        p = pts[0]
        if p.score < self.threshold:
            self.stats.misses += 1
            return None

        age = self._clock - p.payload["ts"]
        if self.ttl_s is not None and age > self.ttl_s:
            self.stats.stale_evictions += 1
            self.stats.misses += 1
            self.client.delete(
                collection_name=CACHE_COLLECTION,
                points_selector=models.PointIdsList(points=[p.id]),
            )
            return None

        self.stats.hits += 1
        return CacheHit(
            answer=p.payload["answer"], question=p.payload["question"],
            score=float(p.score), tenant=p.payload["tenant"], age_s=age,
        )

    def peek(self, tenant: str, question: str) -> tuple[float, dict] | None:
        """Nearest cached entry + its score, WITHOUT applying threshold or TTL.

        Lets a notebook sweep every threshold from a single embedding pass
        instead of re-embedding the probe set once per candidate threshold.
        """
        qf = None
        if self.namespaced:
            qf = models.Filter(must=[models.FieldCondition(
                key="tenant", match=models.MatchValue(value=tenant))])
        pts = self.client.query_points(
            collection_name=CACHE_COLLECTION,
            query=self._embed(question), query_filter=qf, limit=1,
        ).points
        return (float(pts[0].score), pts[0].payload) if pts else None

    def put(self, tenant: str, question: str, answer: str) -> None:
        self.client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[models.PointStruct(
                id=self._next_id,
                vector=self._embed(question),
                payload={"tenant": tenant, "question": question,
                         "answer": answer, "ts": self._clock},
            )],
        )
        self._next_id += 1

    def reset_stats(self) -> None:
        self.stats = CacheStats()
