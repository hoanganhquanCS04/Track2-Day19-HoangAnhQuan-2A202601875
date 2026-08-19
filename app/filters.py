"""Filtered vector search: the three strategies, measured side by side (NB5).

Deck reference: "Filtered Search: Cai Bay Recall It Ai Noi Den".

    post-filter   ANN first, drop non-matching after  -> recall CLIFF
    pre-filter    exact scan over the matching subset -> always correct, loses the index
    filtered-ANN  the index itself is filter-aware    -> what you actually want

`FilteredIndex` deliberately REUSES the vectors already computed by
`app.search.Searcher` (pulled back out of Qdrant with `with_vectors=True`)
instead of re-embedding 1000 documents. Embedding is the slow part of this lab;
paying for it twice teaches nothing.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from app.metadata import enrich
from app.search import COLLECTION as BASE_COLLECTION
from app.search import Searcher

FILTERED_COLLECTION = "lab19_filtered"


def _embed_query(embedder, text: str):
    """Query-side embedding, tolerant of a raw fastembed object.

    `app.embeddings.Embedder` exposes the asymmetric pair; tests and older code
    may hand in a bare `TextEmbedding`, which only has `.embed`. Falling back
    keeps both working -- and for symmetric models the two are identical anyway.
    """
    fn = getattr(embedder, "embed_query", None) or embedder.embed
    return next(iter(fn([text])))



@dataclass
class Result:
    """One strategy's answer for one query."""
    strategy: str
    doc_ids: list[str]
    latency_ms: float
    # How many candidates the engine actually looked at before filtering.
    # This is what makes the post-filter cliff visible.
    fetched: int = 0

    def recall_against(self, truth: list[str]) -> float:
        if not truth:
            return 1.0
        return len(set(self.doc_ids) & set(truth)) / len(truth)


@dataclass
class FilteredIndex:
    client: QdrantClient
    docs: list[dict]
    vectors: np.ndarray                     # (N, dim) float32, row i <-> docs[i]
    embedder: object
    _id_of: dict[str, int] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_searcher(cls, searcher: Searcher) -> "FilteredIndex":
        """Clone the base collection into a filter-aware one with rich payloads."""
        client = searcher.client
        assert client is not None, "searcher was not built"

        points, _ = client.scroll(
            collection_name=BASE_COLLECTION,
            limit=100_000,
            with_vectors=True,
            with_payload=True,
        )
        points = sorted(points, key=lambda p: p.id)

        # The base collection's payload only carries doc_id/title/text, so pull
        # the FULL corpus record (which has `topic`) from the searcher and key
        # it by doc_id. Without this, a topic filter matches zero documents and
        # every topic-filtered query silently returns nothing.
        by_id = {d["doc_id"]: d for d in searcher.docs}
        docs = [enrich({**by_id.get(p.payload["doc_id"], {}), **p.payload}) for p in points]
        vectors = np.asarray([p.vector for p in points], dtype=np.float32)

        if FILTERED_COLLECTION in {c.name for c in client.get_collections().collections}:
            client.delete_collection(FILTERED_COLLECTION)
        client.create_collection(
            collection_name=FILTERED_COLLECTION,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        client.upsert(
            collection_name=FILTERED_COLLECTION,
            points=[
                models.PointStruct(id=i, vector=vectors[i].tolist(), payload=docs[i])
                for i in range(len(docs))
            ],
        )
        # Payload indexes are what let a real Qdrant deployment filter *inside*
        # the HNSW walk instead of after it.
        for fname, ftype in (
            ("tenant", models.PayloadSchemaType.KEYWORD),
            ("access", models.PayloadSchemaType.KEYWORD),
            ("topic", models.PayloadSchemaType.KEYWORD),
            ("published_ts", models.PayloadSchemaType.INTEGER),
        ):
            try:
                # Local in-memory Qdrant filters correctly but ignores payload
                # indexes, and warns loudly about it every single run. On a real
                # server this call is what keeps filtered-ANN fast at scale.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    client.create_payload_index(FILTERED_COLLECTION, fname,
                                                field_schema=ftype)
            except Exception:
                pass

        idx = cls(client=client, docs=docs, vectors=vectors, embedder=searcher.embedder)
        idx._id_of = {d["doc_id"]: i for i, d in enumerate(docs)}
        return idx

    def embed(self, query: str) -> np.ndarray:
        return np.asarray(_embed_query(self.embedder, query), dtype=np.float32)

    # ── ground truth ────────────────────────────────────────────────────
    def exact_top_k(self, qv: np.ndarray, predicate: Callable[[dict], bool], k: int) -> list[str]:
        """Brute-force cosine over the matching subset -- the correct answer."""
        keep = [i for i, d in enumerate(self.docs) if predicate(d)]
        if not keep:
            return []
        sub = self.vectors[keep]
        sims = (sub @ qv) / (np.linalg.norm(sub, axis=1) * np.linalg.norm(qv) + 1e-12)
        order = np.argsort(-sims)[:k]
        return [self.docs[keep[i]]["doc_id"] for i in order]

    # ── the three strategies ────────────────────────────────────────────
    def post_filter(self, query: str, predicate, k: int = 10, fetch_k: int | None = None) -> Result:
        """Ask the index for fetch_k, then throw away whatever does not match."""
        fetch_k = fetch_k or k
        qv = self.embed(query)
        t0 = time.perf_counter()
        hits = self.client.query_points(
            collection_name=FILTERED_COLLECTION, query=qv.tolist(), limit=fetch_k
        ).points
        kept = [h.payload["doc_id"] for h in hits if predicate(h.payload)][:k]
        return Result("post-filter", kept, (time.perf_counter() - t0) * 1000, fetched=len(hits))

    def pre_filter(self, query: str, predicate, k: int = 10) -> Result:
        """Filter first, then scan the survivors exactly. Correct, but no index."""
        qv = self.embed(query)
        t0 = time.perf_counter()
        ids = self.exact_top_k(qv, predicate, k)
        n = sum(1 for d in self.docs if predicate(d))
        return Result("pre-filter", ids, (time.perf_counter() - t0) * 1000, fetched=n)

    def filtered_ann(self, query: str, qfilter: models.Filter, k: int = 10) -> Result:
        """Hand the filter to the engine and let it stay inside the index."""
        qv = self.embed(query)
        t0 = time.perf_counter()
        hits = self.client.query_points(
            collection_name=FILTERED_COLLECTION,
            query=qv.tolist(),
            query_filter=qfilter,
            limit=k,
        ).points
        ids = [h.payload["doc_id"] for h in hits]
        return Result("filtered-ANN", ids, (time.perf_counter() - t0) * 1000, fetched=len(ids))


# ── ready-made filters of controlled selectivity (used by NB5) ──────────
def access_filter(level: str = "internal") -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("access") == level          # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="access", match=models.MatchValue(value=level))])
    return pred, qf


def tenant_filter(tenant: str) -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("tenant") == tenant          # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="tenant", match=models.MatchValue(value=tenant))])
    return pred, qf


def recent_filter(since_ts: int) -> tuple[Callable[[dict], bool], models.Filter]:
    pred = lambda d: d.get("published_ts", 0) >= since_ts   # noqa: E731
    qf = models.Filter(must=[models.FieldCondition(
        key="published_ts", range=models.Range(gte=since_ts))])
    return pred, qf


def combo_filter(tenant: str, since_ts: int) -> tuple[Callable[[dict], bool], models.Filter]:
    """Deliberately narrow: ~1/3 x recency. This is where post-filter dies."""
    pred = lambda d: d.get("tenant") == tenant and d.get("published_ts", 0) >= since_ts  # noqa: E731
    qf = models.Filter(must=[
        models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
        models.FieldCondition(key="published_ts", range=models.Range(gte=since_ts)),
    ])
    return pred, qf
