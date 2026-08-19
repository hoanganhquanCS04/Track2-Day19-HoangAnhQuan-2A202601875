# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB2 — Hybrid Search: BM25 + Vector + RRF
#
# **Stack:** `rank-bm25` cho BM25 sparse + `qdrant-client` cho dense + RRF fusion.
# Maps to slide §3 (Hybrid Search Mechanics) + deliverable bullet 2.
#
# > Hybrid search (BM25 + Vector + RRF $k=60$) là mặc định production 2026 —
# > mọi vector DB lớn (Qdrant, Weaviate, OpenSearch, Milvus) đều có sẵn. Mức
# > cải thiện điển hình so với dense-only là **~10–15 điểm Recall@10**, nhưng
# > con số thật phụ thuộc corpus của bạn — nên notebook này **đo trên golden set
# > của chính lab** thay vì trích một con số từ blog.

# %%
import _setup  # noqa: F401
import json
import statistics
from pathlib import Path

from app.embeddings import Embedder, describe
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Reload corpus + build both indices

# %%
docs = [json.loads(line) for line in (DATA / "corpus_vn.jsonl").open(encoding="utf-8")]

# BM25
tokenized = [(d["title"] + " " + d["text"]).lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# Vector
# Same as NB1: model, device and dimension all come from .env. The whole point
# of this notebook is that the embedding model is a VARIABLE -- re-run with
# EMBEDDING_BACKEND=multilingual and compare the paraphrase slice.
embedder = Embedder()
print(describe())
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
)
BATCH = 64
points = []
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed_documents(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i, vector=v.tolist(),
            payload={"doc_id": d["doc_id"], "topic": d["topic"]},
        ))
client.upsert(collection_name="lab19", points=points)
print(f"BM25 + vector indices ready ({len(docs)} docs)")

# %% [markdown]
# ## 2. Per-mode search functions

# %%
TOP_K = 10
RRF_K = 60   # standard default — see slide §3


def search_keyword(query: str, top_k: int = TOP_K) -> list[str]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [docs[i]["doc_id"] for i in ranked]


def search_semantic(query: str, top_k: int = TOP_K) -> list[str]:
    q_vec = next(embedder.embed_query([query])).tolist()
    res = client.query_points(collection_name="lab19", query=q_vec, limit=top_k)
    return [p.payload["doc_id"] for p in res.points]


# %% [markdown]
# ## 3. TODO — implement Reciprocal Rank Fusion
#
# Công thức (deck §3):
#
# $$\text{score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$
#
# `rank_r(d)` là 1-based (vị trí đầu = 1, không phải 0). $k = 60$ là default công nghiệp.
#
# **Bước:**
# 1. Pull top-50 từ BM25 và top-50 từ vector (depth = 5×top_k để có signal sâu).
# 2. Cho mỗi doc, cộng `1 / (k + rank)` từ mỗi retriever (nếu doc không xuất hiện thì bỏ qua).
# 3. Sort theo total score, trả về top-10 doc_id.

# %%
def search_hybrid(query: str, top_k: int = TOP_K, rrf_k: int = RRF_K) -> list[str]:
    depth = max(top_k * 5, 50)
    kw_ids = search_keyword(query, depth)
    sem_ids = search_semantic(query, depth)

    # TODO: implement RRF fusion below.
    # Hint: dict[doc_id, float] cộng 1/(rrf_k + rank) từ mỗi retriever.
    # rank starts at 1, not 0.
    rrf: dict[str, float] = {}
    for rank, doc_id in enumerate(kw_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc_id in enumerate(sem_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    return [doc_id for doc_id, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]


# Quick sanity (1 paraphrase query from data/golden_set.jsonl):
test_q = "co giãn linh hoạt theo nhu cầu sử dụng"
print(f"Query: {test_q}")
print(f"  keyword top-3:  {search_keyword(test_q)[:3]}")
print(f"  semantic top-3: {search_semantic(test_q)[:3]}")
print(f"  hybrid top-3:   {search_hybrid(test_q)[:3]}")

# %% [markdown]
# ## 4. Đánh giá trên golden set (50 queries)
#
# Metric: **Precision@10** = fraction of top-10 thuộc đúng topic.
# (Slide deck dùng "Recall@10" với 1-relevant-per-query setup khác — ở đây dùng
# precision-style để có signal rõ với 100 docs/topic.)

# %%
golden = [json.loads(line) for line in (DATA / "golden_set.jsonl").open(encoding="utf-8")]
doc_topic = {d["doc_id"]: d["topic"] for d in docs}


def precision_at_10(retrieved_ids: list[str], target_topic: str) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for d in retrieved_ids if doc_topic.get(d) == target_topic) / len(retrieved_ids)


p_kw, p_sem, p_hyb = [], [], []
for q in golden:
    p_kw.append(precision_at_10(search_keyword(q["query"]), q["topic"]))
    p_sem.append(precision_at_10(search_semantic(q["query"]), q["topic"]))
    p_hyb.append(precision_at_10(search_hybrid(q["query"]), q["topic"]))

print(f"Precision@10 (avg over {len(golden)} queries):")
print(f"  Keyword (BM25)   : {statistics.mean(p_kw):.1%}")
print(f"  Semantic (vector): {statistics.mean(p_sem):.1%}")
print(f"  Hybrid  (RRF=60) : {statistics.mean(p_hyb):.1%}   <- should win")

# %% [markdown]
# ## 5. Slice theo loại query
#
# Golden set có 3 loại: `exact` (BM25 ưu thế), `paraphrase` (vector ưu thế),
# `mixed` (hybrid ưu thế). In separate scores để thấy *tại sao* hybrid thắng.

# %%
from collections import defaultdict

by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"kw": [], "sem": [], "hyb": []})
for q, kw, sem, hyb in zip(golden, p_kw, p_sem, p_hyb):
    by_type[q["mode_hint"]]["kw"].append(kw)
    by_type[q["mode_hint"]]["sem"].append(sem)
    by_type[q["mode_hint"]]["hyb"].append(hyb)

print(f"  {'type':12} {'n':>3}  {'kw':>7} {'sem':>7} {'hyb':>7}")
for t in ("exact", "paraphrase", "mixed"):
    m = by_type[t]
    print(f"  {t:12} {len(m['kw']):>3}  "
          f"{statistics.mean(m['kw']):>6.1%} "
          f"{statistics.mean(m['sem']):>6.1%} "
          f"{statistics.mean(m['hyb']):>6.1%}")

# %% [markdown]
# ### Diễn giải kết quả
#
# - `exact` queries chứa từ kỹ thuật verbatim trong corpus → BM25 mạnh, hybrid
#   thường ngang bằng (keyword signal đã đủ mạnh).
# - `paraphrase` queries dùng từ Việt **không** xuất hiện verbatim trong docs
#   → cả BM25 và vector đều giảm điểm. Trên synthetic corpus 1000-doc với
#   embedding model `BAAI/bge-small-en-v1.5` (English-trained), semantic
#   recall trên Vietnamese paraphrases yếu (24-32%). **Đổi sang `bge-m3`
#   (full Docker path) sẽ giúp semantic thắng paraphrase queries** — đây là
#   teaching moment cho "embedding model choice matters".
# - `mixed` queries có cả từ exact + ý tưởng paraphrased → **hybrid thắng rõ**
#   (~100% vs 97-98% pure modes). Đây là pattern production-relevant nhất
#   vì user thật ít khi viết query 100% exact term hoặc 100% paraphrase.
#
# Hybrid thắng *trung bình* nhờ robust trên mọi kiểu query — đó là lý do
# production luôn default hybrid (deck §3, slide "Hybrid Search Mechanics").

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 4: bảng Precision@10 với 3 mode, hybrid > kw và > sem.
# 2. Output cell 5: bảng slice theo loại query, exact/paraphrase/mixed.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the per-mode search wrapper functions in §2. AI nailed
# the pattern in 1 shot. Cũng AI tốt cho việc set up bảng kết quả (`statistics.mean`,
# format `{:.1%}`) — chỉ cần spec rõ output schema.
#
# **Think hard yourself:** the RRF formula. Trước khi implement, hỏi AI giải
# thích RRF rồi cross-check với deck §3. Nếu AI viết code mà rank bắt đầu từ 0
# (không phải 1) hoặc cộng 1/rank thay vì 1/(k+rank), đã hỏng — và rất khó debug
# về sau khi quality giảm. Đây là 1 ví dụ "AI write 5 dòng đúng đắn nhưng nếu
# bạn không tự kiểm tra công thức, bug nằm im trong production".
