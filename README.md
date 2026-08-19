# Day 19 — Vector Store + Feature Store Lab (Track 2)

Lab cho **AICB-P2T2 · Ngày 19 · Vector Store + Feature Store**.
Build hybrid search API + Feast feature store hoàn chỉnh, đo Precision@10 và P99 latency.

**Hai paths để chọn:**

| Path | Stack | Setup | RAM | Khi nào dùng |
|---|---|---|---|---|
| **Lite (default)** | `fastembed` + Qdrant in-memory + SQLite Feast + FastAPI | `bash setup-lite.sh` (~60 s) | ~700 MB | Hầu hết học viên — laptop yếu, không Docker, focus vào concept |
| **Docker (full)** | Qdrant server + Redis + Postgres + **bge-m3** (1024d, đa ngữ) | `bash setup-docker.sh` (~3-8 min) | ~6 GB | Muốn stack production thật + embedding tốt cho tiếng Việt |

> Cả hai paths dùng **cùng `qdrant-client` API và Feast definitions** — bạn có
> thể đổi giữa hai paths bất cứ lúc nào bằng cách đổi `QDRANT_MODE` trong `.env`.

> **Embedding model là một biến thật.** `EMBEDDING_BACKEND` trong `.env` chọn
> giữa `fastembed` (bge-small, 384d, tiếng Anh — mặc định lite),
> `multilingual` (e5-large, 1024d), `bge-m3` (1024d, mặc định của path Docker)
> và `openai` (1536d). Đây chính là bài học ở NB2: bge-small yếu trên câu hỏi
> tiếng Việt diễn đạt lại; đổi sang bge-m3 rồi chạy lại NB2 để **tự đo** mức
> cải thiện. Đổi model = đổi số chiều = **phải index lại**.

---

## Quick Start — Lite (recommended)

```bash
git clone https://github.com/<your-username>/Day19-Track2-VectorFeatureStore-Lab.git
cd Day19-Track2-VectorFeatureStore-Lab
bash setup-lite.sh    # ~60 s — venv + deps + seed corpus + smoke test
make api &            # FastAPI on :8000
make benchmark        # Precision@10 + latency table
make lab              # Jupyter Lab on :8888
```

Yêu cầu: **Python 3.10–3.14**. Không cần Docker, không cần GPU, không cần OpenAI key.

> **Python 3.14:** `pyarrow` được nới lên `<26` (bản `<22` không có wheel cho
> 3.14 nên pip cố build từ nguồn và hỏng). Ngoài ra feast pin `dill~=0.3.0`
> nhưng dill 0.3.9 crash trên 3.14 khi serialize UDF của on-demand feature view
> — `setup-lite.sh` **tự phát hiện** và áp `overrides-py314.txt` chỉ khi venv
> thực sự là 3.14. Các phiên bản Python khác cài đúng những gì feast yêu cầu.
> Đã kiểm chứng end-to-end trên **Python 3.13.3 và 3.14.3**.

Khi `setup-lite.sh` báo `All checks passed`, mở
**http://localhost:8888/lab/tree/01_embeddings_index.ipynb** và bắt đầu.

### Tất cả lệnh `make`

```
make setup-lite      Lite: venv + deps + seed + smoke
make verify-lite     Lite: 5-second smoke test
make seed            Both: regenerate data/ files
make api             Lite: FastAPI on :8000
make lab             Lite: Jupyter Lab on :8888
make benchmark       Both: Precision@10 + P99 latency table
make test            Both: pytest (34 tests, ~2 s)
make gen-advanced    Both: regenerate NB6 compound queries + NB8 spend parquet
make notebooks       Both: execute ALL notebooks headless (what the grader runs)
make clean-lite      Lite: wipe venv + data + Feast registry

make setup-docker    Docker: full stack + venv + seed
make verify-docker   Docker: verify Qdrant/Redis/Postgres reachable
make docker-up       Docker: just bring services up
make docker-down     Docker: stop services (data persists)
make docker-clean    Docker: stop + wipe volumes
```

---

## Tuỳ chọn: chạy embedding trên GPU (NVIDIA)

Không bắt buộc. Lite path mặc định chạy CPU và mọi notebook đều pass — **trừ
ngưỡng `P99 < 50 ms` của NB3 trên máy CPU chậm**. Lý do đo được trên máy này:

```
embed 1 query (CPU) : 50.2 ms   <- 94% ngân sách
ANN (Qdrant, 1000d) :  1.8 ms
BM25                :  1.5 ms
```

Tăng `threads` không cứu được (`threads=1` đã nhanh nhất; 12 thread còn tệ hơn) —
đây là giới hạn cứng của một forward pass trên CPU. Bật GPU đưa embed về ~6 ms:

```bash
pip uninstall -y onnxruntime          # xung đột: cả hai cùng cung cấp module `onnxruntime`
pip install onnxruntime-gpu
pip install nvidia-cublas nvidia-cuda-runtime nvidia-cudnn-cu13             nvidia-cufft nvidia-curand
```

Rồi đặt `EMBEDDING_DEVICE=auto` (mặc định) hoặc `cuda` trong `.env`.

> **Cạm bẫy:** cài `onnxruntime-gpu` là **chưa đủ**. DLL của CUDA provider link tới
> `cublasLt64_13.dll` nằm trong `site-packages/nvidia/**/bin` — thư mục Windows
> không tìm kiếm. onnxruntime in **một dòng warning** rồi **âm thầm chạy CPU**:
> không crash, không lỗi, chỉ chậm gấp 9 lần. `app/gpu.py` đăng ký các thư mục
> đó và `cuda_available()` **dựng thử một session CUDA thật** thay vì tin vào
> `get_available_providers()` (hàm này trả `True` ngay cả khi thiếu sạch DLL).
> Đặt `EMBEDDING_DEVICE=cuda` để lỗi nổ ra thay vì im lặng.

Kiểm tra đang chạy trên gì:

```bash
python -c "from app.embeddings import describe; print(describe())"
# fastembed -> BAAI/bge-small-en-v1.5 (384d) on cuda — ...
```

Đo được trên GTX 1660 Ti (6 GB, sm_75) / CUDA 13.3: hybrid P99 **82.2 ms → 12.9 ms**.

---

## Quick Start — Docker (full stack)

```bash
bash setup-docker.sh    # ~3 min on first run (image pulls ~500 MB)
make api &
make benchmark
```

Yêu cầu: RAM ≥ 8 GB free, port 6333/6379/5432 không xung đột.
Endpoints: Qdrant http://localhost:6333 · Redis :6379 · Postgres :5432

### Ba runtime, không phải chỉ Docker

```bash
make runtime-check     # in ra runtime nào có, version bao nhiêu, làm được gì
```

| Runtime | Compose? | Lệnh |
|---|---|---|
| **Docker Desktop** | có | `bash setup-docker.sh` (dùng `docker compose`) |
| **Apple `container`** | **không** | `make container-up` rồi `bash setup-docker.sh` |
| **Podman** | có (podman-compose) | `podman compose up -d` rồi `bash setup-docker.sh` |

[**Apple `container`**](https://github.com/apple/container) chạy OCI image trong
micro-VM trên Apple silicon (macOS 26+), **không có compose**, nên
`docker-compose.yml` không dùng được. `scripts/container-up.sh` là bản tương
đương viết tay: **cùng image tag, cùng port, cùng env, cùng tên volume**, nên
`.env` và cấu hình Feast không phải đổi gì.

```bash
make runtime-check          # xem bạn đang có gì
make container-up           # khởi động Qdrant + Redis + Postgres
make container-down         # dừng (giữ volume)
make container-down ARGS=--wipe   # dừng + xoá volume
```

> **Một khác biệt thật giữa Docker và Apple container:** volume có tên của
> `container` là một filesystem đã format nên đã chứa sẵn `lost+found`.
> Postgres `initdb` từ chối khởi tạo vào thư mục không rỗng và container chết
> ngay. `container-up.sh` xử lý bằng `PGDATA=/var/lib/postgresql/data/pgdata`
> (thư mục con của mount point). Docker không gặp lỗi này vì volume của nó rỗng.
>
> Đã kiểm chứng end-to-end trên macOS 26.5.1 / Apple silicon với
> `container` 1.2.2: cả 3 service chạy, `verify_docker.py` xanh, index 1000 doc
> vào Qdrant server 12.9 s.

---

## Cấu trúc & tiến trình

| Notebook | Skill | Slide deliverable | Pass when… |
|---|---|---|---|
| `01_embeddings_index` | Embed corpus với `fastembed` + index in Qdrant + similarity search | Bullet 1 — top-5 results trả về cho query VN | indexed = 1000 vectors; top-5 cho paraphrase query đúng cluster |
| `02_hybrid_search_rrf` | BM25 + vector + RRF (k=60) + đánh giá Precision@10 trên 50 golden queries | Bullet 2 — hybrid > keyword & semantic | Hybrid wins on `mixed` slice; thắng trung bình tổng thể |
| `03_search_api_benchmark` | FastAPI `/search?q=...&mode=...` + đo P50/P95/P99 latency | Bullet 1 + 4 — REST endpoint < 50ms P99 | Hybrid P99 server-side < 50 ms |
| `04_feast_feature_store` | 3 feature views + `feast apply` + `materialize` + online lookup + PIT join | Bullet 3 — Feast 3 views materialize+online | `materialize` thành công; online lookup P99 < 10ms |

### Khối nâng cao (NB5–NB8) — theo bản deck mở rộng 2026

| Notebook | Skill | Slide | Pass when… |
|---|---|---|---|
| `05_filtered_search` | post-filter vs pre-filter vs filtered-ANN, đo recall theo độ chọn lọc | §3 Filtered Search | post-filter sập ở filter ~4%; filtered-ANN giữ recall 1.00 |
| `06_agent_retrieval` | retrieval-as-a-tool, planner tách câu hỏi, reflection, ghép ngữ cảnh với Feast | §6 Agentic Retrieval | agentic > single-shot về recall **và** balance, ở **cùng ngân sách** |
| `07_semantic_cache` | ngưỡng / TTL / namespace + demo rò chéo tenant | §6 Semantic Cache + Bảo mật | bảng sweep có cả cột "trả lời sai"; demo leak rồi fix được |
| `08_feature_engineering` | 6 họ feature, target-encoding leakage, PIT vs latest join, on-demand feature view | §7 Feature Engineering | leak gap > 0.30 trên `session_id`; ODFV trả 2 giá trị khác nhau cho cùng user |

> **NB1–NB4 là bắt buộc.** NB5–NB8 là khối nâng cao bám sát phần deck mới
> (filtered search, agentic retrieval, semantic cache, feature engineering).
> Chúng dùng lại index đã dựng ở NB1 nên **không** phải embed lại corpus.

**Source format:** Notebooks live as Jupytext `.py` files (small, easy to review).
`bash setup-lite.sh` and `make lab` auto-convert to `.ipynb`. Edit `.ipynb` in
Jupyter and Jupytext keeps both in sync.

---

## Deliverable (notebook đã chạy + ảnh chụp + reflection)

Mapping 1-to-1 với slide deliverable bullets:

1. **NB1** — Indexed 1000 vectors; top-5 results cho query Vietnamese; paraphrase query trả về đúng topic cluster.
2. **NB2** — Bảng Precision@10 với 3 mode (kw/sem/hyb), hybrid trung bình > 2 mode khác; bảng slice theo loại query (`exact`/`paraphrase`/`mixed`).
3. **NB3** — FastAPI `/search` response sample + bảng P50/P95/P99 cho 3 mode; hybrid P99 server-side < 50 ms.
4. **NB4** — `feast apply` thành công + `materialize` log + online lookup result + PIT join DataFrame.

Khối nâng cao:

5. **NB5** — Bảng recall theo độ chọn lọc + over-fetch ladder.
6. **NB6** — Bảng 3 chiến lược ở cùng ngân sách + trace reflection + `build_context()`.
7. **NB7** — Bảng sweep ngưỡng (tiết kiệm **và** trả lời sai) + demo rò chéo tenant.
8. **NB8** — Bảng leakage + PIT vs latest join + on-demand feature view.

Chấm điểm: xem [`rubric.md`](rubric.md). **100 pts core (NB1–NB4) + 50 pts nâng cao
(NB5–NB8) → Track-2 Daily Lab (30%)** + 20 pts bonus optional.

---

## Vibe-coding tips

Lab này được thiết kế cho **vibe-coding era** với CLI workflow: bạn dùng
AI assistant trong terminal (Claude Code, Codex CLI, OpenCode) để
generate boilerplate, focus vào *judgment decisions*. Đọc
[`VIBE-CODING.md`](VIBE-CODING.md) **trước khi bắt đầu NB1** (5–10 phút)
— file đó là general primer cover:

- Spec-Driven Development (SDD) và TDD trong LLM era
- Khi nào delegate cho AI, khi nào tự nghĩ
- 5 prompt patterns universal (specs in / validate before generate / TDD /
  minimal repro / plan-code-review)
- CLI tool recommendations (Claude Code / Codex CLI / OpenCode)
- 3 anti-patterns phổ biến

Mỗi notebook cũng có **vibe-coding callout** ở cuối: nói rõ subtask nào
trong notebook đó *nên* delegate cho AI, subtask nào *phải* bạn tự nghĩ.

---

## Bonus Challenge — Build Your Own AI Memory (optional, ungraded)

Một creative session 4–6 giờ: thiết kế và build minimal **AI assistant với
hybrid memory** kết hợp Vector Store (episodic memory) và Feature Store
(stable user profile). Vibe coding khuyến khích — code boilerplate AI lo,
*architecture decisions* bạn tự nghĩ + viết.

Format: brainstorm-first (15 phút sketching), code-second, làm đôi 2-3
học viên cũng được. Full brief + self-checklist:
[`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md) (tiếng Việt) ·
[`BONUS-CHALLENGE-EN.md`](BONUS-CHALLENGE-EN.md) (English).

---

## Cấu trúc repo

```
.
├── README.md                       # bạn đang đọc
├── VIBE-CODING.md                  # tips vibe-coding workflow
├── BONUS-CHALLENGE.md              # creative bonus brief (tiếng Việt)
├── BONUS-CHALLENGE-EN.md           # creative bonus brief (English)
├── rubric.md                       # 100-pt grading + 20 pt bonus
├── Makefile                        # dual-path orchestration
├── setup-lite.sh                   # lite path bootstrap (no Docker)
├── setup-docker.sh                 # docker path bootstrap (full stack)
├── docker-compose.yml              # Qdrant + Redis + Postgres
├── requirements.txt                # lite deps
├── requirements-full.txt           # docker extras
├── pyproject.toml                  # for `uv` users
├── .env.example                    # env template
├── notebooks/                      # 4 Jupytext .py files (source of truth)
│   ├── _setup.py
│   ├── 01_embeddings_index.py
│   ├── 02_hybrid_search_rrf.py
│   ├── 03_search_api_benchmark.py
│   └── 04_feast_feature_store.py
├── app/
│   ├── main.py                     # FastAPI /search endpoint
│   ├── search.py                   # Searcher class (kw / sem / hybrid)
│   └── feast_repo/
│       ├── feature_store.yaml      # Feast config (lite vs docker)
│       └── feature_views.py        # 3 feature views definition
├── scripts/
│   ├── seed_corpus.py              # 1000 VN docs + 50 golden queries (deterministic)
│   ├── benchmark.py                # Precision@10 + P99 latency harness
│   ├── verify_lite.py              # lite smoke test
│   └── verify_docker.py            # docker smoke test
├── data/                           # corpus + golden_set (regenerated by `make seed`)
└── submission/
    ├── REFLECTION.md               # template (≤ 200 words)
    └── screenshots/                # bạn add ảnh chụp ở đây
```

---

## Troubleshooting

| Triệu chứng | Fix |
|---|---|
| `setup-lite.sh` báo `python3: command not found` | Install Python 3.10+ (https://www.python.org/downloads/) |
| `make api` → port 8000 in use | `lsof -ti:8000 \| xargs kill -9` hoặc đổi `--port 8001` |
| NB1 báo `expected 1000 indexed, got X` | Chưa `make seed`; chạy lại |
| NB2 hybrid không thắng | Check RRF công thức: `1/(k + rank)` **rank 1-based**, không phải 0-based |
| NB3 P99 > 50ms | Bình thường ở cold start. Chạy 10 query warmup trước rồi đo lại. |
| NB4 `feast apply` lỗi | Xoá `app/feast_repo/registry.db` và chạy lại |
| Docker path: `port 6333 already allocated` | `docker compose down` rồi `docker compose up -d` |
| Docker path: Qdrant timeout | Đợi 60s sau `docker compose up`; image lần đầu pull ~200MB |

---

## Submission

**KHÔNG cần PR — chỉ submit GitHub URL công khai vào VinUni LMS.**

1. **Fork hoặc copy repo này lên GitHub account của bạn**, set repo **public**.
   ```bash
   # Hoặc tạo new repo trên github.com:
   git init -b main
   git remote add origin https://github.com/<your-username>/Day19-Track2-VectorFeatureStore-Lab.git
   ```
2. Hoàn thành notebooks (giữ output cells trong `.ipynb`) — NB1–NB4 bắt buộc, NB5–NB8 nâng cao.
3. Add ảnh chụp vào `submission/screenshots/`:
   - NB1: Indexed 1000 vectors + top-5 paraphrase query
   - NB2: Bảng Precision@10 với hybrid > kw/sem
   - NB3: API response + bảng latency P50/P95/P99
   - NB4: `feast apply` STDOUT + online lookup result + PIT join DF
4. Điền `submission/REFLECTION.md` (≤ 200 chữ).
5. Push lên public repo:
   ```bash
   git add -A
   git commit -m "Lab 19 submission — <Họ Tên>"
   git push -u origin main
   ```
6. **Paste public GitHub URL của bạn vào ô submission của Day 19 trong VinUni LMS.** Không cần PR. Không cần fork-back.
7. (Optional) Bonus: thêm folder `bonus/` (xem `BONUS-CHALLENGE.md`) trước khi push.

> **Quan trọng:** Repo phải **public** đến khi điểm được công bố.
> Nếu private, grader không xem được → 0 điểm.

---

© VinUniversity AICB program · A20 cohort 2026 · Track 2 Day 19.
