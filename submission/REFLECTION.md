# Reflection — Lab 19

**Tên:** Hoàng Anh Quân (2A202601875)
**Cohort:** A20
**Path đã chạy:** lite (Qdrant in-memory + SQLite Feast), embedding chạy trên GPU qua `onnxruntime-gpu`

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**`exact` — BM25 thắng** (96.7% vs vector 88.7%). Query chứa nguyên văn thuật ngữ
(`Kubernetes`, `auto-scaling`); đó đúng là chỗ IDF phát huy, còn vector không phân
biệt nổi những token gần giống nhau.

**`paraphrase` — không mode nào thắng** (kw 33.3%, sem 24.0%, hyb 32.0%). Ngược
kỳ vọng của rubric, và nguyên nhân không nằm ở thuật toán: `bge-small-en-v1.5` là
model tiếng Anh, nên câu hỏi tiếng Việt diễn đạt lại bị map sai cụm. Giới hạn của
**model**, không phải của phương pháp — đổi sang model đa ngữ là biến duy nhất
cần thay.

**`mixed` — hybrid thắng tuyệt đối** (100% vs 97.0/98.5). Query nửa thuật ngữ nửa
khái niệm là chỗ RRF thật sự có việc làm: doc nào được **cả hai** retriever đồng
thuận sẽ nổi lên, dù không bên nào xếp nó số 1.

**Không dùng hybrid khi:** (1) domain toàn định danh chính xác (mã lỗi, số hoá
đơn) — BM25 thuần vừa nhanh vừa đúng hơn; (2) ngân sách latency gắt — đo được
keyword P99 2.4 ms còn hybrid 12.9 ms, gấp 5 lần, vì phải chạy cả hai nhánh;
(3) khi embedding model không phủ ngôn ngữ của corpus — như chính lab này, vector
chỉ thêm nhiễu vào slice `paraphrase`.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid P99 trượt ngưỡng 50 ms (82.2 ms) — nhưng **không phải vì search chậm**.
Bóc tách: embed query 50.2 ms, ANN 1.8 ms, BM25 1.5 ms → 94% chi phí là một
forward pass trên CPU. Tăng thread không cứu được (`threads=1` đã nhanh nhất;
12 thread còn tệ hơn), nên đây là giới hạn cứng chứ không phải code dở.

Thủ phạm thứ hai bất ngờ hơn: harness của NB3 dùng `httpx.get()` — mở TCP mới
cho **từng** request. Chi phí bắt tay chạy trên event-loop thread, giành GIL với
thread đang search, nên đồng hồ **server-side** (chỉ bọc quanh `searcher.search()`)
bị tính nhầm cả thời gian bị treo: P50 40.3 ms với kết nối mới so với 9.9 ms khi
tái dùng — **cùng server, cùng query, khác 4 lần, hoàn toàn là artefact đo đạc**.

Bài học chung của cả hai: mình suýt sửa nhầm `main.py` vì đoán rằng endpoint
đồng bộ gây xoay vòng thread. Đo kiểm chứng trước (3 kịch bản thread đều 9–12 ms)
đã bác bỏ giả thuyết đó và giữ cho code vô tội không bị đụng vào.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(không)_
