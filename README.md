# Academy — Kiểm thử tự động cho ứng dụng AI

**12 bài × 90 phút.** Xây một **ứng dụng agentic** trên văn bản pháp luật thật —
RAG + tool call + backend + giao diện web — rồi viết bộ test tự động chấm chất
lượng đầu ra của nó bằng **pytest + DeepEval + RAGAS**.

Toàn bộ chạy bằng **model local (Ollama)**. Không API key, không tốn tiền, chạy
được offline.

👉 **Bắt đầu tại [docs/README.md](docs/README.md)**

## Bắt đầu nhanh

```bash
uv sync && uv run uvicorn rag_qa.api:app --port 8000
```

Mở <http://localhost:8000>. Hoặc chạy bộ test:

```bash
uv run pytest -m unit
```

Corpus và Chroma DB đã có sẵn trong repo. Dựng lại từ đầu nếu cần:

```bash
uv run python scripts/build_corpus.py && uv run python scripts/ingest.py --reset
```

## Tài liệu nguồn

**Luật số 116/2025/QH15 — Luật An ninh mạng**
(Quốc hội khóa XV, Kỳ họp thứ 10 thông qua 10/12/2025, hiệu lực 01/7/2026).
Tải từ [cổng thông tin Chính phủ](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/01/luat116-2025.pdf).

> ⚠️ PDF gốc là **bản scan**; nội dung trong repo đến từ **OCR** nên có sai sót
> về dấu. Đây là dữ liệu **đào tạo kỹ thuật** —
> **không dùng để tra cứu hoặc trích dẫn pháp lý.**

## Kiến trúc

```
web/index.html  ──►  api.py (FastAPI)  ──►  agent_graph.py
                                              │
                    ┌─────────────────────────┼──────────────────────┐
                    ▼           ▼              ▼                      ▼
            tra_cuu_van_ban  kiem_tra_    tinh_thoi_han     chuyen_chuyen_gia
                    │        cap_nhat
                    ▼           ▼
              retriever.py   vault.py
                    │           │
              Chroma (103)   manifest.json

Nạp dữ liệu:  .docx ─► document_loader ─► chunker (theo Điều) ─► embeddings ─► Chroma
```

`rag_graph.py` (pipeline cố định) được giữ nguyên bên cạnh agent để đối chiếu.

## Model (tất cả chạy local)

| Vai trò | Model | Ghi chú |
|---------|-------|---------|
| Ứng dụng | `llama3.1:8b` | Sinh câu trả lời |
| Judge | `llama3.1:8b` | Chấm điểm trong DeepEval/RAGAS |
| Embedding | `nomic-embed-text-v2-moe` | Biến text thành vector |

Judge **không** dùng model suy luận (`qwen3`, `deepseek-r1`): đã đo, chúng mất
180 giây mỗi metric rồi vẫn hỏng vì khối `<think>` phá JSON parser.

## Bộ test

| Lệnh | Số test | Thời gian | Gọi model? |
|------|---------|-----------|-----------|
| `uv run pytest -m unit` | 72 | ~3 giây | không |
| `pytest tests/test_chatbot_deepeval.py` | 16 | ~1,5 phút | judge |
| `pytest tests/test_rag_ragas.py` | 29 | ~10 phút | judge |
| `pytest tests/test_agent_plumbing.py` | 14 | ~1 giây | không |
| `pytest tests/test_agent_trajectory.py` | 21 | ~1 giây | judge (rất nhẹ) |
| `pytest tests/test_vault.py tests/test_api.py` | 31 | ~1 giây | không |
| `pytest tests/test_an_toan.py` | 48 | ~0,5 giây | không |

> ⚠️ **Chạy từng bộ một, đừng chạy `uv run pytest` trần.** Đã đo: chạy toàn bộ
> trong một lần là ~50 phút gọi judge 8B liên tục, và Ollama **đã chết giữa
> chừng** trong lần thử — mọi test sau đó đỏ hàng loạt vì hạ tầng, không phải
> vì logic. Chạy tách như bảng trên, hoặc dùng judge cloud (xem
> [phụ lục CI](docs/99-phu-luc-ci.md)).

Câu trả lời của ứng dụng được **ghi sẵn** trong `tests/fixtures/responses/` nên
bộ test chạy offline. Ghi lại sau khi đổi prompt:

```bash
uv run python scripts/record_responses.py --prune
```

## An toàn thông tin

```bash
uv run --with pip-audit --with bandit python scripts/scan_bao_mat.py
```

Tám nhóm kiểm tra, thoát mã `1` nếu có phát hiện mức CAO — dùng trực tiếp làm
cổng chặn CI. Ba nhóm cuối (`cong-cu`, `van-tay`, `du-lieu`) là bề mặt tấn công
riêng của ứng dụng agentic, không bộ quét truyền thống nào có.

Dừng khẩn agent, không cần khởi động lại dịch vụ:

```bash
touch .agent-kill
```

Mô hình mối đe doạ, quy trình ứng cứu và checklist duyệt MCP server:
[docs/98-phu-luc-an-toan.md](docs/98-phu-luc-an-toan.md).

## Quy ước quan trọng nhất của khoá học

> **Mock LLM của ứng dụng** — hợp lệ. Đó là phát lại câu trả lời đã ghi.
> **Mock LLM của judge** — vô nghĩa. Đó là hardcode điểm số; test không bao giờ đỏ được.

## Cấu trúc thư mục

```
docs/sessions/   12 trang sổ tay, mỗi trang 90 phút
docs/ebook-*.md  bản ebook gộp một file
src/rag_qa/      mã nguồn ứng dụng (RAG + agent + API)
web/             giao diện một file, không build step
tests/           bộ test
scripts/         OCR, tạo corpus, nạp dữ liệu, ghi bản ghi, báo cáo đánh giá
data/            PDF gốc + .docx đã OCR + Chroma DB + vault giả lập
snapshots/       bản sao cứu hộ sau các bài mốc
```
