"""Ghép 10 session + phụ lục thành một ebook Markdown duy nhất.

Vì sao dựng bằng script thay vì copy tay?

Vì tài liệu session là nguồn sự thật — học viên đọc chúng trong lúc học. Nếu
ebook là một bản copy thủ công, hai bên sẽ lệch nhau ngay lần sửa đầu tiên và
không ai biết bản nào đúng. Chạy lại script này sau mỗi lần sửa session.

    uv run python scripts/build_ebook.py
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = ROOT / "docs" / "sessions"
PHU_LUC_CI = ROOT / "docs" / "99-phu-luc-ci.md"
PHU_LUC_AT = ROOT / "docs" / "98-phu-luc-an-toan.md"
PHU_LUC_TN = ROOT / "docs" / "97-phu-luc-tri-nho.md"
BAO_CAO_MAU = ROOT / "docs" / "eval_report-mau.md"
OUT = ROOT / "docs" / "ebook-kiem-thu-ung-dung-ai.md"


def slugify(text: str) -> str:
    """Sinh anchor kiểu GitHub cho tiêu đề tiếng Việt (giữ nguyên dấu)."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\sÀ-ỹ-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text).strip("-")


def day_heading(noi_dung: str) -> str:
    """Đẩy mọi tiêu đề xuống một cấp: H1->H2, H2->H3, H3->H4..."""
    ra = []
    trong_code = False
    for dong in noi_dung.split("\n"):
        if dong.lstrip().startswith("```"):
            trong_code = not trong_code
        if not trong_code and re.match(r"^#{1,5} ", dong):
            dong = "#" + dong
        ra.append(dong)
    return "\n".join(ra)


def don_dep(noi_dung: str) -> str:
    """Bỏ các khối điều hướng của trang sổ tay — sách đọc tuyến tính không cần chúng."""
    for moc in (
        "\n## Chọn bước tiếp theo\n",
        "\n## Tiếp theo\n",
        "\n## Quay lại\n",
    ):
        if moc in noi_dung:
            noi_dung = noi_dung.split(moc)[0]
    # Bỏ dòng breadcrumb và bảng mục lục riêng của từng trang
    dong = noi_dung.split("\n")
    if dong and dong[0].startswith("[Sổ tay]"):
        dong = dong[1:]
    noi_dung = "\n".join(dong).lstrip("\n")
    noi_dung = re.sub(
        r"\n## Nội dung trang này\n.*?\n---\n", "\n", noi_dung, flags=re.DOTALL
    )
    noi_dung = re.sub(
        r"\n## Trước khi bắt đầu\n", "\n### Trước khi bắt đầu\n", noi_dung
    )
    # Link tương đối giữa các file session -> bỏ, giữ lại chữ
    noi_dung = re.sub(r"\[([^\]]+)\]\((?:\.\./)?(?:sessions/)?[\w.-]+\.md\)", r"*\1*", noi_dung)
    return noi_dung.rstrip() + "\n"


def doc_session(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    dong = [d for d in raw.split("\n") if d.startswith("# ")]
    tieu_de = dong[0].lstrip("# ").strip() if dong else path.stem
    return tieu_de, don_dep(day_heading(raw))


FRONT_MATTER = f"""# Kiểm thử tự động cho ứng dụng AI

### pytest · DeepEval · RAGAS — với model chạy local

*Giáo trình 10 buổi, xây trên một văn bản pháp luật thật.*

Cập nhật: {date.today().strftime("%d/%m/%Y")}

---

## Cuốn sách này dành cho ai

Lập trình viên. **Không cần biết Python từ trước** — Chương 5 bắt đầu pytest từ
con số 0. Không cần biết gì về LLM.

Điều duy nhất giả định: bạn đã từng viết code và dùng terminal.

## Bạn sẽ làm được gì sau khi đọc xong

Viết được bộ test tự động **chấm điểm chất lượng đầu ra của một ứng dụng AI**,
chạy được trên CI, và giải thích được cho đồng nghiệp vì sao điểm số đó đáng tin.

Cụ thể, bạn sẽ có một repo chạy được gồm:

- Một **ứng dụng agentic đầy đủ**: RAG trên văn bản luật, agent có 4 công cụ,
  backend FastAPI và giao diện web — chạy hoàn toàn bằng **model local**.
- Một kho tài liệu có phiên bản, để agent cảnh báo khi văn bản đã có bản mới.
- Bộ test ba tầng: đường ống, chất lượng (DeepEval + RAGAS), và quỹ đạo agent.
- Một cơ chế ghi/phát lại câu trả lời LLM — thứ bạn mang về công ty dùng được
  ngay với bất kỳ model nào.

## Vấn đề mà cả cuốn sách xoay quanh

Test truyền thống dựa trên một giả định: **cùng đầu vào thì cùng đầu ra.**

```python
assert cong(2, 2) == 4        # đúng hôm nay, đúng mãi mãi
```

LLM phá vỡ giả định đó. Cùng một câu hỏi, hai lần chạy cho hai câu chữ khác
nhau, mà cả hai đều đúng.

Vậy assert cái gì? Câu trả lời của cả ngành hiện nay là dùng **một model khác để
chấm điểm** — gọi là *judge*. Bạn không so chuỗi ký tự nữa, bạn so **điểm số**
với một **ngưỡng**.

```python
score = do_do_lien_quan(cau_hoi, cau_tra_loi)   # judge chấm
assert score >= 0.7                              # ngưỡng do đội tự đặt
```

Toàn bộ cuốn sách là về việc làm cho câu lệnh trên trở nên đáng tin — và về
việc **chất vấn chính con judge** đã sinh ra điểm số đó.

## Tài liệu nguồn

**Luật số 116/2025/QH15 — Luật An ninh mạng.** Quốc hội khóa XV, Kỳ họp thứ 10
thông qua ngày 10/12/2025, hiệu lực từ 01/7/2026. Tải từ cổng thông tin Chính phủ:
<https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/01/luat116-2025.pdf>

> ⚠️ **CẢNH BÁO QUAN TRỌNG**
>
> Bản PDF gốc là **bản scan**. Toàn bộ nội dung dùng trong sách đến từ **OCR**
> nên có sai sót về dấu ("kiêm tra" thay vì "kiểm tra", "yêu câu" thay vì "yêu cầu").
>
> Đây là dữ liệu dùng để **đào tạo kỹ thuật**.
> **Không dùng để tra cứu hoặc trích dẫn pháp lý.** Hãy dùng bản gốc tại cổng
> thông tin Chính phủ.

Vì sao chọn một văn bản luật mới? Vì `llama3.1:8b` có mốc kiến thức khoảng cuối
2023 — nó **không thể biết** luật này. Tệ hơn (và hay hơn cho việc dạy): nó *có*
biết Luật An ninh mạng 2018 đã bị thay thế, nên khi bộ truy hồi hỏng, nó trả lời
bằng luật cũ, rất trôi chảy và hoàn toàn sai. Bạn nhìn thấy hệ thống nói dối ngay
từ Chương 1.

## Cách đọc cuốn sách này

Sách viết theo **12 buổi × 90 phút**. Sáu chương đầu xây ứng dụng, sáu chương sau
test nó. Đến Chương 7 bạn test `chunker.py` do chính bạn viết ở Chương 2 — đó là
lý do app đi trước.

Mỗi chương có mục **Bài tập** và **Kiểm tra cuối session**. Đừng bỏ qua phần kiểm
tra: nó là điều kiện để chương sau chạy được.

**Bạn tự gõ**: `chunker.py`, `retriever.py`, `tools.py`, và toàn bộ test của
Chương 7–12.
**Bạn nhận sẵn**: khung ứng dụng (loader, embeddings, vector store, llm),
`agent_graph.py`, `vault.py`, `api.py` và giao diện web.

Mục tiêu của sách là **kiểm thử** ứng dụng AI, không phải xây ứng dụng AI. Phần
bạn gõ chính là phần bạn cần thành thạo.

## Quy ước xuyên suốt

> **Mock LLM của ứng dụng** — hợp lệ. Đó là phát lại câu trả lời đã ghi.
>
> **Mock LLM của judge** — vô nghĩa. Đó là hardcode điểm số; test không bao giờ
> đỏ được.

Câu này xuất hiện lại ở Chương 6 và là xương sống của mọi thứ sau đó.

## Mọi con số trong sách đều được đo, không phỏng đoán

Cuốn sách này kể lại cả những lần **hỏng** trong lúc dựng repo: một hệ RAG trả
lời "Nguyên thủ tịch Quốc hội", một trợ lý pháp lý bịa khung tiền phạt, một judge
chấm sai có hệ thống, và một lần Ollama chết giữa chừng làm cả bộ test đỏ.

Đó không phải tai nạn được giữ lại cho vui. Chúng là **nội dung chính**: bạn học
cách chẩn đoán bằng cách xem người khác chẩn đoán.

"""


def main() -> int:
    files = sorted(SESSIONS_DIR.glob("session-*.md"))
    if len(files) != 12:
        print(f"Cảnh báo: tìm thấy {len(files)} bài, mong đợi 12")

    chuong: list[tuple[str, str]] = []
    for i, f in enumerate(files, 1):
        tieu_de, than = doc_session(f)
        tieu_de = re.sub(r"^Session\s+\d+\s*[—-]\s*", "", tieu_de)
        nhan = f"Chương {i} — {tieu_de}"
        than = re.sub(r"^## .*$", f"## {nhan}", than, count=1, flags=re.MULTILINE)
        chuong.append((nhan, than))

    # Phụ lục
    phu_luc: list[tuple[str, str]] = []
    ci = don_dep(day_heading(PHU_LUC_CI.read_text(encoding="utf-8")))
    ci = re.sub(r"^## .*$", "## Phụ lục A — Chạy bộ đánh giá trên CI", ci, count=1, flags=re.MULTILINE)
    tn = don_dep(day_heading(PHU_LUC_TN.read_text(encoding="utf-8")))
    tn = re.sub(r"^## .*$", "## Phụ lục A — Trí nhớ tự sửa", tn, count=1, flags=re.MULTILINE)
    phu_luc.append(("Phụ lục A — Trí nhớ tự sửa", tn))

    at = don_dep(day_heading(PHU_LUC_AT.read_text(encoding="utf-8")))
    at = re.sub(r"^## .*$", "## Phụ lục B — An toàn thông tin khi tích hợp MCP",
                at, count=1, flags=re.MULTILINE)
    phu_luc.append(("Phụ lục B — An toàn thông tin khi tích hợp MCP", at))
    ci = re.sub(r"^## Phụ lục A — Chạy bộ đánh giá trên CI$",
                "## Phụ lục C — Chạy bộ đánh giá trên CI", ci, count=1, flags=re.MULTILINE)
    phu_luc.append(("Phụ lục C — Chạy bộ đánh giá trên CI", ci))
    phu_luc.append(("Phụ lục D — Bảng tra cứu nhanh", PHU_LUC_B))
    phu_luc.append(("Phụ lục E — Mọi con số đã đo", PHU_LUC_C))
    phu_luc.append(("Phụ lục F — Thuật ngữ", PHU_LUC_D))

    bao_cao = day_heading(BAO_CAO_MAU.read_text(encoding="utf-8"))
    bao_cao = re.sub(r"^## .*$", "## Phụ lục G — Báo cáo đánh giá mẫu", bao_cao, count=1, flags=re.MULTILINE)
    phu_luc.append(("Phụ lục G — Báo cáo đánh giá mẫu", bao_cao))

    # Mục lục
    muc_luc = ["## Mục lục", ""]
    for nhan, _ in chuong:
        muc_luc.append(f"- [{nhan}](#{slugify(nhan)})")
    muc_luc.append("")
    for nhan, _ in phu_luc:
        muc_luc.append(f"- [{nhan}](#{slugify(nhan)})")
    muc_luc.append("")

    phan = [FRONT_MATTER, "\n".join(muc_luc), "\n---\n"]
    for nhan, than in chuong:
        phan.append(than)
        phan.append("\n---\n")
    for nhan, than in phu_luc:
        phan.append(than)
        phan.append("\n---\n")
    phan.append(LOI_KET)

    OUT.write_text("\n".join(phan), encoding="utf-8")
    so_dong = OUT.read_text(encoding="utf-8").count("\n")
    print(f"Đã tạo: {OUT}")
    print(f"  {len(chuong)} chương + {len(phu_luc)} phụ lục, {so_dong:,} dòng")
    return 0


PHU_LUC_B = """## Phụ lục D — Bảng tra cứu nhanh

### Cài đặt

```bash
brew install uv                    # macOS/Linux
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
ollama pull llama3.1:8b && ollama pull nomic-embed-text-v2-moe
```

```bash
uv sync
```

### Chạy ứng dụng

```bash
uv run uvicorn rag_qa.api:app --reload --port 8000    # web + API
```

Mở <http://localhost:8000>.

```bash
uv run python -m rag_qa.cli chat      # chatbot dòng lệnh, không tra cứu
```

```bash
uv run python -m rag_qa.cli rag       # RAG dòng lệnh, có tra cứu
```

### Điểm cuối API

| Điểm cuối | Trả về |
|---|---|
| `POST /api/hoi` | câu trả lời + quỹ đạo + ngữ cảnh |
| `GET /api/tai-lieu` | danh sách văn bản trong vault |
| `GET /api/cap-nhat` | có bản mới hơn không |
| `GET /api/suc-khoe` | model, số chunk, top_k |

### Dựng lại dữ liệu

```bash
uv run python scripts/build_corpus.py            # .txt (đã OCR) -> .docx
```

```bash
uv run python scripts/ingest.py --reset          # cắt theo Điều + nạp Chroma
```

```bash
uv run python scripts/ingest.py --reset --cach-cat ky-tu --chunk-size 500
```

### Bản ghi câu trả lời

```bash
uv run python scripts/record_responses.py --prune     # ghi lại + xoá bản cũ
```

### Chạy test — LUÔN CHẠY TÁCH TỪNG BỘ

```bash
uv run pytest -m unit                                        # 74 test, ~2 giây
```

```bash
DEEPEVAL_TELEMETRY_OPT_OUT=YES uv run pytest tests/test_chatbot_deepeval.py
```

```bash
DEEPEVAL_TELEMETRY_OPT_OUT=YES uv run pytest tests/test_rag_ragas.py
```

```bash
uv run pytest tests/test_agent_plumbing.py tests/test_agent_trajectory.py
```

```bash
uv run pytest tests/test_vault.py tests/test_api.py     # 31 test, ~1 giây
```

> Đừng chạy `uv run pytest` trần. Xem Chương 7 mục "Vận hành".

### Hiệu chỉnh ngưỡng

```bash
uv run python scripts/eval_report.py       # chỉ báo cáo, không assert
```

### Ba chế độ của bản ghi

| Lệnh | Hành vi |
|---|---|
| `pytest -m eval` | Phát lại bản ghi — nhanh, offline |
| `pytest -m eval --record` | Gọi model thật rồi ghi lại |
| `pytest -m eval --live` | Gọi model thật, không ghi (kiểm chứng bản ghi cũ) |

### Biến môi trường

| Biến | Mặc định | Tác dụng |
|---|---|---|
| `RAG_QA_APP_MODEL` | `llama3.1:8b` | Model sinh câu trả lời |
| `RAG_QA_JUDGE_MODEL` | `llama3.1:8b` | Model chấm điểm |
| `RAG_QA_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | Model embedding |
| `RAG_QA_TOP_K` | `3` | Số chunk lấy về |
| `RAG_QA_CHUNK_SIZE` | `1200` | Kích thước chunk tối đa |
| `RAG_QA_TIMEOUT` | `120` | Trần thời gian mỗi lời gọi model (giây) |
| `RAG_QA_MAX_TOKENS` | `512` | Trần token sinh ra mỗi lượt |
| `RAG_QA_AGENT_LIVE` | *(tắt)* | `1` để bật test agent chạy thật |

### Bốn công cụ của agent

| Công cụ | Việc |
|---|---|
| `tra_cuu_van_ban` | Tra nội dung Luật 116/2025 trong ChromaDB |
| `kiem_tra_cap_nhat` | Hỏi vault xem văn bản có bản mới hơn không |
| `tinh_thoi_han` | Cộng số tháng vào một mốc ngày |
| `chuyen_chuyen_gia` | Leo thang cho người thật |

### Khi mọi thứ đỏ cùng lúc

```bash
curl -m 10 http://localhost:11434/api/tags
```

`HTTP 000` = Ollama chết. Nghi ngờ hạ tầng trước khi nghi ngờ code.
"""


PHU_LUC_C = """## Phụ lục E — Mọi con số đã đo

Không con số nào trong sách là phỏng đoán. Đây là bảng tổng hợp, kèm chương giải
thích ý nghĩa.

### Chọn judge

| Judge | Thời gian/metric | Kết quả | Chương |
|---|---|---|---|
| `llama3.1:8b` | 13 giây | Parse JSON sạch | 1 |
| `qwen3.8` (17 GB) | 180 giây | `RetryError` — **hỏng** | 1 |
| `deepseek-r1:8b` | tương tự | cùng vấn đề | 1 |

Nguyên nhân: `qwen3` và `deepseek-r1` là model **suy luận**, sinh khối `<think>`
dài trước khi trả lời, làm hỏng bước parse JSON của DeepEval/RAGAS.

### Cách cắt chunk quyết định chất lượng RAG

| Cách cắt | Số chunk | Câu trả lời cho "luật có hiệu lực từ ngày nào?" | Chương |
|---|---|---|---|
| 500 ký tự cố định | 252 | *"Nguyên thủ tịch Quốc hội"* ❌ | 2 |
| Theo Điều, lặp nhãn | 103 | *"...từ ngày 01 tháng 7 năm 2026"* ✅ | 2 |

Cùng model, cùng câu hỏi, cùng retriever. Chỉ đổi cách cắt.

### Có RAG so với không RAG

| Câu hỏi | Có RAG | Không RAG | Chương |
|---|---|---|---|
| Hiệu lực từ ngày nào? | **01/7/2026** ✅ | "01/01/2024" ❌ | 1 |
| Thời hạn cung cấp thông tin? | **24 giờ** ✅ | "12 giờ" ❌ | 1 |
| Luật nào hết hiệu lực? | 86/2015 và 24/2018 ✅ | "không có thông tin" | 1 |

### Judge sai có hệ thống, không phải nhiễu

`Faithfulness`, đo 3 lần trên cùng dữ liệu:

| Case | Ba lần đo | Dao động |
|---|---|---|
| `hieu-luc-thi-hanh` | 0.500, 0.500, 0.500 | 0.000 |
| `luat-het-hieu-luc` | 1.000, 1.000, 1.000 | 0.000 |
| `so-cap-do-he-thong` | 1.000, 1.000, 1.000 | 0.000 |
| `thoi-han-chuyen-tiep` | 1.000, 1.000, 1.000 | 0.000 |

`hieu-luc-thi-hanh` bị chấm 0.500 **cả ba lần** dù câu trả lời trích nguyên văn
Điều 44. Sai ổn định nguy hiểm hơn sai ngẫu nhiên. (Chương 8)

Trên một corpus khác, `ResponseRelevancy` từng dao động tới **0.092**. Độ ổn định
phụ thuộc metric và dữ liệu — phải đo, không suy ra được.

### Hai case đỏ, hai nguyên nhân trái ngược

| Case | Nguyên nhân | Bằng chứng | Chương |
|---|---|---|---|
| `cap-do-5` | **Model ứng dụng** không trích được câu trả lời | 3 bước loại trừ retriever + prompt | 9 |
| `so-cap-do-he-thong` | **Model judge** không chấm nổi | `OutputParserException` 5/5 lần | 9 |

Case thứ hai: ứng dụng trả lời **hoàn toàn đúng**.

### Ngưỡng cuối cùng

| Metric | Nhóm | Thấp nhất đo được | Ngưỡng |
|---|---|---|---|
| Faithfulness | `su-kien-don` | 0.500 | **0.40** |
| Faithfulness | `tong-hop` | 0.400 | **0.30** |
| Response Relevancy | `su-kien-don` | 0.630 | **0.50** |
| Context Precision | `su-kien-don` | 1.000 | **0.80** |
| Context Recall | `su-kien-don` | 0.500 | **0.40** |

Quy tắc: ngưỡng = (thấp nhất) − (biên dao động ≈ 0.10). Đừng chép sang dự án
khác — chạy `eval_report.py` trên dữ liệu của bạn. (Chương 8)

### Tốc độ bộ test

| Bộ | Số test | Thời gian |
|---|---|---|
| `-m unit` | 74 | ~2 giây |
| Agent (đường ống + quỹ đạo) | 27 + 4 skip | ~0,6 giây |
| DeepEval chatbot | 16 | ~2,5 phút |
| RAGAS | 27 + 2 xfail | ~10 phút |

Chạy tất cả một lượt: ~50 phút, và **Ollama đã chết giữa chừng** trong lần thử.
(Chương 7)

### Corpus

| Chỉ số | Giá trị |
|---|---|
| PDF gốc | 37 trang, 16 MB, **bản scan** |
| Text trích xuất được trước OCR | **0 ký tự** |
| Sau OCR | 86.083 ký tự, 8 Chương, 45 Điều |
| Chunk (cắt theo Điều, 1200/100) | 103 |
| Thời gian embedding | ~4 giây |
"""


PHU_LUC_D = """## Phụ lục F — Thuật ngữ

**Chunk** — một đoạn văn bản nhỏ cắt ra từ tài liệu gốc; đơn vị nhỏ nhất mà hệ
thống lưu và truy hồi. Trong sách này, mỗi chunk tương ứng một Điều luật (hoặc
một mảnh của Điều dài, vẫn mang nhãn Điều).

**Embedding** — biểu diễn số học của một đoạn văn bản dưới dạng vector nhiều
chiều, cho phép so sánh ngữ nghĩa bằng phép toán thay vì so từ khoá.

**Ngữ cảnh (context)** — tập hợp các chunk mà retriever trả về và được nhét vào
prompt gửi cho model.

**Judge** — model ngôn ngữ đóng vai người chấm điểm. Khác hoàn toàn với model
ứng dụng. Không bao giờ được mock.

**Câu hỏi vàng (golden question)** — câu hỏi đã có sẵn câu trả lời chuẩn do con
người viết, dùng làm chuẩn so sánh khi đánh giá.

**Bản ghi (recording)** — câu trả lời thật của model ứng dụng, lưu ra đĩa và phát
lại trong test. Khoá cache là băm của (model, system prompt, prompt).

**Quỹ đạo (trajectory)** — chuỗi công cụ mà agent đã gọi, theo đúng thứ tự. Đối
tượng chấm điểm riêng của agent, ngoài câu trả lời cuối.

**Scratchpad** — bộ nhớ ngắn hạn của agent, lưu kết quả từng lần gọi công cụ để
bước sau dùng lại.

**Reducer** (LangGraph) — hàm quy định cách state được cập nhật khi node trả về
giá trị mới. Mặc định là ghi đè; với reducer có thể cộng dồn.

**Faithfulness** — tỷ lệ khẳng định trong câu trả lời truy ngược được về ngữ cảnh.
Điểm thấp = model đang bịa.

**Response Relevancy** — mức độ câu trả lời thực sự trả lời đúng câu được hỏi.

**Context Precision** — trong những chunk lấy về, bao nhiêu phần thực sự hữu ích.
Chấm **retriever**, không chấm model sinh.

**Context Recall** — những thông tin cần thiết đã được lấy về đủ chưa. Cũng chấm
retriever.

**G-Eval** — cơ chế của DeepEval cho phép mô tả tiêu chí chấm bằng ngôn ngữ tự
nhiên và biến nó thành metric chạy được.

**NaN** — kết quả khi judge không sinh nổi JSON hợp lệ. **Không phải điểm 0** mà
là "không đo được". Hai thứ này phải xử lý khác nhau.

**xfail** — đánh dấu một test đã biết là đỏ vì lý do đã khoanh vùng được, nằm
ngoài tầm sửa hiện tại. Không phải cách để giấu lỗi.
"""


LOI_KET = """## Lời kết — Bảy điều mang về công ty

1. **Mock cái tạo ra câu trả lời, đừng bao giờ mock cái chấm điểm.**
   Mock judge = hardcode điểm số; bộ test sẽ xanh kể cả khi bạn xoá sạch ứng dụng.

2. **Judge tốt là model bám định dạng, không phải model thông minh nhất.**
   Model suy luận sinh khối `<think>` phá vỡ JSON parser của mọi framework đánh
   giá hiện nay.

3. **Metric có thể nói dối — và nói dối một cách ổn định.**
   Trước khi tin một điểm số thấp, hãy đọc dữ liệu thô đã sinh ra nó.

4. **Phân biệt cho được ba loại lỗi: retriever, prompt, model.**
   Ba bước khoanh vùng — truy hồi, ngữ cảnh tối thiểu, prompt tối giản — cho bạn
   biết chính xác phải sửa ở đâu, hoặc biết rằng không sửa được.

5. **Ngưỡng rút ra từ dữ liệu, chừa biên cho dao động, và ghi lại lý do.**
   Với judge yếu, một cổng chặn lỏng mà ổn định tốt hơn một cổng chặt mà chập chờn
   — vì cổng chập chờn sẽ bị cả đội bỏ qua trong một tuần.

6. **Cách cắt tài liệu quyết định chất lượng RAG nhiều hơn prompt.**
   Rất nhiều đội đi tối ưu prompt và đổi model trong khi vấn đề nằm ở chunker.

7. **Mọi lời gọi ra ngoài trong bộ test phải có trần thời gian.**
   Một test đỏ thì sửa được. Một test treo thì cả đội chỉ biết bấm Ctrl-C — rồi
   thôi không chạy nữa.

---

### Và một điều cuối

Cuốn sách này kể lại nhiều lần hỏng hơn là lần chạy đúng: một hệ RAG trả lời
"Nguyên thủ tịch Quốc hội", một trợ lý pháp lý bịa khung tiền phạt, một judge
chấm sai có hệ thống, một agent dẫn "Điều 73" trong bộ luật chỉ có 45 điều, và
một lần Ollama chết làm cả bộ test đỏ.

Đó là chủ ý. Kiểm thử ứng dụng AI không phải kỹ năng viết assert — đó là kỹ năng
**không tin một con số cho tới khi bạn hiểu nó từ đâu ra**.

Chúc bạn dựng được bộ test mà đội mình thật sự chạy.
"""


if __name__ == "__main__":
    raise SystemExit(main())
