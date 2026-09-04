"""Cấu hình tập trung. Mọi model đều chạy local qua Ollama — không cần API key."""

import os
from pathlib import Path

# --- Đường dẫn ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "luat-116-2025-an-ninh-mang.docx"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "luat_116_2025"

# --- Model (đọc từ biến môi trường để đổi nhanh khi máy yếu) ---
# APP model: sinh câu trả lời cho người dùng. Nhỏ, nhanh.
APP_MODEL = os.getenv("RAG_QA_APP_MODEL", "llama3.1:8b")

# JUDGE model: chấm điểm trong DeepEval/RAGAS.
#
# ĐÃ ĐO THỰC TẾ, ĐỪNG ĐỔI SANG MODEL SUY LUẬN:
#   llama3.1:8b     13 giây/metric, parse JSON sạch      <- đang dùng
#   qwen3.8        180 giây/metric, RetryError (hỏng)
#   deepseek-r1:8b  cùng vấn đề với qwen3
#
# qwen3 và deepseek-r1 là model suy luận: chúng sinh khối <think> dài trước khi
# trả lời. DeepEval/RAGAS yêu cầu JSON nghiêm ngặt nên parse hỏng -> NaN/RetryError.
# Model judge tốt cho việc chấm điểm là model KHÔNG suy luận, bám định dạng.
#
# Xem mục "Chọn judge" trong docs/04-ragas-rag.md.
JUDGE_MODEL = os.getenv("RAG_QA_JUDGE_MODEL", "llama3.1:8b")

# EMBEDDING model: biến text thành vector.
EMBEDDING_MODEL = os.getenv("RAG_QA_EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Tham số RAG ---
CHUNK_SIZE = int(os.getenv("RAG_QA_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_QA_CHUNK_OVERLAP", "100"))
# top_k giảm từ 5 xuống 3 khi đổi sang chunk theo Điều.
# Lý do: chunk theo Điều dài ~1200 ký tự, gấp 2,4 lần chunk 500 ký tự trước đó.
# Giữ top_k=5 nghĩa là nhét ~6.000 ký tự vào mỗi prompt — và judge của RAGAS
# phải đọc lại toàn bộ chỗ đó cho MỖI metric. Đo được: một case mất >5 phút.
# top_k=3 cho ~3.600 ký tự, tương đương lượng ngữ cảnh cũ.
TOP_K = int(os.getenv("RAG_QA_TOP_K", "3"))

# --- Chặn treo (Part 5) ---
# Một lời gọi model bị kẹt sẽ treo cả bộ test VÔ HẠN. Agent nhân rủi ro đó lên
# theo số bước. Timeout biến "treo mãi" thành "đỏ sau 120 giây", và một test đỏ
# thì sửa được, còn một test treo thì cả đội chỉ biết bấm Ctrl-C.
REQUEST_TIMEOUT = int(os.getenv("RAG_QA_TIMEOUT", "120"))

# Trần số token sinh ra mỗi bước. Không có trần này, một lần model "lan man"
# đủ để đốt vài phút.
MAX_TOKENS = int(os.getenv("RAG_QA_MAX_TOKENS", "512"))

# Nhiệt độ 0 để output ổn định nhất có thể.
# Lưu ý: vẫn KHÔNG deterministic 100% — đó chính là lý do khoá học này tồn tại.
TEMPERATURE = 0.0
