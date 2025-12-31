import os
import torch

# ============================================================
# CUDA 강제 비활성화 (CPU 전용 모드)
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 완전히 비활성화
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# CPU 스레드 설정 - i7 16코어 최적화
os.environ["OMP_NUM_THREADS"] = "8"       # 연산 라이브러리용 스레드
os.environ["MKL_NUM_THREADS"] = "8"       # Intel MKL 라이브러리용 스레드
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # OpenBLAS 라이브러리용 스레드
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # Apple Accelerate 프레임워크 스레드
os.environ["NUMEXPR_NUM_THREADS"] = "8"   # NumExpr 라이브러리용 스레드

# PyTorch CPU 전용 설정
torch.set_num_threads(8)                  # PyTorch 스레드 수 설정
if torch.cuda.is_available():
    print("⚠️  경고: CUDA가 감지되었지만 CPU 모드로 강제 실행됩니다.")

# 병렬 처리 설정
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 토크나이저 병렬 처리 활성화

# 현재 스크립트 파일의 디렉토리 경로
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 모델 설정
LLAMA_MODEL_PATH: str = os.path.join(BASE_DIR, "models/torchtorchkimtorch-Llama-3.2-Korean-GGACHI-1B-Instruct-v1")
EMBEDDING_MODEL_NAME: str = os.path.join(BASE_DIR, "models/ko-sroberta-multitask")

DEVICE: str = "cpu"
CONTEXT_WINDOW: int = 1024  # 컨텍스트 창 크기 감소
TEMPERATURE: float = 0.1
MAX_NEW_TOKENS: int = 128  # 생성 토큰 수 제한

# 검색 설정
RETRIEVAL_TOP_K: int = 2  # 검색 문서 수 감소
CONTEXT_MAX_TOKENS: int = 512  # 컨텍스트 토큰 제한

# 청크 설정
CHUNK_SIZE: int = 256  # 청크 사이즈 감소
CHUNK_OVERLAP: int = 30  # 오버랩 감소

# 캐싱 설정
CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
VECTOR_DB_PATH: str = os.path.join(BASE_DIR, "vector_db")
