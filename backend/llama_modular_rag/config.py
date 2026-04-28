"""런타임 환경 변수와 경로 상수.

부수효과는 :func:`init_runtime` 호출 시점에만 실행된다. 모듈 import만으로는
torch나 환경변수에 손대지 않으므로 FastAPI lifespan 등에서 명시적으로 호출하라.
"""
import os

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLAMA_MODEL_PATH: str = os.path.join(
    BASE_DIR, "models/torchtorchkimtorch-Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
)
EMBEDDING_MODEL_NAME: str = os.path.join(BASE_DIR, "models/ko-sroberta-multitask")

TEMPERATURE: float = 0.1
TOP_P: float = 0.95
MAX_NEW_TOKENS: int = 128

RETRIEVAL_TOP_K: int = 2
CONTEXT_MAX_TOKENS: int = 512

CHUNK_SIZE: int = 256
CHUNK_OVERLAP: int = 30

CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
VECTOR_DB_PATH: str = os.path.join(BASE_DIR, "vector_db")

_RUNTIME_INITIALIZED = False


def init_runtime(num_threads: int | None = None) -> None:
    """CUDA 비활성화·CPU 스레드 수·토크나이저 병렬화를 한 번만 적용한다."""
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    # torch import 이전에 적용해야 CUDA 초기화를 막을 수 있다.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    threads = num_threads or int(os.environ.get("RAG_NUM_THREADS", os.cpu_count() or 4))
    threads = max(1, threads)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch  # noqa: WPS433  부수효과 격리를 위한 지연 import

    torch.set_num_threads(threads)
    _RUNTIME_INITIALIZED = True
