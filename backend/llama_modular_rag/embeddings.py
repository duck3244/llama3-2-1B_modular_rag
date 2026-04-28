from functools import lru_cache
from typing import Any, Dict

from langchain_huggingface import HuggingFaceEmbeddings

from llama_modular_rag.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """프로세스 수명 동안 한 번만 로드되는 임베딩 모델."""
    model_kwargs: Dict[str, Any] = {"device": "cpu"}
    encode_kwargs: Dict[str, Any] = {
        "normalize_embeddings": True,
        "batch_size": 8,
        "device": "cpu",
    }
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
