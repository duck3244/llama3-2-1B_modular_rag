from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME
import os


def get_embedding_model():
    """임베딩 모델 최적화 설정"""
    # 환경 변수로 토크나이저 캐싱 활성화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬처리 비활성화로 메모리 사용 감소

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            'device': 'cpu'
            # compute_dtype 파라미터 제거
        },
        encode_kwargs={
            'normalize_embeddings': True,  # 정규화로 성능 향상
            'batch_size': 8  # 작은 배치 크기로 메모리 사용 감소
        }
    )
    return embeddings