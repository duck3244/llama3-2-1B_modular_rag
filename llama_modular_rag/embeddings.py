import os
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME


def get_embedding_model() -> HuggingFaceEmbeddings:
    """임베딩 모델 최적화 설정 (CPU 전용)"""
    print("🔧 임베딩 모델을 CPU 모드로 로드 중...")
    
    # 환경 변수로 토크나이저 캐싱 활성화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬처리 비활성화로 메모리 사용 감소
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU 비활성화

    model_kwargs: Dict[str, Any] = {
        'device': 'cpu',  # CPU 명시
        'trust_remote_code': True
    }
    
    encode_kwargs: Dict[str, Any] = {
        'normalize_embeddings': True,  # 정규화로 성능 향상
        'batch_size': 8,  # 작은 배치 크기로 메모리 사용 감소
        'device': 'cpu'  # CPU 명시
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("✅ 임베딩 모델이 CPU에 로드되었습니다.")
    
    return embeddings
