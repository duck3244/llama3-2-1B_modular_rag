import hashlib
import json
import os
import pickle
from typing import Any, Optional
from config import CACHE_DIR


class QueryCache:
    """쿼리 결과 캐싱을 위한 클래스"""

    def __init__(self, cache_dir: str = CACHE_DIR) -> None:
        self.cache_dir: str = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, query: str) -> str:
        """쿼리에서 캐시 키 생성"""
        return hashlib.md5(query.encode()).hexdigest()

    def get_cached_result(self, query: str) -> Optional[Any]:
        """캐시된 결과 검색"""
        cache_key: str = self.get_cache_key(query)
        cache_path: str = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def cache_result(self, query: str, result: Any) -> None:
        """결과 캐싱"""
        cache_key: str = self.get_cache_key(query)
        cache_path: str = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
