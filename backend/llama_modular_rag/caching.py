import hashlib
import json
import os
from typing import Any, Dict, Optional

from langchain_core.documents import Document

from llama_modular_rag.config import CACHE_DIR

# 캐시 포맷이 바뀌면 이 값을 올려 기존 캐시를 자동으로 무효화한다.
_CACHE_SCHEMA_VERSION = 2


def _serialize(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result)
    docs = payload.get("documents")
    if docs:
        payload["documents"] = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
    return {"_v": _CACHE_SCHEMA_VERSION, "data": payload}


def _deserialize(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if blob.get("_v") != _CACHE_SCHEMA_VERSION:
        return None
    data = dict(blob.get("data", {}))
    docs = data.get("documents")
    if docs:
        data["documents"] = [
            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            for d in docs
        ]
    return data


class QueryCache:
    """문서별로 분리된 쿼리 결과 캐시 (JSON, SHA-256)."""

    def __init__(self, cache_dir: str = CACHE_DIR) -> None:
        self.cache_dir: str = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, doc_id: str, query: str) -> str:
        material = f"{doc_id}::{query}".encode("utf-8")
        key = hashlib.sha256(material).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def get_cached_result(self, doc_id: str, query: str) -> Optional[Dict[str, Any]]:
        path = self._path(doc_id, query)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return _deserialize(blob)

    def cache_result(self, doc_id: str, query: str, result: Dict[str, Any]) -> None:
        path = self._path(doc_id, query)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_serialize(result), f, ensure_ascii=False)
