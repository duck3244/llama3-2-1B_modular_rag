"""FastAPI 앱 전역 상태 컨테이너.

LLM/벡터스토어/그래프는 단일 프로세스 내에서 단 하나만 존재해야 한다.
모든 추론 요청은 :pyattr:`AppState.lock`을 통해 직렬화된다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from llama_modular_rag.caching import QueryCache
from llama_modular_rag.data_loader import create_vectorstore_from_pdf
from llama_modular_rag.graph_builder import build_rag_graph

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    cache: QueryCache = field(default_factory=QueryCache)
    vectorstore: Optional[Any] = None
    graph: Optional[Any] = None
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def ready(self) -> bool:
        return self.graph is not None and self.doc_id is not None

    def attach_pdf(self, pdf_path: str) -> None:
        """동기 호출: PDF로 벡터스토어를 만들고 그래프를 구축한다."""
        vectorstore, doc_id = create_vectorstore_from_pdf(pdf_path)
        self.vectorstore = vectorstore
        self.doc_id = doc_id
        self.doc_name = os.path.basename(pdf_path)
        self.graph = build_rag_graph(vectorstore)
        logger.info("문서 활성화: %s (doc_id=%s)", self.doc_name, doc_id[:12])
