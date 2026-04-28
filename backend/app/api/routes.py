from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    DocumentRef,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.deps import AppState
from app.streaming import build_prompt, stream_answer_tokens
from llama_modular_rag.config import RETRIEVAL_TOP_K
from llama_modular_rag.llm_setup import get_llama_tokenizer
from llama_modular_rag.retrieval import context_builder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def _state(request: Request) -> AppState:
    return request.app.state.rag


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = _state(request)
    return HealthResponse(
        status="ok",
        ready=state.ready,
        doc_id=state.doc_id,
        doc_name=state.doc_name,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    state = _state(request)
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="문서가 활성화되지 않았습니다. 먼저 PDF를 업로드하세요.",
        )

    started = time.perf_counter()

    cached = state.cache.get_cached_result(state.doc_id, payload.query)
    if cached:
        return _to_response(payload.query, cached, cached=True, started=started)

    async with state.lock:
        cached = state.cache.get_cached_result(state.doc_id, payload.query)
        if cached:
            return _to_response(payload.query, cached, cached=True, started=started)

        result: Dict[str, Any] = await run_in_threadpool(
            state.graph.invoke, {"query": payload.query}
        )
        state.cache.cache_result(state.doc_id, payload.query, result)

    return _to_response(payload.query, result, cached=False, started=started)


@router.post("/query/stream")
async def query_stream(request: Request, payload: QueryRequest) -> EventSourceResponse:
    """SSE로 답변 토큰을 스트리밍한다.

    이벤트 종류:
        ``docs``  — 검색된 참조 문서 (한 번)
        ``token`` — 답변의 디코드 청크
        ``done``  — 종료 신호 (cached, elapsed_ms)
        ``error`` — 처리 중 예외
    """
    state = _state(request)
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="문서가 활성화되지 않았습니다. 먼저 PDF를 업로드하세요.",
        )

    started = time.perf_counter()
    user_query = payload.query
    doc_id = state.doc_id  # type: ignore[assignment]

    async def event_gen():
        try:
            docs = await run_in_threadpool(
                state.vectorstore.similarity_search, user_query, RETRIEVAL_TOP_K
            )
            doc_payload = [
                {"page_content": d.page_content, "metadata": dict(d.metadata or {})}
                for d in docs
            ]
            yield {"event": "docs", "data": json.dumps(doc_payload, ensure_ascii=False)}

            cached = state.cache.get_cached_result(doc_id, user_query)
            if cached:
                yield {
                    "event": "token",
                    "data": json.dumps(cached.get("answer", ""), ensure_ascii=False),
                }
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "cached": True,
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        }
                    ),
                }
                return

            async with state.lock:
                cached = state.cache.get_cached_result(doc_id, user_query)
                if cached:
                    yield {
                        "event": "token",
                        "data": json.dumps(cached.get("answer", ""), ensure_ascii=False),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {
                                "cached": True,
                                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                            }
                        ),
                    }
                    return

                ctx_state = context_builder({"query": user_query, "documents": docs})
                prompt = build_prompt(ctx_state.get("context", ""), user_query)

                # 토크나이저는 lazy-load (첫 요청 시만 로드 비용)
                _ = get_llama_tokenizer()

                full_text_parts: list[str] = []
                async for token in stream_answer_tokens(prompt):
                    if await request.is_disconnected():
                        logger.info("클라이언트 연결 종료, 송신 중단")
                        return
                    full_text_parts.append(token)
                    yield {"event": "token", "data": json.dumps(token, ensure_ascii=False)}

                full_text = "".join(full_text_parts)
                state.cache.cache_result(
                    doc_id,
                    user_query,
                    {"query": user_query, "answer": full_text, "documents": docs},
                )

            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "cached": False,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("스트리밍 중 오류")
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}

    return EventSourceResponse(event_gen())


@router.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(file.filename))
    bytes_written = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1 << 20):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"파일이 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 한도를 초과합니다.",
                    )
                out.write(chunk)

        state = _state(request)
        async with state.lock:
            await run_in_threadpool(state.attach_pdf, tmp_path)
        return UploadResponse(
            doc_id=state.doc_id or "",
            doc_name=state.doc_name or file.filename,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _to_response(
    query_text: str, result: Dict[str, Any], *, cached: bool, started: float
) -> QueryResponse:
    docs = result.get("documents") or []
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return QueryResponse(
        query=query_text,
        answer=(result.get("answer") or "").strip(),
        documents=[
            DocumentRef(page_content=d.page_content, metadata=dict(d.metadata or {}))
            for d in docs
        ],
        cached=cached,
        elapsed_ms=elapsed_ms,
    )
