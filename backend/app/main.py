"""FastAPI 진입점."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from llama_modular_rag.config import init_runtime

# 라우터/모델 로딩보다 먼저 환경변수와 torch 스레드 수를 적용해야 한다.
init_runtime()

from fastapi import FastAPI  # noqa: E402

from app.api.routes import router  # noqa: E402
from app.deps import AppState  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.rag = state

    default_pdf = os.getenv("RAG_DEFAULT_PDF")
    if default_pdf and os.path.exists(default_pdf):
        try:
            state.attach_pdf(default_pdf)
            logger.info("기본 PDF 로드 완료: %s", default_pdf)
        except Exception:
            logger.exception("기본 PDF 로드 실패: %s", default_pdf)
    else:
        logger.info("기본 PDF 미설정 또는 존재하지 않음 — /api/upload 후 활성화됩니다")

    yield


app = FastAPI(title="Llama 3.2 Modular RAG", version="0.1.0", lifespan=lifespan)
app.include_router(router)
