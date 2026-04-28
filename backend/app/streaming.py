"""SSE 토큰 스트리밍.

`TextIteratorStreamer`로 토큰 단위 디코드 결과를 받고,
백그라운드 스레드에서 ``model.generate``를 돌려 이벤트 루프를 막지 않는다.
"""
from __future__ import annotations

import asyncio
import logging
from threading import Thread
from typing import AsyncIterator

from transformers import TextIteratorStreamer

from llama_modular_rag.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from llama_modular_rag.generation import ANSWER_PROMPT_TEXT
from llama_modular_rag.llm_setup import get_llama_model, get_llama_tokenizer

logger = logging.getLogger(__name__)

_DONE = object()


def build_prompt(context: str, query: str) -> str:
    return ANSWER_PROMPT_TEXT.format(context=context, query=query)


async def stream_answer_tokens(prompt_text: str) -> AsyncIterator[str]:
    """프롬프트 텍스트로부터 디코드된 텍스트 청크를 비동기로 yield한다."""
    tokenizer = get_llama_tokenizer()
    model = get_llama_model()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
    thread.start()

    try:
        while True:
            chunk = await asyncio.to_thread(next, streamer, _DONE)
            if chunk is _DONE:
                break
            if chunk:
                yield chunk
    finally:
        # generate 스레드가 max_new_tokens까지 자연 종료되도록 둔다.
        # 클라이언트가 끊어도 generation은 끝까지 돌지만 SSE 송신은 중단됨.
        thread.join(timeout=0.0)
        if thread.is_alive():
            logger.debug("generation 스레드가 백그라운드에서 계속 진행 중")
