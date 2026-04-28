import logging
from functools import lru_cache

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
)

from llama_modular_rag.config import LLAMA_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llama_tokenizer() -> PreTrainedTokenizerBase:
    """프로세스 수명 동안 한 번만 로드되는 토크나이저."""
    return AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)


@lru_cache(maxsize=1)
def get_llama_model() -> PreTrainedModel:
    """프로세스 수명 동안 한 번만 로드되는 raw transformers 모델 (스트리밍용)."""
    logger.info("Llama 모델을 CPU 모드로 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        trust_remote_code=False,
    )
    model.eval()
    logger.info("모델이 CPU에 로드되었습니다. device=%s", next(model.parameters()).device)
    return model


@lru_cache(maxsize=1)
def setup_llama_model() -> HuggingFacePipeline:
    """LangGraph가 사용하는 LangChain 파이프라인. raw 모델/토크나이저를 공유한다."""
    tokenizer = get_llama_tokenizer()
    model = get_llama_model()

    # 모델은 device_map={"": "cpu"}로 이미 CPU에 적재돼 있다.
    # pipeline()에 device 인자를 같이 주면 accelerate와 충돌해 ValueError가 난다.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.1,
        batch_size=1,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)
