from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from config import LLAMA_MODEL_PATH, DEVICE, TEMPERATURE, MAX_NEW_TOKENS


def setup_llama_model():
    """CPU에 최적화된 Llama 모델 설정"""
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)

    # 모델 로드 최적화 옵션
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    # 모델 최적화 (CPU)
    model = model.to("cpu")
    model.eval()  # 추론 모드로 설정

    # 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,  # 토큰 생성 제한
        temperature=TEMPERATURE,
        repetition_penalty=1.1,
        batch_size=1  # CPU에서는 작은 배치 사이즈
    )

    # LangChain 모델 생성
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm