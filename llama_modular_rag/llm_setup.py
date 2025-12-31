import torch
from typing import Any
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import LLAMA_MODEL_PATH, DEVICE, TEMPERATURE, MAX_NEW_TOKENS


def setup_llama_model() -> HuggingFacePipeline:
    """CPU에 최적화된 Llama 모델 설정"""
    print("🔧 Llama 모델을 CPU 모드로 로드 중...")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)

    # 모델 로드 최적화 옵션 (CPU 강제)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float32,  # CPU에서는 float32 사용
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},  # 명시적으로 CPU 지정
        trust_remote_code=True
    )

    # 모델을 CPU로 명시적 이동 및 최적화
    model = model.to("cpu")
    model.eval()  # 추론 모드로 설정
    
    # CUDA 텐서가 있는지 확인
    if next(model.parameters()).is_cuda:
        print("⚠️  경고: 모델이 GPU에 있습니다. CPU로 이동합니다.")
        model = model.cpu()
    
    print(f"✅ 모델이 CPU에 로드되었습니다. 디바이스: {next(model.parameters()).device}")

    # 파이프라인 생성 (CPU 강제)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,  # 토큰 생성 제한
        temperature=TEMPERATURE,
        repetition_penalty=1.1,
        batch_size=1,  # CPU에서는 작은 배치 사이즈
        device=-1  # CPU 사용 (-1은 CPU를 의미)
    )

    # LangChain 모델 생성
    llm: HuggingFacePipeline = HuggingFacePipeline(pipeline=pipe)
    return llm
