# Llama 3.2 & LangGraph를 활용한 모듈형 RAG 시스템

GPU 가속 없이 일반 컴퓨터에서 효율적으로 실행되도록 설계된 CPU 최적화 검색 증강 생성(RAG) 시스템입니다. LangChain과 LangGraph를 기반으로 구축되었습니다.

## 주요 기능

- **모듈형 설계**: 문서 로딩, 임베딩, 검색, 생성, 캐싱을 위한 독립적인 컴포넌트
- **CPU 최적화**: GPU 가속 없이 CPU 환경에서 효율적으로 실행
- **한국어 지원**: 한국어 언어 모델 및 문서 처리 지원
- **벡터 데이터베이스**: 더 빠른 처리를 위한 문서 임베딩의 영구 저장
- **쿼리 캐싱**: 이전 쿼리 결과를 저장하여 응답 시간 개선
- **메모리 관리**: 가비지 컬렉션 및 모니터링을 통한 최적화된 메모리 사용
- **시각화**: RAG 파이프라인 워크플로우의 그래프 시각화

## 프로젝트 구조
```
llama_modular_rag/
├── config.py               # 환경 설정 및 상수
├── data_loader.py          # PDF 로딩 및 벡터 저장소 생성
├── embeddings.py           # 임베딩 모델 설정
├── caching.py              # 캐싱 메커니즘
├── llm_setup.py            # LLM 모델 설정
├── query_processing.py     # 쿼리 처리
├── retrieval.py            # 문서 검색 및 필터링
├── generation.py           # 답변 생성
├── state.py                # 상태 관리
├── graph_builder.py        # 그래프 구성
└── main.py                 # 메인 실행 파일
```

## 설치 방법

### 1. Conda를 사용한 환경 생성 (권장)
```bash
conda env create -f environment.yaml
conda activate rag-env
```

### 2. 또는 pip를 사용한 패키지 설치
```bash
# 먼저 Python 3.10을 사용하고 있는지 확인
python --version  # Python 3.10.x가 표시되어야 함

# 종속성 설치
pip install -r requirements.txt
```

### 3. Python 버전 확인
```bash
python --version
```

### 4. 환경 검증
```bash
python check_environment.py
```

## 사용 방법

### 1. PDF 문서를 프로젝트 디렉토리에 배치
### 2. 메인 스크립트 실행:

```bash
python llama_modular_rag/main.py
```

### 3. 프롬프트가 표시되면 PDF 파일 경로와 질문 입력
### 4. 생성된 답변 및 참조 문서 확인

## 설정

### config.py에서 하드웨어에 맞게 설정 조정:
```python
# CPU 스레드 설정 - CPU에 따라 조정
os.environ["OMP_NUM_THREADS"] = "8"       # 연산 라이브러리 스레드
os.environ["MKL_NUM_THREADS"] = "8"       # Intel MKL 라이브러리 스레드
```

## 사용 모델
### 이 프로젝트는 두 가지 주요 모델을 사용합니다:

- **Llama 3.2 Korean GGACHI 1B Instruct**: 텍스트 생성용
- **Ko-sRoBERTa-multitask**: 임베딩용

### config.py에서 경로를 수정하여 다른 모델로 교체할 수 있습니다.

## 성능 최적화
### CPU 환경을 위한 여러 최적화가 포함되어 있습니다:

- 토큰 생성 제한 감소
- 더 작은 문서 청크
- 쿼리 결과 캐싱
- 메모리 관리 및 모니터링
- 필수 단계만 포함한 간소화된 워크플로우

## Python 3.10 관련 변경사항

- Python 3.10 구문과 호환되는 타입 힌트
- Python 3.10에서 테스트된 종속성 버전
- Python 3.11+ 전용 기능 미사용 (예: 함수 시그니처의 `|` 타입 유니온 연산자 미사용)
- 모든 요구사항에 대한 호환 라이브러리 버전

## 종속성
### 주요 종속성:

- langchain & langgraph (호환 버전)
- transformers (<5.0.0)
- sentence-transformers (<4.0.0)
- chromadb (<0.6.0)
- pypdf (<4.0.0)
- torch (<2.6.0)
- graphviz
- psutil

## 문제 해결

### 버전 충돌
버전 충돌이 발생하는 경우:
```bash
# 기존 환경 제거
conda env remove -n rag-env

# 정확한 버전으로 재생성
conda env create -f environment.yaml
```

### ImportError
임포트 오류가 발생하는 경우, Python 3.10을 사용하고 있는지 확인:
```bash
python --version
# 3.10이 아닌 경우, 올바른 Python 버전으로 환경 재생성
```

## 설치 완료 후 실행

```bash
# 환경 활성화 (Conda)
conda activate rag-env

# 환경 활성화 (venv)
source venv/bin/activate  # Windows: venv\Scripts\activate

# 프로그램 실행
python llama_modular_rag/main.py
```

## 디렉토리 설명

- `cache/`: 쿼리 결과 캐시 저장
- `vector_db/`: 벡터 데이터베이스 저장
- `models/`: 다운로드한 모델 파일
- `visualizations/`: 생성된 그래프 이미지

## 추가 문서

- **설치 가이드**: `INSTALL_GUIDE.md` - 상세한 설치 지침 및 문제 해결
- **변경 이력**: `CHANGELOG.md` - 버전 변경 사항 및 마이그레이션 가이드
- **파일 변경 사항**: `FILE_CHANGES.md` - Python 3.10 마이그레이션 상세 내역

## 사용 예시

```python
# main.py에서 PDF 경로와 쿼리 수정
pdf_path = "your_document.pdf"
query = "문서에서 중요한 정보를 요약해주세요"

# 실행
python llama_modular_rag/main.py
```

## 라이선스
MIT License
