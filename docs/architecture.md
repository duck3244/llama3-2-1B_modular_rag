# 아키텍처 문서

`Llama 3.2 Modular RAG` 프로젝트의 시스템 구성, 모듈 책임, 런타임 흐름을 정리한 문서입니다.
프론트엔드(React + Vite)와 백엔드(FastAPI + LangGraph)로 구성된 dev 모노리포 형태이며,
CPU 단일 프로세스에서 한국어 PDF 기반 RAG 응답을 제공합니다.

---

## 1. 큰 그림 (High-level)

```
┌──────────────────────────────┐        HTTP / SSE         ┌──────────────────────────────────┐
│  Browser                     │  ───────────────────────▶ │  FastAPI (uvicorn, workers=1)    │
│  React 18 + Vite + Tailwind  │                            │  app/main.py · app/api/routes.py │
│  - ChatPanel / DocumentPanel │  ◀───────────────────────  │                                   │
│  - useRagQuery (SSE client)  │       /api/* responses     │  - lifespan → AppState           │
└──────────────────────────────┘                            │  - asyncio.Lock으로 추론 직렬화   │
                                                            └──────────────┬───────────────────┘
                                                                            │
                                                                            ▼
                                              ┌──────────────────────────────────────────────┐
                                              │  llama_modular_rag (RAG 패키지)               │
                                              │                                              │
                                              │  config → embeddings → data_loader (Chroma)  │
                                              │       │                                      │
                                              │       ▼                                      │
                                              │  graph_builder → LangGraph(StateGraph)       │
                                              │   ├─ document_retriever (similarity search)  │
                                              │   ├─ context_builder   (token budget cut)    │
                                              │   └─ answer_generator  (Llama 3.2 1B 한국어)  │
                                              │                                              │
                                              │  caching (sha256(doc_id::query) → JSON)      │
                                              └──────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
                                                            ┌─────────────────────────────┐
                                                            │  로컬 자원                   │
                                                            │  - models/  (HF 가중치)      │
                                                            │  - vector_db/ (Chroma 영속)  │
                                                            │  - cache/  (쿼리 응답 JSON) │
                                                            └─────────────────────────────┘
```

---

## 2. 디렉터리 구조

```
.
├── backend/
│   ├── app/                         # FastAPI 어댑터 레이어
│   │   ├── main.py                  # 앱 부트스트랩, lifespan, init_runtime() 호출
│   │   ├── deps.py                  # AppState (vectorstore/graph/cache/lock)
│   │   ├── streaming.py             # TextIteratorStreamer 기반 토큰 스트리밍
│   │   └── api/
│   │       ├── routes.py            # /api/health, /api/query, /api/query/stream, /api/upload
│   │       └── schemas.py           # Pydantic v2 요청/응답 모델
│   │
│   ├── llama_modular_rag/           # 프레임워크 독립 RAG 코어
│   │   ├── config.py                # 경로/하이퍼파라미터, init_runtime() (CPU·thread 설정)
│   │   ├── embeddings.py            # ko-sroberta (lru_cache singleton)
│   │   ├── llm_setup.py             # Llama 3.2 1B 토크나이저/모델/HF 파이프라인 (lru_cache)
│   │   ├── data_loader.py           # PDF → split → Chroma persist (doc_id sha256)
│   │   ├── retrieval.py             # similarity_search + 토크나이저 기반 컨텍스트 빌더
│   │   ├── generation.py            # PromptTemplate | LLM | StrOutputParser 체인
│   │   ├── state.py                 # RAGState TypedDict
│   │   ├── graph_builder.py         # LangGraph StateGraph 컴파일
│   │   ├── caching.py               # JSON 파일 기반 쿼리 응답 캐시
│   │   └── main.py                  # CLI 진입점 (단독 실행)
│   │
│   ├── models/                      # ko-sroberta-multitask, Llama-3.2-Korean-GGACHI-1B
│   ├── vector_db/                   # Chroma 영속 디렉터리 (doc_id별 분리)
│   └── cache/                       # 쿼리 응답 캐시 (.json)
│
├── frontend/                        # React 18 + Vite 5 + TS 5 + Tailwind 3
│   └── src/
│       ├── main.tsx · App.tsx
│       ├── components/              # ChatPanel · DocumentPanel · MessageList · MessageBubble · InputBox
│       ├── hooks/useRagQuery.ts     # SSE 구독 + AbortController
│       ├── lib/api.ts               # fetch 래퍼 (REST + SSE)
│       ├── lib/sse.ts               # text/event-stream 파서
│       └── types.ts                 # 백엔드 스키마와 동일한 타입 미러
│
└── docs/                            # 본 문서
```

---

## 3. 레이어 구분 및 책임

### 3.1 Adapter Layer (`backend/app/`)
HTTP/SSE 경계를 처리하는 얇은 레이어. RAG 코어를 직접 import 하되, 비즈니스 로직은 두지 않습니다.

| 파일 | 책임 |
| --- | --- |
| `app/main.py` | `init_runtime()`을 가장 먼저 호출, FastAPI 앱과 lifespan 정의. 환경변수 `RAG_DEFAULT_PDF`가 있으면 시작 시 인덱싱. |
| `app/deps.py` | `AppState` 데이터클래스. `vectorstore`, `graph`, `cache`, `doc_id`, `doc_name`, `asyncio.Lock`을 단일 인스턴스로 보관. `attach_pdf()`는 동기 헬퍼. |
| `app/api/routes.py` | 엔드포인트 4종. `run_in_threadpool` + `state.lock`으로 CPU 추론을 직렬화. `/query/stream`은 `EventSourceResponse`로 SSE. |
| `app/api/schemas.py` | Pydantic v2 모델 (`QueryRequest`, `QueryResponse`, `HealthResponse`, `UploadResponse`, `DocumentRef`). |
| `app/streaming.py` | `TextIteratorStreamer` + 백그라운드 `Thread`로 `model.generate`를 실행하고 토큰 청크를 비동기 yield. |

### 3.2 RAG Core (`backend/llama_modular_rag/`)
프레임워크에 독립적이며 CLI에서도 그대로 재사용 가능한 패키지입니다.

| 모듈 | 책임 | 핵심 결정 |
| --- | --- | --- |
| `config.py` | 경로/하이퍼파라미터 상수. `init_runtime(num_threads)`만이 부수효과(CUDA off, OMP/MKL/torch 스레드 설정) 수행. | 모듈 import 만으로는 환경 변수에 손대지 않음 — 다중 진입점에서 재현성 확보. |
| `embeddings.py` | `HuggingFaceEmbeddings` (`ko-sroberta-multitask`, normalized, batch=8, CPU). | `@lru_cache(maxsize=1)`로 프로세스당 한 번만 로드. |
| `llm_setup.py` | `get_llama_tokenizer()`, `get_llama_model()`, `setup_llama_model()` — 모두 lru_cache. raw 모델은 streaming, HF pipeline은 LangGraph가 사용. | `device_map={"": "cpu"}`로 CPU 강제. `pipeline()`에 `device=` 인자 안 줘서 accelerate 충돌 회피. |
| `data_loader.py` | `compute_doc_id(file)` = sha256(파일 바이트). `create_vectorstore_from_pdf()`은 PyPDF → `RecursiveCharacterTextSplitter` → Chroma persist. 컬렉션 이름은 `doc_<sha32>`. | doc_id가 같으면 기존 Chroma 디렉터리 재사용. 한국어 파일명도 컬렉션 이름 제약 통과. |
| `state.py` | `RAGState` TypedDict (`query`, `documents`, `context`, `answer`, `feedback`). | LangGraph 노드들이 공유하는 dict 형태 상태. |
| `retrieval.py` | `document_retriever`(top-k similarity)와 `context_builder`(LLM 토크나이저로 실제 토큰 수 계산하며 자름). | `CONTEXT_MAX_TOKENS=512`로 1B 모델 컨텍스트에 맞게 컷. |
| `generation.py` | `_ANSWER_PROMPT | setup_llama_model() | StrOutputParser()` LCEL 체인. | 프롬프트 템플릿은 `ANSWER_PROMPT_TEXT`로 export — SSE 경로(`app/streaming.py`)도 같은 텍스트 사용해 일관성. |
| `graph_builder.py` | `StateGraph(RAGState)`에 “문서 검색 → 컨텍스트 생성 → 답변 생성 → END” 직선 흐름 컴파일. | 한국어 노드명이지만 LangGraph 내부 식별자로만 사용. |
| `caching.py` | `QueryCache` — 키 = `sha256(doc_id || "::" || query)`, 값 = `{"_v": 2, "data": {...}}`. `Document`는 `page_content/metadata`로 직렬화·역직렬화. | 스키마 버전이 달라지면 자동 미스로 처리. |
| `main.py` | CLI 진입점. `init_runtime()` 호출 후 PDF 인덱싱 → 캐시 확인 → 그래프 invoke → 시각화(graphviz). | FastAPI가 죽어 있어도 RAG 파이프라인 단독 검증 가능. |

### 3.3 Frontend (`frontend/src/`)

| 파일 | 책임 |
| --- | --- |
| `main.tsx` / `App.tsx` | React StrictMode 진입, `ChatPanel` 마운트. |
| `components/ChatPanel.tsx` | 좌측 `DocumentPanel`, 본문 메시지 리스트, 하단 `InputBox`. 5초 폴링으로 `/api/health` 갱신. |
| `components/DocumentPanel.tsx` | PDF 파일 업로드(`multipart/form-data`) UI. 활성 문서 정보 표시. |
| `components/MessageList.tsx` | 메시지 스크롤 컨테이너 (자동 하단 스크롤). |
| `components/MessageBubble.tsx` | user/assistant 말풍선. 스트리밍 중에는 깜빡이는 커서 표시. cached 여부와 elapsed_ms 표시. |
| `components/InputBox.tsx` | textarea + Enter 전송 / Shift+Enter 줄바꿈. pending 시 “취소” 버튼으로 토글. |
| `hooks/useRagQuery.ts` | 메시지 상태 관리. `streamQuery()`로 SSE 구독 → `onDocs/onToken/onDone/onError` 콜백으로 메시지 패치. `AbortController`로 취소. |
| `lib/api.ts` | `getHealth`, `postQuery`, `uploadPdf`, `streamQuery` (SSE) — 단순 fetch 래퍼. |
| `lib/sse.ts` | `ReadableStream<Uint8Array>` → `AsyncGenerator<SSEEvent>` 파서. `event:`/`data:` 라인 누적, 빈 줄에 dispatch. |
| `types.ts` | 백엔드 Pydantic 스키마와 1:1 미러링. |

---

## 4. 런타임 흐름

### 4.1 부팅
1. `uvicorn app.main:app` 시작 → `app/main.py`가 가장 먼저 `init_runtime()` 호출
   → `CUDA_VISIBLE_DEVICES=""`, `OMP/MKL/...=N`, `torch.set_num_threads(N)` 적용.
2. `lifespan(app)`이 `AppState()` 생성 후 `app.state.rag`에 부착.
3. `RAG_DEFAULT_PDF`가 설정돼 있으면 `state.attach_pdf(path)` — 백그라운드가 아니라 부팅 동기 실행이므로
   첫 요청 latency가 줄어듦 (대신 부팅이 길어짐).

### 4.2 PDF 업로드 (`POST /api/upload`)
1. multipart 스트림을 1MB 청크로 받으며 `MAX_UPLOAD_BYTES` 검증 (기본 50MB).
2. tempdir에 저장 후 `state.lock` 안에서 `state.attach_pdf(tmp)` 실행
   → `compute_doc_id` (파일 sha256) → 동일 doc_id면 기존 Chroma 재사용, 아니면 split + 영속.
3. `vectorstore`, `doc_id`, `doc_name`, `graph`(LangGraph 컴파일) 갱신.
4. tempdir 정리 후 `UploadResponse` 반환.

### 4.3 비스트리밍 쿼리 (`POST /api/query`)
1. `state.ready` 검사 (그래프와 doc_id 둘 다 있어야 함).
2. **락 밖**에서 캐시 lookup → 히트면 즉시 응답.
3. **락 안**에서 다시 lookup (double-checked locking) → 미스면 `run_in_threadpool(graph.invoke, ...)` 실행.
4. 결과를 `cache_result`로 영속 후 `QueryResponse` 반환.

### 4.4 스트리밍 쿼리 (`POST /api/query/stream`, SSE)
1. similarity_search → `docs` 이벤트 1회 송출.
2. 캐시 히트면 전체 답변을 단일 `token` 이벤트로 보내고 `done`.
3. 미스면 락 안에서 `context_builder` → `build_prompt` → `stream_answer_tokens()`로 토큰 단위 yield,
   각 청크를 `token` 이벤트로 송출. 매 루프마다 `request.is_disconnected()` 확인 후 끊기면 송신 중단.
4. 누적 텍스트를 캐시에 저장하고 `done` 이벤트(`cached: false`, `elapsed_ms`).
5. 예외 시 `error` 이벤트.

### 4.5 LangGraph 노드 실행 (RAG 코어)
```
{query}
  → document_retriever(state, vectorstore)   # vectorstore.similarity_search(k=2)
  → context_builder(state)                   # tokenizer.encode로 실제 길이 측정, 512 토큰 컷
  → answer_generator(state)                  # PromptTemplate | HF pipeline | StrOutputParser
  → END
```
LangGraph는 dict-merge 방식으로 상태를 누적하므로 각 노드는 `{**state, ...}` 패턴으로 새 키만 추가합니다.

---

## 5. 핵심 설계 결정

- **단일 워커 + 단일 모델 인스턴스**
  Llama 1B 모델/토크나이저/임베딩은 `lru_cache(maxsize=1)`로 프로세스당 1개. uvicorn은 `--workers 1`로 운용.
  추론은 `asyncio.Lock`으로 직렬화 — CPU에서 동시 호출은 OOM/스루풋 모두에 손해.
- **부수효과 격리 (`init_runtime`)**
  CUDA 비활성화/OMP 스레드 수/`torch.set_num_threads`는 모듈 import 부수효과로 두지 않고 명시 호출.
  FastAPI 진입점과 CLI 진입점 양쪽에서 첫 줄에 호출. 이중 호출은 idempotent.
- **doc_id = 파일 sha256**
  같은 PDF는 항상 같은 doc_id → Chroma 디렉터리 재사용 + 캐시 키 재사용.
  컬렉션 이름은 Chroma 제약(영숫자 3–63자) 때문에 `doc_<sha32>`로 정규화 — 한국어 파일명도 안전.
- **캐시 키 = sha256(doc_id || "::" || query)**
  같은 질문이라도 문서가 다르면 자동으로 분리. JSON 파일 기반이라 외부 의존성 0.
  스키마 버전(`_v`)이 다르면 미스 처리 → 포맷 변경 시 수동 비우기 불필요.
- **컨텍스트 컷은 LLM 토크나이저 기준**
  임베딩 토크나이저가 아니라 답변 모델의 토크나이저로 카운트 → 실제 모델이 보는 길이로 제어.
- **스트리밍 = TextIteratorStreamer + 백그라운드 Thread**
  `model.generate`를 worker thread에서 돌려 이벤트 루프 차단을 피하고, 메인은 `asyncio.to_thread(next, streamer, _DONE)`으로 토큰을 한 개씩 꺼냄.
  클라이언트가 끊어도 `generate`는 자연 종료되도록 그대로 둠 (락 점유 시간이 결정적이라 예측 가능).
- **공유 프롬프트 텍스트**
  `ANSWER_PROMPT_TEXT`를 export하여 비스트리밍(LCEL 체인)과 스트리밍(`streaming.build_prompt`) 양쪽이 동일 프롬프트 사용.
- **CORS 없는 dev 모노리포**
  Vite의 `server.proxy`가 `/api`를 8000으로 포워딩 → 백엔드에 CORS 미들웨어 불필요.

---

## 6. 외부 의존성 한눈에

| 영역 | 라이브러리 |
| --- | --- |
| 백엔드 웹 | FastAPI, uvicorn, sse-starlette |
| RAG 파이프라인 | langchain, langchain-community, langchain-huggingface, langgraph |
| 모델/토크나이저 | transformers, torch (CPU only) |
| 벡터스토어 | chromadb (langchain-community 어댑터 경유) |
| PDF | pypdf (PyPDFLoader) |
| 임베딩 모델 | sentence-transformers (`ko-sroberta-multitask`) |
| 프론트 | React 18, Vite 5, TypeScript 5, TailwindCSS 3 |

---

## 7. 환경 변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `RAG_DEFAULT_PDF` | (없음) | 부팅 시 자동 인덱싱할 PDF 경로 |
| `RAG_NUM_THREADS` | `os.cpu_count()` | torch/OMP/MKL 스레드 수 |
| `MAX_UPLOAD_MB` | `50` | 업로드 PDF 최대 크기 (MB) |
| `LOG_LEVEL` | `INFO` | 로깅 레벨 |
| `VITE_BACKEND_URL` | `http://localhost:8000` | (frontend) Vite proxy 대상 |
| `VITE_API_BASE` | `/api` | (frontend) fetch base — 프록시를 안 쓸 때 절대 URL로 덮을 수 있음 |

---

## 8. 확장 포인트

- **다른 LLM/임베딩 모델**: `config.py`의 `LLAMA_MODEL_PATH`, `EMBEDDING_MODEL_NAME`만 바꾸면 lru_cache 싱글톤이 새 모델을 로드.
- **그래프 분기 추가**: `graph_builder.py`에 노드/엣지 추가 — `RAGState`에 새 키만 정의하면 다른 모듈은 영향받지 않음 (`feedback` 키는 이미 예약되어 있음).
- **다른 벡터스토어**: `data_loader.create_vectorstore_from_pdf`만 교체하면 됨. 반환 타입을 LangChain `VectorStore` 인터페이스로 통일하는 것을 권장.
- **외부 캐시**: `caching.QueryCache`를 동일 메서드 시그니처(`get_cached_result`, `cache_result`)의 다른 구현으로 교체 (Redis 등).
