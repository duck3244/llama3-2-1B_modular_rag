# UML 다이어그램

`Llama 3.2 Modular RAG`의 정적 구조와 동적 흐름을 Mermaid 다이어그램으로 정리했습니다.
GitHub/GitLab/Mermaid Live Editor 등에서 그대로 렌더링됩니다.

> **표기 규칙**
> - 클래스 다이어그램은 백엔드 RAG 코어와 FastAPI 어댑터 레이어를 중심으로 작성합니다.
> - 시퀀스 다이어그램은 “업로드 → 비스트리밍 쿼리 → 스트리밍 쿼리” 세 시나리오를 다룹니다.
> - 컴포넌트 다이어그램은 프론트/백엔드/외부 자원의 배치를 보여줍니다.

---

## 1. 컴포넌트 다이어그램 (System Context)

```mermaid
flowchart LR
    subgraph Browser["Browser (React 18 + Vite)"]
        UI[ChatPanel / DocumentPanel / InputBox]
        Hook[useRagQuery]
        ApiClient[lib/api.ts]
        SSE[lib/sse.ts]
        UI --> Hook --> ApiClient
        ApiClient -- "SSE stream" --> SSE
    end

    subgraph FastAPI["FastAPI (uvicorn, workers=1)"]
        Main[app/main.py<br/>lifespan + init_runtime]
        Routes[app/api/routes.py]
        Streaming[app/streaming.py]
        AppState[(AppState<br/>vectorstore · graph<br/>cache · doc_id · lock)]
        Main --> Routes
        Routes --> AppState
        Routes --> Streaming
    end

    subgraph Core["llama_modular_rag (RAG core)"]
        Config[config.py]
        Embed[embeddings.py]
        LLM[llm_setup.py]
        Loader[data_loader.py]
        Retr[retrieval.py]
        Gen[generation.py]
        Graph[graph_builder.py]
        Cache[caching.py]
        State[state.py]
    end

    subgraph Storage["Local Storage"]
        Models[(models/<br/>HF weights)]
        Chroma[(vector_db/<br/>Chroma persist)]
        JSONCache[(cache/<br/>response JSON)]
    end

    Browser -- "/api/health<br/>/api/upload<br/>/api/query<br/>/api/query/stream (SSE)" --> FastAPI
    Routes --> Loader
    Routes --> Graph
    Routes --> Cache
    Streaming --> LLM
    Streaming --> Gen
    Loader --> Embed
    Graph --> Retr
    Graph --> Gen
    Retr --> LLM
    Gen --> LLM
    Embed --> Models
    LLM --> Models
    Loader --> Chroma
    Cache --> JSONCache
```

---

## 2. 클래스 다이어그램 (백엔드)

### 2.1 RAG 코어 + 어댑터 레이어

```mermaid
classDiagram
    direction TB

    class AppState {
        +QueryCache cache
        +Optional vectorstore
        +Optional graph
        +Optional~str~ doc_id
        +Optional~str~ doc_name
        +asyncio.Lock lock
        +bool ready
        +attach_pdf(pdf_path: str) void
    }

    class QueryCache {
        -str cache_dir
        +get_cached_result(doc_id, query) Optional~Dict~
        +cache_result(doc_id, query, result) void
        -_path(doc_id, query) str
    }

    class RAGState {
        <<TypedDict>>
        +str query
        +Optional~List~Document~~ documents
        +Optional~str~ context
        +Optional~str~ answer
        +Optional~Dict~ feedback
    }

    class StateGraph {
        <<LangGraph>>
        +add_node(name, fn)
        +add_edge(src, dst)
        +set_entry_point(name)
        +compile() CompiledGraph
    }

    class CompiledGraph {
        +invoke(state: RAGState) RAGState
        +get_graph() RuntimeGraph
    }

    class document_retriever {
        <<function>>
        +__call__(state, vectorstore) RAGState
    }

    class context_builder {
        <<function>>
        +__call__(state) RAGState
    }

    class answer_generator {
        <<function>>
        +__call__(state) RAGState
    }

    class HuggingFacePipeline {
        <<LangChain>>
    }

    class HuggingFaceEmbeddings {
        <<LangChain>>
    }

    class Chroma {
        <<LangChain>>
        +similarity_search(query, k) List~Document~
        +from_documents(docs, embedding, persist_directory, collection_name) Chroma
    }

    class data_loader {
        <<module>>
        +compute_doc_id(file_path) str
        +create_vectorstore_from_pdf(pdf_path) Tuple~Chroma, str~
        -_collection_name(doc_id) str
    }

    class config {
        <<module>>
        +LLAMA_MODEL_PATH: str
        +EMBEDDING_MODEL_NAME: str
        +RETRIEVAL_TOP_K: int
        +CONTEXT_MAX_TOKENS: int
        +CHUNK_SIZE: int
        +CHUNK_OVERLAP: int
        +TEMPERATURE: float
        +TOP_P: float
        +MAX_NEW_TOKENS: int
        +CACHE_DIR: str
        +VECTOR_DB_PATH: str
        +init_runtime(num_threads?) void
    }

    class llm_setup {
        <<module · lru_cache=1>>
        +get_llama_tokenizer() PreTrainedTokenizerBase
        +get_llama_model() PreTrainedModel
        +setup_llama_model() HuggingFacePipeline
    }

    class embeddings {
        <<module · lru_cache=1>>
        +get_embedding_model() HuggingFaceEmbeddings
    }

    class generation {
        <<module>>
        +ANSWER_PROMPT_TEXT: str
        +answer_generator(state) RAGState
    }

    class graph_builder {
        <<module>>
        +build_rag_graph(vectorstore) CompiledGraph
    }

    class streaming {
        <<module>>
        +build_prompt(context, query) str
        +stream_answer_tokens(prompt) AsyncIterator~str~
    }

    AppState --> QueryCache : owns
    AppState --> Chroma : holds
    AppState --> CompiledGraph : holds
    StateGraph ..> CompiledGraph : compile()
    graph_builder ..> StateGraph : builds
    graph_builder ..> document_retriever
    graph_builder ..> context_builder
    graph_builder ..> answer_generator
    document_retriever ..> Chroma : uses
    document_retriever ..> RAGState
    context_builder ..> llm_setup : tokenizer
    context_builder ..> RAGState
    answer_generator ..> generation
    generation ..> llm_setup : pipeline
    generation ..> HuggingFacePipeline
    llm_setup ..> config
    embeddings ..> config
    data_loader ..> embeddings : get_embedding_model()
    data_loader ..> Chroma : creates / loads
    data_loader ..> config
    streaming ..> llm_setup
    streaming ..> generation : ANSWER_PROMPT_TEXT
```

### 2.2 FastAPI 라우터 / 스키마

```mermaid
classDiagram
    direction LR

    class FastAPIApp {
        +include_router(router)
        +state.rag: AppState
    }

    class APIRouter {
        +health(request) HealthResponse
        +query(request, payload) QueryResponse
        +query_stream(request, payload) EventSourceResponse
        +upload(request, file) UploadResponse
    }

    class HealthResponse {
        +str status
        +bool ready
        +Optional~str~ doc_id
        +Optional~str~ doc_name
    }

    class QueryRequest {
        +str query
    }

    class QueryResponse {
        +str query
        +str answer
        +List~DocumentRef~ documents
        +bool cached
        +int elapsed_ms
    }

    class DocumentRef {
        +str page_content
        +Dict metadata
    }

    class UploadResponse {
        +str doc_id
        +str doc_name
        +bool chunks_indexed
    }

    FastAPIApp --> APIRouter : mount
    APIRouter --> HealthResponse
    APIRouter --> QueryRequest
    APIRouter --> QueryResponse
    APIRouter --> UploadResponse
    QueryResponse --> DocumentRef
    APIRouter --> AppState : reads request.app.state.rag
```

---

## 3. 클래스 다이어그램 (프론트엔드)

```mermaid
classDiagram
    direction TB

    class App {
        +render() JSX
    }

    class ChatPanel {
        -HealthResponse? health
        +refreshHealth() Promise~void~
        +render() JSX
    }

    class DocumentPanel {
        -bool uploading
        -string? error
        +onChange(e) Promise~void~
        +render() JSX
    }

    class MessageList {
        +messages: ChatMessage[]
        +render() JSX
    }

    class MessageBubble {
        +message: ChatMessage
        +render() JSX
    }

    class InputBox {
        -string value
        +submit() void
        +onKeyDown(e) void
        +render() JSX
    }

    class useRagQuery {
        <<hook>>
        -ChatMessage[] messages
        -bool pending
        -AbortController? abortRef
        +send(query) Promise~void~
        +cancel() void
        +clear() void
    }

    class api {
        <<module>>
        +getHealth(signal?) Promise~HealthResponse~
        +postQuery(query, signal?) Promise~QueryResponse~
        +uploadPdf(file, signal?) Promise~UploadResponse~
        +streamQuery(query, handlers, signal?) Promise~void~
    }

    class StreamHandlers {
        +onDocs(docs)
        +onToken(text)
        +onDone(info)
        +onError(detail)
    }

    class parseSSE {
        <<async generator>>
        +(body: ReadableStream) AsyncGenerator~SSEEvent~
    }

    class ChatMessage {
        +string id
        +'user'|'assistant' role
        +string text
        +DocumentRef[]? documents
        +bool? cached
        +number? elapsedMs
        +bool? pending
        +string? error
    }

    App --> ChatPanel
    ChatPanel --> DocumentPanel
    ChatPanel --> MessageList
    ChatPanel --> InputBox
    ChatPanel --> useRagQuery : uses
    MessageList --> MessageBubble
    useRagQuery --> api : streamQuery
    api --> parseSSE
    api ..> StreamHandlers
    useRagQuery ..> ChatMessage
    DocumentPanel --> api : uploadPdf
    ChatPanel --> api : getHealth
```

---

## 4. 시퀀스 다이어그램

### 4.1 PDF 업로드

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as DocumentPanel (React)
    participant API as lib/api.ts
    participant FastAPI as POST /api/upload
    participant State as AppState
    participant Loader as data_loader
    participant Embed as embeddings
    participant Chroma as Chroma (persist)
    participant Builder as graph_builder

    User->>FE: PDF 파일 선택
    FE->>API: uploadPdf(file)
    API->>FastAPI: multipart/form-data
    FastAPI->>FastAPI: 1MB chunk write + size 검증
    FastAPI->>State: async with state.lock
    activate State
    State->>Loader: attach_pdf(tmp_path)
    Loader->>Loader: compute_doc_id(sha256)
    alt 기존 디렉터리 존재
        Loader->>Chroma: Chroma(persist_dir, collection)
    else 신규
        Loader->>Embed: get_embedding_model()
        Loader->>Loader: PyPDFLoader.load() + RecursiveCharacterTextSplitter
        Loader->>Chroma: Chroma.from_documents(...)
    end
    Loader-->>State: (vectorstore, doc_id)
    State->>Builder: build_rag_graph(vectorstore)
    Builder-->>State: CompiledGraph
    deactivate State
    FastAPI-->>API: UploadResponse { doc_id, doc_name }
    API-->>FE: response
    FE->>FE: onUploaded() → /api/health 재조회
```

### 4.2 비스트리밍 쿼리 (`POST /api/query`)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as ChatPanel
    participant API as lib/api.ts
    participant Route as routes.query
    participant State as AppState
    participant Cache as QueryCache
    participant Graph as CompiledGraph
    participant Retr as document_retriever
    participant Ctx as context_builder
    participant Gen as answer_generator

    User->>FE: 질문 입력
    FE->>API: postQuery(query)
    API->>Route: POST /api/query
    Route->>State: state.ready 검사
    Route->>Cache: get_cached_result(doc_id, query)
    alt 캐시 히트 (락 밖)
        Cache-->>Route: cached
        Route-->>API: QueryResponse(cached=true)
    else 미스
        Route->>State: async with state.lock
        Route->>Cache: 재확인 (double-check)
        alt 락 안 캐시 히트
            Cache-->>Route: cached
        else 진짜 미스
            Route->>Graph: run_in_threadpool(graph.invoke, {query})
            Graph->>Retr: document_retriever(state, vectorstore)
            Retr-->>Graph: state + documents
            Graph->>Ctx: context_builder(state)
            Ctx-->>Graph: state + context
            Graph->>Gen: answer_generator(state)
            Gen-->>Graph: state + answer
            Graph-->>Route: result
            Route->>Cache: cache_result(doc_id, query, result)
        end
        Route-->>API: QueryResponse(cached=false)
    end
    API-->>FE: response
    FE->>FE: 메시지 갱신
```

### 4.3 스트리밍 쿼리 (`POST /api/query/stream`, SSE)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as useRagQuery
    participant API as streamQuery
    participant SSE as parseSSE
    participant Route as routes.query_stream
    participant Vec as vectorstore
    participant Cache as QueryCache
    participant Stream as streaming.stream_answer_tokens
    participant Streamer as TextIteratorStreamer
    participant GenThread as Thread(model.generate)

    User->>FE: 질문 입력 → send(query)
    FE->>API: streamQuery(query, handlers, signal)
    API->>Route: POST /api/query/stream (Accept: text/event-stream)
    Route->>Vec: similarity_search(query, k=2)
    Vec-->>Route: docs
    Route-->>API: event: docs
    API->>SSE: parseSSE → onDocs(docs)
    SSE->>FE: onDocs → message.documents 갱신

    alt 캐시 히트
        Route->>Cache: get_cached_result
        Cache-->>Route: cached
        Route-->>API: event: token (전체 텍스트 1개)
        Route-->>API: event: done {cached:true, elapsed_ms}
    else 미스
        Route->>Route: async with state.lock
        Route->>Route: build_prompt(context, query)
        Route->>Stream: async for token in stream_answer_tokens(prompt)
        Stream->>GenThread: Thread(model.generate, streamer=streamer).start()
        loop 토큰마다
            GenThread-->>Streamer: 디코드된 청크
            Stream-->>Route: token (asyncio.to_thread(next, streamer))
            Route->>Route: request.is_disconnected() 체크
            alt 연결 살아있음
                Route-->>API: event: token "<chunk>"
                API->>SSE: onToken
                SSE->>FE: m.text += token
            else 연결 끊김
                Route->>Route: 송신 중단 (generate는 백그라운드 자연 종료)
            end
        end
        Route->>Cache: cache_result(...)
        Route-->>API: event: done {cached:false, elapsed_ms}
    end
    API->>SSE: 종료
    SSE->>FE: onDone → message.pending=false
```

### 4.4 취소 (AbortController)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as InputBox / useRagQuery
    participant Ctrl as AbortController
    participant API as streamQuery
    participant Server as FastAPI

    User->>FE: 취소 버튼 클릭
    FE->>Ctrl: ctrl.abort()
    Ctrl-->>API: signal.aborted = true
    API-->>API: fetch reader 종료 → AbortError throw
    API-->>FE: catch AbortError → "취소되었습니다"
    Note over Server: request.is_disconnected() == true<br/>SSE 송신 중단<br/>generate 스레드는 max_new_tokens까지 자연 종료
```

---

## 5. 상태 다이어그램 — RAG 워크플로우 (LangGraph)

```mermaid
stateDiagram-v2
    [*] --> 문서검색: invoke({query})
    문서검색: 문서 검색<br/>vectorstore.similarity_search(k=2)
    컨텍스트생성: 컨텍스트 생성<br/>tokenizer로 ≤512 토큰 컷
    답변생성: 답변 생성<br/>PromptTemplate | LLM | StrOutputParser
    문서검색 --> 컨텍스트생성: state + documents
    컨텍스트생성 --> 답변생성: state + context
    답변생성 --> [*]: state + answer
```

---

## 6. 활동 다이어그램 — `/api/query` 핸들러 결정 흐름

```mermaid
flowchart TD
    Start([POST /api/query]) --> Ready{state.ready?}
    Ready -- no --> E409[409 Conflict<br/>'문서가 활성화되지 않았습니다']
    Ready -- yes --> CacheL1[QueryCache.get<br/>락 밖]
    CacheL1 -- hit --> RespCached[QueryResponse cached=true]
    CacheL1 -- miss --> Lock[acquire state.lock]
    Lock --> CacheL2[QueryCache.get<br/>락 안 재확인]
    CacheL2 -- hit --> RespCached
    CacheL2 -- miss --> Invoke[run_in_threadpool<br/>graph.invoke]
    Invoke --> Save[QueryCache.cache_result]
    Save --> RespNew[QueryResponse cached=false]
    RespCached --> End([HTTP 200])
    RespNew --> End
    E409 --> End
```

---

## 7. 배포 다이어그램 (dev)

```mermaid
flowchart LR
    subgraph Dev["개발자 머신 (Linux/CPU)"]
        subgraph Node["Node 18+"]
            Vite["vite dev server :5173<br/>proxy /api → :8000"]
        end
        subgraph Py["Python 3.10 venv"]
            UV["uvicorn :8000<br/>workers=1<br/>--reload-dir app"]
        end
        FS[(파일시스템)]
        Models["models/ (HF 가중치)"]
        VDB["vector_db/ (Chroma)"]
        CACHE["cache/ (응답 JSON)"]
        FS --- Models
        FS --- VDB
        FS --- CACHE
        Vite -- HTTP --> UV
        UV -- read --> Models
        UV -- read/write --> VDB
        UV -- read/write --> CACHE
    end
    Browser((Browser)) -- :5173 --> Vite
```
