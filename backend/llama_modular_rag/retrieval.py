from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from llama_modular_rag.config import CONTEXT_MAX_TOKENS, RETRIEVAL_TOP_K
from llama_modular_rag.llm_setup import get_llama_tokenizer
from llama_modular_rag.state import RAGState


def document_retriever(state: RAGState, vectorstore: Chroma) -> RAGState:
    """벡터 유사도 검색으로 상위 K개 문서를 가져온다."""
    documents: List[Document] = vectorstore.similarity_search(
        state["query"], k=RETRIEVAL_TOP_K
    )
    return {**state, "documents": documents}


def context_builder(state: RAGState) -> RAGState:
    """LLM 토크나이저로 실제 토큰 수를 측정해 컨텍스트를 구성한다."""
    documents = state.get("documents")
    if not documents:
        return {**state, "context": ""}

    tokenizer = get_llama_tokenizer()
    max_tokens: int = CONTEXT_MAX_TOKENS
    context_parts: List[str] = []
    used_tokens: int = 0

    for i, doc in enumerate(documents):
        chunk: str = f"문서 {i + 1}:\n{doc.page_content}\n"
        chunk_tokens: int = len(tokenizer.encode(chunk, add_special_tokens=False))
        if used_tokens + chunk_tokens > max_tokens:
            break
        context_parts.append(chunk)
        used_tokens += chunk_tokens

    return {**state, "context": "\n".join(context_parts)}
