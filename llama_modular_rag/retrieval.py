from typing import List
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from state import RAGState
from config import RETRIEVAL_TOP_K, CONTEXT_MAX_TOKENS


def document_retriever(state: RAGState, vectorstore: Chroma) -> RAGState:
    """효율적인 문서 검색 구현"""
    query: str = state["query"]

    # similarity_search 사용 (Python 3.10 호환)
    documents: List[Document] = vectorstore.similarity_search(
        query,
        k=RETRIEVAL_TOP_K
    )

    return {**state, "documents": documents}


def context_builder(state: RAGState) -> RAGState:
    """효율적인 컨텍스트 생성 및 토큰 제한"""
    documents = state.get("documents")
    
    if not documents:
        return {**state, "context": ""}

    # 토큰 수 제한을 위한 설정
    max_tokens: int = CONTEXT_MAX_TOKENS
    context_parts: List[str] = []
    token_count: float = 0.0

    for i, doc in enumerate(documents):
        # 대략적인 토큰 수 추정 (영어 기준 평균 단어 4개가 3 토큰)
        content: str = doc.page_content
        approx_tokens: float = len(content.split()) * 0.75

        if token_count + approx_tokens > max_tokens:
            # 토큰 제한 도달
            break

        context_parts.append(f"문서 {i + 1}:\n{content}\n")
        token_count += approx_tokens

    context: str = "\n".join(context_parts)
    return {**state, "context": context}
