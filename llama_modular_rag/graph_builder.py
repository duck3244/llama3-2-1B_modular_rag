from typing import Callable, Any
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from state import RAGState
from retrieval import document_retriever, context_builder
from generation import answer_generator


def build_rag_graph(vectorstore: Chroma) -> Any:
    """최소한의 노드만 사용하는 간소화된 RAG 워크플로우"""
    # 상태 그래프 생성
    graph = StateGraph(RAGState)

    # 최소한의 노드만 추가 - 노드 이름을 보기 좋게 설정
    graph.add_node(
        "문서 검색",
        lambda state: document_retriever(state, vectorstore)
    )
    graph.add_node("컨텍스트 생성", context_builder)
    graph.add_node("답변 생성", answer_generator)

    # 직선형 흐름으로 단순화
    graph.add_edge("문서 검색", "컨텍스트 생성")
    graph.add_edge("컨텍스트 생성", "답변 생성")
    graph.add_edge("답변 생성", END)

    # 시작 노드 설정
    graph.set_entry_point("문서 검색")

    return graph.compile()
