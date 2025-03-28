from typing import TypedDict, List, Dict, Optional
from langchain.schema import Document

class RAGState(TypedDict):
    """RAG 파이프라인의 상태를 나타내는 클래스"""
    query: str
    documents: Optional[List[Document]]
    context: Optional[str]
    answer: Optional[str]
    feedback: Optional[Dict]