import gc
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# `python llama_modular_rag/main.py`로 직접 실행돼도 패키지 import가 동작하도록 보정.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_modular_rag.config import CACHE_DIR, init_runtime

# 가장 먼저 환경변수와 torch 스레드 수를 설정한다.
init_runtime()

import psutil  # noqa: E402

from llama_modular_rag.caching import QueryCache  # noqa: E402
from llama_modular_rag.data_loader import create_vectorstore_from_pdf  # noqa: E402
from llama_modular_rag.graph_builder import build_rag_graph  # noqa: E402

logger = logging.getLogger(__name__)


def visualize_graph(graph: Any, output_path: str = "rag_graph.png") -> bool:
    """컴파일된 LangGraph로부터 실제 노드·엣지를 추출해 이미지로 저장한다."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        from graphviz import Digraph

        runtime_graph = graph.get_graph()

        dot = Digraph(comment="RAG Pipeline")
        terminal = {"__start__", "__end__"}
        for node_id in runtime_graph.nodes:
            shape = "ellipse" if node_id in terminal else "box"
            label = {"__start__": "Start", "__end__": "End"}.get(node_id, node_id)
            dot.node(node_id, label, shape=shape)
        for edge in runtime_graph.edges:
            dot.edge(edge.source, edge.target)

        output_path_base: str = output_path.replace(".png", "")
        dot.render(output_path_base, format="png", cleanup=True)
        logger.info("그래프 시각화가 '%s.png'에 저장되었습니다.", output_path_base)
        return True
    except Exception:
        logger.exception("그래프 시각화 중 오류가 발생했습니다")
        return False


def _memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def run_optimized_rag(pdf_path: str, query: str, visualize: bool = True) -> Dict[str, Any]:
    """CPU에 최적화된 RAG 파이프라인을 실행한다."""
    start_time: float = time.time()
    logger.info("시작 메모리: %.2f MB", _memory_mb())

    os.makedirs(CACHE_DIR, exist_ok=True)
    query_cache = QueryCache()

    logger.info("PDF 처리 중: %s", pdf_path)
    vectorstore, doc_id = create_vectorstore_from_pdf(pdf_path)
    gc.collect()
    logger.info("벡터 저장소 준비 후 메모리: %.2f MB", _memory_mb())

    cached: Optional[Dict[str, Any]] = query_cache.get_cached_result(doc_id, query)
    if cached:
        logger.info("캐시된 결과 사용 (소요 %.2f초)", time.time() - start_time)
        return cached

    logger.info("RAG 그래프 구축")
    rag_graph = build_rag_graph(vectorstore)

    if visualize:
        output_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "visualizations"
        )
        os.makedirs(output_dir, exist_ok=True)
        visualize_graph(rag_graph, os.path.join(output_dir, "rag_graph.png"))

    logger.info("쿼리 처리 시작: %r", query)
    result: Dict[str, Any] = rag_graph.invoke({"query": query})

    query_cache.cache_result(doc_id, query, result)
    logger.info("총 소요 시간: %.2f초, 메모리: %.2f MB", time.time() - start_time, _memory_mb())
    gc.collect()
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    parent_dir: str = os.path.dirname(script_dir)
    pdf_filename: str = "PLAYGROUND_JUNGGU.pdf"
    pdf_path: str = os.path.join(script_dir, pdf_filename)

    if not os.path.exists(pdf_path):
        pdf_path = os.path.join(parent_dir, pdf_filename)

    if not os.path.exists(pdf_path):
        logger.error("PDF 파일을 찾을 수 없습니다: %s", pdf_filename)
        sys.exit(1)

    query: str = "명동에 처음 온 외국인 관광객이 가볼만한 장소를 알려줘?"

    try:
        result = run_optimized_rag(pdf_path, query, visualize=True)
        print("\n" + "=" * 50)
        print("질문:", result["query"])
        print("\n답변:", result.get("answer", "응답을 생성할 수 없습니다."))
        print("=" * 50)

        documents: Optional[List[Any]] = result.get("documents")
        if documents:
            print("\n참조 문서:")
            for i, doc in enumerate(documents[:2]):
                content: str = doc.page_content
                preview: str = content[:200] + "..." if len(content) > 200 else content
                print(f"\n문서 {i + 1}:\n{preview}")
    except Exception:
        logger.exception("처리 중 오류 발생")
        sys.exit(1)
