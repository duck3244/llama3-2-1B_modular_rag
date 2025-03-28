import os
import time
import gc
import psutil
from data_loader import create_vectorstore_from_pdf
from graph_builder import build_rag_graph
from caching import QueryCache
from config import CACHE_DIR
import matplotlib.pyplot as plt


# 그래프 시각화 관련 코드 추가
def visualize_graph(graph, output_path="rag_graph.png"):
    """
    RAG 그래프 구조를 시각화하여 이미지 파일로 저장
    """
    try:
        # 폴더 생성
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 직접 그래프 내용 작성
        from graphviz import Digraph

        dot = Digraph(comment='RAG Pipeline')

        # 노드 추가
        dot.node('start', 'Start', shape='ellipse')
        dot.node('doc_retriever', '문서 검색', shape='box')
        dot.node('context_builder', '컨텍스트 생성', shape='box')
        dot.node('answer_generator', '답변 생성', shape='box')
        dot.node('end', 'End', shape='ellipse')

        # 엣지 추가
        dot.edge('start', 'doc_retriever')
        dot.edge('doc_retriever', 'context_builder')
        dot.edge('context_builder', 'answer_generator')
        dot.edge('answer_generator', 'end')

        # 이미지 저장
        output_path_base = output_path.replace(".png", "")
        dot.render(output_path_base, format="png", cleanup=True)

        print(f"그래프 시각화가 '{output_path_base}.png'에 저장되었습니다.")

        return True
    except Exception as e:
        print(f"그래프 시각화 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_memory_usage():
    """현재 메모리 사용량 출력"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")


def run_optimized_rag(pdf_path: str, query: str, visualize=True):
    """CPU에 최적화된 RAG 시스템 실행"""
    start_time = time.time()
    print("시작 시점 메모리 사용량:")
    print_memory_usage()

    # 캐싱 메커니즘 초기화
    os.makedirs(CACHE_DIR, exist_ok=True)
    query_cache = QueryCache()

    # 이전 결과 캐시 확인
    cached_result = query_cache.get_cached_result(query)
    if cached_result:
        print("캐시된 결과 사용 중...")
        elapsed_time = time.time() - start_time
        print(f"\n처리 소요 시간 (캐시): {elapsed_time:.2f}초")
        print_memory_usage()
        return cached_result

    # 벡터 저장소 생성
    print(f"PDF 처리 중: {pdf_path}")
    vectorstore = create_vectorstore_from_pdf(pdf_path)

    # 중간 메모리 정리
    gc.collect()
    print("벡터 저장소 생성 후 메모리 사용량:")
    print_memory_usage()

    # RAG 그래프 구축
    print("RAG 그래프 구축 중...")
    rag_graph = build_rag_graph(vectorstore)

    # 그래프 시각화 (선택 사항)
    if visualize:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rag_graph.png")
        visualize_graph(rag_graph, output_path)

    # 초기 상태 설정
    initial_state = {"query": query}

    # 그래프 실행
    print(f"쿼리 처리 시작: '{query}'")
    result = rag_graph.invoke(initial_state)

    # 결과 캐싱
    print("결과 캐싱 중...")
    query_cache.cache_result(query, result)

    # 소요 시간 계산
    elapsed_time = time.time() - start_time
    print(f"\n총 처리 소요 시간: {elapsed_time:.2f}초")
    print("최종 메모리 사용량:")
    print_memory_usage()

    # 메모리 정리
    gc.collect()

    return result


if __name__ == "__main__":
    # 환경 변수 설정 - 필요한 경우 여기 추가

    # 처리할 PDF 파일 경로
    pdf_path = "PLAYGROUND_JUNGGU.pdf"
    if not os.path.exists(pdf_path):
        print(f"오류: 파일이 존재하지 않습니다: {pdf_path}")
        exit(1)

    # 사용자 쿼리 입력
    query = "명동에 처음 온 외국인 관광객이 가볼만한 장소를 알려줘?"
    if not query.strip():
        print("오류: 질문이 비어있습니다.")
        exit(1)

    try:
        # RAG 파이프라인 실행 (그래프 시각화 활성화)
        result = run_optimized_rag(pdf_path, query, visualize=True)

        # 결과 표시
        print("\n" + "=" * 50)
        print("질문:", result["query"])
        print("\n답변:", result.get("answer", "응답을 생성할 수 없습니다."))
        print("=" * 50)

        # 참조 문서 표시 (선택 사항)
        if "documents" in result and result["documents"]:
            print("\n참조 문서:")
            for i, doc in enumerate(result["documents"][:2]):
                print(f"\n문서 {i + 1} 일부:")
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

    except Exception as e:
        print(f"\n처리 중 오류가 발생했습니다: {str(e)}")
        import traceback

        traceback.print_exc()