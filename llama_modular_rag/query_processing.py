from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from state import RAGState
from llm_setup import setup_llama_model


def query_rewriter(state: RAGState) -> RAGState:
    """원래 쿼리를 더 효과적인 검색 쿼리로 재작성합니다."""
    llm = setup_llama_model()

    rewrite_prompt = PromptTemplate.from_template(
        """당신은 검색 쿼리 최적화 전문가입니다. 
        사용자의 원래 질문을 검색 엔진에서 더 관련성 높은 결과를 얻을 수 있도록 재작성해주세요.

        원래 질문: {query}

        재작성된 질문:"""
    )

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten_query = rewrite_chain.invoke({"query": state["query"]})

    return {**state, "rewritten_query": rewritten_query}