from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from state import RAGState
from llm_setup import setup_llama_model


def needs_more_context(state: RAGState) -> str:
    """현재 응답이 충분한지, 아니면 더 많은 컨텍스트가 필요한지 결정합니다."""
    llm = setup_llama_model()

    decision_prompt = PromptTemplate.from_template(
        """질문: {query}
        현재 답변: {answer}

        이 답변이 질문에 충분히 대답했습니까? '네' 또는 '아니오'로만 대답하세요."""
    )

    decision_chain = decision_prompt | llm | StrOutputParser()
    decision = decision_chain.invoke({
        "query": state["query"],
        "answer": state.get("refined_answer", "")
    }).strip().lower()

    if "아니" in decision or "no" in decision:
        return "needs_more_context"
    else:
        return "sufficient"