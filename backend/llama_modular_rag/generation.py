from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llama_modular_rag.llm_setup import setup_llama_model
from llama_modular_rag.state import RAGState

ANSWER_PROMPT_TEXT = """다음 정보를 기반으로 질문에 간결하게 답변해주세요.

컨텍스트:
{context}

질문: {query}

답변:"""

_ANSWER_PROMPT = PromptTemplate.from_template(ANSWER_PROMPT_TEXT)


def answer_generator(state: RAGState) -> RAGState:
    """컨텍스트와 쿼리를 사용해 답변을 생성한다."""
    chain = _ANSWER_PROMPT | setup_llama_model() | StrOutputParser()
    answer: str = chain.invoke(
        {"context": state.get("context", ""), "query": state["query"]}
    )
    return {**state, "answer": answer}
