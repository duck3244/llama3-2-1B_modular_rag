from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from state import RAGState
from llm_setup import setup_llama_model


def answer_generator(state: RAGState) -> RAGState:
    """컨텍스트와 쿼리를 사용하여 간소화된 답변 생성"""
    llm = setup_llama_model()

    # 더 간결한 프롬프트 템플릿
    answer_prompt = PromptTemplate.from_template(
        """다음 정보를 기반으로 질문에 간결하게 답변해주세요.

        컨텍스트:
        {context}

        질문: {query}

        답변:"""
    )

    answer_chain = answer_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "context": state.get("context", ""),
        "query": state["query"]
    })

    return {**state, "answer": answer}