import hashlib
import logging
import os
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from llama_modular_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, VECTOR_DB_PATH
from llama_modular_rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def compute_doc_id(file_path: str) -> str:
    """PDF 내용 기반 안정적인 문서 ID."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _collection_name(doc_id: str) -> str:
    """Chroma 컬렉션 이름 제약(영숫자/3-63자)을 만족하는 안전한 이름."""
    return f"doc_{doc_id[:32]}"


def create_vectorstore_from_pdf(pdf_path: str) -> Tuple[Chroma, str]:
    """PDF 파일을 로드하고 (벡터 저장소, doc_id)를 반환한다."""
    doc_id = compute_doc_id(pdf_path)
    persist_dir = os.path.join(VECTOR_DB_PATH, doc_id)
    embeddings = get_embedding_model()
    collection = _collection_name(doc_id)

    if os.path.exists(persist_dir):
        logger.info("기존 벡터 저장소 로드: %s", persist_dir)
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection,
        )
        return vectorstore, doc_id

    logger.info("PDF 로딩 및 벡터 저장소 생성: %s", pdf_path)
    os.makedirs(persist_dir, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    documents: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits: List[Document] = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )
    return vectorstore, doc_id
