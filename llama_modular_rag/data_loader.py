import os
import hashlib
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embeddings import get_embedding_model
from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH


def get_cache_key(file_path: str) -> str:
    """파일의 내용을 기반으로 캐시 키 생성"""
    with open(file_path, 'rb') as f:
        content: bytes = f.read()
    return hashlib.md5(content).hexdigest()


def create_vectorstore_from_pdf(pdf_path: str) -> Chroma:
    """PDF 파일을 로드하고 벡터 저장소를 생성합니다."""
    # 캐시 키 생성
    cache_key: str = get_cache_key(pdf_path)
    pdf_name: str = os.path.basename(pdf_path).replace('.pdf', '')
    vector_db_directory: str = os.path.join(VECTOR_DB_PATH, f"{pdf_name}_{cache_key}")

    # 이미 벡터DB가 존재하면 로드
    if os.path.exists(vector_db_directory):
        print(f"기존 벡터 저장소 로드 중: {vector_db_directory}")
        embeddings = get_embedding_model()
        return Chroma(
            persist_directory=vector_db_directory,
            embedding_function=embeddings
        )

    # 새 벡터 DB 생성
    print(f"PDF 로딩 및 벡터 저장소 생성 중: {pdf_path}")
    os.makedirs(vector_db_directory, exist_ok=True)

    # PDF 로드
    loader = PyPDFLoader(pdf_path)
    documents: List[Document] = loader.load()

    # 문서 분할 - 작은 청크 사용
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits: List[Document] = text_splitter.split_documents(documents)

    # 벡터 저장소 생성
    embeddings = get_embedding_model()
    vectorstore: Chroma = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vector_db_directory,
        collection_name=f"{pdf_name}_{cache_key}"
    )

    return vectorstore
