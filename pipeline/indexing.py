"""Indexing module for handling embedding and vector store persistence."""
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List

# Vector store directory path for persistence
VECTOR_STORE_DIR = "../vectors/chroma_db"

def build_vector_store(documents: List[Document], vector_store_path: str =VECTOR_STORE_DIR) -> Chroma:
    """Builds a vector store from a list of documents (chunks)

    Args:
        documents (list[Document]): List of Documents
        vector_store_path (str): The Vector store path

    Returns:
        Chroma: A vector store instance (Chroma)
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build Vector store
    vector_store = Chroma.from_documents(
        documents, 
        embeddings,
        persist_directory=vector_store_path
    )

    return vector_store