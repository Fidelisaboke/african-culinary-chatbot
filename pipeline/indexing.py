"""Indexing module for handling embedding and vector store persistence."""
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

# Vector store directory path for persistence
VECTOR_STORE_DIR = "../vectors/faiss_index"

def build_vector_store(documents: List[Document], vector_store_path: str = VECTOR_STORE_DIR) -> FAISS:
    """Builds a vector store from a list of documents (chunks)

    Args:
        documents (list[Document]): List of Documents
        vector_store_path (str): The Vector store path

    Returns:
        FAISS: A FAISS vector store instance
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build Vector store
    vector_store = FAISS.from_documents(
        documents, 
        embeddings,
    )
    
    # Persist index to a file
    vector_store.save_local(vector_store_path)

    return vector_store