"""Indexing module for handling embedding and vector store persistence."""
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List

# Vector store directory path for persistence
VECTOR_STORE_DIR = "../vectors/chroma_db"

def build_vector_store(documents: List[Document]) -> Chroma:
    """Builds a vector store from a list of documents (chunks)

    Args:
        documents (list[Document]): List of Documents

    Returns:
        Chroma: A vector store instance (Chroma)
    """

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if a vector store exists
    if os.path.exists(VECTOR_STORE_DIR):
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
    else:    
        vector_store = Chroma.from_documents(
            documents, 
            embeddings,
            persist_directory=VECTOR_STORE_DIR
        )

    return vector_store