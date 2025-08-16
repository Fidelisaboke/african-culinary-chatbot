"""Retriever module with reranking."""
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def create_retriever(vector_store: Chroma, fetch_count: int = 6, reranker_top_n: int = 3) -> ContextualCompressionRetriever:
    """Builds a retriever with reranking.

    Args:
        vector_store (Chroma): The Chroma vector store object containing the embeddings.
        fetch_count (int, optional): Number of docs to fetch from the vector store. Defaults to 6.
        reranker_top_n (int, optional): Number of docs to return after reranking. Defaults to 3.

    Returns:
        ContextualCompressionRetriever: The new retriever after reranking.
    """
    # Stage 1 retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": fetch_count})

    # Reranking
    model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    compressor = CrossEncoderReranker(model=model, top_n=reranker_top_n)

    # Stage 2 retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    return compression_retriever
