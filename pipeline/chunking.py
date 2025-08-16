"""Ingestion module for loading the JSON data of recipes and chunking."""
import json
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def create_chunks(data: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Semantically splits the recipes data into chunks.

    Args:
        data (List[Document]): A list of Document objects representing recipes
        chunk_size (int, optional): The largest size of a chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. 
        Defaults to 50.

    Returns:
        List[Document]: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    split_documents = splitter.split_documents(data)
    print(f"Loaded {len(data)} recipes into {len(split_documents)} chunks")

    return split_documents