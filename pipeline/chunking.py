"""Ingestion module for loading the JSON data of recipes and chunking."""
import json
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def create_chunks(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 50) -> List[Document]:
    """Semantically splits the recipes data into chunks.

    Args:
        data (List[Document]): A list of Document objects representing recipes
        chunk_size (int, optional): The largest size of a chunk. Defaults to 800.
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

    all_chunks = []
    for doc in docs:
        # Split the page_content text, while preserving the same metadata
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_id": f"{doc.metadata['id']}_{i}"}
            )
            # print(chunk_doc)
            all_chunks.append(chunk_doc)

    print(f"Loaded {len(docs)} recipes into {len(all_chunks)} chunks")
    return all_chunks