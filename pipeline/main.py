"""Main file for interacting with the RAG pipeline via the terminal."""

import sys
from dotenv import load_dotenv

from recipes_loader import load_recipes
from chunking import create_chunks
from indexing import build_vector_store
from retrieval import create_retriever
from chain import create_rag_chain

# Load environmental variables from the .env file
load_dotenv()

if __name__ == "__main__":
    print("Loading data...")
    recipes = load_recipes()

    print("Chunking data...")
    chunks = create_chunks(recipes)

    print("Building vector store...")
    vector_store = build_vector_store(chunks)

    print("Building retriever ...")
    retriever = create_retriever(vector_store)

    print("Creating chain...")
    chain = create_rag_chain(retriever)

    while True:
        query = input("\nAsk a recipe question (type 'exit' or 'quit' to end program): ")
        if query.lower() in ["exit", "quit"]:
            sys.exit(0)

        answer = chain.invoke(query)
        print("\nAnswer: ", answer)
