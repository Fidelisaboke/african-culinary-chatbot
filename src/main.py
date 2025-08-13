import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Load environmental variables from .env file
load_dotenv()

# Directory for vector stores
VECTOR_STORE_DIR = "../vectors/chroma_db"

# Load the recipes data
loader = JSONLoader(
    file_path="../data/african_recipes.json",
    jq_schema=".[]",
    text_content=False,
)

docs = loader.load()

# Chunk contents
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Generate embeddings and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=VECTOR_STORE_DIR
)

# Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Groq LLM
llm = ChatGroq(model_name="llama3-8b-8192")

# Prompt template
template = """You are a helpful African recipe assistant.
Provide cooking instructions based on the retrieved context.

Context:
{context}

User question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# QA Chain
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

# Sample query
print(chain.invoke("How do I make sukuma wiki?"))

