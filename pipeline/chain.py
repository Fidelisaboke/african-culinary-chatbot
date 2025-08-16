"""This module builds the Retrieval QA Chain."""
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

def create_rag_chain(retriever: ContextualCompressionRetriever) -> Runnable:
    llm = ChatGroq(model="llama3-8b-8192")

    prompt_template = """You are a helpful African recipe assistant.
    Use the context below to answer the user's question on recipes.
    If the answer isn't found, say so.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain