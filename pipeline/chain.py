"""This module builds the Retrieval QA Chain."""
import json
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .utils import parse_time_to_minutes, get_groq_api_key

def _format_recipe_metadata(doc):
    """Convert a Document's metadata into readable text for the LLM."""
    meta = doc.metadata
    
    # Deserialize JSON fields
    ingredients = json.loads(meta.get("ingredients", "[]"))
    steps = json.loads(meta.get("steps", "[]"))
    nutrition = json.loads(meta.get("nutrition", "{}"))
    
    # Format ingredients and steps nicely
    ingredients_text = "\n".join(f"- {ing}" for ing in ingredients) if ingredients else "N/A"
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps)) if steps else "N/A"

    # Convert ISO 8601 time strings to human-readable formats
    prep_minutes = parse_time_to_minutes(meta.get("prep_time"))
    cook_minutes = parse_time_to_minutes(meta.get("cook_time"))
    total_minutes = parse_time_to_minutes(meta.get("total_time"))

    
    nutrition_text = ""
    if nutrition:
        nutrition_text = "\n".join(f"{k}: {v}" for k, v in nutrition.items())
    
    # Construct readable string
    return (
        "Dish: " + meta.get('dish_name', 'Unknown') + "\n"
        "Origin: " + meta.get('origin', 'Unknown') + "\n"
        "Servings: " + str(meta.get('servings', 'Unknown')) + "\n\n"
        "Prep Time: " + (f"{prep_minutes} minutes" if prep_minutes is not None else "N/A") + "\n"
        "Cook Time: " + (f"{cook_minutes} minutes" if cook_minutes is not None else "N/A") + "\n"
        "Total Time: " + (f"{total_minutes} minutes" if total_minutes is not None else "N/A") + "\n\n"
        "Ingredients:\n" + ingredients_text + "\n\n"
        "Steps:\n" + steps_text + "\n\n"
        "Notes: " + str(meta.get('notes', 'N/A')) + "\n\n"
        "Nutrition:\n" + nutrition_text + "\n\n"
        "Source URL: " + str(meta.get('source_url', 'N/A'))
    )

def create_rag_chain(retriever: ContextualCompressionRetriever) -> Runnable:
    """Create the RAG chain.
    Args:
        retriever (ContextualCompressionRetriever): The retriever object.

    Returns:
        (Runnable): Runnable object representing the RAG chain.
    """
    api_key = get_groq_api_key()
    llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)

    prompt_template = """You are a helpful African recipe assistant.
    Use the context below to answer the user's question on recipes.
    If the answer isn't found, say so.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Wrap formatting function
    format_runnable = RunnableLambda(_format_recipe_metadata)

    # Construct the chain using LCEL
    chain = (
        {
            "context": retriever | format_runnable.map(), 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain