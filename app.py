import streamlit as st
import json
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from typing import List, Tuple
from dotenv import load_dotenv

from pipeline.recipes_loader import load_recipes
from pipeline.chunking import create_chunks
from pipeline.indexing import build_vector_store
from pipeline.retrieval import create_retriever
from pipeline.chain import create_rag_chain

# Load environmental variables
load_dotenv()

def safe_json_loads(value, default=None):
    """
    Try to json.loads if value is str, else return as-is.
    - Useful for loading JSON strings from metadata.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default if default is not None else value
    return value if value is not None else default

# --- Setup ---
st.set_page_config(page_title="African Culinary RAG", page_icon="🍲", layout="wide")

@st.cache_resource(show_spinner=False)
def initialize_pipeline() -> Tuple[Runnable, ContextualCompressionRetriever, List[Document]]:
    """Load data, build vector store, retriever, and chain (cached)."""
    # 1. Load recipes
    print("Loading recipes...")
    recipes = load_recipes(file_path="data/african_recipes.json")

    # 2. Create chunks
    print("Creating chunks...")
    chunks = create_chunks(recipes)

    # 3. Build or load vector store
    print("Building vector store...")
    vector_store = build_vector_store(documents=chunks, vector_store_path="vectors/chroma_db")

    # 4. Build retriever
    print("Building retriever...")
    retriever = create_retriever(vector_store)

    # 5. Create RAG chain
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(retriever)

    return rag_chain, retriever, recipes

rag_chain, retriever, recipes = initialize_pipeline()

# --- Sidebar: Recipe Explorer ---
st.sidebar.header("📖 Explore Recipes")
selected_dish = st.sidebar.selectbox(
    "Browse by dish", [r.metadata["dish_name"] for r in recipes]
)
if selected_dish:
    recipe_doc = next(r for r in recipes if r.metadata["dish_name"] == selected_dish)
    st.sidebar.subheader(recipe_doc.metadata["dish_name"])
    st.sidebar.write(f"🌍 Origin: {recipe_doc.metadata['origin']}")

# --- Main UI ---
st.title("🍲 African Culinary RAG Assistant")
st.markdown("Ask me about African recipes, ingredients, or cooking steps!")

user_question = st.text_input("🔍 Enter your question:")
if st.button("Ask") and user_question:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(user_question)

    st.markdown("### 📝 Answer")
    st.write(answer)

# --- Show retrieved recipes (context) ---
st.markdown("### 📚 Sources / Retrieved Recipes")
docs = retriever.invoke(user_question)

# Deduplicate by recipe id
unique_ids = {d.metadata["id"] for d in docs}
display_recipes = [r for r in recipes if r.metadata["id"] in unique_ids]

if not docs:
    st.info("No recipes were retrieved for your query. Try rephrasing your question!")
else:
    for i, doc in enumerate(display_recipes, start=1):
        # Use metadata first
        dish_name = doc.metadata.get("dish_name", "Unknown Dish")
        origin = doc.metadata.get("origin", "Unknown Origin")
        notes = doc.metadata.get("notes", None)
        
        # Auto json.loads if stored as string
        ingredients = safe_json_loads(doc.metadata.get("ingredients", []), [])
        steps = safe_json_loads(doc.metadata.get("steps", []), [])
        nutrition = safe_json_loads(doc.metadata.get("nutrition", {}), {})

        with st.expander(f"📖 {i}. {dish_name} ({origin})"):
            st.markdown(f"#### 🌍 Origin: {origin}")

            if ingredients:
                st.markdown("#### 🥘 Ingredients")
                st.markdown("\n".join(f"- {ing}" for ing in ingredients))
            else:
                st.warning("⚠️ Ingredients not available for this recipe.")

            if steps:
                st.markdown("#### 👩‍🍳 Steps")
                st.markdown("\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps)))
            else:
                st.warning("⚠️ Steps not available for this recipe.")

            if notes:
                st.info(f"📝 Notes: {notes}")

            if nutrition:
                st.success(f"🍽 Nutrition: {nutrition}")

            st.code("\n".join(ingredients), language="text")
            st.caption("📋 Copy ingredients above for your shopping list.")

