import streamlit as st
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
    vector_store = build_vector_store(chunks)

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
docs = retriever.get_relevant_documents(user_question)

if not docs:
    st.info("No recipes were retrieved for your query. Try rephrasing your question!")
else:
    for i, doc in enumerate(docs, start=1):
        dish_name = doc.metadata.get("dish_name", "Unknown Dish")
        origin = doc.metadata.get("origin", "Unknown Origin")
        notes = doc.metadata.get("notes", None)
        nutrition = doc.metadata.get("nutrition", None)

        with st.expander(f"📖 {i}. {dish_name} ({origin})"):
            st.markdown(f"#### 🌍 Origin: {origin}")

            # Extract recipe sections safely
            content = doc.page_content
            ingredients = []
            steps = []

            if "Ingredients:" in content:
                try:
                    ingredients_text = content.split("Ingredients:")[1].split("Steps:")[0]
                    ingredients = [line.strip() for line in ingredients_text.strip().split("\n") if line.strip()]
                except Exception:
                    pass

            if "Steps:" in content:
                try:
                    steps_text = content.split("Steps:")[1]
                    steps = [line.strip() for line in steps_text.strip().split("\n") if line.strip()]
                except Exception:
                    pass

            # Display ingredients
            if ingredients:
                st.markdown("#### 🥘 Ingredients")
                st.markdown("\n".join(f"- {ing}" for ing in ingredients))
            else:
                st.warning("⚠️ Ingredients not available for this recipe.")

            # Display steps
            if steps:
                st.markdown("#### 👩‍🍳 Steps")
                st.markdown("\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps)))
            else:
                st.warning("⚠️ Steps not available for this recipe.")

            # Optional metadata
            if notes:
                st.info(f"📝 Notes: {notes}")

            if nutrition:
                st.success(f"🍽 Nutrition: {nutrition}")

            # Copy button (Streamlit UI trick)
            st.code("\n".join(ingredients), language="text")
            st.caption("📋 Copy ingredients above for your shopping list.")
