import streamlit as st
import pandas as pd
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
from pipeline.utils import safe_json_loads, parse_time_to_minutes

# Load environmental variables
load_dotenv()    

# --- Setup ---
st.set_page_config(page_title="African Culinary RAG", page_icon=":stew:", layout="wide")

@st.cache_resource(show_spinner=False)
def initialize_pipeline() -> Tuple[Runnable, ContextualCompressionRetriever, List[Document]]:
    """Load data, build vector store, retriever, and chain (cached)."""

    status_text = st.empty()
    progress_bar = st.progress(0)

    total_steps = 5

    # 1. Load recipes
    print("Loading recipes...")
    status_text.text("Step 1/5: Loading recipes...")
    recipes = load_recipes(file_path="data/african_recipes.json")
    progress_bar.progress(1 / total_steps)

    # 2. Create chunks
    print("Creating chunks...")
    status_text.text("Step 2/5: Creating document chunks...")
    chunks = create_chunks(recipes)
    progress_bar.progress(2 / total_steps)

    # 3. Build or load vector store
    print("Building vector store...")
    status_text.text("Step 3/5: Building vector store...")
    vector_store = build_vector_store(documents=chunks, vector_store_path="vectors/faiss_index")
    progress_bar.progress(3 / total_steps)

    # 4. Build retriever
    print("Creating retriever...")
    status_text.text("Step 4/5: Creating retriever...")
    retriever = create_retriever(vector_store)
    progress_bar.progress(4 / total_steps)

    # 5. Create RAG chain
    print("Initializing RAG chain...")
    status_text.text("Step 5/5: Initializing RAG chain...")
    rag_chain = create_rag_chain(retriever)
    progress_bar.progress(5 / total_steps)

    # Done! Clear progress and status
    status_text.empty()
    progress_bar.empty()

    return rag_chain, retriever, recipes

rag_chain, retriever, recipes = initialize_pipeline()

# --- Sidebar ---
st.sidebar.markdown("## :open_book: Explore Recipes")

# Build filtered list
filtered_recipes = [r for r in recipes]

# Build select box with dish_name + origin preview
dish_options = [
    f"{r.metadata['dish_name']} ({r.metadata['origin']})" for r in filtered_recipes
]

selected_dish_label = st.sidebar.selectbox(
    "Select a recipe",
    ["None"] + dish_options,
    index=0
)

# Show minimal recipe preview in sidebar
if selected_dish_label != "None":
    # Map back to the recipe object
    selected_recipe = next(
        r for r in filtered_recipes
        if f"{r.metadata['dish_name']} ({r.metadata['origin']})" == selected_dish_label
    )

    meta = selected_recipe.metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### :clipboard: Recipe Preview")
    st.sidebar.markdown(f"**Dish:** {meta['dish_name']}")
    st.sidebar.markdown(f"**Origin:** {meta['origin']}")
    if meta.get("total_time"):
        total_time = parse_time_to_minutes(meta['total_time'])
        st.sidebar.markdown(f"**Total time:** {total_time} mins")
    if meta.get("servings"):
        st.sidebar.markdown(f"**Servings:** {meta['servings']}")

# --- Main UI ---
st.markdown(
    """
    <h1 style="text-align:center;">
    🍲 African Culinary Assistant 🍲
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h4 style="text-align:center;">
        Ask me about 
        <span style="color:#FF6347; font-weight:bold;">recipes</span>, 
        <span style="color:#32CD32; font-weight:bold;">ingredients</span>, or 
        <span style="color:#1E90FF; font-weight:bold;">cooking steps</span>!
    </h4>
    """,
    unsafe_allow_html=True
)

user_question = st.text_input(":mag: Enter your question:", placeholder="e.g., How do I make Githeri?")
ask_button = st.button("Ask!", use_container_width=True)

# Trigger retreival when the user clicks "Ask"
if ask_button and user_question:
    with st.spinner("Thinking..."):
        # Generate answer
        answer = rag_chain.invoke(user_question)

        # Retrieve related recipes
        docs = retriever.invoke(user_question)

        # Deduplicate and filter recipes
        unique_ids = {d.metadata["id"] for d in docs}
        display_recipes = [r for r in recipes if r.metadata["id"] in unique_ids]

    # --- Answer Display ---
    st.markdown("### :memo: Answer")
    st.write(answer)

    # --- Display Retrieved Recipes ---
    st.markdown("### :books: Sources / Retrieved Recipes")
    st.markdown("#### These are the sources referenced by the chatbot:")

    if not docs:
        st.info("No recipes were retrieved for your query. Try rephrasing your question!")
    else:
        for i, doc in enumerate(display_recipes, start=1):
            # Metadata
            dish_name = doc.metadata.get("dish_name", "Unknown Dish")
            origin = doc.metadata.get("origin", "Unknown Origin")
            notes = doc.metadata.get("notes", None)
            source = doc.metadata.get("source_url", None)
            image_url = doc.metadata.get("image_url", None)

            # Auto json.loads if stored as string
            ingredients = safe_json_loads(doc.metadata.get("ingredients", []), [])
            steps = safe_json_loads(doc.metadata.get("steps", []), [])
            nutrition = safe_json_loads(doc.metadata.get("nutrition", {}), {})

            with st.expander(f":open_book: {i}. {dish_name} ({origin})"):
                # Top section with optional image
                if image_url:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image_url, use_container_width=True, caption=dish_name)
                    with col2:
                        st.markdown(f"**Origin:** {origin}")
                        if doc.metadata.get("servings"):
                            st.markdown(f"**Servings:** {doc.metadata['servings']}")
                        if doc.metadata.get("total_time"):
                            st.markdown(
                                f"**Total Time:** {parse_time_to_minutes(doc.metadata['total_time'])} minutes"
                            )
                else:
                    st.markdown(f"**Origin:** {origin}")

                # Ingredients
                if ingredients:
                    st.markdown("#### :shallow_pan_of_food: Ingredients")
                    st.markdown("\n".join(f"- {ing}" for ing in ingredients))
                    st.download_button(
                        label="📥 Download Ingredients",
                        data="\n".join(ingredients),
                        file_name=f"{dish_name}_ingredients.txt",
                    )
                else:
                    st.warning(":warning: Ingredients not available.")

                # Steps
                if steps:
                    st.markdown("#### :woman_cook: Steps")
                    st.markdown(
                        "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps))
                    )
                else:
                    st.warning(":warning: Steps not available.")

                # Notes
                if notes:
                    st.info(f":memo: Notes: {notes}")

                # Nutrition (table)
                if nutrition:
                    st.markdown("#### :fork_and_knife: Nutrition Facts")
                    if isinstance(nutrition, dict) and nutrition:
                        df_nutrition = pd.DataFrame(
                            list(nutrition.items()), columns=["Nutrient", "Value"]
                        )
                        st.table(df_nutrition)
                    else:
                        st.write(nutrition)

                # Recipe source
                if source:
                    st.markdown(f"Recipe source: [Source]({doc.metadata['source_url']})")

