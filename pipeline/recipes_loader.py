"""Module for loading and preprocessing the recipe data before ingestion."""

import json
import os
from langchain_core.documents import Document
from typing import List, Dict

def _format_data(data: List[Dict]) -> List[Document]:
    """Format recipes data into a clean structure, retaining semantics.

    Args:
        data (List[Dict]): A list of deserialized recipes data.

    Returns:
        List[Document]: A list of Document objects, representing the recipes.
    """
    documents = []
    for record in data:
        # Format page_content cleanly
        page_content = (
            f"Dish: {record['dish_name']}\n"
            f"Origin: {record['origin']}\n\n"
            f"Ingredients:\n- " + "\n- ".join(record["ingredients"]) + "\n\n"
            f"Steps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(record["steps"]))
        )

        # Stable metadata for updates, filters, and reranking
        metadata = {
            "id": f"{record['dish_name'].replace(' ', '_')}_{record['origin']}",
            "dish_name": record["dish_name"],
            "origin": record["origin"],
            "section": "full_recipe"
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def load_recipes(file_path: str = "../data/african_recipes.json") -> List[Document]:
    """Loads the African recipes dataset as a list of Document objects.

    Args:
        file_path (str, optional): Path to recipes data. Defaults to "../data/african_recipes.json".

    Raises:
        FileNotFoundError: Raised when the recipes dataset file does not exist.

    Returns:
        List[Document]: A list of Document objects representing the recipes
    """
    # Load JSON recipes
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recipes = _format_data(data)
    else:
        raise FileNotFoundError("African recipes data file does not exist.")
    
    return recipes
