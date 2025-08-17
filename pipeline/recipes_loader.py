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
        # Build structured page_content dynamically
        sections = [
            f"Dish: {record['dish_name']}",
            f"Origin: {record['origin']}"
        ]

        # Add other sections
        if record.get("ingredients"):
            sections.append("Ingredients:\n- " + "\n- ".join(record["ingredients"]))
        if record.get("steps"):
            sections.append("Steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(record["steps"])))
        if record.get("notes"):
            sections.append("Notes:\n" + record["notes"])
        if record.get("nutrition"):
            sections.append("Nutrition:\n" + "\n".join(f"{k}: {v}" for k, v in record["nutrition"].items()))
        if record.get("source_url"):
            sections.append(f"Source: {record['source_url']}")

        page_content = "\n\n".join(sections)

        # Stable metadata for updates, filters, reranking, and Streamlit UI
        metadata = {
            "id": f"{record['dish_name'].replace(' ', '_')}_{record['origin']}",
            "dish_name": record["dish_name"],
            "origin": record["origin"],
            "prep_time": record.get("prep_time"),
            "cook_time": record.get("cook_time"),
            "total_time": record.get("total_time"),
            "servings": record.get("servings"),
            "ingredients": json.dumps(record.get("ingredients", [])),
            "steps": json.dumps(record.get("steps", [])),
            "notes": record.get("notes"),
            "nutrition": json.dumps(record.get("nutrition", {})),
            "source_url": record.get("source_url"),
            "image_url": record.get("image_url"),
            "section": "full_recipe",
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
