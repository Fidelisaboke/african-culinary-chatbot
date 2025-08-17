"""Utility functions used in the Streamlit app module."""

import json
import pandas as pd
import re
import streamlit as st

def safe_json_loads(value, default=None):
    """Try to json.loads if value is str, else return as-is.
    Useful for loading JSON strings from metadata.

    Args:
        value (_type_): _description_
        default (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default if default is not None else value
    return value if value is not None else default


def parse_time_to_minutes(time_str: str):
    """Convert ISO 8601 duration string like 'PT1H30M' to total minutes.
    Returns None if invalid or empty.

    Args:
        time_str (str): The time string to convert to minutes.

    Returns:
        (int or None): An integer representing the time in minutes, or None.
    """
    if not time_str:
        return None
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", time_str)
    if not match:
        return None
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes


def display_recipe_details(doc):
    """Display recipe details in a Streamlit expander."""
    dish_name = doc.metadata.get("dish_name", "Unknown Dish")
    origin = doc.metadata.get("origin", "Unknown Origin")
    notes = doc.metadata.get("notes", None)
    source = doc.metadata.get("source_url", None)
    image_url = doc.metadata.get("image_url", None)

    ingredients = safe_json_loads(doc.metadata.get("ingredients", []), [])
    steps = safe_json_loads(doc.metadata.get("steps", []), [])
    nutrition = safe_json_loads(doc.metadata.get("nutrition", {}), {})

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
                st.markdown(f"**Total Time:** {parse_time_to_minutes(doc.metadata['total_time'])} minutes")
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
        st.markdown("\n".join(f"{idx+1}. {step}" for idx, step in enumerate(steps)))
    else:
        st.warning(":warning: Steps not available.")

    # Notes
    if notes:
        st.info(f":memo: Notes: {notes}")

    # Nutrition
    if nutrition:
        st.markdown("#### :fork_and_knife: Nutrition Facts")
        if isinstance(nutrition, dict) and nutrition:
            df_nutrition = pd.DataFrame(list(nutrition.items()), columns=["Nutrient", "Value"])
            st.table(df_nutrition)
        else:
            st.write(nutrition)

    # Recipe source
    if source:
        st.markdown(f"Recipe source: [Source]({doc.metadata['source_url']})")
