import json
import re


"""Utility functions used in other modules and the Streamlit app module."""
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