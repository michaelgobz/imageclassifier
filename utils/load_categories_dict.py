#!/usr/bin/env python3

"""File contains functions to load the flower categories from a json file
   author: Michael Goboola
    date: 2023-19-12
    time 20:00
"""

import json


def load_categories(filepath: str):
    """function to load the flower categories from a json file
    Args:
        filepath: path to the json file

    Returns: dict of categories

    """
    with open(filepath, "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)
    return cat_to_name
