#!/usr/bin/env python3
    
"""_summary_
"""

import json
    
def load_categories(path:str):
    """_summary_

    Args:
        path (str): _description_
    """
    with open(path, 'r', encoding='utf-8') as f:
        cat_to_name = json.load(f)
    return cat_to_name
