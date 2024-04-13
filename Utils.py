# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:43:33 2024

@author: aliab
"""
from Transformer import Transformer
import Dataset as ds
from Worker import Worker

import json

def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename):
    with open(filename, 'r') as f:
        # Read the entire contents of the file
        content = f.read()
        # Replace single quotes with double quotes
        content = content.replace("'", '"')
        # Load the JSON data
        config = json.loads(content)
    return config


