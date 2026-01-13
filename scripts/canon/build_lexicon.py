"""
TKS Canon Lexicon Builder (Agent Data)

Generates the initial 'canon_lexicon.json' from the normalized canon.
This file maps raw keywords/synonyms to TKS Symbols (Elements, Noetics).

Structure:
{
  "elements": {
    "keyword": {"symbol": "A1", "weight": 1.0},
    ...
  },
  "noetics": {
    "keyword": {"id": "^1", "weight": 1.0},
    ...
  }
}
"""

import json
import os
from typing import Dict, Any

CANON_DIR = os.path.join("canon", "normalized")

def load_json(filename):
    with open(os.path.join(CANON_DIR, filename), "r") as f:
        return json.load(f)

def build_lexicon():
    elements = load_json("canon_elements_40.json")
    noetics = load_json("canon_noetics_10.json")
    
    lexicon = {
        "elements": {},
        "noetics": {}
    }
    
    # 1. Build Element Lexicon
    # Seed with Name and Definition keywords
    for el in elements:
        symbol = el['symbol']
        name = el['name'].lower()
        
        # Add exact name
        lexicon["elements"][name] = {"symbol": symbol, "weight": 1.0}
        
        # Add world-specific keywords (Naive heuristic for now)
        if symbol.startswith("A"):
            lexicon["elements"]["spiritual " + name] = {"symbol": symbol, "weight": 0.9}
        elif symbol.startswith("B"):
            lexicon["elements"]["mental " + name] = {"symbol": symbol, "weight": 0.9}
        
        # Specific overrides based on known mappings
        if "cause" in name:
            lexicon["elements"]["trigger"] = {"symbol": symbol, "weight": 0.8}
            lexicon["elements"]["source"] = {"symbol": symbol, "weight": 0.8}
            
    # 2. Build Noetic Lexicon
    for n in noetics:
        nid = n['id']
        name = n['name'].lower()
        
        lexicon["noetics"][name] = {"id": nid, "weight": 1.0}
        
        # Synonyms based on definitions
        if name == "mind":
            lexicon["noetics"]["attention"] = {"id": nid, "weight": 0.9}
            lexicon["noetics"]["focus"] = {"id": nid, "weight": 0.9}
        elif name == "above":
            lexicon["noetics"]["cause"] = {"id": nid, "weight": 0.9}
            lexicon["noetics"]["prior"] = {"id": nid, "weight": 0.8}
        elif name == "below":
            lexicon["noetics"]["effect"] = {"id": nid, "weight": 0.9}
            lexicon["noetics"]["outcome"] = {"id": nid, "weight": 0.8}

    # Save Lexicon
    out_path = os.path.join(CANON_DIR, "canon_lexicon.json")
    with open(out_path, "w") as f:
        json.dump(lexicon, f, indent=2)
        
    print(f"Built Lexicon with {len(lexicon['elements'])} element terms and {len(lexicon['noetics'])} noetic terms.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    build_lexicon()
