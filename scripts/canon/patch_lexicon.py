"""
Patch Lexicon (Agent Data)

Adds missing terms (Quantum, Delete, etc.) to the Canon Lexicon.
"""

import json
import os

CANON_DIR = os.path.join("canon", "normalized")
LEXICON_PATH = os.path.join(CANON_DIR, "canon_lexicon.json")

def patch_lexicon():
    with open(LEXICON_PATH, "r") as f:
        lexicon = json.load(f)
        
    # Add Quantum -> A8 (Spiritual Cause)
    lexicon["elements"]["quantum"] = {"symbol": "A8", "weight": 0.85}
    lexicon["elements"]["inference"] = {"symbol": "B6", "weight": 0.8} # Logic
    
    # Add Delete -> D3 (Physical Illness/Negation)
    lexicon["elements"]["delete"] = {"symbol": "D3", "weight": 0.9}
    lexicon["elements"]["remove"] = {"symbol": "D3", "weight": 0.85}
    
    # Add Critical -> Noetic Negative or High Stakes logic (Usually handled by Governance, but symbol mapping helps)
    lexicon["noetics"]["critical"] = {"id": "^3", "weight": 0.7} # Negative context
    
    with open(LEXICON_PATH, "w") as f:
        json.dump(lexicon, f, indent=2)
        
    print(f"Patched Lexicon at {LEXICON_PATH}")

if __name__ == "__main__":
    patch_lexicon()
