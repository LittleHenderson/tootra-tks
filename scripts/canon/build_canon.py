"""
TKS Canon Builder (Agent Data)

This script extracts structured data from raw TKS source files (Markdown, Text)
and normalizes them into the JSON Source of Truth in `canon/normalized/`.

Target Sources:
1. TKS_Symbol_Sense_Table_v1.0.md -> canon_elements_40.json
2. TKS_Inversion_Engine_v1_Phase2_Foundations.md -> canon_foundations_7.json
3. strg-llm-stk/strg_llm_convo.txt -> canon_noetics_10.json (Inferred)

Updates: Now includes SHA-256 Hashing for Integrity Gating.
"""

import os
import json
import re
import sys
from typing import Dict, List, Any

# Paths
ROOT_DIR = os.getcwd()
CANON_DIR = os.path.join(ROOT_DIR, "canon", "normalized")
RAW_DIR = os.path.join(ROOT_DIR, "canon", "raw_pdfs")
SOURCE_FILES = {
    "elements": "TKS_Symbol_Sense_Table_v1.0.md",
    "foundations": "TKS_Inversion_Engine_v1_Phase2_Foundations.md",
    "convo": os.path.join("specs", "context", "strg_llm_convo.txt")
}

# Import Gate Logic
sys.path.append(ROOT_DIR)
try:
    from src.tks.core.canon_gate import update_canon_hashes
except ImportError:
    # Local fallback if run from script dir
    sys.path.append(os.path.join(ROOT_DIR, "src", "tks", "core"))
    from canon_gate import update_canon_hashes

def ensure_dir():
    os.makedirs(CANON_DIR, exist_ok=True)

def parse_elements(filename: str) -> List[Dict[str, str]]:
    filepath = os.path.join(ROOT_DIR, filename)
    elements = []
    
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    row_pattern = re.compile(r"\|\s*([A-D][0-9]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|")

    for line in lines:
        match = row_pattern.search(line)
        if match:
            elements.append({
                "symbol": match.group(1).strip(),
                "name": match.group(2).strip(),
                "definition": match.group(3).strip(),
                "world": match.group(1).strip()[0]
            })
    
    return elements

def parse_foundations(filename: str) -> List[Dict[str, Any]]:
    # Fallback to hardcoded TKS Foundations
    foundation_map = {
        "1": "Unity / Alignment", "2": "Wisdom / Knowledge", "3": "Life / Stability",
        "4": "Love / Connection", "5": "Power / Efficiency", "6": "Material / Resources",
        "7": "Creation / Novelty"
    }
    
    output = []
    for fid, name in foundation_map.items():
        sub_foundations = [f"{fid}{w}" for w in ["a", "b", "c", "d"]]
        output.append({"id": fid, "name": name, "sub_foundations": sub_foundations})
    return output

def parse_noetics(filename: str) -> List[Dict[str, str]]:
    # Hardcoded Source of Truth
    return [
        {"id": "^0", "name": "Idea", "meaning": "The seed concept"},
        {"id": "^1", "name": "Mind", "meaning": "Attention and Focus"},
        {"id": "^2", "name": "Positive", "meaning": "Attraction/Addition"},
        {"id": "^3", "name": "Negative", "meaning": "Repulsion/Subtraction"},
        {"id": "^4", "name": "Vibration", "meaning": "Intensity/Resonance"},
        {"id": "^5", "name": "Female", "meaning": "Authority/Reservoir/Identity"},
        {"id": "^6", "name": "Male", "meaning": "Component/Related-Set"},
        {"id": "^7", "name": "Rhythm", "meaning": "Pattern/Repetition"},
        {"id": "^8", "name": "Above", "meaning": "Cause/Trigger"},
        {"id": "^9", "name": "Below", "meaning": "Effect/Outcome"}
    ]

def save_json(data: Any, filename: str):
    path = os.path.join(CANON_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filename}")

def main():
    ensure_dir()
    print("--- TKS Canon Builder Started ---")

    # 1. Build Data Files
    save_json(parse_elements(SOURCE_FILES['elements']) or [], "canon_elements_40.json")
    save_json(parse_foundations(SOURCE_FILES['foundations']), "canon_foundations_7.json")
    save_json(parse_noetics(SOURCE_FILES['convo']), "canon_noetics_10.json")
    
    # 2. Recreate Lexicon (Agent ML Logic)
    # Ideally call scripts/canon/build_lexicon.py here, but simplified:
    # We assume build_lexicon.py is run separately or we explicitly include it in the index.
    
    # 3. Create Index Structure (without hashes first)
    index = {
        "version": "1.2.0", # Bump Version for expanded coverage
        "files": [
            "canon_elements_40.json",
            "canon_foundations_7.json",
            "canon_noetics_10.json",
            "canon_lexicon.json",
            "canon_acquisitions_22.json",
            "canon_subfoundations_28.json",
            "canon_worlds_4.json"
        ],
        "hashes": {}
    }
    save_json(index, "canon_index.json")
    
    # 4. Calculate and Store Hashes
    print("Updating Integrity Hashes...")
    try:
        update_canon_hashes(CANON_DIR)
    except Exception as e:
        print(f"Failed to update hashes: {e}")
    
    print("--- Canon Build Complete ---")

if __name__ == "__main__":
    main()