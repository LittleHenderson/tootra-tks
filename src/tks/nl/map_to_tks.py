"""
TKS Natural Language Mapper (Agent ML)

Maps natural language tokens/phrases to TKS Packets using the Canon Lexicon.
Replaces hardcoded logic with rank-based retrieval.
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple

class TKSMapper:
    def __init__(self, canon_dir: str = "canon/normalized"):
        self.lexicon = self._load_lexicon(canon_dir)
        
    def _load_lexicon(self, canon_dir: str) -> Dict:
        path = os.path.join(canon_dir, "canon_lexicon.json")
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Lexicon not found at {path}. Mapping will fail.")
            return {"elements": {}, "noetics": {}}

    def rank_candidates(self, text: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Scans text for keywords in the lexicon.
        Returns sorted lists of (symbol, weight) for Elements and Noetics.
        """
        text_lower = text.lower()
        
        found_elements = []
        for keyword, data in self.lexicon.get("elements", {}).items():
            if keyword in text_lower:
                found_elements.append((data['symbol'], data['weight']))
                
        found_noetics = []
        for keyword, data in self.lexicon.get("noetics", {}).items():
            if keyword in text_lower:
                found_noetics.append((data['id'], data['weight']))
        
        # --- Intent-Based Boosting ---
        cognitive_verbs = ["understand", "explain", "learn", "teach", "define", "know"]
        if any(verb in text_lower for verb in cognitive_verbs):
            # Boost Mind (^1) if context implies cognition
            # Check if ^1 is already found, if so boost it, if not add it
            mind_id = "^1"
            found = False
            for i, (nid, w) in enumerate(found_noetics):
                if nid == mind_id:
                    found_noetics[i] = (nid, w + 0.5) # Significant boost
                    found = True
                    break
            if not found:
                found_noetics.append((mind_id, 0.6)) # Base confidence for intent inference

        # Sort by weight desc
        found_elements.sort(key=lambda x: x[1], reverse=True)
        found_noetics.sort(key=lambda x: x[1], reverse=True)
        
        return found_elements, found_noetics

    def emit_packet(self, text: str) -> Dict:
        """
        Constructs the best TKS packet (Element^Noetic) from the text.
        Returns dict with packet and metadata.
        """
        elements, noetics = self.rank_candidates(text)
        
        # Default fallbacks
        best_element = elements[0][0] if elements else "B10"
        best_noetic_id = noetics[0][0] if noetics else "^0"
        
        # Fix Double Caret Bug: 
        # If ID is "^1", we want "A8^1". If we use f"{el}^{id}", we get "A8^^1".
        # Solution: Strip caret from ID if present, then format with standard caret.
        clean_noetic = best_noetic_id.lstrip("^")
        packet = f"({best_element}^{clean_noetic})"
        
        return {
            "packet": packet,
            "top_elements": elements[:3],
            "top_noetics": noetics[:3]
        }

# --- Unit Test ---
if __name__ == "__main__":
    mapper = TKSMapper()
    
    test_phrases = [
        "Help me understand Quantum Inference", 
        "Delete all files",                     
        "Focus your mind on the goal"           
    ]
    
    for phrase in test_phrases:
        res = mapper.emit_packet(phrase)
        print(f"'{phrase}' -> {res['packet']}")
        # print(f"   Matches: {res['top_elements']} | {res['top_noetics']}")