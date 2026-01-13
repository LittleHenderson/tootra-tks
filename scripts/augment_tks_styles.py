"""
TKS Multi-Style Augmentation System

Takes coherent TKS material and generates 8 stylistic variations:
1. 5th Grade - Simple words, short sentences
2. 10th Grade - Standard high school level
3. Collegiate - Academic but accessible
4. Bachelor's - Undergraduate academic style
5. Master's - Graduate-level discourse
6. PhD - Dense academic prose
7. AAVE/Street - African American Vernacular English
8. Upper Class - Formal, refined language

This creates training data for the coherence model to learn that
coherent TKS content can be expressed in many different styles.

Author: TKS Research Pipeline
Date: 2026-01-13
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import argparse

# Style transformation templates and rules
STYLE_CONFIGS = {
    "5th_grade": {
        "description": "Simple words, short sentences, everyday examples",
        "sentence_length": "short",
        "vocabulary": "simple",
        "connectors": ["and", "but", "so", "then", "because"],
        "openers": [
            "Think of it like this:",
            "It's kind of like when",
            "This means that",
            "Basically,",
            "Simply put,",
        ],
        "replacements": {
            "consciousness": "awareness",
            "manifestation": "showing up",
            "transcendent": "amazing",
            "spiritual": "inner",
            "dimension": "level",
            "resonance": "connection",
            "amplifies": "makes stronger",
            "vibrational": "energy",
            "noetic": "mind",
            "realm": "world",
            "principle": "idea",
            "harmonious": "peaceful",
            "equilibrium": "balance",
            "transmutation": "change",
        }
    },
    "10th_grade": {
        "description": "Standard high school reading level",
        "sentence_length": "medium",
        "vocabulary": "standard",
        "connectors": ["however", "therefore", "additionally", "furthermore", "because of this"],
        "openers": [
            "In this context,",
            "This concept shows that",
            "We can understand this as",
            "The key idea is that",
            "Looking at this,",
        ],
        "replacements": {
            "manifestation": "expression",
            "transcendent": "beyond ordinary",
            "noetic": "mental",
        }
    },
    "collegiate": {
        "description": "College-level academic writing",
        "sentence_length": "medium-long",
        "vocabulary": "academic",
        "connectors": ["consequently", "moreover", "in contrast", "similarly", "as a result"],
        "openers": [
            "From an analytical perspective,",
            "This framework suggests that",
            "The underlying principle indicates",
            "Examining this relationship,",
            "In theoretical terms,",
        ],
        "replacements": {}
    },
    "bachelors": {
        "description": "Undergraduate academic discourse",
        "sentence_length": "long",
        "vocabulary": "academic-technical",
        "connectors": ["thus", "hence", "accordingly", "whereby", "insofar as"],
        "openers": [
            "Within this theoretical framework,",
            "The conceptual underpinnings reveal",
            "An analysis of this construct indicates",
            "The operational dynamics suggest",
            "Methodologically speaking,",
        ],
        "replacements": {}
    },
    "masters": {
        "description": "Graduate-level scholarly prose",
        "sentence_length": "complex",
        "vocabulary": "specialized",
        "connectors": ["notwithstanding", "vis-a-vis", "qua", "ipso facto", "mutatis mutandis"],
        "openers": [
            "The epistemological implications of this framework",
            "Synthesizing the theoretical foundations,",
            "The dialectical relationship between",
            "From a meta-analytical standpoint,",
            "The hermeneutical interpretation suggests",
        ],
        "replacements": {
            "shows": "evinces",
            "creates": "engenders",
            "basic": "foundational",
            "important": "salient",
        }
    },
    "phd": {
        "description": "Dense doctoral-level academic prose",
        "sentence_length": "very_complex",
        "vocabulary": "highly_specialized",
        "connectors": ["ergo", "inter alia", "a fortiori", "sui generis", "pace"],
        "openers": [
            "The ontological substrate of this phenomenological construct",
            "Deconstructing the epistemological matrix,",
            "The axiological dimensions inherent in",
            "Interrogating the teleological implications,",
            "The isomorphic correspondence between",
        ],
        "replacements": {
            "idea": "conceptualization",
            "change": "transmutation",
            "connection": "interrelationship",
            "pattern": "morphological structure",
            "energy": "dynamical substrate",
        }
    },
    "aave_street": {
        "description": "African American Vernacular English - authentic, expressive",
        "sentence_length": "varied",
        "vocabulary": "vernacular",
        "connectors": ["so like", "and then", "but check it", "real talk", "feel me"],
        "openers": [
            "Yo, peep this -",
            "Aight so basically,",
            "Check it out,",
            "Real talk though,",
            "So boom,",
            "Let me put you on -",
            "Here's the deal,",
        ],
        "replacements": {
            "understand": "feel",
            "realize": "peep",
            "very": "hella",
            "good": "fire",
            "strong": "solid",
            "important": "major",
            "creating": "putting together",
            "represents": "stands for",
            "combined with": "mixed with",
        },
        "suffixes": [
            ", you feel me?",
            ", that's facts.",
            ", no cap.",
            ", straight up.",
            ", on God.",
            ".",
        ]
    },
    "upper_class": {
        "description": "Formal, refined, aristocratic language",
        "sentence_length": "elaborate",
        "vocabulary": "refined",
        "connectors": ["indeed", "furthermore", "moreover", "quite so", "naturally"],
        "openers": [
            "One might observe that",
            "It is rather evident that",
            "The discerning mind will note",
            "In matters of this nature,",
            "As one considers this proposition,",
            "The cultivated perspective reveals",
        ],
        "replacements": {
            "good": "exemplary",
            "bad": "regrettable",
            "big": "considerable",
            "important": "of paramount significance",
            "shows": "demonstrates with clarity",
            "makes": "renders",
            "gets": "acquires",
        },
        "suffixes": [
            ", as one would expect.",
            ", quite naturally.",
            ", I dare say.",
            ".",
        ]
    }
}


def transform_to_style(text: str, style: str) -> str:
    """Transform text to a specific style."""
    config = STYLE_CONFIGS[style]
    result = text

    # Apply word replacements
    for old, new in config.get("replacements", {}).items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        result = pattern.sub(new, result)

    # Add style-specific opener (sometimes)
    if random.random() < 0.4 and config.get("openers"):
        opener = random.choice(config["openers"])
        # Don't double-add openers
        if not any(result.lower().startswith(o.lower()[:10]) for o in config["openers"]):
            result = f"{opener} {result[0].lower()}{result[1:]}"

    # Add style-specific suffix for certain styles
    if config.get("suffixes") and random.random() < 0.3:
        # Remove existing period if present
        result = result.rstrip(".")
        result += random.choice(config["suffixes"])

    # Style-specific transformations
    if style == "5th_grade":
        # Break long sentences
        sentences = result.split(". ")
        shortened = []
        for sent in sentences:
            if len(sent) > 80:
                # Try to break at connectors
                for conn in [", which", ", that", ", and", ", but"]:
                    if conn in sent:
                        parts = sent.split(conn, 1)
                        shortened.append(parts[0] + ".")
                        shortened.append(parts[1].strip().capitalize())
                        break
                else:
                    shortened.append(sent)
            else:
                shortened.append(sent)
        result = " ".join(shortened)

    elif style == "phd":
        # Add complexity markers
        complexity_insertions = [
            "—a notion central to our inquiry—",
            ", properly understood,",
            ", in the fullest sense,",
            " (sensu stricto)",
        ]
        if random.random() < 0.3:
            words = result.split()
            if len(words) > 10:
                insert_pos = random.randint(5, len(words) - 5)
                words.insert(insert_pos, random.choice(complexity_insertions))
                result = " ".join(words)

    elif style == "aave_street":
        # Make more conversational
        result = result.replace("one's", "your")
        result = result.replace("One's", "Your")
        result = result.replace("oneself", "yourself")

    return result


def augment_entry(entry: Dict, styles: List[str] = None) -> List[Dict]:
    """Augment a single entry with multiple styles."""
    if styles is None:
        styles = list(STYLE_CONFIGS.keys())

    augmented = []

    # Get the text to transform
    if "story" in entry:
        original_text = entry["story"]
        text_field = "story"
    elif "interpretation" in entry:
        original_text = entry["interpretation"]
        text_field = "interpretation"
    else:
        return []

    # Original entry
    original = entry.copy()
    original["style"] = "original"
    original["is_coherent"] = True
    augmented.append(original)

    # Generate each style variation
    for style in styles:
        styled_entry = entry.copy()
        styled_entry[text_field] = transform_to_style(original_text, style)
        styled_entry["style"] = style
        styled_entry["is_coherent"] = True
        styled_entry["original_text"] = original_text
        augmented.append(styled_entry)

    return augmented


def create_incoherent_examples(entry: Dict, num_examples: int = 2) -> List[Dict]:
    """Create incoherent/gibberish examples for contrast training."""
    incoherent = []

    if "story" in entry:
        text = entry["story"]
        text_field = "story"
    elif "interpretation" in entry:
        text = entry["interpretation"]
        text_field = "interpretation"
    else:
        return []

    # Method 1: Word salad (shuffle words)
    words = text.split()
    random.shuffle(words)
    gibberish1 = entry.copy()
    gibberish1[text_field] = " ".join(words)
    gibberish1["style"] = "gibberish_shuffle"
    gibberish1["is_coherent"] = False
    incoherent.append(gibberish1)

    # Method 2: Random TKS symbols insertion
    gibberish2 = entry.copy()
    symbols = ["A1", "B5", "C3", "D7", "+T", "-T", "o", "^", "_d"]
    words = text.split()
    for _ in range(len(words) // 3):
        pos = random.randint(0, len(words))
        words.insert(pos, random.choice(symbols))
    gibberish2[text_field] = " ".join(words)
    gibberish2["style"] = "gibberish_symbols"
    gibberish2["is_coherent"] = False
    incoherent.append(gibberish2)

    return incoherent[:num_examples]


def process_corpus(input_file: str, output_file: str, max_entries: int = None,
                   include_incoherent: bool = True):
    """Process a corpus file and generate multi-style augmented data."""
    print(f"Processing {input_file}...")

    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_entries and i >= max_entries:
                break
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(entries)} entries")

    # Augment each entry
    augmented_data = []
    for i, entry in enumerate(entries):
        if i % 100 == 0:
            print(f"  Processing entry {i}/{len(entries)}...")

        # Get style variations
        styled = augment_entry(entry)
        augmented_data.extend(styled)

        # Add incoherent examples for contrast
        if include_incoherent:
            incoherent = create_incoherent_examples(entry)
            augmented_data.extend(incoherent)

    # Shuffle
    random.shuffle(augmented_data)

    # Write output
    print(f"Writing {len(augmented_data)} augmented entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in augmented_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Print statistics
    style_counts = {}
    coherent_count = 0
    for entry in augmented_data:
        style = entry.get("style", "unknown")
        style_counts[style] = style_counts.get(style, 0) + 1
        if entry.get("is_coherent", True):
            coherent_count += 1

    print("\nStyle distribution:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count}")
    print(f"\nCoherent: {coherent_count}, Incoherent: {len(augmented_data) - coherent_count}")

    return augmented_data


def main():
    parser = argparse.ArgumentParser(description="Augment TKS data with multiple styles")
    parser.add_argument("--input", "-i", default="data/story_eq_pairs.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", "-o", default="data/coherence_training_multistyle.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max", "-m", type=int, default=None,
                        help="Maximum entries to process")
    parser.add_argument("--no-incoherent", action="store_true",
                        help="Don't include incoherent examples")

    args = parser.parse_args()

    # Process the corpus
    process_corpus(
        args.input,
        args.output,
        max_entries=args.max,
        include_incoherent=not args.no_incoherent
    )

    print("\nDone! Multi-style augmented data ready for coherence training.")


if __name__ == "__main__":
    main()
