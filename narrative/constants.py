"""
TKS Narrative Semantics - Canonical Constants

Canonical mappings from:
- TKS v7.4 Formal Mathematical Manual
- TKS_Symbol_Sense_Table_v1.0.md
- TKS_Narrative_Semantics_Rulebook_v1.0.md

All mappings are deterministic and canonical. No new symbols allowed.
"""
from typing import Dict, List, Set, Tuple, Optional

# =============================================================================
# WORLDS (Canonical - A/B/C/D only)
# =============================================================================

WORLDS: Dict[str, str] = {
    "A": "Spiritual",
    "B": "Mental",
    "C": "Emotional",
    "D": "Physical",
}

WORLD_LETTERS: Set[str] = {"A", "B", "C", "D"}

# World keywords for detection
WORLD_KEYWORDS: Dict[str, str] = {
    # A - Spiritual
    "spiritual": "A",
    "divine": "A",
    "soul": "A",
    "spirit": "A",
    "purpose": "A",
    "sacred": "A",
    "holy": "A",
    "transcendent": "A",
    "god": "A",
    "meaning": "A",
    "blueprint": "A",
    "alignment": "A",
    "aligned": "A",
    # B - Mental
    "mental": "B",
    "thought": "B",
    "think": "B",
    "thinking": "B",
    "idea": "B",
    "ideas": "B",
    "belief": "B",
    "believe": "B",
    "mind": "B",
    "intellectual": "B",
    "concept": "B",
    "concepts": "B",
    "understand": "B",
    "understanding": "B",
    "knowledge": "B",
    "awareness": "B",
    "aware": "B",
    "conscious": "B",
    "consciousness": "B",
    "attention": "B",
    "focus": "B",
    "clarity": "B",
    "learning": "B",
    "learn": "B",
    "study": "B",
    "education": "B",
    "wisdom": "B",
    "memory": "B",
    "memories": "B",
    "decision": "B",
    "decisions": "B",
    "logic": "B",
    "reasoning": "B",
    "plan": "B",
    "plans": "B",
    "strategy": "B",
    "vision": "B",
    "faith": "B",
    "trust": "B",
    "confidence": "B",
    "hope": "B",
    "optimism": "B",
    "doubt": "B",
    "pessimism": "B",
    "skepticism": "B",
    # C - Emotional
    "emotional": "C",
    "emotion": "C",
    "feel": "C",
    "feeling": "C",
    "feelings": "C",
    "felt": "C",
    "love": "C",
    "loved": "C",
    "loving": "C",
    "fear": "C",
    "feared": "C",
    "afraid": "C",
    "scared": "C",
    "joy": "C",
    "happy": "C",
    "happiness": "C",
    "sad": "C",
    "sadness": "C",
    "anger": "C",
    "angry": "C",
    "desire": "C",
    "desires": "C",
    "passion": "C",
    "anxiety": "C",
    "anxious": "C",
    "worry": "C",
    "worried": "C",
    "hate": "C",
    "hatred": "C",
    "grief": "C",
    "sorrow": "C",
    "depression": "C",
    "shame": "C",
    "guilt": "C",
    "disgust": "C",
    "aversion": "C",
    "rejection": "C",
    "delight": "C",
    "pleasure": "C",
    "attraction": "C",
    "attracted": "C",
    "excitement": "C",
    "openness": "C",
    "open": "C",
    "receptive": "C",
    "expression": "C",
    "expressive": "C",
    # D - Physical
    "physical": "D",
    "body": "D",
    "material": "D",
    "money": "D",
    "wealth": "D",
    "resources": "D",
    "possessions": "D",
    "property": "D",
    "health": "D",
    "healthy": "D",
    "wellness": "D",
    "vitality": "D",
    "illness": "D",
    "sick": "D",
    "disease": "D",
    "disorder": "D",
    "action": "D",
    "do": "D",
    "did": "D",
    "done": "D",
    "doing": "D",
    "woman": "D",
    "women": "D",
    "man": "D",
    "men": "D",
    "person": "D",
    "people": "D",
    "mother": "D",
    "father": "D",
    "daughter": "D",
    "son": "D",
    "sister": "D",
    "brother": "D",
    "wife": "D",
    "husband": "D",
    "girlfriend": "D",
    "boyfriend": "D",
    "partner": "D",
    "female": "D",
    "male": "D",
    "thing": "D",
    "object": "D",
    "place": "D",
    "world": "D",
    "situation": "D",
    "habit": "D",
    "habits": "D",
    "routine": "D",
    "pattern": "D",
    "cycle": "D",
    "cycles": "D",
    "repetition": "D",
    "energy": "D",
    "energetic": "D",
    "intensity": "D",
    "vibration": "D",
    "trigger": "D",
    "triggers": "D",
    "elevation": "D",
    "result": "D",
    "results": "D",
    "consequence": "D",
    "consequences": "D",
    "effect": "D",
    "effects": "D",
    "control": "D",
    "authority": "D",
    "power": "D",
    "influence": "D",
    "status": "D",
    "structure": "D",
    "framework": "D",
    "vessel": "D",
    "container": "D",
    "order": "D",
    "harmony": "D",
    "chaos": "D",
    "instability": "D",
    "change": "D",
    "transformation": "D",
    "always": "D",
    "never": "D",
    "repeated": "D",
    "repeatedly": "D",
}

# =============================================================================
# NOETICS (Canonical - 1-10 fixed)
# =============================================================================

NOETIC_NAMES: Dict[int, str] = {
    1: "Mind",
    2: "Positive",
    3: "Negative",
    4: "Vibration",
    5: "Female",
    6: "Male",
    7: "Rhythm",
    8: "Cause",
    9: "Effect",
    10: "Idea",
}

# Involution pairs (canonical)
NOETIC_INVOLUTIONS: Dict[int, int] = {
    2: 3, 3: 2,   # Positive <-> Negative
    5: 6, 6: 5,   # Female <-> Male
    8: 9, 9: 8,   # Cause <-> Effect
}

# Self-dual noetics
NOETIC_SELF_DUALS: Set[int] = {1, 4, 7, 10}

# Noetic keywords for detection
NOETIC_KEYWORDS: Dict[str, int] = {
    # N1 - Mind
    "mind": 1,
    "conscious": 1,
    "consciousness": 1,
    "awareness": 1,
    "aware": 1,
    "attention": 1,
    "focus": 1,
    "clarity": 1,
    "clear": 1,
    # N2 - Positive
    "positive": 2,
    "attraction": 2,
    "attract": 2,
    "attracted": 2,
    "order": 2,
    "harmony": 2,
    "health": 2,
    "healthy": 2,
    "wellness": 2,
    "vitality": 2,
    "good": 2,
    "joy": 2,
    "happy": 2,
    "happiness": 2,
    "love": 2,
    "loved": 2,
    "loving": 2,
    "delight": 2,
    "pleasure": 2,
    "faith": 2,
    "trust": 2,
    "confidence": 2,
    "optimism": 2,
    "hope": 2,
    "alignment": 2,
    "aligned": 2,
    # N3 - Negative
    "negative": 3,
    "repulsion": 3,
    "reject": 3,
    "rejection": 3,
    "disorder": 3,
    "chaos": 3,
    "instability": 3,
    "illness": 3,
    "sick": 3,
    "disease": 3,
    "bad": 3,
    "fear": 3,
    "feared": 3,
    "afraid": 3,
    "scared": 3,
    "anxiety": 3,
    "anxious": 3,
    "worry": 3,
    "worried": 3,
    "anger": 3,
    "angry": 3,
    "hate": 3,
    "hatred": 3,
    "aversion": 3,
    "sadness": 3,
    "sad": 3,
    "grief": 3,
    "sorrow": 3,
    "depression": 3,
    "shame": 3,
    "guilt": 3,
    "disgust": 3,
    "doubt": 3,
    "limiting": 3,
    "pessimism": 3,
    "skepticism": 3,
    # N4 - Vibration
    "vibration": 4,
    "intensity": 4,
    "intense": 4,
    "energy": 4,
    "energetic": 4,
    "volume": 4,
    "frequency": 4,
    "passion": 4,
    "excitement": 4,
    "intellectual": 4,
    # N5 - Female
    "female": 5,
    "woman": 5,
    "women": 5,
    "mother": 5,
    "daughter": 5,
    "sister": 5,
    "wife": 5,
    "girlfriend": 5,
    "she": 5,
    "her": 5,
    "receptive": 5,
    "receptivity": 5,
    "receive": 5,
    "receiving": 5,
    "vessel": 5,
    "container": 5,
    "learning": 5,
    "learn": 5,
    "openness": 5,
    "open": 5,
    # N6 - Male
    "male": 6,
    "man": 6,
    "men": 6,
    "father": 6,
    "son": 6,
    "brother": 6,
    "husband": 6,
    "boyfriend": 6,
    "he": 6,
    "him": 6,
    "his": 6,
    "projective": 6,
    "project": 6,
    "projection": 6,
    "structure": 6,
    "framework": 6,
    "deliver": 6,
    "delivery": 6,
    "decision": 6,
    "expression": 6,
    "expressive": 6,
    "assertion": 6,
    # N7 - Rhythm
    "rhythm": 7,
    "pattern": 7,
    "habit": 7,
    "habits": 7,
    "cycle": 7,
    "cycles": 7,
    "repeat": 7,
    "repeated": 7,
    "repeatedly": 7,
    "repetition": 7,
    "routine": 7,
    "always": 7,
    "never": 7,
    # N8 - Cause/Above
    "cause": 8,
    "causes": 8,
    "caused": 8,
    "trigger": 8,
    "triggers": 8,
    "above": 8,
    "elevated": 8,
    "elevation": 8,
    "authority": 8,
    "origin": 8,
    "because": 8,
    "control": 8,
    "power": 8,
    "influence": 8,
    "status": 8,
    # N9 - Effect/Below
    "effect": 9,
    "effects": 9,
    "result": 9,
    "results": 9,
    "resulted": 9,
    "below": 9,
    "grounded": 9,
    "foundation": 9,
    "consequence": 9,
    "consequences": 9,
    "change": 9,
    "transformation": 9,
    # N10 - Idea
    "idea": 10,
    "ideas": 10,
    "concept": 10,
    "concepts": 10,
    "plan": 10,
    "plans": 10,
    "strategy": 10,
    "vision": 10,
    "money": 10,
    "wealth": 10,
    "resources": 10,
    "material": 10,
    "possessions": 10,
    "property": 10,
    "situation": 10,
    "object": 10,
    "thing": 10,
    "god": 10,
    "purpose": 10,
    "meaning": 10,
    "blueprint": 10,
    "world": 10,
    "place": 10,
}

# =============================================================================
# FOUNDATIONS (Canonical - 1-7 fixed)
# =============================================================================

FOUNDATIONS: Dict[int, str] = {
    1: "Unity",
    2: "Wisdom",
    3: "Life",
    4: "Companionship",
    5: "Power",
    6: "Material",
    7: "Lust",
}

# Foundation opposites
FOUNDATION_OPPOSITES: Dict[int, int] = {
    1: 7, 7: 1,   # Unity <-> Lust
    2: 6, 6: 2,   # Wisdom <-> Material
    3: 5, 5: 3,   # Life <-> Power
    4: 4,         # Companionship is self-dual
}

# Foundation keywords for detection
FOUNDATION_KEYWORDS: Dict[str, int] = {
    # F1 - Unity
    "unity": 1,
    "god": 1,
    "divine": 1,
    "oneness": 1,
    "purpose": 1,
    "meaning": 1,
    "spiritual union": 1,
    "connection to god": 1,
    "divine purpose": 1,
    "life purpose": 1,
    "soul mission": 1,
    # F2 - Wisdom
    "wisdom": 2,
    "knowledge": 2,
    "learning": 2,
    "understanding": 2,
    "education": 2,
    "study": 2,
    "intellectual wisdom": 2,
    "spiritual wisdom": 2,
    "intuitive wisdom": 2,
    "practical wisdom": 2,
    "insight": 2,
    "revelation": 2,
    "teaching": 2,
    # F3 - Life
    "life": 3,
    "health": 3,
    "vitality": 3,
    "survival": 3,
    "energy": 3,
    "physical health": 3,
    "mental health": 3,
    "emotional health": 3,
    "spiritual vitality": 3,
    "wellness": 3,
    "healing": 3,
    # F4 - Companionship
    "companionship": 4,
    "love": 4,
    "relationship": 4,
    "relationships": 4,
    "partner": 4,
    "friendship": 4,
    "friend": 4,
    "friends": 4,
    "family": 4,
    "soul connection": 4,
    "emotional relationship": 4,
    "physical companionship": 4,
    "intellectual partnership": 4,
    "connection": 4,
    "intimacy": 4,
    "bonding": 4,
    "togetherness": 4,
    # F5 - Power
    "power": 5,
    "control": 5,
    "influence": 5,
    "authority": 5,
    "status": 5,
    "dominance": 5,
    "spiritual authority": 5,
    "intellectual power": 5,
    "emotional influence": 5,
    "material power": 5,
    "command": 5,
    "leadership": 5,
    # F6 - Material
    "material": 6,
    "money": 6,
    "wealth": 6,
    "resources": 6,
    "possessions": 6,
    "property": 6,
    "financial": 6,
    "physical resources": 6,
    "spiritual abundance": 6,
    "ideas about wealth": 6,
    "feelings about money": 6,
    "abundance": 6,
    "prosperity": 6,
    # F7 - Lust
    "lust": 7,
    "sex": 7,
    "sexual": 7,
    "desire": 7,
    "creation": 7,
    "creative": 7,
    "reproduction": 7,
    "passion": 7,
    "creative spirit": 7,
    "creative ideas": 7,
    "physical creation": 7,
    "procreation": 7,
}

# Sub-foundation mapping: (foundation_id, world) -> label
SUBFOUND_MAP: Dict[Tuple[int, str], str] = {
    # Foundation 1: Unity
    (1, "A"): "Spiritual union",
    (1, "B"): "Mental unity",
    (1, "C"): "Emotional unity",
    (1, "D"): "Physical unity",
    # Foundation 2: Wisdom
    (2, "A"): "Spiritual wisdom",
    (2, "B"): "Intellectual wisdom",
    (2, "C"): "Intuitive wisdom",
    (2, "D"): "Practical wisdom",
    # Foundation 3: Life/Health
    (3, "A"): "Spiritual vitality",
    (3, "B"): "Mental health",
    (3, "C"): "Emotional health",
    (3, "D"): "Physical health",
    # Foundation 4: Companionship
    (4, "A"): "Soul connection",
    (4, "B"): "Intellectual partnership",
    (4, "C"): "Emotional relationship",
    (4, "D"): "Physical companionship",
    # Foundation 5: Power
    (5, "A"): "Spiritual authority",
    (5, "B"): "Intellectual power",
    (5, "C"): "Emotional influence",
    (5, "D"): "Material power",
    # Foundation 6: Material
    (6, "A"): "Spiritual abundance",
    (6, "B"): "Ideas about wealth",
    (6, "C"): "Feelings about money",
    (6, "D"): "Physical resources",
    # Foundation 7: Lust/Creation
    (7, "A"): "Creative spirit",
    (7, "B"): "Creative ideas",
    (7, "C"): "Desire/passion",
    (7, "D"): "Physical creation",
}

# =============================================================================
# OPERATORS (Canonical)
# =============================================================================

# From TKS_Narrative_Semantics_Rulebook_v1.0.md:
# - TOOTRA operators: +_T, -_T, ×_T, /_T
# - Composition: ∘ (sequential), → (causal)
# - Reverse causal: ← (mentioned in some contexts)

OPERATORS: Dict[str, str] = {
    "+T": "together with",
    "-T": "without",
    "*T": "intensified by",       # ×_T as *T
    "/T": "in conflict with",     # /_T as /T
    "o": "then",                  # ∘ as o (sequential composition)
    "->": "causes",
    "<-": "caused by",
    "+": "and",
    "-": "minus",
}

# Canonical operator tokens - all explicitly defined in rulebook
ALLOWED_OPS: Set[str] = {"+T", "-T", "*T", "/T", "o", "->", "<-", "+", "-"}

# Deprecated: Use ALLOWED_OPS instead
OPERATOR_TOKENS: Set[str] = ALLOWED_OPS

# Verb-to-operator mapping (from rulebook)
VERB_TO_OPERATOR: Dict[str, str] = {
    # Combining -> +T
    "with": "+T",
    "and": "+T",
    "together": "+T",
    "combined": "+T",
    "combines": "+T",
    "fused": "+T",
    # Removing -> -T
    "without": "-T",
    "removes": "-T",
    "removed": "-T",
    "hides": "-T",
    "hidden": "-T",
    "lacks": "-T",
    "minus": "-T",
    # Intensifying -> *T (×_T)
    "amplifies": "*T",
    "amplified": "*T",
    "increases": "*T",
    "increased": "*T",
    "multiplies": "*T",
    "multiplied": "*T",
    "intensifies": "*T",
    "intensified": "*T",
    "modulates": "*T",
    "modulated": "*T",
    # Conflicting -> /T (/_T)
    "opposes": "/T",
    "opposed": "/T",
    "fights": "/T",
    "divides": "/T",
    "divided": "/T",
    "conflicts": "/T",
    "conflict": "/T",
    # Causing -> ->
    "causes": "->",
    "caused": "->",
    "leads": "->",
    "led": "->",
    "results": "->",
    "resulted": "->",
    "triggers": "->",
    "triggered": "->",
    "produces": "->",
    "produced": "->",
    "transforms": "->",
    "transformed": "->",
    # Sequencing -> o (∘)
    "then": "o",
    "after": "o",
    "followed": "o",
}

# =============================================================================
# ELEMENT SENSE TABLE (Canonical from Symbol Sense Table v1.0)
# Default senses for common elements
# =============================================================================

# Element -> (default_label, default_sense)
ELEMENT_DEFAULTS: Dict[str, Tuple[str, int]] = {
    # D-world defaults
    "D1": ("body awareness", 1),
    "D2": ("health", 1),
    "D3": ("illness", 1),
    "D4": ("physical energy", 1),
    "D5": ("woman", 1),
    "D6": ("man", 1),
    "D7": ("habit", 1),
    "D8": ("physical authority", 1),
    "D9": ("physical effect", 1),
    "D10": ("physical template", 1),
    # C-world defaults
    "C1": ("emotional awareness", 1),
    "C2": ("joy", 1),
    "C3": ("fear", 1),
    "C4": ("emotional intensity", 1),
    "C5": ("emotional openness", 1),
    "C6": ("emotional expression", 1),
    "C7": ("emotional pattern", 1),
    "C8": ("emotional cause", 1),
    "C9": ("emotional effect", 1),
    "C10": ("emotional template", 1),
    # B-world defaults
    "B1": ("clear thinking", 1),
    "B2": ("positive belief", 1),
    "B3": ("limiting belief", 1),
    "B4": ("mental intensity", 1),
    "B5": ("mental receptivity", 1),
    "B6": ("mental assertion", 1),
    "B7": ("thought pattern", 1),
    "B8": ("mental cause", 1),
    "B9": ("mental effect", 1),
    "B10": ("mental template", 1),
    # A-world defaults
    "A1": ("spiritual awareness", 1),
    "A2": ("divine alignment", 1),
    "A3": ("spiritual misalignment", 1),
    "A4": ("spiritual intensity", 1),
    "A5": ("divine receptivity", 1),
    "A6": ("divine will", 1),
    "A7": ("spiritual rhythm", 1),
    "A8": ("spiritual cause", 1),
    "A9": ("spiritual effect", 1),
    "A10": ("spiritual template", 1),
}

# Sense labels for decoding (expanded from TKS_Symbol_Sense_Table_v1.0.md)
SENSE_LABELS: Dict[str, str] = {
    # D-world (Physical)
    "D1.1": "body awareness",
    "D1.2": "material perception",
    "D1.3": "physical presence",
    "D2.1": "health",
    "D2.2": "material order",
    "D2.3": "physical attraction",
    "D2.4": "positive physical habit",
    "D3.1": "illness",
    "D3.2": "material chaos",
    "D3.3": "physical repulsion",
    "D3.4": "decay",
    "D4.1": "physical energy",
    "D4.2": "material intensity",
    "D4.3": "physical frequency",
    "D5.1": "a woman",
    "D5.2": "receptacle",
    "D5.3": "nurturing environment",
    "D5.4": "physical receptivity",
    "D6.1": "a man",
    "D6.2": "structure",
    "D6.3": "delivery mechanism",
    "D6.4": "physical assertion",
    "D7.1": "habit",
    "D7.2": "biological cycle",
    "D7.3": "environmental cycle",
    "D7.4": "repetitive action",
    "D8.1": "physical elevation",
    "D8.2": "physical trigger",
    "D8.3": "material authority",
    "D8.4": "high quality",
    "D9.1": "physical foundation",
    "D9.2": "physical effect",
    "D9.3": "grounded state",
    "D9.4": "basic needs",
    "D10.1": "physical template",
    "D10.2": "material potential",
    "D10.3": "physical specification",
    # C-world (Emotional)
    "C1.1": "emotional awareness",
    "C1.2": "emotional intelligence",
    "C1.3": "felt sense",
    "C2.1": "joy",
    "C2.2": "emotional attraction",
    "C2.3": "love",
    "C2.4": "enthusiasm",
    "C3.1": "fear",
    "C3.2": "emotional aversion",
    "C3.3": "anger",
    "C3.4": "sadness",
    "C4.1": "emotional intensity",
    "C4.2": "emotional energy",
    "C4.3": "emotional amplitude",
    "C5.1": "emotional receptivity",
    "C5.2": "emotional intimacy",
    "C5.3": "accumulated feelings",
    "C5.4": "empathic reception",
    "C6.1": "emotional expression",
    "C6.2": "emotional boundary",
    "C6.3": "emotional transmission",
    "C6.4": "emotional assertion",
    "C7.1": "mood cycle",
    "C7.2": "emotional habit",
    "C7.3": "feeling rhythm",
    "C8.1": "emotional trigger",
    "C8.2": "elevated emotion",
    "C8.3": "emotional authority",
    "C9.1": "emotional response",
    "C9.2": "emotional foundation",
    "C9.3": "basic emotions",
    "C10.1": "emotional concept",
    "C10.2": "emotional potential",
    "C10.3": "named emotion",
    # B-world (Mental)
    "B1.1": "meta-cognition",
    "B1.2": "mental clarity",
    "B1.3": "cognitive awareness",
    "B2.1": "positive belief",
    "B2.2": "mental order",
    "B2.3": "cognitive attraction",
    "B2.4": "optimistic thinking",
    "B3.1": "limiting belief",
    "B3.2": "mental confusion",
    "B3.3": "cognitive aversion",
    "B3.4": "pessimistic thinking",
    "B4.1": "thought intensity",
    "B4.2": "mental energy",
    "B4.3": "concentration",
    "B5.1": "learning receptivity",
    "B5.2": "accumulated knowledge",
    "B5.3": "mental programming",
    "B5.4": "intuition",
    "B6.1": "logical thinking",
    "B6.2": "idea expression",
    "B6.3": "mental structure",
    "B6.4": "analysis",
    "B7.1": "thought pattern",
    "B7.2": "mental habit",
    "B7.3": "cognitive cycle",
    "B8.1": "thought trigger",
    "B8.2": "higher understanding",
    "B8.3": "mental authority",
    "B9.1": "cognitive response",
    "B9.2": "mental foundation",
    "B9.3": "practical thinking",
    "B10.1": "thought form",
    "B10.2": "mental potential",
    "B10.3": "abstract concept",
    # A-world (Spiritual)
    "A1.1": "soul awareness",
    "A1.2": "divine connection",
    "A1.3": "spiritual presence",
    "A2.1": "spiritual alignment",
    "A2.2": "soul attraction",
    "A2.3": "divine order",
    "A2.4": "spiritual love",
    "A3.1": "spiritual misalignment",
    "A3.2": "soul rejection",
    "A3.3": "spiritual crisis",
    "A4.1": "spiritual frequency",
    "A4.2": "spiritual intensity",
    "A4.3": "aetheric energy",
    "A5.1": "divine receptivity",
    "A5.2": "spiritual inheritance",
    "A5.3": "surrender",
    "A6.1": "divine will",
    "A6.2": "spiritual authority",
    "A6.3": "sacred expression",
    "A7.1": "karmic pattern",
    "A7.2": "soul cycle",
    "A7.3": "sacred rhythm",
    "A8.1": "divine cause",
    "A8.2": "spiritual elevation",
    "A8.3": "divine authority",
    "A9.1": "spiritual effect",
    "A9.2": "soul foundation",
    "A9.3": "grounded spirituality",
    "A10.1": "divine blueprint",
    "A10.2": "spiritual potential",
    "A10.3": "sacred template",
}

# =============================================================================
# SENSE RULES (Explicit overrides from Narrative Semantics Rulebook)
# token -> (world, noetic, sense) - only for tokens with non-default senses
# =============================================================================

SENSE_RULES: Dict[str, Tuple[str, int, int]] = {
    # B-world (Mental) non-default senses
    "experiences": ("B", 5, 2),  # B5.2 = accumulated knowledge
    "experience": ("B", 5, 2),
    "accumulated": ("B", 5, 2),
    "accumulated knowledge": ("B", 5, 2),
    "past experiences": ("B", 5, 2),
    "negative thought": ("B", 3, 3),  # B3.3 = cognitive aversion
    "cognitive aversion": ("B", 3, 3),
    "mental programming": ("B", 5, 3),  # B5.3 = mental programming
    "conditioning": ("B", 5, 3),
    "programmed thinking": ("B", 5, 3),
    "intuition": ("B", 5, 4),  # B5.4 = intuition
    "intuitive": ("B", 5, 4),
    "insight": ("B", 8, 2),  # B8.2 = higher understanding
    "higher understanding": ("B", 8, 2),
    "revelation": ("B", 8, 2),
    "mental clarity": ("B", 1, 2),  # B1.2 = mental clarity
    "clear thinking": ("B", 1, 2),
    "analysis": ("B", 6, 4),  # B6.4 = analysis
    "analyzing": ("B", 6, 4),
    "examination": ("B", 6, 4),
    "mental confusion": ("B", 3, 2),  # B3.2 = mental confusion
    "confusion": ("B", 3, 2),
    "scattered thoughts": ("B", 3, 2),
    "concentration": ("B", 4, 3),  # B4.3 = concentration

    # C-world (Emotional) non-default senses
    "aversion": ("C", 3, 2),  # C3.2 = emotional aversion
    "emotional aversion": ("C", 3, 2),
    "repulsion": ("C", 3, 2),
    "anger": ("C", 3, 3),  # C3.3 = anger
    "hostility": ("C", 3, 3),
    "rage": ("C", 3, 3),
    "resentment": ("C", 3, 3),
    "sadness": ("C", 3, 4),  # C3.4 = sadness/grief
    "grief": ("C", 3, 4),
    "mourning": ("C", 3, 4),
    "sorrow": ("C", 3, 4),
    "loss": ("C", 3, 4),
    "emotional attraction": ("C", 2, 2),  # C2.2 = emotional attraction
    "drawn to": ("C", 2, 2),
    "affection": ("C", 2, 3),  # C2.3 = love/affection
    "care": ("C", 2, 3),
    "caring": ("C", 2, 3),
    "warmth": ("C", 2, 3),
    "enthusiasm": ("C", 2, 4),  # C2.4 = enthusiasm/passion
    "passionate": ("C", 2, 4),
    "excited": ("C", 2, 4),
    "emotional intimacy": ("C", 5, 2),  # C5.2 = emotional intimacy
    "intimacy": ("C", 5, 2),
    "closeness": ("C", 5, 2),
    "emotional baggage": ("C", 5, 3),  # C5.3 = accumulated feelings
    "past feelings": ("C", 5, 3),
    "emotional patterns": ("C", 5, 3),
    "empathy": ("C", 5, 4),  # C5.4 = empathic reception
    "empathic": ("C", 5, 4),
    "feeling others": ("C", 5, 4),
    "emotional boundary": ("C", 6, 2),  # C6.2 = emotional boundary
    "boundaries": ("C", 6, 2),
    "containing feelings": ("C", 6, 2),
    "emotional trigger": ("C", 8, 1),  # C8.1 = emotional trigger (default)
    "triggered": ("C", 8, 1),
    "elevated emotion": ("C", 8, 2),  # C8.2 = elevated emotion
    "peak emotion": ("C", 8, 2),
    "transcendent feeling": ("C", 8, 2),

    # D-world (Physical) non-default senses
    "instability": ("D", 3, 2),  # D3.2 = material chaos
    "chaos": ("D", 3, 2),
    "disorder": ("D", 3, 2),
    "clutter": ("D", 3, 2),
    "mess": ("D", 3, 2),
    "disorganized": ("D", 3, 2),
    "decay": ("D", 3, 4),  # D3.4 = decay/entropy
    "deterioration": ("D", 3, 4),
    "aging": ("D", 3, 4),
    "breakdown": ("D", 3, 4),
    "rot": ("D", 3, 4),
    "entropy": ("D", 3, 4),
    "control": ("D", 8, 3),  # D8.3 = material authority
    "authority": ("D", 8, 3),
    "power over": ("D", 8, 3),
    "dominance": ("D", 8, 3),
    "high quality": ("D", 8, 4),  # D8.4 = high quality
    "premium": ("D", 8, 4),
    "superior": ("D", 8, 4),
    "upgraded": ("D", 8, 4),
    "receptacle": ("D", 5, 2),  # D5.2 = vessel/container
    "vessel": ("D", 5, 2),
    "container": ("D", 5, 2),
    "womb": ("D", 5, 2),
    "holding": ("D", 5, 2),
    "nurturing environment": ("D", 5, 3),  # D5.3 = nurturing environment
    "safe space": ("D", 5, 3),
    "home": ("D", 5, 3),
    "nest": ("D", 5, 3),
    "physical receptivity": ("D", 5, 4),  # D5.4 = physical receptivity
    "absorbing": ("D", 5, 4),
    "taking in": ("D", 5, 4),
    "structure": ("D", 6, 2),  # D6.2 = framework
    "framework": ("D", 6, 2),
    "skeleton": ("D", 6, 2),
    "architecture": ("D", 6, 2),
    "delivery mechanism": ("D", 6, 3),  # D6.3 = delivery mechanism
    "transmission": ("D", 6, 3),
    "delivering": ("D", 6, 3),
    "projecting": ("D", 6, 3),
    "physical assertion": ("D", 6, 4),  # D6.4 = physical assertion
    "asserting": ("D", 6, 4),
    "active behavior": ("D", 6, 4),
    "biological cycle": ("D", 7, 2),  # D7.2 = biological cycle
    "circadian": ("D", 7, 2),
    "sleep cycle": ("D", 7, 2),
    "body rhythm": ("D", 7, 2),
    "digestion": ("D", 7, 2),
    "environmental cycle": ("D", 7, 3),  # D7.3 = environmental cycle
    "seasons": ("D", 7, 3),
    "day and night": ("D", 7, 3),
    "tides": ("D", 7, 3),
    "natural rhythm": ("D", 7, 3),
    "repetitive action": ("D", 7, 4),  # D7.4 = repetitive action
    "repeated action": ("D", 7, 4),
    "doing repeatedly": ("D", 7, 4),
    "material order": ("D", 2, 2),  # D2.2 = material order
    "organization": ("D", 2, 2),
    "tidiness": ("D", 2, 2),
    "organized space": ("D", 2, 2),
    "neat": ("D", 2, 2),
    "physical attraction": ("D", 2, 3),  # D2.3 = physical attraction
    "magnetism": ("D", 2, 3),
    "magnetic pull": ("D", 2, 3),
    "positive habit": ("D", 2, 4),  # D2.4 = positive physical habit
    "healthy routine": ("D", 2, 4),
    "exercise routine": ("D", 2, 4),
    "good practice": ("D", 2, 4),
    "physical repulsion": ("D", 3, 3),  # D3.3 = physical repulsion
    "aversion to": ("D", 3, 3),
    "pushing away": ("D", 3, 3),
    "material potential": ("D", 10, 2),  # D10.2 = material potential
    "unmade things": ("D", 10, 2),
    "could become": ("D", 10, 2),
    "physical specification": ("D", 10, 3),  # D10.3 = physical specification
    "physical blueprint": ("D", 10, 3),  # explicit D-world blueprint
    "recipe": ("D", 10, 3),
    "instructions": ("D", 10, 3),
    "design": ("D", 10, 3),

    # A-world (Spiritual) non-default senses
    "divine connection": ("A", 1, 2),  # A1.2 = divine connection
    "connection to god": ("A", 1, 2),
    "connection to source": ("A", 1, 2),
    "spiritual presence": ("A", 1, 3),  # A1.3 = spiritual presence
    "spiritual awakeness": ("A", 1, 3),
    "soul attraction": ("A", 2, 2),  # A2.2 = soul attraction
    "soul pull": ("A", 2, 2),
    "calling": ("A", 2, 2),
    "divine order": ("A", 2, 3),  # A2.3 = divine order
    "cosmic harmony": ("A", 2, 3),
    "cosmic order": ("A", 2, 3),
    "spiritual love": ("A", 2, 4),  # A2.4 = spiritual love
    "divine love": ("A", 2, 4),
    "unconditional love": ("A", 2, 4),
    "soul rejection": ("A", 3, 2),  # A3.2 = soul rejection
    "spiritual resistance": ("A", 3, 2),
    "soul resistance": ("A", 3, 2),
    "spiritual crisis": ("A", 3, 3),  # A3.3 = spiritual crisis
    "dark night": ("A", 3, 3),
    "dark night of the soul": ("A", 3, 3),
    "spiritual frequency": ("A", 4, 1),  # A4.1 = default
    "vibrational level": ("A", 4, 1),
    "spiritual intensity": ("A", 4, 2),  # A4.2 = spiritual intensity
    "strength of spiritual experience": ("A", 4, 2),
    "aetheric energy": ("A", 4, 3),  # A4.3 = aetheric energy
    "chi": ("A", 4, 3),
    "prana": ("A", 4, 3),
    "spiritual energy": ("A", 4, 3),
    "subtle energy": ("A", 4, 3),
    "spiritual inheritance": ("A", 5, 2),  # A5.2 = spiritual inheritance
    "ancestral wisdom": ("A", 5, 2),
    "spiritual heritage": ("A", 5, 2),
    "inherited wisdom": ("A", 5, 2),
    "surrender": ("A", 5, 3),  # A5.3 = surrender
    "letting go": ("A", 5, 3),
    "surrendering": ("A", 5, 3),
    "spiritual authority": ("A", 6, 2),  # A6.2 = spiritual authority
    "spiritual command": ("A", 6, 2),
    "sacred expression": ("A", 6, 3),  # A6.3 = sacred expression
    "spiritual teaching": ("A", 6, 3),
    "expressing spiritual truth": ("A", 6, 3),
    "karmic pattern": ("A", 7, 1),  # A7.1 = default
    "karma": ("A", 7, 1),
    "spiritual lessons": ("A", 7, 1),
    "soul cycle": ("A", 7, 2),  # A7.2 = soul cycle
    "soul evolution": ("A", 7, 2),
    "spiritual development": ("A", 7, 2),
    "sacred rhythm": ("A", 7, 3),  # A7.3 = sacred rhythm
    "divine timing": ("A", 7, 3),
    "sacred seasons": ("A", 7, 3),
    "spiritual elevation": ("A", 8, 2),  # A8.2 = spiritual elevation
    "transcendence": ("A", 8, 2),
    "spiritual heights": ("A", 8, 2),
    "divine authority": ("A", 8, 3),  # A8.3 = divine authority
    "cosmic command": ("A", 8, 3),
    "ultimate authority": ("A", 8, 3),
    "soul foundation": ("A", 9, 2),  # A9.2 = soul foundation
    "soul essence": ("A", 9, 2),
    "spiritual foundation": ("A", 9, 2),
    "grounded spirituality": ("A", 9, 3),  # A9.3 = grounded spirituality
    "embodied spirituality": ("A", 9, 3),
    "practical spirituality": ("A", 9, 3),
    "spiritual potential": ("A", 10, 2),  # A10.2 = spiritual potential
    "unmanifested purpose": ("A", 10, 2),
    "sacred template": ("A", 10, 3),  # A10.3 = sacred template
    "archetypal form": ("A", 10, 3),
    "sacred pattern": ("A", 10, 3),
}

# =============================================================================
# LEXICON: word/phrase -> element mapping
# Deterministic mapping from natural language to TKS elements
# =============================================================================

LEXICON: Dict[str, Tuple[str, int, Optional[int]]] = {
    # Format: "word" -> (world, noetic, sense_or_none)

    # People/Gender (D-world - Noetic 5 Female, 6 Male)
    "woman": ("D", 5, 1),
    "women": ("D", 5, 1),
    "mother": ("D", 5, 1),
    "daughter": ("D", 5, 1),
    "wife": ("D", 5, 1),
    "girlfriend": ("D", 5, 1),
    "she": ("D", 5, 1),
    "her": ("D", 5, 1),
    "sister": ("D", 5, 1),
    "female": ("D", 5, 1),
    "man": ("D", 6, 1),
    "men": ("D", 6, 1),
    "father": ("D", 6, 1),
    "son": ("D", 6, 1),
    "husband": ("D", 6, 1),
    "boyfriend": ("D", 6, 1),
    "he": ("D", 6, 1),
    "him": ("D", 6, 1),
    "his": ("D", 6, 1),
    "brother": ("D", 6, 1),
    "male": ("D", 6, 1),
    "partner": ("D", 6, 1),  # Default male, context may override
    "person": ("D", 1, 1),  # Neutral person as body awareness
    "people": ("D", 1, 1),

    # Physical Health States (D-world - Noetic 2 Positive, 3 Negative)
    "health": ("D", 2, 1),
    "healthy": ("D", 2, 1),
    "vitality": ("D", 2, 1),
    "wellness": ("D", 2, 1),
    "illness": ("D", 3, 1),
    "sick": ("D", 3, 1),
    "disease": ("D", 3, 1),
    "disorder": ("D", 3, 2),  # D3.2 = material chaos

    # Physical Patterns (D-world - Noetic 7 Rhythm)
    "habit": ("D", 7, 1),
    "habits": ("D", 7, 1),
    "routine": ("D", 7, 1),
    "pattern": ("D", 7, 1),
    "cycle": ("D", 7, 1),
    "cycles": ("D", 7, 1),
    "repetition": ("D", 7, 1),

    # Physical Energy/Intensity (D-world - Noetic 4 Vibration)
    "energy": ("D", 4, 1),
    "energetic": ("D", 4, 1),
    "intensity": ("D", 4, 1),
    "vibration": ("D", 4, 1),

    # Physical Causation (D-world - Noetic 8 Cause, 9 Effect)
    "trigger": ("D", 8, 1),
    "triggers": ("D", 8, 1),
    "elevation": ("D", 8, 1),
    "result": ("D", 9, 1),
    "results": ("D", 9, 1),
    "consequence": ("D", 9, 1),
    "consequences": ("D", 9, 1),
    "effect": ("D", 9, 1),
    "effects": ("D", 9, 1),

    # Physical Objects/Concepts (D-world - Noetic 10 Idea)
    "money": ("D", 10, 1),  # D10.1 = physical template/concept
    "wealth": ("D", 10, 1),
    "resources": ("D", 10, 1),
    "material": ("D", 10, 1),
    "possessions": ("D", 10, 1),
    "property": ("D", 10, 1),
    "situation": ("D", 10, 1),
    "body": ("D", 1, 1),
    "object": ("D", 10, 1),
    "thing": ("D", 10, 1),

    # Physical Actions/Authority (D-world - Noetic 8)
    "control": ("D", 8, 3),  # D8.3 = material authority
    "authority": ("D", 8, 3),
    "power": ("D", 8, 1),
    "influence": ("D", 8, 1),
    "status": ("D", 8, 1),

    # Physical Structure (D-world - Noetic 6)
    "structure": ("D", 6, 2),  # D6.2 = structure
    "framework": ("D", 6, 2),
    "vessel": ("D", 5, 2),  # D5.2 = receptacle
    "container": ("D", 5, 2),

    # Physical Order/Disorder (D-world - Noetic 2/3)
    "order": ("D", 2, 2),
    "harmony": ("D", 2, 1),
    "chaos": ("D", 3, 2),
    "instability": ("D", 3, 2),  # D3.2 = material chaos

    # Emotions - Positive (C-world - Noetic 2 Positive)
    "joy": ("C", 2, 1),
    "happiness": ("C", 2, 1),
    "happy": ("C", 2, 1),
    "love": ("C", 2, 3),  # C2.3 = love (sense 3)
    "loved": ("C", 2, 3),
    "loving": ("C", 2, 3),
    "delight": ("C", 2, 1),
    "pleasure": ("C", 2, 1),
    "attraction": ("C", 2, 1),
    "attracted": ("C", 2, 1),

    # Emotions - Negative (C-world - Noetic 3 Negative)
    # IMPORTANT: Senses here must match SENSE_RULES to stay deterministic.
    # C3.1 = fear (default), C3.2 = aversion, C3.3 = anger, C3.4 = sadness/grief
    "fear": ("C", 3, 1),
    "afraid": ("C", 3, 1),
    "scared": ("C", 3, 1),
    "anxiety": ("C", 3, 1),
    "anxious": ("C", 3, 1),
    "worry": ("C", 3, 1),
    "worried": ("C", 3, 1),
    "anger": ("C", 3, 3),  # C3.3 = anger (matches SENSE_RULES)
    "angry": ("C", 3, 3),
    "hate": ("C", 3, 2),   # C3.2 = emotional aversion
    "hatred": ("C", 3, 2),
    "aversion": ("C", 3, 2),
    "sadness": ("C", 3, 4),  # C3.4 = sadness/grief (matches SENSE_RULES)
    "sad": ("C", 3, 4),
    "grief": ("C", 3, 4),    # C3.4 = grief (matches SENSE_RULES)
    "sorrow": ("C", 3, 4),   # C3.4 = sorrow (matches SENSE_RULES)
    "depression": ("C", 3, 4),  # C3.4 = sadness category
    "shame": ("C", 3, 1),
    "guilt": ("C", 3, 1),
    "disgust": ("C", 3, 2),
    "rejection": ("C", 3, 2),

    # Emotional Awareness (C-world - Noetic 1 Mind)
    "feeling": ("C", 1, 1),
    "feelings": ("C", 1, 1),
    "felt": ("C", 1, 1),
    "feel": ("C", 1, 1),

    # Emotional Intensity (C-world - Noetic 4 Vibration)
    "passion": ("C", 4, 1),
    "intense": ("C", 4, 1),
    "excitement": ("C", 4, 1),

    # Emotional Expression/Receptivity (C-world - Noetic 5/6)
    "openness": ("C", 5, 1),
    "open": ("C", 5, 1),
    "receptive": ("C", 5, 1),
    "expression": ("C", 6, 1),
    "expressive": ("C", 6, 1),

    # Emotional Patterns (C-world - Noetic 7 Rhythm)
    "emotional": ("C", 1, 1),
    "emotion": ("C", 1, 1),

    # Emotional Causation (C-world - Noetic 8/9)
    "desire": ("C", 2, 1),  # Can also be C4.1 depending on context
    "desires": ("C", 2, 1),

    # Mental States - Positive (B-world - Noetic 2 Positive)
    "belief": ("B", 2, 1),
    "believe": ("B", 2, 1),
    "faith": ("B", 2, 1),
    "trust": ("B", 2, 1),
    "confidence": ("B", 2, 1),
    "optimism": ("B", 2, 1),
    "hope": ("B", 2, 1),

    # Mental States - Negative (B-world - Noetic 3 Negative)
    "doubt": ("B", 3, 1),
    "limiting": ("B", 3, 1),
    "negative": ("B", 3, 3),  # B3.3 = negative thought
    "pessimism": ("B", 3, 1),
    "skepticism": ("B", 3, 1),

    # Mental Awareness (B-world - Noetic 1 Mind)
    "thought": ("B", 1, 1),
    "thinking": ("B", 1, 1),
    "think": ("B", 1, 1),
    "awareness": ("B", 1, 1),
    "aware": ("B", 1, 1),
    "conscious": ("B", 1, 1),
    "consciousness": ("B", 1, 1),
    "attention": ("B", 1, 1),
    "focus": ("B", 1, 1),
    "understanding": ("B", 1, 1),
    "understand": ("B", 1, 1),
    "clarity": ("B", 1, 1),
    "clear": ("B", 1, 1),

    # Mental Concepts (B-world - Noetic 10 Idea)
    "idea": ("B", 10, 1),
    "ideas": ("B", 10, 1),
    "concept": ("B", 10, 1),
    "concepts": ("B", 10, 1),
    "plan": ("B", 10, 1),
    "plans": ("B", 10, 1),
    "strategy": ("B", 10, 1),
    "vision": ("B", 10, 1),

    # Mental Receptivity/Learning (B-world - Noetic 5 Female)
    "knowledge": ("B", 5, 2),  # B5.2 = accumulated knowledge
    "experience": ("B", 5, 2),
    "experiences": ("B", 5, 2),
    "learning": ("B", 5, 1),
    "learn": ("B", 5, 1),
    "study": ("B", 5, 1),
    "education": ("B", 5, 2),
    "wisdom": ("B", 5, 2),
    "memory": ("B", 5, 2),
    "memories": ("B", 5, 2),
    "receptivity": ("B", 5, 1),

    # Mental Projection/Structure (B-world - Noetic 6 Male)
    "decision": ("B", 6, 1),
    "decisions": ("B", 6, 1),
    "logic": ("B", 6, 1),
    "reasoning": ("B", 6, 1),
    "assertion": ("B", 6, 1),
    "projection": ("B", 6, 1),

    # Mental Patterns (B-world - Noetic 7 Rhythm)
    "mental": ("B", 1, 1),
    "mind": ("B", 1, 1),

    # Mental Intensity (B-world - Noetic 4 Vibration)
    "intellectual": ("B", 4, 1),

    # Spiritual Awareness (A-world - Noetic 1 Mind)
    "soul": ("A", 1, 1),
    "spirit": ("A", 1, 1),
    "spiritual": ("A", 1, 1),
    "sacred": ("A", 1, 1),
    "holy": ("A", 1, 1),

    # Spiritual Alignment (A-world - Noetic 2 Positive)
    "alignment": ("A", 2, 1),
    "aligned": ("A", 2, 1),

    # Spiritual Divine (A-world - Noetic 6 Male)
    "divine": ("A", 6, 1),
    "transcendent": ("A", 6, 1),

    # Spiritual Purpose (A-world - Noetic 10 Idea)
    "god": ("A", 10, 1),
    "purpose": ("A", 10, 1),
    "meaning": ("A", 10, 1),
    "blueprint": ("A", 10, 1),

    # Action Verbs (primarily D-world)
    "action": ("D", 1, 1),
    "do": ("D", 1, 1),
    "did": ("D", 1, 1),
    "done": ("D", 1, 1),
    "doing": ("D", 1, 1),

    # Time/Temporal markers (D-world Noetic 7)
    "always": ("D", 7, 1),
    "never": ("D", 7, 1),
    "repeated": ("D", 7, 1),
    "repeatedly": ("D", 7, 1),

    # General concepts
    "change": ("D", 9, 1),
    "transformation": ("D", 9, 1),
    "world": ("D", 10, 1),
    "place": ("D", 10, 1),
}

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def is_valid_world(world: str) -> bool:
    """Check if world letter is canonical."""
    return world in WORLD_LETTERS


def is_valid_noetic(noetic: int) -> bool:
    """Check if noetic is canonical (1-10)."""
    return 1 <= noetic <= 10


def is_valid_foundation(fid: int) -> bool:
    """Check if foundation id is canonical (1-7)."""
    return 1 <= fid <= 7


def is_valid_operator(op: str) -> bool:
    """
    Check if operator is canonical.

    Returns True if the operator is defined in the rulebook:
    - TOOTRA operators: +T, -T, *T, /T
    - Composition: o, ->
    - Reverse: <-
    - Basic: +, -
    """
    return op in ALLOWED_OPS


def validate_element(world: str, noetic: int) -> bool:
    """Validate element world and noetic are canonical."""
    return is_valid_world(world) and is_valid_noetic(noetic)


def get_subfound_label(fid: int, world: str) -> Optional[str]:
    """Get sub-foundation label for foundation + world combination."""
    return SUBFOUND_MAP.get((fid, world))


def validate_lexicon_consistency() -> List[Tuple[str, str]]:
    """
    Validate LEXICON entries are consistent with SENSE_RULES.

    Returns list of (word, issue_description) for any conflicts found.
    This ensures deterministic encoding behavior.
    """
    conflicts = []

    for word, (lex_world, lex_noetic, lex_sense) in LEXICON.items():
        # Check if word is also in SENSE_RULES
        if word in SENSE_RULES:
            sr_world, sr_noetic, sr_sense = SENSE_RULES[word]

            # Check for conflicts
            if sr_world != lex_world:
                conflicts.append((
                    word,
                    f"World conflict: LEXICON={lex_world}, SENSE_RULES={sr_world}"
                ))
            elif sr_noetic != lex_noetic:
                conflicts.append((
                    word,
                    f"Noetic conflict: LEXICON={lex_noetic}, SENSE_RULES={sr_noetic}"
                ))
            elif lex_sense is not None and sr_sense != lex_sense:
                conflicts.append((
                    word,
                    f"Sense conflict: LEXICON={lex_sense}, SENSE_RULES={sr_sense}"
                ))

    return conflicts


def get_token_mapping(word: str) -> Optional[Tuple[str, int, Optional[int]]]:
    """
    Get deterministic element mapping for a word.

    Priority order:
    1. SENSE_RULES (explicit sense overrides)
    2. LEXICON (base mappings)
    3. WORLD_KEYWORDS + NOETIC_KEYWORDS (fallback)

    Returns (world, noetic, sense) or None if not found.
    """
    # Priority 1: SENSE_RULES has explicit overrides
    if word in SENSE_RULES:
        return SENSE_RULES[word]

    # Priority 2: LEXICON has base mappings
    if word in LEXICON:
        return LEXICON[word]

    # Priority 3: Infer from keywords
    world = WORLD_KEYWORDS.get(word)
    noetic = NOETIC_KEYWORDS.get(word)

    if world or noetic:
        # Default world if only noetic found
        if not world:
            if noetic in (2, 3):  # Positive/Negative often emotional
                world = "C"
            elif noetic in (5, 6):  # Female/Male often physical
                world = "D"
            else:
                world = "B"  # Default to mental

        # Default noetic if only world found
        if noetic is None:
            noetic = 1

        return (world, noetic, None)

    return None


def check_extended_token_conflicts(token: str) -> Optional[str]:
    """
    Check if an extended token (e.g., B8^5_d5) has any conflicts
    with existing lexicon entries.

    Returns error message if conflict detected, None otherwise.
    """
    import re

    # Parse extended token format
    # Supports: B8, B8.5, B8^5, B8_d5, B8^5_d5

    # Check for foundation suffix conflicts
    if "_" in token:
        parts = token.rsplit("_", 1)
        if len(parts) == 2:
            found_suffix = parts[1]
            # Validate foundation suffix format: [a-d][1-7]
            match = re.match(r'^([a-dA-D])([1-7])$', found_suffix)
            if not match:
                valid_examples = "Valid formats: _a1, _b2, _c3, _d4 ... _d7"
                return f"Invalid foundation suffix '_{found_suffix}'. {valid_examples}"

    # Check for sense notation conflicts (can't have both ^ and .)
    if "^" in token and "." in token:
        # Check if both are present in the base element part
        base = token.split("_")[0] if "_" in token else token
        if "^" in base and "." in base:
            return f"Ambiguous sense notation: use either ^ or . but not both"

    return None
