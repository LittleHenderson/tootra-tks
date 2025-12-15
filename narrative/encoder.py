"""
TKS Narrative Semantics - Encoder

Implements EncodeStory: Natural language story -> TKS expression
Following TKS_Narrative_Semantics_Rulebook_v1.0

Encoding Algorithm:
1. Foundation identification
2. Entity extraction
3. Role assignment
4. Sense selection
5. Operator selection
6. World resolution
7. Temporal mapping
8. Assembly
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

from .constants import (
    WORLDS,
    WORLD_KEYWORDS,
    NOETIC_NAMES,
    NOETIC_KEYWORDS,
    FOUNDATIONS,
    FOUNDATION_KEYWORDS,
    LEXICON,
    VERB_TO_OPERATOR,
    ELEMENT_DEFAULTS,
    OPERATOR_TOKENS,
    ALLOWED_OPS,
    is_valid_world,
    is_valid_noetic,
    is_valid_foundation,
    is_valid_operator,
    SUBFOUND_MAP,
    SENSE_RULES,
    get_subfound_label,
)
from .types import (
    ElementRef,
    TKSExpression,
    TKSToken,
    FoundationTag,
    EncodeResult,
)


# =============================================================================
# TOKENIZER
# =============================================================================

# Stop words to skip during encoding
STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "once",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "here",
    "this", "that", "these", "those", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom",
}


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into words with character spans.

    Returns:
        List of (word, start, end) tuples
    """
    tokens = []
    # Split on whitespace and punctuation, keeping track of positions
    pattern = r'\b\w+\b'
    for match in re.finditer(pattern, text.lower()):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

@dataclass
class ExtractedEntity:
    """An entity extracted from the story."""
    word: str
    world: str
    noetic: int
    sense: Optional[int] = None
    span: Optional[Tuple[int, int]] = None
    role: Optional[str] = None  # AGENT, PATIENT, TARGET, etc.
    confidence: float = 1.0


def extract_entities(tokens: List[Tuple[str, int, int]]) -> List[ExtractedEntity]:
    """
    Extract TKS entities from tokenized text.

    Uses LEXICON for exact matches, then WORLD_KEYWORDS and NOETIC_KEYWORDS
    for partial matching. Applies SENSE_RULES for context-specific sense overrides.
    """
    entities = []
    seen_spans = set()

    for word, start, end in tokens:
        # Skip stop words
        if word in STOP_WORDS:
            continue

        # Skip if we've already processed this span
        if (start, end) in seen_spans:
            continue

        # Try lexicon exact match first
        if word in LEXICON:
            world, noetic, sense = LEXICON[word]

            # Apply sense rule override if available
            if word in SENSE_RULES:
                sense_world, sense_noetic, sense_override = SENSE_RULES[word]
                # Only apply if world and noetic match
                if sense_world == world and sense_noetic == noetic:
                    sense = sense_override

            entities.append(ExtractedEntity(
                word=word,
                world=world,
                noetic=noetic,
                sense=sense,
                span=(start, end),
                confidence=1.0,
            ))
            seen_spans.add((start, end))
            continue

        # Try world + noetic keyword combination
        world = WORLD_KEYWORDS.get(word)
        noetic = NOETIC_KEYWORDS.get(word)

        if world or noetic:
            # Default world to C (emotional) if only noetic found
            if not world:
                # Infer world from noetic context
                if noetic in (2, 3):  # Positive/Negative often emotional
                    world = "C"
                elif noetic in (5, 6):  # Female/Male often physical
                    world = "D"
                else:
                    world = "B"  # Default to mental

            # Default noetic to 1 (Mind/awareness) if only world found
            if noetic is None:
                noetic = 1

            # Check for sense rule
            sense = None
            if word in SENSE_RULES:
                sense_world, sense_noetic, sense_override = SENSE_RULES[word]
                # Only apply if world and noetic match
                if sense_world == world and sense_noetic == noetic:
                    sense = sense_override

            entities.append(ExtractedEntity(
                word=word,
                world=world,
                noetic=noetic,
                sense=sense,
                span=(start, end),
                confidence=0.8,
            ))
            seen_spans.add((start, end))

    return entities


# =============================================================================
# OPERATOR EXTRACTION
# =============================================================================

@dataclass
class ExtractedOperator:
    """An operator extracted from the story."""
    word: str
    operator: str
    span: Tuple[int, int]


def extract_operators(tokens: List[Tuple[str, int, int]]) -> List[ExtractedOperator]:
    """
    Extract TKS operators from tokenized text.

    Maps verbs and conjunctions to TOOTRA operators.
    """
    operators = []

    for word, start, end in tokens:
        if word in VERB_TO_OPERATOR:
            operators.append(ExtractedOperator(
                word=word,
                operator=VERB_TO_OPERATOR[word],
                span=(start, end),
            ))

    return operators


# =============================================================================
# FOUNDATION IDENTIFICATION
# =============================================================================

def detect_foundation(text: str) -> Optional[int]:
    """
    Detect primary foundation from story text using FOUNDATION_KEYWORDS.

    Returns:
        Foundation ID (1-7) or None if no foundation detected
    """
    text_lower = text.lower()
    foundation_scores = {i: 0 for i in range(1, 8)}

    for keyword, fid in FOUNDATION_KEYWORDS.items():
        if keyword in text_lower:
            foundation_scores[fid] += 1

    max_score = max(foundation_scores.values())
    if max_score == 0:
        return None

    # Return first foundation with max score
    for fid, score in foundation_scores.items():
        if score == max_score:
            return fid

    return None


def identify_foundation(text: str) -> Optional[FoundationTag]:
    """
    Identify the primary foundation of the story.

    Returns the foundation with the highest keyword count.
    """
    fid = detect_foundation(text)
    if fid is not None:
        return FoundationTag(foundation=fid)
    return None


# =============================================================================
# ROLE ASSIGNMENT
# =============================================================================

def assign_roles(entities: List[ExtractedEntity], text: str) -> List[ExtractedEntity]:
    """
    Assign semantic roles to extracted entities.

    Roles: AGENT, PATIENT, TARGET, CAUSE, RESULT, INSTRUMENT, CONTEXT
    """
    text_lower = text.lower()

    # Simple heuristics for role assignment
    for entity in entities:
        # People (D5, D6) are typically agents
        if entity.world == "D" and entity.noetic in (5, 6):
            entity.role = "AGENT"
        # Emotions are typically motivations/causes
        elif entity.world == "C":
            entity.role = "MOTIVATION"
        # Physical objects are typically patients
        elif entity.world == "D" and entity.noetic in (10, 1):
            entity.role = "PATIENT"
        # Mental states are typically causes
        elif entity.world == "B":
            entity.role = "CAUSE"
        # Spiritual elements are typically context
        elif entity.world == "A":
            entity.role = "CONTEXT"
        else:
            entity.role = "CONTEXT"

    return entities


# =============================================================================
# TEMPORAL/CAUSAL ORDERING
# =============================================================================

def order_by_causality(
    entities: List[ExtractedEntity],
    operators: List[ExtractedOperator],
    text: str
) -> Tuple[List[ExtractedEntity], List[str]]:
    """
    Order entities by causal/temporal sequence.

    Returns:
        Tuple of (ordered_entities, operators_between)
    """
    if not entities:
        return [], []

    # Simple ordering: CAUSE -> MOTIVATION -> AGENT -> PATIENT -> RESULT
    role_order = {
        "CAUSE": 0,
        "MOTIVATION": 1,
        "AGENT": 2,
        "PATIENT": 3,
        "TARGET": 4,
        "RESULT": 5,
        "CONTEXT": 6,
        "INSTRUMENT": 7,
    }

    # Sort by role, then by span position
    sorted_entities = sorted(
        entities,
        key=lambda e: (role_order.get(e.role, 10), e.span[0] if e.span else 0)
    )

    # Determine operators between entities
    ops = []
    text_lower = text.lower()

    for i in range(len(sorted_entities) - 1):
        curr = sorted_entities[i]
        next_ent = sorted_entities[i + 1]

        # Check for explicit operators between these entities
        found_op = None
        if curr.span and next_ent.span:
            between_text = text_lower[curr.span[1]:next_ent.span[0]]

            for op_extracted in operators:
                if curr.span[1] <= op_extracted.span[0] < next_ent.span[0]:
                    found_op = op_extracted.operator
                    break

        # Default operator based on roles
        if not found_op:
            if curr.role in ("CAUSE", "MOTIVATION") and next_ent.role in ("AGENT", "PATIENT"):
                found_op = "->"  # Causal
            elif "cause" in text_lower or "led" in text_lower or "results" in text_lower:
                found_op = "->"  # Causal from text
            else:
                found_op = "+T"  # Default to combination

        ops.append(found_op)

    return sorted_entities, ops


# =============================================================================
# MAIN ENCODER
# =============================================================================

def EncodeStory(story: str, strict: bool = True) -> TKSExpression:
    """
    Encode a natural language story into a TKS expression.

    This is the main entry point for story encoding.

    Args:
        story: Natural language story text
        strict: If True (default), raise ValueError when unknown tokens are detected.
                If False, warn/skip unknown tokens.

    Returns:
        TKSExpression with elements and operators

    Raises:
        ValueError: If strict=True and unknown tokens are detected
    """
    result = encode_story_full(story, strict=strict)
    return result.expression


def encode_story_full(story: str, strict: bool = True) -> EncodeResult:
    """
    Full encoding with diagnostics.

    Args:
        story: Natural language story text
        strict: If True (default), raise ValueError when unknown tokens are detected.
                If False, warn/skip unknown tokens.

    Returns:
        EncodeResult with expression and diagnostic info

    Raises:
        ValueError: If strict=True and unknown tokens are detected
    """
    warnings = []
    unknown_words = []

    # Step 1: Tokenize
    tokens = tokenize(story)

    # Step 2: Foundation identification
    foundation = identify_foundation(story)

    # Step 3: Entity extraction
    entities = extract_entities(tokens)

    if not entities:
        # Fall back to default element if nothing detected
        warnings.append("No entities detected, using default B1 (mind)")
        entities = [ExtractedEntity(word="default", world="B", noetic=1)]

    # Track unknown words
    for word, start, end in tokens:
        if word not in STOP_WORDS and word not in LEXICON:
            if word not in WORLD_KEYWORDS and word not in NOETIC_KEYWORDS:
                if word not in VERB_TO_OPERATOR:
                    unknown_words.append(word)

    # Raise error in strict mode if unknown words detected
    if strict and unknown_words:
        # Build helpful error message with suggestions
        error_msg = f"Unknown tokens detected: {', '.join(unknown_words[:5])}"
        if len(unknown_words) > 5:
            error_msg += f" (and {len(unknown_words) - 5} more)"

        # Suggest checking lexicon
        error_msg += "\n\nValid token categories:"
        error_msg += "\n  - Words in LEXICON (e.g., 'woman', 'man', 'love', 'fear', 'power')"
        error_msg += "\n  - World keywords (e.g., 'spiritual', 'mental', 'emotional', 'physical')"
        error_msg += "\n  - Noetic keywords (e.g., 'mind', 'positive', 'negative', 'female', 'male')"
        error_msg += "\n  - Verbs/operators (e.g., 'caused', 'combined', 'opposed')"
        error_msg += "\n\nUse --lenient flag to allow unknown tokens with warnings."

        raise ValueError(error_msg)

    # Step 4: Role assignment
    entities = assign_roles(entities, story)

    # Step 5: Operator extraction
    operators = extract_operators(tokens)

    # Step 6: Temporal/causal ordering
    ordered_entities, ops = order_by_causality(entities, operators, story)

    # Step 7: Build expression
    elements = []
    element_refs = []

    for entity in ordered_entities:
        code = f"{entity.world}{entity.noetic}"
        elements.append(code)
        element_refs.append(ElementRef(
            world=entity.world,
            noetic=entity.noetic,
            sense=entity.sense,
        ))

    # Build tokens list
    tks_tokens = []
    for i, elem in enumerate(elements):
        tks_tokens.append(TKSToken.element(elem))
        if i < len(ops):
            try:
                tks_tokens.append(TKSToken.operator(ops[i]))
            except ValueError:
                # Invalid operator, use default
                ops[i] = "+T"
                tks_tokens.append(TKSToken.operator("+T"))

    # Calculate confidence based on unknown word ratio
    total_content_words = len([t for t in tokens if t[0] not in STOP_WORDS])
    if total_content_words > 0:
        confidence = 1.0 - (len(unknown_words) / total_content_words) * 0.5
    else:
        confidence = 0.5

    # Step 8: Detect foundation and create sub-foundation tags
    foundation_tags = []
    detected_fid = detect_foundation(story)

    if detected_fid is not None:
        # Validate foundation ID
        if not is_valid_foundation(detected_fid):
            warnings.append(f"Invalid foundation ID detected: {detected_fid}, must be in [1..7]")
        else:
            # Determine dominant world from entities
            world_counts = {}
            for entity in ordered_entities:
                if is_valid_world(entity.world):
                    world_counts[entity.world] = world_counts.get(entity.world, 0) + 1

            # Get dominant world
            dominant_world = None
            if world_counts:
                dominant_world = max(world_counts.items(), key=lambda x: x[1])[0]

            # Create foundation tag with world context if available
            if dominant_world and is_valid_world(dominant_world):
                foundation_tags.append((detected_fid, dominant_world))
            else:
                # No world context available
                foundation_tags.append((detected_fid, None))

    # Build expression
    expr = TKSExpression(
        elements=elements,
        ops=ops,
        foundations=foundation_tags,
        raw=story,
        tokens=tks_tokens,
        element_refs=element_refs,
        confidence=confidence,
    )

    return EncodeResult(
        expression=expr,
        success=True,
        warnings=warnings,
        unknown_words=unknown_words,
    )


# =============================================================================
# EQUATION PARSER
# =============================================================================

def parse_equation(equation: str, strict: bool = True) -> TKSExpression:
    """
    Parse a TKS equation string into a TKSExpression.

    Accepts formats:
        - "B5,+T,D3,-T,C8" (comma-separated)
        - "B5 +T D3 -T C8" (space-separated)
        - "B5 -> D3 +T C8" (with causal arrows)
        - "C3 *T C3" (multiplication/intensification)
        - "B2 /T B3" (division/conflict)
        - "C3 o D7 o D8" (sequential composition)

    Args:
        equation: TKS equation string
        strict: If True (default), raise ValueError for unknown operators.
                If False, skip unknown operators with warning.

    Returns:
        TKSExpression

    Raises:
        ValueError: If strict=True and unknown operators are detected
    """
    # Normalize separators - handle all canonical operators
    eq = equation.replace(",", " ")
    eq = eq.replace("->", " -> ")
    eq = eq.replace("<-", " <- ")
    # Handle *T and /T carefully to avoid splitting too much
    eq = eq.replace("*T", " *T ")
    eq = eq.replace("/T", " /T ")
    eq = eq.replace("+T", " +T ")
    eq = eq.replace("-T", " -T ")
    # Handle composition operator 'o' - only if standalone
    eq = re.sub(r'\bo\b', ' o ', eq)

    tokens = eq.split()

    elements = []
    ops = []
    element_refs = []
    unknown_ops = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Check if it's an operator
        if token in ALLOWED_OPS:
            ops.append(token)
        elif (token in {"+", "-", "*", "/", "o"} or
              token.endswith("T") or
              (len(token) == 2 and token[0] in "+-*/" and token[1].isalpha())):
            # Potential operator but not in ALLOWED_OPS
            # Matches: single chars (+,-,*,/,o), anything ending in T, or 2-char like +X
            unknown_ops.append(token)
            if strict:
                # Build helpful error message
                valid_ops = sorted(ALLOWED_OPS)
                error_msg = f"Unknown operator '{token}' detected."
                error_msg += f"\n\nValid operators: {', '.join(valid_ops)}"
                error_msg += "\n  - '+T' (combination/addition)"
                error_msg += "\n  - '-T' (subtraction/negation)"
                error_msg += "\n  - '->' (causal forward)"
                error_msg += "\n  - '<-' (causal reverse)"
                error_msg += "\n  - '*T' (intensification/multiplication)"
                error_msg += "\n  - '/T' (conflict/division)"
                error_msg += "\n  - 'o' (sequential composition)"
                error_msg += "\n\nUse --lenient flag to skip unknown operators with warnings."
                raise ValueError(error_msg)
            # In non-strict mode, skip this operator (don't add to ops list)
            continue
        # Check if it's an element (starts with A/B/C/D followed by digit)
        elif len(token) >= 2 and token[0] in "ABCD":
            # Use ElementRef.from_string to parse extended syntax
            # Supports: B8, B8.5, B8^5, B8_d5, B8^5_d5
            try:
                elem_ref = ElementRef.from_string(token)
                # For backward compatibility, use base code in elements list
                elements.append(elem_ref.code)
                element_refs.append(elem_ref)
            except (ValueError, IndexError) as e:
                # Invalid element syntax - provide helpful error in strict mode
                if strict:
                    error_msg = f"Invalid token '{token}'. "
                    error_msg += "\n\nExtended syntax formats:"
                    error_msg += "\n  - Basic: B8, D5 (world + noetic)"
                    error_msg += "\n  - Sense suffix: B8^5 (sense 5) or B8.5 (backward compatible)"
                    error_msg += "\n  - Foundation suffix: B8_d5 (foundation 5 in world D)"
                    error_msg += "\n  - Full extended: B8^5_d5 (sense 5, foundation 5 in world D)"
                    error_msg += "\n\nValid ranges:"
                    error_msg += "\n  - Worlds: A/B/C/D only"
                    error_msg += "\n  - Noetics: 1-10"
                    error_msg += "\n  - Foundations: 1-7"
                    error_msg += "\n  - Foundation worlds: a/b/c/d (case-insensitive)"
                    error_msg += f"\n\nError details: {str(e)}"
                    raise ValueError(error_msg)
                # In non-strict mode, skip this token
                pass

    return TKSExpression(
        elements=elements,
        ops=ops,
        element_refs=element_refs,
        raw=equation,
    )
