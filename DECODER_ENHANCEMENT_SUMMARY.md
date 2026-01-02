# Decoder Enhancement Summary

## Overview
Updated `narrative/decoder.py` to use sense labels and sub-foundation labels when available, providing more precise and contextually rich natural language decoding of TKS expressions.

## Changes Made

### 1. Import New Constants
Added imports for sub-foundation support:
```python
from .constants import (
    # ... existing imports ...
    SUBFOUND_MAP,
    get_subfound_label,
)
```

### 2. Added Helper Function: `get_foundation_context()`
Created a new helper function to generate contextual descriptions for foundations with optional world specification:

```python
def get_foundation_context(fid: int, world: Optional[str] = None) -> str:
    """
    Get contextual description for a foundation with optional world.

    Args:
        fid: Foundation ID (1-7)
        world: Optional world letter (A/B/C/D)

    Returns:
        Contextual phrase like "in emotional relationship context"
    """
```

**Behavior:**
- When world is provided: Uses sub-foundation label from `SUBFOUND_MAP` (e.g., "in emotional relationship context" for F4, C)
- When world is None: Uses foundation name only (e.g., "in companionship context")
- Fallback: Combines foundation + world names if sub-foundation label not found

### 3. Enhanced `get_element_label()` Function
The function already correctly implements sense label lookup:
- First tries to find specific sense label (e.g., "D5.1", "B5.2")
- Falls back to default element label if sense not specified
- Maintains backward compatibility

**Note:** This function was already working correctly, so only a clarifying comment was added.

### 4. Updated `decode_story_full()` Function
Enhanced to include sub-foundation context when available:

**New logic:**
1. Extracts foundation information from `expr.foundations` if available
2. Uses first foundation tag (most expressions have one main context)
3. Generates foundation context using `get_foundation_context()`
4. Appends context to decoded story in appropriate places

**Examples:**
- Single element: `"There is love in emotional relationship context."`
- Two elements: `"A woman together with a man, in physical companionship context."`
- Multiple elements: Adds context at end of narrative

## Sub-Foundation Labels Supported

The system now recognizes all 28 sub-foundations (7 foundations × 4 worlds):

### Foundation 1: Unity
- (1, "A"): Spiritual union
- (1, "B"): Mental unity
- (1, "C"): Emotional unity
- (1, "D"): Physical unity

### Foundation 2: Wisdom
- (2, "A"): Spiritual wisdom
- (2, "B"): Intellectual wisdom
- (2, "C"): Intuitive wisdom
- (2, "D"): Practical wisdom

### Foundation 3: Life/Health
- (3, "A"): Spiritual vitality
- (3, "B"): Mental health
- (3, "C"): Emotional health
- (3, "D"): Physical health

### Foundation 4: Companionship
- (4, "A"): Soul connection
- (4, "B"): Intellectual partnership
- (4, "C"): Emotional relationship
- (4, "D"): Physical companionship

### Foundation 5: Power
- (5, "A"): Spiritual authority
- (5, "B"): Intellectual power
- (5, "C"): Emotional influence
- (5, "D"): Material power

### Foundation 6: Material
- (6, "A"): Spiritual abundance
- (6, "B"): Ideas about wealth
- (6, "C"): Feelings about money
- (6, "D"): Physical resources

### Foundation 7: Lust/Creation
- (7, "A"): Creative spirit
- (7, "B"): Creative ideas
- (7, "C"): Desire/passion
- (7, "D"): Physical creation

## Sense Labels Supported

The system uses specific sense labels from `SENSE_LABELS` constant:

### Examples:
- **D5.1**: "a woman" (vs D5.2: "receptacle")
- **D6.1**: "a man" (vs D6.2: "structure")
- **B5.1**: "learning" (vs B5.2: "accumulated knowledge")
- **C2.1**: "joy" (vs C2.3: "love")
- **D8.1**: "physical trigger" (vs D8.3: "material authority")
- **D3.1**: "illness" (vs D3.2: "material chaos")

## Backward Compatibility

All changes are **backward compatible**:
- If no sense is specified, defaults to default element labels
- If no foundation tag is present, no context is added
- Existing decoding behavior is preserved when new features are not used

## Testing

Created comprehensive test suite in `test_decoder_enhancements.py` covering:
1. Sense label usage for various elements
2. Sub-foundation label usage across different foundations and worlds
3. Foundation context helper function
4. Combined sense labels + sub-foundation labels

All tests pass successfully, demonstrating:
- Correct sense label lookup (e.g., D5.1 → "a woman" vs D5.2 → "receptacle")
- Correct sub-foundation context (e.g., F4,C → "in emotional relationship context")
- Proper integration in narrative output

## Usage Example

```python
from narrative.decoder import DecodeStory
from narrative.types import TKSExpression, ElementRef

# Example: Woman + Man in Physical Companionship context
expr = TKSExpression(
    elements=["D5", "D6"],
    ops=["+T"],
    element_refs=[
        ElementRef("D", 5, 1),  # D5.1 = "a woman"
        ElementRef("D", 6, 1)   # D6.1 = "a man"
    ],
    foundations=[(4, "D")]  # F4, D = Physical companionship
)

story = DecodeStory(expr)
# Output: "A woman together with a man, in physical companionship context."
```

## Files Modified
- `narrative/decoder.py` - Main decoder implementation

## Files Created
- `test_decoder_enhancements.py` - Test suite
- `DECODER_ENHANCEMENT_SUMMARY.md` - This document
