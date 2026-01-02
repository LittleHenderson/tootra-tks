"""
TKS Wrapper - Cognitive Lens for Large Language Models

This module wraps general-purpose LLMs (Claude, Gemini) with TKS
processing layers, enabling TKS-aligned analysis of any input.

Architecture:
    User Input → TKS Pre-Processor → Base LLM → TKS Post-Processor → Output

The wrapper enforces canonical TKS rules while leveraging the full
reasoning capabilities of billion-parameter models.

Usage:
    wrapper = TKSWrapper(provider="anthropic", api_key="...")
    result = wrapper.analyze("I'm torn between career ambition and family time")
    print(result.equation)      # B7 +T D3 -T A5
    print(result.interpretation) # Full TKS analysis
    print(result.guidance)      # Actionable TKS-based advice

Supported Providers:
    - anthropic (Claude)
    - google (Gemini)
    - mock (for testing)

Author: TKS-LLM Project
Date: 2024
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

# Import TKS validator
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from teacher.validator import (
        CanonicalValidator,
        ValidationResult,
        NOETICS,
        WORLDS,
        FOUNDATIONS,
        NOETIC_TO_FOUNDATIONS
    )
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False
    print("Warning: TKS validator not found. Validation will be limited.")

# Load API keys from config
try:
    from config.tks_keys import (
        ANTHROPIC_API_KEY,
        GOOGLE_API_KEY,
        CLAUDE_MODEL,
        GEMINI_MODEL
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    ANTHROPIC_API_KEY = None
    GOOGLE_API_KEY = None
    CLAUDE_MODEL = "claude-opus-4-5-20251101"
    GEMINI_MODEL = "gemini-2.0-flash"


# ==============================================================================
# TKS RULEBOOK (Complete System Prompt)
# ==============================================================================

TKS_RULEBOOK = """
# TOOTRA KABBALISTIC SYSTEM (TKS) - COMPLETE CANONICAL RULEBOOK

You are operating as a TKS-enhanced reasoning system. ALL responses must be
grounded in canonical TKS principles. You have full world knowledge but must
filter all analysis through the TKS framework.

## CORE ONTOLOGY

### The 10 Noetics (N1-N10)
These are the fundamental building blocks of reality in TKS:

| ID | Name      | Meaning                    | RPM Category |
|----|-----------|----------------------------|--------------|
| N1 | Mind      | Consciousness, awareness   | Desire       |
| N2 | Positive  | Attraction, addition       | -            |
| N3 | Negative  | Repulsion, subtraction     | -            |
| N4 | Vibration | Movement, change, energy   | Desire       |
| N5 | Female    | Receptive, nurturing       | Wisdom       |
| N6 | Male      | Active, projecting         | Wisdom       |
| N7 | Rhythm    | Cycles, patterns, timing   | Desire       |
| N8 | Cause     | Origin, above, source      | Power        |
| N9 | Effect    | Result, below, manifestation | Power      |
| N10| Idea      | Divine concept, unity, All | -            |

CRITICAL: N6 is "Male" - NEVER use "MEL" or other variations.

### The 4 Worlds (A, B, C, D)
Reality manifests across four hierarchical planes:

| World | Name      | Domain                     | Examples           |
|-------|-----------|----------------------------|--------------------|
| A     | Spiritual | Divine, archetypal, source | Core beliefs, soul |
| B     | Mental    | Thought, intellect, reason | Ideas, decisions   |
| C     | Emotional | Feelings, astral, desire   | Love, fear, joy    |
| D     | Physical  | Material, action, manifest | Body, money, work  |

CRITICAL: ONLY A, B, C, D are valid. Reject Y, Z, E, F, etc.

### Elements
Elements combine World + Noetic: A1, B7, C3, D10, etc.
- Valid range: [A-D][1-10]
- Examples: A1 (Spiritual Mind), D6 (Physical Male), C5 (Emotional Female)

### The 7 Foundations
Life goals/themes that noetics map to:

1. Unity - Wholeness, oneness (N10)
2. Wisdom - Knowledge, understanding (N1, N4, N6, N7)
3. Life - Vitality, existence (N4)
4. Companionship - Connection, relationship (N2, N5)
5. Power - Agency, control (N6, N8)
6. Material - Physical abundance (N4, N9, N10)
7. Lust - Desire, passion, achievement (N5, N6, N7)

### Operators
TKS equations use these operators:

| Operator | Name           | Meaning                        |
|----------|----------------|--------------------------------|
| +T       | Transformation | Adding/amplifying energy       |
| -T       | Diminishment   | Removing/reducing energy       |
| +        | Simple add     | Basic combination              |
| -        | Simple sub     | Basic removal                  |
| ->       | Flow           | Direction of influence         |
| <-       | Receive        | Incoming influence             |
| *T       | Multiply       | Intensification                |
| /T       | Divide         | Distribution                   |
| o        | Compose        | Deep integration               |

### RPM Protocol (Desire/Wisdom/Power)
Every situation can be analyzed through RPM:

- **Desire (D)**: What is wanted? (N1, N4, N7 - Mind, Vibration, Rhythm)
- **Wisdom (W)**: What is understood? (N5, N6 - Female, Male principles)
- **Power (P)**: What can be done? (N8, N9 - Cause, Effect)

### Involution Pairs
Complementary noetics that compose to N10 (Idea/Unity):
- (N2, N3): Positive + Negative
- (N5, N6): Female + Male
- (N8, N9): Cause + Effect

## RESPONSE FORMAT

For EVERY analysis, provide:

1. **TKS EQUATION**: The canonical equation representing the situation
   Format: Element Op Element Op Element...
   Example: B7 +T D3 -T A5

2. **ELEMENT BREAKDOWN**: Explain each element's meaning in context

3. **RPM ANALYSIS**:
   - Desire: [What is being sought]
   - Wisdom: [What understanding is present/needed]
   - Power: [What agency exists]

4. **FOUNDATION MAPPING**: Which Foundation(s) are active

5. **ATTRACTOR STATE**: Where the current trajectory leads

6. **GUIDANCE**: TKS-aligned recommendations (optional operations to add)

## VALIDATION RULES

Before outputting, verify:
- [ ] All elements use only A, B, C, D worlds
- [ ] All noetics are 1-10
- [ ] N6 is labeled "Male" not "MEL"
- [ ] Operators are from the canonical set
- [ ] RPM categories are correctly assigned
- [ ] Foundation mappings follow canonical rules

## EXAMPLES

Input: "I want to start a business but I'm scared of failure"

TKS Analysis:
```
EQUATION: D8 +T B1 -T C3

ELEMENT BREAKDOWN:
- D8 (Physical Cause): The desire to initiate action in material world
- B1 (Mental Mind): Entrepreneurial thinking and planning
- C3 (Emotional Negative): Fear creating resistance

RPM ANALYSIS:
- Desire: Achievement, material success (D8 activates Power foundation)
- Wisdom: Mental planning present (B1) but unbalanced
- Power: Blocked by emotional resistance (C3 subtracting)

FOUNDATION: Power (N8) conflicting with Emotional polarity

ATTRACTOR: Without intervention, oscillation between action and paralysis

GUIDANCE: Introduce +T A5 (Spiritual Female/receptive) to accept uncertainty,
or +T B6 (Mental Male/active) to strengthen decisive thinking.
```
"""


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class AnalysisMode(Enum):
    """Mode of TKS analysis."""
    FULL = "full"           # Complete analysis with all components
    EQUATION = "equation"   # Just extract the equation
    COACHING = "coaching"   # Focus on guidance
    DIAGNOSTIC = "diagnostic"  # Deep RPM analysis


@dataclass
class TKSElement:
    """A single TKS element (World + Noetic)."""
    world: str
    noetic: int

    @property
    def code(self) -> str:
        return f"{self.world}{self.noetic}"

    @property
    def world_name(self) -> str:
        return WORLDS.get(self.world, "Unknown") if HAS_VALIDATOR else self.world

    @property
    def noetic_name(self) -> str:
        return NOETICS.get(self.noetic, "Unknown") if HAS_VALIDATOR else str(self.noetic)

    def __str__(self) -> str:
        return f"{self.code} ({self.world_name} {self.noetic_name})"


@dataclass
class TKSEquationParsed:
    """Parsed TKS equation with elements and operators."""
    raw: str
    elements: List[TKSElement]
    operators: List[str]
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.raw


@dataclass
class RPMAnalysis:
    """RPM (Desire/Wisdom/Power) breakdown."""
    desire: str
    wisdom: str
    power: str
    desire_elements: List[str] = field(default_factory=list)
    wisdom_elements: List[str] = field(default_factory=list)
    power_elements: List[str] = field(default_factory=list)


@dataclass
class TKSAnalysisResult:
    """Complete result from TKS wrapper analysis."""

    # Core outputs
    equation: TKSEquationParsed
    interpretation: str
    guidance: str

    # Detailed analysis
    rpm: RPMAnalysis
    foundations: List[str]
    attractor: str

    # Validation
    is_canonical: bool
    validation_issues: List[str]
    canon_score: float

    # Metadata
    raw_response: str
    provider: str
    model: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "equation": self.equation.raw,
            "interpretation": self.interpretation,
            "guidance": self.guidance,
            "rpm": {
                "desire": self.rpm.desire,
                "wisdom": self.rpm.wisdom,
                "power": self.rpm.power
            },
            "foundations": self.foundations,
            "attractor": self.attractor,
            "is_canonical": self.is_canonical,
            "canon_score": self.canon_score,
            "validation_issues": self.validation_issues,
            "provider": self.provider,
            "model": self.model
        }

    def summary(self) -> str:
        """Return a concise summary."""
        status = "Canonical" if self.is_canonical else "Non-canonical"
        # Sanitize text to handle Unicode on Windows console
        def sanitize(text: str) -> str:
            if not text:
                return text
            # Replace common problematic Unicode with ASCII equivalents
            replacements = {
                '\u2014': '-',  # em dash
                '\u2013': '-',  # en dash
                '\u2018': "'", '\u2019': "'",  # smart quotes
                '\u201c': '"', '\u201d': '"',  # smart double quotes
                '\u2192': '->',  # arrow right
                '\u2190': '<-',  # arrow left
                '\u2194': '<->',  # bidirectional arrow
                '\u2022': '*',  # bullet
                '\u00b7': '*',  # middle dot
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            # Encode to ASCII with replacement for anything else
            return text.encode('ascii', 'replace').decode('ascii')

        return f"""
TKS Analysis [{status}]
{'=' * 50}

EQUATION: {sanitize(self.equation.raw)}

RPM BREAKDOWN:
  Desire: {sanitize(self.rpm.desire)}
  Wisdom: {sanitize(self.rpm.wisdom)}
  Power:  {sanitize(self.rpm.power)}

FOUNDATIONS: {', '.join(self.foundations)}

ATTRACTOR: {sanitize(self.attractor)[:300]}{'...' if len(self.attractor) > 300 else ''}

GUIDANCE: {sanitize(self.guidance)[:500]}{'...' if len(self.guidance) > 500 else ''}
"""


# ==============================================================================
# EQUATION PARSER
# ==============================================================================

class TKSEquationParser:
    """Parse TKS equations from text."""

    ELEMENT_PATTERN = r'([ABCD])(\d{1,2})'
    OPERATOR_PATTERN = r'(\+T|-T|\+|-|->|<-|\*T|/T|o)'
    EQUATION_PATTERN = r'([ABCD]\d{1,2}(?:\s*(?:\+T|-T|\+|-|->|<-|\*T|/T|o)\s*[ABCD]\d{1,2})+)'

    def parse(self, text: str) -> Optional[TKSEquationParsed]:
        """Extract and parse equation from text."""
        # Find equation pattern
        match = re.search(self.EQUATION_PATTERN, text)
        if not match:
            # Try simpler pattern for single elements
            simple_match = re.search(r'([ABCD]\d{1,2})', text)
            if simple_match:
                raw = simple_match.group(1)
                element = self._parse_element(raw)
                if element:
                    return TKSEquationParsed(
                        raw=raw,
                        elements=[element],
                        operators=[],
                        is_valid=True
                    )
            return None

        raw = match.group(1)
        elements = []
        operators = []
        errors = []

        # Extract elements
        for m in re.finditer(self.ELEMENT_PATTERN, raw):
            elem = self._parse_element(m.group(0))
            if elem:
                elements.append(elem)
                # Validate
                if elem.world not in 'ABCD':
                    errors.append(f"Invalid world '{elem.world}' - must be A, B, C, or D")
                if not 1 <= elem.noetic <= 10:
                    errors.append(f"Invalid noetic {elem.noetic} - must be 1-10")

        # Extract operators
        for m in re.finditer(self.OPERATOR_PATTERN, raw):
            operators.append(m.group(1))

        return TKSEquationParsed(
            raw=raw,
            elements=elements,
            operators=operators,
            is_valid=len(errors) == 0,
            validation_errors=errors
        )

    def _parse_element(self, code: str) -> Optional[TKSElement]:
        """Parse a single element code."""
        match = re.match(r'([ABCD])(\d{1,2})', code)
        if match:
            return TKSElement(
                world=match.group(1),
                noetic=int(match.group(2))
            )
        return None


# ==============================================================================
# LLM PROVIDERS
# ==============================================================================

class LLMProviderBase:
    """Base class for LLM providers."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProviderBase):
    """Claude API provider (Anthropic)."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5-20251101"):
        self.api_key = api_key
        self.model = model
        self.name = "anthropic"

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text


class GeminiProvider(LLMProviderBase):
    """Gemini API provider (Google)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.name = "google"

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=model,
                system_instruction=None  # Will be set per-request
            )
        except ImportError:
            raise ImportError("google-generativeai package required: pip install google-generativeai")

    def complete(self, system: str, user: str) -> str:
        import google.generativeai as genai

        # Create model with system instruction
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system
        )

        response = model.generate_content(user)
        return response.text


class MockProvider(LLMProviderBase):
    """Mock provider for testing without API calls."""

    def __init__(self):
        self.name = "mock"
        self.model = "mock-tks"

    def complete(self, system: str, user: str) -> str:
        # Return a template TKS response
        return """
TKS Analysis:

EQUATION: B1 +T D4 -T C3

ELEMENT BREAKDOWN:
- B1 (Mental Mind): Active thinking and planning
- D4 (Physical Vibration): Energy and action in material world
- C3 (Emotional Negative): Resistance or fear

RPM ANALYSIS:
- Desire: Seeking change and progress through mental clarity
- Wisdom: Understanding present but needs balance
- Power: Capacity for action exists but emotionally blocked

FOUNDATION: Wisdom (N1, N4) with Material undertones

ATTRACTOR: Moving toward productive action if emotional block cleared

GUIDANCE: Consider adding +T A5 (Spiritual Female) to cultivate acceptance,
allowing the transformation to proceed without resistance.
"""


# ==============================================================================
# TKS WRAPPER (Main Class)
# ==============================================================================

class TKSWrapper:
    """
    Main TKS Wrapper class.

    Wraps a general-purpose LLM (Claude or Gemini) with TKS pre/post-processing
    to enable TKS-aligned analysis of any input.

    Supported providers:
        - "anthropic" or "claude": Uses Claude API
        - "google" or "gemini": Uses Gemini API
        - "mock": For testing without API calls
    """

    def __init__(
        self,
        provider: str = "mock",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        strict_validation: bool = True
    ):
        """
        Initialize TKS Wrapper.

        Args:
            provider: LLM provider ("anthropic"/"claude", "google"/"gemini", "mock")
            api_key: API key for the provider (or set env var)
            model: Model to use (provider-specific)
            strict_validation: Enforce strict canonical validation
        """
        self.strict_validation = strict_validation
        self.parser = TKSEquationParser()
        self.validator = CanonicalValidator() if HAS_VALIDATOR else None

        # Normalize provider name
        provider = provider.lower()

        # Initialize provider
        if provider in ["anthropic", "claude"]:
            self.provider = AnthropicProvider(
                api_key=api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"),
                model=model or CLAUDE_MODEL
            )
        elif provider in ["google", "gemini"]:
            self.provider = GeminiProvider(
                api_key=api_key or GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                model=model or GEMINI_MODEL
            )
        elif provider == "mock":
            self.provider = MockProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'gemini', or 'mock'")

    def analyze(
        self,
        user_input: str,
        mode: AnalysisMode = AnalysisMode.FULL,
        context: Optional[str] = None
    ) -> TKSAnalysisResult:
        """
        Analyze user input through TKS lens.

        Args:
            user_input: The situation/question to analyze
            mode: Analysis mode (full, equation, coaching, diagnostic)
            context: Optional additional context

        Returns:
            TKSAnalysisResult with equation, interpretation, and guidance
        """
        import time
        start_time = time.time()

        # Pre-process: Build the prompt
        prompt = self._build_prompt(user_input, mode, context)

        # Call LLM
        raw_response = self.provider.complete(
            system=TKS_RULEBOOK,
            user=prompt
        )

        # Post-process: Extract and validate
        result = self._process_response(raw_response, start_time)

        return result

    def _build_prompt(
        self,
        user_input: str,
        mode: AnalysisMode,
        context: Optional[str]
    ) -> str:
        """Build the user prompt based on mode."""

        mode_instructions = {
            AnalysisMode.FULL: "Provide a complete TKS analysis with equation, RPM breakdown, foundation mapping, and guidance.",
            AnalysisMode.EQUATION: "Extract the TKS equation that represents this situation. Focus on accuracy.",
            AnalysisMode.COACHING: "Provide TKS-based guidance and actionable recommendations.",
            AnalysisMode.DIAGNOSTIC: "Perform deep RPM analysis. Identify the core Desire, Wisdom gaps, and Power dynamics."
        }

        prompt = f"""Analyze the following through the TKS framework:

SITUATION:
{user_input}

{f'ADDITIONAL CONTEXT: {context}' if context else ''}

INSTRUCTIONS:
{mode_instructions[mode]}

Remember:
- Use ONLY canonical worlds (A, B, C, D)
- Use ONLY noetics 1-10
- N6 is "Male" (never "MEL")
- Provide the equation in format: Element Op Element Op Element...
"""
        return prompt

    def _process_response(
        self,
        raw_response: str,
        start_time: float
    ) -> TKSAnalysisResult:
        """Process and validate LLM response."""
        import time

        # Parse equation
        equation = self.parser.parse(raw_response)
        if equation is None:
            equation = TKSEquationParsed(
                raw="[No equation found]",
                elements=[],
                operators=[],
                is_valid=False,
                validation_errors=["Could not extract equation from response"]
            )

        # Validate if validator available
        validation_issues = equation.validation_errors.copy()
        canon_score = 1.0 if equation.is_valid else 0.0

        if self.validator and equation.is_valid:
            # Run through canonical validator
            for elem in equation.elements:
                if elem.world not in 'ABCD':
                    validation_issues.append(f"Non-canonical world: {elem.world}")
                    canon_score -= 0.2

        # Extract RPM analysis
        rpm = self._extract_rpm(raw_response)

        # Extract foundations
        foundations = self._extract_foundations(raw_response)

        # Extract attractor
        attractor = self._extract_section(raw_response, "ATTRACTOR", "No attractor identified")

        # Extract guidance
        guidance = self._extract_section(raw_response, "GUIDANCE", "No guidance provided")

        # Extract interpretation
        interpretation = self._extract_section(
            raw_response,
            "ELEMENT BREAKDOWN",
            raw_response[:500] if len(raw_response) > 500 else raw_response
        )

        processing_time = (time.time() - start_time) * 1000

        return TKSAnalysisResult(
            equation=equation,
            interpretation=interpretation,
            guidance=guidance,
            rpm=rpm,
            foundations=foundations,
            attractor=attractor,
            is_canonical=len(validation_issues) == 0 and equation.is_valid,
            validation_issues=validation_issues,
            canon_score=max(0, canon_score),
            raw_response=raw_response,
            provider=self.provider.name,
            model=getattr(self.provider, 'model', 'unknown'),
            processing_time_ms=processing_time
        )

    def _extract_rpm(self, text: str) -> RPMAnalysis:
        """Extract RPM analysis from response."""
        # Try multiple formats with priority order

        # Format 1: Full section headers like "### DESIRE (N1, N4, N7)"
        desire = self._extract_rpm_section(text, "DESIRE")
        wisdom = self._extract_rpm_section(text, "WISDOM")
        power = self._extract_rpm_section(text, "POWER")

        # Fallback: Simple line formats "Desire:", "- Desire:", "**Desire**:"
        if not desire:
            desire = self._extract_line(text, "Desire:") or self._extract_line(text, "- Desire:") or self._extract_line(text, "**Desire**:")
        if not wisdom:
            wisdom = self._extract_line(text, "Wisdom:") or self._extract_line(text, "- Wisdom:") or self._extract_line(text, "**Wisdom**:")
        if not power:
            power = self._extract_line(text, "Power:") or self._extract_line(text, "- Power:") or self._extract_line(text, "**Power**:")

        return RPMAnalysis(
            desire=desire or "Not specified",
            wisdom=wisdom or "Not specified",
            power=power or "Not specified"
        )

    def _extract_rpm_section(self, text: str, component: str) -> Optional[str]:
        """Extract RPM component from markdown section format."""
        # Match "### DESIRE (N1, N4, N7)" or "**DESIRE**" style headers
        pattern = rf'(?:###?\s*)?(?:\*\*)?{component}(?:\*\*)?(?:\s*\([^)]+\))?\s*\n([\s\S]+?)(?=\n###|\n\*\*[A-Z]+\*\*|\n---|\Z)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Extract bullet points - format: "- **What is wanted**: value"
            bullets = re.findall(r'[-*]\s*\*\*([^*]+)\*\*:\s*(.+?)(?=\n[-*]|\n\n|\Z)', content, re.DOTALL)
            if bullets:
                # Combine values (not labels like "What is wanted")
                values = [val.strip().replace('\n', ' ')[:100] for _, val in bullets[:3]]
                return "; ".join(values)
            # Fallback: look for simple "key: value" format
            simple_bullets = re.findall(r'[-*]\s*([^:\n]+):\s*(.+?)(?=\n|$)', content)
            if simple_bullets:
                return "; ".join([f"{k.strip()}: {v.strip()}" for k, v in simple_bullets[:2]])
            # Last fallback: first meaningful line
            lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('|') and not l.strip().startswith('-')]
            if lines:
                return lines[0][:200]
        return None

    def _extract_foundations(self, text: str) -> List[str]:
        """Extract foundation references from response."""
        foundations = []
        foundation_names = ["Unity", "Wisdom", "Life", "Companionship", "Power", "Material", "Lust"]

        for name in foundation_names:
            if name in text:
                foundations.append(name)

        return foundations if foundations else ["Not identified"]

    def _extract_section(self, text: str, header: str, default: str) -> str:
        """Extract a section from the response."""
        # Look for the header, then capture everything until the next major section
        # Major sections: RPM, EQUATION, ELEMENT, FOUNDATION, ATTRACTOR, GUIDANCE (all caps)
        # But NOT markdown headers like "### Immediate"
        pattern = rf'{header}[:\s]*(.+?)(?=\n(?:RPM|EQUATION|ELEMENT|FOUNDATION|ATTRACTOR|GUIDANCE)[:\s]|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Clean up but preserve meaningful content
            if content:
                return content
        # Fallback: try simpler extraction - get everything after header until double newline
        simple_pattern = rf'{header}[:\s]*([^\n]+(?:\n(?!\n)[^\n]*)*)'
        match = re.search(simple_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return default

    def _extract_line(self, text: str, prefix: str) -> Optional[str]:
        """Extract a single line value, handling various formats."""
        # Escape special regex characters in prefix
        escaped_prefix = re.escape(prefix)
        # Try to match the prefix followed by content until newline
        pattern = rf'{escaped_prefix}\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            # Remove trailing markdown artifacts
            result = re.sub(r'\s*\*\*$', '', result)
            return result if result else None
        return None

    def validate_equation(self, equation_str: str) -> Tuple[bool, List[str]]:
        """Validate an equation string against TKS canon."""
        equation = self.parser.parse(equation_str)
        if equation is None:
            return False, ["Could not parse equation"]
        return equation.is_valid, equation.validation_errors


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Command-line interface for TKS Wrapper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TKS Wrapper - Analyze any situation through the TKS framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with mock provider (no API key needed)
  python tks_wrapper.py "I want to start a business" --provider mock

  # Use Claude API
  python tks_wrapper.py "I'm stressed about work" --provider claude

  # Use Gemini API
  python tks_wrapper.py "Should I move to a new city?" --provider gemini

  # Interactive mode
  python tks_wrapper.py --interactive --provider claude

Environment Variables:
  ANTHROPIC_API_KEY  - For Claude provider
  GOOGLE_API_KEY     - For Gemini provider
  GEMINI_API_KEY     - Alternative for Gemini provider
"""
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Situation to analyze (or use --interactive)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["claude", "gemini", "mock"],
        default="mock",
        help="LLM provider: claude (Anthropic), gemini (Google), or mock (testing)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Specific model to use (e.g., claude-sonnet-4-20250514, gemini-1.5-pro)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "equation", "coaching", "diagnostic"],
        default="full",
        help="Analysis mode"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode - continuous conversation"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="API key (or use environment variable)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show raw LLM response for debugging"
    )

    args = parser.parse_args()

    # Initialize wrapper
    try:
        wrapper = TKSWrapper(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall required package:")
        if "anthropic" in str(e):
            print("  pip install anthropic")
        elif "google" in str(e):
            print("  pip install google-generativeai")
        return
    except Exception as e:
        print(f"Error initializing provider: {e}")
        return

    mode_map = {
        "full": AnalysisMode.FULL,
        "equation": AnalysisMode.EQUATION,
        "coaching": AnalysisMode.COACHING,
        "diagnostic": AnalysisMode.DIAGNOSTIC
    }

    if args.interactive:
        print("=" * 60)
        print("TKS Wrapper - Interactive Mode")
        print(f"Provider: {args.provider.upper()}")
        print("=" * 60)
        print("Enter situations to analyze. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue

                print("\nAnalyzing through TKS lens...")
                result = wrapper.analyze(user_input, mode_map[args.mode])

                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    print(result.summary())
                    if args.debug:
                        print("\n" + "=" * 60)
                        print("DEBUG: Raw LLM Response")
                        print("=" * 60)
                        print(result.raw_response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nGoodbye!")

    elif args.input:
        try:
            result = wrapper.analyze(args.input, mode_map[args.mode])

            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(result.summary())
                if args.debug:
                    print("\n" + "=" * 60)
                    print("DEBUG: Raw LLM Response")
                    print("=" * 60)
                    print(result.raw_response)
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
