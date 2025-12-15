"""
Canonical TKS Validator

Validates LLM responses against TKS canonical rules.
Key principle: LLM consensus does NOT override TKS canon.

The validator checks:
    1. Correct noetic naming (e.g., N6 = "Male" NOT "MEL")
    2. Valid element structure (World + Noetic)
    3. Involution pair relationships
    4. RPM component assignments
    5. Foundation correspondences
    6. World hierarchy consistency
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum


# ==============================================================================
# CANONICAL CONSTANTS
# ==============================================================================

# The 10 Noetics with CANONICAL names
NOETICS = {
    1: "Mind",
    2: "Positive",
    3: "Negative",
    4: "Vibration",
    5: "Female",
    6: "Male",      # CANONICAL: NOT "MEL", NOT "Masculine"
    7: "Rhythm",
    8: "Cause",     # Also "Above"
    9: "Effect",    # Also "Below"
    10: "Idea"
}

# Alternative names that are acceptable
NOETIC_ALIASES = {
    1: ["Mind", "Mentalism"],
    2: ["Positive", "Polarity+"],
    3: ["Negative", "Polarity-"],
    4: ["Vibration"],
    5: ["Female", "Feminine"],
    6: ["Male", "Masculine"],  # But NOT "MEL"
    7: ["Rhythm"],
    8: ["Cause", "Above", "Causation"],
    9: ["Effect", "Below", "Correspondence"],
    10: ["Idea", "Divine Idea", "All"]
}

# FORBIDDEN names (common errors)
FORBIDDEN_NOETIC_NAMES = {
    "MEL": "Use 'Male' for N6",
    "Mel": "Use 'Male' for N6",
    "mel": "Use 'Male' for N6",
}

# The 4 Canonical Worlds (A, B, C, D only)
# NOTE: Only A, B, C, D are canonical TKS worlds.
# Any other world letters (Y, Z, etc.) are non-canonical and rejected.
WORLDS = {
    'A': "Spiritual",   # Spiritual World
    'B': "Mental",      # Mental World
    'C': "Emotional",   # Emotional World
    'D': "Physical",    # Physical World
}

WORLD_HIERARCHY = ['A', 'B', 'C', 'D']  # Lowest to highest

# Non-canonical world letters (for explicit rejection)
# Note: 'N' is excluded because it's used for noetic notation (N1-N10)
NON_CANONICAL_WORLDS = {'Y', 'Z', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'}

# 7 Foundations (canonical)
FOUNDATIONS = [
    "Unity",
    "Wisdom",
    "Life",
    "Companionship",
    "Power",
    "Material",
    "Lust"
]

# Noetic to Foundation mappings (canonical)
NOETIC_TO_FOUNDATIONS = {
    10: {"Unity", "Material"},        # Idea
    1: {"Wisdom"},                    # Mind
    4: {"Wisdom", "Life", "Material"},# Vibration
    6: {"Wisdom", "Power", "Lust"},   # Male
    7: {"Wisdom", "Lust"},            # Rhythm
    2: {"Companionship"},             # Positive
    5: {"Companionship", "Lust"},     # Female
    8: {"Power"},                     # Cause
    9: {"Material"},                  # Effect
}

# Involution pairs: compose to ~N10 (Idea)
INVOLUTION_PAIRS = [(2, 3), (5, 6), (8, 9)]

# RPM groupings (canonical)
RPM_MAPPINGS = {
    "desire": {2, 3},           # Positive + Negative
    "wisdom": {1, 4, 5, 6, 7},  # Mind, Vibration, Female, Male, Rhythm
    "power": {8, 9}             # Cause + Effect
}


# ==============================================================================
# CANONICAL RULES
# ==============================================================================

CANONICAL_RULES = {
    "noetic_naming": {
        "description": "Noetics must use canonical names",
        "severity": "error",
        "details": "N6 is 'Male' (NOT 'MEL'). See NOETICS constant."
    },
    "element_structure": {
        "description": "Elements must be World + Noetic (e.g., B4, C10)",
        "severity": "error",
        "details": "Valid worlds: A, B, C, D only. Valid noetics: 1-10."
    },
    "involution_composition": {
        "description": "Involution pairs compose to approximately N10 (Idea)",
        "severity": "warning",
        "details": "Pairs: (2,3), (5,6), (8,9) -> N10"
    },
    "rpm_consistency": {
        "description": "RPM components must derive from correct noetics",
        "severity": "error",
        "details": "Desire=N2,N3; Wisdom=N1,N4,N5,N6,N7; Power=N8,N9"
    },
    "foundation_correspondence": {
        "description": "Foundations must correspond to correct noetics",
        "severity": "warning",
        "details": "Canonical foundations: Unity, Wisdom, Life, Companionship, Power, Material, Lust"
    },
    "world_consistency": {
        "description": "World references must be valid and consistent",
        "severity": "error",
        "details": "Four canonical worlds: Spiritual (A), Mental (B), Emotional (C), Physical (D)"
    },
    "non_canonical_world": {
        "description": "Non-canonical world letters are rejected",
        "severity": "error",
        "details": "Only A, B, C, D are valid. Y, Z, and others are non-canonical."
    },
    "no_invention": {
        "description": "Do not invent new noetics, worlds, or foundations",
        "severity": "error",
        "details": "Only 10 noetics, 4 worlds (A,B,C,D), 7 foundations exist"
    }
}


# ==============================================================================
# VALIDATION RESULT
# ==============================================================================

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found in LLM response."""
    rule: str
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of canonical validation."""
    is_valid: bool
    canon_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(
        self,
        rule: str,
        severity: str,
        message: str,
        location: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            rule=rule,
            severity=ValidationSeverity(severity),
            message=message,
            location=location,
            suggestion=suggestion
        ))


# ==============================================================================
# CANONICAL VALIDATOR
# ==============================================================================

class CanonicalValidator:
    """
    Validates LLM responses against TKS canonical rules.

    Key principle: LLM consensus does NOT override TKS canon.
    If an LLM makes a canonical error (e.g., calls N6 "MEL"),
    that response is marked invalid regardless of how many
    other LLMs agree with it.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, any error makes response invalid.
                        If False, warnings are tolerated.
        """
        self.strict_mode = strict_mode

        # Compile regex patterns
        # Only A, B, C, D are canonical worlds
        self._element_pattern = re.compile(r'\b([ABCD])(\d{1,2})\b')
        # Pattern to detect non-canonical world letters (for rejection)
        # Note: N is excluded because it's used for noetic notation (N1-N10)
        self._noncanonical_element_pattern = re.compile(r'\b([EFGHIJKLMOPQRSTUVWXYZ])(\d{1,2})\b')
        self._noetic_pattern = re.compile(r'\b(?:N|noetic[s]?)\s*(\d{1,2})\b', re.I)

    def validate(self, text: str, context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate LLM response text against TKS canon.

        Args:
            text: LLM response text to validate
            context: Optional context (e.g., expected elements, task type)

        Returns:
            ValidationResult with issues and canon score
        """
        result = ValidationResult(is_valid=True, canon_score=1.0)
        context = context or {}

        # Run all validation checks
        self._check_forbidden_names(text, result)
        self._check_noetic_naming(text, result)
        self._check_element_structure(text, result)
        self._check_world_references(text, result)
        self._check_rpm_references(text, result)
        self._check_foundation_references(text, result)
        self._check_involution_claims(text, result)
        self._check_invented_concepts(text, result)

        # Context-specific validation
        if "expected_elements" in context:
            self._check_expected_elements(text, context["expected_elements"], result)

        if "expected_rpm" in context:
            self._check_expected_rpm(text, context["expected_rpm"], result)

        # Compute final validity and score
        self._compute_final_score(result)

        return result

    def _check_forbidden_names(self, text: str, result: ValidationResult):
        """Check for forbidden noetic names (e.g., 'MEL')."""
        for forbidden, correction in FORBIDDEN_NOETIC_NAMES.items():
            if forbidden in text:
                result.add_issue(
                    rule="noetic_naming",
                    severity="error",
                    message=f"Forbidden term '{forbidden}' found",
                    location=f"Contains '{forbidden}'",
                    suggestion=correction
                )

    def _check_noetic_naming(self, text: str, result: ValidationResult):
        """Check that noetics are named correctly."""
        # Find noetic references
        for match in self._noetic_pattern.finditer(text):
            noetic_num = int(match.group(1))
            if noetic_num < 1 or noetic_num > 10:
                result.add_issue(
                    rule="noetic_naming",
                    severity="error",
                    message=f"Invalid noetic number: {noetic_num}",
                    location=match.group(0),
                    suggestion="Noetics are numbered 1-10"
                )

    def _check_element_structure(self, text: str, result: ValidationResult):
        """Check that elements follow World+Noetic structure."""
        # First check for non-canonical world letters (Y, Z, etc.)
        for match in self._noncanonical_element_pattern.finditer(text):
            world = match.group(1)
            noetic = int(match.group(2))
            result.add_issue(
                rule="non_canonical_world",
                severity="error",
                message=f"Non-canonical world '{world}' in element '{match.group(0)}'",
                location=match.group(0),
                suggestion=f"Only canonical worlds A, B, C, D are valid. Replace with A{noetic}, B{noetic}, C{noetic}, or D{noetic}"
            )

        # Now check canonical elements
        for match in self._element_pattern.finditer(text):
            world = match.group(1)
            noetic = int(match.group(2))

            if world not in WORLDS:
                result.add_issue(
                    rule="element_structure",
                    severity="error",
                    message=f"Invalid world '{world}' in element",
                    location=match.group(0),
                    suggestion=f"Valid worlds: {', '.join(WORLDS.keys())}"
                )

            if noetic < 1 or noetic > 10:
                result.add_issue(
                    rule="element_structure",
                    severity="error",
                    message=f"Invalid noetic {noetic} in element",
                    location=match.group(0),
                    suggestion="Noetics are numbered 1-10"
                )

    def _check_world_references(self, text: str, result: ValidationResult):
        """Check world references are valid."""
        text_lower = text.lower()

        # Check for valid world names (A=Spiritual, B=Mental, C=Emotional, D=Physical)
        valid_world_names = set(w.lower() for w in WORLDS.values())

        # Check for NON-CANONICAL world names (Yetzirah, Atziluth are not canonical)
        non_canonical_world_names = re.findall(r'\b(yetzirah|atziluth|atzilut)\b', text_lower)
        for world in non_canonical_world_names:
            result.add_issue(
                rule="non_canonical_world",
                severity="error",
                message=f"Non-canonical world name '{world}' found",
                location=world,
                suggestion="Canonical worlds are: Spiritual (A), Mental (B), Emotional (C), Physical (D)"
            )

    def _check_rpm_references(self, text: str, result: ValidationResult):
        """Check RPM references are consistent with canonical mappings."""
        text_lower = text.lower()

        # Check for RPM component mentions
        rpm_patterns = {
            "desire": r'\bdesire\b.*\b(?:N|noetic)\s*(\d+)',
            "wisdom": r'\bwisdom\b.*\b(?:N|noetic)\s*(\d+)',
            "power": r'\bpower\b.*\b(?:N|noetic)\s*(\d+)'
        }

        for component, pattern in rpm_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                noetic_num = int(match.group(1))
                expected = RPM_MAPPINGS[component]
                if noetic_num not in expected:
                    result.add_issue(
                        rule="rpm_consistency",
                        severity="warning",
                        message=f"N{noetic_num} is not canonically associated with {component}",
                        location=match.group(0),
                        suggestion=f"{component.title()} noetics: {sorted(expected)}"
                    )

    def _check_foundation_references(self, text: str, result: ValidationResult):
        """Check foundation references are valid."""
        text_lower = text.lower()

        valid_foundations = set(f.lower() for f in FOUNDATIONS)

        # Look for foundation mentions
        for foundation in valid_foundations:
            if foundation in text_lower:
                # This is fine - foundation exists
                pass

        # Check for invented foundations
        potential_foundations = re.findall(r'\bfoundation of (\w+)\b', text_lower)
        for pf in potential_foundations:
            if pf not in valid_foundations:
                result.add_issue(
                    rule="foundation_correspondence",
                    severity="warning",
                    message=f"Unknown foundation reference: '{pf}'",
                    suggestion=f"Valid foundations: {FOUNDATIONS}"
                )

    def _check_involution_claims(self, text: str, result: ValidationResult):
        """Check involution pair claims are correct."""
        text_lower = text.lower()

        # Look for involution claims
        involution_pattern = r'involution.*?(?:N|noetic)\s*(\d+).*?(?:N|noetic)\s*(\d+)'
        matches = re.finditer(involution_pattern, text_lower)

        for match in matches:
            n1, n2 = int(match.group(1)), int(match.group(2))
            pair = tuple(sorted([n1, n2]))

            valid_pairs = [tuple(sorted(p)) for p in INVOLUTION_PAIRS]
            if pair not in valid_pairs:
                result.add_issue(
                    rule="involution_composition",
                    severity="warning",
                    message=f"({n1}, {n2}) is not a canonical involution pair",
                    location=match.group(0),
                    suggestion=f"Canonical pairs: {INVOLUTION_PAIRS}"
                )

    def _check_invented_concepts(self, text: str, result: ValidationResult):
        """Check for invented noetics, worlds, or concepts."""
        # Check for invented noetics (N11+)
        invented_noetics = re.findall(r'\b(?:N|noetic)\s*(1[1-9]|[2-9]\d+)\b', text, re.I)
        for inv in invented_noetics:
            result.add_issue(
                rule="no_invention",
                severity="error",
                message=f"Invented noetic N{inv} referenced",
                suggestion="Only noetics 1-10 exist in TKS"
            )

        # Check for fifth world references
        fifth_world_patterns = [
            r'\bfifth world\b',
            r'\b5th world\b',
            r'\bworld 5\b'
        ]
        for pattern in fifth_world_patterns:
            if re.search(pattern, text, re.I):
                result.add_issue(
                    rule="no_invention",
                    severity="error",
                    message="Reference to non-existent fifth world",
                    suggestion="Only 4 canonical worlds exist: Spiritual (A), Mental (B), Emotional (C), Physical (D)"
                )

    def _check_expected_elements(
        self,
        text: str,
        expected: List[str],
        result: ValidationResult
    ):
        """Check that expected elements are mentioned."""
        for element in expected:
            if element not in text:
                result.add_issue(
                    rule="element_structure",
                    severity="warning",
                    message=f"Expected element {element} not found in response"
                )

    def _check_expected_rpm(
        self,
        text: str,
        expected: Dict[str, float],
        result: ValidationResult
    ):
        """Check that RPM values mentioned match expected."""
        # This is a soft check - just ensure the dominant component is mentioned
        dominant = max(expected, key=expected.get)
        if dominant.lower() not in text.lower():
            result.add_issue(
                rule="rpm_consistency",
                severity="info",
                message=f"Expected dominant RPM component '{dominant}' not emphasized"
            )

    def _compute_final_score(self, result: ValidationResult):
        """Compute final validity and canon score."""
        # Count issues by severity
        errors = result.error_count
        warnings = result.warning_count

        # Compute score
        # Each error reduces score by 0.2, each warning by 0.05
        score = 1.0 - (errors * 0.2) - (warnings * 0.05)
        result.canon_score = max(0.0, min(1.0, score))

        # Determine validity
        if self.strict_mode:
            result.is_valid = errors == 0
        else:
            result.is_valid = result.canon_score >= 0.5

        # Store counts in metadata
        result.metadata["error_count"] = errors
        result.metadata["warning_count"] = warnings


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def validate_response(text: str, strict: bool = True) -> ValidationResult:
    """Convenience function to validate a response."""
    validator = CanonicalValidator(strict_mode=strict)
    return validator.validate(text)


def is_canonical(text: str) -> bool:
    """Quick check if text passes canonical validation."""
    result = validate_response(text, strict=True)
    return result.is_valid


def get_canon_score(text: str) -> float:
    """Get canon score for text (0.0 to 1.0)."""
    result = validate_response(text, strict=False)
    return result.canon_score
