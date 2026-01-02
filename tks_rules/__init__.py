"""
TKS Canonical Rules Package

This package provides the SINGLE SOURCE OF TRUTH for all TKS definitions.
All other modules should import from here rather than defining their own constants.

CANONICAL STRUCTURE:
    - 10 Noetics (nu_0 through nu_9) - 0-indexed
    - 4 Worlds (A, B, C, D) - cascade A -> B -> C -> D
    - 40 Elements (A1-D10) - 4 Worlds x 10 Noetics, notation: Letter+Digit
    - 7 Foundations (F1-F7)
    - 28 Sub-Foundations (1a-7d) - 7 Foundations x 4 Worlds, notation: Digit+Letter
    - 22 Acquisitions (D1-D7, W1-W7, P1-P7, A22)
    - 9 Operators (+, -, *, /, +T, -T, *T, /T, o, ->, <-)

RPM PROTOCOL (MVR):
    - Desire: {1, 4, 7} - Mind, Vibration, Rhythm (MVR)
    - Wisdom: {5, 6} - Female, Male
    - Power: {8, 9} - Cause, Effect

CRITICAL DISTINCTIONS:
    1. Elements use Capital+Digit (A1, B7, C3, D10)
    2. Sub-Foundations use Digit+Lowercase (1a, 3c, 7d)
    3. Noetics MODIFY other values; they do not participate in direct arithmetic

Usage:
    from tks_rules import (
        # Noetics
        NOETICS, NOETIC_NAMES, get_complement,
        # Worlds
        WORLDS, WORLD_LETTERS, world_distance,
        # Elements
        ELEMENTS, ELEMENT_CODES, validate_element,
        # Foundations
        FOUNDATIONS, SUB_FOUNDATIONS, SUB_FOUNDATION_CODES,
        # RPM (MVR Protocol)
        RPM_DESIRE, RPM_WISDOM, RPM_POWER, RPM_MAPPINGS,
        DESIRE_INDICES, WISDOM_INDICES, POWER_INDICES,
        # Operators
        OPERATORS, ALLOWED_OPS, validate_operator,
        # Acquisitions
        ACQUISITIONS, ALL_ACQUISITIONS,
        # Syntax
        parse_simple_equation, validate_expression_syntax,
    )
"""

__version__ = "1.0.0"

# =============================================================================
# NOETICS
# =============================================================================
from .noetics import (
    NoeticDefinition,
    NOETICS,
    NOETIC_NAMES,
    NOETIC_BY_NAME,
    INVOLUTION_PAIRS,
    SELF_DUAL_NOETICS,
    SELF_DUAL_MVR,
    SPECIAL_PAIR,
    MVR_FRACTAL,
    SENSES,
    get_complement,
    validate_noetic,
)

# =============================================================================
# WORLDS
# =============================================================================
from .worlds import (
    WorldDefinition,
    WORLDS,
    WORLD_LETTERS,
    WORLD_ORDINALS,
    WORLD_NAMES,
    WORLD_DOMAINS,
    WORLD_OFFSETS,
    WORLD_CASCADE_ORDER,
    SUBFOUNDATION_WORLD_LETTERS,
    world_distance,
    validate_world,
    get_world_by_ordinal,
)

# =============================================================================
# ELEMENTS (40)
# =============================================================================
from .elements import (
    ElementDefinition,
    ELEMENTS,
    ELEMENT_CODES,
    ELEMENT_BY_INDEX,
    ELEMENTS_BY_WORLD,
    ELEMENT_PATTERN,
    element_num_to_noetic_idx,
    noetic_idx_to_element_num,
    parse_element,
    validate_element,
    element_to_index,
    index_to_element,
    same_world,
    same_noetic,
    get_element_noetic_idx,
)

# =============================================================================
# FOUNDATIONS (7) AND SUB-FOUNDATIONS (28)
# =============================================================================
from .foundations import (
    FoundationDefinition,
    SubFoundationDefinition,
    FOUNDATIONS,
    FOUNDATION_NAMES,
    FOUNDATION_BY_NAME,
    SUB_FOUNDATIONS,
    SUB_FOUNDATION_CODES,
    SUBFOUNDATION_WORLDS,
    SUBFOUNDATION_WORLD_MAP,
    SUBFOUNDATION_PATTERN,
    parse_subfoundation,
    validate_subfoundation,
    validate_foundation,
    get_foundation_subfoundations,
    get_world_subfoundations,
)

# =============================================================================
# OPERATORS
# =============================================================================
from .operators import (
    OperatorDefinition,
    OPERATORS,
    OPERATOR_SYMBOLS,
    ALLOWED_OPS,
    ARITHMETIC_OPS,
    RELATIONAL_OPS,
    CAUSAL_OPS,
    COMPOSITION_OPS,
    COMMUTATIVE_OPS,
    ASSOCIATIVE_OPS,
    OPERATOR_PRECEDENCE,
    OPERATOR_SEMANTICS_ALT,
    validate_operator,
    get_operator_precedence,
    is_commutative,
    is_associative,
    get_operator_semantics,
)

# =============================================================================
# RPM (MVR PROTOCOL)
# =============================================================================
from .rpm import (
    RPMComponent,
    RPM_COMPONENTS,
    RPM_MAPPINGS,
    RPM_DESIRE,
    RPM_WISDOM,
    RPM_POWER,
    RPM_DESIRE_NOETICS,
    RPM_WISDOM_NOETICS,
    RPM_POWER_NOETICS,
    DESIRE_INDICES,
    WISDOM_INDICES,
    POWER_INDICES,
    DEFAULT_GATE_THRESHOLD,
    get_rpm_component,
    compute_dwp_scores,
    validate_rpm_noetics,
    check_gate_open,
    get_gate_status,
)

# =============================================================================
# ACQUISITIONS (22)
# =============================================================================
from .acquisitions import (
    AcquisitionDefinition,
    ACQUISITIONS,
    DESIRE_CHAIN,
    WISDOM_CHAIN,
    POWER_CHAIN,
    META_ACQUISITIONS,
    ALL_ACQUISITIONS,
    ACQUISITIONS_BY_CATEGORY,
    ACQUISITIONS_BY_FOUNDATION,
    ACQUISITION_PATTERN,
    DESIRE_PHASES,
    POWER_PHASES,
    WISDOM_PHASES,
    foundation_to_desire_id,
    foundation_to_wisdom_id,
    foundation_to_power_id,
    parse_acquisition,
    validate_acquisition,
    get_acquisition_for_foundation,
)

# =============================================================================
# ACQUISITION PATHS (Tree Path mappings for 22 acquisitions)
# =============================================================================
from .acquisition_paths import (
    ACQUISITION_PATHS,
    PATH_COLLISIONS,
    ACQUISITION_CATEGORY,
    ACQUISITION_FOUNDATION,
    ACQUISITION_ORDER,
    manual_idx_to_noetic,
    get_path_for_acquisition,
    get_acquisitions_for_path,
    validate_acquisition_paths,
)

# =============================================================================
# FOUNDATION GRAPH (7x7 adjacency for diffusion)
# =============================================================================
from .foundation_graph import (
    FOUNDATION_OPP,
    FOUNDATION_ADJ,
    FOUNDATION_ADJ_CANON,
    build_foundation_adj,
    build_canon_only_adj,
    get_opposite_foundation,
    get_foundation_neighbors,
    visualize_foundation_graph,
)

# =============================================================================
# SYNTAX AND EXPRESSION RULES
# =============================================================================
from .syntax import (
    SymbolPositionRules,
    NAMESPACES,
    ELEMENT_PATTERN as SYNTAX_ELEMENT_PATTERN,
    SUBFOUNDATION_PATTERN as SYNTAX_SUBFOUNDATION_PATTERN,
    ELEMENT_WITH_NOETIC,
    FOUNDATION_WITH_NOETIC,
    ELEMENT_WITH_FOUNDATION,
    FULL_EXPRESSION,
    OPERATOR_PATTERN,
    INVALID_PATTERNS,
    GOLDEN_RULE,
    FractalNotation,
    FRACTAL_PATTERN_UNICODE,
    FRACTAL_PATTERN_ASCII,
    ParsedExpression,
    parse_fractal,
    format_fractal,
    parse_simple_equation,
    validate_expression_syntax,
)


# =============================================================================
# COMPREHENSIVE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # Noetics
    "NoeticDefinition",
    "NOETICS",
    "NOETIC_NAMES",
    "NOETIC_BY_NAME",
    "INVOLUTION_PAIRS",
    "SELF_DUAL_NOETICS",
    "SELF_DUAL_MVR",
    "SPECIAL_PAIR",
    "MVR_FRACTAL",
    "SENSES",
    "get_complement",
    "validate_noetic",

    # Worlds
    "WorldDefinition",
    "WORLDS",
    "WORLD_LETTERS",
    "WORLD_ORDINALS",
    "WORLD_NAMES",
    "WORLD_DOMAINS",
    "WORLD_OFFSETS",
    "WORLD_CASCADE_ORDER",
    "SUBFOUNDATION_WORLD_LETTERS",
    "world_distance",
    "validate_world",
    "get_world_by_ordinal",

    # Elements
    "ElementDefinition",
    "ELEMENTS",
    "ELEMENT_CODES",
    "ELEMENT_BY_INDEX",
    "ELEMENTS_BY_WORLD",
    "ELEMENT_PATTERN",
    "element_num_to_noetic_idx",
    "noetic_idx_to_element_num",
    "parse_element",
    "validate_element",
    "element_to_index",
    "index_to_element",
    "same_world",
    "same_noetic",
    "get_element_noetic_idx",

    # Foundations
    "FoundationDefinition",
    "SubFoundationDefinition",
    "FOUNDATIONS",
    "FOUNDATION_NAMES",
    "FOUNDATION_BY_NAME",
    "SUB_FOUNDATIONS",
    "SUB_FOUNDATION_CODES",
    "SUBFOUNDATION_WORLDS",
    "SUBFOUNDATION_WORLD_MAP",
    "SUBFOUNDATION_PATTERN",
    "parse_subfoundation",
    "validate_subfoundation",
    "validate_foundation",
    "get_foundation_subfoundations",
    "get_world_subfoundations",

    # Operators
    "OperatorDefinition",
    "OPERATORS",
    "OPERATOR_SYMBOLS",
    "ALLOWED_OPS",
    "ARITHMETIC_OPS",
    "RELATIONAL_OPS",
    "CAUSAL_OPS",
    "COMPOSITION_OPS",
    "COMMUTATIVE_OPS",
    "ASSOCIATIVE_OPS",
    "OPERATOR_PRECEDENCE",
    "OPERATOR_SEMANTICS_ALT",
    "validate_operator",
    "get_operator_precedence",
    "is_commutative",
    "is_associative",
    "get_operator_semantics",

    # RPM
    "RPMComponent",
    "RPM_COMPONENTS",
    "RPM_MAPPINGS",
    "RPM_DESIRE",
    "RPM_WISDOM",
    "RPM_POWER",
    "RPM_DESIRE_NOETICS",
    "RPM_WISDOM_NOETICS",
    "RPM_POWER_NOETICS",
    "DESIRE_INDICES",
    "WISDOM_INDICES",
    "POWER_INDICES",
    "DEFAULT_GATE_THRESHOLD",
    "get_rpm_component",
    "compute_dwp_scores",
    "validate_rpm_noetics",
    "check_gate_open",
    "get_gate_status",

    # Acquisitions
    "AcquisitionDefinition",
    "ACQUISITIONS",
    "DESIRE_CHAIN",
    "WISDOM_CHAIN",
    "POWER_CHAIN",
    "META_ACQUISITIONS",
    "ALL_ACQUISITIONS",
    "ACQUISITIONS_BY_CATEGORY",
    "ACQUISITIONS_BY_FOUNDATION",
    "ACQUISITION_PATTERN",
    "DESIRE_PHASES",
    "POWER_PHASES",
    "WISDOM_PHASES",
    "foundation_to_desire_id",
    "foundation_to_wisdom_id",
    "foundation_to_power_id",
    "parse_acquisition",
    "validate_acquisition",
    "get_acquisition_for_foundation",

    # Acquisition Paths (Tree Path mappings)
    "ACQUISITION_PATHS",
    "PATH_COLLISIONS",
    "ACQUISITION_CATEGORY",
    "ACQUISITION_FOUNDATION",
    "ACQUISITION_ORDER",
    "manual_idx_to_noetic",
    "get_path_for_acquisition",
    "get_acquisitions_for_path",
    "validate_acquisition_paths",

    # Foundation Graph (7x7 adjacency)
    "FOUNDATION_OPP",
    "FOUNDATION_ADJ",
    "FOUNDATION_ADJ_CANON",
    "build_foundation_adj",
    "build_canon_only_adj",
    "get_opposite_foundation",
    "get_foundation_neighbors",
    "visualize_foundation_graph",

    # Syntax
    "SymbolPositionRules",
    "NAMESPACES",
    "SYNTAX_ELEMENT_PATTERN",
    "SYNTAX_SUBFOUNDATION_PATTERN",
    "ELEMENT_WITH_NOETIC",
    "FOUNDATION_WITH_NOETIC",
    "ELEMENT_WITH_FOUNDATION",
    "FULL_EXPRESSION",
    "OPERATOR_PATTERN",
    "INVALID_PATTERNS",
    "GOLDEN_RULE",
    "FractalNotation",
    "FRACTAL_PATTERN_UNICODE",
    "FRACTAL_PATTERN_ASCII",
    "ParsedExpression",
    "parse_fractal",
    "format_fractal",
    "parse_simple_equation",
    "validate_expression_syntax",
]
