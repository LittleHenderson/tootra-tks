"""
TKS Multi-LLM Teacher Module

A teacher ensemble system that queries multiple LLMs to generate
high-quality interpretations of TKS equations, with canonical validation.

Components:
    - providers: LLM provider adapters (OpenAI, Anthropic, Gemini, Local)
    - ensemble: MultiLLMTeacher ensemble class
    - validator: Canonical TKS validation
    - transformer: Training data transformation (E2I, I2E, S2E, E2RPM, E2F)
    - scoring: Agreement, canon, and confidence scoring

Key Principle: LLM consensus does NOT override TKS canon.
"""

from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    LocalProvider,
    MockProvider,
    create_provider,
)

from .ensemble import (
    MultiLLMTeacher,
    TeacherConfig,
    TeacherResponse,
    EnsembleResult,
)

from .validator import (
    CanonicalValidator,
    ValidationResult,
    CANONICAL_RULES,
)

from .transformer import (
    TrainingDataTransformer,
    TaskType,
    TransformedExample,
    TKSEquation,
)

from .scoring import (
    compute_agreement_score,
    compute_canon_score,
    compute_confidence_score,
    aggregate_scores,
)

__all__ = [
    # Providers
    'LLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'LocalProvider',
    'MockProvider',
    'create_provider',
    # Ensemble
    'MultiLLMTeacher',
    'TeacherConfig',
    'TeacherResponse',
    'EnsembleResult',
    # Validator
    'CanonicalValidator',
    'ValidationResult',
    'CANONICAL_RULES',
    # Transformer
    'TrainingDataTransformer',
    'TaskType',
    'TransformedExample',
    'TKSEquation',
    # Scoring
    'compute_agreement_score',
    'compute_canon_score',
    'compute_confidence_score',
    'aggregate_scores',
]
