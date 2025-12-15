"""
Multi-LLM Teacher Ensemble

Coordinates multiple LLM providers to generate high-quality
TKS interpretations with consensus scoring and canonical validation.

Key principle: LLM consensus does NOT override TKS canon.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time

from .providers import LLMProvider, LLMResponse, create_provider
from .validator import CanonicalValidator, ValidationResult
from .scoring import (
    TeacherScores,
    aggregate_scores,
    select_best_response,
    compute_agreement_score,
    compute_canon_score
)
from .transformer import (
    TrainingDataTransformer,
    TKSEquation,
    TransformedExample,
    TaskType
)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class TeacherConfig:
    """Configuration for Multi-LLM Teacher ensemble."""

    # Provider configuration
    providers: List[Dict[str, Any]] = field(default_factory=list)

    # Query settings
    max_concurrent_queries: int = 3
    query_timeout_seconds: float = 60.0
    retry_count: int = 2
    retry_delay_seconds: float = 1.0

    # Validation settings
    min_canon_score: float = 0.8
    min_confidence_score: float = 0.6
    strict_validation: bool = True

    # Response selection
    require_unanimous_canon: bool = False
    min_agreement_for_consensus: float = 0.6

    # System prompt for TKS interpretation
    system_prompt: str = """You are an expert in the TOOTRA Kabbalistic System (TKS).

Key canonical rules:
- 10 Noetics: N1=Mind, N2=Positive, N3=Negative, N4=Vibration, N5=Female, N6=Male (NOT "MEL"), N7=Rhythm, N8=Cause, N9=Effect, N10=Idea
- 4 Canonical Worlds: Spiritual (A), Mental (B), Emotional (C), Physical (D) - ONLY these are valid
- 7 Foundations: Unity, Wisdom, Life, Companionship, Power, Material, Lust
- Involution pairs (2,3), (5,6), (8,9) compose to approximately N10 (Idea)
- RPM: Desire=N2,N3; Wisdom=N1,N4,N5,N6,N7; Power=N8,N9

IMPORTANT: Only use canonical world codes A, B, C, D. Non-canonical codes like Y, Z are rejected.

Provide interpretations that are canonically correct and insightful."""

    @classmethod
    def default(cls) -> 'TeacherConfig':
        """Create default configuration."""
        return cls(providers=[
            {"type": "mock", "model": "mock-teacher"}
        ])

    @classmethod
    def from_file(cls, path: str) -> 'TeacherConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ==============================================================================
# RESPONSE TYPES
# ==============================================================================

@dataclass
class TeacherResponse:
    """Response from a single teacher (LLM provider)."""
    provider_name: str
    model: str
    text: str
    tokens_used: int = 0
    latency_ms: float = 0.0

    # Validation
    is_valid: bool = True
    canon_score: float = 1.0
    validation_issues: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.text) > 0


@dataclass
class EnsembleResult:
    """Result from the full teacher ensemble."""

    # Best response
    best_response: str
    best_provider: str
    best_score: float

    # All responses
    responses: List[TeacherResponse]

    # Aggregate scores
    scores: TeacherScores

    # Validation
    all_valid: bool
    valid_count: int
    total_count: int

    # Metadata
    query_time_ms: float
    equation: Optional[TKSEquation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_response": self.best_response,
            "best_provider": self.best_provider,
            "best_score": self.best_score,
            "scores": {
                "agreement": self.scores.agreement_score,
                "canon": self.scores.canon_score,
                "confidence": self.scores.confidence_score
            },
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms
        }


# ==============================================================================
# MULTI-LLM TEACHER ENSEMBLE
# ==============================================================================

class MultiLLMTeacher:
    """
    Multi-LLM Teacher Ensemble for TKS interpretation.

    Queries multiple LLM providers, validates responses against
    TKS canon, scores agreement, and selects the best response.

    Key principle: LLM consensus does NOT override TKS canon.
    """

    def __init__(self, config: Optional[TeacherConfig] = None):
        """
        Initialize the teacher ensemble.

        Args:
            config: TeacherConfig or None for defaults
        """
        self.config = config or TeacherConfig.default()
        self.validator = CanonicalValidator(strict_mode=self.config.strict_validation)
        self.transformer = TrainingDataTransformer(
            min_canon_score=self.config.min_canon_score,
            min_confidence_score=self.config.min_confidence_score
        )

        # Initialize providers
        self.providers: List[LLMProvider] = []
        self._init_providers()

        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.canonical_rejections = 0

    def _init_providers(self):
        """Initialize LLM providers from config."""
        for pconfig in self.config.providers:
            provider_type = pconfig.pop("type", "mock")
            model = pconfig.pop("model", "default")
            try:
                provider = create_provider(provider_type, model, **pconfig)
                self.providers.append(provider)
            except Exception as e:
                print(f"Warning: Failed to initialize {provider_type} provider: {e}")

        if not self.providers:
            # Add mock provider as fallback
            from .providers import MockProvider
            self.providers.append(MockProvider())

    def add_provider(self, provider: LLMProvider):
        """Add a provider to the ensemble."""
        self.providers.append(provider)

    def query(
        self,
        equation: TKSEquation,
        prompt_template: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> EnsembleResult:
        """
        Query all providers for an interpretation.

        Args:
            equation: TKS equation to interpret
            prompt_template: Optional custom prompt template
            additional_context: Optional additional context

        Returns:
            EnsembleResult with best response and scores
        """
        start_time = time.time()
        self.total_queries += 1

        # Build prompt
        prompt = self._build_prompt(equation, prompt_template, additional_context)

        # Query all providers
        responses = self._query_providers(prompt)

        # Validate responses
        validated_responses = self._validate_responses(responses)

        # Score responses
        response_texts = [r.text for r in validated_responses if r.success]
        scores = aggregate_scores(response_texts, self.validator)

        # Select best response
        best_text, best_idx, best_score = select_best_response(
            response_texts, scores, self.validator
        )

        # Determine best provider
        valid_responses = [r for r in validated_responses if r.success]
        best_provider = valid_responses[best_idx].provider_name if best_idx >= 0 and valid_responses else "none"

        # Count valid responses
        valid_count = sum(1 for r in validated_responses if r.is_valid)
        total_count = len(validated_responses)

        if valid_count > 0:
            self.successful_queries += 1
        self.canonical_rejections += scores.canonical_vetoes

        query_time = (time.time() - start_time) * 1000

        return EnsembleResult(
            best_response=best_text,
            best_provider=best_provider,
            best_score=best_score,
            responses=validated_responses,
            scores=scores,
            all_valid=valid_count == total_count,
            valid_count=valid_count,
            total_count=total_count,
            query_time_ms=query_time,
            equation=equation,
            metadata={
                "prompt": prompt,
                "system_prompt": self.config.system_prompt
            }
        )

    def _build_prompt(
        self,
        equation: TKSEquation,
        template: Optional[str],
        context: Optional[str]
    ) -> str:
        """Build the interpretation prompt."""
        if template:
            prompt = template.format(equation=equation.to_string())
        else:
            # Default prompt
            elements_desc = []
            for elem in equation.elements:
                world = elem[0]
                noetic = int(elem[1:])
                world_name = {"A": "Spiritual", "B": "Mental", "C": "Emotional", "D": "Physical"}.get(world, world)
                noetic_names = {
                    1: "Mind", 2: "Positive", 3: "Negative", 4: "Vibration",
                    5: "Female", 6: "Male", 7: "Rhythm", 8: "Cause", 9: "Effect", 10: "Idea"
                }
                noetic_name = noetic_names.get(noetic, str(noetic))
                elements_desc.append(f"- {elem}: {world_name}-{noetic_name}")

            prompt = f"""Interpret the following TKS equation:

Equation: {equation.to_string()}

Elements:
{chr(10).join(elements_desc)}

Provide:
1. The meaning and purpose of this equation
2. The noetic interactions at play
3. The RPM (Desire/Wisdom/Power) implications
4. The expected effects or manifestations"""

        if context:
            prompt = f"{prompt}\n\nAdditional context: {context}"

        return prompt

    def _query_providers(self, prompt: str) -> List[TeacherResponse]:
        """Query all providers concurrently."""
        responses = []

        # Use thread pool for concurrent queries
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_queries
        ) as executor:
            futures = {
                executor.submit(self._query_single_provider, provider, prompt): provider
                for provider in self.providers
            }

            for future in concurrent.futures.as_completed(
                futures,
                timeout=self.config.query_timeout_seconds
            ):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    provider = futures[future]
                    responses.append(TeacherResponse(
                        provider_name=provider.provider_name,
                        model=provider.model,
                        text="",
                        error=str(e)
                    ))

        return responses

    def _query_single_provider(
        self,
        provider: LLMProvider,
        prompt: str
    ) -> TeacherResponse:
        """Query a single provider with retries."""
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                llm_response = provider.query(
                    prompt,
                    system_prompt=self.config.system_prompt
                )

                if llm_response.success:
                    return TeacherResponse(
                        provider_name=provider.provider_name,
                        model=provider.model,
                        text=llm_response.text,
                        tokens_used=llm_response.tokens_used,
                        latency_ms=llm_response.latency_ms,
                        metadata=llm_response.metadata
                    )
                else:
                    last_error = llm_response.error

            except Exception as e:
                last_error = str(e)

            if attempt < self.config.retry_count:
                time.sleep(self.config.retry_delay_seconds)

        return TeacherResponse(
            provider_name=provider.provider_name,
            model=provider.model,
            text="",
            error=last_error or "Unknown error"
        )

    def _validate_responses(
        self,
        responses: List[TeacherResponse]
    ) -> List[TeacherResponse]:
        """Validate all responses against TKS canon."""
        for response in responses:
            if not response.success:
                response.is_valid = False
                response.canon_score = 0.0
                continue

            result = self.validator.validate(response.text)
            response.is_valid = result.is_valid
            response.canon_score = result.canon_score
            response.validation_issues = [
                f"{issue.severity.value}: {issue.message}"
                for issue in result.issues
            ]

        return responses

    def generate_training_data(
        self,
        equations: List[TKSEquation],
        task_types: Optional[List[TaskType]] = None,
        show_progress: bool = True
    ) -> List[TransformedExample]:
        """
        Generate training data from multiple equations.

        Args:
            equations: List of equations to interpret
            task_types: Task types to generate
            show_progress: Whether to show progress

        Returns:
            List of transformed training examples
        """
        all_examples = []

        for i, equation in enumerate(equations):
            if show_progress:
                print(f"Processing equation {i+1}/{len(equations)}: {equation.to_string()}")

            # Query ensemble
            result = self.query(equation)

            if result.valid_count == 0:
                if show_progress:
                    print(f"  -> No valid responses, skipping")
                continue

            # Transform to training examples
            examples = self.transformer.transform(
                equation=equation,
                interpretation=result.best_response,
                canon_score=result.scores.canon_score,
                confidence_score=result.scores.confidence_score,
                task_types=task_types
            )

            all_examples.extend(examples)

            if show_progress:
                print(f"  -> Generated {len(examples)} examples (canon: {result.scores.canon_score:.2f})")

        return all_examples

    def get_statistics(self) -> Dict[str, Any]:
        """Get query statistics."""
        success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0

        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "canonical_rejections": self.canonical_rejections,
            "success_rate": success_rate,
            "provider_count": len(self.providers),
            "providers": [p.provider_name for p in self.providers]
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_teacher(
    providers: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> MultiLLMTeacher:
    """
    Create a teacher ensemble with specified providers.

    Args:
        providers: List of provider configurations
        **kwargs: Additional TeacherConfig options

    Returns:
        Configured MultiLLMTeacher

    Example:
        teacher = create_teacher(providers=[
            {"type": "openai", "model": "gpt-4"},
            {"type": "anthropic", "model": "claude-3-sonnet-20240229"},
        ])
    """
    config = TeacherConfig(
        providers=providers or [{"type": "mock", "model": "mock"}],
        **kwargs
    )
    return MultiLLMTeacher(config)


def interpret_equation(
    equation: TKSEquation,
    teacher: Optional[MultiLLMTeacher] = None
) -> str:
    """
    Get interpretation for a single equation.

    Args:
        equation: TKS equation to interpret
        teacher: Optional teacher (creates default if None)

    Returns:
        Best interpretation text
    """
    if teacher is None:
        teacher = create_teacher()

    result = teacher.query(equation)
    return result.best_response
