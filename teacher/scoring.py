"""
Multi-LLM Teacher Scoring Module

Computes three key scores for teacher ensemble responses:
    1. Agreement Score: How much do LLMs agree with each other?
    2. Canon Score: How well does the response follow TKS canon?
    3. Confidence Score: Combined confidence in the response quality

Key principle: Canon score can VETO high agreement scores.
LLM consensus does NOT override TKS canon.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

from .validator import CanonicalValidator, ValidationResult


# ==============================================================================
# SCORE DATACLASS
# ==============================================================================

@dataclass
class TeacherScores:
    """Aggregated scores for teacher ensemble output."""
    agreement_score: float      # 0.0 to 1.0 - LLM agreement
    canon_score: float          # 0.0 to 1.0 - Canonical compliance
    confidence_score: float     # 0.0 to 1.0 - Overall confidence

    # Components
    semantic_agreement: float   # Text similarity
    structural_agreement: float # Element/RPM agreement
    coverage: float             # Response completeness

    # Metadata
    num_responses: int
    valid_responses: int
    canonical_vetoes: int       # Responses rejected for canon violations


# ==============================================================================
# AGREEMENT SCORING
# ==============================================================================

def compute_agreement_score(responses: List[str]) -> Dict[str, float]:
    """
    Compute agreement score between multiple LLM responses.

    Returns:
        Dict with:
            - semantic_agreement: Text similarity (0-1)
            - structural_agreement: Element/noetic agreement (0-1)
            - overall_agreement: Weighted combination (0-1)
    """
    if len(responses) < 2:
        return {
            "semantic_agreement": 1.0,
            "structural_agreement": 1.0,
            "overall_agreement": 1.0
        }

    # Compute semantic agreement (simplified n-gram overlap)
    semantic = _compute_semantic_agreement(responses)

    # Compute structural agreement (elements, noetics, RPM)
    structural = _compute_structural_agreement(responses)

    # Weighted combination
    overall = 0.4 * semantic + 0.6 * structural

    return {
        "semantic_agreement": semantic,
        "structural_agreement": structural,
        "overall_agreement": overall
    }


def _compute_semantic_agreement(responses: List[str]) -> float:
    """Compute semantic similarity using n-gram overlap."""
    if len(responses) < 2:
        return 1.0

    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        words = text.lower().split()
        if len(words) < n:
            return set(words)
        return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

    # Compute pairwise Jaccard similarity
    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            ngrams_i = get_ngrams(responses[i])
            ngrams_j = get_ngrams(responses[j])

            if not ngrams_i or not ngrams_j:
                continue

            intersection = len(ngrams_i & ngrams_j)
            union = len(ngrams_i | ngrams_j)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)

    return sum(similarities) / len(similarities) if similarities else 1.0


def _compute_structural_agreement(responses: List[str]) -> float:
    """Compute agreement on structural elements (elements, noetics, RPM)."""
    if len(responses) < 2:
        return 1.0

    # Extract structural features from each response
    features_list = [_extract_structural_features(r) for r in responses]

    # Compare elements
    element_agreement = _compare_feature_sets(
        [f["elements"] for f in features_list]
    )

    # Compare noetics
    noetic_agreement = _compare_feature_sets(
        [f["noetics"] for f in features_list]
    )

    # Compare RPM mentions
    rpm_agreement = _compare_feature_sets(
        [f["rpm_components"] for f in features_list]
    )

    # Weighted average
    return 0.4 * element_agreement + 0.4 * noetic_agreement + 0.2 * rpm_agreement


def _extract_structural_features(text: str) -> Dict[str, Set[str]]:
    """Extract structural features from response text."""
    features = {
        "elements": set(),
        "noetics": set(),
        "rpm_components": set(),
        "foundations": set()
    }

    # Elements (e.g., B4, C10, A1) - Only canonical worlds A, B, C, D
    element_pattern = re.compile(r'\b([ABCD])(\d{1,2})\b')
    for match in element_pattern.finditer(text):
        features["elements"].add(match.group(0))

    # Noetics
    noetic_pattern = re.compile(r'\b(?:N|noetic)\s*(\d{1,2})\b', re.I)
    for match in noetic_pattern.finditer(text):
        features["noetics"].add(f"N{match.group(1)}")

    # RPM components
    for component in ["desire", "wisdom", "power"]:
        if component in text.lower():
            features["rpm_components"].add(component)

    # Foundations (canonical)
    foundations = ["unity", "wisdom", "life", "companionship", "power", "material", "lust"]
    for f in foundations:
        if f in text.lower():
            features["foundations"].add(f)

    return features


def _compare_feature_sets(feature_sets: List[Set[str]]) -> float:
    """Compare multiple feature sets for agreement."""
    if not feature_sets or all(len(f) == 0 for f in feature_sets):
        return 1.0  # No features to compare = agreement

    # Count feature occurrences
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)

    if not all_features:
        return 1.0

    # Compute agreement per feature
    agreements = []
    for feature in all_features:
        count = sum(1 for fs in feature_sets if feature in fs)
        agreement = count / len(feature_sets)
        agreements.append(agreement)

    return sum(agreements) / len(agreements) if agreements else 1.0


# ==============================================================================
# CANON SCORING
# ==============================================================================

def compute_canon_score(
    responses: List[str],
    validator: Optional[CanonicalValidator] = None
) -> Dict[str, float]:
    """
    Compute canon scores for LLM responses.

    Returns:
        Dict with:
            - individual_scores: List of per-response canon scores
            - average_score: Mean canon score
            - min_score: Minimum canon score
            - valid_count: Number of responses passing validation
            - veto_count: Responses rejected for canon violations
    """
    if validator is None:
        validator = CanonicalValidator(strict_mode=True)

    individual_scores = []
    valid_count = 0
    veto_count = 0

    for response in responses:
        result = validator.validate(response)
        individual_scores.append(result.canon_score)

        if result.is_valid:
            valid_count += 1
        else:
            veto_count += 1

    if not individual_scores:
        return {
            "individual_scores": [],
            "average_score": 0.0,
            "min_score": 0.0,
            "valid_count": 0,
            "veto_count": 0
        }

    return {
        "individual_scores": individual_scores,
        "average_score": sum(individual_scores) / len(individual_scores),
        "min_score": min(individual_scores),
        "valid_count": valid_count,
        "veto_count": veto_count
    }


# ==============================================================================
# CONFIDENCE SCORING
# ==============================================================================

def compute_confidence_score(
    agreement_scores: Dict[str, float],
    canon_scores: Dict[str, float],
    coverage: float = 1.0
) -> float:
    """
    Compute overall confidence score.

    Key principle: Canon score can VETO high agreement.
    A response with 100% agreement but 50% canon score
    should have low confidence.

    Args:
        agreement_scores: Output of compute_agreement_score()
        canon_scores: Output of compute_canon_score()
        coverage: Response completeness (0-1)

    Returns:
        Confidence score (0.0 to 1.0)
    """
    agreement = agreement_scores.get("overall_agreement", 0.0)
    canon = canon_scores.get("min_score", 0.0)  # Use minimum (strictest)
    valid_ratio = 1.0
    if canon_scores.get("individual_scores"):
        total = len(canon_scores["individual_scores"])
        valid = canon_scores.get("valid_count", 0)
        valid_ratio = valid / total

    # Canon can veto agreement
    # If canon < 0.5, drastically reduce confidence
    if canon < 0.5:
        canon_factor = canon * 0.5  # Heavily penalized
    else:
        canon_factor = canon

    # Confidence formula:
    # - High agreement + high canon = high confidence
    # - High agreement + low canon = low confidence (canon veto)
    # - Low agreement + high canon = medium confidence
    confidence = (
        0.3 * agreement +
        0.5 * canon_factor +
        0.1 * valid_ratio +
        0.1 * coverage
    )

    return max(0.0, min(1.0, confidence))


# ==============================================================================
# AGGREGATE SCORING
# ==============================================================================

def aggregate_scores(
    responses: List[str],
    validator: Optional[CanonicalValidator] = None
) -> TeacherScores:
    """
    Compute all scores for a set of teacher responses.

    Args:
        responses: List of LLM response texts
        validator: Optional custom validator

    Returns:
        TeacherScores with all computed scores
    """
    if validator is None:
        validator = CanonicalValidator(strict_mode=True)

    # Compute agreement
    agreement = compute_agreement_score(responses)

    # Compute canon scores
    canon = compute_canon_score(responses, validator)

    # Compute coverage (simplified: check for key components)
    coverage = _compute_coverage(responses)

    # Compute confidence
    confidence = compute_confidence_score(agreement, canon, coverage)

    return TeacherScores(
        agreement_score=agreement["overall_agreement"],
        canon_score=canon["average_score"],
        confidence_score=confidence,
        semantic_agreement=agreement["semantic_agreement"],
        structural_agreement=agreement["structural_agreement"],
        coverage=coverage,
        num_responses=len(responses),
        valid_responses=canon["valid_count"],
        canonical_vetoes=canon["veto_count"]
    )


def _compute_coverage(responses: List[str]) -> float:
    """
    Compute response coverage/completeness.

    Checks if responses contain expected components:
    - Element references
    - Noetic content
    - Interpretation text
    """
    if not responses:
        return 0.0

    coverage_scores = []
    for response in responses:
        score = 0.0
        components_checked = 0

        # Has elements? (Only canonical worlds A, B, C, D)
        if re.search(r'\b[ABCD]\d{1,2}\b', response):
            score += 1.0
        components_checked += 1

        # Has noetic references?
        if re.search(r'\b(?:N|noetic)\s*\d+\b', response, re.I):
            score += 1.0
        components_checked += 1

        # Has interpretation (sufficient length)?
        if len(response.split()) >= 20:
            score += 1.0
        components_checked += 1

        # Has RPM or Foundation references?
        if any(term in response.lower() for term in ["desire", "wisdom", "power", "foundation"]):
            score += 1.0
        components_checked += 1

        coverage_scores.append(score / components_checked if components_checked > 0 else 0.0)

    return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0


# ==============================================================================
# SELECTION FUNCTIONS
# ==============================================================================

def select_best_response(
    responses: List[str],
    scores: Optional[TeacherScores] = None,
    validator: Optional[CanonicalValidator] = None
) -> Tuple[str, int, float]:
    """
    Select the best response from the ensemble.

    Selection criteria:
    1. Must pass canonical validation (no errors)
    2. Highest individual canon score
    3. Best coverage

    Returns:
        Tuple of (best_response, index, score)
    """
    if not responses:
        return "", -1, 0.0

    if validator is None:
        validator = CanonicalValidator(strict_mode=True)

    best_idx = -1
    best_score = -1.0
    best_response = ""

    for i, response in enumerate(responses):
        result = validator.validate(response)

        # Must pass validation
        if not result.is_valid:
            continue

        # Score combines canon score and coverage
        coverage = _compute_coverage([response])
        combined_score = 0.7 * result.canon_score + 0.3 * coverage

        if combined_score > best_score:
            best_score = combined_score
            best_idx = i
            best_response = response

    # If no valid response, return highest canon score anyway with warning
    if best_idx == -1 and responses:
        canon_scores = compute_canon_score(responses, validator)
        if canon_scores["individual_scores"]:
            best_idx = canon_scores["individual_scores"].index(
                max(canon_scores["individual_scores"])
            )
            best_response = responses[best_idx]
            best_score = canon_scores["individual_scores"][best_idx]

    return best_response, best_idx, best_score


def filter_canonical_responses(
    responses: List[str],
    min_canon_score: float = 0.8,
    validator: Optional[CanonicalValidator] = None
) -> List[Tuple[str, float]]:
    """
    Filter responses to only those meeting canonical standards.

    Args:
        responses: List of response texts
        min_canon_score: Minimum required canon score
        validator: Optional custom validator

    Returns:
        List of (response, score) tuples for passing responses
    """
    if validator is None:
        validator = CanonicalValidator(strict_mode=True)

    passing = []
    for response in responses:
        result = validator.validate(response)
        if result.is_valid and result.canon_score >= min_canon_score:
            passing.append((response, result.canon_score))

    # Sort by score descending
    passing.sort(key=lambda x: x[1], reverse=True)
    return passing
