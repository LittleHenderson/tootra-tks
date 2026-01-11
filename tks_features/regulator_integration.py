"""
TKS Regulator Integration - Bridges FM/ACBE/MVR with RPM/NJT

This module integrates the TKS regulators (FM, ACBE, MVR) into the neural
architecture by detecting D/W/P gaps and generating appropriate sub-goals.

Gap Detection:
    - D_gap: Desire score < threshold -> Trigger MVR
    - W_gap: Wisdom score < threshold -> Trigger FM
    - P_gap: Power score < threshold -> Trigger ACBE

Integration Points:
    1. After RPM gating computes D/W/P scores
    2. Before RecursionStack push (sub-goals go on stack)
    3. In NJT block as auxiliary loss signal

Author: Claude Code
Date: 2026-01-11
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Noetic indices for D/W/P extraction (from tks_llm_core_v2.py)
# MVR: Mind(1), Vibration(4), Rhythm(7) across 4 worlds
DESIRE_INDICES = [1, 4, 7, 11, 14, 17, 21, 24, 27, 31, 34, 37]
# Female/Male: 5, 6 across 4 worlds
WISDOM_INDICES = [5, 6, 15, 16, 25, 26, 35, 36]
# Cause/Effect: 8, 9 across 4 worlds
POWER_INDICES = [8, 9, 18, 19, 28, 29, 38, 39]


@dataclass
class RegulatorConfig:
    """Configuration for regulator integration."""
    gap_threshold: float = 0.3  # Below this, trigger regulator
    mvr_boost: float = 1.5      # Amplification factor for MVR output
    fm_boost: float = 1.2       # Amplification for FM output
    acbe_boost: float = 1.3     # Amplification for ACBE output
    use_soft_gating: bool = True  # Use soft thresholds vs hard
    hidden_dim: int = 64


class GapDetector(nn.Module):
    """
    Detects D/W/P gaps from noetic representations.

    Input: [batch, seq, 40] noetic tensor
    Output: [batch, seq, 3] gap scores for D/W/P
    """

    def __init__(self, config: RegulatorConfig):
        super().__init__()
        self.config = config

        # Register canonical indices
        self.register_buffer('desire_idx', torch.tensor(DESIRE_INDICES, dtype=torch.long))
        self.register_buffer('wisdom_idx', torch.tensor(WISDOM_INDICES, dtype=torch.long))
        self.register_buffer('power_idx', torch.tensor(POWER_INDICES, dtype=torch.long))

        # Learnable gap detectors
        self.desire_detector = nn.Sequential(
            nn.Linear(len(DESIRE_INDICES), config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.wisdom_detector = nn.Sequential(
            nn.Linear(len(WISDOM_INDICES), config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.power_detector = nn.Sequential(
            nn.Linear(len(POWER_INDICES), config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect gaps in D/W/P.

        Args:
            x: [batch, seq, 40] noetic tensor

        Returns:
            Dict with:
                - 'desire_score': [batch, seq] satisfaction scores
                - 'wisdom_score': [batch, seq]
                - 'power_score': [batch, seq]
                - 'desire_gap': [batch, seq] gap indicators (1 = gap)
                - 'wisdom_gap': [batch, seq]
                - 'power_gap': [batch, seq]
        """
        # Extract canonical noetic subsets
        d_feats = x[..., self.desire_idx]  # [batch, seq, 12]
        w_feats = x[..., self.wisdom_idx]  # [batch, seq, 8]
        p_feats = x[..., self.power_idx]   # [batch, seq, 8]

        # Compute satisfaction scores
        d_score = self.desire_detector(d_feats).squeeze(-1)  # [batch, seq]
        w_score = self.wisdom_detector(w_feats).squeeze(-1)
        p_score = self.power_detector(p_feats).squeeze(-1)

        # Compute gaps (1 = gap detected)
        threshold = self.config.gap_threshold
        if self.config.use_soft_gating:
            # Soft gap: continuous signal
            d_gap = torch.sigmoid((threshold - d_score) * 10)
            w_gap = torch.sigmoid((threshold - w_score) * 10)
            p_gap = torch.sigmoid((threshold - p_score) * 10)
        else:
            # Hard gap: binary
            d_gap = (d_score < threshold).float()
            w_gap = (w_score < threshold).float()
            p_gap = (p_score < threshold).float()

        return {
            'desire_score': d_score,
            'wisdom_score': w_score,
            'power_score': p_score,
            'desire_gap': d_gap,
            'wisdom_gap': w_gap,
            'power_gap': p_gap,
        }


class MVRRegulatorModule(nn.Module):
    """
    MVR (Mind, Vibration, Rhythm) - Desire Booster.

    Triggered when D is low. Applies Noetics 1, 4, 7 to amplify desire.
    """

    def __init__(self, noetic_dim: int = 40, config: RegulatorConfig = None):
        super().__init__()
        self.config = config or RegulatorConfig()

        # MVR transformation: applies rhythm (7) -> vibration (4) -> mind (1)
        # Creates nested amplification of desire
        self.mvr_transform = nn.Sequential(
            nn.Linear(noetic_dim, noetic_dim),  # Rhythm gate
            nn.GELU(),
            nn.Linear(noetic_dim, noetic_dim),  # Vibration modulation
            nn.GELU(),
            nn.Linear(noetic_dim, noetic_dim),  # Mind alignment
        )

        # Mask to focus on desire-related indices
        self.register_buffer('desire_mask', self._create_desire_mask(noetic_dim))

    def _create_desire_mask(self, dim: int) -> torch.Tensor:
        """Create mask that focuses on desire-related noetics."""
        mask = torch.zeros(dim)
        for idx in DESIRE_INDICES:
            if idx < dim:
                mask[idx] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        gap_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply MVR transformation to boost desire.

        Args:
            x: [batch, seq, noetic_dim] input
            gap_strength: [batch, seq] how much gap (0-1)

        Returns:
            Desire-boosted tensor [batch, seq, noetic_dim]
        """
        # Apply MVR transformation
        mvr_out = self.mvr_transform(x)

        # Scale boost by gap strength
        boost = gap_strength.unsqueeze(-1) * self.config.mvr_boost

        # Apply masked boost to desire-related dimensions
        desire_boost = mvr_out * self.desire_mask * boost

        return x + desire_boost


class FMRegulatorModule(nn.Module):
    """
    FM (Foundation Modulator) - Wisdom Booster.

    Triggered when W is low. Applies Noetics 5 (Female/Authority)
    and 6 (Male/Superset) to build wisdom.
    """

    def __init__(self, noetic_dim: int = 40, config: RegulatorConfig = None):
        super().__init__()
        self.config = config or RegulatorConfig()

        # FM transformation: female (authority) and male (superset) processing
        self.female_branch = nn.Sequential(
            nn.Linear(noetic_dim, noetic_dim),
            nn.GELU(),
        )
        self.male_branch = nn.Sequential(
            nn.Linear(noetic_dim, noetic_dim),
            nn.GELU(),
        )

        # Combine branches
        self.fm_combine = nn.Linear(noetic_dim * 2, noetic_dim)

        # Mask to focus on wisdom-related indices
        self.register_buffer('wisdom_mask', self._create_wisdom_mask(noetic_dim))

    def _create_wisdom_mask(self, dim: int) -> torch.Tensor:
        """Create mask that focuses on wisdom-related noetics."""
        mask = torch.zeros(dim)
        for idx in WISDOM_INDICES:
            if idx < dim:
                mask[idx] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        gap_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FM transformation to build wisdom.

        Args:
            x: [batch, seq, noetic_dim] input
            gap_strength: [batch, seq] how much gap (0-1)

        Returns:
            Wisdom-enhanced tensor [batch, seq, noetic_dim]
        """
        # Female branch (authority/reservoir)
        female_out = self.female_branch(x)

        # Male branch (superset/related ideas)
        male_out = self.male_branch(x)

        # Combine
        combined = torch.cat([female_out, male_out], dim=-1)
        fm_out = self.fm_combine(combined)

        # Scale boost by gap strength
        boost = gap_strength.unsqueeze(-1) * self.config.fm_boost

        # Apply masked boost to wisdom-related dimensions
        wisdom_boost = fm_out * self.wisdom_mask * boost

        return x + wisdom_boost


class ACBERegulatorModule(nn.Module):
    """
    ACBE (Above/Cause, Below/Effect) - Capability Booster.

    Triggered when P is low. Applies Noetics 8 (Above/Cause)
    and 9 (Below/Effect) to build power/capability.
    """

    def __init__(self, noetic_dim: int = 40, config: RegulatorConfig = None):
        super().__init__()
        self.config = config or RegulatorConfig()

        # ACBE transformation: cause and effect processing
        self.cause_branch = nn.Sequential(
            nn.Linear(noetic_dim, noetic_dim),
            nn.GELU(),
        )
        self.effect_branch = nn.Sequential(
            nn.Linear(noetic_dim, noetic_dim),
            nn.GELU(),
        )

        # Combine branches
        self.acbe_combine = nn.Linear(noetic_dim * 2, noetic_dim)

        # Mask to focus on power-related indices
        self.register_buffer('power_mask', self._create_power_mask(noetic_dim))

    def _create_power_mask(self, dim: int) -> torch.Tensor:
        """Create mask that focuses on power-related noetics."""
        mask = torch.zeros(dim)
        for idx in POWER_INDICES:
            if idx < dim:
                mask[idx] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        gap_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply ACBE transformation to build capability.

        Args:
            x: [batch, seq, noetic_dim] input
            gap_strength: [batch, seq] how much gap (0-1)

        Returns:
            Power-enhanced tensor [batch, seq, noetic_dim]
        """
        # Cause branch (what enables this capability)
        cause_out = self.cause_branch(x)

        # Effect branch (what this capability enables)
        effect_out = self.effect_branch(x)

        # Combine
        combined = torch.cat([cause_out, effect_out], dim=-1)
        acbe_out = self.acbe_combine(combined)

        # Scale boost by gap strength
        boost = gap_strength.unsqueeze(-1) * self.config.acbe_boost

        # Apply masked boost to power-related dimensions
        power_boost = acbe_out * self.power_mask * boost

        return x + power_boost


class IntegratedRegulatorLayer(nn.Module):
    """
    Unified regulator layer for NJT/RPM integration.

    Detects D/W/P gaps and applies appropriate regulators.
    Can be inserted after RPM gating or within NJT blocks.
    """

    def __init__(self, noetic_dim: int = 40, config: RegulatorConfig = None):
        super().__init__()
        self.config = config or RegulatorConfig()
        self.noetic_dim = noetic_dim

        # Gap detector
        self.gap_detector = GapDetector(self.config)

        # Regulators
        self.mvr = MVRRegulatorModule(noetic_dim, self.config)
        self.fm = FMRegulatorModule(noetic_dim, self.config)
        self.acbe = ACBERegulatorModule(noetic_dim, self.config)

        # Learnable blend weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        return_gap_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Apply regulators based on detected gaps.

        Args:
            x: [batch, seq, noetic_dim] noetic tensor
            return_gap_info: Whether to return gap detection info

        Returns:
            output: Regulated tensor [batch, seq, noetic_dim]
            gap_info: Optional dict with gap detection details
        """
        # Detect gaps
        gap_info = self.gap_detector(x)

        # Apply regulators where gaps exist
        x_regulated = x.clone()

        # MVR for desire gaps
        x_regulated = self.mvr(x_regulated, gap_info['desire_gap'])

        # FM for wisdom gaps
        x_regulated = self.fm(x_regulated, gap_info['wisdom_gap'])

        # ACBE for power gaps
        x_regulated = self.acbe(x_regulated, gap_info['power_gap'])

        # Blend with original based on learnable weight
        blend = torch.sigmoid(self.blend_weight)
        output = blend * x_regulated + (1 - blend) * x

        if return_gap_info:
            return output, gap_info
        return output, None

    def compute_regulator_loss(
        self,
        gap_info: Dict,
        target_dwp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for regulator training.

        If target_dwp is provided, uses supervised loss.
        Otherwise, uses self-supervised loss encouraging higher scores.

        Args:
            gap_info: Gap detection output
            target_dwp: Optional [batch, seq, 3] target D/W/P scores

        Returns:
            Scalar loss tensor
        """
        d_score = gap_info['desire_score']
        w_score = gap_info['wisdom_score']
        p_score = gap_info['power_score']

        if target_dwp is not None:
            # Supervised loss
            pred = torch.stack([d_score, w_score, p_score], dim=-1)
            return F.mse_loss(pred, target_dwp)
        else:
            # Self-supervised: encourage higher scores (minimize gaps)
            gap_penalty = (
                gap_info['desire_gap'].mean() +
                gap_info['wisdom_gap'].mean() +
                gap_info['power_gap'].mean()
            )
            return gap_penalty * 0.1


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RegulatorConfig',
    'GapDetector',
    'MVRRegulatorModule',
    'FMRegulatorModule',
    'ACBERegulatorModule',
    'IntegratedRegulatorLayer',
    'DESIRE_INDICES',
    'WISDOM_INDICES',
    'POWER_INDICES',
]
