"""
TKS Routing Stability Module - Mitigations for Routing Collapse

This module provides mechanisms to prevent and detect routing collapse in the
Traceable TKS-Transformer, where entropy-regularized top-k routing can collapse
early with all tokens choosing the same noetic operators.

Components:
1. RoutingTemperatureScheduler - Warm-start routing with temperature annealing
2. LoadBalancingLoss - Switch Transformer style auxiliary loss for diversity
3. NoisyRouting - Gumbel/Gaussian noise injection with annealing
4. RoutingEntropyMonitor - Real-time collapse detection
5. StableRouting - Integrated wrapper applying all mitigations

Mathematical Foundation:
- Temperature scaling: softmax(logits / T) where T: high -> low
- Load balancing: L_aux = N * sum(f_i * P_i) penalizes concentration
- Gumbel noise: -log(-log(U)) for exploration
- Entropy: H = -sum(p_i * log(p_i)) measures routing diversity

References:
- Switch Transformer (Fedus et al., 2021): Load balancing loss
- Gumbel-Softmax (Jang et al., 2017): Differentiable discrete sampling
- TKS v7.4: 10 noetic operators as routing targets

Author: TKS Compiler Agent
Date: 2025-12-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import deque
import math


# =============================================================================
# TEMPERATURE SCHEDULING FOR ROUTING
# =============================================================================

class RoutingTemperatureScheduler:
    """
    Warm-start routing: begin soft (high temp), gradually harden to top-k.

    Prevents premature collapse by keeping exploration high early in training.
    High temperature -> uniform routing -> diverse operator usage.
    Low temperature -> sharp routing -> specialized operator selection.

    Schedules:
    - linear: T(s) = T_init - (T_init - T_final) * min(1, (s - warmup) / anneal)
    - cosine: T(s) = T_final + 0.5 * (T_init - T_final) * (1 + cos(pi * progress))
    - exponential: T(s) = T_final + (T_init - T_final) * exp(-decay * progress)

    Example:
        scheduler = RoutingTemperatureScheduler(
            initial_temp=2.0,
            final_temp=0.1,
            warmup_steps=1000,
            anneal_steps=10000,
            schedule="cosine"
        )
        temp = scheduler.get_temperature(step=5000)
        scaled_logits = scheduler.apply_to_logits(logits, step=5000)
    """

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        schedule: str = "cosine",
    ):
        """
        Initialize the temperature scheduler.

        Args:
            initial_temp: Starting temperature (high = soft routing)
            final_temp: Ending temperature (low = sharp routing)
            warmup_steps: Steps to maintain initial temperature
            anneal_steps: Steps over which to anneal from initial to final
            schedule: One of "linear", "cosine", "exponential"
        """
        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError("Temperatures must be positive")
        if initial_temp < final_temp:
            raise ValueError("initial_temp should be >= final_temp for annealing")
        if schedule not in ("linear", "cosine", "exponential"):
            raise ValueError(f"Unknown schedule: {schedule}")

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.schedule = schedule

        # For exponential: solve T_final = T_init * exp(-decay * 1) for decay
        # decay = -ln(T_final / T_init)
        if schedule == "exponential":
            self._exp_decay = -math.log(max(final_temp / initial_temp, 1e-10))

    def get_temperature(self, step: int) -> float:
        """
        Get the temperature for a given training step.

        Args:
            step: Current training step (0-indexed)

        Returns:
            Temperature value for this step
        """
        # During warmup, maintain initial temperature
        if step < self.warmup_steps:
            return self.initial_temp

        # Compute progress through annealing (0 to 1)
        anneal_step = step - self.warmup_steps
        if self.anneal_steps <= 0:
            progress = 1.0
        else:
            progress = min(1.0, anneal_step / self.anneal_steps)

        # Already past annealing
        if progress >= 1.0:
            return self.final_temp

        # Apply schedule
        if self.schedule == "linear":
            temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress

        elif self.schedule == "cosine":
            # Cosine annealing: smooth S-curve
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1.0 + math.cos(math.pi * progress)
            )

        elif self.schedule == "exponential":
            # Exponential decay
            temp = self.final_temp + (self.initial_temp - self.final_temp) * math.exp(
                -self._exp_decay * progress
            )

        else:
            temp = self.initial_temp

        return max(temp, self.final_temp)  # Ensure we don't go below final

    def apply_to_logits(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """
        Apply temperature scaling to routing logits.

        Args:
            logits: Routing logits [batch, seq, num_experts] or [batch, num_experts]
            step: Current training step

        Returns:
            Temperature-scaled logits
        """
        temperature = self.get_temperature(step)
        return logits / temperature

    def state_dict(self) -> Dict:
        """Return scheduler state for checkpointing."""
        return {
            "initial_temp": self.initial_temp,
            "final_temp": self.final_temp,
            "warmup_steps": self.warmup_steps,
            "anneal_steps": self.anneal_steps,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.initial_temp = state["initial_temp"]
        self.final_temp = state["final_temp"]
        self.warmup_steps = state["warmup_steps"]
        self.anneal_steps = state["anneal_steps"]
        self.schedule = state["schedule"]
        if self.schedule == "exponential":
            self._exp_decay = -math.log(max(self.final_temp / self.initial_temp, 1e-10))


# =============================================================================
# LOAD-BALANCING LOSS (Switch/Expert-style)
# =============================================================================

class LoadBalancingLoss(nn.Module):
    """
    Encourage diverse noetic operator usage across tokens and layers.

    Based on Switch Transformer auxiliary loss (Fedus et al., 2021).
    Penalizes when too many tokens choose the same operator.

    Loss formulation:
        f_i = (number of tokens routed to expert i) / (total tokens)
        P_i = mean(routing_probability to expert i across all tokens)
        L_aux = num_experts * sum_i(f_i * P_i)

    This loss is minimized when:
    - f_i = 1/N for all i (uniform routing fraction)
    - P_i aligns with f_i (calibrated probabilities)

    For soft routing (no top-k), we use soft assignments:
        f_i = mean(p_i) where p_i = softmax(logits)_i

    Example:
        loss_fn = LoadBalancingLoss(num_experts=10, balance_weight=0.01)
        aux_loss = loss_fn(routing_weights)
        total_loss = main_loss + aux_loss
    """

    def __init__(
        self,
        num_experts: int = 10,
        balance_weight: float = 0.01,
        eps: float = 1e-10,
    ):
        """
        Initialize load balancing loss.

        Args:
            num_experts: Number of routing targets (10 noetic operators)
            balance_weight: Coefficient for the auxiliary loss
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.num_experts = num_experts
        self.balance_weight = balance_weight
        self.eps = eps

    def forward(
        self,
        routing_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        Args:
            routing_weights: Softmax routing probabilities
                             [batch, seq, num_experts] or [batch, num_experts]
            mask: Optional mask [batch, seq] where 1 = valid, 0 = padding

        Returns:
            Scalar auxiliary loss (already weighted)
        """
        # Handle 2D input
        if routing_weights.dim() == 2:
            routing_weights = routing_weights.unsqueeze(1)  # [batch, 1, num_experts]

        batch, seq, num_experts = routing_weights.shape

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [batch, seq, 1]
            routing_weights = routing_weights * mask
            total_tokens = mask.sum() + self.eps
        else:
            total_tokens = batch * seq

        # Compute f_i: fraction of tokens routed to each expert
        # For soft routing, this is the mean probability mass
        # Shape: [num_experts]
        f = routing_weights.sum(dim=(0, 1)) / total_tokens

        # Compute P_i: mean routing probability to each expert
        # This should already sum to ~1 per token, so mean over tokens
        # Shape: [num_experts]
        P = routing_weights.mean(dim=(0, 1))

        # Load balancing loss: N * sum(f_i * P_i)
        # Minimum is 1/N when uniform, so we subtract 1 for interpretability
        aux_loss = self.num_experts * (f * P).sum()

        # Apply weight
        return self.balance_weight * aux_loss

    def compute_metrics(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute interpretable load balancing metrics.

        Args:
            routing_weights: [batch, seq, num_experts]

        Returns:
            Dictionary with balance metrics
        """
        if routing_weights.dim() == 2:
            routing_weights = routing_weights.unsqueeze(1)

        with torch.no_grad():
            # Mean probability mass per expert
            expert_load = routing_weights.mean(dim=(0, 1))  # [num_experts]

            # Coefficient of variation: std/mean (0 = perfectly balanced)
            cv = expert_load.std() / (expert_load.mean() + 1e-10)

            # Max/min ratio (1 = balanced)
            load_max = expert_load.max()
            load_min = expert_load.min() + 1e-10
            max_min_ratio = (load_max / load_min).item()

            # Top expert dominance (what fraction goes to most popular)
            top_dominance = load_max.item()

            return {
                "load_cv": cv.item(),
                "load_max_min_ratio": max_min_ratio,
                "top_expert_dominance": top_dominance,
                "per_expert_load": expert_load.tolist(),
            }


# =============================================================================
# ROUTING NOISE INJECTION
# =============================================================================

class NoisyRouting(nn.Module):
    """
    Add Gumbel, Gaussian, or uniform noise to routing to prevent brittle collapse.

    Noise is annealed down over training:
    - Early: high noise -> exploration of all operators
    - Late: low noise -> stable, learned routing

    Noise types:
    - gumbel: -log(-log(U)), U ~ Uniform(0,1) - Gumbel-Softmax style
    - gaussian: N(0, sigma)
    - uniform: U(-a, a)

    Example:
        noisy = NoisyRouting(
            noise_type="gumbel",
            initial_noise=0.5,
            final_noise=0.0,
            anneal_steps=10000
        )
        noisy_logits = noisy.add_noise(logits, step=5000)
    """

    def __init__(
        self,
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        anneal_steps: int = 10000,
        eps: float = 1e-10,
    ):
        """
        Initialize noisy routing.

        Args:
            noise_type: One of "gumbel", "gaussian", "uniform"
            initial_noise: Starting noise scale
            final_noise: Ending noise scale (typically 0)
            anneal_steps: Steps to anneal from initial to final
            eps: Small constant for Gumbel stability
        """
        super().__init__()

        if noise_type not in ("gumbel", "gaussian", "uniform"):
            raise ValueError(f"Unknown noise_type: {noise_type}")

        self.noise_type = noise_type
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.anneal_steps = anneal_steps
        self.eps = eps

    def get_noise_scale(self, step: int) -> float:
        """
        Get noise scale for current step (linear annealing).

        Args:
            step: Current training step

        Returns:
            Noise scale for this step
        """
        if self.anneal_steps <= 0:
            return self.final_noise

        progress = min(1.0, step / self.anneal_steps)
        return self.initial_noise + (self.final_noise - self.initial_noise) * progress

    def gumbel_noise(
        self,
        shape: Tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1).

        The Gumbel distribution has the property that if G_i are i.i.d. Gumbel,
        then argmax(log(p_i) + G_i) ~ Categorical(p).

        Args:
            shape: Output tensor shape
            device: Target device
            dtype: Target dtype

        Returns:
            Gumbel noise tensor
        """
        u = torch.rand(shape, device=device, dtype=dtype)
        # Clamp for numerical stability
        u = u.clamp(self.eps, 1.0 - self.eps)
        return -torch.log(-torch.log(u))

    def gaussian_noise(
        self,
        shape: Tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate standard Gaussian noise."""
        return torch.randn(shape, device=device, dtype=dtype)

    def uniform_noise(
        self,
        shape: Tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate uniform noise in [-1, 1]."""
        return 2.0 * torch.rand(shape, device=device, dtype=dtype) - 1.0

    def add_noise(
        self,
        logits: torch.Tensor,
        step: int,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Add annealed noise to routing logits.

        Args:
            logits: Routing logits [batch, seq, num_experts] or [batch, num_experts]
            step: Current training step
            training: Whether in training mode (no noise in eval)

        Returns:
            Noisy logits
        """
        if not training:
            return logits

        scale = self.get_noise_scale(step)

        if scale <= 0:
            return logits

        # Generate noise
        if self.noise_type == "gumbel":
            noise = self.gumbel_noise(logits.shape, logits.device, logits.dtype)
        elif self.noise_type == "gaussian":
            noise = self.gaussian_noise(logits.shape, logits.device, logits.dtype)
        elif self.noise_type == "uniform":
            noise = self.uniform_noise(logits.shape, logits.device, logits.dtype)
        else:
            return logits

        return logits + scale * noise

    def forward(
        self,
        logits: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Forward pass - alias for add_noise during training."""
        return self.add_noise(logits, step, training=self.training)


# =============================================================================
# ROUTING ENTROPY MONITOR
# =============================================================================

class RoutingEntropyMonitor:
    """
    Track routing entropy and alert if it drops below floor.

    Low entropy indicates routing collapse:
    - Entropy -> 0: All tokens route to same expert (collapsed)
    - Entropy -> log(N): Uniform routing (max diversity)

    For 10 noetic operators:
    - Max entropy = log(10) ~= 2.30 bits
    - Healthy floor ~= 0.5 bits (maintains some specialization)

    Example:
        monitor = RoutingEntropyMonitor(entropy_floor=0.5, window_size=100)
        metrics = monitor.update(routing_weights)
        if monitor.is_collapsing():
            # Increase temperature, add noise, etc.
    """

    def __init__(
        self,
        entropy_floor: float = 0.5,
        window_size: int = 100,
        num_experts: int = 10,
        eps: float = 1e-10,
    ):
        """
        Initialize entropy monitor.

        Args:
            entropy_floor: Minimum acceptable entropy (bits)
            window_size: Number of samples for moving average
            num_experts: Number of routing targets (for max entropy)
            eps: Small constant for log stability
        """
        self.entropy_floor = entropy_floor
        self.window_size = window_size
        self.num_experts = num_experts
        self.eps = eps

        # Maximum possible entropy
        self.max_entropy = math.log(num_experts)

        # Moving average window
        self.entropy_history: deque = deque(maxlen=window_size)
        self.collapse_count = 0
        self.total_updates = 0

    def compute_entropy(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of routing distribution.

        Args:
            routing_weights: [batch, seq, num_experts] or [batch, num_experts]

        Returns:
            Entropy tensor [batch, seq] or [batch]
        """
        # Clamp for numerical stability
        p = routing_weights.clamp(self.eps, 1.0)
        # Shannon entropy: H = -sum(p * log(p))
        entropy = -(p * torch.log(p)).sum(dim=-1)
        return entropy

    def update(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """
        Update monitor with new routing weights and compute metrics.

        Args:
            routing_weights: [batch, seq, num_experts]

        Returns:
            Dictionary with entropy metrics
        """
        with torch.no_grad():
            # Handle 2D input
            if routing_weights.dim() == 2:
                routing_weights = routing_weights.unsqueeze(1)

            # Compute entropy per position
            entropy = self.compute_entropy(routing_weights)

            # Statistics
            mean_entropy = entropy.mean().item()
            min_entropy = entropy.min().item()
            max_entropy = entropy.max().item()
            std_entropy = entropy.std().item()

            # Update history
            self.entropy_history.append(mean_entropy)
            self.total_updates += 1

            # Check for collapse
            if mean_entropy < self.entropy_floor:
                self.collapse_count += 1

            # Moving average
            moving_avg = sum(self.entropy_history) / len(self.entropy_history)

            # Normalized entropy (0-1 scale)
            normalized = mean_entropy / self.max_entropy if self.max_entropy > 0 else 0

            return {
                "routing_entropy_mean": mean_entropy,
                "routing_entropy_min": min_entropy,
                "routing_entropy_max": max_entropy,
                "routing_entropy_std": std_entropy,
                "routing_entropy_normalized": normalized,
                "routing_entropy_moving_avg": moving_avg,
                "routing_entropy_floor": self.entropy_floor,
                "routing_collapse_rate": self.collapse_count / max(self.total_updates, 1),
                "routing_is_collapsing": self.is_collapsing(),
            }

    def is_collapsing(self) -> bool:
        """
        Check if routing is collapsing based on moving average.

        Returns:
            True if moving average entropy is below floor
        """
        if len(self.entropy_history) == 0:
            return False

        moving_avg = sum(self.entropy_history) / len(self.entropy_history)
        return moving_avg < self.entropy_floor

    def reset(self):
        """Reset monitor state."""
        self.entropy_history.clear()
        self.collapse_count = 0
        self.total_updates = 0

    def get_collapse_severity(self) -> float:
        """
        Get severity of collapse (0 = fine, 1 = severe).

        Returns:
            Collapse severity score
        """
        if len(self.entropy_history) == 0:
            return 0.0

        moving_avg = sum(self.entropy_history) / len(self.entropy_history)

        if moving_avg >= self.entropy_floor:
            return 0.0

        # Severity increases as entropy approaches 0
        severity = 1.0 - (moving_avg / self.entropy_floor)
        return min(1.0, max(0.0, severity))


# =============================================================================
# STABLE ROUTING WRAPPER
# =============================================================================

class StableRouting(nn.Module):
    """
    Wrapper that applies all stability mitigations to routing.

    Combines:
    1. Temperature scheduling - gradual transition from soft to sharp routing
    2. Noise injection - Gumbel/Gaussian exploration
    3. Load balancing loss - penalize expert concentration
    4. Entropy monitoring - detect and flag collapse

    Usage:
        config = StableRoutingConfig(...)
        stable = StableRouting(base_router, config)

        weights, aux_loss, metrics = stable(x, step=current_step)
        total_loss = main_loss + aux_loss
    """

    def __init__(
        self,
        base_router: nn.Module,
        num_experts: int = 10,
        # Temperature scheduling
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        temp_schedule: str = "cosine",
        # Noise injection
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        noise_anneal_steps: int = 10000,
        # Load balancing
        balance_weight: float = 0.01,
        # Entropy monitoring
        entropy_floor: float = 0.5,
        entropy_window: int = 100,
    ):
        """
        Initialize stable routing wrapper.

        Args:
            base_router: The underlying router module that produces logits
            num_experts: Number of routing targets
            initial_temp: Starting temperature for softmax
            final_temp: Ending temperature
            warmup_steps: Steps at initial temperature
            anneal_steps: Steps to anneal temperature
            temp_schedule: "linear", "cosine", or "exponential"
            noise_type: "gumbel", "gaussian", or "uniform"
            initial_noise: Starting noise scale
            final_noise: Ending noise scale
            noise_anneal_steps: Steps to anneal noise
            balance_weight: Weight for load balancing loss
            entropy_floor: Minimum acceptable entropy
            entropy_window: Window size for entropy moving average
        """
        super().__init__()

        self.base_router = base_router
        self.num_experts = num_experts

        # Temperature scheduler
        self.temp_scheduler = RoutingTemperatureScheduler(
            initial_temp=initial_temp,
            final_temp=final_temp,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            schedule=temp_schedule,
        )

        # Noise injection
        self.noise = NoisyRouting(
            noise_type=noise_type,
            initial_noise=initial_noise,
            final_noise=final_noise,
            anneal_steps=noise_anneal_steps,
        )

        # Load balancing loss
        self.load_balance = LoadBalancingLoss(
            num_experts=num_experts,
            balance_weight=balance_weight,
        )

        # Entropy monitor
        self.entropy_monitor = RoutingEntropyMonitor(
            entropy_floor=entropy_floor,
            window_size=entropy_window,
            num_experts=num_experts,
        )

    def forward(
        self,
        x: torch.Tensor,
        step: int,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Apply stable routing with all mitigations.

        Args:
            x: Input to base router
            step: Current training step
            mask: Optional padding mask [batch, seq]
            return_logits: If True, also return raw logits

        Returns:
            - routing_weights: [batch, seq, num_experts] probabilities
            - aux_loss: Load balancing loss (scalar)
            - metrics: Dictionary with stability metrics
        """
        # Get base logits from router
        logits = self.base_router(x)  # [batch, seq, num_experts] or [batch, num_experts]

        # Step 1: Add noise (during training)
        noisy_logits = self.noise.add_noise(logits, step, training=self.training)

        # Step 2: Apply temperature scaling
        scaled_logits = self.temp_scheduler.apply_to_logits(noisy_logits, step)

        # Step 3: Compute routing weights
        weights = F.softmax(scaled_logits, dim=-1)

        # Step 4: Compute load balancing loss
        aux_loss = self.load_balance(weights, mask)

        # Step 5: Update entropy monitor and get metrics
        entropy_metrics = self.entropy_monitor.update(weights)

        # Combine all metrics
        metrics = {
            **entropy_metrics,
            "routing_temperature": self.temp_scheduler.get_temperature(step),
            "routing_noise_scale": self.noise.get_noise_scale(step),
            "routing_aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            **self.load_balance.compute_metrics(weights),
        }

        if return_logits:
            return weights, aux_loss, metrics, logits

        return weights, aux_loss, metrics

    def get_current_settings(self, step: int) -> Dict[str, float]:
        """Get current scheduler settings for logging."""
        return {
            "temperature": self.temp_scheduler.get_temperature(step),
            "noise_scale": self.noise.get_noise_scale(step),
            "entropy_floor": self.entropy_monitor.entropy_floor,
            "is_collapsing": self.entropy_monitor.is_collapsing(),
            "collapse_severity": self.entropy_monitor.get_collapse_severity(),
        }


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

class StableRoutingConfig:
    """Configuration container for StableRouting."""

    def __init__(
        self,
        num_experts: int = 10,
        # Temperature
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        warmup_steps: int = 1000,
        anneal_steps: int = 10000,
        temp_schedule: str = "cosine",
        # Noise
        noise_type: str = "gumbel",
        initial_noise: float = 0.5,
        final_noise: float = 0.0,
        noise_anneal_steps: int = 10000,
        # Balance
        balance_weight: float = 0.01,
        # Entropy
        entropy_floor: float = 0.5,
        entropy_window: int = 100,
    ):
        self.num_experts = num_experts
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.temp_schedule = temp_schedule
        self.noise_type = noise_type
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.noise_anneal_steps = noise_anneal_steps
        self.balance_weight = balance_weight
        self.entropy_floor = entropy_floor
        self.entropy_window = entropy_window

    def to_dict(self) -> Dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "StableRoutingConfig":
        return cls(**d)


def create_stable_routing(
    base_router: nn.Module,
    config: Optional[StableRoutingConfig] = None,
    **kwargs,
) -> StableRouting:
    """
    Factory function to create StableRouting with config.

    Args:
        base_router: The underlying router module
        config: Optional configuration object
        **kwargs: Override config values

    Returns:
        StableRouting instance
    """
    if config is None:
        config = StableRoutingConfig()

    # Override with kwargs
    params = config.to_dict()
    params.update(kwargs)

    return StableRouting(base_router, **params)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TKS ROUTING STABILITY - Demo")
    print("=" * 70)

    # Create a simple base router
    class SimpleRouter(nn.Module):
        def __init__(self, dim: int, num_experts: int):
            super().__init__()
            self.proj = nn.Linear(dim, num_experts)

        def forward(self, x):
            return self.proj(x)

    dim, num_experts = 40, 10
    batch, seq = 4, 16

    base_router = SimpleRouter(dim, num_experts)

    # Create stable routing with default config
    stable = StableRouting(
        base_router,
        num_experts=num_experts,
        initial_temp=2.0,
        final_temp=0.1,
        warmup_steps=100,
        anneal_steps=1000,
        temp_schedule="cosine",
        noise_type="gumbel",
        initial_noise=0.5,
        final_noise=0.0,
        noise_anneal_steps=1000,
        balance_weight=0.01,
        entropy_floor=0.5,
        entropy_window=50,
    )

    print(f"\nBase router params: {sum(p.numel() for p in base_router.parameters()):,}")
    print(f"Stable routing components initialized.")

    # Test input
    x = torch.randn(batch, seq, dim)

    # Simulate training loop at different steps
    print("\n" + "-" * 70)
    print("Simulating training at different steps...")
    print("-" * 70)

    stable.train()
    for step in [0, 100, 500, 1000, 2000]:
        weights, aux_loss, metrics = stable(x, step=step)

        print(f"\nStep {step:>5}:")
        print(f"  Temperature: {metrics['routing_temperature']:.3f}")
        print(f"  Noise scale: {metrics['routing_noise_scale']:.3f}")
        print(f"  Entropy (mean): {metrics['routing_entropy_mean']:.3f} bits")
        print(f"  Entropy (normalized): {metrics['routing_entropy_normalized']:.2%}")
        print(f"  Aux loss: {metrics['routing_aux_loss']:.4f}")
        print(f"  Load CV: {metrics['load_cv']:.3f}")
        print(f"  Is collapsing: {metrics['routing_is_collapsing']}")

    # Test collapse detection
    print("\n" + "-" * 70)
    print("Testing collapse detection...")
    print("-" * 70)

    # Simulate collapsed routing (all mass on one expert)
    collapsed_weights = torch.zeros(batch, seq, num_experts)
    collapsed_weights[..., 0] = 1.0  # All to expert 0

    monitor = RoutingEntropyMonitor(entropy_floor=0.5, window_size=10)

    for i in range(15):
        metrics = monitor.update(collapsed_weights)

    print(f"After 15 collapsed updates:")
    print(f"  Moving avg entropy: {metrics['routing_entropy_moving_avg']:.4f}")
    print(f"  Is collapsing: {monitor.is_collapsing()}")
    print(f"  Collapse severity: {monitor.get_collapse_severity():.2%}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
