"""
TKS Provider Fallback - Automatic provider failover for TKS wrapper.

This module provides automatic failover between AI providers:
- Primary provider fails → automatically try secondary
- Quota exceeded → switch providers
- Seamless redundancy for uninterrupted service
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class ProviderHealth:
    """Health status for a provider."""
    name: str
    status: ProviderStatus
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    cooldown_until: Optional[float] = None


class TKSProviderFallback:
    """
    Automatic failover between TKS AI providers.

    Usage:
        fallback = TKSProviderFallback(
            providers=["gemini", "claude"],
            primary="gemini"
        )

        # Will auto-fallback if primary fails
        result = fallback.execute(
            lambda provider: tks_wrapper.analyze(query, provider=provider)
        )

        # Check provider health
        health = fallback.get_health()
    """

    def __init__(
        self,
        providers: List[str],
        primary: Optional[str] = None,
        max_retries: int = 2,
        cooldown_seconds: float = 60.0,
        failure_threshold: int = 3
    ):
        """
        Initialize fallback manager.

        Args:
            providers: List of available providers
            primary: Preferred primary provider
            max_retries: Max retries per provider
            cooldown_seconds: Cooldown after failures
            failure_threshold: Failures before cooldown
        """
        self.providers = providers
        self.primary = primary or (providers[0] if providers else None)
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.failure_threshold = failure_threshold

        # Initialize health tracking
        self.health: Dict[str, ProviderHealth] = {}
        for p in providers:
            self.health[p] = ProviderHealth(
                name=p,
                status=ProviderStatus.HEALTHY
            )

        # Callbacks
        self.on_fallback: Optional[Callable[[str, str, Exception], None]] = None
        self.on_recovery: Optional[Callable[[str], None]] = None

    def execute(
        self,
        func: Callable[[str], Any],
        skip_providers: Optional[List[str]] = None
    ) -> Any:
        """
        Execute function with automatic fallback.

        Args:
            func: Function that takes provider name and returns result
            skip_providers: Providers to skip

        Returns:
            Result from successful provider

        Raises:
            Exception: If all providers fail
        """
        skip = set(skip_providers or [])
        errors = []

        # Try providers in order
        for provider in self._get_provider_order():
            if provider in skip:
                continue

            health = self.health[provider]

            # Check cooldown
            if health.cooldown_until and time.time() < health.cooldown_until:
                logger.debug(f"Provider {provider} in cooldown")
                continue

            # Try with retries
            for attempt in range(self.max_retries + 1):
                try:
                    result = func(provider)
                    self._record_success(provider)
                    return result

                except Exception as e:
                    error_msg = str(e)
                    errors.append((provider, error_msg))

                    # Check for specific error types
                    if self._is_quota_error(e):
                        self._record_quota_exceeded(provider)
                        break  # Don't retry quota errors
                    elif self._is_rate_limit(e):
                        self._record_rate_limit(provider)
                        time.sleep(1.0 * (attempt + 1))  # Backoff
                    else:
                        self._record_failure(provider, e)

                        if attempt == self.max_retries:
                            break

            # Trigger fallback callback
            if self.on_fallback and len(errors) > 0:
                next_provider = self._get_next_available(provider)
                if next_provider:
                    self.on_fallback(provider, next_provider, errors[-1][1])

        # All providers failed
        error_summary = "; ".join([f"{p}: {e}" for p, e in errors])
        raise Exception(f"All providers failed: {error_summary}")

    def _get_provider_order(self) -> List[str]:
        """Get providers sorted by preference and health."""
        available = []
        cooldown = []

        for p in self.providers:
            health = self.health[p]

            if health.cooldown_until and time.time() < health.cooldown_until:
                cooldown.append(p)
            elif health.status == ProviderStatus.QUOTA_EXCEEDED:
                cooldown.append(p)
            else:
                available.append(p)

        # Sort available by health (healthy first, then by failures)
        available.sort(key=lambda p: (
            0 if self.health[p].status == ProviderStatus.HEALTHY else 1,
            self.health[p].consecutive_failures
        ))

        # Put primary first if healthy
        if self.primary in available:
            available.remove(self.primary)
            available.insert(0, self.primary)

        return available + cooldown

    def _get_next_available(self, current: str) -> Optional[str]:
        """Get next available provider after current."""
        order = self._get_provider_order()
        try:
            idx = order.index(current)
            if idx + 1 < len(order):
                return order[idx + 1]
        except ValueError:
            pass
        return None

    def _record_success(self, provider: str) -> None:
        """Record successful request."""
        health = self.health[provider]
        health.last_success = time.time()
        health.consecutive_failures = 0
        health.total_requests += 1

        # Recover from degraded state
        if health.status != ProviderStatus.HEALTHY:
            health.status = ProviderStatus.HEALTHY
            health.cooldown_until = None
            if self.on_recovery:
                self.on_recovery(provider)

    def _record_failure(self, provider: str, error: Exception) -> None:
        """Record failed request."""
        health = self.health[provider]
        health.last_failure = time.time()
        health.consecutive_failures += 1
        health.total_failures += 1
        health.total_requests += 1

        # Check threshold
        if health.consecutive_failures >= self.failure_threshold:
            health.status = ProviderStatus.UNAVAILABLE
            health.cooldown_until = time.time() + self.cooldown_seconds
            logger.warning(f"Provider {provider} entering cooldown")
        else:
            health.status = ProviderStatus.DEGRADED

    def _record_rate_limit(self, provider: str) -> None:
        """Record rate limit error."""
        health = self.health[provider]
        health.status = ProviderStatus.RATE_LIMITED
        health.last_failure = time.time()
        health.consecutive_failures += 1

        # Short cooldown for rate limits
        health.cooldown_until = time.time() + 30.0

    def _record_quota_exceeded(self, provider: str) -> None:
        """Record quota exceeded error."""
        health = self.health[provider]
        health.status = ProviderStatus.QUOTA_EXCEEDED
        health.last_failure = time.time()

        # Long cooldown for quota (1 hour)
        health.cooldown_until = time.time() + 3600.0
        logger.warning(f"Provider {provider} quota exceeded")

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is quota-related."""
        error_str = str(error).lower()
        return any(kw in error_str for kw in [
            "quota", "exceeded", "limit reached", "billing"
        ])

    def _is_rate_limit(self, error: Exception) -> bool:
        """Check if error is rate limit."""
        error_str = str(error).lower()
        return any(kw in error_str for kw in [
            "rate limit", "too many requests", "429", "throttl"
        ])

    def get_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers."""
        return {
            name: {
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "total_failures": health.total_failures,
                "success_rate": (
                    (health.total_requests - health.total_failures) / health.total_requests
                    if health.total_requests > 0 else 1.0
                ),
                "in_cooldown": (
                    health.cooldown_until is not None and
                    time.time() < health.cooldown_until
                )
            }
            for name, health in self.health.items()
        }

    def reset_provider(self, provider: str) -> None:
        """Reset a provider's health status."""
        if provider in self.health:
            self.health[provider] = ProviderHealth(
                name=provider,
                status=ProviderStatus.HEALTHY
            )

    def set_primary(self, provider: str) -> None:
        """Set primary provider."""
        if provider in self.providers:
            self.primary = provider


class TKSFallbackWrapper:
    """
    High-level wrapper combining TKSWrapper with fallback.

    Usage:
        wrapper = TKSFallbackWrapper()
        result = wrapper.analyze("My situation")
        # Automatically uses fallback if needed
    """

    def __init__(
        self,
        tks_wrapper=None,
        providers: List[str] = None,
        primary: str = "gemini"
    ):
        """
        Initialize wrapper with fallback.

        Args:
            tks_wrapper: Optional TKSWrapper instance
            providers: List of providers
            primary: Primary provider
        """
        self.wrapper = tks_wrapper
        self.providers = providers or ["gemini", "claude"]
        self.fallback = TKSProviderFallback(
            providers=self.providers,
            primary=primary
        )

        # Set up logging callback
        self.fallback.on_fallback = self._log_fallback

    def _log_fallback(self, from_p: str, to_p: str, error: str) -> None:
        """Log fallback event."""
        logger.info(f"Falling back from {from_p} to {to_p}: {error}")

    def analyze(self, query: str, **kwargs) -> Any:
        """
        Analyze with automatic fallback.

        Args:
            query: Query to analyze
            **kwargs: Additional arguments

        Returns:
            Analysis result
        """
        def _do_analyze(provider: str):
            if self.wrapper is None:
                # Import and create wrapper
                from tks_wrapper import TKSWrapper
                self.wrapper = TKSWrapper()

            return self.wrapper.analyze(query, provider=provider, **kwargs)

        return self.fallback.execute(_do_analyze)

    def get_status(self) -> Dict[str, Any]:
        """Get wrapper status including provider health."""
        return {
            "providers": self.providers,
            "primary": self.fallback.primary,
            "health": self.fallback.get_health()
        }


def test_fallback():
    """Test fallback functionality."""
    # Simulate providers
    call_count = {"gemini": 0, "claude": 0}

    def mock_call(provider: str):
        call_count[provider] += 1

        # Simulate gemini failing first 2 times
        if provider == "gemini" and call_count["gemini"] <= 2:
            raise Exception("Simulated gemini failure")

        return f"Success from {provider}"

    fallback = TKSProviderFallback(
        providers=["gemini", "claude"],
        primary="gemini",
        max_retries=1
    )

    # Should fall back to claude
    result = fallback.execute(mock_call)
    print(f"Result: {result}")
    print(f"Call counts: {call_count}")
    print(f"Health: {fallback.get_health()}")


if __name__ == "__main__":
    test_fallback()
