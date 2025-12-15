"""
LLM Provider Adapters for Multi-LLM Teacher

Provides unified interface for querying multiple LLM backends:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google Gemini (Gemini Pro, Gemini Ultra)
    - Local models (Ollama, vLLM, etc.)
    - Mock provider for testing
"""

import os
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


# ==============================================================================
# BASE PROVIDER INTERFACE
# ==============================================================================

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    model: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.text) > 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.config = kwargs
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_enabled = kwargs.get("cache_enabled", True)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Make actual API call. Subclasses implement this."""
        pass

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> LLMResponse:
        """Query the LLM with optional caching."""
        cache_key = self._get_cache_key(prompt, system_prompt)

        if use_cache and self._cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.metadata["from_cache"] = True
            return cached

        start_time = time.time()
        try:
            response = self._call_api(prompt, system_prompt)
            response.latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            response = LLMResponse(
                text="",
                model=self.model,
                provider=self.provider_name,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

        if response.success and self._cache_enabled:
            self._cache[cache_key] = response

        return response

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate cache key from prompt content."""
        content = f"{self.provider_name}:{self.model}:{system_prompt or ''}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()


# ==============================================================================
# OPENAI PROVIDER
# ==============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(model, api_key, **kwargs)

        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")

        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call OpenAI API."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            provider=self.provider_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id
            }
        )


# ==============================================================================
# ANTHROPIC PROVIDER
# ==============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude models)."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(model, api_key, **kwargs)

        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call Anthropic API."""
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_name,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            metadata={
                "stop_reason": response.stop_reason,
                "response_id": response.id
            }
        )


# ==============================================================================
# GEMINI PROVIDER (Google)
# ==============================================================================

class GeminiProvider(LLMProvider):
    """Google Gemini API provider (Gemini Pro, Gemini Ultra, etc.)."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        super().__init__(model, api_key, **kwargs)

        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

        self._client = None
        self._generation_config = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)

                # Configure generation settings
                self._generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call Gemini API."""
        client = self._get_client()

        # Gemini handles system prompts by prepending to the conversation
        # or using system_instruction parameter (model-dependent)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        try:
            response = client.generate_content(
                full_prompt,
                generation_config=self._generation_config,
            )

            # Extract text from response
            text = ""
            if response.text:
                text = response.text
            elif response.parts:
                text = "".join(part.text for part in response.parts if hasattr(part, "text"))

            # Get token counts if available
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = (
                    getattr(response.usage_metadata, "prompt_token_count", 0) +
                    getattr(response.usage_metadata, "candidates_token_count", 0)
                )

            return LLMResponse(
                text=text,
                model=self.model,
                provider=self.provider_name,
                tokens_used=tokens_used,
                metadata={
                    "finish_reason": getattr(response.candidates[0], "finish_reason", None)
                    if response.candidates else None,
                    "safety_ratings": [
                        {"category": str(r.category), "probability": str(r.probability)}
                        for r in getattr(response.candidates[0], "safety_ratings", [])
                    ] if response.candidates else []
                }
            )

        except Exception as e:
            # Handle Gemini-specific errors
            error_msg = str(e)

            # Check for safety blocks
            if "blocked" in error_msg.lower() or "safety" in error_msg.lower():
                return LLMResponse(
                    text="",
                    model=self.model,
                    provider=self.provider_name,
                    error=f"Content blocked by safety filters: {error_msg}"
                )

            raise


# ==============================================================================
# LOCAL PROVIDER (Ollama, vLLM, etc.)
# ==============================================================================

class LocalProvider(LLMProvider):
    """Local LLM provider for self-hosted models (Ollama, vLLM, etc.)."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        provider_type: str = "ollama",
        **kwargs
    ):
        super().__init__(model, api_key=None, **kwargs)

        self.base_url = base_url
        self.provider_type = provider_type
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1024)

    @property
    def provider_name(self) -> str:
        return f"local_{self.provider_type}"

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call local LLM API."""
        import requests

        if self.provider_type == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.provider_type == "vllm":
            return self._call_vllm(prompt, system_prompt)
        else:
            # Generic OpenAI-compatible endpoint
            return self._call_openai_compatible(prompt, system_prompt)

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call Ollama API."""
        import requests

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            provider=self.provider_name,
            tokens_used=data.get("eval_count", 0),
            metadata={
                "total_duration": data.get("total_duration", 0),
                "load_duration": data.get("load_duration", 0)
            }
        )

    def _call_vllm(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call vLLM OpenAI-compatible API."""
        return self._call_openai_compatible(prompt, system_prompt)

    def _call_openai_compatible(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call OpenAI-compatible API (works with vLLM, LM Studio, etc.)."""
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            provider=self.provider_name,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            metadata={"response_id": data.get("id", "")}
        )


# ==============================================================================
# MOCK PROVIDER (for testing)
# ==============================================================================

class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""

    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(model, api_key=None, **kwargs)
        self.responses = responses or {}
        self.default_response = kwargs.get(
            "default_response",
            "This is a mock response for TKS equation interpretation."
        )
        self.call_count = 0
        self.call_history: List[Dict] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Return mock response."""
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "call_number": self.call_count
        })

        # Check for matching response
        for key, response in self.responses.items():
            if key in prompt:
                text = response
                break
        else:
            text = self.default_response

        # Simulate some latency
        time.sleep(0.01)

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_name,
            tokens_used=len(text.split()),
            metadata={"call_number": self.call_count}
        )

    def add_response(self, trigger: str, response: str):
        """Add a trigger-response pair."""
        self.responses[trigger] = response

    def reset(self):
        """Reset call history and count."""
        self.call_count = 0
        self.call_history.clear()


# ==============================================================================
# PROVIDER FACTORY
# ==============================================================================

def create_provider(
    provider_type: str,
    model: str,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: One of 'openai', 'anthropic', 'gemini', 'local', 'mock'
        model: Model name/identifier
        **kwargs: Additional provider configuration

    Returns:
        LLMProvider instance

    Examples:
        # OpenAI
        provider = create_provider("openai", "gpt-4")

        # Anthropic
        provider = create_provider("anthropic", "claude-3-sonnet-20240229")

        # Google Gemini
        provider = create_provider("gemini", "gemini-1.5-pro")

        # Local (Ollama)
        provider = create_provider("local", "llama2", base_url="http://localhost:11434")
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,  # Alias for gemini
        "local": LocalProvider,
        "mock": MockProvider
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(providers.keys())}")

    return providers[provider_type](model=model, **kwargs)
