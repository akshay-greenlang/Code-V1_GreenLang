# -*- coding: utf-8 -*-
"""
Tier 1: Ollama Local LLM Provider

Real LLM intelligence WITHOUT any API key or cloud dependency.
Uses Ollama to run local models (Llama 3, Mistral, Phi-3, etc.)

For open-source developers:
- No API key required
- No cloud dependency
- Data stays on your machine (privacy-first)
- Free to use (just needs local compute)
- Works offline

Setup:
    1. Install Ollama: https://ollama.ai/download
    2. Pull a model: ollama pull llama3.2
    3. GreenLang auto-detects and uses it!

Supported models (recommended for GreenLang):
    - llama3.2 (8B) - Best balance of quality/speed
    - mistral (7B) - Good for reasoning
    - phi3 (3.8B) - Fast, good for explanations
    - codellama (7B) - Good for technical analysis
    - mixtral (8x7B) - Highest quality (needs more RAM)

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready - Local Intelligence Tier
"""

from __future__ import annotations
import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Mapping, Optional
import httpx

from greenlang.agents.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.agents.intelligence.schemas.messages import ChatMessage, Role
from greenlang.agents.intelligence.schemas.tools import ToolDef
from greenlang.agents.intelligence.schemas.responses import (
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
)
from greenlang.agents.intelligence.runtime.budget import Budget

logger = logging.getLogger(__name__)


# Default Ollama configuration
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "default": "llama3.2",
    "fast": "phi3",
    "quality": "mixtral",
    "reasoning": "mistral",
    "code": "codellama",
}

# System prompts for GreenLang domain
GREENLANG_SYSTEM_PROMPT = """You are GreenLang Intelligence, an expert AI assistant for climate and sustainability analysis.

Your capabilities:
- Carbon footprint calculations and explanations
- Energy efficiency analysis
- Process heat optimization
- Regulatory compliance (GHG Protocol, CSRD, EPA, NFPA)
- Emission factor lookups and validation
- Sustainability recommendations

Guidelines:
- Be factual and precise with numbers
- Always cite sources (EPA, IPCC, ISO standards) when referencing emission factors
- Provide actionable recommendations
- Consider regulatory context (CSRD, SBTi, GHG Protocol)
- Never hallucinate emission factors - use standard values or say "verify with authoritative source"
- Format outputs clearly with markdown when appropriate

You are running locally via Ollama - user data stays private and secure."""


class OllamaProvider(LLMProvider):
    """
    Tier 1: Ollama Local LLM Provider

    Provides real LLM intelligence using locally-running models via Ollama.
    No API key required - just install Ollama and pull a model.

    Features:
    - True LLM reasoning (not templates)
    - No API costs
    - Data stays local (privacy)
    - Works offline
    - Multiple model options

    Requirements:
    - Ollama installed (https://ollama.ai)
    - A model pulled (e.g., `ollama pull llama3.2`)
    - ~8GB RAM for 7B models, ~16GB for larger

    Usage:
        # Auto-detected if Ollama is running
        from greenlang.intelligence import create_provider
        provider = create_provider()  # Uses Ollama if available

        # Or explicitly request
        provider = create_provider(model="ollama:llama3.2")

    Environment Variables:
        OLLAMA_HOST: Ollama API URL (default: http://localhost:11434)
        OLLAMA_MODEL: Default model (default: llama3.2)
    """

    def __init__(
        self,
        config: LLMProviderConfig,
        host: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize Ollama Provider.

        Args:
            config: Provider configuration
            host: Ollama API host (default: http://localhost:11434)
            model: Model name (default: llama3.2)
        """
        super().__init__(config)

        self.host = host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.model = model or config.model.replace("ollama:", "") or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

        # Validate model name
        if self.model.startswith("ollama:"):
            self.model = self.model[7:]

        self._client = httpx.AsyncClient(timeout=120.0)
        self._is_available: Optional[bool] = None

        logger.info(
            f"Initialized OllamaProvider (Tier 1) - "
            f"host={self.host}, model={self.model}"
        )

    @property
    def capabilities(self) -> LLMCapabilities:
        """Return provider capabilities."""
        return LLMCapabilities(
            function_calling=False,  # Basic Ollama doesn't support tools
            json_schema_mode=True,   # Can request JSON output
            max_output_tokens=4096,
            context_window_tokens=8192,  # Varies by model
        )

    async def is_available(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if Ollama is accessible and model is pulled
        """
        if self._is_available is not None:
            return self._is_available

        try:
            response = await self._client.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]

                if self.model in models or f"{self.model}:latest" in [m["name"] for m in data.get("models", [])]:
                    self._is_available = True
                    logger.info(f"Ollama available with model: {self.model}")
                else:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available models: {models}. "
                        f"Run: ollama pull {self.model}"
                    )
                    self._is_available = False
            else:
                self._is_available = False

        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._is_available = False

        return self._is_available

    def _build_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessage to Ollama format."""
        ollama_messages = []

        # Add GreenLang system prompt
        ollama_messages.append({
            "role": "system",
            "content": GREENLANG_SYSTEM_PROMPT
        })

        for msg in messages:
            role = "user" if msg.role == Role.user else "assistant"
            if msg.role == Role.system:
                role = "system"

            ollama_messages.append({
                "role": role,
                "content": msg.content or ""
            })

        return ollama_messages

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolDef]] = None,
        json_schema: Optional[Any] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        """
        Execute chat completion via Ollama.

        Args:
            messages: Conversation history
            tools: Not supported by basic Ollama
            json_schema: Request JSON output format
            budget: Budget tracker
            temperature: Sampling temperature
            top_p: Nucleus sampling
            seed: Random seed for reproducibility
            tool_choice: Not supported
            metadata: Additional metadata

        Returns:
            ChatResponse with model output
        """
        # Check availability
        if not await self.is_available():
            raise RuntimeError(
                f"Ollama not available at {self.host}. "
                f"Install Ollama from https://ollama.ai and run: ollama pull {self.model}"
            )

        # Build request
        ollama_messages = self._build_messages(messages)

        # Estimate input tokens
        prompt_text = " ".join(m["content"] for m in ollama_messages)
        prompt_tokens = len(prompt_text) // 4

        # Check budget before calling
        estimated_cost = 0.0  # Local = free!
        budget.check(add_usd=estimated_cost, add_tokens=prompt_tokens)

        request_body = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }

        if seed is not None:
            request_body["options"]["seed"] = seed

        if json_schema:
            request_body["format"] = "json"

        try:
            response = await self._client.post(
                f"{self.host}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama request timed out. The model may be loading or your hardware may be slow. "
                f"Try a smaller model like 'phi3' for faster responses."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        # Parse response
        text = data.get("message", {}).get("content", "")

        # Calculate tokens
        completion_tokens = len(text) // 4
        total_tokens = prompt_tokens + completion_tokens

        # Ollama is free - no cost
        cost = 0.0

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
        )

        provider_info = ProviderInfo(
            provider="ollama",
            model=self.model,
            request_id=data.get("created_at", "")[:20],
        )

        # Track in budget
        budget.add(add_usd=cost, add_tokens=total_tokens)

        logger.debug(
            f"OllamaProvider: {total_tokens} tokens, model={self.model}, "
            f"eval_duration={data.get('eval_duration', 0)/1e9:.2f}s"
        )

        return ChatResponse(
            text=text,
            tool_calls=[],
            usage=usage,
            finish_reason=FinishReason.stop,
            provider_info=provider_info,
            raw=data,
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def check_ollama_available(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """
    Check if Ollama is running.

    Args:
        host: Ollama API host

    Returns:
        True if Ollama is accessible
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{host}/api/tags")
            return response.status_code == 200
    except:
        return False


async def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
    """
    List available Ollama models.

    Args:
        host: Ollama API host

    Returns:
        List of model names
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


def get_recommended_model(use_case: str = "default") -> str:
    """
    Get recommended Ollama model for a use case.

    Args:
        use_case: One of "default", "fast", "quality", "reasoning", "code"

    Returns:
        Model name
    """
    return RECOMMENDED_MODELS.get(use_case, DEFAULT_MODEL)


async def ensure_model_available(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST
) -> bool:
    """
    Check if model is available, provide helpful message if not.

    Args:
        model: Model name
        host: Ollama host

    Returns:
        True if model is available
    """
    models = await list_ollama_models(host)

    # Check exact match or with :latest suffix
    model_base = model.split(":")[0]
    available = any(m.startswith(model_base) for m in models)

    if not available:
        logger.warning(
            f"Model '{model}' not found in Ollama. "
            f"Available models: {models}. "
            f"To install: ollama pull {model}"
        )

    return available


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def check_ollama_sync(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """Synchronous check for Ollama availability."""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{host}/api/tags")
            return response.status_code == 200
    except:
        return False


def list_ollama_models_sync(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
    """Synchronous list of Ollama models."""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


if __name__ == "__main__":
    """Test Ollama provider."""
    import asyncio

    async def test():
        print("=== Ollama Provider Test ===\n")

        # Check availability
        available = await check_ollama_available()
        print(f"Ollama available: {available}")

        if not available:
            print("\nOllama not running. To set up:")
            print("1. Download from https://ollama.ai/download")
            print("2. Run: ollama serve")
            print("3. Pull a model: ollama pull llama3.2")
            return

        # List models
        models = await list_ollama_models()
        print(f"Available models: {models}")

        if not models:
            print("\nNo models found. Run: ollama pull llama3.2")
            return

        # Test chat
        config = LLMProviderConfig(model=models[0], api_key_env="")
        provider = OllamaProvider(config)

        messages = [
            ChatMessage(
                role=Role.user,
                content="Explain the carbon footprint of 100 gallons of diesel fuel in 2 sentences."
            )
        ]

        print(f"\nTesting with model: {provider.model}")

        response = await provider.chat(messages, budget=Budget(max_usd=1.0))

        print(f"\nResponse:\n{response.text}")
        print(f"\nTokens: {response.usage.total_tokens}")
        print(f"Cost: ${response.usage.cost_usd:.6f} (local = free!)")

        await provider.close()

    asyncio.run(test())
