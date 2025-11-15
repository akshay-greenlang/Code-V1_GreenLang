"""
Pytest Configuration and Fixtures for LLM Integration Tests.

This module provides shared fixtures, configuration, and utilities for testing
GreenLang's LLM system with real API calls, mocking, and performance benchmarks.

Test Modes:
- REAL_API: Use real API calls (requires API keys in .env)
- MOCK: Use mocked responses (for CI/CD)
- HYBRID: Mix of real and mock (for cost-effective testing)

Environment Variables Required:
- ANTHROPIC_API_KEY: Anthropic Claude API key
- OPENAI_API_KEY: OpenAI GPT API key
- TEST_MODE: "real" | "mock" | "hybrid" (default: "mock")
- TEST_BUDGET_USD: Maximum test budget in USD (default: 1.00)
"""

import os
import sys
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from unittest.mock import AsyncMock, Mock, patch

# Add parent directories to path
test_dir = Path(__file__).parent
foundation_dir = test_dir.parent.parent
sys.path.insert(0, str(foundation_dir))

# Import LLM modules
from llm.providers.base_provider import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
    ProviderHealth,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from llm.providers.anthropic_provider import AnthropicProvider
from llm.providers.openai_provider import OpenAIProvider
from llm.llm_router import LLMRouter, RoutingStrategy
from llm.circuit_breaker import CircuitBreaker, CircuitState
from llm.rate_limiter import RateLimiter
from llm.cost_tracker import CostTracker

# Load environment variables
load_dotenv()

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfig:
    """Centralized test configuration."""

    # Test mode
    TEST_MODE = os.getenv("TEST_MODE", "mock").lower()  # real, mock, hybrid

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Budget limits
    TEST_BUDGET_USD = float(os.getenv("TEST_BUDGET_USD", "1.00"))

    # Test models (cheaper for testing)
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"  # Cheapest model
    OPENAI_MODEL = "gpt-3.5-turbo"  # Cheapest model

    # Performance targets
    LATENCY_P95_TARGET_MS = 2000  # 2s P95 latency
    THROUGHPUT_TARGET_RPS = 10  # 10 requests/second

    # Timeouts
    DEFAULT_TIMEOUT_S = 30.0
    HEALTH_CHECK_TIMEOUT_S = 10.0

    # Rate limits (test mode - reduced)
    ANTHROPIC_RATE_LIMIT_RPM = 50  # 50 req/min for testing
    OPENAI_RATE_LIMIT_RPM = 100  # 100 req/min for testing

    @classmethod
    def should_skip_real_api_tests(cls) -> bool:
        """Check if real API tests should be skipped."""
        if cls.TEST_MODE == "mock":
            return True
        if not cls.ANTHROPIC_API_KEY and not cls.OPENAI_API_KEY:
            logger.warning("No API keys found - skipping real API tests")
            return True
        return False

    @classmethod
    def get_test_budget_remaining(cls) -> float:
        """Get remaining test budget (tracked in cost tracker)."""
        # This would connect to actual cost tracker in production
        return cls.TEST_BUDGET_USD


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Get test configuration."""
    return TestConfig()


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: Integration tests (may use real APIs)"
    )
    config.addinivalue_line(
        "markers", "real_api: Tests that require real API calls"
    )
    config.addinivalue_line(
        "markers", "anthropic: Tests specific to Anthropic provider"
    )
    config.addinivalue_line(
        "markers", "openai: Tests specific to OpenAI provider"
    )
    config.addinivalue_line(
        "markers", "router: Tests for LLM router"
    )
    config.addinivalue_line(
        "markers", "failover: Tests for failover scenarios"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip real API tests if needed."""
    skip_real_api = pytest.mark.skip(reason="Real API tests disabled (TEST_MODE=mock or no API keys)")

    for item in items:
        if "real_api" in item.keywords and TestConfig.should_skip_real_api_tests():
            item.add_marker(skip_real_api)


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Provider Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_response() -> Dict[str, Any]:
    """Create mock Anthropic API response."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a test response from Claude."}],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        }
    }


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Create mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from GPT."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest_asyncio.fixture
async def mock_anthropic_provider(mock_anthropic_response) -> AsyncMock:
    """Create mocked Anthropic provider."""
    mock = AsyncMock(spec=AnthropicProvider)
    mock.provider_name = "anthropic"
    mock.model_id = "claude-3-haiku-20240307"
    mock.cost_per_1k_input = 0.00025
    mock.cost_per_1k_output = 0.00125

    # Mock generate method
    async def mock_generate(request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="This is a test response from Claude.",
            model_id="claude-3-haiku-20240307",
            provider="anthropic",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                input_cost_usd=0.0000025,
                output_cost_usd=0.000025,
                total_cost_usd=0.0000275
            ),
            finish_reason="end_turn",
            generation_time_ms=150.0,
            metadata={"message_id": "msg_test123"}
        )

    mock.generate.side_effect = mock_generate

    # Mock health check
    async def mock_health_check() -> ProviderHealth:
        from datetime import datetime
        return ProviderHealth(
            is_healthy=True,
            last_check=datetime.utcnow(),
            consecutive_failures=0,
            latency_ms=100.0
        )

    mock.health_check.side_effect = mock_health_check

    return mock


@pytest_asyncio.fixture
async def mock_openai_provider(mock_openai_response) -> AsyncMock:
    """Create mocked OpenAI provider."""
    mock = AsyncMock(spec=OpenAIProvider)
    mock.provider_name = "openai"
    mock.model_id = "gpt-3.5-turbo"
    mock.cost_per_1k_input = 0.0005
    mock.cost_per_1k_output = 0.0015

    # Mock generate method
    async def mock_generate(request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text="This is a test response from GPT.",
            model_id="gpt-3.5-turbo",
            provider="openai",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                input_cost_usd=0.000005,
                output_cost_usd=0.00003,
                total_cost_usd=0.000035
            ),
            finish_reason="stop",
            generation_time_ms=200.0,
            metadata={"completion_id": "chatcmpl-test123"}
        )

    mock.generate.side_effect = mock_generate

    # Mock health check
    async def mock_health_check() -> ProviderHealth:
        from datetime import datetime
        return ProviderHealth(
            is_healthy=True,
            last_check=datetime.utcnow(),
            consecutive_failures=0,
            latency_ms=120.0
        )

    mock.health_check.side_effect = mock_health_check

    # Mock embeddings
    async def mock_generate_embeddings(texts):
        return [[0.1] * 1536 for _ in texts]  # 1536-dim vectors

    mock.generate_embeddings.side_effect = mock_generate_embeddings

    return mock


# ============================================================================
# Real Provider Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def anthropic_provider(test_config) -> Optional[AnthropicProvider]:
    """Create real Anthropic provider (if API key available)."""
    if not test_config.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider(
        model_id=test_config.ANTHROPIC_MODEL,
        api_key=test_config.ANTHROPIC_API_KEY,
        max_retries=3,
        timeout=test_config.DEFAULT_TIMEOUT_S
    )

    yield provider

    # Cleanup
    await provider.close()


@pytest_asyncio.fixture
async def openai_provider(test_config) -> Optional[OpenAIProvider]:
    """Create real OpenAI provider (if API key available)."""
    if not test_config.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")

    provider = OpenAIProvider(
        model_id=test_config.OPENAI_MODEL,
        api_key=test_config.OPENAI_API_KEY,
        max_retries=3,
        timeout=test_config.DEFAULT_TIMEOUT_S
    )

    yield provider

    # Cleanup
    await provider.close()


# ============================================================================
# Router Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def mock_router(mock_anthropic_provider, mock_openai_provider) -> LLMRouter:
    """Create LLM router with mocked providers."""
    router = LLMRouter(
        strategy=RoutingStrategy.PRIORITY,
        health_check_interval=10.0,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0,
        max_retries=3
    )

    # Register mocked providers
    router.register_provider("anthropic", mock_anthropic_provider, priority=1)
    router.register_provider("openai", mock_openai_provider, priority=2)

    yield router

    # Cleanup
    await router.close()


@pytest_asyncio.fixture
async def real_router(anthropic_provider, openai_provider) -> Optional[LLMRouter]:
    """Create LLM router with real providers."""
    if not anthropic_provider and not openai_provider:
        pytest.skip("No API keys available for real router")

    router = LLMRouter(
        strategy=RoutingStrategy.PRIORITY,
        health_check_interval=30.0,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0,
        max_retries=3
    )

    # Register available providers
    if anthropic_provider:
        router.register_provider("anthropic", anthropic_provider, priority=1)
    if openai_provider:
        router.register_provider("openai", openai_provider, priority=2)

    yield router

    # Cleanup
    await router.close()


# ============================================================================
# Circuit Breaker Fixtures
# ============================================================================

@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Create circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_calls=1,
        name="test-breaker"
    )


# ============================================================================
# Rate Limiter Fixtures
# ============================================================================

@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create rate limiter for testing."""
    return RateLimiter(
        requests_per_minute=60,  # 1 per second for testing
        tokens_per_minute=10000,
        enable_queuing=True,
        max_wait_time=5.0,
        name="test-limiter"
    )


# ============================================================================
# Cost Tracker Fixtures
# ============================================================================

@pytest.fixture
def cost_tracker() -> CostTracker:
    """Create cost tracker for testing."""
    tracker = CostTracker(auto_reset_budgets=False)

    # Set test budget
    tracker.set_budget(
        tenant_id="test-tenant",
        monthly_limit_usd=TestConfig.TEST_BUDGET_USD,
        alert_thresholds=[0.8, 0.9, 1.0]
    )

    return tracker


# ============================================================================
# Request/Response Fixtures
# ============================================================================

@pytest.fixture
def simple_request() -> GenerationRequest:
    """Create simple generation request for testing."""
    return GenerationRequest(
        prompt="What is carbon accounting?",
        temperature=0.7,
        max_tokens=100,
        metadata={"test_id": "simple_request"}
    )


@pytest.fixture
def complex_request() -> GenerationRequest:
    """Create complex generation request for testing."""
    return GenerationRequest(
        prompt="Analyze the following ESG data and provide detailed insights...",
        system_prompt="You are an expert ESG analyst.",
        temperature=0.5,
        max_tokens=500,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop_sequences=["END", "STOP"],
        metadata={"test_id": "complex_request"}
    )


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_valid_response():
    """Helper to validate generation responses."""
    def _validate(response: GenerationResponse):
        assert isinstance(response, GenerationResponse)
        assert response.text is not None and len(response.text) > 0
        assert response.model_id is not None
        assert response.provider is not None
        assert isinstance(response.usage, TokenUsage)
        assert response.usage.total_tokens > 0
        assert response.usage.total_cost_usd >= 0
        assert response.generation_time_ms > 0
        assert response.finish_reason is not None
    return _validate


@pytest.fixture
def assert_performance_target():
    """Helper to validate performance targets."""
    def _validate(latency_ms: float, target_ms: float = TestConfig.LATENCY_P95_TARGET_MS):
        assert latency_ms < target_ms, f"Latency {latency_ms}ms exceeds target {target_ms}ms"
    return _validate


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any per-test cleanup goes here
    pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """Cleanup after entire test session."""
    yield

    # Log test summary
    logger.info("=" * 80)
    logger.info("Integration Test Session Complete")
    logger.info("=" * 80)
