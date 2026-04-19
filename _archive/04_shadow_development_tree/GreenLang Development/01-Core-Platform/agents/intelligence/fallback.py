# -*- coding: utf-8 -*-
"""
Model Fallback Chains with Circuit Breaker

Implements intelligent fallback logic for LLM reliability:
- Multi-model fallback chains (GPT-4 -> GPT-3.5 -> Claude)
- Circuit breaker pattern for failure handling
- Smart routing based on query complexity
- Retry logic with exponential backoff
- Quality-based fallback triggering

Fallback Chain:
    Primary Model (GPT-4)
           |
    Rate limit / Timeout / Error
           |
    Fallback Model (GPT-3.5)
           |
    Rate limit / Timeout / Error
           |
    Final Fallback (Claude)

Circuit Breaker States:
- CLOSED: Normal operation
- OPEN: Too many failures, stop trying
- HALF_OPEN: Testing if service recovered

Performance Targets:
- Fallback success rate: >95%
- Average latency: <3s
- Circuit breaker sensitivity: 5 failures -> open
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Too many failures, circuit is open
    HALF_OPEN = "half_open"   # Testing if service recovered


class FallbackReason(Enum):
    """Reasons for fallback"""
    RATE_LIMIT = "rate_limit"             # HTTP 429
    SERVICE_UNAVAILABLE = "service_unavailable"  # HTTP 503
    TIMEOUT = "timeout"                   # Request timeout
    QUALITY_CHECK_FAILED = "quality_check_failed"  # Response quality too low
    GENERIC_ERROR = "generic_error"       # Other errors
    CIRCUIT_OPEN = "circuit_open"         # Circuit breaker is open


@dataclass
class ModelConfig:
    """
    Configuration for a model in fallback chain

    Attributes:
        model: Model identifier
        provider: Provider name (openai, anthropic, etc.)
        max_retries: Maximum retry attempts
        timeout: Request timeout (seconds)
        cost_per_1k_input: Cost per 1K input tokens (USD)
        cost_per_1k_output: Cost per 1K output tokens (USD)
        priority: Priority (lower = higher priority)
    """
    model: str
    provider: str
    max_retries: int = 2
    timeout: float = 30.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    priority: int = 0

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens"""
        return (self.cost_per_1k_input + self.cost_per_1k_output) / 2


# Pre-configured model fallback chains
DEFAULT_FALLBACK_CHAIN = [
    ModelConfig(
        model="gpt-4o",
        provider="openai",
        max_retries=2,
        timeout=30.0,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        priority=0,
    ),
    ModelConfig(
        model="gpt-4-turbo",
        provider="openai",
        max_retries=2,
        timeout=30.0,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        priority=1,
    ),
    ModelConfig(
        model="gpt-3.5-turbo",
        provider="openai",
        max_retries=3,
        timeout=20.0,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        priority=2,
    ),
    ModelConfig(
        model="claude-3-sonnet-20240229",
        provider="anthropic",
        max_retries=2,
        timeout=30.0,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=3,
    ),
]

COST_OPTIMIZED_CHAIN = [
    ModelConfig(
        model="gpt-3.5-turbo",
        provider="openai",
        max_retries=3,
        timeout=20.0,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        priority=0,
    ),
    ModelConfig(
        model="gpt-4o-mini",
        provider="openai",
        max_retries=2,
        timeout=25.0,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        priority=1,
    ),
]


@dataclass
class FallbackAttempt:
    """
    Record of a fallback attempt

    Attributes:
        model: Model that was tried
        success: Whether attempt succeeded
        reason: Reason for fallback (if failed)
        latency: Request latency (seconds)
        timestamp: When attempt was made
        error: Error message (if failed)
    """
    model: str
    success: bool
    reason: Optional[FallbackReason] = None
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class FallbackResult:
    """
    Result of fallback chain execution

    Attributes:
        response: Final response
        model_used: Model that succeeded
        attempts: List of all attempts
        total_latency: Total time across all attempts
        fallback_count: Number of fallbacks
        cost: Estimated cost (USD)
    """
    response: Any
    model_used: str
    attempts: List[FallbackAttempt]
    total_latency: float
    fallback_count: int
    cost: float = 0.0

    @property
    def success(self) -> bool:
        """Whether any attempt succeeded"""
        return any(a.success for a in self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_used": self.model_used,
            "fallback_count": self.fallback_count,
            "total_latency": self.total_latency,
            "success": self.success,
            "cost": self.cost,
            "attempts": [
                {
                    "model": a.model,
                    "success": a.success,
                    "reason": a.reason.value if a.reason else None,
                    "latency": a.latency,
                }
                for a in self.attempts
            ],
        }


class CircuitBreaker:
    """
    Circuit breaker for model failure handling

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation
    - OPEN: Too many failures, don't try
    - HALF_OPEN: Testing recovery

    After N consecutive failures, circuit opens for cooldown period.
    After cooldown, circuit goes to half-open to test recovery.
    If test succeeds, circuit closes. If fails, circuit opens again.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Cooldown period before testing recovery (seconds)
            success_threshold: Number of successes in half-open before closing
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

        logger.info(f"CircuitBreaker initialized (threshold={failure_threshold}, timeout={recovery_timeout}s)")

    def record_success(self):
        """Record successful request"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                # Close circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED (recovered)")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count
            self.failure_count = 0

    def record_failure(self):
        """Record failed request"""
        self.last_failure_time = DeterministicClock.now()

        if self.state == CircuitState.HALF_OPEN:
            # Test failed, open circuit again
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning("Circuit breaker OPEN (recovery test failed)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                # Open circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN ({self.failure_count} consecutive failures)")

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if cooldown period has passed
            if self.last_failure_time:
                elapsed = (DeterministicClock.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    # Move to half-open for testing
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker HALF_OPEN (testing recovery)")
                    return True

            return False

        if self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state


class FallbackManager:
    """
    Manages model fallback chains with circuit breakers

    Handles:
    - Multi-model fallback
    - Circuit breakers per model
    - Retry logic with exponential backoff
    - Smart routing based on complexity
    - Metrics tracking
    """

    def __init__(
        self,
        fallback_chain: Optional[List[ModelConfig]] = None,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize fallback manager

        Args:
            fallback_chain: List of models in fallback order
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.fallback_chain = fallback_chain or DEFAULT_FALLBACK_CHAIN
        self.enable_circuit_breaker = enable_circuit_breaker

        # Circuit breakers per model
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        if enable_circuit_breaker:
            for config in self.fallback_chain:
                self.circuit_breakers[config.model] = CircuitBreaker()

        # Metrics
        self.total_requests = 0
        self.fallback_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}

        logger.info(f"FallbackManager initialized with {len(self.fallback_chain)} models")

    async def execute_with_fallback(
        self,
        execute_fn: Callable[[ModelConfig], Coroutine[Any, Any, Any]],
        quality_check_fn: Optional[Callable[[Any], float]] = None,
        min_quality: float = 0.8,
    ) -> FallbackResult:
        """
        Execute request with fallback chain

        Args:
            execute_fn: Async function to execute request: (config) -> response
            quality_check_fn: Optional quality check: (response) -> score (0-1)
            min_quality: Minimum quality score to accept response

        Returns:
            Fallback result with response and metadata
        """
        self.total_requests += 1
        start_time = time.time()

        attempts: List[FallbackAttempt] = []
        response = None
        model_used = None

        for config in self.fallback_chain:
            # Check circuit breaker
            if self.enable_circuit_breaker:
                circuit = self.circuit_breakers.get(config.model)
                if circuit and not circuit.can_execute():
                    logger.warning(f"Circuit breaker OPEN for {config.model}. Skipping.")
                    attempts.append(FallbackAttempt(
                        model=config.model,
                        success=False,
                        reason=FallbackReason.CIRCUIT_OPEN,
                    ))
                    continue

            # Try model with retries
            attempt_result = await self._try_model(config, execute_fn, quality_check_fn, min_quality)
            attempts.append(attempt_result)

            # Update circuit breaker
            if self.enable_circuit_breaker and circuit:
                if attempt_result.success:
                    circuit.record_success()
                else:
                    circuit.record_failure()

            # Check if succeeded
            if attempt_result.success:
                response = attempt_result
                model_used = config.model
                self.success_counts[config.model] = self.success_counts.get(config.model, 0) + 1
                break
            else:
                # Record fallback
                self.fallback_counts[config.model] = self.fallback_counts.get(config.model, 0) + 1

        # Calculate metrics
        total_latency = time.time() - start_time
        fallback_count = len(attempts) - 1

        # Estimate cost (simplified)
        cost = 0.0
        if model_used:
            config = next((c for c in self.fallback_chain if c.model == model_used), None)
            if config:
                # Assume 500 tokens average
                cost = config.avg_cost_per_1k * 0.5

        result = FallbackResult(
            response=response,
            model_used=model_used or "none",
            attempts=attempts,
            total_latency=total_latency,
            fallback_count=fallback_count,
            cost=cost,
        )

        if result.success:
            logger.info(f"Request succeeded with {model_used} after {fallback_count} fallbacks ({total_latency:.2f}s)")
        else:
            logger.error(f"Request failed after trying all {len(attempts)} models ({total_latency:.2f}s)")

        return result

    async def _try_model(
        self,
        config: ModelConfig,
        execute_fn: Callable[[ModelConfig], Coroutine[Any, Any, Any]],
        quality_check_fn: Optional[Callable[[Any], float]],
        min_quality: float,
    ) -> FallbackAttempt:
        """
        Try executing request with a single model (with retries)

        Args:
            config: Model configuration
            execute_fn: Execution function
            quality_check_fn: Quality check function
            min_quality: Minimum quality score

        Returns:
            Attempt result
        """
        attempt_start = time.time()

        for retry in range(config.max_retries):
            try:
                # Execute with timeout
                response = await asyncio.wait_for(
                    execute_fn(config),
                    timeout=config.timeout,
                )

                # Quality check
                if quality_check_fn:
                    quality_score = quality_check_fn(response)
                    if quality_score < min_quality:
                        logger.warning(f"{config.model}: Quality check failed ({quality_score:.2f} < {min_quality})")
                        if retry < config.max_retries - 1:
                            await asyncio.sleep(self._get_backoff_delay(retry))
                            continue
                        else:
                            return FallbackAttempt(
                                model=config.model,
                                success=False,
                                reason=FallbackReason.QUALITY_CHECK_FAILED,
                                latency=time.time() - attempt_start,
                            )

                # Success!
                return FallbackAttempt(
                    model=config.model,
                    success=True,
                    latency=time.time() - attempt_start,
                )

            except asyncio.TimeoutError:
                logger.warning(f"{config.model}: Timeout (retry {retry+1}/{config.max_retries})")
                if retry < config.max_retries - 1:
                    await asyncio.sleep(self._get_backoff_delay(retry))
                    continue
                else:
                    return FallbackAttempt(
                        model=config.model,
                        success=False,
                        reason=FallbackReason.TIMEOUT,
                        latency=time.time() - attempt_start,
                    )

            except Exception as e:
                error_str = str(e).lower()

                # Determine fallback reason
                if "429" in error_str or "rate limit" in error_str:
                    reason = FallbackReason.RATE_LIMIT
                elif "503" in error_str or "unavailable" in error_str:
                    reason = FallbackReason.SERVICE_UNAVAILABLE
                else:
                    reason = FallbackReason.GENERIC_ERROR

                logger.warning(f"{config.model}: {reason.value} (retry {retry+1}/{config.max_retries}): {e}")

                if retry < config.max_retries - 1:
                    await asyncio.sleep(self._get_backoff_delay(retry))
                    continue
                else:
                    return FallbackAttempt(
                        model=config.model,
                        success=False,
                        reason=reason,
                        latency=time.time() - attempt_start,
                        error=str(e),
                    )

        # Should not reach here
        return FallbackAttempt(
            model=config.model,
            success=False,
            reason=FallbackReason.GENERIC_ERROR,
            latency=time.time() - attempt_start,
        )

    def _get_backoff_delay(self, retry: int) -> float:
        """
        Calculate exponential backoff delay

        Args:
            retry: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        base_delay = 1.0
        max_delay = 10.0
        delay = min(base_delay * (2 ** retry), max_delay)
        return delay

    def route_by_complexity(self, query: str) -> List[ModelConfig]:
        """
        Route query to appropriate model based on complexity

        Simple heuristic:
        - Short query (<50 chars) -> cheap model
        - Medium query (50-200 chars) -> balanced model
        - Long query (>200 chars) -> best model

        Args:
            query: User query

        Returns:
            Reordered fallback chain
        """
        query_length = len(query)

        if query_length < 50:
            # Simple query -> cheap models first
            return sorted(self.fallback_chain, key=lambda c: c.avg_cost_per_1k)
        elif query_length > 200:
            # Complex query -> best models first
            return sorted(self.fallback_chain, key=lambda c: -c.avg_cost_per_1k)
        else:
            # Balanced -> default order
            return self.fallback_chain

    def get_metrics(self) -> Dict[str, Any]:
        """Get fallback metrics"""
        return {
            "total_requests": self.total_requests,
            "fallback_counts": self.fallback_counts,
            "success_counts": self.success_counts,
            "circuit_breaker_states": {
                model: cb.get_state().value
                for model, cb in self.circuit_breakers.items()
            } if self.enable_circuit_breaker else {},
        }


if __name__ == "__main__":
    """
    Demo and testing
    """
    import asyncio

    print("=" * 80)
    print("GreenLang Fallback Manager Demo")
    print("=" * 80)

    async def demo():
        # Initialize fallback manager
        manager = FallbackManager()

        # Simulate execution function
        call_count = {"count": 0}

        async def mock_execute(config: ModelConfig):
            call_count["count"] += 1

            # Simulate failures for first 2 models
            if config.model == "gpt-4o":
                raise Exception("Rate limit exceeded (429)")
            elif config.model == "gpt-4-turbo":
                raise Exception("Service unavailable (503)")
            else:
                # Success with GPT-3.5
                return {"response": "Natural gas emits ~0.185 kg CO2/kWh", "model": config.model}

        # Execute with fallback
        print("\n1. Testing fallback chain:")
        result = await manager.execute_with_fallback(mock_execute)

        print(f"   Result: {result.to_dict()}")
        print(f"   Total attempts: {len(result.attempts)}")
        print(f"   Model used: {result.model_used}")
        print(f"   Success: {result.success}")

        # Test circuit breaker
        print("\n2. Testing circuit breaker:")
        async def always_fail(config: ModelConfig):
            raise Exception("Always fails")

        for i in range(7):
            result = await manager.execute_with_fallback(always_fail)
            print(f"   Attempt {i+1}: {len(result.attempts)} models tried")

        # Show metrics
        print("\n3. Fallback metrics:")
        metrics = manager.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")

    # Run demo
    asyncio.run(demo())

    print("\n" + "=" * 80)
