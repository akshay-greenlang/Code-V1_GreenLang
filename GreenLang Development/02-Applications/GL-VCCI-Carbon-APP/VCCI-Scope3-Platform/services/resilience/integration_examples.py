# -*- coding: utf-8 -*-
"""Integration Examples for Resilience Patterns in GL-VCCI.

Demonstrates how to use retry, timeout, fallback, rate limiting,
and graceful degradation patterns in GL-VCCI agents.

Author: Team 2 - Resilience Patterns
Date: November 2025
"""

import logging
from typing import Any, Dict, List, Optional

# GreenLang resilience patterns
from greenlang.resilience import (
    retry,
    timeout,
    fallback,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    RetryStrategy,
    TimeoutConfig,
    OperationType,
    FallbackStrategy,
    RateLimiter,
    RateLimitConfig,
    get_rate_limiter,
)

# GL-VCCI graceful degradation
from .graceful_degradation import (
    DegradationTier,
    get_degradation_manager,
    degradation_handler,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Example 1: Calculator Agent with Retry + Timeout + Fallback
# ==============================================================================


class ResilientCalculatorAgent:
    """Calculator agent with full resilience patterns.

    Features:
    - Retry with exponential backoff for transient errors
    - Timeout for long-running calculations
    - Fallback to cached emission factors
    - Circuit breaker for factor API
    - Rate limiting for API calls
    """

    def __init__(self):
        """Initialize resilient calculator agent."""
        # Circuit breaker for factor API
        self.factor_api_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                name="factor_api",
                fail_max=5,
                timeout_duration=60,
            )
        )

        # Rate limiter
        self.rate_limiter = get_rate_limiter()
        self.rate_limiter.configure(
            "factor_api:default",
            RateLimitConfig(
                requests_per_second=10.0,
                burst_size=20,
            )
        )

    @retry(
        max_retries=3,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=(ConnectionError, TimeoutError),
    )
    @timeout(operation_type=OperationType.EXTERNAL_API)
    @fallback(
        strategy=FallbackStrategy.CACHED,
        cache_key_func=lambda category: f"factor_{category}",
    )
    @degradation_handler(
        min_tier=DegradationTier.TIER_2_CORE,
        fallback_value=None,
    )
    def get_emission_factor(self, category: str) -> Optional[Dict[str, Any]]:
        """Get emission factor with full resilience.

        Args:
            category: Emission factor category

        Returns:
            Emission factor data or None

        Features:
        - Retries on network errors (3 attempts, exponential backoff)
        - Times out after 15 seconds
        - Falls back to cached value if API fails
        - Respects degradation tier (requires Tier 2+)
        """
        # Check rate limit
        self.rate_limiter.check_limit("factor_api:default")

        # Call through circuit breaker
        return self.factor_api_circuit.call(
            self._fetch_factor_from_api,
            category
        )

    def _fetch_factor_from_api(self, category: str) -> Dict[str, Any]:
        """Internal method to fetch from API.

        Args:
            category: Emission factor category

        Returns:
            Emission factor data
        """
        # Simulated API call
        # In production, this would call the actual factor API
        logger.info(f"Fetching emission factor for category: {category}")

        # Simulate network call
        import time
        time.sleep(0.1)

        return {
            "category": category,
            "factor": 0.45,
            "unit": "kg CO2e/unit",
            "source": "EPA 2024",
        }

    @retry(max_retries=2, base_delay=0.5)
    @timeout(timeout_seconds=30.0)
    def calculate_emissions(
        self,
        activity_data: float,
        category: str,
    ) -> Optional[float]:
        """Calculate emissions with resilience.

        Args:
            activity_data: Activity data value
            category: Emission category

        Returns:
            Calculated emissions or None

        Features:
        - Retries calculation on failure (2 attempts)
        - Times out after 30 seconds
        - Falls back to None if calculation fails
        """
        factor = self.get_emission_factor(category)

        if factor is None:
            logger.warning(f"No factor available for {category}")
            return None

        emissions = activity_data * factor["factor"]

        logger.info(
            f"Calculated emissions: {emissions:.2f} kg CO2e "
            f"(activity={activity_data}, category={category})"
        )

        return emissions


# ==============================================================================
# Example 2: Factor Broker with Rate Limiting
# ==============================================================================


class ResilientFactorBroker:
    """Factor broker with rate limiting and circuit breakers.

    Features:
    - Rate limiting per data source
    - Circuit breakers per external API
    - Fallback to secondary sources
    - Timeout for slow APIs
    """

    def __init__(self):
        """Initialize resilient factor broker."""
        # Circuit breakers for different sources
        self.circuits = {
            "epa": CircuitBreaker(
                CircuitBreakerConfig(
                    name="factor_source_epa",
                    fail_max=5,
                    timeout_duration=60,
                )
            ),
            "defra": CircuitBreaker(
                CircuitBreakerConfig(
                    name="factor_source_defra",
                    fail_max=5,
                    timeout_duration=60,
                )
            ),
            "ecoinvent": CircuitBreaker(
                CircuitBreakerConfig(
                    name="factor_source_ecoinvent",
                    fail_max=3,
                    timeout_duration=120,
                )
            ),
        }

        # Rate limiters
        self.rate_limiter = get_rate_limiter()

        # Configure rate limits per source
        self.rate_limiter.configure(
            "factor_source:epa",
            RateLimitConfig(requests_per_second=50.0, burst_size=100)
        )
        self.rate_limiter.configure(
            "factor_source:defra",
            RateLimitConfig(requests_per_second=20.0, burst_size=40)
        )
        self.rate_limiter.configure(
            "factor_source:ecoinvent",
            RateLimitConfig(requests_per_second=5.0, burst_size=10)
        )

    @timeout(operation_type=OperationType.FACTOR_LOOKUP)
    def get_factor(
        self,
        category: str,
        source: str = "epa",
    ) -> Optional[Dict[str, Any]]:
        """Get emission factor from specified source.

        Args:
            category: Factor category
            source: Data source (epa, defra, ecoinvent)

        Returns:
            Emission factor data or None

        Features:
        - Rate limited by source
        - Circuit breaker protection
        - Timeout after 5 seconds
        """
        # Check rate limit
        rate_limit_key = f"factor_source:{source}"
        self.rate_limiter.check_limit(rate_limit_key)

        # Get circuit breaker
        circuit = self.circuits.get(source)
        if circuit is None:
            logger.error(f"Unknown factor source: {source}")
            return None

        # Call through circuit breaker
        try:
            return circuit.call(
                self._fetch_from_source,
                category,
                source
            )
        except Exception as e:
            logger.error(f"Failed to fetch factor from {source}: {e}")
            return None

    def _fetch_from_source(
        self,
        category: str,
        source: str,
    ) -> Dict[str, Any]:
        """Fetch factor from source.

        Args:
            category: Factor category
            source: Data source

        Returns:
            Emission factor data
        """
        logger.info(f"Fetching factor from {source} for {category}")

        # Simulated API call
        import time
        time.sleep(0.05)

        return {
            "category": category,
            "source": source,
            "factor": 0.45,
            "unit": "kg CO2e/unit",
        }

    @fallback(
        strategy=FallbackStrategy.FUNCTION,
        fallback_function=lambda self, category: self.get_factor(category, "defra"),
    )
    def get_factor_with_fallback(
        self,
        category: str,
    ) -> Optional[Dict[str, Any]]:
        """Get factor with fallback to secondary source.

        Args:
            category: Factor category

        Returns:
            Emission factor data

        Features:
        - Primary: EPA
        - Fallback: DEFRA
        """
        return self.get_factor(category, source="epa")


# ==============================================================================
# Example 3: LLM Provider with Timeout + Fallback
# ==============================================================================


class ResilientLLMProvider:
    """LLM provider with timeout and fallback to cached results.

    Features:
    - Timeout for LLM inference (30 seconds)
    - Fallback to cached responses
    - Circuit breaker for LLM API
    - Degradation-aware (only works in Tier 1)
    """

    def __init__(self):
        """Initialize resilient LLM provider."""
        self.circuit = CircuitBreaker(
            CircuitBreakerConfig(
                name="llm_api",
                fail_max=3,
                timeout_duration=120,
            )
        )

    @retry(max_retries=2, base_delay=2.0)
    @timeout(operation_type=OperationType.LLM_INFERENCE)
    @fallback(
        strategy=FallbackStrategy.CACHED,
        cache_key_func=lambda prompt: f"llm_{hash(prompt)}",
    )
    @degradation_handler(
        min_tier=DegradationTier.TIER_1_FULL,
        fallback_value="[LLM unavailable in degraded mode]",
    )
    def generate(self, prompt: str) -> str:
        """Generate LLM response with resilience.

        Args:
            prompt: Input prompt

        Returns:
            Generated text

        Features:
        - Retries on failure (2 attempts)
        - Times out after 30 seconds
        - Falls back to cached response
        - Only available in Tier 1 (full functionality)
        """
        return self.circuit.call(self._call_llm_api, prompt)

    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        logger.info(f"Calling LLM API with prompt: {prompt[:50]}...")

        # Simulated LLM call
        import time
        time.sleep(0.2)

        return f"Generated response for: {prompt[:30]}..."


# ==============================================================================
# Example 4: ERP Connector with Full Resilience Stack
# ==============================================================================


class ResilientERPConnector:
    """ERP connector with complete resilience stack.

    Features:
    - Retry with exponential backoff
    - Timeout per operation type
    - Circuit breaker
    - Rate limiting
    - Fallback to cached data
    - Degradation awareness
    """

    def __init__(self):
        """Initialize resilient ERP connector."""
        self.circuit = CircuitBreaker(
            CircuitBreakerConfig(
                name="erp_api",
                fail_max=5,
                timeout_duration=60,
            )
        )

        self.rate_limiter = get_rate_limiter()
        self.rate_limiter.configure(
            "erp_api:default",
            RateLimitConfig(
                requests_per_second=5.0,
                burst_size=10,
            )
        )

    @retry(
        max_retries=5,
        base_delay=2.0,
        max_delay=60.0,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=(ConnectionError, TimeoutError),
    )
    @timeout(operation_type=OperationType.ERP_API_CALL)
    @fallback(strategy=FallbackStrategy.CACHED)
    @degradation_handler(min_tier=DegradationTier.TIER_2_CORE)
    def fetch_procurement_data(
        self,
        supplier_id: str,
        date_range: tuple,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch procurement data from ERP.

        Args:
            supplier_id: Supplier identifier
            date_range: Date range tuple (start, end)

        Returns:
            Procurement records or None

        Features:
        - Retries up to 5 times with exponential backoff
        - Times out after 10 seconds
        - Falls back to cached data
        - Requires Tier 2 or better
        - Rate limited to 5 req/s
        """
        # Check rate limit
        self.rate_limiter.check_limit("erp_api:default")

        # Call through circuit breaker
        return self.circuit.call(
            self._fetch_from_erp,
            supplier_id,
            date_range,
        )

    def _fetch_from_erp(
        self,
        supplier_id: str,
        date_range: tuple,
    ) -> List[Dict[str, Any]]:
        """Fetch from ERP API.

        Args:
            supplier_id: Supplier identifier
            date_range: Date range tuple

        Returns:
            Procurement records
        """
        logger.info(
            f"Fetching procurement data for supplier {supplier_id} "
            f"from {date_range[0]} to {date_range[1]}"
        )

        # Simulated ERP call
        import time
        time.sleep(0.3)

        return [
            {
                "supplier_id": supplier_id,
                "amount": 10000.0,
                "date": "2024-01-15",
                "category": "Electronics",
            }
        ]


# ==============================================================================
# Example 5: Complete Integration Example
# ==============================================================================


def example_complete_workflow():
    """Complete workflow demonstrating all resilience patterns.

    This example shows how to use all resilience patterns together
    in a realistic GL-VCCI workflow.
    """
    # Initialize degradation manager
    degradation_manager = get_degradation_manager()

    # Register tier change callback
    def on_tier_change(old_tier, new_tier):
        logger.warning(f"System degradation: {old_tier} -> {new_tier}")

    degradation_manager.register_tier_change_callback(on_tier_change)

    # Update service health
    degradation_manager.update_health("factor_api", healthy=True, response_time_ms=150)
    degradation_manager.update_health("database", healthy=True, response_time_ms=50)
    degradation_manager.update_health("erp_api", healthy=True, response_time_ms=200)
    degradation_manager.update_health("llm_api", healthy=False, error="Timeout")

    # Check current tier
    current_tier = degradation_manager.get_current_tier()
    logger.info(f"Current degradation tier: {current_tier}")

    # Initialize resilient components
    calculator = ResilientCalculatorAgent()
    factor_broker = ResilientFactorBroker()
    llm_provider = ResilientLLMProvider()
    erp_connector = ResilientERPConnector()

    # Example workflow
    try:
        # 1. Fetch emission factor
        factor = calculator.get_emission_factor("electricity")
        logger.info(f"Emission factor: {factor}")

        # 2. Calculate emissions
        emissions = calculator.calculate_emissions(
            activity_data=1000.0,
            category="electricity"
        )
        logger.info(f"Calculated emissions: {emissions}")

        # 3. Fetch from multiple sources with fallback
        factor_with_fallback = factor_broker.get_factor_with_fallback("natural_gas")
        logger.info(f"Factor with fallback: {factor_with_fallback}")

        # 4. LLM generation (only in Tier 1)
        if current_tier == DegradationTier.TIER_1_FULL:
            response = llm_provider.generate("Analyze supplier carbon footprint")
            logger.info(f"LLM response: {response}")

        # 5. ERP data fetch
        procurement_data = erp_connector.fetch_procurement_data(
            supplier_id="SUP-12345",
            date_range=("2024-01-01", "2024-12-31")
        )
        logger.info(f"Procurement data: {procurement_data}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")

    # Get degradation stats
    stats = degradation_manager.get_stats()
    logger.info(f"Degradation stats: {stats}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run example
    example_complete_workflow()
