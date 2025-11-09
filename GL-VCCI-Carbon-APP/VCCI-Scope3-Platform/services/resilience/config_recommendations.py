"""Configuration Recommendations for GL-VCCI Resilience Patterns.

Production-ready configurations for different scenarios and environments.

Author: Team 2 - Resilience Patterns
Date: November 2025
"""

from dataclasses import dataclass
from typing import Dict, Any

from greenlang.resilience import (
    RetryConfig,
    RetryStrategy,
    TimeoutConfig,
    OperationType,
    FallbackStrategy,
    FallbackConfig,
    RateLimitConfig,
    RateLimitAlgorithm,
    CircuitBreakerConfig,
)


# ==============================================================================
# Environment-Specific Configurations
# ==============================================================================


@dataclass
class EnvironmentConfig:
    """Configuration for specific environment."""
    name: str
    retry: RetryConfig
    timeout: Dict[str, TimeoutConfig]
    rate_limits: Dict[str, RateLimitConfig]
    circuit_breakers: Dict[str, CircuitBreakerConfig]


# Development Environment - Relaxed limits for testing
DEVELOPMENT_CONFIG = EnvironmentConfig(
    name="development",
    retry=RetryConfig(
        max_retries=2,
        base_delay=0.5,
        max_delay=5.0,
        strategy=RetryStrategy.EXPONENTIAL,
    ),
    timeout={
        "factor_lookup": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.FACTOR_LOOKUP,
        ),
        "llm_inference": TimeoutConfig(
            timeout_seconds=60.0,
            operation_type=OperationType.LLM_INFERENCE,
        ),
        "erp_api": TimeoutConfig(
            timeout_seconds=20.0,
            operation_type=OperationType.ERP_API_CALL,
        ),
        "database": TimeoutConfig(
            timeout_seconds=15.0,
            operation_type=OperationType.DATABASE_QUERY,
        ),
    },
    rate_limits={
        "factor_api": RateLimitConfig(
            requests_per_second=100.0,
            burst_size=200,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        "llm_api": RateLimitConfig(
            requests_per_second=50.0,
            burst_size=100,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        "erp_api": RateLimitConfig(
            requests_per_second=20.0,
            burst_size=40,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
    },
    circuit_breakers={
        "factor_api": CircuitBreakerConfig(
            name="factor_api_dev",
            fail_max=10,
            timeout_duration=30,
        ),
        "llm_api": CircuitBreakerConfig(
            name="llm_api_dev",
            fail_max=5,
            timeout_duration=60,
        ),
        "erp_api": CircuitBreakerConfig(
            name="erp_api_dev",
            fail_max=10,
            timeout_duration=30,
        ),
    },
)


# Staging Environment - Production-like settings
STAGING_CONFIG = EnvironmentConfig(
    name="staging",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL,
    ),
    timeout={
        "factor_lookup": TimeoutConfig(
            timeout_seconds=5.0,
            operation_type=OperationType.FACTOR_LOOKUP,
        ),
        "llm_inference": TimeoutConfig(
            timeout_seconds=30.0,
            operation_type=OperationType.LLM_INFERENCE,
        ),
        "erp_api": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.ERP_API_CALL,
        ),
        "database": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.DATABASE_QUERY,
        ),
        "report_generation": TimeoutConfig(
            timeout_seconds=60.0,
            operation_type=OperationType.REPORT_GENERATION,
        ),
    },
    rate_limits={
        "factor_api": RateLimitConfig(
            requests_per_second=50.0,
            burst_size=100,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        "llm_api": RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
        "erp_api": RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ),
    },
    circuit_breakers={
        "factor_api": CircuitBreakerConfig(
            name="factor_api_staging",
            fail_max=5,
            timeout_duration=60,
        ),
        "llm_api": CircuitBreakerConfig(
            name="llm_api_staging",
            fail_max=3,
            timeout_duration=120,
        ),
        "erp_api": CircuitBreakerConfig(
            name="erp_api_staging",
            fail_max=5,
            timeout_duration=60,
        ),
    },
)


# Production Environment - Strict limits for reliability
PRODUCTION_CONFIG = EnvironmentConfig(
    name="production",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=True,
        jitter_range=0.1,
    ),
    timeout={
        "factor_lookup": TimeoutConfig(
            timeout_seconds=5.0,
            operation_type=OperationType.FACTOR_LOOKUP,
            raise_on_timeout=True,
        ),
        "llm_inference": TimeoutConfig(
            timeout_seconds=30.0,
            operation_type=OperationType.LLM_INFERENCE,
            raise_on_timeout=True,
        ),
        "erp_api": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.ERP_API_CALL,
            raise_on_timeout=True,
        ),
        "database": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.DATABASE_QUERY,
            raise_on_timeout=True,
        ),
        "report_generation": TimeoutConfig(
            timeout_seconds=60.0,
            operation_type=OperationType.REPORT_GENERATION,
            raise_on_timeout=True,
        ),
    },
    rate_limits={
        "factor_api": RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            raise_on_limit=True,
            wait_on_limit=False,
        ),
        "llm_api": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            raise_on_limit=True,
            wait_on_limit=False,
        ),
        "erp_api": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
            raise_on_limit=True,
            wait_on_limit=True,
        ),
    },
    circuit_breakers={
        "factor_api": CircuitBreakerConfig(
            name="factor_api_prod",
            fail_max=5,
            timeout_duration=60,
            reset_timeout=60,
        ),
        "llm_api": CircuitBreakerConfig(
            name="llm_api_prod",
            fail_max=3,
            timeout_duration=120,
            reset_timeout=120,
        ),
        "erp_api": CircuitBreakerConfig(
            name="erp_api_prod",
            fail_max=5,
            timeout_duration=60,
            reset_timeout=60,
        ),
        "database": CircuitBreakerConfig(
            name="database_prod",
            fail_max=3,
            timeout_duration=30,
            reset_timeout=30,
        ),
    },
)


# ==============================================================================
# Operation-Specific Configurations
# ==============================================================================


class OperationConfigs:
    """Recommended configurations for specific operations."""

    # Calculator operations
    EMISSION_CALCULATION = {
        "retry": RetryConfig(
            max_retries=2,
            base_delay=0.5,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=20.0,
            operation_type=OperationType.COMPUTATION,
        ),
        "fallback": FallbackConfig(
            strategy=FallbackStrategy.CACHED,
        ),
    }

    # Factor lookup operations
    FACTOR_LOOKUP = {
        "retry": RetryConfig(
            max_retries=3,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            retryable_exceptions=(ConnectionError, TimeoutError),
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=5.0,
            operation_type=OperationType.FACTOR_LOOKUP,
        ),
        "fallback": FallbackConfig(
            strategy=FallbackStrategy.CACHED,
        ),
        "rate_limit": RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
        ),
    }

    # LLM operations
    LLM_INFERENCE = {
        "retry": RetryConfig(
            max_retries=2,
            base_delay=2.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=30.0,
            operation_type=OperationType.LLM_INFERENCE,
        ),
        "fallback": FallbackConfig(
            strategy=FallbackStrategy.CACHED,
        ),
        "rate_limit": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
        ),
        "circuit_breaker": CircuitBreakerConfig(
            name="llm_inference",
            fail_max=3,
            timeout_duration=120,
        ),
    }

    # ERP operations
    ERP_DATA_FETCH = {
        "retry": RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=True,
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.ERP_API_CALL,
        ),
        "fallback": FallbackConfig(
            strategy=FallbackStrategy.CACHED,
        ),
        "rate_limit": RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
        ),
        "circuit_breaker": CircuitBreakerConfig(
            name="erp_api",
            fail_max=5,
            timeout_duration=60,
        ),
    }

    # Database operations
    DATABASE_QUERY = {
        "retry": RetryConfig(
            max_retries=3,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=10.0,
            operation_type=OperationType.DATABASE_QUERY,
        ),
        "circuit_breaker": CircuitBreakerConfig(
            name="database",
            fail_max=3,
            timeout_duration=30,
        ),
    }

    # Report generation
    REPORT_GENERATION = {
        "retry": RetryConfig(
            max_retries=2,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        "timeout": TimeoutConfig(
            timeout_seconds=60.0,
            operation_type=OperationType.REPORT_GENERATION,
        ),
        "fallback": FallbackConfig(
            strategy=FallbackStrategy.DEFAULT,
            default_value={"status": "pending", "message": "Report generation in progress"},
        ),
    }


# ==============================================================================
# Tenant-Specific Configurations
# ==============================================================================


class TenantConfigs:
    """Recommended configurations for different tenant tiers."""

    # Free tier - Conservative limits
    FREE_TIER = {
        "rate_limits": {
            "api_calls": RateLimitConfig(
                requests_per_second=1.0,
                burst_size=5,
            ),
            "calculations": RateLimitConfig(
                requests_per_second=2.0,
                burst_size=10,
            ),
        },
        "timeouts": {
            "default": TimeoutConfig(timeout_seconds=5.0),
        },
    }

    # Standard tier - Balanced limits
    STANDARD_TIER = {
        "rate_limits": {
            "api_calls": RateLimitConfig(
                requests_per_second=10.0,
                burst_size=20,
            ),
            "calculations": RateLimitConfig(
                requests_per_second=20.0,
                burst_size=50,
            ),
        },
        "timeouts": {
            "default": TimeoutConfig(timeout_seconds=10.0),
        },
    }

    # Enterprise tier - High limits
    ENTERPRISE_TIER = {
        "rate_limits": {
            "api_calls": RateLimitConfig(
                requests_per_second=100.0,
                burst_size=200,
            ),
            "calculations": RateLimitConfig(
                requests_per_second=200.0,
                burst_size=500,
            ),
        },
        "timeouts": {
            "default": TimeoutConfig(timeout_seconds=30.0),
        },
    }


# ==============================================================================
# Configuration Helpers
# ==============================================================================


def get_config_for_environment(env: str) -> EnvironmentConfig:
    """Get configuration for environment.

    Args:
        env: Environment name (development, staging, production)

    Returns:
        EnvironmentConfig for environment
    """
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "staging": STAGING_CONFIG,
        "production": PRODUCTION_CONFIG,
    }

    return configs.get(env.lower(), PRODUCTION_CONFIG)


def get_config_for_operation(operation: str) -> Dict[str, Any]:
    """Get configuration for operation type.

    Args:
        operation: Operation name

    Returns:
        Configuration dictionary
    """
    configs = {
        "emission_calculation": OperationConfigs.EMISSION_CALCULATION,
        "factor_lookup": OperationConfigs.FACTOR_LOOKUP,
        "llm_inference": OperationConfigs.LLM_INFERENCE,
        "erp_fetch": OperationConfigs.ERP_DATA_FETCH,
        "database_query": OperationConfigs.DATABASE_QUERY,
        "report_generation": OperationConfigs.REPORT_GENERATION,
    }

    return configs.get(operation, {})


def get_config_for_tenant(tier: str) -> Dict[str, Any]:
    """Get configuration for tenant tier.

    Args:
        tier: Tenant tier (free, standard, enterprise)

    Returns:
        Configuration dictionary
    """
    configs = {
        "free": TenantConfigs.FREE_TIER,
        "standard": TenantConfigs.STANDARD_TIER,
        "enterprise": TenantConfigs.ENTERPRISE_TIER,
    }

    return configs.get(tier.lower(), TenantConfigs.STANDARD_TIER)


__all__ = [
    "EnvironmentConfig",
    "DEVELOPMENT_CONFIG",
    "STAGING_CONFIG",
    "PRODUCTION_CONFIG",
    "OperationConfigs",
    "TenantConfigs",
    "get_config_for_environment",
    "get_config_for_operation",
    "get_config_for_tenant",
]
