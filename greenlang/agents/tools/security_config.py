# -*- coding: utf-8 -*-
"""
GreenLang Tool Security Configuration
======================================

Centralized security configuration for tool execution.

Features:
- Global security settings
- Per-tool security overrides
- Input validation configuration
- Rate limiting configuration
- Audit logging configuration
- Execution limits and constraints
- Tool whitelisting/blacklisting

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==============================================================================
# Security Configuration
# ==============================================================================

@dataclass
class SecurityConfig:
    """
    Security configuration for tool execution.

    This provides centralized control over all security features including
    validation, rate limiting, audit logging, and execution constraints.
    """

    # =========================================================================
    # Input Validation
    # =========================================================================

    enable_validation: bool = True
    """Enable input validation."""

    strict_validation: bool = False
    """Fail on warnings (not just errors)."""

    sanitize_inputs: bool = True
    """Automatically sanitize/normalize inputs."""

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    enable_rate_limiting: bool = True
    """Enable rate limiting."""

    default_rate_per_second: int = 10
    """Default rate limit (tokens per second)."""

    default_burst_size: int = 20
    """Default burst capacity."""

    per_tool_limits: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    """
    Per-tool rate limit overrides.
    Format: {"tool_name": (rate_per_second, burst_size)}

    Example:
        {
            "calculate_emissions": (100, 200),  # High-frequency tool
            "grid_integration": (5, 10),        # Resource-intensive tool
        }
    """

    per_user_rate_limiting: bool = False
    """Enable per-user rate limiting."""

    # =========================================================================
    # Audit Logging
    # =========================================================================

    enable_audit_logging: bool = True
    """Enable audit logging."""

    audit_log_dir: Path = field(default_factory=lambda: Path("logs/audit"))
    """Directory for audit log files."""

    audit_retention_days: int = 90
    """Days to retain audit logs."""

    audit_log_successes: bool = True
    """Log successful executions."""

    audit_log_failures: bool = True
    """Log failed executions."""

    # =========================================================================
    # Tool Access Control
    # =========================================================================

    tool_whitelist: Optional[Set[str]] = None
    """
    Whitelist of allowed tools (None = all allowed).
    If set, only tools in this set can be executed.
    """

    tool_blacklist: Set[str] = field(default_factory=set)
    """
    Blacklist of forbidden tools.
    Tools in this set cannot be executed.
    """

    max_concurrent_tools: int = 5
    """Maximum number of concurrent tool executions."""

    # =========================================================================
    # Execution Limits
    # =========================================================================

    max_execution_time_seconds: float = 300.0
    """Maximum execution time per tool call (seconds)."""

    max_memory_mb: int = 1024
    """Maximum memory usage per tool (MB) - not yet enforced."""

    max_retries: int = 3
    """Maximum retry attempts for failed executions."""

    # =========================================================================
    # Privacy & Security
    # =========================================================================

    hash_sensitive_inputs: bool = True
    """Hash sensitive inputs in logs (privacy protection)."""

    sensitive_param_names: Set[str] = field(default_factory=lambda: {
        "password", "api_key", "secret", "token", "credential",
        "ssn", "credit_card", "private_key"
    })
    """Parameter names considered sensitive."""

    block_injection_attempts: bool = True
    """Block SQL/command injection attempts."""

    # =========================================================================
    # Development & Testing
    # =========================================================================

    debug_mode: bool = False
    """Enable debug mode (more verbose logging, relaxed constraints)."""

    test_mode: bool = False
    """Enable test mode (bypass certain checks)."""

    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check if tool is allowed to execute.

        Args:
            tool_name: Tool name to check

        Returns:
            True if allowed, False otherwise
        """
        # Check blacklist first
        if tool_name in self.tool_blacklist:
            return False

        # Check whitelist if configured
        if self.tool_whitelist is not None:
            return tool_name in self.tool_whitelist

        # Default: allow
        return True

    def get_rate_limit(self, tool_name: str) -> Tuple[int, int]:
        """
        Get rate limit for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            Tuple of (rate_per_second, burst_size)
        """
        return self.per_tool_limits.get(
            tool_name,
            (self.default_rate_per_second, self.default_burst_size)
        )

    def is_sensitive_param(self, param_name: str) -> bool:
        """
        Check if parameter name is sensitive.

        Args:
            param_name: Parameter name to check

        Returns:
            True if sensitive, False otherwise
        """
        param_lower = param_name.lower()
        return any(
            sensitive in param_lower
            for sensitive in self.sensitive_param_names
        )

    def to_dict(self) -> Dict[str, any]:
        """Convert configuration to dictionary."""
        return {
            # Input Validation
            "enable_validation": self.enable_validation,
            "strict_validation": self.strict_validation,
            "sanitize_inputs": self.sanitize_inputs,

            # Rate Limiting
            "enable_rate_limiting": self.enable_rate_limiting,
            "default_rate_per_second": self.default_rate_per_second,
            "default_burst_size": self.default_burst_size,
            "per_tool_limits": self.per_tool_limits,
            "per_user_rate_limiting": self.per_user_rate_limiting,

            # Audit Logging
            "enable_audit_logging": self.enable_audit_logging,
            "audit_log_dir": str(self.audit_log_dir),
            "audit_retention_days": self.audit_retention_days,
            "audit_log_successes": self.audit_log_successes,
            "audit_log_failures": self.audit_log_failures,

            # Tool Access Control
            "tool_whitelist": list(self.tool_whitelist) if self.tool_whitelist else None,
            "tool_blacklist": list(self.tool_blacklist),
            "max_concurrent_tools": self.max_concurrent_tools,

            # Execution Limits
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_retries": self.max_retries,

            # Privacy & Security
            "hash_sensitive_inputs": self.hash_sensitive_inputs,
            "sensitive_param_names": list(self.sensitive_param_names),
            "block_injection_attempts": self.block_injection_attempts,

            # Development & Testing
            "debug_mode": self.debug_mode,
            "test_mode": self.test_mode,
        }

    def __repr__(self) -> str:
        return (
            f"SecurityConfig("
            f"validation={self.enable_validation}, "
            f"rate_limiting={self.enable_rate_limiting}, "
            f"audit={self.enable_audit_logging}, "
            f"debug={self.debug_mode})"
        )


# ==============================================================================
# Security Configuration Presets
# ==============================================================================

def development_config() -> SecurityConfig:
    """
    Development configuration with relaxed security.

    Use for local development and testing only.
    """
    return SecurityConfig(
        enable_validation=True,
        strict_validation=False,
        enable_rate_limiting=False,  # Disabled for dev
        enable_audit_logging=True,
        audit_retention_days=7,  # Shorter retention
        max_execution_time_seconds=600.0,  # Longer timeout
        debug_mode=True,
        test_mode=False,
    )


def testing_config() -> SecurityConfig:
    """
    Testing configuration for unit/integration tests.

    Minimal security for fast test execution.
    """
    return SecurityConfig(
        enable_validation=True,
        strict_validation=False,
        enable_rate_limiting=False,  # Disabled for tests
        enable_audit_logging=False,  # Disabled for tests
        max_execution_time_seconds=30.0,
        debug_mode=False,
        test_mode=True,
    )


def production_config() -> SecurityConfig:
    """
    Production configuration with maximum security.

    Use for production deployments.
    """
    return SecurityConfig(
        enable_validation=True,
        strict_validation=True,  # Fail on warnings
        enable_rate_limiting=True,
        default_rate_per_second=10,
        default_burst_size=20,
        per_user_rate_limiting=True,
        enable_audit_logging=True,
        audit_retention_days=90,
        audit_log_successes=True,
        audit_log_failures=True,
        max_concurrent_tools=5,
        max_execution_time_seconds=300.0,
        max_memory_mb=1024,
        max_retries=3,
        hash_sensitive_inputs=True,
        block_injection_attempts=True,
        debug_mode=False,
        test_mode=False,
    )


def high_security_config() -> SecurityConfig:
    """
    High security configuration for sensitive environments.

    Strictest security settings.
    """
    config = production_config()
    config.strict_validation = True
    config.default_rate_per_second = 5  # Lower rate
    config.default_burst_size = 10
    config.max_concurrent_tools = 3
    config.max_execution_time_seconds = 120.0  # Shorter timeout
    config.max_retries = 1  # Fewer retries

    return config


# ==============================================================================
# Global Configuration Management
# ==============================================================================

_global_security_config: Optional[SecurityConfig] = None
_config_lock = None


def get_security_config() -> SecurityConfig:
    """
    Get global security configuration (singleton).

    Returns:
        Global SecurityConfig instance
    """
    global _global_security_config

    if _global_security_config is None:
        # Default to production config for safety
        _global_security_config = production_config()
        logger.info("Initialized security config with production defaults")

    return _global_security_config


def configure_security(
    preset: Optional[str] = None,
    **kwargs
) -> SecurityConfig:
    """
    Configure global security settings.

    Args:
        preset: Preset configuration name:
            - "development": Relaxed security for development
            - "testing": Minimal security for tests
            - "production": Maximum security for production
            - "high_security": Strictest security
            - None: Custom configuration from kwargs
        **kwargs: Security configuration parameters (override preset)

    Returns:
        Configured SecurityConfig instance

    Example:
        # Use production preset
        configure_security(preset="production")

        # Use production preset with overrides
        configure_security(
            preset="production",
            default_rate_per_second=20,
            debug_mode=True
        )

        # Fully custom configuration
        configure_security(
            enable_validation=True,
            enable_rate_limiting=True,
            audit_retention_days=30
        )
    """
    global _global_security_config

    # Select preset or create default
    if preset == "development":
        config = development_config()
    elif preset == "testing":
        config = testing_config()
    elif preset == "production":
        config = production_config()
    elif preset == "high_security":
        config = high_security_config()
    elif preset is None:
        config = SecurityConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Valid presets: development, testing, production, high_security"
        )

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown security config parameter: {key}")

    _global_security_config = config
    logger.info(f"Configured security: {config}")

    return config


def reset_security_config() -> None:
    """Reset global security configuration to default."""
    global _global_security_config
    _global_security_config = None
    logger.info("Reset security configuration")


# ==============================================================================
# Context Manager for Temporary Configuration
# ==============================================================================

class SecurityContext:
    """
    Context manager for temporary security configuration changes.

    Example:
        with SecurityContext(debug_mode=True, enable_rate_limiting=False):
            # Execute with modified config
            result = tool.execute(...)
        # Config restored after context
    """

    def __init__(self, **overrides):
        """
        Initialize context manager.

        Args:
            **overrides: Temporary configuration overrides
        """
        self.overrides = overrides
        self.original_config = None

    def __enter__(self):
        """Save current config and apply overrides."""
        global _global_security_config

        # Save current config
        current = get_security_config()
        self.original_config = SecurityConfig(**current.to_dict())

        # Apply overrides
        for key, value in self.overrides.items():
            if hasattr(_global_security_config, key):
                setattr(_global_security_config, key, value)

        return _global_security_config

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original config."""
        global _global_security_config
        _global_security_config = self.original_config
