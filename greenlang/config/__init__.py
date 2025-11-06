"""
GreenLang Configuration Package
================================

Centralized configuration management with:
- Environment-based configuration
- Type-safe Pydantic schemas
- Hot-reload support
- Dependency injection container
- Override mechanism for testing

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from greenlang.config.schemas import (
    Environment,
    GreenLangConfig,
    LLMProviderConfig,
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    ObservabilityConfig,
    SecurityConfig,
    load_config_from_env,
    load_config_from_file,
    create_test_config,
)

from greenlang.config.manager import (
    ConfigManager,
    get_config,
    reload_config,
    override_config,
    detect_environment,
    validate_config,
)

from greenlang.config.container import (
    ServiceContainer,
    ServiceLifetime,
    get_container,
    reset_container,
    register_default_services,
    inject,
)


__all__ = [
    # Schemas
    "Environment",
    "GreenLangConfig",
    "LLMProviderConfig",
    "DatabaseConfig",
    "CacheConfig",
    "LoggingConfig",
    "ObservabilityConfig",
    "SecurityConfig",
    "load_config_from_env",
    "load_config_from_file",
    "create_test_config",
    # Manager
    "ConfigManager",
    "get_config",
    "reload_config",
    "override_config",
    "detect_environment",
    "validate_config",
    # Container
    "ServiceContainer",
    "ServiceLifetime",
    "get_container",
    "reset_container",
    "register_default_services",
    "inject",
]
