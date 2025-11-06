"""
Tests for Configuration System
===============================

Test coverage:
- Config schemas validation
- ConfigManager functionality
- Hot-reload support
- Override mechanism
- Dependency injection container
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any

from greenlang.config import (
    Environment,
    GreenLangConfig,
    ConfigManager,
    get_config,
    override_config,
    ServiceContainer,
    ServiceLifetime,
    create_test_config,
    validate_config,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def reset_config():
    """Reset config manager before each test."""
    ConfigManager.reset_instance()
    yield
    ConfigManager.reset_instance()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return create_test_config()


@pytest.fixture
def config_manager():
    """Create config manager instance."""
    return ConfigManager.get_instance()


@pytest.fixture
def service_container(test_config):
    """Create service container."""
    return ServiceContainer(test_config)


# ==============================================================================
# Config Schema Tests
# ==============================================================================

def test_default_config_creation():
    """Test creating config with defaults."""
    config = GreenLangConfig()

    assert config.environment == Environment.DEVELOPMENT
    assert config.app_name == "GreenLang"
    assert config.llm.provider == "openai"
    assert config.database.provider == "postgresql"
    assert config.cache.provider == "memory"


def test_test_config_creation():
    """Test creating test configuration."""
    config = create_test_config()

    assert config.environment == Environment.TEST
    assert config.debug is True
    assert config.database.provider == "memory"
    assert config.cache.provider == "memory"


def test_config_validation():
    """Test config field validation."""
    # Valid config
    config = GreenLangConfig(
        llm={"temperature": 0.5, "max_tokens": 1000}
    )
    assert config.llm.temperature == 0.5

    # Invalid temperature (too high)
    with pytest.raises(Exception):  # Pydantic ValidationError
        GreenLangConfig(llm={"temperature": 3.0})

    # Invalid max_tokens (negative)
    with pytest.raises(Exception):
        GreenLangConfig(llm={"max_tokens": -1})


def test_environment_detection():
    """Test environment type methods."""
    dev_config = GreenLangConfig(environment=Environment.DEVELOPMENT)
    assert dev_config.is_development() is True
    assert dev_config.is_production() is False
    assert dev_config.is_test() is False

    prod_config = GreenLangConfig(environment=Environment.PRODUCTION)
    assert prod_config.is_production() is True
    assert prod_config.is_development() is False


def test_database_connection_string():
    """Test database connection string generation."""
    config = GreenLangConfig(
        database={
            "provider": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "user",
            "password": "pass123"
        }
    )

    conn_str = config.database.get_connection_string(hide_password=True)
    assert "postgresql://" in conn_str
    assert "***" in conn_str  # Password hidden
    assert "pass123" not in conn_str

    conn_str_full = config.database.get_connection_string(hide_password=False)
    assert "pass123" in conn_str_full


# ==============================================================================
# ConfigManager Tests
# ==============================================================================

def test_config_manager_singleton(config_manager):
    """Test ConfigManager is singleton."""
    manager1 = ConfigManager.get_instance()
    manager2 = ConfigManager.get_instance()

    assert manager1 is manager2
    assert manager1 is config_manager


def test_config_manager_load_from_env(config_manager):
    """Test loading config from environment."""
    # Set env vars
    os.environ["GL_ENVIRONMENT"] = "test"
    os.environ["GL_DEBUG"] = "true"

    config = config_manager.load_from_env()

    assert config.environment == Environment.TEST
    assert config.debug is True

    # Cleanup
    os.environ.pop("GL_ENVIRONMENT", None)
    os.environ.pop("GL_DEBUG", None)


def test_config_manager_get_config(config_manager):
    """Test getting config (auto-loads if not loaded)."""
    config = config_manager.get_config()

    assert config is not None
    assert isinstance(config, GreenLangConfig)


def test_config_manager_override(config_manager):
    """Test config override mechanism."""
    # Initial config
    config1 = config_manager.get_config()
    initial_debug = config1.debug

    # Override
    with config_manager.override(debug=True, app_name="TestApp"):
        config2 = config_manager.get_config()
        assert config2.debug is True
        assert config2.app_name == "TestApp"

    # After override context
    config3 = config_manager.get_config()
    assert config3.debug == initial_debug  # Restored


def test_config_manager_nested_overrides(config_manager):
    """Test nested config overrides."""
    with config_manager.override(debug=True):
        config1 = config_manager.get_config()
        assert config1.debug is True

        with config_manager.override(app_name="Nested"):
            config2 = config_manager.get_config()
            assert config2.debug is True  # Still True
            assert config2.app_name == "Nested"

        config3 = config_manager.get_config()
        assert config3.debug is True
        assert config3.app_name != "Nested"  # Reverted


def test_config_manager_reload_callback(config_manager):
    """Test reload callbacks."""
    callback_calls = []

    def on_reload(config):
        callback_calls.append(config.environment)

    config_manager.add_reload_callback(on_reload)
    config_manager.reload()

    assert len(callback_calls) == 1


def test_global_get_config():
    """Test global get_config() function."""
    config = get_config()

    assert config is not None
    assert isinstance(config, GreenLangConfig)


def test_global_override_config():
    """Test global override_config() function."""
    with override_config(debug=True):
        config = get_config()
        assert config.debug is True


# ==============================================================================
# Hot-Reload Tests
# ==============================================================================

def test_config_manager_load_from_file():
    """Test loading config from YAML file."""
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
environment: test
app_name: TestApp
debug: true
llm:
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.5
""")
        config_path = Path(f.name)

    try:
        manager = ConfigManager.get_instance()
        config = manager.load_from_file(config_path)

        assert config.environment == Environment.TEST
        assert config.app_name == "TestApp"
        assert config.debug is True
        assert config.llm.model == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.5

    finally:
        config_path.unlink()


# ==============================================================================
# Service Container Tests
# ==============================================================================

class DummyService:
    """Dummy service for testing."""

    def __init__(self, value: str):
        self.value = value


class AnotherService:
    """Another dummy service."""

    def __init__(self, dep: DummyService):
        self.dep = dep


def test_service_container_singleton(service_container):
    """Test singleton service registration."""
    service_container.register_singleton(
        DummyService,
        lambda c: DummyService("singleton")
    )

    instance1 = service_container.resolve(DummyService)
    instance2 = service_container.resolve(DummyService)

    assert instance1 is instance2  # Same instance
    assert instance1.value == "singleton"


def test_service_container_transient(service_container):
    """Test transient service registration."""
    service_container.register_transient(
        DummyService,
        lambda c: DummyService("transient")
    )

    instance1 = service_container.resolve(DummyService)
    instance2 = service_container.resolve(DummyService)

    assert instance1 is not instance2  # Different instances
    assert instance1.value == "transient"
    assert instance2.value == "transient"


def test_service_container_scoped(service_container):
    """Test scoped service registration."""
    service_container.register_scoped(
        DummyService,
        lambda c: DummyService("scoped")
    )

    # Within same scope
    instance1 = service_container.resolve(DummyService)
    instance2 = service_container.resolve(DummyService)
    assert instance1 is instance2  # Same instance in same scope

    # New scope
    service_container.clear_scope()
    instance3 = service_container.resolve(DummyService)
    assert instance3 is not instance1  # Different instance in new scope


def test_service_container_register_instance(service_container):
    """Test registering existing instance."""
    existing = DummyService("existing")
    service_container.register_instance(DummyService, existing)

    resolved = service_container.resolve(DummyService)
    assert resolved is existing


def test_service_container_dependency_resolution(service_container):
    """Test resolving service with dependencies."""
    service_container.register_singleton(
        DummyService,
        lambda c: DummyService("dep")
    )

    service_container.register_singleton(
        AnotherService,
        lambda c: AnotherService(c.resolve(DummyService))
    )

    service = service_container.resolve(AnotherService)

    assert service.dep.value == "dep"


def test_service_container_try_resolve(service_container):
    """Test try_resolve returns None for unregistered services."""
    result = service_container.try_resolve(DummyService)
    assert result is None

    # Register and try again
    service_container.register_singleton(
        DummyService,
        lambda c: DummyService("test")
    )

    result = service_container.try_resolve(DummyService)
    assert result is not None


def test_service_container_is_registered(service_container):
    """Test is_registered check."""
    assert service_container.is_registered(DummyService) is False

    service_container.register_singleton(
        DummyService,
        lambda c: DummyService("test")
    )

    assert service_container.is_registered(DummyService) is True


def test_service_container_scope_context(service_container):
    """Test service scope context manager."""
    service_container.register_scoped(
        DummyService,
        lambda c: DummyService("scoped")
    )

    # First scope
    with service_container.create_scope() as scope:
        instance1 = scope.resolve(DummyService)
        instance2 = scope.resolve(DummyService)
        assert instance1 is instance2

    # Second scope (new instance)
    with service_container.create_scope() as scope:
        instance3 = scope.resolve(DummyService)
        assert instance3 is not instance1


# ==============================================================================
# Config Validation Tests
# ==============================================================================

def test_validate_config_production_warnings():
    """Test config validation detects production issues."""
    prod_config = GreenLangConfig(
        environment=Environment.PRODUCTION,
        debug=True,  # Warning: debug in prod
        database={"provider": "memory"},  # Warning: memory db in prod
        cache={"provider": "memory"},  # Warning: memory cache in prod
        logging={"level": "DEBUG"},  # Warning: debug logging in prod
    )

    warnings = validate_config(prod_config)

    assert len(warnings) > 0
    assert any("debug" in w.lower() for w in warnings)
    assert any("memory database" in w.lower() for w in warnings)


def test_validate_config_security_warnings():
    """Test config validation detects security issues."""
    config = GreenLangConfig(
        security={
            "enable_authentication": True,
            "jwt_secret": None  # Warning: auth enabled but no secret
        }
    )

    warnings = validate_config(config)

    assert any("jwt" in w.lower() or "secret" in w.lower() for w in warnings)


def test_validate_config_valid():
    """Test config validation passes for valid config."""
    config = create_test_config()
    warnings = validate_config(config)

    # Test config should have minimal or no warnings
    assert isinstance(warnings, list)


# ==============================================================================
# Integration Tests
# ==============================================================================

def test_full_workflow():
    """Test complete workflow: config -> container -> services."""
    # 1. Create config
    config = create_test_config(
        llm={"model": "gpt-4", "temperature": 0.0}
    )

    # 2. Create container
    container = ServiceContainer(config)

    # 3. Register service that uses config
    class ConfigAwareService:
        def __init__(self, cfg: GreenLangConfig):
            self.config = cfg

    container.register_instance(GreenLangConfig, config)
    container.register_singleton(
        ConfigAwareService,
        lambda c: ConfigAwareService(c.resolve(GreenLangConfig))
    )

    # 4. Resolve and verify
    service = container.resolve(ConfigAwareService)

    assert service.config is config
    assert service.config.llm.model == "gpt-4"


def test_config_override_with_container():
    """Test config override works with container."""
    manager = ConfigManager.get_instance()

    with manager.override(llm={"model": "gpt-3.5"}):
        config = manager.get_config()
        container = ServiceContainer(config)

        assert container.get_config().llm.model == "gpt-3.5"


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
