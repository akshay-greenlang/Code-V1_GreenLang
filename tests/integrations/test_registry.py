"""
Tests for Integration Registry
================================

Tests for connector registration, discovery, and factory patterns.

Author: GreenLang Backend Team
Date: 2025-12-01
"""

import pytest
from pydantic import BaseModel, Field

from greenlang.integrations.base_connector import (
    BaseConnector,
    ConnectorConfig,
)
from greenlang.integrations.registry import (
    IntegrationRegistry,
    ConnectorRegistration,
)


# Test models
class TestQuery(BaseModel):
    value: int = 1


class TestPayload(BaseModel):
    result: int = 1


class TestConfig(ConnectorConfig):
    test_field: str = "test"


class TestConnectorV1(BaseConnector[TestQuery, TestPayload, TestConfig]):
    """Test connector v1."""

    connector_id = "test-conn"
    connector_version = "1.0.0"

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def _health_check_impl(self) -> bool:
        return True

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        return TestPayload(result=1)


class TestConnectorV2(BaseConnector[TestQuery, TestPayload, TestConfig]):
    """Test connector v2."""

    connector_id = "test-conn"
    connector_version = "2.0.0"

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def _health_check_impl(self) -> bool:
        return True

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        return TestPayload(result=2)


class TestIntegrationRegistry:
    """Test suite for IntegrationRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return IntegrationRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert isinstance(registry, IntegrationRegistry)
        assert len(registry._registry) == 0
        assert len(registry._metadata) == 0

    def test_register_connector(self, registry):
        """Test connector registration."""
        registry.register(
            TestConnectorV1,
            description="Test connector v1",
            supported_protocols=["test"]
        )

        assert registry.is_registered("test-conn")
        assert registry.get_version("test-conn") == "1.0.0"

        # Check metadata
        info = registry.get_connector_info("test-conn")
        assert info is not None
        assert info.connector_type == "test-conn"
        assert info.version == "1.0.0"
        assert info.description == "Test connector v1"
        assert "test" in info.supported_protocols

    def test_register_multiple_versions(self, registry):
        """Test registering multiple versions."""
        # Register v1
        registry.register(TestConnectorV1)
        assert registry.get_version("test-conn") == "1.0.0"

        # Register v2 (should replace)
        registry.register(TestConnectorV2)
        assert registry.get_version("test-conn") == "2.0.0"

    def test_unregister_connector(self, registry):
        """Test unregistering connector."""
        registry.register(TestConnectorV1)
        assert registry.is_registered("test-conn")

        registry.unregister("test-conn")
        assert not registry.is_registered("test-conn")
        assert registry.get_connector_info("test-conn") is None

    def test_create_connector(self, registry):
        """Test creating connector from registry."""
        registry.register(TestConnectorV1)

        config = TestConfig(
            connector_id="instance-1",
            connector_type="test"
        )

        connector = registry.create_connector("test-conn", config)

        assert isinstance(connector, TestConnectorV1)
        assert connector.config == config

    def test_create_connector_not_found(self, registry):
        """Test creating non-existent connector."""
        config = TestConfig(
            connector_id="instance-1",
            connector_type="test"
        )

        with pytest.raises(ValueError, match="not registered"):
            registry.create_connector("non-existent", config)

    def test_get_connector_class(self, registry):
        """Test getting connector class."""
        registry.register(TestConnectorV1)

        connector_class = registry.get_connector_class("test-conn")
        assert connector_class == TestConnectorV1

        # Non-existent
        connector_class = registry.get_connector_class("non-existent")
        assert connector_class is None

    def test_list_connectors(self, registry):
        """Test listing connectors."""
        # Empty initially
        assert len(registry.list_connectors()) == 0

        # Register some connectors
        registry.register(TestConnectorV1)

        class AnotherConnector(TestConnectorV1):
            connector_id = "another-conn"
            connector_version = "1.0.0"

        registry.register(AnotherConnector)

        # List should have 2
        connectors = registry.list_connectors()
        assert len(connectors) == 2

        connector_ids = [c.connector_type for c in connectors]
        assert "test-conn" in connector_ids
        assert "another-conn" in connector_ids

    def test_validate_config(self, registry):
        """Test config validation."""
        registry.register(TestConnectorV1)

        config = TestConfig(
            connector_id="test",
            connector_type="test"
        )

        # Should validate successfully
        assert registry.validate_config("test-conn", config) is True

    def test_validate_config_not_registered(self, registry):
        """Test validation for unregistered connector."""
        config = TestConfig(
            connector_id="test",
            connector_type="test"
        )

        with pytest.raises(ValueError, match="not registered"):
            registry.validate_config("non-existent", config)


class TestConnectorRegistration:
    """Test ConnectorRegistration model."""

    def test_registration_model(self):
        """Test registration model creation."""
        reg = ConnectorRegistration(
            connector_type="test",
            connector_class_name="TestConnector",
            version="1.2.3",
            description="Test connector",
            supported_protocols=["http", "ws"]
        )

        assert reg.connector_type == "test"
        assert reg.version == "1.2.3"
        assert len(reg.supported_protocols) == 2

    def test_invalid_version(self):
        """Test invalid version string."""
        with pytest.raises(ValueError):
            ConnectorRegistration(
                connector_type="test",
                connector_class_name="TestConnector",
                version="not-a-version",
                description="Test"
            )
