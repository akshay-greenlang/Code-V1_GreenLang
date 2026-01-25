"""
Integration Registry - Connector Discovery and Version Management
==================================================================

Central registry for all integration connectors.

Features:
- Connector discovery and registration
- Version management
- Configuration validation
- Capability negotiation
- Factory pattern for connector creation

Example:
    >>> registry = IntegrationRegistry()
    >>> registry.register(SCADAConnector)
    >>>
    >>> # Create connector from registry
    >>> connector = registry.create_connector("scada", config)
    >>>
    >>> # List available connectors
    >>> connectors = registry.list_connectors()

Author: GreenLang Backend Team
Date: 2025-12-01
"""

from typing import Dict, List, Type, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from packaging import version
import logging

from greenlang.integrations.base_connector import (
    BaseConnector,
    ConnectorConfig
)

logger = logging.getLogger(__name__)


class ConnectorRegistration(BaseModel):
    """Connector registration metadata."""

    connector_type: str = Field(..., description="Connector type identifier")
    connector_class_name: str = Field(..., description="Connector class name")
    version: str = Field(..., description="Connector version (semver)")
    description: str = Field(..., description="Connector description")
    supported_protocols: List[str] = Field(default_factory=list, description="Supported protocols")
    config_schema: Optional[Dict[str, Any]] = Field(default=None, description="Config schema")
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Validate version is valid semver."""
        try:
            version.parse(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid version: {v}") from e


class IntegrationRegistry:
    """
    Central registry for integration connectors.

    Provides connector discovery, registration, and factory creation.

    Example:
        >>> from greenlang.integrations import IntegrationRegistry
        >>> from greenlang.integrations.scada_connector import SCADAConnector
        >>>
        >>> # Create registry and register connectors
        >>> registry = IntegrationRegistry()
        >>> registry.register(SCADAConnector)
        >>>
        >>> # Create connector instance
        >>> config = SCADAConfig(...)
        >>> connector = registry.create_connector("scada", config)
        >>>
        >>> # Get connector info
        >>> info = registry.get_connector_info("scada")
        >>> print(f"Version: {info.version}")
    """

    def __init__(self):
        """Initialize integration registry."""
        # Registry: connector_type -> connector_class
        self._registry: Dict[str, Type[BaseConnector]] = {}

        # Metadata: connector_type -> registration info
        self._metadata: Dict[str, ConnectorRegistration] = {}

        logger.info("Initialized IntegrationRegistry")

    def register(
        self,
        connector_class: Type[BaseConnector],
        description: str = "",
        supported_protocols: Optional[List[str]] = None
    ) -> None:
        """
        Register a connector class.

        Args:
            connector_class: Connector class to register
            description: Connector description
            supported_protocols: List of supported protocols

        Raises:
            ValueError: If connector already registered with different version
        """
        # Extract metadata from connector class
        connector_type = getattr(connector_class, 'connector_id', None)
        connector_version = getattr(connector_class, 'connector_version', '0.1.0')

        if not connector_type:
            raise ValueError(
                f"Connector class {connector_class.__name__} must define 'connector_id'"
            )

        # Check if already registered
        if connector_type in self._registry:
            existing_version = self._metadata[connector_type].version
            if version.parse(existing_version) != version.parse(connector_version):
                logger.warning(
                    f"Replacing connector {connector_type} "
                    f"v{existing_version} with v{connector_version}"
                )

        # Register connector
        self._registry[connector_type] = connector_class

        # Store metadata
        self._metadata[connector_type] = ConnectorRegistration(
            connector_type=connector_type,
            connector_class_name=connector_class.__name__,
            version=connector_version,
            description=description or connector_class.__doc__ or "",
            supported_protocols=supported_protocols or []
        )

        logger.info(
            f"Registered connector: {connector_type} "
            f"v{connector_version} ({connector_class.__name__})"
        )

    def unregister(self, connector_type: str) -> None:
        """
        Unregister a connector.

        Args:
            connector_type: Connector type identifier
        """
        if connector_type in self._registry:
            del self._registry[connector_type]
            del self._metadata[connector_type]
            logger.info(f"Unregistered connector: {connector_type}")

    def create_connector(
        self,
        connector_type: str,
        config: ConnectorConfig
    ) -> BaseConnector:
        """
        Create a connector instance from the registry.

        Args:
            connector_type: Connector type identifier
            config: Connector configuration

        Returns:
            Connector instance

        Raises:
            ValueError: If connector type not found
        """
        if connector_type not in self._registry:
            raise ValueError(
                f"Connector type '{connector_type}' not registered. "
                f"Available: {list(self._registry.keys())}"
            )

        connector_class = self._registry[connector_type]

        try:
            connector = connector_class(config)
            logger.info(f"Created connector instance: {connector_type}")
            return connector

        except Exception as e:
            logger.error(f"Failed to create connector {connector_type}: {e}")
            raise

    def get_connector_class(self, connector_type: str) -> Optional[Type[BaseConnector]]:
        """
        Get connector class by type.

        Args:
            connector_type: Connector type identifier

        Returns:
            Connector class or None
        """
        return self._registry.get(connector_type)

    def get_connector_info(self, connector_type: str) -> Optional[ConnectorRegistration]:
        """
        Get connector registration metadata.

        Args:
            connector_type: Connector type identifier

        Returns:
            Registration metadata or None
        """
        return self._metadata.get(connector_type)

    def list_connectors(self) -> List[ConnectorRegistration]:
        """
        List all registered connectors.

        Returns:
            List of connector registrations
        """
        return list(self._metadata.values())

    def is_registered(self, connector_type: str) -> bool:
        """
        Check if a connector type is registered.

        Args:
            connector_type: Connector type identifier

        Returns:
            True if registered
        """
        return connector_type in self._registry

    def get_version(self, connector_type: str) -> Optional[str]:
        """
        Get version of a registered connector.

        Args:
            connector_type: Connector type identifier

        Returns:
            Version string or None
        """
        if connector_type in self._metadata:
            return self._metadata[connector_type].version
        return None

    def validate_config(
        self,
        connector_type: str,
        config: ConnectorConfig
    ) -> bool:
        """
        Validate connector configuration.

        Args:
            connector_type: Connector type identifier
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if connector_type not in self._registry:
            raise ValueError(f"Connector type not registered: {connector_type}")

        # Basic validation - config is already Pydantic model
        # Additional custom validation can be added here
        return True


# Global registry instance
_global_registry: Optional[IntegrationRegistry] = None


def get_registry() -> IntegrationRegistry:
    """
    Get global integration registry instance.

    Returns:
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = IntegrationRegistry()
        _register_builtin_connectors()
    return _global_registry


def _register_builtin_connectors():
    """Register built-in connectors on first access."""
    registry = _global_registry

    try:
        from greenlang.integrations.scada_connector import SCADAConnector
        registry.register(
            SCADAConnector,
            description="SCADA system integration (OPC UA, Modbus, DNP3, BACnet)",
            supported_protocols=["opcua", "modbus", "dnp3", "bacnet"]
        )
    except ImportError as e:
        logger.debug(f"Could not register SCADAConnector: {e}")

    try:
        from greenlang.integrations.erp_connector import ERPConnector
        registry.register(
            ERPConnector,
            description="ERP system integration (SAP, Oracle, Dynamics, NetSuite)",
            supported_protocols=["rest", "odata", "soap"]
        )
    except ImportError as e:
        logger.debug(f"Could not register ERPConnector: {e}")

    try:
        from greenlang.integrations.cems_connector import CEMSConnector
        registry.register(
            CEMSConnector,
            description="Continuous Emissions Monitoring System integration",
            supported_protocols=["modbus", "profibus"]
        )
    except ImportError as e:
        logger.debug(f"Could not register CEMSConnector: {e}")

    try:
        from greenlang.integrations.historian_connector import HistorianConnector
        registry.register(
            HistorianConnector,
            description="Time-series historian integration (PI, PHD, GE)",
            supported_protocols=["pi-sdk", "opc-hda"]
        )
    except ImportError as e:
        logger.debug(f"Could not register HistorianConnector: {e}")

    try:
        from greenlang.integrations.cmms_connector import CMMSConnector
        registry.register(
            CMMSConnector,
            description="Maintenance management system integration (Maximo, SAP PM)",
            supported_protocols=["rest", "soap"]
        )
    except ImportError as e:
        logger.debug(f"Could not register CMMSConnector: {e}")

    logger.info(f"Registered {len(registry._registry)} built-in connectors")
