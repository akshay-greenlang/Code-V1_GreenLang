# -*- coding: utf-8 -*-
"""
Connector sources for licensed emission factor data (Phase 7).

Provides the base connector framework, individual connectors (IEA, ecoinvent,
Electricity Maps), and supporting infrastructure (license management, audit
logging, metrics, registry).
"""

from greenlang.factors.connectors.base import BaseConnector, ConnectorCapabilities, ConnectorStatus
from greenlang.factors.connectors.config import ConnectorConfig
from greenlang.factors.connectors.registry import ConnectorRegistry, get_global_registry
from greenlang.factors.connectors.license_manager import LicenseManager
from greenlang.factors.connectors.audit_log import ConnectorAuditLog
from greenlang.factors.connectors.iea import IEAConnector
from greenlang.factors.connectors.ecoinvent import EcoinventConnector
from greenlang.factors.connectors.electricity_maps import ElectricityMapsConnector

__all__ = [
    "BaseConnector",
    "ConnectorCapabilities",
    "ConnectorConfig",
    "ConnectorRegistry",
    "ConnectorStatus",
    "ConnectorAuditLog",
    "LicenseManager",
    "IEAConnector",
    "EcoinventConnector",
    "ElectricityMapsConnector",
    "get_global_registry",
]


def register_default_connectors() -> ConnectorRegistry:
    """Register all built-in connectors in the global registry."""
    registry = get_global_registry()
    registry.register_class("iea_statistics", IEAConnector)
    registry.register_class("ecoinvent", EcoinventConnector)
    registry.register_class("electricity_maps", ElectricityMapsConnector)
    return registry
