"""
GL-007_FurnacePulse Safety Module

This module provides comprehensive safety management capabilities for industrial
furnace operations, including NFPA86 compliance, LOPA/HAZOP integration,
safety alerting, evidence packaging, and circuit breaker protection for
external integrations.

Components:
    - NFPA86ComplianceManager: Manages NFPA86 compliance checklists and evidence
    - LOPAHAZOPIntegrator: Links telemetry to protection layers and HAZOP actions
    - SafetyAlertManager: Handles safety alert taxonomy and escalation
    - EvidencePackager: Creates immutable evidence packages for audits
    - CircuitBreaker: Protects against cascading failures in external integrations

Example:
    >>> from safety import NFPA86ComplianceManager, SafetyAlertManager, CircuitBreaker
    >>> compliance_mgr = NFPA86ComplianceManager(config)
    >>> alert_mgr = SafetyAlertManager(config)
    >>>
    >>> # Check compliance status
    >>> status = compliance_mgr.get_compliance_status(furnace_id="FRN-001")
    >>>
    >>> # Generate alert
    >>> alert = alert_mgr.create_alert("A-001", furnace_id="FRN-001", details={...})
    >>>
    >>> # Circuit breaker for OPC-UA integration
    >>> breaker = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=5))
    >>> result = await breaker.call(opcua_client.read_sensors, timeout=5.0)

Author: GreenLang Backend Team
Version: 1.1.0
"""

from safety.nfpa86_compliance import NFPA86ComplianceManager
from safety.lopa_hazop_integration import LOPAHAZOPIntegrator
from safety.safety_alerts import SafetyAlertManager
from safety.evidence_packager import EvidencePackager
from safety.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    CircuitBreakerRegistry,
)

__all__ = [
    "NFPA86ComplianceManager",
    "LOPAHAZOPIntegrator",
    "SafetyAlertManager",
    "EvidencePackager",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "FailureType",
    "RecoveryStrategy",
    "FailureRecord",
    "CircuitBreakerRegistry",
]

__version__ = "1.1.0"
__author__ = "GreenLang Backend Team"
