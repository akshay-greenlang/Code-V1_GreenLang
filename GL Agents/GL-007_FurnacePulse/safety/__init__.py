"""
GL-007_FurnacePulse Safety Module

This module provides comprehensive safety management capabilities for industrial
furnace operations, including NFPA86 compliance, LOPA/HAZOP integration,
safety alerting, and evidence packaging for regulatory audits.

Components:
    - NFPA86ComplianceManager: Manages NFPA86 compliance checklists and evidence
    - LOPAHAZOPIntegrator: Links telemetry to protection layers and HAZOP actions
    - SafetyAlertManager: Handles safety alert taxonomy and escalation
    - EvidencePackager: Creates immutable evidence packages for audits

Example:
    >>> from safety import NFPA86ComplianceManager, SafetyAlertManager
    >>> compliance_mgr = NFPA86ComplianceManager(config)
    >>> alert_mgr = SafetyAlertManager(config)
    >>>
    >>> # Check compliance status
    >>> status = compliance_mgr.get_compliance_status(furnace_id="FRN-001")
    >>>
    >>> # Generate alert
    >>> alert = alert_mgr.create_alert("A-001", furnace_id="FRN-001", details={...})

Author: GreenLang Backend Team
Version: 1.0.0
"""

from safety.nfpa86_compliance import NFPA86ComplianceManager
from safety.lopa_hazop_integration import LOPAHAZOPIntegrator
from safety.safety_alerts import SafetyAlertManager
from safety.evidence_packager import EvidencePackager

__all__ = [
    "NFPA86ComplianceManager",
    "LOPAHAZOPIntegrator",
    "SafetyAlertManager",
    "EvidencePackager",
]

__version__ = "1.0.0"
__author__ = "GreenLang Backend Team"
