"""
GreenLang EU Compliance Module

This module provides compliance support for EU environmental regulations
applicable to industrial process heat operations.

Modules:
- ied_compliance: EU Industrial Emissions Directive (2010/75/EU)

Example:
    >>> from greenlang.compliance.eu import IEDComplianceManager
    >>> manager = IEDComplianceManager(installation_id="INST-001")
"""

from greenlang.compliance.eu.ied_compliance import (
    IEDComplianceManager,
    IEDAnnexIActivity,
    ComplianceStatus,
    MonitoringFrequency,
    PollutantCategory,
    BATConclusion,
    BATAEL,
    EmissionLimitValue,
    EmissionMeasurement,
    PermitCondition,
    DerogationRequest,
    ComplianceAssessment,
    AnnualReport,
)

__all__ = [
    "IEDComplianceManager",
    "IEDAnnexIActivity",
    "ComplianceStatus",
    "MonitoringFrequency",
    "PollutantCategory",
    "BATConclusion",
    "BATAEL",
    "EmissionLimitValue",
    "EmissionMeasurement",
    "PermitCondition",
    "DerogationRequest",
    "ComplianceAssessment",
    "AnnualReport",
]
