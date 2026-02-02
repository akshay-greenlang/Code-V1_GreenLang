"""
GreenLang Compliance Module - Regulatory compliance checking and reporting.

This package provides compliance verification engines for various regulatory
frameworks including EPA NSPS standards, EU regulations (EUDR, CSRD, Taxonomy, IED),
and other environmental compliance requirements.

Subpackages:
    epa: EPA regulatory compliance (Part 60 NSPS, etc.)
    eu: EU regulatory compliance (EUDR, CSRD, Taxonomy, IED, etc.)

Example:
    >>> from greenlang.compliance.epa import NSPSComplianceChecker
    >>> checker = NSPSComplianceChecker()
    >>> result = checker.check_subpart_d(facility_data, emissions_data)

    >>> from greenlang.compliance.eu import IEDComplianceManager
    >>> manager = IEDComplianceManager(installation_id="INST-001")
    >>> result = manager.assess_compliance(emissions)
"""

from greenlang.compliance.eu import (
    IEDComplianceManager,
    IEDAnnexIActivity,
    ComplianceStatus,
    BATAEL,
    ComplianceAssessment,
)

__all__ = [
    "NSPSComplianceChecker",
    "IEDComplianceManager",
    "IEDAnnexIActivity",
    "ComplianceStatus",
    "BATAEL",
    "ComplianceAssessment",
]
