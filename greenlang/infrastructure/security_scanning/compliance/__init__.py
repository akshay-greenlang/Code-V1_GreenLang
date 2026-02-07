# -*- coding: utf-8 -*-
"""
Compliance Framework Module - SEC-007

Provides automated compliance checks and reporting for security scanning.
Maps security findings to regulatory framework controls including SOC 2,
ISO 27001, and GDPR.

Exports:
    - ComplianceFramework: Abstract base class for compliance frameworks
    - SOC2Compliance: SOC 2 Type II control mapping
    - ISO27001Compliance: ISO 27001 control mapping
    - GDPRCompliance: GDPR technical controls mapping
    - EvidenceCollector: Automated evidence collection for audits
    - ControlResult: Result of a compliance control check
    - ControlStatus: Status enumeration for control checks

Example:
    >>> from greenlang.infrastructure.security_scanning.compliance import (
    ...     SOC2Compliance,
    ...     EvidenceCollector,
    ... )
    >>> soc2 = SOC2Compliance()
    >>> results = await soc2.check_controls(scan_results)
    >>> report = soc2.generate_report(results)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from greenlang.infrastructure.security_scanning.compliance.base import (
    ComplianceFramework,
    ControlResult,
    ControlStatus,
    ComplianceReport,
    FrameworkType,
)
from greenlang.infrastructure.security_scanning.compliance.soc2_mapping import (
    SOC2Compliance,
)
from greenlang.infrastructure.security_scanning.compliance.iso27001_mapping import (
    ISO27001Compliance,
)
from greenlang.infrastructure.security_scanning.compliance.gdpr_mapping import (
    GDPRCompliance,
)
from greenlang.infrastructure.security_scanning.compliance.evidence_collector import (
    EvidenceCollector,
    EvidencePackage,
    EvidenceItem,
    EvidenceType,
)

__all__ = [
    # Base classes
    "ComplianceFramework",
    "ControlResult",
    "ControlStatus",
    "ComplianceReport",
    "FrameworkType",
    # Framework implementations
    "SOC2Compliance",
    "ISO27001Compliance",
    "GDPRCompliance",
    # Evidence collection
    "EvidenceCollector",
    "EvidencePackage",
    "EvidenceItem",
    "EvidenceType",
]

__version__ = "1.0.0"
