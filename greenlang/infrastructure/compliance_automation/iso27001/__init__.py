# -*- coding: utf-8 -*-
"""
ISO 27001:2022 Compliance Module - SEC-010 Phase 5

Implementation of ISO 27001:2022 Information Security Management System (ISMS)
compliance automation. Maps the 93 controls in Annex A to GreenLang technical
controls (SEC-001 through SEC-010), provides automated evidence collection,
and generates Statement of Applicability (SoA) reports.

Public API:
    - ISO27001Mapper: Control mapping and assessment.
    - ISO27001Evidence: Evidence collection for each control domain.
    - ISO27001Reporter: Compliance report generation.

Example:
    >>> from greenlang.infrastructure.compliance_automation.iso27001 import (
    ...     ISO27001Mapper, ISO27001Evidence, ISO27001Reporter,
    ... )
    >>> mapper = ISO27001Mapper()
    >>> controls = await mapper.get_controls()
    >>> soa = await mapper.generate_soa()
    >>> report = ISO27001Reporter().generate_report(soa)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from greenlang.infrastructure.compliance_automation.iso27001.mapper import (
    ISO27001Mapper,
)
from greenlang.infrastructure.compliance_automation.iso27001.evidence import (
    ISO27001Evidence,
)
from greenlang.infrastructure.compliance_automation.iso27001.reporter import (
    ISO27001Reporter,
)

__all__ = [
    "ISO27001Mapper",
    "ISO27001Evidence",
    "ISO27001Reporter",
]
