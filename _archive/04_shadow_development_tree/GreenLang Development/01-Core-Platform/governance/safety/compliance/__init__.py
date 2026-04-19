"""
Regulatory Compliance Framework - Multi-Standard Compliance Support

This module provides compliance verification and reporting for various
regulatory standards applicable to industrial safety systems.

Components:
- EPAReporter: EPA Part 60/75/98 reporting
- NFPA85Checker: NFPA 85 combustion safeguards
- NFPA86Checker: NFPA 86 furnace compliance
- OSHAPSM: OSHA 1910.119 PSM support
- EUIED: EU Industrial Emissions Directive compliance
- ISA182: ISA 18.2 alarm management

Reference: EPA 40 CFR, NFPA 85/86, OSHA 1910.119, EU IED, ISA 18.2

Example:
    >>> from greenlang.safety.compliance import EPAReporter, NFPA85Checker
    >>> reporter = EPAReporter()
    >>> checker = NFPA85Checker()
"""

from greenlang.safety.compliance.epa_reporter import (
    EPAReporter,
    EPAReport,
    EmissionRecord,
)
from greenlang.safety.compliance.nfpa_85_checker import (
    NFPA85Checker,
    NFPA85CheckResult,
    BurnerSafetyRequirement,
)
from greenlang.safety.compliance.nfpa_86_checker import (
    NFPA86Checker,
    NFPA86CheckResult,
    FurnaceClassification,
)
from greenlang.safety.compliance.osha_psm import (
    OSHAPSM,
    PSMElement,
    PSMAuditResult,
)
from greenlang.safety.compliance.eu_ied import (
    EUIED,
    IEDRequirement,
    BATAssessment,
)
from greenlang.safety.compliance.isa_18_2 import (
    ISA182,
    AlarmRationalization,
    AlarmMetrics,
)

__all__ = [
    # EPA
    "EPAReporter",
    "EPAReport",
    "EmissionRecord",
    # NFPA 85
    "NFPA85Checker",
    "NFPA85CheckResult",
    "BurnerSafetyRequirement",
    # NFPA 86
    "NFPA86Checker",
    "NFPA86CheckResult",
    "FurnaceClassification",
    # OSHA PSM
    "OSHAPSM",
    "PSMElement",
    "PSMAuditResult",
    # EU IED
    "EUIED",
    "IEDRequirement",
    "BATAssessment",
    # ISA 18.2
    "ISA182",
    "AlarmRationalization",
    "AlarmMetrics",
]

__version__ = "1.0.0"
