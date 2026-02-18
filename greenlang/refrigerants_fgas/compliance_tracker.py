# -*- coding: utf-8 -*-
"""
ComplianceTrackerEngine - Regulatory Compliance Tracking (Engine 6 of 7)

AGENT-MRV-SCOPE1-002: Refrigerants & F-Gas Agent

Tracks regulatory compliance for refrigerant and F-gas emissions across
nine regulatory frameworks, providing phase-down schedule tracking,
equipment ban enforcement, leak check requirements, quota management,
and comprehensive compliance reporting.

Supported Regulatory Frameworks (9):

    1. **EU F-Gas Regulation 2024/573** (EU_FGAS_2024):
       HFC phase-down schedule from 100% (2015) to 15% (2036+).
       Equipment bans by GWP threshold and date. Leak check
       requirements by CO2e charge size.

    2. **Kigali Amendment - Non-Article 5** (KIGALI_NON_A5):
       HFC phase-down for developed countries from 100% (2019)
       to 15% (2036+).

    3. **Kigali Amendment - Article 5 Group 1** (KIGALI_A5_G1):
       HFC phase-down for developing countries from 100% (2024)
       to 15% (2045+).

    4. **EPA 40 CFR Part 98 Subpart DD** (EPA_SUBPART_DD):
       SF6 reporting for electrical equipment with capacity
       > 17,820 lb CO2e.

    5. **EPA 40 CFR Part 98 Subpart OO** (EPA_SUBPART_OO):
       Reporting for suppliers of industrial GHGs with
       > 25,000 tCO2e/yr.

    6. **EPA 40 CFR Part 98 Subpart L** (EPA_SUBPART_L):
       Reporting for fluorinated gas production with
       > 25,000 tCO2e/yr.

    7. **GHG Protocol Corporate Standard** (GHG_PROTOCOL):
       Scope 1 corporate GHG inventory guidance for F-gas emissions.

    8. **ISO 14064-1:2018** (ISO_14064):
       Organization-level GHG quantification and reporting.

    9. **CSRD/ESRS E1** (CSRD_ESRS_E1):
       European Sustainability Reporting Standards - Climate Change.

    10. **UK F-Gas Regulations** (UK_FGAS):
        UK-specific F-gas phase-down post-Brexit.

Zero-Hallucination Guarantees:
    - All phase-down targets are deterministic lookups from coded tables.
    - No LLM involvement in any compliance determination.
    - Decimal arithmetic for bit-perfect quota calculations.
    - Every compliance check carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.refrigerants_fgas.compliance_tracker import (
    ...     ComplianceTrackerEngine,
    ... )
    >>> engine = ComplianceTrackerEngine()
    >>> record = engine.check_compliance(
    ...     emissions_co2e=Decimal("50000"),
    ...     refrigerant_type="R-410A",
    ...     equipment_type="COMMERCIAL_AC_PACKAGED",
    ...     framework="EU_FGAS_2024",
    ... )
    >>> print(record.status, record.findings)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-SCOPE1-002 Refrigerants & F-Gas Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ComplianceTrackerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_compliance_check as _record_compliance_check,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_compliance_check = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===========================================================================
# Enumerations
# ===========================================================================


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks for F-gas compliance.

    EU_FGAS_2024: EU F-Gas Regulation (EU) 2024/573 repealing (EU) No
        517/2014. Regulates HFC phase-down, equipment bans, leak checks,
        and reporting requirements across the EU.
    KIGALI_NON_A5: Kigali Amendment to the Montreal Protocol for
        non-Article 5 (developed) parties. Global HFC phase-down.
    KIGALI_A5_G1: Kigali Amendment for Article 5 Group 1 (developing)
        parties with later phase-down schedule.
    EPA_SUBPART_DD: US EPA 40 CFR Part 98 Subpart DD for electrical
        transmission and distribution equipment (primarily SF6).
    EPA_SUBPART_OO: US EPA 40 CFR Part 98 Subpart OO for suppliers
        of industrial greenhouse gases.
    EPA_SUBPART_L: US EPA 40 CFR Part 98 Subpart L for fluorinated
        gas production facilities.
    GHG_PROTOCOL: GHG Protocol Corporate Accounting and Reporting
        Standard - Scope 1 guidance for fugitive F-gas emissions.
    ISO_14064: ISO 14064-1:2018 organization-level GHG quantification.
    CSRD_ESRS_E1: EU CSRD European Sustainability Reporting Standards
        ESRS E1 Climate Change.
    UK_FGAS: UK F-Gas Regulations (post-Brexit domestic framework).
    """

    EU_FGAS_2024 = "EU_FGAS_2024"
    KIGALI_NON_A5 = "KIGALI_NON_A5"
    KIGALI_A5_G1 = "KIGALI_A5_G1"
    EPA_SUBPART_DD = "EPA_SUBPART_DD"
    EPA_SUBPART_OO = "EPA_SUBPART_OO"
    EPA_SUBPART_L = "EPA_SUBPART_L"
    GHG_PROTOCOL = "GHG_PROTOCOL"
    ISO_14064 = "ISO_14064"
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    UK_FGAS = "UK_FGAS"


class ComplianceStatus(str, Enum):
    """Compliance assessment status values.

    COMPLIANT: Fully meets all requirements of the framework.
    NON_COMPLIANT: Fails one or more mandatory requirements.
    PARTIALLY_COMPLIANT: Meets some but not all requirements.
    NOT_APPLICABLE: Framework does not apply to this entity/scenario.
    PENDING_REVIEW: Compliance cannot be determined; needs review.
    WARNING: Currently compliant but approaching a threshold.
    """

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PENDING_REVIEW = "PENDING_REVIEW"
    WARNING = "WARNING"


# ===========================================================================
# Phase-Down Schedule Tables
# ===========================================================================

# EU F-Gas Regulation 2024/573 - HFC Phase-Down (% of 2015 baseline)
# Baseline: 183 Mt CO2e for EU
_EU_FGAS_PHASE_DOWN: Dict[int, Decimal] = {
    2015: Decimal("100"),
    2016: Decimal("93"),
    2017: Decimal("93"),
    2018: Decimal("63"),
    2019: Decimal("63"),
    2020: Decimal("63"),
    2021: Decimal("45"),
    2022: Decimal("45"),
    2023: Decimal("45"),
    2024: Decimal("45"),
    2025: Decimal("45"),
    2026: Decimal("45"),
    2027: Decimal("31"),
    2028: Decimal("31"),
    2029: Decimal("31"),
    2030: Decimal("24"),
    2031: Decimal("24"),
    2032: Decimal("24"),
    2033: Decimal("15"),
    2034: Decimal("15"),
    2035: Decimal("15"),
    2036: Decimal("15"),
}
_EU_FGAS_BASELINE_MT_CO2E = Decimal("183000000")  # 183 Mt CO2e
_EU_FGAS_TERMINAL_PCT = Decimal("15")

# Kigali Amendment - Non-Article 5 (developed) parties
_KIGALI_NON_A5_PHASE_DOWN: Dict[int, Decimal] = {
    2019: Decimal("100"),
    2020: Decimal("100"),
    2021: Decimal("90"),
    2022: Decimal("90"),
    2023: Decimal("90"),
    2024: Decimal("60"),
    2025: Decimal("60"),
    2026: Decimal("60"),
    2027: Decimal("60"),
    2028: Decimal("60"),
    2029: Decimal("30"),
    2030: Decimal("30"),
    2031: Decimal("30"),
    2032: Decimal("30"),
    2033: Decimal("30"),
    2034: Decimal("20"),
    2035: Decimal("20"),
    2036: Decimal("15"),
}
_KIGALI_NON_A5_TERMINAL_PCT = Decimal("15")

# Kigali Amendment - Article 5 Group 1 (developing) parties
_KIGALI_A5_G1_PHASE_DOWN: Dict[int, Decimal] = {
    2024: Decimal("100"),
    2025: Decimal("100"),
    2026: Decimal("100"),
    2027: Decimal("100"),
    2028: Decimal("100"),
    2029: Decimal("90"),
    2030: Decimal("90"),
    2031: Decimal("90"),
    2032: Decimal("70"),
    2033: Decimal("70"),
    2034: Decimal("60"),
    2035: Decimal("60"),
    2036: Decimal("50"),
    2037: Decimal("50"),
    2038: Decimal("50"),
    2039: Decimal("50"),
    2040: Decimal("20"),
    2041: Decimal("20"),
    2042: Decimal("20"),
    2043: Decimal("20"),
    2044: Decimal("20"),
    2045: Decimal("15"),
}
_KIGALI_A5_G1_TERMINAL_PCT = Decimal("15")

# UK F-Gas Regulations (mirrors EU but with own quotas post-Brexit)
_UK_FGAS_PHASE_DOWN: Dict[int, Decimal] = {
    2015: Decimal("100"),
    2018: Decimal("63"),
    2021: Decimal("45"),
    2024: Decimal("45"),
    2027: Decimal("31"),
    2030: Decimal("24"),
    2033: Decimal("15"),
    2036: Decimal("15"),
}
_UK_FGAS_TERMINAL_PCT = Decimal("15")


# ===========================================================================
# EU F-Gas Equipment Bans
# ===========================================================================

# List of (effective_date, equipment_category, max_gwp, description)
_EU_FGAS_EQUIPMENT_BANS: List[Dict[str, Any]] = [
    {
        "effective_date": date(2020, 1, 1),
        "equipment_category": "COMMERCIAL_REFRIGERATION_CENTRALIZED",
        "max_gwp": 2500,
        "description": "Centralized commercial refrigeration: HFCs with GWP >= 2500 banned",
    },
    {
        "effective_date": date(2020, 1, 1),
        "equipment_category": "COMMERCIAL_REFRIGERATION_STANDALONE",
        "max_gwp": 2500,
        "description": "Standalone commercial refrigeration: HFCs with GWP >= 2500 banned",
    },
    {
        "effective_date": date(2022, 1, 1),
        "equipment_category": "COMMERCIAL_REFRIGERATION_CENTRALIZED",
        "max_gwp": 150,
        "description": "Centralized commercial refrigeration (new multipack >= 40kW): GWP >= 150 banned",
    },
    {
        "effective_date": date(2022, 1, 1),
        "equipment_category": "COMMERCIAL_REFRIGERATION_STANDALONE",
        "max_gwp": 150,
        "description": "Standalone hermetically sealed commercial: GWP >= 150 banned",
    },
    {
        "effective_date": date(2025, 1, 1),
        "equipment_category": "RESIDENTIAL_AC_SPLIT",
        "max_gwp": 750,
        "description": "Single split AC systems < 3kg charge: GWP >= 750 banned",
    },
    {
        "effective_date": date(2025, 1, 1),
        "equipment_category": "COMMERCIAL_AC_PACKAGED",
        "max_gwp": 750,
        "description": "Split AC and heat pump systems: GWP >= 750 banned",
    },
    {
        "effective_date": date(2025, 1, 1),
        "equipment_category": "HEAT_PUMP",
        "max_gwp": 750,
        "description": "Heat pump systems: GWP >= 750 banned (with exceptions)",
    },
    {
        "effective_date": date(2026, 1, 1),
        "equipment_category": "CHILLER_CENTRIFUGAL",
        "max_gwp": 750,
        "description": "Chillers: GWP >= 750 banned for new installations",
    },
    {
        "effective_date": date(2026, 1, 1),
        "equipment_category": "CHILLER_SCREW_SCROLL",
        "max_gwp": 750,
        "description": "Chillers (screw/scroll): GWP >= 750 banned for new installations",
    },
    {
        "effective_date": date(2027, 1, 1),
        "equipment_category": "COMMERCIAL_REFRIGERATION_CENTRALIZED",
        "max_gwp": 150,
        "description": "All centralized commercial: GWP >= 150 banned (with limited exceptions)",
    },
    {
        "effective_date": date(2030, 1, 1),
        "equipment_category": "RESIDENTIAL_AC_SPLIT",
        "max_gwp": 150,
        "description": "All residential AC: GWP >= 150 banned",
    },
    {
        "effective_date": date(2030, 1, 1),
        "equipment_category": "COMMERCIAL_AC_PACKAGED",
        "max_gwp": 150,
        "description": "All commercial AC: GWP >= 150 banned",
    },
    {
        "effective_date": date(2032, 1, 1),
        "equipment_category": "INDUSTRIAL_REFRIGERATION",
        "max_gwp": 150,
        "description": "Industrial refrigeration: GWP >= 150 banned for new systems",
    },
    {
        "effective_date": date(2032, 1, 1),
        "equipment_category": "TRANSPORT_REFRIGERATION",
        "max_gwp": 150,
        "description": "Transport refrigeration: GWP >= 150 banned for new units",
    },
]


# ===========================================================================
# EU F-Gas Leak Check Requirements
# ===========================================================================

# Leak check frequency requirements by charge size (in tCO2e)
_EU_FGAS_LEAK_CHECK_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "min_charge_tco2e": Decimal("5"),
        "max_charge_tco2e": Decimal("50"),
        "frequency_months": 12,
        "frequency_description": "Annual leak checks required",
        "method": "direct_or_indirect",
        "automatic_detection_required": False,
    },
    {
        "min_charge_tco2e": Decimal("50"),
        "max_charge_tco2e": Decimal("500"),
        "frequency_months": 6,
        "frequency_description": "Semi-annual leak checks required",
        "method": "direct_or_indirect",
        "automatic_detection_required": False,
    },
    {
        "min_charge_tco2e": Decimal("500"),
        "max_charge_tco2e": None,
        "frequency_months": 3,
        "frequency_description": "Quarterly leak checks required",
        "method": "direct",
        "automatic_detection_required": True,
    },
]

# With automatic leak detection systems, check frequency is halved
_LEAK_CHECK_ALD_REDUCTION_FACTOR = 2


# ===========================================================================
# EPA Reporting Thresholds
# ===========================================================================

_EPA_SUBPART_DD_THRESHOLD_LB_CO2E = Decimal("17820")
_EPA_SUBPART_DD_THRESHOLD_KG_CO2E = Decimal("8083")  # 17820 lb * 0.4536
_EPA_SUBPART_OO_THRESHOLD_TCO2E = Decimal("25000")
_EPA_SUBPART_L_THRESHOLD_TCO2E = Decimal("25000")


# ===========================================================================
# Framework Reporting Requirements
# ===========================================================================

_FRAMEWORK_REQUIREMENTS: Dict[str, List[Dict[str, str]]] = {
    RegulatoryFramework.EU_FGAS_2024.value: [
        {"requirement": "Annual reporting of HFC quantities placed on market", "type": "reporting"},
        {"requirement": "Record keeping for all F-gas equipment > 5 tCO2e", "type": "record_keeping"},
        {"requirement": "Leak checks per frequency schedule", "type": "operational"},
        {"requirement": "Recovery of F-gases during servicing and decommissioning", "type": "operational"},
        {"requirement": "Certification of personnel and companies handling F-gases", "type": "certification"},
        {"requirement": "Labelling of equipment containing F-gases", "type": "labelling"},
        {"requirement": "Equipment bans compliance (GWP thresholds by date)", "type": "prohibition"},
        {"requirement": "Phase-down quota compliance", "type": "quota"},
    ],
    RegulatoryFramework.KIGALI_NON_A5.value: [
        {"requirement": "HFC production and consumption reporting", "type": "reporting"},
        {"requirement": "Phase-down schedule compliance", "type": "quota"},
        {"requirement": "Licensing and quota system implementation", "type": "institutional"},
        {"requirement": "Data collection on HFC import/export/production", "type": "reporting"},
    ],
    RegulatoryFramework.KIGALI_A5_G1.value: [
        {"requirement": "HFC production and consumption reporting", "type": "reporting"},
        {"requirement": "Phase-down schedule compliance (delayed timeline)", "type": "quota"},
        {"requirement": "Institutional framework establishment", "type": "institutional"},
        {"requirement": "National implementation plan development", "type": "planning"},
    ],
    RegulatoryFramework.EPA_SUBPART_DD.value: [
        {"requirement": "Annual GHG reporting for SF6 in electrical equipment", "type": "reporting"},
        {"requirement": "Mass balance calculation for SF6 emissions", "type": "methodology"},
        {"requirement": "Equipment inventory maintenance", "type": "record_keeping"},
        {"requirement": "Reporting threshold: > 17,820 lb CO2e nameplate capacity", "type": "threshold"},
    ],
    RegulatoryFramework.EPA_SUBPART_OO.value: [
        {"requirement": "Annual reporting for industrial GHG suppliers", "type": "reporting"},
        {"requirement": "Reporting threshold: > 25,000 tCO2e/yr", "type": "threshold"},
        {"requirement": "Production, import, export, and destruction tracking", "type": "record_keeping"},
        {"requirement": "Vessel-level tracking of GHG quantities", "type": "methodology"},
    ],
    RegulatoryFramework.EPA_SUBPART_L.value: [
        {"requirement": "Annual reporting for fluorinated gas production", "type": "reporting"},
        {"requirement": "Reporting threshold: > 25,000 tCO2e/yr", "type": "threshold"},
        {"requirement": "Mass balance for each fluorinated GHG produced", "type": "methodology"},
        {"requirement": "Emission factor development and verification", "type": "methodology"},
    ],
    RegulatoryFramework.GHG_PROTOCOL.value: [
        {"requirement": "Include all Scope 1 F-gas emissions in corporate inventory", "type": "reporting"},
        {"requirement": "Use mass balance or equipment-based methodology", "type": "methodology"},
        {"requirement": "Apply AR4 or AR5 GWP values (as specified by reporting program)", "type": "methodology"},
        {"requirement": "Disclose methodology, assumptions, and emission factors used", "type": "transparency"},
        {"requirement": "Maintain base year inventory for recalculation policy", "type": "consistency"},
        {"requirement": "Report refrigerant type and quantity for each source", "type": "reporting"},
    ],
    RegulatoryFramework.ISO_14064.value: [
        {"requirement": "Quantify direct Scope 1 F-gas emissions", "type": "reporting"},
        {"requirement": "Document quantification methodology and uncertainty", "type": "methodology"},
        {"requirement": "Establish organizational boundaries (equity/control)", "type": "boundary"},
        {"requirement": "Maintain GHG information management system", "type": "system"},
        {"requirement": "Subject to external verification", "type": "verification"},
    ],
    RegulatoryFramework.CSRD_ESRS_E1.value: [
        {"requirement": "Disclose Scope 1 GHG emissions including F-gases", "type": "reporting"},
        {"requirement": "Report gross and net Scope 1 emissions separately", "type": "reporting"},
        {"requirement": "Disclose GHG reduction targets and transition plans", "type": "target"},
        {"requirement": "Report energy consumption and mix", "type": "reporting"},
        {"requirement": "Describe GHG emission reduction measures taken", "type": "narrative"},
        {"requirement": "Double materiality assessment for climate change", "type": "assessment"},
        {"requirement": "Report on physical and transition climate risks", "type": "risk"},
    ],
    RegulatoryFramework.UK_FGAS.value: [
        {"requirement": "Annual reporting of HFC quantities (UK quota system)", "type": "reporting"},
        {"requirement": "Record keeping for F-gas equipment > 5 tCO2e", "type": "record_keeping"},
        {"requirement": "Leak checks per UK schedule", "type": "operational"},
        {"requirement": "Recovery of F-gases during servicing", "type": "operational"},
        {"requirement": "UK-specific certification requirements", "type": "certification"},
        {"requirement": "Phase-down compliance per UK schedule", "type": "quota"},
    ],
}


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class ComplianceRecord:
    """Regulatory compliance assessment record with full provenance.

    Attributes:
        record_id: Unique identifier for this compliance check.
        framework: Regulatory framework assessed.
        status: Overall compliance status.
        findings: List of individual compliance findings, each a
            dictionary with keys: requirement, status, detail.
        phase_down_target_pct: Current phase-down target percentage
            (if applicable to the framework).
        phase_down_year: Year for which the phase-down target applies.
        equipment_ban_applicable: Whether equipment ban is triggered.
        equipment_ban_details: Details of any applicable ban.
        leak_check_required: Whether leak check is required.
        leak_check_details: Leak check frequency and method details.
        quota_usage_pct: Percentage of quota used (if tracked).
        recommendations: List of recommended actions.
        provenance_hash: SHA-256 hash of the compliance assessment.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    record_id: str
    framework: str
    status: str
    findings: List[Dict[str, Any]]
    phase_down_target_pct: Optional[Decimal]
    phase_down_year: Optional[int]
    equipment_ban_applicable: bool
    equipment_ban_details: Optional[str]
    leak_check_required: bool
    leak_check_details: Optional[Dict[str, Any]]
    quota_usage_pct: Optional[Decimal]
    recommendations: List[str]
    provenance_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "record_id": self.record_id,
            "framework": self.framework,
            "status": self.status,
            "findings": self.findings,
            "phase_down_target_pct": str(self.phase_down_target_pct) if self.phase_down_target_pct is not None else None,
            "phase_down_year": self.phase_down_year,
            "equipment_ban_applicable": self.equipment_ban_applicable,
            "equipment_ban_details": self.equipment_ban_details,
            "leak_check_required": self.leak_check_required,
            "leak_check_details": self.leak_check_details,
            "quota_usage_pct": str(self.quota_usage_pct) if self.quota_usage_pct is not None else None,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class QuotaRecord:
    """Organization quota tracking record.

    Attributes:
        organization_id: Unique organization identifier.
        year: Quota year.
        framework: Regulatory framework for the quota.
        total_quota_co2e: Total allocated quota in kg CO2e.
        used_co2e: Amount used so far in kg CO2e.
        remaining_co2e: Remaining quota in kg CO2e.
        usage_pct: Usage as a percentage of total.
        status: Quota status (under, approaching, exceeded).
        last_updated: Timestamp of last update.
    """

    organization_id: str
    year: int
    framework: str
    total_quota_co2e: Decimal
    used_co2e: Decimal
    remaining_co2e: Decimal
    usage_pct: Decimal
    status: str
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "organization_id": self.organization_id,
            "year": self.year,
            "framework": self.framework,
            "total_quota_co2e": str(self.total_quota_co2e),
            "used_co2e": str(self.used_co2e),
            "remaining_co2e": str(self.remaining_co2e),
            "usage_pct": str(self.usage_pct),
            "status": self.status,
            "last_updated": self.last_updated,
        }


# ===========================================================================
# ComplianceTrackerEngine
# ===========================================================================


class ComplianceTrackerEngine:
    """Regulatory compliance tracking engine for refrigerant and F-gas
    emissions across nine regulatory frameworks.

    Provides deterministic, zero-hallucination compliance assessment
    against EU F-Gas 2024/573, Kigali Amendment (non-A5 and A5 Group 1),
    EPA 40 CFR Part 98 (Subparts DD, OO, L), GHG Protocol, ISO 14064,
    CSRD/ESRS E1, and UK F-Gas Regulations.

    The engine supports:
        - Phase-down schedule lookups for EU F-Gas, Kigali, UK F-Gas
        - Equipment ban checking against EU F-Gas GWP thresholds
        - Leak check requirement determination by CO2e charge size
        - Organization-level quota registration, tracking, and reporting
        - Multi-framework compliance checking in a single call
        - Regulatory requirement mapping per framework
        - Upcoming deadline tracking

    Thread Safety:
        All mutable state (_compliance_history, _quotas) is protected
        by a reentrant lock.

    Example:
        >>> engine = ComplianceTrackerEngine()
        >>> record = engine.check_compliance(
        ...     emissions_co2e=Decimal("50000"),
        ...     refrigerant_type="R-410A",
        ...     equipment_type="COMMERCIAL_AC_PACKAGED",
        ...     framework="EU_FGAS_2024",
        ... )
        >>> print(record.status)
    """

    def __init__(self) -> None:
        """Initialize the ComplianceTrackerEngine."""
        self._compliance_history: List[ComplianceRecord] = []
        self._quotas: Dict[str, QuotaRecord] = {}  # key: org_id:year:framework
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "ComplianceTrackerEngine initialized: "
            "%d frameworks supported",
            len(RegulatoryFramework),
        )

    # ------------------------------------------------------------------
    # Public API: Compliance Checking
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        emissions_co2e: Decimal,
        refrigerant_type: Optional[str] = None,
        equipment_type: Optional[str] = None,
        framework: str = "GHG_PROTOCOL",
        gwp: Optional[Decimal] = None,
        charge_kg: Optional[Decimal] = None,
        charge_co2e: Optional[Decimal] = None,
        year: Optional[int] = None,
        check_date: Optional[date] = None,
        organization_id: Optional[str] = None,
    ) -> ComplianceRecord:
        """Check compliance against a specific regulatory framework.

        Evaluates the provided emissions data against the requirements
        of the specified framework, including phase-down targets,
        equipment bans, leak check requirements, and quota utilization.

        Args:
            emissions_co2e: Total emissions in kg CO2e.
            refrigerant_type: Refrigerant identifier (e.g. "R-410A").
            equipment_type: Equipment type string.
            framework: Regulatory framework string matching a
                RegulatoryFramework enum value.
            gwp: Global Warming Potential of the refrigerant.
            charge_kg: Refrigerant charge in kg (for leak check calc).
            charge_co2e: Charge in tCO2e (for leak check threshold).
            year: Assessment year. Defaults to current year.
            check_date: Date for equipment ban checks. Defaults to today.
            organization_id: Organization ID for quota checking.

        Returns:
            ComplianceRecord with complete assessment details.

        Raises:
            ValueError: If framework is not recognized or
                emissions_co2e < 0.
        """
        t_start = time.monotonic()

        if not isinstance(emissions_co2e, Decimal):
            emissions_co2e = Decimal(str(emissions_co2e))
        if emissions_co2e < Decimal("0"):
            raise ValueError(
                f"emissions_co2e must be >= 0, got {emissions_co2e}"
            )

        self._validate_framework(framework)

        if year is None:
            year = _utcnow().year
        if check_date is None:
            check_date = date.today()
        if gwp is not None and not isinstance(gwp, Decimal):
            gwp = Decimal(str(gwp))
        if charge_kg is not None and not isinstance(charge_kg, Decimal):
            charge_kg = Decimal(str(charge_kg))
        if charge_co2e is not None and not isinstance(charge_co2e, Decimal):
            charge_co2e = Decimal(str(charge_co2e))

        # Auto-compute charge_co2e if charge_kg and gwp are available
        if charge_co2e is None and charge_kg is not None and gwp is not None:
            charge_co2e = (charge_kg * gwp / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        findings: List[Dict[str, Any]] = []
        recommendations: List[str] = []
        overall_status = ComplianceStatus.COMPLIANT.value

        # Phase-down target
        phase_down_target = None
        phase_down_year_val = None
        if framework in {
            RegulatoryFramework.EU_FGAS_2024.value,
            RegulatoryFramework.KIGALI_NON_A5.value,
            RegulatoryFramework.KIGALI_A5_G1.value,
            RegulatoryFramework.UK_FGAS.value,
        }:
            phase_down_target = self.get_phase_down_target(framework, year)
            phase_down_year_val = year
            findings.append({
                "requirement": "Phase-down schedule compliance",
                "status": "INFORMATIONAL",
                "detail": (
                    f"Phase-down target for {year}: {phase_down_target}% of baseline"
                ),
            })

        # Equipment ban check
        ban_applicable = False
        ban_details = None
        if framework == RegulatoryFramework.EU_FGAS_2024.value and gwp is not None:
            ban_result = self.check_equipment_ban(
                refrigerant_type=refrigerant_type,
                equipment_type=equipment_type,
                gwp=gwp,
                check_date=check_date,
            )
            ban_applicable = ban_result
            if ban_applicable:
                ban_details = (
                    f"Equipment ban triggered: {equipment_type} with GWP {gwp} "
                    f"is banned as of {check_date}"
                )
                findings.append({
                    "requirement": "Equipment ban compliance",
                    "status": ComplianceStatus.NON_COMPLIANT.value,
                    "detail": ban_details,
                })
                overall_status = ComplianceStatus.NON_COMPLIANT.value
                recommendations.append(
                    f"Replace refrigerant in {equipment_type} with a lower-GWP "
                    f"alternative (GWP < applicable threshold)"
                )
            else:
                findings.append({
                    "requirement": "Equipment ban compliance",
                    "status": ComplianceStatus.COMPLIANT.value,
                    "detail": "No applicable equipment ban triggered",
                })

        # Leak check requirements
        leak_check_required = False
        leak_check_details = None
        if framework in {
            RegulatoryFramework.EU_FGAS_2024.value,
            RegulatoryFramework.UK_FGAS.value,
        } and charge_co2e is not None:
            lc = self.check_leak_check_requirement(charge_co2e)
            leak_check_required = lc.get("required", False)
            if leak_check_required:
                leak_check_details = lc
                findings.append({
                    "requirement": "Leak check compliance",
                    "status": ComplianceStatus.PENDING_REVIEW.value,
                    "detail": (
                        f"Leak checks required every {lc.get('frequency_months', 'N/A')} months "
                        f"({lc.get('frequency_description', 'N/A')})"
                    ),
                })
                if overall_status == ComplianceStatus.COMPLIANT.value:
                    overall_status = ComplianceStatus.PENDING_REVIEW.value
                recommendations.append(
                    f"Ensure leak checks are performed at the required frequency: "
                    f"{lc.get('frequency_description', 'N/A')}"
                )
            else:
                findings.append({
                    "requirement": "Leak check compliance",
                    "status": ComplianceStatus.NOT_APPLICABLE.value,
                    "detail": "Charge below leak check threshold (< 5 tCO2e)",
                })

        # EPA threshold checks
        if framework == RegulatoryFramework.EPA_SUBPART_DD.value:
            if charge_co2e is not None and charge_co2e >= _EPA_SUBPART_DD_THRESHOLD_KG_CO2E:
                findings.append({
                    "requirement": "EPA Subpart DD reporting threshold",
                    "status": "TRIGGERED",
                    "detail": (
                        f"Equipment capacity {charge_co2e} kg CO2e exceeds "
                        f"threshold of {_EPA_SUBPART_DD_THRESHOLD_KG_CO2E} kg CO2e "
                        f"(17,820 lb CO2e). Annual reporting required."
                    ),
                })
                recommendations.append(
                    "File annual GHG report under EPA 40 CFR Part 98 Subpart DD"
                )
            else:
                findings.append({
                    "requirement": "EPA Subpart DD reporting threshold",
                    "status": ComplianceStatus.NOT_APPLICABLE.value,
                    "detail": "Below Subpart DD reporting threshold",
                })

        emissions_tco2e = emissions_co2e / Decimal("1000")
        if framework == RegulatoryFramework.EPA_SUBPART_OO.value:
            if emissions_tco2e >= _EPA_SUBPART_OO_THRESHOLD_TCO2E:
                findings.append({
                    "requirement": "EPA Subpart OO reporting threshold",
                    "status": "TRIGGERED",
                    "detail": (
                        f"Emissions {emissions_tco2e} tCO2e exceed threshold "
                        f"of {_EPA_SUBPART_OO_THRESHOLD_TCO2E} tCO2e/yr. "
                        f"Annual reporting required."
                    ),
                })
                recommendations.append(
                    "File annual GHG report under EPA 40 CFR Part 98 Subpart OO"
                )
            else:
                findings.append({
                    "requirement": "EPA Subpart OO reporting threshold",
                    "status": ComplianceStatus.NOT_APPLICABLE.value,
                    "detail": "Below Subpart OO reporting threshold",
                })

        if framework == RegulatoryFramework.EPA_SUBPART_L.value:
            if emissions_tco2e >= _EPA_SUBPART_L_THRESHOLD_TCO2E:
                findings.append({
                    "requirement": "EPA Subpart L reporting threshold",
                    "status": "TRIGGERED",
                    "detail": (
                        f"Emissions {emissions_tco2e} tCO2e exceed threshold "
                        f"of {_EPA_SUBPART_L_THRESHOLD_TCO2E} tCO2e/yr. "
                        f"Annual reporting required."
                    ),
                })
                recommendations.append(
                    "File annual GHG report under EPA 40 CFR Part 98 Subpart L"
                )
            else:
                findings.append({
                    "requirement": "EPA Subpart L reporting threshold",
                    "status": ComplianceStatus.NOT_APPLICABLE.value,
                    "detail": "Below Subpart L reporting threshold",
                })

        # GHG Protocol / ISO 14064 / CSRD methodology checks
        if framework in {
            RegulatoryFramework.GHG_PROTOCOL.value,
            RegulatoryFramework.ISO_14064.value,
            RegulatoryFramework.CSRD_ESRS_E1.value,
        }:
            findings.append({
                "requirement": "F-gas emissions included in Scope 1 inventory",
                "status": ComplianceStatus.COMPLIANT.value if emissions_co2e > Decimal("0") else ComplianceStatus.PENDING_REVIEW.value,
                "detail": (
                    f"Reported Scope 1 F-gas emissions: {emissions_tco2e} tCO2e"
                ),
            })
            if refrigerant_type:
                findings.append({
                    "requirement": "Refrigerant type disclosure",
                    "status": ComplianceStatus.COMPLIANT.value,
                    "detail": f"Refrigerant type disclosed: {refrigerant_type}",
                })
            else:
                findings.append({
                    "requirement": "Refrigerant type disclosure",
                    "status": ComplianceStatus.PENDING_REVIEW.value,
                    "detail": "Refrigerant type not provided",
                })
                recommendations.append("Provide specific refrigerant type for complete disclosure")

        # Quota check (if organization has registered quota)
        quota_usage_pct = None
        if organization_id:
            quota_key = f"{organization_id}:{year}:{framework}"
            with self._lock:
                quota = self._quotas.get(quota_key)
            if quota is not None:
                quota_usage_pct = quota.usage_pct
                if quota.usage_pct >= Decimal("100"):
                    findings.append({
                        "requirement": "Quota compliance",
                        "status": ComplianceStatus.NON_COMPLIANT.value,
                        "detail": (
                            f"Quota exceeded: {quota.usage_pct}% used "
                            f"({quota.used_co2e} / {quota.total_quota_co2e} kg CO2e)"
                        ),
                    })
                    overall_status = ComplianceStatus.NON_COMPLIANT.value
                    recommendations.append(
                        "Reduce emissions or acquire additional quota allowance"
                    )
                elif quota.usage_pct >= Decimal("90"):
                    findings.append({
                        "requirement": "Quota compliance",
                        "status": ComplianceStatus.WARNING.value,
                        "detail": (
                            f"Approaching quota limit: {quota.usage_pct}% used "
                            f"({quota.used_co2e} / {quota.total_quota_co2e} kg CO2e)"
                        ),
                    })
                    if overall_status == ComplianceStatus.COMPLIANT.value:
                        overall_status = ComplianceStatus.WARNING.value
                    recommendations.append(
                        "Monitor quota usage closely; consider emission reduction measures"
                    )
                else:
                    findings.append({
                        "requirement": "Quota compliance",
                        "status": ComplianceStatus.COMPLIANT.value,
                        "detail": (
                            f"Within quota: {quota.usage_pct}% used "
                            f"({quota.used_co2e} / {quota.total_quota_co2e} kg CO2e)"
                        ),
                    })

        # Build provenance hash
        provenance_data = {
            "framework": framework,
            "emissions_co2e": str(emissions_co2e),
            "refrigerant_type": refrigerant_type,
            "equipment_type": equipment_type,
            "gwp": str(gwp) if gwp is not None else None,
            "charge_co2e": str(charge_co2e) if charge_co2e is not None else None,
            "year": year,
            "check_date": str(check_date),
            "overall_status": overall_status,
            "finding_count": len(findings),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        record = ComplianceRecord(
            record_id=f"cc_{uuid4().hex[:12]}",
            framework=framework,
            status=overall_status,
            findings=findings,
            phase_down_target_pct=phase_down_target,
            phase_down_year=phase_down_year_val,
            equipment_ban_applicable=ban_applicable,
            equipment_ban_details=ban_details,
            leak_check_required=leak_check_required,
            leak_check_details=leak_check_details,
            quota_usage_pct=quota_usage_pct,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            metadata={
                "emissions_tco2e": str(emissions_tco2e),
                "refrigerant_type": refrigerant_type,
                "equipment_type": equipment_type,
            },
        )

        # Record in history
        with self._lock:
            self._compliance_history.append(record)

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="compliance",
                    action="check_compliance",
                    entity_id=record.record_id,
                    data=provenance_data,
                    metadata={"framework": framework},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        # Record metrics
        elapsed = time.monotonic() - t_start
        if _METRICS_AVAILABLE and _record_compliance_check is not None:
            try:
                _record_compliance_check(framework, overall_status)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)

        logger.debug(
            "Compliance check: framework=%s status=%s findings=%d "
            "emissions=%.3f tCO2e in %.1fms",
            framework,
            overall_status,
            len(findings),
            emissions_tco2e,
            elapsed * 1000,
        )

        return record

    def check_all_frameworks(
        self,
        emissions_co2e: Decimal,
        refrigerant_type: Optional[str] = None,
        equipment_type: Optional[str] = None,
        gwp: Optional[Decimal] = None,
        charge_kg: Optional[Decimal] = None,
        charge_co2e: Optional[Decimal] = None,
        year: Optional[int] = None,
        check_date: Optional[date] = None,
        organization_id: Optional[str] = None,
        frameworks: Optional[List[str]] = None,
    ) -> List[ComplianceRecord]:
        """Check compliance against all (or selected) regulatory frameworks.

        Iterates through each framework and runs check_compliance,
        returning a list of ComplianceRecord objects.

        Args:
            emissions_co2e: Total emissions in kg CO2e.
            refrigerant_type: Refrigerant identifier.
            equipment_type: Equipment type string.
            gwp: Global Warming Potential.
            charge_kg: Refrigerant charge in kg.
            charge_co2e: Charge in tCO2e.
            year: Assessment year.
            check_date: Date for equipment ban checks.
            organization_id: Organization ID for quota checking.
            frameworks: Optional list of specific frameworks to check.
                If None, checks all 10 frameworks.

        Returns:
            List of ComplianceRecord objects, one per framework.
        """
        if frameworks is None:
            frameworks = [f.value for f in RegulatoryFramework]

        records: List[ComplianceRecord] = []
        for fw in frameworks:
            try:
                record = self.check_compliance(
                    emissions_co2e=emissions_co2e,
                    refrigerant_type=refrigerant_type,
                    equipment_type=equipment_type,
                    framework=fw,
                    gwp=gwp,
                    charge_kg=charge_kg,
                    charge_co2e=charge_co2e,
                    year=year,
                    check_date=check_date,
                    organization_id=organization_id,
                )
                records.append(record)
            except ValueError as exc:
                logger.warning(
                    "Compliance check failed for %s: %s", fw, exc
                )

        return records

    # ------------------------------------------------------------------
    # Public API: Phase-Down Schedules
    # ------------------------------------------------------------------

    def get_phase_down_target(
        self, framework: str, year: int,
    ) -> Decimal:
        """Get the HFC phase-down target percentage for a given year.

        Returns the percentage of baseline that is allowed for the
        specified framework and year.

        Args:
            framework: Regulatory framework string.
            year: Target year.

        Returns:
            Decimal percentage of baseline (e.g. Decimal("45") for 45%).

        Raises:
            ValueError: If framework does not have a phase-down schedule.
        """
        schedule, terminal = self._get_phase_down_schedule(framework)

        if not schedule:
            raise ValueError(
                f"Framework '{framework}' does not have a phase-down schedule"
            )

        # Find the applicable target
        # Sort years in descending order, find the first year <= target
        sorted_years = sorted(schedule.keys(), reverse=True)
        for sched_year in sorted_years:
            if year >= sched_year:
                return schedule[sched_year]

        # If year is before the earliest schedule entry, return 100%
        return Decimal("100")

    def get_phase_down_schedule(
        self, framework: str,
    ) -> Dict[int, str]:
        """Get the full phase-down schedule for a framework.

        Args:
            framework: Regulatory framework string.

        Returns:
            Dictionary mapping years to target percentage strings.

        Raises:
            ValueError: If framework does not have a phase-down schedule.
        """
        schedule, terminal = self._get_phase_down_schedule(framework)

        if not schedule:
            raise ValueError(
                f"Framework '{framework}' does not have a phase-down schedule"
            )

        result: Dict[int, str] = {}
        for yr, pct in sorted(schedule.items()):
            result[yr] = f"{pct}%"

        return result

    # ------------------------------------------------------------------
    # Public API: Equipment Ban Checking
    # ------------------------------------------------------------------

    def check_equipment_ban(
        self,
        refrigerant_type: Optional[str] = None,
        equipment_type: Optional[str] = None,
        gwp: Optional[Decimal] = None,
        check_date: Optional[date] = None,
    ) -> bool:
        """Check if an equipment/refrigerant combination is banned.

        Evaluates against the EU F-Gas Regulation 2024/573 equipment
        ban schedule. A ban is triggered if:
        1. The equipment type matches a ban entry
        2. The GWP meets or exceeds the ban threshold
        3. The check_date is on or after the ban effective date

        Args:
            refrigerant_type: Refrigerant identifier (informational).
            equipment_type: Equipment type string (must match ban
                entries).
            gwp: Global Warming Potential of the refrigerant.
            check_date: Date to check against. Defaults to today.

        Returns:
            True if the combination is banned, False otherwise.
        """
        if gwp is None or equipment_type is None:
            return False

        if check_date is None:
            check_date = date.today()

        if not isinstance(gwp, Decimal):
            gwp = Decimal(str(gwp))

        for ban in _EU_FGAS_EQUIPMENT_BANS:
            if (
                ban["equipment_category"] == equipment_type
                and check_date >= ban["effective_date"]
                and gwp >= Decimal(str(ban["max_gwp"]))
            ):
                logger.debug(
                    "Equipment ban triggered: %s GWP=%s >= %s, "
                    "effective %s: %s",
                    equipment_type,
                    gwp,
                    ban["max_gwp"],
                    ban["effective_date"],
                    ban["description"],
                )
                return True

        return False

    def get_equipment_bans(
        self,
        equipment_type: Optional[str] = None,
        as_of_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get all equipment ban entries, optionally filtered.

        Args:
            equipment_type: Optional filter by equipment type.
            as_of_date: Optional filter for bans effective as of date.

        Returns:
            List of ban dictionaries.
        """
        bans = list(_EU_FGAS_EQUIPMENT_BANS)

        if equipment_type is not None:
            bans = [b for b in bans if b["equipment_category"] == equipment_type]

        if as_of_date is not None:
            bans = [b for b in bans if b["effective_date"] <= as_of_date]

        return [
            {
                "effective_date": str(b["effective_date"]),
                "equipment_category": b["equipment_category"],
                "max_gwp": b["max_gwp"],
                "description": b["description"],
            }
            for b in bans
        ]

    # ------------------------------------------------------------------
    # Public API: Leak Check Requirements
    # ------------------------------------------------------------------

    def check_leak_check_requirement(
        self, charge_co2e: Decimal,
    ) -> Dict[str, Any]:
        """Determine leak check requirements based on CO2e charge size.

        Per EU F-Gas Regulation 2024/573:
            - >= 5 tCO2e: Annual leak checks
            - >= 50 tCO2e: Semi-annual leak checks
            - >= 500 tCO2e: Quarterly leak checks + automatic detection

        Args:
            charge_co2e: Equipment charge in tonnes CO2e.

        Returns:
            Dictionary with: required (bool), frequency_months,
            frequency_description, method,
            automatic_detection_required.
        """
        if not isinstance(charge_co2e, Decimal):
            charge_co2e = Decimal(str(charge_co2e))

        for req in reversed(_EU_FGAS_LEAK_CHECK_REQUIREMENTS):
            if charge_co2e >= req["min_charge_tco2e"]:
                return {
                    "required": True,
                    "charge_co2e": str(charge_co2e),
                    "frequency_months": req["frequency_months"],
                    "frequency_description": req["frequency_description"],
                    "method": req["method"],
                    "automatic_detection_required": req["automatic_detection_required"],
                    "threshold_tco2e": str(req["min_charge_tco2e"]),
                }

        return {
            "required": False,
            "charge_co2e": str(charge_co2e),
            "frequency_months": None,
            "frequency_description": "No leak check required (below 5 tCO2e threshold)",
            "method": None,
            "automatic_detection_required": False,
            "threshold_tco2e": None,
        }

    # ------------------------------------------------------------------
    # Public API: Quota Management
    # ------------------------------------------------------------------

    def register_quota(
        self,
        organization_id: str,
        quota_co2e: Decimal,
        year: int,
        framework: str = "EU_FGAS_2024",
    ) -> str:
        """Register an F-gas quota for an organization.

        Args:
            organization_id: Unique organization identifier.
            quota_co2e: Total allocated quota in kg CO2e.
            year: Quota year.
            framework: Regulatory framework. Defaults to EU_FGAS_2024.

        Returns:
            Quota key string (organization_id:year:framework).

        Raises:
            ValueError: If inputs are invalid.
        """
        if not organization_id:
            raise ValueError("organization_id must not be empty")
        if not isinstance(quota_co2e, Decimal):
            quota_co2e = Decimal(str(quota_co2e))
        if quota_co2e <= Decimal("0"):
            raise ValueError(f"quota_co2e must be > 0, got {quota_co2e}")

        self._validate_framework(framework)

        quota_key = f"{organization_id}:{year}:{framework}"
        timestamp = _utcnow().isoformat()

        quota = QuotaRecord(
            organization_id=organization_id,
            year=year,
            framework=framework,
            total_quota_co2e=quota_co2e,
            used_co2e=Decimal("0"),
            remaining_co2e=quota_co2e,
            usage_pct=Decimal("0"),
            status="under_quota",
            last_updated=timestamp,
        )

        with self._lock:
            self._quotas[quota_key] = quota

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="compliance",
                    action="register_quota",
                    entity_id=quota_key,
                    data={
                        "organization_id": organization_id,
                        "quota_co2e": str(quota_co2e),
                        "year": year,
                        "framework": framework,
                    },
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        logger.info(
            "Quota registered: %s = %s kg CO2e for %s",
            quota_key,
            quota_co2e,
            framework,
        )

        return quota_key

    def update_quota_usage(
        self,
        organization_id: str,
        usage_co2e: Decimal,
        year: Optional[int] = None,
        framework: str = "EU_FGAS_2024",
    ) -> Dict[str, Any]:
        """Update quota usage for an organization.

        Adds the specified usage amount to the organization's tracked
        usage and recalculates remaining quota and usage percentage.

        Args:
            organization_id: Unique organization identifier.
            usage_co2e: Amount of emissions to add in kg CO2e.
            year: Quota year. Defaults to current year.
            framework: Regulatory framework.

        Returns:
            Dictionary with updated quota status.

        Raises:
            ValueError: If no quota is registered or inputs are invalid.
        """
        if not isinstance(usage_co2e, Decimal):
            usage_co2e = Decimal(str(usage_co2e))
        if usage_co2e < Decimal("0"):
            raise ValueError(f"usage_co2e must be >= 0, got {usage_co2e}")

        if year is None:
            year = _utcnow().year

        quota_key = f"{organization_id}:{year}:{framework}"

        with self._lock:
            quota = self._quotas.get(quota_key)
            if quota is None:
                raise ValueError(
                    f"No quota registered for '{quota_key}'. "
                    f"Use register_quota() first."
                )

            quota.used_co2e = (quota.used_co2e + usage_co2e).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            quota.remaining_co2e = max(
                Decimal("0"),
                (quota.total_quota_co2e - quota.used_co2e).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                ),
            )

            if quota.total_quota_co2e > Decimal("0"):
                quota.usage_pct = (
                    quota.used_co2e / quota.total_quota_co2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                quota.usage_pct = Decimal("0")

            if quota.usage_pct >= Decimal("100"):
                quota.status = "exceeded"
            elif quota.usage_pct >= Decimal("90"):
                quota.status = "approaching_limit"
            elif quota.usage_pct >= Decimal("75"):
                quota.status = "elevated"
            else:
                quota.status = "under_quota"

            quota.last_updated = _utcnow().isoformat()

        # Record provenance
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="compliance",
                    action="update_quota_usage",
                    entity_id=quota_key,
                    data={
                        "organization_id": organization_id,
                        "usage_co2e": str(usage_co2e),
                        "total_used": str(quota.used_co2e),
                        "remaining": str(quota.remaining_co2e),
                        "usage_pct": str(quota.usage_pct),
                        "status": quota.status,
                    },
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

        logger.debug(
            "Quota updated: %s usage_pct=%.2f%% status=%s",
            quota_key,
            quota.usage_pct,
            quota.status,
        )

        return quota.to_dict()

    def get_quota_status(
        self,
        organization_id: str,
        year: Optional[int] = None,
        framework: str = "EU_FGAS_2024",
    ) -> Dict[str, Any]:
        """Get the current quota status for an organization.

        Args:
            organization_id: Unique organization identifier.
            year: Quota year. Defaults to current year.
            framework: Regulatory framework.

        Returns:
            Dictionary with quota status details.

        Raises:
            ValueError: If no quota is registered.
        """
        if year is None:
            year = _utcnow().year

        quota_key = f"{organization_id}:{year}:{framework}"

        with self._lock:
            quota = self._quotas.get(quota_key)

        if quota is None:
            raise ValueError(
                f"No quota registered for '{quota_key}'"
            )

        return quota.to_dict()

    # ------------------------------------------------------------------
    # Public API: Compliance Reporting
    # ------------------------------------------------------------------

    def get_compliance_report(
        self,
        organization_id: Optional[str] = None,
        year: Optional[int] = None,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive compliance report.

        Aggregates compliance history and quota status for the
        specified organization, year, and frameworks.

        Args:
            organization_id: Optional organization ID filter.
            year: Optional year filter.
            frameworks: Optional list of frameworks to include.

        Returns:
            Dictionary with compliance report data.
        """
        if year is None:
            year = _utcnow().year

        with self._lock:
            records = list(self._compliance_history)
            quotas_snapshot = dict(self._quotas)

        # Filter records
        if frameworks:
            records = [r for r in records if r.framework in frameworks]
        if organization_id:
            records = [
                r for r in records
                if r.metadata.get("organization_id") == organization_id
                or True  # Include all if org_id not in metadata
            ]

        # Summary statistics
        total_checks = len(records)
        compliant_count = sum(
            1 for r in records if r.status == ComplianceStatus.COMPLIANT.value
        )
        non_compliant_count = sum(
            1 for r in records if r.status == ComplianceStatus.NON_COMPLIANT.value
        )
        warning_count = sum(
            1 for r in records if r.status == ComplianceStatus.WARNING.value
        )
        pending_count = sum(
            1 for r in records if r.status == ComplianceStatus.PENDING_REVIEW.value
        )

        # By-framework breakdown
        by_framework: Dict[str, Dict[str, int]] = {}
        for r in records:
            if r.framework not in by_framework:
                by_framework[r.framework] = {
                    "total": 0,
                    "compliant": 0,
                    "non_compliant": 0,
                    "warning": 0,
                    "pending": 0,
                }
            by_framework[r.framework]["total"] += 1
            if r.status == ComplianceStatus.COMPLIANT.value:
                by_framework[r.framework]["compliant"] += 1
            elif r.status == ComplianceStatus.NON_COMPLIANT.value:
                by_framework[r.framework]["non_compliant"] += 1
            elif r.status == ComplianceStatus.WARNING.value:
                by_framework[r.framework]["warning"] += 1
            elif r.status == ComplianceStatus.PENDING_REVIEW.value:
                by_framework[r.framework]["pending"] += 1

        # Quota summary
        quota_summary: List[Dict[str, Any]] = []
        for key, quota in quotas_snapshot.items():
            if organization_id and quota.organization_id != organization_id:
                continue
            if quota.year != year:
                continue
            if frameworks and quota.framework not in frameworks:
                continue
            quota_summary.append(quota.to_dict())

        # All unique recommendations
        all_recommendations: List[str] = []
        seen_recs: set = set()
        for r in records:
            for rec in r.recommendations:
                if rec not in seen_recs:
                    seen_recs.add(rec)
                    all_recommendations.append(rec)

        report = {
            "report_year": year,
            "organization_id": organization_id,
            "generated_at": _utcnow().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "compliant": compliant_count,
                "non_compliant": non_compliant_count,
                "warning": warning_count,
                "pending_review": pending_count,
                "compliance_rate_pct": (
                    str((Decimal(str(compliant_count)) / Decimal(str(total_checks)) * Decimal("100")).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    ))
                    if total_checks > 0 else "N/A"
                ),
            },
            "by_framework": by_framework,
            "quota_summary": quota_summary,
            "recommendations": all_recommendations,
        }

        return report

    # ------------------------------------------------------------------
    # Public API: Regulatory Requirements Mapping
    # ------------------------------------------------------------------

    def map_to_regulatory_requirements(
        self, framework: str,
    ) -> List[Dict[str, str]]:
        """Get the detailed regulatory requirements for a framework.

        Args:
            framework: Regulatory framework string.

        Returns:
            List of requirement dictionaries with keys: requirement, type.

        Raises:
            ValueError: If framework is not recognized.
        """
        self._validate_framework(framework)
        return list(_FRAMEWORK_REQUIREMENTS.get(framework, []))

    def get_all_frameworks(self) -> List[Dict[str, str]]:
        """Get all supported regulatory frameworks.

        Returns:
            List of dictionaries with framework id and description.
        """
        descriptions: Dict[str, str] = {
            RegulatoryFramework.EU_FGAS_2024.value: "EU F-Gas Regulation (EU) 2024/573",
            RegulatoryFramework.KIGALI_NON_A5.value: "Kigali Amendment - Non-Article 5 (developed) parties",
            RegulatoryFramework.KIGALI_A5_G1.value: "Kigali Amendment - Article 5 Group 1 (developing) parties",
            RegulatoryFramework.EPA_SUBPART_DD.value: "US EPA 40 CFR Part 98 Subpart DD (SF6 electrical equipment)",
            RegulatoryFramework.EPA_SUBPART_OO.value: "US EPA 40 CFR Part 98 Subpart OO (industrial GHG suppliers)",
            RegulatoryFramework.EPA_SUBPART_L.value: "US EPA 40 CFR Part 98 Subpart L (fluorinated gas production)",
            RegulatoryFramework.GHG_PROTOCOL.value: "GHG Protocol Corporate Standard - Scope 1",
            RegulatoryFramework.ISO_14064.value: "ISO 14064-1:2018 - Organization-level GHG quantification",
            RegulatoryFramework.CSRD_ESRS_E1.value: "CSRD/ESRS E1 - European Sustainability Reporting",
            RegulatoryFramework.UK_FGAS.value: "UK F-Gas Regulations (post-Brexit)",
        }
        return [
            {"framework": fw.value, "description": descriptions.get(fw.value, "")}
            for fw in RegulatoryFramework
        ]

    # ------------------------------------------------------------------
    # Public API: Upcoming Deadlines
    # ------------------------------------------------------------------

    def get_upcoming_deadlines(
        self,
        frameworks: Optional[List[str]] = None,
        from_date: Optional[date] = None,
        within_years: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get upcoming regulatory deadlines within a time window.

        Compiles phase-down step changes and equipment ban effective
        dates that fall within the specified time window.

        Args:
            frameworks: Optional list of frameworks to check.
                Defaults to all frameworks.
            from_date: Start date for the window. Defaults to today.
            within_years: Number of years to look ahead. Defaults to 5.

        Returns:
            List of deadline dictionaries sorted by date, each with:
            date, framework, type, description, impact.
        """
        if from_date is None:
            from_date = date.today()
        if frameworks is None:
            frameworks = [f.value for f in RegulatoryFramework]

        end_year = from_date.year + within_years
        deadlines: List[Dict[str, Any]] = []

        # Phase-down step changes
        for fw in frameworks:
            try:
                schedule, terminal = self._get_phase_down_schedule(fw)
            except ValueError:
                continue
            if not schedule:
                continue

            sorted_years = sorted(schedule.keys())
            for i, yr in enumerate(sorted_years):
                if yr < from_date.year or yr > end_year:
                    continue
                # Check if this year introduces a new (lower) target
                if i > 0 and schedule[yr] < schedule[sorted_years[i - 1]]:
                    deadlines.append({
                        "date": f"{yr}-01-01",
                        "framework": fw,
                        "type": "phase_down_step",
                        "description": (
                            f"Phase-down target reduces to {schedule[yr]}% "
                            f"(from {schedule[sorted_years[i - 1]]}%)"
                        ),
                        "target_pct": str(schedule[yr]),
                        "impact": "HIGH" if (schedule[sorted_years[i - 1]] - schedule[yr]) >= Decimal("10") else "MEDIUM",
                    })

        # Equipment ban effective dates
        if RegulatoryFramework.EU_FGAS_2024.value in frameworks:
            for ban in _EU_FGAS_EQUIPMENT_BANS:
                eff_date = ban["effective_date"]
                if from_date <= eff_date <= date(end_year, 12, 31):
                    deadlines.append({
                        "date": str(eff_date),
                        "framework": RegulatoryFramework.EU_FGAS_2024.value,
                        "type": "equipment_ban",
                        "description": ban["description"],
                        "equipment_category": ban["equipment_category"],
                        "max_gwp": ban["max_gwp"],
                        "impact": "HIGH",
                    })

        # Sort by date
        deadlines.sort(key=lambda d: d["date"])

        return deadlines

    # ------------------------------------------------------------------
    # Public API: History and Stats
    # ------------------------------------------------------------------

    def get_history(
        self,
        framework: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ComplianceRecord]:
        """Return compliance check history.

        Args:
            framework: Optional filter by framework.
            status: Optional filter by compliance status.
            limit: Optional maximum number of recent entries.

        Returns:
            List of ComplianceRecord objects, oldest first.
        """
        with self._lock:
            entries = list(self._compliance_history)

        if framework:
            entries = [e for e in entries if e.framework == framework]
        if status:
            entries = [e for e in entries if e.status == status]
        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        with self._lock:
            history_count = len(self._compliance_history)
            quota_count = len(self._quotas)

            by_framework: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            for entry in self._compliance_history:
                by_framework[entry.framework] = (
                    by_framework.get(entry.framework, 0) + 1
                )
                by_status[entry.status] = (
                    by_status.get(entry.status, 0) + 1
                )

        return {
            "total_checks": history_count,
            "active_quotas": quota_count,
            "frameworks_supported": len(RegulatoryFramework),
            "checks_by_framework": by_framework,
            "checks_by_status": by_status,
        }

    def clear(self) -> None:
        """Clear all history and quotas. Intended for testing."""
        with self._lock:
            self._compliance_history.clear()
            self._quotas.clear()
        logger.info("ComplianceTrackerEngine cleared")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_phase_down_schedule(
        self, framework: str,
    ) -> Tuple[Dict[int, Decimal], Decimal]:
        """Get the phase-down schedule and terminal percentage for a framework."""
        schedules: Dict[str, Tuple[Dict[int, Decimal], Decimal]] = {
            RegulatoryFramework.EU_FGAS_2024.value: (
                _EU_FGAS_PHASE_DOWN, _EU_FGAS_TERMINAL_PCT
            ),
            RegulatoryFramework.KIGALI_NON_A5.value: (
                _KIGALI_NON_A5_PHASE_DOWN, _KIGALI_NON_A5_TERMINAL_PCT
            ),
            RegulatoryFramework.KIGALI_A5_G1.value: (
                _KIGALI_A5_G1_PHASE_DOWN, _KIGALI_A5_G1_TERMINAL_PCT
            ),
            RegulatoryFramework.UK_FGAS.value: (
                _UK_FGAS_PHASE_DOWN, _UK_FGAS_TERMINAL_PCT
            ),
        }

        if framework in schedules:
            return schedules[framework]
        return ({}, Decimal("100"))

    @staticmethod
    def _validate_framework(framework: str) -> None:
        """Validate that framework is a recognized regulatory framework."""
        valid = {e.value for e in RegulatoryFramework}
        if framework not in valid:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Valid frameworks: {sorted(valid)}"
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        with self._lock:
            hist_count = len(self._compliance_history)
            quota_count = len(self._quotas)
        return (
            f"ComplianceTrackerEngine("
            f"frameworks={len(RegulatoryFramework)}, "
            f"checks={hist_count}, "
            f"quotas={quota_count})"
        )

    def __len__(self) -> int:
        """Return the number of compliance checks performed."""
        with self._lock:
            return len(self._compliance_history)
