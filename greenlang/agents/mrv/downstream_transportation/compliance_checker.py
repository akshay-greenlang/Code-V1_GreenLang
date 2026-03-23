# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-022 Engine 6

This module implements regulatory compliance checking for Downstream Transportation
& Distribution emissions (GHG Protocol Scope 3 Category 9) against 7 regulatory
frameworks plus a dedicated double-counting prevention checker.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 9 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. ISO 14083:2023 (transport-specific, WTW mandatory)
4. CSRD/ESRS E1 Climate Change
5. CDP Climate Change Questionnaire (C6.5 module)
6. SBTi (Science Based Targets initiative)
7. SB 253 (California Climate Corporate Data Accountability Act)

Category 9-Specific Compliance Rules:
- Incoterm-based boundary enforcement (Cat 4 vs Cat 9)
- Downstream distribution chain completeness (9a-9d sub-activities)
- Double-counting prevention (10 rules: DC-DTO-001 through DC-DTO-010)
- Well-to-Wheel vs Tank-to-Wheel scope for ISO 14083
- Data quality scoring per GHG Protocol method hierarchy
- Mode disclosure and breakdown
- Return logistics and reverse logistics handling
- Warehouse/distribution center emissions boundary
- Last-mile delivery tracking
- Retail storage energy attribution

Double-Counting Prevention Rules:
    DC-DTO-001: No overlap with Category 4 (upstream transport)
    DC-DTO-002: No overlap with Scope 1 (company-owned fleet)
    DC-DTO-003: No overlap with Scope 2 (warehouse electricity)
    DC-DTO-004: No overlap with Category 1 (cradle-to-gate transport)
    DC-DTO-005: No overlap with Category 3 (WTT fuel)
    DC-DTO-006: No overlap with Category 12 (end-of-life transport)
    DC-DTO-007: No retail store Scope 1/2 in Cat 9
    DC-DTO-008: Incoterm boundary: seller-paid vs buyer-paid
    DC-DTO-009: No overlap between sub-activities 9a-9d
    DC-DTO-010: Return logistics direction check

Example:
    >>> engine = get_compliance_checker()
    >>> result = engine.check_all_frameworks(calculation_data)
    >>> summary = engine.get_compliance_summary()
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "dto_compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceFramework(str, Enum):
    """Supported regulatory frameworks for downstream transportation."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14083 = "iso_14083"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class TransportMode(str, Enum):
    """Transport mode classification."""

    ROAD = "road"
    RAIL = "rail"
    SEA = "sea"
    AIR = "air"
    INLAND_WATERWAY = "inland_waterway"
    PIPELINE = "pipeline"
    MULTIMODAL = "multimodal"
    LAST_MILE = "last_mile"


class IncotermCategory(str, Enum):
    """Incoterm-based classification for Cat 4 vs Cat 9."""

    CATEGORY_4 = "CATEGORY_4"  # Buyer pays downstream transport
    CATEGORY_9 = "CATEGORY_9"  # Seller pays downstream transport
    AMBIGUOUS = "AMBIGUOUS"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 9."""

    CATEGORY_1 = "CATEGORY_1"   # Purchased goods (cradle-to-gate)
    CATEGORY_3 = "CATEGORY_3"   # Fuel & energy activities (WTT)
    CATEGORY_4 = "CATEGORY_4"   # Upstream transportation
    CATEGORY_12 = "CATEGORY_12"  # End-of-life transport
    SCOPE_1 = "SCOPE_1"         # Company-owned fleet
    SCOPE_2 = "SCOPE_2"         # Warehouse electricity


class DistributionSubActivity(str, Enum):
    """GHG Protocol Category 9 sub-activities."""

    OUTBOUND_TRANSPORT = "9a"     # Post-sale transport per Incoterms
    OUTBOUND_DISTRIBUTION = "9b"  # Distribution center / warehouse
    RETAIL_STORAGE = "9c"         # Third-party retail energy
    LAST_MILE_DELIVERY = "9d"     # Final delivery to end consumer


# ==============================================================================
# INCOTERM CLASSIFICATION TABLE
# ==============================================================================

# Incoterms 2020: classification for downstream (Category 9) vs upstream (Cat 4)
# Category 9 = seller (reporting company) pays transport AFTER point of sale
# Category 4 = reporting company pays transport BEFORE point of sale
INCOTERM_CLASSIFICATION: Dict[str, IncotermCategory] = {
    # Seller pays downstream transport -> Category 9
    "DDP": IncotermCategory.CATEGORY_9,
    "DAP": IncotermCategory.CATEGORY_9,
    "DPU": IncotermCategory.CATEGORY_9,
    "CIF": IncotermCategory.CATEGORY_9,
    "CIP": IncotermCategory.CATEGORY_9,
    "CPT": IncotermCategory.CATEGORY_9,
    "CFR": IncotermCategory.CATEGORY_9,
    # Buyer pays downstream transport -> Category 4 (not Cat 9)
    "EXW": IncotermCategory.CATEGORY_4,
    "FCA": IncotermCategory.CATEGORY_4,
    "FAS": IncotermCategory.CATEGORY_4,
    "FOB": IncotermCategory.CATEGORY_4,
}

# Incoterms that belong in Category 9 (seller-arranged downstream transport)
CAT_9_INCOTERMS: Set[str] = {
    "DDP", "DAP", "DPU", "CIF", "CIP", "CPT", "CFR",
}

# Incoterms that belong in Category 4 (buyer-arranged, NOT Cat 9)
CAT_4_INCOTERMS: Set[str] = {
    "EXW", "FCA", "FAS", "FOB",
}


# ==============================================================================
# FRAMEWORK WEIGHTS FOR OVERALL SCORING
# ==============================================================================

FRAMEWORK_WEIGHTS: Dict[ComplianceFramework, Decimal] = {
    ComplianceFramework.GHG_PROTOCOL: Decimal("1.00"),
    ComplianceFramework.ISO_14064: Decimal("0.85"),
    ComplianceFramework.ISO_14083: Decimal("0.90"),
    ComplianceFramework.CSRD_ESRS: Decimal("0.90"),
    ComplianceFramework.CDP: Decimal("0.85"),
    ComplianceFramework.SBTI: Decimal("0.80"),
    ComplianceFramework.SB_253: Decimal("0.75"),
}


# ==============================================================================
# REQUIRED DISCLOSURES PER FRAMEWORK
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL.value: [
        "total_co2e",
        "calculation_method",
        "ef_sources",
        "incoterm_boundary",
        "mode_breakdown",
        "data_quality_score",
        "double_counting_documentation",
        "reporting_period",
        "base_year",
    ],
    ComplianceFramework.ISO_14064.value: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
        "boundary_definition",
        "reporting_period",
        "verification_statement",
    ],
    ComplianceFramework.ISO_14083.value: [
        "total_co2e",
        "wtw_emissions",
        "mode_specific_efs",
        "load_factor",
        "empty_running",
        "hub_emissions",
    ],
    ComplianceFramework.CSRD_ESRS.value: [
        "total_co2e",
        "mode_breakdown",
        "time_series",
        "reduction_targets",
        "data_sources",
        "methodology",
        "double_counting_documentation",
        "assurance_statement",
    ],
    ComplianceFramework.CDP.value: [
        "total_co2e",
        "calculation_method",
        "relevance_assessment",
        "verification_status",
        "year_over_year_change",
        "reduction_targets",
    ],
    ComplianceFramework.SBTI.value: [
        "total_co2e",
        "flag_separation",
        "target_boundary",
        "coverage_percentage",
        "base_year",
        "progress_tracking",
    ],
    ComplianceFramework.SB_253.value: [
        "total_co2e",
        "assurance_opinion",
        "carb_reporting_format",
        "completeness_assessment",
        "methodology",
        "reporting_period",
    ],
}


# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass
class ComplianceFinding:
    """Single compliance finding with rule code and severity."""

    rule_code: str
    description: str
    severity: ComplianceSeverity
    framework: str
    status: ComplianceStatus = ComplianceStatus.FAIL
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    regulation_reference: Optional[str] = None


@dataclass
class FrameworkCheckState:
    """Internal state accumulator for a single framework check."""

    framework: ComplianceFramework
    findings: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0

    def add_pass(self, rule_code: str, description: str) -> None:
        """Record a passed check."""
        self.passed_checks += 1
        self.total_checks += 1

    def add_fail(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a failed check with a finding."""
        self.failed_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.FAIL,
                details=details,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def add_warning(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.MEDIUM,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a warning check with a finding."""
        self.warning_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.WARNING,
                details=details,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def compute_score(self) -> Decimal:
        """
        Compute compliance score (0-100) based on pass/fail/warning ratio.

        Weights:
            - Each failed check reduces score proportionally
            - Warnings reduce score at 50% of a failure
            - Perfect score = 100 (all checks passed)
        """
        if self.total_checks == 0:
            return Decimal("100.00")

        penalty_points = (
            Decimal(str(self.failed_checks))
            + Decimal(str(self.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(self.total_checks))
        score = (
            (max_points - penalty_points) / max_points * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if score < Decimal("0"):
            score = Decimal("0.00")
        if score > Decimal("100"):
            score = Decimal("100.00")

        return score

    def compute_status(self) -> ComplianceStatus:
        """Compute overall status from findings."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 0:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS

    def to_result(self) -> Dict[str, Any]:
        """Convert accumulated state to a compliance check result dictionary."""
        findings_dicts = [
            {
                "rule_code": f.rule_code,
                "description": f.description,
                "severity": f.severity.value,
                "status": f.status.value,
                "recommendation": f.recommendation,
                "regulation_reference": f.regulation_reference,
                "details": f.details,
            }
            for f in self.findings
        ]

        recommendations = [
            f.recommendation
            for f in self.findings
            if f.recommendation is not None
        ]

        return {
            "framework": self.framework.value,
            "status": self.compute_status().value,
            "score": float(self.compute_score()),
            "findings": findings_dicts,
            "recommendations": recommendations,
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "warnings": self.warning_checks,
            "total_checks": self.total_checks,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Downstream Transportation & Distribution (Category 9).

    Validates calculation results against 7 regulatory frameworks with
    Category 9-specific rules including Incoterm boundary enforcement,
    downstream distribution chain completeness, double-counting prevention,
    and sub-activity (9a-9d) disclosure requirements.

    Thread Safety:
        Singleton pattern with threading.RLock for concurrent access.

    Attributes:
        _enabled_frameworks: List of enabled compliance framework names
        _check_count: Running count of compliance checks performed
        _last_results: Most recent framework check results (for summary)

    Example:
        >>> engine = get_compliance_checker()
        >>> results = engine.check_all_frameworks(calc_result)
        >>> summary = engine.get_compliance_summary()
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with default configuration."""
        self._enabled_frameworks: List[str] = [fw.value for fw in ComplianceFramework]
        self._check_count: int = 0
        self._last_results: Dict[str, Dict[str, Any]] = {}
        self._strict_mode: bool = False

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d, agent=GL-MRV-S3-009",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
        )

    @classmethod
    def get_instance(cls) -> "ComplianceCheckerEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            ComplianceCheckerEngine singleton instance.

        Example:
            >>> engine = ComplianceCheckerEngine.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level RLock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("ComplianceCheckerEngine singleton reset")

    # ==========================================================================
    # Main Entry Points
    # ==========================================================================

    def check_all_frameworks(
        self, data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked.

        Args:
            data: Calculation result dictionary containing total_co2e,
                mode_breakdown, incoterms, methods, ef_sources, etc.

        Returns:
            Dictionary mapping framework name string to compliance check dict.

        Example:
            >>> all_results = engine.check_all_frameworks(calc_data)
            >>> ghg_result = all_results.get("ghg_protocol")
            >>> ghg_result["status"]
            'pass'
        """
        start_time = time.monotonic()
        logger.info("Running compliance checks for all enabled frameworks")

        all_results: Dict[str, Dict[str, Any]] = {}

        framework_dispatch: Dict[str, Tuple[ComplianceFramework, str]] = {
            ComplianceFramework.GHG_PROTOCOL.value: (
                ComplianceFramework.GHG_PROTOCOL,
                "check_ghg_protocol",
            ),
            ComplianceFramework.ISO_14064.value: (
                ComplianceFramework.ISO_14064,
                "check_iso_14064",
            ),
            ComplianceFramework.ISO_14083.value: (
                ComplianceFramework.ISO_14083,
                "check_iso_14083",
            ),
            ComplianceFramework.CSRD_ESRS.value: (
                ComplianceFramework.CSRD_ESRS,
                "check_csrd_esrs",
            ),
            ComplianceFramework.CDP.value: (
                ComplianceFramework.CDP,
                "check_cdp",
            ),
            ComplianceFramework.SBTI.value: (
                ComplianceFramework.SBTI,
                "check_sbti",
            ),
            ComplianceFramework.SB_253.value: (
                ComplianceFramework.SB_253,
                "check_sb_253",
            ),
        }

        for framework_name in self._enabled_frameworks:
            dispatch = framework_dispatch.get(framework_name)
            if dispatch is None:
                logger.warning(
                    "Unknown framework: '%s', skipping", framework_name,
                )
                continue

            framework_enum, method_name = dispatch
            try:
                check_method = getattr(self, method_name)
                check_result = check_method(data)
                all_results[framework_enum.value] = check_result

                logger.info(
                    "%s compliance: %s (score: %s)",
                    framework_enum.value,
                    check_result["status"],
                    check_result["score"],
                )

            except Exception as e:
                logger.error(
                    "Error checking %s compliance: %s",
                    framework_name,
                    str(e),
                    exc_info=True,
                )
                all_results[framework_enum.value] = {
                    "framework": framework_enum.value,
                    "status": ComplianceStatus.FAIL.value,
                    "score": 0.0,
                    "findings": [
                        {
                            "rule_code": "CHECK_ERROR",
                            "description": f"Compliance check failed: {str(e)}",
                            "severity": ComplianceSeverity.CRITICAL.value,
                            "status": ComplianceStatus.FAIL.value,
                        }
                    ],
                    "recommendations": [
                        "Resolve the compliance check error and rerun."
                    ],
                    "passed": 0,
                    "failed": 1,
                    "warnings": 0,
                    "total_checks": 1,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }

        duration = time.monotonic() - start_time
        self._check_count += 1
        self._last_results = all_results

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.4fs",
            len(all_results),
            duration,
        )

        return all_results

    # ==========================================================================
    # Framework 1: GHG Protocol Scope 3 (Category 9)
    # ==========================================================================

    def check_ghg_protocol(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 9).

        Checks GHG-DTO-001 through GHG-DTO-009:
            GHG-DTO-001: Completeness -- total_co2e present and positive
            GHG-DTO-002: Incoterm boundary -- correct Cat 9 Incoterms only
            GHG-DTO-003: Mode disclosure -- transport modes documented
            GHG-DTO-004: Method hierarchy -- calculation method documented
            GHG-DTO-005: Data quality -- DQI score present
            GHG-DTO-006: WTT inclusion -- well-to-tank documented
            GHG-DTO-007: Return logistics -- return/reverse logistics handled
            GHG-DTO-008: Double-counting -- boundary vs Cat 4/Scope 1/Scope 2
            GHG-DTO-009: Provenance -- audit trail documented

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for GHG Protocol.

        Example:
            >>> res = engine.check_ghg_protocol(calc_data)
            >>> res["status"]
            'pass'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # GHG-DTO-001: Completeness -- total_co2e present and positive
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-DTO-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-DTO-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Ensure total_co2e is calculated and > 0.",
                regulation_reference="GHG Protocol Scope 3, Category 9",
            )

        # GHG-DTO-002: Incoterm boundary
        incoterms = data.get("incoterms") or data.get("incoterm_list") or []
        if incoterms:
            invalid_incoterms = [
                ic for ic in incoterms
                if str(ic).upper() in CAT_4_INCOTERMS
            ]
            if invalid_incoterms:
                state.add_fail(
                    "GHG-DTO-002",
                    f"Category 4 Incoterms found in Category 9 calculation: {invalid_incoterms}",
                    ComplianceSeverity.CRITICAL,
                    details={"invalid_incoterms": invalid_incoterms},
                    recommendation=(
                        "Remove buyer-paid transport (EXW, FCA, FAS, FOB) from "
                        "Category 9; those belong in Category 4."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Category 4 & 9 payment boundary",
                )
            else:
                state.add_pass("GHG-DTO-002", "All Incoterms are valid for Category 9")
        else:
            state.add_warning(
                "GHG-DTO-002",
                "No Incoterms specified for shipments",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Assign Incoterms to all downstream shipments to ensure "
                    "correct category assignment (Cat 4 vs Cat 9)."
                ),
                regulation_reference="GHG Protocol Scope 3, Category 4 & 9 guidance",
            )

        # GHG-DTO-003: Mode disclosure
        mode_breakdown = data.get("mode_breakdown") or data.get("by_mode")
        if mode_breakdown and isinstance(mode_breakdown, dict) and len(mode_breakdown) > 0:
            state.add_pass("GHG-DTO-003", "Transport mode breakdown disclosed")
        else:
            state.add_fail(
                "GHG-DTO-003",
                "Transport mode breakdown not disclosed",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by transport mode "
                    "(road, rail, sea, air, last-mile, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Category 9",
            )

        # GHG-DTO-004: Method hierarchy
        method = data.get("method") or data.get("calculation_method")
        if method:
            state.add_pass("GHG-DTO-004", "Calculation method documented")
        else:
            state.add_fail(
                "GHG-DTO-004",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used (supplier-specific, "
                    "distance-based, average-data, or spend-based)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 9.1",
            )

        # GHG-DTO-005: Data quality
        dqi_score = data.get("dqi_score") or data.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-DTO-005", "DQI score present")
        else:
            state.add_warning(
                "GHG-DTO-005",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-DTO-006: WTT inclusion
        wtt_documented = (
            data.get("wtt_included") is not None
            or data.get("wtt_co2e") is not None
            or data.get("emission_scope") in ("WTW", "wtw")
        )
        if wtt_documented:
            state.add_pass("GHG-DTO-006", "Well-to-Tank emissions documented")
        else:
            state.add_warning(
                "GHG-DTO-006",
                "Well-to-Tank (WTT) emissions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether WTT emissions are included (WTW scope) "
                    "or excluded (TTW scope) and justify the choice."
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 9",
            )

        # GHG-DTO-007: Return logistics
        return_logistics = data.get("return_logistics_documented")
        if return_logistics is not None:
            state.add_pass("GHG-DTO-007", "Return logistics documented")
        else:
            state.add_warning(
                "GHG-DTO-007",
                "Return logistics (reverse logistics) not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document whether return logistics emissions (product returns) "
                    "are included or excluded from Category 9."
                ),
                regulation_reference="GHG Protocol Scope 3, Category 9 guidance",
            )

        # GHG-DTO-008: Double-counting documentation
        dc_documented = data.get("double_counting_documented") or data.get(
            "double_counting_documentation"
        )
        if dc_documented:
            state.add_pass("GHG-DTO-008", "Double-counting prevention documented")
        else:
            state.add_warning(
                "GHG-DTO-008",
                "Double-counting prevention not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document boundaries between Cat 4, Cat 9, Scope 1 (fleet), "
                    "Scope 2 (warehouse), and Cat 3 (WTT) to prevent overlap."
                ),
                regulation_reference="GHG Protocol Scope 3, Double-counting guidance",
            )

        # GHG-DTO-009: Provenance
        provenance = data.get("provenance_hash") or data.get("audit_trail")
        if provenance:
            state.add_pass("GHG-DTO-009", "Provenance audit trail present")
        else:
            state.add_warning(
                "GHG-DTO-009",
                "Provenance audit trail not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Include SHA-256 provenance hash for calculation reproducibility "
                    "and audit readiness."
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 7.3",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 2: ISO 14064-1:2018
    # ==========================================================================

    def check_iso_14064(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with ISO 14064-1:2018.

        Checks ISO-DTO-001 through ISO-DTO-007:
            ISO-DTO-001: Uncertainty analysis present
            ISO-DTO-002: Boundary definition documented
            ISO-DTO-003: Documentation -- methodology described
            ISO-DTO-004: Methodology -- quantification approach
            ISO-DTO-005: Base year documented
            ISO-DTO-006: Recalculation policy defined
            ISO-DTO-007: Verification readiness

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for ISO 14064.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # ISO-DTO-001: Uncertainty analysis
        uncertainty = (
            data.get("uncertainty_analysis")
            or data.get("uncertainty")
            or data.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-DTO-001", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-DTO-001",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis "
                    "(Monte Carlo, analytical, or IPCC Tier 2)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-DTO-002: Boundary definition
        boundary = data.get("boundary_definition") or data.get("boundary_defined")
        if boundary:
            state.add_pass("ISO-DTO-002", "Boundary definition documented")
        else:
            state.add_fail(
                "ISO-DTO-002",
                "Organizational and operational boundaries not defined",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Define boundaries per ISO 14064-1: equity share, "
                    "operational control, or financial control."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-DTO-003: Documentation -- total_co2e present
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-DTO-003", "Total CO2e present for documentation")
        else:
            state.add_fail(
                "ISO-DTO-003",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-DTO-004: Methodology described
        methodology = (
            data.get("methodology")
            or data.get("method")
            or data.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-DTO-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-DTO-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, and calculation approach."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-DTO-005: Base year documented
        base_year = data.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-DTO-005", "Base year documented")
        else:
            state.add_fail(
                "ISO-DTO-005",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-DTO-006: Recalculation policy
        recalculation = data.get("recalculation_policy")
        if recalculation:
            state.add_pass("ISO-DTO-006", "Recalculation policy defined")
        else:
            state.add_warning(
                "ISO-DTO-006",
                "Recalculation policy not defined",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Define triggers for base year recalculation "
                    "(structural changes, methodology changes, etc.)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4.2",
            )

        # ISO-DTO-007: Verification readiness
        verification = data.get("verification_status") or data.get("verified")
        if verification is not None:
            state.add_pass("ISO-DTO-007", "Verification status documented")
        else:
            state.add_warning(
                "ISO-DTO-007",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether emissions have been third-party "
                    "verified per ISO 14064-3."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 10",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 3: ISO 14083:2023
    # ==========================================================================

    def check_iso_14083(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with ISO 14083:2023 (transport-specific standard).

        Checks ISO83-DTO-001 through ISO83-DTO-006:
            ISO83-DTO-001: WTW mandatory -- Well-to-Wheel scope required
            ISO83-DTO-002: Mode-specific EFs -- correct EFs per mode
            ISO83-DTO-003: GLEC alignment -- GLEC Framework compatibility
            ISO83-DTO-004: Load factor -- load factor documented
            ISO83-DTO-005: Empty running -- empty running considered
            ISO83-DTO-006: Hub emissions -- hub/transshipment emissions

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for ISO 14083.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14083)

        # ISO83-DTO-001: WTW mandatory
        emission_scope = data.get("emission_scope") or data.get("scope")
        if emission_scope and str(emission_scope).upper() in ("WTW", "WELL_TO_WHEEL"):
            state.add_pass("ISO83-DTO-001", "Well-to-Wheel (WTW) scope used")
        else:
            state.add_fail(
                "ISO83-DTO-001",
                f"ISO 14083 requires WTW scope, but '{emission_scope}' was used",
                ComplianceSeverity.CRITICAL,
                details={"current_scope": str(emission_scope) if emission_scope else "not_specified"},
                recommendation=(
                    "Use Well-to-Wheel (WTW) emission factors for ISO 14083 "
                    "compliance. TTW alone is not sufficient."
                ),
                regulation_reference="ISO 14083:2023, Section 6.3.2",
            )

        # ISO83-DTO-002: Mode-specific emission factors
        ef_sources = data.get("ef_sources") or data.get("ef_source")
        mode_efs_documented = data.get("mode_specific_efs")
        if mode_efs_documented or ef_sources:
            state.add_pass("ISO83-DTO-002", "Mode-specific emission factors documented")
        else:
            state.add_fail(
                "ISO83-DTO-002",
                "Mode-specific emission factors not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document emission factors per transport mode "
                    "(road, rail, sea, air) with source and version."
                ),
                regulation_reference="ISO 14083:2023, Section 6.2",
            )

        # ISO83-DTO-003: GLEC alignment
        glec_aligned = data.get("glec_aligned") or data.get("glec_factors_used")
        if glec_aligned:
            state.add_pass("ISO83-DTO-003", "GLEC Framework alignment confirmed")
        else:
            state.add_warning(
                "ISO83-DTO-003",
                "GLEC Framework alignment not confirmed",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Use GLEC-accredited emission factors where available "
                    "for alignment with ISO 14083."
                ),
                regulation_reference="ISO 14083:2023, Annex A",
            )

        # ISO83-DTO-004: Load factor
        load_factor = data.get("load_factor") or data.get("load_factor_documented")
        if load_factor is not None:
            state.add_pass("ISO83-DTO-004", "Load factors documented")
        else:
            state.add_warning(
                "ISO83-DTO-004",
                "Load factors not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document load factors (mass or volume utilization) "
                    "for all transport modes."
                ),
                regulation_reference="ISO 14083:2023, Section 6.3.6",
            )

        # ISO83-DTO-005: Empty running
        empty_running = data.get("empty_running_considered") or data.get("empty_running")
        if empty_running is not None:
            state.add_pass("ISO83-DTO-005", "Empty running considered")
        else:
            state.add_warning(
                "ISO83-DTO-005",
                "Empty running not considered",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Consider empty running (empty return trips) in calculations "
                    "or document why it is excluded."
                ),
                regulation_reference="ISO 14083:2023, Section 6.3.5",
            )

        # ISO83-DTO-006: Hub emissions
        hub_emissions = data.get("hub_emissions") or data.get("warehouse_emissions")
        if hub_emissions is not None:
            state.add_pass("ISO83-DTO-006", "Hub/transshipment emissions included")
        else:
            state.add_warning(
                "ISO83-DTO-006",
                "Hub/transshipment emissions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Include emissions from hubs, transshipment points, "
                    "distribution centers, and warehousing in the transport chain."
                ),
                regulation_reference="ISO 14083:2023, Transport chain concept",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 4: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd_esrs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        Checks CSRD-DTO-001 through CSRD-DTO-008:
            CSRD-DTO-001: E1-6 Scope 3 -- total emissions reported
            CSRD-DTO-002: Mode breakdown -- by transport mode
            CSRD-DTO-003: Time series -- year-over-year data
            CSRD-DTO-004: Targets -- reduction targets documented
            CSRD-DTO-005: Data sources -- documented
            CSRD-DTO-006: Methodology -- described
            CSRD-DTO-007: Double-counting -- boundary documentation
            CSRD-DTO-008: Assurance -- assurance statement

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for CSRD/ESRS.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # CSRD-DTO-001: E1-6 Scope 3 total
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-DTO-001", "Scope 3 Category 9 emissions reported")
        else:
            state.add_fail(
                "CSRD-DTO-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 9 emissions.",
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-DTO-002: Mode breakdown
        mode_breakdown = data.get("mode_breakdown") or data.get("by_mode")
        if mode_breakdown and isinstance(mode_breakdown, dict) and len(mode_breakdown) > 0:
            state.add_pass("CSRD-DTO-002", "Mode breakdown present")
        else:
            state.add_fail(
                "CSRD-DTO-002",
                "Mode breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by transport mode "
                    "(road, rail, sea, air, last-mile)."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-DTO-003: Time series
        time_series = (
            data.get("time_series")
            or data.get("year_over_year")
            or data.get("trend")
        )
        if time_series is not None:
            state.add_pass("CSRD-DTO-003", "Time series data present")
        else:
            state.add_warning(
                "CSRD-DTO-003",
                "Time series data not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide year-over-year emissions data to show trends "
                    "per ESRS E1-6 requirements."
                ),
                regulation_reference="ESRS E1-6, para 51(e)",
            )

        # CSRD-DTO-004: Targets
        targets = data.get("targets") or data.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-DTO-004", "Reduction targets documented")
        else:
            state.add_warning(
                "CSRD-DTO-004",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for downstream "
                    "transportation (e.g., modal shift, efficiency gains)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-DTO-005: Data sources
        data_sources = data.get("data_sources") or data.get("ef_sources")
        if data_sources:
            state.add_pass("CSRD-DTO-005", "Data sources documented")
        else:
            state.add_warning(
                "CSRD-DTO-005",
                "Data sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document all data sources including emission factor "
                    "databases, carrier data, and estimation methods."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-DTO-006: Methodology
        methodology = (
            data.get("methodology")
            or data.get("method")
            or data.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-DTO-006", "Methodology described")
        else:
            state.add_fail(
                "CSRD-DTO-006",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for downstream "
                    "transportation emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-DTO-007: Double-counting documentation
        dc_doc = data.get("double_counting_documented") or data.get(
            "double_counting_documentation"
        )
        if dc_doc:
            state.add_pass("CSRD-DTO-007", "Double-counting prevention documented")
        else:
            state.add_warning(
                "CSRD-DTO-007",
                "Double-counting prevention not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document boundary between Category 4 (upstream) and "
                    "Category 9 (downstream) and Scope 1/2 overlap prevention."
                ),
                regulation_reference="ESRS E1-6, Scope 3 guidance",
            )

        # CSRD-DTO-008: Assurance
        assurance = (
            data.get("assurance") or data.get("assurance_statement")
            or data.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("CSRD-DTO-008", "Assurance statement present")
        else:
            state.add_warning(
                "CSRD-DTO-008",
                "Assurance statement not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Obtain limited or reasonable assurance for Scope 3 "
                    "Category 9 emissions per CSRD requirements."
                ),
                regulation_reference="CSRD Art 34(1)(aa)",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 5: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with CDP Climate Change Questionnaire (C6.5 module).

        Checks CDP-DTO-001 through CDP-DTO-006:
            CDP-DTO-001: C6.5 module -- total emissions reported
            CDP-DTO-002: Method -- calculation method documented
            CDP-DTO-003: Relevance -- Category 9 relevance assessment
            CDP-DTO-004: Verification -- verification status
            CDP-DTO-005: Year-over-year -- emissions trend
            CDP-DTO-006: Targets -- reduction targets

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for CDP.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # CDP-DTO-001: C6.5 module -- total emissions
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-DTO-001", "Total CO2e reported for CDP C6.5")
        else:
            state.add_fail(
                "CDP-DTO-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 9 emissions to CDP.",
                regulation_reference="CDP CC Module 7, Q7.3a",
            )

        # CDP-DTO-002: Method
        method = data.get("method") or data.get("calculation_method")
        if method:
            state.add_pass("CDP-DTO-002", "Calculation method documented for CDP")
        else:
            state.add_fail(
                "CDP-DTO-002",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used for Category 9 "
                    "(supplier-specific, distance-based, average-data, or spend-based)."
                ),
                regulation_reference="CDP CC Module 7, Q7.3a",
            )

        # CDP-DTO-003: Relevance assessment
        relevance = data.get("relevance_assessment") or data.get("materiality")
        if relevance is not None:
            state.add_pass("CDP-DTO-003", "Relevance assessment present")
        else:
            state.add_warning(
                "CDP-DTO-003",
                "Category 9 relevance assessment not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the relevance and materiality of Category 9 "
                    "emissions within total Scope 3. CDP requires a relevance "
                    "assessment for each category."
                ),
                regulation_reference="CDP CC Module 7, Q7.1",
            )

        # CDP-DTO-004: Verification
        verification = (
            data.get("verification_status")
            or data.get("verified")
            or data.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-DTO-004", "Verification status documented")
        else:
            state.add_warning(
                "CDP-DTO-004",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 9 emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module 10, Q10.1",
            )

        # CDP-DTO-005: Year-over-year
        yoy = data.get("year_over_year_change") or data.get("trend") or data.get("time_series")
        if yoy is not None:
            state.add_pass("CDP-DTO-005", "Year-over-year change documented")
        else:
            state.add_warning(
                "CDP-DTO-005",
                "Year-over-year change not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Report year-over-year emissions change for Category 9 "
                    "to demonstrate progress and trend."
                ),
                regulation_reference="CDP CC Module 7, Q7.3a",
            )

        # CDP-DTO-006: Targets
        targets = data.get("targets") or data.get("reduction_targets")
        if targets:
            state.add_pass("CDP-DTO-006", "Reduction targets documented for CDP")
        else:
            state.add_warning(
                "CDP-DTO-006",
                "Reduction targets not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document reduction targets for downstream transportation "
                    "(modal shift, route optimization, carrier engagement)."
                ),
                regulation_reference="CDP CC Module 4, Q4.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 6: SBTi
    # ==========================================================================

    def check_sbti(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with Science Based Targets initiative.

        Checks SBTI-DTO-001 through SBTI-DTO-006:
            SBTI-DTO-001: FLAG/non-FLAG -- separation documented
            SBTI-DTO-002: Target boundary -- Cat 9 in target boundary
            SBTI-DTO-003: 67% coverage -- Scope 3 coverage threshold
            SBTI-DTO-004: Base year -- base year defined
            SBTI-DTO-005: Progress -- progress tracking
            SBTI-DTO-006: Method -- calculation method hierarchy

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for SBTi.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI)

        # SBTI-DTO-001: FLAG/non-FLAG separation
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-DTO-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-DTO-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate total Category 9 emissions for SBTi target boundary.",
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        flag_documented = data.get("flag_separation") or data.get("flag_non_flag")
        if flag_documented is not None:
            state.add_pass("SBTI-DTO-001-FLAG", "FLAG/non-FLAG separation documented")
        else:
            state.add_warning(
                "SBTI-DTO-001-FLAG",
                "FLAG/non-FLAG separation not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document whether Category 9 emissions include FLAG "
                    "(Forest, Land and Agriculture) related transport."
                ),
                regulation_reference="SBTi FLAG Guidance 2023",
            )

        # SBTI-DTO-002: Target boundary
        in_target = data.get("cat9_in_target_boundary") or data.get("target_boundary")
        if in_target is not None:
            state.add_pass("SBTI-DTO-002", "Category 9 target boundary documented")
        else:
            state.add_warning(
                "SBTI-DTO-002",
                "Category 9 inclusion in SBTi target boundary not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 9 is included in the "
                    "Scope 3 reduction target boundary."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-DTO-003: 67% coverage
        total_scope3 = data.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat9_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                scope3_coverage = data.get("scope3_coverage_percentage")
                if scope3_coverage is not None:
                    coverage = Decimal(str(scope3_coverage))
                    if coverage >= Decimal("67"):
                        state.add_pass(
                            "SBTI-DTO-003",
                            f"Scope 3 coverage {coverage}% meets SBTi 67% threshold",
                        )
                    else:
                        state.add_warning(
                            "SBTI-DTO-003",
                            f"Scope 3 coverage {coverage}% below SBTi 67% threshold",
                            ComplianceSeverity.HIGH,
                            details={"coverage_percentage": float(coverage)},
                            recommendation=(
                                "Expand Scope 3 coverage to at least 67% of "
                                "total Scope 3 emissions."
                            ),
                            regulation_reference="SBTi Criteria v5.1, C20",
                        )
                else:
                    state.add_warning(
                        "SBTI-DTO-003",
                        "Scope 3 coverage percentage not documented",
                        ComplianceSeverity.MEDIUM,
                        recommendation=(
                            "Document Scope 3 coverage percentage (SBTi "
                            "requires 67% if Scope 3 > 40% of total)."
                        ),
                        regulation_reference="SBTi Criteria v5.1, C20",
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-DTO-003",
                    "Could not calculate Scope 3 coverage (invalid total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-DTO-003",
                "Total Scope 3 emissions not provided for coverage assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 9 coverage."
                ),
            )

        # SBTI-DTO-004: Base year
        base_year = data.get("base_year")
        if base_year is not None:
            state.add_pass("SBTI-DTO-004", "Base year documented for SBTi")
        else:
            state.add_fail(
                "SBTI-DTO-004",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for SBTi target tracking."
                ),
                regulation_reference="SBTi Criteria v5.1, C3",
            )

        # SBTI-DTO-005: Progress tracking
        progress = (
            data.get("progress_tracking")
            or data.get("year_over_year_change")
            or data.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-DTO-005", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-DTO-005",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 9 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-DTO-006: Method hierarchy
        method = data.get("method") or data.get("calculation_method")
        if method:
            method_lower = str(method).lower()
            preferred_methods = {"supplier_specific", "distance_based", "distance-based"}
            if method_lower in preferred_methods or "supplier" in method_lower:
                state.add_pass("SBTI-DTO-006", "Preferred calculation method used")
            else:
                state.add_warning(
                    "SBTI-DTO-006",
                    f"Method '{method}' is not the preferred SBTi approach",
                    ComplianceSeverity.LOW,
                    recommendation=(
                        "SBTi prefers supplier-specific or distance-based methods "
                        "over spend-based for transport categories."
                    ),
                    regulation_reference="SBTi Corporate Manual v2.0",
                )
        else:
            state.add_warning(
                "SBTI-DTO-006",
                "Calculation method not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the calculation method used. SBTi prefers "
                    "supplier-specific or distance-based approaches."
                ),
            )

        return state.to_result()

    # ==========================================================================
    # Framework 7: SB 253
    # ==========================================================================

    def check_sb_253(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Checks SB253-DTO-001 through SB253-DTO-006:
            SB253-DTO-001: Cat 9 mandatory -- mandatory reporting
            SB253-DTO-002: Assurance -- third-party assurance opinion
            SB253-DTO-003: CARB format -- CARB reporting format
            SB253-DTO-004: Completeness -- completeness assessment
            SB253-DTO-005: Methodology -- methodology documented
            SB253-DTO-006: Reporting -- reporting period defined

        Args:
            data: Calculation result dictionary.

        Returns:
            Compliance check result dictionary for SB 253.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # SB253-DTO-001: Cat 9 mandatory
        total_co2e = data.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-DTO-001", "Category 9 emissions reported")
        else:
            state.add_fail(
                "SB253-DTO-001",
                "Category 9 emissions missing or zero (mandatory under SB 253)",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 9 emissions for SB 253 compliance.",
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-DTO-002: Assurance
        assurance = (
            data.get("assurance_opinion")
            or data.get("assurance")
            or data.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-DTO-002", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-DTO-002",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-DTO-003: CARB format
        carb_format = data.get("carb_reporting_format") or data.get("carb_format")
        if carb_format:
            state.add_pass("SB253-DTO-003", "CARB reporting format specified")
        else:
            state.add_warning(
                "SB253-DTO-003",
                "CARB reporting format not specified",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Use CARB-prescribed reporting format for SB 253 "
                    "compliance when format guidance is released."
                ),
                regulation_reference="SB 253, Section 38532(c)",
            )

        # SB253-DTO-004: Completeness
        completeness = data.get("completeness_assessment") or data.get("completeness")
        if completeness is not None:
            state.add_pass("SB253-DTO-004", "Completeness assessment documented")
        else:
            state.add_warning(
                "SB253-DTO-004",
                "Completeness assessment not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document completeness of Category 9 reporting including "
                    "coverage of channels, modes, and geographies."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-DTO-005: Methodology
        methodology = (
            data.get("methodology")
            or data.get("method")
            or data.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-DTO-005", "Methodology documented")
        else:
            state.add_fail(
                "SB253-DTO-005",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-DTO-006: Reporting period
        reporting_period = (
            data.get("reporting_period")
            or data.get("period")
        )
        if reporting_period:
            state.add_pass("SB253-DTO-006", "Reporting period defined")
        else:
            state.add_warning(
                "SB253-DTO-006",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Specify the reporting period (calendar year or fiscal year) "
                    "per SB 253 requirements."
                ),
                regulation_reference="SB 253, Section 38532(a)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention
    # ==========================================================================

    def check_double_counting(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate 10 double-counting prevention rules for downstream transportation.

        Rules:
            DC-DTO-001: No overlap with Category 4 (upstream transport)
            DC-DTO-002: No overlap with Scope 1 (company-owned fleet)
            DC-DTO-003: No overlap with Scope 2 (warehouse electricity)
            DC-DTO-004: No overlap with Category 1 (cradle-to-gate transport)
            DC-DTO-005: No overlap with Category 3 (WTT fuel)
            DC-DTO-006: No overlap with Category 12 (end-of-life transport)
            DC-DTO-007: No retail store Scope 1/2 in Cat 9
            DC-DTO-008: Incoterm boundary: seller-paid vs buyer-paid
            DC-DTO-009: No overlap between sub-activities 9a-9d
            DC-DTO-010: Return logistics direction check

        Args:
            data: Calculation data dictionary with shipments, incoterms,
                boundary flags, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and recommendation.

        Example:
            >>> findings = engine.check_double_counting(calc_data)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[Dict[str, Any]] = []
        shipments = data.get("shipments") or []

        # DC-DTO-001: No overlap with Category 4 (upstream transport)
        also_in_cat4 = data.get("reported_in_cat4", False)
        if also_in_cat4:
            findings.append({
                "rule_code": "DC-DTO-001",
                "description": (
                    "Emissions also reported in Category 4 (upstream transportation). "
                    "Each shipment must be in ONLY one category."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "recommendation": (
                    "Ensure no overlap between Cat 4 (company-paid upstream) and "
                    "Cat 9 (downstream after sale). Use Incoterm boundary."
                ),
            })

        for idx, shipment in enumerate(shipments):
            if shipment.get("reported_in_cat4", False):
                findings.append({
                    "rule_code": "DC-DTO-001",
                    "description": (
                        f"Shipment {idx}: Also reported in Category 4. "
                        "Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "shipment_index": idx,
                    "recommendation": (
                        "Report shipment in only one category. "
                        "Use Incoterms to determine Cat 4 vs Cat 9."
                    ),
                })

        # DC-DTO-002: No overlap with Scope 1 (company-owned fleet)
        for idx, shipment in enumerate(shipments):
            vehicle_ownership = shipment.get("vehicle_ownership", "").lower()
            if vehicle_ownership in ("company_owned", "company", "fleet", "leased"):
                findings.append({
                    "rule_code": "DC-DTO-002",
                    "description": (
                        f"Shipment {idx}: Company-owned/leased vehicle detected. "
                        "Should be reported under Scope 1, not Category 9."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "shipment_index": idx,
                    "recommendation": (
                        "Exclude company-owned fleet from Category 9. "
                        "Report under Scope 1 (mobile combustion)."
                    ),
                })

        # DC-DTO-003: No overlap with Scope 2 (warehouse electricity)
        warehouse_in_scope2 = data.get("warehouse_in_scope2", False)
        warehouse_emissions = data.get("warehouse_emissions")
        if warehouse_in_scope2 and warehouse_emissions:
            findings.append({
                "rule_code": "DC-DTO-003",
                "description": (
                    "Warehouse/DC electricity also reported in Scope 2. "
                    "Third-party warehouse energy is Cat 9; own warehouse is Scope 2."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "recommendation": (
                    "Only include THIRD-PARTY warehouse/DC energy in Cat 9. "
                    "Company-owned warehouse electricity belongs in Scope 2."
                ),
            })

        # DC-DTO-004: No overlap with Category 1 (cradle-to-gate transport)
        cat1_includes_transport = data.get("cat1_includes_downstream_transport", False)
        if cat1_includes_transport:
            findings.append({
                "rule_code": "DC-DTO-004",
                "description": (
                    "Category 1 cradle-to-gate factors may include downstream "
                    "transport. Risk of double-counting with Category 9."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "recommendation": (
                    "Verify Category 1 emission factors exclude downstream "
                    "transport that is separately tracked in Category 9."
                ),
            })

        # DC-DTO-005: No overlap with Category 3 (WTT fuel)
        wtt_in_cat3 = data.get("wtt_reported_in_cat3", False)
        emission_scope = data.get("emission_scope")
        if wtt_in_cat3 and emission_scope and str(emission_scope).upper() == "WTW":
            findings.append({
                "rule_code": "DC-DTO-005",
                "description": (
                    "WTT emissions included in both Category 9 (WTW scope) "
                    "and Category 3 (Fuel & Energy Activities). Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "recommendation": (
                    "If using WTW for Category 9, exclude downstream transport "
                    "fuel WTT from Category 3, or use TTW for Category 9."
                ),
            })

        # DC-DTO-006: No overlap with Category 12 (end-of-life transport)
        cat12_includes_transport = data.get("cat12_includes_transport", False)
        if cat12_includes_transport:
            findings.append({
                "rule_code": "DC-DTO-006",
                "description": (
                    "Category 12 (end-of-life) may include transport to "
                    "disposal/recycling. Risk of overlap with Category 9."
                ),
                "severity": ComplianceSeverity.MEDIUM.value,
                "recommendation": (
                    "Define clear boundary: Category 9 covers transport to "
                    "end consumer; Category 12 covers post-consumer transport."
                ),
            })

        # DC-DTO-007: No retail store Scope 1/2 in Cat 9
        retail_includes_scope12 = data.get("retail_includes_scope12", False)
        if retail_includes_scope12:
            findings.append({
                "rule_code": "DC-DTO-007",
                "description": (
                    "Retail storage emissions include retailer Scope 1/2 "
                    "that should not be in the reporting company's Category 9."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "recommendation": (
                    "Only include energy/emissions attributable to the "
                    "reporting company's products at the retailer, not "
                    "the retailer's total Scope 1/2."
                ),
            })

        # DC-DTO-008: Incoterm boundary
        incoterms = data.get("incoterms") or data.get("incoterm_list") or []
        for idx, ic in enumerate(incoterms):
            ic_upper = str(ic).upper()
            if ic_upper in CAT_4_INCOTERMS:
                findings.append({
                    "rule_code": "DC-DTO-008",
                    "description": (
                        f"Incoterm '{ic_upper}' at index {idx} indicates buyer-paid "
                        "transport, which belongs in Category 4, not Category 9."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "incoterm": ic_upper,
                    "recommendation": (
                        f"Remove {ic_upper} shipments from Category 9. "
                        "Only seller-arranged downstream transport belongs here."
                    ),
                })

        # DC-DTO-009: No overlap between sub-activities 9a-9d
        sub_activity_counts: Dict[str, int] = {}
        for idx, shipment in enumerate(shipments):
            sub_activity = shipment.get("sub_activity")
            shipment_id = shipment.get("shipment_id", f"shipment_{idx}")
            if sub_activity:
                key = f"{shipment_id}|{sub_activity}"
                sub_activity_counts[key] = sub_activity_counts.get(key, 0) + 1

        for key, count in sub_activity_counts.items():
            if count > 1:
                findings.append({
                    "rule_code": "DC-DTO-009",
                    "description": (
                        f"Shipment '{key.split('|')[0]}' has duplicate "
                        f"entries for sub-activity '{key.split('|')[1]}'. "
                        "Potential double-counting within Category 9."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": (
                        "Ensure each shipment is counted once per sub-activity. "
                        "De-duplicate records before aggregation."
                    ),
                })

        # DC-DTO-010: Return logistics direction check
        for idx, shipment in enumerate(shipments):
            direction = shipment.get("direction", "").lower()
            if direction in ("return", "reverse", "inbound"):
                is_return_documented = shipment.get("return_logistics_justified", False)
                if not is_return_documented:
                    findings.append({
                        "rule_code": "DC-DTO-010",
                        "description": (
                            f"Shipment {idx}: Return/reverse logistics detected. "
                            "Ensure these are genuinely downstream and not "
                            "upstream (which would be Category 4)."
                        ),
                        "severity": ComplianceSeverity.MEDIUM.value,
                        "shipment_index": idx,
                        "recommendation": (
                            "Classify return logistics carefully: customer-initiated "
                            "returns = Category 9; company-initiated recalls = Category 4."
                        ),
                    })

        logger.info(
            "Double-counting check: %d findings from %d shipments",
            len(findings),
            len(shipments),
        )

        return findings

    # ==========================================================================
    # Incoterm Boundary Classification
    # ==========================================================================

    def check_incoterm_boundary(
        self, shipments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify shipments into Category 4 vs Category 9 based on Incoterms.

        For downstream transportation (Category 9), the reporting company
        (seller) pays for transport after the sale. Incoterms where the
        seller arranges transport belong in Category 9.

        Args:
            shipments: List of shipment dictionaries, each containing
                an 'incoterm' field.

        Returns:
            Dictionary with:
                - cat_9_shipments: list of shipment indices in Category 9
                - cat_4_shipments: list of shipment indices in Category 4
                - ambiguous_shipments: list with no Incoterm specified
                - classification_summary: counts per category

        Example:
            >>> result = engine.check_incoterm_boundary(shipments)
            >>> result["classification_summary"]
            {'category_9': 8, 'category_4': 2, 'ambiguous': 1}
        """
        cat_9_indices: List[int] = []
        cat_4_indices: List[int] = []
        ambiguous_indices: List[int] = []
        details: List[Dict[str, Any]] = []

        for idx, shipment in enumerate(shipments):
            incoterm = shipment.get("incoterm", "").upper()

            if not incoterm:
                ambiguous_indices.append(idx)
                details.append({
                    "index": idx,
                    "incoterm": None,
                    "category": IncotermCategory.AMBIGUOUS.value,
                    "reason": "No Incoterm specified; cannot classify.",
                })
                continue

            classification = INCOTERM_CLASSIFICATION.get(
                incoterm, IncotermCategory.AMBIGUOUS
            )

            if classification == IncotermCategory.CATEGORY_9:
                cat_9_indices.append(idx)
            elif classification == IncotermCategory.CATEGORY_4:
                cat_4_indices.append(idx)
            else:
                ambiguous_indices.append(idx)

            details.append({
                "index": idx,
                "incoterm": incoterm,
                "category": classification.value,
                "reason": (
                    f"Incoterm {incoterm} -> {classification.value}"
                ),
            })

        result = {
            "cat_9_shipments": cat_9_indices,
            "cat_4_shipments": cat_4_indices,
            "ambiguous_shipments": ambiguous_indices,
            "classification_summary": {
                "category_9": len(cat_9_indices),
                "category_4": len(cat_4_indices),
                "ambiguous": len(ambiguous_indices),
            },
            "details": details,
        }

        logger.info(
            "Incoterm boundary: Cat 9=%d, Cat 4=%d, Ambiguous=%d (total=%d)",
            len(cat_9_indices),
            len(cat_4_indices),
            len(ambiguous_indices),
            len(shipments),
        )

        return result

    # ==========================================================================
    # Batch Compliance
    # ==========================================================================

    def check_batch_compliance(
        self, batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run compliance checks on a batch of calculation results.

        Each item in the batch is independently checked against all
        enabled frameworks. Errors in one item do not affect others.

        Args:
            batch: List of calculation result dictionaries.

        Returns:
            List of compliance results, one per batch item.

        Example:
            >>> batch_results = engine.check_batch_compliance(batch_data)
            >>> len(batch_results)
            10
        """
        results: List[Dict[str, Any]] = []

        for idx, item in enumerate(batch):
            try:
                item_results = self.check_all_frameworks(item)
                results.append({
                    "index": idx,
                    "status": "success",
                    "framework_results": item_results,
                })
            except Exception as e:
                logger.error(
                    "Batch compliance check failed for item %d: %s",
                    idx, str(e),
                )
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(e),
                    "framework_results": {},
                })

        logger.info(
            "Batch compliance check complete: %d items, %d errors",
            len(results),
            sum(1 for r in results if r["status"] == "error"),
        )

        return results

    # ==========================================================================
    # Required Disclosures
    # ==========================================================================

    def get_required_disclosures(
        self, framework: str
    ) -> List[str]:
        """
        Get required disclosures for a specific framework.

        Args:
            framework: Framework name (e.g., 'ghg_protocol', 'csrd_esrs').

        Returns:
            List of required disclosure field names.

        Example:
            >>> disclosures = engine.get_required_disclosures("ghg_protocol")
            >>> "total_co2e" in disclosures
            True
        """
        framework_key = framework.lower()
        return FRAMEWORK_REQUIRED_DISCLOSURES.get(framework_key, [])

    # ==========================================================================
    # Compliance Summary
    # ==========================================================================

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Generate compliance summary from the most recent check_all_frameworks call.

        Calculates a weighted average compliance score across all checked
        frameworks and aggregates findings by severity.

        Returns:
            Summary dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.

        Example:
            >>> summary = engine.get_compliance_summary()
            >>> summary["overall_score"]
            85.5
        """
        results = self._last_results

        if not results:
            return {
                "overall_score": 0.0,
                "overall_status": ComplianceStatus.FAIL.value,
                "frameworks_checked": 0,
                "framework_scores": {},
                "critical_findings": [],
                "high_findings": [],
                "medium_findings": [],
                "low_findings": [],
                "total_findings": 0,
                "recommendations": [],
            }

        # Calculate weighted overall score
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        framework_scores: Dict[str, Dict[str, Any]] = {}

        for framework_name, check_result in results.items():
            try:
                fw_enum = ComplianceFramework(framework_name)
                weight = FRAMEWORK_WEIGHTS.get(fw_enum, Decimal("1.00"))
            except ValueError:
                weight = Decimal("1.00")

            score = Decimal(str(check_result.get("score", 0)))
            weighted_sum += score * weight
            total_weight += weight

            framework_scores[framework_name] = {
                "score": float(score),
                "status": check_result.get("status", "fail"),
                "findings_count": len(check_result.get("findings", [])),
                "weight": float(weight),
                "passed": check_result.get("passed", 0),
                "failed": check_result.get("failed", 0),
                "warnings": check_result.get("warnings", 0),
            }

        overall_score = Decimal("0")
        if total_weight > 0:
            overall_score = (weighted_sum / total_weight).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )

        # Determine overall status
        if overall_score >= Decimal("95"):
            overall_status = ComplianceStatus.PASS
        elif overall_score >= Decimal("70"):
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.FAIL

        # Aggregate findings by severity
        critical_findings: List[Dict[str, Any]] = []
        high_findings: List[Dict[str, Any]] = []
        medium_findings: List[Dict[str, Any]] = []
        low_findings: List[Dict[str, Any]] = []

        all_recommendations: List[str] = []
        seen_recommendations: Set[str] = set()

        for framework_name, check_result in results.items():
            for finding in check_result.get("findings", []):
                finding_with_fw = {
                    "framework": framework_name,
                    **finding,
                }
                severity = finding.get("severity", "")
                if severity == ComplianceSeverity.CRITICAL.value:
                    critical_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.HIGH.value:
                    high_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.MEDIUM.value:
                    medium_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.LOW.value:
                    low_findings.append(finding_with_fw)

                rec = finding.get("recommendation")
                if rec and rec not in seen_recommendations:
                    all_recommendations.append(rec)
                    seen_recommendations.add(rec)

        if overall_score < Decimal("70"):
            priority_msg = (
                "PRIORITY: Overall compliance score is below 70%. "
                "Address critical and high-severity issues immediately."
            )
            if priority_msg not in seen_recommendations:
                all_recommendations.insert(0, priority_msg)

        total_findings = (
            len(critical_findings) + len(high_findings)
            + len(medium_findings) + len(low_findings)
        )

        return {
            "overall_score": float(overall_score),
            "overall_status": overall_status.value,
            "frameworks_checked": len(results),
            "framework_scores": framework_scores,
            "critical_findings": critical_findings,
            "high_findings": high_findings,
            "medium_findings": medium_findings,
            "low_findings": low_findings,
            "total_findings": total_findings,
            "recommendations": all_recommendations,
        }

    # ==========================================================================
    # Sub-Activity Compliance (9a-9d)
    # ==========================================================================

    def check_sub_activity_completeness(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate sub-activity (9a-9d) completeness for Category 9.

        Category 9 is composed of four sub-activities:
            9a: Outbound transportation -- post-sale transport
            9b: Outbound distribution -- warehouse / DC operations
            9c: Retail storage -- third-party retail energy
            9d: Last-mile delivery -- final delivery to consumer

        Complete disclosure requires addressing each applicable sub-activity.

        Args:
            data: Calculation result with sub-activity information.

        Returns:
            Dictionary with completeness assessment per sub-activity.

        Example:
            >>> result = engine.check_sub_activity_completeness(calc_data)
            >>> result["completeness_score"]
            75.0
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # 9a: Outbound transportation
        has_9a = bool(data.get("outbound_transport_emissions")) or any(
            item.get("sub_activity") == "9a"
            for item in data.get("shipment_results", [])
        )
        if has_9a:
            state.add_pass("SUB-9A", "Outbound transportation (9a) emissions documented")
        else:
            state.add_warning(
                "SUB-9A",
                "Outbound transportation (9a) not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Include outbound transportation emissions (post-sale transport "
                    "paid by the reporting company) in Category 9."
                ),
                regulation_reference="GHG Protocol Scope 3, Category 9 sub-activity 9a",
            )

        # 9b: Distribution / warehousing
        has_9b = bool(data.get("warehouse_emissions")) or any(
            item.get("sub_activity") == "9b"
            for item in data.get("warehouse_results", [])
        )
        if has_9b:
            state.add_pass("SUB-9B", "Distribution/warehousing (9b) emissions documented")
        else:
            applicable = data.get("uses_distribution_centers", True)
            if applicable:
                state.add_warning(
                    "SUB-9B",
                    "Distribution/warehousing (9b) not documented",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Include distribution center and warehouse emissions "
                        "(energy, refrigeration) in Category 9 sub-activity 9b."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Category 9 sub-activity 9b",
                )
            else:
                state.add_pass("SUB-9B", "Distribution/warehousing (9b) not applicable")

        # 9c: Retail storage
        has_9c = bool(data.get("retail_storage_emissions")) or any(
            item.get("sub_activity") == "9c"
            for item in data.get("shipment_results", [])
        )
        if has_9c:
            state.add_pass("SUB-9C", "Retail storage (9c) emissions documented")
        else:
            applicable = data.get("sells_through_retail", False)
            if applicable:
                state.add_warning(
                    "SUB-9C",
                    "Retail storage (9c) not documented but company sells through retail",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Include energy consumption attributable to product storage "
                        "at third-party retail locations in sub-activity 9c."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Category 9 sub-activity 9c",
                )
            else:
                state.add_pass("SUB-9C", "Retail storage (9c) not applicable")

        # 9d: Last-mile delivery
        has_9d = bool(data.get("last_mile_emissions")) or any(
            item.get("sub_activity") == "9d"
            for item in data.get("last_mile_results", [])
        )
        if has_9d:
            state.add_pass("SUB-9D", "Last-mile delivery (9d) emissions documented")
        else:
            applicable = data.get("has_last_mile_delivery", False)
            if applicable:
                state.add_warning(
                    "SUB-9D",
                    "Last-mile delivery (9d) not documented but company has direct-to-consumer channel",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Include last-mile delivery emissions (courier, parcel, "
                        "e-commerce deliveries) in sub-activity 9d."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Category 9 sub-activity 9d",
                )
            else:
                state.add_pass("SUB-9D", "Last-mile delivery (9d) not applicable")

        result = state.to_result()

        # Add completeness score
        applicable_count = 4
        documented_count = sum([has_9a, has_9b, has_9c, has_9d])
        completeness_score = (
            float(Decimal(str(documented_count)) / Decimal(str(applicable_count)) * Decimal("100"))
            if applicable_count > 0
            else 100.0
        )
        result["completeness_score"] = completeness_score
        result["sub_activities"] = {
            "9a_outbound_transport": has_9a,
            "9b_distribution_warehousing": has_9b,
            "9c_retail_storage": has_9c,
            "9d_last_mile_delivery": has_9d,
        }

        return result

    # ==========================================================================
    # Distribution Chain Validation
    # ==========================================================================

    def validate_distribution_chain(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the downstream distribution chain for completeness and consistency.

        Checks:
        - Origin-to-destination chain continuity
        - Mode consistency within multi-leg chains
        - Distance plausibility (mode vs distance)
        - Mass conservation (outbound mass = DC throughput = last-mile mass)
        - Temporal consistency (warehouse entry before exit)

        Args:
            data: Calculation data with shipments, warehouses, last_mile.

        Returns:
            Dictionary with validation findings and chain status.

        Example:
            >>> result = engine.validate_distribution_chain(calc_data)
            >>> result["chain_valid"]
            True
        """
        findings: List[Dict[str, Any]] = []

        shipments = data.get("shipment_results", []) or data.get("shipments", [])
        warehouses = data.get("warehouse_results", []) or data.get("warehouses", [])
        last_mile = data.get("last_mile_results", []) or data.get("last_mile", [])

        # Check 1: Distance plausibility by mode
        for idx, shipment in enumerate(shipments):
            mode = str(shipment.get("mode", "road")).lower()
            distance_km = Decimal(str(shipment.get("distance_km", "0")))

            if mode == "air" and distance_km < Decimal("100"):
                findings.append({
                    "type": "DISTANCE_PLAUSIBILITY",
                    "index": idx,
                    "message": (
                        f"Air freight shipment {idx} has distance {distance_km} km "
                        "which is unusually short for air transport."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": "Verify air freight distance; consider road transport for short distances.",
                })

            if mode == "sea" and distance_km < Decimal("50"):
                findings.append({
                    "type": "DISTANCE_PLAUSIBILITY",
                    "index": idx,
                    "message": (
                        f"Sea freight shipment {idx} has distance {distance_km} km "
                        "which is unusually short for maritime transport."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": "Verify sea freight distance; consider inland waterway or road.",
                })

            if mode in ("road", "last_mile") and distance_km > Decimal("5000"):
                findings.append({
                    "type": "DISTANCE_PLAUSIBILITY",
                    "index": idx,
                    "message": (
                        f"Road shipment {idx} has distance {distance_km} km "
                        "which is unusually long for road transport."
                    ),
                    "severity": ComplianceSeverity.LOW.value,
                    "recommendation": "Verify road distance; consider if intermodal transport applies.",
                })

        # Check 2: Warehouse storage duration
        for idx, wh in enumerate(warehouses):
            storage_days = Decimal(str(wh.get("storage_days", "0")))

            if storage_days > Decimal("365"):
                findings.append({
                    "type": "STORAGE_DURATION",
                    "index": idx,
                    "message": (
                        f"Warehouse {idx} has storage duration of {storage_days} days "
                        "which exceeds one year."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": "Verify storage duration; ensure multi-year storage is correctly attributed.",
                })

            if storage_days <= Decimal("0"):
                findings.append({
                    "type": "STORAGE_DURATION",
                    "index": idx,
                    "message": f"Warehouse {idx} has zero or negative storage days.",
                    "severity": ComplianceSeverity.LOW.value,
                    "recommendation": "Use cross-dock EF if products pass through without storage.",
                })

        # Check 3: Mass conservation (total outbound ~ total warehouse ~ total last-mile)
        total_shipment_mass = sum(
            Decimal(str(s.get("mass_tonnes", "0"))) for s in shipments
        )
        total_warehouse_mass = sum(
            Decimal(str(w.get("mass_tonnes", "0"))) for w in warehouses
        )
        total_lastmile_mass = sum(
            Decimal(str(l.get("mass_tonnes", "0"))) for l in last_mile
        )

        if total_shipment_mass > 0 and total_warehouse_mass > 0:
            ratio = total_warehouse_mass / total_shipment_mass
            if ratio < Decimal("0.5") or ratio > Decimal("2.0"):
                findings.append({
                    "type": "MASS_CONSERVATION",
                    "message": (
                        f"Outbound mass ({total_shipment_mass}t) and warehouse "
                        f"throughput ({total_warehouse_mass}t) differ significantly "
                        f"(ratio: {ratio:.2f})."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": "Verify mass consistency across distribution chain stages.",
                })

        # Check 4: Last-mile volume vs shipment volume
        if total_shipment_mass > 0 and total_lastmile_mass > 0:
            lm_ratio = total_lastmile_mass / total_shipment_mass
            if lm_ratio > Decimal("1.5"):
                findings.append({
                    "type": "MASS_CONSERVATION",
                    "message": (
                        f"Last-mile mass ({total_lastmile_mass}t) exceeds "
                        f"outbound shipment mass ({total_shipment_mass}t) by >50%."
                    ),
                    "severity": ComplianceSeverity.LOW.value,
                    "recommendation": "Verify last-mile volume does not double-count outbound transport.",
                })

        chain_valid = all(
            f.get("severity") != ComplianceSeverity.CRITICAL.value
            for f in findings
        )

        return {
            "chain_valid": chain_valid,
            "findings": findings,
            "finding_count": len(findings),
            "mass_summary": {
                "shipment_mass_tonnes": str(total_shipment_mass),
                "warehouse_mass_tonnes": str(total_warehouse_mass),
                "last_mile_mass_tonnes": str(total_lastmile_mass),
            },
        }

    # ==========================================================================
    # Data Quality Assessment
    # ==========================================================================

    def assess_data_quality(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess data quality across 5 dimensions per GHG Protocol.

        Dimensions:
        1. Representativeness -- how well data represents actual operations
        2. Completeness -- coverage of all downstream activities
        3. Temporal -- recency of emission factors and activity data
        4. Geographical -- regional specificity of EFs
        5. Technological -- specificity of vehicle/mode EFs

        DQI scoring: 1 (poor) to 5 (excellent) per dimension.

        Args:
            data: Calculation data with method, EF sources, etc.

        Returns:
            Dictionary with DQI scores per dimension and overall score.

        Example:
            >>> dqi = engine.assess_data_quality(calc_data)
            >>> dqi["overall_score"]
            3.4
        """
        method = str(data.get("method", "average_data")).lower()
        ef_source = str(data.get("ef_source", "")).lower()
        has_primary_data = data.get("has_primary_data", False)

        # Dimension 1: Representativeness
        if method == "supplier_specific" or has_primary_data:
            representativeness = Decimal("5.0")
        elif method == "distance_based":
            representativeness = Decimal("4.0")
        elif method == "average_data":
            representativeness = Decimal("2.5")
        elif method == "spend_based":
            representativeness = Decimal("1.5")
        else:
            representativeness = Decimal("2.0")

        # Dimension 2: Completeness
        sub_activities_covered = sum([
            bool(data.get("outbound_transport_emissions") or data.get("shipment_results")),
            bool(data.get("warehouse_emissions") or data.get("warehouse_results")),
            bool(data.get("retail_storage_emissions")),
            bool(data.get("last_mile_emissions") or data.get("last_mile_results")),
        ])
        if sub_activities_covered >= 4:
            completeness = Decimal("5.0")
        elif sub_activities_covered >= 3:
            completeness = Decimal("4.0")
        elif sub_activities_covered >= 2:
            completeness = Decimal("3.0")
        elif sub_activities_covered >= 1:
            completeness = Decimal("2.0")
        else:
            completeness = Decimal("1.0")

        # Dimension 3: Temporal
        reporting_year = data.get("reporting_year", 2024)
        ef_year = data.get("ef_year", 2024)
        year_gap = abs(int(reporting_year) - int(ef_year))
        if year_gap <= 1:
            temporal = Decimal("5.0")
        elif year_gap <= 2:
            temporal = Decimal("4.0")
        elif year_gap <= 3:
            temporal = Decimal("3.0")
        elif year_gap <= 5:
            temporal = Decimal("2.0")
        else:
            temporal = Decimal("1.0")

        # Dimension 4: Geographical
        if "region" in ef_source or "country" in ef_source or data.get("regional_efs"):
            geographical = Decimal("4.5")
        elif "defra" in ef_source or "epa" in ef_source:
            geographical = Decimal("3.5")
        elif "glec" in ef_source:
            geographical = Decimal("4.0")
        elif "eeio" in ef_source:
            geographical = Decimal("2.0")
        else:
            geographical = Decimal("2.5")

        # Dimension 5: Technological
        has_vehicle_type = data.get("vehicle_type_specified", False)
        has_fuel_type = data.get("fuel_type_specified", False)
        if has_vehicle_type and has_fuel_type:
            technological = Decimal("5.0")
        elif has_vehicle_type or has_fuel_type:
            technological = Decimal("4.0")
        elif method == "distance_based":
            technological = Decimal("3.0")
        elif method == "average_data":
            technological = Decimal("2.0")
        else:
            technological = Decimal("1.5")

        # Overall DQI (simple average)
        overall = (
            (representativeness + completeness + temporal + geographical + technological)
            / Decimal("5")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        return {
            "overall_score": float(overall),
            "dimensions": {
                "representativeness": float(representativeness),
                "completeness": float(completeness),
                "temporal": float(temporal),
                "geographical": float(geographical),
                "technological": float(technological),
            },
            "method": method,
            "sub_activities_covered": sub_activities_covered,
            "guidance": self._dqi_guidance(overall),
        }

    @staticmethod
    def _dqi_guidance(score: Decimal) -> str:
        """Return guidance text based on DQI score."""
        if score >= Decimal("4.5"):
            return "Excellent data quality. Suitable for all reporting frameworks."
        elif score >= Decimal("3.5"):
            return "Good data quality. Meets GHG Protocol minimum requirements."
        elif score >= Decimal("2.5"):
            return "Moderate data quality. Consider upgrading to distance-based or supplier-specific methods."
        elif score >= Decimal("1.5"):
            return "Below average. Prioritize data quality improvement for Category 9."
        else:
            return "Poor data quality. Significant improvement required before external reporting."

    # ==========================================================================
    # Method Hierarchy Validation
    # ==========================================================================

    def validate_method_hierarchy(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that the calculation method follows GHG Protocol hierarchy.

        GHG Protocol method hierarchy (most to least preferred):
        1. Supplier-specific (primary carrier data)
        2. Distance-based (tonne-km with mode-specific EFs)
        3. Average-data (industry averages by channel)
        4. Spend-based (EEIO factors per USD)

        Args:
            data: Calculation data with method information.

        Returns:
            Dictionary with hierarchy assessment and improvement recommendations.
        """
        method = str(data.get("method", "average_data")).lower()

        hierarchy_rank = {
            "supplier_specific": 1,
            "distance_based": 2,
            "distance-based": 2,
            "average_data": 3,
            "average-data": 3,
            "spend_based": 4,
            "spend-based": 4,
        }

        rank = hierarchy_rank.get(method, 3)

        # Check percentage of emissions by method
        by_method = data.get("by_method", {})
        total_co2e = Decimal(str(data.get("total_co2e", "0")))

        method_percentages: Dict[str, float] = {}
        if total_co2e > 0:
            for m, emissions in by_method.items():
                pct = float(
                    (Decimal(str(emissions)) / total_co2e * Decimal("100")).quantize(
                        _QUANT_2DP, rounding=ROUNDING
                    )
                )
                method_percentages[m] = pct

        spend_pct = method_percentages.get("spend_based", 0.0) + method_percentages.get(
            "spend-based", 0.0
        )

        recommendations: List[str] = []
        status = "optimal"

        if rank >= 4:
            status = "suboptimal"
            recommendations.append(
                "Spend-based is the least preferred method. Upgrade to "
                "distance-based by collecting shipment distance and weight data."
            )
        elif rank >= 3:
            status = "acceptable"
            recommendations.append(
                "Average-data method is acceptable but not optimal. "
                "Consider collecting distance data per shipment."
            )

        if spend_pct > 80.0:
            status = "suboptimal"
            recommendations.append(
                f"Spend-based method accounts for {spend_pct:.1f}% of emissions. "
                "GHG Protocol recommends reducing spend-based reliance below 80%."
            )

        return {
            "method": method,
            "hierarchy_rank": rank,
            "status": status,
            "method_percentages": method_percentages,
            "spend_based_percentage": spend_pct,
            "recommendations": recommendations,
        }

    # ==========================================================================
    # Emission Factor Source Validation
    # ==========================================================================

    def validate_ef_sources(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate emission factor sources for currency, version, and suitability.

        Checks:
        - EF source is from a recognized database (DEFRA, EPA, GLEC, ICAO, IMO)
        - EF version year is within acceptable range
        - Mode-specific EFs used (not generic averages)
        - Regional EFs where available
        - WTT EFs included for WTW scope

        Args:
            data: Calculation data with EF source metadata.

        Returns:
            Dictionary with EF source validation results.
        """
        findings: List[Dict[str, Any]] = []
        recognized_sources = {
            "defra", "defra_glec", "epa", "glec", "icao", "imo",
            "ecoinvent", "eeio", "supplier_specific", "industry_average",
        }

        all_results = (
            data.get("shipment_results", [])
            + data.get("warehouse_results", [])
            + data.get("last_mile_results", [])
        )

        sources_used: Set[str] = set()
        for result in all_results:
            source = str(result.get("ef_source", "unknown")).lower()
            sources_used.add(source)

            if source not in recognized_sources and source != "unknown":
                findings.append({
                    "type": "UNRECOGNIZED_SOURCE",
                    "index": result.get("index"),
                    "source": source,
                    "message": f"Unrecognized EF source: {source}",
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "recommendation": (
                        "Use recognized EF sources: DEFRA, EPA SmartWay, GLEC, "
                        "ICAO, IMO, or supplier-specific data."
                    ),
                })

        # Check for all-EEIO usage
        if sources_used == {"eeio"}:
            findings.append({
                "type": "ALL_EEIO",
                "message": "All emission factors are EEIO (spend-based). Consider upgrading.",
                "severity": ComplianceSeverity.MEDIUM.value,
                "recommendation": (
                    "EEIO factors have high uncertainty. Upgrade to "
                    "mode-specific EFs (DEFRA, GLEC) where possible."
                ),
            })

        # Check EF version year freshness (EFs should be within 3 years)
        current_year = datetime.now(timezone.utc).year
        max_ef_age_years = 3
        for result in all_results:
            ef_year = result.get("ef_year") or result.get("ef_version_year")
            if ef_year is not None:
                try:
                    ef_year_int = int(ef_year)
                    age = current_year - ef_year_int
                    if age > max_ef_age_years:
                        findings.append({
                            "type": "STALE_EF",
                            "index": result.get("index"),
                            "ef_year": ef_year_int,
                            "age_years": age,
                            "message": (
                                f"Emission factor from {ef_year_int} is {age} years old "
                                f"(max recommended: {max_ef_age_years})."
                            ),
                            "severity": ComplianceSeverity.LOW.value,
                            "recommendation": (
                                f"Update to EFs from {current_year - 1} or later. "
                                "DEFRA publishes annual updates; GLEC publishes v3.1+."
                            ),
                        })
                except (ValueError, TypeError):
                    pass

        # Check mode-specific EF usage (generic EFs should be flagged)
        for result in all_results:
            mode = str(result.get("transport_mode", "")).lower()
            ef_specificity = str(result.get("ef_specificity", "")).lower()
            if mode and ef_specificity in ("generic", "all_modes", "average"):
                findings.append({
                    "type": "GENERIC_EF",
                    "index": result.get("index"),
                    "mode": mode,
                    "message": (
                        f"Generic EF used for {mode} transport. "
                        "Mode-specific EFs provide higher accuracy."
                    ),
                    "severity": ComplianceSeverity.LOW.value,
                    "recommendation": (
                        f"Use {mode}-specific EF from DEFRA/GLEC instead of generic average."
                    ),
                })

        # Check WTT inclusion for WTW reporting scope
        emission_scope = str(data.get("emission_scope", "ttw")).lower()
        if emission_scope in ("wtw", "well_to_wheel", "well-to-wheel"):
            has_wtt = False
            for result in all_results:
                if result.get("includes_wtt") is True or result.get("wtt_included") is True:
                    has_wtt = True
                    break
            if not has_wtt and all_results:
                findings.append({
                    "type": "WTT_MISSING",
                    "message": (
                        "WTW emission scope declared but no WTT (well-to-tank) "
                        "factors detected in results."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "recommendation": (
                        "ISO 14083 requires WTW reporting. Include WTT upstream "
                        "factors from DEFRA or GLEC Framework v3."
                    ),
                })

        # Check regional EF preference
        for result in all_results:
            region = result.get("region") or result.get("country")
            ef_region = result.get("ef_region") or result.get("ef_country")
            if region and ef_region:
                if str(region).lower() != str(ef_region).lower():
                    findings.append({
                        "type": "REGION_MISMATCH",
                        "index": result.get("index"),
                        "data_region": region,
                        "ef_region": ef_region,
                        "message": (
                            f"Data region ({region}) does not match EF region ({ef_region}). "
                            "Regional EFs improve accuracy."
                        ),
                        "severity": ComplianceSeverity.LOW.value,
                        "recommendation": (
                            f"Use EFs specific to {region} where available."
                        ),
                    })

        return {
            "sources_used": list(sources_used),
            "source_count": len(sources_used),
            "findings": findings,
            "finding_count": len(findings),
            "all_recognized": all(
                str(r.get("ef_source", "unknown")).lower() in recognized_sources
                for r in all_results
            ) if all_results else True,
            "has_stale_efs": any(f["type"] == "STALE_EF" for f in findings),
            "has_wtt_gap": any(f["type"] == "WTT_MISSING" for f in findings),
        }

    # ==========================================================================
    # Reporting Boundary Documentation Check
    # ==========================================================================

    def check_reporting_boundary(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that the reporting boundary is properly documented.

        Category 9 boundary requires clear definition of:
        - Which downstream activities are included (9a-9d)
        - Incoterm-based cutoff between Cat 4 and Cat 9
        - Geographic scope of downstream distribution
        - Exclusions and their justification
        - Consolidation approach (equity share / operational control / financial control)

        Args:
            data: Calculation data with boundary metadata.

        Returns:
            Dictionary with boundary assessment.

        Example:
            >>> result = engine.check_reporting_boundary(calc_data)
            >>> result["boundary_documented"]
            True
        """
        findings: List[Dict[str, Any]] = []

        # Check consolidation approach
        consolidation = data.get("consolidation_approach")
        if not consolidation:
            findings.append({
                "type": "CONSOLIDATION",
                "message": "Consolidation approach not specified (equity share, operational control, or financial control).",
                "severity": ComplianceSeverity.MEDIUM.value,
                "recommendation": "Specify consolidation approach per GHG Protocol Corporate Standard, Chapter 3.",
            })

        # Check geographic scope
        geographic_scope = data.get("geographic_scope")
        if not geographic_scope:
            findings.append({
                "type": "GEOGRAPHIC_SCOPE",
                "message": "Geographic scope of downstream distribution not documented.",
                "severity": ComplianceSeverity.LOW.value,
                "recommendation": "Document the geographic scope (countries/regions) covered by Category 9 reporting.",
            })

        # Check exclusions
        exclusions = data.get("exclusions")
        if exclusions is None:
            findings.append({
                "type": "EXCLUSIONS",
                "message": "No exclusions documented. If all downstream activities are included, state so explicitly.",
                "severity": ComplianceSeverity.LOW.value,
                "recommendation": "Document any excluded downstream activities with justification per GHG Protocol materiality threshold.",
            })

        # Check organizational boundary
        org_boundary = data.get("organizational_boundary")
        if not org_boundary:
            findings.append({
                "type": "ORG_BOUNDARY",
                "message": "Organizational boundary not specified.",
                "severity": ComplianceSeverity.MEDIUM.value,
                "recommendation": "Define which legal entities and operations are within the reporting boundary.",
            })

        # Check temporal boundary
        reporting_period = data.get("reporting_period")
        if not reporting_period:
            findings.append({
                "type": "TEMPORAL_BOUNDARY",
                "message": "Reporting period not specified.",
                "severity": ComplianceSeverity.MEDIUM.value,
                "recommendation": "Specify the reporting period (calendar year, fiscal year, or custom period).",
            })

        # Check distribution channels in scope
        channels_documented = data.get("channels_in_scope")
        if not channels_documented:
            findings.append({
                "type": "CHANNELS",
                "message": "Distribution channels in scope not documented.",
                "severity": ComplianceSeverity.LOW.value,
                "recommendation": "Document which distribution channels are in scope (DTC, wholesale, retail, e-commerce, 3PL).",
            })

        boundary_documented = len(findings) == 0

        return {
            "boundary_documented": boundary_documented,
            "findings": findings,
            "finding_count": len(findings),
            "boundary_elements": {
                "consolidation_approach": consolidation or "not_specified",
                "geographic_scope": geographic_scope or "not_specified",
                "organizational_boundary": org_boundary or "not_specified",
                "reporting_period": reporting_period or "not_specified",
                "exclusions_documented": exclusions is not None,
                "channels_documented": channels_documented is not None,
            },
        }

    # ==========================================================================
    # Compliance Report Generation
    # ==========================================================================

    def generate_compliance_report(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance report combining all checks.

        Runs all framework checks, double-counting validation, sub-activity
        completeness, distribution chain validation, data quality assessment,
        method hierarchy validation, and EF source validation into a single
        unified report.

        Args:
            data: Complete calculation data.

        Returns:
            Comprehensive compliance report dictionary.

        Example:
            >>> report = engine.generate_compliance_report(calc_data)
            >>> report["overall_grade"]
            'B+'
        """
        report: Dict[str, Any] = {
            "report_id": f"dto-compliance-{int(time.time())}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": "GL-MRV-S3-009",
            "agent_component": "AGENT-MRV-022",
            "engine_version": ENGINE_VERSION,
        }

        # Run all framework checks
        framework_results = self.check_all_frameworks(data)
        report["framework_results"] = framework_results

        # Get compliance summary
        summary = self.get_compliance_summary()
        report["summary"] = summary

        # Double-counting check
        dc_findings = self.check_double_counting(data)
        report["double_counting"] = {
            "findings": dc_findings,
            "finding_count": len(dc_findings),
            "has_critical": any(
                f.get("severity") == ComplianceSeverity.CRITICAL.value
                for f in dc_findings
            ),
        }

        # Sub-activity completeness
        sub_activity = self.check_sub_activity_completeness(data)
        report["sub_activity_completeness"] = sub_activity

        # Distribution chain validation
        chain_validation = self.validate_distribution_chain(data)
        report["chain_validation"] = chain_validation

        # Data quality assessment
        dqi = self.assess_data_quality(data)
        report["data_quality"] = dqi

        # Method hierarchy validation
        method_hierarchy = self.validate_method_hierarchy(data)
        report["method_hierarchy"] = method_hierarchy

        # EF source validation
        ef_validation = self.validate_ef_sources(data)
        report["ef_sources"] = ef_validation

        # Reporting boundary
        boundary = self.check_reporting_boundary(data)
        report["boundary"] = boundary

        # Overall grade
        overall_score = Decimal(str(summary.get("overall_score", 0)))
        report["overall_grade"] = self._score_to_grade(overall_score)

        return report

    @staticmethod
    def _score_to_grade(score: Decimal) -> str:
        """Convert numeric score to letter grade."""
        if score >= Decimal("95"):
            return "A+"
        elif score >= Decimal("90"):
            return "A"
        elif score >= Decimal("85"):
            return "A-"
        elif score >= Decimal("80"):
            return "B+"
        elif score >= Decimal("75"):
            return "B"
        elif score >= Decimal("70"):
            return "B-"
        elif score >= Decimal("65"):
            return "C+"
        elif score >= Decimal("60"):
            return "C"
        elif score >= Decimal("55"):
            return "C-"
        elif score >= Decimal("50"):
            return "D"
        else:
            return "F"

    # ==========================================================================
    # Engine Statistics
    # ==========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count, enabled frameworks,
            and last result status.

        Example:
            >>> stats = engine.get_engine_stats()
            >>> stats["engine_id"]
            'dto_compliance_checker_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": "GL-MRV-S3-009",
            "agent_component": "AGENT-MRV-022",
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._strict_mode,
            "last_result_frameworks": list(self._last_results.keys()),
            "supported_frameworks": [fw.value for fw in ComplianceFramework],
            "double_counting_rules": 10,
            "sub_activities": ["9a", "9b", "9c", "9d"],
        }


# ==============================================================================
# MODULE-LEVEL ACCESSORS (thread-safe)
# ==============================================================================


def get_compliance_checker() -> ComplianceCheckerEngine:
    """
    Get the ComplianceCheckerEngine singleton instance.

    Convenience function that delegates to the class-level get_instance().

    Returns:
        ComplianceCheckerEngine singleton.

    Example:
        >>> engine = get_compliance_checker()
        >>> engine.get_engine_stats()["engine_id"]
        'dto_compliance_checker_engine'
    """
    return ComplianceCheckerEngine.get_instance()


def reset_compliance_checker() -> None:
    """
    Reset the ComplianceCheckerEngine singleton instance.

    Used for testing or re-initialization with updated configuration.

    Example:
        >>> reset_compliance_checker()
        >>> engine = get_compliance_checker()  # Fresh instance
    """
    ComplianceCheckerEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "ComplianceFramework",
    "ComplianceStatus",
    "ComplianceSeverity",
    "TransportMode",
    "IncotermCategory",
    "DoubleCountingCategory",
    "DistributionSubActivity",
    "INCOTERM_CLASSIFICATION",
    "CAT_9_INCOTERMS",
    "CAT_4_INCOTERMS",
    "FRAMEWORK_WEIGHTS",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "ComplianceFinding",
    "FrameworkCheckState",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
    "reset_compliance_checker",
]
