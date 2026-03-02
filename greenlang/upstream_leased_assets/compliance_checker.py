# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-021 Engine 6

This module implements regulatory compliance checking for Upstream Leased Assets
emissions (GHG Protocol Scope 3 Category 8) against 7 regulatory frameworks.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 8 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change (lease classification disclosure)
4. CDP Climate Change Questionnaire (C6.5 Leased Assets)
5. SBTi (Science Based Targets initiative, materiality >1% of Scope 3)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 8-Specific Compliance Rules:
- Lease classification (operating vs finance) documentation
- Calculation methodology documentation (asset-specific, lessor, average, spend)
- Emission factor source identification with version/year
- Allocation method justification for multi-tenant assets
- WTT emissions boundary alignment with Category 3
- Scope 1/2 boundary confirmation (no overlap with lessee direct emissions)
- Data quality assessment per asset
- Base year recalculation on structural changes (new/terminated leases)
- Organizational boundary alignment with consolidation approach

Double-Counting Prevention Rules (10 rules: DC-ULA-001 through DC-ULA-010):
    DC-ULA-001: Exclude finance leases (-> Scope 1/2)
    DC-ULA-002: Exclude owned assets (-> Scope 1/2)
    DC-ULA-003: No overlap with Scope 2 (electricity already in S2)
    DC-ULA-004: No overlap with Scope 1 (gas/fuel already in S1)
    DC-ULA-005: No overlap with Cat 1 (purchased goods in leased buildings)
    DC-ULA-006: No overlap with Cat 2 (capital goods in leased buildings)
    DC-ULA-007: No overlap with Cat 3 (WTT already in Cat 3)
    DC-ULA-008: No overlap with Cat 5 (waste from leased buildings)
    DC-ULA-009: No overlap with Cat 13 (lessor assets -> downstream)
    DC-ULA-010: Sub-lease allocation (exclude sub-leased portions)

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-008
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_HUNDRED: Decimal = Decimal("100")

# Lease types that belong in Scope 1/2 under financial/equity share approach
FINANCE_LEASE_TYPES: Set[str] = {
    "finance_lease",
    "capital_lease",
    "finance",
    "capital",
    "ifrs16_right_of_use",
}

# Operating lease types that belong in Category 8
OPERATING_LEASE_TYPES: Set[str] = {
    "operating_lease",
    "operating",
    "short_term",
    "low_value",
}

# Asset categories supported
VALID_ASSET_CATEGORIES: Set[str] = {
    "building",
    "vehicle",
    "equipment",
    "it_asset",
}

# Building subtypes
VALID_BUILDING_TYPES: Set[str] = {
    "office",
    "retail",
    "warehouse",
    "industrial",
    "data_center",
    "hotel",
    "healthcare",
    "education",
}

# Vehicle subtypes
VALID_VEHICLE_TYPES: Set[str] = {
    "small_car",
    "medium_car",
    "large_car",
    "suv",
    "light_van",
    "heavy_van",
    "light_truck",
    "heavy_truck",
}

# Equipment subtypes
VALID_EQUIPMENT_TYPES: Set[str] = {
    "manufacturing",
    "construction",
    "generator",
    "agricultural",
    "mining",
    "hvac",
}

# IT asset subtypes
VALID_IT_TYPES: Set[str] = {
    "server",
    "network_switch",
    "storage",
    "desktop",
    "laptop",
    "printer",
    "copier",
}

# Consolidation approaches
VALID_CONSOLIDATION_APPROACHES: Set[str] = {
    "financial_control",
    "operational_control",
    "equity_share",
}

# Valid calculation methods for Category 8
VALID_CALCULATION_METHODS: Set[str] = {
    "asset_specific",
    "lessor_specific",
    "average_data",
    "spend_based",
}

# Energy source types
ENERGY_SOURCES: Set[str] = {
    "electricity",
    "natural_gas",
    "fuel_oil",
    "district_heating",
    "district_cooling",
    "lpg",
    "diesel",
    "steam",
}


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks (7 frameworks)."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class DoubleCountingCategory(str, Enum):
    """Scope/category that could overlap with Category 8."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_1 = "CATEGORY_1"
    CATEGORY_2 = "CATEGORY_2"
    CATEGORY_3 = "CATEGORY_3"
    CATEGORY_5 = "CATEGORY_5"
    CATEGORY_13 = "CATEGORY_13"


class BoundaryClassification(str, Enum):
    """Category boundary classification for a leased asset."""

    CATEGORY_8 = "CATEGORY_8"
    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_1 = "CATEGORY_1"
    CATEGORY_2 = "CATEGORY_2"
    CATEGORY_3 = "CATEGORY_3"
    CATEGORY_5 = "CATEGORY_5"
    CATEGORY_13 = "CATEGORY_13"
    EXCLUDED = "EXCLUDED"


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
        severity: ComplianceSeverity = ComplianceSeverity.MINOR,
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
            (max_points - penalty_points) / max_points * _HUNDRED
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if score < _ZERO:
            score = Decimal("0.00")
        if score > _HUNDRED:
            score = Decimal("100.00")

        return score

    def compute_status(self) -> ComplianceStatus:
        """Compute overall status from findings."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 0:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert accumulated state to a result dictionary."""
        findings_dicts = [
            {
                "rule_code": f.rule_code,
                "description": f.description,
                "severity": f.severity.value,
                "framework": f.framework,
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
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "total_checks": self.total_checks,
        }


# ==============================================================================
# SCORING WEIGHTS PER FRAMEWORK
# ==============================================================================

FRAMEWORK_WEIGHTS: Dict[ComplianceFramework, Decimal] = {
    ComplianceFramework.GHG_PROTOCOL: Decimal("1.00"),
    ComplianceFramework.ISO_14064: Decimal("0.85"),
    ComplianceFramework.CSRD_ESRS: Decimal("0.90"),
    ComplianceFramework.CDP: Decimal("0.85"),
    ComplianceFramework.SBTI: Decimal("0.80"),
    ComplianceFramework.SB_253: Decimal("0.75"),
    ComplianceFramework.GRI: Decimal("0.70"),
}

# ==============================================================================
# FRAMEWORK REQUIRED DISCLOSURES
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "calculation_method",
        "ef_sources",
        "lease_classification",
        "allocation_method",
        "wtt_included",
        "scope_boundary",
        "data_quality_score",
        "base_year",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
        "reporting_period",
        "organizational_boundary",
        "quantification_approach",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "methodology",
        "targets",
        "asset_breakdown",
        "reduction_actions",
        "lease_policy",
        "intensity_per_sqm",
        "year_over_year_change",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "asset_breakdown",
        "total_leased_area",
        "verification_status",
        "data_coverage",
        "energy_breakdown",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "progress_tracking",
        "materiality_assessment",
        "reduction_initiatives",
        "base_year_policy",
    ],
    ComplianceFramework.SB_253: [
        "total_co2e",
        "methodology",
        "assurance_opinion",
        "materiality_assessment",
        "data_retention",
        "provenance_hash",
    ],
    ComplianceFramework.GRI: [
        "total_co2e",
        "gases_included",
        "base_year",
        "standards_used",
        "ef_sources",
        "consolidation_approach",
        "intensity_ratios",
    ],
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Upstream Leased Assets emissions (Category 8).

    Validates calculation results against 7 regulatory frameworks with
    Category 8-specific rules including lease classification, allocation
    method justification, asset-level data quality, double-counting
    prevention (10 rules), and category boundary enforcement.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _enabled_frameworks: List of enabled compliance frameworks
        _check_count: Running count of compliance checks performed

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_all_frameworks(calc_result)
        >>> summary = engine.get_compliance_summary(results)
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "ComplianceCheckerEngine":
        """Create or return singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with configuration."""
        if self._initialized:
            return
        self._initialized = True

        self._enabled_frameworks: List[str] = [
            "ghg_protocol",
            "iso_14064",
            "csrd_esrs",
            "cdp",
            "sbti",
            "sb_253",
            "gri",
        ]
        self._check_count: int = 0
        self._strict_mode: bool = True
        self._double_counting_check: bool = True

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d, strict_mode=%s, double_counting_check=%s",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
            self._strict_mode,
            self._double_counting_check,
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
            Protected by the class-level lock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("ComplianceCheckerEngine singleton reset")

    # ==========================================================================
    # Main Entry Points
    # ==========================================================================

    def check_all_frameworks(
        self,
        result: dict,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked. An optional
        frameworks list can be passed to restrict which frameworks
        are checked.

        Args:
            result: Calculation result dictionary containing total_co2e,
                asset_breakdown, lease_classification, method, ef_sources, etc.
            frameworks: Optional list of framework names to check. If None,
                all enabled frameworks from config are checked.

        Returns:
            Dictionary mapping framework name string to compliance result dict.

        Example:
            >>> all_results = engine.check_all_frameworks(calc_result)
            >>> ghg_result = all_results.get("ghg_protocol")
            >>> ghg_result["status"]
            'pass'
        """
        start_time = time.monotonic()
        logger.info("Running compliance checks for all enabled frameworks")

        all_results: Dict[str, Any] = {}

        framework_dispatch: Dict[str, Tuple[ComplianceFramework, str]] = {
            "GHG_PROTOCOL_SCOPE3": (
                ComplianceFramework.GHG_PROTOCOL,
                "check_ghg_protocol",
            ),
            "ISO_14064": (
                ComplianceFramework.ISO_14064,
                "check_iso_14064",
            ),
            "CSRD_ESRS_E1": (
                ComplianceFramework.CSRD_ESRS,
                "check_csrd_esrs",
            ),
            "CDP": (ComplianceFramework.CDP, "check_cdp"),
            "SBTI": (ComplianceFramework.SBTI, "check_sbti"),
            "SB_253": (ComplianceFramework.SB_253, "check_sb_253"),
            "GRI_305": (ComplianceFramework.GRI, "check_gri_305"),
            "ghg_protocol": (
                ComplianceFramework.GHG_PROTOCOL,
                "check_ghg_protocol",
            ),
            "iso_14064": (
                ComplianceFramework.ISO_14064,
                "check_iso_14064",
            ),
            "csrd_esrs": (
                ComplianceFramework.CSRD_ESRS,
                "check_csrd_esrs",
            ),
            "cdp": (ComplianceFramework.CDP, "check_cdp"),
            "sbti": (ComplianceFramework.SBTI, "check_sbti"),
            "sb_253": (ComplianceFramework.SB_253, "check_sb_253"),
            "gri": (ComplianceFramework.GRI, "check_gri_305"),
        }

        target_frameworks = frameworks if frameworks else self._enabled_frameworks

        for framework_name in target_frameworks:
            dispatch = framework_dispatch.get(framework_name)
            if dispatch is None:
                logger.warning(
                    "Unknown framework in config: '%s', skipping",
                    framework_name,
                )
                continue

            framework_enum, method_name = dispatch
            try:
                check_method = getattr(self, method_name)
                check_result = check_method(result)
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
                            "description": (
                                f"Compliance check failed: {str(e)}"
                            ),
                            "severity": ComplianceSeverity.CRITICAL.value,
                            "status": ComplianceStatus.FAIL.value,
                        }
                    ],
                    "recommendations": [
                        "Resolve the compliance check error and rerun."
                    ],
                }

        duration = time.monotonic() - start_time
        self._check_count += 1

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.4fs",
            len(all_results),
            duration,
        )

        return all_results

    # ==========================================================================
    # Framework: GHG Protocol Scope 3 (Category 8)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 8).

        Category 8 specific checks:
            - GHG-ULA-001: Lease classification documented (operating vs finance)
            - GHG-ULA-002: Calculation methodology documented
            - GHG-ULA-003: Emission factor sources identified with version/year
            - GHG-ULA-004: Allocation method justified
            - GHG-ULA-005: WTT emissions included
            - GHG-ULA-006: Scope 1/2 boundary confirmed (no overlap)
            - GHG-ULA-007: Data quality assessed per asset
            - GHG-ULA-008: Base year recalculation triggered on structural changes
            - GHG-ULA-009: Organizational boundary aligned with consolidation approach

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for GHG Protocol.

        Example:
            >>> res = engine.check_ghg_protocol(calc_result)
            >>> res["framework"]
            'ghg_protocol'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # GHG-ULA-001: Lease classification documented (operating vs finance)
        lease_classification = (
            result.get("lease_classification")
            or result.get("lease_type")
        )
        if lease_classification:
            lc_lower = str(lease_classification).lower().replace("-", "_").replace(" ", "_")
            if lc_lower in OPERATING_LEASE_TYPES or lc_lower in FINANCE_LEASE_TYPES:
                state.add_pass(
                    "GHG-ULA-001",
                    f"Lease classification documented: {lease_classification}",
                )
            else:
                state.add_warning(
                    "GHG-ULA-001",
                    f"Lease classification '{lease_classification}' is non-standard",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Classify each lease as operating or finance per IFRS 16 / "
                        "ASC 842. Under financial control approach, operating leases "
                        "go to Category 8; finance leases go to Scope 1/2."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )
        else:
            # Check assets for per-asset classification
            assets = result.get("assets") or result.get("asset_breakdown") or []
            if isinstance(assets, list) and len(assets) > 0:
                classified_count = sum(
                    1 for a in assets
                    if isinstance(a, dict) and (
                        a.get("lease_type") or a.get("lease_classification")
                    )
                )
                if classified_count == len(assets):
                    state.add_pass(
                        "GHG-ULA-001",
                        f"Lease classification documented per asset ({classified_count} assets)",
                    )
                elif classified_count > 0:
                    state.add_warning(
                        "GHG-ULA-001",
                        f"Lease classification partial: {classified_count}/{len(assets)} assets classified",
                        ComplianceSeverity.MAJOR,
                        recommendation=(
                            "Classify ALL leased assets as operating or finance. "
                            "Only operating leases belong in Category 8 under "
                            "financial control approach."
                        ),
                        regulation_reference="GHG Protocol Scope 3, Ch 8",
                    )
                else:
                    state.add_fail(
                        "GHG-ULA-001",
                        "Lease classification not documented for any asset",
                        ComplianceSeverity.CRITICAL,
                        recommendation=(
                            "Document lease classification (operating vs finance) for "
                            "each leased asset. This determines whether emissions "
                            "belong in Category 8 or Scope 1/2."
                        ),
                        regulation_reference="GHG Protocol Scope 3, Ch 8",
                    )
            else:
                state.add_fail(
                    "GHG-ULA-001",
                    "Lease classification not documented",
                    ComplianceSeverity.CRITICAL,
                    recommendation=(
                        "Document lease classification (operating vs finance) for all "
                        "leased assets. Under financial control approach, operating "
                        "leases report in Category 8."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )

        # GHG-ULA-002: Calculation methodology documented
        method = result.get("method") or result.get("calculation_method")
        if method:
            method_lower = str(method).lower().replace("-", "_").replace(" ", "_")
            if method_lower in VALID_CALCULATION_METHODS:
                state.add_pass(
                    "GHG-ULA-002",
                    f"Calculation methodology documented: {method}",
                )
            else:
                state.add_warning(
                    "GHG-ULA-002",
                    f"Calculation method '{method}' is non-standard",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Use one of the GHG Protocol recommended methods: "
                        "asset-specific, lessor-specific, average-data, or spend-based."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Table 8.1",
                )
        else:
            state.add_fail(
                "GHG-ULA-002",
                "Calculation methodology not documented",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Document the calculation method used: "
                    "asset-specific (metered data), lessor-specific (landlord data), "
                    "average-data (benchmark EUI), or spend-based (EEIO)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 8.1",
            )

        # GHG-ULA-003: Emission factor sources identified with version/year
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            if isinstance(ef_sources, dict):
                has_version = any(
                    "version" in str(v).lower() or "year" in str(v).lower() or "20" in str(v)
                    for v in ef_sources.values()
                )
                if has_version:
                    state.add_pass(
                        "GHG-ULA-003",
                        "EF sources identified with version/year",
                    )
                else:
                    state.add_warning(
                        "GHG-ULA-003",
                        "EF sources present but version/year not explicitly identified",
                        ComplianceSeverity.MINOR,
                        recommendation=(
                            "Include publication year or version for each emission "
                            "factor source (e.g., 'eGRID 2023', 'DEFRA 2024', "
                            "'IEA 2024')."
                        ),
                        regulation_reference="GHG Protocol Scope 3, Ch 8",
                    )
            else:
                state.add_pass(
                    "GHG-ULA-003",
                    "Emission factor sources documented",
                )
        else:
            state.add_fail(
                "GHG-ULA-003",
                "Emission factor sources not documented",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Document all emission factor sources with version/year "
                    "(eGRID, DEFRA, IEA, EPA, EUI benchmarks, EEIO, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 8",
            )

        # GHG-ULA-004: Allocation method justified
        allocation_method = (
            result.get("allocation_method")
            or result.get("alloc_method")
        )
        allocation_justification = result.get("allocation_justification")
        if allocation_method:
            if allocation_justification:
                state.add_pass(
                    "GHG-ULA-004",
                    f"Allocation method '{allocation_method}' justified",
                )
            else:
                state.add_warning(
                    "GHG-ULA-004",
                    f"Allocation method '{allocation_method}' present but not justified",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Document the justification for the chosen allocation method "
                        "(e.g., floor area, headcount, revenue). Explain why it best "
                        "represents the company's share of leased asset emissions."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )
        else:
            # Check if single-tenant (no allocation needed)
            is_single_tenant = result.get("single_tenant", False)
            multi_tenant_count = result.get("multi_tenant_count", 0)
            if is_single_tenant or multi_tenant_count == 0:
                state.add_pass(
                    "GHG-ULA-004",
                    "Single-tenant assets; no allocation method required",
                )
            else:
                state.add_fail(
                    "GHG-ULA-004",
                    "Allocation method not documented for multi-tenant assets",
                    ComplianceSeverity.MAJOR,
                    recommendation=(
                        "Document the allocation method for multi-tenant leased "
                        "assets (floor area ratio, headcount ratio, revenue ratio, "
                        "or other justifiable basis)."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )

        # GHG-ULA-005: WTT emissions included
        wtt_included = result.get("wtt_included", False)
        wtt_co2e = result.get("wtt_co2e") or result.get("wtt_emissions")
        if wtt_included or wtt_co2e is not None:
            state.add_pass(
                "GHG-ULA-005",
                "WTT emissions included",
            )
        else:
            state.add_warning(
                "GHG-ULA-005",
                "WTT emissions not explicitly included",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Include well-to-tank (WTT) emissions for fuel and electricity "
                    "consumed in leased assets. If WTT is reported separately in "
                    "Category 3, document the boundary clearly."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 8 Guidance",
            )

        # GHG-ULA-006: Scope 1/2 boundary confirmed (no overlap)
        scope_boundary = (
            result.get("scope_boundary")
            or result.get("scope_boundary_confirmed")
            or result.get("no_scope12_overlap")
        )
        consolidation_approach = (
            result.get("consolidation_approach")
            or result.get("org_boundary")
        )
        if scope_boundary:
            state.add_pass(
                "GHG-ULA-006",
                "Scope 1/2 boundary confirmed (no overlap)",
            )
        elif consolidation_approach:
            ca_lower = str(consolidation_approach).lower().replace("-", "_").replace(" ", "_")
            if ca_lower in VALID_CONSOLIDATION_APPROACHES:
                state.add_warning(
                    "GHG-ULA-006",
                    f"Consolidation approach '{consolidation_approach}' documented but "
                    "Scope 1/2 boundary not explicitly confirmed",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Explicitly confirm that no emissions in Category 8 overlap "
                        "with Scope 1 (direct fuel) or Scope 2 (purchased electricity) "
                        "already reported by the lessee."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )
            else:
                state.add_warning(
                    "GHG-ULA-006",
                    "Consolidation approach is non-standard",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Use financial control, operational control, or equity share "
                        "as the consolidation approach."
                    ),
                    regulation_reference="GHG Protocol Corporate Standard, Ch 3",
                )
        else:
            state.add_fail(
                "GHG-ULA-006",
                "Scope 1/2 boundary not confirmed",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Confirm that Category 8 emissions do not overlap with Scope 1 "
                    "(direct fuel combustion) or Scope 2 (purchased electricity). "
                    "Under financial control approach, leased assets not under the "
                    "company's financial control belong in Category 8."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 8",
            )

        # GHG-ULA-007: Data quality assessed per asset
        dqi_score = (
            result.get("dqi_score")
            or result.get("data_quality_score")
            or result.get("data_quality")
        )
        if dqi_score is not None:
            state.add_pass("GHG-ULA-007", "Data quality assessed")
        else:
            # Check asset-level DQI
            assets = result.get("assets") or result.get("asset_results") or []
            if isinstance(assets, list) and len(assets) > 0:
                dqi_count = sum(
                    1 for a in assets
                    if isinstance(a, dict) and (
                        a.get("dqi_score") or a.get("data_quality")
                    )
                )
                if dqi_count > 0:
                    state.add_pass(
                        "GHG-ULA-007",
                        f"Data quality assessed for {dqi_count}/{len(assets)} assets",
                    )
                else:
                    state.add_warning(
                        "GHG-ULA-007",
                        "Data quality not assessed per asset",
                        ComplianceSeverity.MINOR,
                        recommendation=(
                            "Assess data quality for each leased asset across 5 "
                            "dimensions (representativeness, completeness, temporal, "
                            "geographical, technological)."
                        ),
                        regulation_reference="GHG Protocol Scope 3, Ch 8",
                    )
            else:
                state.add_warning(
                    "GHG-ULA-007",
                    "Data quality indicator (DQI) score not present",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Calculate and report DQI scores. Asset-specific data "
                        "yields higher DQI than average-data or spend-based."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 8",
                )

        # GHG-ULA-008: Base year recalculation triggered on structural changes
        base_year = result.get("base_year")
        base_year_policy = (
            result.get("base_year_recalculation_policy")
            or result.get("base_year_policy")
        )
        structural_changes = result.get("structural_changes")
        if base_year_policy:
            state.add_pass(
                "GHG-ULA-008",
                "Base year recalculation policy documented",
            )
        elif base_year:
            if structural_changes:
                state.add_fail(
                    "GHG-ULA-008",
                    "Structural changes detected but base year recalculation policy "
                    "not documented",
                    ComplianceSeverity.MAJOR,
                    recommendation=(
                        "Structural changes (new/terminated leases, acquisitions) "
                        "detected. Document base year recalculation policy and "
                        "determine if recalculation is triggered."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 5",
                )
            else:
                state.add_warning(
                    "GHG-ULA-008",
                    "Base year present but recalculation policy not documented",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Document base year recalculation policy. Significant "
                        "structural changes (new leases, terminated leases, "
                        "acquisitions) may trigger recalculation."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 5",
                )
        else:
            state.add_warning(
                "GHG-ULA-008",
                "Base year and recalculation policy not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document both the base year and recalculation policy."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 5",
            )

        # GHG-ULA-009: Organizational boundary aligned with consolidation approach
        if consolidation_approach:
            ca_lower = str(consolidation_approach).lower().replace("-", "_").replace(" ", "_")
            if ca_lower in VALID_CONSOLIDATION_APPROACHES:
                state.add_pass(
                    "GHG-ULA-009",
                    f"Organizational boundary aligned: {consolidation_approach}",
                )
            else:
                state.add_warning(
                    "GHG-ULA-009",
                    f"Non-standard consolidation approach: {consolidation_approach}",
                    ComplianceSeverity.MINOR,
                    recommendation=(
                        "Use financial control, operational control, or equity share. "
                        "The approach determines which leased assets belong in "
                        "Category 8 vs Scope 1/2."
                    ),
                    regulation_reference="GHG Protocol Corporate Standard, Ch 3",
                )
        else:
            state.add_warning(
                "GHG-ULA-009",
                "Organizational boundary / consolidation approach not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document the consolidation approach (financial control, "
                    "operational control, or equity share). This determines "
                    "which leased assets are in Category 8."
                ),
                regulation_reference="GHG Protocol Corporate Standard, Ch 3",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: ISO 14064
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with ISO 14064-1:2018 (Clause 5.2.4).

        Checks:
            - ISO-ULA-001: Total CO2e present
            - ISO-ULA-002: Uncertainty analysis present
            - ISO-ULA-003: Base year documented
            - ISO-ULA-004: Methodology described
            - ISO-ULA-005: Reporting period defined
            - ISO-ULA-006: Organizational boundary documented
            - ISO-ULA-007: Quantification approach documented

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for ISO 14064.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # ISO-ULA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("ISO-ULA-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate and report total CO2e emissions for "
                    "upstream leased assets (Category 8)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-ULA-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-ULA-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-ULA-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Perform and document uncertainty analysis. Leased asset data "
                    "often has high uncertainty due to reliance on lessor data, "
                    "benchmark EUIs, or spend-based estimates."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-ULA-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-ULA-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-ULA-003",
                "Base year not documented",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Document the base year for emissions comparison. Include "
                    "recalculation policy for lease portfolio changes."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-ULA-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-ULA-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-ULA-004",
                "Methodology not described",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, calculation approach "
                    "(asset-specific, lessor, average-data, spend-based)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-ULA-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
        )
        if reporting_period:
            state.add_pass("ISO-ULA-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-ULA-005",
                "Reporting period not specified",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Specify the reporting period (e.g., 2025, FY2025)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-ULA-006: Organizational boundary documented
        org_boundary = (
            result.get("organizational_boundary")
            or result.get("org_boundary")
            or result.get("consolidation_approach")
        )
        if org_boundary:
            state.add_pass("ISO-ULA-006", "Organizational boundary documented")
        else:
            state.add_warning(
                "ISO-ULA-006",
                "Organizational boundary not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document the organizational boundary approach "
                    "(equity share, financial control, or operational control). "
                    "This determines which leased assets are in Category 8."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-ULA-007: Quantification approach documented
        quant_approach = (
            result.get("quantification_approach")
            or result.get("data_sources")
        )
        if quant_approach:
            state.add_pass("ISO-ULA-007", "Quantification approach documented")
        else:
            state.add_warning(
                "ISO-ULA-007",
                "Quantification approach not documented",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Document whether primary data (metered energy), lessor data, "
                    "secondary data (benchmarks), or financial proxy (EEIO) "
                    "were used and the rationale."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 6",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd_esrs(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        Checks:
            - CSRD-ULA-001: Total CO2e by category
            - CSRD-ULA-002: Methodology description
            - CSRD-ULA-003: Targets documented
            - CSRD-ULA-004: Asset breakdown present
            - CSRD-ULA-005: Reduction actions described
            - CSRD-ULA-006: Lease policy and portfolio disclosed
            - CSRD-ULA-007: Intensity metrics present (per sqm)
            - CSRD-ULA-008: Year-over-year comparison

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for CSRD/ESRS.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # CSRD-ULA-001: Total emissions by category
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("CSRD-ULA-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Scope 3 Category 8 emissions. "
                    "CSRD requires disclosure of all material Scope 3 categories."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-ULA-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-ULA-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-ULA-002",
                "Methodology not described",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Describe the calculation methodology for upstream leased "
                    "asset emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-ULA-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-ULA-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-ULA-003",
                "Reduction targets not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document emission reduction targets for leased asset portfolio "
                    "(e.g., green lease clauses, energy efficiency retrofits, "
                    "transition to renewable energy, portfolio optimization)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-ULA-004: Asset breakdown present
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_type")
            or result.get("by_building_type")
        )
        if (
            asset_breakdown
            and isinstance(asset_breakdown, (dict, list))
            and len(asset_breakdown) > 0
        ):
            state.add_pass("CSRD-ULA-004", "Asset breakdown present")
        else:
            state.add_fail(
                "CSRD-ULA-004",
                "Asset type breakdown not provided",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Provide emissions breakdown by asset type "
                    "(building, vehicle, equipment, IT asset) and by "
                    "building subtype (office, retail, warehouse, etc.)."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-ULA-005: Actions described
        actions = result.get("actions") or result.get("reduction_actions")
        if actions:
            state.add_pass("CSRD-ULA-005", "Reduction actions described")
        else:
            state.add_warning(
                "CSRD-ULA-005",
                "Reduction actions not described",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Describe actions taken or planned to reduce leased asset "
                    "emissions (green lease clauses, energy audits, HVAC upgrades, "
                    "renewable energy procurement, portfolio consolidation)."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        # CSRD-ULA-006: Lease policy and portfolio disclosed
        lease_policy = (
            result.get("lease_policy")
            or result.get("leasing_policy")
        )
        portfolio_summary = (
            result.get("portfolio_summary")
            or result.get("total_leased_area")
            or result.get("asset_count")
        )
        if lease_policy and portfolio_summary is not None:
            state.add_pass(
                "CSRD-ULA-006",
                "Lease policy and portfolio disclosed",
            )
        elif portfolio_summary is not None:
            state.add_warning(
                "CSRD-ULA-006",
                "Portfolio data present but lease policy not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document the company's leasing policy and sustainability "
                    "criteria for leased assets."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )
        elif lease_policy:
            state.add_warning(
                "CSRD-ULA-006",
                "Lease policy documented but portfolio summary not provided",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Provide portfolio summary (total leased area, asset count, "
                    "geographic distribution)."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )
        else:
            state.add_warning(
                "CSRD-ULA-006",
                "Lease policy and portfolio not disclosed",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Disclose the company leasing policy and provide portfolio "
                    "summary for CSRD compliance."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-ULA-007: Intensity metrics present
        intensity = (
            result.get("intensity_per_sqm")
            or result.get("co2e_per_sqm")
            or result.get("emissions_intensity")
        )
        if intensity is not None:
            state.add_pass("CSRD-ULA-007", "Intensity metrics present")
        else:
            state.add_warning(
                "CSRD-ULA-007",
                "Emissions intensity per sqm not provided",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Calculate and report emissions intensity "
                    "(kgCO2e per sqm per year) for benchmarking."
                ),
                regulation_reference="ESRS E1-6, para 54",
            )

        # CSRD-ULA-008: Year-over-year comparison
        yoy_change = (
            result.get("year_over_year_change")
            or result.get("yoy_change")
            or result.get("trend")
        )
        if yoy_change is not None:
            state.add_pass("CSRD-ULA-008", "Year-over-year comparison present")
        else:
            state.add_warning(
                "CSRD-ULA-008",
                "Year-over-year comparison not present",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Include year-over-year comparison of leased asset "
                    "emissions to demonstrate trends and progress."
                ),
                regulation_reference="ESRS E1-6, para 52",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with CDP Climate Change Questionnaire (C6.5).

        Checks:
            - CDP-ULA-001: Total CO2e present
            - CDP-ULA-002: Asset type breakdown (CDP requires this)
            - CDP-ULA-003: Total leased area/asset count for intensity
            - CDP-ULA-004: Verification status
            - CDP-ULA-005: Data coverage documented
            - CDP-ULA-006: Energy breakdown by source

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for CDP.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # CDP-ULA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("CDP-ULA-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CDP-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 8 emissions to CDP. "
                    "Upstream leased assets is a standard CDP C6.5 disclosure."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-ULA-002: Asset type breakdown
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_type")
        )
        if (
            asset_breakdown
            and isinstance(asset_breakdown, (dict, list))
            and len(asset_breakdown) > 0
        ):
            state.add_pass(
                "CDP-ULA-002",
                "Asset type breakdown present",
            )
        else:
            state.add_fail(
                "CDP-ULA-002",
                "Asset type breakdown not provided",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Provide emissions breakdown by asset type "
                    "(building, vehicle, equipment, IT). CDP requires "
                    "detailed breakdown for Category 8."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-ULA-003: Leased area / asset count for intensity
        total_leased_area = (
            result.get("total_leased_area")
            or result.get("total_area_sqm")
        )
        asset_count = result.get("asset_count") or result.get("total_assets")
        if total_leased_area is not None or asset_count is not None:
            state.add_pass("CDP-ULA-003", "Leased area/asset count documented")
        else:
            state.add_warning(
                "CDP-ULA-003",
                "Total leased area or asset count not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Report total leased floor area (sqm) and/or asset count "
                    "for per-unit emissions intensity. CDP uses this for "
                    "benchmarking."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-ULA-004: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-ULA-004", "Verification status documented")
        else:
            state.add_warning(
                "CDP-ULA-004",
                "Verification status not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document whether Category 8 emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module C10.1",
            )

        # CDP-ULA-005: Data coverage
        data_coverage = (
            result.get("data_coverage")
            or result.get("coverage_percentage")
        )
        if data_coverage is not None:
            state.add_pass("CDP-ULA-005", "Data coverage documented")
        else:
            state.add_warning(
                "CDP-ULA-005",
                "Data coverage not documented",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Document the percentage of leased assets covered by "
                    "actual data vs estimated data."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-ULA-006: Energy breakdown by source
        energy_breakdown = (
            result.get("energy_breakdown")
            or result.get("by_energy_source")
        )
        if (
            energy_breakdown
            and isinstance(energy_breakdown, (dict, list))
            and len(energy_breakdown) > 0
        ):
            state.add_pass("CDP-ULA-006", "Energy breakdown by source present")
        else:
            state.add_warning(
                "CDP-ULA-006",
                "Energy breakdown by source not provided",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Provide energy breakdown by source (electricity, gas, "
                    "district heating, fuel oil, etc.) for leased assets."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: SBTi
    # ==========================================================================

    def check_sbti(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with Science Based Targets initiative.

        Checks:
            - SBTI-ULA-001: Total CO2e present
            - SBTI-ULA-002: Target coverage documented
            - SBTI-ULA-003: Progress tracking present
            - SBTI-ULA-004: Materiality assessment (> 1% of Scope 3)
            - SBTI-ULA-005: Reduction initiatives documented
            - SBTI-ULA-006: Base year recalculation policy

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for SBTi.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI)

        # SBTI-ULA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("SBTI-ULA-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate total Category 8 emissions for SBTi "
                    "target boundary assessment."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-ULA-002: Target coverage
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
        )
        if target_coverage is not None:
            state.add_pass("SBTI-ULA-002", "Target coverage documented")
        else:
            state.add_warning(
                "SBTI-ULA-002",
                "SBTi target coverage not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is "
                    "covered by SBTi targets (minimum 67% required). "
                    "If Category 8 is material, it must be in the boundary."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-ULA-003: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-ULA-003", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-ULA-003",
                "Progress tracking not present",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Track year-over-year emissions change for Category 8 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-ULA-004: Materiality assessment (> 1% of Scope 3)
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat8_decimal = self._safe_decimal(total_co2e)
                scope3_decimal = self._safe_decimal(total_scope3)
                if scope3_decimal > _ZERO:
                    cat8_pct = (
                        cat8_decimal / scope3_decimal * _HUNDRED
                    )
                    if cat8_pct >= _ONE:
                        state.add_pass(
                            "SBTI-ULA-004",
                            (
                                f"Category 8 is material "
                                f"({cat8_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                        )
                    else:
                        state.add_warning(
                            "SBTI-ULA-004",
                            (
                                f"Category 8 below materiality threshold "
                                f"({cat8_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                            ComplianceSeverity.INFO,
                            recommendation=(
                                "Category 8 may not need to be included "
                                "in SBTi target boundary if below 1%. "
                                "Continued monitoring is recommended."
                            ),
                        )
                else:
                    state.add_warning(
                        "SBTI-ULA-004",
                        "Total Scope 3 is zero; cannot assess materiality",
                        ComplianceSeverity.INFO,
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-ULA-004",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.INFO,
                )
        else:
            state.add_warning(
                "SBTI-ULA-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 8 "
                    "materiality against 1% threshold."
                ),
            )

        # SBTI-ULA-005: Reduction initiatives
        reduction_initiatives = (
            result.get("reduction_initiatives")
            or result.get("reduction_actions")
            or result.get("actions")
        )
        if reduction_initiatives:
            state.add_pass(
                "SBTI-ULA-005",
                "Reduction initiatives documented",
            )
        else:
            state.add_warning(
                "SBTI-ULA-005",
                "Reduction initiatives not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document reduction initiatives for leased assets "
                    "(green lease clauses, energy efficiency requirements, "
                    "renewable energy procurement, portfolio optimization)."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-ULA-006: Base year recalculation policy
        base_year_policy = (
            result.get("base_year_recalculation_policy")
            or result.get("base_year_policy")
        )
        base_year = result.get("base_year")
        if base_year_policy:
            state.add_pass(
                "SBTI-ULA-006",
                "Base year recalculation policy documented",
            )
        elif base_year:
            state.add_warning(
                "SBTI-ULA-006",
                "Base year present but recalculation policy not documented",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Document the base year recalculation policy. "
                    "Lease portfolio changes (new/terminated leases) "
                    "may trigger base year recalculation per SBTi criteria."
                ),
                regulation_reference="SBTi Criteria v5.1, C14",
            )
        else:
            state.add_warning(
                "SBTI-ULA-006",
                "Base year and recalculation policy not documented",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Document both the base year and recalculation policy."
                ),
                regulation_reference="SBTi Criteria v5.1, C14",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb_253(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Checks:
            - SB253-ULA-001: Total CO2e present
            - SB253-ULA-002: Methodology documented
            - SB253-ULA-003: Assurance opinion available
            - SB253-ULA-004: Materiality > 1% threshold
            - SB253-ULA-005: Data retention documentation
            - SB253-ULA-006: Audit trail / provenance

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for SB 253.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # SB253-ULA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("SB253-ULA-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 8 emissions for "
                    "SB 253 compliance."
                ),
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-ULA-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-ULA-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-ULA-002",
                "Methodology not documented",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-ULA-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-ULA-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-ULA-003",
                "Assurance opinion not available",
                ComplianceSeverity.MAJOR,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-ULA-004: Materiality > 1% threshold
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat8_decimal = self._safe_decimal(total_co2e)
                scope3_decimal = self._safe_decimal(total_scope3)
                if scope3_decimal > _ZERO:
                    cat8_pct = (
                        cat8_decimal / scope3_decimal * _HUNDRED
                    )
                    if cat8_pct > _ONE:
                        state.add_pass(
                            "SB253-ULA-004",
                            (
                                f"Category 8 exceeds 1% materiality "
                                f"({cat8_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                        )
                    else:
                        state.add_warning(
                            "SB253-ULA-004",
                            (
                                f"Category 8 below 1% materiality threshold "
                                f"({cat8_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                            ComplianceSeverity.INFO,
                            recommendation=(
                                "Category 8 is below 1% of total Scope 3. "
                                "Consider whether separate reporting is warranted."
                            ),
                        )
                else:
                    state.add_warning(
                        "SB253-ULA-004",
                        "Total Scope 3 is zero; cannot assess materiality",
                        ComplianceSeverity.INFO,
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-ULA-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.INFO,
                )
        else:
            state.add_warning(
                "SB253-ULA-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 8 materiality against 1% threshold."
                ),
            )

        # SB253-ULA-005: Data retention documentation
        data_retention = (
            result.get("data_retention_policy")
            or result.get("data_retention")
            or result.get("record_keeping")
        )
        if data_retention:
            state.add_pass("SB253-ULA-005", "Data retention documented")
        else:
            state.add_warning(
                "SB253-ULA-005",
                "Data retention policy not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document data retention period for emission "
                    "calculations and source data. SB 253 requires "
                    "data to be available for verification."
                ),
                regulation_reference="SB 253, Section 38532(c)",
            )

        # SB253-ULA-006: Audit trail / provenance
        provenance_hash = (
            result.get("provenance_hash")
            or result.get("audit_trail")
            or result.get("provenance")
        )
        if provenance_hash:
            state.add_pass("SB253-ULA-006", "Audit trail / provenance present")
        else:
            state.add_warning(
                "SB253-ULA-006",
                "Audit trail / provenance hash not present",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Include SHA-256 provenance hash for complete "
                    "audit trail. This supports third-party assurance."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        return state.to_dict()

    # ==========================================================================
    # Framework: GRI 305
    # ==========================================================================

    def check_gri_305(self, result: dict) -> Dict[str, Any]:
        """
        Check compliance with GRI 305 Emissions Standard.

        Checks:
            - GRI-ULA-001: Total CO2e in metric tonnes
            - GRI-ULA-002: Gases included documented
            - GRI-ULA-003: Base year present
            - GRI-ULA-004: Standards referenced
            - GRI-ULA-005: Emission factor sources
            - GRI-ULA-006: Consolidation approach
            - GRI-ULA-007: Intensity ratios

        Args:
            result: Calculation result dictionary.

        Returns:
            Compliance result dictionary for GRI.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GRI)

        # GRI-ULA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._safe_decimal(total_co2e) > _ZERO:
            state.add_pass("GRI-ULA-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-ULA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 8 emissions in metric tonnes "
                    "CO2e per GRI 305-3."
                ),
                regulation_reference="GRI 305-3",
            )

        # GRI-ULA-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-ULA-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-ULA-002",
                "Gases included not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, or CO2e aggregate). Leased asset "
                    "calculations typically report CO2e aggregate."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-ULA-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-ULA-003", "Base year present")
        else:
            state.add_warning(
                "GRI-ULA-003",
                "Base year not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting. "
                    "Consider lease portfolio changes when selecting base year."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-ULA-004: Standards referenced
        standards = (
            result.get("standards_used")
            or result.get("standards")
            or result.get("framework_references")
        )
        if standards:
            state.add_pass("GRI-ULA-004", "Standards referenced")
        else:
            state.add_warning(
                "GRI-ULA-004",
                "Standards used not referenced",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Reference the standards and methodologies used "
                    "(e.g., GHG Protocol Scope 3 Category 8, "
                    "eGRID, DEFRA, IEA, EUI benchmarks)."
                ),
                regulation_reference="GRI 305-3(e)",
            )

        # GRI-ULA-005: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-ULA-005", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-ULA-005",
                "Emission factor sources not documented",
                ComplianceSeverity.MINOR,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., eGRID 2023, DEFRA 2024, IEA 2024)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        # GRI-ULA-006: Consolidation approach
        consolidation = (
            result.get("consolidation_approach")
            or result.get("org_boundary")
            or result.get("boundary_approach")
        )
        if consolidation:
            state.add_pass("GRI-ULA-006", "Consolidation approach documented")
        else:
            state.add_warning(
                "GRI-ULA-006",
                "Consolidation approach not documented",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Document the consolidation approach "
                    "(equity share, financial control, or operational control)."
                ),
                regulation_reference="GRI 305-1(b)",
            )

        # GRI-ULA-007: Intensity ratios
        intensity = (
            result.get("intensity_ratios")
            or result.get("intensity_per_sqm")
            or result.get("co2e_per_sqm")
        )
        if intensity is not None:
            state.add_pass("GRI-ULA-007", "Intensity ratios present")
        else:
            state.add_warning(
                "GRI-ULA-007",
                "Emissions intensity ratios not provided",
                ComplianceSeverity.INFO,
                recommendation=(
                    "Calculate and report intensity ratios "
                    "(e.g., kgCO2e per sqm, tCO2e per leased asset) "
                    "for GRI 305-4 compliance."
                ),
                regulation_reference="GRI 305-4",
            )

        return state.to_dict()

    # ==========================================================================
    # Double-Counting Prevention (10 Rules)
    # ==========================================================================

    def check_double_counting(
        self, assets: list
    ) -> List[dict]:
        """
        Validate 10 double-counting prevention rules for upstream leased assets.

        Rules:
            DC-ULA-001: Exclude finance leases (-> Scope 1/2)
            DC-ULA-002: Exclude owned assets (-> Scope 1/2)
            DC-ULA-003: No overlap with Scope 2 (electricity already in S2)
            DC-ULA-004: No overlap with Scope 1 (gas/fuel already in S1)
            DC-ULA-005: No overlap with Cat 1 (purchased goods in leased buildings)
            DC-ULA-006: No overlap with Cat 2 (capital goods in leased buildings)
            DC-ULA-007: No overlap with Cat 3 (WTT already in Cat 3)
            DC-ULA-008: No overlap with Cat 5 (waste from leased buildings)
            DC-ULA-009: No overlap with Cat 13 (lessor assets -> downstream)
            DC-ULA-010: Sub-lease allocation (exclude sub-leased portions)

        Args:
            assets: List of asset dictionaries.

        Returns:
            List of finding dictionaries.
        """
        findings: List[dict] = []

        for idx, asset in enumerate(assets):
            self._check_dc_ula_001(idx, asset, findings)
            self._check_dc_ula_002(idx, asset, findings)
            self._check_dc_ula_003(idx, asset, findings)
            self._check_dc_ula_004(idx, asset, findings)
            self._check_dc_ula_005(idx, asset, findings)
            self._check_dc_ula_006(idx, asset, findings)
            self._check_dc_ula_007(idx, asset, findings)
            self._check_dc_ula_008(idx, asset, findings)
            self._check_dc_ula_009(idx, asset, findings)
            self._check_dc_ula_010(idx, asset, findings)

        logger.info(
            "Double-counting check: %d assets, %d findings",
            len(assets),
            len(findings),
        )

        return findings

    def _check_dc_ula_001(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-001: Exclude finance leases (-> Scope 1/2).

        Finance leases / capital leases under financial control approach
        should be reported in Scope 1/2, not Category 8.
        """
        lease_type = str(asset.get("lease_type", "")).lower().replace("-", "_").replace(" ", "_")
        if lease_type in FINANCE_LEASE_TYPES:
            findings.append({
                "rule_code": "DC-ULA-001",
                "description": (
                    f"Asset {idx}: Finance/capital lease detected "
                    f"(lease_type='{lease_type}'). Under financial control "
                    "approach, finance leases should be in Scope 1/2."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Exclude finance leases from Category 8. Under financial "
                    "control approach, the lessee has control and reports "
                    "directly in Scope 1/2."
                ),
            })

    def _check_dc_ula_002(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-002: Exclude owned assets (-> Scope 1/2).

        Owned assets should be in Scope 1/2, not Category 8.
        """
        ownership = str(asset.get("ownership", "")).lower()
        is_owned = asset.get("is_owned", False)
        if ownership in ("owned", "company_owned", "purchased") or is_owned:
            findings.append({
                "rule_code": "DC-ULA-002",
                "description": (
                    f"Asset {idx}: Owned asset detected (ownership='{ownership}'). "
                    "Owned assets belong in Scope 1/2, not Category 8."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Exclude owned assets from Category 8. Only leased assets "
                    "not under the reporting company's operational/financial "
                    "control belong in Category 8."
                ),
            })

    def _check_dc_ula_003(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-003: No overlap with Scope 2 (electricity already in S2).

        If the lessee's purchased electricity for a leased asset is already
        reported in Scope 2, it must not also appear in Category 8.
        """
        electricity_in_scope2 = asset.get("electricity_in_scope2", False)
        elec_co2e = self._safe_decimal(asset.get("electricity_co2e", "0"))

        if electricity_in_scope2 and elec_co2e > _ZERO:
            findings.append({
                "rule_code": "DC-ULA-003",
                "description": (
                    f"Asset {idx}: Electricity emissions already in Scope 2 "
                    f"({elec_co2e} kgCO2e). Must not also appear in Cat 8."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.SCOPE_2.value,
                "recommendation": (
                    "If the lessee directly purchases electricity for the "
                    "leased asset and reports it in Scope 2, exclude that "
                    "electricity from Category 8."
                ),
            })

    def _check_dc_ula_004(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-004: No overlap with Scope 1 (gas/fuel already in S1).

        If the lessee directly combusts fuel in a leased asset and reports
        it in Scope 1, it must not also appear in Category 8.
        """
        fuel_in_scope1 = asset.get("fuel_in_scope1", False)
        gas_in_scope1 = asset.get("gas_in_scope1", False)
        fuel_co2e = self._safe_decimal(asset.get("fuel_co2e", "0"))

        if (fuel_in_scope1 or gas_in_scope1) and fuel_co2e > _ZERO:
            findings.append({
                "rule_code": "DC-ULA-004",
                "description": (
                    f"Asset {idx}: Fuel/gas emissions already in Scope 1 "
                    f"({fuel_co2e} kgCO2e). Must not also appear in Cat 8."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "If the lessee directly combusts fuel (natural gas, "
                    "heating oil) in the leased asset and reports it in "
                    "Scope 1, exclude from Category 8."
                ),
            })

    def _check_dc_ula_005(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-005: No overlap with Cat 1 (purchased goods in leased buildings).

        Emissions from purchased goods and services used within leased
        buildings should be in Category 1, not Category 8.
        """
        includes_purchased_goods = asset.get("includes_purchased_goods", False)
        reported_in_cat1 = asset.get("reported_in_cat1", False)

        if includes_purchased_goods or reported_in_cat1:
            findings.append({
                "rule_code": "DC-ULA-005",
                "description": (
                    f"Asset {idx}: Purchased goods emissions detected in "
                    "Category 8 scope. These should be in Category 1."
                ),
                "severity": ComplianceSeverity.MAJOR.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.CATEGORY_1.value,
                "recommendation": (
                    "Emissions from purchased goods and services consumed "
                    "within leased buildings belong in Category 1, not "
                    "Category 8. Category 8 covers building operation "
                    "energy only."
                ),
            })

    def _check_dc_ula_006(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-006: No overlap with Cat 2 (capital goods in leased buildings).

        Emissions from capital goods installed in leased buildings should
        be in Category 2, not Category 8.
        """
        includes_capital_goods = asset.get("includes_capital_goods", False)
        reported_in_cat2 = asset.get("reported_in_cat2", False)

        if includes_capital_goods or reported_in_cat2:
            findings.append({
                "rule_code": "DC-ULA-006",
                "description": (
                    f"Asset {idx}: Capital goods emissions detected in "
                    "Category 8 scope. These should be in Category 2."
                ),
                "severity": ComplianceSeverity.MAJOR.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.CATEGORY_2.value,
                "recommendation": (
                    "Emissions from capital goods (HVAC systems, lifts, "
                    "IT infrastructure) installed in leased buildings "
                    "belong in Category 2, not Category 8."
                ),
            })

    def _check_dc_ula_007(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-007: No overlap with Cat 3 (WTT already in Cat 3).

        Well-to-tank emissions should not be double-counted between
        Category 8 and Category 3.
        """
        wtt_included = asset.get("wtt_included", False)
        wtt_in_cat3 = asset.get("wtt_reported_in_cat3", False)

        if wtt_included and wtt_in_cat3:
            findings.append({
                "rule_code": "DC-ULA-007",
                "description": (
                    f"Asset {idx}: WTT emissions included in Category 8 "
                    "AND reported in Category 3. Double-counting risk."
                ),
                "severity": ComplianceSeverity.MAJOR.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.CATEGORY_3.value,
                "recommendation": (
                    "Report WTT in either Category 3 (Fuel & Energy "
                    "Activities) or include as a component in Category 8, "
                    "not both. Document the WTT boundary clearly."
                ),
            })

    def _check_dc_ula_008(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-008: No overlap with Cat 5 (waste from leased buildings).

        Waste generated in leased buildings should be in Category 5,
        not Category 8.
        """
        includes_waste = asset.get("includes_waste_emissions", False)
        reported_in_cat5 = asset.get("reported_in_cat5", False)

        if includes_waste or reported_in_cat5:
            findings.append({
                "rule_code": "DC-ULA-008",
                "description": (
                    f"Asset {idx}: Waste emissions from leased building "
                    "detected in Category 8. These should be in Category 5."
                ),
                "severity": ComplianceSeverity.MAJOR.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.CATEGORY_5.value,
                "recommendation": (
                    "Waste generated in operations at leased buildings "
                    "belongs in Category 5 (Waste Generated in Operations), "
                    "not Category 8. Category 8 covers building operation "
                    "energy only."
                ),
            })

    def _check_dc_ula_009(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-009: No overlap with Cat 13 (lessor assets -> downstream).

        If the reporting company is also a lessor of the same asset to
        a third party, those emissions belong in Category 13 (Downstream
        Leased Assets), not Category 8.
        """
        is_sub_leased_fully = asset.get("is_sub_leased_fully", False)
        reported_in_cat13 = asset.get("reported_in_cat13", False)
        is_lessor_asset = asset.get("is_lessor_asset", False)

        if is_sub_leased_fully or reported_in_cat13 or is_lessor_asset:
            findings.append({
                "rule_code": "DC-ULA-009",
                "description": (
                    f"Asset {idx}: Asset is fully sub-leased or classified "
                    "as a lessor asset. Emissions belong in Category 13 "
                    "(Downstream Leased Assets)."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "asset_index": idx,
                "category": DoubleCountingCategory.CATEGORY_13.value,
                "recommendation": (
                    "If the reporting company sub-leases the asset to a "
                    "third party, those emissions belong in Category 13 "
                    "(Downstream Leased Assets), not Category 8."
                ),
            })

    def _check_dc_ula_010(
        self, idx: int, asset: dict, findings: List[dict]
    ) -> None:
        """
        DC-ULA-010: Sub-lease allocation (exclude sub-leased portions).

        If a portion of the leased asset is sub-leased, that portion
        should be excluded from Category 8 and reported in Category 13.
        """
        sub_lease_fraction = self._safe_decimal(
            asset.get("sub_lease_fraction", "0")
        )
        sub_lease_area = self._safe_decimal(
            asset.get("sub_lease_area_sqm", "0")
        )
        allocation_factor = self._safe_decimal(
            asset.get("allocation_factor", "1")
        )

        if sub_lease_fraction > _ZERO and sub_lease_fraction < _ONE:
            # Partial sub-lease: check that allocation excludes sub-leased portion
            expected_alloc = _ONE - sub_lease_fraction
            if allocation_factor > expected_alloc + Decimal("0.05"):
                findings.append({
                    "rule_code": "DC-ULA-010",
                    "description": (
                        f"Asset {idx}: Sub-lease fraction is "
                        f"{sub_lease_fraction} but allocation factor is "
                        f"{allocation_factor}. Expected <= {expected_alloc}. "
                        "Sub-leased portion must be excluded."
                    ),
                    "severity": ComplianceSeverity.MAJOR.value,
                    "asset_index": idx,
                    "category": DoubleCountingCategory.CATEGORY_13.value,
                    "recommendation": (
                        "Reduce the allocation factor to exclude the "
                        "sub-leased portion. The sub-leased area should "
                        "be reported in Category 13 by the reporting company."
                    ),
                })

        if sub_lease_area > _ZERO:
            total_area = self._safe_decimal(asset.get("total_area_sqm", "0"))
            if total_area > _ZERO:
                sub_ratio = sub_lease_area / total_area
                if sub_ratio >= _ONE:
                    findings.append({
                        "rule_code": "DC-ULA-010",
                        "description": (
                            f"Asset {idx}: Entire leased area ({total_area} sqm) "
                            f"is sub-leased ({sub_lease_area} sqm). "
                            "Should be Category 13, not Category 8."
                        ),
                        "severity": ComplianceSeverity.CRITICAL.value,
                        "asset_index": idx,
                        "category": DoubleCountingCategory.CATEGORY_13.value,
                        "recommendation": (
                            "If the entire leased asset is sub-leased, "
                            "report in Category 13 (Downstream Leased Assets)."
                        ),
                    })

    # ==========================================================================
    # Lease Classification Validator
    # ==========================================================================

    def check_lease_classification(
        self,
        lease_data: dict,
    ) -> Dict[str, Any]:
        """
        Validate whether a lease is correctly classified as operating vs finance.

        Uses indicators from IFRS 16 / ASC 842 to classify:
        - Transfer of ownership at end of lease term -> finance
        - Bargain purchase option -> finance
        - Lease term >= 75% of asset useful life -> finance
        - PV of lease payments >= 90% of fair value -> finance
        - Specialized asset with no alternative use -> finance
        - Otherwise -> operating

        Args:
            lease_data: Dictionary with lease terms and classification.

        Returns:
            Dictionary with classification result, confidence, and indicators.
        """
        indicators: Dict[str, bool] = {}
        finance_indicators: int = 0
        total_indicators: int = 0

        # Indicator 1: Transfer of ownership
        transfer_ownership = lease_data.get("transfers_ownership", False)
        indicators["transfers_ownership"] = transfer_ownership
        total_indicators += 1
        if transfer_ownership:
            finance_indicators += 1

        # Indicator 2: Bargain purchase option
        bargain_purchase = lease_data.get("bargain_purchase_option", False)
        indicators["bargain_purchase_option"] = bargain_purchase
        total_indicators += 1
        if bargain_purchase:
            finance_indicators += 1

        # Indicator 3: Lease term >= 75% of useful life
        lease_term_months = self._safe_decimal(
            lease_data.get("lease_term_months", "0")
        )
        useful_life_months = self._safe_decimal(
            lease_data.get("useful_life_months", "0")
        )
        if useful_life_months > _ZERO and lease_term_months > _ZERO:
            term_ratio = lease_term_months / useful_life_months
            is_major_part = term_ratio >= Decimal("0.75")
            indicators["major_part_of_useful_life"] = is_major_part
            total_indicators += 1
            if is_major_part:
                finance_indicators += 1

        # Indicator 4: PV of payments >= 90% of fair value
        pv_payments = self._safe_decimal(
            lease_data.get("pv_lease_payments", "0")
        )
        fair_value = self._safe_decimal(
            lease_data.get("fair_value", "0")
        )
        if fair_value > _ZERO and pv_payments > _ZERO:
            pv_ratio = pv_payments / fair_value
            is_substantially_all = pv_ratio >= Decimal("0.90")
            indicators["substantially_all_fair_value"] = is_substantially_all
            total_indicators += 1
            if is_substantially_all:
                finance_indicators += 1

        # Indicator 5: Specialized asset
        specialized = lease_data.get("specialized_asset", False)
        indicators["specialized_asset"] = specialized
        total_indicators += 1
        if specialized:
            finance_indicators += 1

        # Determine classification
        if finance_indicators >= 1:
            classification = "finance_lease"
            ghg_scope = "scope_1_2"
            category_8_eligible = False
        else:
            classification = "operating_lease"
            ghg_scope = "category_8"
            category_8_eligible = True

        # Confidence score
        if total_indicators > 0:
            confidence = Decimal(str(
                max(finance_indicators, total_indicators - finance_indicators)
            )) / Decimal(str(total_indicators))
        else:
            confidence = Decimal("0.5")

        # Check against stated classification
        stated_type = str(
            lease_data.get("lease_type", "")
        ).lower().replace("-", "_").replace(" ", "_")
        classification_matches = True
        mismatch_warning = None

        if stated_type:
            if stated_type in FINANCE_LEASE_TYPES and classification == "operating_lease":
                classification_matches = False
                mismatch_warning = (
                    f"Stated lease type '{stated_type}' is finance, but "
                    f"indicators suggest operating lease ({finance_indicators} "
                    f"of {total_indicators} finance indicators)."
                )
            elif stated_type in OPERATING_LEASE_TYPES and classification == "finance_lease":
                classification_matches = False
                mismatch_warning = (
                    f"Stated lease type '{stated_type}' is operating, but "
                    f"indicators suggest finance lease ({finance_indicators} "
                    f"of {total_indicators} finance indicators)."
                )

        return {
            "classification": classification,
            "ghg_scope": ghg_scope,
            "category_8_eligible": category_8_eligible,
            "finance_indicators_count": finance_indicators,
            "total_indicators_count": total_indicators,
            "indicators": indicators,
            "confidence": float(confidence.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "classification_matches_stated": classification_matches,
            "mismatch_warning": mismatch_warning,
        }

    # ==========================================================================
    # Summary & Recommendations
    # ==========================================================================

    def get_compliance_summary(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Summarize all framework results with overall score and recommendations.

        Calculates a weighted average compliance score across all checked
        frameworks and aggregates findings by severity.

        Args:
            results: Dictionary mapping framework name to compliance result dict.

        Returns:
            Summary dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.
        """
        if not results:
            return {
                "overall_score": 0.0,
                "overall_status": ComplianceStatus.FAIL.value,
                "frameworks_checked": 0,
                "framework_scores": {},
                "critical_findings": [],
                "major_findings": [],
                "minor_findings": [],
                "info_findings": [],
                "total_findings": 0,
                "recommendations": [],
                "provenance_hash": "",
            }

        total_weight = _ZERO
        weighted_sum = _ZERO

        framework_scores: Dict[str, Dict[str, Any]] = {}

        for framework_name, check_result in results.items():
            if not isinstance(check_result, dict):
                continue

            score = self._safe_decimal(check_result.get("score", 0))

            try:
                fw_enum = ComplianceFramework(framework_name)
                weight = FRAMEWORK_WEIGHTS.get(fw_enum, _ONE)
            except ValueError:
                weight = _ONE

            weighted_sum += score * weight
            total_weight += weight

            framework_scores[framework_name] = {
                "score": float(score),
                "status": check_result.get("status", ComplianceStatus.FAIL.value),
                "findings_count": len(check_result.get("findings", [])),
                "weight": float(weight),
            }

        overall_score = _ZERO
        if total_weight > _ZERO:
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
        critical_findings: List[dict] = []
        major_findings: List[dict] = []
        minor_findings: List[dict] = []
        info_findings: List[dict] = []

        all_recommendations: List[str] = []
        seen_recommendations: Set[str] = set()

        for framework_name, check_result in results.items():
            if not isinstance(check_result, dict):
                continue
            for finding in check_result.get("findings", []):
                finding_with_fw = {
                    "framework": framework_name,
                    **finding,
                }
                severity = finding.get("severity", "")
                if severity == ComplianceSeverity.CRITICAL.value:
                    critical_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.MAJOR.value:
                    major_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.MINOR.value:
                    minor_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.INFO.value:
                    info_findings.append(finding_with_fw)

                rec = finding.get("recommendation")
                if rec and rec not in seen_recommendations:
                    all_recommendations.append(rec)
                    seen_recommendations.add(rec)

        if overall_score < Decimal("70"):
            priority_msg = (
                "PRIORITY: Overall compliance score is below 70%. "
                "Address critical and major severity issues immediately."
            )
            if priority_msg not in seen_recommendations:
                all_recommendations.insert(0, priority_msg)

        total_findings = (
            len(critical_findings) + len(major_findings)
            + len(minor_findings) + len(info_findings)
        )

        provenance_input = (
            f"{float(overall_score)}:{overall_status.value}:"
            f"{total_findings}:{len(results)}"
        )
        provenance_hash = hashlib.sha256(
            provenance_input.encode("utf-8")
        ).hexdigest()

        return {
            "overall_score": float(overall_score),
            "overall_status": overall_status.value,
            "frameworks_checked": len(results),
            "framework_scores": framework_scores,
            "critical_findings": critical_findings,
            "major_findings": major_findings,
            "minor_findings": minor_findings,
            "info_findings": info_findings,
            "total_findings": total_findings,
            "recommendations": all_recommendations,
            "provenance_hash": provenance_hash,
        }

    # ==========================================================================
    # Batch Compliance Check
    # ==========================================================================

    def check_batch_compliance(
        self,
        results: List[dict],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run compliance checks across a batch of calculation results.

        Args:
            results: List of calculation result dictionaries.
            frameworks: Optional list of framework names to check.

        Returns:
            Dictionary with batch_size, per-result findings,
            aggregated scores, and overall batch compliance.
        """
        start_time = time.monotonic()
        batch_size = len(results)

        logger.info(
            "Running batch compliance check: %d results",
            batch_size,
        )

        per_result_checks: List[Dict[str, Any]] = []
        all_framework_results: Dict[str, List[Dict[str, Any]]] = {}

        for i, calc_result in enumerate(results):
            try:
                fw_results = self.check_all_frameworks(
                    calc_result, frameworks=frameworks
                )
                summary = self.get_compliance_summary(fw_results)
                per_result_checks.append({
                    "index": i,
                    "status": "success",
                    "overall_score": summary["overall_score"],
                    "overall_status": summary["overall_status"],
                    "total_findings": summary["total_findings"],
                })

                for fw_name, fw_result in fw_results.items():
                    if fw_name not in all_framework_results:
                        all_framework_results[fw_name] = []
                    all_framework_results[fw_name].append(fw_result)

            except Exception as e:
                logger.error(
                    "Batch compliance check failed for result %d: %s",
                    i, str(e), exc_info=True,
                )
                per_result_checks.append({
                    "index": i,
                    "status": "error",
                    "error": str(e),
                    "overall_score": 0.0,
                    "overall_status": ComplianceStatus.FAIL.value,
                    "total_findings": 0,
                })

        # Aggregate framework scores
        aggregated_scores: Dict[str, Dict[str, Any]] = {}
        for fw_name, fw_result_list in all_framework_results.items():
            scores = [
                self._safe_decimal(r.get("score", 0))
                for r in fw_result_list
                if isinstance(r, dict)
            ]
            if scores:
                avg_score = sum(scores, _ZERO) / Decimal(str(len(scores)))
                min_score = min(scores)
                max_score = max(scores)
                aggregated_scores[fw_name] = {
                    "average_score": float(
                        avg_score.quantize(_QUANT_2DP, rounding=ROUNDING)
                    ),
                    "min_score": float(min_score),
                    "max_score": float(max_score),
                    "count": len(scores),
                }

        successful_results = [
            r for r in per_result_checks if r["status"] == "success"
        ]
        batch_scores = [r["overall_score"] for r in successful_results]
        batch_avg = (
            sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
        )

        duration = time.monotonic() - start_time

        logger.info(
            "Batch compliance check complete: %d results, "
            "avg_score=%.2f, duration=%.4fs",
            batch_size, batch_avg, duration,
        )

        return {
            "batch_size": batch_size,
            "successful_checks": len(successful_results),
            "failed_checks": batch_size - len(successful_results),
            "average_score": round(batch_avg, 2),
            "per_result_checks": per_result_checks,
            "aggregated_framework_scores": aggregated_scores,
            "duration_seconds": round(duration, 4),
        }

    # ==========================================================================
    # Required Disclosures
    # ==========================================================================

    def get_required_disclosures(
        self,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Get the required disclosures for specified frameworks.

        Args:
            frameworks: Optional list of framework names. If None, returns all.

        Returns:
            Dictionary mapping framework name to list of required field names.
        """
        result: Dict[str, List[str]] = {}

        target_frameworks = frameworks if frameworks else [
            fw.value for fw in ComplianceFramework
        ]

        for fw_name in target_frameworks:
            try:
                fw_enum = ComplianceFramework(fw_name)
            except ValueError:
                fw_map = {
                    "GHG_PROTOCOL_SCOPE3": ComplianceFramework.GHG_PROTOCOL,
                    "ISO_14064": ComplianceFramework.ISO_14064,
                    "CSRD_ESRS_E1": ComplianceFramework.CSRD_ESRS,
                    "CDP": ComplianceFramework.CDP,
                    "SBTI": ComplianceFramework.SBTI,
                    "SB_253": ComplianceFramework.SB_253,
                    "GRI_305": ComplianceFramework.GRI,
                }
                fw_enum = fw_map.get(fw_name)
                if fw_enum is None:
                    logger.warning(
                        "Unknown framework '%s' for required disclosures",
                        fw_name,
                    )
                    continue

            disclosures = FRAMEWORK_REQUIRED_DISCLOSURES.get(fw_enum, [])
            result[fw_enum.value] = disclosures

        return result

    def check_disclosure_completeness(
        self,
        result: dict,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Check whether a result contains all required disclosures
        for given frameworks.

        Args:
            result: Calculation result dictionary.
            frameworks: Optional list of framework names to check.

        Returns:
            Dictionary with per-framework completeness data.
        """
        required_disclosures = self.get_required_disclosures(frameworks)
        completeness: Dict[str, Any] = {}

        for fw_name, required_fields in required_disclosures.items():
            if not required_fields:
                completeness[fw_name] = {
                    "required": [],
                    "present": [],
                    "missing": [],
                    "completeness_pct": 100.0,
                }
                continue

            present = []
            missing = []

            for field_name in required_fields:
                value = result.get(field_name)
                if value is not None and value != "" and value != []:
                    present.append(field_name)
                else:
                    missing.append(field_name)

            total = len(required_fields)
            pct = (len(present) / total * 100) if total > 0 else 100.0

            completeness[fw_name] = {
                "required": required_fields,
                "present": present,
                "missing": missing,
                "completeness_pct": round(pct, 1),
            }

        # Overall completeness
        all_pcts = [
            v["completeness_pct"]
            for v in completeness.values()
            if isinstance(v, dict)
        ]
        overall_pct = sum(all_pcts) / len(all_pcts) if all_pcts else 0.0

        return {
            "frameworks": completeness,
            "overall_completeness_pct": round(overall_pct, 1),
        }

    # ==========================================================================
    # Engine Statistics
    # ==========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count,
            enabled frameworks, and configuration state.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._strict_mode,
            "double_counting_check": self._double_counting_check,
            "supported_asset_categories": sorted(VALID_ASSET_CATEGORIES),
            "supported_building_types": sorted(VALID_BUILDING_TYPES),
            "supported_vehicle_types": sorted(VALID_VEHICLE_TYPES),
            "supported_equipment_types": sorted(VALID_EQUIPMENT_TYPES),
            "supported_it_types": sorted(VALID_IT_TYPES),
        }

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    @staticmethod
    def _safe_decimal(value: Any) -> Decimal:
        """Safely convert any value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return _ZERO


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_compliance_checker() -> ComplianceCheckerEngine:
    """
    Get the ComplianceCheckerEngine singleton instance.

    Returns:
        ComplianceCheckerEngine singleton.

    Example:
        >>> engine = get_compliance_checker()
        >>> engine.get_engine_stats()["engine_id"]
        'compliance_checker_engine'
    """
    return ComplianceCheckerEngine.get_instance()


def reset_compliance_checker() -> None:
    """
    Reset the ComplianceCheckerEngine singleton (for testing only).

    Example:
        >>> reset_compliance_checker()
    """
    ComplianceCheckerEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "ComplianceSeverity",
    "ComplianceStatus",
    "ComplianceFramework",
    "DoubleCountingCategory",
    "BoundaryClassification",
    "ComplianceFinding",
    "FrameworkCheckState",
    "FRAMEWORK_WEIGHTS",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
    "reset_compliance_checker",
]
