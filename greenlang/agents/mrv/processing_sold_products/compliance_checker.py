# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-023 Engine 6

This module implements regulatory compliance checking for Processing of Sold Products
emissions (GHG Protocol Scope 3 Category 10) against 7 regulatory frameworks with
Category 10-specific compliance rules and 8 double-counting prevention rules.

Regulatory Frameworks (7):
    1. GHG Protocol Scope 3 Standard (Category 10 specific) - 8 rules
    2. ISO 14064-1:2018 (Clause 5.2.4) - 6 rules
    3. CSRD/ESRS E1 Climate Change - 7 rules
    4. CDP Climate Change Questionnaire (C6.5) - 5 rules
    5. SBTi (Science Based Targets initiative) - 6 rules
    6. SB 253 (California Climate Corporate Data Accountability Act) - 5 rules
    7. GRI 305-3 Emissions Standard - 4 rules

Category 10-Specific Compliance Rules:
    - Intermediate product boundary enforcement
    - Processing type classification
    - Customer processing data documentation
    - Calculation method selection and hierarchy
    - DQI (Data Quality Indicator) scoring and reporting
    - Emission factor source traceability
    - Allocation method documentation
    - Multi-step processing chain tracking

Double-Counting Prevention Rules (8):
    DC-PSP-001: Exclude own-facility processing (Scope 1)
    DC-PSP-002: Exclude own-facility electricity (Scope 2)
    DC-PSP-003: No overlap with purchased goods (Category 1)
    DC-PSP-004: No overlap with capital goods (Category 2)
    DC-PSP-005: Exclude transportation legs (Category 4 / 9)
    DC-PSP-006: No overlap with use-phase emissions (Category 11)
    DC-PSP-007: No overlap with end-of-life treatment (Category 12)
    DC-PSP-008: Avoid double-counting in multi-step processing chains

Zero-Hallucination Approach:
    - All validation logic is deterministic (no LLM calls)
    - Framework requirements sourced from official standards
    - Thresholds from published GHG Protocol / SBTi / CSRD guidance
    - SHA-256 provenance hashing for every compliance check

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all(calculation_result)
    >>> summary = engine.generate_compliance_report(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Module: greenlang.agents.mrv.processing_sold_products.compliance_checker
Agent: AGENT-MRV-023 (Processing of Sold Products)
Version: 1.0.0
Author: GreenLang Platform Team
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"
ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
ONE_HUNDRED: Decimal = Decimal("100")
_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class ComplianceFramework(str, Enum):
    """Regulatory / reporting framework for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ComplianceStatus(str, Enum):
    """Compliance check outcome status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 10."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_1 = "CATEGORY_1"
    CATEGORY_2 = "CATEGORY_2"
    CATEGORY_4 = "CATEGORY_4"
    CATEGORY_9 = "CATEGORY_9"
    CATEGORY_11 = "CATEGORY_11"
    CATEGORY_12 = "CATEGORY_12"


class CalculationMethod(str, Enum):
    """Calculation methods for Category 10 emissions."""

    SITE_SPECIFIC_DIRECT = "site_specific_direct"
    SITE_SPECIFIC_ENERGY = "site_specific_energy"
    SITE_SPECIFIC_FUEL = "site_specific_fuel"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class IntermediateProductCategory(str, Enum):
    """Categories of intermediate products sold for further processing."""

    METALS_FERROUS = "metals_ferrous"
    METALS_NON_FERROUS = "metals_non_ferrous"
    PLASTICS_THERMOPLASTIC = "plastics_thermoplastic"
    PLASTICS_THERMOSET = "plastics_thermoset"
    CHEMICALS = "chemicals"
    FOOD_INGREDIENTS = "food_ingredients"
    TEXTILES = "textiles"
    ELECTRONICS_COMPONENTS = "electronics_components"
    GLASS_CERAMICS = "glass_ceramics"
    WOOD_PAPER_PULP = "wood_paper_pulp"
    MINERALS = "minerals"
    AGRICULTURAL_COMMODITIES = "agricultural_commodities"


class ProcessingType(str, Enum):
    """Types of downstream processing operations."""

    MACHINING = "machining"
    STAMPING = "stamping"
    WELDING = "welding"
    HEAT_TREATMENT = "heat_treatment"
    INJECTION_MOLDING = "injection_molding"
    EXTRUSION = "extrusion"
    BLOW_MOLDING = "blow_molding"
    CASTING = "casting"
    FORGING = "forging"
    COATING = "coating"
    ASSEMBLY = "assembly"
    CHEMICAL_REACTION = "chemical_reaction"
    REFINING = "refining"
    MILLING = "milling"
    DRYING = "drying"
    SINTERING = "sintering"
    FERMENTATION = "fermentation"
    TEXTILE_FINISHING = "textile_finishing"


class EFSource(str, Enum):
    """Emission factor data source."""

    CUSTOMER_REPORTED = "customer_reported"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    EPA = "epa"
    DEFRA = "defra"
    EEIO = "eeio"
    IEA = "iea"
    INDUSTRY_ASSOCIATION = "industry_association"
    CUSTOM = "custom"


# ==============================================================================
# FRAMEWORK REQUIRED DISCLOSURES
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_emissions_reported",
        "intermediate_product_boundary",
        "calculation_method",
        "emission_factor_sources",
        "data_quality_indicators",
        "activity_data_documented",
        "processing_type_documented",
        "allocation_method_documented",
    ],
    ComplianceFramework.ISO_14064: [
        "total_emissions_reported",
        "methodology_documented",
        "uncertainty_quantified",
        "verification_evidence",
        "completeness_assessment",
        "boundary_documentation",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_emissions_reported",
        "methodology_disclosed",
        "value_chain_boundary",
        "ef_sources_documented",
        "data_quality_indicators",
        "dnsh_assessment",
        "transition_plan_alignment",
    ],
    ComplianceFramework.CDP: [
        "total_emissions_reported",
        "category_coverage",
        "methodology_description",
        "data_quality_tier",
        "verification_status",
    ],
    ComplianceFramework.SBTI: [
        "total_emissions_reported",
        "target_coverage",
        "base_year_defined",
        "recalculation_triggers",
        "scope3_materiality",
        "progress_tracking",
    ],
    ComplianceFramework.SB_253: [
        "total_emissions_reported",
        "reporting_threshold_met",
        "assurance_level",
        "methodology_documented",
        "data_quality_indicators",
    ],
    ComplianceFramework.GRI: [
        "total_emissions_reported",
        "methodology_documented",
        "gases_included",
        "consolidation_approach",
    ],
}

# ==============================================================================
# FRAMEWORK SCORING WEIGHTS
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
# SBTi COVERAGE THRESHOLDS
# ==============================================================================

SBTI_COVERAGE_THRESHOLD: Decimal = Decimal("67.0")
SBTI_NEAR_TERM_YEARS: int = 5
SBTI_NET_ZERO_YEARS: int = 10

# ==============================================================================
# SB 253 REVENUE THRESHOLD (USD)
# ==============================================================================

SB253_REVENUE_THRESHOLD_USD: Decimal = Decimal("1000000000")

# ==============================================================================
# DOUBLE-COUNTING RULE DEFINITIONS
# ==============================================================================

DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-PSP-001": {
        "name": "Exclude own-facility processing (Scope 1)",
        "description": (
            "Processing performed at the reporting company's own facilities "
            "must be excluded from Category 10 as it is already captured in "
            "Scope 1 (direct) emissions."
        ),
        "overlap_category": "SCOPE_1",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.1",
    },
    "DC-PSP-002": {
        "name": "Exclude own-facility electricity (Scope 2)",
        "description": (
            "Electricity consumed at the reporting company's own facilities "
            "for processing must be excluded from Category 10 as it is already "
            "captured in Scope 2 (indirect) emissions."
        ),
        "overlap_category": "SCOPE_2",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.1",
    },
    "DC-PSP-003": {
        "name": "No overlap with purchased goods (Category 1)",
        "description": (
            "Emissions from processing that has already been accounted for "
            "as cradle-to-gate in Category 1 (Purchased Goods & Services) "
            "must not be double-counted in Category 10."
        ),
        "overlap_category": "CATEGORY_1",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.2",
    },
    "DC-PSP-004": {
        "name": "No overlap with capital goods (Category 2)",
        "description": (
            "Emissions from capital goods used in downstream processing "
            "that are already reported in Category 2 must not be duplicated "
            "in Category 10."
        ),
        "overlap_category": "CATEGORY_2",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.2",
    },
    "DC-PSP-005": {
        "name": "Exclude transportation (Category 4 / 9)",
        "description": (
            "Transportation of intermediate products between processing "
            "facilities must be reported in Category 4 (upstream) or "
            "Category 9 (downstream), not in Category 10."
        ),
        "overlap_category": "CATEGORY_4/9",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.2",
    },
    "DC-PSP-006": {
        "name": "No overlap with use-phase emissions (Category 11)",
        "description": (
            "Emissions from the use phase of the final product must be "
            "reported in Category 11 (Use of Sold Products), not in "
            "Category 10 which covers only processing, not end-use."
        ),
        "overlap_category": "CATEGORY_11",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.2",
    },
    "DC-PSP-007": {
        "name": "No overlap with end-of-life treatment (Category 12)",
        "description": (
            "Emissions from end-of-life treatment of the final product must "
            "be reported in Category 12, not Category 10. Processing of sold "
            "products covers only intermediate processing steps."
        ),
        "overlap_category": "CATEGORY_12",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Table 6.2",
    },
    "DC-PSP-008": {
        "name": "Avoid double-counting in multi-step processing chains",
        "description": (
            "When a product passes through multiple downstream processors, "
            "each processing step must be counted exactly once. The reporting "
            "company must track which steps are included and avoid counting "
            "any step more than once."
        ),
        "overlap_category": "MULTI_STEP",
        "regulation_reference": "GHG Protocol Scope 3, Ch 6, Section 6.3",
    },
}

# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass
class ComplianceFinding:
    """Single compliance finding with rule code and severity."""

    rule_id: str
    description: str
    severity: ComplianceSeverity
    framework: str
    status: ComplianceStatus = ComplianceStatus.FAIL
    evidence: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    regulation_reference: Optional[str] = None


@dataclass
class ComplianceResult:
    """Result of compliance check for one framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    findings: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameworkCheckState:
    """Internal state accumulator for a single framework check."""

    framework: ComplianceFramework
    findings: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0

    def add_pass(self, rule_id: str, description: str) -> None:
        """Record a passed check."""
        self.passed_checks += 1
        self.total_checks += 1

    def add_fail(
        self,
        rule_id: str,
        description: str,
        severity: ComplianceSeverity,
        recommendation: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a failed check with a finding."""
        self.failed_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_id=rule_id,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.FAIL,
                evidence=evidence,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def add_warning(
        self,
        rule_id: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.MEDIUM,
        recommendation: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a warning check with a finding."""
        self.warning_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_id=rule_id,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.WARNING,
                evidence=evidence,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def compute_score(self) -> Decimal:
        """
        Compute compliance score (0-100).

        Scoring:
            - Each failed check reduces score proportionally
            - Warnings reduce score at 50% of a failure
            - Perfect score = 100 (all checks passed)

        Returns:
            Score as Decimal in range [0, 100].
        """
        if self.total_checks == 0:
            return Decimal("100.00")

        penalty_points = (
            Decimal(str(self.failed_checks))
            + Decimal(str(self.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(self.total_checks))
        score = (
            (max_points - penalty_points) / max_points * ONE_HUNDRED
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        if score < ZERO:
            score = Decimal("0.00")
        if score > ONE_HUNDRED:
            score = Decimal("100.00")

        return score

    def compute_status(self) -> ComplianceStatus:
        """Compute overall status from findings."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 0:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS

    def to_result(self, provenance_hash: str = "") -> ComplianceResult:
        """Convert accumulated state to a ComplianceResult."""
        return ComplianceResult(
            framework=self.framework,
            status=self.compute_status(),
            score=self.compute_score(),
            findings=list(self.findings),
            passed_checks=self.passed_checks,
            failed_checks=self.failed_checks,
            warning_checks=self.warning_checks,
            total_checks=self.total_checks,
            checked_at=datetime.now(timezone.utc),
            provenance_hash=provenance_hash,
            metadata={
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "agent_id": AGENT_ID,
            },
        )


@dataclass
class DCCheckResult:
    """Result of a double-counting prevention check."""

    rule_id: str
    rule_name: str
    passed: bool
    overlap_category: str
    description: str
    evidence: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    regulation_reference: Optional[str] = None


# ==============================================================================
# VALID PRODUCT / PROCESSING COMBINATIONS
# ==============================================================================

VALID_PROCESSING_BY_CATEGORY: Dict[IntermediateProductCategory, Set[ProcessingType]] = {
    IntermediateProductCategory.METALS_FERROUS: {
        ProcessingType.MACHINING, ProcessingType.STAMPING, ProcessingType.WELDING,
        ProcessingType.HEAT_TREATMENT, ProcessingType.CASTING, ProcessingType.FORGING,
        ProcessingType.COATING, ProcessingType.ASSEMBLY,
    },
    IntermediateProductCategory.METALS_NON_FERROUS: {
        ProcessingType.MACHINING, ProcessingType.STAMPING, ProcessingType.WELDING,
        ProcessingType.HEAT_TREATMENT, ProcessingType.CASTING, ProcessingType.FORGING,
        ProcessingType.COATING, ProcessingType.ASSEMBLY, ProcessingType.EXTRUSION,
    },
    IntermediateProductCategory.PLASTICS_THERMOPLASTIC: {
        ProcessingType.INJECTION_MOLDING, ProcessingType.EXTRUSION,
        ProcessingType.BLOW_MOLDING, ProcessingType.COATING, ProcessingType.ASSEMBLY,
    },
    IntermediateProductCategory.PLASTICS_THERMOSET: {
        ProcessingType.INJECTION_MOLDING, ProcessingType.COATING,
        ProcessingType.ASSEMBLY, ProcessingType.CHEMICAL_REACTION,
    },
    IntermediateProductCategory.CHEMICALS: {
        ProcessingType.CHEMICAL_REACTION, ProcessingType.REFINING,
        ProcessingType.MILLING, ProcessingType.DRYING,
    },
    IntermediateProductCategory.FOOD_INGREDIENTS: {
        ProcessingType.MILLING, ProcessingType.DRYING, ProcessingType.FERMENTATION,
        ProcessingType.HEAT_TREATMENT, ProcessingType.CHEMICAL_REACTION,
    },
    IntermediateProductCategory.TEXTILES: {
        ProcessingType.TEXTILE_FINISHING, ProcessingType.COATING,
        ProcessingType.DRYING, ProcessingType.ASSEMBLY,
    },
    IntermediateProductCategory.ELECTRONICS_COMPONENTS: {
        ProcessingType.ASSEMBLY, ProcessingType.WELDING, ProcessingType.COATING,
        ProcessingType.HEAT_TREATMENT, ProcessingType.SINTERING,
    },
    IntermediateProductCategory.GLASS_CERAMICS: {
        ProcessingType.HEAT_TREATMENT, ProcessingType.SINTERING,
        ProcessingType.COATING, ProcessingType.MILLING,
    },
    IntermediateProductCategory.WOOD_PAPER_PULP: {
        ProcessingType.MILLING, ProcessingType.DRYING,
        ProcessingType.CHEMICAL_REACTION, ProcessingType.COATING,
    },
    IntermediateProductCategory.MINERALS: {
        ProcessingType.MILLING, ProcessingType.SINTERING,
        ProcessingType.HEAT_TREATMENT, ProcessingType.CHEMICAL_REACTION,
    },
    IntermediateProductCategory.AGRICULTURAL_COMMODITIES: {
        ProcessingType.MILLING, ProcessingType.DRYING, ProcessingType.FERMENTATION,
        ProcessingType.HEAT_TREATMENT, ProcessingType.CHEMICAL_REACTION,
    },
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    ComplianceCheckerEngine for Processing of Sold Products (Category 10).

    Validates calculation results against 7 regulatory frameworks with
    Category 10-specific rules for intermediate product boundary enforcement,
    processing type classification, and 8 double-counting prevention rules.

    Thread Safety:
        Singleton pattern with threading.RLock() for concurrent access.

    Attributes:
        _enabled_frameworks: Set of enabled compliance frameworks.
        _check_count: Running count of compliance checks performed.

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_all(calculation_result)
        >>> for r in results:
        ...     print(f"{r.framework.value}: {r.status.value} ({r.score}%)")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine."""
        self._enabled_frameworks: List[ComplianceFramework] = list(ComplianceFramework)
        self._check_count: int = 0
        self._total_duration_ms: float = 0.0

        logger.info(
            "%s v%s: ComplianceCheckerEngine initialized "
            "(frameworks=%d, dc_rules=%d)",
            AGENT_ID,
            ENGINE_VERSION,
            len(self._enabled_frameworks),
            len(DC_RULES),
        )

    @classmethod
    def get_instance(cls) -> "ComplianceCheckerEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            ComplianceCheckerEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("ComplianceCheckerEngine singleton reset")

    # ==========================================================================
    # MAIN ENTRY POINTS
    # ==========================================================================

    def check_all(
        self,
        result: Dict[str, Any],
        frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> List[ComplianceResult]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked.

        Args:
            result: Calculation result dictionary containing total_emissions_kg_co2e,
                product breakdowns, calculation methods, EF sources, and DQI scores.
            frameworks: Optional list of specific frameworks to check.
                If None, checks all enabled frameworks.

        Returns:
            List of ComplianceResult, one per framework checked.

        Example:
            >>> results = engine.check_all(calc_result)
            >>> compliant = [r for r in results if r.status == ComplianceStatus.PASS]
        """
        start_time = time.monotonic()
        active_frameworks = frameworks or self._enabled_frameworks

        logger.info(
            "Running compliance checks: %d frameworks",
            len(active_frameworks),
        )

        check_results: List[ComplianceResult] = []

        framework_dispatch: Dict[
            ComplianceFramework, Any
        ] = {
            ComplianceFramework.GHG_PROTOCOL: self.check_ghg_protocol,
            ComplianceFramework.ISO_14064: self.check_iso_14064,
            ComplianceFramework.CSRD_ESRS: self.check_csrd,
            ComplianceFramework.CDP: self.check_cdp,
            ComplianceFramework.SBTI: self.check_sbti,
            ComplianceFramework.SB_253: self.check_sb253,
            ComplianceFramework.GRI: self.check_gri,
        }

        for framework in active_frameworks:
            check_fn = framework_dispatch.get(framework)
            if check_fn is None:
                logger.warning("No validator for framework: %s", framework.value)
                continue

            try:
                check_result = check_fn(result)
                check_results.append(check_result)
                logger.info(
                    "%s compliance: %s (score: %s)",
                    framework.value,
                    check_result.status.value,
                    check_result.score,
                )
            except Exception as e:
                logger.error(
                    "Error checking %s compliance: %s",
                    framework.value,
                    str(e),
                    exc_info=True,
                )
                error_result = ComplianceResult(
                    framework=framework,
                    status=ComplianceStatus.FAIL,
                    score=ZERO,
                    findings=[
                        ComplianceFinding(
                            rule_id="CHECK_ERROR",
                            description=f"Compliance check failed: {str(e)}",
                            severity=ComplianceSeverity.CRITICAL,
                            framework=framework.value,
                            status=ComplianceStatus.FAIL,
                        )
                    ],
                    failed_checks=1,
                    total_checks=1,
                )
                check_results.append(error_result)

        duration_ms = (time.monotonic() - start_time) * 1000
        self._check_count += 1
        self._total_duration_ms += duration_ms

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.2fms",
            len(check_results),
            duration_ms,
        )

        return check_results

    # ==========================================================================
    # FRAMEWORK 1: GHG Protocol Scope 3 (Category 10) - 8 Rules
    # ==========================================================================

    def check_ghg_protocol(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against GHG Protocol Scope 3 Standard (Category 10 specific).

        8 Rules:
            GHG-PSP-001: Total emissions reported and > 0
            GHG-PSP-002: Intermediate product boundary defined
            GHG-PSP-003: Calculation method documented
            GHG-PSP-004: Activity data documented
            GHG-PSP-005: Emission factor sources documented
            GHG-PSP-006: Data quality indicators reported
            GHG-PSP-007: Processing type documented per product
            GHG-PSP-008: Allocation method documented

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for GHG Protocol.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # Rule GHG-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("GHG-PSP-001", "Total emissions reported")
        else:
            state.add_fail(
                rule_id="GHG-PSP-001",
                description="Total emissions not reported or zero",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total Category 10 emissions in kg CO2e",
                regulation_reference="GHG Protocol Scope 3, Chapter 6",
            )

        # Rule GHG-PSP-002: Intermediate product boundary
        products = result.get("products", [])
        boundary_defined = self._check_boundary_defined(products)
        if boundary_defined:
            state.add_pass("GHG-PSP-002", "Intermediate product boundary defined")
        else:
            state.add_fail(
                rule_id="GHG-PSP-002",
                description="Intermediate product boundary not defined",
                severity=ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Define which intermediate products are sold for further "
                    "processing and document the boundary between Category 10 "
                    "and other categories"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 6, Section 6.1",
            )

        # Rule GHG-PSP-003: Calculation method documented
        method = result.get("calculation_method")
        if method is not None:
            state.add_pass("GHG-PSP-003", "Calculation method documented")
        else:
            state.add_fail(
                rule_id="GHG-PSP-003",
                description="Calculation method not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used: site-specific, "
                    "average-data, or spend-based"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 6, Table 6.3",
            )

        # Rule GHG-PSP-004: Activity data documented
        has_activity_data = self._check_activity_data(result)
        if has_activity_data:
            state.add_pass("GHG-PSP-004", "Activity data documented")
        else:
            state.add_fail(
                rule_id="GHG-PSP-004",
                description="Activity data not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document activity data: mass/quantity of intermediate "
                    "products sold and processing energy consumption"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 6, Section 6.2",
            )

        # Rule GHG-PSP-005: Emission factor sources
        ef_sources = result.get("ef_sources", [])
        if len(ef_sources) > 0:
            state.add_pass("GHG-PSP-005", "Emission factor sources documented")
        else:
            state.add_fail(
                rule_id="GHG-PSP-005",
                description="Emission factor sources not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document all emission factor sources used (e.g., Ecoinvent, "
                    "EPA, DEFRA, customer-reported)"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 6, Section 6.4",
            )

        # Rule GHG-PSP-006: Data quality indicators
        dqi_score = result.get("data_quality_score")
        dqi_breakdown = result.get("dqi_breakdown")
        if dqi_score is not None or dqi_breakdown is not None:
            state.add_pass("GHG-PSP-006", "Data quality indicators reported")
        else:
            state.add_warning(
                rule_id="GHG-PSP-006",
                description="Data quality indicators not reported",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report DQI scores across 5 dimensions: representativeness, "
                    "completeness, temporal correlation, geographical correlation, "
                    "technological correlation"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 7",
            )

        # Rule GHG-PSP-007: Processing type documented per product
        processing_documented = self._check_processing_type_documented(products)
        if processing_documented:
            state.add_pass("GHG-PSP-007", "Processing type documented per product")
        else:
            state.add_warning(
                rule_id="GHG-PSP-007",
                description="Processing type not documented for all products",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the downstream processing type for each "
                    "intermediate product sold (e.g., machining, welding, extrusion)"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 6",
            )

        # Rule GHG-PSP-008: Allocation method documented
        allocation_method = result.get("allocation_method")
        if allocation_method is not None or len(products) <= 1:
            state.add_pass("GHG-PSP-008", "Allocation method documented")
        else:
            state.add_warning(
                rule_id="GHG-PSP-008",
                description="Allocation method not documented for multi-product portfolio",
                severity=ComplianceSeverity.LOW,
                recommendation=(
                    "Document allocation method when emissions are shared "
                    "across multiple products (mass-based, economic, or other)"
                ),
                regulation_reference="GHG Protocol Scope 3, Chapter 8",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.GHG_PROTOCOL, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 2: ISO 14064-1 (Clause 5.2.4) - 6 Rules
    # ==========================================================================

    def check_iso_14064(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against ISO 14064-1:2018 Clause 5.2.4.

        6 Rules:
            ISO-PSP-001: Total emissions reported
            ISO-PSP-002: Methodology documented
            ISO-PSP-003: Uncertainty quantified
            ISO-PSP-004: Verification evidence provided
            ISO-PSP-005: Completeness assessment documented
            ISO-PSP-006: Boundary documentation complete

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for ISO 14064.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # Rule ISO-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("ISO-PSP-001", "Total emissions reported")
        else:
            state.add_fail(
                rule_id="ISO-PSP-001",
                description="Total emissions not reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 10 emissions in kg CO2e",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # Rule ISO-PSP-002: Methodology documented
        methodology = result.get("methodology") or result.get("calculation_method")
        if methodology is not None:
            state.add_pass("ISO-PSP-002", "Methodology documented")
        else:
            state.add_fail(
                rule_id="ISO-PSP-002",
                description="Methodology not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document calculation methodology per ISO 14064-1 requirements "
                    "including data sources, assumptions, and calculation approach"
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # Rule ISO-PSP-003: Uncertainty quantified
        uncertainty = result.get("uncertainty_percentage") or result.get("uncertainty_pct")
        if uncertainty is not None:
            state.add_pass("ISO-PSP-003", "Uncertainty quantified")
        else:
            state.add_warning(
                rule_id="ISO-PSP-003",
                description="Uncertainty not quantified",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Quantify uncertainty using IPCC Tier 2 error propagation, "
                    "Monte Carlo simulation, or analytical methods"
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.3",
            )

        # Rule ISO-PSP-004: Verification evidence
        verification = result.get("verification_evidence") or result.get("verified")
        if verification:
            state.add_pass("ISO-PSP-004", "Verification evidence provided")
        else:
            state.add_warning(
                rule_id="ISO-PSP-004",
                description="No verification evidence provided",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide verification evidence from an accredited third-party "
                    "verifier per ISO 14064-3"
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # Rule ISO-PSP-005: Completeness assessment
        completeness = result.get("completeness_percentage") or result.get("coverage_percentage")
        if completeness is not None:
            completeness_val = self._safe_decimal(completeness)
            if completeness_val is not None and completeness_val >= Decimal("95"):
                state.add_pass("ISO-PSP-005", "Completeness assessment documented (>=95%)")
            else:
                state.add_warning(
                    rule_id="ISO-PSP-005",
                    description=f"Completeness below 95% ({completeness_val}%)",
                    severity=ComplianceSeverity.MEDIUM,
                    evidence={"completeness_pct": str(completeness_val)},
                    recommendation="Improve data coverage to achieve >=95% completeness",
                    regulation_reference="ISO 14064-1:2018, Clause 5.2",
                )
        else:
            state.add_fail(
                rule_id="ISO-PSP-005",
                description="Completeness assessment not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation="Document the completeness of emissions data for Category 10",
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # Rule ISO-PSP-006: Boundary documentation
        boundary_doc = result.get("boundary_documentation") or result.get("organizational_boundary")
        if boundary_doc is not None:
            state.add_pass("ISO-PSP-006", "Boundary documentation complete")
        else:
            state.add_fail(
                rule_id="ISO-PSP-006",
                description="Boundary documentation missing",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document organizational boundary (equity share or control approach) "
                    "and Category 10 product boundary"
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.ISO_14064, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 3: CSRD/ESRS E1 - 7 Rules
    # ==========================================================================

    def check_csrd(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against CSRD ESRS E1 Climate Change.

        7 Rules:
            CSRD-PSP-001: Total emissions reported (E1-6 disclosure)
            CSRD-PSP-002: Methodology disclosed
            CSRD-PSP-003: Value chain boundary defined
            CSRD-PSP-004: Emission factor sources documented
            CSRD-PSP-005: Data quality indicators provided
            CSRD-PSP-006: DNSH assessment completed
            CSRD-PSP-007: Transition plan alignment disclosed

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for CSRD/ESRS E1.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # Rule CSRD-PSP-001: Total emissions (E1-6)
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("CSRD-PSP-001", "Total emissions reported (E1-6)")
        else:
            state.add_fail(
                rule_id="CSRD-PSP-001",
                description="Total emissions not reported for E1-6 disclosure",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 10 emissions per ESRS E1-6",
                regulation_reference="ESRS E1, Disclosure Requirement E1-6",
            )

        # Rule CSRD-PSP-002: Methodology disclosed
        methodology = result.get("methodology") or result.get("calculation_method")
        if methodology is not None:
            state.add_pass("CSRD-PSP-002", "Methodology disclosed")
        else:
            state.add_fail(
                rule_id="CSRD-PSP-002",
                description="Methodology not disclosed",
                severity=ComplianceSeverity.HIGH,
                recommendation="Disclose calculation methodology per ESRS E1 requirements",
                regulation_reference="ESRS E1, Disclosure Requirement E1-6, AR 46",
            )

        # Rule CSRD-PSP-003: Value chain boundary
        value_chain_boundary = result.get("value_chain_boundary") or result.get("organizational_boundary")
        if value_chain_boundary is not None:
            state.add_pass("CSRD-PSP-003", "Value chain boundary defined")
        else:
            state.add_fail(
                rule_id="CSRD-PSP-003",
                description="Value chain boundary not defined",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Define value chain boundary covering downstream processing "
                    "of sold intermediate products"
                ),
                regulation_reference="ESRS E1, Disclosure Requirement E1-6, AR 44",
            )

        # Rule CSRD-PSP-004: EF sources documented
        ef_sources = result.get("ef_sources", [])
        if len(ef_sources) > 0:
            state.add_pass("CSRD-PSP-004", "Emission factor sources documented")
        else:
            state.add_fail(
                rule_id="CSRD-PSP-004",
                description="Emission factor sources not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation="Document all emission factor sources used in calculations",
                regulation_reference="ESRS E1, Disclosure Requirement E1-6, AR 47",
            )

        # Rule CSRD-PSP-005: Data quality indicators
        dqi_score = result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("CSRD-PSP-005", "Data quality indicators provided")
        else:
            state.add_warning(
                rule_id="CSRD-PSP-005",
                description="Data quality indicators not provided",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Report data quality indicators for Scope 3 emissions estimates",
                regulation_reference="ESRS E1, Disclosure Requirement E1-6, AR 48",
            )

        # Rule CSRD-PSP-006: DNSH assessment
        dnsh_assessment = result.get("dnsh_assessment")
        if dnsh_assessment is not None:
            state.add_pass("CSRD-PSP-006", "DNSH assessment completed")
        else:
            state.add_warning(
                rule_id="CSRD-PSP-006",
                description="Do No Significant Harm (DNSH) assessment not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Complete DNSH assessment per EU Taxonomy Regulation Article 17 "
                    "for downstream processing activities"
                ),
                regulation_reference="ESRS E1, Disclosure Requirement E1-1, AR 16",
            )

        # Rule CSRD-PSP-007: Transition plan alignment
        transition_plan = result.get("transition_plan_alignment")
        if transition_plan is not None:
            state.add_pass("CSRD-PSP-007", "Transition plan alignment disclosed")
        else:
            state.add_warning(
                rule_id="CSRD-PSP-007",
                description="Transition plan alignment not disclosed",
                severity=ComplianceSeverity.LOW,
                recommendation=(
                    "Disclose alignment of Category 10 emissions with "
                    "the company's climate transition plan"
                ),
                regulation_reference="ESRS E1, Disclosure Requirement E1-1",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.CSRD_ESRS, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 4: CDP C6.5 - 5 Rules
    # ==========================================================================

    def check_cdp(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against CDP Climate Change Questionnaire C6.5.

        5 Rules:
            CDP-PSP-001: Total emissions reported
            CDP-PSP-002: Category coverage disclosed
            CDP-PSP-003: Methodology description provided
            CDP-PSP-004: Data quality tier indicated
            CDP-PSP-005: Verification status disclosed

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for CDP.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # Rule CDP-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("CDP-PSP-001", "Total emissions reported")
        else:
            state.add_fail(
                rule_id="CDP-PSP-001",
                description="Total emissions not reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 10 emissions in metric tonnes CO2e",
                regulation_reference="CDP C6.5",
            )

        # Rule CDP-PSP-002: Category coverage
        coverage = result.get("coverage_percentage") or result.get("completeness_percentage")
        if coverage is not None:
            coverage_val = self._safe_decimal(coverage)
            if coverage_val is not None and coverage_val >= Decimal("80"):
                state.add_pass("CDP-PSP-002", "Category coverage adequate (>=80%)")
            else:
                state.add_warning(
                    rule_id="CDP-PSP-002",
                    description=f"Category coverage below 80% ({coverage_val}%)",
                    severity=ComplianceSeverity.MEDIUM,
                    evidence={"coverage_pct": str(coverage_val)},
                    recommendation="Improve coverage to at least 80% of Category 10 emissions",
                    regulation_reference="CDP C6.5, Scoring Methodology",
                )
        else:
            state.add_fail(
                rule_id="CDP-PSP-002",
                description="Category coverage not disclosed",
                severity=ComplianceSeverity.HIGH,
                recommendation="Disclose the percentage of Category 10 emissions covered",
                regulation_reference="CDP C6.5",
            )

        # Rule CDP-PSP-003: Methodology description
        methodology = result.get("methodology") or result.get("calculation_method")
        if methodology is not None:
            state.add_pass("CDP-PSP-003", "Methodology description provided")
        else:
            state.add_fail(
                rule_id="CDP-PSP-003",
                description="Methodology description not provided",
                severity=ComplianceSeverity.HIGH,
                recommendation="Describe the methodology used for Category 10 calculations",
                regulation_reference="CDP C6.5",
            )

        # Rule CDP-PSP-004: Data quality tier
        dqi_tier = result.get("data_quality_tier") or result.get("data_quality_score")
        if dqi_tier is not None:
            state.add_pass("CDP-PSP-004", "Data quality tier indicated")
        else:
            state.add_warning(
                rule_id="CDP-PSP-004",
                description="Data quality tier not indicated",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Indicate data quality tier (primary, secondary, or estimated)",
                regulation_reference="CDP C6.5, Data Quality Assessment",
            )

        # Rule CDP-PSP-005: Verification status
        verification = result.get("verification_status") or result.get("verified")
        if verification is not None:
            state.add_pass("CDP-PSP-005", "Verification status disclosed")
        else:
            state.add_warning(
                rule_id="CDP-PSP-005",
                description="Verification status not disclosed",
                severity=ComplianceSeverity.LOW,
                recommendation="Disclose verification/assurance status for Scope 3 emissions",
                regulation_reference="CDP C10.1, Verification",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.CDP, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 5: SBTi - 6 Rules
    # ==========================================================================

    def check_sbti(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against SBTi Corporate Net-Zero requirements.

        6 Rules:
            SBTI-PSP-001: Total emissions reported
            SBTI-PSP-002: Target coverage >= 67%
            SBTI-PSP-003: Base year defined
            SBTI-PSP-004: Recalculation triggers documented
            SBTI-PSP-005: Scope 3 materiality assessment
            SBTI-PSP-006: Progress tracking disclosed

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for SBTi.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI)

        # Rule SBTI-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("SBTI-PSP-001", "Total emissions reported")
        else:
            state.add_fail(
                rule_id="SBTI-PSP-001",
                description="Total emissions not reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 10 emissions for SBTi target tracking",
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 4",
            )

        # Rule SBTI-PSP-002: Target coverage >= 67%
        target_coverage = self._safe_decimal(result.get("target_coverage_percentage"))
        if target_coverage is not None and target_coverage >= SBTI_COVERAGE_THRESHOLD:
            state.add_pass(
                "SBTI-PSP-002",
                f"Target coverage adequate ({target_coverage}% >= {SBTI_COVERAGE_THRESHOLD}%)",
            )
        elif target_coverage is not None:
            state.add_fail(
                rule_id="SBTI-PSP-002",
                description=(
                    f"Target coverage {target_coverage}% below "
                    f"SBTi threshold of {SBTI_COVERAGE_THRESHOLD}%"
                ),
                severity=ComplianceSeverity.HIGH,
                evidence={"target_coverage_pct": str(target_coverage)},
                recommendation=(
                    f"Increase Scope 3 target coverage to at least "
                    f"{SBTI_COVERAGE_THRESHOLD}% of total Scope 3 emissions"
                ),
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 4.2",
            )
        else:
            state.add_warning(
                rule_id="SBTI-PSP-002",
                description="Target coverage percentage not provided",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose Scope 3 target coverage percentage for SBTi validation",
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 4.2",
            )

        # Rule SBTI-PSP-003: Base year defined
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("SBTI-PSP-003", f"Base year defined ({base_year})")
        else:
            state.add_fail(
                rule_id="SBTI-PSP-003",
                description="Base year not defined",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Define a base year for SBTi target setting. Must be the most "
                    "recent year with verifiable data (no earlier than 2015)"
                ),
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 3.1",
            )

        # Rule SBTI-PSP-004: Recalculation triggers
        recalc_triggers = result.get("recalculation_triggers")
        if recalc_triggers is not None:
            state.add_pass("SBTI-PSP-004", "Recalculation triggers documented")
        else:
            state.add_warning(
                rule_id="SBTI-PSP-004",
                description="Recalculation triggers not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document recalculation triggers (structural changes, "
                    "methodology updates, error corrections exceeding threshold)"
                ),
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 3.3",
            )

        # Rule SBTI-PSP-005: Scope 3 materiality assessment
        total_scope3 = self._safe_decimal(result.get("total_scope3_emissions_kg_co2e"))
        cat10_pct = self._safe_decimal(result.get("category_10_percentage_of_scope3"))
        if total_scope3 is not None and cat10_pct is not None:
            if cat10_pct >= Decimal("1.0"):
                state.add_pass(
                    "SBTI-PSP-005",
                    f"Category 10 is material ({cat10_pct}% of Scope 3)",
                )
            else:
                state.add_warning(
                    rule_id="SBTI-PSP-005",
                    description=(
                        f"Category 10 represents {cat10_pct}% of Scope 3, "
                        "below 1% materiality threshold"
                    ),
                    severity=ComplianceSeverity.LOW,
                    evidence={"cat10_pct_of_scope3": str(cat10_pct)},
                    recommendation=(
                        "Category 10 may be excluded from SBTi target if below "
                        "materiality threshold, but must still be reported"
                    ),
                    regulation_reference="SBTi Corporate Net-Zero Standard, Section 4.2",
                )
        else:
            state.add_warning(
                rule_id="SBTI-PSP-005",
                description="Scope 3 materiality assessment not available for Category 10",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide total Scope 3 emissions and Category 10 percentage "
                    "for materiality assessment"
                ),
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 4.1",
            )

        # Rule SBTI-PSP-006: Progress tracking
        progress = result.get("progress_tracking") or result.get("year_over_year_change")
        if progress is not None:
            state.add_pass("SBTI-PSP-006", "Progress tracking disclosed")
        else:
            state.add_warning(
                rule_id="SBTI-PSP-006",
                description="Progress tracking not disclosed",
                severity=ComplianceSeverity.LOW,
                recommendation=(
                    "Disclose year-over-year change in Category 10 emissions "
                    "relative to the base year target pathway"
                ),
                regulation_reference="SBTi Corporate Net-Zero Standard, Section 5",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.SBTI, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 6: SB 253 - 5 Rules
    # ==========================================================================

    def check_sb253(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against California SB 253 (Climate Corporate Data Accountability Act).

        5 Rules:
            SB253-PSP-001: Total emissions reported
            SB253-PSP-002: Revenue threshold met (>= $1B USD)
            SB253-PSP-003: Assurance level disclosed
            SB253-PSP-004: Methodology documented
            SB253-PSP-005: Data quality indicators disclosed

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for SB 253.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # Rule SB253-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("SB253-PSP-001", "Total emissions reported")
        else:
            state.add_fail(
                rule_id="SB253-PSP-001",
                description="Total emissions not reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 10 emissions per SB 253 requirements",
                regulation_reference="California SB 253, Section 3(a)",
            )

        # Rule SB253-PSP-002: Revenue threshold
        revenue_usd = self._safe_decimal(result.get("company_revenue_usd"))
        if revenue_usd is not None:
            if revenue_usd >= SB253_REVENUE_THRESHOLD_USD:
                state.add_pass(
                    "SB253-PSP-002",
                    f"Revenue threshold met (${revenue_usd:,.0f} >= ${SB253_REVENUE_THRESHOLD_USD:,.0f})",
                )
            else:
                state.add_warning(
                    rule_id="SB253-PSP-002",
                    description=(
                        f"Company revenue ${revenue_usd:,.0f} below SB 253 threshold "
                        f"of ${SB253_REVENUE_THRESHOLD_USD:,.0f}"
                    ),
                    severity=ComplianceSeverity.INFO,
                    evidence={"revenue_usd": str(revenue_usd)},
                    recommendation=(
                        "SB 253 reporting requirement applies to companies with "
                        "annual revenue >= $1 billion doing business in California"
                    ),
                    regulation_reference="California SB 253, Section 2(c)",
                )
        else:
            state.add_warning(
                rule_id="SB253-PSP-002",
                description="Company revenue not provided for SB 253 threshold check",
                severity=ComplianceSeverity.LOW,
                recommendation="Provide company annual revenue for SB 253 applicability assessment",
                regulation_reference="California SB 253, Section 2(c)",
            )

        # Rule SB253-PSP-003: Assurance level
        assurance_level = result.get("assurance_level")
        if assurance_level is not None:
            state.add_pass("SB253-PSP-003", f"Assurance level disclosed ({assurance_level})")
        else:
            state.add_warning(
                rule_id="SB253-PSP-003",
                description="Assurance level not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Disclose assurance level: limited or reasonable. "
                    "SB 253 requires third-party assurance (limited initially, "
                    "reasonable by 2030)"
                ),
                regulation_reference="California SB 253, Section 3(b)",
            )

        # Rule SB253-PSP-004: Methodology documented
        methodology = result.get("methodology") or result.get("calculation_method")
        if methodology is not None:
            state.add_pass("SB253-PSP-004", "Methodology documented")
        else:
            state.add_fail(
                rule_id="SB253-PSP-004",
                description="Methodology not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation="Document calculation methodology per SB 253 requirements",
                regulation_reference="California SB 253, Section 3(a)(2)",
            )

        # Rule SB253-PSP-005: Data quality indicators
        dqi_score = result.get("data_quality_score") or result.get("data_quality_tier")
        if dqi_score is not None:
            state.add_pass("SB253-PSP-005", "Data quality indicators disclosed")
        else:
            state.add_warning(
                rule_id="SB253-PSP-005",
                description="Data quality indicators not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose data quality indicators for Scope 3 emissions estimates",
                regulation_reference="California SB 253, Section 3(a)(3)",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.SB_253, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # FRAMEWORK 7: GRI 305-3 - 4 Rules
    # ==========================================================================

    def check_gri(self, result: Dict[str, Any]) -> ComplianceResult:
        """
        Validate against GRI 305-3 (Other indirect GHG emissions).

        4 Rules:
            GRI-PSP-001: Total emissions reported (305-3a)
            GRI-PSP-002: Methodology documented (305-3b)
            GRI-PSP-003: Gases included (305-3c)
            GRI-PSP-004: Consolidation approach disclosed (305-3d)

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult with findings for GRI 305.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GRI)

        # Rule GRI-PSP-001: Total emissions reported
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e"))
        if total_emissions is not None and total_emissions > ZERO:
            state.add_pass("GRI-PSP-001", "Total emissions reported (305-3a)")
        else:
            state.add_fail(
                rule_id="GRI-PSP-001",
                description="Total indirect GHG emissions not reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 10 emissions in metric tonnes CO2e (305-3a)",
                regulation_reference="GRI 305-3, Disclosure 305-3a",
            )

        # Rule GRI-PSP-002: Methodology documented
        methodology = result.get("methodology") or result.get("calculation_method")
        if methodology is not None:
            state.add_pass("GRI-PSP-002", "Methodology documented (305-3b)")
        else:
            state.add_fail(
                rule_id="GRI-PSP-002",
                description="Methodology not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the standards, methodologies, assumptions and/or "
                    "calculation tools used (GRI 305-3b)"
                ),
                regulation_reference="GRI 305-3, Disclosure 305-3b",
            )

        # Rule GRI-PSP-003: Gases included
        gases_included = result.get("gases_included")
        gas_breakdown = result.get("gas_breakdown")
        if gases_included is not None or gas_breakdown is not None:
            state.add_pass("GRI-PSP-003", "Gases included in calculation disclosed (305-3c)")
        else:
            state.add_warning(
                rule_id="GRI-PSP-003",
                description="Gases included in calculation not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Disclose which GHGs are included (CO2, CH4, N2O, HFCs, "
                    "PFCs, SF6, NF3) per GRI 305-3c"
                ),
                regulation_reference="GRI 305-3, Disclosure 305-3c",
            )

        # Rule GRI-PSP-004: Consolidation approach
        consolidation = result.get("consolidation_approach") or result.get("organizational_boundary")
        if consolidation is not None:
            state.add_pass("GRI-PSP-004", "Consolidation approach disclosed (305-3d)")
        else:
            state.add_warning(
                rule_id="GRI-PSP-004",
                description="Consolidation approach not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Disclose consolidation approach: equity share, financial "
                    "control, or operational control (GRI 305-3d)"
                ),
                regulation_reference="GRI 305-3, Disclosure 305-3d",
            )

        provenance_hash = self._build_provenance(ComplianceFramework.GRI, result)
        return state.to_result(provenance_hash)

    # ==========================================================================
    # DOUBLE-COUNTING PREVENTION - 8 Rules
    # ==========================================================================

    def check_dc_scope1(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-001: Exclude own-facility processing (Scope 1).

        Verifies that processing at the reporting company's own facilities
        has been excluded from Category 10 results, as these emissions
        belong in Scope 1.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Scope 1 overlap detected).
        """
        includes_own_facility = result.get("includes_own_facility_processing", False)
        own_facility_ids = result.get("own_facility_ids", [])
        products = result.get("products", [])

        if includes_own_facility:
            logger.warning(
                "DC-PSP-001 FAIL: Result explicitly includes own-facility processing"
            )
            return False

        for product in products:
            processing_facility = product.get("processing_facility_id", "")
            if processing_facility in own_facility_ids:
                logger.warning(
                    "DC-PSP-001 FAIL: Product %s processed at own facility %s",
                    product.get("product_id", "unknown"),
                    processing_facility,
                )
                return False

        logger.info("DC-PSP-001 PASS: No own-facility processing detected")
        return True

    def check_dc_scope2(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-002: Exclude own-facility electricity (Scope 2).

        Verifies that electricity consumed at the reporting company's own
        facilities for any processing has been excluded, as these are
        Scope 2 emissions.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Scope 2 overlap detected).
        """
        includes_own_electricity = result.get("includes_own_facility_electricity", False)
        if includes_own_electricity:
            logger.warning(
                "DC-PSP-002 FAIL: Result includes own-facility electricity consumption"
            )
            return False

        products = result.get("products", [])
        own_facility_ids = set(result.get("own_facility_ids", []))

        for product in products:
            if product.get("energy_source") == "own_facility_grid":
                if product.get("processing_facility_id", "") in own_facility_ids:
                    logger.warning(
                        "DC-PSP-002 FAIL: Product %s uses own-facility electricity",
                        product.get("product_id", "unknown"),
                    )
                    return False

        logger.info("DC-PSP-002 PASS: No own-facility electricity detected")
        return True

    def check_dc_cat1(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-003: No overlap with purchased goods (Category 1).

        Verifies that emissions already counted in Category 1 (cradle-to-gate)
        are not also counted in Category 10.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Category 1 overlap detected).
        """
        cat1_boundary = result.get("category_1_boundary", "")
        cat1_includes_processing = result.get("category_1_includes_downstream_processing", False)

        if cat1_includes_processing:
            logger.warning(
                "DC-PSP-003 FAIL: Category 1 already includes downstream processing emissions"
            )
            return False

        products = result.get("products", [])
        for product in products:
            if product.get("included_in_cat1_cradle_to_gate", False):
                logger.warning(
                    "DC-PSP-003 FAIL: Product %s processing already in Cat 1 cradle-to-gate",
                    product.get("product_id", "unknown"),
                )
                return False

        logger.info("DC-PSP-003 PASS: No Category 1 overlap detected")
        return True

    def check_dc_cat2(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-004: No overlap with capital goods (Category 2).

        Verifies that capital equipment emissions already in Category 2
        are not duplicated in Category 10 processing calculations.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Category 2 overlap detected).
        """
        includes_capital_goods = result.get("includes_capital_goods_emissions", False)
        if includes_capital_goods:
            logger.warning(
                "DC-PSP-004 FAIL: Category 10 includes capital goods emissions"
            )
            return False

        products = result.get("products", [])
        for product in products:
            if product.get("includes_equipment_depreciation", False):
                logger.warning(
                    "DC-PSP-004 FAIL: Product %s includes equipment depreciation emissions",
                    product.get("product_id", "unknown"),
                )
                return False

        logger.info("DC-PSP-004 PASS: No Category 2 overlap detected")
        return True

    def check_dc_cat4_9(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-005: Exclude transportation (Category 4 / Category 9).

        Verifies that transportation emissions between processing facilities
        are excluded from Category 10 (should be in Cat 4 or Cat 9).

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no transportation overlap detected).
        """
        includes_transport = result.get("includes_transportation_between_facilities", False)
        if includes_transport:
            logger.warning(
                "DC-PSP-005 FAIL: Category 10 includes inter-facility transportation"
            )
            return False

        products = result.get("products", [])
        for product in products:
            if product.get("includes_transport_emissions", False):
                logger.warning(
                    "DC-PSP-005 FAIL: Product %s includes transport emissions",
                    product.get("product_id", "unknown"),
                )
                return False

        logger.info("DC-PSP-005 PASS: No transportation overlap detected")
        return True

    def check_dc_cat11(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-006: No overlap with use-phase emissions (Category 11).

        Verifies that use-phase emissions of the final product are not
        included in Category 10, which covers only processing stages.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Category 11 overlap detected).
        """
        includes_use_phase = result.get("includes_use_phase_emissions", False)
        if includes_use_phase:
            logger.warning(
                "DC-PSP-006 FAIL: Category 10 includes use-phase emissions"
            )
            return False

        products = result.get("products", [])
        for product in products:
            boundary = product.get("boundary", "")
            if "use_phase" in str(boundary).lower():
                logger.warning(
                    "DC-PSP-006 FAIL: Product %s boundary extends to use phase",
                    product.get("product_id", "unknown"),
                )
                return False

        logger.info("DC-PSP-006 PASS: No use-phase overlap detected")
        return True

    def check_dc_cat12(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-007: No overlap with end-of-life treatment (Category 12).

        Verifies that end-of-life treatment emissions are not included
        in Category 10. These belong in Category 12.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no Category 12 overlap detected).
        """
        includes_eol = result.get("includes_end_of_life_emissions", False)
        if includes_eol:
            logger.warning(
                "DC-PSP-007 FAIL: Category 10 includes end-of-life treatment emissions"
            )
            return False

        products = result.get("products", [])
        for product in products:
            boundary = product.get("boundary", "")
            if "end_of_life" in str(boundary).lower() or "eol" in str(boundary).lower():
                logger.warning(
                    "DC-PSP-007 FAIL: Product %s boundary extends to end-of-life",
                    product.get("product_id", "unknown"),
                )
                return False

        logger.info("DC-PSP-007 PASS: No end-of-life overlap detected")
        return True

    def check_dc_multi_step(self, result: Dict[str, Any]) -> bool:
        """
        DC-PSP-008: Avoid double-counting in multi-step processing chains.

        When a product passes through multiple downstream processors,
        each processing step must be counted exactly once. This check
        verifies that no processing step appears more than once.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if check passes (no multi-step double-counting detected).
        """
        products = result.get("products", [])
        seen_steps: Set[str] = set()

        for product in products:
            processing_chain = product.get("processing_chain", [])
            for step in processing_chain:
                step_id = step.get("step_id", "")
                if not step_id:
                    facility_id = step.get("facility_id", "")
                    processing_type = step.get("processing_type", "")
                    step_id = f"{facility_id}:{processing_type}"

                if step_id in seen_steps:
                    logger.warning(
                        "DC-PSP-008 FAIL: Duplicate processing step '%s' in chain",
                        step_id,
                    )
                    return False
                seen_steps.add(step_id)

        logger.info(
            "DC-PSP-008 PASS: No multi-step double-counting detected (%d unique steps)",
            len(seen_steps),
        )
        return True

    def check_all_dc_rules(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run all 8 double-counting prevention rules.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of dictionaries with rule results.

        Example:
            >>> dc_results = engine.check_all_dc_rules(calc_result)
            >>> failed = [r for r in dc_results if not r["passed"]]
            >>> assert len(failed) == 0, f"DC failures: {failed}"
        """
        dc_check_methods = [
            ("DC-PSP-001", self.check_dc_scope1),
            ("DC-PSP-002", self.check_dc_scope2),
            ("DC-PSP-003", self.check_dc_cat1),
            ("DC-PSP-004", self.check_dc_cat2),
            ("DC-PSP-005", self.check_dc_cat4_9),
            ("DC-PSP-006", self.check_dc_cat11),
            ("DC-PSP-007", self.check_dc_cat12),
            ("DC-PSP-008", self.check_dc_multi_step),
        ]

        results: List[Dict[str, Any]] = []

        for rule_id, check_fn in dc_check_methods:
            rule_def = DC_RULES.get(rule_id, {})
            try:
                passed = check_fn(result)
                results.append({
                    "rule_id": rule_id,
                    "rule_name": rule_def.get("name", rule_id),
                    "passed": passed,
                    "overlap_category": rule_def.get("overlap_category", ""),
                    "description": rule_def.get("description", ""),
                    "regulation_reference": rule_def.get("regulation_reference", ""),
                    "recommendation": (
                        None if passed
                        else f"Review and correct {rule_id} double-counting issue"
                    ),
                })
            except Exception as e:
                logger.error("DC check %s failed: %s", rule_id, str(e), exc_info=True)
                results.append({
                    "rule_id": rule_id,
                    "rule_name": rule_def.get("name", rule_id),
                    "passed": False,
                    "overlap_category": rule_def.get("overlap_category", ""),
                    "description": f"Check failed with error: {str(e)}",
                    "regulation_reference": rule_def.get("regulation_reference", ""),
                    "recommendation": f"Resolve error in {rule_id} check and rerun",
                })

        passed_count = sum(1 for r in results if r["passed"])
        logger.info(
            "Double-counting check complete: %d/%d rules passed",
            passed_count,
            len(results),
        )

        return results

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def validate_boundary(self, products: List[Dict[str, Any]]) -> List[str]:
        """
        Validate intermediate product boundary definitions.

        Each product must have a product_category (from IntermediateProductCategory)
        and a processing_type (from ProcessingType). This method validates that
        the processing type is compatible with the product category.

        Args:
            products: List of product dictionaries.

        Returns:
            List of validation error messages (empty if valid).

        Example:
            >>> errors = engine.validate_boundary(products)
            >>> if errors:
            ...     print("Boundary validation failed:")
            ...     for e in errors:
            ...         print(f"  - {e}")
        """
        errors: List[str] = []

        if not products:
            errors.append("No products provided for boundary validation")
            return errors

        for idx, product in enumerate(products):
            product_id = product.get("product_id", f"product_{idx}")

            # Check product category
            category_str = product.get("product_category", "")
            try:
                category = IntermediateProductCategory(category_str)
            except (ValueError, KeyError):
                errors.append(
                    f"Product {product_id}: invalid product category '{category_str}'. "
                    f"Valid categories: {[c.value for c in IntermediateProductCategory]}"
                )
                continue

            # Check processing type
            processing_str = product.get("processing_type", "")
            try:
                processing = ProcessingType(processing_str)
            except (ValueError, KeyError):
                errors.append(
                    f"Product {product_id}: invalid processing type '{processing_str}'. "
                    f"Valid types: {[p.value for p in ProcessingType]}"
                )
                continue

            # Check compatibility
            valid_processing = VALID_PROCESSING_BY_CATEGORY.get(category, set())
            if processing not in valid_processing:
                errors.append(
                    f"Product {product_id}: processing type '{processing_str}' is not "
                    f"compatible with category '{category_str}'. Valid processing "
                    f"types for {category_str}: {sorted([p.value for p in valid_processing])}"
                )

            # Check mass/quantity provided
            mass = product.get("mass_tonnes") or product.get("quantity")
            if mass is None:
                errors.append(
                    f"Product {product_id}: mass_tonnes or quantity must be provided"
                )

        return errors

    def validate_completeness(self, result: Dict[str, Any]) -> float:
        """
        Calculate the percentage of products with calculations.

        Args:
            result: Calculation result dictionary.

        Returns:
            Percentage (0-100) of products with non-zero emissions.

        Example:
            >>> completeness = engine.validate_completeness(calc_result)
            >>> print(f"Completeness: {completeness:.1f}%")
        """
        products = result.get("products", [])
        if not products:
            return 0.0

        calculated_count = 0
        for product in products:
            emissions = product.get("emissions_kg_co2e")
            if emissions is not None:
                emissions_val = self._safe_decimal(emissions)
                if emissions_val is not None and emissions_val > ZERO:
                    calculated_count += 1

        return float(
            (Decimal(str(calculated_count)) / Decimal(str(len(products))) * ONE_HUNDRED)
            .quantize(_QUANT_2DP, rounding=ROUNDING)
        )

    def validate_method_appropriateness(
        self,
        method: str,
        data_available: Dict[str, bool],
    ) -> bool:
        """
        Validate that the chosen calculation method is appropriate
        given the available data.

        Method selection hierarchy (GHG Protocol recommended):
            1. Site-specific direct > 2. Site-specific energy/fuel >
            3. Average-data > 4. Spend-based

        Args:
            method: Calculation method name.
            data_available: Dictionary of data availability flags.

        Returns:
            True if the method is appropriate given available data.

        Example:
            >>> is_ok = engine.validate_method_appropriateness(
            ...     "average_data",
            ...     {"customer_processing_data": False, "product_mass": True}
            ... )
        """
        has_customer_data = data_available.get("customer_processing_data", False)
        has_energy_data = data_available.get("energy_consumption_data", False)
        has_fuel_data = data_available.get("fuel_consumption_data", False)
        has_product_mass = data_available.get("product_mass", False)
        has_spend_data = data_available.get("spend_data", False)

        try:
            method_enum = CalculationMethod(method)
        except (ValueError, KeyError):
            logger.warning("Unknown calculation method: %s", method)
            return False

        if method_enum == CalculationMethod.SITE_SPECIFIC_DIRECT:
            return has_customer_data

        if method_enum == CalculationMethod.SITE_SPECIFIC_ENERGY:
            return has_energy_data

        if method_enum == CalculationMethod.SITE_SPECIFIC_FUEL:
            return has_fuel_data

        if method_enum == CalculationMethod.AVERAGE_DATA:
            if has_customer_data or has_energy_data or has_fuel_data:
                logger.warning(
                    "Average-data method used when higher-quality data is available"
                )
            return has_product_mass

        if method_enum == CalculationMethod.SPEND_BASED:
            if has_customer_data or has_energy_data or has_fuel_data or has_product_mass:
                logger.warning(
                    "Spend-based method used when higher-quality data is available"
                )
            return has_spend_data

        return False

    def generate_compliance_report(
        self,
        results: List[ComplianceResult],
        dc_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report from check results.

        Args:
            results: List of framework compliance results.
            dc_results: Optional double-counting check results.

        Returns:
            Dictionary containing overall score, framework breakdown,
            findings summary, and double-counting assessment.

        Example:
            >>> report = engine.generate_compliance_report(results, dc_results)
            >>> print(f"Overall score: {report['overall_score']}%")
            >>> print(f"Compliant frameworks: {report['compliant_count']}/{report['total_frameworks']}")
        """
        if not results:
            return {
                "overall_score": Decimal("0"),
                "overall_status": ComplianceStatus.FAIL.value,
                "total_frameworks": 0,
                "compliant_count": 0,
                "warning_count": 0,
                "non_compliant_count": 0,
                "frameworks": {},
                "all_findings": [],
                "double_counting": {},
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": AGENT_ID,
                "engine_version": ENGINE_VERSION,
            }

        # Calculate weighted overall score
        weighted_score_sum = ZERO
        weight_sum = ZERO
        compliant_count = 0
        warning_count = 0
        non_compliant_count = 0
        all_findings: List[Dict[str, Any]] = []
        framework_details: Dict[str, Any] = {}

        for r in results:
            weight = FRAMEWORK_WEIGHTS.get(r.framework, Decimal("0.70"))
            weighted_score_sum += r.score * weight
            weight_sum += weight

            if r.status == ComplianceStatus.PASS:
                compliant_count += 1
            elif r.status == ComplianceStatus.WARNING:
                warning_count += 1
            else:
                non_compliant_count += 1

            framework_details[r.framework.value] = {
                "status": r.status.value,
                "score": str(r.score),
                "passed_checks": r.passed_checks,
                "failed_checks": r.failed_checks,
                "warning_checks": r.warning_checks,
                "total_checks": r.total_checks,
                "provenance_hash": r.provenance_hash,
            }

            for finding in r.findings:
                all_findings.append({
                    "rule_id": finding.rule_id,
                    "framework": finding.framework,
                    "severity": finding.severity.value,
                    "status": finding.status.value,
                    "description": finding.description,
                    "recommendation": finding.recommendation,
                    "regulation_reference": finding.regulation_reference,
                })

        overall_score = ZERO
        if weight_sum > ZERO:
            overall_score = (weighted_score_sum / weight_sum).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )

        if non_compliant_count > 0:
            overall_status = ComplianceStatus.FAIL.value
        elif warning_count > 0:
            overall_status = ComplianceStatus.WARNING.value
        else:
            overall_status = ComplianceStatus.PASS.value

        # Double-counting summary
        dc_summary: Dict[str, Any] = {}
        if dc_results is not None:
            dc_passed = sum(1 for d in dc_results if d.get("passed", False))
            dc_summary = {
                "total_rules": len(dc_results),
                "passed_rules": dc_passed,
                "failed_rules": len(dc_results) - dc_passed,
                "all_passed": dc_passed == len(dc_results),
                "rules": dc_results,
            }

        return {
            "overall_score": str(overall_score),
            "overall_status": overall_status,
            "total_frameworks": len(results),
            "compliant_count": compliant_count,
            "warning_count": warning_count,
            "non_compliant_count": non_compliant_count,
            "frameworks": framework_details,
            "all_findings": all_findings,
            "critical_findings": [
                f for f in all_findings if f["severity"] == ComplianceSeverity.CRITICAL.value
            ],
            "double_counting": dc_summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": AGENT_ID,
            "engine_version": ENGINE_VERSION,
        }

    def get_framework_requirements(
        self, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """
        Get detailed requirements for a specific framework.

        Args:
            framework: Compliance framework enum value.

        Returns:
            Dictionary describing framework requirements, disclosures, and thresholds.

        Example:
            >>> reqs = engine.get_framework_requirements(ComplianceFramework.GHG_PROTOCOL)
            >>> print(f"Required disclosures: {reqs['required_disclosures']}")
        """
        required_disclosures = FRAMEWORK_REQUIRED_DISCLOSURES.get(framework, [])
        weight = FRAMEWORK_WEIGHTS.get(framework, Decimal("0.70"))

        framework_info: Dict[ComplianceFramework, Dict[str, Any]] = {
            ComplianceFramework.GHG_PROTOCOL: {
                "name": "GHG Protocol Scope 3 Standard",
                "category": "Category 10 - Processing of Sold Products",
                "reference": "GHG Protocol Scope 3, Chapter 6",
                "rule_count": 8,
                "rule_prefix": "GHG-PSP",
                "key_requirements": [
                    "Total emissions in kg CO2e",
                    "Intermediate product boundary definition",
                    "Calculation method documentation",
                    "Activity data documentation",
                    "Emission factor source traceability",
                    "Data quality indicators",
                    "Processing type per product",
                    "Allocation method documentation",
                ],
            },
            ComplianceFramework.ISO_14064: {
                "name": "ISO 14064-1:2018",
                "category": "Clause 5.2.4 - Indirect GHG Emissions",
                "reference": "ISO 14064-1:2018, Clause 5.2.4",
                "rule_count": 6,
                "rule_prefix": "ISO-PSP",
                "key_requirements": [
                    "Total emissions reported",
                    "Methodology documentation",
                    "Uncertainty quantification",
                    "Verification evidence",
                    "Completeness assessment",
                    "Boundary documentation",
                ],
            },
            ComplianceFramework.CSRD_ESRS: {
                "name": "CSRD ESRS E1 Climate Change",
                "category": "E1-6 Disclosure Requirement",
                "reference": "ESRS E1, Disclosure Requirement E1-6",
                "rule_count": 7,
                "rule_prefix": "CSRD-PSP",
                "key_requirements": [
                    "E1-6 total emissions disclosure",
                    "Methodology disclosure",
                    "Value chain boundary",
                    "EF source documentation",
                    "Data quality indicators",
                    "DNSH assessment",
                    "Transition plan alignment",
                ],
            },
            ComplianceFramework.CDP: {
                "name": "CDP Climate Change Questionnaire",
                "category": "C6.5 - Scope 3 Emissions",
                "reference": "CDP C6.5",
                "rule_count": 5,
                "rule_prefix": "CDP-PSP",
                "key_requirements": [
                    "Total emissions reported",
                    "Category coverage disclosure",
                    "Methodology description",
                    "Data quality tier",
                    "Verification status",
                ],
            },
            ComplianceFramework.SBTI: {
                "name": "SBTi Corporate Net-Zero Standard",
                "category": "Scope 3 Target Setting",
                "reference": "SBTi Corporate Net-Zero Standard, Section 4",
                "rule_count": 6,
                "rule_prefix": "SBTI-PSP",
                "key_requirements": [
                    "Total emissions reported",
                    "Target coverage >= 67%",
                    "Base year defined",
                    "Recalculation triggers",
                    "Scope 3 materiality assessment",
                    "Progress tracking",
                ],
            },
            ComplianceFramework.SB_253: {
                "name": "California SB 253",
                "category": "Climate Corporate Data Accountability Act",
                "reference": "California SB 253, Section 3",
                "rule_count": 5,
                "rule_prefix": "SB253-PSP",
                "key_requirements": [
                    "Total emissions reported",
                    "Revenue threshold (>= $1B)",
                    "Assurance level disclosed",
                    "Methodology documented",
                    "Data quality indicators",
                ],
            },
            ComplianceFramework.GRI: {
                "name": "GRI 305-3 Emissions Standard",
                "category": "Disclosure 305-3 Other Indirect GHG",
                "reference": "GRI 305-3",
                "rule_count": 4,
                "rule_prefix": "GRI-PSP",
                "key_requirements": [
                    "Total emissions reported (305-3a)",
                    "Methodology documented (305-3b)",
                    "Gases included (305-3c)",
                    "Consolidation approach (305-3d)",
                ],
            },
        }

        info = framework_info.get(framework, {})

        return {
            "framework": framework.value,
            "weight": str(weight),
            "required_disclosures": required_disclosures,
            **info,
        }

    def get_compliance_summary(
        self, results: Dict[str, ComplianceResult]
    ) -> Dict[str, Any]:
        """
        Get a brief compliance summary suitable for dashboards.

        Args:
            results: Dictionary of framework name to ComplianceResult.

        Returns:
            Summary dictionary with overall status and per-framework scores.
        """
        if not results:
            return {
                "overall_status": ComplianceStatus.FAIL.value,
                "overall_score": "0.00",
                "framework_count": 0,
            }

        result_list = list(results.values()) if isinstance(results, dict) else results
        scores = [r.score for r in result_list]
        avg_score = sum(scores, ZERO) / Decimal(str(len(scores)))
        avg_score = avg_score.quantize(_QUANT_2DP, rounding=ROUNDING)

        has_fail = any(r.status == ComplianceStatus.FAIL for r in result_list)
        has_warn = any(r.status == ComplianceStatus.WARNING for r in result_list)

        if has_fail:
            overall = ComplianceStatus.FAIL.value
        elif has_warn:
            overall = ComplianceStatus.WARNING.value
        else:
            overall = ComplianceStatus.PASS.value

        return {
            "overall_status": overall,
            "overall_score": str(avg_score),
            "framework_count": len(result_list),
            "frameworks": {
                r.framework.value: {
                    "status": r.status.value,
                    "score": str(r.score),
                }
                for r in result_list
            },
        }

    # ==========================================================================
    # PRIVATE HELPER METHODS
    # ==========================================================================

    def _build_provenance(
        self, framework: ComplianceFramework, result: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for a compliance check.

        Args:
            framework: Framework being checked.
            result: Calculation result used for checking.

        Returns:
            SHA-256 hex digest string.
        """
        provenance_input = {
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "framework": framework.value,
            "total_emissions_kg_co2e": str(result.get("total_emissions_kg_co2e", "0")),
            "calculation_method": str(result.get("calculation_method", "")),
            "product_count": len(result.get("products", [])),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        serialized = json.dumps(provenance_input, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """
        Safely convert a value to Decimal.

        Args:
            value: Value to convert.

        Returns:
            Decimal value, or None if conversion fails.
        """
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return None

    def _check_boundary_defined(self, products: List[Dict[str, Any]]) -> bool:
        """
        Check if intermediate product boundary is defined.

        The boundary is considered defined when at least one product has
        a product_category field from IntermediateProductCategory.

        Args:
            products: List of product dictionaries.

        Returns:
            True if boundary is adequately defined.
        """
        if not products:
            return False

        valid_categories = {c.value for c in IntermediateProductCategory}
        for product in products:
            category = product.get("product_category", "")
            if category in valid_categories:
                return True

        return False

    def _check_activity_data(self, result: Dict[str, Any]) -> bool:
        """
        Check if activity data is documented.

        Activity data includes mass/quantity of products and either
        processing energy data or emission factors applied.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if activity data is adequately documented.
        """
        products = result.get("products", [])
        if not products:
            return False

        for product in products:
            has_mass = product.get("mass_tonnes") is not None or product.get("quantity") is not None
            has_ef = product.get("emission_factor") is not None or product.get("ef_kg_co2e_per_tonne") is not None
            has_energy = product.get("energy_kwh") is not None or product.get("fuel_litres") is not None
            has_spend = product.get("spend_usd") is not None or product.get("revenue_usd") is not None
            has_direct = product.get("direct_emissions_kg_co2e") is not None

            if has_mass and (has_ef or has_energy or has_spend or has_direct):
                return True

        return False

    def _check_processing_type_documented(self, products: List[Dict[str, Any]]) -> bool:
        """
        Check if processing type is documented for each product.

        Args:
            products: List of product dictionaries.

        Returns:
            True if all products have processing_type documented.
        """
        if not products:
            return False

        valid_processing = {p.value for p in ProcessingType}
        for product in products:
            processing = product.get("processing_type", "")
            if processing not in valid_processing:
                return False

        return True


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_compliance_engine() -> ComplianceCheckerEngine:
    """
    Get the singleton ComplianceCheckerEngine instance.

    Returns:
        ComplianceCheckerEngine singleton instance.

    Example:
        >>> engine = get_compliance_engine()
        >>> results = engine.check_all(calc_result)
    """
    return ComplianceCheckerEngine.get_instance()


def check_compliance(
    result: Dict[str, Any],
    frameworks: Optional[List[ComplianceFramework]] = None,
) -> List[ComplianceResult]:
    """
    Convenience function to run compliance checks.

    Args:
        result: Calculation result dictionary.
        frameworks: Optional list of specific frameworks to check.

    Returns:
        List of ComplianceResult.

    Example:
        >>> results = check_compliance(calc_result)
        >>> for r in results:
        ...     print(f"{r.framework.value}: {r.status.value}")
    """
    engine = get_compliance_engine()
    return engine.check_all(result, frameworks)


def check_double_counting(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function to run all double-counting prevention checks.

    Args:
        result: Calculation result dictionary.

    Returns:
        List of DC check result dictionaries.

    Example:
        >>> dc_results = check_double_counting(calc_result)
        >>> all_passed = all(r["passed"] for r in dc_results)
    """
    engine = get_compliance_engine()
    return engine.check_all_dc_rules(result)
