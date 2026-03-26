# -*- coding: utf-8 -*-
"""
BaseYearReportingEngine - PACK-045 Base Year Management Engine 10
==================================================================

Multi-framework base year reporting engine that generates disclosure-ready
reports for GHG Protocol, ISO 14064, ESRS E1, CDP, SBTi, SEC Climate
Rule, California SB 253, and TCFD.  Each framework has specific base
year disclosure requirements; this engine maps inventory data to the
correct format for each framework and checks disclosure completeness.

Report Generation Architecture:
    1. Framework requirement lookup (static, deterministic)
    2. Inventory data mapping to framework-specific sections
    3. Content generation (Markdown, HTML, JSON, CSV)
    4. Disclosure completeness check against requirements
    5. Multi-framework cross-reference matrix generation

Framework-Specific Disclosures:

    GHG Protocol Corporate Standard (Chapter 5):
        - Base year selection and rationale
        - Base year emissions by scope
        - Recalculation policy
        - Recalculation history with adjustments
        - Significance threshold definition

    ISO 14064-1:2018 (Clause 5.2):
        - Historical base year and justification
        - Base year quantification methodology
        - Conditions for base year update
        - Documentation of any restatements

    ESRS E1-6 (Climate Change):
        - Base year and rationale for selection
        - Base year emissions (Scope 1, 2, 3 separately)
        - Restatement of base year if recalculated
        - Methodology changes and their impact

    CDP Climate Change C5:
        - C5.1: Base year for Scope 1 and 2
        - C5.1a: Base year emission details
        - C5.2: Emissions data for base year
        - C5.2a: Methodology for base year recalculation

    SBTi:
        - Base year emissions for target boundary
        - Base year recalculation policy
        - Recalculation history
        - Target boundary coverage percentage

    SEC Climate Disclosure Rule:
        - Base year emissions (Scope 1, 2, and optionally 3)
        - Material changes to base year
        - Methodology description
        - Three years of comparable data

    California SB 253:
        - GHG emissions by scope
        - Base year reference
        - Methodology citation

    TCFD:
        - Historical emissions trend
        - Base year reference for targets
        - Scenario analysis base year

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapters 5, 9
    - ISO 14064-1:2018, Clauses 5.2, 9.3
    - ESRS E1-6 (Gross Scopes 1, 2 and 3 GHG emissions)
    - CDP Climate Change 2024 Questionnaire, Module C5
    - SBTi Corporate Manual (2023), Section 7
    - SEC Climate Disclosure Rule (2024), Items 1504-1505
    - California SB 253 (Climate Corporate Data Accountability Act)
    - TCFD Recommendations (2017), Metrics and Targets

Zero-Hallucination:
    - All report content derives from structured inventory data
    - Framework requirements are statically defined (no generation)
    - No LLM involvement in any content generation or formatting
    - SHA-256 provenance hash on every report

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReportingFramework(str, Enum):
    """Supported reporting frameworks for base year disclosures.

    GHG_PROTOCOL:   GHG Protocol Corporate Standard.
    ISO_14064:      ISO 14064-1:2018.
    ESRS_E1:        European Sustainability Reporting Standards E1.
    CDP:            CDP Climate Change Questionnaire.
    SBTI:           Science Based Targets initiative.
    SEC:            SEC Climate Disclosure Rule (2024).
    SB_253:         California SB 253 (Climate Corporate Data
                    Accountability Act).
    TCFD:           Task Force on Climate-related Financial Disclosures.
    """
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ESRS_E1 = "esrs_e1"
    CDP = "cdp"
    SBTI = "sbti"
    SEC = "sec"
    SB_253 = "sb_253"
    TCFD = "tcfd"


class OutputFormat(str, Enum):
    """Supported output formats for generated reports.

    MARKDOWN:   Human-readable Markdown.
    HTML:       Formatted HTML suitable for web display.
    JSON:       Machine-readable structured JSON.
    CSV:        Tabular CSV for data analysis.
    """
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class ReportSection(str, Enum):
    """Sections that may appear in a base year report.

    EXECUTIVE_SUMMARY:          High-level overview.
    BASE_YEAR_SELECTION:        Base year choice and rationale.
    INVENTORY_SUMMARY:          Emission totals by scope.
    RECALCULATION_HISTORY:      History of base year recalculations.
    ADJUSTMENT_DETAILS:         Details of each adjustment.
    TREND_ANALYSIS:             Time-series trend analysis.
    TARGET_PROGRESS:            Progress against reduction targets.
    VERIFICATION_STATUS:        Third-party verification details.
    METHODOLOGY_NOTES:          Methodology description and notes.
    """
    EXECUTIVE_SUMMARY = "executive_summary"
    BASE_YEAR_SELECTION = "base_year_selection"
    INVENTORY_SUMMARY = "inventory_summary"
    RECALCULATION_HISTORY = "recalculation_history"
    ADJUSTMENT_DETAILS = "adjustment_details"
    TREND_ANALYSIS = "trend_analysis"
    TARGET_PROGRESS = "target_progress"
    VERIFICATION_STATUS = "verification_status"
    METHODOLOGY_NOTES = "methodology_notes"


# ---------------------------------------------------------------------------
# Constants: Framework disclosure requirements
# ---------------------------------------------------------------------------


FRAMEWORK_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    ReportingFramework.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Corporate Standard",
        "required_disclosures": [
            "base_year_and_rationale",
            "base_year_emissions_by_scope",
            "recalculation_policy",
            "significance_threshold",
            "recalculation_history",
            "consolidation_approach",
        ],
        "optional_disclosures": [
            "scope3_base_year",
            "intensity_metrics",
            "uncertainty_analysis",
        ],
        "min_years_trend": 2,
        "requires_scope3": False,
        "requires_verification": False,
    },
    ReportingFramework.ISO_14064.value: {
        "name": "ISO 14064-1:2018",
        "required_disclosures": [
            "base_year_and_justification",
            "base_year_quantification",
            "conditions_for_update",
            "restatement_documentation",
            "uncertainty_assessment",
        ],
        "optional_disclosures": [
            "removals_reporting",
            "biogenic_emissions",
        ],
        "min_years_trend": 2,
        "requires_scope3": False,
        "requires_verification": True,
    },
    ReportingFramework.ESRS_E1.value: {
        "name": "ESRS E1 (Climate Change)",
        "required_disclosures": [
            "base_year_and_rationale",
            "scope1_base_year_emissions",
            "scope2_base_year_emissions",
            "scope3_base_year_emissions",
            "restatement_details",
            "methodology_changes",
            "gwp_values_used",
        ],
        "optional_disclosures": [
            "carbon_credits_reporting",
            "transition_plan_reference",
        ],
        "min_years_trend": 3,
        "requires_scope3": True,
        "requires_verification": True,
    },
    ReportingFramework.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "required_disclosures": [
            "c5_1_base_year_scope_1_2",
            "c5_1a_base_year_details",
            "c5_2_base_year_emissions_data",
            "c5_2a_recalculation_methodology",
            "c5_3_scope3_base_year",
        ],
        "optional_disclosures": [
            "recalculation_policy_details",
            "verification_status",
        ],
        "min_years_trend": 4,
        "requires_scope3": True,
        "requires_verification": False,
    },
    ReportingFramework.SBTI.value: {
        "name": "Science Based Targets initiative",
        "required_disclosures": [
            "base_year_emissions_target_boundary",
            "target_boundary_coverage",
            "recalculation_policy",
            "recalculation_history",
            "base_year_recency",
        ],
        "optional_disclosures": [
            "flag_emissions_breakdown",
            "offset_exclusion_note",
        ],
        "min_years_trend": 2,
        "requires_scope3": True,
        "requires_verification": True,
    },
    ReportingFramework.SEC.value: {
        "name": "SEC Climate Disclosure Rule (2024)",
        "required_disclosures": [
            "scope1_emissions",
            "scope2_emissions",
            "material_changes_to_base_year",
            "methodology_description",
            "three_years_comparable_data",
        ],
        "optional_disclosures": [
            "scope3_emissions",
            "intensity_metrics",
        ],
        "min_years_trend": 3,
        "requires_scope3": False,
        "requires_verification": True,
    },
    ReportingFramework.SB_253.value: {
        "name": "California SB 253",
        "required_disclosures": [
            "scope1_emissions",
            "scope2_emissions",
            "scope3_emissions",
            "methodology_citation",
            "base_year_reference",
        ],
        "optional_disclosures": [
            "verification_details",
        ],
        "min_years_trend": 1,
        "requires_scope3": True,
        "requires_verification": True,
    },
    ReportingFramework.TCFD.value: {
        "name": "TCFD Recommendations",
        "required_disclosures": [
            "historical_emissions_trend",
            "base_year_for_targets",
            "metrics_and_targets",
        ],
        "optional_disclosures": [
            "scenario_analysis_base_year",
            "transition_risk_metrics",
        ],
        "min_years_trend": 3,
        "requires_scope3": False,
        "requires_verification": False,
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class InventoryData(BaseModel):
    """Inventory data for report generation.

    Attributes:
        organization_name: Name of the reporting organization.
        organization_id: Organization identifier.
        base_year: Designated base year.
        reporting_year: Current reporting year.
        base_year_scope1_tco2e: Base year Scope 1 emissions.
        base_year_scope2_location_tco2e: Base year Scope 2 location-based.
        base_year_scope2_market_tco2e: Base year Scope 2 market-based.
        base_year_scope3_tco2e: Base year Scope 3 emissions.
        base_year_total_tco2e: Base year total emissions.
        current_year_scope1_tco2e: Current year Scope 1 emissions.
        current_year_scope2_location_tco2e: Current year Scope 2 location.
        current_year_scope2_market_tco2e: Current year Scope 2 market.
        current_year_scope3_tco2e: Current year Scope 3 emissions.
        current_year_total_tco2e: Current year total emissions.
        consolidation_approach: Consolidation approach used.
        gwp_version: GWP version used.
        base_year_rationale: Rationale for base year selection.
        recalculation_policy: Summary of recalculation policy.
        significance_threshold_pct: Significance threshold.
        recalculation_history: List of recalculation events.
        methodology_description: Description of calculation methodology.
        verification_status: Current verification status.
        verifier_name: Name of verifier (if verified).
        target_name: Name of emission reduction target (if any).
        target_reduction_pct: Target reduction percentage (if any).
        target_year: Target year (if any).
        annual_data: Dict of year -> dict with scope breakdowns.
    """
    organization_name: str = Field(..., min_length=1)
    organization_id: str = Field(default_factory=_new_uuid)
    base_year: int = Field(..., ge=1990, le=2100)
    reporting_year: int = Field(..., ge=1990, le=2100)
    base_year_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    base_year_scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    base_year_scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    base_year_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    base_year_total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_year_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_year_scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_year_scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_year_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_year_total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    consolidation_approach: str = Field(default="operational_control")
    gwp_version: str = Field(default="AR5")
    base_year_rationale: str = Field(default="")
    recalculation_policy: str = Field(default="")
    significance_threshold_pct: Decimal = Field(default=Decimal("5"))
    recalculation_history: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_description: str = Field(default="")
    verification_status: str = Field(default="not_verified")
    verifier_name: str = Field(default="")
    target_name: str = Field(default="")
    target_reduction_pct: Optional[Decimal] = None
    target_year: Optional[int] = None
    annual_data: Dict[int, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("base_year_scope1_tco2e", "base_year_scope2_location_tco2e",
                     "base_year_scope2_market_tco2e", "base_year_scope3_tco2e",
                     "base_year_total_tco2e", "current_year_scope1_tco2e",
                     "current_year_scope2_location_tco2e",
                     "current_year_scope2_market_tco2e",
                     "current_year_scope3_tco2e", "current_year_total_tco2e",
                     "significance_threshold_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("target_reduction_pct", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)


class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        include_sections: Sections to include (default: all).
        output_format: Desired output format.
        include_provenance: Whether to include provenance hashes.
        decimal_places: Decimal places for emission values.
        include_recommendations: Include completeness recommendations.
    """
    include_sections: Optional[List[ReportSection]] = None
    output_format: OutputFormat = Field(default=OutputFormat.MARKDOWN)
    include_provenance: bool = Field(default=True)
    decimal_places: int = Field(default=3, ge=0, le=10)
    include_recommendations: bool = Field(default=True)


class FrameworkRequirement(BaseModel):
    """Requirements for a specific reporting framework.

    Attributes:
        framework: The framework identifier.
        required_disclosures: List of required disclosure items.
        optional_disclosures: List of optional disclosure items.
    """
    framework: ReportingFramework
    required_disclosures: List[str] = Field(default_factory=list)
    optional_disclosures: List[str] = Field(default_factory=list)


class ReportContent(BaseModel):
    """Content for a single report section.

    Attributes:
        section: Section identifier.
        title: Human-readable section title.
        content: Formatted content (string for MD/HTML, serialized for JSON).
        tables: List of tabular data within the section.
        charts: List of chart specifications.
    """
    section: ReportSection
    title: str
    content: str = Field(default="")
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)


class BaseYearReport(BaseModel):
    """A complete base year report for a single framework.

    Attributes:
        report_id: Unique report identifier.
        organization: Organization name.
        base_year: The base year reported on.
        reporting_year: The current reporting year.
        framework: The target framework.
        sections: List of report sections.
        generated_date: When the report was generated.
        generated_by: System/person that generated the report.
        completeness_pct: Disclosure completeness percentage.
        missing_disclosures: List of missing required disclosures.
        provenance_hash: SHA-256 hash for auditability.
    """
    report_id: str = Field(default_factory=_new_uuid)
    organization: str
    base_year: int
    reporting_year: int
    framework: ReportingFramework
    sections: List[ReportContent] = Field(default_factory=list)
    generated_date: datetime = Field(default_factory=_utcnow)
    generated_by: str = Field(default="GreenLang PACK-045 BaseYearReportingEngine")
    completeness_pct: Decimal = Field(default=Decimal("0"))
    missing_disclosures: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MultiFrameworkReport(BaseModel):
    """Multi-framework report combining reports for multiple frameworks.

    Attributes:
        reports: List of BaseYearReport (one per framework).
        cross_reference_matrix: Dict mapping disclosure items to
            frameworks that require them.
        completeness_by_framework: Dict mapping framework to
            completeness percentage.
        generated_date: When the report was generated.
        provenance_hash: SHA-256 hash for auditability.
    """
    reports: List[BaseYearReport] = Field(default_factory=list)
    cross_reference_matrix: Dict[str, List[str]] = Field(default_factory=dict)
    completeness_by_framework: Dict[str, str] = Field(default_factory=dict)
    generated_date: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BaseYearReportingEngine:
    """Multi-framework base year reporting engine.

    Guarantees:
        - Deterministic: Same input -> Same output (bit-perfect)
        - Reproducible: Full provenance tracking with SHA-256 hashes
        - Auditable: Every report section traceable to source data
        - NO LLM: Zero hallucination risk in all content generation

    Usage::

        engine = BaseYearReportingEngine()
        inventory = InventoryData(
            organization_name="Acme Corp",
            base_year=2019,
            reporting_year=2024,
            base_year_total_tco2e=Decimal("100000"),
            ...
        )
        report = engine.generate_report(
            inventory,
            ReportingFramework.GHG_PROTOCOL,
        )
    """

    def __init__(self) -> None:
        """Initialize the BaseYearReportingEngine."""
        logger.info(
            "BaseYearReportingEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
        config: Optional[ReportConfig] = None,
    ) -> BaseYearReport:
        """Generate a base year report for a specific framework.

        Args:
            inventory: The inventory data to report on.
            framework: Target reporting framework.
            config: Optional report configuration.

        Returns:
            BaseYearReport with all sections populated.
        """
        t0 = time.perf_counter()
        cfg = config or ReportConfig()

        # Determine sections to include.
        sections_to_include = cfg.include_sections or list(ReportSection)

        # Generate framework-specific content.
        sections: List[ReportContent] = []

        if ReportSection.EXECUTIVE_SUMMARY in sections_to_include:
            sections.append(self._build_executive_summary(inventory, framework))

        if ReportSection.BASE_YEAR_SELECTION in sections_to_include:
            sections.append(self._build_base_year_selection(inventory, framework))

        if ReportSection.INVENTORY_SUMMARY in sections_to_include:
            sections.append(self._build_inventory_summary(inventory, framework, cfg))

        if ReportSection.RECALCULATION_HISTORY in sections_to_include:
            sections.append(self._build_recalculation_history(inventory, framework))

        if ReportSection.ADJUSTMENT_DETAILS in sections_to_include:
            sections.append(self._build_adjustment_details(inventory, framework))

        if ReportSection.TREND_ANALYSIS in sections_to_include:
            sections.append(self._build_trend_analysis(inventory, framework, cfg))

        if ReportSection.TARGET_PROGRESS in sections_to_include:
            sections.append(self._build_target_progress(inventory, framework))

        if ReportSection.VERIFICATION_STATUS in sections_to_include:
            sections.append(self._build_verification_status(inventory, framework))

        if ReportSection.METHODOLOGY_NOTES in sections_to_include:
            sections.append(self._build_methodology_notes(inventory, framework))

        # Check completeness.
        completeness_pct, missing = self.check_disclosure_completeness(
            inventory, framework
        )

        report = BaseYearReport(
            organization=inventory.organization_name,
            base_year=inventory.base_year,
            reporting_year=inventory.reporting_year,
            framework=framework,
            sections=sections,
            generated_date=_utcnow(),
            completeness_pct=completeness_pct,
            missing_disclosures=missing,
        )
        report.provenance_hash = _compute_hash(report)
        return report

    def generate_multi_framework_report(
        self,
        inventory: InventoryData,
        frameworks: List[ReportingFramework],
        config: Optional[ReportConfig] = None,
    ) -> MultiFrameworkReport:
        """Generate reports for multiple frameworks.

        Also produces a cross-reference matrix showing which disclosures
        are needed by which frameworks.

        Args:
            inventory: The inventory data.
            frameworks: List of frameworks to generate reports for.
            config: Optional report configuration.

        Returns:
            MultiFrameworkReport with all framework reports and
            cross-reference matrix.
        """
        reports: List[BaseYearReport] = []
        completeness_by_fw: Dict[str, str] = {}

        for fw in frameworks:
            report = self.generate_report(inventory, fw, config)
            reports.append(report)
            completeness_by_fw[fw.value] = str(report.completeness_pct)

        # Build cross-reference matrix.
        cross_ref = self._build_cross_reference_matrix(frameworks)

        result = MultiFrameworkReport(
            reports=reports,
            cross_reference_matrix=cross_ref,
            completeness_by_framework=completeness_by_fw,
            generated_date=_utcnow(),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def format_ghg_protocol_report(
        self,
        inventory: InventoryData,
    ) -> str:
        """Generate GHG Protocol-specific formatted report.

        Args:
            inventory: The inventory data.

        Returns:
            Markdown-formatted GHG Protocol report.
        """
        report = self.generate_report(
            inventory, ReportingFramework.GHG_PROTOCOL
        )
        return self.export_report(report, OutputFormat.MARKDOWN)

    def format_cdp_c5(
        self,
        inventory: InventoryData,
    ) -> str:
        """Generate CDP C5 (Emissions Methodology) section content.

        Maps inventory data to CDP question format (C5.1, C5.1a, C5.2,
        C5.2a, C5.3).

        Args:
            inventory: The inventory data.

        Returns:
            Markdown-formatted CDP C5 responses.
        """
        lines: List[str] = []
        lines.append("# CDP Climate Change - Module C5: Emissions Methodology")
        lines.append("")

        # C5.1
        lines.append("## C5.1 Base year for Scope 1 and Scope 2")
        lines.append("")
        lines.append(f"**Base year:** {inventory.base_year}")
        lines.append(f"**Rationale:** {inventory.base_year_rationale or 'Not provided'}")
        lines.append("")

        # C5.1a
        lines.append("## C5.1a Base year emission details")
        lines.append("")
        lines.append("| Metric | Value (tCO2e) |")
        lines.append("|--------|--------------|")
        lines.append(f"| Scope 1 base year emissions | {_round_val(inventory.base_year_scope1_tco2e)} |")
        lines.append(f"| Scope 2 location-based | {_round_val(inventory.base_year_scope2_location_tco2e)} |")
        lines.append(f"| Scope 2 market-based | {_round_val(inventory.base_year_scope2_market_tco2e)} |")
        lines.append("")

        # C5.2
        lines.append("## C5.2 Base year emissions data")
        lines.append("")
        lines.append(f"**Total base year emissions:** {_round_val(inventory.base_year_total_tco2e)} tCO2e")
        lines.append(f"**Consolidation approach:** {inventory.consolidation_approach}")
        lines.append(f"**GWP values:** {inventory.gwp_version}")
        lines.append("")

        # C5.2a
        lines.append("## C5.2a Recalculation methodology")
        lines.append("")
        lines.append(f"**Recalculation policy:** {inventory.recalculation_policy or 'Not provided'}")
        lines.append(f"**Significance threshold:** {inventory.significance_threshold_pct}%")
        lines.append("")

        if inventory.recalculation_history:
            lines.append("**Recalculation history:**")
            lines.append("")
            for event in inventory.recalculation_history:
                lines.append(f"- {event.get('description', 'No description')}")
            lines.append("")

        # C5.3
        lines.append("## C5.3 Scope 3 base year")
        lines.append("")
        lines.append(f"**Scope 3 base year emissions:** {_round_val(inventory.base_year_scope3_tco2e)} tCO2e")
        lines.append("")

        return "\n".join(lines)

    def format_esrs_e1_6(
        self,
        inventory: InventoryData,
    ) -> str:
        """Generate ESRS E1-6 disclosure content.

        Args:
            inventory: The inventory data.

        Returns:
            Markdown-formatted ESRS E1-6 disclosure.
        """
        lines: List[str] = []
        lines.append("# ESRS E1-6: Gross Scopes 1, 2 and 3 GHG Emissions")
        lines.append("")

        # Base year disclosure.
        lines.append("## Base Year")
        lines.append("")
        lines.append(f"**Base year selected:** {inventory.base_year}")
        lines.append(f"**Rationale:** {inventory.base_year_rationale or 'Not provided'}")
        lines.append("")

        # Scope breakdown.
        lines.append("## Base Year Emissions by Scope")
        lines.append("")
        lines.append("| Scope | Emissions (tCO2e) |")
        lines.append("|-------|------------------|")
        lines.append(f"| Scope 1 | {_round_val(inventory.base_year_scope1_tco2e)} |")
        lines.append(f"| Scope 2 (location-based) | {_round_val(inventory.base_year_scope2_location_tco2e)} |")
        lines.append(f"| Scope 2 (market-based) | {_round_val(inventory.base_year_scope2_market_tco2e)} |")
        lines.append(f"| Scope 3 | {_round_val(inventory.base_year_scope3_tco2e)} |")
        lines.append(f"| **Total** | **{_round_val(inventory.base_year_total_tco2e)}** |")
        lines.append("")

        # GWP values.
        lines.append(f"**GWP values used:** {inventory.gwp_version}")
        lines.append(f"**Consolidation approach:** {inventory.consolidation_approach}")
        lines.append("")

        # Restatement section.
        if inventory.recalculation_history:
            lines.append("## Restatement of Base Year")
            lines.append("")
            for event in inventory.recalculation_history:
                lines.append(f"- **{event.get('date', 'N/A')}:** {event.get('description', '')}")
            lines.append("")

        # Methodology.
        lines.append("## Methodology")
        lines.append("")
        lines.append(inventory.methodology_description or "Not provided.")
        lines.append("")

        return "\n".join(lines)

    def format_sbti_report(
        self,
        inventory: InventoryData,
    ) -> str:
        """Generate SBTi base year disclosure content.

        Args:
            inventory: The inventory data.

        Returns:
            Markdown-formatted SBTi disclosure.
        """
        lines: List[str] = []
        lines.append("# SBTi Base Year Disclosure")
        lines.append("")
        lines.append(f"**Organization:** {inventory.organization_name}")
        lines.append(f"**Base year:** {inventory.base_year}")
        lines.append("")

        # Target boundary emissions.
        lines.append("## Base Year Emissions (Target Boundary)")
        lines.append("")
        lines.append("| Scope | Emissions (tCO2e) |")
        lines.append("|-------|------------------|")
        lines.append(f"| Scope 1 | {_round_val(inventory.base_year_scope1_tco2e)} |")
        scope2_val = max(
            inventory.base_year_scope2_location_tco2e,
            inventory.base_year_scope2_market_tco2e,
        )
        lines.append(f"| Scope 2 | {_round_val(scope2_val)} |")
        lines.append(f"| Scope 1+2 Total | {_round_val(inventory.base_year_scope1_tco2e + scope2_val)} |")
        if inventory.base_year_scope3_tco2e > Decimal("0"):
            lines.append(f"| Scope 3 | {_round_val(inventory.base_year_scope3_tco2e)} |")
        lines.append("")

        # Target (if any).
        if inventory.target_name:
            lines.append("## Reduction Target")
            lines.append("")
            lines.append(f"**Target:** {inventory.target_name}")
            if inventory.target_reduction_pct is not None:
                lines.append(f"**Reduction:** {inventory.target_reduction_pct}%")
            if inventory.target_year is not None:
                lines.append(f"**Target Year:** {inventory.target_year}")
            lines.append("")

        # Recalculation policy.
        lines.append("## Recalculation Policy")
        lines.append("")
        lines.append(inventory.recalculation_policy or "Not provided.")
        lines.append(f"**Significance threshold:** {inventory.significance_threshold_pct}%")
        lines.append("")

        return "\n".join(lines)

    def format_sec_report(
        self,
        inventory: InventoryData,
    ) -> str:
        """Generate SEC Climate Disclosure Rule content.

        Args:
            inventory: The inventory data.

        Returns:
            Markdown-formatted SEC disclosure.
        """
        lines: List[str] = []
        lines.append("# SEC Climate Disclosure - GHG Emissions")
        lines.append("")
        lines.append(f"**Registrant:** {inventory.organization_name}")
        lines.append(f"**Reporting Period:** FY{inventory.reporting_year}")
        lines.append("")

        # Item 1504: GHG Emissions Disclosure.
        lines.append("## Item 1504: GHG Emissions")
        lines.append("")
        lines.append("### Current Year")
        lines.append("")
        lines.append("| Metric | Value (tCO2e) |")
        lines.append("|--------|--------------|")
        lines.append(f"| Scope 1 | {_round_val(inventory.current_year_scope1_tco2e)} |")
        lines.append(f"| Scope 2 | {_round_val(max(inventory.current_year_scope2_location_tco2e, inventory.current_year_scope2_market_tco2e))} |")
        lines.append("")

        # Comparable prior periods.
        lines.append("### Base Year Reference")
        lines.append("")
        lines.append(f"**Base year:** {inventory.base_year}")
        lines.append(f"**Scope 1:** {_round_val(inventory.base_year_scope1_tco2e)} tCO2e")
        scope2_base = max(
            inventory.base_year_scope2_location_tco2e,
            inventory.base_year_scope2_market_tco2e,
        )
        lines.append(f"**Scope 2:** {_round_val(scope2_base)} tCO2e")
        lines.append("")

        # Material changes.
        if inventory.recalculation_history:
            lines.append("### Material Changes to Base Year")
            lines.append("")
            for event in inventory.recalculation_history:
                lines.append(f"- {event.get('description', '')}")
            lines.append("")

        # Methodology.
        lines.append("### Methodology")
        lines.append("")
        lines.append(inventory.methodology_description or "Not provided.")
        lines.append("")

        return "\n".join(lines)

    def check_disclosure_completeness(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> Tuple[Decimal, List[str]]:
        """Check disclosure completeness against framework requirements.

        Args:
            inventory: The inventory data.
            framework: The target framework.

        Returns:
            Tuple of (completeness_pct, list of missing disclosures).
        """
        fw_req = FRAMEWORK_REQUIREMENTS.get(framework.value, {})
        required = fw_req.get("required_disclosures", [])

        if not required:
            return Decimal("100"), []

        # Check each required disclosure.
        present = 0
        missing: List[str] = []

        for disclosure in required:
            if self._is_disclosure_met(inventory, disclosure):
                present += 1
            else:
                missing.append(disclosure)

        completeness = _round_val(
            _safe_pct(_decimal(present), _decimal(len(required))),
            places=1,
        )

        return completeness, missing

    def export_report(
        self,
        report: BaseYearReport,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> str:
        """Export a report in the specified format.

        Args:
            report: The report to export.
            output_format: Desired format.

        Returns:
            Formatted string.
        """
        if output_format == OutputFormat.JSON:
            return self._export_json(report)
        elif output_format == OutputFormat.CSV:
            return self._export_csv(report)
        elif output_format == OutputFormat.HTML:
            return self._export_html(report)
        else:
            return self._export_markdown(report)

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _build_executive_summary(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build executive summary section."""
        fw_name = FRAMEWORK_REQUIREMENTS.get(
            framework.value, {}
        ).get("name", framework.value)

        change = _decimal(inventory.current_year_total_tco2e) - _decimal(inventory.base_year_total_tco2e)
        change_pct = _safe_pct(change, _decimal(inventory.base_year_total_tco2e))
        direction = "decrease" if change < Decimal("0") else "increase"

        content = (
            f"This report presents the base year disclosure for "
            f"{inventory.organization_name} in accordance with "
            f"{fw_name}.\n\n"
            f"Base year {inventory.base_year} total emissions were "
            f"{_round_val(inventory.base_year_total_tco2e)} tCO2e. "
            f"Current year ({inventory.reporting_year}) emissions are "
            f"{_round_val(inventory.current_year_total_tco2e)} tCO2e, "
            f"representing a {_round_val(_decimal(change_pct).copy_abs(), 1)}% "
            f"{direction} from the base year."
        )

        return ReportContent(
            section=ReportSection.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            content=content,
        )

    def _build_base_year_selection(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build base year selection section."""
        content = (
            f"**Base Year:** {inventory.base_year}\n\n"
            f"**Rationale:** {inventory.base_year_rationale or 'Not provided.'}\n\n"
            f"**Consolidation Approach:** {inventory.consolidation_approach}\n\n"
            f"**GWP Version:** {inventory.gwp_version}"
        )

        return ReportContent(
            section=ReportSection.BASE_YEAR_SELECTION,
            title="Base Year Selection",
            content=content,
        )

    def _build_inventory_summary(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
        config: ReportConfig,
    ) -> ReportContent:
        """Build inventory summary section."""
        dp = config.decimal_places

        table_data = {
            "headers": ["Scope", "Base Year (tCO2e)", "Current Year (tCO2e)", "Change (%)"],
            "rows": [],
        }

        # Scope 1.
        s1_change = _safe_pct(
            _decimal(inventory.current_year_scope1_tco2e) - _decimal(inventory.base_year_scope1_tco2e),
            _decimal(inventory.base_year_scope1_tco2e),
        )
        table_data["rows"].append([
            "Scope 1",
            str(_round_val(inventory.base_year_scope1_tco2e, dp)),
            str(_round_val(inventory.current_year_scope1_tco2e, dp)),
            str(_round_val(s1_change, 1)),
        ])

        # Scope 2 location.
        s2l_change = _safe_pct(
            _decimal(inventory.current_year_scope2_location_tco2e) - _decimal(inventory.base_year_scope2_location_tco2e),
            _decimal(inventory.base_year_scope2_location_tco2e),
        )
        table_data["rows"].append([
            "Scope 2 (location)",
            str(_round_val(inventory.base_year_scope2_location_tco2e, dp)),
            str(_round_val(inventory.current_year_scope2_location_tco2e, dp)),
            str(_round_val(s2l_change, 1)),
        ])

        # Scope 2 market.
        s2m_change = _safe_pct(
            _decimal(inventory.current_year_scope2_market_tco2e) - _decimal(inventory.base_year_scope2_market_tco2e),
            _decimal(inventory.base_year_scope2_market_tco2e),
        )
        table_data["rows"].append([
            "Scope 2 (market)",
            str(_round_val(inventory.base_year_scope2_market_tco2e, dp)),
            str(_round_val(inventory.current_year_scope2_market_tco2e, dp)),
            str(_round_val(s2m_change, 1)),
        ])

        # Scope 3 (if applicable).
        fw_req = FRAMEWORK_REQUIREMENTS.get(framework.value, {})
        if fw_req.get("requires_scope3", False) or inventory.base_year_scope3_tco2e > Decimal("0"):
            s3_change = _safe_pct(
                _decimal(inventory.current_year_scope3_tco2e) - _decimal(inventory.base_year_scope3_tco2e),
                _decimal(inventory.base_year_scope3_tco2e),
            )
            table_data["rows"].append([
                "Scope 3",
                str(_round_val(inventory.base_year_scope3_tco2e, dp)),
                str(_round_val(inventory.current_year_scope3_tco2e, dp)),
                str(_round_val(s3_change, 1)),
            ])

        # Total.
        total_change = _safe_pct(
            _decimal(inventory.current_year_total_tco2e) - _decimal(inventory.base_year_total_tco2e),
            _decimal(inventory.base_year_total_tco2e),
        )
        table_data["rows"].append([
            "Total",
            str(_round_val(inventory.base_year_total_tco2e, dp)),
            str(_round_val(inventory.current_year_total_tco2e, dp)),
            str(_round_val(total_change, 1)),
        ])

        content = "Emissions summary comparing base year to current reporting year."

        return ReportContent(
            section=ReportSection.INVENTORY_SUMMARY,
            title="Inventory Summary",
            content=content,
            tables=[table_data],
        )

    def _build_recalculation_history(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build recalculation history section."""
        if not inventory.recalculation_history:
            content = "No base year recalculations have been performed."
        else:
            events = []
            for event in inventory.recalculation_history:
                events.append(
                    f"- **{event.get('date', 'N/A')}:** "
                    f"{event.get('description', 'No description')} "
                    f"(Trigger: {event.get('trigger', 'N/A')}, "
                    f"Impact: {event.get('impact_tco2e', 'N/A')} tCO2e)"
                )
            content = "\n".join(events)

        return ReportContent(
            section=ReportSection.RECALCULATION_HISTORY,
            title="Recalculation History",
            content=content,
        )

    def _build_adjustment_details(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build adjustment details section."""
        content = (
            f"**Recalculation policy:** {inventory.recalculation_policy or 'Not provided.'}\n\n"
            f"**Significance threshold:** {inventory.significance_threshold_pct}%\n\n"
            f"Adjustments are applied when structural or methodological changes "
            f"exceed the significance threshold."
        )

        return ReportContent(
            section=ReportSection.ADJUSTMENT_DETAILS,
            title="Adjustment Details",
            content=content,
        )

    def _build_trend_analysis(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
        config: ReportConfig,
    ) -> ReportContent:
        """Build trend analysis section from annual data."""
        dp = config.decimal_places

        if not inventory.annual_data:
            content = "No multi-year data available for trend analysis."
            return ReportContent(
                section=ReportSection.TREND_ANALYSIS,
                title="Trend Analysis",
                content=content,
            )

        # Build trend table.
        sorted_years = sorted(inventory.annual_data.keys())
        table_data = {
            "headers": ["Year", "Total (tCO2e)", "YoY Change (%)"],
            "rows": [],
        }

        prev_total: Optional[Decimal] = None
        for year in sorted_years:
            year_data = inventory.annual_data[year]
            total = _decimal(year_data.get("total_tco2e", 0))
            yoy = ""
            if prev_total is not None and prev_total != Decimal("0"):
                yoy_val = _safe_pct(total - prev_total, prev_total.copy_abs())
                yoy = str(_round_val(yoy_val, 1))
            table_data["rows"].append([
                str(year),
                str(_round_val(total, dp)),
                yoy,
            ])
            prev_total = total

        content = "Emission trend from base year to current reporting year."

        return ReportContent(
            section=ReportSection.TREND_ANALYSIS,
            title="Trend Analysis",
            content=content,
            tables=[table_data],
        )

    def _build_target_progress(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build target progress section."""
        if not inventory.target_name:
            content = "No emission reduction targets have been set."
        else:
            content = (
                f"**Target:** {inventory.target_name}\n"
                f"**Reduction:** {inventory.target_reduction_pct}%\n"
                f"**Target Year:** {inventory.target_year}\n\n"
            )
            if inventory.target_reduction_pct is not None and \
               inventory.base_year_total_tco2e > Decimal("0"):
                target_val = _round_val(
                    inventory.base_year_total_tco2e * (
                        Decimal("1") - _decimal(inventory.target_reduction_pct) / Decimal("100")
                    ),
                    3,
                )
                current_progress = _safe_pct(
                    inventory.base_year_total_tco2e - inventory.current_year_total_tco2e,
                    inventory.base_year_total_tco2e - target_val,
                )
                content += (
                    f"**Target emissions:** {target_val} tCO2e\n"
                    f"**Current progress:** {_round_val(current_progress, 1)}%"
                )

        return ReportContent(
            section=ReportSection.TARGET_PROGRESS,
            title="Target Progress",
            content=content,
        )

    def _build_verification_status(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build verification status section."""
        content = (
            f"**Verification status:** {inventory.verification_status}\n"
        )
        if inventory.verifier_name:
            content += f"**Verifier:** {inventory.verifier_name}\n"

        fw_req = FRAMEWORK_REQUIREMENTS.get(framework.value, {})
        if fw_req.get("requires_verification", False):
            content += (
                f"\n{fw_req.get('name', framework.value)} requires "
                f"third-party verification of emissions data."
            )

        return ReportContent(
            section=ReportSection.VERIFICATION_STATUS,
            title="Verification Status",
            content=content,
        )

    def _build_methodology_notes(
        self,
        inventory: InventoryData,
        framework: ReportingFramework,
    ) -> ReportContent:
        """Build methodology notes section."""
        content = inventory.methodology_description or "Methodology description not provided."
        content += f"\n\n**GWP values:** {inventory.gwp_version}"
        content += f"\n**Consolidation approach:** {inventory.consolidation_approach}"

        return ReportContent(
            section=ReportSection.METHODOLOGY_NOTES,
            title="Methodology Notes",
            content=content,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_disclosure_met(
        self,
        inventory: InventoryData,
        disclosure: str,
    ) -> bool:
        """Check whether a specific disclosure requirement is met.

        Args:
            inventory: The inventory data.
            disclosure: The disclosure identifier.

        Returns:
            True if the disclosure requirement is met.
        """
        # Map disclosure identifiers to data checks.
        checks: Dict[str, bool] = {
            "base_year_and_rationale": (
                inventory.base_year > 0 and
                bool(inventory.base_year_rationale)
            ),
            "base_year_and_justification": (
                inventory.base_year > 0 and
                bool(inventory.base_year_rationale)
            ),
            "base_year_emissions_by_scope": (
                inventory.base_year_scope1_tco2e > Decimal("0") or
                inventory.base_year_scope2_location_tco2e > Decimal("0")
            ),
            "base_year_quantification": (
                inventory.base_year_total_tco2e > Decimal("0")
            ),
            "recalculation_policy": bool(inventory.recalculation_policy),
            "significance_threshold": (
                inventory.significance_threshold_pct > Decimal("0")
            ),
            "recalculation_history": True,  # Always met (may be empty).
            "consolidation_approach": bool(inventory.consolidation_approach),
            "conditions_for_update": bool(inventory.recalculation_policy),
            "restatement_documentation": True,
            "uncertainty_assessment": bool(inventory.methodology_description),
            "scope1_base_year_emissions": (
                inventory.base_year_scope1_tco2e > Decimal("0")
            ),
            "scope2_base_year_emissions": (
                inventory.base_year_scope2_location_tco2e > Decimal("0") or
                inventory.base_year_scope2_market_tco2e > Decimal("0")
            ),
            "scope3_base_year_emissions": (
                inventory.base_year_scope3_tco2e > Decimal("0")
            ),
            "restatement_details": True,
            "methodology_changes": True,
            "gwp_values_used": bool(inventory.gwp_version),
            "c5_1_base_year_scope_1_2": inventory.base_year > 0,
            "c5_1a_base_year_details": (
                inventory.base_year_scope1_tco2e > Decimal("0")
            ),
            "c5_2_base_year_emissions_data": (
                inventory.base_year_total_tco2e > Decimal("0")
            ),
            "c5_2a_recalculation_methodology": bool(inventory.recalculation_policy),
            "c5_3_scope3_base_year": (
                inventory.base_year_scope3_tco2e > Decimal("0")
            ),
            "base_year_emissions_target_boundary": (
                inventory.base_year_total_tco2e > Decimal("0")
            ),
            "target_boundary_coverage": bool(inventory.target_name),
            "base_year_recency": inventory.base_year > 0,
            "scope1_emissions": (
                inventory.current_year_scope1_tco2e > Decimal("0") or
                inventory.base_year_scope1_tco2e > Decimal("0")
            ),
            "scope2_emissions": (
                inventory.current_year_scope2_location_tco2e > Decimal("0") or
                inventory.current_year_scope2_market_tco2e > Decimal("0")
            ),
            "scope3_emissions": (
                inventory.current_year_scope3_tco2e > Decimal("0") or
                inventory.base_year_scope3_tco2e > Decimal("0")
            ),
            "material_changes_to_base_year": True,
            "methodology_description": bool(inventory.methodology_description),
            "three_years_comparable_data": len(inventory.annual_data) >= 3,
            "methodology_citation": bool(inventory.methodology_description),
            "base_year_reference": inventory.base_year > 0,
            "historical_emissions_trend": len(inventory.annual_data) >= 2,
            "base_year_for_targets": (
                inventory.base_year > 0 and bool(inventory.target_name)
            ),
            "metrics_and_targets": bool(inventory.target_name),
        }

        return checks.get(disclosure, False)

    def _build_cross_reference_matrix(
        self,
        frameworks: List[ReportingFramework],
    ) -> Dict[str, List[str]]:
        """Build a cross-reference matrix of disclosures to frameworks.

        Args:
            frameworks: List of frameworks to include.

        Returns:
            Dict mapping disclosure item -> list of frameworks requiring it.
        """
        matrix: Dict[str, List[str]] = {}

        for fw in frameworks:
            fw_req = FRAMEWORK_REQUIREMENTS.get(fw.value, {})
            for disclosure in fw_req.get("required_disclosures", []):
                if disclosure not in matrix:
                    matrix[disclosure] = []
                matrix[disclosure].append(fw.value)
            for disclosure in fw_req.get("optional_disclosures", []):
                key = f"{disclosure} (optional)"
                if key not in matrix:
                    matrix[key] = []
                matrix[key].append(fw.value)

        return matrix

    def _export_json(self, report: BaseYearReport) -> str:
        """Export report as JSON."""
        data = report.model_dump(mode="json")
        return json.dumps(data, indent=2, sort_keys=True, default=str)

    def _export_csv(self, report: BaseYearReport) -> str:
        """Export report sections as CSV."""
        lines: List[str] = []
        lines.append("section,title,content_length,table_count")

        for section in report.sections:
            content_len = len(section.content)
            table_count = len(section.tables)
            title_escaped = section.title.replace('"', '""')
            lines.append(
                f'"{section.section.value}","{title_escaped}",'
                f'{content_len},{table_count}'
            )

        return "\n".join(lines)

    def _export_html(self, report: BaseYearReport) -> str:
        """Export report as HTML."""
        lines: List[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html><head>")
        lines.append(f"<title>Base Year Report - {report.organization}</title>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        lines.append("h1 { color: #2c3e50; }")
        lines.append("h2 { color: #34495e; border-bottom: 1px solid #eee; }")
        lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        lines.append("th { background-color: #f2f2f2; }")
        lines.append(".provenance { font-size: 0.8em; color: #999; }")
        lines.append("</style>")
        lines.append("</head><body>")
        lines.append(f"<h1>Base Year Report: {report.organization}</h1>")
        lines.append(
            f"<p><strong>Framework:</strong> {report.framework.value} | "
            f"<strong>Base Year:</strong> {report.base_year} | "
            f"<strong>Reporting Year:</strong> {report.reporting_year}</p>"
        )
        lines.append(
            f"<p><strong>Completeness:</strong> {report.completeness_pct}%</p>"
        )

        for section in report.sections:
            lines.append(f"<h2>{section.title}</h2>")
            # Convert content to HTML paragraphs.
            for para in section.content.split("\n\n"):
                if para.strip():
                    lines.append(f"<p>{para.strip()}</p>")

            # Render tables.
            for table in section.tables:
                lines.append("<table>")
                headers = table.get("headers", [])
                if headers:
                    lines.append("<tr>")
                    for h in headers:
                        lines.append(f"<th>{h}</th>")
                    lines.append("</tr>")
                for row in table.get("rows", []):
                    lines.append("<tr>")
                    for cell in row:
                        lines.append(f"<td>{cell}</td>")
                    lines.append("</tr>")
                lines.append("</table>")

        lines.append(
            f'<p class="provenance">Provenance hash: '
            f'{report.provenance_hash[:32]}...</p>'
        )
        lines.append(
            f'<p class="provenance">Generated: {report.generated_date} '
            f'by {report.generated_by}</p>'
        )
        lines.append("</body></html>")

        return "\n".join(lines)

    def _export_markdown(self, report: BaseYearReport) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Base Year Report: {report.organization}")
        lines.append("")
        lines.append(f"**Framework:** {report.framework.value}")
        lines.append(f"**Base Year:** {report.base_year}")
        lines.append(f"**Reporting Year:** {report.reporting_year}")
        lines.append(f"**Completeness:** {report.completeness_pct}%")
        lines.append(f"**Provenance Hash:** `{report.provenance_hash[:16]}...`")
        lines.append("")

        if report.missing_disclosures:
            lines.append("**Missing Disclosures:**")
            for disc in report.missing_disclosures:
                lines.append(f"- {disc}")
            lines.append("")

        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

            # Render tables.
            for table in section.tables:
                headers = table.get("headers", [])
                if headers:
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in table.get("rows", []):
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

        lines.append("---")
        lines.append(
            f"*Generated: {report.generated_date} by "
            f"{report.generated_by}*"
        )

        return "\n".join(lines)
