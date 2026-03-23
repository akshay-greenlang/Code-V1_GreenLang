# -*- coding: utf-8 -*-
"""
Dual Reporting Reconciliation Agent Data Models - AGENT-MRV-013

Pydantic v2 data models for the Dual Reporting Reconciliation Agent SDK
covering GHG Protocol Scope 2 dual-reporting reconciliation between
location-based and market-based accounting methods including:

- Collection of upstream results from 4 Scope 2 agents (MRV-009 through
  MRV-012) covering electricity, steam, district heating, and district
  cooling energy types
- Boundary alignment and energy-type mapping across location-based and
  market-based calculation outputs
- Discrepancy analysis with 8 discrepancy types (REC/GO impact, residual
  mix uplift, supplier EF delta, geographic mismatch, temporal mismatch,
  partial coverage, steam/heat method divergence, grid update timing)
- Waterfall decomposition of total discrepancy into individual drivers
- Quality scoring across 4 dimensions (completeness, consistency,
  accuracy, transparency) with weighted composite scoring and grading
- Emission factor hierarchy quality assessment per GHG Protocol Scope 2
  Guidance (supplier-specific through grid average)
- Multi-framework reporting table generation (GHG Protocol, CSRD/ESRS,
  CDP, SBTi, GRI, ISO 14064, RE100)
- Trend analysis with year-over-year, CAGR, PIF (Procurement Impact
  Factor), and RE100 percentage tracking
- Intensity metrics (revenue, FTE, floor area, production unit)
- Compliance checking against 7 regulatory frameworks
- Residual mix factors for 30+ regions worldwide
- SHA-256 provenance chain for complete audit trails

This agent is a RECONCILIATION agent -- it does NOT calculate emissions.
It collects pre-calculated results from upstream Scope 2 agents and
reconciles location-based versus market-based totals.

Enumerations (22):
    - EnergyType, Scope2Method, UpstreamAgent, DiscrepancyType,
      DiscrepancyDirection, MaterialityLevel, QualityDimension,
      QualityGrade, EFHierarchyPriority, ReportingFramework,
      FlagType, FlagSeverity, ReconciliationStatus, IntensityMetric,
      TrendDirection, PipelineStage, ExportFormat, ComplianceStatus,
      DataQualityTier, GWPSource, EmissionGas, BatchStatus

Constants:
    - MATERIALITY_THRESHOLDS: Percentage bands for materiality levels
    - QUALITY_WEIGHTS: Dimensional weights for composite scoring
    - QUALITY_GRADE_THRESHOLDS: Score thresholds for letter grades
    - EF_HIERARCHY_QUALITY_SCORES: Quality scores per EF hierarchy tier
    - RESIDUAL_MIX_FACTORS: 30+ regions with grid/residual EFs and ratios
    - UPSTREAM_AGENT_MAPPING: Energy type to upstream agent mapping
    - FRAMEWORK_REQUIRED_DISCLOSURES: Per-framework disclosure requirements

Data Models (25):
    - ResidualMixFactor, UpstreamResult, EnergyTypeBreakdown,
      FacilityBreakdown, ReconciliationWorkspace, Discrepancy,
      WaterfallItem, WaterfallDecomposition, DiscrepancyReport,
      Flag, QualityScore, QualityAssessment, FrameworkTable,
      ReportingTableSet, TrendDataPoint, TrendReport,
      IntensityResult, ComplianceRequirement, ComplianceCheckResult,
      ReconciliationRequest, ReconciliationReport,
      BatchReconciliationRequest, BatchReconciliationResult,
      ExportRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-013 Dual Reporting Reconciliation (GL-MRV-X-024)
Status: Production Ready
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

#: Agent identifier for registry integration.
AGENT_ID: str = "GL-MRV-X-024"

#: Agent component identifier.
AGENT_COMPONENT: str = "AGENT-MRV-013"

#: Service version string.
VERSION: str = "1.0.0"

#: Database table prefix for all DRR tables.
TABLE_PREFIX: str = "gl_drr_"

#: Maximum number of upstream results per reconciliation request.
MAX_UPSTREAM_RESULTS: int = 10_000

#: Maximum number of periods in a batch reconciliation.
MAX_BATCH_PERIODS: int = 120

#: Maximum number of facilities per reconciliation.
MAX_FACILITIES: int = 50_000

#: Maximum number of discrepancies per report.
MAX_DISCREPANCIES: int = 5_000

#: Maximum number of frameworks per request.
MAX_FRAMEWORKS: int = 20

#: Maximum number of trend data points.
MAX_TREND_POINTS: int = 240

#: Maximum number of compliance requirements per framework.
MAX_REQUIREMENTS_PER_FRAMEWORK: int = 200

#: Default confidence level for quality thresholds.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("95.0")

#: Positive infinity sentinel for Decimal comparisons.
DECIMAL_INF: Decimal = Decimal("Infinity")

#: Number of decimal places for Decimal quantization.
DECIMAL_PLACES: int = 8

#: Decimal zero constant for arithmetic operations.
ZERO: Decimal = Decimal("0")

#: Decimal one hundred constant for percentage calculations.
ONE_HUNDRED: Decimal = Decimal("100")


# =============================================================================
# Enumerations (22)
# =============================================================================


class EnergyType(str, Enum):
    """Types of energy purchased for Scope 2 emissions reporting.

    GHG Protocol Scope 2 Guidance defines four categories of purchased
    energy that must be reported under both location-based and
    market-based methods.

    ELECTRICITY: Purchased electricity from the grid or direct supply
        agreements. Typically the largest Scope 2 source.
    STEAM: Purchased steam from external suppliers for heating or
        industrial processes.
    DISTRICT_HEATING: Purchased heat from centralised district heating
        networks supplied to multiple buildings.
    DISTRICT_COOLING: Purchased cooling from centralised district
        cooling networks (chilled water systems).
    """

    ELECTRICITY = "electricity"
    STEAM = "steam"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"


class Scope2Method(str, Enum):
    """GHG Protocol Scope 2 accounting methods.

    The GHG Protocol Scope 2 Guidance (2015) requires companies to
    report Scope 2 emissions using two parallel methods. Both totals
    must be disclosed; this agent reconciles the difference.

    LOCATION_BASED: Uses average grid emission factors for the region
        where energy consumption occurs. Reflects the average emissions
        intensity of the local grid regardless of contractual
        instruments.
    MARKET_BASED: Uses emission factors from contractual instruments
        (energy attribute certificates, power purchase agreements,
        supplier-specific emission rates, or residual mix factors).
        Reflects the emissions associated with the specific electricity
        a company has chosen to purchase.
    """

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class UpstreamAgent(str, Enum):
    """Upstream MRV agents that provide Scope 2 calculation results.

    Each agent handles one energy type and produces both location-based
    and market-based emission totals that feed into this reconciliation
    agent.

    MRV_009: Purchased Electricity Agent - handles grid electricity
        consumption, renewable energy certificates (RECs), guarantees
        of origin (GOs), and power purchase agreements (PPAs).
    MRV_010: Purchased Steam Agent - handles steam purchased from
        external suppliers for heating and industrial applications.
    MRV_011: District Heating Agent - handles purchased heat from
        centralised district heating networks.
    MRV_012: District Cooling Agent - handles purchased cooling
        from centralised district cooling networks.
    """

    MRV_009 = "mrv_009"
    MRV_010 = "mrv_010"
    MRV_011 = "mrv_011"
    MRV_012 = "mrv_012"


class DiscrepancyType(str, Enum):
    """Types of discrepancies between location-based and market-based totals.

    When location-based and market-based Scope 2 totals differ, the
    discrepancy can be attributed to one or more of these eight root
    causes. The waterfall decomposition assigns every tonne of
    discrepancy to exactly one type.

    REC_GO_IMPACT: Difference caused by Renewable Energy Certificates
        (RECs) or Guarantees of Origin (GOs) that lower market-based
        emissions to zero for the covered quantity. The location-based
        total still reflects the grid average for those same MWh.
    RESIDUAL_MIX_UPLIFT: Difference caused by using residual mix
        emission factors (which strip out tracked renewables) instead
        of grid average factors. Residual mix EFs are typically higher
        than grid average EFs because renewable generation has been
        contractually claimed and removed from the residual pool.
    SUPPLIER_EF_DELTA: Difference caused by using a supplier-specific
        emission factor that differs from the grid average. This may
        increase or decrease market-based emissions depending on the
        supplier's generation profile.
    GEOGRAPHIC_MISMATCH: Difference arising from facilities where
        the location-based grid region differs from the market-based
        contractual instrument region. For example, purchasing RECs
        from a different grid region than where consumption occurs.
    TEMPORAL_MISMATCH: Difference caused by timing discrepancies
        between the vintage of contractual instruments and the
        reporting period of energy consumption.
    PARTIAL_COVERAGE: Difference arising when contractual instruments
        cover only a portion of total energy consumption, leaving
        the uncovered portion subject to residual mix or grid average
        factors.
    STEAM_HEAT_METHOD: Difference in methodological approach between
        location-based and market-based for steam and heating,
        where the efficiency method or CHP allocation may differ.
    GRID_UPDATE_TIMING: Difference caused by using different vintages
        of grid emission factors in location-based versus market-based
        calculations, such as when grid factors are updated mid-year.
    """

    REC_GO_IMPACT = "rec_go_impact"
    RESIDUAL_MIX_UPLIFT = "residual_mix_uplift"
    SUPPLIER_EF_DELTA = "supplier_ef_delta"
    GEOGRAPHIC_MISMATCH = "geographic_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    PARTIAL_COVERAGE = "partial_coverage"
    STEAM_HEAT_METHOD = "steam_heat_method"
    GRID_UPDATE_TIMING = "grid_update_timing"


class DiscrepancyDirection(str, Enum):
    """Direction of the market-based total relative to location-based.

    Indicates whether market-based emissions are lower, higher, or
    equal compared to location-based emissions.

    MARKET_LOWER: Market-based total is lower than location-based.
        Typical when RECs/GOs or clean-energy PPAs are procured.
    MARKET_HIGHER: Market-based total is higher than location-based.
        Occurs when supplier EFs exceed grid average or residual mix
        factors are significantly higher than grid average.
    EQUAL: Market-based and location-based totals are identical
        within rounding tolerance (< 0.01 tCO2e).
    """

    MARKET_LOWER = "market_lower"
    MARKET_HIGHER = "market_higher"
    EQUAL = "equal"


class MaterialityLevel(str, Enum):
    """Materiality classification for discrepancies.

    Classifies the magnitude of a discrepancy as a percentage of the
    larger of the two totals. Thresholds are defined in
    MATERIALITY_THRESHOLDS.

    IMMATERIAL: Discrepancy is less than 5% -- no action required.
    MINOR: Discrepancy is 5-15% -- note in reporting, no remediation.
    MATERIAL: Discrepancy is 15-50% -- investigate root cause,
        disclose in reporting footnotes.
    SIGNIFICANT: Discrepancy is 50-100% -- full investigation and
        management sign-off required.
    EXTREME: Discrepancy exceeds 100% -- immediate escalation,
        likely data quality issue or missing contractual instruments.
    """

    IMMATERIAL = "immaterial"
    MINOR = "minor"
    MATERIAL = "material"
    SIGNIFICANT = "significant"
    EXTREME = "extreme"


class QualityDimension(str, Enum):
    """Data quality dimensions assessed during reconciliation.

    Four quality dimensions are scored independently and combined
    using weighted average (see QUALITY_WEIGHTS) to produce a
    composite quality score.

    COMPLETENESS: Are all energy types, facilities, and periods
        covered by both location-based and market-based results?
    CONSISTENCY: Are methodological choices, GWP values, reporting
        boundaries, and time periods consistent between the two
        methods?
    ACCURACY: Are emission factors from authoritative sources, and
        are calculations free from arithmetic errors?
    TRANSPARENCY: Are all assumptions, data sources, emission
        factors, and contractual instruments fully documented
        with provenance hashes?
    """

    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TRANSPARENCY = "transparency"


class QualityGrade(str, Enum):
    """Letter grade for quality assessment results.

    Grades are assigned based on the composite quality score
    thresholds defined in QUALITY_GRADE_THRESHOLDS.

    A: Excellent (>= 0.90) -- assurance-ready, minimal findings.
    B: Good (>= 0.80) -- minor findings, generally acceptable.
    C: Acceptable (>= 0.65) -- notable findings, improvement needed.
    D: Poor (>= 0.50) -- significant findings, remediation required.
    F: Failing (< 0.50) -- critical findings, not assurance-ready.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class EFHierarchyPriority(str, Enum):
    """Emission factor hierarchy per GHG Protocol Scope 2 Guidance.

    The GHG Protocol Scope 2 Guidance defines a quality hierarchy
    for market-based emission factors, from highest to lowest quality.
    This hierarchy is used in quality scoring to assess whether the
    best available data source was used.

    SUPPLIER_WITH_CERT: Supplier-specific EF with third-party
        certification or verification. Highest quality.
    SUPPLIER_NO_CERT: Supplier-specific EF without third-party
        certification. High quality but unverified.
    BUNDLED_CERT: EF derived from energy attribute certificates
        bundled with energy purchase (e.g. bundled RECs/GOs).
    UNBUNDLED_CERT: EF derived from unbundled energy attribute
        certificates purchased separately from energy.
    RESIDUAL_MIX: Residual mix emission factor after tracked
        instruments are removed from grid average.
    GRID_AVERAGE: Default grid average emission factor. Lowest
        quality for market-based; standard for location-based.
    """

    SUPPLIER_WITH_CERT = "supplier_with_cert"
    SUPPLIER_NO_CERT = "supplier_no_cert"
    BUNDLED_CERT = "bundled_cert"
    UNBUNDLED_CERT = "unbundled_cert"
    RESIDUAL_MIX = "residual_mix"
    GRID_AVERAGE = "grid_average"


class ReportingFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks supported.

    Each framework has specific dual-reporting requirements defined
    in FRAMEWORK_REQUIRED_DISCLOSURES. The reconciliation agent
    generates framework-specific reporting tables and checks
    compliance against each framework's requirements.

    GHG_PROTOCOL: GHG Protocol Scope 2 Guidance (2015). Mandates
        dual reporting of location-based and market-based totals.
    CSRD_ESRS: EU Corporate Sustainability Reporting Directive,
        European Sustainability Reporting Standards E1.
        Requires dual reporting with reconciliation disclosure.
    CDP: Carbon Disclosure Project climate questionnaire (C6/C7).
        Requires both methods plus detailed energy breakdowns.
    SBTI: Science Based Targets initiative. Uses market-based for
        target tracking, location-based for context.
    GRI: Global Reporting Initiative Standards 305-2 (2016).
        Requires both methods with separate disclosure.
    ISO_14064: ISO 14064-1:2018 Greenhouse gas quantification.
        Permits either method with justification; recommends both.
    RE100: RE100 renewable electricity initiative. Tracks market-based
        method for RE100 percentage calculation.
    """

    GHG_PROTOCOL = "ghg_protocol"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    GRI = "gri"
    ISO_14064 = "iso_14064"
    RE100 = "re100"


class FlagType(str, Enum):
    """Classification of flags raised during reconciliation.

    WARNING: Condition that may indicate a data quality issue but
        does not necessarily prevent reporting.
    ERROR: Condition that indicates a definite issue requiring
        remediation before reporting.
    INFO: Informational note for context or documentation purposes.
    RECOMMENDATION: Suggested improvement for future reporting
        periods.
    """

    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    RECOMMENDATION = "recommendation"


class FlagSeverity(str, Enum):
    """Severity level for flags raised during reconciliation.

    LOW: Minor issue with negligible impact on reported totals.
    MEDIUM: Moderate issue that should be addressed but does not
        materially affect reported totals.
    HIGH: Significant issue that materially affects reported totals
        or compliance status.
    CRITICAL: Blocking issue that prevents completion of
        reconciliation or renders results unreliable.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReconciliationStatus(str, Enum):
    """Status of a reconciliation run through the pipeline.

    PENDING: Reconciliation request has been received but processing
        has not started.
    IN_PROGRESS: Pipeline is actively processing the reconciliation
        through one or more stages.
    COMPLETED: All pipeline stages have completed successfully and
        the reconciliation report is available.
    FAILED: One or more pipeline stages failed, and the
        reconciliation could not be completed.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IntensityMetric(str, Enum):
    """Denominator metrics for emission intensity calculations.

    Intensity metrics normalise absolute emissions by a business
    activity metric, enabling comparison across reporting periods,
    facilities, and peer companies.

    REVENUE: Revenue-based intensity (tCO2e per million USD or EUR).
    FTE: Full-time-equivalent employee count intensity.
    FLOOR_AREA: Floor area intensity (tCO2e per square metre).
    PRODUCTION_UNIT: Production-based intensity (tCO2e per unit of
        output, e.g. per tonne of product).
    """

    REVENUE = "revenue"
    FTE = "fte"
    FLOOR_AREA = "floor_area"
    PRODUCTION_UNIT = "production_unit"


class TrendDirection(str, Enum):
    """Direction of a trend across reporting periods.

    INCREASING: Metric is rising over the analysed time range.
    DECREASING: Metric is falling over the analysed time range.
    STABLE: Metric is neither rising nor falling (within +/- 2%).
    """

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class PipelineStage(str, Enum):
    """Stages in the dual reporting reconciliation pipeline.

    The pipeline processes reconciliation requests through ten
    sequential stages. Each stage must complete before the next
    one begins.

    COLLECT_RESULTS: Gather location-based and market-based results
        from upstream agents (MRV-009 through MRV-012).
    ALIGN_BOUNDARIES: Verify that organisational, operational, and
        temporal boundaries match between the two methods.
    MAP_ENERGY_TYPES: Map and align energy types across upstream
        results to ensure consistent categorisation.
    ANALYZE_DISCREPANCIES: Identify, classify, and quantify
        discrepancies between the two totals using waterfall
        decomposition.
    SCORE_QUALITY: Assess data quality across four dimensions
        and assign a composite grade.
    GENERATE_TABLES: Generate framework-specific dual-reporting
        tables with footnotes and disclosures.
    ANALYZE_TRENDS: Compute year-over-year changes, CAGR, PIF
        trends, and RE100 percentage trajectories.
    CHECK_COMPLIANCE: Validate results against each requested
        framework's dual-reporting requirements.
    ASSEMBLE_REPORT: Combine all outputs into a unified
        ReconciliationReport.
    SEAL_PROVENANCE: Calculate SHA-256 provenance hash over the
        entire report for audit trail integrity.
    """

    COLLECT_RESULTS = "collect_results"
    ALIGN_BOUNDARIES = "align_boundaries"
    MAP_ENERGY_TYPES = "map_energy_types"
    ANALYZE_DISCREPANCIES = "analyze_discrepancies"
    SCORE_QUALITY = "score_quality"
    GENERATE_TABLES = "generate_tables"
    ANALYZE_TRENDS = "analyze_trends"
    CHECK_COMPLIANCE = "check_compliance"
    ASSEMBLE_REPORT = "assemble_report"
    SEAL_PROVENANCE = "seal_provenance"


class ExportFormat(str, Enum):
    """Supported export formats for reconciliation outputs.

    JSON: Machine-readable JSON export for API consumers and
        downstream integrations.
    CSV: Comma-separated values for spreadsheet import and
        tabular analysis.
    EXCEL: Microsoft Excel workbook (.xlsx) with formatted sheets,
        charts, and conditional formatting.
    """

    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check.

    COMPLIANT: All requirements of the regulatory framework are met.
    NON_COMPLIANT: One or more mandatory requirements are not met.
    PARTIAL: Some requirements are met but others are missing or
        incomplete.
    NOT_APPLICABLE: The framework's requirements do not apply to
        this entity or reporting period.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class DataQualityTier(str, Enum):
    """Data quality tier per IPCC methodology hierarchy.

    TIER_1: Uses global default emission factors and parameters.
        Simplest approach, highest uncertainty.
    TIER_2: Uses country-specific or region-specific emission
        factors and activity data. Moderate uncertainty.
    TIER_3: Uses facility-specific measurements, continuous
        emissions monitoring, or detailed process models.
        Lowest uncertainty.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class GWPSource(str, Enum):
    """IPCC Assessment Report source for Global Warming Potential values.

    AR4: Fourth Assessment Report (2007) - 100-year GWP values.
        CH4 = 25, N2O = 298.
    AR5: Fifth Assessment Report (2014) - 100-year GWP values.
        CH4 = 28, N2O = 265.
    AR6: Sixth Assessment Report (2021) - 100-year GWP values.
        CH4 = 27.9, N2O = 273.
    AR6_20YR: Sixth Assessment Report - 20-year GWP values.
        CH4 = 82.5, N2O = 273.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 2 emission calculations.

    CO2: Carbon dioxide - primary gas from fossil fuel combustion
        in electricity generation and heat production.
    CH4: Methane - emitted from upstream fuel extraction and
        incomplete combustion in power generation.
    N2O: Nitrous oxide - emitted from combustion processes in
        power generation and heat production.
    CO2E: Carbon dioxide equivalent - aggregate metric combining
        all gases using GWP values.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"


class BatchStatus(str, Enum):
    """Status of a batch reconciliation job.

    PENDING: Batch job has been created but not started.
    RUNNING: Batch job is actively processing periods.
    COMPLETED: All periods in the batch completed successfully.
    FAILED: All periods in the batch failed.
    PARTIAL: Some periods completed successfully while others failed.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================


# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[GWPSource, Dict[EmissionGas, Decimal]] = {
    GWPSource.AR4: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("25"),
        EmissionGas.N2O: Decimal("298"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR5: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("28"),
        EmissionGas.N2O: Decimal("265"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR6: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("27.9"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO2E: Decimal("1"),
    },
    GWPSource.AR6_20YR: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("82.5"),
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.CO2E: Decimal("1"),
    },
}


# ---------------------------------------------------------------------------
# Materiality thresholds: (min_pct inclusive, max_pct exclusive)
# ---------------------------------------------------------------------------

MATERIALITY_THRESHOLDS: Dict[MaterialityLevel, Tuple[Decimal, Decimal]] = {
    MaterialityLevel.IMMATERIAL: (Decimal("0"), Decimal("5")),
    MaterialityLevel.MINOR: (Decimal("5"), Decimal("15")),
    MaterialityLevel.MATERIAL: (Decimal("15"), Decimal("50")),
    MaterialityLevel.SIGNIFICANT: (Decimal("50"), Decimal("100")),
    MaterialityLevel.EXTREME: (Decimal("100"), DECIMAL_INF),
}


# ---------------------------------------------------------------------------
# Quality dimension weights for composite score calculation
# Sum must equal 1.00
# ---------------------------------------------------------------------------

QUALITY_WEIGHTS: Dict[QualityDimension, Decimal] = {
    QualityDimension.COMPLETENESS: Decimal("0.30"),
    QualityDimension.CONSISTENCY: Decimal("0.25"),
    QualityDimension.ACCURACY: Decimal("0.25"),
    QualityDimension.TRANSPARENCY: Decimal("0.20"),
}


# ---------------------------------------------------------------------------
# Quality grade thresholds (minimum composite score for each grade)
# ---------------------------------------------------------------------------

QUALITY_GRADE_THRESHOLDS: Dict[QualityGrade, Decimal] = {
    QualityGrade.A: Decimal("0.90"),
    QualityGrade.B: Decimal("0.80"),
    QualityGrade.C: Decimal("0.65"),
    QualityGrade.D: Decimal("0.50"),
    QualityGrade.F: Decimal("0.0"),
}


# ---------------------------------------------------------------------------
# Emission factor hierarchy quality scores per GHG Protocol Scope 2
# ---------------------------------------------------------------------------

EF_HIERARCHY_QUALITY_SCORES: Dict[EFHierarchyPriority, Decimal] = {
    EFHierarchyPriority.SUPPLIER_WITH_CERT: Decimal("1.00"),
    EFHierarchyPriority.SUPPLIER_NO_CERT: Decimal("0.85"),
    EFHierarchyPriority.BUNDLED_CERT: Decimal("0.75"),
    EFHierarchyPriority.UNBUNDLED_CERT: Decimal("0.65"),
    EFHierarchyPriority.RESIDUAL_MIX: Decimal("0.40"),
    EFHierarchyPriority.GRID_AVERAGE: Decimal("0.20"),
}


# ---------------------------------------------------------------------------
# Upstream agent mapping: energy type -> (location agent, market agent)
# Each energy type has a primary upstream agent for both methods.
# ---------------------------------------------------------------------------

UPSTREAM_AGENT_MAPPING: Dict[
    EnergyType, Tuple[UpstreamAgent, UpstreamAgent]
] = {
    EnergyType.ELECTRICITY: (UpstreamAgent.MRV_009, UpstreamAgent.MRV_009),
    EnergyType.STEAM: (UpstreamAgent.MRV_010, UpstreamAgent.MRV_010),
    EnergyType.DISTRICT_HEATING: (UpstreamAgent.MRV_011, UpstreamAgent.MRV_011),
    EnergyType.DISTRICT_COOLING: (UpstreamAgent.MRV_012, UpstreamAgent.MRV_012),
}


# ---------------------------------------------------------------------------
# Framework required disclosures
# Each framework specifies what must appear in the dual-reporting output.
# ---------------------------------------------------------------------------

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ReportingFramework, List[str]] = {
    ReportingFramework.GHG_PROTOCOL: [
        "location_based_total_tco2e",
        "market_based_total_tco2e",
        "location_by_energy_type",
        "market_by_energy_type",
        "location_by_country",
        "market_by_country",
        "contractual_instruments_summary",
        "residual_mix_disclosure",
        "emission_factor_sources",
        "gwp_values_used",
        "organizational_boundary",
        "base_year_recalculation_policy",
        "exclusions_and_limitations",
    ],
    ReportingFramework.CSRD_ESRS: [
        "location_based_total_tco2e",
        "market_based_total_tco2e",
        "location_by_energy_type",
        "market_by_energy_type",
        "energy_consumption_mwh",
        "renewable_energy_percentage",
        "reconciliation_explanation",
        "emission_factor_sources",
        "gwp_values_used",
        "data_quality_assessment",
        "significant_changes_explanation",
        "base_year_emissions",
        "reduction_targets",
        "value_chain_boundary",
    ],
    ReportingFramework.CDP: [
        "location_based_total_tco2e",
        "market_based_total_tco2e",
        "location_by_country",
        "market_by_country",
        "location_by_activity",
        "market_by_activity",
        "electricity_consumption_mwh",
        "steam_consumption_mwh",
        "heating_consumption_mwh",
        "cooling_consumption_mwh",
        "low_carbon_electricity_percentage",
        "renewable_electricity_percentage",
        "contractual_instruments_details",
        "emission_factor_sources",
        "verification_status",
    ],
    ReportingFramework.SBTI: [
        "market_based_total_tco2e",
        "location_based_total_tco2e",
        "base_year_market_based_tco2e",
        "base_year_location_based_tco2e",
        "target_year",
        "reduction_percentage",
        "re100_progress_percentage",
        "renewable_electricity_mwh",
        "total_electricity_mwh",
        "emission_factor_sources",
        "contractual_instruments_summary",
    ],
    ReportingFramework.GRI: [
        "location_based_total_tco2e",
        "market_based_total_tco2e",
        "location_by_energy_type",
        "market_by_energy_type",
        "emission_factor_sources",
        "gwp_values_used",
        "consolidation_approach",
        "base_year_information",
        "standards_and_methodologies",
        "significant_changes",
    ],
    ReportingFramework.ISO_14064: [
        "location_based_total_tco2e",
        "market_based_total_tco2e",
        "method_justification",
        "emission_by_gas_co2",
        "emission_by_gas_ch4",
        "emission_by_gas_n2o",
        "emission_factor_sources",
        "gwp_values_used",
        "uncertainty_assessment",
        "organizational_boundary",
        "reporting_period",
        "base_year_information",
        "data_quality_assessment",
    ],
    ReportingFramework.RE100: [
        "total_electricity_mwh",
        "renewable_electricity_mwh",
        "re100_percentage",
        "contractual_instruments_breakdown",
        "self_generation_mwh",
        "unbundled_eacs_mwh",
        "bundled_eacs_mwh",
        "ppa_mwh",
        "green_tariff_mwh",
        "market_based_total_tco2e",
        "country_breakdown",
    ],
}


# ---------------------------------------------------------------------------
# Residual mix factors by region
#
# Each entry contains grid_average_ef (tCO2e/MWh), residual_mix_ef
# (tCO2e/MWh), and the ratio (residual / grid_average). These are used
# to quantify the RESIDUAL_MIX_UPLIFT discrepancy type. Values are
# representative 2024 estimates and must be updated annually.
# ---------------------------------------------------------------------------

# Forward declaration -- the ResidualMixFactor model is defined below
# so we build the constant table after the model definition.


# =============================================================================
# Data Models (25) -- Pydantic v2, frozen=True
# =============================================================================


# ---------------------------------------------------------------------------
# 1. ResidualMixFactor
# ---------------------------------------------------------------------------


class ResidualMixFactor(BaseModel):
    """Regional residual mix emission factor data.

    Contains the grid average emission factor, the residual mix emission
    factor (after tracked instruments are removed), and their ratio for
    a given region and reference year.

    The residual mix factor is typically higher than the grid average
    because renewable generation that has been contractually claimed
    via RECs/GOs is removed from the residual pool, concentrating
    fossil generation in the unclaimed portion.

    Attributes:
        region: ISO 3166-1 alpha-2 country code or sub-national grid
            region identifier (e.g. "US-ERCOT", "DE", "GB").
        grid_average_ef: Grid average emission factor in tCO2e per MWh.
        residual_mix_ef: Residual mix emission factor in tCO2e per MWh.
        ratio: Ratio of residual_mix_ef to grid_average_ef. Values
            above 1.0 indicate uplift from tracked renewables removal.
        year: Reference year for the emission factor data.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    region: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description=(
            "ISO 3166-1 alpha-2 country code or sub-national grid "
            "region identifier"
        ),
    )
    grid_average_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid average emission factor in tCO2e per MWh",
    )
    residual_mix_ef: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Residual mix emission factor in tCO2e per MWh",
    )
    ratio: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description=(
            "Ratio of residual_mix_ef to grid_average_ef; values "
            "above 1.0 indicate uplift from tracked renewables removal"
        ),
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reference year for the emission factor data",
    )


# ---------------------------------------------------------------------------
# Now build the RESIDUAL_MIX_FACTORS constant using the model above
# ---------------------------------------------------------------------------

RESIDUAL_MIX_FACTORS: Dict[str, ResidualMixFactor] = {
    # Europe
    "AT": ResidualMixFactor(
        region="AT", grid_average_ef=Decimal("0.0930"),
        residual_mix_ef=Decimal("0.2210"), ratio=Decimal("2.376"), year=2024,
    ),
    "BE": ResidualMixFactor(
        region="BE", grid_average_ef=Decimal("0.1490"),
        residual_mix_ef=Decimal("0.2810"), ratio=Decimal("1.886"), year=2024,
    ),
    "BG": ResidualMixFactor(
        region="BG", grid_average_ef=Decimal("0.3940"),
        residual_mix_ef=Decimal("0.5120"), ratio=Decimal("1.299"), year=2024,
    ),
    "CZ": ResidualMixFactor(
        region="CZ", grid_average_ef=Decimal("0.3850"),
        residual_mix_ef=Decimal("0.5290"), ratio=Decimal("1.374"), year=2024,
    ),
    "DE": ResidualMixFactor(
        region="DE", grid_average_ef=Decimal("0.3380"),
        residual_mix_ef=Decimal("0.5760"), ratio=Decimal("1.704"), year=2024,
    ),
    "DK": ResidualMixFactor(
        region="DK", grid_average_ef=Decimal("0.0940"),
        residual_mix_ef=Decimal("0.3590"), ratio=Decimal("3.819"), year=2024,
    ),
    "ES": ResidualMixFactor(
        region="ES", grid_average_ef=Decimal("0.1380"),
        residual_mix_ef=Decimal("0.2860"), ratio=Decimal("2.072"), year=2024,
    ),
    "FI": ResidualMixFactor(
        region="FI", grid_average_ef=Decimal("0.0680"),
        residual_mix_ef=Decimal("0.2830"), ratio=Decimal("4.162"), year=2024,
    ),
    "FR": ResidualMixFactor(
        region="FR", grid_average_ef=Decimal("0.0520"),
        residual_mix_ef=Decimal("0.2320"), ratio=Decimal("4.462"), year=2024,
    ),
    "GB": ResidualMixFactor(
        region="GB", grid_average_ef=Decimal("0.2070"),
        residual_mix_ef=Decimal("0.3480"), ratio=Decimal("1.681"), year=2024,
    ),
    "GR": ResidualMixFactor(
        region="GR", grid_average_ef=Decimal("0.2910"),
        residual_mix_ef=Decimal("0.4630"), ratio=Decimal("1.591"), year=2024,
    ),
    "HU": ResidualMixFactor(
        region="HU", grid_average_ef=Decimal("0.2110"),
        residual_mix_ef=Decimal("0.3390"), ratio=Decimal("1.607"), year=2024,
    ),
    "IE": ResidualMixFactor(
        region="IE", grid_average_ef=Decimal("0.2960"),
        residual_mix_ef=Decimal("0.4280"), ratio=Decimal("1.446"), year=2024,
    ),
    "IT": ResidualMixFactor(
        region="IT", grid_average_ef=Decimal("0.2560"),
        residual_mix_ef=Decimal("0.4110"), ratio=Decimal("1.605"), year=2024,
    ),
    "NL": ResidualMixFactor(
        region="NL", grid_average_ef=Decimal("0.3280"),
        residual_mix_ef=Decimal("0.5140"), ratio=Decimal("1.567"), year=2024,
    ),
    "NO": ResidualMixFactor(
        region="NO", grid_average_ef=Decimal("0.0080"),
        residual_mix_ef=Decimal("0.3690"), ratio=Decimal("46.125"), year=2024,
    ),
    "PL": ResidualMixFactor(
        region="PL", grid_average_ef=Decimal("0.6230"),
        residual_mix_ef=Decimal("0.7450"), ratio=Decimal("1.196"), year=2024,
    ),
    "PT": ResidualMixFactor(
        region="PT", grid_average_ef=Decimal("0.1580"),
        residual_mix_ef=Decimal("0.3070"), ratio=Decimal("1.943"), year=2024,
    ),
    "RO": ResidualMixFactor(
        region="RO", grid_average_ef=Decimal("0.2530"),
        residual_mix_ef=Decimal("0.3920"), ratio=Decimal("1.549"), year=2024,
    ),
    "SE": ResidualMixFactor(
        region="SE", grid_average_ef=Decimal("0.0120"),
        residual_mix_ef=Decimal("0.3320"), ratio=Decimal("27.667"), year=2024,
    ),
    # North America
    "US-CAMX": ResidualMixFactor(
        region="US-CAMX", grid_average_ef=Decimal("0.2250"),
        residual_mix_ef=Decimal("0.3410"), ratio=Decimal("1.516"), year=2024,
    ),
    "US-ERCOT": ResidualMixFactor(
        region="US-ERCOT", grid_average_ef=Decimal("0.3690"),
        residual_mix_ef=Decimal("0.4520"), ratio=Decimal("1.225"), year=2024,
    ),
    "US-MROE": ResidualMixFactor(
        region="US-MROE", grid_average_ef=Decimal("0.5480"),
        residual_mix_ef=Decimal("0.6120"), ratio=Decimal("1.117"), year=2024,
    ),
    "US-NEWE": ResidualMixFactor(
        region="US-NEWE", grid_average_ef=Decimal("0.2170"),
        residual_mix_ef=Decimal("0.3590"), ratio=Decimal("1.655"), year=2024,
    ),
    "US-NWPP": ResidualMixFactor(
        region="US-NWPP", grid_average_ef=Decimal("0.2780"),
        residual_mix_ef=Decimal("0.3960"), ratio=Decimal("1.424"), year=2024,
    ),
    "US-RFCE": ResidualMixFactor(
        region="US-RFCE", grid_average_ef=Decimal("0.2920"),
        residual_mix_ef=Decimal("0.4380"), ratio=Decimal("1.500"), year=2024,
    ),
    "US-SRMV": ResidualMixFactor(
        region="US-SRMV", grid_average_ef=Decimal("0.3710"),
        residual_mix_ef=Decimal("0.4290"), ratio=Decimal("1.156"), year=2024,
    ),
    "CA": ResidualMixFactor(
        region="CA", grid_average_ef=Decimal("0.1200"),
        residual_mix_ef=Decimal("0.2380"), ratio=Decimal("1.983"), year=2024,
    ),
    # Asia-Pacific
    "AU": ResidualMixFactor(
        region="AU", grid_average_ef=Decimal("0.6560"),
        residual_mix_ef=Decimal("0.7230"), ratio=Decimal("1.102"), year=2024,
    ),
    "JP": ResidualMixFactor(
        region="JP", grid_average_ef=Decimal("0.4570"),
        residual_mix_ef=Decimal("0.5310"), ratio=Decimal("1.162"), year=2024,
    ),
    "KR": ResidualMixFactor(
        region="KR", grid_average_ef=Decimal("0.4150"),
        residual_mix_ef=Decimal("0.4890"), ratio=Decimal("1.178"), year=2024,
    ),
    "IN": ResidualMixFactor(
        region="IN", grid_average_ef=Decimal("0.7080"),
        residual_mix_ef=Decimal("0.7640"), ratio=Decimal("1.079"), year=2024,
    ),
    "CN": ResidualMixFactor(
        region="CN", grid_average_ef=Decimal("0.5810"),
        residual_mix_ef=Decimal("0.6380"), ratio=Decimal("1.098"), year=2024,
    ),
    "SG": ResidualMixFactor(
        region="SG", grid_average_ef=Decimal("0.4080"),
        residual_mix_ef=Decimal("0.4510"), ratio=Decimal("1.105"), year=2024,
    ),
    # Latin America
    "BR": ResidualMixFactor(
        region="BR", grid_average_ef=Decimal("0.0740"),
        residual_mix_ef=Decimal("0.1820"), ratio=Decimal("2.459"), year=2024,
    ),
    "MX": ResidualMixFactor(
        region="MX", grid_average_ef=Decimal("0.4050"),
        residual_mix_ef=Decimal("0.4720"), ratio=Decimal("1.165"), year=2024,
    ),
    # Middle East & Africa
    "ZA": ResidualMixFactor(
        region="ZA", grid_average_ef=Decimal("0.9280"),
        residual_mix_ef=Decimal("0.9540"), ratio=Decimal("1.028"), year=2024,
    ),
    "AE": ResidualMixFactor(
        region="AE", grid_average_ef=Decimal("0.4380"),
        residual_mix_ef=Decimal("0.4720"), ratio=Decimal("1.078"), year=2024,
    ),
}


# ---------------------------------------------------------------------------
# 2. UpstreamResult
# ---------------------------------------------------------------------------


class UpstreamResult(BaseModel):
    """A single emission result from an upstream Scope 2 agent.

    Represents one row of output from MRV-009 through MRV-012, carrying
    the calculated emissions under either location-based or market-based
    methodology for a single energy type, facility, and period.

    This model is the primary input to the reconciliation pipeline.
    The agent collects location-based and market-based UpstreamResult
    objects, aligns them by facility and energy type, and computes
    the discrepancy between the two methods.

    Attributes:
        agent: Identifier of the upstream agent that produced this result.
        method: Scope 2 accounting method used (location or market).
        energy_type: Type of purchased energy.
        emissions_tco2e: Total emissions in tonnes CO2 equivalent.
        emissions_by_gas: Breakdown of emissions by individual gas
            (CO2, CH4, N2O) in tonnes CO2e.
        energy_quantity_mwh: Energy consumed in megawatt-hours.
        ef_used: Emission factor used in tCO2e/MWh.
        ef_source: Source of the emission factor (e.g. "eGRID 2023",
            "AIB Residual Mix 2023", "supplier certificate").
        ef_hierarchy: Position in the EF hierarchy for quality scoring.
        tier: Data quality tier of the calculation.
        gwp_source: IPCC Assessment Report used for GWP values.
        facility_id: Unique identifier of the facility.
        facility_name: Human-readable facility name.
        region: Grid region or country code for the facility.
        tenant_id: Tenant identifier for multi-tenancy scoping.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        provenance_hash: SHA-256 hash from the upstream agent for
            audit trail integrity.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    agent: UpstreamAgent = Field(
        ...,
        description="Upstream agent that produced this result",
    )
    method: Scope2Method = Field(
        ...,
        description="Scope 2 accounting method (location or market)",
    )
    energy_type: EnergyType = Field(
        ...,
        description="Type of purchased energy",
    )
    emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions in tonnes CO2 equivalent",
    )
    emissions_by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description=(
            "Breakdown of emissions by gas (CO2, CH4, N2O) in "
            "tonnes CO2e"
        ),
    )
    energy_quantity_mwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Energy consumed in megawatt-hours",
    )
    ef_used: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor used in tCO2e/MWh",
    )
    ef_source: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Source of the emission factor (e.g. eGRID 2023, "
            "AIB Residual Mix 2023)"
        ),
    )
    ef_hierarchy: Optional[EFHierarchyPriority] = Field(
        default=None,
        description=(
            "Position in the GHG Protocol EF hierarchy for quality "
            "scoring; applicable only to market-based results"
        ),
    )
    tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier of the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="IPCC Assessment Report used for GWP values",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier of the facility",
    )
    facility_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable facility name",
    )
    region: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Grid region or country code for the facility",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy scoping",
    )
    period_start: date = Field(
        ...,
        description="Start date of the reporting period",
    )
    period_end: date = Field(
        ...,
        description="End date of the reporting period",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description=(
            "SHA-256 hash from the upstream agent for audit trail "
            "integrity"
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: date, info: Any) -> date:
        """Validate that period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

    @field_validator("emissions_by_gas")
    @classmethod
    def _validate_gas_keys(
        cls, v: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """Validate that gas keys are recognised emission gas names."""
        valid_keys = {g.value for g in EmissionGas}
        for key in v:
            if key not in valid_keys:
                raise ValueError(
                    f"Unrecognised emission gas key '{key}'; "
                    f"valid keys are {sorted(valid_keys)}"
                )
        return v


# ---------------------------------------------------------------------------
# 3. EnergyTypeBreakdown
# ---------------------------------------------------------------------------


class EnergyTypeBreakdown(BaseModel):
    """Reconciliation breakdown for a single energy type.

    Compares location-based and market-based totals for one energy
    type, computing the absolute and percentage difference and the
    direction of the discrepancy.

    Attributes:
        energy_type: The energy type being compared.
        location_tco2e: Location-based emissions in tCO2e.
        market_tco2e: Market-based emissions in tCO2e.
        difference_tco2e: Absolute difference (location minus market)
            in tCO2e. Positive means market is lower.
        difference_pct: Percentage difference relative to the larger
            of the two totals.
        direction: Whether market-based is lower, higher, or equal.
        energy_mwh: Total energy consumed in MWh for this type.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    energy_type: EnergyType = Field(
        ...,
        description="Energy type being compared",
    )
    location_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based emissions in tCO2e",
    )
    market_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Market-based emissions in tCO2e",
    )
    difference_tco2e: Decimal = Field(
        ...,
        description=(
            "Absolute difference (location minus market) in tCO2e; "
            "positive means market is lower"
        ),
    )
    difference_pct: Decimal = Field(
        ...,
        description=(
            "Percentage difference relative to the larger of the "
            "two totals"
        ),
    )
    direction: DiscrepancyDirection = Field(
        ...,
        description="Whether market-based is lower, higher, or equal",
    )
    energy_mwh: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total energy consumed in MWh for this type",
    )


# ---------------------------------------------------------------------------
# 4. FacilityBreakdown
# ---------------------------------------------------------------------------


class FacilityBreakdown(BaseModel):
    """Reconciliation breakdown for a single facility.

    Compares location-based and market-based totals for one facility
    across all energy types, computing the absolute and percentage
    difference.

    Attributes:
        facility_id: Unique identifier of the facility.
        facility_name: Human-readable name of the facility.
        location_tco2e: Location-based emissions in tCO2e for this
            facility.
        market_tco2e: Market-based emissions in tCO2e for this
            facility.
        difference_tco2e: Absolute difference (location minus market)
            in tCO2e.
        difference_pct: Percentage difference relative to the larger
            of the two totals.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier of the facility",
    )
    facility_name: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable name of the facility",
    )
    location_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based emissions in tCO2e",
    )
    market_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Market-based emissions in tCO2e",
    )
    difference_tco2e: Decimal = Field(
        ...,
        description=(
            "Absolute difference (location minus market) in tCO2e"
        ),
    )
    difference_pct: Decimal = Field(
        ...,
        description=(
            "Percentage difference relative to the larger of the "
            "two totals"
        ),
    )


# ---------------------------------------------------------------------------
# 5. ReconciliationWorkspace
# ---------------------------------------------------------------------------


class ReconciliationWorkspace(BaseModel):
    """Working data structure for a reconciliation run.

    Holds the intermediate state as the pipeline progresses through
    its ten stages. Populated incrementally as upstream results are
    collected, aligned, and analysed.

    Attributes:
        reconciliation_id: Unique identifier for this reconciliation
            run (UUID).
        tenant_id: Tenant identifier for multi-tenancy scoping.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        location_results: List of location-based upstream results
            collected from MRV-009 through MRV-012.
        market_results: List of market-based upstream results
            collected from MRV-009 through MRV-012.
        total_location_tco2e: Aggregate location-based emissions.
        total_market_tco2e: Aggregate market-based emissions.
        by_energy_type: Breakdown by energy type.
        by_facility: Breakdown by facility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reconciliation run identifier (UUID)",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy scoping",
    )
    period_start: date = Field(
        ...,
        description="Start date of the reporting period",
    )
    period_end: date = Field(
        ...,
        description="End date of the reporting period",
    )
    location_results: List[UpstreamResult] = Field(
        default_factory=list,
        description=(
            "Location-based upstream results from MRV-009 "
            "through MRV-012"
        ),
    )
    market_results: List[UpstreamResult] = Field(
        default_factory=list,
        description=(
            "Market-based upstream results from MRV-009 "
            "through MRV-012"
        ),
    )
    total_location_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate location-based emissions in tCO2e",
    )
    total_market_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate market-based emissions in tCO2e",
    )
    by_energy_type: List[EnergyTypeBreakdown] = Field(
        default_factory=list,
        description="Breakdown by energy type",
    )
    by_facility: List[FacilityBreakdown] = Field(
        default_factory=list,
        description="Breakdown by facility",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: date, info: Any) -> date:
        """Validate that period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

    @field_validator("location_results", "market_results")
    @classmethod
    def _validate_results_count(
        cls, v: List[UpstreamResult]
    ) -> List[UpstreamResult]:
        """Validate that upstream results do not exceed maximum."""
        if len(v) > MAX_UPSTREAM_RESULTS:
            raise ValueError(
                f"Maximum {MAX_UPSTREAM_RESULTS} upstream results "
                f"per method, got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 6. Discrepancy
# ---------------------------------------------------------------------------


class Discrepancy(BaseModel):
    """A single identified discrepancy between location and market totals.

    Represents one root cause contributing to the overall difference
    between the two Scope 2 methods. Multiple discrepancies may exist
    per reconciliation, and their absolute contributions should sum
    to the total discrepancy.

    Attributes:
        discrepancy_type: Classification of the discrepancy root cause.
        direction: Whether this discrepancy makes market higher or lower.
        materiality: Materiality level based on percentage thresholds.
        absolute_tco2e: Absolute contribution of this discrepancy
            in tonnes CO2e.
        percentage: Contribution as a percentage of the total
            discrepancy.
        energy_type: Energy type most affected by this discrepancy,
            if applicable.
        facility_id: Facility most affected, if applicable.
        region: Region most affected, if applicable.
        description: Human-readable explanation of the discrepancy.
        recommendation: Suggested action to address the discrepancy.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    discrepancy_type: DiscrepancyType = Field(
        ...,
        description="Classification of the discrepancy root cause",
    )
    direction: DiscrepancyDirection = Field(
        ...,
        description=(
            "Whether this discrepancy makes market higher or lower"
        ),
    )
    materiality: MaterialityLevel = Field(
        ...,
        description=(
            "Materiality level based on percentage thresholds"
        ),
    )
    absolute_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description=(
            "Absolute contribution in tonnes CO2e"
        ),
    )
    percentage: Decimal = Field(
        ...,
        description=(
            "Contribution as a percentage of the total discrepancy"
        ),
    )
    energy_type: Optional[EnergyType] = Field(
        default=None,
        description=(
            "Energy type most affected, if applicable"
        ),
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Facility most affected, if applicable",
    )
    region: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Region most affected, if applicable",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable explanation of the discrepancy",
    )
    recommendation: str = Field(
        default="",
        max_length=2000,
        description="Suggested action to address the discrepancy",
    )


# ---------------------------------------------------------------------------
# 7. WaterfallItem
# ---------------------------------------------------------------------------


class WaterfallItem(BaseModel):
    """A single step in the waterfall decomposition of the total discrepancy.

    The waterfall decomposes the total difference between location-based
    and market-based totals into individual drivers. Starting from the
    location-based total, each WaterfallItem adds or subtracts its
    contribution to arrive at the market-based total.

    Attributes:
        driver: Name of the driver (maps to a DiscrepancyType or
            a summary label such as "Total Location" or "Total Market").
        contribution_tco2e: Signed contribution in tonnes CO2e.
            Negative values reduce the running total toward market-based.
        contribution_pct: Signed percentage contribution relative
            to the location-based total.
        description: Human-readable explanation of this driver step.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    driver: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Name of the driver (maps to a DiscrepancyType or "
            "summary label)"
        ),
    )
    contribution_tco2e: Decimal = Field(
        ...,
        description=(
            "Signed contribution in tonnes CO2e; negative values "
            "reduce toward market-based total"
        ),
    )
    contribution_pct: Decimal = Field(
        ...,
        description=(
            "Signed percentage contribution relative to "
            "location-based total"
        ),
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Human-readable explanation of this driver step",
    )


# ---------------------------------------------------------------------------
# 8. WaterfallDecomposition
# ---------------------------------------------------------------------------


class WaterfallDecomposition(BaseModel):
    """Complete waterfall decomposition from location to market totals.

    Decomposes the total discrepancy between location-based and
    market-based Scope 2 emissions into discrete driver steps. The
    items list should form a bridge from location total to market
    total when contributions are summed.

    Attributes:
        total_discrepancy_tco2e: Total difference (location minus
            market) in tonnes CO2e.
        items: Ordered list of waterfall steps from location total
            to market total.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    total_discrepancy_tco2e: Decimal = Field(
        ...,
        description=(
            "Total difference (location minus market) in tCO2e"
        ),
    )
    items: List[WaterfallItem] = Field(
        default_factory=list,
        description=(
            "Ordered waterfall steps from location to market total"
        ),
    )


# ---------------------------------------------------------------------------
# 9. Flag (defined before DiscrepancyReport to allow forward reference)
# ---------------------------------------------------------------------------


class Flag(BaseModel):
    """A flag raised during reconciliation processing.

    Flags communicate warnings, errors, informational notes, and
    recommendations to the user. They are collected across all
    pipeline stages and included in the final report.

    Attributes:
        flag_type: Classification of the flag.
        severity: Severity level of the flag.
        code: Machine-readable code for programmatic handling
            (e.g. "DRR-W-001", "DRR-E-003").
        message: Human-readable message describing the condition.
        recommendation: Suggested action to resolve the condition.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    flag_type: FlagType = Field(
        ...,
        description="Classification of the flag",
    )
    severity: FlagSeverity = Field(
        ...,
        description="Severity level of the flag",
    )
    code: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "Machine-readable code for programmatic handling "
            "(e.g. DRR-W-001)"
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable message describing the condition",
    )
    recommendation: str = Field(
        default="",
        max_length=2000,
        description="Suggested action to resolve the condition",
    )


# ---------------------------------------------------------------------------
# 10. DiscrepancyReport (model 9 in the specification numbering)
# ---------------------------------------------------------------------------


class DiscrepancyReport(BaseModel):
    """Complete discrepancy analysis report for a reconciliation run.

    Contains all identified discrepancies, the waterfall decomposition,
    a materiality summary, and flags raised during analysis.

    Attributes:
        reconciliation_id: Reference to the reconciliation run.
        discrepancies: List of identified discrepancies.
        materiality_summary: Count of discrepancies by materiality
            level (e.g. {"immaterial": 2, "material": 1}).
        waterfall: Waterfall decomposition from location to market.
        flags: Flags raised during discrepancy analysis.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the reconciliation run",
    )
    discrepancies: List[Discrepancy] = Field(
        default_factory=list,
        description="List of identified discrepancies",
    )
    materiality_summary: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Count of discrepancies by materiality level"
        ),
    )
    waterfall: Optional[WaterfallDecomposition] = Field(
        default=None,
        description=(
            "Waterfall decomposition from location to market total"
        ),
    )
    flags: List[Flag] = Field(
        default_factory=list,
        description="Flags raised during discrepancy analysis",
    )

    @field_validator("discrepancies")
    @classmethod
    def _validate_discrepancy_count(
        cls, v: List[Discrepancy]
    ) -> List[Discrepancy]:
        """Validate that discrepancies do not exceed maximum."""
        if len(v) > MAX_DISCREPANCIES:
            raise ValueError(
                f"Maximum {MAX_DISCREPANCIES} discrepancies per "
                f"report, got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 11. QualityScore
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    """Score for a single quality dimension.

    Each of the four quality dimensions (completeness, consistency,
    accuracy, transparency) receives an independent score between
    0.0 and max_score (default 1.0). Scores are combined using
    QUALITY_WEIGHTS to produce a composite score.

    Attributes:
        dimension: The quality dimension being scored.
        score: Numeric score for this dimension (0.0 to max_score).
        max_score: Maximum possible score (default 1.0).
        findings: List of findings or observations for this dimension.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    dimension: QualityDimension = Field(
        ...,
        description="Quality dimension being scored",
    )
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Numeric score (0.0 to max_score)",
    )
    max_score: Decimal = Field(
        default=Decimal("1.0"),
        gt=Decimal("0"),
        description="Maximum possible score (default 1.0)",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Findings or observations for this dimension",
    )

    @field_validator("score")
    @classmethod
    def _score_within_max(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that score does not exceed max_score."""
        max_s = info.data.get("max_score", Decimal("1.0"))
        if max_s is not None and v > max_s:
            raise ValueError(
                f"Score ({v}) cannot exceed max_score ({max_s})"
            )
        return v


# ---------------------------------------------------------------------------
# 12. QualityAssessment
# ---------------------------------------------------------------------------


class QualityAssessment(BaseModel):
    """Complete quality assessment for a reconciliation run.

    Combines individual dimension scores into a weighted composite
    score and assigns a letter grade. Determines whether the
    reconciliation output meets the threshold for external assurance.

    Attributes:
        reconciliation_id: Reference to the reconciliation run.
        scores: Individual scores for each quality dimension.
        composite_score: Weighted composite score (0.0 to 1.0).
        grade: Letter grade based on composite score thresholds.
        assurance_ready: Whether the output meets the minimum
            threshold for external assurance (grade A or B).
        findings: Aggregate findings across all dimensions.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the reconciliation run",
    )
    scores: List[QualityScore] = Field(
        default_factory=list,
        description="Individual scores for each quality dimension",
    )
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Weighted composite score (0.0 to 1.0)",
    )
    grade: QualityGrade = Field(
        ...,
        description="Letter grade based on composite score thresholds",
    )
    assurance_ready: bool = Field(
        default=False,
        description=(
            "Whether output meets minimum threshold for external "
            "assurance (grade A or B)"
        ),
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Aggregate findings across all dimensions",
    )


# ---------------------------------------------------------------------------
# 13. FrameworkTable
# ---------------------------------------------------------------------------


class FrameworkTable(BaseModel):
    """A reporting table generated for a specific framework.

    Each framework (GHG Protocol, CSRD, CDP, etc.) has its own
    table format with specific rows, columns, and footnotes.
    Tables are generated during the GENERATE_TABLES pipeline stage.

    Attributes:
        framework: The reporting framework this table is for.
        title: Human-readable title of the table.
        rows: List of row dictionaries containing table data.
            Each dict maps column names to values.
        footnotes: List of footnotes for the table.
        generated_at: UTC timestamp of table generation.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ReportingFramework = Field(
        ...,
        description="Reporting framework this table is for",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable title of the table",
    )
    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Row dictionaries containing table data; each dict "
            "maps column names to values"
        ),
    )
    footnotes: List[str] = Field(
        default_factory=list,
        description="Footnotes for the table",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of table generation",
    )


# ---------------------------------------------------------------------------
# 14. ReportingTableSet
# ---------------------------------------------------------------------------


class ReportingTableSet(BaseModel):
    """Collection of framework-specific reporting tables.

    Groups all tables generated for a single reconciliation run,
    one per requested framework.

    Attributes:
        reconciliation_id: Reference to the reconciliation run.
        tables: List of framework-specific tables.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the reconciliation run",
    )
    tables: List[FrameworkTable] = Field(
        default_factory=list,
        description="Framework-specific reporting tables",
    )


# ---------------------------------------------------------------------------
# 15. TrendDataPoint
# ---------------------------------------------------------------------------


class TrendDataPoint(BaseModel):
    """A single data point in a multi-period trend analysis.

    Captures location-based and market-based emissions for one
    reporting period along with derived metrics (PIF, RE100
    percentage).

    PIF (Procurement Impact Factor) = 1 - (market / location).
    A PIF of 0.30 means market-based is 30% lower than location-based,
    indicating procurement of cleaner energy sources.

    RE100 percentage = renewable electricity / total electricity * 100.

    Attributes:
        period: Label for the reporting period (e.g. "2024",
            "2024-Q1").
        location_tco2e: Location-based emissions for this period.
        market_tco2e: Market-based emissions for this period.
        pif: Procurement Impact Factor for this period.
        re100_pct: RE100 renewable electricity percentage.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Label for the reporting period",
    )
    location_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based emissions for this period",
    )
    market_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Market-based emissions for this period",
    )
    pif: Decimal = Field(
        default=Decimal("0"),
        description=(
            "Procurement Impact Factor: 1 - (market / location)"
        ),
    )
    re100_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="RE100 renewable electricity percentage",
    )


# ---------------------------------------------------------------------------
# 16. TrendReport
# ---------------------------------------------------------------------------


class TrendReport(BaseModel):
    """Multi-period trend analysis report.

    Analyses location-based and market-based emissions across multiple
    reporting periods, computing year-over-year changes, compound
    annual growth rates (CAGR), PIF trends, RE100 progress, and
    intensity metrics.

    Attributes:
        tenant_id: Tenant identifier for scoping.
        data_points: Ordered list of trend data points (earliest
            first).
        location_yoy_pct: Year-over-year change in location-based
            emissions (most recent vs. prior period) as a percentage.
        market_yoy_pct: Year-over-year change in market-based
            emissions (most recent vs. prior period) as a percentage.
        location_cagr: Compound annual growth rate for location-based
            emissions across all periods.
        market_cagr: Compound annual growth rate for market-based
            emissions across all periods.
        pif_trend: Direction of the PIF trend across periods.
        re100_trend: Direction of the RE100 percentage trend.
        intensity_metrics: Dictionary of intensity metric type to
            IntensityResult for the current period.
        sbti_on_track: Whether the entity is on track to meet its
            Science Based Target for Scope 2 (market-based).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for scoping",
    )
    data_points: List[TrendDataPoint] = Field(
        default_factory=list,
        description=(
            "Ordered trend data points (earliest first)"
        ),
    )
    location_yoy_pct: Optional[Decimal] = Field(
        default=None,
        description=(
            "Year-over-year change in location-based emissions "
            "as a percentage"
        ),
    )
    market_yoy_pct: Optional[Decimal] = Field(
        default=None,
        description=(
            "Year-over-year change in market-based emissions "
            "as a percentage"
        ),
    )
    location_cagr: Optional[Decimal] = Field(
        default=None,
        description=(
            "Compound annual growth rate for location-based "
            "emissions"
        ),
    )
    market_cagr: Optional[Decimal] = Field(
        default=None,
        description=(
            "Compound annual growth rate for market-based "
            "emissions"
        ),
    )
    pif_trend: Optional[TrendDirection] = Field(
        default=None,
        description="Direction of the PIF trend across periods",
    )
    re100_trend: Optional[TrendDirection] = Field(
        default=None,
        description="Direction of the RE100 percentage trend",
    )
    intensity_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Intensity metric type to IntensityResult for current "
            "period"
        ),
    )
    sbti_on_track: bool = Field(
        default=False,
        description=(
            "Whether entity is on track to meet SBTi Scope 2 target"
        ),
    )

    @field_validator("data_points")
    @classmethod
    def _validate_trend_count(
        cls, v: List[TrendDataPoint]
    ) -> List[TrendDataPoint]:
        """Validate that trend data points do not exceed maximum."""
        if len(v) > MAX_TREND_POINTS:
            raise ValueError(
                f"Maximum {MAX_TREND_POINTS} trend data points, "
                f"got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 17. IntensityResult
# ---------------------------------------------------------------------------


class IntensityResult(BaseModel):
    """Emission intensity result for a single metric type and period.

    Normalises both location-based and market-based emissions by a
    business activity denominator to enable comparison across periods
    and peer companies.

    Attributes:
        metric_type: The intensity denominator type (revenue, FTE,
            floor area, or production unit).
        location_intensity: Location-based intensity value.
        market_intensity: Market-based intensity value.
        unit: Unit string describing the intensity (e.g.
            "tCO2e/million USD", "tCO2e/FTE").
        period: Reporting period label.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    metric_type: IntensityMetric = Field(
        ...,
        description="Intensity denominator type",
    )
    location_intensity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Location-based intensity value",
    )
    market_intensity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Market-based intensity value",
    )
    unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Unit describing the intensity (e.g. tCO2e/million USD)"
        ),
    )
    period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reporting period label",
    )


# ---------------------------------------------------------------------------
# 18. ComplianceRequirement
# ---------------------------------------------------------------------------


class ComplianceRequirement(BaseModel):
    """A single compliance requirement within a framework check.

    Represents one specific requirement that the reconciliation output
    must satisfy for a given reporting framework. Each requirement is
    independently assessed as met or not met.

    Attributes:
        requirement_id: Unique identifier for the requirement
            (e.g. "GHG-S2-001").
        description: Human-readable description of the requirement.
        met: Whether the requirement is met.
        evidence: Evidence or reference demonstrating compliance
            (e.g. "Location total disclosed in Table 1, Row 3").
        notes: Additional notes or context.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    requirement_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique identifier for the requirement",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable description of the requirement",
    )
    met: bool = Field(
        ...,
        description="Whether the requirement is met",
    )
    evidence: str = Field(
        default="",
        max_length=2000,
        description="Evidence demonstrating compliance",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Additional notes or context",
    )


# ---------------------------------------------------------------------------
# 19. ComplianceCheckResult
# ---------------------------------------------------------------------------


class ComplianceCheckResult(BaseModel):
    """Result of a compliance check against a single reporting framework.

    Evaluates the reconciliation output against all requirements of a
    specific framework and produces an overall compliance status with
    a numeric score.

    Attributes:
        framework: The reporting framework checked.
        status: Overall compliance status.
        requirements_total: Total number of requirements checked.
        requirements_met: Number of requirements met.
        requirements: Detailed list of individual requirement results.
        score: Compliance score as a fraction (0.0 to 1.0).
        findings: List of findings or observations.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    framework: ReportingFramework = Field(
        ...,
        description="Reporting framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    requirements_total: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements checked",
    )
    requirements_met: int = Field(
        default=0,
        ge=0,
        description="Number of requirements met",
    )
    requirements: List[ComplianceRequirement] = Field(
        default_factory=list,
        description="Detailed individual requirement results",
    )
    score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Compliance score as a fraction (0.0 to 1.0)",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Findings or observations",
    )

    @field_validator("requirements")
    @classmethod
    def _validate_requirements_count(
        cls, v: List[ComplianceRequirement]
    ) -> List[ComplianceRequirement]:
        """Validate that requirements do not exceed maximum."""
        if len(v) > MAX_REQUIREMENTS_PER_FRAMEWORK:
            raise ValueError(
                f"Maximum {MAX_REQUIREMENTS_PER_FRAMEWORK} "
                f"requirements per framework, got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 20. ReconciliationRequest
# ---------------------------------------------------------------------------


class ReconciliationRequest(BaseModel):
    """Request to perform a dual-reporting reconciliation.

    The primary input to the reconciliation pipeline. Provides
    location-based and market-based upstream results along with
    configuration for which frameworks to check, whether to include
    trend and quality analysis, and optional historical results for
    trend computation.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy scoping.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        location_results: List of location-based upstream results.
        market_results: List of market-based upstream results.
        frameworks: Optional list of reporting frameworks to check.
            If None, all frameworks are checked.
        include_trends: Whether to include trend analysis.
        include_quality: Whether to include quality assessment.
        historical_results: Optional historical reconciliation
            results for trend analysis.
        intensity_denominators: Optional intensity denominators
            for the current period (metric type to value).
        base_year_location_tco2e: Optional base year location-based
            total for target tracking.
        base_year_market_tco2e: Optional base year market-based
            total for target tracking.
        sbti_target_year: Optional SBTi target year.
        sbti_target_reduction_pct: Optional SBTi target reduction
            percentage.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy scoping",
    )
    period_start: date = Field(
        ...,
        description="Start date of the reporting period",
    )
    period_end: date = Field(
        ...,
        description="End date of the reporting period",
    )
    location_results: List[UpstreamResult] = Field(
        default_factory=list,
        description="Location-based upstream results",
    )
    market_results: List[UpstreamResult] = Field(
        default_factory=list,
        description="Market-based upstream results",
    )
    frameworks: Optional[List[ReportingFramework]] = Field(
        default=None,
        description=(
            "Reporting frameworks to check; if None all are checked"
        ),
    )
    include_trends: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )
    include_quality: bool = Field(
        default=True,
        description="Whether to include quality assessment",
    )
    historical_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Historical reconciliation results for trend analysis"
        ),
    )
    intensity_denominators: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description=(
            "Intensity denominators for current period "
            "(metric type to value)"
        ),
    )
    base_year_location_tco2e: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Base year location-based total for target tracking",
    )
    base_year_market_tco2e: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Base year market-based total for target tracking",
    )
    sbti_target_year: Optional[int] = Field(
        default=None,
        ge=2020,
        le=2100,
        description="SBTi target year",
    )
    sbti_target_reduction_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="SBTi target reduction percentage",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )

    @field_validator("period_end")
    @classmethod
    def _period_end_after_start(cls, v: date, info: Any) -> date:
        """Validate that period_end is on or after period_start."""
        start = info.data.get("period_start")
        if start is not None and v < start:
            raise ValueError(
                f"period_end ({v}) must be on or after "
                f"period_start ({start})"
            )
        return v

    @field_validator("location_results", "market_results")
    @classmethod
    def _validate_results_count(
        cls, v: List[UpstreamResult]
    ) -> List[UpstreamResult]:
        """Validate that upstream results do not exceed maximum."""
        if len(v) > MAX_UPSTREAM_RESULTS:
            raise ValueError(
                f"Maximum {MAX_UPSTREAM_RESULTS} upstream results "
                f"per method, got {len(v)}"
            )
        return v

    @field_validator("frameworks")
    @classmethod
    def _validate_frameworks_count(
        cls, v: Optional[List[ReportingFramework]]
    ) -> Optional[List[ReportingFramework]]:
        """Validate that frameworks do not exceed maximum."""
        if v is not None and len(v) > MAX_FRAMEWORKS:
            raise ValueError(
                f"Maximum {MAX_FRAMEWORKS} frameworks per request, "
                f"got {len(v)}"
            )
        return v


# ---------------------------------------------------------------------------
# 21. ReconciliationReport
# ---------------------------------------------------------------------------


class ReconciliationReport(BaseModel):
    """Complete output of a dual-reporting reconciliation run.

    The primary output of the reconciliation pipeline, containing
    all analysis results assembled from the ten pipeline stages.

    Attributes:
        reconciliation_id: Unique identifier for this reconciliation
            run (UUID).
        status: Final status of the reconciliation.
        tenant_id: Tenant identifier.
        period: Human-readable period label (e.g. "2024-01-01 to
            2024-12-31").
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        total_location_tco2e: Aggregate location-based emissions.
        total_market_tco2e: Aggregate market-based emissions.
        pif: Procurement Impact Factor: 1 - (market / location).
        discrepancy_pct: Percentage difference between methods
            relative to the larger total.
        discrepancy_direction: Direction of the discrepancy.
        by_energy_type: Breakdown by energy type.
        by_facility: Breakdown by facility.
        discrepancy_report: Detailed discrepancy analysis.
        quality_assessment: Quality assessment results.
        reporting_tables: Framework-specific reporting tables.
        trend_report: Multi-period trend analysis.
        compliance_results: Compliance check results per framework.
        flags: All flags raised during processing.
        provenance_hash: SHA-256 hash over the entire report.
        timestamp: UTC timestamp of report generation.
        processing_time_ms: Total processing duration in milliseconds.
        pipeline_stages_completed: List of completed pipeline stages.
        metadata: Additional key-value pairs for extensibility.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique reconciliation run identifier (UUID)",
    )
    status: ReconciliationStatus = Field(
        default=ReconciliationStatus.PENDING,
        description="Final status of the reconciliation",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier",
    )
    period: str = Field(
        default="",
        max_length=100,
        description=(
            "Human-readable period label "
            "(e.g. 2024-01-01 to 2024-12-31)"
        ),
    )
    period_start: Optional[date] = Field(
        default=None,
        description="Start date of the reporting period",
    )
    period_end: Optional[date] = Field(
        default=None,
        description="End date of the reporting period",
    )
    total_location_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate location-based emissions in tCO2e",
    )
    total_market_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate market-based emissions in tCO2e",
    )
    pif: Decimal = Field(
        default=Decimal("0"),
        description=(
            "Procurement Impact Factor: 1 - (market / location)"
        ),
    )
    discrepancy_pct: Decimal = Field(
        default=Decimal("0"),
        description=(
            "Percentage difference between methods relative to "
            "the larger total"
        ),
    )
    discrepancy_direction: DiscrepancyDirection = Field(
        default=DiscrepancyDirection.EQUAL,
        description="Direction of the discrepancy",
    )
    by_energy_type: List[EnergyTypeBreakdown] = Field(
        default_factory=list,
        description="Breakdown by energy type",
    )
    by_facility: List[FacilityBreakdown] = Field(
        default_factory=list,
        description="Breakdown by facility",
    )
    discrepancy_report: Optional[DiscrepancyReport] = Field(
        default=None,
        description="Detailed discrepancy analysis",
    )
    quality_assessment: Optional[QualityAssessment] = Field(
        default=None,
        description="Quality assessment results",
    )
    reporting_tables: Optional[ReportingTableSet] = Field(
        default=None,
        description="Framework-specific reporting tables",
    )
    trend_report: Optional[TrendReport] = Field(
        default=None,
        description="Multi-period trend analysis",
    )
    compliance_results: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Compliance check results per framework",
    )
    flags: List[Flag] = Field(
        default_factory=list,
        description="All flags raised during processing",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description="SHA-256 hash over the entire report",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of report generation",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    pipeline_stages_completed: List[str] = Field(
        default_factory=list,
        description="List of completed pipeline stage names",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for extensibility",
    )


# ---------------------------------------------------------------------------
# 22. BatchReconciliationRequest
# ---------------------------------------------------------------------------


class BatchReconciliationRequest(BaseModel):
    """Request to perform reconciliation across multiple periods.

    Enables batch processing of reconciliations for multiple reporting
    periods in a single request, such as monthly reconciliations for
    an entire fiscal year.

    Attributes:
        periods: List of (period_start, period_end) date pairs.
        tenant_id: Tenant identifier for multi-tenancy scoping.
        batch_id: Unique identifier for this batch job (UUID).
        frameworks: Optional list of reporting frameworks to check.
        include_trends: Whether to include trend analysis.
        include_quality: Whether to include quality assessment.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    periods: List[Dict[str, date]] = Field(
        ...,
        min_length=1,
        description=(
            "List of period definitions; each dict must contain "
            "'period_start' and 'period_end' date keys"
        ),
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tenant identifier for multi-tenancy scoping",
    )
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier (UUID)",
    )
    frameworks: Optional[List[ReportingFramework]] = Field(
        default=None,
        description="Reporting frameworks to check",
    )
    include_trends: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )
    include_quality: bool = Field(
        default=True,
        description="Whether to include quality assessment",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    @field_validator("periods")
    @classmethod
    def _validate_periods(
        cls, v: List[Dict[str, date]]
    ) -> List[Dict[str, date]]:
        """Validate periods count and structure."""
        if len(v) > MAX_BATCH_PERIODS:
            raise ValueError(
                f"Maximum {MAX_BATCH_PERIODS} periods per batch, "
                f"got {len(v)}"
            )
        for idx, period in enumerate(v):
            if "period_start" not in period or "period_end" not in period:
                raise ValueError(
                    f"Period at index {idx} must contain "
                    f"'period_start' and 'period_end' keys"
                )
            if period["period_end"] < period["period_start"]:
                raise ValueError(
                    f"Period at index {idx}: period_end "
                    f"({period['period_end']}) must be on or after "
                    f"period_start ({period['period_start']})"
                )
        return v


# ---------------------------------------------------------------------------
# 23. BatchReconciliationResult
# ---------------------------------------------------------------------------


class BatchReconciliationResult(BaseModel):
    """Result of a batch reconciliation across multiple periods.

    Attributes:
        batch_id: Unique identifier of the batch job.
        status: Overall batch status.
        total_periods: Total number of periods in the batch.
        completed: Number of periods completed successfully.
        failed: Number of periods that failed.
        results: List of reconciliation reports, one per completed
            period.
        failed_periods: List of period labels that failed with
            error messages.
        processing_time_ms: Total processing duration in milliseconds.
        timestamp: UTC timestamp of batch completion.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the batch job",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Overall batch status",
    )
    total_periods: int = Field(
        default=0,
        ge=0,
        description="Total number of periods in the batch",
    )
    completed: int = Field(
        default=0,
        ge=0,
        description="Number of periods completed successfully",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Number of periods that failed",
    )
    results: List[ReconciliationReport] = Field(
        default_factory=list,
        description=(
            "Reconciliation reports, one per completed period"
        ),
    )
    failed_periods: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Period labels that failed with error messages"
        ),
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of batch completion",
    )


# ---------------------------------------------------------------------------
# 24. ExportRequest
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """Request to export reconciliation results in a specific format.

    Attributes:
        reconciliation_id: Reference to the reconciliation run to
            export.
        format: Desired export format (JSON, CSV, or EXCEL).
        frameworks: Optional list of frameworks to include in export.
            If None, all available framework tables are included.
        include_discrepancy_report: Whether to include the detailed
            discrepancy analysis.
        include_quality_assessment: Whether to include the quality
            assessment.
        include_trend_report: Whether to include the trend analysis.
        include_compliance: Whether to include compliance results.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the reconciliation run to export",
    )
    format: ExportFormat = Field(
        ...,
        description="Desired export format",
    )
    frameworks: Optional[List[ReportingFramework]] = Field(
        default=None,
        description=(
            "Frameworks to include in export; if None all available "
            "tables are included"
        ),
    )
    include_discrepancy_report: bool = Field(
        default=True,
        description="Whether to include discrepancy analysis",
    )
    include_quality_assessment: bool = Field(
        default=True,
        description="Whether to include quality assessment",
    )
    include_trend_report: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )
    include_compliance: bool = Field(
        default=True,
        description="Whether to include compliance results",
    )


# ---------------------------------------------------------------------------
# 25. AggregationResult
# ---------------------------------------------------------------------------


class AggregationResult(BaseModel):
    """Result of aggregating multiple reconciliation outputs.

    Used for portfolio-level or multi-entity aggregation of Scope 2
    dual-reporting results.

    Attributes:
        reconciliation_id: Identifier for this aggregation result.
        aggregation_type: Type of aggregation performed (e.g.
            "portfolio", "business_unit", "region").
        total_location: Aggregate location-based emissions in tCO2e.
        total_market: Aggregate market-based emissions in tCO2e.
        breakdown: Dictionary of group keys to sub-totals. Each
            sub-total is a dict with "location_tco2e" and
            "market_tco2e" keys.
        count: Number of reconciliation runs aggregated.
        pif: Portfolio-level Procurement Impact Factor.
        discrepancy_pct: Portfolio-level discrepancy percentage.
        direction: Portfolio-level discrepancy direction.
        provenance_hash: SHA-256 hash over the aggregation for
            audit trail integrity.
        timestamp: UTC timestamp of aggregation completion.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    reconciliation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identifier for this aggregation result",
    )
    aggregation_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Type of aggregation (e.g. portfolio, business_unit, "
            "region)"
        ),
    )
    total_location: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate location-based emissions in tCO2e",
    )
    total_market: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Aggregate market-based emissions in tCO2e",
    )
    breakdown: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description=(
            "Group keys to sub-totals; each sub-total has "
            "location_tco2e and market_tco2e keys"
        ),
    )
    count: int = Field(
        default=0,
        ge=0,
        description="Number of reconciliation runs aggregated",
    )
    pif: Decimal = Field(
        default=Decimal("0"),
        description="Portfolio-level Procurement Impact Factor",
    )
    discrepancy_pct: Decimal = Field(
        default=Decimal("0"),
        description="Portfolio-level discrepancy percentage",
    )
    direction: DiscrepancyDirection = Field(
        default=DiscrepancyDirection.EQUAL,
        description="Portfolio-level discrepancy direction",
    )
    provenance_hash: str = Field(
        default="",
        max_length=128,
        description=(
            "SHA-256 hash over the aggregation for audit trail"
        ),
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of aggregation completion",
    )


# =============================================================================
# Type Aliases (backward-compatible names used by pipeline engine)
# =============================================================================

#: Alias for Discrepancy used by the pipeline engine.
DiscrepancyItem = Discrepancy

#: Alias for ComplianceRequirement used by the pipeline engine.
ComplianceIssue = ComplianceRequirement


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Module-level constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "MAX_UPSTREAM_RESULTS",
    "MAX_BATCH_PERIODS",
    "MAX_FACILITIES",
    "MAX_DISCREPANCIES",
    "MAX_FRAMEWORKS",
    "MAX_TREND_POINTS",
    "MAX_REQUIREMENTS_PER_FRAMEWORK",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DECIMAL_INF",
    # Enumerations (22)
    "EnergyType",
    "Scope2Method",
    "UpstreamAgent",
    "DiscrepancyType",
    "DiscrepancyDirection",
    "MaterialityLevel",
    "QualityDimension",
    "QualityGrade",
    "EFHierarchyPriority",
    "ReportingFramework",
    "FlagType",
    "FlagSeverity",
    "ReconciliationStatus",
    "IntensityMetric",
    "TrendDirection",
    "PipelineStage",
    "ExportFormat",
    "ComplianceStatus",
    "DataQualityTier",
    "GWPSource",
    "EmissionGas",
    "BatchStatus",
    # Constant tables
    "GWP_VALUES",
    "MATERIALITY_THRESHOLDS",
    "QUALITY_WEIGHTS",
    "QUALITY_GRADE_THRESHOLDS",
    "EF_HIERARCHY_QUALITY_SCORES",
    "RESIDUAL_MIX_FACTORS",
    "UPSTREAM_AGENT_MAPPING",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    # Data models (25)
    "ResidualMixFactor",
    "UpstreamResult",
    "EnergyTypeBreakdown",
    "FacilityBreakdown",
    "ReconciliationWorkspace",
    "Discrepancy",
    "WaterfallItem",
    "WaterfallDecomposition",
    "DiscrepancyReport",
    "Flag",
    "QualityScore",
    "QualityAssessment",
    "FrameworkTable",
    "ReportingTableSet",
    "TrendDataPoint",
    "TrendReport",
    "IntensityResult",
    "ComplianceRequirement",
    "ComplianceCheckResult",
    "ReconciliationRequest",
    "ReconciliationReport",
    "BatchReconciliationRequest",
    "BatchReconciliationResult",
    "ExportRequest",
    "AggregationResult",
    # Decimal constants used by pipeline engine
    "DECIMAL_PLACES",
    "ZERO",
    "ONE_HUNDRED",
    # Type aliases (backward-compatible names)
    "DiscrepancyItem",
    "ComplianceIssue",
]
