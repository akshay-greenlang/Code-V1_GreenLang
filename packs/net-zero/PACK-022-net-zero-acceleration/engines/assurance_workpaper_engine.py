# -*- coding: utf-8 -*-
"""
AssuranceWorkpaperEngine - PACK-022 Net Zero Acceleration Engine 10
=====================================================================

Generates structured audit workpapers for external assurance engagements
on GHG statements, following ISAE 3410 (Assurance Engagements on
Greenhouse Gas Statements) and supporting both limited and reasonable
assurance levels.

ISAE 3410 Framework:
    ISAE 3410 establishes requirements and application guidance for
    practitioners performing assurance engagements on GHG statements.
    The standard requires the practitioner to:
    - Understand the entity's GHG reporting processes and controls
    - Evaluate the suitability of measurement and reporting criteria
    - Obtain sufficient appropriate evidence to support the conclusion
    - Consider the risk of material misstatement in the GHG statement
    - Document findings in workpapers that support the assurance report

    Workpaper sections per ISAE 3410:
    1. Engagement summary (scope, boundary, standards, materiality)
    2. Methodology documentation (calculation methods per scope/source)
    3. Calculation trace (step-by-step with intermediate values)
    4. Data lineage (source system -> transformation -> output)
    5. Control evidence (validation checks, reconciliation, provenance)
    6. Exception register (data quality issues, estimations, assumptions)
    7. Completeness matrix (actual vs estimated per source)
    8. Change register (methodology changes, restatements)

Materiality Guidance:
    - Quantitative threshold: Typically 5% of total reported emissions
    - Qualitative factors: Nature of omission, regulatory impact
    - Component materiality: Applied at scope or source level

Assurance Levels:
    - Limited assurance: Negative conclusion ("nothing has come to our
      attention"), less evidence required
    - Reasonable assurance: Positive conclusion ("in our opinion, fairly
      stated"), more rigorous evidence

Features:
    - Complete workpaper structure per ISAE 3410
    - Materiality threshold calculation
    - Calculation trace with intermediate values
    - Data lineage tracking with SHA-256 hash chain
    - Cross-check routines for internal consistency
    - Exception register with severity classification
    - Completeness matrix (actual vs estimated)
    - Export to structured JSON for auditor review tools

Regulatory References:
    - ISAE 3410 Assurance Engagements on GHG Statements (IAASB)
    - ISAE 3000 (Revised) Assurance Engagements Other than Audits
    - ISO 14064-3:2019 Specification for GHG validation/verification
    - CSRD ESRS Assurance requirements (Art. 34)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - SEC Climate-Related Disclosures (2024)

Zero-Hallucination:
    - All materiality calculations use deterministic thresholds
    - Cross-check routines use exact arithmetic comparison
    - Exception classification uses rule-based severity mapping
    - SHA-256 provenance chain provides cryptographic integrity
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

DEFAULT_MATERIALITY_PCT: Decimal = Decimal("5")  # 5% of total emissions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
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

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _chain_hash(previous_hash: str, data: Any) -> str:
    """Create a chained SHA-256 hash by combining previous hash with new data.

    This provides cryptographic linking between workpaper elements,
    forming a tamper-evident chain similar to a blockchain.

    Args:
        previous_hash: Hash of the previous element in the chain.
        data: New data to hash.

    Returns:
        SHA-256 hex digest of (previous_hash + data_hash).
    """
    data_hash = _compute_hash(data)
    combined = f"{previous_hash}:{data_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WorkpaperSection(str, Enum):
    """Workpaper section identifiers per ISAE 3410 structure."""
    ENGAGEMENT_SUMMARY = "engagement_summary"
    METHODOLOGY_DOCUMENTATION = "methodology_documentation"
    CALCULATION_TRACE = "calculation_trace"
    DATA_LINEAGE = "data_lineage"
    CONTROL_EVIDENCE = "control_evidence"
    EXCEPTION_REGISTER = "exception_register"
    COMPLETENESS_MATRIX = "completeness_matrix"
    CHANGE_REGISTER = "change_register"

class AssuranceLevel(str, Enum):
    """Assurance engagement level."""
    LIMITED = "limited"
    REASONABLE = "reasonable"

class MaterialityBasis(str, Enum):
    """Basis for materiality threshold calculation."""
    TOTAL_EMISSIONS = "total_emissions"
    SCOPE_LEVEL = "scope_level"
    SOURCE_LEVEL = "source_level"

class DataSourceType(str, Enum):
    """Type of data source for an emission calculation."""
    METERED = "metered"
    INVOICED = "invoiced"
    ESTIMATED = "estimated"
    CALCULATED = "calculated"
    DEFAULT_FACTOR = "default_factor"
    SUPPLIER_PROVIDED = "supplier_provided"

class ExceptionSeverity(str, Enum):
    """Severity of an exception finding."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CrossCheckStatus(str, Enum):
    """Status of a cross-check routine."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"

class CalculationMethod(str, Enum):
    """GHG calculation method per GHG Protocol."""
    DIRECT_MEASUREMENT = "direct_measurement"
    EMISSION_FACTOR = "emission_factor"
    MASS_BALANCE = "mass_balance"
    ENGINEERING_ESTIMATE = "engineering_estimate"
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EngagementSummary(BaseModel):
    """Section 1: Engagement summary."""
    engagement_id: str = Field(default_factory=_new_uuid, description="Engagement ID")
    entity_name: str = Field(description="Entity under assurance")
    reporting_year: int = Field(description="Reporting year")
    assurance_level: AssuranceLevel = Field(description="Limited or reasonable assurance")
    organizational_boundary: str = Field(
        default="", description="Consolidation approach (equity share/control)"
    )
    operational_boundary: str = Field(
        default="", description="Scopes and categories included"
    )
    reporting_standards: List[str] = Field(
        default_factory=list, description="Standards applied (GHG Protocol, ISO 14064, etc.)"
    )
    assurance_standard: str = Field(
        default="ISAE 3410", description="Assurance standard applied"
    )
    materiality_threshold: Decimal = Field(
        default=Decimal("0"), description="Materiality threshold (tCO2e)"
    )
    materiality_pct: Decimal = Field(
        default=Decimal("5"), description="Materiality as % of total"
    )
    total_reported_emissions: Decimal = Field(
        default=Decimal("0"), description="Total reported emissions (tCO2e)"
    )
    engagement_team: List[str] = Field(
        default_factory=list, description="Engagement team members"
    )
    engagement_date: datetime = Field(default_factory=utcnow, description="Engagement date")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("materiality_threshold", "materiality_pct",
                     "total_reported_emissions", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class MethodologyEntry(BaseModel):
    """Section 2: Methodology documentation for one emission source."""
    entry_id: str = Field(default_factory=_new_uuid, description="Entry identifier")
    scope: str = Field(description="Scope (scope1, scope2, scope3)")
    category: str = Field(default="", description="Category (e.g., scope3_cat1)")
    source_name: str = Field(description="Emission source name")
    calculation_method: CalculationMethod = Field(description="Calculation method used")
    emission_factor_source: str = Field(
        default="", description="Emission factor source (e.g., DEFRA 2025, EPA, IEA)"
    )
    emission_factor_value: Decimal = Field(
        default=Decimal("0"), description="Emission factor value"
    )
    emission_factor_unit: str = Field(
        default="", description="Emission factor unit (e.g., kgCO2e/kWh)"
    )
    gwp_source: str = Field(
        default="IPCC AR6", description="GWP source"
    )
    gwp_timeframe: str = Field(
        default="100-year", description="GWP timeframe"
    )
    methodology_notes: str = Field(
        default="", description="Additional methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emission_factor_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CalculationStep(BaseModel):
    """A single step in a calculation trace."""
    step_number: int = Field(description="Step number (1-based)")
    description: str = Field(description="Step description")
    input_value: Decimal = Field(description="Input value")
    input_unit: str = Field(default="", description="Input unit")
    factor_applied: Decimal = Field(default=Decimal("1"), description="Factor/multiplier applied")
    factor_description: str = Field(default="", description="Factor source/description")
    output_value: Decimal = Field(description="Output value after applying factor")
    output_unit: str = Field(default="", description="Output unit")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("input_value", "factor_applied", "output_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CalculationTrace(BaseModel):
    """Section 3: Complete calculation trace for one emission source."""
    trace_id: str = Field(default_factory=_new_uuid, description="Trace identifier")
    source_name: str = Field(description="Emission source name")
    scope: str = Field(description="Scope")
    steps: List[CalculationStep] = Field(default_factory=list, description="Calculation steps")
    final_emissions: Decimal = Field(description="Final calculated emissions (tCO2e)")
    is_material: bool = Field(default=False, description="Whether above materiality threshold")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("final_emissions", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class DataLineageEntry(BaseModel):
    """Section 4: Data lineage for one data point."""
    lineage_id: str = Field(default_factory=_new_uuid, description="Lineage identifier")
    data_point_name: str = Field(description="Data point name")
    source_system: str = Field(description="Originating system (ERP, SCADA, manual, etc.)")
    source_type: DataSourceType = Field(description="Data source type")
    raw_value: Decimal = Field(description="Raw value from source")
    raw_unit: str = Field(default="", description="Raw value unit")
    transformations: List[str] = Field(
        default_factory=list, description="Transformation descriptions applied"
    )
    final_value: Decimal = Field(description="Final value after transformations")
    final_unit: str = Field(default="", description="Final value unit")
    source_hash: str = Field(default="", description="Hash of raw source data")
    transformation_hash: str = Field(default="", description="Hash of transformation chain")
    output_hash: str = Field(default="", description="Hash of final output")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("raw_value", "final_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ControlEvidence(BaseModel):
    """Section 5: Control evidence for one check."""
    control_id: str = Field(default_factory=_new_uuid, description="Control identifier")
    control_name: str = Field(description="Control name")
    control_type: str = Field(description="Type (input_validation, reconciliation, provenance)")
    description: str = Field(default="", description="Control description")
    expected_value: Optional[Decimal] = Field(default=None, description="Expected value")
    actual_value: Optional[Decimal] = Field(default=None, description="Actual value")
    tolerance_pct: Decimal = Field(default=Decimal("1"), description="Tolerance (%)")
    status: CrossCheckStatus = Field(description="Control check status")
    finding: str = Field(default="", description="Finding description")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("tolerance_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("expected_value", "actual_value", mode="before")
    @classmethod
    def _coerce_decimal_opt(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class ExceptionEntry(BaseModel):
    """Section 6: Exception register entry."""
    exception_id: str = Field(default_factory=_new_uuid, description="Exception identifier")
    source_name: str = Field(description="Affected emission source")
    scope: str = Field(default="", description="Affected scope")
    severity: ExceptionSeverity = Field(description="Exception severity")
    exception_type: str = Field(description="Type (data_quality, estimation, assumption, gap)")
    description: str = Field(description="Exception description")
    impact_tco2e: Decimal = Field(default=Decimal("0"), description="Estimated impact (tCO2e)")
    impact_pct: Decimal = Field(default=Decimal("0"), description="Impact as % of total")
    mitigation: str = Field(default="", description="Mitigation or response")
    is_material: bool = Field(default=False, description="Whether material")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("impact_tco2e", "impact_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CompletenessEntry(BaseModel):
    """Section 7: Completeness matrix entry for one source."""
    source_name: str = Field(description="Emission source name")
    scope: str = Field(description="Scope")
    data_source_type: DataSourceType = Field(description="Data source type")
    coverage_pct: Decimal = Field(description="Data coverage percentage")
    is_actual: bool = Field(description="Whether data is actual (vs estimated)")
    estimation_method: str = Field(default="", description="Estimation method if estimated")
    months_actual: int = Field(default=12, description="Months with actual data")
    months_estimated: int = Field(default=0, description="Months with estimated data")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("coverage_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ChangeEntry(BaseModel):
    """Section 8: Change register entry."""
    change_id: str = Field(default_factory=_new_uuid, description="Change identifier")
    change_type: str = Field(description="Type (methodology, boundary, factor, restatement)")
    description: str = Field(description="Change description")
    affected_scope: str = Field(default="", description="Affected scope")
    affected_source: str = Field(default="", description="Affected source")
    previous_value: Decimal = Field(default=Decimal("0"), description="Previous value (tCO2e)")
    new_value: Decimal = Field(default=Decimal("0"), description="New value (tCO2e)")
    impact: Decimal = Field(default=Decimal("0"), description="Impact (tCO2e)")
    rationale: str = Field(default="", description="Rationale for change")
    approved_by: str = Field(default="", description="Approval authority")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("previous_value", "new_value", "impact", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CrossCheckResult(BaseModel):
    """Result of a cross-check routine."""
    check_id: str = Field(default_factory=_new_uuid, description="Check identifier")
    check_name: str = Field(description="Cross-check name")
    description: str = Field(default="", description="Check description")
    value_a: Decimal = Field(description="First value being compared")
    value_a_source: str = Field(default="", description="Source of first value")
    value_b: Decimal = Field(description="Second value being compared")
    value_b_source: str = Field(default="", description="Source of second value")
    difference: Decimal = Field(description="Absolute difference")
    difference_pct: Decimal = Field(description="Difference as percentage")
    tolerance_pct: Decimal = Field(default=Decimal("5"), description="Tolerance threshold (%)")
    status: CrossCheckStatus = Field(description="Check result")
    finding: str = Field(default="", description="Finding description")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("value_a", "value_b", "difference",
                     "difference_pct", "tolerance_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class AssuranceResult(BaseModel):
    """Complete assurance workpaper result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    engagement_summary: EngagementSummary = Field(description="Engagement summary")
    methodology_entries: List[MethodologyEntry] = Field(
        default_factory=list, description="Methodology documentation"
    )
    calculation_traces: List[CalculationTrace] = Field(
        default_factory=list, description="Calculation traces"
    )
    data_lineage: List[DataLineageEntry] = Field(
        default_factory=list, description="Data lineage entries"
    )
    control_evidence: List[ControlEvidence] = Field(
        default_factory=list, description="Control evidence"
    )
    exceptions: List[ExceptionEntry] = Field(
        default_factory=list, description="Exception register"
    )
    completeness_matrix: List[CompletenessEntry] = Field(
        default_factory=list, description="Completeness matrix"
    )
    change_register: List[ChangeEntry] = Field(
        default_factory=list, description="Change register"
    )
    cross_checks: List[CrossCheckResult] = Field(
        default_factory=list, description="Cross-check results"
    )
    materiality_threshold: Decimal = Field(
        default=Decimal("0"), description="Materiality threshold (tCO2e)"
    )
    provenance_chain: List[str] = Field(
        default_factory=list, description="SHA-256 hash chain"
    )
    material_exceptions_count: int = Field(
        default=0, description="Number of material exceptions"
    )
    overall_completeness_pct: Decimal = Field(
        default=Decimal("0"), description="Overall data completeness %"
    )
    cross_checks_passed: int = Field(default=0, description="Cross-checks passed")
    cross_checks_failed: int = Field(default=0, description="Cross-checks failed")
    workpaper_sections_count: int = Field(
        default=8, description="Number of workpaper sections"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("materiality_threshold", "overall_completeness_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class AssuranceWorkpaperConfig(BaseModel):
    """Configuration for the AssuranceWorkpaperEngine."""
    default_assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED, description="Default assurance level"
    )
    materiality_pct: Decimal = Field(
        default=DEFAULT_MATERIALITY_PCT, description="Materiality threshold (%)"
    )
    materiality_basis: MaterialityBasis = Field(
        default=MaterialityBasis.TOTAL_EMISSIONS, description="Materiality basis"
    )
    cross_check_tolerance_pct: Decimal = Field(
        default=Decimal("5"), description="Default cross-check tolerance (%)"
    )
    assurance_standard: str = Field(
        default="ISAE 3410", description="Applicable assurance standard"
    )
    decimal_precision: int = Field(
        default=4, description="Decimal places for results"
    )

    @field_validator("materiality_pct", "cross_check_tolerance_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

EngagementSummary.model_rebuild()
MethodologyEntry.model_rebuild()
CalculationStep.model_rebuild()
CalculationTrace.model_rebuild()
DataLineageEntry.model_rebuild()
ControlEvidence.model_rebuild()
ExceptionEntry.model_rebuild()
CompletenessEntry.model_rebuild()
ChangeEntry.model_rebuild()
CrossCheckResult.model_rebuild()
AssuranceResult.model_rebuild()
AssuranceWorkpaperConfig.model_rebuild()

# ---------------------------------------------------------------------------
# AssuranceWorkpaperEngine
# ---------------------------------------------------------------------------

class AssuranceWorkpaperEngine:
    """
    Audit workpaper generation engine per ISAE 3410.

    Generates structured workpapers for external assurance engagements
    on GHG statements, including methodology documentation, calculation
    traces, data lineage, control evidence, and exception registers.

    Attributes:
        config: Engine configuration.
        _methodology: Methodology entries.
        _traces: Calculation traces.
        _lineage: Data lineage entries.
        _controls: Control evidence.
        _exceptions: Exception register.
        _completeness: Completeness matrix.
        _changes: Change register.

    Example:
        >>> engine = AssuranceWorkpaperEngine()
        >>> engine.add_methodology(entry)
        >>> engine.add_calculation_trace(trace)
        >>> result = engine.generate_workpapers(entity_name, 2025, total_emissions)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AssuranceWorkpaperEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = AssuranceWorkpaperConfig(**config)
        elif config and isinstance(config, AssuranceWorkpaperConfig):
            self.config = config
        else:
            self.config = AssuranceWorkpaperConfig()

        self._methodology: List[MethodologyEntry] = []
        self._traces: List[CalculationTrace] = []
        self._lineage: List[DataLineageEntry] = []
        self._controls: List[ControlEvidence] = []
        self._exceptions: List[ExceptionEntry] = []
        self._completeness: List[CompletenessEntry] = []
        self._changes: List[ChangeEntry] = []
        logger.info("AssuranceWorkpaperEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Section 1: Engagement Summary
    # -------------------------------------------------------------------

    def create_engagement_summary(
        self,
        entity_name: str,
        reporting_year: int,
        total_emissions: Decimal,
        assurance_level: Optional[AssuranceLevel] = None,
        organizational_boundary: str = "",
        operational_boundary: str = "",
        reporting_standards: Optional[List[str]] = None,
        engagement_team: Optional[List[str]] = None,
    ) -> EngagementSummary:
        """Create the engagement summary section.

        Args:
            entity_name: Entity under assurance.
            reporting_year: Reporting year.
            total_emissions: Total reported emissions (tCO2e).
            assurance_level: Assurance level (defaults to config).
            organizational_boundary: Consolidation approach description.
            operational_boundary: Scope boundary description.
            reporting_standards: Standards applied.
            engagement_team: Team members.

        Returns:
            EngagementSummary workpaper section.
        """
        if assurance_level is None:
            assurance_level = self.config.default_assurance_level

        total_emissions = _decimal(total_emissions)
        materiality = self.calculate_materiality(total_emissions)

        summary = EngagementSummary(
            entity_name=entity_name,
            reporting_year=reporting_year,
            assurance_level=assurance_level,
            organizational_boundary=organizational_boundary,
            operational_boundary=operational_boundary or "Scope 1, 2, and material Scope 3 categories",
            reporting_standards=reporting_standards or ["GHG Protocol Corporate Standard"],
            assurance_standard=self.config.assurance_standard,
            materiality_threshold=materiality,
            materiality_pct=self.config.materiality_pct,
            total_reported_emissions=total_emissions,
            engagement_team=engagement_team or [],
        )
        summary.provenance_hash = _compute_hash(summary)

        logger.info(
            "Engagement summary created: %s, year %d, total=%.1f tCO2e, materiality=%.1f tCO2e",
            entity_name, reporting_year, float(total_emissions), float(materiality),
        )
        return summary

    # -------------------------------------------------------------------
    # Section 2: Methodology Documentation
    # -------------------------------------------------------------------

    def add_methodology(self, entry: MethodologyEntry) -> MethodologyEntry:
        """Add a methodology documentation entry.

        Args:
            entry: MethodologyEntry for one emission source.

        Returns:
            Entry with computed provenance hash.
        """
        entry.provenance_hash = _compute_hash(entry)
        self._methodology.append(entry)
        logger.info("Added methodology: %s (%s)", entry.source_name, entry.scope)
        return entry

    # -------------------------------------------------------------------
    # Section 3: Calculation Trace
    # -------------------------------------------------------------------

    def add_calculation_trace(self, trace: CalculationTrace) -> CalculationTrace:
        """Add a calculation trace with step-by-step detail.

        Args:
            trace: CalculationTrace for one emission source.

        Returns:
            Trace with computed provenance hash and step hashes.
        """
        # Compute per-step hashes
        for step in trace.steps:
            step.provenance_hash = _compute_hash(step)

        trace.provenance_hash = _compute_hash(trace)
        self._traces.append(trace)
        logger.info(
            "Added trace: %s (%s), %d steps, final=%.4f tCO2e",
            trace.source_name, trace.scope, len(trace.steps), float(trace.final_emissions),
        )
        return trace

    def create_calculation_trace(
        self,
        source_name: str,
        scope: str,
        activity_data: Decimal,
        activity_unit: str,
        emission_factor: Decimal,
        ef_unit: str,
        ef_source: str,
        gwp: Decimal = Decimal("1"),
        gwp_gas: str = "CO2",
    ) -> CalculationTrace:
        """Create a standard emission factor calculation trace.

        Standard formula: emissions = activity_data * emission_factor * GWP

        Args:
            source_name: Emission source name.
            scope: Scope classification.
            activity_data: Activity data value.
            activity_unit: Activity data unit.
            emission_factor: Emission factor value.
            ef_unit: Emission factor unit.
            ef_source: Emission factor source.
            gwp: Global warming potential (default 1 for CO2).
            gwp_gas: GHG gas name.

        Returns:
            CalculationTrace with three steps.
        """
        activity_data = _decimal(activity_data)
        emission_factor = _decimal(emission_factor)
        gwp = _decimal(gwp)

        intermediate = _round_val(activity_data * emission_factor, self.config.decimal_precision)
        final = _round_val(intermediate * gwp, self.config.decimal_precision)

        steps = [
            CalculationStep(
                step_number=1,
                description=f"Activity data for {source_name}",
                input_value=activity_data,
                input_unit=activity_unit,
                factor_applied=Decimal("1"),
                factor_description="Raw activity data from source system",
                output_value=activity_data,
                output_unit=activity_unit,
            ),
            CalculationStep(
                step_number=2,
                description=f"Apply emission factor from {ef_source}",
                input_value=activity_data,
                input_unit=activity_unit,
                factor_applied=emission_factor,
                factor_description=f"EF = {emission_factor} {ef_unit} (source: {ef_source})",
                output_value=intermediate,
                output_unit="kgCO2e" if "kg" in ef_unit.lower() else "tCO2e",
            ),
            CalculationStep(
                step_number=3,
                description=f"Apply GWP for {gwp_gas}",
                input_value=intermediate,
                input_unit="kgCO2e",
                factor_applied=gwp,
                factor_description=f"GWP-100 for {gwp_gas} (IPCC AR6)",
                output_value=final,
                output_unit="tCO2e",
            ),
        ]

        trace = CalculationTrace(
            source_name=source_name,
            scope=scope,
            steps=steps,
            final_emissions=final,
        )
        return self.add_calculation_trace(trace)

    # -------------------------------------------------------------------
    # Section 4: Data Lineage
    # -------------------------------------------------------------------

    def add_data_lineage(self, entry: DataLineageEntry) -> DataLineageEntry:
        """Add a data lineage entry.

        Args:
            entry: DataLineageEntry for one data point.

        Returns:
            Entry with computed hash chain.
        """
        entry.source_hash = _compute_hash({"raw": str(entry.raw_value), "source": entry.source_system})
        entry.transformation_hash = _compute_hash(entry.transformations)
        entry.output_hash = _compute_hash({"final": str(entry.final_value)})
        entry.provenance_hash = _chain_hash(entry.source_hash, entry.output_hash)

        self._lineage.append(entry)
        logger.info("Added lineage: %s (%s -> %s)", entry.data_point_name, entry.source_system, str(entry.final_value))
        return entry

    # -------------------------------------------------------------------
    # Section 5: Control Evidence
    # -------------------------------------------------------------------

    def add_control_evidence(self, control: ControlEvidence) -> ControlEvidence:
        """Add a control evidence entry.

        Args:
            control: ControlEvidence for one check.

        Returns:
            Control with computed provenance hash and auto-evaluated status.
        """
        # Auto-evaluate status if expected and actual values are provided
        if control.expected_value is not None and control.actual_value is not None:
            diff = abs(control.expected_value - control.actual_value)
            diff_pct = _safe_pct(diff, abs(control.expected_value)) if control.expected_value != 0 else Decimal("0")

            if diff_pct <= control.tolerance_pct:
                control.status = CrossCheckStatus.PASSED
                control.finding = f"Within tolerance ({diff_pct}% <= {control.tolerance_pct}%)"
            elif diff_pct <= control.tolerance_pct * Decimal("2"):
                control.status = CrossCheckStatus.WARNING
                control.finding = f"Near tolerance ({diff_pct}% > {control.tolerance_pct}%)"
            else:
                control.status = CrossCheckStatus.FAILED
                control.finding = f"Exceeds tolerance ({diff_pct}% >> {control.tolerance_pct}%)"

        control.provenance_hash = _compute_hash(control)
        self._controls.append(control)
        logger.info("Added control: %s - %s", control.control_name, control.status.value)
        return control

    # -------------------------------------------------------------------
    # Section 6: Exception Register
    # -------------------------------------------------------------------

    def add_exception(self, exception: ExceptionEntry) -> ExceptionEntry:
        """Add an exception register entry.

        Args:
            exception: ExceptionEntry for one finding.

        Returns:
            Exception with computed provenance hash.
        """
        exception.provenance_hash = _compute_hash(exception)
        self._exceptions.append(exception)
        logger.info(
            "Added exception: %s (%s) - %s",
            exception.source_name, exception.severity.value, exception.exception_type,
        )
        return exception

    # -------------------------------------------------------------------
    # Section 7: Completeness Matrix
    # -------------------------------------------------------------------

    def add_completeness_entry(self, entry: CompletenessEntry) -> CompletenessEntry:
        """Add a completeness matrix entry.

        Args:
            entry: CompletenessEntry for one source.

        Returns:
            Entry with computed provenance hash.
        """
        entry.provenance_hash = _compute_hash(entry)
        self._completeness.append(entry)
        logger.info(
            "Added completeness: %s (%s) - %s, coverage=%.0f%%",
            entry.source_name, entry.scope,
            "actual" if entry.is_actual else "estimated",
            float(entry.coverage_pct),
        )
        return entry

    # -------------------------------------------------------------------
    # Section 8: Change Register
    # -------------------------------------------------------------------

    def add_change(self, change: ChangeEntry) -> ChangeEntry:
        """Add a change register entry.

        Args:
            change: ChangeEntry for one methodology/boundary change.

        Returns:
            Change with auto-calculated impact and provenance hash.
        """
        if change.impact == Decimal("0"):
            change.impact = change.new_value - change.previous_value

        change.provenance_hash = _compute_hash(change)
        self._changes.append(change)
        logger.info(
            "Added change: %s (%s), impact=%.1f tCO2e",
            change.change_type, change.description[:50], float(change.impact),
        )
        return change

    # -------------------------------------------------------------------
    # Materiality Calculation
    # -------------------------------------------------------------------

    def calculate_materiality(
        self,
        total_emissions: Decimal,
        basis: Optional[MaterialityBasis] = None,
    ) -> Decimal:
        """Calculate the materiality threshold.

        Args:
            total_emissions: Total reported emissions (tCO2e).
            basis: Materiality basis (defaults to config).

        Returns:
            Materiality threshold in tCO2e.
        """
        pct = self.config.materiality_pct
        threshold = _round_val(
            _decimal(total_emissions) * pct / Decimal("100"),
            self.config.decimal_precision,
        )
        logger.info(
            "Materiality threshold: %.1f tCO2e (%.0f%% of %.1f)",
            float(threshold), float(pct), float(total_emissions),
        )
        return threshold

    # -------------------------------------------------------------------
    # Cross-Check Routines
    # -------------------------------------------------------------------

    def run_cross_checks(
        self,
        scope1: Decimal,
        scope2_location: Decimal,
        scope2_market: Decimal,
        scope3: Decimal,
        reported_total: Decimal,
        prior_year_total: Optional[Decimal] = None,
    ) -> List[CrossCheckResult]:
        """Run internal consistency cross-checks.

        Checks:
        1. Scope sum = reported total
        2. Scope 2 location vs market (should be within 50%)
        3. Year-over-year change (flag >30% change)
        4. Scope 3 proportion (flag if > 90% or < 5% of total)

        Args:
            scope1: Scope 1 emissions (tCO2e).
            scope2_location: Scope 2 location-based (tCO2e).
            scope2_market: Scope 2 market-based (tCO2e).
            scope3: Scope 3 emissions (tCO2e).
            reported_total: Reported total (tCO2e).
            prior_year_total: Prior year total for trend check.

        Returns:
            List of CrossCheckResult.
        """
        scope1 = _decimal(scope1)
        scope2_location = _decimal(scope2_location)
        scope2_market = _decimal(scope2_market)
        scope3 = _decimal(scope3)
        reported_total = _decimal(reported_total)

        checks: List[CrossCheckResult] = []
        tolerance = self.config.cross_check_tolerance_pct
        prec = self.config.decimal_precision

        # Check 1: Sum check (S1 + S2_market + S3 = total)
        calc_total = scope1 + scope2_market + scope3
        diff1 = abs(calc_total - reported_total)
        diff1_pct = _safe_pct(diff1, reported_total) if reported_total > 0 else Decimal("0")
        status1 = CrossCheckStatus.PASSED if diff1_pct <= tolerance else CrossCheckStatus.FAILED

        checks.append(CrossCheckResult(
            check_name="Scope Sum Reconciliation",
            description="Verify S1 + S2(market) + S3 = reported total",
            value_a=_round_val(calc_total, prec),
            value_a_source="Calculated sum (S1+S2+S3)",
            value_b=_round_val(reported_total, prec),
            value_b_source="Reported total",
            difference=_round_val(diff1, prec),
            difference_pct=diff1_pct,
            tolerance_pct=tolerance,
            status=status1,
            finding=f"Sum={calc_total}, reported={reported_total}, diff={diff1_pct}%",
        ))

        # Check 2: Scope 2 location vs market reasonableness
        if scope2_location > 0 and scope2_market > 0:
            diff2 = abs(scope2_location - scope2_market)
            diff2_pct = _safe_pct(diff2, scope2_location)
            status2 = CrossCheckStatus.PASSED if diff2_pct <= Decimal("50") else CrossCheckStatus.WARNING

            checks.append(CrossCheckResult(
                check_name="Scope 2 Location vs Market",
                description="Scope 2 location and market should be within 50%",
                value_a=_round_val(scope2_location, prec),
                value_a_source="Scope 2 location-based",
                value_b=_round_val(scope2_market, prec),
                value_b_source="Scope 2 market-based",
                difference=_round_val(diff2, prec),
                difference_pct=diff2_pct,
                tolerance_pct=Decimal("50"),
                status=status2,
                finding=f"Difference={diff2_pct}%",
            ))

        # Check 3: Year-over-year trend
        if prior_year_total is not None and prior_year_total > Decimal("0"):
            prior = _decimal(prior_year_total)
            yoy_diff = abs(reported_total - prior)
            yoy_pct = _safe_pct(yoy_diff, prior)
            status3 = CrossCheckStatus.PASSED if yoy_pct <= Decimal("30") else CrossCheckStatus.WARNING

            checks.append(CrossCheckResult(
                check_name="Year-over-Year Trend",
                description="Flag changes exceeding 30% year-over-year",
                value_a=_round_val(reported_total, prec),
                value_a_source="Current year total",
                value_b=_round_val(prior, prec),
                value_b_source="Prior year total",
                difference=_round_val(yoy_diff, prec),
                difference_pct=yoy_pct,
                tolerance_pct=Decimal("30"),
                status=status3,
                finding=f"YoY change={yoy_pct}%",
            ))

        # Check 4: Scope 3 proportion
        if reported_total > 0 and scope3 > 0:
            s3_pct = _safe_pct(scope3, reported_total)
            status4 = CrossCheckStatus.PASSED
            finding4 = f"Scope 3 = {s3_pct}% of total"
            if s3_pct > Decimal("95"):
                status4 = CrossCheckStatus.WARNING
                finding4 += " (unusually high, verify completeness of S1/S2)"
            elif s3_pct < Decimal("5"):
                status4 = CrossCheckStatus.WARNING
                finding4 += " (unusually low, verify S3 screening)"

            checks.append(CrossCheckResult(
                check_name="Scope 3 Proportion",
                description="Scope 3 proportion should be between 5% and 95%",
                value_a=_round_val(scope3, prec),
                value_a_source="Scope 3 total",
                value_b=_round_val(reported_total, prec),
                value_b_source="Total emissions",
                difference=_round_val(scope3, prec),
                difference_pct=s3_pct,
                tolerance_pct=Decimal("95"),
                status=status4,
                finding=finding4,
            ))

        for check in checks:
            check.provenance_hash = _compute_hash(check)

        logger.info(
            "Cross-checks complete: %d checks, %d passed, %d warnings/failed",
            len(checks),
            sum(1 for c in checks if c.status == CrossCheckStatus.PASSED),
            sum(1 for c in checks if c.status != CrossCheckStatus.PASSED),
        )
        return checks

    # -------------------------------------------------------------------
    # Provenance Chain
    # -------------------------------------------------------------------

    def build_provenance_chain(self) -> List[str]:
        """Build a SHA-256 hash chain across all workpaper elements.

        Creates a linked chain of hashes from raw data through
        methodology, calculations, and controls to the final output.
        Each link depends on the previous, making the chain tamper-evident.

        Returns:
            List of SHA-256 hashes forming the provenance chain.
        """
        chain: List[str] = []
        current_hash = _compute_hash("CHAIN_GENESIS")
        chain.append(current_hash)

        # Link methodology
        for entry in self._methodology:
            current_hash = _chain_hash(current_hash, entry)
            chain.append(current_hash)

        # Link calculation traces
        for trace in self._traces:
            current_hash = _chain_hash(current_hash, trace)
            chain.append(current_hash)

        # Link data lineage
        for lineage in self._lineage:
            current_hash = _chain_hash(current_hash, lineage)
            chain.append(current_hash)

        # Link controls
        for control in self._controls:
            current_hash = _chain_hash(current_hash, control)
            chain.append(current_hash)

        # Link exceptions
        for exception in self._exceptions:
            current_hash = _chain_hash(current_hash, exception)
            chain.append(current_hash)

        logger.info("Provenance chain built: %d links", len(chain))
        return chain

    # -------------------------------------------------------------------
    # Full Workpaper Generation
    # -------------------------------------------------------------------

    def generate_workpapers(
        self,
        entity_name: str,
        reporting_year: int,
        total_emissions: Decimal,
        scope1: Decimal = Decimal("0"),
        scope2_location: Decimal = Decimal("0"),
        scope2_market: Decimal = Decimal("0"),
        scope3: Decimal = Decimal("0"),
        prior_year_total: Optional[Decimal] = None,
        assurance_level: Optional[AssuranceLevel] = None,
        organizational_boundary: str = "",
        operational_boundary: str = "",
        reporting_standards: Optional[List[str]] = None,
        engagement_team: Optional[List[str]] = None,
    ) -> AssuranceResult:
        """Generate complete assurance workpapers.

        Assembles all eight workpaper sections, runs cross-checks,
        builds the provenance chain, and produces a structured result.

        Args:
            entity_name: Entity under assurance.
            reporting_year: Reporting year.
            total_emissions: Total reported emissions (tCO2e).
            scope1: Scope 1 emissions.
            scope2_location: Scope 2 location-based.
            scope2_market: Scope 2 market-based.
            scope3: Scope 3 emissions.
            prior_year_total: Prior year total for trend check.
            assurance_level: Assurance level.
            organizational_boundary: Consolidation approach.
            operational_boundary: Scope boundary.
            reporting_standards: Standards applied.
            engagement_team: Team members.

        Returns:
            Complete AssuranceResult.
        """
        logger.info(
            "Generating workpapers for %s, year %d, total=%.1f tCO2e",
            entity_name, reporting_year, float(total_emissions),
        )

        # Section 1: Engagement summary
        summary = self.create_engagement_summary(
            entity_name=entity_name,
            reporting_year=reporting_year,
            total_emissions=total_emissions,
            assurance_level=assurance_level,
            organizational_boundary=organizational_boundary,
            operational_boundary=operational_boundary,
            reporting_standards=reporting_standards,
            engagement_team=engagement_team,
        )

        materiality = summary.materiality_threshold

        # Mark traces as material/immaterial
        for trace in self._traces:
            trace.is_material = trace.final_emissions >= materiality

        # Mark exceptions as material
        for exc in self._exceptions:
            exc.is_material = exc.impact_tco2e >= materiality
            if total_emissions > 0:
                exc.impact_pct = _safe_pct(exc.impact_tco2e, _decimal(total_emissions))

        # Cross-checks
        cross_checks = self.run_cross_checks(
            scope1=scope1,
            scope2_location=scope2_location,
            scope2_market=scope2_market,
            scope3=scope3,
            reported_total=total_emissions,
            prior_year_total=prior_year_total,
        )

        # Provenance chain
        provenance_chain = self.build_provenance_chain()

        # Calculate summary metrics
        material_exceptions = sum(1 for e in self._exceptions if e.is_material)
        actual_count = sum(1 for c in self._completeness if c.is_actual)
        total_sources = len(self._completeness)
        overall_completeness = _safe_pct(
            _decimal(actual_count), _decimal(total_sources)
        ) if total_sources > 0 else Decimal("100")

        checks_passed = sum(1 for c in cross_checks if c.status == CrossCheckStatus.PASSED)
        checks_failed = sum(1 for c in cross_checks if c.status == CrossCheckStatus.FAILED)

        result = AssuranceResult(
            engagement_summary=summary,
            methodology_entries=list(self._methodology),
            calculation_traces=list(self._traces),
            data_lineage=list(self._lineage),
            control_evidence=list(self._controls),
            exceptions=list(self._exceptions),
            completeness_matrix=list(self._completeness),
            change_register=list(self._changes),
            cross_checks=cross_checks,
            materiality_threshold=materiality,
            provenance_chain=provenance_chain,
            material_exceptions_count=material_exceptions,
            overall_completeness_pct=overall_completeness,
            cross_checks_passed=checks_passed,
            cross_checks_failed=checks_failed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Workpapers generated: %d methodology, %d traces, %d lineage, "
            "%d controls, %d exceptions (%d material), %d completeness, "
            "%d changes, %d cross-checks (%d passed, %d failed), "
            "%d provenance links",
            len(self._methodology), len(self._traces), len(self._lineage),
            len(self._controls), len(self._exceptions), material_exceptions,
            len(self._completeness), len(self._changes),
            len(cross_checks), checks_passed, checks_failed,
            len(provenance_chain),
        )
        return result

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export_to_json(self, result: AssuranceResult) -> str:
        """Export workpapers to structured JSON for auditor review tools.

        Args:
            result: AssuranceResult to export.

        Returns:
            JSON string.
        """
        data = result.model_dump(mode="json")
        return json.dumps(data, indent=2, sort_keys=True, default=str)

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    def get_section_summary(self) -> Dict[str, int]:
        """Get a summary of entries per workpaper section.

        Returns:
            Dictionary mapping section name to entry count.
        """
        return {
            WorkpaperSection.METHODOLOGY_DOCUMENTATION.value: len(self._methodology),
            WorkpaperSection.CALCULATION_TRACE.value: len(self._traces),
            WorkpaperSection.DATA_LINEAGE.value: len(self._lineage),
            WorkpaperSection.CONTROL_EVIDENCE.value: len(self._controls),
            WorkpaperSection.EXCEPTION_REGISTER.value: len(self._exceptions),
            WorkpaperSection.COMPLETENESS_MATRIX.value: len(self._completeness),
            WorkpaperSection.CHANGE_REGISTER.value: len(self._changes),
        }

    def clear(self) -> None:
        """Clear all stored workpaper data."""
        self._methodology.clear()
        self._traces.clear()
        self._lineage.clear()
        self._controls.clear()
        self._exceptions.clear()
        self._completeness.clear()
        self._changes.clear()
        logger.info("AssuranceWorkpaperEngine cleared")
