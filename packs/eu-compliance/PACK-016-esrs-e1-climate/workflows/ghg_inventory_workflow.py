# -*- coding: utf-8 -*-
"""
GHG Inventory Workflow
==============================

5-phase workflow for end-to-end GHG inventory per ESRS E1-6 and GHG Protocol.
Implements data collection, emission calculation, scope aggregation, quality
checks, and report generation with full provenance tracking.

Phases:
    1. DataCollection         -- Gather activity data from sources
    2. EmissionCalculation    -- Apply emission factors and calculate tCO2e
    3. ScopeAggregation       -- Aggregate by Scope 1/2/3 and categories
    4. QualityCheck           -- Validate completeness and consistency
    5. ReportGeneration       -- Produce E1-6 disclosure data

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the GHG inventory workflow."""
    DATA_COLLECTION = "data_collection"
    EMISSION_CALCULATION = "emission_calculation"
    SCOPE_AGGREGATION = "scope_aggregation"
    QUALITY_CHECK = "quality_check"
    REPORT_GENERATION = "report_generation"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class GHGScope(str, Enum):
    """GHG Protocol emission scope."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"

class GasType(str, Enum):
    """Greenhouse gas types per GHG Protocol."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCs"
    PFCS = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"
    CO2E = "CO2e"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories."""
    CAT_1 = "purchased_goods_services"
    CAT_2 = "capital_goods"
    CAT_3 = "fuel_energy_activities"
    CAT_4 = "upstream_transportation"
    CAT_5 = "waste_generated"
    CAT_6 = "business_travel"
    CAT_7 = "employee_commuting"
    CAT_8 = "upstream_leased_assets"
    CAT_9 = "downstream_transportation"
    CAT_10 = "processing_sold_products"
    CAT_11 = "use_of_sold_products"
    CAT_12 = "end_of_life_treatment"
    CAT_13 = "downstream_leased_assets"
    CAT_14 = "franchises"
    CAT_15 = "investments"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class EmissionRecord(BaseModel):
    """Single emission record with activity data and calculated emissions."""
    record_id: str = Field(default_factory=lambda: f"em-{_new_uuid()[:8]}")
    source_name: str = Field(..., description="Emission source name")
    scope: GHGScope = Field(..., description="GHG Protocol scope")
    scope_3_category: Optional[Scope3Category] = Field(
        default=None, description="Scope 3 category if applicable"
    )
    activity_data: float = Field(default=0.0, ge=0.0, description="Activity quantity")
    activity_unit: str = Field(default="", description="Unit of activity data")
    emission_factor: float = Field(default=0.0, ge=0.0, description="EF in tCO2e per unit")
    emission_factor_source: str = Field(default="", description="EF database source")
    gas_type: GasType = Field(default=GasType.CO2E, description="Greenhouse gas type")
    emissions_tco2e: float = Field(default=0.0, ge=0.0, description="Calculated emissions in tCO2e")
    gwp_applied: float = Field(default=1.0, description="Global warming potential factor")
    methodology: str = Field(default="calculation", description="Calculation methodology")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    reporting_year: int = Field(default=2025)

class ScopeAggregation(BaseModel):
    """Aggregated emissions by scope."""
    scope: str = Field(..., description="Scope identifier")
    total_tco2e: float = Field(default=0.0, ge=0.0)
    source_count: int = Field(default=0, ge=0)
    category_breakdown: Dict[str, float] = Field(default_factory=dict)
    gas_breakdown: Dict[str, float] = Field(default_factory=dict)
    avg_data_quality: float = Field(default=0.0)

class GHGInventoryInput(BaseModel):
    """Input data model for GHGInventoryWorkflow."""
    emission_records: List[EmissionRecord] = Field(
        default_factory=list, description="Pre-collected emission records"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    base_year: int = Field(default=2019, ge=1990, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    consolidation_approach: str = Field(
        default="operational_control",
        description="financial_control, operational_control, or equity_share"
    )
    include_biogenic: bool = Field(default=True, description="Include biogenic CO2")
    quality_threshold: float = Field(
        default=2.0, ge=0.0, le=5.0, description="Minimum data quality score"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class GHGInventoryResult(BaseModel):
    """Complete result from GHG inventory workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="ghg_inventory")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope_1_tco2e: float = Field(default=0.0)
    scope_2_location_tco2e: float = Field(default=0.0)
    scope_2_market_tco2e: float = Field(default=0.0)
    scope_3_tco2e: float = Field(default=0.0)
    scope_aggregations: List[ScopeAggregation] = Field(default_factory=list)
    emission_records: List[EmissionRecord] = Field(default_factory=list)
    gas_disaggregation: Dict[str, float] = Field(default_factory=dict)
    intensity_metrics: Dict[str, float] = Field(default_factory=dict)
    quality_issues: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    base_year: int = Field(default=2019)
    consolidation_approach: str = Field(default="operational_control")
    provenance_hash: str = Field(default="")

# =============================================================================
# DEFAULT EMISSION FACTORS (tCO2e per unit)
# =============================================================================

DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas_m3": 0.00202,
    "diesel_litre": 0.002674,
    "petrol_litre": 0.002315,
    "electricity_kwh_global": 0.000494,
    "coal_kg": 0.002419,
    "lpg_litre": 0.001557,
    "heating_oil_litre": 0.002674,
    "jet_fuel_litre": 0.002530,
    "biomass_kg": 0.000000,
}

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class GHGInventoryWorkflow:
    """
    5-phase GHG inventory workflow for ESRS E1-6 disclosure.

    Implements end-to-end GHG inventory compilation per the GHG Protocol
    Corporate Standard and ESRS E1-6 disclosure requirements. Collects
    activity data, applies emission factors, aggregates by scope, runs
    quality checks, and generates disclosure-ready output.

    Zero-hallucination: all emission calculations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = GHGInventoryWorkflow()
        >>> inp = GHGInventoryInput(emission_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_emissions_tco2e >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize GHGInventoryWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._records: List[EmissionRecord] = []
        self._aggregations: List[ScopeAggregation] = []
        self._quality_issues: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DATA_COLLECTION.value, "description": "Gather activity data from sources"},
            {"name": WorkflowPhase.EMISSION_CALCULATION.value, "description": "Apply emission factors and calculate tCO2e"},
            {"name": WorkflowPhase.SCOPE_AGGREGATION.value, "description": "Aggregate by Scope 1/2/3 and categories"},
            {"name": WorkflowPhase.QUALITY_CHECK.value, "description": "Validate completeness and consistency"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-6 disclosure data"},
        ]

    def validate_inputs(self, input_data: GHGInventoryInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.emission_records:
            issues.append("No emission records provided")
        if input_data.reporting_year < input_data.base_year:
            issues.append("Reporting year cannot be before base year")
        if input_data.consolidation_approach not in (
            "financial_control", "operational_control", "equity_share"
        ):
            issues.append(f"Invalid consolidation approach: {input_data.consolidation_approach}")
        for rec in input_data.emission_records:
            if rec.activity_data < 0:
                issues.append(f"Negative activity data in record {rec.record_id}")
            if rec.emission_factor < 0:
                issues.append(f"Negative emission factor in record {rec.record_id}")
        return issues

    async def execute(
        self,
        input_data: Optional[GHGInventoryInput] = None,
        emission_records: Optional[List[EmissionRecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> GHGInventoryResult:
        """
        Execute the 5-phase GHG inventory workflow.

        Args:
            input_data: Full input model (preferred).
            emission_records: Emission records (fallback).
            config: Configuration overrides.

        Returns:
            GHGInventoryResult with scope totals, aggregations, and quality issues.
        """
        if input_data is None:
            input_data = GHGInventoryInput(
                emission_records=emission_records or [],
                config=config or {},
            )

        started_at = utcnow()
        self.logger.info("Starting GHG inventory workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_data_collection(input_data))
            phases_done += 1
            phase_results.append(await self._phase_emission_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_scope_aggregation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_quality_check(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("GHG inventory workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        # Calculate scope totals
        scope_1 = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_1)
        scope_2_loc = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_2_LOCATION)
        scope_2_mkt = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_2_MARKET)
        scope_3 = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_3)
        total = scope_1 + max(scope_2_loc, scope_2_mkt) + scope_3

        # Gas disaggregation
        gas_breakdown: Dict[str, float] = {}
        for rec in self._records:
            gas_breakdown[rec.gas_type.value] = gas_breakdown.get(rec.gas_type.value, 0.0) + rec.emissions_tco2e

        result = GHGInventoryResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            total_emissions_tco2e=round(total, 4),
            scope_1_tco2e=round(scope_1, 4),
            scope_2_location_tco2e=round(scope_2_loc, 4),
            scope_2_market_tco2e=round(scope_2_mkt, 4),
            scope_3_tco2e=round(scope_3, 4),
            scope_aggregations=self._aggregations,
            emission_records=self._records,
            gas_disaggregation=gas_breakdown,
            quality_issues=self._quality_issues,
            reporting_year=input_data.reporting_year,
            base_year=input_data.base_year,
            consolidation_approach=input_data.consolidation_approach,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "GHG inventory %s completed in %.2fs: %.2f tCO2e total",
            self.workflow_id, elapsed, total,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: GHGInventoryInput,
    ) -> PhaseResult:
        """Gather emission records from input sources."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._records = list(input_data.emission_records)

        # Count by scope
        scope_counts: Dict[str, int] = {}
        for rec in self._records:
            scope_counts[rec.scope.value] = scope_counts.get(rec.scope.value, 0) + 1

        outputs["records_collected"] = len(self._records)
        outputs["scope_distribution"] = scope_counts
        outputs["unique_sources"] = len(set(r.source_name for r in self._records))
        outputs["consolidation_approach"] = input_data.consolidation_approach

        if not self._records:
            warnings.append("No emission records provided; inventory will be empty")

        # Check for missing scope coverage
        present_scopes = set(r.scope.value for r in self._records)
        expected_scopes = {"scope_1", "scope_2_location", "scope_3"}
        missing = expected_scopes - present_scopes
        if missing:
            warnings.append(f"Missing scope coverage: {', '.join(sorted(missing))}")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataCollection: %d records from %d sources",
            len(self._records), outputs["unique_sources"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DATA_COLLECTION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Emission Calculation
    # -------------------------------------------------------------------------

    async def _phase_emission_calculation(
        self, input_data: GHGInventoryInput,
    ) -> PhaseResult:
        """Apply emission factors and calculate tCO2e for each record."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        calculated_count = 0

        for rec in self._records:
            if rec.emissions_tco2e > 0:
                # Already calculated
                calculated_count += 1
                continue
            if rec.emission_factor > 0 and rec.activity_data > 0:
                rec.emissions_tco2e = round(
                    rec.activity_data * rec.emission_factor * rec.gwp_applied, 6
                )
                calculated_count += 1
            else:
                # Try default emission factor lookup
                ef = self._lookup_default_ef(rec)
                if ef > 0:
                    rec.emission_factor = ef
                    rec.emission_factor_source = "pack_016_defaults"
                    rec.emissions_tco2e = round(
                        rec.activity_data * ef * rec.gwp_applied, 6
                    )
                    calculated_count += 1
                    warnings.append(
                        f"Record {rec.record_id}: used default EF for {rec.source_name}"
                    )
                else:
                    warnings.append(
                        f"Record {rec.record_id}: no emission factor available, emissions=0"
                    )

        total_emissions = sum(r.emissions_tco2e for r in self._records)
        outputs["records_calculated"] = calculated_count
        outputs["records_missing_ef"] = len(self._records) - calculated_count
        outputs["total_emissions_tco2e"] = round(total_emissions, 4)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 EmissionCalculation: %d calculated, %.2f tCO2e total",
            calculated_count, total_emissions,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.EMISSION_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _lookup_default_ef(self, record: EmissionRecord) -> float:
        """Look up default emission factor by source name and unit."""
        key = f"{record.source_name.lower().replace(' ', '_')}_{record.activity_unit.lower()}"
        if key in DEFAULT_EMISSION_FACTORS:
            return DEFAULT_EMISSION_FACTORS[key]
        # Try partial match
        for ef_key, ef_val in DEFAULT_EMISSION_FACTORS.items():
            if record.source_name.lower().replace(" ", "_") in ef_key:
                return ef_val
        return 0.0

    # -------------------------------------------------------------------------
    # Phase 3: Scope Aggregation
    # -------------------------------------------------------------------------

    async def _phase_scope_aggregation(
        self, input_data: GHGInventoryInput,
    ) -> PhaseResult:
        """Aggregate emissions by scope, category, and gas type."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._aggregations = []

        # Group by scope
        scope_groups: Dict[str, List[EmissionRecord]] = {}
        for rec in self._records:
            scope_groups.setdefault(rec.scope.value, []).append(rec)

        for scope_val, records in sorted(scope_groups.items()):
            total = sum(r.emissions_tco2e for r in records)
            cat_breakdown: Dict[str, float] = {}
            gas_bk: Dict[str, float] = {}
            quality_scores: List[float] = []

            for r in records:
                # Category breakdown (scope 3 uses category, others use source)
                cat_key = r.scope_3_category.value if r.scope_3_category else r.source_name
                cat_breakdown[cat_key] = cat_breakdown.get(cat_key, 0.0) + r.emissions_tco2e
                gas_bk[r.gas_type.value] = gas_bk.get(r.gas_type.value, 0.0) + r.emissions_tco2e
                if r.data_quality_score > 0:
                    quality_scores.append(r.data_quality_score)

            avg_quality = (
                round(sum(quality_scores) / len(quality_scores), 2)
                if quality_scores else 0.0
            )

            self._aggregations.append(ScopeAggregation(
                scope=scope_val,
                total_tco2e=round(total, 4),
                source_count=len(records),
                category_breakdown={k: round(v, 4) for k, v in cat_breakdown.items()},
                gas_breakdown={k: round(v, 4) for k, v in gas_bk.items()},
                avg_data_quality=avg_quality,
            ))

        outputs["scopes_aggregated"] = len(self._aggregations)
        outputs["scope_totals"] = {
            agg.scope: agg.total_tco2e for agg in self._aggregations
        }

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ScopeAggregation: %d scopes aggregated",
            len(self._aggregations),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SCOPE_AGGREGATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quality Check
    # -------------------------------------------------------------------------

    async def _phase_quality_check(
        self, input_data: GHGInventoryInput,
    ) -> PhaseResult:
        """Validate completeness and data quality of the inventory."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._quality_issues = []

        # Check 1: Scope coverage
        present_scopes = set(r.scope.value for r in self._records)
        if "scope_1" not in present_scopes:
            self._quality_issues.append("CRITICAL: No Scope 1 emissions reported")
        if "scope_2_location" not in present_scopes and "scope_2_market" not in present_scopes:
            self._quality_issues.append("CRITICAL: No Scope 2 emissions reported")

        # Check 2: Data quality scores
        low_quality = [
            r for r in self._records
            if 0 < r.data_quality_score < input_data.quality_threshold
        ]
        if low_quality:
            self._quality_issues.append(
                f"WARNING: {len(low_quality)} records below quality threshold "
                f"({input_data.quality_threshold})"
            )

        # Check 3: Zero-emission records
        zero_records = [r for r in self._records if r.emissions_tco2e == 0.0]
        if zero_records:
            self._quality_issues.append(
                f"INFO: {len(zero_records)} records with zero emissions"
            )

        # Check 4: Biogenic handling
        if not input_data.include_biogenic:
            biogenic = [r for r in self._records if "biomass" in r.source_name.lower()]
            if biogenic:
                warnings.append(f"{len(biogenic)} biogenic records excluded per configuration")

        # Check 5: Duplicate source detection
        source_scope_pairs = [
            (r.source_name, r.scope.value) for r in self._records
        ]
        seen: set = set()
        duplicates = 0
        for pair in source_scope_pairs:
            if pair in seen:
                duplicates += 1
            seen.add(pair)
        if duplicates > 0:
            self._quality_issues.append(
                f"WARNING: {duplicates} potential duplicate source-scope combinations"
            )

        outputs["quality_issues_found"] = len(self._quality_issues)
        outputs["quality_issues"] = self._quality_issues
        outputs["low_quality_records"] = len(low_quality) if low_quality else 0
        outputs["zero_emission_records"] = len(zero_records) if zero_records else 0
        outputs["quality_pass"] = len(self._quality_issues) == 0 or all(
            not issue.startswith("CRITICAL") for issue in self._quality_issues
        )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 QualityCheck: %d issues found",
            len(self._quality_issues),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.QUALITY_CHECK.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: GHGInventoryInput,
    ) -> PhaseResult:
        """Generate E1-6 disclosure-ready output."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        scope_1 = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_1)
        scope_2_loc = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_2_LOCATION)
        scope_2_mkt = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_2_MARKET)
        scope_3 = sum(r.emissions_tco2e for r in self._records if r.scope == GHGScope.SCOPE_3)

        outputs["e1_6_disclosure"] = {
            "gross_scope_1_tco2e": round(scope_1, 2),
            "gross_scope_2_location_tco2e": round(scope_2_loc, 2),
            "gross_scope_2_market_tco2e": round(scope_2_mkt, 2),
            "total_scope_3_tco2e": round(scope_3, 2),
            "total_ghg_emissions_tco2e": round(scope_1 + max(scope_2_loc, scope_2_mkt) + scope_3, 2),
            "consolidation_approach": input_data.consolidation_approach,
            "reporting_year": input_data.reporting_year,
            "base_year": input_data.base_year,
            "scope_3_categories_reported": len(set(
                r.scope_3_category.value for r in self._records
                if r.scope_3_category is not None
            )),
        }

        # Gas disaggregation for E1-6
        gas_totals: Dict[str, float] = {}
        for rec in self._records:
            gas_totals[rec.gas_type.value] = gas_totals.get(rec.gas_type.value, 0.0) + rec.emissions_tco2e
        outputs["gas_disaggregation"] = {k: round(v, 2) for k, v in gas_totals.items()}

        outputs["report_ready"] = True
        outputs["record_count"] = len(self._records)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReportGeneration: E1-6 disclosure ready, %d records",
            len(self._records),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: GHGInventoryResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
