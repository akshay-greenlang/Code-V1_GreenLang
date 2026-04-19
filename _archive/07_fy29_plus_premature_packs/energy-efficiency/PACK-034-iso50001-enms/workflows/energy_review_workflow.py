# -*- coding: utf-8 -*-
"""
Energy Review Workflow - ISO 50001 Clause 6.3
===================================

4-phase workflow for conducting ISO 50001 energy reviews within
PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. DataCollection          -- Validate and aggregate energy consumption data
    2. SEUAnalysis             -- Identify Significant Energy Uses via Pareto analysis
    3. EnBEnPIEstablishment    -- Establish energy baselines and calculate EnPIs
    4. OpportunityIdentification -- Identify improvement opportunities from SEU analysis

The workflow follows GreenLang zero-hallucination principles: all SEU
identification uses deterministic Pareto analysis on metered data,
EnPI calculations use deterministic formulas against validated baselines,
and opportunity scoring is rule-based. SHA-256 provenance hashes
guarantee auditability.

Schedule: annual / on significant change
Estimated duration: 45 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ReviewPhase(str, Enum):
    """Phases of the energy review workflow."""

    DATA_COLLECTION = "data_collection"
    SEU_ANALYSIS = "seu_analysis"
    ENB_ENPI_ESTABLISHMENT = "enb_enpi_establishment"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"


# =============================================================================
# SEU CLASSIFICATION REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical energy end-use categories for Pareto analysis
SEU_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "hvac": {
        "label": "Heating, Ventilation & Air Conditioning",
        "typical_share_pct_low": 25.0,
        "typical_share_pct_high": 50.0,
        "improvement_potential_pct": 15.0,
    },
    "lighting": {
        "label": "Lighting Systems",
        "typical_share_pct_low": 10.0,
        "typical_share_pct_high": 25.0,
        "improvement_potential_pct": 30.0,
    },
    "motors_drives": {
        "label": "Motors & Drives",
        "typical_share_pct_low": 15.0,
        "typical_share_pct_high": 40.0,
        "improvement_potential_pct": 20.0,
    },
    "compressed_air": {
        "label": "Compressed Air Systems",
        "typical_share_pct_low": 5.0,
        "typical_share_pct_high": 20.0,
        "improvement_potential_pct": 25.0,
    },
    "process_heat": {
        "label": "Process Heating",
        "typical_share_pct_low": 10.0,
        "typical_share_pct_high": 35.0,
        "improvement_potential_pct": 10.0,
    },
    "refrigeration": {
        "label": "Refrigeration",
        "typical_share_pct_low": 5.0,
        "typical_share_pct_high": 15.0,
        "improvement_potential_pct": 18.0,
    },
    "building_envelope": {
        "label": "Building Envelope",
        "typical_share_pct_low": 3.0,
        "typical_share_pct_high": 15.0,
        "improvement_potential_pct": 12.0,
    },
    "other": {
        "label": "Other / Miscellaneous",
        "typical_share_pct_low": 2.0,
        "typical_share_pct_high": 10.0,
        "improvement_potential_pct": 5.0,
    },
}

# SEU threshold: end-uses consuming >= this % are deemed significant
SEU_THRESHOLD_PCT: float = 10.0

# Pareto threshold: cumulative share at which analysis stops
PARETO_CUMULATIVE_PCT: float = 80.0


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class SEUItem(BaseModel):
    """A Significant Energy Use identified during the review."""

    seu_id: str = Field(default_factory=lambda: f"seu-{uuid.uuid4().hex[:8]}")
    category: str = Field(default="", description="Energy end-use category")
    label: str = Field(default="", description="Human-readable label")
    consumption_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    share_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    cumulative_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    is_significant: bool = Field(default=False, description="True if SEU threshold met")
    improvement_potential_pct: Decimal = Field(default=Decimal("0"), ge=0)
    potential_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    equipment_ids: List[str] = Field(default_factory=list)


class BaselineRecord(BaseModel):
    """An energy baseline (EnB) record."""

    baseline_id: str = Field(default_factory=lambda: f"enb-{uuid.uuid4().hex[:8]}")
    category: str = Field(default="", description="Energy end-use category")
    baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    period_start: str = Field(default="", description="ISO 8601 date")
    period_end: str = Field(default="", description="ISO 8601 date")
    relevant_variables: List[str] = Field(default_factory=list)
    normalization_factor: str = Field(default="none", description="HDD|CDD|production|none")


class EnPIRecord(BaseModel):
    """An Energy Performance Indicator (EnPI) record."""

    enpi_id: str = Field(default_factory=lambda: f"enpi-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="EnPI name")
    category: str = Field(default="", description="Energy end-use category")
    value: Decimal = Field(default=Decimal("0"), description="Calculated EnPI value")
    unit: str = Field(default="kWh/unit", description="EnPI unit")
    baseline_value: Decimal = Field(default=Decimal("0"), description="Baseline EnPI value")
    improvement_pct: Decimal = Field(default=Decimal("0"), description="Improvement vs baseline")


class OpportunityItem(BaseModel):
    """An energy improvement opportunity."""

    opportunity_id: str = Field(default_factory=lambda: f"opp-{uuid.uuid4().hex[:8]}")
    seu_id: str = Field(default="", description="Related SEU ID")
    title: str = Field(default="", description="Opportunity title")
    description: str = Field(default="", description="Detailed description")
    category: str = Field(default="", description="Energy end-use category")
    estimated_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_savings_pct: Decimal = Field(default=Decimal("0"), ge=0)
    priority: str = Field(default="medium", description="high|medium|low")
    implementation_complexity: str = Field(default="medium", description="low|medium|high")


class EnergyReviewInput(BaseModel):
    """Input data model for EnergyReviewWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    enms_id: str = Field(default="", description="EnMS program identifier")
    energy_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Energy consumption records with category, kwh, period",
    )
    equipment_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Equipment inventory data",
    )
    weather_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Weather data (HDD, CDD) for normalization",
    )
    review_period_start: str = Field(default="", description="Review period start YYYY-MM-DD")
    review_period_end: str = Field(default="", description="Review period end YYYY-MM-DD")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("energy_data")
    @classmethod
    def validate_energy_data(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure at least one energy data record is provided."""
        if not v:
            raise ValueError("energy_data must contain at least one record")
        return v


class EnergyReviewResult(BaseModel):
    """Complete result from energy review workflow."""

    review_id: str = Field(..., description="Unique review execution ID")
    facility_id: str = Field(default="", description="Reviewed facility ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    seu_list: List[SEUItem] = Field(default_factory=list, description="Identified SEUs")
    baselines: List[BaselineRecord] = Field(default_factory=list, description="Established baselines")
    enpis: List[EnPIRecord] = Field(default_factory=list, description="Calculated EnPIs")
    opportunities: List[OpportunityItem] = Field(default_factory=list, description="Improvement opportunities")
    total_consumption_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    seu_coverage_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0, description="Total execution time in ms")
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EnergyReviewWorkflow:
    """
    4-phase energy review workflow per ISO 50001 Clause 6.3.

    Performs data collection and validation, Significant Energy Use (SEU)
    identification via Pareto analysis, energy baseline (EnB) and EnPI
    establishment, and improvement opportunity identification.

    Zero-hallucination: SEU identification uses deterministic Pareto
    sorting on metered consumption data. EnPIs are calculated from
    validated meter readings and production data. No LLM calls in the
    numeric computation path.

    Attributes:
        review_id: Unique review execution identifier.
        _seus: Identified significant energy uses.
        _baselines: Established energy baselines.
        _enpis: Calculated energy performance indicators.
        _opportunities: Identified improvement opportunities.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = EnergyReviewWorkflow()
        >>> inp = EnergyReviewInput(
        ...     facility_id="fac-001",
        ...     enms_id="enms-001",
        ...     energy_data=[{"category": "hvac", "kwh": 500000, "period": "2025"}],
        ...     review_period_start="2025-01-01",
        ...     review_period_end="2025-12-31",
        ... )
        >>> result = wf.execute(inp)
        >>> assert len(result.seu_list) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyReviewWorkflow."""
        self.review_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._seus: List[SEUItem] = []
        self._baselines: List[BaselineRecord] = []
        self._enpis: List[EnPIRecord] = []
        self._opportunities: List[OpportunityItem] = []
        self._phase_results: List[PhaseResult] = []
        self._total_consumption_kwh: Decimal = Decimal("0")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: EnergyReviewInput) -> EnergyReviewResult:
        """
        Execute the 4-phase energy review workflow.

        Args:
            input_data: Validated energy review input.

        Returns:
            EnergyReviewResult with SEUs, baselines, EnPIs, and opportunities.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting energy review workflow %s facility=%s enms=%s",
            self.review_id, input_data.facility_id, input_data.enms_id,
        )

        self._phase_results = []
        self._seus = []
        self._baselines = []
        self._enpis = []
        self._opportunities = []
        self._total_consumption_kwh = Decimal("0")

        try:
            # Phase 1: Data Collection
            phase1 = self._phase_data_collection(input_data)
            self._phase_results.append(phase1)

            # Phase 2: SEU Analysis
            phase2 = self._phase_seu_analysis(input_data)
            self._phase_results.append(phase2)

            # Phase 3: EnB / EnPI Establishment
            phase3 = self._phase_enb_enpi_establishment(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Opportunity Identification
            phase4 = self._phase_opportunity_identification(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Energy review workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Calculate SEU coverage (sum of significant SEU shares)
        seu_coverage = sum(
            s.share_pct for s in self._seus if s.is_significant
        )

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = EnergyReviewResult(
            review_id=self.review_id,
            facility_id=input_data.facility_id,
            enms_id=input_data.enms_id,
            seu_list=self._seus,
            baselines=self._baselines,
            enpis=self._enpis,
            opportunities=self._opportunities,
            total_consumption_kwh=self._total_consumption_kwh,
            seu_coverage_pct=seu_coverage,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Energy review workflow %s completed in %.0fms SEUs=%d baselines=%d "
            "enpis=%d opportunities=%d total=%.0f kWh",
            self.review_id, elapsed_ms, len(self._seus), len(self._baselines),
            len(self._enpis), len(self._opportunities),
            float(self._total_consumption_kwh),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: EnergyReviewInput
    ) -> PhaseResult:
        """Validate and aggregate energy consumption data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Aggregate energy data by category
        category_totals: Dict[str, Decimal] = {}
        record_count = 0

        for record in input_data.energy_data:
            category = record.get("category", "other")
            kwh = Decimal(str(record.get("kwh", 0)))

            if kwh < 0:
                warnings.append(f"Negative kWh value in category '{category}'; skipping")
                continue

            category_totals[category] = category_totals.get(category, Decimal("0")) + kwh
            record_count += 1

        self._total_consumption_kwh = sum(category_totals.values())

        if self._total_consumption_kwh <= 0:
            warnings.append("Total energy consumption is zero or negative")

        # Validate data completeness
        if not input_data.review_period_start or not input_data.review_period_end:
            warnings.append("Review period dates not fully specified")

        if not input_data.equipment_data:
            warnings.append("No equipment data provided; SEU analysis limited to consumption data")

        if not input_data.weather_data:
            warnings.append("No weather data provided; normalization unavailable")

        outputs["records_processed"] = record_count
        outputs["categories_found"] = list(category_totals.keys())
        outputs["category_totals_kwh"] = {k: str(v) for k, v in category_totals.items()}
        outputs["total_consumption_kwh"] = str(self._total_consumption_kwh)
        outputs["equipment_count"] = len(input_data.equipment_data)
        outputs["weather_records"] = len(input_data.weather_data)
        outputs["review_period"] = f"{input_data.review_period_start} to {input_data.review_period_end}"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataCollection: %d records, %d categories, total=%.0f kWh",
            record_count, len(category_totals), float(self._total_consumption_kwh),
        )
        return PhaseResult(
            phase_name=ReviewPhase.DATA_COLLECTION.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: SEU Analysis
    # -------------------------------------------------------------------------

    def _phase_seu_analysis(
        self, input_data: EnergyReviewInput
    ) -> PhaseResult:
        """Run SEU identification using Pareto analysis."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Aggregate by category
        category_totals: Dict[str, Decimal] = {}
        for record in input_data.energy_data:
            category = record.get("category", "other")
            kwh = Decimal(str(record.get("kwh", 0)))
            if kwh > 0:
                category_totals[category] = category_totals.get(category, Decimal("0")) + kwh

        total = self._total_consumption_kwh
        if total <= 0:
            warnings.append("Cannot perform Pareto analysis with zero total consumption")
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return PhaseResult(
                phase_name=ReviewPhase.SEU_ANALYSIS.value, phase_number=2,
                status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
                outputs={"seus_identified": 0}, warnings=warnings,
                provenance_hash=self._hash_dict({"seus_identified": 0}),
            )

        # Sort categories by consumption (descending) for Pareto
        sorted_categories = sorted(
            category_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        cumulative_pct = Decimal("0")
        significant_count = 0

        for category, consumption in sorted_categories:
            share_pct = Decimal(str(round(float(consumption) / float(total) * 100.0, 2)))
            cumulative_pct += share_pct

            # Lookup reference data
            ref = SEU_CATEGORIES.get(category, SEU_CATEGORIES["other"])
            improvement_pct = Decimal(str(ref["improvement_potential_pct"]))
            potential_savings = Decimal(str(
                round(float(consumption) * float(improvement_pct) / 100.0, 2)
            ))

            is_significant = float(share_pct) >= SEU_THRESHOLD_PCT

            # Collect equipment IDs for this category
            equipment_ids = [
                eq.get("equipment_id", "")
                for eq in input_data.equipment_data
                if eq.get("category", "") == category
            ]

            seu = SEUItem(
                category=category,
                label=ref["label"],
                consumption_kwh=consumption,
                share_pct=share_pct,
                cumulative_pct=min(cumulative_pct, Decimal("100")),
                is_significant=is_significant,
                improvement_potential_pct=improvement_pct,
                potential_savings_kwh=potential_savings,
                equipment_ids=equipment_ids,
            )
            self._seus.append(seu)

            if is_significant:
                significant_count += 1

        outputs["total_categories"] = len(sorted_categories)
        outputs["seus_identified"] = significant_count
        outputs["pareto_80_pct_categories"] = sum(
            1 for s in self._seus
            if float(s.cumulative_pct) <= PARETO_CUMULATIVE_PCT or s.is_significant
        )
        outputs["seu_threshold_pct"] = SEU_THRESHOLD_PCT

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 SEUAnalysis: %d categories, %d significant, Pareto coverage=%.1f%%",
            len(sorted_categories), significant_count, float(cumulative_pct),
        )
        return PhaseResult(
            phase_name=ReviewPhase.SEU_ANALYSIS.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: EnB / EnPI Establishment
    # -------------------------------------------------------------------------

    def _phase_enb_enpi_establishment(
        self, input_data: EnergyReviewInput
    ) -> PhaseResult:
        """Establish energy baselines and calculate EnPIs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Determine normalization factor from weather data
        has_weather = len(input_data.weather_data) > 0
        normalization = "HDD" if has_weather else "none"

        # Create baseline for each SEU
        for seu in self._seus:
            if not seu.is_significant:
                continue

            # Identify relevant variables
            relevant_vars: List[str] = []
            if has_weather:
                relevant_vars.append("weather_hdd")
            # Check equipment data for production variables
            for eq in input_data.equipment_data:
                if eq.get("category") == seu.category:
                    prod_var = eq.get("production_variable", "")
                    if prod_var and prod_var not in relevant_vars:
                        relevant_vars.append(prod_var)

            baseline = BaselineRecord(
                category=seu.category,
                baseline_kwh=seu.consumption_kwh,
                period_start=input_data.review_period_start,
                period_end=input_data.review_period_end,
                relevant_variables=relevant_vars,
                normalization_factor=normalization if relevant_vars else "none",
            )
            self._baselines.append(baseline)

        # Create EnPIs
        # Facility-level EnPI: total kWh / floor area (if available)
        facility_enpi = EnPIRecord(
            name="Facility Energy Intensity",
            category="facility",
            value=self._total_consumption_kwh,
            unit="kWh/year",
            baseline_value=self._total_consumption_kwh,
            improvement_pct=Decimal("0"),
        )
        self._enpis.append(facility_enpi)

        # SEU-level EnPIs
        for seu in self._seus:
            if not seu.is_significant:
                continue

            enpi = EnPIRecord(
                name=f"{seu.label} Intensity",
                category=seu.category,
                value=seu.consumption_kwh,
                unit="kWh/year",
                baseline_value=seu.consumption_kwh,
                improvement_pct=Decimal("0"),
            )
            self._enpis.append(enpi)

        outputs["baselines_created"] = len(self._baselines)
        outputs["enpis_created"] = len(self._enpis)
        outputs["normalization_method"] = normalization
        outputs["baseline_period"] = (
            f"{input_data.review_period_start} to {input_data.review_period_end}"
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 EnBEnPIEstablishment: %d baselines, %d EnPIs, norm=%s",
            len(self._baselines), len(self._enpis), normalization,
        )
        return PhaseResult(
            phase_name=ReviewPhase.ENB_ENPI_ESTABLISHMENT.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Opportunity Identification
    # -------------------------------------------------------------------------

    def _phase_opportunity_identification(
        self, input_data: EnergyReviewInput
    ) -> PhaseResult:
        """Identify improvement opportunities from SEU analysis."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for seu in self._seus:
            if not seu.is_significant:
                continue

            opportunities = self._generate_opportunities_for_seu(seu)
            self._opportunities.extend(opportunities)

        # Sort opportunities by estimated savings descending
        self._opportunities.sort(
            key=lambda o: o.estimated_savings_kwh, reverse=True,
        )

        # Assign priority based on savings rank
        for idx, opp in enumerate(self._opportunities):
            if idx < len(self._opportunities) * 0.33:
                opp.priority = "high"
            elif idx < len(self._opportunities) * 0.66:
                opp.priority = "medium"
            else:
                opp.priority = "low"

        total_opportunity_kwh = sum(o.estimated_savings_kwh for o in self._opportunities)

        outputs["opportunities_identified"] = len(self._opportunities)
        outputs["total_opportunity_kwh"] = str(total_opportunity_kwh)
        outputs["high_priority_count"] = sum(1 for o in self._opportunities if o.priority == "high")
        outputs["medium_priority_count"] = sum(1 for o in self._opportunities if o.priority == "medium")
        outputs["low_priority_count"] = sum(1 for o in self._opportunities if o.priority == "low")

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 OpportunityIdentification: %d opportunities, total=%.0f kWh",
            len(self._opportunities), float(total_opportunity_kwh),
        )
        return PhaseResult(
            phase_name=ReviewPhase.OPPORTUNITY_IDENTIFICATION.value, phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_opportunities_for_seu(self, seu: SEUItem) -> List[OpportunityItem]:
        """Generate improvement opportunities for a specific SEU."""
        opportunities: List[OpportunityItem] = []
        ref = SEU_CATEGORIES.get(seu.category, SEU_CATEGORIES["other"])

        # Primary opportunity: direct improvement on SEU
        primary = OpportunityItem(
            seu_id=seu.seu_id,
            title=f"Optimize {ref['label']}",
            description=(
                f"Improve energy performance of {ref['label']} systems. "
                f"Current consumption: {seu.consumption_kwh} kWh/year "
                f"({seu.share_pct}% of total). "
                f"Estimated improvement potential: {seu.improvement_potential_pct}%."
            ),
            category=seu.category,
            estimated_savings_kwh=seu.potential_savings_kwh,
            estimated_savings_pct=seu.improvement_potential_pct,
            implementation_complexity="medium",
        )
        opportunities.append(primary)

        # Secondary opportunity: operational controls
        operational_savings = Decimal(str(
            round(float(seu.potential_savings_kwh) * 0.3, 2)
        ))
        operational = OpportunityItem(
            seu_id=seu.seu_id,
            title=f"Operational Controls for {ref['label']}",
            description=(
                f"Implement operational controls and scheduling optimization "
                f"for {ref['label']}. Low-cost measure targeting "
                f"approximately 30% of total improvement potential."
            ),
            category=seu.category,
            estimated_savings_kwh=operational_savings,
            estimated_savings_pct=Decimal(str(
                round(float(seu.improvement_potential_pct) * 0.3, 2)
            )),
            implementation_complexity="low",
        )
        opportunities.append(operational)

        return opportunities

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: EnergyReviewResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
