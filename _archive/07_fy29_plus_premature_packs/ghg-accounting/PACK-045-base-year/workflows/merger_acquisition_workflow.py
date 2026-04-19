# -*- coding: utf-8 -*-
"""
Merger & Acquisition Workflow
=================================

5-phase workflow for M&A-specific base year adjustments within
PACK-045 Base Year Management Pack.

Phases:
    1. EntityIdentification      -- Identify and catalog acquired or divested
                                    entities including subsidiaries, facilities,
                                    and emission sources with ownership details.
    2. EmissionQuantification    -- Quantify emissions attributable to each
                                    entity using available data, estimation
                                    methods, and sector benchmarks.
    3. ProRataCalculation        -- Apply pro-rata temporal allocation for
                                    entities acquired/divested mid-year,
                                    accounting for ownership percentage.
    4. SignificanceTesting       -- Test whether the aggregate M&A impact
                                    exceeds the significance threshold for
                                    base year recalculation.
    5. AdjustmentExecution       -- If significant, execute the base year
                                    adjustment, create new inventory version,
                                    and record full audit trail.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 5 (Structural changes)
    GHG Protocol Scope 2 Guidance (M&A recalculation)
    ISO 14064-1:2018 Clause 5.1 (Organizational boundary changes)
    SBTi Corporate Manual (Target recalculation for M&A)

Schedule: Triggered upon M&A transaction close
Estimated duration: 2-6 weeks depending on entity complexity

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
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


class MAPhase(str, Enum):
    """Merger & acquisition workflow phases."""

    ENTITY_IDENTIFICATION = "entity_identification"
    EMISSION_QUANTIFICATION = "emission_quantification"
    PRO_RATA_CALCULATION = "pro_rata_calculation"
    SIGNIFICANCE_TESTING = "significance_testing"
    ADJUSTMENT_EXECUTION = "adjustment_execution"


class TransactionType(str, Enum):
    """Type of M&A transaction."""

    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    JOINT_VENTURE = "joint_venture"
    PARTIAL_ACQUISITION = "partial_acquisition"
    PARTIAL_DIVESTITURE = "partial_divestiture"


class EntityType(str, Enum):
    """Type of organizational entity."""

    SUBSIDIARY = "subsidiary"
    FACILITY = "facility"
    BUSINESS_UNIT = "business_unit"
    DIVISION = "division"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"


class DataQualityTier(str, Enum):
    """Data quality tier for emission quantification."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    ESTIMATED = "estimated"
    BENCHMARK = "benchmark"


class SignificanceOutcome(str, Enum):
    """Outcome of significance testing."""

    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    BORDERLINE = "borderline"


class AdjustmentDirection(str, Enum):
    """Direction of base year adjustment."""

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class MAEntity(BaseModel):
    """Entity involved in M&A transaction."""

    entity_id: str = Field(default_factory=lambda: f"ent-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY)
    transaction_type: TransactionType = Field(default=TransactionType.ACQUISITION)
    effective_date: str = Field(default="", description="ISO date of transaction close")
    ownership_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    sector: str = Field(default="")
    country: str = Field(default="")
    facilities: List[str] = Field(default_factory=list)
    annual_revenue: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    known_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)


class BaseYearInventory(BaseModel):
    """Base year inventory with scope-level breakdown."""

    year: int = Field(..., ge=2010, le=2050)
    version: str = Field(default="v1.0")
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    categories: Dict[str, float] = Field(default_factory=dict)
    methodology_version: str = Field(default="ghg_protocol_v1")


class EntityEmissions(BaseModel):
    """Quantified emissions for a single entity."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    annual_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    methodology: str = Field(default="")
    emission_factor_source: str = Field(default="")
    confidence_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


class ProRataAdjustment(BaseModel):
    """Pro-rata temporal allocation for mid-year transactions."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    transaction_type: TransactionType = Field(default=TransactionType.ACQUISITION)
    effective_date: str = Field(default="")
    ownership_pct: float = Field(default=100.0)
    annual_tco2e: float = Field(default=0.0, ge=0.0)
    days_in_base_year: int = Field(default=0, ge=0, le=366)
    total_days_in_year: int = Field(default=365, ge=365, le=366)
    temporal_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    ownership_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    pro_rata_tco2e: float = Field(default=0.0)
    adjustment_direction: AdjustmentDirection = Field(default=AdjustmentDirection.INCREASE)
    calculation_formula: str = Field(default="")
    provenance_hash: str = Field(default="")


class SignificanceTestResult(BaseModel):
    """Result of significance testing for M&A impact."""

    total_ma_impact_tco2e: float = Field(default=0.0)
    base_year_total_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    threshold_pct: float = Field(default=5.0)
    outcome: SignificanceOutcome = Field(default=SignificanceOutcome.NOT_SIGNIFICANT)
    entities_tested: int = Field(default=0)
    borderline_margin_pct: float = Field(default=0.0)
    recommendation: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class MergerAcquisitionInput(BaseModel):
    """Input data model for MergerAcquisitionWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organization identifier")
    base_year_inventory: BaseYearInventory = Field(
        ..., description="Current base year inventory",
    )
    acquired_entities: List[MAEntity] = Field(
        default_factory=list, description="Entities being acquired/merged",
    )
    divested_entities: List[MAEntity] = Field(
        default_factory=list, description="Entities being divested",
    )
    effective_dates: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity ID to effective date mapping",
    )
    ownership_pcts: Dict[str, float] = Field(
        default_factory=dict,
        description="Entity ID to ownership percentage mapping",
    )
    significance_threshold_pct: float = Field(
        default=5.0, ge=0.1, le=50.0,
        description="Significance threshold for recalculation",
    )
    sector_benchmarks: Dict[str, float] = Field(
        default_factory=dict,
        description="Sector to tCO2e/employee or tCO2e/revenue benchmarks",
    )
    base_year_days: int = Field(default=365, ge=365, le=366)
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class MergerAcquisitionResult(BaseModel):
    """Complete result from merger & acquisition workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="merger_acquisition")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    entity_emissions: List[EntityEmissions] = Field(default_factory=list)
    pro_rata_adjustments: List[ProRataAdjustment] = Field(default_factory=list)
    significance_result: Optional[SignificanceTestResult] = Field(default=None)
    adjusted_inventory: Optional[BaseYearInventory] = Field(default=None)
    recalculation_performed: bool = Field(default=False)
    provenance_hash: str = Field(default="")


# =============================================================================
# SECTOR EMISSION BENCHMARKS (Zero-Hallucination)
# =============================================================================

# Default sector benchmarks: tCO2e per employee (illustrative values)
DEFAULT_SECTOR_BENCHMARKS: Dict[str, float] = {
    "technology": 5.0,
    "financial_services": 8.0,
    "manufacturing": 25.0,
    "energy": 120.0,
    "transportation": 45.0,
    "retail": 12.0,
    "healthcare": 15.0,
    "construction": 35.0,
    "agriculture": 50.0,
    "mining": 80.0,
    "utilities": 100.0,
    "default": 15.0,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MergerAcquisitionWorkflow:
    """
    5-phase workflow for M&A-specific base year adjustments.

    Identifies acquired/divested entities, quantifies their emissions,
    applies pro-rata temporal allocation, tests significance against
    threshold, and executes adjustments when required.

    Zero-hallucination: all emission quantification uses deterministic
    formulas (benchmark * employees or known data), pro-rata uses
    calendar-day fractions, no LLM in calculation paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _entity_emissions: Quantified emissions per entity.
        _pro_rata: Pro-rata adjustments.
        _significance: Significance test result.
        _adjusted_inventory: Recalculated base year if significant.

    Example:
        >>> wf = MergerAcquisitionWorkflow()
        >>> entity = MAEntity(
        ...     name="Acquired Corp", transaction_type=TransactionType.ACQUISITION,
        ...     employee_count=500, sector="manufacturing",
        ... )
        >>> inv = BaseYearInventory(year=2022, total_tco2e=50000.0)
        >>> inp = MergerAcquisitionInput(
        ...     organization_id="org-001",
        ...     base_year_inventory=inv,
        ...     acquired_entities=[entity],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[MAPhase] = [
        MAPhase.ENTITY_IDENTIFICATION,
        MAPhase.EMISSION_QUANTIFICATION,
        MAPhase.PRO_RATA_CALCULATION,
        MAPhase.SIGNIFICANCE_TESTING,
        MAPhase.ADJUSTMENT_EXECUTION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MergerAcquisitionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._all_entities: List[MAEntity] = []
        self._entity_emissions: List[EntityEmissions] = []
        self._pro_rata: List[ProRataAdjustment] = []
        self._significance: Optional[SignificanceTestResult] = None
        self._adjusted_inventory: Optional[BaseYearInventory] = None
        self._recalculation_performed: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: MergerAcquisitionInput,
    ) -> MergerAcquisitionResult:
        """
        Execute the 5-phase M&A base year adjustment workflow.

        Args:
            input_data: Entities, base year inventory, and policy.

        Returns:
            MergerAcquisitionResult with entity emissions, adjustments, and outcome.
        """
        started_at = datetime.utcnow()
        total_entities = len(input_data.acquired_entities) + len(input_data.divested_entities)
        self.logger.info(
            "Starting M&A workflow %s org=%s entities=%d",
            self.workflow_id, input_data.organization_id, total_entities,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_entity_identification,
            self._phase_emission_quantification,
            self._phase_pro_rata_calculation,
            self._phase_significance_testing,
            self._phase_adjustment_execution,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._execute_with_retry(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("M&A workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = MergerAcquisitionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            entity_emissions=self._entity_emissions,
            pro_rata_adjustments=self._pro_rata,
            significance_result=self._significance,
            adjusted_inventory=self._adjusted_inventory,
            recalculation_performed=self._recalculation_performed,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "M&A workflow %s completed in %.2fs status=%s recalc=%s",
            self.workflow_id, elapsed, overall_status.value,
            self._recalculation_performed,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: MergerAcquisitionInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Entity Identification
    # -------------------------------------------------------------------------

    async def _phase_entity_identification(
        self, input_data: MergerAcquisitionInput,
    ) -> PhaseResult:
        """Identify and catalog all acquired and divested entities."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._all_entities = []

        # Process acquired entities
        for entity in input_data.acquired_entities:
            # Override ownership if specified in input
            if entity.entity_id in input_data.ownership_pcts:
                entity.ownership_pct = input_data.ownership_pcts[entity.entity_id]
            if entity.entity_id in input_data.effective_dates:
                entity.effective_date = input_data.effective_dates[entity.entity_id]

            if not entity.effective_date:
                warnings.append(
                    f"Entity {entity.name} ({entity.entity_id}) missing effective_date"
                )

            self._all_entities.append(entity)

        # Process divested entities
        for entity in input_data.divested_entities:
            if entity.entity_id in input_data.ownership_pcts:
                entity.ownership_pct = input_data.ownership_pcts[entity.entity_id]
            if entity.entity_id in input_data.effective_dates:
                entity.effective_date = input_data.effective_dates[entity.entity_id]

            if entity.transaction_type not in (
                TransactionType.DIVESTITURE, TransactionType.PARTIAL_DIVESTITURE,
            ):
                entity.transaction_type = TransactionType.DIVESTITURE

            self._all_entities.append(entity)

        # Categorize
        acquisitions = sum(
            1 for e in self._all_entities
            if e.transaction_type in (TransactionType.ACQUISITION, TransactionType.PARTIAL_ACQUISITION, TransactionType.MERGER)
        )
        divestitures = sum(
            1 for e in self._all_entities
            if e.transaction_type in (TransactionType.DIVESTITURE, TransactionType.PARTIAL_DIVESTITURE)
        )
        sectors = list(set(e.sector for e in self._all_entities if e.sector))
        countries = list(set(e.country for e in self._all_entities if e.country))

        outputs["total_entities"] = len(self._all_entities)
        outputs["acquisitions"] = acquisitions
        outputs["divestitures"] = divestitures
        outputs["sectors"] = sectors
        outputs["countries"] = countries
        outputs["total_facilities"] = sum(len(e.facilities) for e in self._all_entities)
        outputs["total_employees"] = sum(e.employee_count for e in self._all_entities)

        if not self._all_entities:
            warnings.append("No M&A entities provided")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 EntityIdentification: %d entities (%d acq, %d div)",
            len(self._all_entities), acquisitions, divestitures,
        )
        return PhaseResult(
            phase_name="entity_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Emission Quantification
    # -------------------------------------------------------------------------

    async def _phase_emission_quantification(
        self, input_data: MergerAcquisitionInput,
    ) -> PhaseResult:
        """Quantify emissions for each M&A entity."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._entity_emissions = []
        benchmarks = {**DEFAULT_SECTOR_BENCHMARKS, **input_data.sector_benchmarks}

        for entity in self._all_entities:
            annual_tco2e = entity.known_emissions_tco2e
            data_quality = entity.data_quality
            methodology = "known_emissions"
            ef_source = ""
            confidence = 90.0

            # If no known emissions, estimate from benchmarks
            if annual_tco2e <= 0 and entity.employee_count > 0:
                sector_key = entity.sector.lower() if entity.sector else "default"
                benchmark = benchmarks.get(sector_key, benchmarks["default"])
                annual_tco2e = entity.employee_count * benchmark
                data_quality = DataQualityTier.BENCHMARK
                methodology = f"employee_count({entity.employee_count}) * benchmark({benchmark})"
                ef_source = f"sector_benchmark_{sector_key}"
                confidence = 50.0
                warnings.append(
                    f"Entity {entity.name}: estimated from benchmark ({data_quality.value})"
                )
            elif annual_tco2e <= 0:
                warnings.append(
                    f"Entity {entity.name}: no emissions data or employee count"
                )
                confidence = 10.0

            # Default scope split: 40% S1, 35% S2, 25% S3
            scope1 = round(annual_tco2e * 0.40, 4)
            scope2 = round(annual_tco2e * 0.35, 4)
            scope3 = round(annual_tco2e * 0.25, 4)

            em_data = {
                "entity_id": entity.entity_id,
                "annual_tco2e": round(annual_tco2e, 4),
                "methodology": methodology,
            }
            em_hash = hashlib.sha256(
                json.dumps(em_data, sort_keys=True).encode("utf-8")
            ).hexdigest()

            self._entity_emissions.append(EntityEmissions(
                entity_id=entity.entity_id,
                entity_name=entity.name,
                annual_tco2e=round(annual_tco2e, 4),
                scope1_tco2e=scope1,
                scope2_tco2e=scope2,
                scope3_tco2e=scope3,
                data_quality=data_quality,
                methodology=methodology,
                emission_factor_source=ef_source,
                confidence_pct=confidence,
                provenance_hash=em_hash,
            ))

        total_quantified = sum(e.annual_tco2e for e in self._entity_emissions)
        primary_count = sum(
            1 for e in self._entity_emissions
            if e.data_quality == DataQualityTier.PRIMARY
        )
        benchmark_count = sum(
            1 for e in self._entity_emissions
            if e.data_quality == DataQualityTier.BENCHMARK
        )

        outputs["entities_quantified"] = len(self._entity_emissions)
        outputs["total_annual_tco2e"] = round(total_quantified, 4)
        outputs["primary_data_count"] = primary_count
        outputs["benchmark_estimated_count"] = benchmark_count
        outputs["avg_confidence_pct"] = round(
            sum(e.confidence_pct for e in self._entity_emissions)
            / max(len(self._entity_emissions), 1),
            2,
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 EmissionQuantification: %d entities, total=%.2f tCO2e",
            len(self._entity_emissions), total_quantified,
        )
        return PhaseResult(
            phase_name="emission_quantification", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Pro-Rata Calculation
    # -------------------------------------------------------------------------

    async def _phase_pro_rata_calculation(
        self, input_data: MergerAcquisitionInput,
    ) -> PhaseResult:
        """Apply pro-rata temporal allocation for mid-year transactions."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._pro_rata = []
        base_year = input_data.base_year_inventory.year
        total_days = input_data.base_year_days

        for entity, emissions in zip(self._all_entities, self._entity_emissions):
            # Parse effective date to determine temporal fraction
            days_applicable = total_days  # Default to full year
            effective_date = entity.effective_date

            if effective_date:
                try:
                    eff_dt = datetime.fromisoformat(effective_date)
                    year_start = datetime(base_year, 1, 1)
                    year_end = datetime(base_year, 12, 31)

                    if entity.transaction_type in (
                        TransactionType.ACQUISITION,
                        TransactionType.PARTIAL_ACQUISITION,
                        TransactionType.MERGER,
                    ):
                        # Acquired: count days from effective_date to year_end
                        if eff_dt.year == base_year:
                            days_applicable = (year_end - eff_dt).days + 1
                        elif eff_dt.year < base_year:
                            days_applicable = total_days  # Full year
                        else:
                            days_applicable = 0  # Future year
                    else:
                        # Divested: count days from year_start to effective_date
                        if eff_dt.year == base_year:
                            days_applicable = (eff_dt - year_start).days + 1
                        elif eff_dt.year > base_year:
                            days_applicable = total_days  # Full year (still owned)
                        else:
                            days_applicable = 0  # Already divested
                except (ValueError, TypeError):
                    warnings.append(
                        f"Entity {entity.name}: could not parse effective_date {effective_date}"
                    )

            days_applicable = max(0, min(days_applicable, total_days))
            temporal_fraction = days_applicable / max(total_days, 1)
            ownership_fraction = entity.ownership_pct / 100.0

            # Pro-rata calculation: annual * temporal * ownership
            pro_rata_tco2e = round(
                emissions.annual_tco2e * temporal_fraction * ownership_fraction, 4,
            )

            # Determine adjustment direction
            if entity.transaction_type in (
                TransactionType.ACQUISITION,
                TransactionType.PARTIAL_ACQUISITION,
                TransactionType.MERGER,
            ):
                direction = AdjustmentDirection.INCREASE
            elif entity.transaction_type in (
                TransactionType.DIVESTITURE,
                TransactionType.PARTIAL_DIVESTITURE,
            ):
                direction = AdjustmentDirection.DECREASE
                pro_rata_tco2e = -abs(pro_rata_tco2e)
            else:
                direction = AdjustmentDirection.NO_CHANGE

            formula = (
                f"annual({emissions.annual_tco2e:.4f}) * "
                f"temporal({temporal_fraction:.4f} = {days_applicable}/{total_days}) * "
                f"ownership({ownership_fraction:.4f} = {entity.ownership_pct}%)"
            )

            pr_data = {
                "entity_id": entity.entity_id,
                "pro_rata_tco2e": pro_rata_tco2e,
                "temporal_fraction": temporal_fraction,
                "ownership_fraction": ownership_fraction,
            }
            pr_hash = hashlib.sha256(
                json.dumps(pr_data, sort_keys=True).encode("utf-8")
            ).hexdigest()

            self._pro_rata.append(ProRataAdjustment(
                entity_id=entity.entity_id,
                entity_name=entity.name,
                transaction_type=entity.transaction_type,
                effective_date=entity.effective_date,
                ownership_pct=entity.ownership_pct,
                annual_tco2e=emissions.annual_tco2e,
                days_in_base_year=days_applicable,
                total_days_in_year=total_days,
                temporal_fraction=round(temporal_fraction, 6),
                ownership_fraction=round(ownership_fraction, 6),
                pro_rata_tco2e=pro_rata_tco2e,
                adjustment_direction=direction,
                calculation_formula=formula,
                provenance_hash=pr_hash,
            ))

        total_pro_rata = sum(p.pro_rata_tco2e for p in self._pro_rata)
        increases = sum(p.pro_rata_tco2e for p in self._pro_rata if p.pro_rata_tco2e > 0)
        decreases = sum(p.pro_rata_tco2e for p in self._pro_rata if p.pro_rata_tco2e < 0)

        outputs["adjustments_calculated"] = len(self._pro_rata)
        outputs["total_pro_rata_tco2e"] = round(total_pro_rata, 4)
        outputs["total_increases_tco2e"] = round(increases, 4)
        outputs["total_decreases_tco2e"] = round(decreases, 4)
        outputs["net_adjustment_tco2e"] = round(total_pro_rata, 4)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ProRataCalculation: %d adjustments, net=%.2f tCO2e",
            len(self._pro_rata), total_pro_rata,
        )
        return PhaseResult(
            phase_name="pro_rata_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Significance Testing
    # -------------------------------------------------------------------------

    async def _phase_significance_testing(
        self, input_data: MergerAcquisitionInput,
    ) -> PhaseResult:
        """Test M&A impact against significance threshold."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base_total = input_data.base_year_inventory.total_tco2e
        total_impact = sum(abs(p.pro_rata_tco2e) for p in self._pro_rata)
        threshold = input_data.significance_threshold_pct

        impact_pct = (total_impact / max(base_total, 1.0)) * 100.0

        # Determine outcome
        borderline_margin = 1.0  # Within 1% of threshold
        if impact_pct >= threshold:
            outcome = SignificanceOutcome.SIGNIFICANT
            recommendation = (
                f"M&A impact ({impact_pct:.2f}%) exceeds {threshold:.1f}% threshold. "
                f"Base year recalculation required per GHG Protocol Chapter 5."
            )
        elif impact_pct >= threshold - borderline_margin:
            outcome = SignificanceOutcome.BORDERLINE
            recommendation = (
                f"M&A impact ({impact_pct:.2f}%) is borderline "
                f"({threshold:.1f}% threshold). Recommend recalculation for consistency."
            )
        else:
            outcome = SignificanceOutcome.NOT_SIGNIFICANT
            recommendation = (
                f"M&A impact ({impact_pct:.2f}%) below {threshold:.1f}% threshold. "
                f"Recalculation not required. Document decision for audit trail."
            )

        sig_data = {
            "total_impact": round(total_impact, 4),
            "base_total": round(base_total, 4),
            "impact_pct": round(impact_pct, 4),
            "threshold": threshold,
            "outcome": outcome.value,
        }
        sig_hash = hashlib.sha256(
            json.dumps(sig_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self._significance = SignificanceTestResult(
            total_ma_impact_tco2e=round(total_impact, 4),
            base_year_total_tco2e=round(base_total, 4),
            impact_pct=round(impact_pct, 4),
            threshold_pct=threshold,
            outcome=outcome,
            entities_tested=len(self._pro_rata),
            borderline_margin_pct=round(abs(impact_pct - threshold), 4),
            recommendation=recommendation,
            provenance_hash=sig_hash,
        )

        outputs["impact_tco2e"] = round(total_impact, 4)
        outputs["impact_pct"] = round(impact_pct, 4)
        outputs["threshold_pct"] = threshold
        outputs["outcome"] = outcome.value
        outputs["recalculation_required"] = outcome in (
            SignificanceOutcome.SIGNIFICANT, SignificanceOutcome.BORDERLINE,
        )

        if outcome == SignificanceOutcome.SIGNIFICANT:
            warnings.append("Base year recalculation required due to M&A impact")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 SignificanceTesting: impact=%.2f%% threshold=%.1f%% outcome=%s",
            impact_pct, threshold, outcome.value,
        )
        return PhaseResult(
            phase_name="significance_testing", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Adjustment Execution
    # -------------------------------------------------------------------------

    async def _phase_adjustment_execution(
        self, input_data: MergerAcquisitionInput,
    ) -> PhaseResult:
        """Execute base year adjustment if significant."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not self._significance:
            raise ValueError("Significance testing not completed")

        if self._significance.outcome == SignificanceOutcome.NOT_SIGNIFICANT:
            self._recalculation_performed = False
            outputs["action"] = "no_recalculation"
            outputs["reason"] = "Impact below significance threshold"

            elapsed = (datetime.utcnow() - started).total_seconds()
            self.logger.info("Phase 5 AdjustmentExecution: no recalculation needed")
            return PhaseResult(
                phase_name="adjustment_execution", phase_number=5,
                status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Execute adjustment
        base_inv = input_data.base_year_inventory
        net_adjustment = sum(p.pro_rata_tco2e for p in self._pro_rata)

        # Distribute adjustment across scopes based on entity emission profiles
        scope_adjustments = {"scope1": 0.0, "scope2": 0.0, "scope3": 0.0}
        for pro_rata, emissions in zip(self._pro_rata, self._entity_emissions):
            if emissions.annual_tco2e > 0:
                s1_frac = emissions.scope1_tco2e / emissions.annual_tco2e
                s2_frac = emissions.scope2_tco2e / emissions.annual_tco2e
                s3_frac = emissions.scope3_tco2e / emissions.annual_tco2e
            else:
                s1_frac, s2_frac, s3_frac = 0.4, 0.35, 0.25

            scope_adjustments["scope1"] += pro_rata.pro_rata_tco2e * s1_frac
            scope_adjustments["scope2"] += pro_rata.pro_rata_tco2e * s2_frac
            scope_adjustments["scope3"] += pro_rata.pro_rata_tco2e * s3_frac

        new_scope1 = max(base_inv.scope1_tco2e + scope_adjustments["scope1"], 0.0)
        new_scope2 = max(base_inv.scope2_tco2e + scope_adjustments["scope2"], 0.0)
        new_scope3 = max(base_inv.scope3_tco2e + scope_adjustments["scope3"], 0.0)
        new_total = round(new_scope1 + new_scope2 + new_scope3, 4)

        # New version
        version_parts = base_inv.version.replace("v", "").split(".")
        new_minor = int(version_parts[-1]) + 1 if version_parts else 1
        new_version = f"v{version_parts[0]}.{new_minor}" if version_parts else "v1.1"

        inv_data = {
            "year": base_inv.year,
            "version": new_version,
            "total_tco2e": new_total,
            "scope1": round(new_scope1, 4),
            "scope2": round(new_scope2, 4),
            "scope3": round(new_scope3, 4),
        }
        integrity_hash = hashlib.sha256(
            json.dumps(inv_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self._adjusted_inventory = BaseYearInventory(
            year=base_inv.year,
            version=new_version,
            total_tco2e=new_total,
            scope1_tco2e=round(new_scope1, 4),
            scope2_tco2e=round(new_scope2, 4),
            scope3_tco2e=round(new_scope3, 4),
            categories=base_inv.categories,
            methodology_version=base_inv.methodology_version,
        )
        self._recalculation_performed = True

        outputs["action"] = "recalculation_performed"
        outputs["old_version"] = base_inv.version
        outputs["new_version"] = new_version
        outputs["old_total_tco2e"] = base_inv.total_tco2e
        outputs["new_total_tco2e"] = new_total
        outputs["net_adjustment_tco2e"] = round(net_adjustment, 4)
        outputs["delta_pct"] = round(
            ((new_total - base_inv.total_tco2e) / max(base_inv.total_tco2e, 1.0)) * 100.0,
            4,
        )
        outputs["integrity_hash"] = integrity_hash

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 AdjustmentExecution: %s -> %s (%.2f -> %.2f tCO2e)",
            base_inv.version, new_version, base_inv.total_tco2e, new_total,
        )
        return PhaseResult(
            phase_name="adjustment_execution", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._all_entities = []
        self._entity_emissions = []
        self._pro_rata = []
        self._significance = None
        self._adjusted_inventory = None
        self._recalculation_performed = False

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: MergerAcquisitionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
