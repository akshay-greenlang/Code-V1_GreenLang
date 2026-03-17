# -*- coding: utf-8 -*-
"""
Energy Assessment Workflow
==============================

5-phase workflow for energy consumption and mix assessment per ESRS E1-5.
Implements data collection, unit normalization, source classification,
mix calculation, and report generation with full provenance tracking.

Phases:
    1. DataCollection         -- Gather energy consumption data
    2. UnitNormalization      -- Normalize all values to MWh
    3. SourceClassification   -- Classify fossil vs. renewable
    4. MixCalculation         -- Calculate energy mix and intensity
    5. ReportGeneration       -- Produce E1-5 disclosure data

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

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


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
    """Phases of the energy assessment workflow."""
    DATA_COLLECTION = "data_collection"
    UNIT_NORMALIZATION = "unit_normalization"
    SOURCE_CLASSIFICATION = "source_classification"
    MIX_CALCULATION = "mix_calculation"
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


class EnergyType(str, Enum):
    """Energy source type classification."""
    FOSSIL_COAL = "fossil_coal"
    FOSSIL_OIL = "fossil_oil"
    FOSSIL_GAS = "fossil_gas"
    NUCLEAR = "nuclear"
    RENEWABLE_SOLAR = "renewable_solar"
    RENEWABLE_WIND = "renewable_wind"
    RENEWABLE_HYDRO = "renewable_hydro"
    RENEWABLE_BIOMASS = "renewable_biomass"
    RENEWABLE_GEOTHERMAL = "renewable_geothermal"
    RENEWABLE_OTHER = "renewable_other"
    PURCHASED_ELECTRICITY = "purchased_electricity"
    PURCHASED_HEAT = "purchased_heat"
    PURCHASED_STEAM = "purchased_steam"
    PURCHASED_COOLING = "purchased_cooling"
    SELF_GENERATED = "self_generated"


class EnergyCategory(str, Enum):
    """High-level energy category."""
    FOSSIL = "fossil"
    RENEWABLE = "renewable"
    NUCLEAR = "nuclear"
    PURCHASED = "purchased"
    SELF_GENERATED = "self_generated"


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


class EnergySource(BaseModel):
    """Single energy consumption record."""
    source_id: str = Field(default_factory=lambda: f"es-{_new_uuid()[:8]}")
    source_name: str = Field(..., description="Energy source name")
    energy_type: EnergyType = Field(..., description="Energy source type")
    consumption_value: float = Field(default=0.0, ge=0.0, description="Consumption quantity")
    consumption_unit: str = Field(default="MWh", description="Unit of consumption")
    consumption_mwh: float = Field(default=0.0, ge=0.0, description="Normalized MWh value")
    is_renewable: bool = Field(default=False, description="Whether renewable energy")
    is_self_generated: bool = Field(default=False, description="Self-generated or purchased")
    location: str = Field(default="", description="Facility or site location")
    supplier: str = Field(default="", description="Energy supplier name")
    reporting_year: int = Field(default=2025)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)


class EnergyMixResult(BaseModel):
    """Calculated energy mix breakdown."""
    total_consumption_mwh: float = Field(default=0.0)
    fossil_mwh: float = Field(default=0.0)
    renewable_mwh: float = Field(default=0.0)
    nuclear_mwh: float = Field(default=0.0)
    fossil_share_pct: float = Field(default=0.0)
    renewable_share_pct: float = Field(default=0.0)
    nuclear_share_pct: float = Field(default=0.0)
    source_breakdown: Dict[str, float] = Field(default_factory=dict)


class EnergyAssessmentInput(BaseModel):
    """Input data model for EnergyAssessmentWorkflow."""
    energy_sources: List[EnergySource] = Field(
        default_factory=list, description="Energy consumption records"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    revenue_eur: float = Field(default=0.0, ge=0.0, description="Revenue for intensity calc")
    headcount: int = Field(default=0, ge=0, description="FTE for intensity calc")
    renewable_target_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Renewable energy target %"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class EnergyAssessmentResult(BaseModel):
    """Complete result from energy assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="energy_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    total_consumption_mwh: float = Field(default=0.0)
    fossil_consumption_mwh: float = Field(default=0.0)
    renewable_consumption_mwh: float = Field(default=0.0)
    nuclear_consumption_mwh: float = Field(default=0.0)
    renewable_share_pct: float = Field(default=0.0)
    energy_intensity_mwh_per_eur: float = Field(default=0.0)
    energy_intensity_mwh_per_fte: float = Field(default=0.0)
    energy_mix: Optional[EnergyMixResult] = Field(default=None)
    energy_sources: List[EnergySource] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# UNIT CONVERSION FACTORS (to MWh)
# =============================================================================

UNIT_TO_MWH: Dict[str, float] = {
    "mwh": 1.0,
    "kwh": 0.001,
    "gwh": 1000.0,
    "gj": 0.277778,
    "tj": 277.778,
    "mj": 0.000277778,
    "therm": 0.029307,
    "btu": 0.000000293071,
    "mmbtu": 0.293071,
    "litre_diesel": 0.01005,
    "litre_petrol": 0.00887,
    "litre_lpg": 0.00717,
    "m3_natural_gas": 0.01055,
    "kg_coal": 0.00728,
    "tonne_coal": 7.28,
}

# Renewable energy type set
RENEWABLE_TYPES = {
    EnergyType.RENEWABLE_SOLAR,
    EnergyType.RENEWABLE_WIND,
    EnergyType.RENEWABLE_HYDRO,
    EnergyType.RENEWABLE_BIOMASS,
    EnergyType.RENEWABLE_GEOTHERMAL,
    EnergyType.RENEWABLE_OTHER,
}

FOSSIL_TYPES = {
    EnergyType.FOSSIL_COAL,
    EnergyType.FOSSIL_OIL,
    EnergyType.FOSSIL_GAS,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EnergyAssessmentWorkflow:
    """
    5-phase energy consumption and mix assessment workflow for ESRS E1-5.

    Implements energy data collection, unit normalization, source
    classification (fossil/renewable/nuclear), mix calculation with
    intensity metrics, and disclosure-ready report generation.

    Zero-hallucination: all calculations use deterministic arithmetic
    with documented conversion factors.

    Example:
        >>> wf = EnergyAssessmentWorkflow()
        >>> inp = EnergyAssessmentInput(energy_sources=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.renewable_share_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize EnergyAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sources: List[EnergySource] = []
        self._mix: Optional[EnergyMixResult] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DATA_COLLECTION.value, "description": "Gather energy consumption data"},
            {"name": WorkflowPhase.UNIT_NORMALIZATION.value, "description": "Normalize all values to MWh"},
            {"name": WorkflowPhase.SOURCE_CLASSIFICATION.value, "description": "Classify fossil vs renewable"},
            {"name": WorkflowPhase.MIX_CALCULATION.value, "description": "Calculate energy mix and intensity"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-5 disclosure data"},
        ]

    def validate_inputs(self, input_data: EnergyAssessmentInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.energy_sources:
            issues.append("No energy sources provided")
        for src in input_data.energy_sources:
            if src.consumption_value < 0:
                issues.append(f"Negative consumption in source {src.source_id}")
        if input_data.renewable_target_pct > 100:
            issues.append("Renewable target exceeds 100%")
        return issues

    async def execute(
        self,
        input_data: Optional[EnergyAssessmentInput] = None,
        energy_sources: Optional[List[EnergySource]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> EnergyAssessmentResult:
        """
        Execute the 5-phase energy assessment workflow.

        Args:
            input_data: Full input model (preferred).
            energy_sources: Energy sources (fallback).
            config: Configuration overrides.

        Returns:
            EnergyAssessmentResult with mix, intensity, and source details.
        """
        if input_data is None:
            input_data = EnergyAssessmentInput(
                energy_sources=energy_sources or [],
                config=config or {},
            )

        started_at = _utcnow()
        self.logger.info("Starting energy assessment workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_data_collection(input_data))
            phases_done += 1
            phase_results.append(await self._phase_unit_normalization(input_data))
            phases_done += 1
            phase_results.append(await self._phase_source_classification(input_data))
            phases_done += 1
            phase_results.append(await self._phase_mix_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Energy assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        total_mwh = sum(s.consumption_mwh for s in self._sources)
        fossil_mwh = sum(s.consumption_mwh for s in self._sources if s.energy_type in FOSSIL_TYPES)
        renewable_mwh = sum(s.consumption_mwh for s in self._sources if s.is_renewable)
        nuclear_mwh = sum(
            s.consumption_mwh for s in self._sources
            if s.energy_type == EnergyType.NUCLEAR
        )
        renewable_pct = round((renewable_mwh / total_mwh * 100) if total_mwh > 0 else 0.0, 2)

        intensity_eur = round(total_mwh / input_data.revenue_eur, 6) if input_data.revenue_eur > 0 else 0.0
        intensity_fte = round(total_mwh / input_data.headcount, 4) if input_data.headcount > 0 else 0.0

        result = EnergyAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            total_consumption_mwh=round(total_mwh, 4),
            fossil_consumption_mwh=round(fossil_mwh, 4),
            renewable_consumption_mwh=round(renewable_mwh, 4),
            nuclear_consumption_mwh=round(nuclear_mwh, 4),
            renewable_share_pct=renewable_pct,
            energy_intensity_mwh_per_eur=intensity_eur,
            energy_intensity_mwh_per_fte=intensity_fte,
            energy_mix=self._mix,
            energy_sources=self._sources,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Energy assessment %s completed in %.2fs: %.2f MWh total, %.1f%% renewable",
            self.workflow_id, elapsed, total_mwh, renewable_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: EnergyAssessmentInput,
    ) -> PhaseResult:
        """Gather energy consumption records."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._sources = list(input_data.energy_sources)

        type_counts: Dict[str, int] = {}
        for src in self._sources:
            type_counts[src.energy_type.value] = type_counts.get(src.energy_type.value, 0) + 1

        outputs["sources_collected"] = len(self._sources)
        outputs["type_distribution"] = type_counts
        outputs["unique_locations"] = len(set(s.location for s in self._sources if s.location))

        if not self._sources:
            warnings.append("No energy sources provided; assessment will be empty")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 1 DataCollection: %d sources collected", len(self._sources))
        return PhaseResult(
            phase_name=WorkflowPhase.DATA_COLLECTION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Unit Normalization
    # -------------------------------------------------------------------------

    async def _phase_unit_normalization(
        self, input_data: EnergyAssessmentInput,
    ) -> PhaseResult:
        """Normalize all energy values to MWh."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        normalized_count = 0

        for src in self._sources:
            if src.consumption_mwh > 0:
                normalized_count += 1
                continue

            unit_key = src.consumption_unit.lower().replace(" ", "")
            factor = UNIT_TO_MWH.get(unit_key, 0.0)

            if factor > 0:
                src.consumption_mwh = round(src.consumption_value * factor, 6)
                normalized_count += 1
            else:
                # Assume already in MWh if unit not recognized
                src.consumption_mwh = src.consumption_value
                warnings.append(
                    f"Source {src.source_id}: unrecognized unit '{src.consumption_unit}', "
                    f"assuming MWh"
                )
                normalized_count += 1

        total_mwh = sum(s.consumption_mwh for s in self._sources)
        outputs["sources_normalized"] = normalized_count
        outputs["total_consumption_mwh"] = round(total_mwh, 4)
        outputs["conversion_warnings"] = len(warnings)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 UnitNormalization: %d sources normalized, %.2f MWh total",
            normalized_count, total_mwh,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.UNIT_NORMALIZATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Source Classification
    # -------------------------------------------------------------------------

    async def _phase_source_classification(
        self, input_data: EnergyAssessmentInput,
    ) -> PhaseResult:
        """Classify energy sources as fossil, renewable, or nuclear."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        for src in self._sources:
            if src.energy_type in RENEWABLE_TYPES:
                src.is_renewable = True
            elif src.energy_type in FOSSIL_TYPES:
                src.is_renewable = False

            if src.energy_type == EnergyType.SELF_GENERATED:
                src.is_self_generated = True

        renewable_count = sum(1 for s in self._sources if s.is_renewable)
        fossil_count = sum(1 for s in self._sources if s.energy_type in FOSSIL_TYPES)
        nuclear_count = sum(1 for s in self._sources if s.energy_type == EnergyType.NUCLEAR)

        outputs["renewable_sources"] = renewable_count
        outputs["fossil_sources"] = fossil_count
        outputs["nuclear_sources"] = nuclear_count
        outputs["self_generated_sources"] = sum(1 for s in self._sources if s.is_self_generated)
        outputs["purchased_sources"] = len(self._sources) - sum(
            1 for s in self._sources if s.is_self_generated
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SourceClassification: %d renewable, %d fossil, %d nuclear",
            renewable_count, fossil_count, nuclear_count,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SOURCE_CLASSIFICATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Mix Calculation
    # -------------------------------------------------------------------------

    async def _phase_mix_calculation(
        self, input_data: EnergyAssessmentInput,
    ) -> PhaseResult:
        """Calculate energy mix percentages and intensity metrics."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_mwh = sum(s.consumption_mwh for s in self._sources)
        fossil_mwh = sum(s.consumption_mwh for s in self._sources if s.energy_type in FOSSIL_TYPES)
        renewable_mwh = sum(s.consumption_mwh for s in self._sources if s.is_renewable)
        nuclear_mwh = sum(
            s.consumption_mwh for s in self._sources
            if s.energy_type == EnergyType.NUCLEAR
        )

        fossil_pct = round((fossil_mwh / total_mwh * 100) if total_mwh > 0 else 0.0, 2)
        renewable_pct = round((renewable_mwh / total_mwh * 100) if total_mwh > 0 else 0.0, 2)
        nuclear_pct = round((nuclear_mwh / total_mwh * 100) if total_mwh > 0 else 0.0, 2)

        # Source-level breakdown
        source_breakdown: Dict[str, float] = {}
        for src in self._sources:
            source_breakdown[src.energy_type.value] = (
                source_breakdown.get(src.energy_type.value, 0.0) + src.consumption_mwh
            )

        self._mix = EnergyMixResult(
            total_consumption_mwh=round(total_mwh, 4),
            fossil_mwh=round(fossil_mwh, 4),
            renewable_mwh=round(renewable_mwh, 4),
            nuclear_mwh=round(nuclear_mwh, 4),
            fossil_share_pct=fossil_pct,
            renewable_share_pct=renewable_pct,
            nuclear_share_pct=nuclear_pct,
            source_breakdown={k: round(v, 4) for k, v in source_breakdown.items()},
        )

        # Check against target
        if input_data.renewable_target_pct > 0:
            gap = input_data.renewable_target_pct - renewable_pct
            if gap > 0:
                warnings.append(
                    f"Renewable share ({renewable_pct}%) is {gap:.1f}pp below target "
                    f"({input_data.renewable_target_pct}%)"
                )

        outputs["total_consumption_mwh"] = round(total_mwh, 4)
        outputs["fossil_share_pct"] = fossil_pct
        outputs["renewable_share_pct"] = renewable_pct
        outputs["nuclear_share_pct"] = nuclear_pct
        outputs["target_gap_pp"] = round(
            max(0, input_data.renewable_target_pct - renewable_pct), 2
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 MixCalculation: %.1f%% fossil, %.1f%% renewable, %.1f%% nuclear",
            fossil_pct, renewable_pct, nuclear_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.MIX_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: EnergyAssessmentInput,
    ) -> PhaseResult:
        """Generate E1-5 disclosure-ready output."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        mix = self._mix
        outputs["e1_5_disclosure"] = {
            "total_energy_consumption_mwh": mix.total_consumption_mwh if mix else 0.0,
            "fossil_sources_mwh": mix.fossil_mwh if mix else 0.0,
            "renewable_sources_mwh": mix.renewable_mwh if mix else 0.0,
            "nuclear_sources_mwh": mix.nuclear_mwh if mix else 0.0,
            "share_of_renewable_pct": mix.renewable_share_pct if mix else 0.0,
            "energy_intensity_mwh_per_m_eur": round(
                (mix.total_consumption_mwh / (input_data.revenue_eur / 1_000_000))
                if mix and input_data.revenue_eur > 0 else 0.0, 4
            ),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True
        outputs["source_count"] = len(self._sources)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReportGeneration: E1-5 disclosure ready, %d sources",
            len(self._sources),
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

    def _compute_provenance(self, result: EnergyAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
