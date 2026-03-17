# -*- coding: utf-8 -*-
"""
Manufacturing Emissions Workflow
================================

Four-phase workflow for comprehensive manufacturing emissions assessment
covering Scope 1 (process + combustion), Scope 2 (energy), and Scope 3
(upstream/downstream) emissions with intensity metrics per unit produced.

Phases:
    1. DataCollection - Validate facility, production, and energy input data
    2. ProcessCalculation - Compute process + combustion (Scope 1) emissions
    3. EnergyAnalysis - Compute Scope 2 + energy intensity metrics
    4. Consolidation - Aggregate totals, compute intensities, build provenance

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
#  Shared models
# ---------------------------------------------------------------------------

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store outputs from a completed phase."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previously completed phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Update the status of a phase."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check whether a phase has completed successfully."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
#  Input / Result
# ---------------------------------------------------------------------------

class FacilityRecord(BaseModel):
    """Single manufacturing facility record."""
    facility_id: str = Field(..., description="Unique facility identifier")
    facility_name: str = Field(default="", description="Facility display name")
    country: str = Field(default="", description="ISO country code")
    sector: str = Field(default="", description="NACE sector code")
    sub_sector: str = Field(default="", description="Manufacturing sub-sector")


class ProductionVolume(BaseModel):
    """Production volume for a facility / product."""
    facility_id: str = Field(...)
    product_id: str = Field(default="")
    product_name: str = Field(default="")
    quantity: float = Field(..., ge=0.0, description="Quantity produced")
    unit: str = Field(default="tonnes")


class EnergyRecord(BaseModel):
    """Energy consumption record."""
    facility_id: str = Field(...)
    source: str = Field(..., description="Energy source (e.g. natural_gas, electricity)")
    consumption: float = Field(..., ge=0.0, description="Consumption value")
    unit: str = Field(default="MWh")
    emission_factor: float = Field(default=0.0, ge=0.0, description="kgCO2e per unit")
    scope: str = Field(default="scope2", description="scope1 or scope2")


class ManufacturingEmissionsInput(BaseModel):
    """Input for manufacturing emissions workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2000, le=2100)
    facility_data: List[FacilityRecord] = Field(default_factory=list)
    production_volumes: List[ProductionVolume] = Field(default_factory=list)
    energy_data: List[EnergyRecord] = Field(default_factory=list)
    process_emission_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Facility ID -> process emission factor (tCO2e/tonne product)",
    )
    combustion_fuels: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Fuel consumption records for combustion (Scope 1)",
    )
    scope3_upstream: float = Field(default=0.0, ge=0.0, description="Estimated scope 3 upstream tCO2e")
    scope3_downstream: float = Field(default=0.0, ge=0.0, description="Estimated scope 3 downstream tCO2e")
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Ensure reporting year is reasonable."""
        if v < 2015:
            raise ValueError("Reporting year must be 2015 or later")
        return v


class ManufacturingEmissionsResult(WorkflowResult):
    """Result from the manufacturing emissions workflow."""
    scope1_total: float = Field(default=0.0, description="Total Scope 1 tCO2e")
    scope2_total: float = Field(default=0.0, description="Total Scope 2 tCO2e")
    scope3_total: float = Field(default=0.0, description="Total Scope 3 tCO2e")
    process_emissions: float = Field(default=0.0, description="Process emissions tCO2e")
    combustion_emissions: float = Field(default=0.0, description="Combustion emissions tCO2e")
    emission_intensity: float = Field(default=0.0, description="tCO2e per tonne product")
    energy_intensity: float = Field(default=0.0, description="MWh per tonne product")
    methodology_notes: str = Field(default="")


# ---------------------------------------------------------------------------
#  Phase 1: Data Collection
# ---------------------------------------------------------------------------

class DataCollectionPhase:
    """Validate and inventory all input data for manufacturing emissions."""

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Ingest and validate facility, production, and energy data."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            facilities = config.get("facility_data", [])
            production = config.get("production_volumes", [])
            energy = config.get("energy_data", [])

            outputs["facility_count"] = len(facilities)
            outputs["production_records"] = len(production)
            outputs["energy_records"] = len(energy)

            facility_ids = {
                f.get("facility_id", "") if isinstance(f, dict) else f.facility_id
                for f in facilities
            }
            outputs["facility_ids"] = list(facility_ids)

            total_production = sum(
                (p.get("quantity", 0.0) if isinstance(p, dict) else p.quantity)
                for p in production
            )
            outputs["total_production_tonnes"] = total_production

            total_energy = sum(
                (e.get("consumption", 0.0) if isinstance(e, dict) else e.consumption)
                for e in energy
            )
            outputs["total_energy_consumption"] = total_energy

            if not facilities:
                errors.append("No facility data provided")
            if not production:
                warnings.append("No production volume data; intensity metrics unavailable")
            if not energy:
                warnings.append("No energy data; Scope 2 calculations unavailable")

            outputs["validated"] = len(errors) == 0
            status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
            records = len(facilities) + len(production) + len(energy)

        except Exception as exc:
            logger.error("DataCollection failed: %s", exc, exc_info=True)
            errors.append(f"Data collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records,
        )


# ---------------------------------------------------------------------------
#  Phase 2: Process Calculation
# ---------------------------------------------------------------------------

class ProcessCalculationPhase:
    """Compute Scope 1 emissions from industrial processes and combustion."""

    PHASE_NAME = "process_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Calculate process and combustion Scope 1 emissions."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            production = config.get("production_volumes", [])
            pef = config.get("process_emission_factors", {})
            combustion_fuels = config.get("combustion_fuels", [])

            # Process emissions: production * factor
            process_total = 0.0
            facility_process: Dict[str, float] = {}
            for p in production:
                fid = p.get("facility_id", "") if isinstance(p, dict) else p.facility_id
                qty = p.get("quantity", 0.0) if isinstance(p, dict) else p.quantity
                factor = pef.get(fid, 0.0)
                em = qty * factor
                process_total += em
                facility_process[fid] = facility_process.get(fid, 0.0) + em

            outputs["process_emissions_total"] = round(process_total, 4)
            outputs["process_by_facility"] = {
                k: round(v, 4) for k, v in facility_process.items()
            }

            # Combustion emissions: fuel_qty * emission_factor
            combustion_total = 0.0
            facility_combustion: Dict[str, float] = {}
            for fuel in combustion_fuels:
                fid = fuel.get("facility_id", "")
                qty = fuel.get("quantity", 0.0)
                ef = fuel.get("emission_factor", 0.0)
                em = qty * ef
                combustion_total += em
                facility_combustion[fid] = facility_combustion.get(fid, 0.0) + em

            outputs["combustion_emissions_total"] = round(combustion_total, 4)
            outputs["combustion_by_facility"] = {
                k: round(v, 4) for k, v in facility_combustion.items()
            }
            outputs["scope1_total"] = round(process_total + combustion_total, 4)

            if process_total == 0.0 and combustion_total == 0.0:
                warnings.append("No Scope 1 emissions calculated; check input data")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ProcessCalculation failed: %s", exc, exc_info=True)
            errors.append(f"Process calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 3: Energy Analysis
# ---------------------------------------------------------------------------

class EnergyAnalysisPhase:
    """Compute Scope 2 emissions and energy intensity metrics."""

    PHASE_NAME = "energy_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Calculate Scope 2 emissions, energy mix, and intensity."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            energy = config.get("energy_data", [])
            collection = context.get_phase_output("data_collection")
            total_production = collection.get("total_production_tonnes", 0.0)

            scope2_total = 0.0
            scope1_energy = 0.0
            total_energy_mwh = 0.0
            by_source: Dict[str, float] = {}
            by_facility: Dict[str, float] = {}

            for rec in energy:
                fid = rec.get("facility_id", "") if isinstance(rec, dict) else rec.facility_id
                source = rec.get("source", "") if isinstance(rec, dict) else rec.source
                consumption = rec.get("consumption", 0.0) if isinstance(rec, dict) else rec.consumption
                ef = rec.get("emission_factor", 0.0) if isinstance(rec, dict) else rec.emission_factor
                scope = rec.get("scope", "scope2") if isinstance(rec, dict) else rec.scope

                emissions = consumption * ef / 1000.0  # kgCO2e -> tCO2e
                total_energy_mwh += consumption

                if scope == "scope1":
                    scope1_energy += emissions
                else:
                    scope2_total += emissions

                by_source[source] = by_source.get(source, 0.0) + consumption
                by_facility[fid] = by_facility.get(fid, 0.0) + emissions

            outputs["scope2_total"] = round(scope2_total, 4)
            outputs["scope1_energy"] = round(scope1_energy, 4)
            outputs["total_energy_mwh"] = round(total_energy_mwh, 2)
            outputs["by_source"] = {k: round(v, 2) for k, v in by_source.items()}
            outputs["by_facility"] = {k: round(v, 4) for k, v in by_facility.items()}

            if total_production > 0:
                outputs["energy_intensity"] = round(total_energy_mwh / total_production, 4)
            else:
                outputs["energy_intensity"] = 0.0
                warnings.append("Energy intensity unavailable: no production data")

            # Energy mix percentages
            if total_energy_mwh > 0:
                outputs["energy_mix_pct"] = {
                    k: round(v / total_energy_mwh * 100, 2)
                    for k, v in by_source.items()
                }
            else:
                outputs["energy_mix_pct"] = {}

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("EnergyAnalysis failed: %s", exc, exc_info=True)
            errors.append(f"Energy analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Phase 4: Consolidation
# ---------------------------------------------------------------------------

class ConsolidationPhase:
    """Aggregate scope totals, compute intensities, build provenance."""

    PHASE_NAME = "consolidation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Consolidate all scope emissions and compute intensity metrics."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            process = context.get_phase_output("process_calculation")
            energy = context.get_phase_output("energy_analysis")
            collection = context.get_phase_output("data_collection")

            scope1_process = process.get("process_emissions_total", 0.0)
            scope1_combustion = process.get("combustion_emissions_total", 0.0)
            scope1_energy = energy.get("scope1_energy", 0.0)
            scope1_total = scope1_process + scope1_combustion + scope1_energy

            scope2_total = energy.get("scope2_total", 0.0)
            scope3_up = config.get("scope3_upstream", 0.0)
            scope3_down = config.get("scope3_downstream", 0.0)
            scope3_total = scope3_up + scope3_down
            grand_total = scope1_total + scope2_total + scope3_total

            total_production = collection.get("total_production_tonnes", 0.0)
            emission_intensity = (
                round(grand_total / total_production, 4) if total_production > 0 else 0.0
            )
            energy_intensity = energy.get("energy_intensity", 0.0)

            outputs["scope1_total"] = round(scope1_total, 4)
            outputs["scope2_total"] = round(scope2_total, 4)
            outputs["scope3_total"] = round(scope3_total, 4)
            outputs["grand_total"] = round(grand_total, 4)
            outputs["process_emissions"] = round(scope1_process, 4)
            outputs["combustion_emissions"] = round(scope1_combustion, 4)
            outputs["emission_intensity"] = emission_intensity
            outputs["energy_intensity"] = energy_intensity
            outputs["energy_mix"] = energy.get("energy_mix_pct", {})
            outputs["methodology_notes"] = (
                f"Scope 1 = process ({scope1_process:.2f}) + combustion "
                f"({scope1_combustion:.2f}) + energy-scope1 ({scope1_energy:.2f}). "
                f"Scope 2 = {scope2_total:.2f}. "
                f"Scope 3 = upstream ({scope3_up:.2f}) + downstream ({scope3_down:.2f}). "
                f"Intensity = {emission_intensity:.4f} tCO2e/tonne."
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Consolidation failed: %s", exc, exc_info=True)
            errors.append(f"Consolidation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class ManufacturingEmissionsWorkflow:
    """
    Four-phase manufacturing emissions assessment workflow.

    Orchestrates comprehensive Scope 1/2/3 emissions calculation for
    manufacturing facilities, including process emissions, combustion,
    energy analysis, and consolidation with intensity metrics.
    """

    WORKFLOW_NAME = "manufacturing_emissions"

    PHASE_ORDER = [
        "data_collection", "process_calculation",
        "energy_analysis", "consolidation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize ManufacturingEmissionsWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_collection": DataCollectionPhase(),
            "process_calculation": ProcessCalculationPhase(),
            "energy_analysis": EnergyAnalysisPhase(),
            "consolidation": ConsolidationPhase(),
        }

    async def run(self, input_data: ManufacturingEmissionsInput) -> ManufacturingEmissionsResult:
        """Execute the complete 4-phase manufacturing emissions workflow."""
        started_at = _utcnow()
        logger.info(
            "Starting manufacturing emissions workflow %s org=%s year=%d",
            self.workflow_id, input_data.organization_id, input_data.reporting_year,
        )
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            self._notify_progress(
                phase_name, f"Starting: {phase_name}",
                idx / len(self.PHASE_ORDER),
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "data_collection":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=_utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = _utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return ManufacturingEmissionsResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            scope1_total=summary.get("scope1_total", 0.0),
            scope2_total=summary.get("scope2_total", 0.0),
            scope3_total=summary.get("scope3_total", 0.0),
            process_emissions=summary.get("process_emissions", 0.0),
            combustion_emissions=summary.get("combustion_emissions", 0.0),
            emission_intensity=summary.get("emission_intensity", 0.0),
            energy_intensity=summary.get("energy_intensity", 0.0),
            methodology_notes=summary.get("methodology_notes", ""),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from consolidated phase outputs."""
        consol = context.get_phase_output("consolidation")
        return {
            "scope1_total": consol.get("scope1_total", 0.0),
            "scope2_total": consol.get("scope2_total", 0.0),
            "scope3_total": consol.get("scope3_total", 0.0),
            "grand_total": consol.get("grand_total", 0.0),
            "process_emissions": consol.get("process_emissions", 0.0),
            "combustion_emissions": consol.get("combustion_emissions", 0.0),
            "emission_intensity": consol.get("emission_intensity", 0.0),
            "energy_intensity": consol.get("energy_intensity", 0.0),
            "methodology_notes": consol.get("methodology_notes", ""),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
