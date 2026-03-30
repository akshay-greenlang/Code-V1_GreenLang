# -*- coding: utf-8 -*-
"""
Product Carbon Footprint (PCF) Workflow
========================================

Five-phase workflow for calculating product carbon footprint across
the full lifecycle, supporting Digital Product Passport (DPP) data
generation and ISO 14067 compliance.

Phases:
    1. ProductSelection - Identify target products, validate BOM data
    2. BOMMapping - Map bill of materials to emission factors
    3. LifecycleAssessment - Calculate emissions per lifecycle stage
    4. Allocation - Allocate shared process emissions to products
    5. PCFGeneration - Produce PCF label data with DPP fields

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

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
    execution_timestamp: datetime = Field(default_factory=utcnow)
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

class BOMComponent(BaseModel):
    """Single bill-of-materials component."""
    component_id: str = Field(..., description="Component identifier")
    component_name: str = Field(default="")
    material: str = Field(default="", description="Material type")
    mass_kg: float = Field(default=0.0, ge=0.0, description="Mass in kilograms")
    emission_factor: float = Field(default=0.0, ge=0.0, description="kgCO2e per kg material")
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    supplier_country: str = Field(default="")

class ManufacturingProcess(BaseModel):
    """Manufacturing process step with energy and waste data."""
    process_id: str = Field(...)
    process_name: str = Field(default="")
    energy_kwh: float = Field(default=0.0, ge=0.0)
    energy_ef: float = Field(default=0.0, ge=0.0, description="kgCO2e per kWh")
    waste_kg: float = Field(default=0.0, ge=0.0)
    waste_ef: float = Field(default=0.0, ge=0.0, description="kgCO2e per kg waste")
    yield_rate: float = Field(default=1.0, gt=0.0, le=1.0)

class ProductPCFInput(BaseModel):
    """Input for product carbon footprint workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(default="")
    functional_unit: str = Field(default="1 unit", description="Functional unit for PCF")
    bom_components: List[BOMComponent] = Field(default_factory=list)
    manufacturing_processes: List[ManufacturingProcess] = Field(default_factory=list)
    distribution_km: float = Field(default=0.0, ge=0.0, description="Distribution distance km")
    distribution_mode: str = Field(default="road", description="Transport mode")
    distribution_ef: float = Field(default=0.0, ge=0.0, description="kgCO2e per tonne-km")
    use_phase_kwh: float = Field(default=0.0, ge=0.0, description="Energy in use phase")
    use_phase_ef: float = Field(default=0.0, ge=0.0, description="kgCO2e per kWh in use")
    use_phase_years: float = Field(default=1.0, ge=0.0)
    end_of_life_method: str = Field(default="landfill")
    end_of_life_ef: float = Field(default=0.0, ge=0.0, description="kgCO2e per kg at EoL")
    product_mass_kg: float = Field(default=0.0, ge=0.0)
    allocation_method: str = Field(default="mass", description="mass, economic, or energy")
    allocation_factor: float = Field(default=1.0, gt=0.0, le=1.0)
    skip_phases: List[str] = Field(default_factory=list)

class LifecycleBreakdown(BaseModel):
    """Emissions breakdown by lifecycle stage."""
    raw_materials: float = Field(default=0.0)
    manufacturing: float = Field(default=0.0)
    distribution: float = Field(default=0.0)
    use_phase: float = Field(default=0.0)
    end_of_life: float = Field(default=0.0)

class ProductPCFResult(WorkflowResult):
    """Result from the product PCF workflow."""
    total_pcf: float = Field(default=0.0, description="Total PCF in kgCO2e")
    lifecycle_breakdown: LifecycleBreakdown = Field(default_factory=LifecycleBreakdown)
    hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    dpp_data: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
#  Phase 1: Product Selection
# ---------------------------------------------------------------------------

class ProductSelectionPhase:
    """Identify target products and validate BOM data completeness."""

    PHASE_NAME = "product_selection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Validate product data and BOM completeness."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_id = config.get("product_id", "")
            bom = config.get("bom_components", [])
            processes = config.get("manufacturing_processes", [])

            outputs["product_id"] = product_id
            outputs["product_name"] = config.get("product_name", "")
            outputs["functional_unit"] = config.get("functional_unit", "1 unit")
            outputs["bom_component_count"] = len(bom)
            outputs["process_count"] = len(processes)

            total_mass = sum(c.get("mass_kg", 0.0) for c in bom)
            outputs["total_bom_mass_kg"] = total_mass

            missing_ef = sum(1 for c in bom if c.get("emission_factor", 0.0) == 0.0)
            if missing_ef > 0:
                warnings.append(f"{missing_ef} BOM components missing emission factors")
            outputs["missing_ef_count"] = missing_ef

            if not bom:
                errors.append("No BOM components provided")
            if not product_id:
                errors.append("Product ID is required")

            outputs["validated"] = len(errors) == 0
            status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED

        except Exception as exc:
            logger.error("ProductSelection failed: %s", exc, exc_info=True)
            errors.append(f"Product selection failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 2: BOM Mapping
# ---------------------------------------------------------------------------

class BOMMappingPhase:
    """Map bill of materials to emission factors and calculate raw material emissions."""

    PHASE_NAME = "bom_mapping"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Calculate raw material emissions from BOM components."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            bom = config.get("bom_components", [])

            raw_material_total = 0.0
            component_emissions: List[Dict[str, Any]] = []
            material_breakdown: Dict[str, float] = {}

            for comp in bom:
                mass = comp.get("mass_kg", 0.0)
                ef = comp.get("emission_factor", 0.0)
                recycled = comp.get("recycled_content_pct", 0.0)
                # Reduce EF by recycled content proportion (simplified)
                adjusted_ef = ef * (1.0 - recycled / 100.0 * 0.5)
                emissions = mass * adjusted_ef
                raw_material_total += emissions

                material = comp.get("material", "unknown")
                material_breakdown[material] = (
                    material_breakdown.get(material, 0.0) + emissions
                )
                component_emissions.append({
                    "component_id": comp.get("component_id", ""),
                    "component_name": comp.get("component_name", ""),
                    "material": material,
                    "mass_kg": mass,
                    "emission_factor": ef,
                    "adjusted_ef": round(adjusted_ef, 4),
                    "emissions_kgco2e": round(emissions, 4),
                    "recycled_content_pct": recycled,
                })

            outputs["raw_material_emissions"] = round(raw_material_total, 4)
            outputs["component_emissions"] = component_emissions
            outputs["material_breakdown"] = {
                k: round(v, 4) for k, v in material_breakdown.items()
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("BOMMapping failed: %s", exc, exc_info=True)
            errors.append(f"BOM mapping failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 3: Lifecycle Assessment
# ---------------------------------------------------------------------------

class LifecycleAssessmentPhase:
    """Calculate emissions per lifecycle stage (cradle-to-grave)."""

    PHASE_NAME = "lifecycle_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Compute manufacturing, distribution, use, and EoL emissions."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            bom_out = context.get_phase_output("bom_mapping")
            processes = config.get("manufacturing_processes", [])

            # Manufacturing emissions
            mfg_total = 0.0
            for proc in processes:
                energy_em = proc.get("energy_kwh", 0.0) * proc.get("energy_ef", 0.0)
                waste_em = proc.get("waste_kg", 0.0) * proc.get("waste_ef", 0.0)
                yield_rate = proc.get("yield_rate", 1.0)
                # Adjust for yield losses
                proc_em = (energy_em + waste_em) / max(yield_rate, 0.01)
                mfg_total += proc_em

            # Distribution emissions
            dist_km = config.get("distribution_km", 0.0)
            dist_ef = config.get("distribution_ef", 0.0)
            product_mass = config.get("product_mass_kg", 0.0)
            dist_total = dist_km * (product_mass / 1000.0) * dist_ef

            # Use phase emissions
            use_kwh = config.get("use_phase_kwh", 0.0)
            use_ef = config.get("use_phase_ef", 0.0)
            use_years = config.get("use_phase_years", 1.0)
            use_total = use_kwh * use_ef * use_years

            # End of life emissions
            eol_ef = config.get("end_of_life_ef", 0.0)
            eol_total = product_mass * eol_ef

            raw_materials = bom_out.get("raw_material_emissions", 0.0)

            outputs["raw_materials"] = round(raw_materials, 4)
            outputs["manufacturing"] = round(mfg_total, 4)
            outputs["distribution"] = round(dist_total, 4)
            outputs["use_phase"] = round(use_total, 4)
            outputs["end_of_life"] = round(eol_total, 4)
            outputs["total_pre_allocation"] = round(
                raw_materials + mfg_total + dist_total + use_total + eol_total, 4
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("LifecycleAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Lifecycle assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 4: Allocation
# ---------------------------------------------------------------------------

class AllocationPhase:
    """Allocate shared process emissions to individual products."""

    PHASE_NAME = "allocation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Apply allocation factor to lifecycle emissions."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            lca = context.get_phase_output("lifecycle_assessment")
            alloc_method = config.get("allocation_method", "mass")
            alloc_factor = config.get("allocation_factor", 1.0)

            raw = lca.get("raw_materials", 0.0) * alloc_factor
            mfg = lca.get("manufacturing", 0.0) * alloc_factor
            dist = lca.get("distribution", 0.0) * alloc_factor
            use = lca.get("use_phase", 0.0)  # Use phase is product-specific
            eol = lca.get("end_of_life", 0.0)  # EoL is product-specific

            total = raw + mfg + dist + use + eol

            outputs["allocation_method"] = alloc_method
            outputs["allocation_factor"] = alloc_factor
            outputs["allocated_raw_materials"] = round(raw, 4)
            outputs["allocated_manufacturing"] = round(mfg, 4)
            outputs["allocated_distribution"] = round(dist, 4)
            outputs["allocated_use_phase"] = round(use, 4)
            outputs["allocated_end_of_life"] = round(eol, 4)
            outputs["total_pcf"] = round(total, 4)

            if alloc_factor < 1.0:
                warnings.append(
                    f"Allocation factor {alloc_factor} applied ({alloc_method} method)"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Allocation failed: %s", exc, exc_info=True)
            errors.append(f"Allocation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 5: PCF Generation
# ---------------------------------------------------------------------------

class PCFGenerationPhase:
    """Produce PCF label data with DPP fields and hotspot analysis."""

    PHASE_NAME = "pcf_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Generate PCF label, DPP data, and identify emission hotspots."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            alloc = context.get_phase_output("allocation")
            bom_out = context.get_phase_output("bom_mapping")
            selection = context.get_phase_output("product_selection")

            total_pcf = alloc.get("total_pcf", 0.0)
            breakdown = {
                "raw_materials": alloc.get("allocated_raw_materials", 0.0),
                "manufacturing": alloc.get("allocated_manufacturing", 0.0),
                "distribution": alloc.get("allocated_distribution", 0.0),
                "use_phase": alloc.get("allocated_use_phase", 0.0),
                "end_of_life": alloc.get("allocated_end_of_life", 0.0),
            }

            # Identify hotspots (stages contributing >25%)
            hotspots = []
            for stage, value in breakdown.items():
                pct = (value / total_pcf * 100) if total_pcf > 0 else 0.0
                if pct > 25.0:
                    hotspots.append({
                        "stage": stage,
                        "emissions_kgco2e": round(value, 4),
                        "percentage": round(pct, 2),
                    })

            # Component-level hotspots
            comp_emissions = bom_out.get("component_emissions", [])
            sorted_comps = sorted(
                comp_emissions,
                key=lambda c: c.get("emissions_kgco2e", 0.0),
                reverse=True,
            )
            top_components = sorted_comps[:5]

            # DPP data
            dpp_data = {
                "product_id": config.get("product_id", ""),
                "product_name": config.get("product_name", ""),
                "functional_unit": config.get("functional_unit", "1 unit"),
                "pcf_kgco2e": round(total_pcf, 4),
                "lifecycle_breakdown": breakdown,
                "methodology": "ISO 14067:2018",
                "allocation_method": alloc.get("allocation_method", "mass"),
                "data_quality_rating": self._assess_quality(config, bom_out),
                "generated_at": utcnow().isoformat(),
            }

            # Data quality assessment
            data_quality = self._build_quality_assessment(config, bom_out)

            outputs["total_pcf"] = round(total_pcf, 4)
            outputs["lifecycle_breakdown"] = breakdown
            outputs["hotspots"] = hotspots
            outputs["top_components"] = top_components
            outputs["dpp_data"] = dpp_data
            outputs["data_quality"] = data_quality

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PCFGeneration failed: %s", exc, exc_info=True)
            errors.append(f"PCF generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _assess_quality(self, config: Dict[str, Any], bom_out: Dict[str, Any]) -> str:
        """Assess overall data quality rating."""
        missing = bom_out.get("missing_ef_count", 0) if "missing_ef_count" in bom_out else 0
        comp_count = len(config.get("bom_components", []))
        if comp_count == 0:
            return "LOW"
        ratio = missing / comp_count
        if ratio == 0:
            return "HIGH"
        if ratio < 0.2:
            return "MEDIUM"
        return "LOW"

    def _build_quality_assessment(
        self, config: Dict[str, Any], bom_out: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build detailed data quality assessment."""
        bom = config.get("bom_components", [])
        with_ef = sum(1 for c in bom if c.get("emission_factor", 0.0) > 0)
        return {
            "bom_coverage": round(with_ef / max(len(bom), 1) * 100, 1),
            "primary_data_pct": round(with_ef / max(len(bom), 1) * 100, 1),
            "secondary_data_pct": round((len(bom) - with_ef) / max(len(bom), 1) * 100, 1),
            "overall_rating": self._assess_quality(config, bom_out),
        }

# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class ProductPCFWorkflow:
    """
    Five-phase product carbon footprint workflow.

    Calculates cradle-to-grave PCF for manufactured products with BOM
    mapping, lifecycle assessment, allocation, and DPP data generation.
    Follows ISO 14067 methodology.
    """

    WORKFLOW_NAME = "product_pcf"

    PHASE_ORDER = [
        "product_selection", "bom_mapping", "lifecycle_assessment",
        "allocation", "pcf_generation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize ProductPCFWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "product_selection": ProductSelectionPhase(),
            "bom_mapping": BOMMappingPhase(),
            "lifecycle_assessment": LifecycleAssessmentPhase(),
            "allocation": AllocationPhase(),
            "pcf_generation": PCFGenerationPhase(),
        }

    async def run(self, input_data: ProductPCFInput) -> ProductPCFResult:
        """Execute the complete 5-phase product PCF workflow."""
        started_at = utcnow()
        logger.info(
            "Starting product PCF workflow %s org=%s product=%s",
            self.workflow_id, input_data.organization_id, input_data.product_id,
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
                    if phase_name == "product_selection":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
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

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return ProductPCFResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            total_pcf=summary.get("total_pcf", 0.0),
            lifecycle_breakdown=LifecycleBreakdown(**summary.get("lifecycle_breakdown", {})),
            hotspots=summary.get("hotspots", []),
            dpp_data=summary.get("dpp_data", {}),
            data_quality=summary.get("data_quality", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from PCF generation phase outputs."""
        pcf = context.get_phase_output("pcf_generation")
        return {
            "total_pcf": pcf.get("total_pcf", 0.0),
            "lifecycle_breakdown": pcf.get("lifecycle_breakdown", {}),
            "hotspots": pcf.get("hotspots", []),
            "dpp_data": pcf.get("dpp_data", {}),
            "data_quality": pcf.get("data_quality", {}),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
