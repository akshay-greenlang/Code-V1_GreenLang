# -*- coding: utf-8 -*-
"""
Supply Chain Assessment Workflow
=================================

Five-phase workflow for Scope 3 supply chain emissions assessment,
hotspot identification, and supplier engagement planning.

Phases:
    1. SupplierInventory - Catalogue suppliers and map to Scope 3 categories
    2. DataCollection - Gather spend, activity, and supplier-specific data
    3. EmissionCalculation - Calculate Scope 3 by category (spend / hybrid)
    4. HotspotAnalysis - Identify top emitting suppliers and categories
    5. EngagementPlanning - Build supplier engagement and reduction plan

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

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


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
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
#  Input / Result
# ---------------------------------------------------------------------------

class SupplierRecord(BaseModel):
    """Supplier information record."""
    supplier_id: str = Field(...)
    supplier_name: str = Field(default="")
    country: str = Field(default="")
    sector: str = Field(default="")
    scope3_category: int = Field(default=1, ge=1, le=15, description="GHG Protocol Scope 3 category")
    spend_eur: float = Field(default=0.0, ge=0.0)
    emission_factor: float = Field(default=0.0, ge=0.0, description="kgCO2e per EUR spend")
    supplier_reported_emissions: Optional[float] = Field(None, ge=0.0, description="tCO2e if reported")
    data_quality: str = Field(default="estimated", description="primary, secondary, estimated")
    tier: int = Field(default=1, ge=1, description="Supply chain tier")


class TransportRecord(BaseModel):
    """Transport data for upstream logistics."""
    transport_id: str = Field(...)
    mode: str = Field(default="road", description="road, rail, sea, air")
    distance_km: float = Field(default=0.0, ge=0.0)
    weight_tonnes: float = Field(default=0.0, ge=0.0)
    emission_factor: float = Field(default=0.0, ge=0.0, description="kgCO2e per tonne-km")
    supplier_id: str = Field(default="")


class SupplyChainInput(BaseModel):
    """Input for supply chain assessment workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2000, le=2100)
    suppliers: List[SupplierRecord] = Field(default_factory=list)
    bom_data: List[Dict[str, Any]] = Field(default_factory=list)
    transport_data: List[TransportRecord] = Field(default_factory=list)
    priority_categories: List[int] = Field(
        default_factory=lambda: [1, 4, 6, 7],
        description="Scope 3 categories to prioritize",
    )
    skip_phases: List[str] = Field(default_factory=list)


class SupplyChainResult(WorkflowResult):
    """Result from the supply chain assessment workflow."""
    total_scope3: float = Field(default=0.0, description="Total Scope 3 tCO2e")
    category_breakdown: Dict[str, float] = Field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    engagement_plan: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Phase 1: Supplier Inventory
# ---------------------------------------------------------------------------

class SupplierInventoryPhase:
    """Catalogue suppliers and map to Scope 3 categories."""

    PHASE_NAME = "supplier_inventory"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Build supplier inventory with category mapping."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            suppliers = config.get("suppliers", [])

            outputs["total_suppliers"] = len(suppliers)
            total_spend = sum(s.get("spend_eur", 0.0) for s in suppliers)
            outputs["total_spend_eur"] = total_spend

            # Group by category
            by_category: Dict[int, List[Dict[str, Any]]] = {}
            for s in suppliers:
                cat = s.get("scope3_category", 1)
                by_category.setdefault(cat, []).append(s)

            outputs["categories_covered"] = sorted(by_category.keys())
            outputs["suppliers_by_category"] = {
                str(cat): len(sups) for cat, sups in by_category.items()
            }
            outputs["spend_by_category"] = {
                str(cat): round(sum(s.get("spend_eur", 0.0) for s in sups), 2)
                for cat, sups in by_category.items()
            }

            # Data quality distribution
            quality_dist = {"primary": 0, "secondary": 0, "estimated": 0}
            for s in suppliers:
                dq = s.get("data_quality", "estimated")
                quality_dist[dq] = quality_dist.get(dq, 0) + 1
            outputs["data_quality_distribution"] = quality_dist

            if not suppliers:
                errors.append("No supplier data provided")

            status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED

        except Exception as exc:
            logger.error("SupplierInventory failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 2: Data Collection
# ---------------------------------------------------------------------------

class SupplyChainDataCollectionPhase:
    """Gather spend, activity, and supplier-specific emissions data."""

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Validate and consolidate supply chain data sources."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            suppliers = config.get("suppliers", [])
            transport = config.get("transport_data", [])

            with_primary = sum(
                1 for s in suppliers if s.get("supplier_reported_emissions") is not None
            )
            outputs["suppliers_with_primary_data"] = with_primary
            outputs["suppliers_spend_only"] = len(suppliers) - with_primary
            outputs["transport_records"] = len(transport)

            total_transport_emissions = 0.0
            for t in transport:
                em = t.get("distance_km", 0.0) * t.get("weight_tonnes", 0.0) * t.get("emission_factor", 0.0) / 1000.0
                total_transport_emissions += em

            outputs["transport_emissions_tco2e"] = round(total_transport_emissions, 4)

            primary_pct = round(with_primary / max(len(suppliers), 1) * 100, 2)
            outputs["primary_data_coverage_pct"] = primary_pct

            if primary_pct < 20:
                warnings.append(f"Only {primary_pct}% of suppliers have primary emissions data")

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("DataCollection failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 3: Emission Calculation
# ---------------------------------------------------------------------------

class EmissionCalculationPhase:
    """Calculate Scope 3 emissions by category using spend-based or hybrid method."""

    PHASE_NAME = "emission_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Compute emissions per supplier and aggregate by category."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            suppliers = config.get("suppliers", [])
            data_coll = context.get_phase_output("data_collection")
            transport_em = data_coll.get("transport_emissions_tco2e", 0.0)

            total_scope3 = 0.0
            by_category: Dict[str, float] = {}
            supplier_emissions: List[Dict[str, Any]] = []

            for s in suppliers:
                reported = s.get("supplier_reported_emissions")
                if reported is not None and reported > 0:
                    em = reported
                    method = "supplier_specific"
                else:
                    spend = s.get("spend_eur", 0.0)
                    ef = s.get("emission_factor", 0.0)
                    em = spend * ef / 1000.0  # kgCO2e -> tCO2e
                    method = "spend_based"

                total_scope3 += em
                cat_key = f"cat_{s.get('scope3_category', 1)}"
                by_category[cat_key] = by_category.get(cat_key, 0.0) + em

                supplier_emissions.append({
                    "supplier_id": s.get("supplier_id", ""),
                    "supplier_name": s.get("supplier_name", ""),
                    "scope3_category": s.get("scope3_category", 1),
                    "emissions_tco2e": round(em, 4),
                    "method": method,
                    "data_quality": s.get("data_quality", "estimated"),
                })

            # Add transport emissions to category 4
            by_category["cat_4"] = by_category.get("cat_4", 0.0) + transport_em
            total_scope3 += transport_em

            outputs["total_scope3"] = round(total_scope3, 4)
            outputs["category_breakdown"] = {k: round(v, 4) for k, v in by_category.items()}
            outputs["supplier_emissions"] = supplier_emissions

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("EmissionCalculation failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 4: Hotspot Analysis
# ---------------------------------------------------------------------------

class HotspotAnalysisPhase:
    """Identify top emitting suppliers and categories."""

    PHASE_NAME = "hotspot_analysis"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Rank suppliers and categories by emission contribution."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            calc = context.get_phase_output("emission_calculation")
            supplier_em = calc.get("supplier_emissions", [])
            total = calc.get("total_scope3", 0.0)
            category_bd = calc.get("category_breakdown", {})

            # Top 10 suppliers by emissions
            sorted_suppliers = sorted(
                supplier_em, key=lambda x: x.get("emissions_tco2e", 0.0), reverse=True
            )
            top_suppliers = []
            for s in sorted_suppliers[:10]:
                em = s.get("emissions_tco2e", 0.0)
                top_suppliers.append({
                    **s,
                    "pct_of_total": round(em / max(total, 0.001) * 100, 2),
                })

            # Top categories
            sorted_cats = sorted(category_bd.items(), key=lambda x: x[1], reverse=True)
            top_categories = [
                {
                    "category": cat,
                    "emissions_tco2e": round(em, 4),
                    "pct_of_total": round(em / max(total, 0.001) * 100, 2),
                }
                for cat, em in sorted_cats[:5]
            ]

            # Concentration: top 20% of suppliers by emissions
            top_20_pct = int(max(len(sorted_suppliers) * 0.2, 1))
            top_20_em = sum(s.get("emissions_tco2e", 0.0) for s in sorted_suppliers[:top_20_pct])
            concentration = round(top_20_em / max(total, 0.001) * 100, 2)

            outputs["hotspot_suppliers"] = top_suppliers
            outputs["hotspot_categories"] = top_categories
            outputs["concentration_pct"] = concentration
            outputs["total_scope3"] = total

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("HotspotAnalysis failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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
#  Phase 5: Engagement Planning
# ---------------------------------------------------------------------------

class EngagementPlanningPhase:
    """Build supplier engagement and emission reduction plan."""

    PHASE_NAME = "engagement_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Generate targeted engagement plan for top emitting suppliers."""
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            hotspot = context.get_phase_output("hotspot_analysis")
            inv = context.get_phase_output("supplier_inventory")
            top_suppliers = hotspot.get("hotspot_suppliers", [])
            quality_dist = inv.get("data_quality_distribution", {})

            engagement_actions: List[Dict[str, Any]] = []

            for idx, sup in enumerate(top_suppliers):
                dq = sup.get("data_quality", "estimated")
                if dq == "estimated":
                    action = "request_primary_data"
                    timeline = "Q1"
                elif dq == "secondary":
                    action = "verify_and_improve"
                    timeline = "Q2"
                else:
                    action = "collaborate_on_reduction"
                    timeline = "Q3-Q4"

                engagement_actions.append({
                    "priority": idx + 1,
                    "supplier_id": sup.get("supplier_id", ""),
                    "supplier_name": sup.get("supplier_name", ""),
                    "current_emissions_tco2e": sup.get("emissions_tco2e", 0.0),
                    "pct_of_total": sup.get("pct_of_total", 0.0),
                    "data_quality": dq,
                    "recommended_action": action,
                    "target_timeline": timeline,
                })

            # Data quality improvement targets
            estimated_count = quality_dist.get("estimated", 0)
            total_suppliers = sum(quality_dist.values())
            data_quality_target = {
                "current_primary_pct": round(
                    quality_dist.get("primary", 0) / max(total_suppliers, 1) * 100, 1
                ),
                "target_primary_pct": 50.0,
                "current_estimated_pct": round(
                    estimated_count / max(total_suppliers, 1) * 100, 1
                ),
                "improvement_needed": estimated_count,
            }

            outputs["engagement_plan"] = engagement_actions
            outputs["data_quality_target"] = data_quality_target
            outputs["engagement_count"] = len(engagement_actions)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("EngagementPlanning failed: %s", exc, exc_info=True)
            errors.append(str(exc))
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

class SupplyChainAssessmentWorkflow:
    """
    Five-phase supply chain emissions assessment workflow.

    Calculates Scope 3 emissions across supplier categories, identifies
    emission hotspots, and generates supplier engagement plans.
    """

    WORKFLOW_NAME = "supply_chain_assessment"

    PHASE_ORDER = [
        "supplier_inventory", "data_collection", "emission_calculation",
        "hotspot_analysis", "engagement_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize SupplyChainAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "supplier_inventory": SupplierInventoryPhase(),
            "data_collection": SupplyChainDataCollectionPhase(),
            "emission_calculation": EmissionCalculationPhase(),
            "hotspot_analysis": HotspotAnalysisPhase(),
            "engagement_planning": EngagementPlanningPhase(),
        }

    async def run(self, input_data: SupplyChainInput) -> SupplyChainResult:
        """Execute the complete 5-phase supply chain assessment workflow."""
        started_at = _utcnow()
        logger.info("Starting supply chain assessment workflow %s", self.workflow_id)
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

            self._notify_progress(phase_name, f"Starting: {phase_name}", idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "supplier_inventory":
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

        return SupplyChainResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            total_scope3=summary.get("total_scope3", 0.0),
            category_breakdown=summary.get("category_breakdown", {}),
            hotspots=summary.get("hotspots", []),
            engagement_plan=summary.get("engagement_plan", []),
            data_quality=summary.get("data_quality", {}),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build summary from phase outputs."""
        calc = context.get_phase_output("emission_calculation")
        hotspot = context.get_phase_output("hotspot_analysis")
        engage = context.get_phase_output("engagement_planning")
        inv = context.get_phase_output("supplier_inventory")
        return {
            "total_scope3": calc.get("total_scope3", 0.0),
            "category_breakdown": calc.get("category_breakdown", {}),
            "hotspots": hotspot.get("hotspot_suppliers", []),
            "engagement_plan": engage.get("engagement_plan", []),
            "data_quality": inv.get("data_quality_distribution", {}),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if configured."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
