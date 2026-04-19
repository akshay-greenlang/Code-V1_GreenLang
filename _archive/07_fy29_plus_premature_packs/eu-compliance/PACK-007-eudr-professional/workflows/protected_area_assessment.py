# -*- coding: utf-8 -*-
"""
Protected Area Assessment Workflow
====================================

Three-phase workflow for screening supplier plots against protected area
boundaries (national parks, UNESCO sites, indigenous territories, etc.).

This workflow enables:
- GIS overlay analysis with protected area databases
- Risk amplification for plots in/near protected zones
- Targeted mitigation planning

Phases:
    1. Overlay Analysis - Match plot coordinates to protected area boundaries
    2. Risk Amplification - Increase risk scores for protected area exposure
    3. Mitigation Planning - Generate targeted actions for affected plots

Regulatory Context:
    EUDR recitals reference the need to protect biodiversity and indigenous rights.
    Production in protected areas or indigenous territories significantly increases
    deforestation risk and legal compliance risk.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    OVERLAY_ANALYSIS = "overlay_analysis"
    RISK_AMPLIFICATION = "risk_amplification"
    MITIGATION_PLANNING = "mitigation_planning"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProtectedAreaType(str, Enum):
    """Types of protected areas."""
    NATIONAL_PARK = "national_park"
    UNESCO_SITE = "unesco_site"
    INDIGENOUS_TERRITORY = "indigenous_territory"
    BIODIVERSITY_HOTSPOT = "biodiversity_hotspot"
    RAMSAR_WETLAND = "ramsar_wetland"
    FOREST_RESERVE = "forest_reserve"


class ProximityLevel(str, Enum):
    """Proximity to protected area."""
    INSIDE = "inside"
    BUFFER_1KM = "buffer_1km"
    BUFFER_5KM = "buffer_5km"
    BUFFER_10KM = "buffer_10km"
    OUTSIDE = "outside"


# =============================================================================
# DATA MODELS
# =============================================================================


class ProtectedAreaAssessmentConfig(BaseModel):
    """Configuration for protected area assessment workflow."""
    protected_area_databases: List[str] = Field(
        default_factory=lambda: ["WDPA", "UNESCO", "ILO_169"],
        description="Protected area data sources",
    )
    buffer_distance_km: float = Field(default=10.0, ge=0.0, description="Buffer zone distance")
    risk_multiplier_inside: float = Field(default=2.0, ge=1.0, description="Risk multiplier for inside PA")
    risk_multiplier_buffer: float = Field(default=1.5, ge=1.0, description="Risk multiplier for buffer zone")
    operator_id: Optional[str] = Field(None, description="Operator context")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: ProtectedAreaAssessmentConfig = Field(default_factory=ProtectedAreaAssessmentConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the protected area assessment workflow."""
    workflow_name: str = Field(default="protected_area_assessment", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    plots_assessed: int = Field(default=0, ge=0, description="Plots analyzed")
    plots_in_protected_areas: int = Field(default=0, ge=0, description="Plots inside PAs")
    plots_in_buffer_zones: int = Field(default=0, ge=0, description="Plots in buffer zones")
    high_risk_plots: int = Field(default=0, ge=0, description="High-risk plots identified")
    mitigation_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# PROTECTED AREA ASSESSMENT WORKFLOW
# =============================================================================


class ProtectedAreaAssessmentWorkflow:
    """
    Three-phase protected area assessment workflow.

    Performs GIS-based screening of supplier plots against protected areas:
    - Spatial overlay analysis with global protected area databases
    - Risk score amplification for plots in/near protected zones
    - Targeted mitigation planning for high-risk plots

    Example:
        >>> config = ProtectedAreaAssessmentConfig(
        ...     buffer_distance_km=10.0,
        ...     risk_multiplier_inside=2.0,
        ... )
        >>> workflow = ProtectedAreaAssessmentWorkflow(config)
        >>> result = await workflow.run(
        ...     WorkflowContext(config=config, state={"plots": plot_data})
        ... )
        >>> assert result.overall_status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[ProtectedAreaAssessmentConfig] = None) -> None:
        """Initialize the protected area assessment workflow."""
        self.config = config or ProtectedAreaAssessmentConfig()
        self.logger = logging.getLogger(f"{__name__}.ProtectedAreaAssessmentWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 3-phase protected area assessment workflow.

        Args:
            context: Workflow context with configuration and plot data.

        Returns:
            WorkflowResult with overlay analysis, risk amplification, and mitigation.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting protected area assessment workflow execution_id=%s buffer=%dkm",
            context.execution_id,
            self.config.buffer_distance_km,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.OVERLAY_ANALYSIS, self._phase_1_overlay_analysis),
            (Phase.RISK_AMPLIFICATION, self._phase_2_risk_amplification),
            (Phase.MITIGATION_PLANNING, self._phase_3_mitigation_planning),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        plots_assessed = context.state.get("plots_assessed", 0)
        plots_inside = context.state.get("plots_in_protected_areas", 0)
        plots_buffer = context.state.get("plots_in_buffer_zones", 0)
        high_risk = context.state.get("high_risk_plots", 0)
        mitigation_actions = context.state.get("mitigation_actions", [])

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "plots_assessed": plots_assessed,
        })

        self.logger.info(
            "Protected area assessment finished execution_id=%s status=%s "
            "assessed=%d inside_pa=%d buffer=%d",
            context.execution_id,
            overall_status.value,
            plots_assessed,
            plots_inside,
            plots_buffer,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            plots_assessed=plots_assessed,
            plots_in_protected_areas=plots_inside,
            plots_in_buffer_zones=plots_buffer,
            high_risk_plots=high_risk,
            mitigation_actions=mitigation_actions,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Overlay Analysis
    # -------------------------------------------------------------------------

    async def _phase_1_overlay_analysis(self, context: WorkflowContext) -> PhaseResult:
        """
        Match plot coordinates to protected area boundaries.

        GIS operations:
        - Load plot geometries (points or polygons)
        - Query protected area databases (WDPA, UNESCO, ILO 169)
        - Perform spatial intersection/proximity analysis
        - Classify proximity level (inside, 1km buffer, 5km buffer, 10km buffer)
        - Record protected area metadata (name, type, designation year)
        """
        phase = Phase.OVERLAY_ANALYSIS
        plots = context.state.get("plots", self._generate_sample_plots())

        self.logger.info("Performing GIS overlay analysis for %d plots", len(plots))

        await asyncio.sleep(0.1)

        # Simulate GIS overlay analysis
        plot_assessments = []

        for plot in plots:
            # Simulate spatial query (in production, use PostGIS/GeoPandas)
            proximity, pa_details = self._simulate_spatial_query(
                plot.get("latitude", 0),
                plot.get("longitude", 0),
            )

            assessment = {
                "plot_id": plot.get("plot_id", f"PLOT-{uuid.uuid4().hex[:8]}"),
                "latitude": plot.get("latitude", 0),
                "longitude": plot.get("longitude", 0),
                "proximity_level": proximity.value,
                "protected_areas": pa_details,
                "inside_protected_area": proximity == ProximityLevel.INSIDE,
                "in_buffer_zone": proximity in (ProximityLevel.BUFFER_1KM, ProximityLevel.BUFFER_5KM, ProximityLevel.BUFFER_10KM),
            }
            plot_assessments.append(assessment)

        context.state["plot_assessments"] = plot_assessments
        context.state["plots_assessed"] = len(plot_assessments)
        context.state["plots_in_protected_areas"] = len([a for a in plot_assessments if a["inside_protected_area"]])
        context.state["plots_in_buffer_zones"] = len([a for a in plot_assessments if a["in_buffer_zone"]])

        # Group by proximity level
        by_proximity = {}
        for assessment in plot_assessments:
            prox = assessment["proximity_level"]
            by_proximity[prox] = by_proximity.get(prox, 0) + 1

        provenance = self._hash({
            "phase": phase.value,
            "plots_assessed": len(plot_assessments),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "plots_assessed": len(plot_assessments),
                "plots_inside_pa": context.state["plots_in_protected_areas"],
                "plots_in_buffer": context.state["plots_in_buffer_zones"],
                "by_proximity": by_proximity,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Risk Amplification
    # -------------------------------------------------------------------------

    async def _phase_2_risk_amplification(self, context: WorkflowContext) -> PhaseResult:
        """
        Increase risk scores for plots in/near protected areas.

        Risk amplification logic:
        - Plots INSIDE protected areas: Risk * multiplier_inside (default 2.0x)
        - Plots in BUFFER zones: Risk * multiplier_buffer (default 1.5x)
        - Plots OUTSIDE: No change

        Special considerations:
        - Indigenous territories: Additional multiplier (1.2x)
        - UNESCO sites: Additional multiplier (1.2x)
        """
        phase = Phase.RISK_AMPLIFICATION
        plot_assessments = context.state.get("plot_assessments", [])

        self.logger.info("Amplifying risk scores for %d plots", len(plot_assessments))

        high_risk_plots = []

        for assessment in plot_assessments:
            proximity = assessment["proximity_level"]
            protected_areas = assessment.get("protected_areas", [])

            # Base risk score (simulated; in production, retrieve from database)
            base_risk = random.uniform(30, 70)

            # Apply proximity multiplier
            if proximity == ProximityLevel.INSIDE.value:
                amplified_risk = base_risk * self.config.risk_multiplier_inside
            elif proximity in (ProximityLevel.BUFFER_1KM.value, ProximityLevel.BUFFER_5KM.value, ProximityLevel.BUFFER_10KM.value):
                amplified_risk = base_risk * self.config.risk_multiplier_buffer
            else:
                amplified_risk = base_risk

            # Apply special multipliers
            for pa in protected_areas:
                pa_type = pa.get("type", "")
                if pa_type == ProtectedAreaType.INDIGENOUS_TERRITORY.value:
                    amplified_risk *= 1.2
                elif pa_type == ProtectedAreaType.UNESCO_SITE.value:
                    amplified_risk *= 1.2

            amplified_risk = min(100.0, amplified_risk)

            # Track high-risk plots (amplified risk >= 70)
            if amplified_risk >= 70.0:
                high_risk_plots.append({
                    "plot_id": assessment["plot_id"],
                    "base_risk": round(base_risk, 1),
                    "amplified_risk": round(amplified_risk, 1),
                    "proximity": proximity,
                    "protected_areas": [pa.get("name", "") for pa in protected_areas],
                })

            assessment["base_risk"] = round(base_risk, 1)
            assessment["amplified_risk"] = round(amplified_risk, 1)

        context.state["plot_assessments"] = plot_assessments
        context.state["high_risk_plots"] = len(high_risk_plots)
        context.state["high_risk_plot_details"] = high_risk_plots

        provenance = self._hash({
            "phase": phase.value,
            "high_risk_count": len(high_risk_plots),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "plots_with_amplified_risk": len([a for a in plot_assessments if a.get("amplified_risk", 0) > a.get("base_risk", 0)]),
                "high_risk_plots": len(high_risk_plots),
                "avg_amplification": round(
                    sum(a.get("amplified_risk", 0) / max(a.get("base_risk", 1), 1) for a in plot_assessments) / max(len(plot_assessments), 1),
                    2,
                ),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Mitigation Planning
    # -------------------------------------------------------------------------

    async def _phase_3_mitigation_planning(self, context: WorkflowContext) -> PhaseResult:
        """
        Generate targeted actions for high-risk plots.

        Mitigation actions:
        - Plots INSIDE protected areas: Require legal verification, consider exclusion
        - Plots in BUFFER zones: Enhanced monitoring, satellite imagery analysis
        - Indigenous territories: FPIC (Free, Prior, Informed Consent) documentation
        - UNESCO sites: Additional certifications, third-party audits
        """
        phase = Phase.MITIGATION_PLANNING
        high_risk_plots = context.state.get("high_risk_plot_details", [])
        plots_inside = context.state.get("plots_in_protected_areas", 0)
        plots_buffer = context.state.get("plots_in_buffer_zones", 0)

        self.logger.info("Planning mitigation for %d high-risk plots", len(high_risk_plots))

        actions = []

        # Actions for plots inside protected areas
        if plots_inside > 0:
            actions.append(
                f"CRITICAL: {plots_inside} plot(s) located INSIDE protected areas. "
                "Conduct legal review to verify production rights. Consider excluding "
                "plots from supply chain if legal status unclear."
            )
            actions.append(
                "Request land title documentation and environmental permits for all "
                "plots inside protected areas. Verify production predates 2020-12-31."
            )

        # Actions for plots in buffer zones
        if plots_buffer > 0:
            actions.append(
                f"HIGH: {plots_buffer} plot(s) in buffer zones (within {self.config.buffer_distance_km}km of PA). "
                "Implement enhanced monitoring with quarterly satellite imagery review."
            )

        # Special actions for specific protected area types
        indigenous_plots = [
            p for p in high_risk_plots
            if any("indigenous" in pa.lower() for pa in p.get("protected_areas", []))
        ]
        if indigenous_plots:
            actions.append(
                f"{len(indigenous_plots)} plot(s) overlap indigenous territories. "
                "Request FPIC (Free, Prior, Informed Consent) documentation from suppliers."
            )

        unesco_plots = [
            p for p in high_risk_plots
            if any("unesco" in pa.lower() for pa in p.get("protected_areas", []))
        ]
        if unesco_plots:
            actions.append(
                f"{len(unesco_plots)} plot(s) near UNESCO World Heritage Sites. "
                "Require third-party audit certification (FSC, PEFC) for continued sourcing."
            )

        # General high-risk mitigation
        if len(high_risk_plots) > 0:
            actions.append(
                f"Schedule on-site verification visits for top {min(5, len(high_risk_plots))} "
                "highest-risk plots within next 90 days."
            )
            actions.append(
                "Implement continuous deforestation monitoring using GLAD/RADD alerts "
                "for all plots in/near protected areas."
            )

        if not actions:
            actions.append(
                "No high-risk plots identified. Continue standard monitoring protocols."
            )

        context.state["mitigation_actions"] = actions

        provenance = self._hash({
            "phase": phase.value,
            "action_count": len(actions),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "mitigation_actions": actions,
                "action_count": len(actions),
                "plots_requiring_fpic": len(indigenous_plots),
                "plots_requiring_certification": len(unesco_plots),
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_sample_plots(self) -> List[Dict[str, Any]]:
        """Generate sample plot data for testing."""
        plot_count = random.randint(50, 200)
        plots = []

        for i in range(plot_count):
            plots.append({
                "plot_id": f"PLOT-{uuid.uuid4().hex[:8]}",
                "latitude": random.uniform(-30, 10),  # Tropical latitudes
                "longitude": random.uniform(-80, 120),  # Americas and Asia
                "area_hectares": random.uniform(1, 100),
            })

        return plots

    def _simulate_spatial_query(
        self, latitude: float, longitude: float
    ) -> Tuple[ProximityLevel, List[Dict[str, Any]]]:
        """
        Simulate spatial query against protected area database.

        In production, this would use PostGIS or GeoPandas:
        ```sql
        SELECT pa.*, ST_Distance(plot_geom, pa.geom) AS distance
        FROM protected_areas pa
        WHERE ST_DWithin(plot_geom, pa.geom, 10000)  -- 10km buffer
        ORDER BY distance;
        ```
        """
        # Simulate random proximity (weighted towards outside)
        rand_val = random.random()

        if rand_val < 0.05:  # 5% inside
            proximity = ProximityLevel.INSIDE
            pa_details = [{
                "pa_id": f"PA-{uuid.uuid4().hex[:8]}",
                "name": random.choice([
                    "Amazon National Park",
                    "Yasuni UNESCO Site",
                    "Kayapo Indigenous Territory",
                ]),
                "type": random.choice([
                    ProtectedAreaType.NATIONAL_PARK.value,
                    ProtectedAreaType.UNESCO_SITE.value,
                    ProtectedAreaType.INDIGENOUS_TERRITORY.value,
                ]),
                "designation_year": random.randint(1980, 2020),
            }]
        elif rand_val < 0.15:  # 10% in 1km buffer
            proximity = ProximityLevel.BUFFER_1KM
            pa_details = [{
                "pa_id": f"PA-{uuid.uuid4().hex[:8]}",
                "name": "Protected Forest Reserve",
                "type": ProtectedAreaType.FOREST_RESERVE.value,
                "distance_km": random.uniform(0.1, 1.0),
            }]
        elif rand_val < 0.30:  # 15% in 5km buffer
            proximity = ProximityLevel.BUFFER_5KM
            pa_details = [{
                "pa_id": f"PA-{uuid.uuid4().hex[:8]}",
                "name": "Biodiversity Hotspot",
                "type": ProtectedAreaType.BIODIVERSITY_HOTSPOT.value,
                "distance_km": random.uniform(1.0, 5.0),
            }]
        elif rand_val < 0.45:  # 15% in 10km buffer
            proximity = ProximityLevel.BUFFER_10KM
            pa_details = [{
                "pa_id": f"PA-{uuid.uuid4().hex[:8]}",
                "name": "Ramsar Wetland",
                "type": ProtectedAreaType.RAMSAR_WETLAND.value,
                "distance_km": random.uniform(5.0, 10.0),
            }]
        else:  # 55% outside
            proximity = ProximityLevel.OUTSIDE
            pa_details = []

        return proximity, pa_details

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
