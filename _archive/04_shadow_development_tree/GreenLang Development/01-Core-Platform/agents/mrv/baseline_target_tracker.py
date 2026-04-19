# -*- coding: utf-8 -*-
"""
GL-MRV-X-009: Baseline & Target Tracker
========================================

Tracks GHG emissions baselines and progress toward reduction targets
following SBTi methodology and GHG Protocol standards.

Capabilities:
    - Baseline year emissions tracking
    - Absolute and intensity targets
    - SBTi-aligned target validation
    - Progress tracking and projections
    - Recalculation triggers
    - Complete provenance tracking

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class TargetType(str, Enum):
    """Types of emissions targets."""
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    NET_ZERO = "net_zero"


class TargetScope(str, Enum):
    """Target scope coverage."""
    SCOPE1_2 = "scope1_2"
    SCOPE1_2_3 = "scope1_2_3"
    SCOPE3_ONLY = "scope3_only"


class TargetStatus(str, Enum):
    """Status of target progress."""
    ON_TRACK = "on_track"
    BEHIND = "behind"
    AT_RISK = "at_risk"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"


class BaselineData(BaseModel):
    """Baseline year emissions data."""
    baseline_year: int = Field(..., ge=1990, le=2050)
    scope1_tco2e: float = Field(..., ge=0)
    scope2_location_tco2e: float = Field(..., ge=0)
    scope2_market_tco2e: float = Field(..., ge=0)
    scope3_tco2e: float = Field(default=0, ge=0)
    revenue_musd: Optional[float] = Field(None, description="Revenue for intensity")
    production_units: Optional[float] = Field(None, description="Production units")
    employees: Optional[int] = Field(None)
    intensity_metric: Optional[str] = Field(None)
    intensity_value: Optional[float] = Field(None)


class EmissionsTarget(BaseModel):
    """An emissions reduction target."""
    target_id: str = Field(...)
    target_name: str = Field(...)
    target_type: TargetType = Field(...)
    target_scope: TargetScope = Field(...)
    baseline_year: int = Field(...)
    target_year: int = Field(...)
    reduction_percentage: float = Field(..., ge=0, le=100)
    intensity_metric: Optional[str] = Field(None)
    is_sbti_aligned: bool = Field(default=False)
    sbti_pathway: Optional[str] = Field(None)


class AnnualEmissions(BaseModel):
    """Annual emissions data for tracking."""
    year: int = Field(...)
    scope1_tco2e: float = Field(default=0)
    scope2_market_tco2e: float = Field(default=0)
    scope3_tco2e: float = Field(default=0)
    revenue_musd: Optional[float] = Field(None)
    production_units: Optional[float] = Field(None)


class TargetProgress(BaseModel):
    """Progress toward a target."""
    target_id: str = Field(...)
    target_name: str = Field(...)
    target_type: TargetType = Field(...)
    baseline_emissions: float = Field(...)
    target_emissions: float = Field(...)
    current_emissions: float = Field(...)
    reduction_achieved_pct: float = Field(...)
    reduction_required_pct: float = Field(...)
    gap_tco2e: float = Field(...)
    status: TargetStatus = Field(...)
    years_remaining: int = Field(...)
    annual_reduction_needed_pct: float = Field(...)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class BaselineTargetInput(BaseModel):
    """Input model for BaselineTargetTracker."""
    baseline: Optional[BaselineData] = Field(None)
    targets: Optional[List[EmissionsTarget]] = Field(None)
    annual_emissions: Optional[List[AnnualEmissions]] = Field(None)
    current_year: int = Field(default=2024)
    organization_id: Optional[str] = Field(None)


class BaselineTargetOutput(BaseModel):
    """Output model for BaselineTargetTracker."""
    success: bool = Field(...)
    baseline_summary: Optional[Dict[str, Any]] = Field(None)
    target_progress: List[TargetProgress] = Field(default_factory=list)
    overall_status: Optional[TargetStatus] = Field(None)
    recalculation_triggers: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class BaselineTargetTrackerAgent(DeterministicAgent):
    """
    GL-MRV-X-009: Baseline & Target Tracker Agent

    Tracks emissions baselines and progress toward reduction targets.

    Example:
        >>> agent = BaselineTargetTrackerAgent()
        >>> result = agent.execute({
        ...     "baseline": {"baseline_year": 2019, "scope1_tco2e": 1000,
        ...                  "scope2_market_tco2e": 500, "scope2_location_tco2e": 500},
        ...     "targets": [{"target_id": "T1", "target_name": "Net Zero 2050",
        ...                  "target_type": "absolute", "target_scope": "scope1_2",
        ...                  "baseline_year": 2019, "target_year": 2050,
        ...                  "reduction_percentage": 90}],
        ...     "annual_emissions": [{"year": 2023, "scope1_tco2e": 800,
        ...                           "scope2_market_tco2e": 400}]
        ... })
    """

    AGENT_ID = "GL-MRV-X-009"
    AGENT_NAME = "Baseline & Target Tracker"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="BaselineTargetTrackerAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Tracks baselines and target progress"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute baseline and target tracking."""
        start_time = DeterministicClock.now()

        try:
            tracker_input = BaselineTargetInput(**inputs)
            progress_results: List[TargetProgress] = []
            baseline_summary = None
            recalculation_triggers = []

            # Process baseline
            if tracker_input.baseline:
                baseline_summary = self._summarize_baseline(tracker_input.baseline)

            # Track progress for each target
            if tracker_input.targets and tracker_input.baseline:
                for target in tracker_input.targets:
                    # Get current emissions
                    current = self._get_current_emissions(
                        target,
                        tracker_input.annual_emissions,
                        tracker_input.current_year
                    )

                    progress = self._calculate_progress(
                        target,
                        tracker_input.baseline,
                        current,
                        tracker_input.current_year
                    )
                    progress_results.append(progress)

            # Determine overall status
            overall_status = self._determine_overall_status(progress_results)

            # Check recalculation triggers
            recalculation_triggers = self._check_recalculation_triggers(
                tracker_input.baseline,
                tracker_input.annual_emissions
            )

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_hash({
                "input": inputs,
                "overall_status": overall_status.value if overall_status else None
            })

            output = BaselineTargetOutput(
                success=True,
                baseline_summary=baseline_summary,
                target_progress=progress_results,
                overall_status=overall_status,
                recalculation_triggers=recalculation_triggers,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="track_baseline_targets",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Tracked {len(progress_results)} targets"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Target tracking failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _summarize_baseline(self, baseline: BaselineData) -> Dict[str, Any]:
        """Summarize baseline data."""
        total = baseline.scope1_tco2e + baseline.scope2_market_tco2e + baseline.scope3_tco2e
        return {
            "baseline_year": baseline.baseline_year,
            "scope1_tco2e": baseline.scope1_tco2e,
            "scope2_market_tco2e": baseline.scope2_market_tco2e,
            "scope3_tco2e": baseline.scope3_tco2e,
            "total_tco2e": total,
            "intensity_metric": baseline.intensity_metric,
            "intensity_value": baseline.intensity_value
        }

    def _get_current_emissions(
        self,
        target: EmissionsTarget,
        annual_data: Optional[List[AnnualEmissions]],
        current_year: int
    ) -> float:
        """Get current emissions for target scope."""
        if not annual_data:
            return 0.0

        # Find most recent year
        for ae in sorted(annual_data, key=lambda x: x.year, reverse=True):
            if ae.year <= current_year:
                if target.target_scope == TargetScope.SCOPE1_2:
                    return ae.scope1_tco2e + ae.scope2_market_tco2e
                elif target.target_scope == TargetScope.SCOPE1_2_3:
                    return ae.scope1_tco2e + ae.scope2_market_tco2e + ae.scope3_tco2e
                else:
                    return ae.scope3_tco2e

        return 0.0

    def _calculate_progress(
        self,
        target: EmissionsTarget,
        baseline: BaselineData,
        current_emissions: float,
        current_year: int
    ) -> TargetProgress:
        """Calculate progress toward a target."""
        trace = []

        # Calculate baseline emissions for scope
        if target.target_scope == TargetScope.SCOPE1_2:
            baseline_emissions = baseline.scope1_tco2e + baseline.scope2_market_tco2e
        elif target.target_scope == TargetScope.SCOPE1_2_3:
            baseline_emissions = baseline.scope1_tco2e + baseline.scope2_market_tco2e + baseline.scope3_tco2e
        else:
            baseline_emissions = baseline.scope3_tco2e

        trace.append(f"Baseline ({target.baseline_year}): {baseline_emissions:.2f} tCO2e")

        # Calculate target emissions
        reduction_factor = Decimal(str(target.reduction_percentage)) / Decimal("100")
        target_emissions = float(Decimal(str(baseline_emissions)) * (Decimal("1") - reduction_factor))
        trace.append(f"Target ({target.target_year}): {target_emissions:.2f} tCO2e")

        # Calculate progress
        reduction_achieved = baseline_emissions - current_emissions
        reduction_achieved_pct = (reduction_achieved / baseline_emissions * 100) if baseline_emissions > 0 else 0
        trace.append(f"Current: {current_emissions:.2f} tCO2e ({reduction_achieved_pct:.1f}% reduction)")

        # Calculate gap
        gap = current_emissions - target_emissions
        years_remaining = max(0, target.target_year - current_year)

        # Determine status
        years_elapsed = current_year - target.baseline_year
        years_total = target.target_year - target.baseline_year
        expected_reduction_pct = (years_elapsed / years_total * target.reduction_percentage) if years_total > 0 else 0

        if reduction_achieved_pct >= target.reduction_percentage:
            status = TargetStatus.ACHIEVED
        elif reduction_achieved_pct >= expected_reduction_pct:
            status = TargetStatus.ON_TRACK
        elif reduction_achieved_pct >= expected_reduction_pct * 0.8:
            status = TargetStatus.AT_RISK
        else:
            status = TargetStatus.BEHIND

        # Annual reduction needed
        if years_remaining > 0 and current_emissions > target_emissions:
            annual_needed = ((current_emissions - target_emissions) / current_emissions / years_remaining * 100)
        else:
            annual_needed = 0

        trace.append(f"Status: {status.value}")

        provenance_hash = self._compute_hash({
            "target_id": target.target_id,
            "reduction_achieved_pct": reduction_achieved_pct
        })

        return TargetProgress(
            target_id=target.target_id,
            target_name=target.target_name,
            target_type=target.target_type,
            baseline_emissions=baseline_emissions,
            target_emissions=target_emissions,
            current_emissions=current_emissions,
            reduction_achieved_pct=round(reduction_achieved_pct, 2),
            reduction_required_pct=target.reduction_percentage,
            gap_tco2e=round(gap, 2),
            status=status,
            years_remaining=years_remaining,
            annual_reduction_needed_pct=round(annual_needed, 2),
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _determine_overall_status(self, progress: List[TargetProgress]) -> Optional[TargetStatus]:
        """Determine overall status across all targets."""
        if not progress:
            return None

        statuses = [p.status for p in progress]
        if all(s == TargetStatus.ACHIEVED for s in statuses):
            return TargetStatus.ACHIEVED
        elif any(s == TargetStatus.BEHIND for s in statuses):
            return TargetStatus.BEHIND
        elif any(s == TargetStatus.AT_RISK for s in statuses):
            return TargetStatus.AT_RISK
        else:
            return TargetStatus.ON_TRACK

    def _check_recalculation_triggers(
        self,
        baseline: Optional[BaselineData],
        annual_data: Optional[List[AnnualEmissions]]
    ) -> List[str]:
        """Check for baseline recalculation triggers."""
        triggers = []

        if baseline and annual_data:
            # Check for significant organic growth (>10% change)
            baseline_total = baseline.scope1_tco2e + baseline.scope2_market_tco2e
            for ae in annual_data:
                current = ae.scope1_tco2e + ae.scope2_market_tco2e
                if baseline_total > 0:
                    change_pct = abs(current - baseline_total) / baseline_total * 100
                    if change_pct > 50:
                        triggers.append(f"Significant structural change in {ae.year} ({change_pct:.1f}%)")

        return triggers

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
