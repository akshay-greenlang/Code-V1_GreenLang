# -*- coding: utf-8 -*-
"""
Transition Plan Workflow
============================

Four-phase workflow for developing and assessing financial institution
transition plans under CSRD/ESRS E1 and NZBA/GFANZ commitments.

Phases:
    1. BaselineAssessment - Establish current financed emissions baseline
    2. TargetSetting - Set sector and portfolio targets (NZBA/SBTi-FI)
    3. PathwayModeling - Model decarbonization pathways to 2030/2050
    4. CredibilityScoring - Score plan credibility and identify gaps

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
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
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

    def is_phase_completed(self, phase_name: str) -> bool:
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
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
#  Input / Result Models
# ---------------------------------------------------------------------------

class SectorTarget(BaseModel):
    """Sector-level decarbonization target."""
    sector: str = Field(..., description="NACE sector code")
    sector_name: str = Field(default="")
    base_year: int = Field(default=2022)
    baseline_intensity: float = Field(..., ge=0.0, description="tCO2e/EUR M")
    target_year_2030: Optional[float] = Field(None, ge=0.0)
    target_year_2050: Optional[float] = Field(None, ge=0.0)
    pathway_reference: str = Field(default="IEA_NZE", description="Scenario reference")
    current_intensity: Optional[float] = Field(None, ge=0.0)

class TransitionPlanInput(BaseModel):
    """Input for the transition plan workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_period: str = Field(..., description="Reporting period YYYY")
    institution_name: str = Field(default="")
    commitment_framework: str = Field(default="NZBA", description="NZBA, GFANZ, SBTi-FI")
    net_zero_target_year: int = Field(default=2050, ge=2030, le=2100)
    interim_target_year: int = Field(default=2030, ge=2025, le=2050)
    total_financed_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    total_portfolio_eur: float = Field(default=0.0, ge=0.0)
    sector_targets: List[SectorTarget] = Field(default_factory=list)
    exclusion_policies: List[str] = Field(default_factory=list)
    engagement_targets: List[str] = Field(default_factory=list)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        int(v)
        return v

class TransitionPlanResult(WorkflowResult):
    """Result from the transition plan workflow."""
    baseline_emissions_tco2e: float = Field(default=0.0)
    portfolio_intensity: float = Field(default=0.0)
    interim_reduction_target_pct: float = Field(default=0.0)
    net_zero_target_year: int = Field(default=2050)
    sectors_with_targets: int = Field(default=0)
    credibility_score: float = Field(default=0.0)
    gaps_identified: int = Field(default=0)
    on_track_sectors: int = Field(default=0)
    off_track_sectors: int = Field(default=0)

# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------

class BaselineAssessmentPhase:
    """Establish current financed emissions baseline."""
    PHASE_NAME = "baseline_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            total_emissions = config.get("total_financed_emissions_tco2e", 0.0)
            total_portfolio = config.get("total_portfolio_eur", 0.0)
            sector_targets = config.get("sector_targets", [])

            intensity = round(total_emissions / max(total_portfolio / 1_000_000, 0.001), 4)
            outputs["baseline_emissions_tco2e"] = total_emissions
            outputs["portfolio_eur"] = total_portfolio
            outputs["portfolio_intensity_tco2e_per_meur"] = intensity
            outputs["reporting_period"] = config.get("reporting_period", "")

            # Sector baselines
            sector_baselines = []
            for st in sector_targets:
                baseline_int = st.get("baseline_intensity", 0.0)
                current_int = st.get("current_intensity") or baseline_int
                reduction_from_baseline = round(
                    (baseline_int - current_int) / max(baseline_int, 0.001) * 100, 2
                ) if baseline_int > 0 else 0.0

                sector_baselines.append({
                    "sector": st.get("sector", ""),
                    "sector_name": st.get("sector_name", ""),
                    "base_year": st.get("base_year", 2022),
                    "baseline_intensity": baseline_int,
                    "current_intensity": current_int,
                    "reduction_from_baseline_pct": reduction_from_baseline,
                })

            outputs["sector_baselines"] = sector_baselines
            outputs["sectors_count"] = len(sector_baselines)

            if total_emissions == 0.0:
                warnings.append("Total financed emissions is zero; baseline not meaningful")

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("BaselineAssessment failed: %s", exc, exc_info=True)
            errors.append(f"Baseline assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class TargetSettingPhase:
    """Set sector and portfolio targets aligned with NZBA/SBTi-FI."""
    PHASE_NAME = "target_setting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            baseline = context.get_phase_output("baseline_assessment")
            sector_targets = config.get("sector_targets", [])
            framework = config.get("commitment_framework", "NZBA")
            interim_year = config.get("interim_target_year", 2030)
            nz_year = config.get("net_zero_target_year", 2050)

            targets = []
            for st in sector_targets:
                target_2030 = st.get("target_year_2030")
                target_2050 = st.get("target_year_2050")
                baseline_int = st.get("baseline_intensity", 0.0)
                reduction_2030 = round(
                    (baseline_int - (target_2030 or baseline_int)) / max(baseline_int, 0.001) * 100, 2
                ) if baseline_int > 0 else 0.0

                targets.append({
                    "sector": st.get("sector", ""),
                    "sector_name": st.get("sector_name", ""),
                    "baseline_intensity": baseline_int,
                    "target_2030": target_2030,
                    "target_2050": target_2050,
                    "reduction_pct_by_2030": reduction_2030,
                    "pathway_reference": st.get("pathway_reference", "IEA_NZE"),
                })

            outputs["sector_targets"] = targets
            outputs["sectors_with_targets"] = len(targets)
            outputs["framework"] = framework
            outputs["interim_target_year"] = interim_year
            outputs["net_zero_target_year"] = nz_year

            if not targets:
                warnings.append("No sector-level targets defined; NZBA requires sector targets")

            avg_reduction = round(
                sum(t.get("reduction_pct_by_2030", 0.0) for t in targets) / max(len(targets), 1), 2
            )
            outputs["avg_interim_reduction_pct"] = avg_reduction

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("TargetSetting failed: %s", exc, exc_info=True)
            errors.append(f"Target setting failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class PathwayModelingPhase:
    """Model decarbonization pathways to 2030/2050."""
    PHASE_NAME = "pathway_modeling"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            baseline = context.get_phase_output("baseline_assessment")
            targets = context.get_phase_output("target_setting")
            sector_baselines = baseline.get("sector_baselines", [])
            sector_targets = targets.get("sector_targets", [])

            pathways = []
            on_track = 0
            off_track = 0

            target_map = {t["sector"]: t for t in sector_targets}

            for sb in sector_baselines:
                sector = sb.get("sector", "")
                current = sb.get("current_intensity", 0.0)
                baseline_int = sb.get("baseline_intensity", 0.0)
                target = target_map.get(sector, {})
                target_2030 = target.get("target_2030")

                if target_2030 is not None and baseline_int > 0:
                    # Linear pathway check
                    years_elapsed = 2026 - sb.get("base_year", 2022)
                    years_total = 2030 - sb.get("base_year", 2022)
                    expected = baseline_int - (baseline_int - target_2030) * (years_elapsed / max(years_total, 1))
                    is_on_track = current <= expected * 1.05
                    if is_on_track:
                        on_track += 1
                    else:
                        off_track += 1
                else:
                    is_on_track = None

                pathways.append({
                    "sector": sector,
                    "sector_name": sb.get("sector_name", ""),
                    "baseline_intensity": baseline_int,
                    "current_intensity": current,
                    "target_2030": target_2030,
                    "on_track": is_on_track,
                    "gap_to_target": round(current - (target_2030 or current), 2) if target_2030 else None,
                })

            outputs["pathways"] = pathways
            outputs["on_track_count"] = on_track
            outputs["off_track_count"] = off_track
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("PathwayModeling failed: %s", exc, exc_info=True)
            errors.append(f"Pathway modeling failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class CredibilityScoringPhase:
    """Score transition plan credibility and identify gaps."""
    PHASE_NAME = "credibility_scoring"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        try:
            config = context.config
            targets = context.get_phase_output("target_setting")
            pathways = context.get_phase_output("pathway_modeling")

            score = 0.0
            max_score = 100.0
            gaps = []

            # 1. Sector targets defined (25 pts)
            sector_count = targets.get("sectors_with_targets", 0)
            if sector_count >= 5:
                score += 25.0
            elif sector_count >= 3:
                score += 15.0
            elif sector_count >= 1:
                score += 8.0
            else:
                gaps.append("No sector-level targets defined")

            # 2. On-track sectors (25 pts)
            on_track = pathways.get("on_track_count", 0)
            off_track = pathways.get("off_track_count", 0)
            total = on_track + off_track
            if total > 0:
                on_track_pct = on_track / total
                score += 25.0 * on_track_pct
                if on_track_pct < 0.5:
                    gaps.append(f"Only {on_track}/{total} sectors on track")

            # 3. Exclusion policies (15 pts)
            exclusions = config.get("exclusion_policies", [])
            if len(exclusions) >= 3:
                score += 15.0
            elif len(exclusions) >= 1:
                score += 8.0
            else:
                gaps.append("No exclusion policies defined")

            # 4. Engagement targets (15 pts)
            engagement = config.get("engagement_targets", [])
            if len(engagement) >= 3:
                score += 15.0
            elif len(engagement) >= 1:
                score += 8.0
            else:
                gaps.append("No client engagement targets defined")

            # 5. Framework commitment (10 pts)
            framework = config.get("commitment_framework", "")
            if framework in ("NZBA", "GFANZ", "SBTi-FI"):
                score += 10.0
            elif framework:
                score += 5.0
            else:
                gaps.append("No net-zero commitment framework specified")

            # 6. Interim target ambition (10 pts)
            avg_reduction = targets.get("avg_interim_reduction_pct", 0.0)
            if avg_reduction >= 40:
                score += 10.0
            elif avg_reduction >= 25:
                score += 6.0
            elif avg_reduction >= 10:
                score += 3.0
            else:
                gaps.append("Interim reduction targets below 10%")

            outputs["credibility_score"] = round(min(score, max_score), 2)
            outputs["max_score"] = max_score
            outputs["gaps"] = gaps
            outputs["gaps_count"] = len(gaps)
            outputs["on_track_sectors"] = on_track
            outputs["off_track_sectors"] = off_track
            status = PhaseStatus.COMPLETED
        except Exception as exc:
            logger.error("CredibilityScoring failed: %s", exc, exc_info=True)
            errors.append(f"Credibility scoring failed: {str(exc)}")
            status = PhaseStatus.FAILED
        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class TransitionPlanWorkflow:
    """Four-phase transition plan workflow for CSRD FI decarbonization planning."""

    WORKFLOW_NAME = "transition_plan"
    PHASE_ORDER = ["baseline_assessment", "target_setting",
                    "pathway_modeling", "credibility_scoring"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "baseline_assessment": BaselineAssessmentPhase(),
            "target_setting": TargetSettingPhase(),
            "pathway_modeling": PathwayModelingPhase(),
            "credibility_scoring": CredibilityScoringPhase(),
        }

    async def run(self, input_data: TransitionPlanInput) -> TransitionPlanResult:
        """Execute the workflow."""
        started_at = utcnow()
        logger.info("Starting %s workflow %s org=%s", self.WORKFLOW_NAME,
                     self.workflow_id, input_data.organization_id)
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
            if context.is_phase_completed(phase_name):
                continue
            self._notify_progress(phase_name, f"Starting: {phase_name}",
                                  idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == self.PHASE_ORDER[0]:
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
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return TransitionPlanResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            **{k: summary.get(k, v) for k, v in self._result_defaults().items()}
        )

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        baseline = context.get_phase_output("baseline_assessment")
        targets = context.get_phase_output("target_setting")
        credibility = context.get_phase_output("credibility_scoring")
        return {
            "baseline_emissions_tco2e": baseline.get("baseline_emissions_tco2e", 0.0),
            "portfolio_intensity": baseline.get("portfolio_intensity_tco2e_per_meur", 0.0),
            "interim_reduction_target_pct": targets.get("avg_interim_reduction_pct", 0.0),
            "net_zero_target_year": targets.get("net_zero_target_year", 2050),
            "sectors_with_targets": targets.get("sectors_with_targets", 0),
            "credibility_score": credibility.get("credibility_score", 0.0),
            "gaps_identified": credibility.get("gaps_count", 0),
            "on_track_sectors": credibility.get("on_track_sectors", 0),
            "off_track_sectors": credibility.get("off_track_sectors", 0),
        }

    @staticmethod
    def _result_defaults() -> Dict[str, Any]:
        return {
            "baseline_emissions_tco2e": 0.0, "portfolio_intensity": 0.0,
            "interim_reduction_target_pct": 0.0, "net_zero_target_year": 2050,
            "sectors_with_targets": 0, "credibility_score": 0.0,
            "gaps_identified": 0, "on_track_sectors": 0, "off_track_sectors": 0,
        }
