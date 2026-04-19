# -*- coding: utf-8 -*-
"""
GL-REP-PUB-002: Climate Action Progress Report Agent
=====================================================

Tracks and reports progress on municipal climate action plans including
emissions reductions, target achievement, and implementation milestones.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class TargetStatus(str, Enum):
    """Status of climate targets."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"


class ActionStatus(str, Enum):
    """Status of climate actions."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class ReportingPeriod(str, Enum):
    """Reporting period frequency."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    BIENNIAL = "biennial"


class EmissionsProgress(BaseModel):
    """Emissions progress tracking."""
    sector: str = Field(...)
    baseline_year: int = Field(...)
    baseline_emissions_tco2e: float = Field(..., ge=0)
    current_year: int = Field(...)
    current_emissions_tco2e: float = Field(..., ge=0)
    target_year: int = Field(...)
    target_emissions_tco2e: float = Field(..., ge=0)
    reduction_achieved_pct: float = Field(default=0.0)
    reduction_required_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)


class ActionProgress(BaseModel):
    """Progress on individual climate actions."""
    action_id: str = Field(...)
    action_name: str = Field(...)
    status: ActionStatus = Field(default=ActionStatus.PLANNED)
    planned_start_date: Optional[datetime] = Field(None)
    actual_start_date: Optional[datetime] = Field(None)
    planned_completion_date: Optional[datetime] = Field(None)
    actual_completion_date: Optional[datetime] = Field(None)
    budget_allocated_usd: float = Field(default=0.0, ge=0)
    budget_spent_usd: float = Field(default=0.0, ge=0)
    estimated_reduction_tco2e: float = Field(default=0.0, ge=0)
    verified_reduction_tco2e: float = Field(default=0.0, ge=0)
    completion_pct: float = Field(default=0.0, ge=0, le=100)
    notes: Optional[str] = Field(None)


class TargetProgress(BaseModel):
    """Progress toward climate targets."""
    target_id: str = Field(...)
    target_name: str = Field(...)
    target_type: str = Field(...)  # absolute, intensity, renewable
    target_year: int = Field(...)
    target_value: float = Field(...)
    current_value: float = Field(...)
    baseline_value: float = Field(...)
    progress_pct: float = Field(default=0.0)
    status: TargetStatus = Field(default=TargetStatus.NOT_STARTED)
    trajectory_assessment: Optional[str] = Field(None)


class ProgressReport(BaseModel):
    """Climate action progress report."""
    report_id: str = Field(...)
    municipality_name: str = Field(...)
    report_title: str = Field(...)
    reporting_period: ReportingPeriod = Field(...)
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    emissions_progress: List[EmissionsProgress] = Field(default_factory=list)
    target_progress: List[TargetProgress] = Field(default_factory=list)
    action_progress: List[ActionProgress] = Field(default_factory=list)
    total_emissions_baseline_tco2e: float = Field(default=0.0, ge=0)
    total_emissions_current_tco2e: float = Field(default=0.0, ge=0)
    total_reduction_achieved_tco2e: float = Field(default=0.0)
    total_reduction_achieved_pct: float = Field(default=0.0)
    overall_status: TargetStatus = Field(default=TargetStatus.NOT_STARTED)
    key_achievements: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    next_period_priorities: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class ProgressReportInput(BaseModel):
    """Input for Progress Report Agent."""
    action: str = Field(...)
    report_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    reporting_period: Optional[ReportingPeriod] = Field(None)
    period_start: Optional[datetime] = Field(None)
    period_end: Optional[datetime] = Field(None)
    emissions_progress: Optional[EmissionsProgress] = Field(None)
    target_progress: Optional[TargetProgress] = Field(None)
    action_progress: Optional[ActionProgress] = Field(None)
    achievement: Optional[str] = Field(None)
    challenge: Optional[str] = Field(None)
    priority: Optional[str] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_report', 'add_emissions_progress', 'add_target_progress',
                 'add_action_progress', 'add_achievement', 'add_challenge',
                 'add_priority', 'calculate_overall_progress', 'generate_summary',
                 'get_report', 'list_reports'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class ProgressReportOutput(BaseModel):
    """Output from Progress Report Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    report: Optional[ProgressReport] = Field(None)
    reports: Optional[List[ProgressReport]] = Field(None)
    summary: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class ClimateActionProgressReportAgent(BaseAgent):
    """GL-REP-PUB-002: Climate Action Progress Report Agent"""

    AGENT_ID = "GL-REP-PUB-002"
    AGENT_NAME = "Climate Action Progress Report Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Climate action progress tracking and reporting",
                version=self.VERSION,
            )
        super().__init__(config)
        self._reports: Dict[str, ProgressReport] = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            inp = ProgressReportInput(**input_data)
            handlers = {
                'create_report': self._create_report,
                'add_emissions_progress': self._add_emissions_progress,
                'add_target_progress': self._add_target_progress,
                'add_action_progress': self._add_action_progress,
                'add_achievement': self._add_achievement,
                'add_challenge': self._add_challenge,
                'add_priority': self._add_priority,
                'calculate_overall_progress': self._calculate_overall,
                'generate_summary': self._generate_summary,
                'get_report': self._get_report,
                'list_reports': self._list_reports,
            }
            out = handlers[inp.action](inp)
            out.processing_time_ms = (time.time() - start_time) * 1000
            out.provenance_hash = self._hash_output(out)
            return AgentResult(success=out.success, data=out.model_dump(), error=out.error)
        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_report(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.municipality_name or not inp.reporting_period:
            return ProgressReportOutput(success=False, action='create_report', error="Municipality and period required")
        report_id = f"PR-{inp.municipality_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m')}"
        report = ProgressReport(
            report_id=report_id,
            municipality_name=inp.municipality_name,
            report_title=f"{inp.municipality_name} Climate Action Progress Report",
            reporting_period=inp.reporting_period,
            period_start=inp.period_start or DeterministicClock.now(),
            period_end=inp.period_end or DeterministicClock.now(),
        )
        self._reports[report_id] = report
        return ProgressReportOutput(success=True, action='create_report', report=report, calculation_trace=[f"Created {report_id}"])

    def _add_emissions_progress(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.emissions_progress:
            return ProgressReportOutput(success=False, action='add_emissions_progress', error="Report ID and emissions progress required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_emissions_progress', error="Report not found")
        ep = inp.emissions_progress
        if ep.baseline_emissions_tco2e > 0:
            ep.reduction_achieved_pct = ((ep.baseline_emissions_tco2e - ep.current_emissions_tco2e) / ep.baseline_emissions_tco2e) * 100
            ep.reduction_required_pct = ((ep.baseline_emissions_tco2e - ep.target_emissions_tco2e) / ep.baseline_emissions_tco2e) * 100
            years_elapsed = max(1, ep.current_year - ep.baseline_year)
            ep.annual_reduction_rate_pct = ep.reduction_achieved_pct / years_elapsed
        report.emissions_progress.append(ep)
        return ProgressReportOutput(success=True, action='add_emissions_progress', report=report, calculation_trace=[f"Added {ep.sector} emissions progress"])

    def _add_target_progress(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.target_progress:
            return ProgressReportOutput(success=False, action='add_target_progress', error="Report ID and target progress required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_target_progress', error="Report not found")
        tp = inp.target_progress
        if tp.baseline_value != tp.target_value:
            total_change_required = abs(tp.baseline_value - tp.target_value)
            change_achieved = abs(tp.baseline_value - tp.current_value)
            tp.progress_pct = min(100, (change_achieved / total_change_required) * 100) if total_change_required > 0 else 0
        if tp.progress_pct >= 100:
            tp.status = TargetStatus.ACHIEVED
        elif tp.progress_pct >= 70:
            tp.status = TargetStatus.ON_TRACK
        elif tp.progress_pct >= 40:
            tp.status = TargetStatus.AT_RISK
        else:
            tp.status = TargetStatus.OFF_TRACK
        report.target_progress.append(tp)
        return ProgressReportOutput(success=True, action='add_target_progress', report=report, calculation_trace=[f"Added {tp.target_name} progress"])

    def _add_action_progress(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.action_progress:
            return ProgressReportOutput(success=False, action='add_action_progress', error="Report ID and action progress required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_action_progress', error="Report not found")
        report.action_progress.append(inp.action_progress)
        return ProgressReportOutput(success=True, action='add_action_progress', report=report, calculation_trace=[f"Added action {inp.action_progress.action_id}"])

    def _add_achievement(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.achievement:
            return ProgressReportOutput(success=False, action='add_achievement', error="Report ID and achievement required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_achievement', error="Report not found")
        report.key_achievements.append(inp.achievement)
        return ProgressReportOutput(success=True, action='add_achievement', report=report, calculation_trace=["Added achievement"])

    def _add_challenge(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.challenge:
            return ProgressReportOutput(success=False, action='add_challenge', error="Report ID and challenge required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_challenge', error="Report not found")
        report.challenges.append(inp.challenge)
        return ProgressReportOutput(success=True, action='add_challenge', report=report, calculation_trace=["Added challenge"])

    def _add_priority(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id or not inp.priority:
            return ProgressReportOutput(success=False, action='add_priority', error="Report ID and priority required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='add_priority', error="Report not found")
        report.next_period_priorities.append(inp.priority)
        return ProgressReportOutput(success=True, action='add_priority', report=report, calculation_trace=["Added priority"])

    def _calculate_overall(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id:
            return ProgressReportOutput(success=False, action='calculate_overall_progress', error="Report ID required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='calculate_overall_progress', error="Report not found")
        report.total_emissions_baseline_tco2e = sum(ep.baseline_emissions_tco2e for ep in report.emissions_progress)
        report.total_emissions_current_tco2e = sum(ep.current_emissions_tco2e for ep in report.emissions_progress)
        report.total_reduction_achieved_tco2e = report.total_emissions_baseline_tco2e - report.total_emissions_current_tco2e
        if report.total_emissions_baseline_tco2e > 0:
            report.total_reduction_achieved_pct = (report.total_reduction_achieved_tco2e / report.total_emissions_baseline_tco2e) * 100
        status_counts = {s: 0 for s in TargetStatus}
        for tp in report.target_progress:
            status_counts[tp.status] += 1
        if status_counts[TargetStatus.ACHIEVED] == len(report.target_progress) and len(report.target_progress) > 0:
            report.overall_status = TargetStatus.ACHIEVED
        elif status_counts[TargetStatus.OFF_TRACK] > len(report.target_progress) / 2:
            report.overall_status = TargetStatus.OFF_TRACK
        elif status_counts[TargetStatus.AT_RISK] > len(report.target_progress) / 2:
            report.overall_status = TargetStatus.AT_RISK
        elif status_counts[TargetStatus.ON_TRACK] >= len(report.target_progress) / 2:
            report.overall_status = TargetStatus.ON_TRACK
        return ProgressReportOutput(success=True, action='calculate_overall_progress', report=report, calculation_trace=[f"Overall: {report.total_reduction_achieved_pct:.1f}% reduction"])

    def _generate_summary(self, inp: ProgressReportInput) -> ProgressReportOutput:
        if not inp.report_id:
            return ProgressReportOutput(success=False, action='generate_summary', error="Report ID required")
        report = self._reports.get(inp.report_id)
        if not report:
            return ProgressReportOutput(success=False, action='generate_summary', error="Report not found")
        summary = {
            "municipality": report.municipality_name,
            "period": report.reporting_period.value,
            "overall_status": report.overall_status.value,
            "total_reduction_tco2e": report.total_reduction_achieved_tco2e,
            "total_reduction_pct": report.total_reduction_achieved_pct,
            "targets_on_track": sum(1 for tp in report.target_progress if tp.status == TargetStatus.ON_TRACK),
            "targets_achieved": sum(1 for tp in report.target_progress if tp.status == TargetStatus.ACHIEVED),
            "targets_at_risk": sum(1 for tp in report.target_progress if tp.status == TargetStatus.AT_RISK),
            "actions_completed": sum(1 for ap in report.action_progress if ap.status == ActionStatus.COMPLETED),
            "actions_in_progress": sum(1 for ap in report.action_progress if ap.status == ActionStatus.IN_PROGRESS),
            "key_achievements": report.key_achievements,
            "challenges": report.challenges,
        }
        return ProgressReportOutput(success=True, action='generate_summary', report=report, summary=summary, calculation_trace=["Summary generated"])

    def _get_report(self, inp: ProgressReportInput) -> ProgressReportOutput:
        report = self._reports.get(inp.report_id) if inp.report_id else None
        if not report:
            return ProgressReportOutput(success=False, action='get_report', error="Report not found")
        return ProgressReportOutput(success=True, action='get_report', report=report)

    def _list_reports(self, inp: ProgressReportInput) -> ProgressReportOutput:
        return ProgressReportOutput(success=True, action='list_reports', reports=list(self._reports.values()))

    def _hash_output(self, output: ProgressReportOutput) -> str:
        content = {"action": output.action, "success": output.success}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
