# -*- coding: utf-8 -*-
"""
Compliance Engine for GL-010 EmissionsGuardian

This module implements the main compliance evaluation engine that performs
deterministic rule evaluation against emissions data. All calculations follow
EPA 40 CFR Part 75 methods and zero-hallucination principles.

Zero-Hallucination Principle:
- All threshold calculations are deterministic arithmetic
- No LLM inference for numeric compliance determinations
- Complete provenance tracking with SHA-256 hashes
"""

from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from .schemas import (
    AveragingPeriod,
    OperatingState,
    ExceedanceSeverity,
    RegulatoryProgram,
    PermitRule,
    ExceedanceEvent,
    ComplianceStatus,
)

logger = logging.getLogger(__name__)


class EmissionsDataPoint(BaseModel):
    """Single emissions measurement for compliance evaluation."""
    timestamp: datetime = Field(..., description="Measurement timestamp")
    unit_id: str = Field(..., description="Emission unit identifier")
    pollutant: str = Field(..., description="Pollutant code")
    measured_value: Decimal = Field(..., ge=0, description="Measured value")
    measurement_unit: str = Field(..., description="Unit of measurement")
    data_quality: str = Field(default="VALID")
    is_substituted: bool = Field(default=False)
    operating_load_pct: Optional[float] = Field(None, ge=0, le=100)
    heat_input_mmbtu: Optional[Decimal] = Field(None, ge=0)

    @field_validator("pollutant")
    @classmethod
    def normalize_pollutant(cls, v: str) -> str:
        return v.upper()


class HourlyEvaluationInput(BaseModel):
    """Input for hourly compliance evaluation."""
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    evaluation_hour: datetime = Field(...)
    operating_state: OperatingState = Field(...)
    operating_params: Dict[str, Any] = Field(default_factory=dict)
    emissions_data: List[EmissionsDataPoint] = Field(..., min_length=1)

    def calculate_input_hash(self) -> str:
        content = f"{self.facility_id}|{self.unit_id}|{self.evaluation_hour.isoformat()}"
        for dp in sorted(self.emissions_data, key=lambda x: x.pollutant):
            content += f"|{dp.pollutant}:{dp.measured_value}"
        return hashlib.sha256(content.encode()).hexdigest()


class RollingEvaluationInput(BaseModel):
    """Input for rolling average compliance evaluation."""
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    evaluation_end: datetime = Field(...)
    averaging_period: AveragingPeriod = Field(...)
    pollutant: str = Field(...)
    hourly_values: List[Tuple[datetime, Decimal]] = Field(default_factory=list)
    operating_hours_only: bool = Field(default=True)


@dataclass
class RuleEvaluationResult:
    """Result of evaluating a single rule."""
    rule_id: str
    permit_id: str
    pollutant: str
    is_applicable: bool
    limit_value: Decimal
    measured_value: Decimal
    measurement_unit: str
    percentage_of_limit: float
    threshold_status: str
    warning_threshold_pct: float
    action_threshold_pct: float
    exceedance_pct: float
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    exemption_applied: bool = False
    exemption_reason: Optional[str] = None
    data_quality_flag: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "permit_id": self.permit_id,
            "pollutant": self.pollutant,
            "is_applicable": self.is_applicable,
            "limit_value": str(self.limit_value),
            "measured_value": str(self.measured_value),
            "percentage_of_limit": self.percentage_of_limit,
            "threshold_status": self.threshold_status,
            "exceedance_pct": self.exceedance_pct,
            "exemption_applied": self.exemption_applied,
        }


class ComplianceEvaluationOutput(BaseModel):
    """Output from compliance evaluation."""
    evaluation_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_type: str = Field(...)
    averaging_period: Optional[AveragingPeriod] = Field(None)
    overall_status: str = Field(...)
    rules_evaluated: int = Field(..., ge=0)
    rules_applicable: int = Field(..., ge=0)
    rules_compliant: int = Field(..., ge=0)
    rules_warning: int = Field(..., ge=0)
    rules_exceeded: int = Field(..., ge=0)
    rule_results: List[Dict[str, Any]] = Field(default_factory=list)
    exceedance_events: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(..., ge=0)
    input_hash: str = Field(...)
    output_hash: str = Field(...)
    provenance_hash: str = Field(...)

    def calculate_output_hash(self) -> str:
        content = f"{self.evaluation_id}|{self.overall_status}|{self.rules_exceeded}"
        return hashlib.sha256(content.encode()).hexdigest()


class ComplianceEngine:
    """Main compliance evaluation engine for emissions monitoring."""

    ROLLING_PERIODS = {
        AveragingPeriod.ROLLING_3HOUR: 3,
        AveragingPeriod.ROLLING_24HOUR: 24,
        AveragingPeriod.ROLLING_30DAY: 720,
    }

    def __init__(
        self,
        rules_repository: Any,
        warning_threshold_default: float = 90.0,
        enable_exemptions: bool = True,
        cache_size_hours: int = 720,
    ):
        self.rules_repository = rules_repository
        self.warning_threshold_default = warning_threshold_default
        self.enable_exemptions = enable_exemptions
        self.cache_size_hours = cache_size_hours
        self._rolling_cache: Dict[Tuple[str, str], deque] = {}
        self._evaluation_counter = 0
        logger.info(f"ComplianceEngine initialized: warning={warning_threshold_default}%")

    def _generate_evaluation_id(self, eval_type: str) -> str:
        self._evaluation_counter += 1
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"EVAL-{eval_type}-{ts}-{self._evaluation_counter:06d}"

    def _calculate_provenance_hash(self, input_hash: str, output_hash: str) -> str:
        combined = f"{input_hash}|{output_hash}|{datetime.now().isoformat()}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _check_exemption(
        self, operating_state: OperatingState, rule: PermitRule
    ) -> Tuple[bool, Optional[str]]:
        if not self.enable_exemptions:
            return False, None
        if operating_state in rule.exemption_states:
            return True, f"State '{operating_state.value}' exempt per rule {rule.rule_id}"
        return False, None

    def _evaluate_single_rule(
        self,
        rule: PermitRule,
        measured_value: Decimal,
        measurement_unit: str,
        operating_state: OperatingState,
        operating_params: Dict[str, Any],
        check_date: Optional[date] = None,
    ) -> RuleEvaluationResult:
        """DETERMINISTIC calculation - no LLM inference."""
        notes: List[str] = []

        is_applicable = rule.is_applicable(operating_state, operating_params, check_date)
        if not is_applicable:
            return RuleEvaluationResult(
                rule_id=rule.rule_id, permit_id=rule.permit_id, pollutant=rule.pollutant,
                is_applicable=False, limit_value=rule.limit_value, measured_value=measured_value,
                measurement_unit=measurement_unit, percentage_of_limit=0.0,
                threshold_status="NOT_APPLICABLE", warning_threshold_pct=rule.warning_threshold_pct,
                action_threshold_pct=rule.action_threshold_pct, exceedance_pct=0.0, notes=notes,
            )

        exempt, reason = self._check_exemption(operating_state, rule)
        if exempt:
            return RuleEvaluationResult(
                rule_id=rule.rule_id, permit_id=rule.permit_id, pollutant=rule.pollutant,
                is_applicable=True, limit_value=rule.limit_value, measured_value=measured_value,
                measurement_unit=measurement_unit,
                percentage_of_limit=float(measured_value / rule.limit_value * 100),
                threshold_status="EXEMPT", warning_threshold_pct=rule.warning_threshold_pct,
                action_threshold_pct=rule.action_threshold_pct, exceedance_pct=0.0,
                exemption_applied=True, exemption_reason=reason, notes=notes,
            )

        # DETERMINISTIC CALCULATION
        pct = float((measured_value / rule.limit_value * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP))
        warn_thresh = rule.warning_threshold_pct or self.warning_threshold_default
        action_thresh = rule.action_threshold_pct or 100.0

        if pct >= action_thresh:
            status, exc_pct = "EXCEEDED", pct - 100.0
        elif pct >= warn_thresh:
            status, exc_pct = "WARNING", 0.0
        else:
            status, exc_pct = "COMPLIANT", 0.0

        return RuleEvaluationResult(
            rule_id=rule.rule_id, permit_id=rule.permit_id, pollutant=rule.pollutant,
            is_applicable=True, limit_value=rule.limit_value, measured_value=measured_value,
            measurement_unit=measurement_unit, percentage_of_limit=pct, threshold_status=status,
            warning_threshold_pct=warn_thresh, action_threshold_pct=action_thresh,
            exceedance_pct=exc_pct, notes=notes,
        )

    def evaluate_hourly(self, input_data: HourlyEvaluationInput) -> ComplianceEvaluationOutput:
        """Evaluate hourly emissions against applicable permit limits."""
        start = datetime.now()
        eval_id = self._generate_evaluation_id("HOURLY")
        input_hash = input_data.calculate_input_hash()

        rules = self.rules_repository.get_rules_for_unit(
            unit_id=input_data.unit_id, averaging_period=AveragingPeriod.HOURLY,
            check_date=input_data.evaluation_hour.date(),
        )

        emissions = {dp.pollutant: dp.measured_value for dp in input_data.emissions_data}
        units = {dp.pollutant: dp.measurement_unit for dp in input_data.emissions_data}

        results, exc_ids = [], []
        for rule in rules:
            poll = rule.pollutant.upper()
            if poll not in emissions:
                continue
            res = self._evaluate_single_rule(
                rule, emissions[poll], units[poll], input_data.operating_state,
                input_data.operating_params, input_data.evaluation_hour.date(),
            )
            results.append(res)
            if res.threshold_status == "EXCEEDED":
                exc_ids.append(f"EXC-{eval_id}-{rule.rule_id}")

        for poll, val in emissions.items():
            self._update_cache(input_data.unit_id, poll, input_data.evaluation_hour, val)

        n_eval, n_app = len(results), sum(1 for r in results if r.is_applicable)
        n_ok = sum(1 for r in results if r.threshold_status == "COMPLIANT")
        n_warn = sum(1 for r in results if r.threshold_status == "WARNING")
        n_exc = sum(1 for r in results if r.threshold_status == "EXCEEDED")
        status = "EXCEEDED" if n_exc else ("WARNING" if n_warn else "COMPLIANT")

        output = ComplianceEvaluationOutput(
            evaluation_id=eval_id, facility_id=input_data.facility_id, unit_id=input_data.unit_id,
            evaluation_type="HOURLY", averaging_period=AveragingPeriod.HOURLY, overall_status=status,
            rules_evaluated=n_eval, rules_applicable=n_app, rules_compliant=n_ok,
            rules_warning=n_warn, rules_exceeded=n_exc, rule_results=[r.to_dict() for r in results],
            exceedance_events=exc_ids, processing_time_ms=(datetime.now()-start).total_seconds()*1000,
            input_hash=input_hash, output_hash="", provenance_hash="",
        )
        output.output_hash = output.calculate_output_hash()
        output.provenance_hash = self._calculate_provenance_hash(input_hash, output.output_hash)
        return output

    def _update_cache(self, unit_id: str, pollutant: str, ts: datetime, val: Decimal) -> None:
        key = (unit_id, pollutant.upper())
        if key not in self._rolling_cache:
            self._rolling_cache[key] = deque(maxlen=self.cache_size_hours)
        self._rolling_cache[key].append((ts, val))

    def _get_rolling(self, unit_id: str, pollutant: str, end: datetime, hrs: int):
        key = (unit_id, pollutant.upper())
        if key not in self._rolling_cache:
            return []
        start = end - timedelta(hours=hrs)
        return [(t, v) for t, v in self._rolling_cache[key] if start < t <= end]

    def evaluate_rolling(self, input_data: RollingEvaluationInput) -> ComplianceEvaluationOutput:
        """Evaluate rolling average emissions."""
        start = datetime.now()
        eval_id = self._generate_evaluation_id("ROLLING")
        hrs = self.ROLLING_PERIODS.get(input_data.averaging_period)
        if not hrs:
            raise ValueError(f"Invalid period: {input_data.averaging_period}")

        vals = input_data.hourly_values or self._get_rolling(
            input_data.unit_id, input_data.pollutant, input_data.evaluation_end, hrs)
        avg = sum(v for _, v in vals) / Decimal(len(vals)) if vals else Decimal("0")

        content = f"{input_data.facility_id}|{input_data.unit_id}|{input_data.pollutant}"
        input_hash = hashlib.sha256(content.encode()).hexdigest()

        rules = self.rules_repository.get_rules_for_unit(
            unit_id=input_data.unit_id, averaging_period=input_data.averaging_period,
            pollutant=input_data.pollutant, check_date=input_data.evaluation_end.date(),
        )

        results, exc_ids = [], []
        for rule in rules:
            res = self._evaluate_single_rule(
                rule, avg, rule.limit_unit, OperatingState.NORMAL, {},
                input_data.evaluation_end.date(),
            )
            results.append(res)
            if res.threshold_status == "EXCEEDED":
                exc_ids.append(f"EXC-{eval_id}-{rule.rule_id}")

        n_eval, n_app = len(results), sum(1 for r in results if r.is_applicable)
        n_ok = sum(1 for r in results if r.threshold_status == "COMPLIANT")
        n_warn = sum(1 for r in results if r.threshold_status == "WARNING")
        n_exc = sum(1 for r in results if r.threshold_status == "EXCEEDED")
        status = "EXCEEDED" if n_exc else ("WARNING" if n_warn else "COMPLIANT")

        output = ComplianceEvaluationOutput(
            evaluation_id=eval_id, facility_id=input_data.facility_id, unit_id=input_data.unit_id,
            evaluation_type="ROLLING", averaging_period=input_data.averaging_period,
            overall_status=status, rules_evaluated=n_eval, rules_applicable=n_app,
            rules_compliant=n_ok, rules_warning=n_warn, rules_exceeded=n_exc,
            rule_results=[r.to_dict() for r in results], exceedance_events=exc_ids,
            processing_time_ms=(datetime.now()-start).total_seconds()*1000,
            input_hash=input_hash, output_hash="", provenance_hash="",
        )
        output.output_hash = output.calculate_output_hash()
        output.provenance_hash = self._calculate_provenance_hash(input_hash, output.output_hash)
        return output

    def evaluate_permit(
        self, facility_id: str, unit_id: str, permit_id: str, evaluation_date: date,
        emissions_summary: Dict[str, Dict[str, Decimal]],
        operating_state: OperatingState = OperatingState.NORMAL,
        operating_params: Optional[Dict[str, Any]] = None,
    ) -> ComplianceEvaluationOutput:
        """Evaluate all permit rules for a unit."""
        start = datetime.now()
        eval_id = self._generate_evaluation_id("PERMIT")
        operating_params = operating_params or {}

        content = f"{facility_id}|{unit_id}|{permit_id}|{evaluation_date}"
        input_hash = hashlib.sha256(content.encode()).hexdigest()

        rules = self.rules_repository.get_rules_by_permit(permit_id, evaluation_date)
        results, exc_ids = [], []

        for rule in rules:
            poll, period = rule.pollutant.upper(), rule.averaging_period.value
            if poll not in emissions_summary or period not in emissions_summary[poll]:
                continue
            res = self._evaluate_single_rule(
                rule, emissions_summary[poll][period], rule.limit_unit,
                operating_state, operating_params, evaluation_date,
            )
            results.append(res)
            if res.threshold_status == "EXCEEDED":
                exc_ids.append(f"EXC-{eval_id}-{rule.rule_id}")

        n_eval, n_app = len(results), sum(1 for r in results if r.is_applicable)
        n_ok = sum(1 for r in results if r.threshold_status == "COMPLIANT")
        n_warn = sum(1 for r in results if r.threshold_status == "WARNING")
        n_exc = sum(1 for r in results if r.threshold_status == "EXCEEDED")
        status = "EXCEEDED" if n_exc else ("WARNING" if n_warn else "COMPLIANT")

        output = ComplianceEvaluationOutput(
            evaluation_id=eval_id, facility_id=facility_id, unit_id=unit_id,
            evaluation_type="PERMIT", averaging_period=None, overall_status=status,
            rules_evaluated=n_eval, rules_applicable=n_app, rules_compliant=n_ok,
            rules_warning=n_warn, rules_exceeded=n_exc,
            rule_results=[r.to_dict() for r in results], exceedance_events=exc_ids,
            processing_time_ms=(datetime.now()-start).total_seconds()*1000,
            input_hash=input_hash, output_hash="", provenance_hash="",
        )
        output.output_hash = output.calculate_output_hash()
        output.provenance_hash = self._calculate_provenance_hash(input_hash, output.output_hash)
        return output

    def get_compliance_status(
        self, facility_id: str, unit_id: Optional[str] = None, as_of: Optional[datetime] = None
    ) -> ComplianceStatus:
        """Get aggregated compliance status."""
        as_of = as_of or datetime.now()
        status = ComplianceStatus(
            facility_id=facility_id, unit_id=unit_id, status_date=as_of,
            overall_status="COMPLIANT", compliance_score=100.0, pollutant_status={},
            program_status={}, active_exceedances=0, active_warnings=0,
            open_corrective_actions=0, overdue_actions=0, pending_submissions=0,
            overdue_submissions=0, days_in_compliance=0, total_applicable_rules=0,
            rules_in_compliance=0, rules_in_warning=0, rules_exceeded=0, provenance_hash="",
        )
        status.provenance_hash = hashlib.sha256(
            f"{facility_id}|{unit_id}|{as_of}".encode()).hexdigest()
        return status

    def clear_cache(self, unit_id: Optional[str] = None) -> int:
        """Clear rolling data cache."""
        if unit_id:
            keys = [k for k in self._rolling_cache if k[0] == unit_id]
            for k in keys:
                del self._rolling_cache[k]
            return len(keys)
        count = len(self._rolling_cache)
        self._rolling_cache.clear()
        return count
