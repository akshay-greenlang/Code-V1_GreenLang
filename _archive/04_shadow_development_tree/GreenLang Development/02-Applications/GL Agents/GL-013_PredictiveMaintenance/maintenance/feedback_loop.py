"""GL-013 PredictiveMaintenance - Feedback Loop Module

This module provides closed-loop learning from CMMS outcomes to improve
predictive maintenance model accuracy and alert usefulness.

Example:
    >>> config = FeedbackProcessorConfig()
    >>> processor = FeedbackProcessor(config)
    >>> label = processor.process_closure(closure_data, original_prediction)
    >>> metrics = processor.get_usefulness_metrics()
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CMMSClosureCode(str, Enum):
    """CMMS work order closure codes."""
    COMPLETED_AS_PLANNED = "completed_as_planned"
    DEFECT_FOUND_REPAIRED = "defect_found_repaired"
    NO_DEFECT_FOUND = "no_defect_found"
    DEFERRED = "deferred"
    CANCELLED_FALSE_ALARM = "cancelled_false_alarm"
    CANCELLED_OTHER = "cancelled_other"
    PARTIAL_COMPLETION = "partial_completion"
    ESCALATED = "escalated"
    CONDITION_BETTER_THAN_EXPECTED = "condition_better_than_expected"
    CONDITION_WORSE_THAN_EXPECTED = "condition_worse_than_expected"


class PredictionOutcome(str, Enum):
    """Outcome classification for predictions."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"
    EARLY_WARNING = "early_warning"
    LATE_WARNING = "late_warning"
    INDETERMINATE = "indeterminate"


class LabelConfidence(str, Enum):
    """Confidence level for training labels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class CMMSClosureData(BaseModel):
    """CMMS work order closure data."""
    work_order_id: str = Field(..., description="Work order identifier")
    asset_id: str = Field(..., description="Asset identifier")
    closure_code: CMMSClosureCode = Field(..., description="CMMS closure code")
    closed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    technician_notes: Optional[str] = Field(None, description="Technician findings")
    actual_defect_found: bool = Field(default=False)
    defect_description: Optional[str] = Field(None)
    actual_failure_mode: Optional[str] = Field(None)
    repair_duration_hours: Optional[float] = Field(None, ge=0.0)
    parts_replaced: List[str] = Field(default_factory=list)
    cost_usd: Optional[float] = Field(None, ge=0.0)


class OriginalPrediction(BaseModel):
    """Original prediction that triggered the work order."""
    prediction_id: str = Field(..., description="Prediction identifier")
    asset_id: str = Field(..., description="Asset identifier")
    prediction_timestamp: datetime
    predicted_failure_mode: str
    rul_p10_days: float = Field(..., ge=0.0)
    rul_p50_days: float = Field(..., ge=0.0)
    rul_p90_days: float = Field(..., ge=0.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    recommended_action: str
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TrainingLabel(BaseModel):
    """Training label generated from CMMS feedback."""
    label_id: str = Field(..., description="Unique label identifier")
    asset_id: str
    prediction_id: str
    work_order_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: PredictionOutcome
    label_confidence: LabelConfidence
    actual_rul_days: Optional[float] = Field(None, ge=0.0)
    rul_error_days: Optional[float] = Field(None)
    was_useful_alert: bool = Field(default=True)
    failure_mode_correct: bool = Field(default=True)
    severity_assessment_correct: bool = Field(default=True)
    notes: Optional[str] = Field(None)
    provenance_hash: str


class AlertUsefulnessMetrics(BaseModel):
    """Metrics for alert usefulness tracking."""
    period_start: datetime
    period_end: datetime
    total_alerts: int = Field(default=0, ge=0)
    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    true_negatives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    early_warnings: int = Field(default=0, ge=0)
    late_warnings: int = Field(default=0, ge=0)
    indeterminate: int = Field(default=0, ge=0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    false_alarm_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_rul_error_days: Optional[float] = Field(None)
    rul_error_std_days: Optional[float] = Field(None)
    useful_alert_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_hash: str


class FeedbackProcessorConfig(BaseModel):
    """Configuration for feedback processing."""
    early_warning_threshold_days: float = Field(default=7.0, ge=0.0)
    late_warning_threshold_days: float = Field(default=-3.0, le=0.0)
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    medium_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_auto_labeling: bool = Field(default=True)
    min_samples_for_metrics: int = Field(default=10, ge=1)


class FeedbackProcessor:
    """
    Closed-loop learning processor for predictive maintenance feedback.

    This processor analyzes CMMS work order closures to generate training labels,
    track alert usefulness, and identify false alarms for model improvement.

    Attributes:
        config: Processor configuration

    Example:
        >>> processor = FeedbackProcessor()
        >>> label = processor.process_closure(closure_data, original_prediction)
        >>> metrics = processor.get_usefulness_metrics()
    """

    def __init__(self, config: Optional[FeedbackProcessorConfig] = None):
        """Initialize FeedbackProcessor with configuration."""
        self.config = config or FeedbackProcessorConfig()
        self._label_count = 0
        self._labels: List[TrainingLabel] = []
        self._outcomes: Dict[str, List[PredictionOutcome]] = defaultdict(list)
        self._rul_errors: List[float] = []
        logger.info("FeedbackProcessor initialized")

    def process_closure(self, closure: CMMSClosureData, prediction: OriginalPrediction) -> TrainingLabel:
        """
        Process CMMS closure to generate training label.

        Args:
            closure: CMMS work order closure data
            prediction: Original prediction that triggered the work order

        Returns:
            TrainingLabel for model retraining
        """
        self._label_count += 1
        now = datetime.now(timezone.utc)

        # Determine prediction outcome
        outcome = self._determine_outcome(closure, prediction)

        # Calculate RUL error if defect was found
        actual_rul_days = None
        rul_error_days = None
        if closure.actual_defect_found:
            actual_rul_days = (closure.closed_at - prediction.prediction_timestamp).total_seconds() / 86400.0
            rul_error_days = prediction.rul_p50_days - actual_rul_days
            self._rul_errors.append(rul_error_days)

        # Determine label confidence
        label_confidence = self._determine_confidence(closure, prediction)

        # Check if alert was useful
        was_useful = self._check_alert_usefulness(closure, outcome)

        # Check if failure mode was correct
        fm_correct = self._check_failure_mode(closure, prediction)

        # Check severity assessment
        sev_correct = self._check_severity(closure, prediction)

        # Generate provenance hash
        prov_str = f"{closure.work_order_id}|{prediction.prediction_id}|{outcome.value}|{now.isoformat()}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()

        label = TrainingLabel(
            label_id=f"LABEL-{self._label_count:06d}",
            asset_id=closure.asset_id,
            prediction_id=prediction.prediction_id,
            work_order_id=closure.work_order_id,
            generated_at=now,
            outcome=outcome,
            label_confidence=label_confidence,
            actual_rul_days=actual_rul_days,
            rul_error_days=rul_error_days,
            was_useful_alert=was_useful,
            failure_mode_correct=fm_correct,
            severity_assessment_correct=sev_correct,
            provenance_hash=prov_hash
        )

        self._labels.append(label)
        self._outcomes[closure.asset_id].append(outcome)
        logger.info(f"Generated label {label.label_id}: {outcome.value}")
        return label

    def _determine_outcome(self, closure: CMMSClosureData, prediction: OriginalPrediction) -> PredictionOutcome:
        """Determine prediction outcome from closure data."""
        if closure.closure_code == CMMSClosureCode.CANCELLED_FALSE_ALARM:
            return PredictionOutcome.FALSE_POSITIVE

        if closure.closure_code == CMMSClosureCode.NO_DEFECT_FOUND:
            return PredictionOutcome.FALSE_POSITIVE

        if closure.closure_code in (CMMSClosureCode.DEFECT_FOUND_REPAIRED, CMMSClosureCode.COMPLETED_AS_PLANNED):
            if closure.actual_defect_found:
                # Calculate timing accuracy
                days_since_prediction = (closure.closed_at - prediction.prediction_timestamp).total_seconds() / 86400.0
                timing_error = prediction.rul_p50_days - days_since_prediction

                if timing_error > self.config.early_warning_threshold_days:
                    return PredictionOutcome.EARLY_WARNING
                elif timing_error < self.config.late_warning_threshold_days:
                    return PredictionOutcome.LATE_WARNING
                else:
                    return PredictionOutcome.TRUE_POSITIVE
            else:
                return PredictionOutcome.TRUE_POSITIVE

        if closure.closure_code == CMMSClosureCode.CONDITION_WORSE_THAN_EXPECTED:
            return PredictionOutcome.LATE_WARNING

        if closure.closure_code == CMMSClosureCode.CONDITION_BETTER_THAN_EXPECTED:
            return PredictionOutcome.EARLY_WARNING

        return PredictionOutcome.INDETERMINATE

    def _determine_confidence(self, closure: CMMSClosureData, prediction: OriginalPrediction) -> LabelConfidence:
        """Determine label confidence based on closure data quality."""
        # High confidence if clear outcome with supporting evidence
        if closure.closure_code in (CMMSClosureCode.DEFECT_FOUND_REPAIRED, CMMSClosureCode.NO_DEFECT_FOUND):
            if closure.technician_notes and len(closure.technician_notes) > 20:
                if prediction.confidence_score >= self.config.high_confidence_threshold:
                    return LabelConfidence.HIGH
                return LabelConfidence.MEDIUM

        if closure.closure_code == CMMSClosureCode.CANCELLED_FALSE_ALARM:
            return LabelConfidence.HIGH

        if closure.closure_code in (CMMSClosureCode.DEFERRED, CMMSClosureCode.CANCELLED_OTHER):
            return LabelConfidence.UNCERTAIN

        if prediction.confidence_score >= self.config.medium_confidence_threshold:
            return LabelConfidence.MEDIUM

        return LabelConfidence.LOW

    def _check_alert_usefulness(self, closure: CMMSClosureData, outcome: PredictionOutcome) -> bool:
        """Check if the alert was useful."""
        if outcome == PredictionOutcome.FALSE_POSITIVE:
            return False
        if outcome in (PredictionOutcome.TRUE_POSITIVE, PredictionOutcome.EARLY_WARNING):
            return True
        if outcome == PredictionOutcome.LATE_WARNING:
            # Late warning is still useful if defect was found before failure
            return closure.closure_code != CMMSClosureCode.ESCALATED
        return True

    def _check_failure_mode(self, closure: CMMSClosureData, prediction: OriginalPrediction) -> bool:
        """Check if predicted failure mode was correct."""
        if not closure.actual_failure_mode:
            return True  # Cannot determine, assume correct
        return closure.actual_failure_mode.lower() == prediction.predicted_failure_mode.lower()

    def _check_severity(self, closure: CMMSClosureData, prediction: OriginalPrediction) -> bool:
        """Check if severity assessment was correct."""
        if closure.closure_code == CMMSClosureCode.CONDITION_WORSE_THAN_EXPECTED:
            return False
        if closure.closure_code == CMMSClosureCode.CONDITION_BETTER_THAN_EXPECTED:
            return False
        return True

    def get_usefulness_metrics(self, period_start=None, period_end=None):
        now = datetime.now(timezone.utc)
        if period_end is None: period_end = now
        if period_start is None: period_start = period_end - timedelta(days=30)
        period_labels = [l for l in self._labels if period_start <= l.generated_at <= period_end]
        tp = sum(1 for l in period_labels if l.outcome == PredictionOutcome.TRUE_POSITIVE)
        fp = sum(1 for l in period_labels if l.outcome == PredictionOutcome.FALSE_POSITIVE)
        tn = sum(1 for l in period_labels if l.outcome == PredictionOutcome.TRUE_NEGATIVE)
        fn = sum(1 for l in period_labels if l.outcome == PredictionOutcome.FALSE_NEGATIVE)
        early = sum(1 for l in period_labels if l.outcome == PredictionOutcome.EARLY_WARNING)
        late = sum(1 for l in period_labels if l.outcome == PredictionOutcome.LATE_WARNING)
        indet = sum(1 for l in period_labels if l.outcome == PredictionOutcome.INDETERMINATE)
        total = len(period_labels)
        useful = sum(1 for l in period_labels if l.was_useful_alert)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        false_alarm_rate = fp / total if total > 0 else 0.0
        useful_rate = useful / total if total > 0 else 0.0
        period_rul_errors = [l.rul_error_days for l in period_labels if l.rul_error_days is not None]
        mean_rul_error = None
        rul_error_std = None
        if len(period_rul_errors) >= self.config.min_samples_for_metrics:
            mean_rul_error = sum(period_rul_errors) / len(period_rul_errors)
            variance = sum((e - mean_rul_error) ** 2 for e in period_rul_errors) / len(period_rul_errors)
            rul_error_std = math.sqrt(variance)
        prov_str = f"{period_start.isoformat()}|{period_end.isoformat()}|{total}|{tp}|{fp}"
        prov_hash = hashlib.sha256(prov_str.encode()).hexdigest()
        return AlertUsefulnessMetrics(period_start=period_start, period_end=period_end, total_alerts=total, true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn, early_warnings=early, late_warnings=late, indeterminate=indet, precision=precision, recall=recall, f1_score=f1, false_alarm_rate=false_alarm_rate, mean_rul_error_days=mean_rul_error, rul_error_std_days=rul_error_std, useful_alert_rate=useful_rate, provenance_hash=prov_hash)

    def get_false_alarm_analysis(self, asset_id=None):
        false_alarms = [l for l in self._labels if l.outcome == PredictionOutcome.FALSE_POSITIVE and (asset_id is None or l.asset_id == asset_id)]
        if not false_alarms: return {"total_false_alarms": 0, "rate": 0.0, "by_asset": {}, "recommendations": []}
        total_labels = len([l for l in self._labels if asset_id is None or l.asset_id == asset_id])
        rate = len(false_alarms) / total_labels if total_labels > 0 else 0.0
        by_asset = defaultdict(int)
        for fa in false_alarms: by_asset[fa.asset_id] += 1
        high_fa_assets = [aid for aid, count in by_asset.items() if count >= 3]
        recommendations = []
        if rate > 0.2: recommendations.append("Consider increasing risk threshold")
        if high_fa_assets: recommendations.append(f"Review sensor calibration for: {high_fa_assets}")
        if len(false_alarms) > 10: recommendations.append("Consider retraining model")
        return {"total_false_alarms": len(false_alarms), "rate": rate, "by_asset": dict(by_asset), "high_false_alarm_assets": high_fa_assets, "recommendations": recommendations}

    def get_training_labels_for_export(self, min_confidence=None):
        if min_confidence is None: min_confidence = LabelConfidence.MEDIUM
        confidence_order = {LabelConfidence.HIGH: 3, LabelConfidence.MEDIUM: 2, LabelConfidence.LOW: 1, LabelConfidence.UNCERTAIN: 0}
        min_level = confidence_order[min_confidence]
        eligible_labels = [l for l in self._labels if confidence_order[l.label_confidence] >= min_level]
        export_data = []
        for label in eligible_labels:
            export_data.append({"label_id": label.label_id, "asset_id": label.asset_id, "outcome": label.outcome.value, "actual_rul_days": label.actual_rul_days, "rul_error_days": label.rul_error_days, "failure_mode_correct": label.failure_mode_correct, "confidence": label.label_confidence.value, "timestamp": label.generated_at.isoformat(), "provenance_hash": label.provenance_hash})
        logger.info(f"Exported {len(export_data)} training labels")
        return export_data

    def get_asset_performance_summary(self, asset_id):
        asset_labels = [l for l in self._labels if l.asset_id == asset_id]
        if not asset_labels: return {"asset_id": asset_id, "total_predictions": 0, "message": "No feedback data"}
        outcomes = [l.outcome for l in asset_labels]
        useful_count = sum(1 for l in asset_labels if l.was_useful_alert)
        fm_correct = sum(1 for l in asset_labels if l.failure_mode_correct)
        rul_errors = [l.rul_error_days for l in asset_labels if l.rul_error_days is not None]
        mean_rul_error = sum(rul_errors) / len(rul_errors) if rul_errors else None
        return {"asset_id": asset_id, "total_predictions": len(asset_labels), "true_positives": outcomes.count(PredictionOutcome.TRUE_POSITIVE), "false_positives": outcomes.count(PredictionOutcome.FALSE_POSITIVE), "early_warnings": outcomes.count(PredictionOutcome.EARLY_WARNING), "late_warnings": outcomes.count(PredictionOutcome.LATE_WARNING), "useful_alert_rate": useful_count / len(asset_labels), "failure_mode_accuracy": fm_correct / len(asset_labels), "mean_rul_error_days": mean_rul_error}

    def reset_metrics(self):
        self._labels.clear()
        self._outcomes.clear()
        self._rul_errors.clear()
        self._label_count = 0
        logger.info("FeedbackProcessor metrics reset")
