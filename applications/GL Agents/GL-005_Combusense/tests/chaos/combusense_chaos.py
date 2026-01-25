"""
GL-005 Combusense - Agent-Specific Chaos Engineering Components

This module provides chaos engineering components specific to the
Combusense Emissions Analytics Agent, including:

- CEMS (Continuous Emission Monitoring System) failures
- Regulatory reporting disruptions
- Data quality degradation
- Correlation engine failures
- Predictive model failures

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CombusenseChaosConfig:
    """Configuration for Combusense-specific chaos tests."""

    # CEMS parameters
    cems_data_availability_target: float = 99.5  # Percent
    cems_calibration_drift_max: float = 2.5  # Percent

    # Reporting parameters
    report_submission_deadline_minutes: int = 15
    max_missing_data_hours: int = 24

    # Data quality parameters
    quality_score_threshold: float = 0.9
    outlier_threshold_sigma: float = 3.0

    # Correlation parameters
    correlation_timeout_ms: float = 5000.0
    min_correlation_coefficient: float = 0.7

    # Prediction parameters
    prediction_accuracy_target: float = 95.0
    prediction_horizon_hours: int = 24


class CEMSStatus(Enum):
    """CEMS operational status."""
    OPERATIONAL = "operational"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    OFFLINE = "offline"


class DataQualityLevel(Enum):
    """Data quality levels."""
    VALID = "valid"
    QUESTIONABLE = "questionable"
    SUBSTITUTE = "substitute"
    MISSING = "missing"


class ComplianceStatus(Enum):
    """Regulatory compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    EXCEEDANCE = "exceedance"
    DATA_INSUFFICIENT = "data_insufficient"


# =============================================================================
# CEMS Fault Injector
# =============================================================================

class CEMSFaultInjector:
    """
    Inject faults into Continuous Emission Monitoring System.

    Fault types:
    - Analyzer failure
    - Calibration drift
    - Sample line blockage
    - Communication failure
    - Power interruption

    Example:
        >>> injector = CEMSFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "analyzer_failure",
        ...     "analyzer": "nox_analyzer"
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._analyzers: Dict[str, CEMSStatus] = {
            "nox_analyzer": CEMSStatus.OPERATIONAL,
            "so2_analyzer": CEMSStatus.OPERATIONAL,
            "co_analyzer": CEMSStatus.OPERATIONAL,
            "o2_analyzer": CEMSStatus.OPERATIONAL,
            "flow_monitor": CEMSStatus.OPERATIONAL,
            "opacity_monitor": CEMSStatus.OPERATIONAL,
        }
        self._data_availability: Dict[str, float] = {
            analyzer: 100.0 for analyzer in self._analyzers
        }

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject CEMS fault."""
        try:
            self._fault_type = params.get("fault_type", "analyzer_failure")
            self._params = params

            analyzer = params.get("analyzer", "nox_analyzer")

            if self._fault_type == "analyzer_failure":
                self._analyzers[analyzer] = CEMSStatus.FAULT
                self._data_availability[analyzer] = 0.0

            elif self._fault_type == "calibration_drift":
                self._analyzers[analyzer] = CEMSStatus.CALIBRATING

            elif self._fault_type == "communication_failure":
                for a in self._analyzers:
                    self._analyzers[a] = CEMSStatus.OFFLINE

            logger.info(f"CEMSFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"CEMSFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove CEMS fault."""
        try:
            logger.info("CEMSFaultInjector: Rolling back")
            self._active = False
            for analyzer in self._analyzers:
                self._analyzers[analyzer] = CEMSStatus.OPERATIONAL
                self._data_availability[analyzer] = 100.0
            return True

        except Exception as e:
            logger.error(f"CEMSFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_analyzer_status(self, analyzer: str) -> CEMSStatus:
        return self._analyzers.get(analyzer, CEMSStatus.OFFLINE)

    def get_data_availability(self, analyzer: str) -> float:
        return self._data_availability.get(analyzer, 0.0)

    def get_reading(self, analyzer: str) -> Dict[str, Any]:
        """Get analyzer reading with potential faults."""
        status = self._analyzers.get(analyzer, CEMSStatus.OFFLINE)

        if status == CEMSStatus.OFFLINE or status == CEMSStatus.FAULT:
            return {"value": None, "status": status.value, "quality": "invalid"}

        if status == CEMSStatus.CALIBRATING:
            return {"value": None, "status": status.value, "quality": "calibrating"}

        # Normal reading with possible drift
        base_values = {
            "nox_analyzer": 30.0,
            "so2_analyzer": 15.0,
            "co_analyzer": 50.0,
            "o2_analyzer": 3.0,
            "flow_monitor": 10000.0,
            "opacity_monitor": 5.0,
        }

        base = base_values.get(analyzer, 0)

        if self._fault_type == "calibration_drift" and self._active:
            drift = self._params.get("drift_percent", 5) / 100.0
            value = base * (1 + drift)
        else:
            value = base + random.uniform(-base * 0.05, base * 0.05)

        return {"value": value, "status": "operational", "quality": "valid"}


# =============================================================================
# Regulatory Reporting Fault Injector
# =============================================================================

class RegulatoryReportingFaultInjector:
    """
    Inject faults into regulatory reporting system.

    Fault types:
    - Submission failure
    - Data validation error
    - Report generation timeout
    - XML schema error
    - Credential expiration

    Example:
        >>> injector = RegulatoryReportingFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "submission_failure",
        ...     "failure_reason": "network_timeout"
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._pending_reports: List[Dict[str, Any]] = []

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject regulatory reporting fault."""
        try:
            self._fault_type = params.get("fault_type", "submission_failure")
            self._params = params

            logger.info(f"RegulatoryReportingFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"RegulatoryReportingFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove regulatory reporting fault."""
        try:
            logger.info("RegulatoryReportingFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"RegulatoryReportingFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    async def submit_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit regulatory report with potential faults."""
        if not self._active:
            return {
                "status": "submitted",
                "confirmation_id": f"CONF-{random.randint(10000, 99999)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if self._fault_type == "submission_failure":
            reason = self._params.get("failure_reason", "network_timeout")
            return {"status": "failed", "error": reason}

        elif self._fault_type == "validation_error":
            return {
                "status": "rejected",
                "error": "data_validation_failed",
                "details": ["missing_required_field", "value_out_of_range"],
            }

        elif self._fault_type == "timeout":
            timeout_ms = self._params.get("timeout_ms", 30000)
            await asyncio.sleep(timeout_ms / 1000.0)
            return {"status": "timeout"}

        elif self._fault_type == "schema_error":
            return {
                "status": "rejected",
                "error": "xml_schema_validation_failed",
            }

        elif self._fault_type == "credential_expired":
            return {
                "status": "unauthorized",
                "error": "credentials_expired",
            }

        return {"status": "unknown_error"}

    async def generate_report(self, report_type: str, period: str) -> Dict[str, Any]:
        """Generate regulatory report."""
        if not self._active:
            return {
                "status": "generated",
                "report_type": report_type,
                "period": period,
                "data_availability": 99.5,
            }

        if self._fault_type == "generation_timeout":
            await asyncio.sleep(10)
            return {"status": "timeout"}

        return {
            "status": "generated",
            "report_type": report_type,
            "period": period,
            "data_availability": 85.0,  # Degraded
        }


# =============================================================================
# Data Quality Fault Injector
# =============================================================================

class DataQualityFaultInjector:
    """
    Inject data quality issues.

    Fault types:
    - Missing data
    - Outliers
    - Timestamp errors
    - Duplicate records
    - Corrupted values

    Example:
        >>> injector = DataQualityFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "missing_data",
        ...     "missing_percent": 10
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject data quality fault."""
        try:
            self._fault_type = params.get("fault_type", "missing_data")
            self._params = params

            logger.info(f"DataQualityFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"DataQualityFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove data quality fault."""
        try:
            logger.info("DataQualityFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"DataQualityFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def process_data_point(self, value: float, timestamp: datetime) -> Dict[str, Any]:
        """Process data point with potential quality issues."""
        if not self._active:
            return {
                "value": value,
                "timestamp": timestamp,
                "quality": DataQualityLevel.VALID.value,
            }

        if self._fault_type == "missing_data":
            missing_percent = self._params.get("missing_percent", 10)
            if random.random() * 100 < missing_percent:
                return {
                    "value": None,
                    "timestamp": timestamp,
                    "quality": DataQualityLevel.MISSING.value,
                }

        elif self._fault_type == "outliers":
            outlier_percent = self._params.get("outlier_percent", 5)
            if random.random() * 100 < outlier_percent:
                multiplier = random.choice([5, 10, -5])
                return {
                    "value": value * multiplier,
                    "timestamp": timestamp,
                    "quality": DataQualityLevel.QUESTIONABLE.value,
                    "flag": "outlier_detected",
                }

        elif self._fault_type == "timestamp_error":
            # Shift timestamp randomly
            shift = timedelta(minutes=random.randint(-60, 60))
            return {
                "value": value,
                "timestamp": timestamp + shift,
                "quality": DataQualityLevel.QUESTIONABLE.value,
                "flag": "timestamp_suspect",
            }

        elif self._fault_type == "duplicate":
            return {
                "value": value,
                "timestamp": timestamp,
                "quality": DataQualityLevel.VALID.value,
                "flag": "duplicate_record",
            }

        elif self._fault_type == "corrupted":
            return {
                "value": float('nan'),
                "timestamp": timestamp,
                "quality": DataQualityLevel.SUBSTITUTE.value,
            }

        return {
            "value": value,
            "timestamp": timestamp,
            "quality": DataQualityLevel.VALID.value,
        }

    def get_quality_score(self, data_points: List[Dict[str, Any]]) -> float:
        """Calculate overall data quality score."""
        if not data_points:
            return 0.0

        valid_count = sum(
            1 for dp in data_points
            if dp.get("quality") == DataQualityLevel.VALID.value
        )

        return valid_count / len(data_points)


# =============================================================================
# Correlation Engine Fault Injector
# =============================================================================

class CorrelationEngineFaultInjector:
    """
    Inject faults into emission correlation engine.

    Fault types:
    - Calculation timeout
    - Memory overflow
    - Invalid correlation
    - Data alignment error
    - Historical data unavailable

    Example:
        >>> injector = CorrelationEngineFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "calculation_timeout",
        ...     "timeout_ms": 5000
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject correlation engine fault."""
        try:
            self._fault_type = params.get("fault_type", "calculation_timeout")
            self._params = params

            logger.info(f"CorrelationEngineFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"CorrelationEngineFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove correlation engine fault."""
        try:
            logger.info("CorrelationEngineFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"CorrelationEngineFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    async def calculate_correlation(
        self,
        series_a: List[float],
        series_b: List[float]
    ) -> Dict[str, Any]:
        """Calculate correlation with potential faults."""
        if not self._active:
            # Normal correlation calculation
            import statistics
            if len(series_a) != len(series_b) or len(series_a) < 2:
                return {"status": "error", "error": "insufficient_data"}

            try:
                correlation = self._pearson_correlation(series_a, series_b)
                return {
                    "status": "success",
                    "correlation": correlation,
                    "samples": len(series_a),
                }
            except Exception:
                return {"status": "error", "error": "calculation_failed"}

        if self._fault_type == "calculation_timeout":
            timeout_ms = self._params.get("timeout_ms", 5000)
            await asyncio.sleep(timeout_ms / 1000.0)
            return {"status": "timeout"}

        elif self._fault_type == "memory_overflow":
            return {"status": "error", "error": "memory_overflow"}

        elif self._fault_type == "invalid_result":
            return {
                "status": "warning",
                "correlation": float('nan'),
                "error": "invalid_correlation_result",
            }

        elif self._fault_type == "alignment_error":
            return {"status": "error", "error": "data_alignment_mismatch"}

        elif self._fault_type == "historical_unavailable":
            return {"status": "error", "error": "historical_data_not_available"}

        return {"status": "unknown_error"}

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator


# =============================================================================
# Predictive Model Fault Injector
# =============================================================================

class PredictiveModelFaultInjector:
    """
    Inject faults into emission prediction models.

    Fault types:
    - Model timeout
    - Poor accuracy
    - Feature unavailable
    - Model version mismatch
    - Prediction out of bounds

    Example:
        >>> injector = PredictiveModelFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "poor_accuracy",
        ...     "error_margin": 30
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject predictive model fault."""
        try:
            self._fault_type = params.get("fault_type", "model_timeout")
            self._params = params

            logger.info(f"PredictiveModelFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"PredictiveModelFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove predictive model fault."""
        try:
            logger.info("PredictiveModelFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"PredictiveModelFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    async def predict(self, features: Dict[str, float], horizon_hours: int) -> Dict[str, Any]:
        """Generate prediction with potential faults."""
        if not self._active:
            return {
                "status": "success",
                "predictions": [
                    {"hour": h, "value": 30 + random.uniform(-5, 5), "confidence": 0.9}
                    for h in range(horizon_hours)
                ],
                "accuracy": 95.0,
            }

        if self._fault_type == "model_timeout":
            timeout_ms = self._params.get("timeout_ms", 10000)
            await asyncio.sleep(timeout_ms / 1000.0)
            return {"status": "timeout"}

        elif self._fault_type == "poor_accuracy":
            error_margin = self._params.get("error_margin", 30)
            return {
                "status": "degraded",
                "predictions": [
                    {
                        "hour": h,
                        "value": 30 + random.uniform(-error_margin, error_margin),
                        "confidence": 0.5
                    }
                    for h in range(horizon_hours)
                ],
                "accuracy": 65.0,
                "warning": "model_accuracy_degraded",
            }

        elif self._fault_type == "feature_unavailable":
            missing = self._params.get("missing_feature", "temperature")
            return {
                "status": "error",
                "error": f"missing_feature_{missing}",
            }

        elif self._fault_type == "version_mismatch":
            return {
                "status": "error",
                "error": "model_version_mismatch",
            }

        elif self._fault_type == "out_of_bounds":
            return {
                "status": "warning",
                "predictions": [
                    {"hour": h, "value": -100 if h % 2 == 0 else 1000, "confidence": 0.3}
                    for h in range(horizon_hours)
                ],
                "warning": "predictions_out_of_valid_range",
            }

        return {"status": "unknown_error"}


# =============================================================================
# Steady State Hypothesis for Combusense
# =============================================================================

def create_combusense_hypothesis():
    """Create steady state hypothesis specific to Combusense agent."""
    import sys
    import os

    gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
    if gl001_path not in sys.path:
        sys.path.insert(0, gl001_path)

    from steady_state import SteadyStateHypothesis, SteadyStateMetric, ComparisonOperator

    return SteadyStateHypothesis(
        name="Combusense Emissions Analytics Health",
        description="Validates Combusense agent is operating normally",
        metrics=[
            SteadyStateMetric(
                name="cems_availability_percent",
                description="CEMS data availability",
                threshold=99.0,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
            SteadyStateMetric(
                name="data_quality_score",
                description="Overall data quality score",
                threshold=0.9,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
            SteadyStateMetric(
                name="report_submission_success_rate",
                description="Regulatory report submission success rate",
                threshold=99.0,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
            SteadyStateMetric(
                name="prediction_accuracy_percent",
                description="Emission prediction accuracy",
                threshold=90.0,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=False,
            ),
            SteadyStateMetric(
                name="correlation_calculation_time_ms",
                description="Correlation calculation latency",
                threshold=1000,
                operator=ComparisonOperator.LESS_THAN,
                required=False,
            ),
        ],
        pass_threshold=0.8,
        aggregation="weighted",
    )
