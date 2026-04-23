# -*- coding: utf-8 -*-
"""
Anomaly Detection Module for GL-005 COMBUSENSE

Implements real-time anomaly detection with standardized event taxonomy
as specified in GL-005 Playbook Section 9.

Detection Layers (per Section 9.2):
    1. Deterministic rule checks (boundary violations, limits)
    2. Statistical change detection (CUSUM, EWMA)
    3. ML-based multivariate anomaly scoring (optional)

Anomaly Taxonomy (per Section 9.1):
    - COMBUSTION_RICH: Low O2 with elevated CO
    - COMBUSTION_LEAN: High O2/excess air
    - CO_SPIKE: Transient CO excursion
    - NOX_SPIKE: NOx threshold breach
    - FLAME_INSTABILITY: Flame variability/flicker
    - FLAME_LOSS: Flame scanner dropout
    - SENSOR_DRIFT: Analyzer drift detected
    - SENSOR_FAULT: Sensor failure/invalid
    - INTERLOCK_BYPASS: Safety bypass active
    - MODE_ANOMALY: Unexpected operating mode
    - EFFICIENCY_DEGRADATION: Performance trending down
    - AIR_FUEL_IMBALANCE: AFR outside optimal range

Reference: GL-005 Playbook Section 9 (Real-Time Anomaly Detection)

Author: GreenLang GL-005 Team
Version: 1.0.0
Performance Target: <10ms per detection cycle
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Severity level definitions (S1-S4)
SEVERITY_LEVELS = {
    "S1": {"name": "Informational", "priority": 1, "color": "blue"},
    "S2": {"name": "Warning", "priority": 2, "color": "yellow"},
    "S3": {"name": "Urgent", "priority": 3, "color": "orange"},
    "S4": {"name": "Critical", "priority": 4, "color": "red"},
}

# Default detection thresholds
DEFAULT_THRESHOLDS = {
    # Emissions thresholds
    "co_warning_ppm": 100,
    "co_critical_ppm": 200,
    "nox_warning_ppm": 50,
    "nox_critical_ppm": 100,

    # O2 thresholds
    "o2_low_lean": 1.5,
    "o2_high_lean": 6.0,
    "o2_target": 3.0,

    # Flame thresholds
    "flame_intensity_min": 30,
    "flame_variability_max": 15,

    # AFR thresholds
    "afr_low": 14.0,
    "afr_high": 20.0,

    # Hysteresis
    "min_duration_seconds": 10,
    "cooldown_seconds": 60,
}

# CUSUM parameters
CUSUM_K = 0.5  # Slack parameter
CUSUM_H = 5.0  # Decision threshold


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AnomalyType(str, Enum):
    """Standardized anomaly event types (Section 9.1)"""
    COMBUSTION_RICH = "COMBUSTION_RICH"
    COMBUSTION_LEAN = "COMBUSTION_LEAN"
    CO_SPIKE = "CO_SPIKE"
    NOX_SPIKE = "NOX_SPIKE"
    FLAME_INSTABILITY = "FLAME_INSTABILITY"
    FLAME_LOSS = "FLAME_LOSS"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    SENSOR_FAULT = "SENSOR_FAULT"
    INTERLOCK_BYPASS = "INTERLOCK_BYPASS"
    MODE_ANOMALY = "MODE_ANOMALY"
    EFFICIENCY_DEGRADATION = "EFFICIENCY_DEGRADATION"
    AIR_FUEL_IMBALANCE = "AIR_FUEL_IMBALANCE"
    THERMAL_NOX_EXCESSIVE = "THERMAL_NOX_EXCESSIVE"
    INCOMPLETE_COMBUSTION = "INCOMPLETE_COMBUSTION"
    FURNACE_PRESSURE_ANOMALY = "FURNACE_PRESSURE_ANOMALY"
    LOAD_TRANSIENT = "LOAD_TRANSIENT"


class Severity(str, Enum):
    """Severity levels (S1-S4)"""
    S1 = "S1"  # Informational
    S2 = "S2"  # Warning
    S3 = "S3"  # Urgent
    S4 = "S4"  # Critical


class DetectionLayer(str, Enum):
    """Detection layer that triggered the anomaly"""
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"


class AnomalyStatus(str, Enum):
    """Anomaly lifecycle status"""
    ACTIVE = "active"
    UPDATING = "updating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


# =============================================================================
# ANOMALY TAXONOMY DEFINITIONS
# =============================================================================

ANOMALY_TAXONOMY: Dict[AnomalyType, Dict[str, Any]] = {
    AnomalyType.COMBUSTION_RICH: {
        "description": "Low O2 with elevated CO, possible fuel-rich operation, risk of CO excursions and instability",
        "default_severity": Severity.S3,
        "affected_cqi_components": ["emissions", "safety"],
        "typical_causes": [
            "Air flow restriction",
            "Fuel metering error",
            "AFR control fault",
            "Damper malfunction",
        ],
        "recommended_checks": [
            "Check air damper positions",
            "Verify FD fan operation",
            "Review AFR controller output",
            "Inspect fuel flow measurement",
        ],
    },
    AnomalyType.COMBUSTION_LEAN: {
        "description": "High O2/excess air, efficiency loss, possible flame instability at low load",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["efficiency"],
        "typical_causes": [
            "Air in-leakage",
            "O2 trim over-correction",
            "Load reduction",
            "Damper control issue",
        ],
        "recommended_checks": [
            "Inspect for air in-leakage",
            "Review O2 trim setpoint",
            "Check damper calibration",
            "Verify load demand signal",
        ],
    },
    AnomalyType.CO_SPIKE: {
        "description": "Transient CO spike beyond engineered threshold, may indicate poor mixing or ignition issues",
        "default_severity": Severity.S3,
        "affected_cqi_components": ["emissions", "safety"],
        "typical_causes": [
            "Incomplete combustion",
            "Air shortage",
            "Fuel quality change",
            "Burner fouling",
        ],
        "recommended_checks": [
            "Verify O2/CO analyzer status",
            "Inspect burner tips",
            "Check air-fuel ratio",
            "Review recent load changes",
        ],
    },
    AnomalyType.NOX_SPIKE: {
        "description": "NOx threshold breach, thermal or fuel-bound NOx formation",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["emissions"],
        "typical_causes": [
            "High flame temperature",
            "Excess air imbalance",
            "Load increase",
            "Air staging issue",
        ],
        "recommended_checks": [
            "Check flame temperature",
            "Review air staging settings",
            "Verify load transition rate",
            "Inspect burner pattern",
        ],
    },
    AnomalyType.FLAME_INSTABILITY: {
        "description": "Flame variability, pulsation, or pattern degradation",
        "default_severity": Severity.S3,
        "affected_cqi_components": ["stability", "safety"],
        "typical_causes": [
            "Fuel pressure fluctuation",
            "Air/fuel mixing issue",
            "Burner register mismatch",
            "Low load operation",
        ],
        "recommended_checks": [
            "Check fuel supply pressure",
            "Inspect burner registers",
            "Review flame scanner signals",
            "Verify load vs. minimum firing rate",
        ],
    },
    AnomalyType.FLAME_LOSS: {
        "description": "Flame scanner indicates no flame or flame dropout",
        "default_severity": Severity.S4,
        "affected_cqi_components": ["stability", "safety"],
        "typical_causes": [
            "Actual flameout",
            "Flame scanner fault",
            "Fuel supply interruption",
            "Air flow surge",
        ],
        "recommended_checks": [
            "VERIFY FLAME STATUS IMMEDIATELY",
            "Check flame scanner health",
            "Verify fuel supply",
            "Check BMS status",
        ],
    },
    AnomalyType.SENSOR_DRIFT: {
        "description": "Analyzer or sensor showing drift from expected baseline",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["data"],
        "typical_causes": [
            "Calibration drift",
            "Sample line contamination",
            "Sensor aging",
            "Environmental factors",
        ],
        "recommended_checks": [
            "Schedule sensor calibration",
            "Check sample conditioning",
            "Review calibration history",
            "Compare with redundant sensors",
        ],
    },
    AnomalyType.SENSOR_FAULT: {
        "description": "Sensor reporting invalid data or fault status",
        "default_severity": Severity.S3,
        "affected_cqi_components": ["data"],
        "typical_causes": [
            "Sensor failure",
            "Communication loss",
            "Power issue",
            "Wiring fault",
        ],
        "recommended_checks": [
            "Check sensor status codes",
            "Verify communications",
            "Inspect wiring",
            "Replace if necessary",
        ],
    },
    AnomalyType.INTERLOCK_BYPASS: {
        "description": "Safety interlock bypass is active",
        "default_severity": Severity.S3,
        "affected_cqi_components": ["safety"],
        "typical_causes": [
            "Maintenance activity",
            "Testing",
            "Operational override",
            "Fault recovery",
        ],
        "recommended_checks": [
            "Verify bypass authorization",
            "Document bypass reason",
            "Set time limit per procedures",
            "Monitor affected systems",
        ],
    },
    AnomalyType.MODE_ANOMALY: {
        "description": "Unexpected operating mode or state transition",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["safety"],
        "typical_causes": [
            "Control system fault",
            "Unexpected trip",
            "Operator action",
            "Sequence logic issue",
        ],
        "recommended_checks": [
            "Review BMS state machine",
            "Check trip history",
            "Verify sequence logic",
            "Contact operations",
        ],
    },
    AnomalyType.EFFICIENCY_DEGRADATION: {
        "description": "Combustion efficiency trending below baseline",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["efficiency"],
        "typical_causes": [
            "Heat exchanger fouling",
            "Air in-leakage",
            "Burner degradation",
            "Control drift",
        ],
        "recommended_checks": [
            "Review efficiency trend",
            "Schedule heat exchanger inspection",
            "Check for air leaks",
            "Review tuning parameters",
        ],
    },
    AnomalyType.AIR_FUEL_IMBALANCE: {
        "description": "Air-fuel ratio outside optimal operating band",
        "default_severity": Severity.S2,
        "affected_cqi_components": ["efficiency", "emissions"],
        "typical_causes": [
            "Flow measurement error",
            "Control valve issue",
            "Fuel quality change",
            "Air system imbalance",
        ],
        "recommended_checks": [
            "Verify flow measurements",
            "Check control valve positions",
            "Review fuel analysis",
            "Balance air distribution",
        ],
    },
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event"""
    incident_id: str
    anomaly_type: AnomalyType
    severity: Severity
    confidence: float  # 0.0 to 1.0
    status: AnomalyStatus
    detection_layer: DetectionLayer

    # Timestamps
    start_time: datetime
    last_update_time: datetime
    end_time: Optional[datetime] = None

    # Context
    asset_id: str = "default"
    affected_signals: List[str] = field(default_factory=list)
    trigger_values: Dict[str, float] = field(default_factory=dict)
    threshold_values: Dict[str, float] = field(default_factory=dict)

    # CQI impact
    cqi_impact: float = 0.0
    affected_cqi_components: List[str] = field(default_factory=list)

    # Explanation
    description: str = ""
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    recommended_checks: List[str] = field(default_factory=list)

    # Metadata
    suppression_reason: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "incident_id": self.incident_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "detection_layer": self.detection_layer.value,
            "start_time": self.start_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "asset_id": self.asset_id,
            "affected_signals": self.affected_signals,
            "trigger_values": self.trigger_values,
            "cqi_impact": self.cqi_impact,
            "description": self.description,
            "top_drivers": self.top_drivers,
            "recommended_checks": self.recommended_checks,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class DetectionResult:
    """Result from a detection layer"""
    detected: bool
    anomaly_type: Optional[AnomalyType]
    severity: Optional[Severity]
    confidence: float
    trigger_values: Dict[str, float]
    description: str


# =============================================================================
# INPUT MODELS
# =============================================================================

class CombustionState(BaseModel):
    """Current combustion state for anomaly detection"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    asset_id: str = "default"

    # Core signals
    o2_percent: float = Field(default=3.0, ge=0, le=21)
    co_ppm: float = Field(default=50.0, ge=0)
    nox_ppm: float = Field(default=30.0, ge=0)
    fuel_flow_kg_s: float = Field(default=1.0, ge=0)
    air_flow_kg_s: float = Field(default=17.2, ge=0)

    # Flame signals
    flame_intensity: float = Field(default=80.0, ge=0, le=100)
    flame_present: bool = True

    # Furnace conditions
    furnace_temp_c: float = Field(default=800.0, ge=0)
    furnace_pressure_pa: float = Field(default=101325.0, ge=0)

    # Safety status
    interlocks_healthy: bool = True
    bypass_active: bool = False

    # Operating context
    operating_mode: str = "RUN"
    load_percent: float = Field(default=75.0, ge=0, le=100)

    # Signal quality
    o2_valid: bool = True
    co_valid: bool = True
    nox_valid: bool = True


class DetectionConfig(BaseModel):
    """Configuration for anomaly detection"""
    # Emissions thresholds
    co_warning_ppm: float = 100.0
    co_critical_ppm: float = 200.0
    nox_warning_ppm: float = 50.0
    nox_critical_ppm: float = 100.0

    # O2 thresholds
    o2_low_rich: float = 1.5
    o2_high_lean: float = 6.0
    o2_target: float = 3.0

    # Flame thresholds
    flame_intensity_min: float = 30.0
    flame_variability_max: float = 15.0

    # AFR thresholds
    afr_low: float = 14.0
    afr_high: float = 20.0
    afr_stoich: float = 17.2

    # Timing
    min_duration_seconds: float = 10.0
    cooldown_seconds: float = 60.0
    quiet_modes: List[str] = Field(default_factory=lambda: ["PURGE", "IGNITE", "SHUTDOWN"])

    # Hysteresis
    hysteresis_percent: float = 5.0


# =============================================================================
# RULE-BASED DETECTOR
# =============================================================================

class RuleBasedDetector:
    """
    Layer 1: Deterministic rule-based detection

    Implements threshold and limit checks per playbook Section 9.2.
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize rule-based detector

        Args:
            config: Detection configuration
        """
        self.config = config

    def detect(self, state: CombustionState) -> List[DetectionResult]:
        """
        Run all rule-based detection checks

        Args:
            state: Current combustion state

        Returns:
            List of detection results
        """
        results = []

        # Skip detection in quiet modes
        if state.operating_mode.upper() in [m.upper() for m in self.config.quiet_modes]:
            return results

        # CO excursion check
        results.append(self._check_co_excursion(state))

        # NOx excursion check
        results.append(self._check_nox_excursion(state))

        # Rich combustion check
        results.append(self._check_rich_combustion(state))

        # Lean combustion check
        results.append(self._check_lean_combustion(state))

        # Flame checks
        results.append(self._check_flame_loss(state))
        results.append(self._check_flame_instability(state))

        # Sensor validity checks
        results.append(self._check_sensor_faults(state))

        # Interlock bypass check
        results.append(self._check_interlock_bypass(state))

        # AFR check
        results.append(self._check_afr_imbalance(state))

        # Filter to detected anomalies only
        return [r for r in results if r.detected]

    def _check_co_excursion(self, state: CombustionState) -> DetectionResult:
        """Check for CO spike"""
        if not state.co_valid:
            return DetectionResult(False, None, None, 0.0, {}, "")

        if state.co_ppm >= self.config.co_critical_ppm:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.CO_SPIKE,
                severity=Severity.S4,
                confidence=0.95,
                trigger_values={"co_ppm": state.co_ppm},
                description=f"Critical CO spike: {state.co_ppm:.0f} ppm exceeds {self.config.co_critical_ppm:.0f} ppm threshold"
            )
        elif state.co_ppm >= self.config.co_warning_ppm:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.CO_SPIKE,
                severity=Severity.S3,
                confidence=0.85,
                trigger_values={"co_ppm": state.co_ppm},
                description=f"CO elevated: {state.co_ppm:.0f} ppm exceeds {self.config.co_warning_ppm:.0f} ppm warning threshold"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_nox_excursion(self, state: CombustionState) -> DetectionResult:
        """Check for NOx spike"""
        if not state.nox_valid:
            return DetectionResult(False, None, None, 0.0, {}, "")

        if state.nox_ppm >= self.config.nox_critical_ppm:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.NOX_SPIKE,
                severity=Severity.S3,
                confidence=0.90,
                trigger_values={"nox_ppm": state.nox_ppm},
                description=f"NOx elevated: {state.nox_ppm:.0f} ppm exceeds {self.config.nox_critical_ppm:.0f} ppm threshold"
            )
        elif state.nox_ppm >= self.config.nox_warning_ppm:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.NOX_SPIKE,
                severity=Severity.S2,
                confidence=0.80,
                trigger_values={"nox_ppm": state.nox_ppm},
                description=f"NOx warning: {state.nox_ppm:.0f} ppm approaching limit"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_rich_combustion(self, state: CombustionState) -> DetectionResult:
        """Check for rich combustion condition"""
        if not state.o2_valid:
            return DetectionResult(False, None, None, 0.0, {}, "")

        # Rich: low O2 + elevated CO
        if state.o2_percent < self.config.o2_low_rich and state.co_ppm > 50:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.COMBUSTION_RICH,
                severity=Severity.S3,
                confidence=0.90,
                trigger_values={
                    "o2_percent": state.o2_percent,
                    "co_ppm": state.co_ppm
                },
                description=f"Rich combustion: O2 at {state.o2_percent:.1f}% with CO at {state.co_ppm:.0f} ppm"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_lean_combustion(self, state: CombustionState) -> DetectionResult:
        """Check for lean combustion condition"""
        if not state.o2_valid:
            return DetectionResult(False, None, None, 0.0, {}, "")

        if state.o2_percent > self.config.o2_high_lean:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.COMBUSTION_LEAN,
                severity=Severity.S2,
                confidence=0.85,
                trigger_values={"o2_percent": state.o2_percent},
                description=f"Lean combustion: O2 at {state.o2_percent:.1f}% indicates excess air"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_flame_loss(self, state: CombustionState) -> DetectionResult:
        """Check for flame loss"""
        if not state.flame_present and state.operating_mode.upper() == "RUN":
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.FLAME_LOSS,
                severity=Severity.S4,
                confidence=0.99,
                trigger_values={"flame_present": 0},
                description="FLAME LOSS DETECTED - Flame scanner indicates no flame during RUN mode"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_flame_instability(self, state: CombustionState) -> DetectionResult:
        """Check for flame instability"""
        if state.flame_intensity < self.config.flame_intensity_min and state.flame_present:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.FLAME_INSTABILITY,
                severity=Severity.S3,
                confidence=0.80,
                trigger_values={"flame_intensity": state.flame_intensity},
                description=f"Low flame intensity: {state.flame_intensity:.0f}% below {self.config.flame_intensity_min:.0f}% minimum"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_sensor_faults(self, state: CombustionState) -> DetectionResult:
        """Check for sensor faults"""
        invalid_sensors = []
        if not state.o2_valid:
            invalid_sensors.append("O2")
        if not state.co_valid:
            invalid_sensors.append("CO")
        if not state.nox_valid:
            invalid_sensors.append("NOx")

        if invalid_sensors:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.SENSOR_FAULT,
                severity=Severity.S3,
                confidence=0.95,
                trigger_values={f"{s}_valid": False for s in invalid_sensors},
                description=f"Sensor fault detected: {', '.join(invalid_sensors)} analyzer(s) reporting invalid"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_interlock_bypass(self, state: CombustionState) -> DetectionResult:
        """Check for active interlock bypass"""
        if state.bypass_active:
            return DetectionResult(
                detected=True,
                anomaly_type=AnomalyType.INTERLOCK_BYPASS,
                severity=Severity.S3,
                confidence=1.0,
                trigger_values={"bypass_active": True},
                description="Safety interlock bypass is active"
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _check_afr_imbalance(self, state: CombustionState) -> DetectionResult:
        """Check for air-fuel ratio imbalance"""
        if state.fuel_flow_kg_s > 0:
            afr = state.air_flow_kg_s / state.fuel_flow_kg_s

            if afr < self.config.afr_low:
                return DetectionResult(
                    detected=True,
                    anomaly_type=AnomalyType.AIR_FUEL_IMBALANCE,
                    severity=Severity.S3,
                    confidence=0.85,
                    trigger_values={"afr": afr},
                    description=f"Low AFR: {afr:.1f} below {self.config.afr_low:.1f} minimum (fuel-rich risk)"
                )
            elif afr > self.config.afr_high:
                return DetectionResult(
                    detected=True,
                    anomaly_type=AnomalyType.AIR_FUEL_IMBALANCE,
                    severity=Severity.S2,
                    confidence=0.80,
                    trigger_values={"afr": afr},
                    description=f"High AFR: {afr:.1f} above {self.config.afr_high:.1f} maximum (excess air)"
                )

        return DetectionResult(False, None, None, 0.0, {}, "")


# =============================================================================
# STATISTICAL DETECTOR
# =============================================================================

class StatisticalDetector:
    """
    Layer 2: Statistical change detection

    Implements CUSUM and EWMA-based change point detection.
    """

    def __init__(
        self,
        window_size: int = 60,
        cusum_k: float = CUSUM_K,
        cusum_h: float = CUSUM_H
    ):
        """
        Initialize statistical detector

        Args:
            window_size: Number of samples for baseline
            cusum_k: CUSUM slack parameter
            cusum_h: CUSUM decision threshold
        """
        self.window_size = window_size
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        # History buffers
        self.o2_history: Deque[float] = deque(maxlen=window_size)
        self.co_history: Deque[float] = deque(maxlen=window_size)
        self.nox_history: Deque[float] = deque(maxlen=window_size)
        self.flame_history: Deque[float] = deque(maxlen=window_size)

        # CUSUM accumulators
        self.cusum_pos: Dict[str, float] = {}
        self.cusum_neg: Dict[str, float] = {}

    def detect(self, state: CombustionState) -> List[DetectionResult]:
        """
        Run statistical detection

        Args:
            state: Current combustion state

        Returns:
            List of detection results
        """
        results = []

        # Update histories
        if state.o2_valid:
            self.o2_history.append(state.o2_percent)
        if state.co_valid:
            self.co_history.append(state.co_ppm)
        if state.nox_valid:
            self.nox_history.append(state.nox_ppm)
        self.flame_history.append(state.flame_intensity)

        # Need sufficient history
        if len(self.o2_history) < 10:
            return results

        # CUSUM on O2 for efficiency degradation
        o2_result = self._cusum_detect("o2", state.o2_percent, self.o2_history)
        if o2_result.detected:
            results.append(o2_result)

        # CUSUM on flame for instability
        flame_result = self._cusum_detect("flame", state.flame_intensity, self.flame_history)
        if flame_result.detected:
            results.append(flame_result)

        return results

    def _cusum_detect(
        self, signal_name: str, current_value: float, history: Deque[float]
    ) -> DetectionResult:
        """
        CUSUM change detection

        Args:
            signal_name: Name of signal
            current_value: Current value
            history: Historical values

        Returns:
            DetectionResult
        """
        if len(history) < 10:
            return DetectionResult(False, None, None, 0.0, {}, "")

        # Calculate baseline mean and std
        baseline = list(history)[:-1]  # Exclude current
        mean = sum(baseline) / len(baseline)
        std = math.sqrt(sum((x - mean) ** 2 for x in baseline) / len(baseline))

        if std < 0.001:
            return DetectionResult(False, None, None, 0.0, {}, "")

        # Standardized deviation
        z = (current_value - mean) / std

        # Update CUSUM accumulators
        key = signal_name
        if key not in self.cusum_pos:
            self.cusum_pos[key] = 0.0
            self.cusum_neg[key] = 0.0

        self.cusum_pos[key] = max(0, self.cusum_pos[key] + z - self.cusum_k)
        self.cusum_neg[key] = max(0, self.cusum_neg[key] - z - self.cusum_k)

        # Check for detection
        if self.cusum_pos[key] > self.cusum_h:
            self.cusum_pos[key] = 0.0  # Reset
            return self._create_statistical_result(
                signal_name, "increase", current_value, mean, std
            )
        elif self.cusum_neg[key] > self.cusum_h:
            self.cusum_neg[key] = 0.0  # Reset
            return self._create_statistical_result(
                signal_name, "decrease", current_value, mean, std
            )

        return DetectionResult(False, None, None, 0.0, {}, "")

    def _create_statistical_result(
        self, signal: str, direction: str, value: float, mean: float, std: float
    ) -> DetectionResult:
        """Create detection result for statistical anomaly"""
        anomaly_map = {
            "o2_increase": AnomalyType.COMBUSTION_LEAN,
            "o2_decrease": AnomalyType.COMBUSTION_RICH,
            "flame_decrease": AnomalyType.FLAME_INSTABILITY,
            "flame_increase": AnomalyType.FLAME_INSTABILITY,
        }

        key = f"{signal}_{direction}"
        anomaly_type = anomaly_map.get(key, AnomalyType.EFFICIENCY_DEGRADATION)

        return DetectionResult(
            detected=True,
            anomaly_type=anomaly_type,
            severity=Severity.S2,
            confidence=0.75,
            trigger_values={
                signal: value,
                f"{signal}_mean": mean,
                f"{signal}_std": std,
            },
            description=f"Statistical change detected: {signal} {direction} (value={value:.2f}, baseline mean={mean:.2f})"
        )


# =============================================================================
# ANOMALY DETECTOR (MAIN CLASS)
# =============================================================================

class AnomalyDetector:
    """
    Main anomaly detection engine

    Orchestrates multiple detection layers and manages anomaly lifecycle.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        enable_statistical: bool = True,
        enable_ml: bool = False
    ):
        """
        Initialize anomaly detector

        Args:
            config: Detection configuration
            enable_statistical: Enable statistical detection layer
            enable_ml: Enable ML-based detection layer (future)
        """
        self.config = config or DetectionConfig()
        self.rule_detector = RuleBasedDetector(self.config)
        self.statistical_detector = StatisticalDetector() if enable_statistical else None
        self.enable_ml = enable_ml

        # Active anomalies tracker
        self.active_anomalies: Dict[str, AnomalyEvent] = {}

        # Cooldown tracker (to prevent chatter)
        self.cooldown_tracker: Dict[AnomalyType, datetime] = {}

        logger.info("Anomaly Detector initialized")

    def detect(self, state: CombustionState) -> List[AnomalyEvent]:
        """
        Run anomaly detection on current state

        Args:
            state: Current combustion state

        Returns:
            List of new or updated anomaly events
        """
        start_time = time.perf_counter()
        events = []

        # Layer 1: Rule-based detection
        rule_results = self.rule_detector.detect(state)

        # Layer 2: Statistical detection
        stat_results = []
        if self.statistical_detector:
            stat_results = self.statistical_detector.detect(state)

        # Process all results
        all_results = rule_results + stat_results
        for result in all_results:
            event = self._process_detection_result(result, state)
            if event:
                events.append(event)

        # Check for resolved anomalies
        resolved = self._check_resolved_anomalies(state, all_results)
        events.extend(resolved)

        # Log performance
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 10:
            logger.warning(f"Detection cycle took {elapsed_ms:.1f}ms (target: <10ms)")

        return events

    def _process_detection_result(
        self, result: DetectionResult, state: CombustionState
    ) -> Optional[AnomalyEvent]:
        """Process a detection result and create/update anomaly event"""
        if not result.detected or not result.anomaly_type:
            return None

        # Check cooldown
        if self._is_in_cooldown(result.anomaly_type):
            return None

        # Check for existing active anomaly of same type
        existing_key = self._find_existing_anomaly(result.anomaly_type, state.asset_id)

        if existing_key:
            # Update existing anomaly
            anomaly = self.active_anomalies[existing_key]
            anomaly.last_update_time = state.timestamp
            anomaly.status = AnomalyStatus.UPDATING
            anomaly.trigger_values.update(result.trigger_values)
            if result.confidence > anomaly.confidence:
                anomaly.confidence = result.confidence
            if result.severity and result.severity.value > anomaly.severity.value:
                anomaly.severity = result.severity
            return anomaly
        else:
            # Create new anomaly
            return self._create_anomaly_event(result, state)

    def _create_anomaly_event(
        self, result: DetectionResult, state: CombustionState
    ) -> AnomalyEvent:
        """Create a new anomaly event"""
        taxonomy = ANOMALY_TAXONOMY.get(result.anomaly_type, {})

        incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

        event = AnomalyEvent(
            incident_id=incident_id,
            anomaly_type=result.anomaly_type,
            severity=result.severity or taxonomy.get("default_severity", Severity.S2),
            confidence=result.confidence,
            status=AnomalyStatus.ACTIVE,
            detection_layer=DetectionLayer.RULE_BASED,
            start_time=state.timestamp,
            last_update_time=state.timestamp,
            asset_id=state.asset_id,
            affected_signals=list(result.trigger_values.keys()),
            trigger_values=result.trigger_values,
            description=result.description or taxonomy.get("description", ""),
            affected_cqi_components=taxonomy.get("affected_cqi_components", []),
            recommended_checks=taxonomy.get("recommended_checks", []),
            provenance_hash=self._calculate_hash(incident_id, result)
        )

        # Calculate CQI impact (rough estimate)
        event.cqi_impact = self._estimate_cqi_impact(result)

        # Register active anomaly
        self.active_anomalies[incident_id] = event

        logger.info(f"New anomaly detected: {event.anomaly_type.value} ({event.severity.value})")

        return event

    def _check_resolved_anomalies(
        self, state: CombustionState, current_results: List[DetectionResult]
    ) -> List[AnomalyEvent]:
        """Check if any active anomalies are resolved"""
        resolved = []
        current_types = {r.anomaly_type for r in current_results if r.detected}

        for incident_id, anomaly in list(self.active_anomalies.items()):
            if anomaly.anomaly_type not in current_types:
                # Anomaly condition no longer present
                anomaly.status = AnomalyStatus.RESOLVED
                anomaly.end_time = state.timestamp
                resolved.append(anomaly)

                # Set cooldown
                self.cooldown_tracker[anomaly.anomaly_type] = state.timestamp

                # Remove from active
                del self.active_anomalies[incident_id]

                logger.info(f"Anomaly resolved: {anomaly.incident_id}")

        return resolved

    def _find_existing_anomaly(
        self, anomaly_type: AnomalyType, asset_id: str
    ) -> Optional[str]:
        """Find existing active anomaly of the same type"""
        for incident_id, anomaly in self.active_anomalies.items():
            if anomaly.anomaly_type == anomaly_type and anomaly.asset_id == asset_id:
                return incident_id
        return None

    def _is_in_cooldown(self, anomaly_type: AnomalyType) -> bool:
        """Check if anomaly type is in cooldown period"""
        if anomaly_type not in self.cooldown_tracker:
            return False

        cooldown_end = self.cooldown_tracker[anomaly_type] + timedelta(
            seconds=self.config.cooldown_seconds
        )
        return datetime.now(timezone.utc) < cooldown_end

    def _estimate_cqi_impact(self, result: DetectionResult) -> float:
        """Estimate CQI impact from detection result"""
        severity_impacts = {
            Severity.S1: -2.0,
            Severity.S2: -5.0,
            Severity.S3: -15.0,
            Severity.S4: -30.0,
        }
        return severity_impacts.get(result.severity, -5.0)

    def _calculate_hash(self, incident_id: str, result: DetectionResult) -> str:
        """Calculate provenance hash"""
        data = {
            "incident_id": incident_id,
            "anomaly_type": result.anomaly_type.value if result.anomaly_type else "",
            "trigger_values": result.trigger_values,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get_active_anomalies(self) -> List[AnomalyEvent]:
        """Get all active anomalies"""
        return list(self.active_anomalies.values())

    def get_anomaly(self, incident_id: str) -> Optional[AnomalyEvent]:
        """Get specific anomaly by incident ID"""
        return self.active_anomalies.get(incident_id)

    def suppress_anomaly(self, incident_id: str, reason: str) -> bool:
        """Suppress an anomaly"""
        if incident_id in self.active_anomalies:
            self.active_anomalies[incident_id].status = AnomalyStatus.SUPPRESSED
            self.active_anomalies[incident_id].suppression_reason = reason
            return True
        return False


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def create_default_detector() -> AnomalyDetector:
    """Create anomaly detector with default configuration"""
    return AnomalyDetector()


def get_anomaly_taxonomy() -> Dict[AnomalyType, Dict[str, Any]]:
    """Get the full anomaly taxonomy"""
    return ANOMALY_TAXONOMY


def detect_quick(
    o2: float, co: float, nox: float,
    flame_intensity: float = 80.0,
    flame_present: bool = True
) -> List[AnomalyEvent]:
    """Quick detection for testing"""
    detector = AnomalyDetector()
    state = CombustionState(
        o2_percent=o2,
        co_ppm=co,
        nox_ppm=nox,
        flame_intensity=flame_intensity,
        flame_present=flame_present
    )
    return detector.detect(state)
