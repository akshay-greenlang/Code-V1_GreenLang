# -*- coding: utf-8 -*-
"""
Explainability Module for GL-005 COMBUSENSE

Implements SHAP/LIME-style feature attributions and attention visualization
for combustion diagnostics as specified in GL-005 Playbook Section 10.

Key Features:
    - SHAP-style feature attributions for CQI components
    - LIME-style local explanations for anomaly incidents
    - Time-series attention maps showing signal/time-step importance
    - Top-N signal ranking with engineering interpretation
    - Confidence-weighted attributions

Reference: GL-005 Playbook Section 10 (Explainability, Attention Visualization)

Author: GreenLang GL-005 Team
Version: 1.0.0
Performance Target: <20ms per explanation
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Engineering interpretation templates
SIGNAL_INTERPRETATIONS = {
    "o2_percent": {
        "high": "Excess air - efficiency loss, possible lean operation",
        "low": "Insufficient air - risk of incomplete combustion",
        "normal": "O2 within optimal operating band",
    },
    "co_ppm": {
        "high": "Elevated CO indicates incomplete combustion",
        "low": "CO within normal operating limits",
        "normal": "CO levels acceptable",
    },
    "nox_ppm": {
        "high": "Elevated NOx - thermal NOx formation, check flame temperature",
        "low": "NOx within compliance limits",
        "normal": "NOx levels acceptable",
    },
    "fuel_flow": {
        "high": "High fuel flow - increased heat demand",
        "low": "Low fuel flow - reduced load or startup",
        "normal": "Fuel flow consistent with load",
    },
    "air_flow": {
        "high": "High air flow - excess air, check O2 setpoint",
        "low": "Low air flow - risk of rich combustion",
        "normal": "Air flow consistent with fuel flow",
    },
    "flame_intensity": {
        "high": "Strong flame signal - stable combustion",
        "low": "Weak flame signal - possible instability",
        "normal": "Flame intensity within normal range",
    },
    "air_fuel_ratio": {
        "high": "Lean operation - excess air",
        "low": "Rich operation - potential safety concern",
        "normal": "AFR within optimal range",
    },
}

# Attribution thresholds
ATTRIBUTION_SIGNIFICANCE_THRESHOLD = 0.05
TOP_N_DRIVERS = 5


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ExplanationType(str, Enum):
    """Types of explanations"""
    CQI_DEGRADATION = "cqi_degradation"
    ANOMALY_INCIDENT = "anomaly_incident"
    SAFETY_STATUS_CHANGE = "safety_status_change"
    TREND_ALERT = "trend_alert"


class SignalDirection(str, Enum):
    """Direction of signal deviation"""
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"
    INCREASING = "increasing"
    DECREASING = "decreasing"


class ContributionType(str, Enum):
    """Type of contribution to score change"""
    POSITIVE = "positive"  # Improves score
    NEGATIVE = "negative"  # Degrades score
    NEUTRAL = "neutral"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class FeatureAttribution:
    """Single feature attribution (SHAP-style)"""
    feature_name: str
    attribution_value: float  # -1 to +1, positive improves score
    base_value: float  # Feature's current value
    reference_value: float  # Baseline/expected value
    deviation: float  # Difference from reference
    contribution_type: ContributionType
    engineering_interpretation: str
    rank: int


@dataclass(frozen=True)
class TimeSegmentAttribution:
    """Attribution for a time segment (attention-style)"""
    segment_start_idx: int
    segment_end_idx: int
    segment_start_time: datetime
    segment_end_time: datetime
    attention_weight: float  # 0-1
    primary_signal: str
    description: str


@dataclass(frozen=True)
class AttentionMap:
    """Time-series attention map across signals and time"""
    signals: List[str]
    time_steps: int
    attention_weights: List[List[float]]  # [signal_idx][time_idx]
    top_signals: List[Tuple[str, float]]
    top_time_segments: List[TimeSegmentAttribution]
    provenance_hash: str


@dataclass(frozen=True)
class ExplanationResult:
    """Complete explanation result for an event"""
    explanation_id: str
    explanation_type: ExplanationType
    timestamp: datetime
    asset_id: str

    # Attributions
    feature_attributions: List[FeatureAttribution]
    top_drivers: List[FeatureAttribution]

    # Time-series attention (if applicable)
    attention_map: Optional[AttentionMap]

    # Context
    operating_mode: str
    load_context: str
    event_summary: str

    # Signal deltas
    signal_deltas: Dict[str, Dict[str, float]]  # signal -> {before, after, delta}

    # Confidence and metadata
    confidence: float
    model_version: str
    config_version: str
    provenance_hash: str


@dataclass(frozen=True)
class IncidentExplanation:
    """Explanation specifically for an anomaly incident"""
    incident_id: str
    event_type: str
    severity: str
    confidence: float

    # CQI impact
    cqi_before: float
    cqi_after: float
    cqi_delta: float
    component_impacts: Dict[str, float]

    # Top drivers with engineering interpretation
    top_drivers: List[FeatureAttribution]

    # Root cause hypotheses
    root_cause_hypotheses: List[Tuple[str, float, str]]  # (hypothesis, confidence, rationale)

    # Recommended checks
    recommended_checks: List[str]

    # Evidence
    evidence_signals: Dict[str, Dict[str, Any]]
    attention_map: Optional[AttentionMap]

    # Metadata
    explanation: ExplanationResult
    provenance_hash: str


# =============================================================================
# INPUT MODELS
# =============================================================================

class SignalSnapshot(BaseModel):
    """Snapshot of signal values at a point in time"""
    timestamp: datetime
    o2_percent: float = 0.0
    co_ppm: float = 0.0
    nox_ppm: float = 0.0
    fuel_flow_kg_s: float = 0.0
    air_flow_kg_s: float = 0.0
    flame_intensity: float = 80.0
    furnace_temp_c: float = 800.0
    furnace_pressure_pa: float = 101325.0
    load_percent: float = 75.0


class SignalTimeSeries(BaseModel):
    """Time series of signals for attention analysis"""
    timestamps: List[datetime]
    o2_percent: List[float] = Field(default_factory=list)
    co_ppm: List[float] = Field(default_factory=list)
    nox_ppm: List[float] = Field(default_factory=list)
    fuel_flow: List[float] = Field(default_factory=list)
    air_flow: List[float] = Field(default_factory=list)
    flame_intensity: List[float] = Field(default_factory=list)


class ExplainabilityInput(BaseModel):
    """Input for generating explanations"""
    event_type: ExplanationType
    asset_id: str
    incident_id: Optional[str] = None

    # Current and reference snapshots
    current_snapshot: SignalSnapshot
    reference_snapshot: Optional[SignalSnapshot] = None

    # Time series for attention (optional)
    time_series: Optional[SignalTimeSeries] = None

    # CQI values
    cqi_current: float = 0.0
    cqi_previous: float = 0.0

    # Operating context
    operating_mode: str = "RUN"
    load_percent: float = 75.0


# =============================================================================
# SHAP-STYLE EXPLAINER
# =============================================================================

class SHAPStyleExplainer:
    """
    SHAP-style feature attribution explainer

    Uses a simplified SHAP approach for combustion diagnostics:
    - Computes feature contributions to score changes
    - Provides engineering interpretations
    - Supports both global and local explanations
    """

    def __init__(
        self,
        reference_values: Optional[Dict[str, float]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize SHAP-style explainer

        Args:
            reference_values: Baseline/expected values for each feature
            feature_ranges: (min, max) ranges for each feature
        """
        self.reference_values = reference_values or {
            "o2_percent": 3.0,
            "co_ppm": 50.0,
            "nox_ppm": 30.0,
            "fuel_flow": 1.0,
            "air_flow": 17.2,
            "flame_intensity": 80.0,
            "air_fuel_ratio": 17.2,
            "furnace_temp_c": 800.0,
        }

        self.feature_ranges = feature_ranges or {
            "o2_percent": (0.0, 21.0),
            "co_ppm": (0.0, 500.0),
            "nox_ppm": (0.0, 200.0),
            "fuel_flow": (0.0, 10.0),
            "air_flow": (0.0, 200.0),
            "flame_intensity": (0.0, 100.0),
            "air_fuel_ratio": (10.0, 25.0),
            "furnace_temp_c": (400.0, 1200.0),
        }

        # Feature importance weights (learned or configured)
        self.feature_weights = {
            "o2_percent": 0.25,
            "co_ppm": 0.20,
            "nox_ppm": 0.15,
            "fuel_flow": 0.10,
            "air_flow": 0.10,
            "flame_intensity": 0.10,
            "air_fuel_ratio": 0.05,
            "furnace_temp_c": 0.05,
        }

    def explain(
        self,
        current_values: Dict[str, float],
        score_delta: float,
        reference_values: Optional[Dict[str, float]] = None
    ) -> List[FeatureAttribution]:
        """
        Generate SHAP-style feature attributions

        Args:
            current_values: Current feature values
            score_delta: Change in score to explain
            reference_values: Optional custom reference values

        Returns:
            List of feature attributions sorted by absolute attribution
        """
        refs = reference_values or self.reference_values
        attributions = []

        for feature_name, current_value in current_values.items():
            if feature_name not in refs:
                continue

            ref_value = refs[feature_name]
            deviation = current_value - ref_value

            # Calculate normalized attribution
            feature_range = self.feature_ranges.get(feature_name, (0, 100))
            range_size = max(feature_range[1] - feature_range[0], 1e-6)
            normalized_deviation = deviation / range_size

            # Apply feature weight
            weight = self.feature_weights.get(feature_name, 0.1)
            attribution = normalized_deviation * weight * score_delta

            # Determine contribution type
            if abs(attribution) < ATTRIBUTION_SIGNIFICANCE_THRESHOLD:
                contribution_type = ContributionType.NEUTRAL
            elif attribution > 0:
                contribution_type = ContributionType.POSITIVE
            else:
                contribution_type = ContributionType.NEGATIVE

            # Get engineering interpretation
            direction = self._get_signal_direction(feature_name, deviation, ref_value)
            interpretation = self._get_interpretation(feature_name, direction)

            attributions.append(FeatureAttribution(
                feature_name=feature_name,
                attribution_value=round(attribution, 4),
                base_value=round(current_value, 4),
                reference_value=round(ref_value, 4),
                deviation=round(deviation, 4),
                contribution_type=contribution_type,
                engineering_interpretation=interpretation,
                rank=0  # Will be set after sorting
            ))

        # Sort by absolute attribution and assign ranks
        attributions.sort(key=lambda x: abs(x.attribution_value), reverse=True)
        ranked_attributions = []
        for i, attr in enumerate(attributions):
            ranked_attributions.append(FeatureAttribution(
                feature_name=attr.feature_name,
                attribution_value=attr.attribution_value,
                base_value=attr.base_value,
                reference_value=attr.reference_value,
                deviation=attr.deviation,
                contribution_type=attr.contribution_type,
                engineering_interpretation=attr.engineering_interpretation,
                rank=i + 1
            ))

        return ranked_attributions

    def _get_signal_direction(
        self, feature: str, deviation: float, reference: float
    ) -> SignalDirection:
        """Determine signal direction from deviation"""
        threshold = abs(reference * 0.1) if reference != 0 else 1.0

        if deviation > threshold:
            return SignalDirection.HIGH
        elif deviation < -threshold:
            return SignalDirection.LOW
        else:
            return SignalDirection.NORMAL

    def _get_interpretation(self, feature: str, direction: SignalDirection) -> str:
        """Get engineering interpretation for feature and direction"""
        interpretations = SIGNAL_INTERPRETATIONS.get(feature, {})
        return interpretations.get(
            direction.value,
            f"{feature} is {direction.value}"
        )


# =============================================================================
# LIME-STYLE EXPLAINER
# =============================================================================

class LIMEStyleExplainer:
    """
    LIME-style local interpretable model-agnostic explanations

    Provides per-incident explanations using local linear approximations.
    """

    def __init__(self, kernel_width: float = 0.75):
        """
        Initialize LIME-style explainer

        Args:
            kernel_width: Width of the exponential kernel for weighting
        """
        self.kernel_width = kernel_width
        self.shap_explainer = SHAPStyleExplainer()

    def explain_incident(
        self,
        incident_type: str,
        severity: str,
        before_snapshot: SignalSnapshot,
        after_snapshot: SignalSnapshot,
        cqi_delta: float
    ) -> Tuple[List[FeatureAttribution], List[Tuple[str, float, str]]]:
        """
        Generate LIME-style explanation for an incident

        Args:
            incident_type: Type of anomaly incident
            severity: Incident severity
            before_snapshot: Signal values before incident
            after_snapshot: Signal values after/during incident
            cqi_delta: CQI change due to incident

        Returns:
            Tuple of (attributions, root_cause_hypotheses)
        """
        # Extract signal values
        before_values = {
            "o2_percent": before_snapshot.o2_percent,
            "co_ppm": before_snapshot.co_ppm,
            "nox_ppm": before_snapshot.nox_ppm,
            "fuel_flow": before_snapshot.fuel_flow_kg_s,
            "air_flow": before_snapshot.air_flow_kg_s,
            "flame_intensity": before_snapshot.flame_intensity,
        }

        after_values = {
            "o2_percent": after_snapshot.o2_percent,
            "co_ppm": after_snapshot.co_ppm,
            "nox_ppm": after_snapshot.nox_ppm,
            "fuel_flow": after_snapshot.fuel_flow_kg_s,
            "air_flow": after_snapshot.air_flow_kg_s,
            "flame_intensity": after_snapshot.flame_intensity,
        }

        # Get attributions using SHAP explainer with before values as reference
        attributions = self.shap_explainer.explain(
            after_values, cqi_delta, before_values
        )

        # Generate root cause hypotheses based on top drivers
        hypotheses = self._generate_hypotheses(
            incident_type, attributions[:TOP_N_DRIVERS]
        )

        return attributions, hypotheses

    def _generate_hypotheses(
        self,
        incident_type: str,
        top_drivers: List[FeatureAttribution]
    ) -> List[Tuple[str, float, str]]:
        """Generate root cause hypotheses from top drivers"""
        hypotheses = []

        # Map incident types to hypothesis templates
        hypothesis_templates = {
            "CO_SPIKE": [
                ("Incomplete combustion due to air shortage", 0.85,
                 "Check air damper position and FD fan status"),
                ("Fuel quality variation", 0.65,
                 "Verify fuel composition and heating value"),
                ("Burner fouling or plugging", 0.50,
                 "Inspect burner tips and fuel nozzles"),
            ],
            "NOX_SPIKE": [
                ("High flame temperature causing thermal NOx", 0.80,
                 "Check air staging and flame pattern"),
                ("Excess air imbalance", 0.70,
                 "Review O2 trim setpoint and control"),
                ("Load transition effects", 0.55,
                 "Evaluate load change rate"),
            ],
            "COMBUSTION_RICH": [
                ("Air flow restriction", 0.85,
                 "Check dampers, filters, and fan operation"),
                ("Fuel metering error", 0.70,
                 "Verify fuel flow measurement and control valve"),
                ("AFR control fault", 0.60,
                 "Review AFR controller tuning and output"),
            ],
            "COMBUSTION_LEAN": [
                ("Excess air from leakage", 0.80,
                 "Inspect furnace for air in-leakage"),
                ("Air flow control overshoot", 0.70,
                 "Check damper control and FD fan response"),
                ("O2 trim over-correction", 0.55,
                 "Review O2 trim controller parameters"),
            ],
            "FLAME_INSTABILITY": [
                ("Fuel pressure fluctuation", 0.80,
                 "Check fuel supply pressure and regulators"),
                ("Air/fuel mixing issue", 0.75,
                 "Inspect burner register settings"),
                ("Flame scanner fault", 0.50,
                 "Verify flame scanner operation and calibration"),
            ],
        }

        # Get hypotheses for incident type
        templates = hypothesis_templates.get(
            incident_type,
            [("Unknown cause - investigate", 0.5, "Manual investigation required")]
        )

        # Adjust confidence based on top driver alignment
        for hypothesis, base_confidence, rationale in templates:
            # Boost confidence if top drivers align with hypothesis
            confidence_boost = 0.0
            for driver in top_drivers:
                if self._driver_supports_hypothesis(driver, hypothesis):
                    confidence_boost += 0.05

            final_confidence = min(0.95, base_confidence + confidence_boost)
            hypotheses.append((hypothesis, final_confidence, rationale))

        # Sort by confidence
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        return hypotheses[:5]  # Return top 5 hypotheses

    def _driver_supports_hypothesis(
        self, driver: FeatureAttribution, hypothesis: str
    ) -> bool:
        """Check if a driver supports a hypothesis"""
        hypothesis_lower = hypothesis.lower()
        feature = driver.feature_name.lower()

        # Simple keyword matching
        if "air" in hypothesis_lower and ("air" in feature or "o2" in feature):
            return True
        if "fuel" in hypothesis_lower and "fuel" in feature:
            return True
        if "flame" in hypothesis_lower and "flame" in feature:
            return True
        if "combustion" in hypothesis_lower and ("co" in feature or "nox" in feature):
            return True

        return False


# =============================================================================
# ATTENTION VISUALIZER
# =============================================================================

class AttentionVisualizer:
    """
    Time-series attention visualization

    Generates attention maps showing signal and time-step importance
    for combustion diagnostics (Playbook Section 10.3).
    """

    def __init__(self, window_size: int = 120, sensitivity: float = 1.0):
        """
        Initialize attention visualizer

        Args:
            window_size: Number of time steps for attention window
            sensitivity: Sensitivity factor for change detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity

    def compute_attention_map(
        self,
        time_series: SignalTimeSeries,
        event_timestamp: datetime
    ) -> AttentionMap:
        """
        Compute attention map from time series data

        Args:
            time_series: Time series of signals
            event_timestamp: Timestamp of the event to explain

        Returns:
            AttentionMap with signal and time-step importances
        """
        signals = ["o2_percent", "co_ppm", "nox_ppm", "fuel_flow",
                   "air_flow", "flame_intensity"]

        # Get signal arrays
        signal_arrays = {
            "o2_percent": time_series.o2_percent,
            "co_ppm": time_series.co_ppm,
            "nox_ppm": time_series.nox_ppm,
            "fuel_flow": time_series.fuel_flow,
            "air_flow": time_series.air_flow,
            "flame_intensity": time_series.flame_intensity,
        }

        # Compute attention weights based on variability and change
        n_steps = len(time_series.timestamps)
        attention_weights: List[List[float]] = []

        signal_importances = {}

        for signal_name in signals:
            signal_data = signal_arrays.get(signal_name, [])
            if not signal_data or len(signal_data) < 2:
                attention_weights.append([0.0] * n_steps)
                signal_importances[signal_name] = 0.0
                continue

            # Calculate time-step attention based on rate of change
            step_weights = []
            for i in range(len(signal_data)):
                if i == 0:
                    change = 0.0
                else:
                    change = abs(signal_data[i] - signal_data[i - 1])

                # Normalize by signal range
                signal_range = max(signal_data) - min(signal_data) if max(signal_data) != min(signal_data) else 1.0
                normalized_change = change / signal_range * self.sensitivity
                step_weights.append(normalized_change)

            # Normalize weights to sum to 1
            total = sum(step_weights) or 1.0
            normalized_weights = [w / total for w in step_weights]
            attention_weights.append(normalized_weights)

            # Calculate overall signal importance (variance-based)
            variance = np.var(signal_data) if len(signal_data) > 1 else 0.0
            signal_importances[signal_name] = float(variance)

        # Normalize signal importances
        total_importance = sum(signal_importances.values()) or 1.0
        for k in signal_importances:
            signal_importances[k] /= total_importance

        # Get top signals
        top_signals = sorted(
            signal_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_N_DRIVERS]

        # Find top time segments
        top_segments = self._find_top_time_segments(
            time_series, attention_weights, signals
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_hash(attention_weights, signals)

        return AttentionMap(
            signals=signals,
            time_steps=n_steps,
            attention_weights=attention_weights,
            top_signals=top_signals,
            top_time_segments=top_segments,
            provenance_hash=provenance_hash
        )

    def _find_top_time_segments(
        self,
        time_series: SignalTimeSeries,
        attention_weights: List[List[float]],
        signals: List[str]
    ) -> List[TimeSegmentAttribution]:
        """Find the top contributing time segments"""
        segments = []
        timestamps = time_series.timestamps

        if not timestamps or len(timestamps) < 2:
            return segments

        # Find peaks in attention weights
        for signal_idx, signal_name in enumerate(signals):
            weights = attention_weights[signal_idx]
            if not weights:
                continue

            # Find local maxima
            for i in range(1, len(weights) - 1):
                if weights[i] > weights[i - 1] and weights[i] > weights[i + 1]:
                    if weights[i] > 0.1:  # Threshold for significance
                        segment = TimeSegmentAttribution(
                            segment_start_idx=max(0, i - 2),
                            segment_end_idx=min(len(timestamps) - 1, i + 2),
                            segment_start_time=timestamps[max(0, i - 2)],
                            segment_end_time=timestamps[min(len(timestamps) - 1, i + 2)],
                            attention_weight=weights[i],
                            primary_signal=signal_name,
                            description=f"Significant change in {signal_name}"
                        )
                        segments.append(segment)

        # Sort by attention weight and return top segments
        segments.sort(key=lambda x: x.attention_weight, reverse=True)
        return segments[:5]

    def _calculate_hash(
        self,
        attention_weights: List[List[float]],
        signals: List[str]
    ) -> str:
        """Calculate provenance hash for attention map"""
        data = {
            "signals": signals,
            "weights_checksum": sum(
                sum(w) for w in attention_weights if w
            ),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]


# =============================================================================
# MAIN EXPLAINABILITY ENGINE
# =============================================================================

class ExplainabilityEngine:
    """
    Main explainability engine for GL-005 COMBUSENSE

    Combines SHAP, LIME, and attention visualization to produce
    comprehensive explanations for combustion events.
    """

    def __init__(
        self,
        model_version: str = "1.0.0",
        config_version: str = "1.0.0"
    ):
        """
        Initialize explainability engine

        Args:
            model_version: Current model version for audit trail
            config_version: Current config version for audit trail
        """
        self.shap_explainer = SHAPStyleExplainer()
        self.lime_explainer = LIMEStyleExplainer()
        self.attention_visualizer = AttentionVisualizer()
        self.model_version = model_version
        self.config_version = config_version

    def explain_cqi_change(
        self,
        input_data: ExplainabilityInput
    ) -> ExplanationResult:
        """
        Generate explanation for CQI degradation or change

        Args:
            input_data: Explainability input with snapshots and context

        Returns:
            Complete explanation result
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # Get signal values
        current = input_data.current_snapshot
        reference = input_data.reference_snapshot or self._get_default_reference()

        current_values = {
            "o2_percent": current.o2_percent,
            "co_ppm": current.co_ppm,
            "nox_ppm": current.nox_ppm,
            "fuel_flow": current.fuel_flow_kg_s,
            "air_flow": current.air_flow_kg_s,
            "flame_intensity": current.flame_intensity,
        }

        reference_values = {
            "o2_percent": reference.o2_percent,
            "co_ppm": reference.co_ppm,
            "nox_ppm": reference.nox_ppm,
            "fuel_flow": reference.fuel_flow_kg_s,
            "air_flow": reference.air_flow_kg_s,
            "flame_intensity": reference.flame_intensity,
        }

        # Calculate CQI delta
        cqi_delta = input_data.cqi_current - input_data.cqi_previous

        # Get feature attributions
        attributions = self.shap_explainer.explain(
            current_values, cqi_delta, reference_values
        )

        # Get top drivers
        top_drivers = attributions[:TOP_N_DRIVERS]

        # Compute attention map if time series provided
        attention_map = None
        if input_data.time_series:
            attention_map = self.attention_visualizer.compute_attention_map(
                input_data.time_series,
                current.timestamp
            )

        # Calculate signal deltas
        signal_deltas = {}
        for key in current_values:
            signal_deltas[key] = {
                "before": reference_values.get(key, 0.0),
                "after": current_values.get(key, 0.0),
                "delta": current_values.get(key, 0.0) - reference_values.get(key, 0.0),
            }

        # Generate event summary
        event_summary = self._generate_event_summary(
            input_data.event_type, top_drivers, cqi_delta
        )

        # Calculate confidence
        confidence = self._calculate_explanation_confidence(attributions)

        # Generate explanation ID and hash
        explanation_id = f"EXP-{timestamp.strftime('%Y%m%d%H%M%S')}-{input_data.asset_id[:8]}"
        provenance_hash = self._calculate_provenance_hash(
            attributions, timestamp, input_data.asset_id
        )

        return ExplanationResult(
            explanation_id=explanation_id,
            explanation_type=input_data.event_type,
            timestamp=timestamp,
            asset_id=input_data.asset_id,
            feature_attributions=attributions,
            top_drivers=top_drivers,
            attention_map=attention_map,
            operating_mode=input_data.operating_mode,
            load_context=f"Load: {input_data.load_percent:.0f}%",
            event_summary=event_summary,
            signal_deltas=signal_deltas,
            confidence=confidence,
            model_version=self.model_version,
            config_version=self.config_version,
            provenance_hash=provenance_hash
        )

    def explain_incident(
        self,
        incident_id: str,
        incident_type: str,
        severity: str,
        before_snapshot: SignalSnapshot,
        after_snapshot: SignalSnapshot,
        cqi_before: float,
        cqi_after: float,
        time_series: Optional[SignalTimeSeries] = None,
        asset_id: str = "default"
    ) -> IncidentExplanation:
        """
        Generate comprehensive explanation for an anomaly incident

        Args:
            incident_id: Unique incident identifier
            incident_type: Type of anomaly (e.g., CO_SPIKE, FLAME_INSTABILITY)
            severity: Incident severity (S1-S4)
            before_snapshot: Signals before incident
            after_snapshot: Signals during/after incident
            cqi_before: CQI before incident
            cqi_after: CQI after incident
            time_series: Optional time series for attention
            asset_id: Asset identifier

        Returns:
            Complete incident explanation
        """
        cqi_delta = cqi_after - cqi_before

        # Get LIME-style attributions and hypotheses
        attributions, hypotheses = self.lime_explainer.explain_incident(
            incident_type, severity, before_snapshot, after_snapshot, cqi_delta
        )

        # Get attention map if time series provided
        attention_map = None
        if time_series:
            attention_map = self.attention_visualizer.compute_attention_map(
                time_series, after_snapshot.timestamp
            )

        # Calculate component impacts
        component_impacts = self._calculate_component_impacts(
            attributions, cqi_delta
        )

        # Get recommended checks
        recommended_checks = self._get_recommended_checks(
            incident_type, hypotheses
        )

        # Build evidence signals
        evidence_signals = self._build_evidence_signals(
            before_snapshot, after_snapshot
        )

        # Create base explanation
        input_data = ExplainabilityInput(
            event_type=ExplanationType.ANOMALY_INCIDENT,
            asset_id=asset_id,
            incident_id=incident_id,
            current_snapshot=after_snapshot,
            reference_snapshot=before_snapshot,
            time_series=time_series,
            cqi_current=cqi_after,
            cqi_previous=cqi_before,
        )
        base_explanation = self.explain_cqi_change(input_data)

        # Calculate provenance hash
        provenance_hash = self._calculate_incident_hash(
            incident_id, incident_type, attributions
        )

        return IncidentExplanation(
            incident_id=incident_id,
            event_type=incident_type,
            severity=severity,
            confidence=base_explanation.confidence,
            cqi_before=cqi_before,
            cqi_after=cqi_after,
            cqi_delta=cqi_delta,
            component_impacts=component_impacts,
            top_drivers=attributions[:TOP_N_DRIVERS],
            root_cause_hypotheses=hypotheses,
            recommended_checks=recommended_checks,
            evidence_signals=evidence_signals,
            attention_map=attention_map,
            explanation=base_explanation,
            provenance_hash=provenance_hash
        )

    def _get_default_reference(self) -> SignalSnapshot:
        """Get default reference snapshot"""
        return SignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            o2_percent=3.0,
            co_ppm=50.0,
            nox_ppm=30.0,
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            flame_intensity=80.0,
        )

    def _generate_event_summary(
        self,
        event_type: ExplanationType,
        top_drivers: List[FeatureAttribution],
        cqi_delta: float
    ) -> str:
        """Generate human-readable event summary"""
        if not top_drivers:
            return f"CQI changed by {cqi_delta:+.1f} points"

        top_driver = top_drivers[0]
        direction = "increased" if top_driver.deviation > 0 else "decreased"

        return (
            f"CQI changed by {cqi_delta:+.1f} points. "
            f"Primary driver: {top_driver.feature_name} {direction} "
            f"from {top_driver.reference_value:.1f} to {top_driver.base_value:.1f}. "
            f"{top_driver.engineering_interpretation}"
        )

    def _calculate_explanation_confidence(
        self, attributions: List[FeatureAttribution]
    ) -> float:
        """Calculate confidence in explanation"""
        if not attributions:
            return 0.0

        # Confidence based on how much variance is explained by top drivers
        total_attribution = sum(abs(a.attribution_value) for a in attributions)
        top_attribution = sum(abs(a.attribution_value) for a in attributions[:3])

        if total_attribution == 0:
            return 0.5

        explained_ratio = top_attribution / total_attribution
        return min(0.95, explained_ratio + 0.3)

    def _calculate_component_impacts(
        self,
        attributions: List[FeatureAttribution],
        cqi_delta: float
    ) -> Dict[str, float]:
        """Calculate impact on CQI components"""
        # Map features to CQI components
        feature_to_component = {
            "o2_percent": "efficiency",
            "co_ppm": "emissions",
            "nox_ppm": "emissions",
            "fuel_flow": "efficiency",
            "air_flow": "efficiency",
            "flame_intensity": "stability",
        }

        impacts = {
            "efficiency": 0.0,
            "emissions": 0.0,
            "stability": 0.0,
            "safety": 0.0,
            "data": 0.0,
        }

        for attr in attributions:
            component = feature_to_component.get(attr.feature_name, "data")
            impacts[component] += attr.attribution_value

        return impacts

    def _get_recommended_checks(
        self,
        incident_type: str,
        hypotheses: List[Tuple[str, float, str]]
    ) -> List[str]:
        """Get recommended checks based on incident type and hypotheses"""
        # Base checks for all incidents
        checks = [
            "Verify analyzer status and calibration",
            "Check recent control actions and setpoint changes",
        ]

        # Add hypothesis-specific checks
        for _, _, rationale in hypotheses[:3]:
            if rationale not in checks:
                checks.append(rationale)

        # Add incident-type specific checks
        type_specific = {
            "CO_SPIKE": [
                "Inspect burner tips for fouling",
                "Check air damper positions",
                "Review fuel quality logs",
            ],
            "NOX_SPIKE": [
                "Check flame temperature readings",
                "Review air staging settings",
                "Verify load transition rates",
            ],
            "FLAME_INSTABILITY": [
                "Inspect flame scanner readings",
                "Check fuel supply pressure",
                "Review burner register positions",
            ],
        }

        for check in type_specific.get(incident_type, []):
            if check not in checks:
                checks.append(check)

        return checks[:7]  # Limit to 7 checks

    def _build_evidence_signals(
        self,
        before: SignalSnapshot,
        after: SignalSnapshot
    ) -> Dict[str, Dict[str, Any]]:
        """Build evidence signals for incident"""
        return {
            "o2_percent": {
                "before": before.o2_percent,
                "after": after.o2_percent,
                "delta": after.o2_percent - before.o2_percent,
                "trend": "increasing" if after.o2_percent > before.o2_percent else "decreasing",
            },
            "co_ppm": {
                "before": before.co_ppm,
                "after": after.co_ppm,
                "delta": after.co_ppm - before.co_ppm,
                "trend": "increasing" if after.co_ppm > before.co_ppm else "decreasing",
            },
            "nox_ppm": {
                "before": before.nox_ppm,
                "after": after.nox_ppm,
                "delta": after.nox_ppm - before.nox_ppm,
                "trend": "increasing" if after.nox_ppm > before.nox_ppm else "decreasing",
            },
            "flame_intensity": {
                "before": before.flame_intensity,
                "after": after.flame_intensity,
                "delta": after.flame_intensity - before.flame_intensity,
                "trend": "increasing" if after.flame_intensity > before.flame_intensity else "decreasing",
            },
        }

    def _calculate_provenance_hash(
        self,
        attributions: List[FeatureAttribution],
        timestamp: datetime,
        asset_id: str
    ) -> str:
        """Calculate provenance hash for audit trail"""
        data = {
            "attributions": [
                (a.feature_name, round(a.attribution_value, 6))
                for a in attributions
            ],
            "timestamp": timestamp.isoformat(),
            "asset_id": asset_id,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _calculate_incident_hash(
        self,
        incident_id: str,
        incident_type: str,
        attributions: List[FeatureAttribution]
    ) -> str:
        """Calculate provenance hash for incident explanation"""
        data = {
            "incident_id": incident_id,
            "incident_type": incident_type,
            "top_drivers": [
                a.feature_name for a in attributions[:3]
            ],
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_evidence_bundle(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """
        Convert explanation to evidence bundle for narrative generation

        Returns the minimum content required for natural-language narratives
        as specified in Playbook Section 10.4.
        """
        return {
            "operating_mode": explanation.operating_mode,
            "load_context": explanation.load_context,
            "signal_deltas": explanation.signal_deltas,
            "event_type": explanation.explanation_type.value,
            "severity": "S3" if explanation.confidence > 0.7 else "S2",
            "confidence": explanation.confidence,
            "top_attributions": [
                {
                    "feature": d.feature_name,
                    "attribution": d.attribution_value,
                    "interpretation": d.engineering_interpretation,
                }
                for d in explanation.top_drivers
            ],
            "time_segments": (
                [
                    {
                        "signal": s.primary_signal,
                        "weight": s.attention_weight,
                        "description": s.description,
                    }
                    for s in explanation.attention_map.top_time_segments
                ] if explanation.attention_map else []
            ),
            "recommended_checks": [],  # Populated by incident explanation
        }


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def create_default_engine() -> ExplainabilityEngine:
    """Create explainability engine with default configuration"""
    return ExplainabilityEngine()


def explain_cqi_quick(
    o2: float, co: float, nox: float,
    ref_o2: float = 3.0, ref_co: float = 50.0, ref_nox: float = 30.0,
    cqi_delta: float = -10.0
) -> List[FeatureAttribution]:
    """Quick explanation for testing"""
    explainer = SHAPStyleExplainer()
    current = {"o2_percent": o2, "co_ppm": co, "nox_ppm": nox}
    reference = {"o2_percent": ref_o2, "co_ppm": ref_co, "nox_ppm": ref_nox}
    return explainer.explain(current, cqi_delta, reference)
