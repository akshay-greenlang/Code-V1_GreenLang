# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Failure Diagnostics Module

This module provides steam trap failure detection and diagnostics using
ultrasonic, temperature differential, and visual inspection methods.
It implements decision tree logic for failure mode classification.

Features:
    - Ultrasonic analysis for steam flow detection
    - Temperature differential analysis for trap condition
    - Decision tree diagnostic logic
    - Multi-method correlation for confidence scoring
    - Failure mode probability estimation

Diagnostic Methods:
    - Ultrasonic: Detects steam flow by acoustic emission
    - Temperature: Compares inlet/outlet temperatures
    - Visual: Discharge observation (flash steam vs condensate)

Failure Modes:
    - Good: Operating correctly
    - Failed Open: Blowing through (live steam loss)
    - Failed Closed: Blocked (condensate backup)
    - Leaking: Partial steam loss
    - Cold: No steam/condensate flow

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Diagnostic Guidelines

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.failure_diagnostics import (
    ...     TrapDiagnosticsEngine,
    ... )
    >>> engine = TrapDiagnosticsEngine(config)
    >>> result = engine.diagnose(diagnostic_input)
    >>> print(f"Status: {result.condition.status}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import statistics

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    TrapType,
    FailureMode,
    DiagnosticMethod,
    DiagnosticThresholds,
    TrapTypeConfig,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapDiagnosticInput,
    TrapDiagnosticOutput,
    TrapCondition,
    TrapHealthScore,
    FailureModeProbability,
    MaintenanceRecommendation,
    SteamLossEstimate,
    UltrasonicReading,
    TemperatureReading,
    VisualInspectionReading,
    TrapStatus,
    DiagnosisConfidence,
    TrendDirection,
    MaintenancePriority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DIAGNOSTIC RESULT MODELS
# =============================================================================

@dataclass
class UltrasonicDiagnosticResult:
    """Result of ultrasonic analysis."""

    status: TrapStatus
    confidence: float
    average_db: float
    peak_db: float
    cycling_detected: bool
    continuous_flow_detected: bool
    evidence: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class TemperatureDiagnosticResult:
    """Result of temperature differential analysis."""

    status: TrapStatus
    confidence: float
    inlet_temp_f: float
    outlet_temp_f: float
    delta_t_f: float
    subcooling_f: float
    evidence: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CombinedDiagnosticResult:
    """Combined diagnostic result from multiple methods."""

    status: TrapStatus
    confidence: float
    confidence_level: DiagnosisConfidence
    failure_probabilities: Dict[str, float]
    evidence: List[str]
    inconsistencies: List[str]
    methods_used: List[DiagnosticMethod]


# =============================================================================
# ULTRASONIC ANALYZER
# =============================================================================

class UltrasonicAnalyzer:
    """
    Ultrasonic diagnostic analyzer for steam traps.

    Ultrasonic testing detects the acoustic emission from steam and
    condensate flow. Key principles:
    - Steam flow creates high-frequency noise (>30 kHz)
    - Condensate flow is relatively quiet
    - Cycling traps show intermittent sound patterns
    - Failed open traps show continuous high noise

    Reference: Spirax Sarco Steam Trap Management Guide
    """

    def __init__(self, thresholds: DiagnosticThresholds) -> None:
        """
        Initialize ultrasonic analyzer.

        Args:
            thresholds: Diagnostic threshold configuration
        """
        self.thresholds = thresholds.ultrasonic
        self._analysis_count = 0

    def analyze(
        self,
        readings: List[UltrasonicReading],
        trap_type: TrapType,
        trap_config: Optional[TrapTypeConfig] = None,
    ) -> UltrasonicDiagnosticResult:
        """
        Analyze ultrasonic readings for trap status.

        Args:
            readings: List of ultrasonic readings
            trap_type: Type of steam trap
            trap_config: Trap type configuration for adjustments

        Returns:
            UltrasonicDiagnosticResult
        """
        self._analysis_count += 1

        if not readings:
            return UltrasonicDiagnosticResult(
                status=TrapStatus.UNKNOWN,
                confidence=0.0,
                average_db=0.0,
                peak_db=0.0,
                cycling_detected=False,
                continuous_flow_detected=False,
                notes=["No ultrasonic readings provided"],
            )

        # Extract dB levels
        db_levels = [r.decibel_level_db for r in readings]
        average_db = statistics.mean(db_levels)
        peak_db = max(db_levels)
        min_db = min(db_levels)

        # Apply trap-type specific adjustment
        db_adjustment = 0.0
        if trap_config:
            db_adjustment = trap_config.ultrasonic_adjustment_db
        elif trap_type == TrapType.THERMODYNAMIC:
            db_adjustment = 10.0  # TD traps cycle loudly
        elif trap_type == TrapType.INVERTED_BUCKET:
            db_adjustment = 5.0

        adjusted_avg_db = average_db - db_adjustment

        # Check for cycling pattern
        cycling_detected = any(r.cycling_detected for r in readings)
        if not cycling_detected and len(db_levels) > 1:
            # Check for variation indicating cycling
            db_variance = max(db_levels) - min(db_levels)
            if db_variance > 15:  # More than 15 dB variation
                cycling_detected = True

        # Check for continuous flow
        continuous_flow_detected = any(r.continuous_flow_detected for r in readings)
        if not continuous_flow_detected:
            # If all readings are high and consistent, likely continuous
            if adjusted_avg_db > self.thresholds.failed_open_db:
                if max(db_levels) - min(db_levels) < 10:
                    continuous_flow_detected = True

        evidence = []
        notes = []

        # Diagnostic decision tree
        if adjusted_avg_db < self.thresholds.cold_max_db:
            # Very low sound - cold/blocked trap
            status = TrapStatus.FAILED_CLOSED
            confidence = 0.75
            evidence.append(f"Very low ultrasonic level ({average_db:.1f} dB)")
            evidence.append("No flow detected at trap")

        elif continuous_flow_detected or adjusted_avg_db > self.thresholds.failed_open_db:
            # High continuous sound - failed open
            status = TrapStatus.FAILED_OPEN
            confidence = 0.85
            evidence.append(f"High ultrasonic level ({average_db:.1f} dB)")
            evidence.append("Continuous flow pattern detected")

        elif adjusted_avg_db > self.thresholds.leaking_min_db:
            # Elevated sound - leaking
            status = TrapStatus.LEAKING
            confidence = 0.70
            evidence.append(f"Elevated ultrasonic level ({average_db:.1f} dB)")
            if not cycling_detected:
                evidence.append("No cycling pattern detected")

        elif cycling_detected:
            # Normal cycling pattern
            status = TrapStatus.GOOD
            confidence = 0.80
            evidence.append("Normal cycling pattern detected")
            evidence.append(f"Ultrasonic level within range ({average_db:.1f} dB)")

        elif adjusted_avg_db < self.thresholds.good_max_db:
            # Low sound, likely good
            status = TrapStatus.GOOD
            confidence = 0.70
            evidence.append(f"Ultrasonic level acceptable ({average_db:.1f} dB)")

        else:
            # Uncertain
            status = TrapStatus.UNKNOWN
            confidence = 0.50
            notes.append("Unable to determine status from ultrasonic alone")

        # Add trap type notes
        if trap_type == TrapType.THERMODYNAMIC:
            notes.append("TD traps cycle loudly - normal operation")
        elif trap_type == TrapType.INVERTED_BUCKET:
            notes.append("Inverted bucket should show cycling pattern")
        elif trap_type == TrapType.FLOAT_THERMOSTATIC:
            notes.append("F&T traps operate relatively quietly")

        logger.debug(
            f"Ultrasonic analysis: {status.value} (confidence {confidence:.0%})"
        )

        return UltrasonicDiagnosticResult(
            status=status,
            confidence=confidence,
            average_db=round(average_db, 1),
            peak_db=round(peak_db, 1),
            cycling_detected=cycling_detected,
            continuous_flow_detected=continuous_flow_detected,
            evidence=evidence,
            notes=notes,
        )

    @property
    def analysis_count(self) -> int:
        """Get analysis count."""
        return self._analysis_count


# =============================================================================
# TEMPERATURE DIFFERENTIAL ANALYZER
# =============================================================================

class TemperatureDifferentialAnalyzer:
    """
    Temperature differential analyzer for steam traps.

    Temperature-based diagnostics compare inlet (steam side) and outlet
    (condensate side) temperatures. Key principles:
    - Good trap: Inlet at saturation, outlet subcooled
    - Failed open: Outlet near saturation (steam passing through)
    - Failed closed: Large delta T (cold outlet, no flow)

    Reference: DOE Steam Tip Sheet #1
    """

    def __init__(
        self,
        thresholds: DiagnosticThresholds,
        config: SteamTrapMonitorConfig,
    ) -> None:
        """
        Initialize temperature analyzer.

        Args:
            thresholds: Diagnostic threshold configuration
            config: Agent configuration with steam properties
        """
        self.thresholds = thresholds.temperature
        self.config = config
        self._analysis_count = 0

    def analyze(
        self,
        readings: List[TemperatureReading],
        steam_pressure_psig: float,
        trap_type: TrapType,
        trap_config: Optional[TrapTypeConfig] = None,
    ) -> TemperatureDiagnosticResult:
        """
        Analyze temperature readings for trap status.

        Args:
            readings: List of temperature readings
            steam_pressure_psig: Operating steam pressure
            trap_type: Type of steam trap
            trap_config: Trap type configuration

        Returns:
            TemperatureDiagnosticResult
        """
        self._analysis_count += 1

        if not readings:
            return TemperatureDiagnosticResult(
                status=TrapStatus.UNKNOWN,
                confidence=0.0,
                inlet_temp_f=0.0,
                outlet_temp_f=0.0,
                delta_t_f=0.0,
                subcooling_f=0.0,
                notes=["No temperature readings provided"],
            )

        # Average readings
        inlet_temps = [r.inlet_temp_f for r in readings]
        outlet_temps = [r.outlet_temp_f for r in readings]
        inlet_temp = statistics.mean(inlet_temps)
        outlet_temp = statistics.mean(outlet_temps)
        delta_t = inlet_temp - outlet_temp

        # Get saturation temperature
        sat_temp = self.config.get_saturation_temperature(steam_pressure_psig)
        subcooling = sat_temp - outlet_temp

        # Apply trap-type adjustment
        temp_adjustment = 0.0
        expected_subcooling = 0.0
        if trap_config:
            temp_adjustment = trap_config.temperature_adjustment_f
            expected_subcooling = trap_config.subcooling_f
        elif trap_type == TrapType.THERMOSTATIC:
            expected_subcooling = 20.0
        elif trap_type == TrapType.BIMETALLIC:
            expected_subcooling = 40.0
        elif trap_type == TrapType.THERMODYNAMIC:
            expected_subcooling = 5.0

        evidence = []
        notes = []

        # Check if inlet is near saturation
        inlet_deviation = abs(inlet_temp - sat_temp)
        if inlet_deviation > 30:
            notes.append(
                f"Inlet temp {inlet_temp:.0f}F deviates from "
                f"saturation {sat_temp:.0f}F by {inlet_deviation:.0f}F"
            )

        # Diagnostic decision tree
        if inlet_temp < self.thresholds.min_inlet_temp_f:
            # Cold inlet - no steam supply
            status = TrapStatus.COLD
            confidence = 0.85
            evidence.append(f"Inlet temperature too low ({inlet_temp:.0f}F)")
            evidence.append("No steam supply to trap")

        elif delta_t > self.thresholds.failed_closed_delta_t_min_f:
            # Very high delta T - failed closed
            status = TrapStatus.FAILED_CLOSED
            confidence = 0.80
            evidence.append(f"Very high temperature differential ({delta_t:.0f}F)")
            evidence.append("Condensate not flowing through trap")

        elif delta_t < self.thresholds.failed_open_delta_t_max_f:
            # Very low delta T - steam passing through
            if outlet_temp > sat_temp - self.thresholds.outlet_max_above_sat_f:
                status = TrapStatus.FAILED_OPEN
                confidence = 0.85
                evidence.append(f"Very low temperature differential ({delta_t:.0f}F)")
                evidence.append(f"Outlet near saturation ({outlet_temp:.0f}F)")
            else:
                status = TrapStatus.LEAKING
                confidence = 0.70
                evidence.append(f"Low temperature differential ({delta_t:.0f}F)")

        elif delta_t < self.thresholds.good_delta_t_min_f:
            # Low delta T - possible leaking
            status = TrapStatus.LEAKING
            confidence = 0.65
            evidence.append(f"Low temperature differential ({delta_t:.0f}F)")
            evidence.append("Possible steam passage")

        elif self.thresholds.good_delta_t_min_f <= delta_t <= self.thresholds.good_delta_t_max_f:
            # Normal range
            status = TrapStatus.GOOD
            confidence = 0.75

            # Adjust for trap type expected subcooling
            if abs(subcooling - expected_subcooling) < 15:
                evidence.append(
                    f"Subcooling {subcooling:.0f}F appropriate for {trap_type.value}"
                )
                confidence = 0.85
            else:
                evidence.append(f"Temperature differential in normal range ({delta_t:.0f}F)")

        else:
            # High but not extreme delta T
            status = TrapStatus.GOOD
            confidence = 0.70
            evidence.append(f"Temperature differential ({delta_t:.0f}F) acceptable")
            notes.append("Higher than typical - verify flow")

        # Add ambient reference if available
        if readings[0].ambient_temp_f:
            ambient = readings[0].ambient_temp_f
            if outlet_temp < ambient + self.thresholds.ambient_delta_threshold_f:
                notes.append("Outlet temperature close to ambient - check flow")

        logger.debug(
            f"Temperature analysis: {status.value} (confidence {confidence:.0%})"
        )

        return TemperatureDiagnosticResult(
            status=status,
            confidence=confidence,
            inlet_temp_f=round(inlet_temp, 1),
            outlet_temp_f=round(outlet_temp, 1),
            delta_t_f=round(delta_t, 1),
            subcooling_f=round(subcooling, 1),
            evidence=evidence,
            notes=notes,
        )

    @property
    def analysis_count(self) -> int:
        """Get analysis count."""
        return self._analysis_count


# =============================================================================
# DIAGNOSTIC DECISION TREE
# =============================================================================

class DiagnosticDecisionTree:
    """
    Decision tree for combining multiple diagnostic methods.

    This class implements the logic for correlating results from
    ultrasonic, temperature, and visual diagnostics to produce
    a final diagnosis with confidence scoring.

    Decision Logic:
    1. If methods agree, high confidence
    2. If methods disagree, reduce confidence and note inconsistency
    3. Weight by method reliability for application
    """

    def __init__(self, thresholds: DiagnosticThresholds) -> None:
        """
        Initialize decision tree.

        Args:
            thresholds: Diagnostic threshold configuration
        """
        self.thresholds = thresholds

    def combine_results(
        self,
        ultrasonic: Optional[UltrasonicDiagnosticResult],
        temperature: Optional[TemperatureDiagnosticResult],
        visual: Optional[VisualInspectionReading],
    ) -> CombinedDiagnosticResult:
        """
        Combine results from multiple diagnostic methods.

        Args:
            ultrasonic: Ultrasonic analysis result
            temperature: Temperature analysis result
            visual: Visual inspection reading

        Returns:
            CombinedDiagnosticResult
        """
        methods_used = []
        all_evidence = []
        inconsistencies = []

        # Collect individual diagnoses
        diagnoses: List[Tuple[TrapStatus, float, str]] = []

        if ultrasonic and ultrasonic.status != TrapStatus.UNKNOWN:
            diagnoses.append((ultrasonic.status, ultrasonic.confidence, "ultrasonic"))
            methods_used.append(DiagnosticMethod.ULTRASONIC)
            all_evidence.extend(ultrasonic.evidence)

        if temperature and temperature.status != TrapStatus.UNKNOWN:
            diagnoses.append((temperature.status, temperature.confidence, "temperature"))
            methods_used.append(DiagnosticMethod.TEMPERATURE)
            all_evidence.extend(temperature.evidence)

        if visual:
            visual_status = self._interpret_visual(visual)
            if visual_status != TrapStatus.UNKNOWN:
                diagnoses.append((visual_status, 0.60, "visual"))
                methods_used.append(DiagnosticMethod.VISUAL)
                all_evidence.append(f"Visual inspection by {visual.inspector_id}")

        if not diagnoses:
            return CombinedDiagnosticResult(
                status=TrapStatus.UNKNOWN,
                confidence=0.0,
                confidence_level=DiagnosisConfidence.UNCERTAIN,
                failure_probabilities={},
                evidence=["No diagnostic data available"],
                inconsistencies=[],
                methods_used=[],
            )

        # Count status votes (weighted by confidence)
        status_scores: Dict[TrapStatus, float] = {}
        for status, confidence, method in diagnoses:
            current = status_scores.get(status, 0.0)
            status_scores[status] = current + confidence

        # Find consensus
        best_status = max(status_scores, key=status_scores.get)
        best_score = status_scores[best_status]
        total_score = sum(status_scores.values())

        # Check for agreement
        agreeing_methods = sum(
            1 for s, c, m in diagnoses if s == best_status
        )
        total_methods = len(diagnoses)

        # Calculate final confidence
        if agreeing_methods == total_methods:
            # All methods agree
            base_confidence = best_score / total_score
            final_confidence = min(0.95, base_confidence + 0.1)
        else:
            # Methods disagree
            base_confidence = best_score / total_score
            penalty = self.thresholds.disagreement_confidence_penalty
            final_confidence = max(0.30, base_confidence - penalty)

            # Note inconsistencies
            for status, conf, method in diagnoses:
                if status != best_status:
                    inconsistencies.append(
                        f"{method} indicates {status.value} "
                        f"(confidence {conf:.0%})"
                    )

        # Multi-method agreement bonus
        if self.thresholds.require_multi_method_agreement:
            if total_methods >= 2 and agreeing_methods >= 2:
                final_confidence = min(0.95, final_confidence + 0.05)
                all_evidence.append("Multiple methods agree on diagnosis")

        # Determine confidence level
        if final_confidence >= self.thresholds.high_confidence_threshold:
            confidence_level = DiagnosisConfidence.HIGH
        elif final_confidence >= self.thresholds.medium_confidence_threshold:
            confidence_level = DiagnosisConfidence.MEDIUM
        elif final_confidence >= 0.50:
            confidence_level = DiagnosisConfidence.LOW
        else:
            confidence_level = DiagnosisConfidence.UNCERTAIN

        # Calculate failure mode probabilities
        failure_probs = self._calculate_failure_probabilities(
            status_scores, total_score
        )

        logger.debug(
            f"Combined diagnosis: {best_status.value} "
            f"({confidence_level.value}, {final_confidence:.0%})"
        )

        return CombinedDiagnosticResult(
            status=best_status,
            confidence=round(final_confidence, 2),
            confidence_level=confidence_level,
            failure_probabilities=failure_probs,
            evidence=all_evidence,
            inconsistencies=inconsistencies,
            methods_used=methods_used,
        )

    def _interpret_visual(
        self,
        visual: VisualInspectionReading,
    ) -> TrapStatus:
        """Interpret visual inspection results."""
        if visual.visible_steam_discharge:
            return TrapStatus.FAILED_OPEN

        if visual.condensate_visible and visual.trap_cycling_observed:
            return TrapStatus.GOOD

        if not visual.condensate_visible and not visual.visible_steam_discharge:
            # No discharge - could be cold or blocked
            return TrapStatus.COLD

        if visual.leaks_detected:
            return TrapStatus.LEAKING

        return TrapStatus.UNKNOWN

    def _calculate_failure_probabilities(
        self,
        status_scores: Dict[TrapStatus, float],
        total_score: float,
    ) -> Dict[str, float]:
        """Calculate probability for each failure mode."""
        probs = {}

        if total_score > 0:
            for status, score in status_scores.items():
                if status in [TrapStatus.FAILED_OPEN, TrapStatus.FAILED_CLOSED, TrapStatus.LEAKING]:
                    probs[status.value] = round(score / total_score, 3)

        return probs


# =============================================================================
# FAILURE MODE DETECTOR
# =============================================================================

class FailureModeDetector:
    """
    Detector for specific failure modes with root cause indication.

    Provides detailed analysis of each failure mode with
    supporting evidence and recommended actions.
    """

    def analyze_failure_mode(
        self,
        status: TrapStatus,
        combined: CombinedDiagnosticResult,
        trap_type: TrapType,
    ) -> List[FailureModeProbability]:
        """
        Analyze failure mode probabilities with detail.

        Args:
            status: Primary diagnosed status
            combined: Combined diagnostic result
            trap_type: Type of steam trap

        Returns:
            List of FailureModeProbability with details
        """
        results = []

        # Failed Open analysis
        if TrapStatus.FAILED_OPEN.value in combined.failure_probabilities or status == TrapStatus.FAILED_OPEN:
            prob = combined.failure_probabilities.get(
                TrapStatus.FAILED_OPEN.value,
                0.8 if status == TrapStatus.FAILED_OPEN else 0.0
            )

            indicators = []
            contradictors = []

            for evidence in combined.evidence:
                if any(w in evidence.lower() for w in ["continuous", "high", "failed open", "steam"]):
                    indicators.append(evidence)
                elif any(w in evidence.lower() for w in ["cycling", "normal", "good"]):
                    contradictors.append(evidence)

            results.append(FailureModeProbability(
                failure_mode="failed_open",
                probability=prob,
                confidence=combined.confidence_level,
                indicators=indicators,
                contradictors=contradictors,
            ))

        # Failed Closed analysis
        if TrapStatus.FAILED_CLOSED.value in combined.failure_probabilities or status == TrapStatus.FAILED_CLOSED:
            prob = combined.failure_probabilities.get(
                TrapStatus.FAILED_CLOSED.value,
                0.8 if status == TrapStatus.FAILED_CLOSED else 0.0
            )

            indicators = []
            contradictors = []

            for evidence in combined.evidence:
                if any(w in evidence.lower() for w in ["cold", "blocked", "high differential", "no flow"]):
                    indicators.append(evidence)
                elif any(w in evidence.lower() for w in ["flow", "cycling", "steam"]):
                    contradictors.append(evidence)

            results.append(FailureModeProbability(
                failure_mode="failed_closed",
                probability=prob,
                confidence=combined.confidence_level,
                indicators=indicators,
                contradictors=contradictors,
            ))

        # Leaking analysis
        if TrapStatus.LEAKING.value in combined.failure_probabilities or status == TrapStatus.LEAKING:
            prob = combined.failure_probabilities.get(
                TrapStatus.LEAKING.value,
                0.7 if status == TrapStatus.LEAKING else 0.0
            )

            indicators = []
            for evidence in combined.evidence:
                if any(w in evidence.lower() for w in ["elevated", "leak", "partial"]):
                    indicators.append(evidence)

            results.append(FailureModeProbability(
                failure_mode="leaking",
                probability=prob,
                confidence=combined.confidence_level,
                indicators=indicators,
                contradictors=[],
            ))

        return results


# =============================================================================
# MAIN DIAGNOSTICS ENGINE
# =============================================================================

class TrapDiagnosticsEngine:
    """
    Main steam trap diagnostics engine.

    Integrates ultrasonic analysis, temperature differential analysis,
    and visual inspection to provide comprehensive trap diagnosis.
    Implements decision tree logic for failure mode classification.

    All diagnostics are ZERO-HALLUCINATION: deterministic decision
    trees with documented thresholds.

    Example:
        >>> engine = TrapDiagnosticsEngine(config)
        >>> result = engine.diagnose(diagnostic_input)
        >>> print(f"Status: {result.condition.status}")
        >>> print(f"Confidence: {result.condition.confidence_score:.0%}")
    """

    def __init__(self, config: SteamTrapMonitorConfig) -> None:
        """
        Initialize diagnostics engine.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._ultrasonic_analyzer = UltrasonicAnalyzer(config.diagnostics)
        self._temperature_analyzer = TemperatureDifferentialAnalyzer(
            config.diagnostics, config
        )
        self._decision_tree = DiagnosticDecisionTree(config.diagnostics)
        self._failure_detector = FailureModeDetector()
        self._diagnosis_count = 0

        logger.info("TrapDiagnosticsEngine initialized")

    def diagnose(
        self,
        input_data: TrapDiagnosticInput,
    ) -> TrapDiagnosticOutput:
        """
        Perform comprehensive trap diagnosis.

        Args:
            input_data: Diagnostic input data

        Returns:
            TrapDiagnosticOutput with complete diagnosis
        """
        start_time = datetime.now(timezone.utc)
        self._diagnosis_count += 1

        trap_info = input_data.trap_info
        trap_type = TrapType(trap_info.trap_type)
        trap_config = self.config.get_trap_type_config(trap_info.trap_type)

        # Ultrasonic analysis
        ultrasonic_result = None
        if input_data.ultrasonic_readings:
            ultrasonic_result = self._ultrasonic_analyzer.analyze(
                readings=input_data.ultrasonic_readings,
                trap_type=trap_type,
                trap_config=trap_config,
            )

        # Temperature analysis
        temperature_result = None
        if input_data.temperature_readings:
            temperature_result = self._temperature_analyzer.analyze(
                readings=input_data.temperature_readings,
                steam_pressure_psig=input_data.steam_pressure_psig,
                trap_type=trap_type,
                trap_config=trap_config,
            )

        # Combine results
        combined = self._decision_tree.combine_results(
            ultrasonic=ultrasonic_result,
            temperature=temperature_result,
            visual=input_data.visual_inspection,
        )

        # Detailed failure analysis
        failure_probs = self._failure_detector.analyze_failure_mode(
            status=combined.status,
            combined=combined,
            trap_type=trap_type,
        )

        # Build condition assessment
        condition = TrapCondition(
            status=combined.status,
            confidence=combined.confidence_level,
            confidence_score=combined.confidence,
            failed_open_probability=combined.failure_probabilities.get("failed_open", 0.0),
            failed_closed_probability=combined.failure_probabilities.get("failed_closed", 0.0),
            leaking_probability=combined.failure_probabilities.get("leaking", 0.0),
            ultrasonic_assessment=ultrasonic_result.status.value if ultrasonic_result else None,
            temperature_assessment=temperature_result.status.value if temperature_result else None,
            visual_assessment=input_data.visual_inspection.trap_body_condition if input_data.visual_inspection else None,
            evidence=combined.evidence,
            inconsistencies=combined.inconsistencies,
        )

        # Calculate health score
        health_score = self._calculate_health_score(combined)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            combined=combined,
            trap_info=trap_info,
        )

        # Calculate processing time
        end_time = datetime.now(timezone.utc)
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        # Provenance hash
        provenance_data = {
            "trap_id": trap_info.trap_id,
            "status": combined.status.value,
            "confidence": combined.confidence,
            "timestamp": end_time.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        # Check ASME B16.34 compliance
        asme_compliant = True
        if trap_info.pressure_rating_psig < input_data.steam_pressure_psig:
            asme_compliant = False

        logger.info(
            f"Diagnosis complete for {trap_info.trap_id}: "
            f"{combined.status.value} ({combined.confidence:.0%})"
        )

        return TrapDiagnosticOutput(
            request_id=input_data.request_id,
            trap_id=trap_info.trap_id,
            timestamp=end_time,
            status="success",
            processing_time_ms=processing_time_ms,
            condition=condition,
            health_score=health_score,
            failure_probabilities=failure_probs,
            steam_loss=SteamLossEstimate(),  # Calculated separately
            recommendations=recommendations,
            diagnostic_methods_used=[m.value for m in combined.methods_used],
            sensor_data_quality=self._assess_data_quality(input_data),
            asme_b16_34_compliant=asme_compliant,
            pressure_rating_adequate=asme_compliant,
            provenance_hash=provenance_hash,
        )

    def _calculate_health_score(
        self,
        combined: CombinedDiagnosticResult,
    ) -> TrapHealthScore:
        """Calculate trap health score from diagnosis."""
        status = combined.status

        # Base score by status
        score_by_status = {
            TrapStatus.GOOD: 95.0,
            TrapStatus.LEAKING: 60.0,
            TrapStatus.FAILED_OPEN: 20.0,
            TrapStatus.FAILED_CLOSED: 25.0,
            TrapStatus.COLD: 50.0,
            TrapStatus.FLOODED: 40.0,
            TrapStatus.UNKNOWN: 50.0,
        }

        base_score = score_by_status.get(status, 50.0)

        # Adjust by confidence
        confidence_factor = combined.confidence
        adjusted_score = base_score * confidence_factor + 50 * (1 - confidence_factor)

        # Categorize
        if adjusted_score >= 90:
            category = "excellent"
        elif adjusted_score >= 75:
            category = "good"
        elif adjusted_score >= 50:
            category = "fair"
        elif adjusted_score >= 25:
            category = "poor"
        else:
            category = "critical"

        return TrapHealthScore(
            overall_score=round(adjusted_score, 1),
            category=category,
            thermal_efficiency_score=adjusted_score if status == TrapStatus.GOOD else adjusted_score * 0.8,
            mechanical_condition_score=adjusted_score,
            operational_score=adjusted_score,
            trend=TrendDirection.STABLE,  # Requires historical data
        )

    def _generate_recommendations(
        self,
        combined: CombinedDiagnosticResult,
        trap_info,
    ) -> List[MaintenanceRecommendation]:
        """Generate maintenance recommendations based on diagnosis."""
        recommendations = []
        status = combined.status

        if status == TrapStatus.FAILED_OPEN:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.URGENT,
                action="Replace steam trap",
                description=(
                    f"Trap {trap_info.trap_id} has failed open and is passing "
                    "live steam. Replace immediately to stop energy loss."
                ),
                deadline_hours=24.0,
                estimated_duration_hours=2.0,
                estimated_cost_usd=750.0,
                parts_required=["Replacement steam trap", "Gaskets"],
                reason="Failed open - continuous steam loss",
                potential_savings_usd=5000.0,  # Estimated annual
            ))

        elif status == TrapStatus.FAILED_CLOSED:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.HIGH,
                action="Clear or replace steam trap",
                description=(
                    f"Trap {trap_info.trap_id} has failed closed. Condensate "
                    "backup may cause waterhammer or equipment damage."
                ),
                deadline_hours=48.0,
                estimated_duration_hours=2.0,
                estimated_cost_usd=500.0,
                parts_required=["Repair kit or replacement trap", "Gaskets"],
                reason="Failed closed - condensate backup risk",
            ))

        elif status == TrapStatus.LEAKING:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.MEDIUM,
                action="Inspect and repair steam trap",
                description=(
                    f"Trap {trap_info.trap_id} is leaking steam. "
                    "Inspect internals and repair or replace as needed."
                ),
                deadline_hours=168.0,  # 1 week
                estimated_duration_hours=1.5,
                estimated_cost_usd=350.0,
                parts_required=["Repair kit", "Gaskets"],
                reason="Steam leakage detected",
                potential_savings_usd=1500.0,
            ))

        elif status == TrapStatus.COLD:
            recommendations.append(MaintenanceRecommendation(
                priority=MaintenancePriority.LOW,
                action="Verify steam supply and trap operation",
                description=(
                    f"Trap {trap_info.trap_id} shows no activity. "
                    "Verify steam supply is on and trap is not isolated."
                ),
                deadline_hours=336.0,  # 2 weeks
                estimated_duration_hours=0.5,
                estimated_cost_usd=0.0,
                parts_required=[],
                reason="No trap activity detected",
            ))

        return recommendations

    def _assess_data_quality(
        self,
        input_data: TrapDiagnosticInput,
    ) -> float:
        """Assess quality of input sensor data."""
        scores = []

        # Ultrasonic data quality
        if input_data.ultrasonic_readings:
            us_quality = statistics.mean(
                r.quality_score for r in input_data.ultrasonic_readings
            )
            scores.append(us_quality)

        # Temperature data quality
        if input_data.temperature_readings:
            temp_quality = statistics.mean(
                r.quality_score for r in input_data.temperature_readings
            )
            scores.append(temp_quality)

        # Visual inspection adds confidence
        if input_data.visual_inspection:
            scores.append(0.8)

        if not scores:
            return 0.0

        return round(statistics.mean(scores), 2)

    @property
    def diagnosis_count(self) -> int:
        """Get total diagnosis count."""
        return self._diagnosis_count
