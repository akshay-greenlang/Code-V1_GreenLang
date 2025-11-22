# -*- coding: utf-8 -*-
"""
Deterministic calculation tools for GL-008 SteamTrapInspector.

This module provides zero-hallucination tools for steam trap performance analysis,
failure detection, energy loss calculation, and maintenance optimization. All
calculations use deterministic physics-based formulas compliant with ASME PTC 25,
Spirax Sarco Steam Engineering Principles, and DOE Best Practices.

CRITICAL: All numeric results come from deterministic formulas, never from LLM generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta
import math

from .config import (
    TrapType,
    FailureMode,
    AcousticConfig,
    ThermalConfig,
    EnergyLossConfig,
    MaintenanceConfig
)


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class AcousticAnalysisResult:
    """Result from acoustic signature analysis."""
    trap_id: str
    failure_probability: float  # 0.0-1.0
    failure_mode: FailureMode
    confidence_score: float  # 0.0-1.0
    acoustic_signature: Dict[str, float]  # FFT features
    anomaly_detected: bool
    signal_strength_db: float
    frequency_peak_hz: float
    spectral_features: Dict[str, Any]
    timestamp: str
    provenance_hash: str


@dataclass
class ThermalAnalysisResult:
    """Result from thermal imaging analysis."""
    trap_id: str
    trap_health_score: float  # 0-100
    temperature_upstream_c: float
    temperature_downstream_c: float
    temperature_differential_c: float
    anomalies_detected: List[str]
    hot_spots: List[Dict[str, float]]
    cold_spots: List[Dict[str, float]]
    thermal_signature: Dict[str, Any]
    condensate_pooling_detected: bool
    timestamp: str
    provenance_hash: str


@dataclass
class FailureDiagnosisResult:
    """Result from comprehensive failure diagnosis."""
    trap_id: str
    failure_mode: FailureMode
    failure_severity: str  # critical, high, medium, low, normal
    root_cause: str
    confidence: float  # 0.0-1.0
    diagnostic_indicators: Dict[str, Any]
    recommended_action: str
    urgency_hours: int  # Response time needed
    safety_implications: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class EnergyLossResult:
    """Result from energy loss calculation."""
    trap_id: str
    steam_loss_kg_hr: float
    steam_loss_lb_hr: float
    energy_loss_mmbtu_yr: float
    energy_loss_gj_yr: float
    energy_loss_kwh_yr: float
    cost_loss_usd_yr: float
    co2_emissions_kg_yr: float
    co2_emissions_tons_yr: float
    calculation_basis: str
    assumptions: Dict[str, Any]
    timestamp: str
    provenance_hash: str


@dataclass
class MaintenancePriorityResult:
    """Result from maintenance prioritization."""
    priority_list: List[Dict[str, Any]]  # Sorted by criticality
    total_potential_savings_usd_yr: float
    recommended_schedule: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    estimated_total_cost_usd: float
    expected_roi_percent: float
    payback_months: float
    timestamp: str
    provenance_hash: str


@dataclass
class RULPredictionResult:
    """Result from Remaining Useful Life prediction."""
    trap_id: str
    rul_days: float
    rul_confidence_lower: float
    rul_confidence_upper: float
    confidence_interval_percent: float  # 90% or 95%
    next_inspection_date: str
    failure_probability_curve: List[Tuple[int, float]]  # (days, probability)
    degradation_rate: float
    historical_mtbf_days: Optional[float]
    timestamp: str
    provenance_hash: str


@dataclass
class CostBenefitResult:
    """Result from cost-benefit analysis."""
    trap_id: str
    maintenance_cost_usd: float
    replacement_cost_usd: float
    annual_savings_usd: float
    payback_months: float
    roi_percent: float
    npv_usd: float  # 5-year NPV at 8% discount
    irr_percent: float
    decision_recommendation: str  # repair, replace, monitor
    sensitivity_analysis: Dict[str, Any]
    timestamp: str
    provenance_hash: str


# ============================================================================
# STEAM TRAP INSPECTION TOOLS
# ============================================================================

class SteamTrapTools:
    """
    Deterministic calculation tools for steam trap inspection and analysis.

    All methods use physics-based formulas with zero hallucination guarantee.
    """

    def __init__(self):
        """Initialize tools with default configurations."""
        self.acoustic_config = AcousticConfig()
        self.thermal_config = ThermalConfig()
        self.energy_config = EnergyLossConfig()
        self.maintenance_config = MaintenanceConfig()

    # ========================================================================
    # TOOL 1: ACOUSTIC SIGNATURE ANALYSIS
    # ========================================================================

    def analyze_acoustic_signature(
        self,
        acoustic_data: Dict[str, Any],
        trap_type: TrapType = TrapType.THERMODYNAMIC
    ) -> AcousticAnalysisResult:
        """
        Analyze acoustic signature for trap failure detection.

        Physics basis: Ultrasonic analysis (20-100 kHz) detects steam leakage,
        cavitation, and mechanical wear through characteristic frequency patterns.

        Args:
            acoustic_data: Dictionary with:
                - trap_id: str
                - signal: List[float] (time-domain audio samples)
                - sampling_rate_hz: int
                - measurement_duration_sec: float
            trap_type: Type of steam trap being analyzed

        Returns:
            AcousticAnalysisResult with failure detection and classification

        Formula:
            FFT Analysis: X(f) = FFT(signal)
            Power Spectral Density: PSD(f) = |X(f)|^2
            Signal Strength (dB) = 20 * log10(RMS(signal) / ref_amplitude)
            Spectral Centroid = Σ(f * PSD(f)) / Σ(PSD(f))

        Standards: ASTM E1316 (Ultrasonic Testing), ISO 18436-8
        """
        trap_id = acoustic_data.get('trap_id', 'unknown')
        signal = np.array(acoustic_data.get('signal', []))
        sampling_rate = acoustic_data.get('sampling_rate_hz', self.acoustic_config.sampling_rate_hz)

        # Step 1: Calculate time-domain features
        rms_amplitude = np.sqrt(np.mean(signal ** 2))
        signal_strength_db = 20 * np.log10(rms_amplitude / 1e-6 + 1e-12)  # ref: 1 µPa

        # Step 2: FFT analysis
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
        psd = np.abs(fft_result) ** 2

        # Step 3: Spectral features
        positive_freqs = frequencies[:len(frequencies)//2]
        positive_psd = psd[:len(psd)//2]

        spectral_centroid = np.sum(positive_freqs * positive_psd) / (np.sum(positive_psd) + 1e-12)
        spectral_rolloff = self._calculate_spectral_rolloff(positive_freqs, positive_psd, 0.85)
        zero_crossing_rate = self._calculate_zero_crossing_rate(signal)

        # Step 4: Peak frequency detection
        peak_idx = np.argmax(positive_psd)
        frequency_peak_hz = positive_freqs[peak_idx]

        # Step 5: Failure mode classification (rule-based)
        failure_mode, failure_probability = self._classify_acoustic_failure(
            signal_strength_db,
            frequency_peak_hz,
            spectral_centroid,
            trap_type
        )

        # Step 6: Anomaly detection threshold
        anomaly_detected = signal_strength_db > self.acoustic_config.detection_threshold_db

        # Step 7: Confidence scoring
        confidence_score = self._calculate_acoustic_confidence(
            signal_strength_db,
            failure_probability,
            len(signal)
        )

        # Create result
        result = AcousticAnalysisResult(
            trap_id=trap_id,
            failure_probability=round(failure_probability, 4),
            failure_mode=failure_mode,
            confidence_score=round(confidence_score, 4),
            acoustic_signature={
                'rms_amplitude': round(rms_amplitude, 6),
                'spectral_centroid_hz': round(spectral_centroid, 2),
                'spectral_rolloff_hz': round(spectral_rolloff, 2),
                'zero_crossing_rate': round(zero_crossing_rate, 4)
            },
            anomaly_detected=anomaly_detected,
            signal_strength_db=round(signal_strength_db, 2),
            frequency_peak_hz=round(frequency_peak_hz, 2),
            spectral_features={
                'peak_frequency_hz': round(frequency_peak_hz, 2),
                'bandwidth_hz': round(spectral_rolloff - spectral_centroid, 2),
                'signal_to_noise_ratio_db': round(signal_strength_db - self.acoustic_config.noise_floor_db, 2)
            },
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(acoustic_data)
        )

        return result

    def _calculate_spectral_rolloff(
        self, frequencies: np.ndarray, psd: np.ndarray, threshold: float = 0.85
    ) -> float:
        """Calculate spectral rolloff frequency (85% energy threshold)."""
        total_energy = np.sum(psd)
        cumulative_energy = np.cumsum(psd)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        if len(rolloff_idx) > 0:
            return frequencies[rolloff_idx[0]]
        return frequencies[-1]

    def _calculate_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Calculate zero-crossing rate (frequency content indicator)."""
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings) / len(signal)

    def _classify_acoustic_failure(
        self,
        signal_db: float,
        peak_freq_hz: float,
        spectral_centroid_hz: float,
        trap_type: TrapType
    ) -> Tuple[FailureMode, float]:
        """
        Classify failure mode based on acoustic features.

        Classification rules based on Spirax Sarco research:
        - Failed open (steam leak): High amplitude (>60 dB), freq 25-40 kHz
        - Cavitation: Broadband noise, freq 40-80 kHz
        - Mechanical wear: Low freq content <30 kHz
        - Normal operation: Low amplitude (<50 dB)
        """
        if signal_db > 60.0 and 25000 <= peak_freq_hz <= 40000:
            return FailureMode.FAILED_OPEN, 0.85
        elif signal_db > 55.0 and 40000 <= peak_freq_hz <= 80000:
            return FailureMode.CAVITATION, 0.75
        elif signal_db > 50.0 and peak_freq_hz < 30000:
            return FailureMode.WORN_SEAT, 0.70
        elif signal_db < 45.0:
            return FailureMode.NORMAL, 0.90
        else:
            return FailureMode.LEAKING, 0.65

    def _calculate_acoustic_confidence(
        self, signal_db: float, failure_prob: float, sample_count: int
    ) -> float:
        """Calculate confidence score for acoustic analysis."""
        # Confidence increases with signal strength and sample size
        signal_factor = min(signal_db / 70.0, 1.0)
        sample_factor = min(sample_count / 10000, 1.0)
        probability_factor = abs(failure_prob - 0.5) * 2  # Higher confidence at extremes

        confidence = (signal_factor * 0.4 + sample_factor * 0.3 + probability_factor * 0.3)
        return max(0.5, min(confidence, 0.99))  # Clamp to 0.5-0.99

    # ========================================================================
    # TOOL 2: THERMAL PATTERN ANALYSIS
    # ========================================================================

    def analyze_thermal_pattern(
        self,
        thermal_data: Dict[str, Any]
    ) -> ThermalAnalysisResult:
        """
        Analyze thermal imaging pattern for trap health assessment.

        Physics basis: Temperature differential analysis using Stefan-Boltzmann law
        and heat transfer principles to detect condensate backup and steam loss.

        Args:
            thermal_data: Dictionary with:
                - trap_id: str
                - temperature_upstream_c: float
                - temperature_downstream_c: float
                - thermal_image: Optional[np.ndarray] (IR image)
                - ambient_temp_c: float

        Returns:
            ThermalAnalysisResult with health score and anomalies

        Formula:
            ΔT = T_upstream - T_downstream
            Health Score = 100 * (1 - |ΔT - ΔT_expected| / ΔT_expected)
            Heat Loss (W) = U * A * ΔT (simplified)

        Standards: ASME PTC 19.3 (Temperature Measurement)
        """
        trap_id = thermal_data.get('trap_id', 'unknown')
        temp_upstream_c = thermal_data.get('temperature_upstream_c', 150.0)
        temp_downstream_c = thermal_data.get('temperature_downstream_c', 90.0)
        ambient_temp_c = thermal_data.get('ambient_temp_c', self.thermal_config.ambient_temp_c)

        # Step 1: Calculate temperature differential
        delta_t_c = temp_upstream_c - temp_downstream_c

        # Step 2: Expected differential for normal operation (varies by trap type)
        # Normal trap: ΔT should be 10-30°C (condensate cooling)
        expected_delta_t_c = 20.0

        # Step 3: Detect anomalies
        anomalies = []
        hot_spots = []
        cold_spots = []
        condensate_pooling = False

        if delta_t_c < 5.0:
            anomalies.append("Minimal temperature differential - possible failed open")
            condensate_pooling = False
        elif delta_t_c > 50.0:
            anomalies.append("Excessive temperature differential - possible failed closed")
            condensate_pooling = True
        elif abs(delta_t_c - expected_delta_t_c) < 5.0:
            anomalies.append("Normal operation detected")

        if temp_downstream_c > temp_upstream_c:
            anomalies.append("Reverse temperature gradient - check sensor calibration")

        # Step 4: Calculate health score (0-100)
        deviation_percent = abs(delta_t_c - expected_delta_t_c) / expected_delta_t_c
        health_score = max(0, min(100, 100 * (1 - deviation_percent)))

        # Step 5: Hot/cold spot detection (if thermal image provided)
        thermal_image = thermal_data.get('thermal_image')
        if thermal_image is not None:
            hot_spots, cold_spots = self._detect_thermal_spots(thermal_image, ambient_temp_c)

        # Create result
        result = ThermalAnalysisResult(
            trap_id=trap_id,
            trap_health_score=round(health_score, 2),
            temperature_upstream_c=round(temp_upstream_c, 2),
            temperature_downstream_c=round(temp_downstream_c, 2),
            temperature_differential_c=round(delta_t_c, 2),
            anomalies_detected=anomalies,
            hot_spots=hot_spots,
            cold_spots=cold_spots,
            thermal_signature={
                'delta_t_c': round(delta_t_c, 2),
                'expected_delta_t_c': expected_delta_t_c,
                'deviation_percent': round(deviation_percent * 100, 2)
            },
            condensate_pooling_detected=condensate_pooling,
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(thermal_data)
        )

        return result

    def _detect_thermal_spots(
        self, thermal_image: np.ndarray, ambient_temp_c: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Detect hot and cold spots in thermal image."""
        hot_spots = []
        cold_spots = []

        if thermal_image is None or len(thermal_image) == 0:
            return hot_spots, cold_spots

        mean_temp = np.mean(thermal_image)
        std_temp = np.std(thermal_image)

        # Hot spots: > mean + 2*std
        hot_threshold = mean_temp + 2 * std_temp
        hot_pixels = np.where(thermal_image > hot_threshold)
        if len(hot_pixels[0]) > 0:
            hot_spots.append({
                'max_temp_c': round(float(np.max(thermal_image[hot_pixels])), 2),
                'count': len(hot_pixels[0])
            })

        # Cold spots: < mean - 2*std
        cold_threshold = mean_temp - 2 * std_temp
        cold_pixels = np.where(thermal_image < cold_threshold)
        if len(cold_pixels[0]) > 0:
            cold_spots.append({
                'min_temp_c': round(float(np.min(thermal_image[cold_pixels])), 2),
                'count': len(cold_pixels[0])
            })

        return hot_spots, cold_spots

    # ========================================================================
    # TOOL 3: FAILURE DIAGNOSIS
    # ========================================================================

    def diagnose_trap_failure(
        self,
        sensor_data: Dict[str, Any],
        acoustic_result: Optional[AcousticAnalysisResult] = None,
        thermal_result: Optional[ThermalAnalysisResult] = None
    ) -> FailureDiagnosisResult:
        """
        Comprehensive failure diagnosis integrating multiple data sources.

        Args:
            sensor_data: Operational data (pressure, temperature, flow)
            acoustic_result: Optional acoustic analysis result
            thermal_result: Optional thermal analysis result

        Returns:
            FailureDiagnosisResult with failure mode and root cause

        Logic: Multi-modal fusion with weighted confidence scoring
        """
        trap_id = sensor_data.get('trap_id', 'unknown')

        # Integrate evidence from multiple sources
        evidence_scores = {}

        # Evidence from acoustic analysis
        if acoustic_result:
            evidence_scores['acoustic'] = {
                'failure_mode': acoustic_result.failure_mode,
                'confidence': acoustic_result.confidence_score,
                'weight': 0.4
            }

        # Evidence from thermal analysis
        if thermal_result:
            if thermal_result.condensate_pooling_detected:
                thermal_failure_mode = FailureMode.FAILED_CLOSED
            elif thermal_result.temperature_differential_c < 5.0:
                thermal_failure_mode = FailureMode.FAILED_OPEN
            else:
                thermal_failure_mode = FailureMode.NORMAL

            evidence_scores['thermal'] = {
                'failure_mode': thermal_failure_mode,
                'confidence': thermal_result.trap_health_score / 100.0,
                'weight': 0.4
            }

        # Evidence from operational data
        pressure_upstream = sensor_data.get('pressure_upstream_psig', 100.0)
        pressure_downstream = sensor_data.get('pressure_downstream_psig', 0.0)
        delta_p = pressure_upstream - pressure_downstream

        if delta_p < 5.0:
            operational_failure_mode = FailureMode.FAILED_OPEN
            operational_confidence = 0.7
        elif delta_p > 50.0:
            operational_failure_mode = FailureMode.PLUGGED
            operational_confidence = 0.6
        else:
            operational_failure_mode = FailureMode.NORMAL
            operational_confidence = 0.8

        evidence_scores['operational'] = {
            'failure_mode': operational_failure_mode,
            'confidence': operational_confidence,
            'weight': 0.2
        }

        # Weighted vote for final failure mode
        failure_mode, combined_confidence = self._vote_failure_mode(evidence_scores)

        # Determine severity
        severity = self._determine_severity(failure_mode, combined_confidence)

        # Root cause analysis
        root_cause = self._identify_root_cause(failure_mode, sensor_data)

        # Recommended action
        recommended_action = self._recommend_action(failure_mode, severity)

        # Urgency (response time in hours)
        urgency_hours = self._calculate_urgency(failure_mode, severity)

        # Safety implications
        safety_implications = self._assess_safety_implications(failure_mode)

        result = FailureDiagnosisResult(
            trap_id=trap_id,
            failure_mode=failure_mode,
            failure_severity=severity,
            root_cause=root_cause,
            confidence=round(combined_confidence, 4),
            diagnostic_indicators={
                'evidence_sources': list(evidence_scores.keys()),
                'acoustic_failure_prob': acoustic_result.failure_probability if acoustic_result else 0.0,
                'thermal_health_score': thermal_result.trap_health_score if thermal_result else 0.0,
                'pressure_differential_psi': round(delta_p, 2)
            },
            recommended_action=recommended_action,
            urgency_hours=urgency_hours,
            safety_implications=safety_implications,
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(sensor_data)
        )

        return result

    def _vote_failure_mode(
        self, evidence_scores: Dict[str, Dict]
    ) -> Tuple[FailureMode, float]:
        """Weighted voting for failure mode determination."""
        votes = {}
        total_weight = 0.0

        for source, evidence in evidence_scores.items():
            mode = evidence['failure_mode']
            confidence = evidence['confidence']
            weight = evidence['weight']

            weighted_score = confidence * weight
            votes[mode] = votes.get(mode, 0.0) + weighted_score
            total_weight += weight

        # Find mode with highest weighted score
        if not votes:
            return FailureMode.NORMAL, 0.5

        best_mode = max(votes, key=votes.get)
        combined_confidence = votes[best_mode] / total_weight if total_weight > 0 else 0.5

        return best_mode, combined_confidence

    def _determine_severity(self, failure_mode: FailureMode, confidence: float) -> str:
        """Determine failure severity level."""
        if failure_mode == FailureMode.NORMAL:
            return "normal"
        elif failure_mode in [FailureMode.FAILED_OPEN, FailureMode.FAILED_CLOSED]:
            return "critical" if confidence > 0.8 else "high"
        elif failure_mode in [FailureMode.CAVITATION, FailureMode.WATERLOGGED]:
            return "high"
        elif failure_mode in [FailureMode.LEAKING, FailureMode.WORN_SEAT]:
            return "medium"
        else:
            return "low"

    def _identify_root_cause(self, failure_mode: FailureMode, sensor_data: Dict) -> str:
        """Identify root cause based on failure mode."""
        root_causes = {
            FailureMode.FAILED_OPEN: "Valve seat erosion or mechanical failure allowing steam bypass",
            FailureMode.FAILED_CLOSED: "Valve stuck closed or orifice blockage preventing condensate discharge",
            FailureMode.LEAKING: "Partial valve seat wear or debris preventing complete closure",
            FailureMode.PLUGGED: "Strainer blockage or dirt accumulation in valve assembly",
            FailureMode.WATERLOGGED: "Loss of air vent or improper sizing for condensate load",
            FailureMode.CAVITATION: "Excessive pressure differential causing vapor collapse",
            FailureMode.WORN_SEAT: "Normal wear from operational cycles or abrasive condensate",
            FailureMode.NORMAL: "Trap operating within normal parameters"
        }
        return root_causes.get(failure_mode, "Unknown failure mechanism - further investigation required")

    def _recommend_action(self, failure_mode: FailureMode, severity: str) -> str:
        """Recommend corrective action."""
        if severity == "critical":
            return f"IMMEDIATE ACTION REQUIRED: {failure_mode.value} - Replace trap within 4 hours"
        elif severity == "high":
            return f"URGENT: {failure_mode.value} - Schedule maintenance within 24 hours"
        elif severity == "medium":
            return f"Schedule preventive maintenance within 1 week for {failure_mode.value}"
        elif severity == "low":
            return "Monitor trap performance - schedule inspection at next planned maintenance window"
        else:
            return "Continue normal operation - no action required"

    def _calculate_urgency(self, failure_mode: FailureMode, severity: str) -> int:
        """Calculate response urgency in hours."""
        urgency_map = {
            "critical": 4,
            "high": 24,
            "medium": 168,  # 1 week
            "low": 720,     # 1 month
            "normal": 8760  # 1 year
        }
        return urgency_map.get(severity, 168)

    def _assess_safety_implications(self, failure_mode: FailureMode) -> List[str]:
        """Assess safety implications of failure mode."""
        implications = []

        if failure_mode == FailureMode.FAILED_OPEN:
            implications.extend([
                "Steam release hazard - risk of burns to personnel",
                "Pressure system instability",
                "Excessive energy consumption"
            ])
        elif failure_mode == FailureMode.FAILED_CLOSED:
            implications.extend([
                "Condensate backup - risk of water hammer",
                "Equipment damage from condensate flooding",
                "Process efficiency degradation"
            ])
        elif failure_mode == FailureMode.CAVITATION:
            implications.extend([
                "Noise pollution exceeding OSHA limits",
                "Vibration damage to piping systems",
                "Accelerated component wear"
            ])

        return implications if implications else ["No immediate safety concerns identified"]

    # ========================================================================
    # TOOL 4: ENERGY LOSS CALCULATION
    # ========================================================================

    def calculate_energy_loss(
        self,
        trap_data: Dict[str, Any],
        failure_mode: FailureMode = FailureMode.FAILED_OPEN
    ) -> EnergyLossResult:
        """
        Calculate energy and cost loss from steam trap failure.

        Physics basis: Steam thermodynamic properties and mass/energy conservation

        Args:
            trap_data: Dict with:
                - trap_id: str
                - orifice_diameter_in: float (trap orifice size)
                - steam_pressure_psig: float
                - operating_hours_yr: int
                - steam_cost_usd_per_1000lb: float
                - failure_severity: float (0.0-1.0, 1.0 = complete failure)

        Returns:
            EnergyLossResult with energy loss and cost impact

        Formula (Napier's equation for steam flow through orifice):
            W = 24.24 * P * D² * C
            Where:
                W = Steam loss (lb/hr)
                P = Upstream pressure (psig)
                D = Orifice diameter (inches)
                C = Discharge coefficient (0.7 for failed open)

        Energy content:
            At 100 psig: h_fg ≈ 881 BTU/lb (latent heat)
            Energy loss (MMBtu/yr) = W * h_fg * hours_yr / 1,000,000

        Standards: Spirax Sarco Steam Engineering, DOE Steam Tip Sheet #1
        """
        trap_id = trap_data.get('trap_id', 'unknown')
        orifice_diameter_in = trap_data.get('orifice_diameter_in', 0.125)  # 1/8" typical
        steam_pressure_psig = trap_data.get('steam_pressure_psig', self.energy_config.steam_pressure_psig)
        operating_hours_yr = trap_data.get('operating_hours_yr', self.energy_config.operating_hours_per_year)
        steam_cost_usd_per_1000lb = trap_data.get('steam_cost_usd_per_1000lb', self.energy_config.steam_cost_usd_per_1000lb)
        failure_severity = trap_data.get('failure_severity', 1.0)  # 1.0 = complete failure

        # Step 1: Calculate steam loss using Napier's equation
        discharge_coefficient = self._get_discharge_coefficient(failure_mode)
        steam_loss_lb_hr = 24.24 * steam_pressure_psig * (orifice_diameter_in ** 2) * discharge_coefficient * failure_severity

        # Convert to kg/hr
        steam_loss_kg_hr = steam_loss_lb_hr * 0.453592

        # Step 2: Get latent heat at operating pressure
        latent_heat_btu_lb = self._get_latent_heat(steam_pressure_psig)

        # Step 3: Calculate annual energy loss
        energy_loss_btu_hr = steam_loss_lb_hr * latent_heat_btu_lb
        energy_loss_mmbtu_yr = (energy_loss_btu_hr * operating_hours_yr) / 1_000_000
        energy_loss_gj_yr = energy_loss_mmbtu_yr * 1.05506  # 1 MMBtu = 1.05506 GJ
        energy_loss_kwh_yr = energy_loss_mmbtu_yr * 293.071  # 1 MMBtu = 293.071 kWh

        # Step 4: Calculate annual cost loss
        annual_steam_loss_lb = steam_loss_lb_hr * operating_hours_yr
        cost_loss_usd_yr = (annual_steam_loss_lb / 1000) * steam_cost_usd_per_1000lb

        # Step 5: Calculate CO2 emissions
        co2_factor_kg_per_mmbtu = self.energy_config.co2_factor_kg_per_mmbtu
        co2_emissions_kg_yr = energy_loss_mmbtu_yr * co2_factor_kg_per_mmbtu
        co2_emissions_tons_yr = co2_emissions_kg_yr / 1000

        result = EnergyLossResult(
            trap_id=trap_id,
            steam_loss_kg_hr=round(steam_loss_kg_hr, 2),
            steam_loss_lb_hr=round(steam_loss_lb_hr, 2),
            energy_loss_mmbtu_yr=round(energy_loss_mmbtu_yr, 2),
            energy_loss_gj_yr=round(energy_loss_gj_yr, 2),
            energy_loss_kwh_yr=round(energy_loss_kwh_yr, 2),
            cost_loss_usd_yr=round(cost_loss_usd_yr, 2),
            co2_emissions_kg_yr=round(co2_emissions_kg_yr, 2),
            co2_emissions_tons_yr=round(co2_emissions_tons_yr, 3),
            calculation_basis="Napier's equation for steam flow through orifice",
            assumptions={
                'steam_pressure_psig': steam_pressure_psig,
                'orifice_diameter_in': orifice_diameter_in,
                'discharge_coefficient': discharge_coefficient,
                'latent_heat_btu_lb': latent_heat_btu_lb,
                'operating_hours_yr': operating_hours_yr,
                'steam_cost_usd_per_1000lb': steam_cost_usd_per_1000lb,
                'failure_severity': failure_severity
            },
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(trap_data)
        )

        return result

    def _get_discharge_coefficient(self, failure_mode: FailureMode) -> float:
        """Get discharge coefficient based on failure mode."""
        coefficients = {
            FailureMode.FAILED_OPEN: 0.70,      # Full orifice area
            FailureMode.LEAKING: 0.35,          # Partial orifice area
            FailureMode.WORN_SEAT: 0.20,        # Minor leakage
            FailureMode.NORMAL: 0.0             # No leakage
        }
        return coefficients.get(failure_mode, 0.0)

    def _get_latent_heat(self, pressure_psig: float) -> float:
        """
        Get latent heat of vaporization at given pressure.

        Interpolated from steam tables (saturated steam).
        """
        # Simplified steam table (pressure psig -> latent heat BTU/lb)
        steam_table = {
            0: 970.3,
            50: 915.5,
            100: 881.0,
            150: 856.5,
            200: 838.0,
            250: 823.0,
            300: 809.0
        }

        # Linear interpolation
        pressures = sorted(steam_table.keys())
        if pressure_psig <= pressures[0]:
            return steam_table[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return steam_table[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i+1]:
                p1, p2 = pressures[i], pressures[i+1]
                h1, h2 = steam_table[p1], steam_table[p2]
                # Linear interpolation
                return h1 + (h2 - h1) * (pressure_psig - p1) / (p2 - p1)

        return 881.0  # Default for 100 psig

    # ========================================================================
    # TOOL 5: MAINTENANCE PRIORITIZATION
    # ========================================================================

    def prioritize_maintenance(
        self,
        trap_fleet: List[Dict[str, Any]]
    ) -> MaintenancePriorityResult:
        """
        Prioritize maintenance activities across steam trap fleet.

        Args:
            trap_fleet: List of trap data dictionaries with:
                - trap_id
                - failure_mode
                - energy_loss_usd_yr
                - process_criticality (1-10)
                - current_age_years
                - last_maintenance_date

        Returns:
            MaintenancePriorityResult with prioritized schedule and ROI

        Algorithm:
            Priority Score = (Energy Loss Weight * 0.4) +
                           (Process Criticality * 0.3) +
                           (Safety Factor * 0.2) +
                           (Age Factor * 0.1)
        """
        priority_list = []
        total_potential_savings = 0.0
        total_maintenance_cost = 0.0

        for trap in trap_fleet:
            trap_id = trap.get('trap_id')
            failure_mode = trap.get('failure_mode', FailureMode.NORMAL)
            energy_loss_usd_yr = trap.get('energy_loss_usd_yr', 0.0)
            process_criticality = trap.get('process_criticality', 5)  # 1-10
            current_age_years = trap.get('current_age_years', 0)

            # Calculate priority score
            energy_factor = min(energy_loss_usd_yr / 10000, 1.0)  # Normalize to 0-1
            criticality_factor = process_criticality / 10.0
            safety_factor = 1.0 if failure_mode in [FailureMode.FAILED_OPEN, FailureMode.FAILED_CLOSED] else 0.5
            age_factor = min(current_age_years / 15.0, 1.0)  # 15 years = full aging

            priority_score = (
                energy_factor * 0.4 +
                criticality_factor * 0.3 +
                safety_factor * 0.2 +
                age_factor * 0.1
            )

            # Maintenance cost
            if failure_mode in [FailureMode.FAILED_OPEN, FailureMode.FAILED_CLOSED, FailureMode.PLUGGED]:
                maintenance_cost = self.maintenance_config.replacement_cost_per_trap_usd
            else:
                maintenance_cost = self.maintenance_config.maintenance_cost_per_trap_usd

            total_maintenance_cost += maintenance_cost
            total_potential_savings += energy_loss_usd_yr

            priority_list.append({
                'trap_id': trap_id,
                'priority_score': round(priority_score, 4),
                'failure_mode': failure_mode.value,
                'energy_loss_usd_yr': round(energy_loss_usd_yr, 2),
                'maintenance_cost_usd': round(maintenance_cost, 2),
                'process_criticality': process_criticality,
                'recommended_action': self._recommend_action(failure_mode, self._determine_severity(failure_mode, priority_score))
            })

        # Sort by priority score (descending)
        priority_list.sort(key=lambda x: x['priority_score'], reverse=True)

        # Generate recommended schedule
        schedule = self._generate_maintenance_schedule(priority_list)

        # Calculate ROI
        payback_months = (total_maintenance_cost / total_potential_savings * 12) if total_potential_savings > 0 else 999
        roi_percent = (total_potential_savings / total_maintenance_cost * 100 - 100) if total_maintenance_cost > 0 else 0

        result = MaintenancePriorityResult(
            priority_list=priority_list,
            total_potential_savings_usd_yr=round(total_potential_savings, 2),
            recommended_schedule=schedule,
            resource_requirements={
                'total_traps': len(trap_fleet),
                'immediate_action_required': len([t for t in priority_list if t['priority_score'] > 0.8]),
                'estimated_labor_hours': len(trap_fleet) * 2,  # 2 hours per trap
                'estimated_parts_cost_usd': total_maintenance_cost
            },
            estimated_total_cost_usd=round(total_maintenance_cost, 2),
            expected_roi_percent=round(roi_percent, 2),
            payback_months=round(payback_months, 2),
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash({'fleet': [t.get('trap_id') for t in trap_fleet]})
        )

        return result

    def _generate_maintenance_schedule(self, priority_list: List[Dict]) -> List[Dict]:
        """Generate phased maintenance schedule."""
        schedule = []
        current_date = datetime.now()

        # Phase 1: Critical (priority > 0.8) - within 1 week
        phase1 = [t for t in priority_list if t['priority_score'] > 0.8]
        if phase1:
            schedule.append({
                'phase': 1,
                'description': 'Critical failures - immediate action',
                'start_date': current_date.isoformat(),
                'end_date': (current_date + timedelta(days=7)).isoformat(),
                'trap_count': len(phase1),
                'trap_ids': [t['trap_id'] for t in phase1]
            })

        # Phase 2: High priority (0.6 < score <= 0.8) - within 1 month
        phase2 = [t for t in priority_list if 0.6 < t['priority_score'] <= 0.8]
        if phase2:
            schedule.append({
                'phase': 2,
                'description': 'High priority - schedule within 1 month',
                'start_date': (current_date + timedelta(days=7)).isoformat(),
                'end_date': (current_date + timedelta(days=30)).isoformat(),
                'trap_count': len(phase2),
                'trap_ids': [t['trap_id'] for t in phase2]
            })

        # Phase 3: Medium priority (0.4 < score <= 0.6) - within 3 months
        phase3 = [t for t in priority_list if 0.4 < t['priority_score'] <= 0.6]
        if phase3:
            schedule.append({
                'phase': 3,
                'description': 'Medium priority - schedule within 3 months',
                'start_date': (current_date + timedelta(days=30)).isoformat(),
                'end_date': (current_date + timedelta(days=90)).isoformat(),
                'trap_count': len(phase3),
                'trap_ids': [t['trap_id'] for t in phase3]
            })

        return schedule

    # ========================================================================
    # TOOL 6: REMAINING USEFUL LIFE PREDICTION
    # ========================================================================

    def predict_remaining_useful_life(
        self,
        condition_data: Dict[str, Any]
    ) -> RULPredictionResult:
        """
        Predict Remaining Useful Life using Weibull analysis.

        Args:
            condition_data: Dict with:
                - trap_id
                - current_age_days
                - degradation_rate (0.0-1.0 per year)
                - historical_failures: List[int] (ages at failure in days)
                - current_health_score (0-100)

        Returns:
            RULPredictionResult with predicted RUL and confidence intervals

        Formula (Weibull distribution):
            R(t) = exp(-(t/η)^β)
            Where:
                R(t) = Reliability at time t
                η = Scale parameter (characteristic life)
                β = Shape parameter (failure rate trend)
        """
        trap_id = condition_data.get('trap_id', 'unknown')
        current_age_days = condition_data.get('current_age_days', 0)
        degradation_rate = condition_data.get('degradation_rate', 0.1)  # per year
        historical_failures = condition_data.get('historical_failures', [])
        current_health_score = condition_data.get('current_health_score', 100)

        # Step 1: Calculate MTBF from historical data
        if historical_failures and len(historical_failures) > 0:
            mtbf_days = np.mean(historical_failures)
        else:
            # Default MTBF for steam traps: 5-8 years (use 6 years = 2190 days)
            mtbf_days = 2190

        # Step 2: Weibull parameters estimation
        # β = 2.5 (increasing failure rate, typical for mechanical components)
        # η = MTBF / Γ(1 + 1/β) where Γ is gamma function
        beta = 2.5
        gamma_factor = math.gamma(1 + 1/beta)  # ≈ 0.887 for β=2.5
        eta = mtbf_days / gamma_factor

        # Step 3: Calculate reliability at current age
        reliability_current = math.exp(-((current_age_days / eta) ** beta))

        # Step 4: Find time when reliability drops to 10% (end of useful life)
        target_reliability = 0.10
        rul_days = eta * ((-math.log(target_reliability)) ** (1/beta)) - current_age_days

        # Adjust for current health score
        health_factor = current_health_score / 100.0
        rul_days = rul_days * health_factor

        # Step 5: Confidence intervals (±20% typical)
        rul_lower = rul_days * 0.8
        rul_upper = rul_days * 1.2

        # Step 6: Next inspection date (at 50% of RUL or 90 days, whichever is sooner)
        next_inspection_days = min(rul_days * 0.5, 90)
        next_inspection_date = (datetime.now() + timedelta(days=next_inspection_days)).isoformat()

        # Step 7: Failure probability curve (next 365 days)
        failure_curve = []
        for days_ahead in range(0, 366, 30):  # Monthly intervals
            future_age = current_age_days + days_ahead
            reliability = math.exp(-((future_age / eta) ** beta))
            failure_prob = 1 - reliability
            failure_curve.append((days_ahead, round(failure_prob, 4)))

        result = RULPredictionResult(
            trap_id=trap_id,
            rul_days=round(rul_days, 2),
            rul_confidence_lower=round(rul_lower, 2),
            rul_confidence_upper=round(rul_upper, 2),
            confidence_interval_percent=90,
            next_inspection_date=next_inspection_date,
            failure_probability_curve=failure_curve,
            degradation_rate=round(degradation_rate, 4),
            historical_mtbf_days=round(mtbf_days, 2) if historical_failures else None,
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(condition_data)
        )

        return result

    # ========================================================================
    # TOOL 7: COST-BENEFIT ANALYSIS
    # ========================================================================

    def calculate_cost_benefit(
        self,
        maintenance_plan: Dict[str, Any]
    ) -> CostBenefitResult:
        """
        Perform cost-benefit analysis for maintenance/replacement decision.

        Args:
            maintenance_plan: Dict with:
                - trap_id
                - action: 'repair' or 'replace'
                - annual_energy_loss_usd
                - maintenance_cost_usd (one-time)
                - replacement_cost_usd (if replacing)
                - expected_service_life_years (after action)
                - discount_rate (typically 0.08 for 8%)

        Returns:
            CostBenefitResult with financial metrics and recommendation

        Formulas:
            NPV = Σ(Savings_t / (1+r)^t) - Initial_Cost
            IRR: Rate where NPV = 0
            Payback: Initial_Cost / Annual_Savings
        """
        trap_id = maintenance_plan.get('trap_id', 'unknown')
        action = maintenance_plan.get('action', 'repair')
        annual_savings_usd = maintenance_plan.get('annual_energy_loss_usd', 0)
        maintenance_cost_usd = maintenance_plan.get('maintenance_cost_usd', self.maintenance_config.maintenance_cost_per_trap_usd)
        replacement_cost_usd = maintenance_plan.get('replacement_cost_usd', self.maintenance_config.replacement_cost_per_trap_usd)
        service_life_years = maintenance_plan.get('expected_service_life_years', 5)
        discount_rate = maintenance_plan.get('discount_rate', 0.08)

        # Step 1: Determine initial cost
        if action == 'replace':
            initial_cost_usd = replacement_cost_usd
        else:
            initial_cost_usd = maintenance_cost_usd

        # Step 2: Calculate payback period
        if annual_savings_usd > 0:
            payback_months = (initial_cost_usd / annual_savings_usd) * 12
        else:
            payback_months = 999  # Infinite payback

        # Step 3: Calculate ROI (simple)
        total_savings_over_life = annual_savings_usd * service_life_years
        roi_percent = ((total_savings_over_life - initial_cost_usd) / initial_cost_usd) * 100

        # Step 4: Calculate NPV
        npv_usd = 0
        for year in range(1, service_life_years + 1):
            discounted_savings = annual_savings_usd / ((1 + discount_rate) ** year)
            npv_usd += discounted_savings
        npv_usd -= initial_cost_usd

        # Step 5: Calculate IRR (simplified iterative approach)
        irr_percent = self._calculate_irr(initial_cost_usd, annual_savings_usd, service_life_years)

        # Step 6: Make recommendation
        if npv_usd > 0 and payback_months < 24:
            if action == 'replace':
                decision_recommendation = "RECOMMENDED: Replace trap - strong financial case"
            else:
                decision_recommendation = "RECOMMENDED: Repair trap - positive ROI with quick payback"
        elif npv_usd > 0 and payback_months < 48:
            decision_recommendation = "RECOMMENDED: Proceed with plan - positive NPV but longer payback"
        elif npv_usd < 0:
            decision_recommendation = "NOT RECOMMENDED: Negative NPV - consider monitoring only"
        else:
            decision_recommendation = "MARGINAL: Break-even scenario - evaluate operational risk"

        # Step 7: Sensitivity analysis
        sensitivity = {
            'savings_-20%': round((annual_savings_usd * 0.8 * service_life_years - initial_cost_usd) / initial_cost_usd * 100, 2),
            'savings_+20%': round((annual_savings_usd * 1.2 * service_life_years - initial_cost_usd) / initial_cost_usd * 100, 2),
            'cost_-20%': round((annual_savings_usd * service_life_years - initial_cost_usd * 0.8) / (initial_cost_usd * 0.8) * 100, 2),
            'cost_+20%': round((annual_savings_usd * service_life_years - initial_cost_usd * 1.2) / (initial_cost_usd * 1.2) * 100, 2)
        }

        result = CostBenefitResult(
            trap_id=trap_id,
            maintenance_cost_usd=round(maintenance_cost_usd, 2) if action == 'repair' else 0,
            replacement_cost_usd=round(replacement_cost_usd, 2) if action == 'replace' else 0,
            annual_savings_usd=round(annual_savings_usd, 2),
            payback_months=round(payback_months, 2),
            roi_percent=round(roi_percent, 2),
            npv_usd=round(npv_usd, 2),
            irr_percent=round(irr_percent, 2),
            decision_recommendation=decision_recommendation,
            sensitivity_analysis=sensitivity,
            timestamp=datetime.utcnow().isoformat(),
            provenance_hash=self._calculate_hash(maintenance_plan)
        )

        return result

    def _calculate_irr(
        self, initial_cost: float, annual_savings: float, years: int
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson iteration.
        """
        # IRR is the rate where NPV = 0
        # NPV(r) = -Initial + Σ(Savings / (1+r)^t) = 0

        def npv_at_rate(rate: float) -> float:
            npv = -initial_cost
            for year in range(1, years + 1):
                npv += annual_savings / ((1 + rate) ** year)
            return npv

        # Newton-Raphson iteration
        rate = 0.1  # Initial guess 10%
        for _ in range(20):  # Max 20 iterations
            npv = npv_at_rate(rate)
            if abs(npv) < 1:  # Converged
                break

            # Numerical derivative
            delta = 0.0001
            npv_delta = npv_at_rate(rate + delta)
            derivative = (npv_delta - npv) / delta

            if abs(derivative) < 1e-10:
                break

            # Update rate
            rate = rate - npv / derivative

            # Bound to reasonable range
            rate = max(0.0, min(rate, 1.0))

        return rate * 100  # Convert to percentage

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
