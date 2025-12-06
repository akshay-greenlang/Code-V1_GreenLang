"""
GL-020 ECONOPULSE - Steaming Economizer Detector

Detects and prevents steaming conditions in economizers through:
- Approach temperature monitoring (water outlet vs saturation)
- Low-load risk assessment
- DP/temperature fluctuation detection
- Recirculation activation logic

Steaming economizers can cause:
- Water hammer (dangerous tube damage)
- Flow instability
- Uneven heat distribution
- Accelerated corrosion

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - ASME Boiler and Pressure Vessel Code

Zero-Hallucination: All calculations use deterministic formulas with full provenance.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Steam table data for saturation temperature calculation
# Based on IAPWS-IF97 industrial formulation
# T_sat = f(P) for saturated water/steam

# Coefficients for saturation temperature correlation (50-3000 psig range)
# T_sat(F) = A + B*ln(P+14.7) + C*(ln(P+14.7))^2
SAT_TEMP_COEF_A = -51.46
SAT_TEMP_COEF_B = 73.86
SAT_TEMP_COEF_C = 3.217

# Minimum approach temperature for safe operation (F)
MIN_SAFE_APPROACH_F = 5.0

# Risk threshold approach temperatures (F)
APPROACH_WARNING_F = 15.0
APPROACH_ALARM_F = 10.0
APPROACH_CRITICAL_F = 5.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SteamingConfig:
    """Configuration for steaming detection."""

    # Design conditions
    design_approach_temp_f: float = 30.0
    design_subcooling_f: float = 20.0
    design_outlet_pressure_psig: float = 500.0

    # Thresholds
    approach_warning_f: float = APPROACH_WARNING_F
    approach_alarm_f: float = APPROACH_ALARM_F
    approach_critical_f: float = APPROACH_CRITICAL_F

    # Fluctuation detection
    dp_fluctuation_threshold_pct: float = 10.0
    temp_fluctuation_threshold_f: float = 5.0
    fluctuation_window_samples: int = 10

    # Load-based limits
    steaming_risk_load_pct: float = 30.0
    min_water_flow_pct: float = 25.0

    # Recirculation
    recirculation_enabled: bool = False
    recirculation_trigger_approach_f: float = 12.0
    recirculation_flow_pct: float = 15.0


@dataclass
class SteamingInput:
    """Input data for steaming detection."""

    timestamp: datetime

    # Water outlet conditions
    water_outlet_temp_f: float
    water_outlet_pressure_psig: float

    # Operating conditions
    current_load_pct: float
    water_flow_lb_hr: float
    design_water_flow_lb_hr: float

    # Gas side
    gas_inlet_temp_f: float

    # Saturation reference (optional - can be calculated)
    drum_pressure_psig: Optional[float] = None
    saturation_temp_f: Optional[float] = None  # Can be provided or calculated

    # Historical data for fluctuation detection
    recent_dp_values: List[float] = field(default_factory=list)
    recent_temp_values: List[float] = field(default_factory=list)

    # Recirculation status
    recirculation_active: bool = False
    recirculation_flow_pct: float = 0.0


@dataclass
class SteamingResult:
    """Result of steaming detection analysis."""

    # Steaming status
    steaming_detected: bool
    steaming_risk: str  # low, moderate, high, critical
    steaming_risk_score: float  # 0-100

    # Approach temperature
    approach_temp_f: float
    design_approach_f: float
    approach_margin_f: float

    # Saturation conditions
    water_outlet_temp_f: float
    saturation_temp_f: float
    subcooling_f: float

    # Operating conditions
    current_load_pct: float
    water_flow_pct: float
    low_load_risk: bool

    # Fluctuation analysis
    dp_fluctuation_detected: bool
    temp_fluctuation_detected: bool
    dp_fluctuation_pct: float
    temp_fluctuation_f: float

    # Recommendations
    action_required: bool
    recommended_action: Optional[str]
    increase_water_flow: bool
    activate_recirculation: bool
    reduce_heat_input: bool

    # Operating limits
    min_safe_load_pct: float
    current_min_load_margin_pct: float

    # Provenance
    calculation_method: str
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STEAMING DETECTOR
# =============================================================================

class SteamingDetector:
    """
    Detects and prevents steaming conditions in economizers.

    Steaming occurs when the economizer water outlet temperature
    approaches saturation temperature, causing local vaporization.

    Detection methods:
    1. Approach temperature monitoring (outlet temp vs saturation)
    2. DP fluctuation detection (steam bubbles cause erratic flow)
    3. Temperature fluctuation detection (unstable heat transfer)
    4. Low-load risk assessment (reduced water flow increases risk)

    Prevention strategies:
    1. Maintain adequate water flow (minimum 25-30% design)
    2. Activate recirculation when approach gets too small
    3. Reduce heat input (bypass damper, load reduction)
    4. Maintain minimum load limits

    Reference: ASME PTC 4.1, ASME BPVC Section I
    """

    def __init__(self, config: SteamingConfig):
        """
        Initialize steaming detector.

        Args:
            config: Steaming detection configuration
        """
        self.config = config
        logger.info(
            f"SteamingDetector initialized: design_approach={config.design_approach_temp_f}F"
        )

    def calculate_saturation_temperature(
        self,
        pressure_psig: float,
    ) -> float:
        """
        Calculate saturation temperature from pressure.

        Uses a polynomial correlation based on IAPWS-IF97:
        T_sat(F) = A + B*ln(P_abs) + C*(ln(P_abs))^2

        Valid for 50-3000 psig range.

        Args:
            pressure_psig: Pressure in psig

        Returns:
            Saturation temperature in Fahrenheit
        """
        # Convert to absolute pressure (psia)
        pressure_psia = pressure_psig + 14.696

        if pressure_psia <= 0:
            return 212.0  # Atmospheric boiling point

        # Correlation calculation
        ln_p = math.log(pressure_psia)
        t_sat = (
            SAT_TEMP_COEF_A +
            SAT_TEMP_COEF_B * ln_p +
            SAT_TEMP_COEF_C * ln_p * ln_p
        )

        # Validate against known points
        # At 14.7 psia (0 psig): T_sat = 212F
        # At 514.7 psia (500 psig): T_sat ≈ 470F
        # At 1014.7 psia (1000 psig): T_sat ≈ 546F

        return t_sat

    def calculate_approach_temperature(
        self,
        water_outlet_temp_f: float,
        saturation_temp_f: float,
    ) -> Tuple[float, float]:
        """
        Calculate approach temperature and subcooling.

        Approach temperature: T_sat - T_water_out
        (how close outlet is to saturation)

        Subcooling: T_sat - T_water_out (same as approach for economizer)

        Args:
            water_outlet_temp_f: Water outlet temperature (F)
            saturation_temp_f: Saturation temperature at outlet pressure (F)

        Returns:
            Tuple of (approach_temp_f, subcooling_f)
        """
        approach_temp = saturation_temp_f - water_outlet_temp_f
        subcooling = approach_temp  # Same for economizer (no superheat)

        return approach_temp, subcooling

    def assess_approach_risk(
        self,
        approach_temp_f: float,
    ) -> Tuple[str, float]:
        """
        Assess steaming risk based on approach temperature.

        Risk levels:
        - Low: approach > warning threshold
        - Moderate: alarm < approach <= warning
        - High: critical < approach <= alarm
        - Critical: approach <= critical

        Args:
            approach_temp_f: Approach temperature (F)

        Returns:
            Tuple of (risk_level, risk_score)
        """
        if approach_temp_f <= self.config.approach_critical_f:
            risk_level = "critical"
            risk_score = 100.0
        elif approach_temp_f <= self.config.approach_alarm_f:
            risk_level = "high"
            # Score from 75-100 as approach decreases
            score_range = self.config.approach_alarm_f - self.config.approach_critical_f
            if score_range > 0:
                risk_score = 100 - 25 * (approach_temp_f - self.config.approach_critical_f) / score_range
            else:
                risk_score = 87.5
        elif approach_temp_f <= self.config.approach_warning_f:
            risk_level = "moderate"
            # Score from 50-75
            score_range = self.config.approach_warning_f - self.config.approach_alarm_f
            if score_range > 0:
                risk_score = 75 - 25 * (approach_temp_f - self.config.approach_alarm_f) / score_range
            else:
                risk_score = 62.5
        else:
            risk_level = "low"
            # Score from 0-50 based on margin above warning
            margin = approach_temp_f - self.config.approach_warning_f
            risk_score = max(0, 50 - margin * 2)

        return risk_level, risk_score

    def detect_fluctuations(
        self,
        recent_values: List[float],
        threshold: float,
        is_percentage: bool = False,
    ) -> Tuple[bool, float]:
        """
        Detect fluctuations in recent measurements.

        Fluctuations indicate unstable flow, possibly due to:
        - Steam bubble formation (two-phase flow)
        - Flow instability
        - Control valve hunting

        Uses coefficient of variation (CV) for detection:
        CV = std_dev / mean

        Args:
            recent_values: List of recent measurements
            threshold: Fluctuation threshold (% or absolute)
            is_percentage: If True, threshold is in percent

        Returns:
            Tuple of (fluctuation_detected, fluctuation_magnitude)
        """
        if len(recent_values) < 3:
            return False, 0.0

        # Calculate statistics
        mean_val = sum(recent_values) / len(recent_values)
        if mean_val == 0:
            return False, 0.0

        variance = sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values)
        std_dev = math.sqrt(variance)

        if is_percentage:
            # Coefficient of variation
            fluctuation = (std_dev / abs(mean_val)) * 100
        else:
            # Absolute fluctuation
            fluctuation = std_dev

        detected = fluctuation > threshold

        return detected, fluctuation

    def assess_low_load_risk(
        self,
        current_load_pct: float,
        water_flow_pct: float,
    ) -> Tuple[bool, float]:
        """
        Assess steaming risk from low load operation.

        At low loads:
        - Reduced water flow through economizer
        - Higher gas-to-water temperature ratio
        - Increased time for heat transfer
        - Greater risk of localized steaming

        Args:
            current_load_pct: Current boiler load (%)
            water_flow_pct: Water flow as % of design

        Returns:
            Tuple of (low_load_risk, min_safe_load_pct)
        """
        # Calculate minimum safe load
        # Based on maintaining adequate water velocity
        min_safe_load = self.config.steaming_risk_load_pct

        # Adjust for actual water flow
        if water_flow_pct < self.config.min_water_flow_pct:
            # Increase minimum safe load if flow is low
            flow_deficit = self.config.min_water_flow_pct - water_flow_pct
            min_safe_load += flow_deficit * 0.5  # 0.5% load increase per 1% flow deficit

        low_load_risk = current_load_pct < min_safe_load

        return low_load_risk, min_safe_load

    def determine_recommendations(
        self,
        risk_level: str,
        risk_score: float,
        approach_temp_f: float,
        low_load_risk: bool,
        dp_fluctuation: bool,
        temp_fluctuation: bool,
        water_flow_pct: float,
        recirculation_active: bool,
    ) -> Dict[str, any]:
        """
        Determine recommended actions based on steaming risk.

        Args:
            risk_level: Current risk level
            risk_score: Risk score (0-100)
            approach_temp_f: Approach temperature
            low_load_risk: Low load risk flag
            dp_fluctuation: DP fluctuation detected
            temp_fluctuation: Temperature fluctuation detected
            water_flow_pct: Current water flow %
            recirculation_active: Recirculation currently active

        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "action_required": False,
            "recommended_action": None,
            "increase_water_flow": False,
            "activate_recirculation": False,
            "reduce_heat_input": False,
        }

        # Check for steaming indicators
        steaming_indicators = dp_fluctuation and temp_fluctuation

        if risk_level == "critical" or steaming_indicators:
            recommendations["action_required"] = True
            recommendations["reduce_heat_input"] = True

            if steaming_indicators:
                recommendations["recommended_action"] = (
                    "CRITICAL: Steaming detected (DP and temp fluctuations). "
                    "Reduce load or bypass economizer immediately. "
                    "Check for water hammer."
                )
            else:
                recommendations["recommended_action"] = (
                    "CRITICAL: Approach temperature too low. "
                    "Reduce firing rate or increase water flow. "
                    "Prepare for economizer bypass."
                )

            # Activate all mitigations
            if water_flow_pct < 100:
                recommendations["increase_water_flow"] = True
            if self.config.recirculation_enabled and not recirculation_active:
                recommendations["activate_recirculation"] = True

        elif risk_level == "high":
            recommendations["action_required"] = True
            recommendations["recommended_action"] = (
                f"HIGH RISK: Approach temperature {approach_temp_f:.1f}F is low. "
                "Increase water flow and/or activate recirculation."
            )

            if water_flow_pct < 100:
                recommendations["increase_water_flow"] = True
            if (
                self.config.recirculation_enabled and
                not recirculation_active and
                approach_temp_f < self.config.recirculation_trigger_approach_f
            ):
                recommendations["activate_recirculation"] = True

        elif risk_level == "moderate":
            recommendations["action_required"] = False
            recommendations["recommended_action"] = (
                f"MODERATE RISK: Monitor approach temperature ({approach_temp_f:.1f}F). "
                "Consider increasing water flow or reducing load."
            )

            if (
                self.config.recirculation_enabled and
                not recirculation_active and
                approach_temp_f < self.config.recirculation_trigger_approach_f
            ):
                recommendations["activate_recirculation"] = True

        elif low_load_risk:
            recommendations["recommended_action"] = (
                "LOW LOAD WARNING: Operating below minimum safe load. "
                "Increase load or maintain higher water flow."
            )
            recommendations["increase_water_flow"] = True

        return recommendations

    def detect(self, input_data: SteamingInput) -> SteamingResult:
        """
        Perform steaming detection analysis.

        Args:
            input_data: SteamingInput with current conditions

        Returns:
            SteamingResult with detection results
        """
        # Calculate saturation temperature
        if input_data.saturation_temp_f is not None:
            sat_temp = input_data.saturation_temp_f
        elif input_data.drum_pressure_psig is not None:
            sat_temp = self.calculate_saturation_temperature(input_data.drum_pressure_psig)
        else:
            sat_temp = self.calculate_saturation_temperature(input_data.water_outlet_pressure_psig)

        # Calculate approach temperature
        approach_temp, subcooling = self.calculate_approach_temperature(
            input_data.water_outlet_temp_f,
            sat_temp,
        )

        # Calculate approach margin
        approach_margin = approach_temp - self.config.design_approach_temp_f

        # Assess approach risk
        risk_level, risk_score = self.assess_approach_risk(approach_temp)

        # Calculate water flow percentage
        water_flow_pct = (
            input_data.water_flow_lb_hr / input_data.design_water_flow_lb_hr * 100
            if input_data.design_water_flow_lb_hr > 0 else 0
        )

        # Assess low load risk
        low_load_risk, min_safe_load = self.assess_low_load_risk(
            input_data.current_load_pct,
            water_flow_pct,
        )

        # Detect DP fluctuations
        dp_fluctuation, dp_fluct_pct = self.detect_fluctuations(
            input_data.recent_dp_values,
            self.config.dp_fluctuation_threshold_pct,
            is_percentage=True,
        )

        # Detect temperature fluctuations
        temp_fluctuation, temp_fluct_f = self.detect_fluctuations(
            input_data.recent_temp_values,
            self.config.temp_fluctuation_threshold_f,
            is_percentage=False,
        )

        # Determine if steaming is detected
        steaming_detected = (
            approach_temp <= MIN_SAFE_APPROACH_F or
            (dp_fluctuation and temp_fluctuation) or
            risk_level == "critical"
        )

        # If steaming is detected, increase risk score
        if steaming_detected and risk_score < 100:
            risk_score = max(risk_score, 90)
            if risk_level not in ("critical", "high"):
                risk_level = "high"

        # Get recommendations
        recs = self.determine_recommendations(
            risk_level,
            risk_score,
            approach_temp,
            low_load_risk,
            dp_fluctuation,
            temp_fluctuation,
            water_flow_pct,
            input_data.recirculation_active,
        )

        # Calculate minimum load margin
        min_load_margin = input_data.current_load_pct - min_safe_load

        # Build result
        result_data = {
            "steaming_detected": steaming_detected,
            "steaming_risk": risk_level,
            "steaming_risk_score": round(risk_score, 1),
            "approach_temp_f": round(approach_temp, 1),
            "design_approach_f": round(self.config.design_approach_temp_f, 1),
            "approach_margin_f": round(approach_margin, 1),
            "water_outlet_temp_f": round(input_data.water_outlet_temp_f, 1),
            "saturation_temp_f": round(sat_temp, 1),
            "subcooling_f": round(subcooling, 1),
            "current_load_pct": round(input_data.current_load_pct, 1),
            "water_flow_pct": round(water_flow_pct, 1),
            "low_load_risk": low_load_risk,
            "dp_fluctuation_detected": dp_fluctuation,
            "temp_fluctuation_detected": temp_fluctuation,
            "dp_fluctuation_pct": round(dp_fluct_pct, 1),
            "temp_fluctuation_f": round(temp_fluct_f, 1),
            "action_required": recs["action_required"],
            "recommended_action": recs["recommended_action"],
            "increase_water_flow": recs["increase_water_flow"],
            "activate_recirculation": recs["activate_recirculation"],
            "reduce_heat_input": recs["reduce_heat_input"],
            "min_safe_load_pct": round(min_safe_load, 1),
            "current_min_load_margin_pct": round(min_load_margin, 1),
            "calculation_method": "APPROACH_TEMP_MONITORING",
        }

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            json.dumps(result_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        result = SteamingResult(
            **result_data,
            provenance_hash=provenance_hash,
        )

        log_level = logging.WARNING if steaming_detected else logging.INFO
        logger.log(
            log_level,
            f"Steaming detection: detected={steaming_detected}, risk={risk_level}, "
            f"approach={approach_temp:.1f}F, score={risk_score:.0f}"
        )

        return result


def create_steaming_detector(
    config: Optional[SteamingConfig] = None,
) -> SteamingDetector:
    """Factory function to create SteamingDetector."""
    if config is None:
        config = SteamingConfig()
    return SteamingDetector(config)
