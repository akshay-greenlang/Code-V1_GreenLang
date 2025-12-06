"""
GL-020 ECONOPULSE - Gas-Side Fouling Analyzer

Analyzes gas-side fouling through pressure drop and heat transfer degradation.
Differentiates between gas-side and water-side fouling sources.

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: All calculations use deterministic formulas with full provenance.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default thermal conductivity of ash/soot deposits (BTU/hr-ft-F)
# Typical range: 0.05-0.15 for fly ash
ASH_THERMAL_CONDUCTIVITY = 0.1

# Reference conditions
REFERENCE_TEMP_F = 60.0
REFERENCE_PRESSURE_ATM = 1.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GasSideFoulingInput:
    """Input data for gas-side fouling analysis."""

    # Current pressure drop
    current_dp_in_wc: float
    design_dp_in_wc: float

    # Flow conditions
    current_gas_flow_lb_hr: float
    design_gas_flow_lb_hr: float

    # Temperatures
    gas_inlet_temp_f: float
    gas_outlet_temp_f: float

    # Heat transfer values
    current_u_btu_hr_ft2_f: Optional[float] = None
    clean_u_btu_hr_ft2_f: Optional[float] = None

    # Design values
    design_u_btu_hr_ft2_f: float = 10.0
    heat_transfer_area_ft2: float = 5000.0

    # Historical data for trending
    dp_history: List[Tuple[datetime, float]] = field(default_factory=list)
    u_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # Thresholds
    dp_warning_ratio: float = 1.3
    dp_alarm_ratio: float = 1.5
    dp_cleaning_ratio: float = 1.7
    u_degradation_warning_pct: float = 10.0
    u_degradation_alarm_pct: float = 20.0


@dataclass
class GasSideFoulingResult:
    """Result of gas-side fouling analysis."""

    # Fouling detection
    fouling_detected: bool
    fouling_severity: str  # none, light, moderate, severe, critical
    fouling_trend: str  # improving, stable, degrading

    # Pressure drop analysis
    current_dp_in_wc: float
    design_dp_in_wc: float
    corrected_dp_in_wc: float
    dp_ratio: float
    dp_deviation_pct: float

    # Heat transfer degradation
    u_actual_btu_hr_ft2_f: float
    u_clean_btu_hr_ft2_f: float
    u_degradation_pct: float

    # Fouling resistance
    fouling_resistance_hr_ft2_f_btu: float
    estimated_fouling_thickness_in: Optional[float]

    # Performance impact
    efficiency_loss_pct: float
    fuel_waste_pct: float

    # Recommendations
    cleaning_status: str  # not_required, monitor, recommended, required, urgent
    soot_blow_recommended: bool
    estimated_hours_to_cleaning: Optional[float]

    # Provenance
    calculation_method: str
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# GAS-SIDE FOULING ANALYZER
# =============================================================================

class GasSideFoulingAnalyzer:
    """
    Analyzes gas-side fouling through pressure drop and heat transfer metrics.

    Gas-side fouling in economizers is primarily caused by:
    - Fly ash deposition
    - Soot accumulation
    - Sulfate deposits (low-temperature corrosion products)

    Detection methods:
    1. Pressure drop increase (corrected for flow)
    2. Heat transfer coefficient degradation
    3. Gas outlet temperature rise

    This analyzer provides:
    - Flow-corrected pressure drop analysis
    - Fouling resistance calculation
    - Fouling source differentiation (gas vs water side)
    - Trend analysis for predictive maintenance

    Reference: ASME PTC 4.3 Air Heater Test Code
    """

    def __init__(
        self,
        ash_thermal_conductivity: float = ASH_THERMAL_CONDUCTIVITY,
    ):
        """
        Initialize gas-side fouling analyzer.

        Args:
            ash_thermal_conductivity: Thermal conductivity of ash deposits (BTU/hr-ft-F)
        """
        self.ash_thermal_conductivity = ash_thermal_conductivity
        logger.info("GasSideFoulingAnalyzer initialized")

    def analyze_pressure_drop(
        self,
        current_dp: float,
        design_dp: float,
        flow_ratio: float,
    ) -> Tuple[float, float, float]:
        """
        Analyze gas-side fouling from pressure drop.

        Pressure drop in tube banks follows:
        DP = f * (rho * V^2 / 2) * (L/D) * N

        For turbulent flow, DP is proportional to flow^2:
        DP_corrected = DP_actual / (flow_ratio)^2

        Fouling increases DP by:
        1. Reducing flow area (higher velocity)
        2. Increasing surface roughness (higher friction factor)

        Args:
            current_dp: Current measured pressure drop (in. WC)
            design_dp: Design pressure drop at design flow (in. WC)
            flow_ratio: Current flow / design flow

        Returns:
            Tuple of (corrected_dp, dp_ratio, dp_deviation_pct)
        """
        # Ensure valid flow ratio
        if flow_ratio <= 0:
            flow_ratio = 0.01

        # Correct DP to design flow conditions
        # DP is proportional to velocity squared (turbulent flow)
        # DP_corrected = DP_actual / (V_actual/V_design)^2
        corrected_dp = current_dp / (flow_ratio ** 2)

        # Calculate fouling ratio
        if design_dp <= 0:
            design_dp = 0.1  # Prevent division by zero

        dp_ratio = corrected_dp / design_dp
        dp_deviation_pct = (dp_ratio - 1.0) * 100

        logger.debug(
            f"DP analysis: actual={current_dp:.2f}, corrected={corrected_dp:.2f}, "
            f"ratio={dp_ratio:.3f}, deviation={dp_deviation_pct:.1f}%"
        )

        return corrected_dp, dp_ratio, dp_deviation_pct

    def analyze_heat_transfer_degradation(
        self,
        current_u: float,
        clean_u: float,
    ) -> Tuple[float, float]:
        """
        Calculate U-value degradation and fouling resistance.

        The overall heat transfer coefficient (U) decreases with fouling:
        1/U_actual = 1/U_clean + R_fouling

        Therefore:
        R_fouling = 1/U_actual - 1/U_clean

        Args:
            current_u: Actual overall U-value (BTU/hr-ft2-F)
            clean_u: Clean condition U-value (BTU/hr-ft2-F)

        Returns:
            Tuple of (fouling_resistance, u_degradation_pct)
        """
        # Ensure valid values
        if current_u <= 0:
            current_u = 0.01
        if clean_u <= 0:
            clean_u = current_u * 1.2  # Assume 20% higher if not provided

        # Calculate fouling resistance
        # R_fouling = 1/U_actual - 1/U_clean
        r_fouling = (1.0 / current_u) - (1.0 / clean_u)

        # Ensure non-negative (can be negative if current > clean due to measurement error)
        r_fouling = max(0.0, r_fouling)

        # Calculate degradation percentage
        u_degradation_pct = ((clean_u - current_u) / clean_u) * 100
        u_degradation_pct = max(0.0, u_degradation_pct)

        logger.debug(
            f"Heat transfer analysis: U_actual={current_u:.2f}, U_clean={clean_u:.2f}, "
            f"R_fouling={r_fouling:.6f}, degradation={u_degradation_pct:.1f}%"
        )

        return r_fouling, u_degradation_pct

    def estimate_fouling_thickness(
        self,
        fouling_resistance: float,
        thermal_conductivity: Optional[float] = None,
    ) -> float:
        """
        Estimate fouling layer thickness from fouling resistance.

        For a planar deposit layer:
        R_fouling = thickness / k_fouling

        Therefore:
        thickness = R_fouling * k_fouling

        Args:
            fouling_resistance: Fouling resistance (hr-ft2-F/BTU)
            thermal_conductivity: Thermal conductivity of deposit (BTU/hr-ft-F)

        Returns:
            Estimated fouling thickness in inches
        """
        if thermal_conductivity is None:
            thermal_conductivity = self.ash_thermal_conductivity

        # thickness (ft) = R * k
        thickness_ft = fouling_resistance * thermal_conductivity
        thickness_in = thickness_ft * 12.0

        return thickness_in

    def differentiate_fouling_source(
        self,
        dp_trend: List[Tuple[datetime, float]],
        u_trend: List[Tuple[datetime, float]],
        water_dp_trend: Optional[List[Tuple[datetime, float]]] = None,
    ) -> Tuple[str, float, str]:
        """
        Differentiate between gas-side and water-side fouling.

        Differentiation logic:
        - Gas-side fouling: DP increases, U decreases
        - Water-side fouling: DP stable (gas), U decreases, water DP may increase
        - Combined fouling: Both DP and U degrade together

        The analysis examines the correlation between trends:
        - Strong DP-U correlation with increasing DP -> gas-side
        - U decreasing without DP increase -> water-side

        Args:
            dp_trend: Historical gas-side pressure drop data
            u_trend: Historical U-value data
            water_dp_trend: Optional water-side pressure drop data

        Returns:
            Tuple of (fouling_type, confidence, explanation)
        """
        # Need minimum data points for trend analysis
        min_points = 3

        if len(dp_trend) < min_points or len(u_trend) < min_points:
            return ("unknown", 0.0, "Insufficient data for trend analysis")

        # Calculate trends (simple linear regression slope)
        dp_slope = self._calculate_trend_slope(dp_trend)
        u_slope = self._calculate_trend_slope(u_trend)

        # Normalize slopes to percentage per day
        # Assume timestamps are in reasonable order
        dp_trend_pct_per_day = dp_slope * 24 * 100  # Convert to %/day
        u_trend_pct_per_day = u_slope * 24 * 100

        # Thresholds for significant trends
        dp_threshold = 0.5  # %/day
        u_threshold = 0.3  # %/day

        dp_increasing = dp_trend_pct_per_day > dp_threshold
        u_decreasing = u_trend_pct_per_day < -u_threshold

        # Differentiation logic
        if dp_increasing and u_decreasing:
            # Classic gas-side fouling pattern
            fouling_type = "gas_side"
            confidence = min(1.0, abs(dp_trend_pct_per_day) / 2.0)
            explanation = (
                f"Gas-side fouling indicated: DP increasing at {dp_trend_pct_per_day:.2f}%/day, "
                f"U decreasing at {abs(u_trend_pct_per_day):.2f}%/day"
            )

        elif u_decreasing and not dp_increasing:
            # U decreasing without DP increase suggests water-side
            if water_dp_trend and len(water_dp_trend) >= min_points:
                water_dp_slope = self._calculate_trend_slope(water_dp_trend)
                water_dp_pct_per_day = water_dp_slope * 24 * 100

                if water_dp_pct_per_day > dp_threshold:
                    fouling_type = "water_side"
                    confidence = min(1.0, abs(u_trend_pct_per_day) / 1.5)
                    explanation = (
                        f"Water-side fouling indicated: gas DP stable, U decreasing at "
                        f"{abs(u_trend_pct_per_day):.2f}%/day, water DP increasing"
                    )
                else:
                    fouling_type = "water_side"
                    confidence = 0.6
                    explanation = (
                        f"Likely water-side fouling: gas DP stable, U decreasing at "
                        f"{abs(u_trend_pct_per_day):.2f}%/day"
                    )
            else:
                fouling_type = "water_side"
                confidence = 0.5
                explanation = (
                    f"Probable water-side fouling: gas DP stable, U decreasing at "
                    f"{abs(u_trend_pct_per_day):.2f}%/day (water DP data not available)"
                )

        elif dp_increasing and not u_decreasing:
            # DP increasing without U change could be partial blockage
            fouling_type = "gas_side"
            confidence = 0.7
            explanation = (
                f"Possible gas-side restriction: DP increasing at {dp_trend_pct_per_day:.2f}%/day "
                f"without significant U degradation"
            )

        else:
            # No significant trends
            fouling_type = "none"
            confidence = 0.8
            explanation = "No significant fouling trends detected"

        return fouling_type, confidence, explanation

    def _calculate_trend_slope(
        self,
        data: List[Tuple[datetime, float]],
    ) -> float:
        """
        Calculate trend slope using simple linear regression.

        Returns slope in units per hour.
        """
        if len(data) < 2:
            return 0.0

        # Convert to hours from first point
        t0 = data[0][0]
        x = []
        y = []

        for timestamp, value in data:
            hours = (timestamp - t0).total_seconds() / 3600.0
            x.append(hours)
            y.append(value)

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Normalize to percentage change per hour relative to mean
        mean_y = sum_y / n if n > 0 else 1.0
        if abs(mean_y) > 1e-10:
            slope = slope / mean_y

        return slope

    def determine_fouling_severity(
        self,
        dp_ratio: float,
        u_degradation_pct: float,
        config: Optional[Dict] = None,
    ) -> str:
        """
        Determine fouling severity based on indicators.

        Args:
            dp_ratio: Corrected DP ratio (actual/design)
            u_degradation_pct: U-value degradation percentage
            config: Optional thresholds configuration

        Returns:
            Severity level: none, light, moderate, severe, critical
        """
        # Default thresholds
        thresholds = {
            "dp_light": 1.1,
            "dp_moderate": 1.3,
            "dp_severe": 1.5,
            "dp_critical": 1.8,
            "u_light": 5.0,
            "u_moderate": 10.0,
            "u_severe": 20.0,
            "u_critical": 30.0,
        }

        if config:
            thresholds.update(config)

        # Determine severity from DP
        if dp_ratio >= thresholds["dp_critical"]:
            dp_severity = 4  # critical
        elif dp_ratio >= thresholds["dp_severe"]:
            dp_severity = 3  # severe
        elif dp_ratio >= thresholds["dp_moderate"]:
            dp_severity = 2  # moderate
        elif dp_ratio >= thresholds["dp_light"]:
            dp_severity = 1  # light
        else:
            dp_severity = 0  # none

        # Determine severity from U degradation
        if u_degradation_pct >= thresholds["u_critical"]:
            u_severity = 4
        elif u_degradation_pct >= thresholds["u_severe"]:
            u_severity = 3
        elif u_degradation_pct >= thresholds["u_moderate"]:
            u_severity = 2
        elif u_degradation_pct >= thresholds["u_light"]:
            u_severity = 1
        else:
            u_severity = 0

        # Take the worse of the two
        severity = max(dp_severity, u_severity)

        severity_map = {0: "none", 1: "light", 2: "moderate", 3: "severe", 4: "critical"}
        return severity_map.get(severity, "none")

    def determine_cleaning_status(
        self,
        dp_ratio: float,
        u_degradation_pct: float,
        hours_since_cleaning: Optional[float] = None,
    ) -> Tuple[str, bool, Optional[float]]:
        """
        Determine cleaning status and recommendations.

        Args:
            dp_ratio: Corrected DP ratio
            u_degradation_pct: U-value degradation percentage
            hours_since_cleaning: Hours since last cleaning

        Returns:
            Tuple of (cleaning_status, soot_blow_recommended, estimated_hours_to_cleaning)
        """
        # Cleaning thresholds
        soot_blow_dp_trigger = 1.2
        soot_blow_u_trigger = 5.0

        recommended_dp_ratio = 1.4
        required_dp_ratio = 1.6
        urgent_dp_ratio = 1.8

        recommended_u_degradation = 15.0
        required_u_degradation = 25.0
        urgent_u_degradation = 35.0

        # Determine soot blow recommendation
        soot_blow_recommended = (
            dp_ratio >= soot_blow_dp_trigger or
            u_degradation_pct >= soot_blow_u_trigger
        )

        # Determine cleaning status
        if dp_ratio >= urgent_dp_ratio or u_degradation_pct >= urgent_u_degradation:
            cleaning_status = "urgent"
        elif dp_ratio >= required_dp_ratio or u_degradation_pct >= required_u_degradation:
            cleaning_status = "required"
        elif dp_ratio >= recommended_dp_ratio or u_degradation_pct >= recommended_u_degradation:
            cleaning_status = "recommended"
        elif soot_blow_recommended:
            cleaning_status = "monitor"
        else:
            cleaning_status = "not_required"

        # Estimate hours to cleaning (simple linear projection)
        estimated_hours = None
        if hours_since_cleaning is not None and hours_since_cleaning > 0:
            # Assume linear fouling rate
            current_fouling_indicator = max(dp_ratio - 1.0, u_degradation_pct / 100)
            target_fouling_indicator = max(required_dp_ratio - 1.0, required_u_degradation / 100)

            if current_fouling_indicator > 0:
                fouling_rate = current_fouling_indicator / hours_since_cleaning
                remaining_fouling = target_fouling_indicator - current_fouling_indicator

                if remaining_fouling > 0 and fouling_rate > 0:
                    estimated_hours = remaining_fouling / fouling_rate

        return cleaning_status, soot_blow_recommended, estimated_hours

    def calculate_efficiency_impact(
        self,
        u_degradation_pct: float,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        design_outlet_temp_f: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate efficiency and fuel waste impact from fouling.

        Fouling causes higher gas outlet temperature, reducing heat recovery.
        Efficiency loss is approximately proportional to temperature increase.

        Rule of thumb: 1% efficiency loss per 40F increase in stack temperature.

        Args:
            u_degradation_pct: U-value degradation
            gas_inlet_temp_f: Gas inlet temperature
            gas_outlet_temp_f: Actual gas outlet temperature
            design_outlet_temp_f: Design gas outlet temperature

        Returns:
            Tuple of (efficiency_loss_pct, fuel_waste_pct)
        """
        if design_outlet_temp_f is None:
            # Estimate design outlet from degradation
            # If U is degraded, outlet temp is higher than it should be
            # Higher U degradation -> higher outlet temp deviation
            # Rough estimate: 1% U degradation -> 0.5F outlet temp increase
            outlet_temp_increase = u_degradation_pct * 0.5
            design_outlet_temp_f = gas_outlet_temp_f - outlet_temp_increase

        # Temperature rise above design
        temp_deviation_f = gas_outlet_temp_f - design_outlet_temp_f
        temp_deviation_f = max(0.0, temp_deviation_f)

        # Efficiency loss estimation
        # Industry rule of thumb: 1% efficiency loss per 40F stack temp increase
        efficiency_loss_pct = temp_deviation_f / 40.0

        # Fuel waste is related to efficiency loss
        # fuel_waste = efficiency_loss / (100 - efficiency_loss)
        # Approximation for small losses: fuel_waste ≈ efficiency_loss
        fuel_waste_pct = efficiency_loss_pct

        return round(efficiency_loss_pct, 2), round(fuel_waste_pct, 2)

    def analyze(self, input_data: GasSideFoulingInput) -> GasSideFoulingResult:
        """
        Perform complete gas-side fouling analysis.

        Args:
            input_data: GasSideFoulingInput with all required data

        Returns:
            GasSideFoulingResult with analysis results
        """
        start_time = datetime.now(timezone.utc)

        # Calculate flow ratio
        flow_ratio = input_data.current_gas_flow_lb_hr / input_data.design_gas_flow_lb_hr

        # Analyze pressure drop
        corrected_dp, dp_ratio, dp_deviation_pct = self.analyze_pressure_drop(
            input_data.current_dp_in_wc,
            input_data.design_dp_in_wc,
            flow_ratio,
        )

        # Determine U values
        if input_data.current_u_btu_hr_ft2_f is not None:
            current_u = input_data.current_u_btu_hr_ft2_f
        else:
            # Estimate from design with DP-based degradation
            degradation_factor = 1.0 / dp_ratio
            current_u = input_data.design_u_btu_hr_ft2_f * degradation_factor

        if input_data.clean_u_btu_hr_ft2_f is not None:
            clean_u = input_data.clean_u_btu_hr_ft2_f
        else:
            clean_u = input_data.design_u_btu_hr_ft2_f * 1.1  # Assume clean is 10% better than design

        # Analyze heat transfer degradation
        fouling_resistance, u_degradation_pct = self.analyze_heat_transfer_degradation(
            current_u,
            clean_u,
        )

        # Estimate fouling thickness
        fouling_thickness = self.estimate_fouling_thickness(fouling_resistance)

        # Differentiate fouling source
        fouling_type, confidence, explanation = self.differentiate_fouling_source(
            input_data.dp_history,
            input_data.u_history,
        )

        # Determine fouling severity
        severity = self.determine_fouling_severity(dp_ratio, u_degradation_pct)

        # Determine trend
        if len(input_data.dp_history) >= 3:
            dp_slope = self._calculate_trend_slope(input_data.dp_history)
            if dp_slope > 0.001:
                trend = "degrading"
            elif dp_slope < -0.001:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Determine cleaning status
        cleaning_status, soot_blow_recommended, est_hours = self.determine_cleaning_status(
            dp_ratio,
            u_degradation_pct,
        )

        # Calculate efficiency impact
        efficiency_loss, fuel_waste = self.calculate_efficiency_impact(
            u_degradation_pct,
            input_data.gas_inlet_temp_f,
            input_data.gas_outlet_temp_f,
        )

        # Build result
        result_data = {
            "fouling_detected": severity != "none",
            "fouling_severity": severity,
            "fouling_trend": trend,
            "current_dp_in_wc": round(input_data.current_dp_in_wc, 3),
            "design_dp_in_wc": round(input_data.design_dp_in_wc, 3),
            "corrected_dp_in_wc": round(corrected_dp, 3),
            "dp_ratio": round(dp_ratio, 3),
            "dp_deviation_pct": round(dp_deviation_pct, 1),
            "u_actual_btu_hr_ft2_f": round(current_u, 2),
            "u_clean_btu_hr_ft2_f": round(clean_u, 2),
            "u_degradation_pct": round(u_degradation_pct, 1),
            "fouling_resistance_hr_ft2_f_btu": round(fouling_resistance, 6),
            "estimated_fouling_thickness_in": round(fouling_thickness, 4) if fouling_thickness > 0 else None,
            "efficiency_loss_pct": efficiency_loss,
            "fuel_waste_pct": fuel_waste,
            "cleaning_status": cleaning_status,
            "soot_blow_recommended": soot_blow_recommended,
            "estimated_hours_to_cleaning": round(est_hours, 1) if est_hours else None,
            "calculation_method": "ASME_PTC_4.3",
        }

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            json.dumps(result_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        result = GasSideFoulingResult(
            **result_data,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Gas-side fouling analysis complete: severity={severity}, "
            f"dp_ratio={dp_ratio:.3f}, u_degradation={u_degradation_pct:.1f}%"
        )

        return result


def create_gas_side_fouling_analyzer(
    ash_thermal_conductivity: float = ASH_THERMAL_CONDUCTIVITY,
) -> GasSideFoulingAnalyzer:
    """Factory function to create GasSideFoulingAnalyzer."""
    return GasSideFoulingAnalyzer(ash_thermal_conductivity=ash_thermal_conductivity)
