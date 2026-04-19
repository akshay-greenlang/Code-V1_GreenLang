"""
GL-016 WATERGUARD Agent - Deaerator Performance Module

Implements deaerator performance monitoring including:
- Oxygen removal efficiency
- CO2 removal efficiency
- Steam consumption analysis
- Vent rate optimization
- Corrosion potential assessment

All calculations are deterministic with zero hallucination.

References:
    - ASME Performance Test Code for Deaerators
    - Heat Exchange Institute Standards for Deaerators
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    DeaerationInput,
    DeaerationOutput,
    WaterQualityStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class DeaerationConstants:
    """Constants for deaerator calculations."""

    # Saturation temperatures at common DA pressures (psig: temp_F)
    SATURATION_TEMPS = {
        0: 212.0,
        3: 221.5,
        5: 227.1,
        7: 232.4,
        10: 240.1,
        15: 250.3,
        20: 259.3,
        25: 267.3,
    }

    # Oxygen removal efficiency targets (%)
    O2_REMOVAL_EXCELLENT = 99.9
    O2_REMOVAL_GOOD = 99.5
    O2_REMOVAL_MIN = 99.0

    # Outlet O2 limits (ppb)
    OUTLET_O2_EXCELLENT = 5
    OUTLET_O2_LIMIT = 7
    OUTLET_O2_ACTION = 15

    # Vent rate factors
    MIN_VENT_RATE_PCT = 0.5
    MAX_VENT_RATE_PCT = 2.0
    TYPICAL_VENT_RATE_PCT = 1.0

    # Heat capacity of water (BTU/lb-F)
    CP_WATER = 1.0


# =============================================================================
# DEAERATOR ANALYZER CLASS
# =============================================================================

class DeaeratorAnalyzer:
    """
    Analyzes deaerator performance.

    Monitors oxygen removal efficiency, steam consumption, and
    identifies optimization opportunities.

    Attributes:
        o2_limit_ppb: Maximum acceptable outlet O2

    Example:
        >>> analyzer = DeaeratorAnalyzer()
        >>> result = analyzer.analyze(deaeration_input)
        >>> print(f"O2 removal: {result.oxygen_removal_efficiency_pct}%")
    """

    def __init__(
        self,
        o2_limit_ppb: float = 7.0,
    ) -> None:
        """
        Initialize DeaeratorAnalyzer.

        Args:
            o2_limit_ppb: Maximum acceptable outlet O2 (ppb)
        """
        self.o2_limit_ppb = o2_limit_ppb
        logger.info("DeaeratorAnalyzer initialized")

    def analyze(self, input_data: DeaerationInput) -> DeaerationOutput:
        """
        Analyze deaerator performance.

        Args:
            input_data: Deaerator operating data

        Returns:
            DeaerationOutput with performance analysis
        """
        start_time = datetime.now(timezone.utc)
        logger.debug("Analyzing deaerator performance")

        # Calculate oxygen removal efficiency
        o2_removal_eff = self._calculate_o2_removal_efficiency(
            input_data.inlet_dissolved_oxygen_ppb,
            input_data.outlet_dissolved_oxygen_ppb,
        )

        # Calculate CO2 removal efficiency
        co2_removal_eff = None
        if input_data.inlet_co2_ppm and input_data.outlet_co2_ppm:
            co2_removal_eff = self._calculate_co2_removal_efficiency(
                input_data.inlet_co2_ppm,
                input_data.outlet_co2_ppm,
            )

        # Check outlet O2 limit
        o2_within_limit = input_data.outlet_dissolved_oxygen_ppb <= self.o2_limit_ppb

        # Determine performance status
        performance_status = self._determine_performance_status(
            o2_removal_eff,
            input_data.outlet_dissolved_oxygen_ppb,
        )

        # Get saturation temperature
        sat_temp = self._get_saturation_temperature(input_data.deaerator_pressure_psig)

        # Calculate subcooling
        actual_temp = input_data.deaerator_temperature_f or sat_temp
        subcooling = sat_temp - actual_temp

        # Calculate steam requirements
        theoretical_steam = self._calculate_theoretical_steam(
            input_data.total_flow_lb_hr,
            input_data.inlet_water_temperature_f,
            sat_temp,
        )

        # Calculate steam efficiency
        steam_efficiency = None
        if input_data.steam_flow_lb_hr and input_data.steam_flow_lb_hr > 0:
            steam_efficiency = (theoretical_steam / input_data.steam_flow_lb_hr) * 100

        # Calculate recommended vent rate
        recommended_vent = self._calculate_recommended_vent_rate(
            input_data.total_flow_lb_hr,
            input_data.inlet_dissolved_oxygen_ppb,
        )

        # Evaluate vent rate status
        vent_status = self._evaluate_vent_rate(
            input_data.vent_rate_lb_hr,
            recommended_vent,
        )

        # Assess corrosion potential
        corrosion_potential = self._assess_corrosion_potential(
            input_data.outlet_dissolved_oxygen_ppb,
            input_data.outlet_co2_ppm,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            input_data,
            o2_removal_eff,
            subcooling,
            vent_status,
            performance_status,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(input_data)
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return DeaerationOutput(
            timestamp=datetime.now(timezone.utc),
            oxygen_removal_efficiency_pct=round(o2_removal_eff, 2),
            co2_removal_efficiency_pct=round(co2_removal_eff, 2) if co2_removal_eff else None,
            performance_status=performance_status,
            outlet_o2_within_limit=o2_within_limit,
            saturation_temperature_f=round(sat_temp, 1),
            subcooling_f=round(subcooling, 1),
            theoretical_steam_lb_hr=round(theoretical_steam, 1),
            actual_steam_lb_hr=input_data.steam_flow_lb_hr,
            steam_efficiency_pct=round(steam_efficiency, 1) if steam_efficiency else None,
            vent_rate_recommended_lb_hr=round(recommended_vent, 1),
            vent_rate_status=vent_status,
            corrosion_potential=corrosion_potential,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

    def _calculate_o2_removal_efficiency(
        self,
        inlet_o2_ppb: float,
        outlet_o2_ppb: float,
    ) -> float:
        """
        Calculate oxygen removal efficiency.

        Efficiency = (Inlet O2 - Outlet O2) / Inlet O2 * 100

        Args:
            inlet_o2_ppb: Inlet dissolved oxygen (ppb)
            outlet_o2_ppb: Outlet dissolved oxygen (ppb)

        Returns:
            Oxygen removal efficiency (%)
        """
        if inlet_o2_ppb <= 0:
            logger.warning("Inlet O2 is zero or negative")
            return 0.0

        efficiency = ((inlet_o2_ppb - outlet_o2_ppb) / inlet_o2_ppb) * 100
        return max(0, min(efficiency, 100))

    def _calculate_co2_removal_efficiency(
        self,
        inlet_co2_ppm: float,
        outlet_co2_ppm: float,
    ) -> float:
        """Calculate CO2 removal efficiency."""
        if inlet_co2_ppm <= 0:
            return 0.0

        efficiency = ((inlet_co2_ppm - outlet_co2_ppm) / inlet_co2_ppm) * 100
        return max(0, min(efficiency, 100))

    def _determine_performance_status(
        self,
        o2_removal_eff: float,
        outlet_o2_ppb: float,
    ) -> WaterQualityStatus:
        """Determine overall deaerator performance status."""
        # Critical if outlet O2 very high
        if outlet_o2_ppb > DeaerationConstants.OUTLET_O2_ACTION:
            return WaterQualityStatus.CRITICAL

        # Out of spec if above limit
        if outlet_o2_ppb > DeaerationConstants.OUTLET_O2_LIMIT:
            return WaterQualityStatus.OUT_OF_SPEC

        # Excellent performance
        if (o2_removal_eff >= DeaerationConstants.O2_REMOVAL_EXCELLENT and
                outlet_o2_ppb <= DeaerationConstants.OUTLET_O2_EXCELLENT):
            return WaterQualityStatus.EXCELLENT

        # Good performance
        if o2_removal_eff >= DeaerationConstants.O2_REMOVAL_GOOD:
            return WaterQualityStatus.GOOD

        # Acceptable
        if o2_removal_eff >= DeaerationConstants.O2_REMOVAL_MIN:
            return WaterQualityStatus.ACCEPTABLE

        return WaterQualityStatus.WARNING

    def _get_saturation_temperature(self, pressure_psig: float) -> float:
        """Get saturation temperature at given pressure."""
        pressures = sorted(DeaerationConstants.SATURATION_TEMPS.keys())

        if pressure_psig <= pressures[0]:
            return DeaerationConstants.SATURATION_TEMPS[pressures[0]]
        if pressure_psig >= pressures[-1]:
            return DeaerationConstants.SATURATION_TEMPS[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1 = DeaerationConstants.SATURATION_TEMPS[p1]
                t2 = DeaerationConstants.SATURATION_TEMPS[p2]
                return t1 + (t2 - t1) * (pressure_psig - p1) / (p2 - p1)

        return 212.0

    def _calculate_theoretical_steam(
        self,
        water_flow_lb_hr: float,
        inlet_temp_f: float,
        saturation_temp_f: float,
    ) -> float:
        """
        Calculate theoretical steam requirement for heating.

        Steam = Water * Cp * dT / hfg

        Args:
            water_flow_lb_hr: Water flow rate (lb/hr)
            inlet_temp_f: Inlet water temperature (F)
            saturation_temp_f: DA saturation temperature (F)

        Returns:
            Theoretical steam requirement (lb/hr)
        """
        # Temperature rise required
        temp_rise = saturation_temp_f - inlet_temp_f
        if temp_rise <= 0:
            return 0.0

        # Heat required (BTU/hr)
        heat_required = water_flow_lb_hr * DeaerationConstants.CP_WATER * temp_rise

        # Latent heat of steam at DA pressure (approximate)
        # At 5 psig: hfg ~ 960 BTU/lb
        hfg = 960.0

        # Steam required
        steam_required = heat_required / hfg

        return steam_required

    def _calculate_recommended_vent_rate(
        self,
        water_flow_lb_hr: float,
        inlet_o2_ppb: float,
    ) -> float:
        """
        Calculate recommended vent rate.

        Vent rate should be sufficient to carry away released O2 and
        non-condensables, typically 0.5-2% of steam.

        Args:
            water_flow_lb_hr: Water flow rate (lb/hr)
            inlet_o2_ppb: Inlet dissolved oxygen (ppb)

        Returns:
            Recommended vent rate (lb/hr)
        """
        # Base vent rate as percentage of heating steam
        # Assume heating steam is ~5% of water flow for typical temperature rise
        estimated_steam = water_flow_lb_hr * 0.05

        # Adjust vent rate based on O2 loading
        if inlet_o2_ppb > 1000:
            vent_pct = DeaerationConstants.MAX_VENT_RATE_PCT
        elif inlet_o2_ppb > 500:
            vent_pct = DeaerationConstants.TYPICAL_VENT_RATE_PCT * 1.5
        elif inlet_o2_ppb > 200:
            vent_pct = DeaerationConstants.TYPICAL_VENT_RATE_PCT
        else:
            vent_pct = DeaerationConstants.MIN_VENT_RATE_PCT

        recommended_vent = estimated_steam * (vent_pct / 100)

        return recommended_vent

    def _evaluate_vent_rate(
        self,
        actual_vent: Optional[float],
        recommended_vent: float,
    ) -> str:
        """Evaluate vent rate adequacy."""
        if actual_vent is None:
            return "unknown"

        if actual_vent < recommended_vent * 0.5:
            return "insufficient"
        elif actual_vent > recommended_vent * 2.0:
            return "excessive"
        else:
            return "adequate"

    def _assess_corrosion_potential(
        self,
        outlet_o2_ppb: float,
        outlet_co2_ppm: Optional[float],
    ) -> str:
        """
        Assess downstream corrosion potential.

        Args:
            outlet_o2_ppb: Outlet dissolved oxygen (ppb)
            outlet_co2_ppm: Outlet CO2 (ppm)

        Returns:
            Corrosion potential rating (low/medium/high)
        """
        # Score based on O2
        o2_score = 0
        if outlet_o2_ppb > DeaerationConstants.OUTLET_O2_ACTION:
            o2_score = 3
        elif outlet_o2_ppb > DeaerationConstants.OUTLET_O2_LIMIT:
            o2_score = 2
        elif outlet_o2_ppb > DeaerationConstants.OUTLET_O2_EXCELLENT:
            o2_score = 1

        # Score based on CO2
        co2_score = 0
        if outlet_co2_ppm:
            if outlet_co2_ppm > 5:
                co2_score = 2
            elif outlet_co2_ppm > 1:
                co2_score = 1

        total_score = o2_score + co2_score

        if total_score >= 4:
            return "high"
        elif total_score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        input_data: DeaerationInput,
        o2_removal_eff: float,
        subcooling: float,
        vent_status: str,
        performance_status: WaterQualityStatus,
    ) -> List[str]:
        """Generate deaerator optimization recommendations."""
        recommendations = []

        # High outlet O2
        if input_data.outlet_dissolved_oxygen_ppb > self.o2_limit_ppb:
            recommendations.append(
                f"Outlet O2 ({input_data.outlet_dissolved_oxygen_ppb} ppb) exceeds limit "
                f"({self.o2_limit_ppb} ppb) - investigate DA operation"
            )

        # Low removal efficiency
        if o2_removal_eff < DeaerationConstants.O2_REMOVAL_MIN:
            recommendations.append(
                f"O2 removal efficiency ({o2_removal_eff:.1f}%) below minimum - "
                f"check spray nozzles and tray condition"
            )

        # Subcooling
        if subcooling > 5:
            recommendations.append(
                f"Subcooling of {subcooling:.1f}F detected - "
                f"increase DA pressure or steam supply"
            )
        elif subcooling < -2:
            recommendations.append(
                f"DA temperature above saturation - check pressure gauge calibration"
            )

        # Vent rate
        if vent_status == "insufficient":
            recommendations.append(
                "Vent rate may be insufficient - increase vent valve opening "
                "to improve gas removal"
            )
        elif vent_status == "excessive":
            recommendations.append(
                "Vent rate may be excessive - reduce to minimize steam loss"
            )

        # Low pressure
        if input_data.deaerator_pressure_psig < 3:
            recommendations.append(
                "DA pressure below recommended minimum - risk of inadequate O2 removal"
            )

        # Good performance
        if performance_status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]:
            recommendations.append(
                "Deaerator performing well - continue current operation"
            )

        return recommendations

    def _calculate_provenance_hash(self, input_data: DeaerationInput) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data_str = json.dumps(input_data.dict(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


def calculate_deaerator_capacity(
    water_flow_lb_hr: float,
    inlet_temp_f: float,
    da_pressure_psig: float = 5.0,
) -> Dict[str, float]:
    """
    Calculate deaerator capacity and steam requirements.

    Args:
        water_flow_lb_hr: Water flow through DA (lb/hr)
        inlet_temp_f: Inlet water temperature (F)
        da_pressure_psig: DA operating pressure (psig)

    Returns:
        Dictionary with capacity metrics
    """
    # Get saturation temperature
    sat_temps = DeaerationConstants.SATURATION_TEMPS
    pressures = sorted(sat_temps.keys())

    sat_temp = sat_temps.get(da_pressure_psig)
    if sat_temp is None:
        # Interpolate
        for i in range(len(pressures) - 1):
            if pressures[i] <= da_pressure_psig <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1 = sat_temps[p1]
                t2 = sat_temps[p2]
                sat_temp = t1 + (t2 - t1) * (da_pressure_psig - p1) / (p2 - p1)
                break
        else:
            sat_temp = 227.1  # Default to 5 psig

    # Temperature rise
    temp_rise = sat_temp - inlet_temp_f

    # Heat duty
    heat_duty = water_flow_lb_hr * 1.0 * temp_rise

    # Steam requirement (assuming hfg = 960)
    steam_required = heat_duty / 960

    # Storage volume requirement (10 min minimum)
    storage_gallons = (water_flow_lb_hr / 8.34) * (10 / 60)

    return {
        "saturation_temp_f": sat_temp,
        "temp_rise_f": temp_rise,
        "heat_duty_btu_hr": heat_duty,
        "steam_required_lb_hr": steam_required,
        "storage_volume_gal": storage_gallons,
    }
