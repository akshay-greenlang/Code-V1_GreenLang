"""
GL-017 CONDENSYNC Agent - Air Ingress Detection Module

This module implements air ingress detection and source identification
for steam surface condensers. It uses multiple indicators including
dissolved oxygen, subcooling, and vacuum behavior to detect and
quantify air in-leakage.

All calculations are deterministic with zero hallucination.

Example:
    >>> detector = AirIngresDetector(config)
    >>> result = detector.detect_air_ingress(
    ...     dissolved_o2_ppb=25.0,
    ...     subcooling_f=2.5,
    ...     condenser_vacuum_inhga=1.6,
    ... )
    >>> print(f"Air ingress detected: {result.air_ingress_detected}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    AirIngresConfig,
    VacuumSystemConfig,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    AirIngresResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Air Ingress Detection Parameters
# =============================================================================

class AirIngressConstants:
    """Air ingress detection constants."""

    # Dissolved oxygen equilibrium with air (ppb at various temps)
    # At atmospheric pressure, water saturates with O2
    DO_SATURATION_PPB = {
        60: 12_800,
        70: 11_300,
        80: 10_000,
        90: 8_900,
        100: 7_900,
        110: 7_100,
        120: 6_400,
    }

    # Expected DO in deaerated condensate (well-sealed system)
    TARGET_DO_PPB = 7.0  # HEI/EPRI recommended maximum

    # Subcooling thresholds
    NORMAL_SUBCOOLING_F = 1.0
    WARNING_SUBCOOLING_F = 3.0
    ALARM_SUBCOOLING_F = 5.0

    # Heat rate impact from subcooling
    # Each degree F subcooling = ~5 BTU/kWh heat rate penalty
    HEAT_RATE_PENALTY_BTU_KWH_PER_F = 5.0

    # Common air ingress sources and typical SCFM contribution
    COMMON_LEAK_SOURCES = {
        "turbine_shaft_seals": (2.0, 10.0),  # (min, max) SCFM
        "lp_turbine_hood_joints": (1.0, 5.0),
        "condenser_shell_joints": (0.5, 3.0),
        "instrument_taps": (0.1, 0.5),
        "valve_packing": (0.2, 2.0),
        "expansion_joints": (1.0, 5.0),
        "manways": (0.5, 2.0),
        "vacuum_breaker": (0.1, 1.0),
        "hotwell_drains": (0.2, 1.0),
        "feedwater_heater_drains": (0.5, 3.0),
        "condensate_pump_seals": (0.2, 1.0),
    }

    # Leak detection zones for systematic surveys
    DETECTION_ZONES = [
        "LP_turbine_exhaust",
        "condenser_shell_top",
        "condenser_shell_sides",
        "hotwell_area",
        "expansion_joints",
        "instrument_connections",
        "valve_bonnets",
        "manway_covers",
        "drain_connections",
        "pump_seals",
    ]


@dataclass
class AirIngressReading:
    """Historical air ingress indicator reading."""
    timestamp: datetime
    dissolved_o2_ppb: Optional[float]
    subcooling_f: Optional[float]
    vacuum_inhga: float
    air_removal_scfm: Optional[float]


class AirIngressDetector:
    """
    Air ingress detection and source identification.

    This class detects air in-leakage in condensers using multiple
    indicators and provides source identification assistance.

    Detection Methods:
        1. Dissolved oxygen monitoring
        2. Condensate subcooling analysis
        3. Vacuum behavior analysis
        4. Air removal rate trending

    Attributes:
        config: Air ingress configuration
        vacuum_config: Vacuum system configuration
        performance_config: Performance configuration

    Example:
        >>> config = AirIngresConfig()
        >>> detector = AirIngressDetector(config)
        >>> result = detector.detect_air_ingress(...)
    """

    def __init__(
        self,
        air_ingress_config: AirIngresConfig,
        vacuum_config: VacuumSystemConfig,
        performance_config: PerformanceConfig,
    ) -> None:
        """
        Initialize the air ingress detector.

        Args:
            air_ingress_config: Air ingress configuration
            vacuum_config: Vacuum system configuration
            performance_config: Performance configuration
        """
        self.config = air_ingress_config
        self.vacuum_config = vacuum_config
        self.performance_config = performance_config
        self._history: List[AirIngressReading] = []
        self._calculation_count = 0

        logger.info("AirIngressDetector initialized")

    def detect_air_ingress(
        self,
        dissolved_o2_ppb: Optional[float] = None,
        subcooling_f: Optional[float] = None,
        condenser_vacuum_inhga: float = 1.5,
        expected_vacuum_inhga: Optional[float] = None,
        air_removal_scfm: Optional[float] = None,
        saturation_temp_f: Optional[float] = None,
        hotwell_temp_f: Optional[float] = None,
    ) -> AirIngresResult:
        """
        Detect air ingress using multiple indicators.

        Uses a weighted scoring approach combining:
        - Dissolved oxygen level
        - Condensate subcooling
        - Vacuum deviation
        - Air removal rate

        Args:
            dissolved_o2_ppb: Dissolved oxygen in condensate (ppb)
            subcooling_f: Subcooling below saturation (F)
            condenser_vacuum_inhga: Current vacuum (inHgA)
            expected_vacuum_inhga: Expected vacuum (inHgA)
            air_removal_scfm: Measured air removal (SCFM)
            saturation_temp_f: Saturation temperature at vacuum (F)
            hotwell_temp_f: Hotwell temperature (F)

        Returns:
            AirIngresResult with detection analysis
        """
        logger.debug(
            f"Detecting air ingress: DO={dissolved_o2_ppb}, "
            f"SC={subcooling_f}F, vacuum={condenser_vacuum_inhga} inHgA"
        )
        self._calculation_count += 1

        # Calculate subcooling if temperatures provided
        if subcooling_f is None and saturation_temp_f and hotwell_temp_f:
            subcooling_f = saturation_temp_f - hotwell_temp_f

        # Expected vacuum default
        if expected_vacuum_inhga is None:
            expected_vacuum_inhga = self.vacuum_config.design_vacuum_inhga

        # Score each indicator
        do_score = self._score_dissolved_oxygen(dissolved_o2_ppb)
        sc_score = self._score_subcooling(subcooling_f)
        vacuum_score = self._score_vacuum_deviation(
            condenser_vacuum_inhga, expected_vacuum_inhga
        )
        air_rate_score = self._score_air_removal_rate(air_removal_scfm)

        # Combined weighted score
        weights = {
            "dissolved_oxygen": 0.35,
            "subcooling": 0.30,
            "vacuum": 0.20,
            "air_rate": 0.15,
        }

        total_score = (
            weights["dissolved_oxygen"] * do_score +
            weights["subcooling"] * sc_score +
            weights["vacuum"] * vacuum_score +
            weights["air_rate"] * air_rate_score
        )

        # Determine severity
        air_ingress_detected = total_score > 0.3
        severity = self._determine_severity(total_score)

        # Estimate air ingress rate
        estimated_ingress = self._estimate_air_ingress_rate(
            dissolved_o2_ppb, subcooling_f,
            condenser_vacuum_inhga, expected_vacuum_inhga,
            air_removal_scfm
        )

        # Identify probable leak locations
        probable_locations = self._identify_probable_locations(
            dissolved_o2_ppb, subcooling_f,
            estimated_ingress, air_removal_scfm
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            dissolved_o2_ppb is not None,
            subcooling_f is not None,
            air_removal_scfm is not None
        )

        # Calculate impacts
        heat_rate_impact = self._calculate_heat_rate_impact(
            subcooling_f, estimated_ingress
        )
        do_impact = self._assess_do_impact(dissolved_o2_ppb)

        # Determine recommendations
        leak_testing_recommended = total_score > 0.5
        test_method = self._recommend_test_method(severity, probable_locations)

        # Record reading
        self._record_reading(
            dissolved_o2_ppb, subcooling_f,
            condenser_vacuum_inhga, air_removal_scfm
        )

        result = AirIngresResult(
            air_ingress_detected=air_ingress_detected,
            ingress_severity=severity,
            estimated_air_ingress_scfm=round(estimated_ingress, 2),
            subcooling_observed_f=round(subcooling_f or 0.0, 2),
            dissolved_o2_ppb=dissolved_o2_ppb,
            probable_leak_locations=probable_locations,
            confidence_pct=round(confidence, 1),
            heat_rate_impact_btu_kwh=round(heat_rate_impact, 1),
            dissolved_o2_impact=do_impact,
            leak_testing_recommended=leak_testing_recommended,
            recommended_test_method=test_method,
        )

        logger.info(
            f"Air ingress detection complete: detected={air_ingress_detected}, "
            f"severity={severity}, estimated={estimated_ingress:.2f} SCFM"
        )

        return result

    def analyze_leak_survey_results(
        self,
        zone_readings: Dict[str, float],
        baseline_readings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Analyze leak survey results.

        Compares readings from different zones to identify
        leak locations.

        Args:
            zone_readings: Dictionary of zone name to reading value
            baseline_readings: Optional baseline readings for comparison

        Returns:
            Analysis results with identified leak locations
        """
        logger.info("Analyzing leak survey results")

        identified_leaks = []
        total_estimated_ingress = 0.0

        for zone, reading in zone_readings.items():
            baseline = (
                baseline_readings.get(zone, 0.0)
                if baseline_readings else 0.0
            )
            deviation = reading - baseline

            # Check if reading indicates a leak
            if deviation > 0.1:  # Threshold for detection
                # Estimate contribution
                contribution = self._estimate_zone_contribution(zone, deviation)
                total_estimated_ingress += contribution

                identified_leaks.append({
                    "zone": zone,
                    "reading": reading,
                    "baseline": baseline,
                    "deviation": deviation,
                    "estimated_scfm": contribution,
                    "priority": self._prioritize_leak(contribution),
                })

        # Sort by contribution
        identified_leaks.sort(key=lambda x: x["estimated_scfm"], reverse=True)

        return {
            "leaks_identified": len(identified_leaks),
            "total_estimated_ingress_scfm": round(total_estimated_ingress, 2),
            "leak_details": identified_leaks,
            "recommended_repair_order": [
                leak["zone"] for leak in identified_leaks
            ],
            "estimated_repair_benefit_scfm": round(
                total_estimated_ingress * 0.8, 2
            ),  # Assume 80% reduction
        }

    def _score_dissolved_oxygen(
        self,
        do_ppb: Optional[float],
    ) -> float:
        """
        Score dissolved oxygen indicator.

        Args:
            do_ppb: Dissolved oxygen (ppb)

        Returns:
            Score 0-1 (0=no indication, 1=severe indication)
        """
        if do_ppb is None:
            return 0.0

        target = AirIngressConstants.TARGET_DO_PPB
        warning = self.config.do_warning_ppb
        alarm = self.config.do_alarm_ppb

        if do_ppb <= target:
            return 0.0
        elif do_ppb <= warning:
            return (do_ppb - target) / (warning - target) * 0.5
        elif do_ppb <= alarm:
            return 0.5 + (do_ppb - warning) / (alarm - warning) * 0.3
        else:
            return min(1.0, 0.8 + (do_ppb - alarm) / alarm * 0.2)

    def _score_subcooling(
        self,
        subcooling_f: Optional[float],
    ) -> float:
        """
        Score subcooling indicator.

        Args:
            subcooling_f: Subcooling (F)

        Returns:
            Score 0-1
        """
        if subcooling_f is None:
            return 0.0

        normal = AirIngressConstants.NORMAL_SUBCOOLING_F
        warning = self.config.subcooling_warning_f
        alarm = self.config.subcooling_alarm_f

        if subcooling_f <= normal:
            return 0.0
        elif subcooling_f <= warning:
            return (subcooling_f - normal) / (warning - normal) * 0.5
        elif subcooling_f <= alarm:
            return 0.5 + (subcooling_f - warning) / (alarm - warning) * 0.3
        else:
            return min(1.0, 0.8 + (subcooling_f - alarm) / alarm * 0.2)

    def _score_vacuum_deviation(
        self,
        actual_vacuum: float,
        expected_vacuum: float,
    ) -> float:
        """
        Score vacuum deviation indicator.

        Args:
            actual_vacuum: Actual vacuum (inHgA)
            expected_vacuum: Expected vacuum (inHgA)

        Returns:
            Score 0-1
        """
        deviation = actual_vacuum - expected_vacuum

        if deviation <= 0:
            return 0.0

        # Vacuum degradation indicates air ingress
        if deviation < 0.2:
            return deviation / 0.2 * 0.3
        elif deviation < 0.5:
            return 0.3 + (deviation - 0.2) / 0.3 * 0.4
        else:
            return min(1.0, 0.7 + (deviation - 0.5) / 0.5 * 0.3)

    def _score_air_removal_rate(
        self,
        air_removal_scfm: Optional[float],
    ) -> float:
        """
        Score air removal rate indicator.

        Args:
            air_removal_scfm: Measured air removal (SCFM)

        Returns:
            Score 0-1
        """
        if air_removal_scfm is None:
            return 0.0

        design_capacity = self.vacuum_config.air_removal_capacity_scfm
        normal_load = design_capacity * 0.5  # Normal is ~50% of capacity

        if air_removal_scfm <= normal_load:
            return 0.0

        utilization = air_removal_scfm / design_capacity

        if utilization < 0.7:
            return (utilization - 0.5) / 0.2 * 0.3
        elif utilization < 0.9:
            return 0.3 + (utilization - 0.7) / 0.2 * 0.4
        else:
            return min(1.0, 0.7 + (utilization - 0.9) / 0.1 * 0.3)

    def _determine_severity(
        self,
        score: float,
    ) -> str:
        """
        Determine severity from combined score.

        Args:
            score: Combined detection score (0-1)

        Returns:
            Severity level
        """
        if score < 0.3:
            return "none"
        elif score < 0.5:
            return "minor"
        elif score < 0.7:
            return "moderate"
        else:
            return "severe"

    def _estimate_air_ingress_rate(
        self,
        dissolved_o2_ppb: Optional[float],
        subcooling_f: Optional[float],
        actual_vacuum: float,
        expected_vacuum: float,
        air_removal_scfm: Optional[float],
    ) -> float:
        """
        Estimate air ingress rate from indicators.

        Args:
            dissolved_o2_ppb: Dissolved oxygen (ppb)
            subcooling_f: Subcooling (F)
            actual_vacuum: Actual vacuum (inHgA)
            expected_vacuum: Expected vacuum (inHgA)
            air_removal_scfm: Measured air removal (SCFM)

        Returns:
            Estimated air ingress (SCFM)
        """
        # If air removal is measured, that's the best estimate
        if air_removal_scfm is not None:
            return air_removal_scfm

        # Estimate from other indicators
        estimates = []

        # Vacuum-based estimate
        vacuum_deviation = actual_vacuum - expected_vacuum
        if vacuum_deviation > 0:
            # Rule of thumb: 0.1 inHg = ~5 SCFM air ingress
            estimates.append(vacuum_deviation * 50)

        # DO-based estimate (if available)
        if dissolved_o2_ppb is not None:
            target_do = AirIngressConstants.TARGET_DO_PPB
            if dissolved_o2_ppb > target_do:
                # Higher DO indicates more air contact
                do_ratio = dissolved_o2_ppb / target_do
                estimates.append((do_ratio - 1) * 5)  # Rough correlation

        # Subcooling-based estimate
        if subcooling_f is not None:
            if subcooling_f > 1.0:
                # Subcooling indicates air blanket
                estimates.append((subcooling_f - 1.0) * 3)

        if not estimates:
            return 0.0

        return sum(estimates) / len(estimates)

    def _identify_probable_locations(
        self,
        dissolved_o2_ppb: Optional[float],
        subcooling_f: Optional[float],
        estimated_ingress: float,
        air_removal_scfm: Optional[float],
    ) -> List[str]:
        """
        Identify probable leak locations.

        Args:
            dissolved_o2_ppb: Dissolved oxygen (ppb)
            subcooling_f: Subcooling (F)
            estimated_ingress: Estimated air ingress (SCFM)
            air_removal_scfm: Measured air removal (SCFM)

        Returns:
            List of probable leak locations
        """
        if estimated_ingress < 1.0:
            return []

        probable = []

        # Large ingress suggests major sources
        if estimated_ingress > 10:
            probable.extend([
                "LP_turbine_shaft_seals",
                "LP_turbine_hood_joints",
                "expansion_joints",
            ])
        elif estimated_ingress > 5:
            probable.extend([
                "turbine_shaft_seals",
                "condenser_shell_joints",
                "expansion_joints",
            ])
        else:
            probable.extend([
                "instrument_taps",
                "valve_packing",
                "manways",
            ])

        # High DO suggests contact with air in condensate path
        if dissolved_o2_ppb and dissolved_o2_ppb > 50:
            probable.append("condensate_pump_seals")
            probable.append("hotwell_drains")

        # High subcooling suggests air blanket on tubes
        if subcooling_f and subcooling_f > 3:
            probable.append("condenser_shell_top")
            probable.append("air_removal_system_inlet")

        return list(set(probable))[:5]  # Return top 5 unique

    def _calculate_confidence(
        self,
        has_do: bool,
        has_subcooling: bool,
        has_air_rate: bool,
    ) -> float:
        """
        Calculate confidence in detection.

        Args:
            has_do: Dissolved oxygen available
            has_subcooling: Subcooling available
            has_air_rate: Air removal rate available

        Returns:
            Confidence percentage
        """
        base_confidence = 50.0  # Vacuum is always available

        if has_do:
            base_confidence += 20.0
        if has_subcooling:
            base_confidence += 15.0
        if has_air_rate:
            base_confidence += 15.0

        return min(100.0, base_confidence)

    def _calculate_heat_rate_impact(
        self,
        subcooling_f: Optional[float],
        estimated_ingress: float,
    ) -> float:
        """
        Calculate heat rate impact from air ingress.

        Args:
            subcooling_f: Subcooling (F)
            estimated_ingress: Estimated air ingress (SCFM)

        Returns:
            Heat rate impact (BTU/kWh)
        """
        impact = 0.0

        # Subcooling impact
        if subcooling_f and subcooling_f > 1.0:
            impact += (subcooling_f - 1.0) * AirIngressConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_F

        # Air blanket reduces heat transfer - additional penalty
        if estimated_ingress > 5:
            impact += estimated_ingress * 0.5

        return impact

    def _assess_do_impact(
        self,
        dissolved_o2_ppb: Optional[float],
    ) -> str:
        """
        Assess impact of dissolved oxygen on system.

        Args:
            dissolved_o2_ppb: Dissolved oxygen (ppb)

        Returns:
            Impact description
        """
        if dissolved_o2_ppb is None:
            return "unknown"

        if dissolved_o2_ppb <= 7:
            return "none"
        elif dissolved_o2_ppb <= 20:
            return "minor_corrosion_risk"
        elif dissolved_o2_ppb <= 50:
            return "moderate_corrosion_risk"
        else:
            return "severe_corrosion_risk"

    def _recommend_test_method(
        self,
        severity: str,
        probable_locations: List[str],
    ) -> Optional[str]:
        """
        Recommend leak detection test method.

        Args:
            severity: Ingress severity
            probable_locations: Probable leak locations

        Returns:
            Recommended test method
        """
        if severity == "none":
            return None

        if severity == "minor":
            return "ultrasonic_leak_detection"
        elif severity == "moderate":
            return "helium_tracer_survey"
        else:
            return "comprehensive_helium_tracer_with_mass_spectrometer"

    def _estimate_zone_contribution(
        self,
        zone: str,
        deviation: float,
    ) -> float:
        """
        Estimate contribution from a zone based on deviation.

        Args:
            zone: Zone name
            deviation: Reading deviation from baseline

        Returns:
            Estimated SCFM contribution
        """
        # Map zones to typical leak ranges
        zone_ranges = AirIngressConstants.COMMON_LEAK_SOURCES

        # Find matching source
        for source, (min_scfm, max_scfm) in zone_ranges.items():
            if source.lower() in zone.lower():
                # Scale by deviation
                return min_scfm + deviation * (max_scfm - min_scfm)

        # Default estimate
        return deviation * 2.0

    def _prioritize_leak(
        self,
        contribution_scfm: float,
    ) -> str:
        """
        Prioritize leak for repair.

        Args:
            contribution_scfm: Estimated contribution (SCFM)

        Returns:
            Priority level
        """
        if contribution_scfm > 5:
            return "high"
        elif contribution_scfm > 2:
            return "medium"
        else:
            return "low"

    def _record_reading(
        self,
        dissolved_o2: Optional[float],
        subcooling: Optional[float],
        vacuum: float,
        air_removal: Optional[float],
    ) -> None:
        """Record an air ingress reading."""
        reading = AirIngressReading(
            timestamp=datetime.now(timezone.utc),
            dissolved_o2_ppb=dissolved_o2,
            subcooling_f=subcooling,
            vacuum_inhga=vacuum,
            air_removal_scfm=air_removal,
        )
        self._history.append(reading)

        # Trim old history
        cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 3600)
        self._history = [
            r for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

    def get_trend_data(
        self,
        hours: int = 24,
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get trend data for air ingress indicators.

        Args:
            hours: Hours of history

        Returns:
            Dictionary of indicator trends
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)

        relevant = [
            r for r in self._history
            if r.timestamp.timestamp() > cutoff
        ]

        trends = {
            "dissolved_o2": [],
            "subcooling": [],
            "vacuum": [],
            "air_removal": [],
        }

        for r in sorted(relevant, key=lambda x: x.timestamp):
            if r.dissolved_o2_ppb is not None:
                trends["dissolved_o2"].append((r.timestamp, r.dissolved_o2_ppb))
            if r.subcooling_f is not None:
                trends["subcooling"].append((r.timestamp, r.subcooling_f))
            trends["vacuum"].append((r.timestamp, r.vacuum_inhga))
            if r.air_removal_scfm is not None:
                trends["air_removal"].append((r.timestamp, r.air_removal_scfm))

        return trends

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
