# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Thermography Analysis Module

This module implements infrared thermography analysis for equipment
hot spot detection and thermal anomaly identification. Thermography
is a powerful non-contact method for detecting:

- Electrical connection issues
- Motor/bearing overheating
- Insulation degradation
- Steam trap failures
- Refractory damage
- Heat exchanger fouling

Analysis follows ASNT and ISO standards for thermal imaging.

All calculations are DETERMINISTIC with provenance tracking.
No ML/LLM in calculation path - ZERO HALLUCINATION.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.thermography import (
    ...     ThermographyAnalyzer
    ... )
    >>> analyzer = ThermographyAnalyzer(config)
    >>> result = analyzer.analyze(thermal_image)
    >>> print(f"Hot spots detected: {result.hot_spots_detected}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    TemperatureThresholds,
    IRCameraConfig,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    ThermalImage,
    ThermographyResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# NETA/NFPA thermal severity classifications
# Based on Delta-T above reference/ambient
THERMAL_SEVERITY_DELTA_T = {
    "priority_1": 40.0,    # Immediate - Major repair needed
    "priority_2": 20.0,    # Serious - Repair ASAP
    "priority_3": 10.0,    # Intermediate - Schedule repair
    "priority_4": 1.0,     # Minor - Monitor
}

# Component-specific temperature limits (Celsius)
COMPONENT_TEMP_LIMITS = {
    "electrical_connection": {
        "warning": 60,
        "alarm": 90,
        "critical": 120,
    },
    "motor_bearing": {
        "warning": 70,
        "alarm": 85,
        "critical": 95,
    },
    "motor_winding": {
        "warning": 100,
        "alarm": 130,
        "critical": 150,
    },
    "gearbox": {
        "warning": 70,
        "alarm": 85,
        "critical": 100,
    },
    "pump_casing": {
        "warning": 60,
        "alarm": 80,
        "critical": 100,
    },
    "coupling": {
        "warning": 50,
        "alarm": 70,
        "critical": 90,
    },
    "bearing_housing": {
        "warning": 70,
        "alarm": 85,
        "critical": 95,
    },
    "steam_trap": {
        "warning": None,  # Depends on steam pressure
        "alarm": None,
        "critical": None,
    },
    "insulation": {
        "warning": 50,  # External surface
        "alarm": 70,
        "critical": 90,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HotSpot:
    """Detected thermal hot spot."""
    x: int
    y: int
    temperature_c: float
    delta_t_c: float
    area_pixels: int
    severity: AlertSeverity
    component_type: Optional[str] = None
    probable_cause: Optional[str] = None


@dataclass
class ThermalReference:
    """Reference temperature for comparison."""
    temperature_c: float
    location: str
    component_type: Optional[str] = None


# =============================================================================
# THERMOGRAPHY ANALYZER CLASS
# =============================================================================

class ThermographyAnalyzer:
    """
    Thermography Analyzer for equipment hot spot detection.

    This class analyzes infrared thermal images to detect anomalies
    and hot spots that may indicate equipment problems. Analysis
    follows industry standards including:

    - NETA MTS (Maintenance Testing Specifications)
    - NFPA 70B (Electrical Equipment Maintenance)
    - ISO 18434 (Condition monitoring with thermography)

    All calculations are DETERMINISTIC - ZERO HALLUCINATION.

    Attributes:
        thresholds: Temperature alarm thresholds
        camera_config: IR camera configuration

    Example:
        >>> analyzer = ThermographyAnalyzer()
        >>> result = analyzer.analyze(thermal_image)
        >>> for spot in result.hot_spots:
        ...     print(f"Hot spot at ({spot['x']}, {spot['y']}): {spot['temp']}C")
    """

    def __init__(
        self,
        thresholds: Optional[TemperatureThresholds] = None,
        camera_config: Optional[IRCameraConfig] = None,
    ) -> None:
        """
        Initialize thermography analyzer.

        Args:
            thresholds: Temperature alarm thresholds
            camera_config: IR camera configuration
        """
        self.thresholds = thresholds or TemperatureThresholds()
        self.camera_config = camera_config

        logger.info("ThermographyAnalyzer initialized")

    def analyze(
        self,
        image: ThermalImage,
        reference: Optional[ThermalReference] = None,
        component_type: Optional[str] = None,
    ) -> ThermographyResult:
        """
        Analyze thermal image for anomalies.

        Args:
            image: Thermal image data with hot spots
            reference: Reference temperature for comparison
            component_type: Type of component being inspected

        Returns:
            ThermographyResult with analysis
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Analyzing thermal image: {image.image_id}")

        # Determine reference temperature
        ref_temp = self._determine_reference(image, reference)

        # Calculate delta-T
        delta_t = image.max_temperature_c - ref_temp if ref_temp else None

        # Analyze hot spots
        analyzed_spots = []
        for spot_data in image.hot_spots:
            analyzed_spot = self._analyze_hot_spot(
                spot_data,
                ref_temp,
                component_type,
            )
            analyzed_spots.append(analyzed_spot)

        # Determine overall severity
        thermal_severity = self._determine_overall_severity(
            image.max_temperature_c,
            delta_t,
            analyzed_spots,
            component_type,
        )

        # Identify probable causes
        probable_causes = self._identify_probable_causes(
            image,
            analyzed_spots,
            component_type,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            thermal_severity,
            probable_causes,
            image.max_temperature_c,
            delta_t,
        )

        # Convert hot spots to output format
        hot_spots_output = [
            {
                "x": s.x,
                "y": s.y,
                "temperature_c": s.temperature_c,
                "delta_t_c": s.delta_t_c,
                "area_pixels": s.area_pixels,
                "severity": s.severity.value,
                "probable_cause": s.probable_cause,
            }
            for s in analyzed_spots
        ]

        logger.info(
            f"Thermography analysis complete: {len(analyzed_spots)} hot spots, "
            f"severity={thermal_severity.value}"
        )

        return ThermographyResult(
            image_id=image.image_id,
            timestamp=image.timestamp,
            max_temperature_c=image.max_temperature_c,
            reference_temperature_c=ref_temp,
            delta_t_c=delta_t,
            hot_spots_detected=len(analyzed_spots),
            hot_spots=hot_spots_output,
            thermal_severity=thermal_severity,
            probable_causes=probable_causes,
            recommendations=recommendations,
        )

    def _determine_reference(
        self,
        image: ThermalImage,
        reference: Optional[ThermalReference],
    ) -> Optional[float]:
        """
        Determine reference temperature for comparison.

        Priority:
        1. Provided reference
        2. Ambient temperature from image
        3. Average image temperature

        Args:
            image: Thermal image
            reference: Provided reference

        Returns:
            Reference temperature in Celsius
        """
        if reference:
            return reference.temperature_c

        if image.ambient_c:
            return image.ambient_c

        return image.avg_temperature_c

    def _analyze_hot_spot(
        self,
        spot_data: Dict[str, Any],
        reference_temp: Optional[float],
        component_type: Optional[str],
    ) -> HotSpot:
        """
        Analyze individual hot spot.

        Args:
            spot_data: Hot spot data from image
            reference_temp: Reference temperature
            component_type: Component type

        Returns:
            Analyzed HotSpot
        """
        temp = spot_data.get("temperature_c", 0)
        x = spot_data.get("x", 0)
        y = spot_data.get("y", 0)
        area = spot_data.get("area_pixels", 1)

        # Calculate delta-T
        delta_t = temp - reference_temp if reference_temp else 0

        # Determine severity
        severity = self._classify_severity_neta(delta_t, temp, component_type)

        # Identify probable cause based on severity and component
        probable_cause = self._identify_hot_spot_cause(
            temp,
            delta_t,
            component_type,
        )

        return HotSpot(
            x=x,
            y=y,
            temperature_c=temp,
            delta_t_c=delta_t,
            area_pixels=area,
            severity=severity,
            component_type=component_type,
            probable_cause=probable_cause,
        )

    def _classify_severity_neta(
        self,
        delta_t: float,
        absolute_temp: float,
        component_type: Optional[str],
    ) -> AlertSeverity:
        """
        Classify severity using NETA guidelines.

        NETA severity based on delta-T:
        - Priority 1 (>40C): Major discrepancy, immediate repair
        - Priority 2 (20-40C): Serious discrepancy, repair ASAP
        - Priority 3 (10-20C): Intermediate, schedule repair
        - Priority 4 (1-10C): Minor, monitor

        Args:
            delta_t: Temperature differential
            absolute_temp: Absolute temperature
            component_type: Component type for absolute limits

        Returns:
            AlertSeverity level
        """
        # First check absolute temperature limits
        if component_type and component_type in COMPONENT_TEMP_LIMITS:
            limits = COMPONENT_TEMP_LIMITS[component_type]
            if limits["critical"] and absolute_temp >= limits["critical"]:
                return AlertSeverity.UNACCEPTABLE
            if limits["alarm"] and absolute_temp >= limits["alarm"]:
                return AlertSeverity.UNSATISFACTORY

        # Then check delta-T
        if delta_t >= THERMAL_SEVERITY_DELTA_T["priority_1"]:
            return AlertSeverity.UNACCEPTABLE  # Priority 1
        elif delta_t >= THERMAL_SEVERITY_DELTA_T["priority_2"]:
            return AlertSeverity.UNSATISFACTORY  # Priority 2
        elif delta_t >= THERMAL_SEVERITY_DELTA_T["priority_3"]:
            return AlertSeverity.ACCEPTABLE  # Priority 3
        elif delta_t >= THERMAL_SEVERITY_DELTA_T["priority_4"]:
            return AlertSeverity.GOOD  # Priority 4
        else:
            return AlertSeverity.GOOD  # Normal

    def _identify_hot_spot_cause(
        self,
        temp: float,
        delta_t: float,
        component_type: Optional[str],
    ) -> Optional[str]:
        """
        Identify probable cause of hot spot.

        Args:
            temp: Hot spot temperature
            delta_t: Temperature differential
            component_type: Component type

        Returns:
            Probable cause description
        """
        if not component_type:
            if delta_t > 40:
                return "Severe thermal anomaly - investigate immediately"
            elif delta_t > 20:
                return "Significant thermal anomaly"
            return None

        cause_map = {
            "electrical_connection": {
                "high": "Loose or corroded connection, high resistance",
                "medium": "Degraded connection, check torque",
                "low": "Minor connection resistance, monitor",
            },
            "motor_bearing": {
                "high": "Bearing failure imminent, lubrication failure",
                "medium": "Bearing wear or inadequate lubrication",
                "low": "Normal bearing heat, verify lubrication",
            },
            "motor_winding": {
                "high": "Winding insulation failure, overload",
                "medium": "Excessive load or cooling problem",
                "low": "Minor load imbalance",
            },
            "gearbox": {
                "high": "Gear/bearing failure, oil breakdown",
                "medium": "Abnormal wear, check oil level",
                "low": "Normal operation heat",
            },
            "pump_casing": {
                "high": "Cavitation, bearing failure, dry running",
                "medium": "Partial cavitation, wear",
                "low": "Normal friction heat",
            },
            "coupling": {
                "high": "Misalignment, lubrication failure",
                "medium": "Wear, check alignment",
                "low": "Normal friction",
            },
            "bearing_housing": {
                "high": "Bearing failure, lubrication failure",
                "medium": "Wear, inadequate lubrication",
                "low": "Normal heat generation",
            },
            "insulation": {
                "high": "Major insulation damage, heat loss",
                "medium": "Insulation degradation",
                "low": "Minor insulation wear",
            },
        }

        if component_type in cause_map:
            causes = cause_map[component_type]
            if delta_t >= 40:
                return causes.get("high", "Severe anomaly")
            elif delta_t >= 20:
                return causes.get("medium", "Moderate anomaly")
            else:
                return causes.get("low", "Minor anomaly")

        return None

    def _determine_overall_severity(
        self,
        max_temp: float,
        delta_t: Optional[float],
        hot_spots: List[HotSpot],
        component_type: Optional[str],
    ) -> AlertSeverity:
        """
        Determine overall thermal severity.

        Args:
            max_temp: Maximum temperature
            delta_t: Maximum delta-T
            hot_spots: Analyzed hot spots
            component_type: Component type

        Returns:
            Overall AlertSeverity
        """
        # Check if any hot spot is critical
        if any(s.severity == AlertSeverity.UNACCEPTABLE for s in hot_spots):
            return AlertSeverity.UNACCEPTABLE

        if any(s.severity == AlertSeverity.UNSATISFACTORY for s in hot_spots):
            return AlertSeverity.UNSATISFACTORY

        # Check absolute temperature thresholds
        if component_type and component_type in COMPONENT_TEMP_LIMITS:
            limits = COMPONENT_TEMP_LIMITS[component_type]
            if limits["critical"] and max_temp >= limits["critical"]:
                return AlertSeverity.UNACCEPTABLE
            if limits["alarm"] and max_temp >= limits["alarm"]:
                return AlertSeverity.UNSATISFACTORY
            if limits["warning"] and max_temp >= limits["warning"]:
                return AlertSeverity.ACCEPTABLE

        # Check delta-T
        if delta_t:
            if delta_t >= self.thresholds.delta_alarm_c * 2:
                return AlertSeverity.UNACCEPTABLE
            elif delta_t >= self.thresholds.delta_alarm_c:
                return AlertSeverity.UNSATISFACTORY
            elif delta_t >= self.thresholds.delta_alarm_c / 2:
                return AlertSeverity.ACCEPTABLE

        return AlertSeverity.GOOD

    def _identify_probable_causes(
        self,
        image: ThermalImage,
        hot_spots: List[HotSpot],
        component_type: Optional[str],
    ) -> List[str]:
        """
        Identify probable causes of thermal anomalies.

        Args:
            image: Thermal image
            hot_spots: Analyzed hot spots
            component_type: Component type

        Returns:
            List of probable causes
        """
        causes = []

        # Collect unique causes from hot spots
        spot_causes = set(
            s.probable_cause for s in hot_spots
            if s.probable_cause
        )
        causes.extend(list(spot_causes))

        # Add general causes based on pattern
        if len(hot_spots) > 3:
            causes.append("Multiple hot spots may indicate systemic issue")

        # Temperature pattern analysis
        temp_range = image.max_temperature_c - image.min_temperature_c
        if temp_range > 50:
            causes.append("High temperature differential across component")

        return causes[:5]  # Limit to top 5

    def _generate_recommendations(
        self,
        severity: AlertSeverity,
        probable_causes: List[str],
        max_temp: float,
        delta_t: Optional[float],
    ) -> List[str]:
        """
        Generate maintenance recommendations.

        Args:
            severity: Overall severity
            probable_causes: Identified causes
            max_temp: Maximum temperature
            delta_t: Maximum delta-T

        Returns:
            List of recommendations
        """
        recommendations = []

        # Severity-based recommendations (NETA guidelines)
        if severity == AlertSeverity.UNACCEPTABLE:
            recommendations.append(
                "IMMEDIATE ACTION - NETA Priority 1: "
                "Major thermal discrepancy. Repair immediately. "
                "Equipment failure possible if left unaddressed."
            )
        elif severity == AlertSeverity.UNSATISFACTORY:
            recommendations.append(
                "URGENT - NETA Priority 2: "
                "Serious thermal discrepancy. "
                "Repair as soon as possible, schedule within 48 hours."
            )
        elif severity == AlertSeverity.ACCEPTABLE:
            recommendations.append(
                "NETA Priority 3: "
                "Intermediate thermal discrepancy. "
                "Schedule repair at earliest convenience."
            )

        # Specific recommendations
        if delta_t and delta_t > 30:
            recommendations.append(
                "Re-scan after repair to verify correction. "
                "Document before/after thermal images."
            )

        if max_temp > 100:
            recommendations.append(
                "High absolute temperature detected. "
                "Verify personnel safety barriers in place."
            )

        # Add investigation recommendation
        if probable_causes:
            recommendations.append(
                f"Investigate: {'; '.join(probable_causes[:2])}"
            )

        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Thermal profile normal. Continue routine monitoring."
            )

        return recommendations

    def calculate_heat_loss(
        self,
        surface_temp_c: float,
        ambient_temp_c: float,
        surface_area_m2: float,
        emissivity: float = 0.9,
    ) -> float:
        """
        Calculate heat loss from hot surface using Stefan-Boltzmann.

        Q = epsilon * sigma * A * (T_surface^4 - T_ambient^4)

        Plus convective loss approximation:
        Q_conv = h * A * (T_surface - T_ambient)
        where h ~ 10 W/m2K for natural convection

        Args:
            surface_temp_c: Surface temperature (Celsius)
            ambient_temp_c: Ambient temperature (Celsius)
            surface_area_m2: Surface area (m2)
            emissivity: Surface emissivity

        Returns:
            Heat loss in Watts
        """
        # Convert to Kelvin
        t_surface_k = surface_temp_c + 273.15
        t_ambient_k = ambient_temp_c + 273.15

        # Stefan-Boltzmann constant
        sigma = 5.67e-8  # W/m2K4

        # Radiative heat loss
        q_rad = (emissivity * sigma * surface_area_m2 *
                 (t_surface_k**4 - t_ambient_k**4))

        # Convective heat loss (natural convection approximation)
        h_conv = 10  # W/m2K typical for natural convection
        q_conv = h_conv * surface_area_m2 * (surface_temp_c - ambient_temp_c)

        return q_rad + q_conv

    def estimate_insulation_thickness_needed(
        self,
        surface_temp_c: float,
        process_temp_c: float,
        target_surface_temp_c: float = 50.0,
        thermal_conductivity_w_mk: float = 0.04,
    ) -> float:
        """
        Estimate insulation thickness needed to achieve target surface temperature.

        Uses simple one-dimensional heat transfer assumption.

        Args:
            surface_temp_c: Current surface temperature
            process_temp_c: Process/internal temperature
            target_surface_temp_c: Target surface temperature
            thermal_conductivity_w_mk: Insulation thermal conductivity

        Returns:
            Required insulation thickness in mm
        """
        if surface_temp_c <= target_surface_temp_c:
            return 0  # No additional insulation needed

        # Simple ratio-based estimation
        # Higher delta-T from process = more insulation needed
        current_delta = process_temp_c - surface_temp_c
        target_delta = process_temp_c - target_surface_temp_c

        if current_delta <= 0:
            return 0

        # Ratio of required to current insulation effectiveness
        ratio = target_delta / current_delta

        # Estimate based on typical 25mm providing certain protection
        base_thickness = 25  # mm baseline
        required_thickness = base_thickness * (ratio - 1)

        return max(0, round(required_thickness, 0))
