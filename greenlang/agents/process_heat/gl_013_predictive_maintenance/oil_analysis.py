# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Oil Analysis Module

This module implements oil analysis interpretation for equipment health
assessment. Oil analysis is a key predictive maintenance technique that
provides insight into lubricant condition and machine wear.

Key parameters analyzed:
- Viscosity: Lubricant film thickness and flow properties
- Total Acid Number (TAN): Oil degradation indicator
- Particle count (ISO 4406): Contamination level
- Wear metals: Component wear indicators
- Water content: Contamination from leaks or condensation

All calculations are DETERMINISTIC with provenance tracking.
No ML/LLM in calculation path - ZERO HALLUCINATION.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.oil_analysis import (
    ...     OilAnalyzer
    ... )
    >>> analyzer = OilAnalyzer(config)
    >>> result = analyzer.analyze(oil_reading, baseline)
    >>> print(f"Oil condition: {result.oil_condition}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    OilThresholds,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    HealthStatus,
    OilAnalysisReading,
    OilAnalysisResult,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OilBaseline:
    """Baseline oil properties for comparison."""
    viscosity_40c_cst: float
    viscosity_100c_cst: Optional[float] = None
    tan_mg_koh_g: float = 0.3
    tbn_mg_koh_g: Optional[float] = None
    iron_ppm: float = 0.0
    copper_ppm: float = 0.0
    chromium_ppm: float = 0.0
    silicon_ppm: float = 0.0
    water_ppm: float = 50.0
    particle_count_iso_4406: str = "16/14/11"


@dataclass
class WearMetalTrend:
    """Trend data for wear metal analysis."""
    metal: str
    current_ppm: float
    previous_ppm: float
    baseline_ppm: float
    rate_ppm_per_1000h: float
    trend: TrendDirection


# =============================================================================
# OIL ANALYZER CLASS
# =============================================================================

class OilAnalyzer:
    """
    Oil Analysis Interpreter for equipment health assessment.

    This class interprets oil analysis results to assess lubricant
    condition and detect machine wear patterns. Analysis follows
    industry standards including:
    - ISO 4406 for particle counting
    - ASTM D445 for viscosity
    - ASTM D664/D974 for acid number

    All calculations are DETERMINISTIC - ZERO HALLUCINATION.

    Attributes:
        thresholds: Oil analysis alarm thresholds
        baseline: Baseline oil properties

    Example:
        >>> thresholds = OilThresholds()
        >>> analyzer = OilAnalyzer(thresholds)
        >>> result = analyzer.analyze(current_sample, baseline)
        >>> print(f"Condition: {result.oil_condition}")
    """

    # Wear metal source mapping (typical industrial equipment)
    WEAR_METAL_SOURCES = {
        "iron": ["gears", "cylinders", "bearings", "shafts", "valve train"],
        "copper": ["bushings", "thrust washers", "oil cooler", "bronze components"],
        "chromium": ["rings", "liners", "bearings", "stainless steel"],
        "aluminum": ["pistons", "bearings", "pumps", "bushings"],
        "lead": ["bearings", "solder joints"],
        "tin": ["bearings", "bronze alloys", "piston overlay"],
        "nickel": ["valves", "alloy steels", "exhaust components"],
        "silver": ["silver-plated bearings", "wrist pin bushings"],
    }

    # Contaminant sources
    CONTAMINANT_SOURCES = {
        "silicon": ["dirt ingress", "seal failure", "additive"],
        "sodium": ["coolant leak", "salt water ingress"],
        "potassium": ["coolant leak"],
        "boron": ["coolant leak"],
    }

    def __init__(
        self,
        thresholds: Optional[OilThresholds] = None,
        baseline: Optional[OilBaseline] = None,
    ) -> None:
        """
        Initialize oil analyzer.

        Args:
            thresholds: Oil analysis alarm thresholds
            baseline: Baseline oil properties for comparison
        """
        self.thresholds = thresholds or OilThresholds()
        self.baseline = baseline

        logger.info("OilAnalyzer initialized")

    def analyze(
        self,
        reading: OilAnalysisReading,
        baseline: Optional[OilBaseline] = None,
        history: Optional[List[OilAnalysisReading]] = None,
    ) -> OilAnalysisResult:
        """
        Analyze oil sample and generate interpretation.

        Args:
            reading: Current oil analysis reading
            baseline: Baseline properties (overrides instance baseline)
            history: Historical readings for trend analysis

        Returns:
            OilAnalysisResult with interpretation

        Raises:
            ValueError: If reading is invalid
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Analyzing oil sample: {reading.sample_id}")

        # Use provided baseline or instance baseline
        base = baseline or self.baseline
        if base is None:
            # Create default baseline from reading (assuming new oil)
            base = OilBaseline(
                viscosity_40c_cst=reading.viscosity_40c_cst,
                tan_mg_koh_g=0.3,  # Typical new oil
            )

        # Analyze viscosity
        viscosity_status, viscosity_change = self._analyze_viscosity(reading, base)

        # Analyze acid number
        tan_status = self._analyze_tan(reading)

        # Analyze water contamination
        water_status = self._analyze_water(reading)

        # Analyze particle count
        particle_status = self._analyze_particles(reading)

        # Analyze wear metals
        wear_trend, primary_metal, wear_source = self._analyze_wear_metals(
            reading, base, history
        )

        # Determine overall oil condition
        oil_condition = self._determine_overall_condition(
            viscosity_status,
            tan_status,
            water_status,
            particle_status,
            wear_trend,
        )

        # Estimate remaining useful life
        rul_pct = self._estimate_remaining_life(reading, base)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            reading,
            base,
            oil_condition,
            viscosity_status,
            tan_status,
            water_status,
            particle_status,
            wear_trend,
            primary_metal,
        )

        # Determine action flags
        oil_change_recommended = (
            oil_condition in [HealthStatus.CRITICAL, HealthStatus.FAILED] or
            tan_status == AlertSeverity.UNACCEPTABLE or
            (rul_pct is not None and rul_pct < 10)
        )

        filtration_recommended = (
            particle_status in [AlertSeverity.UNSATISFACTORY, AlertSeverity.UNACCEPTABLE] or
            water_status in [AlertSeverity.UNSATISFACTORY, AlertSeverity.UNACCEPTABLE]
        )

        investigation_needed = (
            wear_trend == TrendDirection.INCREASING or
            (primary_metal is not None and
             reading.__dict__.get(f"{primary_metal}_ppm", 0) >
             self.thresholds.__dict__.get(f"{primary_metal}_critical_ppm", float('inf')))
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(reading, oil_condition)

        logger.info(
            f"Oil analysis complete: condition={oil_condition.value}, "
            f"viscosity={viscosity_status}, TAN={tan_status.value}"
        )

        return OilAnalysisResult(
            sample_id=reading.sample_id,
            timestamp=reading.timestamp,
            oil_condition=oil_condition,
            remaining_useful_life_pct=rul_pct,
            viscosity_status=viscosity_status,
            viscosity_change_pct=viscosity_change,
            tan_status=tan_status,
            water_status=water_status,
            particle_status=particle_status,
            wear_trend=wear_trend,
            primary_wear_metal=primary_metal,
            wear_source_probable=wear_source,
            oil_change_recommended=oil_change_recommended,
            filtration_recommended=filtration_recommended,
            investigation_needed=investigation_needed,
            recommendations=recommendations,
        )

    def _analyze_viscosity(
        self,
        reading: OilAnalysisReading,
        baseline: OilBaseline,
    ) -> Tuple[str, float]:
        """
        Analyze viscosity change from baseline.

        Viscosity changes indicate:
        - Decrease: Fuel dilution, wrong oil, shear
        - Increase: Oxidation, contamination, wrong oil

        Args:
            reading: Current reading
            baseline: Baseline properties

        Returns:
            Tuple of (status, change_percentage)
        """
        change_pct = (
            (reading.viscosity_40c_cst - baseline.viscosity_40c_cst)
            / baseline.viscosity_40c_cst * 100
        )

        threshold = self.thresholds.viscosity_change_pct

        if abs(change_pct) <= threshold / 2:
            status = "normal"
        elif change_pct < -threshold:
            status = "low"
        elif change_pct > threshold:
            status = "high"
        else:
            status = "marginal"

        return status, round(change_pct, 1)

    def _analyze_tan(self, reading: OilAnalysisReading) -> AlertSeverity:
        """
        Analyze Total Acid Number.

        Higher TAN indicates oil degradation from oxidation.

        Args:
            reading: Current reading

        Returns:
            AlertSeverity level
        """
        tan = reading.tan_mg_koh_g

        if tan <= self.thresholds.tan_warning_mg_koh_g / 2:
            return AlertSeverity.GOOD
        elif tan <= self.thresholds.tan_warning_mg_koh_g:
            return AlertSeverity.ACCEPTABLE
        elif tan <= self.thresholds.tan_critical_mg_koh_g:
            return AlertSeverity.UNSATISFACTORY
        else:
            return AlertSeverity.UNACCEPTABLE

    def _analyze_water(self, reading: OilAnalysisReading) -> AlertSeverity:
        """
        Analyze water contamination.

        Water in oil causes:
        - Reduced lubrication
        - Corrosion
        - Additive depletion
        - Microbial growth

        Args:
            reading: Current reading

        Returns:
            AlertSeverity level
        """
        water = reading.water_ppm

        if water <= self.thresholds.water_warning_ppm / 2:
            return AlertSeverity.GOOD
        elif water <= self.thresholds.water_warning_ppm:
            return AlertSeverity.ACCEPTABLE
        elif water <= self.thresholds.water_critical_ppm:
            return AlertSeverity.UNSATISFACTORY
        else:
            return AlertSeverity.UNACCEPTABLE

    def _analyze_particles(self, reading: OilAnalysisReading) -> AlertSeverity:
        """
        Analyze particle count per ISO 4406.

        ISO 4406 code format: X/Y/Z where:
        - X: Particles > 4um per mL
        - Y: Particles > 6um per mL
        - Z: Particles > 14um per mL

        Args:
            reading: Current reading

        Returns:
            AlertSeverity level
        """
        if not reading.particle_count_iso_4406:
            return AlertSeverity.ACCEPTABLE  # No data

        try:
            current = self._parse_iso_4406(reading.particle_count_iso_4406)
            warning = self._parse_iso_4406(self.thresholds.particle_count_warning)
            critical = self._parse_iso_4406(self.thresholds.particle_count_critical)

            # Compare using sum of codes (simplified)
            current_sum = sum(current)
            warning_sum = sum(warning)
            critical_sum = sum(critical)

            if current_sum <= warning_sum - 6:
                return AlertSeverity.GOOD
            elif current_sum <= warning_sum:
                return AlertSeverity.ACCEPTABLE
            elif current_sum <= critical_sum:
                return AlertSeverity.UNSATISFACTORY
            else:
                return AlertSeverity.UNACCEPTABLE

        except ValueError:
            logger.warning(
                f"Invalid ISO 4406 code: {reading.particle_count_iso_4406}"
            )
            return AlertSeverity.ACCEPTABLE

    def _parse_iso_4406(self, code: str) -> Tuple[int, int, int]:
        """
        Parse ISO 4406 cleanliness code.

        Args:
            code: ISO 4406 code string (e.g., "18/16/13")

        Returns:
            Tuple of three integer codes
        """
        parts = code.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid ISO 4406 format: {code}")
        return tuple(int(p) for p in parts)

    def _analyze_wear_metals(
        self,
        reading: OilAnalysisReading,
        baseline: OilBaseline,
        history: Optional[List[OilAnalysisReading]],
    ) -> Tuple[TrendDirection, Optional[str], Optional[str]]:
        """
        Analyze wear metal content and trends.

        Args:
            reading: Current reading
            baseline: Baseline properties
            history: Historical readings

        Returns:
            Tuple of (trend_direction, primary_metal, probable_source)
        """
        # Track metals and their changes
        metals = [
            ("iron", reading.iron_ppm, baseline.iron_ppm,
             self.thresholds.iron_warning_ppm),
            ("copper", reading.copper_ppm, baseline.copper_ppm,
             self.thresholds.copper_warning_ppm),
            ("chromium", reading.chromium_ppm, baseline.chromium_ppm,
             self.thresholds.chromium_warning_ppm),
        ]

        # Find most significant metal
        max_ratio = 0
        primary_metal = None
        overall_trend = TrendDirection.STABLE

        for metal, current, baseline_val, warning in metals:
            if warning > 0:
                ratio = current / warning
                if ratio > max_ratio:
                    max_ratio = ratio
                    primary_metal = metal

        # Determine trend from history
        if history and len(history) >= 2 and primary_metal:
            metal_attr = f"{primary_metal}_ppm"
            recent_values = [
                getattr(h, metal_attr, 0)
                for h in sorted(history, key=lambda x: x.timestamp)[-5:]
            ]

            if len(recent_values) >= 2:
                if recent_values[-1] > recent_values[0] * 1.2:
                    overall_trend = TrendDirection.INCREASING
                elif recent_values[-1] < recent_values[0] * 0.8:
                    overall_trend = TrendDirection.DECREASING

        # Determine probable source
        probable_source = None
        if primary_metal and max_ratio > 0.5:
            sources = self.WEAR_METAL_SOURCES.get(primary_metal, [])
            if sources:
                probable_source = sources[0]  # Most common source

        return overall_trend, primary_metal, probable_source

    def _determine_overall_condition(
        self,
        viscosity_status: str,
        tan_status: AlertSeverity,
        water_status: AlertSeverity,
        particle_status: AlertSeverity,
        wear_trend: TrendDirection,
    ) -> HealthStatus:
        """
        Determine overall oil condition.

        Args:
            viscosity_status: Viscosity assessment
            tan_status: TAN severity
            water_status: Water contamination severity
            particle_status: Particle count severity
            wear_trend: Wear metal trend

        Returns:
            Overall HealthStatus
        """
        # Count critical conditions
        severities = [tan_status, water_status, particle_status]
        critical_count = sum(
            1 for s in severities
            if s == AlertSeverity.UNACCEPTABLE
        )
        warning_count = sum(
            1 for s in severities
            if s == AlertSeverity.UNSATISFACTORY
        )

        # Determine overall status
        if critical_count >= 2 or (critical_count == 1 and warning_count >= 1):
            return HealthStatus.FAILED
        elif critical_count == 1:
            return HealthStatus.CRITICAL
        elif warning_count >= 2:
            return HealthStatus.WARNING
        elif (warning_count == 1 or
              viscosity_status in ["low", "high"] or
              wear_trend == TrendDirection.INCREASING):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _estimate_remaining_life(
        self,
        reading: OilAnalysisReading,
        baseline: OilBaseline,
    ) -> Optional[float]:
        """
        Estimate remaining useful life percentage.

        Based on TAN, viscosity change, and contamination levels.

        Args:
            reading: Current reading
            baseline: Baseline properties

        Returns:
            Estimated RUL percentage (0-100)
        """
        # TAN-based estimate (major factor)
        tan_limit = self.thresholds.tan_critical_mg_koh_g
        tan_used = min(reading.tan_mg_koh_g / tan_limit * 100, 100)
        tan_rul = max(0, 100 - tan_used)

        # Viscosity-based estimate
        visc_change = abs(
            (reading.viscosity_40c_cst - baseline.viscosity_40c_cst)
            / baseline.viscosity_40c_cst * 100
        )
        visc_limit = self.thresholds.viscosity_change_pct * 2  # Allow 2x threshold
        visc_used = min(visc_change / visc_limit * 100, 100)
        visc_rul = max(0, 100 - visc_used)

        # Water-based estimate
        water_limit = self.thresholds.water_critical_ppm
        water_used = min(reading.water_ppm / water_limit * 100, 100)
        water_rul = max(0, 100 - water_used)

        # Weighted average (TAN is most important)
        rul = 0.5 * tan_rul + 0.3 * visc_rul + 0.2 * water_rul

        return round(rul, 1)

    def _generate_recommendations(
        self,
        reading: OilAnalysisReading,
        baseline: OilBaseline,
        oil_condition: HealthStatus,
        viscosity_status: str,
        tan_status: AlertSeverity,
        water_status: AlertSeverity,
        particle_status: AlertSeverity,
        wear_trend: TrendDirection,
        primary_metal: Optional[str],
    ) -> List[str]:
        """
        Generate maintenance recommendations.

        Args:
            Multiple analysis parameters

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Oil change recommendations
        if oil_condition == HealthStatus.FAILED:
            recommendations.append(
                "IMMEDIATE OIL CHANGE REQUIRED - Oil has exceeded service limits"
            )
        elif oil_condition == HealthStatus.CRITICAL:
            recommendations.append(
                "Schedule oil change within 48 hours - Oil condition critical"
            )
        elif oil_condition == HealthStatus.WARNING:
            recommendations.append(
                "Plan oil change at next scheduled maintenance - Oil degraded"
            )

        # Viscosity recommendations
        if viscosity_status == "low":
            recommendations.append(
                "Investigate cause of viscosity decrease: "
                "Check for fuel dilution, wrong oil grade, or excessive shear"
            )
        elif viscosity_status == "high":
            recommendations.append(
                "Investigate cause of viscosity increase: "
                "Check for oxidation, contamination, or wrong oil addition"
            )

        # TAN recommendations
        if tan_status == AlertSeverity.UNACCEPTABLE:
            recommendations.append(
                "TAN exceeds limit - Oil is severely oxidized. "
                "Investigate operating temperature and aeration"
            )
        elif tan_status == AlertSeverity.UNSATISFACTORY:
            recommendations.append(
                "TAN elevated - Monitor oil oxidation. "
                "Consider reducing drain interval"
            )

        # Water recommendations
        if water_status == AlertSeverity.UNACCEPTABLE:
            recommendations.append(
                "CRITICAL water contamination - "
                "Identify and repair leak source immediately. "
                "Consider vacuum dehydration"
            )
        elif water_status == AlertSeverity.UNSATISFACTORY:
            recommendations.append(
                "Elevated water content - "
                "Check seals, breathers, and cooler integrity. "
                "Monitor for increasing trend"
            )

        # Particle recommendations
        if particle_status == AlertSeverity.UNACCEPTABLE:
            recommendations.append(
                "CRITICAL particle contamination - "
                "Flush system and replace filters immediately. "
                "Investigate contamination source"
            )
        elif particle_status == AlertSeverity.UNSATISFACTORY:
            recommendations.append(
                "Elevated particle count - "
                "Replace filters and consider enhanced filtration"
            )

        # Wear metal recommendations
        if wear_trend == TrendDirection.INCREASING:
            source_msg = (
                f" ({self.WEAR_METAL_SOURCES.get(primary_metal, ['unknown'])[0]})"
                if primary_metal else ""
            )
            recommendations.append(
                f"Increasing {primary_metal or 'wear metal'} trend detected{source_msg}. "
                "Schedule inspection of wear components"
            )

        # If no specific issues
        if not recommendations:
            recommendations.append(
                "Oil condition satisfactory - Continue current monitoring interval"
            )

        return recommendations

    def _calculate_provenance(
        self,
        reading: OilAnalysisReading,
        condition: HealthStatus,
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            reading: Oil analysis reading
            condition: Determined condition

        Returns:
            SHA-256 hash string
        """
        provenance_str = (
            f"oil_analysis|{reading.sample_id}|{reading.timestamp.isoformat()}|"
            f"{reading.viscosity_40c_cst:.4f}|{reading.tan_mg_koh_g:.4f}|"
            f"{reading.iron_ppm:.2f}|{reading.water_ppm:.2f}|"
            f"{condition.value}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_oil_change_interval(
        self,
        current_sample: OilAnalysisReading,
        previous_sample: OilAnalysisReading,
        hours_between_samples: float,
    ) -> Optional[float]:
        """
        Calculate optimal oil change interval based on degradation rate.

        Args:
            current_sample: Current oil sample
            previous_sample: Previous oil sample
            hours_between_samples: Operating hours between samples

        Returns:
            Recommended oil change interval in hours
        """
        if hours_between_samples <= 0:
            return None

        # Calculate TAN degradation rate
        tan_rate = (
            (current_sample.tan_mg_koh_g - previous_sample.tan_mg_koh_g)
            / hours_between_samples
        )

        if tan_rate <= 0:
            return None  # No degradation observed

        # Calculate hours to reach critical TAN
        tan_remaining = (
            self.thresholds.tan_critical_mg_koh_g -
            current_sample.tan_mg_koh_g
        )

        if tan_remaining <= 0:
            return 0  # Already exceeded

        hours_to_limit = tan_remaining / tan_rate

        # Apply safety factor (change at 80% of limit)
        recommended_interval = hours_to_limit * 0.8

        return round(recommended_interval, 0)
