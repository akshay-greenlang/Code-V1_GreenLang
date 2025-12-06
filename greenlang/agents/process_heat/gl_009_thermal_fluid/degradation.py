"""
GL-009 THERMALIQ Agent - Thermal Fluid Degradation Monitoring

This module provides comprehensive degradation monitoring for thermal fluids,
analyzing laboratory test results to assess fluid condition and predict
remaining useful life.

Monitored parameters:
    - Viscosity changes (thermal cracking/polymerization indicators)
    - Thermal conductivity degradation
    - Flash point reduction (low boiler formation)
    - Total acid number (oxidation indicator)
    - Carbon residue (high boiler/coke formation)
    - Moisture content
    - Low boilers (volatile decomposition products)
    - High boilers (polymeric degradation products)

All calculations are deterministic - ZERO HALLUCINATION guaranteed.

Reference:
    - ASTM D6743: Thermal Stability of Organic Heat Transfer Fluids
    - Manufacturer fluid analysis guidelines (Eastman, Dow)

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.degradation import (
    ...     DegradationMonitor,
    ... )
    >>> monitor = DegradationMonitor(fluid_type=ThermalFluidType.THERMINOL_66)
    >>> result = monitor.analyze(lab_analysis)
    >>> print(f"Degradation level: {result.degradation_level}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pydantic import BaseModel, Field

from .schemas import (
    ThermalFluidType,
    DegradationLevel,
    DegradationAnalysis,
    FluidLabAnalysis,
    ValidationStatus,
)
from .config import DegradationThresholds
from .fluid_properties import ThermalFluidPropertyDatabase

logger = logging.getLogger(__name__)


# =============================================================================
# DEGRADATION LIMITS BY FLUID TYPE
# =============================================================================

@dataclass
class FluidDegradationLimits:
    """Degradation limits for a specific fluid type."""

    # Viscosity limits (% change from new)
    viscosity_warning_pct: float = 10.0
    viscosity_critical_pct: float = 25.0

    # Flash point limits (F drop from new)
    flash_point_warning_drop_f: float = 30.0
    flash_point_critical_drop_f: float = 50.0

    # Baseline flash point for comparison
    new_flash_point_f: float = 340.0

    # Acid number limits (mg KOH/g)
    acid_number_warning: float = 0.2
    acid_number_critical: float = 0.5
    acid_number_new: float = 0.01

    # Carbon residue limits (%)
    carbon_residue_warning_pct: float = 0.5
    carbon_residue_critical_pct: float = 1.0
    carbon_residue_new_pct: float = 0.01

    # Moisture limits (ppm)
    moisture_warning_ppm: float = 500.0
    moisture_critical_ppm: float = 1000.0

    # Low boilers limits (%)
    low_boilers_warning_pct: float = 3.0
    low_boilers_critical_pct: float = 10.0
    low_boilers_new_pct: float = 0.0

    # High boilers limits (%)
    high_boilers_warning_pct: float = 5.0
    high_boilers_critical_pct: float = 15.0
    high_boilers_new_pct: float = 0.0

    # Thermal conductivity limits (% change)
    thermal_conductivity_warning_pct: float = 5.0
    thermal_conductivity_critical_pct: float = 10.0

    # Reference viscosity at 100F (cSt)
    viscosity_new_100f: float = 30.0


# Fluid-specific degradation limits
FLUID_DEGRADATION_LIMITS: Dict[ThermalFluidType, FluidDegradationLimits] = {
    ThermalFluidType.THERMINOL_66: FluidDegradationLimits(
        new_flash_point_f=340.0,
        viscosity_new_100f=29.0,
        acid_number_new=0.01,
    ),
    ThermalFluidType.THERMINOL_VP1: FluidDegradationLimits(
        new_flash_point_f=255.0,
        viscosity_new_100f=3.5,
        acid_number_new=0.02,
        flash_point_warning_drop_f=25.0,  # More sensitive for VP1
        flash_point_critical_drop_f=40.0,
    ),
    ThermalFluidType.DOWTHERM_A: FluidDegradationLimits(
        new_flash_point_f=255.0,
        viscosity_new_100f=3.7,
        acid_number_new=0.02,
        flash_point_warning_drop_f=25.0,
        flash_point_critical_drop_f=40.0,
    ),
    ThermalFluidType.DOWTHERM_G: FluidDegradationLimits(
        new_flash_point_f=350.0,
        viscosity_new_100f=12.0,
        acid_number_new=0.01,
    ),
    ThermalFluidType.MARLOTHERM_SH: FluidDegradationLimits(
        new_flash_point_f=365.0,
        viscosity_new_100f=24.0,
        acid_number_new=0.01,
    ),
    ThermalFluidType.SYLTHERM_800: FluidDegradationLimits(
        new_flash_point_f=350.0,
        viscosity_new_100f=9.0,
        acid_number_new=0.02,
        # Silicones are more thermally stable
        viscosity_warning_pct=15.0,
        viscosity_critical_pct=30.0,
    ),
}


# =============================================================================
# DEGRADATION MONITOR
# =============================================================================

class DegradationMonitor:
    """
    Thermal fluid degradation monitoring and analysis.

    This class analyzes laboratory test results to assess fluid condition,
    identify degradation mechanisms, and predict remaining useful life.
    All calculations are deterministic with no ML/LLM in the analysis path.

    Degradation mechanisms monitored:
        - Thermal cracking (low boilers, viscosity decrease)
        - Oxidation (acid number, viscosity increase)
        - Polymerization (high boilers, viscosity increase)
        - Contamination (moisture, particulates)

    Example:
        >>> monitor = DegradationMonitor(
        ...     fluid_type=ThermalFluidType.THERMINOL_66,
        ...     thresholds=DegradationThresholds(),
        ... )
        >>> result = monitor.analyze(lab_analysis)
        >>> if result.replacement_recommended:
        ...     print("Replace fluid!")
    """

    def __init__(
        self,
        fluid_type: ThermalFluidType,
        thresholds: Optional[DegradationThresholds] = None,
    ) -> None:
        """
        Initialize the degradation monitor.

        Args:
            fluid_type: Type of thermal fluid
            thresholds: Custom degradation thresholds (uses defaults if None)
        """
        self.fluid_type = fluid_type
        self.thresholds = thresholds or DegradationThresholds()

        # Get fluid-specific limits
        self.limits = FLUID_DEGRADATION_LIMITS.get(
            fluid_type,
            FluidDegradationLimits()
        )

        # Property database for reference values
        self._property_db = ThermalFluidPropertyDatabase()

        # Historical analysis storage
        self._analysis_history: List[Tuple[datetime, DegradationAnalysis]] = []

        self._calculation_count = 0

        logger.info(f"DegradationMonitor initialized for {fluid_type}")

    def analyze(
        self,
        lab_analysis: FluidLabAnalysis,
        baseline_analysis: Optional[FluidLabAnalysis] = None,
    ) -> DegradationAnalysis:
        """
        Analyze fluid degradation from laboratory results.

        Args:
            lab_analysis: Current laboratory analysis results
            baseline_analysis: Optional baseline (new fluid) analysis

        Returns:
            DegradationAnalysis with condition assessment
        """
        self._calculation_count += 1

        findings: List[str] = []
        recommendations: List[str] = []

        # Analyze each parameter
        viscosity_result = self._analyze_viscosity(lab_analysis, baseline_analysis)
        flash_point_result = self._analyze_flash_point(lab_analysis, baseline_analysis)
        acid_number_result = self._analyze_acid_number(lab_analysis)
        carbon_residue_result = self._analyze_carbon_residue(lab_analysis)
        moisture_result = self._analyze_moisture(lab_analysis)
        low_boilers_result = self._analyze_low_boilers(lab_analysis)
        high_boilers_result = self._analyze_high_boilers(lab_analysis)
        thermal_cond_result = self._analyze_thermal_conductivity(lab_analysis)

        # Collect findings
        findings.extend(viscosity_result["findings"])
        findings.extend(flash_point_result["findings"])
        findings.extend(acid_number_result["findings"])
        findings.extend(carbon_residue_result["findings"])
        findings.extend(moisture_result["findings"])
        findings.extend(low_boilers_result["findings"])
        findings.extend(high_boilers_result["findings"])
        findings.extend(thermal_cond_result["findings"])

        # Collect recommendations
        recommendations.extend(viscosity_result["recommendations"])
        recommendations.extend(flash_point_result["recommendations"])
        recommendations.extend(acid_number_result["recommendations"])
        recommendations.extend(carbon_residue_result["recommendations"])
        recommendations.extend(moisture_result["recommendations"])
        recommendations.extend(low_boilers_result["recommendations"])
        recommendations.extend(high_boilers_result["recommendations"])
        recommendations.extend(thermal_cond_result["recommendations"])

        # Calculate composite degradation score (0-100)
        degradation_score = self._calculate_degradation_score(
            viscosity_result,
            flash_point_result,
            acid_number_result,
            carbon_residue_result,
            moisture_result,
            low_boilers_result,
            high_boilers_result,
        )

        # Determine degradation level
        degradation_level = self._determine_degradation_level(degradation_score)

        # Calculate remaining life estimate
        remaining_life_pct = max(0.0, 100.0 - degradation_score)

        # Determine if replacement recommended
        replacement_recommended = (
            degradation_level in [DegradationLevel.POOR, DegradationLevel.CRITICAL] or
            degradation_score >= 80
        )

        # Determine next sample date
        next_sample_date = self._calculate_next_sample_date(
            degradation_level,
            lab_analysis.sample_date,
        )

        # Create result
        result = DegradationAnalysis(
            degradation_level=degradation_level,
            remaining_life_pct=round(remaining_life_pct, 1),
            replacement_recommended=replacement_recommended,
            viscosity_status=viscosity_result["status"],
            thermal_conductivity_status=thermal_cond_result["status"],
            flash_point_status=flash_point_result["status"],
            acid_number_status=acid_number_result["status"],
            carbon_residue_status=carbon_residue_result["status"],
            moisture_status=moisture_result["status"],
            low_boilers_status=low_boilers_result["status"],
            high_boilers_status=high_boilers_result["status"],
            degradation_score=round(degradation_score, 1),
            findings=findings,
            recommendations=recommendations,
            next_sample_date=next_sample_date,
        )

        # Store in history
        self._analysis_history.append((lab_analysis.sample_date, result))

        return result

    def _analyze_viscosity(
        self,
        lab: FluidLabAnalysis,
        baseline: Optional[FluidLabAnalysis],
    ) -> Dict[str, Any]:
        """Analyze viscosity changes."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.viscosity_cst_100f is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        # Determine baseline
        if baseline and baseline.viscosity_cst_100f:
            baseline_visc = baseline.viscosity_cst_100f
        else:
            baseline_visc = self.limits.viscosity_new_100f

        # Calculate change
        change_pct = abs(lab.viscosity_cst_100f - baseline_visc) / baseline_visc * 100

        if change_pct >= self.limits.viscosity_critical_pct:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Viscosity changed {change_pct:.1f}% from baseline "
                f"(current: {lab.viscosity_cst_100f:.1f} cSt, baseline: {baseline_visc:.1f} cSt)"
            )
            recommendations.append("Fluid replacement required due to viscosity degradation")

        elif change_pct >= self.limits.viscosity_warning_pct:
            status = ValidationStatus.WARNING
            score = change_pct / self.limits.viscosity_critical_pct * 100
            findings.append(
                f"WARNING: Viscosity changed {change_pct:.1f}% from baseline"
            )
            recommendations.append("Monitor viscosity trend closely")

        else:
            score = change_pct / self.limits.viscosity_warning_pct * 50

        # Determine degradation mechanism
        if lab.viscosity_cst_100f < baseline_visc * 0.9:
            findings.append("Viscosity decrease indicates thermal cracking")
        elif lab.viscosity_cst_100f > baseline_visc * 1.1:
            findings.append("Viscosity increase indicates oxidation or polymerization")

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
            "change_pct": change_pct,
        }

    def _analyze_flash_point(
        self,
        lab: FluidLabAnalysis,
        baseline: Optional[FluidLabAnalysis],
    ) -> Dict[str, Any]:
        """Analyze flash point changes."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.flash_point_f is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        # Determine baseline
        if baseline and baseline.flash_point_f:
            baseline_fp = baseline.flash_point_f
        else:
            baseline_fp = self.limits.new_flash_point_f

        # Calculate drop
        drop_f = baseline_fp - lab.flash_point_f

        if drop_f >= self.limits.flash_point_critical_drop_f:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Flash point dropped {drop_f:.0f}F "
                f"(current: {lab.flash_point_f:.0f}F, new: {baseline_fp:.0f}F)"
            )
            recommendations.append(
                "Immediate action required - low boilers formation indicates thermal degradation"
            )
            recommendations.append("Consider fluid replacement or distillation")

        elif drop_f >= self.limits.flash_point_warning_drop_f:
            status = ValidationStatus.WARNING
            score = drop_f / self.limits.flash_point_critical_drop_f * 100
            findings.append(f"WARNING: Flash point dropped {drop_f:.0f}F from baseline")
            recommendations.append("Increase venting/degassing frequency")
            recommendations.append("Monitor for low boiler accumulation")

        else:
            score = max(0, drop_f) / self.limits.flash_point_warning_drop_f * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
            "drop_f": drop_f,
        }

    def _analyze_acid_number(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze total acid number."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.total_acid_number_mg_koh_g is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        tan = lab.total_acid_number_mg_koh_g

        if tan >= self.limits.acid_number_critical:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Total Acid Number {tan:.2f} mg KOH/g exceeds limit"
            )
            recommendations.append("Fluid is heavily oxidized - replacement recommended")

        elif tan >= self.limits.acid_number_warning:
            status = ValidationStatus.WARNING
            score = tan / self.limits.acid_number_critical * 100
            findings.append(f"WARNING: Total Acid Number elevated at {tan:.2f} mg KOH/g")
            recommendations.append("Check expansion tank nitrogen blanket")
            recommendations.append("Verify system is sealed from air ingress")

        else:
            score = tan / self.limits.acid_number_warning * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _analyze_carbon_residue(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze carbon residue (Conradson)."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.carbon_residue_pct is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        cr = lab.carbon_residue_pct

        if cr >= self.limits.carbon_residue_critical_pct:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Carbon residue {cr:.2f}% indicates high boiler accumulation"
            )
            recommendations.append("High risk of coking on heater tubes")
            recommendations.append("Consider fluid replacement")

        elif cr >= self.limits.carbon_residue_warning_pct:
            status = ValidationStatus.WARNING
            score = cr / self.limits.carbon_residue_critical_pct * 100
            findings.append(f"WARNING: Carbon residue elevated at {cr:.2f}%")
            recommendations.append("Monitor heater tube condition")

        else:
            score = cr / self.limits.carbon_residue_warning_pct * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _analyze_moisture(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze moisture content."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.moisture_ppm is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        moisture = lab.moisture_ppm

        if moisture >= self.limits.moisture_critical_ppm:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Moisture content {moisture:.0f} ppm exceeds limit"
            )
            recommendations.append("High moisture can cause pump cavitation and system upset")
            recommendations.append("Heat fluid to drive off moisture")
            recommendations.append("Check for heat exchanger leaks")

        elif moisture >= self.limits.moisture_warning_ppm:
            status = ValidationStatus.WARNING
            score = moisture / self.limits.moisture_critical_ppm * 100
            findings.append(f"WARNING: Moisture content elevated at {moisture:.0f} ppm")
            recommendations.append("Check expansion tank condition")
            recommendations.append("Verify seals and gaskets")

        else:
            score = moisture / self.limits.moisture_warning_ppm * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _analyze_low_boilers(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze low boiling compounds (volatile degradation products)."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.low_boilers_pct is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        lb = lab.low_boilers_pct

        if lb >= self.limits.low_boilers_critical_pct:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Low boilers {lb:.1f}% indicates severe thermal cracking"
            )
            recommendations.append("System is experiencing significant thermal degradation")
            recommendations.append("Check for hot spots and film temperature exceedances")
            recommendations.append("Consider flash distillation or fluid replacement")

        elif lb >= self.limits.low_boilers_warning_pct:
            status = ValidationStatus.WARNING
            score = lb / self.limits.low_boilers_critical_pct * 100
            findings.append(f"WARNING: Low boilers elevated at {lb:.1f}%")
            recommendations.append("Verify film temperature is within limits")
            recommendations.append("Increase venting frequency")

        else:
            score = lb / self.limits.low_boilers_warning_pct * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _analyze_high_boilers(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze high boiling compounds (polymeric degradation products)."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.high_boilers_pct is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        hb = lab.high_boilers_pct

        if hb >= self.limits.high_boilers_critical_pct:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: High boilers {hb:.1f}% indicates polymerization/oxidation"
            )
            recommendations.append("High coking potential on heated surfaces")
            recommendations.append("Consider fluid replacement")

        elif hb >= self.limits.high_boilers_warning_pct:
            status = ValidationStatus.WARNING
            score = hb / self.limits.high_boilers_critical_pct * 100
            findings.append(f"WARNING: High boilers elevated at {hb:.1f}%")
            recommendations.append("Monitor system for fouling")
            recommendations.append("Check nitrogen blanket on expansion tank")

        else:
            score = hb / self.limits.high_boilers_warning_pct * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _analyze_thermal_conductivity(
        self,
        lab: FluidLabAnalysis,
    ) -> Dict[str, Any]:
        """Analyze thermal conductivity changes."""
        findings = []
        recommendations = []
        status = ValidationStatus.VALID
        score = 0.0

        if lab.thermal_conductivity_change_pct is None:
            return {
                "status": status,
                "score": 0.0,
                "findings": [],
                "recommendations": [],
            }

        change = abs(lab.thermal_conductivity_change_pct)

        if change >= self.limits.thermal_conductivity_critical_pct:
            status = ValidationStatus.INVALID
            score = 100.0
            findings.append(
                f"CRITICAL: Thermal conductivity changed {change:.1f}%"
            )
            recommendations.append("Heat transfer efficiency significantly degraded")

        elif change >= self.limits.thermal_conductivity_warning_pct:
            status = ValidationStatus.WARNING
            score = change / self.limits.thermal_conductivity_critical_pct * 100
            findings.append(f"WARNING: Thermal conductivity changed {change:.1f}%")

        else:
            score = change / self.limits.thermal_conductivity_warning_pct * 50

        return {
            "status": status,
            "score": min(100, score),
            "findings": findings,
            "recommendations": recommendations,
        }

    def _calculate_degradation_score(
        self,
        viscosity: Dict,
        flash_point: Dict,
        acid_number: Dict,
        carbon_residue: Dict,
        moisture: Dict,
        low_boilers: Dict,
        high_boilers: Dict,
    ) -> float:
        """
        Calculate composite degradation score (0-100).

        Weighting reflects relative importance of each parameter.
        """
        # Weights for each parameter
        weights = {
            "viscosity": 15,
            "flash_point": 20,
            "acid_number": 15,
            "carbon_residue": 15,
            "moisture": 10,
            "low_boilers": 15,
            "high_boilers": 10,
        }

        total_weight = sum(weights.values())

        weighted_score = (
            viscosity["score"] * weights["viscosity"] +
            flash_point["score"] * weights["flash_point"] +
            acid_number["score"] * weights["acid_number"] +
            carbon_residue["score"] * weights["carbon_residue"] +
            moisture["score"] * weights["moisture"] +
            low_boilers["score"] * weights["low_boilers"] +
            high_boilers["score"] * weights["high_boilers"]
        )

        return weighted_score / total_weight

    def _determine_degradation_level(
        self,
        degradation_score: float,
    ) -> DegradationLevel:
        """Determine degradation level from score."""
        if degradation_score >= 80:
            return DegradationLevel.CRITICAL
        elif degradation_score >= 60:
            return DegradationLevel.POOR
        elif degradation_score >= 40:
            return DegradationLevel.FAIR
        elif degradation_score >= 20:
            return DegradationLevel.GOOD
        else:
            return DegradationLevel.EXCELLENT

    def _calculate_next_sample_date(
        self,
        degradation_level: DegradationLevel,
        current_sample_date: datetime,
    ) -> datetime:
        """Calculate recommended next sample date."""
        if degradation_level == DegradationLevel.CRITICAL:
            months = 1  # Monthly monitoring
        elif degradation_level == DegradationLevel.POOR:
            months = self.thresholds.sampling_interval_critical_months
        elif degradation_level == DegradationLevel.FAIR:
            months = self.thresholds.sampling_interval_months // 2
        else:
            months = self.thresholds.sampling_interval_months

        return current_sample_date + timedelta(days=30 * months)

    def get_trending_analysis(
        self,
        analyses: List[FluidLabAnalysis],
    ) -> Dict[str, Any]:
        """
        Analyze degradation trends over multiple samples.

        Args:
            analyses: List of lab analyses in chronological order

        Returns:
            Dictionary with trending analysis
        """
        if len(analyses) < 2:
            return {"message": "Insufficient data for trending"}

        # Sort by date
        analyses = sorted(analyses, key=lambda x: x.sample_date)

        trends = {}

        # Analyze viscosity trend
        if all(a.viscosity_cst_100f for a in analyses):
            viscosities = [a.viscosity_cst_100f for a in analyses]
            trends["viscosity"] = self._calculate_trend(viscosities)

        # Analyze flash point trend
        if all(a.flash_point_f for a in analyses):
            flash_points = [a.flash_point_f for a in analyses]
            trends["flash_point"] = self._calculate_trend(flash_points)

        # Analyze acid number trend
        if all(a.total_acid_number_mg_koh_g for a in analyses):
            acid_numbers = [a.total_acid_number_mg_koh_g for a in analyses]
            trends["acid_number"] = self._calculate_trend(acid_numbers)

        return trends

    def _calculate_trend(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        """Calculate linear trend for a parameter."""
        n = len(values)
        if n < 2:
            return {"slope": 0, "direction": "stable"}

        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        if abs(slope) < 0.01 * y_mean:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "slope": slope,
            "direction": direction,
            "first_value": values[0],
            "last_value": values[-1],
            "change_pct": (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0,
        }

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
