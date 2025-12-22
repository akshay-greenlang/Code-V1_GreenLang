# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Temperature Differential Calculator

This module provides deterministic temperature-based diagnostics for steam trap
analysis following GreenLang zero-hallucination principles.

Features:
    - Inlet/outlet temperature differential analysis
    - Subcooling detection and quantification
    - Failed-open and failed-closed detection
    - Saturation temperature comparison
    - Seasonal temperature compensation
    - Condensate backup detection

Zero-Hallucination Guarantee:
    All calculations use deterministic thermodynamic formulas from steam tables.
    No LLM or AI inference in any calculation path.
    Same temperature inputs always produce identical diagnostic outputs.

Engineering References:
    - ASME Steam Tables (IF-97)
    - ASHRAE Fundamentals
    - Armstrong Steam Trap Handbook

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TrapCondition(str, Enum):
    """Steam trap condition based on temperature analysis."""
    OPERATING_NORMALLY = "operating_normally"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    LEAKING = "leaking"
    SUBCOOLED = "subcooled"
    FLOODED = "flooded"
    COLD = "cold"
    UNKNOWN = "unknown"


class SeasonalFactor(str, Enum):
    """Seasonal adjustment factors for temperature analysis."""
    WINTER = "winter"    # Higher ambient losses
    SPRING = "spring"
    SUMMER = "summer"    # Lower ambient losses
    FALL = "fall"


class TemperatureSeverity(str, Enum):
    """Severity levels for temperature diagnostics."""
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    """Confidence levels for diagnostic results."""
    HIGH = "high"         # >= 85%
    MEDIUM = "medium"     # 60-84%
    LOW = "low"           # 40-59%
    VERY_LOW = "very_low" # < 40%


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TemperatureConfig:
    """
    Configuration for temperature differential analysis.

    Attributes:
        normal_differential_min_c: Minimum normal temp differential (inlet-outlet)
        normal_differential_max_c: Maximum normal temp differential
        failed_open_threshold_c: Threshold for failed-open detection
        failed_closed_threshold_c: Threshold for failed-closed detection
        subcooling_threshold_c: Minimum subcooling to flag
        saturation_tolerance_c: Tolerance for saturation comparison
        ambient_reference_c: Reference ambient temperature
        seasonal_adjustment_enabled: Whether to apply seasonal factors
    """
    normal_differential_min_c: Decimal = field(default_factory=lambda: Decimal("5.0"))
    normal_differential_max_c: Decimal = field(default_factory=lambda: Decimal("30.0"))
    failed_open_threshold_c: Decimal = field(default_factory=lambda: Decimal("3.0"))
    failed_closed_threshold_c: Decimal = field(default_factory=lambda: Decimal("50.0"))
    subcooling_threshold_c: Decimal = field(default_factory=lambda: Decimal("15.0"))
    saturation_tolerance_c: Decimal = field(default_factory=lambda: Decimal("5.0"))
    ambient_reference_c: Decimal = field(default_factory=lambda: Decimal("20.0"))
    seasonal_adjustment_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.normal_differential_min_c >= self.normal_differential_max_c:
            raise ValueError("Minimum differential must be less than maximum")
        if self.failed_open_threshold_c >= self.normal_differential_min_c:
            raise ValueError("Failed-open threshold must be less than normal minimum")


# =============================================================================
# STEAM TABLE LOOKUP DATA (Deterministic)
# =============================================================================

# Simplified saturation temperature lookup table
# Pressure (bar gauge) -> Saturation Temperature (Celsius)
SATURATION_TABLE: Dict[Decimal, Decimal] = {
    Decimal("0.0"): Decimal("100.0"),
    Decimal("0.5"): Decimal("111.4"),
    Decimal("1.0"): Decimal("120.2"),
    Decimal("1.5"): Decimal("127.4"),
    Decimal("2.0"): Decimal("133.5"),
    Decimal("2.5"): Decimal("138.9"),
    Decimal("3.0"): Decimal("143.6"),
    Decimal("3.5"): Decimal("147.9"),
    Decimal("4.0"): Decimal("151.8"),
    Decimal("4.5"): Decimal("155.5"),
    Decimal("5.0"): Decimal("158.8"),
    Decimal("6.0"): Decimal("164.9"),
    Decimal("7.0"): Decimal("170.4"),
    Decimal("8.0"): Decimal("175.4"),
    Decimal("9.0"): Decimal("179.9"),
    Decimal("10.0"): Decimal("184.1"),
    Decimal("12.0"): Decimal("191.6"),
    Decimal("14.0"): Decimal("198.3"),
    Decimal("16.0"): Decimal("204.3"),
    Decimal("18.0"): Decimal("209.8"),
    Decimal("20.0"): Decimal("214.9"),
}

# Seasonal adjustment factors for ambient heat loss
SEASONAL_FACTORS: Dict[SeasonalFactor, Decimal] = {
    SeasonalFactor.WINTER: Decimal("1.15"),   # 15% higher losses
    SeasonalFactor.SPRING: Decimal("1.05"),
    SeasonalFactor.SUMMER: Decimal("0.90"),   # 10% lower losses
    SeasonalFactor.FALL: Decimal("1.05"),
}


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

@dataclass(frozen=True)
class ProvenanceStep:
    """Single step in calculation provenance chain."""
    step_name: str
    inputs: Dict[str, Any]
    formula: str
    output: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "inputs": self.inputs,
            "formula": self.formula,
            "output": str(self.output) if isinstance(self.output, Decimal) else self.output,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProvenanceTracker:
    """Tracks complete calculation provenance for audit trails."""
    calculation_id: str = ""
    steps: List[ProvenanceStep] = field(default_factory=list)

    def add_step(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        formula: str,
        output: Any
    ) -> None:
        """Add a calculation step to the provenance chain."""
        step = ProvenanceStep(
            step_name=step_name,
            inputs=inputs,
            formula=formula,
            output=output,
        )
        self.steps.append(step)

    def get_hash(self) -> str:
        """Generate SHA-256 hash of entire provenance chain."""
        chain_str = "|".join(
            f"{s.step_name}:{s.inputs}:{s.formula}:{s.output}"
            for s in self.steps
        )
        return hashlib.sha256(chain_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance chain to dictionary."""
        return {
            "calculation_id": self.calculation_id,
            "steps": [s.to_dict() for s in self.steps],
            "provenance_hash": self.get_hash(),
        }


# =============================================================================
# INPUT DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class TemperatureReading:
    """
    Temperature measurement data for a steam trap.

    Attributes:
        trap_id: Unique identifier for the steam trap
        reading_timestamp: When the measurement was taken
        inlet_temperature_c: Temperature at trap inlet (steam side)
        outlet_temperature_c: Temperature at trap outlet (condensate side)
        ambient_temperature_c: Ambient temperature at trap location
        steam_pressure_bar: System steam pressure in bar gauge
        sensor_accuracy_c: Accuracy of temperature sensors
        measurement_method: Method used (contact, IR, etc.)
    """
    trap_id: str
    reading_timestamp: datetime
    inlet_temperature_c: Decimal
    outlet_temperature_c: Decimal
    ambient_temperature_c: Decimal
    steam_pressure_bar: Decimal
    sensor_accuracy_c: Decimal = field(default_factory=lambda: Decimal("1.0"))
    measurement_method: str = "contact"

    def __post_init__(self) -> None:
        """Validate temperature reading values."""
        if self.inlet_temperature_c < Decimal("-50") or self.inlet_temperature_c > Decimal("350"):
            raise ValueError(f"Inlet temperature {self.inlet_temperature_c} outside valid range")
        if self.outlet_temperature_c < Decimal("-50") or self.outlet_temperature_c > Decimal("350"):
            raise ValueError(f"Outlet temperature {self.outlet_temperature_c} outside valid range")
        if self.steam_pressure_bar < Decimal("0") or self.steam_pressure_bar > Decimal("50"):
            raise ValueError(f"Steam pressure {self.steam_pressure_bar} outside valid range")


# =============================================================================
# OUTPUT DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class DifferentialAnalysisResult:
    """
    Result of temperature differential analysis.

    Attributes:
        temperature_differential_c: Inlet minus outlet temperature
        differential_ratio: Ratio of differential to expected
        is_within_normal_range: Whether differential is normal
        adjusted_differential_c: Differential adjusted for ambient
    """
    temperature_differential_c: Decimal
    differential_ratio: Decimal
    is_within_normal_range: bool
    adjusted_differential_c: Decimal


@dataclass(frozen=True)
class SubcoolingAnalysis:
    """
    Subcooling analysis result.

    Attributes:
        saturation_temperature_c: Expected saturation temp at pressure
        actual_outlet_temperature_c: Measured outlet temperature
        subcooling_amount_c: Degrees of subcooling
        subcooling_percentage: Subcooling as percentage of saturation
        is_excessive: Whether subcooling exceeds threshold
    """
    saturation_temperature_c: Decimal
    actual_outlet_temperature_c: Decimal
    subcooling_amount_c: Decimal
    subcooling_percentage: Decimal
    is_excessive: bool


@dataclass(frozen=True)
class FailureDetectionResult:
    """
    Trap failure detection result.

    Attributes:
        detected_condition: Most likely trap condition
        confidence: Confidence score (0.0 - 1.0)
        contributing_factors: Evidence supporting the diagnosis
        alternative_conditions: Other possible conditions
    """
    detected_condition: TrapCondition
    confidence: Decimal
    contributing_factors: List[str]
    alternative_conditions: Dict[TrapCondition, Decimal]


@dataclass(frozen=True)
class SaturationComparison:
    """
    Comparison of measured temperatures to saturation.

    Attributes:
        inlet_vs_saturation_c: Inlet temp minus saturation temp
        outlet_vs_saturation_c: Outlet temp minus saturation temp
        inlet_at_saturation: Whether inlet is at saturation
        steam_quality_indicator: Indicator of steam quality
    """
    inlet_vs_saturation_c: Decimal
    outlet_vs_saturation_c: Decimal
    inlet_at_saturation: bool
    steam_quality_indicator: str


@dataclass(frozen=True)
class TemperatureDiagnosticResult:
    """
    Complete temperature diagnostic analysis result.

    Attributes:
        trap_id: Steam trap identifier
        analysis_timestamp: When analysis was performed
        differential_analysis: Differential analysis results
        subcooling_analysis: Subcooling analysis results
        failure_detection: Failure detection results
        saturation_comparison: Saturation comparison results
        recommended_action: Suggested maintenance action
        diagnostic_severity: Overall severity assessment
        provenance_hash: SHA-256 hash for audit trail
        calculation_details: Detailed provenance chain
    """
    trap_id: str
    analysis_timestamp: datetime
    differential_analysis: DifferentialAnalysisResult
    subcooling_analysis: SubcoolingAnalysis
    failure_detection: FailureDetectionResult
    saturation_comparison: SaturationComparison
    recommended_action: str
    diagnostic_severity: TemperatureSeverity
    provenance_hash: str
    calculation_details: Dict[str, Any]


# =============================================================================
# TEMPERATURE DIFFERENTIAL CALCULATOR
# =============================================================================

class TemperatureDifferentialCalculator:
    """
    Deterministic temperature differential calculator for steam trap diagnostics.

    This calculator analyzes inlet and outlet temperature measurements to
    diagnose steam trap conditions using thermodynamic principles. All
    calculations are deterministic - same inputs always produce same outputs.

    Zero-Hallucination Design:
        - Saturation temperatures from ASME steam tables lookup
        - Differential thresholds from engineering standards
        - No ML/AI inference in any calculation path
        - Complete provenance tracking for audit trails

    Example:
        >>> config = TemperatureConfig()
        >>> calculator = TemperatureDifferentialCalculator(config)
        >>> reading = TemperatureReading(
        ...     trap_id="TRAP-001",
        ...     reading_timestamp=datetime.now(),
        ...     inlet_temperature_c=Decimal("150.0"),
        ...     outlet_temperature_c=Decimal("148.0"),
        ...     ambient_temperature_c=Decimal("25.0"),
        ...     steam_pressure_bar=Decimal("4.0")
        ... )
        >>> result = calculator.analyze_temperature(reading)
        >>> print(result.failure_detection.detected_condition)
        TrapCondition.FAILED_OPEN

    Attributes:
        config: Calculator configuration parameters
    """

    def __init__(self, config: Optional[TemperatureConfig] = None):
        """
        Initialize the temperature differential calculator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or TemperatureConfig()
        logger.info(f"TemperatureDifferentialCalculator initialized")

    def analyze_temperature(
        self,
        reading: TemperatureReading,
        season: Optional[SeasonalFactor] = None,
        include_provenance: bool = True
    ) -> TemperatureDiagnosticResult:
        """
        Perform complete temperature diagnostic analysis.

        This is the main entry point for temperature analysis. It performs:
        1. Temperature differential calculation
        2. Subcooling analysis
        3. Saturation temperature comparison
        4. Failure condition detection
        5. Recommendation generation

        Args:
            reading: Temperature reading data
            season: Optional seasonal adjustment factor
            include_provenance: Whether to include detailed provenance

        Returns:
            Complete diagnostic result with all analysis components

        Raises:
            ValueError: If reading data is invalid
        """
        logger.info(f"Analyzing temperature for trap {reading.trap_id}")
        start_time = datetime.utcnow()

        # Initialize provenance tracker
        provenance = ProvenanceTracker(
            calculation_id=f"temp_{reading.trap_id}_{start_time.isoformat()}"
        )

        # Step 1: Calculate temperature differential
        differential_analysis = self._analyze_differential(reading, season, provenance)

        # Step 2: Analyze subcooling
        subcooling_analysis = self._analyze_subcooling(reading, provenance)

        # Step 3: Compare to saturation
        saturation_comparison = self._compare_to_saturation(reading, provenance)

        # Step 4: Detect failure condition
        failure_detection = self._detect_failure(
            reading, differential_analysis, subcooling_analysis,
            saturation_comparison, provenance
        )

        # Step 5: Determine severity
        severity = self._determine_severity(failure_detection, provenance)

        # Step 6: Generate recommendation
        recommendation = self._generate_recommendation(
            failure_detection, severity, provenance
        )

        # Generate final provenance hash
        provenance_hash = provenance.get_hash()

        result = TemperatureDiagnosticResult(
            trap_id=reading.trap_id,
            analysis_timestamp=start_time,
            differential_analysis=differential_analysis,
            subcooling_analysis=subcooling_analysis,
            failure_detection=failure_detection,
            saturation_comparison=saturation_comparison,
            recommended_action=recommendation,
            diagnostic_severity=severity,
            provenance_hash=provenance_hash,
            calculation_details=provenance.to_dict() if include_provenance else {},
        )

        logger.info(
            f"Temperature analysis complete for {reading.trap_id}: "
            f"condition={failure_detection.detected_condition.value}, "
            f"severity={severity.value}"
        )

        return result

    def _analyze_differential(
        self,
        reading: TemperatureReading,
        season: Optional[SeasonalFactor],
        provenance: ProvenanceTracker
    ) -> DifferentialAnalysisResult:
        """
        Analyze temperature differential between inlet and outlet.

        Args:
            reading: Temperature reading
            season: Optional seasonal factor for adjustment
            provenance: Provenance tracker

        Returns:
            Differential analysis result
        """
        # Calculate raw differential
        differential = reading.inlet_temperature_c - reading.outlet_temperature_c

        provenance.add_step(
            step_name="calculate_differential",
            inputs={
                "inlet_c": str(reading.inlet_temperature_c),
                "outlet_c": str(reading.outlet_temperature_c)
            },
            formula="differential = inlet - outlet",
            output=str(differential)
        )

        # Apply seasonal adjustment if enabled
        if self.config.seasonal_adjustment_enabled and season:
            seasonal_factor = SEASONAL_FACTORS.get(season, Decimal("1.0"))
            adjusted_differential = differential * seasonal_factor
        else:
            adjusted_differential = differential

        provenance.add_step(
            step_name="apply_seasonal_adjustment",
            inputs={
                "raw_differential": str(differential),
                "season": season.value if season else "none",
                "factor": str(SEASONAL_FACTORS.get(season, Decimal("1.0"))) if season else "1.0"
            },
            formula="adjusted = differential * seasonal_factor",
            output=str(adjusted_differential)
        )

        # Calculate expected differential ratio
        expected_mid = (
            self.config.normal_differential_min_c +
            self.config.normal_differential_max_c
        ) / Decimal("2")

        if expected_mid > 0:
            differential_ratio = (adjusted_differential / expected_mid).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            differential_ratio = Decimal("0")

        # Check if within normal range
        is_normal = (
            self.config.normal_differential_min_c <=
            adjusted_differential <=
            self.config.normal_differential_max_c
        )

        provenance.add_step(
            step_name="evaluate_differential_range",
            inputs={
                "adjusted_differential": str(adjusted_differential),
                "min_normal": str(self.config.normal_differential_min_c),
                "max_normal": str(self.config.normal_differential_max_c)
            },
            formula="is_normal = min <= differential <= max",
            output=is_normal
        )

        return DifferentialAnalysisResult(
            temperature_differential_c=differential.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            differential_ratio=differential_ratio,
            is_within_normal_range=is_normal,
            adjusted_differential_c=adjusted_differential.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        )

    def _get_saturation_temperature(
        self,
        pressure_bar: Decimal,
        provenance: ProvenanceTracker
    ) -> Decimal:
        """
        Look up saturation temperature from steam tables.

        Uses linear interpolation between table values.

        Args:
            pressure_bar: Steam pressure in bar gauge
            provenance: Provenance tracker

        Returns:
            Saturation temperature in Celsius
        """
        # Find bracketing pressures
        pressures = sorted(SATURATION_TABLE.keys())

        if pressure_bar <= pressures[0]:
            sat_temp = SATURATION_TABLE[pressures[0]]
        elif pressure_bar >= pressures[-1]:
            sat_temp = SATURATION_TABLE[pressures[-1]]
        else:
            # Linear interpolation
            lower_p = max(p for p in pressures if p <= pressure_bar)
            upper_p = min(p for p in pressures if p > pressure_bar)

            lower_t = SATURATION_TABLE[lower_p]
            upper_t = SATURATION_TABLE[upper_p]

            # Interpolate
            fraction = (pressure_bar - lower_p) / (upper_p - lower_p)
            sat_temp = lower_t + fraction * (upper_t - lower_t)

        sat_temp = sat_temp.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        provenance.add_step(
            step_name="lookup_saturation_temperature",
            inputs={"pressure_bar": str(pressure_bar)},
            formula="linear_interpolation(SATURATION_TABLE)",
            output=str(sat_temp)
        )

        return sat_temp

    def _analyze_subcooling(
        self,
        reading: TemperatureReading,
        provenance: ProvenanceTracker
    ) -> SubcoolingAnalysis:
        """
        Analyze subcooling at trap outlet.

        Subcooling is the difference between saturation temperature
        and actual outlet temperature.

        Args:
            reading: Temperature reading
            provenance: Provenance tracker

        Returns:
            Subcooling analysis result
        """
        # Get saturation temperature at system pressure
        sat_temp = self._get_saturation_temperature(
            reading.steam_pressure_bar, provenance
        )

        # Calculate subcooling
        subcooling = sat_temp - reading.outlet_temperature_c

        provenance.add_step(
            step_name="calculate_subcooling",
            inputs={
                "saturation_c": str(sat_temp),
                "outlet_c": str(reading.outlet_temperature_c)
            },
            formula="subcooling = saturation - outlet",
            output=str(subcooling)
        )

        # Calculate subcooling percentage
        if sat_temp > Decimal("0"):
            subcooling_pct = (subcooling / sat_temp * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            subcooling_pct = Decimal("0")

        # Check if excessive
        is_excessive = subcooling > self.config.subcooling_threshold_c

        provenance.add_step(
            step_name="evaluate_subcooling",
            inputs={
                "subcooling_c": str(subcooling),
                "threshold_c": str(self.config.subcooling_threshold_c)
            },
            formula="is_excessive = subcooling > threshold",
            output=is_excessive
        )

        return SubcoolingAnalysis(
            saturation_temperature_c=sat_temp,
            actual_outlet_temperature_c=reading.outlet_temperature_c,
            subcooling_amount_c=subcooling.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            subcooling_percentage=subcooling_pct,
            is_excessive=is_excessive
        )

    def _compare_to_saturation(
        self,
        reading: TemperatureReading,
        provenance: ProvenanceTracker
    ) -> SaturationComparison:
        """
        Compare measured temperatures to saturation.

        Args:
            reading: Temperature reading
            provenance: Provenance tracker

        Returns:
            Saturation comparison result
        """
        sat_temp = self._get_saturation_temperature(
            reading.steam_pressure_bar, provenance
        )

        inlet_vs_sat = reading.inlet_temperature_c - sat_temp
        outlet_vs_sat = reading.outlet_temperature_c - sat_temp

        # Check if inlet is at saturation (within tolerance)
        inlet_at_saturation = abs(inlet_vs_sat) <= self.config.saturation_tolerance_c

        # Determine steam quality indicator
        if inlet_at_saturation:
            if reading.inlet_temperature_c > sat_temp:
                quality_indicator = "superheated"
            else:
                quality_indicator = "saturated"
        elif reading.inlet_temperature_c < sat_temp - self.config.saturation_tolerance_c:
            quality_indicator = "subcooled_inlet"
        else:
            quality_indicator = "normal"

        provenance.add_step(
            step_name="compare_to_saturation",
            inputs={
                "inlet_c": str(reading.inlet_temperature_c),
                "outlet_c": str(reading.outlet_temperature_c),
                "saturation_c": str(sat_temp),
                "tolerance_c": str(self.config.saturation_tolerance_c)
            },
            formula="inlet_at_sat = abs(inlet - sat) <= tolerance",
            output={
                "inlet_vs_sat": str(inlet_vs_sat),
                "outlet_vs_sat": str(outlet_vs_sat),
                "quality": quality_indicator
            }
        )

        return SaturationComparison(
            inlet_vs_saturation_c=inlet_vs_sat.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            outlet_vs_saturation_c=outlet_vs_sat.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            inlet_at_saturation=inlet_at_saturation,
            steam_quality_indicator=quality_indicator
        )

    def _detect_failure(
        self,
        reading: TemperatureReading,
        differential: DifferentialAnalysisResult,
        subcooling: SubcoolingAnalysis,
        saturation: SaturationComparison,
        provenance: ProvenanceTracker
    ) -> FailureDetectionResult:
        """
        Detect trap failure condition from temperature analysis.

        Uses deterministic rules based on engineering thresholds.

        Args:
            reading: Temperature reading
            differential: Differential analysis result
            subcooling: Subcooling analysis result
            saturation: Saturation comparison result
            provenance: Provenance tracker

        Returns:
            Failure detection result
        """
        contributing_factors: List[str] = []
        condition_scores: Dict[TrapCondition, Decimal] = {
            condition: Decimal("0") for condition in TrapCondition
        }

        # Rule 1: Failed Open - Very low differential
        if differential.temperature_differential_c < self.config.failed_open_threshold_c:
            condition_scores[TrapCondition.FAILED_OPEN] += Decimal("0.5")
            contributing_factors.append(
                f"Very low differential ({differential.temperature_differential_c}C) "
                f"below threshold ({self.config.failed_open_threshold_c}C)"
            )

        # Rule 2: Failed Open - Outlet near saturation
        if not subcooling.is_excessive and saturation.outlet_vs_saturation_c > Decimal("-5"):
            condition_scores[TrapCondition.FAILED_OPEN] += Decimal("0.3")
            contributing_factors.append(
                "Outlet temperature near saturation indicates steam passing through"
            )

        # Rule 3: Failed Closed - Very high differential
        if differential.temperature_differential_c > self.config.failed_closed_threshold_c:
            condition_scores[TrapCondition.FAILED_CLOSED] += Decimal("0.6")
            contributing_factors.append(
                f"Very high differential ({differential.temperature_differential_c}C) "
                f"above threshold ({self.config.failed_closed_threshold_c}C)"
            )

        # Rule 4: Failed Closed - Outlet much cooler than expected
        if subcooling.subcooling_amount_c > Decimal("40"):
            condition_scores[TrapCondition.FAILED_CLOSED] += Decimal("0.3")
            contributing_factors.append(
                f"Excessive subcooling ({subcooling.subcooling_amount_c}C) indicates blocked trap"
            )

        # Rule 5: Cold trap - Inlet below saturation
        if saturation.steam_quality_indicator == "subcooled_inlet":
            condition_scores[TrapCondition.COLD] += Decimal("0.7")
            contributing_factors.append(
                "Inlet temperature below saturation indicates no steam supply"
            )

        # Rule 6: Flooded - Excessive subcooling with moderate differential
        if (subcooling.is_excessive and
            self.config.normal_differential_min_c < differential.temperature_differential_c <
            self.config.failed_closed_threshold_c):
            condition_scores[TrapCondition.FLOODED] += Decimal("0.5")
            contributing_factors.append(
                "Excessive subcooling with moderate differential suggests flooding"
            )

        # Rule 7: Leaking - Moderate issues
        if (differential.temperature_differential_c < self.config.normal_differential_min_c and
            differential.temperature_differential_c >= self.config.failed_open_threshold_c):
            condition_scores[TrapCondition.LEAKING] += Decimal("0.4")
            contributing_factors.append(
                "Low but not critical differential may indicate partial leak"
            )

        # Rule 8: Normal operation
        if (differential.is_within_normal_range and
            not subcooling.is_excessive and
            saturation.inlet_at_saturation):
            condition_scores[TrapCondition.OPERATING_NORMALLY] += Decimal("0.8")
            contributing_factors.append(
                "Temperature readings within normal operating parameters"
            )

        # Find condition with highest score
        best_condition = max(condition_scores, key=lambda k: condition_scores[k])
        best_score = condition_scores[best_condition]

        # If no clear diagnosis, mark as unknown
        if best_score < Decimal("0.3"):
            best_condition = TrapCondition.UNKNOWN
            best_score = Decimal("0.5")
            contributing_factors.append(
                "Insufficient evidence for definitive diagnosis"
            )

        # Calculate confidence (normalize score)
        confidence = min(best_score, Decimal("1.0"))

        # Get alternatives
        alternatives = {
            k: v for k, v in condition_scores.items()
            if k != best_condition and v > Decimal("0.1")
        }

        provenance.add_step(
            step_name="detect_failure_condition",
            inputs={
                "differential_c": str(differential.temperature_differential_c),
                "subcooling_c": str(subcooling.subcooling_amount_c),
                "inlet_at_saturation": saturation.inlet_at_saturation
            },
            formula="rule_based_scoring_system",
            output={
                "condition": best_condition.value,
                "confidence": str(confidence),
                "scores": {k.value: str(v) for k, v in condition_scores.items()}
            }
        )

        return FailureDetectionResult(
            detected_condition=best_condition,
            confidence=confidence.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            contributing_factors=contributing_factors,
            alternative_conditions=alternatives
        )

    def _determine_severity(
        self,
        failure: FailureDetectionResult,
        provenance: ProvenanceTracker
    ) -> TemperatureSeverity:
        severity_map = {
            TrapCondition.OPERATING_NORMALLY: TemperatureSeverity.NORMAL,
            TrapCondition.FAILED_OPEN: TemperatureSeverity.CRITICAL,
            TrapCondition.FAILED_CLOSED: TemperatureSeverity.CRITICAL,
            TrapCondition.LEAKING: TemperatureSeverity.WARNING,
            TrapCondition.SUBCOOLED: TemperatureSeverity.WATCH,
            TrapCondition.FLOODED: TemperatureSeverity.WARNING,
            TrapCondition.COLD: TemperatureSeverity.WARNING,
            TrapCondition.UNKNOWN: TemperatureSeverity.WATCH,
        }
        severity = severity_map.get(failure.detected_condition, TemperatureSeverity.WATCH)
        if failure.confidence < Decimal("0.5") and severity == TemperatureSeverity.CRITICAL:
            severity = TemperatureSeverity.WARNING
        provenance.add_step(
            step_name="determine_severity",
            inputs={"condition": failure.detected_condition.value, "confidence": str(failure.confidence)},
            formula="severity_lookup_table[condition] adjusted for confidence",
            output=severity.value
        )
        return severity

    def _generate_recommendation(
        self,
        failure: FailureDetectionResult,
        severity: TemperatureSeverity,
        provenance: ProvenanceTracker
    ) -> str:
        recommendations = {
            (TrapCondition.OPERATING_NORMALLY, TemperatureSeverity.NORMAL):
                "No action required. Trap operating within normal parameters.",
            (TrapCondition.FAILED_OPEN, TemperatureSeverity.CRITICAL):
                "IMMEDIATE: Replace failed-open trap. Significant steam loss occurring.",
            (TrapCondition.FAILED_OPEN, TemperatureSeverity.WARNING):
                "URGENT: Schedule trap inspection within 48 hours.",
            (TrapCondition.FAILED_CLOSED, TemperatureSeverity.CRITICAL):
                "IMMEDIATE: Replace blocked trap. Risk of equipment damage from condensate backup.",
            (TrapCondition.FAILED_CLOSED, TemperatureSeverity.WARNING):
                "URGENT: Inspect trap for blockage. Check inlet strainer.",
            (TrapCondition.LEAKING, TemperatureSeverity.WARNING):
                "Schedule repair within 1 week. Monitor energy loss.",
            (TrapCondition.LEAKING, TemperatureSeverity.WATCH):
                "Add to watch list. Re-survey in 2 weeks.",
            (TrapCondition.FLOODED, TemperatureSeverity.WARNING):
                "URGENT: Check for water hammer risk. Inspect drainage.",
            (TrapCondition.COLD, TemperatureSeverity.WARNING):
                "Check steam supply. Verify isolation valves are open.",
            (TrapCondition.UNKNOWN, TemperatureSeverity.WATCH):
                "Re-survey with additional diagnostics. Consider acoustic testing.",
        }
        key = (failure.detected_condition, severity)
        recommendation = recommendations.get(
            key,
            f"Review trap condition. Detected: {failure.detected_condition.value}, Confidence: {failure.confidence}"
        )
        provenance.add_step(
            step_name="generate_recommendation",
            inputs={"condition": failure.detected_condition.value, "severity": severity.value},
            formula="lookup_table[condition, severity]",
            output=recommendation
        )
        return recommendation

    def analyze_batch(
        self,
        readings: List[TemperatureReading],
        season: Optional[SeasonalFactor] = None,
        include_provenance: bool = False
    ) -> List[TemperatureDiagnosticResult]:
        logger.info(f"Starting batch analysis of {len(readings)} readings")
        results = []
        for reading in readings:
            try:
                result = self.analyze_temperature(reading, season, include_provenance)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze reading for trap {reading.trap_id}: {e}")
                raise
        logger.info(f"Batch analysis complete. Processed {len(results)} readings.")
        return results

    def get_statistics(self, results: List[TemperatureDiagnosticResult]) -> Dict[str, Any]:
        if not results:
            return {
                "total_analyzed": 0,
                "condition_distribution": {},
                "severity_distribution": {},
                "average_confidence": "0.00"
            }
        cond_dist: Dict[str, int] = {}
        for r in results:
            cond = r.failure_detection.detected_condition.value
            cond_dist[cond] = cond_dist.get(cond, 0) + 1
        sev_dist: Dict[str, int] = {}
        for r in results:
            sev = r.diagnostic_severity.value
            sev_dist[sev] = sev_dist.get(sev, 0) + 1
        avg_confidence = sum(r.failure_detection.confidence for r in results) / Decimal(len(results))
        avg_differential = sum(r.differential_analysis.temperature_differential_c for r in results) / Decimal(len(results))
        return {
            "total_analyzed": len(results),
            "condition_distribution": cond_dist,
            "severity_distribution": sev_dist,
            "average_confidence": str(avg_confidence.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
            "average_differential_c": str(avg_differential.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
            "critical_count": sev_dist.get("critical", 0),
            "warning_count": sev_dist.get("warning", 0),
            "normal_count": cond_dist.get("operating_normally", 0),
        }


__all__ = [
    "TemperatureDifferentialCalculator",
    "TemperatureConfig",
    "TrapCondition",
    "SeasonalFactor",
    "TemperatureSeverity",
    "ConfidenceLevel",
    "TemperatureReading",
    "DifferentialAnalysisResult",
    "SubcoolingAnalysis",
    "FailureDetectionResult",
    "SaturationComparison",
    "TemperatureDiagnosticResult",
    "ProvenanceTracker",
    "ProvenanceStep",
    "SATURATION_TABLE",
    "SEASONAL_FACTORS",
]
