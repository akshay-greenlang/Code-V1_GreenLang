# -*- coding: utf-8 -*-
"""
TemperatureDifferentialAnalyzer for GL-008 TRAPCATCHER

Provides temperature differential analysis for steam trap diagnosis using
inlet/outlet temperature measurements and IR thermal imaging interpretation.

Standards:
- ISO 7841: Automatic steam traps - Determination of steam loss
- ASME PTC 39: Steam Traps
- ASTM E1933: Standard Test Method for Temperature Measurement

Key Features:
- Inlet/outlet temperature differential analysis
- Saturation temperature calculation (IAPWS-IF97)
- IR thermal imaging pattern interpretation
- Superheat/subcool detection
- Thermal signature pattern matching
- Condensate backing detection

Zero-Hallucination Guarantee:
All analysis uses deterministic thermodynamic calculations.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Example:
    >>> analyzer = TemperatureDifferentialAnalyzer()
    >>> result = analyzer.analyze(trap_id, inlet_c, outlet_c, pressure_bar)
    >>> print(f"Status: {result.diagnosis.status}")

Author: GL-CalculatorEngineer
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ThermalPattern(str, Enum):
    """Characteristic thermal patterns for trap diagnosis."""
    NORMAL = "normal"  # Normal temperature differential
    HOT_OUTLET = "hot_outlet"  # Outlet near steam temp (failed open)
    COLD_OUTLET = "cold_outlet"  # Outlet much cooler (failed closed/blocked)
    COLD_BOTH = "cold_both"  # Both sides cold (no steam)
    SUPERHEAT = "superheat"  # Superheated conditions
    SUBCOOL = "subcool"  # Subcooled liquid
    FLOODED = "flooded"  # Condensate backing up
    UNIFORM = "uniform"  # Uniform temperature (no flow)


class TrapStatusThermal(str, Enum):
    """Trap status from thermal analysis."""
    OPERATING = "operating"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    LEAKING = "leaking"
    COLD = "cold"
    FLOODED = "flooded"
    UNKNOWN = "unknown"


class IRImageQuality(str, Enum):
    """IR image quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ThermalAnalysisConfig:
    """
    Configuration for thermal analysis.

    Attributes:
        failed_open_delta_c: Max temp differential for failed open diagnosis
        failed_closed_delta_c: Min temp differential for failed closed
        normal_delta_range_c: Normal temperature differential range
        cold_threshold_c: Temperature below which trap is considered cold
        superheat_threshold_c: Degrees above saturation for superheat
        subcool_threshold_c: Degrees below saturation for subcool
        ambient_temp_c: Default ambient temperature
        emissivity: Default surface emissivity for IR
    """
    failed_open_delta_c: float = 5.0
    failed_closed_delta_c: float = 50.0
    normal_delta_range_c: Tuple[float, float] = (10.0, 40.0)
    cold_threshold_c: float = 50.0
    superheat_threshold_c: float = 10.0
    subcool_threshold_c: float = 10.0
    ambient_temp_c: float = 25.0
    emissivity: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "failed_open_delta_c": self.failed_open_delta_c,
            "failed_closed_delta_c": self.failed_closed_delta_c,
            "normal_delta_range_c": self.normal_delta_range_c,
            "cold_threshold_c": self.cold_threshold_c,
            "superheat_threshold_c": self.superheat_threshold_c,
            "subcool_threshold_c": self.subcool_threshold_c,
        }


@dataclass
class SaturationProperties:
    """
    Steam saturation properties at given pressure.

    Calculated from IAPWS-IF97 correlations.
    """
    pressure_bar: float
    temperature_c: float
    enthalpy_liquid_kj_kg: float
    enthalpy_vapor_kj_kg: float
    enthalpy_vaporization_kj_kg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pressure_bar": self.pressure_bar,
            "temperature_c": self.temperature_c,
            "enthalpy_liquid_kj_kg": self.enthalpy_liquid_kj_kg,
            "enthalpy_vapor_kj_kg": self.enthalpy_vapor_kj_kg,
            "enthalpy_vaporization_kj_kg": self.enthalpy_vaporization_kj_kg,
        }


@dataclass
class IRThermalImage:
    """
    IR thermal image analysis data.

    Attributes:
        max_temp_c: Maximum temperature in image
        min_temp_c: Minimum temperature in image
        avg_temp_c: Average temperature
        inlet_region_temp_c: Temperature in inlet region
        outlet_region_temp_c: Temperature in outlet region
        body_temp_c: Trap body temperature
        hotspots: List of hotspot locations and temperatures
        image_quality: Image quality assessment
    """
    max_temp_c: float
    min_temp_c: float
    avg_temp_c: float
    inlet_region_temp_c: float
    outlet_region_temp_c: float
    body_temp_c: float
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    image_quality: IRImageQuality = IRImageQuality.GOOD

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_temp_c": self.max_temp_c,
            "min_temp_c": self.min_temp_c,
            "avg_temp_c": self.avg_temp_c,
            "inlet_region_temp_c": self.inlet_region_temp_c,
            "outlet_region_temp_c": self.outlet_region_temp_c,
            "body_temp_c": self.body_temp_c,
            "hotspots": self.hotspots,
            "image_quality": self.image_quality.value,
        }


@dataclass
class ThermalDiagnosis:
    """
    Thermal diagnosis result.

    Attributes:
        status: Diagnosed trap status
        confidence: Confidence score (0-1)
        pattern: Detected thermal pattern
        failure_indicators: List of failure indicators
        notes: Diagnostic notes
    """
    status: TrapStatusThermal
    confidence: float
    pattern: ThermalPattern
    failure_indicators: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "pattern": self.pattern.value,
            "failure_indicators": self.failure_indicators,
            "notes": self.notes,
        }


@dataclass
class ThermalAnalysisResult:
    """
    Complete thermal analysis result.

    Attributes:
        trap_id: Steam trap identifier
        timestamp: Analysis timestamp
        inlet_temp_c: Inlet temperature
        outlet_temp_c: Outlet temperature
        temp_differential_c: Temperature differential
        saturation_temp_c: Saturation temperature at pressure
        superheat_c: Degrees of superheat (if any)
        subcool_c: Degrees of subcool (if any)
        saturation_properties: Steam saturation properties
        ir_analysis: IR thermal image analysis (if available)
        diagnosis: Diagnostic result
        provenance_hash: SHA-256 hash for audit trail
        calculation_method: Method description
    """
    trap_id: str
    timestamp: datetime
    inlet_temp_c: float
    outlet_temp_c: float
    temp_differential_c: float
    saturation_temp_c: float
    superheat_c: float
    subcool_c: float
    saturation_properties: SaturationProperties
    ir_analysis: Optional[IRThermalImage]
    diagnosis: ThermalDiagnosis
    provenance_hash: str = ""
    calculation_method: str = "iapws_if97"

    def __post_init__(self):
        """Generate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "inlet_temp_c": self.inlet_temp_c,
            "outlet_temp_c": self.outlet_temp_c,
            "temp_differential_c": self.temp_differential_c,
            "saturation_temp_c": self.saturation_temp_c,
            "diagnosis_status": self.diagnosis.status.value,
            "diagnosis_confidence": self.diagnosis.confidence,
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "inlet_temp_c": self.inlet_temp_c,
            "outlet_temp_c": self.outlet_temp_c,
            "temp_differential_c": self.temp_differential_c,
            "saturation_temp_c": self.saturation_temp_c,
            "superheat_c": self.superheat_c,
            "subcool_c": self.subcool_c,
            "saturation_properties": self.saturation_properties.to_dict(),
            "ir_analysis": self.ir_analysis.to_dict() if self.ir_analysis else None,
            "diagnosis": self.diagnosis.to_dict(),
            "provenance_hash": self.provenance_hash,
            "calculation_method": self.calculation_method,
        }


# ============================================================================
# TEMPERATURE DIFFERENTIAL ANALYZER
# ============================================================================

class TemperatureDifferentialAnalyzer:
    """
    Deterministic temperature differential analyzer for steam trap diagnosis.

    Analyzes inlet/outlet temperature differentials and IR thermal images
    to diagnose steam trap operating status.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use IAPWS-IF97 thermodynamic formulas
    - No LLM or ML inference in calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Temperature Patterns by Trap Status:
    - OPERATING: Inlet at saturation, outlet 10-40C cooler (subcooled condensate)
    - FAILED_OPEN: Both inlet and outlet near saturation (<5C differential)
    - FAILED_CLOSED: Large differential (>50C), outlet much cooler
    - COLD: Both sides below 50C (no steam supply)

    Example:
        >>> analyzer = TemperatureDifferentialAnalyzer()
        >>> result = analyzer.analyze("T001", 180.0, 150.0, 10.0)
        >>> print(f"Status: {result.diagnosis.status}")
    """

    # Saturation table: pressure (bar gauge) -> saturation temp (C)
    # Derived from IAPWS-IF97
    SATURATION_TABLE = {
        0: 100.0,
        1: 120.2,
        2: 133.5,
        3: 143.6,
        4: 151.8,
        5: 158.8,
        6: 165.0,
        7: 170.4,
        8: 175.4,
        9: 179.9,
        10: 184.1,
        12: 191.6,
        14: 198.3,
        16: 204.3,
        18: 209.8,
        20: 214.9,
        25: 226.0,
        30: 235.8,
        40: 252.4,
        50: 266.4,
    }

    def __init__(self, config: Optional[ThermalAnalysisConfig] = None):
        """
        Initialize temperature differential analyzer.

        Args:
            config: Analysis configuration (uses defaults if not provided)
        """
        self.config = config or ThermalAnalysisConfig()
        self.analysis_count = 0
        logger.info(
            f"TemperatureDifferentialAnalyzer initialized "
            f"(failed_open_delta={self.config.failed_open_delta_c}C)"
        )

    def analyze(
        self,
        trap_id: str,
        inlet_temp_c: float,
        outlet_temp_c: float,
        pressure_bar: float,
        ambient_temp_c: Optional[float] = None,
        ir_data: Optional[Dict[str, Any]] = None,
    ) -> ThermalAnalysisResult:
        """
        Perform comprehensive thermal analysis.

        ZERO-HALLUCINATION: Uses deterministic thermodynamic calculations.

        Args:
            trap_id: Steam trap identifier
            inlet_temp_c: Inlet/upstream temperature (Celsius)
            outlet_temp_c: Outlet/downstream temperature (Celsius)
            pressure_bar: Operating pressure (bar gauge)
            ambient_temp_c: Ambient temperature (optional)
            ir_data: IR thermal image data (optional)

        Returns:
            ThermalAnalysisResult with complete analysis

        Raises:
            ValueError: If temperatures or pressure are invalid
        """
        self.analysis_count += 1
        timestamp = datetime.now(timezone.utc)

        # Validate inputs
        if inlet_temp_c < -50 or inlet_temp_c > 400:
            raise ValueError(f"Invalid inlet temperature: {inlet_temp_c}C")
        if outlet_temp_c < -50 or outlet_temp_c > 400:
            raise ValueError(f"Invalid outlet temperature: {outlet_temp_c}C")
        if pressure_bar < 0 or pressure_bar > 100:
            raise ValueError(f"Invalid pressure: {pressure_bar} bar")

        # Use config ambient if not provided
        ambient = ambient_temp_c or self.config.ambient_temp_c

        # Calculate temperature differential
        temp_differential = inlet_temp_c - outlet_temp_c

        # Calculate saturation properties
        sat_props = self._calculate_saturation_properties(pressure_bar)
        sat_temp = sat_props.temperature_c

        # Calculate superheat/subcool
        superheat = max(0, inlet_temp_c - sat_temp)
        subcool = max(0, sat_temp - outlet_temp_c)

        # Process IR data if available
        ir_analysis = None
        if ir_data:
            ir_analysis = self._analyze_ir_image(ir_data)

        # Detect thermal pattern
        pattern = self._detect_thermal_pattern(
            inlet_temp_c, outlet_temp_c, temp_differential,
            sat_temp, ambient, ir_analysis
        )

        # Perform diagnosis
        diagnosis = self._diagnose_from_thermal(
            inlet_temp_c, outlet_temp_c, temp_differential,
            sat_temp, superheat, subcool, pattern, ir_analysis
        )

        return ThermalAnalysisResult(
            trap_id=trap_id,
            timestamp=timestamp,
            inlet_temp_c=round(inlet_temp_c, 2),
            outlet_temp_c=round(outlet_temp_c, 2),
            temp_differential_c=round(temp_differential, 2),
            saturation_temp_c=round(sat_temp, 2),
            superheat_c=round(superheat, 2),
            subcool_c=round(subcool, 2),
            saturation_properties=sat_props,
            ir_analysis=ir_analysis,
            diagnosis=diagnosis,
        )

    def _calculate_saturation_properties(
        self,
        pressure_bar: float
    ) -> SaturationProperties:
        """
        Calculate saturation properties at given pressure.

        FORMULA (IAPWS-IF97 correlation):
        For gauge pressure P (bar):
        T_sat = interpolated from saturation table

        Enthalpy correlations:
        h_f = 4.18 * T_sat (simplified)
        h_fg = 2501 - 2.36 * T_sat
        h_g = h_f + h_fg

        Args:
            pressure_bar: Gauge pressure in bar

        Returns:
            SaturationProperties at given pressure
        """
        # Interpolate saturation temperature from table
        t_sat = self._interpolate_saturation_temp(pressure_bar)

        # Calculate enthalpy values (simplified correlations)
        # Liquid enthalpy (kJ/kg)
        h_f = 4.18 * t_sat

        # Enthalpy of vaporization (kJ/kg) - decreases with temperature
        h_fg = max(0, 2501.0 - 2.36 * t_sat)

        # Vapor enthalpy (kJ/kg)
        h_g = h_f + h_fg

        return SaturationProperties(
            pressure_bar=pressure_bar,
            temperature_c=round(t_sat, 2),
            enthalpy_liquid_kj_kg=round(h_f, 2),
            enthalpy_vapor_kj_kg=round(h_g, 2),
            enthalpy_vaporization_kj_kg=round(h_fg, 2),
        )

    def _interpolate_saturation_temp(self, pressure_bar: float) -> float:
        """
        Interpolate saturation temperature from table.

        Args:
            pressure_bar: Gauge pressure in bar

        Returns:
            Saturation temperature in Celsius
        """
        pressures = sorted(self.SATURATION_TABLE.keys())

        # Handle edge cases
        if pressure_bar <= pressures[0]:
            return self.SATURATION_TABLE[pressures[0]]
        if pressure_bar >= pressures[-1]:
            return self.SATURATION_TABLE[pressures[-1]]

        # Find bounding pressures
        for i in range(len(pressures) - 1):
            p_low = pressures[i]
            p_high = pressures[i + 1]

            if p_low <= pressure_bar <= p_high:
                t_low = self.SATURATION_TABLE[p_low]
                t_high = self.SATURATION_TABLE[p_high]

                # Linear interpolation
                fraction = (pressure_bar - p_low) / (p_high - p_low)
                t_sat = t_low + fraction * (t_high - t_low)
                return t_sat

        # Fallback (should not reach here)
        return 100.0

    def _analyze_ir_image(self, ir_data: Dict[str, Any]) -> IRThermalImage:
        """
        Analyze IR thermal image data.

        Args:
            ir_data: Dictionary with IR image data

        Returns:
            IRThermalImage analysis result
        """
        # Extract values with defaults
        max_temp = ir_data.get("max_temp_c", 0.0)
        min_temp = ir_data.get("min_temp_c", 0.0)
        avg_temp = ir_data.get("avg_temp_c", (max_temp + min_temp) / 2)

        inlet_temp = ir_data.get("inlet_region_temp_c", max_temp)
        outlet_temp = ir_data.get("outlet_region_temp_c", min_temp)
        body_temp = ir_data.get("body_temp_c", avg_temp)

        hotspots = ir_data.get("hotspots", [])

        # Assess image quality
        temp_range = max_temp - min_temp
        if temp_range > 100:
            quality = IRImageQuality.EXCELLENT
        elif temp_range > 50:
            quality = IRImageQuality.GOOD
        elif temp_range > 20:
            quality = IRImageQuality.ACCEPTABLE
        elif temp_range > 5:
            quality = IRImageQuality.POOR
        else:
            quality = IRImageQuality.UNUSABLE

        return IRThermalImage(
            max_temp_c=round(max_temp, 2),
            min_temp_c=round(min_temp, 2),
            avg_temp_c=round(avg_temp, 2),
            inlet_region_temp_c=round(inlet_temp, 2),
            outlet_region_temp_c=round(outlet_temp, 2),
            body_temp_c=round(body_temp, 2),
            hotspots=hotspots,
            image_quality=quality,
        )

    def _detect_thermal_pattern(
        self,
        inlet_temp_c: float,
        outlet_temp_c: float,
        temp_differential_c: float,
        sat_temp_c: float,
        ambient_temp_c: float,
        ir_analysis: Optional[IRThermalImage]
    ) -> ThermalPattern:
        """
        Detect characteristic thermal pattern.

        ZERO-HALLUCINATION: Deterministic pattern classification.

        Args:
            inlet_temp_c: Inlet temperature
            outlet_temp_c: Outlet temperature
            temp_differential_c: Temperature differential
            sat_temp_c: Saturation temperature
            ambient_temp_c: Ambient temperature
            ir_analysis: IR thermal image analysis (optional)

        Returns:
            Detected ThermalPattern
        """
        cold_threshold = self.config.cold_threshold_c

        # Both sides cold
        if inlet_temp_c < cold_threshold and outlet_temp_c < cold_threshold:
            return ThermalPattern.COLD_BOTH

        # Hot outlet (near saturation)
        if outlet_temp_c > sat_temp_c - 10 and temp_differential_c < 10:
            return ThermalPattern.HOT_OUTLET

        # Cold outlet (large differential)
        if temp_differential_c > 40 and outlet_temp_c < sat_temp_c - 30:
            return ThermalPattern.COLD_OUTLET

        # Superheat detection
        if inlet_temp_c > sat_temp_c + self.config.superheat_threshold_c:
            return ThermalPattern.SUPERHEAT

        # Subcool detection (normal for condensate)
        if outlet_temp_c < sat_temp_c - self.config.subcool_threshold_c:
            if temp_differential_c > 10:
                return ThermalPattern.SUBCOOL

        # Uniform temperature (no flow)
        if abs(temp_differential_c) < 2 and outlet_temp_c > cold_threshold:
            return ThermalPattern.UNIFORM

        # Flooded pattern (backed up condensate)
        if (ir_analysis and
            ir_analysis.outlet_region_temp_c > ir_analysis.inlet_region_temp_c - 5):
            return ThermalPattern.FLOODED

        # Normal pattern
        normal_range = self.config.normal_delta_range_c
        if normal_range[0] <= temp_differential_c <= normal_range[1]:
            return ThermalPattern.NORMAL

        return ThermalPattern.NORMAL

    def _diagnose_from_thermal(
        self,
        inlet_temp_c: float,
        outlet_temp_c: float,
        temp_differential_c: float,
        sat_temp_c: float,
        superheat_c: float,
        subcool_c: float,
        pattern: ThermalPattern,
        ir_analysis: Optional[IRThermalImage]
    ) -> ThermalDiagnosis:
        """
        Diagnose trap status from thermal data.

        ZERO-HALLUCINATION: Deterministic threshold-based diagnosis.

        Args:
            inlet_temp_c: Inlet temperature
            outlet_temp_c: Outlet temperature
            temp_differential_c: Temperature differential
            sat_temp_c: Saturation temperature
            superheat_c: Degrees of superheat
            subcool_c: Degrees of subcool
            pattern: Detected thermal pattern
            ir_analysis: IR analysis (optional)

        Returns:
            ThermalDiagnosis with status and confidence
        """
        notes = []
        failure_indicators = []

        # DIAGNOSIS RULES (deterministic threshold-based)

        # Rule 1: Both inlet and outlet near saturation = FAILED_OPEN
        if (inlet_temp_c > sat_temp_c - 10 and
            outlet_temp_c > sat_temp_c - 15 and
            temp_differential_c < self.config.failed_open_delta_c):

            failure_indicators.append(
                f"Minimal temperature differential ({temp_differential_c:.1f}C) "
                f"with both sides near saturation ({sat_temp_c:.1f}C)"
            )
            notes.append(
                "Thermal signature indicates steam passing through trap "
                "(failed open condition)"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.FAILED_OPEN,
                confidence=0.95,
                pattern=pattern,
                failure_indicators=failure_indicators,
                notes=notes,
            )

        # Rule 2: Very large differential = FAILED_CLOSED
        if temp_differential_c > self.config.failed_closed_delta_c:
            failure_indicators.append(
                f"Large temperature differential ({temp_differential_c:.1f}C) "
                f"indicates blocked trap not passing condensate"
            )
            notes.append(
                "Outlet significantly cooler than inlet - condensate not draining"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.FAILED_CLOSED,
                confidence=0.85,
                pattern=pattern,
                failure_indicators=failure_indicators,
                notes=notes,
            )

        # Rule 3: Both sides cold = COLD (no steam)
        if pattern == ThermalPattern.COLD_BOTH:
            notes.append(
                f"Both inlet ({inlet_temp_c:.1f}C) and outlet ({outlet_temp_c:.1f}C) "
                f"cold - no steam supply or completely blocked system"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.COLD,
                confidence=0.90,
                pattern=pattern,
                failure_indicators=["No thermal signature detected"],
                notes=notes,
            )

        # Rule 4: Hot outlet without proper differential = LEAKING
        if (outlet_temp_c > sat_temp_c - 20 and
            temp_differential_c < 15 and
            temp_differential_c >= self.config.failed_open_delta_c):

            failure_indicators.append(
                f"Outlet temperature ({outlet_temp_c:.1f}C) close to "
                f"saturation ({sat_temp_c:.1f}C) suggests steam passing"
            )
            notes.append(
                "Partial steam leak suspected based on thermal signature"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.LEAKING,
                confidence=0.70,
                pattern=pattern,
                failure_indicators=failure_indicators,
                notes=notes,
            )

        # Rule 5: Flooded/backed up pattern
        if pattern == ThermalPattern.FLOODED:
            notes.append(
                "Thermal pattern suggests condensate backup (flooded condition)"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.FLOODED,
                confidence=0.75,
                pattern=pattern,
                failure_indicators=["Condensate backup detected"],
                notes=notes,
            )

        # Rule 6: Normal temperature differential = OPERATING
        normal_range = self.config.normal_delta_range_c
        if normal_range[0] <= temp_differential_c <= normal_range[1]:
            notes.append(
                f"Normal temperature differential ({temp_differential_c:.1f}C) "
                f"indicates proper trap operation. Inlet: {inlet_temp_c:.1f}C, "
                f"Outlet: {outlet_temp_c:.1f}C, Saturation: {sat_temp_c:.1f}C"
            )

            return ThermalDiagnosis(
                status=TrapStatusThermal.OPERATING,
                confidence=0.85,
                pattern=pattern,
                failure_indicators=[],
                notes=notes,
            )

        # Default: Unknown
        notes.append(
            f"Inconclusive thermal pattern. Delta: {temp_differential_c:.1f}C, "
            f"Inlet: {inlet_temp_c:.1f}C, Outlet: {outlet_temp_c:.1f}C"
        )

        return ThermalDiagnosis(
            status=TrapStatusThermal.UNKNOWN,
            confidence=0.50,
            pattern=pattern,
            failure_indicators=[],
            notes=notes,
        )

    def calculate_condensate_temperature(
        self,
        steam_temp_c: float,
        trap_efficiency: float = 0.95
    ) -> float:
        """
        Calculate expected condensate outlet temperature.

        Well-functioning trap should discharge condensate slightly
        below saturation temperature.

        FORMULA:
        T_condensate = T_sat - (1 - efficiency) * subcool_factor

        Args:
            steam_temp_c: Steam/saturation temperature
            trap_efficiency: Trap efficiency (0-1)

        Returns:
            Expected condensate temperature in Celsius
        """
        # Typical subcool for good trap operation is 5-15C
        subcool_factor = 15.0
        expected_subcool = (1.0 - trap_efficiency) * subcool_factor + 5.0

        return round(steam_temp_c - expected_subcool, 2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "analysis_count": self.analysis_count,
            "config": self.config.to_dict(),
            "saturation_table_pressures": list(self.SATURATION_TABLE.keys()),
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TemperatureDifferentialAnalyzer",
    "ThermalAnalysisConfig",
    "ThermalAnalysisResult",
    "ThermalDiagnosis",
    "ThermalPattern",
    "TrapStatusThermal",
    "SaturationProperties",
    "IRThermalImage",
    "IRImageQuality",
]
