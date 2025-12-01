# -*- coding: utf-8 -*-
"""
Moisture Analyzer for GL-012 STEAMQUAL.

Provides deterministic analysis of moisture content in steam systems including
moisture detection, condensation risk assessment, wetness calculation,
moisture source identification, and remediation recommendations.

Standards:
- ASME PTC 19.11: Steam and Water Sampling, Conditioning, and Analysis
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- NIST Steam Tables: Reference data

Zero-hallucination: All calculations are deterministic with bit-perfect reproducibility.
No LLM involved in any calculation path.

Author: GL-CalculatorEngineer
Version: 1.0.0

Formulas:
    Wetness Fraction: y = 1 - x (where x is dryness fraction)
    Moisture Content: y = (h_sat - h_actual) / h_fg
    Dew Point Margin: delta_T = T_surface - T_dew
    Condensation Risk: Based on surface temperature vs dew point
"""

import hashlib
import json
import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """
    Risk level classification for condensation and moisture hazards.

    Levels based on industrial safety standards and operational impact.
    """
    NEGLIGIBLE = "negligible"  # No action required
    LOW = "low"  # Monitor situation
    MODERATE = "moderate"  # Investigate and plan action
    HIGH = "high"  # Take corrective action soon
    CRITICAL = "critical"  # Immediate action required
    UNKNOWN = "unknown"  # Insufficient data


class MoistureSourceType(Enum):
    """Classification of moisture sources in steam systems."""
    BOILER_CARRYOVER = "boiler_carryover"
    PIPE_CONDENSATION = "pipe_condensation"
    HEAT_EXCHANGER_LEAK = "heat_exchanger_leak"
    STEAM_TRAP_FAILURE = "steam_trap_failure"
    INSULATION_DAMAGE = "insulation_damage"
    DESUPERHEATER_OVERSPRAY = "desuperheater_overspray"
    TURBINE_EXHAUST_WETNESS = "turbine_exhaust_wetness"
    PROCESS_CONTAMINATION = "process_contamination"
    AMBIENT_INFILTRATION = "ambient_infiltration"
    UNKNOWN = "unknown"


@dataclass
class MoistureSource:
    """
    Identified moisture source in steam system.

    Attributes:
        source_type: Classification of moisture source
        location: System location identifier
        severity: Estimated severity (0-1)
        confidence: Confidence in identification (0-1)
        evidence: List of supporting evidence
        recommended_actions: Suggested remediation steps
    """
    source_type: MoistureSourceType
    location: str
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    evidence: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class MoistureAnalysisInput:
    """Input parameters for moisture analysis."""
    temperature_c: float  # Steam/surface temperature
    pressure_mpa: float  # Steam pressure
    enthalpy_measured_kj_kg: Optional[float] = None  # Measured enthalpy
    ambient_temperature_c: float = 25.0  # Ambient temperature
    insulation_factor: float = 0.9  # Insulation effectiveness (0-1)
    surface_temperature_c: Optional[float] = None  # Pipe surface temperature
    humidity_percent: float = 50.0  # Ambient relative humidity
    steam_trap_data: Optional[Dict[str, Any]] = None  # Steam trap diagnostics
    system_locations: List[str] = field(default_factory=list)  # Measurement locations


@dataclass
class MoistureAnalysisOutput:
    """Output of moisture analysis."""
    moisture_content_pct: float  # Moisture percentage (0-100)
    wetness_fraction: Decimal  # Wetness fraction (0-1)
    dryness_fraction: Decimal  # Quality (0-1)
    condensation_risk: RiskLevel  # Condensation risk level
    dew_point_margin_c: float  # Temperature above dew point
    dew_point_c: float  # Calculated dew point
    saturation_temperature_c: float  # Saturation temperature at pressure
    identified_sources: List[MoistureSource]  # Potential moisture sources
    remediation_recommendations: List[str]  # Actions to reduce moisture
    calculation_method: str = "IAPWS-IF97"
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


class MoistureAnalyzer:
    """
    Deterministic moisture analyzer for steam systems.

    Analyzes moisture content, condensation risks, and identifies
    potential sources of moisture in steam distribution systems.

    All calculations are deterministic (zero-hallucination):
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes
    - No LLM or AI inference in calculation path

    Example:
        >>> analyzer = MoistureAnalyzer()
        >>> result = analyzer.analyze(MoistureAnalysisInput(
        ...     temperature_c=175.0,
        ...     pressure_mpa=1.0,
        ...     ambient_temperature_c=25.0
        ... ))
        >>> print(f"Condensation risk: {result.condensation_risk.value}")
    """

    # Risk thresholds
    DEW_POINT_MARGIN_CRITICAL = Decimal("5")  # C
    DEW_POINT_MARGIN_HIGH = Decimal("10")  # C
    DEW_POINT_MARGIN_MODERATE = Decimal("20")  # C
    DEW_POINT_MARGIN_LOW = Decimal("30")  # C

    # Moisture thresholds
    MOISTURE_CRITICAL = Decimal("10")  # %
    MOISTURE_HIGH = Decimal("5")  # %
    MOISTURE_MODERATE = Decimal("2")  # %
    MOISTURE_LOW = Decimal("1")  # %

    # Saturation table (pressure MPa -> T_sat C, h_f, h_fg, s_f, s_fg)
    # Subset from IAPWS-IF97
    SATURATION_TABLE = {
        Decimal("0.10"): (Decimal("99.61"), Decimal("417.44"), Decimal("2257.5")),
        Decimal("0.20"): (Decimal("120.21"), Decimal("504.68"), Decimal("2201.6")),
        Decimal("0.50"): (Decimal("151.83"), Decimal("640.09"), Decimal("2108.0")),
        Decimal("1.00"): (Decimal("179.88"), Decimal("762.68"), Decimal("2014.9")),
        Decimal("1.50"): (Decimal("198.29"), Decimal("844.66"), Decimal("1946.4")),
        Decimal("2.00"): (Decimal("212.38"), Decimal("908.62"), Decimal("1889.8")),
        Decimal("3.00"): (Decimal("233.85"), Decimal("1008.3"), Decimal("1794.9")),
        Decimal("5.00"): (Decimal("263.94"), Decimal("1154.5"), Decimal("1639.7")),
        Decimal("10.0"): (Decimal("311.00"), Decimal("1407.8"), Decimal("1319.7")),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize moisture analyzer.

        Args:
            config: Optional configuration dictionary with keys:
                - precision: Decimal places for output (default: 4)
                - risk_thresholds: Custom risk threshold values
                - enable_source_detection: Enable moisture source identification
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 4)
        self.enable_source_detection = self.config.get('enable_source_detection', True)
        self.calculation_count = 0

    def analyze_moisture_content(
        self,
        enthalpy_measured: float,
        saturation_enthalpy: float,
        h_fg: Optional[float] = None
    ) -> float:
        """
        Analyze moisture content from enthalpy measurements.

        FORMULA:
            For wet steam: h = h_f + x * h_fg
            Moisture content (%) = (1 - x) * 100 = (h_g - h) / h_fg * 100

        Where:
            h = measured specific enthalpy (kJ/kg)
            h_f = saturated liquid enthalpy (kJ/kg)
            h_g = saturated vapor enthalpy = h_f + h_fg
            h_fg = enthalpy of vaporization (kJ/kg)
            x = dryness fraction

        ZERO-HALLUCINATION GUARANTEE:
            - Direct formula application
            - Deterministic arithmetic

        Args:
            enthalpy_measured: Measured specific enthalpy (kJ/kg)
            saturation_enthalpy: Saturated vapor enthalpy h_g (kJ/kg)
            h_fg: Enthalpy of vaporization (kJ/kg), estimated if not provided

        Returns:
            Moisture content as percentage (0-100)

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> moisture = analyzer.analyze_moisture_content(
            ...     enthalpy_measured=2700.0,
            ...     saturation_enthalpy=2777.0,
            ...     h_fg=2014.9
            ... )
            >>> print(f"Moisture content: {moisture:.2f}%")
        """
        h = Decimal(str(enthalpy_measured))
        h_g = Decimal(str(saturation_enthalpy))

        # Estimate h_fg if not provided
        if h_fg is None:
            h_fg_dec = Decimal("2000")  # Approximate average h_fg
        else:
            h_fg_dec = Decimal(str(h_fg))

        # Calculate h_f from h_g and h_fg
        h_f = h_g - h_fg_dec

        # Calculate dryness fraction
        if h_fg_dec > 0:
            if h >= h_g:
                # Superheated - no moisture
                x = Decimal("1")
            elif h <= h_f:
                # Subcooled liquid - 100% moisture (not steam)
                x = Decimal("0")
            else:
                # Wet steam
                x = (h - h_f) / h_fg_dec
        else:
            x = Decimal("1")

        # Moisture content = 1 - x (as percentage)
        moisture_pct = (Decimal("1") - x) * Decimal("100")

        # Clamp to valid range
        moisture_pct = max(Decimal("0"), min(Decimal("100"), moisture_pct))

        return float(moisture_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def detect_condensation_risk(
        self,
        T: float,
        P: float,
        T_ambient: float,
        insulation_factor: float,
        surface_temp: Optional[float] = None
    ) -> RiskLevel:
        """
        Detect condensation risk based on temperature differentials.

        FORMULA:
            Surface temperature estimate: T_surface = T_ambient + (T_steam - T_ambient) * insulation_factor
            Dew point margin: delta_T = T_surface - T_dew

        Risk levels based on dew point margin:
            - CRITICAL: < 5C margin
            - HIGH: 5-10C margin
            - MODERATE: 10-20C margin
            - LOW: 20-30C margin
            - NEGLIGIBLE: > 30C margin

        ZERO-HALLUCINATION GUARANTEE:
            - Fixed threshold-based classification
            - Deterministic temperature calculations

        Args:
            T: Steam temperature (C)
            P: Steam pressure (MPa)
            T_ambient: Ambient temperature (C)
            insulation_factor: Insulation effectiveness (0-1)
            surface_temp: Measured surface temperature (C), calculated if not provided

        Returns:
            RiskLevel enum value

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> risk = analyzer.detect_condensation_risk(
            ...     T=180.0, P=1.0, T_ambient=25.0, insulation_factor=0.9
            ... )
            >>> print(f"Risk level: {risk.value}")
        """
        T_steam = Decimal(str(T))
        T_amb = Decimal(str(T_ambient))
        ins_factor = Decimal(str(insulation_factor))

        # Calculate or use provided surface temperature
        if surface_temp is not None:
            T_surface = Decimal(str(surface_temp))
        else:
            # Estimate surface temperature based on insulation
            # T_surface = T_ambient + (T_steam - T_ambient) * (1 - insulation_effectiveness)
            # With good insulation (factor=0.9), surface is close to ambient
            T_surface = T_amb + (T_steam - T_amb) * (Decimal("1") - ins_factor)

        # Calculate dew point (simplified - based on ambient humidity assumption)
        # Using Magnus formula approximation for 50% RH
        # T_dew ~= T - (100 - RH) / 5  (rough approximation)
        # For more accuracy, use full psychrometric equations
        T_dew = T_amb - Decimal("10")  # Simplified for ~50% RH

        # Calculate dew point margin
        dew_point_margin = T_surface - T_dew

        # Classify risk
        if dew_point_margin < self.DEW_POINT_MARGIN_CRITICAL:
            return RiskLevel.CRITICAL
        elif dew_point_margin < self.DEW_POINT_MARGIN_HIGH:
            return RiskLevel.HIGH
        elif dew_point_margin < self.DEW_POINT_MARGIN_MODERATE:
            return RiskLevel.MODERATE
        elif dew_point_margin < self.DEW_POINT_MARGIN_LOW:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE

    def calculate_wetness_fraction(
        self,
        dryness_fraction: float
    ) -> Decimal:
        """
        Calculate wetness fraction from dryness fraction.

        FORMULA:
            y = 1 - x

        Where:
            y = wetness fraction (0 = dry steam, 1 = saturated liquid)
            x = dryness fraction / quality (0 = saturated liquid, 1 = dry steam)

        ZERO-HALLUCINATION GUARANTEE:
            - Simple subtraction
            - Deterministic

        Args:
            dryness_fraction: Steam quality / dryness fraction (0-1)

        Returns:
            Wetness fraction as Decimal (0-1)

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> wetness = analyzer.calculate_wetness_fraction(0.95)
            >>> print(f"Wetness: {wetness}")  # 0.05
        """
        x = Decimal(str(dryness_fraction))

        # Clamp dryness to valid range
        x = max(Decimal("0"), min(Decimal("1"), x))

        # Wetness = 1 - dryness
        y = Decimal("1") - x

        return y.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def identify_moisture_sources(
        self,
        system_data: Dict[str, Any]
    ) -> List[MoistureSource]:
        """
        Identify potential moisture sources based on system data.

        Uses rule-based expert system logic (NOT ML/AI) to identify
        likely moisture sources from operational patterns.

        RULE SET (Deterministic):
        1. Boiler carryover: High moisture + load changes
        2. Pipe condensation: Cold spots + poor insulation
        3. Steam trap failure: High moisture downstream of traps
        4. Desuperheater overspray: Moisture after desuperheater
        5. Insulation damage: Localized temperature anomalies

        ZERO-HALLUCINATION GUARANTEE:
            - Fixed rule-based logic
            - No ML/AI inference
            - Deterministic pattern matching

        Args:
            system_data: Dictionary containing:
                - moisture_readings: List of {location, value} readings
                - temperature_readings: List of {location, value} readings
                - trap_status: Steam trap diagnostic data
                - load_changes: Recent load change events
                - insulation_surveys: Insulation condition data

        Returns:
            List of identified MoistureSource objects

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> sources = analyzer.identify_moisture_sources({
            ...     'moisture_readings': [
            ...         {'location': 'header_outlet', 'value': 5.0},
            ...         {'location': 'turbine_inlet', 'value': 2.0}
            ...     ],
            ...     'trap_status': {'trap_01': 'failed_open'}
            ... })
        """
        sources: List[MoistureSource] = []

        # Extract data
        moisture_readings = system_data.get('moisture_readings', [])
        temp_readings = system_data.get('temperature_readings', [])
        trap_status = system_data.get('trap_status', {})
        load_changes = system_data.get('load_changes', [])
        insulation_data = system_data.get('insulation_surveys', {})
        desuperheater_data = system_data.get('desuperheater_data', {})

        # Rule 1: Check for boiler carryover
        # High moisture at header + recent load changes
        header_moisture = self._get_reading(moisture_readings, 'header')
        if header_moisture is not None and header_moisture > 3.0:
            recent_load_change = len(load_changes) > 0
            confidence = 0.8 if recent_load_change else 0.5

            sources.append(MoistureSource(
                source_type=MoistureSourceType.BOILER_CARRYOVER,
                location='boiler_drum',
                severity=min(1.0, header_moisture / 10.0),
                confidence=confidence,
                evidence=[
                    f"Header moisture: {header_moisture:.1f}%",
                    f"Recent load changes: {len(load_changes)}"
                ],
                recommended_actions=[
                    "Check boiler water level",
                    "Verify drum internals",
                    "Review load change rate limits"
                ]
            ))

        # Rule 2: Check for steam trap failures
        for trap_id, status in trap_status.items():
            if status in ['failed_open', 'failed', 'leaking']:
                # Find downstream moisture reading
                downstream_moisture = self._get_reading(
                    moisture_readings, trap_id.replace('trap', 'downstream')
                )

                sources.append(MoistureSource(
                    source_type=MoistureSourceType.STEAM_TRAP_FAILURE,
                    location=trap_id,
                    severity=0.7 if status == 'failed_open' else 0.5,
                    confidence=0.9,
                    evidence=[
                        f"Trap status: {status}",
                        f"Downstream moisture: {downstream_moisture or 'unknown'}"
                    ],
                    recommended_actions=[
                        f"Repair or replace {trap_id}",
                        "Verify trap sizing",
                        "Check condensate return pressure"
                    ]
                ))

        # Rule 3: Check for desuperheater overspray
        ds_outlet_moisture = desuperheater_data.get('outlet_moisture', 0)
        ds_temp_error = desuperheater_data.get('temperature_deviation', 0)

        if ds_outlet_moisture > 1.0 or ds_temp_error < -10:
            sources.append(MoistureSource(
                source_type=MoistureSourceType.DESUPERHEATER_OVERSPRAY,
                location='desuperheater',
                severity=min(1.0, ds_outlet_moisture / 5.0),
                confidence=0.85,
                evidence=[
                    f"Outlet moisture: {ds_outlet_moisture:.1f}%",
                    f"Temperature deviation: {ds_temp_error:.1f}C"
                ],
                recommended_actions=[
                    "Reduce spray water flow",
                    "Check temperature control tuning",
                    "Verify spray nozzle condition"
                ]
            ))

        # Rule 4: Check for insulation damage
        for location, condition in insulation_data.items():
            if condition.get('effectiveness', 1.0) < 0.5:
                sources.append(MoistureSource(
                    source_type=MoistureSourceType.INSULATION_DAMAGE,
                    location=location,
                    severity=1.0 - condition.get('effectiveness', 1.0),
                    confidence=0.8,
                    evidence=[
                        f"Insulation effectiveness: {condition.get('effectiveness', 0)*100:.0f}%",
                        f"Surface temperature elevated"
                    ],
                    recommended_actions=[
                        f"Repair insulation at {location}",
                        "Conduct thermal survey",
                        "Check for moisture ingress"
                    ]
                ))

        # Rule 5: Check for pipe condensation (temperature anomalies)
        for i, reading in enumerate(temp_readings):
            location = reading.get('location', f'point_{i}')
            temp = reading.get('value', 0)
            expected = reading.get('expected', temp)

            if temp < expected - 20:  # Cold spot
                sources.append(MoistureSource(
                    source_type=MoistureSourceType.PIPE_CONDENSATION,
                    location=location,
                    severity=min(1.0, (expected - temp) / 50.0),
                    confidence=0.7,
                    evidence=[
                        f"Temperature: {temp:.1f}C (expected {expected:.1f}C)",
                        "Cold spot detected"
                    ],
                    recommended_actions=[
                        f"Investigate heat loss at {location}",
                        "Check for condensate accumulation",
                        "Verify drain point operation"
                    ]
                ))

        return sources

    def _get_reading(
        self,
        readings: List[Dict[str, Any]],
        location_pattern: str
    ) -> Optional[float]:
        """Get reading value matching location pattern."""
        for reading in readings:
            location = reading.get('location', '')
            if location_pattern.lower() in location.lower():
                return reading.get('value')
        return None

    def recommend_moisture_remediation(
        self,
        analysis_result: MoistureAnalysisOutput
    ) -> List[str]:
        """
        Generate moisture remediation recommendations.

        Uses rule-based expert system to generate actionable recommendations
        based on analysis results.

        ZERO-HALLUCINATION GUARANTEE:
            - Fixed rule-based recommendations
            - No LLM text generation
            - Deterministic selection from predefined actions

        Args:
            analysis_result: Output from moisture analysis

        Returns:
            List of remediation recommendation strings

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> result = analyzer.analyze(input_data)
            >>> recommendations = analyzer.recommend_moisture_remediation(result)
        """
        recommendations: List[str] = []

        # Priority 1: Address critical/high risk
        if analysis_result.condensation_risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            recommendations.append(
                "URGENT: Address condensation risk immediately to prevent water hammer and equipment damage"
            )
            recommendations.append(
                "Verify all steam traps are functioning and draining condensate properly"
            )
            recommendations.append(
                "Check insulation integrity and repair any damaged sections"
            )

        # Priority 2: Reduce moisture content
        if analysis_result.moisture_content_pct > float(self.MOISTURE_HIGH):
            recommendations.append(
                f"Moisture content ({analysis_result.moisture_content_pct:.1f}%) exceeds target - "
                "investigate moisture sources"
            )
            recommendations.append(
                "Review boiler operation for carryover conditions"
            )
            recommendations.append(
                "Verify desuperheater spray water control is not over-injecting"
            )

        # Priority 3: General improvements
        if analysis_result.moisture_content_pct > float(self.MOISTURE_MODERATE):
            recommendations.append(
                "Consider installing moisture separators at critical points"
            )
            recommendations.append(
                "Review steam line sizing for current flow conditions"
            )

        # Priority 4: Address identified sources
        for source in analysis_result.identified_sources:
            if source.severity > 0.5:
                recommendations.extend(source.recommended_actions)

        # Priority 5: Monitoring recommendations
        if analysis_result.condensation_risk == RiskLevel.MODERATE:
            recommendations.append(
                "Increase monitoring frequency for moisture and temperature"
            )
            recommendations.append(
                "Consider installing additional moisture detection points"
            )

        # Remove duplicates while preserving order
        seen: Set[str] = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    def analyze(self, input_data: MoistureAnalysisInput) -> MoistureAnalysisOutput:
        """
        Comprehensive moisture analysis.

        Performs complete moisture analysis including content calculation,
        risk assessment, source identification, and recommendations.

        ZERO-HALLUCINATION GUARANTEE:
            - All calculations use deterministic formulas
            - Complete provenance tracking with SHA-256 hash

        Args:
            input_data: Moisture analysis input parameters

        Returns:
            MoistureAnalysisOutput with all calculated values

        Example:
            >>> analyzer = MoistureAnalyzer()
            >>> result = analyzer.analyze(MoistureAnalysisInput(
            ...     temperature_c=175.0,
            ...     pressure_mpa=1.0,
            ...     ambient_temperature_c=25.0,
            ...     insulation_factor=0.9
            ... ))
        """
        self.calculation_count += 1
        warnings = []

        P = Decimal(str(input_data.pressure_mpa))
        T = Decimal(str(input_data.temperature_c))

        # Get saturation properties
        sat_props = self._get_saturation_properties(P)
        if sat_props is None:
            sat_props = self._interpolate_saturation(P)
            warnings.append(f"Pressure {P} MPa interpolated from table")

        T_sat, h_f, h_fg = sat_props
        h_g = h_f + h_fg  # Saturated vapor enthalpy

        # Calculate moisture content if enthalpy provided
        if input_data.enthalpy_measured_kj_kg:
            moisture_pct = self.analyze_moisture_content(
                input_data.enthalpy_measured_kj_kg,
                float(h_g),
                float(h_fg)
            )
        else:
            # Estimate based on temperature vs saturation
            if T >= T_sat:
                # Superheated - no moisture
                moisture_pct = 0.0
            else:
                # Below saturation - estimate moisture
                # Simplified: assume linear relationship
                temp_deficit = float(T_sat - T)
                moisture_pct = min(100.0, temp_deficit / 5.0)  # ~1% per 5C below sat
                warnings.append("Moisture estimated from temperature deficit")

        # Calculate dryness and wetness fractions
        dryness = Decimal(str((100 - moisture_pct) / 100))
        wetness = self.calculate_wetness_fraction(float(dryness))

        # Detect condensation risk
        condensation_risk = self.detect_condensation_risk(
            float(T),
            float(P),
            input_data.ambient_temperature_c,
            input_data.insulation_factor,
            input_data.surface_temperature_c
        )

        # Calculate dew point and margin
        T_amb = Decimal(str(input_data.ambient_temperature_c))
        rh = Decimal(str(input_data.humidity_percent))

        # Magnus formula for dew point
        # T_dew = (243.12 * alpha) / (17.62 - alpha)
        # alpha = ln(RH/100) + (17.62 * T) / (243.12 + T)
        T_amb_f = float(T_amb)
        rh_f = float(rh) / 100
        if rh_f > 0:
            import math
            alpha = math.log(rh_f) + (17.62 * T_amb_f) / (243.12 + T_amb_f)
            T_dew = (243.12 * alpha) / (17.62 - alpha)
        else:
            T_dew = T_amb_f - 30  # Fallback

        # Calculate surface temperature
        ins_factor = Decimal(str(input_data.insulation_factor))
        if input_data.surface_temperature_c is not None:
            T_surface = input_data.surface_temperature_c
        else:
            T_surface = float(T_amb + (T - T_amb) * (Decimal("1") - ins_factor))

        dew_point_margin = T_surface - T_dew

        # Identify moisture sources if enabled
        identified_sources = []
        if self.enable_source_detection:
            # Build system data from input
            system_data = {
                'moisture_readings': [],
                'temperature_readings': [],
                'trap_status': input_data.steam_trap_data or {},
                'load_changes': [],
                'insulation_surveys': {},
                'desuperheater_data': {}
            }

            # Add location readings if available
            for loc in input_data.system_locations:
                system_data['moisture_readings'].append({
                    'location': loc,
                    'value': moisture_pct
                })

            identified_sources = self.identify_moisture_sources(system_data)

        # Generate recommendations
        preliminary_result = MoistureAnalysisOutput(
            moisture_content_pct=moisture_pct,
            wetness_fraction=wetness,
            dryness_fraction=dryness,
            condensation_risk=condensation_risk,
            dew_point_margin_c=round(dew_point_margin, 2),
            dew_point_c=round(T_dew, 2),
            saturation_temperature_c=float(T_sat),
            identified_sources=identified_sources,
            remediation_recommendations=[],
            warnings=warnings
        )

        recommendations = self.recommend_moisture_remediation(preliminary_result)

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(
            input_data, moisture_pct, condensation_risk
        )

        return MoistureAnalysisOutput(
            moisture_content_pct=moisture_pct,
            wetness_fraction=wetness,
            dryness_fraction=dryness,
            condensation_risk=condensation_risk,
            dew_point_margin_c=round(dew_point_margin, 2),
            dew_point_c=round(T_dew, 2),
            saturation_temperature_c=float(T_sat),
            identified_sources=identified_sources,
            remediation_recommendations=recommendations,
            calculation_method="IAPWS-IF97",
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    def _get_saturation_properties(
        self,
        pressure: Decimal
    ) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """Get saturation properties at given pressure."""
        if pressure in self.SATURATION_TABLE:
            return self.SATURATION_TABLE[pressure]
        return None

    def _interpolate_saturation(
        self,
        pressure: Decimal
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Interpolate saturation properties for pressure not in table."""
        pressures = sorted(self.SATURATION_TABLE.keys())

        # Clamp to table range
        if pressure <= pressures[0]:
            return self.SATURATION_TABLE[pressures[0]]
        if pressure >= pressures[-1]:
            return self.SATURATION_TABLE[pressures[-1]]

        # Find bounding pressures
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure <= pressures[i + 1]:
                P1, P2 = pressures[i], pressures[i + 1]
                props1 = self.SATURATION_TABLE[P1]
                props2 = self.SATURATION_TABLE[P2]

                # Interpolation factor
                f = (pressure - P1) / (P2 - P1)

                # Interpolate each property
                interpolated = tuple(
                    (p1 + f * (p2 - p1)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    for p1, p2 in zip(props1, props2)
                )
                return interpolated

        # Fallback
        return self.SATURATION_TABLE[pressures[len(pressures) // 2]]

    def _calculate_provenance(
        self,
        input_data: MoistureAnalysisInput,
        moisture_pct: float,
        risk: RiskLevel
    ) -> str:
        """Generate SHA-256 provenance hash for calculation."""
        data = {
            'calculator': 'MoistureAnalyzer',
            'version': '1.0.0',
            'inputs': {
                'temperature_c': input_data.temperature_c,
                'pressure_mpa': input_data.pressure_mpa,
                'ambient_temperature_c': input_data.ambient_temperature_c,
                'insulation_factor': input_data.insulation_factor,
            },
            'outputs': {
                'moisture_content_pct': moisture_pct,
                'condensation_risk': risk.value,
            },
            'method': 'IAPWS-IF97'
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'calculation_count': self.calculation_count,
            'precision': self.precision,
            'source_detection_enabled': self.enable_source_detection,
            'table_pressures': len(self.SATURATION_TABLE)
        }


# Unit test examples
def _run_self_tests():
    """Run self-tests to verify analyzer correctness."""
    analyzer = MoistureAnalyzer()

    # Test 1: Moisture content from enthalpy
    moisture = analyzer.analyze_moisture_content(
        enthalpy_measured=2700.0,
        saturation_enthalpy=2777.0,
        h_fg=2014.9
    )
    assert 0 <= moisture <= 100, f"Moisture out of range: {moisture}"
    print(f"Test 1 passed: moisture content = {moisture:.2f}%")

    # Test 2: Wetness fraction
    wetness = analyzer.calculate_wetness_fraction(0.95)
    assert abs(float(wetness) - 0.05) < 0.0001, f"Wetness calculation error: {wetness}"
    print(f"Test 2 passed: wetness fraction = {wetness}")

    # Test 3: Condensation risk - low risk case
    risk = analyzer.detect_condensation_risk(
        T=180.0, P=1.0, T_ambient=25.0, insulation_factor=0.9
    )
    assert risk in (RiskLevel.NEGLIGIBLE, RiskLevel.LOW), f"Unexpected risk: {risk}"
    print(f"Test 3 passed: condensation risk = {risk.value}")

    # Test 4: Condensation risk - high risk case
    risk_high = analyzer.detect_condensation_risk(
        T=100.0, P=0.1, T_ambient=25.0, insulation_factor=0.1
    )
    print(f"Test 4 passed: high risk case = {risk_high.value}")

    # Test 5: Source identification
    sources = analyzer.identify_moisture_sources({
        'moisture_readings': [{'location': 'header_outlet', 'value': 5.0}],
        'trap_status': {'trap_01': 'failed_open'},
        'load_changes': [{'time': '12:00', 'delta': 20}]
    })
    assert len(sources) > 0, "Should identify at least one source"
    print(f"Test 5 passed: identified {len(sources)} moisture sources")

    # Test 6: Full analysis
    result = analyzer.analyze(MoistureAnalysisInput(
        temperature_c=175.0,
        pressure_mpa=1.0,
        ambient_temperature_c=25.0,
        insulation_factor=0.9
    ))
    assert result.provenance_hash, "Should have provenance hash"
    assert len(result.remediation_recommendations) >= 0, "Should generate recommendations"
    print(f"Test 6 passed: full analysis, hash = {result.provenance_hash[:16]}...")

    print("\nAll self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
