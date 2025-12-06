"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Steam Quality Monitoring Module

This module provides steam quality monitoring per ASME standards including:
- Dryness fraction (steam quality) measurement and validation
- Total Dissolved Solids (TDS) monitoring
- Cation conductivity analysis
- Silica tracking
- Carryover estimation

Consolidates GL-012 STEAMQUAL functionality into the unified optimizer.

Standards Reference:
    - ASME Boiler and Pressure Vessel Code
    - ABMA Recommended Practices
    - EPRI Guidelines for Cycle Chemistry

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.quality import (
    ...     SteamQualityMonitor,
    ... )
    >>>
    >>> monitor = SteamQualityMonitor(config)
    >>> analysis = monitor.analyze_quality(reading)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

from .config import QualityMonitoringConfig, SteamQualityStandard
from .schemas import (
    SteamQualityReading,
    SteamQualityAnalysis,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - ASME/ABMA QUALITY LIMITS
# =============================================================================

class ASMEQualityLimits:
    """
    ASME recommended steam quality limits by pressure range.

    Based on ASME guidelines and EPRI cycle chemistry recommendations.
    """

    # TDS limits (ppm) by pressure range (psig)
    TDS_LIMITS = {
        "0-300": 3500.0,      # Low pressure
        "300-450": 3000.0,    # Medium pressure
        "450-600": 2500.0,    # Medium-high pressure
        "600-900": 1500.0,    # High pressure
        "900-1500": 1000.0,   # Very high pressure
        "1500+": 500.0,       # Ultra-high pressure
    }

    # Cation conductivity limits (microS/cm)
    CATION_CONDUCTIVITY_LIMITS = {
        "drum_boiler": 0.3,
        "once_through": 0.15,
        "turbine_inlet": 0.2,
    }

    # Silica limits (ppm)
    SILICA_LIMITS = {
        "0-300": 0.05,
        "300-600": 0.025,
        "600-1000": 0.015,
        "1000+": 0.010,
    }

    # Dissolved oxygen limits (ppb)
    DISSOLVED_O2_LIMIT = 7.0  # ASME feedwater limit

    # Sodium limits (ppb)
    SODIUM_LIMITS = {
        "drum_boiler": 10.0,
        "once_through": 3.0,
    }

    # Minimum dryness fraction
    MIN_DRYNESS_FRACTION = 0.95
    TARGET_DRYNESS_FRACTION = 0.98


class CarryoverEstimation:
    """Constants for mechanical carryover estimation."""

    # Carryover coefficients by operating condition
    # Based on steam drum design correlations
    CARRYOVER_FACTORS = {
        "normal": 0.001,      # 0.1% base carryover
        "high_load": 0.003,   # 0.3% at high load
        "upset": 0.010,       # 1.0% during upsets
    }

    # TDS ratio for carryover calculation
    # Steam TDS / Boiler Water TDS = Carryover fraction
    MAX_ACCEPTABLE_RATIO = 0.005  # 0.5% carryover


# =============================================================================
# QUALITY LIMIT CALCULATOR
# =============================================================================

class QualityLimitCalculator:
    """
    Calculator for determining applicable quality limits.

    Determines appropriate limits based on operating pressure,
    boiler type, and applicable standards.
    """

    def __init__(
        self,
        config: QualityMonitoringConfig,
    ) -> None:
        """
        Initialize quality limit calculator.

        Args:
            config: Quality monitoring configuration
        """
        self.config = config

        logger.debug("QualityLimitCalculator initialized")

    def get_tds_limit(self, pressure_psig: float) -> float:
        """
        Get TDS limit for given pressure.

        Args:
            pressure_psig: Operating pressure (psig)

        Returns:
            Maximum TDS limit (ppm)
        """
        # Use config overrides if specified
        if pressure_psig < 300:
            return self.config.max_tds_ppm_lp
        elif pressure_psig < 450:
            return self.config.max_tds_ppm_mp
        else:
            return self.config.max_tds_ppm_hp

    def get_cation_conductivity_limit(
        self,
        boiler_type: str = "drum_boiler",
    ) -> float:
        """
        Get cation conductivity limit.

        Args:
            boiler_type: Type of boiler (drum_boiler, once_through)

        Returns:
            Maximum cation conductivity (microS/cm)
        """
        if self.config.max_cation_conductivity_us_cm > 0:
            return self.config.max_cation_conductivity_us_cm

        return ASMEQualityLimits.CATION_CONDUCTIVITY_LIMITS.get(
            boiler_type,
            0.3
        )

    def get_silica_limit(self, pressure_psig: float) -> float:
        """
        Get silica limit for given pressure.

        Args:
            pressure_psig: Operating pressure (psig)

        Returns:
            Maximum silica limit (ppm)
        """
        if self.config.max_silica_ppm > 0:
            return self.config.max_silica_ppm

        if pressure_psig < 300:
            return ASMEQualityLimits.SILICA_LIMITS["0-300"]
        elif pressure_psig < 600:
            return ASMEQualityLimits.SILICA_LIMITS["300-600"]
        elif pressure_psig < 1000:
            return ASMEQualityLimits.SILICA_LIMITS["600-1000"]
        else:
            return ASMEQualityLimits.SILICA_LIMITS["1000+"]

    def get_dryness_limits(self) -> Tuple[float, float]:
        """
        Get minimum and target dryness fraction.

        Returns:
            Tuple of (minimum, target) dryness fractions
        """
        return (
            self.config.min_dryness_fraction,
            self.config.target_dryness_fraction,
        )

    def get_warning_threshold(self, limit: float) -> float:
        """
        Get warning threshold for a parameter limit.

        Args:
            limit: The parameter limit value

        Returns:
            Warning threshold value
        """
        return limit * (self.config.warning_threshold_pct / 100)

    def get_critical_threshold(self, limit: float) -> float:
        """
        Get critical threshold for a parameter limit.

        Args:
            limit: The parameter limit value

        Returns:
            Critical threshold value
        """
        return limit * (self.config.critical_threshold_pct / 100)


# =============================================================================
# DRYNESS FRACTION CALCULATOR
# =============================================================================

class DrynessFractionCalculator:
    """
    Calculator for steam dryness fraction (quality).

    Methods:
        - Throttling calorimeter
        - Separating calorimeter
        - Conductivity-based estimation
        - Temperature-based estimation
    """

    def __init__(self) -> None:
        """Initialize dryness fraction calculator."""
        logger.debug("DrynessFractionCalculator initialized")

    def calculate_from_throttling(
        self,
        inlet_pressure_psig: float,
        outlet_pressure_psig: float,
        outlet_temperature_f: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dryness using throttling calorimeter method.

        The throttling calorimeter works by throttling wet steam to
        low pressure where it becomes superheated. The superheat
        indicates the original quality.

        Args:
            inlet_pressure_psig: Steam inlet pressure (psig)
            outlet_pressure_psig: Calorimeter outlet pressure (psig)
            outlet_temperature_f: Measured outlet temperature (F)

        Returns:
            Tuple of (dryness_fraction, calculation_details)
        """
        # Get saturation properties
        from .distribution import SteamPropertyCalculator
        calc = SteamPropertyCalculator()

        inlet_props = calc.get_saturation_properties(inlet_pressure_psig)
        outlet_props = calc.get_saturation_properties(outlet_pressure_psig)

        h_f_in = inlet_props["h_f_btu_lb"]
        h_fg_in = inlet_props["h_fg_btu_lb"]
        t_sat_out = outlet_props["saturation_temp_f"]
        h_g_out = outlet_props["h_g_btu_lb"]

        # Calculate outlet enthalpy from superheat
        superheat = outlet_temperature_f - t_sat_out
        if superheat < 0:
            # Not superheated - cannot use this method
            return 0.0, {
                "error": "Steam not superheated at outlet",
                "superheat_f": superheat,
            }

        # h_outlet = h_g + Cp * superheat
        cp_superheated = 0.48  # BTU/lb-F
        h_outlet = h_g_out + cp_superheated * superheat

        # For throttling, h_in = h_out (isenthalpic)
        # h_in = h_f + x * h_fg
        # x = (h_in - h_f) / h_fg
        dryness = (h_outlet - h_f_in) / h_fg_in

        # Validate
        dryness = max(0.0, min(1.0, dryness))

        details = {
            "method": "throttling_calorimeter",
            "inlet_pressure_psig": inlet_pressure_psig,
            "outlet_pressure_psig": outlet_pressure_psig,
            "outlet_temperature_f": outlet_temperature_f,
            "saturation_temp_outlet_f": t_sat_out,
            "superheat_f": superheat,
            "enthalpy_outlet_btu_lb": h_outlet,
            "h_f_inlet_btu_lb": h_f_in,
            "h_fg_inlet_btu_lb": h_fg_in,
        }

        return dryness, details

    def estimate_from_conductivity(
        self,
        steam_tds_ppm: float,
        boiler_water_tds_ppm: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate dryness from TDS carryover ratio.

        Mechanical carryover is indicated by TDS in steam relative
        to boiler water TDS.

        Args:
            steam_tds_ppm: Measured steam TDS (ppm)
            boiler_water_tds_ppm: Boiler water TDS (ppm)

        Returns:
            Tuple of (dryness_fraction, calculation_details)
        """
        if boiler_water_tds_ppm <= 0:
            return 1.0, {"error": "Invalid boiler water TDS"}

        # Carryover fraction = Steam TDS / Boiler Water TDS
        carryover_fraction = steam_tds_ppm / boiler_water_tds_ppm

        # Dryness = 1 - carryover (moisture carries dissolved solids)
        # This is a simplification - actual relationship depends on
        # mechanical entrainment vs vapor carry-over
        dryness = 1.0 - carryover_fraction

        # Validate
        dryness = max(0.0, min(1.0, dryness))

        details = {
            "method": "conductivity_estimation",
            "steam_tds_ppm": steam_tds_ppm,
            "boiler_water_tds_ppm": boiler_water_tds_ppm,
            "carryover_fraction": carryover_fraction,
            "carryover_pct": carryover_fraction * 100,
        }

        return dryness, details

    def estimate_from_temperature(
        self,
        measured_temperature_f: float,
        pressure_psig: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate dryness from temperature (for superheated steam check).

        If temperature equals saturation, steam is saturated.
        If below saturation, steam is wet.
        If above saturation, steam is superheated (dry).

        Args:
            measured_temperature_f: Measured steam temperature (F)
            pressure_psig: Steam pressure (psig)

        Returns:
            Tuple of (dryness_fraction, calculation_details)
        """
        from .distribution import SteamPropertyCalculator
        calc = SteamPropertyCalculator()

        sat_props = calc.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        if measured_temperature_f >= t_sat:
            # Superheated - definitely dry
            dryness = 1.0
            superheat = measured_temperature_f - t_sat
        else:
            # Below saturation - estimate wetness
            # This is approximate - actual wet steam is at saturation temp
            # A temperature reading below saturation suggests measurement error
            # or significant moisture content
            temp_deficit = t_sat - measured_temperature_f
            # Rough estimation: 1F deficit ~ 0.5% moisture
            moisture_estimate = temp_deficit * 0.005
            dryness = max(0.5, 1.0 - moisture_estimate)
            superheat = 0.0

        details = {
            "method": "temperature_estimation",
            "measured_temperature_f": measured_temperature_f,
            "saturation_temperature_f": t_sat,
            "pressure_psig": pressure_psig,
            "superheat_f": superheat if measured_temperature_f >= t_sat else 0,
            "note": "Estimate only - use calorimeter for accurate measurement",
        }

        return dryness, details


# =============================================================================
# CARRYOVER ANALYZER
# =============================================================================

class CarryoverAnalyzer:
    """
    Analyzer for steam carryover estimation and diagnosis.

    Carryover types:
        - Mechanical entrainment (water droplets)
        - Volatile carryover (dissolved solids in vapor)
    """

    def __init__(self) -> None:
        """Initialize carryover analyzer."""
        logger.debug("CarryoverAnalyzer initialized")

    def estimate_mechanical_carryover(
        self,
        steam_tds_ppm: float,
        boiler_water_tds_ppm: float,
    ) -> Dict[str, Any]:
        """
        Estimate mechanical carryover from TDS ratio.

        Args:
            steam_tds_ppm: Steam TDS (ppm)
            boiler_water_tds_ppm: Boiler water TDS (ppm)

        Returns:
            Dictionary with carryover analysis
        """
        if boiler_water_tds_ppm <= 0:
            return {
                "error": "Invalid boiler water TDS",
                "carryover_pct": None,
            }

        carryover_ratio = steam_tds_ppm / boiler_water_tds_ppm
        carryover_pct = carryover_ratio * 100

        # Classify severity
        if carryover_pct < 0.1:
            severity = "normal"
            status = "acceptable"
        elif carryover_pct < 0.5:
            severity = "elevated"
            status = "warning"
        elif carryover_pct < 1.0:
            severity = "high"
            status = "action_required"
        else:
            severity = "critical"
            status = "immediate_action"

        # Diagnose potential causes
        causes = []
        if carryover_pct > 0.1:
            causes.append("High boiler water level")
            causes.append("Drum internals damage")
            causes.append("Rapid load changes")
        if carryover_pct > 0.5:
            causes.append("Foaming due to contamination")
            causes.append("High TDS in boiler water")
            causes.append("Drum pressure fluctuations")
        if carryover_pct > 1.0:
            causes.append("Severe mechanical damage")
            causes.append("Critical water treatment failure")

        return {
            "carryover_pct": carryover_pct,
            "carryover_ratio": carryover_ratio,
            "severity": severity,
            "status": status,
            "potential_causes": causes,
            "max_acceptable_pct": CarryoverEstimation.MAX_ACCEPTABLE_RATIO * 100,
        }

    def estimate_silica_carryover(
        self,
        steam_silica_ppm: float,
        boiler_water_silica_ppm: float,
        pressure_psig: float,
    ) -> Dict[str, Any]:
        """
        Estimate silica carryover (volatile + mechanical).

        Silica volatility increases significantly with pressure.

        Args:
            steam_silica_ppm: Steam silica content (ppm)
            boiler_water_silica_ppm: Boiler water silica (ppm)
            pressure_psig: Operating pressure (psig)

        Returns:
            Dictionary with silica carryover analysis
        """
        if boiler_water_silica_ppm <= 0:
            return {"error": "Invalid boiler water silica"}

        # Silica distribution ratio (vapor/liquid) increases with pressure
        # Approximate relationship based on Straub correlations
        pressure_psia = pressure_psig + 14.696

        # Distribution ratio approximation
        if pressure_psia < 400:
            expected_ratio = 0.001
        elif pressure_psia < 800:
            expected_ratio = 0.01
        elif pressure_psia < 1200:
            expected_ratio = 0.05
        elif pressure_psia < 1800:
            expected_ratio = 0.15
        else:
            expected_ratio = 0.30

        actual_ratio = steam_silica_ppm / boiler_water_silica_ppm

        # Mechanical contribution = actual - expected volatile
        if actual_ratio > expected_ratio:
            mechanical_contribution = actual_ratio - expected_ratio
            volatile_contribution = expected_ratio
        else:
            mechanical_contribution = 0
            volatile_contribution = actual_ratio

        return {
            "steam_silica_ppm": steam_silica_ppm,
            "boiler_water_silica_ppm": boiler_water_silica_ppm,
            "pressure_psig": pressure_psig,
            "actual_ratio": actual_ratio,
            "expected_volatile_ratio": expected_ratio,
            "volatile_contribution_pct": volatile_contribution * 100,
            "mechanical_contribution_pct": mechanical_contribution * 100,
            "total_carryover_pct": actual_ratio * 100,
        }


# =============================================================================
# STEAM QUALITY MONITOR
# =============================================================================

class SteamQualityMonitor:
    """
    Comprehensive steam quality monitoring per ASME standards.

    Monitors and analyzes:
        - Dryness fraction (steam quality)
        - TDS content
        - Cation conductivity
        - Silica content
        - Dissolved oxygen
        - Carryover estimation

    Example:
        >>> config = QualityMonitoringConfig()
        >>> monitor = SteamQualityMonitor(config)
        >>>
        >>> reading = SteamQualityReading(
        ...     location_id="BLR-001-STEAM",
        ...     pressure_psig=150,
        ...     temperature_f=366,
        ...     dryness_fraction=0.98,
        ...     tds_ppm=10,
        ...     cation_conductivity_us_cm=0.15,
        ... )
        >>>
        >>> analysis = monitor.analyze_quality(reading)
        >>> print(analysis.overall_status)
    """

    def __init__(
        self,
        config: QualityMonitoringConfig,
    ) -> None:
        """
        Initialize steam quality monitor.

        Args:
            config: Quality monitoring configuration
        """
        self.config = config
        self.limit_calc = QualityLimitCalculator(config)
        self.dryness_calc = DrynessFractionCalculator()
        self.carryover_analyzer = CarryoverAnalyzer()

        # History for trending
        self._reading_history: List[SteamQualityReading] = []
        self._max_history = 1000

        logger.info(
            f"SteamQualityMonitor initialized (standard: {config.standard})"
        )

    def analyze_quality(
        self,
        reading: SteamQualityReading,
        boiler_water_tds_ppm: Optional[float] = None,
    ) -> SteamQualityAnalysis:
        """
        Analyze steam quality reading.

        Args:
            reading: Steam quality reading
            boiler_water_tds_ppm: Optional boiler water TDS for carryover calc

        Returns:
            SteamQualityAnalysis with validation and recommendations
        """
        start_time = datetime.now(timezone.utc)

        limits_exceeded = []
        limits_warning = []
        recommendations = []

        # Analyze dryness fraction
        dryness_status = self._analyze_dryness(reading, limits_exceeded, limits_warning)

        # Analyze TDS
        tds_status = self._analyze_tds(
            reading, limits_exceeded, limits_warning, recommendations
        )

        # Analyze cation conductivity
        conductivity_status = self._analyze_conductivity(
            reading, limits_exceeded, limits_warning, recommendations
        )

        # Analyze silica
        silica_status = self._analyze_silica(
            reading, limits_exceeded, limits_warning, recommendations
        )

        # Estimate carryover if boiler water data available
        carryover_estimate = None
        if boiler_water_tds_ppm and reading.tds_ppm:
            carryover_result = self.carryover_analyzer.estimate_mechanical_carryover(
                steam_tds_ppm=reading.tds_ppm,
                boiler_water_tds_ppm=boiler_water_tds_ppm,
            )
            carryover_estimate = carryover_result.get("carryover_pct")

            if carryover_estimate and carryover_estimate > 0.5:
                recommendations.append(
                    f"Carryover at {carryover_estimate:.2f}% - "
                    "check drum internals and water level"
                )

        # Determine overall status
        overall_status = self._determine_overall_status(
            dryness_status,
            tds_status,
            conductivity_status,
            silica_status,
        )

        # Store in history
        self._update_history(reading)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(reading)

        return SteamQualityAnalysis(
            reading=reading,
            overall_status=overall_status,
            dryness_status=dryness_status,
            tds_status=tds_status,
            conductivity_status=conductivity_status,
            silica_status=silica_status,
            limits_exceeded=limits_exceeded,
            limits_warning=limits_warning,
            recommendations=recommendations,
            estimated_carryover_pct=carryover_estimate,
            provenance_hash=provenance_hash,
        )

    def _analyze_dryness(
        self,
        reading: SteamQualityReading,
        limits_exceeded: List[str],
        limits_warning: List[str],
    ) -> ValidationStatus:
        """Analyze dryness fraction."""
        min_dryness, target_dryness = self.limit_calc.get_dryness_limits()
        dryness = reading.dryness_fraction

        if dryness < min_dryness:
            limits_exceeded.append(
                f"Dryness fraction {dryness:.3f} below minimum {min_dryness}"
            )
            return ValidationStatus.INVALID
        elif dryness < target_dryness:
            warning_threshold = min_dryness + 0.8 * (target_dryness - min_dryness)
            if dryness < warning_threshold:
                limits_warning.append(
                    f"Dryness fraction {dryness:.3f} below target {target_dryness}"
                )
                return ValidationStatus.WARNING
            return ValidationStatus.VALID
        else:
            return ValidationStatus.VALID

    def _analyze_tds(
        self,
        reading: SteamQualityReading,
        limits_exceeded: List[str],
        limits_warning: List[str],
        recommendations: List[str],
    ) -> ValidationStatus:
        """Analyze TDS content."""
        if reading.tds_ppm is None:
            return ValidationStatus.UNCHECKED

        tds_limit = self.limit_calc.get_tds_limit(reading.pressure_psig)
        warning_threshold = self.limit_calc.get_warning_threshold(tds_limit)
        critical_threshold = self.limit_calc.get_critical_threshold(tds_limit)

        tds = reading.tds_ppm

        if tds > tds_limit:
            limits_exceeded.append(
                f"TDS {tds:.1f} ppm exceeds limit {tds_limit:.0f} ppm"
            )
            recommendations.append(
                "Increase blowdown rate or improve water treatment"
            )
            return ValidationStatus.INVALID
        elif tds > critical_threshold:
            limits_warning.append(
                f"TDS {tds:.1f} ppm approaching limit ({tds_limit:.0f} ppm)"
            )
            return ValidationStatus.WARNING
        elif tds > warning_threshold:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALID

    def _analyze_conductivity(
        self,
        reading: SteamQualityReading,
        limits_exceeded: List[str],
        limits_warning: List[str],
        recommendations: List[str],
    ) -> ValidationStatus:
        """Analyze cation conductivity."""
        if reading.cation_conductivity_us_cm is None:
            return ValidationStatus.UNCHECKED

        cond_limit = self.limit_calc.get_cation_conductivity_limit()
        warning_threshold = self.limit_calc.get_warning_threshold(cond_limit)
        critical_threshold = self.limit_calc.get_critical_threshold(cond_limit)

        cond = reading.cation_conductivity_us_cm

        if cond > cond_limit:
            limits_exceeded.append(
                f"Cation conductivity {cond:.3f} uS/cm exceeds "
                f"limit {cond_limit:.2f} uS/cm"
            )
            recommendations.append(
                "Check for contamination or condensate polisher issues"
            )
            return ValidationStatus.INVALID
        elif cond > critical_threshold:
            limits_warning.append(
                f"Cation conductivity {cond:.3f} uS/cm approaching limit"
            )
            return ValidationStatus.WARNING
        elif cond > warning_threshold:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALID

    def _analyze_silica(
        self,
        reading: SteamQualityReading,
        limits_exceeded: List[str],
        limits_warning: List[str],
        recommendations: List[str],
    ) -> ValidationStatus:
        """Analyze silica content."""
        if reading.silica_ppm is None:
            return ValidationStatus.UNCHECKED

        silica_limit = self.limit_calc.get_silica_limit(reading.pressure_psig)
        warning_threshold = self.limit_calc.get_warning_threshold(silica_limit)

        silica = reading.silica_ppm

        if silica > silica_limit:
            limits_exceeded.append(
                f"Silica {silica:.4f} ppm exceeds limit {silica_limit:.3f} ppm"
            )
            recommendations.append(
                "Increase blowdown or improve silica removal in makeup water"
            )
            return ValidationStatus.INVALID
        elif silica > warning_threshold:
            limits_warning.append(
                f"Silica {silica:.4f} ppm approaching limit"
            )
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALID

    def _determine_overall_status(
        self,
        dryness_status: ValidationStatus,
        tds_status: ValidationStatus,
        conductivity_status: ValidationStatus,
        silica_status: ValidationStatus,
    ) -> ValidationStatus:
        """Determine overall quality status from individual statuses."""
        statuses = [dryness_status, tds_status, conductivity_status, silica_status]

        # Filter out unchecked
        checked = [s for s in statuses if s != ValidationStatus.UNCHECKED]

        if not checked:
            return ValidationStatus.UNCHECKED

        if any(s == ValidationStatus.INVALID for s in checked):
            return ValidationStatus.INVALID
        elif any(s == ValidationStatus.WARNING for s in checked):
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALID

    def _update_history(self, reading: SteamQualityReading) -> None:
        """Update reading history."""
        self._reading_history.append(reading)
        if len(self._reading_history) > self._max_history:
            self._reading_history = self._reading_history[-self._max_history:]

    def _calculate_provenance_hash(
        self,
        reading: SteamQualityReading,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "location_id": reading.location_id,
            "timestamp": reading.timestamp.isoformat(),
            "pressure_psig": reading.pressure_psig,
            "temperature_f": reading.temperature_f,
            "dryness_fraction": reading.dryness_fraction,
            "tds_ppm": reading.tds_ppm,
            "cation_conductivity_us_cm": reading.cation_conductivity_us_cm,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_quality_trend(
        self,
        parameter: str,
        num_readings: int = 10,
    ) -> Dict[str, Any]:
        """
        Get trend analysis for a quality parameter.

        Args:
            parameter: Parameter name (tds, conductivity, dryness, silica)
            num_readings: Number of recent readings to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(self._reading_history) < 2:
            return {"trend": "insufficient_data", "data_points": 0}

        recent = self._reading_history[-num_readings:]

        # Extract parameter values
        values = []
        timestamps = []
        for reading in recent:
            if parameter == "tds" and reading.tds_ppm is not None:
                values.append(reading.tds_ppm)
                timestamps.append(reading.timestamp)
            elif parameter == "conductivity" and reading.cation_conductivity_us_cm is not None:
                values.append(reading.cation_conductivity_us_cm)
                timestamps.append(reading.timestamp)
            elif parameter == "dryness":
                values.append(reading.dryness_fraction)
                timestamps.append(reading.timestamp)
            elif parameter == "silica" and reading.silica_ppm is not None:
                values.append(reading.silica_ppm)
                timestamps.append(reading.timestamp)

        if len(values) < 2:
            return {"trend": "insufficient_data", "data_points": len(values)}

        # Calculate trend
        n = len(values)
        mean_value = sum(values) / n
        variance = sum((v - mean_value) ** 2 for v in values) / n

        # Simple linear trend
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (v - mean_value) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0

        # Classify trend
        slope_pct = (slope / mean_value * 100) if mean_value > 0 else 0

        if slope_pct > 5:
            trend = "increasing"
        elif slope_pct < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope_per_reading": slope,
            "slope_pct": slope_pct,
            "mean_value": mean_value,
            "std_deviation": variance ** 0.5,
            "min_value": min(values),
            "max_value": max(values),
            "data_points": len(values),
            "latest_value": values[-1] if values else None,
        }
