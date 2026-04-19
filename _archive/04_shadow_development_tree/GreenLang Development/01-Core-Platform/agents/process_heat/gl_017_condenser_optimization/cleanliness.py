"""
GL-017 CONDENSYNC Agent - HEI Cleanliness Factor Calculator

This module implements HEI (Heat Exchange Institute) Standards compliant
cleanliness factor calculations for steam surface condensers.

All calculations are deterministic with zero hallucination.
Formula references: HEI Standards for Steam Surface Condensers, 12th Edition.

Example:
    >>> calculator = HEICleanlinessCalculator(config)
    >>> result = calculator.calculate_cleanliness(
    ...     heat_duty=500e6,
    ...     lmtd=15.0,
    ...     surface_area=150000.0,
    ... )
    >>> print(f"Cleanliness: {result.cleanliness_factor:.3f}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CleanlinessConfig,
    TubeFoulingConfig,
    TubeMaterial,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CleanlinessResult,
    CleaningStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - HEI Standards Reference Values
# =============================================================================

class HEIConstants:
    """HEI Standards reference constants."""

    # Base heat transfer coefficient at reference conditions
    # Reference: 70F inlet, 7 ft/s velocity, clean admiralty tubes
    U_BASE_BTU_HR_FT2_F = 650.0

    # Tube material factors (Table 5-1, HEI Standards 12th Ed)
    MATERIAL_FACTORS = {
        TubeMaterial.ADMIRALTY_BRASS: 1.00,
        TubeMaterial.ALUMINUM_BRASS: 0.98,
        TubeMaterial.COPPER_NICKEL_90_10: 0.92,
        TubeMaterial.COPPER_NICKEL_70_30: 0.88,
        TubeMaterial.STAINLESS_304: 0.78,
        TubeMaterial.STAINLESS_316: 0.76,
        TubeMaterial.TITANIUM: 0.73,
        TubeMaterial.DUPLEX_2205: 0.75,
    }

    # Tube wall thickness by gauge (BWG) in inches
    TUBE_WALL_THICKNESS = {
        14: 0.083,
        15: 0.072,
        16: 0.065,
        17: 0.058,
        18: 0.049,
        19: 0.042,
        20: 0.035,
        21: 0.032,
        22: 0.028,
        24: 0.022,
    }

    # Thermal conductivity of tube materials (BTU/hr-ft-F)
    THERMAL_CONDUCTIVITY = {
        TubeMaterial.ADMIRALTY_BRASS: 64.0,
        TubeMaterial.ALUMINUM_BRASS: 58.0,
        TubeMaterial.COPPER_NICKEL_90_10: 26.0,
        TubeMaterial.COPPER_NICKEL_70_30: 17.0,
        TubeMaterial.STAINLESS_304: 9.4,
        TubeMaterial.STAINLESS_316: 9.4,
        TubeMaterial.TITANIUM: 12.0,
        TubeMaterial.DUPLEX_2205: 9.4,
    }

    # Inlet water correction factors (Table 5-2)
    # Based on water quality: freshwater=1.0, brackish=0.9, seawater=0.85
    INLET_WATER_FACTORS = {
        "freshwater": 1.00,
        "brackish": 0.90,
        "seawater": 0.85,
        "cooling_tower": 0.85,
    }

    # Velocity correction exponent
    VELOCITY_EXPONENT = 0.5

    # Temperature correction coefficient
    TEMP_COEFFICIENT = 0.01  # Per degree F deviation from 70F


@dataclass
class HEICalculationInputs:
    """Input parameters for HEI calculation."""
    heat_duty_btu_hr: float
    lmtd_f: float
    surface_area_ft2: float
    cw_velocity_fps: float
    cw_inlet_temp_f: float
    tube_material: TubeMaterial
    tube_od_in: float
    tube_gauge: int
    design_cleanliness: float


class HEICleanlinessCalculator:
    """
    HEI Standards compliant cleanliness factor calculator.

    This calculator implements the methodology from HEI Standards for
    Steam Surface Condensers (12th Edition) for determining condenser
    cleanliness factors.

    The cleanliness factor is defined as:
        CF = U_actual / U_clean

    Where:
        U_actual = Q / (A * LMTD)
        U_clean = f(velocity, inlet temp, tube material, gauge)

    Attributes:
        config: HEI cleanliness configuration
        fouling_config: Tube fouling configuration

    Example:
        >>> config = CleanlinessConfig()
        >>> calculator = HEICleanlinessCalculator(config)
        >>> result = calculator.calculate_cleanliness(inputs)
    """

    def __init__(
        self,
        cleanliness_config: CleanlinessConfig,
        fouling_config: TubeFoulingConfig,
    ) -> None:
        """
        Initialize the HEI cleanliness calculator.

        Args:
            cleanliness_config: HEI cleanliness configuration
            fouling_config: Tube fouling configuration
        """
        self.config = cleanliness_config
        self.fouling_config = fouling_config
        self._calculation_count = 0

        logger.info(
            f"HEICleanlinessCalculator initialized: "
            f"HEI {self.config.hei_edition} Edition"
        )

    def calculate_cleanliness(
        self,
        heat_duty_btu_hr: float,
        lmtd_f: float,
        surface_area_ft2: float,
        cw_velocity_fps: float,
        cw_inlet_temp_f: float,
        tube_material: Optional[TubeMaterial] = None,
        tube_od_in: Optional[float] = None,
        tube_gauge: Optional[int] = None,
        water_type: str = "cooling_tower",
    ) -> CleanlinessResult:
        """
        Calculate HEI cleanliness factor.

        This method implements the HEI heat transfer coefficient method
        for determining condenser cleanliness.

        Args:
            heat_duty_btu_hr: Condenser heat duty (BTU/hr)
            lmtd_f: Log mean temperature difference (F)
            surface_area_ft2: Heat transfer surface area (ft2)
            cw_velocity_fps: Cooling water velocity (ft/s)
            cw_inlet_temp_f: Cooling water inlet temperature (F)
            tube_material: Tube material (optional, uses config default)
            tube_od_in: Tube OD in inches (optional, uses config default)
            tube_gauge: Tube gauge BWG (optional, uses config default)
            water_type: Water type for inlet factor

        Returns:
            CleanlinessResult with calculated cleanliness factor

        Raises:
            ValueError: If input parameters are invalid
        """
        logger.debug(f"Calculating HEI cleanliness factor")
        self._calculation_count += 1
        start_time = datetime.now(timezone.utc)

        # Use config defaults if not provided
        tube_material = tube_material or self.fouling_config.tube_material
        if isinstance(tube_material, str):
            tube_material = TubeMaterial(tube_material)
        tube_od_in = tube_od_in or self.fouling_config.tube_od_in
        tube_gauge = tube_gauge or self.fouling_config.tube_gauge

        # Input validation
        self._validate_inputs(
            heat_duty_btu_hr, lmtd_f, surface_area_ft2,
            cw_velocity_fps, cw_inlet_temp_f
        )

        # Calculate actual overall heat transfer coefficient
        u_actual = self._calculate_actual_u(
            heat_duty_btu_hr, lmtd_f, surface_area_ft2
        )

        # Calculate clean tube heat transfer coefficient (HEI method)
        u_clean = self._calculate_clean_u(
            cw_velocity_fps=cw_velocity_fps,
            cw_inlet_temp_f=cw_inlet_temp_f,
            tube_material=tube_material,
            tube_od_in=tube_od_in,
            tube_gauge=tube_gauge,
            water_type=water_type,
        )

        # Calculate design U (with design cleanliness factor)
        design_cleanliness = self.fouling_config.design_cleanliness_factor
        u_design = u_clean * design_cleanliness

        # Calculate cleanliness factor
        cleanliness_factor = u_actual / u_clean if u_clean > 0 else 0.0

        # Calculate cleanliness ratio (actual vs design)
        cleanliness_ratio = (
            cleanliness_factor / design_cleanliness
            if design_cleanliness > 0 else 0.0
        )

        # Calculate fouling factor
        fouling_factor = self._calculate_fouling_factor(
            u_actual, u_clean
        )

        # Estimate fouling thickness
        fouling_thickness = self._estimate_fouling_thickness(fouling_factor)

        # Determine cleaning status
        cleaning_status = self._determine_cleaning_status(cleanliness_factor)

        # Estimate days to cleaning
        days_to_cleaning = self._estimate_days_to_cleaning(
            cleanliness_factor, cleaning_status
        )

        # Create result
        result = CleanlinessResult(
            cleanliness_factor=round(cleanliness_factor, 4),
            design_cleanliness=design_cleanliness,
            cleanliness_ratio=round(cleanliness_ratio, 3),
            u_actual_btu_hr_ft2_f=round(u_actual, 2),
            u_clean_btu_hr_ft2_f=round(u_clean, 2),
            u_design_btu_hr_ft2_f=round(u_design, 2),
            fouling_factor_hr_ft2_f_btu=round(fouling_factor, 6),
            estimated_fouling_thickness_mils=(
                round(fouling_thickness, 2) if fouling_thickness else None
            ),
            cleaning_status=cleaning_status,
            estimated_days_to_cleaning=days_to_cleaning,
            lmtd_f=round(lmtd_f, 2),
            heat_duty_btu_hr=round(heat_duty_btu_hr, 0),
            surface_area_ft2=round(surface_area_ft2, 0),
            calculation_method="HEI_STANDARD",
            formula_reference=f"HEI Standards {self.config.hei_edition} Ed. Section 5",
        )

        logger.info(
            f"Cleanliness calculation complete: CF={cleanliness_factor:.3f}, "
            f"Status={cleaning_status.value}"
        )

        return result

    def calculate_lmtd(
        self,
        hot_inlet_temp_f: float,
        hot_outlet_temp_f: float,
        cold_inlet_temp_f: float,
        cold_outlet_temp_f: float,
    ) -> float:
        """
        Calculate log mean temperature difference.

        LMTD = (dT1 - dT2) / ln(dT1 / dT2)

        For a condenser:
        - Hot side inlet = saturation temperature
        - Hot side outlet = saturation temperature (isothermal condensation)

        Args:
            hot_inlet_temp_f: Hot side inlet (saturation temp)
            hot_outlet_temp_f: Hot side outlet (saturation temp)
            cold_inlet_temp_f: Cold side inlet (CW inlet)
            cold_outlet_temp_f: Cold side outlet (CW outlet)

        Returns:
            Log mean temperature difference (F)
        """
        dt1 = hot_inlet_temp_f - cold_outlet_temp_f  # Terminal temp difference
        dt2 = hot_outlet_temp_f - cold_inlet_temp_f  # Initial temp difference

        # Handle edge cases
        if dt1 <= 0 or dt2 <= 0:
            logger.warning(
                f"Temperature cross detected: dT1={dt1:.1f}, dT2={dt2:.1f}"
            )
            return abs(dt1 + dt2) / 2

        if abs(dt1 - dt2) < 0.1:
            # Avoid log(1) = 0
            return dt1

        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        return lmtd

    def _validate_inputs(
        self,
        heat_duty: float,
        lmtd: float,
        area: float,
        velocity: float,
        inlet_temp: float,
    ) -> None:
        """Validate input parameters."""
        if heat_duty <= 0:
            raise ValueError("Heat duty must be positive")
        if lmtd <= 0:
            raise ValueError("LMTD must be positive")
        if area <= 0:
            raise ValueError("Surface area must be positive")
        if velocity <= 0:
            raise ValueError("Velocity must be positive")
        if velocity > 15:
            logger.warning(f"Velocity {velocity:.1f} fps exceeds typical range")
        if inlet_temp < 32 or inlet_temp > 120:
            logger.warning(
                f"Inlet temperature {inlet_temp:.1f}F outside typical range"
            )

    def _calculate_actual_u(
        self,
        heat_duty_btu_hr: float,
        lmtd_f: float,
        surface_area_ft2: float,
    ) -> float:
        """
        Calculate actual overall heat transfer coefficient.

        U = Q / (A * LMTD)

        Args:
            heat_duty_btu_hr: Heat duty (BTU/hr)
            lmtd_f: Log mean temperature difference (F)
            surface_area_ft2: Surface area (ft2)

        Returns:
            Actual U value (BTU/hr-ft2-F)
        """
        u_actual = heat_duty_btu_hr / (surface_area_ft2 * lmtd_f)
        return u_actual

    def _calculate_clean_u(
        self,
        cw_velocity_fps: float,
        cw_inlet_temp_f: float,
        tube_material: TubeMaterial,
        tube_od_in: float,
        tube_gauge: int,
        water_type: str = "cooling_tower",
    ) -> float:
        """
        Calculate clean tube heat transfer coefficient per HEI Standards.

        U_clean = U_base * F_material * F_velocity * F_temperature * F_water

        Args:
            cw_velocity_fps: Cooling water velocity (ft/s)
            cw_inlet_temp_f: Cooling water inlet temperature (F)
            tube_material: Tube material
            tube_od_in: Tube OD (inches)
            tube_gauge: Tube gauge (BWG)
            water_type: Water type for correction

        Returns:
            Clean tube U value (BTU/hr-ft2-F)
        """
        # Base U value
        u_base = HEIConstants.U_BASE_BTU_HR_FT2_F

        # Material factor
        f_material = HEIConstants.MATERIAL_FACTORS.get(tube_material, 0.78)

        # Velocity correction
        # U proportional to V^0.5
        reference_velocity = self.config.reference_velocity_fps
        if self.config.include_velocity_correction:
            f_velocity = (cw_velocity_fps / reference_velocity) ** HEIConstants.VELOCITY_EXPONENT
        else:
            f_velocity = 1.0

        # Temperature correction
        # U increases with inlet temperature
        reference_temp = self.config.reference_inlet_temp_f
        if self.config.include_temperature_correction:
            temp_diff = cw_inlet_temp_f - reference_temp
            f_temperature = 1.0 + (HEIConstants.TEMP_COEFFICIENT * temp_diff)
        else:
            f_temperature = 1.0

        # Inlet water factor
        f_water = HEIConstants.INLET_WATER_FACTORS.get(water_type, 0.85)

        # Tube wall resistance correction
        wall_thickness = HEIConstants.TUBE_WALL_THICKNESS.get(tube_gauge, 0.049)
        k_tube = HEIConstants.THERMAL_CONDUCTIVITY.get(tube_material, 9.4)

        # Wall resistance = t / k (ft2-hr-F/BTU)
        wall_resistance = (wall_thickness / 12) / k_tube

        # Calculate clean U
        u_clean = u_base * f_material * f_velocity * f_temperature * f_water

        # Adjust for tube wall resistance (small correction)
        # 1/U_corrected = 1/U_clean + R_wall
        u_clean_corrected = 1.0 / (1.0 / u_clean + wall_resistance)

        logger.debug(
            f"Clean U calculation: base={u_base:.0f}, "
            f"Fm={f_material:.2f}, Fv={f_velocity:.3f}, "
            f"Ft={f_temperature:.3f}, Fw={f_water:.2f}, "
            f"U_clean={u_clean_corrected:.1f}"
        )

        return u_clean_corrected

    def _calculate_fouling_factor(
        self,
        u_actual: float,
        u_clean: float,
    ) -> float:
        """
        Calculate fouling factor from U values.

        R_fouling = 1/U_actual - 1/U_clean

        Args:
            u_actual: Actual overall U (BTU/hr-ft2-F)
            u_clean: Clean tube U (BTU/hr-ft2-F)

        Returns:
            Fouling factor (hr-ft2-F/BTU)
        """
        if u_actual <= 0 or u_clean <= 0:
            return 0.0

        if u_actual >= u_clean:
            # No fouling (or negative - measurement error)
            return 0.0

        fouling_factor = (1.0 / u_actual) - (1.0 / u_clean)
        return max(0.0, fouling_factor)

    def _estimate_fouling_thickness(
        self,
        fouling_factor: float,
        fouling_conductivity: float = 0.6,
    ) -> Optional[float]:
        """
        Estimate fouling layer thickness from fouling factor.

        t = R_f * k_fouling

        Assumes typical biofilm/scale conductivity of 0.6 BTU/hr-ft-F.

        Args:
            fouling_factor: Fouling resistance (hr-ft2-F/BTU)
            fouling_conductivity: Fouling layer conductivity (BTU/hr-ft-F)

        Returns:
            Estimated thickness in mils (thousandths of inch)
        """
        if fouling_factor <= 0:
            return None

        # Thickness in feet
        thickness_ft = fouling_factor * fouling_conductivity

        # Convert to mils (1 ft = 12000 mils)
        thickness_mils = thickness_ft * 12000

        return thickness_mils

    def _determine_cleaning_status(
        self,
        cleanliness_factor: float,
    ) -> CleaningStatus:
        """
        Determine cleaning status based on cleanliness factor.

        Args:
            cleanliness_factor: Calculated cleanliness factor

        Returns:
            CleaningStatus enum value
        """
        if cleanliness_factor >= self.fouling_config.cleanliness_warning_threshold:
            return CleaningStatus.NOT_REQUIRED
        elif cleanliness_factor >= self.fouling_config.cleanliness_alarm_threshold:
            return CleaningStatus.RECOMMENDED
        elif cleanliness_factor >= self.fouling_config.cleaning_trigger_threshold:
            return CleaningStatus.REQUIRED
        else:
            return CleaningStatus.URGENT

    def _estimate_days_to_cleaning(
        self,
        cleanliness_factor: float,
        current_status: CleaningStatus,
        fouling_rate_per_day: float = 0.001,
    ) -> Optional[int]:
        """
        Estimate days until cleaning is required.

        Uses linear extrapolation based on typical fouling rate.

        Args:
            cleanliness_factor: Current cleanliness factor
            current_status: Current cleaning status
            fouling_rate_per_day: Assumed fouling rate

        Returns:
            Estimated days to cleaning, or None if already required
        """
        if current_status in [CleaningStatus.REQUIRED, CleaningStatus.URGENT]:
            return 0

        target_cf = self.fouling_config.cleaning_trigger_threshold
        cf_margin = cleanliness_factor - target_cf

        if cf_margin <= 0:
            return 0

        # Days = margin / fouling rate
        days = int(cf_margin / fouling_rate_per_day)

        return min(days, 365)  # Cap at 1 year

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


class CleanlinessMonitor:
    """
    Monitors cleanliness trends over time.

    This class tracks cleanliness factor history and provides
    trend analysis and cleaning schedule predictions.
    """

    def __init__(
        self,
        calculator: HEICleanlinessCalculator,
        history_days: int = 90,
    ) -> None:
        """
        Initialize the cleanliness monitor.

        Args:
            calculator: HEI cleanliness calculator instance
            history_days: Days of history to maintain
        """
        self.calculator = calculator
        self.history_days = history_days
        self._history: List[Tuple[datetime, float]] = []

    def record_cleanliness(
        self,
        cleanliness_factor: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a cleanliness measurement.

        Args:
            cleanliness_factor: Measured cleanliness factor
            timestamp: Measurement timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        self._history.append((timestamp, cleanliness_factor))

        # Trim old history
        cutoff = datetime.now(timezone.utc).timestamp() - (
            self.history_days * 24 * 3600
        )
        self._history = [
            (ts, cf) for ts, cf in self._history
            if ts.timestamp() > cutoff
        ]

    def get_fouling_rate(self) -> Optional[float]:
        """
        Calculate fouling rate from history.

        Returns:
            Fouling rate (cleanliness factor loss per day)
        """
        if len(self._history) < 2:
            return None

        # Sort by timestamp
        sorted_history = sorted(self._history, key=lambda x: x[0])

        # Calculate rate using linear regression
        n = len(sorted_history)
        t0 = sorted_history[0][0].timestamp()

        sum_t = sum((h[0].timestamp() - t0) / 86400 for h in sorted_history)
        sum_cf = sum(h[1] for h in sorted_history)
        sum_t_cf = sum(
            ((h[0].timestamp() - t0) / 86400) * h[1]
            for h in sorted_history
        )
        sum_t2 = sum(
            ((h[0].timestamp() - t0) / 86400) ** 2
            for h in sorted_history
        )

        denominator = n * sum_t2 - sum_t ** 2
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_t_cf - sum_t * sum_cf) / denominator

        # Fouling rate is negative of slope (CF decreases as fouling increases)
        return -slope

    def predict_cleaning_date(
        self,
        current_cf: float,
        target_cf: Optional[float] = None,
    ) -> Optional[datetime]:
        """
        Predict when cleaning will be required.

        Args:
            current_cf: Current cleanliness factor
            target_cf: Target cleanliness for cleaning

        Returns:
            Predicted cleaning date, or None if insufficient data
        """
        fouling_rate = self.get_fouling_rate()
        if fouling_rate is None or fouling_rate <= 0:
            return None

        target = (
            target_cf or
            self.calculator.fouling_config.cleaning_trigger_threshold
        )

        if current_cf <= target:
            return datetime.now(timezone.utc)

        days_to_cleaning = (current_cf - target) / fouling_rate
        cleaning_date = datetime.now(timezone.utc).timestamp() + (
            days_to_cleaning * 86400
        )

        return datetime.fromtimestamp(cleaning_date, timezone.utc)

    def get_trend(self) -> str:
        """
        Get current fouling trend.

        Returns:
            Trend description (improving, stable, degrading)
        """
        fouling_rate = self.get_fouling_rate()
        if fouling_rate is None:
            return "unknown"

        if fouling_rate < -0.0005:
            return "improving"
        elif fouling_rate > 0.002:
            return "degrading_fast"
        elif fouling_rate > 0.0005:
            return "degrading"
        else:
            return "stable"
