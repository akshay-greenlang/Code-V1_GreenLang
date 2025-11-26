"""Steam Energy Calculator.

This module implements steam property calculations for thermal efficiency
analysis, including enthalpy from pressure/temperature, condensate return
energy, flash steam calculations, and steam quality impact.

Standards:
    - IAPWS-IF97: Industrial Formulation for Water/Steam Properties
    - ASME Steam Tables
    - ISO 12241: Thermal insulation calculation

The calculator uses simplified polynomial fits based on IAPWS-IF97
for rapid deterministic calculations.

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime
import math


class SteamPhase(Enum):
    """Phase of steam/water."""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"


@dataclass(frozen=True)
class SteamState:
    """Thermodynamic state of steam.

    Attributes:
        pressure_kpa: Absolute pressure (kPa)
        temperature_c: Temperature (Celsius)
        specific_enthalpy_kj_kg: Specific enthalpy (kJ/kg)
        specific_entropy_kj_kg_k: Specific entropy (kJ/kg-K)
        specific_volume_m3_kg: Specific volume (m3/kg)
        phase: Phase of steam
        quality: Steam quality (0-1) for wet steam
    """
    pressure_kpa: float
    temperature_c: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    phase: SteamPhase
    quality: Optional[float] = None


@dataclass(frozen=True)
class SteamQuality:
    """Steam quality (dryness fraction) details.

    Attributes:
        quality_fraction: Dryness fraction (0-1)
        liquid_fraction: Wetness fraction (1 - quality)
        enthalpy_kj_kg: Enthalpy at this quality
        moisture_percent: Moisture percentage
    """
    quality_fraction: float
    liquid_fraction: float
    enthalpy_kj_kg: float
    moisture_percent: float


@dataclass
class FlashSteamResult:
    """Result of flash steam calculation.

    Attributes:
        flash_steam_fraction: Fraction of condensate that flashes
        flash_steam_flow_kg_s: Flash steam flow rate (kg/s)
        condensate_flow_kg_s: Remaining condensate flow (kg/s)
        flash_steam_enthalpy_kj_kg: Enthalpy of flash steam
        energy_in_flash_steam_kw: Energy in flash steam (kW)
        low_pressure_kpa: Flash tank pressure
    """
    flash_steam_fraction: float
    flash_steam_flow_kg_s: float
    condensate_flow_kg_s: float
    flash_steam_enthalpy_kj_kg: float
    energy_in_flash_steam_kw: float
    low_pressure_kpa: float


@dataclass
class CondensateReturn:
    """Condensate return energy analysis.

    Attributes:
        condensate_temperature_c: Condensate temperature
        condensate_enthalpy_kj_kg: Condensate enthalpy
        energy_returned_kw: Energy returned with condensate (kW)
        energy_lost_kw: Energy lost (not returned)
        return_rate_percent: Condensate return rate (%)
        makeup_water_required_kg_s: Makeup water flow needed
    """
    condensate_temperature_c: float
    condensate_enthalpy_kj_kg: float
    energy_returned_kw: float
    energy_lost_kw: float
    return_rate_percent: float
    makeup_water_required_kg_s: float


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class SteamEnergyResult:
    """Complete steam energy calculation result.

    Attributes:
        steam_state: Thermodynamic state of steam
        mass_flow_kg_s: Steam mass flow rate (kg/s)
        energy_flow_kw: Energy flow rate (kW)
        reference_enthalpy_kj_kg: Reference enthalpy (feedwater)
        energy_added_kw: Energy added to generate steam (kW)
        flash_steam_result: Flash steam analysis (if applicable)
        condensate_return: Condensate return analysis
        calculation_steps: Audit trail
        provenance_hash: SHA-256 hash
        calculation_timestamp: When calculated
        warnings: Any warnings
    """
    steam_state: SteamState
    mass_flow_kg_s: float
    energy_flow_kw: float
    reference_enthalpy_kj_kg: float
    energy_added_kw: float
    flash_steam_result: Optional[FlashSteamResult]
    condensate_return: Optional[CondensateReturn]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_timestamp: str
    calculator_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "steam_state": {
                "pressure_kpa": self.steam_state.pressure_kpa,
                "temperature_c": self.steam_state.temperature_c,
                "enthalpy_kj_kg": self.steam_state.specific_enthalpy_kj_kg,
                "phase": self.steam_state.phase.value
            },
            "mass_flow_kg_s": self.mass_flow_kg_s,
            "energy_flow_kw": self.energy_flow_kw,
            "energy_added_kw": self.energy_added_kw,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
            "warnings": self.warnings
        }


class SteamEnergyCalculator:
    """Steam Property and Energy Calculator.

    Calculates steam properties and energy content using simplified
    IAPWS-IF97 formulations for industrial applications.

    The calculator provides:
    - Saturation properties (temperature-pressure relationship)
    - Enthalpy from P-T or P-x conditions
    - Entropy calculations
    - Flash steam calculations
    - Condensate return analysis

    Note: This uses polynomial fits for speed and determinism.
    For high-precision work, use validated IAPWS-IF97 libraries.

    Example:
        >>> calculator = SteamEnergyCalculator()
        >>> state = calculator.get_steam_properties(
        ...     pressure_kpa=1000,
        ...     temperature_c=200
        ... )
        >>> print(f"Enthalpy: {state.specific_enthalpy_kj_kg} kJ/kg")
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    # Physical constants
    CRITICAL_PRESSURE_KPA: float = 22064.0
    CRITICAL_TEMPERATURE_C: float = 373.946
    CRITICAL_DENSITY_KG_M3: float = 322.0

    # Reference conditions
    REFERENCE_TEMPERATURE_C: float = 25.0
    REFERENCE_ENTHALPY_KJ_KG: float = 104.89  # Water at 25C

    def __init__(self, precision: int = 4) -> None:
        """Initialize the Steam Energy Calculator.

        Args:
            precision: Decimal places for rounding
        """
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def get_saturation_temperature(self, pressure_kpa: float) -> float:
        """Get saturation temperature for a given pressure.

        Uses Antoine equation approximation:
            T_sat = B / (A - log10(P)) - C

        Valid range: 1 kPa to 22,064 kPa (critical point)

        Args:
            pressure_kpa: Absolute pressure (kPa)

        Returns:
            Saturation temperature (Celsius)
        """
        if pressure_kpa <= 0:
            raise ValueError("Pressure must be positive")
        if pressure_kpa > self.CRITICAL_PRESSURE_KPA:
            raise ValueError("Pressure exceeds critical point")

        # Antoine coefficients for water
        # Modified for kPa input
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Convert kPa to mmHg for Antoine equation
        p_mmhg = pressure_kpa * 7.50062

        # Antoine equation
        t_sat = B / (A - math.log10(p_mmhg)) - C

        self._add_calculation_step(
            description="Calculate saturation temperature from pressure",
            operation="antoine_equation",
            inputs={"pressure_kpa": pressure_kpa},
            output_value=t_sat,
            output_name="saturation_temperature_c",
            formula="T_sat = B / (A - log10(P)) - C"
        )

        return self._round_value(t_sat)

    def get_saturation_pressure(self, temperature_c: float) -> float:
        """Get saturation pressure for a given temperature.

        Uses Antoine equation:
            log10(P) = A - B / (T + C)

        Args:
            temperature_c: Temperature (Celsius)

        Returns:
            Saturation pressure (kPa)
        """
        if temperature_c < 0:
            raise ValueError("Temperature must be positive (Celsius)")
        if temperature_c > self.CRITICAL_TEMPERATURE_C:
            raise ValueError("Temperature exceeds critical point")

        # Antoine coefficients
        A = 8.07131
        B = 1730.63
        C = 233.426

        # Antoine equation (gives mmHg)
        log_p = A - B / (temperature_c + C)
        p_mmhg = math.pow(10, log_p)

        # Convert mmHg to kPa
        p_kpa = p_mmhg / 7.50062

        self._add_calculation_step(
            description="Calculate saturation pressure from temperature",
            operation="antoine_equation",
            inputs={"temperature_c": temperature_c},
            output_value=p_kpa,
            output_name="saturation_pressure_kpa",
            formula="log10(P) = A - B / (T + C)"
        )

        return self._round_value(p_kpa)

    def get_saturated_liquid_enthalpy(self, pressure_kpa: float) -> float:
        """Get saturated liquid enthalpy (hf) at given pressure.

        Uses polynomial fit to IAPWS-IF97 data.

        Args:
            pressure_kpa: Absolute pressure (kPa)

        Returns:
            Saturated liquid enthalpy (kJ/kg)
        """
        # Get saturation temperature
        t_sat = self.get_saturation_temperature(pressure_kpa)

        # Polynomial fit for hf vs T_sat (valid 0-350C)
        # hf = a0 + a1*T + a2*T^2 + a3*T^3
        T = t_sat
        hf = 0.0 + 4.186 * T + 0.0003 * T * T

        self._add_calculation_step(
            description="Calculate saturated liquid enthalpy (hf)",
            operation="polynomial_fit",
            inputs={"pressure_kpa": pressure_kpa, "t_sat_c": t_sat},
            output_value=hf,
            output_name="hf_kj_kg",
            formula="hf = f(T_sat)"
        )

        return self._round_value(hf)

    def get_saturated_vapor_enthalpy(self, pressure_kpa: float) -> float:
        """Get saturated vapor enthalpy (hg) at given pressure.

        Uses polynomial fit to IAPWS-IF97 data.

        Args:
            pressure_kpa: Absolute pressure (kPa)

        Returns:
            Saturated vapor enthalpy (kJ/kg)
        """
        # Get saturation temperature
        t_sat = self.get_saturation_temperature(pressure_kpa)

        # Polynomial fit for hg vs T_sat
        # At low pressures: hg ~ 2500 + 1.8*T
        # Decreases as approaching critical point
        T = t_sat
        hg = 2500.0 + 1.84 * T - 0.002 * T * T

        self._add_calculation_step(
            description="Calculate saturated vapor enthalpy (hg)",
            operation="polynomial_fit",
            inputs={"pressure_kpa": pressure_kpa, "t_sat_c": t_sat},
            output_value=hg,
            output_name="hg_kj_kg",
            formula="hg = f(T_sat)"
        )

        return self._round_value(hg)

    def get_enthalpy_of_vaporization(self, pressure_kpa: float) -> float:
        """Get enthalpy of vaporization (hfg) at given pressure.

        hfg = hg - hf

        Args:
            pressure_kpa: Absolute pressure (kPa)

        Returns:
            Enthalpy of vaporization (kJ/kg)
        """
        hf = self.get_saturated_liquid_enthalpy(pressure_kpa)
        hg = self.get_saturated_vapor_enthalpy(pressure_kpa)
        hfg = hg - hf

        self._add_calculation_step(
            description="Calculate enthalpy of vaporization (hfg)",
            operation="subtract",
            inputs={"hg_kj_kg": hg, "hf_kj_kg": hf},
            output_value=hfg,
            output_name="hfg_kj_kg",
            formula="hfg = hg - hf"
        )

        return self._round_value(hfg)

    def get_superheated_steam_enthalpy(
        self,
        pressure_kpa: float,
        temperature_c: float
    ) -> float:
        """Get superheated steam enthalpy.

        For superheated steam (T > T_sat):
            h = hg + Cp_steam * (T - T_sat)

        Args:
            pressure_kpa: Absolute pressure (kPa)
            temperature_c: Actual temperature (Celsius)

        Returns:
            Specific enthalpy (kJ/kg)
        """
        t_sat = self.get_saturation_temperature(pressure_kpa)

        if temperature_c < t_sat:
            self._warnings.append(
                f"Temperature {temperature_c}C < T_sat {t_sat}C: not superheated"
            )

        hg = self.get_saturated_vapor_enthalpy(pressure_kpa)

        # Specific heat of steam (varies with P and T, using average ~2.0 kJ/kg-K)
        cp_steam = 2.0
        superheat = max(0, temperature_c - t_sat)

        h = hg + cp_steam * superheat

        self._add_calculation_step(
            description="Calculate superheated steam enthalpy",
            operation="enthalpy_superheated",
            inputs={
                "pressure_kpa": pressure_kpa,
                "temperature_c": temperature_c,
                "t_sat_c": t_sat,
                "hg_kj_kg": hg,
                "superheat_c": superheat
            },
            output_value=h,
            output_name="h_superheated_kj_kg",
            formula="h = hg + Cp x (T - T_sat)"
        )

        return self._round_value(h)

    def get_wet_steam_enthalpy(
        self,
        pressure_kpa: float,
        quality: float
    ) -> float:
        """Get enthalpy of wet steam (two-phase mixture).

        h = hf + x * hfg

        Where x is the steam quality (dryness fraction).

        Args:
            pressure_kpa: Absolute pressure (kPa)
            quality: Steam quality, 0-1 (0 = all liquid, 1 = all vapor)

        Returns:
            Specific enthalpy (kJ/kg)
        """
        if quality < 0 or quality > 1:
            raise ValueError("Quality must be between 0 and 1")

        hf = self.get_saturated_liquid_enthalpy(pressure_kpa)
        hfg = self.get_enthalpy_of_vaporization(pressure_kpa)

        h = hf + quality * hfg

        self._add_calculation_step(
            description="Calculate wet steam enthalpy",
            operation="wet_steam_enthalpy",
            inputs={
                "pressure_kpa": pressure_kpa,
                "quality": quality,
                "hf_kj_kg": hf,
                "hfg_kj_kg": hfg
            },
            output_value=h,
            output_name="h_wet_kj_kg",
            formula="h = hf + x * hfg"
        )

        return self._round_value(h)

    def get_steam_properties(
        self,
        pressure_kpa: float,
        temperature_c: Optional[float] = None,
        quality: Optional[float] = None
    ) -> SteamState:
        """Get complete steam state properties.

        Determines phase and calculates all properties.

        Args:
            pressure_kpa: Absolute pressure (kPa)
            temperature_c: Temperature (C) for superheated/subcooled
            quality: Quality for wet steam (0-1)

        Returns:
            SteamState with all properties
        """
        self._reset_calculation_state()

        t_sat = self.get_saturation_temperature(pressure_kpa)

        # Determine phase and calculate enthalpy
        if quality is not None:
            # Wet steam specified
            phase = SteamPhase.WET_STEAM
            temperature_c = t_sat
            enthalpy = self.get_wet_steam_enthalpy(pressure_kpa, quality)

        elif temperature_c is not None:
            if temperature_c > t_sat + 0.1:
                # Superheated vapor
                phase = SteamPhase.SUPERHEATED_VAPOR
                enthalpy = self.get_superheated_steam_enthalpy(pressure_kpa, temperature_c)
                quality = None
            elif temperature_c < t_sat - 0.1:
                # Subcooled liquid
                phase = SteamPhase.SUBCOOLED_LIQUID
                # Approximate as saturated liquid at that temperature
                p_sat = self.get_saturation_pressure(temperature_c)
                enthalpy = self.get_saturated_liquid_enthalpy(p_sat)
                quality = 0.0
            else:
                # At saturation - assume saturated vapor
                phase = SteamPhase.SATURATED_VAPOR
                temperature_c = t_sat
                enthalpy = self.get_saturated_vapor_enthalpy(pressure_kpa)
                quality = 1.0
        else:
            # No temperature or quality - assume saturated vapor
            phase = SteamPhase.SATURATED_VAPOR
            temperature_c = t_sat
            enthalpy = self.get_saturated_vapor_enthalpy(pressure_kpa)
            quality = 1.0

        # Estimate entropy (simplified)
        if phase == SteamPhase.SATURATED_VAPOR:
            entropy = 6.5 + 0.005 * (100 - pressure_kpa / 100)  # Approximate
        elif phase == SteamPhase.SUPERHEATED_VAPOR:
            entropy = 7.0 + 0.002 * (temperature_c - t_sat)
        else:
            entropy = 1.3 + 0.003 * temperature_c

        # Estimate specific volume
        if phase in [SteamPhase.SATURATED_VAPOR, SteamPhase.SUPERHEATED_VAPOR]:
            # Ideal gas approximation for vapor
            R = 0.4615  # kJ/kg-K for water
            T_k = temperature_c + 273.15
            specific_volume = R * T_k / pressure_kpa
        else:
            # Liquid is approximately incompressible
            specific_volume = 0.001  # m3/kg

        return SteamState(
            pressure_kpa=self._round_value(pressure_kpa),
            temperature_c=self._round_value(temperature_c),
            specific_enthalpy_kj_kg=enthalpy,
            specific_entropy_kj_kg_k=self._round_value(entropy),
            specific_volume_m3_kg=self._round_value(specific_volume, 6),
            phase=phase,
            quality=quality
        )

    def calculate_steam_energy(
        self,
        pressure_kpa: float,
        mass_flow_kg_s: float,
        temperature_c: Optional[float] = None,
        quality: Optional[float] = None,
        feedwater_temperature_c: float = 25.0
    ) -> SteamEnergyResult:
        """Calculate steam energy flow rate.

        Energy = mass_flow * (h_steam - h_feedwater)

        Args:
            pressure_kpa: Steam pressure (kPa)
            mass_flow_kg_s: Steam mass flow (kg/s)
            temperature_c: Steam temperature (C) if superheated
            quality: Steam quality if wet steam
            feedwater_temperature_c: Feedwater temperature for reference

        Returns:
            SteamEnergyResult with complete analysis
        """
        self._reset_calculation_state()

        # Get steam properties
        steam_state = self.get_steam_properties(pressure_kpa, temperature_c, quality)

        # Reference enthalpy (feedwater)
        p_fw = self.get_saturation_pressure(feedwater_temperature_c)
        h_ref = self.get_saturated_liquid_enthalpy(p_fw)

        # Energy flow rate
        energy_flow = mass_flow_kg_s * steam_state.specific_enthalpy_kj_kg

        # Energy added (above feedwater)
        energy_added = mass_flow_kg_s * (steam_state.specific_enthalpy_kj_kg - h_ref)

        self._add_calculation_step(
            description="Calculate steam energy flow",
            operation="energy_flow",
            inputs={
                "mass_flow_kg_s": mass_flow_kg_s,
                "steam_enthalpy_kj_kg": steam_state.specific_enthalpy_kj_kg,
                "reference_enthalpy_kj_kg": h_ref
            },
            output_value=energy_flow,
            output_name="energy_flow_kw",
            formula="Q = m_dot x h_steam"
        )

        # Generate provenance
        provenance = self._generate_provenance_hash(
            pressure_kpa, mass_flow_kg_s, steam_state.specific_enthalpy_kj_kg
        )
        timestamp = datetime.utcnow().isoformat() + "Z"

        return SteamEnergyResult(
            steam_state=steam_state,
            mass_flow_kg_s=mass_flow_kg_s,
            energy_flow_kw=self._round_value(energy_flow),
            reference_enthalpy_kj_kg=self._round_value(h_ref),
            energy_added_kw=self._round_value(energy_added),
            flash_steam_result=None,
            condensate_return=None,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance,
            calculation_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def calculate_flash_steam(
        self,
        condensate_pressure_kpa: float,
        condensate_flow_kg_s: float,
        flash_tank_pressure_kpa: float
    ) -> FlashSteamResult:
        """Calculate flash steam from high-pressure condensate.

        When hot condensate at high pressure is released to lower
        pressure, some of it "flashes" to steam.

        Flash Fraction = (hf_high - hf_low) / hfg_low

        Args:
            condensate_pressure_kpa: Inlet condensate pressure (kPa)
            condensate_flow_kg_s: Condensate flow rate (kg/s)
            flash_tank_pressure_kpa: Flash tank pressure (kPa)

        Returns:
            FlashSteamResult with flash steam analysis
        """
        if flash_tank_pressure_kpa >= condensate_pressure_kpa:
            raise ValueError("Flash tank pressure must be lower than condensate pressure")

        # Get enthalpies
        hf_high = self.get_saturated_liquid_enthalpy(condensate_pressure_kpa)
        hf_low = self.get_saturated_liquid_enthalpy(flash_tank_pressure_kpa)
        hfg_low = self.get_enthalpy_of_vaporization(flash_tank_pressure_kpa)
        hg_low = self.get_saturated_vapor_enthalpy(flash_tank_pressure_kpa)

        # Flash fraction
        flash_fraction = (hf_high - hf_low) / hfg_low
        flash_fraction = min(max(0, flash_fraction), 1)  # Clamp to 0-1

        # Flow rates
        flash_steam_flow = condensate_flow_kg_s * flash_fraction
        remaining_condensate = condensate_flow_kg_s * (1 - flash_fraction)

        # Energy in flash steam
        energy_flash = flash_steam_flow * hg_low

        self._add_calculation_step(
            description="Calculate flash steam fraction",
            operation="flash_steam",
            inputs={
                "hf_high_kj_kg": hf_high,
                "hf_low_kj_kg": hf_low,
                "hfg_low_kj_kg": hfg_low,
                "condensate_flow_kg_s": condensate_flow_kg_s
            },
            output_value=flash_fraction,
            output_name="flash_fraction",
            formula="x_flash = (hf_high - hf_low) / hfg_low"
        )

        return FlashSteamResult(
            flash_steam_fraction=self._round_value(flash_fraction),
            flash_steam_flow_kg_s=self._round_value(flash_steam_flow),
            condensate_flow_kg_s=self._round_value(remaining_condensate),
            flash_steam_enthalpy_kj_kg=self._round_value(hg_low),
            energy_in_flash_steam_kw=self._round_value(energy_flash),
            low_pressure_kpa=flash_tank_pressure_kpa
        )

    def calculate_condensate_return(
        self,
        steam_flow_kg_s: float,
        steam_pressure_kpa: float,
        condensate_return_percent: float,
        condensate_temperature_c: Optional[float] = None,
        makeup_water_temperature_c: float = 15.0
    ) -> CondensateReturn:
        """Calculate condensate return energy savings.

        Returning hot condensate to the boiler reduces:
        - Fuel consumption (less heating required)
        - Water treatment costs
        - Makeup water costs

        Args:
            steam_flow_kg_s: Steam flow rate (kg/s)
            steam_pressure_kpa: Steam pressure for condensate temp
            condensate_return_percent: Percentage of condensate returned
            condensate_temperature_c: Condensate temperature (if known)
            makeup_water_temperature_c: Makeup water temperature

        Returns:
            CondensateReturn with energy analysis
        """
        # Condensate temperature (saturated liquid if not specified)
        if condensate_temperature_c is None:
            condensate_temperature_c = self.get_saturation_temperature(steam_pressure_kpa)

        # Get enthalpies
        p_cond = self.get_saturation_pressure(condensate_temperature_c)
        h_condensate = self.get_saturated_liquid_enthalpy(p_cond)

        p_makeup = self.get_saturation_pressure(makeup_water_temperature_c)
        h_makeup = self.get_saturated_liquid_enthalpy(p_makeup)

        # Calculate flows
        return_fraction = condensate_return_percent / 100
        condensate_returned_kg_s = steam_flow_kg_s * return_fraction
        makeup_required_kg_s = steam_flow_kg_s * (1 - return_fraction)

        # Energy calculations
        # Energy returned = condensate flow * (h_cond - h_makeup)
        energy_returned = condensate_returned_kg_s * (h_condensate - h_makeup)

        # Energy lost = makeup required * (h_cond - h_makeup)
        # (This is the energy we "would have had" if all condensate returned)
        energy_lost = makeup_required_kg_s * (h_condensate - h_makeup)

        self._add_calculation_step(
            description="Calculate condensate return energy",
            operation="condensate_return",
            inputs={
                "steam_flow_kg_s": steam_flow_kg_s,
                "return_percent": condensate_return_percent,
                "h_condensate_kj_kg": h_condensate,
                "h_makeup_kj_kg": h_makeup
            },
            output_value=energy_returned,
            output_name="energy_returned_kw",
            formula="Q = m_cond x (h_cond - h_makeup)"
        )

        return CondensateReturn(
            condensate_temperature_c=self._round_value(condensate_temperature_c),
            condensate_enthalpy_kj_kg=self._round_value(h_condensate),
            energy_returned_kw=self._round_value(energy_returned),
            energy_lost_kw=self._round_value(energy_lost),
            return_rate_percent=condensate_return_percent,
            makeup_water_required_kg_s=self._round_value(makeup_required_kg_s)
        )

    def calculate_steam_quality_impact(
        self,
        pressure_kpa: float,
        actual_quality: float,
        target_quality: float = 1.0
    ) -> SteamQuality:
        """Calculate impact of steam quality on energy content.

        Low quality steam (wet steam) has less usable energy.

        Args:
            pressure_kpa: Steam pressure (kPa)
            actual_quality: Actual steam quality (0-1)
            target_quality: Target/design quality (typically 1.0)

        Returns:
            SteamQuality analysis
        """
        h_actual = self.get_wet_steam_enthalpy(pressure_kpa, actual_quality)
        h_target = self.get_wet_steam_enthalpy(pressure_kpa, target_quality)

        moisture_percent = (1 - actual_quality) * 100
        liquid_fraction = 1 - actual_quality

        self._add_calculation_step(
            description="Analyze steam quality impact",
            operation="quality_analysis",
            inputs={
                "pressure_kpa": pressure_kpa,
                "actual_quality": actual_quality,
                "target_quality": target_quality
            },
            output_value=h_actual,
            output_name="actual_enthalpy_kj_kg",
            formula="h = hf + x * hfg"
        )

        return SteamQuality(
            quality_fraction=actual_quality,
            liquid_fraction=self._round_value(liquid_fraction),
            enthalpy_kj_kg=h_actual,
            moisture_percent=self._round_value(moisture_percent)
        )

    def _reset_calculation_state(self) -> None:
        """Reset calculation state."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        pressure: float,
        flow: float,
        enthalpy: float
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "calculator": "SteamEnergyCalculator",
            "version": self.VERSION,
            "pressure_kpa": pressure,
            "mass_flow_kg_s": flow,
            "enthalpy_kj_kg": enthalpy
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float, precision: Optional[int] = None) -> float:
        """Round value to precision."""
        if precision is None:
            precision = self.precision

        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
