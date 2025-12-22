"""
GL-002 FLAMEGUARD - Efficiency Calculator

Boiler efficiency calculations per ASME PTC 4.1 (Fired Steam Generators).

Implements both direct and indirect (heat loss) methods with
deterministic, reproducible calculations and full provenance tracking.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class FuelProperties:
    """Fuel chemical and physical properties."""

    fuel_type: str
    higher_heating_value_btu_lb: float
    lower_heating_value_btu_lb: float
    carbon_percent: float
    hydrogen_percent: float
    sulfur_percent: float
    nitrogen_percent: float
    oxygen_percent: float
    moisture_percent: float
    ash_percent: float
    stoichiometric_air_ratio: float  # lb air / lb fuel


# Standard fuel properties database
FUEL_DATABASE = {
    "natural_gas": FuelProperties(
        fuel_type="natural_gas",
        higher_heating_value_btu_lb=23875.0,
        lower_heating_value_btu_lb=21500.0,
        carbon_percent=75.0,
        hydrogen_percent=25.0,
        sulfur_percent=0.0,
        nitrogen_percent=0.0,
        oxygen_percent=0.0,
        moisture_percent=0.0,
        ash_percent=0.0,
        stoichiometric_air_ratio=17.2,
    ),
    "fuel_oil_no2": FuelProperties(
        fuel_type="fuel_oil_no2",
        higher_heating_value_btu_lb=19500.0,
        lower_heating_value_btu_lb=18300.0,
        carbon_percent=86.5,
        hydrogen_percent=12.5,
        sulfur_percent=0.3,
        nitrogen_percent=0.1,
        oxygen_percent=0.1,
        moisture_percent=0.5,
        ash_percent=0.01,
        stoichiometric_air_ratio=14.1,
    ),
    "coal_bituminous": FuelProperties(
        fuel_type="coal_bituminous",
        higher_heating_value_btu_lb=12500.0,
        lower_heating_value_btu_lb=11800.0,
        carbon_percent=70.0,
        hydrogen_percent=5.0,
        sulfur_percent=2.0,
        nitrogen_percent=1.5,
        oxygen_percent=7.5,
        moisture_percent=5.0,
        ash_percent=9.0,
        stoichiometric_air_ratio=9.5,
    ),
}


@dataclass
class EfficiencyInput:
    """Input data for efficiency calculation."""

    # Process conditions
    steam_flow_klb_hr: float
    steam_pressure_psig: float
    steam_temperature_f: float
    feedwater_temperature_f: float
    fuel_flow_rate: float  # lb/hr or scfh
    fuel_flow_unit: str = "lb_hr"

    # Flue gas analysis
    flue_gas_temperature_f: float
    flue_gas_o2_percent: float
    flue_gas_co_ppm: float = 0.0
    flue_gas_co2_percent: float = 0.0

    # Ambient conditions
    ambient_temperature_f: float = 77.0
    ambient_humidity_percent: float = 60.0
    barometric_pressure_psia: float = 14.696

    # Combustion air
    combustion_air_temperature_f: float = 80.0

    # Blowdown
    blowdown_rate_percent: float = 3.0

    # Fuel
    fuel_type: str = "natural_gas"
    fuel_properties: Optional[FuelProperties] = None

    # Ash (for solid fuels)
    ash_unburned_carbon_percent: float = 2.0


@dataclass
class EfficiencyResult:
    """Boiler efficiency calculation result."""

    # Identification
    calculation_id: str
    timestamp: datetime
    method: str  # "direct" or "indirect"

    # Overall efficiency
    efficiency_hhv_percent: float
    efficiency_lhv_percent: float
    uncertainty_percent: float

    # Heat flows
    fuel_input_mmbtu_hr: float
    steam_output_mmbtu_hr: float
    total_losses_mmbtu_hr: float

    # Individual losses (percent of fuel input)
    dry_flue_gas_loss_percent: float
    moisture_in_fuel_loss_percent: float
    hydrogen_combustion_loss_percent: float
    moisture_in_air_loss_percent: float
    unburned_carbon_loss_percent: float
    co_loss_percent: float
    radiation_loss_percent: float
    blowdown_loss_percent: float
    other_losses_percent: float
    total_losses_percent: float

    # Derived values
    excess_air_percent: float
    air_fuel_ratio: float
    flue_gas_mass_flow_lb_hr: float

    # Provenance
    input_hash: str
    output_hash: str
    formula_version: str = "ASME_PTC_4.1_2013"


class EfficiencyCalculator:
    """
    ASME PTC 4.1 compliant boiler efficiency calculator.

    Provides deterministic, reproducible calculations with:
    - Direct method (heat output / heat input)
    - Indirect method (100% - losses)
    - Full provenance tracking via SHA-256 hashes
    - Uncertainty analysis

    Example:
        >>> calc = EfficiencyCalculator()
        >>> input_data = EfficiencyInput(
        ...     steam_flow_klb_hr=100.0,
        ...     steam_pressure_psig=150.0,
        ...     steam_temperature_f=366.0,
        ...     feedwater_temperature_f=227.0,
        ...     fuel_flow_rate=8000.0,
        ...     flue_gas_temperature_f=400.0,
        ...     flue_gas_o2_percent=3.0,
        ... )
        >>> result = calc.calculate(input_data)
        >>> print(f"Efficiency: {result.efficiency_hhv_percent:.2f}%")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "ASME_PTC_4.1_2013"

    def __init__(
        self,
        fuel_database: Optional[Dict[str, FuelProperties]] = None,
    ) -> None:
        """
        Initialize the efficiency calculator.

        Args:
            fuel_database: Custom fuel properties database
        """
        self.fuel_database = fuel_database or FUEL_DATABASE

    def calculate(
        self,
        input_data: EfficiencyInput,
        method: str = "indirect",
    ) -> EfficiencyResult:
        """
        Calculate boiler efficiency.

        Args:
            input_data: Calculation inputs
            method: "direct" or "indirect"

        Returns:
            EfficiencyResult with all losses and provenance
        """
        # Get fuel properties
        fuel = input_data.fuel_properties or self.fuel_database.get(
            input_data.fuel_type,
            FUEL_DATABASE["natural_gas"],
        )

        # Compute input hash for provenance
        input_hash = self._compute_hash(input_data.__dict__)

        # Calculate based on method
        if method == "direct":
            result = self._calculate_direct(input_data, fuel)
        else:
            result = self._calculate_indirect(input_data, fuel)

        # Add provenance
        result.calculation_id = f"EFF-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        result.timestamp = datetime.now(timezone.utc)
        result.method = method
        result.input_hash = input_hash
        result.output_hash = self._compute_hash({
            "efficiency": result.efficiency_hhv_percent,
            "losses": result.total_losses_percent,
        })
        result.formula_version = self.FORMULA_VERSION

        return result

    def _calculate_indirect(
        self,
        inp: EfficiencyInput,
        fuel: FuelProperties,
    ) -> EfficiencyResult:
        """
        Calculate efficiency using indirect (heat loss) method.

        Per ASME PTC 4.1:
        Efficiency = 100% - (sum of all losses)
        """
        # Convert fuel flow to lb/hr if needed
        if inp.fuel_flow_unit == "scfh":
            # Natural gas: ~0.042 lb/scf at STP
            fuel_mass_flow = inp.fuel_flow_rate * 0.042
        else:
            fuel_mass_flow = inp.fuel_flow_rate

        # Calculate fuel heat input
        fuel_input_btu_hr = fuel_mass_flow * fuel.higher_heating_value_btu_lb
        fuel_input_mmbtu = fuel_input_btu_hr / 1e6

        # Calculate excess air from O2
        excess_air = self._o2_to_excess_air(inp.flue_gas_o2_percent)
        actual_air_ratio = fuel.stoichiometric_air_ratio * (1 + excess_air / 100)

        # Mass flow of combustion products
        actual_air_mass = fuel_mass_flow * actual_air_ratio
        flue_gas_mass = fuel_mass_flow + actual_air_mass

        # ======================
        # LOSS CALCULATIONS
        # ======================

        # 1. Dry Flue Gas Loss (L1)
        # L1 = Cp_fg * (Tfg - Ta) * m_fg / Q_fuel * 100
        cp_flue_gas = 0.24  # BTU/(lb·°F) average
        temp_diff = inp.flue_gas_temperature_f - inp.ambient_temperature_f
        l1_dry_flue_gas = (
            cp_flue_gas * temp_diff * actual_air_ratio /
            fuel.higher_heating_value_btu_lb * 100
        )

        # 2. Moisture in Fuel Loss (L2)
        # Includes latent heat of vaporization
        if fuel.moisture_percent > 0:
            # 1055 BTU/lb latent heat at 212°F
            # Plus sensible heat to stack temp
            moisture_enthalpy = (
                1055 +
                0.46 * (inp.flue_gas_temperature_f - 212)
            )
            l2_moisture_fuel = (
                fuel.moisture_percent / 100 * moisture_enthalpy /
                fuel.higher_heating_value_btu_lb * 100
            )
        else:
            l2_moisture_fuel = 0.0

        # 3. Hydrogen Combustion Moisture Loss (L3)
        # 9 lb H2O per lb H2 combusted
        h2_water = 9 * fuel.hydrogen_percent / 100
        h2_moisture_enthalpy = (
            1055 +
            0.46 * (inp.flue_gas_temperature_f - 212)
        )
        l3_hydrogen = h2_water * h2_moisture_enthalpy / fuel.higher_heating_value_btu_lb * 100

        # 4. Moisture in Air Loss (L4)
        # Assume specific humidity from RH
        # At 80°F, 60% RH: ~0.013 lb moisture/lb dry air
        humidity_ratio = 0.013 * inp.ambient_humidity_percent / 60
        air_moisture = humidity_ratio * actual_air_ratio
        l4_air_moisture = (
            air_moisture * 0.46 * temp_diff /
            fuel.higher_heating_value_btu_lb * 100
        )

        # 5. Unburned Carbon Loss (L5)
        # For solid fuels only
        if fuel.ash_percent > 0:
            # Unburned carbon heating value: 14,500 BTU/lb
            unburned_c = (
                fuel.ash_percent / 100 *
                inp.ash_unburned_carbon_percent / 100
            )
            l5_unburned_carbon = (
                unburned_c * 14500 /
                fuel.higher_heating_value_btu_lb * 100
            )
        else:
            l5_unburned_carbon = 0.0

        # 6. CO Loss (L6)
        # CO heating value: 10,100 BTU/lb
        if inp.flue_gas_co_ppm > 0:
            # Approximate CO mass fraction
            co_mass_frac = inp.flue_gas_co_ppm / 1e6 * 28 / 29
            l6_co = (
                co_mass_frac * actual_air_ratio * 10100 /
                fuel.higher_heating_value_btu_lb * 100
            )
        else:
            l6_co = 0.0

        # 7. Radiation and Convection Loss (L7)
        # Use ABMA correlation for surface losses
        # L7 ≈ 1.5 / (capacity)^0.15 for capacity in MMBTU/hr
        if fuel_input_mmbtu > 0:
            l7_radiation = 1.5 / (fuel_input_mmbtu ** 0.15)
            l7_radiation = max(0.3, min(3.0, l7_radiation))
        else:
            l7_radiation = 1.0

        # 8. Blowdown Loss (L8)
        # Heat lost in blowdown water
        if inp.blowdown_rate_percent > 0:
            # Steam enthalpy at operating pressure
            steam_h = self._steam_enthalpy(inp.steam_pressure_psig)
            fw_h = inp.feedwater_temperature_f - 32  # Approximate
            steam_output_btu = (
                inp.steam_flow_klb_hr * 1000 *
                (steam_h - fw_h)
            )
            blowdown_heat = (
                inp.blowdown_rate_percent / 100 *
                steam_output_btu *
                (steam_h - fw_h) / steam_h
            )
            l8_blowdown = blowdown_heat / fuel_input_btu_hr * 100 if fuel_input_btu_hr > 0 else 0
        else:
            l8_blowdown = 0.0

        # 9. Other losses (L9) - manufacturer margin
        l9_other = 0.5

        # Total losses
        total_losses = (
            l1_dry_flue_gas +
            l2_moisture_fuel +
            l3_hydrogen +
            l4_air_moisture +
            l5_unburned_carbon +
            l6_co +
            l7_radiation +
            l8_blowdown +
            l9_other
        )

        # Efficiency
        efficiency_hhv = 100 - total_losses
        efficiency_hhv = max(50.0, min(100.0, efficiency_hhv))

        # LHV efficiency (typically ~10% higher for gas)
        hhv_lhv_ratio = (
            fuel.higher_heating_value_btu_lb /
            fuel.lower_heating_value_btu_lb
        )
        efficiency_lhv = min(100.0, efficiency_hhv * hhv_lhv_ratio)

        # Steam output
        steam_h = self._steam_enthalpy(inp.steam_pressure_psig)
        fw_h = inp.feedwater_temperature_f - 32
        steam_output_mmbtu = (
            inp.steam_flow_klb_hr * 1000 *
            (steam_h - fw_h) / 1e6
        )

        # Uncertainty (simplified)
        # Per ASME PTC 4.1, typical uncertainty ~0.5-1.0%
        uncertainty = 0.5 + 0.1 * (100 - efficiency_hhv) / 20

        return EfficiencyResult(
            calculation_id="",  # Set later
            timestamp=datetime.now(timezone.utc),
            method="indirect",
            efficiency_hhv_percent=round(efficiency_hhv, 2),
            efficiency_lhv_percent=round(efficiency_lhv, 2),
            uncertainty_percent=round(uncertainty, 2),
            fuel_input_mmbtu_hr=round(fuel_input_mmbtu, 3),
            steam_output_mmbtu_hr=round(steam_output_mmbtu, 3),
            total_losses_mmbtu_hr=round(fuel_input_mmbtu * total_losses / 100, 3),
            dry_flue_gas_loss_percent=round(l1_dry_flue_gas, 2),
            moisture_in_fuel_loss_percent=round(l2_moisture_fuel, 2),
            hydrogen_combustion_loss_percent=round(l3_hydrogen, 2),
            moisture_in_air_loss_percent=round(l4_air_moisture, 2),
            unburned_carbon_loss_percent=round(l5_unburned_carbon, 2),
            co_loss_percent=round(l6_co, 3),
            radiation_loss_percent=round(l7_radiation, 2),
            blowdown_loss_percent=round(l8_blowdown, 2),
            other_losses_percent=round(l9_other, 2),
            total_losses_percent=round(total_losses, 2),
            excess_air_percent=round(excess_air, 1),
            air_fuel_ratio=round(actual_air_ratio, 2),
            flue_gas_mass_flow_lb_hr=round(flue_gas_mass, 0),
            input_hash="",
            output_hash="",
        )

    def _calculate_direct(
        self,
        inp: EfficiencyInput,
        fuel: FuelProperties,
    ) -> EfficiencyResult:
        """
        Calculate efficiency using direct method.

        Efficiency = Steam Output / Fuel Input * 100
        """
        # Fuel input
        if inp.fuel_flow_unit == "scfh":
            fuel_mass_flow = inp.fuel_flow_rate * 0.042
        else:
            fuel_mass_flow = inp.fuel_flow_rate

        fuel_input_btu_hr = fuel_mass_flow * fuel.higher_heating_value_btu_lb
        fuel_input_mmbtu = fuel_input_btu_hr / 1e6

        # Steam output
        steam_h = self._steam_enthalpy(inp.steam_pressure_psig)
        fw_h = inp.feedwater_temperature_f - 32
        steam_output_btu = inp.steam_flow_klb_hr * 1000 * (steam_h - fw_h)
        steam_output_mmbtu = steam_output_btu / 1e6

        # Efficiency
        if fuel_input_btu_hr > 0:
            efficiency_hhv = steam_output_btu / fuel_input_btu_hr * 100
        else:
            efficiency_hhv = 0.0

        efficiency_hhv = max(50.0, min(100.0, efficiency_hhv))

        # LHV efficiency
        hhv_lhv_ratio = (
            fuel.higher_heating_value_btu_lb /
            fuel.lower_heating_value_btu_lb
        )
        efficiency_lhv = min(100.0, efficiency_hhv * hhv_lhv_ratio)

        # For direct method, losses are calculated by difference
        total_losses = 100 - efficiency_hhv

        return EfficiencyResult(
            calculation_id="",
            timestamp=datetime.now(timezone.utc),
            method="direct",
            efficiency_hhv_percent=round(efficiency_hhv, 2),
            efficiency_lhv_percent=round(efficiency_lhv, 2),
            uncertainty_percent=1.0,  # Direct method typically less accurate
            fuel_input_mmbtu_hr=round(fuel_input_mmbtu, 3),
            steam_output_mmbtu_hr=round(steam_output_mmbtu, 3),
            total_losses_mmbtu_hr=round(fuel_input_mmbtu * total_losses / 100, 3),
            dry_flue_gas_loss_percent=0.0,
            moisture_in_fuel_loss_percent=0.0,
            hydrogen_combustion_loss_percent=0.0,
            moisture_in_air_loss_percent=0.0,
            unburned_carbon_loss_percent=0.0,
            co_loss_percent=0.0,
            radiation_loss_percent=0.0,
            blowdown_loss_percent=0.0,
            other_losses_percent=total_losses,
            total_losses_percent=round(total_losses, 2),
            excess_air_percent=self._o2_to_excess_air(inp.flue_gas_o2_percent),
            air_fuel_ratio=fuel.stoichiometric_air_ratio * (1 + self._o2_to_excess_air(inp.flue_gas_o2_percent) / 100),
            flue_gas_mass_flow_lb_hr=0.0,
            input_hash="",
            output_hash="",
        )

    def _o2_to_excess_air(self, o2_percent: float) -> float:
        """Convert O2 percentage to excess air percentage."""
        if o2_percent >= 21:
            return 500.0
        if o2_percent <= 0:
            return 0.0
        return o2_percent / (21 - o2_percent) * 100

    def _steam_enthalpy(self, pressure_psig: float) -> float:
        """
        Approximate steam enthalpy at saturation.

        Uses polynomial fit for saturated steam tables.
        """
        psia = pressure_psig + 14.696
        # Polynomial approximation for hg (BTU/lb) vs pressure
        # Valid 15-300 psia
        if psia < 15:
            return 1150.0
        elif psia > 300:
            return 1204.0
        else:
            # Fit: hg ≈ 1150 + 0.3*P - 0.0005*P^2
            return 1150 + 0.3 * psia - 0.0005 * psia ** 2

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
