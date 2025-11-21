# -*- coding: utf-8 -*-
"""
Heat Output Calculator for GL-005 CombustionControlAgent

Calculates actual heat output, heat release rate, and thermal efficiency.
Zero-hallucination design using thermodynamic principles and heat transfer equations.

Reference Standards:
- ASME PTC 4.1: Fired Steam Generators - Performance Test Codes
- ISO 9001: Heat Balance Calculation Method
- DIN EN 12952: Water-tube boilers - Heat balance calculation
- ASHRAE Fundamentals: Heat Transfer chapter

Mathematical Formulas:
- Heat Release Rate: Q̇ = ṁ_fuel * LHV * η_combustion
- Thermal Efficiency: η = Q_useful / Q_input = Q_useful / (ṁ_fuel * LHV)
- Heat Loss (Stack): Q_stack = ṁ_flue * Cp * (T_flue - T_ambient)
- Radiation Loss: Q_rad = ε * σ * A * (T_surface^4 - T_ambient^4)
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HeatLossCategory(str, Enum):
    """Categories of heat loss"""
    STACK_LOSS = "stack_loss"
    RADIATION_LOSS = "radiation_loss"
    CONVECTION_LOSS = "convection_loss"
    INCOMPLETE_COMBUSTION = "incomplete_combustion"
    UNBURNED_FUEL = "unburned_fuel"
    MOISTURE_LOSS = "moisture_loss"
    BLOWDOWN_LOSS = "blowdown_loss"


@dataclass
class HeatLossComponent:
    """Individual heat loss component"""
    category: HeatLossCategory
    loss_kw: float
    loss_percent: float
    description: str


class HeatOutputInput(BaseModel):
    """Input parameters for heat output calculation"""

    # Fuel input
    fuel_flow_rate_kg_per_hr: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Fuel flow rate in kg/hr"
    )
    fuel_lower_heating_value_mj_per_kg: float = Field(
        ...,
        gt=0,
        le=100,
        description="Lower heating value of fuel in MJ/kg"
    )
    fuel_higher_heating_value_mj_per_kg: Optional[float] = Field(
        None,
        gt=0,
        le=100,
        description="Higher heating value of fuel in MJ/kg (optional)"
    )

    # Air input
    air_flow_rate_kg_per_hr: float = Field(
        ...,
        ge=0,
        description="Air flow rate in kg/hr"
    )

    # Flue gas measurements
    flue_gas_temperature_c: float = Field(
        ...,
        ge=0,
        le=2000,
        description="Flue gas temperature in Celsius"
    )
    flue_gas_o2_percent: float = Field(
        ...,
        ge=0,
        le=21,
        description="O2 in flue gas (dry basis) in percent"
    )
    flue_gas_co2_percent: Optional[float] = Field(
        None,
        ge=0,
        le=20,
        description="CO2 in flue gas (dry basis) in percent"
    )
    flue_gas_co_ppm: float = Field(
        default=0,
        ge=0,
        le=10000,
        description="CO in flue gas in ppm"
    )

    # Environmental conditions
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50,
        le=60,
        description="Ambient temperature in Celsius"
    )
    ambient_pressure_pa: float = Field(
        default=101325,
        ge=80000,
        le=110000,
        description="Ambient pressure in Pa"
    )

    # Equipment parameters
    combustor_surface_area_m2: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="External surface area for radiation loss calculation"
    )
    surface_temperature_c: Optional[float] = Field(
        None,
        ge=0,
        le=500,
        description="External surface temperature in Celsius"
    )
    surface_emissivity: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Surface emissivity (0-1)"
    )

    # Fuel composition for moisture calculation
    fuel_hydrogen_percent: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Hydrogen content in fuel (mass %)"
    )
    fuel_moisture_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Moisture content in fuel (mass %)"
    )

    # Target heat output (for validation)
    target_heat_output_kw: Optional[float] = Field(
        None,
        gt=0,
        description="Target heat output in kW (for comparison)"
    )


class HeatOutputResult(BaseModel):
    """Heat output calculation results"""

    # Heat input
    gross_heat_input_kw: float = Field(
        ...,
        description="Gross heat input (HHV basis) in kW"
    )
    net_heat_input_kw: float = Field(
        ...,
        description="Net heat input (LHV basis) in kW"
    )

    # Heat output
    net_heat_output_kw: float = Field(
        ...,
        description="Net useful heat output in kW"
    )
    heat_release_rate_kw_per_m3: Optional[float] = Field(
        None,
        description="Volumetric heat release rate in kW/m³"
    )

    # Efficiency
    gross_thermal_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Gross efficiency (HHV basis)"
    )
    net_thermal_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Net efficiency (LHV basis)"
    )
    combustion_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Combustion efficiency"
    )

    # Heat losses
    total_heat_loss_kw: float
    total_heat_loss_percent: float
    heat_losses: List[Dict[str, any]] = Field(
        default_factory=list,
        description="Detailed breakdown of heat losses"
    )

    # Stack loss details
    stack_loss_kw: float
    stack_loss_percent: float
    stack_gas_flow_rate_kg_per_hr: float

    # Performance metrics
    excess_air_percent: float
    specific_fuel_consumption_kg_per_kwh: float

    # Validation
    target_deviation_percent: Optional[float] = Field(
        None,
        description="Deviation from target heat output"
    )
    meets_target: Optional[bool] = Field(
        None,
        description="Whether output meets target (within ±5%)"
    )

    # Recommendations
    recommendations: List[str]


class HeatOutputCalculator:
    """
    Calculate actual heat output and thermal efficiency using heat balance method.

    This calculator uses ASME PTC 4.1 indirect method (heat loss method) to
    determine actual heat output and efficiency. All calculations are deterministic.

    Heat Balance Equation:
        Q_input = Q_output + Q_losses
        Q_input = ṁ_fuel * LHV

        Q_losses = Q_stack + Q_radiation + Q_convection + Q_incomplete + Q_moisture

    Efficiency:
        η_thermal = Q_output / Q_input = 1 - (Q_losses / Q_input)
    """

    # Physical constants
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
    SPECIFIC_HEAT_AIR = 1.005   # kJ/(kg·K)
    SPECIFIC_HEAT_FLUE_GAS = 1.08  # kJ/(kg·K) approximate
    LATENT_HEAT_WATER = 2442  # kJ/kg at 25°C

    # Molecular weights
    MW_H2O = 18.015  # kg/kmol
    MW_H2 = 2.016    # kg/kmol

    def __init__(self):
        """Initialize heat output calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_heat_output(
        self,
        heat_input: HeatOutputInput
    ) -> HeatOutputResult:
        """
        Calculate actual heat output using heat balance method.

        Algorithm:
            1. Calculate gross and net heat input
            2. Calculate stack loss (largest loss component)
            3. Calculate radiation and convection losses
            4. Calculate moisture loss (from H2 combustion and fuel moisture)
            5. Calculate incomplete combustion loss (from CO)
            6. Sum all losses
            7. Calculate net heat output = input - losses
            8. Calculate thermal efficiency
            9. Validate against target (if provided)

        Args:
            heat_input: Heat output calculation inputs

        Returns:
            HeatOutputResult with complete heat balance
        """
        self.logger.info("Calculating heat output and efficiency")

        # Step 1: Calculate heat input
        net_heat_input_kw = self._calculate_net_heat_input(
            heat_input.fuel_flow_rate_kg_per_hr,
            heat_input.fuel_lower_heating_value_mj_per_kg
        )

        # Calculate HHV if not provided (approximate)
        if heat_input.fuel_higher_heating_value_mj_per_kg:
            hhv = heat_input.fuel_higher_heating_value_mj_per_kg
        else:
            # HHV ≈ LHV + 2.442 * 9 * H2_fraction
            h2_fraction = heat_input.fuel_hydrogen_percent / 100
            hhv = heat_input.fuel_lower_heating_value_mj_per_kg + 2.442 * 9 * h2_fraction

        gross_heat_input_kw = self._calculate_net_heat_input(
            heat_input.fuel_flow_rate_kg_per_hr,
            hhv
        )

        # Step 2: Calculate excess air
        excess_air_percent = self._calculate_excess_air(
            heat_input.flue_gas_o2_percent
        )

        # Step 3: Calculate flue gas flow rate
        flue_gas_flow = self._calculate_flue_gas_flow(
            heat_input.fuel_flow_rate_kg_per_hr,
            heat_input.air_flow_rate_kg_per_hr
        )

        # Step 4: Calculate stack loss (dry flue gas)
        stack_loss_kw, stack_loss_percent = self._calculate_stack_loss(
            flue_gas_flow,
            heat_input.flue_gas_temperature_c,
            heat_input.ambient_temperature_c,
            net_heat_input_kw
        )

        # Step 5: Calculate moisture loss
        moisture_loss_kw, moisture_loss_percent = self._calculate_moisture_loss(
            heat_input.fuel_flow_rate_kg_per_hr,
            heat_input.fuel_hydrogen_percent,
            heat_input.fuel_moisture_percent,
            net_heat_input_kw
        )

        # Step 6: Calculate incomplete combustion loss
        incomplete_loss_kw, incomplete_loss_percent = self._calculate_incomplete_combustion_loss(
            heat_input.flue_gas_co_ppm,
            flue_gas_flow,
            net_heat_input_kw
        )

        # Step 7: Calculate radiation and convection loss
        if heat_input.combustor_surface_area_m2 and heat_input.surface_temperature_c:
            radiation_loss_kw, radiation_loss_percent = self._calculate_radiation_loss(
                heat_input.combustor_surface_area_m2,
                heat_input.surface_temperature_c,
                heat_input.ambient_temperature_c,
                heat_input.surface_emissivity,
                net_heat_input_kw
            )
        else:
            # Use typical radiation loss (1-2% for industrial equipment)
            radiation_loss_percent = 1.5
            radiation_loss_kw = net_heat_input_kw * radiation_loss_percent / 100

        # Step 8: Sum all losses
        heat_losses = [
            HeatLossComponent(
                category=HeatLossCategory.STACK_LOSS,
                loss_kw=stack_loss_kw,
                loss_percent=stack_loss_percent,
                description="Heat loss through hot flue gases"
            ),
            HeatLossComponent(
                category=HeatLossCategory.MOISTURE_LOSS,
                loss_kw=moisture_loss_kw,
                loss_percent=moisture_loss_percent,
                description="Latent heat of water vapor in flue gas"
            ),
            HeatLossComponent(
                category=HeatLossCategory.INCOMPLETE_COMBUSTION,
                loss_kw=incomplete_loss_kw,
                loss_percent=incomplete_loss_percent,
                description="Unburned CO in flue gas"
            ),
            HeatLossComponent(
                category=HeatLossCategory.RADIATION_LOSS,
                loss_kw=radiation_loss_kw,
                loss_percent=radiation_loss_percent,
                description="Radiation and convection from external surfaces"
            )
        ]

        total_loss_kw = sum(loss.loss_kw for loss in heat_losses)
        total_loss_percent = sum(loss.loss_percent for loss in heat_losses)

        # Step 9: Calculate net heat output
        net_heat_output_kw = net_heat_input_kw - total_loss_kw

        # Step 10: Calculate efficiencies
        net_efficiency = (net_heat_output_kw / net_heat_input_kw * 100) if net_heat_input_kw > 0 else 0
        gross_efficiency = (net_heat_output_kw / gross_heat_input_kw * 100) if gross_heat_input_kw > 0 else 0

        # Combustion efficiency (excludes only incomplete combustion loss)
        combustion_efficiency = 100 - incomplete_loss_percent

        # Step 11: Calculate specific fuel consumption
        specific_fuel_consumption = (
            heat_input.fuel_flow_rate_kg_per_hr / net_heat_output_kw
            if net_heat_output_kw > 0 else 0
        )

        # Step 12: Validate against target
        target_deviation = None
        meets_target = None
        if heat_input.target_heat_output_kw:
            target_deviation = (
                (net_heat_output_kw - heat_input.target_heat_output_kw) /
                heat_input.target_heat_output_kw * 100
            )
            meets_target = abs(target_deviation) <= 5.0  # Within ±5%

        # Step 13: Generate recommendations
        recommendations = self._generate_heat_output_recommendations(
            net_efficiency,
            stack_loss_percent,
            excess_air_percent,
            heat_input.flue_gas_temperature_c,
            incomplete_loss_percent
        )

        return HeatOutputResult(
            gross_heat_input_kw=self._round_decimal(gross_heat_input_kw, 2),
            net_heat_input_kw=self._round_decimal(net_heat_input_kw, 2),
            net_heat_output_kw=self._round_decimal(net_heat_output_kw, 2),
            heat_release_rate_kw_per_m3=None,  # Requires combustor volume
            gross_thermal_efficiency_percent=self._round_decimal(gross_efficiency, 2),
            net_thermal_efficiency_percent=self._round_decimal(net_efficiency, 2),
            combustion_efficiency_percent=self._round_decimal(combustion_efficiency, 2),
            total_heat_loss_kw=self._round_decimal(total_loss_kw, 2),
            total_heat_loss_percent=self._round_decimal(total_loss_percent, 2),
            heat_losses=[
                {
                    'category': loss.category.value,
                    'loss_kw': self._round_decimal(loss.loss_kw, 2),
                    'loss_percent': self._round_decimal(loss.loss_percent, 2),
                    'description': loss.description
                }
                for loss in heat_losses
            ],
            stack_loss_kw=self._round_decimal(stack_loss_kw, 2),
            stack_loss_percent=self._round_decimal(stack_loss_percent, 2),
            stack_gas_flow_rate_kg_per_hr=self._round_decimal(flue_gas_flow, 2),
            excess_air_percent=self._round_decimal(excess_air_percent, 2),
            specific_fuel_consumption_kg_per_kwh=self._round_decimal(specific_fuel_consumption, 4),
            target_deviation_percent=self._round_decimal(target_deviation, 2) if target_deviation else None,
            meets_target=meets_target,
            recommendations=recommendations
        )

    def _calculate_net_heat_input(
        self,
        fuel_flow_kg_per_hr: float,
        heating_value_mj_per_kg: float
    ) -> float:
        """
        Calculate net heat input.

        Formula:
            Q̇_input = ṁ_fuel * LHV
            Convert: MJ/hr to kW = MJ/hr * 1000 / 3600 = MJ/hr / 3.6
        """
        heat_input_mj_per_hr = fuel_flow_kg_per_hr * heating_value_mj_per_kg
        heat_input_kw = heat_input_mj_per_hr / 3.6  # Convert MJ/hr to kW

        return heat_input_kw

    def calculate_heat_release_rate(
        self,
        fuel_flow_kg_per_hr: float,
        heating_value_mj_per_kg: float,
        combustor_volume_m3: float,
        combustion_efficiency: float = 0.99
    ) -> float:
        """
        Calculate volumetric heat release rate.

        Formula:
            HRR = (ṁ_fuel * LHV * η_comb) / V_combustor

        Args:
            fuel_flow_kg_per_hr: Fuel flow rate in kg/hr
            heating_value_mj_per_kg: Fuel heating value in MJ/kg
            combustor_volume_m3: Combustor volume in m³
            combustion_efficiency: Combustion efficiency (0-1)

        Returns:
            Heat release rate in kW/m³
        """
        heat_input_kw = self._calculate_net_heat_input(fuel_flow_kg_per_hr, heating_value_mj_per_kg)
        actual_heat_release = heat_input_kw * combustion_efficiency

        if combustor_volume_m3 <= 0:
            raise ValueError("Combustor volume must be positive")

        hrr = actual_heat_release / combustor_volume_m3

        return hrr

    def calculate_thermal_efficiency(
        self,
        heat_output_kw: float,
        fuel_flow_kg_per_hr: float,
        heating_value_mj_per_kg: float
    ) -> float:
        """
        Calculate thermal efficiency.

        Formula:
            η_thermal = Q_output / Q_input

        Args:
            heat_output_kw: Useful heat output in kW
            fuel_flow_kg_per_hr: Fuel flow rate in kg/hr
            heating_value_mj_per_kg: Fuel heating value in MJ/kg

        Returns:
            Thermal efficiency as percentage
        """
        heat_input_kw = self._calculate_net_heat_input(fuel_flow_kg_per_hr, heating_value_mj_per_kg)

        if heat_input_kw <= 0:
            return 0.0

        efficiency = (heat_output_kw / heat_input_kw) * 100

        return min(efficiency, 100.0)  # Cap at 100%

    def validate_against_target(
        self,
        actual_output_kw: float,
        target_output_kw: float,
        tolerance_percent: float = 5.0
    ) -> Dict[str, any]:
        """
        Validate actual heat output against target.

        Args:
            actual_output_kw: Actual heat output in kW
            target_output_kw: Target heat output in kW
            tolerance_percent: Acceptable tolerance in percent

        Returns:
            Dictionary with validation results
        """
        deviation_kw = actual_output_kw - target_output_kw
        deviation_percent = (deviation_kw / target_output_kw * 100) if target_output_kw > 0 else 0

        within_tolerance = abs(deviation_percent) <= tolerance_percent

        return {
            'actual_output_kw': actual_output_kw,
            'target_output_kw': target_output_kw,
            'deviation_kw': deviation_kw,
            'deviation_percent': deviation_percent,
            'tolerance_percent': tolerance_percent,
            'within_tolerance': within_tolerance,
            'status': 'OK' if within_tolerance else 'OUT_OF_TOLERANCE'
        }

    def _calculate_excess_air(self, o2_percent: float) -> float:
        """
        Calculate excess air from flue gas O2.

        Formula:
            EA% = (O2 / (21 - O2)) * 100

        Reference: ASHRAE Fundamentals
        """
        if o2_percent >= 21:
            return 100.0  # Maximum practical value

        excess_air = (o2_percent / (21 - o2_percent)) * 100

        return excess_air

    def _calculate_flue_gas_flow(
        self,
        fuel_flow: float,
        air_flow: float
    ) -> float:
        """
        Calculate flue gas flow rate.

        Simplified: Flue gas = Fuel + Air (neglecting ash removal)
        """
        return fuel_flow + air_flow

    def _calculate_stack_loss(
        self,
        flue_gas_flow: float,
        flue_gas_temp: float,
        ambient_temp: float,
        heat_input_kw: float
    ) -> Tuple[float, float]:
        """
        Calculate stack loss (sensible heat in flue gas).

        Formula:
            Q_stack = ṁ_flue * Cp * (T_flue - T_ambient)

        Args:
            flue_gas_flow: Flue gas flow rate in kg/hr
            flue_gas_temp: Flue gas temperature in °C
            ambient_temp: Ambient temperature in °C
            heat_input_kw: Heat input in kW

        Returns:
            Tuple of (loss_kw, loss_percent)
        """
        temp_diff = flue_gas_temp - ambient_temp

        # Convert flow rate from kg/hr to kg/s
        flue_gas_flow_kg_per_s = flue_gas_flow / 3600

        # Stack loss (kW) = ṁ * Cp * ΔT
        stack_loss_kw = flue_gas_flow_kg_per_s * self.SPECIFIC_HEAT_FLUE_GAS * temp_diff

        # Loss percentage
        stack_loss_percent = (stack_loss_kw / heat_input_kw * 100) if heat_input_kw > 0 else 0

        return stack_loss_kw, stack_loss_percent

    def _calculate_moisture_loss(
        self,
        fuel_flow: float,
        hydrogen_percent: float,
        moisture_percent: float,
        heat_input_kw: float
    ) -> Tuple[float, float]:
        """
        Calculate moisture loss (latent heat of water vapor).

        Sources of moisture:
        1. H2 combustion: H2 + 0.5*O2 -> H2O (9 kg H2O per kg H2)
        2. Moisture in fuel

        Formula:
            Q_moisture = (m_H2O_from_H2 + m_H2O_fuel) * h_fg

        Args:
            fuel_flow: Fuel flow rate in kg/hr
            hydrogen_percent: Hydrogen content in percent
            moisture_percent: Moisture content in percent
            heat_input_kw: Heat input in kW

        Returns:
            Tuple of (loss_kw, loss_percent)
        """
        # H2O from hydrogen combustion (9 kg H2O per 1 kg H2)
        h2_flow = fuel_flow * (hydrogen_percent / 100)
        h2o_from_h2 = h2_flow * 9

        # Moisture in fuel
        h2o_from_fuel = fuel_flow * (moisture_percent / 100)

        # Total water vapor
        total_h2o = h2o_from_h2 + h2o_from_fuel

        # Latent heat loss (kW)
        # Convert: kg/hr * kJ/kg = kJ/hr, then /3600 = kJ/s = kW
        moisture_loss_kw = (total_h2o * self.LATENT_HEAT_WATER) / 3600

        # Loss percentage
        moisture_loss_percent = (moisture_loss_kw / heat_input_kw * 100) if heat_input_kw > 0 else 0

        return moisture_loss_kw, moisture_loss_percent

    def _calculate_incomplete_combustion_loss(
        self,
        co_ppm: float,
        flue_gas_flow: float,
        heat_input_kw: float
    ) -> Tuple[float, float]:
        """
        Calculate heat loss from incomplete combustion (CO formation).

        CO represents unburned carbon that should have formed CO2.
        Heat loss = mass_CO * (HHV_CO to CO2) ≈ 10.1 MJ/kg CO

        Args:
            co_ppm: CO concentration in ppm
            flue_gas_flow: Flue gas flow rate in kg/hr
            heat_input_kw: Heat input in kW

        Returns:
            Tuple of (loss_kw, loss_percent)
        """
        if co_ppm == 0:
            return 0.0, 0.0

        # Convert ppm to mass fraction (assuming ideal gas at STP)
        # CO ppm ≈ CO mg/Nm³ for combustion calculations
        co_mg_per_nm3 = co_ppm

        # Flue gas volume (Nm³/hr) - rough approximation
        # Density of flue gas ≈ 1.3 kg/Nm³
        flue_gas_volume_nm3_per_hr = flue_gas_flow / 1.3

        # CO mass flow (kg/hr)
        co_mass_flow = (co_mg_per_nm3 * flue_gas_volume_nm3_per_hr) / 1e6

        # Heat loss from CO (10.1 MJ/kg CO)
        co_heating_value = 10.1  # MJ/kg
        co_loss_mj_per_hr = co_mass_flow * co_heating_value

        # Convert to kW
        co_loss_kw = co_loss_mj_per_hr / 3.6

        # Loss percentage
        co_loss_percent = (co_loss_kw / heat_input_kw * 100) if heat_input_kw > 0 else 0

        return co_loss_kw, co_loss_percent

    def _calculate_radiation_loss(
        self,
        surface_area: float,
        surface_temp: float,
        ambient_temp: float,
        emissivity: float,
        heat_input_kw: float
    ) -> Tuple[float, float]:
        """
        Calculate radiation and convection heat loss.

        Formula (Stefan-Boltzmann):
            Q_rad = ε * σ * A * (T_surface⁴ - T_ambient⁴)

        Args:
            surface_area: External surface area in m²
            surface_temp: Surface temperature in °C
            ambient_temp: Ambient temperature in °C
            emissivity: Surface emissivity (0-1)
            heat_input_kw: Heat input in kW

        Returns:
            Tuple of (loss_kw, loss_percent)
        """
        # Convert temperatures to Kelvin
        t_surface_k = surface_temp + 273.15
        t_ambient_k = ambient_temp + 273.15

        # Radiation loss (W)
        radiation_loss_w = (
            emissivity *
            self.STEFAN_BOLTZMANN *
            surface_area *
            (t_surface_k**4 - t_ambient_k**4)
        )

        # Convection loss (simplified: h = 10 W/(m²·K) for natural convection)
        convection_coefficient = 10.0  # W/(m²·K)
        convection_loss_w = convection_coefficient * surface_area * (surface_temp - ambient_temp)

        # Total loss (kW)
        total_loss_kw = (radiation_loss_w + convection_loss_w) / 1000

        # Loss percentage
        loss_percent = (total_loss_kw / heat_input_kw * 100) if heat_input_kw > 0 else 0

        return total_loss_kw, loss_percent

    def _generate_heat_output_recommendations(
        self,
        efficiency: float,
        stack_loss_percent: float,
        excess_air_percent: float,
        flue_gas_temp: float,
        incomplete_loss_percent: float
    ) -> List[str]:
        """Generate recommendations based on heat output analysis"""
        recommendations = []

        if efficiency < 70:
            recommendations.append("CRITICAL: Thermal efficiency below 70% - immediate action required")

        if efficiency < 80:
            recommendations.append("Low efficiency detected - investigate heat losses")

        if stack_loss_percent > 15:
            recommendations.append(f"High stack loss ({stack_loss_percent:.1f}%) - reduce flue gas temperature")

        if flue_gas_temp > 300:
            recommendations.append(f"Flue gas temperature high ({flue_gas_temp:.0f}°C) - consider heat recovery")

        if excess_air_percent > 30:
            recommendations.append(f"Excess air too high ({excess_air_percent:.1f}%) - reduce air flow for better efficiency")

        if excess_air_percent < 10:
            recommendations.append(f"Excess air too low ({excess_air_percent:.1f}%) - risk of incomplete combustion")

        if incomplete_loss_percent > 1:
            recommendations.append(f"Incomplete combustion detected ({incomplete_loss_percent:.2f}%) - adjust air-fuel ratio")

        if 80 <= efficiency <= 90 and 10 <= excess_air_percent <= 20:
            recommendations.append("System operating efficiently within target range")

        return recommendations

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return None
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
