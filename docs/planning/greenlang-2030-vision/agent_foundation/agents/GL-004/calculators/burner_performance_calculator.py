# -*- coding: utf-8 -*-
"""Burner Performance Calculator for GL-004 BURNMASTER.

Implements deterministic calculation of burner performance metrics
using physics-based combustion engineering principles.

This module provides zero-hallucination performance calculations for
industrial burners including thermal efficiency, combustion efficiency,
turndown ratio, and flame stability metrics.

Key Features:
- Physics-based thermal efficiency calculation
- Stack loss calculation (sensible heat, latent heat)
- Combustion efficiency from O2/CO measurements
- Turndown ratio and stability analysis
- Heat output calculation from fuel heating values
- Complete provenance tracking with SHA-256 hashes

Author: GreenLang AI Agent Factory
License: Proprietary
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import hashlib
import json
import math


@dataclass
class PerformanceMetrics:
    """Comprehensive burner performance metrics.

    All metrics are deterministically calculated from
    sensor inputs with full provenance tracking.
    """
    thermal_efficiency: float  # Overall thermal efficiency (0-1)
    combustion_efficiency: float  # Combustion efficiency (0-1)
    burner_load_percent: float  # Current load percentage (0-100)
    heat_output_mmbtu_hr: float  # Heat output in MMBtu/hr
    turndown_ratio: float  # Turndown ratio (max/current)
    flame_stability_index: float  # Flame stability (0-1)
    provenance_hash: str  # SHA-256 hash for verification

    # Extended metrics
    stack_loss_percent: float = 0.0
    radiation_loss_percent: float = 0.0
    unburned_loss_percent: float = 0.0
    heat_input_mmbtu_hr: float = 0.0
    excess_air_percent: float = 0.0
    air_fuel_ratio: float = 0.0


# Heating values for common fuels
# Higher Heating Value (HHV) in BTU per unit
FUEL_HEATING_VALUES: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "hhv_btu_scf": 1020.0,  # BTU per standard cubic foot
        "hhv_btu_kg": 23875.0,  # BTU per kg
        "lhv_btu_scf": 920.0,
        "lhv_btu_kg": 21500.0,
        "density_lb_scf": 0.0458,
    },
    "propane": {
        "hhv_btu_gal": 91500.0,  # BTU per gallon
        "hhv_btu_kg": 21661.0,
        "lhv_btu_gal": 84250.0,
        "lhv_btu_kg": 19944.0,
        "density_lb_gal": 4.2,
    },
    "fuel_oil_2": {
        "hhv_btu_gal": 140000.0,  # BTU per gallon
        "hhv_btu_kg": 19560.0,
        "lhv_btu_gal": 131500.0,
        "lhv_btu_kg": 18370.0,
        "density_lb_gal": 7.15,
    },
    "fuel_oil_6": {
        "hhv_btu_gal": 152000.0,
        "hhv_btu_kg": 18270.0,
        "lhv_btu_gal": 143500.0,
        "lhv_btu_kg": 17250.0,
        "density_lb_gal": 8.3,
    },
    "coal": {
        "hhv_btu_lb": 12500.0,  # BTU per pound (bituminous)
        "hhv_btu_kg": 27550.0,
        "lhv_btu_lb": 11800.0,
        "lhv_btu_kg": 26000.0,
    },
}


class BurnerPerformanceCalculator:
    """Calculate comprehensive burner performance metrics.

    Implements deterministic performance calculations based on
    established combustion engineering principles. All calculations
    are reproducible and include full provenance tracking.

    Key calculations:
    1. Thermal efficiency using indirect method
    2. Stack losses (dry gas, moisture, radiation)
    3. Combustion efficiency from O2/CO2 analysis
    4. Heat output from fuel flow and heating values
    5. Flame stability analysis

    Example:
        calculator = BurnerPerformanceCalculator()
        metrics = calculator.calculate(
            fuel_flow=100.0,
            burner_load=80.0,
            max_capacity=125.0,
            fuel_type="natural_gas"
        )
    """

    # Stack loss coefficients
    STACK_LOSS_COEFFICIENTS: Dict[str, float] = {
        "natural_gas": 0.35,  # Coefficient for sensible heat loss
        "propane": 0.38,
        "fuel_oil_2": 0.45,
        "fuel_oil_6": 0.47,
        "coal": 0.50,
    }

    # Radiation loss by burner size (MMBtu/hr capacity)
    RADIATION_LOSS_FACTORS: Dict[str, float] = {
        "small": 0.03,    # < 10 MMBtu/hr
        "medium": 0.02,   # 10-50 MMBtu/hr
        "large": 0.015,   # 50-100 MMBtu/hr
        "xlarge": 0.01,   # > 100 MMBtu/hr
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the burner performance calculator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._calculation_steps: List[Dict[str, Any]] = []

    def calculate(
        self,
        fuel_flow: float,
        burner_load: float,
        max_capacity: float,
        fuel_type: str = "natural_gas",
        inlet_air_temp_f: float = 70.0,
        stack_temp_f: float = 350.0,
        o2_percent: float = 3.0,
        co_ppm: float = 50.0
    ) -> PerformanceMetrics:
        """Calculate comprehensive burner performance metrics.

        Args:
            fuel_flow: Fuel flow rate (scf/hr for gas, gal/hr for oil, lb/hr for coal)
            burner_load: Current burner load (percentage or absolute)
            max_capacity: Maximum burner capacity (MMBtu/hr)
            fuel_type: Type of fuel being burned
            inlet_air_temp_f: Combustion air inlet temperature (Fahrenheit)
            stack_temp_f: Stack/flue gas temperature (Fahrenheit)
            o2_percent: O2 percentage in flue gas (dry basis)
            co_ppm: CO concentration in flue gas (ppm)

        Returns:
            PerformanceMetrics with complete performance data
        """
        self._calculation_steps = []

        # Get heating values for fuel
        heating_values = FUEL_HEATING_VALUES.get(fuel_type)
        if not heating_values:
            heating_values = FUEL_HEATING_VALUES["natural_gas"]

        # Calculate heat input
        if fuel_type == "natural_gas":
            hhv = heating_values["hhv_btu_scf"]
            heat_input_btu_hr = fuel_flow * hhv
        elif fuel_type in ["propane", "fuel_oil_2", "fuel_oil_6"]:
            hhv = heating_values["hhv_btu_gal"]
            heat_input_btu_hr = fuel_flow * hhv
        else:
            hhv = heating_values.get("hhv_btu_lb", 12000)
            heat_input_btu_hr = fuel_flow * hhv

        heat_input_mmbtu_hr = heat_input_btu_hr / 1_000_000
        self._log_step("calculate_heat_input", {
            "fuel_flow": fuel_flow,
            "hhv": hhv,
            "fuel_type": fuel_type
        }, heat_input_mmbtu_hr)

        # Calculate load percentage
        if burner_load > 1.0:
            # Assume burner_load is already a percentage
            load_percent = burner_load
        else:
            # Calculate from heat input vs capacity
            load_percent = (heat_input_mmbtu_hr / max_capacity) * 100 if max_capacity > 0 else 0

        load_percent = min(100.0, max(0.0, load_percent))
        self._log_step("calculate_load_percent", {
            "burner_load": burner_load,
            "max_capacity": max_capacity,
            "heat_input": heat_input_mmbtu_hr
        }, load_percent)

        # Calculate excess air from O2 measurement
        excess_air_percent = self._calculate_excess_air(o2_percent)
        self._log_step("calculate_excess_air", {"o2_percent": o2_percent}, excess_air_percent)

        # Calculate stack loss (sensible heat)
        stack_loss = self._calculate_stack_loss(
            stack_temp_f, inlet_air_temp_f, excess_air_percent, fuel_type
        )
        self._log_step("calculate_stack_loss", {
            "stack_temp": stack_temp_f,
            "inlet_temp": inlet_air_temp_f,
            "excess_air": excess_air_percent
        }, stack_loss)

        # Calculate radiation loss
        radiation_loss = self._calculate_radiation_loss(max_capacity, load_percent)
        self._log_step("calculate_radiation_loss", {
            "capacity": max_capacity,
            "load": load_percent
        }, radiation_loss)

        # Calculate unburned combustibles loss
        unburned_loss = self._calculate_unburned_loss(co_ppm, excess_air_percent)
        self._log_step("calculate_unburned_loss", {
            "co_ppm": co_ppm,
            "excess_air": excess_air_percent
        }, unburned_loss)

        # Calculate combustion efficiency (from O2 and CO)
        combustion_eff = self._calculate_combustion_efficiency(
            o2_percent, co_ppm, fuel_type
        )
        self._log_step("calculate_combustion_efficiency", {
            "o2_percent": o2_percent,
            "co_ppm": co_ppm
        }, combustion_eff)

        # Calculate thermal efficiency
        thermal_eff = 1.0 - stack_loss - radiation_loss - unburned_loss
        thermal_eff = max(0.50, min(0.95, thermal_eff))
        self._log_step("calculate_thermal_efficiency", {
            "stack_loss": stack_loss,
            "radiation_loss": radiation_loss,
            "unburned_loss": unburned_loss
        }, thermal_eff)

        # Calculate heat output
        heat_output_mmbtu_hr = heat_input_mmbtu_hr * thermal_eff
        self._log_step("calculate_heat_output", {
            "heat_input": heat_input_mmbtu_hr,
            "thermal_eff": thermal_eff
        }, heat_output_mmbtu_hr)

        # Calculate turndown ratio
        current_output = heat_output_mmbtu_hr
        turndown = max_capacity / max(current_output, 0.1)
        turndown = max(1.0, min(20.0, turndown))
        self._log_step("calculate_turndown", {
            "max_capacity": max_capacity,
            "current_output": current_output
        }, turndown)

        # Calculate flame stability index
        stability = self._calculate_flame_stability(
            load_percent, excess_air_percent, co_ppm
        )
        self._log_step("calculate_flame_stability", {
            "load": load_percent,
            "excess_air": excess_air_percent,
            "co_ppm": co_ppm
        }, stability)

        # Calculate air-fuel ratio
        stoich_afr = {"natural_gas": 17.2, "propane": 15.7, "fuel_oil_2": 14.7}.get(fuel_type, 17.0)
        actual_afr = stoich_afr * (1 + excess_air_percent / 100.0)

        # Generate provenance hash
        result_data = {
            "fuel_flow": fuel_flow,
            "burner_load": burner_load,
            "thermal_eff": thermal_eff,
            "heat_output": heat_output_mmbtu_hr
        }
        provenance = self._calculate_provenance_hash(result_data)

        return PerformanceMetrics(
            thermal_efficiency=round(thermal_eff, 4),
            combustion_efficiency=round(combustion_eff, 4),
            burner_load_percent=round(load_percent, 1),
            heat_output_mmbtu_hr=round(heat_output_mmbtu_hr, 2),
            turndown_ratio=round(turndown, 1),
            flame_stability_index=round(stability, 2),
            provenance_hash=provenance,
            stack_loss_percent=round(stack_loss * 100, 2),
            radiation_loss_percent=round(radiation_loss * 100, 2),
            unburned_loss_percent=round(unburned_loss * 100, 2),
            heat_input_mmbtu_hr=round(heat_input_mmbtu_hr, 2),
            excess_air_percent=round(excess_air_percent, 1),
            air_fuel_ratio=round(actual_afr, 2)
        )

    def _calculate_excess_air(self, o2_percent: float) -> float:
        """Calculate excess air from O2 measurement.

        Uses standard combustion formula:
        EA% = (O2 / (21 - O2)) * 100

        Args:
            o2_percent: O2 percentage in dry flue gas

        Returns:
            Excess air percentage
        """
        if o2_percent >= 20.9:
            return 1000.0  # Essentially all air (no combustion)
        if o2_percent <= 0.0:
            return 0.0

        excess_air = (o2_percent / (21.0 - o2_percent)) * 100.0
        return excess_air

    def _calculate_stack_loss(
        self,
        stack_temp_f: float,
        inlet_temp_f: float,
        excess_air_percent: float,
        fuel_type: str
    ) -> float:
        """Calculate stack loss (sensible heat loss in flue gas).

        Uses Siegert formula approximation:
        Stack Loss = K * (T_stack - T_ambient) / (CO2_percent)

        Simplified version based on excess air:
        Stack Loss = A * (T_stack - T_inlet) * (1 + EA/100) / 1000

        Args:
            stack_temp_f: Stack temperature (Fahrenheit)
            inlet_temp_f: Combustion air inlet temperature (Fahrenheit)
            excess_air_percent: Excess air percentage
            fuel_type: Type of fuel

        Returns:
            Stack loss as fraction (0-1)
        """
        # Temperature difference
        delta_t = stack_temp_f - inlet_temp_f

        # Coefficient based on fuel type
        coeff = self.STACK_LOSS_COEFFICIENTS.get(fuel_type, 0.38)

        # Stack loss increases with excess air
        air_factor = 1.0 + (excess_air_percent / 100.0) * 0.5

        # Calculate stack loss
        # Approximately 1% loss per 40F temperature difference at stoichiometric
        stack_loss = (delta_t / 40.0) * 0.01 * coeff * air_factor

        # Clamp to realistic range
        stack_loss = max(0.02, min(0.30, stack_loss))

        return stack_loss

    def _calculate_radiation_loss(
        self,
        capacity_mmbtu_hr: float,
        load_percent: float
    ) -> float:
        """Calculate radiation and convection losses.

        Radiation losses are relatively fixed and become a larger
        percentage of output at lower loads.

        Args:
            capacity_mmbtu_hr: Burner capacity (MMBtu/hr)
            load_percent: Current load percentage

        Returns:
            Radiation loss as fraction (0-1)
        """
        # Determine burner size category
        if capacity_mmbtu_hr < 10:
            base_loss = self.RADIATION_LOSS_FACTORS["small"]
        elif capacity_mmbtu_hr < 50:
            base_loss = self.RADIATION_LOSS_FACTORS["medium"]
        elif capacity_mmbtu_hr < 100:
            base_loss = self.RADIATION_LOSS_FACTORS["large"]
        else:
            base_loss = self.RADIATION_LOSS_FACTORS["xlarge"]

        # Radiation loss as % increases at lower loads
        # (fixed absolute loss / lower output = higher %)
        if load_percent > 0:
            load_factor = 100.0 / max(load_percent, 10.0)
        else:
            load_factor = 10.0

        radiation_loss = base_loss * (1 + (load_factor - 1) * 0.5)

        # Clamp to realistic range
        radiation_loss = max(0.005, min(0.10, radiation_loss))

        return radiation_loss

    def _calculate_unburned_loss(
        self,
        co_ppm: float,
        excess_air_percent: float
    ) -> float:
        """Calculate loss from unburned combustibles.

        Based on CO levels as indicator of incomplete combustion.

        Args:
            co_ppm: CO concentration in flue gas (ppm)
            excess_air_percent: Excess air percentage

        Returns:
            Unburned combustibles loss as fraction (0-1)
        """
        # CO indicates incomplete combustion
        # Each 100 ppm CO represents approximately 0.05% loss

        if excess_air_percent < 0:
            # Rich mixture - significant losses
            co_factor = 2.0
        else:
            co_factor = 1.0

        unburned_loss = (co_ppm / 100.0) * 0.0005 * co_factor

        # Clamp to realistic range
        unburned_loss = max(0.001, min(0.05, unburned_loss))

        return unburned_loss

    def _calculate_combustion_efficiency(
        self,
        o2_percent: float,
        co_ppm: float,
        fuel_type: str
    ) -> float:
        """Calculate combustion efficiency from flue gas analysis.

        Uses O2 and CO measurements to determine completeness
        of combustion reaction.

        Args:
            o2_percent: O2 percentage in dry flue gas
            co_ppm: CO concentration in ppm
            fuel_type: Type of fuel

        Returns:
            Combustion efficiency as fraction (0-1)
        """
        # Base efficiency assuming complete combustion
        base_eff = 1.0

        # Penalty for high excess air (dilution)
        if o2_percent > 5.0:
            excess_penalty = (o2_percent - 5.0) * 0.005
        else:
            excess_penalty = 0.0

        # Penalty for CO (incomplete combustion)
        if co_ppm > 100:
            co_penalty = (co_ppm - 100) * 0.0001
        elif co_ppm > 50:
            co_penalty = (co_ppm - 50) * 0.00005
        else:
            co_penalty = 0.0

        # Penalty for low O2 (rich combustion)
        if o2_percent < 1.0:
            low_o2_penalty = (1.0 - o2_percent) * 0.02
        else:
            low_o2_penalty = 0.0

        combustion_eff = base_eff - excess_penalty - co_penalty - low_o2_penalty

        # Clamp to realistic range
        combustion_eff = max(0.85, min(1.0, combustion_eff))

        return combustion_eff

    def _calculate_flame_stability(
        self,
        load_percent: float,
        excess_air_percent: float,
        co_ppm: float
    ) -> float:
        """Calculate flame stability index.

        Combines multiple factors affecting flame stability:
        - Load level (low loads = less stable)
        - Excess air (too much or too little = less stable)
        - CO levels (high CO = unstable combustion)

        Args:
            load_percent: Current burner load percentage
            excess_air_percent: Excess air percentage
            co_ppm: CO concentration in ppm

        Returns:
            Stability index (0-1, where 1 = perfectly stable)
        """
        # Load factor - stability decreases below 30% load
        if load_percent < 20:
            load_stability = load_percent / 20.0 * 0.5
        elif load_percent < 30:
            load_stability = 0.5 + (load_percent - 20) / 10.0 * 0.3
        else:
            load_stability = 0.8 + min((load_percent - 30) / 70.0 * 0.2, 0.2)

        # Excess air factor - optimal is 10-20%
        if 10 <= excess_air_percent <= 20:
            air_stability = 1.0
        elif 5 <= excess_air_percent < 10:
            air_stability = 0.8 + (excess_air_percent - 5) / 5.0 * 0.2
        elif 20 < excess_air_percent <= 30:
            air_stability = 1.0 - (excess_air_percent - 20) / 10.0 * 0.2
        else:
            # Outside optimal range
            if excess_air_percent < 5:
                air_stability = 0.5 + excess_air_percent / 5.0 * 0.3
            else:  # > 30%
                air_stability = max(0.3, 0.8 - (excess_air_percent - 30) / 50.0 * 0.5)

        # CO factor - high CO indicates unstable combustion
        if co_ppm < 50:
            co_stability = 1.0
        elif co_ppm < 100:
            co_stability = 1.0 - (co_ppm - 50) / 50.0 * 0.2
        elif co_ppm < 200:
            co_stability = 0.8 - (co_ppm - 100) / 100.0 * 0.3
        else:
            co_stability = max(0.2, 0.5 - (co_ppm - 200) / 500.0 * 0.3)

        # Combined stability (weighted average)
        stability = 0.4 * load_stability + 0.35 * air_stability + 0.25 * co_stability

        return stability

    def _log_step(
        self,
        operation: str,
        inputs: Dict[str, Any],
        output: Any
    ) -> None:
        """Log a calculation step for provenance tracking.

        Args:
            operation: Name of the operation
            inputs: Input values
            output: Output value(s)
        """
        self._calculation_steps.append({
            "step": len(self._calculation_steps) + 1,
            "operation": operation,
            "inputs": inputs,
            "output": output
        })

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash.

        Args:
            data: Data to hash

        Returns:
            First 16 characters of SHA-256 hash
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]

    def get_calculation_steps(self) -> List[Dict[str, Any]]:
        """Get all calculation steps for audit trail.

        Returns:
            List of calculation step dictionaries
        """
        return self._calculation_steps.copy()


# Example usage and testing
if __name__ == "__main__":
    calculator = BurnerPerformanceCalculator()

    print("=" * 60)
    print("Burner Performance Calculator Test")
    print("=" * 60)

    # Test with typical natural gas burner
    metrics = calculator.calculate(
        fuel_flow=1000.0,  # 1000 scf/hr natural gas
        burner_load=80.0,  # 80% load
        max_capacity=1.5,  # 1.5 MMBtu/hr max
        fuel_type="natural_gas",
        inlet_air_temp_f=70.0,
        stack_temp_f=350.0,
        o2_percent=3.5,
        co_ppm=35.0
    )

    print(f"\nInput Parameters:")
    print(f"  Fuel Flow: 1000 scf/hr (natural gas)")
    print(f"  Stack Temp: 350 F")
    print(f"  O2: 3.5%")
    print(f"  CO: 35 ppm")

    print(f"\nPerformance Metrics:")
    print(f"  Thermal Efficiency: {metrics.thermal_efficiency * 100:.1f}%")
    print(f"  Combustion Efficiency: {metrics.combustion_efficiency * 100:.1f}%")
    print(f"  Burner Load: {metrics.burner_load_percent:.1f}%")
    print(f"  Heat Input: {metrics.heat_input_mmbtu_hr:.2f} MMBtu/hr")
    print(f"  Heat Output: {metrics.heat_output_mmbtu_hr:.2f} MMBtu/hr")
    print(f"  Turndown Ratio: {metrics.turndown_ratio:.1f}:1")
    print(f"  Flame Stability: {metrics.flame_stability_index:.2f}")

    print(f"\nLoss Breakdown:")
    print(f"  Stack Loss: {metrics.stack_loss_percent:.2f}%")
    print(f"  Radiation Loss: {metrics.radiation_loss_percent:.2f}%")
    print(f"  Unburned Loss: {metrics.unburned_loss_percent:.2f}%")

    print(f"\nCombustion Parameters:")
    print(f"  Excess Air: {metrics.excess_air_percent:.1f}%")
    print(f"  Air-Fuel Ratio: {metrics.air_fuel_ratio:.1f}")

    print(f"\nProvenance Hash: {metrics.provenance_hash}")

    print(f"\nCalculation Steps ({len(calculator.get_calculation_steps())}):")
    for step in calculator.get_calculation_steps()[:5]:
        print(f"  Step {step['step']}: {step['operation']}")

    print("\n" + "=" * 60)
