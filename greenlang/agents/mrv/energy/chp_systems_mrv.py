# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-007: CHP Systems MRV Agent

Calculates emissions from combined heat and power (cogeneration) systems
with proper allocation between electricity and heat outputs.

Standards Reference:
    - GHG Protocol: Scope 1 Guidance
    - EU CHP Directive: Efficiency methodology
    - EPA CHP Partnership: Emission allocations
    - ISO 50001: Energy management

Example:
    >>> agent = CHPSystemsMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "CHP-PLANT-001",
    ...     "chp_id": "COGEN-1",
    ...     "fuel_type": "natural_gas",
    ...     "fuel_consumption": 50000,
    ...     "fuel_unit": "MMBTU",
    ...     "electricity_output_mwh": 4000,
    ...     "heat_output_mmbtu": 25000,
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    CHPSystemsInput,
    CHPSystemsOutput,
    FuelType,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class CHPSystemsMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-007: CHP Systems MRV Agent

    Calculates:
    - Total CHP emissions
    - Allocation to electricity and heat
    - Efficiency metrics
    - Primary energy savings vs separate production

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-007",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Combined heat and power (CHP) emissions MRV"
    )

    # EU reference efficiencies for separate production
    REFERENCE_EFFICIENCIES = {
        "electricity": {
            "natural_gas": 0.525,
            "coal_bituminous": 0.445,
            "fuel_oil_no2": 0.44,
            "biomass_wood": 0.33,
        },
        "heat": {
            "natural_gas": 0.90,
            "coal_bituminous": 0.88,
            "fuel_oil_no2": 0.89,
            "biomass_wood": 0.86,
        },
    }

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize CHP Systems MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-007",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CHP systems MRV calculation.

        Args:
            inputs: Dictionary with CHPSystemsInput fields

        Returns:
            Dictionary with CHPSystemsOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = CHPSystemsInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        fuel_key = validated_input.fuel_type.value
        calculation_trace.append(
            f"Processing CHP system {validated_input.chp_id}"
        )

        # Convert fuel to MMBTU
        fuel_mmbtu = self._convert_fuel_to_mmbtu(
            validated_input.fuel_consumption,
            validated_input.fuel_unit
        )
        calculation_trace.append(f"Fuel input: {fuel_mmbtu:.2f} MMBTU")

        # Convert outputs to common units
        electricity_mmbtu = validated_input.electricity_output_mwh * 3.412
        heat_mmbtu = validated_input.heat_output_mmbtu

        if validated_input.heat_output_mwh_thermal is not None:
            heat_mmbtu = validated_input.heat_output_mwh_thermal * 3.412

        calculation_trace.append(
            f"Electricity output: {validated_input.electricity_output_mwh:.2f} MWh "
            f"({electricity_mmbtu:.2f} MMBTU)"
        )
        calculation_trace.append(f"Heat output: {heat_mmbtu:.2f} MMBTU")

        # Calculate efficiencies
        if fuel_mmbtu > 0:
            electrical_efficiency = electricity_mmbtu / fuel_mmbtu
            thermal_efficiency = heat_mmbtu / fuel_mmbtu
            overall_efficiency = (electricity_mmbtu + heat_mmbtu) / fuel_mmbtu
        else:
            electrical_efficiency = 0
            thermal_efficiency = 0
            overall_efficiency = 0

        power_to_heat_ratio = (
            electricity_mmbtu / heat_mmbtu
            if heat_mmbtu > 0 else 0
        )

        calculation_trace.append(
            f"Electrical efficiency: {electrical_efficiency:.1%}"
        )
        calculation_trace.append(
            f"Thermal efficiency: {thermal_efficiency:.1%}"
        )
        calculation_trace.append(
            f"Overall efficiency: {overall_efficiency:.1%}"
        )
        calculation_trace.append(
            f"Power-to-heat ratio: {power_to_heat_ratio:.3f}"
        )

        # Calculate total emissions
        co2_factor = self.get_emission_factor("co2_mmbtu", fuel_key)
        total_emissions = fuel_mmbtu * co2_factor / 1000  # tonnes

        calculation_trace.append(
            f"Total emissions: {fuel_mmbtu:.2f} * {co2_factor:.2f} / 1000 = "
            f"{total_emissions:.2f} tonnes"
        )

        # Allocate emissions using energy content method
        total_output = electricity_mmbtu + heat_mmbtu
        if total_output > 0:
            electricity_share = electricity_mmbtu / total_output
            heat_share = heat_mmbtu / total_output
        else:
            electricity_share = 0.5
            heat_share = 0.5

        electricity_emissions = total_emissions * electricity_share
        heat_emissions = total_emissions * heat_share

        calculation_trace.append(
            f"Allocation - Electricity: {electricity_share:.1%}, Heat: {heat_share:.1%}"
        )
        calculation_trace.append(
            f"Electricity emissions: {electricity_emissions:.2f} tonnes"
        )
        calculation_trace.append(
            f"Heat emissions: {heat_emissions:.2f} tonnes"
        )

        # Calculate emission intensities
        if validated_input.electricity_output_mwh > 0:
            electricity_intensity = (
                electricity_emissions * 1000 /
                validated_input.electricity_output_mwh
            )
        else:
            electricity_intensity = 0

        if heat_mmbtu > 0:
            # Convert heat MMBTU to MWh thermal for consistency
            heat_mwh = heat_mmbtu / 3.412
            heat_intensity = heat_emissions * 1000 / heat_mwh
        else:
            heat_intensity = 0

        calculation_trace.append(
            f"Electricity intensity: {electricity_intensity:.2f} kg/MWh"
        )
        calculation_trace.append(
            f"Heat intensity: {heat_intensity:.2f} kg/MWh_th"
        )

        # Calculate Primary Energy Savings (PES) per EU methodology
        ref_elec_eff = self.REFERENCE_EFFICIENCIES["electricity"].get(
            fuel_key, validated_input.reference_electricity_efficiency
        )
        ref_heat_eff = self.REFERENCE_EFFICIENCIES["heat"].get(
            fuel_key, validated_input.reference_heat_efficiency
        )

        # PES = 1 - 1 / (CHP_H/Ref_H + CHP_E/Ref_E)
        if thermal_efficiency > 0 and electrical_efficiency > 0:
            pes_denominator = (
                thermal_efficiency / ref_heat_eff +
                electrical_efficiency / ref_elec_eff
            )
            if pes_denominator > 0:
                pes = (1 - 1 / pes_denominator) * 100
            else:
                pes = 0
        else:
            pes = 0

        calculation_trace.append(
            f"Reference efficiencies - Elec: {ref_elec_eff:.1%}, Heat: {ref_heat_eff:.1%}"
        )
        calculation_trace.append(f"Primary Energy Savings: {pes:.1f}%")

        # Calculate avoided emissions vs separate production
        # Separate production fuel for same outputs
        separate_elec_fuel = electricity_mmbtu / ref_elec_eff
        separate_heat_fuel = heat_mmbtu / ref_heat_eff
        separate_total_fuel = separate_elec_fuel + separate_heat_fuel

        separate_emissions = separate_total_fuel * co2_factor / 1000
        avoided_emissions = max(0, separate_emissions - total_emissions)

        calculation_trace.append(
            f"Separate production would use: {separate_total_fuel:.2f} MMBTU"
        )
        calculation_trace.append(
            f"Avoided emissions: {avoided_emissions:.2f} tonnes"
        )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "combustion"
        )

        # Warnings
        warnings = []
        if overall_efficiency < 0.60:
            warnings.append(
                f"Low overall efficiency ({overall_efficiency:.1%}) - "
                f"verify CHP is operating optimally"
            )
        if pes < 10 and pes > 0:
            warnings.append(
                f"Low PES ({pes:.1f}%) - may not qualify as high-efficiency CHP"
            )

        output = {
            "facility_id": validated_input.facility_id,
            "chp_id": validated_input.chp_id,
            "electrical_efficiency": round(electrical_efficiency, 4),
            "thermal_efficiency": round(thermal_efficiency, 4),
            "overall_efficiency": round(overall_efficiency, 4),
            "power_to_heat_ratio": round(power_to_heat_ratio, 4),
            "total_emissions_tonnes": round(total_emissions, 2),
            "electricity_allocation_tonnes": round(electricity_emissions, 2),
            "heat_allocation_tonnes": round(heat_emissions, 2),
            "electricity_intensity_kg_mwh": round(electricity_intensity, 2),
            "heat_intensity_kg_mwh": round(heat_intensity, 2),
            "primary_energy_savings_pct": round(pes, 1),
            "avoided_emissions_tonnes": round(avoided_emissions, 2),
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output

    def _convert_fuel_to_mmbtu(self, value: float, unit: str) -> float:
        """Convert fuel consumption to MMBTU."""
        unit_lower = unit.lower()
        if unit_lower == "mmbtu":
            return value
        elif unit_lower == "kwh":
            return value * 0.003412
        elif unit_lower == "mwh":
            return value * 3.412
        elif unit_lower == "gj":
            return value * 0.9478
        elif unit_lower == "therm":
            return value * 0.1
        elif unit_lower == "mcf":
            return value * 1.028
        else:
            return value
