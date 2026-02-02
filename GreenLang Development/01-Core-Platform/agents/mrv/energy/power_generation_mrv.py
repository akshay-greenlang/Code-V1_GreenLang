# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-001: Power Generation MRV Agent

Calculates greenhouse gas emissions from power generation facilities
including fossil fuel combustion, emission intensities, and regulatory
compliance tracking.

Standards Reference:
    - GHG Protocol: Corporate Standard (Scope 1)
    - EPA Part 98 Subpart D: Electricity Generation
    - EPA AP-42: Compilation of Air Emission Factors
    - ISO 14064-1: GHG accounting principles

Example:
    >>> agent = PowerGenerationMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "PLANT-001",
    ...     "unit_id": "UNIT-1",
    ...     "generation_type": "combined_cycle_gas_turbine",
    ...     "fuel_type": "natural_gas",
    ...     "fuel_consumption": 50000,
    ...     "fuel_consumption_unit": "MMBTU",
    ...     "net_generation_mwh": 5000,
    ...     "reporting_period_start": "2024-01-01T00:00:00Z",
    ...     "reporting_period_end": "2024-01-31T23:59:59Z",
    ... })
    >>> print(f"CO2: {result['co2_tonnes']:.2f} tonnes")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    PowerGenerationInput,
    PowerGenerationOutput,
    GenerationType,
    FuelType,
    EmissionScope,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class PowerGenerationMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-001: Power Generation MRV Agent

    Calculates emissions from power generation using:
    - Fuel consumption method (primary)
    - CEMS data integration (when available)
    - Mass balance approach (for specific fuels)

    This agent is CRITICAL PATH - all calculations are deterministic
    with zero AI/LLM involvement. Full audit trail is maintained.

    Attributes:
        agent_id: GL-MRV-ENE-001
        category: CRITICAL (zero hallucination)
        supported_technologies: List of supported generation types
    """

    AGENT_ID = "GL-MRV-ENE-001"
    AGENT_NAME = "Power Generation MRV Agent"
    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-001",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Power generation emissions MRV with EPA/GHG Protocol compliance"
    )

    # Supported generation technologies
    SUPPORTED_TECHNOLOGIES = [
        GenerationType.CCGT,
        GenerationType.OCGT,
        GenerationType.COAL_SUBCRITICAL,
        GenerationType.COAL_SUPERCRITICAL,
        GenerationType.COAL_USC,
        GenerationType.CHP_GAS,
        GenerationType.CHP_COAL,
        GenerationType.CHP_BIOMASS,
        GenerationType.BIOMASS,
    ]

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Power Generation MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-001",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

        self.logger.info("PowerGenerationMRVAgent initialized")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute power generation emissions calculation.

        Calculation methodology:
        1. Validate inputs and determine calculation approach
        2. Calculate CO2 from fuel combustion
        3. Calculate CH4 and N2O emissions
        4. Calculate emission intensity (kg/MWh)
        5. Apply quality/uncertainty assessment

        Args:
            inputs: Dictionary with PowerGenerationInput fields

        Returns:
            Dictionary with PowerGenerationOutput fields
        """
        calculation_trace: List[str] = []

        # Parse and validate inputs
        try:
            validated_input = PowerGenerationInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        calculation_trace.append(
            f"Processing unit {validated_input.unit_id} at facility {validated_input.facility_id}"
        )

        # Determine calculation method
        use_cems = validated_input.cems_co2_tons is not None
        calculation_method = "CEMS" if use_cems else "fuel_combustion"
        calculation_trace.append(f"Using calculation method: {calculation_method}")

        # Calculate CO2 emissions
        if use_cems:
            # Use CEMS data directly
            co2_tonnes = validated_input.cems_co2_tons * 0.907185  # US tons to metric
            calculation_trace.append(
                f"CO2 from CEMS: {validated_input.cems_co2_tons} US tons = {co2_tonnes:.2f} metric tonnes"
            )
            emission_factor_co2 = None
        else:
            # Calculate from fuel consumption
            if validated_input.fuel_type is None:
                raise ValueError("fuel_type required for non-CEMS calculation")
            if validated_input.fuel_consumption is None:
                raise ValueError("fuel_consumption required for non-CEMS calculation")

            # Get emission factor (kg CO2 / MMBTU)
            fuel_key = validated_input.fuel_type.value
            emission_factor_co2 = self.get_emission_factor("co2_mmbtu", fuel_key)

            # Convert fuel consumption to MMBTU if needed
            fuel_mmbtu = self._convert_to_mmbtu(
                validated_input.fuel_consumption,
                validated_input.fuel_consumption_unit,
                validated_input.fuel_type
            )

            # Calculate CO2 (kg -> tonnes)
            co2_kg = fuel_mmbtu * emission_factor_co2
            co2_tonnes = co2_kg / 1000

            calculation_trace.append(
                f"Fuel: {fuel_mmbtu:.2f} MMBTU of {fuel_key}"
            )
            calculation_trace.append(
                f"CO2 factor: {emission_factor_co2:.2f} kg/MMBTU"
            )
            calculation_trace.append(
                f"CO2 = {fuel_mmbtu:.2f} * {emission_factor_co2:.2f} / 1000 = {co2_tonnes:.2f} tonnes"
            )

        # Calculate CH4 emissions
        ch4_tonnes_co2e = 0.0
        if validated_input.fuel_type and validated_input.fuel_consumption:
            try:
                ch4_factor = self.get_emission_factor(
                    "ch4_mmbtu",
                    validated_input.fuel_type.value,
                    default=0.0
                )
                fuel_mmbtu = self._convert_to_mmbtu(
                    validated_input.fuel_consumption,
                    validated_input.fuel_consumption_unit,
                    validated_input.fuel_type
                )
                ch4_g = fuel_mmbtu * ch4_factor
                ch4_tonnes = ch4_g / 1_000_000
                ch4_tonnes_co2e = ch4_tonnes * self.GWP_CH4

                calculation_trace.append(
                    f"CH4 = {fuel_mmbtu:.2f} * {ch4_factor:.2f} g/MMBTU = {ch4_g:.2f} g"
                )
                calculation_trace.append(
                    f"CH4 CO2e = {ch4_tonnes:.6f} * {self.GWP_CH4} = {ch4_tonnes_co2e:.4f} tonnes"
                )
            except KeyError:
                calculation_trace.append("CH4 factor not available for fuel type")

        # Calculate N2O emissions
        n2o_tonnes_co2e = 0.0
        if validated_input.fuel_type and validated_input.fuel_consumption:
            try:
                n2o_factor = self.get_emission_factor(
                    "n2o_mmbtu",
                    validated_input.fuel_type.value,
                    default=0.0
                )
                fuel_mmbtu = self._convert_to_mmbtu(
                    validated_input.fuel_consumption,
                    validated_input.fuel_consumption_unit,
                    validated_input.fuel_type
                )
                n2o_g = fuel_mmbtu * n2o_factor
                n2o_tonnes = n2o_g / 1_000_000
                n2o_tonnes_co2e = n2o_tonnes * self.GWP_N2O

                calculation_trace.append(
                    f"N2O = {fuel_mmbtu:.2f} * {n2o_factor:.2f} g/MMBTU = {n2o_g:.2f} g"
                )
                calculation_trace.append(
                    f"N2O CO2e = {n2o_tonnes:.6f} * {self.GWP_N2O} = {n2o_tonnes_co2e:.4f} tonnes"
                )
            except KeyError:
                calculation_trace.append("N2O factor not available for fuel type")

        # Total GHG emissions
        total_ghg_tonnes = co2_tonnes + ch4_tonnes_co2e + n2o_tonnes_co2e
        calculation_trace.append(
            f"Total GHG = {co2_tonnes:.2f} + {ch4_tonnes_co2e:.4f} + {n2o_tonnes_co2e:.4f} = {total_ghg_tonnes:.2f} tonnes CO2e"
        )

        # Calculate emission intensity
        if validated_input.net_generation_mwh > 0:
            emission_intensity = (total_ghg_tonnes * 1000) / validated_input.net_generation_mwh
            calculation_trace.append(
                f"Intensity = {total_ghg_tonnes * 1000:.2f} kg / {validated_input.net_generation_mwh:.2f} MWh = {emission_intensity:.2f} kg/MWh"
            )
        else:
            emission_intensity = 0.0
            calculation_trace.append("No generation - intensity set to 0")

        # Determine uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "combustion"
        )

        # Build emission factors used dictionary
        emission_factors_used = {}
        if emission_factor_co2:
            emission_factors_used["co2_kg_mmbtu"] = emission_factor_co2
        if validated_input.fuel_type:
            try:
                emission_factors_used["ch4_g_mmbtu"] = self.get_emission_factor(
                    "ch4_mmbtu", validated_input.fuel_type.value, 0.0
                )
                emission_factors_used["n2o_g_mmbtu"] = self.get_emission_factor(
                    "n2o_mmbtu", validated_input.fuel_type.value, 0.0
                )
            except KeyError:
                pass

        # Determine validation status
        validation_status = "PASS"
        warnings = []

        if emission_intensity > 1000:
            warnings.append(
                f"High emission intensity ({emission_intensity:.0f} kg/MWh) - verify heat rate"
            )
            validation_status = "WARN"

        if validated_input.heat_rate_btu_kwh and validated_input.heat_rate_btu_kwh > 15000:
            warnings.append(
                f"High heat rate ({validated_input.heat_rate_btu_kwh:.0f} BTU/kWh) - check efficiency"
            )

        # Prepare output
        output = {
            "facility_id": validated_input.facility_id,
            "unit_id": validated_input.unit_id,
            "generation_type": validated_input.generation_type.value,
            "co2_tonnes": round(co2_tonnes, 2),
            "ch4_tonnes_co2e": round(ch4_tonnes_co2e, 4),
            "n2o_tonnes_co2e": round(n2o_tonnes_co2e, 4),
            "total_ghg_tonnes_co2e": round(total_ghg_tonnes, 2),
            "emission_intensity_kg_mwh": round(emission_intensity, 2),
            "scope": EmissionScope.SCOPE_1.value,
            "methodology": f"GHG Protocol / EPA Part 98 - {calculation_method}",
            "emission_factors_used": emission_factors_used,
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": validation_status,
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output

    def _convert_to_mmbtu(
        self,
        value: float,
        unit: str,
        fuel_type: FuelType
    ) -> float:
        """
        Convert fuel consumption to MMBTU.

        Args:
            value: Fuel consumption value
            unit: Current unit (MMBTU, MCF, therms, tons, gallons)
            fuel_type: Fuel type for heating value lookup

        Returns:
            Fuel consumption in MMBTU
        """
        unit_lower = unit.lower()

        if unit_lower == "mmbtu":
            return value
        elif unit_lower == "mcf":
            return value * self.get_emission_factor(
                "heating_value_mmbtu", "natural_gas_mcf"
            )
        elif unit_lower in ("therm", "therms"):
            return value * self.get_emission_factor(
                "heating_value_mmbtu", "natural_gas_therm"
            )
        elif unit_lower in ("ton", "tons", "short_ton"):
            return value * self.get_emission_factor(
                "heating_value_mmbtu", "coal_ton"
            )
        elif unit_lower in ("gallon", "gallons", "gal"):
            if "oil" in fuel_type.value or fuel_type == FuelType.DIESEL:
                return value * self.get_emission_factor(
                    "heating_value_mmbtu", "fuel_oil_gallon"
                )
            elif fuel_type == FuelType.LPG:
                return value * self.get_emission_factor(
                    "heating_value_mmbtu", "lpg_gallon"
                )
            else:
                return value * 0.138  # Default diesel equivalent
        elif unit_lower == "kwh":
            return value * 0.003412  # 1 kWh = 0.003412 MMBTU
        elif unit_lower == "mwh":
            return value * 3.412  # 1 MWh = 3.412 MMBTU
        else:
            raise ValueError(f"Unknown fuel unit: {unit}")


# Convenience function for direct usage
def calculate_power_generation_emissions(
    facility_id: str,
    unit_id: str,
    generation_type: str,
    fuel_type: str,
    fuel_consumption: float,
    fuel_consumption_unit: str,
    net_generation_mwh: float,
    reporting_period_start: str,
    reporting_period_end: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate power generation emissions.

    Convenience function for direct usage without instantiating agent.

    Args:
        facility_id: Facility identifier
        unit_id: Generation unit identifier
        generation_type: Generation technology type
        fuel_type: Primary fuel type
        fuel_consumption: Fuel consumption value
        fuel_consumption_unit: Fuel consumption unit
        net_generation_mwh: Net electricity generation (MWh)
        reporting_period_start: Start of reporting period (ISO format)
        reporting_period_end: End of reporting period (ISO format)
        **kwargs: Additional optional parameters

    Returns:
        Calculation results dictionary
    """
    agent = PowerGenerationMRVAgent()
    return agent.process({
        "facility_id": facility_id,
        "unit_id": unit_id,
        "generation_type": generation_type,
        "fuel_type": fuel_type,
        "fuel_consumption": fuel_consumption,
        "fuel_consumption_unit": fuel_consumption_unit,
        "net_generation_mwh": net_generation_mwh,
        "reporting_period_start": reporting_period_start,
        "reporting_period_end": reporting_period_end,
        **kwargs
    })
