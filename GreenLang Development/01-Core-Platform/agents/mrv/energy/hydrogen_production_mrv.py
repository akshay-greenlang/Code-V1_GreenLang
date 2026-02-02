# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-008: Hydrogen Production MRV Agent

Calculates emissions from hydrogen production across all production
pathways including SMR, electrolysis, and gasification.

Standards Reference:
    - EU Delegated Act: Renewable hydrogen definition
    - US DOE: Clean Hydrogen Production Standard
    - CertifHy: European hydrogen certification
    - ISO 14687: Hydrogen fuel quality

Example:
    >>> agent = HydrogenProductionMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "H2-PLANT-001",
    ...     "production_id": "ELECTROLYZER-1",
    ...     "production_method": "electrolysis_renewable",
    ...     "hydrogen_output_kg": 10000,
    ...     "feedstock_consumption": 0,
    ...     "feedstock_unit": "MMBTU",
    ...     "electricity_consumption_kwh": 550000,
    ...     "grid_region": "us_camx",
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    HydrogenProductionInput,
    HydrogenProductionOutput,
    HydrogenProductionMethod,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class HydrogenProductionMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-008: Hydrogen Production MRV Agent

    Calculates emissions and carbon intensity for:
    - Steam methane reforming (SMR) with/without CCS
    - Electrolysis (grid, renewable, nuclear)
    - Coal gasification with/without CCS
    - Biomass gasification

    Determines hydrogen "color" classification and low-carbon eligibility.

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    AGENT_ID = "GL-MRV-ENE-008"
    AGENT_NAME = "Hydrogen Production MRV Agent"
    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-008",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Hydrogen production emissions MRV and certification"
    )

    # Process emission factors (kg CO2e per kg H2 before electricity)
    PROCESS_FACTORS = {
        "steam_methane_reforming": 9.0,  # Grey hydrogen baseline
        "steam_methane_reforming_with_ccs": 1.5,  # Blue with 85% capture
        "autothermal_reforming": 8.5,
        "autothermal_reforming_with_ccs": 1.2,
        "coal_gasification": 18.0,
        "coal_gasification_with_ccs": 3.0,
        "biomass_gasification": 2.0,  # Biogenic not counted
        "electrolysis_grid": 0.0,  # Electricity-only
        "electrolysis_renewable": 0.0,
        "electrolysis_nuclear": 0.0,
    }

    # Electricity consumption (kWh per kg H2)
    ELECTRICITY_CONSUMPTION = {
        "steam_methane_reforming": 1.0,
        "steam_methane_reforming_with_ccs": 3.5,  # CCS energy penalty
        "autothermal_reforming": 0.8,
        "autothermal_reforming_with_ccs": 3.0,
        "coal_gasification": 2.0,
        "coal_gasification_with_ccs": 5.0,
        "biomass_gasification": 2.5,
        "electrolysis_grid": 55.0,  # PEM/Alkaline typical
        "electrolysis_renewable": 55.0,
        "electrolysis_nuclear": 55.0,
    }

    # Hydrogen color classification thresholds (kg CO2e / kg H2)
    COLOR_THRESHOLDS = {
        "green": 1.0,  # Renewable electrolysis
        "pink": 1.0,  # Nuclear electrolysis
        "blue": 4.0,  # SMR with CCS
        "turquoise": 3.0,  # Methane pyrolysis
        "yellow": 8.0,  # Grid electrolysis (varies)
        "grey": 999.0,  # Unabated SMR
    }

    # Low-carbon thresholds by standard
    LOW_CARBON_THRESHOLDS = {
        "us_doe": 4.0,  # US DOE Clean Hydrogen Standard
        "eu_delegated": 3.4,  # EU Renewable/Low-carbon H2
        "certifhy": 4.4,  # CertifHy Green Hydrogen
    }

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Hydrogen Production MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-008",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hydrogen production MRV calculation.

        Args:
            inputs: Dictionary with HydrogenProductionInput fields

        Returns:
            Dictionary with HydrogenProductionOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = HydrogenProductionInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        method_key = validated_input.production_method.value
        calculation_trace.append(
            f"Processing {validated_input.hydrogen_output_kg:.2f} kg H2 via {method_key}"
        )

        # Get process emission factor
        process_factor = self.PROCESS_FACTORS.get(method_key, 10.0)
        process_emissions = (
            validated_input.hydrogen_output_kg * process_factor / 1000
        )

        calculation_trace.append(
            f"Process factor: {process_factor:.2f} kg CO2e/kg H2"
        )
        calculation_trace.append(
            f"Process emissions: {validated_input.hydrogen_output_kg:.2f} * "
            f"{process_factor:.2f} / 1000 = {process_emissions:.4f} tonnes"
        )

        # Calculate electricity emissions
        if validated_input.electricity_consumption_kwh > 0:
            # Get electricity emission factor
            if "renewable" in method_key:
                elec_factor = 0.0  # Zero for dedicated renewables
                calculation_trace.append(
                    "Using zero emission factor for renewable electricity"
                )
            elif "nuclear" in method_key:
                elec_factor = 0.012  # Lifecycle nuclear
                calculation_trace.append(
                    "Using nuclear lifecycle factor: 0.012 kg/kWh"
                )
            elif validated_input.electricity_emission_factor_kg_kwh is not None:
                elec_factor = validated_input.electricity_emission_factor_kg_kwh
                calculation_trace.append(
                    f"Using provided electricity factor: {elec_factor:.4f} kg/kWh"
                )
            elif validated_input.grid_region is not None:
                grid_factor = self.get_emission_factor(
                    "grid_kg_mwh",
                    validated_input.grid_region.value
                )
                elec_factor = grid_factor / 1000  # Convert to kg/kWh
                calculation_trace.append(
                    f"Using grid factor ({validated_input.grid_region.value}): "
                    f"{elec_factor:.4f} kg/kWh"
                )
            else:
                elec_factor = 0.4  # Default average
                calculation_trace.append(
                    "Using default electricity factor: 0.4 kg/kWh"
                )

            electricity_emissions = (
                validated_input.electricity_consumption_kwh * elec_factor / 1000
            )
            calculation_trace.append(
                f"Electricity emissions: {validated_input.electricity_consumption_kwh:.2f} kWh * "
                f"{elec_factor:.4f} / 1000 = {electricity_emissions:.4f} tonnes"
            )
        else:
            electricity_emissions = 0.0
            elec_factor = 0.0

        # Calculate feedstock upstream emissions
        feedstock_emissions = 0.0
        if validated_input.feedstock_consumption > 0:
            # Natural gas upstream ~0.5 kg CO2e/kg H2 for SMR
            if "methane" in method_key or "reforming" in method_key:
                feedstock_factor = 0.5
            elif "coal" in method_key:
                feedstock_factor = 1.5
            else:
                feedstock_factor = 0.2

            feedstock_emissions = (
                validated_input.hydrogen_output_kg * feedstock_factor / 1000
            )
            calculation_trace.append(
                f"Feedstock upstream: {feedstock_emissions:.4f} tonnes"
            )

        # Calculate captured emissions (CCS)
        captured_emissions = 0.0
        if validated_input.ccs_capture_rate is not None:
            base_process_factor = self.PROCESS_FACTORS.get(
                method_key.replace("_with_ccs", ""), process_factor
            )
            captured_emissions = (
                validated_input.hydrogen_output_kg *
                base_process_factor *
                validated_input.ccs_capture_rate / 1000
            )
            calculation_trace.append(
                f"CO2 captured ({validated_input.ccs_capture_rate:.0%}): "
                f"{captured_emissions:.4f} tonnes"
            )

        # Total emissions
        total_emissions = (
            process_emissions +
            electricity_emissions +
            feedstock_emissions
        )
        calculation_trace.append(
            f"Total emissions: {process_emissions:.4f} + {electricity_emissions:.4f} + "
            f"{feedstock_emissions:.4f} = {total_emissions:.4f} tonnes"
        )

        # Carbon intensity
        if validated_input.hydrogen_output_kg > 0:
            carbon_intensity = total_emissions * 1000 / validated_input.hydrogen_output_kg
        else:
            carbon_intensity = 0.0

        calculation_trace.append(
            f"Carbon intensity: {total_emissions * 1000:.2f} / "
            f"{validated_input.hydrogen_output_kg:.2f} = {carbon_intensity:.2f} kg CO2e/kg H2"
        )

        # Specific energy consumption
        if validated_input.hydrogen_output_kg > 0:
            specific_energy = (
                validated_input.electricity_consumption_kwh /
                validated_input.hydrogen_output_kg
            )
        else:
            specific_energy = 0.0

        # Determine hydrogen color
        hydrogen_color = self._determine_color(
            carbon_intensity,
            method_key
        )
        calculation_trace.append(f"Hydrogen classification: {hydrogen_color}")

        # Check low-carbon eligibility
        low_carbon_eligible = carbon_intensity <= self.LOW_CARBON_THRESHOLDS["us_doe"]
        certification_threshold = self.LOW_CARBON_THRESHOLDS["us_doe"]

        calculation_trace.append(
            f"Low-carbon eligible (US DOE <{certification_threshold} kg): "
            f"{'Yes' if low_carbon_eligible else 'No'}"
        )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "combustion" if "reforming" in method_key else "grid"
        )

        # Warnings
        warnings = []
        if specific_energy > 60 and "electrolysis" in method_key:
            warnings.append(
                f"High specific energy ({specific_energy:.1f} kWh/kg) - "
                f"typical is 50-55 kWh/kg"
            )
        if carbon_intensity > 10:
            warnings.append(
                f"High carbon intensity ({carbon_intensity:.1f} kg/kg H2)"
            )

        output = {
            "facility_id": validated_input.facility_id,
            "production_id": validated_input.production_id,
            "production_method": validated_input.production_method.value,
            "hydrogen_output_kg": round(validated_input.hydrogen_output_kg, 2),
            "specific_energy_consumption_kwh_kg": round(specific_energy, 2),
            "process_emissions_tonnes": round(process_emissions, 4),
            "electricity_emissions_tonnes": round(electricity_emissions, 4),
            "feedstock_emissions_tonnes": round(feedstock_emissions, 4),
            "captured_emissions_tonnes": round(captured_emissions, 4),
            "total_emissions_tonnes": round(total_emissions, 4),
            "carbon_intensity_kg_co2_kg_h2": round(carbon_intensity, 2),
            "hydrogen_color": hydrogen_color,
            "low_carbon_eligible": low_carbon_eligible,
            "certification_threshold_kg_co2_kg_h2": certification_threshold,
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output

    def _determine_color(
        self,
        carbon_intensity: float,
        method: str
    ) -> str:
        """
        Determine hydrogen color classification.

        Args:
            carbon_intensity: kg CO2e per kg H2
            method: Production method

        Returns:
            Hydrogen color (green, blue, grey, etc.)
        """
        if "renewable" in method and carbon_intensity <= 1.0:
            return "green"
        elif "nuclear" in method and carbon_intensity <= 1.0:
            return "pink"
        elif "biomass" in method and carbon_intensity <= 3.0:
            return "turquoise"
        elif "ccs" in method and carbon_intensity <= 4.0:
            return "blue"
        elif "electrolysis" in method:
            if carbon_intensity <= 4.0:
                return "yellow"  # Low-carbon grid
            else:
                return "grey"  # High-carbon grid
        else:
            return "grey"  # Unabated fossil
