# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-004: Storage Systems MRV Agent

Calculates emissions associated with energy storage systems including
charging emissions, avoided emissions from dispatch, and lifecycle impacts.

Standards Reference:
    - GHG Protocol: Scope 2 Guidance
    - IPCC: Lifecycle assessment guidelines
    - IRENA: Battery storage LCA methodology

Example:
    >>> agent = StorageSystemsMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "BESS-001",
    ...     "storage_id": "LI-ION-ARRAY-1",
    ...     "technology": "li_ion_nmc",
    ...     "rated_capacity_mwh": 100,
    ...     "rated_power_mw": 25,
    ...     "energy_charged_mwh": 2500,
    ...     "energy_discharged_mwh": 2200,
    ...     "round_trip_efficiency": 0.88,
    ...     "grid_region": "us_camx",
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    StorageSystemsInput,
    StorageSystemsOutput,
    StorageTechnology,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class StorageSystemsMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-004: Storage Systems MRV Agent

    Calculates:
    - Charging emissions (Scope 2)
    - Avoided emissions from dispatch
    - Net emissions impact
    - Embedded/lifecycle emissions

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-004",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Energy storage systems emissions MRV"
    )

    # Embedded emissions by technology (kg CO2e / kWh capacity)
    EMBEDDED_EMISSIONS = {
        "li_ion_nmc": 175.0,
        "li_ion_lfp": 150.0,
        "li_ion_nca": 180.0,
        "lead_acid": 80.0,
        "flow_vanadium": 120.0,
        "flow_zinc_bromine": 100.0,
        "sodium_sulfur": 130.0,
        "pumped_hydro": 5.0,
        "caes": 8.0,
        "flywheel": 50.0,
        "hydrogen_storage": 30.0,
    }

    # Expected lifetime cycles by technology
    LIFETIME_CYCLES = {
        "li_ion_nmc": 4000,
        "li_ion_lfp": 6000,
        "li_ion_nca": 3500,
        "lead_acid": 1500,
        "flow_vanadium": 10000,
        "flow_zinc_bromine": 8000,
        "sodium_sulfur": 4500,
        "pumped_hydro": 50000,
        "caes": 30000,
        "flywheel": 100000,
        "hydrogen_storage": 20000,
    }

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Storage Systems MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-004",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute storage systems MRV calculation.

        Args:
            inputs: Dictionary with StorageSystemsInput fields

        Returns:
            Dictionary with StorageSystemsOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = StorageSystemsInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        calculation_trace.append(
            f"Processing {validated_input.technology.value} storage "
            f"{validated_input.storage_id}"
        )

        # Calculate operational metrics
        storage_losses = (
            validated_input.energy_charged_mwh -
            validated_input.energy_discharged_mwh
        )
        actual_efficiency = (
            validated_input.energy_discharged_mwh /
            validated_input.energy_charged_mwh
            if validated_input.energy_charged_mwh > 0 else 0
        )

        # Equivalent full cycles
        total_cycles = (
            validated_input.energy_discharged_mwh /
            validated_input.rated_capacity_mwh
            if validated_input.rated_capacity_mwh > 0 else 0
        )

        # Capacity utilization (based on theoretical max cycles)
        theoretical_max_cycles = 365  # One cycle per day annually
        capacity_utilization = min(total_cycles / theoretical_max_cycles, 1.0)

        calculation_trace.append(
            f"Charged: {validated_input.energy_charged_mwh:.2f} MWh, "
            f"Discharged: {validated_input.energy_discharged_mwh:.2f} MWh"
        )
        calculation_trace.append(
            f"Round-trip efficiency: {actual_efficiency:.1%}, "
            f"Losses: {storage_losses:.2f} MWh"
        )
        calculation_trace.append(
            f"Equivalent full cycles: {total_cycles:.1f}"
        )

        # Get charging emission factor
        grid_key = validated_input.grid_region.value
        if validated_input.charging_emission_factor_kg_mwh is not None:
            charging_factor = validated_input.charging_emission_factor_kg_mwh
            calculation_trace.append(
                f"Using provided charging factor: {charging_factor:.2f} kg/MWh"
            )
        else:
            charging_factor = self.get_emission_factor("grid_kg_mwh", grid_key)
            calculation_trace.append(
                f"Using grid factor ({grid_key}): {charging_factor:.2f} kg/MWh"
            )

        # Calculate charging emissions
        charging_emissions = (
            validated_input.energy_charged_mwh * charging_factor / 1000
        )
        calculation_trace.append(
            f"Charging emissions: {validated_input.energy_charged_mwh:.2f} * "
            f"{charging_factor:.2f} / 1000 = {charging_emissions:.2f} tonnes"
        )

        # Calculate avoided emissions
        # Storage typically dispatches during high-emission periods
        # Use 1.2x grid average as proxy for marginal emissions avoided
        marginal_factor = charging_factor * 1.2
        avoided_emissions = (
            validated_input.energy_discharged_mwh * marginal_factor / 1000
        )
        calculation_trace.append(
            f"Avoided emissions (marginal): {validated_input.energy_discharged_mwh:.2f} * "
            f"{marginal_factor:.2f} / 1000 = {avoided_emissions:.2f} tonnes"
        )

        # Net emissions impact
        net_emissions = charging_emissions - avoided_emissions
        calculation_trace.append(
            f"Net emissions: {charging_emissions:.2f} - {avoided_emissions:.2f} = "
            f"{net_emissions:.2f} tonnes"
        )

        # Emission intensity of discharged energy
        if validated_input.energy_discharged_mwh > 0:
            emission_intensity = (
                charging_emissions * 1000 /
                validated_input.energy_discharged_mwh
            )
        else:
            emission_intensity = 0.0

        # Calculate embedded emissions (annualized)
        tech_key = validated_input.technology.value
        embedded_per_kwh = self.EMBEDDED_EMISSIONS.get(tech_key, 100.0)
        lifetime_cycles = self.LIFETIME_CYCLES.get(tech_key, 5000)

        total_embedded = (
            validated_input.rated_capacity_mwh * 1000 *  # MWh to kWh
            embedded_per_kwh / 1000  # kg to tonnes
        )
        annual_embedded = total_embedded * (total_cycles / lifetime_cycles)

        calculation_trace.append(
            f"Embedded emissions: {validated_input.rated_capacity_mwh * 1000:.0f} kWh * "
            f"{embedded_per_kwh:.0f} kg/kWh = {total_embedded:.2f} tonnes total"
        )
        calculation_trace.append(
            f"Annualized: {total_embedded:.2f} * ({total_cycles:.1f} / {lifetime_cycles}) = "
            f"{annual_embedded:.4f} tonnes"
        )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "grid"
        )

        # Warnings
        warnings = []
        if actual_efficiency < validated_input.round_trip_efficiency - 0.05:
            warnings.append(
                f"Actual efficiency ({actual_efficiency:.1%}) below rated "
                f"({validated_input.round_trip_efficiency:.1%})"
            )
        if total_cycles > 500:
            warnings.append(
                f"High cycle count ({total_cycles:.0f}) - verify battery degradation"
            )

        output = {
            "facility_id": validated_input.facility_id,
            "storage_id": validated_input.storage_id,
            "technology": validated_input.technology.value,
            "total_cycles": round(total_cycles, 2),
            "capacity_utilization": round(capacity_utilization, 4),
            "storage_losses_mwh": round(storage_losses, 2),
            "charging_emissions_tonnes": round(charging_emissions, 4),
            "avoided_emissions_tonnes": round(avoided_emissions, 4),
            "net_emissions_tonnes": round(net_emissions, 4),
            "emission_intensity_kg_mwh": round(emission_intensity, 2),
            "embedded_emissions_tonnes": round(annual_embedded, 4),
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output
