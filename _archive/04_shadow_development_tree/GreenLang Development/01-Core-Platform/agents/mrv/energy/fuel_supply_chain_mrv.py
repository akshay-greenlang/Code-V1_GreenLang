# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-006: Fuel Supply Chain MRV Agent

Calculates upstream (Scope 3) emissions from fuel extraction, processing,
and transportation for energy sector applications.

Standards Reference:
    - GHG Protocol: Scope 3 Category 3 (Fuel and Energy)
    - GREET Model: Well-to-tank factors
    - IPCC: Upstream emission factors
    - EU RED II: Fuel lifecycle methodology

Example:
    >>> agent = FuelSupplyChainMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "PLANT-001",
    ...     "fuel_type": "natural_gas",
    ...     "fuel_quantity": 100000,
    ...     "fuel_unit": "MMBTU",
    ...     "origin_country": "US",
    ...     "transport_mode": "pipeline",
    ...     "transport_distance_km": 500,
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    FuelSupplyChainInput,
    FuelSupplyChainOutput,
    FuelType,
    EmissionScope,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class FuelSupplyChainMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-006: Fuel Supply Chain MRV Agent

    Calculates upstream emissions including:
    - Extraction/production
    - Processing/refining
    - Transportation
    - Fugitive emissions (methane leakage)

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    AGENT_ID = "GL-MRV-ENE-006"
    AGENT_NAME = "Fuel Supply Chain MRV Agent"
    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-006",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Fuel supply chain (Scope 3) emissions MRV"
    )

    # Upstream emission factors (kg CO2e per MMBTU of fuel delivered)
    UPSTREAM_FACTORS = {
        # Natural gas by region
        "natural_gas_us": {
            "extraction": 4.2,
            "processing": 1.8,
            "fugitive": 3.5,  # Methane leakage 1.5%
        },
        "natural_gas_russia": {
            "extraction": 5.5,
            "processing": 2.2,
            "fugitive": 6.0,
        },
        "natural_gas_qatar": {
            "extraction": 3.8,
            "processing": 2.0,
            "fugitive": 2.5,
        },
        # Coal
        "coal_bituminous": {
            "extraction": 8.5,
            "processing": 1.0,
            "fugitive": 2.0,  # Methane from mining
        },
        "coal_subbituminous": {
            "extraction": 7.5,
            "processing": 0.8,
            "fugitive": 1.5,
        },
        # Fuel oil
        "fuel_oil_no2": {
            "extraction": 5.0,
            "processing": 8.0,  # Refining
            "fugitive": 0.5,
        },
        "fuel_oil_no6": {
            "extraction": 5.0,
            "processing": 6.0,
            "fugitive": 0.5,
        },
        "diesel": {
            "extraction": 5.0,
            "processing": 9.0,
            "fugitive": 0.5,
        },
    }

    # Transport emission factors (kg CO2e per tonne-km)
    TRANSPORT_FACTORS = {
        "pipeline": 0.005,  # Gas pipeline
        "pipeline_liquid": 0.003,  # Liquid pipeline
        "ship": 0.010,  # Bulk carrier
        "rail": 0.025,
        "truck": 0.062,
    }

    # Fuel densities for transport calculation (kg/MMBTU equivalent)
    FUEL_DENSITIES = {
        "natural_gas": 19.2,  # kg/MMBTU
        "coal_bituminous": 47.6,
        "coal_subbituminous": 55.0,
        "fuel_oil_no2": 7.14,  # gallons * density
        "fuel_oil_no6": 6.67,
        "diesel": 7.14,
        "lpg": 11.0,
    }

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Fuel Supply Chain MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-006",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fuel supply chain MRV calculation.

        Args:
            inputs: Dictionary with FuelSupplyChainInput fields

        Returns:
            Dictionary with FuelSupplyChainOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = FuelSupplyChainInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        fuel_key = validated_input.fuel_type.value
        calculation_trace.append(
            f"Processing {validated_input.fuel_quantity:.2f} {validated_input.fuel_unit} "
            f"of {fuel_key}"
        )

        # Convert to MMBTU for standardized calculation
        fuel_mmbtu = self._convert_to_mmbtu(
            validated_input.fuel_quantity,
            validated_input.fuel_unit,
            validated_input.fuel_type
        )
        calculation_trace.append(f"Fuel quantity: {fuel_mmbtu:.2f} MMBTU")

        # Get upstream factors
        region_key = f"{fuel_key}_{validated_input.origin_country.lower()}"
        if region_key not in self.UPSTREAM_FACTORS:
            region_key = fuel_key  # Fall back to generic factors

        factors = self.UPSTREAM_FACTORS.get(region_key)
        if factors is None:
            # Default factors for unknown fuels
            factors = {"extraction": 5.0, "processing": 3.0, "fugitive": 1.0}
            calculation_trace.append(f"Using default factors for {fuel_key}")
        else:
            calculation_trace.append(f"Using factors for {region_key}")

        # Calculate extraction emissions
        extraction_emissions = fuel_mmbtu * factors["extraction"] / 1000
        calculation_trace.append(
            f"Extraction: {fuel_mmbtu:.2f} * {factors['extraction']:.2f} / 1000 = "
            f"{extraction_emissions:.4f} tonnes"
        )

        # Calculate processing emissions
        processing_emissions = fuel_mmbtu * factors["processing"] / 1000
        calculation_trace.append(
            f"Processing: {fuel_mmbtu:.2f} * {factors['processing']:.2f} / 1000 = "
            f"{processing_emissions:.4f} tonnes"
        )

        # Calculate fugitive emissions
        fugitive_emissions = fuel_mmbtu * factors["fugitive"] / 1000
        calculation_trace.append(
            f"Fugitive: {fuel_mmbtu:.2f} * {factors['fugitive']:.2f} / 1000 = "
            f"{fugitive_emissions:.4f} tonnes"
        )

        # Calculate transport emissions
        fuel_density = self.FUEL_DENSITIES.get(fuel_key, 20.0)  # kg/MMBTU
        fuel_mass_tonnes = (fuel_mmbtu * fuel_density) / 1000

        transport_mode = validated_input.transport_mode.lower()
        transport_factor = self.TRANSPORT_FACTORS.get(transport_mode, 0.02)

        transport_emissions = (
            fuel_mass_tonnes *
            validated_input.transport_distance_km *
            transport_factor / 1000
        )

        calculation_trace.append(
            f"Transport: {fuel_mass_tonnes:.2f} t * {validated_input.transport_distance_km:.0f} km * "
            f"{transport_factor:.4f} / 1000 = {transport_emissions:.4f} tonnes"
        )

        # Total upstream emissions
        total_upstream = (
            extraction_emissions +
            processing_emissions +
            transport_emissions +
            fugitive_emissions
        )

        calculation_trace.append(
            f"Total upstream: {extraction_emissions:.4f} + {processing_emissions:.4f} + "
            f"{transport_emissions:.4f} + {fugitive_emissions:.4f} = {total_upstream:.4f} tonnes"
        )

        # Calculate intensities
        if fuel_mmbtu > 0:
            upstream_intensity = total_upstream * 1000 / fuel_mmbtu  # kg/MMBTU
            # Convert to kg/kWh (1 MMBTU = 293.07 kWh)
            wtt_factor = upstream_intensity / 293.07
        else:
            upstream_intensity = 0
            wtt_factor = 0

        calculation_trace.append(
            f"Upstream intensity: {upstream_intensity:.2f} kg/MMBTU = "
            f"{wtt_factor:.4f} kg/kWh"
        )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "fugitive"
        )

        # Warnings
        warnings = []
        if validated_input.transport_distance_km > 5000:
            warnings.append(
                f"Long transport distance ({validated_input.transport_distance_km:.0f} km)"
            )
        if fugitive_emissions > extraction_emissions:
            warnings.append("High fugitive emissions - verify methane leakage rate")

        output = {
            "facility_id": validated_input.facility_id,
            "fuel_type": validated_input.fuel_type.value,
            "extraction_emissions_tonnes": round(extraction_emissions, 4),
            "processing_emissions_tonnes": round(processing_emissions, 4),
            "transport_emissions_tonnes": round(transport_emissions, 4),
            "fugitive_emissions_tonnes": round(fugitive_emissions, 4),
            "total_upstream_emissions_tonnes": round(total_upstream, 4),
            "upstream_intensity_kg_unit": round(upstream_intensity, 2),
            "wtt_factor_kg_kwh": round(wtt_factor, 4),
            "scope": EmissionScope.SCOPE_3_UPSTREAM.value,
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
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
        """Convert fuel quantity to MMBTU."""
        unit_lower = unit.lower()

        if unit_lower == "mmbtu":
            return value
        elif unit_lower in ("tonne", "tonnes", "metric_ton"):
            # Convert based on fuel type
            heating_values = {
                "natural_gas": 52.0,
                "coal_bituminous": 21.0,
                "coal_subbituminous": 18.0,
                "fuel_oil_no2": 38.0,
                "diesel": 37.0,
            }
            hv = heating_values.get(fuel_type.value, 25.0)
            return value * hv
        elif unit_lower in ("kg", "kilogram"):
            return value / 1000 * 25.0  # Approximate
        elif unit_lower == "mcf":
            return value * 1.028
        elif unit_lower in ("m3", "cubic_meter"):
            return value * 0.0364  # Natural gas
        elif unit_lower == "kwh":
            return value * 0.003412
        elif unit_lower == "mwh":
            return value * 3.412
        else:
            return value  # Assume MMBTU if unknown
