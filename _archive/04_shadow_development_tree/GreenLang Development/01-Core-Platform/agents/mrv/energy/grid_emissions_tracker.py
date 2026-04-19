# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-002: Grid Emissions Tracker Agent

Tracks grid emission factors and calculates Scope 2 emissions using
both location-based and market-based accounting methods per GHG Protocol.

Standards Reference:
    - GHG Protocol: Scope 2 Guidance (2015)
    - EPA eGRID: US grid emission factors
    - IEA: International grid factors
    - RE100: Renewable electricity criteria

Example:
    >>> agent = GridEmissionsTrackerAgent()
    >>> result = agent.process({
    ...     "facility_id": "OFFICE-001",
    ...     "grid_region": "us_camx",
    ...     "electricity_consumption_mwh": 1000,
    ...     "accounting_method": "dual",
    ...     "renewable_certificates_mwh": 500,
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    GridEmissionsInput,
    GridEmissionsOutput,
    GridRegion,
    EmissionScope,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class GridEmissionsTrackerAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-002: Grid Emissions Tracker Agent

    Calculates Scope 2 emissions using:
    - Location-based method (grid average factors)
    - Market-based method (contractual instruments)

    Supports:
    - REC/GO applications
    - PPA allocations
    - Supplier-specific factors
    - Residual mix calculations

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    AGENT_ID = "GL-MRV-ENE-002"
    AGENT_NAME = "Grid Emissions Tracker Agent"
    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-002",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Grid emissions tracking for Scope 2 accounting"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Grid Emissions Tracker Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-002",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

        # Residual mix factors by region (kg CO2e/MWh)
        self._residual_mix_factors = {
            "us_wecc": 398.0,
            "us_npcc": 287.0,
            "us_rfce": 430.0,
            "us_camx": 294.0,
            "eu_nordic": 320.0,
            "eu_central_west": 450.0,
            "eu_british": 380.0,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute grid emissions calculation.

        Args:
            inputs: Dictionary with GridEmissionsInput fields

        Returns:
            Dictionary with GridEmissionsOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = GridEmissionsInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        calculation_trace.append(
            f"Processing {validated_input.electricity_consumption_mwh:.2f} MWh "
            f"for facility {validated_input.facility_id}"
        )

        # Get grid emission factor
        grid_key = validated_input.grid_region.value
        location_factor = self.get_emission_factor("grid_kg_mwh", grid_key)
        calculation_trace.append(
            f"Location-based factor for {grid_key}: {location_factor:.2f} kg CO2e/MWh"
        )

        # Calculate location-based emissions
        location_emissions_tonnes = (
            validated_input.electricity_consumption_mwh * location_factor / 1000
        )
        calculation_trace.append(
            f"Location-based: {validated_input.electricity_consumption_mwh:.2f} * "
            f"{location_factor:.2f} / 1000 = {location_emissions_tonnes:.2f} tonnes"
        )

        # Calculate market-based emissions
        rec_reduction = 0.0
        ppa_reduction = 0.0

        remaining_mwh = validated_input.electricity_consumption_mwh

        # Apply RECs/GOs (zero emission)
        if validated_input.renewable_certificates_mwh:
            rec_mwh = min(
                validated_input.renewable_certificates_mwh,
                remaining_mwh
            )
            rec_reduction = rec_mwh * location_factor / 1000
            remaining_mwh -= rec_mwh
            calculation_trace.append(
                f"RECs applied: {rec_mwh:.2f} MWh, reduction: {rec_reduction:.2f} tonnes"
            )

        # Apply PPA allocation
        if validated_input.ppa_allocation_mwh:
            ppa_mwh = min(validated_input.ppa_allocation_mwh, remaining_mwh)
            ppa_reduction = ppa_mwh * location_factor / 1000
            remaining_mwh -= ppa_mwh
            calculation_trace.append(
                f"PPA applied: {ppa_mwh:.2f} MWh, reduction: {ppa_reduction:.2f} tonnes"
            )

        # Calculate market-based factor
        if validated_input.supplier_emission_factor is not None:
            market_factor = validated_input.supplier_emission_factor
            calculation_trace.append(
                f"Using supplier-specific factor: {market_factor:.2f} kg/MWh"
            )
        elif validated_input.residual_mix_factor is not None:
            market_factor = validated_input.residual_mix_factor
            calculation_trace.append(
                f"Using provided residual mix: {market_factor:.2f} kg/MWh"
            )
        elif grid_key in self._residual_mix_factors:
            market_factor = self._residual_mix_factors[grid_key]
            calculation_trace.append(
                f"Using default residual mix: {market_factor:.2f} kg/MWh"
            )
        else:
            market_factor = location_factor
            calculation_trace.append(
                f"No residual mix available, using grid average: {market_factor:.2f} kg/MWh"
            )

        # Calculate remaining emissions
        remaining_emissions = remaining_mwh * market_factor / 1000
        market_emissions_tonnes = remaining_emissions

        # Effective market factor
        if validated_input.electricity_consumption_mwh > 0:
            effective_market_factor = (
                market_emissions_tonnes * 1000 /
                validated_input.electricity_consumption_mwh
            )
        else:
            effective_market_factor = 0.0

        calculation_trace.append(
            f"Market-based: {remaining_mwh:.2f} MWh * {market_factor:.2f} / 1000 = "
            f"{market_emissions_tonnes:.2f} tonnes"
        )
        calculation_trace.append(
            f"Effective market factor: {effective_market_factor:.2f} kg/MWh"
        )

        # Determine uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "grid"
        )

        # Warnings
        warnings = []
        if location_factor > 700:
            warnings.append(f"High carbon grid ({location_factor:.0f} kg/MWh)")

        output = {
            "facility_id": validated_input.facility_id,
            "grid_region": validated_input.grid_region.value,
            "location_based_co2e_tonnes": round(location_emissions_tonnes, 2),
            "location_emission_factor_kg_mwh": round(location_factor, 2),
            "market_based_co2e_tonnes": round(market_emissions_tonnes, 2),
            "market_emission_factor_kg_mwh": round(effective_market_factor, 2),
            "rec_reduction_tonnes": round(rec_reduction, 2),
            "ppa_reduction_tonnes": round(ppa_reduction, 2),
            "scope_2_location": round(location_emissions_tonnes, 2),
            "scope_2_market": round(market_emissions_tonnes, 2),
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output
