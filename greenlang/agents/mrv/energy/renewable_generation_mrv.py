# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-003: Renewable Generation MRV Agent

Tracks renewable energy generation, calculates avoided emissions,
and manages renewable energy certificate (REC) eligibility.

Standards Reference:
    - GHG Protocol: Scope 2 Guidance
    - RE100: Technical Criteria
    - IPCC: Lifecycle emission factors
    - I-REC Standard: Certificate eligibility

Example:
    >>> agent = RenewableGenerationMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "SOLAR-FARM-001",
    ...     "asset_id": "PV-ARRAY-1",
    ...     "technology": "solar_pv_utility",
    ...     "installed_capacity_mw": 50,
    ...     "net_generation_mwh": 8500,
    ...     "grid_region": "us_camx",
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    RenewableGenerationInput,
    RenewableGenerationOutput,
    GenerationType,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class RenewableGenerationMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-003: Renewable Generation MRV Agent

    Calculates:
    - Avoided emissions from renewable generation
    - Lifecycle emissions (manufacturing, installation, decommissioning)
    - Net climate benefit
    - REC eligibility

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-003",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Renewable energy generation MRV and avoided emissions"
    )

    RENEWABLE_TECHNOLOGIES = [
        GenerationType.SOLAR_PV_UTILITY,
        GenerationType.SOLAR_PV_ROOFTOP,
        GenerationType.SOLAR_CSP,
        GenerationType.WIND_ONSHORE,
        GenerationType.WIND_OFFSHORE,
        GenerationType.HYDRO_RUN_OF_RIVER,
        GenerationType.HYDRO_RESERVOIR,
        GenerationType.GEOTHERMAL,
    ]

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Renewable Generation MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-003",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute renewable generation MRV calculation.

        Args:
            inputs: Dictionary with RenewableGenerationInput fields

        Returns:
            Dictionary with RenewableGenerationOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = RenewableGenerationInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        calculation_trace.append(
            f"Processing {validated_input.technology.value} asset "
            f"{validated_input.asset_id}"
        )

        # Calculate gross and net generation
        gross_generation = (
            validated_input.net_generation_mwh +
            validated_input.auxiliary_consumption_mwh +
            validated_input.curtailment_mwh
        )

        calculation_trace.append(
            f"Gross generation: {validated_input.net_generation_mwh:.2f} + "
            f"{validated_input.auxiliary_consumption_mwh:.2f} + "
            f"{validated_input.curtailment_mwh:.2f} = {gross_generation:.2f} MWh"
        )

        # Calculate capacity factor
        hours_in_period = 8760  # Full year default
        theoretical_generation = validated_input.installed_capacity_mw * hours_in_period
        capacity_factor = gross_generation / theoretical_generation if theoretical_generation > 0 else 0

        calculation_trace.append(
            f"Capacity factor: {gross_generation:.2f} / "
            f"({validated_input.installed_capacity_mw:.2f} * {hours_in_period}) = "
            f"{capacity_factor:.3f}"
        )

        # Get grid emission factor for avoided emissions
        grid_key = validated_input.grid_region.value
        grid_factor = self.get_emission_factor("grid_kg_mwh", grid_key)
        calculation_trace.append(f"Grid factor ({grid_key}): {grid_factor:.2f} kg/MWh")

        # Calculate avoided emissions
        avoided_tonnes = validated_input.net_generation_mwh * grid_factor / 1000
        calculation_trace.append(
            f"Avoided emissions: {validated_input.net_generation_mwh:.2f} * "
            f"{grid_factor:.2f} / 1000 = {avoided_tonnes:.2f} tonnes CO2e"
        )

        # Get lifecycle emission factor
        tech_key = validated_input.technology.value
        try:
            lifecycle_factor = self.get_emission_factor("lifecycle_g_kwh", tech_key)
        except KeyError:
            lifecycle_factor = 20.0  # Default for unlisted technologies

        calculation_trace.append(
            f"Lifecycle factor ({tech_key}): {lifecycle_factor:.2f} g/kWh"
        )

        # Calculate lifecycle emissions
        lifecycle_tonnes = (
            validated_input.net_generation_mwh * 1000 *  # MWh to kWh
            lifecycle_factor / 1_000_000  # g to tonnes
        )
        lifecycle_intensity = lifecycle_factor

        calculation_trace.append(
            f"Lifecycle emissions: {validated_input.net_generation_mwh * 1000:.2f} kWh * "
            f"{lifecycle_factor:.2f} g/kWh / 1e6 = {lifecycle_tonnes:.4f} tonnes"
        )

        # REC eligibility (all net generation for recognized technologies)
        rec_eligible = validated_input.net_generation_mwh
        if validated_input.technology not in self.RENEWABLE_TECHNOLOGIES:
            rec_eligible = 0.0
            calculation_trace.append(
                f"Technology {tech_key} not eligible for RECs"
            )
        else:
            calculation_trace.append(
                f"REC eligible: {rec_eligible:.2f} MWh"
            )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "lifecycle"
        )

        # Warnings
        warnings = []
        if capacity_factor > 0.5 and "solar" in tech_key:
            warnings.append(
                f"High solar capacity factor ({capacity_factor:.1%}) - verify data"
            )
        if capacity_factor < 0.15 and "wind" in tech_key:
            warnings.append(
                f"Low wind capacity factor ({capacity_factor:.1%}) - may indicate issues"
            )

        output = {
            "facility_id": validated_input.facility_id,
            "asset_id": validated_input.asset_id,
            "technology": validated_input.technology.value,
            "gross_generation_mwh": round(gross_generation, 2),
            "net_generation_mwh": round(validated_input.net_generation_mwh, 2),
            "capacity_factor": round(capacity_factor, 4),
            "avoided_co2e_tonnes": round(avoided_tonnes, 2),
            "avoided_emission_factor_kg_mwh": round(grid_factor, 2),
            "lifecycle_co2e_tonnes": round(lifecycle_tonnes, 4),
            "lifecycle_intensity_g_kwh": round(lifecycle_intensity, 2),
            "rec_eligible_mwh": round(rec_eligible, 2),
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output
