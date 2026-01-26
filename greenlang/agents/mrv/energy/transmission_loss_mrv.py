# -*- coding: utf-8 -*-
"""
GL-MRV-ENE-005: Transmission Loss MRV Agent

Calculates emissions from transmission and distribution (T&D) losses
and provides loss-adjusted emission factors for accurate Scope 2/3 accounting.

Standards Reference:
    - GHG Protocol: Scope 2/3 Guidance
    - FERC Form 714: Transmission loss data
    - IEA: T&D loss methodologies

Example:
    >>> agent = TransmissionLossMRVAgent()
    >>> result = agent.process({
    ...     "facility_id": "UTILITY-001",
    ...     "network_id": "GRID-WEST",
    ...     "voltage_level": "transmission",
    ...     "energy_injected_mwh": 1000000,
    ...     "energy_delivered_mwh": 950000,
    ...     "grid_region": "us_camx",
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.mrv.energy.base import MRVEnergyBaseAgent
from greenlang.agents.mrv.energy.schemas import (
    TransmissionLossInput,
    TransmissionLossOutput,
    UncertaintyLevel,
)


logger = logging.getLogger(__name__)


class TransmissionLossMRVAgent(MRVEnergyBaseAgent):
    """
    GL-MRV-ENE-005: Transmission Loss MRV Agent

    Calculates:
    - Technical T&D losses
    - Emissions from losses
    - Loss-adjusted emission factors

    This agent is CRITICAL PATH - all calculations are deterministic.
    """

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="GL-MRV-ENE-005",
        category=AgentCategory.CRITICAL,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Transmission and distribution loss emissions MRV"
    )

    # Typical loss percentages by voltage level
    TYPICAL_LOSSES = {
        "transmission": 2.5,  # High voltage transmission
        "subtransmission": 1.5,  # Medium voltage
        "distribution": 4.0,  # Low voltage distribution
        "combined": 6.5,  # Total T&D
    }

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Transmission Loss MRV Agent."""
        super().__init__(
            agent_id="GL-MRV-ENE-005",
            version="1.0.0",
            enable_audit_trail=enable_audit_trail
        )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transmission loss MRV calculation.

        Args:
            inputs: Dictionary with TransmissionLossInput fields

        Returns:
            Dictionary with TransmissionLossOutput fields
        """
        calculation_trace: List[str] = []

        # Validate inputs
        try:
            validated_input = TransmissionLossInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        calculation_trace.append(
            f"Processing network {validated_input.network_id}"
        )

        # Calculate total losses
        total_losses = (
            validated_input.energy_injected_mwh -
            validated_input.energy_delivered_mwh
        )
        loss_percentage = (
            (total_losses / validated_input.energy_injected_mwh) * 100
            if validated_input.energy_injected_mwh > 0 else 0
        )

        calculation_trace.append(
            f"Total losses: {validated_input.energy_injected_mwh:.2f} - "
            f"{validated_input.energy_delivered_mwh:.2f} = {total_losses:.2f} MWh "
            f"({loss_percentage:.2f}%)"
        )

        # Split technical vs non-technical losses
        # Assume 90% of losses are technical
        technical_losses = total_losses * 0.90
        non_technical_losses = total_losses * 0.10

        calculation_trace.append(
            f"Technical: {technical_losses:.2f} MWh, "
            f"Non-technical: {non_technical_losses:.2f} MWh"
        )

        # Adjust for transformer losses if provided
        if validated_input.transformer_losses_mwh is not None:
            transformer_pct = (
                validated_input.transformer_losses_mwh /
                validated_input.energy_injected_mwh * 100
                if validated_input.energy_injected_mwh > 0 else 0
            )
            calculation_trace.append(
                f"Transformer losses: {validated_input.transformer_losses_mwh:.2f} MWh "
                f"({transformer_pct:.2f}%)"
            )

        # Get grid emission factor
        grid_key = validated_input.grid_region.value
        grid_factor = self.get_emission_factor("grid_kg_mwh", grid_key)
        calculation_trace.append(
            f"Grid factor ({grid_key}): {grid_factor:.2f} kg/MWh"
        )

        # Calculate emissions from losses
        loss_emissions = total_losses * grid_factor / 1000
        calculation_trace.append(
            f"Loss emissions: {total_losses:.2f} * {grid_factor:.2f} / 1000 = "
            f"{loss_emissions:.2f} tonnes"
        )

        # Calculate loss-adjusted emission factor
        # This is the factor to apply to delivered energy
        if validated_input.energy_delivered_mwh > 0:
            loss_adjusted_factor = grid_factor * (1 + loss_percentage / 100)
        else:
            loss_adjusted_factor = grid_factor

        calculation_trace.append(
            f"Loss-adjusted factor: {grid_factor:.2f} * (1 + {loss_percentage:.2f}/100) = "
            f"{loss_adjusted_factor:.2f} kg/MWh"
        )

        # Compare to typical losses
        typical_loss = self.TYPICAL_LOSSES.get(
            validated_input.voltage_level.lower(),
            self.TYPICAL_LOSSES["combined"]
        )

        # Uncertainty
        uncertainty_pct = self.calculate_uncertainty(
            validated_input.data_quality.value,
            "grid"
        )

        # Warnings
        warnings = []
        if loss_percentage > typical_loss * 1.5:
            warnings.append(
                f"High losses ({loss_percentage:.1f}%) vs typical "
                f"({typical_loss:.1f}%) for {validated_input.voltage_level}"
            )
        if loss_percentage < 0:
            warnings.append("Negative losses detected - check meter data")

        output = {
            "facility_id": validated_input.facility_id,
            "network_id": validated_input.network_id,
            "total_losses_mwh": round(total_losses, 2),
            "loss_percentage": round(loss_percentage, 2),
            "technical_losses_mwh": round(technical_losses, 2),
            "non_technical_losses_mwh": round(non_technical_losses, 2),
            "loss_emissions_tonnes": round(loss_emissions, 2),
            "emission_factor_kg_mwh": round(grid_factor, 2),
            "loss_adjusted_factor_kg_mwh": round(loss_adjusted_factor, 2),
            "data_quality": validated_input.data_quality.value,
            "uncertainty_pct": uncertainty_pct,
            "validation_status": "PASS",
            "warnings": warnings,
            "calculation_trace": calculation_trace,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output
