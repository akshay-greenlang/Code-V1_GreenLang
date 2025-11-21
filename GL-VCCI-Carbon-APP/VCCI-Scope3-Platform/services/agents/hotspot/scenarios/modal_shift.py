# -*- coding: utf-8 -*-
"""
Modal Shift Scenario Module
GL-VCCI Scope 3 Platform

Stub implementation for transport modal shift scenarios.
Full implementation planned for Week 27+ with route optimization.

Version: 1.0.0 (Stub)
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import Dict, Any, List, Optional

from ..models import ModalShiftScenario

logger = logging.getLogger(__name__)


class ModalShiftModule:
    """
    Transport modal shift scenario analyzer.

    STUB IMPLEMENTATION: Framework only.
    Full implementation in Week 27+ will include:
    - Route-specific emission factors (ISO 14083)
    - Multi-modal optimization
    - Cost-time-emission trade-off analysis
    - Capacity and infrastructure constraints
    """

    def __init__(self):
        """Initialize modal shift module."""
        logger.info("Initialized ModalShiftModule (Stub v1.0)")

    def suggest_modal_shifts(
        self,
        current_mode: str,
        routes: List[str],
        baseline_emissions: float
    ) -> List[Dict[str, Any]]:
        """
        Suggest modal shift opportunities.

        STUB: Returns placeholder structure.
        Full implementation will calculate route-specific emission factors.

        Args:
            current_mode: Current transport mode
            routes: Routes to analyze
            baseline_emissions: Current emissions (tCO2e)

        Returns:
            List of modal shift opportunities
        """
        logger.info(
            f"Suggesting modal shifts from {current_mode} for {len(routes)} routes (STUB)"
        )

        # STUB: Return placeholder structure
        opportunities = []

        # Air to sea
        if current_mode.lower() == "air":
            opportunities.append({
                "from_mode": "air",
                "to_mode": "sea",
                "reduction_potential_tco2e": baseline_emissions * 0.6,
                "cost_delta_pct": -15.0,  # Negative = savings
                "time_delta_days": +14,
                "feasibility": "High",
                "notes": "Stub data - full implementation in Week 27+"
            })

        # Road to rail
        if current_mode.lower() == "road":
            opportunities.append({
                "from_mode": "road",
                "to_mode": "rail",
                "reduction_potential_tco2e": baseline_emissions * 0.4,
                "cost_delta_pct": -5.0,
                "time_delta_days": +3,
                "feasibility": "Medium",
                "notes": "Stub data - full implementation in Week 27+"
            })

        return opportunities

    def calculate_modal_emissions(
        self,
        mode: str,
        distance_km: float,
        weight_tonnes: float
    ) -> Dict[str, Any]:
        """
        Calculate emissions for specific transport mode.

        STUB: Uses simplified emission factors.
        Full implementation will use ISO 14083 methodology.

        Args:
            mode: Transport mode
            distance_km: Distance
            weight_tonnes: Weight

        Returns:
            Emission calculation
        """
        # STUB: Simplified emission factors (kgCO2e per tonne-km)
        emission_factors = {
            "air": 0.500,
            "road": 0.062,
            "rail": 0.022,
            "sea": 0.010
        }

        ef = emission_factors.get(mode.lower(), 0.062)
        emissions_kgco2e = ef * distance_km * weight_tonnes

        return {
            "mode": mode,
            "distance_km": distance_km,
            "weight_tonnes": weight_tonnes,
            "emission_factor": ef,
            "emissions_kgco2e": emissions_kgco2e,
            "emissions_tco2e": emissions_kgco2e / 1000,
            "notes": "Stub calculation - full ISO 14083 in Week 27+"
        }


__all__ = ["ModalShiftModule"]
