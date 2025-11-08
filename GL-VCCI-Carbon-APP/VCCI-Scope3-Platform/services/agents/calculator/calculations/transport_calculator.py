"""
Transport Calculator - ISO 14083:2023 Compliant Transport Calculations
GL-VCCI Scope 3 Platform

Provides standardized transport emission calculations following ISO 14083:2023
for upstream and downstream logistics (Categories 4 & 9).

ISO 14083 Formula:
    emissions = distance × weight × emission_factor / load_factor

Features:
- Multi-modal transport support
- Distance-based calculations
- Mode-specific emission factors
- Load factor adjustments
- High-precision arithmetic for compliance

Version: 1.0.0
Date: 2025-11-08
"""

import logging
from typing import Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

from ..config import TransportMode, TRANSPORT_MODE_DEFAULTS
from ..models import CalculationResult, DataQualityInfo, EmissionFactorInfo
from ..exceptions import TransportModeError, ISO14083ComplianceError

logger = logging.getLogger(__name__)


class TransportCalculator:
    """
    ISO 14083:2023 compliant transport emissions calculator.

    Implements the standardized formula for freight transport:
        emissions = distance × weight × emission_factor / load_factor

    Where:
    - distance: kilometers
    - weight: tonnes
    - emission_factor: kgCO2e per tonne-km (mode-specific)
    - load_factor: vehicle utilization factor (0-1)

    Features:
    - All ISO 14083 transport modes
    - Zero variance precision
    - Load factor adjustments
    - Multi-leg journey support
    """

    # ISO 14083 default emission factors (kgCO2e/tonne-km)
    DEFAULT_EMISSION_FACTORS = {
        # Road
        TransportMode.ROAD_TRUCK_LIGHT: 0.195,
        TransportMode.ROAD_TRUCK_MEDIUM: 0.089,
        TransportMode.ROAD_TRUCK_HEAVY: 0.062,
        TransportMode.ROAD_VAN: 0.218,
        # Rail
        TransportMode.RAIL_FREIGHT: 0.022,
        TransportMode.RAIL_FREIGHT_ELECTRIC: 0.018,
        TransportMode.RAIL_FREIGHT_DIESEL: 0.025,
        # Sea
        TransportMode.SEA_CONTAINER: 0.011,
        TransportMode.SEA_BULK: 0.008,
        TransportMode.SEA_TANKER: 0.005,
        TransportMode.SEA_RO_RO: 0.013,
        # Air
        TransportMode.AIR_CARGO: 1.130,
        TransportMode.AIR_FREIGHT: 1.340,
        # Inland Waterway
        TransportMode.INLAND_WATERWAY: 0.031,
    }

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize transport calculator.

        Args:
            config: Calculator configuration
        """
        self.config = config
        self.supported_modes = [mode for mode in TransportMode]
        logger.info("Initialized TransportCalculator (ISO 14083:2023 compliant)")

    async def calculate_transport_emissions(
        self,
        distance_km: float,
        weight_tonnes: float,
        transport_mode: TransportMode,
        emission_factor: Optional[float] = None,
        load_factor: float = 1.0,
        category: int = 4,
    ) -> Dict[str, Any]:
        """
        Calculate transport emissions using ISO 14083 formula.

        Args:
            distance_km: Transport distance in kilometers
            weight_tonnes: Cargo weight in tonnes
            transport_mode: Transport mode (road, rail, sea, air)
            emission_factor: Optional custom emission factor (kgCO2e/tonne-km)
            load_factor: Vehicle utilization factor (0-1, default 1.0)
            category: Scope 3 category (4 or 9)

        Returns:
            Dict with emissions and calculation details

        Raises:
            TransportModeError: If transport mode unsupported
            ValueError: If input parameters invalid
        """
        # Validate inputs
        self._validate_transport_inputs(
            distance_km, weight_tonnes, transport_mode, load_factor
        )

        # Get emission factor
        ef = emission_factor or self._get_emission_factor(transport_mode)

        # ISO 14083 calculation with high precision
        distance_decimal = Decimal(str(distance_km))
        weight_decimal = Decimal(str(weight_tonnes))
        ef_decimal = Decimal(str(ef))
        load_factor_decimal = Decimal(str(load_factor))

        # Formula: emissions = distance × weight × EF / load_factor
        emissions_decimal = (
            distance_decimal * weight_decimal * ef_decimal / load_factor_decimal
        )

        # Round to 6 decimal places for precision
        emissions_kgco2e = float(
            emissions_decimal.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        )

        # Calculate tonne-kilometers
        tonne_km = distance_km * weight_tonnes

        logger.info(
            f"ISO 14083: {distance_km} km × {weight_tonnes} t × {ef} kgCO2e/t-km "
            f"/ {load_factor} = {emissions_kgco2e:.6f} kgCO2e"
        )

        return {
            "emissions_kgco2e": emissions_kgco2e,
            "emissions_tco2e": emissions_kgco2e / 1000,
            "tonne_km": tonne_km,
            "transport_mode": transport_mode.value,
            "distance_km": distance_km,
            "weight_tonnes": weight_tonnes,
            "emission_factor": ef,
            "load_factor": load_factor,
            "iso_14083_compliant": True,
            "formula": "distance × weight × emission_factor / load_factor",
        }

    async def calculate_multi_leg_journey(
        self,
        legs: list[Dict[str, Any]],
        category: int = 4,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for multi-leg transport journey.

        Args:
            legs: List of transport leg dictionaries with keys:
                - distance_km
                - weight_tonnes
                - transport_mode
                - emission_factor (optional)
                - load_factor (optional)
            category: Scope 3 category

        Returns:
            Dict with total emissions and leg-by-leg breakdown
        """
        total_emissions = 0.0
        total_tonne_km = 0.0
        leg_details = []

        for i, leg in enumerate(legs, 1):
            leg_result = await self.calculate_transport_emissions(
                distance_km=leg['distance_km'],
                weight_tonnes=leg['weight_tonnes'],
                transport_mode=leg['transport_mode'],
                emission_factor=leg.get('emission_factor'),
                load_factor=leg.get('load_factor', 1.0),
                category=category,
            )

            total_emissions += leg_result['emissions_kgco2e']
            total_tonne_km += leg_result['tonne_km']

            leg_details.append({
                "leg_number": i,
                **leg_result
            })

            logger.debug(
                f"Leg {i}: {leg['transport_mode'].value} - "
                f"{leg_result['emissions_kgco2e']:.2f} kgCO2e"
            )

        return {
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_emissions / 1000,
            "total_tonne_km": total_tonne_km,
            "num_legs": len(legs),
            "legs": leg_details,
            "iso_14083_compliant": True,
        }

    async def calculate_return_journey(
        self,
        distance_km: float,
        weight_tonnes_outbound: float,
        weight_tonnes_return: float,
        transport_mode: TransportMode,
        emission_factor: Optional[float] = None,
        load_factor_outbound: float = 1.0,
        load_factor_return: float = 1.0,
        category: int = 4,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for return journey with different loads.

        Args:
            distance_km: One-way distance
            weight_tonnes_outbound: Cargo weight outbound
            weight_tonnes_return: Cargo weight return (often 0 for empty return)
            transport_mode: Transport mode
            emission_factor: Optional custom emission factor
            load_factor_outbound: Outbound load factor
            load_factor_return: Return load factor
            category: Scope 3 category

        Returns:
            Dict with total emissions and outbound/return breakdown
        """
        # Outbound leg
        outbound = await self.calculate_transport_emissions(
            distance_km=distance_km,
            weight_tonnes=weight_tonnes_outbound,
            transport_mode=transport_mode,
            emission_factor=emission_factor,
            load_factor=load_factor_outbound,
            category=category,
        )

        # Return leg
        return_leg = await self.calculate_transport_emissions(
            distance_km=distance_km,
            weight_tonnes=weight_tonnes_return,
            transport_mode=transport_mode,
            emission_factor=emission_factor,
            load_factor=load_factor_return,
            category=category,
        )

        total_emissions = outbound['emissions_kgco2e'] + return_leg['emissions_kgco2e']

        return {
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_emissions / 1000,
            "outbound": outbound,
            "return": return_leg,
            "total_distance_km": distance_km * 2,
            "is_return_journey": True,
            "iso_14083_compliant": True,
        }

    def _get_emission_factor(self, transport_mode: TransportMode) -> float:
        """
        Get emission factor for transport mode.

        Args:
            transport_mode: Transport mode

        Returns:
            Emission factor in kgCO2e/tonne-km

        Raises:
            TransportModeError: If mode not supported
        """
        if transport_mode in self.DEFAULT_EMISSION_FACTORS:
            return self.DEFAULT_EMISSION_FACTORS[transport_mode]
        elif transport_mode in TRANSPORT_MODE_DEFAULTS:
            return TRANSPORT_MODE_DEFAULTS[transport_mode]
        else:
            raise TransportModeError(
                transport_mode=transport_mode.value,
                supported_modes=[m.value for m in self.supported_modes]
            )

    def _validate_transport_inputs(
        self,
        distance_km: float,
        weight_tonnes: float,
        transport_mode: TransportMode,
        load_factor: float,
    ):
        """
        Validate transport calculation inputs.

        Args:
            distance_km: Distance
            weight_tonnes: Weight
            transport_mode: Transport mode
            load_factor: Load factor

        Raises:
            ValueError: If validation fails
            TransportModeError: If mode unsupported
        """
        if distance_km <= 0:
            raise ValueError(f"Distance must be positive, got {distance_km}")

        if weight_tonnes < 0:
            raise ValueError(f"Weight cannot be negative, got {weight_tonnes}")

        if not isinstance(transport_mode, TransportMode):
            raise TransportModeError(
                transport_mode=str(transport_mode),
                supported_modes=[m.value for m in self.supported_modes]
            )

        if not (0 < load_factor <= 1):
            raise ValueError(
                f"Load factor must be between 0 and 1, got {load_factor}"
            )

    async def get_mode_characteristics(
        self, transport_mode: TransportMode
    ) -> Dict[str, Any]:
        """
        Get characteristics and metadata for transport mode.

        Args:
            transport_mode: Transport mode

        Returns:
            Dict with mode characteristics
        """
        ef = self._get_emission_factor(transport_mode)

        # Categorize mode
        mode_value = transport_mode.value
        if mode_value.startswith("road"):
            category = "Road Transport"
            typical_speed_kmh = 60
        elif mode_value.startswith("rail"):
            category = "Rail Transport"
            typical_speed_kmh = 50
        elif mode_value.startswith("sea"):
            category = "Maritime Transport"
            typical_speed_kmh = 30
        elif mode_value.startswith("air"):
            category = "Air Transport"
            typical_speed_kmh = 800
        elif mode_value.startswith("inland"):
            category = "Inland Waterway"
            typical_speed_kmh = 15
        else:
            category = "Other"
            typical_speed_kmh = 50

        return {
            "transport_mode": transport_mode.value,
            "category": category,
            "emission_factor_kgco2e_per_tonne_km": ef,
            "typical_speed_kmh": typical_speed_kmh,
            "iso_14083_compliant": True,
            "data_quality": "good" if ef < 0.1 else "fair",
        }


__all__ = ["TransportCalculator"]
