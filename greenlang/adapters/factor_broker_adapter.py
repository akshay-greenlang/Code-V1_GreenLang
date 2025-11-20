"""
FactorBroker Adapter for EmissionFactorClient

This adapter provides backward compatibility between the new EmissionFactorClient SDK
and existing code that uses the FactorBroker pattern (e.g., VCCI Scope 3 Platform).

This enables zero-downtime migration to the database-backed emission factor system.

Example:
    >>> from greenlang.sdk.emission_factor_client import EmissionFactorClient
    >>> from greenlang.adapters.factor_broker_adapter import FactorBrokerAdapter
    >>>
    >>> ef_client = EmissionFactorClient()
    >>> broker = FactorBrokerAdapter(ef_client)
    >>>
    >>> # Use with existing code expecting FactorBroker interface
    >>> factor = broker.get_factor("diesel_fuel", unit="gallons")
    >>> emissions = broker.calculate("diesel_fuel", 100.0, "gallons")
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import EmissionFactorClient SDK
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.sdk.emission_factor_client import (
    EmissionFactorClient,
    EmissionFactorNotFoundError,
    UnitNotAvailableError,
    DatabaseConnectionError
)
from greenlang.models.emission_factor import EmissionFactor, EmissionResult

logger = logging.getLogger(__name__)


class FactorBrokerAdapter:
    """
    Adapter that wraps EmissionFactorClient to provide FactorBroker interface.

    This adapter enables existing code to use the new database-backed emission
    factor system without modification.

    Features:
    - Drop-in replacement for FactorBroker
    - Geographic fallback logic
    - Unit-aware calculations
    - Complete audit trails
    - Caching for performance

    Example:
        >>> broker = FactorBrokerAdapter()
        >>> factor = broker.get_factor("natural_gas", region="US")
        >>> result = broker.calculate("diesel", 100, "gallons")
    """

    def __init__(
        self,
        ef_client: Optional[EmissionFactorClient] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize FactorBrokerAdapter.

        Args:
            ef_client: EmissionFactorClient instance (if None, creates new one)
            db_path: Path to emission factors database (if ef_client not provided)
        """
        if ef_client is None:
            try:
                self.ef_client = EmissionFactorClient(db_path=db_path)
                logger.info("FactorBrokerAdapter initialized with new EmissionFactorClient")
            except DatabaseConnectionError as e:
                logger.error(f"Failed to initialize EmissionFactorClient: {e}")
                raise
        else:
            self.ef_client = ef_client
            logger.info("FactorBrokerAdapter initialized with provided EmissionFactorClient")

    def get_factor(
        self,
        factor_id: str,
        unit: Optional[str] = None,
        region: Optional[str] = None,
        year: Optional[int] = None
    ) -> float:
        """
        Get emission factor value.

        This method provides the FactorBroker interface for backward compatibility.

        Args:
            factor_id: Factor identifier
            unit: Optional unit (if omitted, returns base unit factor)
            region: Optional geographic region for fallback
            year: Optional year for temporal matching

        Returns:
            Emission factor value in kg CO2e per unit

        Raises:
            EmissionFactorNotFoundError: If factor not found
            UnitNotAvailableError: If unit not available

        Example:
            >>> factor_value = broker.get_factor("diesel", unit="gallons")
            10.21
        """
        try:
            # Get factor from database
            factor = self.ef_client.get_factor(
                factor_id=factor_id,
                geography=region,
                year=year
            )

            # Get value for specified unit
            if unit:
                return factor.get_factor_for_unit(unit)
            else:
                return factor.emission_factor_kg_co2e

        except EmissionFactorNotFoundError:
            # Try fuzzy matching for backward compatibility
            logger.warning(f"Factor not found: {factor_id}, attempting fuzzy match")
            return self._fuzzy_match_factor(factor_id, unit, region)

        except UnitNotAvailableError as e:
            logger.error(f"Unit not available: {e}")
            raise

    def calculate(
        self,
        factor_id: str,
        activity_amount: float,
        activity_unit: str,
        region: Optional[str] = None,
        year: Optional[int] = None
    ) -> float:
        """
        Calculate emissions (returns only the numeric value for backward compatibility).

        Args:
            factor_id: Emission factor ID
            activity_amount: Activity quantity
            activity_unit: Activity unit
            region: Optional geographic region
            year: Optional year

        Returns:
            Emissions in kg CO2e

        Example:
            >>> emissions = broker.calculate("diesel", 100.0, "gallons")
            1021.0
        """
        result = self.calculate_detailed(
            factor_id=factor_id,
            activity_amount=activity_amount,
            activity_unit=activity_unit,
            region=region,
            year=year
        )
        return result.emissions_kg_co2e

    def calculate_detailed(
        self,
        factor_id: str,
        activity_amount: float,
        activity_unit: str,
        region: Optional[str] = None,
        year: Optional[int] = None
    ) -> EmissionResult:
        """
        Calculate emissions with full result object.

        Args:
            factor_id: Emission factor ID
            activity_amount: Activity quantity
            activity_unit: Activity unit
            region: Optional geographic region
            year: Optional year

        Returns:
            EmissionResult with complete audit trail

        Example:
            >>> result = broker.calculate_detailed("diesel", 100, "gallons")
            >>> print(f"Emissions: {result.emissions_kg_co2e} kg CO2e")
            >>> print(f"Source: {result.factor_used.source.source_uri}")
        """
        return self.ef_client.calculate_emissions(
            factor_id=factor_id,
            activity_amount=activity_amount,
            activity_unit=activity_unit,
            geography=region,
            year=year
        )

    def get_grid_factor(
        self,
        region: str,
        unit: str = "kwh",
        year: Optional[int] = None
    ) -> float:
        """
        Get electricity grid emission factor.

        Args:
            region: Grid region (e.g., "CAISO", "US_NATIONAL", "California")
            unit: Unit (default: "kwh")
            year: Optional year

        Returns:
            Grid emission factor in kg CO2e per kWh

        Example:
            >>> factor = broker.get_grid_factor("CAISO")
            0.234
        """
        try:
            factor = self.ef_client.get_grid_factor(region=region, year=year)
            return factor.get_factor_for_unit(unit)
        except EmissionFactorNotFoundError:
            # Try fallback to national average
            logger.warning(f"Grid factor not found for {region}, falling back to national average")
            return self.get_grid_factor("US_NATIONAL", unit=unit, year=year)

    def get_fuel_factor(
        self,
        fuel_type: str,
        unit: Optional[str] = None
    ) -> float:
        """
        Get fuel emission factor.

        Args:
            fuel_type: Fuel type (e.g., "diesel", "natural gas", "gasoline")
            unit: Optional unit

        Returns:
            Fuel emission factor in kg CO2e per unit

        Example:
            >>> factor = broker.get_fuel_factor("diesel", "gallons")
            10.21
        """
        factor = self.ef_client.get_fuel_factor(fuel_type=fuel_type, unit=unit)
        if unit:
            return factor.get_factor_for_unit(unit)
        else:
            return factor.emission_factor_kg_co2e

    def get_transport_factor(
        self,
        mode: str,
        unit: str = "ton_km"
    ) -> float:
        """
        Get transportation emission factor.

        Args:
            mode: Transport mode (e.g., "truck", "ship", "air", "rail")
            unit: Unit (default: "ton_km")

        Returns:
            Transport emission factor in kg CO2e per ton-km

        Example:
            >>> factor = broker.get_transport_factor("truck")
            0.062
        """
        # Map common mode names to factor IDs
        mode_mapping = {
            "truck": "freight_truck_diesel",
            "ship": "ocean_freight_container",
            "ocean": "ocean_freight_container",
            "air": "air_freight",
            "rail": "rail_freight",
            "freight_truck": "freight_truck_diesel",
        }

        factor_id = mode_mapping.get(mode.lower(), f"transport_{mode}")

        try:
            factor = self.ef_client.get_factor(factor_id)
            return factor.get_factor_for_unit(unit)
        except EmissionFactorNotFoundError:
            # Try searching by name
            factors = self.ef_client.get_factor_by_name(mode)
            if factors and len(factors) > 0:
                return factors[0].get_factor_for_unit(unit)
            raise

    def _fuzzy_match_factor(
        self,
        factor_id: str,
        unit: Optional[str] = None,
        region: Optional[str] = None
    ) -> float:
        """
        Attempt fuzzy matching for factor_id.

        This provides backward compatibility for factor IDs that may have
        changed format between systems.

        Args:
            factor_id: Factor ID to match
            unit: Optional unit
            region: Optional region

        Returns:
            Matched emission factor value

        Raises:
            EmissionFactorNotFoundError: If no match found
        """
        # Try searching by name
        factors = self.ef_client.get_factor_by_name(factor_id)

        if not factors or len(factors) == 0:
            raise EmissionFactorNotFoundError(f"No factor found matching: {factor_id}")

        # Use first match (could enhance with better scoring)
        factor = factors[0]
        logger.info(f"Fuzzy matched '{factor_id}' to '{factor.factor_id}'")

        if unit:
            return factor.get_factor_for_unit(unit)
        else:
            return factor.emission_factor_kg_co2e

    def list_available_factors(
        self,
        category: Optional[str] = None,
        scope: Optional[str] = None
    ) -> List[str]:
        """
        List available emission factors.

        Args:
            category: Optional category filter
            scope: Optional scope filter

        Returns:
            List of factor IDs

        Example:
            >>> factors = broker.list_available_factors(category="fuels")
            ['fuels_diesel', 'fuels_gasoline', 'fuels_natural_gas', ...]
        """
        if category:
            factors = self.ef_client.get_by_category(category)
        elif scope:
            factors = self.ef_client.get_by_scope(scope)
        else:
            # Get all - would need to implement in EmissionFactorClient
            logger.warning("Listing all factors not yet implemented")
            return []

        return [f.factor_id for f in factors]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get emission factor database statistics.

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = broker.get_statistics()
            >>> print(f"Total factors: {stats['total_factors']}")
        """
        return self.ef_client.get_statistics()

    def close(self):
        """Close database connection."""
        if self.ef_client:
            self.ef_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for quick initialization
def create_factor_broker(db_path: Optional[str] = None) -> FactorBrokerAdapter:
    """
    Create a FactorBrokerAdapter instance.

    Args:
        db_path: Optional path to emission factors database

    Returns:
        FactorBrokerAdapter instance

    Example:
        >>> broker = create_factor_broker()
        >>> factor = broker.get_factor("diesel", unit="gallons")
    """
    return FactorBrokerAdapter(db_path=db_path)
