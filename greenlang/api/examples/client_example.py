"""
GreenLang API Client Example

Demonstrates how to use the GreenLang Emission Factor API from Python.
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime


class GreenLangAPIClient:
    """
    Python client for GreenLang Emission Factor API.

    Example:
        >>> client = GreenLangAPIClient("http://localhost:8000")
        >>> result = client.calculate_emissions("diesel", 100, "gallons")
        >>> print(f"Emissions: {result['emissions_kg_co2e']} kg CO2e")
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            base_url: API base URL (default: http://localhost:8000)
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def list_factors(
        self,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        page: int = 1,
        limit: int = 100
    ) -> Dict:
        """
        List emission factors.

        Args:
            fuel_type: Filter by fuel type
            geography: Filter by geography
            scope: Filter by scope
            page: Page number
            limit: Results per page

        Returns:
            Dictionary with factors list and pagination info
        """
        params = {"page": page, "limit": limit}
        if fuel_type:
            params["fuel_type"] = fuel_type
        if geography:
            params["geography"] = geography
        if scope:
            params["scope"] = scope

        response = self.session.get(f"{self.base_url}/api/v1/factors", params=params)
        response.raise_for_status()
        return response.json()

    def get_factor(self, factor_id: str) -> Dict:
        """
        Get specific emission factor by ID.

        Args:
            factor_id: Factor ID (e.g., "EF:US:diesel:2024:v1")

        Returns:
            Factor details dictionary
        """
        response = self.session.get(f"{self.base_url}/api/v1/factors/{factor_id}")
        response.raise_for_status()
        return response.json()

    def search_factors(self, query: str, geography: Optional[str] = None, limit: int = 20) -> Dict:
        """
        Search emission factors.

        Args:
            query: Search query
            geography: Filter by geography
            limit: Maximum results

        Returns:
            Search results dictionary
        """
        params = {"q": query, "limit": limit}
        if geography:
            params["geography"] = geography

        response = self.session.get(f"{self.base_url}/api/v1/factors/search", params=params)
        response.raise_for_status()
        return response.json()

    def calculate_emissions(
        self,
        fuel_type: str,
        activity_amount: float,
        activity_unit: str,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion"
    ) -> Dict:
        """
        Calculate emissions for an activity.

        Args:
            fuel_type: Fuel type (diesel, gasoline, natural_gas, electricity, etc.)
            activity_amount: Activity amount
            activity_unit: Activity unit (gallons, kWh, therms, etc.)
            geography: Geography (US, UK, EU, etc.)
            scope: GHG scope (1, 2, or 3)
            boundary: Emission boundary (combustion, WTT, WTW)

        Returns:
            Calculation result dictionary
        """
        request = {
            "fuel_type": fuel_type,
            "activity_amount": activity_amount,
            "activity_unit": activity_unit,
            "geography": geography,
            "scope": scope,
            "boundary": boundary
        }

        response = self.session.post(f"{self.base_url}/api/v1/calculate", json=request)
        response.raise_for_status()
        return response.json()

    def calculate_batch(self, calculations: List[Dict]) -> Dict:
        """
        Calculate emissions for multiple activities.

        Args:
            calculations: List of calculation requests (max 100)

        Returns:
            Batch calculation results dictionary
        """
        request = {"calculations": calculations}

        response = self.session.post(f"{self.base_url}/api/v1/calculate/batch", json=request)
        response.raise_for_status()
        return response.json()

    def calculate_scope1(
        self,
        fuel_type: str,
        consumption: float,
        unit: str,
        geography: str = "US"
    ) -> Dict:
        """
        Calculate Scope 1 (direct) emissions.

        Args:
            fuel_type: Fuel type
            consumption: Fuel consumption
            unit: Consumption unit
            geography: Geography

        Returns:
            Emission result dictionary
        """
        request = {
            "fuel_type": fuel_type,
            "consumption": consumption,
            "unit": unit,
            "geography": geography
        }

        response = self.session.post(f"{self.base_url}/api/v1/calculate/scope1", json=request)
        response.raise_for_status()
        return response.json()

    def calculate_scope2(
        self,
        electricity_kwh: float,
        geography: str = "US",
        market_based_factor: Optional[float] = None
    ) -> Dict:
        """
        Calculate Scope 2 (purchased electricity) emissions.

        Args:
            electricity_kwh: Electricity consumption in kWh
            geography: Grid geography
            market_based_factor: Optional market-based factor (for renewable energy)

        Returns:
            Emission result dictionary
        """
        request = {
            "electricity_kwh": electricity_kwh,
            "geography": geography
        }
        if market_based_factor is not None:
            request["market_based_factor"] = market_based_factor

        response = self.session.post(f"{self.base_url}/api/v1/calculate/scope2", json=request)
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict:
        """
        Get API statistics.

        Returns:
            Statistics dictionary
        """
        response = self.session.get(f"{self.base_url}/api/v1/stats")
        response.raise_for_status()
        return response.json()

    def get_coverage_stats(self) -> Dict:
        """
        Get coverage statistics.

        Returns:
            Coverage statistics dictionary
        """
        response = self.session.get(f"{self.base_url}/api/v1/stats/coverage")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict:
        """
        Check API health.

        Returns:
            Health status dictionary
        """
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()


# ==================== USAGE EXAMPLES ====================

def example_basic_calculation():
    """Example: Basic emission calculation"""
    print("\n=== Basic Calculation Example ===")

    client = GreenLangAPIClient()

    # Calculate diesel emissions
    result = client.calculate_emissions(
        fuel_type="diesel",
        activity_amount=100,
        activity_unit="gallons",
        geography="US"
    )

    print(f"Calculation ID: {result['calculation_id']}")
    print(f"Total Emissions: {result['emissions_kg_co2e']:.2f} kg CO2e")
    print(f"                 {result['emissions_tonnes_co2e']:.4f} tonnes CO2e")
    print(f"\nEmissions by gas:")
    for gas, amount in result['emissions_by_gas'].items():
        print(f"  {gas}: {amount:.4f} kg")

    print(f"\nFactor used: {result['factor_used']['factor_id']}")
    print(f"Source: {result['factor_used']['source']} ({result['factor_used']['source_year']})")
    print(f"Data quality: {result['factor_used']['data_quality_score']}/5.0")


def example_batch_calculation():
    """Example: Batch calculation for multiple fuels"""
    print("\n=== Batch Calculation Example ===")

    client = GreenLangAPIClient()

    # Calculate emissions for multiple fuel types
    calculations = [
        {
            "fuel_type": "diesel",
            "activity_amount": 100,
            "activity_unit": "gallons",
            "geography": "US"
        },
        {
            "fuel_type": "natural_gas",
            "activity_amount": 500,
            "activity_unit": "therms",
            "geography": "US"
        },
        {
            "fuel_type": "electricity",
            "activity_amount": 10000,
            "activity_unit": "kWh",
            "geography": "US",
            "scope": "2"
        }
    ]

    result = client.calculate_batch(calculations)

    print(f"Batch ID: {result['batch_id']}")
    print(f"Total emissions: {result['total_emissions_kg_co2e']:.2f} kg CO2e")
    print(f"                 {result['total_emissions_tonnes_co2e']:.4f} tonnes CO2e")
    print(f"\nIndividual calculations:")

    for i, calc in enumerate(result['calculations'], 1):
        print(f"  {i}. {calc['factor_used']['fuel_type']}: "
              f"{calc['emissions_kg_co2e']:.2f} kg CO2e")


def example_scope_calculations():
    """Example: Scope-specific calculations"""
    print("\n=== Scope-Specific Calculations ===")

    client = GreenLangAPIClient()

    # Scope 1: Direct emissions (fuel combustion)
    scope1 = client.calculate_scope1(
        fuel_type="natural_gas",
        consumption=500,
        unit="therms",
        geography="US"
    )
    print(f"Scope 1 (natural gas): {scope1['emissions_kg_co2e']:.2f} kg CO2e")

    # Scope 2: Purchased electricity
    scope2 = client.calculate_scope2(
        electricity_kwh=10000,
        geography="US"
    )
    print(f"Scope 2 (electricity): {scope2['emissions_kg_co2e']:.2f} kg CO2e")

    # Scope 2 with renewable energy (market-based)
    scope2_renewable = client.calculate_scope2(
        electricity_kwh=10000,
        geography="US",
        market_based_factor=0.0  # 100% renewable
    )
    print(f"Scope 2 (renewable): {scope2_renewable['emissions_kg_co2e']:.2f} kg CO2e")


def example_search_factors():
    """Example: Search and list factors"""
    print("\n=== Search Factors Example ===")

    client = GreenLangAPIClient()

    # Search for diesel factors
    results = client.search_factors("diesel", geography="US")

    print(f"Found {results['count']} factors matching 'diesel' in US")
    print(f"Search time: {results['search_time_ms']:.2f}ms\n")

    for factor in results['factors'][:3]:  # Show first 3
        print(f"  {factor['factor_id']}")
        print(f"    Fuel: {factor['fuel_type']} ({factor['unit']})")
        print(f"    CO2e: {factor['co2e_per_unit']:.4f} kg/unit")
        print(f"    Scope: {factor['scope']}, Boundary: {factor['boundary']}")
        print()


def example_coverage_stats():
    """Example: Get coverage statistics"""
    print("\n=== Coverage Statistics ===")

    client = GreenLangAPIClient()

    stats = client.get_coverage_stats()

    print(f"Total factors: {stats['total_factors']}")
    print(f"Geographies covered: {stats['geographies']}")
    print(f"Fuel types: {stats['fuel_types']}")
    print(f"\nFactors by scope:")
    for scope, count in stats['scopes'].items():
        print(f"  Scope {scope}: {count} factors")

    print(f"\nTop geographies:")
    top_geos = sorted(stats['by_geography'].items(), key=lambda x: x[1], reverse=True)[:5]
    for geo, count in top_geos:
        print(f"  {geo}: {count} factors")


if __name__ == "__main__":
    """Run all examples"""

    # Check API health first
    client = GreenLangAPIClient()
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Version: {health['version']}")
    except Exception as e:
        print(f"Error: API is not available at http://localhost:8000")
        print(f"Please start the API first: uvicorn greenlang.api.main:app")
        exit(1)

    # Run examples
    example_basic_calculation()
    example_batch_calculation()
    example_scope_calculations()
    example_search_factors()
    example_coverage_stats()

    print("\n=== All examples completed successfully! ===")
