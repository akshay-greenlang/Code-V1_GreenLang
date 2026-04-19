"""
Category 4: Upstream Transportation and Distribution Agent

Calculates emissions from transportation and distribution of products
purchased by the reporting company in the reporting year between
tier 1 suppliers and the reporting company's operations.

GHG Protocol Reference:
- Chapter 5.4 of Scope 3 Standard
- Category 4 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result, Scope3CalculationStep


class TransportMode(Enum):
    """Transportation modes with different emission factors."""
    TRUCK = "truck"
    RAIL = "rail"
    SHIP = "ship"
    AIR = "air"
    BARGE = "barge"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"


class UpstreamTransportInput(Scope3InputData):
    """Input data for Category 4: Upstream Transportation and Distribution."""

    calculation_method: str = Field(
        ...,
        description="Method: distance-based, spend-based, or fuel-based"
    )

    # Distance-based method
    shipments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of shipments with distance, weight, and mode"
    )

    # Spend-based method
    transport_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Transportation spend by mode or category"
    )

    # Fuel-based method
    transport_fuel: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Fuel consumed by transport providers"
    )

    # Third-party logistics
    logistics_providers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Third-party logistics provider data"
    )

    # Warehousing and storage
    warehousing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Warehousing and storage operations data"
    )

    include_warehousing: bool = Field(
        default=True,
        description="Include warehousing emissions"
    )

    include_empty_returns: bool = Field(
        default=True,
        description="Include empty return trips"
    )


class UpstreamTransportAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 4: Upstream Transportation and Distribution.

    Includes:
    - Transportation of purchased products from tier 1 suppliers to own facilities
    - Third-party transportation and distribution services
    - Inbound logistics
    - Warehousing of purchased products

    Excludes:
    - Outbound logistics (Category 9)
    - Transportation of raw materials by tier 1 suppliers (Category 1)
    """

    def __init__(self):
        """Initialize Upstream Transportation agent."""
        super().__init__()
        self.category_number = 4
        self.category_name = "Upstream Transportation and Distribution"

        # Load transport emission factors
        self.transport_factors = self._load_transport_factors()
        self.warehousing_factors = self._load_warehousing_factors()

    def _load_transport_factors(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Load transportation emission factors by mode.

        Sources: EPA SmartWay, DEFRA, GLEC Framework
        """
        return {
            # Distance-based factors (kg CO2e per tonne-km)
            "truck": {
                "small": Decimal("0.205"),  # <7.5 tonnes
                "medium": Decimal("0.133"),  # 7.5-17 tonnes
                "large": Decimal("0.096"),  # >17 tonnes
                "average": Decimal("0.145")
            },
            "rail": {
                "diesel": Decimal("0.028"),
                "electric": Decimal("0.019"),
                "average": Decimal("0.024")
            },
            "ship": {
                "container": Decimal("0.016"),
                "bulk": Decimal("0.008"),
                "tanker": Decimal("0.012"),
                "average": Decimal("0.012")
            },
            "air": {
                "domestic": Decimal("1.233"),
                "short_haul": Decimal("1.516"),
                "long_haul": Decimal("0.602"),
                "average": Decimal("0.809")
            },
            "barge": {
                "inland": Decimal("0.031"),
                "average": Decimal("0.031")
            },

            # Fuel-based factors (kg CO2e per liter)
            "diesel": Decimal("2.68"),
            "gasoline": Decimal("2.31"),
            "marine_fuel": Decimal("3.17"),
            "aviation_fuel": Decimal("2.52"),

            # Spend-based factors (kg CO2e per USD)
            "transport_spend": {
                "truck": Decimal("1.105"),
                "rail": Decimal("0.289"),
                "water": Decimal("0.168"),
                "air": Decimal("2.153"),
                "average": Decimal("0.748")
            }
        }

    def _load_warehousing_factors(self) -> Dict[str, Decimal]:
        """Load warehousing emission factors."""
        return {
            # kg CO2e per m2 per year
            "ambient": Decimal("35"),
            "refrigerated": Decimal("120"),
            "frozen": Decimal("180"),

            # kg CO2e per pallet-day
            "pallet_day": Decimal("0.25"),

            # kg CO2e per tonne-day
            "tonne_day": Decimal("0.15")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> UpstreamTransportInput:
        """Parse and validate input data."""
        return UpstreamTransportInput(**input_data)

    async def calculate_emissions(self, input_data: UpstreamTransportInput) -> Scope3Result:
        """
        Calculate Category 4 emissions.

        Formulas:
        Distance-based: Σ(Distance × Weight × Mode_Factor)
        Spend-based: Σ(Spend × Spend_Factor)
        Fuel-based: Σ(Fuel × Fuel_Factor)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Calculate transport emissions
        if input_data.calculation_method == "distance-based":
            transport_emissions = self._calculate_distance_based(input_data)
        elif input_data.calculation_method == "spend-based":
            transport_emissions = self._calculate_spend_based(input_data)
        elif input_data.calculation_method == "fuel-based":
            transport_emissions = self._calculate_fuel_based(input_data)
        else:
            raise ValueError(f"Invalid calculation method: {input_data.calculation_method}")

        total_emissions += transport_emissions

        # Add warehousing emissions if applicable
        if input_data.include_warehousing and input_data.warehousing:
            warehousing_emissions = self._calculate_warehousing(input_data)
            total_emissions += warehousing_emissions

        # Calculate data quality score
        method_quality = {
            "fuel-based": 2.0,
            "distance-based": 2.5,
            "spend-based": 4.0
        }

        data_quality = self._calculate_data_quality_score(
            temporal_correlation=2.0 if input_data.reporting_year == 2024 else 3.0,
            geographical_correlation=2.5,
            technological_correlation=2.5,
            completeness=3.0,
            reliability=method_quality.get(input_data.calculation_method, 3.0)
        )

        # Generate result
        result = Scope3Result(
            category="Category 4: Upstream Transportation and Distribution",
            category_number=4,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_t_co2e=total_emissions / Decimal("1000"),
            calculation_methodology=input_data.calculation_method,
            data_quality_score=data_quality,
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=self._calculate_provenance_hash(
                input_data.dict(),
                self.calculation_steps,
                total_emissions
            ),
            calculation_timestamp=str(self._get_timestamp()),
            ghg_protocol_compliance=True,
            uncertainty_range=self._estimate_uncertainty(
                data_quality,
                input_data.calculation_method
            )
        )

        return result

    def _calculate_distance_based(self, input_data: UpstreamTransportInput) -> Decimal:
        """
        Calculate emissions using distance-based method.

        Most accurate for transportation emissions.
        Formula: Emissions = Distance × Weight × Mode_Factor
        """
        total = Decimal("0")

        if not input_data.shipments:
            raise ValueError("shipments data required for distance-based method")

        for shipment in input_data.shipments:
            distance_km = Decimal(str(shipment.get("distance_km", 0)))
            weight_tonnes = Decimal(str(shipment.get("weight_tonnes", 0)))
            mode = shipment.get("mode", "truck")
            vehicle_type = shipment.get("vehicle_type", "average")

            # Get emission factor for mode
            mode_factors = self.transport_factors.get(mode, self.transport_factors["truck"])
            ef = mode_factors.get(vehicle_type, mode_factors.get("average"))

            # Calculate tonne-km
            tonne_km = distance_km * weight_tonnes

            # Apply empty return factor if applicable
            if input_data.include_empty_returns:
                empty_return_factor = Decimal(str(shipment.get("empty_return_factor", 1.3)))
                tonne_km *= empty_return_factor

            # Calculate emissions
            emissions = tonne_km * ef

            self._record_calculation_step(
                description=f"Calculate emissions for {mode} shipment",
                operation="multiply",
                inputs={
                    "distance_km": distance_km,
                    "weight_tonnes": weight_tonnes,
                    "emission_factor": ef,
                    "mode": mode
                },
                output_value=emissions,
                output_name=f"emissions_shipment_{shipment.get('id', 'unknown')}",
                formula="Emissions = Distance × Weight × Mode_Factor",
                unit="kg CO2e"
            )

            self.factors_used[f"{mode}_{vehicle_type}"] = {
                "value": float(ef),
                "unit": "kg CO2e/tonne-km",
                "source": "GLEC Framework 2024",
                "mode": mode,
                "vehicle_type": vehicle_type
            }

            total += emissions

        return total

    def _calculate_spend_based(self, input_data: UpstreamTransportInput) -> Decimal:
        """
        Calculate emissions using spend-based method.

        Less accurate but easier when only financial data available.
        """
        total = Decimal("0")

        if not input_data.transport_spend:
            raise ValueError("transport_spend required for spend-based method")

        spend_factors = self.transport_factors["transport_spend"]

        for category, spend in input_data.transport_spend.items():
            spend_decimal = Decimal(str(spend))

            # Get spend-based emission factor
            ef = spend_factors.get(category, spend_factors["average"])

            # Calculate emissions
            emissions = spend_decimal * ef

            self._record_calculation_step(
                description=f"Calculate emissions for {category} transport spend",
                operation="multiply",
                inputs={
                    "spend_usd": spend,
                    "emission_factor": ef
                },
                output_value=emissions,
                output_name=f"emissions_spend_{category}",
                formula="Emissions = Spend × Spend_Factor",
                unit="kg CO2e"
            )

            self.factors_used[f"spend_{category}"] = {
                "value": float(ef),
                "unit": "kg CO2e/USD",
                "source": "EPA EEIO 2024",
                "category": category
            }

            total += emissions

        return total

    def _calculate_fuel_based(self, input_data: UpstreamTransportInput) -> Decimal:
        """
        Calculate emissions using fuel-based method.

        Most accurate when fuel consumption data is available.
        """
        total = Decimal("0")

        if not input_data.transport_fuel:
            raise ValueError("transport_fuel required for fuel-based method")

        for fuel_type, fuel_data in input_data.transport_fuel.items():
            quantity = Decimal(str(fuel_data.get("quantity", 0)))
            unit = fuel_data.get("unit", "liter")

            # Get fuel emission factor
            ef = self.transport_factors.get(fuel_type, Decimal("2.5"))

            # Convert to liters if needed
            if unit == "gallon":
                quantity *= Decimal("3.78541")  # gallons to liters

            # Calculate emissions
            emissions = quantity * ef

            self._record_calculation_step(
                description=f"Calculate emissions for {fuel_type} consumption",
                operation="multiply",
                inputs={
                    "fuel_quantity": quantity,
                    "unit": "liter",
                    "emission_factor": ef
                },
                output_value=emissions,
                output_name=f"emissions_fuel_{fuel_type}",
                formula="Emissions = Fuel × Fuel_Factor",
                unit="kg CO2e"
            )

            self.factors_used[f"fuel_{fuel_type}"] = {
                "value": float(ef),
                "unit": "kg CO2e/liter",
                "source": "IPCC 2024",
                "fuel_type": fuel_type
            }

            total += emissions

        return total

    def _calculate_warehousing(self, input_data: UpstreamTransportInput) -> Decimal:
        """Calculate emissions from warehousing and storage."""
        total = Decimal("0")
        warehouse_data = input_data.warehousing

        # Method 1: Area-based
        if "area_m2" in warehouse_data:
            area = Decimal(str(warehouse_data["area_m2"]))
            storage_type = warehouse_data.get("type", "ambient")
            occupancy_rate = Decimal(str(warehouse_data.get("occupancy_rate", 0.7)))
            days = Decimal(str(warehouse_data.get("days", 365)))

            ef = self.warehousing_factors[storage_type]
            annual_emissions = area * ef * occupancy_rate
            emissions = (annual_emissions * days) / Decimal("365")

            self._record_calculation_step(
                description=f"Calculate {storage_type} warehousing emissions",
                operation="multiply",
                inputs={
                    "area_m2": area,
                    "days": days,
                    "emission_factor": ef,
                    "occupancy_rate": occupancy_rate
                },
                output_value=emissions,
                output_name="warehousing_emissions",
                formula="Emissions = Area × EF × Occupancy × (Days/365)",
                unit="kg CO2e"
            )

            total += emissions

        # Method 2: Throughput-based
        elif "throughput_tonnes" in warehouse_data:
            throughput = Decimal(str(warehouse_data["throughput_tonnes"]))
            avg_storage_days = Decimal(str(warehouse_data.get("avg_storage_days", 7)))

            ef = self.warehousing_factors["tonne_day"]
            emissions = throughput * avg_storage_days * ef

            self._record_calculation_step(
                description="Calculate warehousing emissions by throughput",
                operation="multiply",
                inputs={
                    "throughput_tonnes": throughput,
                    "storage_days": avg_storage_days,
                    "emission_factor": ef
                },
                output_value=emissions,
                output_name="warehousing_emissions",
                formula="Emissions = Throughput × Days × EF",
                unit="kg CO2e"
            )

            total += emissions

        return total

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()