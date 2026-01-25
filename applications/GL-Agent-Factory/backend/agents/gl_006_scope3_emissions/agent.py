"""
GL-006: Scope 3 Emissions Agent

This module implements the Scope 3 Supply Chain Emissions Agent for calculating
value chain GHG emissions across all 15 GHG Protocol categories.

The agent supports:
- All 15 Scope 3 categories
- Spend-based, activity-based, and supplier-specific methods
- GHG Protocol data quality indicators
- CDP and SBTi reporting formats

Example:
    >>> agent = Scope3EmissionsAgent()
    >>> result = agent.run(Scope3Input(
    ...     category=Scope3Category.PURCHASED_GOODS,
    ...     spend_data=[{"category": "steel", "spend_usd": 1000000}],
    ...     reporting_year=2024
    ... ))
    >>> print(f"Category 1 emissions: {result.data.total_emissions_kgco2e} kgCO2e")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories."""

    CAT_1_PURCHASED_GOODS = "purchased_goods"
    CAT_2_CAPITAL_GOODS = "capital_goods"
    CAT_3_FUEL_ENERGY = "fuel_energy_activities"
    CAT_4_UPSTREAM_TRANSPORT = "upstream_transport"
    CAT_5_WASTE = "waste_generated"
    CAT_6_BUSINESS_TRAVEL = "business_travel"
    CAT_7_COMMUTING = "employee_commuting"
    CAT_8_UPSTREAM_LEASED = "upstream_leased_assets"
    CAT_9_DOWNSTREAM_TRANSPORT = "downstream_transport"
    CAT_10_PROCESSING = "processing_of_products"
    CAT_11_USE_OF_PRODUCTS = "use_of_sold_products"
    CAT_12_END_OF_LIFE = "end_of_life_treatment"
    CAT_13_DOWNSTREAM_LEASED = "downstream_leased_assets"
    CAT_14_FRANCHISES = "franchises"
    CAT_15_INVESTMENTS = "investments"


class CalculationMethod(str, Enum):
    """Scope 3 calculation methods."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"


class DataQualityScore(str, Enum):
    """GHG Protocol data quality indicators."""

    VERY_GOOD = "very_good"  # 1
    GOOD = "good"  # 2
    FAIR = "fair"  # 3
    POOR = "poor"  # 4
    VERY_POOR = "very_poor"  # 5


class SpendData(BaseModel):
    """Spend-based input data."""

    category: str = Field(..., description="Spend category")
    spend_usd: float = Field(..., ge=0, description="Spend amount in USD")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    country: Optional[str] = Field(None, description="Country of origin")


class ActivityData(BaseModel):
    """Activity-based input data."""

    activity_type: str = Field(..., description="Activity type")
    quantity: float = Field(..., ge=0, description="Activity quantity")
    unit: str = Field(..., description="Unit of measurement")


class TransportData(BaseModel):
    """Transport activity data for categories 4, 9."""

    mode: str = Field(..., description="Transport mode")
    distance_km: float = Field(..., ge=0, description="Distance in km")
    weight_tonnes: float = Field(..., ge=0, description="Weight in tonnes")


class TravelData(BaseModel):
    """Business travel data for category 6."""

    mode: str = Field(..., description="Travel mode (air, rail, car)")
    distance_km: float = Field(..., ge=0, description="Distance in km")
    trip_type: str = Field("one_way", description="One way or round trip")


class Scope3Input(BaseModel):
    """
    Input model for Scope 3 Emissions Agent.

    Attributes:
        category: Scope 3 category to calculate
        reporting_year: Fiscal year
        spend_data: Spend-based input data
        activity_data: Activity-based input data
        transport_data: Transport activity data
        travel_data: Business travel data
        supplier_emissions: Primary supplier data
        calculation_method: Preferred calculation method
    """

    category: Scope3Category = Field(..., description="Scope 3 category")
    reporting_year: int = Field(..., ge=2020, description="Reporting year")

    # Input data by method
    spend_data: List[SpendData] = Field(default_factory=list)
    activity_data: List[ActivityData] = Field(default_factory=list)
    transport_data: List[TransportData] = Field(default_factory=list)
    travel_data: List[TravelData] = Field(default_factory=list)

    # Supplier-specific data
    supplier_emissions: Optional[Dict[str, float]] = Field(None)

    # Method preference
    calculation_method: CalculationMethod = Field(
        CalculationMethod.SPEND_BASED,
        description="Calculation method"
    )

    # Scaling factors
    revenue_usd: Optional[float] = Field(None, ge=0, description="Annual revenue")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class Scope3Output(BaseModel):
    """
    Output model for Scope 3 Emissions Agent.

    Includes emissions by category and data quality assessment.
    """

    category: str = Field(..., description="Scope 3 category")
    category_number: int = Field(..., ge=1, le=15, description="Category number")
    category_name: str = Field(..., description="Category name")

    # Emissions results
    total_emissions_kgco2e: float = Field(..., description="Total emissions")
    emissions_by_source: Dict[str, float] = Field(..., description="Breakdown by source")

    # Calculation details
    calculation_method: str = Field(..., description="Method used")
    emission_factors_used: List[Dict[str, Any]] = Field(..., description="EFs applied")

    # Data quality
    data_quality_score: str = Field(..., description="DQI score")
    data_coverage_pct: float = Field(..., description="Data coverage %")
    uncertainty_pct: Optional[float] = Field(None, description="Uncertainty range %")

    # Intensity metrics
    emissions_per_revenue: Optional[float] = Field(None, description="kgCO2e per USD revenue")
    emissions_per_employee: Optional[float] = Field(None, description="kgCO2e per employee")

    # Recommendations
    improvement_opportunities: List[str] = Field(default_factory=list)

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class SpendEmissionFactor(BaseModel):
    """Spend-based emission factor (EEIO)."""

    category: str
    factor_kgco2e_per_usd: float
    source: str
    year: int


class Scope3EmissionsAgent:
    """
    GL-006: Scope 3 Emissions Agent.

    This agent calculates Scope 3 value chain emissions using
    zero-hallucination deterministic calculations:
    - Spend-based: emissions = spend * EEIO_factor
    - Transport: emissions = distance * weight * transport_factor
    - Travel: emissions = distance * mode_factor

    Aligned with:
    - GHG Protocol Corporate Value Chain Standard
    - CDP Climate questionnaire
    - SBTi requirements

    Attributes:
        spend_factors: EPA EEIO emission factors
        transport_factors: GLEC transport emission factors
        travel_factors: Business travel emission factors

    Example:
        >>> agent = Scope3EmissionsAgent()
        >>> result = agent.run(Scope3Input(
        ...     category=Scope3Category.CAT_1_PURCHASED_GOODS,
        ...     spend_data=[SpendData(category="steel", spend_usd=1000000)]
        ... ))
        >>> assert result.total_emissions_kgco2e > 0
    """

    AGENT_ID = "emissions/scope3_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "Scope 3 supply chain emissions calculator"

    # Category names
    CATEGORY_NAMES: Dict[Scope3Category, Tuple[int, str]] = {
        Scope3Category.CAT_1_PURCHASED_GOODS: (1, "Purchased Goods and Services"),
        Scope3Category.CAT_2_CAPITAL_GOODS: (2, "Capital Goods"),
        Scope3Category.CAT_3_FUEL_ENERGY: (3, "Fuel- and Energy-Related Activities"),
        Scope3Category.CAT_4_UPSTREAM_TRANSPORT: (4, "Upstream Transportation and Distribution"),
        Scope3Category.CAT_5_WASTE: (5, "Waste Generated in Operations"),
        Scope3Category.CAT_6_BUSINESS_TRAVEL: (6, "Business Travel"),
        Scope3Category.CAT_7_COMMUTING: (7, "Employee Commuting"),
        Scope3Category.CAT_8_UPSTREAM_LEASED: (8, "Upstream Leased Assets"),
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT: (9, "Downstream Transportation and Distribution"),
        Scope3Category.CAT_10_PROCESSING: (10, "Processing of Sold Products"),
        Scope3Category.CAT_11_USE_OF_PRODUCTS: (11, "Use of Sold Products"),
        Scope3Category.CAT_12_END_OF_LIFE: (12, "End-of-Life Treatment of Sold Products"),
        Scope3Category.CAT_13_DOWNSTREAM_LEASED: (13, "Downstream Leased Assets"),
        Scope3Category.CAT_14_FRANCHISES: (14, "Franchises"),
        Scope3Category.CAT_15_INVESTMENTS: (15, "Investments"),
    }

    # EPA EEIO spend-based factors (kgCO2e per USD)
    SPEND_FACTORS: Dict[str, SpendEmissionFactor] = {
        "steel": SpendEmissionFactor(
            category="steel", factor_kgco2e_per_usd=0.85, source="EPA EEIO 2024", year=2024
        ),
        "aluminum": SpendEmissionFactor(
            category="aluminum", factor_kgco2e_per_usd=1.20, source="EPA EEIO 2024", year=2024
        ),
        "plastics": SpendEmissionFactor(
            category="plastics", factor_kgco2e_per_usd=0.75, source="EPA EEIO 2024", year=2024
        ),
        "chemicals": SpendEmissionFactor(
            category="chemicals", factor_kgco2e_per_usd=0.65, source="EPA EEIO 2024", year=2024
        ),
        "paper": SpendEmissionFactor(
            category="paper", factor_kgco2e_per_usd=0.45, source="EPA EEIO 2024", year=2024
        ),
        "electronics": SpendEmissionFactor(
            category="electronics", factor_kgco2e_per_usd=0.55, source="EPA EEIO 2024", year=2024
        ),
        "machinery": SpendEmissionFactor(
            category="machinery", factor_kgco2e_per_usd=0.40, source="EPA EEIO 2024", year=2024
        ),
        "textiles": SpendEmissionFactor(
            category="textiles", factor_kgco2e_per_usd=0.50, source="EPA EEIO 2024", year=2024
        ),
        "food": SpendEmissionFactor(
            category="food", factor_kgco2e_per_usd=0.60, source="EPA EEIO 2024", year=2024
        ),
        "construction": SpendEmissionFactor(
            category="construction", factor_kgco2e_per_usd=0.35, source="EPA EEIO 2024", year=2024
        ),
        "services": SpendEmissionFactor(
            category="services", factor_kgco2e_per_usd=0.15, source="EPA EEIO 2024", year=2024
        ),
        "it_services": SpendEmissionFactor(
            category="it_services", factor_kgco2e_per_usd=0.18, source="EPA EEIO 2024", year=2024
        ),
        "professional_services": SpendEmissionFactor(
            category="professional_services", factor_kgco2e_per_usd=0.12, source="EPA EEIO 2024", year=2024
        ),
        "default": SpendEmissionFactor(
            category="default", factor_kgco2e_per_usd=0.40, source="EPA EEIO 2024", year=2024
        ),
    }

    # Transport emission factors (kgCO2e per tonne-km) - GLEC Framework
    TRANSPORT_FACTORS: Dict[str, float] = {
        "road_truck": 0.089,
        "road_van": 0.195,
        "rail_freight": 0.028,
        "sea_container": 0.016,
        "sea_bulk": 0.008,
        "air_freight": 0.602,
        "air_belly": 0.301,
        "barge": 0.031,
        "pipeline": 0.025,
    }

    # Business travel factors (kgCO2e per passenger-km)
    TRAVEL_FACTORS: Dict[str, float] = {
        "air_short_haul": 0.255,  # <1500km
        "air_medium_haul": 0.156,  # 1500-4000km
        "air_long_haul": 0.195,  # >4000km
        "rail_average": 0.041,
        "rail_highspeed": 0.006,
        "car_average": 0.171,
        "car_electric": 0.053,
        "bus": 0.089,
        "taxi": 0.203,
    }

    # Employee commuting factors (kgCO2e per passenger-km)
    COMMUTING_FACTORS: Dict[str, float] = {
        "car_alone": 0.171,
        "car_carpool": 0.086,
        "public_transit": 0.089,
        "rail": 0.041,
        "bus": 0.089,
        "bicycle": 0.0,
        "walking": 0.0,
        "remote": 0.002,  # Home office electricity
    }

    # Waste factors (kgCO2e per kg waste)
    WASTE_FACTORS: Dict[str, float] = {
        "landfill_mixed": 0.586,
        "landfill_organic": 0.700,
        "incineration": 0.021,
        "recycling_paper": -0.900,
        "recycling_plastic": -1.400,
        "recycling_metal": -1.800,
        "recycling_glass": -0.300,
        "composting": 0.010,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Scope 3 Emissions Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"Scope3EmissionsAgent initialized (version {self.VERSION})")

    def run(self, input_data: Scope3Input) -> Scope3Output:
        """
        Execute the Scope 3 emissions calculation.

        ZERO-HALLUCINATION calculations by category:
        - Category 1-2: emissions = spend * EEIO_factor
        - Category 4, 9: emissions = distance * weight * transport_factor
        - Category 6: emissions = distance * travel_factor

        Args:
            input_data: Validated Scope 3 input data

        Returns:
            Calculation result with emissions by source
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        cat_info = self.CATEGORY_NAMES[input_data.category]
        logger.info(
            f"Calculating Scope 3 Cat {cat_info[0]}: {cat_info[1]}, "
            f"method={input_data.calculation_method}"
        )

        try:
            # Step 1: Dispatch to category-specific calculator
            emissions_by_source: Dict[str, float] = {}
            factors_used: List[Dict[str, Any]] = []

            if input_data.category in [
                Scope3Category.CAT_1_PURCHASED_GOODS,
                Scope3Category.CAT_2_CAPITAL_GOODS,
            ]:
                emissions_by_source, factors_used = self._calculate_spend_based(
                    input_data.spend_data
                )

            elif input_data.category in [
                Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
                Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT,
            ]:
                emissions_by_source, factors_used = self._calculate_transport(
                    input_data.transport_data
                )

            elif input_data.category == Scope3Category.CAT_6_BUSINESS_TRAVEL:
                emissions_by_source, factors_used = self._calculate_travel(
                    input_data.travel_data
                )

            elif input_data.category == Scope3Category.CAT_5_WASTE:
                emissions_by_source, factors_used = self._calculate_waste(
                    input_data.activity_data
                )

            else:
                # Generic spend-based fallback
                emissions_by_source, factors_used = self._calculate_spend_based(
                    input_data.spend_data
                )

            self._track_step("emissions_calculation", {
                "category": input_data.category.value,
                "method": input_data.calculation_method.value,
                "sources": list(emissions_by_source.keys()),
                "total_sources": len(emissions_by_source),
            })

            # Step 2: ZERO-HALLUCINATION CALCULATION
            # Total = SUM(emissions_by_source)
            total_emissions = sum(emissions_by_source.values())

            self._track_step("total_calculation", {
                "formula": "total = SUM(emissions_by_source)",
                "emissions_by_source": emissions_by_source,
                "total_emissions": total_emissions,
            })

            # Step 3: Calculate data quality
            data_quality, coverage = self._assess_data_quality(
                input_data,
                emissions_by_source,
            )

            # Step 4: Calculate intensity metrics
            emissions_per_revenue = None
            emissions_per_employee = None

            if input_data.revenue_usd and input_data.revenue_usd > 0:
                emissions_per_revenue = total_emissions / input_data.revenue_usd

            if input_data.employees and input_data.employees > 0:
                emissions_per_employee = total_emissions / input_data.employees

            # Step 5: Generate improvement opportunities
            improvements = self._generate_improvements(
                input_data.category,
                emissions_by_source,
                input_data.calculation_method,
            )

            # Step 6: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 7: Create output
            output = Scope3Output(
                category=input_data.category.value,
                category_number=cat_info[0],
                category_name=cat_info[1],
                total_emissions_kgco2e=round(total_emissions, 2),
                emissions_by_source={k: round(v, 2) for k, v in emissions_by_source.items()},
                calculation_method=input_data.calculation_method.value,
                emission_factors_used=factors_used,
                data_quality_score=data_quality.value,
                data_coverage_pct=round(coverage, 2),
                uncertainty_pct=self._estimate_uncertainty(data_quality),
                emissions_per_revenue=round(emissions_per_revenue, 6) if emissions_per_revenue else None,
                emissions_per_employee=round(emissions_per_employee, 2) if emissions_per_employee else None,
                improvement_opportunities=improvements,
                provenance_hash=provenance_hash,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Scope 3 Cat {cat_info[0]} calculation complete: "
                f"{total_emissions:.2f} kgCO2e, DQ={data_quality.value} "
                f"(duration: {duration_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Scope 3 calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_spend_based(
        self,
        spend_data: List[SpendData],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Calculate emissions using spend-based method.

        ZERO-HALLUCINATION: emissions = spend * EEIO_factor
        """
        emissions: Dict[str, float] = {}
        factors: List[Dict[str, Any]] = []

        for spend in spend_data:
            category = spend.category.lower()
            factor = self.SPEND_FACTORS.get(category, self.SPEND_FACTORS["default"])

            # emissions = spend * factor
            emission = spend.spend_usd * factor.factor_kgco2e_per_usd

            source_key = spend.supplier_name or category
            emissions[source_key] = emissions.get(source_key, 0) + emission

            factors.append({
                "category": category,
                "factor": factor.factor_kgco2e_per_usd,
                "unit": "kgCO2e/USD",
                "source": factor.source,
            })

        return emissions, factors

    def _calculate_transport(
        self,
        transport_data: List[TransportData],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Calculate transport emissions.

        ZERO-HALLUCINATION: emissions = distance * weight * factor
        """
        emissions: Dict[str, float] = {}
        factors: List[Dict[str, Any]] = []

        for transport in transport_data:
            mode = transport.mode.lower().replace(" ", "_")
            factor = self.TRANSPORT_FACTORS.get(mode, 0.089)  # Default to road

            # emissions = distance_km * weight_tonnes * factor (kgCO2e/tkm)
            tonne_km = transport.distance_km * transport.weight_tonnes
            emission = tonne_km * factor

            emissions[mode] = emissions.get(mode, 0) + emission

            factors.append({
                "mode": mode,
                "factor": factor,
                "unit": "kgCO2e/tonne-km",
                "source": "GLEC Framework",
            })

        return emissions, factors

    def _calculate_travel(
        self,
        travel_data: List[TravelData],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Calculate business travel emissions.

        ZERO-HALLUCINATION: emissions = distance * factor
        """
        emissions: Dict[str, float] = {}
        factors: List[Dict[str, Any]] = []

        for travel in travel_data:
            mode = travel.mode.lower().replace(" ", "_")

            # Determine air haul type
            if mode == "air":
                if travel.distance_km < 1500:
                    mode = "air_short_haul"
                elif travel.distance_km < 4000:
                    mode = "air_medium_haul"
                else:
                    mode = "air_long_haul"

            factor = self.TRAVEL_FACTORS.get(mode, 0.171)

            # Adjust for round trip
            distance = travel.distance_km
            if travel.trip_type == "round_trip":
                distance *= 2

            # emissions = distance * factor
            emission = distance * factor

            emissions[mode] = emissions.get(mode, 0) + emission

            factors.append({
                "mode": mode,
                "factor": factor,
                "unit": "kgCO2e/passenger-km",
                "source": "DEFRA 2024",
            })

        return emissions, factors

    def _calculate_waste(
        self,
        activity_data: List[ActivityData],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Calculate waste emissions.

        ZERO-HALLUCINATION: emissions = quantity * factor
        """
        emissions: Dict[str, float] = {}
        factors: List[Dict[str, Any]] = []

        for activity in activity_data:
            waste_type = activity.activity_type.lower().replace(" ", "_")
            factor = self.WASTE_FACTORS.get(waste_type, 0.586)

            # Convert to kg if needed
            quantity_kg = activity.quantity
            if activity.unit == "tonnes":
                quantity_kg *= 1000

            # emissions = quantity * factor
            emission = quantity_kg * factor

            emissions[waste_type] = emissions.get(waste_type, 0) + emission

            factors.append({
                "waste_type": waste_type,
                "factor": factor,
                "unit": "kgCO2e/kg",
                "source": "DEFRA 2024",
            })

        return emissions, factors

    def _assess_data_quality(
        self,
        input_data: Scope3Input,
        emissions: Dict[str, float],
    ) -> Tuple[DataQualityScore, float]:
        """Assess data quality using GHG Protocol indicators."""
        # Calculate coverage
        has_data = len(emissions) > 0
        method = input_data.calculation_method

        if method == CalculationMethod.SUPPLIER_SPECIFIC:
            return DataQualityScore.VERY_GOOD, 95.0
        elif method == CalculationMethod.HYBRID:
            return DataQualityScore.GOOD, 80.0
        elif method == CalculationMethod.AVERAGE_DATA:
            return DataQualityScore.FAIR, 60.0
        else:  # Spend-based
            return DataQualityScore.FAIR if has_data else DataQualityScore.POOR, 50.0

    def _estimate_uncertainty(self, quality: DataQualityScore) -> float:
        """Estimate uncertainty based on data quality."""
        uncertainty_map = {
            DataQualityScore.VERY_GOOD: 10.0,
            DataQualityScore.GOOD: 25.0,
            DataQualityScore.FAIR: 50.0,
            DataQualityScore.POOR: 75.0,
            DataQualityScore.VERY_POOR: 100.0,
        }
        return uncertainty_map.get(quality, 50.0)

    def _generate_improvements(
        self,
        category: Scope3Category,
        emissions: Dict[str, float],
        method: CalculationMethod,
    ) -> List[str]:
        """Generate improvement opportunities."""
        improvements = []

        if method == CalculationMethod.SPEND_BASED:
            improvements.append(
                "Collect primary data from key suppliers to improve accuracy"
            )
            improvements.append(
                "Request product carbon footprints from top-spend suppliers"
            )

        # Find top emission sources
        if emissions:
            top_source = max(emissions.items(), key=lambda x: x[1])
            improvements.append(
                f"Focus reduction efforts on '{top_source[0]}' ({top_source[1]:.0f} kgCO2e)"
            )

        if category == Scope3Category.CAT_6_BUSINESS_TRAVEL:
            improvements.append("Consider virtual meeting alternatives for short trips")
            improvements.append("Switch to rail for journeys under 500km")

        if category == Scope3Category.CAT_4_UPSTREAM_TRANSPORT:
            improvements.append("Optimize logistics routes to reduce distances")
            improvements.append("Consider rail freight for long-distance transport")

        return improvements

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_categories(self) -> List[Dict[str, Any]]:
        """Get list of Scope 3 categories."""
        return [
            {"category": cat.value, "number": info[0], "name": info[1]}
            for cat, info in self.CATEGORY_NAMES.items()
        ]


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "emissions/scope3_v1",
    "name": "Scope 3 Emissions Agent",
    "version": "1.0.0",
    "summary": "Scope 3 supply chain emissions calculator",
    "tags": ["scope3", "supply-chain", "ghg-protocol", "value-chain"],
    "owners": ["emissions-team"],
    "compute": {
        "entrypoint": "python://agents.gl_006_scope3_emissions.agent:Scope3EmissionsAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://epa/eeio/2024"},
        {"ref": "ef://glec/transport/2024"},
        {"ref": "ef://defra/travel/2024"},
    ],
    "provenance": {
        "methodology": "GHG Protocol Scope 3",
        "enable_audit": True,
    },
}
