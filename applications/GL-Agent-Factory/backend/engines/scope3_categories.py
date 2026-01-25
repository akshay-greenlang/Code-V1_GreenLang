"""
Scope 3 GHG Protocol Categories 8-15 Implementation

This module implements the remaining Scope 3 emission categories according to
the GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard.

Categories implemented:
- Category 8:  Upstream leased assets
- Category 9:  Downstream transportation and distribution
- Category 10: Processing of sold products
- Category 11: Use of sold products
- Category 12: End-of-life treatment of sold products
- Category 13: Downstream leased assets
- Category 14: Franchises
- Category 15: Investments

References:
- GHG Protocol: Corporate Value Chain (Scope 3) Standard
- GHG Protocol: Technical Guidance for Calculating Scope 3 Emissions
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Common Models
# =============================================================================


class CalculationMethod(Enum):
    """Scope 3 calculation methodologies."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    ASSET_SPECIFIC = "asset_specific"
    LESSOR_SPECIFIC = "lessor_specific"
    LESSEE_SPECIFIC = "lessee_specific"
    INVESTMENT_SPECIFIC = "investment_specific"
    AVERAGE_PRODUCT = "average_product"


class LeaseType(Enum):
    """Types of lease arrangements."""

    OPERATING = "operating"
    FINANCE = "finance"
    CAPITAL = "capital"


class AssetType(Enum):
    """Types of leased assets."""

    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_EQUIPMENT = "it_equipment"
    MACHINERY = "machinery"


class InvestmentType(Enum):
    """Types of investments for Category 15."""

    EQUITY = "equity"
    DEBT = "debt"
    PROJECT_FINANCE = "project_finance"


class EndOfLifeTreatment(Enum):
    """End-of-life treatment methods."""

    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    WASTEWATER = "wastewater"
    REUSE = "reuse"


@dataclass
class EmissionResult:
    """Result of an emission calculation."""

    emissions_tco2e: Decimal
    methodology: str
    methodology_tier: int  # 1=least accurate, 3=most accurate
    data_quality_score: float  # 0-100
    uncertainty_pct: float
    breakdown: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Calculator
# =============================================================================


class Scope3CategoryCalculator(ABC):
    """Base class for Scope 3 category calculators."""

    CATEGORY_NUMBER: int = 0
    CATEGORY_NAME: str = ""
    GHG_PROTOCOL_REFERENCE: str = "GHG Protocol Scope 3 Standard"

    @abstractmethod
    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate emissions for this category."""
        pass

    def _to_decimal(self, value: Union[float, int, Decimal, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))


# =============================================================================
# Category 8: Upstream Leased Assets
# =============================================================================


@dataclass
class LeasedAssetInput:
    """Input for leased asset emission calculation."""

    asset_type: AssetType
    lease_type: LeaseType
    floor_area_m2: Optional[Decimal] = None
    energy_consumption_kwh: Optional[Decimal] = None
    fuel_consumption_liters: Optional[Decimal] = None
    operating_hours: Optional[Decimal] = None
    region: str = "global"
    year: int = 2024


class Category8UpstreamLeasedAssets(Scope3CategoryCalculator):
    """
    Category 8: Upstream Leased Assets

    Emissions from the operation of assets leased by the reporting company
    (lessee) in the reporting year, not already included in Scope 1 and 2.

    Calculation Methods:
    1. Asset-specific method (Tier 3): Direct energy/fuel data per asset
    2. Lessor-specific method (Tier 2): Data from lessor
    3. Average-data method (Tier 1): Floor area × emission factor
    """

    CATEGORY_NUMBER = 8
    CATEGORY_NAME = "Upstream Leased Assets"

    # Average emission factors (kg CO2e/m²/year) by building type
    BUILDING_EMISSION_FACTORS = {
        "office": Decimal("100"),
        "warehouse": Decimal("60"),
        "retail": Decimal("150"),
        "industrial": Decimal("120"),
        "data_center": Decimal("500"),
        "default": Decimal("100"),
    }

    # Equipment emission factors (kg CO2e/hour)
    EQUIPMENT_EMISSION_FACTORS = {
        "forklift_electric": Decimal("5"),
        "forklift_diesel": Decimal("15"),
        "copier": Decimal("0.5"),
        "server": Decimal("2"),
        "default": Decimal("5"),
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate upstream leased asset emissions."""
        assets = inputs.get("leased_assets", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, asset_data in enumerate(assets):
            asset = LeasedAssetInput(**asset_data) if isinstance(asset_data, dict) else asset_data

            if asset.energy_consumption_kwh:
                # Asset-specific method (Tier 3)
                emissions = self._calculate_from_energy(asset)
                method = CalculationMethod.ASSET_SPECIFIC.value
                tier = 3
            elif asset.floor_area_m2 and asset.asset_type == AssetType.BUILDING:
                # Average-data method (Tier 1)
                emissions = self._calculate_from_floor_area(asset)
                method = CalculationMethod.AVERAGE_DATA.value
                tier = 1
            elif asset.operating_hours:
                # Equipment method
                emissions = self._calculate_from_operating_hours(asset)
                method = CalculationMethod.AVERAGE_DATA.value
                tier = 2
            else:
                continue

            asset_key = f"asset_{i}_{asset.asset_type.value}"
            breakdown[asset_key] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology=f"Category 8 - {method}",
            methodology_tier=tier if assets else 1,
            data_quality_score=70.0 if tier == 3 else 50.0,
            uncertainty_pct=15.0 if tier == 3 else 30.0,
            breakdown=breakdown,
            metadata={"asset_count": len(assets)},
        )

    def _calculate_from_energy(self, asset: LeasedAssetInput) -> Decimal:
        """Calculate from direct energy consumption."""
        # Grid emission factor (kg CO2e/kWh)
        grid_factor = Decimal("0.4")  # Average, should use region-specific
        return asset.energy_consumption_kwh * grid_factor

    def _calculate_from_floor_area(self, asset: LeasedAssetInput) -> Decimal:
        """Calculate from floor area."""
        factor = self.BUILDING_EMISSION_FACTORS.get(
            "default", self.BUILDING_EMISSION_FACTORS["default"]
        )
        return asset.floor_area_m2 * factor

    def _calculate_from_operating_hours(self, asset: LeasedAssetInput) -> Decimal:
        """Calculate from operating hours."""
        factor = self.EQUIPMENT_EMISSION_FACTORS.get(
            "default", self.EQUIPMENT_EMISSION_FACTORS["default"]
        )
        return asset.operating_hours * factor


# =============================================================================
# Category 9: Downstream Transportation and Distribution
# =============================================================================


@dataclass
class DownstreamTransportInput:
    """Input for downstream transport emission calculation."""

    product_mass_kg: Decimal
    transport_mode: str  # "road", "rail", "sea", "air"
    distance_km: Decimal
    vehicle_type: Optional[str] = None
    load_factor: Decimal = Decimal("1.0")


class Category9DownstreamTransport(Scope3CategoryCalculator):
    """
    Category 9: Downstream Transportation and Distribution

    Emissions from transportation and distribution of sold products
    in vehicles not owned or controlled by the reporting company.
    """

    CATEGORY_NUMBER = 9
    CATEGORY_NAME = "Downstream Transportation and Distribution"

    # Emission factors (kg CO2e/tonne-km)
    TRANSPORT_EMISSION_FACTORS = {
        "road_truck": Decimal("0.062"),
        "road_van": Decimal("0.25"),
        "rail_freight": Decimal("0.025"),
        "sea_container": Decimal("0.016"),
        "sea_bulk": Decimal("0.008"),
        "air_freight": Decimal("0.602"),
        "air_belly": Decimal("0.459"),
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate downstream transport emissions."""
        shipments = inputs.get("shipments", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, shipment_data in enumerate(shipments):
            shipment = DownstreamTransportInput(**shipment_data) if isinstance(shipment_data, dict) else shipment_data

            # Get emission factor
            factor_key = f"{shipment.transport_mode}_{shipment.vehicle_type or 'default'}"
            factor = self.TRANSPORT_EMISSION_FACTORS.get(
                factor_key, Decimal("0.062")  # Default to road truck
            )

            # Calculate tonne-km
            tonne_km = (shipment.product_mass_kg / Decimal("1000")) * shipment.distance_km

            # Apply load factor adjustment
            emissions = tonne_km * factor / shipment.load_factor

            breakdown[f"shipment_{i}"] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 9 - Distance-based method",
            methodology_tier=2,
            data_quality_score=65.0,
            uncertainty_pct=25.0,
            breakdown=breakdown,
            metadata={"shipment_count": len(shipments)},
        )


# =============================================================================
# Category 10: Processing of Sold Products
# =============================================================================


@dataclass
class ProcessingInput:
    """Input for processing emission calculation."""

    product_type: str
    quantity_sold: Decimal
    unit: str
    processing_energy_kwh_per_unit: Optional[Decimal] = None
    processing_type: str = "default"


class Category10ProcessingSoldProducts(Scope3CategoryCalculator):
    """
    Category 10: Processing of Sold Products

    Emissions from processing of sold intermediate products by third parties.
    """

    CATEGORY_NUMBER = 10
    CATEGORY_NAME = "Processing of Sold Products"

    # Default processing emission factors (kg CO2e/unit)
    PROCESSING_FACTORS = {
        "steel_fabrication": Decimal("50"),  # per tonne
        "plastic_molding": Decimal("0.5"),  # per kg
        "textile_finishing": Decimal("2"),  # per kg
        "food_processing": Decimal("0.3"),  # per kg
        "chemical_processing": Decimal("1.5"),  # per kg
        "default": Decimal("0.5"),
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate processing emissions."""
        products = inputs.get("products", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, product_data in enumerate(products):
            product = ProcessingInput(**product_data) if isinstance(product_data, dict) else product_data

            if product.processing_energy_kwh_per_unit:
                # Site-specific method
                grid_factor = Decimal("0.4")
                emissions = (
                    product.quantity_sold
                    * product.processing_energy_kwh_per_unit
                    * grid_factor
                )
            else:
                # Average-data method
                factor = self.PROCESSING_FACTORS.get(
                    product.processing_type, self.PROCESSING_FACTORS["default"]
                )
                emissions = product.quantity_sold * factor

            breakdown[f"product_{i}_{product.product_type}"] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 10 - Average-data method",
            methodology_tier=2,
            data_quality_score=55.0,
            uncertainty_pct=35.0,
            breakdown=breakdown,
        )


# =============================================================================
# Category 11: Use of Sold Products
# =============================================================================


@dataclass
class SoldProductUseInput:
    """Input for use-phase emission calculation."""

    product_type: str
    units_sold: Decimal
    lifetime_years: Decimal
    energy_per_use_kwh: Optional[Decimal] = None
    uses_per_year: Optional[Decimal] = None
    fuel_per_use_liters: Optional[Decimal] = None
    direct_ghg_emissions_kg: Optional[Decimal] = None


class Category11UseSoldProducts(Scope3CategoryCalculator):
    """
    Category 11: Use of Sold Products

    Emissions from the use of goods and services sold by the reporting company.
    """

    CATEGORY_NUMBER = 11
    CATEGORY_NAME = "Use of Sold Products"

    # Direct use-phase emission factors
    DIRECT_USE_FACTORS = {
        "air_conditioner": {"refrigerant_leak_rate": Decimal("0.04"), "gwp": Decimal("1430")},
        "vehicle": {"fuel_factor_kg_per_liter": Decimal("2.31")},
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate use-phase emissions."""
        products = inputs.get("products", [])
        total_emissions = Decimal("0")
        breakdown = {}
        direct_emissions = Decimal("0")
        indirect_emissions = Decimal("0")

        grid_factor = Decimal(str(inputs.get("grid_factor", 0.4)))

        for i, product_data in enumerate(products):
            product = SoldProductUseInput(**product_data) if isinstance(product_data, dict) else product_data

            product_emissions = Decimal("0")

            # Direct emissions (e.g., fuel combustion, refrigerant leaks)
            if product.direct_ghg_emissions_kg:
                product_direct = product.units_sold * product.direct_ghg_emissions_kg
                product_emissions += product_direct
                direct_emissions += product_direct

            if product.fuel_per_use_liters and product.uses_per_year:
                fuel_emissions = (
                    product.units_sold
                    * product.fuel_per_use_liters
                    * product.uses_per_year
                    * product.lifetime_years
                    * Decimal("2.31")  # kg CO2/liter diesel
                )
                product_emissions += fuel_emissions
                direct_emissions += fuel_emissions

            # Indirect emissions (electricity use)
            if product.energy_per_use_kwh and product.uses_per_year:
                energy_emissions = (
                    product.units_sold
                    * product.energy_per_use_kwh
                    * product.uses_per_year
                    * product.lifetime_years
                    * grid_factor
                )
                product_emissions += energy_emissions
                indirect_emissions += energy_emissions

            breakdown[f"product_{i}_{product.product_type}"] = product_emissions
            total_emissions += product_emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 11 - Direct use-phase method",
            methodology_tier=2,
            data_quality_score=60.0,
            uncertainty_pct=30.0,
            breakdown=breakdown,
            metadata={
                "direct_emissions_kg": float(direct_emissions),
                "indirect_emissions_kg": float(indirect_emissions),
            },
        )


# =============================================================================
# Category 12: End-of-Life Treatment of Sold Products
# =============================================================================


@dataclass
class EndOfLifeInput:
    """Input for end-of-life emission calculation."""

    product_type: str
    units_sold: Decimal
    product_mass_kg: Decimal
    material_composition: Dict[str, Decimal]  # material -> percentage
    treatment_methods: Dict[str, Decimal]  # method -> percentage


class Category12EndOfLife(Scope3CategoryCalculator):
    """
    Category 12: End-of-Life Treatment of Sold Products

    Emissions from waste disposal and treatment of products sold
    by the reporting company at the end of their life.
    """

    CATEGORY_NUMBER = 12
    CATEGORY_NAME = "End-of-Life Treatment of Sold Products"

    # Emission factors by treatment (kg CO2e/kg material)
    TREATMENT_FACTORS = {
        EndOfLifeTreatment.LANDFILL: {
            "paper": Decimal("1.3"),
            "plastic": Decimal("0.04"),
            "metal": Decimal("0.02"),
            "glass": Decimal("0.02"),
            "organic": Decimal("0.5"),
            "default": Decimal("0.3"),
        },
        EndOfLifeTreatment.INCINERATION: {
            "paper": Decimal("1.5"),
            "plastic": Decimal("2.5"),
            "metal": Decimal("0.02"),
            "glass": Decimal("0.02"),
            "organic": Decimal("0.1"),
            "default": Decimal("1.0"),
        },
        EndOfLifeTreatment.RECYCLING: {
            "paper": Decimal("0.1"),
            "plastic": Decimal("0.5"),
            "metal": Decimal("0.3"),
            "glass": Decimal("0.1"),
            "default": Decimal("0.2"),
        },
        EndOfLifeTreatment.COMPOSTING: {
            "organic": Decimal("0.1"),
            "paper": Decimal("0.2"),
            "default": Decimal("0.1"),
        },
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate end-of-life emissions."""
        products = inputs.get("products", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, product_data in enumerate(products):
            product = EndOfLifeInput(**product_data) if isinstance(product_data, dict) else product_data

            total_mass = product.units_sold * product.product_mass_kg
            product_emissions = Decimal("0")

            for treatment_str, treatment_pct in product.treatment_methods.items():
                treatment = EndOfLifeTreatment(treatment_str)
                treatment_factors = self.TREATMENT_FACTORS.get(treatment, {})

                for material, material_pct in product.material_composition.items():
                    material_mass = total_mass * material_pct * treatment_pct
                    factor = treatment_factors.get(
                        material, treatment_factors.get("default", Decimal("0.3"))
                    )
                    emissions = material_mass * factor
                    product_emissions += emissions

            breakdown[f"product_{i}_{product.product_type}"] = product_emissions
            total_emissions += product_emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 12 - Waste-type-specific method",
            methodology_tier=2,
            data_quality_score=55.0,
            uncertainty_pct=40.0,
            breakdown=breakdown,
        )


# =============================================================================
# Category 13: Downstream Leased Assets
# =============================================================================


class Category13DownstreamLeasedAssets(Scope3CategoryCalculator):
    """
    Category 13: Downstream Leased Assets

    Emissions from the operation of assets owned by the reporting company
    (lessor) and leased to other entities.
    """

    CATEGORY_NUMBER = 13
    CATEGORY_NAME = "Downstream Leased Assets"

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate downstream leased asset emissions."""
        # Similar to Category 8 but from lessor perspective
        assets = inputs.get("leased_assets", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, asset_data in enumerate(assets):
            floor_area = Decimal(str(asset_data.get("floor_area_m2", 0)))
            emission_factor = Decimal(str(asset_data.get("emission_factor", 100)))

            emissions = floor_area * emission_factor
            breakdown[f"asset_{i}"] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 13 - Asset-specific method",
            methodology_tier=2,
            data_quality_score=60.0,
            uncertainty_pct=25.0,
            breakdown=breakdown,
        )


# =============================================================================
# Category 14: Franchises
# =============================================================================


@dataclass
class FranchiseInput:
    """Input for franchise emission calculation."""

    franchise_id: str
    floor_area_m2: Optional[Decimal] = None
    energy_consumption_kwh: Optional[Decimal] = None
    fuel_consumption_liters: Optional[Decimal] = None
    franchise_type: str = "retail"


class Category14Franchises(Scope3CategoryCalculator):
    """
    Category 14: Franchises

    Emissions from the operation of franchises not included in Scope 1 and 2.
    Applicable to franchisors.
    """

    CATEGORY_NUMBER = 14
    CATEGORY_NAME = "Franchises"

    # Average franchise emission factors (kg CO2e/m²/year)
    FRANCHISE_FACTORS = {
        "fast_food": Decimal("200"),
        "retail": Decimal("120"),
        "hotel": Decimal("150"),
        "gas_station": Decimal("80"),
        "default": Decimal("120"),
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate franchise emissions."""
        franchises = inputs.get("franchises", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, franchise_data in enumerate(franchises):
            franchise = FranchiseInput(**franchise_data) if isinstance(franchise_data, dict) else franchise_data

            if franchise.energy_consumption_kwh:
                # Direct data method
                grid_factor = Decimal("0.4")
                emissions = franchise.energy_consumption_kwh * grid_factor
            elif franchise.floor_area_m2:
                # Average-data method
                factor = self.FRANCHISE_FACTORS.get(
                    franchise.franchise_type, self.FRANCHISE_FACTORS["default"]
                )
                emissions = franchise.floor_area_m2 * factor
            else:
                emissions = Decimal("0")

            breakdown[f"franchise_{franchise.franchise_id}"] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions / Decimal("1000"),
            methodology="Category 14 - Franchise-specific method",
            methodology_tier=2,
            data_quality_score=60.0,
            uncertainty_pct=30.0,
            breakdown=breakdown,
            metadata={"franchise_count": len(franchises)},
        )


# =============================================================================
# Category 15: Investments
# =============================================================================


@dataclass
class InvestmentInput:
    """Input for investment emission calculation."""

    company_name: str
    investment_type: InvestmentType
    share_of_investment: Decimal  # Percentage ownership
    investee_scope1_emissions: Optional[Decimal] = None
    investee_scope2_emissions: Optional[Decimal] = None
    investee_revenue: Optional[Decimal] = None
    sector_emission_intensity: Optional[Decimal] = None  # tCO2e per $M revenue


class Category15Investments(Scope3CategoryCalculator):
    """
    Category 15: Investments

    Emissions associated with the reporting company's investments in the
    reporting year, not already included in Scope 1 and 2.
    """

    CATEGORY_NUMBER = 15
    CATEGORY_NAME = "Investments"

    # Sector average emission intensities (tCO2e per $M revenue)
    SECTOR_INTENSITIES = {
        "oil_gas": Decimal("500"),
        "utilities": Decimal("400"),
        "materials": Decimal("300"),
        "industrials": Decimal("150"),
        "consumer": Decimal("80"),
        "technology": Decimal("30"),
        "financial": Decimal("20"),
        "healthcare": Decimal("40"),
        "default": Decimal("100"),
    }

    def calculate(self, inputs: Dict[str, Any]) -> EmissionResult:
        """Calculate investment emissions."""
        investments = inputs.get("investments", [])
        total_emissions = Decimal("0")
        breakdown = {}

        for i, inv_data in enumerate(investments):
            investment = InvestmentInput(**inv_data) if isinstance(inv_data, dict) else inv_data

            if investment.investee_scope1_emissions is not None:
                # Investment-specific method
                investee_emissions = (
                    (investment.investee_scope1_emissions or Decimal("0"))
                    + (investment.investee_scope2_emissions or Decimal("0"))
                )
                emissions = investee_emissions * (investment.share_of_investment / Decimal("100"))
                method = "investment-specific"
            elif investment.investee_revenue and investment.sector_emission_intensity:
                # Average-data method with sector intensity
                revenue_millions = investment.investee_revenue / Decimal("1000000")
                total_investee_emissions = revenue_millions * investment.sector_emission_intensity
                emissions = total_investee_emissions * (investment.share_of_investment / Decimal("100"))
                method = "sector-intensity"
            else:
                emissions = Decimal("0")
                method = "no-data"

            breakdown[f"investment_{investment.company_name}"] = emissions
            total_emissions += emissions

        return EmissionResult(
            emissions_tco2e=total_emissions,
            methodology=f"Category 15 - {method}",
            methodology_tier=2,
            data_quality_score=50.0,
            uncertainty_pct=40.0,
            breakdown=breakdown,
            metadata={"investment_count": len(investments)},
        )


# =============================================================================
# Scope 3 Calculator Factory
# =============================================================================


class Scope3CalculatorFactory:
    """Factory for creating Scope 3 category calculators."""

    _calculators: Dict[int, type] = {
        8: Category8UpstreamLeasedAssets,
        9: Category9DownstreamTransport,
        10: Category10ProcessingSoldProducts,
        11: Category11UseSoldProducts,
        12: Category12EndOfLife,
        13: Category13DownstreamLeasedAssets,
        14: Category14Franchises,
        15: Category15Investments,
    }

    @classmethod
    def get_calculator(cls, category: int) -> Scope3CategoryCalculator:
        """Get calculator for a specific category."""
        if category not in cls._calculators:
            raise ValueError(f"Unknown Scope 3 category: {category}")
        return cls._calculators[category]()

    @classmethod
    def list_categories(cls) -> Dict[int, str]:
        """List all available categories."""
        return {
            cat: calc.CATEGORY_NAME
            for cat, calc in sorted(cls._calculators.items())
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Scope3CategoryCalculator",
    "Scope3CalculatorFactory",
    "Category8UpstreamLeasedAssets",
    "Category9DownstreamTransport",
    "Category10ProcessingSoldProducts",
    "Category11UseSoldProducts",
    "Category12EndOfLife",
    "Category13DownstreamLeasedAssets",
    "Category14Franchises",
    "Category15Investments",
    "EmissionResult",
    "CalculationMethod",
    "LeaseType",
    "AssetType",
    "InvestmentType",
    "EndOfLifeTreatment",
]
