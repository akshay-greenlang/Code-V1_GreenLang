"""
Scope 3 Emission Allocation Module.

This module provides comprehensive emission allocation methodologies for
Scope 3 greenhouse gas inventory reporting according to the GHG Protocol
Corporate Value Chain (Scope 3) Standard.

Supported Methodologies:
- Spend-based: Using EEIO (Environmentally Extended Input-Output) factors
- Activity-based: Using supplier-specific emission factors
- Hybrid: Combining spend and activity data for improved accuracy
- Average-data: Using industry average emission factors

Scope 3 Categories Covered:
- Category 1: Purchased goods and services
- Category 2: Capital goods
- Category 3: Fuel- and energy-related activities
- Category 4: Upstream transportation and distribution
- Category 5: Waste generated in operations
- Category 6: Business travel
- Category 7: Employee commuting
- Category 8: Upstream leased assets
- Category 9: Downstream transportation and distribution
- Category 10: Processing of sold products
- Category 11: Use of sold products
- Category 12: End-of-life treatment of sold products
- Category 13: Downstream leased assets
- Category 14: Franchises
- Category 15: Investments

Example:
    >>> from greenlang.supply_chain.scope3 import Scope3Allocator, AllocationMethod
    >>> allocator = Scope3Allocator()
    >>>
    >>> # Add supplier emission profiles
    >>> allocator.add_supplier_profile(profile)
    >>>
    >>> # Calculate Scope 3 emissions
    >>> results = allocator.calculate_category1_emissions(
    ...     spend_data=procurement_data,
    ...     method=AllocationMethod.HYBRID
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from greenlang.supply_chain.models.entity import Supplier

logger = logging.getLogger(__name__)


class Scope3Category(Enum):
    """
    GHG Protocol Scope 3 emission categories.

    Upstream categories (1-8): Emissions from purchased/acquired goods and services
    Downstream categories (9-15): Emissions from sold goods and services
    """
    # Upstream categories
    PURCHASED_GOODS_SERVICES = 1
    CAPITAL_GOODS = 2
    FUEL_ENERGY_ACTIVITIES = 3
    UPSTREAM_TRANSPORTATION = 4
    WASTE_OPERATIONS = 5
    BUSINESS_TRAVEL = 6
    EMPLOYEE_COMMUTING = 7
    UPSTREAM_LEASED_ASSETS = 8

    # Downstream categories
    DOWNSTREAM_TRANSPORTATION = 9
    PROCESSING_SOLD_PRODUCTS = 10
    USE_SOLD_PRODUCTS = 11
    END_OF_LIFE_PRODUCTS = 12
    DOWNSTREAM_LEASED_ASSETS = 13
    FRANCHISES = 14
    INVESTMENTS = 15

    @property
    def name_description(self) -> str:
        """Get category name and description."""
        descriptions = {
            1: "Purchased goods and services",
            2: "Capital goods",
            3: "Fuel- and energy-related activities (not included in Scope 1 or 2)",
            4: "Upstream transportation and distribution",
            5: "Waste generated in operations",
            6: "Business travel",
            7: "Employee commuting",
            8: "Upstream leased assets",
            9: "Downstream transportation and distribution",
            10: "Processing of sold products",
            11: "Use of sold products",
            12: "End-of-life treatment of sold products",
            13: "Downstream leased assets",
            14: "Franchises",
            15: "Investments",
        }
        return descriptions.get(self.value, "Unknown category")

    @property
    def is_upstream(self) -> bool:
        """Check if this is an upstream category."""
        return self.value <= 8

    @property
    def is_downstream(self) -> bool:
        """Check if this is a downstream category."""
        return self.value >= 9


class AllocationMethod(Enum):
    """
    Scope 3 emission allocation methodologies.

    From GHG Protocol Scope 3 guidance:
    - SPEND_BASED: Uses economic input-output emission factors (kg CO2e/$)
    - ACTIVITY_BASED: Uses physical activity data with emission factors
    - HYBRID: Combines spend and activity data for best available accuracy
    - AVERAGE_DATA: Uses industry/sector average emission intensities
    - SUPPLIER_SPECIFIC: Uses primary data from suppliers
    """
    SPEND_BASED = "spend_based"
    ACTIVITY_BASED = "activity_based"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class DataQuality(Enum):
    """
    Data quality classification for emission factors.

    Based on GHG Protocol data quality hierarchy:
    - PRIMARY: Supplier-specific data (highest quality)
    - SECONDARY: Industry average or representative data
    - TERTIARY: Economic input-output or modeled data
    """
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


@dataclass
class EmissionFactor:
    """
    Emission factor with metadata.

    Attributes:
        value: Emission factor value
        unit: Unit of measure (e.g., "kg_co2e_per_kg", "kg_co2e_per_usd")
        source: Data source
        year: Reference year
        geography: Geographic scope
        data_quality: Quality classification
        uncertainty_pct: Uncertainty percentage (+/-)
    """
    value: Decimal
    unit: str
    source: str
    year: int
    geography: str = "GLO"  # Global
    data_quality: DataQuality = DataQuality.SECONDARY
    uncertainty_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": str(self.value),
            "unit": self.unit,
            "source": self.source,
            "year": self.year,
            "geography": self.geography,
            "data_quality": self.data_quality.value,
            "uncertainty_pct": self.uncertainty_pct,
        }


@dataclass
class SupplierEmissionProfile:
    """
    Emission profile for a supplier.

    Contains emission intensity data for allocation calculations.

    Attributes:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        industry_code: NAICS/SIC industry code
        emission_intensity_spend: kg CO2e per $ of spend
        emission_intensity_activity: kg CO2e per unit of activity
        activity_unit: Unit for activity-based calculation
        scope1_emissions: Supplier's Scope 1 emissions (if known)
        scope2_emissions: Supplier's Scope 2 emissions (if known)
        scope3_emissions: Supplier's Scope 3 emissions (if known)
        total_emissions: Total emissions
        data_quality: Data quality classification
        reporting_year: Reporting year
        verification_status: Whether data is third-party verified
        source: Data source
    """
    supplier_id: str
    supplier_name: str
    industry_code: Optional[str] = None
    emission_intensity_spend: Optional[Decimal] = None  # kg CO2e / $
    emission_intensity_activity: Optional[Decimal] = None  # kg CO2e / unit
    activity_unit: str = "kg"
    scope1_emissions: Optional[Decimal] = None
    scope2_emissions: Optional[Decimal] = None
    scope3_emissions: Optional[Decimal] = None
    total_emissions: Optional[Decimal] = None
    data_quality: DataQuality = DataQuality.TERTIARY
    reporting_year: int = 2023
    verification_status: str = "unverified"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "industry_code": self.industry_code,
            "emission_intensity_spend": str(self.emission_intensity_spend) if self.emission_intensity_spend else None,
            "emission_intensity_activity": str(self.emission_intensity_activity) if self.emission_intensity_activity else None,
            "activity_unit": self.activity_unit,
            "scope1_emissions": str(self.scope1_emissions) if self.scope1_emissions else None,
            "scope2_emissions": str(self.scope2_emissions) if self.scope2_emissions else None,
            "scope3_emissions": str(self.scope3_emissions) if self.scope3_emissions else None,
            "total_emissions": str(self.total_emissions) if self.total_emissions else None,
            "data_quality": self.data_quality.value,
            "reporting_year": self.reporting_year,
            "verification_status": self.verification_status,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class EmissionAllocation:
    """
    Result of emission allocation calculation.

    Attributes:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        category: Scope 3 category
        method: Allocation method used
        spend_amount: Procurement spend amount
        spend_currency: Currency code
        activity_amount: Activity amount (if applicable)
        activity_unit: Activity unit
        emission_factor: Emission factor used
        emission_factor_unit: Emission factor unit
        emissions_kg_co2e: Calculated emissions in kg CO2e
        emissions_tonnes_co2e: Calculated emissions in tonnes CO2e
        uncertainty_low: Lower bound of uncertainty range
        uncertainty_high: Upper bound of uncertainty range
        data_quality: Data quality classification
        calculation_date: Date of calculation
        notes: Additional notes
    """
    supplier_id: str
    supplier_name: str
    category: Scope3Category
    method: AllocationMethod
    spend_amount: Optional[Decimal] = None
    spend_currency: str = "USD"
    activity_amount: Optional[Decimal] = None
    activity_unit: Optional[str] = None
    emission_factor: Decimal = Decimal("0")
    emission_factor_unit: str = "kg_co2e_per_usd"
    emissions_kg_co2e: Decimal = Decimal("0")
    emissions_tonnes_co2e: Decimal = Decimal("0")
    uncertainty_low: Optional[Decimal] = None
    uncertainty_high: Optional[Decimal] = None
    data_quality: DataQuality = DataQuality.TERTIARY
    calculation_date: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def __post_init__(self):
        """Calculate tonnes from kg."""
        if self.emissions_kg_co2e and not self.emissions_tonnes_co2e:
            self.emissions_tonnes_co2e = self.emissions_kg_co2e / Decimal("1000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "category": self.category.value,
            "category_name": self.category.name_description,
            "method": self.method.value,
            "spend_amount": str(self.spend_amount) if self.spend_amount else None,
            "spend_currency": self.spend_currency,
            "activity_amount": str(self.activity_amount) if self.activity_amount else None,
            "activity_unit": self.activity_unit,
            "emission_factor": str(self.emission_factor),
            "emission_factor_unit": self.emission_factor_unit,
            "emissions_kg_co2e": str(self.emissions_kg_co2e),
            "emissions_tonnes_co2e": str(self.emissions_tonnes_co2e),
            "uncertainty_low": str(self.uncertainty_low) if self.uncertainty_low else None,
            "uncertainty_high": str(self.uncertainty_high) if self.uncertainty_high else None,
            "data_quality": self.data_quality.value,
            "calculation_date": self.calculation_date.isoformat(),
            "notes": self.notes,
        }


@dataclass
class SpendRecord:
    """
    Procurement spend record for allocation.

    Attributes:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        amount: Spend amount
        currency: Currency code
        category_code: Procurement category code
        category_name: Procurement category name
        industry_code: NAICS/SIC code
        period_start: Period start date
        period_end: Period end date
        po_number: Purchase order number
        invoice_number: Invoice number
    """
    supplier_id: str
    supplier_name: str
    amount: Decimal
    currency: str = "USD"
    category_code: Optional[str] = None
    category_name: Optional[str] = None
    industry_code: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    po_number: Optional[str] = None
    invoice_number: Optional[str] = None


@dataclass
class ActivityRecord:
    """
    Activity data record for allocation.

    Attributes:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        activity_type: Type of activity
        quantity: Activity quantity
        unit: Unit of measure
        product_id: Product/material ID
        product_name: Product/material name
        period_start: Period start date
        period_end: Period end date
    """
    supplier_id: str
    supplier_name: str
    activity_type: str
    quantity: Decimal
    unit: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class Scope3Allocator:
    """
    Scope 3 emission allocation engine.

    Calculates Scope 3 emissions using various methodologies according
    to the GHG Protocol Corporate Value Chain Standard.

    Example:
        >>> allocator = Scope3Allocator(reporting_year=2024)
        >>>
        >>> # Add supplier emission profiles
        >>> allocator.add_supplier_profile(SupplierEmissionProfile(
        ...     supplier_id="SUP001",
        ...     supplier_name="Acme Corp",
        ...     emission_intensity_spend=Decimal("0.5")  # kg CO2e per $
        ... ))
        >>>
        >>> # Calculate emissions from spend data
        >>> spend_records = [
        ...     SpendRecord("SUP001", "Acme Corp", Decimal("100000")),
        ... ]
        >>> results = allocator.calculate_category1_emissions(spend_records)
    """

    # Default EEIO emission factors by industry (kg CO2e per $ spent)
    # Based on EPA/DEFRA/Exiobase data
    DEFAULT_EEIO_FACTORS: Dict[str, Decimal] = {
        # Manufacturing
        "31": Decimal("0.65"),  # Food manufacturing
        "32": Decimal("0.45"),  # Beverage and tobacco
        "33": Decimal("0.55"),  # Textile mills
        "325": Decimal("0.75"),  # Chemical manufacturing
        "326": Decimal("0.68"),  # Plastics and rubber
        "331": Decimal("1.20"),  # Primary metals
        "332": Decimal("0.52"),  # Fabricated metals
        "333": Decimal("0.38"),  # Machinery
        "334": Decimal("0.25"),  # Computer and electronics
        "335": Decimal("0.42"),  # Electrical equipment
        "336": Decimal("0.48"),  # Transportation equipment

        # Services
        "48": Decimal("0.85"),  # Transportation
        "49": Decimal("0.72"),  # Transportation support
        "51": Decimal("0.15"),  # Information
        "52": Decimal("0.08"),  # Finance and insurance
        "54": Decimal("0.12"),  # Professional services
        "56": Decimal("0.18"),  # Administrative services

        # Agriculture
        "11": Decimal("1.80"),  # Agriculture

        # Mining
        "21": Decimal("1.50"),  # Mining

        # Utilities
        "22": Decimal("2.20"),  # Utilities

        # Construction
        "23": Decimal("0.58"),  # Construction

        # Default
        "default": Decimal("0.40"),
    }

    def __init__(
        self,
        reporting_year: int = 2024,
        base_currency: str = "USD",
        default_method: AllocationMethod = AllocationMethod.HYBRID,
    ):
        """
        Initialize the Scope 3 allocator.

        Args:
            reporting_year: GHG reporting year
            base_currency: Base currency for spend data
            default_method: Default allocation methodology
        """
        self.reporting_year = reporting_year
        self.base_currency = base_currency
        self.default_method = default_method

        # Supplier emission profiles
        self._supplier_profiles: Dict[str, SupplierEmissionProfile] = {}

        # Custom emission factors
        self._custom_factors: Dict[str, EmissionFactor] = {}

        # Calculation results
        self._allocations: List[EmissionAllocation] = []

        logger.info(
            f"Scope3Allocator initialized for year {reporting_year}"
        )

    def add_supplier_profile(self, profile: SupplierEmissionProfile) -> None:
        """
        Add or update a supplier emission profile.

        Args:
            profile: Supplier emission profile
        """
        self._supplier_profiles[profile.supplier_id] = profile
        logger.debug(f"Added supplier profile: {profile.supplier_id}")

    def get_supplier_profile(
        self,
        supplier_id: str
    ) -> Optional[SupplierEmissionProfile]:
        """Get supplier emission profile by ID."""
        return self._supplier_profiles.get(supplier_id)

    def set_custom_factor(
        self,
        key: str,
        factor: EmissionFactor
    ) -> None:
        """
        Set a custom emission factor.

        Args:
            key: Factor key (e.g., industry code, product code)
            factor: Emission factor
        """
        self._custom_factors[key] = factor

    def get_emission_factor(
        self,
        supplier_id: Optional[str] = None,
        industry_code: Optional[str] = None,
    ) -> Tuple[Decimal, str, DataQuality]:
        """
        Get the best available emission factor for allocation.

        Priority:
        1. Supplier-specific factor (primary data)
        2. Custom factor by industry code
        3. Default EEIO factor by industry
        4. Global default factor

        Args:
            supplier_id: Supplier identifier
            industry_code: NAICS/SIC industry code

        Returns:
            Tuple of (factor_value, unit, data_quality)
        """
        # 1. Check supplier-specific profile
        if supplier_id and supplier_id in self._supplier_profiles:
            profile = self._supplier_profiles[supplier_id]
            if profile.emission_intensity_spend:
                return (
                    profile.emission_intensity_spend,
                    "kg_co2e_per_usd",
                    profile.data_quality,
                )

        # 2. Check custom factors
        if industry_code and industry_code in self._custom_factors:
            factor = self._custom_factors[industry_code]
            return (factor.value, factor.unit, factor.data_quality)

        # 3. Check default EEIO factors
        if industry_code:
            # Try exact match
            if industry_code in self.DEFAULT_EEIO_FACTORS:
                return (
                    self.DEFAULT_EEIO_FACTORS[industry_code],
                    "kg_co2e_per_usd",
                    DataQuality.TERTIARY,
                )
            # Try prefix match (2-digit NAICS)
            prefix = industry_code[:2] if len(industry_code) >= 2 else industry_code
            if prefix in self.DEFAULT_EEIO_FACTORS:
                return (
                    self.DEFAULT_EEIO_FACTORS[prefix],
                    "kg_co2e_per_usd",
                    DataQuality.TERTIARY,
                )

        # 4. Return default factor
        return (
            self.DEFAULT_EEIO_FACTORS["default"],
            "kg_co2e_per_usd",
            DataQuality.TERTIARY,
        )

    def calculate_category1_emissions(
        self,
        spend_records: List[SpendRecord],
        activity_records: Optional[List[ActivityRecord]] = None,
        method: Optional[AllocationMethod] = None,
    ) -> List[EmissionAllocation]:
        """
        Calculate Category 1 (Purchased goods and services) emissions.

        Args:
            spend_records: List of procurement spend records
            activity_records: Optional list of activity data records
            method: Allocation method to use

        Returns:
            List of EmissionAllocation results
        """
        method = method or self.default_method
        allocations: List[EmissionAllocation] = []

        # Group spend by supplier
        supplier_spend: Dict[str, Decimal] = defaultdict(Decimal)
        supplier_names: Dict[str, str] = {}
        supplier_industry: Dict[str, str] = {}

        for record in spend_records:
            supplier_spend[record.supplier_id] += record.amount
            supplier_names[record.supplier_id] = record.supplier_name
            if record.industry_code:
                supplier_industry[record.supplier_id] = record.industry_code

        # Group activity by supplier (if provided)
        supplier_activity: Dict[str, List[ActivityRecord]] = defaultdict(list)
        if activity_records:
            for record in activity_records:
                supplier_activity[record.supplier_id].append(record)

        # Calculate allocations
        for supplier_id, spend_amount in supplier_spend.items():
            allocation = self._calculate_supplier_allocation(
                supplier_id=supplier_id,
                supplier_name=supplier_names.get(supplier_id, "Unknown"),
                spend_amount=spend_amount,
                industry_code=supplier_industry.get(supplier_id),
                activity_records=supplier_activity.get(supplier_id),
                category=Scope3Category.PURCHASED_GOODS_SERVICES,
                method=method,
            )
            allocations.append(allocation)

        # Store results
        self._allocations.extend(allocations)

        return allocations

    def calculate_category2_emissions(
        self,
        capital_spend: List[SpendRecord],
        asset_lifetimes: Optional[Dict[str, int]] = None,
    ) -> List[EmissionAllocation]:
        """
        Calculate Category 2 (Capital goods) emissions.

        Capital goods emissions can be allocated over the asset lifetime
        or recognized in the year of purchase.

        Args:
            capital_spend: List of capital expenditure records
            asset_lifetimes: Optional dictionary of asset type to lifetime in years

        Returns:
            List of EmissionAllocation results
        """
        allocations: List[EmissionAllocation] = []

        for record in capital_spend:
            # Get emission factor
            ef, ef_unit, data_quality = self.get_emission_factor(
                supplier_id=record.supplier_id,
                industry_code=record.industry_code,
            )

            # Capital goods typically have higher emission intensity
            # Apply capital goods multiplier
            ef = ef * Decimal("1.2")

            emissions_kg = record.amount * ef

            allocation = EmissionAllocation(
                supplier_id=record.supplier_id,
                supplier_name=record.supplier_name,
                category=Scope3Category.CAPITAL_GOODS,
                method=AllocationMethod.SPEND_BASED,
                spend_amount=record.amount,
                spend_currency=record.currency,
                emission_factor=ef,
                emission_factor_unit=ef_unit,
                emissions_kg_co2e=emissions_kg,
                data_quality=data_quality,
                notes=f"Capital goods allocation for {record.category_name or 'unspecified asset'}",
            )
            allocations.append(allocation)

        self._allocations.extend(allocations)
        return allocations

    def calculate_category4_emissions(
        self,
        transport_records: List[ActivityRecord],
    ) -> List[EmissionAllocation]:
        """
        Calculate Category 4 (Upstream transportation) emissions.

        Args:
            transport_records: List of transportation activity records

        Returns:
            List of EmissionAllocation results
        """
        # Transport emission factors (kg CO2e per tonne-km)
        transport_factors = {
            "road_truck": Decimal("0.062"),
            "road_van": Decimal("0.195"),
            "rail_freight": Decimal("0.028"),
            "sea_container": Decimal("0.016"),
            "sea_bulk": Decimal("0.005"),
            "air_freight": Decimal("0.602"),
            "air_express": Decimal("0.850"),
            "default": Decimal("0.062"),
        }

        allocations: List[EmissionAllocation] = []

        for record in transport_records:
            # Get appropriate factor
            factor = transport_factors.get(
                record.activity_type.lower(),
                transport_factors["default"]
            )

            emissions_kg = record.quantity * factor

            allocation = EmissionAllocation(
                supplier_id=record.supplier_id,
                supplier_name=record.supplier_name,
                category=Scope3Category.UPSTREAM_TRANSPORTATION,
                method=AllocationMethod.ACTIVITY_BASED,
                activity_amount=record.quantity,
                activity_unit=record.unit,
                emission_factor=factor,
                emission_factor_unit="kg_co2e_per_tonne_km",
                emissions_kg_co2e=emissions_kg,
                data_quality=DataQuality.SECONDARY,
                notes=f"Transport mode: {record.activity_type}",
            )
            allocations.append(allocation)

        self._allocations.extend(allocations)
        return allocations

    def calculate_all_categories(
        self,
        spend_records: List[SpendRecord],
        activity_records: Optional[List[ActivityRecord]] = None,
    ) -> Dict[Scope3Category, List[EmissionAllocation]]:
        """
        Calculate emissions for all applicable categories.

        Args:
            spend_records: List of procurement spend records
            activity_records: Optional list of activity data records

        Returns:
            Dictionary mapping categories to allocation results
        """
        results: Dict[Scope3Category, List[EmissionAllocation]] = {}

        # Categorize spend records
        category1_spend = []
        category2_spend = []

        for record in spend_records:
            # Simple categorization based on category code
            if record.category_code and record.category_code.startswith("CAPEX"):
                category2_spend.append(record)
            else:
                category1_spend.append(record)

        # Calculate Category 1
        if category1_spend:
            results[Scope3Category.PURCHASED_GOODS_SERVICES] = (
                self.calculate_category1_emissions(
                    spend_records=category1_spend,
                    activity_records=activity_records,
                )
            )

        # Calculate Category 2
        if category2_spend:
            results[Scope3Category.CAPITAL_GOODS] = (
                self.calculate_category2_emissions(capital_spend=category2_spend)
            )

        # Calculate Category 4 (if transport activity data provided)
        if activity_records:
            transport_records = [
                r for r in activity_records
                if r.activity_type.lower() in [
                    "road_truck", "rail_freight", "sea_container",
                    "air_freight", "transport"
                ]
            ]
            if transport_records:
                results[Scope3Category.UPSTREAM_TRANSPORTATION] = (
                    self.calculate_category4_emissions(transport_records)
                )

        return results

    def _calculate_supplier_allocation(
        self,
        supplier_id: str,
        supplier_name: str,
        spend_amount: Decimal,
        industry_code: Optional[str],
        activity_records: Optional[List[ActivityRecord]],
        category: Scope3Category,
        method: AllocationMethod,
    ) -> EmissionAllocation:
        """
        Calculate emission allocation for a single supplier.

        Args:
            supplier_id: Supplier identifier
            supplier_name: Supplier name
            spend_amount: Total spend amount
            industry_code: Industry classification code
            activity_records: Activity data for supplier
            category: Scope 3 category
            method: Allocation method

        Returns:
            EmissionAllocation result
        """
        # Determine which method to use based on data availability
        profile = self._supplier_profiles.get(supplier_id)

        actual_method = method
        emissions_kg = Decimal("0")
        emission_factor = Decimal("0")
        ef_unit = "kg_co2e_per_usd"
        data_quality = DataQuality.TERTIARY
        activity_amount = None
        activity_unit = None

        # Hybrid method: Use best available data
        if method == AllocationMethod.HYBRID:
            if profile and profile.emission_intensity_activity and activity_records:
                actual_method = AllocationMethod.ACTIVITY_BASED
            elif profile and profile.emission_intensity_spend:
                actual_method = AllocationMethod.SUPPLIER_SPECIFIC
            else:
                actual_method = AllocationMethod.SPEND_BASED

        # Calculate based on method
        if actual_method == AllocationMethod.ACTIVITY_BASED and activity_records:
            # Sum activity-based emissions
            if profile and profile.emission_intensity_activity:
                total_activity = sum(r.quantity for r in activity_records)
                emissions_kg = total_activity * profile.emission_intensity_activity
                emission_factor = profile.emission_intensity_activity
                ef_unit = f"kg_co2e_per_{profile.activity_unit}"
                data_quality = profile.data_quality
                activity_amount = total_activity
                activity_unit = profile.activity_unit

        elif actual_method == AllocationMethod.SUPPLIER_SPECIFIC and profile:
            # Use supplier-specific spend intensity
            if profile.emission_intensity_spend:
                emissions_kg = spend_amount * profile.emission_intensity_spend
                emission_factor = profile.emission_intensity_spend
                ef_unit = "kg_co2e_per_usd"
                data_quality = profile.data_quality

        else:
            # Fall back to spend-based with EEIO factors
            ef, ef_unit, data_quality = self.get_emission_factor(
                supplier_id=supplier_id,
                industry_code=industry_code,
            )
            emissions_kg = spend_amount * ef
            emission_factor = ef
            actual_method = AllocationMethod.SPEND_BASED

        # Calculate uncertainty range (simplified)
        uncertainty_pct = {
            DataQuality.PRIMARY: Decimal("0.10"),
            DataQuality.SECONDARY: Decimal("0.30"),
            DataQuality.TERTIARY: Decimal("0.50"),
        }.get(data_quality, Decimal("0.50"))

        uncertainty_low = emissions_kg * (1 - uncertainty_pct)
        uncertainty_high = emissions_kg * (1 + uncertainty_pct)

        return EmissionAllocation(
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            category=category,
            method=actual_method,
            spend_amount=spend_amount,
            spend_currency=self.base_currency,
            activity_amount=activity_amount,
            activity_unit=activity_unit,
            emission_factor=emission_factor,
            emission_factor_unit=ef_unit,
            emissions_kg_co2e=emissions_kg,
            uncertainty_low=uncertainty_low,
            uncertainty_high=uncertainty_high,
            data_quality=data_quality,
        )

    def get_summary_by_category(self) -> Dict[int, Dict[str, Any]]:
        """
        Get emission summary by Scope 3 category.

        Returns:
            Dictionary with category summaries
        """
        summary: Dict[int, Dict[str, Any]] = {}

        for allocation in self._allocations:
            cat_num = allocation.category.value

            if cat_num not in summary:
                summary[cat_num] = {
                    "category": cat_num,
                    "category_name": allocation.category.name_description,
                    "total_emissions_kg_co2e": Decimal("0"),
                    "total_emissions_tonnes_co2e": Decimal("0"),
                    "total_spend": Decimal("0"),
                    "supplier_count": 0,
                    "data_quality_breakdown": {
                        "primary": 0,
                        "secondary": 0,
                        "tertiary": 0,
                    },
                    "suppliers": [],
                }

            summary[cat_num]["total_emissions_kg_co2e"] += allocation.emissions_kg_co2e
            summary[cat_num]["total_emissions_tonnes_co2e"] += allocation.emissions_tonnes_co2e
            if allocation.spend_amount:
                summary[cat_num]["total_spend"] += allocation.spend_amount
            summary[cat_num]["supplier_count"] += 1
            summary[cat_num]["data_quality_breakdown"][allocation.data_quality.value] += 1
            summary[cat_num]["suppliers"].append({
                "id": allocation.supplier_id,
                "name": allocation.supplier_name,
                "emissions_kg": str(allocation.emissions_kg_co2e),
            })

        return summary

    def get_summary_by_supplier(self) -> List[Dict[str, Any]]:
        """
        Get emission summary by supplier.

        Returns:
            List of supplier emission summaries sorted by emissions
        """
        supplier_totals: Dict[str, Dict[str, Any]] = {}

        for allocation in self._allocations:
            sid = allocation.supplier_id

            if sid not in supplier_totals:
                supplier_totals[sid] = {
                    "supplier_id": sid,
                    "supplier_name": allocation.supplier_name,
                    "total_emissions_kg_co2e": Decimal("0"),
                    "total_emissions_tonnes_co2e": Decimal("0"),
                    "total_spend": Decimal("0"),
                    "categories": set(),
                    "data_quality": allocation.data_quality.value,
                }

            supplier_totals[sid]["total_emissions_kg_co2e"] += allocation.emissions_kg_co2e
            supplier_totals[sid]["total_emissions_tonnes_co2e"] += allocation.emissions_tonnes_co2e
            if allocation.spend_amount:
                supplier_totals[sid]["total_spend"] += allocation.spend_amount
            supplier_totals[sid]["categories"].add(allocation.category.value)

        # Convert to list and sort by emissions
        result = []
        for summary in supplier_totals.values():
            summary["categories"] = list(summary["categories"])
            summary["total_emissions_kg_co2e"] = str(summary["total_emissions_kg_co2e"])
            summary["total_emissions_tonnes_co2e"] = str(summary["total_emissions_tonnes_co2e"])
            summary["total_spend"] = str(summary["total_spend"])
            result.append(summary)

        result.sort(
            key=lambda x: Decimal(x["total_emissions_kg_co2e"]),
            reverse=True
        )

        return result

    def get_pareto_suppliers(
        self,
        threshold: float = 0.80
    ) -> List[Dict[str, Any]]:
        """
        Identify Pareto (80/20) suppliers by emissions.

        Args:
            threshold: Cumulative emission threshold

        Returns:
            List of suppliers contributing to threshold
        """
        supplier_summary = self.get_summary_by_supplier()

        total_emissions = sum(
            Decimal(s["total_emissions_kg_co2e"])
            for s in supplier_summary
        )

        if total_emissions == 0:
            return []

        cumulative = Decimal("0")
        pareto_suppliers = []

        for supplier in supplier_summary:
            supplier_emissions = Decimal(supplier["total_emissions_kg_co2e"])
            cumulative += supplier_emissions
            supplier["cumulative_pct"] = float(cumulative / total_emissions)
            supplier["emission_pct"] = float(supplier_emissions / total_emissions)
            pareto_suppliers.append(supplier)

            if float(cumulative / total_emissions) >= threshold:
                break

        return pareto_suppliers

    def export_results(self) -> Dict[str, Any]:
        """
        Export all allocation results.

        Returns:
            Complete allocation results and summaries
        """
        return {
            "reporting_year": self.reporting_year,
            "calculation_date": datetime.utcnow().isoformat(),
            "methodology": self.default_method.value,
            "allocations": [a.to_dict() for a in self._allocations],
            "summary_by_category": {
                str(k): {
                    **v,
                    "total_emissions_kg_co2e": str(v["total_emissions_kg_co2e"]),
                    "total_emissions_tonnes_co2e": str(v["total_emissions_tonnes_co2e"]),
                    "total_spend": str(v["total_spend"]),
                }
                for k, v in self.get_summary_by_category().items()
            },
            "summary_by_supplier": self.get_summary_by_supplier(),
            "pareto_suppliers": self.get_pareto_suppliers(),
            "total_emissions_tonnes_co2e": str(sum(
                a.emissions_tonnes_co2e for a in self._allocations
            )),
            "supplier_count": len(set(a.supplier_id for a in self._allocations)),
        }
