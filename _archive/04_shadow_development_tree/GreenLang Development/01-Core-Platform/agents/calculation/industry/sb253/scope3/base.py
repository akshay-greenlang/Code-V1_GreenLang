# -*- coding: utf-8 -*-
"""
Scope 3 Base Calculator - Foundation for all 15 category calculators

This module provides the base classes and common functionality for all
Scope 3 emission calculators required under California SB 253.

Key Features:
- Zero-Hallucination calculation guarantee (no LLM in calculation path)
- EPA EEIO emission factors for spend-based calculations
- GHG Protocol Technical Guidance formulas
- SHA-256 audit trail generation
- Pydantic models for type safety and validation

Reference: GHG Protocol Scope 3 Technical Guidance (2013)
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CalculationMethod(str, Enum):
    """Supported calculation methods for Scope 3 emissions."""

    # Spend-based methods
    SPEND_BASED = "spend_based"
    EEIO = "eeio"  # Environmentally Extended Input-Output

    # Activity-based methods
    ACTIVITY_BASED = "activity_based"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"

    # Distance-based methods
    DISTANCE_BASED = "distance_based"

    # Asset-specific methods
    ASSET_SPECIFIC = "asset_specific"
    AVERAGE_DATA = "average_data"

    # Investment-specific methods
    EQUITY_SHARE = "equity_share"
    PCAF = "pcaf"  # Partnership for Carbon Accounting Financials

    # Industry-average methods
    INDUSTRY_AVERAGE = "industry_average"


class EmissionFactorSource(str, Enum):
    """Sources for emission factors."""

    EPA_EEIO = "epa_eeio"  # EPA Environmentally Extended Input-Output
    EPA_GHG = "epa_ghg"  # EPA GHG Emission Factors Hub
    DEFRA = "defra"  # UK DEFRA emission factors
    IPCC = "ipcc"  # IPCC emission factors
    ECOINVENT = "ecoinvent"  # ecoinvent database
    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol guidance factors
    CARB = "carb"  # California Air Resources Board
    SUPPLIER_SPECIFIC = "supplier_specific"  # Direct supplier data
    CUSTOM = "custom"  # Organization-specific factors


class DataQualityTier(str, Enum):
    """Data quality tiers per GHG Protocol guidance."""

    TIER_1 = "tier_1"  # Supplier-specific or primary data
    TIER_2 = "tier_2"  # Average data from published sources
    TIER_3 = "tier_3"  # Spend-based or highly aggregated data


class EmissionFactorRecord(BaseModel):
    """
    Emission factor with complete provenance.

    Tracks the source, version, and uncertainty of each factor used
    for regulatory-grade audit trails.
    """

    factor_id: str = Field(..., description="Unique factor identifier")
    factor_value: Decimal = Field(..., description="Emission factor value")
    factor_unit: str = Field(..., description="Factor unit (e.g., kg CO2e/USD)")
    source: EmissionFactorSource = Field(..., description="Factor source")
    source_uri: str = Field("", description="URI to source documentation")
    version: str = Field(..., description="Factor version/year")
    last_updated: str = Field(..., description="Last update date")
    uncertainty_pct: Optional[float] = Field(None, ge=0, le=100, description="Uncertainty percentage")
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_3, description="Data quality tier"
    )
    geographic_scope: str = Field("global", description="Geographic applicability")
    naics_code: Optional[str] = Field(None, description="NAICS code if applicable")
    sic_code: Optional[str] = Field(None, description="SIC code if applicable")

    class Config:
        """Pydantic config."""
        use_enum_values = True


class Scope3CalculationInput(BaseModel):
    """
    Base input model for Scope 3 calculations.

    All category-specific inputs inherit from this base class.
    """

    # Reporting context
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    organization_id: str = Field(..., description="Organization identifier")

    # Calculation parameters
    calculation_method: CalculationMethod = Field(
        CalculationMethod.SPEND_BASED, description="Calculation method to use"
    )

    # Geographic context
    region: str = Field("US", description="Geographic region")
    sub_region: Optional[str] = Field(None, description="Sub-region (e.g., state)")

    # Optional identifiers
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    class Config:
        """Pydantic config."""
        use_enum_values = True

    @validator("reporting_year")
    def validate_reporting_year(cls, v: int) -> int:
        """Validate reporting year is reasonable."""
        current_year = datetime.now().year
        if v > current_year + 1:
            raise ValueError(f"Reporting year {v} is too far in the future")
        return v


class CalculationStep(BaseModel):
    """Single step in the calculation audit trail."""

    step_number: int = Field(..., ge=1, description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: Optional[str] = Field(None, description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output: Optional[str] = Field(None, description="Output value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Step timestamp")


class Scope3CalculationResult(BaseModel):
    """
    Complete result from a Scope 3 calculation with full audit trail.

    Provides:
    - Emission values in multiple units (kg, metric tons)
    - Complete calculation provenance
    - Factor resolution details
    - SHA-256 audit hash
    - Data quality indicators
    """

    # Category identification
    category_number: int = Field(..., ge=1, le=15, description="Scope 3 category (1-15)")
    category_name: str = Field(..., description="Category name")

    # Emission results
    emissions_kg_co2e: Decimal = Field(..., description="Emissions in kg CO2e")
    emissions_mt_co2e: Decimal = Field(..., description="Emissions in metric tons CO2e")

    # Gas breakdown (optional)
    co2_kg: Optional[Decimal] = Field(None, description="CO2 component in kg")
    ch4_kg: Optional[Decimal] = Field(None, description="CH4 component in kg")
    n2o_kg: Optional[Decimal] = Field(None, description="N2O component in kg")

    # Calculation details
    calculation_method: CalculationMethod = Field(..., description="Method used")
    emission_factor_used: EmissionFactorRecord = Field(..., description="Factor applied")

    # Input summary
    activity_data_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of activity data"
    )

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list, description="Complete calculation steps"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash of calculation")

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When calculation was performed"
    )
    calculation_duration_ms: float = Field(0.0, description="Calculation duration in ms")

    # Data quality
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_3, description="Overall data quality"
    )
    uncertainty_pct: Optional[float] = Field(None, description="Uncertainty percentage")

    # Status
    status: str = Field("success", description="Calculation status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    # Metadata
    calculator_version: str = Field("1.0.0", description="Calculator version")

    class Config:
        """Pydantic config."""
        use_enum_values = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return json.loads(self.json())

    def verify_provenance(self) -> bool:
        """Verify the provenance hash is valid."""
        expected_hash = self._calculate_provenance_hash()
        return self.provenance_hash == expected_hash

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of calculation data."""
        # Exclude the hash itself and timestamps from the hash calculation
        data = {
            "category_number": self.category_number,
            "emissions_kg_co2e": str(self.emissions_kg_co2e),
            "calculation_method": self.calculation_method,
            "activity_data_summary": self.activity_data_summary,
            "calculation_steps": [s.dict() for s in self.calculation_steps],
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


# EPA EEIO Emission Factors (kg CO2e per USD of expenditure)
# Source: EPA Supply Chain Emission Factors v1.2 (2023)
EPA_EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    # Manufacturing sectors
    "111": {"name": "Crop production", "factor": 0.67, "naics": "111"},
    "112": {"name": "Animal production", "factor": 0.95, "naics": "112"},
    "211": {"name": "Oil and gas extraction", "factor": 1.23, "naics": "211"},
    "212": {"name": "Mining (except oil and gas)", "factor": 0.78, "naics": "212"},
    "221": {"name": "Utilities", "factor": 1.45, "naics": "221"},
    "236": {"name": "Construction of buildings", "factor": 0.32, "naics": "236"},
    "311": {"name": "Food manufacturing", "factor": 0.45, "naics": "311"},
    "312": {"name": "Beverage and tobacco", "factor": 0.28, "naics": "312"},
    "313": {"name": "Textile mills", "factor": 0.52, "naics": "313"},
    "314": {"name": "Textile product mills", "factor": 0.41, "naics": "314"},
    "315": {"name": "Apparel manufacturing", "factor": 0.38, "naics": "315"},
    "316": {"name": "Leather and allied products", "factor": 0.35, "naics": "316"},
    "321": {"name": "Wood products", "factor": 0.42, "naics": "321"},
    "322": {"name": "Paper manufacturing", "factor": 0.68, "naics": "322"},
    "323": {"name": "Printing", "factor": 0.25, "naics": "323"},
    "324": {"name": "Petroleum and coal products", "factor": 2.15, "naics": "324"},
    "325": {"name": "Chemical manufacturing", "factor": 0.89, "naics": "325"},
    "326": {"name": "Plastics and rubber products", "factor": 0.72, "naics": "326"},
    "327": {"name": "Nonmetallic mineral products", "factor": 0.95, "naics": "327"},
    "331": {"name": "Primary metals", "factor": 1.35, "naics": "331"},
    "332": {"name": "Fabricated metal products", "factor": 0.48, "naics": "332"},
    "333": {"name": "Machinery manufacturing", "factor": 0.35, "naics": "333"},
    "334": {"name": "Computer and electronic products", "factor": 0.28, "naics": "334"},
    "335": {"name": "Electrical equipment", "factor": 0.42, "naics": "335"},
    "336": {"name": "Transportation equipment", "factor": 0.38, "naics": "336"},
    "337": {"name": "Furniture manufacturing", "factor": 0.32, "naics": "337"},
    "339": {"name": "Miscellaneous manufacturing", "factor": 0.35, "naics": "339"},

    # Service sectors
    "42": {"name": "Wholesale trade", "factor": 0.12, "naics": "42"},
    "44-45": {"name": "Retail trade", "factor": 0.15, "naics": "44-45"},
    "48-49": {"name": "Transportation and warehousing", "factor": 0.52, "naics": "48-49"},
    "51": {"name": "Information", "factor": 0.08, "naics": "51"},
    "52": {"name": "Finance and insurance", "factor": 0.05, "naics": "52"},
    "53": {"name": "Real estate", "factor": 0.12, "naics": "53"},
    "54": {"name": "Professional services", "factor": 0.07, "naics": "54"},
    "55": {"name": "Management of companies", "factor": 0.06, "naics": "55"},
    "56": {"name": "Administrative services", "factor": 0.09, "naics": "56"},
    "61": {"name": "Educational services", "factor": 0.08, "naics": "61"},
    "62": {"name": "Health care", "factor": 0.11, "naics": "62"},
    "71": {"name": "Arts and entertainment", "factor": 0.14, "naics": "71"},
    "72": {"name": "Accommodation and food services", "factor": 0.25, "naics": "72"},
    "81": {"name": "Other services", "factor": 0.15, "naics": "81"},

    # Default/fallback factor
    "default": {"name": "Average across sectors", "factor": 0.40, "naics": "default"},
}


class Scope3CategoryCalculator(ABC):
    """
    Abstract base class for Scope 3 category calculators.

    All 15 category calculators inherit from this class and implement
    the calculate() method with category-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - REGULATORY COMPLIANT: GHG Protocol and SB 253 aligned

    Example:
        >>> calculator = Category01PurchasedGoodsCalculator()
        >>> input_data = PurchasedGoodsInput(
        ...     reporting_year=2024,
        ...     organization_id="ORG001",
        ...     spend_usd=1000000,
        ...     naics_code="331"
        ... )
        >>> result = calculator.calculate(input_data)
        >>> print(f"Emissions: {result.emissions_mt_co2e} MT CO2e")
    """

    # Class attributes to be overridden by subclasses
    CATEGORY_NUMBER: int = 0
    CATEGORY_NAME: str = "Base Category"
    SUPPORTED_METHODS: List[CalculationMethod] = [CalculationMethod.SPEND_BASED]

    def __init__(self):
        """Initialize the calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._eeio_factors = EPA_EEIO_FACTORS
        self.logger.info(f"Initialized {self.__class__.__name__} v1.0.0")

    @abstractmethod
    def calculate(self, input_data: Scope3CalculationInput) -> Scope3CalculationResult:
        """
        Execute the emission calculation.

        Args:
            input_data: Category-specific input data

        Returns:
            Complete calculation result with audit trail

        Raises:
            ValueError: If input validation fails
            CalculationError: If calculation fails
        """
        pass

    def _validate_method(self, method: CalculationMethod) -> None:
        """Validate that the calculation method is supported."""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Method {method} not supported for Category {self.CATEGORY_NUMBER}. "
                f"Supported methods: {[m.value for m in self.SUPPORTED_METHODS]}"
            )

    def _get_eeio_factor(
        self,
        naics_code: Optional[str] = None,
        sector_name: Optional[str] = None,
    ) -> EmissionFactorRecord:
        """
        Retrieve EPA EEIO emission factor.

        Args:
            naics_code: NAICS code (2-6 digits)
            sector_name: Sector name for lookup

        Returns:
            EmissionFactorRecord with factor details
        """
        factor_data = None

        # Try exact NAICS match first
        if naics_code:
            # Try full code, then progressively shorter prefixes
            for length in range(len(naics_code), 1, -1):
                prefix = naics_code[:length]
                if prefix in self._eeio_factors:
                    factor_data = self._eeio_factors[prefix]
                    break

        # Fallback to default
        if factor_data is None:
            factor_data = self._eeio_factors["default"]
            self.logger.warning(
                f"Using default EEIO factor for NAICS {naics_code}. "
                "Consider using more specific sector data."
            )

        return EmissionFactorRecord(
            factor_id=f"epa_eeio_{factor_data['naics']}",
            factor_value=Decimal(str(factor_data["factor"])),
            factor_unit="kg CO2e/USD",
            source=EmissionFactorSource.EPA_EEIO,
            source_uri="https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=349324",
            version="2023",
            last_updated="2023-01-01",
            uncertainty_pct=30.0,  # EEIO factors typically have 20-50% uncertainty
            data_quality_tier=DataQualityTier.TIER_3,
            geographic_scope="US",
            naics_code=factor_data["naics"],
        )

    def _calculate_spend_based(
        self,
        spend_usd: Decimal,
        emission_factor: EmissionFactorRecord,
    ) -> Decimal:
        """
        Calculate emissions using spend-based method.

        Formula: Emissions (kg CO2e) = Spend (USD) x Emission Factor (kg CO2e/USD)

        Args:
            spend_usd: Total spend in USD
            emission_factor: EEIO emission factor

        Returns:
            Emissions in kg CO2e
        """
        emissions_kg = spend_usd * emission_factor.factor_value
        return emissions_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _kg_to_metric_tons(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tons."""
        return (kg / Decimal("1000")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _generate_provenance_hash(
        self,
        category_number: int,
        emissions_kg: Decimal,
        method: CalculationMethod,
        activity_data: Dict[str, Any],
        steps: List[CalculationStep],
    ) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            category_number: Scope 3 category
            emissions_kg: Calculated emissions
            method: Calculation method used
            activity_data: Input activity data
            steps: Calculation steps

        Returns:
            SHA-256 hex digest
        """
        provenance_data = {
            "category_number": category_number,
            "emissions_kg_co2e": str(emissions_kg),
            "calculation_method": method,
            "activity_data_summary": activity_data,
            "calculation_steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "output": s.output,
                }
                for s in steps
            ],
        }
        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_result(
        self,
        emissions_kg: Decimal,
        method: CalculationMethod,
        emission_factor: EmissionFactorRecord,
        activity_data: Dict[str, Any],
        steps: List[CalculationStep],
        start_time: datetime,
        warnings: Optional[List[str]] = None,
    ) -> Scope3CalculationResult:
        """
        Create a complete calculation result.

        Args:
            emissions_kg: Calculated emissions in kg
            method: Calculation method used
            emission_factor: Factor applied
            activity_data: Summary of inputs
            steps: Calculation steps
            start_time: Calculation start time
            warnings: Any warning messages

        Returns:
            Complete Scope3CalculationResult
        """
        emissions_mt = self._kg_to_metric_tons(emissions_kg)
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        provenance_hash = self._generate_provenance_hash(
            category_number=self.CATEGORY_NUMBER,
            emissions_kg=emissions_kg,
            method=method,
            activity_data=activity_data,
            steps=steps,
        )

        return Scope3CalculationResult(
            category_number=self.CATEGORY_NUMBER,
            category_name=self.CATEGORY_NAME,
            emissions_kg_co2e=emissions_kg,
            emissions_mt_co2e=emissions_mt,
            calculation_method=method,
            emission_factor_used=emission_factor,
            activity_data_summary=activity_data,
            calculation_steps=steps,
            provenance_hash=provenance_hash,
            calculation_timestamp=end_time,
            calculation_duration_ms=duration_ms,
            data_quality_tier=emission_factor.data_quality_tier,
            uncertainty_pct=emission_factor.uncertainty_pct,
            warnings=warnings or [],
        )
