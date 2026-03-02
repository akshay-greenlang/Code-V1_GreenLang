# -*- coding: utf-8 -*-
"""
SiteSpecificCalculatorEngine - Site-specific processing emissions calculator.

This module implements the SiteSpecificCalculatorEngine for AGENT-MRV-023
(Processing of Sold Products, GHG Protocol Scope 3 Category 10). It calculates
emissions from downstream processing of intermediate products using customer-
provided, site-specific data -- the highest-quality data tier per GHG Protocol.

GHG Protocol Scope 3 Category 10 requires companies to account for emissions
from the processing of sold intermediate products at downstream customer
facilities. Site-specific data is the preferred calculation approach because
it reflects actual processing conditions rather than industry averages.

Three Site-Specific Calculation Methods:
    1. Direct Method: Customer reports processing emissions per unit directly.
       E = SUM(Quantity_i * EF_customer_i)
    2. Energy-Based Method: Customer reports energy consumption per unit.
       E = SUM(Quantity_i * EnergyPerUnit_i * GridEF_region)
    3. Fuel-Based Method: Customer reports fuel consumption per unit.
       E = SUM(Quantity_i * FuelPerUnit_i * FuelEF_type)

Data Quality Indicators (DQI):
    Site-specific methods receive the highest DQI scores because they use
    primary data from the actual processing facility:
    - Direct: 90/100 (primary emissions data)
    - Energy-based: 80/100 (primary energy, secondary grid EF)
    - Fuel-based: 75/100 (primary fuel, secondary combustion EF)

Uncertainty Ranges (95% CI half-width):
    - Direct: +/-10% (lowest uncertainty, customer-reported)
    - Energy-based: +/-15% (moderate, grid EF adds uncertainty)
    - Fuel-based: +/-20% (higher, fuel mix and combustion variability)

Thread Safety:
    Uses the __new__ singleton pattern with threading.RLock for thread-safe
    instantiation. All mutable state is protected by locks.

Example:
    >>> from greenlang.processing_sold_products.site_specific_calculator import (
    ...     SiteSpecificCalculatorEngine,
    ... )
    >>> engine = SiteSpecificCalculatorEngine()
    >>> products = [
    ...     {
    ...         "product_id": "STEEL-001",
    ...         "product_name": "Hot-rolled steel coil",
    ...         "category": "METALS_FERROUS",
    ...         "quantity_tonnes": "500",
    ...         "customer_ef": "265.5",
    ...         "customer_id": "CUST-AUTO-01",
    ...         "country": "US",
    ...     }
    ... ]
    >>> result = engine.calculate_direct(products, "ORG-001", 2024)
    >>> result.total_co2e
    Decimal('132750.00000000')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-010
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic import ConfigDict

from greenlang.processing_sold_products.processing_database import (
    ProcessingDatabaseEngine,
    get_database_engine,
    calculate_provenance_hash,
    ProductCategory,
    ProcessingType,
    GridRegion,
    FuelType,
    EFSource,
    GRID_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    PRODUCT_CATEGORY_EFS,
    PROCESSING_ENERGY_INTENSITIES,
    PRODUCT_PROCESSING_COMPATIBILITY,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")

# Agent metadata
AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

# Calculation method identifiers
METHOD_DIRECT: str = "site_specific_direct"
METHOD_ENERGY: str = "site_specific_energy"
METHOD_FUEL: str = "site_specific_fuel"

# Data Quality Indicator base scores (out of 100)
DQI_BASE_SCORES: Dict[str, Decimal] = {
    METHOD_DIRECT: Decimal("90"),
    METHOD_ENERGY: Decimal("80"),
    METHOD_FUEL: Decimal("75"),
}

# Uncertainty half-widths (fraction of central estimate, 95% CI)
UNCERTAINTY_FRACTIONS: Dict[str, Decimal] = {
    METHOD_DIRECT: Decimal("0.10"),
    METHOD_ENERGY: Decimal("0.15"),
    METHOD_FUEL: Decimal("0.20"),
}

# DQI dimension weights (sum to 1.0) - GHG Protocol Scope 3 guidance
DQI_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "representativeness": Decimal("0.30"),
    "completeness": Decimal("0.25"),
    "temporal": Decimal("0.15"),
    "geographical": Decimal("0.15"),
    "technological": Decimal("0.15"),
}

# DQI dimension default scores by method (1-5 scale, 5 = best)
DQI_DIMENSION_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    METHOD_DIRECT: {
        "representativeness": Decimal("5"),
        "completeness": Decimal("4"),
        "temporal": Decimal("5"),
        "geographical": Decimal("5"),
        "technological": Decimal("4"),
    },
    METHOD_ENERGY: {
        "representativeness": Decimal("4"),
        "completeness": Decimal("4"),
        "temporal": Decimal("4"),
        "geographical": Decimal("4"),
        "technological": Decimal("4"),
    },
    METHOD_FUEL: {
        "representativeness": Decimal("4"),
        "completeness": Decimal("3"),
        "temporal": Decimal("4"),
        "geographical": Decimal("4"),
        "technological": Decimal("3"),
    },
}

# Country code to grid region mapping (expanded beyond the 16 core regions)
COUNTRY_TO_GRID_REGION: Dict[str, str] = {
    "US": "US",
    "GB": "GB",
    "UK": "GB",
    "DE": "DE",
    "FR": "FR",
    "CN": "CN",
    "IN": "IN",
    "JP": "JP",
    "KR": "KR",
    "BR": "BR",
    "CA": "CA",
    "AU": "AU",
    "MX": "MX",
    "IT": "IT",
    "ES": "ES",
    "PL": "PL",
}

# Country code to fuel type hint mapping (default industrial fuel by country)
COUNTRY_DEFAULT_FUEL: Dict[str, str] = {
    "US": "NATURAL_GAS",
    "GB": "NATURAL_GAS",
    "DE": "NATURAL_GAS",
    "FR": "NATURAL_GAS",
    "CN": "COAL",
    "IN": "COAL",
    "JP": "NATURAL_GAS",
    "KR": "NATURAL_GAS",
    "BR": "NATURAL_GAS",
    "CA": "NATURAL_GAS",
    "AU": "COAL",
    "MX": "NATURAL_GAS",
    "IT": "NATURAL_GAS",
    "ES": "NATURAL_GAS",
    "PL": "COAL",
}


# =============================================================================
# RESULT MODELS
# =============================================================================


class ProductBreakdown(BaseModel):
    """
    Emission breakdown for a single product in the calculation.

    Contains the per-product emissions, input parameters used, and
    the provenance hash for audit trail verification.
    """

    product_id: str = Field(
        ..., description="Unique product identifier"
    )
    product_name: str = Field(
        default="", description="Human-readable product name"
    )
    category: str = Field(
        ..., description="Product category (e.g., METALS_FERROUS)"
    )
    quantity_tonnes: Decimal = Field(
        ..., description="Quantity in metric tonnes"
    )
    emission_factor: Decimal = Field(
        ..., description="Applied emission factor (kgCO2e per tonne)"
    )
    emissions_kg_co2e: Decimal = Field(
        ..., description="Calculated emissions (kgCO2e)"
    )
    ef_source: str = Field(
        ..., description="Emission factor data source"
    )
    method: str = Field(
        ..., description="Calculation method used"
    )
    customer_id: Optional[str] = Field(
        default=None, description="Downstream customer identifier"
    )
    country: Optional[str] = Field(
        default=None, description="Processing country"
    )
    calculation_detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step-by-step calculation detail for audit"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """
    Data quality assessment result for a product calculation.

    Scores are on a 1-5 scale per dimension (5 = best) and a composite
    score on a 0-100 scale, following GHG Protocol Scope 3 DQI guidance.
    """

    overall_score: Decimal = Field(
        ..., description="Composite DQI score (0-100)"
    )
    dimensions: Dict[str, Decimal] = Field(
        ..., description="Per-dimension scores (1-5)"
    )
    classification: str = Field(
        ..., description="Quality classification (Excellent/Good/Fair/Poor)"
    )
    method: str = Field(
        ..., description="Calculation method assessed"
    )
    tier: str = Field(
        ..., description="Data quality tier (Tier 1 = best)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """
    Uncertainty quantification result for emissions calculations.

    Provides the 95% confidence interval around the central estimate
    using IPCC Tier 2 default uncertainty ranges.
    """

    central_estimate: Decimal = Field(
        ..., description="Central emissions estimate (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="95% CI lower bound (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="95% CI upper bound (kgCO2e)"
    )
    half_width_fraction: Decimal = Field(
        ..., description="Half-width as fraction of central estimate"
    )
    method: str = Field(
        ..., description="Calculation method"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"),
        description="Confidence level"
    )

    model_config = ConfigDict(frozen=True)


class CalculationResult(BaseModel):
    """
    Complete result from a site-specific emissions calculation.

    Contains the total emissions, per-product breakdowns, data quality
    assessment, uncertainty quantification, and provenance chain.
    """

    org_id: str = Field(
        ..., description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    method: str = Field(
        ..., description="Calculation method used"
    )
    total_co2e: Decimal = Field(
        ..., description="Total emissions (kgCO2e)"
    )
    total_co2e_tonnes: Decimal = Field(
        ..., description="Total emissions (tCO2e)"
    )
    product_count: int = Field(
        ..., description="Number of products calculated"
    )
    product_breakdowns: List[ProductBreakdown] = Field(
        ..., description="Per-product emission breakdowns"
    )
    dqi_score: DataQualityScore = Field(
        ..., description="Data quality assessment"
    )
    uncertainty: UncertaintyResult = Field(
        ..., description="Uncertainty quantification"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing duration in milliseconds"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash of entire result"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent identifier"
    )
    version: str = Field(
        default=VERSION,
        description="Agent version"
    )

    model_config = ConfigDict(frozen=True)


# =============================================================================
# ENGINE CLASS
# =============================================================================


class SiteSpecificCalculatorEngine:
    """
    Thread-safe singleton engine for site-specific processing emissions.

    Implements three site-specific calculation methods for GHG Protocol
    Scope 3 Category 10 (Processing of Sold Products):

    1. Direct Method: Customer-reported processing emissions per unit.
       Formula: E = SUM(Quantity_i * EF_customer_i)
       DQI: 90/100, Uncertainty: +/-10%

    2. Energy-Based Method: Customer-reported energy consumption.
       Formula: E = SUM(Quantity_i * EnergyPerUnit_i * GridEF_region)
       DQI: 80/100, Uncertainty: +/-15%

    3. Fuel-Based Method: Customer-reported fuel consumption.
       Formula: E = SUM(Quantity_i * FuelPerUnit_i * FuelEF_type)
       DQI: 75/100, Uncertainty: +/-20%

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python Decimal arithmetic. No LLM calls are used for any numeric
    computation. Grid and fuel emission factors are retrieved from the
    ProcessingDatabaseEngine singleton.

    Thread Safety:
        Uses __new__ singleton pattern with threading.RLock. All mutable
        state (calculation_count) is protected by a dedicated lock.

    Attributes:
        _db_engine: ProcessingDatabaseEngine for EF lookups.
        _calculation_count: Total calculations performed.

    Example:
        >>> engine = SiteSpecificCalculatorEngine()
        >>> products = [{
        ...     "product_id": "STEEL-001",
        ...     "category": "METALS_FERROUS",
        ...     "quantity_tonnes": "500",
        ...     "customer_ef": "265.5",
        ...     "country": "US",
        ... }]
        >>> result = engine.calculate_direct(products, "ORG-001", 2024)
        >>> result.total_co2e
        Decimal('132750.00000000')
    """

    _instance: Optional["SiteSpecificCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "SiteSpecificCalculatorEngine":
        """Thread-safe singleton instantiation using double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the site-specific calculator engine (once only)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._db_engine: ProcessingDatabaseEngine = get_database_engine()
        self._calculation_count: int = 0
        self._calc_lock: threading.RLock = threading.RLock()

        logger.info(
            "SiteSpecificCalculatorEngine initialized: "
            "methods=[direct, energy, fuel], "
            "dqi_scores={direct=90, energy=80, fuel=75}, "
            "uncertainty={direct=10%%, energy=15%%, fuel=20%%}",
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_calculation(self) -> None:
        """Increment the calculation counter in a thread-safe manner."""
        with self._calc_lock:
            self._calculation_count += 1

    def _quantize(self, value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
        """
        Quantize a Decimal value to the specified precision with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision (default 8 decimal places).

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    def _safe_decimal(self, value: Any, field_name: str) -> Decimal:
        """
        Safely convert a value to Decimal with validation.

        Args:
            value: Value to convert (string, int, float, or Decimal).
            field_name: Field name for error messaging.

        Returns:
            Validated Decimal value.

        Raises:
            ValueError: If conversion fails or value is invalid.
        """
        if value is None:
            raise ValueError(
                f"Field '{field_name}' is required but was None"
            )

        if isinstance(value, Decimal):
            return value

        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                f"Field '{field_name}' cannot be converted to Decimal: "
                f"'{value}' ({type(value).__name__}). Error: {exc}"
            ) from exc

        if result.is_nan() or result.is_infinite():
            raise ValueError(
                f"Field '{field_name}' must be a finite number, got '{value}'"
            )

        return result

    def _validate_product_common(self, product: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Validate common product fields shared across all methods.

        Args:
            product: Raw product dictionary.
            index: Product index for error messaging.

        Returns:
            Validated product dictionary with normalized values.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Product ID
        product_id = product.get("product_id")
        if not product_id:
            raise ValueError(
                f"Product[{index}]: 'product_id' is required"
            )
        product_id = str(product_id).strip()

        # Product name (optional)
        product_name = str(product.get("product_name", "")).strip()

        # Category
        category = product.get("category")
        if not category:
            raise ValueError(
                f"Product[{index}] (id={product_id}): 'category' is required"
            )
        category = str(category).strip().upper()
        if category not in PRODUCT_CATEGORY_EFS:
            available = sorted(PRODUCT_CATEGORY_EFS.keys())
            raise ValueError(
                f"Product[{index}] (id={product_id}): Unknown category "
                f"'{category}'. Available: {available}"
            )

        # Quantity
        quantity = self._safe_decimal(
            product.get("quantity_tonnes"),
            f"Product[{index}].quantity_tonnes",
        )
        if quantity <= Decimal("0"):
            raise ValueError(
                f"Product[{index}] (id={product_id}): "
                f"quantity_tonnes must be positive, got {quantity}"
            )

        # Country (optional, used for grid/fuel EF resolution)
        country = product.get("country")
        if country is not None:
            country = str(country).strip().upper()

        # Customer ID (optional)
        customer_id = product.get("customer_id")
        if customer_id is not None:
            customer_id = str(customer_id).strip()

        return {
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "quantity_tonnes": quantity,
            "country": country,
            "customer_id": customer_id,
        }

    def _resolve_grid_ef(self, country: Optional[str]) -> Decimal:
        """
        Resolve the grid emission factor for a country code.

        Maps the country code to a grid region, then retrieves the
        emission factor from the ProcessingDatabaseEngine. Falls back
        to the GLOBAL factor if the country is not mapped.

        Args:
            country: ISO 3166-1 alpha-2 country code, or None.

        Returns:
            Grid emission factor in kgCO2e per kWh.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> engine._resolve_grid_ef("US")
            Decimal('0.41700000')
            >>> engine._resolve_grid_ef("ZZ")
            Decimal('0.47500000')
        """
        if country is None:
            region = "GLOBAL"
        else:
            region = COUNTRY_TO_GRID_REGION.get(country.upper(), "GLOBAL")

        ef = self._db_engine.get_grid_ef(region)

        logger.debug(
            "Resolved grid EF: country=%s, region=%s, ef=%s kgCO2e/kWh",
            country, region, ef,
        )

        return ef

    def _resolve_fuel_ef(self, fuel_type: Optional[str]) -> Decimal:
        """
        Resolve the fuel combustion emission factor.

        If fuel_type is provided, retrieves it from the database engine.
        Falls back to NATURAL_GAS if fuel_type is None.

        Args:
            fuel_type: Fuel type identifier, or None.

        Returns:
            Fuel emission factor in kgCO2e per unit.

        Raises:
            ValueError: If fuel_type is not recognized.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> engine._resolve_fuel_ef("DIESEL")
            Decimal('2.70600000')
            >>> engine._resolve_fuel_ef(None)
            Decimal('2.02400000')
        """
        if fuel_type is None:
            fuel_type = "NATURAL_GAS"
        else:
            fuel_type = str(fuel_type).strip().upper()

        ef = self._db_engine.get_fuel_ef(fuel_type)

        logger.debug(
            "Resolved fuel EF: type=%s, ef=%s kgCO2e/unit",
            fuel_type, ef,
        )

        return ef

    def _build_provenance(
        self,
        method: str,
        inputs: Any,
        result: Any,
    ) -> str:
        """
        Build a SHA-256 provenance hash from method, inputs, and result.

        The provenance hash ensures audit trail integrity by creating a
        deterministic fingerprint of the calculation.

        Args:
            method: Calculation method identifier.
            inputs: Input data (dict, list, or Pydantic model).
            result: Calculation result (Decimal, dict, or Pydantic model).

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        hash_parts = [
            str(method),
            json.dumps(inputs, sort_keys=True, default=str),
            json.dumps(result, sort_keys=True, default=str) if isinstance(result, dict)
            else str(result),
            datetime.now(timezone.utc).isoformat(),
            AGENT_ID,
            VERSION,
        ]
        combined = "|".join(hash_parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # =========================================================================
    # DATA QUALITY INDICATOR
    # =========================================================================

    def compute_dqi_score(
        self,
        product: Dict[str, Any],
        method: str,
    ) -> DataQualityScore:
        """
        Compute the Data Quality Indicator score for a product calculation.

        Scores are based on GHG Protocol Scope 3 DQI guidance. Site-specific
        methods receive high scores because they use primary data. The
        composite score is a weighted average of 5 dimensions scaled to 0-100.

        Dimensions (1-5 scale, 5 = best):
            - Representativeness: How well data represents actual activity
            - Completeness: Fraction of data coverage
            - Temporal: Temporal correlation to reporting year
            - Geographical: Geographical correlation to activity location
            - Technological: Technological correlation to actual process

        Args:
            product: Product data dictionary.
            method: Calculation method identifier.

        Returns:
            DataQualityScore with composite and per-dimension scores.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> dqi = engine.compute_dqi_score(
            ...     {"product_id": "P1", "country": "US"},
            ...     "site_specific_direct",
            ... )
            >>> dqi.overall_score
            Decimal('90.00000000')
        """
        # Get dimension defaults for the method
        defaults = DQI_DIMENSION_DEFAULTS.get(method, DQI_DIMENSION_DEFAULTS[METHOD_DIRECT])
        dimensions: Dict[str, Decimal] = {}

        # Calculate per-dimension scores with adjustments
        for dim, weight in DQI_DIMENSION_WEIGHTS.items():
            base_score = defaults.get(dim, Decimal("3"))

            # Adjust geographical score if country is provided
            if dim == "geographical":
                country = product.get("country")
                if country and country.upper() in COUNTRY_TO_GRID_REGION:
                    base_score = min(base_score + Decimal("1"), Decimal("5"))
                elif country is None:
                    base_score = max(base_score - Decimal("1"), Decimal("1"))

            # Adjust completeness if key fields are present
            if dim == "completeness":
                has_customer_id = bool(product.get("customer_id"))
                has_product_name = bool(product.get("product_name"))
                if has_customer_id and has_product_name:
                    base_score = min(base_score + Decimal("1"), Decimal("5"))

            dimensions[dim] = self._quantize(base_score)

        # Compute weighted composite score (on 1-5 scale)
        composite_1_5 = Decimal("0")
        for dim, weight in DQI_DIMENSION_WEIGHTS.items():
            composite_1_5 += dimensions[dim] * weight
        composite_1_5 = self._quantize(composite_1_5)

        # Scale to 0-100
        overall = self._quantize(composite_1_5 * Decimal("20"))

        # Override with the method-specific base score as a floor
        base_dqi = DQI_BASE_SCORES.get(method, Decimal("70"))
        if overall < base_dqi:
            overall = self._quantize(base_dqi)

        # Classify
        if overall >= Decimal("85"):
            classification = "Excellent"
        elif overall >= Decimal("70"):
            classification = "Good"
        elif overall >= Decimal("50"):
            classification = "Fair"
        else:
            classification = "Poor"

        # Determine tier
        if overall >= Decimal("80"):
            tier = "Tier 1"
        elif overall >= Decimal("60"):
            tier = "Tier 2"
        else:
            tier = "Tier 3"

        return DataQualityScore(
            overall_score=overall,
            dimensions=dimensions,
            classification=classification,
            method=method,
            tier=tier,
        )

    # =========================================================================
    # UNCERTAINTY QUANTIFICATION
    # =========================================================================

    def compute_uncertainty(
        self,
        emissions: Decimal,
        method: str,
    ) -> UncertaintyResult:
        """
        Compute the uncertainty range for emissions using IPCC Tier 2 defaults.

        The uncertainty is expressed as the half-width of the 95% confidence
        interval as a fraction of the central estimate. Site-specific methods
        have narrower uncertainty ranges than average-data or spend-based.

        Args:
            emissions: Central emissions estimate in kgCO2e.
            method: Calculation method identifier.

        Returns:
            UncertaintyResult with confidence interval bounds.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> unc = engine.compute_uncertainty(Decimal("1000"), "site_specific_direct")
            >>> unc.ci_lower
            Decimal('900.00000000')
            >>> unc.ci_upper
            Decimal('1100.00000000')
        """
        fraction = UNCERTAINTY_FRACTIONS.get(method, Decimal("0.20"))

        half_width = emissions * fraction
        ci_lower = self._quantize(emissions - half_width)
        ci_upper = self._quantize(emissions + half_width)

        # Floor at zero for the lower bound
        if ci_lower < Decimal("0"):
            ci_lower = Decimal("0").quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        return UncertaintyResult(
            central_estimate=self._quantize(emissions),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            half_width_fraction=self._quantize(fraction),
            method=method,
            confidence_level=Decimal("0.95"),
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_site_specific_data(
        self,
        products: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Validate site-specific product data and return a list of warnings.

        This method performs non-blocking validation: it does not raise
        exceptions but instead collects warnings about data quality issues,
        missing optional fields, and compatibility concerns.

        Checks performed:
            - Product list is non-empty
            - Each product has required fields (product_id, category, quantity_tonnes)
            - Quantity is positive and reasonable (< 10,000,000 tonnes)
            - Category is recognized
            - Country code is mapped to a grid region
            - Customer EF is non-negative (if provided)
            - Energy and fuel consumption are non-negative (if provided)
            - Processing type is compatible with category (if provided)
            - No duplicate product IDs

        Args:
            products: List of product data dictionaries.

        Returns:
            List of warning message strings. Empty list means no warnings.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> warnings = engine.validate_site_specific_data([
            ...     {"product_id": "P1", "category": "METALS_FERROUS",
            ...      "quantity_tonnes": "500"}
            ... ])
            >>> len(warnings)
            0
        """
        warnings: List[str] = []

        if not products:
            warnings.append("Product list is empty; no emissions to calculate.")
            return warnings

        seen_ids: set = set()

        for idx, product in enumerate(products):
            prefix = f"Product[{idx}]"

            # Product ID
            pid = product.get("product_id")
            if not pid:
                warnings.append(f"{prefix}: Missing 'product_id'.")
            else:
                pid_str = str(pid).strip()
                if pid_str in seen_ids:
                    warnings.append(
                        f"{prefix}: Duplicate product_id '{pid_str}'."
                    )
                seen_ids.add(pid_str)

            # Category
            category = product.get("category")
            if not category:
                warnings.append(f"{prefix}: Missing 'category'.")
            else:
                cat = str(category).strip().upper()
                if cat not in PRODUCT_CATEGORY_EFS:
                    available = sorted(PRODUCT_CATEGORY_EFS.keys())
                    warnings.append(
                        f"{prefix}: Unknown category '{cat}'. "
                        f"Available: {available}"
                    )

            # Quantity
            qty_raw = product.get("quantity_tonnes")
            if qty_raw is None:
                warnings.append(f"{prefix}: Missing 'quantity_tonnes'.")
            else:
                try:
                    qty = Decimal(str(qty_raw))
                    if qty <= Decimal("0"):
                        warnings.append(
                            f"{prefix}: quantity_tonnes must be positive, got {qty}."
                        )
                    elif qty > Decimal("10000000"):
                        warnings.append(
                            f"{prefix}: quantity_tonnes={qty} exceeds 10M tonnes; "
                            "verify this is correct."
                        )
                except (InvalidOperation, ValueError):
                    warnings.append(
                        f"{prefix}: Cannot parse quantity_tonnes='{qty_raw}' as number."
                    )

            # Country
            country = product.get("country")
            if country:
                cc = str(country).strip().upper()
                if cc not in COUNTRY_TO_GRID_REGION:
                    warnings.append(
                        f"{prefix}: Country '{cc}' not mapped to a grid region; "
                        "GLOBAL default will be used."
                    )

            # Customer EF (for direct method)
            customer_ef = product.get("customer_ef")
            if customer_ef is not None:
                try:
                    ef = Decimal(str(customer_ef))
                    if ef < Decimal("0"):
                        warnings.append(
                            f"{prefix}: customer_ef cannot be negative, got {ef}."
                        )
                    elif ef == Decimal("0"):
                        warnings.append(
                            f"{prefix}: customer_ef is zero; emissions will be zero."
                        )
                except (InvalidOperation, ValueError):
                    warnings.append(
                        f"{prefix}: Cannot parse customer_ef='{customer_ef}' as number."
                    )

            # Energy per unit (for energy method)
            energy_per_unit = product.get("energy_per_unit_kwh")
            if energy_per_unit is not None:
                try:
                    ep = Decimal(str(energy_per_unit))
                    if ep < Decimal("0"):
                        warnings.append(
                            f"{prefix}: energy_per_unit_kwh cannot be negative, got {ep}."
                        )
                except (InvalidOperation, ValueError):
                    warnings.append(
                        f"{prefix}: Cannot parse energy_per_unit_kwh='{energy_per_unit}'."
                    )

            # Fuel per unit (for fuel method)
            fuel_per_unit = product.get("fuel_per_unit")
            if fuel_per_unit is not None:
                try:
                    fp = Decimal(str(fuel_per_unit))
                    if fp < Decimal("0"):
                        warnings.append(
                            f"{prefix}: fuel_per_unit cannot be negative, got {fp}."
                        )
                except (InvalidOperation, ValueError):
                    warnings.append(
                        f"{prefix}: Cannot parse fuel_per_unit='{fuel_per_unit}'."
                    )

            # Processing type compatibility
            proc_type = product.get("processing_type")
            if proc_type and category:
                pt = str(proc_type).strip().upper()
                cat_str = str(category).strip().upper()
                if pt in PROCESSING_ENERGY_INTENSITIES and cat_str in PRODUCT_PROCESSING_COMPATIBILITY:
                    compatible = PRODUCT_PROCESSING_COMPATIBILITY[cat_str]
                    if pt not in compatible:
                        warnings.append(
                            f"{prefix}: Processing type '{pt}' is not standard "
                            f"for category '{cat_str}'."
                        )

        return warnings

    # =========================================================================
    # PER-PRODUCT CALCULATION: DIRECT METHOD
    # =========================================================================

    def calculate_product_emissions_direct(
        self,
        product: Dict[str, Any],
    ) -> ProductBreakdown:
        """
        Calculate emissions for a single product using the direct method.

        Formula: emissions = quantity_tonnes * customer_ef

        The customer_ef is the emission factor reported by the downstream
        customer in kgCO2e per tonne of processed product. This is the
        highest-quality data source for Category 10 calculations.

        Args:
            product: Validated product dictionary with keys:
                - product_id (str)
                - product_name (str, optional)
                - category (str)
                - quantity_tonnes (Decimal or str)
                - customer_ef (Decimal or str) - kgCO2e per tonne
                - customer_id (str, optional)
                - country (str, optional)

        Returns:
            ProductBreakdown with calculated emissions.

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> breakdown = engine.calculate_product_emissions_direct({
            ...     "product_id": "STEEL-001",
            ...     "category": "METALS_FERROUS",
            ...     "quantity_tonnes": "500",
            ...     "customer_ef": "265.5",
            ... })
            >>> breakdown.emissions_kg_co2e
            Decimal('132750.00000000')
        """
        # Extract and validate fields
        product_id = str(product["product_id"]).strip()
        product_name = str(product.get("product_name", "")).strip()
        category = str(product["category"]).strip().upper()
        quantity = self._safe_decimal(product["quantity_tonnes"], "quantity_tonnes")
        customer_ef = self._safe_decimal(product.get("customer_ef"), "customer_ef")
        customer_id = product.get("customer_id")
        country = product.get("country")

        if quantity <= Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': quantity_tonnes must be positive, got {quantity}"
            )
        if customer_ef < Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': customer_ef must be non-negative, got {customer_ef}"
            )

        # Calculate: E = Q * EF_customer
        emissions = self._quantize(quantity * customer_ef)

        # Build calculation detail for audit
        calc_detail = {
            "formula": "E = quantity_tonnes * customer_ef",
            "quantity_tonnes": str(quantity),
            "customer_ef_kgco2e_per_tonne": str(customer_ef),
            "emissions_kgco2e": str(emissions),
            "step_1": f"E = {quantity} * {customer_ef}",
            "step_2": f"E = {emissions} kgCO2e",
        }

        # Provenance
        provenance = calculate_provenance_hash(
            product_id, category, quantity, customer_ef, emissions, METHOD_DIRECT,
        )

        logger.debug(
            "Direct product calc: id=%s, Q=%s t, EF=%s kgCO2e/t, E=%s kgCO2e",
            product_id, quantity, customer_ef, emissions,
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=self._quantize(quantity),
            emission_factor=self._quantize(customer_ef),
            emissions_kg_co2e=emissions,
            ef_source=EFSource.CUSTOMER.value,
            method=METHOD_DIRECT,
            customer_id=customer_id,
            country=country,
            calculation_detail=calc_detail,
            provenance_hash=provenance,
        )

    # =========================================================================
    # PER-PRODUCT CALCULATION: ENERGY-BASED METHOD
    # =========================================================================

    def calculate_product_emissions_energy(
        self,
        product: Dict[str, Any],
        grid_ef: Decimal,
    ) -> ProductBreakdown:
        """
        Calculate emissions for a single product using the energy-based method.

        Formula: emissions = quantity_tonnes * energy_per_unit_kwh * grid_ef

        The energy_per_unit_kwh is the electricity consumption per tonne
        reported by the downstream customer. The grid_ef is the regional
        grid emission factor resolved from the customer's country.

        Args:
            product: Validated product dictionary with keys:
                - product_id (str)
                - product_name (str, optional)
                - category (str)
                - quantity_tonnes (Decimal or str)
                - energy_per_unit_kwh (Decimal or str) - kWh per tonne
                - customer_id (str, optional)
                - country (str, optional)
            grid_ef: Grid emission factor in kgCO2e per kWh.

        Returns:
            ProductBreakdown with calculated emissions.

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> grid_ef = Decimal("0.417")
            >>> breakdown = engine.calculate_product_emissions_energy(
            ...     {
            ...         "product_id": "STEEL-002",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "energy_per_unit_kwh": "300",
            ...     },
            ...     grid_ef,
            ... )
            >>> breakdown.emissions_kg_co2e
            Decimal('62550.00000000')
        """
        # Extract and validate fields
        product_id = str(product["product_id"]).strip()
        product_name = str(product.get("product_name", "")).strip()
        category = str(product["category"]).strip().upper()
        quantity = self._safe_decimal(product["quantity_tonnes"], "quantity_tonnes")
        energy_per_unit = self._safe_decimal(
            product.get("energy_per_unit_kwh"), "energy_per_unit_kwh"
        )
        customer_id = product.get("customer_id")
        country = product.get("country")

        if quantity <= Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': quantity_tonnes must be positive, got {quantity}"
            )
        if energy_per_unit < Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': energy_per_unit_kwh must be non-negative, "
                f"got {energy_per_unit}"
            )

        # Calculate: E = Q * EP * GridEF
        emissions = self._quantize(quantity * energy_per_unit * grid_ef)

        # The effective EF for reporting is energy_per_unit * grid_ef
        effective_ef = self._quantize(energy_per_unit * grid_ef)

        # Build calculation detail for audit
        calc_detail = {
            "formula": "E = quantity_tonnes * energy_per_unit_kwh * grid_ef",
            "quantity_tonnes": str(quantity),
            "energy_per_unit_kwh": str(energy_per_unit),
            "grid_ef_kgco2e_per_kwh": str(grid_ef),
            "effective_ef_kgco2e_per_tonne": str(effective_ef),
            "emissions_kgco2e": str(emissions),
            "step_1": f"effective_ef = {energy_per_unit} * {grid_ef} = {effective_ef} kgCO2e/t",
            "step_2": f"E = {quantity} * {effective_ef} = {emissions} kgCO2e",
            "grid_region": country or "GLOBAL",
        }

        # Provenance
        provenance = calculate_provenance_hash(
            product_id, category, quantity, energy_per_unit,
            grid_ef, emissions, METHOD_ENERGY,
        )

        logger.debug(
            "Energy product calc: id=%s, Q=%s t, EP=%s kWh/t, "
            "GridEF=%s kgCO2e/kWh, E=%s kgCO2e",
            product_id, quantity, energy_per_unit, grid_ef, emissions,
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=self._quantize(quantity),
            emission_factor=effective_ef,
            emissions_kg_co2e=emissions,
            ef_source=EFSource.IEA.value,
            method=METHOD_ENERGY,
            customer_id=customer_id,
            country=country,
            calculation_detail=calc_detail,
            provenance_hash=provenance,
        )

    # =========================================================================
    # PER-PRODUCT CALCULATION: FUEL-BASED METHOD
    # =========================================================================

    def calculate_product_emissions_fuel(
        self,
        product: Dict[str, Any],
        fuel_ef: Decimal,
    ) -> ProductBreakdown:
        """
        Calculate emissions for a single product using the fuel-based method.

        Formula: emissions = quantity_tonnes * fuel_per_unit * fuel_ef

        The fuel_per_unit is the fuel consumption per tonne of processed
        material reported by the downstream customer (in fuel units: m3
        for natural gas, litres for diesel/LPG, kg for coal/HFO/biomass).

        Args:
            product: Validated product dictionary with keys:
                - product_id (str)
                - product_name (str, optional)
                - category (str)
                - quantity_tonnes (Decimal or str)
                - fuel_per_unit (Decimal or str) - fuel units per tonne
                - fuel_type (str, optional) - defaults to NATURAL_GAS
                - customer_id (str, optional)
                - country (str, optional)
            fuel_ef: Fuel emission factor in kgCO2e per fuel unit.

        Returns:
            ProductBreakdown with calculated emissions.

        Raises:
            ValueError: If required fields are missing or invalid.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> fuel_ef = Decimal("2.024")  # natural gas
            >>> breakdown = engine.calculate_product_emissions_fuel(
            ...     {
            ...         "product_id": "STEEL-003",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "fuel_per_unit": "15",
            ...         "fuel_type": "NATURAL_GAS",
            ...     },
            ...     fuel_ef,
            ... )
            >>> breakdown.emissions_kg_co2e
            Decimal('15180.00000000')
        """
        # Extract and validate fields
        product_id = str(product["product_id"]).strip()
        product_name = str(product.get("product_name", "")).strip()
        category = str(product["category"]).strip().upper()
        quantity = self._safe_decimal(product["quantity_tonnes"], "quantity_tonnes")
        fuel_per_unit = self._safe_decimal(
            product.get("fuel_per_unit"), "fuel_per_unit"
        )
        fuel_type = product.get("fuel_type", "NATURAL_GAS")
        if fuel_type is not None:
            fuel_type = str(fuel_type).strip().upper()
        customer_id = product.get("customer_id")
        country = product.get("country")

        if quantity <= Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': quantity_tonnes must be positive, got {quantity}"
            )
        if fuel_per_unit < Decimal("0"):
            raise ValueError(
                f"Product '{product_id}': fuel_per_unit must be non-negative, "
                f"got {fuel_per_unit}"
            )

        # Calculate: E = Q * FP * FuelEF
        emissions = self._quantize(quantity * fuel_per_unit * fuel_ef)

        # The effective EF for reporting is fuel_per_unit * fuel_ef
        effective_ef = self._quantize(fuel_per_unit * fuel_ef)

        # Build calculation detail for audit
        calc_detail = {
            "formula": "E = quantity_tonnes * fuel_per_unit * fuel_ef",
            "quantity_tonnes": str(quantity),
            "fuel_per_unit": str(fuel_per_unit),
            "fuel_type": fuel_type,
            "fuel_ef_kgco2e_per_unit": str(fuel_ef),
            "effective_ef_kgco2e_per_tonne": str(effective_ef),
            "emissions_kgco2e": str(emissions),
            "step_1": f"effective_ef = {fuel_per_unit} * {fuel_ef} = {effective_ef} kgCO2e/t",
            "step_2": f"E = {quantity} * {effective_ef} = {emissions} kgCO2e",
        }

        # Provenance
        provenance = calculate_provenance_hash(
            product_id, category, quantity, fuel_per_unit,
            fuel_ef, emissions, METHOD_FUEL,
        )

        logger.debug(
            "Fuel product calc: id=%s, Q=%s t, FP=%s units/t, "
            "FuelEF=%s kgCO2e/unit, E=%s kgCO2e",
            product_id, quantity, fuel_per_unit, fuel_ef, emissions,
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=self._quantize(quantity),
            emission_factor=effective_ef,
            emissions_kg_co2e=emissions,
            ef_source=EFSource.DEFRA.value,
            method=METHOD_FUEL,
            customer_id=customer_id,
            country=country,
            calculation_detail=calc_detail,
            provenance_hash=provenance,
        )

    # =========================================================================
    # AGGREGATE CALCULATION: DIRECT METHOD
    # =========================================================================

    def calculate_direct(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """
        Calculate total emissions using customer-reported emission factors.

        This is the highest-quality site-specific method. Each product must
        include a customer_ef field representing the customer's reported
        processing emission factor in kgCO2e per tonne.

        Formula per product: E_i = Q_i * EF_customer_i
        Total: E_total = SUM(E_i)

        Args:
            products: List of product dictionaries, each containing:
                - product_id (str, required)
                - product_name (str, optional)
                - category (str, required) - ProductCategory value
                - quantity_tonnes (str/Decimal, required) - positive
                - customer_ef (str/Decimal, required) - kgCO2e/tonne
                - customer_id (str, optional)
                - country (str, optional)
            org_id: Organization identifier for the reporting company.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult with total emissions, breakdowns, DQI, uncertainty.

        Raises:
            ValueError: If products list is empty or has invalid data.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> result = engine.calculate_direct(
            ...     [{
            ...         "product_id": "STEEL-001",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "customer_ef": "265.5",
            ...     }],
            ...     "ORG-001",
            ...     2024,
            ... )
            >>> result.total_co2e
            Decimal('132750.00000000')
        """
        start_time = time.monotonic()
        self._increment_calculation()

        if not products:
            raise ValueError("Products list must not be empty")

        logger.info(
            "Starting direct calculation: org=%s, year=%d, products=%d",
            org_id, reporting_year, len(products),
        )

        # Validate all products
        warnings = self.validate_site_specific_data(products)

        # Calculate per product
        breakdowns: List[ProductBreakdown] = []
        total_co2e = Decimal("0")

        for idx, raw_product in enumerate(products):
            try:
                validated = self._validate_product_common(raw_product, idx)

                # Ensure customer_ef is present
                customer_ef_raw = raw_product.get("customer_ef")
                if customer_ef_raw is None:
                    raise ValueError(
                        f"Product[{idx}] (id={validated['product_id']}): "
                        "'customer_ef' is required for direct method"
                    )

                product_dict = {
                    **validated,
                    "customer_ef": customer_ef_raw,
                }

                breakdown = self.calculate_product_emissions_direct(product_dict)
                breakdowns.append(breakdown)
                total_co2e += breakdown.emissions_kg_co2e

                logger.debug(
                    "Direct calc product[%d/%d]: id=%s, E=%s kgCO2e",
                    idx + 1, len(products),
                    breakdown.product_id, breakdown.emissions_kg_co2e,
                )

            except (ValueError, KeyError) as exc:
                logger.error(
                    "Direct calc product[%d] failed: %s",
                    idx, str(exc),
                )
                raise ValueError(
                    f"Direct calculation failed for product index {idx}: {exc}"
                ) from exc

        total_co2e = self._quantize(total_co2e)
        total_co2e_tonnes = self._quantize(total_co2e / Decimal("1000"))

        # DQI score (use first product as representative)
        representative = products[0] if products else {}
        dqi = self.compute_dqi_score(representative, METHOD_DIRECT)

        # Uncertainty
        uncertainty = self.compute_uncertainty(total_co2e, METHOD_DIRECT)

        # Processing time
        elapsed_ms = self._quantize(
            Decimal(str((time.monotonic() - start_time) * 1000))
        )

        # Provenance
        provenance = self._build_provenance(
            METHOD_DIRECT,
            {"org_id": org_id, "year": reporting_year, "product_count": len(products)},
            {"total_co2e": str(total_co2e)},
        )

        logger.info(
            "Direct calculation complete: org=%s, year=%d, products=%d, "
            "total=%s kgCO2e (%s tCO2e), DQI=%s, time=%s ms",
            org_id, reporting_year, len(breakdowns),
            total_co2e, total_co2e_tonnes, dqi.overall_score, elapsed_ms,
        )

        return CalculationResult(
            org_id=org_id,
            reporting_year=reporting_year,
            method=METHOD_DIRECT,
            total_co2e=total_co2e,
            total_co2e_tonnes=total_co2e_tonnes,
            product_count=len(breakdowns),
            product_breakdowns=breakdowns,
            dqi_score=dqi,
            uncertainty=uncertainty,
            processing_time_ms=elapsed_ms,
            warnings=warnings,
            provenance_hash=provenance,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # =========================================================================
    # AGGREGATE CALCULATION: ENERGY-BASED METHOD
    # =========================================================================

    def calculate_energy_based(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """
        Calculate total emissions using customer-reported energy consumption.

        Each product must include energy_per_unit_kwh (kWh per tonne of
        processed material). The grid emission factor is resolved from the
        customer's country code via the ProcessingDatabaseEngine.

        Formula per product: E_i = Q_i * EP_i * GridEF_region_i
        Total: E_total = SUM(E_i)

        Args:
            products: List of product dictionaries, each containing:
                - product_id (str, required)
                - product_name (str, optional)
                - category (str, required)
                - quantity_tonnes (str/Decimal, required)
                - energy_per_unit_kwh (str/Decimal, required) - kWh/tonne
                - customer_id (str, optional)
                - country (str, optional) - for grid EF resolution
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult with total emissions, breakdowns, DQI, uncertainty.

        Raises:
            ValueError: If products list is empty or has invalid data.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> result = engine.calculate_energy_based(
            ...     [{
            ...         "product_id": "STEEL-002",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "energy_per_unit_kwh": "300",
            ...         "country": "US",
            ...     }],
            ...     "ORG-001",
            ...     2024,
            ... )
            >>> result.total_co2e
            Decimal('62550.00000000')
        """
        start_time = time.monotonic()
        self._increment_calculation()

        if not products:
            raise ValueError("Products list must not be empty")

        logger.info(
            "Starting energy-based calculation: org=%s, year=%d, products=%d",
            org_id, reporting_year, len(products),
        )

        warnings = self.validate_site_specific_data(products)

        breakdowns: List[ProductBreakdown] = []
        total_co2e = Decimal("0")

        for idx, raw_product in enumerate(products):
            try:
                validated = self._validate_product_common(raw_product, idx)

                # Ensure energy_per_unit_kwh is present
                energy_raw = raw_product.get("energy_per_unit_kwh")
                if energy_raw is None:
                    raise ValueError(
                        f"Product[{idx}] (id={validated['product_id']}): "
                        "'energy_per_unit_kwh' is required for energy-based method"
                    )

                # Resolve grid EF for this product's country
                grid_ef = self._resolve_grid_ef(validated.get("country"))

                product_dict = {
                    **validated,
                    "energy_per_unit_kwh": energy_raw,
                }

                breakdown = self.calculate_product_emissions_energy(
                    product_dict, grid_ef
                )
                breakdowns.append(breakdown)
                total_co2e += breakdown.emissions_kg_co2e

                logger.debug(
                    "Energy calc product[%d/%d]: id=%s, GridEF=%s, E=%s kgCO2e",
                    idx + 1, len(products),
                    breakdown.product_id, grid_ef, breakdown.emissions_kg_co2e,
                )

            except (ValueError, KeyError) as exc:
                logger.error(
                    "Energy calc product[%d] failed: %s",
                    idx, str(exc),
                )
                raise ValueError(
                    f"Energy-based calculation failed for product index {idx}: {exc}"
                ) from exc

        total_co2e = self._quantize(total_co2e)
        total_co2e_tonnes = self._quantize(total_co2e / Decimal("1000"))

        representative = products[0] if products else {}
        dqi = self.compute_dqi_score(representative, METHOD_ENERGY)
        uncertainty = self.compute_uncertainty(total_co2e, METHOD_ENERGY)

        elapsed_ms = self._quantize(
            Decimal(str((time.monotonic() - start_time) * 1000))
        )

        provenance = self._build_provenance(
            METHOD_ENERGY,
            {"org_id": org_id, "year": reporting_year, "product_count": len(products)},
            {"total_co2e": str(total_co2e)},
        )

        logger.info(
            "Energy-based calculation complete: org=%s, year=%d, products=%d, "
            "total=%s kgCO2e (%s tCO2e), DQI=%s, time=%s ms",
            org_id, reporting_year, len(breakdowns),
            total_co2e, total_co2e_tonnes, dqi.overall_score, elapsed_ms,
        )

        return CalculationResult(
            org_id=org_id,
            reporting_year=reporting_year,
            method=METHOD_ENERGY,
            total_co2e=total_co2e,
            total_co2e_tonnes=total_co2e_tonnes,
            product_count=len(breakdowns),
            product_breakdowns=breakdowns,
            dqi_score=dqi,
            uncertainty=uncertainty,
            processing_time_ms=elapsed_ms,
            warnings=warnings,
            provenance_hash=provenance,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # =========================================================================
    # AGGREGATE CALCULATION: FUEL-BASED METHOD
    # =========================================================================

    def calculate_fuel_based(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """
        Calculate total emissions using customer-reported fuel consumption.

        Each product must include fuel_per_unit (fuel consumption per tonne
        of processed material in the appropriate fuel unit) and optionally
        fuel_type (defaults to NATURAL_GAS).

        Formula per product: E_i = Q_i * FP_i * FuelEF_type_i
        Total: E_total = SUM(E_i)

        Args:
            products: List of product dictionaries, each containing:
                - product_id (str, required)
                - product_name (str, optional)
                - category (str, required)
                - quantity_tonnes (str/Decimal, required)
                - fuel_per_unit (str/Decimal, required)
                - fuel_type (str, optional) - defaults to NATURAL_GAS
                - customer_id (str, optional)
                - country (str, optional)
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult with total emissions, breakdowns, DQI, uncertainty.

        Raises:
            ValueError: If products list is empty or has invalid data.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> result = engine.calculate_fuel_based(
            ...     [{
            ...         "product_id": "STEEL-003",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "fuel_per_unit": "15",
            ...         "fuel_type": "NATURAL_GAS",
            ...     }],
            ...     "ORG-001",
            ...     2024,
            ... )
            >>> result.total_co2e
            Decimal('15180.00000000')
        """
        start_time = time.monotonic()
        self._increment_calculation()

        if not products:
            raise ValueError("Products list must not be empty")

        logger.info(
            "Starting fuel-based calculation: org=%s, year=%d, products=%d",
            org_id, reporting_year, len(products),
        )

        warnings = self.validate_site_specific_data(products)

        breakdowns: List[ProductBreakdown] = []
        total_co2e = Decimal("0")

        for idx, raw_product in enumerate(products):
            try:
                validated = self._validate_product_common(raw_product, idx)

                # Ensure fuel_per_unit is present
                fuel_raw = raw_product.get("fuel_per_unit")
                if fuel_raw is None:
                    raise ValueError(
                        f"Product[{idx}] (id={validated['product_id']}): "
                        "'fuel_per_unit' is required for fuel-based method"
                    )

                # Get fuel type (default to NATURAL_GAS)
                fuel_type = raw_product.get("fuel_type", "NATURAL_GAS")

                # Resolve fuel EF
                fuel_ef = self._resolve_fuel_ef(fuel_type)

                product_dict = {
                    **validated,
                    "fuel_per_unit": fuel_raw,
                    "fuel_type": fuel_type,
                }

                breakdown = self.calculate_product_emissions_fuel(
                    product_dict, fuel_ef
                )
                breakdowns.append(breakdown)
                total_co2e += breakdown.emissions_kg_co2e

                logger.debug(
                    "Fuel calc product[%d/%d]: id=%s, FuelEF=%s, E=%s kgCO2e",
                    idx + 1, len(products),
                    breakdown.product_id, fuel_ef, breakdown.emissions_kg_co2e,
                )

            except (ValueError, KeyError) as exc:
                logger.error(
                    "Fuel calc product[%d] failed: %s",
                    idx, str(exc),
                )
                raise ValueError(
                    f"Fuel-based calculation failed for product index {idx}: {exc}"
                ) from exc

        total_co2e = self._quantize(total_co2e)
        total_co2e_tonnes = self._quantize(total_co2e / Decimal("1000"))

        representative = products[0] if products else {}
        dqi = self.compute_dqi_score(representative, METHOD_FUEL)
        uncertainty = self.compute_uncertainty(total_co2e, METHOD_FUEL)

        elapsed_ms = self._quantize(
            Decimal(str((time.monotonic() - start_time) * 1000))
        )

        provenance = self._build_provenance(
            METHOD_FUEL,
            {"org_id": org_id, "year": reporting_year, "product_count": len(products)},
            {"total_co2e": str(total_co2e)},
        )

        logger.info(
            "Fuel-based calculation complete: org=%s, year=%d, products=%d, "
            "total=%s kgCO2e (%s tCO2e), DQI=%s, time=%s ms",
            org_id, reporting_year, len(breakdowns),
            total_co2e, total_co2e_tonnes, dqi.overall_score, elapsed_ms,
        )

        return CalculationResult(
            org_id=org_id,
            reporting_year=reporting_year,
            method=METHOD_FUEL,
            total_co2e=total_co2e,
            total_co2e_tonnes=total_co2e_tonnes,
            product_count=len(breakdowns),
            product_breakdowns=breakdowns,
            dqi_score=dqi,
            uncertainty=uncertainty,
            processing_time_ms=elapsed_ms,
            warnings=warnings,
            provenance_hash=provenance,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # =========================================================================
    # METHOD DISPATCHER
    # =========================================================================

    def calculate(
        self,
        products: List[Dict[str, Any]],
        method: str,
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """
        Dispatch calculation to the appropriate site-specific method.

        This is the main entry point for the calculator engine. It routes
        the request to the correct method implementation based on the
        method parameter.

        Supported methods:
            - "site_specific_direct" or "direct": Customer-reported EFs
            - "site_specific_energy" or "energy": Energy consumption data
            - "site_specific_fuel" or "fuel": Fuel consumption data

        Args:
            products: List of product dictionaries.
            method: Calculation method identifier.
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult from the selected method.

        Raises:
            ValueError: If method is not recognized.
            ValueError: If products list is empty or invalid.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> result = engine.calculate(
            ...     [{
            ...         "product_id": "STEEL-001",
            ...         "category": "METALS_FERROUS",
            ...         "quantity_tonnes": "500",
            ...         "customer_ef": "265.5",
            ...     }],
            ...     "direct",
            ...     "ORG-001",
            ...     2024,
            ... )
            >>> result.method
            'site_specific_direct'
        """
        # Normalize method name
        method_lower = method.strip().lower()

        # Map short names to full identifiers
        method_map: Dict[str, str] = {
            "direct": METHOD_DIRECT,
            "site_specific_direct": METHOD_DIRECT,
            "energy": METHOD_ENERGY,
            "site_specific_energy": METHOD_ENERGY,
            "energy_based": METHOD_ENERGY,
            "fuel": METHOD_FUEL,
            "site_specific_fuel": METHOD_FUEL,
            "fuel_based": METHOD_FUEL,
        }

        resolved = method_map.get(method_lower)
        if resolved is None:
            available = sorted(set(method_map.values()))
            raise ValueError(
                f"Unknown calculation method '{method}'. "
                f"Available methods: {available}"
            )

        logger.info(
            "Dispatching calculation: method=%s (resolved=%s), "
            "org=%s, year=%d, products=%d",
            method, resolved, org_id, reporting_year, len(products),
        )

        if resolved == METHOD_DIRECT:
            return self.calculate_direct(products, org_id, reporting_year)
        elif resolved == METHOD_ENERGY:
            return self.calculate_energy_based(products, org_id, reporting_year)
        elif resolved == METHOD_FUEL:
            return self.calculate_fuel_based(products, org_id, reporting_year)
        else:
            # Should never reach here due to validation above
            raise ValueError(f"Internal error: unhandled method '{resolved}'")

    # =========================================================================
    # ENGINE STATUS / DIAGNOSTICS
    # =========================================================================

    @property
    def calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations since engine initialization.
        """
        with self._calc_lock:
            return self._calculation_count

    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get the current engine status for health checks and diagnostics.

        Returns:
            Dict with engine metadata and status indicators.

        Example:
            >>> engine = SiteSpecificCalculatorEngine()
            >>> status = engine.get_engine_status()
            >>> status["agent_id"]
            'GL-MRV-S3-010'
            >>> status["supported_methods"]
            ['site_specific_direct', 'site_specific_energy', 'site_specific_fuel']
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "calculation_count": self.calculation_count,
            "supported_methods": [METHOD_DIRECT, METHOD_ENERGY, METHOD_FUEL],
            "dqi_scores": {k: str(v) for k, v in DQI_BASE_SCORES.items()},
            "uncertainty_fractions": {
                k: str(v) for k, v in UNCERTAINTY_FRACTIONS.items()
            },
            "db_engine_status": self._db_engine.get_engine_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# =============================================================================


def get_site_specific_engine() -> SiteSpecificCalculatorEngine:
    """
    Get the singleton SiteSpecificCalculatorEngine instance.

    This is the recommended way to obtain the engine instance in application
    code. It ensures thread-safe singleton access.

    Returns:
        The SiteSpecificCalculatorEngine singleton instance.

    Example:
        >>> engine = get_site_specific_engine()
        >>> result = engine.calculate_direct([...], "ORG-001", 2024)
    """
    return SiteSpecificCalculatorEngine()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "METHOD_DIRECT",
    "METHOD_ENERGY",
    "METHOD_FUEL",
    "DQI_BASE_SCORES",
    "UNCERTAINTY_FRACTIONS",
    "DQI_DIMENSION_WEIGHTS",
    "DQI_DIMENSION_DEFAULTS",
    "COUNTRY_TO_GRID_REGION",
    "COUNTRY_DEFAULT_FUEL",

    # Result models
    "ProductBreakdown",
    "DataQualityScore",
    "UncertaintyResult",
    "CalculationResult",

    # Engine
    "SiteSpecificCalculatorEngine",
    "get_site_specific_engine",
]
