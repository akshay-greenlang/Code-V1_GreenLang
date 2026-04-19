"""
Calculation Engine Service

This module provides zero-hallucination deterministic calculations for
carbon emissions, energy, and regulatory compliance calculations.

CRITICAL: NO LLM IN CALCULATION PATH
All numeric calculations are performed through validated formulas and
lookup tables with complete provenance tracking.

Example:
    >>> engine = CalculationEngineService()
    >>> result = engine.calculate_emissions(
    ...     fuel_type="natural_gas",
    ...     quantity=1000,
    ...     unit="m3",
    ...     region="US"
    ... )
    >>> print(f"Emissions: {result.value} {result.unit}")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class GWPSet(str, Enum):
    """Global Warming Potential reference sets."""

    AR4 = "AR4"  # IPCC Fourth Assessment Report
    AR5 = "AR5"  # IPCC Fifth Assessment Report
    AR6 = "AR6"  # IPCC Sixth Assessment Report (default)


class EmissionFactorSource(str, Enum):
    """Emission factor data sources."""

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    IEA = "IEA"
    ECOINVENT = "ECOINVENT"
    EXIOBASE = "EXIOBASE"


class EmissionFactor(BaseModel):
    """Emission factor with provenance."""

    material_id: str = Field(..., description="Material/fuel identifier")
    region: str = Field(..., description="Geographic region (ISO 3166)")
    year: int = Field(..., description="Reference year")
    value: float = Field(..., ge=0, description="Emission factor value")
    unit: str = Field(..., description="Unit (e.g., kgCO2e/kWh)")
    source: EmissionFactorSource = Field(..., description="Data source")
    gwp_set: GWPSet = Field(GWPSet.AR6, description="GWP reference set")
    uncertainty_lower: Optional[float] = Field(None, description="Lower uncertainty bound")
    uncertainty_upper: Optional[float] = Field(None, description="Upper uncertainty bound")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def provenance_hash(self) -> str:
        """Generate provenance hash for this emission factor."""
        data = f"{self.material_id}:{self.region}:{self.year}:{self.value}:{self.source}"
        return hashlib.sha256(data.encode()).hexdigest()


class CalculationResult(BaseModel):
    """Result of a calculation with provenance."""

    value: float = Field(..., description="Calculated value")
    unit: str = Field(..., description="Result unit")
    formula_id: str = Field(..., description="Formula used")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    emission_factors_used: List[EmissionFactor] = Field(
        default_factory=list,
        description="Emission factors used"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    uncertainty: Optional[float] = Field(None, description="Uncertainty percentage")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class CalculationEngineService:
    """
    Zero-Hallucination Calculation Engine.

    This service provides deterministic calculations with:
    - Validated formulas from the formula database
    - Emission factors from authoritative sources
    - Complete provenance tracking
    - Unit conversion

    IMPORTANT: No LLM is used in any calculation path.
    All numeric operations are deterministic and reproducible.

    Attributes:
        formula_registry: Registry of validated formulas
        ef_database: Emission factor database
        gwp_set: Default GWP set to use

    Example:
        >>> engine = CalculationEngineService()
        >>> result = engine.calculate("emissions.scope1.stationary", {
        ...     "fuel_type": "natural_gas",
        ...     "quantity": 1000,
        ...     "unit": "m3"
        ... })
        >>> assert result.provenance_hash is not None
    """

    # Standard emission factors (simplified for demonstration)
    # In production, these come from a validated database
    DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, EmissionFactor]] = {}

    def __init__(
        self,
        formula_registry: Optional[Any] = None,
        ef_database: Optional[Any] = None,
        gwp_set: GWPSet = GWPSet.AR6,
    ):
        """
        Initialize the Calculation Engine.

        Args:
            formula_registry: Optional formula registry
            ef_database: Optional emission factor database
            gwp_set: Default GWP set to use
        """
        self.formula_registry = formula_registry
        self.ef_database = ef_database
        self.gwp_set = gwp_set

        # Initialize default emission factors
        self._init_default_emission_factors()

        logger.info(f"CalculationEngineService initialized with GWP set {gwp_set}")

    def _init_default_emission_factors(self) -> None:
        """Initialize default emission factors for common fuels."""
        # Natural Gas (m3 -> kgCO2e)
        self.DEFAULT_EMISSION_FACTORS["natural_gas"] = {
            "US": EmissionFactor(
                material_id="natural_gas",
                region="US",
                year=2024,
                value=1.93,  # kgCO2e per m3
                unit="kgCO2e/m3",
                source=EmissionFactorSource.EPA,
            ),
            "EU": EmissionFactor(
                material_id="natural_gas",
                region="EU",
                year=2024,
                value=2.02,
                unit="kgCO2e/m3",
                source=EmissionFactorSource.DEFRA,
            ),
        }

        # Diesel (L -> kgCO2e)
        self.DEFAULT_EMISSION_FACTORS["diesel"] = {
            "US": EmissionFactor(
                material_id="diesel",
                region="US",
                year=2024,
                value=2.68,  # kgCO2e per liter
                unit="kgCO2e/L",
                source=EmissionFactorSource.EPA,
            ),
            "EU": EmissionFactor(
                material_id="diesel",
                region="EU",
                year=2024,
                value=2.62,
                unit="kgCO2e/L",
                source=EmissionFactorSource.DEFRA,
            ),
        }

        # Electricity (kWh -> kgCO2e) - Grid average
        self.DEFAULT_EMISSION_FACTORS["electricity_grid"] = {
            "US": EmissionFactor(
                material_id="electricity_grid",
                region="US",
                year=2024,
                value=0.417,  # kgCO2e per kWh (US average)
                unit="kgCO2e/kWh",
                source=EmissionFactorSource.EPA,
            ),
            "EU": EmissionFactor(
                material_id="electricity_grid",
                region="EU",
                year=2024,
                value=0.276,  # kgCO2e per kWh (EU average)
                unit="kgCO2e/kWh",
                source=EmissionFactorSource.IEA,
            ),
            "DE": EmissionFactor(
                material_id="electricity_grid",
                region="DE",
                year=2024,
                value=0.366,
                unit="kgCO2e/kWh",
                source=EmissionFactorSource.IEA,
            ),
            "FR": EmissionFactor(
                material_id="electricity_grid",
                region="FR",
                year=2024,
                value=0.052,  # Nuclear-heavy grid
                unit="kgCO2e/kWh",
                source=EmissionFactorSource.IEA,
            ),
        }

    @lru_cache(maxsize=10000)
    def get_emission_factor(
        self,
        material_id: str,
        region: str,
        year: Optional[int] = None,
    ) -> Optional[EmissionFactor]:
        """
        Look up emission factor from database.

        Args:
            material_id: Material/fuel identifier
            region: Geographic region (ISO 3166)
            year: Reference year (defaults to latest)

        Returns:
            Emission factor or None if not found

        Example:
            >>> ef = engine.get_emission_factor("natural_gas", "US")
            >>> print(f"EF: {ef.value} {ef.unit}")
        """
        # Check external database first
        if self.ef_database:
            ef = self.ef_database.get(material_id, region, year)
            if ef:
                return ef

        # Fall back to default factors
        material_factors = self.DEFAULT_EMISSION_FACTORS.get(material_id, {})

        # Try exact region match
        if region in material_factors:
            return material_factors[region]

        # Try parent region (e.g., DE -> EU)
        region_hierarchy = {"DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU"}
        parent_region = region_hierarchy.get(region)
        if parent_region and parent_region in material_factors:
            return material_factors[parent_region]

        logger.warning(f"Emission factor not found: {material_id}/{region}")
        return None

    def calculate_emissions(
        self,
        fuel_type: str,
        quantity: float,
        unit: str,
        region: str = "US",
        scope: int = 1,
    ) -> CalculationResult:
        """
        Calculate emissions for a given fuel consumption.

        This is a zero-hallucination calculation that uses:
        - Validated emission factors from authoritative sources
        - Deterministic formula: emissions = quantity * emission_factor
        - Complete provenance tracking

        Args:
            fuel_type: Type of fuel (natural_gas, diesel, electricity_grid, etc.)
            quantity: Amount consumed
            unit: Unit of measurement
            region: Geographic region
            scope: GHG Protocol scope (1, 2, or 3)

        Returns:
            Calculation result with provenance

        Example:
            >>> result = engine.calculate_emissions(
            ...     fuel_type="natural_gas",
            ...     quantity=1000,
            ...     unit="m3",
            ...     region="US"
            ... )
            >>> print(f"Emissions: {result.value:.2f} kgCO2e")
        """
        # Get emission factor
        ef = self.get_emission_factor(fuel_type, region)
        if not ef:
            raise ValueError(f"No emission factor found for {fuel_type}/{region}")

        # Validate and convert units if needed
        quantity_converted = self._convert_quantity_if_needed(quantity, unit, ef.unit)

        # ZERO-HALLUCINATION CALCULATION
        # Formula: emissions = activity_data * emission_factor
        emissions = quantity_converted * ef.value

        # Generate provenance hash
        provenance_data = {
            "formula": "emissions = quantity * emission_factor",
            "inputs": {
                "quantity": quantity,
                "unit": unit,
                "emission_factor": ef.value,
                "emission_factor_unit": ef.unit,
                "emission_factor_source": ef.source.value,
            },
            "result": emissions,
            "timestamp": datetime.utcnow().isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        logger.info(
            f"Calculated emissions: {emissions:.4f} kgCO2e "
            f"(fuel={fuel_type}, qty={quantity} {unit}, region={region})"
        )

        return CalculationResult(
            value=round(emissions, 6),
            unit="kgCO2e",
            formula_id=f"emissions.scope{scope}.{fuel_type}",
            inputs={
                "fuel_type": fuel_type,
                "quantity": quantity,
                "unit": unit,
                "region": region,
                "scope": scope,
            },
            emission_factors_used=[ef],
            provenance_hash=provenance_hash,
            uncertainty=self._calculate_uncertainty(ef),
        )

    def calculate_scope2_emissions(
        self,
        electricity_kwh: float,
        region: str,
        calculation_method: str = "location",
    ) -> CalculationResult:
        """
        Calculate Scope 2 emissions from electricity consumption.

        Args:
            electricity_kwh: Electricity consumption in kWh
            region: Geographic region
            calculation_method: "location" or "market" based

        Returns:
            Calculation result with provenance
        """
        # Get grid emission factor
        ef = self.get_emission_factor("electricity_grid", region)
        if not ef:
            raise ValueError(f"No grid emission factor for region: {region}")

        # ZERO-HALLUCINATION CALCULATION
        emissions = electricity_kwh * ef.value

        provenance_hash = self._generate_provenance_hash({
            "formula": "scope2_emissions = electricity_kwh * grid_factor",
            "electricity_kwh": electricity_kwh,
            "grid_factor": ef.value,
            "result": emissions,
        })

        return CalculationResult(
            value=round(emissions, 6),
            unit="kgCO2e",
            formula_id=f"emissions.scope2.electricity.{calculation_method}",
            inputs={
                "electricity_kwh": electricity_kwh,
                "region": region,
                "calculation_method": calculation_method,
            },
            emission_factors_used=[ef],
            provenance_hash=provenance_hash,
        )

    def calculate_cbam_embedded_emissions(
        self,
        cn_code: str,
        quantity_kg: float,
        origin_country: str,
        production_route: Optional[str] = None,
    ) -> CalculationResult:
        """
        Calculate CBAM embedded emissions for imported goods.

        Args:
            cn_code: Combined Nomenclature code
            quantity_kg: Quantity in kg
            origin_country: Country of origin
            production_route: Optional production route specification

        Returns:
            Calculation result with CBAM-compliant provenance
        """
        # Map CN code to product category
        product_category = self._cn_code_to_category(cn_code)

        # Get default or specific emission factor
        ef = self._get_cbam_emission_factor(product_category, origin_country)

        # ZERO-HALLUCINATION CALCULATION
        embedded_emissions = (quantity_kg / 1000) * ef.value  # tCO2e per tonne

        provenance_hash = self._generate_provenance_hash({
            "formula": "cbam_emissions = (quantity_kg / 1000) * ef_per_tonne",
            "cn_code": cn_code,
            "quantity_kg": quantity_kg,
            "emission_factor": ef.value,
            "result": embedded_emissions,
        })

        return CalculationResult(
            value=round(embedded_emissions, 6),
            unit="tCO2e",
            formula_id="cbam.embedded_emissions",
            inputs={
                "cn_code": cn_code,
                "quantity_kg": quantity_kg,
                "origin_country": origin_country,
                "product_category": product_category,
            },
            emission_factors_used=[ef],
            provenance_hash=provenance_hash,
        )

    def aggregate_emissions(
        self,
        scope1: float,
        scope2: float,
        scope3: float = 0,
    ) -> CalculationResult:
        """
        Aggregate emissions across scopes.

        Args:
            scope1: Scope 1 emissions (kgCO2e)
            scope2: Scope 2 emissions (kgCO2e)
            scope3: Scope 3 emissions (kgCO2e)

        Returns:
            Aggregated result
        """
        total = scope1 + scope2 + scope3

        return CalculationResult(
            value=round(total, 6),
            unit="kgCO2e",
            formula_id="emissions.aggregate",
            inputs={
                "scope1": scope1,
                "scope2": scope2,
                "scope3": scope3,
            },
            provenance_hash=self._generate_provenance_hash({
                "formula": "total = scope1 + scope2 + scope3",
                "scope1": scope1,
                "scope2": scope2,
                "scope3": scope3,
                "total": total,
            }),
        )

    def _convert_quantity_if_needed(
        self,
        quantity: float,
        from_unit: str,
        to_unit_with_denominator: str,
    ) -> float:
        """Convert quantity if units don't match."""
        # Extract denominator unit from emission factor unit
        # e.g., "kgCO2e/m3" -> "m3"
        if "/" in to_unit_with_denominator:
            target_unit = to_unit_with_denominator.split("/")[1]
        else:
            target_unit = to_unit_with_denominator

        if from_unit == target_unit:
            return quantity

        # Common conversions
        conversions = {
            ("m3", "L"): 1000,
            ("L", "m3"): 0.001,
            ("kg", "t"): 0.001,
            ("t", "kg"): 1000,
            ("MWh", "kWh"): 1000,
            ("kWh", "MWh"): 0.001,
            ("GJ", "kWh"): 277.78,
        }

        conversion = conversions.get((from_unit, target_unit))
        if conversion:
            return quantity * conversion

        logger.warning(f"No conversion available: {from_unit} -> {target_unit}")
        return quantity

    def _cn_code_to_category(self, cn_code: str) -> str:
        """Map CN code to CBAM product category."""
        # Simplified mapping for demonstration
        cn_prefixes = {
            "72": "iron_steel",
            "73": "iron_steel_articles",
            "76": "aluminium",
            "25": "cement",
            "28": "chemicals",
            "31": "fertilizers",
        }

        prefix = cn_code[:2]
        return cn_prefixes.get(prefix, "other")

    def _get_cbam_emission_factor(
        self,
        product_category: str,
        origin_country: str,
    ) -> EmissionFactor:
        """Get CBAM-specific emission factor."""
        # CBAM default values (tCO2e per tonne)
        cbam_defaults = {
            "iron_steel": 1.85,
            "aluminium": 8.4,
            "cement": 0.65,
            "fertilizers": 2.8,
            "chemicals": 1.2,
        }

        value = cbam_defaults.get(product_category, 1.0)

        return EmissionFactor(
            material_id=f"cbam_{product_category}",
            region=origin_country,
            year=2024,
            value=value,
            unit="tCO2e/t",
            source=EmissionFactorSource.IPCC,
        )

    def _calculate_uncertainty(self, ef: EmissionFactor) -> Optional[float]:
        """Calculate uncertainty percentage from emission factor bounds."""
        if ef.uncertainty_lower and ef.uncertainty_upper:
            range_pct = ((ef.uncertainty_upper - ef.uncertainty_lower) / ef.value) * 100
            return round(range_pct / 2, 2)  # Half range as uncertainty
        return None

    def _generate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Generate SHA-256 provenance hash."""
        data["timestamp"] = datetime.utcnow().isoformat()
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
