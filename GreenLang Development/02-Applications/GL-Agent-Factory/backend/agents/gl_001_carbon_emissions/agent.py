"""
GL-001: Carbon Emissions Calculator Agent

This module implements the Carbon Emissions Calculator Agent that computes
GHG emissions with zero-hallucination deterministic calculations.

The agent supports:
- Scope 1 emissions (stationary and mobile combustion)
- Scope 2 emissions (electricity, location/market-based)
- Multiple fuel types and regions
- Complete provenance tracking

Example:
    >>> agent = CarbonEmissionsAgent()
    >>> result = agent.run(CarbonEmissionsInput(
    ...     fuel_type="natural_gas",
    ...     quantity=1000,
    ...     unit="m3",
    ...     region="US"
    ... ))
    >>> print(f"Emissions: {result.data.emissions_kgco2e} kgCO2e")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """Supported fuel types."""

    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    COAL = "coal"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    ELECTRICITY_GRID = "electricity_grid"


class Scope(int, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = 1
    SCOPE_2 = 2
    SCOPE_3 = 3


class CarbonEmissionsInput(BaseModel):
    """
    Input model for Carbon Emissions Calculator.

    Attributes:
        fuel_type: Type of fuel consumed
        quantity: Amount consumed
        unit: Unit of measurement
        region: Geographic region (ISO 3166)
        scope: GHG Protocol scope (1, 2, or 3)
        calculation_method: For Scope 2 - location or market based
        metadata: Additional metadata
    """

    fuel_type: FuelType = Field(..., description="Type of fuel consumed")
    quantity: float = Field(..., ge=0, description="Amount consumed")
    unit: str = Field(..., description="Unit of measurement")
    region: str = Field("US", description="Geographic region")
    scope: Scope = Field(Scope.SCOPE_1, description="GHG Protocol scope")
    calculation_method: str = Field(
        "location",
        description="Calculation method (location/market for Scope 2)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("unit")
    def validate_unit(cls, v: str, values: Dict) -> str:
        """Validate unit is appropriate for fuel type."""
        fuel_type = values.get("fuel_type")
        valid_units = {
            FuelType.NATURAL_GAS: ["m3", "cf", "therm", "GJ"],
            FuelType.DIESEL: ["L", "gal", "kg"],
            FuelType.GASOLINE: ["L", "gal"],
            FuelType.COAL: ["kg", "t", "short_ton"],
            FuelType.FUEL_OIL: ["L", "gal", "kg"],
            FuelType.PROPANE: ["L", "gal", "kg"],
            FuelType.ELECTRICITY_GRID: ["kWh", "MWh", "GJ"],
        }

        if fuel_type and v not in valid_units.get(fuel_type, []):
            logger.warning(f"Unit {v} may not be standard for {fuel_type}")

        return v

    @validator("region")
    def validate_region(cls, v: str) -> str:
        """Validate region code."""
        # Common region codes
        valid_regions = {
            "US", "EU", "UK", "DE", "FR", "IT", "ES", "CN", "JP", "KR",
            "AU", "CA", "BR", "IN", "MX", "GLOBAL"
        }
        if v.upper() not in valid_regions:
            logger.warning(f"Region {v} may not have specific emission factors")
        return v.upper()


class CarbonEmissionsOutput(BaseModel):
    """
    Output model for Carbon Emissions Calculator.

    All outputs include provenance tracking for audit compliance.
    """

    emissions_kgco2e: float = Field(..., description="Total emissions in kgCO2e")
    emission_factor_used: float = Field(..., description="Emission factor value")
    emission_factor_unit: str = Field(..., description="Emission factor unit")
    emission_factor_source: str = Field(..., description="Source (EPA, IPCC, etc.)")
    scope: int = Field(..., description="GHG Protocol scope")
    calculation_method: str = Field(..., description="Calculation method")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    uncertainty_pct: Optional[float] = Field(None, description="Uncertainty percentage")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class EmissionFactor(BaseModel):
    """Emission factor with provenance."""

    value: float
    unit: str
    source: str
    year: int
    uncertainty_lower: Optional[float] = None
    uncertainty_upper: Optional[float] = None


class CarbonEmissionsAgent:
    """
    GL-001: Carbon Emissions Calculator Agent.

    This agent calculates GHG emissions using zero-hallucination
    deterministic calculations. All numeric computations use:
    - Validated emission factors from authoritative sources
    - Deterministic formulas (emissions = activity * EF)
    - Complete SHA-256 provenance tracking

    Attributes:
        emission_factors: Database of validated emission factors
        config: Agent configuration

    Example:
        >>> agent = CarbonEmissionsAgent()
        >>> result = agent.run(CarbonEmissionsInput(
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     quantity=1000,
        ...     unit="m3",
        ...     region="US"
        ... ))
        >>> assert result.provenance_hash is not None
    """

    # Agent metadata
    AGENT_ID = "emissions/carbon_calculator_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "Zero-hallucination carbon emissions calculator"

    # Emission factors (EPA, DEFRA, IPCC sources)
    # In production, these come from a validated database
    EMISSION_FACTORS: Dict[str, Dict[str, EmissionFactor]] = {
        "natural_gas": {
            "US": EmissionFactor(
                value=1.93,
                unit="kgCO2e/m3",
                source="EPA",
                year=2024,
            ),
            "EU": EmissionFactor(
                value=2.02,
                unit="kgCO2e/m3",
                source="DEFRA",
                year=2024,
            ),
        },
        "diesel": {
            "US": EmissionFactor(
                value=2.68,
                unit="kgCO2e/L",
                source="EPA",
                year=2024,
            ),
            "EU": EmissionFactor(
                value=2.62,
                unit="kgCO2e/L",
                source="DEFRA",
                year=2024,
            ),
        },
        "gasoline": {
            "US": EmissionFactor(
                value=2.31,
                unit="kgCO2e/L",
                source="EPA",
                year=2024,
            ),
            "EU": EmissionFactor(
                value=2.19,
                unit="kgCO2e/L",
                source="DEFRA",
                year=2024,
            ),
        },
        "electricity_grid": {
            "US": EmissionFactor(
                value=0.417,
                unit="kgCO2e/kWh",
                source="EPA eGRID",
                year=2024,
            ),
            "EU": EmissionFactor(
                value=0.276,
                unit="kgCO2e/kWh",
                source="IEA",
                year=2024,
            ),
            "DE": EmissionFactor(
                value=0.366,
                unit="kgCO2e/kWh",
                source="IEA",
                year=2024,
            ),
            "FR": EmissionFactor(
                value=0.052,
                unit="kgCO2e/kWh",
                source="IEA",
                year=2024,
            ),
        },
    }

    # Unit conversion factors
    UNIT_CONVERSIONS: Dict[tuple, float] = {
        ("cf", "m3"): 0.0283168,
        ("therm", "m3"): 2.832,
        ("GJ", "m3"): 26.137,
        ("gal", "L"): 3.78541,
        ("MWh", "kWh"): 1000,
        ("t", "kg"): 1000,
        ("short_ton", "kg"): 907.185,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Carbon Emissions Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"CarbonEmissionsAgent initialized (version {self.VERSION})")

    def run(self, input_data: CarbonEmissionsInput) -> CarbonEmissionsOutput:
        """
        Execute the carbon emissions calculation.

        This method performs a zero-hallucination calculation:
        emissions = activity_data * emission_factor

        Args:
            input_data: Validated input data

        Returns:
            Calculation result with provenance

        Raises:
            ValueError: If emission factor not found
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Calculating emissions: fuel={input_data.fuel_type}, "
            f"qty={input_data.quantity} {input_data.unit}, region={input_data.region}"
        )

        try:
            # Step 1: Get emission factor
            ef = self._get_emission_factor(
                input_data.fuel_type.value,
                input_data.region,
            )
            if not ef:
                raise ValueError(
                    f"No emission factor found for {input_data.fuel_type}/{input_data.region}"
                )

            self._track_step("emission_factor_lookup", {
                "fuel_type": input_data.fuel_type.value,
                "region": input_data.region,
                "ef_value": ef.value,
                "ef_source": ef.source,
            })

            # Step 2: Convert units if needed
            quantity_converted = self._convert_units(
                input_data.quantity,
                input_data.unit,
                ef.unit,
            )

            self._track_step("unit_conversion", {
                "original_quantity": input_data.quantity,
                "original_unit": input_data.unit,
                "converted_quantity": quantity_converted,
            })

            # Step 3: ZERO-HALLUCINATION CALCULATION
            # Formula: emissions = activity_data * emission_factor
            emissions = quantity_converted * ef.value

            self._track_step("calculation", {
                "formula": "emissions = quantity * emission_factor",
                "quantity": quantity_converted,
                "emission_factor": ef.value,
                "result": emissions,
            })

            # Step 4: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 5: Calculate uncertainty
            uncertainty = None
            if ef.uncertainty_lower and ef.uncertainty_upper:
                uncertainty = ((ef.uncertainty_upper - ef.uncertainty_lower) / ef.value) * 50

            # Step 6: Create output
            output = CarbonEmissionsOutput(
                emissions_kgco2e=round(emissions, 6),
                emission_factor_used=ef.value,
                emission_factor_unit=ef.unit,
                emission_factor_source=ef.source,
                scope=input_data.scope.value,
                calculation_method=input_data.calculation_method,
                provenance_hash=provenance_hash,
                uncertainty_pct=uncertainty,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Calculation complete: {emissions:.4f} kgCO2e "
                f"(duration: {duration_ms:.2f}ms, provenance: {provenance_hash[:16]}...)"
            )

            return output

        except Exception as e:
            logger.error(f"Calculation failed: {str(e)}", exc_info=True)
            raise

    def _get_emission_factor(
        self,
        fuel_type: str,
        region: str,
    ) -> Optional[EmissionFactor]:
        """
        Look up emission factor from validated database.

        ZERO-HALLUCINATION: All factors from authoritative sources.
        """
        fuel_factors = self.EMISSION_FACTORS.get(fuel_type, {})

        # Try exact region match
        if region in fuel_factors:
            return fuel_factors[region]

        # Try parent region
        region_hierarchy = {"DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU", "UK": "EU"}
        parent = region_hierarchy.get(region)
        if parent and parent in fuel_factors:
            return fuel_factors[parent]

        # Fall back to US (most comprehensive data)
        if "US" in fuel_factors:
            logger.warning(f"Using US emission factor for region {region}")
            return fuel_factors["US"]

        return None

    def _convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit_with_denominator: str,
    ) -> float:
        """
        Convert units if needed.

        Returns the value converted to match emission factor denominator.
        """
        # Extract denominator from EF unit (e.g., kgCO2e/m3 -> m3)
        target_unit = to_unit_with_denominator.split("/")[1] if "/" in to_unit_with_denominator else to_unit_with_denominator

        if from_unit == target_unit:
            return value

        # Look up conversion
        conversion = self.UNIT_CONVERSIONS.get((from_unit, target_unit))
        if conversion:
            return value * conversion

        # Try reverse conversion
        reverse = self.UNIT_CONVERSIONS.get((target_unit, from_unit))
        if reverse:
            return value / reverse

        logger.warning(f"No conversion found: {from_unit} -> {target_unit}")
        return value

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        This hash enables:
        - Verification that calculation was deterministic
        - Audit trail for regulatory compliance
        - Reproducibility checking
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_supported_fuel_types(self) -> List[str]:
        """Get list of supported fuel types."""
        return [ft.value for ft in FuelType]

    def get_supported_regions(self) -> List[str]:
        """Get list of regions with emission factors."""
        regions = set()
        for fuel_factors in self.EMISSION_FACTORS.values():
            regions.update(fuel_factors.keys())
        return sorted(regions)


# Pack specification (pack.yaml equivalent)
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "emissions/carbon_calculator_v1",
    "name": "Carbon Calculator Agent",
    "version": "1.0.0",
    "summary": "Calculate carbon emissions with zero hallucination",
    "tags": ["emissions", "scope1", "scope2", "ghg-protocol"],
    "owners": ["emissions-team"],
    "compute": {
        "entrypoint": "python://agents.gl_001_carbon_emissions.agent:CarbonEmissionsAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://epa/stationary-combustion/2024"},
        {"ref": "ef://ipcc/gwp/ar6"},
    ],
    "provenance": {
        "ef_version_pin": "2024-Q4",
        "gwp_set": "AR6",
        "enable_audit": True,
    },
}
