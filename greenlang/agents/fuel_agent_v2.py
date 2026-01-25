# -*- coding: utf-8 -*-
"""
FuelAgent V2 - Migrated to AgentSpecV2Base Pattern
===================================================

This is the migrated version of FuelAgent demonstrating the standardized
AgentSpecV2Base + DeterministicMixin inheritance pattern.

This agent calculates emissions from fuel consumption with:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail for regulatory compliance
- Provenance tracking with SHA-256 hashes
- Type-safe Pydantic models
- Standard lifecycle management

Category: DETERMINISTIC
Use for: Scope 1 fuel emission calculations

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready (Migration Example)
"""

from typing import Optional, Dict, Any
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import json

from pydantic import BaseModel, Field, validator

from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import DeterministicMixin
from greenlang.data.emission_factors import EmissionFactors
from greenlang.utils.unit_converter import UnitConverter
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Input/Output Models (Pydantic)
# ==============================================================================

class FuelInputV2(BaseModel):
    """
    Input data model for FuelAgent V2.

    All inputs are validated using Pydantic before reaching execute_impl().
    """

    fuel_type: str = Field(
        ...,
        description="Type of fuel (natural_gas, diesel, gasoline, electricity, etc.)"
    )
    amount: float = Field(
        ...,
        ge=0,
        description="Fuel consumption amount (must be non-negative)"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (kWh, therms, gallons, liters, etc.)"
    )
    country: str = Field(
        default="US",
        description="Country code for emission factor selection (ISO 3166-1 alpha-2)"
    )
    year: int = Field(
        default=2025,
        ge=2000,
        le=2050,
        description="Year for emission factor vintage"
    )
    efficiency: Optional[float] = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Equipment efficiency factor (0.0-1.0)"
    )
    renewable_percentage: Optional[float] = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Renewable energy percentage (0-100)"
    )

    @validator('fuel_type')
    def validate_fuel_type(cls, v):
        """Validate fuel type is known."""
        valid_fuels = {
            "natural_gas", "diesel", "gasoline", "propane", "fuel_oil",
            "coal", "biomass", "electricity", "lpg", "heating_oil"
        }
        if v.lower() not in valid_fuels:
            logger.warning(f"Fuel type '{v}' not in standard list. Will attempt lookup anyway.")
        return v.lower()

    @validator('unit')
    def validate_unit(cls, v):
        """Validate unit format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Unit cannot be empty")
        return v.strip()


class FuelOutputV2(BaseModel):
    """
    Output data model for FuelAgent V2.

    All outputs conform to this schema and are validated before returning.
    """

    co2e_emissions_kg: float = Field(
        ...,
        description="Total CO2 equivalent emissions in kilograms"
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type processed (normalized)"
    )
    consumption_amount: float = Field(
        ...,
        description="Fuel consumption amount"
    )
    consumption_unit: str = Field(
        ...,
        description="Unit of fuel consumption"
    )
    emission_factor: float = Field(
        ...,
        description="Emission factor used in calculation"
    )
    emission_factor_unit: str = Field(
        ...,
        description="Unit of emission factor"
    )
    country: str = Field(
        ...,
        description="Country for emission factor"
    )
    scope: str = Field(
        ...,
        description="GHG Protocol scope (1, 2, or 3)"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )
    calculation_time_ms: float = Field(
        ...,
        description="Calculation execution time in milliseconds"
    )
    renewable_offset_applied: bool = Field(
        default=False,
        description="Whether renewable energy offset was applied"
    )
    efficiency_adjusted: bool = Field(
        default=False,
        description="Whether efficiency adjustment was applied"
    )


# ==============================================================================
# FuelAgent V2 Implementation
# ==============================================================================

class FuelAgentV2(AgentSpecV2Base[FuelInputV2, FuelOutputV2], DeterministicMixin):
    """
    FuelAgent V2 - Standardized deterministic fuel emissions calculator.

    This agent demonstrates the AgentSpecV2Base + DeterministicMixin pattern:
    - Inherits from AgentSpecV2Base[InT, OutT] for lifecycle management
    - Uses DeterministicMixin for zero-hallucination guarantee
    - Implements execute_impl() for core calculation logic
    - Provides full audit trail for regulatory compliance

    Category: DETERMINISTIC
    - NO LLM calls in calculation path
    - Fully reproducible results
    - Complete audit trail
    - Provenance tracking

    Example:
        >>> agent = FuelAgentV2()
        >>> input_data = FuelInputV2(
        ...     fuel_type="natural_gas",
        ...     amount=100,
        ...     unit="therms",
        ...     country="US",
        ...     year=2025
        ... )
        >>> result = agent.run(input_data)
        >>> print(result.data["co2e_emissions_kg"])
        530.0
        >>> audit_trail = agent.get_audit_trail()
        >>> print(audit_trail[0]["calculation_trace"])
    """

    def __init__(self, **kwargs):
        """
        Initialize FuelAgent V2.

        Args:
            **kwargs: Additional keyword arguments passed to AgentSpecV2Base
        """
        super().__init__(
            agent_id="fuel_agent_v2",
            enable_metrics=True,
            enable_citations=False,  # Not needed for deterministic calculations
            enable_validation=True,
            **kwargs
        )

        # Agent-specific resources (loaded in initialize_impl)
        self.emission_factors: Optional[EmissionFactors] = None
        self.unit_converter: Optional[UnitConverter] = None
        self.fuel_config: Optional[Dict[str, Any]] = None

        # Fuel type mapping for aliases
        self.fuel_type_mapping = {
            "lpg": "propane",
            "heating_oil": "fuel_oil",
            "wood": "biomass",
            "electric": "electricity",
        }

    def initialize_impl(self) -> None:
        """
        Custom initialization logic for FuelAgent V2.

        This method is called during agent initialization to:
        - Load emission factors database
        - Initialize unit converter
        - Load fuel properties configuration
        """
        try:
            # Load emission factors
            self.emission_factors = EmissionFactors()

            # Load unit converter
            self.unit_converter = UnitConverter()

            # Load fuel properties config
            self.fuel_config = self._load_fuel_config()

            self.logger.info("FuelAgent V2 initialized successfully")

        except Exception as e:
            self.logger.error(f"FuelAgent V2 initialization failed: {e}", exc_info=True)
            raise

    @lru_cache(maxsize=1)
    def _load_fuel_config(self) -> Dict[str, Any]:
        """
        Load fuel properties configuration from external file.

        Returns:
            Fuel configuration dictionary
        """
        config_path = Path(__file__).parent.parent / "configs" / "fuel_properties.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # Fallback to basic properties
            return {
                "fuel_properties": {
                    "electricity": {"energy_content": {"value": 3412, "unit": "Btu/kWh"}},
                    "natural_gas": {"energy_content": {"value": 100000, "unit": "Btu/therm"}},
                    "diesel": {"energy_content": {"value": 138690, "unit": "Btu/gallon"}},
                }
            }

    def validate_input_impl(
        self,
        input_data: FuelInputV2,
        context: AgentExecutionContext
    ) -> FuelInputV2:
        """
        Custom input validation logic beyond Pydantic schema validation.

        Args:
            input_data: Validated input data from Pydantic
            context: Execution context

        Returns:
            Validated input data (possibly transformed)

        Raises:
            ValueError: If validation fails
        """
        # Normalize fuel type using mapping
        fuel_type = input_data.fuel_type.lower()
        if fuel_type in self.fuel_type_mapping:
            normalized_fuel = self.fuel_type_mapping[fuel_type]
            self.logger.info(f"Normalized fuel type: {fuel_type} → {normalized_fuel}")

            # Create new instance with normalized fuel type
            input_data = input_data.copy(update={"fuel_type": normalized_fuel})

        return input_data

    def execute_impl(
        self,
        validated_input: FuelInputV2,
        context: AgentExecutionContext
    ) -> FuelOutputV2:
        """
        Core execution logic - ZERO HALLUCINATION.

        This method implements deterministic fuel emissions calculation:
        1. Lookup emission factor from database (deterministic)
        2. Calculate emissions using pure arithmetic (deterministic)
        3. Apply efficiency/renewable adjustments (deterministic)
        4. Capture complete audit trail
        5. Calculate provenance hash

        NO LLM CALLS ALLOWED IN THIS METHOD.

        Args:
            validated_input: Validated and normalized input data
            context: Execution context with metadata

        Returns:
            Calculation results with provenance hash

        Raises:
            ValueError: If emission factor not found
        """
        start_time = DeterministicClock.now()

        try:
            # Step 1: Get emission factor (DETERMINISTIC DATABASE LOOKUP)
            emission_factor = self._get_emission_factor(
                fuel_type=validated_input.fuel_type,
                unit=validated_input.unit,
                region=validated_input.country
            )

            if emission_factor is None:
                raise ValueError(
                    f"No emission factor found for {validated_input.fuel_type} "
                    f"in {validated_input.country} ({validated_input.unit})"
                )

            # Step 2: Calculate base emissions (DETERMINISTIC ARITHMETIC)
            co2e_emissions_kg = validated_input.amount * emission_factor

            # Step 3: Apply renewable offset if specified (DETERMINISTIC)
            renewable_offset_applied = False
            if validated_input.renewable_percentage > 0 and validated_input.fuel_type == "electricity":
                offset = co2e_emissions_kg * (validated_input.renewable_percentage / 100)
                co2e_emissions_kg -= offset
                renewable_offset_applied = True

                self.logger.info(
                    f"Applied {validated_input.renewable_percentage}% renewable offset: -{offset:.2f} kg CO2e"
                )

            # Step 4: Apply efficiency adjustment if specified (DETERMINISTIC)
            efficiency_adjusted = False
            if validated_input.efficiency and validated_input.efficiency < 1.0:
                co2e_emissions_kg = co2e_emissions_kg / validated_input.efficiency
                efficiency_adjusted = True

                self.logger.info(
                    f"Adjusted for {validated_input.efficiency*100}% efficiency"
                )

            # Step 5: Determine GHG Protocol scope (DETERMINISTIC MAPPING)
            scope = self._determine_scope(validated_input.fuel_type)

            # Step 6: Capture audit trail (REQUIRED FOR DETERMINISTIC MIXIN)
            calculation_trace = [
                f"fuel_type = {validated_input.fuel_type}",
                f"amount = {validated_input.amount} {validated_input.unit}",
                f"emission_factor = {emission_factor} kgCO2e/{validated_input.unit}",
                f"base_emissions = {validated_input.amount} * {emission_factor} = {validated_input.amount * emission_factor} kg CO2e",
            ]

            if renewable_offset_applied:
                calculation_trace.append(
                    f"renewable_offset = {validated_input.renewable_percentage}% = -{offset:.2f} kg CO2e"
                )

            if efficiency_adjusted:
                calculation_trace.append(
                    f"efficiency_adjustment = {validated_input.efficiency} → {co2e_emissions_kg:.2f} kg CO2e"
                )

            calculation_trace.append(f"final_emissions = {co2e_emissions_kg} kg CO2e")

            self.capture_audit_entry(
                operation="fuel_emissions_calculation",
                inputs=validated_input.dict(),
                outputs={
                    "co2e_emissions_kg": co2e_emissions_kg,
                    "scope": scope,
                    "emission_factor": emission_factor
                },
                calculation_trace=calculation_trace,
                metadata={
                    "country": validated_input.country,
                    "year": validated_input.year,
                    "renewable_offset_applied": renewable_offset_applied,
                    "efficiency_adjusted": efficiency_adjusted
                }
            )

            # Step 7: Calculate provenance hash (REQUIRED FOR DETERMINISTIC MIXIN)
            provenance_hash = self.calculate_provenance_hash(
                inputs=validated_input.dict(),
                outputs={
                    "co2e_emissions_kg": co2e_emissions_kg,
                    "emission_factor": emission_factor,
                    "scope": scope
                }
            )

            # Step 8: Calculate execution time
            execution_time_ms = (DeterministicClock.now() - start_time).total_seconds() * 1000

            # Step 9: Create output
            return FuelOutputV2(
                co2e_emissions_kg=co2e_emissions_kg,
                fuel_type=validated_input.fuel_type,
                consumption_amount=validated_input.amount,
                consumption_unit=validated_input.unit,
                emission_factor=emission_factor,
                emission_factor_unit=f"kgCO2e/{validated_input.unit}",
                country=validated_input.country,
                scope=scope,
                provenance_hash=provenance_hash,
                calculation_time_ms=execution_time_ms,
                renewable_offset_applied=renewable_offset_applied,
                efficiency_adjusted=efficiency_adjusted
            )

        except Exception as e:
            self.logger.error(f"Fuel calculation failed: {e}", exc_info=True)
            raise

    @lru_cache(maxsize=256)
    def _get_emission_factor(self, fuel_type: str, unit: str, region: str) -> Optional[float]:
        """
        Get emission factor with caching for performance.

        This method uses LRU cache to avoid repeated database lookups for the
        same fuel type/unit/region combination.

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            region: Country or region code

        Returns:
            Emission factor (kgCO2e/unit) or None if not found
        """
        return self.emission_factors.get_factor(
            fuel_type=fuel_type,
            unit=unit,
            region=region
        )

    def _determine_scope(self, fuel_type: str) -> str:
        """
        Determine GHG Protocol scope for fuel type.

        Scope 1: Direct emissions from owned/controlled sources
        Scope 2: Indirect emissions from purchased electricity/heat/cooling
        Scope 3: All other indirect emissions

        Args:
            fuel_type: Type of fuel

        Returns:
            Scope number as string ("1", "2", or "3")
        """
        scope_mapping = {
            "natural_gas": "1",
            "diesel": "1",
            "gasoline": "1",
            "propane": "1",
            "fuel_oil": "1",
            "coal": "1",
            "biomass": "1",
            "electricity": "2",
            "district_heating": "2",
            "district_cooling": "2",
        }

        return scope_mapping.get(fuel_type, "1")

    def validate_output_impl(
        self,
        output: FuelOutputV2,
        context: AgentExecutionContext
    ) -> FuelOutputV2:
        """
        Custom output validation logic.

        Args:
            output: Output data to validate
            context: Execution context

        Returns:
            Validated output

        Raises:
            ValueError: If output validation fails
        """
        # Sanity checks
        if output.co2e_emissions_kg < 0:
            raise ValueError(f"Emissions cannot be negative: {output.co2e_emissions_kg}")

        if output.emission_factor <= 0:
            raise ValueError(f"Emission factor must be positive: {output.emission_factor}")

        return output

    def finalize_impl(
        self,
        result: "AgentResult[FuelOutputV2]",
        context: AgentExecutionContext
    ) -> "AgentResult[FuelOutputV2]":
        """
        Custom finalization logic.

        Args:
            result: Agent result to finalize
            context: Execution context

        Returns:
            Finalized result
        """
        # Add audit trail metadata
        audit_trail = self.get_audit_trail()

        result.metadata.update({
            "audit_trail_count": len(audit_trail),
            "category": self.category,
            "deterministic": True,
            "version": "2.0.0"
        })

        return result


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    "FuelAgentV2",
    "FuelInputV2",
    "FuelOutputV2",
]
