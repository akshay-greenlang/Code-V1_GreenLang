# -*- coding: utf-8 -*-
"""
SB 253 Scope 1 Stationary Combustion Calculator
===============================================

Calculates direct GHG emissions from stationary combustion sources:
- Boilers
- Furnaces
- Heaters
- Generators
- Other stationary equipment

Emission Factors: EPA GHG Emission Factors Hub 2024
GWP Values: IPCC AR6 (GWP-100)

Formula:
    Emissions (kg CO2e) = Fuel Consumed (units) x Emission Factor (kg CO2e/unit)

Accuracy Target: +/- 1%

Author: GreenLang Framework Team
Version: 1.0.0
Date: 2025-12-04
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..base import BaseCalculator, CalculationResult, AuditRecord, UnitConverter


@dataclass
class StationaryCombustionInput:
    """Input data for stationary combustion calculation."""
    facility_id: str
    fuel_type: str
    quantity: float
    unit: str
    reporting_period_start: str
    reporting_period_end: str
    equipment_type: Optional[str] = None
    source_document_id: Optional[str] = None


@dataclass
class EmissionFactorData:
    """Emission factor with metadata for audit trail."""
    factor_value: float
    factor_unit: str
    source: str
    source_uri: str
    version: str
    gwp_basis: str
    co2_factor: float
    ch4_factor: float
    n2o_factor: float


class StationaryCombustionCalculator(BaseCalculator):
    """
    Calculate Scope 1 emissions from stationary combustion.

    This calculator is DETERMINISTIC - no AI/estimation is used.
    All calculations follow:
        Emissions = Fuel Quantity x Emission Factor

    Supports:
        - Natural gas (therms, kWh, MCF, MMBtu)
        - Diesel (gallons, liters)
        - Propane/LPG (gallons, liters)
        - Fuel oil #2, #4, #6 (gallons, liters)
        - Kerosene (gallons, liters)

    Accuracy Target: +/- 1%
    """

    CALCULATOR_ID = "sb253-scope1-stationary-v1"
    CALCULATOR_VERSION = "1.0.0"

    # EPA GHG Emission Factors Hub 2024 (kg CO2e per standard unit)
    # Source: https://www.epa.gov/climateleadership/ghg-emission-factors-hub
    EMISSION_FACTORS: Dict[str, EmissionFactorData] = {
        "natural_gas": EmissionFactorData(
            factor_value=5.30,
            factor_unit="kg CO2e/therm",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=5.27,
            ch4_factor=0.005,
            n2o_factor=0.0001
        ),
        "diesel": EmissionFactorData(
            factor_value=10.21,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.15,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "propane": EmissionFactorData(
            factor_value=5.72,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=5.68,
            ch4_factor=0.0003,
            n2o_factor=0.0001
        ),
        "lpg": EmissionFactorData(
            factor_value=5.72,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=5.68,
            ch4_factor=0.0003,
            n2o_factor=0.0001
        ),
        "fuel_oil_2": EmissionFactorData(
            factor_value=10.21,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.15,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "fuel_oil_4": EmissionFactorData(
            factor_value=11.27,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=11.21,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "fuel_oil_6": EmissionFactorData(
            factor_value=11.27,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=11.21,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "kerosene": EmissionFactorData(
            factor_value=10.15,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.10,
            ch4_factor=0.0003,
            n2o_factor=0.0001
        ),
    }

    def __init__(self):
        super().__init__(
            calculator_id=self.CALCULATOR_ID,
            version=self.CALCULATOR_VERSION
        )

    def calculate(
        self,
        inputs: List[Dict[str, Any]]
    ) -> CalculationResult:
        """
        Calculate stationary combustion emissions for multiple fuel sources.

        This method is DETERMINISTIC - same inputs always produce same outputs.

        Args:
            inputs: List of fuel consumption records with:
                - facility_id: str
                - fuel_type: str (natural_gas, diesel, propane, etc.)
                - quantity: float (amount consumed)
                - unit: str (therms, gallons, liters, etc.)
                - reporting_period_start: str (ISO date)
                - reporting_period_end: str (ISO date)

        Returns:
            CalculationResult with:
                - total_emissions_kg_co2e
                - total_emissions_mt_co2e
                - emissions_by_source (by fuel type)
                - audit_records (complete provenance)

        Raises:
            ValueError: If fuel type or unit is unsupported
        """
        total_emissions_kg = 0.0
        total_co2_kg = 0.0
        total_ch4_kg_co2e = 0.0
        total_n2o_kg_co2e = 0.0
        emissions_by_fuel: Dict[str, float] = {}
        audit_records: List[AuditRecord] = []

        for input_dict in inputs:
            # Parse input
            input_data = self._parse_input(input_dict)

            # Validate fuel type
            fuel_type = input_data.fuel_type.lower()
            if fuel_type not in self.EMISSION_FACTORS:
                raise ValueError(
                    f"Unsupported fuel type: {input_data.fuel_type}. "
                    f"Supported types: {list(self.EMISSION_FACTORS.keys())}"
                )

            # Get emission factor
            ef_data = self.EMISSION_FACTORS[fuel_type]

            # Convert units to standard (therms for gas, gallons for liquids)
            quantity_standardized = self._convert_to_standard_unit(
                input_data.quantity,
                input_data.unit,
                fuel_type
            )

            # Calculate emissions (DETERMINISTIC)
            emissions_kg = quantity_standardized * ef_data.factor_value

            # Calculate gas breakdown
            co2_kg = quantity_standardized * ef_data.co2_factor
            ch4_kg_co2e = quantity_standardized * ef_data.ch4_factor
            n2o_kg_co2e = quantity_standardized * ef_data.n2o_factor

            # Aggregate totals
            total_emissions_kg += emissions_kg
            total_co2_kg += co2_kg
            total_ch4_kg_co2e += ch4_kg_co2e
            total_n2o_kg_co2e += n2o_kg_co2e

            # Track by fuel type
            if fuel_type not in emissions_by_fuel:
                emissions_by_fuel[fuel_type] = 0.0
            emissions_by_fuel[fuel_type] += emissions_kg

            # Create audit record with SHA-256 provenance
            audit_record = self._create_audit_record(
                input_data=input_data,
                ef_data=ef_data,
                quantity_standardized=quantity_standardized,
                emissions_kg=emissions_kg,
                co2_kg=co2_kg,
                ch4_kg_co2e=ch4_kg_co2e,
                n2o_kg_co2e=n2o_kg_co2e
            )
            audit_records.append(audit_record)

        # Convert to metric tonnes
        total_emissions_mt = total_emissions_kg / 1000.0

        return CalculationResult(
            success=True,
            scope="1",
            category="stationary_combustion",
            total_emissions_kg_co2e=self.round_emissions(total_emissions_kg),
            total_emissions_mt_co2e=self.round_emissions(total_emissions_mt, 3),
            emissions_by_source=emissions_by_fuel,
            co2_kg=self.round_emissions(total_co2_kg),
            ch4_kg_co2e=self.round_emissions(total_ch4_kg_co2e),
            n2o_kg_co2e=self.round_emissions(total_n2o_kg_co2e),
            audit_records=audit_records,
            calculation_timestamp=self.get_timestamp(),
            calculator_id=self.CALCULATOR_ID,
            calculator_version=self.CALCULATOR_VERSION,
            metadata={
                "num_fuel_records": len(inputs),
                "fuel_types": list(emissions_by_fuel.keys()),
                "emission_factor_source": "EPA GHG Emission Factors Hub 2024",
            }
        )

    def _parse_input(self, input_dict: Dict[str, Any]) -> StationaryCombustionInput:
        """Parse dictionary input into typed dataclass."""
        return StationaryCombustionInput(
            facility_id=input_dict["facility_id"],
            fuel_type=input_dict["fuel_type"],
            quantity=float(input_dict["quantity"]),
            unit=input_dict["unit"],
            reporting_period_start=input_dict["reporting_period_start"],
            reporting_period_end=input_dict["reporting_period_end"],
            equipment_type=input_dict.get("equipment_type"),
            source_document_id=input_dict.get("source_document_id"),
        )

    def _convert_to_standard_unit(
        self,
        quantity: float,
        unit: str,
        fuel_type: str
    ) -> float:
        """
        Convert input units to standard units for emission factor application.

        Natural gas: Standard unit is therms
        Liquid fuels: Standard unit is gallons
        """
        unit_lower = unit.lower()

        # Natural gas conversions (to therms)
        if fuel_type == "natural_gas":
            conversions = {
                "therms": 1.0,
                "therm": 1.0,
                "kwh": 0.0341296,
                "mcf": 10.0,
                "mmbtu": 10.0,
                "m3": 0.3531,
                "ccf": 1.0,
            }
            if unit_lower not in conversions:
                raise ValueError(
                    f"Unsupported unit '{unit}' for natural gas. "
                    f"Supported: {list(conversions.keys())}"
                )
            return quantity * conversions[unit_lower]

        # Liquid fuel conversions (to gallons)
        liquid_conversions = {
            "gallons": 1.0,
            "gallon": 1.0,
            "gal": 1.0,
            "liters": 0.2642,
            "liter": 0.2642,
            "l": 0.2642,
            "barrels": 42.0,
            "barrel": 42.0,
            "bbl": 42.0,
        }

        if unit_lower not in liquid_conversions:
            raise ValueError(
                f"Unsupported unit '{unit}' for {fuel_type}. "
                f"Supported: {list(liquid_conversions.keys())}"
            )

        return quantity * liquid_conversions[unit_lower]

    def _create_audit_record(
        self,
        input_data: StationaryCombustionInput,
        ef_data: EmissionFactorData,
        quantity_standardized: float,
        emissions_kg: float,
        co2_kg: float,
        ch4_kg_co2e: float,
        n2o_kg_co2e: float
    ) -> AuditRecord:
        """
        Create SHA-256 verified audit record for assurance.

        Each calculation gets a unique audit record with:
        - Input hash (SHA-256)
        - Output hash (SHA-256)
        - Complete emission factor provenance
        - Calculation formula documentation
        """
        # Create deterministic input dictionary for hashing
        input_dict = {
            "facility_id": input_data.facility_id,
            "fuel_type": input_data.fuel_type,
            "quantity": input_data.quantity,
            "unit": input_data.unit,
            "reporting_period_start": input_data.reporting_period_start,
            "reporting_period_end": input_data.reporting_period_end,
        }

        # SHA-256 hash of inputs
        input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode()
        ).hexdigest()

        # Create deterministic output dictionary for hashing
        output_dict = {
            "emissions_kg_co2e": self.round_emissions(emissions_kg),
            "co2_kg": self.round_emissions(co2_kg),
            "ch4_kg_co2e": self.round_emissions(ch4_kg_co2e),
            "n2o_kg_co2e": self.round_emissions(n2o_kg_co2e),
        }

        # SHA-256 hash of outputs
        output_hash = hashlib.sha256(
            json.dumps(output_dict, sort_keys=True).encode()
        ).hexdigest()

        return AuditRecord(
            calculation_id=f"{self.CALCULATOR_ID}-{input_hash[:12]}",
            timestamp=self.get_timestamp(),
            scope="1",
            category="stationary_combustion",
            input_hash=input_hash,
            output_hash=output_hash,
            emission_factor_source=ef_data.source,
            emission_factor_version=ef_data.version,
            emission_factor_value=ef_data.factor_value,
            emission_factor_unit=ef_data.factor_unit,
            gwp_basis=ef_data.gwp_basis,
            calculation_formula=(
                f"emissions = {quantity_standardized:.4f} {ef_data.factor_unit.split('/')[1]} "
                f"x {ef_data.factor_value} {ef_data.factor_unit} "
                f"= {emissions_kg:.4f} kg CO2e"
            ),
            inputs=input_dict,
            outputs=output_dict
        )

    def get_supported_fuels(self) -> List[str]:
        """Get list of supported fuel types."""
        return list(self.EMISSION_FACTORS.keys())

    def get_emission_factor(self, fuel_type: str) -> Optional[EmissionFactorData]:
        """Get emission factor data for a fuel type."""
        return self.EMISSION_FACTORS.get(fuel_type.lower())
