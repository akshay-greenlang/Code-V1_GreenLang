# -*- coding: utf-8 -*-
"""
SB 253 Calculator Base Classes
==============================

Base classes and data structures for all SB 253 emission calculators.
Provides audit trail generation and SHA-256 provenance tracking.

Author: GreenLang Framework Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import json


@dataclass
class AuditRecord:
    """
    Audit trail record for assurance-ready calculations.

    Provides complete provenance tracking with SHA-256 hashing
    for third-party verification (Big 4 audit support).
    """
    calculation_id: str
    timestamp: str
    scope: str  # "1", "2", "3"
    category: str

    # SHA-256 provenance hashes
    input_hash: str
    output_hash: str

    # Emission factor metadata
    emission_factor_source: str
    emission_factor_version: str
    emission_factor_value: float
    emission_factor_unit: str
    gwp_basis: str

    # Calculation details
    calculation_formula: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp,
            "scope": self.scope,
            "category": self.category,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "emission_factor": {
                "source": self.emission_factor_source,
                "version": self.emission_factor_version,
                "value": self.emission_factor_value,
                "unit": self.emission_factor_unit,
                "gwp_basis": self.gwp_basis,
            },
            "calculation_formula": self.calculation_formula,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


@dataclass
class CalculationResult:
    """
    Result of an emission calculation.

    Includes emissions values, audit records, and metadata
    for assurance package generation.
    """
    success: bool
    scope: str
    category: str

    # Emission values
    total_emissions_kg_co2e: float = 0.0
    total_emissions_mt_co2e: float = 0.0

    # Breakdown by source
    emissions_by_source: Dict[str, Any] = field(default_factory=dict)

    # Gas breakdown (for detailed reporting)
    co2_kg: float = 0.0
    ch4_kg_co2e: float = 0.0
    n2o_kg_co2e: float = 0.0

    # Audit trail
    audit_records: List[AuditRecord] = field(default_factory=list)

    # Metadata
    calculation_timestamp: str = ""
    calculator_id: str = ""
    calculator_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "scope": self.scope,
            "category": self.category,
            "emissions": {
                "total_kg_co2e": self.total_emissions_kg_co2e,
                "total_mt_co2e": self.total_emissions_mt_co2e,
                "by_source": self.emissions_by_source,
                "gas_breakdown": {
                    "co2_kg": self.co2_kg,
                    "ch4_kg_co2e": self.ch4_kg_co2e,
                    "n2o_kg_co2e": self.n2o_kg_co2e,
                }
            },
            "audit_trail": [r.to_dict() for r in self.audit_records],
            "calculation_timestamp": self.calculation_timestamp,
            "calculator_id": self.calculator_id,
            "calculator_version": self.calculator_version,
            "metadata": self.metadata,
            "error": self.error_message,
        }


class BaseCalculator(ABC):
    """
    Abstract base class for SB 253 emission calculators.

    All calculators must:
    1. Be deterministic (no AI/estimation in core calculations)
    2. Generate SHA-256 audit trails
    3. Track emission factor provenance
    4. Support assurance verification
    """

    def __init__(self, calculator_id: str, version: str):
        """
        Initialize base calculator.

        Args:
            calculator_id: Unique calculator identifier
            version: Calculator version string
        """
        self.calculator_id = calculator_id
        self.version = version

    @abstractmethod
    def calculate(self, inputs: List[Dict[str, Any]]) -> CalculationResult:
        """
        Execute the emission calculation.

        Args:
            inputs: List of input records

        Returns:
            CalculationResult with emissions and audit trail
        """
        pass

    def create_hash(self, data: Dict[str, Any]) -> str:
        """
        Create SHA-256 hash of data for provenance tracking.

        Args:
            data: Dictionary to hash

        Returns:
            SHA-256 hex digest
        """
        # Sort keys for deterministic hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    def validate_positive(self, value: float, field_name: str) -> None:
        """Validate that a value is non-negative."""
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative, got {value}")

    def round_emissions(self, value: float, decimals: int = 6) -> float:
        """Round emission values consistently."""
        return round(value, decimals)


class UnitConverter:
    """
    Unit conversion utilities for emission calculations.

    Provides standardized conversions for:
    - Fuel quantities (therms, gallons, kWh, etc.)
    - Energy units (MMBtu, MJ, kWh)
    - Mass units (kg, tonnes, tons)
    """

    # Fuel unit conversions to standard units
    FUEL_CONVERSIONS = {
        # Natural gas conversions (to therms)
        "natural_gas": {
            "therms": 1.0,
            "kWh": 0.0341296,  # 1 kWh = 0.0341296 therms
            "MCF": 10.0,       # 1 MCF = 10 therms
            "MMBtu": 10.0,     # 1 MMBtu = 10 therms
            "m3": 0.3531,      # 1 m3 = 0.3531 therms
            "GJ": 9.4782,      # 1 GJ = 9.4782 therms
        },
        # Liquid fuel conversions (to gallons)
        "diesel": {
            "gallons": 1.0,
            "liters": 0.2642,
            "barrels": 42.0,
        },
        "gasoline": {
            "gallons": 1.0,
            "liters": 0.2642,
            "barrels": 42.0,
        },
        "propane": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "fuel_oil_2": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "fuel_oil_6": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "kerosene": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
    }

    # Energy conversions
    ENERGY_CONVERSIONS = {
        "kWh_to_MJ": 3.6,
        "MJ_to_kWh": 0.2778,
        "MMBtu_to_MJ": 1055.06,
        "MMBtu_to_kWh": 293.07,
        "therm_to_kWh": 29.3,
        "therm_to_MJ": 105.5,
    }

    # Mass conversions
    MASS_CONVERSIONS = {
        "kg_to_tonnes": 0.001,
        "tonnes_to_kg": 1000,
        "kg_to_lbs": 2.2046,
        "lbs_to_kg": 0.4536,
        "tons_to_kg": 907.185,  # US short tons
        "tonnes_to_tons": 1.1023,  # metric to US short
    }

    @classmethod
    def convert_fuel(
        cls,
        value: float,
        from_unit: str,
        fuel_type: str
    ) -> float:
        """
        Convert fuel quantity to standard unit.

        Args:
            value: Quantity to convert
            from_unit: Source unit
            fuel_type: Type of fuel

        Returns:
            Quantity in standard unit for that fuel type
        """
        conversions = cls.FUEL_CONVERSIONS.get(fuel_type)
        if not conversions:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        factor = conversions.get(from_unit)
        if factor is None:
            raise ValueError(
                f"Unknown unit '{from_unit}' for fuel '{fuel_type}'. "
                f"Supported: {list(conversions.keys())}"
            )

        return value * factor

    @classmethod
    def kg_to_mt(cls, kg: float) -> float:
        """Convert kilograms to metric tonnes."""
        return kg / 1000.0

    @classmethod
    def mt_to_kg(cls, mt: float) -> float:
        """Convert metric tonnes to kilograms."""
        return mt * 1000.0
