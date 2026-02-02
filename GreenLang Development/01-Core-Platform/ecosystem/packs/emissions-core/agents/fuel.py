# -*- coding: utf-8 -*-
"""
Fuel Emissions Agent
====================

Domain-specific agent for calculating fuel-based emissions.
This is where domain logic lives, not in the core framework.
"""

from typing import Dict, Any
from greenlang.sdk.base import Agent, Result, Metadata
import json
from pathlib import Path


class FuelAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Calculate emissions from fuel consumption
    
    This agent contains domain-specific knowledge about:
    - Fuel types and their emission factors
    - Unit conversions for fuel quantities
    - Regional variations in fuel composition
    """
    
    def __init__(self):
        """Initialize with fuel-specific metadata"""
        super().__init__(
            metadata=Metadata(
                id="fuel-emissions",
                name="Fuel Emissions Calculator",
                version="1.0.0",
                description="Calculate CO2e emissions from fuel consumption",
                tags=["emissions", "fuel", "carbon"]
            )
        )
        
        # Load emission factors from pack data
        self.emission_factors = self._load_emission_factors()
    
    def _load_emission_factors(self) -> Dict[str, float]:
        """Load emission factors from pack data"""
        data_file = Path(__file__).parent.parent / "data" / "emission_factors.json"
        
        # Default factors if file doesn't exist
        default_factors = {
            "natural_gas": 0.18,  # kg CO2e/kWh
            "diesel": 2.68,        # kg CO2e/liter
            "gasoline": 2.31,      # kg CO2e/liter
            "coal": 0.34,          # kg CO2e/kWh
            "propane": 1.51,       # kg CO2e/liter
            "biomass": 0.0,        # Carbon neutral (simplified)
        }
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f).get("fuel_factors", default_factors)
        
        return default_factors
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate fuel emissions input"""
        required = ["fuel_type", "amount", "unit"]
        
        # Check required fields
        for field in required:
            if field not in input_data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Check fuel type is supported
        if input_data["fuel_type"] not in self.emission_factors:
            self.logger.error(f"Unsupported fuel type: {input_data['fuel_type']}")
            return False
        
        # Check amount is positive
        if input_data["amount"] <= 0:
            self.logger.error("Amount must be positive")
            return False
        
        return True
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fuel emissions"""
        fuel_type = input_data["fuel_type"]
        amount = input_data["amount"]
        unit = input_data["unit"]
        
        # Get emission factor
        factor = self.emission_factors[fuel_type]
        
        # Convert units if needed
        amount_standard = self._convert_to_standard_unit(amount, unit, fuel_type)
        
        # Calculate emissions
        co2e_kg = amount_standard * factor
        
        return {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "co2e_kg": round(co2e_kg, 2),
            "co2e_tons": round(co2e_kg / 1000, 3),
            "emission_factor": factor,
            "methodology": "IPCC 2006",
            "confidence": "high"
        }
    
    def _convert_to_standard_unit(self, amount: float, unit: str, fuel_type: str) -> float:
        """Convert to standard units for calculation"""
        # Simplified unit conversion
        conversions = {
            "gallons": 3.785,  # to liters
            "therms": 29.3,    # to kWh
            "mmbtu": 293.1,    # to kWh
            "kg": 1.0,         # already standard
            "liters": 1.0,     # already standard
            "kwh": 1.0,        # already standard
        }
        
        unit_lower = unit.lower()
        if unit_lower in conversions:
            return amount * conversions[unit_lower]
        
        return amount
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for input"""
        return {
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "enum": list(self.emission_factors.keys())
                },
                "amount": {
                    "type": "number",
                    "minimum": 0
                },
                "unit": {
                    "type": "string",
                    "enum": ["liters", "gallons", "kg", "kwh", "therms", "mmbtu"]
                }
            },
            "required": ["fuel_type", "amount", "unit"]
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for output"""
        return {
            "type": "object",
            "properties": {
                "fuel_type": {"type": "string"},
                "amount": {"type": "number"},
                "unit": {"type": "string"},
                "co2e_kg": {"type": "number"},
                "co2e_tons": {"type": "number"},
                "emission_factor": {"type": "number"},
                "methodology": {"type": "string"},
                "confidence": {"type": "string"}
            }
        }