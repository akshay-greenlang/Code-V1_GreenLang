#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 6: Schema and Business Rules Validation
================================================

This example demonstrates comprehensive validation:
- JSON Schema validation for structure
- Business rules validation for logic
- Data quality checks
- Error reporting and handling

Run: python examples/06_validation_framework.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from greenlang.sdk.base import Validator, Result, Agent, Metadata


class SchemaValidator(Validator[Dict[str, Any]]):
    """
    JSON Schema validator for building energy data.
    """

    def __init__(self):
        self.schema = {
            "type": "object",
            "required": ["building_id", "energy_consumption"],
            "properties": {
                "building_id": {"type": "string", "minLength": 1},
                "building_name": {"type": "string"},
                "area_sqm": {"type": "number", "minimum": 0},
                "energy_consumption": {
                    "type": "object",
                    "required": ["electricity", "gas"],
                    "properties": {
                        "electricity": {
                            "type": "object",
                            "required": ["value", "unit"],
                            "properties": {
                                "value": {"type": "number", "minimum": 0},
                                "unit": {"type": "string", "enum": ["kWh", "MWh"]}
                            }
                        },
                        "gas": {
                            "type": "object",
                            "required": ["value", "unit"],
                            "properties": {
                                "value": {"type": "number", "minimum": 0},
                                "unit": {"type": "string", "enum": ["therms", "m3"]}
                            }
                        }
                    }
                },
                "location": {"type": "string"},
                "year": {"type": "integer", "minimum": 2000, "maximum": 2100}
            }
        }

    def validate(self, data: Dict[str, Any]) -> Result:
        """Validate data against schema"""
        errors = []

        # Check required fields
        for field in self.schema["required"]:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Validate building_id
        if "building_id" in data:
            if not isinstance(data["building_id"], str) or len(data["building_id"]) == 0:
                errors.append("building_id must be a non-empty string")

        # Validate energy_consumption
        if "energy_consumption" in data:
            energy = data["energy_consumption"]
            if not isinstance(energy, dict):
                errors.append("energy_consumption must be an object")
            else:
                for fuel in ["electricity", "gas"]:
                    if fuel not in energy:
                        errors.append(f"Missing required field: energy_consumption.{fuel}")
                    elif not isinstance(energy[fuel], dict):
                        errors.append(f"energy_consumption.{fuel} must be an object")
                    else:
                        # Validate value and unit
                        if "value" not in energy[fuel]:
                            errors.append(f"Missing required field: energy_consumption.{fuel}.value")
                        elif not isinstance(energy[fuel]["value"], (int, float)) or energy[fuel]["value"] < 0:
                            errors.append(f"energy_consumption.{fuel}.value must be a non-negative number")

                        if "unit" not in energy[fuel]:
                            errors.append(f"Missing required field: energy_consumption.{fuel}.unit")

        # Validate optional fields
        if "area_sqm" in data:
            if not isinstance(data["area_sqm"], (int, float)) or data["area_sqm"] < 0:
                errors.append("area_sqm must be a non-negative number")

        if "year" in data:
            if not isinstance(data["year"], int) or data["year"] < 2000 or data["year"] > 2100:
                errors.append("year must be an integer between 2000 and 2100")

        if errors:
            return Result(
                success=False,
                error="Schema validation failed",
                metadata={"validation_errors": errors}
            )

        return Result(success=True, data={"message": "Schema validation passed"})


class BusinessRulesValidator(Validator[Dict[str, Any]]):
    """
    Business rules validator for data quality and logic.
    """

    def __init__(self):
        self.rules = [
            self._check_energy_intensity,
            self._check_reasonable_consumption,
            self._check_data_completeness,
            self._check_temporal_consistency
        ]

    def validate(self, data: Dict[str, Any]) -> Result:
        """Apply business rules validation"""
        warnings = []
        errors = []

        for rule in self.rules:
            result = rule(data)
            if result["type"] == "error":
                errors.append(result["message"])
            elif result["type"] == "warning":
                warnings.append(result["message"])

        if errors:
            return Result(
                success=False,
                error="Business rules validation failed",
                metadata={"errors": errors, "warnings": warnings}
            )

        return Result(
            success=True,
            data={"message": "Business rules validation passed", "warnings": warnings}
        )

    def _check_energy_intensity(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Check if energy intensity is reasonable"""
        if "area_sqm" not in data or data["area_sqm"] == 0:
            return {"type": "info", "message": "Cannot calculate intensity without area"}

        energy = data.get("energy_consumption", {})
        elec_kwh = energy.get("electricity", {}).get("value", 0)

        # Convert MWh to kWh if needed
        if energy.get("electricity", {}).get("unit") == "MWh":
            elec_kwh *= 1000

        intensity = elec_kwh / data["area_sqm"]

        # Typical office: 100-300 kWh/sqm/year
        if intensity > 500:
            return {"type": "error", "message": f"Energy intensity too high: {intensity:.1f} kWh/sqm (max 500)"}
        elif intensity > 300:
            return {"type": "warning", "message": f"Energy intensity high: {intensity:.1f} kWh/sqm (typical max 300)"}

        return {"type": "info", "message": "Energy intensity is reasonable"}

    def _check_reasonable_consumption(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Check if consumption values are reasonable"""
        energy = data.get("energy_consumption", {})

        elec_kwh = energy.get("electricity", {}).get("value", 0)
        if energy.get("electricity", {}).get("unit") == "MWh":
            elec_kwh *= 1000

        # Check for unreasonably high consumption
        if elec_kwh > 10000000:  # 10 GWh
            return {"type": "error", "message": f"Electricity consumption unreasonably high: {elec_kwh:,.0f} kWh"}

        # Check for suspiciously low consumption
        if elec_kwh < 100 and elec_kwh > 0:
            return {"type": "warning", "message": f"Electricity consumption very low: {elec_kwh:,.0f} kWh"}

        return {"type": "info", "message": "Consumption values are reasonable"}

    def _check_data_completeness(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Check data completeness and quality"""
        recommended_fields = ["building_name", "location", "year", "area_sqm"]
        missing = [f for f in recommended_fields if f not in data or not data[f]]

        if len(missing) > 2:
            return {"type": "warning", "message": f"Missing recommended fields: {', '.join(missing)}"}

        return {"type": "info", "message": "Data completeness is good"}

    def _check_temporal_consistency(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Check temporal consistency"""
        if "year" in data:
            current_year = 2025
            if data["year"] > current_year:
                return {"type": "error", "message": f"Future year not allowed: {data['year']}"}
            elif data["year"] < current_year - 5:
                return {"type": "warning", "message": f"Data is old: {data['year']} (>5 years)"}

        return {"type": "info", "message": "Temporal consistency is good"}


class ValidatedCalculatorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Calculator with comprehensive validation.
    """

    def __init__(self):
        metadata = Metadata(
            id="validated_calculator",
            name="Validated Calculator Agent",
            version="1.0.0",
            description="Calculator with schema and business rules validation",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        self.schema_validator = SchemaValidator()
        self.business_validator = BusinessRulesValidator()

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Multi-stage validation"""
        # Stage 1: Schema validation
        schema_result = self.schema_validator.validate(input_data)
        if not schema_result.success:
            self.logger.error(f"Schema validation failed: {schema_result.metadata.get('validation_errors')}")
            return False

        # Stage 2: Business rules validation
        business_result = self.business_validator.validate(input_data)
        if not business_result.success:
            self.logger.error(f"Business rules failed: {business_result.metadata.get('errors')}")
            return False

        # Log warnings if any
        if business_result.data and business_result.data.get("warnings"):
            for warning in business_result.data["warnings"]:
                self.logger.warning(warning)

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with validation"""
        # Simplified calculation (validation already done)
        return {
            "building_id": input_data["building_id"],
            "validation_passed": True,
            "message": "Calculation completed successfully"
        }


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 6: Schema and Business Rules Validation")
    print("="*70 + "\n")

    # Test 1: Valid data
    print("Test 1: Valid Data")
    print("-" * 70)

    valid_data = {
        "building_id": "B001",
        "building_name": "Office Tower A",
        "area_sqm": 5000,
        "energy_consumption": {
            "electricity": {"value": 50000, "unit": "kWh"},
            "gas": {"value": 1000, "unit": "therms"}
        },
        "location": "San Francisco",
        "year": 2024
    }

    agent = ValidatedCalculatorAgent()
    result = agent.run(valid_data)
    print(f"Success: {result.success}")
    if result.success:
        print(f"Data: {result.data}")

    # Test 2: Missing required fields
    print("\n\nTest 2: Missing Required Fields")
    print("-" * 70)

    invalid_data = {
        "building_name": "Test Building"
    }

    result = agent.run(invalid_data)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 3: Invalid values
    print("\n\nTest 3: Invalid Values (Negative Consumption)")
    print("-" * 70)

    bad_values = {
        "building_id": "B002",
        "area_sqm": 5000,
        "energy_consumption": {
            "electricity": {"value": -1000, "unit": "kWh"},
            "gas": {"value": 500, "unit": "therms"}
        }
    }

    result = agent.run(bad_values)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 4: Business rules warning
    print("\n\nTest 4: Business Rules Warning (High Intensity)")
    print("-" * 70)

    high_intensity = {
        "building_id": "B003",
        "area_sqm": 100,
        "energy_consumption": {
            "electricity": {"value": 40000, "unit": "kWh"},
            "gas": {"value": 100, "unit": "therms"}
        },
        "year": 2024
    }

    result = agent.run(high_intensity)
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
