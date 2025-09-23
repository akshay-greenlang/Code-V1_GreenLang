from typing import Any, Dict, List
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig


class InputValidatorAgent(BaseAgent):
    """Agent for validating input data for emissions calculations"""

    VALID_FUEL_TYPES = [
        "electricity",
        "natural_gas",
        "diesel",
        "gasoline",
        "propane",
        "fuel_oil",
        "coal",
        "biomass",
        "solar",
        "wind",
        "hydro",
    ]

    VALID_UNITS = {
        "electricity": ["kWh", "MWh", "GWh"],
        "natural_gas": ["therms", "ccf", "mcf", "m3", "MMBtu"],
        "diesel": ["gallons", "liters", "kg"],
        "gasoline": ["gallons", "liters", "kg"],
        "propane": ["gallons", "liters", "kg"],
        "fuel_oil": ["gallons", "liters", "kg"],
        "coal": ["tons", "kg", "lbs"],
        "biomass": ["tons", "kg"],
    }

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="InputValidatorAgent",
                description="Validates input data for emissions calculations",
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        validation_errors = []
        validated_data = {}
        warnings = []

        if "fuels" in input_data:
            validated_fuels, fuel_errors, fuel_warnings = self._validate_fuels(
                input_data["fuels"]
            )
            validated_data["fuels"] = validated_fuels
            validation_errors.extend(fuel_errors)
            warnings.extend(fuel_warnings)

        if "building_info" in input_data:
            validated_building, building_errors = self._validate_building_info(
                input_data["building_info"]
            )
            validated_data["building_info"] = validated_building
            validation_errors.extend(building_errors)

        if "period" in input_data:
            validated_period, period_errors = self._validate_period(
                input_data["period"]
            )
            validated_data["period"] = validated_period
            validation_errors.extend(period_errors)

        if validation_errors:
            return AgentResult(
                success=False,
                error="Validation failed",
                data={"errors": validation_errors, "warnings": warnings},
            )

        return AgentResult(
            success=True,
            data={
                "validated_data": validated_data,
                "warnings": warnings,
                "summary": f"Validated {len(validated_data.get('fuels', []))} fuel entries",
            },
            metadata={"agent": "InputValidatorAgent", "warnings_count": len(warnings)},
        )

    def _validate_fuels(self, fuels: List[Dict]) -> tuple:
        validated_fuels = []
        errors = []
        warnings = []

        for idx, fuel in enumerate(fuels):
            fuel_type = fuel.get("type", "").lower()
            consumption = fuel.get("consumption")
            unit = fuel.get("unit")

            if fuel_type not in self.VALID_FUEL_TYPES:
                errors.append(f"Fuel {idx+1}: Invalid fuel type '{fuel_type}'")
                continue

            if consumption is None:
                errors.append(f"Fuel {idx+1}: Missing consumption value")
                continue

            try:
                consumption = float(consumption)
                if consumption < 0:
                    errors.append(f"Fuel {idx+1}: Negative consumption value")
                    continue
                elif consumption == 0:
                    warnings.append(f"Fuel {idx+1}: Zero consumption value")
            except (ValueError, TypeError):
                errors.append(f"Fuel {idx+1}: Invalid consumption value")
                continue

            if unit not in self.VALID_UNITS.get(fuel_type, []):
                errors.append(f"Fuel {idx+1}: Invalid unit '{unit}' for {fuel_type}")
                continue

            validated_fuel = {
                "fuel_type": fuel_type,
                "consumption": consumption,
                "unit": unit,
                "region": fuel.get("region", "US"),
            }
            validated_fuels.append(validated_fuel)

        return validated_fuels, errors, warnings

    def _validate_building_info(self, building_info: Dict) -> tuple:
        validated = {}
        errors = []

        if "area" in building_info:
            try:
                area = float(building_info["area"])
                if area <= 0:
                    errors.append("Building area must be positive")
                else:
                    validated["area"] = area
                    validated["area_unit"] = building_info.get("area_unit", "sqft")
            except (ValueError, TypeError):
                errors.append("Invalid building area value")

        if "occupancy" in building_info:
            try:
                occupancy = int(building_info["occupancy"])
                if occupancy < 0:
                    errors.append("Occupancy cannot be negative")
                else:
                    validated["occupancy"] = occupancy
            except (ValueError, TypeError):
                errors.append("Invalid occupancy value")

        if "type" in building_info:
            validated["type"] = building_info["type"]

        return validated, errors

    def _validate_period(self, period: Dict) -> tuple:
        validated = {}
        errors = []

        if "start_date" in period:
            validated["start_date"] = period["start_date"]

        if "end_date" in period:
            validated["end_date"] = period["end_date"]

        if "duration" in period:
            validated["duration"] = period["duration"]
            validated["duration_unit"] = period.get("duration_unit", "month")

        return validated, errors
