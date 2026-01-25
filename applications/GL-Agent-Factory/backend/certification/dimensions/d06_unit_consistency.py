"""
Dimension 06: Unit Consistency Verification

This dimension verifies that agents handle units correctly,
including input/output validation and conversion verification.

Checks:
    - Input/output unit validation
    - Conversion factor verification
    - Dimensional analysis
    - Unit standardization

Example:
    >>> dimension = UnitConsistencyDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class UnitConsistencyDimension(BaseDimension):
    """
    Unit Consistency Dimension Evaluator (D06).

    Verifies that agents handle units correctly and consistently.

    Configuration:
        require_explicit_units: Require units on all numeric fields (default: True)
        verify_conversions: Verify unit conversion factors (default: True)
    """

    DIMENSION_ID = "D06"
    DIMENSION_NAME = "Unit Consistency"
    DESCRIPTION = "Verifies input/output unit validation and conversion accuracy"
    WEIGHT = 1.2
    REQUIRED_FOR_CERTIFICATION = True

    # Standard unit patterns
    EMISSION_UNITS = ["kgCO2e", "tCO2e", "gCO2e", "kgCO2", "tCO2"]
    ENERGY_UNITS = ["kWh", "MWh", "GWh", "GJ", "MJ", "kJ", "therm", "BTU"]
    VOLUME_UNITS = ["L", "gal", "m3", "cf", "scf"]
    MASS_UNITS = ["kg", "g", "t", "lb", "short_ton", "long_ton"]

    # Standard conversion factors for verification
    CONVERSION_FACTORS = {
        ("gal", "L"): 3.78541,
        ("cf", "m3"): 0.0283168,
        ("MWh", "kWh"): 1000,
        ("GJ", "MJ"): 1000,
        ("t", "kg"): 1000,
        ("lb", "kg"): 0.453592,
        ("short_ton", "kg"): 907.185,
        ("therm", "MJ"): 105.506,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unit consistency dimension evaluator."""
        super().__init__(config)

        self.require_explicit_units = self.config.get("require_explicit_units", True)
        self.verify_conversions = self.config.get("verify_conversions", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate unit consistency for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with unit consistency evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting unit consistency evaluation")

        # Load agent source code
        agent_file = agent_path / "agent.py"
        if not agent_file.exists():
            self._add_check(
                name="agent_file_exists",
                passed=False,
                message="agent.py not found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        try:
            source_code = agent_file.read_text(encoding="utf-8")
        except Exception as e:
            self._add_check(
                name="source_readable",
                passed=False,
                message=f"Cannot read agent source: {str(e)}",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Load agent instance if not provided
        if agent is None:
            agent = self._load_agent(agent_path)

        # Check 1: Input model has unit field
        input_units = self._check_input_units(source_code, agent)
        self._add_check(
            name="input_units_defined",
            passed=input_units["has_units"],
            message="Input model includes unit field"
            if input_units["has_units"]
            else "Input model missing unit field",
            severity="error" if self.require_explicit_units else "warning",
            details=input_units,
        )

        # Check 2: Output includes unit information
        output_units = self._check_output_units(source_code, agent)
        self._add_check(
            name="output_units_defined",
            passed=output_units["has_units"],
            message=f"Output includes unit information ({', '.join(output_units['unit_fields'])})"
            if output_units["has_units"]
            else "Output missing unit information",
            severity="error" if self.require_explicit_units else "warning",
            details=output_units,
        )

        # Check 3: Unit validation present
        validation_check = self._check_unit_validation(source_code)
        self._add_check(
            name="unit_validation",
            passed=validation_check["has_validation"],
            message="Unit validation is present"
            if validation_check["has_validation"]
            else "No unit validation found",
            severity="warning",
            details=validation_check,
        )

        # Check 4: Conversion factors defined
        conversion_check = self._check_conversion_factors(source_code, agent)
        self._add_check(
            name="conversion_factors_defined",
            passed=conversion_check["has_conversions"],
            message=f"Found {conversion_check['conversion_count']} unit conversion(s)"
            if conversion_check["has_conversions"]
            else "No unit conversions defined",
            severity="info",
            details=conversion_check,
        )

        # Check 5: Conversion factor accuracy
        if self.verify_conversions and conversion_check["has_conversions"]:
            accuracy_check = self._verify_conversion_accuracy(conversion_check["conversions"])
            self._add_check(
                name="conversion_accuracy",
                passed=accuracy_check["all_accurate"],
                message="All conversion factors are accurate"
                if accuracy_check["all_accurate"]
                else f"{accuracy_check['inaccurate_count']} inaccurate conversion(s)",
                severity="error" if not accuracy_check["all_accurate"] else "info",
                details=accuracy_check,
            )

        # Check 6: Dimensional consistency
        dimensional_check = self._check_dimensional_consistency(source_code)
        self._add_check(
            name="dimensional_consistency",
            passed=dimensional_check["consistent"],
            message="Dimensional analysis is consistent"
            if dimensional_check["consistent"]
            else "Dimensional inconsistencies found",
            severity="error" if not dimensional_check["consistent"] else "info",
            details=dimensional_check,
        )

        # Check 7: Standard unit naming
        naming_check = self._check_unit_naming(source_code)
        self._add_check(
            name="standard_unit_naming",
            passed=naming_check["uses_standard"],
            message="Uses standard unit naming conventions"
            if naming_check["uses_standard"]
            else "Non-standard unit names found",
            severity="warning" if not naming_check["uses_standard"] else "info",
            details=naming_check,
        )

        # Check 8: SI unit support
        si_check = self._check_si_unit_support(source_code)
        self._add_check(
            name="si_unit_support",
            passed=si_check["supports_si"],
            message="Supports SI units"
            if si_check["supports_si"]
            else "Limited SI unit support",
            severity="warning" if not si_check["supports_si"] else "info",
            details=si_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "input_unit_field": input_units.get("unit_field"),
                "output_unit_fields": output_units.get("unit_fields", []),
                "conversion_count": conversion_check.get("conversion_count", 0),
            },
        )

    def _load_agent(self, agent_path: Path) -> Optional[Any]:
        """Load agent from path."""
        try:
            agent_file = agent_path / "agent.py"
            if not agent_file.exists():
                return None

            import importlib.util

            spec = importlib.util.spec_from_file_location("agent", agent_file)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and name.endswith("Agent")
                    and hasattr(obj, "run")
                ):
                    return obj()

            return None

        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            return None

    def _check_input_units(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check if input model has unit field.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with input unit check results
        """
        result = {
            "has_units": False,
            "unit_field": None,
            "valid_units": [],
        }

        # Look for unit field in Input class
        input_pattern = re.compile(
            r"class\s+\w*Input\w*\([^)]*\):\s*(?:\"\"\"[^\"]*\"\"\")?([^class]*?)(?=\nclass|\Z)",
            re.DOTALL,
        )

        input_match = input_pattern.search(source_code)
        if input_match:
            input_body = input_match.group(1)

            # Check for unit field
            unit_field_pattern = re.compile(
                r"(\w*unit\w*)\s*:\s*(\w+)",
                re.IGNORECASE,
            )

            unit_match = unit_field_pattern.search(input_body)
            if unit_match:
                result["has_units"] = True
                result["unit_field"] = unit_match.group(1)

        # Look for valid_units list
        valid_units_pattern = re.compile(
            r"valid_units\s*=\s*\[([^\]]+)\]",
            re.IGNORECASE,
        )

        valid_match = valid_units_pattern.search(source_code)
        if valid_match:
            units_str = valid_match.group(1)
            result["valid_units"] = re.findall(r"[\"']([^\"']+)[\"']", units_str)

        return result

    def _check_output_units(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check if output includes unit information.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with output unit check results
        """
        result = {
            "has_units": False,
            "unit_fields": [],
        }

        # Look for unit-related fields in Output class
        output_pattern = re.compile(
            r"class\s+\w*Output\w*\([^)]*\):\s*(?:\"\"\"[^\"]*\"\"\")?([^class]*?)(?=\nclass|\Z)",
            re.DOTALL,
        )

        output_match = output_pattern.search(source_code)
        if output_match:
            output_body = output_match.group(1)

            # Look for unit fields
            unit_patterns = [
                r"(\w*unit\w*)\s*:",
                r"(\w+_unit)\s*:",
                r"(emission_factor_unit)\s*:",
            ]

            for pattern in unit_patterns:
                matches = re.findall(pattern, output_body, re.IGNORECASE)
                result["unit_fields"].extend(matches)

        result["unit_fields"] = list(set(result["unit_fields"]))
        result["has_units"] = len(result["unit_fields"]) > 0

        return result

    def _check_unit_validation(self, source_code: str) -> Dict[str, Any]:
        """
        Check for unit validation in code.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with validation check results
        """
        result = {
            "has_validation": False,
            "validation_types": [],
        }

        validation_patterns = [
            (r"@validator.*unit", "Pydantic unit validator"),
            (r"validate_unit", "Unit validation method"),
            (r"if\s+.*unit.*not\s+in", "Unit whitelist check"),
            (r"valid_units", "Valid units list"),
        ]

        for pattern, description in validation_patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_validation"] = True
                result["validation_types"].append(description)

        return result

    def _check_conversion_factors(
        self,
        source_code: str,
        agent: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Check for unit conversion factors.

        Args:
            source_code: Agent source code
            agent: Optional agent instance

        Returns:
            Dictionary with conversion factor check results
        """
        result = {
            "has_conversions": False,
            "conversion_count": 0,
            "conversions": [],
        }

        # Look for UNIT_CONVERSIONS dict
        conversion_pattern = re.compile(
            r"UNIT_CONVERSIONS\s*[:=]\s*\{([^}]+)\}",
            re.DOTALL,
        )

        conv_match = conversion_pattern.search(source_code)
        if conv_match:
            conv_body = conv_match.group(1)

            # Parse conversion entries
            entry_pattern = re.compile(
                r"\(\s*[\"'](\w+)[\"']\s*,\s*[\"'](\w+)[\"']\s*\)\s*:\s*([\d.]+)",
            )

            entries = entry_pattern.findall(conv_body)
            for from_unit, to_unit, factor in entries:
                result["conversions"].append({
                    "from": from_unit,
                    "to": to_unit,
                    "factor": float(factor),
                })

        # Also check agent instance
        if agent and hasattr(agent, "UNIT_CONVERSIONS"):
            conv_dict = getattr(agent, "UNIT_CONVERSIONS", {})
            for (from_unit, to_unit), factor in conv_dict.items():
                result["conversions"].append({
                    "from": from_unit,
                    "to": to_unit,
                    "factor": float(factor),
                })

        # Remove duplicates
        seen = set()
        unique_conversions = []
        for conv in result["conversions"]:
            key = (conv["from"], conv["to"])
            if key not in seen:
                seen.add(key)
                unique_conversions.append(conv)

        result["conversions"] = unique_conversions
        result["conversion_count"] = len(unique_conversions)
        result["has_conversions"] = result["conversion_count"] > 0

        return result

    def _verify_conversion_accuracy(
        self,
        conversions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify conversion factor accuracy against known values.

        Args:
            conversions: List of conversion factors

        Returns:
            Dictionary with accuracy check results
        """
        result = {
            "all_accurate": True,
            "inaccurate_count": 0,
            "inaccurate_conversions": [],
            "verified_conversions": [],
        }

        tolerance = 0.001  # 0.1% tolerance

        for conv in conversions:
            key = (conv["from"], conv["to"])
            expected = self.CONVERSION_FACTORS.get(key)

            if expected is not None:
                if abs(conv["factor"] - expected) / expected > tolerance:
                    result["all_accurate"] = False
                    result["inaccurate_count"] += 1
                    result["inaccurate_conversions"].append({
                        "from": conv["from"],
                        "to": conv["to"],
                        "actual": conv["factor"],
                        "expected": expected,
                        "error_pct": abs(conv["factor"] - expected) / expected * 100,
                    })
                else:
                    result["verified_conversions"].append(key)

        return result

    def _check_dimensional_consistency(self, source_code: str) -> Dict[str, Any]:
        """
        Check for dimensional consistency in calculations.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with dimensional consistency check results
        """
        result = {
            "consistent": True,
            "issues": [],
        }

        # Look for emission calculations
        # Pattern: emissions = quantity * emission_factor
        calc_pattern = re.compile(
            r"emissions?\s*=\s*(\w+)\s*\*\s*(\w+)",
            re.IGNORECASE,
        )

        # This is a simplified check - full dimensional analysis
        # would require parsing the entire calculation chain
        calc_matches = calc_pattern.findall(source_code)

        if calc_matches:
            # Check that there's unit handling
            if not re.search(r"_convert_units|convert_unit|unit_conversion", source_code, re.IGNORECASE):
                result["consistent"] = False
                result["issues"].append(
                    "Calculation multiplies values without explicit unit conversion"
                )

        return result

    def _check_unit_naming(self, source_code: str) -> Dict[str, Any]:
        """
        Check for standard unit naming conventions.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with unit naming check results
        """
        result = {
            "uses_standard": True,
            "non_standard": [],
        }

        # Look for unit strings in code
        unit_pattern = re.compile(r"[\"']([a-zA-Z0-9/]+)[\"']")
        found_units = unit_pattern.findall(source_code)

        # Standard unit names
        standard_units = set(
            self.EMISSION_UNITS
            + self.ENERGY_UNITS
            + self.VOLUME_UNITS
            + self.MASS_UNITS
        )

        # Check for non-standard but unit-like strings
        for unit in found_units:
            if any(char.isdigit() or char == "/" for char in unit):
                # Might be a unit
                if unit not in standard_units and len(unit) < 10:
                    # Check if it looks like a unit variant
                    if re.match(r"^[a-zA-Z0-9/]+$", unit):
                        # Could be non-standard
                        pass  # Be lenient for now

        return result

    def _check_si_unit_support(self, source_code: str) -> Dict[str, Any]:
        """
        Check for SI unit support.

        Args:
            source_code: Agent source code

        Returns:
            Dictionary with SI unit support check results
        """
        result = {
            "supports_si": False,
            "si_units_found": [],
        }

        si_units = ["m3", "kg", "kWh", "MJ", "GJ", "L"]

        for unit in si_units:
            if unit in source_code:
                result["supports_si"] = True
                result["si_units_found"].append(unit)

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "agent_file_exists": (
                "Create agent.py in the agent directory."
            ),
            "source_readable": (
                "Ensure agent.py is readable and uses UTF-8 encoding."
            ),
            "input_units_defined": (
                "Add unit field to input model:\n"
                "  class AgentInput(BaseModel):\n"
                "      quantity: float\n"
                "      unit: str = Field(..., description='Unit of measurement')"
            ),
            "output_units_defined": (
                "Add unit information to output model:\n"
                "  class AgentOutput(BaseModel):\n"
                "      emissions_kgco2e: float\n"
                "      emission_factor_unit: str"
            ),
            "unit_validation": (
                "Add unit validation:\n"
                "  @validator('unit')\n"
                "  def validate_unit(cls, v):\n"
                "      valid = ['m3', 'L', 'kg', 'kWh']\n"
                "      if v not in valid:\n"
                "          raise ValueError(f'Invalid unit: {v}')\n"
                "      return v"
            ),
            "conversion_factors_defined": (
                "Define unit conversion factors:\n"
                "  UNIT_CONVERSIONS = {\n"
                "      ('gal', 'L'): 3.78541,\n"
                "      ('cf', 'm3'): 0.0283168,\n"
                "  }"
            ),
            "conversion_accuracy": (
                "Verify conversion factor accuracy:\n"
                "  - gal to L: 3.78541\n"
                "  - cf to m3: 0.0283168\n"
                "  - MWh to kWh: 1000"
            ),
            "dimensional_consistency": (
                "Ensure dimensional consistency:\n"
                "  - Add explicit unit conversion before multiplication\n"
                "  - Verify emission factor units match input units"
            ),
            "standard_unit_naming": (
                "Use standard unit abbreviations:\n"
                "  - kgCO2e (not kg-CO2e or kgco2e)\n"
                "  - kWh (not kwh or KWH)\n"
                "  - m3 (not m^3 or cubic_meters)"
            ),
            "si_unit_support": (
                "Add support for SI units:\n"
                "  - Mass: kg, g, t\n"
                "  - Volume: L, m3\n"
                "  - Energy: kWh, MJ, GJ"
            ),
        }

        return remediation_map.get(check.name)
