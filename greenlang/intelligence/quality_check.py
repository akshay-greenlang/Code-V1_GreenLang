"""
Quality Check and Validation for LLM Responses

Validates LLM outputs and calculates confidence scores to trigger fallbacks:
- JSON format validation
- Required field checking
- Numeric range validation
- Hallucination detection
- Confidence scoring (0-1)
- Automatic fallback if confidence < threshold

Quality Checks:
1. Format Validation - Check JSON structure
2. Schema Validation - Verify required fields
3. Range Validation - Check numeric bounds
4. Semantic Validation - Cross-reference with knowledge base
5. Confidence Scoring - Calculate overall confidence

Triggers Fallback When:
- Confidence < 0.8
- Required fields missing
- Invalid data types
- Hallucination detected
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """
    Validation error details

    Attributes:
        field: Field that failed validation
        error_type: Type of error
        message: Error message
        expected: Expected value/format
        actual: Actual value
    """
    field: str
    error_type: str
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class QualityScore:
    """
    Quality score for LLM response

    Attributes:
        overall: Overall confidence score (0-1)
        format_score: Format validation score (0-1)
        schema_score: Schema validation score (0-1)
        range_score: Range validation score (0-1)
        semantic_score: Semantic validation score (0-1)
        errors: List of validation errors
        warnings: List of warnings
        metadata: Additional metadata
    """
    overall: float = 0.0
    format_score: float = 1.0
    schema_score: float = 1.0
    range_score: float = 1.0
    semantic_score: float = 1.0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None):
        """
        Calculate overall score as weighted average

        Args:
            weights: Custom weights for each score component
        """
        default_weights = {
            "format": 0.3,
            "schema": 0.3,
            "range": 0.2,
            "semantic": 0.2,
        }
        weights = weights or default_weights

        self.overall = (
            self.format_score * weights["format"] +
            self.schema_score * weights["schema"] +
            self.range_score * weights["range"] +
            self.semantic_score * weights["semantic"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall": self.overall,
            "format_score": self.format_score,
            "schema_score": self.schema_score,
            "range_score": self.range_score,
            "semantic_score": self.semantic_score,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class FormatValidator:
    """
    Validate response format

    Checks:
    - Valid JSON
    - Proper structure
    - No malformed data
    """

    def validate(self, response: str, expected_format: str = "json") -> Tuple[float, List[ValidationError]]:
        """
        Validate response format

        Args:
            response: LLM response text
            expected_format: Expected format (json, text, etc.)

        Returns:
            Tuple of (score, errors)
        """
        errors = []
        score = 1.0

        if expected_format == "json":
            # Check valid JSON
            try:
                json.loads(response)
            except json.JSONDecodeError as e:
                errors.append(ValidationError(
                    field="response",
                    error_type="invalid_json",
                    message=f"Invalid JSON: {e}",
                    expected="valid JSON",
                    actual=response[:100],
                ))
                score = 0.0

        elif expected_format == "text":
            # Check non-empty text
            if not response or not response.strip():
                errors.append(ValidationError(
                    field="response",
                    error_type="empty_response",
                    message="Response is empty",
                    expected="non-empty text",
                    actual=response,
                ))
                score = 0.0

        return score, errors


class SchemaValidator:
    """
    Validate response against schema

    Checks:
    - Required fields present
    - Correct data types
    - Valid enums
    """

    def validate(
        self,
        response: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Tuple[float, List[ValidationError]]:
        """
        Validate response against schema

        Args:
            response: Parsed response dictionary
            schema: Schema definition

        Returns:
            Tuple of (score, errors)
        """
        errors = []

        # Check required fields
        required_fields = schema.get("required", [])
        missing_fields = []

        for field in required_fields:
            if field not in response:
                missing_fields.append(field)
                errors.append(ValidationError(
                    field=field,
                    error_type="missing_required_field",
                    message=f"Required field '{field}' is missing",
                    expected=field,
                    actual=None,
                ))

        # Check data types
        properties = schema.get("properties", {})
        type_errors = 0

        for field, field_schema in properties.items():
            if field in response:
                expected_type = field_schema.get("type")
                actual_value = response[field]

                if not self._check_type(actual_value, expected_type):
                    type_errors += 1
                    errors.append(ValidationError(
                        field=field,
                        error_type="invalid_type",
                        message=f"Field '{field}' has wrong type",
                        expected=expected_type,
                        actual=type(actual_value).__name__,
                    ))

        # Calculate score
        total_checks = len(required_fields) + len(properties)
        if total_checks == 0:
            score = 1.0
        else:
            errors_count = len(missing_fields) + type_errors
            score = max(0.0, 1.0 - (errors_count / total_checks))

        return score, errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip check

        return isinstance(value, expected_python_type)


class RangeValidator:
    """
    Validate numeric ranges

    Checks:
    - Values within expected bounds
    - Positive values where required
    - Reasonable magnitude
    """

    def __init__(self):
        """Initialize range validator"""
        # Known reasonable ranges for climate data
        self.known_ranges = {
            "carbon_intensity": (0.0, 2.0),      # kg CO2/kWh
            "electricity_kwh": (0.0, 1000000.0), # kWh
            "natural_gas_kwh": (0.0, 1000000.0), # kWh
            "emissions_kg": (0.0, 1000000.0),    # kg CO2
            "temperature_c": (-50.0, 50.0),      # Celsius
            "efficiency": (0.0, 1.0),            # 0-100%
            "cop": (1.0, 6.0),                   # Coefficient of Performance
        }

    def validate(
        self,
        response: Dict[str, Any],
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[float, List[ValidationError]]:
        """
        Validate numeric ranges

        Args:
            response: Parsed response
            ranges: Custom range constraints

        Returns:
            Tuple of (score, errors)
        """
        ranges = ranges or {}
        errors = []
        total_checks = 0
        passed_checks = 0

        # Combine known ranges with custom ranges
        all_ranges = {**self.known_ranges, **ranges}

        # Check each field
        for field, (min_val, max_val) in all_ranges.items():
            if field in response:
                value = response[field]

                # Skip non-numeric values
                if not isinstance(value, (int, float)):
                    continue

                total_checks += 1

                # Check range
                if value < min_val or value > max_val:
                    errors.append(ValidationError(
                        field=field,
                        error_type="out_of_range",
                        message=f"Value {value} is outside expected range [{min_val}, {max_val}]",
                        expected=f"[{min_val}, {max_val}]",
                        actual=value,
                    ))
                else:
                    passed_checks += 1

        # Calculate score
        if total_checks == 0:
            score = 1.0
        else:
            score = passed_checks / total_checks

        return score, errors


class SemanticValidator:
    """
    Validate semantic correctness

    Checks:
    - Cross-reference with knowledge base
    - Detect obvious hallucinations
    - Check consistency with known facts
    """

    def __init__(self):
        """Initialize semantic validator"""
        # Known facts for climate domain
        self.known_facts = {
            # Grid carbon intensity (kg CO2/kWh)
            "grid_intensity_us_avg": 0.4,
            "grid_intensity_california": 0.24,
            "grid_intensity_coal": 0.9,
            "grid_intensity_renewable": 0.05,

            # Fuel carbon intensity (kg CO2/kWh)
            "natural_gas_intensity": 0.185,
            "heating_oil_intensity": 0.265,
            "propane_intensity": 0.215,

            # Common conversions
            "kwh_per_btu": 0.000293,
            "kg_per_tonne": 1000,
        }

    def validate(
        self,
        response: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, List[ValidationError]]:
        """
        Validate semantic correctness

        Args:
            response: Parsed response
            context: Context for validation

        Returns:
            Tuple of (score, errors)
        """
        errors = []
        warnings = []
        total_checks = 0
        passed_checks = 0

        # Check carbon intensity values
        if "carbon_intensity" in response:
            value = response["carbon_intensity"]
            if isinstance(value, (int, float)):
                total_checks += 1

                # Check if within reasonable range
                if value > 2.0:
                    errors.append(ValidationError(
                        field="carbon_intensity",
                        error_type="hallucination",
                        message=f"Carbon intensity {value} is unreasonably high (max ~2.0 kg/kWh for coal)",
                        expected="< 2.0",
                        actual=value,
                    ))
                else:
                    passed_checks += 1

        # Check emissions calculations
        if "emissions_kg" in response and "electricity_kwh" in response:
            total_checks += 1

            emissions = response["emissions_kg"]
            electricity = response["electricity_kwh"]

            # Estimate expected emissions (US avg grid)
            expected_emissions = electricity * self.known_facts["grid_intensity_us_avg"]

            # Allow 50% variance
            if abs(emissions - expected_emissions) / expected_emissions > 0.5:
                warnings.append(
                    f"Emissions calculation may be inaccurate: "
                    f"expected ~{expected_emissions:.0f} kg, got {emissions:.0f} kg"
                )
            else:
                passed_checks += 1

        # Calculate score
        if total_checks == 0:
            score = 1.0
        else:
            score = passed_checks / total_checks

        return score, errors


class QualityChecker:
    """
    Main quality checker for LLM responses

    Combines multiple validators and calculates overall confidence score.
    """

    def __init__(
        self,
        min_confidence: float = 0.8,
        enable_format_check: bool = True,
        enable_schema_check: bool = True,
        enable_range_check: bool = True,
        enable_semantic_check: bool = True,
    ):
        """
        Initialize quality checker

        Args:
            min_confidence: Minimum confidence for acceptance
            enable_format_check: Enable format validation
            enable_schema_check: Enable schema validation
            enable_range_check: Enable range validation
            enable_semantic_check: Enable semantic validation
        """
        self.min_confidence = min_confidence

        # Initialize validators
        self.format_validator = FormatValidator() if enable_format_check else None
        self.schema_validator = SchemaValidator() if enable_schema_check else None
        self.range_validator = RangeValidator() if enable_range_check else None
        self.semantic_validator = SemanticValidator() if enable_semantic_check else None

        logger.info(f"QualityChecker initialized (min_confidence={min_confidence})")

    def check(
        self,
        response: str,
        expected_format: str = "json",
        schema: Optional[Dict[str, Any]] = None,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> QualityScore:
        """
        Check response quality

        Args:
            response: LLM response text
            expected_format: Expected format
            schema: Schema for validation
            ranges: Range constraints
            context: Context for semantic validation

        Returns:
            Quality score with confidence and errors
        """
        quality_score = QualityScore()

        # 1. Format validation
        if self.format_validator:
            format_score, format_errors = self.format_validator.validate(response, expected_format)
            quality_score.format_score = format_score
            quality_score.errors.extend(format_errors)

            # Stop if format check failed
            if format_score == 0.0:
                quality_score.calculate_overall()
                return quality_score

        # Parse response for further checks
        try:
            if expected_format == "json":
                parsed_response = json.loads(response)
            else:
                parsed_response = {"text": response}
        except json.JSONDecodeError:
            # Already caught in format validation
            quality_score.calculate_overall()
            return quality_score

        # 2. Schema validation
        if self.schema_validator and schema:
            schema_score, schema_errors = self.schema_validator.validate(parsed_response, schema)
            quality_score.schema_score = schema_score
            quality_score.errors.extend(schema_errors)

        # 3. Range validation
        if self.range_validator:
            range_score, range_errors = self.range_validator.validate(parsed_response, ranges)
            quality_score.range_score = range_score
            quality_score.errors.extend(range_errors)

        # 4. Semantic validation
        if self.semantic_validator:
            semantic_score, semantic_errors = self.semantic_validator.validate(parsed_response, context)
            quality_score.semantic_score = semantic_score
            quality_score.errors.extend(semantic_errors)

        # Calculate overall score
        quality_score.calculate_overall()

        # Add metadata
        quality_score.metadata["timestamp"] = datetime.now().isoformat()
        quality_score.metadata["response_length"] = len(response)
        quality_score.metadata["num_errors"] = len(quality_score.errors)

        logger.info(f"Quality check complete: confidence={quality_score.overall:.3f}, errors={len(quality_score.errors)}")

        return quality_score

    def should_fallback(self, quality_score: QualityScore) -> bool:
        """
        Determine if fallback should be triggered

        Args:
            quality_score: Quality score

        Returns:
            True if should fallback
        """
        return quality_score.overall < self.min_confidence


if __name__ == "__main__":
    """
    Demo and testing
    """
    print("=" * 80)
    print("GreenLang Quality Checker Demo")
    print("=" * 80)

    # Initialize quality checker
    checker = QualityChecker(min_confidence=0.8)

    # Test cases
    test_cases = [
        # Valid response
        {
            "response": json.dumps({
                "carbon_intensity": 0.185,
                "electricity_kwh": 1000,
                "emissions_kg": 400,
            }),
            "schema": {
                "required": ["carbon_intensity", "electricity_kwh", "emissions_kg"],
                "properties": {
                    "carbon_intensity": {"type": "number"},
                    "electricity_kwh": {"type": "number"},
                    "emissions_kg": {"type": "number"},
                }
            },
            "description": "Valid response",
        },

        # Missing required field
        {
            "response": json.dumps({
                "carbon_intensity": 0.185,
            }),
            "schema": {
                "required": ["carbon_intensity", "electricity_kwh"],
                "properties": {
                    "carbon_intensity": {"type": "number"},
                    "electricity_kwh": {"type": "number"},
                }
            },
            "description": "Missing required field",
        },

        # Out of range
        {
            "response": json.dumps({
                "carbon_intensity": 10.0,  # Too high!
                "electricity_kwh": 1000,
            }),
            "schema": {
                "required": ["carbon_intensity", "electricity_kwh"],
                "properties": {
                    "carbon_intensity": {"type": "number"},
                    "electricity_kwh": {"type": "number"},
                }
            },
            "description": "Out of range value",
        },

        # Invalid JSON
        {
            "response": "This is not valid JSON {incomplete",
            "schema": {},
            "description": "Invalid JSON",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}:")
        print(f"   Response: {test['response'][:80]}...")

        quality_score = checker.check(
            response=test["response"],
            schema=test.get("schema"),
        )

        print(f"   Overall confidence: {quality_score.overall:.3f}")
        print(f"   Format score: {quality_score.format_score:.3f}")
        print(f"   Schema score: {quality_score.schema_score:.3f}")
        print(f"   Range score: {quality_score.range_score:.3f}")
        print(f"   Should fallback: {checker.should_fallback(quality_score)}")
        if quality_score.errors:
            print(f"   Errors:")
            for error in quality_score.errors:
                print(f"     - {error.error_type}: {error.message}")

    print("\n" + "=" * 80)
