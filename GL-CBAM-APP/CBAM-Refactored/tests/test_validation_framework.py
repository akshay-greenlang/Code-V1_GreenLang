"""
Validation Framework Test Suite

Tests the GreenLang validation framework components:
- ValidationFramework
- Schema validation (JSON Schema)
- Business rules engine
- ValidationException handling
- Custom validators
- Validation reporting

Validates replacement of custom validation logic with framework

Author: GreenLang CBAM Team
Date: 2025-10-16
"""

import json
import pytest
import yaml
from pathlib import Path
from typing import Dict, List

# Import framework validation
from greenlang.validation import (
    ValidationFramework,
    ValidationException,
    ValidationIssue,
    SchemaValidator,
    RulesEngine
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "validation_test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def shipment_schema(test_data_dir):
    """Create JSON schema for CBAM shipment validation."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["shipment_id", "import_date", "cn_code", "net_mass_kg"],
        "properties": {
            "shipment_id": {
                "type": "string",
                "minLength": 1
            },
            "import_date": {
                "type": "string",
                "format": "date"
            },
            "quarter": {
                "type": "string",
                "pattern": "^Q[1-4]-20[0-9]{2}$"
            },
            "cn_code": {
                "type": "string",
                "pattern": "^[0-9]{8}$"
            },
            "net_mass_kg": {
                "type": "number",
                "minimum": 0.001,
                "exclusiveMinimum": 0
            },
            "origin_iso": {
                "type": "string",
                "pattern": "^[A-Z]{2}$"
            },
            "importer_country": {
                "type": "string",
                "enum": ["DE", "FR", "IT", "ES", "PL", "NL", "BE", "SE", "AT", "DK"]
            }
        }
    }

    schema_path = test_data_dir / "shipment_schema.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f)

    return schema_path


@pytest.fixture
def validation_rules(test_data_dir):
    """Create validation rules YAML."""
    rules = {
        "rules": [
            {
                "id": "VAL-001",
                "name": "CN Code Format",
                "description": "CN code must be 8 digits",
                "field": "cn_code",
                "type": "regex",
                "pattern": "^[0-9]{8}$",
                "severity": "error"
            },
            {
                "id": "VAL-002",
                "name": "Positive Mass",
                "description": "Mass must be positive",
                "field": "net_mass_kg",
                "type": "range",
                "min": 0.001,
                "severity": "error"
            },
            {
                "id": "VAL-003",
                "name": "EU Importer",
                "description": "Importer must be in EU",
                "field": "importer_country",
                "type": "enum",
                "allowed": ["DE", "FR", "IT", "ES", "PL", "NL", "BE", "SE", "AT", "DK"],
                "severity": "error"
            },
            {
                "id": "VAL-004",
                "name": "Future Date Warning",
                "description": "Import date should not be in future",
                "field": "import_date",
                "type": "date_range",
                "max": "today",
                "severity": "warning"
            }
        ]
    }

    rules_path = test_data_dir / "validation_rules.yaml"
    with open(rules_path, 'w') as f:
        yaml.dump(rules, f)

    return rules_path


@pytest.fixture
def valid_shipment():
    """Create valid shipment data."""
    return {
        "shipment_id": "SHIP001",
        "import_date": "2024-01-15",
        "quarter": "Q1-2024",
        "cn_code": "72071100",
        "net_mass_kg": 10000.0,
        "origin_iso": "CN",
        "importer_country": "DE"
    }


@pytest.fixture
def invalid_shipment():
    """Create invalid shipment data."""
    return {
        "shipment_id": "",  # Empty - invalid
        "import_date": "2024-13-45",  # Invalid date
        "quarter": "Q5-2024",  # Invalid quarter
        "cn_code": "1234",  # Too short
        "net_mass_kg": -100,  # Negative - invalid
        "origin_iso": "USA",  # Wrong format
        "importer_country": "US"  # Not EU
    }


# ============================================================================
# TEST VALIDATION FRAMEWORK
# ============================================================================

class TestValidationFramework:
    """Test ValidationFramework core functionality."""

    def test_framework_initialization(self, shipment_schema, validation_rules):
        """Test framework initializes with schema and rules."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        assert validator is not None
        assert validator.schema is not None
        assert validator.rules is not None

    def test_validate_valid_data(self, shipment_schema, validation_rules, valid_shipment):
        """Test validating valid data passes."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        result = validator.validate(valid_shipment)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_data(self, shipment_schema, validation_rules, invalid_shipment):
        """Test validating invalid data fails."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        result = validator.validate(invalid_shipment)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validation_exception_raising(self, shipment_schema, validation_rules, invalid_shipment):
        """Test ValidationException is raised for invalid data."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules),
            strict=True  # Raise on errors
        )

        with pytest.raises(ValidationException):
            validator.validate(invalid_shipment)


# ============================================================================
# TEST SCHEMA VALIDATION
# ============================================================================

class TestSchemaValidation:
    """Test JSON Schema validation."""

    def test_schema_required_fields(self, shipment_schema):
        """Test schema validates required fields."""
        validator = SchemaValidator(schema_path=str(shipment_schema))

        # Missing required field
        incomplete = {
            "shipment_id": "SHIP001",
            # Missing: import_date, cn_code, net_mass_kg
        }

        result = validator.validate(incomplete)

        assert result.is_valid is False
        assert len(result.errors) >= 3  # At least 3 missing fields

    def test_schema_type_validation(self, shipment_schema):
        """Test schema validates field types."""
        validator = SchemaValidator(schema_path=str(shipment_schema))

        # Wrong types
        wrong_types = {
            "shipment_id": "SHIP001",
            "import_date": "2024-01-15",
            "cn_code": "72071100",
            "net_mass_kg": "not a number"  # Should be number
        }

        result = validator.validate(wrong_types)

        assert result.is_valid is False

    def test_schema_pattern_validation(self, shipment_schema):
        """Test schema validates patterns."""
        validator = SchemaValidator(schema_path=str(shipment_schema))

        # Invalid patterns
        invalid_patterns = {
            "shipment_id": "SHIP001",
            "import_date": "2024-01-15",
            "cn_code": "ABC12345",  # Should be 8 digits
            "net_mass_kg": 1000,
            "quarter": "Quarter 1"  # Wrong pattern
        }

        result = validator.validate(invalid_patterns)

        assert result.is_valid is False

    def test_schema_range_validation(self, shipment_schema):
        """Test schema validates numeric ranges."""
        validator = SchemaValidator(schema_path=str(shipment_schema))

        # Negative mass
        negative_mass = {
            "shipment_id": "SHIP001",
            "import_date": "2024-01-15",
            "cn_code": "72071100",
            "net_mass_kg": -100  # Invalid
        }

        result = validator.validate(negative_mass)

        assert result.is_valid is False


# ============================================================================
# TEST RULES ENGINE
# ============================================================================

class TestRulesEngine:
    """Test business rules engine."""

    def test_rules_engine_initialization(self, validation_rules):
        """Test rules engine initializes."""
        engine = RulesEngine(rules_path=str(validation_rules))

        assert engine is not None
        assert len(engine.rules) >= 4

    def test_regex_rule(self, validation_rules):
        """Test regex rule validation."""
        engine = RulesEngine(rules_path=str(validation_rules))

        # Valid CN code
        valid_data = {"cn_code": "72071100"}
        result = engine.apply_rules(valid_data)
        assert all(not issue.is_error for issue in result if issue.field == "cn_code")

        # Invalid CN code
        invalid_data = {"cn_code": "1234"}
        result = engine.apply_rules(invalid_data)
        assert any(issue.is_error and issue.field == "cn_code" for issue in result)

    def test_range_rule(self, validation_rules):
        """Test range rule validation."""
        engine = RulesEngine(rules_path=str(validation_rules))

        # Valid mass
        valid_data = {"net_mass_kg": 1000}
        result = engine.apply_rules(valid_data)
        assert all(not issue.is_error for issue in result if issue.field == "net_mass_kg")

        # Invalid mass
        invalid_data = {"net_mass_kg": -100}
        result = engine.apply_rules(invalid_data)
        assert any(issue.is_error and issue.field == "net_mass_kg" for issue in result)

    def test_enum_rule(self, validation_rules):
        """Test enum rule validation."""
        engine = RulesEngine(rules_path=str(validation_rules))

        # Valid EU country
        valid_data = {"importer_country": "DE"}
        result = engine.apply_rules(valid_data)
        assert all(not issue.is_error for issue in result if issue.field == "importer_country")

        # Invalid non-EU country
        invalid_data = {"importer_country": "US"}
        result = engine.apply_rules(invalid_data)
        assert any(issue.is_error and issue.field == "importer_country" for issue in result)

    def test_warning_vs_error(self, validation_rules):
        """Test distinction between warnings and errors."""
        engine = RulesEngine(rules_path=str(validation_rules))

        data = {
            "cn_code": "72071100",  # Valid
            "net_mass_kg": 1000,  # Valid
            "importer_country": "DE",  # Valid
            "import_date": "2099-01-01"  # Future date (warning)
        }

        result = engine.apply_rules(data)

        # Should have warnings but no errors
        warnings = [issue for issue in result if issue.severity == "warning"]
        errors = [issue for issue in result if issue.is_error]

        assert len(warnings) > 0
        assert len(errors) == 0


# ============================================================================
# TEST CUSTOM VALIDATORS
# ============================================================================

class TestCustomValidators:
    """Test custom validator functionality."""

    def test_create_custom_validator(self):
        """Test creating custom validator function."""

        def validate_cbam_cn_code(value):
            """Custom validator for CBAM CN codes."""
            if not isinstance(value, str):
                raise ValidationException("CN code must be string")

            if len(value) != 8:
                raise ValidationException("CN code must be 8 characters")

            if not value.isdigit():
                raise ValidationException("CN code must be numeric")

            # Check CBAM coverage (mock)
            cbam_codes = ["72071100", "72072000", "28112100"]
            if value not in cbam_codes:
                raise ValidationException(f"CN code {value} not CBAM-covered")

        # Valid
        validate_cbam_cn_code("72071100")  # Should not raise

        # Invalid
        with pytest.raises(ValidationException):
            validate_cbam_cn_code("99999999")

    def test_register_custom_validator(self, shipment_schema, validation_rules):
        """Test registering custom validator with framework."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        def custom_quarter_validator(value):
            """Validate quarter format and date consistency."""
            if not value:
                raise ValidationException("Quarter required")

            # Extract quarter and year
            import re
            match = re.match(r'^Q([1-4])-20([0-9]{2})$', value)
            if not match:
                raise ValidationException("Invalid quarter format")

        # Register custom validator
        validator.register_custom_validator("quarter", custom_quarter_validator)

        # Test with valid quarter
        valid_data = {
            "shipment_id": "SHIP001",
            "import_date": "2024-01-15",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "net_mass_kg": 1000
        }

        result = validator.validate(valid_data)
        # Should pass if custom validator works


# ============================================================================
# TEST VALIDATION ISSUES
# ============================================================================

class TestValidationIssues:
    """Test ValidationIssue model."""

    def test_create_validation_issue(self):
        """Test creating validation issue."""
        issue = ValidationIssue(
            field="cn_code",
            message="Invalid CN code format",
            severity="error",
            rule_id="VAL-001"
        )

        assert issue.field == "cn_code"
        assert issue.is_error is True

    def test_validation_issue_severity(self):
        """Test validation issue severity levels."""
        error = ValidationIssue(field="field1", message="Error", severity="error")
        warning = ValidationIssue(field="field2", message="Warning", severity="warning")
        info = ValidationIssue(field="field3", message="Info", severity="info")

        assert error.is_error is True
        assert warning.is_error is False
        assert info.is_error is False

    def test_validation_result_summary(self):
        """Test validation result summary."""
        issues = [
            ValidationIssue(field="field1", message="Error 1", severity="error"),
            ValidationIssue(field="field2", message="Error 2", severity="error"),
            ValidationIssue(field="field3", message="Warning 1", severity="warning")
        ]

        errors = [i for i in issues if i.is_error]
        warnings = [i for i in issues if i.severity == "warning"]

        assert len(errors) == 2
        assert len(warnings) == 1


# ============================================================================
# TEST BATCH VALIDATION
# ============================================================================

class TestBatchValidation:
    """Test batch validation functionality."""

    def test_validate_batch(self, shipment_schema, validation_rules, valid_shipment):
        """Test validating batch of records."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        # Create batch
        batch = [valid_shipment.copy() for _ in range(10)]
        batch[3]['cn_code'] = "1234"  # Make one invalid

        results = validator.validate_batch(batch)

        assert len(results) == 10
        assert results[3].is_valid is False  # Invalid record
        assert all(r.is_valid for i, r in enumerate(results) if i != 3)  # Rest valid

    def test_batch_validation_statistics(self, shipment_schema, validation_rules, valid_shipment, invalid_shipment):
        """Test batch validation statistics."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        # Mixed batch
        batch = [valid_shipment] * 7 + [invalid_shipment] * 3

        results = validator.validate_batch(batch)

        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = sum(1 for r in results if not r.is_valid)

        assert valid_count == 7
        assert invalid_count == 3


# ============================================================================
# TEST ERROR REPORTING
# ============================================================================

class TestErrorReporting:
    """Test validation error reporting."""

    def test_error_message_clarity(self, shipment_schema, validation_rules, invalid_shipment):
        """Test error messages are clear and actionable."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        result = validator.validate(invalid_shipment)

        # Check error messages are descriptive
        for error in result.errors:
            assert len(error.message) > 0
            assert error.field is not None

    def test_generate_validation_report(self, shipment_schema, validation_rules, invalid_shipment):
        """Test generating validation report."""
        validator = ValidationFramework(
            schema=str(shipment_schema),
            rules=str(validation_rules)
        )

        result = validator.validate(invalid_shipment)

        # Generate report
        report = result.to_report()

        assert "errors" in report.lower() or "validation" in report.lower()
        assert len(report) > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
