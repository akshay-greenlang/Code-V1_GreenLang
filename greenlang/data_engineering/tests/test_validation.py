"""
Test Validation Rules Engine
============================

Unit tests for the validation rules engine.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

import pytest
from datetime import date
from decimal import Decimal

from greenlang.data_engineering.validation.rules_engine import (
    ValidationRulesEngine,
    ValidationRule,
    RuleType,
    RuleSeverity,
)


class TestValidationRulesEngine:
    """Test suite for ValidationRulesEngine."""

    @pytest.fixture
    def engine(self):
        """Create validation engine with default rules."""
        return ValidationRulesEngine()

    @pytest.fixture
    def valid_emission_factor(self):
        """Create a valid emission factor record."""
        return {
            "factor_id": "ef-001",
            "factor_hash": "abc123def456",
            "factor_value": 0.5,
            "factor_unit": "kgCO2e/kWh",
            "industry": "electricity",
            "region": "europe",
            "scope_type": "scope_2_location",
            "ghg_type": "CO2e",
            "reference_year": 2024,
            "valid_from": "2024-01-01",
        }

    def test_validate_valid_record(self, engine, valid_emission_factor):
        """Test validation passes for valid record."""
        violations = engine.validate_record(valid_emission_factor)
        errors = [v for v in violations if v.severity == RuleSeverity.ERROR]
        assert len(errors) == 0

    def test_validate_missing_required_field(self, engine, valid_emission_factor):
        """Test validation fails for missing required field."""
        del valid_emission_factor["factor_id"]
        violations = engine.validate_record(valid_emission_factor)
        assert any(v.rule_id == "EF001" for v in violations)

    def test_validate_negative_factor_value(self, engine, valid_emission_factor):
        """Test validation fails for negative factor value."""
        valid_emission_factor["factor_value"] = -0.5
        violations = engine.validate_record(valid_emission_factor)
        assert any("factor_value" in (v.field or "") for v in violations)

    def test_validate_invalid_reference_year(self, engine, valid_emission_factor):
        """Test validation fails for out-of-range reference year."""
        valid_emission_factor["reference_year"] = 1980
        violations = engine.validate_record(valid_emission_factor)
        assert any("reference_year" in (v.field or "") for v in violations)

    def test_validate_invalid_ghg_type(self, engine, valid_emission_factor):
        """Test validation fails for invalid GHG type."""
        valid_emission_factor["ghg_type"] = "INVALID"
        violations = engine.validate_record(valid_emission_factor)
        assert any("ghg_type" in (v.field or "") for v in violations)

    def test_validate_batch(self, engine, valid_emission_factor):
        """Test batch validation."""
        records = [valid_emission_factor.copy() for _ in range(10)]
        # Make one invalid
        records[5]["factor_value"] = -1

        result = engine.validate_batch(records)

        assert result.total_records == 10
        assert result.valid_records == 9
        assert result.invalid_records == 1

    def test_uniqueness_validation(self, engine, valid_emission_factor):
        """Test uniqueness validation for factor_hash."""
        records = [valid_emission_factor.copy() for _ in range(3)]
        # All have same hash - duplicates
        result = engine.validate_batch(records)

        # Should have uniqueness violations
        unique_violations = [v for v in result.violations if "Duplicate" in v.get("message", "")]
        assert len(unique_violations) == 2  # First is ok, next 2 are duplicates

    def test_custom_rule(self, engine, valid_emission_factor):
        """Test adding custom validation rule."""
        # Add custom rule
        custom_rule = ValidationRule(
            rule_id="CUSTOM001",
            rule_type=RuleType.CUSTOM,
            field="factor_value",
            severity=RuleSeverity.WARNING,
            message="Factor value should be less than 10",
            parameters={"validator": "check_max_factor"},
        )
        engine.add_rule(custom_rule)
        engine.register_custom_validator(
            "check_max_factor",
            lambda v, r: v is None or float(v) < 10
        )

        valid_emission_factor["factor_value"] = 15
        violations = engine.validate_record(valid_emission_factor)

        assert any(v.rule_id == "CUSTOM001" for v in violations)

    def test_quality_score_calculation(self, engine, valid_emission_factor):
        """Test quality score in validation result."""
        records = [valid_emission_factor.copy() for _ in range(100)]
        result = engine.validate_batch(records)

        assert result.quality_score > 0
        assert result.quality_score <= 100


class TestValidationRules:
    """Test individual validation rules."""

    def test_cn_code_rule(self):
        """Test CN code format validation."""
        engine = ValidationRulesEngine()
        engine.add_rule(ValidationRule(
            rule_id="CN001",
            rule_type=RuleType.CN_CODE,
            field="product_code",
            severity=RuleSeverity.ERROR,
            message="Invalid CN code",
        ))

        # Valid CN code
        valid_record = {"product_code": "72061000", "factor_id": "test"}
        violations = engine.validate_record(valid_record)
        cn_violations = [v for v in violations if v.rule_id == "CN001"]
        assert len(cn_violations) == 0

        # Invalid CN code
        invalid_record = {"product_code": "7206", "factor_id": "test"}
        violations = engine.validate_record(invalid_record)
        cn_violations = [v for v in violations if v.rule_id == "CN001"]
        assert len(cn_violations) == 1

    def test_iso_country_code_rule(self):
        """Test ISO country code validation."""
        engine = ValidationRulesEngine()
        engine.add_rule(ValidationRule(
            rule_id="ISO001",
            rule_type=RuleType.ISO_COUNTRY,
            field="country_code",
            severity=RuleSeverity.ERROR,
            message="Invalid country code",
        ))

        # Valid country code
        valid_record = {"country_code": "US", "factor_id": "test"}
        violations = engine.validate_record(valid_record)
        iso_violations = [v for v in violations if v.rule_id == "ISO001"]
        assert len(iso_violations) == 0

        # Invalid country code
        invalid_record = {"country_code": "XX", "factor_id": "test"}
        violations = engine.validate_record(invalid_record)
        iso_violations = [v for v in violations if v.rule_id == "ISO001"]
        assert len(iso_violations) == 1

    def test_date_sequence_rule(self):
        """Test date sequence validation."""
        engine = ValidationRulesEngine()
        engine.add_rule(ValidationRule(
            rule_id="DATE001",
            rule_type=RuleType.DATE_SEQUENCE,
            severity=RuleSeverity.ERROR,
            message="valid_from must be before valid_to",
            parameters={
                "start_field": "valid_from",
                "end_field": "valid_to",
            },
        ))

        # Valid sequence
        valid_record = {
            "factor_id": "test",
            "valid_from": "2024-01-01",
            "valid_to": "2024-12-31",
        }
        violations = engine.validate_record(valid_record)
        date_violations = [v for v in violations if v.rule_id == "DATE001"]
        assert len(date_violations) == 0

        # Invalid sequence
        invalid_record = {
            "factor_id": "test",
            "valid_from": "2024-12-31",
            "valid_to": "2024-01-01",
        }
        violations = engine.validate_record(invalid_record)
        date_violations = [v for v in violations if v.rule_id == "DATE001"]
        assert len(date_violations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
