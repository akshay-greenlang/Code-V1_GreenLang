# -*- coding: utf-8 -*-
"""
Test Suite for Zero-Hallucination Calculation Engine

Comprehensive tests to verify:
- Bit-perfect reproducibility
- Deterministic calculations
- Provenance tracking
- Regulatory compliance
- Error handling
"""

import pytest
from decimal import Decimal
from datetime import date

from formula_engine import FormulaEngine, FormulaLibrary, Formula, FormulaParameter, FormulaStep
from emission_factors import EmissionFactorDatabase, EmissionFactor
from calculation_engine import CalculationEngine, CalculationResult
from unit_converter import UnitConverter, UnitType
from validators import CalculationValidator, ValidationResult


class TestFormulaEngine:
    """Test formula engine functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = FormulaEngine()

    def test_basic_expression_evaluation(self):
        """Test basic mathematical expression evaluation."""
        result = self.engine.evaluate_expression(
            "a * b + c",
            {"a": 10, "b": 5, "c": 3},
            precision=2
        )
        assert result == Decimal("53.00")

    def test_division_expression(self):
        """Test division operation."""
        result = self.engine.evaluate_expression(
            "numerator / denominator",
            {"numerator": 100, "denominator": 4},
            precision=2
        )
        assert result == Decimal("25.00")

    def test_complex_expression(self):
        """Test complex expression with parentheses."""
        result = self.engine.evaluate_expression(
            "(a + b) * c",
            {"a": 10, "b": 5, "c": 2},
            precision=2
        )
        assert result == Decimal("30.00")

    def test_safe_evaluation_rejects_dangerous_code(self):
        """Test that unsafe operations are rejected."""
        with pytest.raises(ValueError):
            # Should reject function calls not in allowed list
            self.engine.evaluate_expression(
                "exec('malicious code')",
                {},
                precision=2
            )

    def test_precision_rounding(self):
        """Test precision rounding behavior."""
        result = self.engine.evaluate_expression(
            "a / b",
            {"a": 10, "b": 3},
            precision=3
        )
        assert result == Decimal("3.333")

    def test_hash_calculation_deterministic(self):
        """Test that hash calculation is deterministic."""
        data = {"key": "value", "number": 42}
        hash1 = self.engine.calculate_hash(data)
        hash2 = self.engine.calculate_hash(data)
        assert hash1 == hash2

    def test_hash_changes_with_data(self):
        """Test that hash changes when data changes."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        hash1 = self.engine.calculate_hash(data1)
        hash2 = self.engine.calculate_hash(data2)
        assert hash1 != hash2


class TestEmissionFactorDatabase:
    """Test emission factor database functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.db = EmissionFactorDatabase()

    def test_insert_and_retrieve_factor(self):
        """Test inserting and retrieving emission factor."""
        factor = EmissionFactor(
            factor_id="test_diesel_2024",
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            unit="kg_co2e_per_liter",
            factor_co2=Decimal("2.68"),
            factor_ch4=Decimal("0.0001"),
            factor_n2o=Decimal("0.0001"),
            factor_co2e=Decimal("2.69"),
            region="GB",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="DEFRA",
            source_year=2024,
            source_version="2024",
            data_quality="high"
        )

        self.db.insert_factor(factor)

        retrieved = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            region="GB",
            reference_date=date(2024, 6, 1)
        )

        assert retrieved is not None
        assert retrieved.factor_co2e == Decimal("2.69")
        assert retrieved.source == "DEFRA"

    def test_temporal_validity(self):
        """Test temporal validity of emission factors."""
        # Insert factor valid only in 2024
        factor = EmissionFactor(
            factor_id="test_temp_2024",
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="natural_gas",
            unit="kg_co2e_per_m3",
            factor_co2=Decimal("2.0"),
            factor_co2e=Decimal("2.0"),
            region="GB",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="TEST",
            source_year=2024,
            source_version="1.0",
            data_quality="high"
        )

        self.db.insert_factor(factor)

        # Should find for 2024 date
        found_2024 = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="natural_gas",
            region="GB",
            reference_date=date(2024, 6, 1)
        )
        assert found_2024 is not None

        # Should not find for 2025 date
        found_2025 = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="natural_gas",
            region="GB",
            reference_date=date(2025, 6, 1)
        )
        assert found_2025 is None

    def test_regional_fallback(self):
        """Test fallback to GLOBAL when regional factor not found."""
        # Insert only global factor
        global_factor = EmissionFactor(
            factor_id="test_global_coal",
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="coal",
            unit="kg_co2e_per_kg",
            factor_co2=Decimal("2.5"),
            factor_co2e=Decimal("2.5"),
            region="GLOBAL",
            valid_from=date(2024, 1, 1),
            source="IPCC",
            source_year=2024,
            source_version="1.0",
            data_quality="medium"
        )

        self.db.insert_factor(global_factor)

        # Search for specific region (should fallback to GLOBAL)
        found = self.db.get_factor(
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="coal",
            region="FR",  # France-specific not available
            reference_date=date(2024, 6, 1)
        )

        assert found is not None
        assert found.region == "GLOBAL"

    def teardown_method(self):
        """Cleanup test database."""
        self.db.close()


class TestUnitConverter:
    """Test unit converter functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.converter = UnitConverter()

    def test_energy_conversion(self):
        """Test energy unit conversions."""
        # kWh to MWh
        result = self.converter.convert(1000, "kWh", "MWh", precision=2)
        assert result == Decimal("1.00")

        # MWh to kWh
        result = self.converter.convert(1, "MWh", "kWh", precision=0)
        assert result == Decimal("1000")

    def test_mass_conversion(self):
        """Test mass unit conversions."""
        # kg to tonnes
        result = self.converter.convert(2500, "kg", "t", precision=2)
        assert result == Decimal("2.50")

        # tonnes to kg
        result = self.converter.convert(1, "t", "kg", precision=0)
        assert result == Decimal("1000")

    def test_emissions_conversion(self):
        """Test emissions unit conversions."""
        # kg CO2e to tonnes CO2e
        result = self.converter.convert(2690, "kg_co2e", "t_co2e", precision=3)
        assert result == Decimal("2.690")

    def test_incompatible_units_error(self):
        """Test that converting incompatible units raises error."""
        with pytest.raises(ValueError, match="Incompatible unit types"):
            self.converter.convert(100, "kg", "kWh")

    def test_unknown_unit_error(self):
        """Test that unknown unit raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            self.converter.convert(100, "xyz", "abc")

    def test_conversion_factor(self):
        """Test getting conversion factor."""
        factor = self.converter.get_conversion_factor("kg", "t")
        assert factor == Decimal("0.001")


class TestCalculationEngine:
    """Test calculation engine functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.formula_library = FormulaLibrary()
        self.emission_db = EmissionFactorDatabase()
        self.engine = CalculationEngine(self.formula_library, self.emission_db)

        # Insert test emission factor
        test_factor = EmissionFactor(
            factor_id="test_diesel_gb_2024",
            category="scope1",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            unit="kg_co2e_per_liter",
            factor_co2=Decimal("2.68"),
            factor_co2e=Decimal("2.69"),
            region="GB",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="DEFRA",
            source_year=2024,
            source_version="2024",
            data_quality="high",
            uncertainty_percentage=5.0
        )
        self.emission_db.insert_factor(test_factor)

    def test_reproducibility(self):
        """Test that calculations are bit-perfect reproducible."""
        # This would require a loaded formula - skipped if none available
        formula_count = self.formula_library.load_formulas()

        if formula_count > 0:
            # Run same calculation twice
            params = {
                "fuel_quantity": 1000,
                "fuel_type": "diesel",
                "region": "GB"
            }

            result1 = self.engine.calculate("scope1_stationary_combustion", params)
            result2 = self.engine.calculate("scope1_stationary_combustion", params)

            # Results must be identical
            assert result1.output_value == result2.output_value
            assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_tracking(self):
        """Test that provenance hash is generated."""
        formula_count = self.formula_library.load_formulas()

        if formula_count > 0:
            params = {
                "fuel_quantity": 1000,
                "fuel_type": "diesel",
                "region": "GB"
            }

            result = self.engine.calculate("scope1_stationary_combustion", params)

            assert result.provenance_hash is not None
            assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_calculation_steps_recorded(self):
        """Test that all calculation steps are recorded."""
        formula_count = self.formula_library.load_formulas()

        if formula_count > 0:
            params = {
                "fuel_quantity": 1000,
                "fuel_type": "diesel",
                "region": "GB"
            }

            result = self.engine.calculate("scope1_stationary_combustion", params)

            assert len(result.calculation_steps) > 0
            # Each step should have required fields
            for step in result.calculation_steps:
                assert step.step_number > 0
                assert step.description
                assert step.operation
                assert step.output_name

    def teardown_method(self):
        """Cleanup."""
        self.emission_db.close()


class TestCalculationValidator:
    """Test calculation validator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = CalculationValidator()

    def test_validate_positive_emissions(self):
        """Test validation catches negative emissions."""
        from calculation_engine import CalculationResult

        result = CalculationResult(
            formula_id="test_formula",
            formula_version="1.0",
            output_value=Decimal("-100.0"),  # Negative (invalid)
            output_unit="kg_co2e",
            calculation_steps=[],
            provenance_hash="test_hash",
            calculation_time_ms=5.0,
            input_parameters={}
        )

        validation = self.validator.validate_result(result)
        assert not validation.is_valid
        assert any("negative" in msg.message.lower() for msg in validation.errors)

    def test_validate_missing_provenance(self):
        """Test validation catches missing provenance."""
        from calculation_engine import CalculationResult

        result = CalculationResult(
            formula_id="test_formula",
            formula_version="1.0",
            output_value=Decimal("100.0"),
            output_unit="kg_co2e",
            calculation_steps=[],
            provenance_hash="",  # Missing
            calculation_time_ms=5.0,
            input_parameters={}
        )

        validation = self.validator.validate_result(result)
        assert not validation.is_valid
        assert any("provenance" in msg.message.lower() for msg in validation.errors)

    def test_validate_high_uncertainty_warning(self):
        """Test validation warns on high uncertainty."""
        from calculation_engine import CalculationResult

        result = CalculationResult(
            formula_id="test_formula",
            formula_version="1.0",
            output_value=Decimal("100.0"),
            output_unit="kg_co2e",
            calculation_steps=[],
            provenance_hash="valid_hash",
            calculation_time_ms=5.0,
            input_parameters={},
            uncertainty_percentage=15.0  # >10%
        )

        validation = self.validator.validate_result(result)
        assert any("uncertainty" in msg.message.lower() for msg in validation.warnings)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
