# -*- coding: utf-8 -*-
"""
Unit tests for hallucination detection

Tests HallucinationDetector functionality including:
- extract_numeric_claims() finds all numbers
- normalize_value() converts units correctly
- fuzzy_match() handles rounding (±1%)
- verify_citation() accepts matching values
- verify_citation() rejects hallucinations
- verify_response() end-to-end validation
- Multiple claims with multiple tools
- Scientific notation handling
"""

import pytest
from greenlang.intelligence.verification import (
    HallucinationDetector,
    HallucinationDetected,
    NumericClaim,
    Citation,
)


class TestExtractNumericClaims:
    """Test extract_numeric_claims() method"""

    def test_extracts_simple_number(self):
        """Should extract simple numbers"""
        detector = HallucinationDetector()
        text = "The value is 450"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 450

    def test_extracts_number_with_unit(self):
        """Should extract numbers with units"""
        detector = HallucinationDetector()
        text = "Grid intensity is 450 gCO2/kWh"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 450
        assert claims[0].unit == "gCO2/kWh"

    def test_extracts_multiple_numbers(self):
        """Should extract multiple numbers"""
        detector = HallucinationDetector()
        text = "Grid is 450 gCO2/kWh, emissions are 1021 kg"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 2
        assert claims[0].value == 450
        assert claims[0].unit == "gCO2/kWh"
        assert claims[1].value == 1021
        assert claims[1].unit == "kg"

    def test_extracts_decimal_numbers(self):
        """Should extract decimal numbers"""
        detector = HallucinationDetector()
        text = "The result is 42.7 MWh"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 42.7
        assert claims[0].unit == "MWh"

    def test_extracts_scientific_notation(self):
        """Should extract scientific notation"""
        detector = HallucinationDetector()
        text = "The value is 1.5e3 kg"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 1500.0
        assert claims[0].unit == "kg"

    def test_extracts_negative_numbers(self):
        """Should extract negative numbers"""
        detector = HallucinationDetector()
        text = "Temperature is -15 C"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == -15

    def test_extracts_comma_separated_thousands(self):
        """Should extract numbers with comma separators"""
        detector = HallucinationDetector()
        text = "Total is 1,234,567 kg"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 1234567

    def test_extracts_citation_tool(self):
        """Should extract citation tool from [tool:name] format"""
        detector = HallucinationDetector()
        text = "Grid is 450 gCO2/kWh [tool:get_grid_intensity]"

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 450
        assert claims[0].citation_tool == "get_grid_intensity"

    def test_extracts_citation_various_brackets(self):
        """Should extract citations with various bracket types"""
        detector = HallucinationDetector()

        tests = [
            ("Value is 100 [tool:calc]", "calc"),
            ("Value is 100 (tool:calc)", "calc"),
            ("Value is 100 {tool:calc}", "calc"),
            ("Value is 100 <tool:calc>", "calc"),
        ]

        for text, expected_tool in tests:
            claims = detector.extract_numeric_claims(text)
            assert claims[0].citation_tool == expected_tool

    def test_ignores_numbers_in_chemical_formulas(self):
        """Should not extract numbers from chemical formulas like CO2"""
        detector = HallucinationDetector()
        text = "CO2 emissions are 450 kg"

        claims = detector.extract_numeric_claims(text)

        # Should only extract 450, not the 2 from CO2
        assert len(claims) == 1
        assert claims[0].value == 450

    def test_includes_context(self):
        """Should include surrounding context"""
        detector = HallucinationDetector()
        text = "Based on the calculation, the grid intensity is 450 gCO2/kWh which is typical."

        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert "grid intensity" in claims[0].context.lower()


class TestNormalizeValue:
    """Test normalize_value() method"""

    def test_normalizes_grams_to_kilograms(self):
        """Should normalize grams to kilograms"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(1500, "g")

        assert value == 1.5
        assert unit == "kg"

    def test_normalizes_mwh_to_kwh(self):
        """Should normalize MWh to kWh"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(2.5, "MWh")

        assert value == 2500
        assert unit == "kwh"

    def test_normalizes_case_insensitive(self):
        """Should normalize case-insensitively"""
        detector = HallucinationDetector()

        value1, unit1 = detector.normalize_value(1000, "KG")
        value2, unit2 = detector.normalize_value(1000, "kg")

        assert value1 == value2
        assert unit1 == unit2

    def test_handles_unknown_units(self):
        """Should return as-is for unknown units"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(100, "widgets")

        assert value == 100
        assert unit == "widgets"

    def test_normalizes_gco2_kwh(self):
        """Should normalize grid intensity units"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(450, "gCO2/kWh")

        assert value == 450
        assert unit == "gco2/kwh"

    def test_normalizes_kgco2_kwh_to_gco2_kwh(self):
        """Should normalize kgCO2/kWh to gCO2/kWh"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(0.45, "kgCO2/kWh")

        assert value == 450
        assert unit == "gco2/kwh"

    def test_normalizes_tonnes_to_kg(self):
        """Should normalize tonnes to kg"""
        detector = HallucinationDetector()

        value, unit = detector.normalize_value(1.5, "tonne")

        assert value == 1500
        assert unit == "kg"

    def test_custom_normalizations(self):
        """Should support custom unit normalizations"""
        detector = HallucinationDetector(
            unit_normalizations={"custom": ("base", 10.0)}
        )

        value, unit = detector.normalize_value(5, "custom")

        assert value == 50
        assert unit == "base"


class TestFuzzyMatch:
    """Test fuzzy_match() method"""

    def test_exact_match_passes(self):
        """Should pass on exact match"""
        detector = HallucinationDetector(tolerance=0.01)

        assert detector.fuzzy_match(450, 450)

    def test_within_tolerance_passes(self):
        """Should pass when within tolerance"""
        detector = HallucinationDetector(tolerance=0.01)

        # 450.3 is within 1% of 450
        assert detector.fuzzy_match(450, 450.3)

    def test_outside_tolerance_fails(self):
        """Should fail when outside tolerance"""
        detector = HallucinationDetector(tolerance=0.01)

        # 460 is outside 1% of 450
        assert not detector.fuzzy_match(450, 460)

    def test_handles_zero_actual(self):
        """Should handle zero actual value"""
        detector = HallucinationDetector(tolerance=0.01)

        assert detector.fuzzy_match(0, 0)
        assert detector.fuzzy_match(0.005, 0)  # Within tolerance
        assert not detector.fuzzy_match(1.0, 0)  # Outside tolerance

    def test_custom_tolerance(self):
        """Should accept custom tolerance"""
        detector = HallucinationDetector(tolerance=0.01)

        # Within 5% but not 1%
        assert not detector.fuzzy_match(450, 465, tolerance=0.01)
        assert detector.fuzzy_match(450, 465, tolerance=0.05)

    def test_both_zero_passes(self):
        """Should pass when both values are zero"""
        detector = HallucinationDetector()

        assert detector.fuzzy_match(0, 0)

    def test_negative_numbers(self):
        """Should handle negative numbers"""
        detector = HallucinationDetector(tolerance=0.01)

        assert detector.fuzzy_match(-450, -450.3)
        assert not detector.fuzzy_match(-450, -460)


class TestVerifyCitation:
    """Test verify_citation() method"""

    def test_accepts_matching_value(self):
        """Should accept citation when value matches"""
        detector = HallucinationDetector(tolerance=0.01)

        claim = NumericClaim(
            value=450,
            unit="gCO2/kWh",
            citation_tool="get_grid_intensity"
        )

        tool_response = {
            "result": {
                "intensity": 450.3,
                "unit": "gCO2/kWh"
            }
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is not None
        assert citation.tool == "get_grid_intensity"
        assert citation.value == 450

    def test_rejects_non_matching_value(self):
        """Should reject citation when value doesn't match"""
        detector = HallucinationDetector(tolerance=0.01)

        claim = NumericClaim(
            value=999,
            unit="gCO2/kWh",
            citation_tool="get_grid_intensity"
        )

        tool_response = {
            "result": {
                "intensity": 450,
                "unit": "gCO2/kWh"
            }
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is None

    def test_handles_nested_response(self):
        """Should find value in nested response"""
        detector = HallucinationDetector(tolerance=0.01)

        claim = NumericClaim(
            value=450,
            unit="gCO2/kWh",
            citation_tool="calc"
        )

        tool_response = {
            "data": {
                "grid": {
                    "intensity": 450.2
                }
            }
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is not None

    def test_normalizes_units_before_comparison(self):
        """Should normalize units before comparison"""
        detector = HallucinationDetector(tolerance=0.01)

        # Claim in kg, response in g
        claim = NumericClaim(
            value=1.5,
            unit="kg",
            citation_tool="calc"
        )

        tool_response = {
            "result": {
                "value": 1500,
                "unit": "g"
            }
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is not None

    def test_finds_direct_numeric_value(self):
        """Should find direct numeric values (not in value/unit structure)"""
        detector = HallucinationDetector(tolerance=0.01)

        claim = NumericClaim(
            value=450,
            unit="",
            citation_tool="calc"
        )

        tool_response = {
            "result": 450.2
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is not None

    def test_citation_includes_source_path(self):
        """Should include JSON path in citation"""
        detector = HallucinationDetector(tolerance=0.01)

        claim = NumericClaim(
            value=450,
            unit="gCO2/kWh",
            citation_tool="calc"
        )

        tool_response = {
            "result": {
                "intensity": 450
            }
        }

        citation = detector.verify_citation(claim, tool_response)

        assert citation is not None
        assert citation.source  # Should have source path


class TestVerifyResponse:
    """Test verify_response() end-to-end validation"""

    def test_accepts_valid_response(self):
        """Should accept response with valid citations"""
        detector = HallucinationDetector(tolerance=0.01)

        response_text = "Grid is 450 gCO2/kWh [tool:grid]"
        tool_calls = [{"name": "grid", "arguments": {}}]
        tool_responses = [{"result": {"intensity": 450.3, "unit": "gCO2/kWh"}}]

        citations = detector.verify_response(response_text, tool_calls, tool_responses)

        assert len(citations) == 1
        assert citations[0].value == 450

    def test_raises_on_missing_citation(self):
        """Should raise when claim has no citation"""
        detector = HallucinationDetector(require_citations=True)

        response_text = "Grid is 450 gCO2/kWh"  # No [tool:name]
        tool_calls = []
        tool_responses = []

        with pytest.raises(HallucinationDetected, match="without tool citation"):
            detector.verify_response(response_text, tool_calls, tool_responses)

    def test_raises_on_non_existent_tool(self):
        """Should raise when cited tool doesn't exist"""
        detector = HallucinationDetector()

        response_text = "Grid is 450 gCO2/kWh [tool:nonexistent]"
        tool_calls = [{"name": "actual_tool", "arguments": {}}]
        tool_responses = [{"result": 450}]

        with pytest.raises(HallucinationDetected, match="non-existent tool"):
            detector.verify_response(response_text, tool_calls, tool_responses)

    def test_raises_on_value_mismatch(self):
        """Should raise when claimed value doesn't match tool output"""
        detector = HallucinationDetector(tolerance=0.01)

        response_text = "Grid is 999 gCO2/kWh [tool:grid]"
        tool_calls = [{"name": "grid", "arguments": {}}]
        tool_responses = [{"result": {"intensity": 450}}]

        with pytest.raises(HallucinationDetected, match="does not match tool output"):
            detector.verify_response(response_text, tool_calls, tool_responses)

    def test_multiple_claims_multiple_tools(self):
        """Should verify multiple claims with multiple tools"""
        detector = HallucinationDetector(tolerance=0.01)

        response_text = "Grid is 450 gCO2/kWh [tool:grid], emissions are 1,021 kg [tool:calc]"
        tool_calls = [
            {"name": "grid", "arguments": {"region": "CA"}},
            {"name": "calc", "arguments": {"kwh": 2000}}
        ]
        tool_responses = [
            {"result": {"intensity": 450.3, "unit": "gCO2/kWh"}},
            {"result": {"emissions": 1021.5, "unit": "kg"}}
        ]

        citations = detector.verify_response(response_text, tool_calls, tool_responses)

        assert len(citations) == 2
        assert citations[0].tool == "grid"
        assert citations[1].tool == "calc"

    def test_accepts_response_without_numbers(self):
        """Should accept response with no numeric claims"""
        detector = HallucinationDetector()

        response_text = "The calculation is in progress."
        tool_calls = []
        tool_responses = []

        citations = detector.verify_response(response_text, tool_calls, tool_responses)

        assert len(citations) == 0

    def test_optional_citations_mode(self):
        """Should allow uncited claims when require_citations=False"""
        detector = HallucinationDetector(require_citations=False)

        response_text = "The value is approximately 450"  # No citation
        tool_calls = []
        tool_responses = []

        # Should not raise
        citations = detector.verify_response(response_text, tool_calls, tool_responses)

        assert len(citations) == 0


class TestScientificNotation:
    """Test scientific notation handling"""

    def test_extracts_scientific_notation_positive_exponent(self):
        """Should extract scientific notation with positive exponent"""
        detector = HallucinationDetector()

        text = "The value is 1.5e3 kg"
        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 1500

    def test_extracts_scientific_notation_negative_exponent(self):
        """Should extract scientific notation with negative exponent"""
        detector = HallucinationDetector()

        text = "The value is 4.5e-2 kg"
        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 0.045

    def test_extracts_uppercase_e_notation(self):
        """Should extract uppercase E notation"""
        detector = HallucinationDetector()

        text = "The value is 1.5E3 kg"
        claims = detector.extract_numeric_claims(text)

        assert len(claims) == 1
        assert claims[0].value == 1500

    def test_verifies_scientific_notation_claims(self):
        """Should verify claims with scientific notation"""
        detector = HallucinationDetector(tolerance=0.01)

        response_text = "Result is 1.5e3 kg [tool:calc]"
        tool_calls = [{"name": "calc", "arguments": {}}]
        tool_responses = [{"result": 1500}]

        citations = detector.verify_response(response_text, tool_calls, tool_responses)

        assert len(citations) == 1
        assert citations[0].value == 1500


class TestHallucinationDetectorConfiguration:
    """Test HallucinationDetector configuration options"""

    def test_default_tolerance(self):
        """Should use default 1% tolerance"""
        detector = HallucinationDetector()

        assert detector.tolerance == 0.01

    def test_custom_tolerance(self):
        """Should accept custom tolerance"""
        detector = HallucinationDetector(tolerance=0.05)

        assert detector.tolerance == 0.05

    def test_require_citations_default(self):
        """Should require citations by default"""
        detector = HallucinationDetector()

        assert detector.require_citations is True

    def test_optional_citations(self):
        """Should allow optional citations"""
        detector = HallucinationDetector(require_citations=False)

        assert detector.require_citations is False

    def test_custom_unit_normalizations(self):
        """Should merge custom normalizations with defaults"""
        custom = {"custom_unit": ("base", 100.0)}
        detector = HallucinationDetector(unit_normalizations=custom)

        # Should have both custom and default normalizations
        assert "custom_unit" in detector.unit_normalizations
        assert "kg" in detector.unit_normalizations  # Default


class TestHallucinationDetectedException:
    """Test HallucinationDetected exception details"""

    def test_includes_claim_details(self):
        """Exception should include claim details"""
        claim = NumericClaim(value=999, unit="kg", citation_tool="calc")

        exc = HallucinationDetected(
            message="Test hallucination",
            claim=claim,
            tool_response={"result": 450},
            expected_citation="calc"
        )

        error_str = str(exc)
        assert "999" in error_str
        assert "kg" in error_str

    def test_includes_tool_response(self):
        """Exception should include tool response"""
        exc = HallucinationDetected(
            message="Mismatch",
            claim=None,
            tool_response={"result": 450},
            expected_citation="calc"
        )

        error_str = str(exc)
        assert "450" in error_str

    def test_includes_expected_citation(self):
        """Exception should include expected citation"""
        exc = HallucinationDetected(
            message="Missing citation",
            claim=None,
            tool_response=None,
            expected_citation="expected_tool"
        )

        error_str = str(exc)
        assert "expected_tool" in error_str
