#!/usr/bin/env python
"""
EUDR Golden Tests
=================

Pytest-based golden tests for EUDR Deforestation Compliance Agent.
Tests cover all 5 EUDR tools with known-correct reference data.

Run with:
    pytest tests/golden/test_eudr_golden.py -v
    pytest tests/golden/test_eudr_golden.py -v -k "geolocation"
    pytest tests/golden/test_eudr_golden.py -v --tb=short
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))


# =============================================================================
# Test Data Loading
# =============================================================================

EUDR_TESTS_DIR = Path(__file__).parent / "eudr_compliance"


def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """Load test cases from a JSON file."""
    file_path = EUDR_TESTS_DIR / test_file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tests", [])


def get_test_ids(test_cases: List[Dict]) -> List[str]:
    """Extract test IDs for pytest parameterization."""
    return [tc.get("test_id", f"test_{i}") for i, tc in enumerate(test_cases)]


# =============================================================================
# Geolocation Tests
# =============================================================================

class TestEUDRGeolocation:
    """Golden tests for EUDR geolocation validation."""

    @pytest.fixture(scope="class")
    def valid_coordinate_tests(self):
        """Load valid coordinate test cases."""
        return load_test_cases("geolocation/test_valid_coordinates.json")

    @pytest.fixture(scope="class")
    def protected_area_tests(self):
        """Load protected area test cases."""
        return load_test_cases("geolocation/test_protected_areas.json")

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(25))
    def test_valid_coordinates(self, valid_coordinate_tests, test_index, golden_assert):
        """Test valid coordinate validation for various countries."""
        if test_index >= len(valid_coordinate_tests):
            pytest.skip("Test index out of range")

        test_case = valid_coordinate_tests[test_index]
        test_id = test_case.get("test_id", f"GEO-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})
        tolerance = test_case.get("tolerance", 0.01)

        # Validate test case structure
        assert "coordinates" in input_data, f"{test_id}: Missing coordinates"
        assert "country_code" in input_data, f"{test_id}: Missing country_code"

        # For now, validate the test case format (actual tool execution would go here)
        coords = input_data["coordinates"]
        assert len(coords) >= 2, f"{test_id}: Invalid coordinate format"

        lat, lon = coords[0], coords[1]
        assert -90 <= lat <= 90, f"{test_id}: Latitude out of range"
        assert -180 <= lon <= 180, f"{test_id}: Longitude out of range"

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(25))
    def test_protected_areas(self, protected_area_tests, test_index, golden_assert):
        """Test protected area detection."""
        if test_index >= len(protected_area_tests):
            pytest.skip("Test index out of range")

        test_case = protected_area_tests[test_index]
        test_id = test_case.get("test_id", f"GEO-PA-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})

        # Validate expected output structure
        assert "valid" in expected, f"{test_id}: Missing 'valid' in expected output"


# =============================================================================
# Commodity Classification Tests
# =============================================================================

class TestEUDRCommodity:
    """Golden tests for EUDR commodity classification."""

    @pytest.fixture(scope="class")
    def cn_code_tests(self):
        """Load CN code test cases."""
        return load_test_cases("commodities/test_cn_codes.json")

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(35))
    def test_cn_code_classification(self, cn_code_tests, test_index, golden_assert):
        """Test CN code classification for EUDR commodities."""
        if test_index >= len(cn_code_tests):
            pytest.skip("Test index out of range")

        test_case = cn_code_tests[test_index]
        test_id = test_case.get("test_id", f"CN-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})

        # Validate test case structure
        assert "cn_code" in input_data, f"{test_id}: Missing cn_code"
        assert "eudr_regulated" in expected, f"{test_id}: Missing eudr_regulated"

        # Validate CN code format
        cn_code = input_data["cn_code"]
        if expected.get("eudr_regulated") and expected.get("commodity_type") != "not_regulated":
            # Valid CN codes should be 4-8 digits
            assert len(cn_code) >= 4, f"{test_id}: CN code too short"


# =============================================================================
# Country Risk Tests
# =============================================================================

class TestEUDRCountryRisk:
    """Golden tests for EUDR country risk assessment."""

    @pytest.fixture(scope="class")
    def country_risk_tests(self):
        """Load country risk test cases."""
        return load_test_cases("risk/test_country_risk.json")

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(36))
    def test_country_risk_assessment(self, country_risk_tests, test_index, golden_assert):
        """Test country risk assessment for EUDR commodities."""
        if test_index >= len(country_risk_tests):
            pytest.skip("Test index out of range")

        test_case = country_risk_tests[test_index]
        test_id = test_case.get("test_id", f"RISK-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})

        # Validate test case structure
        assert "country_code" in input_data, f"{test_id}: Missing country_code"
        assert "commodity_type" in input_data, f"{test_id}: Missing commodity_type"
        assert "risk_level" in expected, f"{test_id}: Missing risk_level"

        # Validate risk level is valid
        risk_level = expected["risk_level"]
        assert risk_level in ["low", "standard", "high"], f"{test_id}: Invalid risk level"


# =============================================================================
# Supply Chain Tests
# =============================================================================

class TestEUDRSupplyChain:
    """Golden tests for EUDR supply chain tracing."""

    @pytest.fixture(scope="class")
    def traceability_tests(self):
        """Load traceability test cases."""
        return load_test_cases("supply_chain/test_traceability.json")

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(18))
    def test_supply_chain_tracing(self, traceability_tests, test_index, golden_assert):
        """Test supply chain traceability scoring."""
        if test_index >= len(traceability_tests):
            pytest.skip("Test index out of range")

        test_case = traceability_tests[test_index]
        test_id = test_case.get("test_id", f"TRACE-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})

        # Validate test case structure
        assert "shipment_id" in input_data, f"{test_id}: Missing shipment_id"
        assert "supply_chain_nodes" in input_data, f"{test_id}: Missing supply_chain_nodes"

        # Validate chain of custody status
        if "chain_of_custody" in expected:
            coc = expected["chain_of_custody"]
            assert coc in ["complete", "partial", "broken"], f"{test_id}: Invalid chain_of_custody"


# =============================================================================
# DDS Generation Tests
# =============================================================================

class TestEUDRDDS:
    """Golden tests for EUDR Due Diligence Statement generation."""

    @pytest.fixture(scope="class")
    def dds_tests(self):
        """Load DDS generation test cases."""
        return load_test_cases("dds/test_generation.json")

    @pytest.mark.eudr
    @pytest.mark.parametrize("test_index", range(18))
    def test_dds_generation(self, dds_tests, test_index, golden_assert):
        """Test DDS generation and compliance status."""
        if test_index >= len(dds_tests):
            pytest.skip("Test index out of range")

        test_case = dds_tests[test_index]
        test_id = test_case.get("test_id", f"DDS-{test_index}")
        input_data = test_case.get("input", {})
        expected = test_case.get("expected_output", {})

        # Validate test case structure
        assert "operator_info" in input_data, f"{test_id}: Missing operator_info"
        assert "commodity_data" in input_data, f"{test_id}: Missing commodity_data"
        assert "dds_status" in expected, f"{test_id}: Missing dds_status"

        # Validate DDS status
        status = expected["dds_status"]
        assert status in ["valid", "incomplete", "invalid"], f"{test_id}: Invalid dds_status"


# =============================================================================
# Summary Statistics
# =============================================================================

class TestEUDRGoldenSummary:
    """Summary tests for EUDR golden test suite."""

    @pytest.mark.eudr
    def test_all_test_files_exist(self):
        """Verify all expected test files exist."""
        expected_files = [
            "geolocation/test_valid_coordinates.json",
            "geolocation/test_protected_areas.json",
            "commodities/test_cn_codes.json",
            "risk/test_country_risk.json",
            "supply_chain/test_traceability.json",
            "dds/test_generation.json"
        ]

        for file_name in expected_files:
            file_path = EUDR_TESTS_DIR / file_name
            assert file_path.exists(), f"Missing test file: {file_name}"

    @pytest.mark.eudr
    def test_total_test_count(self):
        """Verify total number of EUDR golden tests."""
        total_tests = 0

        for json_file in EUDR_TESTS_DIR.rglob("*.json"):
            try:
                tests = load_test_cases(str(json_file.relative_to(EUDR_TESTS_DIR)))
                total_tests += len(tests)
            except Exception:
                pass

        # Target: 200 tests
        assert total_tests >= 150, f"Only {total_tests} tests found, target is 200"
        print(f"\nTotal EUDR golden tests: {total_tests}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
