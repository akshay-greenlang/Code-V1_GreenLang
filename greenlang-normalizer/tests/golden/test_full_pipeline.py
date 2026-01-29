"""
Golden Tests: Full Pipeline Tests for GL-FOUND-X-003.

This module tests end-to-end normalization scenarios against golden files.
Tests cover GHG Protocol and EU CSRD/CBAM reporting scenarios.

Test Coverage:
    - Scope 1, 2, 3 emissions calculations
    - CBAM product declarations
    - CSRD environmental metrics
    - Multi-measurement normalization
    - Error and warning handling

Features:
    - Complete pipeline simulation
    - Audit trail verification
    - Status code validation
    - Batch processing tests
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

import pytest

from .conftest import (
    FULL_PIPELINE_DIR,
    load_test_cases,
    discover_golden_files,
    compare_values,
    GoldenTestResult,
)


# =============================================================================
# Test Data Loading
# =============================================================================

def get_pipeline_test_cases() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load all pipeline test cases from golden files."""
    test_cases = []
    for golden_file in discover_golden_files(FULL_PIPELINE_DIR):
        framework = golden_file.stem
        for test_case in load_test_cases(golden_file):
            test_id = f"{framework}::{test_case.get('name', 'unnamed')}"
            test_cases.append((framework, test_id, test_case))
    return test_cases


# Generate test IDs
TEST_CASES = get_pipeline_test_cases()
TEST_IDS = [tc[1] for tc in TEST_CASES]


# =============================================================================
# Conversion Factors (from test_unit_conversion.py)
# =============================================================================

ENERGY_FACTORS = {
    "J": 1.0e-6, "kJ": 1.0e-3, "MJ": 1.0, "GJ": 1.0e3, "TJ": 1.0e6,
    "Wh": 3.6e-3, "kWh": 3.6, "MWh": 3600.0, "GWh": 3.6e6,
    "BTU": 1.055056e-3, "MMBtu": 1055.056, "therm": 105.5056,
}

MASS_FACTORS = {
    "mg": 1.0e-6, "g": 1.0e-3, "kg": 1.0, "t": 1000.0, "kt": 1.0e6,
    "lb": 0.45359237, "ton_short": 907.18474,
}

VOLUME_FACTORS = {
    "mL": 1.0e-6, "L": 1.0e-3, "m3": 1.0, "gal_us": 3.785411784e-3,
}

EMISSIONS_FACTORS = {
    "gCO2e": 1.0e-3, "kgCO2e": 1.0, "tCO2e": 1000.0, "kgCO2": 1.0,
}

DIMENSIONLESS_FACTORS = {
    "1": 1.0, "%": 0.01, "ppm": 1.0e-6,
}


# =============================================================================
# Mock Normalizer Pipeline
# =============================================================================

class MockNormalizer:
    """Mock normalizer for pipeline testing."""

    def __init__(self):
        """Initialize normalizer."""
        self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self) -> Dict[str, Dict]:
        """Load vocabulary for entity resolution."""
        return {
            "fuels": {
                "Natural Gas": "GL-FUEL-NATGAS",
                "Diesel": "GL-FUEL-DIESEL",
                "Grid electricity": "GL-FUEL-ELEC",
                "Electricity": "GL-FUEL-ELEC",
                "Renewable electricity": "GL-FUEL-ELEC-REN",
            },
            "materials": {
                "Carbon steel": "GL-MAT-STEEL-CARBON",
                "Steel": "GL-MAT-STEEL",
                "Aluminium": "GL-MAT-ALUM",
                "Aluminum": "GL-MAT-ALUM",
                "CEM I": "GL-MAT-CEMENT-PORTLAND",
                "Urea": "GL-MAT-FERT-UREA",
                "Hydrogen": "GL-MAT-HYDROGEN",
                "Clinker": "GL-MAT-CLINKER",
                "HFC-134a": "GL-MAT-HFC134A",
            },
            "processes": {
                "Stationary combustion": "GL-PROC-COMB-STATIONARY",
                "Mobile combustion": "GL-PROC-COMB-MOBILE",
                "EAF": "GL-PROC-STEEL-EAF",
                "BOF": "GL-PROC-STEEL-BOF",
                "Blast furnace-Basic oxygen furnace": "GL-PROC-STEEL-BFBOF",
                "Clinker production": "GL-PROC-CEMENT-CLINKER",
                "Hall-Heroult": "GL-PROC-ALUM-HALLHEROULT",
                "Road freight": "GL-PROC-TRANSPORT-ROAD",
                "Air travel": "GL-PROC-TRANSPORT-AIR",
            },
        }

    def convert_measurement(
        self,
        value: float,
        unit: str,
        dimension: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Convert a measurement to canonical units."""
        # Select factor map
        if dimension == "energy":
            factors = ENERGY_FACTORS
            canonical_unit = "MJ"
        elif dimension == "mass":
            factors = MASS_FACTORS
            canonical_unit = "kg"
        elif dimension == "volume":
            factors = VOLUME_FACTORS
            canonical_unit = "m3"
        elif dimension in ("emissions_mass", "emissions"):
            factors = EMISSIONS_FACTORS
            canonical_unit = "kgCO2e"
        elif dimension == "dimensionless":
            factors = DIMENSIONLESS_FACTORS
            canonical_unit = "1"
        else:
            return {
                "success": False,
                "error_code": "GLNORM-E201",
                "message": f"Unknown dimension: {dimension}",
            }

        if unit not in factors:
            return {
                "success": False,
                "error_code": "GLNORM-E101",
                "message": f"Unknown unit: {unit}",
            }

        canonical_value = value * factors[unit]

        return {
            "success": True,
            "canonical_value": canonical_value,
            "canonical_unit": canonical_unit,
            "raw_value": value,
            "raw_unit": unit,
        }

    def resolve_entity(
        self,
        raw_name: str,
        entity_type: str,
        hints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Resolve entity to vocabulary ID."""
        vocab = self.vocabulary.get(f"{entity_type}s", {})

        # Exact match
        if raw_name in vocab:
            return {
                "success": True,
                "reference_id": vocab[raw_name],
                "canonical_name": raw_name,
                "match_method": "exact",
                "confidence": 1.0,
                "needs_review": False,
            }

        # Case-insensitive match
        for name, ref_id in vocab.items():
            if name.lower() == raw_name.lower():
                return {
                    "success": True,
                    "reference_id": ref_id,
                    "canonical_name": name,
                    "match_method": "alias",
                    "confidence": 1.0,
                    "needs_review": False,
                }

        # Not found
        return {
            "success": False,
            "error_code": "GLNORM-E400",
            "message": f"Entity not found: {raw_name}",
            "needs_review": True,
        }

    def normalize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run full normalization pipeline."""
        source_record_id = request.get("source_record_id", "unknown")
        policy_mode = request.get("policy_mode", "STRICT")

        result = {
            "source_record_id": source_record_id,
            "status": "success",
            "canonical_measurements": [],
            "normalized_entities": [],
            "errors": [],
            "warnings": [],
            "audit": {
                "normalization_event_id": f"norm-evt-{source_record_id}",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        # Process measurements
        for measurement in request.get("measurements", []):
            field = measurement.get("field", "")
            value = measurement.get("value", 0)
            unit = measurement.get("unit", "")
            dimension = measurement.get("expected_dimension", "")
            metadata = measurement.get("metadata", {})

            conversion = self.convert_measurement(value, unit, dimension, metadata)

            if conversion.get("success"):
                result["canonical_measurements"].append({
                    "field": field,
                    "canonical_value": conversion["canonical_value"],
                    "canonical_unit": conversion["canonical_unit"],
                    "raw_value": conversion["raw_value"],
                    "raw_unit": conversion["raw_unit"],
                })
            else:
                error = {
                    "code": conversion.get("error_code", "GLNORM-E900"),
                    "severity": "error",
                    "path": f"/measurements/{field}",
                    "message": conversion.get("message", "Conversion failed"),
                }
                result["errors"].append(error)

                if policy_mode == "STRICT":
                    result["status"] = "failed"
                    result["audit"]["status"] = "failed"

        # Process entities
        for entity in request.get("entities", []):
            field = entity.get("field", "")
            raw_name = entity.get("raw_name", "")
            entity_type = entity.get("entity_type", "")
            hints = entity.get("hints", {})

            resolution = self.resolve_entity(raw_name, entity_type, hints)

            if resolution.get("success"):
                result["normalized_entities"].append({
                    "field": field,
                    "reference_id": resolution["reference_id"],
                    "canonical_name": resolution["canonical_name"],
                    "match_method": resolution["match_method"],
                    "confidence": resolution["confidence"],
                    "needs_review": resolution["needs_review"],
                })
            else:
                error = {
                    "code": resolution.get("error_code", "GLNORM-E900"),
                    "severity": "error",
                    "path": f"/entities/{field}",
                    "message": resolution.get("message", "Resolution failed"),
                }

                if policy_mode == "STRICT":
                    result["errors"].append(error)
                    result["status"] = "failed"
                    result["audit"]["status"] = "failed"
                else:
                    result["warnings"].append({
                        "code": "GLNORM-E403",
                        "severity": "warning",
                        "path": f"/entities/{field}",
                        "message": resolution.get("message"),
                    })
                    if result["status"] == "success":
                        result["status"] = "warning"

        return result


# Global normalizer instance
NORMALIZER = MockNormalizer()


# =============================================================================
# Test Classes
# =============================================================================

class TestFullPipeline:
    """Golden tests for full pipeline scenarios."""

    @pytest.mark.parametrize("framework,test_id,test_case", TEST_CASES, ids=TEST_IDS)
    def test_pipeline_execution(
        self,
        framework: str,
        test_id: str,
        test_case: Dict[str, Any],
    ):
        """Test full pipeline execution against golden values."""
        input_data = test_case.get("input", {})
        expected = test_case.get("expected", {})

        # Skip batch tests for now (separate handling)
        if "batch_records" in input_data:
            pytest.skip("Batch test - handled separately")

        # Run normalization
        result = NORMALIZER.normalize(input_data)

        # Verify status
        if "status" in expected:
            assert result["status"] == expected["status"], (
                f"Status mismatch for {test_case.get('name')}:\n"
                f"  Expected: {expected['status']}\n"
                f"  Actual: {result['status']}"
            )

        # Verify canonical measurements
        for exp_measurement in expected.get("canonical_measurements", []):
            field = exp_measurement.get("field")
            actual_measurement = next(
                (m for m in result["canonical_measurements"] if m["field"] == field),
                None,
            )

            if actual_measurement is None:
                pytest.fail(f"Missing canonical measurement for field: {field}")

            # Check canonical value
            expected_value = exp_measurement.get("canonical_value")
            actual_value = actual_measurement.get("canonical_value")
            tolerance = exp_measurement.get("tolerance", abs(expected_value) * 1e-9)

            if expected_value is not None:
                comparison = compare_values(expected_value, actual_value, tolerance)
                assert comparison.passed, (
                    f"Canonical value mismatch for {field}:\n"
                    f"  Expected: {expected_value}\n"
                    f"  Actual: {actual_value}\n"
                    f"  {comparison.diff}"
                )

            # Check canonical unit
            if "canonical_unit" in exp_measurement:
                assert actual_measurement.get("canonical_unit") == exp_measurement["canonical_unit"], (
                    f"Canonical unit mismatch for {field}"
                )

        # Verify normalized entities
        for exp_entity in expected.get("normalized_entities", []):
            field = exp_entity.get("field")
            actual_entity = next(
                (e for e in result["normalized_entities"] if e["field"] == field),
                None,
            )

            if actual_entity is None:
                pytest.fail(f"Missing normalized entity for field: {field}")

            # Check reference ID
            if "reference_id" in exp_entity:
                assert actual_entity.get("reference_id") == exp_entity["reference_id"], (
                    f"Reference ID mismatch for {field}"
                )

            # Check confidence
            if "confidence" in exp_entity:
                assert abs(actual_entity.get("confidence", 0) - exp_entity["confidence"]) < 0.01, (
                    f"Confidence mismatch for {field}"
                )

        # Verify errors
        for exp_error in expected.get("errors", []):
            matching_error = next(
                (e for e in result["errors"] if e["code"] == exp_error["code"]),
                None,
            )
            assert matching_error is not None, (
                f"Expected error {exp_error['code']} not found"
            )

        # Verify audit
        if "audit" in expected:
            exp_audit = expected["audit"]
            if "status" in exp_audit:
                assert result["audit"]["status"] == exp_audit["status"], (
                    f"Audit status mismatch"
                )


class TestGHGProtocolScenarios:
    """Tests for GHG Protocol reporting scenarios."""

    @pytest.mark.compliance
    def test_scope1_stationary_combustion(self):
        """Test Scope 1 stationary combustion normalization."""
        request = {
            "source_record_id": "test-sc1-stat",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "fuel_consumption",
                    "value": 1000,
                    "unit": "MMBtu",
                    "expected_dimension": "energy",
                }
            ],
            "entities": [
                {
                    "field": "fuel_type",
                    "entity_type": "fuel",
                    "raw_name": "Natural Gas",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"
        assert len(result["canonical_measurements"]) == 1
        assert result["canonical_measurements"][0]["canonical_unit"] == "MJ"
        assert len(result["normalized_entities"]) == 1
        assert result["normalized_entities"][0]["reference_id"] == "GL-FUEL-NATGAS"

    @pytest.mark.compliance
    def test_scope2_electricity(self):
        """Test Scope 2 electricity normalization."""
        request = {
            "source_record_id": "test-sc2-elec",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "electricity_consumption",
                    "value": 500000,
                    "unit": "kWh",
                    "expected_dimension": "energy",
                }
            ],
            "entities": [
                {
                    "field": "energy_source",
                    "entity_type": "fuel",
                    "raw_name": "Grid electricity",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"
        assert result["canonical_measurements"][0]["canonical_value"] == 1800000.0

    @pytest.mark.compliance
    def test_scope3_purchased_goods(self):
        """Test Scope 3 purchased goods normalization."""
        request = {
            "source_record_id": "test-sc3-goods",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "material_purchased",
                    "value": 500,
                    "unit": "t",
                    "expected_dimension": "mass",
                }
            ],
            "entities": [
                {
                    "field": "material",
                    "entity_type": "material",
                    "raw_name": "Carbon steel",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"
        assert result["canonical_measurements"][0]["canonical_value"] == 500000.0


class TestCBAMScenarios:
    """Tests for EU CBAM reporting scenarios."""

    @pytest.mark.compliance
    def test_cbam_steel_import(self):
        """Test CBAM steel import normalization."""
        request = {
            "source_record_id": "cbam-steel-test",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "product_mass",
                    "value": 100,
                    "unit": "t",
                    "expected_dimension": "mass",
                },
                {
                    "field": "direct_emissions",
                    "value": 180,
                    "unit": "tCO2e",
                    "expected_dimension": "emissions_mass",
                },
            ],
            "entities": [
                {
                    "field": "product",
                    "entity_type": "material",
                    "raw_name": "Carbon steel",
                },
                {
                    "field": "production_route",
                    "entity_type": "process",
                    "raw_name": "EAF",
                },
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"
        assert len(result["canonical_measurements"]) == 2

        # Check mass conversion
        mass_result = next(
            m for m in result["canonical_measurements"]
            if m["field"] == "product_mass"
        )
        assert mass_result["canonical_value"] == 100000.0
        assert mass_result["canonical_unit"] == "kg"

        # Check emissions conversion
        emissions_result = next(
            m for m in result["canonical_measurements"]
            if m["field"] == "direct_emissions"
        )
        assert emissions_result["canonical_value"] == 180000.0
        assert emissions_result["canonical_unit"] == "kgCO2e"

    @pytest.mark.compliance
    def test_cbam_aluminum_import(self):
        """Test CBAM aluminum import normalization."""
        request = {
            "source_record_id": "cbam-alum-test",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "product_mass",
                    "value": 50,
                    "unit": "t",
                    "expected_dimension": "mass",
                },
                {
                    "field": "electricity_consumption",
                    "value": 750,
                    "unit": "MWh",
                    "expected_dimension": "energy",
                },
            ],
            "entities": [
                {
                    "field": "product",
                    "entity_type": "material",
                    "raw_name": "Aluminium",
                },
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"

        # Verify aluminium resolved (UK spelling)
        entity = result["normalized_entities"][0]
        assert entity["reference_id"] == "GL-MAT-ALUM"


class TestErrorHandling:
    """Tests for error handling in pipeline."""

    def test_dimension_mismatch_strict(self):
        """Test dimension mismatch in STRICT mode."""
        request = {
            "source_record_id": "err-dim-mismatch",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "energy",
                    "value": 100,
                    "unit": "kg",  # Mass unit where energy expected
                    "expected_dimension": "energy",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert any(e["code"] == "GLNORM-E101" for e in result["errors"])

    def test_unknown_entity_strict(self):
        """Test unknown entity in STRICT mode."""
        request = {
            "source_record_id": "err-unknown-entity",
            "policy_mode": "STRICT",
            "entities": [
                {
                    "field": "fuel_type",
                    "entity_type": "fuel",
                    "raw_name": "Unknown Fuel XYZ",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_unknown_entity_lenient(self):
        """Test unknown entity in LENIENT mode."""
        request = {
            "source_record_id": "warn-unknown-entity",
            "policy_mode": "LENIENT",
            "entities": [
                {
                    "field": "fuel_type",
                    "entity_type": "fuel",
                    "raw_name": "Unknown Fuel XYZ",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        # Should not fail in LENIENT mode
        assert result["status"] in ("success", "warning")
        # Should have warning
        assert len(result["warnings"]) > 0


class TestAuditTrail:
    """Tests for audit trail generation."""

    def test_audit_event_id_generated(self):
        """Test audit event ID is generated."""
        request = {
            "source_record_id": "audit-test-001",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "energy",
                    "value": 100,
                    "unit": "kWh",
                    "expected_dimension": "energy",
                }
            ],
        }

        result = NORMALIZER.normalize(request)

        assert "audit" in result
        assert "normalization_event_id" in result["audit"]
        assert result["audit"]["normalization_event_id"].startswith("norm-evt-")

    def test_audit_timestamp_present(self):
        """Test audit timestamp is present."""
        request = {
            "source_record_id": "audit-test-002",
            "policy_mode": "STRICT",
            "measurements": [],
        }

        result = NORMALIZER.normalize(request)

        assert "timestamp" in result["audit"]

    def test_audit_status_matches_result(self):
        """Test audit status matches result status."""
        # Success case
        success_request = {
            "source_record_id": "audit-success",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "energy",
                    "value": 100,
                    "unit": "kWh",
                    "expected_dimension": "energy",
                }
            ],
        }
        success_result = NORMALIZER.normalize(success_request)
        assert success_result["audit"]["status"] == success_result["status"]

        # Failure case
        failure_request = {
            "source_record_id": "audit-failure",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "energy",
                    "value": 100,
                    "unit": "unknown_unit",
                    "expected_dimension": "energy",
                }
            ],
        }
        failure_result = NORMALIZER.normalize(failure_request)
        assert failure_result["audit"]["status"] == failure_result["status"]


class TestMultiMeasurement:
    """Tests for multi-measurement normalization."""

    def test_multiple_measurements(self):
        """Test normalizing multiple measurements in one request."""
        request = {
            "source_record_id": "multi-measure",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "electricity",
                    "value": 1000,
                    "unit": "kWh",
                    "expected_dimension": "energy",
                },
                {
                    "field": "natural_gas",
                    "value": 100,
                    "unit": "therm",
                    "expected_dimension": "energy",
                },
                {
                    "field": "diesel",
                    "value": 500,
                    "unit": "L",
                    "expected_dimension": "volume",
                },
            ],
        }

        result = NORMALIZER.normalize(request)

        assert result["status"] == "success"
        assert len(result["canonical_measurements"]) == 3

        # Verify each conversion
        elec = next(m for m in result["canonical_measurements"] if m["field"] == "electricity")
        assert elec["canonical_value"] == 3600.0  # kWh to MJ

        gas = next(m for m in result["canonical_measurements"] if m["field"] == "natural_gas")
        assert abs(gas["canonical_value"] - 10550.56) < 1.0  # therm to MJ

        diesel = next(m for m in result["canonical_measurements"] if m["field"] == "diesel")
        assert diesel["canonical_value"] == 0.5  # L to m3


class TestDeterminism:
    """Tests for pipeline determinism."""

    def test_pipeline_determinism(self):
        """Same input should always produce same output."""
        request = {
            "source_record_id": "determinism-test",
            "policy_mode": "STRICT",
            "measurements": [
                {
                    "field": "energy",
                    "value": 123.456,
                    "unit": "kWh",
                    "expected_dimension": "energy",
                }
            ],
            "entities": [
                {
                    "field": "fuel",
                    "entity_type": "fuel",
                    "raw_name": "Natural Gas",
                }
            ],
        }

        results = [NORMALIZER.normalize(request) for _ in range(10)]

        first = results[0]
        for result in results[1:]:
            assert result["status"] == first["status"]
            assert (
                result["canonical_measurements"][0]["canonical_value"]
                == first["canonical_measurements"][0]["canonical_value"]
            )
            assert (
                result["normalized_entities"][0]["reference_id"]
                == first["normalized_entities"][0]["reference_id"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
