"""
CBAM Importer Copilot - Pytest Configuration & Fixtures

Provides shared fixtures for all tests.

Version: 1.0.0
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cbam_pipeline import CBAMPipeline
from sdk.cbam_sdk import CBAMConfig


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_shipments_data() -> list:
    """Sample shipment records for testing."""
    return [
        {
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "2025-09-15",
            "supplier_id": "SUP-CN-001",
            "invoice_number": "INV-2025-001"
        },
        {
            "cn_code": "72071210",
            "country_of_origin": "TR",
            "quantity_tons": 8.2,
            "import_date": "2025-09-20",
            "supplier_id": "SUP-TR-002",
            "invoice_number": "INV-2025-002"
        },
        {
            "cn_code": "76011000",
            "country_of_origin": "RU",
            "quantity_tons": 12.0,
            "import_date": "2025-09-25",
            "supplier_id": "SUP-RU-001",
            "invoice_number": "INV-2025-003"
        },
        {
            "cn_code": "25232900",
            "country_of_origin": "UA",
            "quantity_tons": 20.5,
            "import_date": "2025-09-28",
            "supplier_id": "SUP-UA-001",
            "invoice_number": "INV-2025-004"
        },
        {
            "cn_code": "28342100",
            "country_of_origin": "IN",
            "quantity_tons": 5.5,
            "import_date": "2025-09-30",
            "supplier_id": "SUP-IN-001",
            "invoice_number": "INV-2025-005"
        }
    ]


@pytest.fixture
def sample_shipments_csv(tmp_path, sample_shipments_data) -> str:
    """Create temporary CSV file with sample shipments."""
    csv_path = tmp_path / "test_shipments.csv"

    # Write CSV
    df = pd.DataFrame(sample_shipments_data)
    df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def sample_shipments_excel(tmp_path, sample_shipments_data) -> str:
    """Create temporary Excel file with sample shipments."""
    excel_path = tmp_path / "test_shipments.xlsx"

    # Write Excel
    df = pd.DataFrame(sample_shipments_data)
    df.to_excel(excel_path, index=False, engine='openpyxl')

    return str(excel_path)


@pytest.fixture
def sample_shipments_json(tmp_path, sample_shipments_data) -> str:
    """Create temporary JSON file with sample shipments."""
    json_path = tmp_path / "test_shipments.json"

    # Write JSON
    with open(json_path, 'w') as f:
        json.dump(sample_shipments_data, f, indent=2)

    return str(json_path)


@pytest.fixture
def sample_shipments_dataframe(sample_shipments_data) -> pd.DataFrame:
    """Create pandas DataFrame with sample shipments."""
    return pd.DataFrame(sample_shipments_data)


@pytest.fixture
def invalid_shipments_data() -> list:
    """Sample shipment records with errors for testing validation."""
    return [
        {
            # Missing cn_code (error)
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "2025-09-15"
        },
        {
            "cn_code": "INVALID",  # Invalid format
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "2025-09-15"
        },
        {
            "cn_code": "72071100",
            "country_of_origin": "XX",  # Invalid country
            "quantity_tons": 15.5,
            "import_date": "2025-09-15"
        },
        {
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": -5.0,  # Negative quantity
            "import_date": "2025-09-15"
        },
        {
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "invalid-date"  # Invalid date
        }
    ]


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def cbam_config() -> CBAMConfig:
    """Standard CBAM configuration for testing."""
    return CBAMConfig(
        importer_name="Test Company BV",
        importer_country="NL",
        importer_eori="NL123456789012",
        declarant_name="Test User",
        declarant_position="Test Manager"
    )


@pytest.fixture
def importer_info() -> Dict[str, Any]:
    """Importer information dictionary for testing."""
    return {
        "name": "Test Company BV",
        "country": "NL",
        "eori": "NL123456789012",
        "declarant_name": "Test User",
        "declarant_position": "Test Manager"
    }


# ============================================================================
# Pipeline Fixtures
# ============================================================================

@pytest.fixture
def cbam_pipeline() -> CBAMPipeline:
    """Initialize CBAM pipeline with test configuration."""
    return CBAMPipeline(
        cn_codes_path="data/cn_codes.json",
        cbam_rules_path="rules/cbam_rules.yaml",
        suppliers_path=None,
        enable_provenance=True
    )


# ============================================================================
# File Path Fixtures
# ============================================================================

@pytest.fixture
def test_output_dir(tmp_path) -> str:
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def cn_codes_path() -> str:
    """Path to CN codes database."""
    return "data/cn_codes.json"


@pytest.fixture
def cbam_rules_path() -> str:
    """Path to CBAM validation rules."""
    return "rules/cbam_rules.yaml"


@pytest.fixture
def suppliers_path() -> str:
    """Path to demo suppliers file."""
    return "examples/demo_suppliers.yaml"


# ============================================================================
# Expected Results Fixtures
# ============================================================================

@pytest.fixture
def expected_emissions() -> Dict[str, float]:
    """Expected emission factors for test CN codes."""
    return {
        "72071100": 0.8,   # Iron/steel
        "72071210": 0.75,  # Iron/steel
        "76011000": 1.2,   # Aluminum
        "25232900": 0.5,   # Cement
        "28342100": 0.9    # Fertilizer
    }


@pytest.fixture
def expected_total_emissions() -> float:
    """Expected total emissions for sample data."""
    # 15.5 * 0.8 + 8.2 * 0.75 + 12.0 * 1.2 + 20.5 * 0.5 + 5.5 * 0.9
    return 12.4 + 6.15 + 14.4 + 10.25 + 4.95  # = 48.15


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def large_shipments_data(sample_shipments_data) -> list:
    """Generate large dataset for performance testing (1000 records)."""
    large_data = []
    for i in range(200):  # 200 copies of 5 records = 1000 records
        for shipment in sample_shipments_data:
            large_data.append({
                **shipment,
                "invoice_number": f"INV-2025-{i:04d}"
            })
    return large_data


@pytest.fixture
def large_shipments_csv(tmp_path, large_shipments_data) -> str:
    """Create large CSV file for performance testing."""
    csv_path = tmp_path / "large_shipments.csv"
    df = pd.DataFrame(large_shipments_data)
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ============================================================================
# Provenance Test Fixtures
# ============================================================================

@pytest.fixture
def sample_provenance_record() -> Dict[str, Any]:
    """Sample provenance record for testing."""
    return {
        "report_id": "TEST-2025Q4-001",
        "generated_at": "2025-10-15T14:30:00Z",
        "input_file_integrity": {
            "file_name": "test_shipments.csv",
            "sha256_hash": "abc123def456...",
            "file_size_bytes": 1024,
            "hash_timestamp": "2025-10-15T14:29:58Z"
        },
        "execution_environment": {
            "python_version": "3.11.5",
            "os": "Linux",
            "timestamp": "2025-10-15T14:30:00Z"
        },
        "dependencies": {
            "pandas": "2.1.0",
            "pydantic": "2.4.0"
        },
        "agent_execution": [
            {
                "agent_name": "ShipmentIntakeAgent",
                "start_time": "2025-10-15T14:30:00Z",
                "end_time": "2025-10-15T14:30:02Z",
                "status": "success"
            }
        ],
        "reproducibility": {
            "deterministic": True,
            "zero_hallucination": True
        }
    }


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def assert_valid_report():
    """Helper function to validate CBAM report structure."""
    def _validate(report: Dict[str, Any]) -> bool:
        """Validate report has required structure."""
        required_keys = [
            "report_metadata",
            "emissions_summary",
            "detailed_goods",
            "aggregations",
            "validation_results",
            "provenance"
        ]

        for key in required_keys:
            assert key in report, f"Missing required key: {key}"

        # Validate report_metadata
        metadata = report["report_metadata"]
        assert "report_id" in metadata
        assert "generated_at" in metadata
        assert "importer" in metadata

        # Validate emissions_summary
        summary = report["emissions_summary"]
        assert "total_embedded_emissions_tco2" in summary
        assert "total_quantity_tons" in summary
        assert "total_shipments" in summary

        # Validate detailed_goods is list
        assert isinstance(report["detailed_goods"], list)

        # Validate provenance
        provenance = report["provenance"]
        assert "input_file_integrity" in provenance
        assert "execution_environment" in provenance
        assert "reproducibility" in provenance

        return True

    return _validate


@pytest.fixture
def assert_zero_hallucination():
    """Helper function to verify zero hallucination guarantee."""
    def _validate(report: Dict[str, Any]) -> bool:
        """Validate zero hallucination architecture."""
        provenance = report.get("provenance", {})
        reproducibility = provenance.get("reproducibility", {})

        assert reproducibility.get("deterministic") == True, \
            "Report must be deterministic"
        assert reproducibility.get("zero_hallucination") == True, \
            "Report must guarantee zero hallucination"

        # Check all emissions have calculation method = "deterministic"
        for good in report.get("detailed_goods", []):
            calc_method = good.get("calculation_method")
            assert calc_method == "deterministic", \
                f"Non-deterministic calculation found: {calc_method}"

        return True

    return _validate


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup any test files created during tests."""
    yield
    # Cleanup happens after test
    # tmp_path fixture handles cleanup automatically


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "compliance: Compliance tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (skip with -m 'not slow')"
    )
