# -*- coding: utf-8 -*-
"""
GL-CBAM-APP - Shared Test Fixtures and Configuration

This conftest.py provides common fixtures and configuration for all tests
across both CBAM-Importer-Copilot and CBAM-Refactored.

Version: 1.0.0
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from greenlang.determinism import DeterministicClock

# Add project roots to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "CBAM-Importer-Copilot"))
sys.path.insert(0, str(PROJECT_ROOT / "CBAM-Refactored"))


# ============================================================================
# SESSION-LEVEL FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cbam_copilot_root():
    """Return CBAM-Importer-Copilot root directory."""
    return PROJECT_ROOT / "CBAM-Importer-Copilot"


@pytest.fixture(scope="session")
def cbam_refactored_root():
    """Return CBAM-Refactored root directory."""
    return PROJECT_ROOT / "CBAM-Refactored"


# ============================================================================
# COMMON TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_cn_codes() -> Dict[str, Dict[str, Any]]:
    """Sample CN codes database for testing."""
    return {
        "72071100": {
            "product_group": "Iron and Steel",
            "description": "Semi-finished products of iron or non-alloy steel",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 1.85
        },
        "72071210": {
            "product_group": "Iron and Steel",
            "description": "Flat-rolled products of iron or non-alloy steel",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 1.75
        },
        "76011000": {
            "product_group": "Aluminum",
            "description": "Unwrought aluminum, not alloyed",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 2.20
        },
        "25232900": {
            "product_group": "Cement",
            "description": "Portland cement",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 0.85
        },
        "28112100": {
            "product_group": "Hydrogen",
            "description": "Hydrogen in gaseous state",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 3.50
        },
        "28342100": {
            "product_group": "Fertilizers",
            "description": "Nitrates of potassium",
            "cbam_covered": True,
            "default_emission_factor_tco2_per_ton": 1.20
        }
    }


@pytest.fixture
def sample_suppliers() -> Dict[str, List[Dict[str, Any]]]:
    """Sample suppliers database for testing."""
    return {
        "suppliers": [
            {
                "supplier_id": "SUP-CN-001",
                "company_name": "China Steel Manufacturing Co Ltd",
                "country": "CN",
                "has_actual_data": True,
                "actual_emissions_data": {
                    "direct_emissions_tco2_per_ton": 1.65,
                    "indirect_emissions_tco2_per_ton": 0.25,
                    "total_emissions_tco2_per_ton": 1.90,
                    "data_quality": "high",
                    "verification": "ISO 14064-1",
                    "verification_date": "2024-09-01"
                }
            },
            {
                "supplier_id": "SUP-TR-002",
                "company_name": "Turkish Steel Industries",
                "country": "TR",
                "has_actual_data": True,
                "actual_emissions_data": {
                    "direct_emissions_tco2_per_ton": 1.70,
                    "indirect_emissions_tco2_per_ton": 0.20,
                    "total_emissions_tco2_per_ton": 1.90,
                    "data_quality": "medium",
                    "verification": "Third-party audit"
                }
            },
            {
                "supplier_id": "SUP-RU-001",
                "company_name": "Russian Aluminum Corp",
                "country": "RU",
                "has_actual_data": False
            },
            {
                "supplier_id": "SUP-UA-001",
                "company_name": "Ukraine Cement Industries",
                "country": "UA",
                "has_actual_data": False
            },
            {
                "supplier_id": "SUP-IN-001",
                "company_name": "India Fertilizers Ltd",
                "country": "IN",
                "has_actual_data": False
            }
        ]
    }


@pytest.fixture
def sample_shipments_data() -> List[Dict[str, Any]]:
    """Generate realistic sample shipment data."""
    return [
        {
            "shipment_id": "SHIP-2024-001",
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "net_mass_kg": 15500,
            "import_date": "2024-09-15",
            "quarter": "Q3-2024",
            "supplier_id": "SUP-CN-001",
            "invoice_number": "INV-2024-001",
            "importer_country": "NL",
            "importer_eori": "NL123456789012",
            "has_actual_emissions": "YES"
        },
        {
            "shipment_id": "SHIP-2024-002",
            "cn_code": "72071210",
            "country_of_origin": "TR",
            "quantity_tons": 8.2,
            "net_mass_kg": 8200,
            "import_date": "2024-09-20",
            "quarter": "Q3-2024",
            "supplier_id": "SUP-TR-002",
            "invoice_number": "INV-2024-002",
            "importer_country": "DE",
            "importer_eori": "DE987654321098",
            "has_actual_emissions": "YES"
        },
        {
            "shipment_id": "SHIP-2024-003",
            "cn_code": "76011000",
            "country_of_origin": "RU",
            "quantity_tons": 12.0,
            "net_mass_kg": 12000,
            "import_date": "2024-09-25",
            "quarter": "Q3-2024",
            "supplier_id": "SUP-RU-001",
            "invoice_number": "INV-2024-003",
            "importer_country": "FR",
            "importer_eori": "FR111222333444",
            "has_actual_emissions": "NO"
        },
        {
            "shipment_id": "SHIP-2024-004",
            "cn_code": "25232900",
            "country_of_origin": "UA",
            "quantity_tons": 20.5,
            "net_mass_kg": 20500,
            "import_date": "2024-09-28",
            "quarter": "Q3-2024",
            "supplier_id": "SUP-UA-001",
            "invoice_number": "INV-2024-004",
            "importer_country": "IT",
            "importer_eori": "IT555666777888",
            "has_actual_emissions": "NO"
        },
        {
            "shipment_id": "SHIP-2024-005",
            "cn_code": "28342100",
            "country_of_origin": "IN",
            "quantity_tons": 5.5,
            "net_mass_kg": 5500,
            "import_date": "2024-09-30",
            "quarter": "Q3-2024",
            "supplier_id": "SUP-IN-001",
            "invoice_number": "INV-2024-005",
            "importer_country": "ES",
            "importer_eori": "ES999888777666",
            "has_actual_emissions": "NO"
        }
    ]


@pytest.fixture
def invalid_shipments_data() -> List[Dict[str, Any]]:
    """Sample invalid shipment data for validation testing."""
    return [
        {
            # Missing cn_code (error)
            "shipment_id": "SHIP-ERR-001",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "2024-09-15"
        },
        {
            # Invalid CN code format
            "shipment_id": "SHIP-ERR-002",
            "cn_code": "INVALID",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "2024-09-15"
        },
        {
            # Invalid country code
            "shipment_id": "SHIP-ERR-003",
            "cn_code": "72071100",
            "country_of_origin": "XX",
            "quantity_tons": 15.5,
            "import_date": "2024-09-15"
        },
        {
            # Negative quantity
            "shipment_id": "SHIP-ERR-004",
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": -5.0,
            "import_date": "2024-09-15"
        },
        {
            # Invalid date format
            "shipment_id": "SHIP-ERR-005",
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "import_date": "invalid-date"
        }
    ]


# ============================================================================
# PERFORMANCE TEST FIXTURES
# ============================================================================

@pytest.fixture
def shipments_1k(sample_shipments_data) -> List[Dict[str, Any]]:
    """Generate 1,000 shipments for performance testing."""
    shipments = []
    for i in range(200):  # 200 × 5 = 1,000
        for shipment in sample_shipments_data:
            shipments.append({
                **shipment,
                "shipment_id": f"SHIP-1K-{i:04d}-{shipment['shipment_id'][-3:]}",
                "invoice_number": f"INV-1K-{i:04d}"
            })
    return shipments


@pytest.fixture
def shipments_10k(sample_shipments_data) -> List[Dict[str, Any]]:
    """Generate 10,000 shipments for performance testing."""
    shipments = []
    for i in range(2000):  # 2,000 × 5 = 10,000
        for shipment in sample_shipments_data:
            shipments.append({
                **shipment,
                "shipment_id": f"SHIP-10K-{i:05d}-{shipment['shipment_id'][-3:]}",
                "invoice_number": f"INV-10K-{i:05d}"
            })
    return shipments


@pytest.fixture
def shipments_100k(sample_shipments_data) -> List[Dict[str, Any]]:
    """Generate 100,000 shipments for performance testing (memory test)."""
    # Note: Only use this for specific performance tests
    # May consume significant memory
    shipments = []
    for i in range(20000):  # 20,000 × 5 = 100,000
        for shipment in sample_shipments_data:
            shipments.append({
                **shipment,
                "shipment_id": f"SHIP-100K-{i:06d}-{shipment['shipment_id'][-3:]}",
                "invoice_number": f"INV-100K-{i:06d}"
            })
    return shipments


# ============================================================================
# FILE FIXTURES
# ============================================================================

@pytest.fixture
def sample_csv_file(tmp_path, sample_shipments_data):
    """Create temporary CSV file with sample shipments."""
    csv_path = tmp_path / "test_shipments.csv"
    df = pd.DataFrame(sample_shipments_data)
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_excel_file(tmp_path, sample_shipments_data):
    """Create temporary Excel file with sample shipments."""
    excel_path = tmp_path / "test_shipments.xlsx"
    df = pd.DataFrame(sample_shipments_data)
    df.to_excel(excel_path, index=False, engine='openpyxl')
    return str(excel_path)


@pytest.fixture
def sample_json_file(tmp_path, sample_shipments_data):
    """Create temporary JSON file with sample shipments."""
    json_path = tmp_path / "test_shipments.json"
    with open(json_path, 'w') as f:
        json.dump(sample_shipments_data, f, indent=2)
    return str(json_path)


@pytest.fixture
def cn_codes_file(tmp_path, sample_cn_codes):
    """Create temporary CN codes database file."""
    cn_path = tmp_path / "cn_codes.json"
    with open(cn_path, 'w') as f:
        json.dump(sample_cn_codes, f, indent=2)
    return str(cn_path)


@pytest.fixture
def suppliers_file(tmp_path, sample_suppliers):
    """Create temporary suppliers database file."""
    import yaml
    suppliers_path = tmp_path / "suppliers.yaml"
    with open(suppliers_path, 'w') as f:
        yaml.dump(sample_suppliers, f)
    return str(suppliers_path)


# ============================================================================
# VALIDATION & ASSERTION HELPERS
# ============================================================================

@pytest.fixture
def assert_valid_cbam_report():
    """Helper to validate CBAM report structure."""
    def _validate(report: Dict[str, Any]) -> bool:
        """Validate report has required CBAM structure."""
        required_sections = [
            "report_metadata",
            "emissions_summary",
            "detailed_goods"
        ]

        for section in required_sections:
            assert section in report, f"Missing required section: {section}"

        # Validate metadata
        metadata = report["report_metadata"]
        assert "report_id" in metadata
        assert "generated_at" in metadata

        # Validate emissions summary
        summary = report["emissions_summary"]
        assert "total_embedded_emissions_tco2" in summary
        assert isinstance(summary["total_embedded_emissions_tco2"], (int, float, Decimal))

        # Validate detailed goods
        assert isinstance(report["detailed_goods"], list)

        return True

    return _validate


@pytest.fixture
def assert_zero_hallucination():
    """Helper to verify zero hallucination guarantee."""
    def _validate(report: Dict[str, Any]) -> bool:
        """Validate zero hallucination architecture."""
        # Check all emissions calculations are deterministic
        for good in report.get("detailed_goods", []):
            calc_method = good.get("calculation_method", "")
            assert calc_method in ["deterministic", "actual_data", "default_values"], \
                f"Non-deterministic calculation detected: {calc_method}"

        # Check provenance if present
        if "provenance" in report:
            provenance = report["provenance"]
            if "reproducibility" in provenance:
                repro = provenance["reproducibility"]
                assert repro.get("zero_hallucination") == True, \
                    "Zero hallucination guarantee not confirmed"

        return True

    return _validate


@pytest.fixture
def assert_performance_target():
    """Helper to verify performance targets."""
    def _validate(duration: float, record_count: int, target_ms_per_record: float = 3.0) -> bool:
        """Validate performance meets target."""
        ms_per_record = (duration * 1000) / record_count
        assert ms_per_record < target_ms_per_record, \
            f"Performance below target: {ms_per_record:.2f}ms/record (target: {target_ms_per_record}ms)"
        return True

    return _validate


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def cbam_config():
    """Standard CBAM configuration for testing."""
    return {
        "importer": {
            "name": "Test Company BV",
            "country": "NL",
            "eori_number": "NL123456789012"
        },
        "declarant": {
            "name": "Test User",
            "position": "CBAM Manager"
        },
        "reporting_period": {
            "quarter": "Q3",
            "year": 2024
        }
    }


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup happens via tmp_path fixture


# ============================================================================
# CUSTOM MARKERS
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    markers = {
        "unit": "Unit tests (fast, isolated)",
        "integration": "Integration tests (multiple components)",
        "performance": "Performance and benchmark tests",
        "compliance": "CBAM compliance tests (CRITICAL)",
        "security": "Security-related tests",
        "slow": "Slow-running tests",
        "smoke": "Smoke tests (quick validation)",
        "e2e": "End-to-end tests (full pipeline)"
    }

    for marker, description in markers.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


# ============================================================================
# TEST COLLECTION HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)

        # Auto-mark slow tests
        if "large" in item.nodeid or "100k" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Auto-mark compliance tests
        if "zero_hallucination" in item.nodeid or "compliance" in item.nodeid:
            item.add_marker(pytest.mark.compliance)


# ============================================================================
# REPORTING HOOKS
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_run_info(request):
    """Capture test run information."""
    start_time = DeterministicClock.now()

    yield

    end_time = DeterministicClock.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n\n{'='*60}")
    print(f"Test Execution Summary")
    print(f"{'='*60}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"{'='*60}\n")
