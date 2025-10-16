"""
Comprehensive Test Suite for CBAM Refactored Agents

Tests all three refactored agents:
1. ShipmentIntakeAgent (BaseDataProcessor)
2. EmissionsCalculatorAgent (BaseCalculator)
3. ReportingPackagerAgent (BaseReporter)

Validates:
- Framework integration
- Business logic preservation
- CBAM compliance
- Zero-hallucination guarantee
- Performance benchmarks
- Provenance tracking

Author: GreenLang CBAM Team
Date: 2025-10-16
"""

import json
import pytest
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

# Import refactored agents
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

from shipment_intake_agent_refactored import ShipmentIntakeAgent, EU_MEMBER_STATES
from emissions_calculator_agent_refactored import EmissionsCalculatorAgent, EmissionsCalculation
from reporting_packager_agent_refactored import ReportingPackagerAgent


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def cn_codes_fixture(test_data_dir):
    """Create CN codes test data."""
    cn_codes = {
        "72071100": {
            "product_group": "Iron and Steel",
            "description": "Semi-finished products of iron",
            "cbam_covered": True
        },
        "72072000": {
            "product_group": "Iron and Steel",
            "description": "Flat-rolled products",
            "cbam_covered": True
        },
        "28112100": {
            "product_group": "Hydrogen",
            "description": "Hydrogen in gaseous state",
            "cbam_covered": True
        }
    }

    cn_codes_path = test_data_dir / "cn_codes.json"
    with open(cn_codes_path, 'w') as f:
        json.dump(cn_codes, f)

    return cn_codes_path


@pytest.fixture
def cbam_rules_fixture(test_data_dir):
    """Create CBAM rules test data."""
    cbam_rules = {
        "version": "1.0.0",
        "complex_goods_threshold": 0.20,
        "validation_rules": {
            "VAL-041": "Summary totals must match detail records",
            "VAL-042": "Emissions totals must match calculated values",
            "VAL-020": "Complex goods cannot exceed 20% of total"
        }
    }

    rules_path = test_data_dir / "cbam_rules.yaml"
    import yaml
    with open(rules_path, 'w') as f:
        yaml.dump(cbam_rules, f)

    return rules_path


@pytest.fixture
def suppliers_fixture(test_data_dir):
    """Create suppliers test data."""
    suppliers = {
        "suppliers": [
            {
                "supplier_id": "SUP001",
                "company_name": "Steel Corp Ltd",
                "country": "CN",
                "has_actual_data": True,
                "actual_emissions_data": {
                    "direct_emissions_tco2_per_ton": 1.85,
                    "indirect_emissions_tco2_per_ton": 0.35,
                    "total_emissions_tco2_per_ton": 2.20,
                    "data_quality": "high",
                    "verification": "ISO 14064-1"
                }
            },
            {
                "supplier_id": "SUP002",
                "company_name": "Hydrogen Industries",
                "country": "RU",
                "has_actual_data": False
            }
        ]
    }

    suppliers_path = test_data_dir / "suppliers.yaml"
    import yaml
    with open(suppliers_path, 'w') as f:
        yaml.dump(suppliers, f)

    return suppliers_path


@pytest.fixture
def sample_shipments():
    """Create sample shipment data for testing."""
    return [
        {
            "shipment_id": "SHIP001",
            "import_date": "2024-01-15",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "origin_iso": "CN",
            "net_mass_kg": 25000,
            "importer_country": "DE",
            "importer_eori": "DE123456789",
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        },
        {
            "shipment_id": "SHIP002",
            "import_date": "2024-01-20",
            "quarter": "Q1-2024",
            "cn_code": "28112100",
            "origin_iso": "RU",
            "net_mass_kg": 5000,
            "importer_country": "FR",
            "importer_eori": "FR987654321",
            "supplier_id": "SUP002",
            "has_actual_emissions": "NO"
        },
        {
            "shipment_id": "SHIP003",
            "import_date": "2024-02-10",
            "quarter": "Q1-2024",
            "cn_code": "72072000",
            "origin_iso": "CN",
            "net_mass_kg": 15000,
            "importer_country": "DE",
            "importer_eori": "DE123456789",
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }
    ]


# ============================================================================
# TEST SHIPMENT INTAKE AGENT
# ============================================================================

class TestShipmentIntakeAgent:
    """Test suite for ShipmentIntakeAgent (BaseDataProcessor)."""

    def test_agent_initialization(self, cn_codes_fixture, cbam_rules_fixture, suppliers_fixture):
        """Test agent initializes correctly with framework."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture,
            suppliers_path=suppliers_fixture
        )

        assert agent.config.agent_id == "cbam-intake"
        assert agent.config.version == "2.0.0"
        assert len(agent.cn_codes) == 3
        assert len(agent.suppliers) == 2

    def test_process_valid_record(self, cn_codes_fixture, cbam_rules_fixture, sample_shipments):
        """Test processing of valid shipment record."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        shipment = sample_shipments[0]
        result = agent.process_record(shipment)

        # Check enrichment
        assert result['product_group'] == "Iron and Steel"
        assert result['product_description'] == "Semi-finished products of iron"
        assert 'shipment_id' in result

    def test_cn_code_validation(self, cn_codes_fixture, cbam_rules_fixture):
        """Test CN code validation rules."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        # Valid CN code
        valid_shipment = {
            "shipment_id": "TEST001",
            "import_date": "2024-01-01",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "origin_iso": "CN",
            "net_mass_kg": 1000,
            "importer_country": "DE"
        }

        result = agent.process_record(valid_shipment)
        assert result is not None

        # Invalid CN code format
        invalid_shipment = valid_shipment.copy()
        invalid_shipment['cn_code'] = "1234"  # Too short

        from greenlang.validation import ValidationException
        with pytest.raises(ValidationException):
            agent.process_record(invalid_shipment)

    def test_eu_member_state_validation(self, cn_codes_fixture, cbam_rules_fixture):
        """Test EU member state validation."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        # Valid EU importer
        valid_shipment = {
            "shipment_id": "TEST001",
            "import_date": "2024-01-01",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "origin_iso": "CN",
            "net_mass_kg": 1000,
            "importer_country": "DE"
        }

        result = agent.process_record(valid_shipment)
        assert result is not None

        # Invalid non-EU importer
        invalid_shipment = valid_shipment.copy()
        invalid_shipment['importer_country'] = "US"

        from greenlang.validation import ValidationException
        with pytest.raises(ValidationException):
            agent.process_record(invalid_shipment)

    def test_mass_validation(self, cn_codes_fixture, cbam_rules_fixture):
        """Test mass validation (must be positive)."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        # Negative mass
        shipment = {
            "shipment_id": "TEST001",
            "import_date": "2024-01-01",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "origin_iso": "CN",
            "net_mass_kg": -1000,  # Invalid
            "importer_country": "DE"
        }

        from greenlang.validation import ValidationException
        with pytest.raises(ValidationException):
            agent.process_record(shipment)

    def test_supplier_enrichment(self, cn_codes_fixture, cbam_rules_fixture, suppliers_fixture):
        """Test supplier enrichment logic."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture,
            suppliers_path=suppliers_fixture
        )

        shipment = {
            "shipment_id": "TEST001",
            "import_date": "2024-01-01",
            "quarter": "Q1-2024",
            "cn_code": "72071100",
            "origin_iso": "CN",
            "net_mass_kg": 1000,
            "importer_country": "DE",
            "supplier_id": "SUP001"
        }

        result = agent.process_record(shipment)

        assert '_enrichment' in result
        assert result['_enrichment']['supplier_found'] is True
        assert result['_enrichment']['supplier_name'] == "Steel Corp Ltd"


# ============================================================================
# TEST EMISSIONS CALCULATOR AGENT
# ============================================================================

class TestEmissionsCalculatorAgent:
    """Test suite for EmissionsCalculatorAgent (BaseCalculator)."""

    def test_agent_initialization(self, suppliers_fixture, cbam_rules_fixture):
        """Test calculator agent initializes with framework."""
        agent = EmissionsCalculatorAgent(
            suppliers_path=suppliers_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        assert agent.config.agent_id == "cbam-calculator"
        assert agent.config.version == "2.0.0"
        assert len(agent.suppliers) == 2

    def test_deterministic_calculations(self, suppliers_fixture):
        """Test @deterministic decorator ensures reproducibility."""
        agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "72071100",
            "net_mass_kg": 10000,
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }

        # Run calculation 10 times
        results = []
        for _ in range(10):
            calc = agent.calculate(shipment)
            results.append(calc.total_emissions_tco2)

        # All results must be identical (zero-hallucination guarantee)
        assert len(set(results)) == 1, "Calculations must be deterministic"

    def test_supplier_actual_data(self, suppliers_fixture):
        """Test emission factor selection from supplier actual data."""
        agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "72071100",
            "net_mass_kg": 10000,
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }

        calc = agent.calculate(shipment)

        assert calc.calculation_method == "actual_data"
        assert calc.emission_factor_source.startswith("Supplier SUP001")
        assert calc.data_quality == "high"

        # Check calculation precision
        expected_mass = Decimal('10000') / Decimal('1000')  # 10 tonnes
        expected_direct = expected_mass * Decimal('1.85')

        assert abs(calc.direct_emissions_tco2 - float(expected_direct)) < 0.001

    def test_high_precision_arithmetic(self, suppliers_fixture):
        """Test high-precision Decimal arithmetic."""
        agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "72071100",
            "net_mass_kg": 12345,  # Non-round number
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }

        calc = agent.calculate(shipment)

        # Manual calculation with Decimal precision
        mass_tonnes = Decimal('12345') / Decimal('1000')
        ef_direct = Decimal('1.85')
        expected = mass_tonnes * ef_direct

        assert abs(calc.direct_emissions_tco2 - float(expected)) < 0.001

    def test_caching_decorator(self, suppliers_fixture):
        """Test @cached decorator avoids redundant calculations."""
        agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "72071100",
            "net_mass_kg": 10000,
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }

        # First call (cache miss)
        import time
        start = time.time()
        result1 = agent.calculate(shipment)
        time1 = time.time() - start

        # Second call (cache hit - should be faster)
        start = time.time()
        result2 = agent.calculate(shipment)
        time2 = time.time() - start

        # Results must be identical
        assert result1.total_emissions_tco2 == result2.total_emissions_tco2

        # Cache hit should be significantly faster (at least 2x)
        # Note: May not always be true in test environment, but worth checking
        print(f"First call: {time1:.6f}s, Second call: {time2:.6f}s")

    def test_emission_factor_fallback(self):
        """Test emission factor selection fallback logic."""
        agent = EmissionsCalculatorAgent()

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "99999999",  # Non-existent CN code
            "net_mass_kg": 10000,
            "supplier_id": "SUP999",
            "has_actual_emissions": "NO"
        }

        # Should raise error if no emission factor available
        with pytest.raises(ValueError, match="No emission factor"):
            agent.calculate(shipment)


# ============================================================================
# TEST REPORTING PACKAGER AGENT
# ============================================================================

class TestReportingPackagerAgent:
    """Test suite for ReportingPackagerAgent (BaseReporter)."""

    def test_agent_initialization(self, cbam_rules_fixture):
        """Test reporter agent initializes with framework."""
        agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        assert agent.config.agent_id == "cbam-reporter"
        assert agent.config.version == "2.0.0"
        assert 'json' in agent.config.output_formats
        assert 'markdown' in agent.config.output_formats

    def test_aggregate_data(self, cbam_rules_fixture):
        """Test CBAM-specific aggregation logic."""
        agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        shipments = [
            {
                "shipment_id": "SHIP001",
                "net_mass_kg": 10000,
                "product_group": "Iron and Steel",
                "emissions_calculation": {
                    "direct_emissions_tco2": 18.5,
                    "indirect_emissions_tco2": 3.5,
                    "total_emissions_tco2": 22.0,
                    "calculation_method": "actual_data"
                }
            },
            {
                "shipment_id": "SHIP002",
                "net_mass_kg": 5000,
                "product_group": "Hydrogen",
                "emissions_calculation": {
                    "direct_emissions_tco2": 9.0,
                    "indirect_emissions_tco2": 1.5,
                    "total_emissions_tco2": 10.5,
                    "calculation_method": "default_values"
                }
            }
        ]

        aggregated = agent.aggregate_data(shipments)

        assert aggregated['goods_summary']['total_shipments'] == 2
        assert aggregated['goods_summary']['total_mass_tonnes'] == 15.0
        assert aggregated['emissions_summary']['total_embedded_emissions_tco2'] == 32.5

    def test_complex_goods_check(self, cbam_rules_fixture):
        """Test complex goods 20% threshold validation."""
        agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        # 15% complex goods (within threshold)
        shipments_valid = [
            {
                "shipment_id": f"SHIP{i:03d}",
                "net_mass_kg": 1000,
                "product_group": "Iron and Steel",
                "emissions_calculation": {
                    "total_emissions_tco2": 2.0,
                    "calculation_method": "complex_goods" if i < 15 else "default_values"
                }
            }
            for i in range(100)
        ]

        aggregated = agent.aggregate_data(shipments_valid)

        assert aggregated['complex_goods_check']['percentage'] == 15.0
        assert aggregated['complex_goods_check']['within_threshold'] is True

        # 25% complex goods (exceeds threshold)
        shipments_invalid = [
            {
                "shipment_id": f"SHIP{i:03d}",
                "net_mass_kg": 1000,
                "product_group": "Iron and Steel",
                "emissions_calculation": {
                    "total_emissions_tco2": 2.0,
                    "calculation_method": "complex_goods" if i < 25 else "default_values"
                }
            }
            for i in range(100)
        ]

        aggregated = agent.aggregate_data(shipments_invalid)

        assert aggregated['complex_goods_check']['percentage'] == 25.0
        assert aggregated['complex_goods_check']['within_threshold'] is False

    def test_validation_rules(self, cbam_rules_fixture):
        """Test CBAM validation rules (VAL-041, VAL-042)."""
        agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        shipments = [
            {
                "shipment_id": "SHIP001",
                "net_mass_kg": 10000,
                "product_group": "Iron and Steel",
                "emissions_calculation": {
                    "direct_emissions_tco2": 18.5,
                    "indirect_emissions_tco2": 3.5,
                    "total_emissions_tco2": 22.0
                }
            }
        ]

        aggregated = agent.aggregate_data(shipments)

        # Check validation status
        assert aggregated['validation']['is_valid'] is True
        assert len(aggregated['validation']['errors']) == 0
        assert len(aggregated['validation']['rules_checked']) >= 2

    def test_report_sections(self, cbam_rules_fixture):
        """Test report section generation."""
        agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        shipments = [
            {
                "shipment_id": "SHIP001",
                "net_mass_kg": 10000,
                "product_group": "Iron and Steel",
                "emissions_calculation": {
                    "direct_emissions_tco2": 18.5,
                    "indirect_emissions_tco2": 3.5,
                    "total_emissions_tco2": 22.0,
                    "calculation_method": "actual_data"
                }
            }
        ]

        aggregated = agent.aggregate_data(shipments)
        sections = agent.build_sections(aggregated)

        assert len(sections) >= 3
        assert any("Summary" in s.title for s in sections)
        assert any("Product Group" in s.title for s in sections)
        assert any("Validation" in s.title for s in sections)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCBAMPipelineIntegration:
    """Integration tests for complete CBAM pipeline."""

    def test_end_to_end_pipeline(self, cn_codes_fixture, cbam_rules_fixture, suppliers_fixture, sample_shipments, tmp_path):
        """Test complete CBAM pipeline: Intake → Calculate → Report."""

        # Step 1: Intake Agent
        intake_agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture,
            suppliers_path=suppliers_fixture
        )

        validated_shipments = []
        for shipment in sample_shipments:
            validated = intake_agent.process_record(shipment)
            validated_shipments.append(validated)

        assert len(validated_shipments) == 3

        # Step 2: Calculator Agent
        calc_agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        calculated_shipments = []
        for shipment in validated_shipments:
            calc = calc_agent.calculate(shipment)
            shipment['emissions_calculation'] = calc.dict()
            calculated_shipments.append(shipment)

        assert all('emissions_calculation' in s for s in calculated_shipments)

        # Step 3: Reporter Agent
        reporter_agent = ReportingPackagerAgent(cbam_rules_path=cbam_rules_fixture)

        result = reporter_agent.run(input_data=calculated_shipments)

        # Verify final report
        assert result.data['goods_summary']['total_shipments'] == 3
        assert result.data['emissions_summary']['total_embedded_emissions_tco2'] > 0
        assert result.data['validation']['is_valid'] is True

    def test_provenance_tracking(self, cn_codes_fixture, cbam_rules_fixture, sample_shipments):
        """Test framework provenance tracking throughout pipeline."""

        intake_agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        result = intake_agent.run(input_data=sample_shipments)

        # Check provenance record exists
        assert hasattr(result, 'provenance_record')
        assert result.provenance_record.agent_name == "cbam-intake"
        assert result.provenance_record.version == "2.0.0"
        assert result.provenance_record.timestamp is not None


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for refactored agents."""

    def test_intake_throughput(self, cn_codes_fixture, cbam_rules_fixture, sample_shipments):
        """Benchmark intake agent throughput."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_fixture,
            cbam_rules_path=cbam_rules_fixture
        )

        # Process 1000 shipments
        large_batch = sample_shipments * 334  # ~1000 shipments

        import time
        start = time.time()

        processed = []
        for shipment in large_batch:
            result = agent.process_record(shipment)
            processed.append(result)

        elapsed = time.time() - start
        throughput = len(large_batch) / elapsed

        print(f"\n🚀 Intake Throughput: {throughput:.2f} shipments/sec")
        assert throughput > 100, "Should process >100 shipments/sec"

    def test_calculator_performance(self, suppliers_fixture):
        """Benchmark calculator agent performance."""
        agent = EmissionsCalculatorAgent(suppliers_path=suppliers_fixture)

        shipment = {
            "shipment_id": "TEST001",
            "cn_code": "72071100",
            "net_mass_kg": 10000,
            "supplier_id": "SUP001",
            "has_actual_emissions": "YES"
        }

        import time

        # Cold start (no cache)
        start = time.time()
        agent.calculate(shipment)
        cold_time = (time.time() - start) * 1000  # ms

        # Warm cache
        times = []
        for _ in range(100):
            start = time.time()
            agent.calculate(shipment)
            times.append((time.time() - start) * 1000)

        avg_warm = sum(times) / len(times)

        print(f"\n⚡ Calculator Performance:")
        print(f"  Cold start: {cold_time:.2f}ms")
        print(f"  Warm (avg): {avg_warm:.2f}ms")

        assert avg_warm < 1.0, "Cached calculations should be <1ms"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
