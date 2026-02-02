# -*- coding: utf-8 -*-
"""
Integration Tests: Supplier Data Prioritization
================================================

Tests supplier actual emissions data prioritization over defaults:
- Supplier actual emissions prioritized
- Fallback to defaults when supplier data missing
- Supplier data quality scoring
- Supplier profile linking accuracy

Target: Maturity score +1 point (data quality management)
Version: 1.0.0
Author: GL-TestEngineer
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.emissions_calculator_agent_v2 import EmissionsCalculatorAgent_v2, CalculatorInput


# ============================================================================
# Supplier Data Priority Tests
# ============================================================================

@pytest.mark.integration
class TestSupplierDataPriority:
    """Test supplier actual emissions are prioritized over defaults."""

    def test_supplier_actual_emissions_prioritized(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test supplier actual emissions are used when available.

        Priority order:
        1. Supplier actual data (highest priority)
        2. EU default values
        3. Error (no data available)
        """
        # Create calculator agent with supplier data
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Test shipment with supplier that has actual emissions
        shipment = {
            "shipment_id": "SHIP-001",
            "cn_code": "72071100",
            "product_group": "iron_steel",
            "origin_iso": "CN",
            "net_mass_kg": 10000,  # 10 tonnes
            "supplier_id": "SUP-CN-001",
            "has_actual_emissions": "YES"
        }

        # Calculate emissions
        calculation, warnings = calculator.calculate_emissions(shipment)

        # Verify actual supplier data was used
        assert calculation is not None, "Calculation should succeed"
        assert calculation.calculation_method == "actual_data", \
            f"Should use actual_data, got: {calculation.calculation_method}"
        assert "SUP-CN-001" in calculation.emission_factor_source, \
            "Should cite supplier as source"

        # Verify actual emissions values used (from supplier profile)
        # Supplier SUP-CN-001 should have specific emission factors
        assert calculation.data_quality in ["high", "medium"], \
            "Actual supplier data should have high/medium quality"

        print("\n[Supplier Priority Test]")
        print(f"  ✓ Used actual supplier data for SUP-CN-001")
        print(f"  ✓ Calculation method: {calculation.calculation_method}")
        print(f"  ✓ Emission factor source: {calculation.emission_factor_source}")
        print(f"  ✓ Data quality: {calculation.data_quality}")

    def test_fallback_to_defaults_when_supplier_data_missing(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test fallback to EU defaults when supplier has no actual data.

        Critical for ensuring calculations always complete.
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Test shipment with supplier that has NO actual emissions
        shipment = {
            "shipment_id": "SHIP-002",
            "cn_code": "72071100",
            "product_group": "iron_steel",
            "origin_iso": "TR",
            "net_mass_kg": 10000,  # 10 tonnes
            "supplier_id": "SUP-TR-002",
            "has_actual_emissions": "NO"  # No actual data
        }

        # Calculate emissions
        calculation, warnings = calculator.calculate_emissions(shipment)

        # Verify default values were used as fallback
        assert calculation is not None, "Calculation should succeed with fallback"
        assert calculation.calculation_method == "default_values", \
            f"Should use default_values, got: {calculation.calculation_method}"
        assert "Default" in calculation.emission_factor_source or "EU" in calculation.emission_factor_source, \
            "Should cite EU defaults as source"

        print("\n[Fallback Test]")
        print(f"  ✓ Fell back to EU defaults for SUP-TR-002")
        print(f"  ✓ Calculation method: {calculation.calculation_method}")
        print(f"  ✓ Emission factor source: {calculation.emission_factor_source}")

    def test_supplier_data_vs_defaults_comparison(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test comparison between supplier actual and default emissions.

        Validates that supplier actuals differ from defaults (proves prioritization).
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Same product, same mass, different supplier data availability
        shipment_with_actuals = {
            "shipment_id": "SHIP-003A",
            "cn_code": "72071100",
            "product_group": "iron_steel",
            "origin_iso": "CN",
            "net_mass_kg": 10000,
            "supplier_id": "SUP-CN-001",
            "has_actual_emissions": "YES"
        }

        shipment_with_defaults = {
            "shipment_id": "SHIP-003B",
            "cn_code": "72071100",
            "product_group": "iron_steel",
            "origin_iso": "CN",
            "net_mass_kg": 10000,
            "supplier_id": "SUP-CN-999",  # Supplier without actuals
            "has_actual_emissions": "NO"
        }

        calc_actual, _ = calculator.calculate_emissions(shipment_with_actuals)
        calc_default, _ = calculator.calculate_emissions(shipment_with_defaults)

        # Both calculations should succeed
        assert calc_actual is not None
        assert calc_default is not None

        # Methods should differ
        assert calc_actual.calculation_method == "actual_data"
        assert calc_default.calculation_method == "default_values"

        # Emission factors should differ (proves different sources)
        # Note: They might be same value, but sources should differ
        assert calc_actual.emission_factor_source != calc_default.emission_factor_source, \
            "Sources should differ between actual and default"

        print("\n[Comparison Test]")
        print(f"  Actual data emissions: {calc_actual.total_emissions_tco2:.3f} tCO2")
        print(f"  Default data emissions: {calc_default.total_emissions_tco2:.3f} tCO2")
        print(f"  Difference: {abs(calc_actual.total_emissions_tco2 - calc_default.total_emissions_tco2):.3f} tCO2")


# ============================================================================
# Supplier Profile Linking Tests
# ============================================================================

@pytest.mark.integration
class TestSupplierProfileLinking:
    """Test supplier profile linking accuracy."""

    def test_supplier_profile_linked_correctly(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test supplier profiles are correctly linked to shipments.

        Validates supplier_id matching and data retrieval.
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Verify suppliers loaded
        assert len(calculator.suppliers) > 0, "Should have loaded supplier data"

        # Test known supplier lookup
        supplier_id = "SUP-CN-001"
        assert supplier_id in calculator.suppliers, f"Supplier {supplier_id} should be loaded"

        supplier_profile = calculator.suppliers[supplier_id]

        # Verify profile structure
        assert "supplier_id" in supplier_profile
        assert "company_name" in supplier_profile
        assert "actual_emissions_available" in supplier_profile

        print("\n[Supplier Linking Test]")
        print(f"  ✓ Loaded {len(calculator.suppliers)} supplier profiles")
        print(f"  ✓ Found profile for {supplier_id}: {supplier_profile.get('company_name')}")

    def test_multiple_suppliers_same_product(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test different suppliers for same product use correct data.

        Critical: Must not mix supplier data.
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Three shipments, same product, different suppliers
        shipments = [
            {
                "shipment_id": "SHIP-004A",
                "cn_code": "72071100",
                "product_group": "iron_steel",
                "origin_iso": "CN",
                "net_mass_kg": 10000,
                "supplier_id": "SUP-CN-001",
                "has_actual_emissions": "YES"
            },
            {
                "shipment_id": "SHIP-004B",
                "cn_code": "72071100",
                "product_group": "iron_steel",
                "origin_iso": "CN",
                "net_mass_kg": 10000,
                "supplier_id": "SUP-CN-002",
                "has_actual_emissions": "YES"
            },
            {
                "shipment_id": "SHIP-004C",
                "cn_code": "72071100",
                "product_group": "iron_steel",
                "origin_iso": "CN",
                "net_mass_kg": 10000,
                "supplier_id": "SUP-CN-003",
                "has_actual_emissions": "NO"
            }
        ]

        calculations = []
        for shipment in shipments:
            calc, _ = calculator.calculate_emissions(shipment)
            calculations.append(calc)

        # Verify calculations completed
        assert all(c is not None for c in calculations), "All calculations should succeed"

        # Verify different suppliers have different data sources
        sources = [c.emission_factor_source for c in calculations]
        methods = [c.calculation_method for c in calculations]

        print("\n[Multiple Suppliers Test]")
        for i, calc in enumerate(calculations):
            print(f"  Shipment {i+1}: {methods[i]} - {sources[i]}")

        # First two should use actual data, third should use defaults
        assert methods[0] == "actual_data"
        assert methods[1] == "actual_data"
        assert methods[2] == "default_values"

        # Sources should be supplier-specific for actuals
        assert "SUP-CN-001" in sources[0] or "actual" in sources[0].lower()
        assert "SUP-CN-002" in sources[1] or "actual" in sources[1].lower()


# ============================================================================
# Data Quality Scoring Tests
# ============================================================================

@pytest.mark.integration
class TestSupplierDataQuality:
    """Test supplier data quality scoring."""

    def test_data_quality_scoring(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test data quality scores are assigned correctly.

        Quality levels: high > medium > low
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        # Test shipments with different data quality
        test_cases = [
            {
                "shipment": {
                    "shipment_id": "SHIP-Q1",
                    "cn_code": "72071100",
                    "product_group": "iron_steel",
                    "origin_iso": "CN",
                    "net_mass_kg": 10000,
                    "supplier_id": "SUP-CN-001",
                    "has_actual_emissions": "YES"
                },
                "expected_quality": ["high", "medium"],  # Should be high or medium for actuals
                "description": "Supplier with actual EPD data"
            },
            {
                "shipment": {
                    "shipment_id": "SHIP-Q2",
                    "cn_code": "72071100",
                    "product_group": "iron_steel",
                    "origin_iso": "CN",
                    "net_mass_kg": 10000,
                    "supplier_id": "SUP-CN-999",
                    "has_actual_emissions": "NO"
                },
                "expected_quality": ["medium"],  # EU defaults are medium quality
                "description": "Supplier using EU defaults"
            }
        ]

        print("\n[Data Quality Test]")

        for test_case in test_cases:
            calc, _ = calculator.calculate_emissions(test_case["shipment"])

            assert calc is not None, f"Calculation failed for {test_case['description']}"
            assert calc.data_quality in test_case["expected_quality"], \
                f"Quality {calc.data_quality} not in expected {test_case['expected_quality']}"

            print(f"  {test_case['description']}: {calc.data_quality} ✓")

    def test_data_quality_affects_reporting(
        self,
        suppliers_with_actuals,
        cbam_rules_path
    ):
        """
        Test data quality is tracked in emissions calculations.

        Must be included in provenance for audit trail.
        """
        calculator = EmissionsCalculatorAgent_v2(
            suppliers_path=suppliers_with_actuals,
            cbam_rules_path=cbam_rules_path
        )

        shipment = {
            "shipment_id": "SHIP-Q3",
            "cn_code": "72071100",
            "product_group": "iron_steel",
            "origin_iso": "CN",
            "net_mass_kg": 10000,
            "supplier_id": "SUP-CN-001",
            "has_actual_emissions": "YES"
        }

        calc, _ = calculator.calculate_emissions(shipment)

        # Verify data quality is included in calculation output
        assert hasattr(calc, 'data_quality'), "Calculation should include data_quality field"
        assert calc.data_quality is not None, "Data quality should not be None"
        assert calc.data_quality in ["high", "medium", "low"], \
            f"Invalid data quality value: {calc.data_quality}"

        # Verify provenance tracking
        assert hasattr(calc, 'emission_factor_source'), "Should track emission factor source"
        assert hasattr(calc, 'calculation_method'), "Should track calculation method"

        print("\n[Provenance Test]")
        print(f"  ✓ Data quality tracked: {calc.data_quality}")
        print(f"  ✓ Source tracked: {calc.emission_factor_source}")
        print(f"  ✓ Method tracked: {calc.calculation_method}")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def suppliers_with_actuals(tmp_path):
    """Create suppliers YAML file with actual emissions data."""
    suppliers_data = {
        "suppliers": [
            {
                "supplier_id": "SUP-CN-001",
                "company_name": "China Steel Manufacturing Co.",
                "country": "CN",
                "actual_emissions_available": True,
                "actual_emissions_data": {
                    "direct_emissions_tco2_per_ton": 1.8,
                    "indirect_emissions_tco2_per_ton": 0.4,
                    "total_emissions_tco2_per_ton": 2.2,
                    "data_quality": "high",
                    "verification_status": "third_party_verified",
                    "last_updated": "2025-01-15"
                }
            },
            {
                "supplier_id": "SUP-CN-002",
                "company_name": "Shanghai Iron Works Ltd.",
                "country": "CN",
                "actual_emissions_available": True,
                "actual_emissions_data": {
                    "direct_emissions_tco2_per_ton": 2.0,
                    "indirect_emissions_tco2_per_ton": 0.3,
                    "total_emissions_tco2_per_ton": 2.3,
                    "data_quality": "medium",
                    "verification_status": "self_reported",
                    "last_updated": "2024-12-01"
                }
            },
            {
                "supplier_id": "SUP-CN-003",
                "company_name": "Beijing Metal Products",
                "country": "CN",
                "actual_emissions_available": False
            },
            {
                "supplier_id": "SUP-TR-002",
                "company_name": "Turkey Steel Corp",
                "country": "TR",
                "actual_emissions_available": False
            },
            {
                "supplier_id": "SUP-CN-999",
                "company_name": "Generic Supplier",
                "country": "CN",
                "actual_emissions_available": False
            }
        ]
    }

    suppliers_path = tmp_path / "test_suppliers.yaml"
    with open(suppliers_path, 'w') as f:
        yaml.dump(suppliers_data, f)

    return str(suppliers_path)


@pytest.fixture
def cbam_rules_path():
    """Path to CBAM rules file."""
    return "rules/cbam_rules.yaml"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
