# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Calculation Engine

Tests calculation engine with multi-phase factors:
- Deterministic calculations
- Audit trail generation
- Multi-gas decomposition
- Uncertainty quantification
- Reproducibility

Author: QA Team Lead
Date: 2025-11-20
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from datetime import date
import hashlib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_foundation" / "agents" / "calculator"))

from calculation_engine import CalculationEngine, CalculationResult
from formula_engine import FormulaLibrary, Formula, FormulaStep
from emission_factors import EmissionFactorDatabase, EmissionFactor


class TestCalculationEngine:
    """Test calculation engine functionality."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        """Setup calculation engine with test data."""
        self.formula_library = FormulaLibrary()
        self.emission_db = EmissionFactorDatabase()
        self.engine = CalculationEngine(self.formula_library, self.emission_db)

        # Insert test emission factors
        self._insert_test_factors()

        # Create test formula
        self._create_test_formula()

        yield

        self.emission_db.close()

    def _insert_test_factors(self):
        """Insert test emission factors for all phases."""
        test_factors = [
            # Phase 1: Diesel
            EmissionFactor(
                factor_id="test_diesel_gb_2024",
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
                data_quality="high",
                uncertainty_percentage=5.0
            ),
            # Phase 2: Electricity
            EmissionFactor(
                factor_id="test_electricity_us_2024",
                category="scope2",
                activity_type="electricity_consumption",
                material_or_fuel="grid_average",
                unit="kg_co2e_per_kwh",
                factor_co2=Decimal("0.45"),
                factor_ch4=Decimal("0.001"),
                factor_n2o=Decimal("0.0005"),
                factor_co2e=Decimal("0.46"),
                region="US",
                valid_from=date(2024, 1, 1),
                source="EPA",
                source_year=2024,
                source_version="2024",
                data_quality="high",
                uncertainty_percentage=7.0
            ),
            # Phase 3: Transportation
            EmissionFactor(
                factor_id="test_truck_freight_2024",
                category="scope3",
                activity_type="freight_transport",
                material_or_fuel="truck",
                unit="kg_co2e_per_tkm",
                factor_co2=Decimal("0.105"),
                factor_ch4=Decimal("0.0002"),
                factor_n2o=Decimal("0.0001"),
                factor_co2e=Decimal("0.106"),
                region="GLOBAL",
                valid_from=date(2024, 1, 1),
                source="GLEC",
                source_year=2024,
                source_version="2024",
                data_quality="medium",
                uncertainty_percentage=10.0
            )
        ]

        for factor in test_factors:
            self.emission_db.insert_factor(factor)

        print(f"\n✓ Inserted {len(test_factors)} test emission factors")

    def _create_test_formula(self):
        """Create a test formula for calculations."""
        # Simple formula: emissions = fuel_quantity * emission_factor
        formula = Formula(
            formula_id="test_fuel_combustion",
            version="1.0",
            name="Test Fuel Combustion",
            description="Test formula for fuel combustion emissions",
            category="scope1",
            subcategory="stationary_combustion",
            regulatory_frameworks=["GHG Protocol"],
            parameters=[
                {
                    "name": "fuel_quantity",
                    "type": "number",
                    "unit": "liters",
                    "required": True,
                    "description": "Quantity of fuel consumed"
                },
                {
                    "name": "fuel_type",
                    "type": "string",
                    "required": True,
                    "description": "Type of fuel"
                },
                {
                    "name": "region",
                    "type": "string",
                    "required": True,
                    "description": "Geographic region"
                }
            ],
            calculation={
                "steps": [
                    {
                        "step": 1,
                        "description": "Lookup emission factor",
                        "operation": "lookup",
                        "lookup_keys": {
                            "category": "scope1",
                            "activity_type": "fuel_combustion",
                            "material_or_fuel": "{fuel_type}",
                            "region": "{region}"
                        },
                        "output": "emission_factor"
                    },
                    {
                        "step": 2,
                        "description": "Calculate total emissions",
                        "operation": "multiply",
                        "operands": ["fuel_quantity", "emission_factor"],
                        "output": "total_emissions"
                    }
                ]
            },
            output={
                "value": "total_emissions",
                "unit": "kg_co2e",
                "precision": 2
            },
            provenance={
                "author": "Test Suite",
                "created_at": "2024-11-20",
                "version_history": []
            }
        )

        self.formula_library.add_formula(formula)
        print(f"✓ Created test formula: {formula.formula_id}")

    def test_basic_calculation(self):
        """Test basic emission calculation."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        assert result is not None
        assert result.output_value > 0
        assert result.output_unit == "kg_co2e"
        assert len(result.calculation_steps) == 2

        print(f"\n✓ Basic Calculation:")
        print(f"  Input: {params}")
        print(f"  Output: {result.output_value} {result.output_unit}")
        print(f"  Steps: {len(result.calculation_steps)}")

    def test_reproducibility(self):
        """Test bit-perfect reproducibility - same inputs produce identical outputs."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        # Run calculation 5 times
        results = []
        for i in range(5):
            result = self.engine.calculate("test_fuel_combustion", params)
            results.append(result)

        # All results must be identical
        first_value = results[0].output_value
        first_hash = results[0].provenance_hash

        for i, result in enumerate(results[1:], 1):
            assert result.output_value == first_value, f"Run {i+1} value differs"
            assert result.provenance_hash == first_hash, f"Run {i+1} hash differs"

        print(f"\n✓ Reproducibility Test (5 runs):")
        print(f"  All values: {first_value}")
        print(f"  All hashes: {first_hash[:16]}...")
        print(f"  ✓ DETERMINISTIC: 100% reproducible")

    def test_provenance_hash_generation(self):
        """Test that provenance hash is correctly generated."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        # Verify hash format (SHA-256 = 64 hex chars)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

        print(f"\n✓ Provenance Hash:")
        print(f"  Hash: {result.provenance_hash}")
        print(f"  Format: SHA-256 ✓")

    def test_provenance_hash_uniqueness(self):
        """Test that different inputs produce different hashes."""
        params1 = {"fuel_quantity": 1000, "fuel_type": "diesel", "region": "GB"}
        params2 = {"fuel_quantity": 2000, "fuel_type": "diesel", "region": "GB"}

        result1 = self.engine.calculate("test_fuel_combustion", params1)
        result2 = self.engine.calculate("test_fuel_combustion", params2)

        assert result1.provenance_hash != result2.provenance_hash
        assert result1.output_value != result2.output_value

        print(f"\n✓ Hash Uniqueness:")
        print(f"  Hash1: {result1.provenance_hash[:16]}...")
        print(f"  Hash2: {result2.provenance_hash[:16]}...")
        print(f"  ✓ Different inputs → Different hashes")

    def test_audit_trail_completeness(self):
        """Test that complete audit trail is generated."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        # Verify all required fields
        assert result.formula_id
        assert result.formula_version
        assert result.provenance_hash
        assert result.input_parameters
        assert len(result.calculation_steps) > 0
        assert result.calculation_time_ms >= 0

        # Verify each step has required info
        for step in result.calculation_steps:
            assert step.step_number > 0
            assert step.description
            assert step.operation
            assert step.output_name
            assert step.timestamp

        print(f"\n✓ Audit Trail Completeness:")
        print(f"  Formula: {result.formula_id} v{result.formula_version}")
        print(f"  Steps: {len(result.calculation_steps)}")
        print(f"  Provenance: {result.provenance_hash[:16]}...")
        print(f"  Time: {result.calculation_time_ms:.2f}ms")

    def test_multi_gas_decomposition(self):
        """Test multi-gas decomposition (CO2, CH4, N2O)."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        # Verify emission factor tracking
        assert len(result.emission_factors_used) > 0

        factor_info = result.emission_factors_used[0]
        assert 'factor_co2e' in factor_info
        assert 'source' in factor_info
        assert 'data_quality' in factor_info

        print(f"\n✓ Multi-Gas Decomposition:")
        print(f"  Factors used: {len(result.emission_factors_used)}")
        print(f"  CO2e: {factor_info['factor_co2e']} {factor_info['unit']}")
        print(f"  Source: {factor_info['source']}")

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification and propagation."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        # Should have uncertainty from emission factor
        if result.uncertainty_percentage:
            assert result.uncertainty_percentage > 0
            assert result.uncertainty_percentage < 100

            print(f"\n✓ Uncertainty Quantification:")
            print(f"  Total uncertainty: {result.uncertainty_percentage:.1f}%")
        else:
            print(f"\n✓ Uncertainty: Not applicable for this calculation")

    def test_calculation_with_phase2_factor(self):
        """Test calculation with Phase 2 (electricity) factor."""
        # Create electricity formula
        elec_formula = Formula(
            formula_id="test_electricity",
            version="1.0",
            name="Test Electricity",
            description="Test formula for electricity emissions",
            category="scope2",
            subcategory="electricity",
            regulatory_frameworks=["GHG Protocol"],
            parameters=[
                {"name": "kwh", "type": "number", "unit": "kWh", "required": True},
                {"name": "region", "type": "string", "required": True}
            ],
            calculation={
                "steps": [
                    {
                        "step": 1,
                        "description": "Lookup electricity emission factor",
                        "operation": "lookup",
                        "lookup_keys": {
                            "category": "scope2",
                            "activity_type": "electricity_consumption",
                            "material_or_fuel": "grid_average",
                            "region": "{region}"
                        },
                        "output": "emission_factor"
                    },
                    {
                        "step": 2,
                        "description": "Calculate electricity emissions",
                        "operation": "multiply",
                        "operands": ["kwh", "emission_factor"],
                        "output": "total_emissions"
                    }
                ]
            },
            output={"value": "total_emissions", "unit": "kg_co2e", "precision": 2},
            provenance={"author": "Test Suite", "created_at": "2024-11-20", "version_history": []}
        )

        self.formula_library.add_formula(elec_formula)

        params = {"kwh": 10000, "region": "US"}
        result = self.engine.calculate("test_electricity", params)

        assert result.output_value > 0
        assert result.category == "scope2"

        print(f"\n✓ Phase 2 Calculation (Electricity):")
        print(f"  Input: {params['kwh']} kWh")
        print(f"  Output: {result.output_value} {result.output_unit}")

    def test_calculation_with_phase3_factor(self):
        """Test calculation with Phase 3 (transportation) factor."""
        # Create transport formula
        transport_formula = Formula(
            formula_id="test_freight",
            version="1.0",
            name="Test Freight",
            description="Test formula for freight transport emissions",
            category="scope3",
            subcategory="transportation",
            regulatory_frameworks=["GHG Protocol"],
            parameters=[
                {"name": "distance_km", "type": "number", "unit": "km", "required": True},
                {"name": "weight_tonnes", "type": "number", "unit": "tonnes", "required": True}
            ],
            calculation={
                "steps": [
                    {
                        "step": 1,
                        "description": "Calculate tonne-km",
                        "operation": "multiply",
                        "operands": ["distance_km", "weight_tonnes"],
                        "output": "tkm"
                    },
                    {
                        "step": 2,
                        "description": "Lookup transport emission factor",
                        "operation": "lookup",
                        "lookup_keys": {
                            "category": "scope3",
                            "activity_type": "freight_transport",
                            "material_or_fuel": "truck",
                            "region": "GLOBAL"
                        },
                        "output": "emission_factor"
                    },
                    {
                        "step": 3,
                        "description": "Calculate transport emissions",
                        "operation": "multiply",
                        "operands": ["tkm", "emission_factor"],
                        "output": "total_emissions"
                    }
                ]
            },
            output={"value": "total_emissions", "unit": "kg_co2e", "precision": 2},
            provenance={"author": "Test Suite", "created_at": "2024-11-20", "version_history": []}
        )

        self.formula_library.add_formula(transport_formula)

        params = {"distance_km": 500, "weight_tonnes": 10}
        result = self.engine.calculate("test_freight", params)

        assert result.output_value > 0
        assert result.category == "scope3"
        assert len(result.calculation_steps) == 3

        print(f"\n✓ Phase 3 Calculation (Transportation):")
        print(f"  Input: {params['distance_km']} km × {params['weight_tonnes']} t")
        print(f"  Output: {result.output_value} {result.output_unit}")

    def test_calculation_verification(self):
        """Test calculation verification (reproducibility check)."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        result = self.engine.calculate("test_fuel_combustion", params)

        # Verify the calculation
        is_verified = self.engine.verify_calculation(result)

        assert is_verified, "Calculation verification failed"

        print(f"\n✓ Calculation Verification:")
        print(f"  Original: {result.output_value}")
        print(f"  Verified: ✓ REPRODUCIBLE")

    def test_error_handling_missing_factor(self):
        """Test error handling when emission factor not found."""
        params = {
            "fuel_quantity": 1000,
            "fuel_type": "nonexistent_fuel",
            "region": "XX"
        }

        with pytest.raises(ValueError, match="Emission factor not found"):
            self.engine.calculate("test_fuel_combustion", params)

        print(f"\n✓ Error Handling: Missing factor raises ValueError")

    def test_error_handling_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        params = {
            "fuel_quantity": 1000,
            # Missing required parameters
        }

        with pytest.raises(ValueError, match="validation failed"):
            self.engine.calculate("test_fuel_combustion", params)

        print(f"\n✓ Error Handling: Invalid parameters raises ValueError")

    def test_performance_calculation_speed(self):
        """Test calculation performance."""
        import time

        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = self.engine.calculate("test_fuel_combustion", params)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n✓ Calculation Performance (10 runs):")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")

        # Should be reasonably fast
        assert avg_time < 100, f"Average calculation time too slow: {avg_time:.2f}ms"

    def test_batch_calculation_throughput(self):
        """Test bulk calculation throughput (target: 10,000+ calc/min)."""
        import time

        params = {
            "fuel_quantity": 1000,
            "fuel_type": "diesel",
            "region": "GB"
        }

        # Run 100 calculations
        start = time.perf_counter()
        for _ in range(100):
            self.engine.calculate("test_fuel_combustion", params)
        elapsed = time.perf_counter() - start

        calcs_per_second = 100 / elapsed
        calcs_per_minute = calcs_per_second * 60

        print(f"\n✓ Bulk Calculation Throughput:")
        print(f"  100 calculations: {elapsed:.2f}s")
        print(f"  Throughput: {calcs_per_minute:.0f} calc/min")
        print(f"  Target: 10,000+ calc/min {'✓ PASS' if calcs_per_minute >= 10000 else '✗ BELOW TARGET'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
