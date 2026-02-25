"""
Test suite for AverageDataCalculatorEngine - AGENT-MRV-014

This module tests the AverageDataCalculatorEngine for the Purchased Goods & Services Agent.
Tests cover average-data (physical) calculations, unit conversions, transport adders,
material resolution, multi-material BOM, DQI scoring, aggregation, and uncertainty.

Coverage target: 85%+
Test count: 60+ tests
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional

from greenlang.purchased_goods_services.average_data_calculator import (
    AverageDataCalculatorEngine,
    PhysicalRecord,
    PhysicalCalculationResult,
    MaterialAllocation,
    TransportAdder,
    DQIScore,
    UncertaintyRange
)
from greenlang.purchased_goods_services.procurement_database import (
    ProcurementDatabaseEngine
)


class TestAverageDataCalculatorSingleton:
    """Test singleton pattern for AverageDataCalculatorEngine."""

    def test_singleton_creation(self):
        """Test that engine can be created."""
        engine = AverageDataCalculatorEngine()
        assert engine is not None
        assert isinstance(engine, AverageDataCalculatorEngine)

    def test_singleton_identity(self):
        """Test that multiple calls return same instance."""
        engine1 = AverageDataCalculatorEngine()
        engine2 = AverageDataCalculatorEngine()
        assert engine1 is engine2

    def test_singleton_reset(self):
        """Test that singleton can be reset."""
        engine1 = AverageDataCalculatorEngine()
        AverageDataCalculatorEngine._instance = None
        engine2 = AverageDataCalculatorEngine()
        assert engine1 is not engine2


class TestSingleCalculation:
    """Test single average-data calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_basic_kg_calculation(self, engine):
        """Test basic calculation with kg input."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-001",
            description="Hot rolled steel"
        )
        result = engine.calculate(record)

        assert result is not None
        assert result.total_emissions > Decimal("0")
        assert result.quantity_kg == Decimal("1000.00")
        assert result.material_key == "steel_hot_rolled"
        assert result.calculation_method == "average_data_physical"

    def test_tonnes_conversion(self, engine):
        """Test tonnes to kg conversion."""
        record = PhysicalRecord(
            quantity=Decimal("5.0"),
            unit="tonnes",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-002",
            description="Steel in tonnes"
        )
        result = engine.calculate(record)

        # 5 tonnes = 5000 kg
        assert result.quantity_kg == Decimal("5000.00")
        assert result.original_quantity == Decimal("5.0")
        assert result.original_unit == "tonnes"

    def test_grams_conversion(self, engine):
        """Test grams to kg conversion."""
        record = PhysicalRecord(
            quantity=Decimal("500000.0"),
            unit="g",
            material_key="aluminum_primary",
            region="GLOBAL",
            supplier_id="SUP-003",
            description="Aluminum in grams"
        )
        result = engine.calculate(record)

        # 500000 g = 500 kg
        assert result.quantity_kg == Decimal("500.00")

    def test_material_key_resolution(self, engine):
        """Test material key resolution from database."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-004",
            description="Steel resolution test"
        )
        result = engine.calculate(record)

        assert result.material_key == "steel_hot_rolled"
        assert result.emission_factor > Decimal("0")
        assert result.material_category == "metals"

    def test_waste_factor_application(self, engine):
        """Test waste/scrap factor application."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-005",
            waste_factor=Decimal("0.05")  # 5% waste
        )
        result = engine.calculate(record)

        # Waste increases quantity: 1000 * 1.05 = 1050 kg
        expected_quantity = Decimal("1000.00") * (Decimal("1") + Decimal("0.05"))
        assert result.quantity_kg == expected_quantity

    def test_zero_quantity(self, engine):
        """Test handling of zero quantity."""
        record = PhysicalRecord(
            quantity=Decimal("0.0"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-006"
        )
        result = engine.calculate(record)

        assert result.total_emissions == Decimal("0")

    def test_missing_material_key(self, engine):
        """Test handling of missing material key."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key=None,
            region="GLOBAL",
            supplier_id="SUP-007"
        )
        with pytest.raises(ValueError, match="Material key required"):
            engine.calculate(record)

    def test_negative_quantity(self, engine):
        """Test handling of negative quantity (credit/return)."""
        record = PhysicalRecord(
            quantity=Decimal("-500.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-008"
        )
        result = engine.calculate(record)

        # Negative quantity should produce negative emissions
        assert result.total_emissions < Decimal("0")

    def test_very_large_quantity(self, engine):
        """Test handling of very large quantities."""
        record = PhysicalRecord(
            quantity=Decimal("1000000.0"),  # 1000 tonnes
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-009"
        )
        result = engine.calculate(record)

        assert result.total_emissions > Decimal("0")
        # Should handle large numbers without overflow
        assert result.total_emissions < Decimal("100000000")

    def test_high_precision_calculation(self, engine):
        """Test calculation maintains high precision."""
        record = PhysicalRecord(
            quantity=Decimal("1234.56"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-010"
        )
        result = engine.calculate(record)

        # Result should preserve precision
        assert result.total_emissions.as_tuple().exponent <= -2


class TestUnitConversion:
    """Test unit conversion for various mass and volume units."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_kg_identity(self, engine):
        """Test kg to kg conversion is identity."""
        converted = engine.convert_to_kg(
            quantity=Decimal("1000.00"),
            unit="kg"
        )
        assert converted == Decimal("1000.00")

    def test_tonnes_to_kg(self, engine):
        """Test tonnes to kg conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("5.0"),
            unit="tonnes"
        )
        assert converted == Decimal("5000.00")

    def test_grams_to_kg(self, engine):
        """Test grams to kg conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("5000.0"),
            unit="g"
        )
        assert converted == Decimal("5.00")

    def test_pounds_to_kg(self, engine):
        """Test pounds to kg conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("2204.62"),
            unit="lb"
        )
        # 2204.62 lb ≈ 1000 kg
        tolerance = Decimal("0.1")
        assert abs(converted - Decimal("1000.00")) < tolerance

    def test_liters_to_kg(self, engine):
        """Test liters to kg conversion (requires density)."""
        converted = engine.convert_to_kg(
            quantity=Decimal("1000.0"),
            unit="liters",
            density=Decimal("0.85")  # Diesel density
        )
        # 1000 L * 0.85 kg/L = 850 kg
        assert converted == Decimal("850.00")

    def test_cubic_meters_to_kg(self, engine):
        """Test cubic meters to kg conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("1.0"),
            unit="m3",
            density=Decimal("2400.0")  # Concrete density
        )
        # 1 m³ * 2400 kg/m³ = 2400 kg
        assert converted == Decimal("2400.00")

    def test_gallons_to_kg(self, engine):
        """Test gallons to kg conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("264.172"),
            unit="gallons",
            density=Decimal("1.0")  # Water density
        )
        # 264.172 gal ≈ 1000 L ≈ 1000 kg (for water)
        tolerance = Decimal("1.0")
        assert abs(converted - Decimal("1000.00")) < tolerance

    def test_missing_unit(self, engine):
        """Test handling of missing unit."""
        with pytest.raises(ValueError, match="Unit required"):
            engine.convert_to_kg(
                quantity=Decimal("1000.00"),
                unit=None
            )

    def test_unknown_unit(self, engine):
        """Test handling of unknown unit."""
        with pytest.raises(ValueError, match="Unknown unit|not supported"):
            engine.convert_to_kg(
                quantity=Decimal("1000.00"),
                unit="furlong"
            )

    def test_volume_without_density(self, engine):
        """Test volume conversion without density raises error."""
        with pytest.raises(ValueError, match="Density required"):
            engine.convert_to_kg(
                quantity=Decimal("1000.0"),
                unit="liters"
            )

    def test_metric_ton_conversion(self, engine):
        """Test metric ton (same as tonne)."""
        converted = engine.convert_to_kg(
            quantity=Decimal("10.0"),
            unit="metric_ton"
        )
        assert converted == Decimal("10000.00")

    def test_short_ton_conversion(self, engine):
        """Test US short ton conversion."""
        converted = engine.convert_to_kg(
            quantity=Decimal("1.0"),
            unit="short_ton"
        )
        # 1 short ton = 907.185 kg
        tolerance = Decimal("0.01")
        assert abs(converted - Decimal("907.185")) < tolerance


class TestTransportAdder:
    """Test transport emission adders for purchased goods."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_road_transport(self, engine):
        """Test road transport emission adder."""
        transport_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="road",
            distance_km=Decimal("500.0")
        )
        assert transport_emissions > Decimal("0")

    def test_rail_transport(self, engine):
        """Test rail transport emission adder."""
        transport_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="rail",
            distance_km=Decimal("500.0")
        )
        assert transport_emissions > Decimal("0")
        # Rail typically lower than road
        road_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="road",
            distance_km=Decimal("500.0")
        )
        assert transport_emissions < road_emissions

    def test_sea_freight(self, engine):
        """Test sea freight emission adder."""
        transport_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="sea",
            distance_km=Decimal("10000.0")
        )
        assert transport_emissions > Decimal("0")
        # Sea freight has lowest emissions per tkm
        road_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="road",
            distance_km=Decimal("10000.0")
        )
        assert transport_emissions < road_emissions

    def test_air_freight(self, engine):
        """Test air freight emission adder."""
        transport_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="air",
            distance_km=Decimal("5000.0")
        )
        assert transport_emissions > Decimal("0")
        # Air freight has highest emissions
        road_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="road",
            distance_km=Decimal("5000.0")
        )
        assert transport_emissions > road_emissions

    def test_zero_distance(self, engine):
        """Test zero transport distance."""
        transport_emissions = engine.calculate_transport_emissions(
            quantity_kg=Decimal("1000.00"),
            transport_mode="road",
            distance_km=Decimal("0.0")
        )
        assert transport_emissions == Decimal("0")

    def test_unknown_transport_mode(self, engine):
        """Test handling of unknown transport mode."""
        with pytest.raises(ValueError, match="Unknown transport mode|not supported"):
            engine.calculate_transport_emissions(
                quantity_kg=Decimal("1000.00"),
                transport_mode="teleportation",
                distance_km=Decimal("500.0")
            )


class TestMaterialResolution:
    """Test material key resolution from various inputs."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_resolve_by_category(self, engine):
        """Test material resolution by category."""
        materials = engine.get_materials_by_category("metals")
        assert len(materials) > 0
        assert all(m.category == "metals" for m in materials)

    def test_resolve_by_keyword(self, engine):
        """Test material resolution by description keyword."""
        materials = engine.search_materials("steel")
        assert len(materials) > 0
        assert all("steel" in m.description.lower() for m in materials)

    def test_resolve_explicit_key(self, engine):
        """Test material resolution with explicit key."""
        material = engine.resolve_material(
            material_key="steel_hot_rolled"
        )
        assert material is not None
        assert material.material_key == "steel_hot_rolled"

    def test_resolve_from_description(self, engine):
        """Test material resolution from description."""
        material = engine.resolve_material_from_description(
            description="Hot rolled steel sheets"
        )
        assert material is not None
        assert "steel" in material.material_key.lower()

    def test_resolve_ambiguous_description(self, engine):
        """Test handling of ambiguous description."""
        # "Metal" could match multiple materials
        materials = engine.search_materials("metal")
        # Should return multiple matches
        assert len(materials) > 1

    def test_resolve_nonexistent_material(self, engine):
        """Test handling of non-existent material."""
        material = engine.resolve_material(
            material_key="unobtainium"
        )
        assert material is None


class TestMultiMaterial:
    """Test multi-material Bill of Materials (BOM) allocation."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_simple_bom(self, engine):
        """Test simple BOM with two materials."""
        allocations = [
            MaterialAllocation(
                material_key="steel_hot_rolled",
                quantity_kg=Decimal("800.0"),
                percentage=Decimal("0.8")
            ),
            MaterialAllocation(
                material_key="aluminum_primary",
                quantity_kg=Decimal("200.0"),
                percentage=Decimal("0.2")
            )
        ]
        result = engine.calculate_bom(
            allocations=allocations,
            total_quantity_kg=Decimal("1000.0"),
            region="GLOBAL",
            supplier_id="SUP-020"
        )

        assert result is not None
        assert result.total_emissions > Decimal("0")
        assert len(result.material_breakdown) == 2

    def test_bom_allocation_percentages(self, engine):
        """Test BOM allocation by percentages."""
        allocations = [
            MaterialAllocation(
                material_key="steel_hot_rolled",
                percentage=Decimal("0.7")
            ),
            MaterialAllocation(
                material_key="plastic_pet",
                percentage=Decimal("0.3")
            )
        ]
        result = engine.calculate_bom(
            allocations=allocations,
            total_quantity_kg=Decimal("1000.0"),
            region="GLOBAL",
            supplier_id="SUP-021"
        )

        # Should allocate 700 kg steel, 300 kg plastic
        steel_breakdown = [b for b in result.material_breakdown if b.material_key == "steel_hot_rolled"][0]
        plastic_breakdown = [b for b in result.material_breakdown if b.material_key == "plastic_pet"][0]

        assert steel_breakdown.quantity_kg == Decimal("700.0")
        assert plastic_breakdown.quantity_kg == Decimal("300.0")

    def test_bom_percentages_sum_to_one(self, engine):
        """Test validation that BOM percentages sum to 1.0."""
        allocations = [
            MaterialAllocation(
                material_key="steel_hot_rolled",
                percentage=Decimal("0.5")
            ),
            MaterialAllocation(
                material_key="aluminum_primary",
                percentage=Decimal("0.3")
            )
            # Only sums to 0.8
        ]
        with pytest.raises(ValueError, match="percentages must sum to 1|100%"):
            engine.calculate_bom(
                allocations=allocations,
                total_quantity_kg=Decimal("1000.0"),
                region="GLOBAL",
                supplier_id="SUP-022"
            )

    def test_empty_bom(self, engine):
        """Test handling of empty BOM."""
        with pytest.raises(ValueError, match="At least one material required"):
            engine.calculate_bom(
                allocations=[],
                total_quantity_kg=Decimal("1000.0"),
                region="GLOBAL",
                supplier_id="SUP-023"
            )


class TestDQIScoring:
    """Test Data Quality Indicator (DQI) scoring for average data."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_default_dqi_scores(self, engine):
        """Test default DQI scores for average-data method."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-030"
        )
        result = engine.calculate(record)

        assert result.dqi is not None
        # Average-data has better quality than spend-based (Tier 2)
        assert result.dqi.tier == 2
        assert 1.5 <= result.dqi.composite_score <= 3.0

    def test_source_year_impact(self, engine):
        """Test that data source year impacts temporal score."""
        record_recent = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-031",
            source_year=2022
        )
        record_old = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-032",
            source_year=2010
        )

        result_recent = engine.calculate(record_recent)
        result_old = engine.calculate(record_old)

        # Older data should have worse temporal score
        assert result_old.dqi.temporal_representativeness >= \
               result_recent.dqi.temporal_representativeness

    def test_regional_data_quality(self, engine):
        """Test regional data quality impact."""
        record_global = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-033"
        )
        record_specific = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="CHINA",
            supplier_id="SUP-034"
        )

        result_global = engine.calculate(record_global)
        result_specific = engine.calculate(record_specific)

        # Regional-specific should have better geographical score
        assert result_specific.dqi.geographical_representativeness <= \
               result_global.dqi.geographical_representativeness

    def test_dqi_dimensions(self, engine):
        """Test individual DQI dimensions."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-035"
        )
        result = engine.calculate(record)

        dqi = result.dqi
        # All dimensions should be scored 1-5
        assert 1 <= dqi.technological_representativeness <= 5
        assert 1 <= dqi.geographical_representativeness <= 5
        assert 1 <= dqi.temporal_representativeness <= 5
        assert 1 <= dqi.completeness <= 5
        assert 1 <= dqi.reliability <= 5

    def test_composite_dqi_calculation(self, engine):
        """Test composite DQI calculation formula."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-036"
        )
        result = engine.calculate(record)

        dqi = result.dqi
        # Composite = average of 5 dimensions
        expected = (
            dqi.technological_representativeness +
            dqi.geographical_representativeness +
            dqi.temporal_representativeness +
            dqi.completeness +
            dqi.reliability
        ) / Decimal("5")

        tolerance = Decimal("0.01")
        assert abs(dqi.composite_score - expected) < tolerance


class TestBatchCalculation:
    """Test batch calculation of multiple physical records."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_multiple_items(self, engine):
        """Test batch calculation of multiple items."""
        records = [
            PhysicalRecord(
                quantity=Decimal("1000.00"),
                unit="kg",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-040"
            ),
            PhysicalRecord(
                quantity=Decimal("500.00"),
                unit="kg",
                material_key="aluminum_primary",
                region="GLOBAL",
                supplier_id="SUP-041"
            ),
            PhysicalRecord(
                quantity=Decimal("2000.00"),
                unit="kg",
                material_key="concrete_ready_mix",
                region="GLOBAL",
                supplier_id="SUP-042"
            )
        ]
        results = engine.calculate_batch(records)

        assert len(results) == 3
        assert all(r.total_emissions > Decimal("0") for r in results)

    def test_mixed_materials(self, engine):
        """Test batch calculation with diverse materials."""
        materials = ["steel_hot_rolled", "aluminum_primary", "plastic_pet", "concrete_ready_mix"]
        records = [
            PhysicalRecord(
                quantity=Decimal("1000.00"),
                unit="kg",
                material_key=material,
                region="GLOBAL",
                supplier_id=f"SUP-{i+50}"
            )
            for i, material in enumerate(materials)
        ]
        results = engine.calculate_batch(records)

        assert len(results) == 4
        # Emission factors should vary by material
        efs = [r.emission_factor for r in results]
        assert len(set(efs)) > 1

    def test_mixed_units(self, engine):
        """Test batch calculation with mixed units."""
        records = [
            PhysicalRecord(
                quantity=Decimal("1000.00"),
                unit="kg",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-054"
            ),
            PhysicalRecord(
                quantity=Decimal("1.0"),
                unit="tonnes",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-055"
            ),
            PhysicalRecord(
                quantity=Decimal("1000000.0"),
                unit="g",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-056"
            )
        ]
        results = engine.calculate_batch(records)

        # All should convert to same kg quantity
        assert all(r.quantity_kg == Decimal("1000.00") for r in results)

    def test_batch_performance(self, engine):
        """Test batch calculation performance (100+ records)."""
        records = [
            PhysicalRecord(
                quantity=Decimal("1000.00"),
                unit="kg",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id=f"SUP-PERF-{i}"
            )
            for i in range(100)
        ]

        import time
        start = time.time()
        results = engine.calculate_batch(records)
        elapsed = time.time() - start

        assert len(results) == 100
        # Should complete in reasonable time (< 5s for 100 records)
        assert elapsed < 5.0

    def test_empty_batch(self, engine):
        """Test handling of empty batch."""
        results = engine.calculate_batch([])
        assert results == []


class TestAggregation:
    """Test aggregation and summarization of physical calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    @pytest.fixture
    def sample_records(self):
        """Create sample physical records."""
        return [
            PhysicalRecord(
                quantity=Decimal("1000.00"),
                unit="kg",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-060"
            ),
            PhysicalRecord(
                quantity=Decimal("2000.00"),
                unit="kg",
                material_key="steel_hot_rolled",
                region="GLOBAL",
                supplier_id="SUP-061"
            ),
            PhysicalRecord(
                quantity=Decimal("500.00"),
                unit="kg",
                material_key="aluminum_primary",
                region="GLOBAL",
                supplier_id="SUP-062"
            )
        ]

    def test_aggregate_by_material(self, engine, sample_records):
        """Test aggregation by material type."""
        results = engine.calculate_batch(sample_records)
        aggregation = engine.aggregate_by_material(results)

        assert "steel_hot_rolled" in aggregation
        assert "aluminum_primary" in aggregation

        # Steel: 1000 + 2000 = 3000 kg
        assert aggregation["steel_hot_rolled"].total_quantity_kg == Decimal("3000.00")
        # Aluminum: 500 kg
        assert aggregation["aluminum_primary"].total_quantity_kg == Decimal("500.00")

    def test_aggregate_totals(self, engine, sample_records):
        """Test total aggregation across all records."""
        results = engine.calculate_batch(sample_records)
        totals = engine.aggregate_totals(results)

        # Total quantity: 1000 + 2000 + 500 = 3500 kg
        assert totals.total_quantity_kg == Decimal("3500.00")
        assert totals.total_emissions > Decimal("0")
        assert totals.record_count == 3

    def test_coverage_calculation(self, engine, sample_records):
        """Test data coverage calculation."""
        results = engine.calculate_batch(sample_records)
        coverage = engine.calculate_coverage(results)

        assert 0.0 <= coverage.mass_coverage <= 1.0
        assert coverage.records_with_data == 3
        assert coverage.total_records == 3

    def test_material_breakdown(self, engine, sample_records):
        """Test material-level breakdown."""
        results = engine.calculate_batch(sample_records)
        breakdown = engine.get_material_breakdown(results)

        assert len(breakdown) == 2  # Two materials
        # Each material should have quantity and emissions
        for material in breakdown:
            assert material.material_key is not None
            assert material.quantity_kg > Decimal("0")
            assert material.emissions > Decimal("0")


class TestUncertainty:
    """Test uncertainty quantification for average-data calculations."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_base_uncertainty_range(self, engine):
        """Test base uncertainty range for average-data method."""
        record = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-070"
        )
        result = engine.calculate(record)

        assert result.uncertainty is not None
        # Average-data typically has ±30% uncertainty
        lower_bound = result.total_emissions * Decimal("0.7")
        upper_bound = result.total_emissions * Decimal("1.3")

        assert result.uncertainty.lower_bound <= result.total_emissions
        assert result.uncertainty.upper_bound >= result.total_emissions
        tolerance = Decimal("5.0")
        assert abs(result.uncertainty.lower_bound - lower_bound) < tolerance
        assert abs(result.uncertainty.upper_bound - upper_bound) < tolerance

    def test_material_specific_uncertainty(self, engine):
        """Test that uncertainty varies by material."""
        record_steel = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-071"
        )
        record_plastic = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="plastic_pet",
            region="GLOBAL",
            supplier_id="SUP-072"
        )

        result_steel = engine.calculate(record_steel)
        result_plastic = engine.calculate(record_plastic)

        # Uncertainty ranges should differ by material
        assert result_steel.uncertainty.relative_uncertainty != \
               result_plastic.uncertainty.relative_uncertainty

    def test_regional_uncertainty_variation(self, engine):
        """Test uncertainty variation by region."""
        record_global = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="GLOBAL",
            supplier_id="SUP-073"
        )
        record_china = PhysicalRecord(
            quantity=Decimal("1000.00"),
            unit="kg",
            material_key="steel_hot_rolled",
            region="CHINA",
            supplier_id="SUP-074"
        )

        result_global = engine.calculate(record_global)
        result_china = engine.calculate(record_china)

        # Regional-specific should have lower uncertainty
        assert result_china.uncertainty.relative_uncertainty <= \
               result_global.uncertainty.relative_uncertainty


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AverageDataCalculatorEngine()

    def test_health_check_returns_dict(self, engine):
        """Test health check returns valid response."""
        health = engine.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_all_fields(self, engine):
        """Test health check contains all expected fields."""
        health = engine.health_check()
        expected_fields = [
            "status",
            "database_connection",
            "physical_efs_loaded",
            "material_count",
            "transport_modes_available",
            "last_calculation_time_ms"
        ]
        for field in expected_fields:
            assert field in health
