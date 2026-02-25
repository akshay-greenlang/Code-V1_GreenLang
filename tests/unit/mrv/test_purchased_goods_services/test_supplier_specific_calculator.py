"""
Unit tests for SupplierSpecificCalculatorEngine (AGENT-MRV-014).

Tests cover:
- Singleton pattern
- Product-level calculation
- Facility allocation
- EPD integration
- CDP integration
- Boundary verification
- Supplier engagement
- DQI scoring
- Batch processing and aggregation
- Health checks
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

try:
    from greenlang.purchased_goods_services.supplier_specific_calculator import (
        SupplierSpecificCalculatorEngine,
        SupplierSpecificInput,
        SupplierSpecificOutput,
        AllocationMethod,
        EngagementLevel,
        EPDData,
        CDPData,
        ProductEmissions,
    )
except ImportError:
    pytest.skip("SupplierSpecificCalculatorEngine not available", allow_module_level=True)


class TestSupplierSpecificCalculatorSingleton:
    """Test singleton pattern for SupplierSpecificCalculatorEngine."""

    def test_singleton_same_instance(self):
        """Test that get_instance returns same instance."""
        engine1 = SupplierSpecificCalculatorEngine.get_instance()
        engine2 = SupplierSpecificCalculatorEngine.get_instance()
        assert engine1 is engine2

    def test_singleton_thread_safe(self):
        """Test thread-safe singleton creation."""
        import threading
        instances = []

        def get_instance():
            instances.append(SupplierSpecificCalculatorEngine.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_singleton_reset(self):
        """Test singleton reset functionality."""
        engine1 = SupplierSpecificCalculatorEngine.get_instance()
        SupplierSpecificCalculatorEngine.reset_instance()
        engine2 = SupplierSpecificCalculatorEngine.get_instance()
        assert engine1 is not engine2


class TestProductLevelCalculation:
    """Test product-level emission calculations."""

    def test_basic_quantity_ef_multiplication(self):
        """Test basic quantity × emission factor calculation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("1000"),
            unit="kg",
            emission_factor=Decimal("2.5"),
            ef_unit="kg_co2e_per_kg",
        )

        result = engine.calculate_product_emissions(product)

        assert result["total_emissions"] == Decimal("2500.0")
        assert result["product_id"] == "P001"
        assert result["calculation_method"] == "product_ef"

    def test_epd_data_calculation(self):
        """Test calculation with EPD data."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Steel Rebar",
            declared_unit=Decimal("1000"),
            declared_unit_type="kg",
            gwp_total=Decimal("1800"),
            gwp_a1a3=Decimal("1500"),
            valid_until=date(2026, 12, 31),
            boundary="cradle_to_gate",
            verified=True,
        )

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("5000"),
            unit="kg",
            epd_data=epd,
        )

        result = engine.calculate_product_emissions(product)

        # 5000 kg / 1000 kg × 1800 kg CO2e = 9000 kg CO2e
        assert result["total_emissions"] == Decimal("9000.0")
        assert result["calculation_method"] == "epd"
        assert result["epd_id"] == "EPD001"

    def test_pcf_value_direct(self):
        """Test calculation with direct PCF value."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("100"),
            unit="units",
            pcf_value=Decimal("50.5"),
            pcf_unit="kg_co2e_per_unit",
        )

        result = engine.calculate_product_emissions(product)

        assert result["total_emissions"] == Decimal("5050.0")
        assert result["calculation_method"] == "pcf"

    def test_zero_quantity(self):
        """Test handling of zero quantity."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("0"),
            unit="kg",
            emission_factor=Decimal("2.5"),
        )

        result = engine.calculate_product_emissions(product)

        assert result["total_emissions"] == Decimal("0")

    def test_negative_quantity_raises_error(self):
        """Test that negative quantity raises error."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("-100"),
            unit="kg",
            emission_factor=Decimal("2.5"),
        )

        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            engine.calculate_product_emissions(product)

    def test_missing_emission_data_raises_error(self):
        """Test that missing emission data raises error."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("100"),
            unit="kg",
        )

        with pytest.raises(ValueError, match="No emission data provided"):
            engine.calculate_product_emissions(product)

    def test_unit_conversion_in_calculation(self):
        """Test unit conversion during calculation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("2"),
            unit="tonnes",
            emission_factor=Decimal("1.5"),
            ef_unit="kg_co2e_per_kg",
        )

        result = engine.calculate_product_emissions(product)

        # 2 tonnes × 1000 kg/tonne × 1.5 kg CO2e/kg = 3000 kg CO2e
        assert result["total_emissions"] == Decimal("3000.0")

    def test_multiple_products_batch(self):
        """Test batch calculation of multiple products."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        products = [
            ProductEmissions(
                product_id="P001",
                quantity=Decimal("100"),
                unit="kg",
                emission_factor=Decimal("2.0"),
            ),
            ProductEmissions(
                product_id="P002",
                quantity=Decimal("200"),
                unit="kg",
                emission_factor=Decimal("1.5"),
            ),
        ]

        results = engine.calculate_batch(products)

        assert len(results) == 2
        assert results[0]["total_emissions"] == Decimal("200.0")
        assert results[1]["total_emissions"] == Decimal("300.0")


class TestFacilityAllocation:
    """Test facility-level allocation methods."""

    def test_revenue_based_allocation(self):
        """Test revenue-based allocation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("10000")
        product_revenue = Decimal("50000")
        total_revenue = Decimal("200000")

        allocated = engine.allocate_facility_emissions(
            facility_emissions=facility_emissions,
            allocation_method=AllocationMethod.REVENUE,
            product_value=product_revenue,
            total_value=total_revenue,
        )

        # 10000 × (50000 / 200000) = 2500
        assert allocated == Decimal("2500.0")

    def test_mass_based_allocation(self):
        """Test mass-based allocation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("5000")
        product_mass = Decimal("1000")
        total_mass = Decimal("4000")

        allocated = engine.allocate_facility_emissions(
            facility_emissions=facility_emissions,
            allocation_method=AllocationMethod.MASS,
            product_value=product_mass,
            total_value=total_mass,
        )

        # 5000 × (1000 / 4000) = 1250
        assert allocated == Decimal("1250.0")

    def test_economic_allocation(self):
        """Test economic value allocation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("8000")
        product_value = Decimal("30000")
        total_value = Decimal("100000")

        allocated = engine.allocate_facility_emissions(
            facility_emissions=facility_emissions,
            allocation_method=AllocationMethod.ECONOMIC,
            product_value=product_value,
            total_value=total_value,
        )

        # 8000 × (30000 / 100000) = 2400
        assert allocated == Decimal("2400.0")

    def test_physical_unit_allocation(self):
        """Test physical unit allocation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("12000")
        product_units = Decimal("500")
        total_units = Decimal("2000")

        allocated = engine.allocate_facility_emissions(
            facility_emissions=facility_emissions,
            allocation_method=AllocationMethod.PHYSICAL_UNIT,
            product_value=product_units,
            total_value=total_units,
        )

        # 12000 × (500 / 2000) = 3000
        assert allocated == Decimal("3000.0")

    def test_equal_allocation(self):
        """Test equal allocation across products."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("6000")
        num_products = 3

        allocated = engine.allocate_facility_emissions(
            facility_emissions=facility_emissions,
            allocation_method=AllocationMethod.EQUAL,
            num_products=num_products,
        )

        # 6000 / 3 = 2000
        assert allocated == Decimal("2000.0")

    def test_zero_total_value_raises_error(self):
        """Test that zero total value raises error."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        with pytest.raises(ValueError, match="Total value cannot be zero"):
            engine.allocate_facility_emissions(
                facility_emissions=Decimal("1000"),
                allocation_method=AllocationMethod.REVENUE,
                product_value=Decimal("100"),
                total_value=Decimal("0"),
            )

    def test_product_value_exceeds_total(self):
        """Test that product value > total value raises error."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        with pytest.raises(ValueError, match="Product value cannot exceed total"):
            engine.allocate_facility_emissions(
                facility_emissions=Decimal("1000"),
                allocation_method=AllocationMethod.MASS,
                product_value=Decimal("1000"),
                total_value=Decimal("500"),
            )

    def test_compute_allocation_factor(self):
        """Test allocation factor computation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        factor = engine.compute_allocation_factor(
            product_value=Decimal("25000"),
            total_value=Decimal("100000"),
        )

        assert factor == Decimal("0.25")

    def test_allocation_factor_bounds(self):
        """Test allocation factor is between 0 and 1."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        factor = engine.compute_allocation_factor(
            product_value=Decimal("100"),
            total_value=Decimal("100"),
        )

        assert Decimal("0") <= factor <= Decimal("1")

    def test_multiple_allocation_methods_same_facility(self):
        """Test applying different allocation methods to same facility."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        facility_emissions = Decimal("10000")

        revenue_alloc = engine.allocate_facility_emissions(
            facility_emissions, AllocationMethod.REVENUE,
            Decimal("50000"), Decimal("200000")
        )
        mass_alloc = engine.allocate_facility_emissions(
            facility_emissions, AllocationMethod.MASS,
            Decimal("1000"), Decimal("4000")
        )

        assert revenue_alloc == Decimal("2500.0")
        assert mass_alloc == Decimal("2500.0")


class TestEPDIntegration:
    """Test Environmental Product Declaration integration."""

    def test_valid_epd_acceptance(self):
        """Test valid EPD is accepted."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Concrete",
            declared_unit=Decimal("1"),
            declared_unit_type="m3",
            gwp_total=Decimal("350"),
            gwp_a1a3=Decimal("300"),
            valid_until=date(2027, 6, 30),
            boundary="cradle_to_gate",
            verified=True,
        )

        is_valid = engine.validate_epd(epd)
        assert is_valid is True

    def test_expired_epd_rejection(self):
        """Test expired EPD is rejected."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Steel",
            declared_unit=Decimal("1000"),
            declared_unit_type="kg",
            gwp_total=Decimal("1800"),
            valid_until=date(2024, 12, 31),
            boundary="cradle_to_gate",
            verified=True,
        )

        is_valid = engine.validate_epd(epd)
        assert is_valid is False

    def test_invalid_boundary_flagged(self):
        """Test invalid system boundary is flagged."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Aluminum",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("12"),
            boundary="gate_to_gate",
            valid_until=date(2027, 12, 31),
            verified=True,
        )

        warnings = engine.check_epd_boundary(epd)
        assert len(warnings) > 0
        assert "gate_to_gate" in warnings[0].lower()

    def test_epd_gwp_a1a3_extraction(self):
        """Test extraction of GWP A1-A3 from EPD."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Cement",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("1.0"),
            gwp_a1a3=Decimal("0.9"),
            valid_until=date(2027, 12, 31),
            boundary="cradle_to_gate",
            verified=True,
        )

        gwp_a1a3 = engine.extract_gwp_a1a3(epd)
        assert gwp_a1a3 == Decimal("0.9")

    def test_epd_without_verification(self):
        """Test unverified EPD handling."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Glass",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("0.8"),
            valid_until=date(2027, 12, 31),
            boundary="cradle_to_gate",
            verified=False,
        )

        dqi_penalty = engine.compute_dqi_penalty(epd)
        assert dqi_penalty > 0

    def test_validate_epd_boundary(self):
        """Test EPD system boundary validation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd_cradle_gate = EPDData(
            epd_id="EPD001",
            product_name="Product A",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("10"),
            boundary="cradle_to_gate",
            valid_until=date(2027, 12, 31),
            verified=True,
        )

        epd_cradle_grave = EPDData(
            epd_id="EPD002",
            product_name="Product B",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("15"),
            boundary="cradle_to_grave",
            valid_until=date(2027, 12, 31),
            verified=True,
        )

        assert engine.is_boundary_acceptable(epd_cradle_gate.boundary) is True
        assert engine.is_boundary_acceptable(epd_cradle_grave.boundary) is False


class TestCDPIntegration:
    """Test CDP (Carbon Disclosure Project) integration."""

    def test_cdp_with_revenue_allocation(self):
        """Test CDP data with revenue allocation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        cdp = CDPData(
            supplier_id="SUP001",
            reporting_year=2024,
            scope3_category3=Decimal("500000"),
            total_revenue=Decimal("10000000"),
            cdp_score="A",
        )

        allocated = engine.allocate_cdp_emissions(
            cdp_data=cdp,
            purchase_value=Decimal("250000"),
        )

        # 500000 × (250000 / 10000000) = 12500
        assert allocated == Decimal("12500.0")

    def test_cdp_score_quality_mapping(self):
        """Test CDP score to data quality mapping."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        dqi_a = engine.map_cdp_score_to_dqi("A")
        dqi_b = engine.map_cdp_score_to_dqi("B")
        dqi_c = engine.map_cdp_score_to_dqi("C")
        dqi_d = engine.map_cdp_score_to_dqi("D")

        assert dqi_a > dqi_b > dqi_c > dqi_d

    def test_cdp_missing_category3_data(self):
        """Test handling of missing Scope 3 Category 3 data."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        cdp = CDPData(
            supplier_id="SUP001",
            reporting_year=2024,
            total_revenue=Decimal("5000000"),
            cdp_score="B",
        )

        with pytest.raises(ValueError, match="CDP data missing Scope 3 Category 3"):
            engine.allocate_cdp_emissions(cdp, Decimal("100000"))

    def test_cdp_zero_revenue(self):
        """Test CDP with zero total revenue."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        cdp = CDPData(
            supplier_id="SUP001",
            reporting_year=2024,
            scope3_category3=Decimal("100000"),
            total_revenue=Decimal("0"),
            cdp_score="C",
        )

        with pytest.raises(ValueError, match="Total revenue cannot be zero"):
            engine.allocate_cdp_emissions(cdp, Decimal("50000"))

    def test_cdp_score_None_handling(self):
        """Test handling of None CDP score."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        dqi = engine.map_cdp_score_to_dqi(None)
        assert dqi == Decimal("2.0")  # Default medium quality


class TestBoundaryVerification:
    """Test system boundary verification."""

    def test_cradle_to_gate_valid(self):
        """Test cradle-to-gate boundary is valid."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        is_valid = engine.verify_boundary("cradle_to_gate")
        assert is_valid is True

    def test_cradle_to_grave_rejected(self):
        """Test cradle-to-grave boundary is rejected."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        is_valid = engine.verify_boundary("cradle_to_grave")
        assert is_valid is False

    def test_gate_to_gate_flagged(self):
        """Test gate-to-gate boundary is flagged with warning."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        warnings = engine.check_boundary_warnings("gate_to_gate")
        assert len(warnings) > 0

    def test_unknown_boundary_rejected(self):
        """Test unknown boundary is rejected."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        is_valid = engine.verify_boundary("unknown_boundary")
        assert is_valid is False


class TestSupplierEngagement:
    """Test supplier engagement level scoring."""

    def test_level_1_primary_data(self):
        """Test Level 1 (primary data) scoring."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        score = engine.score_engagement_level(EngagementLevel.PRIMARY_DATA)
        assert score == 5

    def test_level_2_epd_pcf(self):
        """Test Level 2 (EPD/PCF) scoring."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        score = engine.score_engagement_level(EngagementLevel.EPD_PCF)
        assert score == 4

    def test_level_3_cdp_disclosure(self):
        """Test Level 3 (CDP disclosure) scoring."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        score = engine.score_engagement_level(EngagementLevel.CDP_DISCLOSURE)
        assert score == 3

    def test_level_4_questionnaire(self):
        """Test Level 4 (questionnaire) scoring."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        score = engine.score_engagement_level(EngagementLevel.QUESTIONNAIRE)
        assert score == 2

    def test_level_5_no_engagement(self):
        """Test Level 5 (no engagement) scoring."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        score = engine.score_engagement_level(EngagementLevel.NO_ENGAGEMENT)
        assert score == 1

    def test_engagement_recommendations(self):
        """Test engagement level recommendations."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        recommendations = engine.get_engagement_recommendations(
            current_level=EngagementLevel.QUESTIONNAIRE,
            supplier_size="large",
            spend_materiality="high",
        )

        assert len(recommendations) > 0
        assert any("primary data" in rec.lower() for rec in recommendations)


class TestDQIScoring:
    """Test Data Quality Indicator scoring."""

    def test_verified_epd_high_dqi(self):
        """Test verified EPD gets high DQI."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Steel",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("2.0"),
            valid_until=date(2027, 12, 31),
            boundary="cradle_to_gate",
            verified=True,
        )

        dqi = engine.compute_dqi(epd=epd)
        assert dqi <= Decimal("1.5")  # High quality = low DQI score

    def test_unverified_data_lower_dqi(self):
        """Test unverified data gets lower DQI."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        epd = EPDData(
            epd_id="EPD001",
            product_name="Aluminum",
            declared_unit=Decimal("1"),
            declared_unit_type="kg",
            gwp_total=Decimal("10.0"),
            valid_until=date(2027, 12, 31),
            boundary="cradle_to_gate",
            verified=False,
        )

        dqi = engine.compute_dqi(epd=epd)
        assert dqi >= Decimal("2.0")  # Lower quality = higher DQI score

    def test_dqi_source_dependent(self):
        """Test DQI varies by data source."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        dqi_primary = engine.compute_dqi_by_source("primary_data")
        dqi_secondary = engine.compute_dqi_by_source("secondary_data")
        dqi_estimated = engine.compute_dqi_by_source("estimated")

        assert dqi_primary < dqi_secondary < dqi_estimated

    def test_dqi_components(self):
        """Test DQI component breakdown."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        components = engine.compute_dqi_components(
            temporal_representativeness=Decimal("1.0"),
            geographic_representativeness=Decimal("1.2"),
            technological_representativeness=Decimal("1.1"),
            completeness=Decimal("1.0"),
            reliability=Decimal("1.3"),
        )

        # DQI = sqrt(sum of squares / 5)
        expected = (
            (Decimal("1.0")**2 + Decimal("1.2")**2 + Decimal("1.1")**2 +
             Decimal("1.0")**2 + Decimal("1.3")**2) / 5
        ).sqrt()

        assert abs(components["composite_dqi"] - expected) < Decimal("0.01")


class TestBatchAndAggregation:
    """Test batch processing and aggregation."""

    def test_batch_processing_multiple_products(self):
        """Test batch processing of multiple products."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        products = [
            ProductEmissions(
                product_id=f"P{i:03d}",
                quantity=Decimal("100"),
                unit="kg",
                emission_factor=Decimal("2.0"),
            )
            for i in range(10)
        ]

        results = engine.calculate_batch(products)

        assert len(results) == 10
        assert all(r["total_emissions"] == Decimal("200.0") for r in results)

    def test_aggregation_by_supplier(self):
        """Test aggregation by supplier."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        products = [
            ProductEmissions(
                product_id="P001",
                supplier_id="SUP001",
                quantity=Decimal("100"),
                unit="kg",
                emission_factor=Decimal("2.0"),
            ),
            ProductEmissions(
                product_id="P002",
                supplier_id="SUP001",
                quantity=Decimal("200"),
                unit="kg",
                emission_factor=Decimal("1.5"),
            ),
            ProductEmissions(
                product_id="P003",
                supplier_id="SUP002",
                quantity=Decimal("150"),
                unit="kg",
                emission_factor=Decimal("3.0"),
            ),
        ]

        aggregated = engine.aggregate_by_supplier(products)

        assert aggregated["SUP001"]["total_emissions"] == Decimal("500.0")
        assert aggregated["SUP002"]["total_emissions"] == Decimal("450.0")

    def test_aggregation_by_product_category(self):
        """Test aggregation by product category."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        products = [
            ProductEmissions(
                product_id="P001",
                category="raw_materials",
                quantity=Decimal("1000"),
                unit="kg",
                emission_factor=Decimal("1.5"),
            ),
            ProductEmissions(
                product_id="P002",
                category="raw_materials",
                quantity=Decimal("500"),
                unit="kg",
                emission_factor=Decimal("2.0"),
            ),
            ProductEmissions(
                product_id="P003",
                category="packaging",
                quantity=Decimal("200"),
                unit="kg",
                emission_factor=Decimal("0.5"),
            ),
        ]

        aggregated = engine.aggregate_by_category(products)

        assert aggregated["raw_materials"]["total_emissions"] == Decimal("2500.0")
        assert aggregated["packaging"]["total_emissions"] == Decimal("100.0")

    def test_coverage_calculation(self):
        """Test coverage calculation."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        total_spend = Decimal("1000000")
        covered_spend = Decimal("750000")

        coverage = engine.calculate_coverage(
            covered_spend=covered_spend,
            total_spend=total_spend,
        )

        assert coverage == Decimal("75.0")

    def test_empty_batch_handling(self):
        """Test handling of empty batch."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        results = engine.calculate_batch([])

        assert len(results) == 0


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(self):
        """Test health check returns healthy status."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert "engine" in health
        assert health["engine"] == "SupplierSpecificCalculatorEngine"

    def test_health_check_includes_stats(self):
        """Test health check includes statistics."""
        engine = SupplierSpecificCalculatorEngine.get_instance()

        # Perform some calculations
        product = ProductEmissions(
            product_id="P001",
            quantity=Decimal("100"),
            unit="kg",
            emission_factor=Decimal("2.0"),
        )
        engine.calculate_product_emissions(product)

        health = engine.health_check()

        assert "calculations_performed" in health
        assert health["calculations_performed"] > 0
