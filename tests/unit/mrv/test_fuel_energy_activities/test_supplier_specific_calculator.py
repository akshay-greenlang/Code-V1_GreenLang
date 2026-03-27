"""
Unit tests for SupplierSpecificCalculatorEngine (Engine 5)

Tests supplier-specific emission factor calculations for fuel and electricity.
Validates EPD verification, MIQ grades, OGMP levels, and allocation methods.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.agents.mrv.fuel_energy_activities.engines.supplier_specific_calculator import (
    SupplierSpecificCalculatorEngine,
    SupplierSpecificInput,
    SupplierSpecificOutput,
    SupplierDataType,
    AllocationMethod,
    MIQGrade,
    OGMPLevel,
)
from greenlang.agents.mrv.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
    VerificationLevel,
)
from greenlang_core import AgentConfig
from greenlang_core.exceptions import ValidationError


# Fixtures
@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="supplier_specific_calculator",
        version="1.0.0",
        environment="test"
    )


@pytest.fixture
def engine(agent_config):
    """Create SupplierSpecificCalculatorEngine instance for testing."""
    return SupplierSpecificCalculatorEngine(agent_config)


@pytest.fixture
def fuel_upstream_input():
    """Create fuel upstream supplier-specific input."""
    return SupplierSpecificInput(
        supplier_id="SUP-001",
        supplier_name="CleanFuel Corp",
        data_type=SupplierDataType.FUEL_UPSTREAM,
        fuel_type=FuelType.NATURAL_GAS,
        quantity=Decimal("1000"),  # 1000 mmBtu
        supplier_ef_kgco2e_per_unit=Decimal("5.2"),  # kg CO2e per mmBtu
        verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
        reporting_period="2025-Q1"
    )


@pytest.fixture
def electricity_upstream_input():
    """Create electricity upstream supplier-specific input."""
    return SupplierSpecificInput(
        supplier_id="SUP-002",
        supplier_name="GreenPower Inc",
        data_type=SupplierDataType.ELECTRICITY_UPSTREAM,
        electricity_kwh=Decimal("50000"),
        supplier_ef_kgco2e_per_kwh=Decimal("0.15"),  # Low-carbon source
        verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
        reporting_period="2025-Q1"
    )


@pytest.fixture
def epd_input():
    """Create EPD (Environmental Product Declaration) input."""
    return SupplierSpecificInput(
        supplier_id="SUP-003",
        supplier_name="EPD Supplier",
        data_type=SupplierDataType.EPD,
        fuel_type=FuelType.DIESEL,
        quantity=Decimal("5000"),  # 5000 liters
        epd_document_id="EPD-2024-12345",
        epd_gwp_total_kgco2e_per_unit=Decimal("0.35"),
        epd_verification_body="ISO 14025 Certified Body",
        epd_issue_date=date(2024, 1, 1),
        epd_expiry_date=date(2027, 1, 1),
        reporting_period="2025-Q1"
    )


@pytest.fixture
def miq_grade_a_input():
    """Create MIQ Grade A natural gas input."""
    return SupplierSpecificInput(
        supplier_id="SUP-004",
        supplier_name="MIQ A Gas Supplier",
        data_type=SupplierDataType.MIQ_CERTIFIED,
        fuel_type=FuelType.NATURAL_GAS,
        quantity=Decimal("2000"),  # mmBtu
        miq_grade=MIQGrade.GRADE_A,
        miq_certificate_id="MIQ-A-2024-001",
        verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
        reporting_period="2025-Q1"
    )


# Test Class
class TestSupplierSpecificCalculatorEngine:
    """Test suite for SupplierSpecificCalculatorEngine."""

    def test_initialization(self, agent_config):
        """Test engine initializes correctly."""
        engine = SupplierSpecificCalculatorEngine(agent_config)

        assert engine.config == agent_config
        assert engine.default_ef_db is not None

    def test_calculate_fuel_upstream_with_supplier_ef(self, engine, fuel_upstream_input):
        """Test calculating fuel upstream with supplier-specific EF."""
        result = engine.calculate(fuel_upstream_input)

        assert isinstance(result, SupplierSpecificOutput)

        # 1000 mmBtu × 5.2 kg/mmBtu = 5200 kg CO2e
        expected_emissions = Decimal("1000") * Decimal("5.2")
        assert result.total_emissions_kgco2e == pytest.approx(expected_emissions, rel=Decimal("0.001"))

        assert result.activity_type == ActivityType.ACTIVITY_3A
        assert result.supplier_id == "SUP-001"
        assert result.data_quality_tier == "SUPPLIER_SPECIFIC"
        assert result.provenance_hash is not None

    def test_calculate_electricity_upstream_with_supplier(self, engine, electricity_upstream_input):
        """Test calculating electricity upstream with supplier data."""
        result = engine.calculate(electricity_upstream_input)

        assert isinstance(result, SupplierSpecificOutput)

        # 50000 kWh × 0.15 kg/kWh = 7500 kg CO2e
        expected_emissions = Decimal("50000") * Decimal("0.15")
        assert result.total_emissions_kgco2e == pytest.approx(expected_emissions, rel=Decimal("0.001"))

        assert result.activity_type == ActivityType.ACTIVITY_3B
        assert result.supplier_id == "SUP-002"

    def test_calculate_batch_mixed_records(self, engine, fuel_upstream_input, electricity_upstream_input):
        """Test batch calculation with mixed record types."""
        inputs = [fuel_upstream_input, electricity_upstream_input]

        results = engine.calculate_batch(inputs)

        assert len(results) == 2
        assert results[0].activity_type == ActivityType.ACTIVITY_3A
        assert results[1].activity_type == ActivityType.ACTIVITY_3B

    def test_validate_supplier_data_complete(self, engine, fuel_upstream_input):
        """Test validation of complete supplier data."""
        is_valid = engine.validate_supplier_data(fuel_upstream_input)

        assert is_valid is True

    def test_validate_supplier_data_missing_fields(self, engine):
        """Test validation fails for missing required fields."""
        incomplete_input = SupplierSpecificInput(
            supplier_id="SUP-001",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            quantity=Decimal("1000"),
            # Missing supplier_ef_kgco2e_per_unit
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError, match="supplier.*emission factor"):
            engine.validate_supplier_data(incomplete_input, raise_on_invalid=True)

    def test_validate_epd_complete(self, engine, epd_input):
        """Test validation of complete EPD."""
        is_valid = engine.validate_epd(epd_input)

        assert is_valid is True

    def test_validate_epd_invalid(self, engine):
        """Test validation fails for invalid EPD."""
        invalid_epd = SupplierSpecificInput(
            supplier_id="SUP-003",
            data_type=SupplierDataType.EPD,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("5000"),
            epd_document_id="EPD-INVALID",
            # Missing EPD details
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError):
            engine.validate_epd(invalid_epd, raise_on_invalid=True)

    def test_assess_miq_grade_a(self, engine, miq_grade_a_input):
        """Test MIQ Grade A assessment (best-in-class)."""
        result = engine.calculate(miq_grade_a_input)

        # Grade A should have lower upstream emissions
        miq_adjustment = engine.get_miq_upstream_adjustment(MIQGrade.GRADE_A)

        assert miq_adjustment < Decimal("1.0")  # <1.0 means reduction
        assert result.miq_grade == MIQGrade.GRADE_A
        assert result.miq_adjustment_factor == miq_adjustment

    def test_assess_miq_grade_c(self, engine):
        """Test MIQ Grade C assessment (industry average)."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-005",
            supplier_name="MIQ C Gas Supplier",
            data_type=SupplierDataType.MIQ_CERTIFIED,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("2000"),
            miq_grade=MIQGrade.GRADE_C,
            miq_certificate_id="MIQ-C-2024-001",
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Grade C should be ~1.0 (baseline)
        assert result.miq_adjustment_factor == pytest.approx(Decimal("1.0"), rel=Decimal("0.05"))

    def test_assess_miq_grade_f(self, engine):
        """Test MIQ Grade F assessment (poor performance)."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-006",
            supplier_name="MIQ F Gas Supplier",
            data_type=SupplierDataType.MIQ_CERTIFIED,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("2000"),
            miq_grade=MIQGrade.GRADE_F,
            miq_certificate_id="MIQ-F-2024-001",
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Grade F should have higher upstream emissions
        assert result.miq_adjustment_factor > Decimal("1.0")

    def test_get_miq_upstream_adjustment_all_grades(self, engine):
        """Test MIQ upstream adjustment for all grades."""
        grades = [
            MIQGrade.GRADE_A, MIQGrade.GRADE_B, MIQGrade.GRADE_C,
            MIQGrade.GRADE_D, MIQGrade.GRADE_E, MIQGrade.GRADE_F
        ]

        adjustments = {grade: engine.get_miq_upstream_adjustment(grade) for grade in grades}

        # A should be lowest, F should be highest
        assert adjustments[MIQGrade.GRADE_A] < adjustments[MIQGrade.GRADE_F]

        # Should be monotonically increasing
        assert adjustments[MIQGrade.GRADE_A] <= adjustments[MIQGrade.GRADE_B]
        assert adjustments[MIQGrade.GRADE_B] <= adjustments[MIQGrade.GRADE_C]
        assert adjustments[MIQGrade.GRADE_C] <= adjustments[MIQGrade.GRADE_D]
        assert adjustments[MIQGrade.GRADE_D] <= adjustments[MIQGrade.GRADE_E]
        assert adjustments[MIQGrade.GRADE_E] <= adjustments[MIQGrade.GRADE_F]

    def test_assess_ogmp_level_1_through_5(self, engine):
        """Test OGMP (Oil & Gas Methane Partnership) levels 1-5."""
        levels = [
            OGMPLevel.LEVEL_1, OGMPLevel.LEVEL_2, OGMPLevel.LEVEL_3,
            OGMPLevel.LEVEL_4, OGMPLevel.LEVEL_5
        ]

        for level in levels:
            input_data = SupplierSpecificInput(
                supplier_id=f"SUP-OGMP-{level.value}",
                supplier_name=f"OGMP Level {level.value} Supplier",
                data_type=SupplierDataType.OGMP_CERTIFIED,
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("1000"),
                ogmp_level=level,
                reporting_period="2025-Q1"
            )

            result = engine.calculate(input_data)

            assert result.ogmp_level == level
            # Higher levels should have better data quality
            if level == OGMPLevel.LEVEL_5:
                assert result.dqi_score >= Decimal("4.0")

    def test_calculate_ppa_upstream_solar(self, engine):
        """Test PPA (Power Purchase Agreement) upstream for solar."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-PPA-SOLAR",
            supplier_name="Solar PPA",
            data_type=SupplierDataType.PPA,
            electricity_kwh=Decimal("100000"),
            ppa_technology="SOLAR",
            supplier_ef_kgco2e_per_kwh=Decimal("0.02"),  # Very low for solar
            verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Solar should have very low emissions
        assert result.total_emissions_kgco2e < Decimal("3000")  # <0.03 kg/kWh
        assert result.ppa_technology == "SOLAR"

    def test_calculate_ppa_upstream_wind(self, engine):
        """Test PPA upstream for wind."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-PPA-WIND",
            supplier_name="Wind PPA",
            data_type=SupplierDataType.PPA,
            electricity_kwh=Decimal("100000"),
            ppa_technology="WIND",
            supplier_ef_kgco2e_per_kwh=Decimal("0.01"),  # Very low for wind
            verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Wind should have very low emissions
        assert result.total_emissions_kgco2e < Decimal("2000")

    def test_allocate_supplier_emissions_revenue(self, engine):
        """Test allocating supplier emissions by revenue."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-007",
            supplier_name="Multi-Product Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            supplier_total_emissions_kgco2e=Decimal("100000"),  # Total supplier emissions
            allocation_method=AllocationMethod.REVENUE,
            purchased_revenue=Decimal("50000"),  # $50k
            total_supplier_revenue=Decimal("500000"),  # $500k total
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Should allocate 50000/500000 = 10% of total emissions
        expected_allocated = Decimal("100000") * Decimal("0.1")
        assert result.allocated_emissions_kgco2e == pytest.approx(expected_allocated, rel=Decimal("0.01"))

    def test_allocate_supplier_emissions_production_volume(self, engine):
        """Test allocating supplier emissions by production volume."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-008",
            supplier_name="Volume-based Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("5000"),
            supplier_total_emissions_kgco2e=Decimal("50000"),
            allocation_method=AllocationMethod.PRODUCTION_VOLUME,
            purchased_volume=Decimal("5000"),
            total_supplier_volume=Decimal("20000"),
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Should allocate 5000/20000 = 25% of total emissions
        expected_allocated = Decimal("50000") * Decimal("0.25")
        assert result.allocated_emissions_kgco2e == pytest.approx(expected_allocated, rel=Decimal("0.01"))

    def test_allocate_supplier_emissions_energy(self, engine):
        """Test allocating supplier emissions by energy content."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-009",
            supplier_name="Energy-based Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("1000"),
            supplier_total_emissions_kgco2e=Decimal("80000"),
            allocation_method=AllocationMethod.ENERGY,
            purchased_energy_mj=Decimal("40000"),
            total_supplier_energy_mj=Decimal("200000"),
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Should allocate 40000/200000 = 20% of total emissions
        expected_allocated = Decimal("80000") * Decimal("0.2")
        assert result.allocated_emissions_kgco2e == pytest.approx(expected_allocated, rel=Decimal("0.01"))

    def test_blend_with_average_50_50(self, engine, fuel_upstream_input):
        """Test blending supplier-specific with average (50/50 mix)."""
        # Calculate with 50% supplier-specific, 50% average
        result_blended = engine.calculate_with_average_blend(
            fuel_upstream_input,
            blend_ratio=Decimal("0.5")
        )

        # Calculate 100% supplier-specific
        result_supplier = engine.calculate(fuel_upstream_input)

        # Blended should be between supplier and average
        # (assuming average is different from supplier)
        assert result_blended.total_emissions_kgco2e != result_supplier.total_emissions_kgco2e

    def test_assess_coverage_100_pct(self, engine):
        """Test assessing 100% supplier-specific coverage."""
        coverage_data = {
            "total_spend": Decimal("1000000"),
            "supplier_specific_spend": Decimal("1000000"),
        }

        coverage = engine.assess_coverage(coverage_data)

        assert coverage["coverage_pct"] == Decimal("100")
        assert coverage["quality_tier"] == "SUPPLIER_SPECIFIC"

    def test_assess_coverage_partial(self, engine):
        """Test assessing partial supplier-specific coverage."""
        coverage_data = {
            "total_spend": Decimal("1000000"),
            "supplier_specific_spend": Decimal("600000"),
        }

        coverage = engine.assess_coverage(coverage_data)

        assert coverage["coverage_pct"] == Decimal("60")
        assert coverage["quality_tier"] == "HYBRID"

    def test_assess_verification_level_third_party(self, engine, fuel_upstream_input):
        """Test verification level assessment for third-party verified."""
        result = engine.calculate(fuel_upstream_input)

        assert result.verification_level == VerificationLevel.THIRD_PARTY_VERIFIED
        assert result.verification_confidence >= Decimal("0.95")  # High confidence

    def test_assess_verification_level_unverified(self, engine):
        """Test verification level assessment for unverified data."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-010",
            supplier_name="Unverified Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.COAL,
            quantity=Decimal("1000"),
            supplier_ef_kgco2e_per_unit=Decimal("95"),
            verification_level=VerificationLevel.UNVERIFIED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        assert result.verification_level == VerificationLevel.UNVERIFIED
        assert result.verification_confidence < Decimal("0.7")  # Lower confidence

    def test_compare_with_average(self, engine, fuel_upstream_input):
        """Test comparing supplier-specific with industry average."""
        result = engine.calculate(fuel_upstream_input)

        comparison = engine.compare_with_average(result, fuel_upstream_input)

        assert "supplier_ef" in comparison
        assert "average_ef" in comparison
        assert "difference_pct" in comparison
        assert "better_or_worse" in comparison

    def test_assess_dqi_high_quality_supplier(self, engine, fuel_upstream_input):
        """Test DQI assessment for high-quality supplier data."""
        result = engine.calculate(fuel_upstream_input)

        # Third-party verified supplier-specific should have high DQI
        assert result.dqi_score >= Decimal("4.0")

    def test_quantify_uncertainty_narrowed_by_verification(self, engine):
        """Test uncertainty is narrowed by verification."""
        # Verified supplier data
        verified_input = SupplierSpecificInput(
            supplier_id="SUP-011",
            supplier_name="Verified Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("1000"),
            supplier_ef_kgco2e_per_unit=Decimal("5.5"),
            verification_level=VerificationLevel.THIRD_PARTY_VERIFIED,
            reporting_period="2025-Q1"
        )

        # Unverified supplier data
        unverified_input = SupplierSpecificInput(
            supplier_id="SUP-012",
            supplier_name="Unverified Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("1000"),
            supplier_ef_kgco2e_per_unit=Decimal("5.5"),
            verification_level=VerificationLevel.UNVERIFIED,
            reporting_period="2025-Q1"
        )

        verified_result = engine.calculate(verified_input)
        unverified_result = engine.calculate(unverified_input)

        # Verified should have lower uncertainty
        assert verified_result.uncertainty_pct < unverified_result.uncertainty_pct

    def test_aggregate_by_supplier(self, engine):
        """Test aggregating emissions by supplier."""
        inputs = [
            SupplierSpecificInput(
                supplier_id="SUP-001",
                supplier_name="Supplier A",
                data_type=SupplierDataType.FUEL_UPSTREAM,
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("1000"),
                supplier_ef_kgco2e_per_unit=Decimal("3.0"),
                reporting_period="2025-Q1"
            ),
            SupplierSpecificInput(
                supplier_id="SUP-001",
                supplier_name="Supplier A",
                data_type=SupplierDataType.FUEL_UPSTREAM,
                fuel_type=FuelType.GASOLINE,
                quantity=Decimal("500"),
                supplier_ef_kgco2e_per_unit=Decimal("2.5"),
                reporting_period="2025-Q2"
            ),
            SupplierSpecificInput(
                supplier_id="SUP-002",
                supplier_name="Supplier B",
                data_type=SupplierDataType.FUEL_UPSTREAM,
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("2000"),
                supplier_ef_kgco2e_per_unit=Decimal("5.0"),
                reporting_period="2025-Q1"
            ),
        ]

        results = engine.calculate_batch(inputs)
        aggregated = engine.aggregate_by_supplier(results)

        assert len(aggregated) == 2  # SUP-001 and SUP-002
        assert "SUP-001" in aggregated
        assert "SUP-002" in aggregated

        # SUP-001 should have combined emissions from both periods
        assert aggregated["SUP-001"]["total_emissions_kgco2e"] == (
            Decimal("1000") * Decimal("3.0") + Decimal("500") * Decimal("2.5")
        )

    def test_get_supplier_performance_ranking(self, engine):
        """Test ranking suppliers by emission intensity."""
        suppliers = [
            {"supplier_id": "SUP-001", "ef": Decimal("5.0")},
            {"supplier_id": "SUP-002", "ef": Decimal("3.0")},  # Best
            {"supplier_id": "SUP-003", "ef": Decimal("8.0")},  # Worst
        ]

        ranking = engine.rank_suppliers_by_performance(suppliers)

        assert ranking[0]["supplier_id"] == "SUP-002"  # Lowest EF first
        assert ranking[1]["supplier_id"] == "SUP-001"
        assert ranking[2]["supplier_id"] == "SUP-003"

    def test_calculate_supplier_switching_impact(self, engine):
        """Test calculating impact of switching suppliers."""
        current_supplier = SupplierSpecificInput(
            supplier_id="SUP-CURRENT",
            supplier_name="Current Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            supplier_ef_kgco2e_per_unit=Decimal("3.5"),
            reporting_period="2025-Q1"
        )

        alternative_supplier = SupplierSpecificInput(
            supplier_id="SUP-ALT",
            supplier_name="Alternative Supplier",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("10000"),
            supplier_ef_kgco2e_per_unit=Decimal("2.8"),  # Lower EF
            reporting_period="2025-Q1"
        )

        current_result = engine.calculate(current_supplier)
        alternative_result = engine.calculate(alternative_supplier)

        impact = engine.calculate_switching_impact(current_result, alternative_result)

        assert impact["emissions_reduction_kgco2e"] > Decimal("0")
        assert impact["reduction_pct"] > Decimal("0")

    def test_provenance_tracking(self, engine, fuel_upstream_input):
        """Test provenance tracking is deterministic."""
        result1 = engine.calculate(fuel_upstream_input)
        result2 = engine.calculate(fuel_upstream_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_epd_expiry_check(self, engine):
        """Test EPD expiry date checking."""
        expired_epd = SupplierSpecificInput(
            supplier_id="SUP-EPD-EXPIRED",
            data_type=SupplierDataType.EPD,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("5000"),
            epd_document_id="EPD-EXPIRED",
            epd_gwp_total_kgco2e_per_unit=Decimal("0.35"),
            epd_expiry_date=date(2020, 1, 1),  # Expired
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError, match="expired"):
            engine.validate_epd(expired_epd, raise_on_invalid=True)

    def test_miq_certificate_validation(self, engine, miq_grade_a_input):
        """Test MIQ certificate validation."""
        is_valid = engine.validate_miq_certificate(
            miq_grade_a_input.miq_certificate_id,
            miq_grade_a_input.miq_grade
        )

        assert is_valid is True

    def test_get_statistics(self, engine, fuel_upstream_input):
        """Test getting engine statistics."""
        engine.calculate(fuel_upstream_input)
        engine.calculate(fuel_upstream_input)

        stats = engine.get_statistics()

        assert stats["calculations_performed"] == 2
        assert stats["total_emissions_kgco2e"] > Decimal("0")

    def test_reset(self, engine, fuel_upstream_input):
        """Test resetting engine state."""
        engine.calculate(fuel_upstream_input)

        engine.reset()

        stats = engine.get_statistics()
        assert stats["calculations_performed"] == 0

    def test_error_handling_missing_allocation_data(self, engine):
        """Test error handling for missing allocation data."""
        input_data = SupplierSpecificInput(
            supplier_id="SUP-BAD",
            data_type=SupplierDataType.FUEL_UPSTREAM,
            fuel_type=FuelType.DIESEL,
            quantity=Decimal("1000"),
            supplier_total_emissions_kgco2e=Decimal("50000"),
            allocation_method=AllocationMethod.REVENUE,
            # Missing purchased_revenue and total_supplier_revenue
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError, match="allocation"):
            engine.calculate(input_data)

    def test_performance_batch_processing(self, engine, benchmark):
        """Test batch processing performance."""
        inputs = [
            SupplierSpecificInput(
                supplier_id=f"SUP-{i}",
                supplier_name=f"Supplier {i}",
                data_type=SupplierDataType.FUEL_UPSTREAM,
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("1000"),
                supplier_ef_kgco2e_per_unit=Decimal("5.0"),
                reporting_period="2025-Q1"
            )
            for i in range(100)
        ]

        def run_batch():
            return engine.calculate_batch(inputs)

        results = benchmark(run_batch)

        assert len(results) == 100


# Integration Tests
class TestSupplierSpecificCalculatorIntegration:
    """Integration tests for SupplierSpecificCalculatorEngine."""

    @pytest.mark.integration
    def test_integration_with_supplier_database(self, engine):
        """Test integration with supplier master database."""
        pass

    @pytest.mark.integration
    def test_integration_with_verification_service(self, engine):
        """Test integration with third-party verification service."""
        pass


# Performance Tests
class TestSupplierSpecificCalculatorPerformance:
    """Performance tests for SupplierSpecificCalculatorEngine."""

    @pytest.mark.performance
    def test_throughput_target(self, engine):
        """Test engine meets throughput target (1000 calculations/sec)."""
        num_records = 10000
        inputs = [
            SupplierSpecificInput(
                supplier_id=f"SUP-{i}",
                supplier_name=f"Supplier {i}",
                data_type=SupplierDataType.FUEL_UPSTREAM,
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("1000"),
                supplier_ef_kgco2e_per_unit=Decimal("5.0"),
                reporting_period="2025-Q1"
            )
            for i in range(num_records)
        ]

        start_time = datetime.now()
        results = engine.calculate_batch(inputs)
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_records / duration_seconds

        assert throughput >= 1000
        assert len(results) == num_records
