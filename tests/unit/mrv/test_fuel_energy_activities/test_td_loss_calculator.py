"""
Unit tests for TDLossCalculatorEngine (Engine 4 - Activity 3c)

Tests transmission and distribution loss calculations for purchased electricity.
Validates loss rate factors for 50+ countries, eGRID subregions, and decomposition.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.agents.mrv.fuel_energy_activities.engines.td_loss_calculator import (
    TDLossCalculatorEngine,
    TDLossInput,
    TDLossOutput,
    LossBasis,
    TDLossComponent,
)
from greenlang.agents.mrv.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
    ComplianceFramework,
)
from greenlang_core import AgentConfig
from greenlang_core.exceptions import ValidationError


# Fixtures
@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="td_loss_calculator",
        version="1.0.0",
        environment="test"
    )


@pytest.fixture
def engine(agent_config):
    """Create TDLossCalculatorEngine instance for testing."""
    return TDLossCalculatorEngine(agent_config)


@pytest.fixture
def us_input():
    """Create US T&D loss input."""
    return TDLossInput(
        electricity_consumption_kwh=Decimal("100000"),
        country="US",
        egrid_subregion="RFCW",  # RFC West
        loss_basis=LossBasis.DELIVERED,
        reporting_period="2025-Q1"
    )


@pytest.fixture
def uk_input():
    """Create UK T&D loss input."""
    return TDLossInput(
        electricity_consumption_kwh=Decimal("50000"),
        country="GB",
        loss_basis=LossBasis.DELIVERED,
        reporting_period="2025-Q1"
    )


@pytest.fixture
def india_input():
    """Create India T&D loss input (high loss rate)."""
    return TDLossInput(
        electricity_consumption_kwh=Decimal("200000"),
        country="IN",
        loss_basis=LossBasis.DELIVERED,
        reporting_period="2025-Q1"
    )


# Test Class
class TestTDLossCalculatorEngine:
    """Test suite for TDLossCalculatorEngine."""

    def test_initialization(self, agent_config):
        """Test engine initializes correctly."""
        engine = TDLossCalculatorEngine(agent_config)

        assert engine.config == agent_config
        assert engine.td_loss_factors is not None
        assert len(engine.td_loss_factors) >= 50  # 50+ countries

    def test_calculate_us_td_losses(self, engine, us_input):
        """Test US T&D loss calculation.

        US average T&D loss ~5%
        100000 kWh × 5%/(1-5%) × 0.37 kg/kWh = ~1947 kgCO2e generation + upstream
        """
        result = engine.calculate(us_input)

        assert isinstance(result, TDLossOutput)
        assert result.total_emissions_kgco2e > Decimal("0")
        assert result.generation_emissions_kgco2e > Decimal("0")
        assert result.upstream_emissions_kgco2e > Decimal("0")

        # Total = generation + upstream
        assert result.total_emissions_kgco2e == (
            result.generation_emissions_kgco2e + result.upstream_emissions_kgco2e
        )

        # Verify loss rate is reasonable
        assert Decimal("0.03") <= result.td_loss_rate <= Decimal("0.08")  # 3-8%

        # Verify energy lost
        expected_loss_kwh = us_input.electricity_consumption_kwh * result.td_loss_rate / (
            Decimal("1") - result.td_loss_rate
        )
        assert result.energy_lost_kwh == pytest.approx(expected_loss_kwh, rel=Decimal("0.001"))

        assert result.activity_type == ActivityType.ACTIVITY_3C
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_calculate_uk_td_losses(self, engine, uk_input):
        """Test UK T&D loss calculation (7.8% loss rate)."""
        result = engine.calculate(uk_input)

        assert isinstance(result, TDLossOutput)
        assert result.total_emissions_kgco2e > Decimal("0")

        # UK has ~7.8% T&D losses
        assert result.td_loss_rate == pytest.approx(Decimal("0.078"), rel=Decimal("0.01"))

        # Energy lost = 50000 * 0.078 / (1 - 0.078) = ~4230 kWh
        expected_loss = Decimal("50000") * Decimal("0.078") / (Decimal("1") - Decimal("0.078"))
        assert result.energy_lost_kwh == pytest.approx(expected_loss, rel=Decimal("0.01"))

        assert result.country == "GB"
        assert result.components is not None
        assert len(result.components) == 2  # Transmission + Distribution

    def test_calculate_india_td_losses(self, engine, india_input):
        """Test India T&D loss calculation (19% loss rate - high)."""
        result = engine.calculate(india_input)

        assert isinstance(result, TDLossOutput)

        # India has high T&D losses ~19%
        assert result.td_loss_rate > Decimal("0.15")  # >15%
        assert result.td_loss_rate <= Decimal("0.25")  # <25%

        # High losses mean more emissions
        assert result.total_emissions_kgco2e > Decimal("10000")

        # Energy lost should be significant
        assert result.energy_lost_kwh > Decimal("40000")  # >40000 kWh lost

        assert result.country == "IN"

    def test_calculate_auto_country_resolution(self, engine):
        """Test automatic country resolution from facility location."""
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("75000"),
            facility_id="FAC-DE-001",
            facility_country="DE",  # Germany
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        assert result.country == "DE"
        assert result.total_emissions_kgco2e > Decimal("0")

        # Germany has moderate T&D losses ~5-6%
        assert Decimal("0.04") <= result.td_loss_rate <= Decimal("0.08")

    def test_calculate_batch(self, engine):
        """Test batch calculation for multiple facilities."""
        inputs = [
            TDLossInput(
                electricity_consumption_kwh=Decimal("100000"),
                country="US",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
            TDLossInput(
                electricity_consumption_kwh=Decimal("50000"),
                country="GB",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
            TDLossInput(
                electricity_consumption_kwh=Decimal("200000"),
                country="IN",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
        ]

        results = engine.calculate_batch(inputs)

        assert len(results) == 3
        assert all(isinstance(r, TDLossOutput) for r in results)
        assert results[0].country == "US"
        assert results[1].country == "GB"
        assert results[2].country == "IN"

        # India should have highest emissions due to highest T&D losses
        assert results[2].total_emissions_kgco2e > results[0].total_emissions_kgco2e
        assert results[2].total_emissions_kgco2e > results[1].total_emissions_kgco2e

    def test_calculate_generation_component_only(self, engine, us_input):
        """Test calculation of generation component only."""
        result = engine.calculate(us_input)

        # Generation component = energy_lost × grid_ef
        grid_ef = engine.get_grid_generation_ef("US", "RFCW")
        expected_generation = result.energy_lost_kwh * grid_ef

        assert result.generation_emissions_kgco2e == pytest.approx(
            expected_generation, rel=Decimal("0.001")
        )

    def test_calculate_upstream_component_only(self, engine, us_input):
        """Test calculation of upstream component only."""
        result = engine.calculate(us_input)

        # Upstream component = energy_lost × upstream_ef
        upstream_ef = engine.get_upstream_ef("US")
        expected_upstream = result.energy_lost_kwh * upstream_ef

        assert result.upstream_emissions_kgco2e == pytest.approx(
            expected_upstream, rel=Decimal("0.001")
        )

    def test_convert_loss_basis_delivered_to_generated(self, engine):
        """Test converting loss basis from delivered to generated."""
        # If given 100000 kWh generated, calculate delivered
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("100000"),
            country="US",
            loss_basis=LossBasis.GENERATED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Generated = Delivered / (1 - loss_rate)
        # Delivered = Generated × (1 - loss_rate)
        assert result.loss_basis == LossBasis.GENERATED
        assert result.electricity_consumption_kwh == Decimal("100000")
        assert result.energy_lost_kwh > Decimal("0")

    def test_get_td_loss_factor_50_countries(self, engine):
        """Test getting T&D loss factors for 50+ countries."""
        countries = [
            "US", "GB", "DE", "FR", "IN", "CN", "JP", "BR", "CA", "AU",
            "IT", "ES", "NL", "SE", "NO", "DK", "FI", "BE", "AT", "CH",
            "PL", "CZ", "HU", "RO", "BG", "GR", "PT", "IE", "NZ", "SG",
            "MY", "TH", "VN", "ID", "PH", "KR", "TW", "HK", "MX", "AR",
            "CL", "CO", "PE", "ZA", "EG", "NG", "KE", "MA", "TR", "SA",
        ]

        for country in countries:
            loss_factor = engine.get_td_loss_factor(country)

            assert loss_factor > Decimal("0")
            assert loss_factor < Decimal("0.30")  # <30% (even high-loss countries)

    def test_get_td_loss_by_egrid_26_subregions(self, engine):
        """Test getting T&D loss by eGRID subregion (26 US regions)."""
        egrid_subregions = [
            "AKGD", "AKMS", "AZNM", "CAMX", "ERCT", "FRCC", "HIMS", "HIOA",
            "MROE", "MROW", "NEWE", "NWPP", "NYCW", "NYLI", "NYUP", "RFCE",
            "RFCM", "RFCW", "RMPA", "SPNO", "SPSO", "SRMV", "SRMW", "SRSO",
            "SRTV", "SRVC",
        ]

        for subregion in egrid_subregions:
            loss_factor = engine.get_td_loss_by_egrid(subregion)

            assert loss_factor > Decimal("0")
            assert loss_factor < Decimal("0.15")  # <15% for US

    def test_get_grid_generation_ef(self, engine):
        """Test getting grid generation emission factors."""
        # US average
        us_ef = engine.get_grid_generation_ef("US")
        assert Decimal("0.3") <= us_ef <= Decimal("0.5")  # kg CO2e/kWh

        # UK (cleaner grid)
        uk_ef = engine.get_grid_generation_ef("GB")
        assert Decimal("0.15") <= uk_ef <= Decimal("0.35")

        # India (coal-heavy grid)
        in_ef = engine.get_grid_generation_ef("IN")
        assert in_ef >= Decimal("0.6")  # Higher emissions

    def test_decompose_td_losses(self, engine, us_input):
        """Test decomposing T&D losses into transmission vs distribution."""
        result = engine.calculate(us_input)

        assert result.components is not None
        assert len(result.components) == 2

        transmission = next(c for c in result.components if c.component_type == TDLossComponent.TRANSMISSION)
        distribution = next(c for c in result.components if c.component_type == TDLossComponent.DISTRIBUTION)

        # Transmission losses typically 2-3%
        assert Decimal("0.01") <= transmission.loss_rate <= Decimal("0.04")

        # Distribution losses typically 3-5%
        assert Decimal("0.02") <= distribution.loss_rate <= Decimal("0.06")

        # Total loss = transmission + distribution (approximately)
        total_component_loss = transmission.loss_rate + distribution.loss_rate
        assert result.td_loss_rate == pytest.approx(total_component_loss, rel=Decimal("0.05"))

    def test_calculate_per_gas_breakdown(self, engine, us_input):
        """Test per-gas breakdown of T&D loss emissions."""
        result = engine.calculate(us_input)

        assert result.emissions_by_gas is not None
        assert "CO2" in result.emissions_by_gas
        assert "CH4" in result.emissions_by_gas
        assert "N2O" in result.emissions_by_gas

        # CO2 should dominate
        assert result.emissions_by_gas["CO2"] > result.emissions_by_gas["CH4"]
        assert result.emissions_by_gas["CO2"] > result.emissions_by_gas["N2O"]

        # Sum should equal total
        total_gas_emissions = sum(result.emissions_by_gas.values())
        assert total_gas_emissions == pytest.approx(result.total_emissions_kgco2e, rel=Decimal("0.001"))

    def test_compare_countries_side_by_side(self, engine):
        """Test comparing T&D losses across countries."""
        countries = ["US", "GB", "DE", "IN", "CN"]
        consumption = Decimal("100000")

        results = []
        for country in countries:
            input_data = TDLossInput(
                electricity_consumption_kwh=consumption,
                country=country,
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            )
            results.append(engine.calculate(input_data))

        # India and China should have higher losses than US/GB/DE
        india_result = next(r for r in results if r.country == "IN")
        us_result = next(r for r in results if r.country == "US")

        assert india_result.td_loss_rate > us_result.td_loss_rate
        assert india_result.total_emissions_kgco2e > us_result.total_emissions_kgco2e

    def test_aggregate_by_country(self, engine):
        """Test aggregating T&D losses by country."""
        inputs = [
            TDLossInput(
                electricity_consumption_kwh=Decimal("100000"),
                country="US",
                facility_id="FAC-US-001",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
            TDLossInput(
                electricity_consumption_kwh=Decimal("150000"),
                country="US",
                facility_id="FAC-US-002",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
            TDLossInput(
                electricity_consumption_kwh=Decimal("50000"),
                country="GB",
                facility_id="FAC-GB-001",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
        ]

        results = engine.calculate_batch(inputs)
        aggregated = engine.aggregate_by_country(results)

        assert len(aggregated) == 2  # US and GB
        assert "US" in aggregated
        assert "GB" in aggregated

        # US should have combined consumption of 250000 kWh
        assert aggregated["US"]["total_consumption_kwh"] == Decimal("250000")
        assert aggregated["GB"]["total_consumption_kwh"] == Decimal("50000")

    def test_aggregate_by_facility(self, engine):
        """Test aggregating T&D losses by facility."""
        inputs = [
            TDLossInput(
                electricity_consumption_kwh=Decimal("100000"),
                country="US",
                facility_id="FAC-001",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            ),
            TDLossInput(
                electricity_consumption_kwh=Decimal("150000"),
                country="US",
                facility_id="FAC-001",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q2"
            ),
        ]

        results = engine.calculate_batch(inputs)
        aggregated = engine.aggregate_by_facility(results)

        assert len(aggregated) == 1
        assert "FAC-001" in aggregated
        assert aggregated["FAC-001"]["total_consumption_kwh"] == Decimal("250000")

    def test_assess_dqi(self, engine, us_input):
        """Test data quality index assessment for T&D losses."""
        result = engine.calculate(us_input)

        assert result.dqi_score is not None
        assert Decimal("0") <= result.dqi_score <= Decimal("5")

        # With eGRID subregion, should have good quality
        if us_input.egrid_subregion:
            assert result.dqi_score >= Decimal("3.5")

    def test_quantify_uncertainty(self, engine, us_input):
        """Test uncertainty quantification for T&D losses."""
        result = engine.calculate(us_input)

        assert result.uncertainty_pct is not None
        assert result.uncertainty_pct > Decimal("0")
        assert result.uncertainty_pct < Decimal("50")  # <50% uncertainty

        # T&D losses have moderate uncertainty ~10-20%
        assert result.uncertainty_pct <= Decimal("25")

    def test_sensitivity_analysis(self, engine, us_input):
        """Test sensitivity analysis for T&D loss parameters."""
        # Vary loss rate ±20%
        base_result = engine.calculate(us_input)

        sensitivity = engine.sensitivity_analysis(
            us_input,
            parameter="td_loss_rate",
            variation_pct=Decimal("20")
        )

        assert "low" in sensitivity
        assert "base" in sensitivity
        assert "high" in sensitivity

        assert sensitivity["low"]["total_emissions_kgco2e"] < base_result.total_emissions_kgco2e
        assert sensitivity["high"]["total_emissions_kgco2e"] > base_result.total_emissions_kgco2e

    def test_check_double_counting_with_scope2(self, engine, us_input):
        """Test checking for double counting with Scope 2."""
        result = engine.calculate(us_input)

        # T&D losses should flag potential double counting with Scope 2
        assert result.double_counting_risk is not None
        assert result.double_counting_risk.get("scope2_overlap") is not None

        # Should provide guidance
        assert "exclude_from_scope2" in result.double_counting_risk

    def test_validate_td_factor_valid(self, engine):
        """Test validation of valid T&D loss factor."""
        valid_factor = Decimal("0.05")  # 5%

        is_valid = engine.validate_td_factor(valid_factor)

        assert is_valid is True

    def test_validate_td_factor_negative(self, engine):
        """Test validation rejects negative T&D loss factor."""
        invalid_factor = Decimal("-0.05")

        with pytest.raises(ValidationError):
            engine.validate_td_factor(invalid_factor, raise_on_invalid=True)

    def test_zero_td_loss_returns_zero(self, engine):
        """Test zero T&D loss returns zero emissions."""
        # Mock country with zero losses (hypothetical)
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("100000"),
            country="ZERO",  # Hypothetical
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        with patch.object(engine, 'get_td_loss_factor', return_value=Decimal("0")):
            result = engine.calculate(input_data)

            assert result.energy_lost_kwh == Decimal("0")
            assert result.total_emissions_kgco2e == Decimal("0")

    def test_get_statistics(self, engine, us_input):
        """Test getting engine statistics."""
        # Perform some calculations
        engine.calculate(us_input)
        engine.calculate(us_input)

        stats = engine.get_statistics()

        assert stats["calculations_performed"] == 2
        assert stats["total_energy_lost_kwh"] > Decimal("0")
        assert stats["total_emissions_kgco2e"] > Decimal("0")

    def test_reset(self, engine, us_input):
        """Test resetting engine state."""
        # Perform calculation
        engine.calculate(us_input)

        # Reset
        engine.reset()

        stats = engine.get_statistics()
        assert stats["calculations_performed"] == 0
        assert stats["total_emissions_kgco2e"] == Decimal("0")

    def test_compare_loss_basis_delivered_vs_generated(self, engine):
        """Test comparing delivered vs generated loss basis."""
        consumption = Decimal("100000")

        delivered_input = TDLossInput(
            electricity_consumption_kwh=consumption,
            country="US",
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        generated_input = TDLossInput(
            electricity_consumption_kwh=consumption,
            country="US",
            loss_basis=LossBasis.GENERATED,
            reporting_period="2025-Q1"
        )

        delivered_result = engine.calculate(delivered_input)
        generated_result = engine.calculate(generated_input)

        # Generated basis should have lower energy lost
        # (same number represents generated, so delivered is lower)
        assert delivered_result.energy_lost_kwh > generated_result.energy_lost_kwh

    def test_high_consumption_high_losses(self, engine):
        """Test high consumption with high T&D losses."""
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("1000000"),  # 1 GWh
            country="IN",  # High T&D losses
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Should have significant losses
        assert result.energy_lost_kwh > Decimal("150000")  # >150 MWh lost
        assert result.total_emissions_kgco2e > Decimal("100000")  # >100 tonnes

    def test_low_consumption_accurate_precision(self, engine):
        """Test low consumption maintains precision."""
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("100"),  # Small amount
            country="US",
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        result = engine.calculate(input_data)

        # Should still calculate accurately
        assert result.energy_lost_kwh > Decimal("0")
        assert result.total_emissions_kgco2e > Decimal("0")

        # Should maintain precision
        assert result.total_emissions_kgco2e < Decimal("100")

    def test_egrid_subregion_overrides_country_default(self, engine):
        """Test eGRID subregion provides more specific data than country average."""
        # US average
        country_input = TDLossInput(
            electricity_consumption_kwh=Decimal("100000"),
            country="US",
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        # Specific eGRID subregion
        egrid_input = TDLossInput(
            electricity_consumption_kwh=Decimal("100000"),
            country="US",
            egrid_subregion="CAMX",  # California
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        country_result = engine.calculate(country_input)
        egrid_result = engine.calculate(egrid_input)

        # Should have different DQI (eGRID more specific)
        assert egrid_result.dqi_score > country_result.dqi_score

    def test_provenance_tracking_deterministic(self, engine, us_input):
        """Test provenance hash is deterministic."""
        result1 = engine.calculate(us_input)
        result2 = engine.calculate(us_input)

        # Same input → Same provenance hash
        assert result1.provenance_hash == result2.provenance_hash

    def test_metadata_fields_populated(self, engine, us_input):
        """Test metadata fields are populated correctly."""
        result = engine.calculate(us_input)

        assert result.calculation_timestamp is not None
        assert result.engine_version is not None
        assert result.data_sources is not None
        assert len(result.data_sources) > 0

    def test_export_to_dict(self, engine, us_input):
        """Test exporting result to dictionary."""
        result = engine.calculate(us_input)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "total_emissions_kgco2e" in result_dict
        assert "td_loss_rate" in result_dict
        assert "energy_lost_kwh" in result_dict

    def test_performance_batch_processing(self, engine, benchmark):
        """Test batch processing performance."""
        inputs = [
            TDLossInput(
                electricity_consumption_kwh=Decimal("100000"),
                country="US",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            )
            for _ in range(100)
        ]

        def run_batch():
            return engine.calculate_batch(inputs)

        results = benchmark(run_batch)

        assert len(results) == 100

    def test_error_handling_missing_country(self, engine):
        """Test error handling for missing country."""
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("100000"),
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
            # No country or facility_country
        )

        with pytest.raises(ValidationError, match="country"):
            engine.calculate(input_data)

    def test_error_handling_negative_consumption(self, engine):
        """Test error handling for negative consumption."""
        input_data = TDLossInput(
            electricity_consumption_kwh=Decimal("-100000"),
            country="US",
            loss_basis=LossBasis.DELIVERED,
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError, match="consumption"):
            engine.calculate(input_data)

    def test_generate_report(self, engine, us_input):
        """Test generating human-readable report."""
        result = engine.calculate(us_input)

        report = engine.generate_report(result)

        assert isinstance(report, str)
        assert "T&D Loss" in report
        assert "kWh" in report
        assert "kgCO2e" in report


# Integration Tests
class TestTDLossCalculatorIntegration:
    """Integration tests for TDLossCalculatorEngine."""

    @pytest.mark.integration
    def test_integration_with_scope2_calculation(self, engine):
        """Test integration with Scope 2 calculation to avoid double counting."""
        # This would test with actual Scope 2 agent
        pass

    @pytest.mark.integration
    def test_integration_with_grid_emission_factors(self, engine):
        """Test integration with grid emission factor database."""
        # This would test with actual EF database
        pass


# Performance Tests
class TestTDLossCalculatorPerformance:
    """Performance tests for TDLossCalculatorEngine."""

    @pytest.mark.performance
    def test_throughput_target(self, engine):
        """Test engine meets throughput target (1000 calculations/sec)."""
        num_records = 10000
        inputs = [
            TDLossInput(
                electricity_consumption_kwh=Decimal("100000"),
                country="US",
                loss_basis=LossBasis.DELIVERED,
                reporting_period="2025-Q1"
            )
            for _ in range(num_records)
        ]

        start_time = datetime.now()
        results = engine.calculate_batch(inputs)
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_records / duration_seconds

        assert throughput >= 1000  # Target: 1000 calculations/sec
        assert len(results) == num_records
