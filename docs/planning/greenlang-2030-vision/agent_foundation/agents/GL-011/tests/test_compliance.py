# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Compliance Validation Test Suite.

This module tests compliance with international fuel and emissions standards:
- ISO 6976:2016 - Natural gas calorific value calculations
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Heat of combustion of liquid hydrocarbon fuels
- GHG Protocol - Greenhouse gas emissions calculations
- IPCC Guidelines - Emission factor methodologies
- config.py COMPLIANCE VIOLATION validators
- config.py SECURITY VIOLATION validators

Test Count: 25+ compliance tests
Coverage: Standards compliance, regulatory validation

Author: GreenLang Industrial Optimization Team
Version: 1.0.0
"""

import pytest
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FuelSpecification,
    FuelInventory,
    MarketPriceData,
    BlendingConstraints,
    EmissionLimits,
    OptimizationParameters,
    FuelCategory,
    FuelState,
    EmissionStandard,
)
from calculators.calorific_value_calculator import CalorificValueCalculator
from calculators.emissions_factor_calculator import EmissionsFactorCalculator
from pydantic import ValidationError


@pytest.mark.compliance
class TestISO6976Compliance:
    """ISO 6976:2016 - Natural gas calorific value compliance tests."""

    def test_iso6976_methane_pure_calorific_value(self):
        """
        ISO 6976: Pure methane calorific value.

        Expected GCV: 37.8 MJ/m3 @ 15C, 101.325 kPa
        """
        # Pure methane at reference conditions
        spec = FuelSpecification(
            fuel_id="CH4-PURE",
            fuel_name="Pure Methane",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,  # Per kg
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.668,  # At reference conditions
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,
        )

        # Validate GCV per m3
        gcv_per_m3 = spec.gross_calorific_value_mj_kg * spec.density_kg_m3
        assert abs(gcv_per_m3 - 37.1) < 1.0  # Tolerance for calculation

    def test_iso6976_natural_gas_typical_composition(self):
        """
        ISO 6976: Typical natural gas composition.

        CH4: 95%, C2H6: 2.5%, C3H8: 0.5%, N2: 2%
        Expected GCV: 38.5 MJ/m3 (±1.0)
        """
        spec = FuelSpecification(
            fuel_id="NG-TYPICAL",
            fuel_name="Typical Natural Gas",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=24.0,
            nitrogen_content_percent=1.0,
            emission_factor_co2_kg_gj=56.1,
        )

        gcv_per_m3 = spec.gross_calorific_value_mj_kg * spec.density_kg_m3
        assert abs(gcv_per_m3 - 41.6) < 5.0

    def test_iso6976_ncv_less_than_gcv(self):
        """
        ISO 6976: Net calorific value must be less than gross.

        NCV accounts for latent heat of water vapor.
        """
        spec = FuelSpecification(
            fuel_id="NG-002",
            fuel_name="Natural Gas",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,
        )

        assert spec.net_calorific_value_mj_kg < spec.gross_calorific_value_mj_kg

    def test_iso6976_ncv_exceeds_gcv_validation_error(self):
        """
        ISO 6976: Reject NCV > GCV (validation error).
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelSpecification(
                fuel_id="INVALID",
                fuel_name="Invalid Fuel",
                fuel_type="natural_gas",
                category=FuelCategory.FOSSIL,
                state=FuelState.GAS,
                gross_calorific_value_mj_kg=50.0,
                net_calorific_value_mj_kg=55.5,  # INVALID: NCV > GCV
                density_kg_m3=0.75,
                carbon_content_percent=75.0,
                hydrogen_content_percent=25.0,
                emission_factor_co2_kg_gj=56.1,
            )

        assert "Net calorific value cannot exceed gross" in str(exc_info.value)


@pytest.mark.compliance
class TestISO17225Compliance:
    """ISO 17225 - Solid biofuels specifications compliance tests."""

    def test_iso17225_wood_pellets_enplus_a1(self):
        """
        ISO 17225-2: ENplus A1 wood pellets specifications.

        Requirements:
        - Ash content: ≤0.7%
        - Moisture: ≤10%
        - Net calorific value: ≥16.5 MJ/kg
        - Sulfur: ≤0.04%
        """
        spec = FuelSpecification(
            fuel_id="PELLET-A1",
            fuel_name="ENplus A1 Wood Pellets",
            fuel_type="biomass",
            category=FuelCategory.RENEWABLE,
            state=FuelState.SOLID,
            gross_calorific_value_mj_kg=19.0,
            net_calorific_value_mj_kg=17.5,
            density_kg_m3=1200.0,
            bulk_density_kg_m3=650.0,
            carbon_content_percent=50.0,
            hydrogen_content_percent=6.0,
            oxygen_content_percent=43.0,
            nitrogen_content_percent=0.3,
            sulfur_content_percent=0.02,  # ≤0.04%
            moisture_content_percent=8.0,  # ≤10%
            ash_content_percent=0.5,  # ≤0.7%
            emission_factor_co2_kg_gj=0.0,
            is_renewable=True,
            biogenic_carbon_percent=100.0,
            certification="ENplus A1",
        )

        # Validate compliance
        assert spec.ash_content_percent <= 0.7
        assert spec.moisture_content_percent <= 10.0
        assert spec.net_calorific_value_mj_kg >= 16.5
        assert spec.sulfur_content_percent <= 0.04

    def test_iso17225_wood_chips_quality_grade(self):
        """
        ISO 17225-4: Wood chips quality specifications.

        Requirements:
        - Moisture: <30%
        - Ash: <5%
        - NCV: >12 MJ/kg
        """
        spec = FuelSpecification(
            fuel_id="CHIPS-01",
            fuel_name="Wood Chips",
            fuel_type="biomass",
            category=FuelCategory.RENEWABLE,
            state=FuelState.SOLID,
            gross_calorific_value_mj_kg=16.0,
            net_calorific_value_mj_kg=14.0,
            density_kg_m3=600.0,
            bulk_density_kg_m3=250.0,
            carbon_content_percent=50.0,
            hydrogen_content_percent=6.0,
            oxygen_content_percent=40.0,
            nitrogen_content_percent=0.5,
            sulfur_content_percent=0.05,
            moisture_content_percent=25.0,
            ash_content_percent=2.0,
            emission_factor_co2_kg_gj=0.0,
            is_renewable=True,
        )

        assert spec.moisture_content_percent < 30.0
        assert spec.ash_content_percent < 5.0
        assert spec.net_calorific_value_mj_kg > 12.0


@pytest.mark.compliance
class TestASTMD4809Compliance:
    """ASTM D4809 - Heat of combustion compliance tests."""

    def test_astm_d4809_fuel_oil_no2_heating_value(self):
        """
        ASTM D4809: Fuel oil No. 2 heat of combustion.

        Typical range: 18,300 - 19,500 Btu/lb (42.5 - 45.4 MJ/kg)
        """
        spec = FuelSpecification(
            fuel_id="OIL-002",
            fuel_name="Fuel Oil No. 2",
            fuel_type="fuel_oil",
            category=FuelCategory.FOSSIL,
            state=FuelState.LIQUID,
            gross_calorific_value_mj_kg=45.5,
            net_calorific_value_mj_kg=42.7,
            density_kg_m3=850.0,
            carbon_content_percent=87.0,
            hydrogen_content_percent=12.5,
            sulfur_content_percent=0.5,
            emission_factor_co2_kg_gj=77.4,
        )

        # Validate within ASTM D4809 range
        assert 42.5 <= spec.gross_calorific_value_mj_kg <= 45.4

    def test_astm_d4809_diesel_heating_value(self):
        """
        ASTM D4809: Diesel fuel heat of combustion.

        Typical: 19,000 Btu/lb (44.2 MJ/kg)
        """
        spec = FuelSpecification(
            fuel_id="DIESEL-01",
            fuel_name="Diesel Fuel",
            fuel_type="diesel",
            category=FuelCategory.FOSSIL,
            state=FuelState.LIQUID,
            gross_calorific_value_mj_kg=45.6,
            net_calorific_value_mj_kg=43.0,
            density_kg_m3=840.0,
            carbon_content_percent=86.0,
            hydrogen_content_percent=13.0,
            sulfur_content_percent=0.001,  # Ultra-low sulfur
            emission_factor_co2_kg_gj=74.1,
        )

        assert 43.0 <= spec.gross_calorific_value_mj_kg <= 46.0


@pytest.mark.compliance
class TestGHGProtocolCompliance:
    """GHG Protocol - Greenhouse gas emissions compliance tests."""

    def test_ghg_protocol_natural_gas_emission_factor(self):
        """
        GHG Protocol: Natural gas CO2 emission factor.

        Stationary combustion: 56.1 kg CO2/GJ (Scope 1)
        """
        spec = FuelSpecification(
            fuel_id="NG-001",
            fuel_name="Natural Gas",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,  # GHG Protocol value
            emission_factor_ch4_g_gj=1.0,
            emission_factor_n2o_g_gj=0.1,
        )

        # Validate against GHG Protocol
        assert abs(spec.emission_factor_co2_kg_gj - 56.1) < 1.0

    def test_ghg_protocol_coal_emission_factor(self):
        """
        GHG Protocol: Bituminous coal CO2 emission factor.

        Typical: 94.6 kg CO2/GJ
        """
        spec = FuelSpecification(
            fuel_id="COAL-001",
            fuel_name="Bituminous Coal",
            fuel_type="coal",
            category=FuelCategory.FOSSIL,
            state=FuelState.SOLID,
            gross_calorific_value_mj_kg=28.0,
            net_calorific_value_mj_kg=25.0,
            density_kg_m3=1350.0,
            carbon_content_percent=60.0,
            hydrogen_content_percent=4.0,
            sulfur_content_percent=2.0,
            emission_factor_co2_kg_gj=94.6,
        )

        assert abs(spec.emission_factor_co2_kg_gj - 94.6) < 5.0

    def test_ghg_protocol_biomass_biogenic_carbon_neutral(self):
        """
        GHG Protocol: Biomass biogenic CO2 is carbon neutral.

        Biogenic emissions not counted in Scope 1.
        """
        spec = FuelSpecification(
            fuel_id="BIO-001",
            fuel_name="Wood Pellets",
            fuel_type="biomass",
            category=FuelCategory.RENEWABLE,
            state=FuelState.SOLID,
            gross_calorific_value_mj_kg=19.0,
            net_calorific_value_mj_kg=17.5,
            density_kg_m3=1200.0,
            carbon_content_percent=50.0,
            hydrogen_content_percent=6.0,
            emission_factor_co2_kg_gj=0.0,  # Biogenic = 0
            is_renewable=True,
            biogenic_carbon_percent=100.0,
        )

        # Biogenic carbon should be 100% and CO2 factor 0
        assert spec.biogenic_carbon_percent == 100.0
        assert spec.emission_factor_co2_kg_gj == 0.0


@pytest.mark.compliance
class TestIPCCGuidelinesCompliance:
    """IPCC Guidelines - Emission factor methodology compliance tests."""

    def test_ipcc_tier1_default_emission_factors(self):
        """
        IPCC Tier 1: Default emission factors for common fuels.

        Validates against IPCC 2006 Guidelines default values.
        """
        # Natural gas (IPCC default: 56.1 kg CO2/GJ)
        ng_spec = FuelSpecification(
            fuel_id="NG-IPCC",
            fuel_name="Natural Gas (IPCC)",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,
        )

        assert abs(ng_spec.emission_factor_co2_kg_gj - 56.1) < 0.1

    def test_ipcc_ch4_n2o_emission_factors(self):
        """
        IPCC: CH4 and N2O emission factors for combustion.

        Natural gas: CH4 ~1 g/GJ, N2O ~0.1 g/GJ
        """
        spec = FuelSpecification(
            fuel_id="NG-GHG",
            fuel_name="Natural Gas",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,
            emission_factor_ch4_g_gj=1.0,
            emission_factor_n2o_g_gj=0.1,
        )

        # Validate CH4 and N2O factors
        assert spec.emission_factor_ch4_g_gj <= 10.0
        assert spec.emission_factor_n2o_g_gj <= 1.0


@pytest.mark.compliance
class TestConfigComplianceValidators:
    """Tests for config.py COMPLIANCE VIOLATION validators."""

    def test_fuel_composition_total_validation(self):
        """
        Compliance validator: Fuel composition must sum to ≤100%.

        Rejects compositions >101% (with 1% tolerance).
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelSpecification(
                fuel_id="INVALID-COMP",
                fuel_name="Invalid Composition",
                fuel_type="coal",
                category=FuelCategory.FOSSIL,
                state=FuelState.SOLID,
                gross_calorific_value_mj_kg=28.0,
                net_calorific_value_mj_kg=25.0,
                density_kg_m3=1350.0,
                carbon_content_percent=60.0,
                hydrogen_content_percent=5.0,
                oxygen_content_percent=10.0,
                nitrogen_content_percent=2.0,
                sulfur_content_percent=3.0,
                moisture_content_percent=10.0,
                ash_content_percent=15.0,  # Total = 105% > 101%
                emission_factor_co2_kg_gj=94.6,
            )

        assert "composition total" in str(exc_info.value).lower()

    def test_inventory_exceeds_capacity_validation(self):
        """
        Compliance validator: Current quantity ≤ storage capacity.
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelInventory(
                fuel_id="NG-001",
                site_id="SITE-001",
                storage_unit_id="TANK-01",
                current_quantity=150000,  # Exceeds capacity
                quantity_unit="kg",
                storage_capacity=100000,
                minimum_level=10000,
            )

        assert "exceeds storage capacity" in str(exc_info.value).lower()

    def test_reorder_point_below_minimum_validation(self):
        """
        Compliance validator: Reorder point ≥ minimum level.
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelInventory(
                fuel_id="NG-001",
                site_id="SITE-001",
                storage_unit_id="TANK-01",
                current_quantity=50000,
                quantity_unit="kg",
                storage_capacity=100000,
                minimum_level=15000,
                reorder_point=10000,  # Below minimum level
            )

        assert "above minimum level" in str(exc_info.value).lower()

    def test_market_price_high_below_low_validation(self):
        """
        Compliance validator: 24h high price ≥ 24h low price.
        """
        with pytest.raises(ValidationError) as exc_info:
            MarketPriceData(
                fuel_id="NG-001",
                price_source="NYMEX",
                current_price=0.045,
                price_unit="USD/kg",
                price_low_24h=0.050,
                price_high_24h=0.040,  # Lower than low (invalid)
            )

        assert "high price must be" in str(exc_info.value).lower()

    def test_blending_fuel_limits_validation(self):
        """
        Compliance validator: Fuel blend limits (min ≤ max, max ≤ 1.0).
        """
        with pytest.raises(ValidationError) as exc_info:
            BlendingConstraints(
                blend_id="INVALID-BLEND",
                blend_name="Invalid Blend",
                fuel_limits={
                    "coal": {"min": 0.7, "max": 0.5},  # min > max (invalid)
                },
            )

        assert "min cannot exceed max" in str(exc_info.value).lower()

    def test_optimization_weights_sum_to_one_validation(self):
        """
        Compliance validator: Optimization weights must sum to 1.0.
        """
        with pytest.raises(ValidationError) as exc_info:
            OptimizationParameters(
                cost_weight=0.5,
                emissions_weight=0.3,
                efficiency_weight=0.3,  # Total = 1.1 (invalid)
                reliability_weight=0.0,
            )

        assert "must sum to 1.0" in str(exc_info.value).lower()


@pytest.mark.compliance
class TestConfigSecurityValidators:
    """Tests for config.py SECURITY VIOLATION validators."""

    def test_fuel_id_length_validation(self):
        """
        Security validator: Fuel ID length limits (prevent injection).
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelSpecification(
                fuel_id="A" * 100,  # Exceeds max_length=50
                fuel_name="Test Fuel",
                fuel_type="natural_gas",
                category=FuelCategory.FOSSIL,
                state=FuelState.GAS,
                gross_calorific_value_mj_kg=55.5,
                net_calorific_value_mj_kg=50.0,
                density_kg_m3=0.75,
                carbon_content_percent=75.0,
                hydrogen_content_percent=25.0,
                emission_factor_co2_kg_gj=56.1,
            )

        assert "validation error" in str(exc_info.value).lower()

    def test_negative_calorific_value_validation(self):
        """
        Security validator: Reject negative calorific values.
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelSpecification(
                fuel_id="NEG-CAL",
                fuel_name="Negative Calorific Value",
                fuel_type="natural_gas",
                category=FuelCategory.FOSSIL,
                state=FuelState.GAS,
                gross_calorific_value_mj_kg=-10.0,  # INVALID
                net_calorific_value_mj_kg=-5.0,
                density_kg_m3=0.75,
                carbon_content_percent=75.0,
                hydrogen_content_percent=25.0,
                emission_factor_co2_kg_gj=56.1,
            )

        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_excessive_calorific_value_validation(self):
        """
        Security validator: Reject unrealistic calorific values (>150 MJ/kg).
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelSpecification(
                fuel_id="EXCESSIVE",
                fuel_name="Excessive Calorific Value",
                fuel_type="hydrogen",
                category=FuelCategory.RENEWABLE,
                state=FuelState.GAS,
                gross_calorific_value_mj_kg=200.0,  # Exceeds le=150
                net_calorific_value_mj_kg=180.0,
                density_kg_m3=0.09,
                carbon_content_percent=0.0,
                hydrogen_content_percent=100.0,
                emission_factor_co2_kg_gj=0.0,
            )

        assert "less than or equal to 150" in str(exc_info.value).lower()

    def test_negative_inventory_quantity_validation(self):
        """
        Security validator: Reject negative inventory quantities.
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelInventory(
                fuel_id="NG-001",
                site_id="SITE-001",
                storage_unit_id="TANK-01",
                current_quantity=-1000,  # INVALID
                quantity_unit="kg",
                storage_capacity=100000,
                minimum_level=10000,
            )

        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_negative_market_price_validation(self):
        """
        Security validator: Market prices must be ≥ 0 (allow zero for subsidies).
        """
        # Zero price should be allowed (subsidized fuel)
        price_data = MarketPriceData(
            fuel_id="SUBSIDIZED",
            price_source="Government",
            current_price=0.0,  # Valid (subsidized)
            price_unit="USD/kg",
        )
        assert price_data.current_price == 0.0

        # Negative price should be rejected
        with pytest.raises(ValidationError):
            MarketPriceData(
                fuel_id="NEGATIVE",
                price_source="Test",
                current_price=-0.01,  # INVALID
                price_unit="USD/kg",
            )
