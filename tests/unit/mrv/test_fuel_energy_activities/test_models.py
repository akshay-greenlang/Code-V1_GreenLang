# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-016 Fuel & Energy Activities Agent models.

Tests all enums, constant tables, and Pydantic models for data validation,
serialization, and business logic.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any

from greenlang.fuel_energy_activities.models import (
    # Enums
    FuelType,
    ActivityType,
    CalculationMethod,
    ElectricitySource,
    QualityTier,
    RegulatoryFramework,

    # Constants
    WTT_FUEL_EMISSION_FACTORS,
    UPSTREAM_ELECTRICITY_FACTORS,
    TD_LOSS_FACTORS,
    EGRID_TD_LOSS_FACTORS,
    FUEL_HEATING_VALUES,
    FUEL_DENSITY_FACTORS,
    DQI_WEIGHTS,
    COVERAGE_THRESHOLDS,
    FRAMEWORK_DISCLOSURE_REQUIREMENTS,
    AGENT_ID,
    TABLE_PREFIX,

    # Input Models
    FuelConsumptionRecord,
    ElectricityConsumptionRecord,
    WTTEmissionFactor,
    UpstreamElectricityFactor,
    TDLossFactor,
    SupplierFuelData,

    # Output Models
    Activity3aResult,
    Activity3bResult,
    Activity3cResult,
    CalculationResult,
    FuelBreakdown,
    ElectricityBreakdown,
    TDLossBreakdown,
    GasBreakdown,

    # Compliance Models
    ComplianceCheckResult,
    FrameworkDisclosure,

    # DQI Models
    DQIAssessment,
    DQIScore,

    # Batch Models
    BatchRequest,
    BatchResult,
    YoYDecomposition,
)


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEnums:
    """Test all enum definitions."""

    def test_fuel_type_enum_values(self):
        """Test FuelType enum has all expected fuel types."""
        expected_fuels = [
            "NATURAL_GAS",
            "DIESEL",
            "GASOLINE",
            "FUEL_OIL",
            "LPG",
            "KEROSENE",
            "JET_FUEL",
            "COAL",
            "ANTHRACITE_COAL",
            "BITUMINOUS_COAL",
            "LIGNITE",
            "PETCOKE",
            "PROPANE",
            "BUTANE",
            "ETHANOL",
            "BIODIESEL",
            "METHANOL",
            "HYDROGEN",
            "BIOGAS",
            "RESIDUAL_FUEL_OIL",
            "WOOD",
            "WOOD_PELLETS",
            "WOOD_CHIPS",
            "MUNICIPAL_SOLID_WASTE",
            "OTHER"
        ]

        for fuel in expected_fuels:
            assert hasattr(FuelType, fuel), f"FuelType missing {fuel}"

    def test_activity_type_enum(self):
        """Test ActivityType enum has all Scope 3 Category 3 activities."""
        assert hasattr(ActivityType, "ACTIVITY_3A")
        assert hasattr(ActivityType, "ACTIVITY_3B")
        assert hasattr(ActivityType, "ACTIVITY_3C")

        assert ActivityType.ACTIVITY_3A.value == "activity_3a_fuel_wtt"
        assert ActivityType.ACTIVITY_3B.value == "activity_3b_electricity_upstream"
        assert ActivityType.ACTIVITY_3C.value == "activity_3c_td_losses"

    def test_calculation_method_enum(self):
        """Test CalculationMethod enum."""
        assert hasattr(CalculationMethod, "FUEL_BASED")
        assert hasattr(CalculationMethod, "LOCATION_BASED")
        assert hasattr(CalculationMethod, "MARKET_BASED")
        assert hasattr(CalculationMethod, "SUPPLIER_SPECIFIC")
        assert hasattr(CalculationMethod, "AVERAGE_DATA")

    def test_electricity_source_enum(self):
        """Test ElectricitySource enum."""
        assert hasattr(ElectricitySource, "GRID")
        assert hasattr(ElectricitySource, "ONSITE_SOLAR")
        assert hasattr(ElectricitySource, "ONSITE_WIND")
        assert hasattr(ElectricitySource, "PURCHASED_RENEWABLE")
        assert hasattr(ElectricitySource, "DISTRICT_HEATING")
        assert hasattr(ElectricitySource, "DISTRICT_COOLING")
        assert hasattr(ElectricitySource, "STEAM")
        assert hasattr(ElectricitySource, "OTHER")

    def test_quality_tier_enum(self):
        """Test QualityTier enum."""
        assert hasattr(QualityTier, "TIER_1")
        assert hasattr(QualityTier, "TIER_2")
        assert hasattr(QualityTier, "TIER_3")

    def test_regulatory_framework_enum(self):
        """Test RegulatoryFramework enum."""
        frameworks = [
            "GHG_PROTOCOL",
            "ISO_14064",
            "CSRD",
            "TCFD",
            "CDP",
            "SBTi",
            "GRI"
        ]

        for framework in frameworks:
            assert hasattr(RegulatoryFramework, framework)


# ============================================================================
# CONSTANT TABLE TESTS
# ============================================================================

class TestConstantTables:
    """Test all constant lookup tables."""

    def test_wtt_fuel_emission_factors_completeness(self):
        """Test WTT_FUEL_EMISSION_FACTORS has all 25 fuel types."""
        assert len(WTT_FUEL_EMISSION_FACTORS) == 25

        # Check key fuels are present
        key_fuels = [
            FuelType.NATURAL_GAS,
            FuelType.DIESEL,
            FuelType.GASOLINE,
            FuelType.COAL,
            FuelType.LPG,
            FuelType.JET_FUEL
        ]

        for fuel in key_fuels:
            assert fuel in WTT_FUEL_EMISSION_FACTORS

    def test_wtt_factors_all_decimal(self):
        """Test all WTT emission factors are Decimal type."""
        for fuel_type, factors in WTT_FUEL_EMISSION_FACTORS.items():
            assert isinstance(factors["co2_kg_per_unit"], Decimal)
            assert isinstance(factors["ch4_kg_per_unit"], Decimal)
            assert isinstance(factors["n2o_kg_per_unit"], Decimal)
            assert factors["co2_kg_per_unit"] >= 0
            assert factors["ch4_kg_per_unit"] >= 0
            assert factors["n2o_kg_per_unit"] >= 0

    def test_upstream_electricity_factors_countries(self):
        """Test UPSTREAM_ELECTRICITY_FACTORS has major countries."""
        key_countries = ["US", "GB", "DE", "FR", "CN", "IN", "JP"]

        for country in key_countries:
            assert country in UPSTREAM_ELECTRICITY_FACTORS

    def test_td_loss_factors_countries(self):
        """Test TD_LOSS_FACTORS has 50+ countries."""
        assert len(TD_LOSS_FACTORS) >= 50

        # Check key countries
        assert "US" in TD_LOSS_FACTORS
        assert "GB" in TD_LOSS_FACTORS
        assert "DE" in TD_LOSS_FACTORS

        # Validate percentage format
        for country, loss_pct in TD_LOSS_FACTORS.items():
            assert isinstance(loss_pct, Decimal)
            assert Decimal("0.0") <= loss_pct <= Decimal("20.0")  # Max 20% loss

    def test_egrid_td_loss_factors(self):
        """Test EGRID_TD_LOSS_FACTORS has 26 US subregions."""
        assert len(EGRID_TD_LOSS_FACTORS) == 26

        # Check key subregions
        key_subregions = ["NEWE", "CAMX", "ERCT", "MROW", "RFCE"]
        for subregion in key_subregions:
            assert subregion in EGRID_TD_LOSS_FACTORS

    def test_fuel_heating_values(self):
        """Test FUEL_HEATING_VALUES completeness."""
        assert len(FUEL_HEATING_VALUES) >= 20

        # Check key fuels
        assert FuelType.NATURAL_GAS in FUEL_HEATING_VALUES
        assert FuelType.DIESEL in FUEL_HEATING_VALUES
        assert FuelType.COAL in FUEL_HEATING_VALUES

        # Validate units
        for fuel, data in FUEL_HEATING_VALUES.items():
            assert "value_mj_per_unit" in data
            assert "unit" in data
            assert isinstance(data["value_mj_per_unit"], Decimal)
            assert data["value_mj_per_unit"] > 0

    def test_fuel_density_factors(self):
        """Test FUEL_DENSITY_FACTORS for liquid/gas fuels."""
        assert len(FUEL_DENSITY_FACTORS) >= 10

        # Check key liquid fuels
        assert FuelType.DIESEL in FUEL_DENSITY_FACTORS
        assert FuelType.GASOLINE in FUEL_DENSITY_FACTORS

        # Validate density
        for fuel, density in FUEL_DENSITY_FACTORS.items():
            assert isinstance(density, Decimal)
            assert density > 0

    def test_dqi_score_values(self):
        """Test DQI_WEIGHTS sum to 1.0."""
        total_weight = sum(DQI_WEIGHTS.values())
        assert total_weight == Decimal("1.0")

        # Check all dimensions present
        dimensions = ["completeness", "accuracy", "consistency", "timeliness", "reliability"]
        for dim in dimensions:
            assert dim in DQI_WEIGHTS
            assert DQI_WEIGHTS[dim] > 0

    def test_coverage_thresholds(self):
        """Test COVERAGE_THRESHOLDS."""
        assert "minimum_coverage_percentage" in COVERAGE_THRESHOLDS
        assert "high_coverage_threshold" in COVERAGE_THRESHOLDS

        assert isinstance(COVERAGE_THRESHOLDS["minimum_coverage_percentage"], Decimal)
        assert COVERAGE_THRESHOLDS["minimum_coverage_percentage"] >= Decimal("70.0")

    def test_framework_disclosures(self):
        """Test FRAMEWORK_DISCLOSURE_REQUIREMENTS."""
        frameworks = [
            RegulatoryFramework.GHG_PROTOCOL,
            RegulatoryFramework.ISO_14064,
            RegulatoryFramework.CSRD
        ]

        for framework in frameworks:
            assert framework in FRAMEWORK_DISCLOSURE_REQUIREMENTS
            requirements = FRAMEWORK_DISCLOSURE_REQUIREMENTS[framework]
            assert isinstance(requirements, list)
            assert len(requirements) > 0


# ============================================================================
# INPUT MODEL TESTS
# ============================================================================

class TestInputModels:
    """Test input Pydantic models."""

    def test_fuel_consumption_record_creation(self, sample_fuel_record):
        """Test FuelConsumptionRecord creation and validation."""
        assert sample_fuel_record.facility_id == "FAC-001"
        assert sample_fuel_record.fuel_type == FuelType.NATURAL_GAS
        assert sample_fuel_record.fuel_quantity == Decimal("10000.0")
        assert sample_fuel_record.fuel_quantity_unit == "m3"
        assert sample_fuel_record.activity_type == ActivityType.ACTIVITY_3A
        assert sample_fuel_record.data_quality_score == Decimal("0.85")

    def test_fuel_consumption_record_validation_negative_quantity(self):
        """Test FuelConsumptionRecord rejects negative quantity."""
        with pytest.raises(ValueError):
            FuelConsumptionRecord(
                facility_id="FAC-001",
                reporting_period="2024-Q1",
                fuel_type=FuelType.NATURAL_GAS,
                fuel_quantity=Decimal("-100.0"),  # Negative
                fuel_quantity_unit="m3",
                activity_type=ActivityType.ACTIVITY_3A,
                country="US",
                region="Northeast",
                sector="Manufacturing",
                calculation_method=CalculationMethod.FUEL_BASED,
                has_renewable_content=False,
                renewable_fraction=Decimal("0.0"),
                data_quality_score=Decimal("0.85")
            )

    def test_fuel_consumption_record_renewable_fraction_validation(self):
        """Test renewable_fraction must be between 0 and 1."""
        with pytest.raises(ValueError):
            FuelConsumptionRecord(
                facility_id="FAC-001",
                reporting_period="2024-Q1",
                fuel_type=FuelType.DIESEL,
                fuel_quantity=Decimal("1000.0"),
                fuel_quantity_unit="L",
                activity_type=ActivityType.ACTIVITY_3A,
                country="US",
                region="Northeast",
                sector="Transportation",
                calculation_method=CalculationMethod.FUEL_BASED,
                has_renewable_content=True,
                renewable_fraction=Decimal("1.5"),  # > 1.0
                data_quality_score=Decimal("0.85")
            )

    def test_electricity_consumption_record_creation(self, sample_electricity_record):
        """Test ElectricityConsumptionRecord creation."""
        assert sample_electricity_record.facility_id == "FAC-006"
        assert sample_electricity_record.electricity_quantity == Decimal("100000.0")
        assert sample_electricity_record.electricity_unit == "kWh"
        assert sample_electricity_record.activity_type == ActivityType.ACTIVITY_3B
        assert sample_electricity_record.country == "US"
        assert sample_electricity_record.egrid_subregion == "NEWE"

    def test_electricity_consumption_record_validation_negative_quantity(self):
        """Test ElectricityConsumptionRecord rejects negative quantity."""
        with pytest.raises(ValueError):
            ElectricityConsumptionRecord(
                facility_id="FAC-006",
                reporting_period="2024-Q1",
                electricity_quantity=Decimal("-50000.0"),  # Negative
                electricity_unit="kWh",
                activity_type=ActivityType.ACTIVITY_3B,
                country="US",
                region="Northeast",
                electricity_source=ElectricitySource.GRID,
                supplier_name="Utility",
                calculation_method=CalculationMethod.LOCATION_BASED,
                has_renewable_content=False,
                renewable_fraction=Decimal("0.0"),
                data_quality_score=Decimal("0.92")
            )

    def test_wtt_emission_factor_model(self, sample_wtt_factor):
        """Test WTTEmissionFactor model."""
        assert sample_wtt_factor.fuel_type == FuelType.NATURAL_GAS
        assert sample_wtt_factor.country == "US"
        assert sample_wtt_factor.wtt_co2_kg_per_unit == Decimal("0.185")
        assert sample_wtt_factor.quality_tier == QualityTier.TIER_2
        assert sample_wtt_factor.source == "GREET_2023"

    def test_upstream_electricity_factor_model(self, sample_upstream_ef):
        """Test UpstreamElectricityFactor model."""
        assert sample_upstream_ef.country == "US"
        assert sample_upstream_ef.egrid_subregion == "NEWE"
        assert sample_upstream_ef.upstream_co2_kg_per_kwh == Decimal("0.082")
        assert sample_upstream_ef.quality_tier == QualityTier.TIER_1

    def test_td_loss_factor_model(self, sample_td_loss_factor):
        """Test TDLossFactor model."""
        assert sample_td_loss_factor.country == "US"
        assert sample_td_loss_factor.egrid_subregion == "NEWE"
        assert sample_td_loss_factor.td_loss_percentage == Decimal("5.0")
        assert sample_td_loss_factor.source == "EIA_2023"

    def test_supplier_fuel_data_model(self, sample_supplier_data):
        """Test SupplierFuelData model."""
        assert sample_supplier_data.supplier_name == "Green_Fuel_Corp"
        assert sample_supplier_data.fuel_type == FuelType.DIESEL
        assert sample_supplier_data.has_epd is True
        assert sample_supplier_data.epd_reference == "EPD-GFC-2023-001"
        assert sample_supplier_data.verification_status == "third_party_verified"


# ============================================================================
# OUTPUT MODEL TESTS
# ============================================================================

class TestOutputModels:
    """Test output Pydantic models."""

    def test_fuel_breakdown_model(self):
        """Test FuelBreakdown model."""
        breakdown = FuelBreakdown(
            fuel_type=FuelType.NATURAL_GAS,
            fuel_quantity=Decimal("10000.0"),
            fuel_unit="m3",
            wtt_emissions_tco2e=Decimal("20.5"),
            co2_tco2e=Decimal("18.5"),
            ch4_tco2e=Decimal("1.8"),
            n2o_tco2e=Decimal("0.2"),
            calculation_method=CalculationMethod.FUEL_BASED,
            emission_factor_source="GREET_2023"
        )

        assert breakdown.fuel_type == FuelType.NATURAL_GAS
        assert breakdown.wtt_emissions_tco2e == Decimal("20.5")
        assert breakdown.co2_tco2e + breakdown.ch4_tco2e + breakdown.n2o_tco2e == breakdown.wtt_emissions_tco2e

    def test_electricity_breakdown_model(self):
        """Test ElectricityBreakdown model."""
        breakdown = ElectricityBreakdown(
            electricity_quantity=Decimal("100000.0"),
            electricity_unit="kWh",
            country="US",
            egrid_subregion="NEWE",
            upstream_emissions_tco2e=Decimal("8.2"),
            co2_tco2e=Decimal("7.8"),
            ch4_tco2e=Decimal("0.35"),
            n2o_tco2e=Decimal("0.05"),
            calculation_method=CalculationMethod.LOCATION_BASED,
            emission_factor_source="EPA_eGRID_2023"
        )

        assert breakdown.electricity_quantity == Decimal("100000.0")
        assert breakdown.upstream_emissions_tco2e == Decimal("8.2")

    def test_td_loss_breakdown_model(self):
        """Test TDLossBreakdown model."""
        breakdown = TDLossBreakdown(
            electricity_quantity=Decimal("100000.0"),
            electricity_unit="kWh",
            country="US",
            egrid_subregion="NEWE",
            td_loss_percentage=Decimal("5.0"),
            td_loss_quantity_kwh=Decimal("5000.0"),
            td_loss_emissions_tco2e=Decimal("2.5"),
            co2_tco2e=Decimal("2.3"),
            ch4_tco2e=Decimal("0.18"),
            n2o_tco2e=Decimal("0.02"),
            calculation_method=CalculationMethod.LOCATION_BASED,
            emission_factor_source="EPA_eGRID_2023"
        )

        assert breakdown.td_loss_percentage == Decimal("5.0")
        assert breakdown.td_loss_quantity_kwh == Decimal("5000.0")

    def test_gas_breakdown_model(self):
        """Test GasBreakdown model."""
        breakdown = GasBreakdown(
            co2_tco2e=Decimal("28.6"),
            ch4_tco2e=Decimal("2.33"),
            n2o_tco2e=Decimal("0.27"),
            ch4_percentage=Decimal("7.47"),
            n2o_percentage=Decimal("0.87")
        )

        assert breakdown.co2_tco2e == Decimal("28.6")
        assert breakdown.ch4_tco2e == Decimal("2.33")
        assert breakdown.n2o_tco2e == Decimal("0.27")

        # Validate percentages
        total = breakdown.co2_tco2e + breakdown.ch4_tco2e + breakdown.n2o_tco2e
        ch4_pct = (breakdown.ch4_tco2e / total * 100).quantize(Decimal("0.01"))
        assert ch4_pct == breakdown.ch4_percentage

    def test_activity_3a_result_model(self, sample_activity_3a_result):
        """Test Activity3aResult model."""
        assert sample_activity_3a_result.activity_type == ActivityType.ACTIVITY_3A
        assert sample_activity_3a_result.total_wtt_emissions_tco2e == Decimal("20.5")
        assert len(sample_activity_3a_result.fuel_breakdown) == 1
        assert sample_activity_3a_result.provenance_hash == "abc123def456"

    def test_activity_3b_result_model(self, sample_activity_3b_result):
        """Test Activity3bResult model."""
        assert sample_activity_3b_result.activity_type == ActivityType.ACTIVITY_3B
        assert sample_activity_3b_result.total_upstream_emissions_tco2e == Decimal("8.2")
        assert len(sample_activity_3b_result.electricity_breakdown) == 1

    def test_activity_3c_result_model(self, sample_activity_3c_result):
        """Test Activity3cResult model."""
        assert sample_activity_3c_result.activity_type == ActivityType.ACTIVITY_3C
        assert sample_activity_3c_result.total_td_loss_emissions_tco2e == Decimal("2.5")
        assert len(sample_activity_3c_result.td_loss_breakdown) == 1

    def test_calculation_result_model(self, sample_calculation_result):
        """Test CalculationResult model."""
        assert sample_calculation_result.total_scope3_category3_emissions_tco2e == Decimal("31.2")
        assert sample_calculation_result.activity_3a_emissions_tco2e == Decimal("20.5")
        assert sample_calculation_result.activity_3b_emissions_tco2e == Decimal("8.2")
        assert sample_calculation_result.activity_3c_emissions_tco2e == Decimal("2.5")

        # Validate sum
        total = (
            sample_calculation_result.activity_3a_emissions_tco2e +
            sample_calculation_result.activity_3b_emissions_tco2e +
            sample_calculation_result.activity_3c_emissions_tco2e
        )
        assert total == sample_calculation_result.total_scope3_category3_emissions_tco2e

    def test_calculation_result_gas_breakdown(self, sample_calculation_result):
        """Test CalculationResult has correct gas breakdown."""
        assert sample_calculation_result.total_co2_tco2e == Decimal("28.6")
        assert sample_calculation_result.total_ch4_tco2e == Decimal("2.33")
        assert sample_calculation_result.total_n2o_tco2e == Decimal("0.27")

        assert sample_calculation_result.gas_breakdown.co2_tco2e == Decimal("28.6")
        assert sample_calculation_result.gas_breakdown.ch4_tco2e == Decimal("2.33")
        assert sample_calculation_result.gas_breakdown.n2o_tco2e == Decimal("0.27")


# ============================================================================
# COMPLIANCE MODEL TESTS
# ============================================================================

class TestComplianceModels:
    """Test compliance-related models."""

    def test_framework_disclosure_model(self):
        """Test FrameworkDisclosure model."""
        disclosure = FrameworkDisclosure(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            compliant=True,
            disclosure_requirements=[
                "Activity 3a WTT emissions calculated",
                "Gas-level breakdown provided"
            ],
            missing_requirements=[],
            recommendations=["Consider supplier-specific data"],
            compliance_percentage=Decimal("100.0")
        )

        assert disclosure.framework == RegulatoryFramework.GHG_PROTOCOL
        assert disclosure.compliant is True
        assert len(disclosure.disclosure_requirements) == 2
        assert disclosure.compliance_percentage == Decimal("100.0")

    def test_compliance_check_result_model(self, sample_compliance_result):
        """Test ComplianceCheckResult model."""
        assert sample_compliance_result.overall_compliant is True
        assert sample_compliance_result.compliance_score == Decimal("0.95")
        assert len(sample_compliance_result.frameworks_checked) == 3
        assert sample_compliance_result.checks_passed == 12
        assert sample_compliance_result.checks_failed == 0

    def test_compliance_check_result_framework_results(self, sample_compliance_result):
        """Test ComplianceCheckResult framework_results mapping."""
        assert RegulatoryFramework.GHG_PROTOCOL in sample_compliance_result.framework_results

        ghg_disclosure = sample_compliance_result.framework_results[RegulatoryFramework.GHG_PROTOCOL]
        assert ghg_disclosure.compliant is True
        assert ghg_disclosure.compliance_percentage == Decimal("100.0")


# ============================================================================
# DQI MODEL TESTS
# ============================================================================

class TestDQIModels:
    """Test DQI (Data Quality Indicator) models."""

    def test_dqi_score_model(self):
        """Test DQIScore model."""
        score = DQIScore(
            dimension="completeness",
            score=Decimal("0.95"),
            weight=Decimal("0.20"),
            weighted_score=Decimal("0.19"),
            assessment="High - All required fields present",
            issues=[]
        )

        assert score.dimension == "completeness"
        assert score.score == Decimal("0.95")
        assert score.weight == Decimal("0.20")
        assert score.weighted_score == Decimal("0.19")

    def test_dqi_assessment_model(self, sample_dqi_assessment):
        """Test DQIAssessment model."""
        assert sample_dqi_assessment.overall_score == Decimal("0.89")
        assert sample_dqi_assessment.tier == QualityTier.TIER_2
        assert len(sample_dqi_assessment.dimension_scores) == 5

    def test_dqi_assessment_dimension_scores(self, sample_dqi_assessment):
        """Test DQIAssessment has all 5 dimensions."""
        dimensions = ["completeness", "accuracy", "consistency", "timeliness", "reliability"]

        for dim in dimensions:
            assert dim in sample_dqi_assessment.dimension_scores
            assert isinstance(sample_dqi_assessment.dimension_scores[dim], DQIScore)

    def test_dqi_assessment_weighted_scores_sum(self, sample_dqi_assessment):
        """Test DQIAssessment weighted scores sum to overall score."""
        total_weighted = sum(
            score.weighted_score
            for score in sample_dqi_assessment.dimension_scores.values()
        )

        assert abs(total_weighted - sample_dqi_assessment.overall_score) < Decimal("0.01")


# ============================================================================
# BATCH MODEL TESTS
# ============================================================================

class TestBatchModels:
    """Test batch processing models."""

    def test_batch_request_validation(self, sample_batch_request):
        """Test BatchRequest model."""
        assert sample_batch_request.batch_id == "BATCH-2024-Q1-001"
        assert len(sample_batch_request.fuel_records) == 1
        assert len(sample_batch_request.electricity_records) == 1
        assert sample_batch_request.include_uncertainty is True
        assert sample_batch_request.include_dqi is True

    def test_batch_request_compliance_frameworks(self, sample_batch_request):
        """Test BatchRequest compliance_frameworks."""
        assert RegulatoryFramework.GHG_PROTOCOL in sample_batch_request.compliance_frameworks
        assert RegulatoryFramework.ISO_14064 in sample_batch_request.compliance_frameworks

    def test_yoy_decomposition_validation(self):
        """Test YoYDecomposition model."""
        decomp = YoYDecomposition(
            current_period="2024-Q1",
            prior_period="2023-Q1",
            total_change_tco2e=Decimal("5.5"),
            total_change_percentage=Decimal("10.5"),
            activity_3a_change_tco2e=Decimal("2.5"),
            activity_3a_change_percentage=Decimal("12.2"),
            activity_3b_change_tco2e=Decimal("2.0"),
            activity_3b_change_percentage=Decimal("24.4"),
            activity_3c_change_tco2e=Decimal("1.0"),
            activity_3c_change_percentage=Decimal("40.0"),
            drivers=[
                "Increased natural gas consumption (+15%)",
                "Grid decarbonization (-5%)"
            ],
            metadata={"analysis_date": "2024-04-15"}
        )

        assert decomp.total_change_tco2e == Decimal("5.5")
        assert decomp.total_change_percentage == Decimal("10.5")
        assert len(decomp.drivers) == 2


# ============================================================================
# MODEL IMMUTABILITY TESTS
# ============================================================================

class TestModelImmutability:
    """Test model immutability (frozen=True)."""

    def test_frozen_models_immutable(self, sample_fuel_record):
        """Test that frozen models cannot be modified."""
        with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
            sample_fuel_record.fuel_quantity = Decimal("99999.0")


# ============================================================================
# CONSTANT TESTS
# ============================================================================

class TestConstants:
    """Test module-level constants."""

    def test_agent_id_constant(self):
        """Test AGENT_ID constant."""
        assert AGENT_ID == "GL-MRV-S3-003"

    def test_table_prefix_constant(self):
        """Test TABLE_PREFIX constant."""
        assert TABLE_PREFIX == "gl_fea_"
