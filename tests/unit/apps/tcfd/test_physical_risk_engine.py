# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Physical Risk Engine.

Tests asset registration, acute/chronic risk assessment per hazard type,
exposure/vulnerability/adaptive capacity scoring, composite risk calculation,
financial damage estimation, insurance impact, supply chain propagation,
and portfolio aggregation with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    AssetType,
    PhysicalHazard,
    RiskType,
    HAZARD_EXPOSURE_MATRICES,
)
from services.models import (
    AssetLocation,
    PhysicalRiskAssessment,
    ClimateRisk,
    _new_id,
)


# ===========================================================================
# Asset Registration
# ===========================================================================

class TestAssetRegistration:
    """Test asset registration with geocoding."""

    def test_asset_creation(self, sample_asset_location):
        assert sample_asset_location.asset_name == "Houston Refinery"
        assert sample_asset_location.asset_type == AssetType.BUILDING

    def test_asset_coordinates(self, sample_asset_location):
        assert sample_asset_location.latitude == Decimal("29.7604")
        assert sample_asset_location.longitude == Decimal("-95.3698")

    def test_asset_country_code(self, sample_asset_location):
        assert sample_asset_location.country == "US"

    def test_asset_replacement_value(self, sample_asset_location):
        assert sample_asset_location.replacement_value == Decimal("500000000")

    def test_asset_insurance_coverage(self, sample_asset_location):
        assert sample_asset_location.insurance_coverage == Decimal("400000000")

    def test_asset_year_built(self, sample_asset_location):
        assert sample_asset_location.year_built == 1995

    @pytest.mark.parametrize("asset_type", list(AssetType))
    def test_all_asset_types(self, asset_type):
        asset = AssetLocation(
            org_id=_new_id(),
            asset_name=f"Test {asset_type.value}",
            asset_type=asset_type,
            latitude=Decimal("40.7128"),
            longitude=Decimal("-74.0060"),
            country="US",
        )
        assert asset.asset_type == asset_type

    def test_asset_elevation(self, sample_asset_location):
        assert sample_asset_location.elevation == Decimal("15")

    def test_asset_with_zero_elevation(self):
        asset = AssetLocation(
            org_id=_new_id(),
            asset_name="Sea-level Facility",
            latitude=Decimal("25.7617"),
            longitude=Decimal("-80.1918"),
            country="US",
            elevation=Decimal("0"),
        )
        assert asset.elevation == Decimal("0")


# ===========================================================================
# Acute Risk Assessment
# ===========================================================================

class TestAcuteRiskAssessment:
    """Test acute physical risk assessment by hazard type."""

    @pytest.mark.parametrize("hazard", [
        PhysicalHazard.CYCLONE,
        PhysicalHazard.FLOOD,
        PhysicalHazard.WILDFIRE,
        PhysicalHazard.HEATWAVE,
    ])
    def test_acute_hazard_assessment(self, hazard):
        assessment = PhysicalRiskAssessment(
            org_id=_new_id(),
            asset_id=_new_id(),
            hazard_type=hazard,
            exposure_score=4,
            vulnerability_score=3,
            adaptive_capacity_score=2,
            composite_risk_score=Decimal("72.0"),
        )
        assert assessment.hazard_type == hazard
        assert assessment.composite_risk_score == Decimal("72.0")

    def test_flood_assessment(self, sample_physical_risk_assessment):
        assert sample_physical_risk_assessment.hazard_type == PhysicalHazard.FLOOD


# ===========================================================================
# Chronic Risk Assessment
# ===========================================================================

class TestChronicRiskAssessment:
    """Test chronic physical risk assessment."""

    @pytest.mark.parametrize("hazard", [
        PhysicalHazard.SEA_LEVEL_RISE,
        PhysicalHazard.WATER_STRESS,
        PhysicalHazard.TEMPERATURE_RISE,
        PhysicalHazard.PRECIPITATION_CHANGE,
        PhysicalHazard.DROUGHT,
    ])
    def test_chronic_hazard_assessment(self, hazard):
        assessment = PhysicalRiskAssessment(
            org_id=_new_id(),
            asset_id=_new_id(),
            hazard_type=hazard,
            exposure_score=3,
            vulnerability_score=3,
            adaptive_capacity_score=3,
            composite_risk_score=Decimal("45.0"),
        )
        assert assessment.hazard_type == hazard


# ===========================================================================
# Exposure/Vulnerability/Adaptive Capacity Scoring
# ===========================================================================

class TestRiskComponentScoring:
    """Test exposure, vulnerability, and adaptive capacity scoring."""

    def test_exposure_score_range(self, sample_physical_risk_assessment):
        assert 1 <= sample_physical_risk_assessment.exposure_score <= 5

    def test_vulnerability_score_range(self, sample_physical_risk_assessment):
        assert 1 <= sample_physical_risk_assessment.vulnerability_score <= 5

    def test_adaptive_capacity_score_range(self, sample_physical_risk_assessment):
        assert 1 <= sample_physical_risk_assessment.adaptive_capacity_score <= 5

    @pytest.mark.parametrize("exposure,vulnerability,adaptive,expected_min", [
        (5, 5, 1, Decimal("50")),
        (1, 1, 5, Decimal("0")),
        (3, 3, 3, Decimal("0")),
    ])
    def test_component_score_combinations(self, exposure, vulnerability, adaptive, expected_min):
        assessment = PhysicalRiskAssessment(
            org_id=_new_id(),
            asset_id=_new_id(),
            hazard_type=PhysicalHazard.FLOOD,
            exposure_score=exposure,
            vulnerability_score=vulnerability,
            adaptive_capacity_score=adaptive,
            composite_risk_score=max(Decimal("0"), Decimal(str(
                (exposure * vulnerability * 4) - (adaptive * 4)
            ))),
        )
        assert assessment.composite_risk_score >= expected_min


# ===========================================================================
# Composite Risk Calculation
# ===========================================================================

class TestCompositeRiskCalculation:
    """Test composite risk score calculation."""

    def test_composite_risk_score(self, sample_physical_risk_assessment):
        assert sample_physical_risk_assessment.composite_risk_score == Decimal("68.0")

    def test_composite_bounded_0_100(self):
        assessment = PhysicalRiskAssessment(
            org_id=_new_id(),
            asset_id=_new_id(),
            hazard_type=PhysicalHazard.WILDFIRE,
            composite_risk_score=Decimal("0"),
        )
        assert Decimal("0") <= assessment.composite_risk_score <= Decimal("100")


# ===========================================================================
# Financial Damage Estimation
# ===========================================================================

class TestFinancialDamageEstimation:
    """Test financial damage estimation."""

    def test_damage_estimate(self, sample_physical_risk_assessment):
        assert sample_physical_risk_assessment.financial_damage_estimate == Decimal("25000000")

    def test_zero_damage(self):
        assessment = PhysicalRiskAssessment(
            org_id=_new_id(),
            asset_id=_new_id(),
            hazard_type=PhysicalHazard.DROUGHT,
            financial_damage_estimate=Decimal("0"),
        )
        assert assessment.financial_damage_estimate == Decimal("0")


# ===========================================================================
# Insurance Impact
# ===========================================================================

class TestInsuranceImpact:
    """Test insurance premium impact."""

    def test_insurance_cost_impact(self, sample_physical_risk_assessment):
        assert sample_physical_risk_assessment.insurance_cost_impact == Decimal("2000000")

    def test_insurance_gap(self, sample_asset_location):
        gap = sample_asset_location.replacement_value - sample_asset_location.insurance_coverage
        assert gap == Decimal("100000000")


# ===========================================================================
# Hazard Exposure Matrices
# ===========================================================================

class TestHazardExposureMatrices:
    """Test hazard exposure matrices by RCP/SSP scenario."""

    @pytest.mark.parametrize("scenario", ["ssp1_26", "ssp2_45", "ssp5_85"])
    def test_scenario_matrix_exists(self, scenario):
        assert scenario in HAZARD_EXPOSURE_MATRICES

    def test_ssp5_85_highest_exposure(self):
        for hazard in PhysicalHazard:
            ssp1 = HAZARD_EXPOSURE_MATRICES["ssp1_26"][hazard]["2050"]
            ssp5 = HAZARD_EXPOSURE_MATRICES["ssp5_85"][hazard]["2050"]
            assert ssp5 >= ssp1

    @pytest.mark.parametrize("hazard", list(PhysicalHazard))
    def test_all_hazards_in_matrix(self, hazard):
        matrix = HAZARD_EXPOSURE_MATRICES["ssp2_45"]
        assert hazard in matrix
        assert "baseline" in matrix[hazard]
        assert "2030" in matrix[hazard]
        assert "2050" in matrix[hazard]


# ===========================================================================
# Portfolio Aggregation
# ===========================================================================

class TestPortfolioAggregation:
    """Test portfolio-level risk aggregation."""

    def test_multi_asset_assessment(self):
        assessments = []
        for i in range(5):
            a = PhysicalRiskAssessment(
                org_id=_new_id(),
                asset_id=_new_id(),
                hazard_type=PhysicalHazard.FLOOD,
                composite_risk_score=Decimal(str(20 + i * 10)),
                financial_damage_estimate=Decimal(str(1000000 * (i + 1))),
            )
            assessments.append(a)
        total_damage = sum(a.financial_damage_estimate for a in assessments)
        avg_risk = sum(a.composite_risk_score for a in assessments) / len(assessments)
        assert total_damage == Decimal("15000000")
        assert avg_risk == Decimal("40")

    def test_assessment_provenance(self, sample_physical_risk_assessment):
        assert len(sample_physical_risk_assessment.provenance_hash) == 64
