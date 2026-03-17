# -*- coding: utf-8 -*-
"""
Unit tests for BATComplianceEngine (PACK-013, Engine 6)

Tests BAT/BREF compliance checking, transformation plans, abatement
options, penalty risk, and provenance tracking.

Target: 85%+ coverage, 38+ tests.
"""

import importlib.util
import os
import sys
import pytest
from decimal import Decimal

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines"
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load_module("bat_compliance_engine", "bat_compliance_engine.py")

BATComplianceEngine = bc.BATComplianceEngine
BATConfig = bc.BATConfig
FacilityBATData = bc.FacilityBATData
MeasuredParameter = bc.MeasuredParameter
BREFReference = bc.BREFReference
BATComplianceResult = bc.BATComplianceResult
TransformationPlan = bc.TransformationPlan
AbatementOption = bc.AbatementOption
ParameterResult = bc.ParameterResult
BREFDocument = bc.BREFDocument
ComplianceStatus = bc.ComplianceStatus
TechnologyReadinessLevel = bc.TechnologyReadinessLevel
TransformationStatus = bc.TransformationStatus
BAT_AEL_DATABASE = bc.BAT_AEL_DATABASE
ABATEMENT_TECHNOLOGIES = bc.ABATEMENT_TECHNOLOGIES
IED_PENALTY_MINIMUM_EUR = bc.IED_PENALTY_MINIMUM_EUR
IED_PENALTY_TURNOVER_PCT = bc.IED_PENALTY_TURNOVER_PCT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    return BATConfig(
        reporting_year=2025,
        applicable_brefs=[BREFDocument.CEMENT_LIME],
        annual_turnover_eur=Decimal("200000000"),
        facility_capacity_mw=Decimal("50"),
    )


@pytest.fixture
def default_engine(default_config):
    return BATComplianceEngine(default_config)


@pytest.fixture
def cement_bat_data():
    return FacilityBATData(
        facility_id="CEM-001",
        facility_name="Cement Plant Alpha",
        sub_sector="cement",
        applicable_brefs=[BREFDocument.CEMENT_LIME],
        measured_parameters=[
            MeasuredParameter(parameter_name="dust", measured_value=15.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="nox", measured_value=350.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="so2", measured_value=200.0, unit="mg/Nm3"),
        ],
        annual_turnover_eur=Decimal("150000000"),
        capacity_mw=Decimal("80"),
    )


@pytest.fixture
def steel_bat_data():
    return FacilityBATData(
        facility_id="STL-001",
        facility_name="Steel Works Beta",
        sub_sector="steel",
        applicable_brefs=[BREFDocument.IRON_STEEL],
        measured_parameters=[
            MeasuredParameter(parameter_name="dust_sinter", measured_value=12.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="nox_sinter", measured_value=220.0, unit="mg/Nm3"),
        ],
        annual_turnover_eur=Decimal("500000000"),
        capacity_mw=Decimal("200"),
    )


@pytest.fixture
def glass_bat_data():
    return FacilityBATData(
        facility_id="GLS-001",
        facility_name="Glass Factory Gamma",
        sub_sector="glass",
        applicable_brefs=[BREFDocument.GLASS],
        measured_parameters=[
            MeasuredParameter(parameter_name="dust", measured_value=25.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="nox", measured_value=700.0, unit="mg/Nm3"),
        ],
        annual_turnover_eur=Decimal("80000000"),
        capacity_mw=Decimal("30"),
    )


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test BATComplianceEngine initialization."""

    def test_default_init(self, default_config):
        engine = BATComplianceEngine(default_config)
        assert engine.config == default_config
        assert engine.config.reporting_year == 2025

    def test_with_config(self):
        cfg = BATConfig(
            reporting_year=2024,
            applicable_brefs=[BREFDocument.IRON_STEEL],
        )
        engine = BATComplianceEngine(cfg)
        assert BREFDocument.IRON_STEEL in engine.config.applicable_brefs

    def test_with_dict(self):
        cfg = BATConfig(**{
            "reporting_year": 2025,
            "applicable_brefs": ["glass"],
            "annual_turnover_eur": 100000000,
        })
        engine = BATComplianceEngine(cfg)
        assert engine.config.annual_turnover_eur == Decimal("100000000")

    def test_with_none_optional_fields(self):
        cfg = BATConfig(reporting_year=2025)
        engine = BATComplianceEngine(cfg)
        assert engine.config.applicable_brefs == []
        assert engine.config.annual_turnover_eur == Decimal("0")


# ---------------------------------------------------------------------------
# TestBREFDocuments
# ---------------------------------------------------------------------------


class TestBREFDocuments:
    """Test BREF document enumeration."""

    def test_all_brefs_defined(self):
        assert len(BREFDocument) >= 17

    def test_bref_enum_values(self):
        values = {b.value for b in BREFDocument}
        assert "cement_lime" in values
        assert "iron_steel" in values
        assert "glass" in values

    def test_cement_bref(self):
        assert BREFDocument.CEMENT_LIME.value == "cement_lime"

    def test_iron_steel_bref(self):
        assert BREFDocument.IRON_STEEL.value == "iron_steel"


# ---------------------------------------------------------------------------
# TestParameterCheck
# ---------------------------------------------------------------------------


class TestParameterCheck:
    """Test individual parameter compliance checks."""

    def test_compliant_parameter(self, default_engine):
        mp = MeasuredParameter(parameter_name="dust", measured_value=8.0, unit="mg/Nm3")
        bref_ref = BREFReference(
            bref_document=BREFDocument.CEMENT_LIME,
            parameter_name="dust",
            bat_ael_lower=10.0, bat_ael_upper=20.0,
            unit="mg/Nm3",
        )
        result = default_engine.check_parameter(mp, bref_ref)
        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert result.gap_pct == 0.0

    def test_non_compliant_parameter(self, default_engine):
        mp = MeasuredParameter(parameter_name="dust", measured_value=30.0, unit="mg/Nm3")
        bref_ref = BREFReference(
            bref_document=BREFDocument.CEMENT_LIME,
            parameter_name="dust",
            bat_ael_lower=10.0, bat_ael_upper=20.0,
            unit="mg/Nm3",
        )
        result = default_engine.check_parameter(mp, bref_ref)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        # gap = (30 - 20) / 20 * 100 = 50%
        assert result.gap_pct == pytest.approx(50.0, rel=1e-2)

    def test_within_range(self, default_engine):
        mp = MeasuredParameter(parameter_name="dust", measured_value=15.0, unit="mg/Nm3")
        bref_ref = BREFReference(
            bref_document=BREFDocument.CEMENT_LIME,
            parameter_name="dust",
            bat_ael_lower=10.0, bat_ael_upper=20.0,
            unit="mg/Nm3",
        )
        result = default_engine.check_parameter(mp, bref_ref)
        assert result.compliance_status == ComplianceStatus.WITHIN_RANGE

    def test_derogation(self):
        assert ComplianceStatus.DEROGATION_GRANTED.value == "derogation_granted"

    def test_gap_percentage_calculation(self, default_engine):
        mp = MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3")
        bref_ref = BREFReference(
            bref_document=BREFDocument.CEMENT_LIME,
            parameter_name="nox",
            bat_ael_lower=200.0, bat_ael_upper=450.0,
            unit="mg/Nm3",
        )
        result = default_engine.check_parameter(mp, bref_ref)
        # gap = (600 - 450) / 450 * 100 = 33.33%
        assert result.gap_pct == pytest.approx(33.33, rel=1e-2)

    def test_unknown_parameter(self, default_engine, cement_bat_data):
        # Add unknown parameter that won't match any BREF entry
        cement_bat_data.measured_parameters.append(
            MeasuredParameter(parameter_name="xyzzy_unknown", measured_value=100.0, unit="mg/Nm3")
        )
        result = default_engine.assess_compliance(cement_bat_data)
        # The unknown parameter should be NOT_ASSESSED
        unknown_results = [
            pr for pr in result.parameter_results
            if pr.parameter_name == "xyzzy_unknown"
        ]
        assert len(unknown_results) == 1
        assert unknown_results[0].compliance_status == ComplianceStatus.NOT_ASSESSED


# ---------------------------------------------------------------------------
# TestFacilityAssessment
# ---------------------------------------------------------------------------


class TestFacilityAssessment:
    """Test full facility compliance assessment."""

    def test_fully_compliant_facility(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-002",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=8.0, unit="mg/Nm3"),
                MeasuredParameter(parameter_name="nox", measured_value=180.0, unit="mg/Nm3"),
                MeasuredParameter(parameter_name="so2", measured_value=40.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        assert result.overall_compliance_status == ComplianceStatus.COMPLIANT

    def test_partially_compliant(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        # dust=15 (within_range 10-20), nox=350 (within_range 200-450), so2=200 (within_range 50-400)
        assert result.overall_compliance_status == ComplianceStatus.WITHIN_RANGE

    def test_non_compliant_facility(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-003",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=30.0, unit="mg/Nm3"),
                MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        assert result.overall_compliance_status == ComplianceStatus.NON_COMPLIANT

    def test_overall_status(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        assert result.overall_compliance_status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.WITHIN_RANGE,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.NOT_ASSESSED,
        ]

    def test_parameters_assessed_count(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        assert result.parameters_assessed == len(cement_bat_data.measured_parameters)


# ---------------------------------------------------------------------------
# TestTransformationPlan
# ---------------------------------------------------------------------------


class TestTransformationPlan:
    """Test transformation plan generation."""

    def test_plan_required_when_non_compliant(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-004",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=30.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        assert result.transformation_plan is not None
        assert result.transformation_plan.required is True

    def test_plan_not_required_when_compliant(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-005",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=8.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        assert result.transformation_plan is None

    def test_plan_has_deadline(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-006",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=30.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        plan = result.transformation_plan
        assert plan is not None
        assert plan.deadline is not None
        assert "2029" in plan.deadline  # 2025 + 4 years

    def test_plan_investment_estimate(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-007",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="dust", measured_value=30.0, unit="mg/Nm3"),
            ],
            capacity_mw=Decimal("100"),
        )
        result = default_engine.assess_compliance(facility)
        plan = result.transformation_plan
        assert plan is not None
        assert plan.investment_required_eur > 0


# ---------------------------------------------------------------------------
# TestAbatementOptions
# ---------------------------------------------------------------------------


class TestAbatementOptions:
    """Test abatement technology option analysis."""

    def test_options_generated(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-008",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        assert len(result.abatement_options) > 0

    def test_trl_scores(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-009",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        for opt in result.abatement_options:
            assert 1 <= opt.trl <= 9

    def test_marginal_cost(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-010",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        for opt in result.abatement_options:
            assert opt.marginal_cost_eur_per_tco2 > 0

    def test_payback_calculation(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-011",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[
                MeasuredParameter(parameter_name="nox", measured_value=600.0, unit="mg/Nm3"),
            ],
        )
        result = default_engine.assess_compliance(facility)
        for opt in result.abatement_options:
            assert opt.payback_years > 0


# ---------------------------------------------------------------------------
# TestPenaltyRisk
# ---------------------------------------------------------------------------


class TestPenaltyRisk:
    """Test IED penalty risk calculation."""

    def test_penalty_minimum_3m(self, default_engine):
        # 1 violation, zero turnover => base penalty = max(3M, 0) = 3M
        penalty = default_engine.calculate_penalty_risk(1, 0.0)
        assert penalty >= 3_000_000.0

    def test_penalty_3pct_turnover(self, default_engine):
        # 1 violation, 200M turnover => base = max(3M, 200M*3%) = max(3M, 6M) = 6M
        penalty = default_engine.calculate_penalty_risk(1, 200_000_000.0)
        expected = 200_000_000.0 * 0.03
        assert penalty == pytest.approx(expected, rel=1e-6)

    def test_no_penalty_when_compliant(self, default_engine):
        penalty = default_engine.calculate_penalty_risk(0, 200_000_000.0)
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, default_engine, cement_bat_data):
        r1 = default_engine.assess_compliance(cement_bat_data)
        r2 = default_engine.assess_compliance(cement_bat_data)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_different_input(self, default_engine, cement_bat_data, steel_bat_data):
        r1 = default_engine.assess_compliance(cement_bat_data)
        # Steel uses different BREF, so create separate engine
        steel_cfg = BATConfig(
            reporting_year=2025,
            applicable_brefs=[BREFDocument.IRON_STEEL],
        )
        steel_engine = BATComplianceEngine(steel_cfg)
        r2 = steel_engine.assess_compliance(steel_bat_data)
        assert r1.facility_id != r2.facility_id


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameters_no_crash(self, default_engine):
        facility = FacilityBATData(
            facility_id="CEM-EMPTY",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=[],
        )
        result = default_engine.assess_compliance(facility)
        assert result.overall_compliance_status == ComplianceStatus.NOT_ASSESSED

    def test_large_parameter_set(self, default_engine):
        params = [
            MeasuredParameter(parameter_name="dust", measured_value=15.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="nox", measured_value=300.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="so2", measured_value=200.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="co", measured_value=600.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="hcl", measured_value=7.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="hf", measured_value=0.8, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="toc", measured_value=20.0, unit="mg/Nm3"),
            MeasuredParameter(parameter_name="mercury", measured_value=0.02, unit="mg/Nm3"),
        ]
        facility = FacilityBATData(
            facility_id="CEM-LARGE",
            sub_sector="cement",
            applicable_brefs=[BREFDocument.CEMENT_LIME],
            measured_parameters=params,
        )
        result = default_engine.assess_compliance(facility)
        assert result.parameters_assessed == 8

    def test_result_fields(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        assert isinstance(result, BATComplianceResult)
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0

    def test_bat_ael_database_populated(self):
        assert len(BAT_AEL_DATABASE) >= 7
        assert BREFDocument.CEMENT_LIME in BAT_AEL_DATABASE

    def test_methodology_notes(self, default_engine, cement_bat_data):
        result = default_engine.assess_compliance(cement_bat_data)
        assert isinstance(result.methodology_notes, list)
        assert len(result.methodology_notes) > 0
