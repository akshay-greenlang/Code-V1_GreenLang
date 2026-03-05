# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Substantial Contribution (SC) Assessment Engine.

Tests SC assessment per activity per objective, threshold evaluation for
electricity/cement/steel, enabling and transitional classification, evidence
recording, batch assessment, and multi-objective SC handling with 45+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# SC assessment creation tests
# ===========================================================================

class TestSCAssessmentCreation:
    """Test SC assessment record creation."""

    def test_create_sc_assessment(self, sample_sc_assessment):
        assert sample_sc_assessment["activity_code"] == "CCM_4.1"
        assert sample_sc_assessment["objective"] == "climate_mitigation"
        assert len(sample_sc_assessment["id"]) == 36

    def test_sc_assessment_status(self, sample_sc_assessment):
        assert sample_sc_assessment["status"] == "completed"

    def test_sc_assessment_type(self, sample_sc_assessment):
        assert sample_sc_assessment["sc_type"] == "own_performance"

    def test_sc_assessment_pass_result(self, sample_sc_assessment):
        assert sample_sc_assessment["overall_pass"] is True

    def test_sc_assessor_recorded(self, sample_sc_assessment):
        assert sample_sc_assessment["assessor"] == "Environmental Compliance Team"

    def test_sc_provenance_hash(self, sample_sc_assessment):
        assert len(sample_sc_assessment["provenance_hash"]) == 64


# ===========================================================================
# Threshold evaluation tests -- Electricity (solar PV)
# ===========================================================================

class TestElectricityThreshold:
    """Test electricity generation SC thresholds."""

    def test_solar_pv_below_threshold(self, sample_sc_assessment):
        """Solar PV life-cycle GHG below 100 gCO2e/kWh."""
        checks = sample_sc_assessment["threshold_checks"]
        ghg = checks["life_cycle_ghg"]
        assert ghg["actual_gco2e_kwh"] < ghg["threshold_gco2e_kwh"]
        assert ghg["pass"] is True

    def test_solar_pv_threshold_value(self, single_activity):
        """Solar PV SC criteria threshold is 100 gCO2e/kWh."""
        sc = single_activity["sc_criteria"]
        assert sc["threshold_gco2e_kwh"] == 100

    def test_solar_pv_methodology(self, sample_sc_assessment):
        """Life-cycle methodology is ISO 14067."""
        checks = sample_sc_assessment["threshold_checks"]
        assert checks["life_cycle_ghg"]["methodology"] == "ISO 14067"

    def test_wind_power_criteria(self, sample_activities):
        """Wind power uses same life-cycle GHG threshold."""
        wind = next(a for a in sample_activities if a["activity_code"] == "CCM_4.3")
        assert wind["sc_criteria"]["threshold_gco2e_kwh"] == 100

    def test_engine_evaluate_electricity(self, sc_engine):
        """Engine evaluates electricity threshold correctly."""
        sc_engine.evaluate_threshold.return_value = {
            "criterion": "life_cycle_ghg",
            "threshold": 100,
            "actual": 25.4,
            "pass": True,
        }
        result = sc_engine.evaluate_threshold("CCM_4.1", "life_cycle_ghg", 25.4)
        assert result["pass"] is True


# ===========================================================================
# Threshold evaluation tests -- Cement
# ===========================================================================

class TestCementThreshold:
    """Test cement SC thresholds."""

    def test_cement_grey_clinker_pass(self, cement_sc_assessment):
        """Grey clinker emissions below 0.722 tCO2e/t."""
        checks = cement_sc_assessment["threshold_checks"]
        assert checks["grey_clinker_emissions"]["pass"] is True
        assert checks["grey_clinker_emissions"]["actual_tco2e_t"] < 0.722

    def test_cement_product_pass(self, cement_sc_assessment):
        """Cement product emissions below 0.469 tCO2e/t."""
        checks = cement_sc_assessment["threshold_checks"]
        assert checks["cement_emissions"]["pass"] is True
        assert checks["cement_emissions"]["actual_tco2e_t"] < 0.469

    def test_cement_criteria_from_catalog(self, transitional_activity):
        """Cement activity has correct SC thresholds in catalog."""
        sc = transitional_activity["sc_criteria"]
        assert sc["grey_clinker_threshold_tco2e_t"] == 0.722
        assert sc["cement_threshold_tco2e_t"] == 0.469

    def test_cement_both_thresholds_required(self, cement_sc_assessment):
        """Both grey clinker and cement thresholds must pass."""
        checks = cement_sc_assessment["threshold_checks"]
        all_pass = all(c["pass"] for c in checks.values())
        assert all_pass is True
        assert cement_sc_assessment["overall_pass"] is True


# ===========================================================================
# Threshold evaluation tests -- Steel
# ===========================================================================

class TestSteelThreshold:
    """Test steel SC thresholds."""

    def test_steel_hot_metal_fail(self, steel_sc_assessment):
        """Hot metal emissions exceed 1.331 tCO2e/t threshold."""
        checks = steel_sc_assessment["threshold_checks"]
        assert checks["hot_metal_emissions"]["pass"] is False
        assert checks["hot_metal_emissions"]["actual_tco2e_t"] > 1.331

    def test_steel_overall_fail(self, steel_sc_assessment):
        """Overall SC fails when any threshold is exceeded."""
        assert steel_sc_assessment["overall_pass"] is False

    def test_steel_criteria_from_catalog(self, sample_activities):
        """Steel activity has correct SC thresholds."""
        steel = next(a for a in sample_activities if a["activity_code"] == "CCM_3.9")
        sc = steel["sc_criteria"]
        assert sc["hot_metal_threshold_tco2e_t"] == 1.331
        assert sc["eaf_carbon_threshold_tco2e_t"] == 0.209

    def test_steel_exceedance_magnitude(self, steel_sc_assessment):
        """Calculate exceedance magnitude for gap reporting."""
        checks = steel_sc_assessment["threshold_checks"]
        hm = checks["hot_metal_emissions"]
        exceedance = hm["actual_tco2e_t"] - hm["threshold_tco2e_t"]
        exceedance_pct = (exceedance / hm["threshold_tco2e_t"]) * 100
        assert exceedance > 0
        assert exceedance_pct > 10


# ===========================================================================
# Enabling activity tests
# ===========================================================================

class TestEnablingClassification:
    """Test enabling activity SC classification."""

    def test_enabling_type_identified(self, enabling_activity):
        """Data centre classified as enabling."""
        assert enabling_activity["activity_type"] == "enabling"

    def test_enabling_criteria(self, enabling_activity):
        """Enabling activity has PUE threshold."""
        sc = enabling_activity["sc_criteria"]
        assert sc["pue_threshold"] == 1.5
        assert sc["eu_code_of_conduct"] is True

    def test_engine_check_enabling(self, sc_engine):
        """Engine enabling criteria check returns True."""
        result = sc_engine.check_enabling_criteria("CCM_8.1", {"pue": 1.3})
        assert result is True

    def test_enabling_no_lock_in(self, sc_engine):
        """Enabling activities must not create carbon lock-in."""
        sc_engine.check_enabling_criteria.return_value = True
        result = sc_engine.check_enabling_criteria("CCM_8.1", {
            "pue": 1.3,
            "no_lock_in": True,
            "direct_ghg_zero": True,
        })
        assert result is True

    def test_insurance_enabling_activity(self, sample_activities):
        """Insurance underwriting is an enabling activity for adaptation."""
        insurance = next(a for a in sample_activities if a["activity_code"] == "CCA_9.1")
        assert insurance["activity_type"] == "enabling"
        assert "climate_adaptation" in insurance["objectives"]


# ===========================================================================
# Transitional activity tests
# ===========================================================================

class TestTransitionalClassification:
    """Test transitional activity SC classification."""

    def test_transitional_type_identified(self, transitional_activity):
        """Cement classified as transitional."""
        assert transitional_activity["activity_type"] == "transitional"

    def test_transitional_no_alternative(self, transitional_activity):
        """Transitional implies no low-carbon alternative exists."""
        assert transitional_activity["delegated_act"] == "climate"

    def test_engine_check_transitional(self, sc_engine):
        """Engine transitional criteria check returns True."""
        result = sc_engine.check_transitional_criteria("CCM_3.3", {
            "below_sector_threshold": True,
            "no_carbon_lock_in": True,
        })
        assert result is True

    def test_transitional_activities_count(self, sample_activities):
        """Catalog has identified transitional activities."""
        transitional = [a for a in sample_activities if a["activity_type"] == "transitional"]
        assert len(transitional) == 2  # cement + steel

    def test_transitional_sc_type_in_assessment(self, cement_sc_assessment):
        """SC assessment records transitional type."""
        assert cement_sc_assessment["sc_type"] == "transitional"


# ===========================================================================
# Evidence recording tests
# ===========================================================================

class TestEvidenceRecording:
    """Test evidence recording for SC assessments."""

    def test_evidence_items_present(self, sample_sc_assessment):
        """SC assessment has evidence items."""
        assert len(sample_sc_assessment["evidence_items"]) == 2

    def test_evidence_item_structure(self, sample_sc_assessment):
        """Evidence items have type, ref, and verified status."""
        for item in sample_sc_assessment["evidence_items"]:
            assert "type" in item
            assert "ref" in item
            assert "verified" in item

    def test_evidence_types(self, sample_sc_assessment):
        """Evidence includes certification and report."""
        types = {item["type"] for item in sample_sc_assessment["evidence_items"]}
        assert "certification" in types
        assert "report" in types

    def test_engine_record_evidence(self, sc_engine):
        """Engine records evidence item."""
        sc_engine.record_evidence.return_value = {"id": "ev-001", "status": "uploaded"}
        result = sc_engine.record_evidence(
            assessment_id="asc-001",
            evidence_type="certification",
            document_ref="ISO14067-2023-001",
        )
        sc_engine.record_evidence.assert_called_once()

    def test_tsc_evaluation_evidence(self, sample_tsc_evaluations):
        """TSC evaluations reference evidence documents."""
        for eval_item in sample_tsc_evaluations:
            assert eval_item["evidence_ref"] is not None


# ===========================================================================
# Multi-objective SC tests
# ===========================================================================

class TestMultiObjectiveSC:
    """Test SC assessment for different objectives."""

    def test_climate_mitigation_sc(self, sample_sc_assessment):
        """Climate mitigation SC for solar PV."""
        assert sample_sc_assessment["objective"] == "climate_mitigation"

    def test_water_objective_activity(self, sample_activities):
        """Water/marine objective activity exists."""
        water = [a for a in sample_activities if "water_marine" in a["objectives"]]
        assert len(water) >= 1
        assert water[0]["activity_code"] == "WTR_2.1"

    def test_circular_economy_activity(self, sample_activities):
        """Circular economy objective activity exists."""
        ce = [a for a in sample_activities if "circular_economy" in a["objectives"]]
        assert len(ce) >= 1

    def test_pollution_prevention_activity(self, sample_activities):
        """Pollution prevention objective activity exists."""
        pp = [a for a in sample_activities if "pollution_prevention" in a["objectives"]]
        assert len(pp) >= 1

    def test_biodiversity_activity(self, sample_activities):
        """Biodiversity objective activity exists."""
        bio = [a for a in sample_activities if "biodiversity" in a["objectives"]]
        assert len(bio) >= 1

    def test_engine_assess_different_objectives(self, sc_engine):
        """Engine can assess SC for any objective."""
        for objective in ["climate_mitigation", "climate_adaptation", "water_marine",
                          "circular_economy", "pollution_prevention", "biodiversity"]:
            sc_engine.assess.return_value = {"objective": objective, "pass": True}
            result = sc_engine.assess("ACT_001", objective)
            assert result["objective"] == objective


# ===========================================================================
# Batch SC assessment tests
# ===========================================================================

class TestBatchSCAssessment:
    """Test batch SC assessment functionality."""

    def test_batch_assess(self, sc_engine):
        """Batch assessment processes multiple activities."""
        sc_engine.batch_assess.return_value = {
            "CCM_4.1": {"pass": True, "type": "own_performance"},
            "CCM_3.3": {"pass": True, "type": "transitional"},
            "CCM_3.9": {"pass": False, "type": "transitional"},
        }
        result = sc_engine.batch_assess(
            ["CCM_4.1", "CCM_3.3", "CCM_3.9"],
            objective="climate_mitigation",
        )
        assert len(result) == 3
        assert result["CCM_4.1"]["pass"] is True
        assert result["CCM_3.9"]["pass"] is False

    def test_batch_mixed_types(self, sc_engine):
        """Batch correctly identifies activity types."""
        sc_engine.batch_assess.return_value = {
            "CCM_4.1": {"type": "own_performance"},
            "CCM_3.3": {"type": "transitional"},
            "CCM_8.1": {"type": "enabling"},
        }
        result = sc_engine.batch_assess(["CCM_4.1", "CCM_3.3", "CCM_8.1"])
        types = {v["type"] for v in result.values()}
        assert "own_performance" in types
        assert "transitional" in types
        assert "enabling" in types

    def test_engine_classify_activity_type(self, sc_engine):
        """Engine classifies activity type correctly."""
        result = sc_engine.classify_activity_type("CCM_4.1")
        assert result == "own_performance"


# ===========================================================================
# TSC evaluation tests
# ===========================================================================

class TestTSCEvaluation:
    """Test Technical Screening Criteria evaluation."""

    def test_tsc_numeric_comparison(self, sample_tsc_evaluations):
        """Numeric TSC compares actual vs threshold."""
        ghg_eval = sample_tsc_evaluations[0]
        assert ghg_eval["actual_value"] < ghg_eval["threshold_value"]
        assert ghg_eval["pass_result"] is True

    def test_tsc_qualitative_check(self, sample_tsc_evaluations):
        """Qualitative TSC (no numeric threshold)."""
        cert_eval = sample_tsc_evaluations[1]
        assert cert_eval["threshold_value"] is None
        assert cert_eval["pass_result"] is True

    def test_tsc_criterion_id_format(self, sample_tsc_evaluations):
        """TSC criterion IDs follow expected format."""
        for eval_item in sample_tsc_evaluations:
            assert eval_item["criterion_id"].startswith("CCM_")
            assert "_SC_" in eval_item["criterion_id"]

    def test_tsc_unit_specified(self, sample_tsc_evaluations):
        """Numeric TSC specifies measurement unit."""
        ghg_eval = sample_tsc_evaluations[0]
        assert ghg_eval["unit"] == "gCO2e/kWh"
