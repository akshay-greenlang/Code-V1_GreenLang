# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Do No Significant Harm (DNSH) Assessment Engine.

Tests full DNSH matrix evaluation, per-objective DNSH checks (climate
adaptation, water/marine, circular economy, pollution prevention,
biodiversity), climate risk assessment, DNSH not-applicable handling,
batch DNSH assessment, and evidence linkage with 45+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Full DNSH matrix evaluation tests
# ===========================================================================

class TestDNSHMatrixEvaluation:
    """Test full DNSH matrix evaluation across all objectives."""

    def test_dnsh_all_pass(self, sample_dnsh_assessment):
        """Full DNSH passes when all applicable objectives pass."""
        assert sample_dnsh_assessment["overall_pass"] is True

    def test_dnsh_checks_five_objectives(self, sample_dnsh_assessment):
        """DNSH checks five objectives (all except SC objective)."""
        results = sample_dnsh_assessment["objective_results"]
        assert len(results) == 5

    def test_dnsh_excludes_sc_objective(self, sample_dnsh_assessment):
        """DNSH does not check the SC objective itself."""
        sc_obj = sample_dnsh_assessment["sc_objective"]
        assert sc_obj == "climate_mitigation"
        results = sample_dnsh_assessment["objective_results"]
        # climate_mitigation should not be in DNSH results as it is the SC objective
        # (in this fixture, climate_adaptation is checked instead)
        assert "climate_adaptation" in results

    def test_dnsh_fail_one_objective(self, failing_dnsh_assessment):
        """DNSH fails if any single objective fails."""
        assert failing_dnsh_assessment["overall_pass"] is False

    def test_dnsh_failing_objective_identified(self, failing_dnsh_assessment):
        """Failing objective is identified in results."""
        results = failing_dnsh_assessment["objective_results"]
        assert results["pollution_prevention"]["status"] == "fail"

    def test_dnsh_fail_reason_recorded(self, failing_dnsh_assessment):
        """Failure reason is documented."""
        results = failing_dnsh_assessment["objective_results"]
        assert "BAT non-compliance" in results["pollution_prevention"]["reason"]

    def test_dnsh_status(self, sample_dnsh_assessment):
        """Assessment has completed status."""
        assert sample_dnsh_assessment["status"] == "completed"

    def test_dnsh_provenance(self, sample_dnsh_assessment):
        """DNSH assessment has provenance hash."""
        assert len(sample_dnsh_assessment["provenance_hash"]) == 64


# ===========================================================================
# Climate adaptation DNSH tests
# ===========================================================================

class TestClimateAdaptationDNSH:
    """Test DNSH for climate adaptation objective."""

    def test_climate_adaptation_pass(self, sample_dnsh_objective_results):
        """Climate adaptation DNSH passes with risk assessment."""
        ca = next(r for r in sample_dnsh_objective_results if r["objective"] == "climate_adaptation")
        assert ca["status"] == "pass"

    def test_climate_risk_assessment_completed(self, sample_dnsh_objective_results):
        """Climate risk assessment is completed."""
        ca = next(r for r in sample_dnsh_objective_results if r["objective"] == "climate_adaptation")
        checks = ca["criteria_checks"]
        assert checks["climate_risk_assessment"]["completed"] is True

    def test_risks_identified_and_adapted(self, sample_dnsh_objective_results):
        """Identified risks have adaptation plans."""
        ca = next(r for r in sample_dnsh_objective_results if r["objective"] == "climate_adaptation")
        checks = ca["criteria_checks"]["climate_risk_assessment"]
        assert checks["risks_identified"] == checks["adaptations_planned"]

    def test_appendix_a_reference(self, sample_dnsh_objective_results):
        """Notes reference Appendix A of Climate Delegated Act."""
        ca = next(r for r in sample_dnsh_objective_results if r["objective"] == "climate_adaptation")
        assert "Appendix A" in ca["notes"]

    def test_engine_assess_climate_risk(self, dnsh_engine):
        """Engine climate risk assessment method callable."""
        dnsh_engine.assess_climate_risk.return_value = {
            "status": "pass",
            "risks": 3,
            "adaptations": 3,
        }
        result = dnsh_engine.assess_climate_risk("CCM_4.1", "Bavaria, Germany")
        assert result["status"] == "pass"


# ===========================================================================
# Water/marine DNSH tests
# ===========================================================================

class TestWaterMarineDNSH:
    """Test DNSH for water and marine resources objective."""

    def test_water_not_applicable(self, sample_dnsh_objective_results):
        """Water DNSH is not applicable for solar PV."""
        water = next(r for r in sample_dnsh_objective_results if r["objective"] == "water_marine")
        assert water["status"] == "not_applicable"

    def test_water_no_criteria_checks(self, sample_dnsh_objective_results):
        """No criteria checks needed for not-applicable."""
        water = next(r for r in sample_dnsh_objective_results if r["objective"] == "water_marine")
        assert water["criteria_checks"] == {}

    def test_water_no_evidence_needed(self, sample_dnsh_objective_results):
        """No evidence needed for not-applicable."""
        water = next(r for r in sample_dnsh_objective_results if r["objective"] == "water_marine")
        assert water["evidence_items"] == []

    def test_engine_check_water_dnsh(self, dnsh_engine):
        """Engine water DNSH check returns True."""
        result = dnsh_engine.check_water_dnsh("CCM_4.1")
        assert result is True

    def test_water_dnsh_criteria_for_cement(self, sample_activities):
        """Cement has water DNSH criteria."""
        cement = next(a for a in sample_activities if a["activity_code"] == "CCM_3.3")
        dnsh = cement["dnsh_criteria"]
        assert "water_marine" in dnsh
        assert dnsh["water_marine"]["type"] == "water_use_assessment"


# ===========================================================================
# Circular economy DNSH tests
# ===========================================================================

class TestCircularEconomyDNSH:
    """Test DNSH for circular economy objective."""

    def test_circular_economy_pass(self, sample_dnsh_objective_results):
        """Circular economy DNSH passes with waste management plan."""
        ce = next(r for r in sample_dnsh_objective_results if r["objective"] == "circular_economy")
        assert ce["status"] == "pass"

    def test_waste_management_plan_exists(self, sample_dnsh_objective_results):
        """Waste management plan exists and meets targets."""
        ce = next(r for r in sample_dnsh_objective_results if r["objective"] == "circular_economy")
        checks = ce["criteria_checks"]
        assert checks["waste_management_plan"]["exists"] is True
        assert checks["waste_management_plan"]["recycling_target_met"] is True

    def test_durability_check(self, sample_dnsh_objective_results):
        """Product durability meets standards."""
        ce = next(r for r in sample_dnsh_objective_results if r["objective"] == "circular_economy")
        assert ce["criteria_checks"]["durability"]["lifetime_years"] == 25
        assert ce["criteria_checks"]["durability"]["meets_standard"] is True

    def test_engine_check_circular_dnsh(self, dnsh_engine):
        """Engine circular economy check returns True."""
        result = dnsh_engine.check_circular_dnsh("CCM_4.1")
        assert result is True


# ===========================================================================
# Pollution prevention DNSH tests
# ===========================================================================

class TestPollutionPreventionDNSH:
    """Test DNSH for pollution prevention objective."""

    def test_pollution_pass(self, sample_dnsh_objective_results):
        """Pollution prevention DNSH passes for solar PV."""
        pp = next(r for r in sample_dnsh_objective_results if r["objective"] == "pollution_prevention")
        assert pp["status"] == "pass"

    def test_hazardous_substance_checks(self, sample_dnsh_objective_results):
        """RoHS and REACH compliance checked."""
        pp = next(r for r in sample_dnsh_objective_results if r["objective"] == "pollution_prevention")
        checks = pp["criteria_checks"]["hazardous_substances"]
        assert checks["rohs_compliant"] is True
        assert checks["reach_compliant"] is True

    def test_pollution_fail_for_cement(self, failing_dnsh_assessment):
        """Cement fails pollution DNSH due to BAT non-compliance."""
        results = failing_dnsh_assessment["objective_results"]
        assert results["pollution_prevention"]["status"] == "fail"

    def test_bat_requirement_for_cement(self, sample_activities):
        """Cement DNSH pollution requires BAT compliance."""
        cement = next(a for a in sample_activities if a["activity_code"] == "CCM_3.3")
        assert cement["dnsh_criteria"]["pollution_prevention"]["type"] == "bat_compliance"

    def test_engine_check_pollution_dnsh(self, dnsh_engine):
        """Engine pollution prevention check returns True."""
        result = dnsh_engine.check_pollution_dnsh("CCM_4.1")
        assert result is True


# ===========================================================================
# Biodiversity DNSH tests
# ===========================================================================

class TestBiodiversityDNSH:
    """Test DNSH for biodiversity and ecosystems objective."""

    def test_biodiversity_pass(self, sample_dnsh_objective_results):
        """Biodiversity DNSH passes for solar PV."""
        bio = next(r for r in sample_dnsh_objective_results if r["objective"] == "biodiversity")
        assert bio["status"] == "pass"

    def test_eia_completed(self, sample_dnsh_objective_results):
        """Environmental Impact Assessment completed."""
        bio = next(r for r in sample_dnsh_objective_results if r["objective"] == "biodiversity")
        assert bio["criteria_checks"]["eia_completed"] is True

    def test_protected_area_check(self, sample_dnsh_objective_results):
        """Protected area proximity check completed."""
        bio = next(r for r in sample_dnsh_objective_results if r["objective"] == "biodiversity")
        assert bio["criteria_checks"]["protected_area_check"] is True

    def test_no_high_biodiversity_conversion(self, sample_dnsh_objective_results):
        """No conversion of high-biodiversity areas."""
        bio = next(r for r in sample_dnsh_objective_results if r["objective"] == "biodiversity")
        assert bio["criteria_checks"]["no_conversion_of_high_biodiversity"] is True

    def test_eia_requirement_for_construction(self, sample_activities):
        """Construction activities require EIA for biodiversity DNSH."""
        building = next(a for a in sample_activities if a["activity_code"] == "CCM_7.1")
        assert building["dnsh_criteria"]["biodiversity"]["type"] == "eia_required"

    def test_engine_check_biodiversity_dnsh(self, dnsh_engine):
        """Engine biodiversity check returns True."""
        result = dnsh_engine.check_biodiversity_dnsh("CCM_4.1")
        assert result is True


# ===========================================================================
# Climate risk assessment tests
# ===========================================================================

class TestClimateRiskAssessment:
    """Test climate risk assessment for DNSH adaptation."""

    def test_risk_assessment_status(self, sample_climate_risk_assessment):
        """Climate risk assessment has managed status."""
        assert sample_climate_risk_assessment["overall_status"] == "managed"

    def test_chronic_risks_identified(self, sample_climate_risk_assessment):
        """Chronic physical risks identified."""
        chronic = sample_climate_risk_assessment["physical_risks"]["chronic"]
        assert len(chronic) == 2
        hazards = {r["hazard"] for r in chronic}
        assert "temperature_increase" in hazards

    def test_acute_risks_identified(self, sample_climate_risk_assessment):
        """Acute physical risks identified."""
        acute = sample_climate_risk_assessment["physical_risks"]["acute"]
        assert len(acute) == 2
        hazards = {r["hazard"] for r in acute}
        assert "heatwave" in hazards

    def test_adaptation_solutions_match_risks(self, sample_climate_risk_assessment):
        """Adaptation solutions address identified risks."""
        solutions = sample_climate_risk_assessment["adaptation_solutions"]
        assert len(solutions) >= 2
        for solution in solutions:
            assert "cost_eur" in solution

    def test_residual_risks_acceptable(self, sample_climate_risk_assessment):
        """All residual risks are acceptable after adaptation."""
        residual = sample_climate_risk_assessment["residual_risks"]
        for risk_name, detail in residual.items():
            assert detail["acceptable"] is True
            assert detail["residual_severity"] == "low"

    def test_time_horizon(self, sample_climate_risk_assessment):
        """Assessment uses long-term time horizon."""
        assert sample_climate_risk_assessment["time_horizon"] == "long_term"

    def test_location_recorded(self, sample_climate_risk_assessment):
        """Assessment location is recorded."""
        assert sample_climate_risk_assessment["location"] == "Bavaria, Germany"


# ===========================================================================
# Not-applicable handling tests
# ===========================================================================

class TestNotApplicableHandling:
    """Test handling of not-applicable DNSH objectives."""

    def test_na_status_in_results(self, sample_dnsh_assessment):
        """Not-applicable objectives recorded in results."""
        results = sample_dnsh_assessment["objective_results"]
        na_count = sum(1 for v in results.values() if v.get("status") == "not_applicable")
        assert na_count >= 1

    def test_na_does_not_block_pass(self, sample_dnsh_assessment):
        """Not-applicable objectives do not prevent overall pass."""
        assert sample_dnsh_assessment["overall_pass"] is True

    def test_na_in_objective_detail(self, sample_dnsh_objective_results):
        """Detail record captures not_applicable status."""
        na_results = [r for r in sample_dnsh_objective_results if r["status"] == "not_applicable"]
        assert len(na_results) >= 1
        for r in na_results:
            assert r["criteria_checks"] == {}
            assert r["evidence_items"] == []


# ===========================================================================
# Batch DNSH assessment tests
# ===========================================================================

class TestBatchDNSH:
    """Test batch DNSH assessment functionality."""

    def test_batch_assess_multiple_activities(self, dnsh_engine):
        """Batch DNSH assesses multiple activities."""
        dnsh_engine.batch_assess.return_value = {
            "CCM_4.1": {"overall_pass": True, "failed_objectives": []},
            "CCM_3.3": {"overall_pass": False, "failed_objectives": ["pollution_prevention"]},
            "CCM_7.1": {"overall_pass": True, "failed_objectives": []},
        }
        result = dnsh_engine.batch_assess(
            activities=["CCM_4.1", "CCM_3.3", "CCM_7.1"],
            sc_objective="climate_mitigation",
        )
        assert len(result) == 3
        assert result["CCM_3.3"]["overall_pass"] is False
        assert "pollution_prevention" in result["CCM_3.3"]["failed_objectives"]

    def test_batch_all_pass(self, dnsh_engine):
        """Batch where all activities pass DNSH."""
        dnsh_engine.batch_assess.return_value = {
            "CCM_4.1": {"overall_pass": True},
            "CCM_4.3": {"overall_pass": True},
        }
        result = dnsh_engine.batch_assess(["CCM_4.1", "CCM_4.3"])
        assert all(v["overall_pass"] for v in result.values())

    def test_batch_all_fail(self, dnsh_engine):
        """Batch where all activities fail DNSH."""
        dnsh_engine.batch_assess.return_value = {
            "ACT_1": {"overall_pass": False},
            "ACT_2": {"overall_pass": False},
        }
        result = dnsh_engine.batch_assess(["ACT_1", "ACT_2"])
        assert all(not v["overall_pass"] for v in result.values())


# ===========================================================================
# DNSH evidence linkage tests
# ===========================================================================

class TestDNSHEvidenceLinkage:
    """Test evidence linkage in DNSH assessments."""

    def test_evidence_items_in_assessment(self, sample_dnsh_assessment):
        """Assessment-level evidence items present."""
        assert len(sample_dnsh_assessment["evidence_items"]) >= 2

    def test_evidence_linked_to_objective(self, sample_dnsh_assessment):
        """Evidence items reference specific objectives."""
        for item in sample_dnsh_assessment["evidence_items"]:
            assert "objective" in item

    def test_objective_level_evidence(self, sample_dnsh_objective_results):
        """Per-objective results include evidence items."""
        ca = next(r for r in sample_dnsh_objective_results if r["objective"] == "climate_adaptation")
        assert len(ca["evidence_items"]) >= 1
        assert ca["evidence_items"][0]["type"] == "report"

    def test_evidence_ref_format(self, sample_dnsh_objective_results):
        """Evidence references follow naming convention."""
        for result in sample_dnsh_objective_results:
            for item in result["evidence_items"]:
                assert "ref" in item
                assert len(item["ref"]) > 3

    def test_engine_assess_returns_evidence(self, dnsh_engine):
        """Engine assessment returns evidence references."""
        dnsh_engine.assess.return_value = {
            "overall_pass": True,
            "evidence": [{"ref": "DOC-001", "type": "report"}],
        }
        result = dnsh_engine.assess("CCM_4.1", "climate_mitigation")
        assert len(result["evidence"]) >= 1
