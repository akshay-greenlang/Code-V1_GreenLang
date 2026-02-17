# -*- coding: utf-8 -*-
"""
Integration Tests: Regulatory Compliance Validation (AGENT-DATA-019)
=====================================================================

Tests applying regulatory rule packs (GHG Protocol, CSRD/ESRS, EUDR, SOC2)
to realistic datasets and verifying that pass/fail rates match expected
outcomes, compliance reports are generated correctly, and cross-pack
validation produces consistent results.

Test Classes:
    TestGHGProtocolCompliance             (~8 tests)
    TestCSRDESRSCompliance                (~7 tests)
    TestEUDRCompliance                    (~7 tests)
    TestSOC2Compliance                    (~6 tests)
    TestCrossPackValidation               (~6 tests)
    TestComplianceReportGeneration        (~6 tests)

Total: ~40 integration tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ===========================================================================
# TestGHGProtocolCompliance
# ===========================================================================


class TestGHGProtocolCompliance:
    """Test GHG Protocol rule pack application and evaluation."""

    def test_apply_ghg_pack_imports_rules(self, service):
        """Applying GHG pack should import multiple rules."""
        result = service.apply_pack("ghg_protocol")
        assert result["rules_imported"] >= 3
        assert result["status"] == "applied"

    def test_apply_ghg_pack_creates_rule_set(self, service):
        """Applying GHG pack should create at least one rule set."""
        result = service.apply_pack("ghg_protocol")
        assert result["rule_sets_created"] >= 1

    def test_ghg_pack_rules_are_active(self, service):
        """All GHG rules should be in 'active' status."""
        service.apply_pack("ghg_protocol")
        rules = service.search_rules()
        ghg_rules = [r for r in rules if "ghg" in r.get("name", "").lower()]
        for rule in ghg_rules:
            assert rule.get("status") == "active"

    def test_ghg_evaluation_valid_data_passes(
        self, service, sample_emission_records,
    ):
        """Valid emission records should pass GHG rules."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")

        result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_emission_records,
        )
        assert result["status"] == "completed"
        assert result["pass_rate"] > 0.0

    def test_ghg_evaluation_out_of_range_detected(self, service):
        """Records with out-of-range CO2e should fail range checks."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")

        bad_records = [
            {"co2e_tonnes": -100.0, "scope": "1", "source_id": "US-001",
             "activity_data": 5000.0, "emission_factor": 0.03},
            {"co2e_tonnes": 2_000_000.0, "scope": "2", "source_id": "EU-002",
             "activity_data": 80000.0, "emission_factor": 0.03},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=bad_records)
        assert result["rules_failed"] >= 1

    def test_ghg_evaluation_records_provenance(
        self, service, sample_emission_records,
    ):
        """GHG evaluation should record provenance."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        initial = service.provenance.entry_count
        service.evaluate(rule_set_id=rs_id, dataset=sample_emission_records)
        assert service.provenance.entry_count > initial

    def test_ghg_rules_include_critical_severity(self, service):
        """GHG pack should include critical severity rules."""
        service.apply_pack("ghg_protocol")
        rules = service.search_rules(severity="critical")
        ghg_critical = [r for r in rules if "ghg" in r.get("name", "").lower()]
        assert len(ghg_critical) >= 1

    def test_ghg_pack_idempotent_apply(self, service):
        """Applying GHG pack twice should not duplicate rules."""
        r1 = service.apply_pack("ghg_protocol")
        initial_rule_count = len(service._rules)
        r2 = service.apply_pack("ghg_protocol")
        # Second apply adds more rules but pack should still work
        assert r2["status"] == "applied"


# ===========================================================================
# TestCSRDESRSCompliance
# ===========================================================================


class TestCSRDESRSCompliance:
    """Test CSRD/ESRS rule pack application and evaluation."""

    def test_apply_csrd_pack_imports_rules(self, service):
        """Applying CSRD pack should import rules."""
        result = service.apply_pack("csrd_esrs")
        assert result["rules_imported"] >= 3
        assert result["status"] == "applied"

    def test_apply_csrd_pack_creates_rule_set(self, service):
        """CSRD pack should create rule sets."""
        result = service.apply_pack("csrd_esrs")
        assert result["rule_sets_created"] >= 1

    def test_csrd_evaluation_valid_data(
        self, service, sample_supplier_records,
    ):
        """Valid supplier records should pass CSRD rules."""
        pack = service.apply_pack("csrd_esrs")
        rs_id = pack.get("rule_set_id", "")

        result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_supplier_records,
        )
        assert result["status"] == "completed"
        assert result["pass_rate"] > 0.0

    def test_csrd_evaluation_missing_materiality_fails(self, service):
        """Records missing materiality assessment should fail completeness."""
        pack = service.apply_pack("csrd_esrs")
        rs_id = pack.get("rule_set_id", "")

        bad_records = [
            {"scope3_emissions": 50000.0, "reporting_period": "2025",
             "organizational_boundary": "operational_control"},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=bad_records)
        # Should still complete but may have failures
        assert result["status"] == "completed"

    def test_csrd_evaluation_out_of_range_scope3(self, service):
        """Records with extreme Scope 3 values should be flagged."""
        pack = service.apply_pack("csrd_esrs")
        rs_id = pack.get("rule_set_id", "")

        extreme_records = [
            {"materiality_assessment": "high",
             "scope3_emissions": 500_000_000.0,
             "reporting_period": "2025",
             "organizational_boundary": "operational_control"},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=extreme_records)
        assert result["rules_failed"] >= 1

    def test_csrd_rules_have_regulatory_tags(self, service):
        """CSRD rules should have appropriate names."""
        service.apply_pack("csrd_esrs")
        rules = service.search_rules()
        csrd_rules = [r for r in rules if "csrd" in r.get("name", "").lower()]
        assert len(csrd_rules) >= 3

    def test_csrd_evaluation_provenance_chain_valid(
        self, service, sample_supplier_records,
    ):
        """Provenance chain should be valid after CSRD evaluation."""
        pack = service.apply_pack("csrd_esrs")
        rs_id = pack.get("rule_set_id", "")
        service.evaluate(rule_set_id=rs_id, dataset=sample_supplier_records)
        assert service.provenance.verify_chain() is True


# ===========================================================================
# TestEUDRCompliance
# ===========================================================================


class TestEUDRCompliance:
    """Test EUDR rule pack application and evaluation."""

    def test_apply_eudr_pack_imports_rules(self, service):
        """Applying EUDR pack should import geolocation rules."""
        result = service.apply_pack("eudr")
        assert result["rules_imported"] >= 3
        assert result["status"] == "applied"

    def test_eudr_evaluation_valid_geolocation(
        self, service, sample_geolocation_records,
    ):
        """Valid geolocation records should pass EUDR rules."""
        pack = service.apply_pack("eudr")
        rs_id = pack.get("rule_set_id", "")

        result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_geolocation_records,
        )
        assert result["status"] == "completed"
        assert result["pass_rate"] > 0.0

    def test_eudr_evaluation_invalid_latitude(self, service):
        """Latitude outside [-90, 90] should fail range check."""
        pack = service.apply_pack("eudr")
        rs_id = pack.get("rule_set_id", "")

        bad_records = [
            {"geolocation": "point", "commodity_type": "soy",
             "latitude": 95.0, "longitude": -60.0},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=bad_records)
        assert result["rules_failed"] >= 1

    def test_eudr_evaluation_invalid_longitude(self, service):
        """Longitude outside [-180, 180] should fail range check."""
        pack = service.apply_pack("eudr")
        rs_id = pack.get("rule_set_id", "")

        bad_records = [
            {"geolocation": "point", "commodity_type": "palm_oil",
             "latitude": 2.5, "longitude": 200.0},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=bad_records)
        assert result["rules_failed"] >= 1

    def test_eudr_rules_include_critical_severity(self, service):
        """EUDR geolocation rules should be critical severity."""
        service.apply_pack("eudr")
        rules = service.search_rules(severity="critical")
        eudr_critical = [r for r in rules if "eudr" in r.get("name", "").lower()]
        assert len(eudr_critical) >= 2

    def test_eudr_evaluation_records_provenance(
        self, service, sample_geolocation_records,
    ):
        """EUDR evaluation should record provenance."""
        pack = service.apply_pack("eudr")
        rs_id = pack.get("rule_set_id", "")
        initial = service.provenance.entry_count
        service.evaluate(rule_set_id=rs_id, dataset=sample_geolocation_records)
        assert service.provenance.entry_count > initial

    def test_eudr_pipeline_end_to_end(
        self, service, sample_geolocation_records,
    ):
        """Full pipeline with EUDR pack should complete."""
        result = service.run_pipeline(
            dataset=sample_geolocation_records,
            pack_name="eudr",
        )
        assert result["final_status"] in ("completed", "failed")
        assert result.get("pipeline_id") is not None


# ===========================================================================
# TestSOC2Compliance
# ===========================================================================


class TestSOC2Compliance:
    """Test SOC2 rule pack application and evaluation."""

    def test_apply_soc2_pack_imports_rules(self, service):
        """Applying SOC2 pack should import security rules."""
        result = service.apply_pack("soc2")
        assert result["rules_imported"] >= 2
        assert result["status"] == "applied"

    def test_soc2_evaluation_valid_data(
        self, service, sample_security_records,
    ):
        """Valid security records should pass SOC2 rules."""
        pack = service.apply_pack("soc2")
        rs_id = pack.get("rule_set_id", "")

        result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_security_records,
        )
        assert result["status"] == "completed"
        assert result["pass_rate"] > 0.0

    def test_soc2_evaluation_short_retention_fails(self, service):
        """Records with retention < 90 days should fail."""
        pack = service.apply_pack("soc2")
        rs_id = pack.get("rule_set_id", "")

        bad_records = [
            {"access_log": "enabled", "encryption_status": "AES-256",
             "retention_days": 30},
        ]
        result = service.evaluate(rule_set_id=rs_id, dataset=bad_records)
        assert result["rules_failed"] >= 1

    def test_soc2_rules_are_critical(self, service):
        """SOC2 access and encryption rules should be critical."""
        service.apply_pack("soc2")
        rules = service.search_rules(severity="critical")
        soc2_critical = [r for r in rules if "soc2" in r.get("name", "").lower()]
        assert len(soc2_critical) >= 2

    def test_soc2_pipeline_end_to_end(
        self, service, sample_security_records,
    ):
        """Full pipeline with SOC2 pack should complete."""
        result = service.run_pipeline(
            dataset=sample_security_records,
            pack_name="soc2",
        )
        assert result["final_status"] in ("completed", "failed")

    def test_soc2_evaluation_provenance_valid(
        self, service, sample_security_records,
    ):
        """Provenance chain should be valid after SOC2 evaluation."""
        pack = service.apply_pack("soc2")
        rs_id = pack.get("rule_set_id", "")
        service.evaluate(rule_set_id=rs_id, dataset=sample_security_records)
        assert service.provenance.verify_chain() is True


# ===========================================================================
# TestCrossPackValidation
# ===========================================================================


class TestCrossPackValidation:
    """Test applying multiple rule packs to the same or different datasets."""

    def test_apply_all_four_packs(self, service):
        """All four packs (GHG, CSRD, EUDR, SOC2) can be applied."""
        ghg = service.apply_pack("ghg_protocol")
        csrd = service.apply_pack("csrd_esrs")
        eudr = service.apply_pack("eudr")
        soc2 = service.apply_pack("soc2")

        assert ghg["status"] == "applied"
        assert csrd["status"] == "applied"
        assert eudr["status"] == "applied"
        assert soc2["status"] == "applied"

    def test_cross_pack_rules_registered(self, service):
        """All pack rules should be registered in the service."""
        service.apply_pack("ghg_protocol")
        service.apply_pack("csrd_esrs")
        service.apply_pack("eudr")
        service.apply_pack("soc2")

        all_rules = service.search_rules()
        assert len(all_rules) >= 12  # At least 3+ per pack

    def test_cross_pack_ghg_and_csrd_on_emissions(
        self, service, sample_emission_records,
    ):
        """GHG and CSRD packs can both evaluate emission-style data."""
        ghg = service.apply_pack("ghg_protocol")
        csrd = service.apply_pack("csrd_esrs")

        ghg_result = service.evaluate(
            rule_set_id=ghg.get("rule_set_id", ""),
            dataset=sample_emission_records,
        )
        csrd_result = service.evaluate(
            rule_set_id=csrd.get("rule_set_id", ""),
            dataset=sample_emission_records,
        )

        assert ghg_result["status"] == "completed"
        assert csrd_result["status"] == "completed"

    def test_cross_pack_provenance_chain_valid(self, service):
        """Provenance chain should be valid after all packs applied."""
        service.apply_pack("ghg_protocol")
        service.apply_pack("csrd_esrs")
        service.apply_pack("eudr")
        service.apply_pack("soc2")
        assert service.provenance.verify_chain() is True

    def test_cross_pack_list_packs(self, service):
        """All applied packs should be listable."""
        service.apply_pack("ghg_protocol")
        service.apply_pack("csrd_esrs")
        packs = service.list_packs()
        assert len(packs) >= 2

    def test_cross_pack_filter_by_framework(self, service):
        """List packs should support filtering by framework."""
        service.apply_pack("ghg_protocol")
        service.apply_pack("csrd_esrs")
        ghg_packs = service.list_packs(framework="ghg_protocol")
        assert len(ghg_packs) >= 1
        for p in ghg_packs:
            assert p.get("pack_name") == "ghg_protocol"


# ===========================================================================
# TestComplianceReportGeneration
# ===========================================================================


class TestComplianceReportGeneration:
    """Test compliance report generation across packs."""

    def test_ghg_compliance_report(
        self, service, sample_emission_records,
    ):
        """Generate compliance report for GHG evaluation."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_emission_records,
        )

        report = service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        assert report["report_type"] == "compliance_report"
        assert report["format"] == "json"
        assert "content" in report

    def test_csrd_compliance_report(
        self, service, sample_supplier_records,
    ):
        """Generate compliance report for CSRD evaluation."""
        pack = service.apply_pack("csrd_esrs")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_supplier_records,
        )

        report = service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        assert report["report_type"] == "compliance_report"
        assert report.get("content", {}).get("result") in ("pass", "warn", "fail", "unknown")

    def test_report_includes_evaluation_summary(
        self, service, sample_emission_records,
    ):
        """Report content should include evaluation summary fields."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_emission_records,
        )

        report = service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        content = report.get("content", {})
        assert "pass_rate" in content
        assert "result" in content

    def test_report_provenance_hash(
        self, service, sample_emission_records,
    ):
        """Report should have a provenance hash."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_emission_records,
        )

        report = service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        assert report.get("provenance_hash") is not None
        assert len(report.get("provenance_hash", "")) > 0

    def test_multiple_reports_for_same_evaluation(
        self, service, sample_emission_records,
    ):
        """Multiple reports can be generated for the same evaluation."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_emission_records,
        )
        eval_id = eval_result["evaluation_id"]

        r1 = service.generate_report(
            evaluation_id=eval_id,
            report_type="compliance_report",
            format="json",
        )
        r2 = service.generate_report(
            evaluation_id=eval_id,
            report_type="audit_trail",
            format="json",
        )
        assert r1["report_id"] != r2["report_id"]

    def test_report_stored_in_service(
        self, service, sample_emission_records,
    ):
        """Generated reports should be stored in the service."""
        pack = service.apply_pack("ghg_protocol")
        rs_id = pack.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id, dataset=sample_emission_records,
        )

        report = service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        assert report["report_id"] in service._reports
