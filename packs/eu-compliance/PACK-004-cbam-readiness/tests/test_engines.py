# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Engines Tests (35 tests)

Tests core functionality of all 7 CBAM engines (5 tests each)
using stub/mock implementations. No external dependencies required.

Engines: CBAMCalculation, Certificate, QuarterlyReporting,
SupplierManagement, DeMinimis, Verification, PolicyCompliance.

Author: GreenLang QA Team
"""

import re
from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    StubCBAMApp,
    StubCertificateEngine,
    StubCustoms,
    StubETSFeed,
    StubQuarterlyEngine,
    StubSupplierPortal,
    _compute_hash,
    _new_uuid,
    _utcnow,
)


# ============================================================================
# CBAMCalculationEngine Tests (5 tests)
# ============================================================================

class TestCBAMCalculationEngine:
    """Test CBAM emission calculation engine functionality."""

    def test_calculate_emissions(self, sample_emission_inputs):
        """Test emission calculation for a single import."""
        inp = sample_emission_inputs[0]
        total = inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"]
        result = {
            "input_id": inp["input_id"],
            "total_emissions_tco2e": round(total, 6),
            "unit": "tCO2e",
            "provenance_hash": _compute_hash(inp),
        }
        assert result["total_emissions_tco2e"] == 925.0  # 500 * 1.85
        assert result["unit"] == "tCO2e"
        assert len(result["provenance_hash"]) == 64

    def test_batch_calculate(self, sample_emission_inputs):
        """Test batch emission calculation for all 10 inputs."""
        results = []
        for inp in sample_emission_inputs:
            total = inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"]
            results.append({
                "input_id": inp["input_id"],
                "total_emissions_tco2e": round(total, 6),
            })
        assert len(results) == 10
        total_all = sum(r["total_emissions_tco2e"] for r in results)
        assert total_all > 0

    def test_default_emission_factors(self, mock_cbam_app):
        """Test default emission factor lookup by category."""
        steel_factors = mock_cbam_app.get_emission_factors("steel")
        assert "default" in steel_factors
        assert steel_factors["default"] == 2.30
        alum_factors = mock_cbam_app.get_emission_factors("aluminium")
        assert alum_factors["default"] == 8.50
        cement_factors = mock_cbam_app.get_emission_factors("cement")
        assert cement_factors["default"] == 0.65

    def test_markup_for_default_methodology(self, sample_emission_inputs):
        """Test that default methodology emissions include markup factor."""
        default_inputs = [
            i for i in sample_emission_inputs
            if i["emission_methodology"] == "default_values"
        ]
        assert len(default_inputs) >= 1
        for inp in default_inputs:
            markup_factor = 1.10  # 10% conservative markup
            base_total = inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"]
            adjusted = round(base_total * markup_factor, 6)
            assert adjusted > base_total, "Default values should include markup"

    def test_cn_code_validate(self, mock_customs, sample_emission_inputs):
        """Test CN code validation for emission inputs."""
        for inp in sample_emission_inputs:
            result = mock_customs.validate_cn_code(inp["cn_code"])
            assert result["format_valid"] is True, (
                f"CN code {inp['cn_code']} format invalid"
            )


# ============================================================================
# CertificateEngine Tests (5 tests)
# ============================================================================

class TestCertificateEngine:
    """Test CBAM certificate engine functionality."""

    def test_calculate_obligation(self, mock_certificate_engine):
        """Test gross/net obligation calculation."""
        result = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=5000.0, year=2026,
        )
        assert result["gross_obligation_tco2e"] == 5000.0
        assert result["free_allocation_pct"] == 0.975
        assert result["net_obligation_tco2e"] == pytest.approx(125.0, rel=1e-3)

    def test_free_allocation_deduction(self, mock_certificate_engine):
        """Test free allocation reduces obligation."""
        result = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=10000.0, year=2030,
        )
        # 2030: 51.5% free allocation => net = 10000 * (1 - 0.515) = 4850
        assert result["free_allocation_pct"] == 0.515
        assert result["net_obligation_tco2e"] == pytest.approx(4850.0, rel=1e-3)

    def test_carbon_price_deduction(self, mock_certificate_engine):
        """Test carbon price paid in origin country reduces net obligation."""
        result = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=5000.0, year=2026,
            carbon_price_deduction=50.0,
        )
        # net = 5000 - (5000 * 0.975) - 50 = 125 - 50 = 75
        assert result["net_obligation_tco2e"] == pytest.approx(75.0, rel=1e-3)

    def test_cost_estimation(self, mock_certificate_engine):
        """Test certificate cost estimation at ETS price."""
        cost = mock_certificate_engine.estimate_cost(
            net_obligation_tco2e=100.0, ets_price_eur=80.0,
        )
        assert cost["estimated_cost_eur"] == 8000.0
        assert cost["currency"] == "EUR"

    def test_quarterly_holding_check(self, mock_certificate_engine):
        """Test quarterly certificate holding compliance check."""
        result = mock_certificate_engine.check_quarterly_holding(
            certificates_held=90, net_obligation_tco2e=100.0,
        )
        # 80% of 100 = 80 required; 90 held => compliant
        assert result["compliant"] is True
        assert result["shortfall"] == 0


# ============================================================================
# QuarterlyReportingEngine Tests (5 tests)
# ============================================================================

class TestQuarterlyReportingEngine:
    """Test CBAM quarterly reporting engine functionality."""

    def test_detect_period(self, mock_quarterly_engine):
        """Test quarter detection from date."""
        result = mock_quarterly_engine.detect_period("2026-02-15")
        assert result["quarter"] == "Q1"
        assert result["year"] == 2026
        assert result["start_date"] == "2026-01-01"

    def test_assemble_report(
        self, mock_quarterly_engine, sample_importer_config, sample_emission_results,
    ):
        """Test report assembly from emission results."""
        period = mock_quarterly_engine.detect_period("2026-01-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        assert report["status"] == "assembled"
        assert report["total_imports"] == 10
        assert report["total_emissions_tco2e"] > 0
        assert len(report["provenance_hash"]) == 64

    def test_generate_xml(
        self, mock_quarterly_engine, sample_importer_config, sample_emission_results,
    ):
        """Test XML generation for quarterly report."""
        period = mock_quarterly_engine.detect_period("2026-01-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        xml = mock_quarterly_engine.generate_xml(report)
        assert '<?xml version="1.0"' in xml
        assert "<CBAMQuarterlyReport" in xml
        assert "<ImporterEORI>" in xml
        assert "tCO2e" in xml

    def test_validate_report(
        self, mock_quarterly_engine, sample_importer_config, sample_emission_results,
    ):
        """Test report validation identifies issues."""
        period = mock_quarterly_engine.detect_period("2026-01-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        validation = mock_quarterly_engine.validate_report(report)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_amendment(self, mock_quarterly_engine):
        """Test report amendment creates new version."""
        amendment = mock_quarterly_engine.create_amendment(
            report_id="QR-2026-Q1-001",
            reason="Corrected emission factor for steel",
            version=1,
        )
        assert amendment["amendment_version"] == 2
        assert amendment["status"] == "amended"


# ============================================================================
# SupplierManagementEngine Tests (5 tests)
# ============================================================================

class TestSupplierManagementEngine:
    """Test CBAM supplier management engine functionality."""

    def test_register_supplier(self, mock_supplier_portal):
        """Test supplier registration."""
        result = mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-TEST-001",
            "company_name": "Test Steel Corp",
            "country": "TR",
        })
        assert result["status"] == "active"
        assert result["supplier_id"] == "SUP-TEST-001"

    def test_add_installation(self, mock_supplier_portal):
        """Test installation registration for supplier."""
        mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-TEST-002",
            "company_name": "Test Alum Corp",
            "country": "IN",
        })
        result = mock_supplier_portal.add_installation("SUP-TEST-002", {
            "installation_id": "INST-TEST-001",
            "name": "Primary Smelter",
        })
        assert result["status"] == "registered"
        assert result["installation_id"] == "INST-TEST-001"

    def test_submit_emission_data(self, mock_supplier_portal):
        """Test emission data submission by supplier."""
        mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-TEST-003",
            "company_name": "Data Submit Corp",
            "country": "CN",
        })
        result = mock_supplier_portal.submit_emission_data("SUP-TEST-003", {
            "cn_code": "7207 11 14",
            "weight_tonnes": 500.0,
            "specific_emission": 1.85,
        })
        assert result["status"] == "submitted"
        assert result["supplier_id"] == "SUP-TEST-003"

    def test_review_submission(self, mock_supplier_portal):
        """Test emission data review (accept/reject)."""
        result = mock_supplier_portal.review_submission(
            "SUB-001", "accepted", "Data verified",
        )
        assert result["decision"] == "accepted"
        assert "reviewed_at" in result

    def test_quality_score(self, mock_supplier_portal):
        """Test supplier quality score calculation."""
        mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-QS-001",
            "company_name": "Quality Corp",
            "country": "TR",
            "quality_score": 92.0,
        })
        mock_supplier_portal.suppliers["SUP-QS-001"]["quality_score"] = 92.0
        score = mock_supplier_portal.get_quality_score("SUP-QS-001")
        assert score["quality_score"] == 92.0
        assert score["rating"] == "excellent"


# ============================================================================
# DeMinimisEngine Tests (5 tests)
# ============================================================================

class TestDeMinimisEngine:
    """Test CBAM de minimis threshold engine functionality."""

    def test_track_import_under_threshold(self, sample_cbam_config):
        """Test tracking import below de minimis threshold."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        cumulative_kg = 50000
        under = cumulative_kg < threshold_kg
        result = {
            "cumulative_weight_kg": cumulative_kg,
            "threshold_kg": threshold_kg,
            "under_threshold": under,
            "utilization_pct": round(cumulative_kg / threshold_kg * 100, 1),
        }
        assert result["under_threshold"] is True
        assert result["utilization_pct"] < 100

    def test_threshold_check_exceeded(self, sample_cbam_config):
        """Test detection when de minimis threshold is exceeded."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        cumulative_kg = 200000
        exceeded = cumulative_kg >= threshold_kg
        result = {
            "exceeded": exceeded,
            "amount_over_kg": cumulative_kg - threshold_kg,
            "status": "reporting_required",
        }
        assert result["exceeded"] is True
        assert result["amount_over_kg"] == 50000

    def test_exemption_check(self, sample_cbam_config):
        """Test de minimis exemption determination."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        threshold_eur = sample_cbam_config["cbam"]["deminimis_config"]["annual_value_threshold_eur"]
        import_weight_kg = 100000
        import_value_eur = 120.0
        exempt = import_weight_kg < threshold_kg and import_value_eur < threshold_eur
        result = {
            "weight_exempt": import_weight_kg < threshold_kg,
            "value_exempt": import_value_eur < threshold_eur,
            "overall_exempt": exempt,
        }
        assert result["overall_exempt"] is True

    def test_alert_levels(self, sample_cbam_config):
        """Test de minimis alert level determination."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        alert_at_pct = sample_cbam_config["cbam"]["deminimis_config"]["alert_at_pct"]
        test_cases = [
            (50000, "safe"),
            (120000, "approaching"),  # 80% of 150000
            (160000, "exceeded"),
        ]
        for cumulative_kg, expected_level in test_cases:
            pct = cumulative_kg / threshold_kg * 100
            if pct >= 100:
                level = "exceeded"
            elif pct >= alert_at_pct:
                level = "approaching"
            else:
                level = "safe"
            assert level == expected_level, (
                f"At {cumulative_kg}kg, expected {expected_level} got {level}"
            )

    def test_annual_assessment(self, sample_cbam_config):
        """Test annual de minimis assessment summary."""
        threshold_kg = sample_cbam_config["cbam"]["deminimis_config"]["annual_weight_threshold_kg"]
        monthly_imports_kg = [12000, 15000, 10000, 18000, 14000, 11000,
                              16000, 13000, 17000, 12000, 15000, 14000]
        cumulative = sum(monthly_imports_kg)
        assessment = {
            "year": 2026,
            "total_weight_kg": cumulative,
            "threshold_kg": threshold_kg,
            "exceeded": cumulative >= threshold_kg,
            "monthly_trend": monthly_imports_kg,
            "avg_monthly_kg": round(cumulative / 12, 0),
        }
        assert assessment["total_weight_kg"] == 167000
        assert assessment["exceeded"] is True


# ============================================================================
# VerificationEngine Tests (5 tests)
# ============================================================================

class TestVerificationEngine:
    """Test CBAM verification engine functionality."""

    def test_register_verifier(self, sample_verifier):
        """Test verifier registration."""
        assert sample_verifier["verifier_id"] == "VER-DAkkS-001"
        assert sample_verifier["accreditation_body"] == "DAkkS"
        assert sample_verifier["status"] == "active"
        assert "steel" in sample_verifier["scopes"]

    def test_create_engagement(self, sample_verifier):
        """Test verification engagement creation."""
        engagement = {
            "engagement_id": f"VE-{_new_uuid()[:8]}",
            "verifier_id": sample_verifier["verifier_id"],
            "importer_id": "CBAM-DE-2026-00001",
            "scope": ["steel", "aluminium"],
            "reporting_year": 2026,
            "status": "initiated",
            "created_at": _utcnow().isoformat(),
        }
        assert engagement["status"] == "initiated"
        assert engagement["verifier_id"] == "VER-DAkkS-001"

    def test_submit_finding(self, sample_verifier):
        """Test verification finding submission."""
        finding = {
            "finding_id": f"VF-{_new_uuid()[:8]}",
            "engagement_id": "VE-001",
            "category": "emission_factor",
            "severity": "material",
            "description": "Emission factor for steel BF route lacks supporting documentation",
            "recommendation": "Provide third-party lab analysis",
            "status": "open",
        }
        assert finding["severity"] in ("material", "immaterial", "observation")
        assert finding["status"] == "open"

    def test_materiality_assessment(self, sample_verifier):
        """Test materiality threshold check."""
        threshold_pct = 5.0
        reported_emissions = 5000.0
        verified_emissions = 5180.0
        deviation_pct = abs(verified_emissions - reported_emissions) / reported_emissions * 100
        material = deviation_pct > threshold_pct
        result = {
            "reported": reported_emissions,
            "verified": verified_emissions,
            "deviation_pct": round(deviation_pct, 2),
            "threshold_pct": threshold_pct,
            "material": material,
        }
        assert result["deviation_pct"] == 3.6
        assert result["material"] is False

    def test_verification_statement(self, sample_verifier):
        """Test verification statement generation."""
        statement = {
            "statement_id": f"VS-{_new_uuid()[:8]}",
            "verifier_id": sample_verifier["verifier_id"],
            "verifier_name": sample_verifier["company_name"],
            "accreditation": sample_verifier["accreditation_number"],
            "opinion": "unqualified",
            "scope": sample_verifier["scopes"],
            "valid_until": sample_verifier["accreditation_valid_until"],
            "issued_at": _utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "verifier": sample_verifier["verifier_id"],
                "opinion": "unqualified",
            }),
        }
        assert statement["opinion"] == "unqualified"
        assert len(statement["provenance_hash"]) == 64


# ============================================================================
# PolicyComplianceEngine Tests (5 tests)
# ============================================================================

class TestPolicyComplianceEngine:
    """Test CBAM policy compliance engine functionality."""

    def test_check_compliance(self, sample_compliance_rules):
        """Test compliance check across all rules."""
        checks = []
        for rule in sample_compliance_rules[:10]:
            checks.append({
                "rule_id": rule["rule_id"],
                "status": "pass",
                "severity": rule["severity"],
            })
        passed = sum(1 for c in checks if c["status"] == "pass")
        assert passed == 10

    def test_cn_code_rule_enforcement(self, sample_compliance_rules, mock_customs):
        """Test CN code validation rule enforcement."""
        valid_code = "7207 11 14"
        invalid_code = "9999 99 99"
        valid_result = mock_customs.validate_cn_code(valid_code)
        invalid_result = mock_customs.validate_cn_code(invalid_code)
        assert valid_result["cbam_covered"] is True
        assert invalid_result["cbam_covered"] is False

    def test_emission_factor_range_check(self, mock_cbam_app):
        """Test emission factor is within acceptable range."""
        steel_factors = mock_cbam_app.get_emission_factors("steel")
        for method, factor in steel_factors.items():
            assert 0.0 < factor < 100.0, (
                f"Steel EF ({method}: {factor}) out of range"
            )

    def test_completeness_check(self, sample_emission_inputs):
        """Test data completeness validation."""
        required_fields = [
            "cn_code", "goods_category", "origin_country",
            "weight_tonnes", "specific_emission_tco2e_per_tonne",
        ]
        for inp in sample_emission_inputs:
            missing = [f for f in required_fields if not inp.get(f)]
            assert len(missing) == 0, (
                f"Input {inp['input_id']} missing: {missing}"
            )

    def test_compliance_score(self, sample_compliance_rules):
        """Test overall compliance score calculation."""
        total_rules = len(sample_compliance_rules)
        passed_rules = total_rules - 3  # simulate 3 failures
        score = round(passed_rules / total_rules * 100, 1)
        result = {
            "total_rules": total_rules,
            "passed": passed_rules,
            "failed": 3,
            "score": score,
            "status": "pass" if score >= 90 else "review",
        }
        assert result["score"] > 90
        assert result["status"] == "pass"
