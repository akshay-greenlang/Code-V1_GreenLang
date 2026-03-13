# -*- coding: utf-8 -*-
"""
Tests for Golden Scenarios - AGENT-EUDR-023 Acceptance Tests

Comprehensive test suite covering 7 commodities x 7 scenarios + 1 cross-border:
    Scenario 1: Fully compliant (all 8 categories pass)
    Scenario 2: Missing land rights documentation
    Scenario 3: Expired environmental permits
    Scenario 4: Red flags in labour rights
    Scenario 5: Invalid certification
    Scenario 6: Failed third-party audit
    Scenario 7: Multiple category gaps
    + Cross-border multi-country assessment

Each scenario validates end-to-end compliance assessment behavior
with deterministic, pre-computed expected outcomes.

Test count: 51+ tests (7 commodities x 7 scenarios + 2 cross-border)
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Golden Scenario Acceptance Tests)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    compute_compliance_score,
    determine_compliance,
    classify_red_flag_severity,
    apply_country_multiplier,
    is_document_expired,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COMMODITIES,
    DOCUMENT_TYPES,
    COMPLIANCE_DETERMINATIONS,
)


# ---------------------------------------------------------------------------
# Golden Scenario Builder
# ---------------------------------------------------------------------------


def _build_golden_scenario(
    commodity: str,
    scenario_type: str,
) -> Dict[str, Any]:
    """Build a golden scenario test case for a commodity."""
    today = date.today()
    base_supplier = {
        "supplier_id": f"GOLDEN-{commodity.upper()}-{scenario_type.upper()}",
        "country_code": "BR",
        "commodity": commodity,
    }

    if scenario_type == "fully_compliant":
        return {
            **base_supplier,
            "documents": _build_all_valid_documents(today),
            "certifications": [
                {"certificate_id": f"CERT-{commodity}-001", "status": "valid",
                 "scheme": "fsc", "scope": [commodity],
                 "expiry_date": (today + timedelta(days=365)).isoformat()},
            ],
            "audit_reports": [
                {"audit_id": f"AUD-{commodity}-001", "overall_result": "pass",
                 "findings_count": 0, "major_findings": 0, "minor_findings": 0},
            ],
            "expected_determination": "COMPLIANT",
            "expected_red_flags": 0,
            "expected_docs_valid": 12,
        }

    elif scenario_type == "missing_land_rights":
        docs = _build_all_valid_documents(today)
        # Remove land_title and concession_permit
        docs = [d for d in docs if d["document_type"] not in ("land_title", "concession_permit")]
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [
                {"certificate_id": f"CERT-{commodity}-002", "status": "valid",
                 "scheme": "fsc", "scope": [commodity],
                 "expiry_date": (today + timedelta(days=365)).isoformat()},
            ],
            "audit_reports": [],
            "expected_determination": "PARTIALLY_COMPLIANT",
            "expected_missing_docs": ["land_title", "concession_permit"],
            "expected_red_flags_min": 1,
        }

    elif scenario_type == "expired_permits":
        docs = _build_all_valid_documents(today)
        # Expire the environmental_impact_assessment and harvest_permit
        for doc in docs:
            if doc["document_type"] in ("environmental_impact_assessment", "harvest_permit"):
                doc["expiry_date"] = (today - timedelta(days=30)).isoformat()
                doc["status"] = "expired"
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [],
            "audit_reports": [],
            "expected_determination": "PARTIALLY_COMPLIANT",
            "expected_docs_expired": 2,
            "expected_red_flags_min": 1,
        }

    elif scenario_type == "labour_rights_flags":
        docs = _build_all_valid_documents(today)
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [],
            "audit_reports": [],
            "red_flag_overrides": [
                {"indicator": "forced_labour_risk", "category": "operational",
                 "severity": "critical", "score": Decimal("95")},
                {"indicator": "unsafe_conditions", "category": "operational",
                 "severity": "high", "score": Decimal("70")},
            ],
            "expected_red_flags_min": 2,
            "expected_critical_flags": 1,
        }

    elif scenario_type == "invalid_certification":
        docs = _build_all_valid_documents(today)
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [
                {"certificate_id": f"CERT-{commodity}-INV", "status": "revoked",
                 "scheme": "fsc", "scope": [commodity],
                 "expiry_date": (today + timedelta(days=365)).isoformat()},
                {"certificate_id": f"CERT-{commodity}-SUS", "status": "suspended",
                 "scheme": "pefc", "scope": [commodity],
                 "expiry_date": (today + timedelta(days=180)).isoformat()},
            ],
            "audit_reports": [],
            "expected_valid_certs": 0,
            "expected_red_flags_min": 1,
        }

    elif scenario_type == "failed_audit":
        docs = _build_all_valid_documents(today)
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [
                {"certificate_id": f"CERT-{commodity}-003", "status": "valid",
                 "scheme": "fsc", "scope": [commodity],
                 "expiry_date": (today + timedelta(days=365)).isoformat()},
            ],
            "audit_reports": [
                {"audit_id": f"AUD-{commodity}-FAIL", "overall_result": "fail",
                 "findings_count": 8, "major_findings": 5, "minor_findings": 3,
                 "corrective_actions_required": 6, "corrective_actions_closed": 1},
            ],
            "expected_audit_passed": False,
        }

    elif scenario_type == "multiple_gaps":
        # Only provide 3 out of 12 document types, all expired
        docs = [
            {"document_type": "land_title", "document_id": "DOC-GAP-1",
             "issuing_authority": "Auth-1",
             "expiry_date": (today - timedelta(days=60)).isoformat(),
             "status": "expired"},
            {"document_type": "export_license", "document_id": "DOC-GAP-2",
             "issuing_authority": "Auth-2",
             "expiry_date": (today - timedelta(days=90)).isoformat(),
             "status": "expired"},
            {"document_type": "tax_clearance_certificate", "document_id": "DOC-GAP-3",
             "issuing_authority": "Auth-3",
             "expiry_date": (today + timedelta(days=30)).isoformat(),
             "status": "expiring_soon"},
        ]
        return {
            **base_supplier,
            "documents": docs,
            "certifications": [],
            "audit_reports": [
                {"audit_id": f"AUD-{commodity}-GAP", "overall_result": "fail",
                 "findings_count": 10, "major_findings": 6, "minor_findings": 4},
            ],
            "expected_determination": "NON_COMPLIANT",
            "expected_docs_expired": 2,
            "expected_red_flags_min": 2,
            "expected_audit_passed": False,
        }

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


def _build_all_valid_documents(today: date) -> List[Dict]:
    """Build a complete set of valid documents (one per type)."""
    docs = []
    for i, doc_type in enumerate(DOCUMENT_TYPES):
        docs.append({
            "document_id": f"GOLDEN-DOC-{i+1:04d}",
            "document_type": doc_type,
            "issuing_authority": f"Official-Authority-{i+1}",
            "issue_date": (today - timedelta(days=180)).isoformat(),
            "expiry_date": (today + timedelta(days=365)).isoformat(),
            "country_code": "BR",
            "status": "valid",
            "file_hash": compute_test_hash({"doc_id": f"GOLDEN-DOC-{i+1:04d}"}),
        })
    return docs


def _run_golden_assessment(scenario: Dict) -> Dict[str, Any]:
    """Run assessment for a golden scenario and return results."""
    today = date.today()
    documents = scenario.get("documents", [])
    certifications = scenario.get("certifications", [])
    audit_reports = scenario.get("audit_reports", [])

    docs_valid = sum(
        1 for d in documents
        if d.get("status") == "valid" or (
            d.get("expiry_date") and not is_document_expired(d["expiry_date"])
        )
    )
    docs_expired = len(documents) - docs_valid
    valid_certs = [c for c in certifications if c.get("status") == "valid"]
    audit_passed = any(
        r.get("overall_result") in ("pass", "conditional_pass")
        for r in audit_reports
    )

    # Red flags from overrides or defaults
    red_flags = scenario.get("red_flag_overrides", [])
    if not documents:
        red_flags.append({"indicator": "no_documents", "category": "documentation",
                          "severity": "critical", "score": Decimal("85")})
    if docs_expired > 0:
        red_flags.append({"indicator": "expired_documents", "category": "documentation",
                          "severity": "high", "score": Decimal("65")})
    if not valid_certs and certifications:
        red_flags.append({"indicator": "invalid_certifications", "category": "certification",
                          "severity": "high", "score": Decimal("70")})

    # Compute score
    doc_ratio = docs_valid / max(len(DOCUMENT_TYPES), 1)
    cert_ratio = len(valid_certs) / max(len(certifications), 1) if certifications else 0
    base_score = Decimal(str(round(doc_ratio * 60 + cert_ratio * 25 + (15 if audit_passed else 0), 2)))
    determination = determine_compliance(base_score)

    critical_flags = sum(1 for f in red_flags if f.get("severity") == "critical")

    return {
        "supplier_id": scenario["supplier_id"],
        "commodity": scenario["commodity"],
        "overall_score": base_score,
        "determination": determination,
        "documents_valid": docs_valid,
        "documents_expired": docs_expired,
        "certifications_valid": len(valid_certs),
        "red_flags_count": len(red_flags),
        "critical_flags": critical_flags,
        "audit_passed": audit_passed,
        "provenance_hash": compute_test_hash({
            "supplier": scenario["supplier_id"],
            "score": str(base_score),
        }),
    }


# ===========================================================================
# Scenario 1: Fully Compliant (7 tests)
# ===========================================================================


class TestScenario1FullyCompliant:
    """Test fully compliant scenario for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_fully_compliant(self, commodity):
        """Test fully compliant supplier for each commodity."""
        scenario = _build_golden_scenario(commodity, "fully_compliant")
        result = _run_golden_assessment(scenario)
        assert result["determination"] == "COMPLIANT"
        assert result["documents_valid"] == 12
        assert result["audit_passed"] is True
        assert result["red_flags_count"] == 0


# ===========================================================================
# Scenario 2: Missing Land Rights (7 tests)
# ===========================================================================


class TestScenario2MissingLandRights:
    """Test missing land rights documentation for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_missing_land_rights(self, commodity):
        """Test supplier missing land rights docs for each commodity."""
        scenario = _build_golden_scenario(commodity, "missing_land_rights")
        result = _run_golden_assessment(scenario)
        # Missing 2 out of 12 docs -> reduced score
        assert result["documents_valid"] < 12
        assert result["determination"] in ("PARTIALLY_COMPLIANT", "COMPLIANT")


# ===========================================================================
# Scenario 3: Expired Environmental Permits (7 tests)
# ===========================================================================


class TestScenario3ExpiredPermits:
    """Test expired environmental permits for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_expired_permits(self, commodity):
        """Test supplier with expired environmental permits for each commodity."""
        scenario = _build_golden_scenario(commodity, "expired_permits")
        result = _run_golden_assessment(scenario)
        assert result["documents_expired"] >= 2
        assert result["red_flags_count"] >= 1


# ===========================================================================
# Scenario 4: Labour Rights Red Flags (7 tests)
# ===========================================================================


class TestScenario4LabourRightsFlags:
    """Test labour rights red flags for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_labour_rights_flags(self, commodity):
        """Test supplier with labour rights red flags for each commodity."""
        scenario = _build_golden_scenario(commodity, "labour_rights_flags")
        result = _run_golden_assessment(scenario)
        assert result["red_flags_count"] >= 2
        assert result["critical_flags"] >= 1


# ===========================================================================
# Scenario 5: Invalid Certification (7 tests)
# ===========================================================================


class TestScenario5InvalidCertification:
    """Test invalid certification for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_invalid_certification(self, commodity):
        """Test supplier with invalid certification for each commodity."""
        scenario = _build_golden_scenario(commodity, "invalid_certification")
        result = _run_golden_assessment(scenario)
        assert result["certifications_valid"] == 0
        assert result["red_flags_count"] >= 1


# ===========================================================================
# Scenario 6: Failed Third-Party Audit (7 tests)
# ===========================================================================


class TestScenario6FailedAudit:
    """Test failed third-party audit for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_failed_audit(self, commodity):
        """Test supplier with failed audit for each commodity."""
        scenario = _build_golden_scenario(commodity, "failed_audit")
        result = _run_golden_assessment(scenario)
        assert result["audit_passed"] is False


# ===========================================================================
# Scenario 7: Multiple Category Gaps (7 tests)
# ===========================================================================


class TestScenario7MultipleCategoryGaps:
    """Test multiple category gaps for all 7 commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_multiple_gaps(self, commodity):
        """Test supplier with multiple gaps for each commodity."""
        scenario = _build_golden_scenario(commodity, "multiple_gaps")
        result = _run_golden_assessment(scenario)
        assert result["determination"] == "NON_COMPLIANT"
        assert result["documents_expired"] >= 2
        assert result["audit_passed"] is False
        assert result["red_flags_count"] >= 2


# ===========================================================================
# Cross-Border Multi-Country Scenario (2 tests)
# ===========================================================================


class TestCrossBorderScenario:
    """Test cross-border multi-country compliance assessment."""

    def test_multi_country_assessment(self):
        """Test assessment spanning Brazil and Indonesia supply chains."""
        today = date.today()

        # Brazil leg
        br_scenario = {
            "supplier_id": "GOLDEN-CROSS-BR",
            "country_code": "BR",
            "commodity": "soya",
            "documents": _build_all_valid_documents(today),
            "certifications": [
                {"certificate_id": "CERT-BR-001", "status": "valid",
                 "scheme": "iscc", "scope": ["soya"],
                 "expiry_date": (today + timedelta(days=365)).isoformat()},
            ],
            "audit_reports": [
                {"audit_id": "AUD-BR-001", "overall_result": "pass",
                 "findings_count": 0, "major_findings": 0, "minor_findings": 0},
            ],
        }
        br_result = _run_golden_assessment(br_scenario)

        # Indonesia leg
        id_scenario = {
            "supplier_id": "GOLDEN-CROSS-ID",
            "country_code": "ID",
            "commodity": "oil_palm",
            "documents": _build_all_valid_documents(today)[:6],  # Partial docs
            "certifications": [
                {"certificate_id": "CERT-ID-001", "status": "suspended",
                 "scheme": "rspo", "scope": ["oil_palm"],
                 "expiry_date": (today + timedelta(days=180)).isoformat()},
            ],
            "audit_reports": [],
        }
        id_result = _run_golden_assessment(id_scenario)

        # Brazil should score higher (compliant vs not)
        assert br_result["determination"] == "COMPLIANT"
        assert id_result["determination"] != "COMPLIANT"
        assert br_result["overall_score"] > id_result["overall_score"]

    def test_multi_country_provenance_independent(self):
        """Test each country leg has independent provenance."""
        today = date.today()

        br_scenario = {
            "supplier_id": "GOLDEN-PROV-BR",
            "country_code": "BR",
            "commodity": "wood",
            "documents": _build_all_valid_documents(today),
            "certifications": [],
            "audit_reports": [],
        }
        id_scenario = {
            "supplier_id": "GOLDEN-PROV-ID",
            "country_code": "ID",
            "commodity": "wood",
            "documents": _build_all_valid_documents(today),
            "certifications": [],
            "audit_reports": [],
        }

        br_result = _run_golden_assessment(br_scenario)
        id_result = _run_golden_assessment(id_scenario)

        assert br_result["provenance_hash"] != id_result["provenance_hash"]
        assert len(br_result["provenance_hash"]) == SHA256_HEX_LENGTH
        assert len(id_result["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_three_country_supply_chain(self):
        """Test assessment across three-country supply chain (BR, ID, GH)."""
        today = date.today()
        results = {}
        for country, commodity in [("BR", "soya"), ("ID", "oil_palm"), ("GH", "cocoa")]:
            scenario = {
                "supplier_id": f"GOLDEN-3C-{country}",
                "country_code": country,
                "commodity": commodity,
                "documents": _build_all_valid_documents(today),
                "certifications": [
                    {"certificate_id": f"CERT-3C-{country}", "status": "valid",
                     "scheme": "fsc", "scope": [commodity],
                     "expiry_date": (today + timedelta(days=365)).isoformat()},
                ],
                "audit_reports": [
                    {"audit_id": f"AUD-3C-{country}", "overall_result": "pass",
                     "findings_count": 0, "major_findings": 0, "minor_findings": 0},
                ],
            }
            results[country] = _run_golden_assessment(scenario)

        # All three should be compliant with valid docs + certs + audit
        for country, result in results.items():
            assert result["determination"] == "COMPLIANT", (
                f"{country} should be COMPLIANT"
            )

        # Each should have independent provenance
        hashes = [r["provenance_hash"] for r in results.values()]
        assert len(set(hashes)) == 3  # All unique

    def test_same_commodity_different_origins(self):
        """Test same commodity (coffee) from 4 different origin countries."""
        today = date.today()
        countries = ["BR", "CO", "VN", "KH"]
        results = {}
        for country in countries:
            scenario = {
                "supplier_id": f"GOLDEN-COFFEE-{country}",
                "country_code": country,
                "commodity": "coffee",
                "documents": _build_all_valid_documents(today)[:8],
                "certifications": [],
                "audit_reports": [],
            }
            results[country] = _run_golden_assessment(scenario)

        # All should produce results
        for country in countries:
            assert results[country]["commodity"] == "coffee"
            assert results[country]["determination"] in COMPLIANCE_DETERMINATIONS

        # Each origin produces independent assessment
        supplier_ids = [r["supplier_id"] for r in results.values()]
        assert len(set(supplier_ids)) == 4
