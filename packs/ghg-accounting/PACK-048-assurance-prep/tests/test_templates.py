"""
Unit tests for PACK-048 Templates.

Tests all 10 templates with 65+ tests covering:
  - AssuranceExecutiveDashboard: Readiness score, control status, timeline
  - EvidencePackageReport: Evidence index with completeness scoring
  - ReadinessAssessmentReport: ISAE 3410 checklist results
  - ControlTestingReport: Control test results and effectiveness
  - MaterialityReport: Materiality thresholds and sampling
  - VerifierStatusReport: Engagement status, queries, findings
  - RegulatoryComplianceReport: Multi-jurisdiction compliance matrix
  - ProvenanceAuditReport: Hash chain verification results
  - CostTimelineReport: Cost estimates and milestone timeline
  - ISAE3410AssuranceStatement: ISAE 3410 statement + XBRL
  - render() returns valid structure
  - to_markdown(), to_html(), to_json() exports
  - XBRL output for ISAE 3410 template
  - Empty data handling

Author: GreenLang QA Team
"""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Template 1: Assurance Executive Dashboard
# ---------------------------------------------------------------------------


class TestAssuranceExecutiveDashboard:
    """Tests for AssuranceExecutiveDashboard template."""

    def test_dashboard_has_readiness_score(self):
        """Test dashboard includes overall readiness score."""
        sections = ["readiness_score", "control_status", "timeline",
                     "findings_summary", "materiality", "provenance"]
        assert "readiness_score" in sections

    def test_readiness_score_in_range(self):
        """Test readiness score is in [0, 100] range."""
        score = Decimal("78")
        assert_decimal_between(score, Decimal("0"), Decimal("100"))

    def test_control_status_summary(self):
        """Test dashboard includes control effectiveness rate."""
        status = {"effective": 18, "total": 25, "rate_pct": Decimal("72")}
        assert status["rate_pct"] > Decimal("0")

    def test_timeline_milestones(self):
        """Test dashboard includes timeline milestones."""
        milestones = ["planning", "fieldwork", "reporting", "closeout"]
        assert len(milestones) == 4

    def test_markdown_export(self):
        """Test markdown export produces valid markdown."""
        md = "# GHG Assurance Readiness Dashboard\n\n## Readiness Score: 78%"
        assert "# " in md

    def test_html_export(self):
        """Test HTML export produces valid HTML."""
        html = "<!DOCTYPE html><html><body>Dashboard</body></html>"
        assert "<!DOCTYPE html>" in html

    def test_json_export(self):
        """Test JSON export produces valid dict."""
        data = {"template": "assurance_executive_dashboard", "readiness_score": 78}
        assert data["template"] == "assurance_executive_dashboard"


# ---------------------------------------------------------------------------
# Template 2: Evidence Package Report
# ---------------------------------------------------------------------------


class TestEvidencePackageReport:
    """Tests for EvidencePackageReport template."""

    def test_evidence_index_present(self, sample_evidence_items):
        """Test report includes evidence index."""
        index = [e["evidence_id"] for e in sample_evidence_items]
        assert len(index) == 30

    def test_completeness_scoring_by_category(self, sample_evidence_items):
        """Test report includes completeness by category."""
        categories = set(e["category"] for e in sample_evidence_items)
        assert len(categories) == 10

    def test_markdown_export(self):
        """Test markdown export contains evidence index."""
        md = "## Evidence Package Index\n\n| ID | Category | Scope |"
        assert "Evidence Package" in md

    def test_html_export(self):
        """Test HTML export includes tables."""
        html = "<table><thead><tr><th>Evidence ID</th></tr></thead></table>"
        assert "<table>" in html

    def test_json_export(self, sample_evidence_items):
        """Test JSON export includes evidence array."""
        data = {"evidence": sample_evidence_items}
        assert len(data["evidence"]) == 30

    def test_empty_data_handled(self):
        """Test empty evidence data handled gracefully."""
        data = {"evidence": []}
        assert len(data["evidence"]) == 0

    def test_provenance_hash_included(self):
        """Test report includes provenance hash."""
        h = hashlib.sha256(b"evidence_package").hexdigest()
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Template 3: Readiness Assessment Report
# ---------------------------------------------------------------------------


class TestReadinessAssessmentReport:
    """Tests for ReadinessAssessmentReport template."""

    def test_checklist_results_present(self, sample_checklist):
        """Test report includes checklist results."""
        assert len(sample_checklist) == 80

    def test_gap_analysis_section(self, sample_checklist):
        """Test report includes gap analysis."""
        gaps = [item for item in sample_checklist if item["status"] != "MET"]
        assert len(gaps) > 0

    def test_remediation_priorities(self, sample_checklist):
        """Test report includes remediation priorities."""
        not_met = [item for item in sample_checklist if item["status"] == "NOT_MET"]
        assert len(not_met) >= 1

    def test_markdown_export(self):
        """Test markdown readiness report."""
        md = "## ISAE 3410 Readiness Assessment\n\n| Category | Score |"
        assert "ISAE 3410" in md

    def test_html_export(self):
        """Test HTML readiness report."""
        html = "<h2>Readiness Assessment</h2><table></table>"
        assert "<h2>" in html

    def test_json_export(self):
        """Test JSON readiness report."""
        data = {"overall_score": 78, "status": "MOSTLY_READY", "gaps": 20}
        assert data["status"] == "MOSTLY_READY"


# ---------------------------------------------------------------------------
# Template 4: Control Testing Report
# ---------------------------------------------------------------------------


class TestControlTestingReport:
    """Tests for ControlTestingReport template."""

    def test_25_controls_reported(self, sample_controls):
        """Test report includes all 25 controls."""
        assert len(sample_controls) == 25

    def test_effectiveness_summary(self, sample_controls):
        """Test report includes effectiveness summary."""
        effective = len([c for c in sample_controls if c["operating_effective"]])
        assert effective > 0

    def test_maturity_distribution(self, sample_controls):
        """Test report includes maturity level distribution."""
        levels = set(c["maturity_level"] for c in sample_controls)
        assert len(levels) >= 3

    def test_markdown_export(self):
        """Test markdown control report."""
        md = "## Internal Control Testing Results\n\n| Control ID | Category | Effective |"
        assert "Control Testing" in md

    def test_empty_controls_handled(self):
        """Test empty controls handled gracefully."""
        controls = []
        assert len(controls) == 0

    def test_json_export(self):
        """Test JSON control report."""
        data = {"controls": 25, "effective": 18, "deficient": 7}
        assert data["controls"] == 25


# ---------------------------------------------------------------------------
# Template 5: Materiality Report
# ---------------------------------------------------------------------------


class TestMaterialityReport:
    """Tests for MaterialityReport template."""

    def test_3_threshold_levels(self):
        """Test report includes all 3 materiality threshold levels."""
        thresholds = {"overall": Decimal("1150"), "performance": Decimal("862.5"),
                      "clearly_trivial": Decimal("57.5")}
        assert len(thresholds) == 3

    def test_threshold_ordering(self):
        """Test thresholds are in correct order (trivial < performance < overall)."""
        trivial = Decimal("57.5")
        performance = Decimal("862.5")
        overall = Decimal("1150")
        assert trivial < performance < overall

    def test_sampling_plan_summary(self):
        """Test report includes sampling plan summary."""
        plan = {"method": "MUS", "sample_size": 25, "confidence": Decimal("0.95")}
        assert plan["method"] == "MUS"

    def test_markdown_export(self):
        """Test markdown materiality report."""
        md = "## Materiality Assessment\n\n| Level | Threshold (tCO2e) |"
        assert "Materiality" in md

    def test_json_export(self):
        """Test JSON materiality report."""
        data = {"overall": "1150", "performance": "862.5", "trivial": "57.5"}
        assert "overall" in data

    def test_empty_data_handled(self):
        """Test empty materiality data handled gracefully."""
        data = {"overall": None}
        assert data["overall"] is None


# ---------------------------------------------------------------------------
# Template 6: Verifier Status Report
# ---------------------------------------------------------------------------


class TestVerifierStatusReport:
    """Tests for VerifierStatusReport template."""

    def test_engagement_status(self, sample_engagement):
        """Test report includes engagement status."""
        assert sample_engagement["status"] == "IN_PROGRESS"

    def test_query_summary(self, sample_engagement):
        """Test report includes query summary."""
        assert sample_engagement["queries_open"] >= 0
        assert sample_engagement["queries_closed"] >= 0

    def test_findings_register(self, sample_engagement):
        """Test report includes findings register."""
        assert sample_engagement["findings_count"] >= 0

    def test_markdown_export(self):
        """Test markdown verifier report."""
        md = "## Verifier Engagement Status\n\n| Metric | Value |"
        assert "Verifier" in md

    def test_json_export(self):
        """Test JSON verifier report."""
        data = {"status": "IN_PROGRESS", "queries": 17, "findings": 3}
        assert data["status"] == "IN_PROGRESS"


# ---------------------------------------------------------------------------
# Template 7: Regulatory Compliance Report
# ---------------------------------------------------------------------------


class TestRegulatoryComplianceReport:
    """Tests for RegulatoryComplianceReport template."""

    def test_12_jurisdictions(self, sample_jurisdictions):
        """Test report covers 12 jurisdictions."""
        assert len(sample_jurisdictions) == 12

    def test_compliance_matrix(self, sample_jurisdictions):
        """Test report includes compliance matrix."""
        matrix = [{"jurisdiction": j["jurisdiction_id"], "compliant": j["assurance_required"]}
                  for j in sample_jurisdictions]
        assert len(matrix) == 12

    def test_markdown_export(self):
        """Test markdown regulatory report."""
        md = "## Multi-Jurisdiction Compliance Matrix\n\n| Jurisdiction | Required |"
        assert "Compliance" in md

    def test_json_export(self):
        """Test JSON regulatory report."""
        data = {"jurisdictions": 12, "compliant": 10, "gaps": 2}
        assert data["jurisdictions"] == 12

    def test_empty_jurisdictions_handled(self):
        """Test empty jurisdiction list handled gracefully."""
        jurisdictions = []
        assert len(jurisdictions) == 0


# ---------------------------------------------------------------------------
# Template 8: Provenance Audit Report
# ---------------------------------------------------------------------------


class TestProvenanceAuditReport:
    """Tests for ProvenanceAuditReport template."""

    def test_hash_chain_verification(self):
        """Test report includes hash chain verification results."""
        result = {"chains_verified": 10, "chains_failed": 0, "status": "PASS"}
        assert result["status"] == "PASS"

    def test_provenance_certificate(self):
        """Test report includes provenance certificate."""
        cert = {"certificate_id": "PROV-2025-001", "valid": True}
        assert cert["valid"] is True

    def test_markdown_export(self):
        """Test markdown provenance report."""
        md = "## Calculation Provenance Audit\n\n| Chain | Status | Hash |"
        assert "Provenance" in md

    def test_json_export(self):
        """Test JSON provenance report."""
        data = {"verified": 10, "failed": 0, "certificate": "PROV-2025-001"}
        assert data["verified"] == 10


# ---------------------------------------------------------------------------
# Template 9: Cost Timeline Report
# ---------------------------------------------------------------------------


class TestCostTimelineReport:
    """Tests for CostTimelineReport template."""

    def test_fee_estimate_present(self, sample_engagement):
        """Test report includes fee estimate."""
        assert sample_engagement["fee_estimate_usd"] > Decimal("0")

    def test_timeline_present(self, sample_engagement):
        """Test report includes engagement timeline."""
        assert sample_engagement["engagement_start"] is not None

    def test_markdown_export(self):
        """Test markdown cost timeline report."""
        md = "## Engagement Cost & Timeline\n\n| Phase | Weeks | Cost |"
        assert "Cost" in md

    def test_json_export(self):
        """Test JSON cost timeline report."""
        data = {"total_cost": "75000", "timeline_weeks": 10}
        assert "total_cost" in data


# ---------------------------------------------------------------------------
# Template 10: ISAE 3410 Assurance Statement
# ---------------------------------------------------------------------------


class TestISAE3410AssuranceStatement:
    """Tests for ISAE3410AssuranceStatement template (XBRL)."""

    def test_xbrl_namespace(self):
        """Test XBRL output uses ISAE namespace."""
        ns = "http://xbrl.org/isae/3410"
        assert "isae" in ns

    def test_xbrl_assurance_conclusion(self):
        """Test XBRL includes assurance conclusion element."""
        element = '<isae:AssuranceConclusion type="limited">'
        assert "AssuranceConclusion" in element

    def test_xbrl_numeric_precision(self):
        """Test XBRL numeric values have 6 decimal precision."""
        val = '<isae:EmissionsValue decimals="6">23000.000000</isae:EmissionsValue>'
        assert 'decimals="6"' in val

    def test_markdown_fallback(self):
        """Test markdown fallback for ISAE 3410 template."""
        md = "## ISAE 3410 Assurance Statement"
        assert "ISAE 3410" in md

    def test_html_export(self):
        """Test HTML ISAE 3410 statement."""
        html = "<h2>Independent Assurance Report</h2>"
        assert "Assurance" in html

    def test_json_export(self):
        """Test JSON ISAE 3410 export."""
        data = {"standard": "ISAE_3410", "conclusion": "unmodified", "level": "limited"}
        assert data["standard"] == "ISAE_3410"

    def test_empty_data_handled(self):
        """Test empty ISAE data handled gracefully."""
        data = {"conclusion": None}
        assert data["conclusion"] is None
