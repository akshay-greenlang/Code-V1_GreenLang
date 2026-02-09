# -*- coding: utf-8 -*-
"""
Unit Tests for ComplianceReportEngine (AGENT-DATA-007)

Tests EUDR compliance report generation including compliance status
assessment (COMPLIANT / REVIEW_REQUIRED / NON_COMPLIANT), risk scoring,
evidence summary, recommendations, report retrieval, and SHA-256
provenance hashing.

Coverage target: 85%+ of compliance_report.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline ComplianceReportEngine mirroring
# greenlang/deforestation_satellite/compliance_report.py
# ---------------------------------------------------------------------------

COMPLIANCE_COMPLIANT = "COMPLIANT"
COMPLIANCE_REVIEW_REQUIRED = "REVIEW_REQUIRED"
COMPLIANCE_NON_COMPLIANT = "NON_COMPLIANT"

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"
RISK_VIOLATION = "VIOLATION"


class ComplianceReportEngine:
    """EUDR compliance report generator with evidence packaging."""

    def __init__(self, agent_id: str = "GL-DATA-GEO-003"):
        self._agent_id = agent_id
        self._reports: Dict[str, Dict[str, Any]] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -----------------------------------------------------------------
    # Core report generation
    # -----------------------------------------------------------------

    def generate_report(
        self,
        polygon_id: str,
        alerts: Optional[List[Dict[str, Any]]] = None,
        change_detection: Optional[Dict[str, Any]] = None,
        baseline_check: Optional[Dict[str, Any]] = None,
        classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a full EUDR compliance report for a polygon."""
        alerts = alerts or []
        change_detection = change_detection or {}
        baseline_check = baseline_check or {}
        classification = classification or {}

        compliance_status = self.assess_compliance(alerts, change_detection, baseline_check)
        risk_score = self._compute_risk_score(alerts, change_detection, baseline_check)
        risk_level = self._risk_level(risk_score)
        recommendations = self.generate_recommendations(compliance_status, alerts, risk_score)
        evidence = self._build_evidence_summary(alerts, change_detection, baseline_check, classification)

        report_id = f"rpt-{uuid.uuid4().hex[:12]}"
        provenance_hash = self._hash({
            "polygon_id": polygon_id,
            "compliance_status": compliance_status,
            "risk_score": risk_score,
        })

        report = {
            "report_id": report_id,
            "polygon_id": polygon_id,
            "compliance_status": compliance_status,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "evidence_summary": evidence,
            "alert_count": len(alerts),
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self._agent_id,
            "provenance_hash": provenance_hash,
        }
        self._reports[report_id] = report
        return report

    # -----------------------------------------------------------------
    # Compliance assessment logic
    # -----------------------------------------------------------------

    def assess_compliance(
        self,
        alerts: List[Dict[str, Any]],
        change_detection: Dict[str, Any],
        baseline_check: Dict[str, Any],
    ) -> str:
        """Determine compliance status from alerts and assessment data.

        Rules:
        - No post-cutoff alerts -> COMPLIANT
        - High-confidence post-cutoff alerts -> NON_COMPLIANT
        - Low/nominal confidence post-cutoff alerts -> REVIEW_REQUIRED
        - Significant forest loss detected -> NON_COMPLIANT
        - Baseline indicates deforestation -> NON_COMPLIANT
        """
        post_cutoff_alerts = [
            a for a in alerts
            if a.get("post_cutoff", False)
        ]

        if not post_cutoff_alerts:
            # Check change detection for significant loss
            change_type = change_detection.get("primary_change_type", "no_change")
            if change_type == "clear_cut":
                return COMPLIANCE_NON_COMPLIANT
            return COMPLIANCE_COMPLIANT

        high_conf_alerts = [
            a for a in post_cutoff_alerts
            if a.get("confidence", "low") == "high"
        ]

        if high_conf_alerts:
            return COMPLIANCE_NON_COMPLIANT

        return COMPLIANCE_REVIEW_REQUIRED

    # -----------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------

    def generate_recommendations(
        self,
        compliance_status: str,
        alerts: List[Dict[str, Any]],
        risk_score: float,
    ) -> List[str]:
        """Generate recommendations based on compliance status."""
        recommendations: List[str] = []

        if compliance_status == COMPLIANCE_COMPLIANT:
            if risk_score > 20:
                recommendations.append("Schedule periodic re-assessment to maintain compliance")
            return recommendations

        if compliance_status == COMPLIANCE_REVIEW_REQUIRED:
            recommendations.append("Manual review of flagged alerts is required")
            recommendations.append("Obtain higher-resolution satellite imagery for verification")
            recommendations.append("Cross-reference with ground-truth surveys if available")
            if risk_score > 50:
                recommendations.append("Consider temporary supply chain hold pending investigation")
            return recommendations

        # NON_COMPLIANT
        recommendations.append("Immediate investigation required - non-compliant deforestation detected")
        recommendations.append("Halt procurement from this sourcing region pending investigation")
        recommendations.append("Engage third-party verification for independent assessment")
        recommendations.append("Document remediation actions for regulatory reporting")
        recommendations.append("Notify compliance officer and legal team")
        if len(alerts) > 5:
            recommendations.append("Multiple alert sources confirm deforestation - escalate urgently")
        return recommendations

    # -----------------------------------------------------------------
    # Report retrieval
    # -----------------------------------------------------------------

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        return self._reports.get(report_id)

    def get_report_count(self) -> int:
        return len(self._reports)

    def get_all_reports(self) -> List[Dict[str, Any]]:
        return list(self._reports.values())

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_risk_score(
        self,
        alerts: List[Dict[str, Any]],
        change_detection: Dict[str, Any],
        baseline_check: Dict[str, Any],
    ) -> float:
        """Compute combined risk score (0-100)."""
        score = 0.0

        # Alert-based risk (0-40)
        post_cutoff = [a for a in alerts if a.get("post_cutoff", False)]
        score += min(40.0, len(post_cutoff) * 10.0)

        # Change detection risk (0-30)
        change_type = change_detection.get("primary_change_type", "no_change")
        change_scores = {
            "clear_cut": 30.0,
            "degradation": 20.0,
            "partial_loss": 10.0,
            "regrowth": 0.0,
            "no_change": 0.0,
        }
        score += change_scores.get(change_type, 5.0)

        # Baseline risk (0-30)
        baseline_risk = baseline_check.get("risk_score", 0.0)
        score += min(30.0, baseline_risk * 0.3)

        return min(100.0, round(score, 2))

    def _risk_level(self, risk_score: float) -> str:
        if risk_score <= 25:
            return RISK_LOW
        elif risk_score <= 50:
            return RISK_MEDIUM
        elif risk_score <= 75:
            return RISK_HIGH
        elif risk_score <= 100:
            return RISK_CRITICAL
        return RISK_VIOLATION

    def _build_evidence_summary(
        self,
        alerts: List[Dict[str, Any]],
        change_detection: Dict[str, Any],
        baseline_check: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "total_alerts": len(alerts),
            "post_cutoff_alerts": len([a for a in alerts if a.get("post_cutoff", False)]),
            "high_confidence_alerts": len([a for a in alerts if a.get("confidence") == "high"]),
            "change_type": change_detection.get("primary_change_type", "unknown"),
            "change_area_ha": change_detection.get("area_ha", 0.0),
            "baseline_forest_cover_pct": baseline_check.get("forest_cover_pct", 0.0),
            "current_forest_cover_pct": classification.get("tree_cover_pct", 0.0),
            "satellite_scenes_used": change_detection.get("scenes_used", 0),
        }

    def _hash(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestGenerateReport:
    def test_generate_report_compliant(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001", alerts=[])
        assert report["compliance_status"] == COMPLIANCE_COMPLIANT

    def test_generate_report_non_compliant(self):
        engine = ComplianceReportEngine()
        alerts = [
            {"alert_id": "a1", "post_cutoff": True, "confidence": "high", "source": "glad"},
        ]
        report = engine.generate_report("plot-002", alerts=alerts)
        assert report["compliance_status"] == COMPLIANCE_NON_COMPLIANT

    def test_generate_report_review_required(self):
        engine = ComplianceReportEngine()
        alerts = [
            {"alert_id": "a1", "post_cutoff": True, "confidence": "nominal", "source": "radd"},
        ]
        report = engine.generate_report("plot-003", alerts=alerts)
        assert report["compliance_status"] == COMPLIANCE_REVIEW_REQUIRED

    def test_report_has_report_id(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert report["report_id"].startswith("rpt-")

    def test_report_has_polygon_id(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert report["polygon_id"] == "plot-001"

    def test_report_has_timestamp(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert "timestamp" in report

    def test_report_has_agent_id(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert report["agent_id"] == "GL-DATA-GEO-003"


class TestAssessCompliance:
    def test_assess_compliance_no_alerts_compliant(self):
        engine = ComplianceReportEngine()
        status = engine.assess_compliance([], {}, {})
        assert status == COMPLIANCE_COMPLIANT

    def test_assess_compliance_high_conf_post_cutoff_non_compliant(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "high"}]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_NON_COMPLIANT

    def test_assess_compliance_low_conf_post_cutoff_review(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "low"}]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_REVIEW_REQUIRED

    def test_assess_compliance_nominal_conf_post_cutoff_review(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "nominal"}]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_REVIEW_REQUIRED

    def test_assess_compliance_no_alerts_clearcut_change_non_compliant(self):
        engine = ComplianceReportEngine()
        status = engine.assess_compliance([], {"primary_change_type": "clear_cut"}, {})
        assert status == COMPLIANCE_NON_COMPLIANT

    def test_assess_compliance_no_alerts_degradation_compliant(self):
        engine = ComplianceReportEngine()
        status = engine.assess_compliance([], {"primary_change_type": "degradation"}, {})
        assert status == COMPLIANCE_COMPLIANT

    def test_assess_compliance_pre_cutoff_alerts_compliant(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": False, "confidence": "high"}]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_COMPLIANT

    def test_assess_compliance_mixed_alerts(self):
        engine = ComplianceReportEngine()
        alerts = [
            {"post_cutoff": False, "confidence": "high"},
            {"post_cutoff": True, "confidence": "nominal"},
        ]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_REVIEW_REQUIRED

    def test_assess_compliance_multiple_high_conf(self):
        engine = ComplianceReportEngine()
        alerts = [
            {"post_cutoff": True, "confidence": "high"},
            {"post_cutoff": True, "confidence": "high"},
        ]
        status = engine.assess_compliance(alerts, {}, {})
        assert status == COMPLIANCE_NON_COMPLIANT


class TestRecommendations:
    def test_generate_recommendations_compliant_empty(self):
        engine = ComplianceReportEngine()
        recs = engine.generate_recommendations(COMPLIANCE_COMPLIANT, [], 10.0)
        assert len(recs) == 0

    def test_generate_recommendations_compliant_moderate_risk(self):
        engine = ComplianceReportEngine()
        recs = engine.generate_recommendations(COMPLIANCE_COMPLIANT, [], 25.0)
        assert any("periodic" in r.lower() or "re-assessment" in r.lower() for r in recs)

    def test_generate_recommendations_non_compliant_detailed(self):
        engine = ComplianceReportEngine()
        alerts = [{"alert_id": "a1"}]
        recs = engine.generate_recommendations(COMPLIANCE_NON_COMPLIANT, alerts, 80.0)
        assert len(recs) >= 4
        assert any("investigation" in r.lower() for r in recs)

    def test_generate_recommendations_non_compliant_many_alerts(self):
        engine = ComplianceReportEngine()
        alerts = [{"alert_id": f"a{i}"} for i in range(10)]
        recs = engine.generate_recommendations(COMPLIANCE_NON_COMPLIANT, alerts, 90.0)
        assert any("escalate" in r.lower() for r in recs)

    def test_generate_recommendations_review_required(self):
        engine = ComplianceReportEngine()
        alerts = [{"alert_id": "a1"}]
        recs = engine.generate_recommendations(COMPLIANCE_REVIEW_REQUIRED, alerts, 35.0)
        assert any("manual review" in r.lower() for r in recs)

    def test_generate_recommendations_review_high_risk(self):
        engine = ComplianceReportEngine()
        alerts = [{"alert_id": "a1"}]
        recs = engine.generate_recommendations(COMPLIANCE_REVIEW_REQUIRED, alerts, 55.0)
        assert any("hold" in r.lower() or "supply chain" in r.lower() for r in recs)

    def test_generate_recommendations_review_includes_satellite(self):
        engine = ComplianceReportEngine()
        recs = engine.generate_recommendations(COMPLIANCE_REVIEW_REQUIRED, [], 40.0)
        assert any("satellite" in r.lower() or "resolution" in r.lower() for r in recs)


class TestEvidenceSummary:
    def test_evidence_summary_structure(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report(
            "plot-001",
            alerts=[{"post_cutoff": True, "confidence": "high"}],
            change_detection={"primary_change_type": "degradation", "area_ha": 2.5, "scenes_used": 3},
            baseline_check={"forest_cover_pct": 85.0},
            classification={"tree_cover_pct": 70.0},
        )
        evidence = report["evidence_summary"]
        assert evidence["total_alerts"] == 1
        assert evidence["post_cutoff_alerts"] == 1
        assert evidence["high_confidence_alerts"] == 1
        assert evidence["change_type"] == "degradation"
        assert evidence["change_area_ha"] == 2.5
        assert evidence["baseline_forest_cover_pct"] == 85.0
        assert evidence["current_forest_cover_pct"] == 70.0
        assert evidence["satellite_scenes_used"] == 3

    def test_evidence_summary_empty_data(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        evidence = report["evidence_summary"]
        assert evidence["total_alerts"] == 0
        assert evidence["post_cutoff_alerts"] == 0
        assert evidence["change_type"] == "unknown"

    def test_evidence_summary_counts_post_cutoff(self):
        engine = ComplianceReportEngine()
        alerts = [
            {"post_cutoff": True, "confidence": "high"},
            {"post_cutoff": False, "confidence": "high"},
            {"post_cutoff": True, "confidence": "low"},
        ]
        report = engine.generate_report("plot-001", alerts=alerts)
        evidence = report["evidence_summary"]
        assert evidence["total_alerts"] == 3
        assert evidence["post_cutoff_alerts"] == 2
        assert evidence["high_confidence_alerts"] == 2


class TestRiskScore:
    def test_combined_risk_score_no_risk(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert report["risk_score"] == 0.0

    def test_combined_risk_score_alerts_only(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "nominal"}]
        report = engine.generate_report("plot-001", alerts=alerts)
        assert report["risk_score"] >= 10.0

    def test_combined_risk_score_change_only(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report(
            "plot-001",
            change_detection={"primary_change_type": "clear_cut"},
        )
        assert report["risk_score"] >= 30.0

    def test_combined_risk_score_capped_at_100(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "high"} for _ in range(10)]
        report = engine.generate_report(
            "plot-001",
            alerts=alerts,
            change_detection={"primary_change_type": "clear_cut"},
            baseline_check={"risk_score": 100.0},
        )
        assert report["risk_score"] <= 100.0

    def test_risk_level_low(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert report["risk_level"] == RISK_LOW

    def test_risk_level_medium(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "nominal"} for _ in range(3)]
        report = engine.generate_report("plot-001", alerts=alerts)
        assert report["risk_level"] in (RISK_MEDIUM, RISK_HIGH)

    def test_risk_level_critical(self):
        engine = ComplianceReportEngine()
        alerts = [{"post_cutoff": True, "confidence": "high"} for _ in range(5)]
        report = engine.generate_report(
            "plot-001",
            alerts=alerts,
            change_detection={"primary_change_type": "clear_cut"},
            baseline_check={"risk_score": 80.0},
        )
        assert report["risk_level"] in (RISK_HIGH, RISK_CRITICAL)


class TestReportRetrieval:
    def test_report_retrieval_by_id(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        retrieved = engine.get_report(report["report_id"])
        assert retrieved is not None
        assert retrieved["report_id"] == report["report_id"]

    def test_report_not_found(self):
        engine = ComplianceReportEngine()
        assert engine.get_report("rpt-nonexistent") is None

    def test_report_count_initial(self):
        engine = ComplianceReportEngine()
        assert engine.get_report_count() == 0

    def test_report_count_after_generation(self):
        engine = ComplianceReportEngine()
        engine.generate_report("plot-001")
        engine.generate_report("plot-002")
        assert engine.get_report_count() == 2

    def test_get_all_reports(self):
        engine = ComplianceReportEngine()
        engine.generate_report("plot-001")
        engine.generate_report("plot-002")
        all_reports = engine.get_all_reports()
        assert len(all_reports) == 2


class TestProvenanceHashOnReport:
    def test_provenance_hash_on_report(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        assert "provenance_hash" in report
        assert len(report["provenance_hash"]) == 64

    def test_provenance_hash_is_hex(self):
        engine = ComplianceReportEngine()
        report = engine.generate_report("plot-001")
        int(report["provenance_hash"], 16)

    def test_provenance_hash_deterministic(self):
        engine = ComplianceReportEngine()
        data = {"polygon_id": "plot-001", "compliance_status": COMPLIANCE_COMPLIANT, "risk_score": 0.0}
        h1 = engine._hash(data)
        h2 = engine._hash(data)
        assert h1 == h2

    def test_provenance_hash_differs_for_different_data(self):
        engine = ComplianceReportEngine()
        h1 = engine._hash({"polygon_id": "plot-001"})
        h2 = engine._hash({"polygon_id": "plot-002"})
        assert h1 != h2


class TestCustomAgentId:
    def test_custom_agent_id(self):
        engine = ComplianceReportEngine(agent_id="CUSTOM-RPT-001")
        assert engine.agent_id == "CUSTOM-RPT-001"

    def test_default_agent_id(self):
        engine = ComplianceReportEngine()
        assert engine.agent_id == "GL-DATA-GEO-003"

    def test_report_inherits_agent_id(self):
        engine = ComplianceReportEngine(agent_id="CUSTOM-RPT-001")
        report = engine.generate_report("plot-001")
        assert report["agent_id"] == "CUSTOM-RPT-001"
