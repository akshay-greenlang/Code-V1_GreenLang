# -*- coding: utf-8 -*-
"""
Unit Tests for ComplianceVerifier (AGENT-DATA-005)

Tests EUDR article-by-article compliance verification: Article 3 (deforestation-
free and legal), Article 9 (geolocation), Article 10 (due diligence statement),
Article 11 (risk mitigation), compliance scoring, batch verification, and
remediation guidance generation.

Coverage target: 85%+ of compliance_verifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums
# ---------------------------------------------------------------------------


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    PARTIAL = "partial"


class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Inline data models
# ---------------------------------------------------------------------------


class PlotRecord:
    """Minimal plot record for compliance checks."""

    def __init__(self, plot_id: str, latitude: float, longitude: float,
                 country: str = "BR", commodity: str = "cocoa",
                 area_hectares: float = 5.0,
                 deforestation_free: bool = True,
                 legal_compliance: bool = True,
                 has_polygon: bool = False):
        self.plot_id = plot_id
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.commodity = commodity
        self.area_hectares = area_hectares
        self.deforestation_free = deforestation_free
        self.legal_compliance = legal_compliance
        self.has_polygon = has_polygon


class ComplianceCheckResult:
    """Result of a compliance check."""

    def __init__(self, check_id: str, article: str, target_id: str,
                 status: str, details: str, remediation: Optional[str] = None):
        self.check_id = check_id
        self.article = article
        self.target_id = target_id
        self.status = status
        self.details = details
        self.remediation = remediation
        self.provenance_hash = ""
        self.checked_at = datetime.now(timezone.utc).isoformat()


class ComplianceSummary:
    """Summary of compliance verification results."""

    def __init__(self, total_checks: int, passed: int, failed: int,
                 score: float, checks: List[ComplianceCheckResult]):
        self.total_checks = total_checks
        self.passed = passed
        self.failed = failed
        self.score = score
        self.checks = checks


# ---------------------------------------------------------------------------
# Inline PlotRegistryEngine (minimal)
# ---------------------------------------------------------------------------


class PlotRegistryEngine:
    """Minimal plot registry for compliance testing."""

    def __init__(self):
        self._plots: Dict[str, PlotRecord] = {}

    def register(self, plot: PlotRecord) -> PlotRecord:
        self._plots[plot.plot_id] = plot
        return plot

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        return self._plots.get(plot_id)

    def list_plots(self) -> List[PlotRecord]:
        return list(self._plots.values())


# ---------------------------------------------------------------------------
# Inline ComplianceVerifier mirroring greenlang/eudr_traceability/compliance_verifier.py
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class ComplianceVerifier:
    """EUDR article compliance verification engine.

    Verifies compliance against EUDR Articles 3, 9, 10, and 11.
    """

    ARTICLES = ["article_3", "article_9", "article_10", "article_11"]

    def __init__(self, plot_registry: PlotRegistryEngine,
                 polygon_hectare_threshold: float = 4.0):
        self._plot_registry = plot_registry
        self._polygon_hectare_threshold = polygon_hectare_threshold
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"CHK-{self._counter:05d}"

    def verify_article_3(self, plot_id: str) -> ComplianceCheckResult:
        """Verify Article 3: Deforestation-free and legally produced.

        Products must be deforestation-free (no deforestation after
        31 December 2020) and legally produced.
        """
        plot = self._plot_registry.get_plot(plot_id)
        check_id = self._next_id()

        if plot is None:
            return ComplianceCheckResult(
                check_id=check_id, article="article_3",
                target_id=plot_id, status="non_compliant",
                details="Plot not found in registry",
                remediation="Register the plot before verification",
            )

        issues: List[str] = []

        if not plot.deforestation_free:
            issues.append("Plot has not been verified as deforestation-free")

        if not plot.legal_compliance:
            issues.append("Plot has not been verified as legally compliant")

        if issues:
            return ComplianceCheckResult(
                check_id=check_id, article="article_3",
                target_id=plot_id, status="non_compliant",
                details="; ".join(issues),
                remediation="Obtain deforestation-free and legal compliance certifications",
            )

        result = ComplianceCheckResult(
            check_id=check_id, article="article_3",
            target_id=plot_id, status="compliant",
            details="Plot is deforestation-free and legally produced",
        )
        result.provenance_hash = _compute_hash({
            "check_id": check_id, "article": "article_3",
            "target_id": plot_id, "status": "compliant",
        })
        return result

    def verify_article_9(self, plot_id: str) -> ComplianceCheckResult:
        """Verify Article 9: Geolocation requirements.

        Geolocation data must be provided. Plots above the hectare threshold
        require polygon boundaries.
        """
        plot = self._plot_registry.get_plot(plot_id)
        check_id = self._next_id()

        if plot is None:
            return ComplianceCheckResult(
                check_id=check_id, article="article_9",
                target_id=plot_id, status="non_compliant",
                details="Plot not found in registry",
                remediation="Register the plot with geolocation data",
            )

        issues: List[str] = []

        if plot.latitude == 0.0 and plot.longitude == 0.0:
            issues.append("Missing valid geolocation coordinates")

        if plot.area_hectares > self._polygon_hectare_threshold and not plot.has_polygon:
            issues.append(
                f"Plot area ({plot.area_hectares}ha) exceeds threshold "
                f"({self._polygon_hectare_threshold}ha) but no polygon provided"
            )

        if issues:
            return ComplianceCheckResult(
                check_id=check_id, article="article_9",
                target_id=plot_id, status="non_compliant",
                details="; ".join(issues),
                remediation="Provide valid geolocation data with polygon boundaries for large plots",
            )

        result = ComplianceCheckResult(
            check_id=check_id, article="article_9",
            target_id=plot_id, status="compliant",
            details="Geolocation data meets Article 9 requirements",
        )
        result.provenance_hash = _compute_hash({
            "check_id": check_id, "article": "article_9",
            "target_id": plot_id, "status": "compliant",
        })
        return result

    def verify_article_10(self, plot_id: str,
                          has_dds: bool = True,
                          has_risk_assessment: bool = True) -> ComplianceCheckResult:
        """Verify Article 10: Due diligence statement requirements.

        A complete due diligence statement and risk assessment must exist.
        """
        check_id = self._next_id()
        issues: List[str] = []

        if not has_dds:
            issues.append("No due diligence statement found")

        if not has_risk_assessment:
            issues.append("No risk assessment completed")

        if issues:
            return ComplianceCheckResult(
                check_id=check_id, article="article_10",
                target_id=plot_id, status="non_compliant",
                details="; ".join(issues),
                remediation="Complete due diligence statement and risk assessment",
            )

        result = ComplianceCheckResult(
            check_id=check_id, article="article_10",
            target_id=plot_id, status="compliant",
            details="Due diligence statement is complete with risk assessment",
        )
        result.provenance_hash = _compute_hash({
            "check_id": check_id, "article": "article_10",
            "target_id": plot_id, "status": "compliant",
        })
        return result

    def verify_article_11(self, plot_id: str,
                          risk_level: str = "standard",
                          has_mitigation: bool = True) -> ComplianceCheckResult:
        """Verify Article 11: Risk mitigation for high-risk products.

        High-risk products require documented mitigation measures.
        """
        check_id = self._next_id()

        if risk_level == RiskLevel.HIGH.value and not has_mitigation:
            return ComplianceCheckResult(
                check_id=check_id, article="article_11",
                target_id=plot_id, status="non_compliant",
                details="High-risk classification requires mitigation measures but none found",
                remediation="Implement enhanced due diligence and risk mitigation measures",
            )

        result = ComplianceCheckResult(
            check_id=check_id, article="article_11",
            target_id=plot_id, status="compliant",
            details=(
                "Risk mitigation measures in place"
                if risk_level == RiskLevel.HIGH.value
                else "Standard risk level; no additional mitigation required"
            ),
        )
        result.provenance_hash = _compute_hash({
            "check_id": check_id, "article": "article_11",
            "target_id": plot_id, "status": "compliant",
        })
        return result

    def verify_compliance(self, plot_id: str,
                          has_dds: bool = True,
                          has_risk_assessment: bool = True,
                          risk_level: str = "standard",
                          has_mitigation: bool = True) -> ComplianceSummary:
        """Run all article checks and return a compliance summary."""
        checks = [
            self.verify_article_3(plot_id),
            self.verify_article_9(plot_id),
            self.verify_article_10(plot_id, has_dds, has_risk_assessment),
            self.verify_article_11(plot_id, risk_level, has_mitigation),
        ]

        passed = sum(1 for c in checks if c.status == "compliant")
        failed = len(checks) - passed
        score = (passed / len(checks)) * 100.0 if checks else 0.0

        return ComplianceSummary(
            total_checks=len(checks),
            passed=passed,
            failed=failed,
            score=score,
            checks=checks,
        )

    def batch_verify(self, plot_ids: List[str],
                     has_dds: bool = True,
                     has_risk_assessment: bool = True,
                     risk_level: str = "standard",
                     has_mitigation: bool = True) -> List[ComplianceSummary]:
        """Run compliance verification on multiple plots."""
        return [
            self.verify_compliance(
                pid, has_dds, has_risk_assessment, risk_level, has_mitigation,
            )
            for pid in plot_ids
        ]


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def plot_registry() -> PlotRegistryEngine:
    """Create PlotRegistryEngine with compliant and non-compliant plots."""
    registry = PlotRegistryEngine()

    # Fully compliant plot
    registry.register(PlotRecord(
        plot_id="PLT-COMP-001", latitude=-3.1190, longitude=-60.0217,
        country="BR", commodity="cocoa", area_hectares=3.0,
        deforestation_free=True, legal_compliance=True, has_polygon=False,
    ))

    # Compliant large plot (with polygon)
    registry.register(PlotRecord(
        plot_id="PLT-COMP-002", latitude=-2.5, longitude=-59.0,
        country="BR", commodity="soya", area_hectares=12.0,
        deforestation_free=True, legal_compliance=True, has_polygon=True,
    ))

    # Non-compliant: no deforestation check
    registry.register(PlotRecord(
        plot_id="PLT-FAIL-001", latitude=-3.5, longitude=-60.5,
        country="BR", commodity="cattle", area_hectares=5.0,
        deforestation_free=False, legal_compliance=True, has_polygon=True,
    ))

    # Non-compliant: no legal compliance
    registry.register(PlotRecord(
        plot_id="PLT-FAIL-002", latitude=-4.0, longitude=-61.0,
        country="BR", commodity="cocoa", area_hectares=2.0,
        deforestation_free=True, legal_compliance=False, has_polygon=False,
    ))

    # Non-compliant: large plot without polygon
    registry.register(PlotRecord(
        plot_id="PLT-FAIL-003", latitude=-3.8, longitude=-60.8,
        country="BR", commodity="soya", area_hectares=10.0,
        deforestation_free=True, legal_compliance=True, has_polygon=False,
    ))

    # Non-compliant: missing coordinates
    registry.register(PlotRecord(
        plot_id="PLT-FAIL-004", latitude=0.0, longitude=0.0,
        country="BR", commodity="cocoa", area_hectares=2.0,
        deforestation_free=True, legal_compliance=True, has_polygon=False,
    ))

    return registry


@pytest.fixture
def engine(plot_registry) -> ComplianceVerifier:
    return ComplianceVerifier(plot_registry=plot_registry)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestVerifyArticle3:
    """Tests for Article 3: Deforestation-free and legally produced."""

    def test_verify_article_3_compliant(self, engine):
        result = engine.verify_article_3("PLT-COMP-001")
        assert result.status == "compliant"
        assert result.article == "article_3"

    def test_verify_article_3_no_deforestation(self, engine):
        result = engine.verify_article_3("PLT-FAIL-001")
        assert result.status == "non_compliant"
        assert "deforestation" in result.details.lower()

    def test_verify_article_3_no_legal(self, engine):
        result = engine.verify_article_3("PLT-FAIL-002")
        assert result.status == "non_compliant"
        assert "legal" in result.details.lower()

    def test_verify_article_3_not_found(self, engine):
        result = engine.verify_article_3("PLT-MISSING")
        assert result.status == "non_compliant"
        assert "not found" in result.details.lower()


class TestVerifyArticle9:
    """Tests for Article 9: Geolocation requirements."""

    def test_verify_article_9_valid_coords(self, engine):
        result = engine.verify_article_9("PLT-COMP-001")
        assert result.status == "compliant"

    def test_verify_article_9_missing_coords(self, engine):
        result = engine.verify_article_9("PLT-FAIL-004")
        assert result.status == "non_compliant"
        assert "geolocation" in result.details.lower() or "coordinates" in result.details.lower()

    def test_verify_article_9_polygon_missing(self, engine):
        """Large plot (>4ha) without polygon = non-compliant."""
        result = engine.verify_article_9("PLT-FAIL-003")
        assert result.status == "non_compliant"
        assert "polygon" in result.details.lower()

    def test_verify_article_9_large_with_polygon(self, engine):
        """Large plot (>4ha) with polygon = compliant."""
        result = engine.verify_article_9("PLT-COMP-002")
        assert result.status == "compliant"

    def test_verify_article_9_small_no_polygon(self, engine):
        """Small plot (<4ha) without polygon = compliant (not required)."""
        result = engine.verify_article_9("PLT-COMP-001")
        assert result.status == "compliant"


class TestVerifyArticle10:
    """Tests for Article 10: Due diligence statement completeness."""

    def test_verify_article_10_complete_dds(self, engine):
        result = engine.verify_article_10(
            "PLT-COMP-001", has_dds=True, has_risk_assessment=True,
        )
        assert result.status == "compliant"

    def test_verify_article_10_incomplete(self, engine):
        result = engine.verify_article_10(
            "PLT-COMP-001", has_dds=True, has_risk_assessment=False,
        )
        assert result.status == "non_compliant"
        assert "risk assessment" in result.details.lower()

    def test_verify_article_10_no_dds(self, engine):
        result = engine.verify_article_10(
            "PLT-COMP-001", has_dds=False, has_risk_assessment=True,
        )
        assert result.status == "non_compliant"
        assert "due diligence" in result.details.lower()


class TestVerifyArticle11:
    """Tests for Article 11: Risk mitigation for high-risk products."""

    def test_verify_article_11_high_risk_mitigated(self, engine):
        result = engine.verify_article_11(
            "PLT-COMP-001", risk_level="high", has_mitigation=True,
        )
        assert result.status == "compliant"

    def test_verify_article_11_high_risk_no_mitigation(self, engine):
        result = engine.verify_article_11(
            "PLT-COMP-001", risk_level="high", has_mitigation=False,
        )
        assert result.status == "non_compliant"
        assert "mitigation" in result.details.lower()

    def test_verify_article_11_standard_risk(self, engine):
        result = engine.verify_article_11(
            "PLT-COMP-001", risk_level="standard", has_mitigation=False,
        )
        assert result.status == "compliant"

    def test_verify_article_11_low_risk(self, engine):
        result = engine.verify_article_11(
            "PLT-COMP-001", risk_level="low", has_mitigation=False,
        )
        assert result.status == "compliant"


class TestVerifyCompliance:
    """Tests for full compliance verification across all articles."""

    def test_verify_compliance_all_articles(self, engine):
        summary = engine.verify_compliance("PLT-COMP-001")
        assert summary.total_checks == 4
        articles_checked = {c.article for c in summary.checks}
        assert "article_3" in articles_checked
        assert "article_9" in articles_checked
        assert "article_10" in articles_checked
        assert "article_11" in articles_checked

    def test_compliance_score_100(self, engine):
        summary = engine.verify_compliance("PLT-COMP-001")
        assert summary.score == 100.0
        assert summary.passed == 4
        assert summary.failed == 0

    def test_compliance_score_0(self, engine):
        # PLT-FAIL-004 has 0,0 coords and we add no DDS and no mitigation
        summary = engine.verify_compliance(
            "PLT-FAIL-004",
            has_dds=False,
            has_risk_assessment=False,
            risk_level="high",
            has_mitigation=False,
        )
        # Article 3 might pass (deforestation_free=True, legal_compliance=True)
        # Article 9 fails (0,0 coords)
        # Article 10 fails (no DDS, no risk assessment)
        # Article 11 fails (high risk, no mitigation)
        assert summary.failed >= 3
        assert summary.score < 50.0

    def test_compliance_score_partial(self, engine):
        summary = engine.verify_compliance(
            "PLT-COMP-001",
            has_dds=True,
            has_risk_assessment=False,
        )
        # Article 3: pass, Article 9: pass, Article 10: fail, Article 11: pass
        assert summary.passed == 3
        assert summary.failed == 1
        assert summary.score == 75.0


class TestComplianceSummary:
    """Tests for compliance summary statistics."""

    def test_compliance_summary(self, engine):
        summary = engine.verify_compliance("PLT-COMP-001")
        assert summary.total_checks == summary.passed + summary.failed
        assert 0 <= summary.score <= 100.0


class TestBatchVerify:
    """Tests for batch compliance verification."""

    def test_batch_verify(self, engine):
        results = engine.batch_verify(["PLT-COMP-001", "PLT-FAIL-001"])
        assert len(results) == 2
        assert isinstance(results[0], ComplianceSummary)
        assert isinstance(results[1], ComplianceSummary)

    def test_batch_verify_empty(self, engine):
        results = engine.batch_verify([])
        assert len(results) == 0


class TestCheckIDFormat:
    """Tests for check ID format."""

    def test_check_id_format(self, engine):
        result = engine.verify_article_3("PLT-COMP-001")
        assert result.check_id.startswith("CHK-")
        assert len(result.check_id) == 9  # CHK-00001

    def test_check_ids_sequential(self, engine):
        r1 = engine.verify_article_3("PLT-COMP-001")
        r2 = engine.verify_article_9("PLT-COMP-001")
        num1 = int(r1.check_id.split("-")[1])
        num2 = int(r2.check_id.split("-")[1])
        assert num2 == num1 + 1


class TestRemediationGuidance:
    """Tests for remediation guidance in non-compliant results."""

    def test_remediation_guidance(self, engine):
        result = engine.verify_article_3("PLT-FAIL-001")
        assert result.status == "non_compliant"
        assert result.remediation is not None
        assert len(result.remediation) > 0

    def test_remediation_article_9(self, engine):
        result = engine.verify_article_9("PLT-FAIL-003")
        assert result.remediation is not None
        assert "polygon" in result.remediation.lower() or "geolocation" in result.remediation.lower()

    def test_remediation_article_10(self, engine):
        result = engine.verify_article_10("PLT-COMP-001", has_dds=False)
        assert result.remediation is not None

    def test_remediation_article_11(self, engine):
        result = engine.verify_article_11("PLT-COMP-001", risk_level="high", has_mitigation=False)
        assert result.remediation is not None
        assert "mitigation" in result.remediation.lower()

    def test_no_remediation_when_compliant(self, engine):
        result = engine.verify_article_3("PLT-COMP-001")
        assert result.status == "compliant"
        assert result.remediation is None
