# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.watch.release_orchestrator (F053)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.watch.release_orchestrator import (
    QAGateResult,
    ReleaseReport,
    prepare_release,
    publish_release,
)


def _mock_repo(factor_count=100, certified=80, preview=20):
    """Create a mock FactorCatalogRepository."""
    repo = MagicMock()
    repo.resolve_edition.return_value = "2026.04.0"
    repo.coverage_stats.return_value = {
        "total_factors": factor_count,
        "total_catalog": factor_count,
        "certified": certified,
        "preview": preview,
        "connector_visible": 0,
        "geographies": 5,
        "fuel_types": 10,
        "scopes": {"1": 40, "2": 30, "3": 30},
        "boundaries": {"combustion": 50, "well_to_tank": 50},
        "by_geography": {"US": 50, "GB": 30, "DE": 20},
        "by_fuel_type": {"diesel": 20, "natural_gas": 20, "electricity": 20, "coal": 20, "other": 20},
        "by_status": {"certified": certified, "preview": preview},
    }
    repo.list_factor_summaries.return_value = [
        {"factor_id": f"EF:{i}", "content_hash": f"hash_{i}", "factor_status": "certified"}
        for i in range(factor_count)
    ]
    repo.search_facets.return_value = {
        "facets": {
            "source_id": {"epa": 40, "defra": 30, "ipcc": 30},
        }
    }
    repo.get_changelog.return_value = ["Initial release"]
    repo.get_manifest_dict.return_value = {
        "edition_id": "2026.04.0",
        "status": "pending",
        "label": "Test edition",
        "total_factors": factor_count,
    }
    return repo


class TestQAGateResult:
    def test_passed_gate(self):
        g = QAGateResult(gate_name="Q1_schema", passed=True)
        assert g.passed
        assert g.errors == 0

    def test_failed_gate(self):
        g = QAGateResult(gate_name="Q1_schema", passed=False, errors=1, details=["No factors"])
        assert not g.passed
        assert g.errors == 1


class TestReleaseReport:
    def test_all_gates_passed(self):
        r = ReleaseReport(
            edition_id="2026.04.0",
            previous_edition_id=None,
            timestamp="2026-04-17T00:00:00Z",
            status="ready",
            qa_gates=[
                QAGateResult(gate_name="Q1", passed=True),
                QAGateResult(gate_name="Q2", passed=True),
            ],
        )
        assert r.all_gates_passed()

    def test_not_all_gates_passed(self):
        r = ReleaseReport(
            edition_id="2026.04.0",
            previous_edition_id=None,
            timestamp="2026-04-17T00:00:00Z",
            status="blocked",
            qa_gates=[
                QAGateResult(gate_name="Q1", passed=True),
                QAGateResult(gate_name="Q2", passed=False),
            ],
        )
        assert not r.all_gates_passed()

    def test_is_ready(self):
        r = ReleaseReport(
            edition_id="2026.04.0",
            previous_edition_id=None,
            timestamp="2026-04-17T00:00:00Z",
            status="ready",
            qa_gates=[QAGateResult(gate_name="Q1", passed=True)],
        )
        assert r.is_ready()

    def test_not_ready_with_license_violations(self):
        r = ReleaseReport(
            edition_id="2026.04.0",
            previous_edition_id=None,
            timestamp="2026-04-17T00:00:00Z",
            status="ready",
            qa_gates=[QAGateResult(gate_name="Q1", passed=True)],
            license_violations=3,
        )
        assert not r.is_ready()

    def test_to_dict(self):
        r = ReleaseReport(
            edition_id="2026.04.0",
            previous_edition_id=None,
            timestamp="2026-04-17T00:00:00Z",
            status="ready",
            factor_count=100,
        )
        d = r.to_dict()
        assert d["edition_id"] == "2026.04.0"
        assert d["factor_count"] == 100
        assert "all_gates_passed" in d
        assert "is_ready" in d


class TestPrepareRelease:
    def test_prepare_success(self):
        repo = _mock_repo()
        report = prepare_release(repo, "2026.04.0")
        assert report.edition_id == "2026.04.0"
        assert report.factor_count == 100
        assert report.certified_count == 80
        assert report.preview_count == 20
        assert len(report.qa_gates) == 6
        assert report.all_gates_passed()
        assert len(report.changelog_lines) > 0

    def test_prepare_with_previous_edition(self):
        repo = _mock_repo()
        # Mock compare_editions
        from greenlang.factors.service import FactorCatalogService

        with patch.object(FactorCatalogService, "__init__", return_value=None):
            with patch.object(FactorCatalogService, "compare_editions", return_value={
                "left_edition_id": "2026.03.0",
                "right_edition_id": "2026.04.0",
                "added_factor_ids": ["EF:new1"],
                "removed_factor_ids": [],
                "changed_factor_ids": ["EF:1"],
                "unchanged_count": 99,
            }):
                svc = FactorCatalogService.__new__(FactorCatalogService)
                svc.repo = repo
                with patch("greenlang.factors.watch.release_orchestrator.FactorCatalogService", return_value=svc):
                    report = prepare_release(repo, "2026.04.0", previous_edition_id="2026.03.0")

        assert report.edition_id == "2026.04.0"
        assert report.previous_edition_id == "2026.03.0"

    def test_prepare_invalid_edition(self):
        repo = _mock_repo()
        repo.resolve_edition.side_effect = ValueError("Unknown edition")
        report = prepare_release(repo, "nonexistent")
        assert report.status == "errors"
        assert len(report.blocking_issues) > 0

    def test_prepare_empty_edition(self):
        repo = _mock_repo(factor_count=0, certified=0, preview=0)
        report = prepare_release(repo, "2026.04.0")
        assert not report.all_gates_passed()
        assert report.status == "blocked"

    def test_signoff_checklist(self):
        repo = _mock_repo()
        report = prepare_release(repo, "2026.04.0")
        checklist = report.signoff_checklist
        assert "all_qa_gates_pass" in checklist
        assert "methodology_lead_signoff" in checklist
        assert checklist["all_qa_gates_pass"] is True
        # Human-required items should be False
        assert checklist["methodology_lead_signoff"] is False
        assert checklist["regression_test_passed"] is False


class TestPublishRelease:
    def test_publish_success(self):
        repo = _mock_repo()
        result = publish_release(repo, "2026.04.0", "alice@greenlang.io")
        assert result["edition_id"] == "2026.04.0"
        assert result["status"] == "stable"
        assert result["approved_by"] == "alice@greenlang.io"
        repo.upsert_edition.assert_called_once()

    def test_publish_no_manifest(self):
        repo = _mock_repo()
        repo.get_manifest_dict.return_value = {}
        with pytest.raises(ValueError, match="no manifest"):
            publish_release(repo, "2026.04.0", "alice")

    def test_publish_unknown_edition(self):
        repo = _mock_repo()
        repo.resolve_edition.side_effect = ValueError("Unknown edition")
        with pytest.raises(ValueError):
            publish_release(repo, "nonexistent", "alice")

    def test_publish_with_custom_changelog(self):
        repo = _mock_repo()
        result = publish_release(
            repo, "2026.04.0", "bob",
            changelog=["Custom changelog entry 1", "Custom changelog entry 2"],
        )
        assert result["status"] == "stable"
        # Verify the changelog includes the custom entries + approval
        call_args = repo.upsert_edition.call_args
        final_changelog = call_args.kwargs.get("changelog") or call_args[1].get("changelog")
        assert "Custom changelog entry 1" in final_changelog
        assert any("bob" in line for line in final_changelog)
