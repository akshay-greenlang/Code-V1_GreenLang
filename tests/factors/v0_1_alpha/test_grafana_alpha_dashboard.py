# -*- coding: utf-8 -*-
"""
Static tests for the v0.1 Alpha Grafana dashboard (WS9-T3).

Asserts that ``deployment/observability/grafana/dashboards/factors-v0.1-alpha.json``
matches the WS9-T3 contract:

* schemaVersion 39, version 1, exact alpha tags, refresh "30s",
  time = now-1h .. now
* exactly 8 panels, in the order specified by the WS9-T3 work item
* each panel has the expected title, type, runbook_url annotation, and
  a non-empty PromQL target
* every PromQL parses (using ``promql_parser`` if available, otherwise
  a regex sanity pass)
* the alpha alert rules YAML references the same metric names and
  carries a ``runbook_url`` annotation per alert
* the underlying Prometheus metrics exposed by
  ``greenlang.factors.observability.prometheus_exporter.FactorsMetrics``
  exist and accept the expected labels (so the dashboard is not broken
  silently by a metric rename).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pytest

# Repo root: tests/factors/v0_1_alpha/test_grafana_alpha_dashboard.py
# parents[3] = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHBOARD_PATH = (
    _REPO_ROOT
    / "deployment"
    / "observability"
    / "grafana"
    / "dashboards"
    / "factors-v0.1-alpha.json"
)
_ALERTS_PATH = (
    _REPO_ROOT
    / "deployment"
    / "observability"
    / "prometheus"
    / "factors-v0.1-alpha-alerts.yaml"
)
_RUNBOOK_PATH = (
    _REPO_ROOT
    / "docs"
    / "factors"
    / "runbooks"
    / "factors-v0.1-alpha-alerts.md"
)


# ---------------------------------------------------------------------------
# Expected panel contract  (panel id -> title, type, query substring)
# ---------------------------------------------------------------------------

_EXPECTED_PANELS = [
    {
        "id": 1,
        "title": "Alpha API request rate",
        "type": "timeseries",
        "query_must_contain": [
            "rate(http_requests_total{",
            'release_profile="alpha-v0.1"',
            "by (path)",
        ],
    },
    {
        "id": 2,
        "title": "Alpha API p50 / p95 / p99 latency (per endpoint)",
        "type": "timeseries",
        "query_must_contain": [
            "histogram_quantile(0.95",
            "http_request_duration_seconds_bucket",
            'release_profile="alpha-v0.1"',
            "by (le, path)",
        ],
    },
    {
        "id": 3,
        "title": "Alpha API error rate (4xx / 5xx)",
        "type": "timeseries",
        "query_must_contain": [
            "http_requests_total",
            'release_profile="alpha-v0.1"',
            'status=~"4..|5.."',
            "by (status)",
        ],
    },
    {
        "id": 4,
        "title": "Edition served (current edition_id)",
        "type": "stat",
        "query_must_contain": [
            "factors_current_edition_id_info",
            "by (edition)",
        ],
    },
    {
        "id": 5,
        "title": "Schema validation failure rate",
        "type": "timeseries",
        "query_must_contain": [
            "factors_schema_validation_failures_total",
            'schema="factor_record_v0_1"',
            "by (source)",
        ],
    },
    {
        "id": 6,
        "title": "Provenance gate rejection rate",
        "type": "timeseries",
        "query_must_contain": [
            "factors_alpha_provenance_gate_rejections_total",
            "by (source, reason)",
        ],
    },
    {
        "id": 7,
        "title": "Ingestion success rate per source",
        "type": "stat",
        "query_must_contain": [
            "factors_ingestion_runs_total",
            'status="success"',
            "ipcc-ar6",
            "defra-2025",
            "epa-ghg-hub",
            "epa-egrid",
            "india-cea-baseline",
            "eu-cbam-defaults",
            "by (source)",
        ],
    },
    {
        "id": 8,
        "title": "Parser error rate",
        "type": "timeseries",
        "query_must_contain": [
            "factors_parser_errors_total",
            "by (source, error_type)",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_panel_queries(panel: dict) -> List[str]:
    return [t.get("expr", "") for t in panel.get("targets", []) if t.get("expr")]


def _promql_sanity_regex(expr: str) -> None:
    """Catch obviously broken PromQL (unbalanced braces / parens / aggregations)."""
    if not isinstance(expr, str) or not expr.strip():
        raise AssertionError(f"empty PromQL: {expr!r}")
    if expr.count("{") != expr.count("}"):
        raise AssertionError(f"unbalanced curly braces in PromQL: {expr!r}")
    if expr.count("(") != expr.count(")"):
        raise AssertionError(f"unbalanced parens in PromQL: {expr!r}")
    if expr.count('"') % 2 != 0:
        raise AssertionError(f"unbalanced double-quotes in PromQL: {expr!r}")
    # Aggregation operator must be followed by '(' or 'by/without' clause.
    bad = re.search(r"\b(sum|avg|max|min|count|topk)\s*$", expr)
    if bad:
        raise AssertionError(f"aggregator not applied: {expr!r}")


def _promql_parse(expr: str) -> None:
    """Try a real PromQL parser; fall back to regex sanity if unavailable."""
    try:
        import promql_parser  # type: ignore  # noqa: PLC0415
    except Exception:
        _promql_sanity_regex(expr)
        return
    try:
        promql_parser.parse(expr)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"promql_parser failed to parse expr: {expr!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Top-level fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard() -> dict:
    assert _DASHBOARD_PATH.is_file(), f"dashboard JSON missing: {_DASHBOARD_PATH}"
    text = _DASHBOARD_PATH.read_text(encoding="utf-8")
    return json.loads(text)


# ---------------------------------------------------------------------------
# Top-level metadata tests
# ---------------------------------------------------------------------------


def test_dashboard_json_is_valid(dashboard: dict) -> None:
    """The dashboard JSON parses (sanity, in case the file got corrupted)."""
    assert isinstance(dashboard, dict)
    assert dashboard.get("title")
    assert dashboard.get("uid")


def test_dashboard_schema_version_is_39(dashboard: dict) -> None:
    assert dashboard["schemaVersion"] == 39


def test_dashboard_version_is_1(dashboard: dict) -> None:
    assert dashboard["version"] == 1


def test_dashboard_tags_match_alpha_contract(dashboard: dict) -> None:
    expected = {"greenlang", "factors", "alpha", "v0.1"}
    actual = set(dashboard.get("tags", []))
    assert expected == actual, (
        f"alpha dashboard tags must be exactly {expected}; got {actual}"
    )


def test_dashboard_refresh_is_30s(dashboard: dict) -> None:
    assert dashboard["refresh"] == "30s"


def test_dashboard_time_window_is_one_hour(dashboard: dict) -> None:
    assert dashboard["time"] == {"from": "now-1h", "to": "now"}


def test_dashboard_uid_is_alpha_specific(dashboard: dict) -> None:
    assert dashboard["uid"] == "greenlang-factors-v0-1-alpha"


# ---------------------------------------------------------------------------
# Panel-count + per-panel tests
# ---------------------------------------------------------------------------


def test_dashboard_has_exactly_eight_panels(dashboard: dict) -> None:
    panels = dashboard.get("panels", [])
    assert len(panels) == 8, (
        f"WS9-T3 alpha dashboard must have exactly 8 panels; got {len(panels)}"
    )


@pytest.mark.parametrize("expected", _EXPECTED_PANELS, ids=lambda x: f"panel-{x['id']}")
def test_panel_matches_contract(dashboard: dict, expected: dict) -> None:
    panels_by_id = {p["id"]: p for p in dashboard["panels"]}
    assert expected["id"] in panels_by_id, (
        f"panel id {expected['id']} not found"
    )
    panel = panels_by_id[expected["id"]]

    # Title
    assert panel["title"] == expected["title"], (
        f"panel {expected['id']} title mismatch: "
        f"want {expected['title']!r}, got {panel['title']!r}"
    )
    # Type
    assert panel["type"] == expected["type"], (
        f"panel {expected['id']} type mismatch: "
        f"want {expected['type']!r}, got {panel['type']!r}"
    )
    # Runbook URL annotation (top-level OR description-embedded)
    runbook_url_top = panel.get("runbook_url", "")
    desc = panel.get("description", "")
    expected_runbook_anchor = (
        f"docs/factors/runbooks/factors-v0.1-alpha-alerts.md#panel-{expected['id']}"
    )
    has_runbook = (
        expected_runbook_anchor in runbook_url_top
        or expected_runbook_anchor in desc
    )
    assert has_runbook, (
        f"panel {expected['id']} missing runbook_url anchor "
        f"{expected_runbook_anchor!r}"
    )

    # PromQL substrings
    queries = _all_panel_queries(panel)
    assert queries, f"panel {expected['id']} has no targets/expr"
    combined = " ".join(queries)
    for needle in expected["query_must_contain"]:
        assert needle in combined, (
            f"panel {expected['id']} PromQL missing fragment {needle!r}; "
            f"queries were: {queries}"
        )


# ---------------------------------------------------------------------------
# PromQL parse tests (one assertion per panel, all panels)
# ---------------------------------------------------------------------------


def test_every_panel_promql_parses(dashboard: dict) -> None:
    for panel in dashboard["panels"]:
        for expr in _all_panel_queries(panel):
            _promql_parse(expr)


# ---------------------------------------------------------------------------
# Alert rules cross-check (Deliverable 4 invariants)
# ---------------------------------------------------------------------------


def test_alert_rules_file_exists() -> None:
    assert _ALERTS_PATH.is_file(), f"alert rules YAML missing: {_ALERTS_PATH}"


def test_alert_rules_reference_dashboard_metrics() -> None:
    text = _ALERTS_PATH.read_text(encoding="utf-8")
    # Every metric the dashboard depends on must appear in at least one alert.
    expected_metrics = [
        "http_request_duration_seconds_bucket",
        "http_requests_total",
        "factors_schema_validation_failures_total",
        "factors_alpha_provenance_gate_rejections_total",
        "factors_ingestion_runs_total",
        "factors_parser_errors_total",
    ]
    for m in expected_metrics:
        assert m in text, f"alert rules missing metric reference {m!r}"


def test_alert_rules_have_runbook_urls() -> None:
    text = _ALERTS_PATH.read_text(encoding="utf-8")
    runbook_count = text.count("runbook_url:")
    # We ship at least 6 alerts per the WS9-T3 spec.
    alert_count = text.count("- alert:")
    assert alert_count >= 6, f"expected >= 6 alerts; got {alert_count}"
    assert runbook_count >= alert_count, (
        f"every alert must carry runbook_url; got {runbook_count} for {alert_count} alerts"
    )


# ---------------------------------------------------------------------------
# Runbook cross-check
# ---------------------------------------------------------------------------


def test_runbook_has_anchor_per_panel() -> None:
    assert _RUNBOOK_PATH.is_file(), f"runbook missing: {_RUNBOOK_PATH}"
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    for i in range(1, 9):
        anchor = f'<a id="panel-{i}"></a>'
        assert anchor in text, f"runbook missing anchor for panel {i}: {anchor}"


# ---------------------------------------------------------------------------
# Metric existence on FactorsMetrics  (Deliverable 2 invariants)
# ---------------------------------------------------------------------------


def test_factors_metrics_exposes_alpha_counters() -> None:
    """The dashboard would silently break if the metrics were renamed.

    We exercise each new alpha emit method on the FactorsMetrics
    singleton; if any one of them is missing we fail fast. Using the
    singleton avoids ``Duplicated timeseries in CollectorRegistry``
    when the global default Prometheus registry has already seen the
    factors metrics from another test in the suite.
    """
    from greenlang.factors.observability.prometheus_exporter import (
        get_factors_metrics,
    )

    fm = get_factors_metrics()
    # Method existence
    for name in (
        "record_schema_validation_failure",
        "record_alpha_provenance_rejection",
        "record_ingestion_run",
        "record_parser_error",
        "set_current_edition",
    ):
        assert hasattr(fm, name), f"FactorsMetrics missing method {name!r}"

    # Smoke-call each one (must not raise)
    fm.record_schema_validation_failure(schema="factor_record_v0_1", source="test")
    fm.record_alpha_provenance_rejection(source="test", reason="schema_violation")
    fm.record_ingestion_run(status="success", source="ipcc-ar6")
    fm.record_parser_error(source="defra-2025", error_type="JSONDecodeError")
    fm.set_current_edition(edition="alpha-v0.1.0", active=True)
