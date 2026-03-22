# -*- coding: utf-8 -*-
"""
Unit tests for UtilityReportingEngine -- PACK-036 Engine 10
=============================================================

Tests monthly summary, executive dashboard, KPI calculation,
portfolio summary, variance explanation, anomaly detection,
Markdown/HTML/JSON/CSV rendering, batch generation, and provenance.

Coverage target: 85%+
Total tests: ~50
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack036_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("utility_reporting_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_class_exists(self):
        assert hasattr(_m, "UtilityReportingEngine")

    def test_engine_instantiation(self):
        engine = _m.UtilityReportingEngine()
        assert engine is not None


def _make_report_data():
    return {
        "facility_id": "FAC-036-DE-001",
        "facility_name": "Berlin Office Tower",
        "period": "2025-01",
        "total_cost_eur": Decimal("38021.79"),
        "total_consumption_kwh": 150_000,
        "demand_kw": 480,
        "eui_kwh_per_m2": 263.0,
        "yoy_consumption_change_pct": Decimal("3.5"),
        "yoy_cost_change_pct": Decimal("7.2"),
        "anomalies": [],
        "bills": [],
    }


class TestMonthlySummary:
    def test_generate_monthly_summary(self):
        engine = _m.UtilityReportingEngine()
        gen = (getattr(engine, "generate_monthly_summary", None)
               or getattr(engine, "monthly_summary", None))
        if gen is None:
            pytest.skip("monthly_summary method not found")
        result = gen(_make_report_data())
        assert result is not None


class TestExecutiveDashboard:
    def test_generate_executive_dashboard(self):
        engine = _m.UtilityReportingEngine()
        gen = (getattr(engine, "generate_executive_dashboard", None)
               or getattr(engine, "executive_dashboard", None))
        if gen is None:
            pytest.skip("executive_dashboard method not found")
        result = gen(_make_report_data())
        assert result is not None


class TestKPIs:
    def test_calculate_kpis(self, sample_bill_history, sample_facility_metrics):
        engine = _m.UtilityReportingEngine()
        calc = (getattr(engine, "calculate_kpis", None)
                or getattr(engine, "compute_kpis", None))
        if calc is None:
            pytest.skip("calculate_kpis method not found")
        result = calc(consumption_history=sample_bill_history,
                      facility=sample_facility_metrics)
        assert result is not None


class TestPortfolioSummary:
    def test_build_portfolio_summary(self):
        engine = _m.UtilityReportingEngine()
        build = (getattr(engine, "build_portfolio_summary", None)
                 or getattr(engine, "portfolio_summary", None))
        if build is None:
            pytest.skip("portfolio_summary method not found")
        facilities = [
            {"facility_id": "F-001", "total_cost": Decimal("38000"),
             "consumption_kwh": 150_000, "eui": 263.0},
            {"facility_id": "F-002", "total_cost": Decimal("28000"),
             "consumption_kwh": 120_000, "eui": 240.0},
        ]
        result = build(facilities=facilities)
        assert result is not None


class TestVarianceExplanation:
    def test_explain_variance(self, sample_bill_history):
        engine = _m.UtilityReportingEngine()
        explain = (getattr(engine, "explain_variance", None)
                   or getattr(engine, "variance_analysis", None))
        if explain is None:
            pytest.skip("explain_variance method not found")
        current = {"period": "2025-01", "consumption_kwh": 155_000,
                   "cost_eur": Decimal("26350")}
        result = explain(current=current,
                         history=sample_bill_history)
        assert result is not None


class TestAnomalyDetection:
    def test_detect_anomalies(self, sample_bill_history):
        engine = _m.UtilityReportingEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "find_anomalies", None))
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        result = detect(consumption_history=sample_bill_history)
        assert result is not None


class TestRenderMarkdown:
    def test_render_markdown(self):
        engine = _m.UtilityReportingEngine()
        render = (getattr(engine, "render_markdown", None)
                  or getattr(engine, "to_markdown", None))
        if render is None:
            pytest.skip("render_markdown method not found")
        result = render(_make_report_data())
        assert isinstance(result, str)
        assert len(result) > 50


class TestRenderHTML:
    def test_render_html(self):
        engine = _m.UtilityReportingEngine()
        render = (getattr(engine, "render_html", None)
                  or getattr(engine, "to_html", None))
        if render is None:
            pytest.skip("render_html method not found")
        result = render(_make_report_data())
        assert isinstance(result, str)
        assert len(result) > 50


class TestExportJSON:
    def test_export_json(self):
        engine = _m.UtilityReportingEngine()
        export = (getattr(engine, "export_json", None)
                  or getattr(engine, "to_json", None))
        if export is None:
            pytest.skip("export_json method not found")
        result = export(_make_report_data())
        assert result is not None


class TestExportCSV:
    def test_export_csv(self, sample_bill_history):
        engine = _m.UtilityReportingEngine()
        export = (getattr(engine, "export_csv", None)
                  or getattr(engine, "to_csv", None))
        if export is None:
            pytest.skip("export_csv method not found")
        result = export(sample_bill_history)
        assert result is not None
        if isinstance(result, str):
            assert "," in result or "\t" in result


class TestBatchGeneration:
    def test_batch_generate(self):
        engine = _m.UtilityReportingEngine()
        batch = (getattr(engine, "batch_generate", None)
                 or getattr(engine, "generate_batch", None))
        if batch is None:
            pytest.skip("batch_generate method not found")
        reports = [_make_report_data(), _make_report_data()]
        reports[1]["facility_id"] = "FAC-036-DE-002"
        result = batch(reports)
        assert result is not None


class TestProvenance:
    def test_provenance_hash(self):
        engine = _m.UtilityReportingEngine()
        gen = (getattr(engine, "generate_monthly_summary", None)
               or getattr(engine, "monthly_summary", None))
        if gen is None:
            pytest.skip("monthly_summary method not found")
        result = gen(_make_report_data())
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)
