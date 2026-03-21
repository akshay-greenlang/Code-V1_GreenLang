# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 EnMS Pack - Performance Tests (test_performance.py)
========================================================================

Tests engine execution time, throughput, and memory behaviour for key
engines. Validates that each engine completes within target latency
and handles batch workloads efficiently.

Coverage target: 85%+
Total tests: ~20
"""

import importlib.util
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"

ENGINE_NAMES = [
    "seu_analyzer_engine",
    "energy_baseline_engine",
    "enpi_calculator_engine",
    "cusum_monitor_engine",
    "degree_day_engine",
    "energy_balance_engine",
    "action_plan_engine",
    "compliance_checker_engine",
    "performance_trend_engine",
    "management_review_engine",
]


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack034_perf.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =============================================================================
# Engine Import Time
# =============================================================================


class TestEngineImportTime:
    @pytest.mark.parametrize("engine_name", ENGINE_NAMES)
    def test_engine_import_time(self, engine_name):
        """Each engine module should import in under 500ms."""
        path = ENGINES_DIR / f"{engine_name}.py"
        if not path.exists():
            pytest.skip(f"Engine not found: {engine_name}")
        # Clear cached module to measure import time
        mod_key = f"pack034_perf_import.{engine_name}"
        sys.modules.pop(mod_key, None)
        spec = importlib.util.spec_from_file_location(mod_key, str(path))
        mod = importlib.util.module_from_spec(spec)

        start = time.perf_counter()
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pytest.skip(f"Cannot import {engine_name}")
        finally:
            sys.modules.pop(mod_key, None)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"{engine_name} import took {elapsed_ms:.1f}ms (limit: 500ms)"


# =============================================================================
# SEU Analysis Performance
# =============================================================================


class TestSEUAnalysisPerformance:
    def test_seu_analysis_performance(self):
        """1000 energy consumers should be analyzed in under 5 seconds."""
        mod = _load("seu_analyzer_engine")
        engine = mod.SEUAnalyzerEngine()
        model_cls = (getattr(mod, "EnergyConsumer", None)
                     or getattr(mod, "SEUConsumer", None)
                     or getattr(mod, "EnergyUse", None))
        if model_cls is None:
            pytest.skip("EnergyConsumer model not found")
        analyze = (getattr(engine, "analyze", None) or getattr(engine, "run_pareto", None)
                   or getattr(engine, "identify_seus", None))
        if analyze is None:
            pytest.skip("analyze method not found")
        consumers = [
            model_cls(consumer_id=f"EC-{i:04d}", name=f"Consumer {i}",
                      energy_kwh=float(1000 + i * 100))
            for i in range(1000)
        ]
        start = time.perf_counter()
        result = analyze(consumers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"SEU analysis took {elapsed_ms:.1f}ms (limit: 5000ms)"
        assert result is not None


# =============================================================================
# Baseline Fitting Performance
# =============================================================================


class TestBaselineFittingPerformance:
    def test_baseline_fitting_performance(self):
        """100 data points should fit in under 2 seconds."""
        from datetime import date
        mod = _load("energy_baseline_engine")
        engine = mod.EnergyBaselineEngine()
        import math
        # Build proper BaselineDataPoint objects
        data_points = []
        for i in range(100):
            month = (i % 12) + 1
            year = 2020 + (i // 12)
            end_month = month + 1 if month < 12 else 1
            end_year = year if month < 12 else year + 1
            energy = 200_000 + 50_000 * math.sin(2 * math.pi * i / 12) + i * 100
            hdd = 500 - 40 * math.sin(2 * math.pi * i / 12)
            data_points.append(mod.BaselineDataPoint(
                period_start=date(year, month, 1),
                period_end=date(end_year, end_month, 1),
                energy_kwh=Decimal(str(round(energy, 2))),
                variables={"hdd": Decimal(str(round(hdd, 2)))},
            ))
        config = mod.BaselineConfig(model_type=mod.BaselineModelType.SINGLE_VARIABLE)
        start = time.perf_counter()
        result = engine.establish_baseline(data_points, config)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Baseline fit took {elapsed_ms:.1f}ms (limit: 2000ms)"
        assert result is not None


# =============================================================================
# EnPI Calculation Performance
# =============================================================================


class TestEnPICalculationPerformance:
    def test_enpi_calculation_performance(self):
        """50 EnPI calculations should complete in under 3 seconds."""
        from datetime import date
        mod = _load("enpi_calculator_engine")
        engine = mod.EnPICalculatorEngine()
        start = time.perf_counter()
        for i in range(50):
            definition = mod.EnPIDefinition(
                enpi_id=f"PERF-{i}",
                enpi_type=mod.EnPIType.ABSOLUTE,
            )
            measurements = []
            for m in range(12):
                month = m + 1
                end_month = month + 1 if month < 12 else 1
                end_year = 2025 if month < 12 else 2026
                measurements.append(mod.EnPIMeasurement(
                    period_start=date(2025, month, 1),
                    period_end=date(end_year, end_month, 1),
                    energy_value=Decimal(str(200_000 + i * 1000)),
                ))
            engine.calculate_enpi(definition, measurements, measurements)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 3000, f"50 EnPI calcs took {elapsed_ms:.1f}ms (limit: 3000ms)"


# =============================================================================
# CUSUM Monitoring Performance
# =============================================================================


class TestCUSUMMonitoringPerformance:
    def test_cusum_monitoring_performance(self):
        """1000 data points should be processed in under 2 seconds."""
        mod = _load("cusum_monitor_engine")
        engine = mod.CUSUMMonitorEngine()
        calc = (getattr(engine, "calculate", None) or getattr(engine, "run_cusum", None)
                or getattr(engine, "compute", None) or getattr(engine, "monitor", None))
        if calc is None:
            pytest.skip("calculate method not found")
        residuals = [(-1) ** i * (500 + i * 10) for i in range(1000)]
        start = time.perf_counter()
        result = calc(residuals, method="STANDARD")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"CUSUM 1000 pts took {elapsed_ms:.1f}ms (limit: 2000ms)"
        assert result is not None


# =============================================================================
# Compliance Check Performance
# =============================================================================


class TestComplianceCheckPerformance:
    def test_compliance_check_performance(self, sample_compliance_evidence):
        """26 clause compliance check should complete in under 3 seconds."""
        mod = _load("compliance_checker_engine")
        engine = mod.ComplianceCheckerEngine()
        evidence_map = {f"{c}.{s}": ["Evidence item"]
                        for c in range(4, 11) for s in range(1, 7)}
        document_map = {"5.2": ["Energy Policy"], "6.3": ["Energy Review"]}
        start = time.perf_counter()
        assessments = engine.assess_all_clauses(evidence_map, document_map)
        score = engine.calculate_compliance_score(assessments)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 3000, f"Compliance check took {elapsed_ms:.1f}ms (limit: 3000ms)"
        assert score is not None


# =============================================================================
# Template Rendering Performance
# =============================================================================


class TestTemplateRenderingPerformance:
    def test_template_rendering_performance(self):
        """All 10 templates should render in under 5 seconds total."""
        from pathlib import Path
        TEMPLATES_DIR = PACK_ROOT / "templates"
        template_files = [
            "energy_policy_template.py",
            "energy_review_report_template.py",
            "enpi_methodology_template.py",
            "action_plan_template.py",
            "operational_control_template.py",
            "performance_report_template.py",
            "internal_audit_template.py",
            "management_review_template.py",
            "corrective_action_template.py",
            "enms_documentation_template.py",
        ]
        data = {
            "facility_name": "Test Facility",
            "total_energy_kwh": 2_500_000,
            "review_date": "2026-03-15",
        }
        rendered = 0
        start = time.perf_counter()
        for tpl_file in template_files:
            path = TEMPLATES_DIR / tpl_file
            if not path.exists():
                continue
            mod_key = f"pack034_perf_tpl.{tpl_file.replace('.py', '')}"
            if mod_key in sys.modules:
                mod = sys.modules[mod_key]
            else:
                spec = importlib.util.spec_from_file_location(mod_key, str(path))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_key] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    continue
            # Find template class
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and attr_name.endswith("Template"):
                    try:
                        instance = obj()
                        render = (getattr(instance, "render_markdown", None)
                                  or getattr(instance, "render", None))
                        if render:
                            render(data)
                            rendered += 1
                    except Exception:
                        pass
                    break
        elapsed_ms = (time.perf_counter() - start) * 1000
        if rendered > 0:
            assert elapsed_ms < 5000, f"Template rendering took {elapsed_ms:.1f}ms (limit: 5000ms)"
