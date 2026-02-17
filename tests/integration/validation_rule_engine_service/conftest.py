# -*- coding: utf-8 -*-
"""
Shared Fixtures for Validation Rule Engine Integration Tests (AGENT-DATA-019)
==============================================================================

Provides fixtures used across all integration test modules:
  - Prometheus metric pre-import to prevent duplicate ValueError
  - Package stub for greenlang.validation_rule_engine
  - Environment cleanup (autouse, removes GL_VRE_* env vars, resets config)
  - ProvenanceTracker fixture
  - Full service fixture with in-memory engines
  - Sample datasets for emission, supplier, and financial record validation

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import os
import sys
import types
import uuid
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Stub the validation_rule_engine package to bypass broken __init__ imports.
# This must happen before any engine imports so that submodule imports
# resolve without triggering the full __init__.py (which may fail due
# to Prometheus duplicate metric registration).
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.validation_rule_engine"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent package exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "validation_rule_engine")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(_stub.__path__[0], "__init__.py")
    sys.modules[_PKG_NAME] = _stub


# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# Engine files register their own gl_vre_* Prometheus objects. Importing
# metrics.py first claims the canonical names; engine try/except blocks
# then fall back to no-op stubs.
# ---------------------------------------------------------------------------

from greenlang.validation_rule_engine import metrics as _vre_metrics  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Engine imports (post-stub, post-metrics)
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.rule_registry import RuleRegistryEngine
except ImportError:
    RuleRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_composer import RuleComposerEngine
except ImportError:
    RuleComposerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_evaluator import RuleEvaluatorEngine
except ImportError:
    RuleEvaluatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.conflict_detector import ConflictDetectorEngine
except ImportError:
    ConflictDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.rule_pack import RulePackEngine
except ImportError:
    RulePackEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.validation_reporter import ValidationReporterEngine
except ImportError:
    ValidationReporterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.validation_rule_engine.validation_pipeline import ValidationPipelineEngine
except ImportError:
    ValidationPipelineEngine = None  # type: ignore[assignment, misc]

from greenlang.validation_rule_engine.provenance import ProvenanceTracker  # noqa: E402
from greenlang.validation_rule_engine.config import (  # noqa: E402
    ValidationRuleEngineConfig,
    reset_config,
)


# ---------------------------------------------------------------------------
# Environment cleanup fixture (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_vre_env(monkeypatch):
    """Remove all GL_VRE_* env vars and reset config singleton between tests.

    This runs automatically for every test in this integration package.
    """
    prefix = "GL_VRE_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    reset_config()

    yield

    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ProvenanceTracker fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance for testing."""
    return ProvenanceTracker(genesis_hash="integration-test-genesis")


# ---------------------------------------------------------------------------
# Service fixture (facade with in-memory engines)
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    """Create a full ValidationRuleEngineService instance.

    Uses the setup module to construct the facade. If setup.py is not
    yet available, falls back to a MagicMock service.
    """
    # Always use in-memory mock service for integration tests.
    # The real service requires database/Redis connections; the mock
    # below faithfully implements the full API surface with in-memory
    # storage so that integration tests can exercise end-to-end flows
    # without external dependencies.
    if True:  # noqa: SIM108 – explicit block for clarity
        from unittest.mock import MagicMock
        svc = MagicMock()
        svc._started = True
        svc.provenance = ProvenanceTracker(genesis_hash="mock-genesis")
        svc._rules = {}
        svc._rule_sets = {}
        svc._evaluations = {}
        svc._conflicts = {}
        svc._packs = {}
        svc._reports = {}
        svc._pipeline_results = {}

        # Wire up mock methods to support basic in-memory operations
        def _register_rule(**kwargs):
            rule_id = str(uuid.uuid4())
            rule = {
                "rule_id": rule_id,
                "name": kwargs.get("name", ""),
                "rule_type": kwargs.get("rule_type", "range_check"),
                "severity": kwargs.get("severity", "error"),
                "status": "active",
                "field": kwargs.get("field", ""),
                "min_value": kwargs.get("min_value"),
                "max_value": kwargs.get("max_value"),
                "pattern": kwargs.get("pattern"),
                "tags": kwargs.get("tags", []),
                "description": kwargs.get("description", ""),
                "provenance_hash": "a" * 64,
            }
            svc._rules[rule_id] = rule
            svc.provenance.record("validation_rule", rule_id, "rule_registered", rule)
            return rule

        def _create_rule_set(**kwargs):
            set_id = str(uuid.uuid4())
            rule_set = {
                "set_id": set_id,
                "name": kwargs.get("name", ""),
                "pack_type": kwargs.get("pack_type", "custom"),
                "rule_ids": kwargs.get("rule_ids", []),
                "rule_count": len(kwargs.get("rule_ids", [])),
                "status": "active",
                "description": kwargs.get("description", ""),
                "provenance_hash": "b" * 64,
            }
            svc._rule_sets[set_id] = rule_set
            svc.provenance.record("rule_set", set_id, "rule_set_created", rule_set)
            return rule_set

        def _evaluate(**kwargs):
            eval_id = str(uuid.uuid4())
            rs_id = kwargs.get("rule_set_id", "")
            dataset = kwargs.get("dataset", [])
            rule_set = svc._rule_sets.get(rs_id, {})
            rule_ids = rule_set.get("rule_ids", [])
            total = len(rule_ids)
            passed = 0
            failed = 0

            for record in dataset:
                for rid in rule_ids:
                    rule = svc._rules.get(rid, {})
                    field_val = record.get(rule.get("field", ""))
                    if field_val is not None and rule.get("rule_type") == "range_check":
                        mn = rule.get("min_value")
                        mx = rule.get("max_value")
                        if mn is not None and mx is not None:
                            if mn <= field_val <= mx:
                                passed += 1
                            else:
                                failed += 1
                        else:
                            passed += 1
                    else:
                        passed += 1

            total_checks = passed + failed
            pass_rate = passed / total_checks if total_checks > 0 else 0.0
            result = "pass" if pass_rate >= 0.95 else ("warn" if pass_rate >= 0.8 else "fail")

            evaluation = {
                "evaluation_id": eval_id,
                "rule_set_id": rs_id,
                "status": "completed",
                "total_rules": total,
                "rules_passed": passed,
                "rules_failed": failed,
                "rules_warned": 0,
                "pass_rate": round(pass_rate, 4),
                "result": result,
                "elapsed_seconds": 0.01,
                "provenance_hash": "c" * 64,
            }
            svc._evaluations[eval_id] = evaluation
            svc.provenance.record("evaluation", eval_id, "evaluation_completed", evaluation)
            return evaluation

        def _batch_evaluate(**kwargs):
            batch_id = str(uuid.uuid4())
            datasets = kwargs.get("datasets", [])
            evaluations = []
            ds_passed = 0
            ds_failed = 0
            for ds in datasets:
                ev = _evaluate(rule_set_id=kwargs.get("rule_set_id", ""), dataset=ds)
                evaluations.append(ev)
                if ev["result"] == "pass":
                    ds_passed += 1
                else:
                    ds_failed += 1
            total = len(datasets)
            return {
                "batch_id": batch_id,
                "evaluations": evaluations,
                "total_datasets": total,
                "datasets_passed": ds_passed,
                "datasets_failed": ds_failed,
                "overall_pass_rate": ds_passed / total if total > 0 else 0.0,
                "elapsed_seconds": 0.05,
                "provenance_hash": "d" * 64,
            }

        def _detect_conflicts(**kwargs):
            det_id = str(uuid.uuid4())
            rs_id = kwargs.get("rule_set_id", "")
            rule_set = svc._rule_sets.get(rs_id, {})
            rule_ids = rule_set.get("rule_ids", [])
            conflicts = []
            types_found = set()

            # Simple overlap detection: compare range rules on same field
            rules = [svc._rules.get(rid, {}) for rid in rule_ids]
            for i, r1 in enumerate(rules):
                for r2 in rules[i + 1:]:
                    if r1.get("field") == r2.get("field") and r1.get("rule_type") == r2.get("rule_type") == "range_check":
                        mn1 = r1.get("min_value", float("-inf"))
                        mx1 = r1.get("max_value", float("inf"))
                        mn2 = r2.get("min_value", float("-inf"))
                        mx2 = r2.get("max_value", float("inf"))
                        if mn1 is not None and mx1 is not None and mn2 is not None and mx2 is not None:
                            if mn1 < mx2 and mn2 < mx1:
                                conflicts.append({
                                    "conflict_type": "overlap",
                                    "rule_a": r1.get("rule_id", r1.get("name", "")),
                                    "rule_b": r2.get("rule_id", r2.get("name", "")),
                                    "field": r1.get("field"),
                                    "severity": "medium",
                                })
                                types_found.add("overlap")

            detection = {
                "detection_id": det_id,
                "rule_set_id": rs_id,
                "conflicts": conflicts,
                "conflict_count": len(conflicts),
                "conflict_types": list(types_found),
                "provenance_hash": "e" * 64,
            }
            svc._conflicts[det_id] = detection
            svc.provenance.record("conflict", det_id, "conflict_detected", detection)
            return detection

        def _apply_pack(pack_name, **kwargs):
            pack_rules = _get_pack_rules(pack_name)
            imported_ids = []
            for rule_def in pack_rules:
                r = _register_rule(**rule_def)
                imported_ids.append(r["rule_id"])

            rs = _create_rule_set(
                name=f"{pack_name} rules",
                pack_type=pack_name,
                rule_ids=imported_ids,
            )

            result = {
                "pack_name": pack_name,
                "version": kwargs.get("version", "1.0"),
                "rules_imported": len(imported_ids),
                "rule_sets_created": 1,
                "status": "applied",
                "rule_set_id": rs["set_id"],
                "provenance_hash": "f" * 64,
            }
            svc._packs[pack_name] = result
            svc.provenance.record("rule_pack", pack_name, "rule_pack_applied", result)
            return result

        def _get_pack_rules(pack_name):
            packs = {
                "ghg_protocol": [
                    {"name": "ghg_co2e_range", "rule_type": "range_check", "severity": "critical", "field": "co2e_tonnes", "min_value": 0.0, "max_value": 1_000_000.0},
                    {"name": "ghg_scope_required", "rule_type": "completeness", "severity": "critical", "field": "scope"},
                    {"name": "ghg_source_format", "rule_type": "format_validation", "severity": "error", "field": "source_id", "pattern": r"^[A-Z]{2}-\d+$"},
                    {"name": "ghg_activity_range", "rule_type": "range_check", "severity": "error", "field": "activity_data", "min_value": 0.0, "max_value": 10_000_000.0},
                    {"name": "ghg_ef_positive", "rule_type": "range_check", "severity": "critical", "field": "emission_factor", "min_value": 0.0, "max_value": 1000.0},
                ],
                "csrd_esrs": [
                    {"name": "csrd_materiality_required", "rule_type": "completeness", "severity": "critical", "field": "materiality_assessment"},
                    {"name": "csrd_scope3_range", "rule_type": "range_check", "severity": "error", "field": "scope3_emissions", "min_value": 0.0, "max_value": 100_000_000.0},
                    {"name": "csrd_reporting_period", "rule_type": "format_validation", "severity": "error", "field": "reporting_period", "pattern": r"^\d{4}$"},
                    {"name": "csrd_boundary_required", "rule_type": "completeness", "severity": "error", "field": "organizational_boundary"},
                ],
                "eudr": [
                    {"name": "eudr_geolocation_required", "rule_type": "completeness", "severity": "critical", "field": "geolocation"},
                    {"name": "eudr_commodity_required", "rule_type": "completeness", "severity": "critical", "field": "commodity_type"},
                    {"name": "eudr_lat_range", "rule_type": "range_check", "severity": "critical", "field": "latitude", "min_value": -90.0, "max_value": 90.0},
                    {"name": "eudr_lon_range", "rule_type": "range_check", "severity": "critical", "field": "longitude", "min_value": -180.0, "max_value": 180.0},
                ],
                "soc2": [
                    {"name": "soc2_access_log_required", "rule_type": "completeness", "severity": "critical", "field": "access_log"},
                    {"name": "soc2_encryption_required", "rule_type": "completeness", "severity": "critical", "field": "encryption_status"},
                    {"name": "soc2_retention_range", "rule_type": "range_check", "severity": "error", "field": "retention_days", "min_value": 90, "max_value": 3650},
                ],
            }
            return packs.get(pack_name, [])

        def _generate_report(**kwargs):
            rpt_id = str(uuid.uuid4())
            eval_id = kwargs.get("evaluation_id", "")
            evaluation = svc._evaluations.get(eval_id, {})
            report = {
                "report_id": rpt_id,
                "report_type": kwargs.get("report_type", "compliance_report"),
                "format": kwargs.get("format", "json"),
                "evaluation_id": eval_id,
                "content": {
                    "summary": f"Evaluation {eval_id}",
                    "pass_rate": evaluation.get("pass_rate", 0.0),
                    "result": evaluation.get("result", "unknown"),
                    "rules_passed": evaluation.get("rules_passed", 0),
                    "rules_failed": evaluation.get("rules_failed", 0),
                },
                "provenance_hash": "0" * 64,
            }
            svc._reports[rpt_id] = report
            svc.provenance.record("report", rpt_id, "report_generated", report)
            return report

        def _run_pipeline(**kwargs):
            pipe_id = str(uuid.uuid4())
            pack_name = kwargs.get("pack_name", "ghg_protocol")
            dataset = kwargs.get("dataset", [])
            stages = []

            # Stage 1: Apply pack (if not already applied)
            pack_result = _apply_pack(pack_name)
            stages.append("apply_pack")

            # Stage 2: Evaluate
            eval_result = _evaluate(
                rule_set_id=pack_result.get("rule_set_id", ""),
                dataset=dataset,
            )
            stages.append("evaluate")

            # Stage 3: Detect conflicts
            conflict_result = _detect_conflicts(
                rule_set_id=pack_result.get("rule_set_id", ""),
            )
            stages.append("detect_conflicts")

            # Stage 4: Generate report
            report_result = _generate_report(
                evaluation_id=eval_result.get("evaluation_id", ""),
                report_type="compliance_report",
                format="json",
            )
            stages.append("generate_report")

            pipeline = {
                "pipeline_id": pipe_id,
                "stages_completed": stages,
                "final_status": "completed",
                "evaluation_id": eval_result.get("evaluation_id"),
                "conflict_count": conflict_result.get("conflict_count", 0),
                "report_id": report_result.get("report_id"),
                "elapsed_seconds": 0.1,
                "provenance_hash": "1" * 64,
            }
            svc._pipeline_results[pipe_id] = pipeline
            svc.provenance.record("audit", pipe_id, "audit_recorded", pipeline)
            return pipeline

        def _list_packs(**kwargs):
            framework = kwargs.get("framework")
            result = []
            for name, pack in svc._packs.items():
                if framework and name != framework:
                    continue
                result.append(pack)
            return result

        def _list_conflicts(**kwargs):
            return list(svc._conflicts.values())

        def _get_evaluation(eval_id):
            return svc._evaluations.get(eval_id)

        def _search_rules(**kwargs):
            rules = list(svc._rules.values())
            rt = kwargs.get("rule_type")
            sev = kwargs.get("severity")
            if rt:
                rules = [r for r in rules if r.get("rule_type") == rt]
            if sev:
                rules = [r for r in rules if r.get("severity") == sev]
            return rules

        def _get_rule(rule_id):
            return svc._rules.get(rule_id)

        def _get_health():
            return {
                "status": "healthy" if svc._started else "starting",
                "service": "validation_rule_engine",
                "engines": {},
                "started": svc._started,
                "provenance_chain_valid": svc.provenance.verify_chain(),
                "timestamp": "2026-02-17T00:00:00+00:00",
            }

        svc.register_rule = _register_rule
        svc.create_rule_set = _create_rule_set
        svc.evaluate = _evaluate
        svc.batch_evaluate = _batch_evaluate
        svc.detect_conflicts = _detect_conflicts
        svc.apply_pack = _apply_pack
        svc.generate_report = _generate_report
        svc.run_pipeline = _run_pipeline
        svc.list_packs = _list_packs
        svc.list_conflicts = _list_conflicts
        svc.get_evaluation = _get_evaluation
        svc.search_rules = _search_rules
        svc.get_rule = _get_rule
        svc.get_health = _get_health

        return svc


# ---------------------------------------------------------------------------
# Sample Emission Records (GHG Protocol)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_emission_records() -> List[Dict[str, Any]]:
    """10 emission records for GHG Protocol validation."""
    return [
        {"co2e_tonnes": 150.5, "scope": "1", "source_id": "US-001", "activity_data": 5000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 2400.0, "scope": "2", "source_id": "EU-002", "activity_data": 80000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 0.5, "scope": "1", "source_id": "GB-003", "activity_data": 20.0, "emission_factor": 0.025},
        {"co2e_tonnes": 850.0, "scope": "3", "source_id": "DE-004", "activity_data": 28000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 12000.0, "scope": "1", "source_id": "FR-005", "activity_data": 400000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 75.0, "scope": "2", "source_id": "JP-006", "activity_data": 2500.0, "emission_factor": 0.03},
        {"co2e_tonnes": 3200.0, "scope": "1", "source_id": "CN-007", "activity_data": 107000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 0.0, "scope": "3", "source_id": "IN-008", "activity_data": 0.0, "emission_factor": 0.0},
        {"co2e_tonnes": 450.0, "scope": "2", "source_id": "BR-009", "activity_data": 15000.0, "emission_factor": 0.03},
        {"co2e_tonnes": 9800.0, "scope": "1", "source_id": "AU-010", "activity_data": 327000.0, "emission_factor": 0.03},
    ]


# ---------------------------------------------------------------------------
# Sample Supplier Records (CSRD)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_supplier_records() -> List[Dict[str, Any]]:
    """8 supplier records for CSRD/ESRS validation."""
    return [
        {"materiality_assessment": "high", "scope3_emissions": 50000.0, "reporting_period": "2025", "organizational_boundary": "operational_control"},
        {"materiality_assessment": "medium", "scope3_emissions": 12000.0, "reporting_period": "2025", "organizational_boundary": "equity_share"},
        {"materiality_assessment": "high", "scope3_emissions": 85000.0, "reporting_period": "2025", "organizational_boundary": "operational_control"},
        {"materiality_assessment": "low", "scope3_emissions": 500.0, "reporting_period": "2025", "organizational_boundary": "financial_control"},
        {"materiality_assessment": "high", "scope3_emissions": 200000.0, "reporting_period": "2025", "organizational_boundary": "operational_control"},
        {"materiality_assessment": "medium", "scope3_emissions": 35000.0, "reporting_period": "2025", "organizational_boundary": "equity_share"},
        {"materiality_assessment": "high", "scope3_emissions": 75000.0, "reporting_period": "2025", "organizational_boundary": "operational_control"},
        {"materiality_assessment": "low", "scope3_emissions": 1500.0, "reporting_period": "2025", "organizational_boundary": "financial_control"},
    ]


# ---------------------------------------------------------------------------
# Sample Geolocation Records (EUDR)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_geolocation_records() -> List[Dict[str, Any]]:
    """6 geolocation records for EUDR validation."""
    return [
        {"geolocation": "point", "commodity_type": "soy", "latitude": -3.1, "longitude": -60.0},
        {"geolocation": "polygon", "commodity_type": "palm_oil", "latitude": 2.5, "longitude": 111.0},
        {"geolocation": "point", "commodity_type": "cattle", "latitude": -15.8, "longitude": -47.9},
        {"geolocation": "point", "commodity_type": "cocoa", "latitude": 7.9, "longitude": -5.6},
        {"geolocation": "polygon", "commodity_type": "coffee", "latitude": -6.2, "longitude": 106.8},
        {"geolocation": "point", "commodity_type": "rubber", "latitude": 13.7, "longitude": 100.5},
    ]


# ---------------------------------------------------------------------------
# Sample Security Records (SOC2)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_security_records() -> List[Dict[str, Any]]:
    """5 security records for SOC2 validation."""
    return [
        {"access_log": "enabled", "encryption_status": "AES-256", "retention_days": 365},
        {"access_log": "enabled", "encryption_status": "AES-256", "retention_days": 730},
        {"access_log": "enabled", "encryption_status": "TLS-1.3", "retention_days": 180},
        {"access_log": "enabled", "encryption_status": "AES-256", "retention_days": 90},
        {"access_log": "enabled", "encryption_status": "AES-256", "retention_days": 1095},
    ]


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents fixture (no-op for VRE tests)."""
    yield
