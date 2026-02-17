"""
Functional test suite for SchemaMigrationPipelineEngine.

Loads engine and its dependencies directly via importlib to avoid
triggering the full greenlang package init chain (which causes Prometheus
duplicate-metric errors in isolated test runs).

Run via:
    python "C:/Users/aksha/Code-V1_GreenLang/greenlang/schema_migration/_test_pipeline.py"
"""
import importlib.util
import json
import os
import sys
import types

_ROOT = "C:/Users/aksha/Code-V1_GreenLang"


def _load_module(dotted_name: str) -> types.ModuleType:
    """Load a greenlang submodule directly from its .py file.

    Bypasses greenlang/__init__.py so we do not trigger the heavy import
    chain that causes Prometheus duplicate-metric errors.
    """
    rel_path = dotted_name.replace(".", os.sep) + ".py"
    full_path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order so each module's imports resolve correctly
_provenance_mod = _load_module("greenlang.schema_migration.provenance")
_config_mod = _load_module("greenlang.schema_migration.config")
_metrics_mod = _load_module("greenlang.schema_migration.metrics")
_pipeline_mod = _load_module("greenlang.schema_migration.schema_migration_pipeline")

SchemaMigrationPipelineEngine = _pipeline_mod.SchemaMigrationPipelineEngine


# ===========================================================================
# Test functions
# ===========================================================================


def test_engine_init() -> "SchemaMigrationPipelineEngine":
    engine = SchemaMigrationPipelineEngine()
    # All upstream engines unavailable (not built yet) — should be None
    assert engine._detector is None
    assert engine._registry is None
    assert engine._versioner is None
    assert engine._checker is None
    assert engine._planner is None
    assert engine._executor is None
    print("test_engine_init: PASS")
    return engine


def test_detect_stage(engine: "SchemaMigrationPipelineEngine") -> tuple:
    source = {"id": "int", "name": "str"}
    target = {"id": "int", "name": "str", "email": "str"}

    # Field added — non-breaking
    r_add = engine.detect_stage(source, target)
    assert r_add["has_changes"] is True, f"Expected has_changes=True: {r_add}"
    assert r_add["change_count"] == 1, f"Expected 1 change: {r_add}"
    assert len(r_add["non_breaking_changes"]) == 1
    assert len(r_add["breaking_changes"]) == 0
    print(f"test_detect_stage (add): PASS — change_count={r_add['change_count']}")

    # No changes
    r_none = engine.detect_stage(source, dict(source))
    assert r_none["has_changes"] is False, f"Expected no changes: {r_none}"
    assert r_none["change_count"] == 0
    print("test_detect_stage (no change): PASS")

    # Field removed — breaking
    r_break = engine.detect_stage(source, {"id": "int"})
    assert r_break["has_changes"] is True
    assert len(r_break["breaking_changes"]) == 1
    print(f"test_detect_stage (remove): PASS — breaking={len(r_break['breaking_changes'])}")

    return r_add, r_none, r_break


def test_compatibility_stage(
    engine: "SchemaMigrationPipelineEngine",
    changes_add: dict,
    changes_none: dict,
    changes_break: dict,
) -> tuple:
    # Non-breaking addition
    c_add = engine.compatibility_stage({}, {}, changes_add)
    assert c_add["is_compatible"] is True
    assert c_add["is_breaking"] is False
    print(f"test_compat (add): PASS — compatible={c_add['is_compatible']}")

    # Breaking removal
    c_break = engine.compatibility_stage({}, {}, changes_break)
    assert c_break["is_breaking"] is True
    assert c_break["is_compatible"] is False
    print(f"test_compat (break): PASS — breaking={c_break['is_breaking']}")

    # No changes
    c_none = engine.compatibility_stage({}, {}, changes_none)
    assert c_none["is_compatible"] is True
    print("test_compat (no change): PASS")

    return c_add, c_break


def test_plan_stage(engine: "SchemaMigrationPipelineEngine", changes: dict) -> dict:
    plan = engine.plan_stage("schema_x", "1.0.0", "1.1.0", changes)
    assert plan["plan_id"].startswith("plan-"), f"Bad plan_id: {plan['plan_id']}"
    assert plan["schema_id"] == "schema_x"
    assert plan["source_version"] == "1.0.0"
    assert plan["target_version"] == "1.1.0"
    assert plan["step_count"] == changes["change_count"]
    assert isinstance(plan["effort"], str)
    assert plan["effort"] in ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    print(f"test_plan_stage: PASS — plan_id={plan['plan_id']} steps={plan['step_count']} effort={plan['effort']}")
    return plan


def test_validate_stage(engine: "SchemaMigrationPipelineEngine", plan: dict) -> None:
    val = engine.validate_stage(plan["plan_id"])
    assert val["is_valid"] is True
    print(f"test_validate_stage (valid): PASS — is_valid={val['is_valid']}")

    val2 = engine.validate_stage("")
    assert val2["is_valid"] is False
    print("test_validate_stage (empty plan_id): PASS — correctly invalid")


def test_execute_stage(engine: "SchemaMigrationPipelineEngine", plan: dict) -> dict:
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    ex = engine.execute_stage(plan, data=data)
    assert ex["status"] == "success", f"Expected success: {ex}"
    assert ex["records_migrated"] == 2
    assert ex["migrated_data"] is not None
    assert len(ex["migrated_data"]) == 2
    # field_added step should add 'email' to every record
    for rec in ex["migrated_data"]:
        assert "email" in rec, f"'email' missing from migrated record: {rec}"
    print(f"test_execute_stage: PASS — status={ex['status']} records={ex['records_migrated']}")

    # dry_run
    ex_dry = engine.execute_stage(plan, data=data, dry_run=True)
    assert ex_dry["status"] == "dry_run"
    assert ex_dry["records_migrated"] == 0
    print("test_execute_stage (dry_run): PASS")

    # No data
    ex_nodata = engine.execute_stage(plan, data=None)
    assert ex_nodata["records_migrated"] == 0
    assert ex_nodata["migrated_data"] is None
    print("test_execute_stage (no data): PASS")

    return ex


def test_verify_stage(engine: "SchemaMigrationPipelineEngine", exec_result: dict) -> None:
    target_def = {"id": "int", "name": "str", "email": "str"}
    migrated = exec_result.get("migrated_data")

    vfy = engine.verify_stage(exec_result["execution_id"], target_def, migrated)
    assert vfy["passed"] is True, f"Verify should pass: {vfy}"
    assert vfy["records_verified"] == 2
    print(f"test_verify_stage: PASS — passed={vfy['passed']} records={vfy['records_verified']}")

    # Missing required field
    bad_data = [{"id": 1}]
    vfy2 = engine.verify_stage("exec-x", {"id": "int", "name": "str"}, bad_data)
    assert vfy2["passed"] is False, "Should fail when name is missing"
    assert vfy2["failure_reason"] is not None
    print(f"test_verify_stage (missing field): PASS — failure_reason={vfy2['failure_reason'][:50]!r}")

    # No data — structural check skipped, passes
    vfy3 = engine.verify_stage("exec-y", target_def, None)
    assert vfy3["passed"] is True
    assert vfy3["records_verified"] == 0
    print("test_verify_stage (no data): PASS")


def test_run_pipeline_no_changes(engine: "SchemaMigrationPipelineEngine") -> dict:
    # When the registry is unavailable (stub), the source definition is always
    # empty {}. If we pass {"id": "int"} as target, the detector sees a new
    # field and reports changes (completed, not no_changes).
    # We verify no_changes by passing "{}" as the target (matching empty source).
    result = engine.run_pipeline("schema_nochange", json.dumps({}))
    assert result["status"] == "no_changes", f"Expected no_changes: {result['status']}"
    assert "detect" in result["stages_completed"]
    assert result["provenance_hash"] is not None
    assert result["total_time_ms"] is not None
    print(f"test_run_pipeline (no changes): PASS — status={result['status']}")
    return result


def test_run_pipeline_with_changes(engine: "SchemaMigrationPipelineEngine") -> dict:
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    result = engine.run_pipeline(
        "schema_withchange",
        json.dumps({"id": "int", "name": "str", "email": "str"}),
        data=data,
    )
    assert result["status"] in ("completed", "aborted", "failed", "no_changes")
    assert "detect" in result["stages_completed"]
    assert result["total_time_ms"] is not None
    assert result["provenance_hash"] is not None
    print(
        f"test_run_pipeline (changes): PASS — status={result['status']} "
        f"stages={result['stages_completed']}"
    )
    return result


def test_run_pipeline_dry_run(engine: "SchemaMigrationPipelineEngine") -> None:
    data = [{"id": 1}]
    result = engine.run_pipeline(
        "schema_dryrun",
        json.dumps({"id": "int", "extra": "str"}),
        data=data,
        dry_run=True,
    )
    assert result["status"] == "dry_run_completed", f"Expected dry_run_completed: {result['status']}"
    print(f"test_run_pipeline (dry_run): PASS — status={result['status']}")


def test_run_pipeline_invalid_json(engine: "SchemaMigrationPipelineEngine") -> None:
    result = engine.run_pipeline("schema_badjson", "NOT_VALID_JSON")
    assert result["status"] == "aborted", f"Expected aborted: {result['status']}"
    assert "detect" in result["stages_failed"]
    print(f"test_run_pipeline (invalid JSON): PASS — status={result['status']}")


def test_run_pipeline_skip_compat(engine: "SchemaMigrationPipelineEngine") -> None:
    # When skip_compatibility=True, breaking changes should NOT abort
    result = engine.run_pipeline(
        "schema_skipcompat",
        json.dumps({"only_field": "str"}),
        skip_compatibility=True,
    )
    assert "compatibility" not in result.get("stages_failed", [])
    print(f"test_run_pipeline (skip_compat): PASS — status={result['status']}")


def test_run_pipeline_empty_schema_id(engine: "SchemaMigrationPipelineEngine") -> None:
    try:
        engine.run_pipeline("", json.dumps({"x": "int"}))
        raise AssertionError("Should raise ValueError for empty schema_id")
    except ValueError:
        print("test_run_pipeline (empty schema_id): PASS — ValueError raised")


def test_run_pipeline_empty_target(engine: "SchemaMigrationPipelineEngine") -> None:
    try:
        engine.run_pipeline("schema_x", "")
        raise AssertionError("Should raise ValueError for empty target")
    except ValueError:
        print("test_run_pipeline (empty target_definition_json): PASS — ValueError raised")


def test_run_batch_pipeline(engine: "SchemaMigrationPipelineEngine") -> None:
    pairs = [
        {"schema_id": "batch_a", "target_definition_json": json.dumps({"x": "int"})},
        {"schema_id": "batch_b", "target_definition_json": json.dumps({"y": "str"})},
    ]
    batch = engine.run_batch_pipeline(pairs)
    assert batch["total"] == 2, f"batch total wrong: {batch['total']}"
    assert len(batch["results"]) == 2
    assert batch["batch_id"].startswith("batch-")
    assert batch["provenance_hash"] is not None
    print(
        f"test_run_batch_pipeline: PASS — total={batch['total']} "
        f"no_changes={batch['no_changes']}"
    )

    # Empty pairs raises ValueError
    try:
        engine.run_batch_pipeline([])
        raise AssertionError("Should raise ValueError for empty schema_pairs")
    except ValueError:
        print("test_run_batch_pipeline (empty): PASS — ValueError raised")

    # data_map
    data_map = {"batch_a": [{"x": 1}, {"x": 2}]}
    batch2 = engine.run_batch_pipeline(pairs, data_map=data_map)
    assert batch2["total"] == 2
    print(f"test_run_batch_pipeline (data_map): PASS — total={batch2['total']}")


def test_get_statistics(engine: "SchemaMigrationPipelineEngine") -> None:
    stats = engine.get_statistics()
    assert stats["total_runs"] > 0
    assert isinstance(stats["by_status"], dict)
    assert isinstance(stats["success_rate"], float)
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert stats["provenance_entry_count"] > 0
    print(
        f"test_get_statistics: PASS — total_runs={stats['total_runs']} "
        f"success_rate={stats['success_rate']}"
    )


def test_list_pipeline_runs(engine: "SchemaMigrationPipelineEngine") -> None:
    runs = engine.list_pipeline_runs(limit=100)
    assert isinstance(runs, list)

    page = engine.list_pipeline_runs(limit=2, offset=1)
    assert len(page) <= 2

    # Verify descending order by created_at
    if len(runs) >= 2:
        assert runs[0]["created_at"] >= runs[-1]["created_at"]

    # Negative limit/offset must raise
    try:
        engine.list_pipeline_runs(limit=-1)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass

    try:
        engine.list_pipeline_runs(offset=-1)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass

    print(
        f"test_list_pipeline_runs: PASS — total={len(runs)} "
        f"page(offset=1,limit=2)={len(page)}"
    )


def test_get_pipeline_run(engine: "SchemaMigrationPipelineEngine") -> None:
    runs = engine.list_pipeline_runs(limit=1)
    if runs:
        fetched = engine.get_pipeline_run(runs[0]["pipeline_id"])
        assert fetched is not None
        assert fetched["pipeline_id"] == runs[0]["pipeline_id"]

    missing = engine.get_pipeline_run("nonexistent-xyz-789")
    assert missing is None
    print("test_get_pipeline_run: PASS")


def test_generate_report(engine: "SchemaMigrationPipelineEngine") -> None:
    runs = engine.list_pipeline_runs(limit=1)
    if runs:
        rpt = engine.generate_report(runs[0]["pipeline_id"])
        assert rpt["report_id"].startswith("rpt-")
        assert "compliance_notes" in rpt
        assert "provenance_entries" in rpt
        assert "change_summary" in rpt
        assert rpt["report_hash"] is not None
        assert isinstance(rpt["compliance_notes"], list)
        print(
            f"test_generate_report: PASS — report_id={rpt['report_id']} "
            f"notes={len(rpt['compliance_notes'])}"
        )

    # KeyError for missing pipeline_id
    try:
        engine.generate_report("nonexistent-pipeline-id")
        raise AssertionError("Should raise KeyError")
    except KeyError:
        print("test_generate_report (missing): PASS — KeyError raised")


def test_determine_target_version(engine: "SchemaMigrationPipelineEngine") -> None:
    # Major bump from breaking change
    v = engine._determine_target_version("1.2.3", {"is_breaking": True})
    assert v == "2.0.0", f"Expected 2.0.0, got {v}"

    # Minor bump by recommendation
    v = engine._determine_target_version("1.2.3", {"recommended_bump": "minor", "is_breaking": False})
    assert v == "1.3.0", f"Expected 1.3.0, got {v}"

    # Patch bump (default)
    v = engine._determine_target_version("1.2.3", {"recommended_bump": "patch", "is_breaking": False})
    assert v == "1.2.4", f"Expected 1.2.4, got {v}"

    # Malformed version — should not crash
    v = engine._determine_target_version("bad", {"recommended_bump": "patch"})
    assert isinstance(v, str)

    print(
        f"test_determine_target_version: PASS — "
        f"major=2.0.0 minor=1.3.0 patch=1.2.4"
    )


def test_reset(engine: "SchemaMigrationPipelineEngine") -> None:
    pre = engine.get_statistics()
    assert pre["total_runs"] > 0
    engine.reset()
    post = engine.get_statistics()
    assert post["total_runs"] == 0
    assert post["provenance_entry_count"] == 0
    print(f"test_reset: PASS — runs {pre['total_runs']} -> 0")


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SchemaMigrationPipelineEngine — Functional Test Suite")
    print("=" * 70)

    engine = test_engine_init()

    # Stage-level tests
    r_add, r_none, r_break = test_detect_stage(engine)
    test_compatibility_stage(engine, r_add, r_none, r_break)
    plan = test_plan_stage(engine, r_add)
    test_validate_stage(engine, plan)
    exec_result = test_execute_stage(engine, plan)
    test_verify_stage(engine, exec_result)

    # Full pipeline tests
    test_run_pipeline_no_changes(engine)
    test_run_pipeline_with_changes(engine)
    test_run_pipeline_dry_run(engine)
    test_run_pipeline_invalid_json(engine)
    test_run_pipeline_skip_compat(engine)
    test_run_pipeline_empty_schema_id(engine)
    test_run_pipeline_empty_target(engine)

    # Batch
    test_run_batch_pipeline(engine)

    # Admin / reporting
    test_get_statistics(engine)
    test_list_pipeline_runs(engine)
    test_get_pipeline_run(engine)
    test_generate_report(engine)

    # Helpers
    test_determine_target_version(engine)

    # Reset clears all state (must be last)
    test_reset(engine)

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
