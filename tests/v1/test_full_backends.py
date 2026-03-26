from pathlib import Path
import shutil
import json

import greenlang.v1.backends as backends
from greenlang.v1.standards import compare_artifact_hashes


def _contract_subset(root: Path, artifacts: list[str]) -> Path:
    subset = root / "_contract_subset"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        source = root / artifact
        if source.exists():
            target = subset / artifact
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return subset


def test_cbam_backend_adapter_requires_native_in_strict_mode(tmp_path, monkeypatch) -> None:
    input_path = Path("cbam-pack-mvp/examples/sample_config.yaml")
    out_dir = tmp_path / "cbam_native"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        (out_dir / "audit").mkdir(parents=True, exist_ok=True)
        (out_dir / "cbam_report.xml").write_text("<cbam/>", encoding="utf-8")
        (out_dir / "report_summary.xlsx").write_text("binary-placeholder", encoding="utf-8")
        (out_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (out_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return 0, "native cbam ok"

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)

    result = backends.run_cbam_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert result.success
    assert result.native_backend_used
    assert not result.fallback_used
    assert (out_dir / "audit" / "observability_event.json").exists()


def test_cbam_backend_strict_mode_fails_without_fallback(tmp_path, monkeypatch) -> None:
    input_path = Path("cbam-pack-mvp/examples/sample_config.yaml")
    out_dir = tmp_path / "cbam_strict_fail"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        return 1, "native cbam failed"

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)

    result = backends.run_cbam_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert not result.success
    assert not result.native_backend_used
    assert not result.fallback_used
    assert result.exit_code == 1
    assert any("cbam backend command failed" in err for err in result.errors)


def test_csrd_backend_adapter_requires_native_in_strict_mode(tmp_path, monkeypatch) -> None:
    input_path = Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv")
    out_dir = tmp_path / "csrd_native"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        return 0, "native csrd ok"

    def _fake_materialize(output_dir):
        (output_dir / "esrs_report.json").write_text('{"status":"ok"}', encoding="utf-8")
        return True

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)
    monkeypatch.setattr(backends, "_materialize_csrd_contract_artifact", _fake_materialize)

    result = backends.run_csrd_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert result.success
    assert result.native_backend_used
    assert not result.fallback_used
    assert (out_dir / "esrs_report.json").exists()
    assert (out_dir / "audit" / "run_manifest.json").exists()
    assert (out_dir / "audit" / "checksums.json").exists()


def test_csrd_backend_strict_mode_fails_without_fallback(tmp_path, monkeypatch) -> None:
    input_path = Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv")
    out_dir = tmp_path / "csrd_strict_fail"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        return 1, "native csrd failed"

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)

    result = backends.run_csrd_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert not result.success
    assert not result.native_backend_used
    assert not result.fallback_used
    assert result.exit_code == 1
    assert any("csrd backend command failed" in err for err in result.errors)


def test_csrd_backend_adapter_generates_required_artifacts(tmp_path) -> None:
    input_path = Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv")
    run1 = tmp_path / "csrd_run1"
    run2 = tmp_path / "csrd_run2"
    result_1 = backends.run_csrd_backend(input_path=input_path, output_dir=run1, strict=False, allow_fallback=True)
    result_2 = backends.run_csrd_backend(input_path=input_path, output_dir=run2, strict=False, allow_fallback=True)
    assert result_1.success
    assert result_2.success
    assert (run1 / "esrs_report.json").exists()
    assert (run1 / "audit" / "run_manifest.json").exists()
    assert (run1 / "audit" / "checksums.json").exists()
    determinism = compare_artifact_hashes(
        _contract_subset(run1, ["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"]),
        _contract_subset(run2, ["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"]),
    )
    assert determinism.same_fileset
    assert determinism.diff_count == 0


def test_vcci_backend_adapter_requires_native_in_strict_mode(tmp_path, monkeypatch) -> None:
    input_path = Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv")
    out_dir = tmp_path / "vcci_native"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        return 0, "native vcci ok"

    def _fake_materialize(output_dir):
        (output_dir / "scope3_inventory.json").write_text('{"status":"ok"}', encoding="utf-8")
        return True

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)
    monkeypatch.setattr(backends, "_materialize_vcci_contract_artifact", _fake_materialize)

    result = backends.run_vcci_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert result.success
    assert result.native_backend_used
    assert not result.fallback_used
    assert (out_dir / "scope3_inventory.json").exists()
    assert (out_dir / "audit" / "run_manifest.json").exists()
    assert (out_dir / "audit" / "checksums.json").exists()


def test_vcci_backend_strict_mode_fails_without_fallback(tmp_path, monkeypatch) -> None:
    input_path = Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv")
    out_dir = tmp_path / "vcci_strict_fail"

    def _fake_run_subprocess(command, cwd, env_overrides=None):
        return 1, "native vcci failed"

    monkeypatch.setattr(backends, "_run_subprocess", _fake_run_subprocess)

    result = backends.run_vcci_backend(
        input_path=input_path,
        output_dir=out_dir,
        strict=True,
        allow_fallback=False,
    )
    assert not result.success
    assert not result.native_backend_used
    assert not result.fallback_used
    assert result.exit_code == 1
    assert any("vcci backend command failed" in err for err in result.errors)


def test_vcci_backend_adapter_generates_required_artifacts(tmp_path) -> None:
    input_path = Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv")
    run1 = tmp_path / "vcci_run1"
    run2 = tmp_path / "vcci_run2"
    result_1 = backends.run_vcci_backend(input_path=input_path, output_dir=run1, strict=False, allow_fallback=True)
    result_2 = backends.run_vcci_backend(input_path=input_path, output_dir=run2, strict=False, allow_fallback=True)
    assert result_1.success
    assert result_2.success
    assert (run1 / "scope3_inventory.json").exists()
    assert (run1 / "audit" / "run_manifest.json").exists()
    assert (run1 / "audit" / "checksums.json").exists()
    determinism = compare_artifact_hashes(
        _contract_subset(run1, ["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"]),
        _contract_subset(run2, ["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"]),
    )
    assert determinism.same_fileset
    assert determinism.diff_count == 0


def test_csrd_materialization_preserves_semantic_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "csrd_semantic"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline_id": "csrd-esrs-core",
        "execution_timestamp": "2026-01-01T00:00:00Z",
        "status": "success",
        "compliance_status": "PASS",
        "total_data_points_processed": 12,
        "csrd_report": {
            "metadata": {
                "validation_status": "PASS",
                "validation_errors": 0,
                "validation_warnings": 1,
                "total_xbrl_facts": 42,
                "narratives_generated": 5,
                "locale": "en",
            },
            "outputs": {
                "esef_package": {
                    "file_path": "out/esef.zip",
                    "file_size_bytes": 1024,
                    "package_id": "pkg-1",
                }
            },
        },
        "compliance_audit": {
            "compliance_report": {
                "total_rules_checked": 12,
                "rules_passed": 11,
                "rules_failed": 1,
                "critical_failures": 0,
            }
        },
    }
    (output_dir / "pipeline_result.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    assert backends._materialize_csrd_contract_artifact(output_dir)
    contract = json.loads((output_dir / "esrs_report.json").read_text(encoding="utf-8"))
    assert contract["records_processed"] == 12
    assert contract["report_metadata"]["total_xbrl_facts"] == 42
    assert contract["report_metadata"]["validation_status"] == "PASS"
    assert contract["esef_package"]["file_count"] >= 0
    assert contract["compliance_audit"]["rules_checked"] == 12


def test_vcci_materialization_uses_scope3_report_totals(tmp_path: Path) -> None:
    output_dir = tmp_path / "vcci_semantic"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": "RUN-20260101000000",
        "calculation_results": {
            "total_emissions": 1234.567,
            "categories": {
                "1": {"emissions_tco2e": 1000.0},
                "2": {"emissions_tco2e": 234.567},
            },
        },
    }
    (output_dir / "scope3_report_RUN-20260101000000.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    assert backends._materialize_vcci_contract_artifact(output_dir)
    contract = json.loads((output_dir / "scope3_inventory.json").read_text(encoding="utf-8"))
    assert contract["records_processed"] == 2
    assert contract["total_emissions_kgco2e"] == 1234.567
