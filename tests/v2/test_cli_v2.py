import importlib
import json
import sys
from types import ModuleType
from pathlib import Path

from typer.testing import CliRunner

from greenlang.cli.main import app
from greenlang.v1.backends import BackendRunResult


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _install_fake_cbam_pipeline(monkeypatch, success: bool, exit_code: int, errors: list[str]) -> None:
    cbam_pkg = ModuleType("cbam_pack")
    pipeline_mod = ModuleType("cbam_pack.pipeline")

    class _Result:
        def __init__(self) -> None:
            self.success = success
            self.exit_code = exit_code
            self.errors = errors

    class _Pipeline:
        def __init__(self, config_path, imports_path, output_dir, verbose=False, dry_run=False):
            del config_path, imports_path, verbose, dry_run
            self._output_dir = output_dir

        def run(self):
            audit_dir = self._output_dir / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            (self._output_dir / "cbam_report.xml").write_text("<cbam/>", encoding="utf-8")
            (self._output_dir / "report_summary.xlsx").write_text("xlsx-placeholder", encoding="utf-8")
            (audit_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
            (audit_dir / "checksums.json").write_text("{}", encoding="utf-8")
            return _Result()

    pipeline_mod.CBAMPipeline = _Pipeline
    cbam_pkg.pipeline = pipeline_mod
    monkeypatch.setitem(sys.modules, "cbam_pack", cbam_pkg)
    monkeypatch.setitem(sys.modules, "cbam_pack.pipeline", pipeline_mod)


def test_v2_expanded_profiles_run_with_unified_exit_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    eudr_in = tmp_path / "eudr.json"
    ghg_in = tmp_path / "ghg.json"
    iso_in = tmp_path / "iso.json"
    _write_json(eudr_in, {"suppliers": [{"id": "s1", "risk": "low"}]})
    _write_json(
        ghg_in,
        {"activities": [{"quantity": 10, "emission_factor": 1.25}]},
    )
    _write_json(
        iso_in,
        {"controls": [{"id": "c1", "passed": True}, {"id": "c2", "passed": False}]},
    )

    eudr_out = tmp_path / "eudr_out"
    ghg_out = tmp_path / "ghg_out"
    iso_out = tmp_path / "iso_out"
    eudr = runner.invoke(app, ["run", "eudr", str(eudr_in), str(eudr_out)])
    ghg = runner.invoke(app, ["run", "ghg", str(ghg_in), str(ghg_out)])
    iso = runner.invoke(app, ["run", "iso14064", str(iso_in), str(iso_out)])

    assert eudr.exit_code == 0
    assert ghg.exit_code == 0
    assert iso.exit_code == 0
    assert (eudr_out / "due_diligence_statement.json").exists()
    assert (ghg_out / "ghg_inventory.json").exists()
    assert (iso_out / "iso14064_verification_report.json").exists()
    assert (eudr_out / "audit" / "run_manifest.json").exists()
    assert (ghg_out / "audit" / "run_manifest.json").exists()
    assert (iso_out / "audit" / "run_manifest.json").exists()


def test_v2_expanded_profiles_fail_with_validation_exit_two_for_missing_input(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "eudr", str(tmp_path / "missing.json"), str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "EUDR input not found" in result.output


def test_v2_profile_blocked_exit_is_consistent(monkeypatch, tmp_path: Path) -> None:
    def _blocked_backend(*args, **kwargs):
        return BackendRunResult(
            success=True,
            exit_code=4,
            artifacts=["audit/run_manifest.json", "audit/checksums.json"],
            errors=[],
            warnings=["policy gate blocked export"],
            native_backend_used=True,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_v2_profile_backend", _blocked_backend)
    runner = CliRunner()
    input_path = tmp_path / "eudr_input.json"
    _write_json(input_path, {"suppliers": []})
    result = runner.invoke(app, ["run", "eudr", str(input_path), str(tmp_path / "eudr_out")])
    assert result.exit_code == 4
    assert "export blocked" in result.output.lower()


def test_v2_profile_execution_error_is_nonzero(monkeypatch, tmp_path: Path) -> None:
    def _failed_backend(*args, **kwargs):
        return BackendRunResult(
            success=False,
            exit_code=27,
            artifacts=[],
            errors=["v2 runtime failure"],
            warnings=[],
            native_backend_used=False,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_v2_profile_backend", _failed_backend)
    runner = CliRunner()
    input_path = tmp_path / "ghg_input.json"
    _write_json(input_path, {"activities": []})
    result = runner.invoke(app, ["run", "ghg", str(input_path), str(tmp_path / "ghg_out")])
    assert result.exit_code == 27
    assert "v2 runtime failure" in result.output


def test_v2_selected_apps_success_exit_taxonomy(monkeypatch, tmp_path: Path) -> None:
    _install_fake_cbam_pipeline(monkeypatch, success=True, exit_code=0, errors=[])

    def _ok_backend(*args, **kwargs):
        return BackendRunResult(
            success=True,
            exit_code=0,
            artifacts=["audit/run_manifest.json", "audit/checksums.json"],
            errors=[],
            warnings=[],
            native_backend_used=True,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_csrd_backend", _ok_backend)
    monkeypatch.setattr(cli_main_module, "run_vcci_backend", _ok_backend)

    cbam_cfg = tmp_path / "cbam.yaml"
    cbam_imports = tmp_path / "imports.csv"
    cbam_cfg.write_text("year: 2026\n", encoding="utf-8")
    cbam_imports.write_text("importer,quantity\nA,1\n", encoding="utf-8")
    csrd_input = tmp_path / "csrd.json"
    vcci_input = tmp_path / "vcci.json"
    csrd_input.write_text("{}", encoding="utf-8")
    vcci_input.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    cbam = runner.invoke(app, ["run", "cbam", str(cbam_cfg), str(cbam_imports), str(tmp_path / "cbam_out")])
    csrd = runner.invoke(app, ["run", "csrd", str(csrd_input), str(tmp_path / "csrd_out")])
    vcci = runner.invoke(app, ["run", "vcci", str(vcci_input), str(tmp_path / "vcci_out")])

    assert cbam.exit_code == 0
    assert csrd.exit_code == 0
    assert vcci.exit_code == 0


def test_v2_selected_apps_blocked_exit_taxonomy(monkeypatch, tmp_path: Path) -> None:
    _install_fake_cbam_pipeline(
        monkeypatch,
        success=True,
        exit_code=4,
        errors=["policy gate blocked export"],
    )

    def _blocked_backend(*args, **kwargs):
        return BackendRunResult(
            success=True,
            exit_code=4,
            artifacts=["audit/run_manifest.json", "audit/checksums.json"],
            errors=[],
            warnings=["policy gate blocked export"],
            native_backend_used=True,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_csrd_backend", _blocked_backend)
    monkeypatch.setattr(cli_main_module, "run_vcci_backend", _blocked_backend)

    cbam_cfg = tmp_path / "cbam.yaml"
    cbam_imports = tmp_path / "imports.csv"
    cbam_cfg.write_text("year: 2026\n", encoding="utf-8")
    cbam_imports.write_text("importer,quantity\nA,1\n", encoding="utf-8")
    csrd_input = tmp_path / "csrd.json"
    vcci_input = tmp_path / "vcci.json"
    csrd_input.write_text("{}", encoding="utf-8")
    vcci_input.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    cbam = runner.invoke(app, ["run", "cbam", str(cbam_cfg), str(cbam_imports), str(tmp_path / "cbam_out")])
    csrd = runner.invoke(app, ["run", "csrd", str(csrd_input), str(tmp_path / "csrd_out")])
    vcci = runner.invoke(app, ["run", "vcci", str(vcci_input), str(tmp_path / "vcci_out")])

    assert cbam.exit_code == 4
    assert csrd.exit_code == 4
    assert vcci.exit_code == 4
