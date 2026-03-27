import importlib
import json

from typer.testing import CliRunner

from greenlang.cli.main import app
from greenlang.v1.backends import BackendRunResult


def test_v1_status_command_lists_targets() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["v1", "status"])
    assert result.exit_code == 0
    assert "GL-CBAM-APP" in result.output
    assert "GL-CSRD-APP" in result.output
    assert "GL-VCCI-Carbon-APP" in result.output


def test_v1_gate_command_passes() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["v1", "gate"])
    assert result.exit_code == 0
    assert "v1 release gates passed" in result.output


def test_v1_contract_and_policy_commands_pass() -> None:
    runner = CliRunner()
    contracts = runner.invoke(app, ["v1", "validate-contracts"])
    policy = runner.invoke(app, ["v1", "check-policy"])
    smoke = runner.invoke(app, ["v1", "smoke"])
    assert contracts.exit_code == 0
    assert policy.exit_code == 0
    assert smoke.exit_code == 0


def test_v1_run_profile_smoke_for_all_profiles(tmp_path) -> None:
    runner = CliRunner()
    cbam_out = tmp_path / "cbam"
    csrd_out = tmp_path / "csrd"
    vcci_out = tmp_path / "vcci"
    cbam_input = "applications/GL-CBAM-APP/v1/smoke_input.json"
    csrd_input = "applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1/smoke_input.json"
    vcci_input = "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1/smoke_input.json"

    cbam = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "cbam",
            cbam_input,
            "placeholder-imports.csv",
            str(cbam_out),
            "true",
        ],
    )
    csrd = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "csrd",
            csrd_input,
            "-",
            str(csrd_out),
            "true",
        ],
    )
    vcci = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "vcci",
            vcci_input,
            "-",
            str(vcci_out),
            "true",
        ],
    )
    assert cbam.exit_code == 0
    assert csrd.exit_code == 0
    assert vcci.exit_code == 0


def test_v1_run_profile_full_backend_csrd_vcci_with_optional_fallback(tmp_path, monkeypatch) -> None:
    # Resilient integration lane for local/dev environments.
    monkeypatch.setenv("GL_V1_ALLOW_BACKEND_FALLBACK", "1")
    runner = CliRunner()
    csrd_out = tmp_path / "csrd_full"
    vcci_out = tmp_path / "vcci_full"
    csrd_input = "applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv"
    vcci_input = "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv"

    csrd = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "csrd",
            csrd_input,
            "-",
            str(csrd_out),
            "false",
        ],
    )
    vcci = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "vcci",
            vcci_input,
            "-",
            str(vcci_out),
            "false",
        ],
    )
    assert csrd.exit_code == 0
    assert vcci.exit_code == 0
    assert (csrd_out / "esrs_report.json").exists()
    assert (csrd_out / "audit" / "run_manifest.json").exists()
    assert (vcci_out / "scope3_inventory.json").exists()
    assert (vcci_out / "audit" / "run_manifest.json").exists()


def test_v1_run_profile_full_backend_csrd_vcci_strict_native(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GL_V1_ALLOW_BACKEND_FALLBACK", "0")
    runner = CliRunner()
    csrd_out = tmp_path / "csrd_full_native"
    vcci_out = tmp_path / "vcci_full_native"
    csrd_input = "applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv"
    vcci_input = "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv"

    csrd = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "csrd",
            csrd_input,
            "-",
            str(csrd_out),
            "false",
        ],
    )
    vcci = runner.invoke(
        app,
        [
            "v1",
            "run-profile",
            "vcci",
            vcci_input,
            "-",
            str(vcci_out),
            "false",
        ],
    )
    assert csrd.exit_code == 0
    assert vcci.exit_code == 0
    csrd_manifest = json.loads((csrd_out / "audit" / "run_manifest.json").read_text(encoding="utf-8"))
    vcci_manifest = json.loads((vcci_out / "audit" / "run_manifest.json").read_text(encoding="utf-8"))
    csrd_report = json.loads((csrd_out / "esrs_report.json").read_text(encoding="utf-8"))
    vcci_inventory = json.loads((vcci_out / "scope3_inventory.json").read_text(encoding="utf-8"))
    assert csrd_manifest["execution_mode"] == "native"
    assert vcci_manifest["execution_mode"] == "native"
    assert csrd_manifest["app_id"] == "GL-CSRD-APP"
    assert vcci_manifest["app_id"] == "GL-VCCI-Carbon-APP"
    assert csrd_report.get("app_id") == "GL-CSRD-APP"
    assert isinstance(csrd_report.get("records_processed", 0), int)
    assert csrd_report.get("records_processed", 0) >= 0
    assert vcci_inventory.get("app_id") == "GL-VCCI-Carbon-APP"
    assert isinstance(vcci_inventory.get("records_processed", 0), int)
    assert vcci_inventory.get("records_processed", 0) >= 0
    assert isinstance(vcci_inventory.get("total_emissions_kgco2e", 0.0), (int, float))
    assert vcci_inventory.get("total_emissions_kgco2e", 0.0) >= 0


def test_v1_full_backend_checks_command_passes() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["v1", "full-backend-checks"])
    assert result.exit_code == 0


def test_run_csrd_strict_mode_exits_nonzero_without_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GL_V1_ALLOW_BACKEND_FALLBACK", "0")

    def _fail_csrd_backend(*args, **kwargs):
        return BackendRunResult(
            success=False,
            exit_code=19,
            artifacts=[],
            errors=["native csrd unavailable"],
            warnings=[],
            native_backend_used=False,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_csrd_backend", _fail_csrd_backend)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "csrd",
            "applications/GL-CSRD-APP/CSRD-Reporting-Platform/examples/demo_esg_data.csv",
            str(tmp_path / "csrd_out"),
        ],
    )
    assert result.exit_code == 19
    assert "native csrd unavailable" in result.output


def test_run_vcci_strict_mode_exits_nonzero_without_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GL_V1_ALLOW_BACKEND_FALLBACK", "0")

    def _fail_vcci_backend(*args, **kwargs):
        return BackendRunResult(
            success=False,
            exit_code=23,
            artifacts=[],
            errors=["native vcci unavailable"],
            warnings=[],
            native_backend_used=False,
            fallback_used=False,
        )

    cli_main_module = importlib.import_module("greenlang.cli.main")
    monkeypatch.setattr(cli_main_module, "run_vcci_backend", _fail_vcci_backend)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "vcci",
            "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/examples/sample_category1_batch.csv",
            str(tmp_path / "vcci_out"),
        ],
    )
    assert result.exit_code == 23
    assert "native vcci unavailable" in result.output

