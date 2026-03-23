from pathlib import Path

from typer.testing import CliRunner

from cbam_pack.pipeline import CBAMPipeline
from greenlang.cli.main import app


def _fixtures() -> tuple[Path, Path]:
    mvp_root = Path(__file__).resolve().parents[1]
    return (
        mvp_root / "examples" / "sample_config.yaml",
        mvp_root / "examples" / "sample_imports.csv",
    )


def test_gl_run_cbam_requires_required_inputs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "cbam"])
    assert result.exit_code == 2
    assert "Usage for CBAM" in result.output


def test_gl_run_cbam_generates_artifacts(tmp_path: Path) -> None:
    config_path, imports_path = _fixtures()
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["run", "cbam", str(config_path), str(imports_path), str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "CBAM run completed" in result.output


def test_pipeline_reports_xml_validation_status(tmp_path: Path) -> None:
    config_path, imports_path = _fixtures()
    result = CBAMPipeline(
        config_path=config_path,
        imports_path=imports_path,
        output_dir=tmp_path,
        verbose=False,
        dry_run=False,
    ).run()

    assert result.success is True
    assert result.xml_validation is not None
    assert result.xml_validation.get("status") == "PASS"
    assert (tmp_path / "cbam_report.xml").exists()
    assert (tmp_path / "report_summary.xlsx").exists()
    assert (tmp_path / "audit" / "run_manifest.json").exists()
    assert (tmp_path / "audit" / "checksums.json").exists()
