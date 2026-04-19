from pathlib import Path
import hashlib

from cbam_pack.pipeline import CBAMPipeline


def _fixtures() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "examples" / "sample_config.yaml",
        root / "examples" / "sample_imports.csv",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_excel_output_is_bit_for_bit_deterministic(tmp_path: Path) -> None:
    config_path, imports_path = _fixtures()
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"

    result1 = CBAMPipeline(
        config_path=config_path,
        imports_path=imports_path,
        output_dir=run1,
        verbose=False,
        dry_run=False,
    ).run()
    result2 = CBAMPipeline(
        config_path=config_path,
        imports_path=imports_path,
        output_dir=run2,
        verbose=False,
        dry_run=False,
    ).run()

    assert result1.success is True
    assert result2.success is True

    excel1 = run1 / "report_summary.xlsx"
    excel2 = run2 / "report_summary.xlsx"
    assert excel1.exists()
    assert excel2.exists()
    assert _sha256(excel1) == _sha256(excel2)
