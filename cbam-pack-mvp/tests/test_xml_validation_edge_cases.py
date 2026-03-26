from pathlib import Path
import hashlib

from cbam_pack.exporters.xml_generator import CBAMXMLGenerator
from cbam_pack.pipeline import CBAMPipeline


def _fixtures() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "examples" / "sample_config.yaml",
        root / "examples" / "sample_imports.csv",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_xml_schema_is_loaded_from_bundled_file() -> None:
    generator = CBAMXMLGenerator()
    xsd = generator._load_xsd_schema()
    assert "<xs:schema" in xsd
    assert "CBAMReport" in xsd


def test_structural_validation_flags_missing_summary() -> None:
    generator = CBAMXMLGenerator()
    xml = """<?xml version="1.0"?>
<CBAMReport xmlns="urn:cbam:transitional:v1" version="1.0">
  <Header><GeneratedAt>x</GeneratedAt><ReportType>TRANSITIONAL_QUARTERLY</ReportType><SchemaVersion>1.0.0</SchemaVersion></Header>
  <Declarant><Name>x</Name><EORINumber>x</EORINumber><Address><Street>x</Street><City>x</City><PostalCode>x</PostalCode><Country>x</Country></Address><Contact><Name>x</Name><Email>x</Email></Contact></Declarant>
  <ReportingPeriod><Quarter>Q1</Quarter><Year>2025</Year><StartDate>2025-01-01</StartDate><EndDate>2025-03-31</EndDate></ReportingPeriod>
  <ImportedGoods />
</CBAMReport>"""
    errors = generator._structural_validation(xml)
    assert any("Missing required element: Summary" in error for error in errors)


def test_xml_output_is_bit_for_bit_deterministic(tmp_path: Path) -> None:
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
    xml1 = run1 / "cbam_report.xml"
    xml2 = run2 / "cbam_report.xml"
    assert xml1.exists()
    assert xml2.exists()
    assert _sha256(xml1) == _sha256(xml2)
