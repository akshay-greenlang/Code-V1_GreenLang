from pathlib import Path

from greenlang.v2.contracts import validate_v2_pack, validate_v2_pipeline


V2_TARGETS = [
    Path("applications/GL-CBAM-APP/v2"),
    Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v2"),
    Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v2"),
    Path("applications/GL-EUDR-APP/v2"),
    Path("applications/GL-GHG-APP/v2"),
    Path("applications/GL-ISO14064-APP/v2"),
]


def test_v2_pack_contracts_validate() -> None:
    for target in V2_TARGETS:
        finding = validate_v2_pack(target / "pack.yaml")
        assert finding.ok, f"{finding.path}: {finding.errors}"


def test_v2_pipeline_contracts_validate() -> None:
    for target in V2_TARGETS:
        finding = validate_v2_pipeline(target / "gl.yaml")
        assert finding.ok, f"{finding.path}: {finding.errors}"

