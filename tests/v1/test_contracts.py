from pathlib import Path

import pytest

from greenlang.v1.contracts import validate_v1_pack, validate_v1_pipeline


V1_TARGETS = [
    Path("applications/GL-CBAM-APP/v1"),
    Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1"),
    Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1"),
]


def test_v1_pack_contracts_validate() -> None:
    for target in V1_TARGETS:
        finding = validate_v1_pack(target / "pack.yaml")
        assert finding.ok, f"{finding.path}: {finding.errors}"


def test_v1_pipeline_contracts_validate() -> None:
    for target in V1_TARGETS:
        finding = validate_v1_pipeline(target / "gl.yaml")
        assert finding.ok, f"{finding.path}: {finding.errors}"


INVALID_FIXTURES_ROOT = Path("tests/v1/fixtures/contracts")


@pytest.mark.parametrize(
    "fixture_name,expected_substring",
    [
        ("invalid_pack_missing_owner.yaml", "non-empty string"),
        ("invalid_pack_signed_without_signatures.yaml", "requires at least one signature"),
    ],
)
def test_v1_pack_contracts_reject_invalid_fixtures(
    fixture_name: str, expected_substring: str
) -> None:
    finding = validate_v1_pack(INVALID_FIXTURES_ROOT / fixture_name)
    assert not finding.ok
    assert any(expected_substring in error for error in finding.errors), finding.errors


@pytest.mark.parametrize(
    "fixture_name,expected_substring",
    [
        ("invalid_pipeline_wrong_stage_order.yaml", "stages must exactly match"),
        ("invalid_pipeline_empty_artifacts.yaml", "artifact_contract must be non-empty"),
        ("invalid_pipeline_empty_command.yaml", "command must be a non-empty string"),
    ],
)
def test_v1_pipeline_contracts_reject_invalid_fixtures(
    fixture_name: str, expected_substring: str
) -> None:
    finding = validate_v1_pipeline(INVALID_FIXTURES_ROOT / fixture_name)
    assert not finding.ok
    assert any(expected_substring in error for error in finding.errors), finding.errors

