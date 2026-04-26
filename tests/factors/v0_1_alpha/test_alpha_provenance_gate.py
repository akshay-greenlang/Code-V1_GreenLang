"""Wave B / WS2-T1 — Alpha Provenance Gate enforcement tests.

Validates that ``AlphaProvenanceGate`` rejects every record missing the
v0.1 alpha required provenance / review metadata, plus the additional
format-level constraints (sha256, parser_commit, semver, operator,
gwp_basis, approved_by/approved_at, rejection_reason).

CTO doc reference: §6.1, §19.1 (FY27 Q1 alpha — provenance fields complete).
Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGate,
    AlphaProvenanceGateError,
)


# Mirror of the minimal good factor used by test_factor_record_v0_1_schema_loads.py.
_GOOD_FACTOR: dict = {
    "urn": "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1",
    "factor_id_alias": "EF:IPCC:stationary-combustion:natural-gas-residential:v1",
    "source_urn": "urn:gl:source:ipcc-ar6",
    "factor_pack_urn": "urn:gl:pack:ipcc-ar6:tier-1-defaults:v1",
    "name": "Stationary combustion of natural gas (residential), CO2e",
    "description": (
        "Default Tier 1 emission factor for residential stationary combustion of "
        "natural gas, expressed in kgCO2e/TJ on a net calorific value (NCV) basis. "
        "Boundary excludes upstream extraction and distribution losses."
    ),
    "category": "fuel",
    "value": 56100.0,
    "unit_urn": "urn:gl:unit:kgco2e/tj",
    "gwp_basis": "ar6",
    "gwp_horizon": 100,
    "geography_urn": "urn:gl:geo:global:world",
    "vintage_start": "2021-01-01",
    "vintage_end": "2099-12-31",
    "resolution": "annual",
    "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
    "boundary": (
        "Net calorific value basis. Excludes upstream extraction and distribution "
        "losses."
    ),
    "licence": "IPCC-PUBLIC",
    "citations": [
        {
            "type": "publication",
            "value": "IPCC AR6 WG3 Annex III, Table 1.4",
            "title": "IPCC Sixth Assessment Report — Working Group III, Annex III",
        },
        {"type": "url", "value": "https://www.ipcc.ch/report/ar6/wg3/"},
    ],
    "published_at": "2026-04-25T12:00:00Z",
    "extraction": {
        "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
        "source_record_id": "annex-iii;table-1.4;row=natural-gas-residential",
        "source_publication": (
            "IPCC Sixth Assessment Report — Working Group III, Annex III"
        ),
        "source_version": "AR6-WG3-Annex-III",
        "raw_artifact_uri": "s3://greenlang-factors-raw/ipcc/ar6/wg3-annex-iii.pdf",
        "raw_artifact_sha256": (
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd00"
        ),
        "parser_id": "greenlang.factors.ingestion.parsers.ipcc_defaults",
        "parser_version": "0.1.0",
        "parser_commit": "deadbeefcafe",
        "row_ref": (
            "Sheet=N/A; Table=1.4; Row=Natural Gas (Residential); "
            "Column=Default EF (kgCO2e/TJ)"
        ),
        "ingested_at": "2026-04-25T11:55:00Z",
        "operator": "bot:parser_ipcc_defaults",
    },
    "review": {
        "review_status": "approved",
        "reviewer": "human:methodology-lead@greenlang.io",
        "reviewed_at": "2026-04-25T11:58:00Z",
        "approved_by": "human:methodology-lead@greenlang.io",
        "approved_at": "2026-04-25T11:59:00Z",
    },
}


REQUIRED_TOP_LEVEL_FIELDS = sorted(
    [
        "urn",
        "source_urn",
        "factor_pack_urn",
        "name",
        "description",
        "category",
        "value",
        "unit_urn",
        "gwp_basis",
        "gwp_horizon",
        "geography_urn",
        "vintage_start",
        "vintage_end",
        "resolution",
        "methodology_urn",
        "boundary",
        "licence",
        "citations",
        "published_at",
        "extraction",
        "review",
    ]
)


REQUIRED_EXTRACTION_FIELDS = sorted(AlphaProvenanceGate.REQUIRED_EXTRACTION_FIELDS)
REQUIRED_REVIEW_FIELDS = sorted(AlphaProvenanceGate.REQUIRED_REVIEW_FIELDS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def good_factor() -> dict:
    return copy.deepcopy(_GOOD_FACTOR)


@pytest.fixture(scope="module")
def gate() -> AlphaProvenanceGate:
    return AlphaProvenanceGate()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_minimal_good_factor_passes(gate: AlphaProvenanceGate, good_factor: dict) -> None:
    assert gate.validate(good_factor) == []


def test_assert_valid_does_not_raise_on_good(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    gate.assert_valid(good_factor)


def test_default_schema_path_resolves(tmp_path: Path) -> None:
    g = AlphaProvenanceGate()
    # Force the validator to load.
    g._ensure_validator()
    assert g._schema is not None
    assert g._schema["$id"].endswith("factor_record_v0_1.schema.json")


# ---------------------------------------------------------------------------
# Missing top-level required fields (21 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", REQUIRED_TOP_LEVEL_FIELDS)
def test_missing_top_level_required_field_fails(
    gate: AlphaProvenanceGate, good_factor: dict, field_name: str
) -> None:
    bad = copy.deepcopy(good_factor)
    bad.pop(field_name, None)
    failures = gate.validate(bad)
    assert failures, f"removing {field_name!r} should produce at least one failure"
    # ``review`` removal won't surface as schema[review.*] but rather as a
    # top-level required-field error; either way we expect the field name
    # to appear in at least one failure message.
    assert any(field_name in msg for msg in failures), (
        f"expected {field_name!r} in some failure message; got {failures!r}"
    )


# ---------------------------------------------------------------------------
# Missing extraction sub-fields (12 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sub_field", REQUIRED_EXTRACTION_FIELDS)
def test_missing_extraction_sub_field_fails(
    gate: AlphaProvenanceGate, good_factor: dict, sub_field: str
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["extraction"].pop(sub_field, None)
    failures = gate.validate(bad)
    assert failures, f"removing extraction.{sub_field} must fail the gate"
    assert any(sub_field in msg for msg in failures)


# ---------------------------------------------------------------------------
# Missing review sub-fields (3 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sub_field", REQUIRED_REVIEW_FIELDS)
def test_missing_review_sub_field_fails(
    gate: AlphaProvenanceGate, good_factor: dict, sub_field: str
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["review"].pop(sub_field, None)
    failures = gate.validate(bad)
    assert failures
    assert any(sub_field in msg for msg in failures)


# ---------------------------------------------------------------------------
# review.approved without approved_by / approved_at (2 cases)
# ---------------------------------------------------------------------------


def test_review_approved_without_approved_by_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["review"].pop("approved_by", None)
    failures = gate.validate(bad)
    assert any("approved_by" in msg for msg in failures)


def test_review_approved_without_approved_at_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["review"].pop("approved_at", None)
    failures = gate.validate(bad)
    assert any("approved_at" in msg for msg in failures)


# ---------------------------------------------------------------------------
# review.rejected without rejection_reason
# ---------------------------------------------------------------------------


def test_review_rejected_without_rejection_reason_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["review"] = {
        "review_status": "rejected",
        "reviewer": "human:reviewer@greenlang.io",
        "reviewed_at": "2026-04-25T11:58:00Z",
        # rejection_reason intentionally missing
    }
    failures = gate.validate(bad)
    assert any("rejection_reason" in msg for msg in failures)


# ---------------------------------------------------------------------------
# Format-level alpha checks
# ---------------------------------------------------------------------------


def test_bad_sha256_format_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["extraction"]["raw_artifact_sha256"] = "not-a-real-hash"
    failures = gate.validate(bad)
    assert any("raw_artifact_sha256" in msg for msg in failures)


def test_uppercase_sha256_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    # 64 chars but uppercase — schema regex is lowercase-only.
    bad["extraction"]["raw_artifact_sha256"] = "A" * 64
    failures = gate.validate(bad)
    assert any("raw_artifact_sha256" in msg for msg in failures)


def test_bad_parser_commit_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["extraction"]["parser_commit"] = "ZZZZ"  # too short and non-hex
    failures = gate.validate(bad)
    assert any("parser_commit" in msg for msg in failures)


def test_bad_parser_version_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["extraction"]["parser_version"] = "v0.1"  # not semver
    failures = gate.validate(bad)
    assert any("parser_version" in msg for msg in failures)


def test_bad_operator_format_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["extraction"]["operator"] = "anonymous"
    failures = gate.validate(bad)
    assert any("operator" in msg for msg in failures)


def test_human_operator_format_passes(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    rec = copy.deepcopy(good_factor)
    rec["extraction"]["operator"] = "human:engineer@greenlang.io"
    assert gate.validate(rec) == []


# ---------------------------------------------------------------------------
# gwp_basis must be 'ar6'
# ---------------------------------------------------------------------------


def test_gwp_basis_ar5_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["gwp_basis"] = "ar5"
    failures = gate.validate(bad)
    assert any("gwp_basis" in msg for msg in failures)


def test_gwp_basis_missing_fails(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad.pop("gwp_basis", None)
    failures = gate.validate(bad)
    assert any("gwp_basis" in msg for msg in failures)


# ---------------------------------------------------------------------------
# assert_valid + AlphaProvenanceGateError contract
# ---------------------------------------------------------------------------


def test_assert_valid_raises_with_all_failures_concatenated(
    gate: AlphaProvenanceGate, good_factor: dict
) -> None:
    bad = copy.deepcopy(good_factor)
    bad["gwp_basis"] = "ar5"
    bad["extraction"]["raw_artifact_sha256"] = "not-a-hash"
    bad["extraction"]["parser_commit"] = "zzz"
    bad["review"].pop("approved_by", None)

    with pytest.raises(AlphaProvenanceGateError) as excinfo:
        gate.assert_valid(bad)

    err = excinfo.value
    assert isinstance(err.failures, list)
    assert len(err.failures) >= 4
    msg = str(err)
    assert "gwp_basis" in msg
    assert "raw_artifact_sha256" in msg
    assert "parser_commit" in msg
    assert "approved_by" in msg


def test_validate_non_dict_returns_failure(gate: AlphaProvenanceGate) -> None:
    failures = gate.validate("not a record")  # type: ignore[arg-type]
    assert failures and "dict" in failures[0]


# ---------------------------------------------------------------------------
# check_alpha_source soft check
# ---------------------------------------------------------------------------


def test_check_alpha_source_unknown_returns_false(
    gate: AlphaProvenanceGate,
) -> None:
    assert gate.check_alpha_source({"source_urn": "urn:gl:source:not-listed"}) is False


def test_check_alpha_source_missing_field_returns_false(
    gate: AlphaProvenanceGate,
) -> None:
    assert gate.check_alpha_source({}) is False


def test_check_alpha_source_non_dict_returns_false(
    gate: AlphaProvenanceGate,
) -> None:
    assert gate.check_alpha_source("not a record") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Env-var driven alpha_gate_enabled toggle
# ---------------------------------------------------------------------------


def test_alpha_gate_enabled_env_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    from greenlang.factors.quality.alpha_provenance_gate import alpha_gate_enabled

    monkeypatch.setenv("GL_FACTORS_ALPHA_PROVENANCE_GATE", "1")
    assert alpha_gate_enabled(default_on=False) is True


def test_alpha_gate_enabled_env_falsy(monkeypatch: pytest.MonkeyPatch) -> None:
    from greenlang.factors.quality.alpha_provenance_gate import alpha_gate_enabled

    monkeypatch.setenv("GL_FACTORS_ALPHA_PROVENANCE_GATE", "off")
    assert alpha_gate_enabled(default_on=True) is False


def test_alpha_gate_enabled_env_unset_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from greenlang.factors.quality.alpha_provenance_gate import alpha_gate_enabled

    monkeypatch.delenv("GL_FACTORS_ALPHA_PROVENANCE_GATE", raising=False)
    assert alpha_gate_enabled(default_on=True) is True
    assert alpha_gate_enabled(default_on=False) is False
