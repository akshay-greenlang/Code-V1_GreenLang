"""Schema-load + minimal-fixture validation for factor_record_v0_1.

This is the first gate of the alpha schema. CI must run this on every PR.
A failure here means the v0.1 alpha contract has been broken or is unloadable.

CTO doc reference: §6.1, §19.1 (FY27 Q1 alpha — exit criterion: "Schema approved").
Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
Schema file: config/schemas/factor_record_v0_1.schema.json (FROZEN 2026-04-25)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "config" / "schemas" / "factor_record_v0_1.schema.json"
FREEZE_NOTE = REPO_ROOT / "config" / "schemas" / "FACTOR_RECORD_V0_1_FREEZE.md"
COMPAT_MAP = REPO_ROOT / "config" / "schemas" / "factor_record_v0_1_to_v1_map.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator(schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture()
def good_factor() -> dict:
    """Minimal valid v0.1 alpha factor: an IPCC AR6 stationary-combustion default."""
    return {
        "urn": "urn:gl:factor:ipcc-ar6:stationary-combustion:natural-gas-residential:v1",
        "factor_id_alias": "EF:IPCC:stationary-combustion:natural-gas-residential:v1",
        "source_urn": "urn:gl:source:ipcc-ar6",
        "factor_pack_urn": "urn:gl:pack:ipcc-ar6:tier-1-defaults:2021.0",
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
        "boundary": "Net calorific value basis. Excludes upstream extraction and distribution losses.",
        "licence": "IPCC-PUBLIC",
        "citations": [
            {
                "type": "publication",
                "value": "IPCC AR6 WG3 Annex III, Table 1.4",
                "title": "IPCC Sixth Assessment Report — Working Group III, Annex III",
            },
            {
                "type": "url",
                "value": "https://www.ipcc.ch/report/ar6/wg3/",
            },
        ],
        "published_at": "2026-04-25T12:00:00Z",
        "extraction": {
            "source_url": "https://www.ipcc.ch/report/ar6/wg3/",
            "source_record_id": "annex-iii;table-1.4;row=natural-gas-residential",
            "source_publication": "IPCC Sixth Assessment Report — Working Group III, Annex III",
            "source_version": "AR6-WG3-Annex-III",
            "raw_artifact_uri": "s3://greenlang-factors-raw/ipcc/ar6/wg3-annex-iii.pdf",
            "raw_artifact_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd00",
            "parser_id": "greenlang.factors.ingestion.parsers.ipcc_defaults",
            "parser_version": "0.1.0",
            "parser_commit": "deadbeefcafe",
            "row_ref": "Sheet=N/A; Table=1.4; Row=Natural Gas (Residential); Column=Default EF (kgCO2e/TJ)",
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


# ---------------------------------------------------------------------------
# Schema-load gates (a failure here breaks every downstream test)
# ---------------------------------------------------------------------------


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file(), f"missing {SCHEMA_PATH}"


def test_freeze_note_exists() -> None:
    assert FREEZE_NOTE.is_file(), f"missing freeze note {FREEZE_NOTE}"


def test_compat_map_exists_and_parses() -> None:
    assert COMPAT_MAP.is_file()
    data = json.loads(COMPAT_MAP.read_text(encoding="utf-8"))
    assert "mapping" in data
    assert "v1_fields_intentionally_omitted_in_v0_1" in data
    # Every v0.1 required top-level field must appear in the mapping
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for required_field in schema["required"]:
        # Compound fields (extraction.*, review.*) live under the parent key only at top level
        assert (
            required_field in data["mapping"]
        ), f"compat map missing required v0_1 field: {required_field}"


def test_schema_id_is_canonical(schema: dict) -> None:
    assert (
        schema["$id"]
        == "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json"
    )


def test_schema_uses_2020_12(schema: dict) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_schema_validator_accepts_itself(schema: dict) -> None:
    Draft202012Validator.check_schema(schema)


# ---------------------------------------------------------------------------
# Required-field invariants — every alpha contract field is locked in
# ---------------------------------------------------------------------------


EXPECTED_REQUIRED = {
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
}


def test_required_top_level_fields_match_freeze(schema: dict) -> None:
    actual = set(schema["required"])
    assert actual == EXPECTED_REQUIRED, (
        f"required field set drift detected — additions or removals require a v0.2 schema bump.\n"
        f"missing: {EXPECTED_REQUIRED - actual}\n"
        f"extra:   {actual - EXPECTED_REQUIRED}"
    )


def test_extraction_required_fields_complete(schema: dict) -> None:
    extraction = schema["properties"]["extraction"]
    assert set(extraction["required"]) == {
        "source_url",
        "source_record_id",
        "source_publication",
        "source_version",
        "raw_artifact_uri",
        "raw_artifact_sha256",
        "parser_id",
        "parser_version",
        "parser_commit",
        "row_ref",
        "ingested_at",
        "operator",
    }


def test_gwp_basis_is_ar6_only(schema: dict) -> None:
    assert schema["properties"]["gwp_basis"]["enum"] == ["ar6"]


def test_review_status_enum(schema: dict) -> None:
    assert schema["properties"]["review"]["properties"]["review_status"]["enum"] == [
        "pending",
        "approved",
        "rejected",
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_minimal_good_factor_validates(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    errors = sorted(validator.iter_errors(good_factor), key=lambda e: list(e.path))
    assert errors == [], "\n".join(e.message for e in errors)


# ---------------------------------------------------------------------------
# Negative paths — every required field rejected when missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", sorted(EXPECTED_REQUIRED))
def test_missing_required_field_fails(
    validator: Draft202012Validator, good_factor: dict, missing_field: str
) -> None:
    bad = dict(good_factor)
    bad.pop(missing_field, None)
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_bad_urn_pattern_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor, urn="urn:gl:factor:no-version-suffix")
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_value_zero_fails(validator: Draft202012Validator, good_factor: dict) -> None:
    bad = dict(good_factor, value=0)
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_gwp_basis_ar5_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor, gwp_basis="ar5")
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_geography_urn_pattern_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor, geography_urn="urn:gl:geo:bad:in")
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_review_approved_without_approved_by_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor)
    bad["review"] = {
        "review_status": "approved",
        "reviewer": "human:reviewer@greenlang.io",
        "reviewed_at": "2026-04-25T11:58:00Z",
        # approved_by + approved_at intentionally missing
    }
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_review_rejected_without_reason_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor)
    bad["review"] = {
        "review_status": "rejected",
        "reviewer": "human:reviewer@greenlang.io",
        "reviewed_at": "2026-04-25T11:58:00Z",
        # rejection_reason intentionally missing
    }
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_extraction_missing_parser_commit_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor)
    bad["extraction"] = dict(good_factor["extraction"])
    bad["extraction"].pop("parser_commit")
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_extraction_missing_raw_artifact_sha256_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor)
    bad["extraction"] = dict(good_factor["extraction"])
    bad["extraction"].pop("raw_artifact_sha256")
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_extraction_bad_sha256_fails(
    validator: Draft202012Validator, good_factor: dict
) -> None:
    bad = dict(good_factor)
    bad["extraction"] = dict(good_factor["extraction"])
    bad["extraction"]["raw_artifact_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        validator.validate(bad)
