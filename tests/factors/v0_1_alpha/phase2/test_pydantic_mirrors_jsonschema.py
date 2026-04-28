# -*- coding: utf-8 -*-
"""Phase 2 Block 1 gate — Pydantic schema vs frozen JSON Schema parity.

Calls :func:`FactorRecordV0_1.model_json_schema` and compares its
semantic surface against the frozen ``factor_record_v0_1.schema.json``.
Cosmetic differences (key ordering, ``$defs`` placement, OpenAPI
metadata) are deliberately ignored; the test focuses on:

* Same set of REQUIRED fields.
* Same set of declared properties.
* Same enum values (per property).
* Same regex patterns (per property).
* ``additionalProperties`` policy matches.
* Same nested-object required + property names for ``extraction``,
  ``review``, ``citations`` items, ``uncertainty``.

CTO Phase 2 §2.1 acceptance: the typed mirror MUST stay structurally
aligned with the frozen contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from greenlang.factors.schemas.factor_record_v0_1 import (
    FactorRecordV0_1,
    FrozenSchemaPath,
    model_to_jsonschema_diff,
)


def _load_frozen() -> Dict[str, Any]:
    return json.loads(FrozenSchemaPath.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pydantic_required_fields_match_frozen_schema() -> None:
    """Set of REQUIRED fields is identical between mirror and frozen schema."""
    diff = model_to_jsonschema_diff()
    assert not diff["missing_required"], (
        "Pydantic mirror is missing required fields that the frozen "
        f"schema declares: {diff['missing_required']}"
    )
    assert not diff["extra_required"], (
        "Pydantic mirror declares required fields the frozen schema "
        f"does not: {diff['extra_required']}"
    )


def test_pydantic_property_names_match_frozen_schema() -> None:
    """Set of declared properties is identical."""
    diff = model_to_jsonschema_diff()
    assert not diff["missing_properties"], (
        "Pydantic mirror is missing properties the frozen schema "
        f"declares: {diff['missing_properties']}"
    )
    assert not diff["extra_properties"], (
        "Pydantic mirror declares properties the frozen schema does "
        f"not: {diff['extra_properties']}"
    )


def test_pydantic_enum_values_match_frozen_schema() -> None:
    """Per-property enum values are identical."""
    diff = model_to_jsonschema_diff()
    assert not diff["enum_mismatches"], (
        f"enum mismatches between Pydantic and frozen schema: "
        f"{diff['enum_mismatches']}"
    )


def test_pydantic_patterns_match_frozen_schema() -> None:
    """Per-property regex patterns are identical."""
    diff = model_to_jsonschema_diff()
    assert not diff["pattern_mismatches"], (
        f"pattern mismatches between Pydantic and frozen schema: "
        f"{diff['pattern_mismatches']}"
    )


def test_pydantic_nested_objects_match_frozen_schema() -> None:
    """Nested ``extraction`` and ``review`` objects align on required + names.

    The frozen schema declares ``extraction.required`` (12 fields) and
    ``review.required`` (3 base fields). The Pydantic mirror MUST agree
    on both the required set and the full property-name set so callers
    cannot smuggle extra keys past the typed layer.
    """
    diff = model_to_jsonschema_diff()
    assert not diff["nested_required_mismatches"], (
        f"nested-required mismatches: "
        f"{diff['nested_required_mismatches']}"
    )
    assert not diff["nested_property_name_mismatches"], (
        f"nested-property-name mismatches: "
        f"{diff['nested_property_name_mismatches']}"
    )


def test_pydantic_additional_properties_policy_matches() -> None:
    """``additionalProperties: false`` policy survives the mirror."""
    diff = model_to_jsonschema_diff()
    assert diff["additional_properties_match"], (
        "additionalProperties policy diverges between Pydantic and "
        "frozen schema (frozen is additionalProperties=false)"
    )
    assert diff["type_match"], (
        "top-level 'type' diverges between Pydantic and frozen schema"
    )


def test_full_diff_is_empty_or_documented() -> None:
    """Single aggregated assertion: the diff is empty.

    This is the umbrella test the CI gate relies on. Any non-empty
    section is a failure unless explicitly documented in this test as
    an acceptable cosmetic divergence.
    """
    diff = model_to_jsonschema_diff()
    # Acceptable cosmetic diffs — none today. If any cosmetic diff is
    # ever added it MUST be documented here with a written reason and
    # tracked in the Phase 2 KPI dashboard.
    acceptable_cosmetic_diffs: Dict[str, Any] = {}

    non_empty = {
        k: v
        for k, v in diff.items()
        if (isinstance(v, list) and v) or (isinstance(v, bool) and not v)
    }
    # Drop boolean keys that ARE meant to be True.
    bool_ok = ("additional_properties_match", "type_match")
    for k in bool_ok:
        if k in non_empty and diff.get(k) is True:
            non_empty.pop(k, None)

    unaccepted = {
        k: v
        for k, v in non_empty.items()
        if k not in acceptable_cosmetic_diffs
    }
    assert not unaccepted, (
        "Pydantic mirror has diverged from the frozen JSON Schema. "
        "Either (a) update the Pydantic mirror, or (b) document the "
        "diff as cosmetic in this test's `acceptable_cosmetic_diffs` "
        "dict.\n\n"
        f"Unaccepted diffs:\n{json.dumps(unaccepted, indent=2, default=str)}"
    )


def test_pydantic_schema_top_level_type_is_object() -> None:
    """Top-level type must be 'object' to match the frozen schema."""
    pyd = FactorRecordV0_1.model_json_schema()
    assert pyd.get("type") == "object", (
        f"Pydantic top-level 'type' is {pyd.get('type')!r}; expected "
        "'object' to match the frozen schema"
    )


def test_frozen_schema_id_unchanged() -> None:
    """Frozen schema's ``$id`` is the locked Phase 2 contract id.

    Any change here is a breaking-change-without-migration, blocked by
    Phase 2 §2.6 (Schema Evolution Policy).
    """
    schema = _load_frozen()
    assert (
        schema.get("$id")
        == "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json"
    ), (
        "frozen schema $id has changed — this is a breaking change. "
        "Open a v0.2 schema with a NEW $id and document the migration "
        "per Phase 2 §2.6."
    )
