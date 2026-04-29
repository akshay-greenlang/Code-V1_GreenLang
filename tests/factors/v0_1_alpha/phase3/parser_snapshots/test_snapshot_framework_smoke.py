# -*- coding: utf-8 -*-
"""Phase 3 — meta-test for the parser-snapshot framework itself.

Verifies the helpers in :mod:`_helper`:

  * a synthetic golden + matching parsed output passes;
  * an off-by-one parsed output fails with a readable diff;
  * a parsed output with a missing provenance field fails the
    provenance drift detector;
  * the regen flag (``UPDATE_PARSER_SNAPSHOT=1``) writes a new golden
    and skips the comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.factors.v0_1_alpha.phase3.parser_snapshots import _helper

# ``pytest.fail`` raises ``Failed`` which inherits from BaseException
# (not Exception) so ``pytest.raises(Exception)`` would miss it. Bind the
# canonical class once.
_PytestFailed = pytest.fail.Exception


def _good_rows() -> List[Dict[str, Any]]:
    return [
        {
            "factor_id": "EF:SMOKE:row-1",
            "value": 0.111,
            "unit": "kgCO2e/kWh",
            "row_ref": "Sheet=A;Row=1",
            "licence": "CC-BY-4.0",
            "raw_artifact_sha256": "a" * 64,
            "citations": [{"type": "url", "value": "https://example.test/1"}],
        },
        {
            "factor_id": "EF:SMOKE:row-2",
            "value": 0.222,
            "unit": "kgCO2e/kWh",
            "row_ref": "Sheet=A;Row=2",
            "licence": "CC-BY-4.0",
            "raw_artifact_sha256": "b" * 64,
            "citations": [{"type": "url", "value": "https://example.test/2"}],
        },
    ]


def _write_golden(tmp_path: Path, parser_id: str, parser_version: str, rows: Any) -> Path:
    p = _helper.snapshot_path(parser_id, parser_version, snapshot_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return p


def test_compare_to_snapshot_passes_on_match(tmp_path: Path):
    rows = _good_rows()
    _write_golden(tmp_path, "smoke", "0.1.0", rows)
    # Should not raise / fail.
    _helper.compare_to_snapshot("smoke", "0.1.0", rows, snapshot_dir=tmp_path)


def test_compare_to_snapshot_fails_on_off_by_one_value(tmp_path: Path):
    golden = _good_rows()
    _write_golden(tmp_path, "smoke", "0.1.0", golden)
    drifted = _good_rows()
    drifted[0]["value"] = drifted[0]["value"] + 0.001  # off-by-one drift

    with pytest.raises(_PytestFailed) as exc_info:
        _helper.compare_to_snapshot(
            "smoke", "0.1.0", drifted, snapshot_dir=tmp_path
        )
    msg = str(exc_info.value)
    assert "smoke@0.1.0" in msg
    assert "row 0" in msg or "first divergence" in msg or "drift" in msg


def test_compare_to_snapshot_fails_on_table_shape_drift(tmp_path: Path):
    golden = _good_rows()
    _write_golden(tmp_path, "smoke", "0.1.0", golden)
    parsed = _good_rows()
    parsed[0]["new_column"] = "drift"  # added column drift

    with pytest.raises(_PytestFailed) as exc_info:
        _helper.compare_to_snapshot(
            "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
        )
    assert "table-shape drift" in str(exc_info.value)


def test_compare_to_snapshot_flags_missing_licence(tmp_path: Path):
    golden = _good_rows()
    _write_golden(tmp_path, "smoke", "0.1.0", golden)
    parsed = _good_rows()
    parsed[0].pop("licence")  # provenance drift

    with pytest.raises(_PytestFailed) as exc_info:
        _helper.compare_to_snapshot(
            "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
        )
    assert "missing required provenance" in str(exc_info.value)
    assert "licence" in str(exc_info.value)


def test_compare_to_snapshot_flags_missing_sha256(tmp_path: Path):
    golden = _good_rows()
    _write_golden(tmp_path, "smoke", "0.1.0", golden)
    parsed = _good_rows()
    parsed[0].pop("raw_artifact_sha256")

    with pytest.raises(_PytestFailed) as exc_info:
        _helper.compare_to_snapshot(
            "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
        )
    assert "raw_artifact_sha256" in str(exc_info.value)


def test_compare_to_snapshot_missing_golden_fails_with_helpful_msg(tmp_path: Path):
    parsed = _good_rows()
    with pytest.raises(_PytestFailed) as exc_info:
        _helper.compare_to_snapshot(
            "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
        )
    assert "snapshot missing" in str(exc_info.value)
    assert "UPDATE_PARSER_SNAPSHOT" in str(exc_info.value)


def test_regenerate_if_env_writes_golden_and_skips(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("UPDATE_PARSER_SNAPSHOT", "1")
    parsed = _good_rows()
    with pytest.raises(pytest.skip.Exception):
        _helper.regenerate_if_env(
            "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
        )
    written = _helper.snapshot_path(
        "smoke", "0.1.0", snapshot_dir=tmp_path
    )
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    # JSON canonicalises key order — payload should round-trip.
    assert isinstance(payload, list)
    assert len(payload) == len(parsed)


def test_regenerate_if_env_no_op_without_flag(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("UPDATE_PARSER_SNAPSHOT", raising=False)
    parsed = _good_rows()
    # Must NOT raise / skip / write.
    _helper.regenerate_if_env(
        "smoke", "0.1.0", parsed, snapshot_dir=tmp_path
    )
    written = _helper.snapshot_path(
        "smoke", "0.1.0", snapshot_dir=tmp_path
    )
    assert not written.exists()


# ---------------------------------------------------------------------------
# Phase 3 audit gap E — impossible-values detector smoke cases.
# ---------------------------------------------------------------------------


def _impossible_value_rules(diffs: List[Dict[str, Any]]) -> List[str]:
    """Return the ``rule`` field of every impossible_value diff row."""
    return [d.get("rule") for d in diffs if d.get("kind") == "impossible_value"]


def test_impossible_value_negative_outside_allowed_category():
    """value < 0 fails for category=fuel (not in allow-list)."""
    parsed = [{"category": "fuel", "value": -0.5, "unit": "kgCO2e/kWh"}]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert "negative_value_outside_allowed_categories" in _impossible_value_rules(diffs)


def test_impossible_value_negative_allowed_for_sequestration():
    """value < 0 is OK when category is sequestration / land-use / biogenic."""
    for cat in (
        "forestry-and-land-use",
        "sequestration",
        "biogenic-removal",
    ):
        parsed = [{"category": cat, "value": -1.5, "unit": "kgCO2e/ha"}]
        diffs = _helper.diff_impossible_values(parsed, golden=[])
        # Must NOT flag negative_value violation.
        assert "negative_value_outside_allowed_categories" not in _impossible_value_rules(
            diffs
        ), f"unexpected violation for category={cat!r}"


def test_impossible_value_zero_outside_cbam_currency_pair():
    """value == 0 fails for non-CBAM categories regardless of unit."""
    parsed = [{"category": "fuel", "value": 0.0, "unit": "kgCO2e/kWh"}]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert "zero_value_outside_allowed_categories" in _impossible_value_rules(diffs)


def test_impossible_value_zero_allowed_for_cbam_currency_unit():
    """value == 0 is OK for category=cbam_default with currency unit."""
    for unit in ("kg/usd", "kgCO2e/eur", "kg/USD"):
        parsed = [{"category": "cbam_default", "value": 0, "unit": unit}]
        diffs = _helper.diff_impossible_values(parsed, golden=[])
        assert "zero_value_outside_allowed_categories" not in _impossible_value_rules(
            diffs
        ), f"unexpected zero-violation for unit={unit!r}"


def test_impossible_value_zero_cbam_with_non_currency_unit_still_flags():
    """Even cbam_default rows must use currency units when value==0."""
    parsed = [{"category": "cbam_default", "value": 0, "unit": "kgCO2e/kWh"}]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert "zero_value_outside_allowed_categories" in _impossible_value_rules(diffs)


def test_impossible_value_gwp_horizon_outside_allowed_set():
    """gwp_horizon must be one of {20, 100, 500}."""
    parsed = [
        {"category": "fuel", "value": 0.5, "unit": "kgCO2e/kWh", "gwp_horizon": 50},
    ]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert "gwp_horizon_outside_allowed_set" in _impossible_value_rules(diffs)


def test_impossible_value_gwp_horizon_accepts_canonical_set():
    """gwp_horizon=20/100/500 is silent."""
    for h in (20, 100, 500, "100"):
        parsed = [
            {
                "category": "fuel",
                "value": 0.5,
                "unit": "kgCO2e/kWh",
                "gwp_horizon": h,
            }
        ]
        diffs = _helper.diff_impossible_values(parsed, golden=[])
        assert "gwp_horizon_outside_allowed_set" not in _impossible_value_rules(diffs)


def test_impossible_value_vintage_end_before_start():
    """vintage_end < vintage_start is a violation."""
    parsed = [
        {
            "category": "fuel",
            "value": 0.5,
            "unit": "kgCO2e/kWh",
            "vintage_start": "2024-06-01",
            "vintage_end": "2024-01-01",
        }
    ]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert "vintage_end_before_vintage_start" in _impossible_value_rules(diffs)


def test_impossible_value_confidence_outside_unit_interval():
    """confidence < 0 or > 1 is a violation."""
    for cf in (-0.1, 1.1, 5):
        parsed = [
            {
                "category": "fuel",
                "value": 0.5,
                "unit": "kgCO2e/kWh",
                "confidence": cf,
            }
        ]
        diffs = _helper.diff_impossible_values(parsed, golden=[])
        assert "confidence_outside_unit_interval" in _impossible_value_rules(diffs)


def test_impossible_value_confidence_inside_unit_interval_silent():
    """confidence in [0, 1] does not trigger the rule."""
    for cf in (0.0, 0.5, 1.0):
        parsed = [
            {
                "category": "fuel",
                "value": 0.5,
                "unit": "kgCO2e/kWh",
                "confidence": cf,
            }
        ]
        diffs = _helper.diff_impossible_values(parsed, golden=[])
        assert "confidence_outside_unit_interval" not in _impossible_value_rules(diffs)


def test_impossible_value_clean_row_emits_no_diffs():
    """A row that satisfies every rule emits zero impossible_value diffs."""
    parsed = [
        {
            "category": "fuel",
            "value": 0.5,
            "unit": "kgCO2e/kWh",
            "gwp_horizon": 100,
            "vintage_start": "2024-01-01",
            "vintage_end": "2024-12-31",
            "confidence": 0.9,
        }
    ]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert diffs == []


def test_impossible_value_diff_row_shape_carries_kind_marker():
    """Every flagged violation emits kind='impossible_value'."""
    parsed = [{"category": "fuel", "value": -1, "unit": "kgCO2e/kWh"}]
    diffs = _helper.diff_impossible_values(parsed, golden=[])
    assert all(d.get("kind") == "impossible_value" for d in diffs)
    assert all("rule" in d for d in diffs)
    assert all("row_index" in d for d in diffs)
