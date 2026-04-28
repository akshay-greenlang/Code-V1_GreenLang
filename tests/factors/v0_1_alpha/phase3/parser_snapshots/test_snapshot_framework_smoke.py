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
