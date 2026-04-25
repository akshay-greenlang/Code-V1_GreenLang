# -*- coding: utf-8 -*-
"""Parser-snapshot tests for the v0.1 Alpha source list (CTO doc §19.1).

Wave D / TaskCreate #7-#12 — "Lock 6 alpha-source vintages".

For each of the six sources listed in CTO doc §19.1 as in scope for the
v0.1 alpha launch, this module:

1. Loads a canonical raw fixture from
   ``tests/factors/fixtures/{source_id}/{vintage}/raw.json``.
2. Resolves the parser via ``parser_module``/``parser_function`` from
   ``greenlang/factors/data/source_registry.yaml`` (the same path the
   ETL uses in production).
3. Invokes the parser on the raw fixture.
4. Normalises **all** wall-clock timestamp fields
   (``created_at``, ``updated_at``) to a fixed sentinel so the snapshot
   is reproducible across CI runs and developer machines.
5. Asserts byte-equality against
   ``tests/factors/fixtures/{source_id}/{vintage}/expected.json``.

When a publisher (DESNZ, EPA, eGRID, CEA, EU Commission, IPCC) revises
their column shape, the snapshot test fails loudly with a clear diff
hint — this is the "parser drift" guard called out in CTO doc §19.1
risk list.

To regenerate snapshots after an *intended* change::

    pytest tests/factors/v0_1_alpha/test_alpha_source_snapshots.py \
        --update-source-snapshots

The flag is wired in this module's local ``pytest_addoption`` /
``pytest_configure`` hooks; it is opt-in and never set in CI.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = (
    REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"
)
FIXTURES_ROOT = REPO_ROOT / "tests" / "factors" / "fixtures"

# Six alpha sources, paired with the vintage label used in the fixture
# directory layout. The vintage labels match the ``source_version`` in
# ``source_registry.yaml`` exactly.
ALPHA_SOURCES = [
    ("ipcc_2006_nggi", "2019.1"),
    ("desnz_ghg_conversion", "2024.1"),
    ("epa_hub", "2024.1"),
    ("egrid", "2022.1"),
    ("india_cea_co2_baseline", "20.0"),
    ("cbam_default_values", "2024.1"),
]

# Fields that carry wall-clock timestamps and MUST be normalised before
# the snapshot comparison. Parsers stamp these on every record via
# ``_stamp(...)`` / ``_now_iso()``.
_TIMESTAMP_FIELDS = ("created_at", "updated_at")
# Sentinel value used in the snapshot. Anything sortable + ISO-8601
# would do; this string is chosen to be obviously synthetic so a human
# reading the fixture knows it was normalised.
_TS_SENTINEL = "1970-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# CLI: --update-source-snapshots
# ---------------------------------------------------------------------------
# Defined in tests/factors/v0_1_alpha/conftest.py (pytest_addoption only
# fires from conftest.py, not from inside a test module).


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry() -> Dict[str, Dict[str, Any]]:
    """Return ``{source_id: source_dict}`` from the YAML registry."""
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {s["source_id"]: s for s in raw["sources"]}


def _resolve_parser(source: Dict[str, Any]):
    """Import and return the registry-pinned parser callable."""
    mod_name = source["parser_module"]
    fn_name = source["parser_function"]
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def _coerce_to_dict(rec: Any) -> Dict[str, Any]:
    """Normalise a parser output record to a plain dict.

    Some parsers (notably ``india_cea.parse_india_cea_rows``) return
    ``EmissionFactorRecord`` dataclass instances; others return plain
    dicts. We coerce both into a JSON-serialisable dict.
    """
    if isinstance(rec, dict):
        return rec
    # Prefer an explicit ``to_dict`` if present (it handles enums + dates).
    if hasattr(rec, "to_dict") and callable(rec.to_dict):
        return rec.to_dict()
    if is_dataclass(rec):
        return asdict(rec)
    raise TypeError(
        f"Cannot coerce parser record of type {type(rec).__name__} to dict"
    )


def _normalise_timestamps(obj: Any) -> Any:
    """Recursively replace timestamp fields with ``_TS_SENTINEL``.

    Parsers stamp ``created_at`` / ``updated_at`` with
    ``datetime.now(timezone.utc)``, which makes byte-equality
    impossible without normalisation.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _TIMESTAMP_FIELDS and isinstance(v, str):
                out[k] = _TS_SENTINEL
            else:
                out[k] = _normalise_timestamps(v)
        return out
    if isinstance(obj, list):
        return [_normalise_timestamps(x) for x in obj]
    return obj


def _serialise(records: List[Any]) -> str:
    """Serialise a list of records to a deterministic JSON string."""
    payload = [_coerce_to_dict(r) for r in records]
    payload = _normalise_timestamps(payload)
    # ``sort_keys=True`` + 2-space indent + trailing newline -> stable.
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _fixture_paths(source_id: str, vintage: str) -> tuple[Path, Path]:
    base = FIXTURES_ROOT / source_id / vintage
    return base / "raw.json", base / "expected.json"


def _run_parser_on_raw(
    source_id: str, parser, raw: Dict[str, Any]
) -> List[Any]:
    """Invoke a parser, dispatching on its expected input shape.

    Most alpha parsers take the raw payload dict directly. The CEA
    parser is the lone exception — it takes an iterable of pre-extracted
    rows and a ``default_source_year`` keyword.
    """
    if source_id == "india_cea_co2_baseline":
        return parser(
            raw["rows"], default_source_year=raw["metadata"].get(
                "default_source_year", 2024
            )
        )
    return parser(raw)


# ---------------------------------------------------------------------------
# Pre-test snapshot generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry() -> Dict[str, Dict[str, Any]]:
    return _load_registry()


def _ensure_expected_exists(
    source_id: str, vintage: str, registry: Dict[str, Dict[str, Any]]
) -> Path:
    """Generate ``expected.json`` from ``raw.json`` if it is missing.

    On first run, raw.json fixtures exist but expected.json does not
    (this is the "generate by running the parser" step in the brief).
    Subsequent runs assert byte-equality.
    """
    raw_path, expected_path = _fixture_paths(source_id, vintage)
    if not raw_path.exists():
        pytest.fail(
            f"Missing raw fixture for {source_id} v{vintage}: {raw_path}. "
            "Wave D / TaskCreate #7-12 should have committed these."
        )
    if expected_path.exists():
        return expected_path

    parser = _resolve_parser(registry[source_id])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = _run_parser_on_raw(source_id, parser, raw)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(_serialise(records), encoding="utf-8")
    return expected_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.alpha_v0_1_acceptance
@pytest.mark.parametrize(
    ("source_id", "vintage"),
    ALPHA_SOURCES,
    ids=[f"{sid}-v{v}" for sid, v in ALPHA_SOURCES],
)
def test_alpha_source_parser_snapshot(
    source_id: str,
    vintage: str,
    registry: Dict[str, Dict[str, Any]],
    request: pytest.FixtureRequest,
) -> None:
    """Assert parser output matches the saved snapshot, byte-for-byte.

    On drift, the assertion message points the developer to the
    regenerate command. Drift typically indicates the upstream
    publisher revised their column shape (e.g. DESNZ rename
    ``co2_factor`` -> ``co2e_factor``); the methodology lead must
    review before the snapshot is regenerated.
    """
    assert source_id in registry, (
        f"source_id {source_id!r} missing from source_registry.yaml"
    )

    expected_path = _ensure_expected_exists(source_id, vintage, registry)
    raw_path, _ = _fixture_paths(source_id, vintage)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    parser = _resolve_parser(registry[source_id])
    records = _run_parser_on_raw(source_id, parser, raw)
    actual = _serialise(records)

    if request.config.getoption("--update-source-snapshots"):
        expected_path.write_text(actual, encoding="utf-8")
        return

    expected = expected_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"\nParser drift detected for {source_id} v{vintage}.\n"
        f"  raw fixture:      {raw_path}\n"
        f"  expected snapshot: {expected_path}\n"
        f"\n"
        f"If the change is intentional (parser update reviewed by\n"
        f"methodology lead), regenerate via:\n"
        f"    pytest tests/factors/v0_1_alpha/test_alpha_source_snapshots.py \\\n"
        f"        --update-source-snapshots\n"
        f"\n"
        f"Otherwise the upstream publisher likely revised their column\n"
        f"shape — open a methodology ticket before regenerating.\n"
    )


@pytest.mark.alpha_v0_1_acceptance
def test_all_six_alpha_sources_have_status(registry: Dict[str, Dict[str, Any]]) -> None:
    """Sanity: every alpha source carries the Wave D vintage-audit fields."""
    valid_status = {"locked", "preview", "update_pending"}
    for source_id, _vintage in ALPHA_SOURCES:
        src = registry[source_id]
        assert src.get("alpha_v0_1") is True, (
            f"{source_id}: missing alpha_v0_1 flag in registry"
        )
        status = src.get("alpha_v0_1_status")
        assert status in valid_status, (
            f"{source_id}: alpha_v0_1_status={status!r} not in {valid_status}"
        )
        assert "alpha_v0_1_vintage_target" in src, (
            f"{source_id}: missing alpha_v0_1_vintage_target field"
        )
        # If status == preview the methodology exception path MUST exist
        # on disk — otherwise launch is unsigned.
        if status == "preview":
            ex = src.get("alpha_v0_1_methodology_exception")
            assert ex, (
                f"{source_id}: status=preview requires "
                f"alpha_v0_1_methodology_exception path"
            )
            ex_path = REPO_ROOT / ex
            assert ex_path.exists(), (
                f"{source_id}: methodology exception file missing at {ex_path}"
            )
        elif status == "locked":
            # Locked sources MUST set the field to None / null explicitly,
            # not just omit it — keeps the registry self-documenting.
            assert (
                src.get("alpha_v0_1_methodology_exception") is None
            ), (
                f"{source_id}: status=locked must set "
                f"alpha_v0_1_methodology_exception: null"
            )
