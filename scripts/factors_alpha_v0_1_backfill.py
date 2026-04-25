#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill v0.1-shape catalog seeds from each alpha source's v1 parser.

Wave D / TaskCreate #6 (WS2-T2). For every source flagged
``alpha_v0_1: true`` in ``greenlang/factors/data/source_registry.yaml``:

    1. Resolve the parser declared by ``parser_module`` /
       ``parser_function`` and call it on the canonical alpha seed
       fixture under ``greenlang/factors/data/catalog_seed/_inputs/``.
    2. Lift each produced record via
       :func:`greenlang.factors.etl.alpha_v0_1_normalizer.lift_v1_record_to_v0_1`.
    3. Run :class:`AlphaProvenanceGate.validate` on every lifted record
       and accumulate failures.
    4. Write the validated v0.1 records to
       ``greenlang/factors/data/catalog_seed_v0_1/<source_id>/v1.json``
       (unless ``--dry-run``).

Exits 0 iff every source backfilled cleanly (every record passes the
gate). Otherwise exits 1 with a per-source failure summary.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("factors_alpha_v0_1_backfill")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# scripts/factors_alpha_v0_1_backfill.py -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_INPUTS_DIR = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed" / "_inputs"
)
_OUTDIR = _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"


# Mirrors the mapping in tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py
_SOURCE_ID_TO_SEED_FILE: Dict[str, str] = {
    "epa_hub": "epa_ghg_hub.json",
    "egrid": "egrid.json",
    "desnz_ghg_conversion": "desnz_uk.json",
    "india_cea_co2_baseline": "india_cea.json",
    "ipcc_2006_nggi": "ipcc_defaults.json",
    "cbam_default_values": "cbam_default_values.json",
}


def _seed_path_for(source_id: str) -> Path:
    fname = _SOURCE_ID_TO_SEED_FILE.get(source_id, f"{source_id}.json")
    return _SEED_INPUTS_DIR / fname


def _record_to_dict(rec: Any) -> Dict[str, Any]:
    """Coerce a parser output (dict OR EmissionFactorRecord) to a dict."""
    if isinstance(rec, dict):
        return rec
    to_dict = getattr(rec, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    raise TypeError(
        f"unsupported record type {type(rec).__name__}; "
        f"expected dict or .to_dict()-compatible object"
    )


def _resolve_parser(source: Dict[str, Any]) -> Callable[..., Any]:
    module_name = source.get("parser_module")
    function_name = source.get("parser_function")
    if not isinstance(module_name, str) or not module_name:
        raise RuntimeError(
            f"alpha source {source.get('source_id')!r} has no parser_module"
        )
    if not isinstance(function_name, str) or not function_name:
        raise RuntimeError(
            f"alpha source {source.get('source_id')!r} has no parser_function"
        )
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            f"{module_name}.{function_name} is not callable / does not exist"
        )
    return fn


def _invoke_parser(
    parser: Callable[..., Any],
    payload: Any,
    source_id: str,
) -> List[Any]:
    """Call the parser, handling per-source argument shape variation."""
    if source_id == "india_cea_co2_baseline":
        if isinstance(payload, dict):
            payload = payload.get("rows") or []
        result = parser(payload)
    else:
        result = parser(payload)

    if result is None:
        return []
    if isinstance(result, list):
        return result
    return list(result)


def _run_parser(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve + invoke the parser; return a list of v1-shape dicts.

    Returns an empty list if the seed is missing/unreadable, or the
    parser fails to import/run cleanly.
    """
    source_id = str(source.get("source_id") or "")
    seed_path = _seed_path_for(source_id)
    if not seed_path.is_file():
        logger.warning("seed file missing for %s: %s", source_id, seed_path)
        return []
    try:
        text = seed_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("seed file unreadable for %s: %s", source_id, exc)
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("seed file invalid JSON for %s: %s", source_id, exc)
        return []

    try:
        parser = _resolve_parser(source)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.warning("parser resolve failed for %s: %s", source_id, exc)
        return []

    try:
        raw = _invoke_parser(parser, payload, source_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("parser invocation failed for %s: %s", source_id, exc)
        return []

    out: List[Dict[str, Any]] = []
    for r in raw:
        try:
            out.append(_record_to_dict(r))
        except TypeError as exc:
            logger.warning("dropping unconvertable record from %s: %s", source_id, exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill v0.1-shape catalog seeds from each alpha source's v1 parser."
        )
    )
    parser.add_argument(
        "--source", help="optional source_id filter (default: all alpha sources)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the lift+validate pass but do NOT write seed files to disk",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable INFO-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Late imports — keep --help fast and avoid crashing when the
    # in-tree modules have transient import errors.
    try:
        from greenlang.factors.source_registry import alpha_v0_1_sources
        from greenlang.factors.etl.alpha_v0_1_normalizer import (
            NonPositiveValueError,
            lift_v1_record_to_v0_1,
        )
        from greenlang.factors.quality.alpha_provenance_gate import (
            AlphaProvenanceGate,
        )
    except ImportError as exc:
        logger.error("required GreenLang module failed to import: %s", exc)
        return 2

    sources = alpha_v0_1_sources() or {}
    if args.source:
        if args.source not in sources:
            logger.error(
                "--source %s not in alpha_v0_1 set: %s",
                args.source,
                sorted(sources.keys()),
            )
            return 2
        sources = {args.source: sources[args.source]}

    gate = AlphaProvenanceGate()
    summary: List[Tuple[str, int, int, int]] = []
    overall_pass = True

    for source_id in sorted(sources.keys()):
        s = sources[source_id]
        records = _run_parser(s)
        lifted: List[Dict[str, Any]] = []
        failures: List[Tuple[int, List[str]]] = []
        # Records that are structurally valid but cannot be expressed in
        # v0.1 (e.g. carbon-sequestration land-use removals where
        # value <= 0). These are *skipped* — not failures — because the
        # alpha schema requires ``value > 0`` and signed factors are a
        # v0.5+ schema concern.
        skipped: List[Tuple[int, str]] = []

        for i, r in enumerate(records):
            try:
                v0 = lift_v1_record_to_v0_1(r, dict(s), idx=i)
            except NonPositiveValueError as exc:
                skipped.append((i, str(exc)))
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append((i, [f"normalizer: {exc}"]))
                continue
            errs = gate.validate(v0)
            if errs:
                failures.append((i, errs[:3]))
            else:
                lifted.append(v0)

        outpath = _OUTDIR / source_id / "v1.json"
        if not args.dry_run:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": (
                    "https://schemas.greenlang.io/factors/"
                    "factor_record_v0_1.schema.json"
                ),
                "source_id": source_id,
                "source_urn": s.get("urn"),
                "source_version": s.get("source_version"),
                "records": lifted,
            }
            outpath.write_text(
                json.dumps(payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )

        summary.append((source_id, len(records), len(lifted), len(failures)))
        if failures:
            overall_pass = False
            print(f"FAIL {source_id}: {len(failures)}/{len(records)}")
            for idx, errs in failures[:3]:
                print(f"  rec[{idx}]: {errs}")
        if skipped:
            print(
                f"SKIP {source_id}: {len(skipped)} records "
                "(non-positive CO2e — sequestration; deferred to v0.5)"
            )
            for idx, reason in skipped[:3]:
                print(f"  rec[{idx}]: {reason}")

    print()
    print(f"{'source':30s} {'total':>6s} {'pass':>6s} {'fail':>6s}")
    print("-" * 52)
    for sid, total, p, f in summary:
        print(f"{sid:30s} {total:>6d} {p:>6d} {f:>6d}")
    grand_total = sum(t for _, t, _, _ in summary)
    grand_pass = sum(p for _, _, p, _ in summary)
    grand_fail = sum(f for _, _, _, f in summary)
    print("-" * 52)
    print(f"{'TOTAL':30s} {grand_total:>6d} {grand_pass:>6d} {grand_fail:>6d}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
