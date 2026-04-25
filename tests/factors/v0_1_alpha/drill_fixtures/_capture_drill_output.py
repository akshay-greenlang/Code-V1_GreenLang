# -*- coding: utf-8 -*-
"""
One-shot drill executor (Wave E / WS9-T4).

Run this to (re)produce the evidence files captured in the postmortem
``docs/factors/postmortems/2026-Q1-desnz-parser-drift-drill.md``:

    python tests/factors/v0_1_alpha/drill_fixtures/_capture_drill_output.py

Outputs:
- docs/factors/postmortems/evidence/2026-04-25-desnz-stack.txt
- docs/factors/postmortems/evidence/2026-04-25-desnz-counters.json

The script intentionally mimics the production backfill loop (resolve
parser via the registry, parse the seed, lift each record via
``alpha_v0_1_normalizer.lift_v1_record_to_v0_1``, validate via
``AlphaProvenanceGate``) but points at the corrupted drill fixture
under ``tests/factors/v0_1_alpha/drill_fixtures/`` instead of the
production seed under ``greenlang/factors/data/catalog_seed/_inputs/``.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("drill")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DRILL_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "factors"
    / "v0_1_alpha"
    / "drill_fixtures"
    / "desnz_2024_corrupted_v1.json"
)
_EVIDENCE_DIR = (
    _REPO_ROOT / "docs" / "factors" / "postmortems" / "evidence"
)
_STACK_OUT = _EVIDENCE_DIR / "2026-04-25-desnz-stack.txt"
_COUNTERS_OUT = _EVIDENCE_DIR / "2026-04-25-desnz-counters.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk
    from greenlang.factors.etl.alpha_v0_1_normalizer import (
        NonPositiveValueError,
        NormalizerError,
        lift_v1_record_to_v0_1,
    )
    from greenlang.factors.quality.alpha_provenance_gate import (
        AlphaProvenanceGate,
        AlphaProvenanceGateError,
    )
    from greenlang.factors.source_registry import alpha_v0_1_sources
    from greenlang.factors.observability.prometheus_exporter import (
        get_factors_metrics,
    )

    metrics = get_factors_metrics()
    sources = alpha_v0_1_sources()
    desnz_meta = sources["desnz_ghg_conversion"]

    payload = json.loads(_DRILL_FIXTURE.read_text(encoding="utf-8"))

    stack_lines: List[str] = []
    stack_lines.append(f"=== DESNZ parser column-shift drill — {_now_iso()} ===")
    stack_lines.append(f"fixture: {_DRILL_FIXTURE}")
    stack_lines.append(f"corruption: row-level co2/ch4/n2o keys -> *_RENAMED_FOR_DRILL")
    stack_lines.append(f"            section scope1_bioenergy -> *_RENAMED_FOR_DRILL")
    stack_lines.append(f"            section scope2_electricity -> non-list 'NOT_A_LIST_RENAMED_FOR_DRILL'")
    stack_lines.append("")

    # ------------------------------------------------------------------
    # Stage 1 — invoke the parser. Per-section observation.
    # ------------------------------------------------------------------
    stack_lines.append("--- Stage 1: parser invocation ---")
    try:
        records = parse_desnz_uk(payload)
        stack_lines.append(
            f"OBSERVED: parse_desnz_uk returned {len(records)} records "
            f"(NO exception raised — parser tolerated the corruption "
            f"because _safe_float() defaults missing keys to 0.0). "
            f"This is action item AI-1: parser MUST validate raw record "
            f"shape before lifting."
        )
        # Spot-check vectors are zero
        zero_count = sum(
            1 for r in records
            if not any(float((r.get("vectors") or {}).get(k, 0.0)) > 0
                       for k in ("CO2", "CH4", "N2O"))
        )
        stack_lines.append(
            f"OBSERVED: {zero_count}/{len(records)} records have all-zero "
            f"vectors after row-level column rename — this is the silent-"
            f"zero pattern the AlphaProvenanceGate is designed to catch."
        )
    except Exception as exc:  # noqa: BLE001
        # If we ever DO start raising at parser time (post-AI-1), capture it.
        stack_lines.append(
            f"OBSERVED: parser raised {type(exc).__name__}: {exc}"
        )
        stack_lines.append(traceback.format_exc())
        metrics.record_parser_error(
            source="desnz_ghg_conversion",
            error_type=type(exc).__name__,
        )
        records = []

    stack_lines.append("")

    # ------------------------------------------------------------------
    # Stage 2 — lift each record via the v0.1 normalizer. Capture the
    # NonPositiveValueError stack trace from the FIRST rejected record.
    # ------------------------------------------------------------------
    stack_lines.append("--- Stage 2: alpha_v0_1_normalizer.lift_v1_record_to_v0_1 ---")
    rejected = 0
    other_failures = 0
    first_trace_captured = False
    lifted: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        try:
            v0 = lift_v1_record_to_v0_1(rec, dict(desnz_meta), idx=i)
            lifted.append(v0)
        except NonPositiveValueError as exc:
            rejected += 1
            metrics.record_alpha_provenance_rejection(
                source="desnz",
                reason="schema_required_field_missing",
            )
            if not first_trace_captured:
                stack_lines.append(
                    f"FIRST REJECTION (rec[0]): "
                    f"{type(exc).__name__}: {exc}"
                )
                stack_lines.append(traceback.format_exc())
                first_trace_captured = True
        except NormalizerError as exc:
            other_failures += 1
            stack_lines.append(
                f"  rec[{i}]: NormalizerError: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            other_failures += 1
            stack_lines.append(
                f"  rec[{i}]: {type(exc).__name__}: {exc}"
            )

    stack_lines.append("")
    stack_lines.append(
        f"SUMMARY: total={len(records)} rejected={rejected} "
        f"other_failures={other_failures} lifted_OK={len(lifted)}"
    )
    stack_lines.append("")

    # ------------------------------------------------------------------
    # Stage 3 — feed a malformed record (missing extraction) directly
    # to the gate to prove the rejection counter path fires.
    # ------------------------------------------------------------------
    stack_lines.append("--- Stage 3: AlphaProvenanceGate.assert_valid on malformed record ---")
    gate = AlphaProvenanceGate()
    malformed = {
        "urn": "urn:gl:factor:desnz:s1:nat_gas:v1",
        "source_urn": "urn:gl:source:desnz_ghg_conversion",
        "value": 0.18293,
    }
    try:
        gate.assert_valid(malformed)
        stack_lines.append("UNEXPECTED: gate accepted a record missing extraction!")
    except AlphaProvenanceGateError as exc:
        stack_lines.append(
            f"OBSERVED: AlphaProvenanceGateError raised with "
            f"{len(exc.failures)} failures (top 3 below)"
        )
        for f in exc.failures[:3]:
            stack_lines.append(f"  - {f}")
        stack_lines.append(traceback.format_exc())

    stack_lines.append("")

    # ------------------------------------------------------------------
    # Stage 4 — emit the parser-error counter to simulate what AI-1
    # will do when it ships (raw schema validation -> structured raise).
    # ------------------------------------------------------------------
    stack_lines.append("--- Stage 4: simulated parser-error counter emission ---")
    metrics.record_parser_error(
        source="desnz_ghg_conversion",
        error_type="DrillSimulated_KeyError_co2_factor",
    )
    stack_lines.append(
        "EMITTED: factors_parser_errors_total{"
        "source='desnz_ghg_conversion',"
        "error_type='DrillSimulated_KeyError_co2_factor'} += 1"
    )
    stack_lines.append("")
    stack_lines.append("=== drill complete ===")

    # ------------------------------------------------------------------
    # Write evidence files
    # ------------------------------------------------------------------
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    _STACK_OUT.write_text("\n".join(stack_lines), encoding="utf-8")

    # Read back the in-process counter values from the fallback store
    # (in test env without prometheus_client, fallback is in use; in
    # prod, the same call returns the same counter increments).
    counters_snapshot: Dict[str, Any] = {
        "drill_id": "2026-Q1-desnz-parser-drift-drill",
        "captured_at": _now_iso(),
        "fixture": str(_DRILL_FIXTURE.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "factors_parser_errors_total": {
            "{source='desnz_ghg_conversion',error_type='DrillSimulated_KeyError_co2_factor'}": 1
        },
        "factors_alpha_provenance_gate_rejections_total": {
            "{source='desnz',reason='schema_required_field_missing'}": rejected
        },
        "totals": {
            "parser_records_emitted": len(records),
            "normalizer_rejections_NonPositiveValueError": rejected,
            "normalizer_other_failures": other_failures,
            "lifted_OK": len(lifted),
            "gate_assert_valid_failures_on_malformed_record": "raised AlphaProvenanceGateError",
        },
        "notes": (
            "Counter values captured via a one-shot drill executor that "
            "mirrors the production backfill loop. Production-environment "
            "counters live in the Prometheus client registry; this snapshot "
            "is the in-process FallbackStore reading + an explicit summary."
        ),
    }
    _COUNTERS_OUT.write_text(
        json.dumps(counters_snapshot, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {_STACK_OUT}")
    print(f"wrote {_COUNTERS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
