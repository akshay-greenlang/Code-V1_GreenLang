#!/usr/bin/env python3
"""Offline smoke eval for match pipeline (M5 starter)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository
from greenlang.factors.matching.pipeline import MatchRequest, run_match


def main() -> int:
    gold_path = REPO / "tests" / "factors" / "fixtures" / "gold_eval_smoke.json"
    cases = json.loads(gold_path.read_text(encoding="utf-8"))
    db = EmissionFactorDatabase(enable_cache=False)
    repo = MemoryFactorCatalogRepository("builtin-v1.0.0", "eval", db)
    edition = repo.get_default_edition_id()
    hits = 0
    for c in cases:
        req = MatchRequest(
            activity_description=c["activity"],
            geography=c.get("geography"),
            limit=5,
        )
        out = run_match(repo, edition, req)
        top = out[0]["factor_id"] if out else ""
        rec = repo.get_factor(edition, top) if top else None
        ok = bool(rec and rec.fuel_type.lower() == c["expected_fuel_type"])
        hits += int(ok)
        print(c["id"], "ok" if ok else "miss", top)
    print("scorecard", {"cases": len(cases), "precision_at_1": hits / max(1, len(cases))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
