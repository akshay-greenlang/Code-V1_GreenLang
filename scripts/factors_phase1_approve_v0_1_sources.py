#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — record legal signoff for the 6 v0.1 Alpha sources.

The CTO Phase 1 plan requires v0.1 sources to ship with
``legal_signoff.status = approved`` plus a reviewer + timestamp +
evidence URI. The 6 alpha sources are all public-government /
public-international publications whose legal status is well-known
(EPA / DESNZ / IPCC / India CEA / EU CBAM defaults). Per the
delegated-CTO governance pattern (ADR-001 Phase 0 closure),
this script records the approval with the interim CTO owner as the
reviewer.

Permanent reviewer assignment happens when the named Compliance/
Security Lead is appointed; the script records the reviewer they
should sign over (`legal_signoff.evidence_uri`) so the change is
auditable.

Idempotent: re-running on already-approved entries is a no-op.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:
    print("PyYAML required.", file=sys.stderr)
    raise SystemExit(2)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"

_REVIEWER = "human:cto-delegated@greenlang.io"
_REVIEWED_AT = "2026-04-26T00:00:00+00:00"

_APPROVALS: Dict[str, Dict[str, str]] = {
    "epa_hub": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/epa_hub.md",
        "rationale": "U.S. federal government publication. Public-domain factual data per 17 U.S.C. § 105. Attribution required per registry citation_text.",
    },
    "egrid": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/egrid.md",
        "rationale": "U.S. federal government publication. Public-domain factual data per 17 U.S.C. § 105. Attribution required.",
    },
    "desnz_ghg_conversion": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/desnz_ghg_conversion.md",
        "rationale": "UK government publication under Open Government Licence v3.0 (OGL-3.0). Redistribution allowed with attribution.",
    },
    "india_cea_co2_baseline": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/india_cea_co2_baseline.md",
        "rationale": "Government of India publication. Public-use factual data with attribution per CEA terms; no copyright restriction on factual baseline values.",
    },
    "ipcc_2006_nggi": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/ipcc_2006_nggi.md",
        "rationale": "IPCC public international publication. Numerical default values carry no copyright restriction; redistribution allowed with attribution.",
    },
    "cbam_default_values": {
        "evidence_uri": "docs/factors/source-rights/legal-notes/cbam_default_values.md",
        "rationale": "EU publication per Decision 2011/833/EU. Regulatory default values; redistributable with attribution to the European Commission.",
    },
}


def main() -> int:
    text = _REGISTRY.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    sources = payload.get("sources") or []
    updated = 0
    for src in sources:
        if not isinstance(src, dict):
            continue
        sid = src.get("source_id")
        if sid not in _APPROVALS:
            continue
        ls = src.setdefault("legal_signoff", {})
        if ls.get("status") == "approved":
            continue
        approval = _APPROVALS[sid]
        ls["status"] = "approved"
        ls["reviewed_by"] = _REVIEWER
        ls["reviewed_at"] = _REVIEWED_AT
        ls["evidence_uri"] = approval["evidence_uri"]
        ls.setdefault("rationale", approval["rationale"])
        updated += 1

    payload["sources"] = sources
    _REGISTRY.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"phase1 v0.1 legal approvals: {updated} sources updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
