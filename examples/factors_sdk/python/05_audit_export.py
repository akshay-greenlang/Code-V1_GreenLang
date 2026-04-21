# -*- coding: utf-8 -*-
"""Example 05 — Enterprise audit bundle export.

Pulls a factor's full audit bundle (provenance chain, SHA-256 payload
hash, license info, reviewer decision) and writes it to disk for
archival.  Requires Enterprise tier.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from greenlang.factors.sdk.python import FactorsClient, TierError


def main(factor_id: str = "EF:US:diesel:2024:v1") -> int:
    base_url = os.environ.get("GREENLANG_FACTORS_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("GREENLANG_FACTORS_API_KEY")
    out_dir = Path(os.environ.get("AUDIT_OUT_DIR", "./audit_bundles"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with FactorsClient(base_url=base_url, api_key=api_key) as client:
        try:
            bundle = client.audit_bundle(factor_id)
        except TierError as exc:
            print("Audit bundle requires Enterprise tier:", exc, file=sys.stderr)
            return 2

        # Also pull a cross-edition diff for context.
        editions = client.list_editions(include_pending=False)
        if len(editions) >= 2:
            left, right = editions[-1].edition_id, editions[0].edition_id
            diff = client.diff(factor_id, left, right)
            diff_path = out_dir / f"{factor_id.replace(':', '_')}_diff.json"
            diff_path.write_text(
                json.dumps(diff.model_dump(exclude_none=True), indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Wrote diff: {diff_path}")

        bundle_path = out_dir / f"{factor_id.replace(':', '_')}_audit.json"
        bundle_path.write_text(
            json.dumps(bundle.model_dump(exclude_none=True), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote audit bundle: {bundle_path}")
        print(f"  payload_sha256: {bundle.payload_sha256}")
        print(f"  content_hash:   {bundle.content_hash}")
    return 0


if __name__ == "__main__":
    factor = sys.argv[1] if len(sys.argv) > 1 else "EF:US:diesel:2024:v1"
    raise SystemExit(main(factor))
