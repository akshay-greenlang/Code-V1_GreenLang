# -*- coding: utf-8 -*-
"""Example 01 — Basic factor search.

Run:

    export GREENLANG_FACTORS_BASE_URL=https://api.greenlang.io
    export GREENLANG_FACTORS_API_KEY=gl_fac_...
    python 01_basic_search.py
"""
from __future__ import annotations

import os

from greenlang.factors.sdk.python import FactorsClient


def main() -> None:
    base_url = os.environ.get("GREENLANG_FACTORS_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("GREENLANG_FACTORS_API_KEY")

    with FactorsClient(base_url=base_url, api_key=api_key) as client:
        # Quick full-text search:
        hits = client.search("diesel US Scope 1", limit=5)
        print(f"Found {hits.count} / {hits.total_count} factors (edition={hits.edition_id})")
        for f in hits.factors:
            print(
                f"  - {f.factor_id:<40} "
                f"co2e={f.co2e_per_unit} {f.unit}  "
                f"source={f.source.organization if f.source else '?'}"
            )

        # Advanced search with sort + filters:
        adv = client.search_v2(
            "electricity",
            geography="US",
            scope="2",
            dqs_min=80.0,
            sort_by="dqs_score",
            sort_order="desc",
            limit=10,
        )
        print(f"\nAdvanced search: {adv.count} of {adv.total_count}")
        for f in adv.factors:
            print(
                f"  - {f.factor_id:<40} "
                f"dqs={f.data_quality.overall_score if f.data_quality else '?'}"
            )


if __name__ == "__main__":
    main()
