# -*- coding: utf-8 -*-
"""Example 02 — Full resolution + explain cascade.

Submits a ResolutionRequest and prints the chosen factor with its
7-step derivation chain + alternates.  Requires Pro+ tier.
"""
from __future__ import annotations

import json
import os

from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.models import ResolutionRequest


def main() -> None:
    base_url = os.environ.get("GREENLANG_FACTORS_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("GREENLANG_FACTORS_API_KEY")

    request = ResolutionRequest(
        activity="diesel combustion stationary",
        method_profile="corporate_scope1",
        jurisdiction="US",
        reporting_date="2026-06-01",
    )

    with FactorsClient(base_url=base_url, api_key=api_key) as client:
        resolved = client.resolve(request, alternates=5)

    print("Chosen factor:", resolved.chosen_factor_id)
    print("Fallback rank:", resolved.fallback_rank, "/", 7)
    print("Step label:  ", resolved.step_label)
    print("Why chosen:  ", resolved.why_chosen)
    print("Method:      ", resolved.method_profile)
    print("Edition:     ", resolved.edition_id)

    if resolved.quality_score:
        print(f"Quality:     {resolved.quality_score.overall_score} ({resolved.quality_score.rating})")

    if resolved.gas_breakdown:
        gb = resolved.gas_breakdown
        print(
            f"Gas breakdown: CO2={gb.CO2} CH4={gb.CH4} N2O={gb.N2O} "
            f"(CH4_gwp={gb.ch4_gwp}, N2O_gwp={gb.n2o_gwp})"
        )

    print(f"\nAlternates considered ({len(resolved.alternates)}):")
    for a in resolved.alternates:
        print("  -", a.get("factor_id"), "score=", a.get("score"))

    if resolved.assumptions:
        print("\nAssumptions:")
        for note in resolved.assumptions:
            print("  *", note)

    print("\nRaw explain payload:")
    print(json.dumps(resolved.explain, indent=2, default=str))


if __name__ == "__main__":
    main()
