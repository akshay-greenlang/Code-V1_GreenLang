# -*- coding: utf-8 -*-
"""Example 03 — Batch resolution with polling.

Submits 3 resolution requests at once and polls until the server-side
job reaches a terminal state, then prints per-item results.
"""
from __future__ import annotations

import os
from typing import List

from greenlang.factors.sdk.python import FactorsAPIError, FactorsClient
from greenlang.factors.sdk.python.models import ResolutionRequest


def build_requests() -> List[ResolutionRequest]:
    return [
        ResolutionRequest(
            activity="diesel combustion",
            method_profile="corporate_scope1",
            jurisdiction="US",
        ),
        ResolutionRequest(
            activity="gasoline fleet",
            method_profile="corporate_scope1",
            jurisdiction="US",
        ),
        ResolutionRequest(
            activity="electricity purchased",
            method_profile="corporate_scope2_location_based",
            jurisdiction="US-CA",
        ),
    ]


def main() -> None:
    base_url = os.environ.get("GREENLANG_FACTORS_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("GREENLANG_FACTORS_API_KEY")

    with FactorsClient(base_url=base_url, api_key=api_key) as client:
        handle = client.resolve_batch(build_requests())
        print(f"Submitted job: {handle.job_id} ({handle.status})")

        try:
            final = client.wait_for_batch(handle, poll_interval=1.5, timeout=120.0)
        except FactorsAPIError as exc:
            print("Batch failed:", exc)
            return

        print(f"Job {final.job_id} finished: {final.status}")
        if final.results:
            for i, row in enumerate(final.results):
                print(f"  [{i}] factor_id={row.get('chosen_factor_id')} "
                      f"rank={row.get('fallback_rank')}")


if __name__ == "__main__":
    main()
