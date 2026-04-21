# -*- coding: utf-8 -*-
"""Example 04 — Tenant-scoped factor override.

Consulting/Platform tier customers can override a published factor's
emission intensity with their own supplier-specific value.  Every
override carries a justification for audit and a validity window.
"""
from __future__ import annotations

import os

from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.models import Override


def main() -> None:
    base_url = os.environ.get("GREENLANG_FACTORS_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("GREENLANG_FACTORS_API_KEY")
    tenant_id = os.environ.get("GREENLANG_FACTORS_TENANT_ID", "acme-corp")

    override = Override(
        factor_id="EF:US:diesel:2024:v1",
        tenant_id=tenant_id,
        co2e_per_unit=2.55,  # supplier-measured value, lower than published 2.68
        justification="Supplier audit 2026-Q1 (document ref: ACME-FUEL-2026-A11)",
        effective_from="2026-01-01",
        effective_to="2026-12-31",
        metadata={"source": "supplier_audit", "document": "ACME-FUEL-2026-A11"},
    )

    with FactorsClient(base_url=base_url, api_key=api_key) as client:
        saved = client.set_override(override)
        print("Override saved for", saved.factor_id)

        overrides = client.list_overrides(tenant_id=tenant_id)
        print(f"Tenant {tenant_id} has {len(overrides)} override(s):")
        for o in overrides:
            print(f"  - {o.factor_id}: {o.co2e_per_unit} (valid {o.effective_from} - {o.effective_to})")


if __name__ == "__main__":
    main()
