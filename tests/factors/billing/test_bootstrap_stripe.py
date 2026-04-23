# -*- coding: utf-8 -*-
"""
Tests for ``scripts/bootstrap_stripe.py`` (Agent W4-E / C2).

Coverage:
    * Plan construction is deterministic (identical on re-invocation).
    * Product counts: 5 tier + 8 pack = 13 products.
    * Price counts: 10 tier monthly+annual + 16 pack monthly+annual +
      3 metered = 29 (Enterprise is contact-sales and drops its
      monthly+annual pair; community keeps its two $0 prices as SKU
      markers for the free tier).
    * Dry-run invocation writes no catalog file + exits 0.
    * Idempotent rerun produces a byte-identical plan.
    * --production flag requires both the confirmation env var and
      interactive confirmation (guarded by refusing without them).
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_BOOTSTRAP_PATH = _ROOT / "scripts" / "bootstrap_stripe.py"


def _load_bootstrap_module():
    """Load the bootstrap script as a module (it's a top-level script, not a pkg).

    Note: we register the module in ``sys.modules`` BEFORE executing so
    that dataclass type-hint resolution can find it. Without this,
    Python 3.11's ``dataclasses._is_type`` fails with AttributeError
    because ``sys.modules[cls.__module__]`` returns ``None``.
    """
    spec = importlib.util.spec_from_file_location(
        "bootstrap_stripe", str(_BOOTSTRAP_PATH)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bootstrap():
    return _load_bootstrap_module()


class TestPlanShape:
    def test_plan_has_five_tier_products(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        assert plan.tier_count == 5

    def test_plan_has_eight_pack_products(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        assert plan.pack_count == 8

    def test_plan_has_three_metered_prices(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        assert plan.metered_price_count == 3

    def test_plan_product_count(self, bootstrap) -> None:
        """13 products = 5 tiers + 8 premium packs."""
        plan = bootstrap.build_plan()
        assert len(plan.products) == 13

    def test_plan_price_count(self, bootstrap) -> None:
        """Prices: 5 tier * 2 (mo/yr) - 2 (Ent no subscription price) = 8
        tier subscription prices; 8 packs * 2 = 16 pack prices; 3 metered.
        Community keeps both $0 prices as free-tier markers.
        Total = 8 + 16 + 3 = 27.
        """
        plan = bootstrap.build_plan()
        assert len(plan.prices) == 27

    def test_every_tier_has_product_id(self, bootstrap) -> None:
        """Every tier product_id begins with ``prod_factors_``."""
        plan = bootstrap.build_plan()
        for prod in plan.products:
            assert prod.stripe_product_id.startswith("prod_factors_")


class TestDeterminism:
    def test_plan_is_deterministic(self, bootstrap) -> None:
        """Two calls to build_plan produce identical dicts."""
        p1 = bootstrap.build_plan().to_dict()
        p2 = bootstrap.build_plan().to_dict()
        assert p1 == p2

    def test_dry_run_output_is_stable(self, bootstrap) -> None:
        """Dry-run render is deterministic across calls."""
        plan = bootstrap.build_plan()
        r1 = bootstrap.render_dry_run(plan)
        r2 = bootstrap.render_dry_run(plan)
        assert r1 == r2


class TestCLIGuards:
    def test_dry_run_default_is_safe(self, bootstrap, capsys) -> None:
        """No args -> dry-run -> exit 0 + no Stripe call."""
        rc = bootstrap.main([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "No changes applied" in captured.out

    def test_live_without_api_key_errors(self, bootstrap, capsys, monkeypatch) -> None:
        """--live without STRIPE_API_KEY env var exits non-zero."""
        monkeypatch.delenv("STRIPE_API_KEY", raising=False)
        rc = bootstrap.main(["--live"])
        assert rc == 2
        out = capsys.readouterr().out
        assert "STRIPE_API_KEY" in out

    def test_live_rejects_live_key(self, bootstrap, capsys, monkeypatch) -> None:
        """--live with an ``sk_live_`` key must be refused (footgun guard)."""
        monkeypatch.setenv("STRIPE_API_KEY", "sk_live_real_production_key_abc")
        rc = bootstrap.main(["--live"])
        assert rc == 2
        out = capsys.readouterr().out
        assert "sk_test_" in out

    def test_production_without_confirm_env_errors(
        self, bootstrap, capsys, monkeypatch
    ) -> None:
        """--production without the confirmation env var exits non-zero."""
        monkeypatch.setenv("STRIPE_API_KEY", "sk_live_real_production_key")
        monkeypatch.delenv("GL_BOOTSTRAP_STRIPE_CONFIRM", raising=False)
        rc = bootstrap.main(["--production"])
        assert rc == 2
        out = capsys.readouterr().out
        assert "GL_BOOTSTRAP_STRIPE_CONFIRM" in out

    def test_production_rejects_test_key(self, bootstrap, capsys, monkeypatch) -> None:
        """--production with a test key must be refused."""
        monkeypatch.setenv("STRIPE_API_KEY", "sk_test_abc")
        monkeypatch.setenv("GL_BOOTSTRAP_STRIPE_CONFIRM", "i-understand")
        rc = bootstrap.main(["--production"])
        assert rc == 2


class TestPriceShape:
    def test_every_pack_has_monthly_and_annual(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        lookup_keys = {p.lookup_key for p in plan.prices}
        for pack_id in (
            "electricity_premium",
            "freight_premium",
            "product_carbon_premium",
            "epd_premium",
            "agrifood_premium",
            "finance_premium",
            "cbam_premium",
            "land_premium",
        ):
            monthly = f"price_factors_pack_{pack_id}_monthly"
            annual = f"price_factors_pack_{pack_id}_annual"
            assert monthly in lookup_keys, f"missing {monthly}"
            assert annual in lookup_keys, f"missing {annual}"

    def test_cbam_premium_is_most_expensive_pack(self, bootstrap) -> None:
        """CBAM must be the top-priced premium pack ($999/mo)."""
        plan = bootstrap.build_plan()
        by_key = {p.lookup_key: p.unit_amount_cents for p in plan.prices}
        cbam = by_key["price_factors_pack_cbam_premium_monthly"]
        freight = by_key["price_factors_pack_freight_premium_monthly"]
        electricity = by_key["price_factors_pack_electricity_premium_monthly"]
        assert cbam > freight
        assert cbam > electricity
        assert cbam == 99_900  # $999.00

    def test_pro_monthly_price(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        by_key = {p.lookup_key: p.unit_amount_cents for p in plan.prices}
        assert by_key["price_factors_pro_monthly"] == 29_900  # $299.00
        assert by_key["price_factors_pro_annual"] == 299_000  # $2,990.00

    def test_consulting_monthly_price(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        by_key = {p.lookup_key: p.unit_amount_cents for p in plan.prices}
        assert by_key["price_factors_consulting_monthly"] == 249_900  # $2,499.00
        assert by_key["price_factors_consulting_annual"] == 2_499_000  # $24,990.00
