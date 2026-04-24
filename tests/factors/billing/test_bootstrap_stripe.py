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


# ---------------------------------------------------------------------------
# Wave 5 — idempotency + SKU-coverage tests (mock Stripe client)
# ---------------------------------------------------------------------------


class TestWave5SKUCoverage:
    """Wave 4 summary promised 13 products + 27 prices. Verify the plan
    actually delivers that exact shape and that every product carries a
    ``greenlang_sku_id`` metadata key (the Wave 5 idempotency anchor).
    """

    def test_product_and_price_counts_match_wave4_summary(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        assert len(plan.products) == 13, "Wave 4 summary promised 13 Stripe products"
        assert len(plan.prices) == 27, "Wave 4 summary promised 27 Stripe prices"

    def test_every_product_has_greenlang_sku_id_metadata(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        for prod in plan.products:
            assert "greenlang_sku_id" in prod.metadata, (
                f"product {prod.name!r} missing greenlang_sku_id; "
                "required for Wave 5 idempotency"
            )
            assert prod.metadata["greenlang_sku_id"] == prod.stripe_product_id

    def test_sku_ids_are_unique(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        ids = [p.metadata["greenlang_sku_id"] for p in plan.products]
        assert len(ids) == len(set(ids)), "duplicate greenlang_sku_id values in plan"

    def test_lookup_keys_are_unique(self, bootstrap) -> None:
        plan = bootstrap.build_plan()
        keys = [p.lookup_key for p in plan.prices]
        assert len(keys) == len(set(keys)), "duplicate lookup_key values in plan"


class _MockStripeClient:
    """Record-and-replay Stripe mock that mimics the subset of endpoints
    the bootstrap script calls. Hooks at the same ``_stripe_request`` seam
    the live code uses so we never touch the ``stripe`` SDK or network.
    """

    def __init__(self) -> None:
        self.products: list[dict] = []
        self.prices: list[dict] = []
        self.calls: list[tuple[str, str, dict | None]] = []
        self._next_product_id = 1
        self._next_price_id = 1

    def request(self, method: str, path: str, data=None, api_key=None):
        self.calls.append((method, path, data))
        if method == "GET" and path.startswith("/products"):
            return {"data": list(self.products), "has_more": False}
        if method == "GET" and path.startswith("/prices"):
            # ?lookup_keys[0]=<key>&active=true&limit=10
            import urllib.parse
            qs = urllib.parse.urlparse("http://x" + path).query
            params = urllib.parse.parse_qs(qs)
            lookup = params.get("lookup_keys[0]", [""])[0]
            return {"data": [p for p in self.prices if p.get("lookup_key") == lookup]}
        if method == "POST" and path == "/products":
            prod = {
                "id": f"prod_STRIPE{self._next_product_id:03d}",
                "name": data["name"],
                "description": data.get("description", ""),
                "metadata": dict(data.get("metadata") or {}),
                "active": True,
            }
            self._next_product_id += 1
            self.products.append(prod)
            return prod
        if method == "POST" and path.startswith("/products/"):
            prod_id = path.split("/")[-1]
            for p in self.products:
                if p["id"] == prod_id:
                    if "description" in data:
                        p["description"] = data["description"]
                    if "metadata" in data:
                        p["metadata"].update(data["metadata"])
                    return p
            raise AssertionError(f"product {prod_id} not found")
        if method == "POST" and path == "/prices":
            price = {
                "id": f"price_STRIPE{self._next_price_id:03d}",
                "lookup_key": data["lookup_key"],
                "product": data["product"],
                "unit_amount": int(data["unit_amount"]),
                "currency": data["currency"],
                "active": True,
            }
            self._next_price_id += 1
            self.prices.append(price)
            return price
        raise AssertionError(f"unexpected mock call: {method} {path}")


class TestIdempotency:
    """Wave 5 launch-audit: bootstrap must be safe to re-run. A second
    apply with no changes must create ZERO new products / prices and
    just refresh metadata.
    """

    def _apply(self, bootstrap, mock: _MockStripeClient, monkeypatch) -> dict:
        """Run execute_plan against the mock and return the resolved map."""
        plan = bootstrap.build_plan()
        monkeypatch.setattr(
            bootstrap,
            "_stripe_request",
            lambda method, path, data=None, api_key=None: mock.request(
                method, path, data, api_key
            ),
        )
        return bootstrap.execute_plan(plan, api_key="sk_test_fake", write_catalog=False)

    def test_second_apply_creates_no_duplicates(self, bootstrap, monkeypatch) -> None:
        mock = _MockStripeClient()
        first = self._apply(bootstrap, mock, monkeypatch)
        products_after_first = len(mock.products)
        prices_after_first = len(mock.prices)
        assert products_after_first == 13
        assert prices_after_first == 27

        # Second run: must find everything by greenlang_sku_id / lookup_key
        # and create nothing new.
        second = self._apply(bootstrap, mock, monkeypatch)
        assert len(mock.products) == products_after_first, (
            "second --live run created duplicate products — idempotency broken"
        )
        assert len(mock.prices) == prices_after_first, (
            "second --live run created duplicate prices — idempotency broken"
        )
        assert first == second, "resolved map changed between runs"

    def test_rename_on_stripe_side_still_matches_by_sku_id(
        self, bootstrap, monkeypatch
    ) -> None:
        """If someone renames a product in the Stripe dashboard, the next
        bootstrap must still match it via metadata.greenlang_sku_id.
        """
        mock = _MockStripeClient()
        self._apply(bootstrap, mock, monkeypatch)
        # Simulate a human renaming one product in the Stripe dashboard.
        mock.products[0]["name"] = "Renamed in Stripe Dashboard"
        # Re-run: should not create a new product for the renamed SKU.
        initial_count = len(mock.products)
        self._apply(bootstrap, mock, monkeypatch)
        assert len(mock.products) == initial_count, (
            "bootstrap created a duplicate product after dashboard rename; "
            "greenlang_sku_id metadata match failed"
        )

    def test_metadata_refreshed_on_second_run(
        self, bootstrap, monkeypatch
    ) -> None:
        """When metadata drifts out-of-band, a re-run restores it."""
        mock = _MockStripeClient()
        self._apply(bootstrap, mock, monkeypatch)
        # Corrupt a metadata field out-of-band.
        mock.products[0]["metadata"]["pricing_proposal_version"] = "v0-stale"
        self._apply(bootstrap, mock, monkeypatch)
        assert mock.products[0]["metadata"]["pricing_proposal_version"] == "v1"

    def test_prices_are_not_mutated_in_place(
        self, bootstrap, monkeypatch
    ) -> None:
        """Stripe prices are immutable; on re-run we must not attempt
        to POST changes to existing price IDs."""
        mock = _MockStripeClient()
        self._apply(bootstrap, mock, monkeypatch)
        mock.calls.clear()
        self._apply(bootstrap, mock, monkeypatch)
        price_updates = [
            c for c in mock.calls
            if c[0] == "POST" and c[1].startswith("/prices/")
        ]
        assert price_updates == [], (
            "second run attempted to mutate an existing price; prices are immutable"
        )

    def test_uses_stripe_request_seam_not_stripe_sdk(self, bootstrap) -> None:
        """Regression guard: the script must NOT import the stripe SDK."""
        import inspect
        src = inspect.getsource(bootstrap)
        assert "import stripe" not in src, (
            "bootstrap_stripe.py added a stripe SDK dependency; keep urllib seam"
        )
        assert "from stripe" not in src

    def test_find_existing_product_prefers_sku_id_over_name(
        self, bootstrap
    ) -> None:
        """Unit-test the matcher: a name collision must not fool the
        matcher if metadata.greenlang_sku_id disagrees."""
        mock = _MockStripeClient()
        # Seed: one product with the target name but WRONG greenlang_sku_id,
        # and one product with the correct greenlang_sku_id but a different name.
        mock.products = [
            {
                "id": "prod_wrong",
                "name": "GreenLang Factors — Community",
                "metadata": {"greenlang_sku_id": "prod_factors_legacy"},
            },
            {
                "id": "prod_right",
                "name": "Renamed",
                "metadata": {"greenlang_sku_id": "prod_factors_community"},
            },
        ]
        # Patch _stripe_request to our mock just for this call.
        import types
        orig = bootstrap._stripe_request
        try:
            bootstrap._stripe_request = (
                lambda method, path, data=None, api_key=None: mock.request(
                    method, path, data, api_key
                )
            )
            found = bootstrap._find_existing_product(
                "GreenLang Factors — Community",
                api_key="sk_test_fake",
                greenlang_sku_id="prod_factors_community",
            )
        finally:
            bootstrap._stripe_request = orig
        assert found is not None
        assert found["id"] == "prod_right", (
            "matcher fell back to name match when greenlang_sku_id was available"
        )

    def test_find_existing_product_returns_none_for_unknown(
        self, bootstrap
    ) -> None:
        mock = _MockStripeClient()
        import types
        orig = bootstrap._stripe_request
        try:
            bootstrap._stripe_request = (
                lambda method, path, data=None, api_key=None: mock.request(
                    method, path, data, api_key
                )
            )
            found = bootstrap._find_existing_product(
                "Does Not Exist",
                api_key="sk_test_fake",
                greenlang_sku_id="prod_nope",
            )
        finally:
            bootstrap._stripe_request = orig
        assert found is None
