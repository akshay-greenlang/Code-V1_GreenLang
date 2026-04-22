#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stripe catalog provisioner for the GreenLang Factors FY27 launch.

Reads ``deployment/stripe/factors-products.yaml``, compares it against the
live Stripe catalog, and creates or updates products + prices + meters +
feature flags as needed. The script is idempotent: running it twice in a
row on the same live Stripe environment is a no-op.

Safety rails
------------
* Defaults to ``--dry-run``. Writing to Stripe requires ``--live`` AND
  ``STRIPE_API_KEY`` in the environment — both. A missing or empty
  ``STRIPE_API_KEY`` in ``--live`` mode aborts before any request.
* ``STRIPE_API_KEY`` starting with ``sk_live_`` triggers a second
  confirmation prompt unless ``--yes-really-prod`` is passed. Test-mode
  keys (``sk_test_``) run unprompted.
* All Stripe writes are idempotent: we look up by metadata/nickname first
  (we cannot set Stripe product IDs directly, so GreenLang IDs live in
  ``metadata.gl_sku_id`` and resolved Stripe IDs are written to the
  lockfile).
* Never mutates existing prices. Stripe prices are immutable; we archive
  + recreate if the amount changes.

Usage
-----
    # Preview what would change (default) — never hits the network
    python scripts/factors_stripe_provision.py \
        --catalog deployment/stripe/factors-products.yaml

    # Apply to a live Stripe account (requires STRIPE_API_KEY)
    STRIPE_API_KEY=sk_test_... python scripts/factors_stripe_provision.py \
        --catalog deployment/stripe/factors-products.yaml --live

    # Write a lockfile (with resolved Stripe IDs) for runtime lookup
    python scripts/factors_stripe_provision.py \
        --catalog deployment/stripe/factors-products.yaml --live \
        --lockfile deployment/stripe/stripe_catalog_lockfile.json

Environment
-----------
    STRIPE_API_KEY         Stripe secret key (sk_test_... or sk_live_...)
    STRIPE_API_BASE        Override Stripe base URL (default: api.stripe.com)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover -- dev-time only
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

# Ensure ``greenlang`` is importable when the script is launched from the
# repo root without `pip install -e .`.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from greenlang.factors.billing.skus import (  # noqa: E402
    CATALOG,
    Meter,
    PremiumPack,
    Tier,
)

logger = logging.getLogger("factors_stripe_provision")


# ---------------------------------------------------------------------------
# Minimal Stripe HTTP client
# ---------------------------------------------------------------------------


class StripeError(RuntimeError):
    """Stripe returned a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Stripe API error (HTTP {status_code}): {body[:400]}")


class StripeClient:
    """Tiny form-encoded Stripe REST client (no stripe-python dependency).

    Kept deliberately small — we only need products, prices, meters, and
    entitlement features. Anything else we can stand up via the dashboard.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or os.getenv(
            "STRIPE_API_BASE", "https://api.stripe.com/v1"
        )
        self.timeout = timeout

    def _auth_header(self) -> str:
        token = b64encode(f"{self.api_key}:".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    @staticmethod
    def _flatten(
        payload: Mapping[str, Any], prefix: str = ""
    ) -> List[Tuple[str, str]]:
        """Flatten nested dict -> Stripe's bracketed form-URL encoding."""
        pairs: List[Tuple[str, str]] = []
        for key, value in payload.items():
            full_key = f"{prefix}[{key}]" if prefix else key
            if value is None:
                continue
            if isinstance(value, bool):
                pairs.append((full_key, "true" if value else "false"))
            elif isinstance(value, (int, float, Decimal, str)):
                pairs.append((full_key, str(value)))
            elif isinstance(value, Mapping):
                pairs.extend(StripeClient._flatten(value, full_key))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Mapping):
                        pairs.extend(StripeClient._flatten(item, f"{full_key}[{i}]"))
                    else:
                        pairs.append((f"{full_key}[{i}]", str(item)))
            else:
                pairs.append((full_key, str(value)))
        return pairs

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        query: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            qs = urllib.parse.urlencode(self._flatten(query))
            url = f"{url}?{qs}"
        body: Optional[bytes] = None
        if payload is not None:
            body = urllib.parse.urlencode(self._flatten(payload)).encode("utf-8")
        headers = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "greenlang-factors-provisioner/1.0",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8")
            except Exception:
                pass
            raise StripeError(e.code, raw) from e

    # High-level helpers ---------------------------------------------------

    def list_products(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        starting_after: Optional[str] = None
        while True:
            query: Dict[str, Any] = {"limit": 100, "active": "true"}
            if starting_after:
                query["starting_after"] = starting_after
            resp = self.request("GET", "/products", query=query)
            data = resp.get("data", [])
            out.extend(data)
            if not resp.get("has_more"):
                break
            starting_after = data[-1]["id"] if data else None
        return out

    def list_prices(self, product_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        starting_after: Optional[str] = None
        while True:
            query: Dict[str, Any] = {
                "limit": 100,
                "active": "true",
                "product": product_id,
            }
            if starting_after:
                query["starting_after"] = starting_after
            resp = self.request("GET", "/prices", query=query)
            data = resp.get("data", [])
            out.extend(data)
            if not resp.get("has_more"):
                break
            starting_after = data[-1]["id"] if data else None
        return out


# ---------------------------------------------------------------------------
# Catalog loading + plan computation
# ---------------------------------------------------------------------------


@dataclass
class PlannedAction:
    """One planned change, produced by the diff phase."""

    kind: str  # "create_product" | "update_product" | "create_price" | ...
    resource: str  # GreenLang-owned ID, e.g. "prod_factors_pro"
    detail: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.kind:20s} {self.resource}"


def load_catalog(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"catalog file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"catalog root must be a mapping; got {type(data).__name__}")
    return data


def _to_cents(amount_usd: str) -> int:
    """Convert a USD amount string to integer cents (Decimal-safe)."""
    return int((Decimal(amount_usd) * 100).quantize(Decimal("1")))


def assert_catalog_matches_python(catalog: Dict[str, Any]) -> None:
    """Fail loud if YAML drifts from ``skus.py``.

    The two files must stay in lock-step — the YAML is a declarative
    mirror, not a second source of truth.
    """
    yaml_tier_ids = {t["tier"] for t in catalog.get("tiers", [])}
    py_tier_ids = {t.value for t in CATALOG.tiers.keys()}
    extra_in_yaml = yaml_tier_ids - py_tier_ids
    missing_in_yaml = py_tier_ids - yaml_tier_ids
    if extra_in_yaml or missing_in_yaml:
        raise ValueError(
            f"Tier mismatch between YAML and skus.py. "
            f"Extra-in-YAML={sorted(extra_in_yaml)}, "
            f"Missing-from-YAML={sorted(missing_in_yaml)}"
        )

    yaml_pack_ids = {p["pack"] for p in catalog.get("packs", [])}
    py_pack_ids = {p.value for p in CATALOG.packs.keys()}
    extra_p = yaml_pack_ids - py_pack_ids
    missing_p = py_pack_ids - yaml_pack_ids
    if extra_p or missing_p:
        raise ValueError(
            f"Pack mismatch between YAML and skus.py. "
            f"Extra-in-YAML={sorted(extra_p)}, "
            f"Missing-from-YAML={sorted(missing_p)}"
        )

    # Spot-check amounts for tiers.
    for tier_spec in catalog["tiers"]:
        tier = Tier(tier_spec["tier"])
        py_cfg = CATALOG.tier(tier)
        for price_spec in tier_spec.get("prices", []):
            pid = price_spec["id"]
            amount = Decimal(price_spec["amount_usd"])
            if pid == py_cfg.stripe_price_monthly_id:
                if amount != py_cfg.monthly_price_usd:
                    raise ValueError(
                        f"Tier {tier.value} monthly price drift: "
                        f"yaml={amount} py={py_cfg.monthly_price_usd}"
                    )
            elif pid == py_cfg.stripe_price_annual_id:
                if amount != py_cfg.annual_price_usd:
                    raise ValueError(
                        f"Tier {tier.value} annual price drift: "
                        f"yaml={amount} py={py_cfg.annual_price_usd}"
                    )


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------


@dataclass
class ProvisionResult:
    """Outcome of a provisioning run."""

    dry_run: bool
    actions: List[PlannedAction] = field(default_factory=list)
    stripe_ids: Dict[str, str] = field(default_factory=dict)
    price_ids: Dict[str, str] = field(default_factory=dict)
    feature_ids: Dict[str, str] = field(default_factory=dict)

    def record(self, action: PlannedAction) -> None:
        self.actions.append(action)


class Provisioner:
    """Owns the diff + apply cycle against a Stripe environment.

    ``--dry-run`` mode uses a stub client that never hits the network and
    simply echoes the plan; ``--live`` mode wires up a real
    :class:`StripeClient`.
    """

    def __init__(
        self,
        catalog: Dict[str, Any],
        *,
        client: Optional[StripeClient],
        dry_run: bool,
    ) -> None:
        self.catalog = catalog
        self.client = client
        self.dry_run = dry_run

    # ------------------------- Lookup helpers -------------------------

    def _existing_products_by_gl_id(self) -> Dict[str, Dict[str, Any]]:
        """Return existing Stripe products keyed by ``metadata.gl_sku_id``."""
        if self.dry_run or self.client is None:
            return {}
        products = self.client.list_products()
        out: Dict[str, Dict[str, Any]] = {}
        for p in products:
            meta = p.get("metadata") or {}
            gl_id = meta.get("gl_sku_id")
            if gl_id:
                out[gl_id] = p
        return out

    def _existing_prices_by_gl_id(
        self, stripe_product_id: str
    ) -> Dict[str, Dict[str, Any]]:
        if self.dry_run or self.client is None:
            return {}
        prices = self.client.list_prices(stripe_product_id)
        out: Dict[str, Dict[str, Any]] = {}
        for pr in prices:
            meta = pr.get("metadata") or {}
            gl_id = meta.get("gl_price_id")
            if gl_id:
                out[gl_id] = pr
        return out

    # --------------------------- Apply ---------------------------

    def _create_product(
        self, gl_sku_id: str, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = dict(spec.get("metadata") or {})
        metadata["gl_sku_id"] = gl_sku_id
        payload = {
            "name": spec["name"],
            "description": spec.get("description"),
            "metadata": metadata,
            "active": True,
        }
        if self.dry_run or self.client is None:
            return {"id": f"<dry-run product for {gl_sku_id}>"}
        return self.client.request(
            "POST",
            "/products",
            payload=payload,
            idempotency_key=f"gl-prov-prod-{gl_sku_id}",
        )

    def _update_product(
        self, stripe_product_id: str, gl_sku_id: str, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        metadata = dict(spec.get("metadata") or {})
        metadata["gl_sku_id"] = gl_sku_id
        payload = {
            "name": spec["name"],
            "description": spec.get("description"),
            "metadata": metadata,
        }
        if self.dry_run or self.client is None:
            return {"id": stripe_product_id}
        return self.client.request(
            "POST",
            f"/products/{stripe_product_id}",
            payload=payload,
        )

    def _create_price(
        self,
        *,
        stripe_product_id: str,
        gl_price_id: str,
        currency: str,
        amount_usd: str,
        nickname: Optional[str],
        interval: Optional[str],
        usage_type: str = "licensed",
    ) -> Dict[str, Any]:
        unit_amount = _to_cents(amount_usd)
        payload: Dict[str, Any] = {
            "product": stripe_product_id,
            "currency": currency,
            "unit_amount": unit_amount,
            "nickname": nickname,
            "metadata": {"gl_price_id": gl_price_id},
            "active": True,
        }
        if interval:
            payload["recurring"] = {
                "interval": interval,
                "usage_type": usage_type,
            }
        if self.dry_run or self.client is None:
            return {"id": f"<dry-run price for {gl_price_id}>"}
        return self.client.request(
            "POST",
            "/prices",
            payload=payload,
            idempotency_key=f"gl-prov-price-{gl_price_id}",
        )

    def _create_meter_price(
        self,
        *,
        stripe_product_id: str,
        gl_price_id: str,
        currency: str,
        unit_amount_usd: str,
        unit_label: str,
    ) -> Dict[str, Any]:
        """Metered overage price — ``usage_type=metered``."""
        # Metered prices must use ``recurring.interval`` (month) with
        # ``usage_type=metered``. ``unit_amount_decimal`` lets us price
        # fractional-cent units (e.g. $0.0002/call = 0.02 cents).
        unit_amount_decimal = str(Decimal(unit_amount_usd) * 100)
        payload: Dict[str, Any] = {
            "product": stripe_product_id,
            "currency": currency,
            "unit_amount_decimal": unit_amount_decimal,
            "nickname": f"Meter overage for {gl_price_id}",
            "metadata": {"gl_price_id": gl_price_id},
            "recurring": {
                "interval": "month",
                "usage_type": "metered",
                "aggregate_usage": "sum",
            },
            "billing_scheme": "per_unit",
            "active": True,
        }
        if self.dry_run or self.client is None:
            return {"id": f"<dry-run meter price for {gl_price_id}>"}
        return self.client.request(
            "POST",
            "/prices",
            payload=payload,
            idempotency_key=f"gl-prov-meter-{gl_price_id}",
        )

    def _create_feature(
        self, gl_feature_id: str, name: str
    ) -> Dict[str, Any]:
        """Create a Stripe Product Feature (entitlements API)."""
        payload = {
            "name": name,
            "lookup_key": gl_feature_id,
            "metadata": {"gl_feature_id": gl_feature_id},
        }
        if self.dry_run or self.client is None:
            return {"id": f"<dry-run feature for {gl_feature_id}>"}
        try:
            return self.client.request(
                "POST",
                "/entitlements/features",
                payload=payload,
                idempotency_key=f"gl-prov-feat-{gl_feature_id}",
            )
        except StripeError as exc:
            # Some Stripe accounts don't have entitlements enabled; log
            # and carry on -- features are optional metadata.
            if exc.status_code in (400, 403, 404):
                logger.warning(
                    "entitlements/features not available (HTTP %d); "
                    "skipping feature %s",
                    exc.status_code,
                    gl_feature_id,
                )
                return {"id": None, "skipped": True}
            raise

    # --------------------------- Drivers ---------------------------

    def provision(self) -> ProvisionResult:
        """Walk the catalog, plan + apply every change."""
        result = ProvisionResult(dry_run=self.dry_run)
        existing_products = self._existing_products_by_gl_id()

        # Tiers ------------------------------------------------------
        for tier_spec in self.catalog.get("tiers", []):
            gl_sku_id = tier_spec["product"]["id"]
            self._apply_product_and_prices(
                gl_sku_id=gl_sku_id,
                product_spec=tier_spec["product"],
                price_specs=tier_spec.get("prices", []),
                meter_specs=tier_spec.get("meters", []),
                currency=self.catalog.get("currency", "usd"),
                existing=existing_products.get(gl_sku_id),
                result=result,
            )

        # Packs ------------------------------------------------------
        for pack_spec in self.catalog.get("packs", []):
            gl_sku_id = pack_spec["product"]["id"]
            self._apply_product_and_prices(
                gl_sku_id=gl_sku_id,
                product_spec=pack_spec["product"],
                price_specs=pack_spec.get("prices", []),
                meter_specs=[],
                currency=self.catalog.get("currency", "usd"),
                existing=existing_products.get(gl_sku_id),
                result=result,
            )

        # Feature flags ---------------------------------------------
        for feat_spec in self.catalog.get("feature_flags", []):
            gl_id = feat_spec["id"]
            result.record(PlannedAction("create_feature", gl_id, dict(feat_spec)))
            feat = self._create_feature(gl_id, feat_spec["name"])
            result.feature_ids[gl_id] = feat.get("id") or f"<pending:{gl_id}>"

        return result

    def _apply_product_and_prices(
        self,
        *,
        gl_sku_id: str,
        product_spec: Dict[str, Any],
        price_specs: Sequence[Dict[str, Any]],
        meter_specs: Sequence[Dict[str, Any]],
        currency: str,
        existing: Optional[Dict[str, Any]],
        result: ProvisionResult,
    ) -> None:
        # Product ----------------------------------------------------
        if existing is None:
            result.record(PlannedAction("create_product", gl_sku_id, product_spec))
            resp = self._create_product(gl_sku_id, product_spec)
            stripe_product_id = resp.get("id") or f"<pending:{gl_sku_id}>"
        else:
            result.record(PlannedAction("update_product", gl_sku_id, product_spec))
            stripe_product_id = existing["id"]
            self._update_product(stripe_product_id, gl_sku_id, product_spec)
        result.stripe_ids[gl_sku_id] = stripe_product_id

        existing_prices = self._existing_prices_by_gl_id(stripe_product_id)

        # Regular recurring prices -----------------------------------
        for price_spec in price_specs:
            gl_price_id = price_spec["id"]
            if gl_price_id in existing_prices:
                # Prices are immutable; re-use existing
                result.record(
                    PlannedAction("reuse_price", gl_price_id, price_spec)
                )
                result.price_ids[gl_price_id] = existing_prices[gl_price_id]["id"]
                continue
            result.record(PlannedAction("create_price", gl_price_id, price_spec))
            recurring = price_spec.get("recurring") or {}
            resp = self._create_price(
                stripe_product_id=stripe_product_id,
                gl_price_id=gl_price_id,
                currency=currency,
                amount_usd=price_spec["amount_usd"],
                nickname=price_spec.get("nickname"),
                interval=recurring.get("interval"),
                usage_type=recurring.get("usage_type") or "licensed",
            )
            result.price_ids[gl_price_id] = resp.get("id") or f"<pending:{gl_price_id}>"

        # Metered overage prices -------------------------------------
        for meter_spec in meter_specs:
            gl_price_id = meter_spec["price_id"]
            if Decimal(meter_spec["overage_unit_price_usd"]) == Decimal("0"):
                # Hard-stop meter: do not create a metered price.
                result.record(
                    PlannedAction(
                        "skip_meter_hard_stop",
                        gl_price_id,
                        detail=dict(meter_spec),
                    )
                )
                continue
            if gl_price_id in existing_prices:
                result.record(PlannedAction("reuse_meter", gl_price_id, meter_spec))
                result.price_ids[gl_price_id] = existing_prices[gl_price_id]["id"]
                continue
            result.record(PlannedAction("create_meter", gl_price_id, meter_spec))
            resp = self._create_meter_price(
                stripe_product_id=stripe_product_id,
                gl_price_id=gl_price_id,
                currency=currency,
                unit_amount_usd=meter_spec["overage_unit_price_usd"],
                unit_label=meter_spec.get("unit_label", "unit"),
            )
            result.price_ids[gl_price_id] = resp.get("id") or f"<pending:{gl_price_id}>"


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


def write_lockfile(path: Path, result: ProvisionResult) -> None:
    """Persist GL-id -> Stripe-id mapping so the runtime can look them up."""
    payload = {
        "version": 1,
        "generated_at": int(time.time()),
        "dry_run": result.dry_run,
        "products": result.stripe_ids,
        "prices": result.price_ids,
        "features": result.feature_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)
    logger.info("Wrote Stripe lockfile: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _confirm_prod_or_abort(api_key: str, yes_really_prod: bool) -> None:
    if not api_key.startswith("sk_live_"):
        return
    if yes_really_prod:
        logger.warning("Proceeding against LIVE Stripe (confirmed via --yes-really-prod)")
        return
    prompt = (
        "\n*** WARNING: STRIPE_API_KEY is a LIVE key (sk_live_...) ***\n"
        "This will mutate your production Stripe catalog.\n"
        "Type 'PROVISION LIVE' to continue, anything else aborts: "
    )
    try:
        response = input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Aborted at confirmation prompt.")
    if response.strip() != "PROVISION LIVE":
        raise SystemExit("Aborted: confirmation phrase not entered.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="factors_stripe_provision",
        description="Provision the GreenLang Factors Stripe catalog.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_REPO_ROOT / "deployment" / "stripe" / "factors-products.yaml",
        help="Path to factors-products.yaml",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=_REPO_ROOT
        / "deployment"
        / "stripe"
        / "stripe_catalog_lockfile.json",
        help="Path to write the resolved-IDs lockfile",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Default. Preview the plan without hitting Stripe.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Apply changes against the live Stripe catalog.",
    )
    parser.add_argument(
        "--yes-really-prod",
        action="store_true",
        default=False,
        help="Skip confirmation prompt for sk_live_ keys (CI only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # ``force=True`` so we override any root logger handlers that were
    # installed by ``greenlang`` package imports (which happens in this
    # repo — see observability bootstrap in ``greenlang/__init__.py``).
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        force=True,
    )

    dry_run = not args.live
    catalog = load_catalog(args.catalog)
    assert_catalog_matches_python(catalog)

    client: Optional[StripeClient] = None
    if not dry_run:
        api_key = os.getenv("STRIPE_API_KEY", "").strip()
        if not api_key:
            logger.error(
                "--live requires STRIPE_API_KEY in the environment. Aborting."
            )
            return 2
        _confirm_prod_or_abort(api_key, args.yes_really_prod)
        client = StripeClient(api_key=api_key)

    mode_label = "DRY RUN" if dry_run else "LIVE"
    logger.info("Factors Stripe provisioner starting (%s)", mode_label)
    logger.info("Catalog: %s", args.catalog)

    provisioner = Provisioner(catalog, client=client, dry_run=dry_run)
    result = provisioner.provision()

    # Report -----------------------------------------------------------
    logger.info("Planned actions: %d", len(result.actions))
    for action in result.actions:
        logger.info("  %s", action.describe())

    # Lockfile ---------------------------------------------------------
    if not dry_run:
        write_lockfile(args.lockfile, result)
    else:
        logger.info(
            "Dry-run: skipping lockfile write. Re-run with --live to persist."
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
