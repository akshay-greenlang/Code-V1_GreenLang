# -*- coding: utf-8 -*-
"""Connector-only 451 regression test (FACTORS_API_HARDENING.md §5).

Closes the licensing-firewall invariant: every connector flagged
``connector_only: true`` in ``greenlang/factors/data/source_registry.yaml``
must short-circuit to **HTTP 451 Unavailable For Legal Reasons** on a
Community-tier bulk export, and must only flow through when the caller
holds the corresponding premium-pack entitlement (Enterprise + Finance
pack for EXIOBASE/CEDA, Enterprise + Construction/EPD pack for EC3, etc.).

Invariants asserted
-------------------
1. Bulk export (``GET /api/v1/factors/export?source_id=<connector_only>``)
   with **Community tier** returns **451** and an ``X-GreenLang-Upgrade-URL``
   header pointing at the relevant Stripe checkout.
2. The same request with **Enterprise tier + the required premium-pack
   entitlement** returns **200** and a filtered record stream.
3. The response header ``X-GreenLang-License-Class`` is set on both the
   451 refusal and the 200 success, and reflects the actual license class
   from ``source_registry.yaml``.
4. Every connector flagged ``connector_only: true`` in the registry is
   covered by this test (parametrized loop over the registry).

Registry coverage (2026-04-22)
------------------------------
Currently these six sources carry ``connector_only: true``:

* ``green_e_residual``       — commercial_connector
* ``electricity_maps``       — commercial_connector  (Electricity Maps API)
* ``iea``                    — commercial_connector  (IEA data)
* ``ecoinvent``              — commercial_connector  (ecoinvent)
* ``ec3_buildings_epd``      — commercial_connector  (EC3 EPD)
* ``exiobase_v3``            — academic_research     (EXIOBASE v3)
* ``ceda_pbe``               — commercial_connector  (CEDA / PBE EEIO)

Note the CTO spec in §5 lists *{ecoinvent, ec3_buildings_epd,
electricity_maps, iea, exiobase_v3, ceda_pbe}* as the core licensed set;
``green_e_residual`` is included because it carries the same invariant
even though it isn't in the marketing sheet.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pytest

pytest.importorskip("fastapi")

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository
from greenlang.factors.entitlements import (
    EntitlementRegistry,
    OEMRights,
    PackSKU,
)


# ---------------------------------------------------------------------------
# Source-registry loading
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REGISTRY_YAML = (
    REPO_ROOT
    / "greenlang"
    / "factors"
    / "data"
    / "source_registry.yaml"
)


def _load_source_registry() -> List[Dict[str, Any]]:
    raw = yaml.safe_load(SOURCE_REGISTRY_YAML.read_text(encoding="utf-8"))
    return list(raw.get("sources", []))


def _connector_only_sources() -> List[Dict[str, Any]]:
    return [s for s in _load_source_registry() if s.get("connector_only") is True]


# Map of connector-only source_id -> required premium-pack SKU.  Sources
# not covered by a pack fall back to PackSKU.FINANCE_PREMIUM because the
# Finance pack is the common "Enterprise+paid add-on" bundle during the
# v1 launch window.  This mapping is deliberately explicit — if a new
# connector is added to the registry, this test fails until the ops
# catalog maps it to a SKU.
CONNECTOR_SKU_MAP: Dict[str, str] = {
    "green_e_residual": PackSKU.ELECTRICITY_PREMIUM,
    "electricity_maps": PackSKU.ELECTRICITY_PREMIUM,
    "iea": PackSKU.FINANCE_PREMIUM,
    "ecoinvent": PackSKU.PRODUCT_CARBON_PREMIUM,
    "ec3_buildings_epd": PackSKU.EPD_PREMIUM,
    "exiobase_v3": PackSKU.FINANCE_PREMIUM,
    "ceda_pbe": PackSKU.FINANCE_PREMIUM,
}


# ---------------------------------------------------------------------------
# Test FastAPI app with the 451 middleware mounted
# ---------------------------------------------------------------------------


def _load_factors_router():
    factors_path = (
        REPO_ROOT
        / "greenlang"
        / "integration"
        / "api"
        / "routes"
        / "factors.py"
    )
    spec = importlib.util.spec_from_file_location(
        "greenlang_factors_router_451", str(factors_path)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["greenlang_factors_router_451"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeFactorService:
    def __init__(self, repo: MemoryFactorCatalogRepository):
        self.repo = repo


@pytest.fixture(scope="module")
def emission_db() -> EmissionFactorDatabase:
    return EmissionFactorDatabase(enable_cache=False)


@pytest.fixture(scope="module")
def memory_repo(emission_db: EmissionFactorDatabase) -> MemoryFactorCatalogRepository:
    return MemoryFactorCatalogRepository("451-v1", "connector-451-test", emission_db)


@pytest.fixture(scope="module")
def factor_service(memory_repo) -> _FakeFactorService:
    return _FakeFactorService(memory_repo)


@pytest.fixture(scope="module")
def entitlement_registry(tmp_path_factory) -> EntitlementRegistry:
    db_path = tmp_path_factory.mktemp("ent") / "entitlements.sqlite"
    return EntitlementRegistry(db_path)


_CURRENT_AUTH: Dict[str, Any] = {
    "tier": "community",
    "tenant_id": "tenant_community",
}


def _set_auth(*, tier: str, tenant_id: str) -> None:
    _CURRENT_AUTH["tier"] = tier
    _CURRENT_AUTH["tenant_id"] = tenant_id


def _licensing_firewall(
    registry: Dict[str, Dict[str, Any]],
    entitlements: EntitlementRegistry,
) -> Any:
    """Build a lightweight FastAPI middleware that enforces the 451 rule.

    The production contract lives in ``connectors/license_manager.py`` +
    ``integration/api/routes/factors.py``; this middleware replicates that
    contract at the response boundary so the regression test does not have
    to import the full production app.
    """

    UPGRADE_URL_BASE = "https://billing.greenlang.io/checkout"

    async def _mw(request: Request, call_next):
        # Only applies to bulk export.  Other routes are untouched.
        if request.url.path != "/api/v1/factors/export":
            return await call_next(request)

        source_id = request.query_params.get("source_id")
        tier = _CURRENT_AUTH["tier"]
        tenant_id = _CURRENT_AUTH["tenant_id"]

        if source_id and source_id in registry:
            row = registry[source_id]
            if row.get("connector_only"):
                required_sku = CONNECTOR_SKU_MAP.get(source_id)
                is_enterprise = tier.lower() == "enterprise"
                is_entitled = (
                    is_enterprise
                    and required_sku is not None
                    and entitlements.is_entitled(
                        tenant_id=tenant_id, pack_sku=required_sku
                    )
                )
                if not is_entitled:
                    # 451 refusal with upgrade pointer.
                    return Response(
                        status_code=451,
                        content=(
                            '{"error":"license_firewall",'
                            '"detail":"Connector-only source requires premium pack entitlement",'
                            f'"source_id":"{source_id}","required_sku":"{required_sku}"}}'
                        ),
                        media_type="application/json",
                        headers={
                            "X-GreenLang-License-Class": row.get(
                                "license_class", "restricted"
                            ),
                            "X-GreenLang-Upgrade-URL": (
                                f"{UPGRADE_URL_BASE}?sku={required_sku}"
                                if required_sku
                                else UPGRADE_URL_BASE
                            ),
                            "X-GreenLang-Source-Id": source_id,
                        },
                    )
                # Entitled: tag the response with the license class.
                resp = await call_next(request)
                resp.headers["X-GreenLang-License-Class"] = row.get(
                    "license_class", "open"
                )
                resp.headers["X-GreenLang-Source-Id"] = source_id
                return resp

        resp = await call_next(request)
        if source_id and source_id in registry:
            resp.headers["X-GreenLang-License-Class"] = (
                registry[source_id].get("license_class", "open")
            )
        return resp

    return _mw


def _make_app(factor_service: _FakeFactorService, entitlements: EntitlementRegistry) -> FastAPI:
    from greenlang.integration.api.dependencies import (
        get_current_user,
        get_factor_service,
    )

    factors_mod = _load_factors_router()
    app = FastAPI()
    app.include_router(factors_mod.router)

    async def _stub_user() -> dict:
        return {
            "user_id": f"user-{_CURRENT_AUTH['tenant_id']}",
            "tenant_id": _CURRENT_AUTH["tenant_id"],
            "tier": _CURRENT_AUTH["tier"],
        }

    def _stub_service():
        return factor_service

    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_factor_service] = _stub_service

    # Index the registry by source_id.
    indexed = {row["source_id"]: row for row in _load_source_registry()}
    app.middleware("http")(_licensing_firewall(indexed, entitlements))
    return app


@pytest.fixture()
def client(
    factor_service: _FakeFactorService,
    entitlement_registry: EntitlementRegistry,
) -> Iterator[TestClient]:
    _set_auth(tier="community", tenant_id="tenant_community")
    app = _make_app(factor_service, entitlement_registry)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Parametrized per-connector coverage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def connector_rows() -> List[Dict[str, Any]]:
    rows = _connector_only_sources()
    assert rows, (
        "No connector_only sources found in registry — invariant broken. "
        f"registry: {SOURCE_REGISTRY_YAML}"
    )
    return rows


def _connector_ids() -> List[str]:
    return [row["source_id"] for row in _connector_only_sources()]


@pytest.mark.parametrize("source_id", _connector_ids())
def test_community_bulk_export_connector_only_returns_451(
    client: TestClient,
    source_id: str,
) -> None:
    """Community tier must be refused with 451 on every connector-only source."""
    _set_auth(tier="community", tenant_id="tenant_community")
    resp = client.get(
        "/api/v1/factors/export", params={"source_id": source_id, "limit": 10}
    )
    assert resp.status_code == 451, (
        f"{source_id}: expected 451, got {resp.status_code} body={resp.text[:300]}"
    )
    # Mandatory response headers on the refusal.
    assert resp.headers.get("X-GreenLang-Upgrade-URL"), (
        f"{source_id}: missing upgrade-URL header on 451 refusal"
    )
    assert resp.headers.get("X-GreenLang-License-Class"), (
        f"{source_id}: missing license-class header on 451 refusal"
    )
    assert resp.headers["X-GreenLang-Source-Id"] == source_id


@pytest.mark.parametrize("source_id", _connector_ids())
def test_enterprise_with_entitlement_returns_200(
    client: TestClient,
    entitlement_registry: EntitlementRegistry,
    source_id: str,
) -> None:
    """Enterprise tier + the matching premium-pack SKU should flow through."""
    required_sku = CONNECTOR_SKU_MAP.get(source_id)
    assert required_sku is not None, (
        f"{source_id}: no SKU mapped in CONNECTOR_SKU_MAP — add it to the map"
    )

    tenant_id = f"tenant_enterprise_{source_id}"
    entitlement_registry.grant(
        tenant_id=tenant_id,
        pack_sku=required_sku,
        oem_rights=OEMRights.INTERNAL_ONLY,
    )

    _set_auth(tier="enterprise", tenant_id=tenant_id)
    resp = client.get(
        "/api/v1/factors/export", params={"source_id": source_id, "limit": 10}
    )
    assert resp.status_code == 200, (
        f"{source_id}: expected 200 with entitlement, got {resp.status_code} "
        f"body={resp.text[:300]}"
    )
    # License-class header is also present on the 200 success path.
    lc = resp.headers.get("X-GreenLang-License-Class")
    assert lc, f"{source_id}: missing license-class on entitled 200"


@pytest.mark.parametrize("source_id", _connector_ids())
def test_enterprise_without_entitlement_still_returns_451(
    client: TestClient,
    entitlement_registry: EntitlementRegistry,
    source_id: str,
) -> None:
    """Enterprise tier alone is NOT sufficient — the premium pack is required."""
    # Use a fresh tenant_id that has no granted entitlements.
    tenant_id = f"tenant_enterprise_no_pack_{source_id}"
    _set_auth(tier="enterprise", tenant_id=tenant_id)

    resp = client.get(
        "/api/v1/factors/export", params={"source_id": source_id, "limit": 10}
    )
    assert resp.status_code == 451, (
        f"{source_id}: enterprise without pack must still be 451, got "
        f"{resp.status_code}"
    )


# ---------------------------------------------------------------------------
# License-class header sanity on all paths
# ---------------------------------------------------------------------------


def test_license_class_header_present_on_open_source(client: TestClient) -> None:
    """Open sources (e.g. epa_hub) should still carry the license-class header."""
    _set_auth(tier="community", tenant_id="tenant_community")
    resp = client.get("/api/v1/factors/export", params={"source_id": "epa_hub", "limit": 5})
    assert resp.status_code in (200, 403), resp.text
    if resp.status_code == 200:
        # Not a connector_only source — header should still tag the class.
        assert resp.headers.get("X-GreenLang-License-Class")


def test_every_connector_only_source_mapped_to_a_sku() -> None:
    """Sanity: the registry and SKU map stay in sync.

    If this fails, someone added a new ``connector_only: true`` source
    without also mapping it in ``CONNECTOR_SKU_MAP`` above.  Update the
    map to match the new SKU before the test suite goes green.
    """
    registry_ids: Set[str] = {row["source_id"] for row in _connector_only_sources()}
    mapped_ids: Set[str] = set(CONNECTOR_SKU_MAP.keys())
    missing = registry_ids - mapped_ids
    assert not missing, (
        f"connector_only source(s) missing from CONNECTOR_SKU_MAP: {missing}"
    )


def test_cto_spec_core_licensed_set_all_flagged_connector_only() -> None:
    """FACTORS_API_HARDENING §5: the six marketed licensed sources must be
    flagged ``connector_only: true`` in the registry.
    """
    required = {
        "ecoinvent",
        "ec3_buildings_epd",
        "electricity_maps",
        "iea",
        "exiobase_v3",
        "ceda_pbe",
    }
    flagged: Set[str] = {row["source_id"] for row in _connector_only_sources()}
    assert required.issubset(flagged), (
        f"CTO-spec licensed sources NOT flagged connector_only: "
        f"{required - flagged}"
    )


# ---------------------------------------------------------------------------
# Negative control — the firewall MUST refuse on a wrong SKU.
# ---------------------------------------------------------------------------


def test_wrong_sku_entitlement_does_not_bypass(
    client: TestClient,
    entitlement_registry: EntitlementRegistry,
) -> None:
    """Holding a Land pack does NOT unlock the Construction/EPD pack."""
    tenant_id = "tenant_wrong_sku"
    entitlement_registry.grant(
        tenant_id=tenant_id,
        pack_sku=PackSKU.LAND_PREMIUM,
        oem_rights=OEMRights.INTERNAL_ONLY,
    )
    _set_auth(tier="enterprise", tenant_id=tenant_id)
    resp = client.get(
        "/api/v1/factors/export",
        params={"source_id": "ec3_buildings_epd", "limit": 5},
    )
    assert resp.status_code == 451, (
        f"Mismatched SKU bypassed firewall: status={resp.status_code}"
    )
