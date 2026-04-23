# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the GreenLang Factors test suite.

This conftest is the single source of shared infrastructure for
`tests/factors/`.  It is split into two zones:

1.  **Legacy fixtures** (kept verbatim so existing tests do not break):
    ``emission_db``, ``sample_factor``, ``sample_factor_dict``,
    ``sqlite_catalog``, ``memory_catalog``, ``factor_service``,
    ``api_client``, ``source_registry``, ``gold_eval_cases``.

2.  **New FY27 reproducibility fixtures** (added 2026-04-23 to close the
    CTO-flagged "no reproducible test environment" gap):
    ``factors_app``, ``factors_client``, ``mock_pg``, ``mock_redis``,
    ``mock_stripe``, ``seed_factors``, ``signing_key_pair``.

The new fixtures are designed to:
  * Skip cleanly (``pytest.skip(...)``) when an optional dep is missing
    instead of raising at collection time.
  * Prefer in-memory / fakes when no real backend is available, but use
    the real one (PG via testcontainers / `DATABASE_URL`, Redis via
    `REDIS_URL`) when the docker-compose stack is up.
  * Never call out to real Stripe — the SDK is monkeypatched.
  * Provide a tiny seed catalog (10 factors across families) so E2E
    tests do not need a multi-minute ingest of the full database.

Run via ``make factors-test`` or ``bash scripts/run_factors_tests.sh``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# =============================================================================
# Legacy fixtures (DO NOT MODIFY without coordinating with test owners)
# =============================================================================


@pytest.fixture(scope="session")
def emission_db():
    """Session-scoped EmissionFactorDatabase with cache disabled."""
    from greenlang.data.emission_factor_database import EmissionFactorDatabase

    return EmissionFactorDatabase(enable_cache=False)


@pytest.fixture()
def sample_factor(emission_db):
    """First factor from the built-in database."""
    return next(iter(emission_db.factors.values()))


@pytest.fixture()
def sample_factor_dict(sample_factor):
    """Sample factor serialized to dict (for QA tests)."""
    return sample_factor.to_dict()


@pytest.fixture()
def sqlite_catalog(tmp_path, emission_db):
    """Ingested SQLite DB + SqliteFactorCatalogRepository."""
    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "test_catalog.sqlite"
    ingest_builtin_database(dbfile, "test-edition", label="conftest")
    repo = SqliteFactorCatalogRepository(dbfile)
    return repo


@pytest.fixture()
def memory_catalog(emission_db):
    """MemoryFactorCatalogRepository wrapping built-in DB."""
    from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

    return MemoryFactorCatalogRepository("memory-v1", "memory-test", emission_db)


@pytest.fixture()
def factor_service(sqlite_catalog):
    """FactorCatalogService wrapping SQLite catalog."""
    from greenlang.factors.service import FactorCatalogService

    return FactorCatalogService(sqlite_catalog)


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """FastAPI TestClient with GL_ENV=test and a pre-ingested SQLite catalog."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GL_ENV", "test")
    dbfile = tmp_path / "api_test.sqlite"
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))

    from greenlang.factors.etl.ingest import ingest_builtin_database

    ingest_builtin_database(dbfile, "api-test-edition", label="api-test")

    from greenlang.integration.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def source_registry():
    """Loaded SourceRegistryEntry list from default YAML."""
    from greenlang.factors.source_registry import load_source_registry

    return load_source_registry()


@pytest.fixture()
def gold_eval_cases():
    """Load gold eval smoke cases from fixtures."""
    path = FIXTURES_DIR / "gold_eval_smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# FY27 reproducibility fixtures (added 2026-04-23)
# =============================================================================
#
# These complement (do not replace) the legacy fixtures above.  They are the
# canonical surface for any *new* Factors test that needs an HTTP client,
# a Postgres / Redis / Stripe boundary, or a signed-receipt keypair.


# ---------------------------------------------------------------------------
# FastAPI app + TestClient (new style — uses the legacy api_client glue
# under the hood so behavior is identical, but exposes a name that matches
# the FY27 test plan).
# ---------------------------------------------------------------------------


@pytest.fixture()
def factors_app(monkeypatch, tmp_path):
    """FastAPI app instance with a tiny in-memory factors catalog.

    Tries `greenlang.factors.api.create_factors_app` first (FY27 wiring);
    falls back to `greenlang.integration.api.main:app` so existing tests
    keep working until the dedicated factors app is split out.
    """
    pytest.importorskip("fastapi")

    monkeypatch.setenv("GL_ENV", "test")
    dbfile = tmp_path / "factors_app.sqlite"
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))

    # Seed a real (small) catalog so endpoints have something to return.
    try:
        from greenlang.factors.etl.ingest import ingest_builtin_database

        ingest_builtin_database(dbfile, "factors-app-test", label="factors-app")
    except Exception as exc:  # pragma: no cover — surface-only path
        pytest.skip(f"Factors catalog seed failed: {exc!r}")

    # Prefer dedicated factory if it exists.
    try:
        from greenlang.factors.api import create_factors_app  # type: ignore

        return create_factors_app()
    except (ImportError, AttributeError):
        from greenlang.integration.api.main import app  # type: ignore

        return app


@pytest.fixture()
def factors_client(factors_app):
    """``TestClient(factors_app)`` — the canonical HTTP client for new tests."""
    from fastapi.testclient import TestClient

    with TestClient(factors_app) as client:
        yield client


# ---------------------------------------------------------------------------
# Postgres boundary
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_pg() -> Iterator[str]:
    """Yield a Postgres DSN, or skip cleanly if no PG is available.

    Resolution order:
      1. Honor ``DATABASE_URL`` if set (this is what docker-compose injects).
      2. Honor ``GL_FACTORS_PG_DSN`` if set.
      3. Try ``testcontainers.postgres`` so local dev gets an ephemeral PG.
      4. Skip the test with a clear reason.
    """
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("GL_FACTORS_PG_DSN")
    if dsn:
        yield dsn
        return

    try:
        from testcontainers.postgres import PostgresContainer  # type: ignore
    except Exception:
        pytest.skip(
            "No Postgres available: set DATABASE_URL, run via "
            "`make factors-test`, or `pip install testcontainers`."
        )
        return  # unreachable, satisfies type checker

    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


# ---------------------------------------------------------------------------
# Redis boundary
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    """Return a redis client, preferring real Redis, falling back to fakeredis.

    * If ``REDIS_URL`` is set (docker-compose path), connect to it.
    * Otherwise instantiate ``fakeredis.FakeRedis()`` so unit tests work
      without any infra.
    * Skip if neither is importable.
    """
    url = os.environ.get("REDIS_URL")
    if url:
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(url)
            client.ping()
            return client
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"REDIS_URL={url} but connect failed: {exc!r}")

    try:
        import fakeredis  # type: ignore
    except ImportError:
        pytest.skip("fakeredis not installed: `pip install -e .[factors-test]`.")
    return fakeredis.FakeRedis()


# ---------------------------------------------------------------------------
# Stripe boundary — never calls real Stripe.
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_stripe(monkeypatch):
    """Monkeypatch the ``stripe`` SDK to return canned responses.

    Returns a dict of the patched callables so a test can assert which
    methods were invoked and inspect the recorded payloads.
    """
    stripe = pytest.importorskip("stripe")
    monkeypatch.setattr(stripe, "api_key", "test_key_mock", raising=False)

    recorded: Dict[str, List[Any]] = {
        "Customer.create": [],
        "Subscription.create": [],
        "PaymentIntent.create": [],
        "Webhook.construct_event": [],
    }

    def _customer_create(**kwargs: Any) -> Dict[str, Any]:
        recorded["Customer.create"].append(kwargs)
        return {"id": "cus_test_mock", "object": "customer", **kwargs}

    def _subscription_create(**kwargs: Any) -> Dict[str, Any]:
        recorded["Subscription.create"].append(kwargs)
        return {
            "id": "sub_test_mock",
            "object": "subscription",
            "status": "active",
            **kwargs,
        }

    def _payment_intent_create(**kwargs: Any) -> Dict[str, Any]:
        recorded["PaymentIntent.create"].append(kwargs)
        return {
            "id": "pi_test_mock",
            "object": "payment_intent",
            "status": "succeeded",
            **kwargs,
        }

    def _webhook_construct_event(payload: Any, sig_header: Any, secret: Any) -> Dict[str, Any]:
        recorded["Webhook.construct_event"].append(
            {"payload": payload, "sig": sig_header, "secret": secret}
        )
        body = payload if isinstance(payload, dict) else json.loads(payload)
        return {
            "id": "evt_test_mock",
            "object": "event",
            "type": body.get("type", "invoice.paid"),
            "data": {"object": body.get("data", {})},
        }

    monkeypatch.setattr(stripe.Customer, "create", _customer_create, raising=False)
    monkeypatch.setattr(stripe.Subscription, "create", _subscription_create, raising=False)
    monkeypatch.setattr(stripe.PaymentIntent, "create", _payment_intent_create, raising=False)
    monkeypatch.setattr(stripe.Webhook, "construct_event", _webhook_construct_event, raising=False)
    return recorded


# ---------------------------------------------------------------------------
# Tiny seed catalog — covers all 7 method packs for E2E surface tests.
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_factors() -> List[Dict[str, Any]]:
    """Yield a tiny catalog (10 factors across families).

    Tests that just need *some* factors to query against can use this
    instead of paying for a full ingest of the built-in database.  The
    shape mirrors :class:`EmissionFactorRecord.to_dict()` minimally —
    enough for resolution / matching / signing surfaces.
    """
    return [
        # Electricity (location-based, market-based)
        {
            "factor_id": "elec_us_egrid_us_2023",
            "family": "electricity",
            "scope": 2,
            "region": "US",
            "year": 2023,
            "value": 0.385,
            "unit": "kgCO2e/kWh",
            "source": "EPA eGRID",
            "method": "location_based",
        },
        {
            "factor_id": "elec_uk_desnz_2024",
            "family": "electricity",
            "scope": 2,
            "region": "UK",
            "year": 2024,
            "value": 0.207,
            "unit": "kgCO2e/kWh",
            "source": "DESNZ",
            "method": "location_based",
        },
        # Combustion (stationary + mobile)
        {
            "factor_id": "comb_natgas_us_2023",
            "family": "combustion",
            "scope": 1,
            "region": "US",
            "year": 2023,
            "value": 53.06,
            "unit": "kgCO2e/MMBtu",
            "source": "EPA GHG Hub",
            "method": "ipcc_tier1",
        },
        {
            "factor_id": "comb_diesel_global_2023",
            "family": "combustion",
            "scope": 1,
            "region": "GLOBAL",
            "year": 2023,
            "value": 2.68,
            "unit": "kgCO2e/L",
            "source": "IPCC 2006",
            "method": "ipcc_tier1",
        },
        # Freight
        {
            "factor_id": "freight_truck_us_2023",
            "family": "freight",
            "scope": 3,
            "region": "US",
            "year": 2023,
            "value": 0.105,
            "unit": "kgCO2e/tonne-km",
            "source": "EPA SmartWay",
            "method": "ghg_protocol",
        },
        # Material / CBAM
        {
            "factor_id": "material_steel_eu_2024",
            "family": "material",
            "scope": 3,
            "region": "EU",
            "year": 2024,
            "value": 1.85,
            "unit": "kgCO2e/kg",
            "source": "CBAM Default",
            "method": "cbam_default",
        },
        # Land use
        {
            "factor_id": "land_palm_id_2023",
            "family": "land",
            "scope": 3,
            "region": "ID",
            "year": 2023,
            "value": 7.3,
            "unit": "kgCO2e/kg",
            "source": "EUDR Reference",
            "method": "ipcc_landuse",
        },
        # Product LCA
        {
            "factor_id": "product_cement_global_2023",
            "family": "product",
            "scope": 3,
            "region": "GLOBAL",
            "year": 2023,
            "value": 0.93,
            "unit": "kgCO2e/kg",
            "source": "Ecoinvent",
            "method": "lca_cradle_to_gate",
        },
        # Finance
        {
            "factor_id": "finance_pcaf_us_2023",
            "family": "finance",
            "scope": 3,
            "region": "US",
            "year": 2023,
            "value": 0.000123,
            "unit": "tCO2e/USD",
            "source": "PCAF",
            "method": "pcaf_score4",
        },
        # Refrigerants (cross-cutting GWP test)
        {
            "factor_id": "refrigerant_r410a_global",
            "family": "refrigerant",
            "scope": 1,
            "region": "GLOBAL",
            "year": 2023,
            "value": 2088.0,
            "unit": "kgCO2e/kg",
            "source": "IPCC AR5",
            "method": "gwp100",
        },
    ]


# ---------------------------------------------------------------------------
# Ed25519 signing key pair for signed-receipt tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def signing_key_pair():
    """Generate an ephemeral Ed25519 keypair for signed-receipt tests.

    Returns a dict::

        {
            "key_id":       "test-kid-<hex>",
            "private_pem":  bytes,
            "public_pem":   bytes,
            "private_key":  Ed25519PrivateKey,
            "public_key":   Ed25519PublicKey,
        }

    Skips if `cryptography` is not installed (it is part of the
    `factors-test` extras).
    """
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    import secrets

    private = Ed25519PrivateKey.generate()
    public = private.public_key()

    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return {
        "key_id": f"test-kid-{secrets.token_hex(4)}",
        "private_pem": private_pem,
        "public_pem": public_pem,
        "private_key": private,
        "public_key": public,
        "_cryptography_version": getattr(crypto, "__version__", "unknown"),
    }
