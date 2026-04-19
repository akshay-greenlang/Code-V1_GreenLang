# -*- coding: utf-8 -*-
"""Commercial tier enforcement + Connect connector tests.

Covers PLATFORM 2 (#26) and PLATFORM 3 (#27).
"""

from __future__ import annotations

import asyncio

import pytest

from greenlang.commercial import Tier, TIER_SPECS, feature_allowed
from greenlang.commercial.enforcement import InMemoryCounter
from greenlang.commercial.metering import LogSink, UsageEmitter, UsageEvent
from greenlang.connect import default_registry
from greenlang.connect.base import SourceSpec


# ---- Commercial: Tiers ----


def test_tier_specs_have_all_three_tiers():
    assert set(TIER_SPECS.keys()) == set(Tier)


def test_community_tier_restricts_scope_engine():
    assert not feature_allowed(Tier.COMMUNITY, "scope_engine.compute")


def test_pro_tier_allows_compute_and_preview_factors():
    assert feature_allowed(Tier.PRO, "scope_engine.compute")
    assert "preview" in TIER_SPECS[Tier.PRO].factor_visibility


def test_enterprise_tier_has_all_features():
    assert feature_allowed(Tier.ENTERPRISE, "factors.connector")
    assert feature_allowed(Tier.ENTERPRISE, "sla.99_95")
    assert TIER_SPECS[Tier.ENTERPRISE].daily_request_limit == 0  # unlimited


def test_tier_prices_ascending():
    assert TIER_SPECS[Tier.COMMUNITY].price_usd_monthly == 0
    assert TIER_SPECS[Tier.PRO].price_usd_monthly == 99


# ---- Commercial: Counter ----


def test_in_memory_counter_increments():
    counter = InMemoryCounter()
    assert asyncio.run(counter.increment("t1:pro")) == 1
    assert asyncio.run(counter.increment("t1:pro")) == 2
    assert asyncio.run(counter.increment("t2:pro")) == 1


# ---- Commercial: Metering ----


def test_usage_event_serializes():
    event = UsageEvent(tenant_id="t1", tier="pro", metric="api.requests", quantity=1.0)
    data = event.to_dict()
    assert data["tenant_id"] == "t1"
    assert data["metric"] == "api.requests"
    assert data["quantity"] == 1.0


def test_usage_emitter_fans_out_without_raising():
    emitter = UsageEmitter(sinks=[LogSink()])
    event = UsageEvent(tenant_id="t1", tier="pro", metric="api.requests", quantity=1.0)
    asyncio.run(emitter.emit(event))  # no assertion — just shouldn't raise


# ---- Connect: Connector registry ----


def test_three_default_connectors_registered():
    registry = default_registry()
    assert {"sap-s4hana", "snowflake", "aws-cost-explorer"} <= set(registry.available())


def test_unknown_connector_raises():
    with pytest.raises(ValueError):
        default_registry().get("nonexistent")


def test_sap_healthcheck():
    connector = default_registry().get("sap-s4hana")
    ok = asyncio.run(
        connector.healthcheck({"base_url": "https://x", "client_id": "y"})
    )
    assert ok is True

    bad = asyncio.run(connector.healthcheck({}))
    assert bad is False


def test_snowflake_requires_query():
    connector = default_registry().get("snowflake")
    with pytest.raises(ValueError, match="query"):
        asyncio.run(
            connector.extract(
                SourceSpec(
                    tenant_id="t1",
                    connector_id="snowflake",
                    credentials={"account": "x", "user": "y", "password": "z"},
                    filters={},
                )
            )
        )


def test_connector_extract_returns_result_with_checksum():
    connector = default_registry().get("sap-s4hana")
    result = asyncio.run(
        connector.extract(
            SourceSpec(tenant_id="t1", connector_id="sap-s4hana", credentials={})
        )
    )
    assert result.connector_id == "sap-s4hana"
    assert result.row_count == 0
    assert len(result.checksum) == 64
