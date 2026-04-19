# -*- coding: utf-8 -*-
"""
Locust load test configuration for GreenLang Factors API.

Target: 1000 req/s sustained, p95 < 50ms.

Scenarios:
  - FactorSearchUser (40%): random searches with varying queries
  - FactorListUser (20%): paginated list with filters
  - FactorGetUser (20%): get individual factors by ID
  - FactorMatchUser (10%): activity-to-factor matching
  - EditionBrowseUser (5%): list editions, get changelog
  - ExportUser (5%): bulk export with tier=enterprise

Usage:
    locust -f tests/factors/load/locustfile.py --host http://localhost:8000

Environment variables:
    GL_FACTORS_BASE_URL: Override the target host
    GL_LOAD_TEST_API_KEY: Override the test API key
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

from locust import HttpUser, between, tag, task

# ── Configuration ───────────────────────────────────────────────────

API_KEY = os.getenv(
    "GL_LOAD_TEST_API_KEY",
    "gl_test_load_key_xxxxxxxxxxxxxxxxxxxxxxxx",
)

SEARCH_QUERIES = [
    "diesel",
    "natural gas",
    "electricity",
    "gasoline",
    "coal",
    "fuel oil",
    "propane",
    "biomass",
    "solar",
    "wind",
    "jet fuel",
    "LPG",
    "CNG",
    "hydrogen",
    "ethanol",
    "biodiesel",
    "refrigerant R-410A",
    "HFC-134a",
    "methane fugitive",
    "waste landfill",
]

GEOGRAPHIES = ["US", "GB", "DE", "FR", "JP", "AU", "CA", "IN", "BR", "CN"]
SCOPES = ["scope_1", "scope_2", "scope_3"]
FUEL_TYPES = [
    "diesel", "natural_gas", "gasoline", "coal_bituminous",
    "fuel_oil_2", "propane", "electricity",
]

ACTIVITY_DESCRIPTIONS = [
    "Burned 1000 gallons of diesel in stationary generators",
    "Purchased 500 MWh of grid electricity in California",
    "Fleet consumed 2000 liters of petrol",
    "Used 300 therms of natural gas for heating",
    "Refrigerant leak of 5 kg R-410A from HVAC",
    "Employee commuting 10000 km by car",
    "Air freight 2000 kg-km international",
    "Waste sent to landfill 50 tonnes",
    "Business travel 5000 passenger-km domestic flights",
    "Purchased 100 tonnes of steel",
]


def _auth_headers() -> Dict[str, str]:
    """Return common auth headers for all requests."""
    return {
        "Authorization": "Bearer %s" % API_KEY,
        "Accept": "application/json",
    }


# ── Scenario 1: Factor Search (40% weight) ─────────────────────────


class FactorSearchUser(HttpUser):
    """Searches for emission factors with varying queries and filters."""

    weight = 40
    wait_time = between(0.1, 0.5)

    _factor_ids: List[str] = []

    @tag("search")
    @task(6)
    def search_simple(self) -> None:
        """Simple keyword search."""
        query = random.choice(SEARCH_QUERIES)
        with self.client.get(
            "/v2/factors/search",
            params={"q": query, "limit": 10},
            headers=_auth_headers(),
            name="/v2/factors/search",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                factors = data.get("factors", [])
                for f in factors[:3]:
                    fid = f.get("factor_id")
                    if fid and fid not in self._factor_ids:
                        self._factor_ids.append(fid)
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("search")
    @task(3)
    def search_filtered(self) -> None:
        """Search with geography and scope filters."""
        query = random.choice(SEARCH_QUERIES)
        params: Dict[str, Any] = {
            "q": query,
            "geography": random.choice(GEOGRAPHIES),
            "limit": 20,
        }
        if random.random() < 0.5:
            params["scope"] = random.choice(SCOPES)
        with self.client.get(
            "/v2/factors/search",
            params=params,
            headers=_auth_headers(),
            name="/v2/factors/search",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("search")
    @task(1)
    def search_sorted(self) -> None:
        """Search with sort parameters."""
        query = random.choice(SEARCH_QUERIES)
        sort_field = random.choice(["relevance", "dqs_score", "co2e_total"])
        sort_order = random.choice(["asc", "desc"])
        with self.client.get(
            "/v2/factors/search",
            params={
                "q": query,
                "sort_by": sort_field,
                "sort_order": sort_order,
                "limit": 25,
            },
            headers=_auth_headers(),
            name="/v2/factors/search",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)


# ── Scenario 2: Factor List (20% weight) ───────────────────────────


class FactorListUser(HttpUser):
    """Browses paginated factor lists with various filters."""

    weight = 20
    wait_time = between(0.2, 0.8)

    @tag("list")
    @task(5)
    def list_first_page(self) -> None:
        """List factors - first page."""
        with self.client.get(
            "/v2/factors",
            params={"limit": 25, "offset": 0},
            headers=_auth_headers(),
            name="/v2/factors",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("list")
    @task(3)
    def list_with_geography(self) -> None:
        """List factors filtered by geography."""
        geo = random.choice(GEOGRAPHIES)
        with self.client.get(
            "/v2/factors",
            params={"geography": geo, "limit": 25},
            headers=_auth_headers(),
            name="/v2/factors",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("list")
    @task(2)
    def list_paginated(self) -> None:
        """List factors - deeper page."""
        offset = random.choice([25, 50, 100, 200])
        with self.client.get(
            "/v2/factors",
            params={"limit": 25, "offset": offset},
            headers=_auth_headers(),
            name="/v2/factors",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)


# ── Scenario 3: Factor Get (20% weight) ────────────────────────────


class FactorGetUser(HttpUser):
    """Retrieves individual factors by ID."""

    weight = 20
    wait_time = between(0.1, 0.5)

    _known_ids: List[str] = []

    def on_start(self) -> None:
        """Seed factor IDs by searching first."""
        resp = self.client.get(
            "/v2/factors/search",
            params={"q": "diesel", "limit": 20},
            headers=_auth_headers(),
        )
        if resp.status_code == 200:
            data = resp.json()
            for f in data.get("factors", []):
                fid = f.get("factor_id")
                if fid:
                    self._known_ids.append(fid)

    @tag("get")
    @task(7)
    def get_factor_by_id(self) -> None:
        """Retrieve a single factor by ID."""
        if not self._known_ids:
            return
        factor_id = random.choice(self._known_ids)
        with self.client.get(
            "/v2/factors/%s" % factor_id,
            headers=_auth_headers(),
            name="/v2/factors/{factor_id}",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 404:
                resp.success()  # expected for some IDs
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("get")
    @task(3)
    def get_factor_with_etag(self) -> None:
        """Retrieve a factor with conditional ETag."""
        if not self._known_ids:
            return
        factor_id = random.choice(self._known_ids)
        headers = _auth_headers()
        # First fetch to get ETag
        resp1 = self.client.get(
            "/v2/factors/%s" % factor_id,
            headers=headers,
            name="/v2/factors/{factor_id}",
        )
        etag = resp1.headers.get("ETag")
        if etag:
            headers["If-None-Match"] = etag
            with self.client.get(
                "/v2/factors/%s" % factor_id,
                headers=headers,
                name="/v2/factors/{factor_id}",
                catch_response=True,
            ) as resp2:
                if resp2.status_code in (200, 304):
                    resp2.success()
                else:
                    resp2.failure("Status %d" % resp2.status_code)


# ── Scenario 4: Factor Match (10% weight) ──────────────────────────


class FactorMatchUser(HttpUser):
    """Tests activity-to-factor matching endpoint."""

    weight = 10
    wait_time = between(0.5, 1.5)

    @tag("match")
    @task
    def match_activity(self) -> None:
        """Match a natural-language activity description to factors."""
        description = random.choice(ACTIVITY_DESCRIPTIONS)
        payload = {
            "activity_description": description,
            "max_candidates": 5,
        }
        geo = random.choice(GEOGRAPHIES)
        if random.random() < 0.5:
            payload["geography"] = geo
        with self.client.post(
            "/v2/factors/match",
            json=payload,
            headers=_auth_headers(),
            name="/v2/factors/match",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)


# ── Scenario 5: Edition Browse (5% weight) ─────────────────────────


class EditionBrowseUser(HttpUser):
    """Browses edition listings and changelogs."""

    weight = 5
    wait_time = between(1.0, 3.0)

    _edition_ids: List[str] = []

    def on_start(self) -> None:
        """Fetch available editions."""
        resp = self.client.get(
            "/v2/editions",
            headers=_auth_headers(),
        )
        if resp.status_code == 200:
            data = resp.json()
            for ed in data.get("editions", []):
                eid = ed.get("edition_id")
                if eid:
                    self._edition_ids.append(eid)

    @tag("edition")
    @task(3)
    def list_editions(self) -> None:
        """List all available editions."""
        with self.client.get(
            "/v2/editions",
            headers=_auth_headers(),
            name="/v2/editions",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("edition")
    @task(2)
    def get_edition_changelog(self) -> None:
        """Get changelog for a specific edition."""
        if not self._edition_ids:
            return
        edition_id = random.choice(self._edition_ids)
        with self.client.get(
            "/v2/editions/%s/changelog" % edition_id,
            headers=_auth_headers(),
            name="/v2/editions/{edition_id}/changelog",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)


# ── Scenario 6: Export (5% weight) ─────────────────────────────────


class ExportUser(HttpUser):
    """Tests bulk export functionality (enterprise tier)."""

    weight = 5
    wait_time = between(2.0, 5.0)

    @tag("export")
    @task(3)
    def export_all(self) -> None:
        """Bulk export all factors (enterprise tier)."""
        with self.client.get(
            "/v2/factors/export",
            params={"tier": "enterprise", "format": "json"},
            headers=_auth_headers(),
            name="/v2/factors/export",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)

    @tag("export")
    @task(2)
    def export_filtered(self) -> None:
        """Bulk export with geography filter."""
        geo = random.choice(GEOGRAPHIES)
        with self.client.get(
            "/v2/factors/export",
            params={
                "tier": "enterprise",
                "format": "json",
                "geography": geo,
            },
            headers=_auth_headers(),
            name="/v2/factors/export",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure("Status %d" % resp.status_code)
