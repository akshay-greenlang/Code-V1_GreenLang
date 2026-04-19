# -*- coding: utf-8 -*-
"""
Pre-built sample queries for each Factors API persona.

Provides ready-to-use query examples based on the personas defined in
docs/factors/personas_and_use_cases.md. Each query includes a description,
a curl command for quick testing, and the expected response fields.

Personas (from FY27 product doc):
  1. API Developer - embeds factor lookup in SaaS; needs pinning and OpenAPI
  2. Data Engineer - runs ingestion pipelines; needs artifacts and checksums
  3. Sustainability Consultant - defends methodology; needs citations and audit export
  4. Climate Platform PM - scopes integration; needs licensing clarity
  5. Methodology Lead - approves certified promotions; needs review queue
  6. Enterprise Security - reviews data residency and connector mode

Extended personas for onboarding kit:
  - Sustainability Manager: Scope 1+2 for US manufacturing
  - Carbon Accountant: DESNZ factors for UK portfolio
  - ESG Analyst: Cross-geography benchmarking
  - Developer: API integration patterns

Example:
    >>> from greenlang.factors.onboarding.sample_queries import get_queries_for_persona
    >>> queries = get_queries_for_persona("sustainability_manager")
    >>> for q in queries:
    ...     print(q.description)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Default base URL for curl examples
_BASE_URL = os.getenv("GL_FACTORS_BASE_URL", "https://api.greenlang.io")
_SAMPLE_KEY = "gl_partner_YOUR_KEY_HERE"


@dataclass
class SampleQuery:
    """A pre-built sample query for partner onboarding.

    Attributes:
        persona: Target persona for this query.
        description: Human-readable description of what this query does.
        curl_command: Ready-to-run curl command.
        expected_fields: List of expected fields in the response.
        notes: Additional usage notes.
    """

    persona: str
    description: str
    curl_command: str
    expected_fields: List[str]
    notes: str = ""

    def as_tuple(self) -> Tuple[str, str, List[str]]:
        """Return (description, curl_command, expected_fields) tuple."""
        return (self.description, self.curl_command, self.expected_fields)


def _curl(method: str, path: str, params: str = "", body: str = "") -> str:
    """Build a curl command string.

    Args:
        method: HTTP method (GET, POST).
        path: API path (e.g., /v2/factors/search).
        params: Query string (without leading ?).
        body: JSON body for POST requests.

    Returns:
        Formatted curl command string.
    """
    url = "%s%s" % (_BASE_URL, path)
    if params:
        url = "%s?%s" % (url, params)

    parts = [
        "curl -s",
        '-H "Authorization: Bearer %s"' % _SAMPLE_KEY,
        '-H "Accept: application/json"',
    ]

    if method == "POST":
        parts.append('-X POST -H "Content-Type: application/json"')
        if body:
            parts.append("-d '%s'" % body)

    parts.append('"%s"' % url)
    return " \\\n  ".join(parts)


# ── Sustainability Manager queries ─────────────────────────────────


def _sustainability_manager_queries() -> List[SampleQuery]:
    """Queries for the Sustainability Manager persona.

    Focus: Scope 1+2 emission factors for US manufacturing.
    """
    return [
        SampleQuery(
            persona="sustainability_manager",
            description="Search for US natural gas emission factors (Scope 1 stationary combustion)",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=natural+gas&geography=US&scope=scope_1&limit=10",
            ),
            expected_fields=[
                "factors", "total_count", "factors[].factor_id",
                "factors[].co2e_per_unit", "factors[].unit",
                "factors[].geography", "factors[].source_org",
            ],
            notes="Filter by scope_1 to get stationary combustion factors.",
        ),
        SampleQuery(
            persona="sustainability_manager",
            description="Search for US grid electricity factors (Scope 2 location-based)",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=electricity+grid&geography=US&scope=scope_2&limit=10",
            ),
            expected_fields=[
                "factors", "total_count", "factors[].factor_id",
                "factors[].co2e_per_unit", "factors[].unit",
            ],
            notes="Location-based Scope 2 factors from eGRID.",
        ),
        SampleQuery(
            persona="sustainability_manager",
            description="List all certified diesel factors for US manufacturing",
            curl_command=_curl(
                "GET",
                "/v2/factors",
                "fuel_type=diesel&geography=US&status=certified&limit=25",
            ),
            expected_fields=[
                "factors", "total_count", "offset",
                "factors[].factor_id", "factors[].fuel_type",
            ],
        ),
        SampleQuery(
            persona="sustainability_manager",
            description="Get detailed provenance for a specific emission factor",
            curl_command=_curl(
                "GET",
                "/v2/factors/EF:EPA:natural_gas:US:2024:v1",
            ),
            expected_fields=[
                "factor_id", "co2e_per_unit", "unit", "provenance",
                "provenance.source_org", "provenance.methodology",
                "provenance.citation", "dqs",
            ],
            notes="Replace factor_id with an actual ID from search results.",
        ),
    ]


# ── Carbon Accountant queries ──────────────────────────────────────


def _carbon_accountant_queries() -> List[SampleQuery]:
    """Queries for the Carbon Accountant persona.

    Focus: DESNZ factors for UK portfolio accounting.
    """
    return [
        SampleQuery(
            persona="carbon_accountant",
            description="Search for DESNZ UK conversion factors (all fuel types)",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=DESNZ&geography=GB&limit=25&sort_by=fuel_type",
            ),
            expected_fields=[
                "factors", "total_count",
                "factors[].factor_id", "factors[].source_org",
                "factors[].co2e_per_unit",
            ],
            notes="DESNZ (formerly BEIS/Defra) publishes the UK Government GHG "
                  "Conversion Factors annually.",
        ),
        SampleQuery(
            persona="carbon_accountant",
            description="Get UK electricity grid factor for Scope 2 reporting",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=electricity+grid&geography=GB&scope=scope_2&limit=5",
            ),
            expected_fields=[
                "factors", "factors[].co2e_per_unit", "factors[].unit",
                "factors[].source_year",
            ],
        ),
        SampleQuery(
            persona="carbon_accountant",
            description="Export audit bundle for a factor (compliance evidence)",
            curl_command=_curl(
                "GET",
                "/v2/factors/EF:DESNZ:natural_gas:GB:2024:v1/audit",
            ),
            expected_fields=[
                "factor_id", "edition_id", "provenance", "license_info",
                "verification_chain", "verification_chain.payload_sha256",
                "quality", "quality.dqs_overall",
            ],
            notes="Audit bundles provide full provenance chain for assurance.",
        ),
        SampleQuery(
            persona="carbon_accountant",
            description="Compare factors between two editions for year-on-year change",
            curl_command=_curl(
                "GET",
                "/v2/editions/2025.12.1/diff/2026.04.1",
            ),
            expected_fields=[
                "left_edition_id", "right_edition_id",
                "added_factor_ids", "removed_factor_ids",
                "changed_factor_ids", "unchanged_count",
            ],
        ),
    ]


# ── ESG Analyst queries ───────────────────────────────────────────


def _esg_analyst_queries() -> List[SampleQuery]:
    """Queries for the ESG Analyst persona.

    Focus: Cross-geography benchmarking and comparison.
    """
    return [
        SampleQuery(
            persona="esg_analyst",
            description="Compare electricity grid factors across major economies",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=electricity+grid&scope=scope_2&limit=50&sort_by=co2e_total&sort_order=desc",
            ),
            expected_fields=[
                "factors", "total_count",
                "factors[].geography", "factors[].co2e_per_unit",
                "factors[].source_year",
            ],
            notes="Sort by co2e_total descending to see highest-carbon grids first.",
        ),
        SampleQuery(
            persona="esg_analyst",
            description="Benchmark natural gas factors: US vs EU vs UK",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=natural+gas+stationary&limit=30&sort_by=geography",
            ),
            expected_fields=[
                "factors", "factors[].geography",
                "factors[].co2e_per_unit", "factors[].unit",
            ],
        ),
        SampleQuery(
            persona="esg_analyst",
            description="Find highest-quality factors (DQS >= 4.0) for reporting",
            curl_command=_curl(
                "GET",
                "/v2/factors/search",
                "q=diesel&dqs_min=4.0&limit=20&sort_by=dqs_score&sort_order=desc",
            ),
            expected_fields=[
                "factors", "factors[].dqs_score", "factors[].factor_id",
            ],
            notes="DQS (Data Quality Score) ranges from 1-5; higher is better.",
        ),
        SampleQuery(
            persona="esg_analyst",
            description="Bulk export all certified factors for portfolio analysis",
            curl_command=_curl(
                "GET",
                "/v2/factors/export",
                "tier=enterprise&format=json&status=certified",
            ),
            expected_fields=[
                "factors[]", "export_manifest",
                "export_manifest.edition_id", "export_manifest.exported_rows",
            ],
            notes="Requires enterprise tier. Community/pro get subset.",
        ),
    ]


# ── Developer queries ──────────────────────────────────────────────


def _developer_queries() -> List[SampleQuery]:
    """Queries for the Developer persona.

    Focus: API integration patterns, edition pinning, and caching.
    """
    return [
        SampleQuery(
            persona="developer",
            description="List available editions for version pinning",
            curl_command=_curl("GET", "/v2/editions"),
            expected_fields=[
                "editions", "editions[].edition_id",
                "editions[].status", "editions[].label",
            ],
            notes="Pin your integration to a specific edition via X-Factors-Edition header.",
        ),
        SampleQuery(
            persona="developer",
            description="Search with edition pinning header",
            curl_command=(
                "curl -s \\\n"
                '  -H "Authorization: Bearer %s" \\\n'
                '  -H "Accept: application/json" \\\n'
                '  -H "X-Factors-Edition: 2026.04.1" \\\n'
                '  "%s/v2/factors/search?q=diesel&limit=5"'
            ) % (_SAMPLE_KEY, _BASE_URL),
            expected_fields=[
                "factors", "edition_id", "total_count",
            ],
            notes="X-Factors-Edition header ensures reproducible results.",
        ),
        SampleQuery(
            persona="developer",
            description="Retrieve factor with ETag for conditional caching",
            curl_command=_curl("GET", "/v2/factors/EF:EPA:diesel:US:2024:v1"),
            expected_fields=[
                "factor_id", "co2e_per_unit", "ETag (response header)",
                "Cache-Control (response header)",
            ],
            notes="Use If-None-Match header with ETag for 304 Not Modified responses.",
        ),
        SampleQuery(
            persona="developer",
            description="Match activity description to emission factors (POST)",
            curl_command=_curl(
                "POST",
                "/v2/factors/match",
                body='{"activity_description": "Burned 500 gallons of diesel in backup generators", "max_candidates": 5}',
            ),
            expected_fields=[
                "candidates", "candidates[].factor_id",
                "candidates[].score", "candidates[].explanation",
            ],
            notes="Natural language matching for activities without known factor IDs.",
        ),
        SampleQuery(
            persona="developer",
            description="Check API health status",
            curl_command=_curl("GET", "/health"),
            expected_fields=[
                "status", "version", "edition_id",
                "factor_count", "components",
            ],
        ),
    ]


# ── Public API ─────────────────────────────────────────────────────


_PERSONA_REGISTRY = {
    "sustainability_manager": _sustainability_manager_queries,
    "carbon_accountant": _carbon_accountant_queries,
    "esg_analyst": _esg_analyst_queries,
    "developer": _developer_queries,
}


def get_queries_for_persona(persona: str) -> List[SampleQuery]:
    """Get all sample queries for a specific persona.

    Args:
        persona: One of "sustainability_manager", "carbon_accountant",
            "esg_analyst", or "developer".

    Returns:
        List of SampleQuery objects for the persona.

    Raises:
        ValueError: If persona is not recognized.
    """
    factory = _PERSONA_REGISTRY.get(persona.lower().strip())
    if factory is None:
        valid = ", ".join(sorted(_PERSONA_REGISTRY.keys()))
        raise ValueError("Unknown persona %r. Valid personas: %s" % (persona, valid))
    return factory()


def get_sample_queries() -> List[SampleQuery]:
    """Get all sample queries across all personas.

    Returns:
        List of all SampleQuery objects, grouped by persona.
    """
    all_queries: List[SampleQuery] = []
    for factory in _PERSONA_REGISTRY.values():
        all_queries.extend(factory())
    return all_queries


def list_personas() -> List[str]:
    """List all available persona names.

    Returns:
        Sorted list of persona identifier strings.
    """
    return sorted(_PERSONA_REGISTRY.keys())
