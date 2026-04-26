#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 follow-up — add 3 CTO-required placeholder sources that were
missed in the initial Phase 1 placeholder pass.

Adds (idempotently):

* ``defra_wtt`` (release_milestone=v0.5) — DEFRA Well-To-Tank /
  indirect emissions companion to DESNZ. The CTO wording lists it
  under v0.1 only "if included"; the v0.1 alpha catalog does not
  include WTT records, so this placeholder is registered at v0.5
  pending legal review.
* ``community_contributions`` (release_milestone=v1.5) — open
  community pack accepting upstream-licenced contributed factors
  through the v1.5 marketplace.
* ``nies_idea`` (release_milestone=v2.5) — NIES IDEA, the dedicated
  Japanese hourly LCI / IO database (distinct from the broader
  ``nies_japan`` v2.0 row which covers the national LCA database).

The script is idempotent: re-running on a registry that already
contains these rows is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:
    print("PyYAML required.", file=sys.stderr)
    raise SystemExit(2)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"


_PLACEHOLDERS: List[Dict[str, Any]] = [
    {
        "source_id": "defra_wtt",
        "urn": "urn:gl:source:defra-wtt",
        "display_name": "UK DEFRA / DESNZ — Well-To-Tank (WTT) and indirect emissions companion",
        "authority": "UK Department for Energy Security and Net Zero (DESNZ)",
        "publisher": "UK Department for Energy Security and Net Zero (DESNZ)",
        "jurisdiction": ["UK", "GB"],
        "licence_class": "community_open",
        "licence": "OGL-UK-3.0",
        "redistribution_class": "attribution_required",
        "cadence": "annual",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting-of-greenhouse-gas-emissions",
        "citation_text": "UK DESNZ Well-To-Tank (WTT) and indirect GHG conversion factors, latest annual edition. Open Government Licence v3.0.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v0.5",
        "latest_source_version": None,
        "latest_ingestion_timestamp": None,
        "notes": (
            "CTO Phase-1 plan listed DEFRA WTT/indirect under v0.1 with the "
            "qualifier 'if included'. The v0.1 alpha catalog seed does NOT "
            "include WTT records (alpha pilot uses direct-combustion factors "
            "only via desnz_ghg_conversion). This row is registered at v0.5 "
            "release_milestone — the next eligible release window — so a WTT "
            "ingestion pass landing post-alpha validates against a known "
            "licence baseline (OGL-3.0). To promote to v0.1 retroactively, "
            "Compliance/Security must promote legal_signoff.status to approved "
            "AND populate latest_source_version + latest_ingestion_timestamp."
        ),
    },
    {
        "source_id": "community_contributions",
        "urn": "urn:gl:source:community-contributions",
        "display_name": "GreenLang Community Contributions — curated 3rd-party submissions",
        "authority": "GreenLang Community Contributors (CLA-bound)",
        "publisher": "GreenLang",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
        "cadence": "continuous",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "community_curated",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://greenlang.io/docs/factors/community-contributions",
        "citation_text": "GreenLang Community Contributions, per-factor contributor + CLA on file.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v1.5",
        "latest_source_version": None,
        "latest_ingestion_timestamp": None,
        "notes": (
            "Container row for community-submitted factors gated by a CLA + "
            "methodology review. Each contribution carries its own provenance "
            "subrow under this source URN. Activated when the v1.5 marketplace "
            "lands."
        ),
    },
    {
        "source_id": "nies_idea",
        "urn": "urn:gl:source:nies-idea",
        "display_name": "NIES IDEA — Inventory Database for Environmental Analysis (Japan)",
        "authority": "NIES Japan (National Institute for Environmental Studies)",
        "publisher": "National Institute for Environmental Studies (Japan) / IDEA Consortium",
        "jurisdiction": ["JP"],
        "licence_class": "commercial_licensed",
        "redistribution_class": "tenant_entitled_only",
        "cadence": "annual",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://idea-lca.com/en/",
        "citation_text": "NIES IDEA — Inventory Database for Environmental Analysis (Japan), latest annual release. Commercial licence required (separate from the broader NIES Japan national LCA database).",
        "entitlement_rules": {"model": "tenant_entitlement_required", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
        "latest_source_version": None,
        "latest_ingestion_timestamp": None,
        "notes": (
            "Distinct from the broader nies_japan (v2.0) national LCA "
            "database. NIES IDEA is the IDEA-Consortium-distributed hourly "
            "LCI / IO dataset commonly used for Japanese product-carbon "
            "calculations; it ships under a separate commercial licence "
            "and at a different release cadence."
        ),
    },
]


def _existing_ids(payload: Dict[str, Any]) -> set:
    return {s.get("source_id") for s in payload.get("sources", []) if isinstance(s, dict)}


def main() -> int:
    text = _REGISTRY.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        print("source_registry.yaml has no sources list", file=sys.stderr)
        return 2

    existing = _existing_ids(payload)
    appended = 0
    for ph in _PLACEHOLDERS:
        if ph["source_id"] in existing:
            continue
        sources.append(ph)
        appended += 1

    if appended == 0:
        print("phase1 placeholders: 0 added (already present)")
        return 0

    payload["sources"] = sources
    _REGISTRY.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"phase1 placeholders: {appended} added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
