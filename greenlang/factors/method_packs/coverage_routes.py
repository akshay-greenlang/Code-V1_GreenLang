# -*- coding: utf-8 -*-
"""Method-pack coverage endpoint (MP14, Wave 4-G).

Exposes ``GET /v1/method-packs/coverage`` so operators and tenants can see,
at a glance, which packs are resolving cases today and which are running
hot on the cannot-resolve-safely contract.

The endpoint is read-only + auth-light: it does NOT return individual
factor records, so we skip the licensing-scan middleware. The shape is:

.. code-block:: json

    {
      "packs": [
        {
          "pack_id": "corporate_inventory",
          "version": "0.1.0",
          "status": "preview",
          "supported_families": ["electricity", "combustion", ...],
          "resolved_case_count_7d": 4532,
          "unresolved_case_count_7d": 211,
          "cannot_resolve_safely_count_7d": 17,
          "deprecation_status": "active",
          "replacement_pack_id": null
        }
      ]
    }

The stats aggregation reads from prometheus counters when available, and
falls back to a zero-valued scaffold so the endpoint NEVER 500s just
because metrics are missing. Each zero-valued entry is tagged with a
``_stats_source`` key so callers / dashboards can see whether the number
is real or scaffolded.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from greenlang.factors.method_packs import (
    list_packs,
)
from greenlang.factors.method_packs.registry import _packs as _pack_registry  # noqa: E501
from greenlang.factors.method_packs.finance_proxy import (
    PCAF_CHANGELOG,
    list_pcaf_variants,
    get_pcaf_variant,
)
from greenlang.factors.method_packs.land_removals import (
    LSR_CHANGELOG,
    list_lsr_variants,
    get_lsr_variant,
)
from greenlang.factors.method_packs.product_carbon import (
    PRODUCT_CARBON_CHANGELOG,
)
from greenlang.factors.method_packs.product_lca_variants import (
    PRODUCT_LCA_CHANGELOG,
    get_product_lca_variant,
    list_product_lca_variants,
)

logger = logging.getLogger(__name__)


method_packs_router = APIRouter(
    prefix="/v1/method-packs",
    tags=["factors-v1", "method-packs"],
)


# ---------------------------------------------------------------------------
# Stats aggregation — prometheus counter query stubs
# ---------------------------------------------------------------------------
#
# The resolver emits three counters per pack:
#   - gl_factors_resolved_total{pack_id}
#   - gl_factors_unresolved_total{pack_id}
#   - gl_factors_cannot_resolve_safely_total{pack_id}
#
# When the prometheus client is available we query it via the registry;
# otherwise we return zero counts with a ``_stats_source="scaffold"``
# marker so dashboards can see the data is not real yet.
# TODO(W4-G): wire to prometheus_exporter.get_counter_value() once that
# helper lands from the observability track.


def _query_counter(counter_name: str, pack_id: str, window_seconds: int = 7 * 86400) -> int:
    """Return the cumulative counter value for ``pack_id`` over the window.

    Zero-tolerant: any exception (counter not registered, client missing,
    registry unresponsive) downgrades to ``0`` so the endpoint continues
    to respond. The caller should combine this with a ``_stats_source``
    marker to signal scaffolded data.

    TODO(W4-G): hook to the shared prometheus_exporter aggregation helper
    when that module's ``get_counter_over_window(name, labels, window)``
    surfaces (tracked under OBS-001).
    """
    try:
        from prometheus_client import REGISTRY  # type: ignore

        metric = None
        for family in REGISTRY.collect():
            if family.name == counter_name:
                metric = family
                break
        if metric is None:
            return 0

        total = 0.0
        for sample in metric.samples:
            labels = getattr(sample, "labels", {}) or {}
            if labels.get("pack_id") == pack_id:
                total += float(sample.value)
        return int(total)
    except Exception as exc:  # noqa: BLE001
        logger.debug("counter %s query failed for %s: %s", counter_name, pack_id, exc)
        return 0


def _pack_stats(pack_id: str) -> Dict[str, Any]:
    """Return the three-counter stats block for a pack id."""
    resolved = _query_counter("gl_factors_resolved_total", pack_id)
    unresolved = _query_counter("gl_factors_unresolved_total", pack_id)
    cannot_resolve = _query_counter(
        "gl_factors_cannot_resolve_safely_total", pack_id
    )
    stats_source = "prometheus" if (resolved or unresolved or cannot_resolve) else "scaffold"
    return {
        "resolved_case_count_7d": resolved,
        "unresolved_case_count_7d": unresolved,
        "cannot_resolve_safely_count_7d": cannot_resolve,
        "_stats_source": stats_source,
    }


# ---------------------------------------------------------------------------
# Family + status inference per pack
# ---------------------------------------------------------------------------


def _supported_families(pack) -> List[str]:
    """Extract the sorted unique family list from a pack's SelectionRule."""
    try:
        families = pack.selection_rule.allowed_families
        return sorted({getattr(f, "value", str(f)) for f in families})
    except Exception:  # noqa: BLE001
        return []


def _pack_status(pack) -> str:
    """Derive a coarse ``preview`` / ``certified`` / ``deprecated`` label.

    We look at the pack_version + tags + deprecation.replacement_pack_id to
    infer status. A version of ``0.x.y`` is treated as preview; ``1.x.y+``
    is certified unless a replacement pointer is set.
    """
    version = getattr(pack, "pack_version", "0.0.0")
    replacement = getattr(pack, "replacement_pack_id", None)
    tags = set(getattr(pack, "tags", ()) or ())

    if replacement:
        return "deprecated"
    if "deprecated" in tags:
        return "deprecated"
    major = version.split(".", 1)[0]
    if major == "0":
        return "preview"
    return "certified"


def _deprecation_status(pack) -> str:
    replacement = getattr(pack, "replacement_pack_id", None)
    tags = set(getattr(pack, "tags", ()) or ())
    if replacement:
        return "sunsetting"
    if "deprecated" in tags:
        return "deprecated"
    return "active"


# ---------------------------------------------------------------------------
# MP12: v0.1.0 baseline changelog entries for the Wave 2 hardened packs
# (corporate, electricity, eu_policy, freight). We maintain these here in
# the coverage module so we do NOT have to touch the pack files (which
# are off-limits for this wave).
#
# Each entry follows MP12's {version, date, changes, impact, migration_notes}
# shape. The three v0.2 families (product_carbon, land_removals,
# finance_proxy) carry their own changelogs in their respective modules.
# ---------------------------------------------------------------------------


_WAVE2_BASELINE_CHANGELOG: Dict[str, List[Dict[str, Any]]] = {
    "corporate_scope1": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "GHG Protocol Corporate Standard Scope 1 selection + boundary rules",
                "Fossil-only biogenic treatment",
                "IPCC AR6 100-year GWP basis",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "corporate_scope2_location_based": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "GHG Protocol Scope 2 Guidance location-based method",
                "Market instruments explicitly PROHIBITED",
                "Offsets excluded from Scope 2 inventory",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "corporate_scope2_market_based": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "GHG Protocol Scope 2 Guidance market-based method",
                "RECs / GOs / PPAs + residual mix support",
                "Offsets excluded from market-based inventory",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "corporate_scope3": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "GHG Protocol Scope 3 Standard (15 categories)",
                "Cat 11 use-phase parameter support",
                "SBTi FLAG / PCAF / CSRD E1 reporting labels",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "electricity_location": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "Location-based grid-intensity factors",
                "Jurisdictional filter helper (_jurisdiction_filter)",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "electricity_market": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "Market-based grid-intensity factors",
                "AIB / Green-e / NGA / METI / DESNZ / CER / KEMCO / EMA residual-mix routing",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "eu_cbam": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "CBAM (Regulation EU 2023/956) CN-code coverage",
                "Verification-required SelectionRule",
                "Biogenic CO2 EXCLUDED per Article 7(2)",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "eu_dpp": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "EU Digital Product Passport (ESPR, Regulation 2024/1781) scaffold",
                "Category-open inclusion list pending delegated acts",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "eu_dpp_battery": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "EU Battery Regulation 2023/1542 Article 7 CFP disclosure",
                "Battery class / energy / weight metadata",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
    "freight_iso_14083": [
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "ISO 14083:2023 + GLEC Framework v3.0 transport-chain support",
                "Road / sea / air / rail / inland waterway WTW + TTW labelling",
            ],
            "impact": "baseline",
            "migration_notes": "N/A — pack introduction.",
        },
    ],
}


def _changelog_for(pack_id: str) -> List[Dict[str, Any]]:
    """Return the changelog for ``pack_id`` from the per-pack registries."""
    if pack_id in LSR_CHANGELOG:
        return list(LSR_CHANGELOG[pack_id])
    if pack_id in PCAF_CHANGELOG:
        return list(PCAF_CHANGELOG[pack_id])
    if pack_id in PRODUCT_LCA_CHANGELOG:
        return list(PRODUCT_LCA_CHANGELOG[pack_id])
    if pack_id == "product_carbon":
        return list(PRODUCT_CARBON_CHANGELOG)
    if pack_id in _WAVE2_BASELINE_CHANGELOG:
        return list(_WAVE2_BASELINE_CHANGELOG[pack_id])
    return []


def _collect_all_packs() -> List[Dict[str, Any]]:
    """Walk the method-pack registry and return a coverage entry per pack."""
    seen_ids: set = set()
    entries: List[Dict[str, Any]] = []

    # Profile-registered packs (corporate, electricity, freight, etc.).
    try:
        profile_map = dict(_pack_registry)
    except Exception:  # noqa: BLE001
        profile_map = {}

    for profile, pack in profile_map.items():
        pack_id = getattr(profile, "value", str(profile))
        if pack_id in seen_ids:
            continue
        seen_ids.add(pack_id)
        stats = _pack_stats(pack_id)
        entries.append({
            "pack_id": pack_id,
            "pack_name": pack.name,
            "version": pack.pack_version,
            "status": _pack_status(pack),
            "supported_families": _supported_families(pack),
            "deprecation_status": _deprecation_status(pack),
            "replacement_pack_id": getattr(pack, "replacement_pack_id", None),
            "deprecation_notice_days": getattr(pack, "deprecation_notice_days", 180),
            "cannot_resolve_action": getattr(
                getattr(pack, "cannot_resolve_action", None), "value", None
            ),
            "global_default_tier_allowed": getattr(
                pack, "global_default_tier_allowed", False
            ),
            "reporting_labels": list(pack.reporting_labels),
            "changelog": _changelog_for(pack_id),
            **stats,
        })

    # Variant-registered packs (PCAF asset classes + LSR variants + product LCA).
    for variant_name in list_pcaf_variants():
        if variant_name in seen_ids:
            continue
        seen_ids.add(variant_name)
        pack = get_pcaf_variant(variant_name)
        stats = _pack_stats(variant_name)
        entries.append({
            "pack_id": variant_name,
            "pack_name": pack.name,
            "version": pack.pack_version,
            "status": _pack_status(pack),
            "supported_families": _supported_families(pack),
            "deprecation_status": _deprecation_status(pack),
            "replacement_pack_id": getattr(pack, "replacement_pack_id", None),
            "deprecation_notice_days": getattr(pack, "deprecation_notice_days", 180),
            "cannot_resolve_action": getattr(
                getattr(pack, "cannot_resolve_action", None), "value", None
            ),
            "global_default_tier_allowed": getattr(
                pack, "global_default_tier_allowed", False
            ),
            "reporting_labels": list(pack.reporting_labels),
            "changelog": _changelog_for(variant_name),
            **stats,
        })

    for variant_name in list_lsr_variants():
        if variant_name in seen_ids:
            continue
        seen_ids.add(variant_name)
        pack = get_lsr_variant(variant_name)
        stats = _pack_stats(variant_name)
        entries.append({
            "pack_id": variant_name,
            "pack_name": pack.name,
            "version": pack.pack_version,
            "status": _pack_status(pack),
            "supported_families": _supported_families(pack),
            "deprecation_status": _deprecation_status(pack),
            "replacement_pack_id": getattr(pack, "replacement_pack_id", None),
            "deprecation_notice_days": getattr(pack, "deprecation_notice_days", 180),
            "cannot_resolve_action": getattr(
                getattr(pack, "cannot_resolve_action", None), "value", None
            ),
            "global_default_tier_allowed": getattr(
                pack, "global_default_tier_allowed", False
            ),
            "reporting_labels": list(pack.reporting_labels),
            "changelog": _changelog_for(variant_name),
            **stats,
        })

    for variant_name, pack in list_product_lca_variants().items():
        if variant_name in seen_ids:
            continue
        seen_ids.add(variant_name)
        stats = _pack_stats(variant_name)
        entries.append({
            "pack_id": variant_name,
            "pack_name": pack.name,
            "version": pack.pack_version,
            "status": _pack_status(pack),
            "supported_families": _supported_families(pack),
            "deprecation_status": _deprecation_status(pack),
            "replacement_pack_id": getattr(pack, "replacement_pack_id", None),
            "deprecation_notice_days": getattr(pack, "deprecation_notice_days", 180),
            "cannot_resolve_action": getattr(
                getattr(pack, "cannot_resolve_action", None), "value", None
            ),
            "global_default_tier_allowed": getattr(
                pack, "global_default_tier_allowed", False
            ),
            "reporting_labels": list(pack.reporting_labels),
            "changelog": _changelog_for(variant_name),
            **stats,
        })

    entries.sort(key=lambda e: e["pack_id"])
    return entries


@method_packs_router.get("/coverage")
def method_packs_coverage(request: Request) -> Dict[str, Any]:
    """List all registered method packs with v0.2 coverage metadata.

    Skips the licensing-scan middleware (no factor records returned).
    """
    try:
        request.state.skip_licensing_scan = True
    except Exception:  # noqa: BLE001
        pass
    packs = _collect_all_packs()
    return {"packs": packs, "total": len(packs)}


__all__ = ["method_packs_router"]
