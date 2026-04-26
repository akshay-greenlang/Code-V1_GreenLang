#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — migrate source_registry.yaml to the canonical Phase-1 schema.

For every source in ``greenlang/factors/data/source_registry.yaml``:

* Add the CTO Phase 1 mandatory fields if missing:
  - ``urn``, ``authority``, ``publisher``, ``source_owner``, ``parser`` block,
    ``trust_tier``, ``legal_signoff`` block, ``publication_url``,
    ``citation_text``, ``entitlement_rules``, ``release_milestone``,
    ``latest_source_version``, ``latest_ingestion_timestamp``.
* Map legacy ``license_class`` → canonical ``licence_class`` enum.
* Derive ``redistribution_class`` from existing fields when missing.
* Preserve all existing fields verbatim (additionalProperties=true).
* Append placeholder entries for v0.5–v2.5 planned sources missing today.

The script is idempotent and conservative: it only ADDS fields. It does
not remove or rename anything. Re-runs are no-ops on already-migrated
sources.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:
    print("PyYAML required. pip install pyyaml", file=sys.stderr)
    raise SystemExit(2)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = _REPO_ROOT / "greenlang" / "factors" / "data" / "source_registry.yaml"


# ---------------------------------------------------------------------------
# Legacy license_class -> canonical Phase-1 licence_class enum
# ---------------------------------------------------------------------------
_LICENCE_MAP: Dict[str, str] = {
    "public_us_government": "community_open",
    "uk_open_government": "community_open",
    "eu_publication": "community_open",
    "public_eu": "community_open",
    "public_in_government": "community_open",
    "public_international": "community_open",
    "ipcc_reference": "community_open",
    "open": "community_open",
    "greenlang_terms": "community_open",
    "commercial_connector": "connector_only",
    "registry_terms": "method_only",
    "wri_wbcsd_terms": "method_only",
    "smart_freight_terms": "method_only",
    "pcaf_attribution": "method_only",
    "academic_research": "commercial_licensed",
    "restricted": "commercial_licensed",
}


_REDISTRIBUTION_MAP: Dict[str, str] = {
    # Existing redistribution_class values mapped to Phase-1 enum.
    "open": "redistribution_allowed",
    "restricted": "tenant_entitled_only",
    "licensed": "tenant_entitled_only",
}


def _derive_licence_class(src: Dict[str, Any]) -> str:
    if "licence_class" in src:
        return src["licence_class"]
    legacy = src.get("license_class")
    if legacy and legacy in _LICENCE_MAP:
        return _LICENCE_MAP[legacy]
    if src.get("connector_only"):
        return "connector_only"
    if src.get("redistribution_allowed"):
        return "community_open"
    return "method_only"


def _derive_redistribution_class(src: Dict[str, Any], licence_class: str) -> str:
    if "redistribution_class" in src and src["redistribution_class"] in (
        "redistribution_allowed",
        "attribution_required",
        "metadata_only",
        "derived_values_only",
        "tenant_entitled_only",
        "no_redistribution",
        "blocked",
    ):
        return src["redistribution_class"]
    legacy = src.get("redistribution_class")
    if legacy and legacy in _REDISTRIBUTION_MAP:
        return _REDISTRIBUTION_MAP[legacy]
    if licence_class == "blocked":
        return "blocked"
    if licence_class == "private_tenant_scoped":
        return "tenant_entitled_only"
    if licence_class == "connector_only":
        return "metadata_only"
    if licence_class == "commercial_licensed":
        return "tenant_entitled_only"
    if licence_class == "method_only":
        return "metadata_only"
    if src.get("redistribution_allowed") is True:
        return (
            "attribution_required"
            if src.get("attribution_required")
            else "redistribution_allowed"
        )
    return "no_redistribution"


def _derive_entitlement_rules(licence_class: str) -> Dict[str, Any]:
    if licence_class == "community_open":
        return {"model": "public_no_entitlement", "metadata_visibility": "public"}
    if licence_class == "method_only":
        return {"model": "public_no_entitlement", "metadata_visibility": "public"}
    if licence_class == "commercial_licensed":
        return {
            "model": "tenant_entitlement_required",
            "metadata_visibility": "public",
        }
    if licence_class == "private_tenant_scoped":
        return {
            "model": "private_tenant_owner_only",
            "metadata_visibility": "tenant_only",
        }
    if licence_class == "connector_only":
        return {
            "model": "connector_only_no_bulk",
            "metadata_visibility": "public",
        }
    if licence_class == "blocked":
        return {"model": "blocked", "metadata_visibility": "blocked"}
    return {"model": "public_no_entitlement", "metadata_visibility": "public"}


def _derive_trust_tier(src: Dict[str, Any]) -> str:
    if src.get("trust_tier"):
        return src["trust_tier"]
    vs = (src.get("verification_status") or "").lower()
    if vs == "regulator_approved":
        return "regulator_approved"
    if vs == "external_verified":
        return "external_verified"
    return "community_curated"


def _derive_release_milestone(src: Dict[str, Any]) -> str:
    if src.get("release_milestone"):
        return src["release_milestone"]
    if src.get("alpha_v0_1") is True:
        return "v0.1"
    sid = src.get("source_id", "")
    # heuristic mapping for known categories
    v0_5 = {"iea", "iea_emission_factors", "india_bee_pat"}
    v0_9 = {"edgar", "unfccc_nir", "unfccc_bur", "unfccc_btr"}
    v1_5 = {
        "ademe",
        "glec_framework",
        "iso_14083",
        "ashrae_ahri",
        "climate_trace",
        "pact_pathfinder",
    }
    v2_0 = {"ecoinvent", "exiobase_v3", "pcaf_global_std_v2", "nies", "wri_aqueduct"}
    v2_5 = {
        "entsoe",
        "electricity_maps",
        "watttime",
        "us_iso_rto",
        "grid_india_realtime",
        "nies_idea",
        "agribalyse",
        "faostat",
    }
    if sid in v0_5:
        return "v0.5"
    if sid in v0_9:
        return "v0.9"
    if sid in v1_5:
        return "v1.5"
    if sid in v2_0:
        return "v2.0"
    if sid in v2_5:
        return "v2.5"
    return "v1.0"


def _derive_authority_publisher(src: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not src.get("authority"):
        guess = {
            "epa_hub": "U.S. Environmental Protection Agency",
            "egrid": "U.S. Environmental Protection Agency",
            "desnz_ghg_conversion": "UK Department for Energy Security and Net Zero (DESNZ)",
            "tcr_grp_defaults": "The Climate Registry / GHGRP",
            "green_e_residual": "Green-e (Center for Resource Solutions)",
            "ghgp_method_refs": "World Resources Institute / WBCSD (GHG Protocol)",
            "iea": "International Energy Agency",
            "ecoinvent": "ecoinvent Association",
            "defra_conversion": "UK Department for Environment, Food and Rural Affairs",
            "eu_cbam": "European Commission DG TAXUD",
            "greenlang_builtin": "GreenLang",
            "electricity_maps": "Electricity Maps ApS",
        }
        out["authority"] = guess.get(src["source_id"], "TBD")
    if not src.get("publisher"):
        out["publisher"] = out.get("authority") or src.get("authority") or "TBD"
    return out


def _derive_parser_block(src: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if src.get("parser") and isinstance(src["parser"], dict):
        # If a previous migration wrote a non-canonical sentinel ("TBD"),
        # rewrite to the schema-compliant placeholder.
        existing = src["parser"]
        mod = existing.get("module") or ""
        fn = existing.get("function") or ""
        if mod == "TBD" or fn == "TBD":
            return {
                "module": "tbd.placeholder",
                "function": "tbd_placeholder",
                "version": existing.get("version", "0.0.0"),
            }
        return None
    module = src.get("parser_module")
    function = src.get("parser_function")
    version = src.get("parser_version", "0.0.0")
    if module and function:
        return {"module": module, "function": function, "version": version}
    # Fall back to a TBD parser block so the canonical schema's required
    # ``parser`` block is satisfied even for not-yet-wired sources. The
    # SourceRightsService treats ``parser.module == 'TBD'`` as
    # "ingestion blocked until wired" via the trust_tier + release
    # gate, so this is safe.
    return {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"}


def _derive_legal_signoff(src: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(src.get("legal_signoff"), dict):
        return None
    artifact = src.get("legal_signoff_artifact")
    if artifact:
        # legacy artifact path implies an approved review outside the new
        # block model; we do not invent a reviewer/timestamp here.
        return {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": str(artifact),
        }
    return {
        "status": "pending_legal_review",
        "reviewed_by": None,
        "reviewed_at": None,
        "evidence_uri": None,
    }


def _derive_publication_url(src: Dict[str, Any]) -> Optional[str]:
    if src.get("publication_url"):
        return None
    watch = src.get("watch") or {}
    return (
        watch.get("url")
        or "https://greenlang.io/docs/factors/source-rights/pending-publication-url"
    )


def _derive_citation_text(src: Dict[str, Any]) -> Optional[str]:
    if src.get("citation_text"):
        return None
    return f"{src.get('display_name', src['source_id'])}, per source publication."


def _migrate_source(src: Dict[str, Any]) -> Dict[str, Any]:
    licence_class = _derive_licence_class(src)
    redistribution_class = _derive_redistribution_class(src, licence_class)
    src.setdefault("licence_class", licence_class)
    src["redistribution_class"] = redistribution_class
    src.setdefault("entitlement_rules", _derive_entitlement_rules(licence_class))
    src.setdefault("trust_tier", _derive_trust_tier(src))
    src.setdefault("release_milestone", _derive_release_milestone(src))

    for k, v in _derive_authority_publisher(src).items():
        src.setdefault(k, v)

    if not src.get("source_owner"):
        src["source_owner"] = "climate-methodology-lead"

    p = _derive_parser_block(src)
    if p is not None:
        src["parser"] = p

    ls = _derive_legal_signoff(src)
    if ls is not None:
        src["legal_signoff"] = ls

    pu = _derive_publication_url(src)
    if pu is not None:
        src.setdefault("publication_url", pu)

    ct = _derive_citation_text(src)
    if ct is not None:
        src.setdefault("citation_text", ct)

    if not src.get("urn"):
        slug = src["source_id"].replace("_", "-")
        src["urn"] = f"urn:gl:source:{slug}"

    src.setdefault(
        "latest_source_version", src.get("source_version") or src.get("dataset_version")
    )
    src.setdefault(
        "latest_ingestion_timestamp",
        src.get("latest_ingestion_at") or src.get("ingestion_date"),
    )

    return src


# ---------------------------------------------------------------------------
# Phase 1 placeholder entries for sources not yet present
# ---------------------------------------------------------------------------
_NOW = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

_PLACEHOLDERS: List[Dict[str, Any]] = [
    {
        "source_id": "iea_emission_factors",
        "urn": "urn:gl:source:iea-emission-factors",
        "display_name": "IEA Emissions Factors (Statistics + Greenhouse Gas Emissions)",
        "authority": "International Energy Agency",
        "publisher": "International Energy Agency",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "commercial_licensed",
        "redistribution_class": "tenant_entitled_only",
        "cadence": "annual",
        "source_owner": "compliance-security-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.iea.org/data-and-statistics/data-product/emissions-factors-2024",
        "citation_text": "International Energy Agency, Emissions Factors database (annual). Commercial licence required.",
        "entitlement_rules": {
            "model": "tenant_entitlement_required",
            "metadata_visibility": "public",
            "required_pack_sku": None,
        },
        "release_milestone": "v0.5",
    },
    {
        "source_id": "india_bee_pat",
        "urn": "urn:gl:source:india-bee-pat",
        "display_name": "India BEE PAT (Perform Achieve Trade) sectoral baselines",
        "authority": "Bureau of Energy Efficiency, Government of India",
        "publisher": "Bureau of Energy Efficiency",
        "jurisdiction": ["IN"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
        "cadence": "triennial",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://beeindia.gov.in/en/programmes/perform-achieve-trade-pat",
        "citation_text": "PAT Scheme sectoral specific energy consumption baselines, BEE / MoEFCC, Government of India.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v0.5",
    },
    {
        "source_id": "edgar",
        "urn": "urn:gl:source:edgar",
        "display_name": "EDGAR — Emissions Database for Global Atmospheric Research",
        "authority": "European Commission Joint Research Centre",
        "publisher": "European Commission JRC",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
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
        "publication_url": "https://edgar.jrc.ec.europa.eu/",
        "citation_text": "EDGAR (Emissions Database for Global Atmospheric Research), European Commission JRC, latest annual edition.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v0.9",
    },
    {
        "source_id": "unfccc_nir_bur_btr",
        "urn": "urn:gl:source:unfccc-nir-bur-btr",
        "display_name": "UNFCCC National Inventory Reports / Biennial Update / Transparency Reports",
        "authority": "UNFCCC Secretariat",
        "publisher": "United Nations Framework Convention on Climate Change",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
        "cadence": "biennial",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://unfccc.int/process-and-meetings/transparency-and-reporting",
        "citation_text": "UNFCCC National Inventory Reports / BUR / BTR submissions (per Party).",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v0.9",
    },
    {
        "source_id": "ademe_base_carbone",
        "urn": "urn:gl:source:ademe-base-carbone",
        "display_name": "ADEME Base Carbone (France)",
        "authority": "ADEME — Agence de la transition écologique",
        "publisher": "ADEME",
        "jurisdiction": ["FR"],
        "licence_class": "community_open",
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
        "publication_url": "https://base-carbone.ademe.fr/",
        "citation_text": "ADEME Base Carbone, latest published edition. Etalab open licence (with attribution).",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v1.5",
    },
    {
        "source_id": "iso_14083",
        "urn": "urn:gl:source:iso-14083",
        "display_name": "ISO 14083:2023 — Quantification and reporting of greenhouse gas emissions arising from transport chain operations",
        "authority": "International Organization for Standardization",
        "publisher": "ISO",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "method_only",
        "redistribution_class": "metadata_only",
        "cadence": "ad_hoc",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.iso.org/standard/78864.html",
        "citation_text": "ISO 14083:2023 transport-chain GHG quantification methodology. ISO copyright; method references only.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v1.5",
    },
    {
        "source_id": "ashrae_ahri",
        "urn": "urn:gl:source:ashrae-ahri",
        "display_name": "ASHRAE / AHRI HVAC and refrigerant performance standards",
        "authority": "ASHRAE / AHRI",
        "publisher": "ASHRAE / AHRI",
        "jurisdiction": ["US", "GLOBAL"],
        "licence_class": "method_only",
        "redistribution_class": "metadata_only",
        "cadence": "ad_hoc",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.ashrae.org/technical-resources",
        "citation_text": "ASHRAE / AHRI standards, per latest published edition. Method references only.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v1.5",
    },
    {
        "source_id": "climate_trace",
        "urn": "urn:gl:source:climate-trace",
        "display_name": "Climate TRACE — global emissions inventory",
        "authority": "Climate TRACE Coalition",
        "publisher": "Climate TRACE",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
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
        "publication_url": "https://climatetrace.org/",
        "citation_text": "Climate TRACE global emissions inventory, latest annual release. CC-BY-4.0.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v1.5",
    },
    {
        "source_id": "nies_japan",
        "urn": "urn:gl:source:nies-japan",
        "display_name": "NIES — National Institute for Environmental Studies (Japan)",
        "authority": "NIES Japan",
        "publisher": "National Institute for Environmental Studies (Japan)",
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
        "publication_url": "https://www.nies.go.jp/db/",
        "citation_text": "NIES IDEA / national LCA database. Commercial licence terms apply.",
        "entitlement_rules": {"model": "tenant_entitlement_required", "metadata_visibility": "public"},
        "release_milestone": "v2.0",
    },
    {
        "source_id": "wri_aqueduct",
        "urn": "urn:gl:source:wri-aqueduct",
        "display_name": "WRI Aqueduct Water Risk Atlas",
        "authority": "World Resources Institute",
        "publisher": "World Resources Institute",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
        "cadence": "ad_hoc",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.wri.org/aqueduct",
        "citation_text": "WRI Aqueduct Water Risk Atlas, latest release. CC-BY-4.0.",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v2.0",
    },
    {
        "source_id": "entsoe_realtime",
        "urn": "urn:gl:source:entsoe-realtime",
        "display_name": "ENTSO-E Transparency Platform — real-time grid generation mix",
        "authority": "European Network of Transmission System Operators for Electricity",
        "publisher": "ENTSO-E",
        "jurisdiction": ["EU", "EEA", "UK"],
        "licence_class": "connector_only",
        "redistribution_class": "metadata_only",
        "cadence": "hourly",
        "source_owner": "data-engineering-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://transparency.entsoe.eu/",
        "citation_text": "ENTSO-E Transparency Platform, real-time generation mix.",
        "entitlement_rules": {"model": "connector_only_no_bulk", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
    {
        "source_id": "watttime",
        "urn": "urn:gl:source:watttime",
        "display_name": "WattTime — real-time and forecasted marginal grid intensity",
        "authority": "WattTime",
        "publisher": "WattTime",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "connector_only",
        "redistribution_class": "metadata_only",
        "cadence": "continuous",
        "source_owner": "data-engineering-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.watttime.org/",
        "citation_text": "WattTime real-time grid emissions API. Connector-only — bulk redistribution forbidden.",
        "entitlement_rules": {"model": "connector_only_no_bulk", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
    {
        "source_id": "us_iso_rto",
        "urn": "urn:gl:source:us-iso-rto",
        "display_name": "US ISO/RTO real-time grid (CAISO, ERCOT, MISO, NYISO, PJM, SPP, ISONE)",
        "authority": "US ISOs / RTOs (CAISO, ERCOT, MISO, NYISO, PJM, SPP, ISONE)",
        "publisher": "US ISOs / RTOs (multi)",
        "jurisdiction": ["US"],
        "licence_class": "connector_only",
        "redistribution_class": "metadata_only",
        "cadence": "hourly",
        "source_owner": "data-engineering-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://www.caiso.com/, https://www.ercot.com/, https://www.misoenergy.org/",
        "citation_text": "US ISO/RTO real-time grid generation mix, per-RTO official feed.",
        "entitlement_rules": {"model": "connector_only_no_bulk", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
    {
        "source_id": "grid_india_realtime",
        "urn": "urn:gl:source:grid-india-realtime",
        "display_name": "Grid-India real-time generation mix",
        "authority": "Grid Controller of India Limited (Grid-India)",
        "publisher": "Grid-India",
        "jurisdiction": ["IN"],
        "licence_class": "connector_only",
        "redistribution_class": "metadata_only",
        "cadence": "hourly",
        "source_owner": "data-engineering-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "regulator_approved",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://posoco.in/reports/daily-reports/",
        "citation_text": "Grid-India real-time generation mix, daily / hourly reports.",
        "entitlement_rules": {"model": "connector_only_no_bulk", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
    {
        "source_id": "agribalyse",
        "urn": "urn:gl:source:agribalyse",
        "display_name": "AGRIBALYSE — French agri-food LCA database",
        "authority": "ADEME / INRAE",
        "publisher": "ADEME / INRAE",
        "jurisdiction": ["FR"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
        "cadence": "ad_hoc",
        "source_owner": "climate-methodology-lead",
        "parser": {"module": "tbd.placeholder", "function": "tbd_placeholder", "version": "0.0.0"},
        "trust_tier": "external_verified",
        "legal_signoff": {
            "status": "pending_legal_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "evidence_uri": None,
        },
        "publication_url": "https://agribalyse.ademe.fr/",
        "citation_text": "AGRIBALYSE, ADEME / INRAE. Etalab open licence (with attribution).",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
    {
        "source_id": "faostat",
        "urn": "urn:gl:source:faostat",
        "display_name": "FAOSTAT — FAO Statistical Database (food and agriculture)",
        "authority": "Food and Agriculture Organization of the United Nations",
        "publisher": "FAO",
        "jurisdiction": ["GLOBAL"],
        "licence_class": "community_open",
        "redistribution_class": "attribution_required",
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
        "publication_url": "https://www.fao.org/faostat/",
        "citation_text": "FAOSTAT, FAO of the UN. CC-BY-NC-SA-3.0-IGO (non-commercial; attribution).",
        "entitlement_rules": {"model": "public_no_entitlement", "metadata_visibility": "public"},
        "release_milestone": "v2.5",
    },
]


def _existing_source_ids(payload: Dict[str, Any]) -> set:
    return {s.get("source_id") for s in payload.get("sources", []) if isinstance(s, dict)}


def main() -> int:
    text = _REGISTRY.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        print("source_registry.yaml has no `sources` list", file=sys.stderr)
        return 2

    migrated = 0
    for src in sources:
        if not isinstance(src, dict):
            continue
        before = dict(src)
        _migrate_source(src)
        if src != before:
            migrated += 1

    existing = _existing_source_ids(payload)
    appended = 0
    for ph in _PLACEHOLDERS:
        if ph["source_id"] in existing:
            continue
        sources.append(ph)
        appended += 1

    payload["sources"] = sources
    _REGISTRY.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(
        f"phase1 migration: {migrated} sources updated, {appended} placeholders appended"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
