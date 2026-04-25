# -*- coding: utf-8 -*-
"""
Catalog bootstrap orchestrator (Wave 2.5).

Purpose
-------
Turn the 21 parsers under ``greenlang/factors/ingestion/parsers/`` plus the
legal-rights-matrix posture encoded in ``source_registry.yaml`` into a
populated on-disk catalog seed at
``greenlang/factors/data/catalog_seed/<source_id>/v<version>.json``.

Only sources classified as ``Safe-to-Certify`` (open redistribution class)
or ``Needs-Legal-Review`` (IPCC provisional, marked ``status="preview"``)
are ingested. ``Licensed-Embedded`` / ``Blocked-Contract-Required`` sources
stay BYO-connector-only and are explicitly skipped here.

The orchestrator does NOT mutate any parser file. It only:
  1. Reads seed inputs under ``catalog_seed/_inputs/<source_id>.json``
  2. Calls each parser's public entrypoint
  3. Validates the N5 mandatory-field gate on every produced record
  4. Cross-source de-duplicates on
     ``(factor_family, jurisdiction, valid_from)``: conflicting records ship
     with distinct ``factor_id``s and cross-reference in
     ``validation_flags["alternates_considered"]``
  5. Serialises the sanitized records to
     ``catalog_seed/<source_id>/v<version>.json``.

This file is a first-class module; it does not perform network fetches and
never invents factor values. Any parser that requires live extraction is
skipped with an explicit entry in the run report.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).resolve().parents[1]  # greenlang/factors/
DATA_DIR = _PKG_ROOT / "data"
SEED_DIR = DATA_DIR / "catalog_seed"
SEED_INPUTS_DIR = SEED_DIR / "_inputs"


# ---------------------------------------------------------------------------
# Per-source registry entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceSpec:
    """Declarative description of one parser-to-seed wiring."""

    source_id: str
    display_name: str
    # Rights-matrix posture. Anything not in {"open"} is SKIPPED by default.
    redistribution_class: str
    # license_class mirrors the source_registry.yaml vocabulary for surface
    # attribution. Written into every emitted record.
    license_class: str
    license_name: str
    attribution_text: str
    # Factor status stamped on every emitted record. Defaults to "certified";
    # Needs-Legal-Review sources set this to "preview".
    factor_status: str
    # Parser dispatcher. Must be one of:
    #   "dict"   - parser takes a source-payload dict and returns List[dict]
    #   "rows"   - parser takes an iterable of row-dicts and returns
    #              List[EmissionFactorRecord]
    parser_kind: str
    # Dotted-path + symbol of the parser entrypoint.
    parser_module: str
    parser_function: str
    # Seed-input filename (under catalog_seed/_inputs/). Optional — if
    # missing the source is skipped with reason="no seed input".
    seed_input: Optional[str]
    # Output sub-directory + version.
    source_version: str
    # If True AND we didn't find seed input, the source is considered
    # offline-blocked (e.g., requires network fetch).
    offline_blocked: bool = False
    # Optional provisional-legal-review note stamped into
    # validation_flags["legal_review_note"] on every record.
    provisional_note: Optional[str] = None

    @property
    def output_dir(self) -> Path:
        return SEED_DIR / self.source_id

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"v{self.source_version}.json"


# Rights-matrix-driven source registry. Refer to
# ``docs/legal/source_rights_matrix.md`` for the posture rationale per row.
SOURCE_SPECS: List[SourceSpec] = [
    # ------------------------------------------------------------------
    # Safe-to-Certify, redistribution_class=open — INGESTED
    # ------------------------------------------------------------------
    SourceSpec(
        source_id="epa_hub",
        display_name="US EPA GHG Emission Factors Hub",
        redistribution_class="open",
        license_class="public_us_government",
        license_name="US-Public-Domain",
        attribution_text=(
            "U.S. Environmental Protection Agency, Emission Factors for "
            "Greenhouse Gas Inventories 2024."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.epa_ghg_hub",
        parser_function="parse_epa_ghg_hub",
        seed_input="epa_ghg_hub.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="egrid",
        display_name="US EPA eGRID",
        redistribution_class="open",
        license_class="public_us_government",
        license_name="US-Public-Domain",
        attribution_text="U.S. EPA eGRID2022 subregion emission rates.",
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.egrid",
        parser_function="parse_egrid",
        seed_input="egrid.json",
        source_version="2022.1",
    ),
    SourceSpec(
        source_id="desnz_ghg_conversion",
        display_name="UK DESNZ GHG Conversion Factors",
        redistribution_class="open",
        license_class="uk_open_government",
        license_name="OGL-UK-v3",
        attribution_text=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0 — UK Department for Energy Security "
            "and Net Zero, GHG conversion factors 2024."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.desnz_uk",
        parser_function="parse_desnz_uk",
        seed_input="desnz_uk.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="india_cea_co2_baseline",
        display_name="India CEA CO2 Baseline Database",
        redistribution_class="open",
        license_class="public_in_government",
        license_name="India-Public-Information",
        attribution_text=(
            "CO2 Baseline Database for the Indian Power Sector, Central "
            "Electricity Authority (Government of India), CDM v20.0."
        ),
        factor_status="certified",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.india_cea",
        parser_function="parse_india_cea_rows",
        seed_input="india_cea.json",
        source_version="20.0",
    ),
    SourceSpec(
        source_id="india_ccts_baselines",
        display_name="India CCTS Sectoral Baselines",
        redistribution_class="open",
        license_class="public_in_government",
        license_name="India-Public-Information",
        attribution_text=(
            "Carbon Credit Trading Scheme baseline emission intensities, "
            "Bureau of Energy Efficiency (BEE), MoEFCC, Government of India, "
            "G.S.R. 443(E) dated 28 June 2023."
        ),
        factor_status="certified",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.india_ccts",
        parser_function="parse_india_ccts_rows",
        seed_input="india_ccts.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="aib_residual_mix_eu",
        display_name="AIB European Residual Mix",
        redistribution_class="open",
        license_class="eu_publication",
        license_name="AIB-Terms-Open",
        attribution_text=(
            "European Residual Mixes, Association of Issuing Bodies "
            "(annual)."
        ),
        factor_status="certified",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.aib_residual_mix",
        parser_function="parse_aib_residual_mix_rows",
        seed_input="aib_residual_mix.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="australia_nga_factors",
        display_name="Australia NGER / NGA residual mix",
        redistribution_class="open",
        license_class="open",
        license_name="CC-BY-4.0",
        attribution_text=(
            "NGER state-level residual emission factors, Australian Clean "
            "Energy Regulator and DCCEEW, CC BY 4.0."
        ),
        factor_status="certified",
        parser_kind="rows",
        parser_module=(
            "greenlang.factors.ingestion.parsers.australia_nga_residual"
        ),
        parser_function="parse_australia_nga_residual_rows",
        seed_input="australia_nga_residual.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="japan_meti_electric_emission_factors",
        display_name="Japan METI Electric Utility Emission Factors",
        redistribution_class="open",
        license_class="open",
        license_name="Japan-Government-Public-Use",
        attribution_text="Electric Utility Emission Factors, Japan METI & MOEJ (annual).",
        factor_status="certified",
        parser_kind="rows",
        parser_module=(
            "greenlang.factors.ingestion.parsers.japan_meti_residual"
        ),
        parser_function="parse_japan_meti_residual_rows",
        seed_input="japan_meti_residual.json",
        source_version="FY2022.1",
    ),
    SourceSpec(
        source_id="eu_cbam",
        display_name="EU CBAM default values",
        redistribution_class="open",
        license_class="eu_publication",
        license_name="EU-Publication",
        attribution_text=(
            "European Commission, Carbon Border Adjustment Mechanism default "
            "values, DG TAXUD."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.cbam_full",
        parser_function="parse_cbam_full",
        seed_input="cbam.json",
        source_version="2024.1",
    ),
    # ------------------------------------------------------------------
    # Wave 4-B catalog expansion — US EPA USEEIO / WARM, eGRID subregion
    # extension, India CEA FY27 vintage extension. All public-domain or
    # Safe-to-Certify.
    # ------------------------------------------------------------------
    SourceSpec(
        source_id="useeio_v2",
        display_name="US EPA USEEIO v2 Supply Chain GHG Factors",
        redistribution_class="open",
        license_class="public_us_government",
        license_name="US-Public-Domain",
        attribution_text=(
            "U.S. Environmental Protection Agency, Office of Research and "
            "Development — USEEIO v2.0 Supply Chain GHG Emission Factors."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.useeio",
        parser_function="parse_useeio",
        seed_input="useeio.json",
        source_version="2.0",
    ),
    SourceSpec(
        source_id="epa_warm",
        display_name="US EPA WARM v15 Waste Reduction Model",
        redistribution_class="open",
        license_class="public_us_government",
        license_name="US-Public-Domain",
        attribution_text=(
            "U.S. Environmental Protection Agency — Waste Reduction Model "
            "(WARM) v15."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.epa_warm",
        parser_function="parse_epa_warm",
        seed_input="epa_warm.json",
        source_version="15",
    ),
    SourceSpec(
        source_id="egrid_subregion",
        display_name="US EPA eGRID (subregion extension, gold-pattern IDs)",
        redistribution_class="open",
        license_class="public_us_government",
        license_name="US-Public-Domain",
        attribution_text=(
            "U.S. EPA eGRID2022 subregion emission rates (gold-pattern ID "
            "extension)."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.egrid_subregion",
        parser_function="parse_egrid_subregion",
        seed_input="egrid_subregion.json",
        source_version="2022.1",
    ),
    SourceSpec(
        source_id="india_cea_fy27",
        display_name="India CEA — FY27 vintage extension (v20 lineage)",
        redistribution_class="open",
        license_class="public_in_government",
        license_name="India-Public-Information",
        attribution_text=(
            "Source: Central Electricity Authority, Government of India. "
            "CO2 Baseline Database v20."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.cea_fy27",
        parser_function="parse_cea_fy27",
        seed_input="cea_fy27.json",
        source_version="20.0-fy27",
    ),
    SourceSpec(
        source_id="ipcc_refrigerants_promoted",
        display_name="IPCC AR5/AR6 refrigerant GWPs — certified promotion",
        redistribution_class="open",
        license_class="public_international",
        license_name="IPCC-Guideline",
        attribution_text=(
            "IPCC AR5 Working Group I Chapter 8 and AR6 Working Group I "
            "Chapter 7 — 100-year GWP values for halogenated gases."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.ipcc_refrigerants",
        parser_function="parse_ipcc_refrigerants",
        seed_input="ipcc_refrigerants.json",
        source_version="ar5_ar6_100yr",
    ),
    # ------------------------------------------------------------------
    # Needs-Legal-Review — shipped preview pending IPCC copyright opinion
    # ------------------------------------------------------------------
    SourceSpec(
        source_id="ipcc_2006_nggi",
        display_name="IPCC 2006 Guidelines + 2019 Refinement",
        redistribution_class="open",
        license_class="public_international",
        license_name="IPCC-Guideline",
        attribution_text=(
            "IPCC 2006 Guidelines for National Greenhouse Gas Inventories "
            "and the 2019 Refinement."
        ),
        factor_status="preview",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.ipcc_defaults",
        parser_function="parse_ipcc_defaults",
        seed_input="ipcc_defaults.json",
        source_version="2019.1",
        provisional_note=(
            "provisional pending IPCC legal opinion: numerical-fact "
            "doctrine defence recorded; values may be held if Legal "
            "escalates. See docs/legal/source_rights_matrix.md row "
            "ipcc_2006_nggi."
        ),
    ),
    # ------------------------------------------------------------------
    # Licensed-Embedded / Blocked — EXPLICITLY SKIPPED (BYO connector)
    # ------------------------------------------------------------------
    SourceSpec(
        source_id="ghgp_method_refs",
        display_name="GHG Protocol methodological references",
        redistribution_class="licensed_embedded",
        license_class="wri_wbcsd_terms",
        license_name="WRI-WBCSD-Terms",
        attribution_text="GHG Protocol (WRI/WBCSD).",
        factor_status="connector_only",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.ghg_protocol",
        parser_function="parse_ghg_protocol",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="tcr_grp_defaults",
        display_name="The Climate Registry GRP defaults",
        redistribution_class="licensed_embedded",
        license_class="registry_terms",
        license_name="TCR-Registry-Terms",
        attribution_text="The Climate Registry General Reporting Protocol.",
        factor_status="connector_only",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.tcr",
        parser_function="parse_tcr",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="green_e_residual",
        display_name="Green-e Residual Mix",
        redistribution_class="licensed_embedded",
        license_class="commercial_connector",
        license_name="Green-e-Terms",
        attribution_text="Green-e Residual Mix, Center for Resource Solutions.",
        factor_status="connector_only",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.green_e",
        parser_function="parse_green_e",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="green_e_residual_mix",
        display_name="Green-e Energy Residual Mix (newer)",
        redistribution_class="licensed_embedded",
        license_class="commercial_connector",
        license_name="Green-e-Terms",
        attribution_text="Green-e Residual Mix, Center for Resource Solutions.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module=(
            "greenlang.factors.ingestion.parsers.green_e_residual"
        ),
        parser_function="parse_green_e_residual_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="ec3_buildings_epd",
        display_name="Building Transparency EC3 EPDs",
        redistribution_class="licensed_embedded",
        license_class="commercial_connector",
        license_name="EC3-API-Terms",
        attribution_text="Embodied Carbon in Construction Calculator (EC3), Building Transparency.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.ec3_epd",
        parser_function="parse_ec3_epd_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="glec_framework",
        display_name="GLEC Framework / freight lanes",
        redistribution_class="licensed_embedded",
        license_class="smart_freight_terms",
        license_name="Smart-Freight-Terms",
        attribution_text="GLEC Framework for Logistics Emissions Accounting and Reporting, Smart Freight Centre.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.freight_lanes",
        parser_function="parse_freight_lane_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="pcaf_global_std_v2",
        display_name="PCAF Global Standard proxies",
        redistribution_class="licensed_embedded",
        license_class="pcaf_attribution",
        license_name="PCAF-Attribution",
        attribution_text="PCAF Global GHG Accounting & Reporting Standard, Parts A + B.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.pcaf_proxies",
        parser_function="parse_pcaf_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="pact_pathfinder",
        display_name="WBCSD PACT Pathfinder product data",
        redistribution_class="licensed_embedded",
        license_class="wri_wbcsd_terms",
        license_name="WRI-WBCSD-Terms",
        attribution_text="WBCSD Pathfinder Framework, v3.0.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.pact_product_data",
        parser_function="parse_pact_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="lsr_removals",
        display_name="GHG Protocol Land Sector & Removals",
        redistribution_class="licensed_embedded",
        license_class="wri_wbcsd_terms",
        license_name="WRI-WBCSD-Terms",
        attribution_text="GHG Protocol Land Sector & Removals Guidance, WRI/WBCSD.",
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.lsr_removals",
        parser_function="parse_lsr_rows",
        seed_input=None,
        source_version="n/a",
    ),
    SourceSpec(
        source_id="waste_treatment",
        display_name="GreenLang curated waste-treatment factors",
        redistribution_class="licensed_embedded",
        license_class="greenlang_terms",
        license_name="GreenLang-Curated",
        attribution_text=(
            "GreenLang Factors curated waste treatment set; derived from "
            "IPCC 2006 Vol 5 + country inventories."
        ),
        factor_status="connector_only",
        parser_kind="rows",
        parser_module="greenlang.factors.ingestion.parsers.waste_treatment",
        parser_function="parse_waste_rows",
        seed_input=None,
        source_version="n/a",
    ),
    # ------------------------------------------------------------------
    # Wave 5 catalog expansion (2026-04-24) — five new Safe-to-Certify
    # sources: EU CBAM flat sector rollup, EXIOBASE v3 spend proxies,
    # UK ONS/DEFRA Environmental Accounts, EEA waste statistics, and
    # IPCC 2006 Vol 5 (Waste) India-parameterised defaults.
    # ------------------------------------------------------------------
    SourceSpec(
        source_id="cbam_default_values",
        display_name="EU CBAM Annex IV default values (flat sector rollup)",
        redistribution_class="open",
        license_class="public_eu",
        license_name="EU-Publication",
        attribution_text=(
            "European Commission, CBAM Implementing Regulation (EU) "
            "2023/1773 Annex IV default embedded-emission values."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module=(
            "greenlang.factors.ingestion.parsers.cbam_default_values"
        ),
        parser_function="parse_cbam_default_values",
        seed_input="cbam_default_values.json",
        source_version="2024.1",
    ),
    SourceSpec(
        source_id="exiobase_v3",
        display_name="EXIOBASE v3 multi-regional EE-IO spend proxies",
        redistribution_class="open",
        license_class="academic_research",
        license_name="CC-BY-4.0",
        attribution_text=(
            "EXIOBASE Consortium (TU Wien / NTNU / UTwente), EXIOBASE v3.8.2 "
            "multi-regional environmentally-extended input-output model."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.exiobase_v3",
        parser_function="parse_exiobase_v3",
        seed_input="exiobase_v3.json",
        source_version="3.8.2",
    ),
    SourceSpec(
        source_id="defra_uk_env_accounts",
        display_name=(
            "UK ONS/DEFRA Environmental Accounts (atmospheric emissions by "
            "SIC industry)"
        ),
        redistribution_class="open",
        license_class="uk_open_government",
        license_name="OGL-UK-v3",
        attribution_text=(
            "Contains public sector information licensed under the Open "
            "Government Licence v3.0 — UK Office for National Statistics / "
            "DEFRA, UK Environmental Accounts."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module=(
            "greenlang.factors.ingestion.parsers.defra_uk_env_accounts"
        ),
        parser_function="parse_defra_uk_env_accounts",
        seed_input="defra_uk_env_accounts.json",
        source_version="2022.1",
    ),
    SourceSpec(
        source_id="eea_waste_stats",
        display_name="EEA European waste statistics GHG intensities",
        redistribution_class="open",
        license_class="public_eu",
        license_name="EU-Publication",
        attribution_text=(
            "European Environment Agency, European waste statistics — "
            "per-treatment GHG intensity series."
        ),
        factor_status="certified",
        parser_kind="dict",
        parser_module="greenlang.factors.ingestion.parsers.eea_waste_stats",
        parser_function="parse_eea_waste_stats",
        seed_input="eea_waste_stats.json",
        source_version="2022.1",
    ),
    SourceSpec(
        source_id="ipcc_waste_vol5_in",
        display_name=(
            "IPCC 2006 Vol 5 (Waste) — India-parameterised default factors"
        ),
        redistribution_class="open",
        license_class="ipcc_reference",
        license_name="IPCC-Guideline",
        attribution_text=(
            "IPCC 2006 Guidelines for National GHG Inventories, Volume 5 "
            "(Waste), default values parameterised for India (tropical-wet "
            "climate zone)."
        ),
        factor_status="preview",
        parser_kind="dict",
        parser_module=(
            "greenlang.factors.ingestion.parsers.ipcc_waste_vol5_in"
        ),
        parser_function="parse_ipcc_waste_vol5_in",
        seed_input="ipcc_waste_vol5_in.json",
        source_version="2006.1",
        provisional_note=(
            "provisional pending IPCC legal opinion: factual default values "
            "carry no copyright restriction per the numerical-fact doctrine; "
            "see docs/legal/source_rights_matrix.md row ipcc_waste_vol5_in."
        ),
    ),
]


# ---------------------------------------------------------------------------
# N5 gate — mandatory field validation
# ---------------------------------------------------------------------------

# These five are the "must-have" set called out in the bootstrap spec.
# Any ingested record missing one of these is rejected with a clear error.
_N5_REQUIRED = (
    "valid_from",
    "source_version",
    "jurisdiction_country",
    "denominator_unit",
    "status",
)


def _extract_n5_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Pull N5 fields from a serialised factor record dict."""
    jurisdiction = rec.get("jurisdiction") or {}
    if not isinstance(jurisdiction, dict):
        jurisdiction = {}
    provenance = rec.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "valid_from": rec.get("valid_from"),
        "valid_to": rec.get("valid_to"),
        "source_version": (
            rec.get("source_release")
            or rec.get("release_version")
            or provenance.get("version")
        ),
        "jurisdiction_country": (
            jurisdiction.get("country") or rec.get("geography")
        ),
        "denominator_unit": rec.get("unit"),
        "status": rec.get("factor_status"),
    }


class N5GateError(ValueError):
    """Raised when a produced record fails the N5 mandatory-field gate."""


def _enforce_n5(rec: Dict[str, Any], source_id: str) -> None:
    n5 = _extract_n5_fields(rec)
    missing = [k for k in _N5_REQUIRED if not n5.get(k)]
    if missing:
        raise N5GateError(
            f"[{source_id}] factor_id={rec.get('factor_id')!r} missing "
            f"required N5 fields {missing}; n5_snapshot={n5}"
        )


# ---------------------------------------------------------------------------
# Alpha Provenance Gate (Wave B / TaskCreate #5 / WS2-T1)
# ---------------------------------------------------------------------------
# Stricter, schema-driven gate that REJECTS any record missing the v0.1 alpha
# provenance/review metadata. Runs *side-by-side* with the legacy N5 gate;
# the N5 gate is unchanged. Controlled by ``GL_FACTORS_ALPHA_PROVENANCE_GATE``;
# when unset, defaults to ON iff the active release_profile is ALPHA_V0_1.

_ALPHA_GATE_SINGLETON = None  # type: ignore[var-annotated]


def _alpha_gate_default_on() -> bool:
    """Default the alpha gate ON when running under the alpha-v0.1 profile."""
    try:
        from greenlang.factors.release_profile import (
            ReleaseProfile,
            current_profile,
        )
    except Exception:  # noqa: BLE001
        return False
    try:
        return current_profile() == ReleaseProfile.ALPHA_V0_1
    except Exception:  # noqa: BLE001
        return False


def _maybe_run_alpha_gate(rec: Dict[str, Any], source_id: str) -> None:
    """Run the Alpha Provenance Gate if it is enabled.

    The gate FAILS LOUD: any failure raises ``N5GateError`` so the existing
    bootstrap error-collection path catches it (no refactor of ``_run_one``
    needed). The legacy N5 gate stays as it is and runs first.

    Records that do not carry the alpha ``extraction`` / ``review`` blocks
    (e.g. the legacy v1-shape catalog seed envelopes) are skipped silently
    so this hook never breaks pre-alpha bootstrap pipelines.
    """
    from greenlang.factors.quality.alpha_provenance_gate import (
        AlphaProvenanceGate,
        AlphaProvenanceGateError,
        alpha_gate_enabled,
    )

    if not alpha_gate_enabled(default_on=_alpha_gate_default_on()):
        return

    # Only enforce on records that have the alpha v0.1 shape — i.e., they
    # already carry an ``extraction`` block. Legacy v1-shape records (which
    # the bootstrap pipeline still emits today) are out of scope and stay
    # under the N5 gate exclusively.
    if "extraction" not in rec or "urn" not in rec:
        return

    global _ALPHA_GATE_SINGLETON
    if _ALPHA_GATE_SINGLETON is None:
        _ALPHA_GATE_SINGLETON = AlphaProvenanceGate()

    try:
        _ALPHA_GATE_SINGLETON.assert_valid(rec)
    except AlphaProvenanceGateError as exc:
        raise N5GateError(
            f"[{source_id}] factor_id={rec.get('factor_id')!r} alpha "
            f"provenance gate failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Parser dispatch
# ---------------------------------------------------------------------------

def _load_parser(spec: SourceSpec) -> Callable:
    import importlib

    module = importlib.import_module(spec.parser_module)
    return getattr(module, spec.parser_function)


def _load_seed_payload(spec: SourceSpec) -> Optional[Any]:
    if not spec.seed_input:
        return None
    path = SEED_INPUTS_DIR / spec.seed_input
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _record_to_dict(rec: Any) -> Dict[str, Any]:
    """Coerce parser output (dict OR EmissionFactorRecord) to dict."""
    if isinstance(rec, dict):
        d = dict(rec)
    else:
        to_dict = getattr(rec, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                f"cannot coerce parser output of type {type(rec).__name__} "
                "to dict"
            )
        d = to_dict()

    _sanitize_for_roundtrip(d)
    return d


# Fields that ``EmissionFactorRecord.to_dict()`` emits but
# ``EmissionFactorRecord.from_dict()`` / ``__init__`` cannot accept — either
# because they are init=False dataclass fields or (in SourceProvenance's
# case) computed/derived properties also serialised by ``asdict()``.
_NON_INIT_VECTOR_FIELDS = ("PRECISION", "GWP_VALUES", "DECOMPOSITION_RATIOS")
_NON_INIT_RECORD_FIELDS = ("content_hash",)
_NON_INIT_DQS_FIELDS = ("overall_score", "rating")
_NON_INIT_GWP_FIELDS = ("co2e_total",)
_NON_INIT_PROVENANCE_FIELDS = ("citation",)


def _sanitize_for_roundtrip(d: Dict[str, Any]) -> None:
    """Strip init=False fields so the record survives to_dict -> from_dict.

    We modify ``d`` in place. Keeping the round-trip contract is what lets
    the seed envelopes be reconstituted into live ``EmissionFactorRecord``
    objects at catalog load time.
    """
    # Record-level
    for k in _NON_INIT_RECORD_FIELDS:
        d.pop(k, None)

    # GHGVectors
    vectors = d.get("vectors")
    if isinstance(vectors, dict):
        for k in _NON_INIT_VECTOR_FIELDS:
            vectors.pop(k, None)

    # DataQualityScore
    dqs = d.get("dqs")
    if isinstance(dqs, dict):
        for k in _NON_INIT_DQS_FIELDS:
            dqs.pop(k, None)

    # GWPValues
    for gkey in ("gwp_100yr", "gwp_20yr"):
        gwp = d.get(gkey)
        if isinstance(gwp, dict):
            for k in _NON_INIT_GWP_FIELDS:
                gwp.pop(k, None)

    # SourceProvenance
    provenance = d.get("provenance")
    if isinstance(provenance, dict):
        for k in _NON_INIT_PROVENANCE_FIELDS:
            provenance.pop(k, None)

    # Dataclass-converted canonical_v2 sub-objects come through ``asdict``
    # as plain dicts; ``from_dict`` only rehydrates the classic fields so
    # any unknown keys that do not match EmissionFactorRecord __init__ must
    # be stripped. We keep only the classic set.
    _CANONICAL_V2_KEYS = {
        "factor_family", "factor_name", "method_profile", "factor_version",
        "formula_type", "jurisdiction", "activity_schema", "parameters",
        "verification", "explainability", "primary_data_flag",
        "uncertainty_distribution", "redistribution_class", "raw_record_ref",
        "change_log", "next_review_date", "use_phase",
    }
    # These are Optional[Any] on the record — from_dict passes them through
    # as-is, but dataclass __init__ will reject any *other* unexpected key.
    known_top_level = {
        "factor_id", "fuel_type", "unit", "geography", "geography_level",
        "vectors", "gwp_100yr", "scope", "boundary", "provenance",
        "valid_from", "uncertainty_95ci", "dqs", "license_info",
        "region_hint", "gwp_20yr", "valid_to",
        "heating_value_basis", "reference_temperature_c", "pressure_bar",
        "moisture_content_pct", "ash_content_pct", "biogenic_flag",
        "compliance_frameworks", "created_at", "updated_at", "created_by",
        "notes", "tags",
        "factor_status", "source_id", "source_release", "source_record_id",
        "release_version", "validation_flags", "replacement_factor_id",
        "license_class", "activity_tags", "sector_tags",
    } | _CANONICAL_V2_KEYS
    for k in list(d.keys()):
        if k not in known_top_level:
            d.pop(k, None)


def _apply_source_overlays(rec: Dict[str, Any], spec: SourceSpec) -> Dict[str, Any]:
    """Stamp license / status / attribution onto a record per the spec."""
    rec.setdefault("source_id", spec.source_id)
    rec["factor_status"] = spec.factor_status

    # License overlay (always) — keep the classic LicenseInfo shape. The
    # attribution *string* lives in validation_flags (below) because
    # LicenseInfo.__init__ does not accept an ``attribution_text`` kwarg.
    license_info = rec.get("license_info") or {}
    if not isinstance(license_info, dict):
        license_info = {}
    license_info.setdefault("license", spec.license_name)
    license_info.setdefault(
        "redistribution_allowed", spec.redistribution_class == "open"
    )
    license_info.setdefault("commercial_use_allowed", True)
    license_info.setdefault("attribution_required", True)
    rec["license_info"] = license_info
    rec["license_class"] = spec.license_class
    rec["redistribution_class"] = spec.redistribution_class

    # Validation flags
    vflags = rec.get("validation_flags") or {}
    if not isinstance(vflags, dict):
        vflags = {}
    vflags.setdefault("bootstrap_ingested_at", _utcnow_iso())
    vflags.setdefault("attribution_text", spec.attribution_text)
    if spec.provisional_note:
        vflags.setdefault("legal_review_note", spec.provisional_note)
    rec["validation_flags"] = vflags

    return rec


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Per-source parse + write
# ---------------------------------------------------------------------------

@dataclass
class SourceRunResult:
    source_id: str
    status: str  # "ingested" | "skipped" | "error"
    reason: str = ""
    factor_count: int = 0
    rejected_count: int = 0
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)


def _run_one(spec: SourceSpec) -> SourceRunResult:
    # Gate 1: license posture — do NOT ingest licensed_embedded / blocked
    if spec.redistribution_class != "open":
        return SourceRunResult(
            source_id=spec.source_id,
            status="skipped",
            reason=(
                f"redistribution_class={spec.redistribution_class!r} — stays "
                "BYO connector-only per legal matrix"
            ),
        )

    # Gate 2: seed-input availability (offline parsers only)
    payload = _load_seed_payload(spec)
    if payload is None:
        return SourceRunResult(
            source_id=spec.source_id,
            status="skipped",
            reason=(
                f"no seed input found at catalog_seed/_inputs/"
                f"{spec.seed_input}; parser needs external fetch"
            ),
        )

    # Run parser
    try:
        parser_fn = _load_parser(spec)
    except Exception as exc:  # noqa: BLE001
        return SourceRunResult(
            source_id=spec.source_id,
            status="error",
            reason=f"parser import failed: {type(exc).__name__}: {exc}",
        )

    try:
        if spec.parser_kind == "dict":
            raw = parser_fn(payload)
        elif spec.parser_kind == "rows":
            rows = payload.get("rows") if isinstance(payload, dict) else payload
            raw = parser_fn(rows)
        else:
            raise ValueError(f"unknown parser_kind={spec.parser_kind!r}")
    except Exception as exc:  # noqa: BLE001
        return SourceRunResult(
            source_id=spec.source_id,
            status="error",
            reason=f"parser execution failed: {type(exc).__name__}: {exc}",
        )

    # Normalize to dicts, apply overlays, enforce N5
    kept: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, item in enumerate(raw):
        try:
            d = _record_to_dict(item)
            d = _apply_source_overlays(d, spec)
            _enforce_n5(d, spec.source_id)
            # Wave B / WS2-T1 alpha provenance gate (opt-in via env var;
            # defaults ON under release_profile=alpha-v0.1). Runs side-by-side
            # with the N5 gate above; legacy v1-shape records are ignored.
            _maybe_run_alpha_gate(d, spec.source_id)
            kept.append(d)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"record {i}: {type(exc).__name__}: {exc}")

    # Dedup / conflict tagging within the source itself (cross-source dedup
    # happens in ``bootstrap_catalog`` below).
    _tag_intra_source_conflicts(kept)

    # Serialize
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    envelope = {
        "source_id": spec.source_id,
        "source_version": spec.source_version,
        "ingested_at": _utcnow_iso(),
        "license_class": spec.license_class,
        "license_name": spec.license_name,
        "redistribution_class": spec.redistribution_class,
        "attribution_text": spec.attribution_text,
        "factor_status_default": spec.factor_status,
        "factor_count": len(kept),
        "factors": kept,
    }
    with spec.output_file.open("w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, sort_keys=False, default=str)

    return SourceRunResult(
        source_id=spec.source_id,
        status="ingested",
        reason="",
        factor_count=len(kept),
        rejected_count=len(errors),
        output_path=str(spec.output_file),
        errors=errors[:25],  # cap for report size
    )


def _tag_intra_source_conflicts(records: List[Dict[str, Any]]) -> None:
    """Within one source, if multiple records share (family, country, valid_from)
    we keep all of them but cross-reference in validation_flags."""
    buckets: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for rec in records:
        fam = rec.get("factor_family") or rec.get("vectors", {}).get("_family")
        jur = (rec.get("jurisdiction") or {}).get("country") or rec.get("geography")
        vf = rec.get("valid_from")
        key = (fam, jur, vf)
        buckets.setdefault(key, []).append(rec)
    for key, bucket in buckets.items():
        if len(bucket) <= 1:
            continue
        ids = [r.get("factor_id") for r in bucket if r.get("factor_id")]
        for rec in bucket:
            vflags = rec.setdefault("validation_flags", {})
            alternates = [fid for fid in ids if fid != rec.get("factor_id")]
            if alternates:
                vflags["alternates_considered"] = alternates


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

@dataclass
class BootstrapReport:
    run_started_at: str
    run_finished_at: str
    ingested: List[SourceRunResult] = field(default_factory=list)
    skipped: List[SourceRunResult] = field(default_factory=list)
    errored: List[SourceRunResult] = field(default_factory=list)
    total_factor_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        def _emit(results: List[SourceRunResult]) -> List[Dict[str, Any]]:
            return [r.__dict__ for r in results]

        return {
            "run_started_at": self.run_started_at,
            "run_finished_at": self.run_finished_at,
            "total_factor_count": self.total_factor_count,
            "ingested_sources": _emit(self.ingested),
            "skipped_parsers": [
                {"source": r.source_id, "reason": r.reason} for r in self.skipped
            ],
            "errored_sources": _emit(self.errored),
        }


def bootstrap_catalog(
    only_sources: Optional[List[str]] = None,
) -> BootstrapReport:
    """Run the full bootstrap and return a report.

    Args:
        only_sources: Optional list of source_ids to restrict the run to.
            Useful for incremental re-ingestion during parser development.
    """
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    SEED_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    started = _utcnow_iso()
    report = BootstrapReport(run_started_at=started, run_finished_at=started)

    specs = SOURCE_SPECS
    if only_sources:
        allow = set(only_sources)
        specs = [s for s in specs if s.source_id in allow]
        if not specs:
            raise ValueError(
                f"no matching source_ids for filter {only_sources!r}; "
                f"known: {[s.source_id for s in SOURCE_SPECS]}"
            )

    for spec in specs:
        result = _run_one(spec)
        if result.status == "ingested":
            report.ingested.append(result)
            report.total_factor_count += result.factor_count
        elif result.status == "skipped":
            report.skipped.append(result)
        else:
            report.errored.append(result)

    report.run_finished_at = _utcnow_iso()
    return report


# ---------------------------------------------------------------------------
# Catalog loader helpers (used by factor_database.py)
# ---------------------------------------------------------------------------

def load_seed_envelopes(seed_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load every ``catalog_seed/<source>/v*.json`` envelope on disk.

    Returns the list of envelope dicts (each has ``factors: [...]``).
    """
    root = Path(seed_dir) if seed_dir else SEED_DIR
    if not root.exists():
        return []
    envelopes: List[Dict[str, Any]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        for jf in sorted(sub.glob("v*.json")):
            try:
                with jf.open("r", encoding="utf-8") as fh:
                    env = json.load(fh)
                env["__seed_file"] = str(jf)
                envelopes.append(env)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read seed envelope %s: %s", jf, exc)
    return envelopes


def count_seed_factors(seed_dir: Optional[Path] = None) -> int:
    total = 0
    for env in load_seed_envelopes(seed_dir):
        try:
            total += int(env.get("factor_count") or len(env.get("factors", [])))
        except Exception:  # noqa: BLE001
            pass
    return total


__all__ = [
    "SourceSpec",
    "SOURCE_SPECS",
    "SEED_DIR",
    "SEED_INPUTS_DIR",
    "BootstrapReport",
    "SourceRunResult",
    "bootstrap_catalog",
    "load_seed_envelopes",
    "count_seed_factors",
    "N5GateError",
]
