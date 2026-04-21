# -*- coding: utf-8 -*-
"""
Green-e Residual Mix parser — US + Canada (GAP-10 Wave 2).

Source: Green-e / Center for Resource Solutions, *Green-e Energy Residual
Mix Emission Rates* — https://www.green-e.org/residual-mix.  Annual
publication, typically mid-year (previous calendar year's data).  Values
are published as CO2, CH4, N2O rates per MWh at NERC subregion
granularity for the US and per-province for Canada (CER/ECCC sourced).

This parser mirrors the AIB European Residual Mix parser
(``aib_residual_mix.py``).  Under the GHG Protocol Scope 2 Quality
Criteria, a residual mix is used for market-based accounting when no
contractual instrument (REC, GO, I-REC, utility green tariff) applies.

License class: ``restricted`` — Green-e Terms require attribution and
prohibit commercial redistribution without a license.  This parser
therefore emits :class:`EmissionFactorRecord` instances tagged with
``RedistributionClass.RESTRICTED``.

Input row shape (one record per NERC subregion or Canadian province)::

    {
        "country": "US",                 # ISO 3166-1 alpha-2 (US | CA)
        "region": "NPCC",                # NERC region code (US) or province
        "subregion": "NEWE",             # NERC subregion / CER grid ID
        "calendar_year": 2023,
        "co2_lb_mwh": 521.0,             # lb CO2 per MWh
        "ch4_lb_mwh": 0.041,             # lb CH4 per MWh  (optional)
        "n2o_lb_mwh": 0.006,             # lb N2O per MWh  (optional)
        "version": "Green-e-2024-v1",    # optional
        "publication_date": "2024-07-15",# optional
    }

Alternative units also accepted: ``kg_co2_per_kwh`` / ``kg_ch4_per_kwh``
/ ``kg_n2o_per_kwh`` — useful when caller has already normalised upstream.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set

from greenlang.data.canonical_v2 import (
    ActivitySchema,
    ElectricityBasis,
    Explainability,
    FactorFamily,
    FactorParameters,
    FormulaType,
    Jurisdiction,
    MethodProfile,
    PrimaryDataFlag,
    RedistributionClass,
    UncertaintyDistribution,
    Verification,
    VerificationStatus,
)
from greenlang.data.emission_factor_record import (
    Boundary,
    DataQualityScore,
    EmissionFactorRecord,
    GeographyLevel,
    GHGVectors,
    GWPSet,
    GWPValues,
    LicenseInfo,
    Methodology,
    Scope,
    SourceProvenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

#: 1 lb = 0.453592 kg; 1 MWh = 1000 kWh -> lb/MWh -> kg/kWh factor.
_LB_MWH_TO_KG_KWH = 0.453592 / 1000.0

#: NERC regions that Green-e reports for the US — used for validation /
#: graceful handling of unknown region codes.
_KNOWN_US_NERC_REGIONS: Set[str] = {
    "ERCOT",  # Electric Reliability Council of Texas
    "FRCC",   # Florida Reliability Coordinating Council
    "MRO",    # Midwest Reliability Organization
    "NPCC",   # Northeast Power Coordinating Council
    "RFC",    # ReliabilityFirst Corporation
    "SERC",   # SERC Reliability Corporation
    "SPP",    # Southwest Power Pool
    "TRE",    # Texas Reliability Entity (legacy alias of ERCOT)
    "WECC",   # Western Electricity Coordinating Council
}

#: Canadian provinces / territories in CER (Canada Energy Regulator) data.
_KNOWN_CA_PROVINCES: Set[str] = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU",
    "ON", "PE", "QC", "SK", "YT",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce value to float, returning ``default`` on failure/None."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_vectors(row: Dict[str, Any]) -> GHGVectors:
    """Return a GHGVectors tuple in kg/kWh.

    Supports either native-kg fields or lb/MWh fields (auto-converts).
    """
    # Prefer explicit kg/kWh fields if present.
    if any(
        k in row
        for k in ("kg_co2_per_kwh", "kg_ch4_per_kwh", "kg_n2o_per_kwh")
    ):
        return GHGVectors(
            CO2=_safe_float(row.get("kg_co2_per_kwh")),
            CH4=_safe_float(row.get("kg_ch4_per_kwh")),
            N2O=_safe_float(row.get("kg_n2o_per_kwh")),
        )

    # Fall back to lb/MWh -> kg/kWh conversion.
    return GHGVectors(
        CO2=_safe_float(row.get("co2_lb_mwh")) * _LB_MWH_TO_KG_KWH,
        CH4=_safe_float(row.get("ch4_lb_mwh")) * _LB_MWH_TO_KG_KWH,
        N2O=_safe_float(row.get("n2o_lb_mwh")) * _LB_MWH_TO_KG_KWH,
    )


def _resolve_geography(
    country: str,
    region: Optional[str],
    subregion: Optional[str],
) -> tuple[str, Optional[str], GeographyLevel]:
    """Return ``(geography, grid_region, geography_level)``.

    - ``geography`` is the ISO-2 country code.
    - ``grid_region`` is the most specific NERC/CER identifier available.
    - ``geography_level`` distinguishes COUNTRY / GRID_ZONE / STATE.
    """
    country = country.upper()
    # Prefer subregion if provided (higher resolution).
    grid_region = (subregion or region or "").upper() or None
    if grid_region is None:
        return country, None, GeographyLevel.COUNTRY

    # Validate gracefully: unknown codes still parse but log a warning.
    if country == "US":
        top = (region or "").upper()
        if top and top not in _KNOWN_US_NERC_REGIONS:
            logger.warning(
                "Unknown US NERC region %s; emitting record anyway", top
            )
        return country, grid_region, GeographyLevel.GRID_ZONE

    if country == "CA":
        # Canadian province codes
        prov = (region or subregion or "").upper()
        if prov and prov not in _KNOWN_CA_PROVINCES:
            logger.warning(
                "Unknown Canadian province %s; emitting record anyway", prov
            )
        # CER publishes at province level -> treat as STATE for our taxonomy.
        return country, grid_region, GeographyLevel.STATE

    return country, grid_region, GeographyLevel.GRID_ZONE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_green_e_residual_rows(
    rows: Iterable[Dict[str, Any]],
) -> List[EmissionFactorRecord]:
    """Parse Green-e residual-mix rows into EmissionFactorRecord objects.

    Malformed rows are logged and skipped (never raise).
    """
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            record = _row_to_record(row)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping Green-e residual row %d: %s", i, exc)
            continue
        out.append(record)
    logger.info("Parsed %d Green-e residual-mix factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    country = str(row["country"]).upper()
    if country not in ("US", "CA"):
        raise ValueError(
            f"Green-e residual mix only covers US/CA, got country={country!r}"
        )

    calendar_year = int(row["calendar_year"])
    region = row.get("region")
    subregion = row.get("subregion")
    vectors = _extract_vectors(row)

    if vectors.CO2 <= 0:
        raise ValueError(
            "Green-e residual row missing positive CO2 vector; "
            f"country={country} region={region} subregion={subregion}"
        )

    geography, grid_region, geo_level = _resolve_geography(
        country, region, subregion
    )

    version = str(row.get("version") or f"Green-e-{calendar_year + 1}-v1")
    valid_from = date(calendar_year, 1, 1)
    valid_to = date(calendar_year, 12, 31)

    region_slug = (grid_region or country).lower().replace("-", "_")
    factor_id = (
        f"EF:{country}:residual_mix:{region_slug}:{calendar_year}:"
        f"{version.lower()}"
    )

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="electricity",
        unit="kWh",
        geography=geography,
        geography_level=geo_level,
        vectors=vectors,
        gwp_100yr=GWPValues(
            gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273
        ),
        scope=Scope.SCOPE_2,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org="Green-e / Center for Resource Solutions",
            source_publication="Green-e Energy Residual Mix",
            source_year=calendar_year + 1,
            methodology=Methodology.HYBRID,
            source_url="https://www.green-e.org/residual-mix",
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.08,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5 if grid_region else 3,
            technological=3,
            representativeness=5,
            methodological=5,
        ),
        license_info=LicenseInfo(
            license="Green-e Terms — attribution required; no commercial redistribution without license",
            redistribution_allowed=False,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        region_hint=grid_region,
        source_id="green_e_residual_mix",
        source_release=version,
        source_record_id=f"green_e:{country}:{grid_region or country}:{calendar_year}",
        release_version=version,
        license_class="restricted",
        compliance_frameworks=[
            "GHG_Protocol_Scope2_MarketBased",
            "CDP",
            "IFRS_S2",
            "SEC_Climate",
        ],
        activity_tags=[
            "purchased_electricity",
            "market_based",
            "residual_mix",
        ],
        sector_tags=["power_sector", "utility"],
        tags=[
            "green_e",
            "residual_mix",
            "north_america",
            country.lower(),
            str(calendar_year),
        ],
        # ---- Canonical v2 ----
        factor_family=FactorFamily.RESIDUAL_MIX.value,
        factor_name=(
            f"Green-e residual mix — {grid_region or country} {calendar_year}"
        ),
        method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET.value,
        factor_version="1.0.0",
        formula_type=FormulaType.RESIDUAL_MIX.value,
        jurisdiction=Jurisdiction(
            country=country,
            region=subregion if country == "CA" else None,
            grid_region=grid_region,
        ),
        activity_schema=ActivitySchema(
            category="purchased_electricity",
            sub_category="residual_mix",
            classification_codes=["NAICS:221112", "ISIC:D351"],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope2"],
            electricity_basis=ElectricityBasis.RESIDUAL_MIX,
            residual_mix_applicable=True,
            supplier_specific=False,
            transmission_loss_included=False,
        ),
        verification=Verification(
            status=VerificationStatus.EXTERNAL_VERIFIED,
            verified_by="Center for Resource Solutions",
        ),
        explainability=Explainability(
            assumptions=[
                "Use for Scope 2 market-based accounting in US/Canada when "
                "no supplier-specific contract (PPA, REC, I-REC, utility "
                "green tariff) applies.",
                "Residual mix already nets out certified renewable attributes "
                "tracked via the WREGIS / NEPOOL-GIS / M-RETS / PJM-GATS / "
                "ERCOT / Canadian CER provincial registries.",
                "Transmission and distribution losses NOT included; apply "
                "separately per GHG Protocol Scope 2 Guidance.",
            ],
            fallback_rank=5,
            rationale=(
                f"Green-e residual mix for {grid_region or country}, "
                f"calendar year {calendar_year}. Applied per GHG Protocol "
                f"Scope 2 Quality Criteria when no contractual instrument "
                f"is available."
            ),
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


__all__ = [
    "parse_green_e_residual_rows",
    "_KNOWN_US_NERC_REGIONS",
    "_KNOWN_CA_PROVINCES",
]
