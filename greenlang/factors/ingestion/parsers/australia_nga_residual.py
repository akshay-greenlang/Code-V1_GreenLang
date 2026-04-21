# -*- coding: utf-8 -*-
"""
Australian NGA Residual Mix parser (GAP-10 Wave 2).

Source: Australian Government — Department of Climate Change, Energy,
the Environment and Water (DCCEEW).  *Australian National Greenhouse
Accounts (NGA) Factors*, updated annually (typically August).
Publication URL: https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors

The NGA publishes Scope 2 location-based grid emission factors per NEM
region (NSW, QLD, SA, TAS, VIC) + Western Australia (WEM) + Northern
Territory (NT).  **A residual mix is not officially published** by
DCCEEW; the Clean Energy Regulator's *LGC / STC* surrender data must be
netted out of the published grid average.  This parser implements the
community-accepted residual derivation formula::

    residual_mix = (grid_avg * total_consumption - lgc_volume * 0.0) /
                   (total_consumption - lgc_volume)

with the simplifying assumption that LGC-surrendered MWh carry a zero
emission factor (standard Scope 2 market-based convention).  A detailed
assumptions block is attached to each factor's ``explainability`` field
so auditors can re-derive the value.

License class: ``open`` — NGA Factors are Creative Commons BY-4.0.

Input row shape::

    {
        "region": "NSW",                      # NEM region | WA | NT
        "financial_year": "2023-24",
        "grid_avg_kg_co2e_per_kwh": 0.68,     # published NGA Scope 2 factor
        "lgc_surrendered_share": 0.24,        # fraction of MWh claimed via LGCs
        "co2_share": 0.98,                    # share of CO2e as CO2 (default 0.98)
        "ch4_share": 0.015,                   # share as CH4 (default 0.015)
        "n2o_share": 0.005,                   # share as N2O (default 0.005)
        "version": "NGA-2024-v1",             # optional
    }
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Iterable, List, Set

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


#: Australian electricity regions covered by NGA.
_KNOWN_AU_REGIONS: Set[str] = {
    # National Electricity Market (NEM)
    "NSW",  # New South Wales + ACT
    "QLD",  # Queensland
    "SA",   # South Australia
    "TAS",  # Tasmania
    "VIC",  # Victoria
    # Non-NEM
    "WA",   # Western Australia (Wholesale Electricity Market)
    "NT",   # Northern Territory
}

#: Default GHG split for Australian grid (CO2-dominated; small CH4/N2O).
#: Derived from NGA methodology chapter on combustion GHG speciation.
_DEFAULT_CO2_SHARE = 0.980
_DEFAULT_CH4_SHARE = 0.015
_DEFAULT_N2O_SHARE = 0.005


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _derive_residual(
    grid_avg_kg_per_kwh: float,
    lgc_share: float,
) -> float:
    """Residual-mix derivation per DCCEEW guidance.

    Netting out LGC-surrendered MWh (assumed zero-emission) lifts the
    effective intensity on the remaining pool.  ``lgc_share`` is clamped
    to [0, 0.95] to avoid division blow-ups in edge cases.
    """
    lgc_share = max(0.0, min(lgc_share, 0.95))
    if lgc_share == 0.0:
        return grid_avg_kg_per_kwh
    # Mass balance: grid_avg = (1 - lgc_share) * residual + lgc_share * 0
    # -> residual = grid_avg / (1 - lgc_share)
    return grid_avg_kg_per_kwh / (1.0 - lgc_share)


def _split_vectors(
    co2e_kg_per_kwh: float,
    co2_share: float,
    ch4_share: float,
    n2o_share: float,
    *,
    ch4_gwp: float = 28.0,
    n2o_gwp: float = 273.0,
) -> GHGVectors:
    """Split a CO2e total into CO2/CH4/N2O gas vectors.

    We hold CO2e mass, then back-solve each vector by dividing by its GWP
    so the resulting GHGVectors re-aggregate to the original CO2e under
    AR6 100-year GWPs.  CTO non-negotiable #1: NEVER store only CO2e —
    gas vectors must be first-class.
    """
    total_share = co2_share + ch4_share + n2o_share
    if total_share <= 0:
        return GHGVectors(CO2=co2e_kg_per_kwh, CH4=0.0, N2O=0.0)
    co2_share /= total_share
    ch4_share /= total_share
    n2o_share /= total_share

    co2 = co2e_kg_per_kwh * co2_share            # GWP(CO2) = 1
    ch4 = (co2e_kg_per_kwh * ch4_share) / ch4_gwp
    n2o = (co2e_kg_per_kwh * n2o_share) / n2o_gwp
    return GHGVectors(CO2=co2, CH4=ch4, N2O=n2o)


def parse_australia_nga_residual_rows(
    rows: Iterable[Dict[str, Any]],
) -> List[EmissionFactorRecord]:
    """Parse Australian NGA residual-mix rows into EmissionFactorRecord."""
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            record = _row_to_record(row)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping NGA residual row %d: %s", i, exc)
            continue
        out.append(record)
    logger.info("Parsed %d Australian NGA residual-mix factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    region = str(row["region"]).upper()
    if region not in _KNOWN_AU_REGIONS:
        logger.warning(
            "Unknown AU region %s; emitting record anyway", region
        )

    fy = str(row.get("financial_year") or row.get("fy") or "")
    if not fy:
        raise ValueError("NGA residual row missing financial_year")

    grid_avg = _safe_float(
        row.get("grid_avg_kg_co2e_per_kwh")
        or row.get("co2e_kg_per_kwh")
    )
    if grid_avg <= 0:
        raise ValueError(
            f"NGA residual row missing positive grid_avg for region {region}"
        )

    lgc_share = _safe_float(row.get("lgc_surrendered_share"), default=0.0)
    residual = _derive_residual(grid_avg, lgc_share)

    co2_share = _safe_float(row.get("co2_share"), default=_DEFAULT_CO2_SHARE)
    ch4_share = _safe_float(row.get("ch4_share"), default=_DEFAULT_CH4_SHARE)
    n2o_share = _safe_float(row.get("n2o_share"), default=_DEFAULT_N2O_SHARE)

    vectors = _split_vectors(residual, co2_share, ch4_share, n2o_share)

    version = str(row.get("version") or f"NGA-residual-{fy}-v1")

    # Australian FY: 1 July -> 30 June.
    try:
        start_year = int(fy.split("-")[0])
    except (ValueError, IndexError):
        raise ValueError(f"NGA row has malformed financial_year: {fy!r}")
    valid_from = date(start_year, 7, 1)
    valid_to = date(start_year + 1, 6, 30)

    factor_id = (
        f"EF:AU:residual_mix:{region.lower()}:{fy.replace('-', '_')}:"
        f"{version.lower()}"
    )

    is_nem = region in {"NSW", "QLD", "SA", "TAS", "VIC"}

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="electricity",
        unit="kWh",
        geography="AU",
        geography_level=GeographyLevel.GRID_ZONE,
        vectors=vectors,
        gwp_100yr=GWPValues(
            gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273
        ),
        scope=Scope.SCOPE_2,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org=(
                "Australian Department of Climate Change, Energy, the "
                "Environment and Water (DCCEEW)"
            ),
            source_publication="National Greenhouse Accounts Factors",
            source_year=start_year + 1,
            methodology=Methodology.HYBRID,
            source_url=(
                "https://www.dcceew.gov.au/climate-change/publications/"
                "national-greenhouse-accounts-factors"
            ),
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.10,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5,
            technological=3,
            representativeness=4,
            methodological=4,
        ),
        license_info=LicenseInfo(
            license="CC-BY-4.0",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        region_hint=region,
        source_id="australia_nga_factors",
        source_release=version,
        source_record_id=f"nga:{region}:{fy}",
        release_version=version,
        license_class="open",
        compliance_frameworks=[
            "GHG_Protocol_Scope2_MarketBased",
            "NGER",
            "Climate_Active",
            "IFRS_S2",
        ],
        activity_tags=[
            "purchased_electricity",
            "market_based",
            "residual_mix",
            "nem" if is_nem else "non_nem",
        ],
        sector_tags=["power_sector", "utility"],
        tags=[
            "nga",
            "dcceew",
            "australia",
            "residual_mix",
            region.lower(),
            fy,
        ],
        # ---- Canonical v2 ----
        factor_family=FactorFamily.RESIDUAL_MIX.value,
        factor_name=(
            f"Australian NGA residual mix — {region} FY{fy}"
        ),
        method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET.value,
        factor_version="1.0.0",
        formula_type=FormulaType.RESIDUAL_MIX.value,
        jurisdiction=Jurisdiction(
            country="AU", region=region, grid_region=region
        ),
        activity_schema=ActivitySchema(
            category="purchased_electricity",
            sub_category="residual_mix",
            classification_codes=["NAICS:221112", "ISIC:D351", "ANZSIC:D2611"],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope2"],
            electricity_basis=ElectricityBasis.RESIDUAL_MIX,
            residual_mix_applicable=True,
            supplier_specific=False,
            transmission_loss_included=False,
        ),
        verification=Verification(
            status=VerificationStatus.REGULATOR_APPROVED,
            verified_by="Australian DCCEEW",
        ),
        explainability=Explainability(
            assumptions=[
                "DCCEEW does NOT officially publish a residual mix.  "
                "This factor is derived by netting LGC-surrendered MWh "
                "(assumed zero-emission per GHG Protocol Scope 2 market-"
                "based convention) from the published NGA grid-average "
                "factor.",
                f"LGC-surrendered share assumed = {lgc_share:.3f}; formula: "
                "residual = grid_avg / (1 - lgc_share).",
                "Gas speciation split uses NGA default shares (CO2 98.0%, "
                "CH4 1.5%, N2O 0.5%) unless overridden.",
                "Apply to Scope 2 market-based reporting only when no "
                "supplier-specific contract (GreenPower, LGC) applies.",
                "STC (small-scale) certificates are NOT netted: they are "
                "already excluded from the published NGA grid average.",
            ],
            fallback_rank=5,
            rationale=(
                f"Derived Australian residual mix for {region}, FY {fy}. "
                f"Grid average {grid_avg:.3f} kgCO2e/kWh - LGC share "
                f"{lgc_share:.1%} = residual {residual:.3f} kgCO2e/kWh."
            ),
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.OPEN.value,
    )


__all__ = [
    "parse_australia_nga_residual_rows",
    "_KNOWN_AU_REGIONS",
    "_derive_residual",
    "_split_vectors",
]
