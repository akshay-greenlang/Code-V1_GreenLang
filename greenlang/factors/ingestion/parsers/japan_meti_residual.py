# -*- coding: utf-8 -*-
"""
Japan METI Residual Mix parser (GAP-10 Wave 2).

Source: Japan Ministry of Economy, Trade and Industry (METI) — annual
*Electric Utility Emission Factors* (電気事業者別排出係数) published
jointly with the Ministry of the Environment (MOEJ) under the Act on
Promotion of Global Warming Countermeasures.
Publication URL: https://ghg-santeikohyo.env.go.jp/calc

METI publishes two factors per utility service area:

    1. ``基礎排出係数`` (Basic emission factor) — location-based grid avg
    2. ``調整後排出係数`` (Adjusted emission factor) — location-based AFTER
       utility-level Non-Fossil Certificate (J-Credit, Green Power
       Certificate, 非化石証書) purchases are netted in

A proper *residual mix* is derived here as::

    residual_mix = basic_factor / (1 - system_level_non_fossil_share)

where ``system_level_non_fossil_share`` is the share of non-fossil-
certificated MWh claimed across Japan's tradeable certificate markets
(J-Credit, Green Value Certification, Non-Fossil Value Trading System).
This approximation mirrors the AIB / Green-e residual-mix formula
adapted to Japan's certificate market structure.

License class: ``open`` — METI publications are public-domain government
data (attribution required).

Input row shape::

    {
        "utility_area": "Tokyo",                  # 10 regional service areas
        "fiscal_year": "2022",                    # Japan FY (Apr-Mar)
        "basic_factor_kg_co2_per_kwh": 0.452,     # 基礎排出係数
        "adjusted_factor_kg_co2_per_kwh": 0.441,  # 調整後排出係数 (optional)
        "non_fossil_certificate_share": 0.18,     # J-Credit/NFC netted share
        "version": "METI-FY2022-v1",              # optional
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


#: Ten general electricity utility service areas in Japan.
_KNOWN_JP_UTILITIES: Set[str] = {
    "Hokkaido",   # 北海道電力
    "Tohoku",     # 東北電力
    "Tokyo",      # 東京電力
    "Chubu",      # 中部電力
    "Hokuriku",   # 北陸電力
    "Kansai",     # 関西電力
    "Chugoku",    # 中国電力
    "Shikoku",    # 四国電力
    "Kyushu",     # 九州電力
    "Okinawa",    # 沖縄電力
}

#: Japanese grid gas split — fossil-heavy with minimal CH4/N2O.
_DEFAULT_CO2_SHARE = 0.985
_DEFAULT_CH4_SHARE = 0.010
_DEFAULT_N2O_SHARE = 0.005


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _normalize_utility(name: str) -> str:
    """Return canonical casing (``Tokyo``, ``Kyushu``, ...)."""
    if not name:
        return ""
    n = name.strip()
    # Preserve case mapping for exact lookup
    lower_map = {u.lower(): u for u in _KNOWN_JP_UTILITIES}
    return lower_map.get(n.lower(), n.title())


def _derive_residual(
    basic_factor: float,
    non_fossil_share: float,
) -> float:
    """Residual-mix derivation netting out non-fossil-certificated MWh.

    Non-fossil certificates (J-Credit, Green Power Certificate, non-fossil
    value trading system) represent zero-emission claims.  Once these are
    subtracted from the system mix, the remaining (residual) pool carries
    a proportionally higher intensity::

        residual = basic_factor / (1 - non_fossil_share)
    """
    non_fossil_share = max(0.0, min(non_fossil_share, 0.95))
    if non_fossil_share == 0.0:
        return basic_factor
    return basic_factor / (1.0 - non_fossil_share)


def _split_vectors(
    co2e_kg_per_kwh: float,
    *,
    co2_share: float = _DEFAULT_CO2_SHARE,
    ch4_share: float = _DEFAULT_CH4_SHARE,
    n2o_share: float = _DEFAULT_N2O_SHARE,
    ch4_gwp: float = 28.0,
    n2o_gwp: float = 273.0,
) -> GHGVectors:
    """Split a CO2e total into CO2/CH4/N2O mass vectors (kg/kWh)."""
    total = co2_share + ch4_share + n2o_share
    if total <= 0:
        return GHGVectors(CO2=co2e_kg_per_kwh, CH4=0.0, N2O=0.0)
    co2_share /= total
    ch4_share /= total
    n2o_share /= total
    return GHGVectors(
        CO2=co2e_kg_per_kwh * co2_share,
        CH4=(co2e_kg_per_kwh * ch4_share) / ch4_gwp,
        N2O=(co2e_kg_per_kwh * n2o_share) / n2o_gwp,
    )


def parse_japan_meti_residual_rows(
    rows: Iterable[Dict[str, Any]],
) -> List[EmissionFactorRecord]:
    """Parse METI residual-mix rows into EmissionFactorRecord objects."""
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            record = _row_to_record(row)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping METI residual row %d: %s", i, exc)
            continue
        out.append(record)
    logger.info("Parsed %d Japan METI residual-mix factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    utility = _normalize_utility(str(row["utility_area"]))
    if utility not in _KNOWN_JP_UTILITIES:
        logger.warning(
            "Unknown JP utility %s; emitting record anyway", utility
        )

    fy = str(row.get("fiscal_year") or row.get("fy") or "")
    if not fy:
        raise ValueError("METI row missing fiscal_year")

    basic = _safe_float(
        row.get("basic_factor_kg_co2_per_kwh")
        or row.get("basic_factor")
    )
    if basic <= 0:
        raise ValueError(
            f"METI row missing positive basic_factor for utility {utility}"
        )

    nf_share = _safe_float(
        row.get("non_fossil_certificate_share")
        or row.get("non_fossil_share"),
        default=0.0,
    )
    residual = _derive_residual(basic, nf_share)

    co2_share = _safe_float(row.get("co2_share"), default=_DEFAULT_CO2_SHARE)
    ch4_share = _safe_float(row.get("ch4_share"), default=_DEFAULT_CH4_SHARE)
    n2o_share = _safe_float(row.get("n2o_share"), default=_DEFAULT_N2O_SHARE)

    vectors = _split_vectors(
        residual,
        co2_share=co2_share,
        ch4_share=ch4_share,
        n2o_share=n2o_share,
    )

    version = str(row.get("version") or f"METI-FY{fy}-v1")

    # Japanese fiscal year: 1 April year -> 31 March (year + 1).
    try:
        start_year = int(fy)
    except ValueError:
        try:
            start_year = int(fy.split("-")[0])
        except (ValueError, IndexError):
            raise ValueError(f"METI row has malformed fiscal_year: {fy!r}")
    valid_from = date(start_year, 4, 1)
    valid_to = date(start_year + 1, 3, 31)

    utility_slug = utility.lower()
    factor_id = (
        f"EF:JP:residual_mix:{utility_slug}:fy{start_year}:"
        f"{version.lower()}"
    )

    adjusted = _safe_float(row.get("adjusted_factor_kg_co2_per_kwh"))

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="electricity",
        unit="kWh",
        geography="JP",
        geography_level=GeographyLevel.GRID_ZONE,
        vectors=vectors,
        gwp_100yr=GWPValues(
            gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273
        ),
        scope=Scope.SCOPE_2,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org=(
                "Japan Ministry of Economy, Trade and Industry (METI) / "
                "Ministry of the Environment (MOEJ)"
            ),
            source_publication="Electric Utility Emission Factors",
            source_year=start_year + 1,
            methodology=Methodology.HYBRID,
            source_url="https://ghg-santeikohyo.env.go.jp/calc",
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.09,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5,
            technological=3,
            representativeness=5,
            methodological=4,
        ),
        license_info=LicenseInfo(
            license="Japan Government Open Data — public use with attribution",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        region_hint=utility,
        source_id="japan_meti_electric_emission_factors",
        source_release=version,
        source_record_id=f"meti:{utility}:FY{start_year}",
        release_version=version,
        license_class="open",
        compliance_frameworks=[
            "GHG_Protocol_Scope2_MarketBased",
            "TCFD",
            "IFRS_S2",
            "Japan_SSBJ",
        ],
        activity_tags=[
            "purchased_electricity",
            "market_based",
            "residual_mix",
        ],
        sector_tags=["power_sector", "utility"],
        tags=[
            "meti",
            "japan",
            "residual_mix",
            utility_slug,
            f"fy{start_year}",
        ],
        # ---- Canonical v2 ----
        factor_family=FactorFamily.RESIDUAL_MIX.value,
        factor_name=(
            f"Japan METI residual mix — {utility} FY{start_year}"
        ),
        method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET.value,
        factor_version="1.0.0",
        formula_type=FormulaType.RESIDUAL_MIX.value,
        jurisdiction=Jurisdiction(
            country="JP", region=utility, grid_region=utility
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
            status=VerificationStatus.REGULATOR_APPROVED,
            verified_by="Japan METI / MOEJ",
        ),
        explainability=Explainability(
            assumptions=[
                "METI publishes Basic (基礎排出係数) and Adjusted "
                "(調整後排出係数) factors.  The residual mix is derived "
                "by netting non-fossil-certificate surrenders (J-Credit, "
                "Green Value Certification, Non-Fossil Value Trading "
                "System) from the Basic factor.",
                f"Non-fossil certificate share assumed = {nf_share:.3f}.",
                "Formula: residual = basic_factor / (1 - non_fossil_share).",
                (
                    "Adjusted factor reported by METI is UTILITY-level, not "
                    "system-level; the residual here is system-wide so "
                    "values will differ from the adjusted factor"
                    + (
                        f" ({adjusted:.3f} kgCO2/kWh)" if adjusted else ""
                    )
                    + "."
                ),
                "Gas speciation split uses IPCC 2006 defaults "
                "(CO2 98.5%, CH4 1.0%, N2O 0.5%).",
                "Apply to Scope 2 market-based reporting when no supplier-"
                "specific contract or non-fossil certificate applies.",
            ],
            fallback_rank=5,
            rationale=(
                f"Japan METI residual mix for {utility}, FY{start_year}. "
                f"Basic factor {basic:.3f} kgCO2/kWh, non-fossil share "
                f"{nf_share:.1%} -> residual {residual:.3f} kgCO2/kWh."
            ),
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.OPEN.value,
    )


__all__ = [
    "parse_japan_meti_residual_rows",
    "_KNOWN_JP_UTILITIES",
    "_derive_residual",
    "_split_vectors",
    "_normalize_utility",
]
