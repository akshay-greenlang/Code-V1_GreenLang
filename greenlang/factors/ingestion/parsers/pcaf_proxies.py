# -*- coding: utf-8 -*-
"""
PCAF financed-emissions proxy parser (Phase F9).

PCAF (Partnership for Carbon Accounting Financials) publishes sector-
level emission intensities used for financed-emissions attribution.
Each proxy row is a sector × geography × asset_class tuple.

Input row shape::

    {
        "asset_class": "listed_equity",
        "sector_nace": "D35.11",
        "geography": "EU",
        "intensity_tco2e_per_m_eur_revenue": 450.0,
        "pcaf_dqs_score": 4,
        "year": 2024
    }
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Iterable, List

from greenlang.data.canonical_v2 import (
    ActivitySchema,
    Explainability,
    FactorFamily,
    FactorParameters,
    FormulaType,
    Jurisdiction,
    MethodProfile,
    PrimaryDataFlag,
    RedistributionClass,
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


def parse_pcaf_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping PCAF row %d: %s", i, exc)
    logger.info("Parsed %d PCAF proxy factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    asset = str(row["asset_class"])
    nace = str(row.get("sector_nace", "unknown"))
    geography = str(row.get("geography", "GLOBAL"))
    intensity = float(row["intensity_tco2e_per_m_eur_revenue"])
    # Store as kg CO2e per EUR revenue; callers multiply by revenue (EUR).
    kg_per_eur = intensity * 1000.0 / 1_000_000.0       # tCO2e/M EUR → kg/EUR
    year = int(row.get("year", 2024))
    dqs = int(row.get("pcaf_dqs_score", 3))             # 1 (best) .. 5 (worst)

    factor_id = f"EF:{geography}:pcaf:{asset}:{nace}:{year}"

    # DQS on the PCAF 1-5 scale; map to our 5-point schema.
    # (Note: PCAF uses 1 = best, our DataQualityScore uses 5 = best — invert.)
    inverted = 6 - max(1, min(5, dqs))

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="finance_proxy",
        unit="EUR",
        geography=geography,
        geography_level=GeographyLevel.GLOBAL if geography == "GLOBAL" else GeographyLevel.COUNTRY,
        vectors=GHGVectors(CO2=kg_per_eur, CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.CRADLE_TO_GRAVE,
        provenance=SourceProvenance(
            source_org="PCAF (Partnership for Carbon Accounting Financials)",
            source_publication="PCAF Global GHG Accounting & Reporting Standard",
            source_year=year,
            methodology=Methodology.SPEND_BASED,
        ),
        valid_from=date(year, 1, 1),
        valid_to=date(year, 12, 31),
        uncertainty_95ci=0.35,                          # PCAF proxies are inherently fuzzy
        dqs=DataQualityScore(
            temporal=inverted,
            geographical=inverted,
            technological=max(1, inverted - 1),
            representativeness=inverted,
            methodological=5,
        ),
        license_info=LicenseInfo(
            license="PCAF terms — attribution required",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="pcaf_proxies",
        source_release=f"PCAF-{year}",
        source_record_id=f"{asset}:{nace}:{geography}",
        release_version=f"pcaf-{year}",
        license_class="registry_terms",
        compliance_frameworks=["PCAF", "GHG_Protocol_Scope3_Cat15"],
        activity_tags=["financed_emissions", asset, nace],
        sector_tags=[nace],
        tags=["pcaf", asset, nace],
        factor_family=FactorFamily.FINANCE_PROXY.value,
        factor_name=f"PCAF {asset} intensity — {nace}, {geography}",
        method_profile=MethodProfile.FINANCE_PROXY.value,
        factor_version="1.0.0",
        formula_type=FormulaType.SPEND_PROXY.value,
        jurisdiction=Jurisdiction(country=geography if len(geography) == 2 else None),
        activity_schema=ActivitySchema(
            category="financed_emissions",
            sub_category=asset,
            classification_codes=[f"NACE:{nace}"],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope3"],
        ),
        verification=Verification(status=VerificationStatus.INTERNAL_REVIEW),
        explainability=Explainability(
            assumptions=[
                f"PCAF data-quality score {dqs}/5 (1 = best, 5 = worst).",
                "Attribution method: EVIC or revenue share per PCAF Standard §5.2.",
            ],
            fallback_rank=6,
            rationale=f"PCAF proxy — {asset}, NACE {nace}, {geography}, {year}.",
        ),
        primary_data_flag=PrimaryDataFlag.PROXY.value,
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


__all__ = ["parse_pcaf_rows"]
