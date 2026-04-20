# -*- coding: utf-8 -*-
"""
AIB European Residual Mix parser (Phase F2.2).

Source: Association of Issuing Bodies (AIB) — ``https://www.aib-net.org/facts/european-residual-mix``.
AIB publishes the annual European Residual Mix per member country (EU-27
+ EEA). Under the GHG Protocol Scope 2 Quality Criteria, the residual
mix is what Scope 2 market-based accounting uses when no contractual
instrument (REC / GO / PPA) applies.

Publication cadence: annual, typically May/June (previous calendar year's data).

License class: ``registry_terms`` — redistribution permitted with
attribution; commercial embedding allowed if AIB terms are honoured.

Input row shape::

    {
        "country": "DE",
        "calendar_year": 2023,
        "residual_mix_g_co2_per_kwh": 498.0,
        "publication_date": "2024-06-15",
        "version": "AIB-2024-v1",
    }
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Iterable, List

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


def parse_aib_residual_mix_rows(
    rows: Iterable[Dict[str, Any]],
) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            record = _row_to_record(row)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping AIB row %d: %s", i, exc)
            continue
        out.append(record)
    logger.info("Parsed %d AIB residual-mix factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    country = str(row["country"]).upper()
    calendar_year = int(row["calendar_year"])
    g_per_kwh = float(row["residual_mix_g_co2_per_kwh"])
    kg_per_kwh = g_per_kwh / 1000.0
    version = str(row.get("version") or f"AIB-{calendar_year+1}-v1")

    valid_from = date(calendar_year, 1, 1)
    valid_to = date(calendar_year, 12, 31)

    factor_id = f"EF:{country}:residual_mix:{calendar_year}:{version.lower()}"

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="electricity",
        unit="kWh",
        geography=country,
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(CO2=kg_per_kwh, CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_2,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org="AIB (Association of Issuing Bodies)",
            source_publication="European Residual Mix",
            source_year=calendar_year + 1,          # published in year+1
            methodology=Methodology.HYBRID,
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.07,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5,
            technological=3,
            representativeness=5,
            methodological=5,
        ),
        license_info=LicenseInfo(
            license="AIB Residual Mix — redistribution with attribution",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="aib_residual_mix_eu",
        source_release=version,
        source_record_id=f"aib:{country}:{calendar_year}",
        release_version=version,
        license_class="registry_terms",
        compliance_frameworks=["GHG_Protocol_Scope2_MarketBased", "CSRD_E1", "IFRS_S2"],
        activity_tags=["purchased_electricity", "market_based", "residual_mix"],
        sector_tags=["power_sector", "utility"],
        tags=["aib", "residual_mix", "europe", str(calendar_year)],
        # ---- Canonical v2 ----
        factor_family=FactorFamily.RESIDUAL_MIX.value,
        factor_name=f"European residual mix — {country} {calendar_year}",
        method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET.value,
        factor_version="1.0.0",
        formula_type=FormulaType.RESIDUAL_MIX.value,
        jurisdiction=Jurisdiction(country=country),
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
            verified_by="AIB member Issuing Bodies",
        ),
        explainability=Explainability(
            assumptions=[
                "Use for Scope 2 market-based accounting when no supplier-specific "
                "contract (PPA, GO, REC) applies.",
                "Residual mix already nets out allocated attributes (tracked via "
                "Guarantees of Origin).",
            ],
            fallback_rank=5,
            rationale=(
                f"AIB European Residual Mix for {country}, calendar year "
                f"{calendar_year}. Applied per GHG Protocol Scope 2 Quality "
                f"Criteria when no contractual instrument is available."
            ),
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


__all__ = ["parse_aib_residual_mix_rows"]
