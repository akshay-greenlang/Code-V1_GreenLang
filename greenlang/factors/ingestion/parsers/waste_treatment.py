# -*- coding: utf-8 -*-
"""
Waste treatment factors parser (Phase F9).

Input row shape::

    {
        "waste_type": "mixed_msw",
        "treatment": "landfill",
        "country": "US",
        "ch4_kg_per_t": 62.0,
        "co2_fossil_kg_per_t": 0.0,
        "biogenic_co2_kg_per_t": 320.0,
        "n2o_kg_per_t": 0.0,
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


def parse_waste_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping waste row %d: %s", i, exc)
    logger.info("Parsed %d waste-treatment factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    waste = row["waste_type"]
    treatment = row["treatment"]
    country = str(row.get("country", "GLOBAL"))
    year = int(row.get("year", 2024))

    co2 = float(row.get("co2_fossil_kg_per_t", 0.0))
    ch4 = float(row.get("ch4_kg_per_t", 0.0))
    n2o = float(row.get("n2o_kg_per_t", 0.0))
    bio = float(row.get("biogenic_co2_kg_per_t", 0.0))

    factor_id = f"EF:{country}:waste:{waste}:{treatment}:{year}"

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=waste,
        unit="t",
        geography=country,
        geography_level=GeographyLevel.COUNTRY if len(country) == 2 else GeographyLevel.GLOBAL,
        vectors=GHGVectors(
            CO2=co2,
            CH4=ch4,
            N2O=n2o,
            biogenic_CO2=bio,
        ),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.CRADLE_TO_GRAVE,
        provenance=SourceProvenance(
            source_org="IPCC 2006 Vol.5 / EPA WARM",
            source_publication=f"Waste treatment factors {year}",
            source_year=year,
            methodology=Methodology.IPCC_TIER_1,
        ),
        valid_from=date(year, 1, 1),
        valid_to=date(year, 12, 31),
        uncertainty_95ci=0.30,
        dqs=DataQualityScore(
            temporal=4, geographical=3, technological=3,
            representativeness=4, methodological=5,
        ),
        license_info=LicenseInfo(
            license="Public / attribution required",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="waste_treatment",
        source_release=f"IPCC-EPA-{year}",
        source_record_id=f"{waste}:{treatment}:{country}",
        release_version=f"waste-{year}",
        license_class="public_us_government",
        compliance_frameworks=["IPCC_2006", "GHG_Protocol_Scope3_Cat5", "EPA_WARM"],
        activity_tags=["waste", treatment, waste],
        sector_tags=["waste_management"],
        tags=["waste", treatment, country],
        factor_family=FactorFamily.WASTE_TREATMENT.value,
        factor_name=f"{waste} — {treatment} ({country})",
        method_profile=MethodProfile.CORPORATE_SCOPE3.value,
        factor_version="1.0.0",
        formula_type=FormulaType.DIRECT_FACTOR.value,
        jurisdiction=Jurisdiction(country=country if len(country) == 2 else None),
        activity_schema=ActivitySchema(
            category="waste_treatment",
            sub_category=treatment,
            classification_codes=[],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope3"],
            biogenic_share=bio / (co2 + bio) if (co2 + bio) > 0 else 0.0,
        ),
        verification=Verification(status=VerificationStatus.EXTERNAL_VERIFIED),
        explainability=Explainability(
            assumptions=[
                "CH4 and N2O reported as mass, aggregated via AR6-100 GWP.",
                "Biogenic CO2 reported separately per GHG Protocol Scope 3.",
            ],
            fallback_rank=5,
            rationale=f"Waste treatment factor for {waste} via {treatment} in {country}.",
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        redistribution_class=RedistributionClass.OPEN.value,
    )


__all__ = ["parse_waste_rows"]
