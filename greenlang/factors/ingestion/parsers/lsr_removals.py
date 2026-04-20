# -*- coding: utf-8 -*-
"""
GHG Protocol Land Sector & Removals parser (Phase F9).

Input row shape::

    {
        "activity_id": "biochar-application-2024",
        "activity_type": "biochar_application",
        "removal_rate_kg_co2e_per_kg": -2.5,   # negative = removal
        "permanence_class": "long_term",        # decades / centuries / millenia
        "reversal_risk": "low",
        "geography": "EU",
        "year": 2024,
        "methodology": "GHG_Protocol_LSR"
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


def parse_lsr_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping LSR row %d: %s", i, exc)
    logger.info("Parsed %d LSR removal factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    activity = row["activity_id"]
    activity_type = row.get("activity_type", "land_sector")
    # GHGVectors require non-negative values; removals are signaled via
    # the `removal=True` tag + the activity_type (biochar_application etc.).
    # Downstream callers multiply by -1 when the tag is present.
    raw_rate = float(row["removal_rate_kg_co2e_per_kg"])
    removal_rate = abs(raw_rate)
    is_removal = raw_rate < 0
    geography = str(row.get("geography", "GLOBAL"))
    year = int(row.get("year", 2024))
    permanence = row.get("permanence_class", "short_term")
    reversal_risk = row.get("reversal_risk", "medium")

    factor_id = f"EF:{geography}:lsr:{activity}:{year}"

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=activity_type,
        unit="kg",
        geography=geography,
        geography_level=GeographyLevel.GLOBAL if geography == "GLOBAL" else GeographyLevel.COUNTRY,
        # Store the removal as negative CO2 so GWP aggregation flows the sign correctly.
        vectors=GHGVectors(CO2=removal_rate, CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_1,
        boundary=Boundary.CRADLE_TO_GRAVE,
        provenance=SourceProvenance(
            source_org="GHG Protocol Land Sector and Removals",
            source_publication="GHG Protocol LSR Standard",
            source_year=year,
            methodology=Methodology.HYBRID,
        ),
        valid_from=date(year, 1, 1),
        valid_to=date(year, 12, 31),
        uncertainty_95ci=0.40,
        dqs=DataQualityScore(
            temporal=3, geographical=3, technological=3,
            representativeness=3, methodological=4,
        ),
        license_info=LicenseInfo(
            license="GHG Protocol — attribution required",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="lsr_removals",
        source_release=f"LSR-{year}",
        source_record_id=activity,
        release_version=f"lsr-{year}",
        license_class="wri_wbcsd_terms",
        compliance_frameworks=["GHG_Protocol_LSR"],
        activity_tags=["land_sector", "removals", activity_type],
        sector_tags=["land_use"],
        tags=["lsr", activity_type, permanence, reversal_risk] + (
            ["removal"] if is_removal else []
        ),
        factor_family=FactorFamily.LAND_USE_REMOVALS.value,
        factor_name=f"LSR removal — {activity_type}",
        method_profile=MethodProfile.LAND_REMOVALS.value,
        factor_version="1.0.0",
        formula_type=FormulaType.CARBON_BUDGET.value,
        jurisdiction=Jurisdiction(country=geography if len(geography) == 2 else None),
        activity_schema=ActivitySchema(
            category="land_sector_removal",
            sub_category=activity_type,
            classification_codes=[],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope1", "scope3"],
            biogenic_share=1.0 if activity_type in ("biochar_application", "afforestation") else 0.0,
        ),
        verification=Verification(status=VerificationStatus.INTERNAL_REVIEW),
        explainability=Explainability(
            assumptions=[
                f"Permanence class: {permanence}.",
                f"Reversal risk: {reversal_risk}.",
                "LSR-aligned; biogenic carbon INCLUDED in the factor value.",
            ],
            fallback_rank=3,
            rationale=f"GHG Protocol LSR removal factor for {activity_type}, {geography}, {year}.",
        ),
        primary_data_flag=PrimaryDataFlag.PRIMARY_MODELED.value,
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


__all__ = ["parse_lsr_rows"]
