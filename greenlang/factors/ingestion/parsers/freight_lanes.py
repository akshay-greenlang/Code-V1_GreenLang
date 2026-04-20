# -*- coding: utf-8 -*-
"""
Freight lane factors parser (Phase F9).

Per-mode, per-payload, per-lane emission factors aligned with GLEC
Framework v3 / ISO 14083:2023.

Input row shape::

    {
        "lane_id": "TRUCK-40T-EU-DIESEL",
        "mode": "road",
        "vehicle_class": "heavy_truck_40t",
        "fuel": "diesel",
        "payload_utilization": 0.7,
        "empty_running_factor": 1.25,
        "wtt_gco2e_per_tkm": 21.0,
        "ttw_gco2e_per_tkm": 75.0,
        "wtw_gco2e_per_tkm": 96.0,
        "valid_year": 2024,
        "geography": "EU"
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


def parse_freight_lane_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping freight-lane row %d: %s", i, exc)
    logger.info("Parsed %d freight-lane factors", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    lane_id = row["lane_id"]
    mode = row["mode"]
    fuel = row.get("fuel", "diesel")
    year = int(row.get("valid_year", 2024))
    geography = str(row.get("geography", "GLOBAL"))
    wtw = float(row["wtw_gco2e_per_tkm"]) / 1000.0      # g → kg CO2e / t·km

    factor_id = f"EF:{geography}:freight:{lane_id}:{year}"

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=fuel,
        unit="t_km",                                    # canonical
        geography=geography,
        geography_level=GeographyLevel.COUNTRY if len(geography) == 2 else GeographyLevel.GLOBAL,
        vectors=GHGVectors(CO2=wtw, CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.WTW,
        provenance=SourceProvenance(
            source_org="Smart Freight Centre (GLEC v3)",
            source_publication="GLEC Framework v3 / ISO 14083:2023 lanes",
            source_year=year,
            methodology=Methodology.HYBRID,
        ),
        valid_from=date(year, 1, 1),
        valid_to=date(year, 12, 31),
        uncertainty_95ci=0.20,
        dqs=DataQualityScore(
            temporal=4, geographical=4, technological=4,
            representativeness=4, methodological=5,
        ),
        license_info=LicenseInfo(
            license="GLEC framework + customer-lane overlay",
            redistribution_allowed=False,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="freight_lanes",
        source_release=f"GLEC-v3-{year}",
        source_record_id=lane_id,
        release_version=f"freight-{year}",
        license_class="commercial_connector",
        compliance_frameworks=["ISO_14083", "GLEC", "GHG_Protocol_Scope3"],
        activity_tags=["freight", mode, fuel],
        sector_tags=["transport"],
        tags=["freight", "glec", mode],
        factor_family=FactorFamily.TRANSPORT_LANE.value,
        factor_name=f"Freight {mode} {lane_id}",
        method_profile=MethodProfile.FREIGHT_ISO_14083.value,
        factor_version="1.0.0",
        formula_type=FormulaType.TRANSPORT_CHAIN.value,
        jurisdiction=Jurisdiction(country=geography if len(geography) == 2 else None),
        activity_schema=ActivitySchema(
            category="transport_freight",
            sub_category=mode,
            classification_codes=[],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope3"],
        ),
        verification=Verification(status=VerificationStatus.INTERNAL_REVIEW),
        explainability=Explainability(
            assumptions=[
                f"Payload utilization: {row.get('payload_utilization', 'unknown')}",
                f"Empty-running factor: {row.get('empty_running_factor', 'unknown')}",
                "WTW basis per GLEC Framework v3.",
            ],
            fallback_rank=5,
            rationale=f"Freight lane {lane_id} ({mode}, {fuel}, {year}).",
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        redistribution_class=RedistributionClass.LICENSED.value,
    )


__all__ = ["parse_freight_lane_rows"]
