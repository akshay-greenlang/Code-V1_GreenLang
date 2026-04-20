# -*- coding: utf-8 -*-
"""
EC3 / EPD International parser (Phase F9).

Building Transparency's Embodied Carbon in Construction Calculator
(EC3) aggregates verified Environmental Product Declarations (EPDs)
for construction materials.  Each EPD is an ISO 14025 / EN 15804-
compliant PCF with modular impact (A1-A3 cradle-to-gate, optional
A4-C4) used by embodied-carbon tools.

Input shape::

    {
        "epd_id": "EPD-REG-20240001-EN",
        "product_name": "Ready-mix concrete 30 MPa",
        "category": "concrete",
        "functional_unit": "m3",
        "declared_unit_co2e_kg": 278.0,
        "modules_reported": ["A1", "A2", "A3"],
        "program_operator": "EPD International",
        "verification_date": "2024-03-15",
        "valid_until": "2029-03-15",
        "country": "SE",
        "manufacturer": "NCC AB"
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


def parse_ec3_epd_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping EC3/EPD row %d: %s", i, exc)
    logger.info("Parsed %d EC3/EPD records", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    epd_id = row["epd_id"]
    product_name = row.get("product_name", epd_id)
    country = str(row.get("country", "GLOBAL"))[:2].upper()
    declared_unit = str(row.get("functional_unit", "kg"))
    co2e_kg = float(row["declared_unit_co2e_kg"])

    valid_from = date.fromisoformat(str(row.get("verification_date", "2024-01-01")))
    valid_to = date.fromisoformat(str(row.get("valid_until", "2029-01-01")))
    modules = row.get("modules_reported", ["A1", "A2", "A3"])

    # EPD module A1-A3 = cradle-to-gate; A4-A5 adds transport to site;
    # A-C = full lifecycle.  Pick the right boundary label.
    if any(m in modules for m in ("B1", "B2", "C1", "C2", "C3", "C4")):
        boundary = Boundary.CRADLE_TO_GRAVE
    else:
        boundary = Boundary.CRADLE_TO_GATE

    factor_id = f"EF:EC3:{epd_id}"

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=product_name,
        unit=declared_unit,
        geography=country,
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(CO2=co2e_kg, CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=boundary,
        provenance=SourceProvenance(
            source_org=row.get("program_operator", "EPD International"),
            source_publication=f"EPD {epd_id} ({row.get('manufacturer', 'unknown')})",
            source_year=valid_from.year,
            methodology=Methodology.LCA,
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.20,
        dqs=DataQualityScore(
            temporal=4, geographical=4, technological=5,
            representativeness=5, methodological=5,
        ),
        license_info=LicenseInfo(
            license="EC3 / EPD program terms",
            redistribution_allowed=False,       # EPD API is permissioned
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="ec3_epd",
        source_release=row.get("program_operator", "EPD_INT"),
        source_record_id=epd_id,
        release_version=f"epd-{valid_from.isoformat()}",
        license_class="commercial_connector",
        compliance_frameworks=["ISO_14025", "EN_15804", "GHG_Protocol_Product"],
        activity_tags=["embodied_carbon", "construction", row.get("category", "unknown")],
        sector_tags=[row.get("category", "unknown")],
        tags=["ec3", "epd", product_name, country],
        factor_family=FactorFamily.MATERIAL_EMBODIED.value,
        factor_name=product_name,
        method_profile=MethodProfile.PRODUCT_CARBON.value,
        factor_version="1.0.0",
        formula_type=FormulaType.LCA.value,
        jurisdiction=Jurisdiction(country=country),
        activity_schema=ActivitySchema(
            category="construction_material",
            sub_category=row.get("category"),
            classification_codes=[],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope3"],
        ),
        verification=Verification(
            status=VerificationStatus.EXTERNAL_VERIFIED,
            verified_by=row.get("program_operator", "EPD program operator"),
            verified_at=None,
        ),
        explainability=Explainability(
            assumptions=[
                "Boundary derived from EPD modules reported.",
                "EPD expiry treated as hard validity end.",
                f"Functional unit: {declared_unit}.",
            ],
            fallback_rank=2,
            rationale=f"EC3/EPD record for {product_name} ({country}).",
        ),
        primary_data_flag=PrimaryDataFlag.PRIMARY.value,
        redistribution_class=RedistributionClass.LICENSED.value,
    )


__all__ = ["parse_ec3_epd_rows"]
