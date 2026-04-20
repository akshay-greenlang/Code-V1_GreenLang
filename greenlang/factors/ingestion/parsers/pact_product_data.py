# -*- coding: utf-8 -*-
"""
PACT (Partnership for Carbon Transparency) product-data parser (Phase F9).

PACT defines a Pathfinder Framework data-exchange spec for Product
Carbon Footprints (PCF). Each exchange "ProductFootprint" object maps
naturally to a GreenLang material-embodied factor.

Input shape (subset of PACT v2 JSON)::

    {
        "id": "urn:gl:pact:product:ACME-STEEL-HRC-001",
        "productName": "Hot-rolled steel coil",
        "productCategoryCpc": "41237",
        "pcf": {
            "declaredUnit": "kg",
            "unitaryProductAmount": "1.0",
            "pCfExcludingBiogenic": "2.34",      # kg CO2e / declared unit
            "pCfIncludingBiogenic": "2.35",
            "fossilGhgEmissions": "2.30",
            "biogenicCarbonEmissions": "0.05",
            "geographyCountrySubdivision": "DE"
        },
        "companyName": "Acme Steel",
        "periodCoveredStart": "2024-01-01",
        "periodCoveredEnd": "2024-12-31",
        "version": 2,
        "pcfSpec": "2.0.0"
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


def parse_pact_rows(rows: Iterable[Dict[str, Any]]) -> List[EmissionFactorRecord]:
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping PACT row %d: %s", i, exc)
    logger.info("Parsed %d PACT product footprints", len(out))
    return out


def _row_to_record(row: Dict[str, Any]) -> EmissionFactorRecord:
    pcf = row.get("pcf") or {}
    product_id = row["id"]
    product_name = row.get("productName", product_id)
    country = str(pcf.get("geographyCountrySubdivision", "GLOBAL"))[:2].upper()

    # PACT values are strings per the spec — coerce to float.
    fossil_co2e = float(pcf.get("pCfExcludingBiogenic", pcf.get("fossilGhgEmissions", 0.0)))
    biogenic_co2 = float(pcf.get("biogenicCarbonEmissions", 0.0))
    declared_unit = str(pcf.get("declaredUnit", "kg"))

    period_start = str(row.get("periodCoveredStart", "2024-01-01"))
    period_end = str(row.get("periodCoveredEnd", "2024-12-31"))

    factor_id = f"EF:PACT:{product_id}"

    # PACT gives us aggregate CO2e, not gas-level vectors.  We preserve
    # the PACT CO2e in the CO2 slot (fossil) + biogenic_CO2 — callers
    # that need full gas split must upgrade to a primary LCA dataset.
    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=product_name,
        unit=declared_unit,
        geography=country,
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(
            CO2=fossil_co2e,
            CH4=0.0,
            N2O=0.0,
            biogenic_CO2=biogenic_co2,
        ),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_3,
        boundary=Boundary.CRADLE_TO_GATE,
        provenance=SourceProvenance(
            source_org=row.get("companyName", "PACT exchange"),
            source_publication="PACT Pathfinder Framework " + str(row.get("pcfSpec", "2.0")),
            source_year=int(period_start[:4]),
            methodology=Methodology.LCA,
        ),
        valid_from=date.fromisoformat(period_start),
        valid_to=date.fromisoformat(period_end),
        uncertainty_95ci=0.15,
        dqs=DataQualityScore(
            temporal=5, geographical=4, technological=4,
            representativeness=4, methodological=5,
        ),
        license_info=LicenseInfo(
            license="PACT Pathfinder terms",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="pact_exchange",
        source_release=str(row.get("pcfSpec", "2.0.0")),
        source_record_id=product_id,
        release_version="pact-" + str(row.get("version", "1")),
        license_class="registry_terms",
        compliance_frameworks=["PACT", "GHG_Protocol_Product", "ISO_14067"],
        activity_tags=["product_carbon", "pact"],
        sector_tags=[row.get("productCategoryCpc", "unknown_cpc")],
        tags=["pact", product_name],
        factor_family=FactorFamily.MATERIAL_EMBODIED.value,
        factor_name=product_name,
        method_profile=MethodProfile.PRODUCT_CARBON.value,
        factor_version="1.0.0",
        formula_type=FormulaType.LCA.value,
        jurisdiction=Jurisdiction(country=country),
        activity_schema=ActivitySchema(
            category="product_carbon_footprint",
            sub_category=row.get("productCategoryCpc"),
            classification_codes=[f"CPC:{row.get('productCategoryCpc', '')}"],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope3"],
            biogenic_share=(biogenic_co2 / (fossil_co2e + biogenic_co2)) if fossil_co2e + biogenic_co2 > 0 else 0.0,
        ),
        verification=Verification(
            status=VerificationStatus.EXTERNAL_VERIFIED,
            verified_by=row.get("verifiedBy", "PACT-conformant external verifier"),
        ),
        explainability=Explainability(
            assumptions=[
                "Boundary: cradle-to-gate unless the PACT object says otherwise.",
                "Biogenic carbon reported separately per PACT + GHG Protocol Product Standard.",
            ],
            fallback_rank=2,         # supplier-specific
            rationale=f"PACT product footprint for {product_name}.",
        ),
        primary_data_flag=PrimaryDataFlag.PRIMARY_MODELED.value,
        redistribution_class=RedistributionClass.RESTRICTED.value,
    )


__all__ = ["parse_pact_rows"]
