# -*- coding: utf-8 -*-
"""
India Carbon Credit Trading Scheme (CCTS) baseline-benchmark parser.

Source: Ministry of Environment, Forest and Climate Change (MoEFCC)
notification G.S.R. 443(E) dated 28 June 2023 establishing the Carbon
Credit Trading Scheme (CCTS) + Bureau of Energy Efficiency (BEE)
notifications specifying obligated sectors + emission-intensity
baselines. The first compliance cycle covers ~490 obligated entities
across eight energy-intensive sectors:

* cement
* iron & steel
* aluminum
* pulp & paper
* petrochemicals
* fertilizer
* chlor-alkali
* textiles (wet processing / man-made fibres)

The Bureau of Energy Efficiency under the Energy Conservation (Amendment)
Act, 2022 publishes per-sector emission-intensity benchmarks (tCO2 /
tonne of output) against which obligated entities generate or surrender
Carbon Credit Certificates (CCCs).

Publication cadence: sector-wise BEE gazette notifications, triennial
target revision; baselines refreshed annually.

License class: ``public_in_government`` — Indian government
notifications are public documents with attribution required.

Input row shape::

    {
        "sector": "cement",
        "sub_sector": "OPC_grade_53",
        "baseline_intensity_tco2_per_tonne_output": 0.82,
        "unit_of_output": "tonne_cement",
        "target_year": "2025-26",
        "compliance_cycle": "CCTS-1",
        "obligated_entity_scope": "installed_capacity_above_0.5Mtpa",
        "notification_reference": "BEE/CCTS/cement/2024",
    }
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

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


# Sector -> human-readable descriptor + NIC (National Industrial
# Classification) code for activity_schema.classification_codes.
_SECTORS: Dict[str, Dict[str, str]] = {
    "cement": {
        "display": "Cement manufacturing",
        "nic": "NIC:2394",
        "isic": "ISIC:C2394",
        "output_unit": "tonne_cement",
    },
    "iron_and_steel": {
        "display": "Iron and steel manufacturing",
        "nic": "NIC:2410",
        "isic": "ISIC:C2410",
        "output_unit": "tonne_crude_steel",
    },
    "steel": {
        "display": "Iron and steel manufacturing",
        "nic": "NIC:2410",
        "isic": "ISIC:C2410",
        "output_unit": "tonne_crude_steel",
    },
    "aluminum": {
        "display": "Primary aluminum smelting",
        "nic": "NIC:2420",
        "isic": "ISIC:C2420",
        "output_unit": "tonne_aluminum",
    },
    "aluminium": {
        "display": "Primary aluminum smelting",
        "nic": "NIC:2420",
        "isic": "ISIC:C2420",
        "output_unit": "tonne_aluminum",
    },
    "paper": {
        "display": "Pulp and paper manufacturing",
        "nic": "NIC:1701",
        "isic": "ISIC:C1701",
        "output_unit": "tonne_paper",
    },
    "pulp_and_paper": {
        "display": "Pulp and paper manufacturing",
        "nic": "NIC:1701",
        "isic": "ISIC:C1701",
        "output_unit": "tonne_paper",
    },
    "petrochemicals": {
        "display": "Petrochemicals manufacturing",
        "nic": "NIC:2011",
        "isic": "ISIC:C2011",
        "output_unit": "tonne_product",
    },
    "fertilizer": {
        "display": "Fertilizer manufacturing",
        "nic": "NIC:2012",
        "isic": "ISIC:C2012",
        "output_unit": "tonne_nutrient",
    },
    "fertiliser": {
        "display": "Fertilizer manufacturing",
        "nic": "NIC:2012",
        "isic": "ISIC:C2012",
        "output_unit": "tonne_nutrient",
    },
    "chlor_alkali": {
        "display": "Chlor-alkali manufacturing",
        "nic": "NIC:2011",
        "isic": "ISIC:C2011",
        "output_unit": "tonne_caustic_soda",
    },
    "textile": {
        "display": "Textile wet processing / man-made fibres",
        "nic": "NIC:1313",
        "isic": "ISIC:C1313",
        "output_unit": "tonne_textile",
    },
    "textiles": {
        "display": "Textile wet processing / man-made fibres",
        "nic": "NIC:1313",
        "isic": "ISIC:C1313",
        "output_unit": "tonne_textile",
    },
}


def parse_india_ccts_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    default_source_year: int = 2024,
) -> List[EmissionFactorRecord]:
    """Parse pre-extracted India CCTS baseline rows into EmissionFactorRecords.

    The parser is intentionally tolerant of sector-name synonyms (e.g.,
    ``aluminum`` / ``aluminium``). Rows that fail validation are logged
    and skipped.
    """
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            out.append(_row_to_record(row, default_source_year=default_source_year))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping India CCTS row %d: %s", i, exc)
    logger.info("Parsed %d India CCTS baseline rows", len(out))
    return out


def _row_to_record(
    row: Dict[str, Any], *, default_source_year: int
) -> EmissionFactorRecord:
    sector_raw = str(row["sector"]).strip().lower().replace("-", "_").replace(" ", "_")
    if sector_raw not in _SECTORS:
        raise ValueError(f"Unknown India CCTS sector: {row['sector']!r}")
    meta = _SECTORS[sector_raw]

    sub_sector = row.get("sub_sector")
    target_year = str(row.get("target_year") or f"{default_source_year}-{(default_source_year + 1) % 100:02d}")
    compliance_cycle = str(row.get("compliance_cycle") or "CCTS-1")
    notification_ref = str(row.get("notification_reference") or "BEE/CCTS")
    output_unit = str(row.get("unit_of_output") or meta["output_unit"])

    baseline_intensity = Decimal(str(row["baseline_intensity_tco2_per_tonne_output"]))
    # Store as kg CO2 per unit of output for canonical-unit alignment.
    kg_co2_per_unit = baseline_intensity * Decimal("1000")

    fy_start = int(target_year.split("-")[0])
    valid_from = date(fy_start, 4, 1)          # India FY starts 1 April
    valid_to = date(fy_start + 1, 3, 31)

    factor_id = (
        f"EF:IN:ccts:{sector_raw}"
        + (f":{str(sub_sector).lower()}" if sub_sector else "")
        + f":{target_year}:{compliance_cycle}"
    )

    obligated_entity_scope = row.get("obligated_entity_scope")

    return EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type=f"{sector_raw}_output",
        unit=output_unit,
        geography="IN",
        geography_level=GeographyLevel.COUNTRY,
        vectors=GHGVectors(CO2=float(kg_co2_per_unit), CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_1,
        boundary=Boundary.CRADLE_TO_GATE,
        provenance=SourceProvenance(
            source_org="Bureau of Energy Efficiency (BEE), Government of India",
            source_publication=(
                "Carbon Credit Trading Scheme (CCTS) baseline emission "
                "intensities per MoEFCC notification G.S.R. 443(E) and "
                "BEE sector notifications"
            ),
            source_year=default_source_year,
            methodology=Methodology.HYBRID,
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.1,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5,
            technological=4,
            representativeness=4,
            methodological=4,
        ),
        license_info=LicenseInfo(
            license="Government of India — public notification (attribution required)",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        source_id="india_ccts_baselines",
        source_release=notification_ref,
        source_record_id=f"ccts:{sector_raw}:{target_year}",
        release_version=f"{compliance_cycle}-{target_year}",
        license_class="public_in_government",
        compliance_frameworks=[
            "India_CCTS",
            "India_BRSR",
            "Energy_Conservation_Amendment_Act_2022",
        ],
        activity_tags=["ccts_baseline", "india", sector_raw],
        sector_tags=[sector_raw, "obligated_entity"],
        tags=["ccts", "india", sector_raw, target_year, compliance_cycle],
        factor_family=FactorFamily.EMISSIONS.value,
        factor_name=(
            f"India CCTS baseline — {meta['display']}"
            + (f" ({sub_sector})" if sub_sector else "")
            + f", FY {target_year}"
        ),
        method_profile=MethodProfile.INDIA_CCTS.value,
        factor_version="1.0.0",
        formula_type=FormulaType.DIRECT_FACTOR.value,
        jurisdiction=Jurisdiction(country="IN"),
        activity_schema=ActivitySchema(
            category="ccts_baseline_intensity",
            sub_category=str(sub_sector) if sub_sector else sector_raw,
            classification_codes=[meta["nic"], meta["isic"]],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope1"],
            residual_mix_applicable=False,
            supplier_specific=False,
            transmission_loss_included=False,
            biogenic_share=0.0,
        ),
        verification=Verification(
            status=VerificationStatus.REGULATOR_APPROVED,
            verified_by="Bureau of Energy Efficiency (BEE), Government of India",
            verification_reference=notification_ref,
        ),
        explainability=Explainability(
            assumptions=[
                (
                    "India CCTS baseline intensity published by the Bureau of "
                    "Energy Efficiency per the Energy Conservation (Amendment) "
                    "Act, 2022 and MoEFCC Carbon Credit Trading Scheme "
                    "notification G.S.R. 443(E)."
                ),
                (
                    "Applies to obligated entities in the {sector} sector; "
                    "non-obligated entities may use the baseline as a "
                    "sectoral reference.".format(sector=meta["display"].lower())
                ),
                (
                    f"Compliance cycle: {compliance_cycle}. "
                    f"Target year: {target_year}."
                ),
                *(
                    [f"Obligated-entity scope: {obligated_entity_scope}."]
                    if obligated_entity_scope
                    else []
                ),
                "Baseline values are subject to triennial BEE revision.",
            ],
            fallback_rank=5,
            rationale=(
                f"India CCTS {compliance_cycle} baseline for "
                f"{meta['display']}, FY {target_year}. Notification "
                f"reference: {notification_ref}."
            ),
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.OPEN.value,
    )


__all__ = [
    "parse_india_ccts_rows",
]
