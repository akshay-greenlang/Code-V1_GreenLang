# -*- coding: utf-8 -*-
"""
India CEA CO2 Baseline Database parser (Phase F2.2).

Source: Central Electricity Authority (CEA), Ministry of Power, Government
of India. The CO2 Baseline Database for the Indian Power Sector is
published annually (latest user manual: v20.0, Dec 2024 reporting 2023-24
data). It publishes grid-average CO2 intensities for regional grids
(NEWNE, S, NER) and for all-India composite.

Publication URL: https://cea.nic.in/cdm-co2-baseline-database/

License class: ``public_in_government`` (no formal licence; attribution
required). Safe to redistribute as a factor with citation.

This parser is deliberately tolerant: the CEA publishes as PDF + Excel.
Callers pass pre-extracted rows (list of dicts) and this module shapes
them into :class:`~greenlang.data.emission_factor_record.EmissionFactorRecord`
objects tagged with the ``ElectricityLocationPack`` method profile.

Shape of each input row::

    {
        "grid": "All India" | "NEWNE" | "S" | "NER",
        "financial_year": "2023-24",
        "co2_intensity_t_per_mwh": 0.727,          # tCO2/MWh
        "publication_version": "v20.0",
        "transmission_loss_included": False,
    }
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
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


#: Map CEA regional grid codes to more verbose descriptions for audit text.
_GRID_REGIONS = {
    "All India": "All-India composite grid",
    "NEWNE": "Northern + Eastern + Western + North-Eastern regional grids",
    "S": "Southern regional grid",
    "NER": "North-Eastern regional grid (standalone)",
    "N": "Northern regional grid",
    "E": "Eastern regional grid",
    "W": "Western regional grid",
}


def parse_india_cea_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    default_source_year: int = 2024,
) -> List[EmissionFactorRecord]:
    """Parse pre-extracted CEA rows into EmissionFactorRecord instances."""
    out: List[EmissionFactorRecord] = []
    for i, row in enumerate(rows):
        try:
            record = _row_to_record(row, default_source_year=default_source_year)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping CEA row %d: %s", i, exc)
            continue
        out.append(record)
    logger.info("Parsed %d CEA grid factors", len(out))
    return out


def _row_to_record(
    row: Dict[str, Any], *, default_source_year: int
) -> EmissionFactorRecord:
    grid = str(row["grid"])
    fy = str(row.get("financial_year") or f"{default_source_year-1}-{default_source_year % 100}")
    intensity_t_mwh = Decimal(str(row["co2_intensity_t_per_mwh"]))
    # Convert tCO2/MWh -> kgCO2/kWh for canonical unit alignment.
    kg_co2_per_kwh = intensity_t_mwh
    pub_version = str(row.get("publication_version") or "v20.0")
    transmission_loss_included = bool(row.get("transmission_loss_included", False))

    start_year = int(fy.split("-")[0])
    valid_from = date(start_year, 4, 1)          # India FY starts 1 April
    valid_to = date(start_year + 1, 3, 31)

    grid_region = grid if grid != "All India" else None

    factor_id = f"EF:IN:{grid.lower().replace(' ', '_')}:{fy}:cea-{pub_version}"

    record = EmissionFactorRecord(
        factor_id=factor_id,
        fuel_type="electricity",
        unit="kWh",
        geography="IN",
        geography_level=GeographyLevel.COUNTRY if grid == "All India" else GeographyLevel.GRID_ZONE,
        vectors=GHGVectors(CO2=float(kg_co2_per_kwh), CH4=0.0, N2O=0.0),
        gwp_100yr=GWPValues(gwp_set=GWPSet.IPCC_AR6_100, CH4_gwp=28, N2O_gwp=273),
        scope=Scope.SCOPE_2,
        boundary=Boundary.COMBUSTION,
        provenance=SourceProvenance(
            source_org="CEA (Central Electricity Authority, India)",
            source_publication="CO2 Baseline Database for the Indian Power Sector",
            source_year=default_source_year,
            methodology=Methodology.DIRECT_MEASUREMENT,
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        uncertainty_95ci=0.05,
        dqs=DataQualityScore(
            temporal=5,
            geographical=5 if grid_region else 4,
            technological=4,
            representativeness=5,
            methodological=5,
        ),
        license_info=LicenseInfo(
            license="Government of India — public use with attribution",
            redistribution_allowed=True,
            commercial_use_allowed=True,
            attribution_required=True,
        ),
        region_hint=grid_region,
        source_id="india_cea_co2_baseline",
        source_release=pub_version,
        source_record_id=f"cea:{grid}:{fy}",
        release_version=f"cea-{pub_version}",
        license_class="public_in_government",
        compliance_frameworks=["GHG_Protocol_Scope2_LocationBased", "IFRS_S2"],
        activity_tags=["purchased_electricity", "grid_average", "india"],
        sector_tags=["power_sector", "utility"],
        tags=["cea", "india", "grid", fy],
        # -------- Canonical v2 fields (Phase F1) --------
        factor_family=FactorFamily.GRID_INTENSITY.value,
        factor_name=f"India grid electricity — {_GRID_REGIONS.get(grid, grid)}, {fy}",
        method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION.value,
        factor_version="1.0.0",
        formula_type=FormulaType.DIRECT_FACTOR.value,
        jurisdiction=Jurisdiction(country="IN", region=None, grid_region=grid_region),
        activity_schema=ActivitySchema(
            category="purchased_electricity",
            sub_category="grid_average",
            classification_codes=["NAICS:221112", "ISIC:D351"],
        ),
        parameters=FactorParameters(
            scope_applicability=["scope2"],
            electricity_basis=ElectricityBasis.LOCATION_BASED,
            residual_mix_applicable=False,
            supplier_specific=False,
            transmission_loss_included=transmission_loss_included,
            biogenic_share=0.0,
        ),
        verification=Verification(
            status=VerificationStatus.REGULATOR_APPROVED,
            verified_by="Government of India — CEA",
        ),
        explainability=Explainability(
            assumptions=[
                "Use for purchased grid electricity in India when supplier-specific factor is unavailable.",
                "CEA methodology excludes captive-generation and auxiliary consumption.",
                "Based on generation-weighted grid mix for the given financial year.",
            ],
            fallback_rank=5,
            rationale=f"India CEA CO2 baseline {pub_version} for {_GRID_REGIONS.get(grid, grid)}, FY {fy}.",
        ),
        primary_data_flag=PrimaryDataFlag.SECONDARY.value,
        uncertainty_distribution=UncertaintyDistribution.NORMAL.value,
        redistribution_class=RedistributionClass.OPEN.value,
    )
    return record


__all__ = ["parse_india_cea_rows"]
