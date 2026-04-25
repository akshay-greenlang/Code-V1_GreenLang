# -*- coding: utf-8 -*-
"""
Normalize upstream JSON into EmissionFactorRecord-compatible dicts.

CBAM and DEFRA Agent-Factory shapes are supported as row expansion (not full LCA).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

from greenlang.data.emission_factor_record import (
    Boundary,
    DataQualityScore,
    GeographyLevel,
    GWPSet,
    GWPValues,
    GHGVectors,
    LicenseInfo,
    Methodology,
    Scope,
    SourceProvenance,
)


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return x or "unknown"


def _stamp_record_dict(rec: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rec = dict(rec)
    rec.setdefault("created_at", now)
    rec.setdefault("updated_at", now)
    rec.setdefault("created_by", "greenlang_factors_etl")
    rec.setdefault("compliance_frameworks", ["GHG_Protocol", "IPCC_2006"])
    gwp = dict(rec.get("gwp_100yr") or {})
    gwp.pop("co2e_total", None)
    rec["gwp_100yr"] = gwp
    return rec


def iter_cbam_factor_dicts(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Expand cbam_defaults_2024.json style structure."""
    meta = data.get("metadata") or {}
    year = 2024
    try:
        ver = str(meta.get("version", "2024"))
        year = int(ver.split(".")[0])
    except (TypeError, ValueError):
        year = 2024
    factors = data.get("factors") or {}
    if not isinstance(factors, dict):
        return
    for product_key, pdata in factors.items():
        if not isinstance(pdata, dict):
            continue
        by_country = pdata.get("by_country") or {}
        if not isinstance(by_country, dict):
            continue
        for country, row in by_country.items():
            if not isinstance(row, dict):
                continue
            direct = float(row.get("direct_emissions_factor", 0) or 0)
            indirect = float(row.get("indirect_emissions_factor", 0) or 0)
            # tCO2e/tonne product ~= kg CO2e per kg product; store as CO2 only proxy
            co2_kg_per_kg = (direct + indirect) * 1000.0 / 1000.0
            fid = f"EF:CBAM:{_slug(product_key)}:{country}:{year}:v1"
            prov = SourceProvenance(
                source_org="EU Commission",
                source_publication=str(
                    meta.get("description") or "CBAM default emission factors"
                ),
                source_year=year,
                methodology=Methodology.IPCC_TIER_1,
                source_url="https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en",
                version=str(meta.get("version", "v1")),
            )
            dqs = DataQualityScore(
                temporal=4, geographical=4, technological=3, representativeness=3, methodological=4
            )
            lic = LicenseInfo(
                license="EU-legal-text",
                redistribution_allowed=False,
                commercial_use_allowed=True,
                attribution_required=True,
            )
            record = {
                "factor_id": fid,
                "fuel_type": f"cbam_{_slug(product_key)}",
                "unit": "kg_product",
                "geography": str(country).upper(),
                "geography_level": GeographyLevel.COUNTRY.value,
                "vectors": {"CO2": co2_kg_per_kg, "CH4": 0.0, "N2O": 0.0},
                "gwp_100yr": {
                    "gwp_set": GWPSet.IPCC_AR6_100.value,
                    "CH4_gwp": 28.0,
                    "N2O_gwp": 273.0,
                },
                "scope": Scope.SCOPE_3.value,
                "boundary": Boundary.CRADLE_TO_GATE.value,
                "provenance": {
                    "source_org": prov.source_org,
                    "source_publication": prov.source_publication,
                    "source_year": prov.source_year,
                    "methodology": prov.methodology.value,
                    "source_url": prov.source_url,
                    "version": prov.version,
                },
                "valid_from": date(year, 1, 1).isoformat(),
                "uncertainty_95ci": 0.15,
                "dqs": {
                    "temporal": dqs.temporal,
                    "geographical": dqs.geographical,
                    "technological": dqs.technological,
                    "representativeness": dqs.representativeness,
                    "methodological": dqs.methodological,
                },
                "license_info": {
                    "license": lic.license,
                    "redistribution_allowed": lic.redistribution_allowed,
                    "commercial_use_allowed": lic.commercial_use_allowed,
                    "attribution_required": lic.attribution_required,
                },
                "tags": ["cbam", product_key],
                "notes": "CBAM default combined direct+indirect intensity as CO2 proxy vector.",
            }
            yield _stamp_record_dict(record)


def iter_defra_scope1_dicts(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Expand DEFRA scope1_fuels.json style nested lists."""
    meta = data.get("metadata") or {}
    region = str(meta.get("region", "UK")).upper()
    year = 2024
    try:
        year = int(str(meta.get("version", "2024")).split(".")[0])
    except (TypeError, ValueError):
        year = 2024
    gwp_set = GWPSet.IPCC_AR5_100.value
    for key, val in data.items():
        if key == "metadata" or not isinstance(val, list):
            continue
        for fuel in val:
            if not isinstance(fuel, dict):
                continue
            fuel_type = _slug(str(fuel.get("fuel_type") or fuel.get("fuel_name") or "fuel"))
            native_id = str(fuel.get("id") or "")
            units = fuel.get("units") or []
            if not isinstance(units, list):
                continue
            for urow in units:
                if not isinstance(urow, dict):
                    continue
                unit_key = _slug(str(urow.get("unit")))
                co2 = float(urow.get("co2_factor", 0) or 0)
                ch4 = float(urow.get("ch4_factor", 0) or 0)
                n2o = float(urow.get("n2o_factor", 0) or 0)
                fid = f"EF:{region}:{fuel_type}_{unit_key}:{year}:v1"
                prov = SourceProvenance(
                    source_org="DEFRA",
                    source_publication=str(meta.get("source_document") or "DEFRA GHG factors"),
                    source_year=year,
                    methodology=Methodology.IPCC_TIER_1,
                    source_url=str(meta.get("source_url") or ""),
                    version=str(meta.get("version", "v1")),
                )
                dqs = DataQualityScore(
                    temporal=5, geographical=5, technological=4, representativeness=4, methodological=5
                )
                lic = LicenseInfo(
                    license="OGL-UK",
                    redistribution_allowed=True,
                    commercial_use_allowed=True,
                    attribution_required=True,
                )
                yield _stamp_record_dict(
                    {
                        "factor_id": fid,
                        "fuel_type": fuel_type,
                        "unit": unit_key.replace("_", " "),
                        "geography": region,
                        "geography_level": GeographyLevel.COUNTRY.value,
                        "vectors": {"CO2": co2, "CH4": ch4, "N2O": n2o},
                        "gwp_100yr": {
                            "gwp_set": gwp_set,
                            "CH4_gwp": 28.0,
                            "N2O_gwp": 265.0,
                        },
                        "scope": Scope.SCOPE_1.value,
                        "boundary": Boundary.COMBUSTION.value,
                        "provenance": {
                            "source_org": prov.source_org,
                            "source_publication": prov.source_publication,
                            "source_year": prov.source_year,
                            "methodology": prov.methodology.value,
                            "source_url": prov.source_url,
                            "version": prov.version,
                        },
                        "valid_from": date(year, 1, 1).isoformat(),
                        "uncertainty_95ci": 0.05,
                        "dqs": {
                            "temporal": dqs.temporal,
                            "geographical": dqs.geographical,
                            "technological": dqs.technological,
                            "representativeness": dqs.representativeness,
                            "methodological": dqs.methodological,
                        },
                        "license_info": {
                            "license": lic.license,
                            "redistribution_allowed": lic.redistribution_allowed,
                            "commercial_use_allowed": lic.commercial_use_allowed,
                            "attribution_required": lic.attribution_required,
                        },
                        "tags": ["defra", "scope1", native_id],
                        "notes": None,
                    }
                )


def dict_to_emission_factor_record(data: Dict[str, Any]) -> Any:
    """Convert normalized dict to EmissionFactorRecord (recomputes content_hash)."""
    from greenlang.data.emission_factor_record import EmissionFactorRecord

    return EmissionFactorRecord.from_dict(dict(data))


# ---------------------------------------------------------------------------
# Alpha Provenance Gate hook (Wave B / TaskCreate #5 / WS2-T1)
# ---------------------------------------------------------------------------


def run_alpha_provenance_gate(
    records: Iterable[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    *,
    default_on: Optional[bool] = None,
    report_filename: str = "validation_report.json",
) -> Dict[str, Any]:
    """Run the v0.1 Alpha Provenance Gate over a batch of normalised records.

    Collects every failure into ``validation_report.json`` written to
    ``output_dir`` (when supplied) and returns the report as a dict.

    The gate is opt-in/opt-out via ``GL_FACTORS_ALPHA_PROVENANCE_GATE``; if
    that env var is unset the gate runs iff the active release profile is
    ``alpha-v0.1`` (override via ``default_on=True/False``).
    """
    from greenlang.factors.quality.alpha_provenance_gate import (
        AlphaProvenanceGate,
        alpha_gate_enabled,
    )

    if default_on is None:
        try:
            from greenlang.factors.release_profile import (
                ReleaseProfile,
                current_profile,
            )

            default_on = current_profile() == ReleaseProfile.ALPHA_V0_1
        except Exception:  # noqa: BLE001
            default_on = False

    rec_list = list(records)

    if not alpha_gate_enabled(default_on=default_on):
        report = {
            "gate": "alpha_provenance_gate",
            "enabled": False,
            "total_records": len(rec_list),
            "passed": len(rec_list),
            "failed": 0,
            "failures": [],
        }
    else:
        gate = AlphaProvenanceGate()
        failures: List[Dict[str, Any]] = []
        for idx, rec in enumerate(rec_list):
            errs = gate.validate(rec)
            if errs:
                failures.append(
                    {
                        "record_index": idx,
                        "factor_urn": rec.get("urn") if isinstance(rec, dict) else None,
                        "failures": errs,
                    }
                )
        report = {
            "gate": "alpha_provenance_gate",
            "enabled": True,
            "total_records": len(rec_list),
            "passed": len(rec_list) - len(failures),
            "failed": len(failures),
            "failures": failures,
        }

    if output_dir is not None:
        out = Path(output_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / report_filename).write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "alpha_provenance_gate: could not write %s/%s: %s",
                out,
                report_filename,
                exc,
            )

    return report
