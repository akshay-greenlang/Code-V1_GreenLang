# -*- coding: utf-8 -*-
"""
Generate the GreenLang Factors v1 public gold-label evaluation set.

This is a one-off generator script, run by hand to produce the JSON files
under ``greenlang/factors/data/gold_set/v1/``.  The generated cases are
then used by ``tests/factors/test_gold_set_eval.py`` (and by the
``factors-gold-eval`` GitHub Actions job) to compute precision@1 against
``greenlang.factors.matching.pipeline.run_match`` and
``greenlang.factors.resolution.engine.ResolutionEngine.resolve``.

Run it with::

    python scripts/generate_gold_set_v1.py

It writes 7 JSON files + index.json + README.md (README is left intact
if it already exists).  Re-running is safe — the generator is
deterministic.

Factor-id convention used (matches the on-disk parsers under
``greenlang/factors/ingestion/parsers/``):

    EF:<authority_or_jurisdiction>:<fuel_or_product>:<geo>:<year>:v<n>

Examples emitted by the parsers in this repo::

    EF:IN:northern_grid:2026-27:cea-v20.0      india_cea.py
    EF:DESNZ:s1_bio_natural_gas_kwh:UK:2026:v1 desnz_uk.py
    EF:IEA:diesel:US:2026:v1                   iea.py
    EF:CBAM:steel:CN:2024:v1                   policy_factor_map.example.yaml
    EF:<country>:residual_mix:<year>:v1.0      aib_residual_mix.py

Where the catalog seed does not yet hold a real id we set
``expected.factor_id`` to ``null`` and assert on family + a published
range — see the README under ``greenlang/factors/data/gold_set/`` for
the precision@1 bar interpretation.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "greenlang" / "factors" / "data" / "gold_set" / "v1"
TODAY = "2026-04-23"
DEFAULT_FY = 2027


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def case(
    *,
    case_id: str,
    description: str,
    quantity: float,
    unit: str,
    metadata: Dict[str, Any],
    method_profile: str,
    factor_id: str | None,
    factor_family: str,
    source_authority: str,
    fallback_rank: int,
    co2e_min: float,
    co2e_max: float,
    co2e_unit: str,
    must_include_assumptions: List[str],
    tags: List[str],
) -> Dict[str, Any]:
    """Build one gold-set case.  All keyword-only to avoid argument drift."""
    return {
        "case_id": case_id,
        "activity": {
            "description": description,
            "quantity": quantity,
            "unit": unit,
            "metadata": metadata,
        },
        "method_profile": method_profile,
        "expected": {
            "factor_id": factor_id,
            "factor_family": factor_family,
            "source_authority": source_authority,
            "fallback_rank": fallback_rank,
            "co2e_per_unit_min": co2e_min,
            "co2e_per_unit_max": co2e_max,
            "co2e_unit": co2e_unit,
            "must_include_assumptions": must_include_assumptions,
        },
        "tags": tags + ["fy27_launch"],
    }


# ---------------------------------------------------------------------------
# 1. ELECTRICITY (60 cases)
# ---------------------------------------------------------------------------


# Real CEA grid factors for India (CO2 Baseline Database v20.0, FY2024-25
# values; FY2026-27 used as the activity year for the launch profile).
CEA_GRID_FACTORS = {
    # name : (kg CO2/kWh min, max, region_slug)
    "all_india":         (0.715, 0.745, "all_india"),
    "northern_grid":     (0.500, 0.560, "northern_grid"),
    "western_grid":      (0.860, 0.910, "western_grid"),
    "southern_grid":     (0.660, 0.720, "southern_grid"),
    "eastern_grid":      (0.940, 0.990, "eastern_grid"),
    "north_eastern_grid":(0.220, 0.280, "north_eastern_grid"),
}


def build_electricity_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # ---- INDIA — 20 cases ----
    # 12 location-based across grids/years
    india_grid_cycle = [
        ("all_india", 2027, "FY2026-27", 12500),
        ("northern_grid", 2027, "FY2026-27", 8200),
        ("western_grid", 2027, "FY2026-27", 14300),
        ("southern_grid", 2027, "FY2026-27", 9800),
        ("eastern_grid", 2027, "FY2026-27", 6400),
        ("north_eastern_grid", 2027, "FY2026-27", 1100),
        ("all_india", 2026, "FY2025-26", 5600),
        ("western_grid", 2026, "FY2025-26", 22000),
        ("southern_grid", 2026, "FY2025-26", 17400),
        ("northern_grid", 2026, "FY2025-26", 33000),
        ("all_india", 2025, "FY2024-25", 9200),
        ("eastern_grid", 2025, "FY2024-25", 4100),
    ]
    for i, (grid, yr, fy_label, qty) in enumerate(india_grid_cycle, start=1):
        lo, hi, slug = CEA_GRID_FACTORS[grid]
        out.append(case(
            case_id=f"elec_in_{slug}_{yr}_lb_{i:03d}",
            description=(
                f"Purchased grid electricity, India {grid.replace('_', ' ').title()} "
                f"({fy_label}), location-based"
            ),
            quantity=qty,
            unit="kWh",
            metadata={"country": "IN", "grid_region": grid, "year": yr,
                      "fy_label": fy_label, "scope": "scope2"},
            method_profile="corporate_scope2_location_based",
            factor_id=f"EF:IN:{slug}:{fy_label.replace('FY','')}:cea-v20.0",
            factor_family="grid_intensity",
            source_authority="CEA",
            fallback_rank=4,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["grid average", "location-based", "CEA"],
            tags=["electricity", "india", "location_based", grid],
        ))

    # 4 market-based supplier-specific PPAs
    india_market = [
        ("Tata Power solar PPA, Karnataka", 5400, 0.04, 0.06,
         "supplier_specific", 2),
        ("ReNew Power wind PPA, Tamil Nadu", 7200, 0.01, 0.03,
         "supplier_specific", 2),
        ("Adani Green hybrid PPA, Rajasthan", 12500, 0.03, 0.05,
         "supplier_specific", 2),
        ("ACME Solar rooftop PPA, Maharashtra", 2400, 0.04, 0.07,
         "supplier_specific", 2),
    ]
    for i, (desc, qty, lo, hi, profile_label, rank) in enumerate(india_market, start=1):
        out.append(case(
            case_id=f"elec_in_market_supplier_{i:03d}",
            description=f"Market-based purchased electricity, {desc}",
            quantity=qty,
            unit="kWh",
            metadata={"country": "IN", "year": 2027, "scope": "scope2",
                      "instrument": "PPA"},
            method_profile="corporate_scope2_market_based",
            factor_id=None,  # supplier-specific records are tenant data
            factor_family="grid_intensity",
            source_authority="supplier",
            fallback_rank=rank,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["market-based", "supplier-specific"],
            tags=["electricity", "india", "market_based", "ppa"],
        ))

    # 4 market-based without instrument → falls back to country/grid average
    for i, (grid, qty) in enumerate(
        [("all_india", 11000), ("western_grid", 6800),
         ("southern_grid", 9400), ("northern_grid", 14200)], start=1):
        lo, hi, slug = CEA_GRID_FACTORS[grid]
        out.append(case(
            case_id=f"elec_in_market_residual_fallback_{i:03d}",
            description=(
                f"Market-based electricity with no contractual instrument, "
                f"India {grid.replace('_', ' ')} — defaults to CEA grid-average"
            ),
            quantity=qty,
            unit="kWh",
            metadata={"country": "IN", "grid_region": grid, "year": 2027,
                      "scope": "scope2", "instrument": None},
            method_profile="corporate_scope2_market_based",
            factor_id=f"EF:IN:{slug}:2026-27:cea-v20.0",
            factor_family="grid_intensity",
            source_authority="CEA",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=[
                "market-based", "no instrument applied", "grid-average fallback"],
            tags=["electricity", "india", "market_based", "fallback"],
        ))

    # ---- EU — 15 cases ----
    # 6 location-based national factors (AIB Production Mix proxy ranges)
    eu_country_lb = [
        ("FR", 0.035, 0.060, 18000),  # nuclear-heavy
        ("DE", 0.330, 0.420, 25000),
        ("PL", 0.690, 0.780, 8500),
        ("ES", 0.140, 0.210, 12000),
        ("IT", 0.220, 0.300, 9800),
        ("NL", 0.260, 0.340, 14500),
    ]
    for cc, lo, hi, qty in eu_country_lb:
        out.append(case(
            case_id=f"elec_eu_{cc.lower()}_lb_001",
            description=f"Purchased grid electricity, {cc} national grid average, location-based",
            quantity=qty,
            unit="kWh",
            metadata={"country": cc, "year": 2027, "scope": "scope2"},
            method_profile="corporate_scope2_location_based",
            factor_id=None,  # EU country LB factor IDs vary by source feed
            factor_family="grid_intensity",
            source_authority="AIB",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["location-based"],
            tags=["electricity", "eu", "location_based", cc.lower()],
        ))

    # 6 EU AIB residual-mix
    eu_residual = [
        ("FR", 0.060, 0.110, 9000),
        ("DE", 0.470, 0.560, 16000),
        ("ES", 0.330, 0.420, 7800),
        ("IT", 0.420, 0.510, 6200),
        ("NL", 0.430, 0.520, 10400),
        ("PL", 0.760, 0.870, 5500),
    ]
    for cc, lo, hi, qty in eu_residual:
        out.append(case(
            case_id=f"elec_eu_{cc.lower()}_residual_001",
            description=(
                f"Residual-mix electricity, {cc}, market-based with no contractual instrument"
            ),
            quantity=qty,
            unit="kWh",
            metadata={"country": cc, "year": 2027, "scope": "scope2",
                      "instrument": None},
            method_profile="corporate_scope2_market_based",
            factor_id=f"EF:{cc}:residual_mix:2026:v1.0",
            factor_family="residual_mix",
            source_authority="AIB",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["residual mix", "AIB", "market-based"],
            tags=["electricity", "eu", "residual_mix", cc.lower()],
        ))

    # 3 EU supplier-specific REGO/GO certificates
    for i, (cc, lo, hi, qty, certificate) in enumerate(
        [
            ("FR", 0.0, 0.005, 8000, "EECS-GO hydro"),
            ("DE", 0.0, 0.010, 12000, "EECS-GO wind"),
            ("ES", 0.005, 0.020, 5500, "GdO solar"),
        ],
        start=1,
    ):
        out.append(case(
            case_id=f"elec_eu_{cc.lower()}_supplier_{i:03d}",
            description=(
                f"Market-based electricity backed by {certificate}, {cc}"
            ),
            quantity=qty,
            unit="kWh",
            metadata={"country": cc, "year": 2027, "scope": "scope2",
                      "instrument": "GO"},
            method_profile="corporate_scope2_market_based",
            factor_id=None,
            factor_family="grid_intensity",
            source_authority="supplier",
            fallback_rank=2,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=[
                "market-based", "guarantee of origin", "supplier-specific"],
            tags=["electricity", "eu", "market_based", "go", cc.lower()],
        ))

    # ---- UK — 10 cases ----
    uk_lb = [
        ("UK national grid 2026 (DESNZ)", 0.180, 0.220, 14000, 2026,
         "EF:DESNZ:grid_electricity_kwh:UK:2026:v1", "location-based"),
        ("UK national grid 2025 (DESNZ)", 0.190, 0.240, 8200, 2025,
         "EF:DESNZ:grid_electricity_kwh:UK:2025:v1", "location-based"),
        ("UK national grid 2027 (DESNZ)", 0.170, 0.210, 19500, 2027,
         "EF:DESNZ:grid_electricity_kwh:UK:2027:v1", "location-based"),
        ("UK T&D losses 2026 (DESNZ)", 0.012, 0.018, 14000, 2026,
         "EF:DESNZ:t_d_losses_kwh:UK:2026:v1", "location-based"),
    ]
    for i, (desc, lo, hi, qty, year, fid, basis) in enumerate(uk_lb, start=1):
        out.append(case(
            case_id=f"elec_uk_lb_{year}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="kWh",
            metadata={"country": "UK", "year": year, "scope": "scope2"},
            method_profile="corporate_scope2_location_based",
            factor_id=fid,
            factor_family="grid_intensity",
            source_authority="DESNZ",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=[basis, "DESNZ"],
            tags=["electricity", "uk", "location_based"],
        ))

    # 4 UK residual mix (DESNZ residual)
    for i, (year, lo, hi, qty) in enumerate(
        [(2026, 0.260, 0.330, 14000), (2025, 0.270, 0.340, 8200),
         (2027, 0.250, 0.320, 19500), (2024, 0.280, 0.360, 7600)],
        start=1,
    ):
        out.append(case(
            case_id=f"elec_uk_residual_{year}_{i:03d}",
            description=f"UK national residual-mix electricity {year}, market-based",
            quantity=qty,
            unit="kWh",
            metadata={"country": "UK", "year": year, "scope": "scope2",
                      "instrument": None},
            method_profile="corporate_scope2_market_based",
            factor_id=f"EF:UK:residual_mix:{year}:v1.0",
            factor_family="residual_mix",
            source_authority="DESNZ",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["residual mix", "REGO netting", "DESNZ"],
            tags=["electricity", "uk", "residual_mix"],
        ))

    # 2 UK REGO-backed supplier
    for i, (cert, lo, hi, qty) in enumerate(
        [("REGO 100% wind", 0.0, 0.010, 9500),
         ("REGO 100% hydro", 0.0, 0.010, 6300)], start=1):
        out.append(case(
            case_id=f"elec_uk_supplier_{i:03d}",
            description=f"UK market-based electricity backed by {cert}",
            quantity=qty,
            unit="kWh",
            metadata={"country": "UK", "year": 2027, "scope": "scope2",
                      "instrument": "REGO"},
            method_profile="corporate_scope2_market_based",
            factor_id=None,
            factor_family="grid_intensity",
            source_authority="supplier",
            fallback_rank=2,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["market-based", "REGO", "supplier-specific"],
            tags=["electricity", "uk", "market_based", "rego"],
        ))

    # ---- US — 15 cases ----
    # 9 eGRID subregion location-based (ranges from eGRID 2022 public data)
    egrid_subregions = [
        ("RFCE", 0.300, 0.380, 22000),
        ("RFCM", 0.430, 0.520, 18500),
        ("RFCW", 0.470, 0.570, 14000),
        ("SRMW", 0.500, 0.620, 9700),
        ("SERC", 0.330, 0.420, 30000),
        ("ERCT", 0.370, 0.460, 25500),
        ("CAMX", 0.180, 0.260, 12000),
        ("NWPP", 0.260, 0.340, 8000),
        ("NEWE", 0.220, 0.300, 6400),
    ]
    for sub, lo, hi, qty in egrid_subregions:
        out.append(case(
            case_id=f"elec_us_{sub.lower()}_lb_001",
            description=f"Purchased grid electricity, US eGRID {sub}, location-based",
            quantity=qty,
            unit="kWh",
            metadata={"country": "US", "grid_region": sub, "year": 2026,
                      "scope": "scope2"},
            method_profile="corporate_scope2_location_based",
            factor_id=f"EF:eGRID:{sub.lower()}:US:2026:v1",
            factor_family="grid_intensity",
            source_authority="eGRID",
            fallback_rank=4,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["location-based", "eGRID", sub],
            tags=["electricity", "us", "location_based", sub.lower()],
        ))

    # 4 US Green-e residual mix (subregion)
    us_residual = [
        ("RFCE", 0.420, 0.520, 11000),
        ("CAMX", 0.260, 0.350, 8500),
        ("ERCT", 0.460, 0.560, 14400),
        ("SERC", 0.430, 0.520, 9300),
    ]
    for sub, lo, hi, qty in us_residual:
        out.append(case(
            case_id=f"elec_us_{sub.lower()}_residual_001",
            description=f"US residual-mix electricity, eGRID {sub}, Green-e",
            quantity=qty,
            unit="kWh",
            metadata={"country": "US", "grid_region": sub, "year": 2026,
                      "scope": "scope2", "instrument": None},
            method_profile="corporate_scope2_market_based",
            factor_id=f"EF:US:residual_mix:{sub.lower()}:2026:v1.0",
            factor_family="residual_mix",
            source_authority="Green-e",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["residual mix", "Green-e", "REC netting"],
            tags=["electricity", "us", "residual_mix", sub.lower()],
        ))

    # 2 US supplier-specific REC
    for i, (sub, lo, hi, qty, cert) in enumerate(
        [("CAMX", 0.0, 0.020, 7800, "Green-e Energy REC solar"),
         ("ERCT", 0.0, 0.020, 9100, "Green-e Energy REC wind")],
        start=1,
    ):
        out.append(case(
            case_id=f"elec_us_supplier_{i:03d}",
            description=f"US market-based electricity backed by {cert}",
            quantity=qty,
            unit="kWh",
            metadata={"country": "US", "grid_region": sub, "year": 2026,
                      "scope": "scope2", "instrument": "REC"},
            method_profile="corporate_scope2_market_based",
            factor_id=None,
            factor_family="grid_intensity",
            source_authority="supplier",
            fallback_rank=2,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/kWh",
            must_include_assumptions=["market-based", "REC", "supplier-specific"],
            tags=["electricity", "us", "market_based", "rec"],
        ))
    return out


# ---------------------------------------------------------------------------
# 2. FUEL COMBUSTION (50 cases)
# ---------------------------------------------------------------------------


def build_fuel_combustion_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Natural gas — 12 cases
    # IPCC tier-1 NG combustion: 56.1 kgCO2/GJ ≈ 1.93 kgCO2e/m³ ≈
    # 0.184 kgCO2e/kWh net.
    ng_cases = [
        # (geo, unit, quantity, lo, hi, source_authority, label)
        ("US", "therm", 12000, 5.20, 5.40, "EPA", "stationary combustion"),
        ("US", "MMBtu", 850, 52.0, 54.0, "EPA", "stationary combustion"),
        ("US", "scf", 1500000, 0.0540, 0.0570, "EPA", "stationary combustion"),
        ("UK", "kWh", 240000, 0.180, 0.190, "DESNZ", "gross CV stationary"),
        ("UK", "m3", 22500, 2.00, 2.07, "DESNZ", "gross CV stationary"),
        ("EU", "GJ", 1850, 55.5, 57.0, "IPCC", "tier-1 default"),
        ("DE", "kWh", 130000, 0.180, 0.190, "DESNZ", "gross CV proxy"),
        ("FR", "kWh", 88000, 0.180, 0.190, "DESNZ", "gross CV proxy"),
        ("IN", "m3", 18500, 1.93, 2.05, "IPCC", "tier-1 default"),
        ("IN", "kg", 14200, 2.65, 2.78, "IPCC", "tier-1 default"),
        ("US", "therm", 35000, 5.20, 5.40, "EPA", "commercial boiler"),
        ("UK", "kWh", 412000, 0.180, 0.190, "DESNZ", "industrial boiler"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(ng_cases, start=1):
        out.append(case(
            case_id=f"fuel_ng_{geo.lower()}_{unit}_{i:03d}",
            description=f"Natural gas combustion, stationary, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "natural_gas"},
            method_profile="corporate_scope1",
            factor_id=None if src == "IPCC" else (
                f"EF:DESNZ:s1_natural_gas_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:natural_gas_{unit}:US:2026:v1"
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["stationary combustion", label],
            tags=["fuel", "natural_gas", geo.lower()],
        ))

    # Diesel — 10 cases (incl. mobile + stationary)
    diesel_cases = [
        ("US", "gallon", 4200, 10.10, 10.30, "EPA", "stationary genset"),
        ("US", "gallon", 9800, 10.10, 10.30, "EPA", "mobile, on-road HDV"),
        ("UK", "litre", 18500, 2.50, 2.70, "DESNZ", "stationary"),
        ("UK", "litre", 32000, 2.50, 2.70, "DESNZ", "mobile, articulated truck"),
        ("EU", "litre", 24000, 2.60, 2.75, "IPCC", "tier-1 stationary"),
        ("DE", "litre", 18500, 2.60, 2.75, "IPCC", "tier-1 stationary"),
        ("FR", "litre", 8400, 2.60, 2.75, "IPCC", "tier-1 stationary"),
        ("IN", "litre", 22000, 2.65, 2.80, "IPCC", "tier-1 stationary"),
        ("IN", "litre", 36500, 2.65, 2.80, "IPCC", "mobile, BS-VI HDV"),
        ("US", "gallon", 2400, 10.10, 10.30, "EPA", "marine bunkered"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(diesel_cases, start=1):
        out.append(case(
            case_id=f"fuel_diesel_{geo.lower()}_{unit}_{i:03d}",
            description=f"Diesel combustion, {label}, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "diesel"},
            method_profile="corporate_scope1",
            factor_id=(
                f"EF:DESNZ:s1_diesel_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:diesel_{unit}:US:2026:v1" if src == "EPA"
                else None
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[label, "diesel"],
            tags=["fuel", "diesel", geo.lower()],
        ))

    # Gasoline / Petrol — 8 cases
    gasoline_cases = [
        ("US", "gallon", 1500, 8.70, 8.92, "EPA", "mobile light-duty"),
        ("US", "gallon", 4200, 8.70, 8.92, "EPA", "fleet light-duty"),
        ("UK", "litre", 9800, 2.18, 2.30, "DESNZ", "average car"),
        ("UK", "litre", 4200, 2.18, 2.30, "DESNZ", "small car"),
        ("EU", "litre", 6500, 2.20, 2.34, "IPCC", "tier-1 mobile"),
        ("DE", "litre", 12000, 2.20, 2.34, "IPCC", "tier-1 mobile"),
        ("IN", "litre", 8400, 2.30, 2.45, "IPCC", "tier-1 mobile"),
        ("IN", "litre", 17500, 2.30, 2.45, "IPCC", "BS-VI passenger car"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(gasoline_cases, start=1):
        out.append(case(
            case_id=f"fuel_gasoline_{geo.lower()}_{unit}_{i:03d}",
            description=f"Gasoline (petrol) combustion, {label}, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "gasoline"},
            method_profile="corporate_scope1",
            factor_id=(
                f"EF:DESNZ:s1_gasoline_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:gasoline_{unit}:US:2026:v1" if src == "EPA"
                else None
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[label, "gasoline"],
            tags=["fuel", "gasoline", geo.lower()],
        ))

    # Coal — 8 cases
    coal_cases = [
        ("US", "short_ton", 250, 2240.0, 2400.0, "EPA", "bituminous boiler"),
        ("US", "short_ton", 180, 1900.0, 2080.0, "EPA", "subbituminous boiler"),
        ("UK", "tonne", 95, 2380.0, 2520.0, "DESNZ", "industrial bituminous"),
        ("EU", "tonne", 1200, 2200.0, 2520.0, "IPCC", "hard coal tier-1"),
        ("DE", "tonne", 850, 2200.0, 2520.0, "IPCC", "hard coal tier-1"),
        ("IN", "tonne", 18500, 1850.0, 2050.0, "IPCC", "Indian coal tier-1"),
        ("IN", "tonne", 24500, 1850.0, 2050.0, "IPCC", "thermal power plant"),
        ("IN", "tonne", 6700, 950.0, 1150.0, "IPCC", "lignite tier-1"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(coal_cases, start=1):
        out.append(case(
            case_id=f"fuel_coal_{geo.lower()}_{unit}_{i:03d}",
            description=f"Coal combustion, {label}, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "coal"},
            method_profile="corporate_scope1",
            factor_id=(
                f"EF:DESNZ:s1_coal_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:coal_{unit}:US:2026:v1" if src == "EPA"
                else None
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[label, "coal"],
            tags=["fuel", "coal", geo.lower()],
        ))

    # LPG — 6 cases
    lpg_cases = [
        ("US", "gallon", 1200, 5.65, 5.85, "EPA", "stationary heating"),
        ("UK", "litre", 4500, 1.50, 1.62, "DESNZ", "stationary heating"),
        ("UK", "kg", 2300, 2.85, 3.00, "DESNZ", "stationary heating"),
        ("EU", "kg", 8800, 2.95, 3.10, "IPCC", "tier-1 stationary"),
        ("IN", "kg", 14500, 2.95, 3.10, "IPCC", "commercial cooking"),
        ("IN", "kg", 4200, 2.95, 3.10, "IPCC", "industrial process heat"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(lpg_cases, start=1):
        out.append(case(
            case_id=f"fuel_lpg_{geo.lower()}_{unit}_{i:03d}",
            description=f"LPG combustion, {label}, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "lpg"},
            method_profile="corporate_scope1",
            factor_id=(
                f"EF:DESNZ:s1_lpg_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:lpg_{unit}:US:2026:v1" if src == "EPA"
                else None
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[label, "LPG"],
            tags=["fuel", "lpg", geo.lower()],
        ))

    # Fuel oil (residual / distillate) — 6 cases
    fuel_oil_cases = [
        ("US", "gallon", 2400, 11.20, 11.60, "EPA", "residual fuel oil #6"),
        ("US", "gallon", 1800, 10.20, 10.50, "EPA", "distillate fuel oil #2"),
        ("UK", "litre", 6800, 3.10, 3.30, "DESNZ", "burning oil"),
        ("UK", "litre", 9800, 3.18, 3.36, "DESNZ", "fuel oil"),
        ("EU", "tonne", 220, 3100.0, 3260.0, "IPCC", "residual fuel oil tier-1"),
        ("IN", "tonne", 480, 3050.0, 3220.0, "IPCC", "industrial furnace oil"),
    ]
    for i, (geo, unit, qty, lo, hi, src, label) in enumerate(fuel_oil_cases, start=1):
        out.append(case(
            case_id=f"fuel_oil_{geo.lower()}_{unit}_{i:03d}",
            description=f"Fuel oil combustion, {label}, {geo}",
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": "fuel_oil"},
            method_profile="corporate_scope1",
            factor_id=(
                f"EF:DESNZ:s1_fuel_oil_{unit}:{geo}:2026:v1" if src == "DESNZ"
                else f"EF:EPA:fuel_oil_{unit}:US:2026:v1" if src == "EPA"
                else None
            ),
            factor_family="emissions",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[label, "fuel oil"],
            tags=["fuel", "fuel_oil", geo.lower()],
        ))

    return out


# ---------------------------------------------------------------------------
# 3. REFRIGERANTS (30 cases)
# ---------------------------------------------------------------------------


# AR5 100-yr GWP and AR6 100-yr GWP for the 7 refrigerants in scope.
# Sources:
#   AR5: IPCC AR5 WG1 Annex II, Table 8.A.1
#   AR6: IPCC AR6 WG1 Chapter 7, Table 7.SM.7
REFRIGERANTS = {
    # name : (AR5 lo, AR5 hi, AR6 lo, AR6 hi, family_hint)
    "R-22":     (1760, 1820, 1750, 1810, "HCFC, phase-out"),
    "R-32":     (670,  690,  760,  790,  "low-GWP HFC, R-410A successor"),
    "R-134a":   (1290, 1320, 1490, 1550, "single-component HFC"),
    "R-410A":   (1900, 2090, 2240, 2340, "R-32/R-125 50/50 blend"),
    "R-404A":   (3900, 4000, 4720, 4810, "R-125/R-143a/R-134a blend"),
    "R-507":    (3940, 4020, 4400, 4500, "R-125/R-143a 50/50 blend"),
    "R-1234yf": (1, 5, 1, 5, "HFO single-component"),
}


def build_refrigerant_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # 7 refrigerants × ≈4-5 cases each (AR5/AR6 × geos/charges) → 30 total
    pairs = [
        ("US", "kg", 12.5, "EPA", 2026, "rooftop AC leak"),
        ("US", "kg", 6.0, "EPA", 2026, "fleet MAC service"),
        ("UK", "kg", 8.0, "DESNZ", 2026, "supermarket refrigeration"),
        ("EU", "kg", 14.0, "IPCC", 2026, "industrial chiller leak"),
        ("IN", "kg", 22.0, "IPCC", 2026, "cold chain warehouse"),
    ]
    cur_id = 0
    for ref_name, (ar5_lo, ar5_hi, ar6_lo, ar6_hi, family_hint) in REFRIGERANTS.items():
        # 2 AR5 cases + 2-3 AR6 cases per refrigerant
        ar5_count = 2
        ar6_count = (
            3 if ref_name in ("R-410A", "R-32", "R-134a", "R-22")  # commonly used
            else 2
        )
        cases_for_ref = pairs[:(ar5_count + ar6_count)]
        for j, (geo, unit, qty, src, year, scenario) in enumerate(cases_for_ref):
            cur_id += 1
            is_ar6 = j >= ar5_count
            gwp_lo, gwp_hi = (ar6_lo, ar6_hi) if is_ar6 else (ar5_lo, ar5_hi)
            basis = "IPCC_AR6_100" if is_ar6 else "IPCC_AR5_100"
            out.append(case(
                case_id=f"ref_{ref_name.lower().replace('-', '_')}_{basis.lower()}_{cur_id:03d}",
                description=(
                    f"Refrigerant {ref_name} fugitive emission, {scenario}, "
                    f"{geo} ({basis})"
                ),
                quantity=qty,
                unit=unit,
                metadata={"country": geo, "year": year, "scope": "scope1",
                          "refrigerant": ref_name, "gwp_basis": basis,
                          "fuel_type": "refrigerant"},
                method_profile="corporate_scope1",
                factor_id=f"EF:IPCC:{ref_name.lower().replace('-', '_')}:GLOBAL:{basis.lower()}:v1",
                factor_family="refrigerant_gwp",
                source_authority="IPCC",
                fallback_rank=6,
                co2e_min=float(gwp_lo),
                co2e_max=float(gwp_hi),
                co2e_unit="kgCO2e/kg",
                must_include_assumptions=[
                    "fugitive emission", basis, family_hint, ref_name],
                tags=["refrigerant", ref_name.lower(), basis.lower(), geo.lower()],
            ))
    return out


# ---------------------------------------------------------------------------
# 4. FREIGHT (50 cases)
# ---------------------------------------------------------------------------


def build_freight_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Road — 15 cases (WTW + TTW per GLEC)
    road = [
        # (mode, geo, payload_t, distance_km, lo, hi, scope, label)
        ("HGV ≥40t artic, EU avg", "EU", 18.0, 850, 0.080, 0.110, "WTW", "diesel"),
        ("HGV ≥40t artic, EU avg", "EU", 24.0, 1200, 0.080, 0.110, "TTW", "diesel"),
        ("HGV 26-40t rigid, UK", "UK", 14.0, 320, 0.110, 0.150, "WTW", "diesel"),
        ("HGV 7.5-17t rigid, UK", "UK", 6.0, 180, 0.180, 0.230, "WTW", "diesel"),
        ("Van <3.5t, UK last-mile", "UK", 0.6, 120, 0.450, 0.620, "WTW", "diesel"),
        ("HGV ≥40t artic, US class 8", "US", 22.0, 2400, 0.060, 0.085, "WTW", "diesel"),
        ("HGV class 6 box truck, US", "US", 4.0, 350, 0.180, 0.240, "WTW", "diesel"),
        ("HGV BS-VI 40t, India NH", "IN", 18.0, 1400, 0.090, 0.120, "WTW", "diesel"),
        ("HGV BS-VI 25t, India SH", "IN", 12.0, 600, 0.110, 0.140, "WTW", "diesel"),
        ("HGV BS-VI 16t, India urban", "IN", 6.0, 220, 0.180, 0.230, "WTW", "diesel"),
        ("HGV ≥40t empty backhaul, EU", "EU", 0.0, 400, 0.450, 0.700, "TTW",
         "empty running"),
        ("Refrigerated HGV 26t, EU", "EU", 14.0, 950, 0.180, 0.230, "WTW",
         "refrigerated, diesel + reefer load"),
        ("HGV electric 19t, EU pilot", "EU", 9.0, 280, 0.040, 0.080, "WTW",
         "battery electric"),
        ("HGV LNG 40t, EU", "EU", 18.0, 1050, 0.060, 0.090, "WTW", "LNG"),
        ("HGV BEV 26t, US California", "US", 12.0, 320, 0.030, 0.080, "WTW",
         "battery electric"),
    ]
    for i, (mode, geo, payload, dist, lo, hi, basis, label) in enumerate(road, start=1):
        out.append(case(
            case_id=f"freight_road_{geo.lower()}_{i:03d}",
            description=f"Road freight: {mode}, {payload}t × {dist}km ({basis})",
            quantity=payload * dist,
            unit="tonne_km",
            metadata={"country": geo, "year": 2026, "mode": "road",
                      "boundary": basis, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority="GLEC",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/tonne_km",
            must_include_assumptions=[basis, "road freight", label, "GLEC"],
            tags=["freight", "road", basis.lower(), geo.lower()],
        ))

    # Sea — 12 cases
    sea = [
        ("Container ship 8000 TEU, deepsea", "GLOBAL", 6.5, 19500, 0.0080, 0.0120,
         "WTW", "container deepsea"),
        ("Container ship 4000 TEU, intra-Asia", "GLOBAL", 4.0, 4200, 0.0140, 0.0180,
         "WTW", "container short-sea"),
        ("Container ship 14000 TEU, Asia-EU", "GLOBAL", 8.0, 21500, 0.0070, 0.0110,
         "WTW", "ULCV container"),
        ("Bulk carrier Capesize 180kt", "GLOBAL", 150.0, 18000, 0.0030, 0.0060,
         "WTW", "dry bulk"),
        ("Bulk carrier Panamax 75kt", "GLOBAL", 60.0, 12500, 0.0040, 0.0070,
         "WTW", "dry bulk"),
        ("Tanker VLCC 280kt", "GLOBAL", 250.0, 21000, 0.0030, 0.0050,
         "WTW", "crude oil"),
        ("Tanker MR product 50kt", "GLOBAL", 38.0, 9800, 0.0050, 0.0080,
         "WTW", "refined products"),
        ("Ro-Ro ferry, EU short-sea", "EU", 0.5, 280, 0.040, 0.070,
         "WTW", "ferry car deck"),
        ("Container feeder 1500 TEU, Med", "EU", 1.2, 1200, 0.022, 0.038,
         "WTW", "feeder"),
        ("Reefer container deepsea, Asia-EU", "GLOBAL", 6.0, 19000, 0.022, 0.030,
         "WTW", "reefer container"),
        ("Inland barge Rhine, EU", "EU", 0.8, 350, 0.030, 0.045,
         "WTW", "inland barge"),
        ("LNG carrier 170k m³, GLOBAL", "GLOBAL", 80.0, 18000, 0.0060, 0.0090,
         "WTW", "LNG dual fuel"),
    ]
    for i, (mode, geo, payload, dist, lo, hi, basis, label) in enumerate(sea, start=1):
        out.append(case(
            case_id=f"freight_sea_{i:03d}",
            description=f"Maritime freight: {mode}, {payload} kt × {dist}km ({basis})",
            quantity=payload * dist * 1000,  # kt → tonne
            unit="tonne_km",
            metadata={"country": geo, "year": 2026, "mode": "sea",
                      "boundary": basis, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority="GLEC",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/tonne_km",
            must_include_assumptions=[basis, "maritime freight", label, "GLEC"],
            tags=["freight", "sea", basis.lower()],
        ))

    # Air — 12 cases
    air = [
        ("Wide-body freighter long-haul", "GLOBAL", 90.0, 9500, 0.50, 0.80,
         "WTW", "B777F long-haul"),
        ("Wide-body freighter trans-Pacific", "GLOBAL", 90.0, 11500, 0.50, 0.80,
         "WTW", "B777F trans-Pacific"),
        ("Narrow-body freighter regional", "GLOBAL", 18.0, 1800, 0.85, 1.20,
         "WTW", "B737F regional"),
        ("Belly-hold long-haul, EU-US", "EU", 12.0, 6800, 0.45, 0.70,
         "WTW", "wide-body belly"),
        ("Belly-hold short-haul intra-EU", "EU", 4.0, 1200, 0.95, 1.40,
         "WTW", "narrow-body belly"),
        ("Belly-hold trans-Atlantic, US-EU", "US", 14.0, 7500, 0.45, 0.70,
         "WTW", "wide-body belly"),
        ("Air freight intra-Asia, narrow body", "GLOBAL", 10.0, 2800, 0.85, 1.20,
         "WTW", "narrow-body freighter"),
        ("Long-haul freighter, Asia-EU", "GLOBAL", 100.0, 9000, 0.50, 0.80,
         "WTW", "B747-8F"),
        ("Domestic India air freight", "IN", 6.0, 1600, 0.85, 1.30,
         "WTW", "narrow-body domestic"),
        ("Long-haul belly, India-EU", "IN", 12.0, 7100, 0.45, 0.70,
         "WTW", "wide-body belly"),
        ("Express integrator EU intra", "EU", 3.0, 1100, 0.95, 1.40,
         "WTW", "express integrator"),
        ("Domestic US air freight, narrow", "US", 8.0, 2200, 0.85, 1.30,
         "WTW", "narrow-body domestic"),
    ]
    for i, (mode, geo, payload, dist, lo, hi, basis, label) in enumerate(air, start=1):
        out.append(case(
            case_id=f"freight_air_{i:03d}",
            description=f"Air freight: {mode}, {payload}t × {dist}km ({basis})",
            quantity=payload * dist,
            unit="tonne_km",
            metadata={"country": geo, "year": 2026, "mode": "air",
                      "boundary": basis, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority="GLEC",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/tonne_km",
            must_include_assumptions=[basis, "air freight", label, "GLEC"],
            tags=["freight", "air", basis.lower(), geo.lower()],
        ))

    # Rail — 6 cases
    rail = [
        ("Diesel rail freight EU, intermodal", "EU", 600, 1200, 0.020, 0.035,
         "WTW", "diesel locomotive"),
        ("Electric rail freight EU, container", "EU", 600, 950, 0.012, 0.022,
         "WTW", "electric, EU grid mix"),
        ("Diesel rail freight US, class I", "US", 1200, 2500, 0.015, 0.025,
         "WTW", "BNSF/UP class I"),
        ("Electric rail freight UK, intermodal", "UK", 400, 320, 0.015, 0.025,
         "WTW", "electric, UK grid"),
        ("Indian Railways freight, electric", "IN", 1500, 1100, 0.015, 0.030,
         "WTW", "IR electric WAG"),
        ("Indian Railways freight, diesel", "IN", 1200, 800, 0.022, 0.038,
         "WTW", "IR diesel WDG"),
    ]
    for i, (mode, geo, payload, dist, lo, hi, basis, label) in enumerate(rail, start=1):
        out.append(case(
            case_id=f"freight_rail_{geo.lower()}_{i:03d}",
            description=f"Rail freight: {mode}, {payload}t × {dist}km ({basis})",
            quantity=payload * dist,
            unit="tonne_km",
            metadata={"country": geo, "year": 2026, "mode": "rail",
                      "boundary": basis, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority="GLEC",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/tonne_km",
            must_include_assumptions=[basis, "rail freight", label, "GLEC"],
            tags=["freight", "rail", basis.lower(), geo.lower()],
        ))

    # Intermodal — 5 cases
    intermodal = [
        ("Sea + rail + drayage Asia-EU", "GLOBAL", 18.0, 21000, 0.012, 0.022,
         "WTW", "sea-dominant"),
        ("Rail + truck US transcon, intermodal", "US", 25.0, 4200, 0.025, 0.045,
         "WTW", "rail-dominant"),
        ("Truck + ferry + truck UK-Ireland", "UK", 14.0, 720, 0.060, 0.110,
         "WTW", "truck-dominant short-sea"),
        ("India ICD multi-modal hub-spoke", "IN", 22.0, 1800, 0.050, 0.090,
         "WTW", "rail+truck"),
        ("EU multi-modal short-sea + truck", "EU", 16.0, 1100, 0.040, 0.080,
         "WTW", "sea+truck balanced"),
    ]
    for i, (mode, geo, payload, dist, lo, hi, basis, label) in enumerate(intermodal, start=1):
        out.append(case(
            case_id=f"freight_intermodal_{geo.lower()}_{i:03d}",
            description=f"Intermodal freight: {mode}, {payload}t × {dist}km ({basis})",
            quantity=payload * dist,
            unit="tonne_km",
            metadata={"country": geo, "year": 2026, "mode": "intermodal",
                      "boundary": basis, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority="GLEC",
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit="kgCO2e/tonne_km",
            must_include_assumptions=[basis, "intermodal", label, "GLEC"],
            tags=["freight", "intermodal", basis.lower(), geo.lower()],
        ))

    return out


# ---------------------------------------------------------------------------
# 5. MATERIALS (50 cases)
# ---------------------------------------------------------------------------


def build_materials_cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Steel — 10 cases (BF-BOF + EAF + scrap mixes)
    steel = [
        ("Steel BF-BOF primary, EU avg", "EU", 1500, 2.10, 2.40, "EPD/Worldsteel",
         "BF-BOF primary"),
        ("Steel BF-BOF primary, China", "CN", 1200, 2.40, 2.80, "EPD/Worldsteel",
         "BF-BOF primary"),
        ("Steel BF-BOF primary, India", "IN", 800, 2.30, 2.65, "EPD/Worldsteel",
         "BF-BOF primary"),
        ("Steel BF-BOF primary, JP", "JP", 600, 2.05, 2.30, "EPD/Worldsteel",
         "BF-BOF primary"),
        ("Steel EAF scrap, EU avg", "EU", 900, 0.40, 0.80, "EPD/Worldsteel",
         "EAF scrap-route"),
        ("Steel EAF scrap, US avg", "US", 1100, 0.40, 0.85, "EPD/Worldsteel",
         "EAF scrap-route"),
        ("Steel EAF DRI, India", "IN", 750, 1.20, 1.60, "EPD/Worldsteel",
         "EAF DRI mix"),
        ("Steel HRC, EU EPD avg", "EU", 1400, 1.80, 2.20, "EPD",
         "hot-rolled coil"),
        ("Steel rebar, India market avg", "IN", 1800, 1.80, 2.30, "EPD",
         "rebar"),
        ("Steel hot-dipped galvanized, EU", "EU", 1200, 2.40, 2.80, "EPD",
         "HDG coated"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(steel, start=1):
        out.append(case(
            case_id=f"mat_steel_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "steel"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "steel"],
            tags=["material", "steel", geo.lower()],
        ))

    # Aluminium — 8 cases
    aluminium = [
        ("Primary Al ingot, EU IAI", "EU", 600, 6.50, 8.20, "IAI/EPD",
         "primary, EU mix"),
        ("Primary Al ingot, China", "CN", 950, 16.0, 19.0, "IAI/EPD",
         "primary, China coal mix"),
        ("Primary Al ingot, Iceland (hydro)", "IS", 280, 2.20, 3.50, "IAI/EPD",
         "primary, hydro"),
        ("Primary Al ingot, India", "IN", 480, 14.0, 18.0, "IAI/EPD",
         "primary, India coal mix"),
        ("Recycled Al ingot, EU avg", "EU", 320, 0.40, 0.80, "IAI/EPD",
         "secondary recycled"),
        ("Recycled Al ingot, US avg", "US", 410, 0.40, 0.80, "IAI/EPD",
         "secondary recycled"),
        ("Aluminium extrusion EU EPD", "EU", 220, 4.50, 6.20, "EPD",
         "extruded profile"),
        ("Aluminium sheet rolled, EU", "EU", 180, 5.20, 7.00, "EPD",
         "rolled sheet"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(aluminium, start=1):
        out.append(case(
            case_id=f"mat_al_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "aluminium"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "aluminium"],
            tags=["material", "aluminium", geo.lower()],
        ))

    # Cement — 8 cases
    cement = [
        ("CEM I OPC, EU EPD avg", "EU", 2200, 0.80, 0.95, "EPD",
         "OPC clinker-rich"),
        ("CEM II Portland-composite, EU", "EU", 1800, 0.55, 0.75, "EPD",
         "blended cement"),
        ("CEM III blast furnace slag, EU", "EU", 1200, 0.40, 0.55, "EPD",
         "GGBFS-blended"),
        ("OPC PPC, India avg", "IN", 3500, 0.65, 0.85, "EPD",
         "fly-ash blended"),
        ("OPC 53 grade, India", "IN", 2800, 0.85, 1.05, "EPD",
         "OPC clinker-rich"),
        ("Type II Portland, US avg", "US", 1400, 0.85, 1.00, "EPD",
         "OPC"),
        ("Limestone Portland LC3, EU pilot", "EU", 600, 0.40, 0.55, "EPD",
         "LC3"),
        ("Type IL portland-limestone, US", "US", 1100, 0.65, 0.85, "EPD",
         "PLC type IL"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(cement, start=1):
        out.append(case(
            case_id=f"mat_cement_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "cement"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "cement"],
            tags=["material", "cement", geo.lower()],
        ))

    # Plastics — 8 cases
    plastics = [
        ("PE-LD virgin, EU PlasticsEurope", "EU", 90, 1.85, 2.20, "PlasticsEurope EPD",
         "low-density polyethylene"),
        ("PE-HD virgin, EU PlasticsEurope", "EU", 120, 1.75, 2.05, "PlasticsEurope EPD",
         "high-density polyethylene"),
        ("PP virgin, EU PlasticsEurope", "EU", 150, 1.85, 2.10, "PlasticsEurope EPD",
         "polypropylene"),
        ("PET resin virgin, EU PCI", "EU", 200, 2.10, 2.45, "EPD",
         "polyethylene terephthalate"),
        ("PVC suspension, EU PlasticsEurope", "EU", 180, 2.20, 2.55, "PlasticsEurope EPD",
         "polyvinyl chloride"),
        ("Recycled PET (rPET) flake, EU", "EU", 80, 0.80, 1.20, "EPD",
         "rPET secondary"),
        ("PE-LD virgin, US APR avg", "US", 110, 1.80, 2.15, "EPD",
         "LDPE US"),
        ("PP virgin, India market avg", "IN", 220, 1.95, 2.30, "EPD",
         "PP India"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(plastics, start=1):
        out.append(case(
            case_id=f"mat_plastic_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "plastic"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "plastic"],
            tags=["material", "plastic", geo.lower()],
        ))

    # Paper — 6 cases
    paper = [
        ("Kraft pulp bleached, EU CEPI", "EU", 80, 0.80, 1.20, "CEPI EPD",
         "bleached kraft pulp"),
        ("Newsprint, EU CEPI", "EU", 65, 0.40, 0.80, "CEPI EPD", "newsprint"),
        ("Corrugated case material, EU", "EU", 120, 0.45, 0.80, "CEPI EPD",
         "OCC + virgin mix"),
        ("Office paper, EU avg", "EU", 35, 0.85, 1.20, "CEPI EPD", "office paper"),
        ("Corrugated CCM, US avg", "US", 200, 0.55, 0.95, "EPD", "OCC US"),
        ("Office paper, India avg", "IN", 28, 0.95, 1.40, "EPD", "office paper IN"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(paper, start=1):
        out.append(case(
            case_id=f"mat_paper_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "paper"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "paper"],
            tags=["material", "paper", geo.lower()],
        ))

    # Fertilizer — 5 cases
    fertilizer = [
        ("Urea N fertilizer, EU avg", "EU", 220, 1.80, 3.50, "Fertilizers Europe EPD",
         "urea, EU SCR"),
        ("Urea N fertilizer, India avg", "IN", 1500, 2.80, 4.20, "EPD",
         "urea, India natural-gas"),
        ("Ammonium nitrate, EU avg", "EU", 180, 2.80, 4.50, "Fertilizers Europe EPD",
         "AN, EU"),
        ("DAP fertilizer, EU avg", "EU", 160, 1.20, 2.10, "EPD",
         "diammonium phosphate"),
        ("Compound NPK 15-15-15, EU", "EU", 200, 1.50, 2.80, "EPD",
         "NPK blended"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(fertilizer, start=1):
        out.append(case(
            case_id=f"mat_fert_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "fertilizer"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "fertilizer"],
            tags=["material", "fertilizer", geo.lower()],
        ))

    # Glass — 5 cases
    glass = [
        ("Container glass virgin, EU FEVE", "EU", 60, 0.80, 1.20, "FEVE EPD",
         "container glass"),
        ("Float glass virgin, EU Glass-for-Europe", "EU", 80, 0.95, 1.40, "EPD",
         "float glass"),
        ("Container glass 60% recycled cullet, EU", "EU", 50, 0.55, 0.80, "FEVE EPD",
         "high cullet"),
        ("Float glass, US avg", "US", 70, 1.00, 1.45, "EPD", "float glass US"),
        ("Container glass, India avg", "IN", 40, 0.95, 1.45, "EPD", "container IN"),
    ]
    for i, (desc, geo, qty, lo, hi, src, label) in enumerate(glass, start=1):
        out.append(case(
            case_id=f"mat_glass_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit="tonne",
            metadata={"country": geo, "year": 2026, "scope": "scope3",
                      "category": "purchased_goods", "material": "glass"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=["cradle-to-gate", label, "glass"],
            tags=["material", "glass", geo.lower()],
        ))

    return out


# ---------------------------------------------------------------------------
# 6. CBAM (35 cases)
# ---------------------------------------------------------------------------


def build_cbam_cases() -> List[Dict[str, Any]]:
    """CBAM cases use EU default values from Implementing Regulation
    2023/1773 + Annex II of Reg. 2023/956 + the 2024-2025 default tables."""
    out: List[Dict[str, Any]] = []

    # Iron and Steel — 12 cases (CN codes 7206-7229)
    steel = [
        # (desc, origin_country, qty, lo, hi, cn_code, label)
        ("Steel hot-rolled coil, China",  "CN", 250, 2.40, 2.80, "7208",
         "EU CBAM default 2.06 t embedded direct + indirect"),
        ("Steel cold-rolled coil, China", "CN", 180, 2.50, 2.95, "7209",
         "CBAM default direct"),
        ("Steel rebar, India",            "IN", 320, 2.30, 2.65, "7214",
         "Indian BF-BOF with CBAM default fallback"),
        ("Steel rebar, Turkey",           "TR", 200, 1.20, 1.60, "7214",
         "Turkish EAF scrap-route"),
        ("Steel HDG sheet, India",        "IN", 150, 2.40, 2.80, "7210",
         "Indian primary steel + galvanizing"),
        ("Iron ore pellets, Brazil",      "BR", 1500, 0.10, 0.18, "2601",
         "iron ore preparation"),
        ("Pig iron, Russia",              "RU", 800, 1.80, 2.20, "7201",
         "blast furnace pig iron"),
        ("Steel wire, China",             "CN", 90, 2.60, 3.00, "7217",
         "drawn wire"),
        ("Steel tubes, India",            "IN", 220, 2.40, 2.80, "7304",
         "seamless tubes"),
        ("Stainless steel, China",        "CN", 110, 5.50, 7.00, "7219",
         "stainless 304 grade"),
        ("Pre-painted steel, Korea",      "KR", 130, 2.60, 3.10, "7210",
         "Korean coated steel"),
        ("Slab semi-finished, Ukraine",   "UA", 600, 1.95, 2.30, "7207",
         "semi-finished slab"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(steel, start=1):
        out.append(case(
            case_id=f"cbam_steel_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="tonne",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "iron_and_steel",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:steel:{origin}:2024:v1",
            factor_family="material_embodied",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "embedded emissions"],
            tags=["cbam", "steel", origin.lower(), f"cn{cn}"],
        ))

    # Aluminium — 8 cases (CN 7601-7616)
    aluminium = [
        ("Primary Al ingot, China",  "CN", 80, 18.0, 22.0, "7601",
         "China coal-fired smelter"),
        ("Primary Al ingot, India",  "IN", 65, 16.0, 19.0, "7601",
         "India coal-fired smelter"),
        ("Primary Al ingot, UAE",    "AE", 50, 8.0, 11.0, "7601",
         "UAE gas-fired smelter"),
        ("Primary Al ingot, Russia", "RU", 90, 4.0, 7.0, "7601",
         "Russian hydro smelter"),
        ("Aluminium wire/rod, China","CN", 30, 18.5, 23.0, "7605",
         "China primary + drawing"),
        ("Aluminium foil, China",    "CN", 25, 19.0, 23.5, "7607",
         "rolled foil"),
        ("Aluminium structural, India","IN", 40, 16.5, 20.0, "7610",
         "structural extrusion"),
        ("Aluminium tube, Turkey",   "TR", 20, 12.0, 15.5, "7608",
         "Turkish extrusion"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(aluminium, start=1):
        out.append(case(
            case_id=f"cbam_al_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="tonne",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "aluminium",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:aluminium:{origin}:2024:v1",
            factor_family="material_embodied",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "embedded emissions"],
            tags=["cbam", "aluminium", origin.lower(), f"cn{cn}"],
        ))

    # Cement — 5 cases (CN 2523)
    cement = [
        ("Cement clinker, Turkey",  "TR", 1200, 0.85, 1.05, "252310",
         "clinker default"),
        ("Cement Portland, Turkey", "TR", 800, 0.65, 0.85, "252329",
         "Portland cement"),
        ("Cement clinker, Egypt",   "EG", 1500, 0.85, 1.05, "252310",
         "clinker default"),
        ("Cement Portland, India",  "IN", 950, 0.70, 0.90, "252329",
         "Indian PPC"),
        ("White cement, Egypt",     "EG", 220, 1.00, 1.30, "252321",
         "white cement"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(cement, start=1):
        out.append(case(
            case_id=f"cbam_cement_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="tonne",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "cement",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:cement:{origin}:2024:v1",
            factor_family="material_embodied",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "embedded emissions"],
            tags=["cbam", "cement", origin.lower(), f"cn{cn}"],
        ))

    # Fertilizers — 5 cases (CN 2808, 2814, 2834, 3102, 3105)
    fertilizers = [
        ("Urea, Russia",            "RU", 600, 1.80, 2.40, "310210",
         "urea CBAM default"),
        ("Ammonia, Russia",         "RU", 400, 2.20, 2.85, "281410",
         "ammonia CBAM default"),
        ("Ammonium nitrate, Egypt", "EG", 350, 2.50, 3.50, "310230",
         "AN CBAM default"),
        ("Nitric acid, Egypt",      "EG", 200, 5.50, 6.50, "280800",
         "nitric acid + N2O abatement"),
        ("NPK fertilizer, Morocco", "MA", 280, 1.60, 2.40, "310520",
         "compound NPK"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(fertilizers, start=1):
        out.append(case(
            case_id=f"cbam_fert_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="tonne",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "fertilizers",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:fertilizer:{origin}:2024:v1",
            factor_family="material_embodied",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "embedded emissions"],
            tags=["cbam", "fertilizer", origin.lower(), f"cn{cn}"],
        ))

    # Hydrogen — 3 cases (CN 280410)
    hydrogen = [
        ("Grey H2, SMR, Saudi Arabia", "SA", 50, 9.0, 11.0, "280410",
         "SMR fossil hydrogen"),
        ("Grey H2, SMR, Russia",      "RU", 80, 9.0, 11.0, "280410",
         "SMR fossil hydrogen"),
        ("Blue H2, SMR+CCS, Norway",  "NO", 30, 1.5, 4.0, "280410",
         "SMR with carbon capture"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(hydrogen, start=1):
        out.append(case(
            case_id=f"cbam_h2_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="tonne",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "hydrogen",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:hydrogen:{origin}:2024:v1",
            factor_family="material_embodied",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/tonne",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "embedded emissions"],
            tags=["cbam", "hydrogen", origin.lower(), f"cn{cn}"],
        ))

    # Electricity — 2 cases (CN 27160000)
    electricity = [
        ("Imported electricity from non-EU, Serbia", "RS", 5000, 0.700, 0.900, "27160000",
         "fossil-heavy import"),
        ("Imported electricity from non-EU, Ukraine", "UA", 2500, 0.450, 0.700, "27160000",
         "mixed import"),
    ]
    for i, (desc, origin, qty, lo, hi, cn, label) in enumerate(electricity, start=1):
        out.append(case(
            case_id=f"cbam_elec_{origin.lower()}_{i:03d}",
            description=f"CBAM goods import: {desc} (CN {cn})",
            quantity=qty,
            unit="MWh",
            metadata={"country": origin, "year": 2026,
                      "cbam_product": "electricity",
                      "cn_code": cn, "scope": "embedded_emissions"},
            method_profile="eu_cbam",
            factor_id=f"EF:CBAM:electricity:{origin}:2024:v1",
            factor_family="grid_intensity",
            source_authority="EU CBAM",
            fallback_rank=6,
            co2e_min=lo * 1000,  # MWh
            co2e_max=hi * 1000,
            co2e_unit="kgCO2e/MWh",
            must_include_assumptions=[
                "CBAM default value", label, f"CN code {cn}", "imported electricity"],
            tags=["cbam", "electricity", origin.lower(), f"cn{cn}"],
        ))

    return out


# ---------------------------------------------------------------------------
# 7. METHODOLOGY PROFILES (25 cases)
# ---------------------------------------------------------------------------


def build_methodology_profile_cases() -> List[Dict[str, Any]]:
    """Cross-cutting cases that exercise method-profile selection routing."""
    out: List[Dict[str, Any]] = []

    # Corporate scope 1 — 5 cases
    s1 = [
        ("Stationary natural-gas boiler, US", "US", 12500, "MMBtu", 52.0, 54.0,
         "natural_gas", "EPA"),
        ("Mobile diesel HDV, EU", "EU", 9800, "litre", 2.60, 2.75,
         "diesel", "IPCC"),
        ("Process emissions cement kiln CO2, EU", "EU", 1100, "tonne_clinker",
         460.0, 540.0, "process_co2", "IPCC"),
        ("Refrigerant R-410A leak, US", "US", 6.0, "kg", 2240.0, 2340.0,
         "refrigerant", "IPCC"),
        ("Stationary LPG, IN", "IN", 4500, "kg", 2.95, 3.10,
         "lpg", "IPCC"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, fuel, src) in enumerate(s1, start=1):
        out.append(case(
            case_id=f"meth_scope1_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope1",
                      "fuel_type": fuel},
            method_profile="corporate_scope1",
            factor_id=None,
            factor_family="emissions" if fuel != "refrigerant" else "refrigerant_gwp",
            source_authority=src,
            fallback_rank=5 if src in ("EPA", "DESNZ") else 6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["scope 1", "GHG Protocol", fuel],
            tags=["methodology", "scope1", geo.lower(), fuel],
        ))

    # Corporate scope 2 (location) — 3 cases
    s2_lb = [
        ("Purchased electricity, India southern grid, location-based", "IN",
         9800, "kWh", 0.660, 0.720, "CEA"),
        ("Purchased electricity, US eGRID NWPP, location-based", "US",
         8000, "kWh", 0.260, 0.340, "eGRID"),
        ("Purchased electricity, UK national grid 2026, location-based", "UK",
         14000, "kWh", 0.180, 0.220, "DESNZ"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src) in enumerate(s2_lb, start=1):
        out.append(case(
            case_id=f"meth_scope2_lb_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope2"},
            method_profile="corporate_scope2_location_based",
            factor_id=None,
            factor_family="grid_intensity",
            source_authority=src,
            fallback_rank=4,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["scope 2 location-based", "GHG Protocol", src],
            tags=["methodology", "scope2_lb", geo.lower()],
        ))

    # Corporate scope 2 (market) — 3 cases
    s2_mb = [
        ("Purchased electricity, EU residual mix DE, market-based", "DE",
         16000, "kWh", 0.470, 0.560, "AIB", "residual_mix"),
        ("Purchased electricity, EU GO-backed FR, market-based", "FR",
         8000, "kWh", 0.0, 0.005, "supplier", "grid_intensity"),
        ("Purchased electricity, US REC-backed CAMX, market-based", "US",
         7800, "kWh", 0.0, 0.020, "supplier", "grid_intensity"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src, family) in enumerate(s2_mb, start=1):
        out.append(case(
            case_id=f"meth_scope2_mb_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope2",
                      "instrument": "GO" if src == "supplier" and geo == "FR"
                      else ("REC" if src == "supplier" else None)},
            method_profile="corporate_scope2_market_based",
            factor_id=None,
            factor_family=family,
            source_authority=src,
            fallback_rank=2 if src == "supplier" else 5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["scope 2 market-based", "GHG Protocol"],
            tags=["methodology", "scope2_mb", geo.lower()],
        ))

    # Corporate scope 3 — 3 cases
    s3 = [
        ("Cat 1 purchased steel HRC, EU", "EU", 1400, "tonne",
         1800.0, 2200.0, "EPD", "material_embodied"),
        ("Cat 4 upstream road freight, EU", "EU", 18000, "tonne_km",
         0.080, 0.110, "GLEC", "transport_lane"),
        ("Cat 6 business travel air long-haul", "GLOBAL", 14000, "tonne_km",
         0.45, 0.70, "DESNZ", "transport_lane"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src, family) in enumerate(s3, start=1):
        out.append(case(
            case_id=f"meth_scope3_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope3"},
            method_profile="corporate_scope3",
            factor_id=None,
            factor_family=family,
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["scope 3", "GHG Protocol"],
            tags=["methodology", "scope3", geo.lower()],
        ))

    # GHG Protocol Product / ISO 14067 (product carbon) — 3 cases
    pcf = [
        ("Product carbon footprint: 1 kg cheese", "EU", 1.0, "kg",
         8.0, 12.0, "EPD", "ISO 14067 cradle-to-gate"),
        ("Product carbon footprint: 1 unit T-shirt cotton", "GLOBAL", 1.0, "unit",
         3.0, 8.0, "EPD", "GHG Protocol Product"),
        ("Product carbon footprint: 1 kg whey protein", "EU", 1.0, "kg",
         12.0, 18.0, "EPD", "ISO 14067"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src, basis) in enumerate(pcf, start=1):
        out.append(case(
            case_id=f"meth_pcf_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026,
                      "category": "product_carbon"},
            method_profile="product_carbon",
            factor_id=None,
            factor_family="material_embodied",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=[basis, "product carbon", "PACT"],
            tags=["methodology", "product_carbon", geo.lower()],
        ))

    # Freight ISO 14083 — 3 cases
    fr = [
        ("ISO 14083 road lane, EU 40t artic", "EU", 21000, "tonne_km",
         0.080, 0.110, "GLEC"),
        ("ISO 14083 sea lane, container deepsea", "GLOBAL", 130000000, "tonne_km",
         0.0070, 0.0120, "GLEC"),
        ("ISO 14083 air lane, wide-body long-haul", "GLOBAL", 855000, "tonne_km",
         0.45, 0.80, "GLEC"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src) in enumerate(fr, start=1):
        out.append(case(
            case_id=f"meth_freight_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope3"},
            method_profile="freight_iso_14083",
            factor_id=None,
            factor_family="transport_lane",
            source_authority=src,
            fallback_rank=5,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["ISO 14083", "GLEC", "well-to-wheel"],
            tags=["methodology", "freight_iso_14083", geo.lower()],
        ))

    # PCAF — 3 cases
    pcaf = [
        ("PCAF listed equity, manufacturing", "GLOBAL", 50000000, "USD",
         0.10, 0.40, "PCAF", "scope 1+2 attribution"),
        ("PCAF business loan, agriculture", "GLOBAL", 12000000, "USD",
         0.15, 0.55, "PCAF", "scope 1+2 attribution"),
        ("PCAF mortgage, residential building", "GLOBAL", 850000, "USD",
         0.02, 0.10, "PCAF", "scope 1+2 building energy"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src, label) in enumerate(pcaf, start=1):
        out.append(case(
            case_id=f"meth_pcaf_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "scope": "scope3"},
            method_profile="finance_proxy",
            factor_id=None,
            factor_family="finance_proxy",
            source_authority=src,
            fallback_rank=6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["PCAF", "Global Standard v2", label],
            tags=["methodology", "pcaf", "finance_proxy"],
        ))

    # Land removals — 2 cases
    land = [
        ("Afforestation removal, temperate broadleaf", "GLOBAL", 1.0, "ha",
         -8000.0, -3000.0, "IPCC", "GHG LSR"),
        ("Soil organic carbon, no-till cropland", "US", 1.0, "ha",
         -1500.0, -300.0, "IPCC", "GHG LSR"),
    ]
    for i, (desc, geo, qty, unit, lo, hi, src, label) in enumerate(land, start=1):
        out.append(case(
            case_id=f"meth_land_{geo.lower()}_{i:03d}",
            description=desc,
            quantity=qty,
            unit=unit,
            metadata={"country": geo, "year": 2026, "category": "removals"},
            method_profile="land_removals",
            factor_id=None,
            factor_family="land_use_removals",
            source_authority=src,
            fallback_rank=6,
            co2e_min=lo,
            co2e_max=hi,
            co2e_unit=f"kgCO2e/{unit}",
            must_include_assumptions=["GHG LSR", "removal", label, "IPCC AFOLU"],
            tags=["methodology", "land_removals", geo.lower()],
        ))

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FAMILY_BUILDERS = [
    ("electricity",          build_electricity_cases),
    ("fuel_combustion",      build_fuel_combustion_cases),
    ("refrigerants",         build_refrigerant_cases),
    ("freight",              build_freight_cases),
    ("materials",            build_materials_cases),
    ("cbam",                 build_cbam_cases),
    ("methodology_profiles", build_methodology_profile_cases),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    families: List[Dict[str, Any]] = []
    total = 0
    for name, builder in FAMILY_BUILDERS:
        cases = builder()
        path = OUT_DIR / f"{name}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cases, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        families.append({"family": name, "case_count": len(cases),
                         "file": f"{name}.json"})
        total += len(cases)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(cases)} cases)")

    index = {
        "version": "1.0",
        "case_count": total,
        "families": families,
        "created": TODAY,
        "schema_version": "1.0",
        "schema_doc": "greenlang/factors/data/gold_set/README.md",
    }
    index_path = OUT_DIR / "index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {index_path.relative_to(REPO_ROOT)} (total = {total})")


if __name__ == "__main__":
    main()
