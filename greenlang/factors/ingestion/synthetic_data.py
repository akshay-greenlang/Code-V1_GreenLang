# -*- coding: utf-8 -*-
"""
Synthetic emission factor generator for development and testing (F019-SYN).

Generates realistic but clearly-marked synthetic emission factors covering
all major fuel types, geographies, scopes, and boundaries. Useful for:
- Populating development/staging catalogs to 10K-50K rows
- Performance testing the Factors API at scale
- Integration testing pipelines without real source data

All generated factors are marked with ``source_id="synthetic"`` and
``factor_status="preview"`` so they are never confused with certified data.

Example:
    >>> from greenlang.factors.ingestion.synthetic_data import generate_synthetic_factors
    >>> factor_dicts = generate_synthetic_factors(count=25000, seed=42)
    >>> len(factor_dicts)
    25000
"""

from __future__ import annotations

import hashlib
import logging
import random
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference data tables: fuel types, geographies, scopes, boundaries
# ---------------------------------------------------------------------------

# Each tuple: (fuel_slug, display_name, unit, typical_co2_kg, scope, boundary, heating_value_basis)
# typical_co2_kg is the approximate kg CO2 per unit for baseline generation.
FUEL_PROFILES: List[Tuple[str, str, str, float, str, str, Optional[str]]] = [
    # Liquid fuels - Scope 1 Combustion
    ("diesel", "Diesel", "gallons", 10.18, "1", "combustion", "HHV"),
    ("gasoline", "Motor Gasoline", "gallons", 8.78, "1", "combustion", "HHV"),
    ("jet_fuel", "Jet Fuel (Kerosene-Type)", "gallons", 9.75, "1", "combustion", "HHV"),
    ("kerosene", "Kerosene", "gallons", 9.75, "1", "combustion", "HHV"),
    ("fuel_oil", "Residual Fuel Oil No. 6", "gallons", 11.27, "1", "combustion", "HHV"),
    ("lpg", "Liquefied Petroleum Gas", "gallons", 5.79, "1", "combustion", "HHV"),
    ("propane", "Propane", "gallons", 5.72, "1", "combustion", "HHV"),
    ("butane", "Butane", "gallons", 6.67, "1", "combustion", "HHV"),
    ("ethanol", "Ethanol (E100)", "gallons", 5.56, "1", "combustion", "HHV"),
    ("biodiesel", "Biodiesel (B100)", "gallons", 9.45, "1", "combustion", "HHV"),
    ("crude_oil", "Crude Oil", "gallons", 10.29, "1", "combustion", "HHV"),
    ("aviation_gasoline", "Aviation Gasoline", "gallons", 8.32, "1", "combustion", "HHV"),
    # Gaseous fuels - Scope 1 Combustion
    ("natural_gas", "Natural Gas", "scf", 0.0551, "1", "combustion", "HHV"),
    ("cng", "Compressed Natural Gas", "scf", 0.0551, "1", "combustion", "HHV"),
    ("lng", "Liquefied Natural Gas", "gallons", 4.46, "1", "combustion", "HHV"),
    ("biogas", "Biogas / Landfill Gas", "scf", 0.0340, "1", "combustion", "HHV"),
    ("landfill_gas", "Landfill Gas", "scf", 0.0326, "1", "combustion", "HHV"),
    # Solid fuels - Scope 1 Combustion
    ("anthracite_coal", "Anthracite Coal", "tonnes", 2602.0, "1", "combustion", "HHV"),
    ("bituminous_coal", "Bituminous Coal", "tonnes", 2328.0, "1", "combustion", "HHV"),
    ("sub_bituminous_coal", "Sub-Bituminous Coal", "tonnes", 1826.0, "1", "combustion", "HHV"),
    ("lignite_coal", "Lignite Coal", "tonnes", 1389.0, "1", "combustion", "HHV"),
    ("coal_coke", "Coal Coke", "tonnes", 2875.0, "1", "combustion", "HHV"),
    ("petroleum_coke", "Petroleum Coke", "tonnes", 3072.0, "1", "combustion", "HHV"),
    ("wood_waste", "Wood and Wood Waste", "tonnes", 1640.0, "1", "combustion", None),
    ("msw", "Municipal Solid Waste", "tonnes", 902.0, "1", "combustion", None),
    # Electricity - Scope 2
    ("electricity_grid", "Grid Electricity", "kwh", 0.386, "2", "combustion", None),
    ("electricity_renewable", "Renewable Electricity", "kwh", 0.0, "2", "combustion", None),
    # Steam/Heat - Scope 2
    ("steam", "Purchased Steam", "mmbtu", 66.33, "2", "combustion", None),
    ("district_cooling", "District Cooling", "kwh", 0.21, "2", "combustion", None),
    ("district_heating", "District Heating", "kwh", 0.18, "2", "combustion", None),
    # Scope 3 - WTT upstream
    ("diesel_wtt", "Diesel (Well-to-Tank)", "gallons", 2.43, "3", "WTT", None),
    ("gasoline_wtt", "Gasoline (Well-to-Tank)", "gallons", 2.18, "3", "WTT", None),
    ("natural_gas_wtt", "Natural Gas (Well-to-Tank)", "scf", 0.0129, "3", "WTT", None),
    ("electricity_t_d", "Electricity T&D Losses", "kwh", 0.029, "3", "combustion", None),
    # Scope 3 - Cradle-to-gate materials
    ("steel_bof", "Steel (BOF Route)", "kg_product", 2.33, "3", "cradle_to_gate", None),
    ("aluminium_primary", "Primary Aluminium", "kg_product", 8.14, "3", "cradle_to_gate", None),
    ("cement_portland", "Portland Cement", "kg_product", 0.83, "3", "cradle_to_gate", None),
    ("plastic_pet", "PET Plastic", "kg_product", 2.73, "3", "cradle_to_gate", None),
    ("paper_virgin", "Virgin Paper", "kg_product", 1.09, "3", "cradle_to_gate", None),
    ("glass_container", "Container Glass", "kg_product", 0.86, "3", "cradle_to_gate", None),
    ("fertilizer_urea", "Urea Fertilizer", "kg_product", 3.24, "3", "cradle_to_gate", None),
]

# Geography pools: (iso_code, geography_level, region_hint_or_none)
GEOGRAPHIES: List[Tuple[str, str, Optional[str]]] = [
    # Major countries
    ("US", "country", None),
    ("GB", "country", None),
    ("DE", "country", None),
    ("FR", "country", None),
    ("IT", "country", None),
    ("ES", "country", None),
    ("NL", "country", None),
    ("BE", "country", None),
    ("SE", "country", None),
    ("NO", "country", None),
    ("DK", "country", None),
    ("FI", "country", None),
    ("AT", "country", None),
    ("CH", "country", None),
    ("IE", "country", None),
    ("PT", "country", None),
    ("PL", "country", None),
    ("CZ", "country", None),
    ("RO", "country", None),
    ("HU", "country", None),
    ("GR", "country", None),
    ("BG", "country", None),
    ("HR", "country", None),
    ("SK", "country", None),
    ("LT", "country", None),
    ("LV", "country", None),
    ("EE", "country", None),
    ("SI", "country", None),
    ("LU", "country", None),
    ("MT", "country", None),
    ("CY", "country", None),
    ("CA", "country", None),
    ("AU", "country", None),
    ("NZ", "country", None),
    ("JP", "country", None),
    ("KR", "country", None),
    ("CN", "country", None),
    ("IN", "country", None),
    ("BR", "country", None),
    ("MX", "country", None),
    ("ZA", "country", None),
    ("SG", "country", None),
    ("MY", "country", None),
    ("TH", "country", None),
    ("ID", "country", None),
    ("PH", "country", None),
    ("VN", "country", None),
    ("AE", "country", None),
    ("SA", "country", None),
    ("EG", "country", None),
    # US states (selected)
    ("US-CA", "state", "CA"),
    ("US-TX", "state", "TX"),
    ("US-NY", "state", "NY"),
    ("US-FL", "state", "FL"),
    ("US-PA", "state", "PA"),
    ("US-IL", "state", "IL"),
    ("US-OH", "state", "OH"),
    ("US-GA", "state", "GA"),
    ("US-WA", "state", "WA"),
    ("US-MA", "state", "MA"),
    # Regions/continents
    ("EU27", "continent", None),
    ("OECD", "continent", None),
    ("GLOBAL", "global", None),
    ("APAC", "continent", None),
    ("LATAM", "continent", None),
    ("MENA", "continent", None),
]

# Synthetic source configurations
SYNTHETIC_SOURCES = [
    ("synthetic_epa", "Synthetic EPA-like", "US-Public-Domain", True),
    ("synthetic_defra", "Synthetic DEFRA-like", "OGL-UK", True),
    ("synthetic_ipcc", "Synthetic IPCC-like", "CC-BY-4.0", True),
    ("synthetic_cbam", "Synthetic CBAM-like", "EU-legal-text", False),
    ("synthetic_egrid", "Synthetic eGRID-like", "US-Public-Domain", True),
    ("synthetic_general", "Synthetic General", "CC0-1.0", True),
]

# GWP set options for variation
GWP_SETS = [
    ("IPCC_AR6_100", 27.9, 273.0),
    ("IPCC_AR5_100", 28.0, 265.0),
]

# Methodology options
METHODOLOGIES = [
    "IPCC_Tier_1",
    "IPCC_Tier_2",
    "IPCC_Tier_3",
    "direct_measurement",
    "lifecycle_assessment",
]

# Compliance framework combinations
COMPLIANCE_COMBOS = [
    ["GHG_Protocol", "IPCC_2006"],
    ["GHG_Protocol", "IPCC_2006", "EPA_MRR"],
    ["GHG_Protocol", "IPCC_2006", "ISO_14064"],
    ["GHG_Protocol"],
]


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _deterministic_seed(factor_id: str) -> int:
    """Generate a deterministic seed from factor_id for reproducibility."""
    return int(hashlib.md5(factor_id.encode("utf-8")).hexdigest()[:8], 16)


def _jitter(base: float, pct: float, rng: random.Random) -> float:
    """Apply +/- pct jitter to base value. Never returns negative."""
    factor = 1.0 + rng.uniform(-pct, pct)
    return max(0.0, base * factor)


def _build_ghg_vectors(
    co2_base: float,
    fuel_slug: str,
    rng: random.Random,
) -> Dict[str, float]:
    """Build realistic GHG vector from a CO2 baseline and fuel profile.

    Uses the IPCC decomposition ratios embedded in GHGVectors to produce
    realistic CH4 and N2O values relative to CO2.
    """
    # Apply geographic/temporal jitter to CO2 (up to +/- 15%)
    co2 = _jitter(co2_base, 0.15, rng)

    # CH4/N2O ratios by fuel category (simplified from IPCC profiles)
    ch4_ratio_map = {
        "natural_gas": 0.008, "cng": 0.008, "lng": 0.009,
        "biogas": 0.012, "landfill_gas": 0.015,
        "coal": 0.003, "anthracite_coal": 0.002, "bituminous_coal": 0.004,
        "sub_bituminous_coal": 0.004, "lignite_coal": 0.005,
        "wood_waste": 0.010, "msw": 0.012,
        "diesel": 0.0008, "gasoline": 0.0012, "jet_fuel": 0.0005,
        "electricity_renewable": 0.0, "electricity_grid": 0.0002,
    }
    n2o_ratio_map = {
        "natural_gas": 0.002, "cng": 0.002, "lng": 0.002,
        "biogas": 0.003, "landfill_gas": 0.003,
        "coal": 0.005, "anthracite_coal": 0.004, "bituminous_coal": 0.005,
        "sub_bituminous_coal": 0.005, "lignite_coal": 0.006,
        "wood_waste": 0.005, "msw": 0.005,
        "diesel": 0.003, "gasoline": 0.003, "jet_fuel": 0.003,
        "electricity_renewable": 0.0, "electricity_grid": 0.0001,
    }

    ch4_ratio = ch4_ratio_map.get(fuel_slug, 0.002)
    n2o_ratio = n2o_ratio_map.get(fuel_slug, 0.004)

    # Apply jitter to ratios too
    ch4 = co2 * _jitter(ch4_ratio, 0.20, rng)
    n2o = co2 * _jitter(n2o_ratio, 0.20, rng)

    # Renewable electricity has zero emissions
    if fuel_slug == "electricity_renewable":
        co2, ch4, n2o = 0.0, 0.0, 0.0

    return {
        "CO2": round(co2, 8),
        "CH4": round(ch4, 8),
        "N2O": round(n2o, 8),
    }


def _build_dqs(rng: random.Random, geo_level: str) -> Dict[str, int]:
    """Generate realistic DQS scores. Higher for country-level, lower for global."""
    base_geo = {"country": 4, "state": 5, "grid_zone": 5}.get(geo_level, 3)
    return {
        "temporal": rng.choice([4, 5]),
        "geographical": min(5, max(1, base_geo + rng.randint(-1, 1))),
        "technological": rng.choice([3, 4, 4, 5]),
        "representativeness": rng.choice([3, 3, 4, 4, 5]),
        "methodological": rng.choice([4, 4, 5, 5]),
    }


def _select_source(rng: random.Random) -> Tuple[str, str, str, bool]:
    """Pick a synthetic source configuration."""
    return rng.choice(SYNTHETIC_SOURCES)


def generate_factor_dict(
    fuel_profile: Tuple[str, str, str, float, str, str, Optional[str]],
    geo: Tuple[str, str, Optional[str]],
    year: int,
    version: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Generate a single synthetic emission factor dict.

    Args:
        fuel_profile: Fuel specification tuple from FUEL_PROFILES.
        geo: Geography tuple from GEOGRAPHIES.
        year: Reporting year.
        version: Factor version number.
        rng: Random instance for reproducibility.

    Returns:
        Dict compatible with ``EmissionFactorRecord.from_dict()``.
    """
    fuel_slug, fuel_name, unit, co2_base, scope, boundary, hvb = fuel_profile
    geo_code, geo_level, region_hint = geo

    source_id, source_name, license_id, redist = _select_source(rng)
    gwp_set_name, ch4_gwp, n2o_gwp = rng.choice(GWP_SETS)
    methodology = rng.choice(METHODOLOGIES)

    factor_id = f"EF:{source_id}:{fuel_slug}:{geo_code}:{year}:v{version}"

    # Use factor_id as seed for deterministic vector generation
    factor_rng = random.Random(_deterministic_seed(factor_id))
    vectors = _build_ghg_vectors(co2_base, fuel_slug, factor_rng)
    dqs = _build_dqs(factor_rng, geo_level)

    uncertainty = round(rng.uniform(0.03, 0.20), 4)
    compliance = rng.choice(COMPLIANCE_COMBOS)

    # Build tags
    tags = ["synthetic", fuel_slug, scope, geo_code.lower()]
    if boundary == "WTT":
        tags.append("wtt")
    if boundary == "cradle_to_gate":
        tags.append("cradle_to_gate")

    # Activity and sector tags
    activity_tags = [fuel_slug]
    if scope == "1":
        activity_tags.append("combustion")
    elif scope == "2":
        activity_tags.append("purchased_energy")
    elif scope == "3":
        activity_tags.append("value_chain")

    sector_tags = []
    if fuel_slug in ("electricity_grid", "electricity_renewable", "steam", "district_heating", "district_cooling"):
        sector_tags.append("energy")
    elif fuel_slug.endswith("_wtt") or fuel_slug == "electricity_t_d":
        sector_tags.append("energy")
        sector_tags.append("upstream")
    elif fuel_slug in ("steel_bof", "aluminium_primary", "cement_portland"):
        sector_tags.append("industrial")
        sector_tags.append("materials")
    elif fuel_slug in ("plastic_pet", "paper_virgin", "glass_container"):
        sector_tags.append("manufacturing")
    elif fuel_slug == "fertilizer_urea":
        sector_tags.append("agriculture")
    else:
        sector_tags.append("energy")

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "factor_id": factor_id,
        "fuel_type": fuel_slug,
        "unit": unit,
        "geography": geo_code,
        "geography_level": geo_level,
        "region_hint": region_hint,
        "vectors": vectors,
        "gwp_100yr": {
            "gwp_set": gwp_set_name,
            "CH4_gwp": ch4_gwp,
            "N2O_gwp": n2o_gwp,
        },
        "scope": scope,
        "boundary": boundary,
        "provenance": {
            "source_org": source_name,
            "source_publication": f"{source_name} Synthetic Emission Factors {year}",
            "source_year": year,
            "methodology": methodology,
            "source_url": None,
            "version": f"v{version}",
        },
        "valid_from": date(year, 1, 1).isoformat(),
        "valid_to": date(year, 12, 31).isoformat(),
        "uncertainty_95ci": uncertainty,
        "dqs": dqs,
        "license_info": {
            "license": license_id,
            "redistribution_allowed": redist,
            "commercial_use_allowed": True,
            "attribution_required": True,
        },
        "heating_value_basis": hvb,
        "biogenic_flag": fuel_slug in (
            "wood_waste", "biogas", "landfill_gas", "biodiesel",
            "ethanol", "msw",
        ),
        "compliance_frameworks": compliance,
        "tags": tags,
        "notes": f"Synthetic factor for {fuel_name} in {geo_code} ({year}). NOT FOR PRODUCTION USE.",
        "created_at": now,
        "updated_at": now,
        "created_by": "greenlang_synthetic_generator",
        # CTO governance fields
        "factor_status": "preview",
        "source_id": "synthetic",
        "source_release": f"synthetic-{year}",
        "source_record_id": factor_id,
        "release_version": f"syn-{year}.{version}",
        "validation_flags": {"synthetic": True, "not_for_production": True},
        "replacement_factor_id": None,
        "license_class": "synthetic_dev",
        "activity_tags": activity_tags,
        "sector_tags": sector_tags,
    }


def generate_synthetic_factors(
    count: int = 25000,
    seed: int = 42,
    years: Optional[List[int]] = None,
    max_versions: int = 2,
) -> List[Dict[str, Any]]:
    """Generate a batch of synthetic emission factor dicts.

    Iterates through all combinations of fuels, geographies, and years,
    generating unique factors up to ``count``. The generation is
    deterministic given the same ``seed``.

    Args:
        count: Target number of factors to generate (10,000 to 50,000).
        seed: Random seed for reproducibility.
        years: Reporting years to cover (default: [2022, 2023, 2024, 2025]).
        max_versions: Maximum version variants per factor (1-3).

    Returns:
        List of factor dicts ready for QA validation and catalog insertion.

    Example:
        >>> dicts = generate_synthetic_factors(count=10000, seed=123)
        >>> assert all(d["factor_status"] == "preview" for d in dicts)
        >>> assert all(d["source_id"] == "synthetic" for d in dicts)
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]
    if count < 1:
        raise ValueError("count must be >= 1")

    rng = random.Random(seed)
    factors: List[Dict[str, Any]] = []
    seen_ids: set = set()

    logger.info(
        "Starting synthetic factor generation: target=%d seed=%d years=%s fuels=%d geos=%d",
        count, seed, years, len(FUEL_PROFILES), len(GEOGRAPHIES),
    )

    # Calculate how many passes through the combinatorial space we need
    combos_per_pass = len(FUEL_PROFILES) * len(GEOGRAPHIES)
    total_combos = combos_per_pass * len(years) * max_versions
    logger.info(
        "Combinatorial space: %d fuel*geo per year, %d total possible",
        combos_per_pass, total_combos,
    )

    # Shuffle order for diversity in early truncation
    fuel_list = list(FUEL_PROFILES)
    geo_list = list(GEOGRAPHIES)

    for year in years:
        for version in range(1, max_versions + 1):
            rng.shuffle(fuel_list)
            rng.shuffle(geo_list)
            for fuel_profile in fuel_list:
                for geo in geo_list:
                    if len(factors) >= count:
                        break

                    fuel_slug = fuel_profile[0]
                    geo_code = geo[0]
                    factor_id = f"EF:synthetic_{_pick_source_suffix(fuel_slug, rng)}:{fuel_slug}:{geo_code}:{year}:v{version}"

                    # Ensure uniqueness
                    if factor_id in seen_ids:
                        continue
                    seen_ids.add(factor_id)

                    fd = generate_factor_dict(fuel_profile, geo, year, version, rng)
                    factors.append(fd)

                if len(factors) >= count:
                    break
            if len(factors) >= count:
                break
        if len(factors) >= count:
            break

    # If we still need more factors, generate additional variants
    while len(factors) < count:
        fuel_profile = rng.choice(FUEL_PROFILES)
        geo = rng.choice(GEOGRAPHIES)
        year = rng.choice(years)
        version = rng.randint(1, max_versions + 5)
        fd = generate_factor_dict(fuel_profile, geo, year, version, rng)
        fid = fd["factor_id"]
        if fid not in seen_ids:
            seen_ids.add(fid)
            factors.append(fd)

    logger.info(
        "Synthetic factor generation complete: generated=%d unique_ids=%d",
        len(factors), len(seen_ids),
    )
    return factors[:count]


def _pick_source_suffix(fuel_slug: str, rng: random.Random) -> str:
    """Pick a source suffix for factor_id based on fuel characteristics."""
    # This is used internally for ID generation diversity, but the actual
    # source_id in the record is always "synthetic"
    suffixes = ["epa", "defra", "ipcc", "cbam", "egrid", "general"]
    return rng.choice(suffixes)


def generate_and_validate(
    count: int = 25000,
    seed: int = 42,
    years: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate synthetic factors and run QA validation.

    Args:
        count: Number of factors to generate.
        seed: Random seed.
        years: Reporting years.

    Returns:
        Tuple of (valid_factors, total_generated, total_rejected).
    """
    from greenlang.factors.etl.qa import validate_factor_dict

    all_dicts = generate_synthetic_factors(count=count, seed=seed, years=years)
    valid: List[Dict[str, Any]] = []
    rejected = 0

    for fd in all_dicts:
        ok, errors = validate_factor_dict(fd)
        if ok:
            valid.append(fd)
        else:
            rejected += 1
            logger.debug("Synthetic factor %s failed QA: %s", fd.get("factor_id"), errors)

    logger.info(
        "Synthetic QA complete: generated=%d valid=%d rejected=%d pass_rate=%.2f%%",
        len(all_dicts), len(valid), rejected,
        (len(valid) / max(len(all_dicts), 1)) * 100,
    )
    return valid, len(all_dicts), rejected
