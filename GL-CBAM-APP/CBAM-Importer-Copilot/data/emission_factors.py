"""
CBAM Emission Factors Database (Demo Mode - Synthetic Data)

This module contains illustrative emission factors derived from publicly available sources.
These values are for DEMONSTRATION PURPOSES ONLY and should NOT be used for actual CBAM filings.

For production CBAM filings, use official EU Commission default values published at:
https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en

Sources:
- IEA (International Energy Agency) - Cement Technology Roadmap 2018
- IPCC (Intergovernmental Panel on Climate Change) - 2006/2019 Guidelines
- World Steel Association - Steel Climate Impact Report 2023
- IAI (International Aluminum Institute) - GHG Emissions Protocol 2023
- UNFCCC - National Inventory Reports

All values are in tCO2e per ton of product (metric tons).
"""

from typing import Dict, Any, List
from datetime import datetime

# Version and metadata
VERSION = "1.0.0-demo"
LAST_UPDATED = "2025-10-15"
DISCLAIMER = (
    "⚠️  DEMO MODE: These are illustrative emission factors from public sources. "
    "For actual CBAM filings, use official EU Commission default values."
)


# ============================================================================
# CEMENT & CEMENT CLINKER
# ============================================================================

CEMENT_FACTORS = {
    "cement_portland_grey": {
        "product_name": "Portland Cement (Grey)",
        "cbam_product_group": "cement",
        "cn_codes": ["25231000"],
        "default_direct_tco2_per_ton": 0.766,
        "default_indirect_tco2_per_ton": 0.134,
        "default_total_tco2_per_ton": 0.900,
        "source": "IEA Cement Technology Roadmap 2018, Table 4.2",
        "source_url": "https://www.iea.org/reports/technology-roadmap-low-carbon-transition-in-the-cement-industry",
        "vintage": 2018,
        "uncertainty_pct": 15,
        "scope": "Cradle-to-gate including clinker production and grinding",
        "notes": "Average for grey portland cement, includes fuel combustion (direct) and electricity (indirect)",
    },
    "cement_portland_white": {
        "product_name": "Portland Cement (White)",
        "cbam_product_group": "cement",
        "cn_codes": ["25232900"],
        "default_direct_tco2_per_ton": 0.855,
        "default_indirect_tco2_per_ton": 0.150,
        "default_total_tco2_per_ton": 1.005,
        "source": "IEA Cement Technology Roadmap 2018 + industry corrections",
        "source_url": "https://www.iea.org/reports/technology-roadmap-low-carbon-transition-in-the-cement-industry",
        "vintage": 2018,
        "uncertainty_pct": 18,
        "scope": "Cradle-to-gate",
        "notes": "White cement requires more energy due to purity requirements (+12% vs grey)",
    },
    "cement_clinker": {
        "product_name": "Cement Clinker",
        "cbam_product_group": "cement",
        "cn_codes": ["25231010"],
        "default_direct_tco2_per_ton": 0.850,
        "default_indirect_tco2_per_ton": 0.100,
        "default_total_tco2_per_ton": 0.950,
        "source": "IPCC 2006 Guidelines Volume 3, Chapter 2",
        "source_url": "https://www.ipcc-nggip.iges.or.jp/public/2006gl/",
        "vintage": 2006,
        "uncertainty_pct": 12,
        "scope": "Process emissions + fuel combustion",
        "notes": "Clinker is the intermediate product before cement grinding",
    },
}


# ============================================================================
# STEEL & IRON PRODUCTS
# ============================================================================

STEEL_FACTORS = {
    "steel_basic_oxygen_furnace": {
        "product_name": "Steel (Basic Oxygen Furnace - BOF)",
        "cbam_product_group": "steel",
        "cn_codes": ["72031000", "72044100"],
        "default_direct_tco2_per_ton": 1.850,
        "default_indirect_tco2_per_ton": 0.150,
        "default_total_tco2_per_ton": 2.000,
        "source": "World Steel Association - Steel Climate Impact Report 2023",
        "source_url": "https://worldsteel.org/steel-topics/climate-change/",
        "vintage": 2023,
        "uncertainty_pct": 20,
        "scope": "Ironmaking + steelmaking (BOF route)",
        "notes": "Primary steel from iron ore via blast furnace + BOF",
    },
    "steel_electric_arc_furnace": {
        "product_name": "Steel (Electric Arc Furnace - EAF)",
        "cbam_product_group": "steel",
        "cn_codes": ["72031000", "72044900"],
        "default_direct_tco2_per_ton": 0.385,
        "default_indirect_tco2_per_ton": 0.415,
        "default_total_tco2_per_ton": 0.800,
        "source": "World Steel Association - Steel Climate Impact Report 2023",
        "source_url": "https://worldsteel.org/steel-topics/climate-change/",
        "vintage": 2023,
        "uncertainty_pct": 25,
        "scope": "Secondary steel from scrap via EAF",
        "notes": "Recycled steel, lower direct emissions but higher electricity use",
    },
    "steel_hot_rolled": {
        "product_name": "Hot-rolled Steel Products",
        "cbam_product_group": "steel",
        "cn_codes": ["72081000", "72082500", "72083600"],
        "default_direct_tco2_per_ton": 1.950,
        "default_indirect_tco2_per_ton": 0.200,
        "default_total_tco2_per_ton": 2.150,
        "source": "World Steel Association (average BOF + rolling)",
        "source_url": "https://worldsteel.org/steel-topics/climate-change/",
        "vintage": 2023,
        "uncertainty_pct": 22,
        "scope": "Steel production + hot rolling",
        "notes": "Includes upstream steelmaking and hot rolling process",
    },
    "iron_pig_iron": {
        "product_name": "Pig Iron",
        "cbam_product_group": "iron",
        "cn_codes": ["72011000"],
        "default_direct_tco2_per_ton": 1.800,
        "default_indirect_tco2_per_ton": 0.100,
        "default_total_tco2_per_ton": 1.900,
        "source": "World Steel Association + IPCC 2019",
        "source_url": "https://worldsteel.org/",
        "vintage": 2023,
        "uncertainty_pct": 18,
        "scope": "Blast furnace ironmaking",
        "notes": "Intermediate product from blast furnace",
    },
}


# ============================================================================
# ALUMINUM
# ============================================================================

ALUMINUM_FACTORS = {
    "aluminum_primary_unwrought": {
        "product_name": "Aluminum, Unwrought (Primary)",
        "cbam_product_group": "aluminum",
        "cn_codes": ["76011000"],
        "default_direct_tco2_per_ton": 1.700,
        "default_indirect_tco2_per_ton": 9.800,
        "default_total_tco2_per_ton": 11.500,
        "source": "IAI (International Aluminum Institute) - GHG Protocol 2023",
        "source_url": "https://international-aluminium.org/statistics/greenhouse-gas-emissions/",
        "vintage": 2023,
        "uncertainty_pct": 35,
        "scope": "Alumina refining + aluminum smelting (global average)",
        "notes": "Highly electricity-intensive; emissions depend heavily on grid carbon intensity",
    },
    "aluminum_alloys_unwrought": {
        "product_name": "Aluminum Alloys, Unwrought",
        "cbam_product_group": "aluminum",
        "cn_codes": ["76012000"],
        "default_direct_tco2_per_ton": 1.600,
        "default_indirect_tco2_per_ton": 9.500,
        "default_total_tco2_per_ton": 11.100,
        "source": "IAI - GHG Protocol 2023",
        "source_url": "https://international-aluminium.org/",
        "vintage": 2023,
        "uncertainty_pct": 35,
        "scope": "Primary aluminum + alloying",
        "notes": "Similar to primary aluminum with minor alloying adjustments",
    },
    "aluminum_secondary": {
        "product_name": "Aluminum (Secondary/Recycled)",
        "cbam_product_group": "aluminum",
        "cn_codes": ["76020000"],
        "default_direct_tco2_per_ton": 0.350,
        "default_indirect_tco2_per_ton": 0.250,
        "default_total_tco2_per_ton": 0.600,
        "source": "IAI - GHG Protocol 2023",
        "source_url": "https://international-aluminium.org/",
        "vintage": 2023,
        "uncertainty_pct": 30,
        "scope": "Recycled aluminum smelting",
        "notes": "Recycling saves ~95% energy compared to primary production",
    },
}


# ============================================================================
# FERTILIZERS
# ============================================================================

FERTILIZER_FACTORS = {
    "fertilizers_ammonia": {
        "product_name": "Ammonia (Anhydrous)",
        "cbam_product_group": "fertilizers",
        "cn_codes": ["28141000"],
        "default_direct_tco2_per_ton": 2.200,
        "default_indirect_tco2_per_ton": 0.300,
        "default_total_tco2_per_ton": 2.500,
        "source": "IPCC 2019 Refinement, Volume 3, Chapter 3",
        "source_url": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
        "vintage": 2019,
        "uncertainty_pct": 25,
        "scope": "Haber-Bosch process (natural gas feedstock + process heat)",
        "notes": "Natural gas is both feedstock and fuel; emissions vary by process efficiency",
    },
    "fertilizers_urea": {
        "product_name": "Urea",
        "cbam_product_group": "fertilizers",
        "cn_codes": ["31021000"],
        "default_direct_tco2_per_ton": 1.500,
        "default_indirect_tco2_per_ton": 0.200,
        "default_total_tco2_per_ton": 1.700,
        "source": "IPCC 2019 Refinement + FAO studies",
        "source_url": "https://www.ipcc-nggip.iges.or.jp/",
        "vintage": 2019,
        "uncertainty_pct": 22,
        "scope": "Ammonia → Urea synthesis",
        "notes": "Includes CO2 from ammonia feedstock",
    },
    "fertilizers_nitric_acid": {
        "product_name": "Nitric Acid",
        "cbam_product_group": "fertilizers",
        "cn_codes": ["28080000"],
        "default_direct_tco2_per_ton": 0.900,
        "default_indirect_tco2_per_ton": 0.150,
        "default_total_tco2_per_ton": 1.050,
        "source": "IPCC 2019 + European Fertilizer Manufacturers Association",
        "source_url": "https://www.fertilizerseurope.com/",
        "vintage": 2020,
        "uncertainty_pct": 20,
        "scope": "Ammonia oxidation to nitric acid",
        "notes": "Includes N2O emissions (high GWP)",
    },
}


# ============================================================================
# HYDROGEN (if time permits - less urgent for MVP)
# ============================================================================

HYDROGEN_FACTORS = {
    "hydrogen_grey": {
        "product_name": "Hydrogen (Grey - from natural gas)",
        "cbam_product_group": "hydrogen",
        "cn_codes": ["28041000"],
        "default_direct_tco2_per_ton": 10.000,
        "default_indirect_tco2_per_ton": 1.000,
        "default_total_tco2_per_ton": 11.000,
        "source": "IEA - Global Hydrogen Review 2023",
        "source_url": "https://www.iea.org/reports/global-hydrogen-review-2023",
        "vintage": 2023,
        "uncertainty_pct": 30,
        "scope": "Steam methane reforming (SMR) without CCS",
        "notes": "Highest carbon intensity hydrogen production method",
    },
}


# ============================================================================
# CONSOLIDATED DATABASE
# ============================================================================

EMISSION_FACTORS_DB: Dict[str, Dict[str, Any]] = {
    **CEMENT_FACTORS,
    **STEEL_FACTORS,
    **ALUMINUM_FACTORS,
    **FERTILIZER_FACTORS,
    **HYDROGEN_FACTORS,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_emission_factor_by_product_key(product_key: str) -> Dict[str, Any]:
    """
    Retrieve emission factor data by product key.

    Args:
        product_key: Key from EMISSION_FACTORS_DB

    Returns:
        Dict with emission factor data

    Raises:
        KeyError: If product_key not found
    """
    if product_key not in EMISSION_FACTORS_DB:
        raise KeyError(f"Product key '{product_key}' not found in emission factors database")

    return EMISSION_FACTORS_DB[product_key]


def get_emission_factor_by_cn_code(cn_code: str) -> List[Dict[str, Any]]:
    """
    Retrieve emission factors by CN code.

    Args:
        cn_code: 8-digit CN (Combined Nomenclature) code

    Returns:
        List of matching emission factor records (may have multiple if CN code maps to different products)
    """
    matches = []
    for product_key, data in EMISSION_FACTORS_DB.items():
        if cn_code in data.get("cn_codes", []):
            matches.append({
                "product_key": product_key,
                **data
            })

    return matches


def get_all_product_groups() -> List[str]:
    """Get list of all CBAM product groups covered."""
    product_groups = set()
    for data in EMISSION_FACTORS_DB.values():
        product_groups.add(data["cbam_product_group"])

    return sorted(list(product_groups))


def get_all_cn_codes() -> List[str]:
    """Get list of all CN codes in the database."""
    cn_codes = set()
    for data in EMISSION_FACTORS_DB.values():
        cn_codes.update(data.get("cn_codes", []))

    return sorted(list(cn_codes))


def generate_summary_statistics() -> Dict[str, Any]:
    """Generate summary statistics about the emission factors database."""
    return {
        "version": VERSION,
        "last_updated": LAST_UPDATED,
        "total_products": len(EMISSION_FACTORS_DB),
        "product_groups": get_all_product_groups(),
        "total_cn_codes": len(get_all_cn_codes()),
        "vintage_range": {
            "min": min(d["vintage"] for d in EMISSION_FACTORS_DB.values()),
            "max": max(d["vintage"] for d in EMISSION_FACTORS_DB.values()),
        },
        "disclaimer": DISCLAIMER,
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # Print summary when run as script
    import json

    summary = generate_summary_statistics()
    print("=" * 80)
    print("CBAM EMISSION FACTORS DATABASE - DEMO MODE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 80)
    print(f"⚠️  {DISCLAIMER}")
    print("=" * 80)

    print("\n\nSample Products:")
    print("-" * 80)
    for i, (key, data) in enumerate(list(EMISSION_FACTORS_DB.items())[:5]):
        print(f"\n{i+1}. {data['product_name']}")
        print(f"   CN Codes: {', '.join(data['cn_codes'])}")
        print(f"   Total Emissions: {data['default_total_tco2_per_ton']:.3f} tCO2/ton")
        print(f"   Source: {data['source']}")
