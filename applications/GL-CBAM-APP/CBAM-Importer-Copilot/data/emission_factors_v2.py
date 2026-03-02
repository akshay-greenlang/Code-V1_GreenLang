# -*- coding: utf-8 -*-
"""
CBAM Emission Factors Database v2.0 - Regional and Product-Specific Factors

Expanded emission factors database with 30+ product variants and regional
emission factors for 10+ major producing regions. Includes JRC published
default values, year-based markup schedules, and electricity grid EFs.

Regulatory references:
    - EU Commission Implementing Regulation (EU) 2023/1773 (Default Values)
    - JRC Technical Reports on CBAM Default Values (2025)
    - IEA World Energy Outlook 2024 (Grid Emission Factors)
    - IPCC 2006/2019 Guidelines (Process Emission Factors)
    - Omnibus Simplification Package COM(2025) 508 (Markup Schedule)

All values are in tCO2e per tonne of product unless otherwise noted.
Electricity grid factors are in tCO2e/MWh.

IMPORTANT: For actual CBAM filings, always use the latest official EU
Commission default values published in the Official Journal.

Version: 2.0.0
Author: GreenLang CBAM Team
License: Proprietary
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

VERSION = "2.0.0"
LAST_UPDATED = "2026-02-28"
DISCLAIMER = (
    "Emission factors from public sources (IEA, IPCC, JRC, World Steel, IAI). "
    "For actual CBAM filings, use official EU Commission default values."
)


# ============================================================================
# CEMENT EMISSION FACTORS
# ============================================================================

CEMENT_FACTORS: Dict[str, Dict[str, Any]] = {
    "cement_clinker": {
        "product_code": "CBAM-CEM-001",
        "product_name": "Cement Clinker",
        "cbam_sector": "cement",
        "cn_codes": ["25231000"],
        "world_average_ef": Decimal("0.950"),
        "regional_ef": {
            "EU": Decimal("0.850"),
            "China": Decimal("0.900"),
            "India": Decimal("1.050"),
            "Russia": Decimal("0.950"),
            "Turkey": Decimal("0.920"),
            "South_Korea": Decimal("0.880"),
            "USA": Decimal("0.910"),
            "Brazil": Decimal("0.850"),
            "Japan": Decimal("0.870"),
            "Middle_East": Decimal("0.980"),
        },
        "jrc_default_value": Decimal("0.950"),
        "default_with_markup_2026": Decimal("1.045"),
        "default_with_markup_2027": Decimal("1.140"),
        "default_with_markup_2028": Decimal("1.235"),
        "source": "IPCC 2006 Vol 3 Ch 2; JRC 2025 CBAM Default Values",
        "vintage": 2025,
        "uncertainty_pct": 12,
    },
    "cement_portland_grey": {
        "product_code": "CBAM-CEM-002",
        "product_name": "Portland Cement (Grey)",
        "cbam_sector": "cement",
        "cn_codes": ["25232900"],
        "world_average_ef": Decimal("0.900"),
        "regional_ef": {
            "EU": Decimal("0.766"),
            "China": Decimal("0.850"),
            "India": Decimal("0.980"),
            "Russia": Decimal("0.900"),
            "Turkey": Decimal("0.870"),
            "South_Korea": Decimal("0.830"),
            "USA": Decimal("0.860"),
            "Brazil": Decimal("0.680"),
            "Japan": Decimal("0.810"),
            "Middle_East": Decimal("0.920"),
        },
        "jrc_default_value": Decimal("0.900"),
        "default_with_markup_2026": Decimal("0.990"),
        "default_with_markup_2027": Decimal("1.080"),
        "default_with_markup_2028": Decimal("1.170"),
        "source": "IEA Cement Roadmap 2018; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 15,
    },
    "cement_portland_white": {
        "product_code": "CBAM-CEM-003",
        "product_name": "Portland Cement (White)",
        "cbam_sector": "cement",
        "cn_codes": ["25232100"],
        "world_average_ef": Decimal("1.005"),
        "regional_ef": {
            "EU": Decimal("0.920"),
            "China": Decimal("0.970"),
            "India": Decimal("1.100"),
            "Turkey": Decimal("0.980"),
            "USA": Decimal("0.960"),
        },
        "jrc_default_value": Decimal("1.005"),
        "default_with_markup_2026": Decimal("1.106"),
        "default_with_markup_2027": Decimal("1.206"),
        "default_with_markup_2028": Decimal("1.307"),
        "source": "IEA + industry corrections; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 18,
    },
    "cement_aluminous": {
        "product_code": "CBAM-CEM-004",
        "product_name": "Aluminous Cement",
        "cbam_sector": "cement",
        "cn_codes": ["25233000"],
        "world_average_ef": Decimal("0.850"),
        "regional_ef": {
            "EU": Decimal("0.800"),
            "China": Decimal("0.830"),
            "Turkey": Decimal("0.860"),
        },
        "jrc_default_value": Decimal("0.850"),
        "default_with_markup_2026": Decimal("0.935"),
        "default_with_markup_2027": Decimal("1.020"),
        "default_with_markup_2028": Decimal("1.105"),
        "source": "Industry estimates; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 20,
    },
    "cement_other_hydraulic": {
        "product_code": "CBAM-CEM-005",
        "product_name": "Other Hydraulic Cements",
        "cbam_sector": "cement",
        "cn_codes": ["25239000"],
        "world_average_ef": Decimal("0.780"),
        "regional_ef": {
            "EU": Decimal("0.650"),
            "China": Decimal("0.750"),
            "India": Decimal("0.800"),
        },
        "jrc_default_value": Decimal("0.780"),
        "default_with_markup_2026": Decimal("0.858"),
        "default_with_markup_2027": Decimal("0.936"),
        "default_with_markup_2028": Decimal("1.014"),
        "source": "IPCC + blended cement ratios; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
}


# ============================================================================
# IRON & STEEL EMISSION FACTORS
# ============================================================================

STEEL_FACTORS: Dict[str, Dict[str, Any]] = {
    "iron_pig_iron": {
        "product_code": "CBAM-STL-001",
        "product_name": "Pig Iron",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72011000", "72012000", "72015000"],
        "world_average_ef": Decimal("1.900"),
        "regional_ef": {
            "EU": Decimal("1.700"),
            "China": Decimal("2.100"),
            "India": Decimal("2.500"),
            "Russia": Decimal("1.900"),
            "Turkey": Decimal("1.800"),
            "South_Korea": Decimal("1.750"),
            "USA": Decimal("1.800"),
            "Brazil": Decimal("1.600"),
            "Japan": Decimal("1.700"),
            "Middle_East": Decimal("2.000"),
        },
        "jrc_default_value": Decimal("1.900"),
        "default_with_markup_2026": Decimal("2.090"),
        "default_with_markup_2027": Decimal("2.280"),
        "default_with_markup_2028": Decimal("2.470"),
        "source": "World Steel Association 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 18,
    },
    "steel_bof_crude": {
        "product_code": "CBAM-STL-002",
        "product_name": "Crude Steel (BOF Route)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72061000"],
        "world_average_ef": Decimal("2.000"),
        "regional_ef": {
            "EU": Decimal("1.800"),
            "China": Decimal("2.200"),
            "India": Decimal("2.600"),
            "Russia": Decimal("2.000"),
            "Turkey": Decimal("1.900"),
            "South_Korea": Decimal("1.850"),
            "USA": Decimal("1.850"),
            "Brazil": Decimal("1.700"),
            "Japan": Decimal("1.800"),
            "Middle_East": Decimal("2.100"),
        },
        "jrc_default_value": Decimal("2.000"),
        "default_with_markup_2026": Decimal("2.200"),
        "default_with_markup_2027": Decimal("2.400"),
        "default_with_markup_2028": Decimal("2.600"),
        "source": "World Steel Association 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 20,
    },
    "steel_eaf_crude": {
        "product_code": "CBAM-STL-003",
        "product_name": "Crude Steel (EAF Route)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72061000"],
        "world_average_ef": Decimal("0.800"),
        "regional_ef": {
            "EU": Decimal("0.500"),
            "China": Decimal("0.900"),
            "India": Decimal("1.100"),
            "Russia": Decimal("0.800"),
            "Turkey": Decimal("0.650"),
            "South_Korea": Decimal("0.600"),
            "USA": Decimal("0.550"),
            "Brazil": Decimal("0.400"),
            "Japan": Decimal("0.550"),
            "Middle_East": Decimal("0.750"),
        },
        "jrc_default_value": Decimal("0.800"),
        "default_with_markup_2026": Decimal("0.880"),
        "default_with_markup_2027": Decimal("0.960"),
        "default_with_markup_2028": Decimal("1.040"),
        "source": "World Steel Association 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 25,
    },
    "steel_hot_rolled_flat": {
        "product_code": "CBAM-STL-004",
        "product_name": "Hot-Rolled Flat Products",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72081000", "72082500"],
        "world_average_ef": Decimal("2.150"),
        "regional_ef": {
            "EU": Decimal("1.950"),
            "China": Decimal("2.350"),
            "India": Decimal("2.750"),
            "Russia": Decimal("2.150"),
            "Turkey": Decimal("2.050"),
            "South_Korea": Decimal("2.000"),
            "USA": Decimal("2.000"),
            "Japan": Decimal("1.950"),
        },
        "jrc_default_value": Decimal("2.150"),
        "default_with_markup_2026": Decimal("2.365"),
        "default_with_markup_2027": Decimal("2.580"),
        "default_with_markup_2028": Decimal("2.795"),
        "source": "World Steel + rolling energy; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
    "steel_cold_rolled_flat": {
        "product_code": "CBAM-STL-005",
        "product_name": "Cold-Rolled Flat Products",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72091500"],
        "world_average_ef": Decimal("2.350"),
        "regional_ef": {
            "EU": Decimal("2.100"),
            "China": Decimal("2.550"),
            "India": Decimal("2.950"),
            "South_Korea": Decimal("2.200"),
            "Japan": Decimal("2.150"),
        },
        "jrc_default_value": Decimal("2.350"),
        "default_with_markup_2026": Decimal("2.585"),
        "default_with_markup_2027": Decimal("2.820"),
        "default_with_markup_2028": Decimal("3.055"),
        "source": "World Steel + cold rolling energy; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
    "steel_long_products": {
        "product_code": "CBAM-STL-006",
        "product_name": "Long Products (Bars, Rods, Sections)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72131000", "72142000", "72163100"],
        "world_average_ef": Decimal("2.100"),
        "regional_ef": {
            "EU": Decimal("1.800"),
            "China": Decimal("2.300"),
            "India": Decimal("2.700"),
            "Turkey": Decimal("1.900"),
            "USA": Decimal("1.900"),
        },
        "jrc_default_value": Decimal("2.100"),
        "default_with_markup_2026": Decimal("2.310"),
        "default_with_markup_2027": Decimal("2.520"),
        "default_with_markup_2028": Decimal("2.730"),
        "source": "World Steel Association; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
    "steel_wire": {
        "product_code": "CBAM-STL-007",
        "product_name": "Steel Wire",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72171000"],
        "world_average_ef": Decimal("2.300"),
        "regional_ef": {
            "EU": Decimal("2.000"),
            "China": Decimal("2.500"),
            "India": Decimal("2.800"),
        },
        "jrc_default_value": Decimal("2.300"),
        "default_with_markup_2026": Decimal("2.530"),
        "default_with_markup_2027": Decimal("2.760"),
        "default_with_markup_2028": Decimal("2.990"),
        "source": "World Steel + drawing; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 24,
    },
    "steel_stainless": {
        "product_code": "CBAM-STL-008",
        "product_name": "Stainless Steel Flat Products",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72193100"],
        "world_average_ef": Decimal("5.500"),
        "regional_ef": {
            "EU": Decimal("4.500"),
            "China": Decimal("6.500"),
            "India": Decimal("7.500"),
            "South_Korea": Decimal("5.000"),
            "Japan": Decimal("4.800"),
        },
        "jrc_default_value": Decimal("5.500"),
        "default_with_markup_2026": Decimal("6.050"),
        "default_with_markup_2027": Decimal("6.600"),
        "default_with_markup_2028": Decimal("7.150"),
        "source": "ISSF 2023 + alloy contribution; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 28,
    },
    "steel_seamless_tubes": {
        "product_code": "CBAM-STL-009",
        "product_name": "Seamless Tubes and Pipes",
        "cbam_sector": "iron_steel",
        "cn_codes": ["73041900"],
        "world_average_ef": Decimal("2.500"),
        "regional_ef": {
            "EU": Decimal("2.200"),
            "China": Decimal("2.700"),
            "Russia": Decimal("2.500"),
            "Japan": Decimal("2.300"),
        },
        "jrc_default_value": Decimal("2.500"),
        "default_with_markup_2026": Decimal("2.750"),
        "default_with_markup_2027": Decimal("3.000"),
        "default_with_markup_2028": Decimal("3.250"),
        "source": "World Steel + tube processing; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 24,
    },
    "ferro_manganese": {
        "product_code": "CBAM-STL-010",
        "product_name": "Ferro-Manganese (HC)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72021100"],
        "world_average_ef": Decimal("3.500"),
        "regional_ef": {
            "EU": Decimal("2.800"),
            "China": Decimal("3.800"),
            "India": Decimal("4.200"),
            "South_Africa": Decimal("3.600"),
        },
        "jrc_default_value": Decimal("3.500"),
        "default_with_markup_2026": Decimal("3.850"),
        "default_with_markup_2027": Decimal("4.200"),
        "default_with_markup_2028": Decimal("4.550"),
        "source": "IPCC 2006 Vol 3; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 25,
    },
    "ferro_silicon": {
        "product_code": "CBAM-STL-011",
        "product_name": "Ferro-Silicon (>55% Si)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72022100"],
        "world_average_ef": Decimal("5.000"),
        "regional_ef": {
            "EU": Decimal("3.500"),
            "China": Decimal("6.000"),
            "India": Decimal("6.500"),
            "Russia": Decimal("4.500"),
            "Norway": Decimal("2.000"),
        },
        "jrc_default_value": Decimal("5.000"),
        "default_with_markup_2026": Decimal("5.500"),
        "default_with_markup_2027": Decimal("6.000"),
        "default_with_markup_2028": Decimal("6.500"),
        "source": "IPCC + electricity intensity; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 30,
    },
    "ferro_chrome": {
        "product_code": "CBAM-STL-012",
        "product_name": "Ferro-Chromium (HC)",
        "cbam_sector": "iron_steel",
        "cn_codes": ["72024100"],
        "world_average_ef": Decimal("4.200"),
        "regional_ef": {
            "EU": Decimal("3.200"),
            "South_Africa": Decimal("5.500"),
            "India": Decimal("5.800"),
            "Kazakhstan": Decimal("4.500"),
            "Turkey": Decimal("3.800"),
        },
        "jrc_default_value": Decimal("4.200"),
        "default_with_markup_2026": Decimal("4.620"),
        "default_with_markup_2027": Decimal("5.040"),
        "default_with_markup_2028": Decimal("5.460"),
        "source": "ICDA 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 28,
    },
}


# ============================================================================
# ALUMINIUM EMISSION FACTORS
# ============================================================================

ALUMINIUM_FACTORS: Dict[str, Dict[str, Any]] = {
    "aluminium_primary_unwrought": {
        "product_code": "CBAM-ALU-001",
        "product_name": "Primary Aluminium, Unwrought",
        "cbam_sector": "aluminium",
        "cn_codes": ["76011000"],
        "world_average_ef": Decimal("11.500"),
        "regional_ef": {
            "EU": Decimal("6.800"),
            "China": Decimal("16.500"),
            "India": Decimal("18.000"),
            "Russia": Decimal("5.500"),
            "Middle_East": Decimal("10.500"),
            "South_Korea": Decimal("12.000"),
            "USA": Decimal("8.500"),
            "Brazil": Decimal("4.500"),
            "Norway": Decimal("2.500"),
            "Iceland": Decimal("2.000"),
            "Canada": Decimal("5.000"),
            "Australia": Decimal("14.000"),
        },
        "jrc_default_value": Decimal("11.500"),
        "default_with_markup_2026": Decimal("12.650"),
        "default_with_markup_2027": Decimal("13.800"),
        "default_with_markup_2028": Decimal("14.950"),
        "source": "IAI GHG Protocol 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 35,
    },
    "aluminium_alloys_unwrought": {
        "product_code": "CBAM-ALU-002",
        "product_name": "Aluminium Alloys, Unwrought",
        "cbam_sector": "aluminium",
        "cn_codes": ["76012000"],
        "world_average_ef": Decimal("11.100"),
        "regional_ef": {
            "EU": Decimal("6.500"),
            "China": Decimal("16.000"),
            "India": Decimal("17.500"),
            "Russia": Decimal("5.200"),
            "Middle_East": Decimal("10.200"),
        },
        "jrc_default_value": Decimal("11.100"),
        "default_with_markup_2026": Decimal("12.210"),
        "default_with_markup_2027": Decimal("13.320"),
        "default_with_markup_2028": Decimal("14.430"),
        "source": "IAI GHG Protocol 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 35,
    },
    "aluminium_secondary": {
        "product_code": "CBAM-ALU-003",
        "product_name": "Secondary Aluminium (Recycled)",
        "cbam_sector": "aluminium",
        "cn_codes": ["76020000"],
        "world_average_ef": Decimal("0.600"),
        "regional_ef": {
            "EU": Decimal("0.400"),
            "China": Decimal("0.700"),
            "India": Decimal("0.800"),
            "USA": Decimal("0.450"),
        },
        "jrc_default_value": Decimal("0.600"),
        "default_with_markup_2026": Decimal("0.660"),
        "default_with_markup_2027": Decimal("0.720"),
        "default_with_markup_2028": Decimal("0.780"),
        "source": "IAI 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 30,
    },
    "aluminium_extrusions": {
        "product_code": "CBAM-ALU-004",
        "product_name": "Aluminium Extrusions (Profiles)",
        "cbam_sector": "aluminium",
        "cn_codes": ["76041000", "76042100"],
        "world_average_ef": Decimal("12.000"),
        "regional_ef": {
            "EU": Decimal("7.500"),
            "China": Decimal("17.000"),
            "India": Decimal("18.500"),
            "Turkey": Decimal("11.000"),
        },
        "jrc_default_value": Decimal("12.000"),
        "default_with_markup_2026": Decimal("13.200"),
        "default_with_markup_2027": Decimal("14.400"),
        "default_with_markup_2028": Decimal("15.600"),
        "source": "IAI + extrusion energy; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 32,
    },
    "aluminium_flat_rolled": {
        "product_code": "CBAM-ALU-005",
        "product_name": "Aluminium Plates, Sheets, Strip",
        "cbam_sector": "aluminium",
        "cn_codes": ["76061100"],
        "world_average_ef": Decimal("12.500"),
        "regional_ef": {
            "EU": Decimal("7.800"),
            "China": Decimal("17.500"),
            "India": Decimal("19.000"),
        },
        "jrc_default_value": Decimal("12.500"),
        "default_with_markup_2026": Decimal("13.750"),
        "default_with_markup_2027": Decimal("15.000"),
        "default_with_markup_2028": Decimal("16.250"),
        "source": "IAI + rolling energy; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 33,
    },
    "aluminium_foil": {
        "product_code": "CBAM-ALU-006",
        "product_name": "Aluminium Foil",
        "cbam_sector": "aluminium",
        "cn_codes": ["76071100"],
        "world_average_ef": Decimal("13.000"),
        "regional_ef": {
            "EU": Decimal("8.200"),
            "China": Decimal("18.000"),
        },
        "jrc_default_value": Decimal("13.000"),
        "default_with_markup_2026": Decimal("14.300"),
        "default_with_markup_2027": Decimal("15.600"),
        "default_with_markup_2028": Decimal("16.900"),
        "source": "IAI + thin rolling; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 35,
    },
}


# ============================================================================
# FERTILISER EMISSION FACTORS
# ============================================================================

FERTILISER_FACTORS: Dict[str, Dict[str, Any]] = {
    "ammonia_anhydrous": {
        "product_code": "CBAM-FER-001",
        "product_name": "Ammonia (Anhydrous)",
        "cbam_sector": "fertilisers",
        "cn_codes": ["28141000"],
        "world_average_ef": Decimal("2.500"),
        "regional_ef": {
            "EU": Decimal("1.900"),
            "China": Decimal("3.800"),
            "India": Decimal("2.800"),
            "Russia": Decimal("2.300"),
            "Turkey": Decimal("2.600"),
            "USA": Decimal("1.800"),
            "Middle_East": Decimal("1.600"),
            "Brazil": Decimal("2.200"),
            "Japan": Decimal("2.100"),
            "South_Korea": Decimal("2.400"),
        },
        "jrc_default_value": Decimal("2.500"),
        "default_with_markup_2026": Decimal("2.750"),
        "default_with_markup_2027": Decimal("3.000"),
        "default_with_markup_2028": Decimal("3.250"),
        "source": "IPCC 2019 Vol 3 Ch 3; IFA 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 25,
    },
    "nitric_acid": {
        "product_code": "CBAM-FER-002",
        "product_name": "Nitric Acid",
        "cbam_sector": "fertilisers",
        "cn_codes": ["28080000"],
        "world_average_ef": Decimal("1.050"),
        "regional_ef": {
            "EU": Decimal("0.600"),
            "China": Decimal("1.500"),
            "India": Decimal("1.300"),
            "Russia": Decimal("1.200"),
            "USA": Decimal("0.700"),
        },
        "jrc_default_value": Decimal("1.050"),
        "default_with_markup_2026": Decimal("1.155"),
        "default_with_markup_2027": Decimal("1.260"),
        "default_with_markup_2028": Decimal("1.365"),
        "source": "IPCC 2019 + N2O abatement; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 20,
    },
    "urea": {
        "product_code": "CBAM-FER-003",
        "product_name": "Urea",
        "cbam_sector": "fertilisers",
        "cn_codes": ["31021000"],
        "world_average_ef": Decimal("1.700"),
        "regional_ef": {
            "EU": Decimal("1.300"),
            "China": Decimal("3.200"),
            "India": Decimal("2.000"),
            "Russia": Decimal("1.600"),
            "Middle_East": Decimal("1.200"),
            "USA": Decimal("1.200"),
        },
        "jrc_default_value": Decimal("1.700"),
        "default_with_markup_2026": Decimal("1.870"),
        "default_with_markup_2027": Decimal("2.040"),
        "default_with_markup_2028": Decimal("2.210"),
        "source": "IPCC + FAO; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
    "ammonium_nitrate": {
        "product_code": "CBAM-FER-004",
        "product_name": "Ammonium Nitrate",
        "cbam_sector": "fertilisers",
        "cn_codes": ["31023000"],
        "world_average_ef": Decimal("2.800"),
        "regional_ef": {
            "EU": Decimal("2.000"),
            "China": Decimal("3.500"),
            "India": Decimal("3.200"),
            "Russia": Decimal("2.600"),
        },
        "jrc_default_value": Decimal("2.800"),
        "default_with_markup_2026": Decimal("3.080"),
        "default_with_markup_2027": Decimal("3.360"),
        "default_with_markup_2028": Decimal("3.640"),
        "source": "IPCC + industry; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 22,
    },
    "npk_compound": {
        "product_code": "CBAM-FER-005",
        "product_name": "NPK Compound Fertiliser",
        "cbam_sector": "fertilisers",
        "cn_codes": ["31052000"],
        "world_average_ef": Decimal("1.600"),
        "regional_ef": {
            "EU": Decimal("1.200"),
            "China": Decimal("2.200"),
            "India": Decimal("1.800"),
            "Russia": Decimal("1.500"),
        },
        "jrc_default_value": Decimal("1.600"),
        "default_with_markup_2026": Decimal("1.760"),
        "default_with_markup_2027": Decimal("1.920"),
        "default_with_markup_2028": Decimal("2.080"),
        "source": "IFA + weighted N content; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 25,
    },
    "dap": {
        "product_code": "CBAM-FER-006",
        "product_name": "Di-Ammonium Phosphate (DAP)",
        "cbam_sector": "fertilisers",
        "cn_codes": ["31053000"],
        "world_average_ef": Decimal("1.500"),
        "regional_ef": {
            "EU": Decimal("1.100"),
            "China": Decimal("2.100"),
            "India": Decimal("1.700"),
            "USA": Decimal("1.200"),
            "Morocco": Decimal("1.300"),
        },
        "jrc_default_value": Decimal("1.500"),
        "default_with_markup_2026": Decimal("1.650"),
        "default_with_markup_2027": Decimal("1.800"),
        "default_with_markup_2028": Decimal("1.950"),
        "source": "IFA 2023; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 23,
    },
}


# ============================================================================
# HYDROGEN EMISSION FACTORS
# ============================================================================

HYDROGEN_FACTORS: Dict[str, Dict[str, Any]] = {
    "hydrogen_grey": {
        "product_code": "CBAM-HYD-001",
        "product_name": "Grey Hydrogen (SMR)",
        "cbam_sector": "hydrogen",
        "cn_codes": ["28041000"],
        "world_average_ef": Decimal("11.000"),
        "regional_ef": {
            "EU": Decimal("9.500"),
            "China": Decimal("13.000"),
            "India": Decimal("12.000"),
            "Russia": Decimal("10.500"),
            "USA": Decimal("9.000"),
            "Middle_East": Decimal("9.500"),
        },
        "jrc_default_value": Decimal("11.000"),
        "default_with_markup_2026": Decimal("12.100"),
        "default_with_markup_2027": Decimal("13.200"),
        "default_with_markup_2028": Decimal("14.300"),
        "source": "IEA Global Hydrogen Review 2024; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 30,
    },
    "hydrogen_blue": {
        "product_code": "CBAM-HYD-002",
        "product_name": "Blue Hydrogen (SMR + CCS)",
        "cbam_sector": "hydrogen",
        "cn_codes": ["28042100"],
        "world_average_ef": Decimal("3.000"),
        "regional_ef": {
            "EU": Decimal("2.500"),
            "USA": Decimal("2.200"),
            "Middle_East": Decimal("2.800"),
            "Norway": Decimal("2.000"),
        },
        "jrc_default_value": Decimal("3.000"),
        "default_with_markup_2026": Decimal("3.300"),
        "default_with_markup_2027": Decimal("3.600"),
        "default_with_markup_2028": Decimal("3.900"),
        "source": "IEA GHR 2024; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 35,
    },
    "hydrogen_green": {
        "product_code": "CBAM-HYD-003",
        "product_name": "Green Hydrogen (Electrolysis)",
        "cbam_sector": "hydrogen",
        "cn_codes": ["28042900"],
        "world_average_ef": Decimal("0.500"),
        "regional_ef": {
            "EU": Decimal("0.300"),
            "USA": Decimal("0.200"),
            "Middle_East": Decimal("0.400"),
            "Australia": Decimal("0.350"),
            "Chile": Decimal("0.200"),
        },
        "jrc_default_value": Decimal("0.500"),
        "default_with_markup_2026": Decimal("0.550"),
        "default_with_markup_2027": Decimal("0.600"),
        "default_with_markup_2028": Decimal("0.650"),
        "source": "IEA GHR 2024; JRC 2025",
        "vintage": 2025,
        "uncertainty_pct": 40,
    },
}


# ============================================================================
# ELECTRICITY GRID EMISSION FACTORS (tCO2e/MWh)
# ============================================================================

ELECTRICITY_GRID_FACTORS: Dict[str, Dict[str, Any]] = {
    "EU_average": {"country": "EU-27", "grid_ef_tco2e_per_mwh": Decimal("0.230"), "year": 2024, "source": "EEA"},
    "DE": {"country": "Germany", "grid_ef_tco2e_per_mwh": Decimal("0.350"), "year": 2024, "source": "UBA"},
    "FR": {"country": "France", "grid_ef_tco2e_per_mwh": Decimal("0.052"), "year": 2024, "source": "RTE/ADEME"},
    "PL": {"country": "Poland", "grid_ef_tco2e_per_mwh": Decimal("0.670"), "year": 2024, "source": "KOBiZE"},
    "CN": {"country": "China", "grid_ef_tco2e_per_mwh": Decimal("0.581"), "year": 2024, "source": "MEE"},
    "IN": {"country": "India", "grid_ef_tco2e_per_mwh": Decimal("0.708"), "year": 2024, "source": "CEA"},
    "RU": {"country": "Russia", "grid_ef_tco2e_per_mwh": Decimal("0.420"), "year": 2024, "source": "IEA"},
    "TR": {"country": "Turkey", "grid_ef_tco2e_per_mwh": Decimal("0.440"), "year": 2024, "source": "TEIAS"},
    "KR": {"country": "South Korea", "grid_ef_tco2e_per_mwh": Decimal("0.459"), "year": 2024, "source": "KEEI"},
    "US": {"country": "USA", "grid_ef_tco2e_per_mwh": Decimal("0.370"), "year": 2024, "source": "EPA eGRID"},
    "BR": {"country": "Brazil", "grid_ef_tco2e_per_mwh": Decimal("0.075"), "year": 2024, "source": "MCTIC"},
    "JP": {"country": "Japan", "grid_ef_tco2e_per_mwh": Decimal("0.470"), "year": 2024, "source": "METI"},
    "NO": {"country": "Norway", "grid_ef_tco2e_per_mwh": Decimal("0.008"), "year": 2024, "source": "NVE"},
    "IS": {"country": "Iceland", "grid_ef_tco2e_per_mwh": Decimal("0.001"), "year": 2024, "source": "Orkustofnun"},
    "AU": {"country": "Australia", "grid_ef_tco2e_per_mwh": Decimal("0.680"), "year": 2024, "source": "CER"},
    "CA": {"country": "Canada", "grid_ef_tco2e_per_mwh": Decimal("0.120"), "year": 2024, "source": "ECCC"},
    "ZA": {"country": "South Africa", "grid_ef_tco2e_per_mwh": Decimal("0.950"), "year": 2024, "source": "Eskom"},
    "SA": {"country": "Saudi Arabia", "grid_ef_tco2e_per_mwh": Decimal("0.720"), "year": 2024, "source": "IEA"},
    "AE": {"country": "UAE", "grid_ef_tco2e_per_mwh": Decimal("0.580"), "year": 2024, "source": "IEA"},
    "MX": {"country": "Mexico", "grid_ef_tco2e_per_mwh": Decimal("0.450"), "year": 2024, "source": "SEMARNAT"},
    "ID": {"country": "Indonesia", "grid_ef_tco2e_per_mwh": Decimal("0.760"), "year": 2024, "source": "PLN"},
    "VN": {"country": "Vietnam", "grid_ef_tco2e_per_mwh": Decimal("0.620"), "year": 2024, "source": "EVN"},
    "EG": {"country": "Egypt", "grid_ef_tco2e_per_mwh": Decimal("0.490"), "year": 2024, "source": "IEA"},
    "UA": {"country": "Ukraine", "grid_ef_tco2e_per_mwh": Decimal("0.380"), "year": 2024, "source": "IEA"},
}


# ============================================================================
# CONSOLIDATED DATABASE
# ============================================================================

EMISSION_FACTORS_DB_V2: Dict[str, Dict[str, Any]] = {
    **CEMENT_FACTORS,
    **STEEL_FACTORS,
    **ALUMINIUM_FACTORS,
    **FERTILISER_FACTORS,
    **HYDROGEN_FACTORS,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_emission_factor_v2(product_key: str) -> Dict[str, Any]:
    """
    Retrieve emission factor data by product key.

    Args:
        product_key: Key from EMISSION_FACTORS_DB_V2.

    Returns:
        Dict with complete emission factor data including regional EFs.

    Raises:
        KeyError: If product_key not found.
    """
    if product_key not in EMISSION_FACTORS_DB_V2:
        raise KeyError(f"Product key '{product_key}' not found in v2 database")
    return EMISSION_FACTORS_DB_V2[product_key]


def get_regional_ef(product_key: str, region: str) -> Optional[Decimal]:
    """
    Retrieve a region-specific emission factor.

    Args:
        product_key: Key from EMISSION_FACTORS_DB_V2.
        region: Region name (e.g. "China", "EU", "India").

    Returns:
        Regional EF in tCO2e/t, or None if region not available.
    """
    data = get_emission_factor_v2(product_key)
    return data.get("regional_ef", {}).get(region)


def get_default_with_markup(product_key: str, year: int) -> Decimal:
    """
    Retrieve the JRC default value with year-appropriate markup.

    Markup schedule per Omnibus Simplification:
        2026: +10%, 2027: +20%, 2028: +30%, 2029+: +30% (capped)

    Args:
        product_key: Key from EMISSION_FACTORS_DB_V2.
        year: Calendar year.

    Returns:
        Default value with markup applied.
    """
    data = get_emission_factor_v2(product_key)
    markup_key = f"default_with_markup_{year}"
    if markup_key in data:
        return data[markup_key]

    # Fallback: compute from JRC default
    jrc = data.get("jrc_default_value", data.get("world_average_ef", Decimal("0")))
    if year <= 2025:
        return jrc
    elif year == 2026:
        return (jrc * Decimal("1.10")).quantize(Decimal("0.001"))
    elif year == 2027:
        return (jrc * Decimal("1.20")).quantize(Decimal("0.001"))
    else:
        return (jrc * Decimal("1.30")).quantize(Decimal("0.001"))


def get_ef_by_cn_code_v2(cn_code: str) -> List[Dict[str, Any]]:
    """
    Retrieve all emission factors matching a CN code.

    Args:
        cn_code: 8-digit CN code (spaces/dots stripped internally).

    Returns:
        List of matching emission factor records.
    """
    normalized = "".join(c for c in cn_code if c.isdigit())
    matches: List[Dict[str, Any]] = []
    for key, data in EMISSION_FACTORS_DB_V2.items():
        if normalized in data.get("cn_codes", []):
            matches.append({"product_key": key, **data})
    return matches


def get_grid_ef(country_code: str) -> Optional[Decimal]:
    """
    Retrieve electricity grid emission factor for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 code or 'EU_average'.

    Returns:
        Grid EF in tCO2e/MWh, or None if country not found.
    """
    entry = ELECTRICITY_GRID_FACTORS.get(country_code)
    if entry:
        return entry["grid_ef_tco2e_per_mwh"]
    return None


def get_all_product_keys() -> List[str]:
    """Return all product keys in the v2 database."""
    return sorted(EMISSION_FACTORS_DB_V2.keys())


def get_all_sectors_v2() -> List[str]:
    """Return all CBAM sectors covered in the v2 database."""
    sectors = set()
    for data in EMISSION_FACTORS_DB_V2.values():
        sectors.add(data["cbam_sector"])
    return sorted(sectors)


def generate_summary_v2() -> Dict[str, Any]:
    """Generate summary statistics for the v2 database."""
    all_cn: set = set()
    for data in EMISSION_FACTORS_DB_V2.values():
        all_cn.update(data.get("cn_codes", []))

    return {
        "version": VERSION,
        "last_updated": LAST_UPDATED,
        "total_products": len(EMISSION_FACTORS_DB_V2),
        "total_cn_codes": len(all_cn),
        "sectors": get_all_sectors_v2(),
        "grid_ef_countries": len(ELECTRICITY_GRID_FACTORS),
        "disclaimer": DISCLAIMER,
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    import json

    summary = generate_summary_v2()
    print("=" * 80)
    print("CBAM EMISSION FACTORS DATABASE v2.0")
    print("=" * 80)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n{DISCLAIMER}")
