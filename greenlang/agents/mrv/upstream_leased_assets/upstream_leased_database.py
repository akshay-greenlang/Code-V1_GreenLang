# -*- coding: utf-8 -*-
"""
UpstreamLeasedDatabaseEngine - Engine 1: Upstream Leased Assets Agent (AGENT-MRV-021)

Thread-safe singleton providing emission factor lookups for all upstream leased
asset categories: buildings, vehicles, equipment, IT assets, and spend-based EEIO
factors. Supports building EUI by type and climate zone, grid and fuel emission
factors, vehicle emission factors by type/fuel/age, equipment benchmarks, IT power
ratings, EEIO factors, currency conversion, CPI deflation, refrigerant GWP,
district heating, and allocation defaults.

Zero-Hallucination Guarantees:
    - All emission factors from DEFRA 2024, EPA, IEA, IPCC AR5/AR6 official publications
    - All values stored as Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the lookup path
    - Every lookup is deterministic and reproducible

Features:
    - Building EUI (Energy Use Intensity) by building type x climate zone (40 entries)
    - Grid emission factors for 12 countries/regions
    - Fuel emission factors for 7 fuel types with WTT
    - Vehicle emission factors for 8 vehicle types by fuel and age band
    - Equipment benchmarks for 6 equipment categories
    - IT asset power ratings for 7 IT asset types with PUE
    - EEIO spend-based factors for 10 NAICS codes
    - Currency conversion (20 currencies to USD)
    - CPI deflation (2015-2025)
    - Climate zone mapping by country
    - Refrigerant GWP values (AR5 and AR6)
    - Allocation method defaults and precedence
    - District heating emission factors by region
    - Unified search across all factor tables
    - Thread-safe singleton pattern with __new__

Example:
    >>> engine = UpstreamLeasedDatabaseEngine()
    >>> factor = engine.get_building_eui("office", "temperate")
    >>> factor["eui_kwh_sqm"]
    Decimal('150.00000000')
    >>> grid = engine.get_grid_emission_factor("US")
    >>> grid["co2e_per_kwh"]
    Decimal('0.37170000')

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-021 Upstream Leased Assets (GL-MRV-S3-008)
Status: Production Ready
"""

import logging
import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# ENGINE METADATA
# =============================================================================

ENGINE_ID: str = "upstream_leased_database_engine"
ENGINE_VERSION: str = "1.0.0"

# =============================================================================
# DECIMAL PRECISION
# =============================================================================

_QUANT_8DP = Decimal("0.00000001")


# =============================================================================
# BUILDING EUI DATA (kWh/sqm/year) BY BUILDING TYPE x CLIMATE ZONE
# Source: ASHRAE 90.1-2019, CBECS 2018, IEA Building Energy Data 2024
# 8 building types x 5 climate zones = 40 entries
# gas_fraction: proportion of total energy from natural gas (remainder from grid)
# =============================================================================

BUILDING_EUI_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "office": {
        "tropical": {
            "eui_kwh_sqm": Decimal("210"),
            "gas_fraction": Decimal("0.10"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("190"),
            "gas_fraction": Decimal("0.15"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("150"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("180"),
            "gas_fraction": Decimal("0.35"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("200"),
            "gas_fraction": Decimal("0.45"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "retail": {
        "tropical": {
            "eui_kwh_sqm": Decimal("270"),
            "gas_fraction": Decimal("0.08"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("240"),
            "gas_fraction": Decimal("0.12"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("190"),
            "gas_fraction": Decimal("0.25"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("220"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("250"),
            "gas_fraction": Decimal("0.40"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "warehouse": {
        "tropical": {
            "eui_kwh_sqm": Decimal("100"),
            "gas_fraction": Decimal("0.05"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("85"),
            "gas_fraction": Decimal("0.10"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("65"),
            "gas_fraction": Decimal("0.20"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("80"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("90"),
            "gas_fraction": Decimal("0.40"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "industrial": {
        "tropical": {
            "eui_kwh_sqm": Decimal("240"),
            "gas_fraction": Decimal("0.15"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("210"),
            "gas_fraction": Decimal("0.20"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("170"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("200"),
            "gas_fraction": Decimal("0.35"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("220"),
            "gas_fraction": Decimal("0.45"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "data_center": {
        "tropical": {
            "eui_kwh_sqm": Decimal("920"),
            "gas_fraction": Decimal("0.02"),
            "source": "Uptime Institute 2024/ASHRAE TC 9.9",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("850"),
            "gas_fraction": Decimal("0.02"),
            "source": "Uptime Institute 2024/ASHRAE TC 9.9",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("800"),
            "gas_fraction": Decimal("0.03"),
            "source": "Uptime Institute 2024/ASHRAE TC 9.9",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("820"),
            "gas_fraction": Decimal("0.03"),
            "source": "Uptime Institute 2024/ASHRAE TC 9.9",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("810"),
            "gas_fraction": Decimal("0.03"),
            "source": "Uptime Institute 2024/ASHRAE TC 9.9",
        },
    },
    "hotel": {
        "tropical": {
            "eui_kwh_sqm": Decimal("310"),
            "gas_fraction": Decimal("0.12"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("280"),
            "gas_fraction": Decimal("0.18"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("220"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("250"),
            "gas_fraction": Decimal("0.35"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("270"),
            "gas_fraction": Decimal("0.42"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "healthcare": {
        "tropical": {
            "eui_kwh_sqm": Decimal("410"),
            "gas_fraction": Decimal("0.15"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("370"),
            "gas_fraction": Decimal("0.20"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("300"),
            "gas_fraction": Decimal("0.30"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("350"),
            "gas_fraction": Decimal("0.35"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("380"),
            "gas_fraction": Decimal("0.42"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
    "education": {
        "tropical": {
            "eui_kwh_sqm": Decimal("170"),
            "gas_fraction": Decimal("0.10"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "arid": {
            "eui_kwh_sqm": Decimal("150"),
            "gas_fraction": Decimal("0.15"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "temperate": {
            "eui_kwh_sqm": Decimal("120"),
            "gas_fraction": Decimal("0.28"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "continental": {
            "eui_kwh_sqm": Decimal("140"),
            "gas_fraction": Decimal("0.35"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
        "polar": {
            "eui_kwh_sqm": Decimal("155"),
            "gas_fraction": Decimal("0.42"),
            "source": "ASHRAE 90.1/CBECS 2018",
        },
    },
}


# =============================================================================
# GRID EMISSION FACTORS (kgCO2e per kWh, location-based)
# Source: IEA Emission Factors 2024, EPA eGRID 2024, DEFRA 2024
# =============================================================================

GRID_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "co2e_per_kwh": Decimal("0.37170"),
        "wtt_per_kwh": Decimal("0.04348"),
        "source": "EPA eGRID 2024",
    },
    "GB": {
        "co2e_per_kwh": Decimal("0.20707"),
        "wtt_per_kwh": Decimal("0.02324"),
        "source": "DEFRA 2024",
    },
    "DE": {
        "co2e_per_kwh": Decimal("0.33800"),
        "wtt_per_kwh": Decimal("0.04051"),
        "source": "IEA 2024",
    },
    "FR": {
        "co2e_per_kwh": Decimal("0.05100"),
        "wtt_per_kwh": Decimal("0.00816"),
        "source": "IEA 2024",
    },
    "JP": {
        "co2e_per_kwh": Decimal("0.43400"),
        "wtt_per_kwh": Decimal("0.05425"),
        "source": "IEA 2024",
    },
    "CN": {
        "co2e_per_kwh": Decimal("0.55600"),
        "wtt_per_kwh": Decimal("0.07228"),
        "source": "IEA 2024",
    },
    "IN": {
        "co2e_per_kwh": Decimal("0.70800"),
        "wtt_per_kwh": Decimal("0.09204"),
        "source": "IEA 2024",
    },
    "AU": {
        "co2e_per_kwh": Decimal("0.65600"),
        "wtt_per_kwh": Decimal("0.08528"),
        "source": "IEA 2024",
    },
    "CA": {
        "co2e_per_kwh": Decimal("0.12000"),
        "wtt_per_kwh": Decimal("0.01560"),
        "source": "IEA 2024",
    },
    "BR": {
        "co2e_per_kwh": Decimal("0.07400"),
        "wtt_per_kwh": Decimal("0.00962"),
        "source": "IEA 2024",
    },
    "SE": {
        "co2e_per_kwh": Decimal("0.00800"),
        "wtt_per_kwh": Decimal("0.00104"),
        "source": "IEA 2024",
    },
    "GLOBAL": {
        "co2e_per_kwh": Decimal("0.43600"),
        "wtt_per_kwh": Decimal("0.05668"),
        "source": "IEA 2024 World Average",
    },
}


# =============================================================================
# FUEL EMISSION FACTORS (kgCO2e per kWh and per litre)
# Source: DEFRA 2024 Conversion Factors
# =============================================================================

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "natural_gas": {
        "ef_per_kwh": Decimal("0.18316"),
        "wtt_per_kwh": Decimal("0.02510"),
        "ef_per_litre": Decimal("0.00000"),
        "wtt_per_litre": Decimal("0.00000"),
        "source": "DEFRA 2024",
    },
    "heating_oil": {
        "ef_per_kwh": Decimal("0.24674"),
        "wtt_per_kwh": Decimal("0.05756"),
        "ef_per_litre": Decimal("2.54032"),
        "wtt_per_litre": Decimal("0.59268"),
        "source": "DEFRA 2024",
    },
    "lpg": {
        "ef_per_kwh": Decimal("0.21449"),
        "wtt_per_kwh": Decimal("0.03424"),
        "ef_per_litre": Decimal("1.55363"),
        "wtt_per_litre": Decimal("0.24813"),
        "source": "DEFRA 2024",
    },
    "coal": {
        "ef_per_kwh": Decimal("0.32390"),
        "wtt_per_kwh": Decimal("0.03887"),
        "ef_per_litre": Decimal("0.00000"),
        "wtt_per_litre": Decimal("0.00000"),
        "source": "DEFRA 2024",
    },
    "wood_pellets": {
        "ef_per_kwh": Decimal("0.01553"),
        "wtt_per_kwh": Decimal("0.01243"),
        "ef_per_litre": Decimal("0.00000"),
        "wtt_per_litre": Decimal("0.00000"),
        "source": "DEFRA 2024",
    },
    "district_heating": {
        "ef_per_kwh": Decimal("0.16200"),
        "wtt_per_kwh": Decimal("0.02106"),
        "ef_per_litre": Decimal("0.00000"),
        "wtt_per_litre": Decimal("0.00000"),
        "source": "DEFRA 2024",
    },
    "district_cooling": {
        "ef_per_kwh": Decimal("0.07100"),
        "wtt_per_kwh": Decimal("0.00923"),
        "ef_per_litre": Decimal("0.00000"),
        "wtt_per_litre": Decimal("0.00000"),
        "source": "DEFRA 2024 / IEA Estimate",
    },
}


# =============================================================================
# VEHICLE EMISSION FACTORS (kgCO2e per km) BY VEHICLE TYPE AND FUEL
# Source: DEFRA 2024 Conversion Factors
# Age bands: new_0_3yr, mid_4_7yr, old_8_plus
# =============================================================================

VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "small_car_petrol": {
        "petrol": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.14130"),
                "wtt_per_km": Decimal("0.03624"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.14930"),
                "wtt_per_km": Decimal("0.03829"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.15890"),
                "wtt_per_km": Decimal("0.04075"),
                "source": "DEFRA 2024",
            },
        },
    },
    "medium_car_petrol": {
        "petrol": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.17230"),
                "wtt_per_km": Decimal("0.04419"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.18210"),
                "wtt_per_km": Decimal("0.04670"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.19390"),
                "wtt_per_km": Decimal("0.04973"),
                "source": "DEFRA 2024",
            },
        },
    },
    "large_car_petrol": {
        "petrol": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.20970"),
                "wtt_per_km": Decimal("0.05379"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.22180"),
                "wtt_per_km": Decimal("0.05689"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.23610"),
                "wtt_per_km": Decimal("0.06056"),
                "source": "DEFRA 2024",
            },
        },
    },
    "suv_petrol": {
        "petrol": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.19850"),
                "wtt_per_km": Decimal("0.05091"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.20980"),
                "wtt_per_km": Decimal("0.05381"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.22340"),
                "wtt_per_km": Decimal("0.05730"),
                "source": "DEFRA 2024",
            },
        },
    },
    "light_van_diesel": {
        "diesel": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.21210"),
                "wtt_per_km": Decimal("0.04808"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.22430"),
                "wtt_per_km": Decimal("0.05085"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.23870"),
                "wtt_per_km": Decimal("0.05411"),
                "source": "DEFRA 2024",
            },
        },
    },
    "heavy_van_diesel": {
        "diesel": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.29520"),
                "wtt_per_km": Decimal("0.06691"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.31200"),
                "wtt_per_km": Decimal("0.07072"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.33210"),
                "wtt_per_km": Decimal("0.07528"),
                "source": "DEFRA 2024",
            },
        },
    },
    "light_truck_diesel": {
        "diesel": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.29520"),
                "wtt_per_km": Decimal("0.06691"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.31200"),
                "wtt_per_km": Decimal("0.07072"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.33210"),
                "wtt_per_km": Decimal("0.07528"),
                "source": "DEFRA 2024",
            },
        },
    },
    "heavy_truck_diesel": {
        "diesel": {
            "new_0_3yr": {
                "ef_per_km": Decimal("0.55440"),
                "wtt_per_km": Decimal("0.12563"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "ef_per_km": Decimal("0.58600"),
                "wtt_per_km": Decimal("0.13279"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "ef_per_km": Decimal("0.62370"),
                "wtt_per_km": Decimal("0.14133"),
                "source": "DEFRA 2024",
            },
        },
    },
}


# =============================================================================
# EEIO SPEND-BASED FACTORS (kgCO2e per USD, base year 2021)
# Source: EPA USEEIO v2.0, NAICS codes for leasing activities
# =============================================================================

EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {
        "co2e_per_usd": Decimal("0.19"),
        "description": "Lessors of residential buildings and dwellings",
        "source": "EPA USEEIO v2.0",
    },
    "531120": {
        "co2e_per_usd": Decimal("0.22"),
        "description": "Lessors of nonresidential buildings (except miniwarehouses)",
        "source": "EPA USEEIO v2.0",
    },
    "531130": {
        "co2e_per_usd": Decimal("0.18"),
        "description": "Lessors of miniwarehouses and self-storage units",
        "source": "EPA USEEIO v2.0",
    },
    "531190": {
        "co2e_per_usd": Decimal("0.20"),
        "description": "Lessors of other real estate property",
        "source": "EPA USEEIO v2.0",
    },
    "532100": {
        "co2e_per_usd": Decimal("0.24"),
        "description": "Automotive equipment rental and leasing",
        "source": "EPA USEEIO v2.0",
    },
    "532400": {
        "co2e_per_usd": Decimal("0.28"),
        "description": "Commercial and industrial machinery rental and leasing",
        "source": "EPA USEEIO v2.0",
    },
    "532200": {
        "co2e_per_usd": Decimal("0.21"),
        "description": "Consumer goods rental",
        "source": "EPA USEEIO v2.0",
    },
    "518210": {
        "co2e_per_usd": Decimal("0.35"),
        "description": "Data processing, hosting, and related services",
        "source": "EPA USEEIO v2.0",
    },
    "541500": {
        "co2e_per_usd": Decimal("0.16"),
        "description": "Computer systems design and related services",
        "source": "EPA USEEIO v2.0",
    },
    "238000": {
        "co2e_per_usd": Decimal("0.32"),
        "description": "Specialty trade contractors",
        "source": "EPA USEEIO v2.0",
    },
}


# =============================================================================
# EQUIPMENT BENCHMARKS (Default operating parameters)
# Source: EPA AP-42, Industry Standards, Equipment Manufacturer Data
# =============================================================================

EQUIPMENT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "default_hours": Decimal("4000"),
        "load_factor": Decimal("0.65"),
        "fuel_type": "natural_gas",
        "fuel_consumption_kwh_per_hr": Decimal("50.00"),
        "source": "EPA AP-42 / Industry Standard",
    },
    "construction": {
        "default_hours": Decimal("1500"),
        "load_factor": Decimal("0.55"),
        "fuel_type": "diesel",
        "fuel_consumption_kwh_per_hr": Decimal("80.00"),
        "source": "EPA AP-42 / Industry Standard",
    },
    "generator": {
        "default_hours": Decimal("500"),
        "load_factor": Decimal("0.70"),
        "fuel_type": "diesel",
        "fuel_consumption_kwh_per_hr": Decimal("120.00"),
        "source": "EPA AP-42 / Industry Standard",
    },
    "agricultural": {
        "default_hours": Decimal("1200"),
        "load_factor": Decimal("0.60"),
        "fuel_type": "diesel",
        "fuel_consumption_kwh_per_hr": Decimal("65.00"),
        "source": "EPA AP-42 / USDA",
    },
    "mining": {
        "default_hours": Decimal("3000"),
        "load_factor": Decimal("0.70"),
        "fuel_type": "diesel",
        "fuel_consumption_kwh_per_hr": Decimal("150.00"),
        "source": "EPA AP-42 / Industry Standard",
    },
    "hvac": {
        "default_hours": Decimal("4380"),
        "load_factor": Decimal("0.50"),
        "fuel_type": "electricity",
        "fuel_consumption_kwh_per_hr": Decimal("15.00"),
        "source": "ASHRAE / Industry Standard",
    },
}


# =============================================================================
# IT ASSET POWER RATINGS (kW typical power draw, utilization, PUE)
# Source: ASHRAE TC 9.9, Uptime Institute, EPA ENERGY STAR
# =============================================================================

IT_POWER_RATINGS: Dict[str, Dict[str, Any]] = {
    "server": {
        "typical_power_kw": Decimal("0.500"),
        "utilization": Decimal("0.30"),
        "pue_default": Decimal("1.58"),
        "annual_hours": Decimal("8760"),
        "source": "Uptime Institute 2024 / ASHRAE TC 9.9",
    },
    "network_switch": {
        "typical_power_kw": Decimal("0.150"),
        "utilization": Decimal("0.90"),
        "pue_default": Decimal("1.58"),
        "annual_hours": Decimal("8760"),
        "source": "Uptime Institute 2024 / Industry Standard",
    },
    "storage": {
        "typical_power_kw": Decimal("0.200"),
        "utilization": Decimal("0.50"),
        "pue_default": Decimal("1.58"),
        "annual_hours": Decimal("8760"),
        "source": "Uptime Institute 2024 / Industry Standard",
    },
    "desktop": {
        "typical_power_kw": Decimal("0.150"),
        "utilization": Decimal("0.40"),
        "pue_default": Decimal("1.00"),
        "annual_hours": Decimal("2080"),
        "source": "EPA ENERGY STAR / Industry Standard",
    },
    "laptop": {
        "typical_power_kw": Decimal("0.045"),
        "utilization": Decimal("0.35"),
        "pue_default": Decimal("1.00"),
        "annual_hours": Decimal("2080"),
        "source": "EPA ENERGY STAR / Industry Standard",
    },
    "printer": {
        "typical_power_kw": Decimal("0.200"),
        "utilization": Decimal("0.15"),
        "pue_default": Decimal("1.00"),
        "annual_hours": Decimal("2080"),
        "source": "EPA ENERGY STAR / Industry Standard",
    },
    "copier": {
        "typical_power_kw": Decimal("0.300"),
        "utilization": Decimal("0.10"),
        "pue_default": Decimal("1.00"),
        "annual_hours": Decimal("2080"),
        "source": "EPA ENERGY STAR / Industry Standard",
    },
}


# =============================================================================
# CURRENCY CONVERSION RATES (to USD, as of Jan 2024)
# Source: World Bank / IMF Exchange Rate Data 2024
# =============================================================================

CURRENCY_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.00000000"),
    "EUR": Decimal("1.09000000"),
    "GBP": Decimal("1.27000000"),
    "JPY": Decimal("0.00670000"),
    "CNY": Decimal("0.13900000"),
    "INR": Decimal("0.01200000"),
    "AUD": Decimal("0.65000000"),
    "CAD": Decimal("0.74000000"),
    "CHF": Decimal("1.18000000"),
    "SEK": Decimal("0.09600000"),
    "NOK": Decimal("0.09500000"),
    "DKK": Decimal("0.14600000"),
    "BRL": Decimal("0.20500000"),
    "KRW": Decimal("0.00076000"),
    "MXN": Decimal("0.05800000"),
    "SGD": Decimal("0.74500000"),
    "HKD": Decimal("0.12800000"),
    "NZD": Decimal("0.61000000"),
    "ZAR": Decimal("0.05300000"),
    "AED": Decimal("0.27200000"),
}


# =============================================================================
# CPI DEFLATORS (base year 2021 = 1.0)
# Source: US BLS CPI-U, IMF World Economic Outlook
# =============================================================================

CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.83250000"),
    2016: Decimal("0.85970000"),
    2017: Decimal("0.87810000"),
    2018: Decimal("0.89970000"),
    2019: Decimal("0.91530000"),
    2020: Decimal("0.92710000"),
    2021: Decimal("1.00000000"),
    2022: Decimal("1.08000000"),
    2023: Decimal("1.11520000"),
    2024: Decimal("1.14900000"),
    2025: Decimal("1.17800000"),
}


# =============================================================================
# CLIMATE ZONE MAPPING BY COUNTRY
# Source: Koppen-Geiger Climate Classification, simplified
# =============================================================================

CLIMATE_ZONE_BY_COUNTRY: Dict[str, str] = {
    "US": "temperate",
    "GB": "temperate",
    "DE": "continental",
    "FR": "temperate",
    "JP": "temperate",
    "CN": "continental",
    "IN": "tropical",
    "AU": "arid",
    "CA": "continental",
    "BR": "tropical",
    "SE": "polar",
    "NO": "polar",
    "FI": "polar",
    "RU": "continental",
    "SA": "arid",
    "AE": "arid",
    "EG": "arid",
    "MX": "tropical",
    "ID": "tropical",
    "TH": "tropical",
    "SG": "tropical",
    "MY": "tropical",
    "NG": "tropical",
    "ZA": "temperate",
    "KR": "continental",
    "IT": "temperate",
    "ES": "temperate",
    "NL": "temperate",
    "CH": "continental",
    "AT": "continental",
    "GLOBAL": "temperate",
}


# =============================================================================
# REFRIGERANT GWP VALUES
# Source: IPCC AR5 (2014) and AR6 (2021)
# =============================================================================

REFRIGERANT_GWP: Dict[str, Dict[str, Any]] = {
    "R-22": {
        "gwp_ar6": Decimal("1960"),
        "gwp_ar5": Decimal("1810"),
        "source": "IPCC AR6 (2021)",
    },
    "R-32": {
        "gwp_ar6": Decimal("771"),
        "gwp_ar5": Decimal("675"),
        "source": "IPCC AR6 (2021)",
    },
    "R-134a": {
        "gwp_ar6": Decimal("1530"),
        "gwp_ar5": Decimal("1430"),
        "source": "IPCC AR6 (2021)",
    },
    "R-404A": {
        "gwp_ar6": Decimal("4728"),
        "gwp_ar5": Decimal("3922"),
        "source": "IPCC AR6 (2021)",
    },
    "R-407C": {
        "gwp_ar6": Decimal("1908"),
        "gwp_ar5": Decimal("1774"),
        "source": "IPCC AR6 (2021)",
    },
    "R-410A": {
        "gwp_ar6": Decimal("2256"),
        "gwp_ar5": Decimal("2088"),
        "source": "IPCC AR6 (2021)",
    },
    "R-507A": {
        "gwp_ar6": Decimal("4680"),
        "gwp_ar5": Decimal("3985"),
        "source": "IPCC AR6 (2021)",
    },
    "R-290": {
        "gwp_ar6": Decimal("0.02"),
        "gwp_ar5": Decimal("3"),
        "source": "IPCC AR6 (2021) (propane)",
    },
    "R-600a": {
        "gwp_ar6": Decimal("0.0006"),
        "gwp_ar5": Decimal("3"),
        "source": "IPCC AR6 (2021) (isobutane)",
    },
    "R-717": {
        "gwp_ar6": Decimal("0"),
        "gwp_ar5": Decimal("0"),
        "source": "IPCC AR6 (2021) (ammonia)",
    },
    "R-744": {
        "gwp_ar6": Decimal("1"),
        "gwp_ar5": Decimal("1"),
        "source": "IPCC AR6 (2021) (CO2)",
    },
    "R-1234yf": {
        "gwp_ar6": Decimal("0.501"),
        "gwp_ar5": Decimal("4"),
        "source": "IPCC AR6 (2021)",
    },
    "R-1234ze": {
        "gwp_ar6": Decimal("1.37"),
        "gwp_ar5": Decimal("6"),
        "source": "IPCC AR6 (2021)",
    },
}


# =============================================================================
# ALLOCATION METHOD DEFAULTS
# Source: GHG Protocol Scope 3 Guidance, Chapter 8
# =============================================================================

ALLOCATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "floor_area": {
        "description": "Allocate by proportion of leased floor area to total floor area",
        "precedence": Decimal("1"),
    },
    "headcount": {
        "description": "Allocate by proportion of lessee headcount to total headcount",
        "precedence": Decimal("2"),
    },
    "revenue": {
        "description": "Allocate by proportion of lessee revenue to total revenue",
        "precedence": Decimal("3"),
    },
    "equal_share": {
        "description": "Allocate equally among all tenants",
        "precedence": Decimal("4"),
    },
    "time_based": {
        "description": "Allocate by proportion of lease period to total period",
        "precedence": Decimal("5"),
    },
    "energy_metered": {
        "description": "Direct metered energy allocation (preferred where available)",
        "precedence": Decimal("0"),
    },
}


# =============================================================================
# DISTRICT HEATING EMISSION FACTORS BY REGION (kgCO2e per kWh)
# Source: DEFRA 2024, IEA District Heating Database 2024
# =============================================================================

DISTRICT_HEATING_FACTORS: Dict[str, Dict[str, Any]] = {
    "UK": {
        "ef_per_kwh": Decimal("0.16200"),
        "wtt_per_kwh": Decimal("0.02106"),
        "source": "DEFRA 2024",
    },
    "DE": {
        "ef_per_kwh": Decimal("0.18500"),
        "wtt_per_kwh": Decimal("0.02405"),
        "source": "IEA 2024",
    },
    "FR": {
        "ef_per_kwh": Decimal("0.10800"),
        "wtt_per_kwh": Decimal("0.01404"),
        "source": "IEA 2024",
    },
    "SE": {
        "ef_per_kwh": Decimal("0.04900"),
        "wtt_per_kwh": Decimal("0.00637"),
        "source": "IEA 2024",
    },
    "DK": {
        "ef_per_kwh": Decimal("0.06500"),
        "wtt_per_kwh": Decimal("0.00845"),
        "source": "IEA 2024",
    },
    "FI": {
        "ef_per_kwh": Decimal("0.10300"),
        "wtt_per_kwh": Decimal("0.01339"),
        "source": "IEA 2024",
    },
    "PL": {
        "ef_per_kwh": Decimal("0.28300"),
        "wtt_per_kwh": Decimal("0.03679"),
        "source": "IEA 2024",
    },
    "CN": {
        "ef_per_kwh": Decimal("0.31200"),
        "wtt_per_kwh": Decimal("0.04056"),
        "source": "IEA 2024",
    },
    "RU": {
        "ef_per_kwh": Decimal("0.25600"),
        "wtt_per_kwh": Decimal("0.03328"),
        "source": "IEA 2024",
    },
    "US": {
        "ef_per_kwh": Decimal("0.19800"),
        "wtt_per_kwh": Decimal("0.02574"),
        "source": "EPA 2024 / EIA",
    },
    "GLOBAL": {
        "ef_per_kwh": Decimal("0.16200"),
        "wtt_per_kwh": Decimal("0.02106"),
        "source": "IEA 2024 World Average",
    },
}


# =============================================================================
# ENGINE CLASS
# =============================================================================


class UpstreamLeasedDatabaseEngine:
    """
    Thread-safe singleton engine for upstream leased assets emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all upstream
    leased asset categories: buildings (EUI by type and climate zone), vehicles
    (by type, fuel, age), equipment (benchmarks), IT assets (power ratings),
    and spend-based EEIO factors. Also provides refrigerant GWP, district
    heating factors, allocation defaults, currency conversion, and CPI deflation.

    This engine does NOT perform any LLM calls. All factors are retrieved from
    validated, frozen constant tables defined in this module.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        ENGINE_ID: Unique engine identifier string.
        ENGINE_VERSION: Semantic version of the engine.
        _lookup_count: Total number of factor lookups performed.

    Example:
        >>> engine = UpstreamLeasedDatabaseEngine()
        >>> eui = engine.get_building_eui("office", "temperate")
        >>> eui["eui_kwh_sqm"]
        Decimal('150.00000000')
        >>> grid = engine.get_grid_emission_factor("US")
        >>> grid["co2e_per_kwh"]
        Decimal('0.37170000')
    """

    ENGINE_ID: str = ENGINE_ID
    ENGINE_VERSION: str = ENGINE_VERSION

    _instance: Optional["UpstreamLeasedDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "UpstreamLeasedDatabaseEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the database engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._lookup_count: int = 0
        self._lookup_lock: threading.Lock = threading.Lock()

        logger.info(
            "UpstreamLeasedDatabaseEngine initialized: "
            "building_types=%d, climate_zones=%d, grid_countries=%d, "
            "fuel_types=%d, vehicle_types=%d, equipment_types=%d, "
            "it_asset_types=%d, eeio_codes=%d, currencies=%d, "
            "refrigerants=%d, dh_regions=%d",
            len(BUILDING_EUI_DATA),
            5,
            len(GRID_EMISSION_FACTORS),
            len(FUEL_EMISSION_FACTORS),
            len(VEHICLE_EMISSION_FACTORS),
            len(EQUIPMENT_BENCHMARKS),
            len(IT_POWER_RATINGS),
            len(EEIO_FACTORS),
            len(CURRENCY_RATES),
            len(REFRIGERANT_GWP),
            len(DISTRICT_HEATING_FACTORS),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _quantize(self, value: Decimal) -> Decimal:
        """
        Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    # =========================================================================
    # 1. BUILDING EUI
    # =========================================================================

    def get_building_eui(
        self,
        building_type: str,
        climate_zone: str,
    ) -> Dict[str, Any]:
        """
        Get building energy use intensity (EUI) by building type and climate zone.

        Looks up from BUILDING_EUI_DATA constant table. Falls back to
        office/temperate if the requested combination is not found.

        Args:
            building_type: Building type (office, retail, warehouse, industrial,
                data_center, hotel, healthcare, education).
            climate_zone: Climate zone (tropical, arid, temperate, continental,
                polar).

        Returns:
            Dict with keys: eui_kwh_sqm, gas_fraction, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> eui = engine.get_building_eui("office", "temperate")
            >>> eui["eui_kwh_sqm"]
            Decimal('150.00000000')
        """
        self._increment_lookup()

        btype = building_type.lower().strip().replace(" ", "_")
        zone = climate_zone.lower().strip()

        building_data = BUILDING_EUI_DATA.get(btype)
        if building_data is not None:
            zone_data = building_data.get(zone)
            if zone_data is not None:
                result = {
                    "eui_kwh_sqm": self._quantize(zone_data["eui_kwh_sqm"]),
                    "gas_fraction": self._quantize(zone_data["gas_fraction"]),
                    "source": zone_data["source"],
                }
                logger.debug(
                    "Building EUI lookup: type=%s, zone=%s, eui=%s",
                    btype, zone, result["eui_kwh_sqm"],
                )
                return result

        # Fallback to office / temperate
        logger.warning(
            "Building EUI not found for type=%s, zone=%s; "
            "falling back to office/temperate",
            btype, zone,
        )
        fallback = BUILDING_EUI_DATA["office"]["temperate"]
        return {
            "eui_kwh_sqm": self._quantize(fallback["eui_kwh_sqm"]),
            "gas_fraction": self._quantize(fallback["gas_fraction"]),
            "source": fallback["source"] + " (fallback)",
        }

    # =========================================================================
    # 2. GRID EMISSION FACTORS
    # =========================================================================

    def get_grid_emission_factor(
        self,
        country_code: str,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get electricity grid emission factor by country code.

        Priority: country-specific > GLOBAL average.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").
            region: Optional sub-region code (reserved for future eGRID
                support; currently ignored).

        Returns:
            Dict with keys: co2e_per_kwh, wtt_per_kwh, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> ef = engine.get_grid_emission_factor("US")
            >>> ef["co2e_per_kwh"]
            Decimal('0.37170000')
        """
        self._increment_lookup()

        code = country_code.upper().strip()

        # Priority 1: Country-level
        country_data = GRID_EMISSION_FACTORS.get(code)
        if country_data is not None:
            result = {
                "co2e_per_kwh": self._quantize(country_data["co2e_per_kwh"]),
                "wtt_per_kwh": self._quantize(country_data["wtt_per_kwh"]),
                "source": country_data["source"],
            }
            logger.debug(
                "Grid EF lookup: country=%s, co2e_per_kwh=%s",
                code, result["co2e_per_kwh"],
            )
            return result

        # Priority 2: GLOBAL fallback
        logger.warning(
            "Grid EF not found for country=%s; falling back to GLOBAL",
            code,
        )
        global_data = GRID_EMISSION_FACTORS["GLOBAL"]
        return {
            "co2e_per_kwh": self._quantize(global_data["co2e_per_kwh"]),
            "wtt_per_kwh": self._quantize(global_data["wtt_per_kwh"]),
            "source": global_data["source"] + " (fallback)",
        }

    # =========================================================================
    # 3. FUEL EMISSION FACTORS
    # =========================================================================

    def get_fuel_emission_factor(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """
        Get fuel emission factor by fuel type.

        Provides both per-kWh and per-litre factors with WTT upstream values.

        Args:
            fuel_type: Fuel type key (natural_gas, heating_oil, lpg, coal,
                wood_pellets, district_heating, district_cooling).

        Returns:
            Dict with keys: ef_per_kwh, wtt_per_kwh, ef_per_litre,
            wtt_per_litre, source. All Decimal values quantized to 8dp.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> ef = engine.get_fuel_emission_factor("natural_gas")
            >>> ef["ef_per_kwh"]
            Decimal('0.18316000')
        """
        self._increment_lookup()

        ftype = fuel_type.lower().strip().replace(" ", "_")

        fuel_data = FUEL_EMISSION_FACTORS.get(ftype)
        if fuel_data is not None:
            result = {
                "ef_per_kwh": self._quantize(fuel_data["ef_per_kwh"]),
                "wtt_per_kwh": self._quantize(fuel_data["wtt_per_kwh"]),
                "ef_per_litre": self._quantize(fuel_data["ef_per_litre"]),
                "wtt_per_litre": self._quantize(fuel_data["wtt_per_litre"]),
                "source": fuel_data["source"],
            }
            logger.debug(
                "Fuel EF lookup: type=%s, ef_per_kwh=%s",
                ftype, result["ef_per_kwh"],
            )
            return result

        # Fallback to natural_gas
        logger.warning(
            "Fuel EF not found for type=%s; falling back to natural_gas",
            ftype,
        )
        fallback = FUEL_EMISSION_FACTORS["natural_gas"]
        return {
            "ef_per_kwh": self._quantize(fallback["ef_per_kwh"]),
            "wtt_per_kwh": self._quantize(fallback["wtt_per_kwh"]),
            "ef_per_litre": self._quantize(fallback["ef_per_litre"]),
            "wtt_per_litre": self._quantize(fallback["wtt_per_litre"]),
            "source": fallback["source"] + " (fallback)",
        }

    # =========================================================================
    # 4. VEHICLE EMISSION FACTORS
    # =========================================================================

    def get_vehicle_emission_factor(
        self,
        vehicle_type: str,
        fuel_type: str,
        vehicle_age: str = "mid_4_7yr",
    ) -> Dict[str, Any]:
        """
        Get vehicle emission factor by type, fuel, and age band.

        Falls back to medium_car_petrol / petrol / mid_4_7yr if the requested
        combination is not found.

        Args:
            vehicle_type: Vehicle type key (small_car_petrol, medium_car_petrol,
                large_car_petrol, suv_petrol, light_van_diesel, heavy_van_diesel,
                light_truck_diesel, heavy_truck_diesel).
            fuel_type: Fuel type (petrol, diesel).
            vehicle_age: Age band (new_0_3yr, mid_4_7yr, old_8_plus).
                Defaults to mid_4_7yr.

        Returns:
            Dict with keys: ef_per_km, wtt_per_km, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> ef = engine.get_vehicle_emission_factor(
            ...     "medium_car_petrol", "petrol"
            ... )
            >>> ef["ef_per_km"]
            Decimal('0.18210000')
        """
        self._increment_lookup()

        vtype = vehicle_type.lower().strip()
        ftype = fuel_type.lower().strip()
        age = vehicle_age.lower().strip()

        vehicle_data = VEHICLE_EMISSION_FACTORS.get(vtype)
        if vehicle_data is not None:
            fuel_data = vehicle_data.get(ftype)
            if fuel_data is not None:
                age_data = fuel_data.get(age)
                if age_data is not None:
                    result = {
                        "ef_per_km": self._quantize(age_data["ef_per_km"]),
                        "wtt_per_km": self._quantize(age_data["wtt_per_km"]),
                        "source": age_data["source"],
                    }
                    logger.debug(
                        "Vehicle EF lookup: type=%s, fuel=%s, age=%s, "
                        "ef_per_km=%s",
                        vtype, ftype, age, result["ef_per_km"],
                    )
                    return result

        # Fallback to medium_car_petrol / petrol / mid_4_7yr
        logger.warning(
            "Vehicle EF not found for type=%s, fuel=%s, age=%s; "
            "falling back to medium_car_petrol/petrol/mid_4_7yr",
            vtype, ftype, age,
        )
        fallback = VEHICLE_EMISSION_FACTORS["medium_car_petrol"]["petrol"]["mid_4_7yr"]
        return {
            "ef_per_km": self._quantize(fallback["ef_per_km"]),
            "wtt_per_km": self._quantize(fallback["wtt_per_km"]),
            "source": fallback["source"] + " (fallback)",
        }

    # =========================================================================
    # 5. EQUIPMENT BENCHMARKS
    # =========================================================================

    def get_equipment_benchmark(
        self,
        equipment_type: str,
    ) -> Dict[str, Any]:
        """
        Get equipment operating benchmark by equipment type.

        Provides default operating hours, load factor, fuel type, and fuel
        consumption rate for calculating emissions from leased equipment.

        Args:
            equipment_type: Equipment category (manufacturing, construction,
                generator, agricultural, mining, hvac).

        Returns:
            Dict with keys: default_hours, load_factor, fuel_type,
            fuel_consumption, source. Decimal values quantized to 8dp.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> bench = engine.get_equipment_benchmark("manufacturing")
            >>> bench["default_hours"]
            Decimal('4000.00000000')
            >>> bench["load_factor"]
            Decimal('0.65000000')
        """
        self._increment_lookup()

        etype = equipment_type.lower().strip().replace(" ", "_")

        equip_data = EQUIPMENT_BENCHMARKS.get(etype)
        if equip_data is not None:
            result = {
                "default_hours": self._quantize(equip_data["default_hours"]),
                "load_factor": self._quantize(equip_data["load_factor"]),
                "fuel_type": equip_data["fuel_type"],
                "fuel_consumption": self._quantize(
                    equip_data["fuel_consumption_kwh_per_hr"]
                ),
                "source": equip_data["source"],
            }
            logger.debug(
                "Equipment benchmark lookup: type=%s, hours=%s, load=%s",
                etype, result["default_hours"], result["load_factor"],
            )
            return result

        # Fallback to manufacturing
        logger.warning(
            "Equipment benchmark not found for type=%s; "
            "falling back to manufacturing",
            etype,
        )
        fallback = EQUIPMENT_BENCHMARKS["manufacturing"]
        return {
            "default_hours": self._quantize(fallback["default_hours"]),
            "load_factor": self._quantize(fallback["load_factor"]),
            "fuel_type": fallback["fuel_type"],
            "fuel_consumption": self._quantize(
                fallback["fuel_consumption_kwh_per_hr"]
            ),
            "source": fallback["source"] + " (fallback)",
        }

    # =========================================================================
    # 6. IT ASSET POWER RATINGS
    # =========================================================================

    def get_it_power_rating(
        self,
        it_type: str,
    ) -> Dict[str, Any]:
        """
        Get IT asset power rating and utilization parameters.

        Provides typical power draw, utilization fraction, PUE default, and
        annual operating hours for calculating emissions from leased IT assets.

        Args:
            it_type: IT asset type (server, network_switch, storage, desktop,
                laptop, printer, copier).

        Returns:
            Dict with keys: typical_power_kw, utilization, pue_default,
            annual_hours, source. Decimal values quantized to 8dp.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> it = engine.get_it_power_rating("server")
            >>> it["typical_power_kw"]
            Decimal('0.50000000')
            >>> it["pue_default"]
            Decimal('1.58000000')
        """
        self._increment_lookup()

        itype = it_type.lower().strip().replace(" ", "_")

        it_data = IT_POWER_RATINGS.get(itype)
        if it_data is not None:
            result = {
                "typical_power_kw": self._quantize(it_data["typical_power_kw"]),
                "utilization": self._quantize(it_data["utilization"]),
                "pue_default": self._quantize(it_data["pue_default"]),
                "annual_hours": self._quantize(it_data["annual_hours"]),
                "source": it_data["source"],
            }
            logger.debug(
                "IT power rating lookup: type=%s, power_kw=%s, pue=%s",
                itype, result["typical_power_kw"], result["pue_default"],
            )
            return result

        # Fallback to server
        logger.warning(
            "IT power rating not found for type=%s; falling back to server",
            itype,
        )
        fallback = IT_POWER_RATINGS["server"]
        return {
            "typical_power_kw": self._quantize(fallback["typical_power_kw"]),
            "utilization": self._quantize(fallback["utilization"]),
            "pue_default": self._quantize(fallback["pue_default"]),
            "annual_hours": self._quantize(fallback["annual_hours"]),
            "source": fallback["source"] + " (fallback)",
        }

    # =========================================================================
    # 7. EEIO FACTORS
    # =========================================================================

    def get_eeio_factor(
        self,
        naics_code: str,
    ) -> Dict[str, Any]:
        """
        Get EEIO spend-based emission factor by NAICS code.

        Args:
            naics_code: NAICS industry code (e.g., "531120", "532100").

        Returns:
            Dict with keys: co2e_per_usd, description, source.
            Decimal values quantized to 8 decimal places.

        Raises:
            ValueError: If the NAICS code is not found in the EEIO table.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> ef = engine.get_eeio_factor("531120")
            >>> ef["co2e_per_usd"]
            Decimal('0.22000000')
        """
        self._increment_lookup()

        code = naics_code.strip()

        eeio_data = EEIO_FACTORS.get(code)
        if eeio_data is not None:
            result = {
                "co2e_per_usd": self._quantize(eeio_data["co2e_per_usd"]),
                "description": eeio_data["description"],
                "source": eeio_data["source"],
            }
            logger.debug(
                "EEIO factor lookup: naics=%s, co2e_per_usd=%s",
                code, result["co2e_per_usd"],
            )
            return result

        raise ValueError(
            f"EEIO factor not available for NAICS code '{code}'. "
            f"Available codes: {sorted(EEIO_FACTORS.keys())}"
        )

    # =========================================================================
    # 8. CURRENCY CONVERSION
    # =========================================================================

    def get_currency_rate(
        self,
        currency_code: str,
    ) -> Decimal:
        """
        Get the conversion rate from a currency to USD.

        Args:
            currency_code: ISO 4217 currency code (e.g., "EUR", "GBP").

        Returns:
            Decimal conversion rate to USD, quantized to 8 decimal places.

        Raises:
            ValueError: If the currency code is not found.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> engine.get_currency_rate("EUR")
            Decimal('1.09000000')
        """
        self._increment_lookup()

        code = currency_code.upper().strip()

        rate = CURRENCY_RATES.get(code)
        if rate is not None:
            return self._quantize(rate)

        raise ValueError(
            f"Currency rate not available for '{code}'. "
            f"Available currencies: {sorted(CURRENCY_RATES.keys())}"
        )

    # =========================================================================
    # 9. CPI DEFLATOR
    # =========================================================================

    def get_cpi_deflator(
        self,
        year: int,
    ) -> Decimal:
        """
        Get the CPI deflator for a given year (base year 2021 = 1.0).

        Used to adjust spend-based emission calculations to the EEIO base year.

        Args:
            year: Calendar year (2015-2025 supported).

        Returns:
            Decimal CPI deflator value, quantized to 8 decimal places.

        Raises:
            ValueError: If the year is not found in the CPI deflator table.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> engine.get_cpi_deflator(2024)
            Decimal('1.14900000')
        """
        self._increment_lookup()

        deflator = CPI_DEFLATORS.get(year)
        if deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {year}. "
                f"Available years: {sorted(CPI_DEFLATORS.keys())}"
            )

        return self._quantize(deflator)

    # =========================================================================
    # 10. CLIMATE ZONE
    # =========================================================================

    def get_climate_zone(
        self,
        country_code: str,
    ) -> str:
        """
        Get the simplified climate zone for a given country.

        Uses Koppen-Geiger simplified mapping. Falls back to "temperate"
        if the country is not found.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US", "IN").

        Returns:
            Climate zone string: tropical, arid, temperate, continental,
            or polar.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> engine.get_climate_zone("IN")
            'tropical'
            >>> engine.get_climate_zone("SE")
            'polar'
        """
        self._increment_lookup()

        code = country_code.upper().strip()

        zone = CLIMATE_ZONE_BY_COUNTRY.get(code)
        if zone is not None:
            logger.debug("Climate zone lookup: country=%s, zone=%s", code, zone)
            return zone

        logger.warning(
            "Climate zone not found for country=%s; defaulting to temperate",
            code,
        )
        return "temperate"

    # =========================================================================
    # 11. REFRIGERANT GWP
    # =========================================================================

    def get_refrigerant_gwp(
        self,
        refrigerant: str,
    ) -> Dict[str, Any]:
        """
        Get refrigerant GWP (Global Warming Potential) values.

        Provides both AR5 (2014) and AR6 (2021) GWP values for the
        specified refrigerant.

        Args:
            refrigerant: Refrigerant designator (e.g., "R-410A", "R-134a").

        Returns:
            Dict with keys: gwp_ar6, gwp_ar5, source.
            Decimal values quantized to 8 decimal places.

        Raises:
            ValueError: If the refrigerant is not found.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> gwp = engine.get_refrigerant_gwp("R-410A")
            >>> gwp["gwp_ar6"]
            Decimal('2256.00000000')
        """
        self._increment_lookup()

        ref = refrigerant.upper().strip()
        # Normalize to R-xxx format if given without hyphen
        if ref.startswith("R") and len(ref) > 1 and ref[1] != "-":
            ref = "R-" + ref[1:]

        gwp_data = REFRIGERANT_GWP.get(ref)
        if gwp_data is not None:
            result = {
                "gwp_ar6": self._quantize(gwp_data["gwp_ar6"]),
                "gwp_ar5": self._quantize(gwp_data["gwp_ar5"]),
                "source": gwp_data["source"],
            }
            logger.debug(
                "Refrigerant GWP lookup: ref=%s, gwp_ar6=%s",
                ref, result["gwp_ar6"],
            )
            return result

        raise ValueError(
            f"Refrigerant GWP not available for '{ref}'. "
            f"Available refrigerants: {sorted(REFRIGERANT_GWP.keys())}"
        )

    # =========================================================================
    # 12. ALLOCATION DEFAULTS
    # =========================================================================

    def get_allocation_defaults(
        self,
        method: str,
    ) -> Dict[str, Any]:
        """
        Get allocation method description and precedence.

        Provides guidance on the preferred allocation method hierarchy for
        multi-tenant leased assets per GHG Protocol Scope 3 Category 8.

        Args:
            method: Allocation method key (floor_area, headcount, revenue,
                equal_share, time_based, energy_metered).

        Returns:
            Dict with keys: description, precedence.

        Raises:
            ValueError: If the allocation method is not recognized.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> alloc = engine.get_allocation_defaults("floor_area")
            >>> alloc["precedence"]
            Decimal('1.00000000')
        """
        self._increment_lookup()

        method_key = method.lower().strip().replace(" ", "_")

        alloc_data = ALLOCATION_DEFAULTS.get(method_key)
        if alloc_data is not None:
            result = {
                "description": alloc_data["description"],
                "precedence": self._quantize(alloc_data["precedence"]),
            }
            logger.debug(
                "Allocation defaults lookup: method=%s, precedence=%s",
                method_key, result["precedence"],
            )
            return result

        raise ValueError(
            f"Allocation method not recognized: '{method_key}'. "
            f"Available methods: {sorted(ALLOCATION_DEFAULTS.keys())}"
        )

    # =========================================================================
    # 13. DISTRICT HEATING EMISSION FACTORS
    # =========================================================================

    def get_district_heating_ef(
        self,
        region: str,
    ) -> Dict[str, Any]:
        """
        Get district heating emission factor by region.

        Falls back to GLOBAL average if the region is not found.

        Args:
            region: Region or country code (e.g., "UK", "DE", "SE", "GLOBAL").

        Returns:
            Dict with keys: ef_per_kwh, wtt_per_kwh, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> dh = engine.get_district_heating_ef("SE")
            >>> dh["ef_per_kwh"]
            Decimal('0.04900000')
        """
        self._increment_lookup()

        reg = region.upper().strip()

        dh_data = DISTRICT_HEATING_FACTORS.get(reg)
        if dh_data is not None:
            result = {
                "ef_per_kwh": self._quantize(dh_data["ef_per_kwh"]),
                "wtt_per_kwh": self._quantize(dh_data["wtt_per_kwh"]),
                "source": dh_data["source"],
            }
            logger.debug(
                "District heating EF lookup: region=%s, ef_per_kwh=%s",
                reg, result["ef_per_kwh"],
            )
            return result

        # Fallback to GLOBAL
        logger.warning(
            "District heating EF not found for region=%s; "
            "falling back to GLOBAL",
            reg,
        )
        global_data = DISTRICT_HEATING_FACTORS["GLOBAL"]
        return {
            "ef_per_kwh": self._quantize(global_data["ef_per_kwh"]),
            "wtt_per_kwh": self._quantize(global_data["wtt_per_kwh"]),
            "source": global_data["source"] + " (fallback)",
        }

    # =========================================================================
    # 14. SEARCH EMISSION FACTORS
    # =========================================================================

    def search_emission_factors(
        self,
        category: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search emission factors by category and query string (case-insensitive).

        Searches across the appropriate factor table based on category:
        - "building" / "eui": BUILDING_EUI_DATA
        - "grid" / "electricity": GRID_EMISSION_FACTORS
        - "fuel": FUEL_EMISSION_FACTORS
        - "vehicle" / "car" / "truck": VEHICLE_EMISSION_FACTORS
        - "equipment": EQUIPMENT_BENCHMARKS
        - "it" / "it_asset": IT_POWER_RATINGS
        - "eeio" / "spend": EEIO_FACTORS
        - "refrigerant": REFRIGERANT_GWP
        - "district_heating" / "dh": DISTRICT_HEATING_FACTORS

        Args:
            category: Factor category to search.
            query: Search query (partial name or code, case-insensitive).

        Returns:
            List of matching factor dicts. Empty list if no matches.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> results = engine.search_emission_factors("building", "office")
            >>> len(results) > 0
            True
        """
        self._increment_lookup()

        cat_lower = category.lower().strip()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        matches: List[Dict[str, Any]] = []

        if cat_lower in ("building", "eui"):
            matches = self._search_building_eui(query_lower)
        elif cat_lower in ("grid", "electricity"):
            matches = self._search_grid_factors(query_lower)
        elif cat_lower in ("fuel",):
            matches = self._search_fuel_factors(query_lower)
        elif cat_lower in ("vehicle", "car", "truck"):
            matches = self._search_vehicle_factors(query_lower)
        elif cat_lower in ("equipment",):
            matches = self._search_equipment_factors(query_lower)
        elif cat_lower in ("it", "it_asset"):
            matches = self._search_it_factors(query_lower)
        elif cat_lower in ("eeio", "spend"):
            matches = self._search_eeio_factors(query_lower)
        elif cat_lower in ("refrigerant",):
            matches = self._search_refrigerant_factors(query_lower)
        elif cat_lower in ("district_heating", "dh"):
            matches = self._search_dh_factors(query_lower)
        else:
            logger.warning(
                "Unknown search category '%s'; returning empty results",
                cat_lower,
            )

        logger.debug(
            "EF search: category=%s, query='%s', matches=%d",
            cat_lower, query_lower, len(matches),
        )

        return matches

    def _search_building_eui(self, query: str) -> List[Dict[str, Any]]:
        """Search building EUI factors matching query."""
        matches: List[Dict[str, Any]] = []
        for btype, zones in BUILDING_EUI_DATA.items():
            for zone, data in zones.items():
                key = f"{btype}_{zone}"
                if query in key.lower() or query in btype.lower():
                    matches.append({
                        "building_type": btype,
                        "climate_zone": zone,
                        "eui_kwh_sqm": self._quantize(data["eui_kwh_sqm"]),
                        "gas_fraction": self._quantize(data["gas_fraction"]),
                        "source": data["source"],
                    })
        return matches

    def _search_grid_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search grid emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for code, data in GRID_EMISSION_FACTORS.items():
            if query in code.lower():
                matches.append({
                    "country_code": code,
                    "co2e_per_kwh": self._quantize(data["co2e_per_kwh"]),
                    "wtt_per_kwh": self._quantize(data["wtt_per_kwh"]),
                    "source": data["source"],
                })
        return matches

    def _search_fuel_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search fuel emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for ftype, data in FUEL_EMISSION_FACTORS.items():
            if query in ftype.lower():
                matches.append({
                    "fuel_type": ftype,
                    "ef_per_kwh": self._quantize(data["ef_per_kwh"]),
                    "wtt_per_kwh": self._quantize(data["wtt_per_kwh"]),
                    "source": data["source"],
                })
        return matches

    def _search_vehicle_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search vehicle emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for vtype, fuels in VEHICLE_EMISSION_FACTORS.items():
            if query in vtype.lower():
                for ftype, ages in fuels.items():
                    for age_band, data in ages.items():
                        matches.append({
                            "vehicle_type": vtype,
                            "fuel_type": ftype,
                            "age_band": age_band,
                            "ef_per_km": self._quantize(data["ef_per_km"]),
                            "source": data["source"],
                        })
        return matches

    def _search_equipment_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search equipment benchmarks matching query."""
        matches: List[Dict[str, Any]] = []
        for etype, data in EQUIPMENT_BENCHMARKS.items():
            if query in etype.lower():
                matches.append({
                    "equipment_type": etype,
                    "default_hours": self._quantize(data["default_hours"]),
                    "load_factor": self._quantize(data["load_factor"]),
                    "fuel_type": data["fuel_type"],
                    "source": data["source"],
                })
        return matches

    def _search_it_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search IT asset power ratings matching query."""
        matches: List[Dict[str, Any]] = []
        for itype, data in IT_POWER_RATINGS.items():
            if query in itype.lower():
                matches.append({
                    "it_type": itype,
                    "typical_power_kw": self._quantize(data["typical_power_kw"]),
                    "utilization": self._quantize(data["utilization"]),
                    "pue_default": self._quantize(data["pue_default"]),
                    "source": data["source"],
                })
        return matches

    def _search_eeio_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search EEIO factors matching query."""
        matches: List[Dict[str, Any]] = []
        for code, data in EEIO_FACTORS.items():
            if query in code.lower() or query in data["description"].lower():
                matches.append({
                    "naics_code": code,
                    "co2e_per_usd": self._quantize(data["co2e_per_usd"]),
                    "description": data["description"],
                    "source": data["source"],
                })
        return matches

    def _search_refrigerant_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search refrigerant GWP values matching query."""
        matches: List[Dict[str, Any]] = []
        for ref, data in REFRIGERANT_GWP.items():
            if query in ref.lower():
                matches.append({
                    "refrigerant": ref,
                    "gwp_ar6": self._quantize(data["gwp_ar6"]),
                    "gwp_ar5": self._quantize(data["gwp_ar5"]),
                    "source": data["source"],
                })
        return matches

    def _search_dh_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search district heating factors matching query."""
        matches: List[Dict[str, Any]] = []
        for reg, data in DISTRICT_HEATING_FACTORS.items():
            if query in reg.lower():
                matches.append({
                    "region": reg,
                    "ef_per_kwh": self._quantize(data["ef_per_kwh"]),
                    "wtt_per_kwh": self._quantize(data["wtt_per_kwh"]),
                    "source": data["source"],
                })
        return matches

    # =========================================================================
    # 15. GET ALL BUILDING TYPES
    # =========================================================================

    def get_all_building_types(self) -> List[str]:
        """
        Get list of all available building types.

        Returns:
            Sorted list of building type strings.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> types = engine.get_all_building_types()
            >>> "office" in types
            True
            >>> "data_center" in types
            True
        """
        return sorted(BUILDING_EUI_DATA.keys())

    # =========================================================================
    # 16. GET ALL VEHICLE TYPES
    # =========================================================================

    def get_all_vehicle_types(self) -> List[str]:
        """
        Get list of all available vehicle types.

        Returns:
            Sorted list of vehicle type strings.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> types = engine.get_all_vehicle_types()
            >>> "medium_car_petrol" in types
            True
        """
        return sorted(VEHICLE_EMISSION_FACTORS.keys())

    # =========================================================================
    # 17. GET ALL EQUIPMENT TYPES
    # =========================================================================

    def get_all_equipment_types(self) -> List[str]:
        """
        Get list of all available equipment types.

        Returns:
            Sorted list of equipment type strings.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> types = engine.get_all_equipment_types()
            >>> "manufacturing" in types
            True
        """
        return sorted(EQUIPMENT_BENCHMARKS.keys())

    # =========================================================================
    # 18. GET ALL IT ASSET TYPES
    # =========================================================================

    def get_all_it_asset_types(self) -> List[str]:
        """
        Get list of all available IT asset types.

        Returns:
            Sorted list of IT asset type strings.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> types = engine.get_all_it_asset_types()
            >>> "server" in types
            True
        """
        return sorted(IT_POWER_RATINGS.keys())

    # =========================================================================
    # LOOKUP COUNT
    # =========================================================================

    def get_lookup_count(self) -> int:
        """
        Get the total number of factor lookups performed.

        Returns:
            Integer count of lookups.
        """
        with self._lookup_lock:
            return self._lookup_count

    # =========================================================================
    # DATABASE SUMMARY
    # =========================================================================

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database contents including counts of all factor
        tables and total lookups performed.

        Returns:
            Dict with counts of all factor categories.

        Example:
            >>> engine = UpstreamLeasedDatabaseEngine()
            >>> summary = engine.get_database_summary()
            >>> summary["building_types"]
            8
            >>> summary["grid_countries"]
            12
        """
        # Count total building x zone combinations
        building_combos = 0
        for _btype, zones in BUILDING_EUI_DATA.items():
            building_combos += len(zones)

        # Count total vehicle x fuel x age combinations
        vehicle_combos = 0
        for _vtype, fuels in VEHICLE_EMISSION_FACTORS.items():
            for _ftype, ages in fuels.items():
                vehicle_combos += len(ages)

        return {
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "building_types": len(BUILDING_EUI_DATA),
            "building_type_zone_combinations": building_combos,
            "grid_countries": len(GRID_EMISSION_FACTORS),
            "fuel_types": len(FUEL_EMISSION_FACTORS),
            "vehicle_types": len(VEHICLE_EMISSION_FACTORS),
            "vehicle_fuel_age_combinations": vehicle_combos,
            "equipment_types": len(EQUIPMENT_BENCHMARKS),
            "it_asset_types": len(IT_POWER_RATINGS),
            "eeio_naics_codes": len(EEIO_FACTORS),
            "currencies": len(CURRENCY_RATES),
            "cpi_years": len(CPI_DEFLATORS),
            "climate_zone_countries": len(CLIMATE_ZONE_BY_COUNTRY),
            "refrigerants": len(REFRIGERANT_GWP),
            "allocation_methods": len(ALLOCATION_DEFAULTS),
            "district_heating_regions": len(DISTRICT_HEATING_FACTORS),
            "total_lookups": self.get_lookup_count(),
        }

    # =========================================================================
    # RESET (TESTING ONLY)
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This method is intended for use in test fixtures only.
        Do not call in production code.
        """
        with cls._lock:
            cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSORS
# =============================================================================

_engine_instance: Optional[UpstreamLeasedDatabaseEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_database_engine() -> UpstreamLeasedDatabaseEngine:
    """
    Get the singleton UpstreamLeasedDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance.

    Returns:
        UpstreamLeasedDatabaseEngine singleton instance.

    Example:
        >>> engine = get_database_engine()
        >>> eui = engine.get_building_eui("office", "temperate")
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = UpstreamLeasedDatabaseEngine()
        return _engine_instance


def reset_database_engine() -> None:
    """
    Reset the module-level engine instance (for testing only).

    Warning: This function is intended for use in test fixtures only.
    Do not call in production code.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    UpstreamLeasedDatabaseEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "UpstreamLeasedDatabaseEngine",
    "get_database_engine",
    "reset_database_engine",
    # Extended constant tables (accessible for direct import if needed)
    "BUILDING_EUI_DATA",
    "GRID_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "VEHICLE_EMISSION_FACTORS",
    "EEIO_FACTORS",
    "EQUIPMENT_BENCHMARKS",
    "IT_POWER_RATINGS",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "CLIMATE_ZONE_BY_COUNTRY",
    "REFRIGERANT_GWP",
    "ALLOCATION_DEFAULTS",
    "DISTRICT_HEATING_FACTORS",
    "ENGINE_ID",
    "ENGINE_VERSION",
]
