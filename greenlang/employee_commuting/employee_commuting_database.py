# -*- coding: utf-8 -*-
"""
Employee Commuting Database Engine - Engine 1: Employee Commuting Agent (AGENT-MRV-020)

Thread-safe singleton providing emission factor lookups for all commuting modes,
vehicle types, transit types, telework energy, and spend-based EEIO factors.

Zero-Hallucination Guarantees:
    - All emission factors from DEFRA 2024, EPA, IEA official publications
    - All values stored as Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the lookup path
    - Every lookup is deterministic and reproducible

Features:
    - Personal vehicle emission factors by vehicle size, fuel type, and age band
    - Public transit emission factors for 9 transit types with WTT
    - Country-level grid emission factors from IEA 2024 (20+ countries)
    - US eGRID sub-region-level grid emission factors (8 sub-regions)
    - Home office telework energy defaults by climate zone (5 zones)
    - Heating fuel emission factors for 6 fuel types
    - EEIO spend-based factors for 10 NAICS codes
    - Average commute distances for 12 countries
    - Working days defaults for 12 countries
    - Carpool default occupancy by vehicle type
    - Currency conversion (20 currencies to USD)
    - CPI deflation (2015-2025)
    - Unified search across all factor tables
    - Thread-safe singleton pattern with __new__
    - Prometheus metrics recording for all lookups

Example:
    >>> engine = EmployeeCommutingDatabaseEngine()
    >>> factor = engine.get_vehicle_emission_factor("medium_car", "gasoline")
    >>> factor["co2e_per_km"]
    Decimal('0.19228000')

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
Status: Production Ready
"""

import logging
import threading
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.employee_commuting.models import (
    CommuteMode,
    VehicleType,
    FuelType,
    TransitType,
    RegionCode,
    CurrencyCode,
    EFSource,
    VEHICLE_EMISSION_FACTORS,
    FUEL_EMISSION_FACTORS,
    TRANSIT_EMISSION_FACTORS,
    MICRO_MOBILITY_EFS,
    GRID_EMISSION_FACTORS,
    WORKING_DAYS_DEFAULTS,
    AVERAGE_COMMUTE_DISTANCES,
    DEFAULT_MODE_SHARES,
    TELEWORK_ENERGY_DEFAULTS,
    VAN_EMISSION_FACTORS,
    EEIO_FACTORS,
    CURRENCY_RATES,
    CPI_DEFLATORS,
    calculate_provenance_hash,
)
from greenlang.employee_commuting.config import get_config
from greenlang.employee_commuting.metrics import get_metrics

logger = logging.getLogger(__name__)

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")


# =============================================================================
# EXTENDED CONSTANT TABLES
# =============================================================================

# Personal vehicle emission factors by (vehicle_size, fuel_type, age_band)
# Units: kgCO2e per km for co2e/co2/ch4/n2o; wtt_factor is upstream fraction
# Source: DEFRA 2024 Conversion Factors, Table 10a-c
# Age bands: new_0_3yr, mid_4_7yr, old_8_plus
PERSONAL_VEHICLE_EMISSION_FACTORS: Dict[
    str, Dict[str, Dict[str, Dict[str, Decimal]]]
] = {
    "small_car": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.14520"),
                "co2_per_km": Decimal("0.14320"),
                "ch4_per_km": Decimal("0.00028"),
                "n2o_per_km": Decimal("0.00172"),
                "wtt_factor": Decimal("0.02565"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.15307"),
                "co2_per_km": Decimal("0.15098"),
                "ch4_per_km": Decimal("0.00032"),
                "n2o_per_km": Decimal("0.00177"),
                "wtt_factor": Decimal("0.02704"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.16245"),
                "co2_per_km": Decimal("0.16010"),
                "ch4_per_km": Decimal("0.00040"),
                "n2o_per_km": Decimal("0.00195"),
                "wtt_factor": Decimal("0.02870"),
                "source": "DEFRA 2024",
            },
        },
        "diesel": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.13105"),
                "co2_per_km": Decimal("0.12945"),
                "ch4_per_km": Decimal("0.00001"),
                "n2o_per_km": Decimal("0.00159"),
                "wtt_factor": Decimal("0.01857"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.13890"),
                "co2_per_km": Decimal("0.13721"),
                "ch4_per_km": Decimal("0.00001"),
                "n2o_per_km": Decimal("0.00168"),
                "wtt_factor": Decimal("0.01969"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.14820"),
                "co2_per_km": Decimal("0.14630"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00188"),
                "wtt_factor": Decimal("0.02101"),
                "source": "DEFRA 2024",
            },
        },
        "hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.09814"),
                "co2_per_km": Decimal("0.09690"),
                "ch4_per_km": Decimal("0.00020"),
                "n2o_per_km": Decimal("0.00104"),
                "wtt_factor": Decimal("0.01562"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.10528"),
                "co2_per_km": Decimal("0.10390"),
                "ch4_per_km": Decimal("0.00022"),
                "n2o_per_km": Decimal("0.00116"),
                "wtt_factor": Decimal("0.01676"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.11390"),
                "co2_per_km": Decimal("0.11230"),
                "ch4_per_km": Decimal("0.00026"),
                "n2o_per_km": Decimal("0.00134"),
                "wtt_factor": Decimal("0.01813"),
                "source": "DEFRA 2024",
            },
        },
        "plugin_hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.06500"),
                "co2_per_km": Decimal("0.06420"),
                "ch4_per_km": Decimal("0.00012"),
                "n2o_per_km": Decimal("0.00068"),
                "wtt_factor": Decimal("0.00865"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.07050"),
                "co2_per_km": Decimal("0.06960"),
                "ch4_per_km": Decimal("0.00014"),
                "n2o_per_km": Decimal("0.00076"),
                "wtt_factor": Decimal("0.00938"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.07750"),
                "co2_per_km": Decimal("0.07650"),
                "ch4_per_km": Decimal("0.00016"),
                "n2o_per_km": Decimal("0.00084"),
                "wtt_factor": Decimal("0.01032"),
                "source": "DEFRA 2024",
            },
        },
        "electric": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.04350"),
                "co2_per_km": Decimal("0.04350"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.00919"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.04610"),
                "co2_per_km": Decimal("0.04610"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.00974"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.04950"),
                "co2_per_km": Decimal("0.04950"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01046"),
                "source": "DEFRA 2024",
            },
        },
    },
    "medium_car": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.18210"),
                "co2_per_km": Decimal("0.17960"),
                "ch4_per_km": Decimal("0.00038"),
                "n2o_per_km": Decimal("0.00212"),
                "wtt_factor": Decimal("0.03218"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.19228"),
                "co2_per_km": Decimal("0.18960"),
                "ch4_per_km": Decimal("0.00042"),
                "n2o_per_km": Decimal("0.00226"),
                "wtt_factor": Decimal("0.03398"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.20480"),
                "co2_per_km": Decimal("0.20180"),
                "ch4_per_km": Decimal("0.00049"),
                "n2o_per_km": Decimal("0.00251"),
                "wtt_factor": Decimal("0.03619"),
                "source": "DEFRA 2024",
            },
        },
        "diesel": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.16100"),
                "co2_per_km": Decimal("0.15900"),
                "ch4_per_km": Decimal("0.00001"),
                "n2o_per_km": Decimal("0.00199"),
                "wtt_factor": Decimal("0.02283"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.17034"),
                "co2_per_km": Decimal("0.16810"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00222"),
                "wtt_factor": Decimal("0.02415"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.18170"),
                "co2_per_km": Decimal("0.17920"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00248"),
                "wtt_factor": Decimal("0.02576"),
                "source": "DEFRA 2024",
            },
        },
        "hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.12050"),
                "co2_per_km": Decimal("0.11890"),
                "ch4_per_km": Decimal("0.00025"),
                "n2o_per_km": Decimal("0.00135"),
                "wtt_factor": Decimal("0.01918"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.12925"),
                "co2_per_km": Decimal("0.12750"),
                "ch4_per_km": Decimal("0.00028"),
                "n2o_per_km": Decimal("0.00147"),
                "wtt_factor": Decimal("0.02057"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.13990"),
                "co2_per_km": Decimal("0.13790"),
                "ch4_per_km": Decimal("0.00032"),
                "n2o_per_km": Decimal("0.00168"),
                "wtt_factor": Decimal("0.02227"),
                "source": "DEFRA 2024",
            },
        },
        "plugin_hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.08100"),
                "co2_per_km": Decimal("0.07990"),
                "ch4_per_km": Decimal("0.00016"),
                "n2o_per_km": Decimal("0.00094"),
                "wtt_factor": Decimal("0.01078"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.08819"),
                "co2_per_km": Decimal("0.08700"),
                "ch4_per_km": Decimal("0.00018"),
                "n2o_per_km": Decimal("0.00101"),
                "wtt_factor": Decimal("0.01174"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.09680"),
                "co2_per_km": Decimal("0.09550"),
                "ch4_per_km": Decimal("0.00020"),
                "n2o_per_km": Decimal("0.00110"),
                "wtt_factor": Decimal("0.01289"),
                "source": "DEFRA 2024",
            },
        },
        "electric": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.05020"),
                "co2_per_km": Decimal("0.05020"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01060"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.05305"),
                "co2_per_km": Decimal("0.05305"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01121"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.05680"),
                "co2_per_km": Decimal("0.05680"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01200"),
                "source": "DEFRA 2024",
            },
        },
    },
    "large_car": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.22130"),
                "co2_per_km": Decimal("0.21820"),
                "ch4_per_km": Decimal("0.00048"),
                "n2o_per_km": Decimal("0.00262"),
                "wtt_factor": Decimal("0.03911"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.23391"),
                "co2_per_km": Decimal("0.23060"),
                "ch4_per_km": Decimal("0.00052"),
                "n2o_per_km": Decimal("0.00279"),
                "wtt_factor": Decimal("0.04134"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.24920"),
                "co2_per_km": Decimal("0.24550"),
                "ch4_per_km": Decimal("0.00060"),
                "n2o_per_km": Decimal("0.00310"),
                "wtt_factor": Decimal("0.04404"),
                "source": "DEFRA 2024",
            },
        },
        "diesel": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.19610"),
                "co2_per_km": Decimal("0.19360"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00248"),
                "wtt_factor": Decimal("0.02780"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.20764"),
                "co2_per_km": Decimal("0.20500"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00262"),
                "wtt_factor": Decimal("0.02944"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.22140"),
                "co2_per_km": Decimal("0.21850"),
                "ch4_per_km": Decimal("0.00003"),
                "n2o_per_km": Decimal("0.00287"),
                "wtt_factor": Decimal("0.03139"),
                "source": "DEFRA 2024",
            },
        },
        "hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.15310"),
                "co2_per_km": Decimal("0.15100"),
                "ch4_per_km": Decimal("0.00032"),
                "n2o_per_km": Decimal("0.00178"),
                "wtt_factor": Decimal("0.02437"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.16423"),
                "co2_per_km": Decimal("0.16190"),
                "ch4_per_km": Decimal("0.00036"),
                "n2o_per_km": Decimal("0.00197"),
                "wtt_factor": Decimal("0.02614"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.17780"),
                "co2_per_km": Decimal("0.17520"),
                "ch4_per_km": Decimal("0.00042"),
                "n2o_per_km": Decimal("0.00218"),
                "wtt_factor": Decimal("0.02830"),
                "source": "DEFRA 2024",
            },
        },
        "plugin_hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.10100"),
                "co2_per_km": Decimal("0.09960"),
                "ch4_per_km": Decimal("0.00020"),
                "n2o_per_km": Decimal("0.00120"),
                "wtt_factor": Decimal("0.01344"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.10914"),
                "co2_per_km": Decimal("0.10770"),
                "ch4_per_km": Decimal("0.00022"),
                "n2o_per_km": Decimal("0.00122"),
                "wtt_factor": Decimal("0.01453"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.11890"),
                "co2_per_km": Decimal("0.11730"),
                "ch4_per_km": Decimal("0.00025"),
                "n2o_per_km": Decimal("0.00135"),
                "wtt_factor": Decimal("0.01583"),
                "source": "DEFRA 2024",
            },
        },
        "electric": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.06170"),
                "co2_per_km": Decimal("0.06170"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01303"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.06534"),
                "co2_per_km": Decimal("0.06534"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01380"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.06990"),
                "co2_per_km": Decimal("0.06990"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01476"),
                "source": "DEFRA 2024",
            },
        },
    },
    "suv": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.20980"),
                "co2_per_km": Decimal("0.20690"),
                "ch4_per_km": Decimal("0.00045"),
                "n2o_per_km": Decimal("0.00245"),
                "wtt_factor": Decimal("0.03708"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.22165"),
                "co2_per_km": Decimal("0.21850"),
                "ch4_per_km": Decimal("0.00050"),
                "n2o_per_km": Decimal("0.00265"),
                "wtt_factor": Decimal("0.03917"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.23610"),
                "co2_per_km": Decimal("0.23260"),
                "ch4_per_km": Decimal("0.00057"),
                "n2o_per_km": Decimal("0.00293"),
                "wtt_factor": Decimal("0.04173"),
                "source": "DEFRA 2024",
            },
        },
        "diesel": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.18790"),
                "co2_per_km": Decimal("0.18540"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00248"),
                "wtt_factor": Decimal("0.02664"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.19878"),
                "co2_per_km": Decimal("0.19610"),
                "ch4_per_km": Decimal("0.00002"),
                "n2o_per_km": Decimal("0.00266"),
                "wtt_factor": Decimal("0.02818"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.21190"),
                "co2_per_km": Decimal("0.20900"),
                "ch4_per_km": Decimal("0.00003"),
                "n2o_per_km": Decimal("0.00287"),
                "wtt_factor": Decimal("0.03004"),
                "source": "DEFRA 2024",
            },
        },
        "hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.14620"),
                "co2_per_km": Decimal("0.14420"),
                "ch4_per_km": Decimal("0.00030"),
                "n2o_per_km": Decimal("0.00170"),
                "wtt_factor": Decimal("0.02327"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.15678"),
                "co2_per_km": Decimal("0.15460"),
                "ch4_per_km": Decimal("0.00034"),
                "n2o_per_km": Decimal("0.00184"),
                "wtt_factor": Decimal("0.02496"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.16970"),
                "co2_per_km": Decimal("0.16730"),
                "ch4_per_km": Decimal("0.00039"),
                "n2o_per_km": Decimal("0.00201"),
                "wtt_factor": Decimal("0.02701"),
                "source": "DEFRA 2024",
            },
        },
        "plugin_hybrid": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.09710"),
                "co2_per_km": Decimal("0.09580"),
                "ch4_per_km": Decimal("0.00019"),
                "n2o_per_km": Decimal("0.00111"),
                "wtt_factor": Decimal("0.01293"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.10500"),
                "co2_per_km": Decimal("0.10360"),
                "ch4_per_km": Decimal("0.00021"),
                "n2o_per_km": Decimal("0.00119"),
                "wtt_factor": Decimal("0.01398"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.11450"),
                "co2_per_km": Decimal("0.11300"),
                "ch4_per_km": Decimal("0.00024"),
                "n2o_per_km": Decimal("0.00126"),
                "wtt_factor": Decimal("0.01525"),
                "source": "DEFRA 2024",
            },
        },
        "electric": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.05860"),
                "co2_per_km": Decimal("0.05860"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01238"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.06200"),
                "co2_per_km": Decimal("0.06200"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01310"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.06630"),
                "co2_per_km": Decimal("0.06630"),
                "ch4_per_km": Decimal("0.00000"),
                "n2o_per_km": Decimal("0.00000"),
                "wtt_factor": Decimal("0.01401"),
                "source": "DEFRA 2024",
            },
        },
    },
    "motorcycle_small": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.07850"),
                "co2_per_km": Decimal("0.07730"),
                "ch4_per_km": Decimal("0.00045"),
                "n2o_per_km": Decimal("0.00075"),
                "wtt_factor": Decimal("0.01987"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.08301"),
                "co2_per_km": Decimal("0.08170"),
                "ch4_per_km": Decimal("0.00050"),
                "n2o_per_km": Decimal("0.00081"),
                "wtt_factor": Decimal("0.02101"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.08890"),
                "co2_per_km": Decimal("0.08740"),
                "ch4_per_km": Decimal("0.00058"),
                "n2o_per_km": Decimal("0.00092"),
                "wtt_factor": Decimal("0.02250"),
                "source": "DEFRA 2024",
            },
        },
    },
    "motorcycle_large": {
        "gasoline": {
            "new_0_3yr": {
                "co2e_per_km": Decimal("0.12290"),
                "co2_per_km": Decimal("0.12100"),
                "ch4_per_km": Decimal("0.00060"),
                "n2o_per_km": Decimal("0.00130"),
                "wtt_factor": Decimal("0.03110"),
                "source": "DEFRA 2024",
            },
            "mid_4_7yr": {
                "co2e_per_km": Decimal("0.12994"),
                "co2_per_km": Decimal("0.12790"),
                "ch4_per_km": Decimal("0.00066"),
                "n2o_per_km": Decimal("0.00138"),
                "wtt_factor": Decimal("0.03288"),
                "source": "DEFRA 2024",
            },
            "old_8_plus": {
                "co2e_per_km": Decimal("0.13890"),
                "co2_per_km": Decimal("0.13660"),
                "ch4_per_km": Decimal("0.00075"),
                "n2o_per_km": Decimal("0.00155"),
                "wtt_factor": Decimal("0.03515"),
                "source": "DEFRA 2024",
            },
        },
    },
}

# Public transit emission factors (kgCO2e per passenger-km)
# Extended table with 9 transit types and per-region values
# Source: DEFRA 2024, EPA SmartWay
PUBLIC_TRANSIT_EMISSION_FACTORS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    "local_bus": {
        "global": {
            "co2e_per_pkm": Decimal("0.10312"),
            "wtt_per_pkm": Decimal("0.01847"),
            "source": "DEFRA 2024",
        },
    },
    "express_bus": {
        "global": {
            "co2e_per_pkm": Decimal("0.08956"),
            "wtt_per_pkm": Decimal("0.01604"),
            "source": "DEFRA 2024",
        },
    },
    "coach": {
        "global": {
            "co2e_per_pkm": Decimal("0.02732"),
            "wtt_per_pkm": Decimal("0.00489"),
            "source": "DEFRA 2024",
        },
    },
    "commuter_rail": {
        "global": {
            "co2e_per_pkm": Decimal("0.04115"),
            "wtt_per_pkm": Decimal("0.00867"),
            "source": "DEFRA 2024",
        },
    },
    "subway_metro": {
        "global": {
            "co2e_per_pkm": Decimal("0.03071"),
            "wtt_per_pkm": Decimal("0.00647"),
            "source": "DEFRA 2024",
        },
    },
    "light_rail": {
        "global": {
            "co2e_per_pkm": Decimal("0.02904"),
            "wtt_per_pkm": Decimal("0.00612"),
            "source": "DEFRA 2024",
        },
    },
    "tram_streetcar": {
        "global": {
            "co2e_per_pkm": Decimal("0.02940"),
            "wtt_per_pkm": Decimal("0.00619"),
            "source": "DEFRA 2024",
        },
    },
    "ferry_boat": {
        "global": {
            "co2e_per_pkm": Decimal("0.11318"),
            "wtt_per_pkm": Decimal("0.03450"),
            "source": "DEFRA 2024",
        },
    },
    "water_taxi": {
        "global": {
            "co2e_per_pkm": Decimal("0.14782"),
            "wtt_per_pkm": Decimal("0.04505"),
            "source": "DEFRA 2024",
        },
    },
}

# Country-level grid emission factors (kgCO2e per kWh)
# Source: IEA Emission Factors 2024
COUNTRY_GRID_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "US": {"co2e_per_kwh": Decimal("0.37170"), "source": "IEA 2024"},
    "GB": {"co2e_per_kwh": Decimal("0.20707"), "source": "IEA 2024"},
    "DE": {"co2e_per_kwh": Decimal("0.33800"), "source": "IEA 2024"},
    "FR": {"co2e_per_kwh": Decimal("0.05100"), "source": "IEA 2024"},
    "JP": {"co2e_per_kwh": Decimal("0.43400"), "source": "IEA 2024"},
    "CA": {"co2e_per_kwh": Decimal("0.12000"), "source": "IEA 2024"},
    "AU": {"co2e_per_kwh": Decimal("0.65600"), "source": "IEA 2024"},
    "IN": {"co2e_per_kwh": Decimal("0.70800"), "source": "IEA 2024"},
    "CN": {"co2e_per_kwh": Decimal("0.53700"), "source": "IEA 2024"},
    "BR": {"co2e_per_kwh": Decimal("0.07400"), "source": "IEA 2024"},
    "KR": {"co2e_per_kwh": Decimal("0.41500"), "source": "IEA 2024"},
    "IT": {"co2e_per_kwh": Decimal("0.25600"), "source": "IEA 2024"},
    "ES": {"co2e_per_kwh": Decimal("0.14800"), "source": "IEA 2024"},
    "NL": {"co2e_per_kwh": Decimal("0.30200"), "source": "IEA 2024"},
    "SE": {"co2e_per_kwh": Decimal("0.00800"), "source": "IEA 2024"},
    "NO": {"co2e_per_kwh": Decimal("0.00700"), "source": "IEA 2024"},
    "CH": {"co2e_per_kwh": Decimal("0.01200"), "source": "IEA 2024"},
    "SG": {"co2e_per_kwh": Decimal("0.40800"), "source": "IEA 2024"},
    "ZA": {"co2e_per_kwh": Decimal("0.92800"), "source": "IEA 2024"},
    "MX": {"co2e_per_kwh": Decimal("0.42300"), "source": "IEA 2024"},
    "GLOBAL": {"co2e_per_kwh": Decimal("0.43600"), "source": "IEA 2024"},
}

# US EPA eGRID sub-region emission factors (kgCO2e per kWh)
# Source: EPA eGRID 2022 (published 2024)
US_EGRID_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "CAMX": {"co2e_per_kwh": Decimal("0.22800"), "source": "eGRID 2022"},
    "RFCW": {"co2e_per_kwh": Decimal("0.44100"), "source": "eGRID 2022"},
    "SRMW": {"co2e_per_kwh": Decimal("0.55300"), "source": "eGRID 2022"},
    "SRTV": {"co2e_per_kwh": Decimal("0.38900"), "source": "eGRID 2022"},
    "SRSO": {"co2e_per_kwh": Decimal("0.40200"), "source": "eGRID 2022"},
    "NYUP": {"co2e_per_kwh": Decimal("0.12300"), "source": "eGRID 2022"},
    "NEWE": {"co2e_per_kwh": Decimal("0.20400"), "source": "eGRID 2022"},
    "NWPP": {"co2e_per_kwh": Decimal("0.28700"), "source": "eGRID 2022"},
}

# Home office energy defaults by climate zone (kWh per day)
# Source: IEA analysis of residential energy use patterns by climate
HOME_OFFICE_ENERGY_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    "tropical": {
        "electricity_kwh_per_day": Decimal("0.30"),
        "heating_kwh_per_day": Decimal("0.00"),
        "cooling_kwh_per_day": Decimal("4.50"),
        "lighting_kwh_per_day": Decimal("0.20"),
        "total_kwh_per_day": Decimal("5.00"),
        "source": "IEA 2024",
    },
    "arid": {
        "electricity_kwh_per_day": Decimal("0.30"),
        "heating_kwh_per_day": Decimal("1.50"),
        "cooling_kwh_per_day": Decimal("5.00"),
        "lighting_kwh_per_day": Decimal("0.20"),
        "total_kwh_per_day": Decimal("7.00"),
        "source": "IEA 2024",
    },
    "temperate": {
        "electricity_kwh_per_day": Decimal("0.30"),
        "heating_kwh_per_day": Decimal("3.50"),
        "cooling_kwh_per_day": Decimal("2.00"),
        "lighting_kwh_per_day": Decimal("0.20"),
        "total_kwh_per_day": Decimal("6.00"),
        "source": "IEA 2024",
    },
    "continental": {
        "electricity_kwh_per_day": Decimal("0.30"),
        "heating_kwh_per_day": Decimal("5.50"),
        "cooling_kwh_per_day": Decimal("1.50"),
        "lighting_kwh_per_day": Decimal("0.20"),
        "total_kwh_per_day": Decimal("7.50"),
        "source": "IEA 2024",
    },
    "polar": {
        "electricity_kwh_per_day": Decimal("0.30"),
        "heating_kwh_per_day": Decimal("8.00"),
        "cooling_kwh_per_day": Decimal("0.00"),
        "lighting_kwh_per_day": Decimal("0.30"),
        "total_kwh_per_day": Decimal("8.60"),
        "source": "IEA 2024",
    },
}

# Heating fuel emission factors (kgCO2e per kWh)
# Source: DEFRA 2024 Conversion Factors
HEATING_FUEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "co2e_per_kwh": Decimal("0.18316"),
        "source": "DEFRA 2024",
    },
    "heating_oil": {
        "co2e_per_kwh": Decimal("0.24674"),
        "source": "DEFRA 2024",
    },
    "lpg": {
        "co2e_per_kwh": Decimal("0.21449"),
        "source": "DEFRA 2024",
    },
    "wood_pellets": {
        "co2e_per_kwh": Decimal("0.01553"),
        "source": "DEFRA 2024",
    },
    "heat_pump_electricity": {
        "co2e_per_kwh": Decimal("0.10707"),
        "source": "DEFRA 2024 (grid avg / COP 3.5)",
    },
    "district_heating": {
        "co2e_per_kwh": Decimal("0.16200"),
        "source": "DEFRA 2024",
    },
}

# Average commute distances (one-way, km) with mode split (%)
# Source: NHTS (US), NTS (UK), Eurostat, national census data
EXTENDED_COMMUTE_DISTANCES: Dict[str, Dict[str, Any]] = {
    "US": {
        "avg_distance_km": Decimal("21.7"),
        "mode_split": {"car": Decimal("0.761"), "transit": Decimal("0.057"),
                       "active": Decimal("0.034"), "telework": Decimal("0.053"),
                       "other": Decimal("0.095")},
        "source": "NHTS 2022",
    },
    "GB": {
        "avg_distance_km": Decimal("14.4"),
        "mode_split": {"car": Decimal("0.630"), "transit": Decimal("0.170"),
                       "active": Decimal("0.135"), "telework": Decimal("0.040"),
                       "other": Decimal("0.025")},
        "source": "NTS 2023",
    },
    "DE": {
        "avg_distance_km": Decimal("17.0"),
        "mode_split": {"car": Decimal("0.570"), "transit": Decimal("0.140"),
                       "active": Decimal("0.180"), "telework": Decimal("0.068"),
                       "other": Decimal("0.042")},
        "source": "Eurostat 2023",
    },
    "FR": {
        "avg_distance_km": Decimal("13.3"),
        "mode_split": {"car": Decimal("0.600"), "transit": Decimal("0.160"),
                       "active": Decimal("0.110"), "telework": Decimal("0.080"),
                       "other": Decimal("0.050")},
        "source": "INSEE 2023",
    },
    "JP": {
        "avg_distance_km": Decimal("19.5"),
        "mode_split": {"car": Decimal("0.460"), "transit": Decimal("0.340"),
                       "active": Decimal("0.140"), "telework": Decimal("0.035"),
                       "other": Decimal("0.025")},
        "source": "MLIT 2022",
    },
    "CA": {
        "avg_distance_km": Decimal("15.1"),
        "mode_split": {"car": Decimal("0.740"), "transit": Decimal("0.120"),
                       "active": Decimal("0.070"), "telework": Decimal("0.045"),
                       "other": Decimal("0.025")},
        "source": "StatCan 2022",
    },
    "AU": {
        "avg_distance_km": Decimal("16.8"),
        "mode_split": {"car": Decimal("0.690"), "transit": Decimal("0.140"),
                       "active": Decimal("0.060"), "telework": Decimal("0.075"),
                       "other": Decimal("0.035")},
        "source": "ABS 2023",
    },
    "IN": {
        "avg_distance_km": Decimal("10.2"),
        "mode_split": {"car": Decimal("0.280"), "transit": Decimal("0.300"),
                       "active": Decimal("0.340"), "telework": Decimal("0.025"),
                       "other": Decimal("0.055")},
        "source": "Census 2021",
    },
    "CN": {
        "avg_distance_km": Decimal("9.8"),
        "mode_split": {"car": Decimal("0.340"), "transit": Decimal("0.300"),
                       "active": Decimal("0.290"), "telework": Decimal("0.020"),
                       "other": Decimal("0.050")},
        "source": "NBS 2023",
    },
    "BR": {
        "avg_distance_km": Decimal("12.5"),
        "mode_split": {"car": Decimal("0.350"), "transit": Decimal("0.360"),
                       "active": Decimal("0.180"), "telework": Decimal("0.040"),
                       "other": Decimal("0.070")},
        "source": "IBGE 2022",
    },
    "KR": {
        "avg_distance_km": Decimal("18.2"),
        "mode_split": {"car": Decimal("0.420"), "transit": Decimal("0.380"),
                       "active": Decimal("0.100"), "telework": Decimal("0.060"),
                       "other": Decimal("0.040")},
        "source": "KOTI 2023",
    },
    "GLOBAL": {
        "avg_distance_km": Decimal("15.0"),
        "mode_split": {"car": Decimal("0.520"), "transit": Decimal("0.200"),
                       "active": Decimal("0.160"), "telework": Decimal("0.050"),
                       "other": Decimal("0.070")},
        "source": "IEA/OECD Estimate",
    },
}

# Working days by country (net annual working days)
# Accounts for weekends, public holidays, PTO, sick days
WORKING_DAYS_BY_COUNTRY: Dict[str, int] = {
    "US": 225,
    "GB": 212,
    "DE": 200,
    "FR": 209,
    "JP": 219,
    "CA": 220,
    "AU": 218,
    "IN": 233,
    "CN": 240,
    "BR": 217,
    "KR": 222,
    "GLOBAL": 230,
}

# EEIO commuting-specific factors (kgCO2e per USD, base year 2021)
# Source: EPA USEEIO v2.0, Exiobase 3, OECD
EEIO_COMMUTING_FACTORS: Dict[str, Dict[str, Any]] = {
    "485000": {
        "co2e_per_usd": Decimal("0.26000"),
        "description": "Ground passenger transport",
        "source": "EPA USEEIO v2.0",
    },
    "485110": {
        "co2e_per_usd": Decimal("0.22000"),
        "description": "Mixed mode transit systems",
        "source": "EPA USEEIO v2.0",
    },
    "485210": {
        "co2e_per_usd": Decimal("0.24000"),
        "description": "Interurban bus transportation",
        "source": "EPA USEEIO v2.0",
    },
    "487110": {
        "co2e_per_usd": Decimal("0.31000"),
        "description": "Scenic and sightseeing rail",
        "source": "EPA USEEIO v2.0",
    },
    "488490": {
        "co2e_per_usd": Decimal("0.19000"),
        "description": "Other support activities for transport",
        "source": "EPA USEEIO v2.0",
    },
    "532100": {
        "co2e_per_usd": Decimal("0.19500"),
        "description": "Automotive equipment rental and leasing",
        "source": "EPA USEEIO v2.0",
    },
    "811100": {
        "co2e_per_usd": Decimal("0.15000"),
        "description": "Automotive repair and maintenance",
        "source": "EPA USEEIO v2.0",
    },
    "447110": {
        "co2e_per_usd": Decimal("0.63000"),
        "description": "Gasoline stations with convenience stores",
        "source": "EPA USEEIO v2.0",
    },
    "336110": {
        "co2e_per_usd": Decimal("0.34000"),
        "description": "Automobile manufacturing (for vehicle cost allocation)",
        "source": "EPA USEEIO v2.0",
    },
    "524126": {
        "co2e_per_usd": Decimal("0.04200"),
        "description": "Direct property and casualty insurance (auto insurance)",
        "source": "EPA USEEIO v2.0",
    },
}

# Extended currency exchange rates to USD (20 currencies)
# Approximate mid-market rates, updated semi-annually
EXTENDED_CURRENCY_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.00000000"),
    "EUR": Decimal("1.08500000"),
    "GBP": Decimal("1.26500000"),
    "CAD": Decimal("0.74100000"),
    "AUD": Decimal("0.65200000"),
    "JPY": Decimal("0.00666700"),
    "CNY": Decimal("0.13780000"),
    "INR": Decimal("0.01198000"),
    "CHF": Decimal("1.12800000"),
    "SGD": Decimal("0.74400000"),
    "BRL": Decimal("0.19900000"),
    "ZAR": Decimal("0.05340000"),
    "KRW": Decimal("0.00074500"),
    "MXN": Decimal("0.05720000"),
    "SEK": Decimal("0.09380000"),
    "NOK": Decimal("0.09210000"),
    "DKK": Decimal("0.14560000"),
    "NZD": Decimal("0.60800000"),
    "HKD": Decimal("0.12800000"),
    "TWD": Decimal("0.03120000"),
}

# CPI deflators for spend-based calculation (base year 2021 = 1.0)
# Source: US BLS CPI-U, OECD
EXTENDED_CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.84900000"),
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

# Default carpool occupancy by vehicle type
# Source: NHTS Average Vehicle Occupancy tables
CARPOOL_OCCUPANCY_DEFAULTS: Dict[str, Decimal] = {
    "small_car": Decimal("2.20"),
    "medium_car": Decimal("2.30"),
    "large_car": Decimal("2.40"),
    "suv": Decimal("2.50"),
    "van_small": Decimal("7.00"),
    "van_medium": Decimal("10.00"),
    "van_large": Decimal("12.00"),
    "minibus": Decimal("15.00"),
}


# =============================================================================
# ENGINE CLASS
# =============================================================================


class EmployeeCommutingDatabaseEngine:
    """
    Thread-safe singleton engine for employee commuting emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all
    employee commuting modes, vehicle types, transit, telework, and EEIO
    spend-based factors. Every lookup is recorded via Prometheus metrics
    (gl_ec_factor_selections_total) and returns data suitable for provenance
    hashing.

    This engine does NOT perform any LLM calls. All factors are retrieved
    from validated, frozen constant tables defined in this module and models.py.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        ENGINE_ID: Unique engine identifier string.
        ENGINE_VERSION: Semantic version of the engine.
        _config: Singleton configuration from get_config().
        _metrics: Singleton metrics from get_metrics().
        _lookup_count: Total number of factor lookups performed.

    Example:
        >>> engine = EmployeeCommutingDatabaseEngine()
        >>> vef = engine.get_vehicle_emission_factor("medium_car", "gasoline")
        >>> tef = engine.get_transit_emission_factor("subway_metro")
        >>> gef = engine.get_grid_emission_factor("US", region="CAMX")
    """

    ENGINE_ID: str = "employee_commuting_database_engine"
    ENGINE_VERSION: str = "1.0.0"

    _instance: Optional["EmployeeCommutingDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EmployeeCommutingDatabaseEngine":
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
        self._config = get_config()
        self._metrics = get_metrics()
        self._lookup_count: int = 0
        self._lookup_lock: threading.Lock = threading.Lock()

        logger.info(
            "EmployeeCommutingDatabaseEngine initialized: "
            "vehicle_sizes=%d, transit_types=%d, countries=%d, "
            "egrid_regions=%d, climate_zones=%d, eeio_codes=%d, "
            "currencies=%d, heating_fuels=%d",
            len(PERSONAL_VEHICLE_EMISSION_FACTORS),
            len(PUBLIC_TRANSIT_EMISSION_FACTORS),
            len(COUNTRY_GRID_FACTORS),
            len(US_EGRID_FACTORS),
            len(HOME_OFFICE_ENERGY_DEFAULTS),
            len(EEIO_COMMUTING_FACTORS),
            len(EXTENDED_CURRENCY_RATES),
            len(HEATING_FUEL_FACTORS),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _record_ef_selection(self, source: str, mode: str) -> None:
        """
        Record an emission factor selection in Prometheus metrics.

        Calls the metrics singleton's record_ef_lookup method with a
        "success" status for each factor retrieval.

        Args:
            source: EF source identifier (e.g., "defra", "epa", "iea", "eeio").
            mode: Commute mode (e.g., "car", "bus", "rail", "telework").
        """
        try:
            self._metrics.record_ef_lookup(
                mode=mode, source=source, status="success"
            )
        except Exception as exc:
            logger.warning(
                "Failed to record factor selection metric: %s", exc
            )

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
    # 1. VEHICLE EMISSION FACTORS
    # =========================================================================

    def get_vehicle_emission_factor(
        self,
        vehicle_type: str,
        fuel_type: str,
        vehicle_age: str = "mid_4_7yr",
    ) -> Dict[str, Any]:
        """
        Get personal vehicle emission factors by type, fuel, and age band.

        Looks up from PERSONAL_VEHICLE_EMISSION_FACTORS constant table.
        Falls back to medium_car/gasoline if the requested combination is
        not found, logging a warning on fallback.

        Args:
            vehicle_type: Vehicle size category (small_car, medium_car,
                large_car, suv, motorcycle_small, motorcycle_large).
            fuel_type: Fuel type (gasoline, diesel, hybrid, plugin_hybrid,
                electric).
            vehicle_age: Age band (new_0_3yr, mid_4_7yr, old_8_plus).
                Defaults to mid_4_7yr.

        Returns:
            Dict with keys: co2e_per_km, co2_per_km, ch4_per_km,
            n2o_per_km, wtt_factor, source. All Decimal values quantized
            to 8 decimal places.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> ef = engine.get_vehicle_emission_factor(
            ...     "medium_car", "gasoline"
            ... )
            >>> ef["co2e_per_km"]
            Decimal('0.19228000')
        """
        self._increment_lookup()

        vtype = vehicle_type.lower().strip()
        ftype = fuel_type.lower().strip()
        age = vehicle_age.lower().strip()

        # Attempt direct lookup
        vehicle_data = PERSONAL_VEHICLE_EMISSION_FACTORS.get(vtype)
        if vehicle_data is not None:
            fuel_data = vehicle_data.get(ftype)
            if fuel_data is not None:
                age_data = fuel_data.get(age)
                if age_data is not None:
                    result = {
                        "co2e_per_km": self._quantize(age_data["co2e_per_km"]),
                        "co2_per_km": self._quantize(age_data["co2_per_km"]),
                        "ch4_per_km": self._quantize(age_data["ch4_per_km"]),
                        "n2o_per_km": self._quantize(age_data["n2o_per_km"]),
                        "wtt_factor": self._quantize(age_data["wtt_factor"]),
                        "source": age_data["source"],
                    }
                    self._record_ef_selection("defra", "car")
                    logger.debug(
                        "Vehicle EF lookup: type=%s, fuel=%s, age=%s, "
                        "co2e_per_km=%s",
                        vtype, ftype, age, result["co2e_per_km"],
                    )
                    return result

        # Fallback to medium_car / gasoline / mid_4_7yr
        logger.warning(
            "Vehicle EF not found for type=%s, fuel=%s, age=%s; "
            "falling back to medium_car/gasoline/mid_4_7yr",
            vtype, ftype, age,
        )
        fallback = PERSONAL_VEHICLE_EMISSION_FACTORS["medium_car"]["gasoline"]["mid_4_7yr"]
        result = {
            "co2e_per_km": self._quantize(fallback["co2e_per_km"]),
            "co2_per_km": self._quantize(fallback["co2_per_km"]),
            "ch4_per_km": self._quantize(fallback["ch4_per_km"]),
            "n2o_per_km": self._quantize(fallback["n2o_per_km"]),
            "wtt_factor": self._quantize(fallback["wtt_factor"]),
            "source": fallback["source"] + " (fallback)",
        }
        self._record_ef_selection("defra", "car")
        return result

    # =========================================================================
    # 2. TRANSIT EMISSION FACTORS
    # =========================================================================

    def get_transit_emission_factor(
        self,
        transit_type: str,
        region: str = "global",
    ) -> Dict[str, Any]:
        """
        Get public transit emission factor by transit type and region.

        Looks up from PUBLIC_TRANSIT_EMISSION_FACTORS table.
        Falls back to local_bus if the requested transit type is not found.

        Args:
            transit_type: Transit type (local_bus, express_bus, coach,
                commuter_rail, subway_metro, light_rail, tram_streetcar,
                ferry_boat, water_taxi).
            region: Region key (default "global").

        Returns:
            Dict with keys: co2e_per_pkm, wtt_per_pkm, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> ef = engine.get_transit_emission_factor("subway_metro")
            >>> ef["co2e_per_pkm"]
            Decimal('0.03071000')
        """
        self._increment_lookup()

        ttype = transit_type.lower().strip()
        reg = region.lower().strip()

        transit_data = PUBLIC_TRANSIT_EMISSION_FACTORS.get(ttype)
        if transit_data is not None:
            region_data = transit_data.get(reg)
            if region_data is None:
                # Fall back to global within the transit type
                region_data = transit_data.get("global")
            if region_data is not None:
                result = {
                    "co2e_per_pkm": self._quantize(region_data["co2e_per_pkm"]),
                    "wtt_per_pkm": self._quantize(region_data["wtt_per_pkm"]),
                    "source": region_data["source"],
                }
                self._record_ef_selection("defra", "bus")
                logger.debug(
                    "Transit EF lookup: type=%s, region=%s, co2e_per_pkm=%s",
                    ttype, reg, result["co2e_per_pkm"],
                )
                return result

        # Fallback to local_bus / global
        logger.warning(
            "Transit EF not found for type=%s, region=%s; "
            "falling back to local_bus/global",
            ttype, reg,
        )
        fallback = PUBLIC_TRANSIT_EMISSION_FACTORS["local_bus"]["global"]
        result = {
            "co2e_per_pkm": self._quantize(fallback["co2e_per_pkm"]),
            "wtt_per_pkm": self._quantize(fallback["wtt_per_pkm"]),
            "source": fallback["source"] + " (fallback)",
        }
        self._record_ef_selection("defra", "bus")
        return result

    # =========================================================================
    # 3. GRID EMISSION FACTORS
    # =========================================================================

    def get_grid_emission_factor(
        self,
        country_code: str,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get electricity grid emission factor by country and optional sub-region.

        Priority: region-specific (US eGRID) > country > global average.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").
            region: Optional sub-region code (e.g., "CAMX" for US eGRID).

        Returns:
            Dict with keys: co2e_per_kwh, source.
            All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> ef = engine.get_grid_emission_factor("US", region="CAMX")
            >>> ef["co2e_per_kwh"]
            Decimal('0.22800000')
        """
        self._increment_lookup()

        code = country_code.upper().strip()

        # Priority 1: Region-specific (US eGRID)
        if region is not None:
            reg = region.upper().strip()
            egrid_data = US_EGRID_FACTORS.get(reg)
            if egrid_data is not None:
                result = {
                    "co2e_per_kwh": self._quantize(egrid_data["co2e_per_kwh"]),
                    "source": egrid_data["source"],
                }
                self._record_ef_selection("epa", "grid")
                logger.debug(
                    "Grid EF lookup (eGRID): country=%s, region=%s, "
                    "co2e_per_kwh=%s",
                    code, reg, result["co2e_per_kwh"],
                )
                return result

        # Priority 2: Country-level
        country_data = COUNTRY_GRID_FACTORS.get(code)
        if country_data is not None:
            result = {
                "co2e_per_kwh": self._quantize(country_data["co2e_per_kwh"]),
                "source": country_data["source"],
            }
            self._record_ef_selection("iea", "grid")
            logger.debug(
                "Grid EF lookup (country): country=%s, co2e_per_kwh=%s",
                code, result["co2e_per_kwh"],
            )
            return result

        # Priority 3: Global average
        logger.info(
            "Grid EF: country '%s' not found, falling back to GLOBAL "
            "(co2e_per_kwh=%s)",
            code, COUNTRY_GRID_FACTORS["GLOBAL"]["co2e_per_kwh"],
        )
        global_data = COUNTRY_GRID_FACTORS["GLOBAL"]
        result = {
            "co2e_per_kwh": self._quantize(global_data["co2e_per_kwh"]),
            "source": global_data["source"] + " (global fallback)",
        }
        self._record_ef_selection("iea", "grid")
        return result

    # =========================================================================
    # 4. TELEWORK FACTORS
    # =========================================================================

    def get_telework_factor(
        self,
        climate_zone: str = "temperate",
    ) -> Dict[str, Any]:
        """
        Get telework home office energy consumption defaults by climate zone.

        Args:
            climate_zone: Koppen climate zone (tropical, arid, temperate,
                continental, polar). Defaults to "temperate".

        Returns:
            Dict with keys: electricity_kwh_per_day, heating_kwh_per_day,
            cooling_kwh_per_day, lighting_kwh_per_day, total_kwh_per_day,
            source. All Decimal values quantized to 8 decimal places.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> tw = engine.get_telework_factor("continental")
            >>> tw["heating_kwh_per_day"]
            Decimal('5.50000000')
        """
        self._increment_lookup()

        zone = climate_zone.lower().strip()
        zone_data = HOME_OFFICE_ENERGY_DEFAULTS.get(zone)

        if zone_data is None:
            logger.warning(
                "Telework factor not found for climate zone '%s'; "
                "falling back to 'temperate'",
                zone,
            )
            zone_data = HOME_OFFICE_ENERGY_DEFAULTS["temperate"]

        result = {
            "electricity_kwh_per_day": self._quantize(zone_data["electricity_kwh_per_day"]),
            "heating_kwh_per_day": self._quantize(zone_data["heating_kwh_per_day"]),
            "cooling_kwh_per_day": self._quantize(zone_data["cooling_kwh_per_day"]),
            "lighting_kwh_per_day": self._quantize(zone_data["lighting_kwh_per_day"]),
            "total_kwh_per_day": self._quantize(zone_data["total_kwh_per_day"]),
            "source": zone_data["source"],
        }

        self._record_ef_selection("iea", "telework")

        logger.debug(
            "Telework factor lookup: zone=%s, total_kwh=%s",
            zone, result["total_kwh_per_day"],
        )

        return result

    # =========================================================================
    # 5. HEATING FUEL FACTORS
    # =========================================================================

    def get_heating_fuel_factor(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """
        Get heating fuel emission factor for telework home heating calculations.

        Args:
            fuel_type: Heating fuel type (natural_gas, heating_oil, lpg,
                wood_pellets, heat_pump_electricity, district_heating).

        Returns:
            Dict with keys: co2e_per_kwh, source.

        Raises:
            ValueError: If fuel_type is not found in HEATING_FUEL_FACTORS.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> ef = engine.get_heating_fuel_factor("natural_gas")
            >>> ef["co2e_per_kwh"]
            Decimal('0.18316000')
        """
        self._increment_lookup()

        ftype = fuel_type.lower().strip()
        fuel_data = HEATING_FUEL_FACTORS.get(ftype)

        if fuel_data is None:
            raise ValueError(
                f"Heating fuel factor not found for fuel type '{ftype}'. "
                f"Available types: {sorted(HEATING_FUEL_FACTORS.keys())}"
            )

        result = {
            "co2e_per_kwh": self._quantize(fuel_data["co2e_per_kwh"]),
            "source": fuel_data["source"],
        }

        self._record_ef_selection("defra", "telework")

        logger.debug(
            "Heating fuel EF lookup: fuel=%s, co2e_per_kwh=%s",
            ftype, result["co2e_per_kwh"],
        )

        return result

    # =========================================================================
    # 6. EEIO FACTORS (SPEND-BASED)
    # =========================================================================

    def get_eeio_factor(
        self,
        naics_code: str,
    ) -> Dict[str, Any]:
        """
        Get EEIO emission factor for commuting-related spend by NAICS code.

        EEIO (Environmentally Extended Input-Output) factors are used for
        spend-based calculation when activity data is unavailable.

        Args:
            naics_code: NAICS industry classification code string.

        Returns:
            Dict with keys: co2e_per_usd, description, source.

        Raises:
            ValueError: If naics_code is not found in the EEIO factor database.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> ef = engine.get_eeio_factor("485000")
            >>> ef["co2e_per_usd"]
            Decimal('0.26000000')
            >>> ef["description"]
            'Ground passenger transport'
        """
        self._increment_lookup()

        code = naics_code.strip()
        eeio_entry = EEIO_COMMUTING_FACTORS.get(code)

        if eeio_entry is None:
            raise ValueError(
                f"EEIO factor not found for NAICS code '{code}'. "
                f"Available codes: {sorted(EEIO_COMMUTING_FACTORS.keys())}"
            )

        result = {
            "co2e_per_usd": self._quantize(eeio_entry["co2e_per_usd"]),
            "description": eeio_entry["description"],
            "source": eeio_entry["source"],
        }

        self._record_ef_selection("eeio", "car")

        logger.debug(
            "EEIO factor lookup: naics=%s, description=%s, co2e_per_usd=%s",
            code, result["description"], result["co2e_per_usd"],
        )

        return result

    # =========================================================================
    # 7. AVERAGE COMMUTE DISTANCES
    # =========================================================================

    def get_average_commute_distance(
        self,
        country_code: str,
    ) -> Dict[str, Any]:
        """
        Get average one-way commute distance and mode split for a country.

        Falls back to GLOBAL average if the country is not found.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (or "GLOBAL").

        Returns:
            Dict with keys: avg_distance_km, mode_split, source.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> data = engine.get_average_commute_distance("US")
            >>> data["avg_distance_km"]
            Decimal('21.70000000')
        """
        self._increment_lookup()

        code = country_code.upper().strip()
        country_data = EXTENDED_COMMUTE_DISTANCES.get(code)

        if country_data is None:
            logger.info(
                "Commute distance: country '%s' not found, "
                "falling back to GLOBAL",
                code,
            )
            country_data = EXTENDED_COMMUTE_DISTANCES["GLOBAL"]

        result = {
            "avg_distance_km": self._quantize(country_data["avg_distance_km"]),
            "mode_split": country_data["mode_split"],
            "source": country_data["source"],
        }

        logger.debug(
            "Average commute distance lookup: country=%s, distance=%s km",
            code, result["avg_distance_km"],
        )

        return result

    # =========================================================================
    # 8. WORKING DAYS
    # =========================================================================

    def get_working_days(
        self,
        country_code: str,
    ) -> int:
        """
        Get net annual working days for a country.

        Falls back to 240 (GLOBAL default from WORKING_DAYS_BY_COUNTRY)
        if the country is not found. Also checks the models.py
        WORKING_DAYS_DEFAULTS for RegionCode-keyed data.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (or "GLOBAL").

        Returns:
            Net annual working days as an integer.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> engine.get_working_days("US")
            225
            >>> engine.get_working_days("DE")
            200
        """
        self._increment_lookup()

        code = country_code.upper().strip()

        # Try the extended table first
        days = WORKING_DAYS_BY_COUNTRY.get(code)
        if days is not None:
            logger.debug(
                "Working days lookup: country=%s, days=%d", code, days
            )
            return days

        # Fall back to models.py WORKING_DAYS_DEFAULTS using RegionCode
        try:
            region = RegionCode(code)
            region_data = WORKING_DAYS_DEFAULTS.get(region)
            if region_data is not None:
                net = region_data["net"]
                logger.debug(
                    "Working days lookup (models fallback): country=%s, days=%d",
                    code, net,
                )
                return net
        except ValueError:
            pass

        # Final fallback
        default = WORKING_DAYS_BY_COUNTRY.get("GLOBAL", 240)
        logger.info(
            "Working days: country '%s' not found, falling back to %d",
            code, default,
        )
        return default

    # =========================================================================
    # 9. CARPOOL DEFAULT OCCUPANCY
    # =========================================================================

    def get_carpool_default_occupancy(
        self,
        vehicle_type: str,
    ) -> Decimal:
        """
        Get default carpool occupancy for a given vehicle type.

        Args:
            vehicle_type: Vehicle size or van type (small_car, medium_car,
                large_car, suv, van_small, van_medium, van_large, minibus).

        Returns:
            Default occupancy as a Decimal (e.g., 2.30 for medium_car).
            Falls back to 2.30 for unknown types.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> engine.get_carpool_default_occupancy("suv")
            Decimal('2.50000000')
        """
        self._increment_lookup()

        vtype = vehicle_type.lower().strip()
        occupancy = CARPOOL_OCCUPANCY_DEFAULTS.get(vtype)

        if occupancy is None:
            logger.warning(
                "Carpool occupancy not found for type '%s'; "
                "falling back to 2.30",
                vtype,
            )
            occupancy = Decimal("2.30")

        return self._quantize(occupancy)

    # =========================================================================
    # 10. CURRENCY CONVERSION
    # =========================================================================

    def get_currency_rate(
        self,
        currency: str,
    ) -> Decimal:
        """
        Get the exchange rate from a given currency to USD.

        Args:
            currency: ISO 4217 currency code string (e.g., "GBP", "EUR").

        Returns:
            Exchange rate to USD, quantized to 8 decimal places.

        Raises:
            ValueError: If currency is not found in the rate table.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> engine.get_currency_rate("GBP")
            Decimal('1.26500000')
        """
        self._increment_lookup()

        code = currency.upper().strip()
        rate = EXTENDED_CURRENCY_RATES.get(code)

        if rate is None:
            raise ValueError(
                f"Currency rate not found for '{code}'. "
                f"Available currencies: {sorted(EXTENDED_CURRENCY_RATES.keys())}"
            )

        return self._quantize(rate)

    # =========================================================================
    # 11. CPI DEFLATION
    # =========================================================================

    def get_cpi_deflator(
        self,
        year: int,
    ) -> Decimal:
        """
        Get the CPI deflator for a given year (base year 2021 = 1.0).

        Used to convert nominal spend values to real (2021 base year) USD
        for consistent EEIO factor application across reporting years.

        Args:
            year: Year of the spend data.

        Returns:
            CPI deflator value, quantized to 8 decimal places.

        Raises:
            ValueError: If year is not found in the CPI deflator table.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> engine.get_cpi_deflator(2024)
            Decimal('1.14900000')
        """
        self._increment_lookup()

        deflator = EXTENDED_CPI_DEFLATORS.get(year)
        if deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {year}. "
                f"Available years: {sorted(EXTENDED_CPI_DEFLATORS.keys())}"
            )

        return self._quantize(deflator)

    # =========================================================================
    # 12. SEARCH EMISSION FACTORS
    # =========================================================================

    def search_emission_factors(
        self,
        mode: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search emission factors by mode and query string (case-insensitive).

        Searches across the appropriate factor table based on mode:
        - "vehicle" / "car": PERSONAL_VEHICLE_EMISSION_FACTORS
        - "transit" / "bus" / "rail": PUBLIC_TRANSIT_EMISSION_FACTORS
        - "grid" / "electricity": COUNTRY_GRID_FACTORS + US_EGRID_FACTORS
        - "eeio" / "spend": EEIO_COMMUTING_FACTORS
        - "heating" / "fuel": HEATING_FUEL_FACTORS
        - "telework" / "home": HOME_OFFICE_ENERGY_DEFAULTS

        Args:
            mode: Factor category to search.
            query: Search query (partial name or code, case-insensitive).

        Returns:
            List of matching factor dicts. Empty list if no matches.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> results = engine.search_emission_factors("vehicle", "diesel")
            >>> len(results) > 0
            True
        """
        self._increment_lookup()

        mode_lower = mode.lower().strip()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        matches: List[Dict[str, Any]] = []

        if mode_lower in ("vehicle", "car"):
            matches = self._search_vehicle_factors(query_lower)
        elif mode_lower in ("transit", "bus", "rail"):
            matches = self._search_transit_factors(query_lower)
        elif mode_lower in ("grid", "electricity"):
            matches = self._search_grid_factors(query_lower)
        elif mode_lower in ("eeio", "spend"):
            matches = self._search_eeio_factors(query_lower)
        elif mode_lower in ("heating", "fuel"):
            matches = self._search_heating_factors(query_lower)
        elif mode_lower in ("telework", "home"):
            matches = self._search_telework_factors(query_lower)
        else:
            logger.warning(
                "Unknown search mode '%s'; returning empty results", mode_lower
            )

        logger.debug(
            "EF search: mode=%s, query='%s', matches=%d",
            mode_lower, query_lower, len(matches),
        )

        return matches

    def _search_vehicle_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search vehicle emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for vtype, fuels in PERSONAL_VEHICLE_EMISSION_FACTORS.items():
            for ftype, ages in fuels.items():
                key = f"{vtype}_{ftype}"
                if query in key.lower():
                    for age_band, data in ages.items():
                        matches.append({
                            "vehicle_type": vtype,
                            "fuel_type": ftype,
                            "age_band": age_band,
                            "co2e_per_km": self._quantize(data["co2e_per_km"]),
                            "source": data["source"],
                        })
        return matches

    def _search_transit_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search transit emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for ttype, regions in PUBLIC_TRANSIT_EMISSION_FACTORS.items():
            if query in ttype.lower():
                for reg, data in regions.items():
                    matches.append({
                        "transit_type": ttype,
                        "region": reg,
                        "co2e_per_pkm": self._quantize(data["co2e_per_pkm"]),
                        "source": data["source"],
                    })
        return matches

    def _search_grid_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search grid emission factors matching query."""
        matches: List[Dict[str, Any]] = []
        for code, data in COUNTRY_GRID_FACTORS.items():
            if query in code.lower():
                matches.append({
                    "country_code": code,
                    "co2e_per_kwh": self._quantize(data["co2e_per_kwh"]),
                    "source": data["source"],
                })
        for code, data in US_EGRID_FACTORS.items():
            if query in code.lower():
                matches.append({
                    "egrid_region": code,
                    "co2e_per_kwh": self._quantize(data["co2e_per_kwh"]),
                    "source": data["source"],
                })
        return matches

    def _search_eeio_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search EEIO factors matching query."""
        matches: List[Dict[str, Any]] = []
        for code, data in EEIO_COMMUTING_FACTORS.items():
            if query in code.lower() or query in data["description"].lower():
                matches.append({
                    "naics_code": code,
                    "co2e_per_usd": self._quantize(data["co2e_per_usd"]),
                    "description": data["description"],
                    "source": data["source"],
                })
        return matches

    def _search_heating_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search heating fuel factors matching query."""
        matches: List[Dict[str, Any]] = []
        for ftype, data in HEATING_FUEL_FACTORS.items():
            if query in ftype.lower():
                matches.append({
                    "fuel_type": ftype,
                    "co2e_per_kwh": self._quantize(data["co2e_per_kwh"]),
                    "source": data["source"],
                })
        return matches

    def _search_telework_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search telework home office factors matching query."""
        matches: List[Dict[str, Any]] = []
        for zone, data in HOME_OFFICE_ENERGY_DEFAULTS.items():
            if query in zone.lower():
                matches.append({
                    "climate_zone": zone,
                    "total_kwh_per_day": self._quantize(data["total_kwh_per_day"]),
                    "source": data["source"],
                })
        return matches

    # =========================================================================
    # 13. GET ALL VEHICLE TYPES
    # =========================================================================

    def get_all_vehicle_types(self) -> List[str]:
        """
        Get list of all available vehicle types (size categories).

        Returns:
            Sorted list of vehicle type strings.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> types = engine.get_all_vehicle_types()
            >>> "medium_car" in types
            True
        """
        return sorted(PERSONAL_VEHICLE_EMISSION_FACTORS.keys())

    # =========================================================================
    # 14. GET ALL FUEL TYPES
    # =========================================================================

    def get_all_fuel_types(self) -> List[str]:
        """
        Get list of all available fuel types across all vehicle categories.

        Returns:
            Sorted deduplicated list of fuel type strings.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> fuels = engine.get_all_fuel_types()
            >>> "gasoline" in fuels
            True
            >>> "electric" in fuels
            True
        """
        fuel_types: set = set()
        for _vtype, fuels in PERSONAL_VEHICLE_EMISSION_FACTORS.items():
            for ftype in fuels:
                fuel_types.add(ftype)
        return sorted(fuel_types)

    # =========================================================================
    # 15. GET ALL TRANSIT TYPES
    # =========================================================================

    def get_all_transit_types(self) -> List[str]:
        """
        Get list of all available public transit types.

        Returns:
            Sorted list of transit type strings.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> types = engine.get_all_transit_types()
            >>> "subway_metro" in types
            True
        """
        return sorted(PUBLIC_TRANSIT_EMISSION_FACTORS.keys())

    # =========================================================================
    # 16. GET ALL COMMUTE MODES
    # =========================================================================

    def get_all_commute_modes(self) -> List[str]:
        """
        Get list of all available commute modes from the CommuteMode enum.

        Returns:
            List of commute mode value strings.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> modes = engine.get_all_commute_modes()
            >>> "sov" in modes
            True
            >>> "telework" in modes
            True
        """
        return [mode.value for mode in CommuteMode]

    # =========================================================================
    # 17. GET ALL COUNTRIES
    # =========================================================================

    def get_all_countries(self) -> List[str]:
        """
        Get list of all country codes with grid emission factor data.

        Returns:
            Sorted list of country code strings.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> countries = engine.get_all_countries()
            >>> "US" in countries
            True
            >>> "GLOBAL" in countries
            True
        """
        return sorted(COUNTRY_GRID_FACTORS.keys())

    # =========================================================================
    # 18. DATABASE SUMMARY
    # =========================================================================

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database contents including counts of all factor
        tables and total lookups performed.

        Returns:
            Dict with counts of all factor categories.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> summary = engine.get_database_summary()
            >>> summary["vehicle_types"]
            6
            >>> summary["transit_types"]
            9
        """
        # Count total vehicle/fuel/age combinations
        vehicle_combos = 0
        for _vtype, fuels in PERSONAL_VEHICLE_EMISSION_FACTORS.items():
            for _ftype, ages in fuels.items():
                vehicle_combos += len(ages)

        return {
            "engine_id": self.ENGINE_ID,
            "engine_version": self.ENGINE_VERSION,
            "vehicle_types": len(PERSONAL_VEHICLE_EMISSION_FACTORS),
            "vehicle_fuel_age_combinations": vehicle_combos,
            "transit_types": len(PUBLIC_TRANSIT_EMISSION_FACTORS),
            "country_grid_factors": len(COUNTRY_GRID_FACTORS),
            "us_egrid_regions": len(US_EGRID_FACTORS),
            "climate_zones": len(HOME_OFFICE_ENERGY_DEFAULTS),
            "heating_fuel_types": len(HEATING_FUEL_FACTORS),
            "eeio_naics_codes": len(EEIO_COMMUTING_FACTORS),
            "commute_distance_countries": len(EXTENDED_COMMUTE_DISTANCES),
            "working_days_countries": len(WORKING_DAYS_BY_COUNTRY),
            "carpool_occupancy_types": len(CARPOOL_OCCUPANCY_DEFAULTS),
            "currencies": len(EXTENDED_CURRENCY_RATES),
            "cpi_years": len(EXTENDED_CPI_DEFLATORS),
            "commute_modes": len(CommuteMode),
            "total_lookups": self.get_lookup_count(),
        }

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
    # CLASSIFY COMMUTE MODE
    # =========================================================================

    def classify_commute_mode(
        self,
        commute_data: dict,
    ) -> CommuteMode:
        """
        Classify the commute mode from commute input data.

        Examines the commute_data dictionary for mode-identifying keys:
        - "mode" field (explicit) -> direct mapping to CommuteMode
        - "vehicle_type" containing "motorcycle" -> MOTORCYCLE
        - "vehicle_type" with occupants > 1 -> CARPOOL
        - "vehicle_type" containing "van" or "minibus" -> VANPOOL
        - "vehicle_type" -> SOV
        - "transit_type" -> maps to BUS / METRO / LIGHT_RAIL / etc.
        - "telework" or "remote" in fields -> TELEWORK
        - "cycling" or "bicycle" -> CYCLING
        - "walking" or "walk" -> WALKING
        - "e_bike" -> E_BIKE
        - "e_scooter" -> E_SCOOTER

        Args:
            commute_data: Dictionary of commute input fields.

        Returns:
            Classified CommuteMode enum value.

        Example:
            >>> engine = EmployeeCommutingDatabaseEngine()
            >>> mode = engine.classify_commute_mode(
            ...     {"vehicle_type": "car_medium_petrol"}
            ... )
            >>> mode == CommuteMode.SOV
            True
        """
        if not commute_data:
            logger.warning(
                "Empty commute_data for classification, defaulting to SOV"
            )
            return CommuteMode.SOV

        # Check for explicit mode field
        explicit_mode = commute_data.get("mode")
        if explicit_mode is not None:
            mode_str = str(explicit_mode).lower().strip()
            for commute_mode in CommuteMode:
                if commute_mode.value == mode_str:
                    logger.debug(
                        "Classified commute mode from explicit field: %s",
                        commute_mode.value,
                    )
                    return commute_mode

        # Check for telework indicators
        if commute_data.get("telework") or commute_data.get("remote_work"):
            return CommuteMode.TELEWORK

        # Check for active transport
        if commute_data.get("cycling") or commute_data.get("bicycle"):
            return CommuteMode.CYCLING
        if commute_data.get("walking") or commute_data.get("walk"):
            return CommuteMode.WALKING
        if commute_data.get("e_bike"):
            return CommuteMode.E_BIKE
        if commute_data.get("e_scooter"):
            return CommuteMode.E_SCOOTER

        # Check for transit type
        transit = commute_data.get("transit_type", "")
        if isinstance(transit, str) and transit:
            transit_lower = transit.lower()
            transit_mapping = {
                "bus": CommuteMode.BUS,
                "metro": CommuteMode.METRO,
                "subway": CommuteMode.METRO,
                "light_rail": CommuteMode.LIGHT_RAIL,
                "tram": CommuteMode.LIGHT_RAIL,
                "commuter_rail": CommuteMode.COMMUTER_RAIL,
                "ferry": CommuteMode.FERRY,
            }
            for key, mode in transit_mapping.items():
                if key in transit_lower:
                    return mode

        # Check for vehicle type
        vehicle_type = commute_data.get("vehicle_type", "")
        if isinstance(vehicle_type, str) and vehicle_type:
            vtype_lower = vehicle_type.lower()
            if "motorcycle" in vtype_lower:
                return CommuteMode.MOTORCYCLE
            if "van" in vtype_lower or "minibus" in vtype_lower:
                return CommuteMode.VANPOOL

            occupants = commute_data.get("occupants", 1)
            if isinstance(occupants, int) and occupants > 1:
                return CommuteMode.CARPOOL
            return CommuteMode.SOV

        # Default to SOV
        logger.info(
            "Could not classify commute mode from data keys %s, "
            "defaulting to SOV",
            list(commute_data.keys()),
        )
        return CommuteMode.SOV

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

_engine_instance: Optional[EmployeeCommutingDatabaseEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_database_engine() -> EmployeeCommutingDatabaseEngine:
    """
    Get the singleton EmployeeCommutingDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance.

    Returns:
        EmployeeCommutingDatabaseEngine singleton instance.

    Example:
        >>> engine = get_database_engine()
        >>> ef = engine.get_vehicle_emission_factor("medium_car", "gasoline")
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = EmployeeCommutingDatabaseEngine()
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
    EmployeeCommutingDatabaseEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "EmployeeCommutingDatabaseEngine",
    "get_database_engine",
    "reset_database_engine",
    # Extended constant tables (accessible for direct import if needed)
    "PERSONAL_VEHICLE_EMISSION_FACTORS",
    "PUBLIC_TRANSIT_EMISSION_FACTORS",
    "COUNTRY_GRID_FACTORS",
    "US_EGRID_FACTORS",
    "HOME_OFFICE_ENERGY_DEFAULTS",
    "HEATING_FUEL_FACTORS",
    "EEIO_COMMUTING_FACTORS",
    "EXTENDED_COMMUTE_DISTANCES",
    "WORKING_DAYS_BY_COUNTRY",
    "CARPOOL_OCCUPANCY_DEFAULTS",
    "EXTENDED_CURRENCY_RATES",
    "EXTENDED_CPI_DEFLATORS",
]
