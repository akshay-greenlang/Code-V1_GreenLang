# -*- coding: utf-8 -*-
"""
DownstreamTransportDatabaseEngine - Emission factor database for downstream transport.

This module implements the DownstreamTransportDatabaseEngine for AGENT-MRV-022
(Downstream Transportation & Distribution, GHG Protocol Scope 3 Category 9).

It provides thread-safe singleton access to comprehensive emission factor databases
for all downstream transport modes (road, rail, maritime, air, pipeline), cold chain
uplift factors, warehouse/DC emission factors, last-mile delivery factors, EEIO
spend-based factors, currency conversion, CPI deflation, distribution channel
defaults, Incoterm classifications, and data quality scoring.

Scope 3 Category 9 covers emissions from transportation and distribution of products
sold by the reporting company in the reporting year between the reporting company's
operations and the end consumer (not paid for by the reporting company).

Features:
    - 13 road vehicle types with EF per tonne-km (DEFRA 2024, GLEC v3.0)
    - 4 rail freight types with regional adjustment
    - 16 maritime vessel types with IMO 4th GHG Study factors
    - 5 aircraft types for air freight
    - 3 pipeline types
    - Cold chain uplift factors for CHILLED and FROZEN by mode
    - 6 warehouse/DC emission factors by type and region
    - 8 last-mile delivery types by area (urban/suburban/rural)
    - 18 EEIO sector factors for spend-based calculations
    - 12 currency conversion rates to USD
    - 11-year CPI deflators (2015-2025)
    - 30 country grid emission factors
    - 8 distribution channel defaults with avg distance, mode, weight
    - 13 Incoterm classifications for Cat 4 vs Cat 9 scope
    - Load factor adjustments
    - Return trip multipliers (empty/partial/full)
    - 5-dimension DQI scoring per ISO 14083
    - Uncertainty ranges by calculation method
    - Thread-safe singleton with get/reset pattern
    - Zero-hallucination factor retrieval (no LLM calls)

All emission factors are sourced from:
    - DEFRA/DESNZ 2024 Government GHG Conversion Factors
    - EPA SmartWay Emission Factors
    - GLEC Framework v3.0 (Global Logistics Emissions Council)
    - IMO Fourth GHG Study 2020
    - ICAO Carbon Emissions Calculator v12
    - US EEIO 2.0 (Environmentally Extended Input-Output)
    - IEA CO2 Emissions from Fuel Combustion 2024

Example:
    >>> db = get_downstream_transport_database()
    >>> ef = db.get_transport_ef("ARTICULATED_40_44T")
    >>> print(ef["ef_per_tkm"])
    0.10300000
    >>> cold = db.get_cold_chain_uplift("FROZEN", "ROAD")
    >>> print(cold)
    1.30000000

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009 (AGENT-MRV-022)
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION
# ==============================================================================

_QUANT_8DP = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _q(value: Decimal) -> Decimal:
    """Quantize to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class TransportMode(str, Enum):
    """Transport mode classifications for downstream distribution."""

    ROAD = "ROAD"
    RAIL = "RAIL"
    MARITIME = "MARITIME"
    AIR = "AIR"
    PIPELINE = "PIPELINE"
    INTERMODAL = "INTERMODAL"


class TemperatureRegime(str, Enum):
    """Temperature control regime for cold chain."""

    AMBIENT = "AMBIENT"
    CHILLED = "CHILLED"
    FROZEN = "FROZEN"


class WarehouseType(str, Enum):
    """Warehouse and distribution center types."""

    AMBIENT_WAREHOUSE = "AMBIENT_WAREHOUSE"
    REFRIGERATED_WAREHOUSE = "REFRIGERATED_WAREHOUSE"
    DISTRIBUTION_CENTER = "DISTRIBUTION_CENTER"
    CROSS_DOCK = "CROSS_DOCK"
    RETAIL_STORE = "RETAIL_STORE"
    FULFILLMENT_CENTER = "FULFILLMENT_CENTER"


class DeliveryType(str, Enum):
    """Last-mile delivery types."""

    PARCEL_VAN = "PARCEL_VAN"
    CARGO_BIKE = "CARGO_BIKE"
    ELECTRIC_VAN = "ELECTRIC_VAN"
    DRONE = "DRONE"
    LOCKER_PICKUP = "LOCKER_PICKUP"
    CLICK_AND_COLLECT = "CLICK_AND_COLLECT"
    STANDARD_TRUCK = "STANDARD_TRUCK"
    HEAVY_GOODS = "HEAVY_GOODS"


class DeliveryArea(str, Enum):
    """Delivery area classification."""

    URBAN = "URBAN"
    SUBURBAN = "SUBURBAN"
    RURAL = "RURAL"


class DistributionChannel(str, Enum):
    """Distribution channel types."""

    DIRECT_TO_CONSUMER = "DIRECT_TO_CONSUMER"
    WHOLESALE = "WHOLESALE"
    RETAIL_BRICK_MORTAR = "RETAIL_BRICK_MORTAR"
    ECOMMERCE = "ECOMMERCE"
    THIRD_PARTY_LOGISTICS = "THIRD_PARTY_LOGISTICS"
    FRANCHISE = "FRANCHISE"
    DISTRIBUTOR = "DISTRIBUTOR"
    DROP_SHIP = "DROP_SHIP"


class ReturnType(str, Enum):
    """Return trip type for transport vehicles."""

    EMPTY = "EMPTY"
    PARTIAL_LOAD = "PARTIAL_LOAD"
    FULL_LOAD = "FULL_LOAD"
    NO_RETURN = "NO_RETURN"


class CalculationMethodType(str, Enum):
    """Calculation method for uncertainty ranges."""

    DISTANCE_BASED = "DISTANCE_BASED"
    SPEND_BASED = "SPEND_BASED"
    AVERAGE_DATA = "AVERAGE_DATA"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"


# ==============================================================================
# TRANSPORT EMISSION FACTORS (kgCO2e per tonne-km, WTW)
# ==============================================================================
# Sources: DEFRA 2024, GLEC Framework v3.0, IMO 4th GHG Study, ICAO v12
# Each entry: { ef_per_tkm, wtt_per_tkm, mode, source, unit }
# All values represent Well-to-Wheel (WTW) unless noted.

TRANSPORT_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    # ---- ROAD: Light Commercial Vehicles ----
    "LCV_DIESEL": {
        "ef_per_tkm": Decimal("0.47200000"),
        "wtt_per_tkm": Decimal("0.10100000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "LCV_ELECTRIC": {
        "ef_per_tkm": Decimal("0.08500000"),
        "wtt_per_tkm": Decimal("0.08500000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "ELECTRIC",
        "unit": "kgCO2e/tkm",
    },
    # ---- ROAD: Rigid HGVs ----
    "RIGID_3_5_7_5T": {
        "ef_per_tkm": Decimal("0.35500000"),
        "wtt_per_tkm": Decimal("0.07600000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "RIGID_7_5_17T": {
        "ef_per_tkm": Decimal("0.24000000"),
        "wtt_per_tkm": Decimal("0.05100000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "RIGID_17_26T": {
        "ef_per_tkm": Decimal("0.20200000"),
        "wtt_per_tkm": Decimal("0.04300000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "RIGID_26_28T": {
        "ef_per_tkm": Decimal("0.18500000"),
        "wtt_per_tkm": Decimal("0.03900000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "RIGID_28_32T": {
        "ef_per_tkm": Decimal("0.17200000"),
        "wtt_per_tkm": Decimal("0.03700000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    # ---- ROAD: Articulated HGVs ----
    "ARTICULATED_33T": {
        "ef_per_tkm": Decimal("0.16000000"),
        "wtt_per_tkm": Decimal("0.03400000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "ARTICULATED_40_44T": {
        "ef_per_tkm": Decimal("0.10300000"),
        "wtt_per_tkm": Decimal("0.02800000"),
        "mode": "ROAD",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "ROAD_TRAIN": {
        "ef_per_tkm": Decimal("0.08200000"),
        "wtt_per_tkm": Decimal("0.02200000"),
        "mode": "ROAD",
        "source": "GLEC_V3.0",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    # ---- ROAD: Alternative fuels ----
    "TRUCK_CNG": {
        "ef_per_tkm": Decimal("0.11500000"),
        "wtt_per_tkm": Decimal("0.02300000"),
        "mode": "ROAD",
        "source": "GLEC_V3.0",
        "fuel_type": "CNG",
        "unit": "kgCO2e/tkm",
    },
    "TRUCK_LNG": {
        "ef_per_tkm": Decimal("0.11100000"),
        "wtt_per_tkm": Decimal("0.02600000"),
        "mode": "ROAD",
        "source": "GLEC_V3.0",
        "fuel_type": "LNG",
        "unit": "kgCO2e/tkm",
    },
    "TRUCK_HYDROGEN": {
        "ef_per_tkm": Decimal("0.12000000"),
        "wtt_per_tkm": Decimal("0.12000000"),
        "mode": "ROAD",
        "source": "GLEC_V3.0",
        "fuel_type": "HYDROGEN",
        "unit": "kgCO2e/tkm",
    },
    # ---- RAIL ----
    "FREIGHT_ELECTRIC": {
        "ef_per_tkm": Decimal("0.02500000"),
        "wtt_per_tkm": Decimal("0.02500000"),
        "mode": "RAIL",
        "source": "DEFRA_2024",
        "fuel_type": "ELECTRIC",
        "unit": "kgCO2e/tkm",
    },
    "FREIGHT_DIESEL": {
        "ef_per_tkm": Decimal("0.04600000"),
        "wtt_per_tkm": Decimal("0.01000000"),
        "mode": "RAIL",
        "source": "DEFRA_2024",
        "fuel_type": "DIESEL",
        "unit": "kgCO2e/tkm",
    },
    "INTERMODAL_RAIL": {
        "ef_per_tkm": Decimal("0.03900000"),
        "wtt_per_tkm": Decimal("0.00800000"),
        "mode": "RAIL",
        "source": "GLEC_V3.0",
        "fuel_type": "MIXED",
        "unit": "kgCO2e/tkm",
    },
    "BULK_RAIL": {
        "ef_per_tkm": Decimal("0.03700000"),
        "wtt_per_tkm": Decimal("0.00800000"),
        "mode": "RAIL",
        "source": "GLEC_V3.0",
        "fuel_type": "MIXED",
        "unit": "kgCO2e/tkm",
    },
    # ---- MARITIME: Container ships ----
    "CONTAINER_FEEDER": {
        "ef_per_tkm": Decimal("0.13700000"),
        "wtt_per_tkm": Decimal("0.01600000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "MGO",
        "unit": "kgCO2e/tkm",
    },
    "CONTAINER_FEEDERMAX": {
        "ef_per_tkm": Decimal("0.09700000"),
        "wtt_per_tkm": Decimal("0.01100000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "CONTAINER_PANAMAX": {
        "ef_per_tkm": Decimal("0.05200000"),
        "wtt_per_tkm": Decimal("0.00600000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "CONTAINER_POST_PANAMAX": {
        "ef_per_tkm": Decimal("0.03500000"),
        "wtt_per_tkm": Decimal("0.00400000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "CONTAINER_ULCV": {
        "ef_per_tkm": Decimal("0.01800000"),
        "wtt_per_tkm": Decimal("0.00200000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    # ---- MARITIME: Bulk carriers ----
    "BULK_CARRIER_HANDYSIZE": {
        "ef_per_tkm": Decimal("0.04400000"),
        "wtt_per_tkm": Decimal("0.00500000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "BULK_CARRIER_HANDYMAX": {
        "ef_per_tkm": Decimal("0.03700000"),
        "wtt_per_tkm": Decimal("0.00400000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "BULK_CARRIER_PANAMAX": {
        "ef_per_tkm": Decimal("0.02900000"),
        "wtt_per_tkm": Decimal("0.00300000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "BULK_CARRIER_CAPESIZE": {
        "ef_per_tkm": Decimal("0.02100000"),
        "wtt_per_tkm": Decimal("0.00200000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    # ---- MARITIME: Tankers ----
    "TANKER_AFRAMAX": {
        "ef_per_tkm": Decimal("0.03300000"),
        "wtt_per_tkm": Decimal("0.00400000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "TANKER_SUEZMAX": {
        "ef_per_tkm": Decimal("0.02600000"),
        "wtt_per_tkm": Decimal("0.00300000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    "TANKER_VLCC": {
        "ef_per_tkm": Decimal("0.01900000"),
        "wtt_per_tkm": Decimal("0.00200000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "HFO",
        "unit": "kgCO2e/tkm",
    },
    # ---- MARITIME: Other vessels ----
    "RORO": {
        "ef_per_tkm": Decimal("0.10900000"),
        "wtt_per_tkm": Decimal("0.01300000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "MGO",
        "unit": "kgCO2e/tkm",
    },
    "GENERAL_CARGO": {
        "ef_per_tkm": Decimal("0.12600000"),
        "wtt_per_tkm": Decimal("0.01500000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "MGO",
        "unit": "kgCO2e/tkm",
    },
    "REEFER_VESSEL": {
        "ef_per_tkm": Decimal("0.15400000"),
        "wtt_per_tkm": Decimal("0.01800000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "MGO",
        "unit": "kgCO2e/tkm",
    },
    "FERRY": {
        "ef_per_tkm": Decimal("0.17100000"),
        "wtt_per_tkm": Decimal("0.02000000"),
        "mode": "MARITIME",
        "source": "IMO_4TH_GHG",
        "fuel_type": "MGO",
        "unit": "kgCO2e/tkm",
    },
    # ---- AIR ----
    "SHORT_HAUL_NARROW": {
        "ef_per_tkm": Decimal("1.42800000"),
        "wtt_per_tkm": Decimal("0.17500000"),
        "mode": "AIR",
        "source": "ICAO_V12",
        "fuel_type": "JET_FUEL",
        "unit": "kgCO2e/tkm",
    },
    "MEDIUM_HAUL_NARROW": {
        "ef_per_tkm": Decimal("0.97100000"),
        "wtt_per_tkm": Decimal("0.11900000"),
        "mode": "AIR",
        "source": "ICAO_V12",
        "fuel_type": "JET_FUEL",
        "unit": "kgCO2e/tkm",
    },
    "LONG_HAUL_WIDE": {
        "ef_per_tkm": Decimal("0.62800000"),
        "wtt_per_tkm": Decimal("0.07700000"),
        "mode": "AIR",
        "source": "ICAO_V12",
        "fuel_type": "JET_FUEL",
        "unit": "kgCO2e/tkm",
    },
    "FREIGHTER_NARROW": {
        "ef_per_tkm": Decimal("1.25700000"),
        "wtt_per_tkm": Decimal("0.15400000"),
        "mode": "AIR",
        "source": "ICAO_V12",
        "fuel_type": "JET_FUEL",
        "unit": "kgCO2e/tkm",
    },
    "FREIGHTER_WIDE": {
        "ef_per_tkm": Decimal("0.74200000"),
        "wtt_per_tkm": Decimal("0.09100000"),
        "mode": "AIR",
        "source": "ICAO_V12",
        "fuel_type": "JET_FUEL",
        "unit": "kgCO2e/tkm",
    },
    # ---- PIPELINE ----
    "NATURAL_GAS_PIPELINE": {
        "ef_per_tkm": Decimal("0.00500000"),
        "wtt_per_tkm": Decimal("0.00100000"),
        "mode": "PIPELINE",
        "source": "GLEC_V3.0",
        "fuel_type": "ELECTRICITY",
        "unit": "kgCO2e/tkm",
    },
    "CRUDE_OIL_PIPELINE": {
        "ef_per_tkm": Decimal("0.00300000"),
        "wtt_per_tkm": Decimal("0.00060000"),
        "mode": "PIPELINE",
        "source": "GLEC_V3.0",
        "fuel_type": "ELECTRICITY",
        "unit": "kgCO2e/tkm",
    },
    "REFINED_PRODUCTS_PIPELINE": {
        "ef_per_tkm": Decimal("0.00400000"),
        "wtt_per_tkm": Decimal("0.00080000"),
        "mode": "PIPELINE",
        "source": "GLEC_V3.0",
        "fuel_type": "ELECTRICITY",
        "unit": "kgCO2e/tkm",
    },
}


# ==============================================================================
# COLD CHAIN UPLIFT FACTORS
# ==============================================================================
# Multiplier applied to base transport emissions for temperature-controlled
# transport. Source: DEFRA 2024 Refrigerated Transport, GLEC v3.0.

COLD_CHAIN_UPLIFT_FACTORS: Dict[Tuple[str, str], Decimal] = {
    ("CHILLED", "ROAD"): Decimal("1.15000000"),
    ("CHILLED", "RAIL"): Decimal("1.12000000"),
    ("CHILLED", "MARITIME"): Decimal("1.20000000"),
    ("CHILLED", "AIR"): Decimal("1.05000000"),
    ("FROZEN", "ROAD"): Decimal("1.30000000"),
    ("FROZEN", "RAIL"): Decimal("1.25000000"),
    ("FROZEN", "MARITIME"): Decimal("1.40000000"),
    ("FROZEN", "AIR"): Decimal("1.10000000"),
}


# ==============================================================================
# WAREHOUSE EMISSION FACTORS (kgCO2e per m^2 per year)
# ==============================================================================
# Sources: DEFRA 2024, IEA, GLEC v3.0

WAREHOUSE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "AMBIENT_WAREHOUSE": {
        "electricity_ef": Decimal("12.50000000"),
        "gas_ef": Decimal("5.80000000"),
        "total_ef": Decimal("18.30000000"),
        "source_region": "GLOBAL",
    },
    "REFRIGERATED_WAREHOUSE": {
        "electricity_ef": Decimal("68.40000000"),
        "gas_ef": Decimal("8.20000000"),
        "total_ef": Decimal("76.60000000"),
        "source_region": "GLOBAL",
    },
    "DISTRIBUTION_CENTER": {
        "electricity_ef": Decimal("15.70000000"),
        "gas_ef": Decimal("6.60000000"),
        "total_ef": Decimal("22.30000000"),
        "source_region": "GLOBAL",
    },
    "CROSS_DOCK": {
        "electricity_ef": Decimal("5.90000000"),
        "gas_ef": Decimal("2.80000000"),
        "total_ef": Decimal("8.70000000"),
        "source_region": "GLOBAL",
    },
    "RETAIL_STORE": {
        "electricity_ef": Decimal("45.20000000"),
        "gas_ef": Decimal("12.30000000"),
        "total_ef": Decimal("57.50000000"),
        "source_region": "GLOBAL",
    },
    "FULFILLMENT_CENTER": {
        "electricity_ef": Decimal("25.40000000"),
        "gas_ef": Decimal("7.10000000"),
        "total_ef": Decimal("32.50000000"),
        "source_region": "GLOBAL",
    },
}


# ==============================================================================
# LAST-MILE DELIVERY FACTORS (kgCO2e per delivery)
# ==============================================================================
# Source: DEFRA 2024, GLEC v3.0, Academic literature

LAST_MILE_DELIVERY_FACTORS: Dict[Tuple[str, str], Decimal] = {
    # PARCEL_VAN
    ("PARCEL_VAN", "URBAN"): Decimal("0.21000000"),
    ("PARCEL_VAN", "SUBURBAN"): Decimal("0.35000000"),
    ("PARCEL_VAN", "RURAL"): Decimal("0.68000000"),
    # CARGO_BIKE
    ("CARGO_BIKE", "URBAN"): Decimal("0.00500000"),
    ("CARGO_BIKE", "SUBURBAN"): Decimal("0.01200000"),
    ("CARGO_BIKE", "RURAL"): Decimal("0.02500000"),
    # ELECTRIC_VAN
    ("ELECTRIC_VAN", "URBAN"): Decimal("0.04800000"),
    ("ELECTRIC_VAN", "SUBURBAN"): Decimal("0.07500000"),
    ("ELECTRIC_VAN", "RURAL"): Decimal("0.14500000"),
    # DRONE
    ("DRONE", "URBAN"): Decimal("0.01500000"),
    ("DRONE", "SUBURBAN"): Decimal("0.02200000"),
    ("DRONE", "RURAL"): Decimal("0.03800000"),
    # LOCKER_PICKUP
    ("LOCKER_PICKUP", "URBAN"): Decimal("0.09500000"),
    ("LOCKER_PICKUP", "SUBURBAN"): Decimal("0.15000000"),
    ("LOCKER_PICKUP", "RURAL"): Decimal("0.28000000"),
    # CLICK_AND_COLLECT
    ("CLICK_AND_COLLECT", "URBAN"): Decimal("0.08000000"),
    ("CLICK_AND_COLLECT", "SUBURBAN"): Decimal("0.12500000"),
    ("CLICK_AND_COLLECT", "RURAL"): Decimal("0.22000000"),
    # STANDARD_TRUCK
    ("STANDARD_TRUCK", "URBAN"): Decimal("0.45000000"),
    ("STANDARD_TRUCK", "SUBURBAN"): Decimal("0.62000000"),
    ("STANDARD_TRUCK", "RURAL"): Decimal("1.15000000"),
    # HEAVY_GOODS
    ("HEAVY_GOODS", "URBAN"): Decimal("1.25000000"),
    ("HEAVY_GOODS", "SUBURBAN"): Decimal("1.65000000"),
    ("HEAVY_GOODS", "RURAL"): Decimal("2.80000000"),
}


# ==============================================================================
# EEIO FACTORS (kgCO2e per $1,000 USD, base year 2021)
# ==============================================================================
# Source: US EEIO 2.0 model, EPA

EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "484110": {"sector": "General Freight Trucking, Local", "ef_per_usd": Decimal("523.40000000")},
    "484121": {"sector": "General Freight Trucking, TL Long-Distance", "ef_per_usd": Decimal("489.70000000")},
    "484122": {"sector": "General Freight Trucking, LTL", "ef_per_usd": Decimal("512.30000000")},
    "484220": {"sector": "Specialized Freight Trucking, Local", "ef_per_usd": Decimal("534.60000000")},
    "484230": {"sector": "Specialized Freight Trucking, Long-Distance", "ef_per_usd": Decimal("501.20000000")},
    "482111": {"sector": "Rail Transportation", "ef_per_usd": Decimal("178.50000000")},
    "483111": {"sector": "Deep Sea Freight Transportation", "ef_per_usd": Decimal("89.40000000")},
    "483113": {"sector": "Coastal and Great Lakes Freight", "ef_per_usd": Decimal("102.70000000")},
    "483211": {"sector": "Inland Water Freight", "ef_per_usd": Decimal("95.30000000")},
    "481112": {"sector": "Scheduled Freight Air Transportation", "ef_per_usd": Decimal("1456.80000000")},
    "481212": {"sector": "Nonscheduled Chartered Freight Air", "ef_per_usd": Decimal("1523.40000000")},
    "488510": {"sector": "Freight Transportation Arrangement", "ef_per_usd": Decimal("312.40000000")},
    "493110": {"sector": "General Warehousing and Storage", "ef_per_usd": Decimal("167.30000000")},
    "493120": {"sector": "Refrigerated Warehousing and Storage", "ef_per_usd": Decimal("245.80000000")},
    "454110": {"sector": "Electronic Shopping and Mail-Order", "ef_per_usd": Decimal("278.50000000")},
    "492110": {"sector": "Couriers and Express Delivery", "ef_per_usd": Decimal("635.20000000")},
    "452210": {"sector": "Department Stores", "ef_per_usd": Decimal("195.60000000")},
    "445110": {"sector": "Supermarkets and Grocery Stores", "ef_per_usd": Decimal("210.40000000")},
}


# ==============================================================================
# CURRENCY CONVERSION RATES (to USD, as of 2024-Q4)
# ==============================================================================

CURRENCY_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.00000000"),
    "EUR": Decimal("1.08500000"),
    "GBP": Decimal("1.26300000"),
    "JPY": Decimal("0.00660000"),
    "CNY": Decimal("0.13800000"),
    "INR": Decimal("0.01200000"),
    "CAD": Decimal("0.74200000"),
    "AUD": Decimal("0.65100000"),
    "CHF": Decimal("1.12400000"),
    "BRL": Decimal("0.20100000"),
    "KRW": Decimal("0.00075000"),
    "MXN": Decimal("0.05800000"),
}


# ==============================================================================
# CPI DEFLATORS (base year 2021 = 1.0)
# ==============================================================================
# Source: World Bank, US Bureau of Labor Statistics

CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.87400000"),
    2016: Decimal("0.88500000"),
    2017: Decimal("0.90400000"),
    2018: Decimal("0.92600000"),
    2019: Decimal("0.94400000"),
    2020: Decimal("0.95600000"),
    2021: Decimal("1.00000000"),
    2022: Decimal("1.08000000"),
    2023: Decimal("1.11200000"),
    2024: Decimal("1.14500000"),
    2025: Decimal("1.17300000"),
}


# ==============================================================================
# GRID EMISSION FACTORS (kgCO2e per kWh, 2024)
# ==============================================================================
# Source: IEA 2024, eGRID 2023 (US), EEA (EU)

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.38600000"),
    "GB": Decimal("0.20700000"),
    "DE": Decimal("0.35000000"),
    "FR": Decimal("0.05100000"),
    "IT": Decimal("0.25700000"),
    "ES": Decimal("0.16200000"),
    "NL": Decimal("0.33800000"),
    "BE": Decimal("0.16700000"),
    "SE": Decimal("0.01200000"),
    "NO": Decimal("0.01100000"),
    "DK": Decimal("0.11400000"),
    "PL": Decimal("0.63500000"),
    "JP": Decimal("0.45700000"),
    "CN": Decimal("0.55500000"),
    "IN": Decimal("0.71000000"),
    "KR": Decimal("0.41500000"),
    "AU": Decimal("0.65600000"),
    "CA": Decimal("0.12000000"),
    "BR": Decimal("0.07400000"),
    "MX": Decimal("0.43100000"),
    "ZA": Decimal("0.92800000"),
    "RU": Decimal("0.33400000"),
    "ID": Decimal("0.72000000"),
    "TH": Decimal("0.44600000"),
    "VN": Decimal("0.52000000"),
    "PH": Decimal("0.60000000"),
    "MY": Decimal("0.56000000"),
    "SG": Decimal("0.40800000"),
    "NZ": Decimal("0.09400000"),
    "GLOBAL": Decimal("0.43600000"),
}


# ==============================================================================
# DISTRIBUTION CHANNEL DEFAULTS
# ==============================================================================
# Default parameters by channel for average-data calculations when specific
# transport details are unavailable.

DISTRIBUTION_CHANNEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "DIRECT_TO_CONSUMER": {
        "avg_distance_km": Decimal("150.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "LCV_DIESEL",
        "avg_weight_tonnes": Decimal("0.01000000"),
        "last_mile_type": "PARCEL_VAN",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 1,
        "description": "Direct-to-consumer via parcel delivery",
    },
    "WHOLESALE": {
        "avg_distance_km": Decimal("500.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "ARTICULATED_40_44T",
        "avg_weight_tonnes": Decimal("10.00000000"),
        "last_mile_type": "HEAVY_GOODS",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 2,
        "description": "Wholesale distribution via HGV",
    },
    "RETAIL_BRICK_MORTAR": {
        "avg_distance_km": Decimal("350.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "RIGID_17_26T",
        "avg_weight_tonnes": Decimal("5.00000000"),
        "last_mile_type": "STANDARD_TRUCK",
        "last_mile_area": "URBAN",
        "warehouse_stops": 2,
        "description": "Retail store distribution via rigid HGV",
    },
    "ECOMMERCE": {
        "avg_distance_km": Decimal("250.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "LCV_DIESEL",
        "avg_weight_tonnes": Decimal("0.00500000"),
        "last_mile_type": "PARCEL_VAN",
        "last_mile_area": "URBAN",
        "warehouse_stops": 1,
        "description": "E-commerce fulfillment with last-mile parcel",
    },
    "THIRD_PARTY_LOGISTICS": {
        "avg_distance_km": Decimal("600.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "ARTICULATED_40_44T",
        "avg_weight_tonnes": Decimal("15.00000000"),
        "last_mile_type": "STANDARD_TRUCK",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 3,
        "description": "3PL managed distribution network",
    },
    "FRANCHISE": {
        "avg_distance_km": Decimal("400.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "RIGID_7_5_17T",
        "avg_weight_tonnes": Decimal("3.00000000"),
        "last_mile_type": "STANDARD_TRUCK",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 2,
        "description": "Franchise distribution via regional DC",
    },
    "DISTRIBUTOR": {
        "avg_distance_km": Decimal("550.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "ARTICULATED_33T",
        "avg_weight_tonnes": Decimal("8.00000000"),
        "last_mile_type": "STANDARD_TRUCK",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 2,
        "description": "Third-party distributor network",
    },
    "DROP_SHIP": {
        "avg_distance_km": Decimal("300.00000000"),
        "avg_mode": "ROAD",
        "avg_vehicle_type": "LCV_DIESEL",
        "avg_weight_tonnes": Decimal("0.02000000"),
        "last_mile_type": "PARCEL_VAN",
        "last_mile_area": "SUBURBAN",
        "warehouse_stops": 0,
        "description": "Drop-ship from supplier directly to consumer",
    },
}


# ==============================================================================
# INCOTERM CLASSIFICATIONS
# ==============================================================================
# Defines whether downstream transport falls into Cat 4 (upstream) or Cat 9
# (downstream) based on the Incoterm used, and where physical transfer occurs.
# Source: GHG Protocol Scope 3 Standard, ICC Incoterms 2020.

INCOTERM_CLASSIFICATIONS: Dict[str, Dict[str, Any]] = {
    "EXW": {
        "cat4_scope": True,
        "cat9_scope": False,
        "transfer_point": "seller_premises",
        "seller_responsibility": "none",
        "buyer_responsibility": "all_transport",
        "description": "Ex Works: buyer arranges all transport",
    },
    "FCA": {
        "cat4_scope": True,
        "cat9_scope": False,
        "transfer_point": "named_place",
        "seller_responsibility": "to_carrier",
        "buyer_responsibility": "main_transport",
        "description": "Free Carrier: seller delivers to carrier",
    },
    "FAS": {
        "cat4_scope": True,
        "cat9_scope": False,
        "transfer_point": "alongside_ship",
        "seller_responsibility": "to_port",
        "buyer_responsibility": "loading_and_main",
        "description": "Free Alongside Ship: maritime only",
    },
    "FOB": {
        "cat4_scope": True,
        "cat9_scope": False,
        "transfer_point": "on_board_ship",
        "seller_responsibility": "to_ship",
        "buyer_responsibility": "main_transport",
        "description": "Free On Board: risk transfers at ship rail",
    },
    "CFR": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_port",
        "seller_responsibility": "main_transport",
        "buyer_responsibility": "insurance_and_inland",
        "description": "Cost and Freight: seller pays main freight",
    },
    "CIF": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_port",
        "seller_responsibility": "main_transport_insurance",
        "buyer_responsibility": "inland_from_port",
        "description": "Cost Insurance Freight: seller pays freight+insurance",
    },
    "CPT": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "named_destination",
        "seller_responsibility": "to_destination",
        "buyer_responsibility": "unloading",
        "description": "Carriage Paid To: seller pays carriage",
    },
    "CIP": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "named_destination",
        "seller_responsibility": "carriage_and_insurance",
        "buyer_responsibility": "unloading",
        "description": "Carriage and Insurance Paid To",
    },
    "DAP": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_unloaded",
        "seller_responsibility": "to_destination",
        "buyer_responsibility": "unloading_duties",
        "description": "Delivered At Place: seller delivers to destination",
    },
    "DPU": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_unloaded",
        "seller_responsibility": "to_destination_unloaded",
        "buyer_responsibility": "duties",
        "description": "Delivered at Place Unloaded",
    },
    "DDP": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_cleared",
        "seller_responsibility": "all_including_duties",
        "buyer_responsibility": "none",
        "description": "Delivered Duty Paid: seller bears all costs",
    },
    "DDU": {
        "cat4_scope": False,
        "cat9_scope": True,
        "transfer_point": "destination_uncleared",
        "seller_responsibility": "all_except_duties",
        "buyer_responsibility": "duties_only",
        "description": "Delivered Duty Unpaid (legacy term)",
    },
    "UNKNOWN": {
        "cat4_scope": True,
        "cat9_scope": True,
        "transfer_point": "unknown",
        "seller_responsibility": "unknown",
        "buyer_responsibility": "unknown",
        "description": "Unknown Incoterm: include in both Cat 4 and Cat 9",
    },
}


# ==============================================================================
# LOAD FACTOR ADJUSTMENTS
# ==============================================================================
# Adjustment to EF based on actual load factor vs. assumed average load.
# If actual load factor is lower than average, EF per tkm increases.
# Formula: adjusted_ef = base_ef * (default_load_factor / actual_load_factor)

LOAD_FACTOR_DEFAULTS: Dict[str, Decimal] = {
    "ROAD": Decimal("0.65000000"),
    "RAIL": Decimal("0.70000000"),
    "MARITIME": Decimal("0.75000000"),
    "AIR": Decimal("0.68000000"),
    "PIPELINE": Decimal("0.90000000"),
}


# ==============================================================================
# RETURN TRIP MULTIPLIERS
# ==============================================================================
# Multiplier applied to transport emissions for the return trip of the vehicle.

RETURN_TRIP_MULTIPLIERS: Dict[str, Decimal] = {
    "EMPTY": Decimal("0.35000000"),
    "PARTIAL_LOAD": Decimal("0.20000000"),
    "FULL_LOAD": Decimal("0.00000000"),
    "NO_RETURN": Decimal("0.00000000"),
}


# ==============================================================================
# DATA QUALITY INDICATOR (DQI) SCORING
# ==============================================================================
# 5-dimension DQI per ISO 14083 and GLEC Framework
# Score range: 1 (best, primary data) to 5 (worst, estimated)

DQI_SCORING: Dict[str, Dict[str, Any]] = {
    "technological_representativeness": {
        "description": "How well does EF technology match actual technology?",
        "score_1": "Exact technology match (supplier-specific)",
        "score_2": "Same technology class",
        "score_3": "Similar technology class",
        "score_4": "Different technology, same mode",
        "score_5": "Generic/unknown technology",
    },
    "geographical_representativeness": {
        "description": "How well does EF geography match actual geography?",
        "score_1": "Exact region/country",
        "score_2": "Same continent/region",
        "score_3": "Same climate zone",
        "score_4": "Different region, similar development",
        "score_5": "Global average",
    },
    "temporal_representativeness": {
        "description": "How recent is the EF data?",
        "score_1": "Same year as activity",
        "score_2": "Within 2 years",
        "score_3": "Within 5 years",
        "score_4": "Within 10 years",
        "score_5": "Older than 10 years",
    },
    "completeness": {
        "description": "How complete is the activity data?",
        "score_1": "100% actual data",
        "score_2": ">80% actual data",
        "score_3": ">50% actual data",
        "score_4": ">20% actual data",
        "score_5": "<20% actual, mostly estimated",
    },
    "reliability": {
        "description": "What is the source reliability?",
        "score_1": "Verified primary data",
        "score_2": "Non-verified primary data",
        "score_3": "Published secondary data (DEFRA/GLEC)",
        "score_4": "Industry estimates",
        "score_5": "Unqualified estimate",
    },
}


# ==============================================================================
# UNCERTAINTY RANGES BY METHOD
# ==============================================================================
# Percentage uncertainty ranges (low and high) for each calculation method.
# Source: GHG Protocol uncertainty guidance, GLEC Framework.

UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    "DISTANCE_BASED": {
        "low_pct": Decimal("-15.00000000"),
        "high_pct": Decimal("20.00000000"),
    },
    "SPEND_BASED": {
        "low_pct": Decimal("-40.00000000"),
        "high_pct": Decimal("60.00000000"),
    },
    "AVERAGE_DATA": {
        "low_pct": Decimal("-30.00000000"),
        "high_pct": Decimal("50.00000000"),
    },
    "SUPPLIER_SPECIFIC": {
        "low_pct": Decimal("-5.00000000"),
        "high_pct": Decimal("10.00000000"),
    },
}


# ==============================================================================
# MODE-TO-VEHICLE-TYPE MAPPING (default vehicle types by mode)
# ==============================================================================

MODE_DEFAULT_VEHICLE_TYPES: Dict[str, List[str]] = {
    "ROAD": [
        "LCV_DIESEL", "LCV_ELECTRIC",
        "RIGID_3_5_7_5T", "RIGID_7_5_17T", "RIGID_17_26T",
        "RIGID_26_28T", "RIGID_28_32T",
        "ARTICULATED_33T", "ARTICULATED_40_44T", "ROAD_TRAIN",
        "TRUCK_CNG", "TRUCK_LNG", "TRUCK_HYDROGEN",
    ],
    "RAIL": [
        "FREIGHT_ELECTRIC", "FREIGHT_DIESEL",
        "INTERMODAL_RAIL", "BULK_RAIL",
    ],
    "MARITIME": [
        "CONTAINER_FEEDER", "CONTAINER_FEEDERMAX",
        "CONTAINER_PANAMAX", "CONTAINER_POST_PANAMAX", "CONTAINER_ULCV",
        "BULK_CARRIER_HANDYSIZE", "BULK_CARRIER_HANDYMAX",
        "BULK_CARRIER_PANAMAX", "BULK_CARRIER_CAPESIZE",
        "TANKER_AFRAMAX", "TANKER_SUEZMAX", "TANKER_VLCC",
        "RORO", "GENERAL_CARGO", "REEFER_VESSEL", "FERRY",
    ],
    "AIR": [
        "SHORT_HAUL_NARROW", "MEDIUM_HAUL_NARROW", "LONG_HAUL_WIDE",
        "FREIGHTER_NARROW", "FREIGHTER_WIDE",
    ],
    "PIPELINE": [
        "NATURAL_GAS_PIPELINE", "CRUDE_OIL_PIPELINE",
        "REFINED_PRODUCTS_PIPELINE",
    ],
}


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class DownstreamTransportDatabaseEngine:
    """
    Thread-safe singleton engine for downstream transport emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all downstream
    transport modes, cold chain, warehousing, last-mile delivery, EEIO spend-based,
    currency, CPI deflation, Incoterms, distribution channels, and DQI scoring.

    This engine does NOT perform any LLM calls. All factors are retrieved from
    validated, frozen constant tables embedded in this module.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure only
        one instance is created across all threads. All public methods are
        protected by an instance-level reentrant lock.

    Attributes:
        _lookup_count: Total number of factor lookups performed.

    Example:
        >>> db = get_downstream_transport_database()
        >>> ef = db.get_transport_ef("ARTICULATED_40_44T")
        >>> print(ef["ef_per_tkm"])
        0.10300000
        >>> modes = db.get_all_modes()
        >>> print(modes)
        ['ROAD', 'RAIL', 'MARITIME', 'AIR', 'PIPELINE']
    """

    _instance: Optional["DownstreamTransportDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DownstreamTransportDatabaseEngine":
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
        self._instance_lock: threading.RLock = threading.RLock()

        logger.info(
            "DownstreamTransportDatabaseEngine initialized: "
            "transport_efs=%d, cold_chain=%d, warehouses=%d, "
            "last_mile=%d, eeio=%d, currencies=%d, cpi_years=%d, "
            "grid_efs=%d, channels=%d, incoterms=%d",
            len(TRANSPORT_EMISSION_FACTORS),
            len(COLD_CHAIN_UPLIFT_FACTORS),
            len(WAREHOUSE_EMISSION_FACTORS),
            len(LAST_MILE_DELIVERY_FACTORS),
            len(EEIO_FACTORS),
            len(CURRENCY_RATES),
            len(CPI_DEFLATORS),
            len(GRID_EMISSION_FACTORS),
            len(DISTRIBUTION_CHANNEL_DEFAULTS),
            len(INCOTERM_CLASSIFICATIONS),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        self._lookup_count += 1

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize to 8 decimal places with ROUND_HALF_UP."""
        return value.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

    # =========================================================================
    # 1. get_transport_ef
    # =========================================================================

    def get_transport_ef(self, vehicle_type: str) -> Dict[str, Any]:
        """
        Get transport emission factor for a specific vehicle/vessel type.

        Args:
            vehicle_type: Vehicle or vessel type key (e.g., "ARTICULATED_40_44T",
                         "CONTAINER_POST_PANAMAX", "FREIGHTER_WIDE").

        Returns:
            Dict with keys: ef_per_tkm, wtt_per_tkm, mode, source, fuel_type, unit.

        Raises:
            ValueError: If vehicle_type is not found in the database.

        Example:
            >>> ef = db.get_transport_ef("ARTICULATED_40_44T")
            >>> ef["ef_per_tkm"]
            Decimal('0.10300000')
        """
        with self._instance_lock:
            self._increment_lookup()

            if vehicle_type not in TRANSPORT_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown vehicle type: '{vehicle_type}'. "
                    f"Available types: {sorted(TRANSPORT_EMISSION_FACTORS.keys())}"
                )

            factor = TRANSPORT_EMISSION_FACTORS[vehicle_type].copy()

            logger.debug(
                "get_transport_ef: %s -> ef_per_tkm=%s, wtt=%s, mode=%s",
                vehicle_type,
                factor["ef_per_tkm"],
                factor["wtt_per_tkm"],
                factor["mode"],
            )

            return factor

    # =========================================================================
    # 2. get_transport_efs_by_mode
    # =========================================================================

    def get_transport_efs_by_mode(self, mode: str) -> List[Dict[str, Any]]:
        """
        Get all transport emission factors for a specific transport mode.

        Args:
            mode: Transport mode ("ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE").

        Returns:
            List of dicts, each containing vehicle_type and its EF data.

        Raises:
            ValueError: If mode is not recognized.

        Example:
            >>> rail_efs = db.get_transport_efs_by_mode("RAIL")
            >>> len(rail_efs)
            4
        """
        with self._instance_lock:
            self._increment_lookup()

            mode_upper = mode.upper()
            valid_modes = {"ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE"}
            if mode_upper not in valid_modes:
                raise ValueError(
                    f"Unknown mode: '{mode}'. Must be one of {sorted(valid_modes)}"
                )

            results: List[Dict[str, Any]] = []
            for vtype, data in TRANSPORT_EMISSION_FACTORS.items():
                if data["mode"] == mode_upper:
                    entry = data.copy()
                    entry["vehicle_type"] = vtype
                    results.append(entry)

            logger.debug(
                "get_transport_efs_by_mode: %s -> %d factors",
                mode_upper, len(results),
            )

            return results

    # =========================================================================
    # 3. get_cold_chain_uplift
    # =========================================================================

    def get_cold_chain_uplift(
        self, temperature_regime: str, mode: str
    ) -> Decimal:
        """
        Get cold chain uplift factor for temperature-controlled transport.

        Args:
            temperature_regime: "AMBIENT", "CHILLED", or "FROZEN".
            mode: Transport mode ("ROAD", "RAIL", "MARITIME", "AIR").

        Returns:
            Uplift multiplier (Decimal). Returns 1.0 for AMBIENT.

        Raises:
            ValueError: If temperature_regime is not recognized.

        Example:
            >>> db.get_cold_chain_uplift("FROZEN", "ROAD")
            Decimal('1.30000000')
            >>> db.get_cold_chain_uplift("AMBIENT", "ROAD")
            Decimal('1.00000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            temp_upper = temperature_regime.upper()
            mode_upper = mode.upper()

            if temp_upper not in {"AMBIENT", "CHILLED", "FROZEN"}:
                raise ValueError(
                    f"Unknown temperature regime: '{temperature_regime}'. "
                    f"Must be AMBIENT, CHILLED, or FROZEN."
                )

            if temp_upper == "AMBIENT":
                return self._quantize(_ONE)

            lookup_key = (temp_upper, mode_upper)
            factor = COLD_CHAIN_UPLIFT_FACTORS.get(lookup_key)

            if factor is None:
                logger.warning(
                    "No cold chain uplift for (%s, %s), returning 1.0",
                    temp_upper, mode_upper,
                )
                return self._quantize(_ONE)

            logger.debug(
                "get_cold_chain_uplift: %s/%s -> %s",
                temp_upper, mode_upper, factor,
            )

            return factor

    # =========================================================================
    # 4. get_warehouse_ef
    # =========================================================================

    def get_warehouse_ef(self, warehouse_type: str) -> Dict[str, Decimal]:
        """
        Get warehouse/DC emission factor.

        Args:
            warehouse_type: Warehouse type (e.g., "AMBIENT_WAREHOUSE",
                           "REFRIGERATED_WAREHOUSE", "DISTRIBUTION_CENTER").

        Returns:
            Dict with keys: electricity_ef, gas_ef, total_ef (kgCO2e/m2/year).

        Raises:
            ValueError: If warehouse_type is not found.

        Example:
            >>> wf = db.get_warehouse_ef("DISTRIBUTION_CENTER")
            >>> wf["total_ef"]
            Decimal('22.30000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            wtype_upper = warehouse_type.upper()
            if wtype_upper not in WAREHOUSE_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown warehouse type: '{warehouse_type}'. "
                    f"Available: {sorted(WAREHOUSE_EMISSION_FACTORS.keys())}"
                )

            factor = WAREHOUSE_EMISSION_FACTORS[wtype_upper].copy()

            logger.debug(
                "get_warehouse_ef: %s -> total=%s kgCO2e/m2/year",
                wtype_upper, factor["total_ef"],
            )

            return factor

    # =========================================================================
    # 5. get_last_mile_ef
    # =========================================================================

    def get_last_mile_ef(self, delivery_type: str, area: str) -> Decimal:
        """
        Get last-mile delivery emission factor.

        Args:
            delivery_type: Delivery type (e.g., "PARCEL_VAN", "CARGO_BIKE").
            area: Delivery area ("URBAN", "SUBURBAN", "RURAL").

        Returns:
            Emission factor in kgCO2e per delivery.

        Raises:
            ValueError: If delivery_type or area is not found.

        Example:
            >>> db.get_last_mile_ef("PARCEL_VAN", "URBAN")
            Decimal('0.21000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            dtype_upper = delivery_type.upper()
            area_upper = area.upper()

            lookup_key = (dtype_upper, area_upper)
            factor = LAST_MILE_DELIVERY_FACTORS.get(lookup_key)

            if factor is None:
                raise ValueError(
                    f"No last-mile EF for delivery_type='{delivery_type}', "
                    f"area='{area}'. Available delivery types: "
                    f"{sorted(set(k[0] for k in LAST_MILE_DELIVERY_FACTORS.keys()))}. "
                    f"Available areas: URBAN, SUBURBAN, RURAL."
                )

            logger.debug(
                "get_last_mile_ef: %s/%s -> %s kgCO2e/delivery",
                dtype_upper, area_upper, factor,
            )

            return factor

    # =========================================================================
    # 6. get_eeio_factor
    # =========================================================================

    def get_eeio_factor(self, naics_code: str) -> Dict[str, Any]:
        """
        Get EEIO emission factor for a NAICS sector code.

        Args:
            naics_code: 6-digit NAICS code (e.g., "484121").

        Returns:
            Dict with keys: sector (description), ef_per_usd (kgCO2e/$1000).

        Raises:
            ValueError: If naics_code is not found.

        Example:
            >>> eeio = db.get_eeio_factor("484121")
            >>> eeio["sector"]
            'General Freight Trucking, TL Long-Distance'
            >>> eeio["ef_per_usd"]
            Decimal('489.70000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            if naics_code not in EEIO_FACTORS:
                raise ValueError(
                    f"Unknown NAICS code: '{naics_code}'. "
                    f"Available: {sorted(EEIO_FACTORS.keys())}"
                )

            factor = EEIO_FACTORS[naics_code].copy()

            logger.debug(
                "get_eeio_factor: %s -> %s, %s kgCO2e/$1000",
                naics_code, factor["sector"], factor["ef_per_usd"],
            )

            return factor

    # =========================================================================
    # 7. get_currency_rate
    # =========================================================================

    def get_currency_rate(self, currency: str) -> Decimal:
        """
        Get currency conversion rate to USD.

        Args:
            currency: 3-letter currency code (e.g., "EUR", "GBP", "JPY").

        Returns:
            Conversion rate (multiply by this to get USD).

        Raises:
            ValueError: If currency is not found.

        Example:
            >>> db.get_currency_rate("EUR")
            Decimal('1.08500000')
        """
        with self._instance_lock:
            self._increment_lookup()

            currency_upper = currency.upper()
            if currency_upper not in CURRENCY_RATES:
                raise ValueError(
                    f"Unknown currency: '{currency}'. "
                    f"Available: {sorted(CURRENCY_RATES.keys())}"
                )

            rate = CURRENCY_RATES[currency_upper]

            logger.debug(
                "get_currency_rate: %s -> %s USD",
                currency_upper, rate,
            )

            return rate

    # =========================================================================
    # 8. get_cpi_deflator
    # =========================================================================

    def get_cpi_deflator(self, year: int) -> Decimal:
        """
        Get CPI deflator for a given year (base year 2021 = 1.0).

        Used to deflate spend data to the EEIO base year before applying
        emission factors. Formula: deflated_spend = spend / cpi_deflator.

        Args:
            year: Calendar year (2015-2025).

        Returns:
            CPI deflator value.

        Raises:
            ValueError: If year is outside supported range.

        Example:
            >>> db.get_cpi_deflator(2023)
            Decimal('1.11200000')
        """
        with self._instance_lock:
            self._increment_lookup()

            if year not in CPI_DEFLATORS:
                raise ValueError(
                    f"No CPI deflator for year {year}. "
                    f"Supported range: {min(CPI_DEFLATORS.keys())}-{max(CPI_DEFLATORS.keys())}"
                )

            deflator = CPI_DEFLATORS[year]

            logger.debug("get_cpi_deflator: %d -> %s", year, deflator)

            return deflator

    # =========================================================================
    # 9. get_grid_ef
    # =========================================================================

    def get_grid_ef(self, country: str) -> Decimal:
        """
        Get electricity grid emission factor for a country.

        Args:
            country: 2-letter ISO country code (e.g., "US", "GB") or "GLOBAL".

        Returns:
            Grid emission factor in kgCO2e per kWh.

        Raises:
            ValueError: If country code is not found.

        Example:
            >>> db.get_grid_ef("US")
            Decimal('0.38600000')
        """
        with self._instance_lock:
            self._increment_lookup()

            country_upper = country.upper()
            if country_upper not in GRID_EMISSION_FACTORS:
                raise ValueError(
                    f"Unknown country: '{country}'. "
                    f"Available: {sorted(GRID_EMISSION_FACTORS.keys())}"
                )

            factor = GRID_EMISSION_FACTORS[country_upper]

            logger.debug(
                "get_grid_ef: %s -> %s kgCO2e/kWh",
                country_upper, factor,
            )

            return factor

    # =========================================================================
    # 10. get_distribution_channel_defaults
    # =========================================================================

    def get_distribution_channel_defaults(
        self, channel: str
    ) -> Dict[str, Any]:
        """
        Get default distribution parameters for a channel.

        Args:
            channel: Distribution channel (e.g., "DIRECT_TO_CONSUMER",
                    "WHOLESALE", "ECOMMERCE").

        Returns:
            Dict with avg_distance_km, avg_mode, avg_vehicle_type,
            avg_weight_tonnes, last_mile_type, last_mile_area,
            warehouse_stops, description.

        Raises:
            ValueError: If channel is not found.

        Example:
            >>> ch = db.get_distribution_channel_defaults("ECOMMERCE")
            >>> ch["avg_distance_km"]
            Decimal('250.00000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            channel_upper = channel.upper()
            if channel_upper not in DISTRIBUTION_CHANNEL_DEFAULTS:
                raise ValueError(
                    f"Unknown channel: '{channel}'. "
                    f"Available: {sorted(DISTRIBUTION_CHANNEL_DEFAULTS.keys())}"
                )

            defaults = DISTRIBUTION_CHANNEL_DEFAULTS[channel_upper].copy()

            logger.debug(
                "get_distribution_channel_defaults: %s -> distance=%s km, mode=%s",
                channel_upper,
                defaults["avg_distance_km"],
                defaults["avg_mode"],
            )

            return defaults

    # =========================================================================
    # 11. get_incoterm_classification
    # =========================================================================

    def get_incoterm_classification(self, incoterm: str) -> Dict[str, Any]:
        """
        Get Incoterm classification for Cat 4 vs Cat 9 scope determination.

        Args:
            incoterm: Incoterm code (e.g., "FOB", "CIF", "DDP", "EXW").

        Returns:
            Dict with cat4_scope, cat9_scope, transfer_point,
            seller_responsibility, buyer_responsibility, description.

        Raises:
            ValueError: If incoterm is not found.

        Example:
            >>> inc = db.get_incoterm_classification("DDP")
            >>> inc["cat9_scope"]
            True
        """
        with self._instance_lock:
            self._increment_lookup()

            incoterm_upper = incoterm.upper()
            if incoterm_upper not in INCOTERM_CLASSIFICATIONS:
                raise ValueError(
                    f"Unknown Incoterm: '{incoterm}'. "
                    f"Available: {sorted(INCOTERM_CLASSIFICATIONS.keys())}"
                )

            classification = INCOTERM_CLASSIFICATIONS[incoterm_upper].copy()

            logger.debug(
                "get_incoterm_classification: %s -> cat4=%s, cat9=%s, transfer=%s",
                incoterm_upper,
                classification["cat4_scope"],
                classification["cat9_scope"],
                classification["transfer_point"],
            )

            return classification

    # =========================================================================
    # 12. get_load_factor_adjustment
    # =========================================================================

    def get_load_factor_adjustment(self, load_factor: float) -> Decimal:
        """
        Calculate load factor adjustment multiplier.

        When actual load factor is known but differs from the average assumed
        in the EF, this adjustment corrects the EF. A low load factor increases
        per-tkm emissions (vehicles still burn fuel when partially loaded).

        Formula: adjustment = default_load_factor / actual_load_factor

        Args:
            load_factor: Actual load factor (0.0 < load_factor <= 1.0).

        Returns:
            Adjustment multiplier (Decimal).

        Raises:
            ValueError: If load_factor is out of range.

        Example:
            >>> db.get_load_factor_adjustment(0.50)
            Decimal('1.30000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            lf = Decimal(str(load_factor))
            if lf <= _ZERO or lf > _ONE:
                raise ValueError(
                    f"load_factor must be > 0 and <= 1.0, got {load_factor}"
                )

            # Use ROAD default as a representative average
            default_lf = LOAD_FACTOR_DEFAULTS.get("ROAD", Decimal("0.65000000"))
            adjustment = self._quantize(default_lf / lf)

            logger.debug(
                "get_load_factor_adjustment: actual=%s, default=%s, adjustment=%s",
                lf, default_lf, adjustment,
            )

            return adjustment

    # =========================================================================
    # 13. get_return_factor
    # =========================================================================

    def get_return_factor(self, return_type: str) -> Decimal:
        """
        Get return trip emissions multiplier.

        The return trip factor represents the fraction of outbound transport
        emissions attributed to the return journey. Empty returns cost more
        per tkm (no revenue load), while full returns are accounted separately.

        Args:
            return_type: Return type ("EMPTY", "PARTIAL_LOAD", "FULL_LOAD",
                        "NO_RETURN").

        Returns:
            Return multiplier (Decimal). 0.0 means no return emissions added.

        Raises:
            ValueError: If return_type is not recognized.

        Example:
            >>> db.get_return_factor("EMPTY")
            Decimal('0.35000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            rtype_upper = return_type.upper()
            if rtype_upper not in RETURN_TRIP_MULTIPLIERS:
                raise ValueError(
                    f"Unknown return type: '{return_type}'. "
                    f"Available: {sorted(RETURN_TRIP_MULTIPLIERS.keys())}"
                )

            factor = RETURN_TRIP_MULTIPLIERS[rtype_upper]

            logger.debug(
                "get_return_factor: %s -> %s", rtype_upper, factor,
            )

            return factor

    # =========================================================================
    # 14. get_dqi_scoring
    # =========================================================================

    def get_dqi_scoring(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the 5-dimension Data Quality Indicator scoring framework.

        Returns:
            Dict of 5 DQI dimensions, each containing description and
            score level descriptions (score_1 through score_5).

        Example:
            >>> dqi = db.get_dqi_scoring()
            >>> list(dqi.keys())
            ['technological_representativeness', 'geographical_representativeness',
             'temporal_representativeness', 'completeness', 'reliability']
        """
        with self._instance_lock:
            self._increment_lookup()

            # Deep copy to prevent mutation
            result: Dict[str, Dict[str, Any]] = {}
            for dim, data in DQI_SCORING.items():
                result[dim] = data.copy()

            logger.debug("get_dqi_scoring: returning %d dimensions", len(result))

            return result

    # =========================================================================
    # 15. get_uncertainty_range
    # =========================================================================

    def get_uncertainty_range(self, method: str) -> Dict[str, Decimal]:
        """
        Get uncertainty range for a calculation method.

        Args:
            method: Calculation method ("DISTANCE_BASED", "SPEND_BASED",
                   "AVERAGE_DATA", "SUPPLIER_SPECIFIC").

        Returns:
            Dict with keys: low_pct, high_pct (percentage bounds).

        Raises:
            ValueError: If method is not recognized.

        Example:
            >>> rng = db.get_uncertainty_range("DISTANCE_BASED")
            >>> rng["low_pct"]
            Decimal('-15.00000000')
            >>> rng["high_pct"]
            Decimal('20.00000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            method_upper = method.upper()
            if method_upper not in UNCERTAINTY_RANGES:
                raise ValueError(
                    f"Unknown method: '{method}'. "
                    f"Available: {sorted(UNCERTAINTY_RANGES.keys())}"
                )

            result = UNCERTAINTY_RANGES[method_upper].copy()

            logger.debug(
                "get_uncertainty_range: %s -> [%s%%, %s%%]",
                method_upper, result["low_pct"], result["high_pct"],
            )

            return result

    # =========================================================================
    # 16. get_all_vehicle_types
    # =========================================================================

    def get_all_vehicle_types(self) -> List[str]:
        """
        Get list of all available vehicle/vessel type keys.

        Returns:
            Sorted list of all vehicle type strings.

        Example:
            >>> types = db.get_all_vehicle_types()
            >>> "ARTICULATED_40_44T" in types
            True
        """
        with self._instance_lock:
            self._increment_lookup()

            result = sorted(TRANSPORT_EMISSION_FACTORS.keys())

            logger.debug("get_all_vehicle_types: %d types", len(result))

            return result

    # =========================================================================
    # 17. get_all_modes
    # =========================================================================

    def get_all_modes(self) -> List[str]:
        """
        Get list of all supported transport modes.

        Returns:
            List of mode strings: ROAD, RAIL, MARITIME, AIR, PIPELINE.

        Example:
            >>> db.get_all_modes()
            ['AIR', 'MARITIME', 'PIPELINE', 'RAIL', 'ROAD']
        """
        with self._instance_lock:
            self._increment_lookup()

            modes = sorted(MODE_DEFAULT_VEHICLE_TYPES.keys())

            logger.debug("get_all_modes: %s", modes)

            return modes

    # =========================================================================
    # 18. compare_mode_emissions
    # =========================================================================

    def compare_mode_emissions(
        self, weight_tonnes: float, distance_km: float
    ) -> List[Dict[str, Any]]:
        """
        Compare emissions across all transport modes for a given shipment.

        Calculates estimated emissions using the default vehicle type for each
        mode and ranks them from lowest to highest. Useful for mode-shift
        analysis and sustainability recommendations.

        Args:
            weight_tonnes: Shipment weight in tonnes. Must be > 0.
            distance_km: Transport distance in km. Must be > 0.

        Returns:
            List of dicts sorted by emissions (ascending), each containing:
            mode, vehicle_type, ef_per_tkm, estimated_emissions_kgco2e,
            tonne_km.

        Raises:
            ValueError: If weight_tonnes or distance_km is <= 0.

        Example:
            >>> results = db.compare_mode_emissions(10.0, 500.0)
            >>> results[0]["mode"]  # Lowest emissions mode
            'PIPELINE'
        """
        with self._instance_lock:
            self._increment_lookup()

            weight = Decimal(str(weight_tonnes))
            distance = Decimal(str(distance_km))

            if weight <= _ZERO:
                raise ValueError(f"weight_tonnes must be > 0, got {weight_tonnes}")
            if distance <= _ZERO:
                raise ValueError(f"distance_km must be > 0, got {distance_km}")

            tonne_km = self._quantize(weight * distance)

            comparisons: List[Dict[str, Any]] = []

            # Use a representative vehicle type per mode
            mode_defaults = {
                "ROAD": "ARTICULATED_40_44T",
                "RAIL": "FREIGHT_DIESEL",
                "MARITIME": "CONTAINER_POST_PANAMAX",
                "AIR": "LONG_HAUL_WIDE",
                "PIPELINE": "REFINED_PRODUCTS_PIPELINE",
            }

            for mode, default_vtype in mode_defaults.items():
                ef_data = TRANSPORT_EMISSION_FACTORS.get(default_vtype)
                if ef_data is None:
                    continue

                ef = ef_data["ef_per_tkm"]
                emissions = self._quantize(tonne_km * ef)

                comparisons.append({
                    "mode": mode,
                    "vehicle_type": default_vtype,
                    "ef_per_tkm": ef,
                    "estimated_emissions_kgco2e": emissions,
                    "tonne_km": tonne_km,
                })

            # Sort by emissions ascending
            comparisons.sort(key=lambda x: x["estimated_emissions_kgco2e"])

            logger.info(
                "compare_mode_emissions: %s t, %s km -> %d modes compared, "
                "lowest=%s (%s kgCO2e)",
                weight, distance, len(comparisons),
                comparisons[0]["mode"] if comparisons else "N/A",
                comparisons[0]["estimated_emissions_kgco2e"] if comparisons else "N/A",
            )

            return comparisons

    # =========================================================================
    # ADDITIONAL PUBLIC METHODS
    # =========================================================================

    def get_load_factor_default(self, mode: str) -> Decimal:
        """
        Get default load factor for a transport mode.

        Args:
            mode: Transport mode ("ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE").

        Returns:
            Default load factor (Decimal, 0-1).

        Raises:
            ValueError: If mode is not recognized.

        Example:
            >>> db.get_load_factor_default("ROAD")
            Decimal('0.65000000')
        """
        with self._instance_lock:
            self._increment_lookup()

            mode_upper = mode.upper()
            if mode_upper not in LOAD_FACTOR_DEFAULTS:
                raise ValueError(
                    f"Unknown mode: '{mode}'. "
                    f"Available: {sorted(LOAD_FACTOR_DEFAULTS.keys())}"
                )

            factor = LOAD_FACTOR_DEFAULTS[mode_upper]

            logger.debug("get_load_factor_default: %s -> %s", mode_upper, factor)

            return factor

    def get_vehicle_types_by_mode(self, mode: str) -> List[str]:
        """
        Get all vehicle types available for a specific mode.

        Args:
            mode: Transport mode ("ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE").

        Returns:
            List of vehicle type strings for the given mode.

        Raises:
            ValueError: If mode is not recognized.

        Example:
            >>> db.get_vehicle_types_by_mode("RAIL")
            ['BULK_RAIL', 'FREIGHT_DIESEL', 'FREIGHT_ELECTRIC', 'INTERMODAL_RAIL']
        """
        with self._instance_lock:
            self._increment_lookup()

            mode_upper = mode.upper()
            if mode_upper not in MODE_DEFAULT_VEHICLE_TYPES:
                raise ValueError(
                    f"Unknown mode: '{mode}'. "
                    f"Available: {sorted(MODE_DEFAULT_VEHICLE_TYPES.keys())}"
                )

            result = sorted(MODE_DEFAULT_VEHICLE_TYPES[mode_upper])

            logger.debug(
                "get_vehicle_types_by_mode: %s -> %d types",
                mode_upper, len(result),
            )

            return result

    def get_all_distribution_channels(self) -> List[str]:
        """
        Get all available distribution channel types.

        Returns:
            Sorted list of channel name strings.

        Example:
            >>> db.get_all_distribution_channels()
            ['DIRECT_TO_CONSUMER', 'DISTRIBUTOR', 'DROP_SHIP', ...]
        """
        with self._instance_lock:
            self._increment_lookup()
            return sorted(DISTRIBUTION_CHANNEL_DEFAULTS.keys())

    def get_all_incoterms(self) -> List[str]:
        """
        Get all supported Incoterms.

        Returns:
            Sorted list of Incoterm code strings.

        Example:
            >>> db.get_all_incoterms()
            ['CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DDP', ...]
        """
        with self._instance_lock:
            self._increment_lookup()
            return sorted(INCOTERM_CLASSIFICATIONS.keys())

    def get_cat9_incoterms(self) -> List[str]:
        """
        Get Incoterms where downstream transport falls in Category 9 scope.

        Returns:
            List of Incoterm codes where cat9_scope is True.

        Example:
            >>> db.get_cat9_incoterms()
            ['CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DDP', 'DDU', 'DPU', 'UNKNOWN']
        """
        with self._instance_lock:
            self._increment_lookup()
            return sorted(
                code
                for code, data in INCOTERM_CLASSIFICATIONS.items()
                if data["cat9_scope"]
            )

    def get_all_warehouse_types(self) -> List[str]:
        """
        Get all supported warehouse types.

        Returns:
            Sorted list of warehouse type strings.

        Example:
            >>> db.get_all_warehouse_types()
            ['AMBIENT_WAREHOUSE', 'CROSS_DOCK', ...]
        """
        with self._instance_lock:
            self._increment_lookup()
            return sorted(WAREHOUSE_EMISSION_FACTORS.keys())

    def get_all_last_mile_types(self) -> List[str]:
        """
        Get all supported last-mile delivery types.

        Returns:
            Sorted list of unique delivery type strings.

        Example:
            >>> db.get_all_last_mile_types()
            ['CARGO_BIKE', 'CLICK_AND_COLLECT', 'DRONE', ...]
        """
        with self._instance_lock:
            self._increment_lookup()
            return sorted(set(k[0] for k in LAST_MILE_DELIVERY_FACTORS.keys()))

    def get_lookup_count(self) -> int:
        """
        Get total number of factor lookups performed.

        Returns:
            Integer count of lookups since engine initialization.

        Example:
            >>> db.get_lookup_count()
            42
        """
        with self._instance_lock:
            return self._lookup_count

    def calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            data: Dictionary of input/output data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.

        Example:
            >>> h = db.calculate_provenance_hash({"vehicle": "ARTICULATED_40_44T"})
            >>> len(h)
            64
        """
        import json as json_mod
        serialized = json_mod.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESS FUNCTIONS
# ==============================================================================

_module_instance: Optional[DownstreamTransportDatabaseEngine] = None
_module_lock: threading.Lock = threading.Lock()


def get_downstream_transport_database() -> DownstreamTransportDatabaseEngine:
    """
    Get the singleton DownstreamTransportDatabaseEngine instance.

    Thread-safe. Creates the instance on first call, returns existing
    instance on subsequent calls.

    Returns:
        DownstreamTransportDatabaseEngine singleton.

    Example:
        >>> db = get_downstream_transport_database()
        >>> ef = db.get_transport_ef("ARTICULATED_40_44T")
    """
    global _module_instance
    if _module_instance is None:
        with _module_lock:
            if _module_instance is None:
                _module_instance = DownstreamTransportDatabaseEngine()
    return _module_instance


def reset_downstream_transport_database() -> None:
    """
    Reset the singleton instance (primarily for testing).

    Clears both the module-level reference and the class-level singleton,
    allowing a fresh instance to be created on the next call to
    get_downstream_transport_database().

    Example:
        >>> reset_downstream_transport_database()
        >>> db = get_downstream_transport_database()  # Fresh instance
    """
    global _module_instance
    with _module_lock:
        _module_instance = None
        DownstreamTransportDatabaseEngine._instance = None
        if hasattr(DownstreamTransportDatabaseEngine, "_initialized"):
            # Clear on class is not needed since _initialized is instance attr
            pass

    logger.info("DownstreamTransportDatabaseEngine singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine
    "DownstreamTransportDatabaseEngine",
    "get_downstream_transport_database",
    "reset_downstream_transport_database",
    # Enumerations
    "TransportMode",
    "TemperatureRegime",
    "WarehouseType",
    "DeliveryType",
    "DeliveryArea",
    "DistributionChannel",
    "ReturnType",
    "CalculationMethodType",
    # Data tables (for direct access in other engines)
    "TRANSPORT_EMISSION_FACTORS",
    "COLD_CHAIN_UPLIFT_FACTORS",
    "WAREHOUSE_EMISSION_FACTORS",
    "LAST_MILE_DELIVERY_FACTORS",
    "EEIO_FACTORS",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "GRID_EMISSION_FACTORS",
    "DISTRIBUTION_CHANNEL_DEFAULTS",
    "INCOTERM_CLASSIFICATIONS",
    "LOAD_FACTOR_DEFAULTS",
    "RETURN_TRIP_MULTIPLIERS",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",
    "MODE_DEFAULT_VEHICLE_TYPES",
]
