"""
Downstream Transportation & Distribution Agent Models (AGENT-MRV-022)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 9
(Downstream Transportation & Distribution) emissions calculations.

Supports:
- 4 calculation methods (distance-based, spend-based, average-data, supplier-specific)
- 9 transport modes (road, rail, maritime, air, inland waterway, pipeline, intermodal,
  courier, last-mile)
- 19 vehicle/vessel types with DEFRA 2024, IMO 2020, GLEC v3, ICAO 2024 emission factors
- 11 Incoterms (ICC 2020) with Cat 4 vs Cat 9 boundary classification
- 6 distribution channels with industry-average default parameters
- 5 temperature regimes with cold chain uplift factors
- 7 warehouse types with electricity and gas emission factors per m2/year
- 6 last-mile delivery types with urban/suburban/rural emission factors
- 10 NAICS sector EEIO factors for spend-based screening
- 12 currencies with USD conversion rates
- 11-year CPI deflation table (2015-2025, base year 2024)
- 4 return logistics multipliers (no-return, customer return, recall, reusable)
- 5 load factor adjustments (empty to full)
- 11 country grid emission factors for warehouse electricity
- 5-dimension data quality indicator (DQI) scoring
- 4-method uncertainty ranges (distance/spend/average/supplier)
- 10-stage pipeline provenance tracking
- 7-framework compliance checking (GHG Protocol, ISO 14064, ISO 14083, CSRD, CDP,
  SBTi, SB 253)
- SHA-256 provenance chain with Merkle root computation

All numeric fields use ``Decimal`` for precision in regulatory calculations.
All input models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.downstream_transportation.models import (
    ...     ShipmentInput, TransportMode, VehicleType, Incoterm
    ... )
    >>> shipment = ShipmentInput(
    ...     shipment_id="SHP-001",
    ...     mode=TransportMode.ROAD,
    ...     vehicle_type=VehicleType.ARTICULATED_TRUCK,
    ...     origin="Chicago, IL",
    ...     destination="Dallas, TX",
    ...     distance_km=Decimal("1480"),
    ...     weight_tonnes=Decimal("18.5"),
    ...     incoterm=Incoterm.EXW,
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-009"
AGENT_COMPONENT: str = "AGENT-MRV-022"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dto_"
METRICS_PREFIX: str = "gl_dto_"
API_PREFIX: str = "/api/v1/downstream-transportation"


# ==============================================================================
# ENUMERATIONS (22 Enums)
# ==============================================================================


class TransportMode(str, Enum):
    """Transport modes for downstream freight and distribution.

    Covers the full spectrum of outbound logistics modes from factory gate
    to end consumer, including dedicated last-mile and courier categories
    that have grown with e-commerce.
    """

    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    INLAND_WATERWAY = "inland_waterway"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"
    COURIER = "courier"
    LAST_MILE = "last_mile"


class VehicleType(str, Enum):
    """Vehicle and vessel types with distinct emission profiles.

    Road vehicles span light goods vehicles through articulated trucks.
    Maritime vessels are sized by TEU capacity.  Air freight includes
    dedicated freighters and belly freight on passenger aircraft.
    """

    # Road (9 types)
    LGV_PETROL = "lgv_petrol"
    LGV_DIESEL = "lgv_diesel"
    LGV_ELECTRIC = "lgv_electric"
    RIGID_TRUCK_SMALL = "rigid_truck_small"        # <7.5t
    RIGID_TRUCK_MEDIUM = "rigid_truck_medium"      # 7.5-17t
    RIGID_TRUCK_LARGE = "rigid_truck_large"        # >17t
    ARTICULATED_TRUCK = "articulated_truck"        # >33t
    DELIVERY_VAN = "delivery_van"
    CARGO_BIKE = "cargo_bike"
    # Rail (2 types)
    FREIGHT_TRAIN = "freight_train"
    INTERMODAL_RAIL = "intermodal_rail"
    # Maritime (5 types)
    CONTAINER_SHIP_SMALL = "container_ship_small"   # <1000 TEU
    CONTAINER_SHIP_MEDIUM = "container_ship_medium"  # 1000-5000 TEU
    CONTAINER_SHIP_LARGE = "container_ship_large"   # >5000 TEU
    BULK_CARRIER = "bulk_carrier"
    RO_RO_FERRY = "ro_ro_ferry"
    # Air (3 types)
    FREIGHTER_NARROW = "freighter_narrow"
    FREIGHTER_WIDE = "freighter_wide"
    BELLY_FREIGHT = "belly_freight"


class Incoterm(str, Enum):
    """ICC Incoterms 2020 defining transport cost/risk transfer points.

    Incoterms determine the boundary between Category 4 (seller-paid
    upstream transportation) and Category 9 (buyer-paid downstream
    transportation).  EXW through FOB place most transport in Cat 9;
    CFR through DDP place most or all transport in Cat 4.
    """

    EXW = "exw"    # Ex Works -- buyer arranges all transport (Cat 9)
    FCA = "fca"    # Free Carrier -- seller delivers to carrier (Cat 9 after)
    FAS = "fas"    # Free Alongside Ship (Cat 9 after)
    FOB = "fob"    # Free On Board (Cat 9 after loading)
    CFR = "cfr"    # Cost and Freight -- seller pays main carriage (Cat 4)
    CIF = "cif"    # Cost, Insurance, Freight (Cat 4)
    CIP = "cip"    # Carriage and Insurance Paid (Cat 4)
    CPT = "cpt"    # Carriage Paid To (Cat 4)
    DAP = "dap"    # Delivered At Place (Cat 4)
    DPU = "dpu"    # Delivered at Place Unloaded (Cat 4)
    DDP = "ddp"    # Delivered Duty Paid (Cat 4)


class DistributionChannel(str, Enum):
    """Distribution channels through which sold products reach the end consumer.

    Each channel has different average distances, transport modes, number
    of legs, and storage durations that drive default emission profiles
    when detailed shipment data is unavailable.
    """

    DIRECT_TO_CONSUMER = "direct_to_consumer"
    WHOLESALE = "wholesale"
    RETAIL = "retail"
    E_COMMERCE = "e_commerce"
    DISTRIBUTOR = "distributor"
    FRANCHISE = "franchise"


class CalculationMethod(str, Enum):
    """Calculation methodology per GHG Protocol Scope 3 guidance.

    Supplier-specific is preferred (Tier 1), followed by distance-based
    (Tier 2), average-data (Tier 3), and spend-based EEIO (Tier 3/4).
    """

    DISTANCE_BASED = "distance_based"
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class TemperatureRegime(str, Enum):
    """Temperature regimes for cold chain logistics.

    Cold chain transport adds 15-50% energy uplift depending on mode
    and required temperature.  Deep frozen goods have the highest
    refrigeration energy penalty.
    """

    AMBIENT = "ambient"
    CHILLED = "chilled"            # 2-8 deg C
    FROZEN = "frozen"              # -18 to -25 deg C
    DEEP_FROZEN = "deep_frozen"    # <-25 deg C
    CONTROLLED = "controlled"      # 15-25 deg C


class AllocationMethod(str, Enum):
    """Methods for allocating shared logistics emissions to individual products.

    Mass-based and volume-based are most common for freight; revenue-based
    is an alternative when physical characteristics vary widely.
    """

    MASS = "mass"
    VOLUME = "volume"
    REVENUE = "revenue"
    UNITS_SOLD = "units_sold"
    TEU = "teu"
    PALLET_POSITIONS = "pallet_positions"
    FLOOR_AREA = "floor_area"


class WarehouseType(str, Enum):
    """Warehouse and distribution facility types.

    Each type has a distinct energy intensity profile driven by HVAC,
    refrigeration, lighting, and material-handling requirements.
    """

    DISTRIBUTION_CENTER = "distribution_center"
    CROSS_DOCK = "cross_dock"
    COLD_STORAGE = "cold_storage"
    RETAIL_STORE = "retail_store"
    FULFILLMENT_CENTER = "fulfillment_center"
    DARK_STORE = "dark_store"
    FROZEN_STORAGE = "frozen_storage"


class EnergySource(str, Enum):
    """Energy sources consumed by warehouse and distribution facilities."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    LPG = "lpg"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"


class EmissionFactorSource(str, Enum):
    """Authoritative emission factor databases used for EF resolution.

    The agent follows a hierarchical resolution strategy:
    supplier-specific > DEFRA > EPA SmartWay > GLEC > ICAO > IMO > IEA.
    """

    DEFRA = "defra"
    EPA_SMARTWAY = "epa_smartway"
    GLEC = "glec"
    ICAO = "icao"
    IMO = "imo"
    IEA = "iea"
    ECOINVENT = "ecoinvent"
    CUSTOM = "custom"


class ComplianceFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14083 = "iso_14083"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"


class ComplianceSeverity(str, Enum):
    """Severity levels for compliance findings."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ComplianceStatus(str, Enum):
    """Overall compliance check result status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges and DQI scoring.

    Tier 1 (supplier-specific) is highest quality; Tier 4 (proxy/estimated)
    is lowest.  GHG Protocol recommends progressing from Tier 3/4 screening
    to Tier 1/2 for material categories.
    """

    TIER_1 = "tier_1"   # Supplier-specific / primary data
    TIER_2 = "tier_2"   # Industry average / mode-specific
    TIER_3 = "tier_3"   # Spend-based EEIO
    TIER_4 = "tier_4"   # Proxy / estimated


class ReturnType(str, Enum):
    """Return logistics types affecting reverse-trip emission multipliers."""

    NO_RETURN = "no_return"
    CUSTOMER_RETURN = "customer_return"
    PRODUCT_RECALL = "product_recall"
    REUSABLE_PACKAGING = "reusable_packaging"


class LastMileType(str, Enum):
    """Last-mile delivery service types.

    E-commerce growth has made last-mile delivery a significant and
    rapidly growing source of Category 9 emissions.
    """

    PARCEL_STANDARD = "parcel_standard"
    PARCEL_EXPRESS = "parcel_express"
    SAME_DAY = "same_day"
    CLICK_AND_COLLECT = "click_and_collect"
    LOCKER = "locker"
    CARGO_BIKE = "cargo_bike"


class GWPSource(str, Enum):
    """IPCC Assessment Report versions for Global Warming Potential values."""

    AR4 = "ar4"
    AR5 = "ar5"
    AR6 = "ar6"
    AR6_20YR = "ar6_20yr"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""

    IPCC_TIER_1 = "ipcc_tier_1"
    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    EXPERT_JUDGMENT = "expert_judgment"


class PipelineStage(str, Enum):
    """10-stage processing pipeline stages for provenance tracking.

    Each stage produces an input hash and output hash, chained together
    via SHA-256 to form an immutable provenance record.
    """

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


class LoadFactor(str, Enum):
    """Vehicle load factor / utilization levels.

    Load factor adjustments scale the per-tonne-km emission factor to
    account for actual vs typical vehicle utilization.  Empty vehicles
    still produce deadhead emissions.
    """

    EMPTY = "empty"           # 0% utilization
    PARTIAL = "partial"       # 25-50%
    HALF = "half"             # 50%
    TYPICAL = "typical"       # 60-75%
    FULL = "full"             # 85-100%


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol guidance."""

    TECHNOLOGICAL = "technological"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"


class EmissionGas(str, Enum):
    """Greenhouse gas species relevant to downstream transport emissions."""

    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    CO2_BIOGENIC = "co2_biogenic"


# ==============================================================================
# CONSTANT TABLES (14 Tables)
# ==============================================================================


# ---------------------------------------------------------------------------
# 4.1 TRANSPORT_EMISSION_FACTORS  (per tonne-km, kgCO2e)
# 26 entries: mode/vehicle -> ef_per_tkm (TTW), wtt_per_tkm, source
# Sources: DEFRA 2024, IMO 2020, GLEC v3, ICAO 2024, industry average
# ---------------------------------------------------------------------------

TRANSPORT_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    # -- Road (9 vehicle types) --
    VehicleType.LGV_PETROL.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.LGV_PETROL.value,
        "ef_per_tkm": Decimal("0.584"),
        "wtt_per_tkm": Decimal("0.139"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Light goods vehicle -- petrol",
    },
    VehicleType.LGV_DIESEL.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.LGV_DIESEL.value,
        "ef_per_tkm": Decimal("0.480"),
        "wtt_per_tkm": Decimal("0.112"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Light goods vehicle -- diesel",
    },
    VehicleType.LGV_ELECTRIC.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.LGV_ELECTRIC.value,
        "ef_per_tkm": Decimal("0.118"),
        "wtt_per_tkm": Decimal("0.024"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Light goods vehicle -- electric",
    },
    VehicleType.RIGID_TRUCK_SMALL.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.RIGID_TRUCK_SMALL.value,
        "ef_per_tkm": Decimal("0.441"),
        "wtt_per_tkm": Decimal("0.103"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Rigid truck <7.5t",
    },
    VehicleType.RIGID_TRUCK_MEDIUM.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.RIGID_TRUCK_MEDIUM.value,
        "ef_per_tkm": Decimal("0.213"),
        "wtt_per_tkm": Decimal("0.050"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Rigid truck 7.5-17t",
    },
    VehicleType.RIGID_TRUCK_LARGE.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.RIGID_TRUCK_LARGE.value,
        "ef_per_tkm": Decimal("0.150"),
        "wtt_per_tkm": Decimal("0.035"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Rigid truck >17t",
    },
    VehicleType.ARTICULATED_TRUCK.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.ARTICULATED_TRUCK.value,
        "ef_per_tkm": Decimal("0.107"),
        "wtt_per_tkm": Decimal("0.025"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Articulated truck >33t",
    },
    VehicleType.DELIVERY_VAN.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.DELIVERY_VAN.value,
        "ef_per_tkm": Decimal("0.580"),
        "wtt_per_tkm": Decimal("0.135"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Delivery van",
    },
    VehicleType.CARGO_BIKE.value: {
        "mode": TransportMode.ROAD.value,
        "vehicle_type": VehicleType.CARGO_BIKE.value,
        "ef_per_tkm": Decimal("0.000"),
        "wtt_per_tkm": Decimal("0.000"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Cargo bike -- zero emission",
    },
    # -- Rail (2 types) --
    VehicleType.FREIGHT_TRAIN.value: {
        "mode": TransportMode.RAIL.value,
        "vehicle_type": VehicleType.FREIGHT_TRAIN.value,
        "ef_per_tkm": Decimal("0.028"),
        "wtt_per_tkm": Decimal("0.006"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Freight train",
    },
    VehicleType.INTERMODAL_RAIL.value: {
        "mode": TransportMode.RAIL.value,
        "vehicle_type": VehicleType.INTERMODAL_RAIL.value,
        "ef_per_tkm": Decimal("0.025"),
        "wtt_per_tkm": Decimal("0.005"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Intermodal rail",
    },
    # -- Maritime (5 types) --
    VehicleType.CONTAINER_SHIP_SMALL.value: {
        "mode": TransportMode.MARITIME.value,
        "vehicle_type": VehicleType.CONTAINER_SHIP_SMALL.value,
        "ef_per_tkm": Decimal("0.022"),
        "wtt_per_tkm": Decimal("0.005"),
        "source": EmissionFactorSource.IMO.value,
        "description": "Container ship <1000 TEU",
    },
    VehicleType.CONTAINER_SHIP_MEDIUM.value: {
        "mode": TransportMode.MARITIME.value,
        "vehicle_type": VehicleType.CONTAINER_SHIP_MEDIUM.value,
        "ef_per_tkm": Decimal("0.016"),
        "wtt_per_tkm": Decimal("0.004"),
        "source": EmissionFactorSource.IMO.value,
        "description": "Container ship 1000-5000 TEU",
    },
    VehicleType.CONTAINER_SHIP_LARGE.value: {
        "mode": TransportMode.MARITIME.value,
        "vehicle_type": VehicleType.CONTAINER_SHIP_LARGE.value,
        "ef_per_tkm": Decimal("0.008"),
        "wtt_per_tkm": Decimal("0.002"),
        "source": EmissionFactorSource.IMO.value,
        "description": "Container ship >5000 TEU",
    },
    VehicleType.BULK_CARRIER.value: {
        "mode": TransportMode.MARITIME.value,
        "vehicle_type": VehicleType.BULK_CARRIER.value,
        "ef_per_tkm": Decimal("0.005"),
        "wtt_per_tkm": Decimal("0.001"),
        "source": EmissionFactorSource.IMO.value,
        "description": "Bulk carrier",
    },
    VehicleType.RO_RO_FERRY.value: {
        "mode": TransportMode.MARITIME.value,
        "vehicle_type": VehicleType.RO_RO_FERRY.value,
        "ef_per_tkm": Decimal("0.060"),
        "wtt_per_tkm": Decimal("0.014"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Ro-Ro ferry",
    },
    # -- Air (3 types) --
    VehicleType.FREIGHTER_NARROW.value: {
        "mode": TransportMode.AIR.value,
        "vehicle_type": VehicleType.FREIGHTER_NARROW.value,
        "ef_per_tkm": Decimal("0.602"),
        "wtt_per_tkm": Decimal("0.143"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Narrow-body freighter",
    },
    VehicleType.FREIGHTER_WIDE.value: {
        "mode": TransportMode.AIR.value,
        "vehicle_type": VehicleType.FREIGHTER_WIDE.value,
        "ef_per_tkm": Decimal("0.495"),
        "wtt_per_tkm": Decimal("0.118"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Wide-body freighter",
    },
    VehicleType.BELLY_FREIGHT.value: {
        "mode": TransportMode.AIR.value,
        "vehicle_type": VehicleType.BELLY_FREIGHT.value,
        "ef_per_tkm": Decimal("0.440"),
        "wtt_per_tkm": Decimal("0.105"),
        "source": EmissionFactorSource.ICAO.value,
        "description": "Belly freight on passenger aircraft",
    },
    # -- Inland Waterway (1 type, keyed by mode) --
    "barge": {
        "mode": TransportMode.INLAND_WATERWAY.value,
        "vehicle_type": "barge",
        "ef_per_tkm": Decimal("0.032"),
        "wtt_per_tkm": Decimal("0.007"),
        "source": EmissionFactorSource.GLEC.value,
        "description": "Inland waterway barge",
    },
    # -- Courier (2 types) --
    "parcel_standard_courier": {
        "mode": TransportMode.COURIER.value,
        "vehicle_type": "parcel_standard_courier",
        "ef_per_tkm": Decimal("0.420"),
        "wtt_per_tkm": Decimal("0.098"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Standard parcel courier",
    },
    "parcel_express_courier": {
        "mode": TransportMode.COURIER.value,
        "vehicle_type": "parcel_express_courier",
        "ef_per_tkm": Decimal("0.520"),
        "wtt_per_tkm": Decimal("0.121"),
        "source": EmissionFactorSource.DEFRA.value,
        "description": "Express parcel courier",
    },
    # -- Last Mile (4 types) --
    "same_day_last_mile": {
        "mode": TransportMode.LAST_MILE.value,
        "vehicle_type": "same_day_last_mile",
        "ef_per_tkm": Decimal("0.680"),
        "wtt_per_tkm": Decimal("0.159"),
        "source": EmissionFactorSource.CUSTOM.value,
        "description": "Same-day last-mile delivery",
    },
    "click_collect_last_mile": {
        "mode": TransportMode.LAST_MILE.value,
        "vehicle_type": "click_collect_last_mile",
        "ef_per_tkm": Decimal("0.050"),
        "wtt_per_tkm": Decimal("0.012"),
        "source": EmissionFactorSource.CUSTOM.value,
        "description": "Click-and-collect",
    },
    "locker_last_mile": {
        "mode": TransportMode.LAST_MILE.value,
        "vehicle_type": "locker_last_mile",
        "ef_per_tkm": Decimal("0.040"),
        "wtt_per_tkm": Decimal("0.009"),
        "source": EmissionFactorSource.CUSTOM.value,
        "description": "Parcel locker",
    },
    "cargo_bike_last_mile": {
        "mode": TransportMode.LAST_MILE.value,
        "vehicle_type": "cargo_bike_last_mile",
        "ef_per_tkm": Decimal("0.005"),
        "wtt_per_tkm": Decimal("0.001"),
        "source": EmissionFactorSource.CUSTOM.value,
        "description": "Cargo bike last-mile",
    },
}


# ---------------------------------------------------------------------------
# 4.2 COLD_CHAIN_UPLIFT_FACTORS
# 5 temperature regimes x 4 transport modes
# Multiplier applied to base emission factors for refrigerated transport.
# ---------------------------------------------------------------------------

COLD_CHAIN_UPLIFT_FACTORS: Dict[str, Dict[str, Decimal]] = {
    TemperatureRegime.AMBIENT.value: {
        "road": Decimal("1.00"),
        "rail": Decimal("1.00"),
        "maritime": Decimal("1.00"),
        "air": Decimal("1.00"),
    },
    TemperatureRegime.CHILLED.value: {
        "road": Decimal("1.20"),
        "rail": Decimal("1.15"),
        "maritime": Decimal("1.18"),
        "air": Decimal("1.10"),
    },
    TemperatureRegime.FROZEN.value: {
        "road": Decimal("1.35"),
        "rail": Decimal("1.25"),
        "maritime": Decimal("1.30"),
        "air": Decimal("1.15"),
    },
    TemperatureRegime.DEEP_FROZEN.value: {
        "road": Decimal("1.50"),
        "rail": Decimal("1.35"),
        "maritime": Decimal("1.40"),
        "air": Decimal("1.20"),
    },
    TemperatureRegime.CONTROLLED.value: {
        "road": Decimal("1.05"),
        "rail": Decimal("1.03"),
        "maritime": Decimal("1.04"),
        "air": Decimal("1.02"),
    },
}


# ---------------------------------------------------------------------------
# 4.3 WAREHOUSE_EMISSION_FACTORS  (kgCO2e per m2 per year)
# 7 warehouse types with electricity, gas/heating, and total EFs.
# Sources: CIBSE TM46, industry average
# ---------------------------------------------------------------------------

WAREHOUSE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    WarehouseType.DISTRIBUTION_CENTER.value: {
        "electricity_ef": Decimal("45.0"),
        "gas_ef": Decimal("12.0"),
        "total_ef": Decimal("57.0"),
        "source": "CIBSE TM46",
    },
    WarehouseType.CROSS_DOCK.value: {
        "electricity_ef": Decimal("30.0"),
        "gas_ef": Decimal("8.0"),
        "total_ef": Decimal("38.0"),
        "source": "CIBSE TM46",
    },
    WarehouseType.COLD_STORAGE.value: {
        "electricity_ef": Decimal("120.0"),
        "gas_ef": Decimal("5.0"),
        "total_ef": Decimal("125.0"),
        "source": "Industry avg",
    },
    WarehouseType.FROZEN_STORAGE.value: {
        "electricity_ef": Decimal("180.0"),
        "gas_ef": Decimal("3.0"),
        "total_ef": Decimal("183.0"),
        "source": "Industry avg",
    },
    WarehouseType.RETAIL_STORE.value: {
        "electricity_ef": Decimal("85.0"),
        "gas_ef": Decimal("25.0"),
        "total_ef": Decimal("110.0"),
        "source": "CIBSE TM46",
    },
    WarehouseType.FULFILLMENT_CENTER.value: {
        "electricity_ef": Decimal("55.0"),
        "gas_ef": Decimal("10.0"),
        "total_ef": Decimal("65.0"),
        "source": "Industry avg",
    },
    WarehouseType.DARK_STORE.value: {
        "electricity_ef": Decimal("95.0"),
        "gas_ef": Decimal("15.0"),
        "total_ef": Decimal("110.0"),
        "source": "Industry avg",
    },
}


# ---------------------------------------------------------------------------
# 4.4 LAST_MILE_EMISSION_FACTORS  (kgCO2e per delivery)
# 6 last-mile types x 3 area types (urban, suburban, rural)
# Sources: DEFRA 2024, industry average
# ---------------------------------------------------------------------------

LAST_MILE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    LastMileType.PARCEL_STANDARD.value: {
        "urban": Decimal("0.520"),
        "suburban": Decimal("0.780"),
        "rural": Decimal("1.200"),
        "source": EmissionFactorSource.DEFRA.value,
    },
    LastMileType.PARCEL_EXPRESS.value: {
        "urban": Decimal("0.680"),
        "suburban": Decimal("0.950"),
        "rural": Decimal("1.500"),
        "source": EmissionFactorSource.DEFRA.value,
    },
    LastMileType.SAME_DAY.value: {
        "urban": Decimal("0.850"),
        "suburban": Decimal("1.200"),
        "rural": Decimal("1.800"),
        "source": EmissionFactorSource.CUSTOM.value,
    },
    LastMileType.CLICK_AND_COLLECT.value: {
        "urban": Decimal("0.050"),
        "suburban": Decimal("0.050"),
        "rural": Decimal("0.050"),
        "source": EmissionFactorSource.CUSTOM.value,
    },
    LastMileType.LOCKER.value: {
        "urban": Decimal("0.040"),
        "suburban": Decimal("0.040"),
        "rural": Decimal("0.040"),
        "source": EmissionFactorSource.CUSTOM.value,
    },
    LastMileType.CARGO_BIKE.value: {
        "urban": Decimal("0.010"),
        "suburban": Decimal("0.020"),
        # Rural not applicable for cargo bikes; use None sentinel
        "rural": None,
        "source": EmissionFactorSource.CUSTOM.value,
    },
}


# ---------------------------------------------------------------------------
# 4.5 EEIO_FACTORS  (kgCO2e per USD spent)
# 10 NAICS sectors relevant to downstream transportation and distribution.
# Source: EPA USEEIO v2.0
# ---------------------------------------------------------------------------

EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "484110": {
        "sector": "General Freight Trucking, Local",
        "ef_per_usd": Decimal("0.470"),
        "source": "EPA USEEIO v2.0",
    },
    "484121": {
        "sector": "General Freight Trucking, Long-Distance",
        "ef_per_usd": Decimal("0.380"),
        "source": "EPA USEEIO v2.0",
    },
    "482110": {
        "sector": "Rail Transportation",
        "ef_per_usd": Decimal("0.280"),
        "source": "EPA USEEIO v2.0",
    },
    "483111": {
        "sector": "Deep Sea Freight",
        "ef_per_usd": Decimal("0.210"),
        "source": "EPA USEEIO v2.0",
    },
    "481112": {
        "sector": "Air Freight",
        "ef_per_usd": Decimal("1.250"),
        "source": "EPA USEEIO v2.0",
    },
    "492110": {
        "sector": "Couriers and Express Delivery",
        "ef_per_usd": Decimal("0.520"),
        "source": "EPA USEEIO v2.0",
    },
    "493110": {
        "sector": "General Warehousing and Storage",
        "ef_per_usd": Decimal("0.340"),
        "source": "EPA USEEIO v2.0",
    },
    "493120": {
        "sector": "Refrigerated Warehousing and Storage",
        "ef_per_usd": Decimal("0.580"),
        "source": "EPA USEEIO v2.0",
    },
    "454110": {
        "sector": "Electronic Shopping and Mail-Order",
        "ef_per_usd": Decimal("0.420"),
        "source": "EPA USEEIO v2.0",
    },
    "493130": {
        "sector": "Farm Product Warehousing",
        "ef_per_usd": Decimal("0.310"),
        "source": "EPA USEEIO v2.0",
    },
}


# ---------------------------------------------------------------------------
# 4.6 CURRENCY_CONVERSION_RATES  (to USD, 2024 mid-market approximations)
# 12 currencies
# ---------------------------------------------------------------------------

CURRENCY_CONVERSION_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1.0000"),
    "EUR": Decimal("1.0850"),
    "GBP": Decimal("1.2650"),
    "JPY": Decimal("0.0067"),
    "CAD": Decimal("0.7400"),
    "AUD": Decimal("0.6550"),
    "CHF": Decimal("1.1300"),
    "CNY": Decimal("0.1400"),
    "INR": Decimal("0.0120"),
    "BRL": Decimal("0.2000"),
    "KRW": Decimal("0.0008"),
    "SEK": Decimal("0.0960"),
}


# ---------------------------------------------------------------------------
# 4.7 CPI_DEFLATORS  (base year = 2024)
# 11 years (2015-2025).  Deflator = CPI_year / CPI_2024.
# Source: US BLS CPI-U
# ---------------------------------------------------------------------------

CPI_DEFLATORS: Dict[int, Dict[str, Decimal]] = {
    2015: {"cpi_index": Decimal("237.0"), "deflator": Decimal("0.7983")},
    2016: {"cpi_index": Decimal("240.0"), "deflator": Decimal("0.8084")},
    2017: {"cpi_index": Decimal("245.1"), "deflator": Decimal("0.8256")},
    2018: {"cpi_index": Decimal("251.1"), "deflator": Decimal("0.8458")},
    2019: {"cpi_index": Decimal("255.7"), "deflator": Decimal("0.8613")},
    2020: {"cpi_index": Decimal("258.8"), "deflator": Decimal("0.8717")},
    2021: {"cpi_index": Decimal("270.9"), "deflator": Decimal("0.9125")},
    2022: {"cpi_index": Decimal("292.7"), "deflator": Decimal("0.9859")},
    2023: {"cpi_index": Decimal("304.7"), "deflator": Decimal("1.0264")},
    2024: {"cpi_index": Decimal("296.9"), "deflator": Decimal("1.0000")},
    2025: {"cpi_index": Decimal("303.5"), "deflator": Decimal("1.0222")},
}


# ---------------------------------------------------------------------------
# 4.8 RETURN_LOGISTICS_FACTORS
# Multiplier applied to outbound emissions for return / reverse logistics.
# ---------------------------------------------------------------------------

RETURN_LOGISTICS_FACTORS: Dict[str, Dict[str, Any]] = {
    ReturnType.NO_RETURN.value: {
        "multiplier": Decimal("0.00"),
        "description": "One-way only, no return leg",
    },
    ReturnType.CUSTOMER_RETURN.value: {
        "multiplier": Decimal("0.85"),
        "description": "85% of outbound -- partially consolidated returns",
    },
    ReturnType.PRODUCT_RECALL.value: {
        "multiplier": Decimal("1.00"),
        "description": "Equal to outbound -- full reverse logistics",
    },
    ReturnType.REUSABLE_PACKAGING.value: {
        "multiplier": Decimal("0.50"),
        "description": "50% of outbound -- backhaul of reusable packaging",
    },
}


# ---------------------------------------------------------------------------
# 4.9 LOAD_FACTOR_ADJUSTMENTS
# Adjustment multiplier to base EF based on actual vehicle utilization.
# Typical (60-75%) is the baseline (1.00).
# ---------------------------------------------------------------------------

LOAD_FACTOR_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    LoadFactor.EMPTY.value: {
        "utilization_pct": Decimal("0"),
        "adjustment": Decimal("0.40"),
        "description": "Deadhead / empty repositioning",
    },
    LoadFactor.PARTIAL.value: {
        "utilization_pct": Decimal("37"),
        "adjustment": Decimal("0.65"),
        "description": "25-50% utilization",
    },
    LoadFactor.HALF.value: {
        "utilization_pct": Decimal("50"),
        "adjustment": Decimal("0.80"),
        "description": "50% utilization",
    },
    LoadFactor.TYPICAL.value: {
        "utilization_pct": Decimal("67"),
        "adjustment": Decimal("1.00"),
        "description": "60-75% utilization (baseline)",
    },
    LoadFactor.FULL.value: {
        "utilization_pct": Decimal("92"),
        "adjustment": Decimal("1.15"),
        "description": "85-100% utilization",
    },
}


# ---------------------------------------------------------------------------
# 4.10 DISTRIBUTION_CHANNEL_DEFAULTS
# Average transport parameters by distribution channel, used for
# average-data method when detailed shipment data is unavailable.
# ---------------------------------------------------------------------------

DISTRIBUTION_CHANNEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    DistributionChannel.DIRECT_TO_CONSUMER.value: {
        "avg_distance_km": Decimal("500"),
        "avg_mode": TransportMode.ROAD.value,
        "avg_legs": 2,
        "storage_days": 0,
        "description": "Direct-to-consumer / DTC",
    },
    DistributionChannel.WHOLESALE.value: {
        "avg_distance_km": Decimal("800"),
        "avg_mode": TransportMode.ROAD.value,
        "avg_legs": 1,
        "storage_days": 14,
        "description": "Wholesale distribution",
    },
    DistributionChannel.RETAIL.value: {
        "avg_distance_km": Decimal("600"),
        "avg_mode": TransportMode.ROAD.value,
        "avg_legs": 2,
        "storage_days": 30,
        "description": "Retail store distribution",
    },
    DistributionChannel.E_COMMERCE.value: {
        "avg_distance_km": Decimal("350"),
        "avg_mode": TransportMode.COURIER.value,
        "avg_legs": 3,
        "storage_days": 7,
        "description": "E-commerce / online retail",
    },
    DistributionChannel.DISTRIBUTOR.value: {
        "avg_distance_km": Decimal("1200"),
        "avg_mode": TransportMode.INTERMODAL.value,
        "avg_legs": 2,
        "storage_days": 21,
        "description": "Third-party distributor",
    },
    DistributionChannel.FRANCHISE.value: {
        "avg_distance_km": Decimal("400"),
        "avg_mode": TransportMode.ROAD.value,
        "avg_legs": 1,
        "storage_days": 7,
        "description": "Franchise distribution",
    },
}


# ---------------------------------------------------------------------------
# 4.11 GRID_EMISSION_FACTORS  (kgCO2e per kWh, for warehouse electricity)
# 11 countries / regions.  GLOBAL is fallback default.
# Sources: EPA eGRID 2024, DEFRA 2024, IEA 2024
# ---------------------------------------------------------------------------

GRID_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "ef_kwh": Decimal("0.3937"),
        "source": "EPA eGRID 2024",
    },
    "GB": {
        "ef_kwh": Decimal("0.2121"),
        "source": "DEFRA 2024",
    },
    "DE": {
        "ef_kwh": Decimal("0.3640"),
        "source": "IEA 2024",
    },
    "FR": {
        "ef_kwh": Decimal("0.0569"),
        "source": "IEA 2024",
    },
    "JP": {
        "ef_kwh": Decimal("0.4570"),
        "source": "IEA 2024",
    },
    "CA": {
        "ef_kwh": Decimal("0.1200"),
        "source": "IEA 2024",
    },
    "AU": {
        "ef_kwh": Decimal("0.6100"),
        "source": "IEA 2024",
    },
    "IN": {
        "ef_kwh": Decimal("0.7080"),
        "source": "IEA 2024",
    },
    "CN": {
        "ef_kwh": Decimal("0.5570"),
        "source": "IEA 2024",
    },
    "BR": {
        "ef_kwh": Decimal("0.0740"),
        "source": "IEA 2024",
    },
    "GLOBAL": {
        "ef_kwh": Decimal("0.4360"),
        "source": "IEA 2024",
    },
}


# ---------------------------------------------------------------------------
# 4.12 INCOTERM_CLASSIFICATION
# ICC Incoterms 2020 mapped to Cat 4 (seller-paid) vs Cat 9 (buyer-paid)
# transport scope, with the point of risk/cost transfer.
# ---------------------------------------------------------------------------

INCOTERM_CLASSIFICATION: Dict[str, Dict[str, str]] = {
    Incoterm.EXW.value: {
        "cat4_scope": "No",
        "cat9_scope": "All transport",
        "transfer_point": "Seller's premises",
    },
    Incoterm.FCA.value: {
        "cat4_scope": "To carrier",
        "cat9_scope": "After carrier",
        "transfer_point": "Named place",
    },
    Incoterm.FAS.value: {
        "cat4_scope": "To port",
        "cat9_scope": "After port",
        "transfer_point": "Ship's side",
    },
    Incoterm.FOB.value: {
        "cat4_scope": "To on board",
        "cat9_scope": "After on board",
        "transfer_point": "Ship's rail",
    },
    Incoterm.CFR.value: {
        "cat4_scope": "Main carriage",
        "cat9_scope": "After discharge",
        "transfer_point": "Destination port",
    },
    Incoterm.CIF.value: {
        "cat4_scope": "Main + insurance",
        "cat9_scope": "After discharge",
        "transfer_point": "Destination port",
    },
    Incoterm.CPT.value: {
        "cat4_scope": "To destination",
        "cat9_scope": "After delivery",
        "transfer_point": "Named place",
    },
    Incoterm.CIP.value: {
        "cat4_scope": "To dest + ins",
        "cat9_scope": "After delivery",
        "transfer_point": "Named place",
    },
    Incoterm.DAP.value: {
        "cat4_scope": "To destination",
        "cat9_scope": "Unloading only",
        "transfer_point": "Named place",
    },
    Incoterm.DPU.value: {
        "cat4_scope": "To unloaded",
        "cat9_scope": "None",
        "transfer_point": "Named place",
    },
    Incoterm.DDP.value: {
        "cat4_scope": "All transport",
        "cat9_scope": "None",
        "transfer_point": "Named place",
    },
}


# ---------------------------------------------------------------------------
# 4.13 DQI_SCORING  (Data Quality Indicators)
# 5 dimensions x 3 reference levels (Score 1 = best, Score 5 = worst).
# ---------------------------------------------------------------------------

DQI_SCORING: Dict[str, Dict[str, str]] = {
    DQIDimension.TECHNOLOGICAL.value: {
        "score_1": "Mode-specific EF",
        "score_3": "Generic mode EF",
        "score_5": "Economy-wide avg",
    },
    DQIDimension.TEMPORAL.value: {
        "score_1": "Current year",
        "score_3": "1-3 years old",
        "score_5": ">5 years old",
    },
    DQIDimension.GEOGRAPHICAL.value: {
        "score_1": "Country-specific",
        "score_3": "Regional",
        "score_5": "Global default",
    },
    DQIDimension.COMPLETENESS.value: {
        "score_1": ">95% coverage",
        "score_3": "50-95% coverage",
        "score_5": "<50% coverage",
    },
    DQIDimension.RELIABILITY.value: {
        "score_1": "Measured data",
        "score_3": "Published avg",
        "score_5": "Expert estimate",
    },
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[str, Decimal] = {
    DQIDimension.TECHNOLOGICAL.value: Decimal("0.25"),
    DQIDimension.TEMPORAL.value: Decimal("0.20"),
    DQIDimension.GEOGRAPHICAL.value: Decimal("0.20"),
    DQIDimension.COMPLETENESS.value: Decimal("0.20"),
    DQIDimension.RELIABILITY.value: Decimal("0.15"),
}


# ---------------------------------------------------------------------------
# 4.14 UNCERTAINTY_RANGES
# Low/high percentage bounds for the 95% confidence interval by method.
# ---------------------------------------------------------------------------

UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.DISTANCE_BASED.value: {
        "low_pct": Decimal("-15"),
        "central_pct": Decimal("0"),
        "high_pct": Decimal("20"),
    },
    CalculationMethod.SPEND_BASED.value: {
        "low_pct": Decimal("-30"),
        "central_pct": Decimal("0"),
        "high_pct": Decimal("40"),
    },
    CalculationMethod.AVERAGE_DATA.value: {
        "low_pct": Decimal("-25"),
        "central_pct": Decimal("0"),
        "high_pct": Decimal("35"),
    },
    CalculationMethod.SUPPLIER_SPECIFIC.value: {
        "low_pct": Decimal("-10"),
        "central_pct": Decimal("0"),
        "high_pct": Decimal("15"),
    },
}


# ==============================================================================
# GWP VALUES (for gas-level disaggregation)
# ==============================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    GWPSource.AR4.value: {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
    },
    GWPSource.AR5.value: {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    GWPSource.AR6.value: {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),
        "n2o": Decimal("273"),
    },
    GWPSource.AR6_20YR.value: {
        "co2": Decimal("1"),
        "ch4": Decimal("81.2"),
        "n2o": Decimal("273"),
    },
}


# ==============================================================================
# DOUBLE-COUNTING PREVENTION RULES
# ==============================================================================

DOUBLE_COUNTING_RULES: Dict[str, Dict[str, str]] = {
    "DC-DTO-001": {
        "description": "Exclude company-paid outbound transport (Cat 4 per Incoterms)",
        "boundary": "Cat 4 vs Cat 9",
    },
    "DC-DTO-002": {
        "description": "Exclude transport in owned/controlled vehicles (Scope 1)",
        "boundary": "Scope 1 vs Cat 9",
    },
    "DC-DTO-003": {
        "description": "Exclude electricity for owned warehouses (Scope 2)",
        "boundary": "Scope 2 vs Cat 9",
    },
    "DC-DTO-004": {
        "description": "Exclude transport included in cradle-to-gate EF (Cat 1)",
        "boundary": "Cat 1 vs Cat 9",
    },
    "DC-DTO-005": {
        "description": "Exclude fuel WTT already counted in Cat 3",
        "boundary": "Cat 3 vs Cat 9",
    },
    "DC-DTO-006": {
        "description": "Exclude distribution of leased assets (Cat 8 or Cat 13)",
        "boundary": "Cat 8/13 vs Cat 9",
    },
    "DC-DTO-007": {
        "description": "Exclude end-of-life transport (Cat 12)",
        "boundary": "Cat 12 vs Cat 9",
    },
    "DC-DTO-008": {
        "description": "Exclude customer use-phase transport (Cat 11)",
        "boundary": "Cat 11 vs Cat 9",
    },
    "DC-DTO-009": {
        "description": "Do not double-count multi-leg segments across methods",
        "boundary": "Internal dedup",
    },
    "DC-DTO-010": {
        "description": "Separate biogenic CO2 from fossil CO2 for biofuel transport",
        "boundary": "Biogenic accounting",
    },
}


# ==============================================================================
# FRAMEWORK-REQUIRED DISCLOSURES
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL.value: [
        "total_co2e",
        "method_used",
        "incoterm_boundary",
        "mode_disclosure",
        "method_hierarchy",
        "ef_sources",
        "exclusions",
        "dqi_score",
        "double_counting_checks",
    ],
    ComplianceFramework.ISO_14064.value: [
        "total_co2e",
        "uncertainty_analysis",
        "boundary_completeness",
        "documentation",
        "methodology",
        "base_year",
        "exclusions",
    ],
    ComplianceFramework.ISO_14083.value: [
        "total_co2e",
        "wtw_mandatory",
        "mode_specific",
        "glec_alignment",
        "chain_of_custody",
        "allocation_method",
    ],
    ComplianceFramework.CSRD_ESRS.value: [
        "total_co2e",
        "esrs_e1_scope3",
        "transport_mode_breakdown",
        "time_series",
        "methodology",
        "targets",
        "actions",
        "dqi_score",
    ],
    ComplianceFramework.CDP.value: [
        "total_co2e",
        "module_c6_5",
        "method_disclosure",
        "relevance_assessment",
        "mode_breakdown",
        "verification_status",
    ],
    ComplianceFramework.SBTI.value: [
        "total_co2e",
        "flag_non_flag",
        "target_boundary",
        "coverage_67pct",
        "progress_tracking",
        "method_used",
    ],
    ComplianceFramework.SB_253.value: [
        "total_co2e",
        "category_9_mandatory",
        "third_party_assurance",
        "carb_format",
        "methodology",
        "assurance_opinion",
    ],
}


# ==============================================================================
# MODE DEFAULT VEHICLE TYPE MAPPING
# When a shipment specifies only a transport mode, resolve to a default
# vehicle type for emission factor lookup.
# ==============================================================================

MODE_DEFAULT_VEHICLE: Dict[str, str] = {
    TransportMode.ROAD.value: VehicleType.ARTICULATED_TRUCK.value,
    TransportMode.RAIL.value: VehicleType.FREIGHT_TRAIN.value,
    TransportMode.MARITIME.value: VehicleType.CONTAINER_SHIP_MEDIUM.value,
    TransportMode.AIR.value: VehicleType.FREIGHTER_WIDE.value,
    TransportMode.INLAND_WATERWAY.value: "barge",
    TransportMode.COURIER.value: "parcel_standard_courier",
    TransportMode.LAST_MILE.value: "same_day_last_mile",
    TransportMode.INTERMODAL.value: VehicleType.INTERMODAL_RAIL.value,
    TransportMode.PIPELINE.value: "barge",  # fallback
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_transport_ef(vehicle_type_key: str) -> Optional[Dict[str, Any]]:
    """Look up transport emission factor entry by vehicle type key.

    Args:
        vehicle_type_key: Key into TRANSPORT_EMISSION_FACTORS (usually
            VehicleType.value or a custom string for non-enum entries).

    Returns:
        Dict containing ef_per_tkm, wtt_per_tkm, source, etc. or None
        if the key is not found.
    """
    return TRANSPORT_EMISSION_FACTORS.get(vehicle_type_key)


def get_cold_chain_uplift(
    temperature_regime: str,
    mode: str,
) -> Decimal:
    """Return cold-chain uplift multiplier for the given regime and mode.

    Falls back to 1.00 (ambient) if the regime or mode is not found.

    Args:
        temperature_regime: TemperatureRegime value string.
        mode: TransportMode value string (road, rail, maritime, air).

    Returns:
        Decimal multiplier (>= 1.00).
    """
    regime_factors = COLD_CHAIN_UPLIFT_FACTORS.get(temperature_regime)
    if regime_factors is None:
        return Decimal("1.00")
    # Normalize mode to base category for cold chain lookup
    mode_key = mode
    if mode in ("courier", "last_mile", "intermodal"):
        mode_key = "road"
    elif mode in ("inland_waterway",):
        mode_key = "maritime"
    elif mode in ("pipeline",):
        mode_key = "road"
    return regime_factors.get(mode_key, Decimal("1.00"))


def get_warehouse_ef(warehouse_type: str) -> Optional[Dict[str, Decimal]]:
    """Look up warehouse emission factor entry by warehouse type.

    Args:
        warehouse_type: WarehouseType value string.

    Returns:
        Dict with electricity_ef, gas_ef, total_ef or None.
    """
    return WAREHOUSE_EMISSION_FACTORS.get(warehouse_type)


def get_last_mile_ef(
    delivery_type: str,
    area_type: str = "urban",
) -> Optional[Decimal]:
    """Look up last-mile emission factor per delivery.

    Args:
        delivery_type: LastMileType value string.
        area_type: One of 'urban', 'suburban', 'rural'.

    Returns:
        Decimal kgCO2e per delivery, or None if not found / not applicable.
    """
    entry = LAST_MILE_EMISSION_FACTORS.get(delivery_type)
    if entry is None:
        return None
    return entry.get(area_type)


def get_eeio_factor(naics_code: str) -> Optional[Decimal]:
    """Look up EEIO emission factor by NAICS code.

    Args:
        naics_code: 6-digit NAICS code string.

    Returns:
        Decimal kgCO2e per USD, or None if not found.
    """
    entry = EEIO_FACTORS.get(naics_code)
    if entry is None:
        return None
    return entry.get("ef_per_usd")


def convert_currency_to_usd(amount: Decimal, currency: str) -> Decimal:
    """Convert a monetary amount to USD using static conversion rates.

    Args:
        amount: Monetary amount in the source currency.
        currency: ISO 4217 currency code (e.g., 'EUR', 'GBP').

    Returns:
        Decimal amount in USD.

    Raises:
        ValueError: If the currency code is not found.
    """
    rate = CURRENCY_CONVERSION_RATES.get(currency.upper())
    if rate is None:
        raise ValueError(
            f"Unsupported currency '{currency}'. "
            f"Supported: {list(CURRENCY_CONVERSION_RATES.keys())}"
        )
    return (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def deflate_spend_to_base_year(
    amount_usd: Decimal,
    spend_year: int,
    base_year: int = 2024,
) -> Decimal:
    """Deflate a USD spend amount from spend_year to base_year using CPI.

    Formula: deflated = amount_usd * (deflator_base / deflator_spend)

    Args:
        amount_usd: Spend in USD at spend_year prices.
        spend_year: Year the spend was incurred.
        base_year: Year to deflate to (default 2024).

    Returns:
        Deflated amount in base-year USD.

    Raises:
        ValueError: If CPI data is not available for the requested years.
    """
    spend_entry = CPI_DEFLATORS.get(spend_year)
    base_entry = CPI_DEFLATORS.get(base_year)
    if spend_entry is None:
        raise ValueError(
            f"CPI deflator not available for year {spend_year}. "
            f"Available: {sorted(CPI_DEFLATORS.keys())}"
        )
    if base_entry is None:
        raise ValueError(
            f"CPI deflator not available for base year {base_year}. "
            f"Available: {sorted(CPI_DEFLATORS.keys())}"
        )
    deflator_ratio = base_entry["deflator"] / spend_entry["deflator"]
    return (amount_usd * deflator_ratio).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def get_return_multiplier(return_type: str) -> Decimal:
    """Return the reverse-logistics emission multiplier.

    Args:
        return_type: ReturnType value string.

    Returns:
        Decimal multiplier (0.00 for no-return, up to 1.00 for recall).
    """
    entry = RETURN_LOGISTICS_FACTORS.get(return_type)
    if entry is None:
        return Decimal("0.00")
    return entry["multiplier"]


def get_load_factor_adjustment(load_factor: str) -> Decimal:
    """Return the load factor adjustment multiplier.

    Args:
        load_factor: LoadFactor value string.

    Returns:
        Decimal adjustment multiplier (1.00 = typical baseline).
    """
    entry = LOAD_FACTOR_ADJUSTMENTS.get(load_factor)
    if entry is None:
        return Decimal("1.00")
    return entry["adjustment"]


def get_grid_ef(country: str) -> Decimal:
    """Return grid electricity emission factor for a country.

    Falls back to GLOBAL default if the country is not found.

    Args:
        country: ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB').

    Returns:
        Decimal kgCO2e per kWh.
    """
    entry = GRID_EMISSION_FACTORS.get(country.upper())
    if entry is None:
        entry = GRID_EMISSION_FACTORS["GLOBAL"]
    return entry["ef_kwh"]


def get_uncertainty_range(method: str) -> Dict[str, Decimal]:
    """Return uncertainty percentage bounds for a calculation method.

    Args:
        method: CalculationMethod value string.

    Returns:
        Dict with 'low_pct', 'central_pct', 'high_pct'.
    """
    entry = UNCERTAINTY_RANGES.get(method)
    if entry is None:
        # Default to average-data uncertainty if method unknown
        entry = UNCERTAINTY_RANGES[CalculationMethod.AVERAGE_DATA.value]
    return entry


def classify_incoterm(incoterm: str) -> Dict[str, str]:
    """Classify an Incoterm into Cat 4 and Cat 9 transport scopes.

    Args:
        incoterm: Incoterm value string (e.g., 'exw', 'fob').

    Returns:
        Dict with 'cat4_scope', 'cat9_scope', 'transfer_point'.

    Raises:
        ValueError: If the Incoterm is not recognized.
    """
    classification = INCOTERM_CLASSIFICATION.get(incoterm.lower())
    if classification is None:
        raise ValueError(
            f"Unrecognized Incoterm '{incoterm}'. "
            f"Supported: {list(INCOTERM_CLASSIFICATION.keys())}"
        )
    return classification


def is_cat9_applicable(incoterm: str) -> bool:
    """Determine whether a shipment has Category 9 scope under the Incoterm.

    Returns True if any downstream transport falls in Category 9 (i.e.,
    cat9_scope is not 'None').

    Args:
        incoterm: Incoterm value string.

    Returns:
        True if Cat 9 scope exists for this Incoterm.
    """
    classification = classify_incoterm(incoterm)
    return classification["cat9_scope"].lower() != "none"


def compute_provenance_hash(data: str) -> str:
    """Compute SHA-256 hex digest for provenance tracking.

    Args:
        data: String representation of data to hash.

    Returns:
        64-character lowercase hex digest string.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ==============================================================================
# PYDANTIC INPUT / RESULT MODELS (12 Models)
# ==============================================================================


class ShipmentInput(GreenLangBase):
    """Input for a single outbound shipment (distance-based method).

    Represents one leg of downstream transport from the reporting company's
    gate toward the end consumer.  The Incoterm determines what portion
    of the shipment falls under Category 9.

    Example:
        >>> shipment = ShipmentInput(
        ...     shipment_id="SHP-001",
        ...     mode=TransportMode.ROAD,
        ...     vehicle_type=VehicleType.ARTICULATED_TRUCK,
        ...     origin="Chicago, IL",
        ...     destination="Dallas, TX",
        ...     distance_km=Decimal("1480"),
        ...     weight_tonnes=Decimal("18.5"),
        ...     incoterm=Incoterm.EXW,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    shipment_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique shipment identifier",
    )
    mode: TransportMode = Field(
        ...,
        description="Transport mode for this leg",
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Specific vehicle/vessel type; defaults to mode default if None",
    )
    origin: str = Field(
        ..., min_length=1, max_length=500,
        description="Origin location description or code",
    )
    destination: str = Field(
        ..., min_length=1, max_length=500,
        description="Destination location description or code",
    )
    distance_km: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Transport distance in kilometres",
    )
    weight_tonnes: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Cargo weight in metric tonnes",
    )
    incoterm: Optional[Incoterm] = Field(
        default=Incoterm.EXW,
        description="ICC Incoterm defining Cat 4 / Cat 9 boundary",
    )
    temperature_regime: TemperatureRegime = Field(
        default=TemperatureRegime.AMBIENT,
        description="Cold chain temperature regime",
    )
    load_factor: LoadFactor = Field(
        default=LoadFactor.TYPICAL,
        description="Vehicle utilization level",
    )
    return_type: ReturnType = Field(
        default=ReturnType.NO_RETURN,
        description="Return logistics type",
    )

    @field_validator("distance_km")
    @classmethod
    def validate_distance_km(cls, v: Decimal) -> Decimal:
        """Ensure distance is a positive value."""
        if v <= Decimal("0"):
            raise ValueError("distance_km must be greater than zero")
        return v

    @field_validator("weight_tonnes")
    @classmethod
    def validate_weight_tonnes(cls, v: Decimal) -> Decimal:
        """Ensure weight is a positive value."""
        if v <= Decimal("0"):
            raise ValueError("weight_tonnes must be greater than zero")
        return v


class SpendInput(GreenLangBase):
    """Input for spend-based (EEIO) emissions calculation.

    Used when detailed shipment data is unavailable.  The agent converts
    the spend to base-year USD, then multiplies by the NAICS-specific
    EEIO emission factor.

    Example:
        >>> spend = SpendInput(
        ...     spend_amount=Decimal("25000"),
        ...     currency="EUR",
        ...     spend_year=2023,
        ...     naics_code="484121",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    spend_amount: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Logistics spend amount in the specified currency",
    )
    currency: str = Field(
        default="USD",
        min_length=3, max_length=3,
        description="ISO 4217 currency code",
    )
    spend_year: int = Field(
        default=2024,
        ge=2015, le=2030,
        description="Year the spend was incurred",
    )
    naics_code: Optional[str] = Field(
        default=None,
        description="6-digit NAICS code for EEIO factor lookup",
    )
    logistics_category: Optional[str] = Field(
        default=None,
        description="Descriptive logistics category when NAICS unknown",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Normalize currency to uppercase."""
        return v.upper()


class WarehouseInput(GreenLangBase):
    """Input for warehouse / distribution center storage emissions.

    Emissions are calculated as:
        Annual = floor_area_m2 * warehouse_ef_per_m2_per_year
        Allocated = Annual * allocation_share * (storage_days / 365)

    Example:
        >>> warehouse = WarehouseInput(
        ...     warehouse_type=WarehouseType.COLD_STORAGE,
        ...     floor_area_m2=Decimal("5000"),
        ...     storage_days=30,
        ...     country="US",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    warehouse_type: WarehouseType = Field(
        ...,
        description="Type of warehouse or distribution facility",
    )
    floor_area_m2: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Total floor area in square metres",
    )
    storage_days: int = Field(
        ..., ge=0, le=365,
        description="Number of days the product is stored",
    )
    country: str = Field(
        default="US",
        min_length=2, max_length=10,
        description="Country code for grid emission factor lookup",
    )
    energy_source: EnergySource = Field(
        default=EnergySource.ELECTRICITY,
        description="Primary energy source for the facility",
    )
    temperature_regime: TemperatureRegime = Field(
        default=TemperatureRegime.AMBIENT,
        description="Temperature regime inside the facility",
    )
    allocation_share: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Share of facility allocated to this product (0.0-1.0)",
    )


class LastMileInput(GreenLangBase):
    """Input for last-mile delivery emissions.

    Emissions are calculated per delivery using area-specific factors:
        Total = num_deliveries * ef_per_delivery[type][area]

    Example:
        >>> lm = LastMileInput(
        ...     delivery_type=LastMileType.PARCEL_STANDARD,
        ...     num_deliveries=10000,
        ...     area_type="urban",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    delivery_type: LastMileType = Field(
        ...,
        description="Type of last-mile delivery service",
    )
    num_deliveries: int = Field(
        ..., gt=0,
        description="Number of deliveries",
    )
    area_type: str = Field(
        default="urban",
        description="Delivery area type: urban, suburban, or rural",
    )
    avg_weight_kg: Optional[Decimal] = Field(
        default=None,
        description="Average parcel weight in kg (for weight-based adjustments)",
    )

    @field_validator("area_type")
    @classmethod
    def validate_area_type(cls, v: str) -> str:
        """Ensure area type is one of the recognized values."""
        allowed = {"urban", "suburban", "rural"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"area_type must be one of {allowed}, got '{v}'"
            )
        return v_lower


class AverageDataInput(GreenLangBase):
    """Input for average-data (screening) method.

    Uses distribution channel defaults to estimate transport distance,
    mode, and storage duration when detailed data is unavailable.

    Example:
        >>> avg = AverageDataInput(
        ...     product_category="Consumer Electronics",
        ...     total_weight_tonnes=Decimal("500"),
        ...     distribution_channel=DistributionChannel.E_COMMERCE,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    product_category: str = Field(
        ..., min_length=1,
        description="Product category description",
    )
    total_weight_tonnes: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Total weight of products distributed in tonnes",
    )
    distribution_channel: DistributionChannel = Field(
        ...,
        description="Distribution channel determining default parameters",
    )
    destination_country: str = Field(
        default="US",
        min_length=2, max_length=10,
        description="Primary destination country code",
    )


class CalculationInput(GreenLangBase):
    """Top-level input for a downstream transportation emissions calculation.

    Combines one or more input types (shipments, spend, warehouse,
    last-mile, average-data) into a single calculation request.
    The pipeline processes all provided inputs and aggregates results.

    Example:
        >>> calc_input = CalculationInput(
        ...     tenant_id="TENANT-001",
        ...     reporting_year=2024,
        ...     calculation_method=CalculationMethod.DISTANCE_BASED,
        ...     shipments=[shipment_1, shipment_2],
        ...     warehouse_inputs=[warehouse_1],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Tenant identifier for multi-tenancy isolation",
    )
    reporting_year: int = Field(
        ..., ge=2000, le=2050,
        description="GHG inventory reporting year",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Primary calculation methodology",
    )
    shipments: Optional[List[ShipmentInput]] = Field(
        default=None,
        description="Outbound shipment records for distance-based method",
    )
    spend_inputs: Optional[List[SpendInput]] = Field(
        default=None,
        description="Logistics spend records for spend-based method",
    )
    warehouse_inputs: Optional[List[WarehouseInput]] = Field(
        default=None,
        description="Warehouse / distribution center inputs",
    )
    last_mile_inputs: Optional[List[LastMileInput]] = Field(
        default=None,
        description="Last-mile delivery inputs",
    )
    average_data_inputs: Optional[List[AverageDataInput]] = Field(
        default=None,
        description="Average-data screening inputs",
    )


class CalculationResult(GreenLangBase):
    """Output from a downstream transportation emissions calculation.

    Contains total and component-level emissions with data quality
    indicators, uncertainty bounds, and a provenance hash for audit.

    All emissions values are in kgCO2e unless the field name
    explicitly indicates tonnes (``_t``).
    """

    calculation_id: str = Field(
        ...,
        description="Unique calculation identifier (UUID)",
    )
    tenant_id: str = Field(
        ...,
        description="Tenant identifier",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    total_emissions_kg: Decimal = Field(
        ...,
        description="Total Scope 3 Cat 9 emissions in kgCO2e",
    )
    total_emissions_t: Decimal = Field(
        ...,
        description="Total Scope 3 Cat 9 emissions in tCO2e",
    )
    transport_emissions_kg: Decimal = Field(
        ...,
        description="Transport component emissions in kgCO2e",
    )
    warehouse_emissions_kg: Decimal = Field(
        ...,
        description="Warehouse / storage component in kgCO2e",
    )
    last_mile_emissions_kg: Decimal = Field(
        ...,
        description="Last-mile delivery component in kgCO2e",
    )
    return_emissions_kg: Decimal = Field(
        ...,
        description="Return / reverse logistics emissions in kgCO2e",
    )
    wtt_emissions_kg: Decimal = Field(
        ...,
        description="Well-to-tank upstream fuel emissions in kgCO2e",
    )
    co2_kg: Decimal = Field(
        ...,
        description="CO2 component in kg",
    )
    ch4_kg: Decimal = Field(
        ...,
        description="CH4 component in kg",
    )
    n2o_kg: Decimal = Field(
        ...,
        description="N2O component in kg",
    )
    dqi_score: Decimal = Field(
        ...,
        description="Weighted Data Quality Indicator score (1-5, 5=best)",
    )
    uncertainty_low_pct: Decimal = Field(
        ...,
        description="Lower bound of 95% confidence interval (negative %)",
    )
    uncertainty_high_pct: Decimal = Field(
        ...,
        description="Upper bound of 95% confidence interval (positive %)",
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance chain hash for audit trail",
    )


class ShipmentResult(GreenLangBase):
    """Emission result for a single shipment leg.

    Produced by the DistanceBasedCalculatorEngine for each ShipmentInput.
    """

    shipment_id: str = Field(
        ...,
        description="Corresponding ShipmentInput identifier",
    )
    mode: TransportMode = Field(
        ...,
        description="Transport mode used",
    )
    distance_km: Decimal = Field(
        ...,
        description="Transport distance in km",
    )
    weight_tonnes: Decimal = Field(
        ...,
        description="Cargo weight in tonnes",
    )
    tonne_km: Decimal = Field(
        ...,
        description="Distance x weight (tonne-km)",
    )
    emissions_kg: Decimal = Field(
        ...,
        description="TTW emissions in kgCO2e",
    )
    wtt_emissions_kg: Decimal = Field(
        ...,
        description="Well-to-tank emissions in kgCO2e",
    )
    cold_chain_uplift: Decimal = Field(
        ...,
        description="Cold chain multiplier applied (1.00 = ambient)",
    )
    return_emissions_kg: Decimal = Field(
        ...,
        description="Return logistics emissions in kgCO2e",
    )
    ef_used: Decimal = Field(
        ...,
        description="Emission factor applied (kgCO2e/tkm)",
    )
    ef_source: EmissionFactorSource = Field(
        ...,
        description="Source database of the emission factor",
    )


class WarehouseResult(GreenLangBase):
    """Emission result for a warehouse / storage facility.

    Produced by the WarehouseDistributionEngine for each WarehouseInput.
    """

    warehouse_type: WarehouseType = Field(
        ...,
        description="Type of warehouse facility",
    )
    floor_area_m2: Decimal = Field(
        ...,
        description="Total floor area in m2",
    )
    storage_days: int = Field(
        ...,
        description="Storage duration in days",
    )
    allocated_emissions_kg: Decimal = Field(
        ...,
        description="Allocated emissions after time and share proration",
    )
    electricity_emissions_kg: Decimal = Field(
        ...,
        description="Electricity-related emissions in kgCO2e",
    )
    heating_emissions_kg: Decimal = Field(
        ...,
        description="Gas/heating-related emissions in kgCO2e",
    )


class ComplianceResult(GreenLangBase):
    """Result from a single-framework compliance check.

    Produced by the ComplianceCheckerEngine for each enabled framework.
    """

    framework: ComplianceFramework = Field(
        ...,
        description="Compliance framework evaluated",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status (pass/fail/warning)",
    )
    score: Decimal = Field(
        ...,
        description="Compliance score (0-100)",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual compliance findings with severity",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving compliance",
    )


class AggregationResult(GreenLangBase):
    """Aggregated emissions broken down by various dimensions.

    Produced by the pipeline's AGGREGATE stage after all component
    emissions have been calculated.
    """

    by_mode: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by transport mode (kgCO2e)",
    )
    by_channel: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by distribution channel (kgCO2e)",
    )
    by_destination: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by destination region (kgCO2e)",
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by calculation method (kgCO2e)",
    )
    transport_pct: Decimal = Field(
        default=Decimal("0"),
        description="Transport share of total emissions (%)",
    )
    warehouse_pct: Decimal = Field(
        default=Decimal("0"),
        description="Warehouse share of total emissions (%)",
    )
    last_mile_pct: Decimal = Field(
        default=Decimal("0"),
        description="Last-mile share of total emissions (%)",
    )


class ProvenanceEntry(GreenLangBase):
    """Single entry in the 10-stage provenance chain.

    Each pipeline stage produces an input hash, output hash, and chain hash
    (SHA-256 of previous chain hash + current output hash), forming an
    immutable Merkle-like chain.
    """

    stage: PipelineStage = Field(
        ...,
        description="Pipeline stage that produced this entry",
    )
    input_hash: str = Field(
        ...,
        description="SHA-256 of stage input data",
    )
    output_hash: str = Field(
        ...,
        description="SHA-256 of stage output data",
    )
    chain_hash: str = Field(
        ...,
        description="SHA-256(previous_chain_hash + output_hash)",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of stage completion",
    )
    duration_ms: Optional[Decimal] = Field(
        default=None,
        description="Stage processing duration in milliseconds",
    )
