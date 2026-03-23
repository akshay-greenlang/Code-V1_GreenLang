"""
Upstream Transportation & Distribution Agent Models (AGENT-MRV-017)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 4
(Upstream Transportation & Distribution) emissions calculations.

Supports:
- 5 calculation methods (distance-based, fuel-based, spend-based, supplier-specific, hybrid)
- 6 transport modes (road, rail, maritime, air, pipeline, intermodal)
- Multi-leg transport chains with hub emissions
- Temperature-controlled transport (reefer)
- Warehouse/distribution center emissions
- Incoterm-based allocation
- ISO 14083, GLEC Framework, CSRD ESRS E1 compliance
- Data quality indicators (DQI) and uncertainty quantification

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.upstream_transportation.models import ShipmentInput, TransportLeg
    >>> leg = TransportLeg(
    ...     mode=TransportMode.ROAD,
    ...     vehicle_type=RoadVehicleType.HGV_RIGID_7_5_17T,
    ...     distance_km=Decimal("250"),
    ...     cargo_mass_tonnes=Decimal("10"),
    ...     laden_state=LadenState.FULL
    ... )
    >>> shipment = ShipmentInput(
    ...     shipment_id="SH-2026-001",
    ...     legs=[leg],
    ...     calculation_method=CalculationMethod.DISTANCE_BASED
    ... )
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator, field_validator, model_validator
import hashlib

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-004"
AGENT_COMPONENT: str = "AGENT-MRV-017"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_uto_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method for transport emissions."""

    DISTANCE_BASED = "distance_based"  # Activity data: distance × mass
    FUEL_BASED = "fuel_based"  # Activity data: fuel consumption
    SPEND_BASED = "spend_based"  # Activity data: transport spend × EEIO
    SUPPLIER_SPECIFIC = "supplier_specific"  # Supplier-provided emissions
    HYBRID = "hybrid"  # Combination of methods


class TransportMode(str, Enum):
    """Primary transport mode."""

    ROAD = "road"
    RAIL = "rail"
    MARITIME = "maritime"
    AIR = "air"
    PIPELINE = "pipeline"
    INTERMODAL = "intermodal"  # Multiple modes in single leg


class RoadVehicleType(str, Enum):
    """Road vehicle types per ISO 14083 and GLEC Framework."""

    # Light commercial vehicles
    LCV_PETROL = "lcv_petrol"  # <3.5t petrol
    LCV_DIESEL = "lcv_diesel"  # <3.5t diesel
    LCV_ELECTRIC = "lcv_electric"  # <3.5t battery electric

    # Heavy goods vehicles - Rigid
    HGV_RIGID_3_5_7_5T = "hgv_rigid_3_5_7_5t"
    HGV_RIGID_7_5_17T = "hgv_rigid_7_5_17t"
    HGV_RIGID_17T_PLUS = "hgv_rigid_17t_plus"

    # Heavy goods vehicles - Articulated
    HGV_ARTIC_3_5_33T = "hgv_artic_3_5_33t"
    HGV_ARTIC_33T_PLUS = "hgv_artic_33t_plus"

    # Alternative fuel HGVs
    HGV_CNG = "hgv_cng"  # Compressed natural gas
    HGV_LNG = "hgv_lng"  # Liquified natural gas
    HGV_ELECTRIC = "hgv_electric"  # Battery electric
    HYDROGEN_TRUCK = "hydrogen_truck"  # Hydrogen fuel cell


class RailType(str, Enum):
    """Rail freight types."""

    DIESEL = "diesel"  # Diesel locomotive
    ELECTRIC = "electric"  # Electric locomotive
    AVERAGE = "average"  # Grid average


class MaritimeVesselType(str, Enum):
    """Maritime vessel types per IMO and GLEC Framework."""

    # Container ships
    CONTAINER_FEEDER = "container_feeder"  # <1,000 TEU
    CONTAINER_SMALL = "container_small"  # 1,000-2,000 TEU
    CONTAINER_PANAMAX = "container_panamax"  # 2,000-5,000 TEU
    CONTAINER_POST_PANAMAX = "container_post_panamax"  # 5,000-10,000 TEU
    CONTAINER_ULCV = "container_ulcv"  # >10,000 TEU (Ultra Large Container Vessel)

    # Bulk carriers
    BULK_HANDYSIZE = "bulk_handysize"  # 10,000-40,000 DWT
    BULK_PANAMAX = "bulk_panamax"  # 60,000-80,000 DWT
    BULK_CAPESIZE = "bulk_capesize"  # >100,000 DWT

    # Tankers
    TANKER_PRODUCT = "tanker_product"  # Refined products
    TANKER_CRUDE = "tanker_crude"  # Crude oil
    TANKER_LNG = "tanker_lng"  # Liquified natural gas
    TANKER_LPG = "tanker_lpg"  # Liquified petroleum gas

    # General cargo
    GENERAL_CARGO = "general_cargo"
    RORO = "roro"  # Roll-on/Roll-off
    REEFER_VESSEL = "reefer_vessel"  # Refrigerated cargo

    # Inland waterway
    INLAND_BARGE = "inland_barge"


class AircraftType(str, Enum):
    """Aircraft types for freight."""

    NARROWBODY_FREIGHTER = "narrowbody_freighter"  # B737F, A321F
    WIDEBODY_FREIGHTER = "widebody_freighter"  # B747F, B777F
    LARGE_FREIGHTER = "large_freighter"  # B747-8F, A380F (rare)
    BELLY_FREIGHT = "belly_freight"  # Passenger aircraft cargo hold
    EXPRESS_INTEGRATOR = "express_integrator"  # FedEx, UPS, DHL fleet


class PipelineType(str, Enum):
    """Pipeline transport types."""

    CRUDE_OIL = "crude_oil"
    REFINED_PRODUCTS = "refined_products"  # Gasoline, diesel, jet fuel
    NATURAL_GAS = "natural_gas"
    CHEMICALS = "chemicals"  # Ethylene, ammonia, etc.
    CO2 = "co2"  # Carbon capture pipelines


class TransportFuelType(str, Enum):
    """Fuel types for transport (primarily fuel-based method)."""

    # Liquid fuels
    DIESEL = "diesel"
    PETROL = "petrol"
    BIODIESEL = "biodiesel"
    HVO = "hvo"  # Hydrotreated vegetable oil
    LPG = "lpg"  # Liquified petroleum gas
    HEAVY_FUEL_OIL = "heavy_fuel_oil"  # Marine bunker fuel
    MARINE_GAS_OIL = "marine_gas_oil"
    JET_FUEL = "jet_fuel"  # Aviation turbine fuel
    SUSTAINABLE_AVIATION_FUEL = "sustainable_aviation_fuel"  # SAF

    # Gaseous fuels
    CNG = "cng"  # Compressed natural gas
    LNG = "lng"  # Liquified natural gas
    HYDROGEN = "hydrogen"  # H2 fuel cell

    # Electric
    ELECTRICITY = "electricity"  # Battery electric

    # Emerging
    METHANOL = "methanol"  # Marine methanol
    AMMONIA = "ammonia"  # Marine ammonia (future)


class LadenState(str, Enum):
    """Load state for return/backhaul journeys."""

    EMPTY = "empty"  # 0% load
    HALF = "half"  # ~50% load
    FULL = "full"  # 100% load
    AVERAGE = "average"  # Industry average (~65% for road)


class AllocationMethod(str, Enum):
    """Method for allocating shared transport emissions."""

    MASS = "mass"  # Allocation by cargo weight (tonnes)
    VOLUME = "volume"  # Allocation by cargo volume (m³)
    PALLET_POSITIONS = "pallet_positions"  # Pallet spaces
    TEU = "teu"  # Twenty-foot equivalent units (containers)
    REVENUE = "revenue"  # Revenue-based allocation
    CHARGEABLE_WEIGHT = "chargeable_weight"  # MAX(actual, volumetric) for air
    FLOOR_AREA = "floor_area"  # Floor space (warehousing)


class Incoterm(str, Enum):
    """
    Incoterms 2020 for determining Category 4 vs Category 9 responsibility.

    Category 4 (Upstream): Transport seller pays
    Category 9 (Downstream): Transport buyer pays
    """

    # Category 4 (Upstream) - Seller pays transport
    EXW = "exw"  # Ex Works - buyer pays ALL (but often Category 9 in practice)
    FCA = "fca"  # Free Carrier
    CPT = "cpt"  # Carriage Paid To
    CIP = "cip"  # Carriage and Insurance Paid
    DAP = "dap"  # Delivered at Place
    DPU = "dpu"  # Delivered at Place Unloaded
    DDP = "ddp"  # Delivered Duty Paid

    # Maritime-specific
    FAS = "fas"  # Free Alongside Ship
    FOB = "fob"  # Free On Board
    CFR = "cfr"  # Cost and Freight
    CIF = "cif"  # Cost, Insurance, and Freight


class HubType(str, Enum):
    """Type of distribution/logistics hub."""

    WAREHOUSE = "warehouse"  # Standard warehousing
    COLD_STORAGE = "cold_storage"  # Refrigerated warehouse
    FROZEN_STORAGE = "frozen_storage"  # Deep-freeze warehouse
    CROSS_DOCK = "cross_dock"  # Minimal storage, rapid transfer
    DISTRIBUTION_CENTER = "distribution_center"  # Regional DC
    FULFILLMENT_CENTER = "fulfillment_center"  # E-commerce FC
    CONSOLIDATION_CENTER = "consolidation_center"  # Freight consolidation
    TRANSSHIPMENT_HUB = "transshipment_hub"  # Port/airport transfer


class TemperatureControl(str, Enum):
    """Temperature control requirements (reefer transport)."""

    AMBIENT = "ambient"  # No temperature control
    CHILLED_2_8C = "chilled_2_8c"  # 2-8°C (pharmaceuticals, fresh food)
    FROZEN_MINUS_18C = "frozen_minus_18c"  # -18°C (frozen food)
    DEEP_FROZEN_MINUS_25C = "deep_frozen_minus_25c"  # -25°C (vaccines, biotech)
    HEATED = "heated"  # Heated transport (chemicals)


class EFScope(str, Enum):
    """Emission factor scope (WTW = Well-to-Wheel)."""

    TTW = "ttw"  # Tank-to-Wheel (direct combustion)
    WTT = "wtt"  # Well-to-Tank (upstream fuel production)
    WTW = "wtw"  # Well-to-Wheel (TTW + WTT)


class DistanceMethod(str, Enum):
    """Method for determining transport distance."""

    ACTUAL = "actual"  # GPS/telematics actual distance
    SHORTEST_FEASIBLE = "shortest_feasible"  # Routing engine (OSM, Google, HERE)
    GREAT_CIRCLE = "great_circle"  # Haversine formula (air, maritime)
    ESTIMATED = "estimated"  # Industry average or modeled


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based method."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    SGD = "SGD"
    HKD = "HKD"
    NZD = "NZD"
    KRW = "KRW"
    BRL = "BRL"
    MXN = "MXN"
    ZAR = "ZAR"
    AED = "AED"


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per ISO 14083."""

    RELIABILITY = "reliability"  # Source reliability
    COMPLETENESS = "completeness"  # Data completeness
    TEMPORAL = "temporal"  # Temporal correlation
    GEOGRAPHICAL = "geographical"  # Geographical correlation
    TECHNOLOGICAL = "technological"  # Technological correlation


class DQIScore(str, Enum):
    """Data Quality Indicator scores (1-5 scale)."""

    VERY_GOOD = "very_good"  # 1
    GOOD = "good"  # 2
    FAIR = "fair"  # 3
    POOR = "poor"  # 4
    VERY_POOR = "very_poor"  # 5


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    PEDIGREE_MATRIX = "pedigree_matrix"  # ISO 14083 pedigree matrix


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework."""

    GHG_PROTOCOL_SCOPE3 = "ghg_protocol_scope3"
    ISO_14083 = "iso_14083"  # Quantification and reporting of GHG emissions from transport
    GLEC_FRAMEWORK = "glec_framework"  # Global Logistics Emissions Council
    CSRD_ESRS_E1 = "csrd_esrs_e1"  # EU CSRD ESRS E1-6
    CDP = "cdp"  # CDP Climate Change
    SBTI = "sbti"  # Science Based Targets initiative
    GRI_305 = "gri_305"  # GRI 305 Emissions


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class PipelineStage(str, Enum):
    """Processing pipeline stages."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Transport classification
    NORMALIZE = "normalize"  # Unit normalization
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE_LEGS = "calculate_legs"  # Leg-level calculations
    CALCULATE_HUBS = "calculate_hubs"  # Hub emissions
    ALLOCATE = "allocate"  # Emission allocation
    COMPLIANCE = "compliance"  # Compliance checks
    AGGREGATE = "aggregate"  # Aggregation
    SEAL = "seal"  # Provenance sealing


class ExportFormat(str, Enum):
    """Export format for results."""

    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"


class BatchStatus(str, Enum):
    """Batch calculation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some records failed


class GWPSource(str, Enum):
    """IPCC Global Warming Potential source."""

    AR4 = "ar4"  # Fourth Assessment Report (100-year)
    AR5 = "ar5"  # Fifth Assessment Report (100-year)
    AR6 = "ar6"  # Sixth Assessment Report (100-year)
    AR6_20YR = "ar6_20yr"  # Sixth Assessment Report (20-year)


class EmissionGas(str, Enum):
    """Greenhouse gas types."""

    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFC = "hfc"  # Hydrofluorocarbons (reefer units)
    SF6 = "sf6"  # Sulfur hexafluoride (rare in transport)


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Global Warming Potential (100-year unless stated)
GWP_VALUES: Dict[GWPSource, Dict[EmissionGas, Decimal]] = {
    GWPSource.AR4: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("25"),
        EmissionGas.N2O: Decimal("298"),
        EmissionGas.HFC: Decimal("1430"),  # HFC-134a (common in reefer)
    },
    GWPSource.AR5: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("28"),
        EmissionGas.N2O: Decimal("265"),
        EmissionGas.HFC: Decimal("1300"),  # HFC-134a
    },
    GWPSource.AR6: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("29.8"),  # Fossil CH4 with climate-carbon feedback
        EmissionGas.N2O: Decimal("273"),
        EmissionGas.HFC: Decimal("1530"),  # HFC-134a
    },
    GWPSource.AR6_20YR: {
        EmissionGas.CO2: Decimal("1"),
        EmissionGas.CH4: Decimal("81.2"),  # Fossil CH4, 20-year
        EmissionGas.N2O: Decimal("273"),  # Same as 100-year
        EmissionGas.HFC: Decimal("4144"),  # HFC-134a, 20-year
    },
}

# Road emission factors (kgCO2e per tonne-km) - WTW basis
# Source: GLEC Framework v3.0, DEFRA 2023
ROAD_EMISSION_FACTORS: Dict[RoadVehicleType, Dict[str, Decimal]] = {
    RoadVehicleType.LCV_PETROL: {
        "co2_per_tkm": Decimal("0.512"),
        "ch4_per_tkm": Decimal("0.0008"),
        "n2o_per_tkm": Decimal("0.0012"),
        "total_per_tkm": Decimal("0.514"),
        "laden_full": Decimal("1.0"),  # Multiplier
        "laden_half": Decimal("1.15"),
        "laden_empty": Decimal("1.35"),
    },
    RoadVehicleType.LCV_DIESEL: {
        "co2_per_tkm": Decimal("0.483"),
        "ch4_per_tkm": Decimal("0.0003"),
        "n2o_per_tkm": Decimal("0.0010"),
        "total_per_tkm": Decimal("0.484"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.15"),
        "laden_empty": Decimal("1.35"),
    },
    RoadVehicleType.LCV_ELECTRIC: {
        "co2_per_tkm": Decimal("0.142"),  # WTW (grid emissions)
        "ch4_per_tkm": Decimal("0.0001"),
        "n2o_per_tkm": Decimal("0.0001"),
        "total_per_tkm": Decimal("0.142"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.12"),
        "laden_empty": Decimal("1.28"),
    },
    RoadVehicleType.HGV_RIGID_3_5_7_5T: {
        "co2_per_tkm": Decimal("0.387"),
        "ch4_per_tkm": Decimal("0.0003"),
        "n2o_per_tkm": Decimal("0.0009"),
        "total_per_tkm": Decimal("0.388"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.18"),
        "laden_empty": Decimal("1.42"),
    },
    RoadVehicleType.HGV_RIGID_7_5_17T: {
        "co2_per_tkm": Decimal("0.244"),
        "ch4_per_tkm": Decimal("0.0002"),
        "n2o_per_tkm": Decimal("0.0006"),
        "total_per_tkm": Decimal("0.245"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.20"),
        "laden_empty": Decimal("1.48"),
    },
    RoadVehicleType.HGV_RIGID_17T_PLUS: {
        "co2_per_tkm": Decimal("0.172"),
        "ch4_per_tkm": Decimal("0.0002"),
        "n2o_per_tkm": Decimal("0.0004"),
        "total_per_tkm": Decimal("0.172"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.22"),
        "laden_empty": Decimal("1.52"),
    },
    RoadVehicleType.HGV_ARTIC_3_5_33T: {
        "co2_per_tkm": Decimal("0.118"),
        "ch4_per_tkm": Decimal("0.0001"),
        "n2o_per_tkm": Decimal("0.0003"),
        "total_per_tkm": Decimal("0.118"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.25"),
        "laden_empty": Decimal("1.58"),
    },
    RoadVehicleType.HGV_ARTIC_33T_PLUS: {
        "co2_per_tkm": Decimal("0.085"),
        "ch4_per_tkm": Decimal("0.0001"),
        "n2o_per_tkm": Decimal("0.0002"),
        "total_per_tkm": Decimal("0.085"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.28"),
        "laden_empty": Decimal("1.65"),
    },
    RoadVehicleType.HGV_CNG: {
        "co2_per_tkm": Decimal("0.095"),
        "ch4_per_tkm": Decimal("0.0024"),  # Higher methane slip
        "n2o_per_tkm": Decimal("0.0002"),
        "total_per_tkm": Decimal("0.097"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.26"),
        "laden_empty": Decimal("1.60"),
    },
    RoadVehicleType.HGV_LNG: {
        "co2_per_tkm": Decimal("0.092"),
        "ch4_per_tkm": Decimal("0.0020"),
        "n2o_per_tkm": Decimal("0.0002"),
        "total_per_tkm": Decimal("0.094"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.25"),
        "laden_empty": Decimal("1.58"),
    },
    RoadVehicleType.HGV_ELECTRIC: {
        "co2_per_tkm": Decimal("0.032"),  # WTW (grid emissions)
        "ch4_per_tkm": Decimal("0.0001"),
        "n2o_per_tkm": Decimal("0.0001"),
        "total_per_tkm": Decimal("0.032"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.15"),
        "laden_empty": Decimal("1.35"),
    },
    RoadVehicleType.HYDROGEN_TRUCK: {
        "co2_per_tkm": Decimal("0.048"),  # WTW (H2 production emissions)
        "ch4_per_tkm": Decimal("0.0002"),
        "n2o_per_tkm": Decimal("0.0001"),
        "total_per_tkm": Decimal("0.048"),
        "laden_full": Decimal("1.0"),
        "laden_half": Decimal("1.18"),
        "laden_empty": Decimal("1.42"),
    },
}

# Rail emission factors (kgCO2e per tonne-km) - WTW basis
RAIL_EMISSION_FACTORS: Dict[Tuple[RailType, str], Decimal] = {
    # (rail_type, region): kgCO2e/tkm
    (RailType.DIESEL, "global"): Decimal("0.0224"),
    (RailType.DIESEL, "us"): Decimal("0.0215"),
    (RailType.DIESEL, "eu"): Decimal("0.0232"),
    (RailType.DIESEL, "china"): Decimal("0.0241"),
    (RailType.ELECTRIC, "global"): Decimal("0.0088"),  # Avg grid
    (RailType.ELECTRIC, "us"): Decimal("0.0115"),  # US grid
    (RailType.ELECTRIC, "eu"): Decimal("0.0062"),  # EU grid (lower carbon)
    (RailType.ELECTRIC, "china"): Decimal("0.0142"),  # China grid (coal-heavy)
    (RailType.AVERAGE, "global"): Decimal("0.0156"),  # Weighted average
}

# Maritime emission factors (kgCO2e per tonne-km) - WTW basis
# Source: IMO Fourth GHG Study 2020, GLEC Framework v3.0
MARITIME_EMISSION_FACTORS: Dict[MaritimeVesselType, Decimal] = {
    MaritimeVesselType.CONTAINER_FEEDER: Decimal("0.0183"),
    MaritimeVesselType.CONTAINER_SMALL: Decimal("0.0142"),
    MaritimeVesselType.CONTAINER_PANAMAX: Decimal("0.0098"),
    MaritimeVesselType.CONTAINER_POST_PANAMAX: Decimal("0.0072"),
    MaritimeVesselType.CONTAINER_ULCV: Decimal("0.0054"),  # Most efficient
    MaritimeVesselType.BULK_HANDYSIZE: Decimal("0.0115"),
    MaritimeVesselType.BULK_PANAMAX: Decimal("0.0087"),
    MaritimeVesselType.BULK_CAPESIZE: Decimal("0.0064"),
    MaritimeVesselType.TANKER_PRODUCT: Decimal("0.0102"),
    MaritimeVesselType.TANKER_CRUDE: Decimal("0.0078"),
    MaritimeVesselType.TANKER_LNG: Decimal("0.0118"),  # Higher due to boil-off
    MaritimeVesselType.TANKER_LPG: Decimal("0.0095"),
    MaritimeVesselType.GENERAL_CARGO: Decimal("0.0165"),
    MaritimeVesselType.RORO: Decimal("0.0124"),
    MaritimeVesselType.REEFER_VESSEL: Decimal("0.0215"),  # Higher due to refrigeration
    MaritimeVesselType.INLAND_BARGE: Decimal("0.0312"),
}

# Air emission factors (kgCO2e per tonne-km) - WTW basis
# Source: ICAO Carbon Calculator, GLEC Framework v3.0
AIR_EMISSION_FACTORS: Dict[AircraftType, Decimal] = {
    AircraftType.NARROWBODY_FREIGHTER: Decimal("1.248"),
    AircraftType.WIDEBODY_FREIGHTER: Decimal("0.842"),
    AircraftType.LARGE_FREIGHTER: Decimal("0.726"),
    AircraftType.BELLY_FREIGHT: Decimal("0.952"),  # Allocated from pax flights
    AircraftType.EXPRESS_INTEGRATOR: Decimal("1.456"),  # Less efficient (many stops)
}

# Pipeline emission factors (kgCO2e per tonne-km) - WTW basis
PIPELINE_EMISSION_FACTORS: Dict[PipelineType, Decimal] = {
    PipelineType.CRUDE_OIL: Decimal("0.0042"),
    PipelineType.REFINED_PRODUCTS: Decimal("0.0038"),
    PipelineType.NATURAL_GAS: Decimal("0.0028"),  # Per tonne of LNG equivalent
    PipelineType.CHEMICALS: Decimal("0.0052"),
    PipelineType.CO2: Decimal("0.0015"),  # Ironic but true
}

# Fuel emission factors (kgCO2e per litre or kg)
# Source: IPCC 2006, DEFRA 2023
FUEL_EMISSION_FACTORS: Dict[TransportFuelType, Dict[str, Decimal]] = {
    TransportFuelType.DIESEL: {
        "ttw_kg_per_litre": Decimal("2.687"),  # Tank-to-Wheel
        "wtt_kg_per_litre": Decimal("0.587"),  # Well-to-Tank
        "wtw_kg_per_litre": Decimal("3.274"),  # Total
        "density_kg_per_litre": Decimal("0.845"),
    },
    TransportFuelType.PETROL: {
        "ttw_kg_per_litre": Decimal("2.315"),
        "wtt_kg_per_litre": Decimal("0.532"),
        "wtw_kg_per_litre": Decimal("2.847"),
        "density_kg_per_litre": Decimal("0.742"),
    },
    TransportFuelType.BIODIESEL: {
        "ttw_kg_per_litre": Decimal("2.650"),
        "wtt_kg_per_litre": Decimal("0.398"),  # Lower upstream
        "wtw_kg_per_litre": Decimal("3.048"),
        "density_kg_per_litre": Decimal("0.880"),
    },
    TransportFuelType.HVO: {
        "ttw_kg_per_litre": Decimal("2.680"),
        "wtt_kg_per_litre": Decimal("0.285"),  # Much lower upstream
        "wtw_kg_per_litre": Decimal("2.965"),
        "density_kg_per_litre": Decimal("0.780"),
    },
    TransportFuelType.LPG: {
        "ttw_kg_per_litre": Decimal("1.644"),
        "wtt_kg_per_litre": Decimal("0.342"),
        "wtw_kg_per_litre": Decimal("1.986"),
        "density_kg_per_litre": Decimal("0.538"),
    },
    TransportFuelType.HEAVY_FUEL_OIL: {
        "ttw_kg_per_litre": Decimal("3.114"),
        "wtt_kg_per_litre": Decimal("0.542"),
        "wtw_kg_per_litre": Decimal("3.656"),
        "density_kg_per_litre": Decimal("0.975"),
    },
    TransportFuelType.MARINE_GAS_OIL: {
        "ttw_kg_per_litre": Decimal("2.760"),
        "wtt_kg_per_litre": Decimal("0.598"),
        "wtw_kg_per_litre": Decimal("3.358"),
        "density_kg_per_litre": Decimal("0.890"),
    },
    TransportFuelType.JET_FUEL: {
        "ttw_kg_per_litre": Decimal("2.544"),
        "wtt_kg_per_litre": Decimal("0.612"),
        "wtw_kg_per_litre": Decimal("3.156"),
        "density_kg_per_litre": Decimal("0.804"),
    },
    TransportFuelType.SUSTAINABLE_AVIATION_FUEL: {
        "ttw_kg_per_litre": Decimal("2.544"),  # Same combustion
        "wtt_kg_per_litre": Decimal("0.198"),  # 70-80% lower upstream
        "wtw_kg_per_litre": Decimal("2.742"),
        "density_kg_per_litre": Decimal("0.804"),
    },
    TransportFuelType.CNG: {
        "ttw_kg_per_kg": Decimal("2.750"),  # Per kg of CNG
        "wtt_kg_per_kg": Decimal("0.485"),
        "wtw_kg_per_kg": Decimal("3.235"),
    },
    TransportFuelType.LNG: {
        "ttw_kg_per_kg": Decimal("2.750"),  # Per kg of LNG
        "wtt_kg_per_kg": Decimal("0.624"),  # Higher due to liquefaction
        "wtw_kg_per_kg": Decimal("3.374"),
    },
    TransportFuelType.HYDROGEN: {
        "ttw_kg_per_kg": Decimal("0.0"),  # Zero direct emissions
        "wtt_kg_per_kg": Decimal("10.8"),  # High (SMR production)
        "wtw_kg_per_kg": Decimal("10.8"),
        # Green H2: wtt ~1-2 kgCO2e/kg (electrolysis from renewables)
    },
    TransportFuelType.ELECTRICITY: {
        "ttw_kwh": Decimal("0.0"),  # Zero direct
        "wtt_kwh_global": Decimal("0.475"),  # Global grid average
        "wtt_kwh_us": Decimal("0.417"),  # US grid
        "wtt_kwh_eu": Decimal("0.295"),  # EU grid
        "wtt_kwh_china": Decimal("0.581"),  # China grid
    },
    TransportFuelType.METHANOL: {
        "ttw_kg_per_litre": Decimal("1.375"),
        "wtt_kg_per_litre": Decimal("0.842"),
        "wtw_kg_per_litre": Decimal("2.217"),
        "density_kg_per_litre": Decimal("0.792"),
    },
    TransportFuelType.AMMONIA: {
        "ttw_kg_per_kg": Decimal("0.0"),  # Zero CO2 combustion (produces N2/H2O)
        "wtt_kg_per_kg": Decimal("2.4"),  # Haber-Bosch production
        "wtw_kg_per_kg": Decimal("2.4"),
        # Green ammonia: wtt ~0.5 kgCO2e/kg
    },
}

# EEIO (Environmentally Extended Input-Output) factors for spend-based method
# Source: USEEIO v2.0, EXIOBASE v3.8
# Units: kgCO2e per USD (2023 prices)
EEIO_TRANSPORT_FACTORS: Dict[str, Decimal] = {
    "NAICS_484_truck_transport": Decimal("0.687"),
    "NAICS_482_rail_transport": Decimal("0.412"),
    "NAICS_483_water_transport": Decimal("0.524"),
    "NAICS_481_air_transport": Decimal("1.842"),
    "NAICS_486_pipeline_transport": Decimal("0.285"),
    "NAICS_493_warehousing": Decimal("0.398"),
    "NAICS_488_support_activities": Decimal("0.542"),
}

# Hub/warehouse emission factors (kgCO2e per tonne handled)
# Source: CEFIC Guidelines, GLEC Framework
HUB_EMISSION_FACTORS: Dict[HubType, Decimal] = {
    HubType.WAREHOUSE: Decimal("0.85"),  # Standard warehouse
    HubType.COLD_STORAGE: Decimal("3.42"),  # Refrigeration energy
    HubType.FROZEN_STORAGE: Decimal("5.68"),  # Deep-freeze energy
    HubType.CROSS_DOCK: Decimal("0.24"),  # Minimal storage
    HubType.DISTRIBUTION_CENTER: Decimal("1.12"),
    HubType.FULFILLMENT_CENTER: Decimal("1.85"),  # Automation energy
    HubType.CONSOLIDATION_CENTER: Decimal("0.68"),
    HubType.TRANSSHIPMENT_HUB: Decimal("0.52"),
}

# Reefer (temperature-controlled) uplift factors
# Multiplier on base emission factor for temperature-controlled transport
REEFER_UPLIFT_FACTORS: Dict[TransportMode, Dict[TemperatureControl, Decimal]] = {
    TransportMode.ROAD: {
        TemperatureControl.AMBIENT: Decimal("1.0"),
        TemperatureControl.CHILLED_2_8C: Decimal("1.35"),
        TemperatureControl.FROZEN_MINUS_18C: Decimal("1.62"),
        TemperatureControl.DEEP_FROZEN_MINUS_25C: Decimal("1.85"),
        TemperatureControl.HEATED: Decimal("1.22"),
    },
    TransportMode.MARITIME: {
        TemperatureControl.AMBIENT: Decimal("1.0"),
        TemperatureControl.CHILLED_2_8C: Decimal("1.45"),
        TemperatureControl.FROZEN_MINUS_18C: Decimal("1.78"),
        TemperatureControl.DEEP_FROZEN_MINUS_25C: Decimal("2.05"),
        TemperatureControl.HEATED: Decimal("1.18"),
    },
    TransportMode.AIR: {
        TemperatureControl.AMBIENT: Decimal("1.0"),
        TemperatureControl.CHILLED_2_8C: Decimal("1.28"),
        TemperatureControl.FROZEN_MINUS_18C: Decimal("1.52"),
        TemperatureControl.DEEP_FROZEN_MINUS_25C: Decimal("1.72"),
        TemperatureControl.HEATED: Decimal("1.15"),
    },
}

# Load factor defaults (% of capacity utilized)
LOAD_FACTOR_DEFAULTS: Dict[TransportMode, Decimal] = {
    TransportMode.ROAD: Decimal("0.65"),  # 65% average load
    TransportMode.RAIL: Decimal("0.72"),
    TransportMode.MARITIME: Decimal("0.78"),
    TransportMode.AIR: Decimal("0.68"),
    TransportMode.PIPELINE: Decimal("0.85"),  # Typically high
}

# Empty running rates (% of km traveled empty)
EMPTY_RUNNING_RATES: Dict[TransportMode, Decimal] = {
    TransportMode.ROAD: Decimal("0.28"),  # 28% of road freight km are empty
    TransportMode.RAIL: Decimal("0.15"),
    TransportMode.MARITIME: Decimal("0.12"),  # Ballast voyages
    TransportMode.AIR: Decimal("0.08"),  # Rare for dedicated freighters
}

# Warehouse energy intensities (kWh per m² per year)
WAREHOUSE_ENERGY_INTENSITIES: Dict[str, Decimal] = {
    "standard": Decimal("65"),  # kWh/m²/yr
    "cold_storage": Decimal("285"),
    "frozen_storage": Decimal("425"),
}

# Incoterm to Category mapping
INCOTERM_CATEGORY_MAP: Dict[Incoterm, str] = {
    # Category 4 (Upstream) - Seller arranges and pays for main carriage
    Incoterm.CPT: "CATEGORY_4",
    Incoterm.CIP: "CATEGORY_4",
    Incoterm.CFR: "CATEGORY_4",
    Incoterm.CIF: "CATEGORY_4",
    Incoterm.DAP: "CATEGORY_4",
    Incoterm.DPU: "CATEGORY_4",
    Incoterm.DDP: "CATEGORY_4",

    # Category 9 (Downstream) - Buyer arranges and pays for main carriage
    Incoterm.EXW: "CATEGORY_9",  # Buyer responsible from seller's premises
    Incoterm.FCA: "CATEGORY_9",  # Buyer responsible from carrier
    Incoterm.FAS: "CATEGORY_9",  # Buyer responsible from alongside ship
    Incoterm.FOB: "CATEGORY_9",  # Buyer responsible from on board ship
}

# DQI score numeric values (1=best, 5=worst)
DQI_SCORE_VALUES: Dict[DQIScore, Decimal] = {
    DQIScore.VERY_GOOD: Decimal("1"),
    DQIScore.GOOD: Decimal("2"),
    DQIScore.FAIR: Decimal("3"),
    DQIScore.POOR: Decimal("4"),
    DQIScore.VERY_POOR: Decimal("5"),
}

# DQI composite score to quality tier
DQI_QUALITY_TIERS: Dict[str, Tuple[Decimal, Decimal]] = {
    "tier_1_excellent": (Decimal("1.0"), Decimal("1.5")),  # DQI 1.0-1.5
    "tier_2_good": (Decimal("1.5"), Decimal("2.5")),  # DQI 1.5-2.5
    "tier_3_fair": (Decimal("2.5"), Decimal("3.5")),  # DQI 2.5-3.5
    "tier_4_poor": (Decimal("3.5"), Decimal("4.5")),  # DQI 3.5-4.5
    "tier_5_very_poor": (Decimal("4.5"), Decimal("5.0")),  # DQI 4.5-5.0
}

# Uncertainty ranges by calculation method (± % at 95% confidence)
UNCERTAINTY_RANGES: Dict[CalculationMethod, Decimal] = {
    CalculationMethod.DISTANCE_BASED: Decimal("0.25"),  # ±25%
    CalculationMethod.FUEL_BASED: Decimal("0.15"),  # ±15% (more accurate)
    CalculationMethod.SPEND_BASED: Decimal("0.50"),  # ±50% (least accurate)
    CalculationMethod.SUPPLIER_SPECIFIC: Decimal("0.20"),  # ±20% (depends on supplier data quality)
    CalculationMethod.HYBRID: Decimal("0.22"),  # ±22% (weighted average)
}

# Framework-specific required disclosures
FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL_SCOPE3: [
        "total_scope3_cat4_emissions_tco2e",
        "calculation_methodology",
        "data_quality_assessment",
        "exclusions_with_justification",
    ],
    ComplianceFramework.ISO_14083: [
        "transport_activity_data",
        "emission_factors_with_source",
        "allocation_method",
        "dqi_scores_by_dimension",
        "uncertainty_quantification",
        "system_boundary",
    ],
    ComplianceFramework.GLEC_FRAMEWORK: [
        "shipment_level_emissions",
        "wtw_emissions_breakdown",
        "hub_emissions_separate",
        "reefer_uplift_applied",
        "glec_version_used",
    ],
    ComplianceFramework.CSRD_ESRS_E1: [
        "scope3_category4_absolute_emissions",
        "scope3_category4_intensity_per_revenue",
        "year_over_year_change",
        "reduction_targets",
        "materiality_assessment",
    ],
    ComplianceFramework.CDP: [
        "scope3_cat4_total",
        "percentage_of_total_scope3",
        "verification_status",
        "improvement_initiatives",
    ],
    ComplianceFramework.SBTI: [
        "scope3_cat4_baseline_year",
        "scope3_cat4_target_year",
        "reduction_trajectory",
        "sbti_approved_target",
    ],
    ComplianceFramework.GRI_305: [
        "gri_305_3_scope3_cat4",
        "calculation_methodology_reference",
        "emission_factors_source",
    ],
}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================


class TransportEmissionFactor(BaseModel):
    """
    Transport emission factor with metadata.

    Used for custom/supplier-specific emission factors beyond standard tables.
    """

    factor_id: str = Field(..., description="Unique EF identifier")
    mode: TransportMode
    vehicle_subtype: Optional[str] = Field(None, description="Vehicle subtype (e.g., 'euro_6_diesel')")
    fuel_type: Optional[TransportFuelType] = None
    region: str = Field(default="global", description="Geographic region")
    ef_per_tkm: Decimal = Field(..., ge=0, description="kgCO2e per tonne-km")
    ef_scope: EFScope = Field(default=EFScope.WTW)
    source: str = Field(..., description="Source (e.g., 'GLEC v3.0', 'Supplier X')")
    year: int = Field(..., ge=2000, le=2030, description="Data year")
    valid_from: date
    valid_to: Optional[date] = None

    class Config:
        frozen = True


class RoadVehicleProfile(BaseModel):
    """Road vehicle operational profile."""

    vehicle_type: RoadVehicleType
    fuel_type: TransportFuelType
    euro_standard: Optional[str] = Field(None, description="Euro 6, Euro 5, etc.")
    payload_capacity_tonnes: Decimal = Field(..., gt=0)
    average_load_factor: Decimal = Field(default=Decimal("0.65"), ge=0, le=1)
    empty_running_rate: Decimal = Field(default=Decimal("0.28"), ge=0, le=1)
    fuel_efficiency_litres_per_100km: Optional[Decimal] = Field(None, gt=0)

    class Config:
        frozen = True


class MaritimeVesselProfile(BaseModel):
    """Maritime vessel operational profile."""

    vessel_type: MaritimeVesselType
    fuel_type: TransportFuelType = Field(default=TransportFuelType.HEAVY_FUEL_OIL)
    dwt: Optional[Decimal] = Field(None, gt=0, description="Deadweight tonnage")
    teu_capacity: Optional[int] = Field(None, gt=0, description="TEU capacity (containers)")
    speed_knots: Decimal = Field(default=Decimal("15"), gt=0)
    imo_eedi_rating: Optional[str] = Field(None, description="Energy Efficiency Design Index")

    class Config:
        frozen = True


class AircraftProfile(BaseModel):
    """Aircraft operational profile."""

    aircraft_type: AircraftType
    fuel_type: TransportFuelType = Field(default=TransportFuelType.JET_FUEL)
    cargo_capacity_tonnes: Decimal = Field(..., gt=0)
    average_load_factor: Decimal = Field(default=Decimal("0.68"), ge=0, le=1)
    cruise_speed_kmh: Decimal = Field(default=Decimal("850"), gt=0)

    class Config:
        frozen = True


class TransportLeg(BaseModel):
    """
    Single leg of a transport chain.

    A leg is one continuous journey by one mode of transport (e.g., truck from A to B).
    """

    leg_id: str = Field(default_factory=lambda: f"leg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    mode: TransportMode

    # Vehicle/vessel details
    vehicle_type: Optional[str] = Field(None, description="RoadVehicleType, MaritimeVesselType, etc.")
    vehicle_profile: Optional[Dict[str, Any]] = Field(None, description="Detailed vehicle profile")

    # Route details
    origin: str = Field(..., description="Origin location (name, code, coordinates)")
    destination: str = Field(..., description="Destination location")
    distance_km: Decimal = Field(..., gt=0, description="Distance in kilometers")
    distance_method: DistanceMethod = Field(default=DistanceMethod.SHORTEST_FEASIBLE)

    # Cargo details
    cargo_mass_tonnes: Decimal = Field(..., gt=0, description="Cargo mass for THIS shipment")
    cargo_volume_m3: Optional[Decimal] = Field(None, gt=0, description="Cargo volume (for allocation)")
    laden_state: LadenState = Field(default=LadenState.FULL)

    # Temperature control
    temperature_control: TemperatureControl = Field(default=TemperatureControl.AMBIENT)

    # Fuel (for fuel-based method)
    fuel_type: Optional[TransportFuelType] = None
    fuel_consumed_litres: Optional[Decimal] = Field(None, gt=0)
    fuel_consumed_kg: Optional[Decimal] = Field(None, gt=0)
    electricity_consumed_kwh: Optional[Decimal] = Field(None, gt=0)

    # Emission factor override
    custom_ef_per_tkm: Optional[Decimal] = Field(None, ge=0, description="Custom kgCO2e/tkm")
    ef_source: Optional[str] = Field(None, description="Source of custom EF")

    # Allocation (if shared transport)
    total_vehicle_capacity_tonnes: Optional[Decimal] = Field(None, gt=0)
    load_factor: Optional[Decimal] = Field(None, ge=0, le=1)
    allocation_method: AllocationMethod = Field(default=AllocationMethod.MASS)
    allocation_percentage: Optional[Decimal] = Field(None, ge=0, le=100, description="% of vehicle allocated to this shipment")

    # Metadata
    carrier_name: Optional[str] = None
    service_type: Optional[str] = Field(None, description="Express, Standard, Economy")
    departure_date: Optional[date] = None
    arrival_date: Optional[date] = None

    class Config:
        frozen = True

    @field_validator("allocation_percentage")
    @classmethod
    def validate_allocation(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        if v is not None and (v < 0 or v > 100):
            raise ValueError("allocation_percentage must be 0-100")
        return v


class TransportHub(BaseModel):
    """
    Hub/warehouse/distribution center in transport chain.

    Emissions from storage, handling, refrigeration.
    """

    hub_id: str = Field(default_factory=lambda: f"hub_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    hub_type: HubType
    location: str = Field(..., description="Hub location (name, code)")

    # Throughput
    cargo_mass_tonnes: Decimal = Field(..., gt=0, description="Mass handled")
    dwell_time_days: Decimal = Field(default=Decimal("0"), ge=0, description="Storage duration")

    # Temperature control
    temperature_control: TemperatureControl = Field(default=TemperatureControl.AMBIENT)

    # Energy
    energy_kwh: Optional[Decimal] = Field(None, ge=0, description="Hub energy consumption (if known)")
    floor_area_m2: Optional[Decimal] = Field(None, gt=0, description="Hub floor area")

    # Emission factor override
    custom_ef_per_tonne: Optional[Decimal] = Field(None, ge=0, description="Custom kgCO2e/tonne")
    ef_source: Optional[str] = None

    # Metadata
    operator_name: Optional[str] = None

    class Config:
        frozen = True


class TransportChain(BaseModel):
    """
    Multi-leg transport chain (e.g., truck → warehouse → ship → truck).

    Represents complete journey from supplier to buyer.
    """

    chain_id: str = Field(default_factory=lambda: f"chain_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    legs: List[TransportLeg] = Field(..., min_length=1)
    hubs: List[TransportHub] = Field(default_factory=list)

    # Chain metadata
    origin: str = Field(..., description="Chain origin")
    destination: str = Field(..., description="Final destination")
    incoterm: Optional[Incoterm] = None

    class Config:
        frozen = True

    @model_validator(mode='after')
    def validate_chain_continuity(self) -> 'TransportChain':
        """Ensure legs connect properly (leg[i].destination == leg[i+1].origin)."""
        legs = self.legs
        for i in range(len(legs) - 1):
            if legs[i].destination != legs[i+1].origin:
                raise ValueError(
                    f"Leg continuity broken: leg {i} ends at {legs[i].destination}, "
                    f"but leg {i+1} starts at {legs[i+1].origin}"
                )
        return self


class ShipmentInput(BaseModel):
    """
    Shipment input for distance-based calculation method.

    Primary method for most transport emissions calculations.
    """

    shipment_id: str = Field(..., description="Unique shipment identifier")
    calculation_method: CalculationMethod = Field(default=CalculationMethod.DISTANCE_BASED)

    # Transport chain
    legs: List[TransportLeg] = Field(default_factory=list)
    hubs: List[TransportHub] = Field(default_factory=list)
    transport_chain: Optional[TransportChain] = None  # Alternative to legs+hubs

    # Shipment metadata
    shipment_date: Optional[date] = None
    incoterm: Optional[Incoterm] = None
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None
    buyer_facility_id: Optional[str] = None
    product_category: Optional[str] = None

    # GWP source
    gwp_source: GWPSource = Field(default=GWPSource.AR5)

    # Data quality
    data_source: str = Field(default="erp_system", description="Data source")
    data_quality_tier: Optional[str] = None

    class Config:
        frozen = True

    @model_validator(mode='after')
    def validate_has_transport_data(self) -> 'ShipmentInput':
        """Ensure either legs or transport_chain is provided."""
        if not self.legs and not self.transport_chain:
            raise ValueError("Must provide either 'legs' or 'transport_chain'")
        return self


class FuelConsumptionInput(BaseModel):
    """
    Fuel consumption input for fuel-based calculation method.

    More accurate than distance-based when fuel data available.
    """

    record_id: str = Field(..., description="Unique record identifier")
    calculation_method: CalculationMethod = Field(default=CalculationMethod.FUEL_BASED)

    # Fuel details
    fuel_type: TransportFuelType
    fuel_consumed_litres: Optional[Decimal] = Field(None, gt=0)
    fuel_consumed_kg: Optional[Decimal] = Field(None, gt=0)
    electricity_consumed_kwh: Optional[Decimal] = Field(None, gt=0)

    # Transport context
    mode: TransportMode
    vehicle_type: Optional[str] = None
    route: Optional[str] = Field(None, description="Route description")
    distance_km: Optional[Decimal] = Field(None, gt=0)
    cargo_mass_tonnes: Optional[Decimal] = Field(None, gt=0)

    # Allocation (if shared vehicle)
    allocation_method: AllocationMethod = Field(default=AllocationMethod.MASS)
    allocation_percentage: Decimal = Field(default=Decimal("100"), ge=0, le=100)

    # EF scope
    ef_scope: EFScope = Field(default=EFScope.WTW)
    gwp_source: GWPSource = Field(default=GWPSource.AR5)

    # Metadata
    reporting_period: Optional[str] = Field(None, description="e.g., '2023-Q1'")
    carrier_name: Optional[str] = None
    data_source: str = Field(default="fuel_receipts")

    class Config:
        frozen = True

    @model_validator(mode='after')
    def validate_fuel_quantity(self) -> 'FuelConsumptionInput':
        """Ensure at least one fuel quantity is provided."""
        if not any([self.fuel_consumed_litres, self.fuel_consumed_kg, self.electricity_consumed_kwh]):
            raise ValueError("Must provide at least one fuel/energy quantity")
        return self


class SpendInput(BaseModel):
    """
    Transport spend input for spend-based calculation method.

    Least accurate but useful for screening or when activity data unavailable.
    """

    record_id: str = Field(..., description="Unique record identifier")
    calculation_method: CalculationMethod = Field(default=CalculationMethod.SPEND_BASED)

    # Spend details
    spend_amount: Decimal = Field(..., gt=0, description="Transport spend")
    currency: CurrencyCode = Field(default=CurrencyCode.USD)
    spend_year: int = Field(..., ge=2000, le=2030)

    # Transport type
    transport_type: str = Field(..., description="NAICS code or description (e.g., 'NAICS_484_truck_transport')")
    mode: Optional[TransportMode] = None

    # EEIO factor
    eeio_factor_kg_per_usd: Optional[Decimal] = Field(None, gt=0, description="Custom EEIO factor")
    eeio_source: str = Field(default="USEEIO_v2.0")

    # Context
    supplier_id: Optional[str] = None
    route_description: Optional[str] = None

    # Metadata
    reporting_period: Optional[str] = None
    data_source: str = Field(default="accounts_payable")

    class Config:
        frozen = True


class SupplierEmissionInput(BaseModel):
    """
    Supplier-provided emissions for supplier-specific calculation method.

    Highest accuracy if supplier data is verified/audited.
    """

    record_id: str = Field(..., description="Unique record identifier")
    calculation_method: CalculationMethod = Field(default=CalculationMethod.SUPPLIER_SPECIFIC)

    # Supplier details
    supplier_id: str
    supplier_name: str

    # Emissions
    emissions_kgco2e: Decimal = Field(..., ge=0, description="Supplier-reported kgCO2e")
    emissions_scope: str = Field(..., description="Scope/boundary (e.g., 'WTW', 'TTW')")

    # Shipment context
    shipment_id: Optional[str] = None
    cargo_mass_tonnes: Decimal = Field(..., gt=0)
    distance_km: Optional[Decimal] = Field(None, gt=0)

    # Verification
    verification_status: str = Field(default="unverified", description="verified, audited, self-reported")
    verification_standard: Optional[str] = Field(None, description="e.g., 'ISO 14083', 'GLEC Framework'")
    data_quality_score: Optional[Decimal] = Field(None, ge=1, le=5)

    # Metadata
    reporting_period: Optional[str] = None
    data_source: str = Field(default="supplier_portal")

    class Config:
        frozen = True


class AllocationConfig(BaseModel):
    """Configuration for emission allocation (shared transport)."""

    allocation_method: AllocationMethod

    # Shipment quantities
    shipment_mass_tonnes: Optional[Decimal] = Field(None, gt=0)
    shipment_volume_m3: Optional[Decimal] = Field(None, gt=0)
    shipment_pallet_positions: Optional[int] = Field(None, gt=0)
    shipment_teu: Optional[Decimal] = Field(None, gt=0)
    shipment_revenue: Optional[Decimal] = Field(None, gt=0)
    shipment_chargeable_weight_kg: Optional[Decimal] = Field(None, gt=0)
    shipment_floor_area_m2: Optional[Decimal] = Field(None, gt=0)

    # Total vehicle/vessel quantities
    total_capacity_tonnes: Optional[Decimal] = Field(None, gt=0)
    total_capacity_m3: Optional[Decimal] = Field(None, gt=0)
    total_pallet_positions: Optional[int] = Field(None, gt=0)
    total_teu: Optional[int] = Field(None, gt=0)
    total_revenue: Optional[Decimal] = Field(None, gt=0)
    total_floor_area_m2: Optional[Decimal] = Field(None, gt=0)

    class Config:
        frozen = True


class ReeferConfig(BaseModel):
    """Configuration for refrigerated/temperature-controlled transport."""

    temperature_control: TemperatureControl
    target_temperature_celsius: Optional[Decimal] = None
    reefer_unit_type: Optional[str] = Field(None, description="e.g., 'diesel_genset', 'electric'")
    reefer_fuel_litres: Optional[Decimal] = Field(None, ge=0)
    reefer_electricity_kwh: Optional[Decimal] = Field(None, ge=0)

    # HFC leakage (refrigerant leakage)
    refrigerant_type: Optional[str] = Field(None, description="e.g., 'HFC-134a', 'R-404A'")
    refrigerant_charge_kg: Optional[Decimal] = Field(None, ge=0)
    annual_leakage_rate: Decimal = Field(default=Decimal("0.15"), ge=0, le=1, description="15% default")

    class Config:
        frozen = True


class WarehouseConfig(BaseModel):
    """Configuration for warehouse/hub emissions."""

    hub_type: HubType
    floor_area_m2: Decimal = Field(..., gt=0)
    temperature_control: TemperatureControl = Field(default=TemperatureControl.AMBIENT)

    # Energy
    annual_energy_kwh: Optional[Decimal] = Field(None, ge=0)
    energy_intensity_kwh_per_m2: Optional[Decimal] = Field(None, ge=0)

    # Grid emissions
    grid_emission_factor_kg_per_kwh: Decimal = Field(
        default=Decimal("0.475"),  # Global average
        ge=0
    )

    class Config:
        frozen = True


class LegResult(BaseModel):
    """Result for single transport leg calculation."""

    leg_id: str
    mode: TransportMode
    vehicle_type: Optional[str]

    # Activity data
    distance_km: Decimal
    cargo_mass_tonnes: Decimal
    tonne_km: Decimal = Field(..., description="distance × mass")

    # Emission factor
    ef_per_tkm: Decimal = Field(..., description="kgCO2e per tonne-km")
    ef_source: str
    ef_scope: EFScope

    # Adjustments
    laden_adjustment: Decimal = Field(default=Decimal("1.0"))
    reefer_uplift: Decimal = Field(default=Decimal("1.0"))
    allocation_percentage: Decimal = Field(default=Decimal("100"))

    # Emissions
    emissions_kgco2e: Decimal = Field(..., ge=0)
    emissions_co2_kg: Decimal = Field(default=Decimal("0"), ge=0)
    emissions_ch4_kg: Decimal = Field(default=Decimal("0"), ge=0)
    emissions_n2o_kg: Decimal = Field(default=Decimal("0"), ge=0)

    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class HubResult(BaseModel):
    """Result for hub/warehouse emissions calculation."""

    hub_id: str
    hub_type: HubType
    location: str

    # Activity data
    cargo_mass_tonnes: Decimal
    dwell_time_days: Decimal

    # Emission factor
    ef_per_tonne: Decimal = Field(..., description="kgCO2e per tonne handled")
    ef_source: str

    # Emissions
    emissions_kgco2e: Decimal = Field(..., ge=0)

    # Energy (if applicable)
    energy_kwh: Optional[Decimal] = None

    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class TransportChainResult(BaseModel):
    """Result for complete transport chain."""

    chain_id: str
    origin: str
    destination: str

    # Leg results
    leg_results: List[LegResult]
    hub_results: List[HubResult]

    # Totals
    total_distance_km: Decimal
    total_tonne_km: Decimal
    total_emissions_kgco2e: Decimal = Field(..., ge=0)
    total_emissions_tco2e: Decimal = Field(..., ge=0, description="Metric tonnes CO2e")

    # Breakdown by mode
    emissions_by_mode: Dict[str, Decimal] = Field(default_factory=dict)

    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class CalculationRequest(BaseModel):
    """Request for emissions calculation."""

    request_id: str = Field(default_factory=lambda: f"req_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")
    calculation_method: CalculationMethod

    # Input data (one of these required)
    shipment_input: Optional[ShipmentInput] = None
    fuel_input: Optional[FuelConsumptionInput] = None
    spend_input: Optional[SpendInput] = None
    supplier_input: Optional[SupplierEmissionInput] = None

    # Configuration
    gwp_source: GWPSource = Field(default=GWPSource.AR5)
    ef_scope: EFScope = Field(default=EFScope.WTW)
    include_uncertainty: bool = Field(default=True)
    include_dqi: bool = Field(default=True)

    # Compliance
    compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list)

    class Config:
        frozen = True

    @model_validator(mode='after')
    def validate_input_provided(self) -> 'CalculationRequest':
        """Ensure at least one input is provided."""
        inputs = [self.shipment_input, self.fuel_input, self.spend_input, self.supplier_input]
        if not any(inputs):
            raise ValueError("Must provide at least one input type")
        return self


class CalculationResult(BaseModel):
    """Result from emissions calculation."""

    request_id: str
    calculation_method: CalculationMethod

    # Emissions
    total_emissions_kgco2e: Decimal = Field(..., ge=0)
    total_emissions_tco2e: Decimal = Field(..., ge=0)

    # Breakdown
    emissions_co2_kg: Decimal = Field(default=Decimal("0"), ge=0)
    emissions_ch4_kg: Decimal = Field(default=Decimal("0"), ge=0)
    emissions_n2o_kg: Decimal = Field(default=Decimal("0"), ge=0)
    emissions_hfc_kgco2e: Decimal = Field(default=Decimal("0"), ge=0)

    # Activity data
    total_distance_km: Optional[Decimal] = None
    total_tonne_km: Optional[Decimal] = None
    total_fuel_litres: Optional[Decimal] = None
    total_spend: Optional[Decimal] = None

    # Detailed results (for distance-based)
    transport_chain_result: Optional[TransportChainResult] = None

    # Uncertainty
    uncertainty_range_percent: Optional[Decimal] = None
    lower_bound_kgco2e: Optional[Decimal] = None
    upper_bound_kgco2e: Optional[Decimal] = None

    # Data Quality Indicator
    dqi_reliability: Optional[DQIScore] = None
    dqi_completeness: Optional[DQIScore] = None
    dqi_temporal: Optional[DQIScore] = None
    dqi_geographical: Optional[DQIScore] = None
    dqi_technological: Optional[DQIScore] = None
    dqi_composite_score: Optional[Decimal] = None
    dqi_quality_tier: Optional[str] = None

    # Compliance
    compliance_results: List['ComplianceCheckResult'] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default=VERSION)

    class Config:
        frozen = True


class BatchCalculationRequest(BaseModel):
    """Request for batch emissions calculation."""

    batch_id: str = Field(default_factory=lambda: f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    requests: List[CalculationRequest] = Field(..., min_length=1, max_length=10000)

    # Batch configuration
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=16)
    fail_on_error: bool = Field(default=False, description="Fail entire batch if one record fails")

    class Config:
        frozen = True


class BatchCalculationResult(BaseModel):
    """Result from batch emissions calculation."""

    batch_id: str
    total_records: int
    successful_records: int
    failed_records: int

    # Results
    results: List[CalculationResult]
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # Aggregates
    total_emissions_kgco2e: Decimal = Field(..., ge=0)
    total_emissions_tco2e: Decimal = Field(..., ge=0)

    # Status
    batch_status: BatchStatus

    # Provenance
    processing_time_seconds: Decimal
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class ComplianceRequirement(BaseModel):
    """Compliance requirement definition."""

    framework: ComplianceFramework
    requirement_id: str
    requirement_description: str
    required_fields: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class ComplianceCheckResult(BaseModel):
    """Result from compliance check."""

    framework: ComplianceFramework
    status: ComplianceStatus

    # Checks
    checks_performed: int
    checks_passed: int
    checks_failed: int

    # Issues
    missing_fields: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    class Config:
        frozen = True


class AggregationResult(BaseModel):
    """
    Aggregated emissions result.

    Used for reporting (e.g., total Category 4 emissions for fiscal year).
    """

    aggregation_id: str = Field(default_factory=lambda: f"agg_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    aggregation_level: str = Field(..., description="e.g., 'supplier', 'mode', 'product_category', 'fiscal_year'")
    aggregation_key: str = Field(..., description="e.g., 'SUPPLIER_12345', 'ROAD', '2023'")

    # Aggregated values
    total_shipments: int = Field(..., ge=0)
    total_distance_km: Decimal = Field(default=Decimal("0"), ge=0)
    total_tonne_km: Decimal = Field(default=Decimal("0"), ge=0)
    total_emissions_kgco2e: Decimal = Field(..., ge=0)
    total_emissions_tco2e: Decimal = Field(..., ge=0)

    # Breakdown by mode
    emissions_by_mode: Dict[str, Decimal] = Field(default_factory=dict)

    # Breakdown by method
    emissions_by_method: Dict[str, Decimal] = Field(default_factory=dict)

    # Period
    reporting_period_start: Optional[date] = None
    reporting_period_end: Optional[date] = None

    # Data quality
    average_dqi_score: Optional[Decimal] = None

    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True


class ExportRequest(BaseModel):
    """Request for exporting results."""

    export_id: str = Field(default_factory=lambda: f"export_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    format: ExportFormat

    # Data to export
    calculation_results: Optional[List[CalculationResult]] = None
    aggregation_results: Optional[List[AggregationResult]] = None

    # Export configuration
    include_metadata: bool = Field(default=True)
    include_dqi: bool = Field(default=True)
    include_uncertainty: bool = Field(default=True)
    include_compliance: bool = Field(default=True)

    # Framework-specific
    framework: Optional[ComplianceFramework] = None

    class Config:
        frozen = True


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def calculate_provenance_hash(data: Any) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Args:
        data: Data to hash (will be converted to JSON string)

    Returns:
        SHA-256 hash as hex string
    """
    if isinstance(data, BaseModel):
        data_str = data.model_dump_json(indent=None)
    elif isinstance(data, dict):
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def get_gwp(gas: EmissionGas, source: GWPSource = GWPSource.AR5) -> Decimal:
    """
    Get Global Warming Potential for a gas.

    Args:
        gas: Greenhouse gas
        source: IPCC Assessment Report

    Returns:
        GWP value (unitless)
    """
    return GWP_VALUES.get(source, {}).get(gas, Decimal("0"))


def calculate_co2e(
    co2_kg: Decimal,
    ch4_kg: Decimal,
    n2o_kg: Decimal,
    gwp_source: GWPSource = GWPSource.AR5
) -> Decimal:
    """
    Calculate total CO2e from individual gases.

    Args:
        co2_kg: CO2 mass in kg
        ch4_kg: CH4 mass in kg
        n2o_kg: N2O mass in kg
        gwp_source: IPCC AR source

    Returns:
        Total kgCO2e
    """
    co2e = co2_kg * get_gwp(EmissionGas.CO2, gwp_source)
    co2e += ch4_kg * get_gwp(EmissionGas.CH4, gwp_source)
    co2e += n2o_kg * get_gwp(EmissionGas.N2O, gwp_source)
    return co2e


def get_dqi_composite_score(
    reliability: DQIScore,
    completeness: DQIScore,
    temporal: DQIScore,
    geographical: DQIScore,
    technological: DQIScore
) -> Decimal:
    """
    Calculate composite DQI score (average of five dimensions).

    Per ISO 14083, DQI is 1 (best) to 5 (worst).

    Args:
        reliability: Reliability score
        completeness: Completeness score
        temporal: Temporal correlation score
        geographical: Geographical correlation score
        technological: Technological correlation score

    Returns:
        Composite DQI score (1-5)
    """
    scores = [
        DQI_SCORE_VALUES[reliability],
        DQI_SCORE_VALUES[completeness],
        DQI_SCORE_VALUES[temporal],
        DQI_SCORE_VALUES[geographical],
        DQI_SCORE_VALUES[technological],
    ]
    return sum(scores) / Decimal("5")


def get_dqi_quality_tier(composite_score: Decimal) -> str:
    """
    Map composite DQI score to quality tier.

    Args:
        composite_score: Composite DQI score (1-5)

    Returns:
        Quality tier string
    """
    for tier, (min_score, max_score) in DQI_QUALITY_TIERS.items():
        if min_score <= composite_score < max_score:
            return tier
    return "tier_5_very_poor"  # Default to worst if out of range


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Agent metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enums
    "CalculationMethod",
    "TransportMode",
    "RoadVehicleType",
    "RailType",
    "MaritimeVesselType",
    "AircraftType",
    "PipelineType",
    "TransportFuelType",
    "LadenState",
    "AllocationMethod",
    "Incoterm",
    "HubType",
    "TemperatureControl",
    "EFScope",
    "DistanceMethod",
    "CurrencyCode",
    "DQIDimension",
    "DQIScore",
    "UncertaintyMethod",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ExportFormat",
    "BatchStatus",
    "GWPSource",
    "EmissionGas",

    # Constants
    "GWP_VALUES",
    "ROAD_EMISSION_FACTORS",
    "RAIL_EMISSION_FACTORS",
    "MARITIME_EMISSION_FACTORS",
    "AIR_EMISSION_FACTORS",
    "PIPELINE_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "EEIO_TRANSPORT_FACTORS",
    "HUB_EMISSION_FACTORS",
    "REEFER_UPLIFT_FACTORS",
    "LOAD_FACTOR_DEFAULTS",
    "EMPTY_RUNNING_RATES",
    "WAREHOUSE_ENERGY_INTENSITIES",
    "INCOTERM_CATEGORY_MAP",
    "DQI_SCORE_VALUES",
    "DQI_QUALITY_TIERS",
    "UNCERTAINTY_RANGES",
    "FRAMEWORK_REQUIRED_DISCLOSURES",

    # Models
    "TransportEmissionFactor",
    "RoadVehicleProfile",
    "MaritimeVesselProfile",
    "AircraftProfile",
    "TransportLeg",
    "TransportHub",
    "TransportChain",
    "ShipmentInput",
    "FuelConsumptionInput",
    "SpendInput",
    "SupplierEmissionInput",
    "AllocationConfig",
    "ReeferConfig",
    "WarehouseConfig",
    "LegResult",
    "HubResult",
    "TransportChainResult",
    "CalculationRequest",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "ComplianceRequirement",
    "ComplianceCheckResult",
    "AggregationResult",
    "ExportRequest",

    # Helper functions
    "calculate_provenance_hash",
    "get_gwp",
    "calculate_co2e",
    "get_dqi_composite_score",
    "get_dqi_quality_tier",
]
