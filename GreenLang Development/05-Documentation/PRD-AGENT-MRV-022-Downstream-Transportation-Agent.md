# PRD: AGENT-MRV-022 -- Scope 3 Category 9 Downstream Transportation & Distribution Agent

---

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-009 |
| **Internal Label** | AGENT-MRV-022 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/downstream_transportation/` |
| **DB Migration** | V073 |
| **Metrics Prefix** | `gl_dto_` |
| **Table Prefix** | `gl_dto_` |
| **API** | `/api/v1/downstream-transportation` |
| **Env Prefix** | `GL_DTO_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The **GL-MRV-S3-009 Downstream Transportation & Distribution Agent** implements GHG Protocol Scope 3 Category 9 emissions accounting for the transportation and distribution of products sold by the reporting company in the reporting year between the reporting company's operations and the end consumer (not paid for by the reporting company), including retail and storage.

Category 9 is the outbound logistics counterpart to Category 4 (Upstream Transportation). While Category 4 covers inbound logistics paid for by the reporting company, Category 9 covers outbound logistics of sold products that occur in vehicles and facilities **not owned or controlled** by the reporting company and **not paid for** by the reporting company.

Category 9 encompasses the following sub-activities as defined in the GHG Protocol Scope 3 Standard (Chapter 9):

- **Outbound transportation (9a)** -- Transportation of sold products from the reporting company's gate to the end customer or next point in the value chain, where transportation is paid for by the customer or a third party (determined by Incoterms: EXW, FCA, FOB, FAS, CFR).
- **Outbound distribution (9b)** -- Distribution center and warehouse operations between the reporting company and the end consumer, including cross-docking, sorting, packaging, and inventory holding.
- **Retail storage (9c)** -- Energy consumption at retail outlets, shops, and showrooms where the reporting company's products are stored, displayed, and sold by third-party retailers.
- **Last-mile delivery (9d)** -- Final delivery from distribution hub or retail outlet to the end consumer, including parcel delivery, courier services, and e-commerce fulfillment, which has grown significantly with online shopping.

Category 9 typically represents **2-12% of total Scope 3 emissions** for manufacturing, FMCG, and consumer goods companies. For companies selling physical products through complex distribution networks, this can be highly material. The challenge is that the reporting company often has limited visibility into downstream logistics because the customer or intermediary controls transportation arrangements.

### Justification for Dedicated Agent

1. **Outbound boundary complexity** -- Incoterms (EXW, FCA, FOB, FAS, CFR, CIF, CIP, DAP, DPU, DDP) precisely define the Cat 4 vs Cat 9 boundary; the agent must classify each shipment based on the applicable Incoterm
2. **Distribution channel diversity** -- Products may flow through direct-to-consumer, wholesale, retail, e-commerce, and hybrid channels, each with different transport patterns and emission profiles
3. **Last-mile delivery growth** -- E-commerce has made last-mile delivery (parcels, couriers, urban delivery) a major and growing emission source with unique EF structures (per-parcel vs per-tonne-km)
4. **Warehouse and storage emissions** -- Distribution centers, cold storage, and retail outlets contribute significant emissions from HVAC, lighting, refrigeration, and material handling
5. **Limited data availability** -- Companies often lack direct data on downstream logistics; the agent must support screening methods (average-data, spend-based) alongside detailed methods
6. **Product-level allocation** -- Emissions must be allocated to individual products or product lines for product carbon footprints and eco-design compliance
7. **Cold chain tracking** -- Temperature-controlled distribution (refrigerated, frozen, ambient) adds 15-40% energy uplift depending on mode and temperature regime
8. **Double-counting prevention** -- Complex boundaries with Cat 4 (company-paid outbound), Cat 11 (use of sold products), Cat 12 (end-of-life), and Scope 1/2

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 9
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 9: Category 9
- GHG Protocol Scope 3 Calculation Guidance (online)
- ISO 14083:2023 -- Quantification and reporting of GHG emissions from transport operations
- GLEC Framework v3.0 (2023) -- Global Logistics Emissions Council methodology
- Smart Freight Centre -- Clean Cargo Working Group methodology
- DEFRA/DESNZ Greenhouse Gas Reporting Conversion Factors 2024 -- Freight transport tables
- EPA SmartWay -- Carrier performance data and benchmarks
- IMO Fourth GHG Study 2020 -- Maritime emission factors
- ICAO Carbon Emissions Calculator -- Air freight methodology
- ICC Incoterms 2020 -- International Commercial Terms defining transport responsibility

---

## 2. Architecture

### 7-Engine Design

| Engine # | Class Name | File | Purpose |
|----------|-----------|------|---------|
| 1 | `DownstreamTransportDatabaseEngine` | `downstream_transport_database.py` | Emission factor lookup, vehicle/vessel types, route data, EEIO factors |
| 2 | `DistanceBasedCalculatorEngine` | `distance_based_calculator.py` | Tonne-km method for all transport modes (road, rail, maritime, air, intermodal) |
| 3 | `SpendBasedCalculatorEngine` | `spend_based_calculator.py` | EEIO spend-based method with CPI deflation and currency conversion |
| 4 | `AverageDataCalculatorEngine` | `average_data_calculator.py` | Industry average method using product category and distribution channel defaults |
| 5 | `WarehouseDistributionEngine` | `warehouse_distribution.py` | Distribution center, cold storage, and retail storage emissions |
| 6 | `ComplianceCheckerEngine` | `compliance_checker.py` | 7-framework regulatory compliance validation |
| 7 | `DownstreamTransportPipelineEngine` | `downstream_transport_pipeline.py` | 10-stage orchestration pipeline |

### 10-Stage Pipeline

| Stage | Name | Description |
|-------|------|-------------|
| 1 | VALIDATE | Input validation, schema checks, required field verification |
| 2 | CLASSIFY | Incoterm classification (Cat 4 vs Cat 9), transport mode identification |
| 3 | NORMALIZE | Unit normalization (distances, weights, currencies, energy) |
| 4 | RESOLVE_EFS | Emission factor resolution from DEFRA/EPA/GLEC/IMO/ICAO databases |
| 5 | CALCULATE | Core emissions calculation by method and transport mode |
| 6 | ALLOCATE | Product-level allocation (mass, volume, revenue, units sold) |
| 7 | AGGREGATE | Aggregation by mode, channel, product, destination region |
| 8 | COMPLIANCE | Regulatory compliance checking across 7 frameworks |
| 9 | PROVENANCE | DQI scoring, metadata assembly, provenance hash chain |
| 10 | SEAL | Final SHA-256 sealing and Merkle root computation |

### Core Infrastructure Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package init with AGENT_ID, VERSION, graceful imports |
| `models.py` | Enums, constant tables, Pydantic models |
| `config.py` | GL_DTO_ environment configuration, thread-safe singleton |
| `metrics.py` | Prometheus metrics with gl_dto_ prefix |
| `provenance.py` | SHA-256 chain hashing, Merkle trees, 10-stage provenance tracking |
| `setup.py` | DownstreamTransportService facade wiring 7 engines |
| `api/__init__.py` | API package marker |
| `api/router.py` | 22 REST endpoints at /api/v1/downstream-transportation |

---

## 3. Enumerations (22 Enums)

```python
class TransportMode(str, Enum):
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
    # Road
    LGV_PETROL = "lgv_petrol"
    LGV_DIESEL = "lgv_diesel"
    LGV_ELECTRIC = "lgv_electric"
    RIGID_TRUCK_SMALL = "rigid_truck_small"        # <7.5t
    RIGID_TRUCK_MEDIUM = "rigid_truck_medium"      # 7.5-17t
    RIGID_TRUCK_LARGE = "rigid_truck_large"        # >17t
    ARTICULATED_TRUCK = "articulated_truck"        # >33t
    DELIVERY_VAN = "delivery_van"
    CARGO_BIKE = "cargo_bike"
    # Rail
    FREIGHT_TRAIN = "freight_train"
    INTERMODAL_RAIL = "intermodal_rail"
    # Maritime
    CONTAINER_SHIP_SMALL = "container_ship_small"  # <1000 TEU
    CONTAINER_SHIP_MEDIUM = "container_ship_medium" # 1000-5000 TEU
    CONTAINER_SHIP_LARGE = "container_ship_large"  # >5000 TEU
    BULK_CARRIER = "bulk_carrier"
    RO_RO_FERRY = "ro_ro_ferry"
    # Air
    FREIGHTER_NARROW = "freighter_narrow"
    FREIGHTER_WIDE = "freighter_wide"
    BELLY_FREIGHT = "belly_freight"

class Incoterm(str, Enum):
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
    DIRECT_TO_CONSUMER = "direct_to_consumer"
    WHOLESALE = "wholesale"
    RETAIL = "retail"
    E_COMMERCE = "e_commerce"
    DISTRIBUTOR = "distributor"
    FRANCHISE = "franchise"

class CalculationMethod(str, Enum):
    DISTANCE_BASED = "distance_based"
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"

class TemperatureRegime(str, Enum):
    AMBIENT = "ambient"
    CHILLED = "chilled"           # 2-8°C
    FROZEN = "frozen"             # -18 to -25°C
    DEEP_FROZEN = "deep_frozen"   # <-25°C
    CONTROLLED = "controlled"     # 15-25°C

class AllocationMethod(str, Enum):
    MASS = "mass"
    VOLUME = "volume"
    REVENUE = "revenue"
    UNITS_SOLD = "units_sold"
    TEU = "teu"
    PALLET_POSITIONS = "pallet_positions"
    FLOOR_AREA = "floor_area"

class WarehouseType(str, Enum):
    DISTRIBUTION_CENTER = "distribution_center"
    CROSS_DOCK = "cross_dock"
    COLD_STORAGE = "cold_storage"
    RETAIL_STORE = "retail_store"
    FULFILLMENT_CENTER = "fulfillment_center"
    DARK_STORE = "dark_store"

class EnergySource(str, Enum):
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    LPG = "lpg"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"

class EmissionFactorSource(str, Enum):
    DEFRA = "defra"
    EPA_SMARTWAY = "epa_smartway"
    GLEC = "glec"
    ICAO = "icao"
    IMO = "imo"
    IEA = "iea"
    ECOINVENT = "ecoinvent"
    CUSTOM = "custom"

class ComplianceFramework(str, Enum):
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ISO_14083 = "iso_14083"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"

class ComplianceSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

class ComplianceStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

class DataQualityTier(str, Enum):
    TIER_1 = "tier_1"   # Supplier-specific
    TIER_2 = "tier_2"   # Industry average
    TIER_3 = "tier_3"   # Spend-based EEIO
    TIER_4 = "tier_4"   # Proxy/estimated

class ReturnType(str, Enum):
    NO_RETURN = "no_return"
    CUSTOMER_RETURN = "customer_return"
    PRODUCT_RECALL = "product_recall"
    REUSABLE_PACKAGING = "reusable_packaging"

class LastMileType(str, Enum):
    PARCEL_STANDARD = "parcel_standard"
    PARCEL_EXPRESS = "parcel_express"
    SAME_DAY = "same_day"
    CLICK_AND_COLLECT = "click_and_collect"
    LOCKER = "locker"
    CARGO_BIKE = "cargo_bike"

class GWPSource(str, Enum):
    AR4 = "ar4"
    AR5 = "ar5"
    AR6 = "ar6"
    AR6_20YR = "ar6_20yr"

class UncertaintyMethod(str, Enum):
    IPCC_TIER_1 = "ipcc_tier_1"
    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    EXPERT_JUDGMENT = "expert_judgment"

class PipelineStage(str, Enum):
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
    EMPTY = "empty"           # 0% utilization
    PARTIAL = "partial"       # 25-50%
    HALF = "half"             # 50%
    TYPICAL = "typical"       # 60-75%
    FULL = "full"             # 85-100%
```

---

## 4. Constant Tables (14 Tables)

### 4.1 TRANSPORT_EMISSION_FACTORS (per tonne-km, kgCO2e)

| Mode | Vehicle Type | EF (kgCO2e/tkm) | WTT (kgCO2e/tkm) | Source |
|------|-------------|-----------------|-------------------|--------|
| Road | LGV Petrol | 0.584 | 0.139 | DEFRA 2024 |
| Road | LGV Diesel | 0.480 | 0.112 | DEFRA 2024 |
| Road | LGV Electric | 0.118 | 0.024 | DEFRA 2024 |
| Road | Rigid <7.5t | 0.441 | 0.103 | DEFRA 2024 |
| Road | Rigid 7.5-17t | 0.213 | 0.050 | DEFRA 2024 |
| Road | Rigid >17t | 0.150 | 0.035 | DEFRA 2024 |
| Road | Articulated >33t | 0.107 | 0.025 | DEFRA 2024 |
| Road | Delivery Van | 0.580 | 0.135 | DEFRA 2024 |
| Road | Cargo Bike | 0.000 | 0.000 | Zero emission |
| Rail | Freight Train | 0.028 | 0.006 | DEFRA 2024 |
| Rail | Intermodal Rail | 0.025 | 0.005 | DEFRA 2024 |
| Maritime | Container <1000 TEU | 0.022 | 0.005 | IMO 2020 |
| Maritime | Container 1000-5000 TEU | 0.016 | 0.004 | IMO 2020 |
| Maritime | Container >5000 TEU | 0.008 | 0.002 | IMO 2020 |
| Maritime | Bulk Carrier | 0.005 | 0.001 | IMO 2020 |
| Maritime | Ro-Ro Ferry | 0.060 | 0.014 | DEFRA 2024 |
| Air | Freighter Narrow | 0.602 | 0.143 | DEFRA 2024 |
| Air | Freighter Wide | 0.495 | 0.118 | DEFRA 2024 |
| Air | Belly Freight | 0.440 | 0.105 | ICAO 2024 |
| Inland Waterway | Barge | 0.032 | 0.007 | GLEC v3 |
| Courier | Parcel Standard | 0.420 | 0.098 | DEFRA 2024 |
| Courier | Parcel Express | 0.520 | 0.121 | DEFRA 2024 |
| Last Mile | Same Day | 0.680 | 0.159 | Industry avg |
| Last Mile | Click & Collect | 0.050 | 0.012 | Industry avg |
| Last Mile | Locker | 0.040 | 0.009 | Industry avg |
| Last Mile | Cargo Bike | 0.005 | 0.001 | Industry avg |

### 4.2 COLD_CHAIN_UPLIFT_FACTORS

| Temperature Regime | Road Uplift | Rail Uplift | Maritime Uplift | Air Uplift |
|-------------------|-------------|-------------|-----------------|------------|
| Ambient | 1.00 | 1.00 | 1.00 | 1.00 |
| Chilled (2-8°C) | 1.20 | 1.15 | 1.18 | 1.10 |
| Frozen (-18 to -25°C) | 1.35 | 1.25 | 1.30 | 1.15 |
| Deep Frozen (<-25°C) | 1.50 | 1.35 | 1.40 | 1.20 |
| Controlled (15-25°C) | 1.05 | 1.03 | 1.04 | 1.02 |

### 4.3 WAREHOUSE_EMISSION_FACTORS (kgCO2e per m² per year)

| Warehouse Type | Electricity | Gas/Heating | Total | Source |
|---------------|------------|-------------|-------|--------|
| Distribution Center | 45.0 | 12.0 | 57.0 | CIBSE TM46 |
| Cross-Dock | 30.0 | 8.0 | 38.0 | CIBSE TM46 |
| Cold Storage (Chilled) | 120.0 | 5.0 | 125.0 | Industry avg |
| Cold Storage (Frozen) | 180.0 | 3.0 | 183.0 | Industry avg |
| Retail Store | 85.0 | 25.0 | 110.0 | CIBSE TM46 |
| Fulfillment Center | 55.0 | 10.0 | 65.0 | Industry avg |
| Dark Store | 95.0 | 15.0 | 110.0 | Industry avg |

### 4.4 LAST_MILE_EMISSION_FACTORS (kgCO2e per delivery)

| Last Mile Type | Urban (kgCO2e) | Suburban | Rural | Source |
|---------------|----------------|----------|-------|--------|
| Parcel Standard | 0.520 | 0.780 | 1.200 | DEFRA 2024 |
| Parcel Express | 0.680 | 0.950 | 1.500 | DEFRA 2024 |
| Same Day | 0.850 | 1.200 | 1.800 | Industry avg |
| Click & Collect | 0.050 | 0.050 | 0.050 | Store energy |
| Locker | 0.040 | 0.040 | 0.040 | Locker energy |
| Cargo Bike | 0.010 | 0.020 | N/A | Industry avg |

### 4.5 EEIO_FACTORS (kgCO2e per USD spent)

| NAICS Code | Sector | EF (kgCO2e/$) | Source |
|-----------|--------|--------------|--------|
| 484110 | General Freight Trucking, Local | 0.470 | EPA USEEIO v2.0 |
| 484121 | General Freight Trucking, Long-Distance | 0.380 | EPA USEEIO v2.0 |
| 482110 | Rail Transportation | 0.280 | EPA USEEIO v2.0 |
| 483111 | Deep Sea Freight | 0.210 | EPA USEEIO v2.0 |
| 481112 | Air Freight | 1.250 | EPA USEEIO v2.0 |
| 492110 | Couriers and Express Delivery | 0.520 | EPA USEEIO v2.0 |
| 493110 | General Warehousing and Storage | 0.340 | EPA USEEIO v2.0 |
| 493120 | Refrigerated Warehousing and Storage | 0.580 | EPA USEEIO v2.0 |
| 454110 | Electronic Shopping and Mail-Order | 0.420 | EPA USEEIO v2.0 |
| 493130 | Farm Product Warehousing | 0.310 | EPA USEEIO v2.0 |

### 4.6 CURRENCY_CONVERSION_RATES (to USD)

| Currency | Rate | Year |
|----------|------|------|
| USD | 1.0000 | 2024 |
| EUR | 1.0850 | 2024 |
| GBP | 1.2650 | 2024 |
| JPY | 0.0067 | 2024 |
| CAD | 0.7400 | 2024 |
| AUD | 0.6550 | 2024 |
| CHF | 1.1300 | 2024 |
| CNY | 0.1400 | 2024 |
| INR | 0.0120 | 2024 |
| BRL | 0.2000 | 2024 |
| KRW | 0.0008 | 2024 |
| SEK | 0.0960 | 2024 |

### 4.7 CPI_DEFLATORS (base year = 2024)

| Year | US CPI Index | Deflator |
|------|-------------|----------|
| 2015 | 237.0 | 0.7983 |
| 2016 | 240.0 | 0.8084 |
| 2017 | 245.1 | 0.8256 |
| 2018 | 251.1 | 0.8458 |
| 2019 | 255.7 | 0.8613 |
| 2020 | 258.8 | 0.8717 |
| 2021 | 270.9 | 0.9125 |
| 2022 | 292.7 | 0.9859 |
| 2023 | 304.7 | 1.0264 |
| 2024 | 296.9 | 1.0000 |
| 2025 | 303.5 | 1.0222 |

### 4.8 RETURN_LOGISTICS_FACTORS

| Return Type | Multiplier | Description |
|------------|-----------|-------------|
| No Return | 0.00 | One-way only |
| Customer Return | 0.85 | 85% of outbound (partially consolidated) |
| Product Recall | 1.00 | Equal to outbound |
| Reusable Packaging | 0.50 | 50% of outbound (backhaul) |

### 4.9 LOAD_FACTOR_ADJUSTMENTS

| Load Factor | Utilization % | Adjustment |
|------------|--------------|------------|
| Empty | 0% | 0.40 (deadhead) |
| Partial | 37% | 0.65 |
| Half | 50% | 0.80 |
| Typical | 67% | 1.00 |
| Full | 92% | 1.15 |

### 4.10 DISTRIBUTION_CHANNEL_DEFAULTS

| Channel | Avg Distance (km) | Avg Mode | Avg Legs | Storage Days |
|---------|-------------------|----------|----------|-------------|
| Direct to Consumer | 500 | road | 2 | 0 |
| Wholesale | 800 | road | 1 | 14 |
| Retail | 600 | road | 2 | 30 |
| E-Commerce | 350 | courier | 3 | 7 |
| Distributor | 1200 | intermodal | 2 | 21 |
| Franchise | 400 | road | 1 | 7 |

### 4.11 GRID_EMISSION_FACTORS (for warehouse electricity)

| Country | EF (kgCO2e/kWh) | Source |
|---------|-----------------|--------|
| US | 0.3937 | EPA eGRID 2024 |
| GB | 0.2121 | DEFRA 2024 |
| DE | 0.3640 | IEA 2024 |
| FR | 0.0569 | IEA 2024 |
| JP | 0.4570 | IEA 2024 |
| CA | 0.1200 | IEA 2024 |
| AU | 0.6100 | IEA 2024 |
| IN | 0.7080 | IEA 2024 |
| CN | 0.5570 | IEA 2024 |
| BR | 0.0740 | IEA 2024 |
| GLOBAL | 0.4360 | IEA 2024 |

### 4.12 INCOTERM_CLASSIFICATION

| Incoterm | Cat 4 (Seller) | Cat 9 (Buyer) | Transfer Point |
|----------|---------------|---------------|----------------|
| EXW | No | All transport | Seller's premises |
| FCA | To carrier | After carrier | Named place |
| FAS | To port | After port | Ship's side |
| FOB | To on board | After on board | Ship's rail |
| CFR | Main carriage | After discharge | Destination port |
| CIF | Main + insurance | After discharge | Destination port |
| CPT | To destination | After delivery | Named place |
| CIP | To dest + ins | After delivery | Named place |
| DAP | To destination | Unloading only | Named place |
| DPU | To unloaded | None | Named place |
| DDP | All transport | None | Named place |

### 4.13 DQI_SCORING (Data Quality Indicators)

| Dimension | Score 1 (Best) | Score 3 (Medium) | Score 5 (Worst) |
|-----------|---------------|------------------|-----------------|
| Technological | Mode-specific EF | Generic mode EF | Economy-wide avg |
| Temporal | Current year | 1-3 years old | >5 years old |
| Geographical | Country-specific | Regional | Global default |
| Completeness | >95% coverage | 50-95% coverage | <50% coverage |
| Reliability | Measured data | Published avg | Expert estimate |

### 4.14 UNCERTAINTY_RANGES

| Method | Low (%) | Central (%) | High (%) |
|--------|---------|-------------|----------|
| Distance-Based | -15 | 0 | +20 |
| Spend-Based | -30 | 0 | +40 |
| Average-Data | -25 | 0 | +35 |
| Supplier-Specific | -10 | 0 | +15 |

---

## 5. Pydantic Models (12 Models)

```python
class ShipmentInput(BaseModel, frozen=True):
    shipment_id: str
    mode: TransportMode
    vehicle_type: Optional[VehicleType]
    origin: str
    destination: str
    distance_km: Decimal
    weight_tonnes: Decimal
    incoterm: Optional[Incoterm] = Incoterm.EXW
    temperature_regime: TemperatureRegime = TemperatureRegime.AMBIENT
    load_factor: LoadFactor = LoadFactor.TYPICAL
    return_type: ReturnType = ReturnType.NO_RETURN

class SpendInput(BaseModel, frozen=True):
    spend_amount: Decimal
    currency: str = "USD"
    spend_year: int = 2024
    naics_code: Optional[str]
    logistics_category: Optional[str]

class WarehouseInput(BaseModel, frozen=True):
    warehouse_type: WarehouseType
    floor_area_m2: Decimal
    storage_days: int
    country: str = "US"
    energy_source: EnergySource = EnergySource.ELECTRICITY
    temperature_regime: TemperatureRegime = TemperatureRegime.AMBIENT
    allocation_share: Decimal = Decimal("1.0")

class LastMileInput(BaseModel, frozen=True):
    delivery_type: LastMileType
    num_deliveries: int
    area_type: str = "urban"  # urban, suburban, rural
    avg_weight_kg: Optional[Decimal]

class AverageDataInput(BaseModel, frozen=True):
    product_category: str
    total_weight_tonnes: Decimal
    distribution_channel: DistributionChannel
    destination_country: str = "US"

class CalculationInput(BaseModel, frozen=True):
    tenant_id: str
    reporting_year: int
    calculation_method: CalculationMethod
    shipments: Optional[List[ShipmentInput]]
    spend_inputs: Optional[List[SpendInput]]
    warehouse_inputs: Optional[List[WarehouseInput]]
    last_mile_inputs: Optional[List[LastMileInput]]
    average_data_inputs: Optional[List[AverageDataInput]]

class CalculationResult(BaseModel):
    calculation_id: str
    tenant_id: str
    reporting_year: int
    method: CalculationMethod
    total_emissions_kg: Decimal
    total_emissions_t: Decimal
    transport_emissions_kg: Decimal
    warehouse_emissions_kg: Decimal
    last_mile_emissions_kg: Decimal
    return_emissions_kg: Decimal
    wtt_emissions_kg: Decimal
    co2_kg: Decimal
    ch4_kg: Decimal
    n2o_kg: Decimal
    dqi_score: Decimal
    uncertainty_low_pct: Decimal
    uncertainty_high_pct: Decimal
    provenance_hash: str

class ShipmentResult(BaseModel):
    shipment_id: str
    mode: TransportMode
    distance_km: Decimal
    weight_tonnes: Decimal
    tonne_km: Decimal
    emissions_kg: Decimal
    wtt_emissions_kg: Decimal
    cold_chain_uplift: Decimal
    return_emissions_kg: Decimal
    ef_used: Decimal
    ef_source: EmissionFactorSource

class WarehouseResult(BaseModel):
    warehouse_type: WarehouseType
    floor_area_m2: Decimal
    storage_days: int
    allocated_emissions_kg: Decimal
    electricity_emissions_kg: Decimal
    heating_emissions_kg: Decimal

class ComplianceResult(BaseModel):
    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    findings: List[dict]
    recommendations: List[str]

class AggregationResult(BaseModel):
    by_mode: Dict[str, Decimal]
    by_channel: Dict[str, Decimal]
    by_destination: Dict[str, Decimal]
    by_method: Dict[str, Decimal]
    transport_pct: Decimal
    warehouse_pct: Decimal
    last_mile_pct: Decimal
```

---

## 6. Calculation Formulas

### 6.1 Distance-Based Method (Primary)

```
Transport Emissions (kgCO2e) = distance_km × weight_tonnes × EF_per_tkm × cold_chain_uplift × load_factor_adj
WTT Emissions (kgCO2e) = distance_km × weight_tonnes × WTT_EF_per_tkm × cold_chain_uplift × load_factor_adj
Return Emissions (kgCO2e) = Transport Emissions × return_multiplier
Total Shipment = Transport + WTT + Return
```

### 6.2 Spend-Based Method (Screening)

```
Deflated Spend (USD) = spend_amount × currency_rate × (cpi_base / cpi_year)
Transport Emissions (kgCO2e) = deflated_spend × EEIO_factor
```

### 6.3 Average-Data Method

```
Avg Distance = channel_default_distance
Avg Mode EF = channel_default_mode_ef
Transport Emissions = total_weight_tonnes × avg_distance × avg_mode_ef
Storage Emissions = storage_days × warehouse_ef_per_day × total_weight_tonnes / capacity
```

### 6.4 Warehouse Emissions

```
Annual Emissions (kgCO2e) = floor_area_m2 × warehouse_ef_per_m2_per_year
Allocated Emissions = Annual Emissions × allocation_share × (storage_days / 365)
```

### 6.5 Last-Mile Delivery

```
Emissions (kgCO2e) = num_deliveries × ef_per_delivery[type][area]
```

### 6.6 Product-Level Allocation

```
Product Emissions = Total Emissions × (product_mass / total_mass)
  or = Total Emissions × (product_revenue / total_revenue)
  or = Total Emissions × (product_units / total_units)
```

---

## 7. Double-Counting Prevention Rules (10 Rules)

| Rule ID | Description | Boundary |
|---------|-------------|----------|
| DC-DTO-001 | Exclude company-paid outbound transport (Cat 4 per Incoterms) | Cat 4 vs Cat 9 |
| DC-DTO-002 | Exclude transport in owned/controlled vehicles (Scope 1) | Scope 1 vs Cat 9 |
| DC-DTO-003 | Exclude electricity for owned warehouses (Scope 2) | Scope 2 vs Cat 9 |
| DC-DTO-004 | Exclude transport included in cradle-to-gate EF (Cat 1) | Cat 1 vs Cat 9 |
| DC-DTO-005 | Exclude fuel WTT already counted in Cat 3 | Cat 3 vs Cat 9 |
| DC-DTO-006 | Exclude distribution of leased assets (Cat 8 or Cat 13) | Cat 8/13 vs Cat 9 |
| DC-DTO-007 | Exclude end-of-life transport (Cat 12) | Cat 12 vs Cat 9 |
| DC-DTO-008 | Exclude customer use-phase transport (Cat 11) | Cat 11 vs Cat 9 |
| DC-DTO-009 | Do not double-count multi-leg segments across methods | Internal dedup |
| DC-DTO-010 | Separate biogenic CO2 from fossil CO2 for biofuel transport | Biogenic accounting |

---

## 8. Compliance Frameworks (7 Frameworks)

| Framework | Checks | Key Requirements |
|-----------|--------|-----------------|
| GHG Protocol | GHG-DTO-001 to GHG-DTO-009 | Completeness, Incoterm boundary, mode disclosure, method hierarchy |
| ISO 14064 | ISO-DTO-001 to ISO-DTO-007 | Uncertainty analysis, boundary completeness, documentation |
| ISO 14083 | ISO83-DTO-001 to ISO83-DTO-006 | WTW mandatory, mode-specific, GLEC alignment |
| CSRD ESRS E1 | CSRD-DTO-001 to CSRD-DTO-008 | ESRS E1-6 Scope 3 GHG, transport mode breakdown, time series |
| CDP | CDP-DTO-001 to CDP-DTO-006 | Module C6.5, method disclosure, relevance assessment |
| SBTi | SBTI-DTO-001 to SBTI-DTO-006 | FLAG/non-FLAG, target boundary, 67% coverage |
| SB 253 | SB253-DTO-001 to SB253-DTO-006 | Category 9 mandatory, third-party assurance, CARB format |

---

## 9. Database Schema (V073)

### Tables (16 operational + 5 reference = 21 total)

| Table | Purpose | Key Columns |
|-------|---------|------------|
| `gl_dto_shipments` | Outbound shipment records | shipment_id, tenant_id, mode, origin, destination, distance_km, weight_tonnes, incoterm |
| `gl_dto_transport_emission_factors` | Transport EFs by mode/vehicle | vehicle_type, ef_per_tkm, wtt_per_tkm, source |
| `gl_dto_cold_chain_factors` | Temperature regime uplifts | temperature_regime, road_uplift, rail_uplift, maritime_uplift, air_uplift |
| `gl_dto_warehouses` | Warehouse/DC profiles | warehouse_id, type, floor_area_m2, country, temperature_regime |
| `gl_dto_warehouse_emission_factors` | Warehouse EFs per m² | warehouse_type, electricity_ef, gas_ef, total_ef |
| `gl_dto_last_mile_factors` | Last-mile delivery EFs | delivery_type, urban_ef, suburban_ef, rural_ef |
| `gl_dto_eeio_factors` | EEIO spend-based factors | naics_code, sector, ef_per_usd |
| `gl_dto_currency_rates` | Currency conversion | currency, rate_to_usd, year |
| `gl_dto_cpi_deflators` | CPI deflation | year, cpi_index, deflator |
| `gl_dto_grid_emission_factors` | Grid EFs for warehouse electricity | country, ef_kwh |
| `gl_dto_distribution_channels` | Channel default params | channel, avg_distance, avg_mode, avg_legs, storage_days |
| `gl_dto_incoterm_classification` | Incoterm Cat 4/9 rules | incoterm, cat4_scope, cat9_scope, transfer_point |
| `gl_dto_calculations` | Calculation results | calculation_id, tenant_id, method, total_emissions_kg |
| `gl_dto_calculation_details` | Per-shipment/warehouse results | detail_id, calculation_id, component_type, emissions_kg |
| `gl_dto_compliance_checks` | Compliance check records | check_id, calculation_id, framework, status, score |
| `gl_dto_aggregations` | Aggregated results | aggregation_id, tenant_id, period, by_mode, by_channel |
| `gl_dto_product_allocations` | Product-level allocations | allocation_id, calculation_id, product_id, allocated_emissions_kg |
| `gl_dto_return_logistics` | Return/reverse logistics | return_id, shipment_id, return_type, return_emissions_kg |
| `gl_dto_provenance` | Provenance chain entries | provenance_id, calculation_id, stage, input_hash, output_hash, chain_hash |
| `gl_dto_audit_entries` | Audit trail | audit_id, tenant_id, action, entity_type, entity_id, timestamp |
| `gl_dto_load_factors` | Load factor reference | load_factor, utilization_pct, adjustment |

### Hypertables (3)

- `gl_dto_calculations` -- Partitioned by `calculated_at`
- `gl_dto_calculation_details` -- Partitioned by `calculated_at`
- `gl_dto_aggregations` -- Partitioned by `period_start`

### Continuous Aggregates (2)

- `gl_dto_hourly_emissions` -- Hourly rollup of calculation results
- `gl_dto_daily_emissions` -- Daily rollup with mode breakdown

### Row-Level Security

All tables enforce tenant isolation via `tenant_id = current_setting('app.tenant_id')` policies.

---

## 10. REST API (22 Endpoints)

Prefix: `/api/v1/downstream-transportation`

| Method | Path | Permission | Description |
|--------|------|-----------|-------------|
| POST | `/calculate` | calculate | Full pipeline calculation |
| POST | `/calculate/distance` | calculate | Distance-based calculation |
| POST | `/calculate/spend` | calculate | Spend-based calculation |
| POST | `/calculate/average-data` | calculate | Average-data calculation |
| POST | `/calculate/warehouse` | calculate | Warehouse storage emissions |
| POST | `/calculate/last-mile` | calculate | Last-mile delivery emissions |
| POST | `/calculate/supplier-specific` | calculate | Carrier-specific data |
| POST | `/calculate/batch` | calculate | Batch multi-method calculation |
| POST | `/calculate/portfolio` | calculate | Full logistics portfolio |
| POST | `/compliance/check` | compliance | Multi-framework compliance check |
| GET | `/calculations/{id}` | read | Get calculation by ID |
| GET | `/calculations` | read | List calculations with filters |
| DELETE | `/calculations/{id}` | delete | Delete calculation |
| GET | `/emission-factors/{mode}` | read | Get EFs by transport mode |
| GET | `/warehouse-benchmarks` | read | Get warehouse emission benchmarks |
| GET | `/last-mile-factors` | read | Get last-mile delivery factors |
| GET | `/incoterm-classification` | read | Get Incoterm boundary rules |
| GET | `/aggregations` | read | Get aggregated results |
| GET | `/provenance/{id}` | read | Get provenance chain |
| GET | `/health` | read | Health check |
| POST | `/uncertainty/analyze` | analyze | Uncertainty analysis |
| POST | `/portfolio/analyze` | analyze | Portfolio optimization analysis |

---

## 11. File Manifest (30 files)

### Source Files (15)

| # | File | Estimated Lines |
|---|------|----------------|
| 1 | `__init__.py` | ~120 |
| 2 | `models.py` | ~2,400 |
| 3 | `config.py` | ~2,500 |
| 4 | `metrics.py` | ~1,700 |
| 5 | `provenance.py` | ~3,000 |
| 6 | `downstream_transport_database.py` | ~2,200 |
| 7 | `distance_based_calculator.py` | ~2,000 |
| 8 | `spend_based_calculator.py` | ~1,600 |
| 9 | `average_data_calculator.py` | ~1,800 |
| 10 | `warehouse_distribution.py` | ~2,000 |
| 11 | `compliance_checker.py` | ~3,200 |
| 12 | `downstream_transport_pipeline.py` | ~1,800 |
| 13 | `setup.py` | ~2,100 |
| 14 | `api/__init__.py` | ~1 |
| 15 | `api/router.py` | ~2,500 |

### Test Files (14)

| # | File | Estimated Tests |
|---|------|----------------|
| 1 | `__init__.py` | 0 |
| 2 | `conftest.py` | 30+ fixtures |
| 3 | `test_models.py` | ~200 |
| 4 | `test_config.py` | ~100 |
| 5 | `test_downstream_transport_database.py` | ~60 |
| 6 | `test_distance_based_calculator.py` | ~60 |
| 7 | `test_spend_based_calculator.py` | ~40 |
| 8 | `test_average_data_calculator.py` | ~40 |
| 9 | `test_warehouse_distribution.py` | ~50 |
| 10 | `test_compliance_checker.py` | ~65 |
| 11 | `test_provenance.py` | ~60 |
| 12 | `test_downstream_transport_pipeline.py` | ~50 |
| 13 | `test_api.py` | ~30 |
| 14 | `test_setup.py` | ~25 |

### Migration (1)

| File | Lines |
|------|-------|
| `V073__downstream_transportation_service.sql` | ~1,200 |

**Total Target: 30 files, ~30K+ lines, 700+ tests**
