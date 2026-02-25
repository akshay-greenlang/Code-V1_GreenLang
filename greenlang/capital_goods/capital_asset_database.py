# -*- coding: utf-8 -*-
"""
CapitalAssetDatabaseEngine - Engine 1: Capital Goods Agent (AGENT-MRV-015)

Reference data and emission factor lookup engine for GHG Protocol Scope 3
Category 2 capital goods emissions.  Provides deterministic, zero-hallucination
lookup of EEIO factors, physical emission factors, supplier-specific factors,
NAICS/ISIC/NACE/UNSPSC cross-mappings, CPI deflators, currency exchange rates,
and sector margin adjustments.

All numeric values use ``Decimal`` for precision.  The engine is a thread-safe
singleton (via ``threading.RLock``) and supports runtime registration of custom
emission factors for organisation-specific overrides.

Key capabilities:
    - NAICS-6 sector lookup with progressive prefix matching (6->5->4->3->2)
    - EEIO factor lookup across EPA USEEIO, EXIOBASE, WIOD, GTAP databases
    - Physical emission factor lookup by material/asset type
    - Supplier-specific emission factor registry
    - 8-level GHG Protocol EF hierarchy selection
    - NAICS <-> ISIC <-> NACE <-> UNSPSC cross-classification mapping
    - CPI deflation to base year (2021=100) with indices 2010-2026
    - Purchaser-to-producer margin removal by sector
    - Currency conversion for 20 ISO 4217 currencies
    - Asset classification by category/subcategory with confidence scoring
    - Capitalization threshold checking per accounting policy
    - Useful life range lookup with sub-category overrides

Data Sources:
    - EPA USEEIO v1.2 (1,016 commodities, 2021 USD basis)
    - EXIOBASE 3.8 multi-regional IO (163 sectors x 49 regions)
    - ICE Database v3.0 (Inventory of Carbon & Energy, Univ. Bath)
    - UK DEFRA/DESNZ Conversion Factors 2023
    - World Steel Association Environmental Data 2023
    - International Aluminium Institute Factors 2023
    - BLS Consumer Price Index (CPI-U, base year 2021=100)

Example:
    >>> from greenlang.capital_goods.capital_asset_database import (
    ...     CapitalAssetDatabaseEngine,
    ... )
    >>> db = CapitalAssetDatabaseEngine()
    >>> factor = db.get_eeio_factor("333120")
    >>> print(factor.factor_kg_co2e_per_usd)  # Decimal('0.35')
    >>> pef = db.get_physical_ef("structural_steel")
    >>> print(pef.factor_kg_co2e_per_unit)  # Decimal('1.55')
    >>> usd = db.convert_currency(Decimal("10000"), "EUR", "USD")
    >>> deflated = db.deflate_to_base_year(usd, 2024, 2021)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.capital_goods.models import (
    AssetCategory,
    AssetSubCategory,
    AssetClassification,
    CalculationMethod,
    CapitalAssetRecord,
    CapitalizationThreshold,
    CurrencyCode,
    EEIODatabase,
    EEIOFactor,
    EF_HIERARCHY_PRIORITY,
    PhysicalEF,
    PhysicalEFSource,
    SupplierDataSource,
    SupplierEF,
    UsefulLifeRange,
    CAPITAL_EEIO_EMISSION_FACTORS,
    CAPITAL_PHYSICAL_EMISSION_FACTORS,
    CAPITAL_SECTOR_MARGIN_PERCENTAGES,
    CURRENCY_EXCHANGE_RATES,
    ASSET_USEFUL_LIFE_RANGES,
    ZERO,
    ONE,
    ONE_HUNDRED,
)
from greenlang.capital_goods.metrics import get_metrics
from greenlang.capital_goods.provenance import get_provenance

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level Constants
# =============================================================================

#: Agent identifier.
AGENT_ID: str = "GL-MRV-S3-002"

#: Engine identifier.
ENGINE_ID: str = "CapitalAssetDatabaseEngine"

#: Engine version.
ENGINE_VERSION: str = "1.0.0"

#: Decimal quantize template for 8 decimal places.
_Q8 = Decimal("0.00000001")

#: Decimal quantize template for 2 decimal places.
_Q2 = Decimal("0.01")

#: Default margin percentage when sector not found.
_DEFAULT_MARGIN_PCT = Decimal("20.0")

#: Default EEIO factor when no match found.
_DEFAULT_EEIO_FACTOR = Decimal("0.30")

#: Minimum NAICS prefix length for progressive matching.
_MIN_NAICS_PREFIX = 2


# =============================================================================
# NAICS Capital Sector Names (~80 entries)
# =============================================================================

NAICS_CAPITAL_SECTOR_NAMES: Dict[str, str] = {
    # ---- Construction (NAICS 236-237) ----
    "236210": "Industrial Building Construction",
    "236220": "Commercial & Institutional Building Construction",
    "236116": "New Multifamily Housing Construction (Except For-Sale)",
    "236117": "New Housing For-Sale Builders",
    "236118": "Residential Remodelers",
    "237110": "Water & Sewer Line and Related Structures Construction",
    "237120": "Oil & Gas Pipeline and Related Structures Construction",
    "237130": "Power & Communication Line Construction",
    "237210": "Land Subdivision",
    "237310": "Highway, Street, and Bridge Construction",
    "237990": "Other Heavy and Civil Engineering Construction",
    # ---- Machinery Manufacturing (NAICS 333) ----
    "333111": "Farm Machinery and Equipment Manufacturing",
    "333112": "Lawn and Garden Tractor and Equipment Manufacturing",
    "333120": "Construction Machinery Manufacturing",
    "333131": "Mining Machinery and Equipment Manufacturing",
    "333132": "Oil and Gas Field Machinery and Equipment Manufacturing",
    "333241": "Food Product Machinery Manufacturing",
    "333242": "Semiconductor Machinery Manufacturing",
    "333243": "Sawmill, Woodworking, and Paper Machinery Manufacturing",
    "333244": "Printing Machinery and Equipment Manufacturing",
    "333249": "Other Industrial Machinery Manufacturing",
    "333314": "Optical Instrument and Lens Manufacturing",
    "333316": "Photographic and Photocopying Equipment Manufacturing",
    "333318": "Other Commercial and Service Industry Machinery Manufacturing",
    "333413": "Industrial and Commercial Fan, Blower, and Pump Manufacturing",
    "333414": "Heating Equipment (Except Warm Air Furnaces) Manufacturing",
    "333415": "AC, Refrigeration, and Warm Air Heating Equipment Manufacturing",
    "333511": "Industrial Mold Manufacturing",
    "333514": "Special Die and Tool Manufacturing",
    "333515": "Cutting Tool and Machine Tool Accessory Manufacturing",
    "333517": "Machine Tool Manufacturing",
    "333519": "Rolling Mill and Metalworking Machinery Manufacturing",
    "333611": "Turbine and Turbine Generator Set Units Manufacturing",
    "333612": "Speed Changer, Industrial High-Speed Drive, and Gear Manufacturing",
    "333613": "Mechanical Power Transmission Equipment Manufacturing",
    "333618": "Other Engine Equipment Manufacturing",
    "333912": "Air and Gas Compressor Manufacturing",
    "333914": "Measuring, Dispensing, and Other Pumping Equipment Manufacturing",
    "333921": "Elevator and Moving Stairway Manufacturing",
    "333922": "Conveyor and Conveying Equipment Manufacturing",
    "333923": "Overhead Traveling Crane, Hoist, and Monorail System Manufacturing",
    "333924": "Industrial Truck, Tractor, Trailer, and Stacker Machinery Manufacturing",
    "333991": "Power-Driven Handtool Manufacturing",
    "333993": "Packaging Machinery Manufacturing",
    "333994": "Industrial Process Furnace and Oven Manufacturing",
    "333996": "Fluid Power Pump and Motor Manufacturing",
    "333997": "Scale and Balance Manufacturing",
    # ---- Computer & Electronic Products (NAICS 334) ----
    "334111": "Electronic Computer Manufacturing",
    "334112": "Computer Storage Device Manufacturing",
    "334118": "Computer Terminal and Other Computer Peripheral Equipment Manufacturing",
    "334210": "Telephone Apparatus Manufacturing",
    "334220": "Radio and Television Broadcasting Equipment Manufacturing",
    "334290": "Other Communications Equipment Manufacturing",
    "334310": "Audio and Video Equipment Manufacturing",
    "334413": "Semiconductor and Related Device Manufacturing",
    "334416": "Capacitor, Resistor, Coil, Transformer, and Inductor Manufacturing",
    "334419": "Other Electronic Component Manufacturing",
    "334511": "Search, Detection, Navigation, and Guidance Instruments Manufacturing",
    "334512": "Automatic Environmental Control Manufacturing",
    "334513": "Instruments for Measuring and Testing Manufacturing",
    "334515": "Instrument Manufacturing for Measuring Process Variables",
    "334516": "Analytical Laboratory Instrument Manufacturing",
    # ---- Electrical Equipment (NAICS 335) ----
    "335110": "Electric Lamp Bulb and Part Manufacturing",
    "335210": "Small Electrical Appliance Manufacturing",
    "335220": "Major Household Appliance Manufacturing",
    "335311": "Power, Distribution, and Specialty Transformer Manufacturing",
    "335312": "Motor and Generator Manufacturing",
    "335313": "Switchgear and Switchboard Apparatus Manufacturing",
    "335314": "Relay and Industrial Control Manufacturing",
    "335911": "Storage Battery Manufacturing",
    "335912": "Primary Battery Manufacturing",
    "335999": "All Other Miscellaneous Electrical Equipment Manufacturing",
    # ---- Transportation Equipment (NAICS 336) ----
    "336111": "Automobile Manufacturing",
    "336112": "Light Truck and Utility Vehicle Manufacturing",
    "336120": "Heavy Duty Truck Manufacturing",
    "336211": "Motor Vehicle Body Manufacturing",
    "336310": "Motor Vehicle Gasoline Engine and Parts Manufacturing",
    "336320": "Motor Vehicle Electrical and Electronic Equipment Manufacturing",
    "336330": "Motor Vehicle Steering and Suspension Parts Manufacturing",
    "336340": "Motor Vehicle Brake System Manufacturing",
    "336350": "Motor Vehicle Transmission and Power Train Parts Manufacturing",
    "336360": "Motor Vehicle Seating and Interior Trim Manufacturing",
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine and Engine Parts Manufacturing",
    "336510": "Railroad Rolling Stock Manufacturing",
    "336611": "Ship Building and Repairing",
    "336612": "Boat Building",
    # ---- Furniture (NAICS 337) ----
    "337110": "Wood Kitchen Cabinet and Countertop Manufacturing",
    "337121": "Upholstered Household Furniture Manufacturing",
    "337122": "Nonupholstered Wood Household Furniture Manufacturing",
    "337127": "Institutional Furniture Manufacturing",
    "337211": "Wood Office Furniture Manufacturing",
    "337214": "Office Furniture (Except Wood) Manufacturing",
    "337215": "Showcase, Partition, Shelving, and Locker Manufacturing",
    # ---- Medical & Surgical (NAICS 339) ----
    "339112": "Surgical and Medical Instrument Manufacturing",
    "339113": "Surgical Appliance and Supplies Manufacturing",
    "339114": "Dental Equipment and Supplies Manufacturing",
    "339115": "Ophthalmic Goods Manufacturing",
    # ---- Primary Metals (NAICS 327, 331) ----
    "327310": "Cement Manufacturing",
    "327320": "Ready-Mix Concrete Manufacturing",
    "327330": "Concrete Pipe, Brick, and Block Manufacturing",
    "327390": "Other Concrete Product Manufacturing",
    "331110": "Iron and Steel Mills and Ferroalloy Manufacturing",
    "331210": "Iron and Steel Pipe and Tube Manufacturing from Purchased Steel",
    "331313": "Alumina Refining and Primary Aluminum Production",
    "331314": "Secondary Smelting and Alloying of Aluminum",
    "331420": "Copper Rolling, Drawing, Extruding, and Alloying",
    "331491": "Nonferrous Metal Rolling, Drawing, and Extruding",
    "331511": "Iron Foundries",
    "331512": "Steel Investment Foundries",
    "331524": "Aluminum Foundries (Except Die-Casting)",
    # ---- Fabricated Metal Products (NAICS 332) ----
    "332111": "Iron and Steel Forging",
    "332112": "Nonferrous Forging",
    "332119": "Metal Crown, Closure, and Other Metal Stamping",
    "332312": "Fabricated Structural Metal Manufacturing",
    "332313": "Plate Work Manufacturing",
    "332410": "Power Boiler and Heat Exchanger Manufacturing",
    "332420": "Metal Tank (Heavy Gauge) Manufacturing",
    "332710": "Machine Shops",
    "332721": "Precision Turned Product Manufacturing",
    "332996": "Fabricated Pipe and Pipe Fitting Manufacturing",
    "332999": "All Other Miscellaneous Fabricated Metal Product Manufacturing",
}


# =============================================================================
# NAICS to ISIC Rev 4 Cross-Mapping (~60 entries)
# =============================================================================

NAICS_TO_ISIC: Dict[str, str] = {
    # Construction
    "236210": "4100",
    "236220": "4100",
    "236116": "4100",
    "236117": "4100",
    "236118": "4100",
    "237110": "4220",
    "237120": "4220",
    "237130": "4220",
    "237210": "4290",
    "237310": "4210",
    "237990": "4290",
    # Machinery
    "333111": "2821",
    "333112": "2821",
    "333120": "2824",
    "333131": "2824",
    "333132": "2824",
    "333249": "2829",
    "333318": "2829",
    "333413": "2812",
    "333415": "2819",
    "333511": "2591",
    "333515": "2593",
    "333517": "2822",
    "333611": "2811",
    "333912": "2812",
    "333922": "2816",
    "333923": "2816",
    "333924": "2816",
    "333996": "2813",
    # Electronics / IT
    "334111": "2620",
    "334112": "2620",
    "334118": "2620",
    "334210": "2630",
    "334220": "2630",
    "334290": "2630",
    "334413": "2610",
    "334511": "2651",
    "334513": "2651",
    # Electrical Equipment
    "335110": "2740",
    "335311": "2710",
    "335312": "2710",
    "335313": "2710",
    "335911": "2720",
    "335999": "2790",
    # Vehicles
    "336111": "2910",
    "336112": "2910",
    "336120": "2910",
    "336211": "2920",
    "336310": "2930",
    "336411": "3030",
    "336510": "3020",
    "336611": "3011",
    "336612": "3012",
    # Furniture
    "337110": "3100",
    "337211": "3100",
    "337214": "3100",
    # Medical
    "339112": "3250",
    "339113": "3250",
    # Primary Metals
    "327310": "2394",
    "327320": "2395",
    "331110": "2410",
    "331210": "2410",
    "331313": "2420",
    "331420": "2420",
    "331511": "2431",
}


# =============================================================================
# NACE Rev 2 to ISIC Rev 4 Cross-Mapping (~50 entries)
# =============================================================================

NACE_TO_ISIC: Dict[str, str] = {
    # Construction
    "F41.10": "4100",
    "F41.20": "4100",
    "F42.11": "4210",
    "F42.12": "4210",
    "F42.13": "4220",
    "F42.21": "4220",
    "F42.22": "4220",
    "F42.91": "4290",
    "F42.99": "4290",
    # Machinery
    "C28.11": "2811",
    "C28.12": "2812",
    "C28.13": "2813",
    "C28.14": "2814",
    "C28.15": "2815",
    "C28.21": "2821",
    "C28.22": "2822",
    "C28.23": "2823",
    "C28.24": "2824",
    "C28.25": "2825",
    "C28.29": "2829",
    "C28.30": "2821",
    "C28.41": "2822",
    "C28.49": "2829",
    "C28.91": "2811",
    "C28.92": "2824",
    "C28.93": "2825",
    "C28.94": "2826",
    "C28.95": "2829",
    "C28.96": "2829",
    "C28.99": "2829",
    # Electronics
    "C26.11": "2610",
    "C26.12": "2610",
    "C26.20": "2620",
    "C26.30": "2630",
    "C26.40": "2640",
    "C26.51": "2651",
    "C26.52": "2652",
    # Electrical
    "C27.11": "2710",
    "C27.12": "2710",
    "C27.20": "2720",
    "C27.31": "2731",
    "C27.32": "2732",
    "C27.33": "2733",
    "C27.40": "2740",
    "C27.51": "2750",
    "C27.52": "2750",
    "C27.90": "2790",
    # Vehicles
    "C29.10": "2910",
    "C29.20": "2920",
    "C29.31": "2930",
    "C29.32": "2930",
    "C30.11": "3011",
    "C30.12": "3012",
    "C30.20": "3020",
    "C30.30": "3030",
}


# =============================================================================
# UNSPSC to NAICS Cross-Mapping (~40 entries)
# =============================================================================

UNSPSC_TO_NAICS: Dict[str, str] = {
    # Construction & Building
    "30100000": "236210",  # Structures and building
    "30110000": "236220",  # Commercial buildings
    "30120000": "236210",  # Industrial buildings
    "30150000": "237990",  # Prefabricated structures
    "30170000": "237310",  # Roadway construction
    # Machinery & Industrial Equipment
    "20100000": "333249",  # Mining machinery
    "20110000": "333120",  # Well drilling equipment
    "20120000": "333120",  # Heavy construction machinery
    "23100000": "333249",  # Industrial manufacturing
    "23150000": "333249",  # Industrial process machinery
    "23160000": "333413",  # Pumps and compressors
    "23170000": "333249",  # Material handling machinery
    "23180000": "333511",  # Industrial tooling
    "23200000": "333922",  # Conveying systems
    "23210000": "333923",  # Cranes and hoists
    "23240000": "333996",  # Fluid power systems
    "23250000": "333415",  # HVAC equipment
    "23270000": "333994",  # Industrial furnaces
    # IT & Electronics
    "43200000": "334111",  # Computer equipment
    "43210000": "334112",  # Computer storage
    "43220000": "334118",  # Computer peripherals
    "43230000": "334210",  # Telecommunications equipment
    "43240000": "334210",  # Telephone equipment
    "43250000": "334290",  # Network equipment
    # Electrical Equipment
    "26100000": "335311",  # Power transformers
    "26110000": "335312",  # Motors and generators
    "26120000": "335999",  # Electrical equipment
    "26130000": "335911",  # Batteries
    "26140000": "335314",  # Industrial controls
    # Vehicles
    "25100000": "336111",  # Motor vehicles
    "25110000": "336112",  # Light trucks
    "25120000": "336120",  # Heavy trucks
    "25130000": "336211",  # Specialty vehicles
    "25170000": "336411",  # Aircraft
    "25180000": "336510",  # Rail vehicles
    "25190000": "336612",  # Marine vessels
    # Furniture
    "56100000": "337211",  # Office furniture
    "56110000": "337214",  # Metal office furniture
    "56120000": "337215",  # Shelving and storage
    "56130000": "337127",  # Institutional furniture
    # Medical
    "42200000": "339112",  # Surgical instruments
    "42210000": "339113",  # Surgical appliances
}


# =============================================================================
# AssetCategory to NAICS Code Mapping
# =============================================================================

ASSET_CATEGORY_TO_NAICS: Dict[str, List[str]] = {
    AssetCategory.BUILDINGS.value: [
        "236210", "236220", "236116", "236117", "236118",
        "237110", "237120", "237130", "237210", "237310", "237990",
    ],
    AssetCategory.MACHINERY.value: [
        "333111", "333112", "333120", "333131", "333132",
        "333241", "333242", "333249", "333318", "333413",
        "333414", "333415", "333511", "333514", "333515",
        "333517", "333519", "333611", "333612", "333613",
        "333618", "333912", "333914", "333921", "333922",
        "333923", "333924", "333991", "333993", "333994",
        "333996", "333997",
    ],
    AssetCategory.EQUIPMENT.value: [
        "335311", "335312", "335313", "335314",
        "335911", "335912", "335999",
        "333415", "333912", "333611",
        "332410", "332420",
    ],
    AssetCategory.VEHICLES.value: [
        "336111", "336112", "336120", "336211",
        "336310", "336320", "336330", "336340",
        "336350", "336360", "336411", "336412",
        "336510", "336611", "336612",
    ],
    AssetCategory.IT_INFRASTRUCTURE.value: [
        "334111", "334112", "334118", "334210",
        "334220", "334290", "334310", "334413",
        "334416", "334419", "334511", "334512",
        "334513", "334515", "334516",
    ],
    AssetCategory.FURNITURE_FIXTURES.value: [
        "337110", "337121", "337122", "337127",
        "337211", "337214", "337215",
    ],
    AssetCategory.LAND_IMPROVEMENTS.value: [
        "237310", "237990", "237210",
        "327310", "327320", "327330", "327390",
    ],
    AssetCategory.LEASEHOLD_IMPROVEMENTS.value: [
        "236220", "236118",
        "332312", "332313",
    ],
}


# =============================================================================
# AssetSubCategory to NAICS Code Mapping
# =============================================================================

_SUBCATEGORY_TO_NAICS: Dict[str, str] = {
    AssetSubCategory.OFFICE_BUILDING.value: "236220",
    AssetSubCategory.WAREHOUSE.value: "236210",
    AssetSubCategory.MANUFACTURING_FACILITY.value: "236210",
    AssetSubCategory.RETAIL_STORE.value: "236220",
    AssetSubCategory.CNC_MACHINE.value: "333517",
    AssetSubCategory.PRESS.value: "333519",
    AssetSubCategory.CRANE.value: "333923",
    AssetSubCategory.CONVEYOR.value: "333922",
    AssetSubCategory.INDUSTRIAL_ROBOT.value: "333249",
    AssetSubCategory.HVAC.value: "333415",
    AssetSubCategory.ELECTRICAL_PANEL.value: "335313",
    AssetSubCategory.GENERATOR.value: "335312",
    AssetSubCategory.COMPRESSOR.value: "333912",
    AssetSubCategory.TRANSFORMER.value: "335311",
    AssetSubCategory.PASSENGER_CAR.value: "336111",
    AssetSubCategory.LIGHT_TRUCK.value: "336112",
    AssetSubCategory.HEAVY_TRUCK.value: "336120",
    AssetSubCategory.FORKLIFT.value: "333924",
    AssetSubCategory.VAN.value: "336112",
    AssetSubCategory.SERVER.value: "334111",
    AssetSubCategory.NETWORK_SWITCH.value: "334290",
    AssetSubCategory.STORAGE_ARRAY.value: "334112",
    AssetSubCategory.UPS.value: "335999",
    AssetSubCategory.RACK_ENCLOSURE.value: "334118",
    AssetSubCategory.OFFICE_DESK.value: "337211",
    AssetSubCategory.OFFICE_CHAIR.value: "337121",
    AssetSubCategory.SHELVING.value: "337215",
    AssetSubCategory.PARTITION.value: "337215",
    AssetSubCategory.PAVING.value: "237310",
    AssetSubCategory.LANDSCAPING.value: "237990",
    AssetSubCategory.FENCING.value: "332999",
    AssetSubCategory.DRAINAGE.value: "237110",
    AssetSubCategory.FITOUT_GENERAL.value: "236220",
    AssetSubCategory.INTERIOR_PARTITION.value: "332312",
    AssetSubCategory.FLOORING.value: "236118",
    AssetSubCategory.CEILING.value: "236118",
    AssetSubCategory.SOLAR_PANEL.value: "335999",
    AssetSubCategory.WIND_TURBINE.value: "333611",
    AssetSubCategory.BATTERY_STORAGE.value: "335911",
    AssetSubCategory.ELECTRIC_MOTOR.value: "335312",
}


# =============================================================================
# CPI Index Table (BLS CPI-U, base year 2021 = 100)
# =============================================================================

_CPI_INDICES: Dict[int, Decimal] = {
    2010: Decimal("82.4"),
    2011: Decimal("84.3"),
    2012: Decimal("86.0"),
    2013: Decimal("87.3"),
    2014: Decimal("88.7"),
    2015: Decimal("88.8"),
    2016: Decimal("90.0"),
    2017: Decimal("91.9"),
    2018: Decimal("94.2"),
    2019: Decimal("95.8"),
    2020: Decimal("97.0"),
    2021: Decimal("100.0"),
    2022: Decimal("108.0"),
    2023: Decimal("112.4"),
    2024: Decimal("115.2"),
    2025: Decimal("117.1"),
    2026: Decimal("118.7"),
}


# =============================================================================
# EXIOBASE EEIO Factors (kgCO2e per EUR, 2021 basis)
# Subset of capital-goods-relevant EXIOBASE 3.8 sectors
# =============================================================================

_EXIOBASE_FACTORS: Dict[str, Decimal] = {
    "236210": Decimal("0.51"),
    "236220": Decimal("0.45"),
    "237110": Decimal("0.47"),
    "237310": Decimal("0.49"),
    "333120": Decimal("0.38"),
    "333131": Decimal("0.40"),
    "333249": Decimal("0.35"),
    "333413": Decimal("0.30"),
    "333415": Decimal("0.36"),
    "333611": Decimal("0.42"),
    "333923": Decimal("0.39"),
    "334111": Decimal("0.31"),
    "334112": Decimal("0.29"),
    "334413": Decimal("0.25"),
    "335311": Decimal("0.33"),
    "335312": Decimal("0.32"),
    "336111": Decimal("0.41"),
    "336112": Decimal("0.39"),
    "336120": Decimal("0.44"),
    "336411": Decimal("0.48"),
    "336510": Decimal("0.43"),
    "337110": Decimal("0.20"),
    "337214": Decimal("0.19"),
    "339112": Decimal("0.21"),
    "331110": Decimal("0.58"),
    "331313": Decimal("0.51"),
    "327310": Decimal("0.55"),
}


# =============================================================================
# WIOD EEIO Factors (kgCO2e per USD, 2021 basis)
# Subset of capital-goods-relevant WIOD sectors
# =============================================================================

_WIOD_FACTORS: Dict[str, Decimal] = {
    "236210": Decimal("0.50"),
    "236220": Decimal("0.44"),
    "237310": Decimal("0.48"),
    "333120": Decimal("0.37"),
    "333249": Decimal("0.34"),
    "333611": Decimal("0.41"),
    "334111": Decimal("0.30"),
    "334413": Decimal("0.24"),
    "335311": Decimal("0.32"),
    "335312": Decimal("0.31"),
    "336111": Decimal("0.40"),
    "336120": Decimal("0.43"),
    "336411": Decimal("0.47"),
    "331110": Decimal("0.57"),
    "327310": Decimal("0.54"),
}


# =============================================================================
# GTAP EEIO Factors (kgCO2e per USD, 2021 basis)
# Subset of capital-goods-relevant GTAP v11 sectors
# =============================================================================

_GTAP_FACTORS: Dict[str, Decimal] = {
    "236210": Decimal("0.52"),
    "236220": Decimal("0.46"),
    "237310": Decimal("0.50"),
    "333120": Decimal("0.39"),
    "333249": Decimal("0.36"),
    "333611": Decimal("0.43"),
    "334111": Decimal("0.32"),
    "334413": Decimal("0.26"),
    "335311": Decimal("0.34"),
    "336111": Decimal("0.42"),
    "336120": Decimal("0.45"),
    "336411": Decimal("0.49"),
    "331110": Decimal("0.59"),
    "327310": Decimal("0.56"),
}


# =============================================================================
# Description-based subcategory keyword mapping
# =============================================================================

_SUBCATEGORY_KEYWORDS: Dict[AssetSubCategory, List[str]] = {
    # Buildings
    AssetSubCategory.OFFICE_BUILDING: ["office building", "headquarters", "admin building"],
    AssetSubCategory.WAREHOUSE: ["warehouse", "distribution center", "logistics center", "storage facility"],
    AssetSubCategory.MANUFACTURING_FACILITY: ["factory", "manufacturing", "plant", "production facility"],
    AssetSubCategory.RETAIL_STORE: ["retail", "store", "shop", "outlet"],
    # Machinery
    AssetSubCategory.CNC_MACHINE: ["cnc", "computer numerical control", "machining center"],
    AssetSubCategory.PRESS: ["press", "stamping", "hydraulic press", "mechanical press"],
    AssetSubCategory.CRANE: ["crane", "gantry", "overhead crane", "tower crane"],
    AssetSubCategory.CONVEYOR: ["conveyor", "belt system", "roller conveyor"],
    AssetSubCategory.INDUSTRIAL_ROBOT: ["robot", "robotic arm", "cobot", "automation"],
    # Equipment
    AssetSubCategory.HVAC: ["hvac", "air conditioning", "heating", "ventilation", "chiller"],
    AssetSubCategory.ELECTRICAL_PANEL: ["electrical panel", "switchboard", "switchgear", "panel board"],
    AssetSubCategory.GENERATOR: ["generator", "genset", "standby power", "backup generator"],
    AssetSubCategory.COMPRESSOR: ["compressor", "air compressor", "gas compressor"],
    AssetSubCategory.TRANSFORMER: ["transformer", "power transformer", "distribution transformer"],
    # Vehicles
    AssetSubCategory.PASSENGER_CAR: ["passenger car", "sedan", "automobile", "company car"],
    AssetSubCategory.LIGHT_TRUCK: ["light truck", "pickup", "suv", "utility vehicle"],
    AssetSubCategory.HEAVY_TRUCK: ["heavy truck", "semi", "tractor trailer", "lorry", "18-wheeler"],
    AssetSubCategory.FORKLIFT: ["forklift", "lift truck", "pallet truck"],
    AssetSubCategory.VAN: ["van", "cargo van", "delivery van", "sprinter"],
    # IT Infrastructure
    AssetSubCategory.SERVER: ["server", "rack server", "blade server", "compute node"],
    AssetSubCategory.NETWORK_SWITCH: ["network switch", "router", "firewall", "access point"],
    AssetSubCategory.STORAGE_ARRAY: ["storage array", "san", "nas", "disk array"],
    AssetSubCategory.UPS: ["ups", "uninterruptible power", "battery backup"],
    AssetSubCategory.RACK_ENCLOSURE: ["rack", "server cabinet", "rack enclosure"],
    # Furniture & Fixtures
    AssetSubCategory.OFFICE_DESK: ["desk", "workstation", "standing desk"],
    AssetSubCategory.OFFICE_CHAIR: ["chair", "office chair", "ergonomic chair"],
    AssetSubCategory.SHELVING: ["shelving", "racking", "storage rack"],
    AssetSubCategory.PARTITION: ["partition", "room divider", "cubicle"],
    # Land Improvements
    AssetSubCategory.PAVING: ["paving", "asphalt", "parking lot", "concrete pad"],
    AssetSubCategory.LANDSCAPING: ["landscaping", "garden", "lawn", "irrigation"],
    AssetSubCategory.FENCING: ["fencing", "fence", "gate", "barrier"],
    AssetSubCategory.DRAINAGE: ["drainage", "storm drain", "culvert", "sewer"],
    # Leasehold
    AssetSubCategory.FITOUT_GENERAL: ["fitout", "fit-out", "tenant improvement", "build-out"],
    AssetSubCategory.INTERIOR_PARTITION: ["interior partition", "drywall", "glass partition"],
    AssetSubCategory.FLOORING: ["flooring", "carpet", "tile floor", "vinyl floor"],
    AssetSubCategory.CEILING: ["ceiling", "suspended ceiling", "drop ceiling"],
    # Renewables
    AssetSubCategory.SOLAR_PANEL: ["solar panel", "photovoltaic", "pv module", "solar array"],
    AssetSubCategory.WIND_TURBINE: ["wind turbine", "wind generator", "wind farm"],
    AssetSubCategory.BATTERY_STORAGE: ["battery storage", "energy storage", "li-ion", "lithium"],
    AssetSubCategory.ELECTRIC_MOTOR: ["electric motor", "e-motor", "servo motor"],
}


# =============================================================================
# Default capitalization thresholds by accounting policy
# =============================================================================

_DEFAULT_CAPITALIZATION_THRESHOLDS: Dict[str, Decimal] = {
    "company_defined": Decimal("5000"),
    "ifrs": Decimal("5000"),
    "us_gaap": Decimal("5000"),
    "local_gaap": Decimal("2500"),
}


# =============================================================================
# Physical EF unit mapping for material types
# =============================================================================

_PHYSICAL_EF_UNITS: Dict[str, str] = {
    "structural_steel": "kg",
    "reinforcing_steel": "kg",
    "stainless_steel": "kg",
    "aluminum_sheet": "kg",
    "aluminum_extrusion": "kg",
    "copper_pipe": "kg",
    "copper_wire": "kg",
    "concrete_25mpa": "kg",
    "concrete_32mpa": "kg",
    "concrete_40mpa": "kg",
    "concrete_precast": "kg",
    "brick": "kg",
    "glass_float": "kg",
    "glass_tempered": "kg",
    "glass_double_glazed": "kg",
    "timber_softwood": "kg",
    "timber_hardwood": "kg",
    "timber_glulam": "kg",
    "plasterboard": "kg",
    "ceramic_tiles": "kg",
    "carpet": "kg",
    "vinyl_flooring": "kg",
    "paint_water_based": "kg",
    "paint_solvent_based": "kg",
    "insulation_mineral_wool": "kg",
    "insulation_eps": "kg",
    "insulation_xps": "kg",
    "insulation_pir": "kg",
    "pvc_pipe": "kg",
    "hdpe_pipe": "kg",
    "roofing_membrane": "kg",
    "asphalt": "kg",
    "led_panel_per_unit": "unit",
    "server_per_unit": "unit",
    "laptop_per_unit": "unit",
    "desktop_per_unit": "unit",
    "monitor_per_unit": "unit",
    "network_switch_per_unit": "unit",
    "ups_per_unit": "unit",
    "solar_panel_per_kw": "kW",
    "wind_turbine_per_mw": "MW",
    "battery_li_ion_per_kwh": "kWh",
    "transformer_per_unit": "unit",
    "electric_motor_per_kw": "kW",
}


# =============================================================================
# CapitalAssetDatabaseEngine
# =============================================================================


class CapitalAssetDatabaseEngine:
    """Thread-safe singleton reference data and emission factor lookup engine.

    Manages NAICS sector classifications, EEIO factors, physical emission
    factors, supplier-specific factors, currency conversion, CPI deflation,
    and sector margin removal for GHG Protocol Scope 3 Category 2 capital
    goods emissions calculations.

    The engine is implemented as a thread-safe singleton to ensure consistent
    reference data across all pipeline stages and concurrent requests.  All
    mutable state (custom factor registries) is protected by an
    ``threading.RLock`` to prevent data races.

    Implements the ZERO-HALLUCINATION principle: all emission factors and
    reference data are sourced from embedded deterministic dictionaries.
    No LLM calls are made for any numeric lookups or calculations.

    Attributes:
        _custom_eeio_factors: Registry for user-defined EEIO factors.
        _custom_physical_efs: Registry for user-defined physical EFs.
        _custom_supplier_efs: Registry for user-defined supplier EFs.

    Example:
        >>> db = CapitalAssetDatabaseEngine()
        >>> factor = db.get_eeio_factor("334111")
        >>> assert factor.factor_kg_co2e_per_usd == Decimal("0.28")
        >>> desc = db.get_naics_description("334111")
        >>> assert desc == "Electronic Computer Manufacturing"
    """

    _instance: Optional[CapitalAssetDatabaseEngine] = None
    _singleton_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> CapitalAssetDatabaseEngine:
        """Create or return the singleton instance (thread-safe)."""
        with cls._singleton_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        """Initialize the engine (only once due to singleton guard)."""
        if self._initialized:
            return
        self._lock = threading.RLock()
        self._custom_eeio_factors: Dict[str, EEIOFactor] = {}
        self._custom_physical_efs: Dict[str, PhysicalEF] = {}
        self._custom_supplier_efs: Dict[str, SupplierEF] = {}
        self._lookup_count: int = 0
        self._initialized = True
        logger.info(
            "%s v%s initialized: %d NAICS sectors, %d EEIO factors, "
            "%d physical EFs, %d CPI years",
            ENGINE_ID,
            ENGINE_VERSION,
            len(NAICS_CAPITAL_SECTOR_NAMES),
            len(CAPITAL_EEIO_EMISSION_FACTORS),
            len(CAPITAL_PHYSICAL_EMISSION_FACTORS),
            len(_CPI_INDICES),
        )

    # ------------------------------------------------------------------
    # 1. get_eeio_factor
    # ------------------------------------------------------------------

    def get_eeio_factor(
        self,
        naics_code: str,
        database: EEIODatabase = EEIODatabase.EPA_USEEIO,
    ) -> EEIOFactor:
        """Look up an EEIO emission factor by NAICS code and database.

        Performs progressive prefix matching from 6 digits down to 2
        digits if an exact match is not found.  Returns a structured
        ``EEIOFactor`` model with full provenance metadata.

        Args:
            naics_code: NAICS sector code (2-6 digits).
            database: EEIO database to query (default EPA_USEEIO).

        Returns:
            EEIOFactor with the matched factor and metadata.

        Raises:
            ValueError: If naics_code is empty or not numeric.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> f = db.get_eeio_factor("333120")
            >>> print(f.factor_kg_co2e_per_usd)  # Decimal('0.35')
        """
        start_time = time.monotonic()
        code = str(naics_code).strip()

        if not code or not code.isdigit():
            raise ValueError(
                f"Invalid NAICS code: '{naics_code}' must be numeric"
            )

        # Select factor dictionary based on database
        factor_dict = self._get_factor_dict_for_database(database)

        # Check custom factors first (thread-safe)
        with self._lock:
            if code in self._custom_eeio_factors:
                self._lookup_count += 1
                factor = self._custom_eeio_factors[code]
                self._record_lookup_metric(
                    "eeio_factor", "custom", time.monotonic() - start_time
                )
                logger.debug(
                    "EEIO factor lookup [custom]: NAICS=%s factor=%s",
                    code,
                    factor.factor_kg_co2e_per_usd,
                )
                return factor

        # Progressive prefix match on built-in factors
        matched_code, matched_factor = self._progressive_naics_match(
            code, factor_dict
        )

        # Build description from NAICS sector names
        description = NAICS_CAPITAL_SECTOR_NAMES.get(
            matched_code,
            NAICS_CAPITAL_SECTOR_NAMES.get(
                code, f"NAICS Sector {matched_code}"
            ),
        )

        # Determine region based on database
        region_map = {
            EEIODatabase.EPA_USEEIO: "US",
            EEIODatabase.EXIOBASE: "EU",
            EEIODatabase.WIOD: "GLOBAL",
            EEIODatabase.GTAP: "GLOBAL",
        }

        result = EEIOFactor(
            naics_code=matched_code,
            description=description,
            factor_kg_co2e_per_usd=matched_factor,
            source=database,
            year=2021,
            region=region_map.get(database, "US"),
        )

        with self._lock:
            self._lookup_count += 1

        elapsed = time.monotonic() - start_time
        self._record_lookup_metric("eeio_factor", database.value, elapsed)

        logger.debug(
            "EEIO factor lookup [%s]: NAICS=%s -> matched=%s factor=%s (%.3fms)",
            database.value,
            code,
            matched_code,
            matched_factor,
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 2. get_physical_ef
    # ------------------------------------------------------------------

    def get_physical_ef(
        self,
        material_type: str,
        source: PhysicalEFSource = PhysicalEFSource.ICE_DATABASE,
    ) -> PhysicalEF:
        """Look up a physical emission factor by material type.

        Returns a cradle-to-gate emission factor for the specified
        material or asset type.  Custom factors are checked first,
        then built-in factors from the ICE/DEFRA/ecoinvent databases.

        Args:
            material_type: Material key (e.g. 'structural_steel',
                'server_per_unit').
            source: EF source database (default ICE_DATABASE).

        Returns:
            PhysicalEF with factor, unit, and source metadata.

        Raises:
            ValueError: If material_type is empty.
            KeyError: If material_type not found in any registry.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> ef = db.get_physical_ef("structural_steel")
            >>> print(ef.factor_kg_co2e_per_unit)  # Decimal('1.55')
        """
        start_time = time.monotonic()
        key = str(material_type).strip().lower()

        if not key:
            raise ValueError("material_type must not be empty")

        # Check custom factors first
        with self._lock:
            if key in self._custom_physical_efs:
                self._lookup_count += 1
                ef = self._custom_physical_efs[key]
                self._record_lookup_metric(
                    "physical_ef", "custom", time.monotonic() - start_time
                )
                logger.debug(
                    "Physical EF lookup [custom]: material=%s factor=%s",
                    key, ef.factor_kg_co2e_per_unit,
                )
                return ef

        # Look up in built-in factors
        if key not in CAPITAL_PHYSICAL_EMISSION_FACTORS:
            raise KeyError(
                f"Physical emission factor not found for material: '{key}'. "
                f"Available: {sorted(CAPITAL_PHYSICAL_EMISSION_FACTORS.keys())[:10]}..."
            )

        factor_value = CAPITAL_PHYSICAL_EMISSION_FACTORS[key]
        unit = _PHYSICAL_EF_UNITS.get(key, "kg")

        result = PhysicalEF(
            material_type=key,
            factor_kg_co2e_per_unit=factor_value,
            unit=unit,
            source=source,
            region="GLOBAL",
        )

        with self._lock:
            self._lookup_count += 1

        elapsed = time.monotonic() - start_time
        self._record_lookup_metric("physical_ef", source.value, elapsed)

        logger.debug(
            "Physical EF lookup [%s]: material=%s factor=%s %s (%.3fms)",
            source.value,
            key,
            factor_value,
            unit,
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 3. get_supplier_ef
    # ------------------------------------------------------------------

    def get_supplier_ef(
        self,
        supplier_name: str,
        product_type: str,
    ) -> Optional[SupplierEF]:
        """Look up a supplier-specific emission factor.

        Searches the custom supplier EF registry for a match on
        supplier name and product type.  Returns None if no match.

        Args:
            supplier_name: Name of the capital goods supplier.
            product_type: Product or asset type description.

        Returns:
            SupplierEF if found, None otherwise.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.register_custom_supplier_ef(
            ...     "Komatsu", "excavator", Decimal("45000"), "epd"
            ... )
            >>> ef = db.get_supplier_ef("Komatsu", "excavator")
            >>> print(ef.ef_value)  # Decimal('45000')
        """
        start_time = time.monotonic()
        s_key = str(supplier_name).strip().lower()
        p_key = str(product_type).strip().lower()
        lookup_key = f"{s_key}::{p_key}"

        with self._lock:
            self._lookup_count += 1
            ef = self._custom_supplier_efs.get(lookup_key)

        elapsed = time.monotonic() - start_time
        self._record_lookup_metric(
            "supplier_ef", "custom" if ef else "miss", elapsed
        )

        if ef is not None:
            logger.debug(
                "Supplier EF lookup: supplier=%s product=%s factor=%s (%.3fms)",
                supplier_name, product_type, ef.ef_value, elapsed * 1000,
            )
        else:
            logger.debug(
                "Supplier EF lookup miss: supplier=%s product=%s (%.3fms)",
                supplier_name, product_type, elapsed * 1000,
            )
        return ef

    # ------------------------------------------------------------------
    # 4. classify_asset
    # ------------------------------------------------------------------

    def classify_asset(
        self,
        record: CapitalAssetRecord,
    ) -> AssetClassification:
        """Classify a capital asset by category, subcategory, and NAICS.

        Uses the record's existing category and description to resolve
        the most specific subcategory and NAICS code.  Computes a
        confidence score based on data completeness.

        Args:
            record: Capital asset record to classify.

        Returns:
            AssetClassification with category, subcategory, NAICS code,
            and confidence score.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> record = CapitalAssetRecord(
            ...     asset_category=AssetCategory.MACHINERY,
            ...     description="CNC machining center",
            ...     acquisition_date=date(2024, 6, 15),
            ...     capex_amount=Decimal("250000"),
            ... )
            >>> cls = db.classify_asset(record)
            >>> print(cls.subcategory)  # AssetSubCategory.CNC_MACHINE
        """
        start_time = time.monotonic()

        category = record.asset_category
        subcategory = record.subcategory

        # Resolve subcategory from description if not provided
        if subcategory is None:
            subcategory = self._resolve_subcategory(
                record.description, category
            )

        # Resolve NAICS code: prefer record-level, then subcategory, then category default
        naics_code = record.naics_code
        if naics_code is None and subcategory is not None:
            naics_code = _SUBCATEGORY_TO_NAICS.get(subcategory.value)
        if naics_code is None:
            cat_codes = ASSET_CATEGORY_TO_NAICS.get(category.value, [])
            naics_code = cat_codes[0] if cat_codes else None

        # Calculate confidence
        confidence = self._calculate_confidence_score(
            has_naics=record.naics_code is not None,
            has_subcategory=record.subcategory is not None,
            has_weight=record.weight_kg is not None,
            has_supplier=record.supplier_name is not None,
        )

        # Build classification reason
        reason_parts: List[str] = []
        if record.subcategory is not None:
            reason_parts.append("Subcategory explicitly provided")
        elif subcategory is not None:
            reason_parts.append(
                f"Subcategory inferred from description: '{record.description[:80]}'"
            )
        if record.naics_code is not None:
            reason_parts.append(f"NAICS code provided: {record.naics_code}")
        elif naics_code is not None:
            reason_parts.append(f"NAICS code resolved from classification: {naics_code}")

        reason = "; ".join(reason_parts) if reason_parts else "Default classification"

        result = AssetClassification(
            asset_id=record.asset_id,
            category=category,
            subcategory=subcategory,
            naics_code=naics_code,
            nace_code=record.nace_code,
            is_capital=True,
            capitalization_met=True,
            classification_confidence=confidence,
            classification_reason=reason,
        )

        elapsed = time.monotonic() - start_time
        self._record_lookup_metric("classify_asset", category.value, elapsed)

        logger.debug(
            "Asset classified: id=%s cat=%s subcat=%s naics=%s conf=%.1f%% (%.3fms)",
            record.asset_id,
            category.value,
            subcategory.value if subcategory else "none",
            naics_code or "none",
            float(confidence),
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 5. check_capitalization
    # ------------------------------------------------------------------

    def check_capitalization(
        self,
        record: CapitalAssetRecord,
        threshold: Optional[CapitalizationThreshold] = None,
    ) -> bool:
        """Check if an asset meets the capitalization threshold.

        Determines whether the capital expenditure amount exceeds the
        minimum threshold for capitalization under the specified
        accounting policy.  If no explicit threshold is provided, the
        engine uses the default threshold for the record's policy.

        Args:
            record: Capital asset record to check.
            threshold: Explicit capitalization threshold (optional).

        Returns:
            True if the asset meets capitalization criteria.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> record = CapitalAssetRecord(
            ...     asset_category=AssetCategory.EQUIPMENT,
            ...     description="Air compressor",
            ...     acquisition_date=date(2024, 3, 1),
            ...     capex_amount=Decimal("12000"),
            ... )
            >>> db.check_capitalization(record)  # True
        """
        if threshold is not None:
            # Convert capex to threshold currency if different
            capex_in_threshold_ccy = self.convert_currency(
                record.capex_amount,
                record.currency.value,
                threshold.currency.value,
            )
            meets_amount = capex_in_threshold_ccy >= threshold.threshold_amount
            meets_life = True
            if record.useful_life_years is not None:
                meets_life = record.useful_life_years >= threshold.useful_life_min_years
            return meets_amount and meets_life

        # Use default threshold based on capitalization policy
        policy_key = record.capitalization_policy.value
        default_amount = _DEFAULT_CAPITALIZATION_THRESHOLDS.get(
            policy_key, Decimal("5000")
        )

        capex_usd = self.convert_currency(
            record.capex_amount,
            record.currency.value,
            CurrencyCode.USD.value,
        )
        return capex_usd >= default_amount

    # ------------------------------------------------------------------
    # 6. get_useful_life
    # ------------------------------------------------------------------

    def get_useful_life(
        self,
        category: AssetCategory,
        subcategory: Optional[AssetSubCategory] = None,
    ) -> UsefulLifeRange:
        """Get the useful life range for an asset category or subcategory.

        Checks subcategory-specific overrides first, then falls back
        to the category-level default ranges from ASSET_USEFUL_LIFE_RANGES.

        Args:
            category: Top-level asset category.
            subcategory: Detailed subcategory (optional, for overrides).

        Returns:
            UsefulLifeRange with min, max, and default years.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> life = db.get_useful_life(AssetCategory.VEHICLES)
            >>> print(life.default_years)  # 7
        """
        # Try subcategory override first
        if subcategory is not None:
            key = subcategory.value
            if key in ASSET_USEFUL_LIFE_RANGES:
                min_y, max_y, default_y = ASSET_USEFUL_LIFE_RANGES[key]
                return UsefulLifeRange(
                    asset_category=key,
                    min_years=min_y,
                    max_years=max_y,
                    default_years=default_y,
                )

        # Fall back to category level
        key = category.value
        if key in ASSET_USEFUL_LIFE_RANGES:
            min_y, max_y, default_y = ASSET_USEFUL_LIFE_RANGES[key]
            return UsefulLifeRange(
                asset_category=key,
                min_years=min_y,
                max_years=max_y,
                default_years=default_y,
            )

        # Ultimate fallback: generic range
        logger.warning(
            "No useful life range found for category=%s subcategory=%s, "
            "using generic fallback",
            category.value,
            subcategory.value if subcategory else "none",
        )
        return UsefulLifeRange(
            asset_category=category.value,
            min_years=3,
            max_years=50,
            default_years=10,
        )

    # ------------------------------------------------------------------
    # 7. convert_currency
    # ------------------------------------------------------------------

    def convert_currency(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str,
    ) -> Decimal:
        """Convert a monetary amount between currencies.

        Uses annual average exchange rates stored in
        CURRENCY_EXCHANGE_RATES (rates expressed as units of foreign
        currency per 1 USD).  All conversions route through USD as the
        base currency.

        Args:
            amount: Amount to convert.
            from_currency: Source ISO 4217 currency code.
            to_currency: Target ISO 4217 currency code.

        Returns:
            Converted amount quantized to 8 decimal places.

        Raises:
            ValueError: If either currency code is not supported.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> usd = db.convert_currency(Decimal("10000"), "EUR", "USD")
            >>> print(usd)  # Decimal('10821.33...')
        """
        from_ccy = from_currency.strip().upper()
        to_ccy = to_currency.strip().upper()

        if from_ccy == to_ccy:
            return amount

        # Resolve CurrencyCode enums
        try:
            from_enum = CurrencyCode(from_ccy)
        except ValueError:
            raise ValueError(
                f"Unsupported source currency: '{from_currency}'. "
                f"Supported: {[c.value for c in CurrencyCode]}"
            )
        try:
            to_enum = CurrencyCode(to_ccy)
        except ValueError:
            raise ValueError(
                f"Unsupported target currency: '{to_currency}'. "
                f"Supported: {[c.value for c in CurrencyCode]}"
            )

        from_rate = CURRENCY_EXCHANGE_RATES[from_enum]
        to_rate = CURRENCY_EXCHANGE_RATES[to_enum]

        # Convert: from_currency -> USD -> to_currency
        # amount_in_from / from_rate_per_usd = amount_in_usd
        # amount_in_usd * to_rate_per_usd = amount_in_to
        amount_usd = amount / from_rate
        result = (amount_usd * to_rate).quantize(_Q8, rounding=ROUND_HALF_UP)

        logger.debug(
            "Currency conversion: %s %s -> %s %s (via USD %s)",
            amount, from_ccy, result, to_ccy,
            amount_usd.quantize(_Q2, rounding=ROUND_HALF_UP),
        )
        return result

    # ------------------------------------------------------------------
    # 8. deflate_to_base_year
    # ------------------------------------------------------------------

    def deflate_to_base_year(
        self,
        amount: Decimal,
        from_year: int,
        to_year: int = 2021,
    ) -> Decimal:
        """Deflate a monetary amount from one year to another using CPI.

        Uses the BLS CPI-U index with 2021 as base year (100.0) to
        adjust nominal spend to real terms for consistent EEIO factor
        application.

        Formula: deflated = amount * (CPI_to_year / CPI_from_year)

        Args:
            amount: Nominal amount to deflate.
            from_year: Year of the nominal amount.
            to_year: Target base year (default 2021).

        Returns:
            Deflated amount quantized to 8 decimal places.

        Raises:
            ValueError: If CPI index is not available for either year.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> real = db.deflate_to_base_year(Decimal("100000"), 2024, 2021)
            >>> print(real)  # ~Decimal('86805.55...')
        """
        if from_year == to_year:
            return amount

        cpi_from = self.get_cpi_index(from_year)
        cpi_to = self.get_cpi_index(to_year)

        if cpi_from == ZERO:
            raise ValueError(
                f"CPI index for year {from_year} is zero; cannot deflate"
            )

        result = (amount * cpi_to / cpi_from).quantize(
            _Q8, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "CPI deflation: %s from %d (CPI=%.1f) to %d (CPI=%.1f) = %s",
            amount, from_year, cpi_from, to_year, cpi_to, result,
        )
        return result

    # ------------------------------------------------------------------
    # 9. remove_margin
    # ------------------------------------------------------------------

    def remove_margin(
        self,
        amount: Decimal,
        sector: str,
    ) -> Decimal:
        """Remove trade/transport margin from purchaser price.

        Converts purchaser-price CapEx to producer/basic price for
        more accurate EEIO factor application.  Uses sector-specific
        margin percentages from CAPITAL_SECTOR_MARGIN_PERCENTAGES.

        Formula: producer_price = amount * (1 - margin_pct / 100)

        Args:
            amount: Purchaser price amount.
            sector: Sector key for margin lookup.

        Returns:
            Producer price amount quantized to 8 decimal places.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> prod = db.remove_margin(Decimal("100000"), "machinery")
            >>> print(prod)  # Decimal('80000.00000000')
        """
        sector_key = str(sector).strip().lower()
        margin_pct = self.get_margin_percentage(sector_key)

        margin_factor = ONE - (margin_pct / ONE_HUNDRED)
        result = (amount * margin_factor).quantize(
            _Q8, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "Margin removal: %s * (1 - %.1f%%) = %s (sector=%s)",
            amount, float(margin_pct), result, sector_key,
        )
        return result

    # ------------------------------------------------------------------
    # 10. get_naics_description
    # ------------------------------------------------------------------

    def get_naics_description(self, code: str) -> str:
        """Get the human-readable description for a NAICS code.

        Args:
            code: NAICS code (2-6 digits).

        Returns:
            Description string, or 'Unknown NAICS sector' if not found.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.get_naics_description("334111")
            'Electronic Computer Manufacturing'
        """
        code = str(code).strip()
        desc = NAICS_CAPITAL_SECTOR_NAMES.get(code)
        if desc is None:
            logger.debug("NAICS description not found for code: %s", code)
            return f"Unknown NAICS sector ({code})"
        return desc

    # ------------------------------------------------------------------
    # 11. map_naics_to_isic
    # ------------------------------------------------------------------

    def map_naics_to_isic(self, naics_code: str) -> Optional[str]:
        """Map a NAICS code to the corresponding ISIC Rev 4 code.

        Uses progressive prefix matching if an exact match is not found.

        Args:
            naics_code: NAICS code (2-6 digits).

        Returns:
            ISIC Rev 4 code string, or None if no mapping exists.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.map_naics_to_isic("334111")
            '2620'
        """
        code = str(naics_code).strip()
        # Try exact match first
        if code in NAICS_TO_ISIC:
            return NAICS_TO_ISIC[code]

        # Progressive prefix matching
        for length in range(len(code) - 1, _MIN_NAICS_PREFIX - 1, -1):
            prefix = code[:length]
            for naics, isic in NAICS_TO_ISIC.items():
                if naics.startswith(prefix):
                    logger.debug(
                        "NAICS->ISIC prefix match: %s -> %s (via %s)",
                        code, isic, naics,
                    )
                    return isic

        logger.debug("No NAICS->ISIC mapping found for: %s", code)
        return None

    # ------------------------------------------------------------------
    # 12. map_nace_to_isic
    # ------------------------------------------------------------------

    def map_nace_to_isic(self, nace_code: str) -> Optional[str]:
        """Map a NACE Rev 2 code to the corresponding ISIC Rev 4 code.

        Args:
            nace_code: NACE Rev 2 classification code.

        Returns:
            ISIC Rev 4 code string, or None if no mapping exists.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.map_nace_to_isic("C26.20")
            '2620'
        """
        code = str(nace_code).strip()
        isic = NACE_TO_ISIC.get(code)
        if isic is None:
            logger.debug("No NACE->ISIC mapping found for: %s", code)
        return isic

    # ------------------------------------------------------------------
    # 13. map_unspsc_to_naics
    # ------------------------------------------------------------------

    def map_unspsc_to_naics(self, unspsc_code: str) -> Optional[str]:
        """Map a UNSPSC code to the corresponding NAICS code.

        Args:
            unspsc_code: UNSPSC product/service code.

        Returns:
            NAICS code string, or None if no mapping exists.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.map_unspsc_to_naics("43200000")
            '334111'
        """
        code = str(unspsc_code).strip()
        naics = UNSPSC_TO_NAICS.get(code)
        if naics is None:
            logger.debug("No UNSPSC->NAICS mapping found for: %s", code)
        return naics

    # ------------------------------------------------------------------
    # 14. get_category_naics_codes
    # ------------------------------------------------------------------

    def get_category_naics_codes(
        self, category: AssetCategory,
    ) -> List[str]:
        """Get all NAICS codes associated with an asset category.

        Args:
            category: Top-level asset category.

        Returns:
            List of NAICS code strings for the category.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> codes = db.get_category_naics_codes(AssetCategory.VEHICLES)
            >>> assert "336111" in codes
        """
        return list(ASSET_CATEGORY_TO_NAICS.get(category.value, []))

    # ------------------------------------------------------------------
    # 15. select_best_ef
    # ------------------------------------------------------------------

    def select_best_ef(
        self,
        record: CapitalAssetRecord,
        available_sources: Optional[List[str]] = None,
    ) -> Tuple[Any, str]:
        """Select the best emission factor using the 8-level hierarchy.

        Implements the GHG Protocol Scope 3 Technical Guidance EF
        hierarchy, checking from level 1 (supplier EPD verified) down
        to level 8 (global average EEIO fallback).

        The 8 hierarchy levels are:
            1. supplier_epd_verified
            2. supplier_pcf_verified
            3. supplier_cdp_unverified
            4. product_lca_ecoinvent
            5. material_avg_ice_defra
            6. industry_avg_physical
            7. regional_eeio_exiobase
            8. global_avg_eeio_fallback

        Args:
            record: Capital asset record for EF matching.
            available_sources: Optional list of available source keys
                to constrain the search.

        Returns:
            Tuple of (emission_factor_object, hierarchy_level_key).

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> ef, level = db.select_best_ef(record)
            >>> print(level)  # e.g. 'material_avg_ice_defra'
        """
        start_time = time.monotonic()
        sources = available_sources or list(EF_HIERARCHY_PRIORITY.keys())

        # Level 1-3: Supplier-specific factors
        if record.supplier_name:
            # Level 1: supplier_epd_verified
            if "supplier_epd_verified" in sources:
                ef = self.get_supplier_ef(
                    record.supplier_name,
                    record.subcategory.value if record.subcategory else record.asset_category.value,
                )
                if ef is not None and ef.verification == "verified":
                    if ef.source == SupplierDataSource.EPD:
                        self._record_lookup_metric(
                            "select_best_ef", "supplier_epd_verified",
                            time.monotonic() - start_time,
                        )
                        return ef, "supplier_epd_verified"

            # Level 2: supplier_pcf_verified
            if "supplier_pcf_verified" in sources:
                ef = self.get_supplier_ef(
                    record.supplier_name,
                    record.subcategory.value if record.subcategory else record.asset_category.value,
                )
                if ef is not None and ef.verification == "verified":
                    if ef.source == SupplierDataSource.PCF:
                        self._record_lookup_metric(
                            "select_best_ef", "supplier_pcf_verified",
                            time.monotonic() - start_time,
                        )
                        return ef, "supplier_pcf_verified"

            # Level 3: supplier_cdp_unverified
            if "supplier_cdp_unverified" in sources:
                ef = self.get_supplier_ef(
                    record.supplier_name,
                    record.subcategory.value if record.subcategory else record.asset_category.value,
                )
                if ef is not None:
                    self._record_lookup_metric(
                        "select_best_ef", "supplier_cdp_unverified",
                        time.monotonic() - start_time,
                    )
                    return ef, "supplier_cdp_unverified"

        # Level 4: product_lca_ecoinvent (physical EF from ecoinvent)
        if "product_lca_ecoinvent" in sources and record.weight_kg is not None:
            subcategory_material = self._subcategory_to_material(record)
            if subcategory_material is not None:
                try:
                    ef = self.get_physical_ef(
                        subcategory_material,
                        source=PhysicalEFSource.ECOINVENT,
                    )
                    self._record_lookup_metric(
                        "select_best_ef", "product_lca_ecoinvent",
                        time.monotonic() - start_time,
                    )
                    return ef, "product_lca_ecoinvent"
                except KeyError:
                    pass

        # Level 5: material_avg_ice_defra (ICE or DEFRA physical EF)
        if "material_avg_ice_defra" in sources:
            subcategory_material = self._subcategory_to_material(record)
            if subcategory_material is not None:
                try:
                    ef = self.get_physical_ef(
                        subcategory_material,
                        source=PhysicalEFSource.ICE_DATABASE,
                    )
                    self._record_lookup_metric(
                        "select_best_ef", "material_avg_ice_defra",
                        time.monotonic() - start_time,
                    )
                    return ef, "material_avg_ice_defra"
                except KeyError:
                    pass

        # Level 6: industry_avg_physical (generic physical factor)
        if "industry_avg_physical" in sources:
            # Try generic material based on category
            generic_material = self._category_to_generic_material(
                record.asset_category
            )
            if generic_material is not None:
                try:
                    ef = self.get_physical_ef(
                        generic_material,
                        source=PhysicalEFSource.ICE_DATABASE,
                    )
                    self._record_lookup_metric(
                        "select_best_ef", "industry_avg_physical",
                        time.monotonic() - start_time,
                    )
                    return ef, "industry_avg_physical"
                except KeyError:
                    pass

        # Level 7: regional_eeio_exiobase
        if "regional_eeio_exiobase" in sources:
            naics_code = record.naics_code
            if naics_code is None:
                cat_codes = ASSET_CATEGORY_TO_NAICS.get(
                    record.asset_category.value, []
                )
                naics_code = cat_codes[0] if cat_codes else None
            if naics_code is not None:
                try:
                    ef = self.get_eeio_factor(
                        naics_code, database=EEIODatabase.EXIOBASE
                    )
                    self._record_lookup_metric(
                        "select_best_ef", "regional_eeio_exiobase",
                        time.monotonic() - start_time,
                    )
                    return ef, "regional_eeio_exiobase"
                except (ValueError, KeyError):
                    pass

        # Level 8: global_avg_eeio_fallback (EPA USEEIO)
        naics_code = record.naics_code
        if naics_code is None:
            cat_codes = ASSET_CATEGORY_TO_NAICS.get(
                record.asset_category.value, []
            )
            naics_code = cat_codes[0] if cat_codes else "333249"

        ef = self.get_eeio_factor(
            naics_code, database=EEIODatabase.EPA_USEEIO
        )
        self._record_lookup_metric(
            "select_best_ef", "global_avg_eeio_fallback",
            time.monotonic() - start_time,
        )
        return ef, "global_avg_eeio_fallback"

    # ------------------------------------------------------------------
    # 16. register_custom_eeio_factor
    # ------------------------------------------------------------------

    def register_custom_eeio_factor(
        self,
        naics_code: str,
        factor: Decimal,
        source: str = "custom",
    ) -> None:
        """Register a custom EEIO emission factor for a NAICS code.

        Custom factors take precedence over built-in factors during
        lookup.  Thread-safe via RLock.

        Args:
            naics_code: NAICS 6-digit sector code.
            factor: Emission factor in kgCO2e per USD.
            source: Description of the factor source.

        Raises:
            ValueError: If naics_code is empty or factor is negative.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.register_custom_eeio_factor("999999", Decimal("0.45"), "internal_study")
        """
        code = str(naics_code).strip()
        if not code:
            raise ValueError("naics_code must not be empty")
        if factor < ZERO:
            raise ValueError(
                f"Emission factor must be >= 0, got {factor}"
            )

        ef = EEIOFactor(
            naics_code=code,
            description=NAICS_CAPITAL_SECTOR_NAMES.get(
                code, f"Custom NAICS {code}"
            ),
            factor_kg_co2e_per_usd=factor,
            source=EEIODatabase.EPA_USEEIO,
            year=2021,
            region="CUSTOM",
        )

        with self._lock:
            self._custom_eeio_factors[code] = ef

        logger.info(
            "Registered custom EEIO factor: NAICS=%s factor=%s source=%s",
            code, factor, source,
        )

    # ------------------------------------------------------------------
    # 17. register_custom_physical_ef
    # ------------------------------------------------------------------

    def register_custom_physical_ef(
        self,
        material_type: str,
        factor: Decimal,
        unit: str = "kg",
        source: str = "custom",
    ) -> None:
        """Register a custom physical emission factor for a material.

        Custom factors take precedence over built-in factors during
        lookup.  Thread-safe via RLock.

        Args:
            material_type: Material key for the factor.
            factor: Emission factor in kgCO2e per unit.
            unit: Denominator unit (kg, unit, kW, etc.).
            source: Description of the factor source.

        Raises:
            ValueError: If material_type is empty or factor is negative.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.register_custom_physical_ef(
            ...     "recycled_steel", Decimal("0.85"), "kg", "custom_lca"
            ... )
        """
        key = str(material_type).strip().lower()
        if not key:
            raise ValueError("material_type must not be empty")
        if factor < ZERO:
            raise ValueError(
                f"Emission factor must be >= 0, got {factor}"
            )

        ef = PhysicalEF(
            material_type=key,
            factor_kg_co2e_per_unit=factor,
            unit=unit,
            source=PhysicalEFSource.CUSTOM,
            region="CUSTOM",
        )

        with self._lock:
            self._custom_physical_efs[key] = ef

        logger.info(
            "Registered custom physical EF: material=%s factor=%s/%s source=%s",
            key, factor, unit, source,
        )

    # ------------------------------------------------------------------
    # 18. register_custom_supplier_ef
    # ------------------------------------------------------------------

    def register_custom_supplier_ef(
        self,
        supplier_name: str,
        product_type: str,
        ef_value: Decimal,
        source: str = "direct_measurement",
    ) -> None:
        """Register a custom supplier-specific emission factor.

        Stores the factor in the supplier EF registry keyed by
        the normalized (supplier_name, product_type) pair.

        Args:
            supplier_name: Name of the supplier.
            product_type: Product or asset type.
            ef_value: Emission factor value (kgCO2e per unit).
            source: Source of the supplier data.

        Raises:
            ValueError: If supplier_name or product_type is empty,
                or ef_value is negative.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.register_custom_supplier_ef(
            ...     "Caterpillar", "excavator_320", Decimal("52000"), "epd"
            ... )
        """
        s_key = str(supplier_name).strip()
        p_key = str(product_type).strip()
        if not s_key:
            raise ValueError("supplier_name must not be empty")
        if not p_key:
            raise ValueError("product_type must not be empty")
        if ef_value < ZERO:
            raise ValueError(
                f"Emission factor must be >= 0, got {ef_value}"
            )

        # Resolve SupplierDataSource enum
        try:
            data_source = SupplierDataSource(source)
        except ValueError:
            data_source = SupplierDataSource.DIRECT_MEASUREMENT

        ef = SupplierEF(
            supplier_name=s_key,
            product_type=p_key,
            ef_value=ef_value,
            ef_unit="kgCO2e/unit",
            source=data_source,
            verification="unverified",
        )

        lookup_key = f"{s_key.lower()}::{p_key.lower()}"

        with self._lock:
            self._custom_supplier_efs[lookup_key] = ef

        logger.info(
            "Registered custom supplier EF: supplier=%s product=%s "
            "factor=%s source=%s",
            s_key, p_key, ef_value, source,
        )

    # ------------------------------------------------------------------
    # 19. get_all_eeio_factors
    # ------------------------------------------------------------------

    def get_all_eeio_factors(self) -> Dict[str, Decimal]:
        """Get all built-in and custom EEIO emission factors.

        Returns a merged dictionary with custom factors overriding
        built-in factors for the same NAICS code.

        Returns:
            Dict mapping NAICS code to factor (kgCO2e per USD).

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> factors = db.get_all_eeio_factors()
            >>> print(len(factors))  # 50+
        """
        result = dict(CAPITAL_EEIO_EMISSION_FACTORS)
        with self._lock:
            for code, ef in self._custom_eeio_factors.items():
                result[code] = ef.factor_kg_co2e_per_usd
        return result

    # ------------------------------------------------------------------
    # 20. get_all_physical_efs
    # ------------------------------------------------------------------

    def get_all_physical_efs(self) -> Dict[str, Decimal]:
        """Get all built-in and custom physical emission factors.

        Returns a merged dictionary with custom factors overriding
        built-in factors for the same material type.

        Returns:
            Dict mapping material_type to factor (kgCO2e per unit).

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> efs = db.get_all_physical_efs()
            >>> print(len(efs))  # 40+
        """
        result = dict(CAPITAL_PHYSICAL_EMISSION_FACTORS)
        with self._lock:
            for key, ef in self._custom_physical_efs.items():
                result[key] = ef.factor_kg_co2e_per_unit
        return result

    # ------------------------------------------------------------------
    # 21. get_all_supplier_efs
    # ------------------------------------------------------------------

    def get_all_supplier_efs(self) -> Dict[str, SupplierEF]:
        """Get all registered supplier-specific emission factors.

        Returns:
            Dict mapping 'supplier::product' key to SupplierEF.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> efs = db.get_all_supplier_efs()
        """
        with self._lock:
            return dict(self._custom_supplier_efs)

    # ------------------------------------------------------------------
    # 22. get_cpi_index
    # ------------------------------------------------------------------

    def get_cpi_index(self, year: int) -> Decimal:
        """Get the CPI-U index value for a given year.

        Uses BLS CPI-U with base year 2021 = 100.0.

        Args:
            year: Calendar year (2010-2026).

        Returns:
            CPI index value as Decimal.

        Raises:
            ValueError: If the year is not in the CPI table.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.get_cpi_index(2021)
            Decimal('100.0')
        """
        if year not in _CPI_INDICES:
            raise ValueError(
                f"CPI index not available for year {year}. "
                f"Available years: {sorted(_CPI_INDICES.keys())}"
            )
        return _CPI_INDICES[year]

    # ------------------------------------------------------------------
    # 23. get_margin_percentage
    # ------------------------------------------------------------------

    def get_margin_percentage(self, sector: str) -> Decimal:
        """Get the trade/transport margin percentage for a sector.

        Args:
            sector: Sector key for margin lookup.

        Returns:
            Margin percentage as Decimal (e.g. Decimal('20.0') for 20%).

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.get_margin_percentage("machinery")
            Decimal('20.0')
        """
        key = str(sector).strip().lower()
        margin = CAPITAL_SECTOR_MARGIN_PERCENTAGES.get(key)
        if margin is None:
            logger.debug(
                "Margin not found for sector '%s', using default %.1f%%",
                key, float(_DEFAULT_MARGIN_PCT),
            )
            return _DEFAULT_MARGIN_PCT
        return margin

    # ------------------------------------------------------------------
    # 24. get_exchange_rate
    # ------------------------------------------------------------------

    def get_exchange_rate(self, currency: str) -> Decimal:
        """Get the exchange rate for a currency (units per 1 USD).

        Args:
            currency: ISO 4217 currency code.

        Returns:
            Exchange rate as Decimal.

        Raises:
            ValueError: If the currency code is not supported.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.get_exchange_rate("EUR")
            Decimal('0.92410000')
        """
        ccy = str(currency).strip().upper()
        try:
            ccy_enum = CurrencyCode(ccy)
        except ValueError:
            raise ValueError(
                f"Unsupported currency: '{currency}'. "
                f"Supported: {[c.value for c in CurrencyCode]}"
            )
        return CURRENCY_EXCHANGE_RATES[ccy_enum]

    # ------------------------------------------------------------------
    # 25. validate_naics_code
    # ------------------------------------------------------------------

    def validate_naics_code(self, code: str) -> bool:
        """Validate whether a NAICS code exists in the sector database.

        Checks both the NAICS_CAPITAL_SECTOR_NAMES and the built-in
        EEIO factor tables.  Also accepts valid prefix codes (2-5
        digits) if any 6-digit code starts with that prefix.

        Args:
            code: NAICS code to validate.

        Returns:
            True if the code is valid (found or prefix-matches).

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> db.validate_naics_code("334111")  # True
            >>> db.validate_naics_code("999999")  # False
        """
        code = str(code).strip()
        if not code or not code.isdigit():
            return False

        # Exact match in sector names
        if code in NAICS_CAPITAL_SECTOR_NAMES:
            return True

        # Exact match in EEIO factors
        if code in CAPITAL_EEIO_EMISSION_FACTORS:
            return True

        # Prefix match: check if any known code starts with this prefix
        for known_code in NAICS_CAPITAL_SECTOR_NAMES:
            if known_code.startswith(code):
                return True

        return False

    # ------------------------------------------------------------------
    # 26. get_database_stats
    # ------------------------------------------------------------------

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the reference database.

        Returns counts of all embedded and custom data tables,
        cumulative lookup counts, and engine version information.

        Returns:
            Dictionary with database statistics.

        Example:
            >>> db = CapitalAssetDatabaseEngine()
            >>> stats = db.get_database_stats()
            >>> print(stats['naics_sector_count'])  # ~100+
        """
        with self._lock:
            custom_eeio_count = len(self._custom_eeio_factors)
            custom_physical_count = len(self._custom_physical_efs)
            custom_supplier_count = len(self._custom_supplier_efs)
            total_lookups = self._lookup_count

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "naics_sector_count": len(NAICS_CAPITAL_SECTOR_NAMES),
            "naics_to_isic_count": len(NAICS_TO_ISIC),
            "nace_to_isic_count": len(NACE_TO_ISIC),
            "unspsc_to_naics_count": len(UNSPSC_TO_NAICS),
            "asset_category_count": len(ASSET_CATEGORY_TO_NAICS),
            "builtin_eeio_factor_count": len(CAPITAL_EEIO_EMISSION_FACTORS),
            "builtin_physical_ef_count": len(CAPITAL_PHYSICAL_EMISSION_FACTORS),
            "exiobase_factor_count": len(_EXIOBASE_FACTORS),
            "wiod_factor_count": len(_WIOD_FACTORS),
            "gtap_factor_count": len(_GTAP_FACTORS),
            "cpi_year_count": len(_CPI_INDICES),
            "cpi_year_range": f"{min(_CPI_INDICES.keys())}-{max(_CPI_INDICES.keys())}",
            "margin_sector_count": len(CAPITAL_SECTOR_MARGIN_PERCENTAGES),
            "currency_count": len(CURRENCY_EXCHANGE_RATES),
            "useful_life_entry_count": len(ASSET_USEFUL_LIFE_RANGES),
            "subcategory_naics_count": len(_SUBCATEGORY_TO_NAICS),
            "custom_eeio_factor_count": custom_eeio_count,
            "custom_physical_ef_count": custom_physical_count,
            "custom_supplier_ef_count": custom_supplier_count,
            "total_factor_count": (
                len(CAPITAL_EEIO_EMISSION_FACTORS)
                + len(CAPITAL_PHYSICAL_EMISSION_FACTORS)
                + len(_EXIOBASE_FACTORS)
                + len(_WIOD_FACTORS)
                + len(_GTAP_FACTORS)
                + custom_eeio_count
                + custom_physical_count
                + custom_supplier_count
            ),
            "total_lookups": total_lookups,
        }

    # ==================================================================
    # Private Methods
    # ==================================================================

    def _get_factor_dict_for_database(
        self, database: EEIODatabase,
    ) -> Dict[str, Decimal]:
        """Return the EEIO factor dictionary for the specified database.

        Args:
            database: EEIO database identifier.

        Returns:
            Dict mapping NAICS codes to emission factors.
        """
        if database == EEIODatabase.EPA_USEEIO:
            return CAPITAL_EEIO_EMISSION_FACTORS
        elif database == EEIODatabase.EXIOBASE:
            return _EXIOBASE_FACTORS
        elif database == EEIODatabase.WIOD:
            return _WIOD_FACTORS
        elif database == EEIODatabase.GTAP:
            return _GTAP_FACTORS
        else:
            logger.warning(
                "Unknown EEIO database '%s', falling back to EPA_USEEIO",
                database.value,
            )
            return CAPITAL_EEIO_EMISSION_FACTORS

    def _progressive_naics_match(
        self,
        code: str,
        factors_dict: Dict[str, Decimal],
    ) -> Tuple[str, Decimal]:
        """Progressively match a NAICS code from 6 to 2 digit prefix.

        Starts with the full code and progressively shortens the
        prefix until a match is found.  If no match is found at any
        prefix level, returns the default fallback factor.

        Args:
            code: NAICS code to match.
            factors_dict: Factor dictionary to search.

        Returns:
            Tuple of (matched_code, factor_value).
        """
        # Try exact match first
        if code in factors_dict:
            return code, factors_dict[code]

        # Progressive prefix match: 5->4->3->2 digits
        for length in range(min(len(code), 5), _MIN_NAICS_PREFIX - 1, -1):
            prefix = code[:length]
            for naics_code, factor in factors_dict.items():
                if naics_code.startswith(prefix):
                    logger.debug(
                        "Progressive NAICS match: %s -> %s (prefix=%s, factor=%s)",
                        code, naics_code, prefix, factor,
                    )
                    return naics_code, factor

        # Fallback: use default factor
        logger.warning(
            "No NAICS match found for '%s' in factor dict (%d entries); "
            "using default factor %s",
            code, len(factors_dict), _DEFAULT_EEIO_FACTOR,
        )
        return code, _DEFAULT_EEIO_FACTOR

    def _resolve_subcategory(
        self,
        description: str,
        category: AssetCategory,
    ) -> Optional[AssetSubCategory]:
        """Resolve the asset subcategory from description text.

        Uses keyword matching against the description string to
        determine the most specific subcategory.  Only considers
        subcategories relevant to the given parent category.

        Args:
            description: Human-readable asset description.
            category: Parent asset category for scoping.

        Returns:
            AssetSubCategory if resolved, None otherwise.
        """
        desc_lower = description.lower()

        # Map categories to their valid subcategory groups
        category_subcategories: Dict[AssetCategory, List[AssetSubCategory]] = {
            AssetCategory.BUILDINGS: [
                AssetSubCategory.OFFICE_BUILDING,
                AssetSubCategory.WAREHOUSE,
                AssetSubCategory.MANUFACTURING_FACILITY,
                AssetSubCategory.RETAIL_STORE,
            ],
            AssetCategory.MACHINERY: [
                AssetSubCategory.CNC_MACHINE,
                AssetSubCategory.PRESS,
                AssetSubCategory.CRANE,
                AssetSubCategory.CONVEYOR,
                AssetSubCategory.INDUSTRIAL_ROBOT,
            ],
            AssetCategory.EQUIPMENT: [
                AssetSubCategory.HVAC,
                AssetSubCategory.ELECTRICAL_PANEL,
                AssetSubCategory.GENERATOR,
                AssetSubCategory.COMPRESSOR,
                AssetSubCategory.TRANSFORMER,
                AssetSubCategory.SOLAR_PANEL,
                AssetSubCategory.WIND_TURBINE,
                AssetSubCategory.BATTERY_STORAGE,
                AssetSubCategory.ELECTRIC_MOTOR,
            ],
            AssetCategory.VEHICLES: [
                AssetSubCategory.PASSENGER_CAR,
                AssetSubCategory.LIGHT_TRUCK,
                AssetSubCategory.HEAVY_TRUCK,
                AssetSubCategory.FORKLIFT,
                AssetSubCategory.VAN,
            ],
            AssetCategory.IT_INFRASTRUCTURE: [
                AssetSubCategory.SERVER,
                AssetSubCategory.NETWORK_SWITCH,
                AssetSubCategory.STORAGE_ARRAY,
                AssetSubCategory.UPS,
                AssetSubCategory.RACK_ENCLOSURE,
            ],
            AssetCategory.FURNITURE_FIXTURES: [
                AssetSubCategory.OFFICE_DESK,
                AssetSubCategory.OFFICE_CHAIR,
                AssetSubCategory.SHELVING,
                AssetSubCategory.PARTITION,
            ],
            AssetCategory.LAND_IMPROVEMENTS: [
                AssetSubCategory.PAVING,
                AssetSubCategory.LANDSCAPING,
                AssetSubCategory.FENCING,
                AssetSubCategory.DRAINAGE,
            ],
            AssetCategory.LEASEHOLD_IMPROVEMENTS: [
                AssetSubCategory.FITOUT_GENERAL,
                AssetSubCategory.INTERIOR_PARTITION,
                AssetSubCategory.FLOORING,
                AssetSubCategory.CEILING,
            ],
        }

        valid_subcats = category_subcategories.get(category, [])

        # Score each subcategory by keyword match count
        best_match: Optional[AssetSubCategory] = None
        best_score: int = 0

        for subcat in valid_subcats:
            keywords = _SUBCATEGORY_KEYWORDS.get(subcat, [])
            score = 0
            for keyword in keywords:
                if keyword in desc_lower:
                    score += len(keyword)  # Longer match = higher score
            if score > best_score:
                best_score = score
                best_match = subcat

        if best_match is not None:
            logger.debug(
                "Subcategory resolved: desc='%s' -> %s (score=%d)",
                description[:60], best_match.value, best_score,
            )
        return best_match

    def _calculate_confidence_score(
        self,
        has_naics: bool,
        has_subcategory: bool,
        has_weight: bool,
        has_supplier: bool,
    ) -> Decimal:
        """Calculate classification confidence score (0-100%).

        Scores are based on data completeness: each available data
        point increases confidence.

        Args:
            has_naics: Whether NAICS code was provided.
            has_subcategory: Whether subcategory was provided.
            has_weight: Whether weight data was provided.
            has_supplier: Whether supplier data was provided.

        Returns:
            Confidence percentage as Decimal (0-100).
        """
        # Base confidence: 50% for having a category (always present)
        score = Decimal("50.0")

        if has_naics:
            score += Decimal("20.0")
        if has_subcategory:
            score += Decimal("15.0")
        if has_weight:
            score += Decimal("10.0")
        if has_supplier:
            score += Decimal("5.0")

        return min(score, ONE_HUNDRED)

    def _validate_ef_source(
        self,
        source: str,
        category: AssetCategory,
    ) -> bool:
        """Validate that an EF source is appropriate for an asset category.

        Args:
            source: EF hierarchy level key.
            category: Asset category being calculated.

        Returns:
            True if the source is valid for the category.
        """
        return source in EF_HIERARCHY_PRIORITY

    def _subcategory_to_material(
        self,
        record: CapitalAssetRecord,
    ) -> Optional[str]:
        """Map an asset's subcategory to a physical material type key.

        Used by the EF hierarchy to find physical emission factors
        for assets when subcategory is known.

        Args:
            record: Capital asset record.

        Returns:
            Material type key, or None if no mapping exists.
        """
        subcat_material_map: Dict[str, str] = {
            AssetSubCategory.OFFICE_BUILDING.value: "concrete_32mpa",
            AssetSubCategory.WAREHOUSE.value: "structural_steel",
            AssetSubCategory.MANUFACTURING_FACILITY.value: "structural_steel",
            AssetSubCategory.RETAIL_STORE.value: "concrete_25mpa",
            AssetSubCategory.CNC_MACHINE.value: "structural_steel",
            AssetSubCategory.PRESS.value: "structural_steel",
            AssetSubCategory.CRANE.value: "structural_steel",
            AssetSubCategory.CONVEYOR.value: "structural_steel",
            AssetSubCategory.INDUSTRIAL_ROBOT.value: "aluminum_sheet",
            AssetSubCategory.HVAC.value: "copper_pipe",
            AssetSubCategory.GENERATOR.value: "structural_steel",
            AssetSubCategory.COMPRESSOR.value: "structural_steel",
            AssetSubCategory.TRANSFORMER.value: "transformer_per_unit",
            AssetSubCategory.PASSENGER_CAR.value: "structural_steel",
            AssetSubCategory.LIGHT_TRUCK.value: "structural_steel",
            AssetSubCategory.HEAVY_TRUCK.value: "structural_steel",
            AssetSubCategory.FORKLIFT.value: "structural_steel",
            AssetSubCategory.SERVER.value: "server_per_unit",
            AssetSubCategory.NETWORK_SWITCH.value: "network_switch_per_unit",
            AssetSubCategory.STORAGE_ARRAY.value: "server_per_unit",
            AssetSubCategory.UPS.value: "ups_per_unit",
            AssetSubCategory.OFFICE_DESK.value: "timber_softwood",
            AssetSubCategory.OFFICE_CHAIR.value: "timber_softwood",
            AssetSubCategory.SHELVING.value: "structural_steel",
            AssetSubCategory.PAVING.value: "asphalt",
            AssetSubCategory.FENCING.value: "structural_steel",
            AssetSubCategory.DRAINAGE.value: "pvc_pipe",
            AssetSubCategory.FLOORING.value: "vinyl_flooring",
            AssetSubCategory.CEILING.value: "plasterboard",
            AssetSubCategory.SOLAR_PANEL.value: "solar_panel_per_kw",
            AssetSubCategory.WIND_TURBINE.value: "wind_turbine_per_mw",
            AssetSubCategory.BATTERY_STORAGE.value: "battery_li_ion_per_kwh",
            AssetSubCategory.ELECTRIC_MOTOR.value: "electric_motor_per_kw",
        }

        if record.subcategory is not None:
            material = subcat_material_map.get(record.subcategory.value)
            if material is not None:
                return material

        return None

    def _category_to_generic_material(
        self,
        category: AssetCategory,
    ) -> Optional[str]:
        """Map an asset category to a generic material type for fallback.

        Args:
            category: Asset category.

        Returns:
            Generic material type key, or None.
        """
        category_material_map: Dict[str, str] = {
            AssetCategory.BUILDINGS.value: "concrete_32mpa",
            AssetCategory.MACHINERY.value: "structural_steel",
            AssetCategory.EQUIPMENT.value: "structural_steel",
            AssetCategory.VEHICLES.value: "structural_steel",
            AssetCategory.IT_INFRASTRUCTURE.value: "server_per_unit",
            AssetCategory.FURNITURE_FIXTURES.value: "timber_softwood",
            AssetCategory.LAND_IMPROVEMENTS.value: "asphalt",
            AssetCategory.LEASEHOLD_IMPROVEMENTS.value: "plasterboard",
        }
        return category_material_map.get(category.value)

    def _record_lookup_metric(
        self,
        lookup_type: str,
        source: str,
        elapsed_s: float,
    ) -> None:
        """Record a lookup metric (best-effort, no exceptions raised).

        Args:
            lookup_type: Type of lookup (eeio_factor, physical_ef, etc.).
            source: Source of the lookup result.
            elapsed_s: Elapsed time in seconds.
        """
        try:
            metrics = get_metrics()
            metrics.record_asset_lookup(
                source=source,
                category=lookup_type,
                count=1,
                duration_s=elapsed_s,
            )
        except Exception:
            # Metrics recording is non-critical; swallow errors
            pass

    # ==================================================================
    # Singleton reset (for testing only)
    # ==================================================================

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset the singleton instance (FOR TESTING ONLY).

        This method clears the singleton instance to allow fresh
        initialization in test scenarios.  NEVER call this in
        production code.
        """
        with cls._singleton_lock:
            cls._instance = None
        logger.warning(
            "%s singleton reset -- this should only happen in tests",
            ENGINE_ID,
        )


# =============================================================================
# Module-level convenience function
# =============================================================================


def get_capital_asset_database() -> CapitalAssetDatabaseEngine:
    """Get the singleton CapitalAssetDatabaseEngine instance.

    This is the recommended entry point for obtaining the engine.

    Returns:
        The singleton CapitalAssetDatabaseEngine instance.

    Example:
        >>> db = get_capital_asset_database()
        >>> factor = db.get_eeio_factor("334111")
    """
    return CapitalAssetDatabaseEngine()
