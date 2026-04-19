"""
PACK-020 Battery Passport Prep Pack - Configuration Manager

This module implements the complete configuration system for the Battery
Passport Prep Pack, covering all aspects of EU Battery Regulation (EU)
2023/1542 compliance. It provides Pydantic v2 models for carbon footprint
declaration, recycled content targets, performance and durability ratings,
supply chain due diligence, labelling and marking requirements, end-of-life
collection and recycling obligations, and conformity assessment procedures.

EU Battery Regulation (2023/1542) Key Articles:
    - Art. 7: Carbon footprint declaration (EV, industrial, LMT)
    - Art. 8: Carbon footprint performance classes (A-E)
    - Art. 9: Maximum carbon footprint thresholds
    - Art. 10: Performance and durability requirements
    - Art. 11: Removability and replaceability of portable batteries
    - Art. 13: Labelling and marking requirements
    - Art. 14: Information on state of health and expected lifetime
    - Art. 17: Recycled content documentation
    - Art. 22-26: Due diligence in the supply chain
    - Art. 27: Conformity assessment procedures
    - Art. 65: Battery passport (digital product passport)
    - Art. 69-73: Collection and recycling targets
    - Art. 71: Recycling efficiency rates

Battery Passport Data Requirements (Art. 65, Annex XIII):
    - Unique battery identifier
    - Battery manufacturer and manufacturing plant
    - Battery model and batch/serial number
    - Carbon footprint per lifecycle stage
    - Recycled content share by material (Co, Li, Ni, Pb)
    - State of health (SoH), capacity, power capability
    - Expected battery lifetime under reference conditions
    - Supply chain due diligence report summary
    - Collection and recycling information

Recycled Content Targets (Art. 8, Annex XII):
    Phase 1 (from 18 Aug 2031):
        Cobalt: 16%, Lithium: 6%, Nickel: 6%, Lead: 85%
    Phase 2 (from 18 Aug 2036):
        Cobalt: 26%, Lithium: 12%, Nickel: 15%, Lead: 85%

Carbon Footprint Performance Classes (Delegated Act):
    CLASS_A: Lowest footprint (top 10% of market)
    CLASS_B: Below average footprint (10-25%)
    CLASS_C: Average footprint (25-50%)
    CLASS_D: Above average footprint (50-75%)
    CLASS_E: Highest footprint (bottom 25%)

Collection Targets (Art. 69-70):
    Portable batteries: 45% by 2023, 63% by 2027, 73% by 2030
    LMT batteries: 51% by 2028, 61% by 2031
    EV/Industrial: 100% (take-back obligation)

Recycling Efficiency (Art. 71, Annex XII Part B):
    Li-ion: 65% by 2025, 70% by 2030
    Lead-acid: 75% by 2025, 80% by 2030
    NiCd: 80% (existing)
    Other: 50% (baseline)

Material Recovery Targets (Art. 71, Annex XII Part C):
    Cobalt: 90% by 2027, 95% by 2031
    Copper: 90% by 2027, 95% by 2031
    Lead: 90% by 2027, 95% by 2031
    Lithium: 50% by 2027, 80% by 2031
    Nickel: 90% by 2027, 95% by 2031

Configuration Merge Order (later overrides earlier):
    1. Base pack defaults
    2. Preset YAML (ev_battery / industrial_storage / lmt_battery /
       portable_battery / sli_battery / cell_manufacturer)
    3. Environment overrides (BATTERY_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EU Battery Regulation (EU) 2023/1542
    - Delegated Regulation on carbon footprint calculation methodology
    - Delegated Regulation on carbon footprint performance classes
    - Delegated Regulation on battery passport technical design
    - IEC 62660 (Secondary Li-ion cells for EV)
    - IEC 62619 (Secondary Li cells for industrial)
    - ISO 12405 (Electrically propelled vehicles - test specification)
    - UNECE GTR No. 22 (In-vehicle battery durability)
    - GBA Battery Passport Framework

Example:
    >>> config = PackConfig.from_preset("ev_battery")
    >>> print(config.pack.battery_category)
    BatteryCategory.EV
    >>> print(config.pack.carbon_footprint.max_threshold_per_kwh)
    None
    >>> print(config.pack.recycled_content.targets_2031)
    {'cobalt': 16.0, 'lithium': 6.0, 'nickel': 6.0, 'lead': 85.0}
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Battery Regulation enumeration types (10 enums)
# =============================================================================


class BatteryCategory(str, Enum):
    """Battery category classification per EU Battery Regulation Art. 1(2).

    The regulation defines five battery categories, each with distinct
    compliance obligations regarding passport, labelling, carbon footprint,
    recycled content, collection, and due diligence requirements.
    """

    EV = "EV"                    # Electric vehicle traction battery (>2kWh, vehicle propulsion)
    INDUSTRIAL = "INDUSTRIAL"    # Industrial battery (>2kWh, stationary energy storage)
    LMT = "LMT"                  # Light means of transport (e-bike, e-scooter, <=2kWh traction)
    PORTABLE = "PORTABLE"        # Portable battery (consumer electronics, <=5kg)
    SLI = "SLI"                  # Starting, lighting, ignition (automotive 12V/24V starter)


class BatteryChemistry(str, Enum):
    """Battery chemistry types for technical specification and compliance.

    Chemistry determines applicable emission factors, recycled content
    obligations, recycling efficiency targets, and end-of-life processing
    requirements under the Battery Regulation.
    """

    NMC = "NMC"                  # Lithium Nickel Manganese Cobalt Oxide (generic)
    NCA = "NCA"                  # Lithium Nickel Cobalt Aluminium Oxide
    LFP = "LFP"                  # Lithium Iron Phosphate
    NMC811 = "NMC811"            # NMC 8:1:1 (high nickel, low cobalt)
    NMC622 = "NMC622"            # NMC 6:2:2 (balanced)
    NMC532 = "NMC532"            # NMC 5:3:2 (conventional)
    LMO = "LMO"                  # Lithium Manganese Oxide (spinel)
    LTO = "LTO"                  # Lithium Titanate Oxide (anode chemistry)
    LEAD_ACID = "LEAD_ACID"      # Lead-acid (flooded, AGM, gel)
    NIMH = "NIMH"                # Nickel Metal Hydride
    ALKALINE = "ALKALINE"        # Alkaline (primary/rechargeable)
    SODIUM_ION = "SODIUM_ION"    # Sodium-ion (emerging)
    SOLID_STATE = "SOLID_STATE"  # Solid-state lithium (next-gen)


class LifecycleStage(str, Enum):
    """Battery lifecycle stages for carbon footprint disaggregation.

    Carbon footprint must be declared per lifecycle stage as specified
    in Art. 7 and the delegated act on calculation methodology. Each
    stage uses specific system boundaries and allocation rules.
    """

    RAW_MATERIAL_EXTRACTION = "RAW_MATERIAL_EXTRACTION"  # Mining, refining, precursor production
    MANUFACTURING = "MANUFACTURING"                       # Cell production, module/pack assembly
    DISTRIBUTION = "DISTRIBUTION"                         # Transport and logistics to point of sale
    END_OF_LIFE = "END_OF_LIFE"                           # Collection, recycling, disposal


class CriticalRawMaterial(str, Enum):
    """Critical raw materials subject to due diligence and recycled content.

    These materials are listed in Art. 22-26 (due diligence) and
    Art. 8/Annex XII (recycled content targets). Supply chain traceability
    and responsible sourcing documentation are mandatory.
    """

    COBALT = "COBALT"                    # Primary cathode material (NMC, NCA)
    LITHIUM = "LITHIUM"                  # Electrolyte and cathode component
    NICKEL = "NICKEL"                    # Cathode material (NMC, NCA)
    NATURAL_GRAPHITE = "NATURAL_GRAPHITE"  # Anode material
    MANGANESE = "MANGANESE"              # Cathode material (NMC, LMO)


class CarbonFootprintClass(str, Enum):
    """Carbon footprint performance class per Art. 8 delegated act.

    Performance classes are assigned based on the battery carbon footprint
    relative to peer group distribution. Thresholds are set by delegated
    act and updated periodically based on market data.
    """

    CLASS_A = "CLASS_A"  # Lowest footprint (top 10% of market)
    CLASS_B = "CLASS_B"  # Below average footprint (10th-25th percentile)
    CLASS_C = "CLASS_C"  # Average footprint (25th-50th percentile)
    CLASS_D = "CLASS_D"  # Above average footprint (50th-75th percentile)
    CLASS_E = "CLASS_E"  # Highest footprint (bottom 25%)


class ComplianceStatus(str, Enum):
    """Compliance status for individual regulatory requirements.

    Used to track assessment results for each article and
    obligation under the Battery Regulation.
    """

    COMPLIANT = "COMPLIANT"                    # Fully meets the requirement
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"  # Partially meets; remediation needed
    NON_COMPLIANT = "NON_COMPLIANT"            # Does not meet the requirement
    NOT_APPLICABLE = "NOT_APPLICABLE"          # Requirement not applicable to this battery
    PENDING = "PENDING"                        # Assessment not yet completed


class LabelElement(str, Enum):
    """Labelling elements required per Art. 13 and Annex VI.

    Each battery category has a specific set of required labelling
    elements. The label must be printed on the battery or its packaging
    and must be legible, indelible, and visible.
    """

    CE_MARKING = "CE_MARKING"                    # CE conformity marking
    QR_CODE = "QR_CODE"                          # QR code linking to battery passport
    COLLECTION_SYMBOL = "COLLECTION_SYMBOL"      # Separate collection symbol (crossed-out wheelie bin)
    CAPACITY_LABEL = "CAPACITY_LABEL"            # Rated capacity (Ah) and energy (Wh)
    HAZARDOUS_SUBSTANCE = "HAZARDOUS_SUBSTANCE"  # Hazardous substance identification (Cd, Pb, Hg)
    BATTERY_CHEMISTRY = "BATTERY_CHEMISTRY"      # Chemistry type designation
    CARBON_FOOTPRINT = "CARBON_FOOTPRINT"        # Carbon footprint class label (EV, industrial, LMT)
    SEPARATE_COLLECTION = "SEPARATE_COLLECTION"  # Separate collection information


class ConformityModule(str, Enum):
    """Conformity assessment modules per Art. 27 and Annex VIII.

    Different battery categories require different conformity assessment
    procedures. The manufacturer selects from applicable modules based
    on the battery category and production volume.
    """

    MODULE_A = "MODULE_A"  # Internal production control
    MODULE_B = "MODULE_B"  # EU-type examination
    MODULE_C = "MODULE_C"  # Conformity to type based on internal production control
    MODULE_D = "MODULE_D"  # Conformity to type based on quality assurance of production
    MODULE_E = "MODULE_E"  # Conformity to type based on product quality assurance
    MODULE_G = "MODULE_G"  # Conformity based on unit verification
    MODULE_H = "MODULE_H"  # Conformity based on full quality assurance


class CacheBackend(str, Enum):
    """Cache backend selection for emission factor and lookup caching."""

    MEMORY = "MEMORY"      # In-memory LRU cache (single process)
    REDIS = "REDIS"        # Redis-backed distributed cache
    DISABLED = "DISABLED"  # No caching (every lookup hits the database)


# =============================================================================
# Reference Data Constants
# =============================================================================

# Recycled content targets by phase (Art. 8, Annex XII Part A)
RECYCLED_CONTENT_TARGETS_2031: Dict[str, float] = {
    "cobalt": 16.0,
    "lithium": 6.0,
    "nickel": 6.0,
    "lead": 85.0,
}

RECYCLED_CONTENT_TARGETS_2036: Dict[str, float] = {
    "cobalt": 26.0,
    "lithium": 12.0,
    "nickel": 15.0,
    "lead": 85.0,
}

# Recycling efficiency targets by chemistry (Art. 71, Annex XII Part B)
RECYCLING_EFFICIENCY_TARGETS: Dict[str, Dict[str, float]] = {
    "lithium_ion": {"2025": 65.0, "2030": 70.0},
    "lead_acid": {"2025": 75.0, "2030": 80.0},
    "nickel_cadmium": {"current": 80.0},
    "other": {"current": 50.0},
}

# Material recovery targets (Art. 71, Annex XII Part C)
MATERIAL_RECOVERY_TARGETS: Dict[str, Dict[str, float]] = {
    "cobalt": {"2027": 90.0, "2031": 95.0},
    "copper": {"2027": 90.0, "2031": 95.0},
    "lead": {"2027": 90.0, "2031": 95.0},
    "lithium": {"2027": 50.0, "2031": 80.0},
    "nickel": {"2027": 90.0, "2031": 95.0},
}

# Collection targets by category (Art. 69-70)
COLLECTION_TARGETS: Dict[str, Dict[str, float]] = {
    "portable": {"2023": 45.0, "2027": 63.0, "2030": 73.0},
    "lmt": {"2028": 51.0, "2031": 61.0},
    "ev": {"current": 100.0},
    "industrial": {"current": 100.0},
    "sli": {"current": 100.0},
}

# Labelling requirements by battery category (Art. 13, Annex VI)
REQUIRED_LABEL_ELEMENTS: Dict[str, List[str]] = {
    "EV": [
        "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
        "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
        "SEPARATE_COLLECTION",
    ],
    "INDUSTRIAL": [
        "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
        "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
        "SEPARATE_COLLECTION",
    ],
    "LMT": [
        "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
        "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
        "SEPARATE_COLLECTION",
    ],
    "PORTABLE": [
        "CE_MARKING", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
        "HAZARDOUS_SUBSTANCE", "SEPARATE_COLLECTION",
    ],
    "SLI": [
        "CE_MARKING", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
        "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "SEPARATE_COLLECTION",
    ],
}

# Conformity assessment modules by category (Art. 27, Annex VIII)
CONFORMITY_MODULES_BY_CATEGORY: Dict[str, List[str]] = {
    "EV": ["MODULE_B", "MODULE_C", "MODULE_D", "MODULE_E", "MODULE_H"],
    "INDUSTRIAL": ["MODULE_B", "MODULE_C", "MODULE_D", "MODULE_E", "MODULE_H"],
    "LMT": ["MODULE_A", "MODULE_B", "MODULE_C", "MODULE_D"],
    "PORTABLE": ["MODULE_A"],
    "SLI": ["MODULE_A", "MODULE_B", "MODULE_C"],
}

# Battery passport requirement applicability by category (Art. 65)
PASSPORT_REQUIRED: Dict[str, bool] = {
    "EV": True,
    "INDUSTRIAL": True,
    "LMT": True,
    "PORTABLE": False,
    "SLI": False,
}

# Carbon footprint declaration applicability (Art. 7)
CARBON_FOOTPRINT_REQUIRED: Dict[str, bool] = {
    "EV": True,
    "INDUSTRIAL": True,
    "LMT": True,
    "PORTABLE": False,
    "SLI": False,
}

# Battery Regulation article reference information
BATTERY_REGULATION_ARTICLES: Dict[str, Dict[str, Any]] = {
    "Art. 7": {
        "name": "Carbon Footprint Declaration",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2025 (EV/Industrial), 18 Aug 2028 (LMT)",
        "description": "Carbon footprint per kWh over lifecycle, declared per production batch",
    },
    "Art. 8": {
        "name": "Carbon Footprint Performance Classes",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2026 (EV/Industrial), 18 Aug 2029 (LMT)",
        "description": "Assignment to performance class A-E based on carbon footprint",
    },
    "Art. 9": {
        "name": "Maximum Carbon Footprint Thresholds",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2028 (EV/Industrial), 18 Aug 2031 (LMT)",
        "description": "Maximum allowable carbon footprint per kWh (threshold TBD by DA)",
    },
    "Art. 10": {
        "name": "Performance and Durability Requirements",
        "scope": ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"],
        "mandatory": True,
        "timeline": "18 Aug 2025 (portable), 18 Feb 2027 (others)",
        "description": "Minimum performance and durability parameters by category",
    },
    "Art. 11": {
        "name": "Removability and Replaceability",
        "scope": ["PORTABLE", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2027",
        "description": "Portable batteries must be removable and replaceable by end user",
    },
    "Art. 13": {
        "name": "Labelling and Marking",
        "scope": ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"],
        "mandatory": True,
        "timeline": "18 Aug 2025",
        "description": "Category-specific labelling, CE marking, QR code, symbols",
    },
    "Art. 14": {
        "name": "State of Health and Expected Lifetime",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2027",
        "description": "Real-time SoH data accessible via battery management system",
    },
    "Art. 17": {
        "name": "Recycled Content Documentation",
        "scope": ["EV", "INDUSTRIAL", "LMT", "SLI"],
        "mandatory": True,
        "timeline": "18 Aug 2028 (documentation), 18 Aug 2031 (targets phase 1)",
        "description": "Documentation of recycled content share by material",
    },
    "Art. 22-26": {
        "name": "Supply Chain Due Diligence",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Aug 2025",
        "description": "Due diligence for cobalt, lithium, nickel, natural graphite, manganese",
    },
    "Art. 27": {
        "name": "Conformity Assessment",
        "scope": ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"],
        "mandatory": True,
        "timeline": "18 Aug 2025",
        "description": "Conformity assessment procedures (modules A-H by category)",
    },
    "Art. 65": {
        "name": "Battery Passport",
        "scope": ["EV", "INDUSTRIAL", "LMT"],
        "mandatory": True,
        "timeline": "18 Feb 2027",
        "description": "Digital product passport with unique identifier and QR code",
    },
    "Art. 69-70": {
        "name": "Collection Targets",
        "scope": ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"],
        "mandatory": True,
        "timeline": "Phase-in from 2023 to 2031",
        "description": "Waste battery collection rate targets by category",
    },
    "Art. 71": {
        "name": "Recycling Efficiency and Material Recovery",
        "scope": ["EV", "INDUSTRIAL", "LMT", "PORTABLE", "SLI"],
        "mandatory": True,
        "timeline": "2025 and 2027/2030/2031 phase-in",
        "description": "Recycling efficiency rates and material recovery targets",
    },
}

# Available presets for PACK-020
AVAILABLE_PRESETS: Dict[str, str] = {
    "ev_battery": "EV traction battery - full passport, all requirements, NMC/NCA/LFP",
    "industrial_storage": "Industrial stationary storage >2kWh - carbon footprint + passport",
    "lmt_battery": "Light means of transport (e-bike/e-scooter) - passport required",
    "portable_battery": "Portable consumer battery - labelling focus, no passport",
    "sli_battery": "Automotive SLI starter battery - performance + recycled content",
    "cell_manufacturer": "Cell production facility - carbon footprint + due diligence focus",
}

# Chemistry-to-critical-materials mapping
CHEMISTRY_CRITICAL_MATERIALS: Dict[str, List[str]] = {
    "NMC": ["COBALT", "LITHIUM", "NICKEL", "NATURAL_GRAPHITE", "MANGANESE"],
    "NCA": ["COBALT", "LITHIUM", "NICKEL", "NATURAL_GRAPHITE"],
    "LFP": ["LITHIUM", "NATURAL_GRAPHITE"],
    "NMC811": ["COBALT", "LITHIUM", "NICKEL", "NATURAL_GRAPHITE", "MANGANESE"],
    "NMC622": ["COBALT", "LITHIUM", "NICKEL", "NATURAL_GRAPHITE", "MANGANESE"],
    "NMC532": ["COBALT", "LITHIUM", "NICKEL", "NATURAL_GRAPHITE", "MANGANESE"],
    "LMO": ["LITHIUM", "NATURAL_GRAPHITE", "MANGANESE"],
    "LTO": ["LITHIUM"],
    "LEAD_ACID": [],
    "NIMH": ["NICKEL"],
    "ALKALINE": [],
    "SODIUM_ION": ["NATURAL_GRAPHITE"],
    "SOLID_STATE": ["LITHIUM", "NICKEL", "COBALT"],
}

# High-risk sourcing countries for due diligence (Art. 22-26)
HIGH_RISK_COUNTRIES: List[str] = [
    "CD",   # Democratic Republic of the Congo (cobalt)
    "CN",   # China (lithium, graphite, cobalt processing)
    "MM",   # Myanmar (tin, rare earths)
    "PH",   # Philippines (nickel)
    "ID",   # Indonesia (nickel)
    "RU",   # Russia (nickel, cobalt)
    "ZM",   # Zambia (cobalt)
    "CL",   # Chile (lithium - monitored)
    "AU",   # Australia (lithium - low risk but significant volume)
    "AR",   # Argentina (lithium - monitored)
    "ZW",   # Zimbabwe (lithium)
    "MZ",   # Mozambique (graphite)
    "MG",   # Madagascar (graphite, nickel)
]


# =============================================================================
# Pydantic Sub-Config Models (9 sub-config models)
# =============================================================================


class CarbonFootprintConfig(BaseModel):
    """Configuration for carbon footprint declaration engine.

    Implements Art. 7 carbon footprint declaration, Art. 8 performance
    class assignment, and Art. 9 maximum threshold compliance. The
    carbon footprint is expressed in kg CO2e per kWh of total energy
    provided by the battery over its expected service life.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon footprint calculation and declaration",
    )
    methodology: str = Field(
        "EU_BATTERY_REG_DA",
        description=(
            "Carbon footprint calculation methodology: EU_BATTERY_REG_DA "
            "(Delegated Act under Art. 7), PEF (Product Environmental Footprint), "
            "ISO_14067 (Carbon footprint of products)"
        ),
    )
    functional_unit: str = Field(
        "kgCO2e_per_kWh",
        description="Functional unit for carbon footprint: kg CO2e per kWh of total energy",
    )
    lifecycle_stages: List[LifecycleStage] = Field(
        default_factory=lambda: [
            LifecycleStage.RAW_MATERIAL_EXTRACTION,
            LifecycleStage.MANUFACTURING,
            LifecycleStage.DISTRIBUTION,
            LifecycleStage.END_OF_LIFE,
        ],
        description="Lifecycle stages included in carbon footprint calculation",
    )
    class_thresholds: Dict[str, Optional[float]] = Field(
        default_factory=lambda: {
            "CLASS_A": None,
            "CLASS_B": None,
            "CLASS_C": None,
            "CLASS_D": None,
            "CLASS_E": None,
        },
        description=(
            "Carbon footprint thresholds for performance class boundaries "
            "(kg CO2e/kWh). None means threshold not yet set by delegated act."
        ),
    )
    max_threshold_per_kwh: Optional[float] = Field(
        None,
        description=(
            "Maximum allowable carbon footprint per kWh (Art. 9). "
            "None means threshold not yet set by delegated act."
        ),
    )
    gwp_source: str = Field(
        "IPCC_AR6",
        description="GWP value source: IPCC_AR6 (100-year GWP per IPCC Sixth Assessment Report)",
    )
    emission_factor_database: str = Field(
        "ECOINVENT_3.10",
        description="Primary LCI database for background emission factors",
    )
    secondary_data_sources: List[str] = Field(
        default_factory=lambda: [
            "ECOINVENT_3.10",
            "GaBi_2024",
            "GREET_2024",
        ],
        description="Allowed secondary LCI data sources for background processes",
    )
    allocation_method: str = Field(
        "PHYSICAL",
        description="Allocation method for multi-output processes: PHYSICAL, ECONOMIC, SYSTEM_EXPANSION",
    )
    electricity_mix_source: str = Field(
        "COUNTRY_SPECIFIC",
        description=(
            "Source for grid electricity mix: COUNTRY_SPECIFIC (manufacturing country), "
            "SUPPLIER_SPECIFIC (contractual), EU_AVERAGE"
        ),
    )
    data_quality_rating_required: bool = Field(
        True,
        description="Require data quality rating (DQR) per EU PEF methodology for each data point",
    )
    third_party_verification: bool = Field(
        True,
        description="Require third-party verification of carbon footprint declaration",
    )
    declaration_per_batch: bool = Field(
        True,
        description="Calculate and declare carbon footprint per production batch (not per model)",
    )

    @field_validator("methodology")
    @classmethod
    def validate_methodology(cls, v: str) -> str:
        """Validate carbon footprint methodology is recognized."""
        valid = {"EU_BATTERY_REG_DA", "PEF", "ISO_14067", "GHG_PROTOCOL_PRODUCT"}
        if v not in valid:
            raise ValueError(
                f"Invalid methodology: {v}. Valid: {sorted(valid)}"
            )
        return v

    @field_validator("allocation_method")
    @classmethod
    def validate_allocation_method(cls, v: str) -> str:
        """Validate allocation method is recognized."""
        valid = {"PHYSICAL", "ECONOMIC", "SYSTEM_EXPANSION"}
        if v not in valid:
            raise ValueError(
                f"Invalid allocation method: {v}. Valid: {sorted(valid)}"
            )
        return v


class RecycledContentConfig(BaseModel):
    """Configuration for recycled content tracking engine.

    Implements Art. 8 recycled content documentation and target compliance.
    Tracks recycled content share for cobalt, lithium, nickel, and lead
    in active materials, against phase 1 (2031) and phase 2 (2036) targets.
    """

    enabled: bool = Field(
        True,
        description="Enable recycled content tracking and compliance checking",
    )
    targets_2031: Dict[str, float] = Field(
        default_factory=lambda: {
            "cobalt": 16.0,
            "lithium": 6.0,
            "nickel": 6.0,
            "lead": 85.0,
        },
        description="Phase 1 recycled content targets (% by weight, from 18 Aug 2031)",
    )
    targets_2036: Dict[str, float] = Field(
        default_factory=lambda: {
            "cobalt": 26.0,
            "lithium": 12.0,
            "nickel": 15.0,
            "lead": 85.0,
        },
        description="Phase 2 recycled content targets (% by weight, from 18 Aug 2036)",
    )
    documentation_required_from: str = Field(
        "2028-08-18",
        description="Date from which recycled content documentation is mandatory (ISO 8601)",
    )
    verification_method: str = Field(
        "MASS_BALANCE",
        description=(
            "Verification method for recycled content claims: "
            "MASS_BALANCE, CHAIN_OF_CUSTODY, BOOK_AND_CLAIM"
        ),
    )
    third_party_audit: bool = Field(
        True,
        description="Require third-party audit of recycled content declarations",
    )
    tracking_granularity: str = Field(
        "BATCH",
        description="Granularity for tracking: BATCH (per production batch), MODEL (per battery model)",
    )
    pre_consumer_included: bool = Field(
        True,
        description="Include pre-consumer recycled content in calculations",
    )
    post_consumer_included: bool = Field(
        True,
        description="Include post-consumer recycled content in calculations",
    )

    @field_validator("verification_method")
    @classmethod
    def validate_verification_method(cls, v: str) -> str:
        """Validate recycled content verification method."""
        valid = {"MASS_BALANCE", "CHAIN_OF_CUSTODY", "BOOK_AND_CLAIM"}
        if v not in valid:
            raise ValueError(
                f"Invalid verification method: {v}. Valid: {sorted(valid)}"
            )
        return v

    @field_validator("targets_2031", "targets_2036")
    @classmethod
    def validate_target_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate recycled content target percentages are in valid range."""
        for material, pct in v.items():
            if pct < 0.0 or pct > 100.0:
                raise ValueError(
                    f"Recycled content target for {material} must be 0-100%, got {pct}%"
                )
        return v


class PerformanceConfig(BaseModel):
    """Configuration for performance and durability assessment engine.

    Implements Art. 10 performance and durability requirements and
    Art. 14 state of health (SoH) data access. Covers capacity fade,
    power capability, cycle life, calendar life, and round-trip
    efficiency requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable performance and durability tracking",
    )
    soh_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "excellent": 95.0,
            "good": 85.0,
            "acceptable": 70.0,
            "degraded": 50.0,
            "end_of_life": 0.0,
        },
        description=(
            "State of Health (SoH) classification thresholds (%). "
            "SoH = remaining capacity / initial rated capacity * 100"
        ),
    )
    durability_ratings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ev_minimum_cycle_life": 1500,
            "ev_minimum_soh_at_eol": 70.0,
            "ev_calendar_life_years": 8,
            "ev_warranty_capacity_retention_pct": 70.0,
            "industrial_minimum_cycle_life": 4000,
            "industrial_minimum_soh_at_eol": 60.0,
            "industrial_calendar_life_years": 15,
            "industrial_round_trip_efficiency_pct": 85.0,
            "lmt_minimum_cycle_life": 1000,
            "lmt_minimum_soh_at_eol": 70.0,
            "portable_rated_capacity_deviation_pct": 5.0,
            "sli_cold_cranking_performance_retention_pct": 80.0,
        },
        description="Minimum durability parameters by battery category",
    )
    soh_data_access: bool = Field(
        True,
        description="Enable real-time SoH data access via BMS (Art. 14)",
    )
    soh_parameters: List[str] = Field(
        default_factory=lambda: [
            "remaining_capacity_ah",
            "remaining_capacity_pct",
            "remaining_power_capability_w",
            "remaining_power_capability_pct",
            "state_of_charge_pct",
            "state_of_energy_kwh",
            "cycle_count",
            "internal_resistance_mohm",
            "calendar_age_days",
            "temperature_history_summary",
            "fast_charge_events_count",
            "deep_discharge_events_count",
            "expected_remaining_lifetime_cycles",
            "expected_remaining_lifetime_months",
        ],
        description="SoH parameters to track and expose per Art. 14",
    )
    test_standards: List[str] = Field(
        default_factory=lambda: [
            "IEC_62660_1",   # Li-ion cells for EV - performance testing
            "IEC_62660_2",   # Li-ion cells for EV - reliability and abuse testing
            "IEC_62619",     # Li cells for industrial - safety requirements
            "IEC_61960",     # Secondary Li cells for portable - performance
            "ISO_12405_4",   # EV battery pack test specification
            "UNECE_GTR_22",  # In-vehicle battery durability
        ],
        description="Applicable test standards for performance validation",
    )
    capacity_measurement_standard: str = Field(
        "IEC_62660_1",
        description="Standard for rated capacity measurement",
    )
    temperature_range_c: Dict[str, float] = Field(
        default_factory=lambda: {
            "min_operating": -20.0,
            "max_operating": 45.0,
            "min_charging": 0.0,
            "max_charging": 45.0,
            "reference": 25.0,
        },
        description="Operating and reference temperature range in Celsius",
    )

    @field_validator("soh_thresholds")
    @classmethod
    def validate_soh_thresholds(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate SoH thresholds are in valid range and ordered."""
        for label, pct in v.items():
            if pct < 0.0 or pct > 100.0:
                raise ValueError(
                    f"SoH threshold '{label}' must be 0-100%, got {pct}%"
                )
        return v


class SupplyChainConfig(BaseModel):
    """Configuration for supply chain due diligence engine.

    Implements Art. 22-26 supply chain due diligence obligations for
    critical raw materials: cobalt, lithium, nickel, natural graphite,
    and manganese. Aligned with OECD Due Diligence Guidance for
    Responsible Supply Chains of Minerals from Conflict-Affected and
    High-Risk Areas.
    """

    enabled: bool = Field(
        True,
        description="Enable supply chain due diligence tracking",
    )
    critical_materials: List[CriticalRawMaterial] = Field(
        default_factory=lambda: [
            CriticalRawMaterial.COBALT,
            CriticalRawMaterial.LITHIUM,
            CriticalRawMaterial.NICKEL,
            CriticalRawMaterial.NATURAL_GRAPHITE,
            CriticalRawMaterial.MANGANESE,
        ],
        description="Critical raw materials subject to due diligence (Art. 22)",
    )
    high_risk_countries: List[str] = Field(
        default_factory=lambda: [
            "CD", "CN", "MM", "PH", "ID", "RU", "ZM", "CL", "ZW", "MZ", "MG",
        ],
        description="ISO 3166-1 alpha-2 country codes for high-risk sourcing regions",
    )
    due_diligence_framework: str = Field(
        "OECD_MINERALS",
        description=(
            "Due diligence framework: OECD_MINERALS (OECD Due Diligence Guidance), "
            "RMI_RMAP (Responsible Minerals Assurance Process), "
            "LME_RESPONSIBLE_SOURCING (London Metal Exchange)"
        ),
    )
    audit_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "third_party_audit_required": True,
            "audit_frequency_months": 12,
            "audit_standard": "OECD_MINERALS_ANNEX_II",
            "conflict_minerals_screening": True,
            "human_rights_assessment": True,
            "environmental_impact_assessment": True,
            "child_labour_zero_tolerance": True,
            "forced_labour_zero_tolerance": True,
            "tier_1_coverage_pct": 100,
            "tier_2_coverage_pct": 80,
            "tier_3_coverage_pct": 50,
            "smelter_refiner_identification": True,
            "mine_of_origin_traceability": True,
        },
        description="Audit requirements and coverage targets for due diligence",
    )
    grievance_mechanism: bool = Field(
        True,
        description="Maintain grievance mechanism for affected stakeholders (Art. 24)",
    )
    public_reporting: bool = Field(
        True,
        description="Publish annual due diligence report (Art. 26)",
    )
    chain_of_custody_tracking: bool = Field(
        True,
        description="Track chain of custody from mine to cell production",
    )
    responsible_sourcing_certifications: List[str] = Field(
        default_factory=lambda: [
            "IRMA",        # Initiative for Responsible Mining Assurance
            "RMI_RMAP",    # Responsible Minerals Assurance Process
            "ASI",         # Aluminium Stewardship Initiative (for lithium)
            "TSM",         # Towards Sustainable Mining
            "ICMM",       # International Council on Mining and Metals
        ],
        description="Accepted responsible sourcing certification schemes",
    )
    risk_assessment_frequency_months: int = Field(
        6,
        ge=1,
        le=24,
        description="Frequency of supply chain risk assessment in months",
    )

    @field_validator("due_diligence_framework")
    @classmethod
    def validate_dd_framework(cls, v: str) -> str:
        """Validate due diligence framework is recognized."""
        valid = {"OECD_MINERALS", "RMI_RMAP", "LME_RESPONSIBLE_SOURCING", "CUSTOM"}
        if v not in valid:
            raise ValueError(
                f"Invalid due diligence framework: {v}. Valid: {sorted(valid)}"
            )
        return v


class LabellingConfig(BaseModel):
    """Configuration for labelling and marking compliance engine.

    Implements Art. 13 labelling and marking requirements, including
    CE marking, QR code specifications, separate collection symbol,
    capacity labels, hazardous substance markings, and carbon footprint
    class labels per battery category.
    """

    enabled: bool = Field(
        True,
        description="Enable labelling compliance checking",
    )
    required_elements_by_category: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "EV": [
                "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
                "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
                "SEPARATE_COLLECTION",
            ],
            "INDUSTRIAL": [
                "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
                "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
                "SEPARATE_COLLECTION",
            ],
            "LMT": [
                "CE_MARKING", "QR_CODE", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
                "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "CARBON_FOOTPRINT",
                "SEPARATE_COLLECTION",
            ],
            "PORTABLE": [
                "CE_MARKING", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
                "HAZARDOUS_SUBSTANCE", "SEPARATE_COLLECTION",
            ],
            "SLI": [
                "CE_MARKING", "COLLECTION_SYMBOL", "CAPACITY_LABEL",
                "HAZARDOUS_SUBSTANCE", "BATTERY_CHEMISTRY", "SEPARATE_COLLECTION",
            ],
        },
        description="Required labelling elements per battery category (Art. 13, Annex VI)",
    )
    qr_code_spec: Dict[str, Any] = Field(
        default_factory=lambda: {
            "version": "QR_CODE_V2",
            "error_correction": "M",
            "min_size_mm": 16,
            "encoding": "UTF-8",
            "url_format": "https://passport.battery-reg.eu/{unique_id}",
            "must_link_to_passport": True,
            "must_link_to_declaration_of_conformity": True,
        },
        description="QR code technical specification per delegated act",
    )
    legibility_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "font_min_height_mm": 1.2,
            "contrast_ratio_min": 3.0,
            "indelible": True,
            "visible_without_opening": True,
            "language": "language_of_member_state",
        },
        description="Label legibility and durability requirements",
    )
    hazardous_substance_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "cadmium_ppm": 20.0,
            "lead_ppm": 40.0,
            "mercury_ppm": 5.0,
        },
        description="Hazardous substance concentration thresholds (ppm) triggering labelling",
    )

    @field_validator("required_elements_by_category")
    @classmethod
    def validate_label_elements(
        cls, v: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Validate all label elements are recognized."""
        valid_elements = {e.value for e in LabelElement}
        for category, elements in v.items():
            invalid = [e for e in elements if e not in valid_elements]
            if invalid:
                raise ValueError(
                    f"Invalid label elements for {category}: {invalid}. "
                    f"Valid: {sorted(valid_elements)}"
                )
        return v


class EOLConfig(BaseModel):
    """Configuration for end-of-life collection and recycling engine.

    Implements Art. 69-73 collection targets, Art. 71 recycling efficiency
    rates, and Annex XII material recovery targets. Covers collection
    infrastructure registration, recycling process documentation, and
    second-life battery requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable end-of-life compliance tracking",
    )
    collection_targets: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "portable": {"2023": 45.0, "2027": 63.0, "2030": 73.0},
            "lmt": {"2028": 51.0, "2031": 61.0},
            "ev": {"current": 100.0},
            "industrial": {"current": 100.0},
            "sli": {"current": 100.0},
        },
        description="Collection rate targets by category and year (%)",
    )
    recycling_targets: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "lithium_ion": {"2025": 65.0, "2030": 70.0},
            "lead_acid": {"2025": 75.0, "2030": 80.0},
            "nickel_cadmium": {"current": 80.0},
            "other": {"current": 50.0},
        },
        description="Recycling efficiency targets by chemistry type and year (%)",
    )
    recovery_targets: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "cobalt": {"2027": 90.0, "2031": 95.0},
            "copper": {"2027": 90.0, "2031": 95.0},
            "lead": {"2027": 90.0, "2031": 95.0},
            "lithium": {"2027": 50.0, "2031": 80.0},
            "nickel": {"2027": 90.0, "2031": 95.0},
        },
        description="Material recovery targets by material and year (%)",
    )
    second_life_assessment: bool = Field(
        True,
        description="Enable second-life (repurposing) assessment before recycling",
    )
    second_life_soh_threshold_pct: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum SoH (%) for a battery to be considered for second life",
    )
    waste_shipment_tracking: bool = Field(
        True,
        description="Track waste battery shipments per Basel Convention and EU WSR",
    )
    producer_responsibility: bool = Field(
        True,
        description="Enable extended producer responsibility (EPR) obligation tracking",
    )
    take_back_obligation: bool = Field(
        True,
        description="Track compliance with take-back obligation (EV, industrial, SLI)",
    )
    recycler_certification_required: bool = Field(
        True,
        description="Require recycler facility certification and permit validation",
    )

    @field_validator("collection_targets")
    @classmethod
    def validate_collection_targets(
        cls, v: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Validate collection target percentages are in valid range."""
        for category, targets in v.items():
            for year, pct in targets.items():
                if pct < 0.0 or pct > 100.0:
                    raise ValueError(
                        f"Collection target for {category}/{year} must be 0-100%, got {pct}%"
                    )
        return v


class ConformityConfig(BaseModel):
    """Configuration for conformity assessment engine.

    Implements Art. 27 and Annex VIII conformity assessment procedures.
    Determines applicable assessment modules by battery category and
    tracks documentation requirements for the EU Declaration of
    Conformity and technical file.
    """

    enabled: bool = Field(
        True,
        description="Enable conformity assessment tracking",
    )
    module_by_category: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "EV": ["MODULE_B", "MODULE_C", "MODULE_D", "MODULE_E", "MODULE_H"],
            "INDUSTRIAL": ["MODULE_B", "MODULE_C", "MODULE_D", "MODULE_E", "MODULE_H"],
            "LMT": ["MODULE_A", "MODULE_B", "MODULE_C", "MODULE_D"],
            "PORTABLE": ["MODULE_A"],
            "SLI": ["MODULE_A", "MODULE_B", "MODULE_C"],
        },
        description="Applicable conformity assessment modules by battery category",
    )
    documentation_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "technical_file": True,
            "eu_declaration_of_conformity": True,
            "test_reports": True,
            "risk_assessment": True,
            "design_drawings": True,
            "bill_of_materials": True,
            "manufacturing_process_description": True,
            "quality_management_system": True,
            "notified_body_certificate": False,
            "retention_period_years": 10,
        },
        description="Documentation requirements for conformity assessment",
    )
    notified_body_required: bool = Field(
        False,
        description=(
            "Whether a notified body is required (depends on module selection). "
            "Modules B, D, E, G, H require notified body involvement."
        ),
    )
    quality_management_standard: str = Field(
        "ISO_9001",
        description="Quality management system standard: ISO_9001, IATF_16949",
    )
    eu_doc_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for EU Declaration of Conformity",
    )

    @field_validator("module_by_category")
    @classmethod
    def validate_modules(
        cls, v: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Validate conformity module identifiers."""
        valid_modules = {m.value for m in ConformityModule}
        for category, modules in v.items():
            invalid = [m for m in modules if m not in valid_modules]
            if invalid:
                raise ValueError(
                    f"Invalid conformity modules for {category}: {invalid}. "
                    f"Valid: {sorted(valid_modules)}"
                )
        return v


class ReportingConfig(BaseModel):
    """Configuration for battery passport and compliance report generation."""

    enabled: bool = Field(
        True,
        description="Enable report and passport generation",
    )
    format: ReportFormat = Field(
        ReportFormat.PDF,
        description="Primary output format for compliance reports",
    )
    additional_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.JSON, ReportFormat.HTML],
        description="Additional output formats",
    )
    include_qr: bool = Field(
        True,
        description="Include QR code in generated reports linking to battery passport",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all calculated values",
    )
    passport_api_format: str = Field(
        "JSON_LD",
        description="Battery passport API data format: JSON_LD, JSON, XML",
    )
    passport_schema_version: str = Field(
        "1.0.0",
        description="Battery passport data schema version (per delegated act)",
    )
    review_workflow: bool = Field(
        True,
        description="Enable review and approval workflow before passport publication",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )
    audit_trail_report: bool = Field(
        True,
        description="Generate separate audit trail report for assurance",
    )
    multi_language: bool = Field(
        False,
        description="Enable multi-language report generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with key compliance metrics",
    )
    data_quality_disclosure: bool = Field(
        True,
        description="Disclose data quality ratings for carbon footprint and recycled content",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Report and passport data retention period in years",
    )

    @field_validator("passport_api_format")
    @classmethod
    def validate_passport_format(cls, v: str) -> str:
        """Validate passport API format."""
        valid = {"JSON_LD", "JSON", "XML"}
        if v not in valid:
            raise ValueError(
                f"Invalid passport API format: {v}. Valid: {sorted(valid)}"
            )
        return v


class CacheConfig(BaseModel):
    """Configuration for caching of emission factors and reference lookups."""

    backend: CacheBackend = Field(
        CacheBackend.MEMORY,
        description="Cache backend selection",
    )
    ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds (time-to-live)",
    )
    max_entries: int = Field(
        10000,
        ge=100,
        le=1000000,
        description="Maximum number of cached entries",
    )
    redis_url: Optional[str] = Field(
        None,
        description="Redis connection URL (required if backend is REDIS)",
    )
    warm_on_startup: bool = Field(
        True,
        description="Pre-warm cache with frequently used emission factors on startup",
    )

    @model_validator(mode="after")
    def validate_redis_config(self) -> "CacheConfig":
        """Validate Redis URL is provided when Redis backend is selected."""
        if self.backend == CacheBackend.REDIS and not self.redis_url:
            raise ValueError(
                "redis_url must be provided when backend is REDIS"
            )
        return self


# =============================================================================
# Main Configuration Model
# =============================================================================


class BatteryPassportConfig(BaseModel):
    """Main configuration for PACK-020 Battery Passport Prep Pack.

    This is the root configuration model that contains all sub-configurations
    for complete EU Battery Regulation compliance. The battery category and
    chemistry drive which articles are applicable, what labelling elements
    are required, which conformity modules are available, and what
    recycled content targets apply.
    """

    # Battery identification
    pack_name: str = Field(
        "PACK-020 Battery Passport Prep Pack",
        description="Pack display name",
    )
    version: str = Field(
        "1.0.0",
        description="Pack configuration schema version",
    )
    battery_category: BatteryCategory = Field(
        BatteryCategory.EV,
        description="Primary battery category (determines applicable requirements)",
    )
    chemistry: BatteryChemistry = Field(
        BatteryChemistry.NMC811,
        description="Battery chemistry type (determines critical materials and recycling targets)",
    )
    reporting_year: int = Field(
        2026,
        ge=2024,
        le=2035,
        description="Compliance reporting year",
    )
    manufacturer_name: str = Field(
        "",
        description="Battery manufacturer legal entity name",
    )
    manufacturing_country: str = Field(
        "DE",
        description="ISO 3166-1 alpha-2 country code of manufacturing plant",
    )
    battery_model: str = Field(
        "",
        description="Battery model designation",
    )
    rated_capacity_kwh: Optional[float] = Field(
        None,
        ge=0.001,
        description="Rated energy capacity in kWh",
    )
    rated_capacity_ah: Optional[float] = Field(
        None,
        ge=0.001,
        description="Rated capacity in Ah",
    )
    nominal_voltage_v: Optional[float] = Field(
        None,
        ge=1.0,
        description="Nominal voltage in V",
    )
    weight_kg: Optional[float] = Field(
        None,
        ge=0.001,
        description="Battery weight in kg",
    )

    # Engine sub-configurations
    carbon_footprint: CarbonFootprintConfig = Field(
        default_factory=CarbonFootprintConfig,
        description="Carbon footprint declaration configuration (Art. 7-9)",
    )
    recycled_content: RecycledContentConfig = Field(
        default_factory=RecycledContentConfig,
        description="Recycled content tracking configuration (Art. 8, Annex XII)",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and durability configuration (Art. 10, 14)",
    )
    supply_chain: SupplyChainConfig = Field(
        default_factory=SupplyChainConfig,
        description="Supply chain due diligence configuration (Art. 22-26)",
    )
    labelling: LabellingConfig = Field(
        default_factory=LabellingConfig,
        description="Labelling and marking configuration (Art. 13)",
    )
    eol: EOLConfig = Field(
        default_factory=EOLConfig,
        description="End-of-life collection and recycling configuration (Art. 69-73)",
    )
    conformity: ConformityConfig = Field(
        default_factory=ConformityConfig,
        description="Conformity assessment configuration (Art. 27)",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report and passport generation configuration",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration for emission factor lookups",
    )

    @model_validator(mode="after")
    def validate_category_requirements(self) -> "BatteryPassportConfig":
        """Apply category-specific requirement defaults and warnings."""
        category = self.battery_category.value

        # Disable carbon footprint for portable and SLI
        if category in ("PORTABLE", "SLI"):
            if self.carbon_footprint.enabled:
                logger.info(
                    "Carbon footprint declaration is not required for %s batteries. "
                    "Disabling carbon footprint engine.",
                    category,
                )
                object.__setattr__(
                    self.carbon_footprint, "enabled", False
                )

        # Disable passport-related features for portable
        if category == "PORTABLE":
            if self.performance.soh_data_access:
                logger.info(
                    "SoH data access (Art. 14) is not required for PORTABLE batteries. "
                    "Disabling SoH data access.",
                )
                object.__setattr__(
                    self.performance, "soh_data_access", False
                )

        # Warn about supply chain DD for portable
        if category == "PORTABLE" and self.supply_chain.enabled:
            logger.info(
                "Supply chain due diligence (Art. 22-26) is not required for "
                "PORTABLE batteries. Consider disabling.",
            )

        return self

    @model_validator(mode="after")
    def validate_chemistry_materials(self) -> "BatteryPassportConfig":
        """Validate supply chain materials match battery chemistry."""
        chemistry = self.chemistry.value
        required_materials = CHEMISTRY_CRITICAL_MATERIALS.get(chemistry, [])
        configured_materials = [m.value for m in self.supply_chain.critical_materials]

        missing = [m for m in required_materials if m not in configured_materials]
        if missing and self.supply_chain.enabled:
            logger.warning(
                "Chemistry %s requires due diligence for materials %s, "
                "but they are not configured in supply_chain.critical_materials.",
                chemistry,
                missing,
            )

        return self

    @model_validator(mode="after")
    def validate_eol_chemistry_match(self) -> "BatteryPassportConfig":
        """Validate recycling targets match battery chemistry type."""
        chemistry = self.chemistry.value
        li_ion_chemistries = {
            "NMC", "NCA", "LFP", "NMC811", "NMC622", "NMC532",
            "LMO", "LTO", "SODIUM_ION", "SOLID_STATE",
        }
        if chemistry in li_ion_chemistries:
            if "lithium_ion" not in self.eol.recycling_targets:
                logger.warning(
                    "Battery chemistry %s is lithium-ion type but 'lithium_ion' "
                    "recycling targets are not configured in eol.recycling_targets.",
                    chemistry,
                )
        elif chemistry == "LEAD_ACID":
            if "lead_acid" not in self.eol.recycling_targets:
                logger.warning(
                    "Battery chemistry LEAD_ACID but 'lead_acid' recycling targets "
                    "are not configured in eol.recycling_targets.",
                )
        return self

    def get_applicable_articles(self) -> List[str]:
        """Get list of applicable Battery Regulation articles for this category.

        Returns:
            List of article references applicable to the configured battery category.
        """
        category = self.battery_category.value
        applicable = []
        for article_ref, info in BATTERY_REGULATION_ARTICLES.items():
            if category in info["scope"]:
                applicable.append(article_ref)
        return applicable

    def get_required_label_elements(self) -> List[str]:
        """Get required labelling elements for the configured battery category.

        Returns:
            List of LabelElement values required for this category.
        """
        category = self.battery_category.value
        return REQUIRED_LABEL_ELEMENTS.get(category, [])

    def is_passport_required(self) -> bool:
        """Check if a battery passport is required for this category.

        Returns:
            True if battery passport is mandatory per Art. 65.
        """
        return PASSPORT_REQUIRED.get(self.battery_category.value, False)

    def is_carbon_footprint_required(self) -> bool:
        """Check if carbon footprint declaration is required.

        Returns:
            True if carbon footprint declaration is mandatory per Art. 7.
        """
        return CARBON_FOOTPRINT_REQUIRED.get(self.battery_category.value, False)

    def get_critical_materials(self) -> List[str]:
        """Get critical raw materials for this battery chemistry.

        Returns:
            List of critical material names requiring due diligence.
        """
        return CHEMISTRY_CRITICAL_MATERIALS.get(self.chemistry.value, [])


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging for PACK-020.
    """

    pack: BatteryPassportConfig = Field(
        default_factory=BatteryPassportConfig,
        description="Main Battery Passport configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-020-battery-passport-prep",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (ev_battery, industrial_storage, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        pack_config = BatteryPassportConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = BatteryPassportConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with BATTERY_PACK_ are loaded and
        mapped to configuration keys. Nested keys use double underscore.

        Example: BATTERY_PACK_BATTERY_CATEGORY=EV
        Example: BATTERY_PACK_CARBON_FOOTPRINT__METHODOLOGY=PEF
        """
        overrides: Dict[str, Any] = {}
        prefix = "BATTERY_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance."""
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: BatteryPassportConfig) -> List[str]:
    """Validate a battery passport configuration and return any warnings.

    Args:
        config: BatteryPassportConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []
    category = config.battery_category.value

    # Check passport requirement alignment
    if PASSPORT_REQUIRED.get(category, False) and not config.reporting.enabled:
        warnings.append(
            f"Battery passport is mandatory for {category} batteries (Art. 65) "
            f"but reporting is disabled."
        )

    # Check carbon footprint requirement alignment
    if CARBON_FOOTPRINT_REQUIRED.get(category, False):
        if not config.carbon_footprint.enabled:
            warnings.append(
                f"Carbon footprint declaration is mandatory for {category} "
                f"batteries (Art. 7) but carbon footprint engine is disabled."
            )

    # Check supply chain DD for applicable categories
    dd_required_categories = {"EV", "INDUSTRIAL", "LMT"}
    if category in dd_required_categories and not config.supply_chain.enabled:
        warnings.append(
            f"Supply chain due diligence is mandatory for {category} "
            f"batteries (Art. 22-26) but supply chain engine is disabled."
        )

    # Check labelling requirements
    required_elements = REQUIRED_LABEL_ELEMENTS.get(category, [])
    if config.labelling.enabled:
        configured_elements = config.labelling.required_elements_by_category.get(
            category, []
        )
        missing_elements = [e for e in required_elements if e not in configured_elements]
        if missing_elements:
            warnings.append(
                f"Missing required label elements for {category}: {missing_elements}. "
                f"Art. 13 requires all specified elements."
            )

    # Check recycled content for applicable categories
    rc_categories = {"EV", "INDUSTRIAL", "LMT", "SLI"}
    if category in rc_categories and not config.recycled_content.enabled:
        warnings.append(
            f"Recycled content documentation is required for {category} "
            f"batteries (Art. 17) but recycled content engine is disabled."
        )

    # Check conformity assessment module selection
    if config.conformity.enabled:
        available_modules = CONFORMITY_MODULES_BY_CATEGORY.get(category, [])
        configured_modules = config.conformity.module_by_category.get(category, [])
        invalid_modules = [m for m in configured_modules if m not in available_modules]
        if invalid_modules:
            warnings.append(
                f"Conformity modules {invalid_modules} are not applicable for "
                f"{category} batteries. Available: {available_modules}."
            )

    # Check provenance tracking
    if not config.reporting.sha256_provenance:
        warnings.append(
            "SHA-256 provenance tracking is disabled. Consider enabling "
            "for audit trail integrity."
        )

    # Check SoH data access for passport batteries
    passport_categories = {"EV", "INDUSTRIAL", "LMT"}
    if category in passport_categories and not config.performance.soh_data_access:
        warnings.append(
            f"SoH data access (Art. 14) is required for {category} batteries "
            f"but soh_data_access is disabled."
        )

    # Check end-of-life collection target coverage
    if config.eol.enabled:
        cat_key = category.lower()
        if cat_key not in config.eol.collection_targets:
            warnings.append(
                f"No collection targets configured for {category} category "
                f"in eol.collection_targets."
            )

    # Check chemistry-specific material coverage
    chemistry = config.chemistry.value
    required_materials = CHEMISTRY_CRITICAL_MATERIALS.get(chemistry, [])
    if config.supply_chain.enabled and required_materials:
        configured = [m.value for m in config.supply_chain.critical_materials]
        missing = [m for m in required_materials if m not in configured]
        if missing:
            warnings.append(
                f"Chemistry {chemistry} uses materials {missing} which are "
                f"not in supply_chain.critical_materials for due diligence."
            )

    return warnings


def get_default_config(
    category: str = "EV",
    chemistry: str = "NMC811",
) -> BatteryPassportConfig:
    """Get default battery passport configuration for a given category/chemistry.

    Args:
        category: Battery category identifier.
        chemistry: Battery chemistry identifier.

    Returns:
        BatteryPassportConfig instance with appropriate defaults.
    """
    return BatteryPassportConfig(
        battery_category=BatteryCategory(category),
        chemistry=BatteryChemistry(chemistry),
    )


def get_article_info(article_ref: str) -> Dict[str, Any]:
    """Get detailed information about a Battery Regulation article.

    Args:
        article_ref: Article reference (e.g., "Art. 7", "Art. 65").

    Returns:
        Dictionary with article name, scope, timeline, and description.
    """
    return BATTERY_REGULATION_ARTICLES.get(article_ref, {
        "name": article_ref,
        "scope": [],
        "mandatory": False,
        "timeline": "Unknown",
        "description": "Unknown article reference",
    })


def get_recycled_content_target(
    material: str,
    phase: int = 1,
) -> float:
    """Get recycled content target for a material and phase.

    Args:
        material: Material name (cobalt, lithium, nickel, lead).
        phase: Target phase (1 = 2031, 2 = 2036).

    Returns:
        Target percentage as float.
    """
    targets = RECYCLED_CONTENT_TARGETS_2031 if phase == 1 else RECYCLED_CONTENT_TARGETS_2036
    return targets.get(material.lower(), 0.0)


def get_collection_target(
    category: str,
    year: str,
) -> float:
    """Get collection rate target for a category and year.

    Args:
        category: Battery category (portable, lmt, ev, industrial, sli).
        year: Target year as string.

    Returns:
        Collection target percentage as float.
    """
    return COLLECTION_TARGETS.get(category.lower(), {}).get(year, 0.0)


def get_material_recovery_target(
    material: str,
    year: str,
) -> float:
    """Get material recovery target for a material and year.

    Args:
        material: Material name (cobalt, copper, lead, lithium, nickel).
        year: Target year as string.

    Returns:
        Recovery target percentage as float.
    """
    return MATERIAL_RECOVERY_TARGETS.get(material.lower(), {}).get(year, 0.0)


def get_critical_materials_for_chemistry(chemistry: str) -> List[str]:
    """Get critical raw materials for a battery chemistry.

    Args:
        chemistry: Chemistry identifier (NMC, NCA, LFP, etc.).

    Returns:
        List of critical material names.
    """
    return CHEMISTRY_CRITICAL_MATERIALS.get(chemistry.upper(), [])


def list_available_presets() -> Dict[str, str]:
    """List all available battery passport configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
