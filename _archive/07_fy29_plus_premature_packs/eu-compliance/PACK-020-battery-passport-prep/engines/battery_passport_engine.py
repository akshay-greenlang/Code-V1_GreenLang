# -*- coding: utf-8 -*-
"""
BatteryPassportEngine - PACK-020 Battery Passport Prep Engine 3
================================================================

Compiles and validates all battery passport data per EU Battery
Regulation Art 77-78 and Annex XIII.

Under Regulation (EU) 2023/1542 (the EU Battery Regulation), Articles
77 and 78 mandate the creation of a digital battery passport for EV
batteries, industrial batteries with a capacity above 2 kWh, and LMT
batteries placed on the EU market from 18 February 2027.  Annex XIII
defines the information requirements that must be included in the
battery passport, accessible via a QR code on the battery.

Regulation (EU) 2023/1542 Framework:
    - Art 77(1): From 18 February 2027, each industrial battery with a
      capacity above 2 kWh, each EV battery, and each LMT battery
      placed on the market or put into service shall have an electronic
      record (battery passport).
    - Art 77(2): The battery passport shall be linked to a unique
      identifier of the battery and shall be accessible online via a
      QR code affixed to the battery.
    - Art 77(3): The battery passport shall contain the information
      set out in Annex XIII.
    - Art 78: Access rights to the battery passport information,
      distinguishing between public, notified body, and Commission
      access levels.
    - Annex XIII: Information requirements for the battery passport
      including general info, carbon footprint, supply chain due
      diligence, material composition, performance and durability,
      and end-of-life information.

Annex XIII Information Sections:
    A. General battery and manufacturer information
    B. Carbon footprint information (links to Art 7)
    C. Supply chain due diligence (links to Art 48-52)
    D. Material composition and hazardous substances
    E. Performance and durability information (links to Art 10)
    F. End-of-life information (collection, recycling, second life)

Regulatory References:
    - Regulation (EU) 2023/1542 of the European Parliament and of the
      Council of 12 July 2023 concerning batteries and waste batteries
    - Art 77 - Battery passport
    - Art 78 - Access to battery passport information
    - Annex XIII - Battery passport information requirements
    - Commission Implementing Regulation (EU) 2024/XX (QR code specs)

Zero-Hallucination:
    - Completeness calculation uses deterministic field counting
    - Data quality assessment uses deterministic rule-based scoring
    - QR payload generation uses deterministic JSON serialisation
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PassportField(str, Enum):
    """Battery passport data field per Annex XIII.

    Enumerates all data fields that must or may be included in the
    battery passport.  Fields are organised by the Annex XIII sections
    (A through F) covering general information, carbon footprint,
    supply chain due diligence, material composition, performance
    and durability, and end-of-life information.
    """
    # Section A: General battery and manufacturer information
    MANUFACTURER_ID = "manufacturer_id"
    MANUFACTURING_PLANT = "manufacturing_plant"
    MANUFACTURING_DATE = "manufacturing_date"
    MANUFACTURING_COUNTRY = "manufacturing_country"
    BATTERY_MODEL = "battery_model"
    BATTERY_BATCH = "battery_batch"
    BATTERY_SERIAL = "battery_serial"
    BATTERY_WEIGHT = "battery_weight"
    BATTERY_CATEGORY = "battery_category"
    BATTERY_CHEMISTRY = "battery_chemistry"
    ENERGY_CAPACITY = "energy_capacity"
    VOLTAGE_NOMINAL = "voltage_nominal"

    # Section B: Carbon footprint information
    CARBON_FOOTPRINT_TOTAL = "carbon_footprint_total"
    CARBON_FOOTPRINT_PER_KWH = "carbon_footprint_per_kwh"
    CARBON_FOOTPRINT_CLASS = "carbon_footprint_class"
    CARBON_FOOTPRINT_LIFECYCLE = "carbon_footprint_lifecycle"
    CARBON_FOOTPRINT_METHODOLOGY = "carbon_footprint_methodology"

    # Section C: Supply chain due diligence
    DD_POLICY = "dd_policy"
    DD_THIRD_PARTY_VERIFICATION = "dd_third_party_verification"
    DD_CONFLICT_MINERALS = "dd_conflict_minerals"
    DD_SUPPLY_CHAIN_MAPPING = "dd_supply_chain_mapping"

    # Section D: Material composition
    MATERIAL_COMPOSITION = "material_composition"
    HAZARDOUS_SUBSTANCES = "hazardous_substances"
    CRITICAL_RAW_MATERIALS = "critical_raw_materials"
    RECYCLED_CONTENT = "recycled_content"

    # Section E: Performance and durability
    RATED_CAPACITY = "rated_capacity"
    CYCLE_LIFE_EXPECTED = "cycle_life_expected"
    ENERGY_EFFICIENCY = "energy_efficiency"
    INTERNAL_RESISTANCE = "internal_resistance"
    STATE_OF_HEALTH = "state_of_health"
    TEMPERATURE_RANGE = "temperature_range"
    C_RATE_MAX = "c_rate_max"

    # Section F: End-of-life information
    EOL_COLLECTION_INFO = "eol_collection_info"
    EOL_RECYCLING_INFO = "eol_recycling_info"
    EOL_SECOND_LIFE_INFO = "eol_second_life_info"
    EOL_SAFETY_INSTRUCTIONS = "eol_safety_instructions"

class PassportStatus(str, Enum):
    """Status of the battery passport.

    Tracks the lifecycle of the passport from initial draft through
    validation, publication, and potential revocation.
    """
    DRAFT = "draft"
    VALIDATED = "validated"
    PUBLISHED = "published"
    REVOKED = "revoked"

class DataQuality(str, Enum):
    """Data quality level for a passport field.

    Indicates the completeness and validity of data for each field
    in the battery passport.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    INVALID = "invalid"

class AccessLevel(str, Enum):
    """Access level for battery passport information per Art 78.

    Distinguishes between information that is publicly accessible,
    restricted to notified bodies, or accessible only to the
    Commission and market surveillance authorities.
    """
    PUBLIC = "public"
    NOTIFIED_BODY = "notified_body"
    COMMISSION = "commission"
    MARKET_SURVEILLANCE = "market_surveillance"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required fields by Annex XIII section.
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "A_general": [
        PassportField.MANUFACTURER_ID.value,
        PassportField.MANUFACTURING_PLANT.value,
        PassportField.MANUFACTURING_DATE.value,
        PassportField.BATTERY_MODEL.value,
        PassportField.BATTERY_SERIAL.value,
        PassportField.BATTERY_WEIGHT.value,
        PassportField.BATTERY_CATEGORY.value,
        PassportField.BATTERY_CHEMISTRY.value,
        PassportField.ENERGY_CAPACITY.value,
        PassportField.VOLTAGE_NOMINAL.value,
    ],
    "B_carbon_footprint": [
        PassportField.CARBON_FOOTPRINT_TOTAL.value,
        PassportField.CARBON_FOOTPRINT_PER_KWH.value,
        PassportField.CARBON_FOOTPRINT_CLASS.value,
    ],
    "C_due_diligence": [
        PassportField.DD_POLICY.value,
        PassportField.DD_THIRD_PARTY_VERIFICATION.value,
    ],
    "D_material_composition": [
        PassportField.MATERIAL_COMPOSITION.value,
        PassportField.HAZARDOUS_SUBSTANCES.value,
        PassportField.CRITICAL_RAW_MATERIALS.value,
        PassportField.RECYCLED_CONTENT.value,
    ],
    "E_performance": [
        PassportField.RATED_CAPACITY.value,
        PassportField.CYCLE_LIFE_EXPECTED.value,
        PassportField.ENERGY_EFFICIENCY.value,
    ],
    "F_end_of_life": [
        PassportField.EOL_COLLECTION_INFO.value,
        PassportField.EOL_RECYCLING_INFO.value,
        PassportField.EOL_SAFETY_INSTRUCTIONS.value,
    ],
}

# Optional but recommended fields.
OPTIONAL_FIELDS: Dict[str, List[str]] = {
    "A_general": [
        PassportField.MANUFACTURING_COUNTRY.value,
        PassportField.BATTERY_BATCH.value,
    ],
    "B_carbon_footprint": [
        PassportField.CARBON_FOOTPRINT_LIFECYCLE.value,
        PassportField.CARBON_FOOTPRINT_METHODOLOGY.value,
    ],
    "C_due_diligence": [
        PassportField.DD_CONFLICT_MINERALS.value,
        PassportField.DD_SUPPLY_CHAIN_MAPPING.value,
    ],
    "D_material_composition": [],
    "E_performance": [
        PassportField.INTERNAL_RESISTANCE.value,
        PassportField.STATE_OF_HEALTH.value,
        PassportField.TEMPERATURE_RANGE.value,
        PassportField.C_RATE_MAX.value,
    ],
    "F_end_of_life": [
        PassportField.EOL_SECOND_LIFE_INFO.value,
    ],
}

# All required fields as a flat set for quick lookup.
ALL_REQUIRED_FIELDS: set = set()
for _section_fields in REQUIRED_FIELDS.values():
    ALL_REQUIRED_FIELDS.update(_section_fields)

# All optional fields as a flat set.
ALL_OPTIONAL_FIELDS: set = set()
for _section_fields in OPTIONAL_FIELDS.values():
    ALL_OPTIONAL_FIELDS.update(_section_fields)

# Section labels.
SECTION_LABELS: Dict[str, str] = {
    "A_general": "A. General battery and manufacturer information",
    "B_carbon_footprint": "B. Carbon footprint information",
    "C_due_diligence": "C. Supply chain due diligence",
    "D_material_composition": "D. Material composition and hazardous substances",
    "E_performance": "E. Performance and durability",
    "F_end_of_life": "F. End-of-life information",
}

# Field access levels per Art 78.
FIELD_ACCESS_LEVELS: Dict[str, str] = {
    PassportField.MANUFACTURER_ID.value: AccessLevel.PUBLIC.value,
    PassportField.MANUFACTURING_PLANT.value: AccessLevel.NOTIFIED_BODY.value,
    PassportField.MANUFACTURING_DATE.value: AccessLevel.PUBLIC.value,
    PassportField.MANUFACTURING_COUNTRY.value: AccessLevel.PUBLIC.value,
    PassportField.BATTERY_MODEL.value: AccessLevel.PUBLIC.value,
    PassportField.BATTERY_BATCH.value: AccessLevel.NOTIFIED_BODY.value,
    PassportField.BATTERY_SERIAL.value: AccessLevel.PUBLIC.value,
    PassportField.BATTERY_WEIGHT.value: AccessLevel.PUBLIC.value,
    PassportField.BATTERY_CATEGORY.value: AccessLevel.PUBLIC.value,
    PassportField.BATTERY_CHEMISTRY.value: AccessLevel.PUBLIC.value,
    PassportField.ENERGY_CAPACITY.value: AccessLevel.PUBLIC.value,
    PassportField.VOLTAGE_NOMINAL.value: AccessLevel.PUBLIC.value,
    PassportField.CARBON_FOOTPRINT_TOTAL.value: AccessLevel.PUBLIC.value,
    PassportField.CARBON_FOOTPRINT_PER_KWH.value: AccessLevel.PUBLIC.value,
    PassportField.CARBON_FOOTPRINT_CLASS.value: AccessLevel.PUBLIC.value,
    PassportField.CARBON_FOOTPRINT_LIFECYCLE.value: AccessLevel.PUBLIC.value,
    PassportField.CARBON_FOOTPRINT_METHODOLOGY.value: AccessLevel.PUBLIC.value,
    PassportField.DD_POLICY.value: AccessLevel.PUBLIC.value,
    PassportField.DD_THIRD_PARTY_VERIFICATION.value: AccessLevel.PUBLIC.value,
    PassportField.DD_CONFLICT_MINERALS.value: AccessLevel.NOTIFIED_BODY.value,
    PassportField.DD_SUPPLY_CHAIN_MAPPING.value: AccessLevel.COMMISSION.value,
    PassportField.MATERIAL_COMPOSITION.value: AccessLevel.PUBLIC.value,
    PassportField.HAZARDOUS_SUBSTANCES.value: AccessLevel.PUBLIC.value,
    PassportField.CRITICAL_RAW_MATERIALS.value: AccessLevel.PUBLIC.value,
    PassportField.RECYCLED_CONTENT.value: AccessLevel.PUBLIC.value,
    PassportField.RATED_CAPACITY.value: AccessLevel.PUBLIC.value,
    PassportField.CYCLE_LIFE_EXPECTED.value: AccessLevel.PUBLIC.value,
    PassportField.ENERGY_EFFICIENCY.value: AccessLevel.PUBLIC.value,
    PassportField.INTERNAL_RESISTANCE.value: AccessLevel.PUBLIC.value,
    PassportField.STATE_OF_HEALTH.value: AccessLevel.PUBLIC.value,
    PassportField.TEMPERATURE_RANGE.value: AccessLevel.PUBLIC.value,
    PassportField.C_RATE_MAX.value: AccessLevel.PUBLIC.value,
    PassportField.EOL_COLLECTION_INFO.value: AccessLevel.PUBLIC.value,
    PassportField.EOL_RECYCLING_INFO.value: AccessLevel.PUBLIC.value,
    PassportField.EOL_SECOND_LIFE_INFO.value: AccessLevel.PUBLIC.value,
    PassportField.EOL_SAFETY_INSTRUCTIONS.value: AccessLevel.PUBLIC.value,
}

# Field descriptions for documentation.
FIELD_DESCRIPTIONS: Dict[str, str] = {
    PassportField.MANUFACTURER_ID.value: "Unique identifier of the battery manufacturer",
    PassportField.MANUFACTURING_PLANT.value: "Name and address of the manufacturing facility",
    PassportField.MANUFACTURING_DATE.value: "Date of battery manufacture",
    PassportField.MANUFACTURING_COUNTRY.value: "Country of manufacture",
    PassportField.BATTERY_MODEL.value: "Battery model or type designation",
    PassportField.BATTERY_BATCH.value: "Batch or lot number",
    PassportField.BATTERY_SERIAL.value: "Unique serial number of the battery",
    PassportField.BATTERY_WEIGHT.value: "Total weight of the battery in kilograms",
    PassportField.BATTERY_CATEGORY.value: "Battery category (EV, industrial, LMT, etc.)",
    PassportField.BATTERY_CHEMISTRY.value: "Electrochemical system / chemistry type",
    PassportField.ENERGY_CAPACITY.value: "Rated energy capacity in kWh",
    PassportField.VOLTAGE_NOMINAL.value: "Nominal voltage in volts",
    PassportField.CARBON_FOOTPRINT_TOTAL.value: "Total lifecycle carbon footprint (kgCO2e)",
    PassportField.CARBON_FOOTPRINT_PER_KWH.value: "Carbon footprint per kWh (kgCO2e/kWh)",
    PassportField.CARBON_FOOTPRINT_CLASS.value: "Carbon footprint performance class (A-E)",
    PassportField.CARBON_FOOTPRINT_LIFECYCLE.value: "Per-stage lifecycle breakdown",
    PassportField.CARBON_FOOTPRINT_METHODOLOGY.value: "Methodology references for footprint calculation",
    PassportField.DD_POLICY.value: "Supply chain due diligence policy documentation",
    PassportField.DD_THIRD_PARTY_VERIFICATION.value: "Third-party verification of due diligence",
    PassportField.DD_CONFLICT_MINERALS.value: "Conflict minerals assessment",
    PassportField.DD_SUPPLY_CHAIN_MAPPING.value: "Supply chain mapping and risk assessment",
    PassportField.MATERIAL_COMPOSITION.value: "Bill of materials with material composition",
    PassportField.HAZARDOUS_SUBSTANCES.value: "Hazardous substances declaration",
    PassportField.CRITICAL_RAW_MATERIALS.value: "Critical raw materials content",
    PassportField.RECYCLED_CONTENT.value: "Recycled content percentages per material",
    PassportField.RATED_CAPACITY.value: "Rated capacity in Ah",
    PassportField.CYCLE_LIFE_EXPECTED.value: "Expected number of charge/discharge cycles",
    PassportField.ENERGY_EFFICIENCY.value: "Round-trip energy efficiency (%)",
    PassportField.INTERNAL_RESISTANCE.value: "Internal resistance (mOhm)",
    PassportField.STATE_OF_HEALTH.value: "State of Health (%)",
    PassportField.TEMPERATURE_RANGE.value: "Operating temperature range (degC)",
    PassportField.C_RATE_MAX.value: "Maximum charge/discharge C-rate",
    PassportField.EOL_COLLECTION_INFO.value: "Collection and take-back information",
    PassportField.EOL_RECYCLING_INFO.value: "Recycling instructions and requirements",
    PassportField.EOL_SECOND_LIFE_INFO.value: "Second-life application suitability",
    PassportField.EOL_SAFETY_INSTRUCTIONS.value: "End-of-life safety and handling instructions",
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GeneralInfo(BaseModel):
    """Section A: General battery and manufacturer information.

    Contains core identification and specification data for the
    battery as required by Annex XIII, Section A.
    """
    manufacturer_id: str = Field(
        default="",
        description="Unique manufacturer identifier",
        max_length=200,
    )
    manufacturing_plant: str = Field(
        default="",
        description="Manufacturing plant name and location",
        max_length=500,
    )
    manufacturing_date: str = Field(
        default="",
        description="Date of manufacture (ISO 8601 format)",
        max_length=50,
    )
    manufacturing_country: str = Field(
        default="",
        description="Country of manufacture (ISO 3166-1 alpha-2)",
        max_length=10,
    )
    battery_model: str = Field(
        default="",
        description="Battery model or type designation",
        max_length=200,
    )
    battery_batch: str = Field(
        default="",
        description="Batch or lot number",
        max_length=200,
    )
    battery_serial: str = Field(
        default="",
        description="Unique battery serial number",
        max_length=200,
    )
    battery_weight_kg: Optional[Decimal] = Field(
        default=None,
        description="Total battery weight in kilograms",
        ge=0,
    )
    battery_category: str = Field(
        default="",
        description="Battery category (ev, industrial, lmt, portable, sli)",
        max_length=50,
    )
    battery_chemistry: str = Field(
        default="",
        description="Electrochemical system / chemistry type",
        max_length=100,
    )
    energy_capacity_kwh: Optional[Decimal] = Field(
        default=None,
        description="Rated energy capacity in kWh",
        ge=0,
    )
    voltage_nominal: Optional[Decimal] = Field(
        default=None,
        description="Nominal voltage in volts",
        ge=0,
    )

class CarbonFootprintInfo(BaseModel):
    """Section B: Carbon footprint information.

    Contains carbon footprint data as required by Annex XIII, Section B,
    linking to the Art 7 carbon footprint declaration.
    """
    total_co2e_kg: Optional[Decimal] = Field(
        default=None,
        description="Total lifecycle CO2e emissions (kg)",
    )
    per_kwh_co2e_kg: Optional[Decimal] = Field(
        default=None,
        description="Carbon footprint intensity (kgCO2e/kWh)",
    )
    performance_class: str = Field(
        default="",
        description="Performance class (class_a through class_e)",
        max_length=20,
    )
    lifecycle_breakdown: Optional[Dict[str, str]] = Field(
        default=None,
        description="Per-stage lifecycle breakdown",
    )
    methodology: Optional[Dict[str, str]] = Field(
        default=None,
        description="Methodology references",
    )

class SupplyChainDD(BaseModel):
    """Section C: Supply chain due diligence.

    Contains due diligence data as required by Annex XIII, Section C,
    linking to the Art 48-52 due diligence obligations.
    """
    dd_policy: str = Field(
        default="",
        description="Summary of supply chain due diligence policy",
        max_length=5000,
    )
    third_party_verification: str = Field(
        default="",
        description="Third-party verification status and body",
        max_length=2000,
    )
    conflict_minerals: str = Field(
        default="",
        description="Conflict minerals assessment summary",
        max_length=5000,
    )
    supply_chain_mapping: str = Field(
        default="",
        description="Supply chain mapping summary",
        max_length=5000,
    )

class MaterialComposition(BaseModel):
    """Section D: Material composition and hazardous substances.

    Contains material composition data as required by Annex XIII,
    Section D.
    """
    bill_of_materials: Optional[Dict[str, str]] = Field(
        default=None,
        description="Bill of materials (material name to weight/percentage)",
    )
    hazardous_substances: Optional[List[str]] = Field(
        default=None,
        description="List of hazardous substances present",
    )
    critical_raw_materials: Optional[Dict[str, str]] = Field(
        default=None,
        description="Critical raw materials content (material to percentage)",
    )
    recycled_content: Optional[Dict[str, str]] = Field(
        default=None,
        description="Recycled content per material (material to percentage)",
    )

class PerformanceDurability(BaseModel):
    """Section E: Performance and durability information.

    Contains performance and durability data as required by Annex XIII,
    Section E, linking to the Art 10 requirements.
    """
    rated_capacity_ah: Optional[Decimal] = Field(
        default=None,
        description="Rated capacity in Ah",
        ge=0,
    )
    cycle_life_expected: Optional[int] = Field(
        default=None,
        description="Expected number of charge/discharge cycles",
        ge=0,
    )
    energy_efficiency_pct: Optional[Decimal] = Field(
        default=None,
        description="Round-trip energy efficiency (%)",
        ge=0,
        le=Decimal("100"),
    )
    internal_resistance_mohm: Optional[Decimal] = Field(
        default=None,
        description="Internal resistance in milliohms",
        ge=0,
    )
    state_of_health_pct: Optional[Decimal] = Field(
        default=None,
        description="State of Health (%)",
        ge=0,
        le=Decimal("100"),
    )
    temperature_range_min: Optional[Decimal] = Field(
        default=None,
        description="Minimum operating temperature (degC)",
    )
    temperature_range_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum operating temperature (degC)",
    )
    c_rate_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum charge/discharge C-rate",
        ge=0,
    )

class EndOfLifeInfo(BaseModel):
    """Section F: End-of-life information.

    Contains end-of-life data as required by Annex XIII, Section F.
    """
    collection_info: str = Field(
        default="",
        description="Collection and take-back scheme information",
        max_length=5000,
    )
    recycling_info: str = Field(
        default="",
        description="Recycling instructions and requirements",
        max_length=5000,
    )
    second_life_info: str = Field(
        default="",
        description="Second-life application suitability assessment",
        max_length=5000,
    )
    safety_instructions: str = Field(
        default="",
        description="End-of-life safety and handling instructions",
        max_length=5000,
    )

class PassportData(BaseModel):
    """Complete passport data covering all Annex XIII sections.

    Aggregates all six sections of the battery passport information
    requirements into a single structured model.
    """
    passport_id: str = Field(
        default_factory=_new_uuid,
        description="Unique passport identifier",
    )
    general_info: GeneralInfo = Field(
        default_factory=GeneralInfo,
        description="Section A: General battery information",
    )
    carbon_footprint: CarbonFootprintInfo = Field(
        default_factory=CarbonFootprintInfo,
        description="Section B: Carbon footprint information",
    )
    supply_chain_dd: SupplyChainDD = Field(
        default_factory=SupplyChainDD,
        description="Section C: Supply chain due diligence",
    )
    material_composition: MaterialComposition = Field(
        default_factory=MaterialComposition,
        description="Section D: Material composition",
    )
    performance_durability: PerformanceDurability = Field(
        default_factory=PerformanceDurability,
        description="Section E: Performance and durability",
    )
    end_of_life: EndOfLifeInfo = Field(
        default_factory=EndOfLifeInfo,
        description="Section F: End-of-life information",
    )

class FieldValidation(BaseModel):
    """Validation result for a single passport field."""
    field_name: str = Field(
        ...,
        description="Passport field identifier",
    )
    field_description: str = Field(
        default="",
        description="Human-readable field description",
    )
    data_quality: DataQuality = Field(
        default=DataQuality.MISSING,
        description="Data quality assessment for this field",
    )
    is_required: bool = Field(
        default=False,
        description="Whether this field is required by Annex XIII",
    )
    access_level: str = Field(
        default=AccessLevel.PUBLIC.value,
        description="Access level for this field per Art 78",
    )
    section: str = Field(
        default="",
        description="Annex XIII section this field belongs to",
    )
    value_present: bool = Field(
        default=False,
        description="Whether the field has a non-empty value",
    )
    note: str = Field(
        default="",
        description="Additional validation note",
        max_length=500,
    )

class PassportValidationResult(BaseModel):
    """Result of battery passport compilation and validation.

    Contains the complete validation assessment of the passport data
    against Annex XIII requirements, with completeness scoring,
    per-field quality assessment, and QR code payload generation.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this validation",
    )
    validated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of validation (UTC)",
    )
    passport_id: str = Field(
        default="",
        description="Passport identifier",
    )
    status: PassportStatus = Field(
        default=PassportStatus.DRAFT,
        description="Passport status",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Overall completeness percentage",
    )
    required_fields_total: int = Field(
        default=0,
        description="Total number of required fields",
    )
    required_fields_met: int = Field(
        default=0,
        description="Number of required fields with data",
    )
    required_fields_missing: int = Field(
        default=0,
        description="Number of required fields without data",
    )
    optional_fields_total: int = Field(
        default=0,
        description="Total number of optional fields",
    )
    optional_fields_met: int = Field(
        default=0,
        description="Number of optional fields with data",
    )
    section_completeness: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-section completeness assessment",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors",
    )
    field_validations: List[FieldValidation] = Field(
        default_factory=list,
        description="Per-field validation results",
    )
    data_quality_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of fields by data quality level",
    )
    qr_payload: Optional[str] = Field(
        default=None,
        description="JSON payload for QR code generation",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving passport completeness",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BatteryPassportEngine:
    """Battery passport engine per EU Battery Regulation Art 77-78.

    Provides deterministic, zero-hallucination compilation and
    validation of:
    - Passport data assembly from all six Annex XIII sections
    - Field-by-field completeness and quality assessment
    - Required vs optional field validation
    - Section-level completeness scoring
    - QR code payload generation for battery marking
    - Status management (draft -> validated -> published)
    - Access level enforcement per Art 78

    All calculations use Decimal arithmetic and are bit-perfect
    reproducible.  No LLM is used in any calculation path.

    Usage::

        engine = BatteryPassportEngine()
        data = PassportData(
            general_info=GeneralInfo(
                manufacturer_id="MFG-001",
                battery_model="EV-75-NMC",
                battery_serial="SN-2025-000001",
                battery_weight_kg=Decimal("450"),
                battery_category="ev",
                battery_chemistry="nmc811",
                energy_capacity_kwh=Decimal("75"),
                voltage_nominal=Decimal("400"),
            ),
        )
        passport = engine.compile_passport(data)
        result = engine.validate_passport(passport)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise BatteryPassportEngine."""
        self._passports: Dict[str, PassportData] = {}
        self._validation_results: List[PassportValidationResult] = []
        logger.info(
            "BatteryPassportEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Passport Compilation                                                 #
    # ------------------------------------------------------------------ #

    def compile_passport(
        self, data: PassportData
    ) -> PassportData:
        """Compile and assemble a battery passport from section data.

        Validates that the passport has a unique identifier and stores
        it in the internal registry for subsequent validation.

        Args:
            data: PassportData with all sections populated.

        Returns:
            Compiled PassportData with assigned passport_id.
        """
        t0 = time.perf_counter()

        if not data.passport_id:
            data.passport_id = _new_uuid()

        self._passports[data.passport_id] = data

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Compiled passport %s in %.3f ms",
            data.passport_id, elapsed_ms,
        )
        return data

    # ------------------------------------------------------------------ #
    # Passport Validation                                                  #
    # ------------------------------------------------------------------ #

    def validate_passport(
        self, passport: PassportData
    ) -> PassportValidationResult:
        """Validate a battery passport against Annex XIII requirements.

        Performs field-by-field validation of all required and optional
        fields, calculates section and overall completeness, assesses
        data quality, and generates QR code payload.

        Args:
            passport: PassportData to validate.

        Returns:
            PassportValidationResult with complete assessment.
        """
        t0 = time.perf_counter()

        # Step 1: Extract all field values
        field_values = self._extract_field_values(passport)

        # Step 2: Validate each field
        field_validations: List[FieldValidation] = []
        required_met = 0
        required_total = len(ALL_REQUIRED_FIELDS)
        optional_met = 0
        optional_total = len(ALL_OPTIONAL_FIELDS)
        validation_errors: List[str] = []
        dq_summary: Dict[str, int] = {}

        for field in PassportField:
            value = field_values.get(field.value)
            is_required = field.value in ALL_REQUIRED_FIELDS
            is_optional = field.value in ALL_OPTIONAL_FIELDS

            # Determine section
            section = self._get_field_section(field.value)

            # Assess data quality
            quality = self._assess_field_quality(value)

            # Check presence
            value_present = quality in (DataQuality.COMPLETE, DataQuality.PARTIAL)

            if is_required and not value_present:
                validation_errors.append(
                    f"Required field '{field.value}' "
                    f"({FIELD_DESCRIPTIONS.get(field.value, '')}) is missing"
                )
            if is_required and value_present:
                required_met += 1
            if is_optional and value_present:
                optional_met += 1

            # Data quality summary
            dq_key = quality.value
            dq_summary[dq_key] = dq_summary.get(dq_key, 0) + 1

            field_validations.append(FieldValidation(
                field_name=field.value,
                field_description=FIELD_DESCRIPTIONS.get(field.value, ""),
                data_quality=quality,
                is_required=is_required,
                access_level=FIELD_ACCESS_LEVELS.get(
                    field.value, AccessLevel.PUBLIC.value
                ),
                section=section,
                value_present=value_present,
            ))

        # Step 3: Section completeness
        section_completeness = self._calculate_section_completeness(
            field_validations
        )

        # Step 4: Overall completeness
        completeness_pct = Decimal("0.00")
        if required_total > 0:
            completeness_pct = _round_val(
                _decimal(required_met) / _decimal(required_total) * Decimal("100"),
                2,
            )

        # Step 5: Determine status
        status = self._determine_status(completeness_pct, validation_errors)

        # Step 6: Generate QR payload
        qr_payload = self.generate_qr_payload(passport)

        # Step 7: Recommendations
        recommendations = self._generate_recommendations(
            field_validations, section_completeness, validation_errors
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PassportValidationResult(
            passport_id=passport.passport_id,
            status=status,
            completeness_pct=completeness_pct,
            required_fields_total=required_total,
            required_fields_met=required_met,
            required_fields_missing=required_total - required_met,
            optional_fields_total=optional_total,
            optional_fields_met=optional_met,
            section_completeness=section_completeness,
            validation_errors=validation_errors,
            field_validations=field_validations,
            data_quality_summary=dq_summary,
            qr_payload=qr_payload,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        self._validation_results.append(result)

        logger.info(
            "Validated passport %s: completeness=%s%%, status=%s, "
            "required=%d/%d, errors=%d in %.3f ms",
            passport.passport_id,
            completeness_pct,
            status.value,
            required_met,
            required_total,
            len(validation_errors),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # QR Code Payload                                                      #
    # ------------------------------------------------------------------ #

    def generate_qr_payload(
        self, passport: PassportData
    ) -> str:
        """Generate a QR code payload for the battery passport.

        Creates a JSON payload containing the public-access fields
        from the battery passport, suitable for encoding into a QR

from greenlang.schemas import utcnow
        code affixed to the battery per Art 77(2).

        Args:
            passport: PassportData to generate QR payload from.

        Returns:
            JSON string for QR code encoding.
        """
        t0 = time.perf_counter()

        gi = passport.general_info
        cf = passport.carbon_footprint

        payload: Dict[str, Any] = {
            "passport_id": passport.passport_id,
            "schema_version": "1.0",
            "regulation": "EU_2023_1542",
            "manufacturer_id": gi.manufacturer_id,
            "battery_serial": gi.battery_serial,
            "battery_model": gi.battery_model,
            "battery_category": gi.battery_category,
            "battery_chemistry": gi.battery_chemistry,
            "energy_capacity_kwh": (
                str(gi.energy_capacity_kwh)
                if gi.energy_capacity_kwh is not None else None
            ),
            "weight_kg": (
                str(gi.battery_weight_kg)
                if gi.battery_weight_kg is not None else None
            ),
            "voltage_nominal": (
                str(gi.voltage_nominal)
                if gi.voltage_nominal is not None else None
            ),
            "carbon_footprint_class": cf.performance_class,
            "carbon_footprint_per_kwh": (
                str(cf.per_kwh_co2e_kg)
                if cf.per_kwh_co2e_kg is not None else None
            ),
            "manufacturing_date": gi.manufacturing_date,
            "passport_url": (
                f"https://battery-passport.eu/p/{passport.passport_id}"
            ),
            "generated_at": str(utcnow()),
        }

        qr_json = json.dumps(payload, sort_keys=True, default=str)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Generated QR payload for passport %s (%d bytes) in %.3f ms",
            passport.passport_id, len(qr_json), elapsed_ms,
        )
        return qr_json

    # ------------------------------------------------------------------ #
    # Data Quality Assessment                                              #
    # ------------------------------------------------------------------ #

    def assess_data_quality(
        self, passport: PassportData
    ) -> Dict[str, Any]:
        """Assess data quality for all passport fields.

        Produces a field-by-field data quality assessment with
        section-level and overall summaries.

        Args:
            passport: PassportData to assess.

        Returns:
            Dict with data quality assessment.
        """
        t0 = time.perf_counter()

        field_values = self._extract_field_values(passport)
        field_results: Dict[str, Dict[str, str]] = {}
        quality_counts: Dict[str, int] = {
            DataQuality.COMPLETE.value: 0,
            DataQuality.PARTIAL.value: 0,
            DataQuality.MISSING.value: 0,
            DataQuality.INVALID.value: 0,
        }

        for field in PassportField:
            value = field_values.get(field.value)
            quality = self._assess_field_quality(value)
            quality_counts[quality.value] = quality_counts.get(
                quality.value, 0
            ) + 1

            field_results[field.value] = {
                "description": FIELD_DESCRIPTIONS.get(field.value, ""),
                "quality": quality.value,
                "required": field.value in ALL_REQUIRED_FIELDS,
                "section": self._get_field_section(field.value),
            }

        total_fields = len(PassportField)
        complete_count = quality_counts.get(DataQuality.COMPLETE.value, 0)
        overall_quality_pct = _round2(
            _safe_divide(float(complete_count), float(total_fields)) * 100.0
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        assessment = {
            "passport_id": passport.passport_id,
            "total_fields": total_fields,
            "quality_distribution": quality_counts,
            "overall_quality_pct": overall_quality_pct,
            "field_results": field_results,
            "processing_time_ms": elapsed_ms,
        }
        assessment["provenance_hash"] = _compute_hash(assessment)

        logger.info(
            "Data quality assessment for %s: %s%% complete in %.3f ms",
            passport.passport_id, overall_quality_pct, elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------ #
    # Batch Validation                                                     #
    # ------------------------------------------------------------------ #

    def validate_batch(
        self, passports: List[PassportData]
    ) -> List[PassportValidationResult]:
        """Validate a batch of battery passports.

        Args:
            passports: List of PassportData objects.

        Returns:
            List of PassportValidationResult objects.
        """
        t0 = time.perf_counter()
        results: List[PassportValidationResult] = []

        for passport in passports:
            result = self.validate_passport(passport)
            results.append(result)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Batch passport validation: %d passports in %.3f ms",
            len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Status Management                                                    #
    # ------------------------------------------------------------------ #

    def update_status(
        self,
        passport_id: str,
        new_status: PassportStatus,
    ) -> Dict[str, Any]:
        """Update the status of a battery passport.

        Args:
            passport_id: Passport identifier.
            new_status: New status to assign.

        Returns:
            Dict with status update confirmation.
        """
        passport = self._passports.get(passport_id)
        if passport is None:
            return {
                "passport_id": passport_id,
                "success": False,
                "error": "Passport not found in registry",
            }

        result = {
            "passport_id": passport_id,
            "success": True,
            "previous_status": "unknown",
            "new_status": new_status.value,
            "updated_at": str(utcnow()),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Updated passport %s status to %s",
            passport_id, new_status.value,
        )
        return result

    # ------------------------------------------------------------------ #
    # Access Level Filtering                                               #
    # ------------------------------------------------------------------ #

    def get_public_fields(
        self, passport: PassportData
    ) -> Dict[str, Any]:
        """Return only public-access fields from the passport.

        Filters passport data to include only fields with public
        access level per Art 78.

        Args:
            passport: PassportData to filter.

        Returns:
            Dict with public-access fields only.
        """
        field_values = self._extract_field_values(passport)
        public_fields: Dict[str, Any] = {}

        for field_name, value in field_values.items():
            access = FIELD_ACCESS_LEVELS.get(
                field_name, AccessLevel.PUBLIC.value
            )
            if access == AccessLevel.PUBLIC.value and value is not None:
                public_fields[field_name] = value

        return {
            "passport_id": passport.passport_id,
            "access_level": AccessLevel.PUBLIC.value,
            "fields": public_fields,
            "provenance_hash": _compute_hash(public_fields),
        }

    # ------------------------------------------------------------------ #
    # Comparison Utilities                                                 #
    # ------------------------------------------------------------------ #

    def compare_passports(
        self, results: List[PassportValidationResult]
    ) -> Dict[str, Any]:
        """Compare validation results across multiple passports.

        Args:
            results: List of PassportValidationResult objects.

        Returns:
            Dict with comparative analysis.
        """
        t0 = time.perf_counter()

        if not results:
            return {
                "count": 0,
                "comparison": [],
                "provenance_hash": _compute_hash({}),
            }

        entries = [
            {
                "passport_id": r.passport_id,
                "status": r.status.value,
                "completeness_pct": str(r.completeness_pct),
                "required_met": r.required_fields_met,
                "required_total": r.required_fields_total,
                "errors": len(r.validation_errors),
            }
            for r in results
        ]

        entries.sort(
            key=lambda x: Decimal(x["completeness_pct"]),
            reverse=True,
        )

        pcts = [r.completeness_pct for r in results]
        avg_pct = _round_val(
            sum(pcts) / _decimal(len(pcts)), 2
        )

        status_dist: Dict[str, int] = {}
        for r in results:
            s = r.status.value
            status_dist[s] = status_dist.get(s, 0) + 1

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        comparison = {
            "count": len(results),
            "ranking": entries,
            "statistics": {
                "min_completeness_pct": str(min(pcts)),
                "max_completeness_pct": str(max(pcts)),
                "avg_completeness_pct": str(avg_pct),
            },
            "status_distribution": status_dist,
            "processing_time_ms": elapsed_ms,
        }
        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Registry Management                                                  #
    # ------------------------------------------------------------------ #

    def get_passport(self, passport_id: str) -> Optional[PassportData]:
        """Retrieve a passport from the internal registry.

        Args:
            passport_id: Passport identifier.

        Returns:
            PassportData or None if not found.
        """
        return self._passports.get(passport_id)

    def get_all_passports(self) -> Dict[str, PassportData]:
        """Return all passports in the registry.

        Returns:
            Dict mapping passport_id to PassportData.
        """
        return dict(self._passports)

    def get_validation_results(self) -> List[PassportValidationResult]:
        """Return all validation results.

        Returns:
            List of PassportValidationResult objects.
        """
        return list(self._validation_results)

    def clear_registry(self) -> None:
        """Clear all stored passports and validation results."""
        self._passports.clear()
        self._validation_results.clear()
        logger.info("BatteryPassportEngine registry cleared")

    # ------------------------------------------------------------------ #
    # Reference Data                                                       #
    # ------------------------------------------------------------------ #

    def get_field_reference(self) -> Dict[str, Dict[str, Any]]:
        """Return reference data for all passport fields.

        Returns:
            Dict mapping field name to description, section, required,
            and access level.
        """
        reference: Dict[str, Dict[str, Any]] = {}
        for field in PassportField:
            reference[field.value] = {
                "description": FIELD_DESCRIPTIONS.get(field.value, ""),
                "section": self._get_field_section(field.value),
                "required": field.value in ALL_REQUIRED_FIELDS,
                "optional": field.value in ALL_OPTIONAL_FIELDS,
                "access_level": FIELD_ACCESS_LEVELS.get(
                    field.value, AccessLevel.PUBLIC.value
                ),
            }
        return reference

    def get_section_reference(self) -> Dict[str, Dict[str, Any]]:
        """Return reference data for all passport sections.

        Returns:
            Dict mapping section key to label, required fields, and
            optional fields.
        """
        reference: Dict[str, Dict[str, Any]] = {}
        for section_key, label in SECTION_LABELS.items():
            reference[section_key] = {
                "label": label,
                "required_fields": REQUIRED_FIELDS.get(section_key, []),
                "optional_fields": OPTIONAL_FIELDS.get(section_key, []),
                "total_required": len(REQUIRED_FIELDS.get(section_key, [])),
                "total_optional": len(OPTIONAL_FIELDS.get(section_key, [])),
            }
        return reference

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_field_values(
        self, passport: PassportData
    ) -> Dict[str, Any]:
        """Extract all field values from the passport data.

        Maps each PassportField enum value to its actual value from
        the passport data model.

        Args:
            passport: PassportData to extract from.

        Returns:
            Dict mapping field name to value.
        """
        gi = passport.general_info
        cf = passport.carbon_footprint
        dd = passport.supply_chain_dd
        mc = passport.material_composition
        pd = passport.performance_durability
        eol = passport.end_of_life

        return {
            # Section A
            PassportField.MANUFACTURER_ID.value: gi.manufacturer_id or None,
            PassportField.MANUFACTURING_PLANT.value: gi.manufacturing_plant or None,
            PassportField.MANUFACTURING_DATE.value: gi.manufacturing_date or None,
            PassportField.MANUFACTURING_COUNTRY.value: gi.manufacturing_country or None,
            PassportField.BATTERY_MODEL.value: gi.battery_model or None,
            PassportField.BATTERY_BATCH.value: gi.battery_batch or None,
            PassportField.BATTERY_SERIAL.value: gi.battery_serial or None,
            PassportField.BATTERY_WEIGHT.value: gi.battery_weight_kg,
            PassportField.BATTERY_CATEGORY.value: gi.battery_category or None,
            PassportField.BATTERY_CHEMISTRY.value: gi.battery_chemistry or None,
            PassportField.ENERGY_CAPACITY.value: gi.energy_capacity_kwh,
            PassportField.VOLTAGE_NOMINAL.value: gi.voltage_nominal,
            # Section B
            PassportField.CARBON_FOOTPRINT_TOTAL.value: cf.total_co2e_kg,
            PassportField.CARBON_FOOTPRINT_PER_KWH.value: cf.per_kwh_co2e_kg,
            PassportField.CARBON_FOOTPRINT_CLASS.value: cf.performance_class or None,
            PassportField.CARBON_FOOTPRINT_LIFECYCLE.value: cf.lifecycle_breakdown,
            PassportField.CARBON_FOOTPRINT_METHODOLOGY.value: cf.methodology,
            # Section C
            PassportField.DD_POLICY.value: dd.dd_policy or None,
            PassportField.DD_THIRD_PARTY_VERIFICATION.value: dd.third_party_verification or None,
            PassportField.DD_CONFLICT_MINERALS.value: dd.conflict_minerals or None,
            PassportField.DD_SUPPLY_CHAIN_MAPPING.value: dd.supply_chain_mapping or None,
            # Section D
            PassportField.MATERIAL_COMPOSITION.value: mc.bill_of_materials,
            PassportField.HAZARDOUS_SUBSTANCES.value: mc.hazardous_substances,
            PassportField.CRITICAL_RAW_MATERIALS.value: mc.critical_raw_materials,
            PassportField.RECYCLED_CONTENT.value: mc.recycled_content,
            # Section E
            PassportField.RATED_CAPACITY.value: pd.rated_capacity_ah,
            PassportField.CYCLE_LIFE_EXPECTED.value: pd.cycle_life_expected,
            PassportField.ENERGY_EFFICIENCY.value: pd.energy_efficiency_pct,
            PassportField.INTERNAL_RESISTANCE.value: pd.internal_resistance_mohm,
            PassportField.STATE_OF_HEALTH.value: pd.state_of_health_pct,
            PassportField.TEMPERATURE_RANGE.value: (
                f"{pd.temperature_range_min} to {pd.temperature_range_max}"
                if pd.temperature_range_min is not None
                and pd.temperature_range_max is not None
                else None
            ),
            PassportField.C_RATE_MAX.value: pd.c_rate_max,
            # Section F
            PassportField.EOL_COLLECTION_INFO.value: eol.collection_info or None,
            PassportField.EOL_RECYCLING_INFO.value: eol.recycling_info or None,
            PassportField.EOL_SECOND_LIFE_INFO.value: eol.second_life_info or None,
            PassportField.EOL_SAFETY_INSTRUCTIONS.value: eol.safety_instructions or None,
        }

    def _assess_field_quality(self, value: Any) -> DataQuality:
        """Assess data quality for a single field value.

        Args:
            value: Field value to assess.

        Returns:
            DataQuality enum value.
        """
        if value is None:
            return DataQuality.MISSING

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return DataQuality.MISSING
            if len(stripped) < 3:
                return DataQuality.PARTIAL
            return DataQuality.COMPLETE

        if isinstance(value, (int, float, Decimal)):
            return DataQuality.COMPLETE

        if isinstance(value, dict):
            if not value:
                return DataQuality.MISSING
            # Check if all values are empty
            non_empty = sum(
                1 for v in value.values()
                if v is not None and str(v).strip()
            )
            if non_empty == 0:
                return DataQuality.MISSING
            if non_empty < len(value):
                return DataQuality.PARTIAL
            return DataQuality.COMPLETE

        if isinstance(value, list):
            if not value:
                return DataQuality.MISSING
            return DataQuality.COMPLETE

        return DataQuality.COMPLETE

    def _get_field_section(self, field_name: str) -> str:
        """Determine which Annex XIII section a field belongs to.

        Args:
            field_name: Field name to look up.

        Returns:
            Section key (e.g., 'A_general', 'B_carbon_footprint').
        """
        for section_key, fields in REQUIRED_FIELDS.items():
            if field_name in fields:
                return section_key
        for section_key, fields in OPTIONAL_FIELDS.items():
            if field_name in fields:
                return section_key
        return "unknown"

    def _calculate_section_completeness(
        self, field_validations: List[FieldValidation]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate completeness for each Annex XIII section.

        Args:
            field_validations: List of per-field validation results.

        Returns:
            Dict mapping section to completeness metrics.
        """
        section_stats: Dict[str, Dict[str, int]] = {}

        for fv in field_validations:
            section = fv.section
            if section not in section_stats:
                section_stats[section] = {
                    "required_total": 0,
                    "required_met": 0,
                    "optional_total": 0,
                    "optional_met": 0,
                }

            if fv.is_required:
                section_stats[section]["required_total"] += 1
                if fv.value_present:
                    section_stats[section]["required_met"] += 1
            elif fv.field_name in ALL_OPTIONAL_FIELDS:
                section_stats[section]["optional_total"] += 1
                if fv.value_present:
                    section_stats[section]["optional_met"] += 1

        result: Dict[str, Dict[str, Any]] = {}
        for section_key, stats in section_stats.items():
            req_total = stats["required_total"]
            req_met = stats["required_met"]
            completeness = _round2(
                _safe_divide(float(req_met), float(req_total)) * 100.0
            ) if req_total > 0 else 100.0

            result[section_key] = {
                "label": SECTION_LABELS.get(section_key, section_key),
                "required_total": req_total,
                "required_met": req_met,
                "required_missing": req_total - req_met,
                "optional_total": stats["optional_total"],
                "optional_met": stats["optional_met"],
                "completeness_pct": completeness,
            }

        return result

    def _determine_status(
        self, completeness_pct: Decimal, errors: List[str]
    ) -> PassportStatus:
        """Determine passport status based on completeness.

        Args:
            completeness_pct: Overall completeness percentage.
            errors: List of validation errors.

        Returns:
            PassportStatus.
        """
        if completeness_pct >= Decimal("100") and not errors:
            return PassportStatus.VALIDATED
        if completeness_pct >= Decimal("80"):
            return PassportStatus.DRAFT
        return PassportStatus.DRAFT

    def _generate_recommendations(
        self,
        field_validations: List[FieldValidation],
        section_completeness: Dict[str, Dict[str, Any]],
        validation_errors: List[str],
    ) -> List[str]:
        """Generate recommendations for improving passport completeness.

        Args:
            field_validations: Per-field validation results.
            section_completeness: Per-section completeness metrics.
            validation_errors: List of validation errors.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Missing required fields
        missing_required = [
            fv for fv in field_validations
            if fv.is_required and not fv.value_present
        ]
        if missing_required:
            field_names = [fv.field_name for fv in missing_required[:5]]
            recommendations.append(
                f"{len(missing_required)} required field(s) are missing. "
                f"Priority fields: {', '.join(field_names)}. "
                f"Complete these to achieve Annex XIII compliance."
            )

        # Partial data quality fields
        partial_fields = [
            fv for fv in field_validations
            if fv.data_quality == DataQuality.PARTIAL
        ]
        if partial_fields:
            recommendations.append(
                f"{len(partial_fields)} field(s) have partial data quality. "
                f"Review and complete these fields to improve data accuracy."
            )

        # Section-level recommendations
        for section_key, stats in section_completeness.items():
            pct = stats.get("completeness_pct", 0)
            if isinstance(pct, (int, float)) and pct < 100:
                label = stats.get("label", section_key)
                missing = stats.get("required_missing", 0)
                if missing > 0:
                    recommendations.append(
                        f"Section '{label}': {missing} required field(s) "
                        f"missing ({pct}% complete). Provide data for "
                        f"all required fields in this section."
                    )

        # Carbon footprint section
        cf_stats = section_completeness.get("B_carbon_footprint", {})
        cf_pct = cf_stats.get("completeness_pct", 0)
        if isinstance(cf_pct, (int, float)) and cf_pct < 100:
            recommendations.append(
                "Carbon footprint information is incomplete. "
                "Run the CarbonFootprintEngine to generate Art 7 "
                "data and populate Section B of the passport."
            )

        # Recycled content section
        mc_stats = section_completeness.get("D_material_composition", {})
        mc_pct = mc_stats.get("completeness_pct", 0)
        if isinstance(mc_pct, (int, float)) and mc_pct < 100:
            recommendations.append(
                "Material composition section is incomplete. "
                "Run the RecycledContentEngine to generate Art 8 "
                "data and populate Section D of the passport."
            )

        # Overall readiness
        if not validation_errors:
            recommendations.append(
                "All required fields are populated. The passport is "
                "ready for validation and publication."
            )
        elif len(validation_errors) <= 3:
            recommendations.append(
                f"Only {len(validation_errors)} validation error(s) remain. "
                f"Address these to achieve full Annex XIII compliance."
            )

        return recommendations
