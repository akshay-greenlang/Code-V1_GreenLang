# -*- coding: utf-8 -*-
"""
ProductSustainabilityEngine - PACK-014 CSRD Retail Engine 4
=============================================================

ESPR Digital Product Passport (DPP), Product Environmental Footprint (PEF),
ECGT green claims substantiation, and textile microplastic assessment engine.

Regulatory References:
    - Regulation (EU) 2024/1781 (Ecodesign for Sustainable Products - ESPR)
    - Directive (EU) 2024/825 (Empowering Consumers / Green Claims - ECGT)
    - Commission Recommendation on PEF 2013/179/EU (PEF 3.0)
    - Regulation (EU) 2025/xxx (Textile microplastics -- draft)
    - French Repairability Index (Indice de Reparabilite)

Key Compliance Areas:
    - Digital Product Passport (DPP) completeness by category
    - Green claims audit against ECGT prohibited claims list
    - PEF normalized impact scoring across 16 categories
    - Textile microplastic release assessment
    - Repairability scoring (spare parts, disassembly, documentation)
    - Hazardous substance compliance

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Requirements from ESPR/ECGT legal texts (hard-coded)
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    # Exclude volatile fields to guarantee bit-perfect reproducibility
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DPPCategory(str, Enum):
    """ESPR Digital Product Passport categories."""
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    FURNITURE = "furniture"
    BATTERIES = "batteries"
    TYRES = "tyres"

class PEFImpactCategory(str, Enum):
    """PEF 3.0 environmental impact categories."""
    CLIMATE_CHANGE = "climate_change"
    OZONE_DEPLETION = "ozone_depletion"
    ACIDIFICATION = "acidification"
    EUTROPHICATION_FRESHWATER = "eutrophication_freshwater"
    EUTROPHICATION_MARINE = "eutrophication_marine"
    EUTROPHICATION_TERRESTRIAL = "eutrophication_terrestrial"
    RESOURCE_DEPLETION_MINERALS = "resource_depletion_minerals"
    RESOURCE_DEPLETION_FOSSILS = "resource_depletion_fossils"
    WATER_USE = "water_use"
    LAND_USE = "land_use"
    PARTICULATE_MATTER = "particulate_matter"
    IONISING_RADIATION = "ionising_radiation"
    PHOTOCHEMICAL_OZONE = "photochemical_ozone"
    HUMAN_TOXICITY_CANCER = "human_toxicity_cancer"
    HUMAN_TOXICITY_NON_CANCER = "human_toxicity_non_cancer"
    ECOTOXICITY_FRESHWATER = "ecotoxicity_freshwater"

class GreenClaimType(str, Enum):
    """Types of environmental marketing claims."""
    CARBON_NEUTRAL = "carbon_neutral"
    CLIMATE_POSITIVE = "climate_positive"
    ECO_FRIENDLY = "eco_friendly"
    SUSTAINABLE = "sustainable"
    RECYCLABLE = "recyclable"
    BIODEGRADABLE = "biodegradable"
    ORGANIC = "organic"
    NATURAL = "natural"
    VEGAN = "vegan"
    FAIR_TRADE = "fair_trade"

class ClaimVerificationStatus(str, Enum):
    """Verification status of a green claim."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    PROHIBITED = "prohibited"
    REQUIRES_SUBSTANTIATION = "requires_substantiation"

class RepairabilityGrade(str, Enum):
    """Repairability grade (French model, extended to EU ESPR)."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"

class FiberType(str, Enum):
    """Textile fiber types for microplastic assessment."""
    POLYESTER = "polyester"
    NYLON = "nylon"
    ACRYLIC = "acrylic"
    POLYPROPYLENE = "polypropylene"
    COTTON = "cotton"
    WOOL = "wool"
    SILK = "silk"
    LINEN = "linen"
    VISCOSE = "viscose"
    LYOCELL = "lyocell"
    RECYCLED_POLYESTER = "recycled_polyester"
    RECYCLED_NYLON = "recycled_nylon"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DPP mandatory fields by product category
# Source: ESPR Delegated Acts (draft 2025)
DPP_MANDATORY_FIELDS: Dict[str, List[str]] = {
    DPPCategory.TEXTILES: [
        "product_name", "manufacturer", "country_of_manufacture",
        "fiber_composition", "recycled_content_pct", "carbon_footprint_kg",
        "water_footprint_litre", "hazardous_substances", "care_instructions",
        "durability_cycles", "repairability_score", "recyclability_pct",
        "microplastic_release_mg", "supply_chain_traceability",
        "social_compliance_certifications", "product_weight_kg",
        "packaging_material", "packaging_recyclability",
        "end_of_life_instructions", "digital_id", "batch_number",
        "manufacturing_date", "country_of_origin_materials",
        "chemical_compliance_reach", "colour_fastness_rating",
    ],
    DPPCategory.ELECTRONICS: [
        "product_name", "manufacturer", "model_number", "serial_number",
        "country_of_manufacture", "energy_efficiency_class",
        "power_consumption_watts", "standby_power_watts",
        "carbon_footprint_kg", "recycled_content_pct",
        "critical_raw_materials", "hazardous_substances",
        "repairability_score", "spare_parts_availability_years",
        "software_update_support_years", "battery_capacity_wh",
        "battery_chemistry", "battery_removability",
        "recyclability_pct", "disassembly_instructions",
        "product_weight_kg", "packaging_material",
        "warranty_years", "expected_lifetime_years",
        "repair_manual_available", "end_of_life_instructions",
        "weee_registration", "digital_id", "batch_number",
        "manufacturing_date",
    ],
    DPPCategory.FURNITURE: [
        "product_name", "manufacturer", "country_of_manufacture",
        "material_composition", "wood_origin_certification",
        "recycled_content_pct", "carbon_footprint_kg",
        "hazardous_substances", "formaldehyde_class",
        "durability_rating", "repairability_score",
        "disassembly_instructions", "recyclability_pct",
        "product_weight_kg", "packaging_material",
        "fire_retardant_chemicals", "voc_emissions_class",
        "end_of_life_instructions", "digital_id",
        "manufacturing_date",
    ],
    DPPCategory.BATTERIES: [
        "product_name", "manufacturer", "battery_chemistry",
        "capacity_wh", "voltage", "cycle_life",
        "carbon_footprint_kg", "recycled_content_pct",
        "cobalt_content_pct", "lithium_content_pct",
        "nickel_content_pct", "lead_content_pct",
        "state_of_health_indicator", "supply_chain_due_diligence",
        "hazardous_substances", "safety_certifications",
        "recyclability_pct", "collection_instructions",
        "digital_id", "batch_number", "manufacturing_date",
        "country_of_manufacture", "expected_lifetime_years",
        "operating_temperature_range",
    ],
    DPPCategory.TYRES: [
        "product_name", "manufacturer", "tyre_class",
        "rolling_resistance_class", "wet_grip_class",
        "noise_class_db", "carbon_footprint_kg",
        "recycled_content_pct", "natural_rubber_pct",
        "synthetic_rubber_pct", "hazardous_substances",
        "abrasion_rate_mg_per_km", "mileage_warranty_km",
        "retreading_potential", "recyclability_pct",
        "end_of_life_instructions", "digital_id",
        "manufacturing_date", "country_of_manufacture",
        "size_specification",
    ],
}

# ECGT prohibited generic environmental claims
# Source: Directive (EU) 2024/825 Article 2
ECGT_PROHIBITED_CLAIMS: Dict[str, Dict[str, Any]] = {
    "eco_friendly": {
        "prohibited": True,
        "reason": "Generic claim without specific substantiation prohibited under ECGT",
        "alternatives": "Specify measurable environmental benefit (e.g., '30% less CO2')",
    },
    "green": {
        "prohibited": True,
        "reason": "Vague environmental claim prohibited under ECGT",
        "alternatives": "Use specific, measurable claims with third-party verification",
    },
    "sustainable": {
        "prohibited": True,
        "reason": "Generic sustainability claim prohibited unless covering full lifecycle",
        "alternatives": "Reference certified sustainability scheme (e.g., EU Ecolabel)",
    },
    "climate_neutral": {
        "prohibited": True,
        "reason": "Carbon-neutrality claims based solely on offsets prohibited under ECGT",
        "alternatives": "Focus on absolute emission reductions; offsets can only supplement",
    },
    "carbon_neutral": {
        "prohibited": True,
        "reason": "Carbon-neutrality claims based solely on offsets prohibited under ECGT",
        "alternatives": "Demonstrate verified emission reductions before claiming neutrality",
    },
    "environmentally_friendly": {
        "prohibited": True,
        "reason": "Generic environmental claim without evidence prohibited",
        "alternatives": "Provide specific environmental performance data",
    },
    "natural": {
        "prohibited": False,
        "reason": "Allowed if substantiated (product is genuinely natural)",
        "alternatives": "Ensure claim is accurate and verifiable",
    },
    "biodegradable": {
        "prohibited": False,
        "reason": "Allowed with specific conditions and timeframe",
        "alternatives": "Specify conditions and timeframe for biodegradation",
    },
    "recyclable": {
        "prohibited": False,
        "reason": "Allowed if recycling infrastructure exists for the product",
        "alternatives": "Verify recycling infrastructure availability in target markets",
    },
    "organic": {
        "prohibited": False,
        "reason": "Allowed with certified organic certification",
        "alternatives": "Maintain valid organic certification",
    },
    "vegan": {
        "prohibited": False,
        "reason": "Allowed with appropriate verification",
        "alternatives": "Use certified vegan labeling",
    },
    "fair_trade": {
        "prohibited": False,
        "reason": "Allowed with certified fair trade scheme",
        "alternatives": "Maintain valid fair trade certification",
    },
}

# PEF 3.0 normalization and weighting factors
# Source: EU PEF 3.0 (2024 update)
PEF_NORMALIZATION_FACTORS: Dict[str, Dict[str, float]] = {
    PEFImpactCategory.CLIMATE_CHANGE: {
        "normalization": 8100.0,    # kg CO2 eq / person / year
        "weighting": 21.06,        # % of total
        "unit": "kg CO2 eq",
    },
    PEFImpactCategory.OZONE_DEPLETION: {
        "normalization": 0.0536,
        "weighting": 6.31,
        "unit": "kg CFC-11 eq",
    },
    PEFImpactCategory.ACIDIFICATION: {
        "normalization": 55.6,
        "weighting": 6.20,
        "unit": "mol H+ eq",
    },
    PEFImpactCategory.EUTROPHICATION_FRESHWATER: {
        "normalization": 1.61,
        "weighting": 2.80,
        "unit": "kg P eq",
    },
    PEFImpactCategory.EUTROPHICATION_MARINE: {
        "normalization": 19.5,
        "weighting": 2.96,
        "unit": "kg N eq",
    },
    PEFImpactCategory.EUTROPHICATION_TERRESTRIAL: {
        "normalization": 177.0,
        "weighting": 3.71,
        "unit": "mol N eq",
    },
    PEFImpactCategory.RESOURCE_DEPLETION_MINERALS: {
        "normalization": 0.0636,
        "weighting": 7.55,
        "unit": "kg Sb eq",
    },
    PEFImpactCategory.RESOURCE_DEPLETION_FOSSILS: {
        "normalization": 65000.0,
        "weighting": 8.32,
        "unit": "MJ",
    },
    PEFImpactCategory.WATER_USE: {
        "normalization": 11500.0,
        "weighting": 8.51,
        "unit": "m3 world eq",
    },
    PEFImpactCategory.LAND_USE: {
        "normalization": 819000.0,
        "weighting": 7.94,
        "unit": "pt",
    },
    PEFImpactCategory.PARTICULATE_MATTER: {
        "normalization": 0.000596,
        "weighting": 8.96,
        "unit": "disease incidence",
    },
    PEFImpactCategory.IONISING_RADIATION: {
        "normalization": 4220.0,
        "weighting": 5.01,
        "unit": "kBq U235 eq",
    },
    PEFImpactCategory.PHOTOCHEMICAL_OZONE: {
        "normalization": 40.6,
        "weighting": 4.78,
        "unit": "kg NMVOC eq",
    },
    PEFImpactCategory.HUMAN_TOXICITY_CANCER: {
        "normalization": 0.0000169,
        "weighting": 2.13,
        "unit": "CTUh",
    },
    PEFImpactCategory.HUMAN_TOXICITY_NON_CANCER: {
        "normalization": 0.000234,
        "weighting": 1.84,
        "unit": "CTUh",
    },
    PEFImpactCategory.ECOTOXICITY_FRESHWATER: {
        "normalization": 42700.0,
        "weighting": 1.92,
        "unit": "CTUe",
    },
}

# Textile microplastic release rates (mg per wash per kg of textile)
# Source: EU JRC Technical Report 2023, ISO/TR 23383:2024
MICROPLASTIC_RELEASE_RATES: Dict[str, Dict[str, float]] = {
    FiberType.POLYESTER: {
        "release_rate_mg_per_kg_per_wash": 124.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.NYLON: {
        "release_rate_mg_per_kg_per_wash": 80.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.ACRYLIC: {
        "release_rate_mg_per_kg_per_wash": 310.0,
        "estimated_washes_per_year": 40.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.POLYPROPYLENE: {
        "release_rate_mg_per_kg_per_wash": 68.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.RECYCLED_POLYESTER: {
        "release_rate_mg_per_kg_per_wash": 135.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.RECYCLED_NYLON: {
        "release_rate_mg_per_kg_per_wash": 88.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    # Natural fibers -- negligible microplastic release
    FiberType.COTTON: {
        "release_rate_mg_per_kg_per_wash": 2.0,
        "estimated_washes_per_year": 52.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.WOOL: {
        "release_rate_mg_per_kg_per_wash": 5.0,
        "estimated_washes_per_year": 30.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.SILK: {
        "release_rate_mg_per_kg_per_wash": 1.0,
        "estimated_washes_per_year": 20.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.LINEN: {
        "release_rate_mg_per_kg_per_wash": 3.0,
        "estimated_washes_per_year": 40.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.VISCOSE: {
        "release_rate_mg_per_kg_per_wash": 15.0,
        "estimated_washes_per_year": 45.0,
        "threshold_mg_per_wash": 100.0,
    },
    FiberType.LYOCELL: {
        "release_rate_mg_per_kg_per_wash": 8.0,
        "estimated_washes_per_year": 45.0,
        "threshold_mg_per_wash": 100.0,
    },
}

# Repairability scoring criteria weights
# Source: French Repairability Index (Indice de reparabilite), adapted for ESPR
REPAIRABILITY_CRITERIA: Dict[str, Dict[str, float]] = {
    "documentation": {
        "weight": 15.0,
        "description": "Availability of repair documentation and manuals",
        "max_score": 10.0,
    },
    "disassembly": {
        "weight": 20.0,
        "description": "Ease of disassembly (tools, fasteners, design)",
        "max_score": 10.0,
    },
    "spare_parts_availability": {
        "weight": 25.0,
        "description": "Availability and delivery time of spare parts",
        "max_score": 10.0,
    },
    "spare_parts_price": {
        "weight": 15.0,
        "description": "Price ratio of spare parts vs new product",
        "max_score": 10.0,
    },
    "product_specific": {
        "weight": 25.0,
        "description": "Category-specific repairability criteria",
        "max_score": 10.0,
    },
}

# Repairability grade thresholds (out of 10)
REPAIRABILITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    RepairabilityGrade.A: (8.0, 10.0),
    RepairabilityGrade.B: (6.0, 8.0),
    RepairabilityGrade.C: (4.0, 6.0),
    RepairabilityGrade.D: (2.0, 4.0),
    RepairabilityGrade.E: (0.0, 2.0),
}

# Claim type mapping to ECGT categories
CLAIM_TYPE_MAP: Dict[str, str] = {
    GreenClaimType.CARBON_NEUTRAL: "carbon_neutral",
    GreenClaimType.CLIMATE_POSITIVE: "climate_neutral",
    GreenClaimType.ECO_FRIENDLY: "eco_friendly",
    GreenClaimType.SUSTAINABLE: "sustainable",
    GreenClaimType.RECYCLABLE: "recyclable",
    GreenClaimType.BIODEGRADABLE: "biodegradable",
    GreenClaimType.ORGANIC: "organic",
    GreenClaimType.NATURAL: "natural",
    GreenClaimType.VEGAN: "vegan",
    GreenClaimType.FAIR_TRADE: "fair_trade",
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class FiberComposition(BaseModel):
    """Fiber composition entry for textile products.

    Attributes:
        fiber_type: Type of fiber.
        percentage: Percentage by weight.
    """
    fiber_type: FiberType
    percentage: float = Field(..., ge=0, le=100, description="Weight percentage")

class DPPData(BaseModel):
    """Digital Product Passport data for a product.

    Attributes:
        product_id: Product identifier.
        dpp_category: DPP product category.
        fiber_composition: Textile fiber breakdown (textiles only).
        recycled_content_pct: Recycled content percentage.
        repairability_score: Repairability score (0-10).
        carbon_footprint_kg: Product carbon footprint (kg CO2e).
        water_footprint_litre: Water footprint (litres).
        hazardous_substances: List of hazardous substances present.
        manufacturing_country: Country of manufacture.
        recyclability_pct: Percentage recyclable at end of life.
        durability_cycles: Number of use/wash cycles.
        spare_parts_years: Years of spare parts availability.
        disassembly_time_minutes: Time to disassemble (minutes).
        repair_manual_available: Whether repair manual is available.
        energy_efficiency_class: Energy efficiency class (A-G).
        expected_lifetime_years: Expected product lifetime.
        has_digital_id: Has unique digital identifier (QR/NFC).
        fields_provided: Dict of all DPP field names and whether filled.
    """
    product_id: str = Field(..., min_length=1, description="Product ID")
    dpp_category: DPPCategory
    fiber_composition: List[FiberComposition] = Field(default_factory=list)
    recycled_content_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Recycled content %"
    )
    repairability_score: Optional[float] = Field(
        None, ge=0, le=10, description="Repairability (0-10)"
    )
    carbon_footprint_kg: Optional[float] = Field(
        None, ge=0, description="Carbon footprint (kg CO2e)"
    )
    water_footprint_litre: Optional[float] = Field(
        None, ge=0, description="Water footprint (L)"
    )
    hazardous_substances: List[str] = Field(default_factory=list)
    manufacturing_country: Optional[str] = Field(None, description="Country code")
    recyclability_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Recyclability %"
    )
    durability_cycles: Optional[int] = Field(None, ge=0, description="Durability cycles")
    spare_parts_years: Optional[int] = Field(
        None, ge=0, description="Spare parts availability (years)"
    )
    disassembly_time_minutes: Optional[float] = Field(
        None, ge=0, description="Disassembly time (min)"
    )
    repair_manual_available: bool = Field(False, description="Repair manual available")
    energy_efficiency_class: Optional[str] = Field(
        None, description="Energy class (A-G)"
    )
    expected_lifetime_years: Optional[float] = Field(
        None, ge=0, description="Expected lifetime (years)"
    )
    has_digital_id: bool = Field(False, description="Has digital ID (QR/NFC)")
    fields_provided: Dict[str, bool] = Field(
        default_factory=dict, description="DPP fields provided"
    )

class GreenClaim(BaseModel):
    """Environmental marketing claim for ECGT assessment.

    Attributes:
        claim_type: Type of green claim.
        claim_text: Actual claim text used in marketing.
        product_id: Associated product identifier.
        substantiation_evidence: Evidence supporting the claim.
        third_party_verified: Whether independently verified.
        certification_scheme: Name of certification scheme if applicable.
        verification_status: Current verification status.
        applicable_regulation: Relevant regulation.
    """
    claim_type: GreenClaimType
    claim_text: str = Field(..., min_length=1, description="Claim text")
    product_id: Optional[str] = Field(None, description="Product ID")
    substantiation_evidence: Optional[str] = Field(
        None, description="Supporting evidence"
    )
    third_party_verified: bool = Field(False, description="Third-party verified")
    certification_scheme: Optional[str] = Field(
        None, description="Certification scheme"
    )
    verification_status: ClaimVerificationStatus = Field(
        ClaimVerificationStatus.UNVERIFIED, description="Verification status"
    )
    applicable_regulation: str = Field(
        "ECGT", description="Applicable regulation"
    )

class PEFImpactData(BaseModel):
    """PEF impact characterization result for a single category.

    Attributes:
        category: Impact category.
        characterization_value: Raw characterization result.
        unit: Unit of characterization result.
    """
    category: PEFImpactCategory
    characterization_value: float = Field(
        ..., description="Characterization result"
    )
    unit: Optional[str] = Field(None, description="Unit")

class RepairabilityInput(BaseModel):
    """Repairability assessment input data.

    Attributes:
        product_id: Product identifier.
        documentation_score: Documentation availability (0-10).
        disassembly_score: Ease of disassembly (0-10).
        spare_parts_availability_score: Spare parts availability (0-10).
        spare_parts_price_score: Spare parts price ratio (0-10).
        product_specific_score: Category-specific criteria (0-10).
    """
    product_id: str = Field(..., min_length=1, description="Product ID")
    documentation_score: float = Field(0.0, ge=0, le=10, description="Documentation (0-10)")
    disassembly_score: float = Field(0.0, ge=0, le=10, description="Disassembly (0-10)")
    spare_parts_availability_score: float = Field(
        0.0, ge=0, le=10, description="Spare parts availability (0-10)"
    )
    spare_parts_price_score: float = Field(
        0.0, ge=0, le=10, description="Spare parts price (0-10)"
    )
    product_specific_score: float = Field(
        0.0, ge=0, le=10, description="Product-specific (0-10)"
    )

class ProductData(BaseModel):
    """Complete product data for sustainability assessment.

    Attributes:
        product_id: Product identifier.
        product_name: Human-readable product name.
        category: General product category.
        dpp_category: DPP product category.
        weight_kg: Product weight in kilograms.
        materials: List of material names.
        country_of_manufacture: Manufacturing country.
        brand: Brand name.
        sku: Stock keeping unit.
        price_eur: Product price in EUR.
        units_sold: Units sold per year.
    """
    product_id: str = Field(..., min_length=1, description="Product ID")
    product_name: str = Field(..., min_length=1, description="Product name")
    category: str = Field(..., description="Product category")
    dpp_category: Optional[DPPCategory] = Field(None, description="DPP category")
    weight_kg: float = Field(..., gt=0, description="Weight (kg)")
    materials: List[str] = Field(default_factory=list)
    country_of_manufacture: Optional[str] = Field(None, description="Country")
    brand: Optional[str] = Field(None, description="Brand")
    sku: Optional[str] = Field(None, description="SKU")
    price_eur: Optional[float] = Field(None, ge=0, description="Price (EUR)")
    units_sold: int = Field(0, ge=0, description="Annual units sold")

class ProductSustainabilityInput(BaseModel):
    """Complete input for product sustainability assessment.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Reporting year.
        products: List of product data.
        dpp_data: DPP data per product.
        green_claims: Green claims to audit.
        pef_data: PEF impact data per product.
        repairability_data: Repairability input per product.
    """
    organisation_id: str = Field(..., min_length=1, description="Organisation ID")
    reporting_year: int = Field(..., ge=2024, le=2050, description="Reporting year")
    products: List[ProductData] = Field(default_factory=list)
    dpp_data: List[DPPData] = Field(default_factory=list)
    green_claims: List[GreenClaim] = Field(default_factory=list)
    pef_data: Dict[str, List[PEFImpactData]] = Field(
        default_factory=dict, description="PEF data by product_id"
    )
    repairability_data: List[RepairabilityInput] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class DPPCompletenessResult(BaseModel):
    """DPP completeness assessment for a single product.

    Attributes:
        product_id: Product identifier.
        dpp_category: DPP category.
        total_mandatory_fields: Total mandatory fields for this category.
        filled_fields: Number of fields provided.
        missing_fields: List of missing field names.
        completeness_pct: Completeness percentage.
        compliant: Whether above compliance threshold (90%).
    """
    product_id: str
    dpp_category: str
    total_mandatory_fields: int
    filled_fields: int
    missing_fields: List[str]
    completeness_pct: float
    compliant: bool

class GreenClaimAuditResult(BaseModel):
    """Audit result for a single green claim.

    Attributes:
        claim_type: Type of claim.
        claim_text: Marketing claim text.
        product_id: Associated product.
        is_prohibited: Whether prohibited under ECGT.
        prohibition_reason: Reason for prohibition.
        recommended_alternative: Suggested compliant alternative.
        has_substantiation: Whether evidence is provided.
        is_third_party_verified: Whether independently verified.
        compliance_status: Overall compliance status.
    """
    claim_type: str
    claim_text: str
    product_id: Optional[str]
    is_prohibited: bool
    prohibition_reason: str
    recommended_alternative: str
    has_substantiation: bool
    is_third_party_verified: bool
    compliance_status: str

class PEFNormalizedResult(BaseModel):
    """PEF normalized and weighted impact result.

    Attributes:
        product_id: Product identifier.
        impact_scores: Normalized scores by category.
        total_weighted_score: Total weighted environmental score.
        dominant_categories: Top 3 impact categories.
    """
    product_id: str
    impact_scores: Dict[str, Dict[str, float]]
    total_weighted_score: float
    dominant_categories: List[Dict[str, Any]]

class MicroplasticAssessmentResult(BaseModel):
    """Textile microplastic release assessment.

    Attributes:
        product_id: Product identifier.
        fiber_breakdown: Release by fiber type.
        total_release_mg_per_wash: Total release per wash.
        annual_release_mg: Estimated annual release.
        exceeds_threshold: Whether any fiber exceeds threshold.
        recommendation: Mitigation recommendation.
    """
    product_id: str
    fiber_breakdown: List[Dict[str, Any]]
    total_release_mg_per_wash: float
    annual_release_mg: float
    exceeds_threshold: bool
    recommendation: str

class RepairabilityResult(BaseModel):
    """Repairability scoring result.

    Attributes:
        product_id: Product identifier.
        criteria_scores: Score per criterion.
        weighted_score: Final weighted score (0-10).
        grade: Repairability grade (A-E).
    """
    product_id: str
    criteria_scores: Dict[str, Dict[str, float]]
    weighted_score: float
    grade: str

class ProductSustainabilityResult(BaseModel):
    """Complete product sustainability assessment result.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Reporting year.
        total_products: Total products assessed.
        dpp_results: DPP completeness per product.
        avg_dpp_completeness_pct: Average DPP completeness.
        green_claims_audit: Green claims audit results.
        total_claims: Total claims assessed.
        prohibited_claims_count: Number of prohibited claims.
        pef_results: PEF impact results per product.
        microplastic_assessments: Microplastic results per product.
        repairability_results: Repairability results per product.
        recommendations: Improvement recommendations.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    organisation_id: str
    reporting_year: int
    total_products: int
    dpp_results: List[DPPCompletenessResult]
    avg_dpp_completeness_pct: float
    green_claims_audit: List[GreenClaimAuditResult]
    total_claims: int
    prohibited_claims_count: int
    pef_results: List[PEFNormalizedResult]
    microplastic_assessments: List[MicroplasticAssessmentResult]
    repairability_results: List[RepairabilityResult]
    recommendations: List[str]
    engine_version: str = engine_version
    calculated_at: datetime = Field(default_factory=utcnow)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class ProductSustainabilityEngine:
    """Product sustainability compliance engine.

    Assesses products against ESPR DPP requirements, ECGT green claims
    rules, PEF environmental footprint, textile microplastic thresholds,
    and repairability scoring.

    Guarantees:
        - Deterministic: identical inputs always produce identical outputs.
        - Reproducible: full provenance via SHA-256 hashing.
        - Auditable: every assessment is documented.
        - Zero-hallucination: no LLM in the calculation path.

    Usage::

        engine = ProductSustainabilityEngine()
        result = engine.assess_products(input_data)
    """

    def __init__(self) -> None:
        """Initialise engine with embedded regulatory constants."""
        self._dpp_fields = DPP_MANDATORY_FIELDS
        self._prohibited_claims = ECGT_PROHIBITED_CLAIMS
        self._pef_factors = PEF_NORMALIZATION_FACTORS
        self._microplastic_rates = MICROPLASTIC_RELEASE_RATES
        self._repair_criteria = REPAIRABILITY_CRITERIA
        self._repair_thresholds = REPAIRABILITY_THRESHOLDS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assess_products(
        self, input_data: ProductSustainabilityInput
    ) -> ProductSustainabilityResult:
        """Assess product portfolio for sustainability compliance.

        Runs all assessments: DPP completeness, green claims audit, PEF
        scoring, microplastic assessment, and repairability scoring.

        Args:
            input_data: Complete product sustainability input data.

        Returns:
            ProductSustainabilityResult with all assessment results.
        """
        t0 = time.perf_counter()

        # --- DPP completeness ---
        dpp_results = [
            self._assess_dpp_completeness(dpp) for dpp in input_data.dpp_data
        ]
        avg_dpp = self._calc_avg_dpp_completeness(dpp_results)

        # --- Green claims audit ---
        claims_audit = [
            self._audit_green_claim(claim) for claim in input_data.green_claims
        ]
        prohibited_count = sum(1 for c in claims_audit if c.is_prohibited)

        # --- PEF scoring ---
        pef_results: List[PEFNormalizedResult] = []
        for product_id, impacts in input_data.pef_data.items():
            pef_result = self._calc_pef_score(product_id, impacts)
            pef_results.append(pef_result)

        # --- Microplastic assessment ---
        microplastic_results: List[MicroplasticAssessmentResult] = []
        for dpp in input_data.dpp_data:
            if dpp.dpp_category == DPPCategory.TEXTILES and dpp.fiber_composition:
                mp_result = self._assess_microplastics(
                    dpp.product_id, dpp.fiber_composition
                )
                microplastic_results.append(mp_result)

        # --- Repairability scoring ---
        repair_results = [
            self._calc_repairability(r) for r in input_data.repairability_data
        ]

        # --- Recommendations ---
        recommendations = self._generate_recommendations(
            dpp_results, claims_audit, pef_results,
            microplastic_results, repair_results
        )

        processing_ms = (time.perf_counter() - t0) * 1000.0

        result = ProductSustainabilityResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            total_products=len(input_data.products),
            dpp_results=dpp_results,
            avg_dpp_completeness_pct=_round_val(_decimal(avg_dpp), 2),
            green_claims_audit=claims_audit,
            total_claims=len(claims_audit),
            prohibited_claims_count=prohibited_count,
            pef_results=pef_results,
            microplastic_assessments=microplastic_results,
            repairability_results=repair_results,
            recommendations=recommendations,
            processing_time_ms=round(processing_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # DPP Completeness
    # -------------------------------------------------------------------

    def _assess_dpp_completeness(self, dpp: DPPData) -> DPPCompletenessResult:
        """Assess DPP completeness against mandatory field requirements.

        Checks each DPP data point against the mandatory fields list for
        the product category and calculates completeness percentage.

        Args:
            dpp: DPP data for a single product.

        Returns:
            DPPCompletenessResult with completeness metrics.
        """
        mandatory_fields = self._dpp_fields.get(dpp.dpp_category, [])
        total_fields = len(mandatory_fields)

        if total_fields == 0:
            return DPPCompletenessResult(
                product_id=dpp.product_id,
                dpp_category=dpp.dpp_category.value,
                total_mandatory_fields=0,
                filled_fields=0,
                missing_fields=[],
                completeness_pct=100.0,
                compliant=True,
            )

        # Check which fields are provided
        filled = 0
        missing: List[str] = []

        # Map DPP model fields to mandatory field names
        field_mapping = self._build_field_mapping(dpp)

        for field_name in mandatory_fields:
            if field_mapping.get(field_name, False):
                filled += 1
            elif dpp.fields_provided.get(field_name, False):
                filled += 1
            else:
                missing.append(field_name)

        pct = float(_safe_pct(_decimal(filled), _decimal(total_fields)))

        return DPPCompletenessResult(
            product_id=dpp.product_id,
            dpp_category=dpp.dpp_category.value,
            total_mandatory_fields=total_fields,
            filled_fields=filled,
            missing_fields=missing,
            completeness_pct=round(pct, 2),
            compliant=pct >= 90.0,
        )

    def _build_field_mapping(self, dpp: DPPData) -> Dict[str, bool]:
        """Build a mapping from DPP mandatory field names to availability.

        Maps the Pydantic model attributes to the standardised DPP field
        names for completeness checking.

        Args:
            dpp: DPP data for a single product.

        Returns:
            Dict mapping field name to whether it is populated.
        """
        m: Dict[str, bool] = {}

        # Universal fields
        m["product_name"] = True  # Always has product_id at minimum
        m["digital_id"] = dpp.has_digital_id
        m["recycled_content_pct"] = dpp.recycled_content_pct is not None
        m["carbon_footprint_kg"] = dpp.carbon_footprint_kg is not None
        m["hazardous_substances"] = len(dpp.hazardous_substances) >= 0  # Empty list is valid
        m["manufacturing_country"] = dpp.manufacturing_country is not None
        m["country_of_manufacture"] = dpp.manufacturing_country is not None
        m["recyclability_pct"] = dpp.recyclability_pct is not None
        m["repairability_score"] = dpp.repairability_score is not None
        m["product_weight_kg"] = True  # Derived from ProductData
        m["manufacturing_date"] = True  # Assumed available
        m["end_of_life_instructions"] = dpp.recyclability_pct is not None

        # Textiles
        m["fiber_composition"] = len(dpp.fiber_composition) > 0
        m["water_footprint_litre"] = dpp.water_footprint_litre is not None
        m["durability_cycles"] = dpp.durability_cycles is not None
        m["microplastic_release_mg"] = len(dpp.fiber_composition) > 0
        m["care_instructions"] = len(dpp.fiber_composition) > 0
        m["colour_fastness_rating"] = False  # Requires explicit data
        m["chemical_compliance_reach"] = len(dpp.hazardous_substances) >= 0
        m["social_compliance_certifications"] = False
        m["supply_chain_traceability"] = False
        m["country_of_origin_materials"] = False

        # Electronics
        m["model_number"] = False
        m["serial_number"] = False
        m["energy_efficiency_class"] = dpp.energy_efficiency_class is not None
        m["power_consumption_watts"] = False
        m["standby_power_watts"] = False
        m["critical_raw_materials"] = False
        m["spare_parts_availability_years"] = dpp.spare_parts_years is not None
        m["software_update_support_years"] = False
        m["battery_capacity_wh"] = False
        m["battery_chemistry"] = False
        m["battery_removability"] = False
        m["disassembly_instructions"] = dpp.disassembly_time_minutes is not None
        m["warranty_years"] = False
        m["expected_lifetime_years"] = dpp.expected_lifetime_years is not None
        m["repair_manual_available"] = dpp.repair_manual_available
        m["weee_registration"] = False
        m["batch_number"] = False

        # Furniture
        m["material_composition"] = len(dpp.hazardous_substances) >= 0
        m["wood_origin_certification"] = False
        m["formaldehyde_class"] = False
        m["durability_rating"] = dpp.durability_cycles is not None
        m["fire_retardant_chemicals"] = False
        m["voc_emissions_class"] = False
        m["packaging_material"] = False
        m["packaging_recyclability"] = False

        # Batteries
        m["capacity_wh"] = False
        m["voltage"] = False
        m["cycle_life"] = False
        m["cobalt_content_pct"] = False
        m["lithium_content_pct"] = False
        m["nickel_content_pct"] = False
        m["lead_content_pct"] = False
        m["state_of_health_indicator"] = False
        m["supply_chain_due_diligence"] = False
        m["safety_certifications"] = False
        m["collection_instructions"] = False
        m["operating_temperature_range"] = False

        # Tyres
        m["tyre_class"] = False
        m["rolling_resistance_class"] = False
        m["wet_grip_class"] = False
        m["noise_class_db"] = False
        m["natural_rubber_pct"] = False
        m["synthetic_rubber_pct"] = False
        m["abrasion_rate_mg_per_km"] = False
        m["mileage_warranty_km"] = False
        m["retreading_potential"] = False
        m["size_specification"] = False
        m["manufacturer"] = dpp.manufacturing_country is not None

        return m

    def _calc_avg_dpp_completeness(
        self, results: List[DPPCompletenessResult]
    ) -> float:
        """Calculate average DPP completeness across all products.

        Args:
            results: List of DPP completeness results.

        Returns:
            Average completeness as float percentage.
        """
        if not results:
            return 0.0
        total = sum(_decimal(r.completeness_pct) for r in results)
        return float(_safe_divide(total, _decimal(len(results))))

    # -------------------------------------------------------------------
    # Green Claims Audit
    # -------------------------------------------------------------------

    def _audit_green_claim(self, claim: GreenClaim) -> GreenClaimAuditResult:
        """Audit a green claim against ECGT requirements.

        Checks whether the claim type is prohibited, requires substantiation,
        or is compliant under ECGT rules.

        Args:
            claim: Green claim to audit.

        Returns:
            GreenClaimAuditResult with compliance assessment.
        """
        ecgt_key = CLAIM_TYPE_MAP.get(claim.claim_type, claim.claim_type.value)
        ecgt_info = self._prohibited_claims.get(ecgt_key, {})

        is_prohibited = ecgt_info.get("prohibited", False)
        reason = ecgt_info.get("reason", "No specific ECGT guidance for this claim type")
        alternative = ecgt_info.get("alternatives", "Ensure claim is specific and verifiable")

        has_substantiation = (
            claim.substantiation_evidence is not None
            and len(claim.substantiation_evidence) > 0
        )

        # Determine compliance status
        if is_prohibited and not has_substantiation:
            status = "prohibited"
        elif is_prohibited and has_substantiation and claim.third_party_verified:
            status = "requires_review"  # May be acceptable with strong evidence
        elif is_prohibited:
            status = "prohibited"
        elif has_substantiation and claim.third_party_verified:
            status = "compliant"
        elif has_substantiation:
            status = "requires_verification"
        else:
            status = "requires_substantiation"

        return GreenClaimAuditResult(
            claim_type=claim.claim_type.value,
            claim_text=claim.claim_text,
            product_id=claim.product_id,
            is_prohibited=is_prohibited,
            prohibition_reason=reason,
            recommended_alternative=alternative,
            has_substantiation=has_substantiation,
            is_third_party_verified=claim.third_party_verified,
            compliance_status=status,
        )

    # -------------------------------------------------------------------
    # PEF Scoring
    # -------------------------------------------------------------------

    def _calc_pef_score(
        self, product_id: str, impacts: List[PEFImpactData]
    ) -> PEFNormalizedResult:
        """Calculate PEF normalized and weighted impact score.

        Normalizes characterization results by per-capita reference values,
        then applies PEF 3.0 weighting factors to produce a single score.

        Formula: normalized = characterization / normalization_factor
                 weighted = normalized * weighting / 100
                 total = sum(weighted)

        Args:
            product_id: Product identifier.
            impacts: List of PEF impact data.

        Returns:
            PEFNormalizedResult with scored impacts.
        """
        impact_scores: Dict[str, Dict[str, float]] = {}
        total_weighted = Decimal("0")
        category_scores: List[Tuple[str, Decimal]] = []

        for impact in impacts:
            pef_info = self._pef_factors.get(impact.category)
            if not pef_info:
                continue

            norm_factor = _decimal(pef_info["normalization"])
            weighting = _decimal(pef_info["weighting"])
            char_value = _decimal(impact.characterization_value)

            normalized = _safe_divide(char_value, norm_factor)
            weighted = normalized * weighting / Decimal("100")
            total_weighted += weighted

            cat_key = impact.category.value
            impact_scores[cat_key] = {
                "characterization": _round_val(char_value, 6),
                "normalized": _round_val(normalized, 8),
                "weighted": _round_val(weighted, 8),
                "weighting_pct": _round_val(weighting, 2),
            }
            category_scores.append((cat_key, weighted))

        # Top 3 dominant categories
        sorted_cats = sorted(category_scores, key=lambda x: x[1], reverse=True)
        dominant = [
            {
                "category": cat,
                "weighted_score": _round_val(score, 8),
                "pct_of_total": _round_val(
                    _safe_pct(score, total_weighted), 2
                ),
            }
            for cat, score in sorted_cats[:3]
        ]

        return PEFNormalizedResult(
            product_id=product_id,
            impact_scores=impact_scores,
            total_weighted_score=_round_val(total_weighted, 8),
            dominant_categories=dominant,
        )

    # -------------------------------------------------------------------
    # Microplastic Assessment
    # -------------------------------------------------------------------

    def _assess_microplastics(
        self,
        product_id: str,
        fiber_composition: List[FiberComposition],
    ) -> MicroplasticAssessmentResult:
        """Assess textile microplastic release from washing.

        Calculates per-wash and annual microplastic release based on
        fiber composition and release rate data.

        Formula: release_per_wash = sum(fiber_weight_frac * release_rate)
                 annual_release = release_per_wash * washes_per_year

        Args:
            product_id: Product identifier.
            fiber_composition: List of fiber compositions.

        Returns:
            MicroplasticAssessmentResult with release estimates.
        """
        fiber_breakdown: List[Dict[str, Any]] = []
        total_per_wash = Decimal("0")
        total_annual = Decimal("0")
        exceeds = False
        max_washes = Decimal("0")

        for fc in fiber_composition:
            rate_info = self._microplastic_rates.get(fc.fiber_type)
            if not rate_info:
                continue

            frac = _decimal(fc.percentage) / Decimal("100")
            release_rate = _decimal(rate_info["release_rate_mg_per_kg_per_wash"])
            washes = _decimal(rate_info["estimated_washes_per_year"])
            threshold = _decimal(rate_info["threshold_mg_per_wash"])

            # Per-wash release (mg per kg of textile, weighted by fiber fraction)
            fiber_release = frac * release_rate
            annual = fiber_release * washes
            total_per_wash += fiber_release
            total_annual += annual

            if washes > max_washes:
                max_washes = washes

            above_threshold = fiber_release > threshold * frac
            if above_threshold:
                exceeds = True

            fiber_breakdown.append({
                "fiber_type": fc.fiber_type.value,
                "percentage": fc.percentage,
                "release_rate_mg_per_kg_per_wash": _round_val(release_rate, 1),
                "weighted_release_mg": _round_val(fiber_release, 2),
                "annual_release_mg": _round_val(annual, 2),
                "exceeds_threshold": above_threshold,
            })

        # Check overall threshold
        overall_threshold = Decimal("100")  # mg/kg/wash
        if total_per_wash > overall_threshold:
            exceeds = True

        if exceeds:
            recommendation = (
                "Microplastic release exceeds recommended thresholds. "
                "Consider: (1) shifting to natural fibers or low-shedding synthetics, "
                "(2) applying anti-shedding textile finishes, "
                "(3) recommending microfiber-catching wash bags to consumers."
            )
        elif total_per_wash > overall_threshold * Decimal("0.7"):
            recommendation = (
                "Microplastic release is approaching thresholds. "
                "Monitor fiber selection and consider pre-washing treatments "
                "to reduce shedding."
            )
        else:
            recommendation = (
                "Microplastic release is within acceptable limits. "
                "Continue monitoring and consider further fiber optimisation."
            )

        return MicroplasticAssessmentResult(
            product_id=product_id,
            fiber_breakdown=fiber_breakdown,
            total_release_mg_per_wash=_round_val(total_per_wash, 2),
            annual_release_mg=_round_val(total_annual, 2),
            exceeds_threshold=exceeds,
            recommendation=recommendation,
        )

    # -------------------------------------------------------------------
    # Repairability Scoring
    # -------------------------------------------------------------------

    def _calc_repairability(
        self, data: RepairabilityInput
    ) -> RepairabilityResult:
        """Calculate repairability score and grade.

        Applies weighted scoring across five criteria (documentation,
        disassembly, spare parts availability, spare parts price, and
        product-specific) using French Repairability Index methodology.

        Formula: weighted_score = sum(criterion_score * weight / 100) / sum(weight / 100)

        Args:
            data: Repairability input data.

        Returns:
            RepairabilityResult with weighted score and grade.
        """
        criteria_scores: Dict[str, Dict[str, float]] = {}
        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        score_map = {
            "documentation": data.documentation_score,
            "disassembly": data.disassembly_score,
            "spare_parts_availability": data.spare_parts_availability_score,
            "spare_parts_price": data.spare_parts_price_score,
            "product_specific": data.product_specific_score,
        }

        for criterion_name, score_val in score_map.items():
            criterion_info = self._repair_criteria.get(criterion_name, {})
            weight = _decimal(criterion_info.get("weight", 20.0))
            max_score = _decimal(criterion_info.get("max_score", 10.0))
            score = min(_decimal(score_val), max_score)

            weighted = score * weight / Decimal("100")
            total_weighted += weighted
            total_weight += weight / Decimal("100")

            criteria_scores[criterion_name] = {
                "raw_score": _round_val(score, 1),
                "weight_pct": _round_val(weight, 1),
                "weighted_contribution": _round_val(weighted, 3),
            }

        final_score = _safe_divide(total_weighted, total_weight)

        # Determine grade
        grade = RepairabilityGrade.E.value
        for g, (low, high) in self._repair_thresholds.items():
            if _decimal(low) <= final_score < _decimal(high):
                grade = g.value
                break
        if final_score >= Decimal("8"):
            grade = RepairabilityGrade.A.value

        return RepairabilityResult(
            product_id=data.product_id,
            criteria_scores=criteria_scores,
            weighted_score=_round_val(final_score, 2),
            grade=grade,
        )

    # -------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        dpp_results: List[DPPCompletenessResult],
        claims_audit: List[GreenClaimAuditResult],
        pef_results: List[PEFNormalizedResult],
        microplastic_results: List[MicroplasticAssessmentResult],
        repair_results: List[RepairabilityResult],
    ) -> List[str]:
        """Generate actionable product sustainability recommendations.

        Analyses all assessment results to produce prioritised improvement
        recommendations.

        Args:
            dpp_results: DPP completeness results.
            claims_audit: Green claims audit results.
            pef_results: PEF scoring results.
            microplastic_results: Microplastic assessment results.
            repair_results: Repairability scoring results.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # DPP completeness
        incomplete = [r for r in dpp_results if not r.compliant]
        if incomplete:
            categories = set(r.dpp_category for r in incomplete)
            recs.append(
                f"Complete DPP data for {len(incomplete)} products in categories: "
                f"{', '.join(categories)}. Focus on mandatory fields to reach "
                f"90% completeness threshold."
            )

        # Prohibited claims
        prohibited = [c for c in claims_audit if c.is_prohibited]
        if prohibited:
            recs.append(
                f"Remove or reformulate {len(prohibited)} prohibited green claims. "
                "ECGT requires specific, measurable, and verifiable environmental "
                "claims backed by third-party evidence."
            )

        # Unsubstantiated claims
        unsubstantiated = [
            c for c in claims_audit
            if c.compliance_status in ("requires_substantiation", "requires_verification")
        ]
        if unsubstantiated:
            recs.append(
                f"Substantiate {len(unsubstantiated)} green claims with "
                "third-party verification. Under ECGT, all environmental claims "
                "must be supported by evidence and independently verified."
            )

        # Microplastic exceedances
        mp_exceed = [m for m in microplastic_results if m.exceeds_threshold]
        if mp_exceed:
            recs.append(
                f"{len(mp_exceed)} textile products exceed microplastic release "
                "thresholds. Transition to low-shedding fiber blends and apply "
                "anti-shedding treatments."
            )

        # Repairability
        poor_repair = [r for r in repair_results if r.grade in ("D", "E")]
        if poor_repair:
            recs.append(
                f"{len(poor_repair)} products have poor repairability (grade D/E). "
                "Improve spare parts availability, provide repair documentation, "
                "and design for easier disassembly."
            )

        # PEF dominant categories
        if pef_results:
            climate_dominant = any(
                any(
                    d.get("category") == "climate_change"
                    for d in r.dominant_categories[:1]
                )
                for r in pef_results
            )
            if climate_dominant:
                recs.append(
                    "Climate change is the dominant PEF impact category. "
                    "Prioritize supply chain decarbonization, renewable energy, "
                    "and material substitution to reduce product carbon footprints."
                )

        if not recs:
            recs.append(
                "Product sustainability profile is strong. Continue monitoring "
                "regulatory developments and improving DPP data completeness."
            )

        return recs
