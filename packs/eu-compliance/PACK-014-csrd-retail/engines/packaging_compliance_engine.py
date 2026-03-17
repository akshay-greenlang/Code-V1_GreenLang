# -*- coding: utf-8 -*-
"""
PackagingComplianceEngine - PACK-014 CSRD Retail Engine 3
==========================================================

PPWR (Packaging and Packaging Waste Regulation) 2025/40 compliance engine.
Covers recycled content targets, EPR eco-modulation fees, labeling
requirements, reuse targets, and portfolio-level compliance assessment.

Regulatory References:
    - Regulation (EU) 2025/40 (PPWR) -- replacing Directive 94/62/EC
    - PPWR Annex II: Recycled content targets by polymer type
    - PPWR Annex V: Reuse targets by packaging format
    - PPWR Article 11: Labeling and marking requirements
    - EU EPR eco-modulation guidelines (2024)
    - EN 13432:2000 for compostable packaging

Key Compliance Areas:
    - Recycled content minimums (2030 and 2040 targets)
    - EPR fee eco-modulation (recyclability grade A-E)
    - Mandatory labeling (material composition, sorting instructions)
    - Reuse and refill targets (transport, e-commerce, grouped)
    - Recyclability-by-design requirements
    - Substance restrictions (PFAS, heavy metals)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Targets and thresholds from PPWR legal text (hard-coded)
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

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class PackagingMaterial(str, Enum):
    """Packaging material types."""
    PET = "PET"
    HDPE = "HDPE"
    PP = "PP"
    PS = "PS"
    PVC = "PVC"
    GLASS = "glass"
    ALUMINIUM = "aluminium"
    STEEL = "steel"
    PAPER_BOARD = "paper_board"
    WOOD = "wood"
    COMPOSITE = "composite"
    BIOPLASTIC = "bioplastic"


class PackagingType(str, Enum):
    """Packaging function/level classification per PPWR."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    TRANSPORT = "transport"
    E_COMMERCE = "e_commerce"


class EPRGrade(str, Enum):
    """Eco-modulation recyclability grade (A = best, E = worst)."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class LabelingStatus(str, Enum):
    """Labeling compliance status."""
    COMPLIANT = "compliant"
    PENDING = "pending"
    NON_COMPLIANT = "non_compliant"
    EXEMPT = "exempt"


# ---------------------------------------------------------------------------
# Constants -- PPWR Targets and Factors
# ---------------------------------------------------------------------------

# PPWR Recycled content targets by polymer type (% by weight)
# Source: PPWR Annex II, Article 7
PPWR_RECYCLED_CONTENT_TARGETS: Dict[str, Dict[int, float]] = {
    # Contact-sensitive PET (food/beverage)
    "PET_contact": {2030: 30.0, 2040: 50.0},
    # Non-contact-sensitive PET
    "PET_non_contact": {2030: 30.0, 2040: 50.0},
    # Other contact-sensitive plastics
    "other_plastic_contact": {2030: 10.0, 2040: 50.0},
    # Other non-contact-sensitive plastics
    "other_plastic_non_contact": {2030: 10.0, 2040: 65.0},
    # HDPE, PP, PS, PVC -- non-contact
    "HDPE": {2030: 10.0, 2040: 65.0},
    "PP": {2030: 10.0, 2040: 65.0},
    "PS": {2030: 10.0, 2040: 65.0},
    "PVC": {2030: 10.0, 2040: 65.0},
    # PET (general default)
    "PET": {2030: 30.0, 2040: 50.0},
}

# PPWR Reuse targets by packaging format (% of units placed on market)
# Source: PPWR Annex V
PPWR_REUSE_TARGETS: Dict[str, Dict[int, float]] = {
    "e_commerce": {2030: 10.0, 2040: 50.0},
    "transport_packaging": {2030: 40.0, 2040: 70.0},
    "grouped_packaging": {2030: 10.0, 2040: 25.0},
    "beverage_alcoholic": {2030: 10.0, 2040: 25.0},
    "beverage_non_alcoholic": {2030: 10.0, 2040: 25.0},
    "household_appliance": {2030: 0.0, 2040: 10.0},
}

# EPR eco-modulation multipliers by recyclability grade
# Source: EU EPR eco-modulation guidelines, typical national implementations
EPR_GRADE_MULTIPLIERS: Dict[str, float] = {
    EPRGrade.A: 0.50,   # Fully recyclable, mono-material
    EPRGrade.B: 0.80,   # Recyclable with minor limitations
    EPRGrade.C: 1.00,   # Baseline -- partially recyclable
    EPRGrade.D: 1.50,   # Difficult to recycle
    EPRGrade.E: 2.00,   # Not recyclable
}

# Public alias for backward-compatible imports
EPR_MODULATION_CRITERIA = EPR_GRADE_MULTIPLIERS

# EPR base rates by material (EUR per tonne)
# Source: Typical EU EPR scheme rates (FR/DE/NL average)
EPR_BASE_RATES: Dict[str, float] = {
    PackagingMaterial.PET: 380.0,
    PackagingMaterial.HDPE: 420.0,
    PackagingMaterial.PP: 410.0,
    PackagingMaterial.PS: 480.0,
    PackagingMaterial.PVC: 520.0,
    PackagingMaterial.GLASS: 65.0,
    PackagingMaterial.ALUMINIUM: 180.0,
    PackagingMaterial.STEEL: 120.0,
    PackagingMaterial.PAPER_BOARD: 95.0,
    PackagingMaterial.WOOD: 45.0,
    PackagingMaterial.COMPOSITE: 550.0,
    PackagingMaterial.BIOPLASTIC: 350.0,
}

# Recycled content bonus: discount on EPR fee per 10% recycled content
RECYCLED_CONTENT_BONUS_PER_10PCT: float = 0.03  # 3% discount per 10% RC

# Compostable packaging bonus multiplier (applied if certified EN 13432)
COMPOSTABLE_BONUS_MULTIPLIER: float = 0.85  # 15% discount

# Labeling requirement deadlines
# Source: PPWR Article 11
LABELING_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "material_composition": {
        "description": "Material identification marking on all packaging",
        "deadline": "2026-08-01",
        "mandatory": True,
        "exemptions": ["very_small_packaging"],
    },
    "sorting_instructions": {
        "description": "Consumer sorting/disposal instructions",
        "deadline": "2026-08-01",
        "mandatory": True,
        "exemptions": ["transport_packaging"],
    },
    "pictograms": {
        "description": "Harmonised EU pictograms for waste sorting",
        "deadline": "2028-08-01",
        "mandatory": True,
        "exemptions": [],
    },
    "digital_watermark": {
        "description": "Digital watermark for automated sorting",
        "deadline": "2030-01-01",
        "mandatory": False,
        "exemptions": [],
    },
    "reusable_marking": {
        "description": "Reusable packaging marking and QR code",
        "deadline": "2026-08-01",
        "mandatory": True,
        "exemptions": ["single_use_only"],
    },
}

# PPWR substance restrictions thresholds
SUBSTANCE_RESTRICTIONS: Dict[str, Dict[str, Any]] = {
    "heavy_metals": {
        "limit_ppm": 100,
        "description": "Sum of Pb, Cd, Hg, Cr(VI) concentration",
        "regulation": "PPWR Article 5",
    },
    "pfas": {
        "limit_ppm": 25,
        "description": "Total PFAS in food-contact packaging",
        "regulation": "PPWR Article 5(4)",
        "effective_date": "2026-01-01",
    },
    "bisphenol_a": {
        "limit_ppm": 0,
        "description": "BPA ban in food-contact packaging",
        "regulation": "PPWR Article 5(5)",
        "effective_date": "2027-01-01",
    },
}

# Packaging carbon footprint factors (kgCO2e per kg of material, virgin)
PACKAGING_CARBON_FACTORS: Dict[str, float] = {
    PackagingMaterial.PET: 3.14,
    PackagingMaterial.HDPE: 2.52,
    PackagingMaterial.PP: 2.34,
    PackagingMaterial.PS: 3.48,
    PackagingMaterial.PVC: 2.41,
    PackagingMaterial.GLASS: 0.86,
    PackagingMaterial.ALUMINIUM: 8.14,
    PackagingMaterial.STEEL: 1.89,
    PackagingMaterial.PAPER_BOARD: 1.29,
    PackagingMaterial.WOOD: 0.31,
    PackagingMaterial.COMPOSITE: 3.80,
    PackagingMaterial.BIOPLASTIC: 2.10,
}

# Recycled content carbon reduction factor (% reduction per % recycled)
# Recycled material typically has 30-70% lower carbon footprint
RECYCLED_CARBON_REDUCTION: Dict[str, float] = {
    PackagingMaterial.PET: 0.60,        # 60% reduction for rPET
    PackagingMaterial.HDPE: 0.55,
    PackagingMaterial.PP: 0.50,
    PackagingMaterial.PS: 0.45,
    PackagingMaterial.PVC: 0.40,
    PackagingMaterial.GLASS: 0.25,
    PackagingMaterial.ALUMINIUM: 0.92,  # Recycled aluminium saves 92%
    PackagingMaterial.STEEL: 0.70,
    PackagingMaterial.PAPER_BOARD: 0.35,
    PackagingMaterial.WOOD: 0.20,
    PackagingMaterial.COMPOSITE: 0.30,
    PackagingMaterial.BIOPLASTIC: 0.25,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class PackagingItem(BaseModel):
    """Individual packaging item with material and compliance data.

    Attributes:
        item_id: Unique item identifier.
        item_name: Human-readable name.
        material: Primary packaging material.
        packaging_type: Packaging function/level.
        weight_grams: Weight per unit in grams.
        units_placed: Number of units placed on market per year.
        recycled_content_pct: Percentage of recycled content by weight.
        is_contact_sensitive: Whether used for food/beverage contact.
        recyclability_grade: EPR recyclability grade (A-E).
        compostable: Whether certified compostable (EN 13432).
        reusable: Whether designed for reuse.
        reuse_cycles: Number of reuse cycles if reusable.
        has_material_marking: Has material composition marking.
        has_sorting_instructions: Has consumer sorting instructions.
        has_pictograms: Has harmonised EU pictograms.
        has_digital_watermark: Has digital watermark.
        has_reusable_marking: Has reusable packaging marking.
        heavy_metals_ppm: Heavy metals concentration in ppm.
        pfas_ppm: PFAS concentration in ppm.
        contains_bpa: Whether packaging contains BPA.
    """
    item_id: str = Field(..., min_length=1, description="Item identifier")
    item_name: str = Field("", description="Item name")
    material: PackagingMaterial
    packaging_type: PackagingType
    weight_grams: float = Field(..., gt=0, description="Weight per unit (g)")
    units_placed: int = Field(1, ge=0, description="Units placed on market/year")
    recycled_content_pct: float = Field(
        0.0, ge=0, le=100, description="Recycled content (%)"
    )
    is_contact_sensitive: bool = Field(
        False, description="Food/beverage contact"
    )
    recyclability_grade: EPRGrade = Field(
        EPRGrade.C, description="Recyclability grade"
    )
    compostable: bool = Field(False, description="EN 13432 certified")
    reusable: bool = Field(False, description="Designed for reuse")
    reuse_cycles: int = Field(0, ge=0, description="Reuse cycles")
    has_material_marking: bool = Field(False, description="Material marking")
    has_sorting_instructions: bool = Field(False, description="Sorting instructions")
    has_pictograms: bool = Field(False, description="EU pictograms")
    has_digital_watermark: bool = Field(False, description="Digital watermark")
    has_reusable_marking: bool = Field(False, description="Reusable marking")
    heavy_metals_ppm: Optional[float] = Field(None, ge=0, description="Heavy metals (ppm)")
    pfas_ppm: Optional[float] = Field(None, ge=0, description="PFAS (ppm)")
    contains_bpa: bool = Field(False, description="Contains BPA")


class PackagingPortfolio(BaseModel):
    """Complete packaging portfolio for compliance assessment.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Assessment year.
        items: List of packaging items.
        country: Primary market country (for EPR rates).
    """
    organisation_id: str = Field(..., min_length=1, description="Organisation ID")
    reporting_year: int = Field(..., ge=2024, le=2050, description="Reporting year")
    items: List[PackagingItem] = Field(..., min_length=1, description="Packaging items")
    country: str = Field("EU", description="Primary market country")


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class RecycledContentAssessment(BaseModel):
    """Recycled content assessment per material.

    Attributes:
        material: Packaging material.
        total_weight_kg: Total weight of this material (kg).
        weighted_recycled_pct: Weight-averaged recycled content.
        target_2030_pct: 2030 PPWR target.
        target_2040_pct: 2040 PPWR target.
        gap_to_2030_pct: Gap to 2030 target (negative = compliant).
        gap_to_2040_pct: Gap to 2040 target (negative = compliant).
        compliant_2030: Whether on track for 2030.
        compliant_2040: Whether on track for 2040.
        item_count: Number of items of this material.
    """
    material: str
    total_weight_kg: float
    weighted_recycled_pct: float
    target_2030_pct: float
    target_2040_pct: float
    gap_to_2030_pct: float
    gap_to_2040_pct: float
    compliant_2030: bool
    compliant_2040: bool
    item_count: int


class EPRFeeDetail(BaseModel):
    """EPR fee calculation detail per material.

    Attributes:
        material: Packaging material.
        weight_tonnes: Weight in tonnes.
        base_rate_per_tonne: Base EPR rate (EUR/tonne).
        grade_distribution: Count of items by grade.
        weighted_multiplier: Weight-averaged grade multiplier.
        recycled_content_discount: Discount from recycled content.
        compostable_discount: Discount from compostable certification.
        gross_fee_eur: Fee before discounts.
        net_fee_eur: Fee after discounts.
    """
    material: str
    weight_tonnes: float
    base_rate_per_tonne: float
    grade_distribution: Dict[str, int]
    weighted_multiplier: float
    recycled_content_discount: float
    compostable_discount: float
    gross_fee_eur: float
    net_fee_eur: float


class LabelingComplianceDetail(BaseModel):
    """Labeling compliance assessment per requirement.

    Attributes:
        requirement: Labeling requirement name.
        description: Requirement description.
        deadline: Compliance deadline.
        mandatory: Whether mandatory.
        total_items: Total items assessed.
        compliant_items: Items meeting requirement.
        compliance_pct: Compliance percentage.
        status: Overall compliance status.
    """
    requirement: str
    description: str
    deadline: str
    mandatory: bool
    total_items: int
    compliant_items: int
    compliance_pct: float
    status: str


class ReuseProgressDetail(BaseModel):
    """Reuse target progress assessment.

    Attributes:
        packaging_format: Packaging format category.
        total_units: Total units placed on market.
        reusable_units: Units that are reusable.
        reuse_pct: Current reuse percentage.
        target_2030_pct: 2030 target.
        target_2040_pct: 2040 target.
        gap_to_2030_pct: Gap to 2030 (negative = compliant).
        on_track: Whether currently on track.
    """
    packaging_format: str
    total_units: int
    reusable_units: int
    reuse_pct: float
    target_2030_pct: float
    target_2040_pct: float
    gap_to_2030_pct: float
    on_track: bool


class SubstanceComplianceDetail(BaseModel):
    """Substance restriction compliance assessment.

    Attributes:
        substance: Restricted substance name.
        limit_ppm: Regulatory limit (ppm).
        items_tested: Number of items with test data.
        items_compliant: Number compliant.
        items_non_compliant: Number non-compliant.
        items_unknown: Number without data.
        compliant: Overall compliance status.
    """
    substance: str
    limit_ppm: float
    items_tested: int
    items_compliant: int
    items_non_compliant: int
    items_unknown: int
    compliant: bool


class CarbonFootprintSummary(BaseModel):
    """Packaging carbon footprint summary.

    Attributes:
        total_virgin_footprint_tco2e: Carbon footprint if all virgin.
        total_actual_footprint_tco2e: Carbon footprint with recycled content.
        avoided_emissions_tco2e: Emissions avoided by recycled content.
        reduction_pct: Percentage reduction from recycled content.
    """
    total_virgin_footprint_tco2e: float
    total_actual_footprint_tco2e: float
    avoided_emissions_tco2e: float
    reduction_pct: float


class PPWRComplianceResult(BaseModel):
    """Complete PPWR compliance assessment result.

    Attributes:
        organisation_id: Organisation identifier.
        reporting_year: Assessment year.
        total_items: Total packaging items assessed.
        total_weight_tonnes: Total packaging weight.
        total_units: Total units placed on market.
        recycled_content_by_material: Recycled content per material.
        overall_recycled_content_pct: Portfolio-wide recycled content.
        epr_fee_details: EPR fee breakdown per material.
        total_epr_fee_eur: Total EPR fee.
        labeling_compliance: Labeling compliance per requirement.
        reuse_progress: Reuse target progress.
        substance_compliance: Substance restriction compliance.
        carbon_footprint: Packaging carbon footprint.
        epr_grade_distribution: Overall grade distribution.
        overall_compliance_score: Composite compliance score (0-100).
        recommendations: Improvement recommendations.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    organisation_id: str
    reporting_year: int
    total_items: int
    total_weight_tonnes: float
    total_units: int
    recycled_content_by_material: List[RecycledContentAssessment]
    overall_recycled_content_pct: float
    epr_fee_details: List[EPRFeeDetail]
    total_epr_fee_eur: float
    labeling_compliance: List[LabelingComplianceDetail]
    reuse_progress: List[ReuseProgressDetail]
    substance_compliance: List[SubstanceComplianceDetail]
    carbon_footprint: CarbonFootprintSummary
    epr_grade_distribution: Dict[str, int]
    overall_compliance_score: float
    recommendations: List[str]
    engine_version: str = engine_version
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class PackagingComplianceEngine:
    """PPWR packaging compliance calculation engine.

    Assesses a packaging portfolio against PPWR 2025/40 requirements
    including recycled content targets, EPR fees, labeling, reuse targets,
    and substance restrictions.

    Guarantees:
        - Deterministic: identical inputs always produce identical outputs.
        - Reproducible: full provenance via SHA-256 hashing.
        - Auditable: every assessment is documented.
        - Zero-hallucination: no LLM in the calculation path.

    Usage::

        engine = PackagingComplianceEngine()
        result = engine.assess_compliance(portfolio)
    """

    def __init__(self) -> None:
        """Initialise engine with embedded PPWR constants."""
        self._rc_targets = PPWR_RECYCLED_CONTENT_TARGETS
        self._reuse_targets = PPWR_REUSE_TARGETS
        self._epr_multipliers = EPR_GRADE_MULTIPLIERS
        self._epr_base_rates = EPR_BASE_RATES
        self._labeling_reqs = LABELING_REQUIREMENTS
        self._substance_limits = SUBSTANCE_RESTRICTIONS
        self._carbon_factors = PACKAGING_CARBON_FACTORS
        self._recycled_reduction = RECYCLED_CARBON_REDUCTION

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assess_compliance(
        self, portfolio: PackagingPortfolio
    ) -> PPWRComplianceResult:
        """Assess a packaging portfolio for PPWR compliance.

        Evaluates all PPWR compliance dimensions: recycled content, EPR
        fees, labeling, reuse targets, and substance restrictions.

        Args:
            portfolio: Complete packaging portfolio to assess.

        Returns:
            PPWRComplianceResult with detailed compliance assessment.
        """
        t0 = time.perf_counter()

        items = portfolio.items
        year = portfolio.reporting_year

        # --- Compute totals ---
        total_weight_tonnes = Decimal("0")
        total_units = 0
        for item in items:
            w_kg = _decimal(item.weight_grams) / Decimal("1000")
            w_tonnes = w_kg * _decimal(item.units_placed) / Decimal("1000")
            total_weight_tonnes += w_tonnes
            total_units += item.units_placed

        # --- Recycled content assessment ---
        rc_assessment = self._assess_recycled_content(items, year)
        overall_rc = self._calc_portfolio_recycled_content(items)

        # --- EPR fees ---
        epr_details = self._calc_epr_fees(items)
        total_epr = sum(_decimal(d.net_fee_eur) for d in epr_details)

        # --- Labeling compliance ---
        labeling = self._assess_labeling(items, year)

        # --- Reuse progress ---
        reuse = self._assess_reuse(items)

        # --- Substance compliance ---
        substances = self._assess_substances(items)

        # --- Carbon footprint ---
        carbon = self._calc_carbon_footprint(items)

        # --- Grade distribution ---
        grade_dist: Dict[str, int] = defaultdict(int)
        for item in items:
            grade_dist[item.recyclability_grade.value] += 1

        # --- Overall compliance score ---
        compliance_score = self._calc_compliance_score(
            rc_assessment, labeling, reuse, substances, grade_dist
        )

        # --- Recommendations ---
        recommendations = self._generate_recommendations(
            rc_assessment, labeling, reuse, substances, grade_dist, year
        )

        processing_ms = (time.perf_counter() - t0) * 1000.0

        result = PPWRComplianceResult(
            organisation_id=portfolio.organisation_id,
            reporting_year=year,
            total_items=len(items),
            total_weight_tonnes=_round_val(total_weight_tonnes, 4),
            total_units=total_units,
            recycled_content_by_material=rc_assessment,
            overall_recycled_content_pct=_round_val(overall_rc, 2),
            epr_fee_details=epr_details,
            total_epr_fee_eur=_round_val(total_epr, 2),
            labeling_compliance=labeling,
            reuse_progress=reuse,
            substance_compliance=substances,
            carbon_footprint=carbon,
            epr_grade_distribution=dict(grade_dist),
            overall_compliance_score=_round_val(_decimal(compliance_score), 1),
            recommendations=recommendations,
            processing_time_ms=round(processing_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Internal assessment methods
    # -------------------------------------------------------------------

    def _assess_recycled_content(
        self, items: List[PackagingItem], year: int
    ) -> List[RecycledContentAssessment]:
        """Assess recycled content compliance per material.

        Calculates weight-averaged recycled content and compares against
        PPWR 2030 and 2040 targets by polymer type.

        Args:
            items: List of packaging items.
            year: Reporting year.

        Returns:
            List of RecycledContentAssessment per material.
        """
        material_data: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {
                "total_weight": Decimal("0"),
                "rc_weight": Decimal("0"),
                "count": Decimal("0"),
                "contact_weight": Decimal("0"),
                "contact_rc_weight": Decimal("0"),
            }
        )

        for item in items:
            unit_weight_kg = _decimal(item.weight_grams) / Decimal("1000")
            total_kg = unit_weight_kg * _decimal(item.units_placed)
            rc_kg = total_kg * _decimal(item.recycled_content_pct) / Decimal("100")

            mat = item.material.value
            material_data[mat]["total_weight"] += total_kg
            material_data[mat]["rc_weight"] += rc_kg
            material_data[mat]["count"] += Decimal("1")

            if item.is_contact_sensitive:
                material_data[mat]["contact_weight"] += total_kg
                material_data[mat]["contact_rc_weight"] += rc_kg

        results: List[RecycledContentAssessment] = []
        for mat, data in material_data.items():
            weighted_rc = _safe_pct(data["rc_weight"], data["total_weight"])

            # Determine target key
            target_key = self._get_rc_target_key(mat, data)
            targets = self._rc_targets.get(target_key, {})
            target_2030 = _decimal(targets.get(2030, 0.0))
            target_2040 = _decimal(targets.get(2040, 0.0))

            gap_2030 = target_2030 - weighted_rc
            gap_2040 = target_2040 - weighted_rc

            results.append(
                RecycledContentAssessment(
                    material=mat,
                    total_weight_kg=_round_val(data["total_weight"], 2),
                    weighted_recycled_pct=_round_val(weighted_rc, 2),
                    target_2030_pct=_round_val(target_2030, 1),
                    target_2040_pct=_round_val(target_2040, 1),
                    gap_to_2030_pct=_round_val(gap_2030, 2),
                    gap_to_2040_pct=_round_val(gap_2040, 2),
                    compliant_2030=gap_2030 <= Decimal("0"),
                    compliant_2040=gap_2040 <= Decimal("0"),
                    item_count=int(data["count"]),
                )
            )

        return results

    def _get_rc_target_key(
        self, material: str, data: Dict[str, Decimal]
    ) -> str:
        """Determine the recycled content target key based on material.

        Maps packaging material to the appropriate PPWR target category,
        considering contact sensitivity.

        Args:
            material: Material identifier.
            data: Material aggregate data.

        Returns:
            Target key for PPWR_RECYCLED_CONTENT_TARGETS lookup.
        """
        is_plastic = material in {"PET", "HDPE", "PP", "PS", "PVC"}
        has_contact = data["contact_weight"] > Decimal("0")

        if material == "PET":
            return "PET_contact" if has_contact else "PET_non_contact"
        elif is_plastic:
            return material if material in self._rc_targets else (
                "other_plastic_contact" if has_contact
                else "other_plastic_non_contact"
            )
        return "other_plastic_non_contact"

    def _calc_portfolio_recycled_content(
        self, items: List[PackagingItem]
    ) -> Decimal:
        """Calculate portfolio-wide weight-averaged recycled content.

        Args:
            items: List of packaging items.

        Returns:
            Overall recycled content percentage as Decimal.
        """
        total_weight = Decimal("0")
        rc_weight = Decimal("0")

        for item in items:
            unit_weight = _decimal(item.weight_grams) / Decimal("1000")
            item_weight = unit_weight * _decimal(item.units_placed)
            total_weight += item_weight
            rc_weight += item_weight * _decimal(item.recycled_content_pct) / Decimal("100")

        return _safe_pct(rc_weight, total_weight)

    def _calc_epr_fees(
        self, items: List[PackagingItem]
    ) -> List[EPRFeeDetail]:
        """Calculate EPR fees with eco-modulation per material.

        Applies base rates, grade multipliers, recycled content discounts,
        and compostable bonuses.

        Formula: fee = weight * base_rate * grade_multiplier * (1 - RC_discount) * compost_bonus

        Args:
            items: List of packaging items.

        Returns:
            List of EPRFeeDetail per material.
        """
        material_groups: Dict[str, List[PackagingItem]] = defaultdict(list)
        for item in items:
            material_groups[item.material.value].append(item)

        results: List[EPRFeeDetail] = []
        for mat, mat_items in material_groups.items():
            mat_enum = PackagingMaterial(mat) if mat in [m.value for m in PackagingMaterial] else None
            base_rate = _decimal(
                self._epr_base_rates.get(mat_enum, 500.0) if mat_enum else 500.0
            )

            total_weight_tonnes = Decimal("0")
            grade_dist: Dict[str, int] = defaultdict(int)
            weighted_mult = Decimal("0")
            weighted_rc = Decimal("0")
            total_item_weight = Decimal("0")
            compostable_weight = Decimal("0")

            for item in mat_items:
                unit_kg = _decimal(item.weight_grams) / Decimal("1000")
                item_tonnes = unit_kg * _decimal(item.units_placed) / Decimal("1000")
                total_weight_tonnes += item_tonnes
                total_item_weight += item_tonnes

                grade_dist[item.recyclability_grade.value] += 1
                mult = _decimal(
                    self._epr_multipliers.get(item.recyclability_grade, 1.0)
                )
                weighted_mult += item_tonnes * mult
                weighted_rc += item_tonnes * _decimal(item.recycled_content_pct)

                if item.compostable:
                    compostable_weight += item_tonnes

            avg_mult = _safe_divide(weighted_mult, total_item_weight, Decimal("1"))
            avg_rc = _safe_divide(weighted_rc, total_item_weight)

            # RC discount: 3% per 10% recycled content
            rc_discount_pct = (avg_rc / Decimal("10")) * _decimal(RECYCLED_CONTENT_BONUS_PER_10PCT)
            rc_discount = min(rc_discount_pct, Decimal("0.30"))  # Cap at 30%

            # Compostable bonus
            compost_frac = _safe_divide(compostable_weight, total_item_weight)
            compost_discount = Decimal("0")
            if compost_frac > Decimal("0"):
                compost_discount = compost_frac * (
                    Decimal("1") - _decimal(COMPOSTABLE_BONUS_MULTIPLIER)
                )

            gross_fee = total_weight_tonnes * base_rate * avg_mult
            net_fee = gross_fee * (Decimal("1") - rc_discount) * (
                Decimal("1") - compost_discount
            )

            results.append(
                EPRFeeDetail(
                    material=mat,
                    weight_tonnes=_round_val(total_weight_tonnes, 4),
                    base_rate_per_tonne=_round_val(base_rate, 2),
                    grade_distribution=dict(grade_dist),
                    weighted_multiplier=_round_val(avg_mult, 3),
                    recycled_content_discount=_round_val(rc_discount * Decimal("100"), 2),
                    compostable_discount=_round_val(compost_discount * Decimal("100"), 2),
                    gross_fee_eur=_round_val(gross_fee, 2),
                    net_fee_eur=_round_val(net_fee, 2),
                )
            )

        return results

    def _assess_labeling(
        self, items: List[PackagingItem], year: int
    ) -> List[LabelingComplianceDetail]:
        """Assess labeling compliance for each PPWR requirement.

        Checks each packaging item against mandatory labeling requirements
        with deadline awareness.

        Args:
            items: List of packaging items.
            year: Reporting year.

        Returns:
            List of LabelingComplianceDetail per requirement.
        """
        results: List[LabelingComplianceDetail] = []

        field_map = {
            "material_composition": "has_material_marking",
            "sorting_instructions": "has_sorting_instructions",
            "pictograms": "has_pictograms",
            "digital_watermark": "has_digital_watermark",
            "reusable_marking": "has_reusable_marking",
        }

        for req_name, req_info in self._labeling_reqs.items():
            attr_name = field_map.get(req_name)
            if not attr_name:
                continue

            deadline_year = int(req_info["deadline"][:4])
            applicable_items = items
            if req_name == "reusable_marking":
                applicable_items = [i for i in items if i.reusable]

            total = len(applicable_items)
            compliant_count = sum(
                1 for i in applicable_items if getattr(i, attr_name, False)
            )
            pct = float(_safe_pct(_decimal(compliant_count), _decimal(total)))

            if not req_info["mandatory"]:
                status = "optional"
            elif year < deadline_year:
                status = "pending" if pct < 100.0 else "compliant"
            elif pct >= 100.0:
                status = "compliant"
            elif pct >= 80.0:
                status = "partially_compliant"
            else:
                status = "non_compliant"

            results.append(
                LabelingComplianceDetail(
                    requirement=req_name,
                    description=req_info["description"],
                    deadline=req_info["deadline"],
                    mandatory=req_info["mandatory"],
                    total_items=total,
                    compliant_items=compliant_count,
                    compliance_pct=round(pct, 1),
                    status=status,
                )
            )

        return results

    def _assess_reuse(
        self, items: List[PackagingItem]
    ) -> List[ReuseProgressDetail]:
        """Assess progress toward PPWR reuse targets.

        Groups packaging items by format and calculates reuse percentage
        against PPWR Annex V targets.

        Args:
            items: List of packaging items.

        Returns:
            List of ReuseProgressDetail per format.
        """
        format_map = {
            PackagingType.E_COMMERCE: "e_commerce",
            PackagingType.TRANSPORT: "transport_packaging",
            PackagingType.SECONDARY: "grouped_packaging",
        }

        format_data: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "reusable": 0}
        )

        for item in items:
            format_key = format_map.get(item.packaging_type)
            if format_key:
                format_data[format_key]["total"] += item.units_placed
                if item.reusable:
                    format_data[format_key]["reusable"] += item.units_placed

        results: List[ReuseProgressDetail] = []
        for fmt, data in format_data.items():
            total = data["total"]
            reusable = data["reusable"]
            pct = float(_safe_pct(_decimal(reusable), _decimal(total)))

            targets = self._reuse_targets.get(fmt, {})
            t2030 = targets.get(2030, 0.0)
            t2040 = targets.get(2040, 0.0)
            gap = t2030 - pct

            results.append(
                ReuseProgressDetail(
                    packaging_format=fmt,
                    total_units=total,
                    reusable_units=reusable,
                    reuse_pct=round(pct, 2),
                    target_2030_pct=t2030,
                    target_2040_pct=t2040,
                    gap_to_2030_pct=round(gap, 2),
                    on_track=gap <= 0,
                )
            )

        return results

    def _assess_substances(
        self, items: List[PackagingItem]
    ) -> List[SubstanceComplianceDetail]:
        """Assess compliance with PPWR substance restrictions.

        Checks heavy metals, PFAS, and BPA against regulatory limits.

        Args:
            items: List of packaging items.

        Returns:
            List of SubstanceComplianceDetail per substance.
        """
        results: List[SubstanceComplianceDetail] = []

        # Heavy metals
        hm_tested = 0
        hm_compliant = 0
        hm_non_compliant = 0
        hm_unknown = 0
        hm_limit = self._substance_limits["heavy_metals"]["limit_ppm"]

        for item in items:
            if item.heavy_metals_ppm is not None:
                hm_tested += 1
                if item.heavy_metals_ppm <= hm_limit:
                    hm_compliant += 1
                else:
                    hm_non_compliant += 1
            else:
                hm_unknown += 1

        results.append(
            SubstanceComplianceDetail(
                substance="heavy_metals",
                limit_ppm=float(hm_limit),
                items_tested=hm_tested,
                items_compliant=hm_compliant,
                items_non_compliant=hm_non_compliant,
                items_unknown=hm_unknown,
                compliant=hm_non_compliant == 0,
            )
        )

        # PFAS (food-contact only)
        pfas_limit = self._substance_limits["pfas"]["limit_ppm"]
        contact_items = [i for i in items if i.is_contact_sensitive]
        pf_tested = 0
        pf_compliant = 0
        pf_non_compliant = 0
        pf_unknown = 0

        for item in contact_items:
            if item.pfas_ppm is not None:
                pf_tested += 1
                if item.pfas_ppm <= pfas_limit:
                    pf_compliant += 1
                else:
                    pf_non_compliant += 1
            else:
                pf_unknown += 1

        results.append(
            SubstanceComplianceDetail(
                substance="pfas",
                limit_ppm=float(pfas_limit),
                items_tested=pf_tested,
                items_compliant=pf_compliant,
                items_non_compliant=pf_non_compliant,
                items_unknown=pf_unknown,
                compliant=pf_non_compliant == 0,
            )
        )

        # BPA (food-contact only)
        bpa_tested = 0
        bpa_compliant = 0
        bpa_non_compliant = 0
        bpa_unknown = 0

        for item in contact_items:
            bpa_tested += 1
            if not item.contains_bpa:
                bpa_compliant += 1
            else:
                bpa_non_compliant += 1

        results.append(
            SubstanceComplianceDetail(
                substance="bisphenol_a",
                limit_ppm=0.0,
                items_tested=bpa_tested,
                items_compliant=bpa_compliant,
                items_non_compliant=bpa_non_compliant,
                items_unknown=bpa_unknown,
                compliant=bpa_non_compliant == 0,
            )
        )

        return results

    def _calc_carbon_footprint(
        self, items: List[PackagingItem]
    ) -> CarbonFootprintSummary:
        """Calculate packaging carbon footprint with recycled content savings.

        Computes virgin-baseline footprint, actual footprint with recycled
        content, and avoided emissions.

        Args:
            items: List of packaging items.

        Returns:
            CarbonFootprintSummary with footprint comparison.
        """
        total_virgin = Decimal("0")
        total_actual = Decimal("0")

        for item in items:
            weight_kg = _decimal(item.weight_grams) / Decimal("1000") * _decimal(item.units_placed)
            mat_enum = item.material

            virgin_ef = _decimal(
                self._carbon_factors.get(mat_enum, 2.0)
            )
            virgin_footprint = weight_kg * virgin_ef / Decimal("1000")  # tCO2e
            total_virgin += virgin_footprint

            # Recycled content reduces footprint
            rc_frac = _decimal(item.recycled_content_pct) / Decimal("100")
            reduction = _decimal(
                self._recycled_reduction.get(mat_enum, 0.30)
            )
            actual_footprint = virgin_footprint * (
                Decimal("1") - rc_frac * reduction
            )
            total_actual += actual_footprint

        avoided = total_virgin - total_actual
        reduction_pct = _safe_pct(avoided, total_virgin)

        return CarbonFootprintSummary(
            total_virgin_footprint_tco2e=_round_val(total_virgin, 6),
            total_actual_footprint_tco2e=_round_val(total_actual, 6),
            avoided_emissions_tco2e=_round_val(avoided, 6),
            reduction_pct=_round_val(reduction_pct, 2),
        )

    def _calc_compliance_score(
        self,
        rc_assessment: List[RecycledContentAssessment],
        labeling: List[LabelingComplianceDetail],
        reuse: List[ReuseProgressDetail],
        substances: List[SubstanceComplianceDetail],
        grade_dist: Dict[str, int],
    ) -> float:
        """Calculate composite compliance score (0-100).

        Weighted scoring:
        - Recycled content: 25%
        - EPR grade: 20%
        - Labeling: 20%
        - Reuse: 15%
        - Substances: 20%

        Args:
            rc_assessment: Recycled content results.
            labeling: Labeling compliance results.
            reuse: Reuse progress results.
            substances: Substance compliance results.
            grade_dist: EPR grade distribution.

        Returns:
            Composite score as float (0-100).
        """
        # RC score: % of materials meeting 2030 targets
        rc_compliant = sum(1 for r in rc_assessment if r.compliant_2030)
        rc_total = max(len(rc_assessment), 1)
        rc_score = _decimal(rc_compliant) / _decimal(rc_total) * Decimal("100")

        # Grade score: weighted by A=100, B=80, C=60, D=30, E=0
        grade_scores = {"A": 100, "B": 80, "C": 60, "D": 30, "E": 0}
        total_items = sum(grade_dist.values()) or 1
        grade_score_sum = sum(
            grade_scores.get(g, 0) * count for g, count in grade_dist.items()
        )
        grade_score = _decimal(grade_score_sum) / _decimal(total_items)

        # Labeling score: average compliance across mandatory requirements
        mandatory_labels = [l for l in labeling if l.mandatory]
        if mandatory_labels:
            label_score = _decimal(
                sum(l.compliance_pct for l in mandatory_labels)
            ) / _decimal(len(mandatory_labels))
        else:
            label_score = Decimal("100")

        # Reuse score: average progress toward 2030 targets
        if reuse:
            reuse_scores = []
            for r in reuse:
                if r.target_2030_pct > 0:
                    progress = min(r.reuse_pct / r.target_2030_pct * 100, 100)
                    reuse_scores.append(progress)
                else:
                    reuse_scores.append(100.0)
            reuse_score = _decimal(sum(reuse_scores)) / _decimal(len(reuse_scores))
        else:
            reuse_score = Decimal("50")  # No data

        # Substance score: all compliant = 100
        if substances:
            sub_compliant = sum(1 for s in substances if s.compliant)
            sub_score = _decimal(sub_compliant) / _decimal(len(substances)) * Decimal("100")
        else:
            sub_score = Decimal("100")

        # Weighted composite
        composite = (
            rc_score * Decimal("0.25")
            + grade_score * Decimal("0.20")
            + label_score * Decimal("0.20")
            + reuse_score * Decimal("0.15")
            + sub_score * Decimal("0.20")
        )

        return float(min(composite, Decimal("100")))

    def _generate_recommendations(
        self,
        rc_assessment: List[RecycledContentAssessment],
        labeling: List[LabelingComplianceDetail],
        reuse: List[ReuseProgressDetail],
        substances: List[SubstanceComplianceDetail],
        grade_dist: Dict[str, int],
        year: int,
    ) -> List[str]:
        """Generate actionable compliance recommendations.

        Analyses assessment results and produces prioritised recommendations.

        Args:
            rc_assessment: Recycled content results.
            labeling: Labeling results.
            reuse: Reuse results.
            substances: Substance results.
            grade_dist: Grade distribution.
            year: Reporting year.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # Recycled content gaps
        non_compliant_mats = [r for r in rc_assessment if not r.compliant_2030]
        if non_compliant_mats:
            mat_names = ", ".join(r.material for r in non_compliant_mats)
            recs.append(
                f"Increase recycled content for {mat_names} to meet 2030 PPWR targets. "
                "Engage recycled material suppliers and consider design-for-recycling."
            )

        # Grade improvements
        poor_grades = grade_dist.get("D", 0) + grade_dist.get("E", 0)
        total = sum(grade_dist.values()) or 1
        if poor_grades / total > 0.2:
            recs.append(
                "Over 20% of packaging has poor recyclability (grade D/E). "
                "Transition to mono-material designs and eliminate problematic "
                "components (mixed materials, non-recyclable adhesives)."
            )

        # Labeling gaps
        non_label = [l for l in labeling if l.status == "non_compliant"]
        if non_label:
            req_names = ", ".join(l.requirement for l in non_label)
            recs.append(
                f"Address non-compliant labeling: {req_names}. "
                "Implement harmonised EU labeling by the applicable deadlines."
            )

        # Reuse gaps
        off_track = [r for r in reuse if not r.on_track and r.target_2030_pct > 0]
        if off_track:
            fmts = ", ".join(r.packaging_format for r in off_track)
            recs.append(
                f"Accelerate reuse programmes for {fmts} to meet 2030 targets. "
                "Consider deposit-return schemes and standardised reusable formats."
            )

        # Substance issues
        non_sub = [s for s in substances if not s.compliant]
        if non_sub:
            sub_names = ", ".join(s.substance for s in non_sub)
            recs.append(
                f"Non-compliant substances detected: {sub_names}. "
                "Reformulate or source alternative materials immediately."
            )

        if not recs:
            recs.append(
                "Packaging portfolio is well-positioned for PPWR compliance. "
                "Continue monitoring targets and maintaining documentation."
            )

        return recs
