# -*- coding: utf-8 -*-
"""
RecycledContentEngine - PACK-020 Battery Passport Prep Engine 2
================================================================

Tracks and calculates recycled content of critical raw materials in
batteries per EU Battery Regulation Art 8.

Under Regulation (EU) 2023/1542 (the EU Battery Regulation), Article 8
establishes mandatory recycled content requirements for batteries placed
on the EU market.  The regulation introduces a phased approach with
documentation obligations from 2028, minimum recycled content targets
from 2031, and increased targets from 2036.

Regulation (EU) 2023/1542 Framework:
    - Art 8(1): From 18 August 2031, industrial batteries with a
      capacity above 2 kWh, EV batteries, SLI batteries, and LMT
      batteries shall contain minimum levels of recycled content.
    - Art 8(2): Minimum recycled content levels for 2031:
      (a) 16% cobalt; (b) 85% lead; (c) 6% lithium; (d) 6% nickel.
    - Art 8(3): Increased recycled content levels from 2036:
      (a) 26% cobalt; (b) 85% lead; (c) 12% lithium; (d) 15% nickel.
    - Art 8(4): From 18 August 2028, batteries shall be accompanied
      by documentation on the recycled content.
    - Art 8(5): The Commission shall adopt implementing acts laying
      down the methodology for calculating and verifying the recycled
      content share.

Regulatory Phases:
    - DOCUMENTATION_2028: Documentation requirement only (no minimums)
    - MINIMUM_2031: First mandatory minimum recycled content targets
    - INCREASED_2036: Increased mandatory recycled content targets

Regulatory References:
    - Regulation (EU) 2023/1542 of the European Parliament and of the
      Council of 12 July 2023 concerning batteries and waste batteries
    - Art 8 - Recycled content
    - Recitals 30-33 (recycled content rationale)
    - Commission Delegated Regulation (EU) 2024/1781 (implementing rules)
    - ISO 14021:2016 - Self-declared environmental claims

Zero-Hallucination:
    - All recycled content percentages use deterministic Decimal division
    - Target compliance uses deterministic threshold comparison
    - Gap calculations use deterministic subtraction
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

from greenlang.schemas import utcnow

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

class CriticalRawMaterial(str, Enum):
    """Critical raw material subject to recycled content requirements.

    Article 8 of the EU Battery Regulation specifies recycled content
    targets for cobalt, lithium, nickel, and lead.  Manganese is
    included as a material of growing regulatory interest.
    """
    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    LEAD = "lead"
    MANGANESE = "manganese"

class RecycledContentPhase(str, Enum):
    """Regulatory phase for recycled content requirements.

    The EU Battery Regulation introduces recycled content requirements
    in three phases: documentation (2028), minimum targets (2031),
    and increased targets (2036).
    """
    DOCUMENTATION_2028 = "documentation_2028"
    MINIMUM_2031 = "minimum_2031"
    INCREASED_2036 = "increased_2036"

class ComplianceStatus(str, Enum):
    """Compliance status for recycled content assessment.

    Indicates whether a material or battery meets, exceeds, or fails
    the applicable recycled content targets.
    """
    EXCEEDS_TARGET = "exceeds_target"
    MEETS_TARGET = "meets_target"
    BELOW_TARGET = "below_target"
    NO_TARGET = "no_target"
    DOCUMENTATION_ONLY = "documentation_only"

class VerificationMethod(str, Enum):
    """Method used to verify recycled content claims.

    Identifies how the recycled content percentage was determined
    and verified for regulatory compliance.
    """
    MASS_BALANCE = "mass_balance"
    CHAIN_OF_CUSTODY = "chain_of_custody"
    BOOK_AND_CLAIM = "book_and_claim"
    THIRD_PARTY_AUDIT = "third_party_audit"
    SELF_DECLARATION = "self_declaration"
    NOT_VERIFIED = "not_verified"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mandatory recycled content targets by phase and material.
# Values are percentages (e.g., 16 means 16%).
# Source: Regulation (EU) 2023/1542, Art 8(2) and Art 8(3).
RECYCLED_CONTENT_TARGETS: Dict[str, Dict[str, Decimal]] = {
    RecycledContentPhase.DOCUMENTATION_2028.value: {
        CriticalRawMaterial.COBALT.value: Decimal("0"),
        CriticalRawMaterial.LITHIUM.value: Decimal("0"),
        CriticalRawMaterial.NICKEL.value: Decimal("0"),
        CriticalRawMaterial.LEAD.value: Decimal("0"),
        CriticalRawMaterial.MANGANESE.value: Decimal("0"),
    },
    RecycledContentPhase.MINIMUM_2031.value: {
        CriticalRawMaterial.COBALT.value: Decimal("16"),
        CriticalRawMaterial.LITHIUM.value: Decimal("6"),
        CriticalRawMaterial.NICKEL.value: Decimal("6"),
        CriticalRawMaterial.LEAD.value: Decimal("85"),
        CriticalRawMaterial.MANGANESE.value: Decimal("0"),
    },
    RecycledContentPhase.INCREASED_2036.value: {
        CriticalRawMaterial.COBALT.value: Decimal("26"),
        CriticalRawMaterial.LITHIUM.value: Decimal("12"),
        CriticalRawMaterial.NICKEL.value: Decimal("15"),
        CriticalRawMaterial.LEAD.value: Decimal("85"),
        CriticalRawMaterial.MANGANESE.value: Decimal("0"),
    },
}

# Phase effective dates.
PHASE_EFFECTIVE_DATES: Dict[str, str] = {
    RecycledContentPhase.DOCUMENTATION_2028.value: "2028-08-18",
    RecycledContentPhase.MINIMUM_2031.value: "2031-08-18",
    RecycledContentPhase.INCREASED_2036.value: "2036-08-18",
}

# Phase descriptions.
PHASE_DESCRIPTIONS: Dict[str, str] = {
    RecycledContentPhase.DOCUMENTATION_2028.value: (
        "Documentation phase: batteries must be accompanied by technical "
        "documentation on the recycled content of cobalt, lead, lithium, "
        "and nickel. No minimum targets apply."
    ),
    RecycledContentPhase.MINIMUM_2031.value: (
        "First mandatory phase: batteries must contain minimum recycled "
        "content levels of 16% cobalt, 6% lithium, 6% nickel, and 85% lead."
    ),
    RecycledContentPhase.INCREASED_2036.value: (
        "Increased targets phase: batteries must contain minimum recycled "
        "content levels of 26% cobalt, 12% lithium, 15% nickel, and 85% lead."
    ),
}

# Material display labels and context.
MATERIAL_LABELS: Dict[str, Dict[str, str]] = {
    CriticalRawMaterial.COBALT.value: {
        "name": "Cobalt (Co)",
        "symbol": "Co",
        "typical_use": "Cathode active material in NMC, NCA chemistries",
        "supply_chain_risk": "High - concentrated in DRC",
    },
    CriticalRawMaterial.LITHIUM.value: {
        "name": "Lithium (Li)",
        "symbol": "Li",
        "typical_use": "Core element in all lithium-ion chemistries",
        "supply_chain_risk": "Medium - Australia, Chile, China dominant",
    },
    CriticalRawMaterial.NICKEL.value: {
        "name": "Nickel (Ni)",
        "symbol": "Ni",
        "typical_use": "Cathode active material in NMC, NCA chemistries",
        "supply_chain_risk": "Medium - Indonesia, Philippines dominant",
    },
    CriticalRawMaterial.LEAD.value: {
        "name": "Lead (Pb)",
        "symbol": "Pb",
        "typical_use": "Primary material in lead-acid batteries",
        "supply_chain_risk": "Low - well-established recycling infrastructure",
    },
    CriticalRawMaterial.MANGANESE.value: {
        "name": "Manganese (Mn)",
        "symbol": "Mn",
        "typical_use": "Cathode active material in NMC, LMO chemistries",
        "supply_chain_risk": "Medium - South Africa, Gabon dominant",
    },
}

# Verification method descriptions.
VERIFICATION_DESCRIPTIONS: Dict[str, str] = {
    VerificationMethod.MASS_BALANCE.value: (
        "Mass balance approach: tracks total recycled content input "
        "and attributes it proportionally to output products"
    ),
    VerificationMethod.CHAIN_OF_CUSTODY.value: (
        "Chain of custody: physical tracing of recycled material "
        "from recycler through processing to battery manufacturer"
    ),
    VerificationMethod.BOOK_AND_CLAIM.value: (
        "Book and claim: certificate-based system where recycled "
        "content credits can be traded independently of physical material"
    ),
    VerificationMethod.THIRD_PARTY_AUDIT.value: (
        "Independent third-party audit of recycled content claims "
        "by an accredited verification body"
    ),
    VerificationMethod.SELF_DECLARATION.value: (
        "Self-declared recycled content by the manufacturer without "
        "independent verification"
    ),
    VerificationMethod.NOT_VERIFIED.value: (
        "Recycled content has not been verified through any method"
    ),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class MaterialInput(BaseModel):
    """Input data for a single material's recycled content.

    Contains the total weight of the material in the battery and the
    weight sourced from recycled streams, used to calculate the
    recycled content percentage.
    """
    material: CriticalRawMaterial = Field(
        ...,
        description="Critical raw material identifier",
    )
    total_kg: Decimal = Field(
        ...,
        description="Total weight of this material in the battery (kg)",
        gt=0,
    )
    recycled_kg: Decimal = Field(
        ...,
        description="Weight sourced from recycled material (kg)",
        ge=0,
    )
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.NOT_VERIFIED,
        description="Method used to verify the recycled content",
    )
    recycler_name: str = Field(
        default="",
        description="Name of the recycler or secondary material supplier",
        max_length=500,
    )
    recycler_location: str = Field(
        default="",
        description="Location of the recycler",
        max_length=500,
    )
    certificate_id: str = Field(
        default="",
        description="Recycled content certificate identifier",
        max_length=200,
    )
    data_quality: str = Field(
        default="primary",
        description="Data quality indicator: 'primary', 'secondary', or 'estimated'",
        max_length=50,
    )
    notes: str = Field(
        default="",
        description="Additional notes about the material sourcing",
        max_length=2000,
    )

    @field_validator("recycled_kg")
    @classmethod
    def validate_recycled_not_exceeding_total(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that recycled weight does not exceed total weight."""
        # Note: Pydantic v2 field_validator with access to other fields
        # is done via info.data
        total = info.data.get("total_kg")
        if total is not None and v > total:
            raise ValueError(
                f"recycled_kg ({v}) cannot exceed total_kg ({total})"
            )
        return v

class RecycledContentInput(BaseModel):
    """Input data for battery recycled content calculation per Art 8.

    Contains battery identification and a list of material entries with
    total and recycled weights for each critical raw material.
    """
    battery_id: str = Field(
        ...,
        description="Unique battery identifier",
        min_length=1,
        max_length=200,
    )
    materials: List[MaterialInput] = Field(
        ...,
        description="List of material entries with recycled content data",
        min_length=1,
    )
    manufacturer_id: str = Field(
        default="",
        description="Manufacturer identifier",
        max_length=200,
    )
    manufacturing_date: str = Field(
        default="",
        description="Manufacturing date or period",
        max_length=50,
    )
    reporting_period: str = Field(
        default="",
        description="Reporting period (e.g., '2025', '2025-Q1')",
        max_length=50,
    )

    @field_validator("materials")
    @classmethod
    def validate_no_duplicate_materials(
        cls, v: List[MaterialInput]
    ) -> List[MaterialInput]:
        """Validate that each material appears at most once."""
        materials_seen = set()
        for m in v:
            if m.material.value in materials_seen:
                raise ValueError(
                    f"Duplicate material entry for {m.material.value}"
                )
            materials_seen.add(m.material.value)
        return v

class MaterialResult(BaseModel):
    """Result for a single material's recycled content assessment.

    Contains the calculated recycled content percentage, applicable
    target, compliance status, and gap analysis.
    """
    material: CriticalRawMaterial = Field(
        ...,
        description="Critical raw material",
    )
    material_label: str = Field(
        default="",
        description="Human-readable material label",
    )
    total_kg: Decimal = Field(
        ...,
        description="Total material weight (kg)",
    )
    recycled_kg: Decimal = Field(
        ...,
        description="Recycled material weight (kg)",
    )
    recycled_content_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Recycled content percentage",
    )
    target_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Applicable target percentage for the current phase",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.NO_TARGET,
        description="Compliance status against target",
    )
    gap_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Gap between actual and target (negative = surplus)",
    )
    gap_kg: Decimal = Field(
        default=Decimal("0.000"),
        description="Additional recycled material needed to meet target (kg)",
    )
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.NOT_VERIFIED,
        description="Verification method used",
    )
    next_phase_target_pct: Optional[Decimal] = Field(
        default=None,
        description="Target for the next regulatory phase",
    )
    next_phase_gap_pct: Optional[Decimal] = Field(
        default=None,
        description="Gap to the next phase target",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash for this material result",
    )

class RecycledContentResult(BaseModel):
    """Result of battery recycled content calculation per Art 8.

    Contains per-material results, overall compliance status,
    applicable phase, and recommendations.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    battery_id: str = Field(
        ...,
        description="Battery identifier",
    )
    material_results: List[MaterialResult] = Field(
        default_factory=list,
        description="Per-material recycled content results",
    )
    overall_compliance: bool = Field(
        default=False,
        description="Whether all materials meet their targets",
    )
    phase: RecycledContentPhase = Field(
        default=RecycledContentPhase.MINIMUM_2031,
        description="Regulatory phase used for target comparison",
    )
    phase_description: str = Field(
        default="",
        description="Description of the applicable regulatory phase",
    )
    phase_effective_date: str = Field(
        default="",
        description="Effective date of the applicable phase",
    )
    targets_met: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-material target compliance summary",
    )
    materials_assessed: int = Field(
        default=0,
        description="Number of materials assessed",
    )
    materials_compliant: int = Field(
        default=0,
        description="Number of materials meeting targets",
    )
    materials_non_compliant: int = Field(
        default=0,
        description="Number of materials below targets",
    )
    overall_recycled_content_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Weighted average recycled content across all materials",
    )
    total_material_weight_kg: Decimal = Field(
        default=Decimal("0.000"),
        description="Total weight of all assessed materials",
    )
    total_recycled_weight_kg: Decimal = Field(
        default=Decimal("0.000"),
        description="Total weight of recycled materials",
    )
    data_quality_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of materials by data quality level",
    )
    verification_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of materials by verification method",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving recycled content",
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

class RecycledContentEngine:
    """Battery recycled content engine per EU Battery Regulation Art 8.

    Provides deterministic, zero-hallucination calculation of:
    - Per-material recycled content percentages
    - Target compliance against applicable regulatory phase
    - Gap analysis showing shortfall or surplus vs targets
    - Forward-looking assessment against next-phase targets
    - Weighted average recycled content across all materials
    - Documentation generation for regulatory submission

    All calculations use Decimal arithmetic and are bit-perfect
    reproducible.  No LLM is used in any calculation path.

    Usage::

        engine = RecycledContentEngine()
        inp = RecycledContentInput(
            battery_id="BAT-EV-2025-001",
            materials=[
                MaterialInput(
                    material=CriticalRawMaterial.COBALT,
                    total_kg=Decimal("12.5"),
                    recycled_kg=Decimal("2.5"),
                ),
                MaterialInput(
                    material=CriticalRawMaterial.LITHIUM,
                    total_kg=Decimal("8.0"),
                    recycled_kg=Decimal("0.6"),
                ),
            ],
        )
        result = engine.calculate_recycled_content(inp)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise RecycledContentEngine."""
        self._results: List[RecycledContentResult] = []
        logger.info(
            "RecycledContentEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Main Calculation                                                     #
    # ------------------------------------------------------------------ #

    def calculate_recycled_content(
        self,
        input_data: RecycledContentInput,
        phase: RecycledContentPhase = RecycledContentPhase.MINIMUM_2031,
    ) -> RecycledContentResult:
        """Calculate recycled content and assess compliance per Art 8.

        Computes the recycled content percentage for each critical raw
        material, compares against the applicable regulatory targets,
        and produces a comprehensive compliance assessment.

        Args:
            input_data: Validated RecycledContentInput with material data.
            phase: Regulatory phase for target comparison.

        Returns:
            RecycledContentResult with complete assessment.

        Raises:
            ValueError: If input validation fails.
        """
        t0 = time.perf_counter()

        # Step 1: Validate input
        validation_errors = self._validate_input(input_data)
        if validation_errors:
            raise ValueError(
                f"Input validation failed: {'; '.join(validation_errors)}"
            )

        # Step 2: Calculate per-material recycled content
        material_results: List[MaterialResult] = []
        targets_met: Dict[str, bool] = {}
        total_weight = Decimal("0.000")
        total_recycled = Decimal("0.000")
        dq_summary: Dict[str, int] = {}
        verification_summary: Dict[str, int] = {}

        for mat_input in input_data.materials:
            # Calculate percentage
            recycled_pct = self._calculate_material_pct(
                mat_input.recycled_kg, mat_input.total_kg
            )

            # Get target for this material and phase
            target_pct = self.get_target_for_material(mat_input.material, phase)

            # Check target compliance
            compliance = self._assess_material_compliance(
                recycled_pct, target_pct, phase
            )

            # Calculate gap
            gap_pct = self._calculate_gap_pct(recycled_pct, target_pct)
            gap_kg = self._calculate_gap_kg(
                mat_input.total_kg, recycled_pct, target_pct
            )

            # Next phase assessment
            next_phase_target, next_phase_gap = self._assess_next_phase(
                mat_input.material, recycled_pct, phase
            )

            # Material label
            mat_label_data = MATERIAL_LABELS.get(mat_input.material.value, {})
            mat_label = mat_label_data.get("name", mat_input.material.value)

            mat_result = MaterialResult(
                material=mat_input.material,
                material_label=mat_label,
                total_kg=_round_val(mat_input.total_kg, 3),
                recycled_kg=_round_val(mat_input.recycled_kg, 3),
                recycled_content_pct=_round_val(recycled_pct, 2),
                target_pct=_round_val(target_pct, 2),
                compliance_status=compliance,
                gap_pct=_round_val(gap_pct, 2),
                gap_kg=_round_val(gap_kg, 3),
                verification_method=mat_input.verification_method,
                next_phase_target_pct=(
                    _round_val(next_phase_target, 2)
                    if next_phase_target is not None else None
                ),
                next_phase_gap_pct=(
                    _round_val(next_phase_gap, 2)
                    if next_phase_gap is not None else None
                ),
            )
            mat_result.provenance_hash = _compute_hash(mat_result)
            material_results.append(mat_result)

            # Track compliance
            is_met = compliance in (
                ComplianceStatus.MEETS_TARGET,
                ComplianceStatus.EXCEEDS_TARGET,
                ComplianceStatus.NO_TARGET,
                ComplianceStatus.DOCUMENTATION_ONLY,
            )
            targets_met[mat_input.material.value] = is_met

            # Aggregate weights
            total_weight += mat_input.total_kg
            total_recycled += mat_input.recycled_kg

            # Data quality
            dq = mat_input.data_quality
            dq_summary[dq] = dq_summary.get(dq, 0) + 1

            # Verification
            vm = mat_input.verification_method.value
            verification_summary[vm] = verification_summary.get(vm, 0) + 1

        # Step 3: Overall compliance
        overall_compliance = all(targets_met.values()) if targets_met else False

        # Step 4: Overall weighted recycled content
        overall_pct = Decimal("0.00")
        if total_weight > 0:
            overall_pct = _round_val(
                (total_recycled / total_weight) * Decimal("100"), 2
            )

        # Step 5: Counts
        compliant_count = sum(1 for v in targets_met.values() if v)
        non_compliant_count = len(targets_met) - compliant_count

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            material_results, phase, overall_compliance
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = RecycledContentResult(
            battery_id=input_data.battery_id,
            material_results=material_results,
            overall_compliance=overall_compliance,
            phase=phase,
            phase_description=PHASE_DESCRIPTIONS.get(phase.value, ""),
            phase_effective_date=PHASE_EFFECTIVE_DATES.get(phase.value, ""),
            targets_met=targets_met,
            materials_assessed=len(material_results),
            materials_compliant=compliant_count,
            materials_non_compliant=non_compliant_count,
            overall_recycled_content_pct=overall_pct,
            total_material_weight_kg=_round_val(total_weight, 3),
            total_recycled_weight_kg=_round_val(total_recycled, 3),
            data_quality_summary=dq_summary,
            verification_summary=verification_summary,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        self._results.append(result)

        logger.info(
            "Calculated recycled content for %s: %d materials, "
            "overall=%s%%, compliance=%s, phase=%s in %.3f ms",
            input_data.battery_id,
            len(material_results),
            overall_pct,
            "PASS" if overall_compliance else "FAIL",
            phase.value,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Target Checks                                                        #
    # ------------------------------------------------------------------ #

    def check_targets(
        self,
        content_pct: Decimal,
        material: CriticalRawMaterial,
        phase: RecycledContentPhase,
    ) -> Dict[str, Any]:
        """Check recycled content against regulatory targets.

        Compares the actual recycled content percentage against the
        target for the specified material and regulatory phase.

        Args:
            content_pct: Actual recycled content percentage.
            material: Critical raw material.
            phase: Regulatory phase.

        Returns:
            Dict with compliance assessment.
        """
        target = self.get_target_for_material(material, phase)
        val = _decimal(content_pct)

        compliance = self._assess_material_compliance(val, target, phase)
        gap = self._calculate_gap_pct(val, target)

        mat_label_data = MATERIAL_LABELS.get(material.value, {})
        mat_label = mat_label_data.get("name", material.value)

        return {
            "material": material.value,
            "material_label": mat_label,
            "actual_pct": str(_round_val(val, 2)),
            "target_pct": str(_round_val(target, 2)),
            "phase": phase.value,
            "phase_effective_date": PHASE_EFFECTIVE_DATES.get(phase.value, ""),
            "compliance_status": compliance.value,
            "gap_pct": str(_round_val(gap, 2)),
            "compliant": compliance in (
                ComplianceStatus.MEETS_TARGET,
                ComplianceStatus.EXCEEDS_TARGET,
                ComplianceStatus.NO_TARGET,
                ComplianceStatus.DOCUMENTATION_ONLY,
            ),
            "provenance_hash": _compute_hash({
                "material": material.value,
                "actual_pct": str(val),
                "target_pct": str(target),
                "phase": phase.value,
            }),
        }

    # ------------------------------------------------------------------ #
    # Target Lookup                                                        #
    # ------------------------------------------------------------------ #

    def get_target_for_material(
        self,
        material: CriticalRawMaterial,
        phase: RecycledContentPhase,
    ) -> Decimal:
        """Look up the recycled content target for a material and phase.

        Args:
            material: Critical raw material.
            phase: Regulatory phase.

        Returns:
            Target percentage as Decimal (0 if no target).
        """
        phase_targets = RECYCLED_CONTENT_TARGETS.get(phase.value, {})
        return phase_targets.get(material.value, Decimal("0"))

    def get_all_targets(
        self, phase: RecycledContentPhase
    ) -> Dict[str, str]:
        """Return all material targets for a regulatory phase.

        Args:
            phase: Regulatory phase.

        Returns:
            Dict mapping material to target percentage.
        """
        phase_targets = RECYCLED_CONTENT_TARGETS.get(phase.value, {})
        return {k: str(v) for k, v in phase_targets.items()}

    def get_phase_info(
        self, phase: RecycledContentPhase
    ) -> Dict[str, Any]:
        """Return detailed information about a regulatory phase.

        Args:
            phase: Regulatory phase.

        Returns:
            Dict with phase details.
        """
        return {
            "phase": phase.value,
            "description": PHASE_DESCRIPTIONS.get(phase.value, ""),
            "effective_date": PHASE_EFFECTIVE_DATES.get(phase.value, ""),
            "targets": self.get_all_targets(phase),
            "regulation_reference": "Regulation (EU) 2023/1542, Art 8",
        }

    # ------------------------------------------------------------------ #
    # Multi-Phase Assessment                                               #
    # ------------------------------------------------------------------ #

    def assess_all_phases(
        self, input_data: RecycledContentInput
    ) -> Dict[str, Any]:
        """Assess recycled content against all regulatory phases.

        Runs the recycled content calculation for each of the three
        regulatory phases (2028, 2031, 2036) and produces a comparative
        view of compliance readiness.

        Args:
            input_data: RecycledContentInput with material data.

        Returns:
            Dict with per-phase results and readiness summary.
        """
        t0 = time.perf_counter()

        phases = [
            RecycledContentPhase.DOCUMENTATION_2028,
            RecycledContentPhase.MINIMUM_2031,
            RecycledContentPhase.INCREASED_2036,
        ]

        phase_results: Dict[str, Any] = {}
        readiness: Dict[str, bool] = {}

        for phase in phases:
            result = self.calculate_recycled_content(input_data, phase)
            phase_results[phase.value] = {
                "overall_compliance": result.overall_compliance,
                "materials_compliant": result.materials_compliant,
                "materials_non_compliant": result.materials_non_compliant,
                "targets_met": result.targets_met,
                "phase_description": result.phase_description,
                "effective_date": result.phase_effective_date,
            }
            readiness[phase.value] = result.overall_compliance

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        assessment = {
            "battery_id": input_data.battery_id,
            "phase_results": phase_results,
            "readiness_summary": readiness,
            "all_phases_compliant": all(readiness.values()),
            "processing_time_ms": elapsed_ms,
        }
        assessment["provenance_hash"] = _compute_hash(assessment)

        logger.info(
            "Multi-phase assessment for %s: 2028=%s, 2031=%s, 2036=%s in %.3f ms",
            input_data.battery_id,
            readiness.get(RecycledContentPhase.DOCUMENTATION_2028.value),
            readiness.get(RecycledContentPhase.MINIMUM_2031.value),
            readiness.get(RecycledContentPhase.INCREASED_2036.value),
            elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------ #
    # Batch Processing                                                     #
    # ------------------------------------------------------------------ #

    def calculate_batch(
        self,
        inputs: List[RecycledContentInput],
        phase: RecycledContentPhase = RecycledContentPhase.MINIMUM_2031,
    ) -> List[RecycledContentResult]:
        """Calculate recycled content for a batch of batteries.

        Args:
            inputs: List of RecycledContentInput objects.
            phase: Regulatory phase for target comparison.

        Returns:
            List of RecycledContentResult objects.
        """
        t0 = time.perf_counter()
        results: List[RecycledContentResult] = []

        for inp in inputs:
            try:
                result = self.calculate_recycled_content(inp, phase)
                results.append(result)
            except ValueError as e:
                logger.warning(
                    "Skipping battery %s due to validation error: %s",
                    inp.battery_id, str(e),
                )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Batch recycled content: %d/%d calculated in %.3f ms",
            len(results), len(inputs), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Documentation Builder                                                #
    # ------------------------------------------------------------------ #

    def build_documentation(
        self, result: RecycledContentResult
    ) -> Dict[str, Any]:
        """Build recycled content documentation per Art 8(4).

        Produces a structured document suitable for inclusion in
        the battery passport and regulatory submission.

        Args:
            result: RecycledContentResult to build documentation from.

        Returns:
            Dict with complete documentation.
        """
        t0 = time.perf_counter()

        doc: Dict[str, Any] = {
            "document_id": _new_uuid(),
            "regulation_reference": "Regulation (EU) 2023/1542, Art 8",
            "document_type": "Recycled Content Documentation",
            "battery_id": result.battery_id,
            "phase": result.phase.value,
            "phase_description": result.phase_description,
            "effective_date": result.phase_effective_date,
            "materials": [],
            "summary": {
                "overall_compliance": result.overall_compliance,
                "overall_recycled_content_pct": str(
                    result.overall_recycled_content_pct
                ),
                "total_material_weight_kg": str(
                    result.total_material_weight_kg
                ),
                "total_recycled_weight_kg": str(
                    result.total_recycled_weight_kg
                ),
                "materials_assessed": result.materials_assessed,
                "materials_compliant": result.materials_compliant,
            },
            "verification_methods": result.verification_summary,
            "generated_at": str(result.calculated_at),
            "engine_version": result.engine_version,
        }

        for mr in result.material_results:
            mat_info = MATERIAL_LABELS.get(mr.material.value, {})
            doc["materials"].append({
                "material": mr.material.value,
                "material_name": mat_info.get("name", mr.material.value),
                "symbol": mat_info.get("symbol", ""),
                "total_kg": str(mr.total_kg),
                "recycled_kg": str(mr.recycled_kg),
                "recycled_content_pct": str(mr.recycled_content_pct),
                "target_pct": str(mr.target_pct),
                "compliance_status": mr.compliance_status.value,
                "gap_pct": str(mr.gap_pct),
                "gap_kg": str(mr.gap_kg),
                "verification_method": mr.verification_method.value,
            })

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        doc["processing_time_ms"] = elapsed_ms
        doc["provenance_hash"] = _compute_hash(doc)

        logger.info(
            "Built recycled content documentation for %s in %.3f ms",
            result.battery_id, elapsed_ms,
        )
        return doc

    # ------------------------------------------------------------------ #
    # Comparison Utilities                                                 #
    # ------------------------------------------------------------------ #

    def compare_results(
        self, results: List[RecycledContentResult]
    ) -> Dict[str, Any]:
        """Compare recycled content across multiple batteries.

        Args:
            results: List of RecycledContentResult objects.

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

        comparison_entries = []
        for r in results:
            entry = {
                "battery_id": r.battery_id,
                "overall_recycled_pct": str(r.overall_recycled_content_pct),
                "overall_compliance": r.overall_compliance,
                "materials_compliant": r.materials_compliant,
                "materials_non_compliant": r.materials_non_compliant,
            }
            for mr in r.material_results:
                entry[f"{mr.material.value}_pct"] = str(mr.recycled_content_pct)
            comparison_entries.append(entry)

        # Sort by overall recycled content (descending)
        comparison_entries.sort(
            key=lambda x: Decimal(x["overall_recycled_pct"]),
            reverse=True,
        )

        # Statistics
        overall_pcts = [r.overall_recycled_content_pct for r in results]
        min_pct = min(overall_pcts)
        max_pct = max(overall_pcts)
        avg_pct = _round_val(
            sum(overall_pcts) / _decimal(len(overall_pcts)), 2
        )

        compliant_count = sum(1 for r in results if r.overall_compliance)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        comparison = {
            "count": len(results),
            "ranking": comparison_entries,
            "statistics": {
                "min_overall_pct": str(min_pct),
                "max_overall_pct": str(max_pct),
                "avg_overall_pct": str(avg_pct),
            },
            "compliance_summary": {
                "compliant_count": compliant_count,
                "non_compliant_count": len(results) - compliant_count,
                "compliance_rate_pct": str(_round_val(
                    _decimal(compliant_count) / _decimal(len(results)) * Decimal("100"),
                    2,
                )),
            },
            "processing_time_ms": elapsed_ms,
        }
        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Material Information                                                 #
    # ------------------------------------------------------------------ #

    def get_material_info(
        self, material: CriticalRawMaterial
    ) -> Dict[str, Any]:
        """Return detailed information about a critical raw material.

        Args:
            material: Critical raw material.

        Returns:
            Dict with material details and targets across phases.
        """
        mat_data = MATERIAL_LABELS.get(material.value, {})

        targets = {}
        for phase in RecycledContentPhase:
            target = self.get_target_for_material(material, phase)
            targets[phase.value] = {
                "target_pct": str(target),
                "effective_date": PHASE_EFFECTIVE_DATES.get(phase.value, ""),
            }

        return {
            "material": material.value,
            "name": mat_data.get("name", material.value),
            "symbol": mat_data.get("symbol", ""),
            "typical_use": mat_data.get("typical_use", ""),
            "supply_chain_risk": mat_data.get("supply_chain_risk", ""),
            "targets_by_phase": targets,
            "regulation_reference": "Regulation (EU) 2023/1542, Art 8",
        }

    # ------------------------------------------------------------------ #
    # Registry Management                                                  #
    # ------------------------------------------------------------------ #

    def get_results(self) -> List[RecycledContentResult]:
        """Return all calculated results.

        Returns:
            List of RecycledContentResult objects.
        """
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        logger.info("RecycledContentEngine results cleared")

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate_input(
        self, input_data: RecycledContentInput
    ) -> List[str]:
        """Validate input data for recycled content calculation.

        Args:
            input_data: RecycledContentInput to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if not input_data.materials:
            errors.append("At least one material entry is required")

        for mat in input_data.materials:
            if mat.recycled_kg > mat.total_kg:
                errors.append(
                    f"{mat.material.value}: recycled_kg ({mat.recycled_kg}) "
                    f"exceeds total_kg ({mat.total_kg})"
                )

            if mat.total_kg <= 0:
                errors.append(
                    f"{mat.material.value}: total_kg must be positive"
                )

        return errors

    def _calculate_material_pct(
        self, recycled_kg: Decimal, total_kg: Decimal
    ) -> Decimal:
        """Calculate recycled content percentage for a material.

        Formula (deterministic):
            pct = (recycled_kg / total_kg) * 100

        Args:
            recycled_kg: Weight of recycled material (kg).
            total_kg: Total weight of material (kg).

        Returns:
            Recycled content percentage.
        """
        if total_kg <= 0:
            return Decimal("0.00")
        return (recycled_kg / total_kg) * Decimal("100")

    def _assess_material_compliance(
        self,
        actual_pct: Decimal,
        target_pct: Decimal,
        phase: RecycledContentPhase,
    ) -> ComplianceStatus:
        """Assess compliance status for a single material.

        Args:
            actual_pct: Actual recycled content percentage.
            target_pct: Target percentage for the phase.
            phase: Regulatory phase.

        Returns:
            ComplianceStatus enum value.
        """
        if phase == RecycledContentPhase.DOCUMENTATION_2028:
            return ComplianceStatus.DOCUMENTATION_ONLY

        if target_pct <= 0:
            return ComplianceStatus.NO_TARGET

        if actual_pct > target_pct:
            return ComplianceStatus.EXCEEDS_TARGET
        if actual_pct == target_pct:
            return ComplianceStatus.MEETS_TARGET
        return ComplianceStatus.BELOW_TARGET

    def _calculate_gap_pct(
        self, actual_pct: Decimal, target_pct: Decimal
    ) -> Decimal:
        """Calculate gap between actual and target percentage.

        A positive gap means the material is below target.
        A negative gap means the material exceeds the target (surplus).

        Args:
            actual_pct: Actual recycled content percentage.
            target_pct: Target percentage.

        Returns:
            Gap in percentage points (positive = shortfall).
        """
        return target_pct - actual_pct

    def _calculate_gap_kg(
        self,
        total_kg: Decimal,
        actual_pct: Decimal,
        target_pct: Decimal,
    ) -> Decimal:
        """Calculate additional recycled material needed in kilograms.

        If the material meets or exceeds the target, returns zero.

        Args:
            total_kg: Total material weight (kg).
            actual_pct: Actual recycled content percentage.
            target_pct: Target percentage.

        Returns:
            Additional recycled material needed (kg). Zero if target met.
        """
        if actual_pct >= target_pct:
            return Decimal("0.000")

        target_kg = total_kg * (target_pct / Decimal("100"))
        actual_kg = total_kg * (actual_pct / Decimal("100"))
        return target_kg - actual_kg

    def _assess_next_phase(
        self,
        material: CriticalRawMaterial,
        actual_pct: Decimal,
        current_phase: RecycledContentPhase,
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Assess readiness for the next regulatory phase.

        Args:
            material: Critical raw material.
            actual_pct: Current recycled content percentage.
            current_phase: Current regulatory phase.

        Returns:
            Tuple of (next_phase_target, gap_to_next_phase).
            Returns (None, None) if no next phase.
        """
        phase_order = [
            RecycledContentPhase.DOCUMENTATION_2028,
            RecycledContentPhase.MINIMUM_2031,
            RecycledContentPhase.INCREASED_2036,
        ]

        current_idx = None
        for i, p in enumerate(phase_order):
            if p == current_phase:
                current_idx = i
                break

        if current_idx is None or current_idx >= len(phase_order) - 1:
            return (None, None)

        next_phase = phase_order[current_idx + 1]
        next_target = self.get_target_for_material(material, next_phase)

        if next_target <= 0:
            return (None, None)

        gap = next_target - actual_pct
        return (next_target, gap)

    def _generate_recommendations(
        self,
        material_results: List[MaterialResult],
        phase: RecycledContentPhase,
        overall_compliance: bool,
    ) -> List[str]:
        """Generate recommendations for improving recycled content.

        Provides actionable guidance based on compliance gaps,
        verification methods, and upcoming phase targets.

        Args:
            material_results: Per-material assessment results.
            phase: Current regulatory phase.
            overall_compliance: Whether overall compliance is achieved.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Non-compliant materials
        for mr in material_results:
            if mr.compliance_status == ComplianceStatus.BELOW_TARGET:
                recommendations.append(
                    f"{mr.material_label}: recycled content of "
                    f"{mr.recycled_content_pct}% is below the "
                    f"{mr.target_pct}% target. An additional "
                    f"{mr.gap_kg} kg of recycled {mr.material.value} "
                    f"is needed to meet the {phase.value} requirement."
                )

        # Verification improvements
        for mr in material_results:
            if mr.verification_method in (
                VerificationMethod.NOT_VERIFIED,
                VerificationMethod.SELF_DECLARATION,
            ):
                recommendations.append(
                    f"{mr.material_label}: currently using "
                    f"{mr.verification_method.value} verification. "
                    f"Consider upgrading to third-party audit or "
                    f"chain of custody for regulatory acceptance."
                )

        # Next phase readiness
        for mr in material_results:
            if (
                mr.next_phase_target_pct is not None
                and mr.next_phase_gap_pct is not None
                and mr.next_phase_gap_pct > 0
            ):
                recommendations.append(
                    f"{mr.material_label}: current level of "
                    f"{mr.recycled_content_pct}% will need to increase "
                    f"to {mr.next_phase_target_pct}% for the next "
                    f"regulatory phase. Begin engaging recycled material "
                    f"suppliers to close the {mr.next_phase_gap_pct} "
                    f"percentage point gap."
                )

        # Overall guidance
        if overall_compliance and phase == RecycledContentPhase.MINIMUM_2031:
            recommendations.append(
                "Battery meets all 2031 minimum targets. "
                "Plan ahead for the 2036 increased targets by "
                "securing long-term recycled material supply agreements."
            )

        if not overall_compliance:
            recommendations.append(
                "Battery does not meet all recycled content targets. "
                "Prioritise sourcing recycled material for non-compliant "
                "materials and establish verification processes."
            )

        # Lead-acid specific guidance
        for mr in material_results:
            if (
                mr.material == CriticalRawMaterial.LEAD
                and mr.recycled_content_pct < Decimal("85")
            ):
                recommendations.append(
                    "Lead recycled content is below the 85% target. "
                    "Lead-acid batteries typically achieve >95% recycled "
                    "content through established recycling infrastructure. "
                    "Review supply chain for recycled lead sources."
                )

        return recommendations
