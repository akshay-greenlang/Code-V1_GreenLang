# -*- coding: utf-8 -*-
"""
EndOfLifeEngine - PACK-020 Battery Passport Engine 7
======================================================

Tracks collection rates, recycling efficiency, and material recovery
targets for batteries at end-of-life per Articles 56-71 of the EU
Battery Regulation (2023/1542).

Articles 56-71 of the EU Battery Regulation establish the obligations
for collection, treatment, recycling, and material recovery of waste
batteries.  The regulation sets progressively tightening targets for
collection rates, recycling efficiency, and the recovery of specific
critical raw materials (cobalt, lithium, nickel, copper, and lead).

Collection Targets:
    - Portable batteries: 63% by 2027, 73% by 2030
    - LMT batteries: 51% by 2028, 61% by 2031
    - EV batteries: 100% take-back obligation
    - Industrial batteries: 100% take-back obligation
    - SLI batteries: 100% take-back obligation

Material Recovery Targets:
    - Lithium: 50% by 31 Dec 2027, 80% by 31 Dec 2031
    - Cobalt: 90% by 31 Dec 2027, 95% by 31 Dec 2031
    - Nickel: 90% by 31 Dec 2027, 95% by 31 Dec 2031
    - Copper: 90% by 31 Dec 2027, 95% by 31 Dec 2031
    - Lead: 90% by 31 Dec 2027, 95% by 31 Dec 2031

Recycling Efficiency:
    - Lead-acid batteries: 75% by weight
    - Lithium-based batteries: 65% by 31 Dec 2025, 70% by 31 Dec 2030

Regulatory References:
    - EU Regulation 2023/1542 (EU Battery Regulation), Art 56-71
    - Directive 2006/66/EC (legacy waste battery directive)
    - Directive 2008/98/EC (Waste Framework Directive)
    - Commission Delegated Regulation on recycling efficiency
    - Basel Convention on hazardous waste shipment

Zero-Hallucination:
    - Collection rates are deterministic weight ratios
    - Recovery percentages use Decimal arithmetic
    - Target comparisons are simple threshold checks
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep Pack
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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BatteryCategory(str, Enum):
    """Battery category per EU Battery Regulation classification.

    Determines which collection, recycling, and recovery targets
    apply to the battery type.
    """
    PORTABLE = "portable"
    LMT = "lmt"
    SLI = "sli"
    EV = "ev"
    INDUSTRIAL = "industrial"

class RecoveryMaterial(str, Enum):
    """Critical raw material subject to recovery targets per Art 71.

    Each material has specific minimum recovery rate targets
    that must be achieved by recycling facilities.
    """
    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    COPPER = "copper"
    LEAD = "lead"

class EOLPhase(str, Enum):
    """Phase in the end-of-life management process.

    Tracks the lifecycle stage of waste battery processing
    from collection through to final recovery or disposal.

from greenlang.schemas import utcnow
    """
    COLLECTION = "collection"
    DISMANTLING = "dismantling"
    RECYCLING = "recycling"
    RECOVERY = "recovery"
    DISPOSAL = "disposal"

class BatteryChemistry(str, Enum):
    """Battery chemistry type for recycling efficiency targets.

    Different chemistries have different recycling efficiency
    requirements under the regulation.
    """
    LEAD_ACID = "lead_acid"
    LITHIUM_ION = "lithium_ion"
    NICKEL_METAL_HYDRIDE = "nickel_metal_hydride"
    OTHER = "other"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Collection rate targets by category and year.
# Values are minimum collection rates as percentages.
COLLECTION_TARGETS: Dict[str, Dict[int, float]] = {
    BatteryCategory.PORTABLE.value: {
        2024: 45.0,
        2025: 45.0,
        2026: 45.0,
        2027: 63.0,
        2028: 63.0,
        2029: 63.0,
        2030: 73.0,
        2031: 73.0,
    },
    BatteryCategory.LMT.value: {
        2026: 51.0,
        2027: 51.0,
        2028: 51.0,
        2029: 51.0,
        2030: 51.0,
        2031: 61.0,
    },
    BatteryCategory.SLI.value: {
        # 100% take-back obligation
        2024: 100.0,
        2025: 100.0,
        2026: 100.0,
        2027: 100.0,
        2028: 100.0,
        2029: 100.0,
        2030: 100.0,
        2031: 100.0,
    },
    BatteryCategory.EV.value: {
        # 100% take-back obligation
        2024: 100.0,
        2025: 100.0,
        2026: 100.0,
        2027: 100.0,
        2028: 100.0,
        2029: 100.0,
        2030: 100.0,
        2031: 100.0,
    },
    BatteryCategory.INDUSTRIAL.value: {
        # 100% take-back obligation
        2024: 100.0,
        2025: 100.0,
        2026: 100.0,
        2027: 100.0,
        2028: 100.0,
        2029: 100.0,
        2030: 100.0,
        2031: 100.0,
    },
}

# Material recovery targets by material and year (%).
MATERIAL_RECOVERY_TARGETS: Dict[str, Dict[int, float]] = {
    RecoveryMaterial.LITHIUM.value: {
        2027: 50.0,
        2028: 50.0,
        2029: 50.0,
        2030: 50.0,
        2031: 80.0,
    },
    RecoveryMaterial.COBALT.value: {
        2027: 90.0,
        2028: 90.0,
        2029: 90.0,
        2030: 90.0,
        2031: 95.0,
    },
    RecoveryMaterial.NICKEL.value: {
        2027: 90.0,
        2028: 90.0,
        2029: 90.0,
        2030: 90.0,
        2031: 95.0,
    },
    RecoveryMaterial.COPPER.value: {
        2027: 90.0,
        2028: 90.0,
        2029: 90.0,
        2030: 90.0,
        2031: 95.0,
    },
    RecoveryMaterial.LEAD.value: {
        2027: 90.0,
        2028: 90.0,
        2029: 90.0,
        2030: 90.0,
        2031: 95.0,
    },
}

# Recycling efficiency targets by chemistry and year (%).
RECYCLING_EFFICIENCY_TARGETS: Dict[str, Dict[int, float]] = {
    BatteryChemistry.LEAD_ACID.value: {
        2025: 75.0,
        2026: 75.0,
        2027: 75.0,
        2028: 75.0,
        2029: 75.0,
        2030: 80.0,
        2031: 80.0,
    },
    BatteryChemistry.LITHIUM_ION.value: {
        2025: 65.0,
        2026: 65.0,
        2027: 65.0,
        2028: 65.0,
        2029: 65.0,
        2030: 70.0,
        2031: 70.0,
    },
    BatteryChemistry.NICKEL_METAL_HYDRIDE.value: {
        2025: 65.0,
        2026: 65.0,
        2027: 65.0,
        2028: 65.0,
        2029: 65.0,
        2030: 70.0,
        2031: 70.0,
    },
    BatteryChemistry.OTHER.value: {
        2025: 50.0,
        2026: 50.0,
        2027: 50.0,
        2028: 50.0,
        2029: 50.0,
        2030: 50.0,
        2031: 50.0,
    },
}

# Phase descriptions for reporting.
EOL_PHASE_DESCRIPTIONS: Dict[str, str] = {
    EOLPhase.COLLECTION.value: (
        "Collection of waste batteries from end-users, collection "
        "points, and take-back schemes"
    ),
    EOLPhase.DISMANTLING.value: (
        "Dismantling of battery packs into modules and cells, "
        "including removal of hazardous components"
    ),
    EOLPhase.RECYCLING.value: (
        "Processing of battery materials through hydrometallurgical "
        "or pyrometallurgical treatment"
    ),
    EOLPhase.RECOVERY.value: (
        "Recovery of critical raw materials (cobalt, lithium, "
        "nickel, copper, lead) from recycled battery fractions"
    ),
    EOLPhase.DISPOSAL.value: (
        "Safe disposal of non-recoverable residues in accordance "
        "with Directive 1999/31/EC (Landfill Directive)"
    ),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CollectionData(BaseModel):
    """Collection data for a battery category in a reporting year.

    Tracks the number of batteries placed on the market versus
    those collected, calculating the collection rate against
    the regulatory target.
    """
    collection_id: str = Field(
        default_factory=_new_uuid,
        description="Unique collection record identifier",
    )
    category: BatteryCategory = Field(
        ...,
        description="Battery category",
    )
    batteries_placed: float = Field(
        ...,
        description="Weight (kg) of batteries placed on market in period",
        ge=0.0,
    )
    batteries_collected: float = Field(
        ...,
        description="Weight (kg) of waste batteries collected in period",
        ge=0.0,
    )
    collection_rate_pct: float = Field(
        default=0.0,
        description="Calculated collection rate (%)",
        ge=0.0,
    )
    target_rate_pct: float = Field(
        default=0.0,
        description="Applicable regulatory target (%)",
        ge=0.0,
    )
    meets_target: bool = Field(
        default=False,
        description="Whether the collection rate meets the target",
    )
    gap_pct: float = Field(
        default=0.0,
        description="Gap between actual and target rate (negative=shortfall)",
    )
    year: int = Field(
        ...,
        description="Reporting year",
        ge=2024,
        le=2050,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class RecyclingData(BaseModel):
    """Recycling efficiency data for a battery processing operation.

    Tracks input weight versus output weight to calculate
    recycling efficiency against regulatory targets.
    """
    recycling_id: str = Field(
        default_factory=_new_uuid,
        description="Unique recycling record identifier",
    )
    chemistry: BatteryChemistry = Field(
        default=BatteryChemistry.LITHIUM_ION,
        description="Battery chemistry type",
    )
    input_weight_kg: float = Field(
        ...,
        description="Total weight of waste batteries input (kg)",
        ge=0.0,
    )
    output_weight_kg: float = Field(
        ...,
        description="Total weight of recycled output (kg)",
        ge=0.0,
    )
    efficiency_pct: float = Field(
        default=0.0,
        description="Calculated recycling efficiency (%)",
        ge=0.0,
    )
    target_pct: float = Field(
        default=0.0,
        description="Applicable regulatory target (%)",
        ge=0.0,
    )
    meets_target: bool = Field(
        default=False,
        description="Whether the efficiency meets the target",
    )
    gap_pct: float = Field(
        default=0.0,
        description="Gap between actual and target (%)",
    )
    year: int = Field(
        default=2027,
        description="Reporting year",
        ge=2024,
        le=2050,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class MaterialRecoveryData(BaseModel):
    """Material recovery data for a specific critical raw material.

    Tracks input versus recovered quantities for each material
    against the regulatory recovery targets.
    """
    recovery_id: str = Field(
        default_factory=_new_uuid,
        description="Unique recovery record identifier",
    )
    material: RecoveryMaterial = Field(
        ...,
        description="Critical raw material",
    )
    input_kg: float = Field(
        ...,
        description="Weight of material in battery input (kg)",
        ge=0.0,
    )
    recovered_kg: float = Field(
        ...,
        description="Weight of material recovered (kg)",
        ge=0.0,
    )
    recovery_pct: float = Field(
        default=0.0,
        description="Calculated recovery rate (%)",
        ge=0.0,
    )
    target_pct: float = Field(
        default=0.0,
        description="Applicable regulatory target (%)",
        ge=0.0,
    )
    meets_target: bool = Field(
        default=False,
        description="Whether recovery meets the target",
    )
    gap_pct: float = Field(
        default=0.0,
        description="Gap between actual and target (%)",
    )
    year: int = Field(
        default=2027,
        description="Reporting year",
        ge=2024,
        le=2050,
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class SecondLifeAssessment(BaseModel):
    """Assessment of second-life potential for a battery.

    Evaluates whether a battery is suitable for second-life
    applications before recycling (e.g., stationary energy
    storage from retired EV batteries).
    """
    battery_id: str = Field(
        default="",
        description="Battery identifier",
    )
    soh_pct: float = Field(
        default=0.0,
        description="State of health (%)",
        ge=0.0,
        le=100.0,
    )
    suitable_for_second_life: bool = Field(
        default=False,
        description="Whether battery is suitable for second-life use",
    )
    estimated_remaining_life_years: float = Field(
        default=0.0,
        description="Estimated remaining useful life in second-life (years)",
        ge=0.0,
    )
    recommended_application: str = Field(
        default="",
        description="Recommended second-life application",
    )
    assessment_notes: str = Field(
        default="",
        description="Additional assessment notes",
        max_length=2000,
    )

class EOLResult(BaseModel):
    """Complete end-of-life assessment result.

    Contains collection compliance, recycling efficiency,
    material recovery results, second-life assessment, and
    overall compliance determination with recommendations.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of assessment (UTC)",
    )
    collection_compliance: List[CollectionData] = Field(
        default_factory=list,
        description="Collection rate compliance results per category",
    )
    recycling_compliance: List[RecyclingData] = Field(
        default_factory=list,
        description="Recycling efficiency compliance results",
    )
    material_recovery_results: List[MaterialRecoveryData] = Field(
        default_factory=list,
        description="Material recovery compliance results per material",
    )
    second_life_assessment: Optional[SecondLifeAssessment] = Field(
        default=None,
        description="Second-life suitability assessment",
    )
    overall_compliance: bool = Field(
        default=False,
        description="Whether all EOL targets are met",
    )
    collection_compliant: bool = Field(
        default=False,
        description="Whether all collection targets are met",
    )
    recycling_compliant: bool = Field(
        default=False,
        description="Whether all recycling efficiency targets are met",
    )
    recovery_compliant: bool = Field(
        default=False,
        description="Whether all material recovery targets are met",
    )
    total_collected_kg: float = Field(
        default=0.0,
        description="Total weight of batteries collected (kg)",
    )
    total_recycled_kg: float = Field(
        default=0.0,
        description="Total weight of batteries recycled (kg)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations",
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

class EndOfLifeEngine:
    """End-of-life management engine per EU Battery Regulation Art 56-71.

    Provides deterministic, zero-hallucination tracking of:
    - Collection rate compliance per battery category
    - Recycling efficiency against chemistry-specific targets
    - Material recovery rates for critical raw materials
    - Second-life suitability assessment
    - Progressive target compliance by year
    - Gap analysis and corrective recommendations

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = EndOfLifeEngine()
        collection = CollectionData(
            category=BatteryCategory.PORTABLE,
            batteries_placed=10000.0,
            batteries_collected=6500.0,
            year=2027,
        )
        result = engine.assess_end_of_life(
            collection_data=[collection],
            recycling_data=[],
            recovery_data=[],
            year=2027,
        )
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise EndOfLifeEngine."""
        logger.info(
            "EndOfLifeEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Full End-of-Life Assessment                                          #
    # ------------------------------------------------------------------ #

    def assess_end_of_life(
        self,
        collection_data: List[CollectionData],
        recycling_data: List[RecyclingData],
        recovery_data: List[MaterialRecoveryData],
        year: int = 2027,
        second_life_battery_id: Optional[str] = None,
        second_life_soh_pct: Optional[float] = None,
    ) -> EOLResult:
        """Perform a complete end-of-life compliance assessment.

        Evaluates collection rates, recycling efficiency, and
        material recovery against the applicable regulatory targets
        for the reporting year.

        Args:
            collection_data: List of CollectionData per category.
            recycling_data: List of RecyclingData per chemistry.
            recovery_data: List of MaterialRecoveryData per material.
            year: Reporting year for target lookup.
            second_life_battery_id: Optional battery ID for second-life check.
            second_life_soh_pct: Optional SoH for second-life assessment.

        Returns:
            EOLResult with complete compliance assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing end-of-life compliance for year %d: "
            "%d collection, %d recycling, %d recovery records",
            year,
            len(collection_data),
            len(recycling_data),
            len(recovery_data),
        )

        # Assess collection targets
        assessed_collections = self.check_collection_targets(
            collection_data, year
        )

        # Assess recycling efficiency
        assessed_recycling = self.check_recycling_efficiency(
            recycling_data, year
        )

        # Assess material recovery
        assessed_recovery = self.check_material_recovery(
            recovery_data, year
        )

        # Second-life assessment
        second_life = None
        if (
            second_life_battery_id is not None
            and second_life_soh_pct is not None
        ):
            second_life = self.assess_second_life(
                battery_id=second_life_battery_id,
                soh_pct=second_life_soh_pct,
            )

        # Overall compliance
        collection_compliant = all(
            c.meets_target for c in assessed_collections
        ) if assessed_collections else True
        recycling_compliant = all(
            r.meets_target for r in assessed_recycling
        ) if assessed_recycling else True
        recovery_compliant = all(
            r.meets_target for r in assessed_recovery
        ) if assessed_recovery else True

        overall = (
            collection_compliant
            and recycling_compliant
            and recovery_compliant
        )

        # Totals
        total_collected = sum(
            c.batteries_collected for c in assessed_collections
        )
        total_recycled = sum(
            r.output_weight_kg for r in assessed_recycling
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            assessed_collections,
            assessed_recycling,
            assessed_recovery,
            year,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EOLResult(
            collection_compliance=assessed_collections,
            recycling_compliance=assessed_recycling,
            material_recovery_results=assessed_recovery,
            second_life_assessment=second_life,
            overall_compliance=overall,
            collection_compliant=collection_compliant,
            recycling_compliant=recycling_compliant,
            recovery_compliant=recovery_compliant,
            total_collected_kg=_round2(total_collected),
            total_recycled_kg=_round2(total_recycled),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "EOL assessment complete: collection=%s, recycling=%s, "
            "recovery=%s, overall=%s in %.3f ms",
            collection_compliant,
            recycling_compliant,
            recovery_compliant,
            overall,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Collection Target Checking                                           #
    # ------------------------------------------------------------------ #

    def check_collection_targets(
        self,
        collection_data: List[CollectionData],
        year: int = 2027,
    ) -> List[CollectionData]:
        """Check collection rates against regulatory targets.

        Calculates the collection rate for each category and
        compares it to the applicable target for the reporting year.

        Args:
            collection_data: List of CollectionData records.
            year: Reporting year for target lookup.

        Returns:
            List of CollectionData with targets and compliance flags set.
        """
        assessed: List[CollectionData] = []

        for data in collection_data:
            # Calculate collection rate
            rate = _round2(
                _safe_divide(
                    data.batteries_collected,
                    data.batteries_placed,
                    0.0,
                ) * 100.0
            )

            # Look up target
            target = self._get_collection_target(
                data.category, year if data.year == 0 else data.year
            )

            meets = rate >= target
            gap = _round2(rate - target)

            assessed_record = data.model_copy(update={
                "collection_rate_pct": rate,
                "target_rate_pct": target,
                "meets_target": meets,
                "gap_pct": gap,
                "year": data.year if data.year > 0 else year,
            })
            assessed_record.provenance_hash = _compute_hash(assessed_record)
            assessed.append(assessed_record)

            logger.debug(
                "Collection %s/%d: rate=%.1f%%, target=%.1f%%, meets=%s",
                data.category.value,
                data.year if data.year > 0 else year,
                rate,
                target,
                meets,
            )

        return assessed

    # ------------------------------------------------------------------ #
    # Recycling Efficiency Checking                                        #
    # ------------------------------------------------------------------ #

    def check_recycling_efficiency(
        self,
        recycling_data: List[RecyclingData],
        year: int = 2027,
    ) -> List[RecyclingData]:
        """Check recycling efficiency against regulatory targets.

        Calculates recycling efficiency (output/input) for each
        record and compares to the applicable chemistry-specific
        target for the reporting year.

        Args:
            recycling_data: List of RecyclingData records.
            year: Reporting year for target lookup.

        Returns:
            List of RecyclingData with targets and compliance flags set.
        """
        assessed: List[RecyclingData] = []

        for data in recycling_data:
            # Calculate efficiency
            efficiency = _round2(
                _safe_divide(
                    data.output_weight_kg,
                    data.input_weight_kg,
                    0.0,
                ) * 100.0
            )

            # Look up target
            effective_year = data.year if data.year > 0 else year
            target = self._get_recycling_target(
                data.chemistry, effective_year
            )

            meets = efficiency >= target
            gap = _round2(efficiency - target)

            assessed_record = data.model_copy(update={
                "efficiency_pct": efficiency,
                "target_pct": target,
                "meets_target": meets,
                "gap_pct": gap,
                "year": effective_year,
            })
            assessed_record.provenance_hash = _compute_hash(assessed_record)
            assessed.append(assessed_record)

            logger.debug(
                "Recycling %s/%d: efficiency=%.1f%%, target=%.1f%%, meets=%s",
                data.chemistry.value,
                effective_year,
                efficiency,
                target,
                meets,
            )

        return assessed

    # ------------------------------------------------------------------ #
    # Material Recovery Checking                                           #
    # ------------------------------------------------------------------ #

    def check_material_recovery(
        self,
        recovery_data: List[MaterialRecoveryData],
        year: int = 2027,
    ) -> List[MaterialRecoveryData]:
        """Check material recovery rates against regulatory targets.

        Calculates recovery percentage for each material and
        compares to the applicable target for the reporting year.

        Args:
            recovery_data: List of MaterialRecoveryData records.
            year: Reporting year for target lookup.

        Returns:
            List of MaterialRecoveryData with targets and compliance set.
        """
        assessed: List[MaterialRecoveryData] = []

        for data in recovery_data:
            # Calculate recovery rate
            recovery = _round2(
                _safe_divide(
                    data.recovered_kg,
                    data.input_kg,
                    0.0,
                ) * 100.0
            )

            # Look up target
            effective_year = data.year if data.year > 0 else year
            target = self._get_recovery_target(
                data.material, effective_year
            )

            meets = recovery >= target
            gap = _round2(recovery - target)

            assessed_record = data.model_copy(update={
                "recovery_pct": recovery,
                "target_pct": target,
                "meets_target": meets,
                "gap_pct": gap,
                "year": effective_year,
            })
            assessed_record.provenance_hash = _compute_hash(assessed_record)
            assessed.append(assessed_record)

            logger.debug(
                "Recovery %s/%d: rate=%.1f%%, target=%.1f%%, meets=%s",
                data.material.value,
                effective_year,
                recovery,
                target,
                meets,
            )

        return assessed

    # ------------------------------------------------------------------ #
    # Second-Life Assessment                                               #
    # ------------------------------------------------------------------ #

    def assess_second_life(
        self,
        battery_id: str,
        soh_pct: float,
        category: BatteryCategory = BatteryCategory.EV,
    ) -> SecondLifeAssessment:
        """Assess whether a battery is suitable for second-life use.

        Batteries with state-of-health above 70% are generally
        considered suitable for second-life applications such as
        stationary energy storage.

        Thresholds:
            SoH >= 80%: Suitable, estimated 8+ years remaining
            SoH 70-79%: Suitable, estimated 4-7 years remaining
            SoH 50-69%: Marginal, estimated 1-3 years remaining
            SoH < 50%: Not suitable, proceed to recycling

        Args:
            battery_id: Battery identifier.
            soh_pct: State of health percentage (0-100).
            category: Battery category.

        Returns:
            SecondLifeAssessment with suitability determination.
        """
        if soh_pct >= 80.0:
            suitable = True
            remaining = 8.0
            application = "Stationary energy storage (grid-scale or commercial)"
            notes = (
                "Battery has excellent remaining capacity. Suitable for "
                "demanding second-life applications with expected 8+ years "
                "of useful service."
            )
        elif soh_pct >= 70.0:
            suitable = True
            remaining = 5.0
            application = "Stationary energy storage (residential or backup)"
            notes = (
                "Battery has good remaining capacity. Suitable for "
                "moderate-duty second-life applications with expected "
                "4-7 years of useful service."
            )
        elif soh_pct >= 50.0:
            suitable = False
            remaining = 2.0
            application = "Limited second-life use (low-duty backup only)"
            notes = (
                "Battery has marginal remaining capacity. Second-life "
                "use is possible but economically questionable. Consider "
                "recycling unless a specific low-duty application exists."
            )
        else:
            suitable = False
            remaining = 0.0
            application = "Not suitable for second life - proceed to recycling"
            notes = (
                "Battery state of health is below 50%. Not suitable for "
                "second-life applications. Proceed directly to recycling "
                "and material recovery per Art 56-71."
            )

        return SecondLifeAssessment(
            battery_id=battery_id,
            soh_pct=soh_pct,
            suitable_for_second_life=suitable,
            estimated_remaining_life_years=remaining,
            recommended_application=application,
            assessment_notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Target Lookup Utilities                                              #
    # ------------------------------------------------------------------ #

    def get_collection_target(
        self, category: BatteryCategory, year: int
    ) -> Dict[str, Any]:
        """Get the applicable collection target for a category and year.

        Args:
            category: Battery category.
            year: Target year.

        Returns:
            Dict with target details.
        """
        target = self._get_collection_target(category, year)
        return {
            "category": category.value,
            "year": year,
            "target_pct": target,
            "is_take_back": category in (
                BatteryCategory.EV,
                BatteryCategory.INDUSTRIAL,
                BatteryCategory.SLI,
            ),
            "provenance_hash": _compute_hash({
                "category": category.value,
                "year": year,
                "target": target,
            }),
        }

    def get_recovery_target(
        self, material: RecoveryMaterial, year: int
    ) -> Dict[str, Any]:
        """Get the applicable material recovery target.

        Args:
            material: Critical raw material.
            year: Target year.

        Returns:
            Dict with target details.
        """
        target = self._get_recovery_target(material, year)
        return {
            "material": material.value,
            "year": year,
            "target_pct": target,
            "provenance_hash": _compute_hash({
                "material": material.value,
                "year": year,
                "target": target,
            }),
        }

    def get_recycling_target(
        self, chemistry: BatteryChemistry, year: int
    ) -> Dict[str, Any]:
        """Get the applicable recycling efficiency target.

        Args:
            chemistry: Battery chemistry type.
            year: Target year.

        Returns:
            Dict with target details.
        """
        target = self._get_recycling_target(chemistry, year)
        return {
            "chemistry": chemistry.value,
            "year": year,
            "target_pct": target,
            "provenance_hash": _compute_hash({
                "chemistry": chemistry.value,
                "year": year,
                "target": target,
            }),
        }

    def get_all_targets_for_year(
        self, year: int
    ) -> Dict[str, Any]:
        """Get all applicable targets for a reporting year.

        Args:
            year: Reporting year.

        Returns:
            Dict with collection, recycling, and recovery targets.
        """
        targets: Dict[str, Any] = {
            "year": year,
            "collection": {},
            "recycling": {},
            "recovery": {},
        }

        for category in BatteryCategory:
            target = self._get_collection_target(category, year)
            targets["collection"][category.value] = target

        for chemistry in BatteryChemistry:
            target = self._get_recycling_target(chemistry, year)
            targets["recycling"][chemistry.value] = target

        for material in RecoveryMaterial:
            target = self._get_recovery_target(material, year)
            targets["recovery"][material.value] = target

        targets["provenance_hash"] = _compute_hash(targets)
        return targets

    # ------------------------------------------------------------------ #
    # Gap Analysis                                                         #
    # ------------------------------------------------------------------ #

    def gap_analysis(
        self, result: EOLResult
    ) -> Dict[str, Any]:
        """Perform a gap analysis on an EOL assessment result.

        Identifies all areas where targets are not met and
        quantifies the gaps.

        Args:
            result: EOLResult to analyse.

        Returns:
            Dict with detailed gap analysis.
        """
        gaps: Dict[str, Any] = {
            "total_gaps": 0,
            "collection_gaps": [],
            "recycling_gaps": [],
            "recovery_gaps": [],
        }

        for c in result.collection_compliance:
            if not c.meets_target:
                gaps["collection_gaps"].append({
                    "category": c.category.value,
                    "year": c.year,
                    "actual_pct": c.collection_rate_pct,
                    "target_pct": c.target_rate_pct,
                    "gap_pct": c.gap_pct,
                    "additional_kg_needed": _round2(
                        (c.target_rate_pct - c.collection_rate_pct)
                        / 100.0 * c.batteries_placed
                    ) if c.batteries_placed > 0 else 0.0,
                })
                gaps["total_gaps"] += 1

        for r in result.recycling_compliance:
            if not r.meets_target:
                gaps["recycling_gaps"].append({
                    "chemistry": r.chemistry.value,
                    "year": r.year,
                    "actual_pct": r.efficiency_pct,
                    "target_pct": r.target_pct,
                    "gap_pct": r.gap_pct,
                })
                gaps["total_gaps"] += 1

        for m in result.material_recovery_results:
            if not m.meets_target:
                gaps["recovery_gaps"].append({
                    "material": m.material.value,
                    "year": m.year,
                    "actual_pct": m.recovery_pct,
                    "target_pct": m.target_pct,
                    "gap_pct": m.gap_pct,
                    "additional_kg_needed": _round2(
                        (m.target_pct - m.recovery_pct)
                        / 100.0 * m.input_kg
                    ) if m.input_kg > 0 else 0.0,
                })
                gaps["total_gaps"] += 1

        gaps["provenance_hash"] = _compute_hash(gaps)
        return gaps

    # ------------------------------------------------------------------ #
    # Summary Utilities                                                    #
    # ------------------------------------------------------------------ #

    def get_eol_summary(self, result: EOLResult) -> Dict[str, Any]:
        """Return a structured summary of an EOL assessment.

        Args:
            result: EOLResult to summarise.

        Returns:
            Dict with summary statistics.
        """
        return {
            "overall_compliance": result.overall_compliance,
            "collection_compliant": result.collection_compliant,
            "recycling_compliant": result.recycling_compliant,
            "recovery_compliant": result.recovery_compliant,
            "collection_records": len(result.collection_compliance),
            "recycling_records": len(result.recycling_compliance),
            "recovery_records": len(result.material_recovery_results),
            "total_collected_kg": result.total_collected_kg,
            "total_recycled_kg": result.total_recycled_kg,
            "has_second_life_assessment": (
                result.second_life_assessment is not None
            ),
            "second_life_suitable": (
                result.second_life_assessment.suitable_for_second_life
                if result.second_life_assessment else None
            ),
            "recommendation_count": len(result.recommendations),
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private: Target Lookups                                              #
    # ------------------------------------------------------------------ #

    def _get_collection_target(
        self, category: BatteryCategory, year: int
    ) -> float:
        """Look up the collection target for a category and year.

        If the exact year is not in the table, the nearest
        applicable year is used.

        Args:
            category: Battery category.
            year: Target year.

        Returns:
            Target collection rate (%).
        """
        targets = COLLECTION_TARGETS.get(category.value, {})
        if not targets:
            return 0.0

        if year in targets:
            return targets[year]

        # Find the nearest applicable year (latest year <= requested)
        applicable_years = sorted(
            y for y in targets.keys() if y <= year
        )
        if applicable_years:
            return targets[applicable_years[-1]]

        # If year is before any target, return the earliest target
        earliest = min(targets.keys())
        return targets[earliest]

    def _get_recycling_target(
        self, chemistry: BatteryChemistry, year: int
    ) -> float:
        """Look up the recycling efficiency target.

        Args:
            chemistry: Battery chemistry.
            year: Target year.

        Returns:
            Target recycling efficiency (%).
        """
        targets = RECYCLING_EFFICIENCY_TARGETS.get(
            chemistry.value, {}
        )
        if not targets:
            return 50.0

        if year in targets:
            return targets[year]

        applicable_years = sorted(
            y for y in targets.keys() if y <= year
        )
        if applicable_years:
            return targets[applicable_years[-1]]

        earliest = min(targets.keys())
        return targets[earliest]

    def _get_recovery_target(
        self, material: RecoveryMaterial, year: int
    ) -> float:
        """Look up the material recovery target.

        Args:
            material: Critical raw material.
            year: Target year.

        Returns:
            Target recovery rate (%).
        """
        targets = MATERIAL_RECOVERY_TARGETS.get(material.value, {})
        if not targets:
            return 0.0

        if year in targets:
            return targets[year]

        applicable_years = sorted(
            y for y in targets.keys() if y <= year
        )
        if applicable_years:
            return targets[applicable_years[-1]]

        # If year is before any target, return 0 (not yet applicable)
        earliest = min(targets.keys())
        if year < earliest:
            return 0.0

        return targets[earliest]

    # ------------------------------------------------------------------ #
    # Private: Recommendations                                             #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        collections: List[CollectionData],
        recycling: List[RecyclingData],
        recovery: List[MaterialRecoveryData],
        year: int,
    ) -> List[str]:
        """Generate actionable recommendations based on assessment.

        Args:
            collections: Assessed collection data.
            recycling: Assessed recycling data.
            recovery: Assessed recovery data.
            year: Reporting year.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Collection gaps
        collection_gaps = [c for c in collections if not c.meets_target]
        if collection_gaps:
            for gap in collection_gaps:
                shortfall = abs(gap.gap_pct)
                recommendations.append(
                    f"Collection gap for {gap.category.value} batteries: "
                    f"actual {gap.collection_rate_pct}% vs target "
                    f"{gap.target_rate_pct}% (shortfall: {shortfall}%). "
                    f"Expand collection infrastructure and awareness."
                )

        # Recycling efficiency gaps
        recycling_gaps = [r for r in recycling if not r.meets_target]
        if recycling_gaps:
            for gap in recycling_gaps:
                shortfall = abs(gap.gap_pct)
                recommendations.append(
                    f"Recycling efficiency gap for {gap.chemistry.value}: "
                    f"actual {gap.efficiency_pct}% vs target "
                    f"{gap.target_pct}% (shortfall: {shortfall}%). "
                    f"Invest in recycling technology upgrades."
                )

        # Material recovery gaps
        recovery_gaps = [r for r in recovery if not r.meets_target]
        if recovery_gaps:
            for gap in recovery_gaps:
                shortfall = abs(gap.gap_pct)
                recommendations.append(
                    f"Material recovery gap for {gap.material.value}: "
                    f"actual {gap.recovery_pct}% vs target "
                    f"{gap.target_pct}% (shortfall: {shortfall}%). "
                    f"Improve hydrometallurgical or pyrometallurgical "
                    f"recovery processes."
                )

        # Lithium-specific (harder to recover)
        lithium_records = [
            r for r in recovery
            if r.material == RecoveryMaterial.LITHIUM
        ]
        lithium_gaps = [r for r in lithium_records if not r.meets_target]
        if lithium_gaps:
            recommendations.append(
                "Lithium recovery is below target. Lithium recovery "
                "is technically challenging; consider partnerships "
                "with specialised lithium recyclers or investment in "
                "direct lithium recovery technologies."
            )

        # Future target warnings
        if year < 2031:
            recommendations.append(
                f"Note: Targets increase in future years. Ensure "
                f"infrastructure and technology are being prepared "
                f"for 2031 targets (e.g., Li recovery 80%, "
                f"Co/Ni/Cu/Pb recovery 95%)."
            )

        # All targets met
        if not collection_gaps and not recycling_gaps and not recovery_gaps:
            recommendations.append(
                f"All end-of-life targets for year {year} are met. "
                f"Continue monitoring and prepare for increased "
                f"targets in subsequent years."
            )

        return recommendations
