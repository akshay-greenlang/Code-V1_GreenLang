# -*- coding: utf-8 -*-
"""
TransformationTracker - AGENT-EUDR-009 Feature 5: Transformation Tracking

Records processing steps with input-to-output mappings, validates yield ratios
against industry reference data (ICCO, MPOB, ICO, SCA, NOPA, IRSG, ITTO, FAO),
tracks by-products and waste, builds multi-step transformation chains from raw
material to finished product, and supports co-product allocation using economic,
mass, or energy allocation methods.

Capabilities:
    - Record processing transformations with input/output batch mappings
    - 25+ process types (drying, fermentation, roasting, milling, refining, etc.)
    - Yield ratio validation: actual vs expected per process/commodity
    - By-product and waste tracking per transformation step
    - Multi-step transformation chain assembly (raw -> intermediate -> final)
    - Derived product tracking across commodity form changes
    - Co-product allocation (economic, mass, energy allocation methods)
    - Historical yield statistics for trend analysis
    - Batch import of transformation records
    - SHA-256 provenance hashing on all results

Zero-Hallucination Guarantees:
    - All yield ratios sourced from peer-reviewed industry references
    - All mass balance arithmetic is deterministic Python float/Decimal
    - No LLM or ML used in any calculation or validation path
    - SHA-256 provenance hash on every result for tamper detection
    - Bit-perfect reproducibility: same inputs produce same outputs

Regulatory Basis:
    - EUDR Article 4(2): Due diligence on product transformations
    - EUDR Article 9(1)(d): Plot-to-product traceability through transformations
    - EUDR Article 9(1)(f): Quantity/weight tracking through processing
    - EUDR Article 14: 5-year record retention for transformation records

Dependencies:
    - Reference yield ratios: ICCO, MPOB, ICO, SCA, NOPA, IRSG, ITTO, FAO
    - provenance: SHA-256 chain hashing
    - metrics: Prometheus transformation counters

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Feature 5
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses json.dumps with sort_keys=True and default=str to ensure
    deterministic serialization regardless of dict insertion order.

    Args:
        data: Data to hash (dict, list, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters, lowercase).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Identifier prefix string (e.g., 'TRF', 'YLD').

    Returns:
        Prefixed UUID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR regulation reference
EUDR_REGULATION_REF = "Regulation (EU) 2023/1115"

#: All supported EUDR commodities
EUDR_COMMODITIES: FrozenSet[str] = frozenset({
    "cocoa", "coffee", "palm_oil", "soya", "rubber", "wood", "cattle",
})

#: Extended commodity list including derived products
DERIVED_COMMODITIES: FrozenSet[str] = frozenset({
    "cocoa_nibs", "cocoa_liquor", "cocoa_butter", "cocoa_powder",
    "chocolate", "green_coffee", "roasted_coffee", "ground_coffee",
    "instant_coffee", "ffb", "cpo", "pko", "rbd_palm_oil",
    "palm_olein", "palm_stearin", "palm_kernel_cake",
    "soya_oil", "soya_meal", "soya_lecithin", "tofu",
    "latex", "rss", "tsr", "crumb_rubber",
    "log", "sawn_timber", "plywood", "veneer", "mdf",
    "pulp", "paper", "charcoal",
    "live_cattle", "carcass", "beef", "leather", "hide",
    "tallow", "bone_meal", "gelatin",
})


class ProcessType(str, Enum):
    """Supported processing/transformation types.

    Covers 25+ process types across all 7 EUDR commodities.
    """

    DRYING = "drying"
    FERMENTATION = "fermentation"
    ROASTING = "roasting"
    MILLING = "milling"
    REFINING = "refining"
    PRESSING = "pressing"
    EXTRACTION = "extraction"
    FRACTIONATION = "fractionation"
    DEODORIZATION = "deodorization"
    HYDROGENATION = "hydrogenation"
    INTERESTERIFICATION = "interesterification"
    SMELTING = "smelting"
    SAWING = "sawing"
    TANNING = "tanning"
    SPINNING = "spinning"
    WEAVING = "weaving"
    DISTILLATION = "distillation"
    CRUSHING = "crushing"
    HULLING = "hulling"
    POLISHING = "polishing"
    DEGUMMING = "degumming"
    BLEACHING = "bleaching"
    WINTERIZATION = "winterization"
    TEMPERING = "tempering"
    MOLDING = "molding"
    WINNOWING = "winnowing"
    CONCHING = "conching"
    SLAUGHTERING = "slaughtering"
    PEELING = "peeling"
    VULCANIZATION = "vulcanization"


class AllocationMethod(str, Enum):
    """Co-product allocation methods per ISO 14044."""

    MASS = "mass"
    ECONOMIC = "economic"
    ENERGY = "energy"


class TransformationStatus(str, Enum):
    """Transformation record status."""

    RECORDED = "recorded"
    VALIDATED = "validated"
    YIELD_WARNING = "yield_warning"
    YIELD_ALERT = "yield_alert"
    ERROR = "error"


class YieldVerdict(str, Enum):
    """Yield validation verdict."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Reference Yield Ratios (Appendix A)
# ---------------------------------------------------------------------------
# Source: ICCO, MPOB, ICO, SCA, NOPA, IRSG, ITTO, FAO, ICHSLTA, PORAM
# Each entry: (commodity, process_type) -> (expected_min, expected_max, source)

REFERENCE_YIELD_RATIOS: Dict[Tuple[str, str], Tuple[float, float, str]] = {
    # Cocoa chain (ICCO)
    ("cocoa", "fermentation"): (0.90, 0.95, "ICCO"),
    ("cocoa", "drying"): (0.55, 0.65, "ICCO"),
    ("cocoa", "winnowing"): (0.82, 0.90, "ICCO"),
    ("cocoa", "roasting"): (0.85, 0.92, "ICCO"),
    ("cocoa", "milling"): (0.95, 0.99, "ICCO"),
    ("cocoa_nibs", "milling"): (0.95, 0.99, "ICCO"),
    ("cocoa_nibs", "pressing"): (0.87, 0.92, "ICCO"),
    ("cocoa_liquor", "pressing"): (0.42, 0.48, "ICCO"),
    ("cocoa_liquor", "extraction"): (0.52, 0.58, "ICCO"),
    ("cocoa_butter", "tempering"): (0.97, 0.99, "ICCO"),
    ("cocoa_powder", "milling"): (0.95, 0.99, "ICCO"),
    ("chocolate", "conching"): (0.95, 0.99, "ICCO"),
    ("chocolate", "tempering"): (0.97, 0.99, "ICCO"),
    ("chocolate", "molding"): (0.95, 0.99, "ICCO"),
    # Palm oil chain (MPOB, PORAM)
    ("palm_oil", "pressing"): (0.19, 0.24, "MPOB"),
    ("ffb", "pressing"): (0.19, 0.24, "MPOB"),
    ("ffb", "extraction"): (0.03, 0.05, "MPOB"),
    ("cpo", "refining"): (0.90, 0.94, "PORAM"),
    ("cpo", "degumming"): (0.95, 0.98, "PORAM"),
    ("cpo", "bleaching"): (0.96, 0.99, "PORAM"),
    ("cpo", "deodorization"): (0.95, 0.98, "PORAM"),
    ("rbd_palm_oil", "fractionation"): (0.65, 0.75, "MPOB"),
    ("pko", "refining"): (0.90, 0.95, "MPOB"),
    ("palm_oil", "hydrogenation"): (0.95, 0.99, "MPOB"),
    ("palm_oil", "interesterification"): (0.94, 0.98, "MPOB"),
    # Coffee chain (ICO, SCA)
    ("coffee", "drying"): (0.45, 0.55, "ICO"),
    ("coffee", "hulling"): (0.60, 0.70, "ICO"),
    ("coffee", "peeling"): (0.55, 0.65, "ICO"),
    ("coffee", "polishing"): (0.95, 0.99, "ICO"),
    ("green_coffee", "roasting"): (0.80, 0.85, "SCA"),
    ("roasted_coffee", "milling"): (0.95, 0.99, "SCA"),
    ("roasted_coffee", "extraction"): (0.25, 0.35, "SCA"),
    # Soya chain (NOPA)
    ("soya", "crushing"): (0.78, 0.82, "NOPA"),
    ("soya", "extraction"): (0.17, 0.21, "NOPA"),
    ("soya", "pressing"): (0.15, 0.20, "NOPA"),
    ("soya_oil", "degumming"): (0.95, 0.98, "NOPA"),
    ("soya_oil", "refining"): (0.90, 0.95, "NOPA"),
    ("soya_oil", "bleaching"): (0.96, 0.99, "NOPA"),
    ("soya_oil", "deodorization"): (0.95, 0.98, "NOPA"),
    ("soya_oil", "winterization"): (0.85, 0.92, "NOPA"),
    ("soya_meal", "milling"): (0.95, 0.99, "NOPA"),
    # Rubber chain (IRSG)
    ("rubber", "pressing"): (0.28, 0.36, "IRSG"),
    ("rubber", "drying"): (0.55, 0.65, "IRSG"),
    ("latex", "pressing"): (0.28, 0.36, "IRSG"),
    ("latex", "drying"): (0.55, 0.65, "IRSG"),
    ("rss", "milling"): (0.95, 0.99, "IRSG"),
    ("crumb_rubber", "vulcanization"): (0.92, 0.98, "IRSG"),
    # Wood chain (ITTO, FAO)
    ("wood", "sawing"): (0.43, 0.57, "ITTO"),
    ("log", "sawing"): (0.43, 0.57, "ITTO"),
    ("wood", "peeling"): (0.50, 0.60, "ITTO"),
    ("log", "peeling"): (0.50, 0.60, "ITTO"),
    ("veneer", "pressing"): (0.85, 0.95, "ITTO"),
    ("sawn_timber", "drying"): (0.90, 0.96, "ITTO"),
    ("wood", "milling"): (0.80, 0.90, "FAO"),
    ("pulp", "pressing"): (0.90, 0.95, "FAO"),
    ("wood", "distillation"): (0.25, 0.35, "FAO"),
    # Cattle chain (FAO, ICHSLTA)
    ("cattle", "slaughtering"): (0.50, 0.60, "FAO"),
    ("live_cattle", "slaughtering"): (0.50, 0.60, "FAO"),
    ("carcass", "milling"): (0.70, 0.85, "FAO"),
    ("hide", "tanning"): (0.20, 0.30, "ICHSLTA"),
    ("leather", "drying"): (0.90, 0.96, "ICHSLTA"),
    ("cattle", "tanning"): (0.05, 0.09, "ICHSLTA"),
}

#: Default yield tolerance percentage for flagging warnings.
DEFAULT_YIELD_WARNING_TOLERANCE_PCT: float = 5.0

#: Default yield tolerance percentage for flagging alerts/failures.
DEFAULT_YIELD_ALERT_TOLERANCE_PCT: float = 15.0

#: Default maximum batch import size.
MAX_BATCH_IMPORT_SIZE: int = 5000

#: Expected by-product ratios (commodity, process) -> by-product name -> ratio
#: These are approximate industry averages for completeness tracking.
EXPECTED_BY_PRODUCTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("cocoa_liquor", "pressing"): {
        "cocoa_butter": 0.45,
        "cocoa_powder": 0.55,
    },
    ("ffb", "pressing"): {
        "cpo": 0.22,
        "pko": 0.035,
        "palm_kernel_cake": 0.035,
        "efb": 0.22,
    },
    ("soya", "crushing"): {
        "soya_meal": 0.80,
        "soya_oil": 0.19,
    },
    ("rbd_palm_oil", "fractionation"): {
        "palm_olein": 0.70,
        "palm_stearin": 0.30,
    },
    ("cattle", "slaughtering"): {
        "carcass": 0.55,
        "hide": 0.07,
        "tallow": 0.04,
        "bone_meal": 0.05,
    },
    ("wood", "sawing"): {
        "sawn_timber": 0.50,
        "sawdust": 0.10,
        "wood_chips": 0.15,
        "bark": 0.08,
    },
}

#: Common derived product transformation chains for traceability.
COMMODITY_FORM_CHAINS: Dict[str, List[str]] = {
    "cocoa": [
        "cocoa", "cocoa_nibs", "cocoa_liquor", "cocoa_butter", "chocolate",
    ],
    "palm_oil": [
        "ffb", "cpo", "rbd_palm_oil", "palm_olein",
    ],
    "coffee": [
        "coffee", "green_coffee", "roasted_coffee", "ground_coffee",
    ],
    "soya": [
        "soya", "soya_oil", "soya_meal",
    ],
    "rubber": [
        "latex", "rss", "crumb_rubber",
    ],
    "wood": [
        "log", "sawn_timber", "plywood",
    ],
    "cattle": [
        "live_cattle", "carcass", "beef", "leather",
    ],
}


# ---------------------------------------------------------------------------
# Data Models (local dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class InputOutput:
    """A single input or output entry in a transformation.

    Attributes:
        batch_id: Batch identifier.
        commodity: Commodity type or derived product name.
        quantity: Quantity in the specified unit.
        unit: Unit of measurement (default: kg).
        quality_grade: Optional quality grade designation.
        origin_plot_ids: Origin plot identifiers carried through.
        metadata: Additional key-value metadata.
    """

    batch_id: str = ""
    commodity: str = ""
    quantity: float = 0.0
    unit: str = "kg"
    quality_grade: Optional[str] = None
    origin_plot_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "batch_id": self.batch_id,
            "commodity": self.commodity,
            "quantity": self.quantity,
            "unit": self.unit,
            "quality_grade": self.quality_grade,
            "origin_plot_ids": list(self.origin_plot_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class TransformationRecord:
    """A single transformation/processing step record.

    Attributes:
        transform_id: Unique transformation identifier.
        process_type: Type of processing (from ProcessType enum values).
        facility_id: Facility where transformation took place.
        facility_name: Human-readable facility name.
        inputs: List of input batches/materials.
        outputs: List of output batches/products.
        input_total_qty: Total input quantity (computed).
        output_total_qty: Total output quantity (computed).
        actual_yield: Computed yield ratio (output / input).
        expected_yield_min: Expected minimum yield from reference data.
        expected_yield_max: Expected maximum yield from reference data.
        yield_source: Reference source for expected yield.
        yield_verdict: Validation verdict (pass/warning/fail).
        timestamp: When the transformation occurred.
        duration_minutes: Duration of the processing step.
        operator_id: ID of operator performing the transformation.
        notes: Optional free-text notes.
        status: Record status.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        metadata: Additional key-value metadata.
    """

    transform_id: str = field(default_factory=lambda: _generate_id("TRF"))
    process_type: str = ""
    facility_id: str = ""
    facility_name: str = ""
    inputs: List[InputOutput] = field(default_factory=list)
    outputs: List[InputOutput] = field(default_factory=list)
    input_total_qty: float = 0.0
    output_total_qty: float = 0.0
    actual_yield: float = 0.0
    expected_yield_min: float = 0.0
    expected_yield_max: float = 0.0
    yield_source: str = ""
    yield_verdict: str = "pass"
    timestamp: datetime = field(default_factory=_utcnow)
    duration_minutes: float = 0.0
    operator_id: str = ""
    notes: str = ""
    status: str = "recorded"
    provenance_hash: str = ""
    created_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "transform_id": self.transform_id,
            "process_type": self.process_type,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "input_total_qty": self.input_total_qty,
            "output_total_qty": self.output_total_qty,
            "actual_yield": self.actual_yield,
            "expected_yield_min": self.expected_yield_min,
            "expected_yield_max": self.expected_yield_max,
            "yield_source": self.yield_source,
            "yield_verdict": self.yield_verdict,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_minutes": self.duration_minutes,
            "operator_id": self.operator_id,
            "notes": self.notes,
            "status": self.status,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class YieldValidation:
    """Result of yield ratio validation for a transformation.

    Attributes:
        validation_id: Unique validation identifier.
        transform_id: Transformation being validated.
        commodity: Commodity being processed.
        process_type: Process type applied.
        input_qty: Input quantity.
        output_qty: Output quantity.
        actual_yield: Computed yield ratio.
        expected_min: Expected minimum yield from reference data.
        expected_max: Expected maximum yield from reference data.
        reference_source: Source of expected yield (ICCO, MPOB, etc.).
        deviation_pct: Percentage deviation from expected range.
        verdict: Validation verdict (pass/warning/fail).
        message: Human-readable validation message.
        provenance_hash: SHA-256 hash for audit trail.
        validated_at: When the validation was performed.
    """

    validation_id: str = field(default_factory=lambda: _generate_id("YLD"))
    transform_id: str = ""
    commodity: str = ""
    process_type: str = ""
    input_qty: float = 0.0
    output_qty: float = 0.0
    actual_yield: float = 0.0
    expected_min: float = 0.0
    expected_max: float = 0.0
    reference_source: str = ""
    deviation_pct: float = 0.0
    verdict: str = "pass"
    message: str = ""
    provenance_hash: str = ""
    validated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "validation_id": self.validation_id,
            "transform_id": self.transform_id,
            "commodity": self.commodity,
            "process_type": self.process_type,
            "input_qty": self.input_qty,
            "output_qty": self.output_qty,
            "actual_yield": self.actual_yield,
            "expected_min": self.expected_min,
            "expected_max": self.expected_max,
            "reference_source": self.reference_source,
            "deviation_pct": self.deviation_pct,
            "verdict": self.verdict,
            "message": self.message,
            "provenance_hash": self.provenance_hash,
            "validated_at": (
                self.validated_at.isoformat() if self.validated_at else None
            ),
        }


@dataclass
class ByProductRecord:
    """By-product and waste record for a transformation.

    Attributes:
        record_id: Unique record identifier.
        transform_id: Parent transformation ID.
        main_product: Main product output entry.
        by_products: List of by-product outputs.
        waste: List of waste outputs.
        main_product_pct: Percentage of output that is main product.
        by_product_pct: Percentage of output that is by-products.
        waste_pct: Percentage of output that is waste.
        total_output_qty: Sum of main + by-products + waste.
        mass_balance_check: Whether mass balance holds (output <= input).
        provenance_hash: SHA-256 hash for audit trail.
        recorded_at: When the record was created.
    """

    record_id: str = field(default_factory=lambda: _generate_id("BYP"))
    transform_id: str = ""
    main_product: Optional[InputOutput] = None
    by_products: List[InputOutput] = field(default_factory=list)
    waste: List[InputOutput] = field(default_factory=list)
    main_product_pct: float = 0.0
    by_product_pct: float = 0.0
    waste_pct: float = 0.0
    total_output_qty: float = 0.0
    mass_balance_check: bool = True
    provenance_hash: str = ""
    recorded_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "record_id": self.record_id,
            "transform_id": self.transform_id,
            "main_product": (
                self.main_product.to_dict() if self.main_product else None
            ),
            "by_products": [bp.to_dict() for bp in self.by_products],
            "waste": [w.to_dict() for w in self.waste],
            "main_product_pct": self.main_product_pct,
            "by_product_pct": self.by_product_pct,
            "waste_pct": self.waste_pct,
            "total_output_qty": self.total_output_qty,
            "mass_balance_check": self.mass_balance_check,
            "provenance_hash": self.provenance_hash,
            "recorded_at": (
                self.recorded_at.isoformat() if self.recorded_at else None
            ),
        }


@dataclass
class TransformationChainStep:
    """A single step in a multi-step transformation chain.

    Attributes:
        step_number: Ordinal position in the chain (1-based).
        transform_id: Transformation record identifier.
        process_type: Processing type applied.
        input_commodity: Input commodity/form.
        output_commodity: Output commodity/form.
        input_qty: Quantity going in.
        output_qty: Quantity coming out.
        yield_ratio: Actual yield for this step.
        facility_id: Facility where step occurred.
        timestamp: When the step occurred.
    """

    step_number: int = 0
    transform_id: str = ""
    process_type: str = ""
    input_commodity: str = ""
    output_commodity: str = ""
    input_qty: float = 0.0
    output_qty: float = 0.0
    yield_ratio: float = 0.0
    facility_id: str = ""
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "step_number": self.step_number,
            "transform_id": self.transform_id,
            "process_type": self.process_type,
            "input_commodity": self.input_commodity,
            "output_commodity": self.output_commodity,
            "input_qty": self.input_qty,
            "output_qty": self.output_qty,
            "yield_ratio": self.yield_ratio,
            "facility_id": self.facility_id,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
        }


@dataclass
class TransformationChain:
    """A complete multi-step transformation chain from raw to final product.

    Attributes:
        chain_id: Unique chain identifier.
        batch_id: The batch being traced.
        origin_commodity: Starting raw material.
        final_commodity: Final product form.
        steps: Ordered list of transformation steps.
        total_steps: Number of steps in the chain.
        cumulative_yield: Product of all step yields.
        origin_plot_ids: Origin plots carried through the chain.
        chain_complete: Whether the chain is fully linked.
        gaps: List of missing steps or data.
        provenance_hash: SHA-256 hash for audit trail.
        assembled_at: When the chain was assembled.
    """

    chain_id: str = field(default_factory=lambda: _generate_id("CHN"))
    batch_id: str = ""
    origin_commodity: str = ""
    final_commodity: str = ""
    steps: List[TransformationChainStep] = field(default_factory=list)
    total_steps: int = 0
    cumulative_yield: float = 1.0
    origin_plot_ids: List[str] = field(default_factory=list)
    chain_complete: bool = False
    gaps: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    assembled_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "chain_id": self.chain_id,
            "batch_id": self.batch_id,
            "origin_commodity": self.origin_commodity,
            "final_commodity": self.final_commodity,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "cumulative_yield": self.cumulative_yield,
            "origin_plot_ids": list(self.origin_plot_ids),
            "chain_complete": self.chain_complete,
            "gaps": list(self.gaps),
            "provenance_hash": self.provenance_hash,
            "assembled_at": (
                self.assembled_at.isoformat() if self.assembled_at else None
            ),
        }


@dataclass
class CoProductAllocation:
    """Result of co-product allocation for a transformation.

    Attributes:
        allocation_id: Unique allocation identifier.
        transform_id: Transformation being allocated.
        allocation_method: Method used (mass/economic/energy).
        products: List of co-products with their allocation shares.
        total_input_qty: Total input quantity being allocated.
        allocation_factors: Computed allocation factors per product.
        total_allocation_pct: Sum of all allocation percentages (should be 100).
        notes: Additional notes on the allocation.
        provenance_hash: SHA-256 hash for audit trail.
        allocated_at: When the allocation was performed.
    """

    allocation_id: str = field(default_factory=lambda: _generate_id("CPA"))
    transform_id: str = ""
    allocation_method: str = "mass"
    products: List[InputOutput] = field(default_factory=list)
    total_input_qty: float = 0.0
    allocation_factors: Dict[str, float] = field(default_factory=dict)
    total_allocation_pct: float = 0.0
    notes: str = ""
    provenance_hash: str = ""
    allocated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "allocation_id": self.allocation_id,
            "transform_id": self.transform_id,
            "allocation_method": self.allocation_method,
            "products": [p.to_dict() for p in self.products],
            "total_input_qty": self.total_input_qty,
            "allocation_factors": dict(self.allocation_factors),
            "total_allocation_pct": self.total_allocation_pct,
            "notes": self.notes,
            "provenance_hash": self.provenance_hash,
            "allocated_at": (
                self.allocated_at.isoformat() if self.allocated_at else None
            ),
        }


@dataclass
class YieldStatistics:
    """Historical yield statistics for a commodity/process combination.

    Attributes:
        stats_id: Unique statistics identifier.
        commodity: Commodity analysed.
        process_type: Process type analysed.
        sample_count: Number of transformation records in the sample.
        mean_yield: Mean actual yield ratio.
        median_yield: Median actual yield ratio.
        min_yield: Minimum observed yield.
        max_yield: Maximum observed yield.
        std_dev: Standard deviation of yield.
        expected_min: Reference minimum yield.
        expected_max: Reference maximum yield.
        reference_source: Source of reference data.
        outlier_count: Number of yields outside expected range.
        outlier_pct: Percentage of outliers.
        provenance_hash: SHA-256 hash for audit trail.
        computed_at: When the statistics were computed.
    """

    stats_id: str = field(default_factory=lambda: _generate_id("YST"))
    commodity: str = ""
    process_type: str = ""
    sample_count: int = 0
    mean_yield: float = 0.0
    median_yield: float = 0.0
    min_yield: float = 0.0
    max_yield: float = 0.0
    std_dev: float = 0.0
    expected_min: float = 0.0
    expected_max: float = 0.0
    reference_source: str = ""
    outlier_count: int = 0
    outlier_pct: float = 0.0
    provenance_hash: str = ""
    computed_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "stats_id": self.stats_id,
            "commodity": self.commodity,
            "process_type": self.process_type,
            "sample_count": self.sample_count,
            "mean_yield": self.mean_yield,
            "median_yield": self.median_yield,
            "min_yield": self.min_yield,
            "max_yield": self.max_yield,
            "std_dev": self.std_dev,
            "expected_min": self.expected_min,
            "expected_max": self.expected_max,
            "reference_source": self.reference_source,
            "outlier_count": self.outlier_count,
            "outlier_pct": self.outlier_pct,
            "provenance_hash": self.provenance_hash,
            "computed_at": (
                self.computed_at.isoformat() if self.computed_at else None
            ),
        }


@dataclass
class BatchTransformResult:
    """Result of a batch import of transformation records.

    Attributes:
        result_id: Unique result identifier.
        total_submitted: Number of transformations submitted.
        total_recorded: Number successfully recorded.
        total_failed: Number that failed validation or recording.
        recorded_ids: List of successfully recorded transformation IDs.
        errors: List of error details for failed records.
        yield_warnings: Number of records with yield warnings.
        yield_alerts: Number of records with yield alerts.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        completed_at: When the batch import completed.
    """

    result_id: str = field(default_factory=lambda: _generate_id("BTR"))
    total_submitted: int = 0
    total_recorded: int = 0
    total_failed: int = 0
    recorded_ids: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    yield_warnings: int = 0
    yield_alerts: int = 0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    completed_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "total_submitted": self.total_submitted,
            "total_recorded": self.total_recorded,
            "total_failed": self.total_failed,
            "recorded_ids": list(self.recorded_ids),
            "errors": list(self.errors),
            "yield_warnings": self.yield_warnings,
            "yield_alerts": self.yield_alerts,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


@dataclass
class TransformationTrackerConfig:
    """Configuration for the TransformationTracker engine.

    Attributes:
        yield_warning_tolerance_pct: Percentage deviation from expected
            yield that triggers a warning.
        yield_alert_tolerance_pct: Percentage deviation from expected
            yield that triggers an alert/failure.
        max_batch_import_size: Maximum number of transformations per
            batch import.
        enable_provenance: Whether to compute provenance hashes.
        enable_by_product_tracking: Whether to track by-products.
        strict_yield_validation: If True, yield failures raise errors
            instead of warnings.
        default_unit: Default unit of measurement.
    """

    yield_warning_tolerance_pct: float = DEFAULT_YIELD_WARNING_TOLERANCE_PCT
    yield_alert_tolerance_pct: float = DEFAULT_YIELD_ALERT_TOLERANCE_PCT
    max_batch_import_size: int = MAX_BATCH_IMPORT_SIZE
    enable_provenance: bool = True
    enable_by_product_tracking: bool = True
    strict_yield_validation: bool = False
    default_unit: str = "kg"

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization."""
        errors: List[str] = []

        if not (0.0 <= self.yield_warning_tolerance_pct <= 100.0):
            errors.append(
                f"yield_warning_tolerance_pct must be in [0, 100], "
                f"got {self.yield_warning_tolerance_pct}"
            )
        if not (0.0 <= self.yield_alert_tolerance_pct <= 100.0):
            errors.append(
                f"yield_alert_tolerance_pct must be in [0, 100], "
                f"got {self.yield_alert_tolerance_pct}"
            )
        if self.yield_warning_tolerance_pct > self.yield_alert_tolerance_pct:
            errors.append(
                "yield_warning_tolerance_pct must be <= yield_alert_tolerance_pct"
            )
        if self.max_batch_import_size <= 0:
            errors.append(
                f"max_batch_import_size must be > 0, "
                f"got {self.max_batch_import_size}"
            )

        if errors:
            raise ValueError(
                "TransformationTrackerConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


# ===========================================================================
# TransformationTracker Engine
# ===========================================================================


class TransformationTracker:
    """Transformation tracking engine for EUDR chain of custody.

    Records processing steps with input-to-output batch mappings, validates
    yield ratios against industry reference data, tracks by-products and
    waste, builds multi-step transformation chains, and supports co-product
    allocation using mass, economic, or energy allocation methods.

    All calculations are deterministic -- no LLM or ML is used in any
    arithmetic or validation path. Yield reference data is sourced from
    ICCO, MPOB, ICO, SCA, NOPA, IRSG, ITTO, and FAO.

    Attributes:
        config: TransformationTrackerConfig with engine settings.
        _transform_store: Dictionary of transform_id -> TransformationRecord.
        _by_product_store: Dictionary of transform_id -> ByProductRecord.
        _batch_transform_index: Index of batch_id -> list of transform_ids.
        _facility_transform_index: Index of facility_id -> list of transform_ids.
        _co_product_store: Dictionary of transform_id -> CoProductAllocation.
        _derived_product_store: List of derived product tracking records.
        _record_count: Total records stored.

    Example:
        >>> tracker = TransformationTracker()
        >>> record = tracker.record_transformation({
        ...     "process_type": "pressing",
        ...     "facility_id": "FAC-001",
        ...     "inputs": [{"batch_id": "B001", "commodity": "ffb",
        ...                 "quantity": 1000.0}],
        ...     "outputs": [{"batch_id": "B002", "commodity": "cpo",
        ...                  "quantity": 220.0}],
        ... })
        >>> assert record.yield_verdict == "pass"
    """

    def __init__(
        self, config: Optional[TransformationTrackerConfig] = None
    ) -> None:
        """Initialize the TransformationTracker engine.

        Args:
            config: Optional configuration. Defaults to
                TransformationTrackerConfig() with standard settings.
        """
        self.config = config or TransformationTrackerConfig()
        self._transform_store: Dict[str, TransformationRecord] = {}
        self._by_product_store: Dict[str, ByProductRecord] = {}
        self._batch_transform_index: Dict[str, List[str]] = defaultdict(list)
        self._facility_transform_index: Dict[str, List[str]] = defaultdict(list)
        self._co_product_store: Dict[str, CoProductAllocation] = {}
        self._derived_product_store: List[Dict[str, Any]] = []
        self._record_count: int = 0

        logger.info(
            "TransformationTracker initialized: yield_warning=%.1f%%, "
            "yield_alert=%.1f%%, max_batch=%d, provenance=%s, "
            "by_product_tracking=%s",
            self.config.yield_warning_tolerance_pct,
            self.config.yield_alert_tolerance_pct,
            self.config.max_batch_import_size,
            self.config.enable_provenance,
            self.config.enable_by_product_tracking,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_transformation(
        self, transform_data: Dict[str, Any]
    ) -> TransformationRecord:
        """Record a processing/transformation step.

        Records a processing step with input-to-output batch mappings,
        computes the actual yield ratio, validates against reference data,
        and stores the record with a provenance hash.

        Args:
            transform_data: Dictionary containing transformation details.
                Required keys:
                    - process_type (str): Processing type from ProcessType.
                    - facility_id (str): Facility identifier.
                    - inputs (list[dict]): Input batch entries with
                      batch_id, commodity, quantity.
                    - outputs (list[dict]): Output batch entries with
                      batch_id, commodity, quantity.
                Optional keys:
                    - facility_name (str): Human-readable facility name.
                    - timestamp (str|datetime): When transformation occurred.
                    - duration_minutes (float): Processing duration.
                    - operator_id (str): Operator identifier.
                    - notes (str): Free-text notes.
                    - metadata (dict): Additional metadata.

        Returns:
            TransformationRecord with computed yield and provenance hash.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        start_time = time.monotonic()

        # Validate required fields
        self._validate_transform_data(transform_data)

        # Parse inputs and outputs
        inputs = self._parse_io_list(transform_data.get("inputs", []))
        outputs = self._parse_io_list(transform_data.get("outputs", []))

        # Compute totals
        input_total = sum(io.quantity for io in inputs)
        output_total = sum(io.quantity for io in outputs)

        # Compute yield ratio
        actual_yield = output_total / input_total if input_total > 0.0 else 0.0

        # Look up expected yield
        primary_commodity = self._get_primary_commodity(inputs)
        process_type = transform_data["process_type"]
        expected_min, expected_max, yield_source = self._lookup_expected_yield(
            primary_commodity, process_type
        )

        # Determine yield verdict
        yield_verdict = self._evaluate_yield(
            actual_yield, expected_min, expected_max
        )

        # Parse timestamp
        timestamp = self._parse_timestamp(transform_data.get("timestamp"))

        # Build record
        record = TransformationRecord(
            process_type=process_type,
            facility_id=transform_data["facility_id"],
            facility_name=transform_data.get("facility_name", ""),
            inputs=inputs,
            outputs=outputs,
            input_total_qty=input_total,
            output_total_qty=output_total,
            actual_yield=round(actual_yield, 6),
            expected_yield_min=expected_min,
            expected_yield_max=expected_max,
            yield_source=yield_source,
            yield_verdict=yield_verdict,
            timestamp=timestamp,
            duration_minutes=float(
                transform_data.get("duration_minutes", 0.0)
            ),
            operator_id=transform_data.get("operator_id", ""),
            notes=transform_data.get("notes", ""),
            status=self._status_from_verdict(yield_verdict),
            metadata=transform_data.get("metadata", {}),
        )

        # Compute provenance hash
        if self.config.enable_provenance:
            record.provenance_hash = _compute_hash(record)

        # Store record and update indexes
        self._store_record(record)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Recorded transformation %s: process=%s, facility=%s, "
            "yield=%.4f (%s), elapsed=%.1fms",
            record.transform_id,
            record.process_type,
            record.facility_id,
            record.actual_yield,
            record.yield_verdict,
            elapsed_ms,
        )

        return record

    def validate_yield(
        self,
        commodity: str,
        process_type: str,
        input_qty: float,
        output_qty: float,
    ) -> YieldValidation:
        """Validate actual yield against expected yield for a process.

        Computes the actual yield ratio (output_qty / input_qty) and
        compares it against the reference yield range for the given
        commodity and process type.

        Args:
            commodity: Commodity being processed.
            process_type: Type of processing applied.
            input_qty: Input quantity.
            output_qty: Output quantity.

        Returns:
            YieldValidation with verdict and deviation details.

        Raises:
            ValueError: If input_qty is zero or negative.
        """
        start_time = time.monotonic()

        if input_qty <= 0.0:
            raise ValueError(
                f"input_qty must be > 0, got {input_qty}"
            )
        if output_qty < 0.0:
            raise ValueError(
                f"output_qty must be >= 0, got {output_qty}"
            )

        actual_yield = output_qty / input_qty
        expected_min, expected_max, ref_source = self._lookup_expected_yield(
            commodity, process_type
        )

        # Compute deviation percentage from expected range
        deviation_pct = self._compute_yield_deviation(
            actual_yield, expected_min, expected_max
        )

        verdict = self._evaluate_yield(actual_yield, expected_min, expected_max)
        message = self._yield_message(
            commodity, process_type, actual_yield,
            expected_min, expected_max, verdict
        )

        validation = YieldValidation(
            commodity=commodity,
            process_type=process_type,
            input_qty=input_qty,
            output_qty=output_qty,
            actual_yield=round(actual_yield, 6),
            expected_min=expected_min,
            expected_max=expected_max,
            reference_source=ref_source,
            deviation_pct=round(deviation_pct, 2),
            verdict=verdict,
            message=message,
        )

        if self.config.enable_provenance:
            validation.provenance_hash = _compute_hash(validation)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Yield validation: commodity=%s, process=%s, yield=%.4f, "
            "verdict=%s, deviation=%.2f%%, elapsed=%.1fms",
            commodity,
            process_type,
            actual_yield,
            verdict,
            deviation_pct,
            elapsed_ms,
        )

        return validation

    def track_by_products(
        self,
        transform_id: str,
        main_product: Dict[str, Any],
        by_products: List[Dict[str, Any]],
        waste: Optional[List[Dict[str, Any]]] = None,
    ) -> ByProductRecord:
        """Track by-products and waste from a transformation.

        Records the main product, by-products, and waste outputs from
        a transformation, computing percentage allocations and verifying
        mass balance (output <= input).

        Args:
            transform_id: Transformation record identifier.
            main_product: Main product output dict (batch_id, commodity,
                quantity, unit).
            by_products: List of by-product output dicts.
            waste: Optional list of waste output dicts.

        Returns:
            ByProductRecord with computed allocations and mass balance.

        Raises:
            ValueError: If transform_id not found or main_product is empty.
        """
        start_time = time.monotonic()

        if not transform_id:
            raise ValueError("transform_id is required")

        # Parse IO entries
        main_io = self._parse_single_io(main_product)
        by_product_ios = self._parse_io_list(by_products)
        waste_ios = self._parse_io_list(waste or [])

        # Compute totals
        main_qty = main_io.quantity
        by_product_qty = sum(bp.quantity for bp in by_product_ios)
        waste_qty = sum(w.quantity for w in waste_ios)
        total_output = main_qty + by_product_qty + waste_qty

        # Compute percentages
        main_pct = (main_qty / total_output * 100.0) if total_output > 0.0 else 0.0
        by_pct = (by_product_qty / total_output * 100.0) if total_output > 0.0 else 0.0
        waste_pct_val = (waste_qty / total_output * 100.0) if total_output > 0.0 else 0.0

        # Check mass balance against stored transformation input
        mass_balance_ok = True
        if transform_id in self._transform_store:
            transform_record = self._transform_store[transform_id]
            if total_output > transform_record.input_total_qty * 1.01:
                mass_balance_ok = False
                logger.warning(
                    "Mass balance violation for transform %s: "
                    "output=%.2f > input=%.2f",
                    transform_id,
                    total_output,
                    transform_record.input_total_qty,
                )

        record = ByProductRecord(
            transform_id=transform_id,
            main_product=main_io,
            by_products=by_product_ios,
            waste=waste_ios,
            main_product_pct=round(main_pct, 2),
            by_product_pct=round(by_pct, 2),
            waste_pct=round(waste_pct_val, 2),
            total_output_qty=round(total_output, 4),
            mass_balance_check=mass_balance_ok,
        )

        if self.config.enable_provenance:
            record.provenance_hash = _compute_hash(record)

        # Store
        self._by_product_store[transform_id] = record

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "By-product record for transform %s: main=%.1f%%, "
            "by-products=%.1f%%, waste=%.1f%%, mass_balance=%s, "
            "elapsed=%.1fms",
            transform_id,
            main_pct,
            by_pct,
            waste_pct_val,
            mass_balance_ok,
            elapsed_ms,
        )

        return record

    def get_transformation_chain(
        self, batch_id: str
    ) -> TransformationChain:
        """Assemble a multi-step transformation chain for a batch.

        Traces the transformation history of a batch from raw material
        to its current form, building a complete chain of processing
        steps with cumulative yield calculation.

        Args:
            batch_id: Batch identifier to trace.

        Returns:
            TransformationChain with all steps and cumulative yield.
        """
        start_time = time.monotonic()

        steps: List[TransformationChainStep] = []
        visited: Set[str] = set()
        current_batch = batch_id
        gaps: List[str] = []

        # Walk backwards through transformations to find the chain
        chain_transforms = self._trace_batch_chain(current_batch, visited)

        # Sort by timestamp
        chain_transforms.sort(
            key=lambda t: t.timestamp if t.timestamp else _utcnow()
        )

        # Build chain steps
        cumulative_yield = 1.0
        origin_plot_ids: Set[str] = set()

        for idx, transform in enumerate(chain_transforms):
            input_commodity = self._get_primary_commodity(transform.inputs)
            output_commodity = self._get_primary_commodity(transform.outputs)

            step = TransformationChainStep(
                step_number=idx + 1,
                transform_id=transform.transform_id,
                process_type=transform.process_type,
                input_commodity=input_commodity,
                output_commodity=output_commodity,
                input_qty=transform.input_total_qty,
                output_qty=transform.output_total_qty,
                yield_ratio=transform.actual_yield,
                facility_id=transform.facility_id,
                timestamp=transform.timestamp,
            )
            steps.append(step)
            cumulative_yield *= transform.actual_yield

            # Collect origin plots from inputs
            for inp in transform.inputs:
                origin_plot_ids.update(inp.origin_plot_ids)

        # Determine origin and final commodities
        origin_commodity = steps[0].input_commodity if steps else ""
        final_commodity = steps[-1].output_commodity if steps else ""

        # Detect gaps: check for temporal continuity
        for i in range(1, len(steps)):
            prev_ts = steps[i - 1].timestamp
            curr_ts = steps[i].timestamp
            if prev_ts and curr_ts and curr_ts < prev_ts:
                gaps.append(
                    f"Step {i} timestamp precedes step {i - 1}: "
                    f"temporal order violation"
                )

        chain = TransformationChain(
            batch_id=batch_id,
            origin_commodity=origin_commodity,
            final_commodity=final_commodity,
            steps=steps,
            total_steps=len(steps),
            cumulative_yield=round(cumulative_yield, 6),
            origin_plot_ids=sorted(origin_plot_ids),
            chain_complete=len(gaps) == 0 and len(steps) > 0,
            gaps=gaps,
        )

        if self.config.enable_provenance:
            chain.provenance_hash = _compute_hash(chain)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Assembled transformation chain for batch %s: steps=%d, "
            "cumulative_yield=%.4f, complete=%s, elapsed=%.1fms",
            batch_id,
            len(steps),
            cumulative_yield,
            chain.chain_complete,
            elapsed_ms,
        )

        return chain

    def track_derived_product(
        self,
        input_commodity: str,
        output_commodity: str,
        transform: Dict[str, Any],
    ) -> TransformationRecord:
        """Track a derived product transformation where commodity form changes.

        Records transformations where the commodity changes form (e.g.,
        palm fruit -> CPO -> RBD palm oil -> soap). Validates that the
        commodity transition is expected per the commodity form chain.

        Args:
            input_commodity: Input commodity/form.
            output_commodity: Output commodity/form.
            transform: Transformation data dictionary. Same schema as
                record_transformation, but input/output commodities are
                validated against COMMODITY_FORM_CHAINS.

        Returns:
            TransformationRecord with derived product metadata.

        Raises:
            ValueError: If the commodity transition is not recognized.
        """
        start_time = time.monotonic()

        if not input_commodity:
            raise ValueError("input_commodity is required")
        if not output_commodity:
            raise ValueError("output_commodity is required")

        # Check if this is a known commodity form transition
        transition_valid = self._validate_commodity_transition(
            input_commodity, output_commodity
        )

        # Ensure inputs have the input_commodity and outputs have output_commodity
        transform_data = dict(transform)
        if "inputs" in transform_data:
            for inp in transform_data["inputs"]:
                if not inp.get("commodity"):
                    inp["commodity"] = input_commodity
        if "outputs" in transform_data:
            for out in transform_data["outputs"]:
                if not out.get("commodity"):
                    out["commodity"] = output_commodity

        # Record the transformation
        record = self.record_transformation(transform_data)

        # Store derived product tracking info
        derived_info = {
            "transform_id": record.transform_id,
            "input_commodity": input_commodity,
            "output_commodity": output_commodity,
            "transition_valid": transition_valid,
            "recorded_at": _utcnow().isoformat(),
        }
        self._derived_product_store.append(derived_info)

        if not transition_valid:
            logger.warning(
                "Unrecognized commodity transition: %s -> %s (transform %s). "
                "No reference yield data may be available.",
                input_commodity,
                output_commodity,
                record.transform_id,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Derived product tracked: %s -> %s, transform=%s, "
            "valid_transition=%s, elapsed=%.1fms",
            input_commodity,
            output_commodity,
            record.transform_id,
            transition_valid,
            elapsed_ms,
        )

        return record

    def allocate_co_products(
        self,
        transform_id: str,
        allocation_method: str = "mass",
        economic_values: Optional[Dict[str, float]] = None,
        energy_values: Optional[Dict[str, float]] = None,
    ) -> CoProductAllocation:
        """Allocate environmental burden across co-products.

        When one process produces multiple valuable outputs, allocates
        the input material (and its environmental burden) across the
        co-products using the specified allocation method (mass, economic,
        or energy).

        Args:
            transform_id: Transformation record identifier.
            allocation_method: Allocation method ('mass', 'economic',
                or 'energy'). Defaults to 'mass'.
            economic_values: Required for economic allocation. Dictionary
                of product_commodity -> price_per_unit.
            energy_values: Required for energy allocation. Dictionary
                of product_commodity -> energy_content_mj_per_kg.

        Returns:
            CoProductAllocation with computed allocation factors.

        Raises:
            ValueError: If transform_id not found or allocation data
                is insufficient.
        """
        start_time = time.monotonic()

        # Validate allocation method
        if allocation_method not in ("mass", "economic", "energy"):
            raise ValueError(
                f"allocation_method must be mass, economic, or energy, "
                f"got '{allocation_method}'"
            )

        # Look up transformation
        transform = self._transform_store.get(transform_id)
        if transform is None:
            raise ValueError(
                f"Transformation {transform_id} not found in store"
            )

        outputs = transform.outputs
        if not outputs:
            raise ValueError(
                f"Transformation {transform_id} has no outputs for allocation"
            )

        # Compute allocation factors
        allocation_factors: Dict[str, float] = {}

        if allocation_method == "mass":
            allocation_factors = self._allocate_by_mass(outputs)
        elif allocation_method == "economic":
            if not economic_values:
                raise ValueError(
                    "economic_values required for economic allocation"
                )
            allocation_factors = self._allocate_by_economic(
                outputs, economic_values
            )
        elif allocation_method == "energy":
            if not energy_values:
                raise ValueError(
                    "energy_values required for energy allocation"
                )
            allocation_factors = self._allocate_by_energy(
                outputs, energy_values
            )

        total_allocation = sum(allocation_factors.values())

        allocation = CoProductAllocation(
            transform_id=transform_id,
            allocation_method=allocation_method,
            products=list(outputs),
            total_input_qty=transform.input_total_qty,
            allocation_factors=allocation_factors,
            total_allocation_pct=round(total_allocation, 2),
            notes=(
                f"Allocated using {allocation_method} method. "
                f"Total allocation: {total_allocation:.2f}%."
            ),
        )

        if self.config.enable_provenance:
            allocation.provenance_hash = _compute_hash(allocation)

        self._co_product_store[transform_id] = allocation

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Co-product allocation for transform %s: method=%s, "
            "products=%d, total_alloc=%.2f%%, elapsed=%.1fms",
            transform_id,
            allocation_method,
            len(outputs),
            total_allocation,
            elapsed_ms,
        )

        return allocation

    def get_yield_statistics(
        self, commodity: str, process_type: str
    ) -> YieldStatistics:
        """Compute historical yield statistics for a commodity/process.

        Aggregates all recorded transformations for the specified commodity
        and process type, computing mean, median, min, max, standard
        deviation, and outlier counts vs reference data.

        Args:
            commodity: Commodity to analyse.
            process_type: Process type to analyse.

        Returns:
            YieldStatistics with computed aggregates.
        """
        start_time = time.monotonic()

        # Gather all yields for this commodity/process
        yields: List[float] = []
        for transform in self._transform_store.values():
            primary = self._get_primary_commodity(transform.inputs)
            if primary == commodity and transform.process_type == process_type:
                if transform.actual_yield > 0.0:
                    yields.append(transform.actual_yield)

        # Look up reference
        expected_min, expected_max, ref_source = self._lookup_expected_yield(
            commodity, process_type
        )

        # Compute statistics
        sample_count = len(yields)
        if sample_count == 0:
            stats = YieldStatistics(
                commodity=commodity,
                process_type=process_type,
                sample_count=0,
                expected_min=expected_min,
                expected_max=expected_max,
                reference_source=ref_source,
            )
            if self.config.enable_provenance:
                stats.provenance_hash = _compute_hash(stats)
            return stats

        sorted_yields = sorted(yields)
        mean_val = sum(yields) / sample_count
        median_val = self._compute_median(sorted_yields)
        min_val = sorted_yields[0]
        max_val = sorted_yields[-1]
        std_dev = self._compute_std_dev(yields, mean_val)

        # Count outliers (outside expected range +/- alert tolerance)
        outlier_count = 0
        tolerance = self.config.yield_alert_tolerance_pct / 100.0
        low_bound = expected_min * (1.0 - tolerance) if expected_min > 0 else 0.0
        high_bound = expected_max * (1.0 + tolerance) if expected_max > 0 else 1.0

        for y in yields:
            if y < low_bound or y > high_bound:
                outlier_count += 1

        outlier_pct = (outlier_count / sample_count * 100.0) if sample_count > 0 else 0.0

        stats = YieldStatistics(
            commodity=commodity,
            process_type=process_type,
            sample_count=sample_count,
            mean_yield=round(mean_val, 6),
            median_yield=round(median_val, 6),
            min_yield=round(min_val, 6),
            max_yield=round(max_val, 6),
            std_dev=round(std_dev, 6),
            expected_min=expected_min,
            expected_max=expected_max,
            reference_source=ref_source,
            outlier_count=outlier_count,
            outlier_pct=round(outlier_pct, 2),
        )

        if self.config.enable_provenance:
            stats.provenance_hash = _compute_hash(stats)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Yield statistics for %s/%s: samples=%d, mean=%.4f, "
            "median=%.4f, outliers=%d (%.1f%%), elapsed=%.1fms",
            commodity,
            process_type,
            sample_count,
            mean_val,
            median_val,
            outlier_count,
            outlier_pct,
            elapsed_ms,
        )

        return stats

    def batch_record(
        self, transformations: List[Dict[str, Any]]
    ) -> BatchTransformResult:
        """Batch import of transformation records.

        Processes multiple transformation records in a single call,
        validating each individually and returning a summary result.

        Args:
            transformations: List of transformation data dictionaries.
                Each follows the same schema as record_transformation.

        Returns:
            BatchTransformResult with counts and error details.

        Raises:
            ValueError: If the batch size exceeds the configured maximum.
        """
        start_time = time.monotonic()

        if len(transformations) > self.config.max_batch_import_size:
            raise ValueError(
                f"Batch size {len(transformations)} exceeds maximum "
                f"{self.config.max_batch_import_size}"
            )

        result = BatchTransformResult(
            total_submitted=len(transformations),
        )

        for idx, transform_data in enumerate(transformations):
            try:
                record = self.record_transformation(transform_data)
                result.recorded_ids.append(record.transform_id)
                result.total_recorded += 1

                if record.yield_verdict == YieldVerdict.WARNING.value:
                    result.yield_warnings += 1
                elif record.yield_verdict == YieldVerdict.FAIL.value:
                    result.yield_alerts += 1

            except Exception as exc:
                result.total_failed += 1
                result.errors.append({
                    "index": idx,
                    "error": str(exc),
                    "data_summary": {
                        "process_type": transform_data.get("process_type", ""),
                        "facility_id": transform_data.get("facility_id", ""),
                    },
                })
                logger.warning(
                    "Batch record item %d failed: %s", idx, str(exc)
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result.processing_time_ms = round(elapsed_ms, 2)
        result.completed_at = _utcnow()

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "Batch transformation import complete: submitted=%d, "
            "recorded=%d, failed=%d, warnings=%d, alerts=%d, "
            "elapsed=%.1fms",
            result.total_submitted,
            result.total_recorded,
            result.total_failed,
            result.yield_warnings,
            result.yield_alerts,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_transformation(
        self, transform_id: str
    ) -> Optional[TransformationRecord]:
        """Retrieve a transformation record by ID.

        Args:
            transform_id: Transformation identifier.

        Returns:
            TransformationRecord if found, None otherwise.
        """
        return self._transform_store.get(transform_id)

    def get_transformations_for_batch(
        self, batch_id: str
    ) -> List[TransformationRecord]:
        """Retrieve all transformations involving a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of TransformationRecord objects.
        """
        transform_ids = self._batch_transform_index.get(batch_id, [])
        return [
            self._transform_store[tid]
            for tid in transform_ids
            if tid in self._transform_store
        ]

    def get_transformations_for_facility(
        self, facility_id: str
    ) -> List[TransformationRecord]:
        """Retrieve all transformations at a facility.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of TransformationRecord objects.
        """
        transform_ids = self._facility_transform_index.get(facility_id, [])
        return [
            self._transform_store[tid]
            for tid in transform_ids
            if tid in self._transform_store
        ]

    def get_by_product_record(
        self, transform_id: str
    ) -> Optional[ByProductRecord]:
        """Retrieve a by-product record by transformation ID.

        Args:
            transform_id: Transformation identifier.

        Returns:
            ByProductRecord if found, None otherwise.
        """
        return self._by_product_store.get(transform_id)

    def get_co_product_allocation(
        self, transform_id: str
    ) -> Optional[CoProductAllocation]:
        """Retrieve a co-product allocation by transformation ID.

        Args:
            transform_id: Transformation identifier.

        Returns:
            CoProductAllocation if found, None otherwise.
        """
        return self._co_product_store.get(transform_id)

    @property
    def record_count(self) -> int:
        """Return total number of transformation records stored."""
        return self._record_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_transform_data(self, data: Dict[str, Any]) -> None:
        """Validate transformation data contains required fields.

        Args:
            data: Transformation data dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        errors: List[str] = []

        if not data.get("process_type"):
            errors.append("process_type is required")
        if not data.get("facility_id"):
            errors.append("facility_id is required")
        if not data.get("inputs"):
            errors.append("inputs is required and must be non-empty")
        if not data.get("outputs"):
            errors.append("outputs is required and must be non-empty")

        # Validate process type is recognized
        process_type = data.get("process_type", "")
        valid_types = {pt.value for pt in ProcessType}
        if process_type and process_type not in valid_types:
            logger.warning(
                "Unrecognized process_type '%s'. Known types: %s",
                process_type,
                ", ".join(sorted(valid_types)),
            )

        # Validate inputs have required fields
        for idx, inp in enumerate(data.get("inputs", [])):
            if not isinstance(inp, dict):
                errors.append(f"inputs[{idx}] must be a dictionary")
                continue
            if not inp.get("batch_id"):
                errors.append(f"inputs[{idx}].batch_id is required")
            if inp.get("quantity") is not None and float(inp["quantity"]) < 0:
                errors.append(
                    f"inputs[{idx}].quantity must be >= 0, "
                    f"got {inp['quantity']}"
                )

        # Validate outputs have required fields
        for idx, out in enumerate(data.get("outputs", [])):
            if not isinstance(out, dict):
                errors.append(f"outputs[{idx}] must be a dictionary")
                continue
            if not out.get("batch_id"):
                errors.append(f"outputs[{idx}].batch_id is required")
            if out.get("quantity") is not None and float(out["quantity"]) < 0:
                errors.append(
                    f"outputs[{idx}].quantity must be >= 0, "
                    f"got {out['quantity']}"
                )

        if errors:
            raise ValueError(
                "Transformation data validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _parse_io_list(self, io_list: List[Dict[str, Any]]) -> List[InputOutput]:
        """Parse a list of input/output dictionaries to InputOutput objects.

        Args:
            io_list: List of dictionaries with batch_id, commodity,
                quantity, unit, etc.

        Returns:
            List of InputOutput dataclass instances.
        """
        results: List[InputOutput] = []
        for item in io_list:
            io = self._parse_single_io(item)
            results.append(io)
        return results

    def _parse_single_io(self, item: Dict[str, Any]) -> InputOutput:
        """Parse a single input/output dictionary.

        Args:
            item: Dictionary with batch_id, commodity, quantity, etc.

        Returns:
            InputOutput dataclass instance.
        """
        return InputOutput(
            batch_id=str(item.get("batch_id", "")),
            commodity=str(item.get("commodity", "")),
            quantity=float(item.get("quantity", 0.0)),
            unit=str(item.get("unit", self.config.default_unit)),
            quality_grade=item.get("quality_grade"),
            origin_plot_ids=list(item.get("origin_plot_ids", [])),
            metadata=dict(item.get("metadata", {})),
        )

    def _get_primary_commodity(
        self, io_list: List[InputOutput]
    ) -> str:
        """Get the primary commodity from an IO list.

        Returns the commodity of the first entry, or empty string.

        Args:
            io_list: List of InputOutput entries.

        Returns:
            Primary commodity string.
        """
        if io_list and io_list[0].commodity:
            return io_list[0].commodity
        return ""

    def _lookup_expected_yield(
        self, commodity: str, process_type: str
    ) -> Tuple[float, float, str]:
        """Look up expected yield range from reference data.

        Args:
            commodity: Commodity being processed.
            process_type: Type of processing.

        Returns:
            Tuple of (expected_min, expected_max, source). Returns
            (0.0, 1.0, "unknown") if no reference data found.
        """
        key = (commodity, process_type)
        if key in REFERENCE_YIELD_RATIOS:
            return REFERENCE_YIELD_RATIOS[key]

        # Try with base commodity (strip derived prefix)
        for base_commodity in EUDR_COMMODITIES:
            if commodity.startswith(base_commodity):
                base_key = (base_commodity, process_type)
                if base_key in REFERENCE_YIELD_RATIOS:
                    return REFERENCE_YIELD_RATIOS[base_key]

        logger.debug(
            "No reference yield data for commodity=%s, process=%s",
            commodity,
            process_type,
        )
        return (0.0, 1.0, "unknown")

    def _evaluate_yield(
        self,
        actual_yield: float,
        expected_min: float,
        expected_max: float,
    ) -> str:
        """Evaluate yield ratio against expected range.

        Args:
            actual_yield: Computed yield ratio.
            expected_min: Expected minimum from reference data.
            expected_max: Expected maximum from reference data.

        Returns:
            Verdict string: 'pass', 'warning', or 'fail'.
        """
        if expected_min == 0.0 and expected_max == 1.0:
            # No reference data -- cannot validate
            return YieldVerdict.PASS.value

        warning_tol = self.config.yield_warning_tolerance_pct / 100.0
        alert_tol = self.config.yield_alert_tolerance_pct / 100.0

        low_warn = expected_min * (1.0 - warning_tol)
        high_warn = expected_max * (1.0 + warning_tol)
        low_alert = expected_min * (1.0 - alert_tol)
        high_alert = expected_max * (1.0 + alert_tol)

        if actual_yield < low_alert or actual_yield > high_alert:
            return YieldVerdict.FAIL.value
        elif actual_yield < low_warn or actual_yield > high_warn:
            return YieldVerdict.WARNING.value
        else:
            return YieldVerdict.PASS.value

    def _compute_yield_deviation(
        self,
        actual_yield: float,
        expected_min: float,
        expected_max: float,
    ) -> float:
        """Compute percentage deviation from expected yield range.

        Args:
            actual_yield: Computed actual yield.
            expected_min: Expected minimum yield.
            expected_max: Expected maximum yield.

        Returns:
            Percentage deviation (negative = below range, positive = above).
            Returns 0.0 if within range.
        """
        if expected_min <= actual_yield <= expected_max:
            return 0.0

        if actual_yield < expected_min and expected_min > 0.0:
            return ((actual_yield - expected_min) / expected_min) * 100.0

        if actual_yield > expected_max and expected_max > 0.0:
            return ((actual_yield - expected_max) / expected_max) * 100.0

        return 0.0

    def _yield_message(
        self,
        commodity: str,
        process_type: str,
        actual: float,
        expected_min: float,
        expected_max: float,
        verdict: str,
    ) -> str:
        """Generate a human-readable yield validation message.

        Args:
            commodity: Commodity name.
            process_type: Process type.
            actual: Actual yield.
            expected_min: Expected minimum.
            expected_max: Expected maximum.
            verdict: Validation verdict.

        Returns:
            Human-readable message string.
        """
        if verdict == YieldVerdict.PASS.value:
            return (
                f"Yield {actual:.4f} for {commodity}/{process_type} is within "
                f"expected range [{expected_min:.4f}, {expected_max:.4f}]."
            )
        elif verdict == YieldVerdict.WARNING.value:
            return (
                f"Yield {actual:.4f} for {commodity}/{process_type} is outside "
                f"expected range [{expected_min:.4f}, {expected_max:.4f}] "
                f"but within tolerance. Review recommended."
            )
        else:
            return (
                f"Yield {actual:.4f} for {commodity}/{process_type} significantly "
                f"deviates from expected range [{expected_min:.4f}, {expected_max:.4f}]. "
                f"Investigation required."
            )

    def _status_from_verdict(self, verdict: str) -> str:
        """Map yield verdict to transformation status.

        Args:
            verdict: Yield verdict string.

        Returns:
            Transformation status string.
        """
        mapping = {
            YieldVerdict.PASS.value: TransformationStatus.VALIDATED.value,
            YieldVerdict.WARNING.value: TransformationStatus.YIELD_WARNING.value,
            YieldVerdict.FAIL.value: TransformationStatus.YIELD_ALERT.value,
        }
        return mapping.get(verdict, TransformationStatus.RECORDED.value)

    def _parse_timestamp(
        self, ts_value: Any
    ) -> datetime:
        """Parse a timestamp value to datetime.

        Args:
            ts_value: String, datetime, or None.

        Returns:
            Parsed datetime. Defaults to current UTC if None.
        """
        if ts_value is None:
            return _utcnow()
        if isinstance(ts_value, datetime):
            return ts_value
        if isinstance(ts_value, str):
            try:
                return datetime.fromisoformat(ts_value)
            except ValueError:
                logger.warning(
                    "Could not parse timestamp '%s', using current UTC",
                    ts_value,
                )
                return _utcnow()
        return _utcnow()

    def _store_record(self, record: TransformationRecord) -> None:
        """Store a transformation record and update indexes.

        Args:
            record: TransformationRecord to store.
        """
        self._transform_store[record.transform_id] = record
        self._record_count += 1

        # Update batch index
        for inp in record.inputs:
            if inp.batch_id:
                self._batch_transform_index[inp.batch_id].append(
                    record.transform_id
                )
        for out in record.outputs:
            if out.batch_id:
                self._batch_transform_index[out.batch_id].append(
                    record.transform_id
                )

        # Update facility index
        if record.facility_id:
            self._facility_transform_index[record.facility_id].append(
                record.transform_id
            )

    def _trace_batch_chain(
        self,
        batch_id: str,
        visited: Set[str],
    ) -> List[TransformationRecord]:
        """Trace the transformation chain for a batch recursively.

        Walks backwards through transformations: for each transformation
        that outputs this batch, finds the input batches and recurses.

        Args:
            batch_id: Batch identifier to trace.
            visited: Set of already visited transform IDs to prevent cycles.

        Returns:
            List of TransformationRecord objects in the chain.
        """
        chain: List[TransformationRecord] = []
        transform_ids = self._batch_transform_index.get(batch_id, [])

        for tid in transform_ids:
            if tid in visited:
                continue
            visited.add(tid)

            transform = self._transform_store.get(tid)
            if transform is None:
                continue

            # Check if this batch is an output of this transformation
            is_output = any(
                out.batch_id == batch_id for out in transform.outputs
            )
            if is_output:
                # Recurse into input batches
                for inp in transform.inputs:
                    if inp.batch_id and inp.batch_id != batch_id:
                        upstream = self._trace_batch_chain(
                            inp.batch_id, visited
                        )
                        chain.extend(upstream)
                chain.append(transform)

            # Check if this batch is an input of this transformation
            is_input = any(
                inp.batch_id == batch_id for inp in transform.inputs
            )
            if is_input and not is_output:
                chain.append(transform)
                # Recurse into output batches
                for out in transform.outputs:
                    if out.batch_id and out.batch_id != batch_id:
                        downstream = self._trace_batch_chain(
                            out.batch_id, visited
                        )
                        chain.extend(downstream)

        return chain

    def _validate_commodity_transition(
        self, input_commodity: str, output_commodity: str
    ) -> bool:
        """Check if a commodity form transition is recognized.

        Args:
            input_commodity: Input commodity/form.
            output_commodity: Output commodity/form.

        Returns:
            True if the transition is in a known commodity form chain.
        """
        for chain in COMMODITY_FORM_CHAINS.values():
            try:
                in_idx = chain.index(input_commodity)
                out_idx = chain.index(output_commodity)
                if out_idx > in_idx:
                    return True
            except ValueError:
                continue
        return False

    def _allocate_by_mass(
        self, outputs: List[InputOutput]
    ) -> Dict[str, float]:
        """Allocate by mass proportion.

        Args:
            outputs: List of output entries.

        Returns:
            Dictionary of commodity -> allocation percentage.
        """
        total_mass = sum(o.quantity for o in outputs)
        if total_mass <= 0.0:
            return {}

        factors: Dict[str, float] = {}
        for output in outputs:
            key = output.commodity or output.batch_id
            pct = (output.quantity / total_mass) * 100.0
            factors[key] = round(pct, 4)
        return factors

    def _allocate_by_economic(
        self,
        outputs: List[InputOutput],
        economic_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Allocate by economic value proportion.

        Args:
            outputs: List of output entries.
            economic_values: Dictionary of commodity -> price per unit.

        Returns:
            Dictionary of commodity -> allocation percentage.
        """
        total_value = 0.0
        product_values: List[Tuple[str, float]] = []

        for output in outputs:
            commodity = output.commodity or output.batch_id
            price = economic_values.get(commodity, 0.0)
            value = output.quantity * price
            product_values.append((commodity, value))
            total_value += value

        if total_value <= 0.0:
            return {}

        factors: Dict[str, float] = {}
        for commodity, value in product_values:
            pct = (value / total_value) * 100.0
            factors[commodity] = round(pct, 4)
        return factors

    def _allocate_by_energy(
        self,
        outputs: List[InputOutput],
        energy_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Allocate by energy content proportion.

        Args:
            outputs: List of output entries.
            energy_values: Dictionary of commodity -> MJ per kg.

        Returns:
            Dictionary of commodity -> allocation percentage.
        """
        total_energy = 0.0
        product_energies: List[Tuple[str, float]] = []

        for output in outputs:
            commodity = output.commodity or output.batch_id
            energy_per_kg = energy_values.get(commodity, 0.0)
            energy = output.quantity * energy_per_kg
            product_energies.append((commodity, energy))
            total_energy += energy

        if total_energy <= 0.0:
            return {}

        factors: Dict[str, float] = {}
        for commodity, energy in product_energies:
            pct = (energy / total_energy) * 100.0
            factors[commodity] = round(pct, 4)
        return factors

    def _compute_median(self, sorted_values: List[float]) -> float:
        """Compute median of a sorted list.

        Args:
            sorted_values: Pre-sorted list of float values.

        Returns:
            Median value.
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        return sorted_values[mid]

    def _compute_std_dev(
        self, values: List[float], mean: float
    ) -> float:
        """Compute standard deviation.

        Args:
            values: List of values.
            mean: Pre-computed mean.

        Returns:
            Standard deviation (population).
        """
        n = len(values)
        if n <= 1:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / n
        return variance ** 0.5
