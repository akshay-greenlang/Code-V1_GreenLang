# -*- coding: utf-8 -*-
"""
LifecycleAssessmentEngine - PACK-018 EU Green Claims Prep Engine 3
===================================================================

Performs PEF/LCA-based lifecycle impact verification per EU Green Claims
Directive Article 3(1), including hotspot identification, data quality
assessment, and system boundary validation.

The EU Green Claims Directive (Proposal COM/2023/166) requires that
environmental claims be substantiated by lifecycle-based evidence.
Article 3(1) explicitly mandates that claims take into account the
full lifecycle of products, using methodologies consistent with the
Product Environmental Footprint (PEF) framework.

Article 3(1) Requirements:
    - Para (a): Environmental claims shall rely on widely recognised
      scientific evidence, identifying significant environmental
      impacts, aspects, or performance from a lifecycle perspective.
    - Para (b): The assessment shall cover all environmental aspects
      that are significant for the product or organisation.
    - Para (c): The lifecycle assessment shall follow the PEF/OEF
      methodology or an equivalent methodology based on ISO 14040/44.
    - Para (d): Primary data shall be used where available; secondary
      data shall be from recognised databases (e.g., EF database).

PEF Methodology:
    The Product Environmental Footprint (PEF) method, as defined in
    Commission Recommendation 2013/179/EU and updated through PEF
    Category Rules (PEFCRs), defines 16 impact categories with
    official weighting factors established by the EU JRC.

Impact Categories (EF 3.1, EU JRC):
    1. Climate change (GWP100)
    2. Ozone depletion
    3. Acidification
    4. Eutrophication, freshwater
    5. Eutrophication, marine
    6. Eutrophication, terrestrial
    7. Photochemical ozone formation
    8. Resource use, minerals and metals
    9. Resource use, fossils
    10. Water use
    11. Land use
    12. Ecotoxicity, freshwater
    13. Human toxicity, cancer
    14. Human toxicity, non-cancer
    15. Particulate matter
    16. Ionising radiation

Regulatory References:
    - EU Green Claims Directive Proposal COM/2023/166, Article 3(1)
    - Commission Recommendation 2013/179/EU (PEF/OEF)
    - ISO 14040:2006 (LCA Principles and Framework)
    - ISO 14044:2006 (LCA Requirements and Guidelines)
    - EU JRC Technical Report: EF reference package 3.1
    - EU JRC Weighting factors for PEF impact categories

Zero-Hallucination:
    - All impact aggregations use deterministic Decimal arithmetic
    - PEF weighting uses official EU JRC weighting factors
    - Hotspot identification uses ranked sorting of weighted scores
    - Data quality assessment uses ratio-based scoring
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
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


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


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


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LifecyclePhase(str, Enum):
    """Product lifecycle phases per PEF/ISO 14040 framework.

    Covers the six standard lifecycle phases from raw material
    extraction through end-of-life treatment.
    """
    RAW_MATERIALS = "raw_materials"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    DISTRIBUTION = "distribution"
    USE = "use"
    END_OF_LIFE = "end_of_life"


class ImpactCategory(str, Enum):
    """PEF impact categories per EU JRC EF reference package 3.1.

    The 16 impact categories defined in the PEF methodology, each
    with its own characterisation model and weighting factor.
    """
    CLIMATE_CHANGE = "climate_change"
    OZONE_DEPLETION = "ozone_depletion"
    ACIDIFICATION = "acidification"
    EUTROPHICATION_FRESHWATER = "eutrophication_freshwater"
    EUTROPHICATION_MARINE = "eutrophication_marine"
    EUTROPHICATION_TERRESTRIAL = "eutrophication_terrestrial"
    PHOTOCHEMICAL_OZONE = "photochemical_ozone"
    RESOURCE_USE_MINERALS = "resource_use_minerals"
    RESOURCE_USE_FOSSILS = "resource_use_fossils"
    WATER_USE = "water_use"
    LAND_USE = "land_use"
    ECOTOXICITY = "ecotoxicity"
    HUMAN_TOXICITY_CANCER = "human_toxicity_cancer"
    HUMAN_TOXICITY_NON_CANCER = "human_toxicity_non_cancer"
    PARTICULATE_MATTER = "particulate_matter"
    IONISING_RADIATION = "ionising_radiation"


class DataQualityRating(str, Enum):
    """Data quality rating per PEF Data Quality Requirements.

    Classifies data sources by their quality level, from high-quality
    measured primary data to low-quality estimates.
    """
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class SystemBoundaryType(str, Enum):
    """System boundary types for LCA scope definition.

    Per PEF methodology, the system boundary must be clearly defined
    to ensure comparability and completeness.
    """
    CRADLE_TO_GRAVE = "cradle_to_grave"
    CRADLE_TO_GATE = "cradle_to_gate"
    GATE_TO_GATE = "gate_to_gate"
    GATE_TO_GRAVE = "gate_to_grave"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Official EU JRC PEF weighting factors (EF 3.1).
PEF_WEIGHTING_FACTORS: Dict[str, Decimal] = {
    ImpactCategory.CLIMATE_CHANGE.value: Decimal("21.06"),
    ImpactCategory.OZONE_DEPLETION.value: Decimal("6.31"),
    ImpactCategory.ACIDIFICATION.value: Decimal("6.20"),
    ImpactCategory.EUTROPHICATION_FRESHWATER.value: Decimal("2.80"),
    ImpactCategory.EUTROPHICATION_MARINE.value: Decimal("2.96"),
    ImpactCategory.EUTROPHICATION_TERRESTRIAL.value: Decimal("3.71"),
    ImpactCategory.PHOTOCHEMICAL_OZONE.value: Decimal("4.78"),
    ImpactCategory.RESOURCE_USE_MINERALS.value: Decimal("7.55"),
    ImpactCategory.RESOURCE_USE_FOSSILS.value: Decimal("8.32"),
    ImpactCategory.WATER_USE.value: Decimal("8.51"),
    ImpactCategory.LAND_USE.value: Decimal("7.94"),
    ImpactCategory.ECOTOXICITY.value: Decimal("1.92"),
    ImpactCategory.HUMAN_TOXICITY_CANCER.value: Decimal("2.13"),
    ImpactCategory.HUMAN_TOXICITY_NON_CANCER.value: Decimal("1.84"),
    ImpactCategory.PARTICULATE_MATTER.value: Decimal("8.96"),
    ImpactCategory.IONISING_RADIATION.value: Decimal("5.01"),
}

PEF_CATEGORY_UNITS: Dict[str, str] = {
    ImpactCategory.CLIMATE_CHANGE.value: "kg CO2 eq",
    ImpactCategory.OZONE_DEPLETION.value: "kg CFC-11 eq",
    ImpactCategory.ACIDIFICATION.value: "mol H+ eq",
    ImpactCategory.EUTROPHICATION_FRESHWATER.value: "kg P eq",
    ImpactCategory.EUTROPHICATION_MARINE.value: "kg N eq",
    ImpactCategory.EUTROPHICATION_TERRESTRIAL.value: "mol N eq",
    ImpactCategory.PHOTOCHEMICAL_OZONE.value: "kg NMVOC eq",
    ImpactCategory.RESOURCE_USE_MINERALS.value: "kg Sb eq",
    ImpactCategory.RESOURCE_USE_FOSSILS.value: "MJ",
    ImpactCategory.WATER_USE.value: "m3 world eq",
    ImpactCategory.LAND_USE.value: "pt",
    ImpactCategory.ECOTOXICITY.value: "CTUe",
    ImpactCategory.HUMAN_TOXICITY_CANCER.value: "CTUh",
    ImpactCategory.HUMAN_TOXICITY_NON_CANCER.value: "CTUh",
    ImpactCategory.PARTICULATE_MATTER.value: "disease incidence",
    ImpactCategory.IONISING_RADIATION.value: "kBq U235 eq",
}

DATA_QUALITY_SCORES: Dict[str, Decimal] = {
    DataQualityRating.EXCELLENT.value: Decimal("100"),
    DataQualityRating.VERY_GOOD.value: Decimal("80"),
    DataQualityRating.GOOD.value: Decimal("60"),
    DataQualityRating.FAIR.value: Decimal("40"),
    DataQualityRating.POOR.value: Decimal("20"),
}

BOUNDARY_REQUIRED_PHASES: Dict[str, List[str]] = {
    SystemBoundaryType.CRADLE_TO_GRAVE.value: [
        LifecyclePhase.RAW_MATERIALS.value, LifecyclePhase.MANUFACTURING.value,
        LifecyclePhase.TRANSPORTATION.value, LifecyclePhase.DISTRIBUTION.value,
        LifecyclePhase.USE.value, LifecyclePhase.END_OF_LIFE.value,
    ],
    SystemBoundaryType.CRADLE_TO_GATE.value: [
        LifecyclePhase.RAW_MATERIALS.value, LifecyclePhase.MANUFACTURING.value,
    ],
    SystemBoundaryType.GATE_TO_GATE.value: [
        LifecyclePhase.MANUFACTURING.value,
    ],
    SystemBoundaryType.GATE_TO_GRAVE.value: [
        LifecyclePhase.MANUFACTURING.value, LifecyclePhase.TRANSPORTATION.value,
        LifecyclePhase.DISTRIBUTION.value, LifecyclePhase.USE.value,
        LifecyclePhase.END_OF_LIFE.value,
    ],
}

MINIMUM_DQR_THRESHOLD: Decimal = Decimal("70")
DEFAULT_HOTSPOT_COUNT: int = 5


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class LifecycleImpact(BaseModel):
    """A single lifecycle impact data point."""
    impact_id: str = Field(default_factory=_new_uuid, description="Unique impact ID")
    phase: LifecyclePhase = Field(..., description="Lifecycle phase")
    category: ImpactCategory = Field(..., description="PEF impact category")
    value: Decimal = Field(..., description="Impact value in category unit")
    unit: str = Field(default="", description="Unit of measurement")
    data_quality_rating: DataQualityRating = Field(
        default=DataQualityRating.FAIR, description="Data quality rating",
    )
    source: str = Field(default="", description="Data source reference", max_length=500)
    is_primary_data: bool = Field(default=False, description="Whether primary data")
    normalised_value: Optional[Decimal] = Field(default=None, description="Normalised value")


class LCAResult(BaseModel):
    """Result of a lifecycle assessment calculation."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    product_name: str = Field(..., description="Assessed product name")
    functional_unit: str = Field(default="", description="Functional unit", max_length=500)
    system_boundary: str = Field(
        default=SystemBoundaryType.CRADLE_TO_GRAVE.value, description="System boundary",
    )
    impacts: List[LifecycleImpact] = Field(default_factory=list, description="Impact data points")
    category_totals: Dict[str, str] = Field(default_factory=dict, description="Total per category")
    phase_totals: Dict[str, str] = Field(default_factory=dict, description="Weighted score per phase")
    total_weighted_score: Decimal = Field(default=Decimal("0.000"), description="PEF single score")
    dominant_phase: str = Field(default="", description="Phase with highest contribution")
    dominant_category: str = Field(default="", description="Category with highest contribution")
    data_quality_ratio: Decimal = Field(default=Decimal("0.00"), description="DQR percentage")
    primary_data_ratio: Decimal = Field(default=Decimal("0.00"), description="Primary data pct")
    hotspots: List[Dict[str, str]] = Field(default_factory=list, description="Top hotspots")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp UTC")
    processing_time_ms: float = Field(default=0.0, description="Processing time ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class DataQualityResult(BaseModel):
    """Result of a data quality assessment."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    total_data_points: int = Field(default=0, description="Total data points")
    quality_distribution: Dict[str, int] = Field(default_factory=dict, description="Count per rating")
    overall_dqr_score: Decimal = Field(default=Decimal("0.00"), description="Overall DQR 0-100")
    primary_data_count: int = Field(default=0, description="Primary data count")
    secondary_data_count: int = Field(default=0, description="Secondary data count")
    primary_data_ratio: Decimal = Field(default=Decimal("0.00"), description="Primary ratio 0-100")
    meets_pef_threshold: bool = Field(default=False, description="Meets PEF DQR threshold")
    phase_quality: Dict[str, str] = Field(default_factory=dict, description="DQR per phase")
    category_quality: Dict[str, str] = Field(default_factory=dict, description="DQR per category")
    issues: List[str] = Field(default_factory=list, description="Issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp UTC")
    processing_time_ms: float = Field(default=0.0, description="Processing time ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class SystemBoundaryResult(BaseModel):
    """Result of a system boundary validation."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    system_boundary: str = Field(default="", description="Declared boundary type")
    claim_scope: str = Field(default="", description="Claim scope")
    required_phases: List[str] = Field(default_factory=list, description="Required phases")
    covered_phases: List[str] = Field(default_factory=list, description="Covered phases")
    missing_phases: List[str] = Field(default_factory=list, description="Missing phases")
    coverage_pct: Decimal = Field(default=Decimal("0.00"), description="Coverage percentage")
    is_valid: bool = Field(default=False, description="Boundary valid for claim")
    boundary_appropriate_for_claim: bool = Field(default=False, description="Boundary appropriate")
    issues: List[str] = Field(default_factory=list, description="Issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp UTC")
    processing_time_ms: float = Field(default=0.0, description="Processing time ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LifecycleAssessmentEngine:
    """Lifecycle assessment engine per EU Green Claims Directive Art. 3(1).

    Provides deterministic, zero-hallucination lifecycle impact
    assessment using the PEF methodology:

    - Calculate lifecycle impacts across 16 PEF categories and 6 phases
    - Identify environmental hotspots by weighted contribution
    - Calculate PEF single scores using official EU JRC weighting
    - Assess data quality against PEF DQR requirements
    - Validate system boundaries against claim scope

    All calculations use Decimal arithmetic with ROUND_HALF_UP rounding.
    Every result includes a SHA-256 provenance hash for audit trail.

    Usage::

        engine = LifecycleAssessmentEngine()
        impacts = [
            LifecycleImpact(
                phase=LifecyclePhase.MANUFACTURING,
                category=ImpactCategory.CLIMATE_CHANGE,
                value=Decimal("125.50"),
                data_quality_rating=DataQualityRating.GOOD,
            ),
        ]
        result = engine.calculate_lifecycle_impacts("Widget X", impacts)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise LifecycleAssessmentEngine."""
        logger.info("LifecycleAssessmentEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Calculate Lifecycle Impacts                                           #
    # ------------------------------------------------------------------ #

    def calculate_lifecycle_impacts(
        self, product_name: str, impacts_data: List[LifecycleImpact],
        functional_unit: str = "",
        system_boundary: str = SystemBoundaryType.CRADLE_TO_GRAVE.value,
    ) -> Dict[str, Any]:
        """Calculate complete lifecycle impacts with PEF weighting.

        Args:
            product_name: Name of the assessed product.
            impacts_data: List of lifecycle impact data points.
            functional_unit: Functional unit for the assessment.
            system_boundary: System boundary type.

        Returns:
            Dict with keys: result (LCAResult), provenance_hash (str).
        """
        t0 = time.perf_counter()

        for impact in impacts_data:
            if not impact.unit:
                impact.unit = PEF_CATEGORY_UNITS.get(impact.category.value, "")

        category_totals: Dict[str, Decimal] = {}
        for impact in impacts_data:
            cat = impact.category.value
            category_totals[cat] = category_totals.get(cat, Decimal("0")) + abs(impact.value)

        weighted_cat: Dict[str, Decimal] = {}
        for cat, total in category_totals.items():
            weight = PEF_WEIGHTING_FACTORS.get(cat, Decimal("0"))
            weighted_cat[cat] = _round_val(total * weight / Decimal("100"), 6)

        phase_weighted: Dict[str, Decimal] = {}
        for impact in impacts_data:
            phase = impact.phase.value
            weight = PEF_WEIGHTING_FACTORS.get(impact.category.value, Decimal("0"))
            contribution = abs(impact.value) * weight / Decimal("100")
            phase_weighted[phase] = phase_weighted.get(phase, Decimal("0")) + contribution

        total_weighted = sum(weighted_cat.values())
        dominant_phase = max(phase_weighted, key=phase_weighted.get) if phase_weighted else ""
        dominant_category = max(weighted_cat, key=weighted_cat.get) if weighted_cat else ""

        dqr = self._calculate_dqr_score(impacts_data)
        primary_ratio = self._calculate_primary_ratio(impacts_data)
        hotspots = self._identify_hotspots_internal(impacts_data, DEFAULT_HOTSPOT_COUNT)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = LCAResult(
            product_name=product_name, functional_unit=functional_unit,
            system_boundary=system_boundary, impacts=impacts_data,
            category_totals={k: str(_round_val(v, 6)) for k, v in category_totals.items()},
            phase_totals={k: str(_round_val(v, 6)) for k, v in phase_weighted.items()},
            total_weighted_score=_round_val(total_weighted, 6),
            dominant_phase=dominant_phase, dominant_category=dominant_category,
            data_quality_ratio=dqr, primary_data_ratio=primary_ratio,
            hotspots=hotspots, processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Calculated lifecycle impacts for '%s': %d impacts, score=%s, "
            "dominant phase=%s, DQR=%s%% in %.3f ms",
            product_name, len(impacts_data), result.total_weighted_score,
            dominant_phase, dqr, elapsed_ms,
        )
        return {"result": result, "provenance_hash": result.provenance_hash}

    # ------------------------------------------------------------------ #
    # Identify Hotspots                                                     #
    # ------------------------------------------------------------------ #

    def identify_hotspots(
        self, impacts: List[LifecycleImpact], top_n: int = DEFAULT_HOTSPOT_COUNT,
    ) -> Dict[str, Any]:
        """Identify the top environmental hotspots.

        Args:
            impacts: List of lifecycle impact data points.
            top_n: Number of top hotspots to return.

        Returns:
            Dict with keys: hotspots, total_weighted_score,
            hotspot_coverage_pct, provenance_hash.
        """
        t0 = time.perf_counter()
        hotspots = self._identify_hotspots_internal(impacts, top_n)

        total = Decimal("0")
        for imp in impacts:
            w = PEF_WEIGHTING_FACTORS.get(imp.category.value, Decimal("0"))
            total += abs(imp.value) * w / Decimal("100")

        hs_total = sum(_decimal(h.get("weighted_score", "0")) for h in hotspots)
        coverage = _safe_divide(hs_total * Decimal("100"), total if total > Decimal("0") else Decimal("1"))

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        result = {
            "hotspots": hotspots,
            "total_weighted_score": str(_round_val(total, 6)),
            "hotspot_coverage_pct": str(_round_val(coverage, 2)),
            "top_n": top_n, "total_impacts": len(impacts),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info("Identified %d hotspots from %d impacts in %.3f ms", len(hotspots), len(impacts), elapsed_ms)
        return result

    # ------------------------------------------------------------------ #
    # Calculate PEF Score                                                   #
    # ------------------------------------------------------------------ #

    def calculate_pef_score(self, impacts: List[LifecycleImpact]) -> Dict[str, Any]:
        """Calculate the PEF single score from lifecycle impacts.

        Args:
            impacts: List of lifecycle impact data points.

        Returns:
            Dict with keys: total_score, category_scores,
            category_contributions_pct, provenance_hash.
        """
        t0 = time.perf_counter()

        cat_totals: Dict[str, Decimal] = {}
        for imp in impacts:
            cat = imp.category.value
            cat_totals[cat] = cat_totals.get(cat, Decimal("0")) + abs(imp.value)

        cat_scores: Dict[str, Decimal] = {}
        for cat, total in cat_totals.items():
            w = PEF_WEIGHTING_FACTORS.get(cat, Decimal("0"))
            cat_scores[cat] = _round_val(total * w / Decimal("100"), 6)

        total_score = sum(cat_scores.values())
        contributions: Dict[str, Decimal] = {}
        for cat, score in cat_scores.items():
            contributions[cat] = _round_val(
                _safe_divide(score * Decimal("100"), total_score if total_score > Decimal("0") else Decimal("1")), 2,
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        result = {
            "total_score": str(_round_val(total_score, 6)),
            "category_scores": {k: str(_round_val(v, 6)) for k, v in cat_scores.items()},
            "category_contributions_pct": {k: str(v) for k, v in contributions.items()},
            "categories_assessed": len(cat_totals),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info("Calculated PEF score: %s across %d categories in %.3f ms", result["total_score"], len(cat_totals), elapsed_ms)
        return result

    # ------------------------------------------------------------------ #
    # Assess Data Quality                                                   #
    # ------------------------------------------------------------------ #

    def assess_data_quality(self, impacts: List[LifecycleImpact]) -> Dict[str, Any]:
        """Assess the data quality of lifecycle impact data.

        Args:
            impacts: List of lifecycle impact data points.

        Returns:
            Dict with keys: result (DataQualityResult), provenance_hash.
        """
        t0 = time.perf_counter()
        total = len(impacts)

        quality_dist: Dict[str, int] = {r.value: 0 for r in DataQualityRating}
        for imp in impacts:
            quality_dist[imp.data_quality_rating.value] = quality_dist.get(imp.data_quality_rating.value, 0) + 1

        dqr_score = self._calculate_dqr_score(impacts)
        primary_count = sum(1 for i in impacts if i.is_primary_data)
        secondary_count = total - primary_count
        primary_ratio = self._calculate_primary_ratio(impacts)
        meets_threshold = dqr_score >= MINIMUM_DQR_THRESHOLD

        phase_quality = self._calculate_phase_quality(impacts)
        category_quality = self._calculate_category_quality(impacts)

        issues: List[str] = []
        recommendations: List[str] = []

        if not meets_threshold:
            issues.append(f"Overall DQR score ({dqr_score}%) is below PEF threshold of {MINIMUM_DQR_THRESHOLD}%")
            recommendations.append("Replace poor/fair quality data with measured or verified primary data")

        poor_count = quality_dist.get(DataQualityRating.POOR.value, 0)
        if poor_count > 0:
            issues.append(f"{poor_count} data point(s) have poor quality rating")
            recommendations.append("Replace poor-quality data with higher-quality sources")

        if primary_ratio < Decimal("50"):
            issues.append(f"Primary data ratio ({primary_ratio}%) is low")
            recommendations.append("Collect primary measurement data for key lifecycle phases")

        phases_covered = {i.phase.value for i in impacts}
        missing_phases = {p.value for p in LifecyclePhase} - phases_covered
        if missing_phases:
            issues.append(f"No impact data for phases: {', '.join(sorted(missing_phases))}")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        result = DataQualityResult(
            total_data_points=total, quality_distribution=quality_dist,
            overall_dqr_score=dqr_score, primary_data_count=primary_count,
            secondary_data_count=secondary_count, primary_data_ratio=primary_ratio,
            meets_pef_threshold=meets_threshold,
            phase_quality={k: str(v) for k, v in phase_quality.items()},
            category_quality={k: str(v) for k, v in category_quality.items()},
            issues=issues, recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Assessed data quality: %d points, DQR=%s%%, primary=%s%%, meets PEF=%s in %.3f ms",
                     total, dqr_score, primary_ratio, meets_threshold, elapsed_ms)
        return {"result": result, "provenance_hash": result.provenance_hash}

    # ------------------------------------------------------------------ #
    # Validate System Boundary                                              #
    # ------------------------------------------------------------------ #

    def validate_system_boundary(
        self, phases: List[str], claim_scope: str,
        system_boundary: str = SystemBoundaryType.CRADLE_TO_GRAVE.value,
    ) -> Dict[str, Any]:
        """Validate the system boundary against claim scope.

        Args:
            phases: Lifecycle phases covered in the assessment.
            claim_scope: Description of the environmental claim scope.
            system_boundary: Declared system boundary type.

        Returns:
            Dict with keys: result (SystemBoundaryResult), provenance_hash.
        """
        t0 = time.perf_counter()

        required = BOUNDARY_REQUIRED_PHASES.get(system_boundary, [])
        covered = set(phases)
        required_set = set(required)
        missing = sorted(required_set - covered)
        coverage_pct = _safe_divide(
            _decimal(len(required_set) - len(missing)) * Decimal("100"),
            _decimal(len(required_set)) if required_set else Decimal("1"),
        )

        is_valid = len(missing) == 0 and len(phases) > 0
        boundary_appropriate = self._is_boundary_appropriate(system_boundary, claim_scope)

        issues: List[str] = []
        recommendations: List[str] = []

        if missing:
            issues.append(f"System boundary '{system_boundary}' missing phases: {missing}")
            recommendations.append(f"Add impact data for: {', '.join(missing)}")
        if not boundary_appropriate:
            issues.append(f"Boundary '{system_boundary}' may not suit claim scope: {claim_scope}")
            recommendations.append("Consider cradle-to-grave for full product claims")
        if not phases:
            issues.append("No lifecycle phases provided")
            recommendations.append("Include impact data for required boundary phases")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        result = SystemBoundaryResult(
            system_boundary=system_boundary, claim_scope=claim_scope,
            required_phases=sorted(required_set), covered_phases=sorted(covered),
            missing_phases=missing, coverage_pct=_round_val(coverage_pct, 2),
            is_valid=is_valid, boundary_appropriate_for_claim=boundary_appropriate,
            issues=issues, recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Validated boundary '%s': valid=%s, coverage=%s%%, appropriate=%s in %.3f ms",
                     system_boundary, is_valid, result.coverage_pct, boundary_appropriate, elapsed_ms)
        return {"result": result, "provenance_hash": result.provenance_hash}

    # ------------------------------------------------------------------ #
    # Private Methods                                                       #
    # ------------------------------------------------------------------ #

    def _identify_hotspots_internal(
        self, impacts: List[LifecycleImpact], top_n: int,
    ) -> List[Dict[str, str]]:
        """Identify top N hotspots by weighted score."""
        scored: List[Tuple[str, str, Decimal]] = []
        for imp in impacts:
            w = PEF_WEIGHTING_FACTORS.get(imp.category.value, Decimal("0"))
            weighted = _round_val(abs(imp.value) * w / Decimal("100"), 6)
            scored.append((imp.phase.value, imp.category.value, weighted))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [
            {"phase": p, "category": c, "weighted_score": str(_round_val(s, 6)),
             "unit": PEF_CATEGORY_UNITS.get(c, "")}
            for p, c, s in scored[:top_n]
        ]

    def _calculate_dqr_score(self, impacts: List[LifecycleImpact]) -> Decimal:
        """Calculate overall data quality ratio score (0-100)."""
        if not impacts:
            return Decimal("0.00")
        total = sum(DATA_QUALITY_SCORES.get(i.data_quality_rating.value, Decimal("20")) for i in impacts)
        return _round_val(_safe_divide(total, _decimal(len(impacts))), 2)

    def _calculate_primary_ratio(self, impacts: List[LifecycleImpact]) -> Decimal:
        """Calculate primary data ratio (0-100)."""
        if not impacts:
            return Decimal("0.00")
        pc = sum(1 for i in impacts if i.is_primary_data)
        return _round_val(_safe_divide(_decimal(pc) * Decimal("100"), _decimal(len(impacts))), 2)

    def _calculate_phase_quality(self, impacts: List[LifecycleImpact]) -> Dict[str, Decimal]:
        """Calculate average DQR score per lifecycle phase."""
        phase_scores: Dict[str, List[Decimal]] = {}
        for imp in impacts:
            s = DATA_QUALITY_SCORES.get(imp.data_quality_rating.value, Decimal("20"))
            phase_scores.setdefault(imp.phase.value, []).append(s)
        return {p: _round_val(_safe_divide(sum(ss), _decimal(len(ss))), 2) for p, ss in phase_scores.items()}

    def _calculate_category_quality(self, impacts: List[LifecycleImpact]) -> Dict[str, Decimal]:
        """Calculate average DQR score per impact category."""
        cat_scores: Dict[str, List[Decimal]] = {}
        for imp in impacts:
            s = DATA_QUALITY_SCORES.get(imp.data_quality_rating.value, Decimal("20"))
            cat_scores.setdefault(imp.category.value, []).append(s)
        return {c: _round_val(_safe_divide(sum(ss), _decimal(len(ss))), 2) for c, ss in cat_scores.items()}

    def _is_boundary_appropriate(self, system_boundary: str, claim_scope: str) -> bool:
        """Check whether the system boundary suits the claim scope."""
        scope_lower = claim_scope.lower()
        full_indicators = [
            "full lifecycle", "entire lifecycle", "cradle to grave",
            "overall", "total", "whole product", "carbon neutral",
            "net zero", "climate positive", "eco-friendly",
            "sustainable", "environmentally friendly",
        ]
        if any(ind in scope_lower for ind in full_indicators):
            return system_boundary == SystemBoundaryType.CRADLE_TO_GRAVE.value

        mfg_indicators = ["manufacturing", "production", "factory", "processing"]
        if any(ind in scope_lower for ind in mfg_indicators):
            return system_boundary in (
                SystemBoundaryType.GATE_TO_GATE.value,
                SystemBoundaryType.CRADLE_TO_GATE.value,
                SystemBoundaryType.CRADLE_TO_GRAVE.value,
            )
        return True
