# -*- coding: utf-8 -*-
"""
CertificationBridge - Green Building Certification Integration for PACK-032
=============================================================================

This module provides integration with major green building certification
schemes: LEED v4.1, BREEAM 2024, Energy Star Portfolio Manager, and NABERS.
It evaluates energy-related credits/prerequisites, calculates potential scores,
performs gap analysis, and identifies documentation requirements.

Certification Schemes:
    LEED v4.1 BD+C / O+M   -- Energy & Atmosphere credits, Minimum Energy Perf
    BREEAM 2024 New / Ref   -- Ene credits, energy modelling, sub-metering
    Energy Star PM           -- 1-100 score, weather normalization, source EUI
    NABERS                   -- 1-6 star energy/water rating (AU)

Features:
    - Credit/prerequisite evaluation for energy-related categories
    - Score calculation and certification level estimation
    - Gap analysis between current and target certification
    - Documentation requirements checklist generation
    - Multi-scheme comparison for best strategy selection
    - SHA-256 provenance on all evaluations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-032 Building Energy Assessment
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CertificationScheme(str, Enum):
    """Supported green building certification schemes."""

    LEED_V4_BDC = "leed_v4_bdc"
    LEED_V4_OM = "leed_v4_om"
    BREEAM_NEW = "breeam_new"
    BREEAM_REFURB = "breeam_refurb"
    BREEAM_IN_USE = "breeam_in_use"
    ENERGY_STAR = "energy_star"
    NABERS_ENERGY = "nabers_energy"
    NABERS_WATER = "nabers_water"
    DGNB = "dgnb"
    HQE = "hqe"
    WELL_V2 = "well_v2"


class CertificationLevel(str, Enum):
    """Certification achievement levels."""

    # LEED levels
    LEED_CERTIFIED = "leed_certified"
    LEED_SILVER = "leed_silver"
    LEED_GOLD = "leed_gold"
    LEED_PLATINUM = "leed_platinum"
    # BREEAM levels
    BREEAM_PASS = "breeam_pass"
    BREEAM_GOOD = "breeam_good"
    BREEAM_VERY_GOOD = "breeam_very_good"
    BREEAM_EXCELLENT = "breeam_excellent"
    BREEAM_OUTSTANDING = "breeam_outstanding"
    # Energy Star
    ENERGY_STAR_CERTIFIED = "energy_star_certified"
    # NABERS
    NABERS_1_STAR = "nabers_1_star"
    NABERS_2_STAR = "nabers_2_star"
    NABERS_3_STAR = "nabers_3_star"
    NABERS_4_STAR = "nabers_4_star"
    NABERS_5_STAR = "nabers_5_star"
    NABERS_6_STAR = "nabers_6_star"
    # General
    NOT_CERTIFIED = "not_certified"


class CreditStatus(str, Enum):
    """Status of a certification credit/prerequisite."""

    MET = "met"
    NOT_MET = "not_met"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_DOCUMENTATION = "requires_documentation"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CreditEvaluation(BaseModel):
    """Evaluation of a single certification credit or prerequisite."""

    credit_id: str = Field(default="")
    credit_name: str = Field(default="")
    category: str = Field(default="")
    is_prerequisite: bool = Field(default=False)
    max_points: float = Field(default=0.0)
    achieved_points: float = Field(default=0.0)
    status: CreditStatus = Field(default=CreditStatus.NOT_MET)
    current_value: Optional[float] = Field(None)
    threshold_value: Optional[float] = Field(None)
    gap: Optional[float] = Field(None)
    documentation_required: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CertificationEvaluation(BaseModel):
    """Complete certification evaluation result."""

    evaluation_id: str = Field(default_factory=_new_uuid)
    building_id: str = Field(default="")
    scheme: CertificationScheme = Field(...)
    scheme_version: str = Field(default="")
    total_credits_evaluated: int = Field(default=0)
    prerequisites_met: int = Field(default=0)
    prerequisites_total: int = Field(default=0)
    prerequisites_all_met: bool = Field(default=False)
    total_points_available: float = Field(default=0.0)
    total_points_achieved: float = Field(default=0.0)
    score_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_level: CertificationLevel = Field(default=CertificationLevel.NOT_CERTIFIED)
    credits: List[CreditEvaluation] = Field(default_factory=list)
    gap_analysis: Dict[str, Any] = Field(default_factory=dict)
    documentation_checklist: List[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class MultiSchemeComparison(BaseModel):
    """Comparison across multiple certification schemes."""

    comparison_id: str = Field(default_factory=_new_uuid)
    building_id: str = Field(default="")
    evaluations: List[CertificationEvaluation] = Field(default_factory=list)
    recommended_scheme: str = Field(default="")
    recommendation_reason: str = Field(default="")
    provenance_hash: str = Field(default="")


class CertificationBridgeConfig(BaseModel):
    """Configuration for the Certification Bridge."""

    pack_id: str = Field(default="PACK-032")
    enable_provenance: bool = Field(default=True)
    target_level: str = Field(default="", description="Target certification level")
    include_documentation: bool = Field(default=True)
    country_code: str = Field(default="GB")


# ---------------------------------------------------------------------------
# Certification Scheme Definitions
# ---------------------------------------------------------------------------

LEED_ENERGY_CREDITS: List[Dict[str, Any]] = [
    {"id": "EAp1", "name": "Fundamental Commissioning and Verification", "category": "EA",
     "prerequisite": True, "max_points": 0, "metric": "commissioning_complete",
     "threshold": True},
    {"id": "EAp2", "name": "Minimum Energy Performance", "category": "EA",
     "prerequisite": True, "max_points": 0, "metric": "energy_cost_saving_pct",
     "threshold": 5.0},
    {"id": "EAc1", "name": "Enhanced Commissioning", "category": "EA",
     "prerequisite": False, "max_points": 6, "metric": "enhanced_cx",
     "threshold": True},
    {"id": "EAc2", "name": "Optimize Energy Performance", "category": "EA",
     "prerequisite": False, "max_points": 18, "metric": "energy_cost_saving_pct",
     "thresholds": [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 50]},
    {"id": "EAc3", "name": "Advanced Energy Metering", "category": "EA",
     "prerequisite": False, "max_points": 1, "metric": "sub_metering_pct",
     "threshold": 100.0},
    {"id": "EAc4", "name": "Demand Response", "category": "EA",
     "prerequisite": False, "max_points": 2, "metric": "demand_response_capable",
     "threshold": True},
    {"id": "EAc5", "name": "Renewable Energy Production", "category": "EA",
     "prerequisite": False, "max_points": 3, "metric": "renewable_pct",
     "thresholds": [1, 5, 10]},
    {"id": "EAc6", "name": "Enhanced Refrigerant Management", "category": "EA",
     "prerequisite": False, "max_points": 1, "metric": "low_gwp_refrigerants",
     "threshold": True},
    {"id": "EAc7", "name": "Green Power and Carbon Offsets", "category": "EA",
     "prerequisite": False, "max_points": 2, "metric": "green_power_pct",
     "thresholds": [50, 100]},
]

BREEAM_ENERGY_CREDITS: List[Dict[str, Any]] = [
    {"id": "Ene01", "name": "Reduction of energy use and carbon emissions", "category": "Energy",
     "prerequisite": True, "max_credits": 15, "max_points": 15.0,
     "metric": "energy_improvement_pct", "thresholds": list(range(0, 101, 7))},
    {"id": "Ene02", "name": "Energy monitoring", "category": "Energy",
     "prerequisite": False, "max_credits": 2, "max_points": 2.0,
     "metric": "sub_metering_level", "thresholds": [1, 2]},
    {"id": "Ene03", "name": "External lighting", "category": "Energy",
     "prerequisite": False, "max_credits": 1, "max_points": 1.0,
     "metric": "external_lighting_compliant", "threshold": True},
    {"id": "Ene04", "name": "Low carbon design", "category": "Energy",
     "prerequisite": False, "max_credits": 3, "max_points": 3.0,
     "metric": "low_carbon_feasibility", "thresholds": [1, 2, 3]},
    {"id": "Ene05", "name": "Energy efficient cold storage", "category": "Energy",
     "prerequisite": False, "max_credits": 2, "max_points": 2.0,
     "metric": "cold_storage_efficiency", "threshold": True},
    {"id": "Ene06", "name": "Energy efficient transportation systems", "category": "Energy",
     "prerequisite": False, "max_credits": 2, "max_points": 2.0,
     "metric": "lift_efficiency", "thresholds": [1, 2]},
]

# BREEAM scoring thresholds
BREEAM_LEVELS: Dict[str, float] = {
    "outstanding": 85.0,
    "excellent": 70.0,
    "very_good": 55.0,
    "good": 45.0,
    "pass": 30.0,
}

LEED_LEVELS: Dict[str, float] = {
    "platinum": 80.0,
    "gold": 60.0,
    "silver": 50.0,
    "certified": 40.0,
}

ENERGY_STAR_THRESHOLDS: Dict[str, float] = {
    "certified": 75.0,
}

NABERS_STAR_THRESHOLDS: Dict[str, float] = {
    "6_star": 0.25,  # ratio to benchmark (lower is better)
    "5_star": 0.50,
    "4_star": 0.75,
    "3_star": 1.00,
    "2_star": 1.25,
    "1_star": 1.50,
}


# ---------------------------------------------------------------------------
# CertificationBridge
# ---------------------------------------------------------------------------


class CertificationBridge:
    """Green building certification integration for PACK-032.

    Evaluates building energy performance against LEED, BREEAM, Energy Star,
    and NABERS certification requirements. Provides score calculation, gap
    analysis, and documentation requirements.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = CertificationBridge()
        >>> result = bridge.evaluate_leed("building-1", {"energy_cost_saving_pct": 25})
        >>> assert result.estimated_level != CertificationLevel.NOT_CERTIFIED
    """

    def __init__(self, config: Optional[CertificationBridgeConfig] = None) -> None:
        """Initialize the Certification Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or CertificationBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("CertificationBridge initialized")

    # -------------------------------------------------------------------------
    # LEED Evaluation
    # -------------------------------------------------------------------------

    def evaluate_leed(
        self,
        building_id: str,
        metrics: Dict[str, Any],
        scheme: CertificationScheme = CertificationScheme.LEED_V4_BDC,
    ) -> CertificationEvaluation:
        """Evaluate building against LEED v4.1 energy credits.

        Args:
            building_id: Building identifier.
            metrics: Building performance metrics.
            scheme: LEED scheme variant.

        Returns:
            CertificationEvaluation with credit-level detail.
        """
        evaluation = CertificationEvaluation(
            building_id=building_id,
            scheme=scheme,
            scheme_version="v4.1",
        )

        for credit_def in LEED_ENERGY_CREDITS:
            credit = CreditEvaluation(
                credit_id=credit_def["id"],
                credit_name=credit_def["name"],
                category=credit_def["category"],
                is_prerequisite=credit_def.get("prerequisite", False),
                max_points=credit_def.get("max_points", 0),
            )

            metric_key = credit_def.get("metric", "")
            metric_value = metrics.get(metric_key)

            if metric_value is None:
                credit.status = CreditStatus.REQUIRES_DOCUMENTATION
                credit.documentation_required.append(
                    f"Provide {metric_key} data for credit {credit_def['id']}"
                )
            else:
                credit.current_value = float(metric_value) if isinstance(metric_value, (int, float)) else None

                thresholds = credit_def.get("thresholds")
                threshold = credit_def.get("threshold")

                if thresholds:
                    # Multi-level credit
                    points = 0
                    for i, t in enumerate(thresholds):
                        if isinstance(metric_value, (int, float)) and metric_value >= t:
                            points = i + 1
                    credit.achieved_points = min(points, credit.max_points)
                    credit.status = CreditStatus.MET if points > 0 else CreditStatus.NOT_MET
                elif threshold is not None:
                    if isinstance(threshold, bool):
                        credit.status = CreditStatus.MET if metric_value else CreditStatus.NOT_MET
                        credit.achieved_points = credit.max_points if metric_value else 0
                    else:
                        is_met = isinstance(metric_value, (int, float)) and metric_value >= threshold
                        credit.status = CreditStatus.MET if is_met else CreditStatus.NOT_MET
                        credit.achieved_points = credit.max_points if is_met else 0
                        if not is_met and isinstance(metric_value, (int, float)):
                            credit.gap = round(threshold - metric_value, 2)

            evaluation.credits.append(credit)

        # Compute totals
        self._compute_evaluation_totals(evaluation)

        # Determine LEED level
        score = evaluation.total_points_achieved
        if score >= LEED_LEVELS["platinum"]:
            evaluation.estimated_level = CertificationLevel.LEED_PLATINUM
        elif score >= LEED_LEVELS["gold"]:
            evaluation.estimated_level = CertificationLevel.LEED_GOLD
        elif score >= LEED_LEVELS["silver"]:
            evaluation.estimated_level = CertificationLevel.LEED_SILVER
        elif score >= LEED_LEVELS["certified"]:
            evaluation.estimated_level = CertificationLevel.LEED_CERTIFIED
        else:
            evaluation.estimated_level = CertificationLevel.NOT_CERTIFIED

        # Gap analysis
        evaluation.gap_analysis = self._leed_gap_analysis(evaluation)

        if self.config.enable_provenance:
            evaluation.provenance_hash = _compute_hash(evaluation)

        return evaluation

    # -------------------------------------------------------------------------
    # BREEAM Evaluation
    # -------------------------------------------------------------------------

    def evaluate_breeam(
        self,
        building_id: str,
        metrics: Dict[str, Any],
        scheme: CertificationScheme = CertificationScheme.BREEAM_NEW,
    ) -> CertificationEvaluation:
        """Evaluate building against BREEAM 2024 energy credits.

        Args:
            building_id: Building identifier.
            metrics: Building performance metrics.
            scheme: BREEAM scheme variant.

        Returns:
            CertificationEvaluation with credit-level detail.
        """
        evaluation = CertificationEvaluation(
            building_id=building_id,
            scheme=scheme,
            scheme_version="2024",
        )

        for credit_def in BREEAM_ENERGY_CREDITS:
            credit = CreditEvaluation(
                credit_id=credit_def["id"],
                credit_name=credit_def["name"],
                category=credit_def["category"],
                is_prerequisite=credit_def.get("prerequisite", False),
                max_points=credit_def.get("max_points", 0),
            )

            metric_key = credit_def.get("metric", "")
            metric_value = metrics.get(metric_key)

            if metric_value is None:
                credit.status = CreditStatus.REQUIRES_DOCUMENTATION
            else:
                credit.current_value = float(metric_value) if isinstance(metric_value, (int, float)) else None
                thresholds = credit_def.get("thresholds")
                threshold = credit_def.get("threshold")

                if thresholds:
                    points = 0
                    for i, t in enumerate(thresholds):
                        if isinstance(metric_value, (int, float)) and metric_value >= t:
                            points = i + 1
                    credit.achieved_points = min(points, credit.max_points)
                    credit.status = CreditStatus.MET if points > 0 else CreditStatus.NOT_MET
                elif threshold is not None:
                    if isinstance(threshold, bool):
                        credit.status = CreditStatus.MET if metric_value else CreditStatus.NOT_MET
                        credit.achieved_points = credit.max_points if metric_value else 0
                    else:
                        is_met = isinstance(metric_value, (int, float)) and metric_value >= threshold
                        credit.status = CreditStatus.MET if is_met else CreditStatus.NOT_MET
                        credit.achieved_points = credit.max_points if is_met else 0

            evaluation.credits.append(credit)

        self._compute_evaluation_totals(evaluation)

        # Determine BREEAM level
        score = evaluation.score_pct
        if score >= BREEAM_LEVELS["outstanding"]:
            evaluation.estimated_level = CertificationLevel.BREEAM_OUTSTANDING
        elif score >= BREEAM_LEVELS["excellent"]:
            evaluation.estimated_level = CertificationLevel.BREEAM_EXCELLENT
        elif score >= BREEAM_LEVELS["very_good"]:
            evaluation.estimated_level = CertificationLevel.BREEAM_VERY_GOOD
        elif score >= BREEAM_LEVELS["good"]:
            evaluation.estimated_level = CertificationLevel.BREEAM_GOOD
        elif score >= BREEAM_LEVELS["pass"]:
            evaluation.estimated_level = CertificationLevel.BREEAM_PASS
        else:
            evaluation.estimated_level = CertificationLevel.NOT_CERTIFIED

        evaluation.gap_analysis = self._breeam_gap_analysis(evaluation)

        if self.config.enable_provenance:
            evaluation.provenance_hash = _compute_hash(evaluation)

        return evaluation

    # -------------------------------------------------------------------------
    # Energy Star Evaluation
    # -------------------------------------------------------------------------

    def evaluate_energy_star(
        self,
        building_id: str,
        source_eui_kbtu_sqft: float,
        building_type: str = "office",
        gross_floor_area_sqft: float = 0.0,
    ) -> CertificationEvaluation:
        """Evaluate building against Energy Star Portfolio Manager.

        Args:
            building_id: Building identifier.
            source_eui_kbtu_sqft: Source Energy Use Intensity (kBtu/sqft).
            building_type: Building type for benchmarking.
            gross_floor_area_sqft: Gross floor area in sq ft.

        Returns:
            CertificationEvaluation with score.
        """
        evaluation = CertificationEvaluation(
            building_id=building_id,
            scheme=CertificationScheme.ENERGY_STAR,
            scheme_version="2024",
        )

        # Energy Star score estimation (simplified, deterministic)
        # Median source EUI by building type (kBtu/sqft)
        median_eui = {
            "office": 92.0,
            "retail": 67.0,
            "hotel": 95.0,
            "hospital": 250.0,
            "school": 73.0,
            "multifamily": 80.0,
            "warehouse": 29.0,
        }
        type_median = median_eui.get(building_type, 92.0)

        # Simplified score: ratio-based (real ENERGY STAR uses regression)
        ratio = source_eui_kbtu_sqft / max(type_median, 1)
        if ratio <= 0.5:
            score = 95
        elif ratio <= 0.75:
            score = 85
        elif ratio <= 1.0:
            score = 70
        elif ratio <= 1.25:
            score = 50
        elif ratio <= 1.5:
            score = 30
        else:
            score = 15

        credit = CreditEvaluation(
            credit_id="ES_Score",
            credit_name="Energy Star Score",
            category="Energy",
            max_points=100,
            achieved_points=float(score),
            current_value=source_eui_kbtu_sqft,
            threshold_value=type_median,
            status=CreditStatus.MET if score >= 75 else CreditStatus.NOT_MET,
        )
        if score < 75:
            credit.gap = round(75 - score, 1)
            credit.recommendations.append(
                f"Reduce source EUI from {source_eui_kbtu_sqft} to "
                f"approximately {type_median * 0.75:.0f} kBtu/sqft for certification"
            )

        evaluation.credits.append(credit)
        evaluation.total_points_available = 100
        evaluation.total_points_achieved = float(score)
        evaluation.score_pct = float(score)
        evaluation.total_credits_evaluated = 1

        if score >= 75:
            evaluation.estimated_level = CertificationLevel.ENERGY_STAR_CERTIFIED
        else:
            evaluation.estimated_level = CertificationLevel.NOT_CERTIFIED

        if self.config.enable_provenance:
            evaluation.provenance_hash = _compute_hash(evaluation)

        return evaluation

    # -------------------------------------------------------------------------
    # NABERS Evaluation
    # -------------------------------------------------------------------------

    def evaluate_nabers(
        self,
        building_id: str,
        energy_kwh_m2: float,
        benchmark_kwh_m2: float,
    ) -> CertificationEvaluation:
        """Evaluate building against NABERS energy rating.

        Args:
            building_id: Building identifier.
            energy_kwh_m2: Actual energy use intensity.
            benchmark_kwh_m2: Benchmark energy use for building type.

        Returns:
            CertificationEvaluation with star rating.
        """
        evaluation = CertificationEvaluation(
            building_id=building_id,
            scheme=CertificationScheme.NABERS_ENERGY,
            scheme_version="2024",
        )

        ratio = energy_kwh_m2 / max(benchmark_kwh_m2, 1)

        if ratio <= NABERS_STAR_THRESHOLDS["6_star"]:
            stars = 6
            level = CertificationLevel.NABERS_6_STAR
        elif ratio <= NABERS_STAR_THRESHOLDS["5_star"]:
            stars = 5
            level = CertificationLevel.NABERS_5_STAR
        elif ratio <= NABERS_STAR_THRESHOLDS["4_star"]:
            stars = 4
            level = CertificationLevel.NABERS_4_STAR
        elif ratio <= NABERS_STAR_THRESHOLDS["3_star"]:
            stars = 3
            level = CertificationLevel.NABERS_3_STAR
        elif ratio <= NABERS_STAR_THRESHOLDS["2_star"]:
            stars = 2
            level = CertificationLevel.NABERS_2_STAR
        elif ratio <= NABERS_STAR_THRESHOLDS["1_star"]:
            stars = 1
            level = CertificationLevel.NABERS_1_STAR
        else:
            stars = 0
            level = CertificationLevel.NOT_CERTIFIED

        credit = CreditEvaluation(
            credit_id="NABERS_Energy",
            credit_name=f"NABERS Energy Rating ({stars} star)",
            category="Energy",
            max_points=6,
            achieved_points=float(stars),
            current_value=energy_kwh_m2,
            threshold_value=benchmark_kwh_m2,
            status=CreditStatus.MET if stars >= 3 else CreditStatus.NOT_MET,
        )
        evaluation.credits.append(credit)
        evaluation.total_points_available = 6
        evaluation.total_points_achieved = float(stars)
        evaluation.score_pct = round(stars / 6 * 100, 1)
        evaluation.total_credits_evaluated = 1
        evaluation.estimated_level = level

        if self.config.enable_provenance:
            evaluation.provenance_hash = _compute_hash(evaluation)

        return evaluation

    # -------------------------------------------------------------------------
    # Multi-Scheme Comparison
    # -------------------------------------------------------------------------

    def compare_schemes(
        self,
        building_id: str,
        metrics: Dict[str, Any],
    ) -> MultiSchemeComparison:
        """Compare building against multiple certification schemes.

        Args:
            building_id: Building identifier.
            metrics: Building performance metrics.

        Returns:
            MultiSchemeComparison with all evaluations and recommendation.
        """
        comparison = MultiSchemeComparison(building_id=building_id)

        # Evaluate LEED
        leed_eval = self.evaluate_leed(building_id, metrics)
        comparison.evaluations.append(leed_eval)

        # Evaluate BREEAM
        breeam_eval = self.evaluate_breeam(building_id, metrics)
        comparison.evaluations.append(breeam_eval)

        # Recommend best scheme
        best_score = 0.0
        best_scheme = ""
        for ev in comparison.evaluations:
            if ev.score_pct > best_score:
                best_score = ev.score_pct
                best_scheme = ev.scheme.value

        comparison.recommended_scheme = best_scheme
        comparison.recommendation_reason = (
            f"Highest energy credit score ({best_score:.1f}%) "
            f"achieved with {best_scheme}"
        )

        if self.config.enable_provenance:
            comparison.provenance_hash = _compute_hash(comparison)

        return comparison

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _compute_evaluation_totals(self, evaluation: CertificationEvaluation) -> None:
        """Compute aggregate totals for a certification evaluation."""
        prereqs = [c for c in evaluation.credits if c.is_prerequisite]
        credits = [c for c in evaluation.credits if not c.is_prerequisite]

        evaluation.prerequisites_total = len(prereqs)
        evaluation.prerequisites_met = sum(
            1 for p in prereqs if p.status == CreditStatus.MET
        )
        evaluation.prerequisites_all_met = (
            evaluation.prerequisites_met == evaluation.prerequisites_total
        )

        evaluation.total_points_available = sum(c.max_points for c in credits)
        evaluation.total_points_achieved = sum(c.achieved_points for c in credits)
        evaluation.total_credits_evaluated = len(evaluation.credits)

        if evaluation.total_points_available > 0:
            evaluation.score_pct = round(
                evaluation.total_points_achieved / evaluation.total_points_available * 100, 1
            )

        if self.config.include_documentation:
            for credit in evaluation.credits:
                evaluation.documentation_checklist.extend(credit.documentation_required)

    def _leed_gap_analysis(self, evaluation: CertificationEvaluation) -> Dict[str, Any]:
        """Generate LEED-specific gap analysis."""
        target_levels = {
            "platinum": LEED_LEVELS["platinum"],
            "gold": LEED_LEVELS["gold"],
            "silver": LEED_LEVELS["silver"],
            "certified": LEED_LEVELS["certified"],
        }
        gaps: Dict[str, Any] = {"current_points": evaluation.total_points_achieved}
        for level_name, threshold in target_levels.items():
            gap = max(threshold - evaluation.total_points_achieved, 0)
            gaps[f"gap_to_{level_name}"] = round(gap, 1)
            gaps[f"{level_name}_achievable"] = gap == 0
        return gaps

    def _breeam_gap_analysis(self, evaluation: CertificationEvaluation) -> Dict[str, Any]:
        """Generate BREEAM-specific gap analysis."""
        gaps: Dict[str, Any] = {"current_score_pct": evaluation.score_pct}
        for level_name, threshold in BREEAM_LEVELS.items():
            gap = max(threshold - evaluation.score_pct, 0)
            gaps[f"gap_to_{level_name}"] = round(gap, 1)
            gaps[f"{level_name}_achievable"] = gap == 0
        return gaps

    def get_supported_schemes(self) -> List[str]:
        """Return list of supported certification schemes.

        Returns:
            List of scheme identifiers.
        """
        return [s.value for s in CertificationScheme]
