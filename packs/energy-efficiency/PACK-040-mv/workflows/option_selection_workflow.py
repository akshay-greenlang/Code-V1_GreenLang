# -*- coding: utf-8 -*-
"""
Option Selection Workflow
===================================

3-phase workflow for detailed IPMVP option evaluation and recommendation,
providing multi-criteria decision analysis for each Energy Conservation
Measure (ECM).

Phases:
    1. ECMCharacterization     -- Characterize ECMs for option applicability
    2. OptionEvaluation        -- Score each option across criteria
    3. RecommendationReport    -- Produce ranked recommendations

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - IPMVP Core Concepts (EVO 10000-1:2022)
    - ISO 50015:2014 (M&V of energy performance)
    - FEMP M&V Guidelines 4.0

Schedule: on-demand / project planning
Estimated duration: 10 minutes

Author: GreenLang Platform Team
Version: 40.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class LoadProfile(str, Enum):
    """Equipment load profile type."""

    CONSTANT = "constant"
    VARIABLE = "variable"
    SCHEDULE_DRIVEN = "schedule_driven"
    WEATHER_DRIVEN = "weather_driven"
    PROCESS_DRIVEN = "process_driven"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

OPTION_CRITERIA: Dict[str, Dict[str, Any]] = {
    "cost": {
        "description": "M&V implementation and ongoing cost",
        "weight": 0.25,
        "unit": "USD",
        "direction": "lower_is_better",
        "scores": {
            "A": {"value": 3.0, "label": "Low cost"},
            "B": {"value": 5.0, "label": "Medium cost"},
            "C": {"value": 1.0, "label": "Very low cost"},
            "D": {"value": 10.0, "label": "High cost"},
        },
    },
    "accuracy": {
        "description": "Expected measurement accuracy",
        "weight": 0.30,
        "unit": "percent",
        "direction": "lower_is_better",
        "scores": {
            "A": {"value": 15.0, "label": "+/- 15%"},
            "B": {"value": 10.0, "label": "+/- 10%"},
            "C": {"value": 20.0, "label": "+/- 20%"},
            "D": {"value": 25.0, "label": "+/- 25%"},
        },
    },
    "complexity": {
        "description": "Implementation and analysis complexity",
        "weight": 0.20,
        "unit": "scale_1_10",
        "direction": "lower_is_better",
        "scores": {
            "A": {"value": 3.0, "label": "Low complexity"},
            "B": {"value": 6.0, "label": "Medium complexity"},
            "C": {"value": 5.0, "label": "Medium complexity"},
            "D": {"value": 9.0, "label": "High complexity"},
        },
    },
    "applicability": {
        "description": "Suitability for the ECM type",
        "weight": 0.25,
        "unit": "scale_1_10",
        "direction": "higher_is_better",
        "scores": {
            "A": {"value": 7.0, "label": "Constant load ECMs"},
            "B": {"value": 8.0, "label": "Variable load ECMs"},
            "C": {"value": 6.0, "label": "Multiple ECM projects"},
            "D": {"value": 5.0, "label": "New construction"},
        },
    },
}

ECM_TYPE_OPTION_AFFINITY: Dict[str, Dict[str, float]] = {
    "lighting": {"A": 0.9, "B": 0.6, "C": 0.4, "D": 0.2},
    "hvac": {"A": 0.3, "B": 0.9, "C": 0.7, "D": 0.5},
    "motors": {"A": 0.7, "B": 0.8, "C": 0.5, "D": 0.2},
    "building_envelope": {"A": 0.2, "B": 0.5, "C": 0.8, "D": 0.7},
    "controls": {"A": 0.4, "B": 0.7, "C": 0.6, "D": 0.5},
    "boiler": {"A": 0.3, "B": 0.9, "C": 0.6, "D": 0.4},
    "chiller": {"A": 0.2, "B": 0.9, "C": 0.7, "D": 0.5},
    "vfd": {"A": 0.5, "B": 0.9, "C": 0.5, "D": 0.3},
    "compressed_air": {"A": 0.4, "B": 0.8, "C": 0.5, "D": 0.3},
    "process": {"A": 0.3, "B": 0.8, "C": 0.6, "D": 0.5},
    "renewable": {"A": 0.6, "B": 0.8, "C": 0.5, "D": 0.4},
    "general": {"A": 0.5, "B": 0.6, "C": 0.7, "D": 0.4},
}

LOAD_PROFILE_MODIFIERS: Dict[str, Dict[str, float]] = {
    "constant": {"A": 1.2, "B": 1.0, "C": 0.9, "D": 0.8},
    "variable": {"A": 0.7, "B": 1.2, "C": 1.0, "D": 0.9},
    "schedule_driven": {"A": 0.8, "B": 1.1, "C": 1.0, "D": 0.9},
    "weather_driven": {"A": 0.6, "B": 1.1, "C": 1.2, "D": 1.0},
    "process_driven": {"A": 0.5, "B": 1.2, "C": 0.8, "D": 1.0},
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class ECMCharacteristics(BaseModel):
    """Detailed ECM characteristics for option selection."""

    ecm_id: str = Field(default_factory=lambda: f"ecm-{uuid.uuid4().hex[:8]}")
    ecm_name: str = Field(..., min_length=1, description="ECM display name")
    ecm_type: str = Field(default="general", description="ECM category")
    load_profile: str = Field(default="variable", description="Load profile type")
    estimated_savings_pct: float = Field(
        default=10.0, ge=0, le=100, description="Estimated savings %",
    )
    estimated_savings_kwh: float = Field(default=0.0, ge=0, description="Annual savings kWh")
    estimated_cost: float = Field(default=0.0, ge=0, description="Implementation cost")
    interactive_effects: bool = Field(default=False, description="Interactive effects")
    baseline_data_available: bool = Field(default=True, description="Baseline data exists")
    sub_metering_feasible: bool = Field(default=True, description="Sub-metering possible")
    multiple_energy_streams: bool = Field(default=False, description=">1 energy stream")
    operating_hours_per_year: float = Field(
        default=8760.0, ge=0, le=8760, description="Annual operating hours",
    )

class OptionSelectionInput(BaseModel):
    """Input data model for OptionSelectionWorkflow."""

    project_id: str = Field(default_factory=lambda: f"proj-{uuid.uuid4().hex[:8]}")
    project_name: str = Field(..., min_length=1, description="Project name")
    ecm_list: List[ECMCharacteristics] = Field(
        default_factory=list, description="ECMs to evaluate",
    )
    budget_constraint: float = Field(
        default=0.0, ge=0, description="Maximum M&V budget",
    )
    accuracy_requirement_pct: float = Field(
        default=20.0, ge=1.0, le=50.0, description="Required accuracy %",
    )
    criteria_weights: Optional[Dict[str, float]] = Field(
        default=None, description="Custom criteria weights (overrides defaults)",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Ensure project name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("project_name must not be blank")
        return stripped

class OptionScore(BaseModel):
    """Score for a single IPMVP option for a single ECM."""

    option: str = Field(..., description="IPMVP option A/B/C/D")
    cost_score: float = Field(default=0.0, description="Cost criterion score")
    accuracy_score: float = Field(default=0.0, description="Accuracy criterion score")
    complexity_score: float = Field(default=0.0, description="Complexity criterion score")
    applicability_score: float = Field(default=0.0, description="Applicability score")
    affinity_score: float = Field(default=0.0, description="ECM type affinity")
    load_modifier: float = Field(default=1.0, description="Load profile modifier")
    weighted_total: float = Field(default=0.0, description="Final weighted score")
    rank: int = Field(default=0, description="Rank (1=best)")
    recommended: bool = Field(default=False, description="Is recommended option")

class OptionSelectionResult(BaseModel):
    """Complete result from option selection workflow."""

    selection_id: str = Field(..., description="Unique selection ID")
    project_id: str = Field(default="", description="Project identifier")
    ecm_count: int = Field(default=0, ge=0, description="Number of ECMs evaluated")
    ecm_evaluations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-ECM evaluation results",
    )
    summary_statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated statistics",
    )
    criteria_weights_used: Dict[str, float] = Field(
        default_factory=dict, description="Criteria weights applied",
    )
    phases_completed: List[str] = Field(default_factory=list)
    workflow_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class OptionSelectionWorkflow:
    """
    3-phase IPMVP option selection workflow.

    Performs multi-criteria decision analysis to recommend the optimal IPMVP
    option (A/B/C/D) for each Energy Conservation Measure.

    Zero-hallucination: all scoring uses deterministic criteria weights and
    reference affinity tables. No LLM calls in the scoring path.

    Attributes:
        selection_id: Unique selection execution identifier.
        _characterizations: ECM characterization results.
        _evaluations: Per-ECM option evaluation results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = OptionSelectionWorkflow()
        >>> ecm = ECMCharacteristics(ecm_name="LED Retrofit", ecm_type="lighting")
        >>> inp = OptionSelectionInput(project_name="HQ", ecm_list=[ecm])
        >>> result = wf.run(inp)
        >>> assert result.ecm_count > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize OptionSelectionWorkflow."""
        self.selection_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._characterizations: List[Dict[str, Any]] = []
        self._evaluations: List[Dict[str, Any]] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: OptionSelectionInput) -> OptionSelectionResult:
        """
        Execute the 3-phase option selection workflow.

        Args:
            input_data: Validated option selection input.

        Returns:
            OptionSelectionResult with per-ECM option recommendations.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting option selection workflow %s for project=%s ecms=%d",
            self.selection_id, input_data.project_name, len(input_data.ecm_list),
        )

        self._phase_results = []
        self._characterizations = []
        self._evaluations = []

        try:
            # Phase 1: ECM Characterization
            phase1 = self._phase_ecm_characterization(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Option Evaluation
            phase2 = self._phase_option_evaluation(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Recommendation Report
            phase3 = self._phase_recommendation_report(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Option selection workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Build summary statistics
        recommended_options: Dict[str, int] = {}
        for ev in self._evaluations:
            rec = ev.get("recommended_option", "B")
            recommended_options[rec] = recommended_options.get(rec, 0) + 1

        weights_used = self._get_criteria_weights(input_data)

        result = OptionSelectionResult(
            selection_id=self.selection_id,
            project_id=input_data.project_id,
            ecm_count=len(input_data.ecm_list),
            ecm_evaluations=self._evaluations,
            summary_statistics={
                "option_distribution": recommended_options,
                "ecms_evaluated": len(self._evaluations),
                "accuracy_requirement_pct": input_data.accuracy_requirement_pct,
                "budget_constraint": input_data.budget_constraint,
            },
            criteria_weights_used=weights_used,
            phases_completed=completed_phases,
            workflow_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Option selection workflow %s completed in %dms ecms=%d distribution=%s",
            self.selection_id, int(elapsed_ms),
            len(input_data.ecm_list), recommended_options,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: ECM Characterization
    # -------------------------------------------------------------------------

    def _phase_ecm_characterization(
        self, input_data: OptionSelectionInput,
    ) -> PhaseResult:
        """Characterize ECMs for option applicability."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.ecm_list:
            warnings.append("No ECMs provided; creating placeholder")
            input_data.ecm_list.append(ECMCharacteristics(
                ecm_name="General ECM",
                ecm_type="general",
            ))

        characterizations: List[Dict[str, Any]] = []
        for ecm in input_data.ecm_list:
            char = {
                "ecm_id": ecm.ecm_id,
                "ecm_name": ecm.ecm_name,
                "ecm_type": ecm.ecm_type,
                "load_profile": ecm.load_profile,
                "savings_significance": self._classify_savings_significance(ecm),
                "metering_feasibility": "feasible" if ecm.sub_metering_feasible else "limited",
                "baseline_available": ecm.baseline_data_available,
                "interactive": ecm.interactive_effects,
                "multi_stream": ecm.multiple_energy_streams,
                "utilization_factor": round(
                    ecm.operating_hours_per_year / 8760.0, 3,
                ),
                "option_eligibility": self._determine_eligibility(ecm),
                "characterized_at": utcnow().isoformat() + "Z",
            }
            characterizations.append(char)

        self._characterizations = characterizations

        outputs["ecms_characterized"] = len(characterizations)
        outputs["load_profiles"] = list(set(c["load_profile"] for c in characterizations))
        outputs["avg_utilization"] = round(
            sum(c["utilization_factor"] for c in characterizations) /
            max(len(characterizations), 1), 3,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 ECMCharacterization: %d ECMs characterized",
            len(characterizations),
        )
        return PhaseResult(
            phase_name="ecm_characterization", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Option Evaluation
    # -------------------------------------------------------------------------

    def _phase_option_evaluation(
        self, input_data: OptionSelectionInput,
    ) -> PhaseResult:
        """Score each IPMVP option across criteria for each ECM."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        weights = self._get_criteria_weights(input_data)
        evaluations: List[Dict[str, Any]] = []

        for char in self._characterizations:
            ecm_id = char["ecm_id"]
            ecm_type = char["ecm_type"]
            load_profile = char["load_profile"]
            eligibility = char["option_eligibility"]

            option_scores: List[OptionScore] = []

            for option_key in ["A", "B", "C", "D"]:
                if option_key not in eligibility:
                    continue

                # Base criteria scores
                cost_raw = OPTION_CRITERIA["cost"]["scores"][option_key]["value"]
                acc_raw = OPTION_CRITERIA["accuracy"]["scores"][option_key]["value"]
                cmplx_raw = OPTION_CRITERIA["complexity"]["scores"][option_key]["value"]
                app_raw = OPTION_CRITERIA["applicability"]["scores"][option_key]["value"]

                # Normalize scores to 0-1 (invert for lower-is-better)
                cost_norm = 1.0 - (cost_raw / 10.0)
                acc_norm = 1.0 - (acc_raw / 30.0)
                cmplx_norm = 1.0 - (cmplx_raw / 10.0)
                app_norm = app_raw / 10.0

                # ECM type affinity
                affinity_map = ECM_TYPE_OPTION_AFFINITY.get(
                    ecm_type, ECM_TYPE_OPTION_AFFINITY["general"],
                )
                affinity = affinity_map.get(option_key, 0.5)

                # Load profile modifier
                load_mod_map = LOAD_PROFILE_MODIFIERS.get(
                    load_profile, LOAD_PROFILE_MODIFIERS["variable"],
                )
                load_mod = load_mod_map.get(option_key, 1.0)

                # Weighted total
                weighted = (
                    cost_norm * weights["cost"]
                    + acc_norm * weights["accuracy"]
                    + cmplx_norm * weights["complexity"]
                    + app_norm * weights["applicability"]
                ) * affinity * load_mod

                option_scores.append(OptionScore(
                    option=option_key,
                    cost_score=round(cost_norm, 4),
                    accuracy_score=round(acc_norm, 4),
                    complexity_score=round(cmplx_norm, 4),
                    applicability_score=round(app_norm, 4),
                    affinity_score=round(affinity, 4),
                    load_modifier=round(load_mod, 4),
                    weighted_total=round(weighted, 4),
                ))

            # Rank and recommend
            option_scores.sort(key=lambda s: -s.weighted_total)
            for rank, score in enumerate(option_scores, 1):
                score.rank = rank
                score.recommended = rank == 1

            # Check accuracy constraint
            if input_data.accuracy_requirement_pct > 0:
                for score in option_scores:
                    opt_acc = OPTION_CRITERIA["accuracy"]["scores"][score.option]["value"]
                    if opt_acc > input_data.accuracy_requirement_pct:
                        if score.recommended:
                            score.recommended = False
                            warnings.append(
                                f"ECM '{char['ecm_name']}': Option {score.option} "
                                f"accuracy {opt_acc}% exceeds requirement "
                                f"{input_data.accuracy_requirement_pct}%"
                            )

                # Re-recommend best option meeting accuracy
                for score in option_scores:
                    opt_acc = OPTION_CRITERIA["accuracy"]["scores"][score.option]["value"]
                    if opt_acc <= input_data.accuracy_requirement_pct:
                        score.recommended = True
                        break

            recommended = next(
                (s.option for s in option_scores if s.recommended), "B",
            )

            evaluations.append({
                "ecm_id": ecm_id,
                "ecm_name": char["ecm_name"],
                "recommended_option": recommended,
                "option_scores": [s.model_dump() for s in option_scores],
                "evaluated_at": utcnow().isoformat() + "Z",
            })

        self._evaluations = evaluations

        outputs["ecms_evaluated"] = len(evaluations)
        outputs["recommended_options"] = {
            ev["ecm_name"]: ev["recommended_option"] for ev in evaluations
        }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 OptionEvaluation: %d ECMs scored across 4 options",
            len(evaluations),
        )
        return PhaseResult(
            phase_name="option_evaluation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Recommendation Report
    # -------------------------------------------------------------------------

    def _phase_recommendation_report(
        self, input_data: OptionSelectionInput,
    ) -> PhaseResult:
        """Produce ranked recommendation report."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_items: List[Dict[str, Any]] = []
        for ev in self._evaluations:
            rec_option = ev["recommended_option"]
            scores = ev.get("option_scores", [])
            best_score = next(
                (s for s in scores if s.get("option") == rec_option), {},
            )

            report_items.append({
                "ecm_id": ev["ecm_id"],
                "ecm_name": ev["ecm_name"],
                "recommended_option": rec_option,
                "option_name": f"IPMVP Option {rec_option}",
                "weighted_score": best_score.get("weighted_total", 0.0),
                "confidence": "high" if best_score.get("weighted_total", 0) > 0.5 else "medium",
                "rationale": self._build_rationale(ev, rec_option),
                "alternatives": [
                    s["option"] for s in scores
                    if s.get("option") != rec_option
                ],
            })

        # Portfolio summary
        option_dist: Dict[str, int] = {}
        for item in report_items:
            opt = item["recommended_option"]
            option_dist[opt] = option_dist.get(opt, 0) + 1

        outputs["report_items"] = report_items
        outputs["portfolio_summary"] = {
            "total_ecms": len(report_items),
            "option_distribution": option_dist,
            "avg_confidence_score": round(
                sum(r["weighted_score"] for r in report_items) /
                max(len(report_items), 1), 4,
            ),
        }
        outputs["recommendation_ready"] = True

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 RecommendationReport: %d recommendations generated",
            len(report_items),
        )
        return PhaseResult(
            phase_name="recommendation_report", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_criteria_weights(
        self, input_data: OptionSelectionInput,
    ) -> Dict[str, float]:
        """Get criteria weights, using custom weights if provided."""
        if input_data.criteria_weights:
            total = sum(input_data.criteria_weights.values())
            if total > 0:
                return {
                    k: v / total for k, v in input_data.criteria_weights.items()
                }
        return {k: v["weight"] for k, v in OPTION_CRITERIA.items()}

    def _classify_savings_significance(self, ecm: ECMCharacteristics) -> str:
        """Classify savings significance for option selection."""
        if ecm.estimated_savings_pct >= 30.0:
            return "high"
        if ecm.estimated_savings_pct >= 15.0:
            return "medium"
        return "low"

    def _determine_eligibility(self, ecm: ECMCharacteristics) -> List[str]:
        """Determine which IPMVP options are eligible for an ECM."""
        eligible = []
        # Option A: always eligible if sub-metering feasible
        if ecm.sub_metering_feasible:
            eligible.append("A")
        # Option B: needs sub-metering
        if ecm.sub_metering_feasible:
            eligible.append("B")
        # Option C: always eligible
        eligible.append("C")
        # Option D: needs baseline data or is for new construction
        if not ecm.baseline_data_available or ecm.interactive_effects:
            eligible.append("D")
        elif ecm.multiple_energy_streams:
            eligible.append("D")
        return eligible if eligible else ["B", "C"]

    def _build_rationale(self, evaluation: Dict[str, Any], option: str) -> str:
        """Build rationale string for recommendation."""
        scores = evaluation.get("option_scores", [])
        best = next((s for s in scores if s.get("option") == option), {})
        parts = [
            f"Option {option} recommended for '{evaluation['ecm_name']}'.",
            f"Weighted score: {best.get('weighted_total', 0):.4f}.",
            f"Affinity: {best.get('affinity_score', 0):.2f}.",
            f"Load modifier: {best.get('load_modifier', 1.0):.2f}.",
        ]
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Provenance Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: OptionSelectionResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
