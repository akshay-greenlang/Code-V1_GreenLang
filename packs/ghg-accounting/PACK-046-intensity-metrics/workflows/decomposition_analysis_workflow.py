# -*- coding: utf-8 -*-
"""
Decomposition Analysis Workflow
====================================

3-phase workflow for Logarithmic Mean Divisia Index (LMDI) decomposition
analysis within PACK-046 Intensity Metrics Pack.

Phases:
    1. PeriodSelection            -- Select base and comparison periods, validate
                                     that emissions and denominator data exist for
                                     both periods, check data quality thresholds.
    2. LMDIDecomposition          -- Run the DecompositionEngine to execute additive
                                     or multiplicative LMDI-I decomposition, breaking
                                     total intensity change into activity effect,
                                     structure effect, and intensity effect for each
                                     sub-sector or business unit.
    3. EffectInterpretation       -- Interpret decomposition effects, classify each
                                     as organic improvement, structural change, growth
                                     effect, or methodology artefact; produce narrative
                                     and visualisation data.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic LMDI formulas. SHA-256 provenance hashes
guarantee auditability.

Regulatory Basis:
    IEA Energy Efficiency Indicators methodology (LMDI-I/II)
    ESRS E1 - Understanding emission intensity drivers
    SBTi SDA v2.0 - Decomposition of target progress
    GHG Protocol - Understanding changes in emissions over time

Schedule: Annually or on-demand for multi-year trend analysis
Estimated duration: 1-2 days

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _logarithmic_mean(a: float, b: float) -> float:
    """
    Compute the logarithmic mean L(a, b) used in LMDI decomposition.

    L(a, b) = (a - b) / (ln(a) - ln(b)) when a != b, else a.
    Returns 0 if either value is non-positive.
    """
    if a <= 0 or b <= 0:
        return 0.0
    if abs(a - b) < 1e-12:
        return a
    return (a - b) / (math.log(a) - math.log(b))


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


class DecompPhase(str, Enum):
    """Decomposition analysis workflow phases."""

    PERIOD_SELECTION = "period_selection"
    LMDI_DECOMPOSITION = "lmdi_decomposition"
    EFFECT_INTERPRETATION = "effect_interpretation"


class DecompositionMethod(str, Enum):
    """LMDI decomposition method variant."""

    ADDITIVE_LMDI_I = "additive_lmdi_i"
    MULTIPLICATIVE_LMDI_I = "multiplicative_lmdi_i"
    ADDITIVE_LMDI_II = "additive_lmdi_ii"
    MULTIPLICATIVE_LMDI_II = "multiplicative_lmdi_ii"


class EffectType(str, Enum):
    """Type of decomposition effect."""

    ACTIVITY = "activity"
    STRUCTURE = "structure"
    INTENSITY = "intensity"
    RESIDUAL = "residual"


class ChangeClassification(str, Enum):
    """Classification of an effect's nature."""

    ORGANIC_IMPROVEMENT = "organic_improvement"
    STRUCTURAL_CHANGE = "structural_change"
    GROWTH_EFFECT = "growth_effect"
    METHODOLOGY_ARTEFACT = "methodology_artefact"
    MIXED = "mixed"
    NEGLIGIBLE = "negligible"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class SubSectorData(BaseModel):
    """Emissions and activity data for one sub-sector in a single period."""

    sub_sector_id: str = Field(..., min_length=1)
    sub_sector_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    activity_value: float = Field(default=0.0, ge=0.0, description="Denominator value")
    intensity: float = Field(default=0.0, ge=0.0, description="tCO2e / activity unit")


class PeriodData(BaseModel):
    """Complete data for a single period across all sub-sectors."""

    period: str = Field(..., description="Reporting period, e.g. 2023")
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    total_activity: float = Field(default=0.0, ge=0.0)
    aggregate_intensity: float = Field(default=0.0, ge=0.0)
    sub_sectors: List[SubSectorData] = Field(default_factory=list)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


class DecompositionEffect(BaseModel):
    """A single decomposition effect from LMDI analysis."""

    effect_id: str = Field(default_factory=lambda: f"de-{_new_uuid()[:8]}")
    effect_type: EffectType = Field(...)
    sub_sector_id: str = Field(default="", description="Empty for aggregate effects")
    value_tco2e: float = Field(default=0.0, description="Additive effect in tCO2e")
    value_pct: float = Field(default=0.0, description="Percentage of total change")
    direction: str = Field(default="neutral", description="increase|decrease|neutral")
    provenance_hash: str = Field(default="")


class EffectInterpretation(BaseModel):
    """Interpretation of a decomposition effect."""

    effect_type: EffectType = Field(...)
    classification: ChangeClassification = Field(...)
    narrative: str = Field(default="")
    magnitude_pct: float = Field(default=0.0)
    is_favorable: bool = Field(default=False, description="Favorable = reduces intensity")
    confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class DecompositionWorkflowInput(BaseModel):
    """Input data model for DecompositionAnalysisWorkflow."""

    organization_id: str = Field(..., min_length=1)
    base_period: PeriodData = Field(
        ..., description="Base period data with sub-sector breakdown",
    )
    comparison_period: PeriodData = Field(
        ..., description="Comparison period data with sub-sector breakdown",
    )
    method: DecompositionMethod = Field(
        default=DecompositionMethod.ADDITIVE_LMDI_I,
        description="LMDI decomposition method",
    )
    minimum_data_quality: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum data quality score for both periods",
    )
    significance_threshold_pct: float = Field(
        default=1.0, ge=0.0, le=50.0,
        description="Minimum effect size to classify as non-negligible",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class DecompositionWorkflowResult(BaseModel):
    """Complete result from decomposition analysis workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="decomposition_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    base_period: str = Field(default="")
    comparison_period: str = Field(default="")
    method: DecompositionMethod = Field(default=DecompositionMethod.ADDITIVE_LMDI_I)
    total_change_tco2e: float = Field(default=0.0)
    total_change_pct: float = Field(default=0.0)
    effects: List[DecompositionEffect] = Field(default_factory=list)
    interpretations: List[EffectInterpretation] = Field(default_factory=list)
    residual_tco2e: float = Field(default=0.0, description="Unexplained residual")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DecompositionAnalysisWorkflow:
    """
    3-phase workflow for LMDI decomposition of emissions intensity changes.

    Decomposes total intensity change between two periods into activity,
    structure, and intensity effects using the Logarithmic Mean Divisia
    Index method.

    Zero-hallucination: all decomposition uses deterministic LMDI formulas
    from Ang (2004/2005); no LLM calls in numeric paths; SHA-256 provenance
    on every effect.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _effects: Decomposition effects.
        _interpretations: Effect interpretations.

    Example:
        >>> wf = DecompositionAnalysisWorkflow()
        >>> base = PeriodData(period="2023", total_emissions_tco2e=10000,
        ...     total_activity=100, aggregate_intensity=100.0)
        >>> comp = PeriodData(period="2024", total_emissions_tco2e=9500,
        ...     total_activity=110, aggregate_intensity=86.36)
        >>> inp = DecompositionWorkflowInput(
        ...     organization_id="org-001",
        ...     base_period=base, comparison_period=comp,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[DecompPhase] = [
        DecompPhase.PERIOD_SELECTION,
        DecompPhase.LMDI_DECOMPOSITION,
        DecompPhase.EFFECT_INTERPRETATION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DecompositionAnalysisWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._effects: List[DecompositionEffect] = []
        self._interpretations: List[EffectInterpretation] = []
        self._total_change_tco2e: float = 0.0
        self._total_change_pct: float = 0.0
        self._residual: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: DecompositionWorkflowInput,
    ) -> DecompositionWorkflowResult:
        """
        Execute the 3-phase decomposition analysis workflow.

        Args:
            input_data: Base and comparison period data with sub-sector breakdown.

        Returns:
            DecompositionWorkflowResult with effects and interpretations.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting decomposition analysis %s org=%s %s->%s method=%s",
            self.workflow_id, input_data.organization_id,
            input_data.base_period.period, input_data.comparison_period.period,
            input_data.method.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_period_selection,
            self._phase_2_lmdi_decomposition,
            self._phase_3_effect_interpretation,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Decomposition analysis failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = DecompositionWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            base_period=input_data.base_period.period,
            comparison_period=input_data.comparison_period.period,
            method=input_data.method,
            total_change_tco2e=self._total_change_tco2e,
            total_change_pct=self._total_change_pct,
            effects=self._effects,
            interpretations=self._interpretations,
            residual_tco2e=self._residual,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Decomposition analysis %s completed in %.2fs status=%s effects=%d",
            self.workflow_id, elapsed, overall_status.value, len(self._effects),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Period Selection
    # -------------------------------------------------------------------------

    async def _phase_1_period_selection(
        self, input_data: DecompositionWorkflowInput,
    ) -> PhaseResult:
        """Validate base and comparison period data availability."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base = input_data.base_period
        comp = input_data.comparison_period

        # Validate data quality
        if base.data_quality_score < input_data.minimum_data_quality:
            warnings.append(
                f"Base period quality {base.data_quality_score:.1f} below "
                f"minimum {input_data.minimum_data_quality:.1f}"
            )
        if comp.data_quality_score < input_data.minimum_data_quality:
            warnings.append(
                f"Comparison period quality {comp.data_quality_score:.1f} below "
                f"minimum {input_data.minimum_data_quality:.1f}"
            )

        # Validate sub-sector alignment
        base_ids = set(s.sub_sector_id for s in base.sub_sectors)
        comp_ids = set(s.sub_sector_id for s in comp.sub_sectors)
        common_ids = base_ids & comp_ids
        base_only = base_ids - comp_ids
        comp_only = comp_ids - base_ids

        if base_only:
            warnings.append(f"Sub-sectors in base only: {base_only}")
        if comp_only:
            warnings.append(f"Sub-sectors in comparison only: {comp_only}")

        # Calculate total change
        self._total_change_tco2e = round(
            comp.total_emissions_tco2e - base.total_emissions_tco2e, 6,
        )
        if base.total_emissions_tco2e > 0:
            self._total_change_pct = round(
                (self._total_change_tco2e / base.total_emissions_tco2e) * 100.0, 4,
            )

        # Validate non-zero values
        if base.total_emissions_tco2e <= 0:
            raise ValueError("Base period total emissions must be positive")
        if base.total_activity <= 0:
            raise ValueError("Base period total activity must be positive")

        outputs["base_period"] = base.period
        outputs["comparison_period"] = comp.period
        outputs["base_emissions_tco2e"] = base.total_emissions_tco2e
        outputs["comparison_emissions_tco2e"] = comp.total_emissions_tco2e
        outputs["total_change_tco2e"] = self._total_change_tco2e
        outputs["total_change_pct"] = self._total_change_pct
        outputs["common_sub_sectors"] = len(common_ids)
        outputs["base_only_sub_sectors"] = len(base_only)
        outputs["comparison_only_sub_sectors"] = len(comp_only)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 PeriodSelection: %s->%s change=%.2f tCO2e (%.2f%%)",
            base.period, comp.period,
            self._total_change_tco2e, self._total_change_pct,
        )
        return PhaseResult(
            phase_name="period_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: LMDI Decomposition
    # -------------------------------------------------------------------------

    async def _phase_2_lmdi_decomposition(
        self, input_data: DecompositionWorkflowInput,
    ) -> PhaseResult:
        """Execute LMDI decomposition for activity, structure, intensity effects."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base = input_data.base_period
        comp = input_data.comparison_period
        self._effects = []

        # Align sub-sectors (only common)
        base_map: Dict[str, SubSectorData] = {s.sub_sector_id: s for s in base.sub_sectors}
        comp_map: Dict[str, SubSectorData] = {s.sub_sector_id: s for s in comp.sub_sectors}
        common_ids = sorted(set(base_map.keys()) & set(comp_map.keys()))

        if not common_ids:
            # Use aggregate-level decomposition
            common_ids = ["aggregate"]
            base_map = {"aggregate": SubSectorData(
                sub_sector_id="aggregate",
                emissions_tco2e=base.total_emissions_tco2e,
                activity_value=base.total_activity,
                intensity=base.aggregate_intensity,
            )}
            comp_map = {"aggregate": SubSectorData(
                sub_sector_id="aggregate",
                emissions_tco2e=comp.total_emissions_tco2e,
                activity_value=comp.total_activity,
                intensity=comp.aggregate_intensity,
            )}
            warnings.append("No common sub-sectors; using aggregate decomposition")

        # Total activity for structure calculation
        base_total_activity = sum(base_map[sid].activity_value for sid in common_ids)
        comp_total_activity = sum(comp_map[sid].activity_value for sid in common_ids)

        # Aggregate effects
        total_activity_effect = 0.0
        total_structure_effect = 0.0
        total_intensity_effect = 0.0

        for sid in common_ids:
            b = base_map[sid]
            c = comp_map[sid]

            # Skip if either period has zero emissions
            if b.emissions_tco2e <= 0 or c.emissions_tco2e <= 0:
                warnings.append(f"Sub-sector {sid} has zero emissions in one period")
                continue

            # LMDI-I Additive decomposition (Ang 2005)
            # Weight: L(C_i^T, C_i^0) where C_i is emissions of sub-sector i
            w_i = _logarithmic_mean(c.emissions_tco2e, b.emissions_tco2e)

            if w_i <= 0:
                continue

            # Activity effect: change in total activity
            if base_total_activity > 0 and comp_total_activity > 0:
                act_effect = w_i * math.log(comp_total_activity / base_total_activity)
            else:
                act_effect = 0.0

            # Structure effect: change in activity share
            base_share = b.activity_value / max(base_total_activity, 1e-12)
            comp_share = c.activity_value / max(comp_total_activity, 1e-12)
            if base_share > 0 and comp_share > 0:
                str_effect = w_i * math.log(comp_share / base_share)
            else:
                str_effect = 0.0

            # Intensity effect: change in sub-sector intensity
            base_intensity = b.emissions_tco2e / max(b.activity_value, 1e-12)
            comp_intensity = c.emissions_tco2e / max(c.activity_value, 1e-12)
            if base_intensity > 0 and comp_intensity > 0:
                int_effect = w_i * math.log(comp_intensity / base_intensity)
            else:
                int_effect = 0.0

            total_activity_effect += act_effect
            total_structure_effect += str_effect
            total_intensity_effect += int_effect

            # Record sub-sector level effects
            for etype, val in [
                (EffectType.ACTIVITY, act_effect),
                (EffectType.STRUCTURE, str_effect),
                (EffectType.INTENSITY, int_effect),
            ]:
                effect_data = {
                    "type": etype.value, "sub_sector": sid,
                    "value": round(val, 6),
                }
                self._effects.append(DecompositionEffect(
                    effect_type=etype,
                    sub_sector_id=sid,
                    value_tco2e=round(val, 6),
                    value_pct=round(
                        (val / abs(self._total_change_tco2e) * 100.0)
                        if self._total_change_tco2e != 0 else 0.0, 4,
                    ),
                    direction="decrease" if val < 0 else ("increase" if val > 0 else "neutral"),
                    provenance_hash=_compute_hash(effect_data),
                ))

        # Record aggregate effects
        for etype, val in [
            (EffectType.ACTIVITY, total_activity_effect),
            (EffectType.STRUCTURE, total_structure_effect),
            (EffectType.INTENSITY, total_intensity_effect),
        ]:
            agg_data = {"type": etype.value, "sub_sector": "aggregate", "value": round(val, 6)}
            self._effects.append(DecompositionEffect(
                effect_type=etype,
                sub_sector_id="aggregate",
                value_tco2e=round(val, 6),
                value_pct=round(
                    (val / abs(self._total_change_tco2e) * 100.0)
                    if self._total_change_tco2e != 0 else 0.0, 4,
                ),
                direction="decrease" if val < 0 else ("increase" if val > 0 else "neutral"),
                provenance_hash=_compute_hash(agg_data),
            ))

        # Residual
        explained = total_activity_effect + total_structure_effect + total_intensity_effect
        self._residual = round(self._total_change_tco2e - explained, 6)

        if abs(self._residual) > 0.01:
            res_data = {"type": "residual", "value": self._residual}
            self._effects.append(DecompositionEffect(
                effect_type=EffectType.RESIDUAL,
                sub_sector_id="aggregate",
                value_tco2e=self._residual,
                value_pct=round(
                    (self._residual / abs(self._total_change_tco2e) * 100.0)
                    if self._total_change_tco2e != 0 else 0.0, 4,
                ),
                direction="decrease" if self._residual < 0 else "increase",
                provenance_hash=_compute_hash(res_data),
            ))

        outputs["method"] = input_data.method.value
        outputs["sub_sectors_decomposed"] = len(common_ids)
        outputs["total_activity_effect_tco2e"] = round(total_activity_effect, 6)
        outputs["total_structure_effect_tco2e"] = round(total_structure_effect, 6)
        outputs["total_intensity_effect_tco2e"] = round(total_intensity_effect, 6)
        outputs["residual_tco2e"] = self._residual
        outputs["effects_count"] = len(self._effects)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 LMDIDecomposition: activity=%.2f structure=%.2f intensity=%.2f residual=%.2f",
            total_activity_effect, total_structure_effect,
            total_intensity_effect, self._residual,
        )
        return PhaseResult(
            phase_name="lmdi_decomposition", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Effect Interpretation
    # -------------------------------------------------------------------------

    async def _phase_3_effect_interpretation(
        self, input_data: DecompositionWorkflowInput,
    ) -> PhaseResult:
        """Interpret decomposition effects and classify changes."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._interpretations = []
        threshold = input_data.significance_threshold_pct

        # Get aggregate effects
        agg_effects: Dict[EffectType, DecompositionEffect] = {}
        for effect in self._effects:
            if effect.sub_sector_id == "aggregate" and effect.effect_type != EffectType.RESIDUAL:
                agg_effects[effect.effect_type] = effect

        for etype in [EffectType.ACTIVITY, EffectType.STRUCTURE, EffectType.INTENSITY]:
            effect = agg_effects.get(etype)
            if not effect:
                continue

            magnitude = abs(effect.value_pct)
            is_favorable = effect.value_tco2e < 0  # Reduction is favorable

            # Classify
            if magnitude < threshold:
                classification = ChangeClassification.NEGLIGIBLE
                confidence = 95.0
            elif etype == EffectType.ACTIVITY:
                classification = ChangeClassification.GROWTH_EFFECT
                confidence = 85.0
            elif etype == EffectType.STRUCTURE:
                classification = ChangeClassification.STRUCTURAL_CHANGE
                confidence = 80.0
            elif etype == EffectType.INTENSITY:
                classification = ChangeClassification.ORGANIC_IMPROVEMENT if is_favorable else ChangeClassification.MIXED
                confidence = 75.0
            else:
                classification = ChangeClassification.MIXED
                confidence = 60.0

            # Generate narrative
            narrative = self._generate_narrative(etype, effect, classification)

            # Generate recommendations
            recommendations = self._generate_recommendations(etype, effect, classification)

            self._interpretations.append(EffectInterpretation(
                effect_type=etype,
                classification=classification,
                narrative=narrative,
                magnitude_pct=round(magnitude, 4),
                is_favorable=is_favorable,
                confidence=confidence,
                recommendations=recommendations,
            ))

        outputs["interpretations_count"] = len(self._interpretations)
        outputs["favorable_effects"] = sum(1 for i in self._interpretations if i.is_favorable)
        outputs["unfavorable_effects"] = sum(1 for i in self._interpretations if not i.is_favorable)
        outputs["classifications"] = [i.classification.value for i in self._interpretations]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 EffectInterpretation: %d interpretations",
            len(self._interpretations),
        )
        return PhaseResult(
            phase_name="effect_interpretation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Narrative Helpers
    # -------------------------------------------------------------------------

    def _generate_narrative(
        self, etype: EffectType, effect: DecompositionEffect,
        classification: ChangeClassification,
    ) -> str:
        """Generate deterministic narrative for an effect."""
        direction = "decreased" if effect.value_tco2e < 0 else "increased"
        abs_val = abs(effect.value_tco2e)
        abs_pct = abs(effect.value_pct)

        if etype == EffectType.ACTIVITY:
            return (
                f"Activity effect {direction} emissions by {abs_val:.1f} tCO2e "
                f"({abs_pct:.1f}% of total change). This reflects changes in "
                f"overall production or business volume."
            )
        elif etype == EffectType.STRUCTURE:
            return (
                f"Structure effect {direction} emissions by {abs_val:.1f} tCO2e "
                f"({abs_pct:.1f}% of total change). This reflects shifts in the "
                f"mix of sub-sectors or business units."
            )
        elif etype == EffectType.INTENSITY:
            return (
                f"Intensity effect {direction} emissions by {abs_val:.1f} tCO2e "
                f"({abs_pct:.1f}% of total change). This reflects changes in "
                f"emission intensity within sub-sectors."
            )
        return f"Effect {direction} emissions by {abs_val:.1f} tCO2e."

    def _generate_recommendations(
        self, etype: EffectType, effect: DecompositionEffect,
        classification: ChangeClassification,
    ) -> List[str]:
        """Generate deterministic recommendations for an effect."""
        recs: List[str] = []
        if classification == ChangeClassification.NEGLIGIBLE:
            recs.append("No action required; effect is within noise threshold.")
            return recs

        if etype == EffectType.ACTIVITY and effect.value_tco2e > 0:
            recs.append("Growth-driven emissions increase; consider decoupling strategies.")
            recs.append("Review intensity reduction targets to offset activity growth.")
        elif etype == EffectType.STRUCTURE:
            recs.append("Review structural shifts for permanence and strategic alignment.")
            recs.append("Update sector benchmarks if structural change is permanent.")
        elif etype == EffectType.INTENSITY and effect.value_tco2e < 0:
            recs.append("Document efficiency improvements for target progress reporting.")
            recs.append("Identify replicable practices across other sub-sectors.")
        elif etype == EffectType.INTENSITY and effect.value_tco2e > 0:
            recs.append("Investigate root cause of intensity increase.")
            recs.append("Prioritise high-impact sub-sectors for improvement programmes.")
        return recs

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: DecompositionWorkflowInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._effects = []
        self._interpretations = []
        self._total_change_tco2e = 0.0
        self._total_change_pct = 0.0
        self._residual = 0.0

    def _compute_provenance(self, result: DecompositionWorkflowResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.base_period}|{result.comparison_period}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
