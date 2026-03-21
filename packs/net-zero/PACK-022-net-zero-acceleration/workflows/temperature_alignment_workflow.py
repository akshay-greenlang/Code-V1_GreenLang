# -*- coding: utf-8 -*-
"""
Temperature Alignment Workflow
====================================

4-phase workflow for calculating portfolio temperature alignment scores
within PACK-022 Net-Zero Acceleration Pack.  The workflow collects entity
targets, calculates temperature scores per target using the SBTi
temperature rating methodology, aggregates to portfolio level using
multiple weighting methods, and generates a temperature alignment report.

Phases:
    1. TargetCollection         -- Collect all entity targets
                                    (Scope 1+2, Scope 3, near-term, long-term)
    2. ScoreCalculation         -- Calculate temperature score per target
                                    using SBTi temperature mapping
    3. PortfolioAggregation     -- Aggregate to portfolio level
                                    (WATS, TETS, MOTS, EOTS)
    4. Reporting                -- Generate temperature alignment report
                                    with contribution analysis

Regulatory references:
    - SBTi Temperature Rating Methodology v2.0
    - SBTi Portfolio Coverage approach
    - TCFD Metrics and Targets recommendations
    - Paris Agreement Art. 2.1(a) - 1.5 deg C goal
    - IPCC AR6 - remaining carbon budgets

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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


class TargetScope(str, Enum):
    """Target scope coverage."""

    S1S2 = "scope_1_2"
    S3 = "scope_3"
    S1S2S3 = "scope_1_2_3"


class TargetTimeframe(str, Enum):
    """Target time horizon."""

    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"


class AggregationMethod(str, Enum):
    """Portfolio aggregation methods."""

    WATS = "wats"   # Weighted Average Temperature Score
    TETS = "tets"   # Total Emissions Weighted Temperature Score
    MOTS = "mots"   # Market Owned Temperature Score
    EOTS = "eots"   # Enterprise Owned Temperature Score


# =============================================================================
# TEMPERATURE RATING REFERENCE DATA (Zero-Hallucination, from SBTi)
# =============================================================================

# SBTi temperature score mapping based on annual reduction rate
# From SBTi Temperature Rating Methodology v2.0
TEMP_SCORE_MAP: Dict[str, Dict[str, float]] = {
    # Annual reduction rate ranges -> temperature score in deg C
    "scope_1_2": {
        "7.0+": 1.5,
        "4.2-7.0": 1.65,
        "2.5-4.2": 1.80,
        "1.0-2.5": 2.00,
        "0.0-1.0": 2.50,
        "no_target": 3.20,
    },
    "scope_3": {
        "7.0+": 1.5,
        "4.2-7.0": 1.75,
        "2.5-4.2": 2.00,
        "1.0-2.5": 2.20,
        "0.0-1.0": 2.70,
        "no_target": 3.20,
    },
}

# Default temperature score when no target exists
DEFAULT_TEMP_SCORE = 3.20  # deg C (current policies scenario)

# Paris-aligned threshold
PARIS_ALIGNED_THRESHOLD = 1.75  # deg C


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EntityTarget(BaseModel):
    """A single entity's emission reduction target."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    scope: TargetScope = Field(default=TargetScope.S1S2)
    timeframe: TargetTimeframe = Field(default=TargetTimeframe.NEAR_TERM)
    base_year: int = Field(default=2020, ge=2010, le=2050)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    is_sbti_validated: bool = Field(default=False)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)


class EntityWeight(BaseModel):
    """Entity weighting for portfolio aggregation."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    portfolio_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    market_cap_usd: float = Field(default=0.0, ge=0.0)
    enterprise_value_usd: float = Field(default=0.0, ge=0.0)
    revenue_usd: float = Field(default=0.0, ge=0.0)


class TemperatureScore(BaseModel):
    """Temperature score for a single entity target."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    scope: TargetScope = Field(default=TargetScope.S1S2)
    timeframe: TargetTimeframe = Field(default=TargetTimeframe.NEAR_TERM)
    annual_reduction_rate_pct: float = Field(default=0.0)
    temperature_score_c: float = Field(default=DEFAULT_TEMP_SCORE)
    is_paris_aligned: bool = Field(default=False)
    has_target: bool = Field(default=False)
    score_basis: str = Field(default="", description="Basis for the score assignment")


class PortfolioTemperature(BaseModel):
    """Portfolio-level temperature score for one aggregation method."""

    method: AggregationMethod = Field(default=AggregationMethod.WATS)
    method_description: str = Field(default="")
    temperature_score_c: float = Field(default=DEFAULT_TEMP_SCORE)
    is_paris_aligned: bool = Field(default=False)
    entity_count: int = Field(default=0)
    entities_with_targets_pct: float = Field(default=0.0)
    contribution_breakdown: Dict[str, float] = Field(default_factory=dict)


class ContributionAnalysis(BaseModel):
    """Entity contribution to portfolio temperature score."""

    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    temperature_score_c: float = Field(default=0.0)
    weight: float = Field(default=0.0)
    weighted_contribution_c: float = Field(default=0.0)
    contribution_pct: float = Field(default=0.0)
    paris_aligned: bool = Field(default=False)


class TemperatureAlignmentConfig(BaseModel):
    """Configuration for the temperature alignment workflow."""

    entities: List[EntityTarget] = Field(default_factory=list)
    entity_weights: List[EntityWeight] = Field(default_factory=list)
    aggregation_method: str = Field(default="wats", description="Primary aggregation method")
    default_score: float = Field(default=DEFAULT_TEMP_SCORE, description="Default temp score for no-target entities")
    include_scope3: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("aggregation_method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        allowed = {"wats", "tets", "mots", "eots"}
        if v not in allowed:
            raise ValueError(f"aggregation_method must be one of {allowed}")
        return v


class TemperatureAlignmentResult(BaseModel):
    """Complete result from the temperature alignment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="temperature_alignment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    entity_scores: List[TemperatureScore] = Field(default_factory=list)
    portfolio_scores: List[PortfolioTemperature] = Field(default_factory=list)
    primary_portfolio_score: Optional[PortfolioTemperature] = Field(None)
    contributions: List[ContributionAnalysis] = Field(default_factory=list)
    paris_aligned_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TemperatureAlignmentWorkflow:
    """
    4-phase temperature alignment workflow.

    Collects entity targets, calculates temperature scores using the SBTi
    methodology, aggregates to portfolio level using multiple weighting
    methods, and generates a temperature alignment report.

    Zero-hallucination: all temperature score mappings come from
    deterministic SBTi reference tables.  No LLM calls in the
    numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = TemperatureAlignmentWorkflow()
        >>> config = TemperatureAlignmentConfig(entities=[...])
        >>> result = await wf.execute(config)
        >>> assert result.primary_portfolio_score is not None
    """

    def __init__(self) -> None:
        """Initialise TemperatureAlignmentWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._entity_scores: List[TemperatureScore] = []
        self._portfolio_scores: List[PortfolioTemperature] = []
        self._primary: Optional[PortfolioTemperature] = None
        self._contributions: List[ContributionAnalysis] = []
        self._paris_pct: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: TemperatureAlignmentConfig) -> TemperatureAlignmentResult:
        """
        Execute the 4-phase temperature alignment workflow.

        Args:
            config: Temperature alignment configuration with entity targets,
                weights, and aggregation preferences.

        Returns:
            TemperatureAlignmentResult with entity scores, portfolio
            scores, and contribution analysis.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting temperature alignment workflow %s, entities=%d",
            self.workflow_id, len(config.entities),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_target_collection(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_score_calculation(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_portfolio_aggregation(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_reporting(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Temperature alignment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = TemperatureAlignmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            entity_scores=self._entity_scores,
            portfolio_scores=self._portfolio_scores,
            primary_portfolio_score=self._primary,
            contributions=self._contributions,
            paris_aligned_pct=round(self._paris_pct, 2),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Temperature alignment workflow %s completed in %.2fs",
            self.workflow_id, elapsed,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Target Collection
    # -------------------------------------------------------------------------

    async def _phase_target_collection(self, config: TemperatureAlignmentConfig) -> PhaseResult:
        """Collect and validate all entity targets."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        entities = config.entities
        if not entities:
            entities = self._generate_sample_entities()
            warnings.append(f"No entities provided; generated {len(entities)} sample entities")

        # Validate targets
        valid_entities: List[EntityTarget] = []
        for entity in entities:
            if entity.target_year <= entity.base_year:
                warnings.append(
                    f"Entity '{entity.entity_name}': target year ({entity.target_year}) "
                    f"<= base year ({entity.base_year}); skipping"
                )
                continue
            valid_entities.append(entity)

        # Generate weights if not provided
        if not config.entity_weights:
            self._generate_default_weights(valid_entities, config)

        # Count unique entities and target types
        unique_entities = set(e.entity_id for e in valid_entities)
        s1s2_targets = sum(1 for e in valid_entities if e.scope in (TargetScope.S1S2, TargetScope.S1S2S3))
        s3_targets = sum(1 for e in valid_entities if e.scope in (TargetScope.S3, TargetScope.S1S2S3))
        nt_targets = sum(1 for e in valid_entities if e.timeframe == TargetTimeframe.NEAR_TERM)
        lt_targets = sum(1 for e in valid_entities if e.timeframe == TargetTimeframe.LONG_TERM)

        outputs["total_targets"] = len(valid_entities)
        outputs["unique_entities"] = len(unique_entities)
        outputs["s1s2_targets"] = s1s2_targets
        outputs["s3_targets"] = s3_targets
        outputs["near_term_targets"] = nt_targets
        outputs["long_term_targets"] = lt_targets
        outputs["sbti_validated_count"] = sum(1 for e in valid_entities if e.is_sbti_validated)

        # Store validated entities back for later phases
        config.entities = valid_entities

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Target collection: %d targets from %d entities",
                         len(valid_entities), len(unique_entities))
        return PhaseResult(
            phase_name="target_collection",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_sample_entities(self) -> List[EntityTarget]:
        """Generate sample entity targets when none provided."""
        return [
            EntityTarget(
                entity_id="ENT-001", entity_name="Company Alpha",
                scope=TargetScope.S1S2, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2020, target_year=2030, reduction_pct=42.0,
                is_sbti_validated=True, base_year_emissions_tco2e=50000,
            ),
            EntityTarget(
                entity_id="ENT-001", entity_name="Company Alpha",
                scope=TargetScope.S3, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2020, target_year=2030, reduction_pct=25.0,
                is_sbti_validated=True, base_year_emissions_tco2e=150000,
            ),
            EntityTarget(
                entity_id="ENT-002", entity_name="Company Beta",
                scope=TargetScope.S1S2, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2021, target_year=2030, reduction_pct=30.0,
                is_sbti_validated=False, base_year_emissions_tco2e=80000,
            ),
            EntityTarget(
                entity_id="ENT-003", entity_name="Company Gamma",
                scope=TargetScope.S1S2, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2022, target_year=2032, reduction_pct=50.0,
                is_sbti_validated=True, base_year_emissions_tco2e=30000,
            ),
            EntityTarget(
                entity_id="ENT-004", entity_name="Company Delta",
                scope=TargetScope.S1S2, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2020, target_year=2030, reduction_pct=10.0,
                is_sbti_validated=False, base_year_emissions_tco2e=120000,
            ),
            EntityTarget(
                entity_id="ENT-005", entity_name="Company Epsilon",
                scope=TargetScope.S1S2, timeframe=TargetTimeframe.NEAR_TERM,
                base_year=2020, target_year=2030, reduction_pct=0.0,
                is_sbti_validated=False, base_year_emissions_tco2e=60000,
            ),
        ]

    def _generate_default_weights(
        self, entities: List[EntityTarget], config: TemperatureAlignmentConfig
    ) -> None:
        """Generate equal weights when no entity weights provided."""
        unique_ids = list(set(e.entity_id for e in entities))
        n = len(unique_ids)
        if n == 0:
            return
        equal_weight = 1.0 / n

        weights: List[EntityWeight] = []
        for eid in unique_ids:
            entity_targets = [e for e in entities if e.entity_id == eid]
            name = entity_targets[0].entity_name if entity_targets else eid
            total_emissions = sum(e.base_year_emissions_tco2e for e in entity_targets)
            weights.append(EntityWeight(
                entity_id=eid,
                entity_name=name,
                portfolio_weight=equal_weight,
                total_emissions_tco2e=total_emissions,
            ))
        config.entity_weights = weights

    # -------------------------------------------------------------------------
    # Phase 2: Score Calculation
    # -------------------------------------------------------------------------

    async def _phase_score_calculation(self, config: TemperatureAlignmentConfig) -> PhaseResult:
        """Calculate temperature score per target using SBTi mapping."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._entity_scores = []
        for target in config.entities:
            score = self._calculate_temperature_score(target, config)
            self._entity_scores.append(score)

        paris_count = sum(1 for s in self._entity_scores if s.is_paris_aligned)
        total_count = len(self._entity_scores)
        self._paris_pct = (paris_count / total_count * 100.0) if total_count > 0 else 0.0

        avg_score = (
            sum(s.temperature_score_c for s in self._entity_scores) / total_count
            if total_count > 0 else DEFAULT_TEMP_SCORE
        )

        outputs["scores_calculated"] = total_count
        outputs["paris_aligned_count"] = paris_count
        outputs["paris_aligned_pct"] = round(self._paris_pct, 2)
        outputs["average_temp_score_c"] = round(avg_score, 2)
        outputs["min_score_c"] = round(min((s.temperature_score_c for s in self._entity_scores), default=0.0), 2)
        outputs["max_score_c"] = round(max((s.temperature_score_c for s in self._entity_scores), default=0.0), 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Score calculation: %d scores, %.1f%% Paris-aligned",
                         total_count, self._paris_pct)
        return PhaseResult(
            phase_name="score_calculation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calculate_temperature_score(
        self, target: EntityTarget, config: TemperatureAlignmentConfig
    ) -> TemperatureScore:
        """Calculate temperature score for a single entity target."""
        has_target = target.reduction_pct > 0
        if not has_target:
            return TemperatureScore(
                entity_id=target.entity_id,
                entity_name=target.entity_name,
                scope=target.scope,
                timeframe=target.timeframe,
                annual_reduction_rate_pct=0.0,
                temperature_score_c=config.default_score,
                is_paris_aligned=False,
                has_target=False,
                score_basis="No target set; default score applied",
            )

        # Calculate annual reduction rate
        years = target.target_year - target.base_year
        if years <= 0:
            years = 1
        annual_rate = target.reduction_pct / years

        # Map annual rate to temperature score
        scope_key = "scope_3" if target.scope == TargetScope.S3 else "scope_1_2"
        temp_score = self._map_rate_to_temperature(annual_rate, scope_key)

        is_aligned = temp_score <= PARIS_ALIGNED_THRESHOLD

        basis = (
            f"Annual reduction rate: {annual_rate:.2f}%/yr "
            f"({target.reduction_pct:.1f}% over {years} years). "
            f"Mapped to {temp_score:.2f} deg C via SBTi methodology."
        )

        return TemperatureScore(
            entity_id=target.entity_id,
            entity_name=target.entity_name,
            scope=target.scope,
            timeframe=target.timeframe,
            annual_reduction_rate_pct=round(annual_rate, 2),
            temperature_score_c=round(temp_score, 2),
            is_paris_aligned=is_aligned,
            has_target=True,
            score_basis=basis,
        )

    def _map_rate_to_temperature(self, annual_rate: float, scope_key: str) -> float:
        """Map annual reduction rate to temperature score using SBTi table."""
        if annual_rate >= 7.0:
            return 1.50
        elif annual_rate >= 4.2:
            # Linear interpolation between 1.5 and next tier
            next_score = 1.65 if scope_key == "scope_1_2" else 1.75
            frac = (annual_rate - 4.2) / (7.0 - 4.2)
            return next_score - frac * (next_score - 1.50)
        elif annual_rate >= 2.5:
            low_score = 1.80 if scope_key == "scope_1_2" else 2.00
            high_score = 1.65 if scope_key == "scope_1_2" else 1.75
            frac = (annual_rate - 2.5) / (4.2 - 2.5)
            return low_score - frac * (low_score - high_score)
        elif annual_rate >= 1.0:
            low_score = 2.00 if scope_key == "scope_1_2" else 2.20
            high_score = 1.80 if scope_key == "scope_1_2" else 2.00
            frac = (annual_rate - 1.0) / (2.5 - 1.0)
            return low_score - frac * (low_score - high_score)
        elif annual_rate > 0:
            low_score = 2.50 if scope_key == "scope_1_2" else 2.70
            high_score = 2.00 if scope_key == "scope_1_2" else 2.20
            frac = annual_rate / 1.0
            return low_score - frac * (low_score - high_score)
        else:
            return DEFAULT_TEMP_SCORE

    # -------------------------------------------------------------------------
    # Phase 3: Portfolio Aggregation
    # -------------------------------------------------------------------------

    async def _phase_portfolio_aggregation(self, config: TemperatureAlignmentConfig) -> PhaseResult:
        """Aggregate entity scores to portfolio level using multiple methods."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._portfolio_scores = []
        weight_map = {w.entity_id: w for w in config.entity_weights}

        # Build entity-level combined scores (best of S1S2 and S3 targets)
        entity_combined = self._build_entity_combined_scores(config)

        # WATS: Weighted Average Temperature Score (equal or portfolio weight)
        wats = self._aggregate_wats(entity_combined, weight_map)
        self._portfolio_scores.append(wats)

        # TETS: Total Emissions Weighted Temperature Score
        tets = self._aggregate_tets(entity_combined, weight_map)
        self._portfolio_scores.append(tets)

        # MOTS: Market Owned Temperature Score (market cap weighted)
        mots = self._aggregate_mots(entity_combined, weight_map)
        self._portfolio_scores.append(mots)

        # EOTS: Enterprise Owned Temperature Score (enterprise value weighted)
        eots = self._aggregate_eots(entity_combined, weight_map)
        self._portfolio_scores.append(eots)

        # Set primary based on config
        method_map = {
            "wats": AggregationMethod.WATS,
            "tets": AggregationMethod.TETS,
            "mots": AggregationMethod.MOTS,
            "eots": AggregationMethod.EOTS,
        }
        primary_method = method_map.get(config.aggregation_method, AggregationMethod.WATS)
        self._primary = next(
            (p for p in self._portfolio_scores if p.method == primary_method),
            self._portfolio_scores[0] if self._portfolio_scores else None,
        )

        for ps in self._portfolio_scores:
            outputs[f"{ps.method.value}_score_c"] = round(ps.temperature_score_c, 2)
            outputs[f"{ps.method.value}_paris_aligned"] = ps.is_paris_aligned

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Portfolio aggregation: WATS=%.2f, TETS=%.2f",
                         wats.temperature_score_c, tets.temperature_score_c)
        return PhaseResult(
            phase_name="portfolio_aggregation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _build_entity_combined_scores(
        self, config: TemperatureAlignmentConfig
    ) -> Dict[str, float]:
        """Build per-entity combined temperature score (best target)."""
        entity_scores: Dict[str, List[float]] = {}
        for score in self._entity_scores:
            if score.entity_id not in entity_scores:
                entity_scores[score.entity_id] = []
            entity_scores[score.entity_id].append(score.temperature_score_c)

        # Use the best (lowest) score per entity
        combined: Dict[str, float] = {}
        for eid, scores in entity_scores.items():
            combined[eid] = min(scores) if scores else DEFAULT_TEMP_SCORE
        return combined

    def _aggregate_wats(
        self, entity_scores: Dict[str, float], weight_map: Dict[str, EntityWeight]
    ) -> PortfolioTemperature:
        """Weighted Average Temperature Score (portfolio weight)."""
        total_weight = 0.0
        weighted_sum = 0.0
        breakdown: Dict[str, float] = {}

        for eid, temp in entity_scores.items():
            w = weight_map.get(eid)
            weight = w.portfolio_weight if w else (1.0 / len(entity_scores) if entity_scores else 1.0)
            weighted_sum += temp * weight
            total_weight += weight
            name = w.entity_name if w else eid
            breakdown[name] = round(temp * weight, 4)

        score = weighted_sum / total_weight if total_weight > 0 else DEFAULT_TEMP_SCORE
        with_targets = sum(1 for t in entity_scores.values() if t < DEFAULT_TEMP_SCORE)
        total = len(entity_scores)

        return PortfolioTemperature(
            method=AggregationMethod.WATS,
            method_description="Weighted Average Temperature Score using portfolio weights",
            temperature_score_c=round(score, 2),
            is_paris_aligned=score <= PARIS_ALIGNED_THRESHOLD,
            entity_count=total,
            entities_with_targets_pct=round((with_targets / total * 100.0) if total > 0 else 0.0, 2),
            contribution_breakdown=breakdown,
        )

    def _aggregate_tets(
        self, entity_scores: Dict[str, float], weight_map: Dict[str, EntityWeight]
    ) -> PortfolioTemperature:
        """Total Emissions Weighted Temperature Score."""
        total_emissions = sum(
            (weight_map.get(eid).total_emissions_tco2e if weight_map.get(eid) else 0.0)
            for eid in entity_scores
        )
        weighted_sum = 0.0
        breakdown: Dict[str, float] = {}

        for eid, temp in entity_scores.items():
            w = weight_map.get(eid)
            emissions = w.total_emissions_tco2e if w else 0.0
            weight = emissions / total_emissions if total_emissions > 0 else 0.0
            weighted_sum += temp * weight
            name = w.entity_name if w else eid
            breakdown[name] = round(temp * weight, 4)

        score = weighted_sum if total_emissions > 0 else DEFAULT_TEMP_SCORE
        total = len(entity_scores)

        return PortfolioTemperature(
            method=AggregationMethod.TETS,
            method_description="Total Emissions Weighted Temperature Score",
            temperature_score_c=round(score, 2),
            is_paris_aligned=score <= PARIS_ALIGNED_THRESHOLD,
            entity_count=total,
            entities_with_targets_pct=round(
                (sum(1 for t in entity_scores.values() if t < DEFAULT_TEMP_SCORE) / total * 100.0)
                if total > 0 else 0.0, 2
            ),
            contribution_breakdown=breakdown,
        )

    def _aggregate_mots(
        self, entity_scores: Dict[str, float], weight_map: Dict[str, EntityWeight]
    ) -> PortfolioTemperature:
        """Market Owned Temperature Score (market cap weighted)."""
        total_mcap = sum(
            (weight_map.get(eid).market_cap_usd if weight_map.get(eid) else 0.0)
            for eid in entity_scores
        )
        weighted_sum = 0.0
        breakdown: Dict[str, float] = {}

        for eid, temp in entity_scores.items():
            w = weight_map.get(eid)
            mcap = w.market_cap_usd if w else 0.0
            weight = mcap / total_mcap if total_mcap > 0 else (1.0 / len(entity_scores) if entity_scores else 1.0)
            weighted_sum += temp * weight
            name = w.entity_name if w else eid
            breakdown[name] = round(temp * weight, 4)

        score = weighted_sum if total_mcap > 0 else DEFAULT_TEMP_SCORE
        # Fallback to equal-weighted if no market cap data
        if total_mcap <= 0 and entity_scores:
            score = sum(entity_scores.values()) / len(entity_scores)
        total = len(entity_scores)

        return PortfolioTemperature(
            method=AggregationMethod.MOTS,
            method_description="Market Owned Temperature Score (market cap weighted)",
            temperature_score_c=round(score, 2),
            is_paris_aligned=score <= PARIS_ALIGNED_THRESHOLD,
            entity_count=total,
            entities_with_targets_pct=round(
                (sum(1 for t in entity_scores.values() if t < DEFAULT_TEMP_SCORE) / total * 100.0)
                if total > 0 else 0.0, 2
            ),
            contribution_breakdown=breakdown,
        )

    def _aggregate_eots(
        self, entity_scores: Dict[str, float], weight_map: Dict[str, EntityWeight]
    ) -> PortfolioTemperature:
        """Enterprise Owned Temperature Score (enterprise value weighted)."""
        total_ev = sum(
            (weight_map.get(eid).enterprise_value_usd if weight_map.get(eid) else 0.0)
            for eid in entity_scores
        )
        weighted_sum = 0.0
        breakdown: Dict[str, float] = {}

        for eid, temp in entity_scores.items():
            w = weight_map.get(eid)
            ev = w.enterprise_value_usd if w else 0.0
            weight = ev / total_ev if total_ev > 0 else (1.0 / len(entity_scores) if entity_scores else 1.0)
            weighted_sum += temp * weight
            name = w.entity_name if w else eid
            breakdown[name] = round(temp * weight, 4)

        score = weighted_sum if total_ev > 0 else DEFAULT_TEMP_SCORE
        if total_ev <= 0 and entity_scores:
            score = sum(entity_scores.values()) / len(entity_scores)
        total = len(entity_scores)

        return PortfolioTemperature(
            method=AggregationMethod.EOTS,
            method_description="Enterprise Owned Temperature Score (enterprise value weighted)",
            temperature_score_c=round(score, 2),
            is_paris_aligned=score <= PARIS_ALIGNED_THRESHOLD,
            entity_count=total,
            entities_with_targets_pct=round(
                (sum(1 for t in entity_scores.values() if t < DEFAULT_TEMP_SCORE) / total * 100.0)
                if total > 0 else 0.0, 2
            ),
            contribution_breakdown=breakdown,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Reporting
    # -------------------------------------------------------------------------

    async def _phase_reporting(self, config: TemperatureAlignmentConfig) -> PhaseResult:
        """Generate temperature alignment report with contribution analysis."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build contribution analysis
        weight_map = {w.entity_id: w for w in config.entity_weights}
        entity_combined = self._build_entity_combined_scores(config)

        self._contributions = []
        total_weight = sum(
            (weight_map.get(eid).portfolio_weight if weight_map.get(eid) else 0.0)
            for eid in entity_combined
        )
        if total_weight <= 0:
            total_weight = 1.0

        portfolio_score = self._primary.temperature_score_c if self._primary else DEFAULT_TEMP_SCORE

        for eid, temp in entity_combined.items():
            w = weight_map.get(eid)
            weight = w.portfolio_weight if w else (1.0 / len(entity_combined) if entity_combined else 1.0)
            weighted_contrib = temp * weight
            contrib_pct = (weighted_contrib / portfolio_score * 100.0) if portfolio_score > 0 else 0.0

            self._contributions.append(ContributionAnalysis(
                entity_id=eid,
                entity_name=w.entity_name if w else eid,
                temperature_score_c=round(temp, 2),
                weight=round(weight, 4),
                weighted_contribution_c=round(weighted_contrib, 4),
                contribution_pct=round(contrib_pct, 2),
                paris_aligned=temp <= PARIS_ALIGNED_THRESHOLD,
            ))

        self._contributions.sort(key=lambda c: c.weighted_contribution_c, reverse=True)

        outputs["portfolio_score_c"] = round(portfolio_score, 2)
        outputs["paris_aligned"] = portfolio_score <= PARIS_ALIGNED_THRESHOLD
        outputs["paris_aligned_entities_pct"] = round(self._paris_pct, 2)
        outputs["contribution_count"] = len(self._contributions)
        outputs["highest_contributor"] = (
            self._contributions[0].entity_name if self._contributions else "none"
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Reporting: portfolio=%.2f deg C, Paris=%s",
                         portfolio_score, portfolio_score <= PARIS_ALIGNED_THRESHOLD)
        return PhaseResult(
            phase_name="reporting",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
