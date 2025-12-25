# Risk Scorer Module
"""GL-013 PredictiveMaintenance - Risk-Based Scoring Module"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib, logging, math
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PlanningHorizon(str, Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class CriticalityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NON_CRITICAL = "non_critical"

class RiskCategory(str, Enum):
    EXTREME = "extreme"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class RiskScorerConfig(BaseModel):
    immediate_threshold_days: int = Field(default=7, ge=1, le=14)
    short_term_threshold_days: int = Field(default=30, ge=14, le=60)
    medium_term_threshold_days: int = Field(default=90, ge=60, le=180)
    extreme_risk_threshold: float = Field(default=80.0)
    high_risk_threshold: float = Field(default=60.0)
    medium_risk_threshold: float = Field(default=40.0)
    low_risk_threshold: float = Field(default=20.0)
    safety_weight: float = Field(default=0.50)
    production_weight: float = Field(default=0.35)
    cost_weight: float = Field(default=0.15)
    criticality_multipliers: Dict[str, float] = Field(default={"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.7, "non_critical": 0.4})

class AssetCriticality(BaseModel):
    asset_id: str
    asset_name: str
    criticality_level: CriticalityLevel
    safety_impact: float = Field(default=0.0, ge=0.0, le=100.0)
    production_dependency: float = Field(default=0.0, ge=0.0, le=100.0)
    redundancy_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    mttr_hours: float = Field(default=4.0, ge=0.0)
    asset_class: Optional[str] = None
    location: Optional[str] = None
    last_assessment_date: Optional[datetime] = None

class ConsequenceFactors(BaseModel):
    injury_potential: float = Field(default=0.0, ge=0.0, le=100.0)
    environmental_impact: float = Field(default=0.0, ge=0.0, le=100.0)
    regulatory_violation: float = Field(default=0.0, ge=0.0, le=100.0)
    downtime_hours: float = Field(default=0.0, ge=0.0)
    production_loss_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    quality_impact: float = Field(default=0.0, ge=0.0, le=100.0)
    cascade_potential: float = Field(default=0.0, ge=0.0, le=100.0)
    repair_cost_usd: float = Field(default=0.0, ge=0.0)
    spare_parts_cost_usd: float = Field(default=0.0, ge=0.0)
    production_loss_usd_per_hour: float = Field(default=0.0, ge=0.0)

class RULDistribution(BaseModel):
    asset_id: str
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rul_mean_days: float = Field(..., ge=0.0)
    rul_median_days: float = Field(..., ge=0.0)
    rul_std_days: float = Field(default=0.0, ge=0.0)
    rul_p10_days: float = Field(..., ge=0.0)
    rul_p50_days: float = Field(..., ge=0.0)
    rul_p90_days: float = Field(..., ge=0.0)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    model_version: Optional[str] = None
    failure_mode: Optional[str] = None

class RiskScore(BaseModel):
    asset_id: str
    score_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_category: RiskCategory
    planning_horizon: PlanningHorizon
    failure_probability: float = Field(..., ge=0.0, le=100.0)
    consequence_score: float = Field(..., ge=0.0, le=100.0)
    criticality_multiplier: float = Field(..., ge=0.0)
    safety_score: float = Field(default=0.0)
    production_score: float = Field(default=0.0)
    cost_score: float = Field(default=0.0)
    uncertainty_level: str = Field(default="medium")
    confidence_interval_lower: float = Field(default=0.0)
    confidence_interval_upper: float = Field(default=100.0)
    recommended_action: str = Field(default="monitor")
    days_until_action: int = Field(default=90, ge=0)
    provenance_hash: str
    processing_time_ms: float = Field(default=0.0, ge=0.0)

class RiskScorer:
    """Risk-based scoring for predictive maintenance using deterministic calculations."""

    def __init__(self, config: Optional[RiskScorerConfig] = None):
        self.config = config or RiskScorerConfig()
        self._score_count = 0

    def calculate_risk(self, rul: RULDistribution, crit: AssetCriticality, cons: ConsequenceFactors) -> RiskScore:
        """Calculate comprehensive risk score using Risk = P(failure) * Consequence * Criticality."""
        start = datetime.now(timezone.utc)
        if rul.asset_id != crit.asset_id:
            raise ValueError(f"Asset ID mismatch: {rul.asset_id} vs {crit.asset_id}")
        fp = self._calc_failure_prob(rul)
        cs, ss, ps, ks = self._calc_consequence(cons)
        cm = self.config.criticality_multipliers.get(crit.criticality_level.value, 1.0)
        rs = min(100.0, max(0.0, (fp / 100.0) * cs * cm))
        rc = self._categorize_risk(rs)
        ph = self._determine_horizon(rul.rul_p10_days)
        ul, cl, cu = self._calc_uncertainty(rul, cs, cm)
        ra = self._determine_action(rc, ul)
        da = self._calc_days_until_action(ph, rul.rul_p10_days)
        self._score_count += 1
        end = datetime.now(timezone.utc)
        prov = hashlib.sha256(f"{rul.asset_id}|{rs:.6f}".encode()).hexdigest()
        return RiskScore(asset_id=rul.asset_id, score_id=f"RISK-{rul.asset_id}-{self._score_count:06d}",
            timestamp=end, risk_score=rs, risk_category=rc, planning_horizon=ph, failure_probability=fp,
            consequence_score=cs, criticality_multiplier=cm, safety_score=ss, production_score=ps, cost_score=ks,
            uncertainty_level=ul, confidence_interval_lower=cl, confidence_interval_upper=cu,
            recommended_action=ra, days_until_action=da, provenance_hash=prov,
            processing_time_ms=(end - start).total_seconds() * 1000)

    def _calc_failure_prob(self, rul: RULDistribution) -> float:
        rul_days = max(0.1, rul.rul_p10_days)
        prob = 100.0 * math.exp(-rul_days / 30.0) * (1.0 + (1.0 - rul.confidence_score) * 0.2)
        return min(100.0, max(0.0, prob))

    def _calc_consequence(self, c: ConsequenceFactors) -> Tuple[float, float, float, float]:
        safety = c.injury_potential * 0.5 + c.environmental_impact * 0.3 + c.regulatory_violation * 0.2
        prod = min(100.0, c.downtime_hours / 24.0 * 100.0) * 0.3 + c.production_loss_rate * 0.3
        prod += c.quality_impact * 0.2 + c.cascade_potential * 0.2
        cost = min(100.0, (c.repair_cost_usd + c.spare_parts_cost_usd) / 1000.0)
        total = safety * self.config.safety_weight + prod * self.config.production_weight + cost * self.config.cost_weight
        return total, safety, prod, cost

    def _categorize_risk(self, score: float) -> RiskCategory:
        if score >= self.config.extreme_risk_threshold: return RiskCategory.EXTREME
        if score >= self.config.high_risk_threshold: return RiskCategory.HIGH
        if score >= self.config.medium_risk_threshold: return RiskCategory.MEDIUM
        if score >= self.config.low_risk_threshold: return RiskCategory.LOW
        return RiskCategory.NEGLIGIBLE

    def _determine_horizon(self, rul_p10: float) -> PlanningHorizon:
        if rul_p10 <= self.config.immediate_threshold_days: return PlanningHorizon.IMMEDIATE
        if rul_p10 <= self.config.short_term_threshold_days: return PlanningHorizon.SHORT_TERM
        if rul_p10 <= self.config.medium_term_threshold_days: return PlanningHorizon.MEDIUM_TERM
        return PlanningHorizon.LONG_TERM

    def _calc_uncertainty(self, rul: RULDistribution, cs: float, cm: float) -> Tuple[str, float, float]:
        cv = rul.rul_std_days / rul.rul_mean_days if rul.rul_mean_days > 0 else 1.0
        if cv < 0.2 and rul.confidence_score > 0.8: level = "low"
        elif cv > 0.5 or rul.confidence_score < 0.5: level = "high"
        else: level = "medium"
        ci_upper = min(100.0, math.exp(-rul.rul_p10_days / 30.0) * cs * cm)
        ci_lower = max(0.0, math.exp(-rul.rul_p90_days / 30.0) * cs * cm)
        return level, ci_lower, ci_upper

    def _determine_action(self, cat: RiskCategory, unc: str) -> str:
        if unc == "high":
            if cat in (RiskCategory.EXTREME, RiskCategory.HIGH): return "inspect_urgent"
            return "inspect_planned"
        action_map = {RiskCategory.EXTREME: "replace_immediate", RiskCategory.HIGH: "replace_scheduled",
            RiskCategory.MEDIUM: "condition_monitor", RiskCategory.LOW: "monitor", RiskCategory.NEGLIGIBLE: "accept"}
        return action_map.get(cat, "accept")

    def _calc_days_until_action(self, horizon: PlanningHorizon, rul_p10: float) -> int:
        days_map = {PlanningHorizon.IMMEDIATE: 3, PlanningHorizon.SHORT_TERM: 14,
            PlanningHorizon.MEDIUM_TERM: 45, PlanningHorizon.LONG_TERM: 90}
        return max(1, min(days_map.get(horizon, 90), int(rul_p10 * 0.7)))
