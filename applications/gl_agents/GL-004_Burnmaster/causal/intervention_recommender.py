"""
InterventionRecommender - Recommends and validates interventions for combustion optimization.

This module provides recommendations for interventions to address combustion issues,
estimates their impact, validates safety constraints, and tracks outcomes.

Example:
    >>> recommender = InterventionRecommender(causal_graph, counterfactual_engine)
    >>> recommendation = recommender.recommend_intervention('high_CO', current_state)
    >>> print(recommendation.proposed_intervention)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from causal.causal_graph import CausalGraph, NodeType
from causal.counterfactual_engine import CounterfactualEngine, Intervention, Effect

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class ImpactEstimate(BaseModel):
    intervention: Intervention = Field(...)
    primary_effect: Effect = Field(...)
    secondary_effects: List[Effect] = Field(default_factory=list)
    estimated_improvement: float = Field(..., description="Percentage improvement expected")
    time_to_effect: str = Field(..., description="Estimated time for effect")
    confidence: float = Field(..., ge=0.0, le=1.0)
    risks: List[str] = Field(default_factory=list)
    provenance_hash: str = Field("")


class RankedIntervention(BaseModel):
    rank: int = Field(...)
    intervention: Intervention = Field(...)
    expected_benefit: float = Field(...)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    feasibility: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(...)
    rationale: str = Field(...)


class SafetyValidation(BaseModel):
    intervention: Intervention = Field(...)
    is_safe: bool = Field(...)
    safety_level: SafetyLevel = Field(...)
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommended_limits: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class InterventionRecommendation(BaseModel):
    problem: str = Field(..., description="Problem being addressed")
    current_state: Dict[str, float] = Field(...)
    proposed_intervention: Intervention = Field(...)
    impact_estimate: ImpactEstimate = Field(...)
    safety_validation: SafetyValidation = Field(...)
    alternatives: List[RankedIntervention] = Field(default_factory=list)
    explanation: str = Field(...)
    recommendation_id: str = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("")

    class Config:
        use_enum_values = True


class InterventionOutcome(BaseModel):
    intervention_id: str = Field(...)
    intervention: Intervention = Field(...)
    predicted_effect: float = Field(...)
    actual_effect: float = Field(...)
    success: bool = Field(...)
    deviation: float = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: str = Field("")


# Type alias for tuple since Pydantic needs it
from typing import Tuple


class InterventionRecommender:
    # Problem to intervention mappings based on combustion physics
    PROBLEM_INTERVENTIONS = {
        "high_CO": [
            {"variable": "air_flow", "direction": "increase", "magnitude": 0.1, "rationale": "Increase excess air to promote CO burnout"},
            {"variable": "O2", "direction": "increase", "magnitude": 0.5, "rationale": "Higher O2 promotes complete combustion"},
        ],
        "low_efficiency": [
            {"variable": "O2", "direction": "decrease", "magnitude": 0.3, "rationale": "Reduce excess air losses"},
            {"variable": "fuel_flow", "direction": "optimize", "magnitude": 0.05, "rationale": "Optimize air-fuel ratio"},
        ],
        "high_NOx": [
            {"variable": "flame_temp", "direction": "decrease", "magnitude": 50, "rationale": "Lower peak flame temperature"},
            {"variable": "air_flow", "direction": "stage", "magnitude": 0.1, "rationale": "Stage air injection"},
        ],
        "instability": [
            {"variable": "fuel_flow", "direction": "increase", "magnitude": 0.02, "rationale": "Increase fuel for flame anchoring"},
            {"variable": "air_flow", "direction": "decrease", "magnitude": 0.05, "rationale": "Reduce air velocity"},
        ],
    }

    SAFETY_LIMITS = {
        "fuel_flow": (0.5, 50.0),
        "air_flow": (5.0, 500.0),
        "O2": (1.0, 10.0),
        "flame_temp": (800.0, 2200.0),
        "CO": (0.0, 500.0),
        "NOx": (0.0, 200.0),
    }

    def __init__(self, graph: CausalGraph, cf_engine: Optional[CounterfactualEngine] = None):
        self.graph = graph
        self._nx_graph = graph.to_networkx()
        self.cf_engine = cf_engine or CounterfactualEngine(graph)
        self._outcome_history: Dict[str, InterventionOutcome] = {}
        logger.info("InterventionRecommender initialized")

    def recommend_intervention(self, problem: str, current_state: Dict[str, float]) -> InterventionRecommendation:
        logger.info(f"Recommending intervention for problem: {problem}")
        recommendation_id = str(uuid.uuid4())
        
        # Get candidate interventions
        candidates = self.PROBLEM_INTERVENTIONS.get(problem.lower(), [])
        if not candidates:
            candidates = self._infer_interventions(problem, current_state)
        
        # Rank interventions
        ranked = self.rank_interventions([
            self._create_intervention(c, current_state) for c in candidates
        ])
        
        if not ranked:
            raise ValueError(f"No valid interventions found for problem: {problem}")
        
        best = ranked[0]
        impact = self.estimate_intervention_impact(best.intervention)
        safety = self.validate_intervention_safety(best.intervention)
        
        explanation = self._generate_explanation(problem, best, impact, safety)
        
        rec_data = f"{problem}{best.intervention.variable}{best.intervention.value}"
        provenance_hash = hashlib.sha256(rec_data.encode()).hexdigest()
        
        recommendation = InterventionRecommendation(
            problem=problem, current_state=current_state,
            proposed_intervention=best.intervention,
            impact_estimate=impact, safety_validation=safety,
            alternatives=ranked[1:4], explanation=explanation,
            recommendation_id=recommendation_id, provenance_hash=provenance_hash)
        
        logger.info(f"Recommendation generated: {best.intervention.variable}={best.intervention.value}")
        return recommendation

    def _create_intervention(self, candidate: Dict, current_state: Dict[str, float]) -> Intervention:
        var = candidate["variable"]
        direction = candidate["direction"]
        magnitude = candidate["magnitude"]
        
        current_val = current_state.get(var, 0.0)
        if current_val == 0.0:
            node = self.graph.nodes.get(var)
            if node and node.bounds:
                current_val = (node.bounds[0] + node.bounds[1]) / 2
        
        if direction == "increase":
            new_val = current_val * (1 + magnitude)
        elif direction == "decrease":
            new_val = current_val * (1 - magnitude)
        else:
            new_val = current_val * (1 + magnitude * 0.5)
        
        return Intervention(variable=var, value=new_val, description=candidate.get("rationale", ""))

    def _infer_interventions(self, problem: str, current_state: Dict[str, float]) -> List[Dict]:
        inferred = []
        input_nodes = [n for n, node in self.graph.nodes.items() if node.node_type == NodeType.INPUT]
        
        for input_node in input_nodes:
            inferred.append({"variable": input_node, "direction": "optimize",
                            "magnitude": 0.05, "rationale": f"Optimize {input_node} for {problem}"})
        return inferred

    def estimate_intervention_impact(self, intervention: Intervention) -> ImpactEstimate:
        logger.info(f"Estimating impact of {intervention.variable}={intervention.value}")
        
        primary_effect = self.cf_engine.estimate_intervention_effect(intervention)
        
        # Get secondary effects on output nodes
        descendants = self.graph.get_descendants(intervention.variable)
        secondary_effects = []
        
        for desc in descendants:
            node = self.graph.nodes.get(desc)
            if node and node.node_type == NodeType.OUTPUT:
                effect = Effect(variable=desc, baseline_value=0.0, counterfactual_value=0.0,
                               effect_size=primary_effect.effect_size * 0.5,
                               effect_direction=primary_effect.effect_direction,
                               confidence=0.7, mechanism=f"Downstream effect from {intervention.variable}")
                secondary_effects.append(effect)
        
        estimated_improvement = abs(primary_effect.effect_size) * 10  # Percentage
        
        risks = []
        safety = self.validate_intervention_safety(intervention)
        if not safety.is_safe:
            risks.extend(safety.violations)
        risks.extend(safety.warnings)
        
        impact_data = f"{intervention.variable}{intervention.value}{estimated_improvement}"
        provenance_hash = hashlib.sha256(impact_data.encode()).hexdigest()
        
        return ImpactEstimate(
            intervention=intervention, primary_effect=primary_effect,
            secondary_effects=secondary_effects, estimated_improvement=estimated_improvement,
            time_to_effect="1-5 minutes", confidence=primary_effect.confidence,
            risks=risks, provenance_hash=provenance_hash)

    def rank_interventions(self, interventions: List[Intervention]) -> List[RankedIntervention]:
        logger.info(f"Ranking {len(interventions)} interventions")
        scored = []
        
        for intervention in interventions:
            impact = self.estimate_intervention_impact(intervention)
            safety = self.validate_intervention_safety(intervention)
            
            benefit = impact.estimated_improvement / 100.0
            risk = 0.0 if safety.is_safe else 0.5
            if safety.safety_level == SafetyLevel.WARNING:
                risk = 0.7
            elif safety.safety_level == SafetyLevel.CRITICAL:
                risk = 1.0
            
            feasibility = 0.9 if intervention.variable in ["air_flow", "fuel_flow"] else 0.7
            overall = (benefit * 0.5 + (1 - risk) * 0.3 + feasibility * 0.2)
            
            scored.append({"intervention": intervention, "benefit": benefit,
                          "risk": risk, "feasibility": feasibility, "overall": overall})
        
        scored.sort(key=lambda x: x["overall"], reverse=True)
        
        ranked = []
        for i, s in enumerate(scored, 1):
            ranked.append(RankedIntervention(
                rank=i, intervention=s["intervention"], expected_benefit=s["benefit"],
                risk_score=s["risk"], feasibility=s["feasibility"],
                overall_score=s["overall"],
                rationale=f"Benefit: {s['benefit']:.2f}, Risk: {s['risk']:.2f}, Feasibility: {s['feasibility']:.2f}"))
        
        return ranked

    def validate_intervention_safety(self, intervention: Intervention) -> SafetyValidation:
        logger.info(f"Validating safety of {intervention.variable}={intervention.value}")
        violations = []
        warnings = []
        
        limits = self.SAFETY_LIMITS.get(intervention.variable)
        if limits:
            if intervention.value < limits[0]:
                violations.append(f"{intervention.variable} below minimum safe limit ({limits[0]})")
            elif intervention.value > limits[1]:
                violations.append(f"{intervention.variable} above maximum safe limit ({limits[1]})")
            elif intervention.value < limits[0] * 1.1:
                warnings.append(f"{intervention.variable} approaching minimum limit")
            elif intervention.value > limits[1] * 0.9:
                warnings.append(f"{intervention.variable} approaching maximum limit")
        
        # Check downstream safety impacts
        descendants = self.graph.get_descendants(intervention.variable)
        for desc in descendants:
            desc_limits = self.SAFETY_LIMITS.get(desc)
            if desc_limits:
                warnings.append(f"Monitor {desc} for safety limits during intervention")
        
        # Determine safety level
        if violations:
            safety_level = SafetyLevel.CRITICAL
        elif len(warnings) > 2:
            safety_level = SafetyLevel.WARNING
        elif warnings:
            safety_level = SafetyLevel.CAUTION
        else:
            safety_level = SafetyLevel.SAFE
        
        is_safe = len(violations) == 0
        
        return SafetyValidation(
            intervention=intervention, is_safe=is_safe, safety_level=safety_level,
            violations=violations, warnings=warnings,
            recommended_limits={intervention.variable: limits} if limits else {})

    def track_intervention_outcome(self, intervention_id: str, outcome: Dict[str, Any]):
        logger.info(f"Tracking outcome for intervention {intervention_id}")
        
        if intervention_id not in self._outcome_history:
            logger.warning(f"Intervention {intervention_id} not found in history")
        
        predicted = outcome.get("predicted_effect", 0.0)
        actual = outcome.get("actual_effect", 0.0)
        deviation = abs(actual - predicted) / max(abs(predicted), 0.001)
        success = deviation < 0.3 and actual * predicted > 0
        
        intervention_outcome = InterventionOutcome(
            intervention_id=intervention_id,
            intervention=outcome.get("intervention", Intervention(variable="unknown", value=0.0)),
            predicted_effect=predicted, actual_effect=actual,
            success=success, deviation=deviation,
            notes=outcome.get("notes", ""))
        
        self._outcome_history[intervention_id] = intervention_outcome
        logger.info(f"Outcome tracked: success={success}, deviation={deviation:.2%}")

    def _generate_explanation(self, problem: str, best: RankedIntervention,
                              impact: ImpactEstimate, safety: SafetyValidation) -> str:
        explanation = f"To address {problem}, recommend adjusting {best.intervention.variable} "
        explanation += f"to {best.intervention.value:.3f}. "
        explanation += f"Expected improvement: {impact.estimated_improvement:.1f}%. "
        explanation += f"Time to effect: {impact.time_to_effect}. "
        
        if safety.is_safe:
            explanation += "Intervention is within safe operating limits. "
        else:
            explanation += f"CAUTION: {', '.join(safety.violations)}. "
        
        if safety.warnings:
            explanation += f"Warnings: {', '.join(safety.warnings[:2])}. "
        
        explanation += f"Overall confidence: {impact.confidence:.0%}."
        return explanation
