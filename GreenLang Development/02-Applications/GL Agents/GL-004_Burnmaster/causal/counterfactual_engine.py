"""
CounterfactualEngine - Counterfactual analysis for combustion systems.

This module implements counterfactual reasoning to answer "what if" questions
about combustion process interventions and their potential effects.

Example:
    >>> engine = CounterfactualEngine(causal_graph)
    >>> query = CounterfactualQuery(variable='O2', actual_value=3.5, counterfactual_value=4.0)
    >>> result = engine.compute_counterfactual(query)
    >>> print(result.effects)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from causal.causal_graph import CausalGraph, CausalEdge, NodeType

logger = logging.getLogger(__name__)


class InterventionType(str, Enum):
    DO = "do"
    OBSERVE = "observe"
    SOFT = "soft"


class Intervention(BaseModel):
    variable: str = Field(..., description="Variable to intervene on")
    value: float = Field(..., description="Intervention value")
    intervention_type: InterventionType = Field(default=InterventionType.DO)
    description: str = Field("", description="Human-readable description")

    class Config:
        use_enum_values = True


class Effect(BaseModel):
    variable: str = Field(..., description="Affected variable")
    baseline_value: float = Field(..., description="Value before intervention")
    counterfactual_value: float = Field(..., description="Value after intervention")
    effect_size: float = Field(..., description="Change in value")
    effect_direction: str = Field(..., description="increase, decrease, or unchanged")
    confidence: float = Field(..., ge=0.0, le=1.0)
    mechanism: str = Field("", description="Causal mechanism")


class CounterfactualQuery(BaseModel):
    variable: str = Field(..., description="Variable of interest")
    actual_value: float = Field(..., description="Actual observed value")
    counterfactual_value: float = Field(..., description="Hypothetical value")
    context: Dict[str, float] = Field(default_factory=dict, description="Current state")
    target_variables: List[str] = Field(default_factory=list, description="Variables to predict")


class CounterfactualResult(BaseModel):
    query: CounterfactualQuery = Field(..., description="Original query")
    effects: List[Effect] = Field(default_factory=list)
    total_effect: float = Field(..., description="Aggregate effect magnitude")
    explanation: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field("")

    class Config:
        use_enum_values = True


class ScenarioComparison(BaseModel):
    baseline: Dict[str, float] = Field(..., description="Baseline scenario values")
    alternative: Dict[str, float] = Field(..., description="Alternative scenario values")
    differences: Dict[str, float] = Field(default_factory=dict)
    improvements: List[str] = Field(default_factory=list)
    degradations: List[str] = Field(default_factory=list)
    net_benefit: float = Field(..., description="Overall benefit score")
    recommendation: str = Field(..., description="Recommended action")
    provenance_hash: str = Field("")


class ValidationResult(BaseModel):
    is_valid: bool = Field(...)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CounterfactualEngine:
    """Counterfactual reasoning engine for combustion systems."""

    def __init__(self, graph: CausalGraph, historical_data: Optional[pd.DataFrame] = None):
        self.graph = graph
        self._nx_graph = graph.to_networkx()
        self.historical_data = historical_data
        self._structural_equations: Dict[str, callable] = {}
        self._build_structural_equations()
        logger.info(f"CounterfactualEngine initialized with {len(graph.nodes)} nodes")

    def _build_structural_equations(self):
        for node_name, node in self.graph.nodes.items():
            parents = self.graph.get_parents(node_name)
            if not parents:
                self._structural_equations[node_name] = lambda x, n=node_name: x.get(n, 0.0)
            else:
                def make_eq(node, parent_list):
                    def equation(values):
                        parent_effects = 0.0
                        for parent in parent_list:
                            edge = next((e for e in self.graph.edges
                                        if e.source == parent and e.target == node), None)
                            if edge:
                                parent_effects += values.get(parent, 0.0) * edge.weight
                        return parent_effects
                    return equation
                self._structural_equations[node_name] = make_eq(node_name, parents)

    def compute_counterfactual(self, query: CounterfactualQuery) -> CounterfactualResult:
        start_time = datetime.now()
        logger.info(f"Computing counterfactual: {query.variable}")

        if query.variable not in self.graph.nodes:
            raise ValueError(f"Variable {query.variable} not in causal graph")

        descendants = self.graph.get_descendants(query.variable)
        target_vars = query.target_variables if query.target_variables else list(descendants)

        baseline_state = {**query.context}
        baseline_state[query.variable] = query.actual_value
        baseline_state = self._propagate_effects(baseline_state)

        cf_state = {**query.context}
        cf_state[query.variable] = query.counterfactual_value
        cf_state = self._propagate_effects(cf_state)

        effects = []
        for var in target_vars:
            if var in baseline_state and var in cf_state:
                baseline_val = baseline_state[var]
                cf_val = cf_state[var]
                effect_size = cf_val - baseline_val

                if effect_size > 0.01:
                    direction = "increase"
                elif effect_size < -0.01:
                    direction = "decrease"
                else:
                    direction = "unchanged"

                path = nx.shortest_path(self._nx_graph, query.variable, var) if nx.has_path(self._nx_graph, query.variable, var) else []
                mechanism = f"Effect propagates via: {' -> '.join(path)}" if path else "Direct effect"

                effects.append(Effect(
                    variable=var, baseline_value=baseline_val, counterfactual_value=cf_val,
                    effect_size=effect_size, effect_direction=direction,
                    confidence=0.8, mechanism=mechanism))

        total_effect = sum(abs(e.effect_size) for e in effects)
        explanation = self._generate_explanation(query, effects)
        confidence = 0.8 if effects else 0.5

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result_data = f"{query.variable}{query.counterfactual_value}{len(effects)}"
        provenance_hash = hashlib.sha256(result_data.encode()).hexdigest()

        result = CounterfactualResult(
            query=query, effects=effects, total_effect=total_effect,
            explanation=explanation, confidence=confidence,
            processing_time_ms=processing_time, provenance_hash=provenance_hash)

        logger.info(f"Counterfactual computed: {len(effects)} effects")
        return result

    def _propagate_effects(self, initial_state: Dict[str, float]) -> Dict[str, float]:
        state = dict(initial_state)
        G = self._nx_graph

        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            logger.warning("Graph has cycles, using node order")
            topo_order = list(G.nodes())

        for node in topo_order:
            if node in state:
                continue

            parents = self.graph.get_parents(node)
            if not parents:
                state[node] = 0.0
                continue

            value = 0.0
            for parent in parents:
                if parent in state:
                    edge = next((e for e in self.graph.edges
                                if e.source == parent and e.target == node), None)
                    if edge:
                        value += state[parent] * edge.weight
            state[node] = value

        return state

    def _generate_explanation(self, query: CounterfactualQuery, effects: List[Effect]) -> str:
        if not effects:
            return f"No downstream effects predicted from changing {query.variable}."

        change_dir = "increasing" if query.counterfactual_value > query.actual_value else "decreasing"
        change_amt = abs(query.counterfactual_value - query.actual_value)

        parts = [
            f"If we had set {query.variable} to {query.counterfactual_value} instead of {query.actual_value}",
            f"({change_dir} by {change_amt:.3f}), the following effects would occur:"
        ]

        for effect in sorted(effects, key=lambda e: abs(e.effect_size), reverse=True)[:5]:
            parts.append(f"- {effect.variable}: {effect.effect_direction} by {abs(effect.effect_size):.3f}")

        return " ".join(parts[:2]) + "\n" + "\n".join(parts[2:])

    def estimate_intervention_effect(self, intervention: Intervention) -> Effect:
        logger.info(f"Estimating effect of intervention: {intervention.variable}={intervention.value}")

        if self.historical_data is not None and intervention.variable in self.historical_data.columns:
            baseline_value = self.historical_data[intervention.variable].mean()
        else:
            node = self.graph.nodes.get(intervention.variable)
            if node and node.bounds:
                baseline_value = (node.bounds[0] + node.bounds[1]) / 2
            else:
                baseline_value = 0.0

        query = CounterfactualQuery(
            variable=intervention.variable,
            actual_value=baseline_value,
            counterfactual_value=intervention.value)

        result = self.compute_counterfactual(query)

        effect_size = sum(e.effect_size for e in result.effects)
        direction = "increase" if effect_size > 0 else "decrease" if effect_size < 0 else "unchanged"

        return Effect(
            variable="aggregate", baseline_value=0.0, counterfactual_value=effect_size,
            effect_size=effect_size, effect_direction=direction,
            confidence=result.confidence, mechanism=f"Intervention on {intervention.variable}")

    def compare_scenarios(self, baseline: Dict[str, float], alternative: Dict[str, float]) -> ScenarioComparison:
        logger.info("Comparing baseline and alternative scenarios")

        baseline_full = self._propagate_effects(baseline)
        alternative_full = self._propagate_effects(alternative)

        differences = {}
        improvements = []
        degradations = []

        all_vars = set(baseline_full.keys()) | set(alternative_full.keys())

        for var in all_vars:
            base_val = baseline_full.get(var, 0.0)
            alt_val = alternative_full.get(var, 0.0)
            diff = alt_val - base_val
            differences[var] = diff

            node = self.graph.nodes.get(var)
            if node:
                if node.node_type == NodeType.OUTPUT:
                    if var in ["efficiency", "stability"]:
                        if diff > 0.01:
                            improvements.append(f"{var}: +{diff:.3f}")
                        elif diff < -0.01:
                            degradations.append(f"{var}: {diff:.3f}")
                    elif var in ["CO", "NOx"]:
                        if diff < -0.01:
                            improvements.append(f"{var}: {diff:.3f} (reduced)")
                        elif diff > 0.01:
                            degradations.append(f"{var}: +{diff:.3f} (increased)")

        benefit_score = len(improvements) - len(degradations)
        net_benefit = benefit_score / max(len(all_vars), 1)

        if net_benefit > 0.2:
            recommendation = "Strongly recommend alternative scenario"
        elif net_benefit > 0:
            recommendation = "Alternative scenario shows marginal improvement"
        elif net_benefit == 0:
            recommendation = "Scenarios are equivalent"
        else:
            recommendation = "Baseline scenario is preferred"

        comparison_data = f"{baseline}{alternative}{net_benefit}"
        provenance_hash = hashlib.sha256(comparison_data.encode()).hexdigest()

        return ScenarioComparison(
            baseline=baseline_full, alternative=alternative_full,
            differences=differences, improvements=improvements,
            degradations=degradations, net_benefit=net_benefit,
            recommendation=recommendation, provenance_hash=provenance_hash)

    def validate_counterfactual(self, result: CounterfactualResult) -> ValidationResult:
        logger.info("Validating counterfactual result")
        errors = []
        warnings = []

        if result.query.variable not in self.graph.nodes:
            errors.append(f"Variable {result.query.variable} not in causal graph")

        node = self.graph.nodes.get(result.query.variable)
        if node and node.bounds:
            if result.query.counterfactual_value < node.bounds[0]:
                warnings.append(f"Counterfactual value below minimum bound ({node.bounds[0]})")
            if result.query.counterfactual_value > node.bounds[1]:
                warnings.append(f"Counterfactual value above maximum bound ({node.bounds[1]})")

        for effect in result.effects:
            if effect.variable not in self.graph.nodes:
                errors.append(f"Effect variable {effect.variable} not in graph")
            if effect.confidence < 0.3:
                warnings.append(f"Low confidence effect on {effect.variable}")

        for effect in result.effects:
            if not nx.has_path(self._nx_graph, result.query.variable, effect.variable):
                if effect.variable != result.query.variable:
                    errors.append(f"No causal path from {result.query.variable} to {effect.variable}")

        is_valid = len(errors) == 0

        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
