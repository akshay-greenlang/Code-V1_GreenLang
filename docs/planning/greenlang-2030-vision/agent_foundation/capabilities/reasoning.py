# -*- coding: utf-8 -*-
"""
Reasoning Framework - Comprehensive reasoning engines for GreenLang agents.

This module implements various reasoning capabilities including:
- Deductive reasoning with logic-based inference
- Inductive reasoning with pattern-based learning
- Abductive reasoning for best explanation finding
- Analogical reasoning with similarity-based inference

Example:
    >>> reasoner = DeductiveReasoner(config)
    >>> result = await reasoner.infer(premises, query)
    >>> confidence = result.confidence_score
"""

import ast
import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
import numpy as np
from scipy import stats
import networkx as nx
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ReasoningType(str, Enum):
    """Types of reasoning methods."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"


class InferenceStatus(str, Enum):
    """Status of inference process."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class Proposition:
    """Logical proposition for reasoning."""

    proposition_id: str
    content: str
    truth_value: Optional[bool] = None
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """Logical rule for inference."""

    rule_id: str
    name: str
    antecedents: List[str]  # If conditions
    consequent: str         # Then conclusion
    confidence: float = 1.0
    rule_type: str = "implication"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Pattern for inductive reasoning."""

    pattern_id: str
    pattern_type: str
    instances: List[Dict[str, Any]]
    frequency: int
    confidence: float
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class Hypothesis:
    """Hypothesis for abductive reasoning."""

    hypothesis_id: str
    explanation: str
    evidence_for: List[str]
    evidence_against: List[str]
    likelihood: float
    simplicity_score: float
    explanatory_power: float


@dataclass
class Analogy:
    """Analogy for analogical reasoning."""

    source_domain: str
    target_domain: str
    mappings: Dict[str, str]
    similarity_score: float
    structural_alignment: float


class InferenceResult(BaseModel):
    """Result from reasoning inference."""

    reasoning_type: ReasoningType
    status: InferenceStatus
    conclusion: Optional[Any] = None
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    reasoning_path: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningEngine(ABC):
    """Abstract base class for reasoning engines."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reasoning engine."""
        self.config = config or {}
        self.knowledge_base = []
        self.inference_cache = {}
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

    @abstractmethod
    async def infer(
        self,
        premises: List[Any],
        query: Any,
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform inference based on premises and query."""
        pass

    @abstractmethod
    def add_knowledge(self, knowledge: Any) -> None:
        """Add knowledge to the reasoning engine."""
        pass

    @abstractmethod
    def validate_inference(self, result: InferenceResult) -> bool:
        """Validate inference result."""
        pass


class DeductiveReasoner(ReasoningEngine):
    """Deductive reasoning with logic-based inference."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize deductive reasoner."""
        super().__init__(config)
        self.rules: List[Rule] = []
        self.facts: Dict[str, Proposition] = {}
        self.inference_method = config.get("inference_method", "forward_chaining")

    async def infer(
        self,
        premises: List[Proposition],
        query: str,
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform deductive inference."""
        start_time = time.time()
        reasoning_path = []

        try:
            # Add premises to facts
            for premise in premises:
                self.facts[premise.content] = premise

            # Perform inference based on method
            if self.inference_method == "forward_chaining":
                conclusion = await self._forward_chaining(query, reasoning_path)
            elif self.inference_method == "backward_chaining":
                conclusion = await self._backward_chaining(query, reasoning_path)
            else:
                conclusion = await self._resolution(query, reasoning_path)

            # Calculate confidence
            confidence = self._calculate_confidence(conclusion, reasoning_path)

            # Create result
            execution_time = (time.time() - start_time) * 1000

            return InferenceResult(
                reasoning_type=ReasoningType.DEDUCTIVE,
                status=InferenceStatus.COMPLETED if conclusion else InferenceStatus.UNCERTAIN,
                conclusion=conclusion,
                confidence_score=confidence,
                evidence=[p.content for p in premises],
                reasoning_path=reasoning_path,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Deductive inference failed: {str(e)}")
            return InferenceResult(
                reasoning_type=ReasoningType.DEDUCTIVE,
                status=InferenceStatus.FAILED,
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def _forward_chaining(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]]
    ) -> Optional[bool]:
        """Forward chaining inference."""
        inferred = set()
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                # Check if all antecedents are satisfied
                if all(ant in self.facts or ant in inferred for ant in rule.antecedents):
                    if rule.consequent not in inferred:
                        inferred.add(rule.consequent)
                        changed = True

                        # Record reasoning step
                        reasoning_path.append({
                            "rule": rule.name,
                            "antecedents": rule.antecedents,
                            "consequent": rule.consequent,
                            "confidence": rule.confidence
                        })

                        # Check if we found the query
                        if rule.consequent == query:
                            return True

        return query in inferred

    async def _backward_chaining(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]]
    ) -> Optional[bool]:
        """Backward chaining inference."""
        return await self._prove_goal(query, set(), reasoning_path)

    async def _prove_goal(
        self,
        goal: str,
        visited: Set[str],
        reasoning_path: List[Dict[str, Any]]
    ) -> bool:
        """Recursively prove a goal."""
        # Check if goal is already known
        if goal in self.facts:
            return self.facts[goal].truth_value

        # Avoid infinite recursion
        if goal in visited:
            return False

        visited.add(goal)

        # Find rules that can prove this goal
        for rule in self.rules:
            if rule.consequent == goal:
                # Try to prove all antecedents
                all_proven = True
                for antecedent in rule.antecedents:
                    if not await self._prove_goal(antecedent, visited, reasoning_path):
                        all_proven = False
                        break

                if all_proven:
                    reasoning_path.append({
                        "rule": rule.name,
                        "goal": goal,
                        "antecedents_proven": rule.antecedents,
                        "method": "backward_chaining"
                    })
                    return True

        return False

    async def _resolution(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]]
    ) -> Optional[bool]:
        """Resolution-based inference."""
        # Convert to CNF and apply resolution
        # Simplified implementation
        clauses = self._convert_to_cnf(self.facts, self.rules)
        negated_query = self._negate(query)
        clauses.append(negated_query)

        new = set()
        while True:
            pairs = [(clauses[i], clauses[j])
                    for i in range(len(clauses))
                    for j in range(i + 1, len(clauses))]

            for (ci, cj) in pairs:
                resolvents = self._resolve(ci, cj)
                if [] in resolvents:
                    # Empty clause derived, query is true
                    reasoning_path.append({
                        "method": "resolution",
                        "result": "empty_clause",
                        "conclusion": query
                    })
                    return True
                new = new.union(set(resolvents))

            if new.issubset(set(clauses)):
                # No new clauses, cannot prove query
                return False

            clauses.extend(new)

    def _convert_to_cnf(
        self,
        facts: Dict[str, Proposition],
        rules: List[Rule]
    ) -> List[List[str]]:
        """Convert facts and rules to CNF."""
        clauses = []

        # Convert facts
        for fact_content, fact in facts.items():
            if fact.truth_value:
                clauses.append([fact_content])
            else:
                clauses.append([f"not_{fact_content}"])

        # Convert rules (simplified)
        for rule in rules:
            # Rule: A1 ∧ A2 → C becomes ¬A1 ∨ ¬A2 ∨ C
            clause = [f"not_{ant}" for ant in rule.antecedents]
            clause.append(rule.consequent)
            clauses.append(clause)

        return clauses

    def _negate(self, proposition: str) -> List[str]:
        """Negate a proposition."""
        if proposition.startswith("not_"):
            return [proposition[4:]]
        return [f"not_{proposition}"]

    def _resolve(self, ci: List[str], cj: List[str]) -> List[List[str]]:
        """Resolve two clauses."""
        resolvents = []

        for literal_i in ci:
            for literal_j in cj:
                if self._are_complementary(literal_i, literal_j):
                    # Create resolvent
                    new_clause = [l for l in ci if l != literal_i]
                    new_clause.extend([l for l in cj if l != literal_j])
                    if new_clause not in resolvents:
                        resolvents.append(new_clause)

        return resolvents

    def _are_complementary(self, lit1: str, lit2: str) -> bool:
        """Check if two literals are complementary."""
        return ((lit1.startswith("not_") and lit2 == lit1[4:]) or
                (lit2.startswith("not_") and lit1 == lit2[4:]))

    def _calculate_confidence(
        self,
        conclusion: Optional[bool],
        reasoning_path: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in conclusion."""
        if conclusion is None:
            return 0.0

        if not reasoning_path:
            return 0.5

        # Calculate based on rules used
        confidences = []
        for step in reasoning_path:
            if "confidence" in step:
                confidences.append(step["confidence"])

        if confidences:
            # Combine confidences (simplified)
            return np.mean(confidences)

        return 0.8  # Default confidence for successful inference

    def add_knowledge(self, knowledge: Union[Rule, Proposition]) -> None:
        """Add knowledge to deductive reasoner."""
        if isinstance(knowledge, Rule):
            self.rules.append(knowledge)
        elif isinstance(knowledge, Proposition):
            self.facts[knowledge.content] = knowledge

    def validate_inference(self, result: InferenceResult) -> bool:
        """Validate deductive inference result."""
        # Check logical consistency
        if result.conclusion is not None:
            # Verify reasoning path
            for step in result.reasoning_path:
                if "rule" in step:
                    # Verify rule exists
                    rule_exists = any(r.name == step["rule"] for r in self.rules)
                    if not rule_exists:
                        return False

        return result.confidence_score >= self.confidence_threshold


class InductiveReasoner(ReasoningEngine):
    """Inductive reasoning with pattern-based learning."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize inductive reasoner."""
        super().__init__(config)
        self.patterns: List[Pattern] = []
        self.observations: List[Dict[str, Any]] = []
        self.min_support = config.get("min_support", 3)
        self.pattern_detector = PatternDetector()

    async def infer(
        self,
        premises: List[Dict[str, Any]],
        query: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform inductive inference."""
        start_time = time.time()

        try:
            # Add observations
            self.observations.extend(premises)

            # Detect patterns
            detected_patterns = await self._detect_patterns(self.observations)

            # Find applicable patterns for query
            applicable_patterns = self._find_applicable_patterns(
                query,
                detected_patterns
            )

            # Make prediction based on patterns
            prediction = await self._make_prediction(
                query,
                applicable_patterns
            )

            # Calculate confidence based on pattern support
            confidence = self._calculate_pattern_confidence(applicable_patterns)

            execution_time = (time.time() - start_time) * 1000

            return InferenceResult(
                reasoning_type=ReasoningType.INDUCTIVE,
                status=InferenceStatus.COMPLETED if prediction else InferenceStatus.UNCERTAIN,
                conclusion=prediction,
                confidence_score=confidence,
                evidence=[str(p) for p in applicable_patterns],
                reasoning_path=[{
                    "method": "pattern_matching",
                    "patterns_found": len(applicable_patterns),
                    "total_observations": len(self.observations)
                }],
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Inductive inference failed: {str(e)}")
            return InferenceResult(
                reasoning_type=ReasoningType.INDUCTIVE,
                status=InferenceStatus.FAILED,
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def _detect_patterns(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Detect patterns in observations."""
        patterns = []

        # Frequency-based pattern detection
        frequency_patterns = self._detect_frequency_patterns(observations)
        patterns.extend(frequency_patterns)

        # Sequence pattern detection
        sequence_patterns = self._detect_sequence_patterns(observations)
        patterns.extend(sequence_patterns)

        # Statistical pattern detection
        statistical_patterns = await self._detect_statistical_patterns(observations)
        patterns.extend(statistical_patterns)

        return patterns

    def _detect_frequency_patterns(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Detect frequency-based patterns."""
        patterns = []

        # Count attribute combinations
        attribute_counts = defaultdict(int)
        for obs in observations:
            for key, value in obs.items():
                attribute_counts[(key, value)] += 1

        # Create patterns for frequent items
        for (key, value), count in attribute_counts.items():
            if count >= self.min_support:
                pattern = Pattern(
                    pattern_id=f"freq_{key}_{value}",
                    pattern_type="frequency",
                    instances=[{key: value}],
                    frequency=count,
                    confidence=count / len(observations)
                )
                patterns.append(pattern)

        return patterns

    def _detect_sequence_patterns(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Detect sequential patterns."""
        patterns = []

        # Look for sequences in temporal data
        if all("timestamp" in obs for obs in observations):
            sorted_obs = sorted(observations, key=lambda x: x["timestamp"])

            # Find recurring sequences
            for i in range(len(sorted_obs) - 1):
                for j in range(i + 1, min(i + 5, len(sorted_obs))):
                    sequence = sorted_obs[i:j+1]
                    if self._is_repeating_sequence(sequence, sorted_obs):
                        pattern = Pattern(
                            pattern_id=f"seq_{i}_{j}",
                            pattern_type="sequence",
                            instances=sequence,
                            frequency=self._count_sequence_occurrences(sequence, sorted_obs),
                            confidence=0.8
                        )
                        patterns.append(pattern)

        return patterns

    async def _detect_statistical_patterns(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Pattern]:
        """Detect statistical patterns."""
        patterns = []

        # Extract numeric features
        numeric_data = self._extract_numeric_features(observations)

        if numeric_data:
            # Detect correlations
            correlations = self._detect_correlations(numeric_data)
            for corr in correlations:
                pattern = Pattern(
                    pattern_id=f"stat_corr_{corr['var1']}_{corr['var2']}",
                    pattern_type="correlation",
                    instances=[corr],
                    frequency=len(observations),
                    confidence=abs(corr['correlation'])
                )
                patterns.append(pattern)

            # Detect trends
            trends = self._detect_trends(numeric_data)
            for trend in trends:
                pattern = Pattern(
                    pattern_id=f"stat_trend_{trend['variable']}",
                    pattern_type="trend",
                    instances=[trend],
                    frequency=len(observations),
                    confidence=trend['confidence']
                )
                patterns.append(pattern)

        return patterns

    def _find_applicable_patterns(
        self,
        query: Dict[str, Any],
        patterns: List[Pattern]
    ) -> List[Pattern]:
        """Find patterns applicable to query."""
        applicable = []

        for pattern in patterns:
            if self._pattern_applies_to_query(pattern, query):
                applicable.append(pattern)

        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)

        return applicable

    def _pattern_applies_to_query(
        self,
        pattern: Pattern,
        query: Dict[str, Any]
    ) -> bool:
        """Check if pattern applies to query."""
        if pattern.pattern_type == "frequency":
            # Check if query matches pattern attributes
            for instance in pattern.instances:
                for key, value in instance.items():
                    if key in query and query[key] != value:
                        return False
            return True

        elif pattern.pattern_type == "sequence":
            # Check if query could be part of sequence
            return any(
                all(k in query for k in instance.keys())
                for instance in pattern.instances
            )

        elif pattern.pattern_type in ["correlation", "trend"]:
            # Check if query involves relevant variables
            for instance in pattern.instances:
                if any(var in query for var in instance.keys()):
                    return True

        return False

    async def _make_prediction(
        self,
        query: Dict[str, Any],
        patterns: List[Pattern]
    ) -> Any:
        """Make prediction based on patterns."""
        if not patterns:
            return None

        # Use strongest pattern
        strongest_pattern = patterns[0]

        if strongest_pattern.pattern_type == "frequency":
            # Predict most frequent value
            return strongest_pattern.instances[0]

        elif strongest_pattern.pattern_type == "sequence":
            # Predict next in sequence
            return self._predict_next_in_sequence(
                query,
                strongest_pattern.instances
            )

        elif strongest_pattern.pattern_type == "correlation":
            # Use correlation for prediction
            return self._predict_from_correlation(
                query,
                strongest_pattern.instances[0]
            )

        elif strongest_pattern.pattern_type == "trend":
            # Extrapolate trend
            return self._extrapolate_trend(
                query,
                strongest_pattern.instances[0]
            )

        return None

    def _calculate_pattern_confidence(self, patterns: List[Pattern]) -> float:
        """Calculate confidence based on patterns."""
        if not patterns:
            return 0.0

        # Weighted average of pattern confidences
        total_weight = 0
        weighted_sum = 0

        for pattern in patterns:
            weight = pattern.frequency
            weighted_sum += pattern.confidence * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight

        return 0.0

    def _is_repeating_sequence(
        self,
        sequence: List[Dict[str, Any]],
        all_obs: List[Dict[str, Any]]
    ) -> bool:
        """Check if sequence repeats in observations."""
        occurrences = self._count_sequence_occurrences(sequence, all_obs)
        return occurrences >= 2

    def _count_sequence_occurrences(
        self,
        sequence: List[Dict[str, Any]],
        all_obs: List[Dict[str, Any]]
    ) -> int:
        """Count occurrences of sequence in observations."""
        count = 0
        seq_len = len(sequence)

        for i in range(len(all_obs) - seq_len + 1):
            if self._sequences_match(sequence, all_obs[i:i+seq_len]):
                count += 1

        return count

    def _sequences_match(
        self,
        seq1: List[Dict[str, Any]],
        seq2: List[Dict[str, Any]]
    ) -> bool:
        """Check if two sequences match."""
        if len(seq1) != len(seq2):
            return False

        for s1, s2 in zip(seq1, seq2):
            # Check key attributes match
            for key in ["type", "action", "category"]:
                if key in s1 and key in s2:
                    if s1[key] != s2[key]:
                        return False

        return True

    def _extract_numeric_features(
        self,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract numeric features from observations."""
        numeric_data = defaultdict(list)

        for obs in observations:
            for key, value in obs.items():
                if isinstance(value, (int, float)):
                    numeric_data[key].append(float(value))

        return dict(numeric_data)

    def _detect_correlations(
        self,
        numeric_data: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Detect correlations between variables."""
        correlations = []
        variables = list(numeric_data.keys())

        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                var1, var2 = variables[i], variables[j]
                data1, data2 = numeric_data[var1], numeric_data[var2]

                if len(data1) == len(data2) and len(data1) > 2:
                    corr_coef, p_value = stats.pearsonr(data1, data2)

                    if abs(corr_coef) > 0.5 and p_value < 0.05:
                        correlations.append({
                            "var1": var1,
                            "var2": var2,
                            "correlation": corr_coef,
                            "p_value": p_value
                        })

        return correlations

    def _detect_trends(
        self,
        numeric_data: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Detect trends in numeric data."""
        trends = []

        for variable, values in numeric_data.items():
            if len(values) > 3:
                # Simple linear trend detection
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

                if abs(r_value) > 0.7 and p_value < 0.05:
                    trends.append({
                        "variable": variable,
                        "trend": "increasing" if slope > 0 else "decreasing",
                        "slope": slope,
                        "confidence": abs(r_value)
                    })

        return trends

    def _predict_next_in_sequence(
        self,
        query: Dict[str, Any],
        sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict next element in sequence."""
        # Simple next element prediction
        if sequence:
            return sequence[-1]  # Return last element as prediction
        return None

    def _predict_from_correlation(
        self,
        query: Dict[str, Any],
        correlation: Dict[str, Any]
    ) -> Any:
        """Make prediction based on correlation."""
        # Simple correlation-based prediction
        if correlation["var1"] in query:
            # Predict var2 based on var1
            return {
                correlation["var2"]: query[correlation["var1"]] * correlation["correlation"]
            }
        return None

    def _extrapolate_trend(
        self,
        query: Dict[str, Any],
        trend: Dict[str, Any]
    ) -> Any:
        """Extrapolate trend for prediction."""
        # Simple trend extrapolation
        if "time_step" in query:
            predicted_value = trend["slope"] * query["time_step"]
            return {trend["variable"]: predicted_value}
        return None

    def add_knowledge(self, knowledge: Union[Dict[str, Any], Pattern]) -> None:
        """Add knowledge to inductive reasoner."""
        if isinstance(knowledge, Pattern):
            self.patterns.append(knowledge)
        else:
            self.observations.append(knowledge)

    def validate_inference(self, result: InferenceResult) -> bool:
        """Validate inductive inference result."""
        return result.confidence_score >= self.confidence_threshold


class AbductiveReasoner(ReasoningEngine):
    """Abductive reasoning for best explanation finding."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize abductive reasoner."""
        super().__init__(config)
        self.hypotheses: List[Hypothesis] = []
        self.observations: List[str] = []
        self.background_knowledge: Dict[str, Any] = {}

    async def infer(
        self,
        premises: List[str],
        query: str,
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform abductive inference to find best explanation."""
        start_time = time.time()

        try:
            # Add observations
            self.observations.extend(premises)

            # Generate hypotheses
            hypotheses = await self._generate_hypotheses(premises, query)

            # Evaluate hypotheses
            evaluated_hypotheses = await self._evaluate_hypotheses(
                hypotheses,
                premises
            )

            # Select best explanation
            best_hypothesis = self._select_best_explanation(evaluated_hypotheses)

            # Calculate confidence
            confidence = self._calculate_explanation_confidence(best_hypothesis)

            execution_time = (time.time() - start_time) * 1000

            return InferenceResult(
                reasoning_type=ReasoningType.ABDUCTIVE,
                status=InferenceStatus.COMPLETED if best_hypothesis else InferenceStatus.UNCERTAIN,
                conclusion=best_hypothesis.explanation if best_hypothesis else None,
                confidence_score=confidence,
                evidence=premises,
                reasoning_path=[{
                    "method": "abduction",
                    "hypotheses_generated": len(hypotheses),
                    "best_hypothesis": best_hypothesis.explanation if best_hypothesis else None,
                    "likelihood": best_hypothesis.likelihood if best_hypothesis else 0
                }],
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Abductive inference failed: {str(e)}")
            return InferenceResult(
                reasoning_type=ReasoningType.ABDUCTIVE,
                status=InferenceStatus.FAILED,
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def _generate_hypotheses(
        self,
        observations: List[str],
        query: str
    ) -> List[Hypothesis]:
        """Generate possible explanations for observations."""
        hypotheses = []

        # Method 1: Rule-based hypothesis generation
        rule_hypotheses = self._generate_rule_based_hypotheses(observations)
        hypotheses.extend(rule_hypotheses)

        # Method 2: Causal hypothesis generation
        causal_hypotheses = self._generate_causal_hypotheses(observations)
        hypotheses.extend(causal_hypotheses)

        # Method 3: Analogical hypothesis generation
        analogical_hypotheses = await self._generate_analogical_hypotheses(
            observations,
            query
        )
        hypotheses.extend(analogical_hypotheses)

        return hypotheses

    def _generate_rule_based_hypotheses(
        self,
        observations: List[str]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on rules."""
        hypotheses = []

        # Common explanation patterns
        patterns = {
            "error": "System malfunction or configuration issue",
            "delay": "Network latency or resource contention",
            "missing": "Data loss or incomplete processing",
            "unexpected": "External interference or edge case"
        }

        for obs in observations:
            for keyword, explanation in patterns.items():
                if keyword in obs.lower():
                    hypothesis = Hypothesis(
                        hypothesis_id=f"rule_{keyword}",
                        explanation=explanation,
                        evidence_for=[obs],
                        evidence_against=[],
                        likelihood=0.6,
                        simplicity_score=0.8,
                        explanatory_power=0.5
                    )
                    hypotheses.append(hypothesis)

        return hypotheses

    def _generate_causal_hypotheses(
        self,
        observations: List[str]
    ) -> List[Hypothesis]:
        """Generate causal explanations."""
        hypotheses = []

        # Build causal graph
        causal_graph = self._build_causal_graph(observations)

        # Find root causes
        for obs in observations:
            root_causes = self._find_root_causes(obs, causal_graph)
            for cause in root_causes:
                hypothesis = Hypothesis(
                    hypothesis_id=f"causal_{cause}",
                    explanation=f"Root cause: {cause}",
                    evidence_for=[obs],
                    evidence_against=[],
                    likelihood=0.7,
                    simplicity_score=0.6,
                    explanatory_power=0.7
                )
                hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_analogical_hypotheses(
        self,
        observations: List[str],
        query: str
    ) -> List[Hypothesis]:
        """Generate hypotheses by analogy."""
        hypotheses = []

        # Find similar past cases
        similar_cases = self._find_similar_cases(observations)

        for case in similar_cases:
            hypothesis = Hypothesis(
                hypothesis_id=f"analog_{case['id']}",
                explanation=f"Similar to: {case['explanation']}",
                evidence_for=observations,
                evidence_against=[],
                likelihood=case['similarity'],
                simplicity_score=0.5,
                explanatory_power=0.6
            )
            hypotheses.append(hypothesis)

        return hypotheses

    async def _evaluate_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        observations: List[str]
    ) -> List[Hypothesis]:
        """Evaluate hypotheses against observations."""
        evaluated = []

        for hypothesis in hypotheses:
            # Check consistency with observations
            consistency = self._check_consistency(hypothesis, observations)

            # Calculate explanatory coverage
            coverage = self._calculate_coverage(hypothesis, observations)

            # Update hypothesis scores
            hypothesis.likelihood *= consistency
            hypothesis.explanatory_power *= coverage

            evaluated.append(hypothesis)

        return evaluated

    def _select_best_explanation(
        self,
        hypotheses: List[Hypothesis]
    ) -> Optional[Hypothesis]:
        """Select best explanation using inference to best explanation."""
        if not hypotheses:
            return None

        # Score each hypothesis
        scored_hypotheses = []
        for h in hypotheses:
            score = self._score_hypothesis(h)
            scored_hypotheses.append((h, score))

        # Sort by score
        scored_hypotheses.sort(key=lambda x: x[1], reverse=True)

        return scored_hypotheses[0][0] if scored_hypotheses else None

    def _score_hypothesis(self, hypothesis: Hypothesis) -> float:
        """Score hypothesis based on multiple criteria."""
        # Weighted combination of factors
        score = (
            hypothesis.likelihood * 0.4 +
            hypothesis.simplicity_score * 0.3 +
            hypothesis.explanatory_power * 0.3
        )

        # Penalty for evidence against
        penalty = len(hypothesis.evidence_against) * 0.1
        score = max(0, score - penalty)

        return score

    def _calculate_explanation_confidence(
        self,
        hypothesis: Optional[Hypothesis]
    ) -> float:
        """Calculate confidence in explanation."""
        if not hypothesis:
            return 0.0

        # Base confidence on hypothesis quality
        confidence = self._score_hypothesis(hypothesis)

        # Adjust based on evidence
        evidence_ratio = len(hypothesis.evidence_for) / max(
            len(hypothesis.evidence_for) + len(hypothesis.evidence_against),
            1
        )
        confidence *= evidence_ratio

        return min(1.0, confidence)

    def _build_causal_graph(
        self,
        observations: List[str]
    ) -> nx.DiGraph:
        """Build causal graph from observations."""
        graph = nx.DiGraph()

        # Simple causal relationships
        # In production, would use more sophisticated causal discovery
        for i, obs1 in enumerate(observations):
            graph.add_node(i, observation=obs1)
            for j, obs2 in enumerate(observations):
                if i != j:
                    # Check for causal keywords
                    if any(word in obs1.lower() for word in ["causes", "leads to", "results in"]):
                        graph.add_edge(i, j, weight=0.8)

        return graph

    def _find_root_causes(
        self,
        observation: str,
        causal_graph: nx.DiGraph
    ) -> List[str]:
        """Find root causes in causal graph."""
        root_causes = []

        # Find node for observation
        obs_node = None
        for node, data in causal_graph.nodes(data=True):
            if data.get("observation") == observation:
                obs_node = node
                break

        if obs_node is not None:
            # Find ancestors (potential causes)
            ancestors = nx.ancestors(causal_graph, obs_node)
            for ancestor in ancestors:
                cause = causal_graph.nodes[ancestor].get("observation", "Unknown")
                root_causes.append(cause)

        return root_causes

    def _find_similar_cases(
        self,
        observations: List[str]
    ) -> List[Dict[str, Any]]:
        """Find similar past cases."""
        similar_cases = []

        # Simulated case database
        past_cases = [
            {
                "id": "case1",
                "observations": ["error in processing", "timeout occurred"],
                "explanation": "Database connection pool exhausted"
            },
            {
                "id": "case2",
                "observations": ["unexpected output", "validation failed"],
                "explanation": "Input data format mismatch"
            }
        ]

        for case in past_cases:
            similarity = self._calculate_case_similarity(
                observations,
                case["observations"]
            )
            if similarity > 0.5:
                similar_cases.append({
                    "id": case["id"],
                    "explanation": case["explanation"],
                    "similarity": similarity
                })

        return similar_cases

    def _calculate_case_similarity(
        self,
        obs1: List[str],
        obs2: List[str]
    ) -> float:
        """Calculate similarity between observation sets."""
        # Simple word overlap similarity
        words1 = set(" ".join(obs1).lower().split())
        words2 = set(" ".join(obs2).lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _check_consistency(
        self,
        hypothesis: Hypothesis,
        observations: List[str]
    ) -> float:
        """Check hypothesis consistency with observations."""
        # Simple consistency check
        consistent_count = 0
        for obs in observations:
            if not self._contradicts(hypothesis.explanation, obs):
                consistent_count += 1

        return consistent_count / len(observations) if observations else 0

    def _contradicts(self, explanation: str, observation: str) -> bool:
        """Check if explanation contradicts observation."""
        # Simple contradiction detection
        contradictory_pairs = [
            ("increase", "decrease"),
            ("success", "failure"),
            ("present", "absent")
        ]

        for word1, word2 in contradictory_pairs:
            if word1 in explanation.lower() and word2 in observation.lower():
                return True
            if word2 in explanation.lower() and word1 in observation.lower():
                return True

        return False

    def _calculate_coverage(
        self,
        hypothesis: Hypothesis,
        observations: List[str]
    ) -> float:
        """Calculate how well hypothesis covers observations."""
        # Simple coverage calculation
        covered_count = 0
        for obs in observations:
            if self._explains(hypothesis.explanation, obs):
                covered_count += 1

        return covered_count / len(observations) if observations else 0

    def _explains(self, explanation: str, observation: str) -> bool:
        """Check if explanation covers observation."""
        # Simple explanation check
        key_words = observation.lower().split()
        explanation_words = explanation.lower().split()

        overlap = len(set(key_words) & set(explanation_words))
        return overlap > 0

    def add_knowledge(self, knowledge: Any) -> None:
        """Add knowledge to abductive reasoner."""
        if isinstance(knowledge, Hypothesis):
            self.hypotheses.append(knowledge)
        elif isinstance(knowledge, str):
            self.observations.append(knowledge)
        elif isinstance(knowledge, dict):
            self.background_knowledge.update(knowledge)

    def validate_inference(self, result: InferenceResult) -> bool:
        """Validate abductive inference result."""
        return result.confidence_score >= self.confidence_threshold


class AnalogicalReasoner(ReasoningEngine):
    """Analogical reasoning with similarity-based inference."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize analogical reasoner."""
        super().__init__(config)
        self.case_base: List[Dict[str, Any]] = []
        self.similarity_threshold = config.get("similarity_threshold", 0.6)

    async def infer(
        self,
        premises: List[Dict[str, Any]],
        query: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform analogical inference."""
        start_time = time.time()

        try:
            # Find similar cases
            similar_cases = await self._find_similar_cases(query, premises)

            # Create analogical mappings
            mappings = self._create_mappings(query, similar_cases)

            # Transfer solutions
            solution = await self._transfer_solution(query, mappings)

            # Calculate confidence
            confidence = self._calculate_analogy_confidence(mappings)

            execution_time = (time.time() - start_time) * 1000

            return InferenceResult(
                reasoning_type=ReasoningType.ANALOGICAL,
                status=InferenceStatus.COMPLETED if solution else InferenceStatus.UNCERTAIN,
                conclusion=solution,
                confidence_score=confidence,
                evidence=[str(m) for m in mappings],
                reasoning_path=[{
                    "method": "analogical_transfer",
                    "similar_cases": len(similar_cases),
                    "mappings_created": len(mappings)
                }],
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Analogical inference failed: {str(e)}")
            return InferenceResult(
                reasoning_type=ReasoningType.ANALOGICAL,
                status=InferenceStatus.FAILED,
                confidence_score=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def _find_similar_cases(
        self,
        query: Dict[str, Any],
        premises: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find similar cases for analogy."""
        similar_cases = []

        # Add premises to case base
        self.case_base.extend(premises)

        for case in self.case_base:
            similarity = self._calculate_similarity(query, case)
            if similarity >= self.similarity_threshold:
                similar_cases.append({
                    "case": case,
                    "similarity": similarity
                })

        # Sort by similarity
        similar_cases.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_cases[:5]  # Return top 5

    def _create_mappings(
        self,
        query: Dict[str, Any],
        similar_cases: List[Dict[str, Any]]
    ) -> List[Analogy]:
        """Create analogical mappings."""
        mappings = []

        for case_info in similar_cases:
            case = case_info["case"]
            similarity = case_info["similarity"]

            # Create structural alignment
            alignment = self._align_structures(query, case)

            # Create mapping
            analogy = Analogy(
                source_domain=str(case),
                target_domain=str(query),
                mappings=alignment,
                similarity_score=similarity,
                structural_alignment=self._calculate_structural_similarity(
                    query,
                    case,
                    alignment
                )
            )
            mappings.append(analogy)

        return mappings

    async def _transfer_solution(
        self,
        query: Dict[str, Any],
        mappings: List[Analogy]
    ) -> Any:
        """Transfer solution from source to target."""
        if not mappings:
            return None

        # Use best mapping
        best_mapping = max(mappings, key=lambda m: m.similarity_score)

        # Extract solution from source
        source_solution = self._extract_solution(best_mapping.source_domain)

        # Adapt solution to target
        adapted_solution = self._adapt_solution(
            source_solution,
            best_mapping.mappings
        )

        return adapted_solution

    def _calculate_similarity(
        self,
        case1: Dict[str, Any],
        case2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between cases."""
        # Surface similarity
        surface_sim = self._surface_similarity(case1, case2)

        # Structural similarity
        structural_sim = self._structural_similarity(case1, case2)

        # Pragmatic similarity
        pragmatic_sim = self._pragmatic_similarity(case1, case2)

        # Weighted combination
        similarity = (
            surface_sim * 0.3 +
            structural_sim * 0.5 +
            pragmatic_sim * 0.2
        )

        return similarity

    def _surface_similarity(
        self,
        case1: Dict[str, Any],
        case2: Dict[str, Any]
    ) -> float:
        """Calculate surface-level similarity."""
        # Compare attribute values
        common_keys = set(case1.keys()) & set(case2.keys())
        if not common_keys:
            return 0.0

        matches = sum(
            1 for key in common_keys
            if case1[key] == case2[key]
        )

        return matches / len(common_keys)

    def _structural_similarity(
        self,
        case1: Dict[str, Any],
        case2: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity."""
        # Compare relationships and patterns
        struct1 = self._extract_structure(case1)
        struct2 = self._extract_structure(case2)

        if not struct1 or not struct2:
            return 0.0

        # Simple structure comparison
        common_relations = len(set(struct1) & set(struct2))
        total_relations = len(set(struct1) | set(struct2))

        return common_relations / total_relations if total_relations > 0 else 0

    def _pragmatic_similarity(
        self,
        case1: Dict[str, Any],
        case2: Dict[str, Any]
    ) -> float:
        """Calculate pragmatic (goal-based) similarity."""
        # Compare goals or purposes
        goal1 = case1.get("goal", case1.get("purpose", ""))
        goal2 = case2.get("goal", case2.get("purpose", ""))

        if not goal1 or not goal2:
            return 0.5  # Neutral if no goals

        # Simple goal comparison
        return 1.0 if goal1 == goal2 else 0.0

    def _align_structures(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any]
    ) -> Dict[str, str]:
        """Align structures between source and target."""
        alignment = {}

        # Simple attribute mapping
        for target_key in target.keys():
            best_match = None
            best_score = 0

            for source_key in source.keys():
                score = self._attribute_similarity(
                    target_key,
                    source_key,
                    target.get(target_key),
                    source.get(source_key)
                )
                if score > best_score:
                    best_score = score
                    best_match = source_key

            if best_match and best_score > 0.5:
                alignment[target_key] = best_match

        return alignment

    def _calculate_structural_similarity(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any],
        alignment: Dict[str, str]
    ) -> float:
        """Calculate structural similarity given alignment."""
        if not alignment:
            return 0.0

        # Check how well aligned attributes match
        match_scores = []
        for target_key, source_key in alignment.items():
            score = self._attribute_similarity(
                target_key,
                source_key,
                target.get(target_key),
                source.get(source_key)
            )
            match_scores.append(score)

        return np.mean(match_scores) if match_scores else 0.0

    def _attribute_similarity(
        self,
        key1: str,
        key2: str,
        val1: Any,
        val2: Any
    ) -> float:
        """Calculate similarity between attributes."""
        # Key similarity
        key_sim = 1.0 if key1 == key2 else 0.5 if key1.lower() == key2.lower() else 0.0

        # Value similarity
        if type(val1) == type(val2):
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    val_sim = 1.0
                else:
                    val_sim = 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                val_sim = 1.0
            else:
                val_sim = 0.0
        else:
            val_sim = 0.0

        return (key_sim + val_sim) / 2

    def _extract_structure(self, case: Dict[str, Any]) -> List[str]:
        """Extract structural relations from case."""
        relations = []

        # Extract simple relations
        for key, value in case.items():
            if isinstance(value, dict):
                for sub_key in value.keys():
                    relations.append(f"{key}->{sub_key}")
            elif isinstance(value, list):
                relations.append(f"{key}->list[{len(value)}]")
            else:
                relations.append(f"{key}->value")

        return relations

    def _extract_solution(self, source: str) -> Any:
        """Extract solution from source case."""
        # Parse source string back to dict
        try:
            source_dict = ast.literal_eval(source)  # SECURITY FIX: Use ast.literal_eval
            return source_dict.get("solution", source_dict.get("result"))
        except Exception as e:
            logger.debug(f"Failed to parse source solution: {e}")
            return None

    def _adapt_solution(
        self,
        solution: Any,
        mappings: Dict[str, str]
    ) -> Any:
        """Adapt solution using mappings."""
        if solution is None:
            return None

        # Simple adaptation
        if isinstance(solution, dict):
            adapted = {}
            for key, value in solution.items():
                # Map key if possible
                mapped_key = next(
                    (k for k, v in mappings.items() if v == key),
                    key
                )
                adapted[mapped_key] = value
            return adapted

        return solution

    def _calculate_analogy_confidence(self, mappings: List[Analogy]) -> float:
        """Calculate confidence in analogical inference."""
        if not mappings:
            return 0.0

        # Average similarity and alignment scores
        similarities = [m.similarity_score for m in mappings]
        alignments = [m.structural_alignment for m in mappings]

        avg_similarity = np.mean(similarities)
        avg_alignment = np.mean(alignments)

        # Weighted combination
        confidence = avg_similarity * 0.6 + avg_alignment * 0.4

        return min(1.0, confidence)

    def add_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Add case to case base."""
        self.case_base.append(knowledge)

    def validate_inference(self, result: InferenceResult) -> bool:
        """Validate analogical inference result."""
        return result.confidence_score >= self.confidence_threshold


class PatternDetector:
    """Helper class for pattern detection."""

    def detect_patterns(self, data: List[Any]) -> List[Pattern]:
        """Detect patterns in data."""
        # Implementation for various pattern detection algorithms
        return []


class ReasoningFramework:
    """Main framework for reasoning capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reasoning framework."""
        self.config = config or {}

        # Initialize reasoning engines
        self.engines = {
            ReasoningType.DEDUCTIVE: DeductiveReasoner(config),
            ReasoningType.INDUCTIVE: InductiveReasoner(config),
            ReasoningType.ABDUCTIVE: AbductiveReasoner(config),
            ReasoningType.ANALOGICAL: AnalogicalReasoner(config)
        }

        self.inference_history = []
        self.metrics = defaultdict(lambda: {
            "total_inferences": 0,
            "successful_inferences": 0,
            "average_confidence": 0.0,
            "average_time_ms": 0.0
        })

    async def reason(
        self,
        reasoning_type: ReasoningType,
        premises: List[Any],
        query: Any,
        context: Dict[str, Any] = None
    ) -> InferenceResult:
        """Perform reasoning using specified type."""
        if reasoning_type not in self.engines:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")

        engine = self.engines[reasoning_type]
        result = await engine.infer(premises, query, context)

        # Update metrics
        self._update_metrics(reasoning_type, result)

        # Store in history
        self.inference_history.append({
            "timestamp": DeterministicClock.now(),
            "type": reasoning_type,
            "result": result
        })

        return result

    async def multi_reason(
        self,
        premises: List[Any],
        query: Any,
        context: Dict[str, Any] = None
    ) -> Dict[ReasoningType, InferenceResult]:
        """Perform multiple types of reasoning and combine results."""
        results = {}

        # Run all applicable reasoning types
        tasks = []
        for reasoning_type, engine in self.engines.items():
            task = engine.infer(premises, query, context)
            tasks.append((reasoning_type, task))

        # Gather results
        for reasoning_type, task in tasks:
            try:
                result = await task
                results[reasoning_type] = result
                self._update_metrics(reasoning_type, result)
            except Exception as e:
                logger.error(f"{reasoning_type} reasoning failed: {str(e)}")

        return results

    def add_knowledge(
        self,
        reasoning_type: ReasoningType,
        knowledge: Any
    ) -> None:
        """Add knowledge to specific reasoning engine."""
        if reasoning_type in self.engines:
            self.engines[reasoning_type].add_knowledge(knowledge)

    def get_metrics(self, reasoning_type: Optional[ReasoningType] = None) -> Dict[str, Any]:
        """Get reasoning metrics."""
        if reasoning_type:
            return self.metrics[reasoning_type]
        return dict(self.metrics)

    def _update_metrics(
        self,
        reasoning_type: ReasoningType,
        result: InferenceResult
    ) -> None:
        """Update metrics for reasoning type."""
        metrics = self.metrics[reasoning_type]

        metrics["total_inferences"] += 1
        if result.status == InferenceStatus.COMPLETED:
            metrics["successful_inferences"] += 1

        # Update running averages
        n = metrics["total_inferences"]
        metrics["average_confidence"] = (
            (metrics["average_confidence"] * (n - 1) + result.confidence_score) / n
        )
        metrics["average_time_ms"] = (
            (metrics["average_time_ms"] * (n - 1) + result.execution_time_ms) / n
        )


