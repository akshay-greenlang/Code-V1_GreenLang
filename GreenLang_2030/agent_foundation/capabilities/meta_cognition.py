"""
Meta-Cognition Framework - Self-reflection and improvement capabilities.

This module implements meta-cognitive capabilities including:
- Self-monitoring and performance tracking
- Confidence estimation and uncertainty handling
- Self-improvement through learning
- Meta-reasoning and strategy selection

Example:
    >>> meta_cognition = MetaCognition(config)
    >>> confidence = await meta_cognition.estimate_confidence(task_result)
    >>> improvements = await meta_cognition.identify_improvements(performance_data)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, validator
import numpy as np
from scipy import stats
import pickle

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Types of performance metrics."""

    ACCURACY = "accuracy"
    SPEED = "speed"
    RESOURCE_USAGE = "resource_usage"
    CONFIDENCE = "confidence"
    ERROR_RATE = "error_rate"
    COMPLETENESS = "completeness"


class ImprovementStrategy(str, Enum):
    """Strategies for self-improvement."""

    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_SWITCHING = "algorithm_switching"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    ERROR_CORRECTION = "error_correction"


class UncertaintyType(str, Enum):
    """Types of uncertainty."""

    ALEATORY = "aleatory"      # Inherent randomness
    EPISTEMIC = "epistemic"    # Knowledge uncertainty
    ONTOLOGICAL = "ontological" # Conceptual uncertainty


@dataclass
class PerformanceRecord:
    """Record of agent performance."""

    task_id: str
    timestamp: datetime
    metrics: Dict[PerformanceMetric, float]
    context: Dict[str, Any]
    outcome: str
    confidence: float
    errors: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class LearningExperience:
    """Experience for learning and improvement."""

    experience_id: str
    task_type: str
    initial_approach: Dict[str, Any]
    outcome: str
    feedback: Optional[str]
    lessons_learned: List[str]
    confidence_delta: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConfidenceEstimate:
    """Confidence estimate with uncertainty quantification."""

    point_estimate: float
    confidence_interval: Tuple[float, float]
    uncertainty_type: UncertaintyType
    contributing_factors: Dict[str, float]
    reliability_score: float


class SelfMonitor:
    """Self-monitoring and performance tracking."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize self-monitor."""
        self.config = config or {}
        self.performance_history: deque = deque(maxlen=1000)
        self.metrics_tracker = defaultdict(list)
        self.anomaly_detector = AnomalyDetector()
        self.monitoring_interval = config.get("monitoring_interval", 10)

    async def monitor_performance(
        self,
        task_id: str,
        task_execution: callable,
        context: Dict[str, Any]
    ) -> PerformanceRecord:
        """Monitor performance during task execution."""
        start_time = time.time()
        start_resources = self._get_resource_usage()
        errors = []

        try:
            # Execute task with monitoring
            result = await task_execution()
            outcome = "success"

        except Exception as e:
            result = None
            outcome = "failure"
            errors.append(str(e))
            logger.error(f"Task {task_id} failed: {str(e)}")

        # Calculate metrics
        execution_time = time.time() - start_time
        end_resources = self._get_resource_usage()
        resource_delta = self._calculate_resource_delta(start_resources, end_resources)

        # Create performance record
        record = PerformanceRecord(
            task_id=task_id,
            timestamp=datetime.now(),
            metrics={
                PerformanceMetric.SPEED: execution_time,
                PerformanceMetric.ACCURACY: 1.0 if outcome == "success" else 0.0,
                PerformanceMetric.ERROR_RATE: len(errors) / max(1, len(errors) + 1),
                PerformanceMetric.RESOURCE_USAGE: sum(resource_delta.values())
            },
            context=context,
            outcome=outcome,
            confidence=self._calculate_execution_confidence(result, errors),
            errors=errors,
            resource_usage=resource_delta
        )

        # Store record
        self.performance_history.append(record)
        self._update_metrics(record)

        # Check for anomalies
        if await self.anomaly_detector.detect_anomaly(record):
            logger.warning(f"Performance anomaly detected in task {task_id}")

        return record

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_io_mb": psutil.disk_io_counters().read_bytes / 1024 / 1024
        }

    def _calculate_resource_delta(
        self,
        start: Dict[str, float],
        end: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate resource usage delta."""
        return {
            key: end.get(key, 0) - start.get(key, 0)
            for key in set(start.keys()) | set(end.keys())
        }

    def _calculate_execution_confidence(
        self,
        result: Any,
        errors: List[str]
    ) -> float:
        """Calculate confidence in execution."""
        if errors:
            return 0.3  # Low confidence with errors

        if result is None:
            return 0.5  # Medium confidence with no result

        return 0.9  # High confidence with successful result

    def _update_metrics(self, record: PerformanceRecord) -> None:
        """Update metrics tracking."""
        for metric, value in record.metrics.items():
            self.metrics_tracker[metric].append({
                "timestamp": record.timestamp,
                "value": value,
                "context": record.context
            })

    async def get_performance_trends(
        self,
        metric: PerformanceMetric,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance trends for a metric."""
        if metric not in self.metrics_tracker:
            return {"trend": "unknown", "data": []}

        data = self.metrics_tracker[metric]

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            data = [d for d in data if d["timestamp"] > cutoff]

        if len(data) < 2:
            return {"trend": "insufficient_data", "data": data}

        # Calculate trend
        values = [d["value"] for d in data]
        timestamps = [(d["timestamp"] - data[0]["timestamp"]).total_seconds()
                     for d in data]

        # Linear regression for trend
        if timestamps and values:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps,
                values
            )

            trend = "improving" if slope > 0 else "declining" if slope < 0 else "stable"

            return {
                "trend": trend,
                "slope": slope,
                "correlation": r_value,
                "p_value": p_value,
                "data": data,
                "current_value": values[-1],
                "average": np.mean(values),
                "std_dev": np.std(values)
            }

        return {"trend": "unknown", "data": data}

    def get_error_patterns(self) -> Dict[str, Any]:
        """Identify error patterns in performance history."""
        error_patterns = defaultdict(int)
        error_contexts = defaultdict(list)

        for record in self.performance_history:
            for error in record.errors:
                # Categorize error
                error_type = self._categorize_error(error)
                error_patterns[error_type] += 1
                error_contexts[error_type].append(record.context)

        # Analyze patterns
        patterns = {}
        for error_type, count in error_patterns.items():
            patterns[error_type] = {
                "frequency": count,
                "percentage": count / len(self.performance_history) * 100,
                "common_contexts": self._find_common_contexts(
                    error_contexts[error_type]
                )
            }

        return patterns

    def _categorize_error(self, error: str) -> str:
        """Categorize error type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower:
            return "memory"
        elif "connection" in error_lower:
            return "connection"
        elif "validation" in error_lower:
            return "validation"
        elif "permission" in error_lower:
            return "permission"
        else:
            return "other"

    def _find_common_contexts(
        self,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find common patterns in contexts."""
        if not contexts:
            return {}

        common = {}

        # Find common keys
        all_keys = set()
        for ctx in contexts:
            all_keys.update(ctx.keys())

        for key in all_keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if values:
                # Find most common value
                value_counts = Counter(values)
                most_common = value_counts.most_common(1)[0]
                if most_common[1] > len(contexts) * 0.5:  # More than 50%
                    common[key] = most_common[0]

        return common


class ConfidenceEstimator:
    """Confidence estimation and uncertainty quantification."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize confidence estimator."""
        self.config = config or {}
        self.calibration_data = []
        self.confidence_model = None
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.3)

    async def estimate_confidence(
        self,
        prediction: Any,
        evidence: List[Any],
        context: Dict[str, Any]
    ) -> ConfidenceEstimate:
        """Estimate confidence in prediction."""
        # Calculate base confidence
        base_confidence = self._calculate_base_confidence(prediction, evidence)

        # Quantify uncertainty
        uncertainty = await self._quantify_uncertainty(prediction, evidence, context)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            base_confidence,
            uncertainty
        )

        # Identify contributing factors
        factors = self._identify_contributing_factors(
            prediction,
            evidence,
            context
        )

        # Calculate reliability
        reliability = self._calculate_reliability(
            base_confidence,
            uncertainty,
            len(evidence)
        )

        return ConfidenceEstimate(
            point_estimate=base_confidence,
            confidence_interval=confidence_interval,
            uncertainty_type=self._classify_uncertainty(uncertainty),
            contributing_factors=factors,
            reliability_score=reliability
        )

    def _calculate_base_confidence(
        self,
        prediction: Any,
        evidence: List[Any]
    ) -> float:
        """Calculate base confidence score."""
        if not evidence:
            return 0.1

        # Evidence strength
        evidence_score = min(1.0, len(evidence) / 10)

        # Prediction consistency
        consistency_score = self._check_prediction_consistency(prediction, evidence)

        # Combine scores
        base_confidence = (evidence_score + consistency_score) / 2

        return base_confidence

    async def _quantify_uncertainty(
        self,
        prediction: Any,
        evidence: List[Any],
        context: Dict[str, Any]
    ) -> float:
        """Quantify uncertainty in prediction."""
        uncertainties = []

        # Aleatory uncertainty (inherent randomness)
        aleatory = self._estimate_aleatory_uncertainty(prediction, context)
        uncertainties.append(aleatory)

        # Epistemic uncertainty (knowledge gaps)
        epistemic = self._estimate_epistemic_uncertainty(evidence, context)
        uncertainties.append(epistemic)

        # Ontological uncertainty (conceptual ambiguity)
        ontological = self._estimate_ontological_uncertainty(prediction, context)
        uncertainties.append(ontological)

        # Combined uncertainty
        return np.mean(uncertainties)

    def _calculate_confidence_interval(
        self,
        point_estimate: float,
        uncertainty: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        # Simple confidence interval based on uncertainty
        margin = uncertainty * 0.5

        lower = max(0.0, point_estimate - margin)
        upper = min(1.0, point_estimate + margin)

        return (lower, upper)

    def _identify_contributing_factors(
        self,
        prediction: Any,
        evidence: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Identify factors contributing to confidence."""
        factors = {}

        # Evidence quality
        factors["evidence_quality"] = self._assess_evidence_quality(evidence)

        # Context relevance
        factors["context_relevance"] = self._assess_context_relevance(context)

        # Prediction complexity
        factors["prediction_complexity"] = self._assess_prediction_complexity(prediction)

        # Historical accuracy
        factors["historical_accuracy"] = self._get_historical_accuracy(context)

        return factors

    def _calculate_reliability(
        self,
        confidence: float,
        uncertainty: float,
        evidence_count: int
    ) -> float:
        """Calculate reliability of confidence estimate."""
        # Higher confidence, lower uncertainty, more evidence = higher reliability
        reliability = (
            confidence * 0.4 +
            (1 - uncertainty) * 0.4 +
            min(1.0, evidence_count / 20) * 0.2
        )

        return reliability

    def _check_prediction_consistency(
        self,
        prediction: Any,
        evidence: List[Any]
    ) -> float:
        """Check consistency between prediction and evidence."""
        # Simplified consistency check
        if not evidence:
            return 0.5

        # Check if evidence supports prediction
        supporting = sum(1 for e in evidence if self._supports(e, prediction))
        consistency = supporting / len(evidence)

        return consistency

    def _supports(self, evidence: Any, prediction: Any) -> bool:
        """Check if evidence supports prediction."""
        # Simple support check - would be more sophisticated in production
        return True  # Placeholder

    def _estimate_aleatory_uncertainty(
        self,
        prediction: Any,
        context: Dict[str, Any]
    ) -> float:
        """Estimate aleatory (inherent) uncertainty."""
        # Check for inherent randomness indicators
        if context.get("stochastic", False):
            return 0.7

        if context.get("noise_level", 0) > 0.5:
            return 0.6

        return 0.2  # Low aleatory uncertainty by default

    def _estimate_epistemic_uncertainty(
        self,
        evidence: List[Any],
        context: Dict[str, Any]
    ) -> float:
        """Estimate epistemic (knowledge) uncertainty."""
        # Based on evidence completeness
        if not evidence:
            return 0.9

        required_evidence = context.get("required_evidence", 10)
        completeness = min(1.0, len(evidence) / required_evidence)

        return 1.0 - completeness

    def _estimate_ontological_uncertainty(
        self,
        prediction: Any,
        context: Dict[str, Any]
    ) -> float:
        """Estimate ontological (conceptual) uncertainty."""
        # Check for conceptual clarity
        if context.get("domain_expertise", 1.0) < 0.5:
            return 0.7

        if context.get("concept_clarity", 1.0) < 0.5:
            return 0.6

        return 0.2  # Low ontological uncertainty by default

    def _classify_uncertainty(self, uncertainty: float) -> UncertaintyType:
        """Classify dominant uncertainty type."""
        # Simplified classification
        if uncertainty > 0.7:
            return UncertaintyType.EPISTEMIC
        elif uncertainty > 0.4:
            return UncertaintyType.ALEATORY
        else:
            return UncertaintyType.ONTOLOGICAL

    def _assess_evidence_quality(self, evidence: List[Any]) -> float:
        """Assess quality of evidence."""
        if not evidence:
            return 0.0

        # Simple quality assessment
        return min(1.0, len(evidence) / 10)

    def _assess_context_relevance(self, context: Dict[str, Any]) -> float:
        """Assess relevance of context."""
        # Check for key context elements
        important_keys = ["domain", "task_type", "constraints"]
        present = sum(1 for key in important_keys if key in context)

        return present / len(important_keys)

    def _assess_prediction_complexity(self, prediction: Any) -> float:
        """Assess complexity of prediction."""
        # Simple complexity measure
        if prediction is None:
            return 0.0

        if isinstance(prediction, (int, float, str, bool)):
            return 0.2  # Simple prediction

        if isinstance(prediction, (list, dict)):
            return 0.5 + min(0.5, len(str(prediction)) / 1000)

        return 0.8  # Complex prediction

    def _get_historical_accuracy(self, context: Dict[str, Any]) -> float:
        """Get historical accuracy for similar contexts."""
        # Would query historical performance database
        return 0.75  # Placeholder

    async def calibrate_confidence(
        self,
        predictions_and_outcomes: List[Tuple[Any, Any, float]]
    ) -> None:
        """Calibrate confidence estimates based on outcomes."""
        # Store calibration data
        self.calibration_data.extend(predictions_and_outcomes)

        # Train calibration model if enough data
        if len(self.calibration_data) > 100:
            await self._train_calibration_model()

    async def _train_calibration_model(self) -> None:
        """Train model for confidence calibration."""
        # Extract features and targets
        X = []
        y = []

        for prediction, outcome, confidence in self.calibration_data:
            # Extract features from prediction
            features = self._extract_calibration_features(prediction, confidence)
            X.append(features)

            # Binary outcome
            y.append(1 if prediction == outcome else 0)

        # Train simple calibration model
        # In production, would use proper ML model
        self.confidence_model = {"mean_accuracy": np.mean(y)}

    def _extract_calibration_features(
        self,
        prediction: Any,
        confidence: float
    ) -> List[float]:
        """Extract features for calibration."""
        return [
            confidence,
            len(str(prediction)),
            1.0 if prediction is not None else 0.0
        ]


class SelfImprovement:
    """Self-improvement through learning and adaptation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize self-improvement module."""
        self.config = config or {}
        self.experience_database = ExperienceDatabase()
        self.strategy_selector = StrategySelector()
        self.parameter_tuner = ParameterTuner()
        self.improvement_threshold = config.get("improvement_threshold", 0.1)

    async def learn_from_feedback(
        self,
        task_result: Dict[str, Any],
        feedback: str,
        context: Dict[str, Any]
    ) -> LearningExperience:
        """Learn from feedback on task performance."""
        # Parse feedback
        feedback_analysis = self._analyze_feedback(feedback)

        # Extract lessons
        lessons = await self._extract_lessons(
            task_result,
            feedback_analysis,
            context
        )

        # Create learning experience
        experience = LearningExperience(
            experience_id=self._generate_experience_id(),
            task_type=context.get("task_type", "unknown"),
            initial_approach=task_result.get("approach", {}),
            outcome=task_result.get("outcome", "unknown"),
            feedback=feedback,
            lessons_learned=lessons,
            confidence_delta=feedback_analysis.get("confidence_change", 0.0)
        )

        # Store experience
        await self.experience_database.store_experience(experience)

        # Apply immediate improvements
        await self._apply_improvements(experience)

        return experience

    def _analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """Analyze feedback to extract insights."""
        analysis = {
            "sentiment": self._analyze_sentiment(feedback),
            "suggestions": self._extract_suggestions(feedback),
            "criticisms": self._extract_criticisms(feedback),
            "confidence_change": 0.0
        }

        # Adjust confidence based on feedback
        if analysis["sentiment"] == "positive":
            analysis["confidence_change"] = 0.1
        elif analysis["sentiment"] == "negative":
            analysis["confidence_change"] = -0.1

        return analysis

    def _analyze_sentiment(self, feedback: str) -> str:
        """Analyze sentiment of feedback."""
        positive_words = ["good", "excellent", "great", "correct", "perfect"]
        negative_words = ["bad", "wrong", "incorrect", "poor", "failure"]

        feedback_lower = feedback.lower()

        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _extract_suggestions(self, feedback: str) -> List[str]:
        """Extract suggestions from feedback."""
        suggestions = []

        # Look for suggestion patterns
        patterns = ["should", "could", "try", "consider", "recommend"]

        for pattern in patterns:
            if pattern in feedback.lower():
                # Extract sentence containing pattern
                sentences = feedback.split(".")
                for sentence in sentences:
                    if pattern in sentence.lower():
                        suggestions.append(sentence.strip())

        return suggestions

    def _extract_criticisms(self, feedback: str) -> List[str]:
        """Extract criticisms from feedback."""
        criticisms = []

        # Look for criticism patterns
        patterns = ["failed", "missed", "wrong", "incorrect", "error"]

        for pattern in patterns:
            if pattern in feedback.lower():
                sentences = feedback.split(".")
                for sentence in sentences:
                    if pattern in sentence.lower():
                        criticisms.append(sentence.strip())

        return criticisms

    async def _extract_lessons(
        self,
        task_result: Dict[str, Any],
        feedback_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract lessons from feedback and results."""
        lessons = []

        # Learn from suggestions
        for suggestion in feedback_analysis["suggestions"]:
            lesson = f"Consider: {suggestion}"
            lessons.append(lesson)

        # Learn from criticisms
        for criticism in feedback_analysis["criticisms"]:
            lesson = f"Avoid: {criticism}"
            lessons.append(lesson)

        # Learn from outcome
        if task_result.get("outcome") == "success":
            lessons.append(f"Successful approach for {context.get('task_type')}")
        else:
            lessons.append(f"Ineffective approach for {context.get('task_type')}")

        return lessons

    async def _apply_improvements(self, experience: LearningExperience) -> None:
        """Apply improvements based on learning experience."""
        # Select improvement strategy
        strategy = await self.strategy_selector.select_strategy(experience)

        if strategy == ImprovementStrategy.PARAMETER_TUNING:
            await self.parameter_tuner.tune_parameters(experience)

        elif strategy == ImprovementStrategy.ALGORITHM_SWITCHING:
            await self._switch_algorithm(experience)

        elif strategy == ImprovementStrategy.KNOWLEDGE_ACQUISITION:
            await self._acquire_knowledge(experience)

        elif strategy == ImprovementStrategy.STRATEGY_ADAPTATION:
            await self._adapt_strategy(experience)

        elif strategy == ImprovementStrategy.ERROR_CORRECTION:
            await self._correct_errors(experience)

    async def _switch_algorithm(self, experience: LearningExperience) -> None:
        """Switch to better algorithm based on experience."""
        # Implementation for algorithm switching
        pass

    async def _acquire_knowledge(self, experience: LearningExperience) -> None:
        """Acquire new knowledge from experience."""
        # Implementation for knowledge acquisition
        pass

    async def _adapt_strategy(self, experience: LearningExperience) -> None:
        """Adapt strategy based on experience."""
        # Implementation for strategy adaptation
        pass

    async def _correct_errors(self, experience: LearningExperience) -> None:
        """Correct errors identified in experience."""
        # Implementation for error correction
        pass

    def _generate_experience_id(self) -> str:
        """Generate unique experience ID."""
        timestamp = datetime.now().isoformat()
        return f"exp_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"

    async def identify_knowledge_gaps(
        self,
        performance_history: List[PerformanceRecord]
    ) -> List[Dict[str, Any]]:
        """Identify gaps in knowledge from performance."""
        gaps = []

        # Analyze failures
        failures = [r for r in performance_history if r.outcome == "failure"]

        # Group failures by context
        failure_contexts = defaultdict(list)
        for failure in failures:
            key = self._context_key(failure.context)
            failure_contexts[key].append(failure)

        # Identify patterns
        for context_key, failures in failure_contexts.items():
            if len(failures) > 2:  # Repeated failures
                gap = {
                    "context": context_key,
                    "failure_count": len(failures),
                    "common_errors": self._find_common_errors(failures),
                    "suggested_learning": self._suggest_learning(failures)
                }
                gaps.append(gap)

        return gaps

    def _context_key(self, context: Dict[str, Any]) -> str:
        """Create key from context for grouping."""
        key_parts = []
        for k in sorted(["task_type", "domain", "complexity"]):
            if k in context:
                key_parts.append(f"{k}:{context[k]}")
        return "|".join(key_parts)

    def _find_common_errors(
        self,
        failures: List[PerformanceRecord]
    ) -> List[str]:
        """Find common errors across failures."""
        all_errors = []
        for failure in failures:
            all_errors.extend(failure.errors)

        # Count occurrences
        error_counts = Counter(all_errors)

        # Return most common
        common = []
        for error, count in error_counts.most_common(3):
            if count > 1:
                common.append(error)

        return common

    def _suggest_learning(
        self,
        failures: List[PerformanceRecord]
    ) -> List[str]:
        """Suggest learning to address failures."""
        suggestions = []

        # Analyze error types
        error_types = set()
        for failure in failures:
            for error in failure.errors:
                error_type = self._categorize_error(error)
                error_types.add(error_type)

        # Map error types to learning suggestions
        learning_map = {
            "timeout": "Learn about optimization and async processing",
            "memory": "Learn about memory management and efficiency",
            "validation": "Learn about input validation and constraints",
            "logic": "Learn about reasoning and problem-solving",
            "data": "Learn about data handling and formats"
        }

        for error_type in error_types:
            if error_type in learning_map:
                suggestions.append(learning_map[error_type])

        return suggestions

    def _categorize_error(self, error: str) -> str:
        """Categorize error for learning suggestions."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower:
            return "memory"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation"
        elif "logic" in error_lower or "reasoning" in error_lower:
            return "logic"
        elif "data" in error_lower or "format" in error_lower:
            return "data"
        else:
            return "other"


class MetaReasoner:
    """Meta-reasoning for strategy selection and resource allocation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize meta-reasoner."""
        self.config = config or {}
        self.strategy_history = []
        self.resource_allocator = ResourceAllocator()

    async def select_reasoning_strategy(
        self,
        task: Dict[str, Any],
        available_strategies: List[str],
        constraints: Dict[str, Any]
    ) -> str:
        """Select best reasoning strategy for task."""
        # Evaluate each strategy
        strategy_scores = {}

        for strategy in available_strategies:
            score = await self._evaluate_strategy(strategy, task, constraints)
            strategy_scores[strategy] = score

        # Select highest scoring strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)

        # Record selection
        self.strategy_history.append({
            "task": task,
            "selected": best_strategy,
            "scores": strategy_scores,
            "timestamp": datetime.now()
        })

        return best_strategy

    async def _evaluate_strategy(
        self,
        strategy: str,
        task: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> float:
        """Evaluate strategy for task."""
        score = 0.0

        # Task compatibility
        compatibility = self._assess_compatibility(strategy, task)
        score += compatibility * 0.4

        # Resource requirements
        resource_fit = self._assess_resource_fit(strategy, constraints)
        score += resource_fit * 0.3

        # Historical performance
        historical = self._get_historical_performance(strategy, task)
        score += historical * 0.3

        return score

    def _assess_compatibility(
        self,
        strategy: str,
        task: Dict[str, Any]
    ) -> float:
        """Assess strategy-task compatibility."""
        # Strategy-task compatibility matrix
        compatibility_matrix = {
            "deductive": {"logical": 0.9, "analytical": 0.8, "creative": 0.3},
            "inductive": {"pattern": 0.9, "prediction": 0.8, "logical": 0.5},
            "abductive": {"diagnostic": 0.9, "explanatory": 0.8, "predictive": 0.4},
            "analogical": {"similar": 0.9, "transfer": 0.8, "novel": 0.6}
        }

        task_type = task.get("type", "unknown")
        return compatibility_matrix.get(strategy, {}).get(task_type, 0.5)

    def _assess_resource_fit(
        self,
        strategy: str,
        constraints: Dict[str, Any]
    ) -> float:
        """Assess if strategy fits resource constraints."""
        # Strategy resource requirements
        resource_requirements = {
            "deductive": {"cpu": 0.3, "memory": 0.2, "time": 0.2},
            "inductive": {"cpu": 0.5, "memory": 0.6, "time": 0.5},
            "abductive": {"cpu": 0.4, "memory": 0.4, "time": 0.4},
            "analogical": {"cpu": 0.3, "memory": 0.5, "time": 0.3}
        }

        requirements = resource_requirements.get(strategy, {})
        available = constraints.get("resources", {})

        # Check if requirements fit within constraints
        fit_scores = []
        for resource, required in requirements.items():
            available_amount = available.get(resource, 1.0)
            fit = min(1.0, available_amount / required)
            fit_scores.append(fit)

        return np.mean(fit_scores) if fit_scores else 0.5

    def _get_historical_performance(
        self,
        strategy: str,
        task: Dict[str, Any]
    ) -> float:
        """Get historical performance of strategy on similar tasks."""
        # Filter history for similar tasks
        similar_tasks = [
            h for h in self.strategy_history
            if h["selected"] == strategy and
            self._tasks_similar(h["task"], task)
        ]

        if not similar_tasks:
            return 0.5  # Default performance

        # Calculate average performance
        # In production, would track actual outcomes
        return 0.7  # Placeholder

    def _tasks_similar(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """Check if two tasks are similar."""
        # Simple similarity check
        return task1.get("type") == task2.get("type")

    async def allocate_computational_budget(
        self,
        tasks: List[Dict[str, Any]],
        total_budget: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Allocate computational resources to tasks."""
        return await self.resource_allocator.allocate(tasks, total_budget)

    def handle_uncertainty(
        self,
        uncertainty_level: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide how to handle uncertainty."""
        handling_strategy = {}

        if uncertainty_level > 0.7:
            # High uncertainty
            handling_strategy = {
                "approach": "conservative",
                "actions": [
                    "gather_more_evidence",
                    "use_fallback_strategy",
                    "increase_confidence_threshold"
                ],
                "risk_tolerance": 0.2
            }

        elif uncertainty_level > 0.4:
            # Medium uncertainty
            handling_strategy = {
                "approach": "balanced",
                "actions": [
                    "proceed_with_caution",
                    "monitor_closely",
                    "prepare_contingencies"
                ],
                "risk_tolerance": 0.5
            }

        else:
            # Low uncertainty
            handling_strategy = {
                "approach": "confident",
                "actions": [
                    "proceed_normally",
                    "optimize_for_performance"
                ],
                "risk_tolerance": 0.8
            }

        return handling_strategy

    async def generate_explanation(
        self,
        decision: Any,
        reasoning_path: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Generate explanation for decision."""
        explanation_parts = []

        # Explain decision
        explanation_parts.append(f"Decision: {decision}")

        # Explain reasoning steps
        explanation_parts.append("\nReasoning process:")
        for i, step in enumerate(reasoning_path, 1):
            explanation_parts.append(
                f"{i}. {step.get('action', 'Step')} - {step.get('reason', 'Processing')}"
            )

        # Explain key factors
        if "factors" in context:
            explanation_parts.append("\nKey factors considered:")
            for factor, weight in context["factors"].items():
                explanation_parts.append(f"- {factor}: {weight:.2f}")

        # Explain confidence
        if "confidence" in context:
            explanation_parts.append(
                f"\nConfidence level: {context['confidence']:.2%}"
            )

        return "\n".join(explanation_parts)


class ExperienceDatabase:
    """Database for storing and retrieving learning experiences."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize experience database."""
        self.storage_path = storage_path or Path("experience_db.pkl")
        self.experiences: List[LearningExperience] = []
        self._load_experiences()

    async def store_experience(self, experience: LearningExperience) -> None:
        """Store learning experience."""
        self.experiences.append(experience)
        await self._save_experiences()

    async def retrieve_similar_experiences(
        self,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[LearningExperience]:
        """Retrieve similar experiences."""
        # Calculate similarity scores
        scored_experiences = []
        for exp in self.experiences:
            similarity = self._calculate_similarity(exp, context)
            scored_experiences.append((exp, similarity))

        # Sort by similarity
        scored_experiences.sort(key=lambda x: x[1], reverse=True)

        # Return top experiences
        return [exp for exp, _ in scored_experiences[:limit]]

    def _calculate_similarity(
        self,
        experience: LearningExperience,
        context: Dict[str, Any]
    ) -> float:
        """Calculate similarity between experience and context."""
        # Simple similarity based on task type
        if experience.task_type == context.get("task_type"):
            return 0.8
        return 0.2

    async def _save_experiences(self) -> None:
        """Save experiences to storage."""
        try:
            with open(self.storage_path, "wb") as f:
                pickle.dump(self.experiences, f)
        except Exception as e:
            logger.error(f"Failed to save experiences: {str(e)}")

    def _load_experiences(self) -> None:
        """Load experiences from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "rb") as f:
                    self.experiences = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load experiences: {str(e)}")
                self.experiences = []


class AnomalyDetector:
    """Detect anomalies in performance."""

    async def detect_anomaly(self, record: PerformanceRecord) -> bool:
        """Detect if record is anomalous."""
        # Simple anomaly detection
        if record.metrics.get(PerformanceMetric.ERROR_RATE, 0) > 0.5:
            return True
        if record.metrics.get(PerformanceMetric.SPEED, 0) > 30:  # 30 seconds
            return True
        return False


class StrategySelector:
    """Select improvement strategies."""

    async def select_strategy(
        self,
        experience: LearningExperience
    ) -> ImprovementStrategy:
        """Select improvement strategy based on experience."""
        # Simple strategy selection
        if "parameter" in " ".join(experience.lessons_learned).lower():
            return ImprovementStrategy.PARAMETER_TUNING
        elif "algorithm" in " ".join(experience.lessons_learned).lower():
            return ImprovementStrategy.ALGORITHM_SWITCHING
        else:
            return ImprovementStrategy.STRATEGY_ADAPTATION


class ParameterTuner:
    """Tune parameters for improvement."""

    async def tune_parameters(self, experience: LearningExperience) -> Dict[str, Any]:
        """Tune parameters based on experience."""
        # Placeholder for parameter tuning
        return {}


class ResourceAllocator:
    """Allocate computational resources."""

    async def allocate(
        self,
        tasks: List[Dict[str, Any]],
        total_budget: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Allocate resources to tasks."""
        allocations = {}

        # Simple proportional allocation based on priority
        total_priority = sum(t.get("priority", 1) for t in tasks)

        for task in tasks:
            task_id = task.get("id", str(id(task)))
            priority = task.get("priority", 1)
            weight = priority / total_priority

            allocation = {}
            for resource, amount in total_budget.items():
                allocation[resource] = amount * weight

            allocations[task_id] = allocation

        return allocations


class PerformanceTracker:
    """Track and analyze performance metrics."""

    def __init__(self):
        """Initialize performance tracker."""
        self.metrics = defaultdict(list)

    def track(self, metric_name: str, value: float) -> None:
        """Track a metric value."""
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now()
        })

    def get_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = [m["value"] for m in self.metrics[metric_name]]

        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }


class MetaCognition:
    """Main meta-cognition framework."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize meta-cognition framework."""
        self.config = config or {}
        self.self_monitor = SelfMonitor(config)
        self.confidence_estimator = ConfidenceEstimator(config)
        self.self_improvement = SelfImprovement(config)
        self.meta_reasoner = MetaReasoner(config)
        self.performance_tracker = PerformanceTracker()

    async def monitor_and_learn(
        self,
        task_id: str,
        task_execution: callable,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor task execution and learn from it."""
        # Monitor performance
        performance = await self.self_monitor.monitor_performance(
            task_id,
            task_execution,
            context
        )

        # Estimate confidence
        confidence = await self.confidence_estimator.estimate_confidence(
            performance.outcome,
            [performance],
            context
        )

        # Identify improvements if needed
        if performance.outcome == "failure" or confidence.point_estimate < 0.5:
            gaps = await self.self_improvement.identify_knowledge_gaps(
                [performance]
            )

            # Learn from experience
            experience = await self.self_improvement.learn_from_feedback(
                {"outcome": performance.outcome, "approach": context},
                "Auto-generated feedback based on performance",
                context
            )

        # Track metrics
        self.performance_tracker.track("confidence", confidence.point_estimate)
        self.performance_tracker.track("execution_time",
                                     performance.metrics.get(PerformanceMetric.SPEED, 0))

        return {
            "performance": performance,
            "confidence": confidence,
            "metrics_summary": self.performance_tracker.get_summary("confidence")
        }

    async def estimate_confidence(
        self,
        task_result: Any,
        evidence: List[Any] = None,
        context: Dict[str, Any] = None
    ) -> ConfidenceEstimate:
        """Estimate confidence in task result."""
        return await self.confidence_estimator.estimate_confidence(
            task_result,
            evidence or [],
            context or {}
        )

    async def identify_improvements(
        self,
        performance_data: List[PerformanceRecord]
    ) -> List[Dict[str, Any]]:
        """Identify potential improvements from performance data."""
        return await self.self_improvement.identify_knowledge_gaps(performance_data)

    async def select_strategy(
        self,
        task: Dict[str, Any],
        available_strategies: List[str],
        constraints: Dict[str, Any] = None
    ) -> str:
        """Select best strategy for task."""
        return await self.meta_reasoner.select_reasoning_strategy(
            task,
            available_strategies,
            constraints or {}
        )

    def get_performance_trends(
        self,
        metric: PerformanceMetric,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance trends."""
        return self.self_monitor.get_performance_trends(metric, time_window)

    def get_error_patterns(self) -> Dict[str, Any]:
        """Get error patterns from performance history."""
        return self.self_monitor.get_error_patterns()

from collections import Counter