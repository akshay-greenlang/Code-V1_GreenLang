# -*- coding: utf-8 -*-
"""
Intelligent Agent Routing System
Routes requests to optimal agents based on confidence and performance
"""

import asyncio
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
from sklearn.ensemble import RandomForestClassifier
import joblib
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random

class RoutingStrategy(Enum):
    """Routing strategies for different scenarios"""
    CONFIDENCE_BASED = "confidence_based"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_FIRST = "performance_first"
    ROUND_ROBIN = "round_robin"
    ENSEMBLE = "ensemble"
    FAILOVER = "failover"

@dataclass
class AgentProfile:
    """Profile of an agent's capabilities"""
    agent_id: str
    agent_type: str
    capabilities: Set[str]
    performance_metrics: Dict[str, float]
    cost_per_request: float
    availability: float
    specializations: List[str]
    confidence_thresholds: Dict[str, float]
    historical_accuracy: float
    average_latency_ms: int
    max_concurrent_requests: int
    current_load: int = 0

@dataclass
class RoutingRequest:
    """Request for agent routing"""
    request_id: str
    task_type: str
    priority: int
    deadline: Optional[datetime]
    required_confidence: float
    max_cost: Optional[float]
    prefer_ensemble: bool
    context: Dict[str, Any]
    retry_count: int = 0

@dataclass
class RoutingDecision:
    """Routing decision with reasoning"""
    request_id: str
    selected_agents: List[str]
    strategy_used: RoutingStrategy
    confidence_score: float
    estimated_cost: float
    estimated_latency_ms: int
    reasoning: str
    fallback_agents: List[str]
    ensemble_weights: Optional[Dict[str, float]] = None

class IntelligentRouter:
    """
    Main router for intelligent agent selection
    """

    def __init__(self):
        self.agents = {}
        self.routing_history = defaultdict(list)
        self.performance_tracker = PerformanceTracker()
        self.confidence_calculator = ConfidenceCalculator()
        self.ensemble_manager = EnsembleManager()
        self.fallback_manager = FallbackManager()
        self.ml_predictor = MLPredictor()

    def register_agent(self, profile: AgentProfile):
        """Register an agent with the router"""
        self.agents[profile.agent_id] = profile
        self.performance_tracker.initialize_agent(profile.agent_id)

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Route request to optimal agent(s)"""

        # Step 1: Filter eligible agents
        eligible_agents = self._filter_eligible_agents(request)

        if not eligible_agents:
            return self._handle_no_agents(request)

        # Step 2: Select routing strategy
        strategy = self._select_strategy(request, eligible_agents)

        # Step 3: Apply routing strategy
        if strategy == RoutingStrategy.ENSEMBLE:
            return await self._route_ensemble(request, eligible_agents)
        elif strategy == RoutingStrategy.CONFIDENCE_BASED:
            return await self._route_by_confidence(request, eligible_agents)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return await self._route_by_cost(request, eligible_agents)
        elif strategy == RoutingStrategy.PERFORMANCE_FIRST:
            return await self._route_by_performance(request, eligible_agents)
        else:
            return await self._route_round_robin(request, eligible_agents)

    def _filter_eligible_agents(self, request: RoutingRequest) -> List[AgentProfile]:
        """Filter agents that can handle the request"""

        eligible = []

        for agent in self.agents.values():
            # Check capability
            if request.task_type not in agent.capabilities:
                continue

            # Check availability
            if agent.availability < 0.5:
                continue

            # Check load
            if agent.current_load >= agent.max_concurrent_requests:
                continue

            # Check cost constraint
            if request.max_cost and agent.cost_per_request > request.max_cost:
                continue

            # Check confidence threshold
            if agent.confidence_thresholds.get(request.task_type, 0) < request.required_confidence:
                continue

            eligible.append(agent)

        return eligible

    def _select_strategy(self, request: RoutingRequest, eligible_agents: List[AgentProfile]) -> RoutingStrategy:
        """Select optimal routing strategy"""

        # Use ensemble for critical requests
        if request.priority == 1 or request.prefer_ensemble:
            return RoutingStrategy.ENSEMBLE

        # Use ML prediction for complex routing
        if len(eligible_agents) > 5:
            predicted_strategy = self.ml_predictor.predict_strategy(request, eligible_agents)
            if predicted_strategy:
                return predicted_strategy

        # Cost optimization for low-priority
        if request.priority > 3 and request.max_cost:
            return RoutingStrategy.COST_OPTIMIZED

        # Performance for time-sensitive
        if request.deadline and (request.deadline - DeterministicClock.utcnow()).total_seconds() < 10:
            return RoutingStrategy.PERFORMANCE_FIRST

        # Default to confidence-based
        return RoutingStrategy.CONFIDENCE_BASED

    async def _route_by_confidence(self, request: RoutingRequest, agents: List[AgentProfile]) -> RoutingDecision:
        """Route based on confidence scores"""

        # Calculate confidence for each agent
        agent_scores = []
        for agent in agents:
            confidence = self.confidence_calculator.calculate(
                agent,
                request.task_type,
                request.context
            )
            agent_scores.append((agent, confidence))

        # Sort by confidence
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top agent
        selected_agent = agent_scores[0][0]
        confidence = agent_scores[0][1]

        # Get fallbacks
        fallbacks = [agent.agent_id for agent, _ in agent_scores[1:4]]

        return RoutingDecision(
            request_id=request.request_id,
            selected_agents=[selected_agent.agent_id],
            strategy_used=RoutingStrategy.CONFIDENCE_BASED,
            confidence_score=confidence,
            estimated_cost=selected_agent.cost_per_request,
            estimated_latency_ms=selected_agent.average_latency_ms,
            reasoning=f"Selected {selected_agent.agent_id} with confidence {confidence:.2f}",
            fallback_agents=fallbacks
        )

    async def _route_ensemble(self, request: RoutingRequest, agents: List[AgentProfile]) -> RoutingDecision:
        """Route to multiple agents for ensemble processing"""

        # Select diverse agents for ensemble
        ensemble_agents = self.ensemble_manager.select_ensemble(
            agents,
            request.task_type,
            target_size=3
        )

        # Calculate ensemble weights
        weights = self.ensemble_manager.calculate_weights(
            ensemble_agents,
            request.task_type
        )

        # Calculate combined metrics
        total_cost = sum(agent.cost_per_request for agent in ensemble_agents)
        max_latency = max(agent.average_latency_ms for agent in ensemble_agents)
        ensemble_confidence = self.ensemble_manager.calculate_ensemble_confidence(
            ensemble_agents,
            weights
        )

        return RoutingDecision(
            request_id=request.request_id,
            selected_agents=[agent.agent_id for agent in ensemble_agents],
            strategy_used=RoutingStrategy.ENSEMBLE,
            confidence_score=ensemble_confidence,
            estimated_cost=total_cost,
            estimated_latency_ms=max_latency,
            reasoning=f"Ensemble of {len(ensemble_agents)} agents for high confidence",
            fallback_agents=[],
            ensemble_weights=weights
        )


class ConfidenceCalculator:
    """
    Calculate confidence scores for agent-task combinations
    """

    def __init__(self):
        self.historical_performance = defaultdict(lambda: defaultdict(list))

    def calculate(self, agent: AgentProfile, task_type: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score"""

        base_confidence = agent.confidence_thresholds.get(task_type, 0.5)

        # Factor 1: Historical accuracy (40%)
        historical = agent.historical_accuracy
        accuracy_factor = historical * 0.4

        # Factor 2: Specialization match (30%)
        specialization_factor = 0.3 if task_type in agent.specializations else 0.15

        # Factor 3: Current performance (20%)
        recent_performance = self._get_recent_performance(agent.agent_id, task_type)
        performance_factor = recent_performance * 0.2

        # Factor 4: Load factor (10%)
        load_ratio = agent.current_load / max(agent.max_concurrent_requests, 1)
        load_factor = (1 - load_ratio) * 0.1

        # Calculate final confidence
        confidence = min(
            base_confidence + accuracy_factor + specialization_factor +
            performance_factor + load_factor,
            1.0
        )

        return confidence

    def _get_recent_performance(self, agent_id: str, task_type: str) -> float:
        """Get recent performance score"""
        recent = self.historical_performance[agent_id][task_type][-10:]
        if not recent:
            return 0.7  # Default

        return sum(recent) / len(recent)


class EnsembleManager:
    """
    Manage ensemble routing strategies
    """

    def select_ensemble(self, agents: List[AgentProfile], task_type: str, target_size: int = 3) -> List[AgentProfile]:
        """Select diverse agents for ensemble"""

        if len(agents) <= target_size:
            return agents

        # Score agents by diversity and performance
        scored_agents = []
        for agent in agents:
            score = self._calculate_ensemble_score(agent, task_type, agents)
            scored_agents.append((agent, score))

        # Sort and select top agents
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in scored_agents[:target_size]]

    def calculate_weights(self, agents: List[AgentProfile], task_type: str) -> Dict[str, float]:
        """Calculate voting weights for ensemble"""

        weights = {}
        total_score = 0

        for agent in agents:
            # Base weight on historical accuracy and specialization
            score = agent.historical_accuracy
            if task_type in agent.specializations:
                score *= 1.5

            weights[agent.agent_id] = score
            total_score += score

        # Normalize weights
        if total_score > 0:
            for agent_id in weights:
                weights[agent_id] /= total_score

        return weights

    def calculate_ensemble_confidence(self, agents: List[AgentProfile], weights: Dict[str, float]) -> float:
        """Calculate combined confidence for ensemble"""

        weighted_confidence = 0
        for agent in agents:
            agent_confidence = agent.historical_accuracy
            weight = weights.get(agent.agent_id, 0)
            weighted_confidence += agent_confidence * weight

        # Boost for ensemble diversity
        diversity_bonus = min(len(agents) * 0.05, 0.15)

        return min(weighted_confidence + diversity_bonus, 0.99)

    def _calculate_ensemble_score(self, agent: AgentProfile, task_type: str, all_agents: List[AgentProfile]) -> float:
        """Calculate score for ensemble selection"""

        # Performance score
        performance_score = agent.historical_accuracy * 0.5

        # Specialization score
        spec_score = 0.3 if task_type in agent.specializations else 0.1

        # Diversity score (different from other agents)
        diversity_score = self._calculate_diversity(agent, all_agents) * 0.2

        return performance_score + spec_score + diversity_score

    def _calculate_diversity(self, agent: AgentProfile, others: List[AgentProfile]) -> float:
        """Calculate diversity of agent from others"""

        if len(others) <= 1:
            return 1.0

        # Calculate based on capability differences
        unique_capabilities = agent.capabilities
        for other in others:
            if other.agent_id != agent.agent_id:
                unique_capabilities -= other.capabilities

        diversity = len(unique_capabilities) / max(len(agent.capabilities), 1)
        return min(diversity, 1.0)


class FallbackManager:
    """
    Manage fallback strategies for failed requests
    """

    def __init__(self):
        self.failure_history = defaultdict(list)
        self.fallback_chains = {}

    def create_fallback_chain(self, primary_agent: str, task_type: str, eligible_agents: List[AgentProfile]) -> List[str]:
        """Create fallback chain for request"""

        # Remove primary from candidates
        candidates = [a for a in eligible_agents if a.agent_id != primary_agent]

        if not candidates:
            return []

        # Sort by reliability and performance
        scored = []
        for agent in candidates:
            score = self._calculate_fallback_score(agent, task_type)
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 as fallback chain
        return [agent.agent_id for agent, _ in scored[:3]]

    def _calculate_fallback_score(self, agent: AgentProfile, task_type: str) -> float:
        """Calculate fallback suitability score"""

        # Prioritize reliability over performance for fallbacks
        reliability_score = agent.availability * 0.5
        accuracy_score = agent.historical_accuracy * 0.3
        capacity_score = (1 - agent.current_load / agent.max_concurrent_requests) * 0.2

        return reliability_score + accuracy_score + capacity_score

    def record_failure(self, agent_id: str, task_type: str, error: str):
        """Record agent failure for analysis"""

        self.failure_history[agent_id].append({
            'task_type': task_type,
            'error': error,
            'timestamp': DeterministicClock.utcnow()
        })

        # Update agent reliability if too many failures
        recent_failures = [
            f for f in self.failure_history[agent_id]
            if f['timestamp'] > DeterministicClock.utcnow() - timedelta(hours=1)
        ]

        if len(recent_failures) > 5:
            # Trigger reliability downgrade
            self._downgrade_reliability(agent_id)

    def _downgrade_reliability(self, agent_id: str):
        """Downgrade agent reliability score"""
        # Implementation would update agent profile
        pass


class PerformanceTracker:
    """
    Track and analyze agent performance
    """

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.aggregates = {}

    def initialize_agent(self, agent_id: str):
        """Initialize tracking for new agent"""
        self.metrics[agent_id] = {
            'latency': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'failures': deque(maxlen=100),
            'load': deque(maxlen=100)
        }

    def record_request(self, agent_id: str, latency_ms: int, success: bool, accuracy: Optional[float] = None):
        """Record request metrics"""

        self.metrics[agent_id]['latency'].append(latency_ms)
        self.metrics[agent_id]['failures'].append(0 if success else 1)

        if accuracy is not None:
            self.metrics[agent_id]['accuracy'].append(accuracy)

    def get_agent_stats(self, agent_id: str) -> Dict[str, float]:
        """Get aggregated stats for agent"""

        metrics = self.metrics[agent_id]

        return {
            'avg_latency_ms': np.mean(metrics['latency']) if metrics['latency'] else 0,
            'p95_latency_ms': np.percentile(metrics['latency'], 95) if metrics['latency'] else 0,
            'success_rate': 1 - (sum(metrics['failures']) / max(len(metrics['failures']), 1)),
            'avg_accuracy': np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
        }

    def get_comparative_analysis(self, task_type: str) -> Dict[str, Any]:
        """Compare agent performance for task type"""

        comparison = {}

        for agent_id, metrics in self.metrics.items():
            stats = self.get_agent_stats(agent_id)
            comparison[agent_id] = {
                'performance_score': self._calculate_performance_score(stats),
                'stats': stats
            }

        # Rank agents
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1]['performance_score'],
            reverse=True
        )

        return {
            'rankings': ranked,
            'best_performer': ranked[0][0] if ranked else None,
            'comparison': comparison
        }

    def _calculate_performance_score(self, stats: Dict[str, float]) -> float:
        """Calculate overall performance score"""

        # Weighted scoring
        latency_score = max(0, 1 - (stats['avg_latency_ms'] / 5000)) * 0.3
        success_score = stats['success_rate'] * 0.4
        accuracy_score = stats['avg_accuracy'] * 0.3

        return latency_score + success_score + accuracy_score


class MLPredictor:
    """
    Machine learning based routing prediction
    """

    def __init__(self):
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.training_data = []

    def predict_strategy(self, request: RoutingRequest, agents: List[AgentProfile]) -> Optional[RoutingStrategy]:
        """Predict optimal routing strategy using ML"""

        if not self.model:
            return None

        # Extract features
        features = self.feature_extractor.extract(request, agents)

        # Predict
        try:
            prediction = self.model.predict([features])[0]
            return RoutingStrategy(prediction)
        except:
            return None

    def train_model(self, historical_data: List[Dict[str, Any]]):
        """Train ML model on historical routing data"""

        if len(historical_data) < 100:
            return  # Not enough data

        # Prepare training data
        X = []
        y = []

        for record in historical_data:
            features = self.feature_extractor.extract(
                record['request'],
                record['agents']
            )
            X.append(features)
            y.append(record['strategy_used'].value)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.model.fit(X, y)

        # Save model
        joblib.dump(self.model, 'routing_model.pkl')

    def update_model(self, request: RoutingRequest, decision: RoutingDecision, outcome: Dict[str, Any]):
        """Update model with new routing outcome"""

        self.training_data.append({
            'request': request,
            'decision': decision,
            'outcome': outcome,
            'timestamp': DeterministicClock.utcnow()
        })

        # Retrain periodically
        if len(self.training_data) % 100 == 0:
            self.train_model(self.training_data)


class FeatureExtractor:
    """
    Extract features for ML routing
    """

    def extract(self, request: RoutingRequest, agents: List[AgentProfile]) -> List[float]:
        """Extract numerical features"""

        features = []

        # Request features
        features.append(float(request.priority))
        features.append(request.required_confidence)
        features.append(request.retry_count)
        features.append(1.0 if request.prefer_ensemble else 0.0)

        # Agent pool features
        features.append(len(agents))
        features.append(np.mean([a.historical_accuracy for a in agents]))
        features.append(np.mean([a.cost_per_request for a in agents]))
        features.append(np.mean([a.average_latency_ms for a in agents]))
        features.append(np.mean([a.availability for a in agents]))

        # Capacity features
        total_capacity = sum(a.max_concurrent_requests for a in agents)
        total_load = sum(a.current_load for a in agents)
        features.append(total_load / max(total_capacity, 1))

        # Specialization match
        specialized = sum(1 for a in agents if request.task_type in a.specializations)
        features.append(specialized / max(len(agents), 1))

        return features


class ABTestingFramework:
    """
    A/B testing for routing strategies
    """

    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(list)

    def create_experiment(self, name: str, strategies: List[RoutingStrategy], split: List[float]):
        """Create A/B test experiment"""

        self.experiments[name] = {
            'strategies': strategies,
            'split': split,
            'created': DeterministicClock.utcnow(),
            'request_count': 0
        }

    def select_variant(self, experiment_name: str) -> RoutingStrategy:
        """Select variant for request"""

        if experiment_name not in self.experiments:
            return RoutingStrategy.CONFIDENCE_BASED

        experiment = self.experiments[experiment_name]
        rand = np.deterministic_random().random()

        cumulative = 0
        for i, split in enumerate(experiment['split']):
            cumulative += split
            if rand < cumulative:
                return experiment['strategies'][i]

        return experiment['strategies'][-1]

    def record_result(self, experiment_name: str, strategy: RoutingStrategy, metrics: Dict[str, Any]):
        """Record experiment result"""

        self.results[experiment_name].append({
            'strategy': strategy,
            'metrics': metrics,
            'timestamp': DeterministicClock.utcnow()
        })

    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze experiment results"""

        if experiment_name not in self.results:
            return {}

        results = self.results[experiment_name]

        # Group by strategy
        strategy_metrics = defaultdict(list)
        for result in results:
            strategy_metrics[result['strategy']].append(result['metrics'])

        # Calculate statistics
        analysis = {}
        for strategy, metrics_list in strategy_metrics.items():
            analysis[strategy.value] = {
                'sample_size': len(metrics_list),
                'avg_latency': np.mean([m['latency'] for m in metrics_list]),
                'avg_accuracy': np.mean([m.get('accuracy', 0) for m in metrics_list]),
                'success_rate': sum(1 for m in metrics_list if m.get('success', False)) / len(metrics_list),
                'avg_cost': np.mean([m.get('cost', 0) for m in metrics_list])
            }

        # Determine winner
        best_strategy = max(
            analysis.items(),
            key=lambda x: x[1]['success_rate'] * 0.5 + x[1]['avg_accuracy'] * 0.5
        )

        return {
            'analysis': analysis,
            'winner': best_strategy[0],
            'confidence': self._calculate_statistical_significance(strategy_metrics)
        }

    def _calculate_statistical_significance(self, strategy_metrics: Dict) -> float:
        """Calculate statistical significance of results"""
        # Simplified - would use proper statistical tests
        if len(strategy_metrics) < 2:
            return 0.0

        sample_sizes = [len(metrics) for metrics in strategy_metrics.values()]
        if min(sample_sizes) < 30:
            return 0.5  # Not enough data

        return 0.95  # Placeholder


# Export main components
__all__ = [
    'IntelligentRouter',
    'RoutingStrategy',
    'RoutingRequest',
    'RoutingDecision',
    'AgentProfile',
    'PerformanceTracker',
    'EnsembleManager',
    'ABTestingFramework'
]