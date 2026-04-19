# -*- coding: utf-8 -*-
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from .explanation_schemas import CausalEdge, RootCauseHypothesis, CausalExplanation, ConfidenceBounds, PredictionType

logger = logging.getLogger(__name__)
DEFAULT_RANDOM_SEED = 42


@dataclass
class CausalAnalyzerConfig:
    random_seed: int = DEFAULT_RANDOM_SEED
    min_effect_threshold: float = 0.05
    confidence_level: float = 0.95
    max_hypotheses: int = 10
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


class CausalGraph:
    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required")
        self.graph = nx.DiGraph()
        self._default_pm_graph()
        if edges:
            self.graph.add_edges_from(edges)

    def _default_pm_graph(self):
        edges = [("load", "vibration"), ("load", "temperature"), ("load", "failure"),
                 ("speed", "vibration"), ("speed", "temperature"), ("speed", "failure"),
                 ("temperature", "vibration"), ("temperature", "failure"),
                 ("vibration", "failure"), ("age", "failure")]
        self.graph.add_edges_from(edges)

    def get_parents(self, node: str) -> List[str]:
        return list(self.graph.predecessors(node))

    def get_ancestors(self, node: str) -> Set[str]:
        return nx.ancestors(self.graph, node)

    def get_descendants(self, node: str) -> Set[str]:
        return nx.descendants(self.graph, node)

    def get_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        parents = set(self.get_parents(treatment))
        try:
            return parents - {outcome} - self.get_descendants(treatment)
        except:
            return parents - {outcome}

    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        return []

    def to_edges(self) -> List[CausalEdge]:
        return [CausalEdge(source=s, target=t, weight=1.0, confidence=0.9, is_direct=True)
                for s, t in self.graph.edges()]


class CausalAnalyzer:
    def __init__(self, config: Optional[CausalAnalyzerConfig] = None, causal_graph: Optional[CausalGraph] = None):
        self.config = config or CausalAnalyzerConfig()
        self.causal_graph = causal_graph or CausalGraph()
        self._cache = {}
        np.random.seed(self.config.random_seed)

    def identify_confounders(self, treatment: str, outcome: str = "failure") -> List[str]:
        parents = set(self.causal_graph.get_parents(treatment))
        try:
            return list(parents & self.causal_graph.get_ancestors(outcome))
        except:
            return list(parents)

    def compute_backdoor_adjustment(self, treatment: str, outcome: str, data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        if treatment not in data or outcome not in data:
            return 0.0, 1.0
        X, Y = data[treatment], data[outcome]
        if len(X) < 2:
            return 0.0, 1.0
        corr = np.corrcoef(X, Y)[0, 1]
        return (float(corr) if not np.isnan(corr) else 0.0, 1.0 / np.sqrt(len(X)))

    def generate_root_cause_hypotheses(self, data: Dict[str, np.ndarray], outcome: str = "failure") -> List[RootCauseHypothesis]:
        hypotheses = []
        for cause in [n for n in self.causal_graph.graph.nodes() if n != outcome]:
            effect, se = self.compute_backdoor_adjustment(cause, outcome, data)
            if abs(effect) < self.config.min_effect_threshold:
                continue
            ci = ConfidenceBounds(lower_bound=effect - 1.96 * se, upper_bound=effect + 1.96 * se,
                                  confidence_level=0.95, method="backdoor")
            hypotheses.append(RootCauseHypothesis(
                hypothesis_id=hashlib.sha256(f"{cause}{effect}".encode()).hexdigest()[:16],
                cause_variable=cause, effect_variable=outcome, causal_effect=effect,
                uncertainty=se, confidence_interval=ci, confounders_adjusted=self.identify_confounders(cause, outcome),
                backdoor_paths_blocked=0, evidence_strength=min(1.0, abs(effect) / (se + 0.01)), rank=0
            ))
        hypotheses.sort(key=lambda h: abs(h.causal_effect), reverse=True)
        for i, h in enumerate(hypotheses):
            h.rank = i + 1
        return hypotheses[:self.config.max_hypotheses]

    def analyze(self, data: Dict[str, np.ndarray], prediction_value: float,
                prediction_type: PredictionType, outcome: str = "failure") -> CausalExplanation:
        start_time = time.time()
        hypotheses = self.generate_root_cause_hypotheses(data, outcome)
        return CausalExplanation(
            explanation_id=hashlib.sha256(f"{start_time}".encode()).hexdigest()[:16],
            prediction_type=prediction_type, prediction_value=prediction_value,
            causal_graph_edges=self.causal_graph.to_edges(),
            root_cause_hypotheses=hypotheses, confounders_identified=[],
            adjustment_set=[], total_effect=sum(h.causal_effect for h in hypotheses),
            direct_effect=sum(h.causal_effect for h in hypotheses), indirect_effect=0.0,
            timestamp=datetime.utcnow(), computation_time_ms=(time.time() - start_time) * 1000
        )

    def clear_cache(self) -> None:
        self._cache.clear()


def identify_confounders(treatment: str, outcome: str, graph: Optional[CausalGraph] = None) -> List[str]:
    return CausalAnalyzer(causal_graph=graph).identify_confounders(treatment, outcome)


def compute_backdoor_adjustment(treatment: str, outcome: str, data: Dict[str, np.ndarray],
                                graph: Optional[CausalGraph] = None) -> Tuple[float, float]:
    return CausalAnalyzer(causal_graph=graph).compute_backdoor_adjustment(treatment, outcome, data)


def rank_root_causes(data: Dict[str, np.ndarray], outcome: str = "failure",
                     graph: Optional[CausalGraph] = None) -> List[RootCauseHypothesis]:
    return CausalAnalyzer(causal_graph=graph).generate_root_cause_hypotheses(data, outcome)
