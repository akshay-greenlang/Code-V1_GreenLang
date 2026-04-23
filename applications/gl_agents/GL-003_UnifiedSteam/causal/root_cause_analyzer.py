"""
GL-003 UNIFIEDSTEAM - Root Cause Analyzer

Performs root cause analysis using causal graphs:
- Deviation identification (metric vs baseline)
- Cause ranking with probability scores
- Evidence aggregation
- RCA report generation

Answers: "What change most likely caused the deviation?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import uuid
import math

from .causal_graph import CausalGraph, CausalNode, CausalEdge, NodeType

logger = logging.getLogger(__name__)


class DeviationType(Enum):
    """Types of deviations."""
    HIGH = "high"  # Above normal range
    LOW = "low"  # Below normal range
    RAPID_CHANGE = "rapid_change"  # Fast change rate
    OSCILLATION = "oscillation"  # Unstable behavior
    STUCK = "stuck"  # No expected change
    TREND = "trend"  # Gradual drift


class EvidenceStrength(Enum):
    """Strength of supporting evidence."""
    STRONG = "strong"  # Clear correlation
    MODERATE = "moderate"  # Some correlation
    WEAK = "weak"  # Minor correlation
    CIRCUMSTANTIAL = "circumstantial"  # Possible but unclear


@dataclass
class Deviation:
    """Represents a deviation from normal operation."""
    deviation_id: str
    metric_name: str
    metric_node_id: str
    deviation_type: DeviationType

    # Values
    baseline_value: float
    current_value: float
    deviation_magnitude: float  # Absolute difference
    deviation_percent: float  # Percentage difference

    # Time
    detected_at: datetime
    duration_seconds: Optional[float] = None

    # Context
    unit: str = ""
    normal_range: Optional[Tuple[float, float]] = None
    threshold_used: Optional[float] = None

    # Severity
    severity: str = "medium"  # "low", "medium", "high", "critical"

    def to_dict(self) -> Dict:
        return {
            "deviation_id": self.deviation_id,
            "metric_name": self.metric_name,
            "metric_node_id": self.metric_node_id,
            "deviation_type": self.deviation_type.value,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "deviation_magnitude": self.deviation_magnitude,
            "deviation_percent": self.deviation_percent,
            "detected_at": self.detected_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "unit": self.unit,
            "normal_range": list(self.normal_range) if self.normal_range else None,
            "threshold_used": self.threshold_used,
            "severity": self.severity,
        }


@dataclass
class CauseEvidence:
    """Evidence supporting a potential cause."""
    evidence_id: str
    evidence_type: str  # "temporal", "statistical", "physical", "historical"
    strength: EvidenceStrength

    # Description
    description: str
    observation: str

    # Quantification
    correlation: Optional[float] = None  # -1 to 1
    time_lag_seconds: Optional[float] = None
    confidence: float = 0.5

    # Source
    source_signal: str = ""
    source_value: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "strength": self.strength.value,
            "description": self.description,
            "observation": self.observation,
            "correlation": self.correlation,
            "time_lag_seconds": self.time_lag_seconds,
            "confidence": self.confidence,
            "source_signal": self.source_signal,
            "source_value": self.source_value,
        }


@dataclass
class RankedCause:
    """A potential cause with probability and evidence."""
    cause_id: str
    cause_node_id: str
    cause_name: str
    cause_description: str

    # Ranking
    probability: float  # 0-1, probability this is the root cause
    rank: int  # 1 = most likely

    # Path
    causal_path: List[str]  # Node IDs from cause to effect
    path_length: int

    # Evidence
    evidence: List[CauseEvidence]
    evidence_summary: str

    # Impact
    estimated_impact: float  # How much this cause contributed
    direction: str  # "positive", "negative"

    # Actionability
    is_actionable: bool = True
    suggested_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "cause_id": self.cause_id,
            "cause_node_id": self.cause_node_id,
            "cause_name": self.cause_name,
            "cause_description": self.cause_description,
            "probability": self.probability,
            "rank": self.rank,
            "causal_path": self.causal_path,
            "path_length": self.path_length,
            "evidence": [e.to_dict() for e in self.evidence],
            "evidence_summary": self.evidence_summary,
            "estimated_impact": self.estimated_impact,
            "direction": self.direction,
            "is_actionable": self.is_actionable,
            "suggested_action": self.suggested_action,
        }


@dataclass
class RankedCauses:
    """Collection of ranked causes for a deviation."""
    analysis_id: str
    deviation_id: str
    timestamp: datetime

    # Causes
    causes: List[RankedCause]
    total_causes_considered: int

    # Top cause
    most_likely_cause: Optional[RankedCause] = None
    top_cause_confidence: float = 0.0

    # Alternative explanations
    alternative_explanations: List[str] = field(default_factory=list)

    # Uncertainty
    uncertainty_note: str = ""

    def to_dict(self) -> Dict:
        return {
            "analysis_id": self.analysis_id,
            "deviation_id": self.deviation_id,
            "timestamp": self.timestamp.isoformat(),
            "causes": [c.to_dict() for c in self.causes],
            "total_causes_considered": self.total_causes_considered,
            "most_likely_cause": self.most_likely_cause.to_dict() if self.most_likely_cause else None,
            "top_cause_confidence": self.top_cause_confidence,
            "alternative_explanations": self.alternative_explanations,
            "uncertainty_note": self.uncertainty_note,
        }


@dataclass
class RCAReport:
    """Complete root cause analysis report."""
    report_id: str
    timestamp: datetime
    analyst: str  # "GL-003"

    # Deviation
    deviation: Deviation

    # Analysis
    ranked_causes: RankedCauses

    # Summary
    executive_summary: str
    technical_summary: str
    recommended_actions: List[str]

    # Confidence
    overall_confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)

    # Audit
    analysis_duration_ms: int = 0
    nodes_examined: int = 0
    evidence_points: int = 0

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "analyst": self.analyst,
            "deviation": self.deviation.to_dict(),
            "ranked_causes": self.ranked_causes.to_dict(),
            "executive_summary": self.executive_summary,
            "technical_summary": self.technical_summary,
            "recommended_actions": self.recommended_actions,
            "overall_confidence": self.overall_confidence,
            "limitations": self.limitations,
            "analysis_duration_ms": self.analysis_duration_ms,
            "nodes_examined": self.nodes_examined,
            "evidence_points": self.evidence_points,
        }


class RootCauseAnalyzer:
    """
    Performs root cause analysis using causal graphs.

    Features:
    - Deviation detection and characterization
    - Causal path traversal
    - Evidence aggregation from multiple sources
    - Probability-ranked cause identification
    - Report generation
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        agent_id: str = "GL-003",
    ) -> None:
        self.graph = causal_graph
        self.agent_id = agent_id

        # Analysis parameters
        self.max_path_depth = 5
        self.min_probability_threshold = 0.05
        self.evidence_weight_physical = 0.4
        self.evidence_weight_statistical = 0.3
        self.evidence_weight_temporal = 0.2
        self.evidence_weight_historical = 0.1

        # Cause-action mapping
        self._action_mapping = self._initialize_action_mapping()

        # Cached analyses
        self._analyses: Dict[str, RankedCauses] = {}
        self._reports: Dict[str, RCAReport] = {}

        logger.info(f"RootCauseAnalyzer initialized with graph: {causal_graph.graph_id}")

    def _initialize_action_mapping(self) -> Dict[str, str]:
        """Initialize mapping from cause types to suggested actions."""
        return {
            # Equipment causes
            "prv": "Check PRV operation and setpoint",
            "desuperheater": "Verify spray water valve position and temperature",
            "boiler": "Review boiler firing rate and steam output",
            "trap": "Inspect steam trap with ultrasonic tester",
            "valve": "Check valve position and actuator",
            "pump": "Verify pump operation and flow rate",

            # Process variable causes
            "pressure": "Review pressure control loop and setpoint",
            "temperature": "Check temperature control and measurements",
            "flow": "Verify flow measurements and control valves",
            "level": "Check level control and transmitter",

            # External causes
            "ambient": "Monitor for ambient temperature effects",
            "demand": "Coordinate with production on steam demand",
            "fuel": "Verify fuel supply pressure and quality",
        }

    def identify_deviation(
        self,
        metric: str,
        baseline: float,
        current: float,
        node_id: Optional[str] = None,
        unit: str = "",
        normal_range: Optional[Tuple[float, float]] = None,
        threshold_percent: float = 10.0,
    ) -> Deviation:
        """
        Identify and characterize a deviation.

        Args:
            metric: Name of the metric
            baseline: Expected/normal value
            current: Current observed value
            node_id: Corresponding node ID in causal graph
            unit: Unit of measurement
            normal_range: Normal operating range
            threshold_percent: Threshold for deviation detection

        Returns:
            Deviation object
        """
        deviation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Calculate deviation
        magnitude = abs(current - baseline)
        if baseline != 0:
            percent = (current - baseline) / baseline * 100
        else:
            percent = 100.0 if current != 0 else 0.0

        # Determine deviation type
        if percent > 0:
            deviation_type = DeviationType.HIGH
        elif percent < 0:
            deviation_type = DeviationType.LOW
        else:
            deviation_type = DeviationType.STUCK

        # Determine severity
        abs_percent = abs(percent)
        if abs_percent > 50:
            severity = "critical"
        elif abs_percent > 25:
            severity = "high"
        elif abs_percent > threshold_percent:
            severity = "medium"
        else:
            severity = "low"

        deviation = Deviation(
            deviation_id=deviation_id,
            metric_name=metric,
            metric_node_id=node_id or metric,
            deviation_type=deviation_type,
            baseline_value=baseline,
            current_value=current,
            deviation_magnitude=magnitude,
            deviation_percent=percent,
            detected_at=timestamp,
            unit=unit,
            normal_range=normal_range,
            threshold_used=threshold_percent,
            severity=severity,
        )

        logger.info(
            f"Identified deviation: {metric} = {current} (baseline: {baseline}, "
            f"{percent:+.1f}%, severity: {severity})"
        )

        return deviation

    def rank_root_causes(
        self,
        deviation: Deviation,
        causal_graph: Optional[CausalGraph] = None,
        evidence: Optional[Dict[str, Any]] = None,
        max_causes: int = 10,
    ) -> RankedCauses:
        """
        Rank potential root causes for a deviation.

        Args:
            deviation: The deviation to analyze
            causal_graph: Optional override for causal graph
            evidence: Additional evidence (signal values, correlations)
            max_causes: Maximum number of causes to return

        Returns:
            RankedCauses with probability-ranked causes
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        graph = causal_graph or self.graph
        evidence = evidence or {}

        # Find the deviation node in the graph
        metric_node = graph.get_node(deviation.metric_node_id)
        if not metric_node:
            logger.warning(f"Metric node {deviation.metric_node_id} not found in graph")
            # Create a minimal response
            return RankedCauses(
                analysis_id=analysis_id,
                deviation_id=deviation.deviation_id,
                timestamp=timestamp,
                causes=[],
                total_causes_considered=0,
                uncertainty_note="Metric not found in causal graph",
            )

        # Get all potential causes (ancestors)
        ancestors = graph.get_ancestors(deviation.metric_node_id, max_depth=self.max_path_depth)

        # Score each potential cause
        scored_causes = []
        for ancestor in ancestors:
            # Get causal path
            paths = graph.get_path(ancestor.node_id, deviation.metric_node_id)
            if not paths:
                continue

            shortest_path = min(paths, key=len)

            # Gather evidence for this cause
            cause_evidence = self._gather_evidence(
                ancestor, deviation, evidence, graph
            )

            # Compute probability
            probability = self.compute_cause_probability(
                ancestor.node_id, cause_evidence, evidence
            )

            if probability < self.min_probability_threshold:
                continue

            # Determine direction
            direction = self._determine_direction(
                ancestor, deviation, graph, shortest_path
            )

            # Get suggested action
            suggested_action = self._get_suggested_action(ancestor)

            scored_causes.append({
                "node": ancestor,
                "probability": probability,
                "path": shortest_path,
                "evidence": cause_evidence,
                "direction": direction,
                "action": suggested_action,
            })

        # Sort by probability
        scored_causes.sort(key=lambda x: x["probability"], reverse=True)

        # Build RankedCause objects
        ranked_causes = []
        for rank, cause_data in enumerate(scored_causes[:max_causes], 1):
            node = cause_data["node"]
            ranked_cause = RankedCause(
                cause_id=str(uuid.uuid4())[:8],
                cause_node_id=node.node_id,
                cause_name=node.name,
                cause_description=node.description or f"Change in {node.name}",
                probability=cause_data["probability"],
                rank=rank,
                causal_path=cause_data["path"],
                path_length=len(cause_data["path"]),
                evidence=cause_data["evidence"],
                evidence_summary=self._summarize_evidence(cause_data["evidence"]),
                estimated_impact=cause_data["probability"] * abs(deviation.deviation_percent),
                direction=cause_data["direction"],
                is_actionable=node.is_controllable,
                suggested_action=cause_data["action"],
            )
            ranked_causes.append(ranked_cause)

        # Identify most likely cause
        most_likely = ranked_causes[0] if ranked_causes else None
        top_confidence = most_likely.probability if most_likely else 0.0

        # Generate alternative explanations
        alternatives = self._generate_alternatives(ranked_causes, deviation)

        # Uncertainty note
        uncertainty = self._generate_uncertainty_note(
            ranked_causes, len(ancestors), top_confidence
        )

        result = RankedCauses(
            analysis_id=analysis_id,
            deviation_id=deviation.deviation_id,
            timestamp=timestamp,
            causes=ranked_causes,
            total_causes_considered=len(ancestors),
            most_likely_cause=most_likely,
            top_cause_confidence=top_confidence,
            alternative_explanations=alternatives,
            uncertainty_note=uncertainty,
        )

        self._analyses[analysis_id] = result
        logger.info(
            f"Ranked {len(ranked_causes)} causes for deviation {deviation.deviation_id}, "
            f"top cause: {most_likely.cause_name if most_likely else 'None'}"
        )

        return result

    def _gather_evidence(
        self,
        cause_node: CausalNode,
        deviation: Deviation,
        evidence_data: Dict[str, Any],
        graph: CausalGraph,
    ) -> List[CauseEvidence]:
        """Gather evidence supporting a potential cause."""
        evidence_list = []

        # Physical evidence (from causal graph structure)
        edge = graph.get_edge_between(cause_node.node_id, deviation.metric_node_id)
        if not edge:
            # Check for indirect path
            paths = graph.get_path(cause_node.node_id, deviation.metric_node_id)
            if paths:
                strength = EvidenceStrength.MODERATE if len(paths[0]) <= 3 else EvidenceStrength.WEAK
                evidence_list.append(CauseEvidence(
                    evidence_id=str(uuid.uuid4())[:8],
                    evidence_type="physical",
                    strength=strength,
                    description=f"Causal path exists with {len(paths[0])} steps",
                    observation=f"Path: {' -> '.join(paths[0])}",
                    confidence=0.7 if strength == EvidenceStrength.MODERATE else 0.4,
                ))
        else:
            evidence_list.append(CauseEvidence(
                evidence_id=str(uuid.uuid4())[:8],
                evidence_type="physical",
                strength=EvidenceStrength.STRONG,
                description="Direct causal relationship in graph",
                observation=f"Direct edge: {edge.relationship_type.value}",
                confidence=edge.confidence,
            ))

        # Statistical evidence (from provided correlations)
        correlation = evidence_data.get(f"correlation_{cause_node.node_id}")
        if correlation is not None:
            strength = (
                EvidenceStrength.STRONG if abs(correlation) > 0.7
                else EvidenceStrength.MODERATE if abs(correlation) > 0.4
                else EvidenceStrength.WEAK
            )
            evidence_list.append(CauseEvidence(
                evidence_id=str(uuid.uuid4())[:8],
                evidence_type="statistical",
                strength=strength,
                description=f"Statistical correlation detected",
                observation=f"Correlation coefficient: {correlation:.3f}",
                correlation=correlation,
                confidence=abs(correlation),
            ))

        # Temporal evidence (from time series)
        time_data = evidence_data.get(f"change_{cause_node.node_id}")
        if time_data:
            change_percent = time_data.get("change_percent", 0)
            if abs(change_percent) > 5:
                strength = EvidenceStrength.MODERATE if abs(change_percent) > 15 else EvidenceStrength.WEAK
                evidence_list.append(CauseEvidence(
                    evidence_id=str(uuid.uuid4())[:8],
                    evidence_type="temporal",
                    strength=strength,
                    description=f"Recent change in {cause_node.name}",
                    observation=f"Changed by {change_percent:+.1f}% recently",
                    time_lag_seconds=time_data.get("lag_seconds", 0),
                    confidence=min(0.8, abs(change_percent) / 30),
                ))

        # Historical evidence
        historical = evidence_data.get(f"historical_{cause_node.node_id}")
        if historical:
            evidence_list.append(CauseEvidence(
                evidence_id=str(uuid.uuid4())[:8],
                evidence_type="historical",
                strength=EvidenceStrength.MODERATE,
                description=f"Similar pattern in history",
                observation=historical.get("description", "Historical match found"),
                confidence=historical.get("confidence", 0.5),
            ))

        return evidence_list

    def compute_cause_probability(
        self,
        cause: str,
        evidence: List[CauseEvidence],
        additional_evidence: Dict[str, Any],
    ) -> float:
        """
        Compute probability that a cause is the root cause.

        Uses weighted evidence aggregation.

        Args:
            cause: Cause node ID
            evidence: List of evidence items
            additional_evidence: Additional evidence dictionary

        Returns:
            Probability (0-1)
        """
        if not evidence:
            return 0.1  # Base probability for any ancestor

        # Weight evidence by type
        type_weights = {
            "physical": self.evidence_weight_physical,
            "statistical": self.evidence_weight_statistical,
            "temporal": self.evidence_weight_temporal,
            "historical": self.evidence_weight_historical,
        }

        # Strength multipliers
        strength_mult = {
            EvidenceStrength.STRONG: 1.0,
            EvidenceStrength.MODERATE: 0.7,
            EvidenceStrength.WEAK: 0.4,
            EvidenceStrength.CIRCUMSTANTIAL: 0.2,
        }

        total_weight = 0
        weighted_confidence = 0

        for ev in evidence:
            weight = type_weights.get(ev.evidence_type, 0.1)
            mult = strength_mult.get(ev.strength, 0.5)
            adjusted_confidence = ev.confidence * mult

            weighted_confidence += weight * adjusted_confidence
            total_weight += weight

        if total_weight > 0:
            base_prob = weighted_confidence / total_weight
        else:
            base_prob = 0.1

        # Apply prior based on node type
        # Controllable nodes are more likely to be root causes
        node = self.graph.get_node(cause)
        if node:
            if node.is_controllable:
                base_prob *= 1.2
            if node.is_exogenous:
                base_prob *= 1.1

        return min(1.0, base_prob)

    def _determine_direction(
        self,
        cause_node: CausalNode,
        deviation: Deviation,
        graph: CausalGraph,
        path: List[str],
    ) -> str:
        """Determine the direction of causal effect."""
        # Check edge directionality along path
        is_positive = True
        for i in range(len(path) - 1):
            edge = graph.get_edge_between(path[i], path[i + 1])
            if edge and not edge.is_positive:
                is_positive = not is_positive

        # Combine with deviation direction
        if deviation.deviation_type == DeviationType.HIGH:
            return "positive" if is_positive else "negative"
        elif deviation.deviation_type == DeviationType.LOW:
            return "negative" if is_positive else "positive"
        else:
            return "neutral"

    def _get_suggested_action(self, node: CausalNode) -> str:
        """Get suggested action for a cause node."""
        # Try to match by node type
        node_type_str = node.node_type.value.lower()
        for key, action in self._action_mapping.items():
            if key in node_type_str:
                return action

        # Try to match by name
        name_lower = node.name.lower()
        for key, action in self._action_mapping.items():
            if key in name_lower:
                return action

        return f"Investigate {node.name}"

    def _summarize_evidence(self, evidence: List[CauseEvidence]) -> str:
        """Generate summary of evidence."""
        if not evidence:
            return "No direct evidence"

        strong = [e for e in evidence if e.strength == EvidenceStrength.STRONG]
        moderate = [e for e in evidence if e.strength == EvidenceStrength.MODERATE]

        parts = []
        if strong:
            parts.append(f"{len(strong)} strong evidence points")
        if moderate:
            parts.append(f"{len(moderate)} moderate evidence points")

        return ", ".join(parts) if parts else "Circumstantial evidence only"

    def _generate_alternatives(
        self,
        ranked_causes: List[RankedCause],
        deviation: Deviation,
    ) -> List[str]:
        """Generate alternative explanations."""
        alternatives = []

        if len(ranked_causes) < 2:
            alternatives.append("Limited alternative causes identified")
            return alternatives

        # Top alternative
        if ranked_causes[1].probability > 0.2:
            alt = ranked_causes[1]
            alternatives.append(
                f"Alternative: {alt.cause_name} ({alt.probability:.0%} probability)"
            )

        # Check for multiple similar probability causes
        top_prob = ranked_causes[0].probability
        similar = [c for c in ranked_causes[1:] if c.probability > top_prob * 0.8]
        if similar:
            alternatives.append(
                f"Multiple causes with similar likelihood ({len(similar) + 1} total)"
            )

        # Check for external factors
        external = [c for c in ranked_causes if self.graph.get_node(c.cause_node_id) and
                   self.graph.get_node(c.cause_node_id).is_exogenous]
        if external:
            alternatives.append(
                f"External factor possible: {external[0].cause_name}"
            )

        return alternatives

    def _generate_uncertainty_note(
        self,
        ranked_causes: List[RankedCause],
        total_considered: int,
        top_confidence: float,
    ) -> str:
        """Generate uncertainty note for the analysis."""
        if top_confidence > 0.8:
            return "High confidence in root cause identification"
        elif top_confidence > 0.5:
            return "Moderate confidence; verification recommended"
        elif top_confidence > 0.3:
            if len(ranked_causes) > 3:
                return "Multiple possible causes; additional investigation needed"
            return "Low confidence; limited evidence available"
        else:
            return "Very low confidence; root cause unclear from available data"

    def generate_rca_report(
        self,
        analysis_result: RankedCauses,
        deviation: Optional[Deviation] = None,
    ) -> RCAReport:
        """
        Generate complete RCA report.

        Args:
            analysis_result: Results from rank_root_causes
            deviation: Original deviation (looked up if not provided)

        Returns:
            Complete RCAReport
        """
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Get deviation from ID if not provided
        if deviation is None:
            # Would look up from stored deviations
            deviation = Deviation(
                deviation_id=analysis_result.deviation_id,
                metric_name="Unknown",
                metric_node_id="unknown",
                deviation_type=DeviationType.HIGH,
                baseline_value=0,
                current_value=0,
                deviation_magnitude=0,
                deviation_percent=0,
                detected_at=timestamp,
            )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            deviation, analysis_result
        )

        # Generate technical summary
        technical_summary = self._generate_technical_summary(
            deviation, analysis_result
        )

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(analysis_result)

        # Calculate overall confidence
        overall_confidence = analysis_result.top_cause_confidence

        # Identify limitations
        limitations = self._identify_limitations(analysis_result)

        report = RCAReport(
            report_id=report_id,
            timestamp=timestamp,
            analyst=self.agent_id,
            deviation=deviation,
            ranked_causes=analysis_result,
            executive_summary=executive_summary,
            technical_summary=technical_summary,
            recommended_actions=recommended_actions,
            overall_confidence=overall_confidence,
            limitations=limitations,
            nodes_examined=analysis_result.total_causes_considered,
            evidence_points=sum(
                len(c.evidence) for c in analysis_result.causes
            ),
        )

        self._reports[report_id] = report
        logger.info(f"Generated RCA report: {report_id}")

        return report

    def _generate_executive_summary(
        self,
        deviation: Deviation,
        analysis: RankedCauses,
    ) -> str:
        """Generate executive summary for report."""
        summary_parts = []

        # Deviation description
        summary_parts.append(
            f"Analysis of {deviation.metric_name} deviation: "
            f"current value {deviation.current_value:.1f} vs "
            f"baseline {deviation.baseline_value:.1f} "
            f"({deviation.deviation_percent:+.1f}%)."
        )

        # Root cause identification
        if analysis.most_likely_cause:
            cause = analysis.most_likely_cause
            summary_parts.append(
                f"Most likely root cause: {cause.cause_name} "
                f"({cause.probability:.0%} confidence)."
            )
        else:
            summary_parts.append("No clear root cause identified.")

        # Recommendation
        if analysis.causes and analysis.causes[0].suggested_action:
            summary_parts.append(
                f"Recommended action: {analysis.causes[0].suggested_action}"
            )

        return " ".join(summary_parts)

    def _generate_technical_summary(
        self,
        deviation: Deviation,
        analysis: RankedCauses,
    ) -> str:
        """Generate technical summary for report."""
        lines = []

        lines.append(f"Deviation Analysis: {deviation.metric_name}")
        lines.append(f"  Type: {deviation.deviation_type.value}")
        lines.append(f"  Magnitude: {deviation.deviation_magnitude:.2f} {deviation.unit}")
        lines.append(f"  Severity: {deviation.severity}")
        lines.append("")

        lines.append(f"Causal Analysis:")
        lines.append(f"  Nodes examined: {analysis.total_causes_considered}")
        lines.append(f"  Causes identified: {len(analysis.causes)}")
        lines.append("")

        lines.append("Top Causes:")
        for cause in analysis.causes[:3]:
            lines.append(f"  {cause.rank}. {cause.cause_name}")
            lines.append(f"     Probability: {cause.probability:.1%}")
            lines.append(f"     Path length: {cause.path_length}")
            lines.append(f"     Evidence: {cause.evidence_summary}")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        analysis: RankedCauses,
    ) -> List[str]:
        """Generate recommended actions from analysis."""
        recommendations = []

        for cause in analysis.causes[:3]:
            if cause.suggested_action:
                recommendations.append(
                    f"{cause.rank}. {cause.suggested_action} "
                    f"(addresses {cause.cause_name})"
                )

        if not recommendations:
            recommendations.append("Continue monitoring; investigate further if deviation persists")

        return recommendations

    def _identify_limitations(
        self,
        analysis: RankedCauses,
    ) -> List[str]:
        """Identify limitations of the analysis."""
        limitations = []

        if analysis.top_cause_confidence < 0.5:
            limitations.append("Low confidence in root cause identification")

        if analysis.total_causes_considered < 3:
            limitations.append("Limited causal graph coverage")

        evidence_count = sum(len(c.evidence) for c in analysis.causes)
        if evidence_count < 5:
            limitations.append("Limited supporting evidence available")

        if len(analysis.alternative_explanations) > 2:
            limitations.append("Multiple alternative explanations possible")

        return limitations

    def get_analysis(self, analysis_id: str) -> Optional[RankedCauses]:
        """Get analysis by ID."""
        return self._analyses.get(analysis_id)

    def get_report(self, report_id: str) -> Optional[RCAReport]:
        """Get report by ID."""
        return self._reports.get(report_id)
