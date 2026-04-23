"""
TripCausalAnalyzer - Analyzes combustion system trip events using causal reasoning.

This module implements causal analysis of trip sequences to identify initiators,
contributing factors, and recommend prevention strategies.

Example:
    >>> analyzer = TripCausalAnalyzer(causal_graph)
    >>> events = [Event(variable='O2', value=0.5, timestamp=t1), ...]
    >>> analysis = analyzer.analyze_trip_sequence(events)
    >>> print(analysis.initiator)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from causal.causal_graph import CausalGraph, NodeType

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    ALARM = "alarm"
    WARNING = "warning"
    TRIP = "trip"
    SETPOINT_CHANGE = "setpoint_change"
    MEASUREMENT = "measurement"
    OPERATOR_ACTION = "operator_action"


class Event(BaseModel):
    variable: str = Field(..., description="Variable involved")
    value: float = Field(..., description="Value at event time")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: EventType = Field(default=EventType.MEASUREMENT)
    description: str = Field("")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class TripEvent(BaseModel):
    trip_id: str = Field(..., description="Unique trip identifier")
    trip_type: str = Field(..., description="Type of trip")
    trip_variable: str = Field(..., description="Variable that triggered trip")
    trip_value: float = Field(..., description="Value at trip")
    trip_setpoint: float = Field(..., description="Trip setpoint")
    timestamp: datetime = Field(...)
    preceding_events: List[Event] = Field(default_factory=list)
    duration_to_trip_seconds: float = Field(...)


class TripInitiator(BaseModel):
    variable: str = Field(..., description="Initiating variable")
    confidence: float = Field(..., ge=0.0, le=1.0)
    time_before_trip: float = Field(..., description="Seconds before trip")
    deviation_from_normal: float = Field(...)
    causal_path_to_trip: List[str] = Field(default_factory=list)
    mechanism: str = Field(...)
    evidence: List[str] = Field(default_factory=list)


class ContributingFactor(BaseModel):
    variable: str = Field(...)
    contribution_score: float = Field(..., ge=0.0, le=1.0)
    timing: str = Field(..., description="early, mid, late")
    role: str = Field(..., description="initiator, amplifier, or enabler")
    evidence: str = Field(...)


class PreventionRecommendation(BaseModel):
    priority: int = Field(..., ge=1, le=5)
    action: str = Field(...)
    target_variable: str = Field(...)
    rationale: str = Field(...)
    expected_effectiveness: float = Field(..., ge=0.0, le=1.0)
    implementation_difficulty: str = Field(...)
    monitoring_requirements: List[str] = Field(default_factory=list)


class TripSequenceAnalysis(BaseModel):
    trip: TripEvent = Field(...)
    event_sequence: List[Event] = Field(...)
    initiator: TripInitiator = Field(...)
    contributing_factors: List[ContributingFactor] = Field(default_factory=list)
    timeline: Dict[str, List[str]] = Field(default_factory=dict)
    causal_chain: List[str] = Field(default_factory=list)
    analysis_confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("")


class TripReport(BaseModel):
    analysis: TripSequenceAnalysis = Field(...)
    executive_summary: str = Field(...)
    detailed_narrative: str = Field(...)
    root_cause_statement: str = Field(...)
    recommendations: List[PreventionRecommendation] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    report_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field("")


class TripCausalAnalyzer:
    """Causal analyzer for combustion system trip events."""

    TRIP_THRESHOLDS = {
        "O2": {"low": 1.0, "high": 12.0},
        "CO": {"high": 1000.0},
        "flame_temp": {"low": 600.0, "high": 2300.0},
        "stability": {"low": 0.2},
        "fuel_flow": {"low": 0.1, "high": 80.0},
        "air_flow": {"low": 1.0, "high": 800.0}
    }

    def __init__(self, graph: CausalGraph, historical_trips: Optional[List[TripEvent]] = None):
        self.graph = graph
        self._nx_graph = graph.to_networkx()
        self.historical_trips = historical_trips or []
        logger.info(f"TripCausalAnalyzer initialized with {len(graph.nodes)} nodes")

    def analyze_trip_sequence(self, events: List[Event]) -> TripSequenceAnalysis:
        logger.info(f"Analyzing trip sequence with {len(events)} events")

        if not events:
            raise ValueError("Event list cannot be empty")

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        trip_event = self._identify_trip_event(sorted_events)
        initiator = self.identify_trip_initiator_from_analysis(sorted_events, trip_event)
        contributing = self.compute_contributing_factors(trip_event)
        timeline = self._build_timeline(sorted_events, trip_event)
        causal_chain = self._trace_causal_chain(initiator.variable, trip_event.trip_variable)
        confidence = self._calculate_analysis_confidence(initiator, contributing, causal_chain)

        analysis_data = f"{trip_event.trip_id}{initiator.variable}{confidence}"
        provenance_hash = hashlib.sha256(analysis_data.encode()).hexdigest()

        analysis = TripSequenceAnalysis(
            trip=trip_event, event_sequence=sorted_events,
            initiator=initiator, contributing_factors=contributing,
            timeline=timeline, causal_chain=causal_chain,
            analysis_confidence=confidence, provenance_hash=provenance_hash)

        logger.info(f"Trip analysis complete: initiator={initiator.variable}")
        return analysis

    def _identify_trip_event(self, events: List[Event]) -> TripEvent:
        trip_events = [e for e in events if e.event_type == EventType.TRIP]

        if trip_events:
            trip_event = trip_events[-1]
        else:
            max_deviation = 0
            trip_event = events[-1]
            for event in events:
                threshold = self.TRIP_THRESHOLDS.get(event.variable, {})
                if "high" in threshold and event.value > threshold["high"]:
                    deviation = event.value / threshold["high"]
                    if deviation > max_deviation:
                        max_deviation = deviation
                        trip_event = event
                elif "low" in threshold and event.value < threshold["low"]:
                    deviation = threshold["low"] / max(event.value, 0.001)
                    if deviation > max_deviation:
                        max_deviation = deviation
                        trip_event = event

        trip_setpoint = self.TRIP_THRESHOLDS.get(trip_event.variable, {}).get("high", trip_event.value * 1.2)
        first_event = events[0]
        duration = (trip_event.timestamp - first_event.timestamp).total_seconds()
        preceding = [e for e in events if e.timestamp < trip_event.timestamp]

        return TripEvent(
            trip_id=f"TRIP_{trip_event.timestamp.strftime('%Y%m%d%H%M%S')}",
            trip_type=f"{trip_event.variable}_trip",
            trip_variable=trip_event.variable,
            trip_value=trip_event.value,
            trip_setpoint=trip_setpoint,
            timestamp=trip_event.timestamp,
            preceding_events=preceding,
            duration_to_trip_seconds=duration)

    def identify_trip_initiator(self, sequence: TripSequenceAnalysis) -> TripInitiator:
        return sequence.initiator

    def identify_trip_initiator_from_analysis(self, events: List[Event], trip: TripEvent) -> TripInitiator:
        logger.info(f"Identifying trip initiator for {trip.trip_variable}")

        ancestors = self.graph.get_ancestors(trip.trip_variable)
        direct_parents = self.graph.get_parents(trip.trip_variable)

        initiator_scores = {}
        for event in events:
            if event.timestamp >= trip.timestamp:
                continue
            if event.variable not in ancestors and event.variable not in direct_parents:
                continue

            time_before = (trip.timestamp - event.timestamp).total_seconds()
            time_score = min(time_before / 60.0, 1.0)

            try:
                path = nx.shortest_path(self._nx_graph, event.variable, trip.trip_variable)
                path_score = 1.0 / len(path)
            except nx.NetworkXNoPath:
                path_score = 0.0

            threshold = self.TRIP_THRESHOLDS.get(event.variable, {})
            if "high" in threshold:
                deviation = event.value / threshold["high"] if threshold["high"] > 0 else 0
            elif "low" in threshold:
                deviation = threshold["low"] / max(event.value, 0.001)
            else:
                deviation = 0.5

            total_score = (time_score * 0.3 + path_score * 0.4 + min(deviation, 1.0) * 0.3)
            initiator_scores[event.variable] = max(initiator_scores.get(event.variable, 0), total_score)

        if initiator_scores:
            best_var = max(initiator_scores, key=initiator_scores.get)
            confidence = initiator_scores[best_var]
            best_event = next((e for e in reversed(events) if e.variable == best_var), events[0])
        else:
            best_var = trip.trip_variable
            confidence = 0.3
            best_event = events[0]

        try:
            causal_path = nx.shortest_path(self._nx_graph, best_var, trip.trip_variable)
        except nx.NetworkXNoPath:
            causal_path = [best_var, trip.trip_variable]

        time_before = (trip.timestamp - best_event.timestamp).total_seconds()

        return TripInitiator(
            variable=best_var, confidence=confidence, time_before_trip=time_before,
            deviation_from_normal=best_event.value, causal_path_to_trip=causal_path,
            mechanism=f"Deviation in {best_var} propagated to {trip.trip_variable}",
            evidence=[f"First anomaly: {best_var}={best_event.value} at T-{time_before:.0f}s"])

    def compute_contributing_factors(self, trip: TripEvent) -> List[ContributingFactor]:
        logger.info(f"Computing contributing factors for trip {trip.trip_id}")

        factors = []
        ancestors = self.graph.get_ancestors(trip.trip_variable)

        for i, event in enumerate(trip.preceding_events):
            if event.variable not in ancestors:
                continue

            total_events = len(trip.preceding_events)
            if i < total_events * 0.3:
                timing = "early"
            elif i < total_events * 0.7:
                timing = "mid"
            else:
                timing = "late"

            if i == 0:
                role = "initiator"
            elif timing == "mid":
                role = "amplifier"
            else:
                role = "enabler"

            try:
                path = nx.shortest_path(self._nx_graph, event.variable, trip.trip_variable)
                contribution = 1.0 / len(path) * 0.8
            except nx.NetworkXNoPath:
                contribution = 0.2

            factors.append(ContributingFactor(
                variable=event.variable, contribution_score=contribution,
                timing=timing, role=role,
                evidence=f"{event.variable}={event.value} at {event.timestamp}"))

        return sorted(factors, key=lambda f: f.contribution_score, reverse=True)[:5]

    def _build_timeline(self, events: List[Event], trip: TripEvent) -> Dict[str, List[str]]:
        timeline = {"early": [], "mid": [], "late": [], "trip": []}

        if not events:
            return timeline

        first_time = events[0].timestamp
        trip_time = trip.timestamp
        total_duration = (trip_time - first_time).total_seconds()

        for event in events:
            elapsed = (event.timestamp - first_time).total_seconds()
            if total_duration > 0:
                progress = elapsed / total_duration
            else:
                progress = 1.0

            entry = f"{event.variable}={event.value:.2f} @ T+{elapsed:.0f}s"

            if event.timestamp >= trip_time:
                timeline["trip"].append(entry)
            elif progress < 0.3:
                timeline["early"].append(entry)
            elif progress < 0.7:
                timeline["mid"].append(entry)
            else:
                timeline["late"].append(entry)

        return timeline

    def _trace_causal_chain(self, initiator: str, trip_var: str) -> List[str]:
        try:
            return nx.shortest_path(self._nx_graph, initiator, trip_var)
        except nx.NetworkXNoPath:
            return [initiator, trip_var]

    def _calculate_analysis_confidence(self, initiator: TripInitiator,
                                        contributing: List[ContributingFactor],
                                        causal_chain: List[str]) -> float:
        confidence = initiator.confidence * 0.5

        if contributing:
            avg_contribution = sum(f.contribution_score for f in contributing) / len(contributing)
            confidence += avg_contribution * 0.3

        if len(causal_chain) > 1:
            path_confidence = 1.0 / len(causal_chain)
            confidence += path_confidence * 0.2

        return min(confidence, 1.0)

    def recommend_prevention(self, analysis: TripSequenceAnalysis) -> PreventionRecommendation:
        logger.info(f"Generating prevention recommendation for {analysis.trip.trip_id}")

        initiator_var = analysis.initiator.variable

        if initiator_var == "fuel_flow":
            action = "Install fuel flow rate limiting and alarming"
            rationale = "Early detection and limiting of fuel flow deviations"
            effectiveness = 0.85
            difficulty = "medium"
            monitoring = ["Fuel flow rate", "Fuel pressure", "Air-fuel ratio"]
        elif initiator_var == "air_flow":
            action = "Implement air flow rate stabilization control"
            rationale = "Maintain stable air supply to prevent combustion instability"
            effectiveness = 0.8
            difficulty = "medium"
            monitoring = ["Air flow rate", "Damper position", "Fan speed"]
        elif initiator_var == "O2":
            action = "Add redundant O2 monitoring with voting logic"
            rationale = "Improve reliability of excess air measurement"
            effectiveness = 0.75
            difficulty = "low"
            monitoring = ["O2 sensors", "Air damper", "Combustion stoichiometry"]
        else:
            action = f"Implement early warning system for {initiator_var} deviations"
            rationale = f"Early detection of {initiator_var} anomalies before trip"
            effectiveness = 0.7
            difficulty = "low"
            monitoring = [initiator_var, "Related process variables"]

        return PreventionRecommendation(
            priority=1, action=action, target_variable=initiator_var,
            rationale=rationale, expected_effectiveness=effectiveness,
            implementation_difficulty=difficulty, monitoring_requirements=monitoring)

    def generate_trip_report(self, analysis: TripSequenceAnalysis) -> TripReport:
        logger.info(f"Generating trip report for {analysis.trip.trip_id}")

        # Executive summary
        summary_lines = [
            f"Trip Event {analysis.trip.trip_id}: {analysis.trip.trip_type}",
            f"Occurred at {analysis.trip.timestamp}",
            f"Trip variable: {analysis.trip.trip_variable} reached {analysis.trip.trip_value:.2f} (setpoint: {analysis.trip.trip_setpoint:.2f})",
            f"Root cause identified: {analysis.initiator.variable} (confidence: {analysis.initiator.confidence:.0%})",
            f"Duration to trip: {analysis.trip.duration_to_trip_seconds:.0f} seconds"
        ]
        executive_summary = "\n".join(summary_lines)

        # Detailed narrative
        narrative_lines = [
            f"Analysis of trip event {analysis.trip.trip_id}:",
            f"The trip was initiated by abnormal conditions in {analysis.initiator.variable}, occurring {analysis.initiator.time_before_trip:.0f} seconds before the trip.",
            f"The causal chain was: {' -> '.join(analysis.causal_chain)}",
            "Contributing factors included:"
        ]
        for factor in analysis.contributing_factors[:3]:
            narrative_lines.append(f"  - {factor.variable} ({factor.role}, {factor.timing}): {factor.evidence}")
        detailed_narrative = "\n".join(narrative_lines)

        # Root cause statement
        root_cause_statement = (
            f"The root cause of this trip was a deviation in {analysis.initiator.variable}. "
            f"{analysis.initiator.mechanism}. "
            f"Evidence: {'; '.join(analysis.initiator.evidence[:2])}"
        )

        # Recommendations
        prevention = self.recommend_prevention(analysis)
        recommendations = [prevention]

        for factor in analysis.contributing_factors[:2]:
            if factor.variable != analysis.initiator.variable:
                rec = PreventionRecommendation(
                    priority=2, action=f"Monitor {factor.variable} for early deviations",
                    target_variable=factor.variable,
                    rationale=f"{factor.variable} was a {factor.role} in this trip",
                    expected_effectiveness=0.6, implementation_difficulty="low",
                    monitoring_requirements=[factor.variable])
                recommendations.append(rec)

        # Lessons learned
        lessons = [
            f"Early detection of {analysis.initiator.variable} anomalies is critical",
            f"Causal chain from {analysis.initiator.variable} to {analysis.trip.trip_variable} should be monitored",
            f"Trip occurred within {analysis.trip.duration_to_trip_seconds:.0f}s - faster response needed"
        ]

        report_data = f"{analysis.trip.trip_id}{executive_summary[:100]}"
        provenance_hash = hashlib.sha256(report_data.encode()).hexdigest()

        report = TripReport(
            analysis=analysis, executive_summary=executive_summary,
            detailed_narrative=detailed_narrative, root_cause_statement=root_cause_statement,
            recommendations=recommendations, lessons_learned=lessons,
            provenance_hash=provenance_hash)

        logger.info(f"Trip report generated for {analysis.trip.trip_id}")
        return report
