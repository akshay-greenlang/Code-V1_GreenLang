"""
GL-006 HEATRECLAIM - Engineering Rationale Generator

Generates deterministic, rule-based explanations based on
pinch analysis principles, thermodynamic laws, and
heat exchanger design heuristics.

This provides the first layer of explainability:
transparent, auditable engineering reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json
import logging
import math

from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    PinchAnalysisResult,
)
from ..core.config import StreamType

logger = logging.getLogger(__name__)


class RationaleCategory(Enum):
    """Category of engineering rationale."""
    PINCH_RULE = "pinch_rule"
    THERMODYNAMIC = "thermodynamic"
    HEURISTIC = "heuristic"
    CONSTRAINT = "constraint"
    ECONOMIC = "economic"
    SAFETY = "safety"


@dataclass
class RationaleItem:
    """Single rationale item."""
    category: RationaleCategory
    rule_name: str
    description: str
    applied: bool
    impact: str
    confidence: float
    references: List[str]


@dataclass
class MatchRationale:
    """Rationale for a specific exchanger match."""
    exchanger_id: str
    hot_stream_id: str
    cold_stream_id: str
    match_rationale: List[RationaleItem]
    alternatives_considered: List[str]
    why_not_alternatives: List[str]
    overall_confidence: float


@dataclass
class EngineeringRationale:
    """Complete engineering rationale for design."""
    design_rationales: List[RationaleItem]
    match_rationales: List[MatchRationale]
    pinch_interpretation: str
    utility_justification: str
    key_constraints: List[str]
    design_trade_offs: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""


class EngineeringRationaleGenerator:
    """
    Generator for engineering-based explanations.

    Applies pinch analysis rules and thermodynamic principles
    to generate transparent, deterministic explanations for
    optimization decisions.

    Golden Rules Applied:
    1. Do not transfer heat across the pinch
    2. Above the pinch: no cold utility
    3. Below the pinch: no hot utility
    4. Match high-FCp with high-FCp first
    5. Tick-off heuristic for match duties

    Example:
        >>> generator = EngineeringRationaleGenerator()
        >>> rationale = generator.generate_rationale(design, pinch_result)
        >>> print(rationale.pinch_interpretation)
    """

    VERSION = "1.0.0"

    # Pinch analysis golden rules
    GOLDEN_RULES = {
        "no_cross_pinch": {
            "name": "No Cross-Pinch Transfer",
            "description": "Heat must not be transferred across the pinch point",
            "category": RationaleCategory.PINCH_RULE,
            "references": ["Linnhoff & Hindmarsh, 1983"],
        },
        "no_cold_utility_above": {
            "name": "No Cold Utility Above Pinch",
            "description": "Cold utility should not be used above the pinch temperature",
            "category": RationaleCategory.PINCH_RULE,
            "references": ["Linnhoff & Flower, 1978"],
        },
        "no_hot_utility_below": {
            "name": "No Hot Utility Below Pinch",
            "description": "Hot utility should not be used below the pinch temperature",
            "category": RationaleCategory.PINCH_RULE,
            "references": ["Linnhoff & Flower, 1978"],
        },
    }

    # FCp matching heuristics
    FCP_HEURISTICS = {
        "above_pinch": {
            "name": "FCp Rule Above Pinch",
            "description": "Above pinch: FCp_cold ≥ FCp_hot for feasible match",
            "category": RationaleCategory.HEURISTIC,
            "references": ["Pinch Design Method"],
        },
        "below_pinch": {
            "name": "FCp Rule Below Pinch",
            "description": "Below pinch: FCp_hot ≥ FCp_cold for feasible match",
            "category": RationaleCategory.HEURISTIC,
            "references": ["Pinch Design Method"],
        },
    }

    def __init__(self) -> None:
        """Initialize rationale generator."""
        pass

    def generate_rationale(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_result: Optional[PinchAnalysisResult] = None,
        delta_t_min: float = 10.0,
    ) -> EngineeringRationale:
        """
        Generate complete engineering rationale for design.

        Args:
            design: The HEN design to explain
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            pinch_result: Pinch analysis results
            delta_t_min: Minimum approach temperature

        Returns:
            EngineeringRationale with full explanation
        """
        # Generate design-level rationales
        design_rationales = self._generate_design_rationales(
            design, hot_streams, cold_streams, pinch_result, delta_t_min
        )

        # Generate match-level rationales
        match_rationales = self._generate_match_rationales(
            design.exchangers, hot_streams, cold_streams,
            pinch_result, delta_t_min
        )

        # Generate pinch interpretation
        pinch_interpretation = self._interpret_pinch(
            pinch_result, hot_streams, cold_streams, delta_t_min
        )

        # Justify utility requirements
        utility_justification = self._justify_utilities(
            design, pinch_result
        )

        # Identify key constraints
        key_constraints = self._identify_constraints(
            design, hot_streams, cold_streams, delta_t_min
        )

        # Identify trade-offs
        design_trade_offs = self._identify_trade_offs(
            design, pinch_result
        )

        # Compute hash
        computation_hash = self._compute_hash(
            design_rationales, match_rationales
        )

        return EngineeringRationale(
            design_rationales=design_rationales,
            match_rationales=match_rationales,
            pinch_interpretation=pinch_interpretation,
            utility_justification=utility_justification,
            key_constraints=key_constraints,
            design_trade_offs=design_trade_offs,
            computation_hash=computation_hash,
        )

    def explain_match(
        self,
        exchanger: HeatExchanger,
        hot_stream: HeatStream,
        cold_stream: HeatStream,
        pinch_temperature: Optional[float] = None,
        delta_t_min: float = 10.0,
    ) -> MatchRationale:
        """
        Generate detailed rationale for a single match.

        Args:
            exchanger: The heat exchanger match
            hot_stream: Hot stream in match
            cold_stream: Cold stream in match
            pinch_temperature: Pinch temperature if known
            delta_t_min: Minimum approach temperature

        Returns:
            MatchRationale with detailed explanation
        """
        rationale_items = []

        # Check temperature driving force
        dt_hot_in = exchanger.hot_inlet_T_C - exchanger.cold_outlet_T_C
        dt_cold_in = exchanger.hot_outlet_T_C - exchanger.cold_inlet_T_C
        dt_min_actual = min(dt_hot_in, dt_cold_in)

        rationale_items.append(RationaleItem(
            category=RationaleCategory.THERMODYNAMIC,
            rule_name="Temperature Driving Force",
            description=(
                f"ΔT at hot end: {dt_hot_in:.1f}°C, "
                f"ΔT at cold end: {dt_cold_in:.1f}°C, "
                f"Minimum: {dt_min_actual:.1f}°C ≥ ΔTmin ({delta_t_min}°C)"
            ),
            applied=dt_min_actual >= delta_t_min,
            impact="Ensures feasible heat transfer",
            confidence=0.99,
            references=["Second Law of Thermodynamics"],
        ))

        # Check FCp constraint
        hot_FCp = hot_stream.FCp_kW_K
        cold_FCp = cold_stream.FCp_kW_K

        # Determine position relative to pinch
        if pinch_temperature is not None:
            above_pinch = exchanger.hot_inlet_T_C >= pinch_temperature

            if above_pinch:
                fcp_satisfied = cold_FCp >= hot_FCp
                rule_desc = f"Above pinch: FCp_cold ({cold_FCp:.1f}) ≥ FCp_hot ({hot_FCp:.1f})"
            else:
                fcp_satisfied = hot_FCp >= cold_FCp
                rule_desc = f"Below pinch: FCp_hot ({hot_FCp:.1f}) ≥ FCp_cold ({cold_FCp:.1f})"

            rationale_items.append(RationaleItem(
                category=RationaleCategory.HEURISTIC,
                rule_name="FCp Matching Rule",
                description=rule_desc,
                applied=fcp_satisfied,
                impact="Ensures pinch-feasible match progression",
                confidence=0.95,
                references=["Pinch Design Method"],
            ))

        # Check duty utilization
        max_duty = min(hot_stream.duty_kW, cold_stream.duty_kW)
        duty_utilization = exchanger.duty_kW / max_duty if max_duty > 0 else 0

        rationale_items.append(RationaleItem(
            category=RationaleCategory.HEURISTIC,
            rule_name="Tick-off Heuristic",
            description=(
                f"Match duty: {exchanger.duty_kW:.1f} kW, "
                f"Maximum possible: {max_duty:.1f} kW, "
                f"Utilization: {duty_utilization*100:.1f}%"
            ),
            applied=duty_utilization > 0.5,
            impact="Maximizes heat recovery per match",
            confidence=0.85,
            references=["Linnhoff & Hindmarsh, 1983"],
        ))

        # Consider alternatives
        alternatives = [
            "Match with different cold stream",
            "Split hot stream for parallel matches",
            "Use utility instead of process stream",
        ]

        why_not = self._generate_alternatives_reasoning(
            exchanger, hot_stream, cold_stream, delta_t_min
        )

        # Calculate overall confidence
        confidence_scores = [r.confidence for r in rationale_items if r.applied]
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        return MatchRationale(
            exchanger_id=exchanger.exchanger_id,
            hot_stream_id=hot_stream.stream_id,
            cold_stream_id=cold_stream.stream_id,
            match_rationale=rationale_items,
            alternatives_considered=alternatives,
            why_not_alternatives=why_not,
            overall_confidence=round(overall_confidence, 4),
        )

    def _generate_design_rationales(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_result: Optional[PinchAnalysisResult],
        delta_t_min: float,
    ) -> List[RationaleItem]:
        """Generate design-level rationales."""
        rationales = []

        # Check pinch rules
        if pinch_result:
            pinch_T = pinch_result.pinch_temperature_C

            # Check for cross-pinch violations
            cross_pinch = False
            for hx in design.exchangers:
                if hx.hot_inlet_T_C > pinch_T and hx.cold_inlet_T_C < pinch_T - delta_t_min:
                    cross_pinch = True
                    break

            rationales.append(RationaleItem(
                category=RationaleCategory.PINCH_RULE,
                rule_name=self.GOLDEN_RULES["no_cross_pinch"]["name"],
                description=self.GOLDEN_RULES["no_cross_pinch"]["description"],
                applied=not cross_pinch,
                impact="Ensures minimum utility consumption",
                confidence=0.99,
                references=self.GOLDEN_RULES["no_cross_pinch"]["references"],
            ))

            # Check utility placement
            hot_utility_below = design.hot_utility_required_kW > 0 and any(
                hx.hot_stream_id == "HOT_UTILITY" and hx.cold_inlet_T_C < pinch_T
                for hx in design.exchangers
            )

            rationales.append(RationaleItem(
                category=RationaleCategory.PINCH_RULE,
                rule_name=self.GOLDEN_RULES["no_hot_utility_below"]["name"],
                description=self.GOLDEN_RULES["no_hot_utility_below"]["description"],
                applied=not hot_utility_below,
                impact="Prevents wasteful utility use",
                confidence=0.95,
                references=self.GOLDEN_RULES["no_hot_utility_below"]["references"],
            ))

        # Heat recovery efficiency
        total_hot_duty = sum(s.duty_kW for s in hot_streams)
        total_cold_duty = sum(s.duty_kW for s in cold_streams)
        max_recovery = min(total_hot_duty, total_cold_duty)
        recovery_efficiency = design.total_heat_recovered_kW / max_recovery if max_recovery > 0 else 0

        rationales.append(RationaleItem(
            category=RationaleCategory.THERMODYNAMIC,
            rule_name="Heat Recovery Efficiency",
            description=(
                f"Recovered {design.total_heat_recovered_kW:.1f} kW of "
                f"{max_recovery:.1f} kW potential ({recovery_efficiency*100:.1f}%)"
            ),
            applied=recovery_efficiency > 0.7,
            impact="Quantifies thermodynamic effectiveness",
            confidence=0.99,
            references=["First Law of Thermodynamics"],
        ))

        # Number of exchangers vs minimum
        n_streams = len(hot_streams) + len(cold_streams)
        n_utilities = (1 if design.hot_utility_required_kW > 0 else 0) + \
                      (1 if design.cold_utility_required_kW > 0 else 0)
        u_min = n_streams + n_utilities - 1

        rationales.append(RationaleItem(
            category=RationaleCategory.HEURISTIC,
            rule_name="Minimum Units Targeting",
            description=(
                f"Design has {design.exchanger_count} exchangers, "
                f"minimum theoretical: {u_min}"
            ),
            applied=design.exchanger_count <= u_min + 2,
            impact="Balances complexity vs performance",
            confidence=0.85,
            references=["Euler's Network Theorem"],
        ))

        return rationales

    def _generate_match_rationales(
        self,
        exchangers: List[HeatExchanger],
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_result: Optional[PinchAnalysisResult],
        delta_t_min: float,
    ) -> List[MatchRationale]:
        """Generate rationales for each match."""
        stream_map = {s.stream_id: s for s in hot_streams + cold_streams}
        match_rationales = []

        pinch_T = pinch_result.pinch_temperature_C if pinch_result else None

        for hx in exchangers:
            hot_stream = stream_map.get(hx.hot_stream_id)
            cold_stream = stream_map.get(hx.cold_stream_id)

            if hot_stream and cold_stream:
                match_rationale = self.explain_match(
                    hx, hot_stream, cold_stream, pinch_T, delta_t_min
                )
                match_rationales.append(match_rationale)

        return match_rationales

    def _interpret_pinch(
        self,
        pinch_result: Optional[PinchAnalysisResult],
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> str:
        """Generate interpretation of pinch analysis."""
        if pinch_result is None:
            return "Pinch analysis not performed. Energy targets unavailable."

        T_pinch = pinch_result.pinch_temperature_C
        Q_Hmin = pinch_result.minimum_hot_utility_kW
        Q_Cmin = pinch_result.minimum_cold_utility_kW
        Q_rec = pinch_result.maximum_heat_recovery_kW

        interpretation = (
            f"The pinch temperature is {T_pinch:.1f}°C (hot side: {T_pinch}°C, "
            f"cold side: {T_pinch - delta_t_min}°C). "
            f"At ΔTmin = {delta_t_min}°C, the energy targets are:\n"
            f"• Minimum hot utility: {Q_Hmin:.1f} kW\n"
            f"• Minimum cold utility: {Q_Cmin:.1f} kW\n"
            f"• Maximum heat recovery: {Q_rec:.1f} kW\n\n"
            f"The pinch divides the problem into two regions:\n"
            f"• Above pinch (>{T_pinch}°C): Heat deficit - hot utility required\n"
            f"• Below pinch (<{T_pinch - delta_t_min}°C): Heat surplus - cold utility required\n\n"
            f"Respecting pinch rules ensures thermodynamically optimal design."
        )

        return interpretation

    def _justify_utilities(
        self,
        design: HENDesign,
        pinch_result: Optional[PinchAnalysisResult],
    ) -> str:
        """Justify utility requirements."""
        Q_H = design.hot_utility_required_kW
        Q_C = design.cold_utility_required_kW

        justification = f"Utility requirements: Hot = {Q_H:.1f} kW, Cold = {Q_C:.1f} kW.\n\n"

        if pinch_result:
            Q_Hmin = pinch_result.minimum_hot_utility_kW
            Q_Cmin = pinch_result.minimum_cold_utility_kW

            if Q_H <= Q_Hmin * 1.05:
                justification += "Hot utility is at or near minimum target (within 5%). "
            else:
                excess = Q_H - Q_Hmin
                justification += (
                    f"Hot utility exceeds minimum by {excess:.1f} kW. "
                    "This may be due to practical constraints or cross-pinch transfer. "
                )

            if Q_C <= Q_Cmin * 1.05:
                justification += "Cold utility is at or near minimum target."
            else:
                excess = Q_C - Q_Cmin
                justification += (
                    f"Cold utility exceeds minimum by {excess:.1f} kW. "
                    "Consider reducing stream outlet temperatures or improving matches."
                )
        else:
            justification += (
                "Without pinch analysis, utility efficiency cannot be assessed. "
                "Consider performing pinch analysis to establish energy targets."
            )

        return justification

    def _identify_constraints(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> List[str]:
        """Identify active constraints."""
        constraints = []

        # ΔTmin constraint
        constraints.append(
            f"Minimum approach temperature: ΔTmin = {delta_t_min}°C"
        )

        # Temperature constraints
        T_max_hot = max(s.T_supply_C for s in hot_streams) if hot_streams else 0
        T_min_cold = min(s.T_supply_C for s in cold_streams) if cold_streams else 0
        constraints.append(
            f"Temperature range: {T_min_cold:.0f}°C to {T_max_hot:.0f}°C"
        )

        # Duty balance
        total_hot = sum(s.duty_kW for s in hot_streams)
        total_cold = sum(s.duty_kW for s in cold_streams)
        constraints.append(
            f"Duty balance: Hot = {total_hot:.0f} kW, Cold = {total_cold:.0f} kW"
        )

        return constraints

    def _identify_trade_offs(
        self,
        design: HENDesign,
        pinch_result: Optional[PinchAnalysisResult],
    ) -> List[str]:
        """Identify design trade-offs."""
        trade_offs = []

        trade_offs.append(
            "ΔTmin trade-off: Lower ΔTmin increases heat recovery but requires "
            "larger (more expensive) exchangers"
        )

        trade_offs.append(
            "Number of exchangers trade-off: More exchangers can improve recovery "
            "but increase capital cost and complexity"
        )

        if pinch_result and pinch_result.is_threshold_problem:
            trade_offs.append(
                "Threshold problem: One utility type dominates. "
                "Limited flexibility in heat integration."
            )

        return trade_offs

    def _generate_alternatives_reasoning(
        self,
        exchanger: HeatExchanger,
        hot_stream: HeatStream,
        cold_stream: HeatStream,
        delta_t_min: float,
    ) -> List[str]:
        """Generate reasoning for why alternatives were not chosen."""
        reasons = []

        # Temperature match quality
        dt1 = exchanger.hot_inlet_T_C - exchanger.cold_outlet_T_C
        dt2 = exchanger.hot_outlet_T_C - exchanger.cold_inlet_T_C

        if min(dt1, dt2) > delta_t_min * 2:
            reasons.append(
                "Good temperature driving force throughout; no need for splitting"
            )
        else:
            reasons.append(
                f"Temperature approach near ΔTmin; this match maximizes the available driving force"
            )

        # Duty considerations
        if exchanger.duty_kW > 0.5 * min(hot_stream.duty_kW, cold_stream.duty_kW):
            reasons.append(
                "Match captures significant portion of available duty"
            )

        return reasons

    def _compute_hash(
        self,
        design_rationales: List[RationaleItem],
        match_rationales: List[MatchRationale],
    ) -> str:
        """Compute SHA-256 hash for provenance."""
        data = {
            "design_rationales": [
                {
                    "rule": r.rule_name,
                    "applied": r.applied,
                    "confidence": r.confidence,
                }
                for r in design_rationales
            ],
            "match_rationales": [
                {
                    "exchanger": m.exchanger_id,
                    "confidence": m.overall_confidence,
                }
                for m in match_rationales
            ],
            "version": self.VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
