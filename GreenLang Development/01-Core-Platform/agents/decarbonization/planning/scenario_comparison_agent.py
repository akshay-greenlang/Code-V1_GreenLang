# -*- coding: utf-8 -*-
"""
GL-DECARB-X-019: Scenario Comparison Agent
===========================================

Compares decarbonization scenarios on multiple dimensions.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class ScenarioMetrics(BaseModel):
    scenario_id: str = Field(...)
    scenario_name: str = Field(...)

    # Emission impact
    total_reduction_tco2e: float = Field(default=0, ge=0)
    reduction_percent: float = Field(default=0, ge=0, le=100)

    # Financial
    total_investment_usd: float = Field(default=0, ge=0)
    npv_usd: float = Field(default=0)
    average_cost_per_tco2e: float = Field(default=0)
    payback_years: Optional[float] = Field(None, ge=0)

    # Risk
    risk_score: float = Field(default=0.5, ge=0, le=1)
    technology_readiness_avg: float = Field(default=7, ge=1, le=9)

    # Implementation
    number_of_projects: int = Field(default=0, ge=0)
    implementation_years: int = Field(default=5, ge=1)


class ScenarioComparison(BaseModel):
    comparison_id: str = Field(...)
    scenarios: List[ScenarioMetrics] = Field(default_factory=list)

    # Rankings
    rank_by_reduction: List[str] = Field(default_factory=list)
    rank_by_cost_effectiveness: List[str] = Field(default_factory=list)
    rank_by_npv: List[str] = Field(default_factory=list)
    rank_by_risk: List[str] = Field(default_factory=list)

    # Recommendations
    recommended_scenario: Optional[str] = Field(None)
    recommendation_rationale: str = Field(default="")

    provenance_hash: str = Field(default="")


class ScenarioComparisonInput(BaseModel):
    operation: str = Field(default="compare")
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioComparisonOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    comparison: Optional[ScenarioComparison] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class ScenarioComparisonAgent(DeterministicAgent):
    """GL-DECARB-X-019: Scenario Comparison Agent"""

    AGENT_ID = "GL-DECARB-X-019"
    AGENT_NAME = "Scenario Comparison Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="ScenarioComparisonAgent",
        category=AgentCategory.CRITICAL,
        description="Compares decarbonization scenarios"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Compares scenarios", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            sc_input = ScenarioComparisonInput(**inputs)
            calculation_trace.append(f"Operation: {sc_input.operation}")

            if sc_input.operation == "compare":
                # Parse scenarios
                scenarios = [ScenarioMetrics(**s) for s in sc_input.scenarios]

                if len(scenarios) < 2:
                    raise ValueError("At least 2 scenarios required for comparison")

                # Rank by different criteria
                by_reduction = sorted(scenarios, key=lambda s: s.total_reduction_tco2e, reverse=True)
                by_cost_eff = sorted(scenarios, key=lambda s: s.average_cost_per_tco2e)
                by_npv = sorted(scenarios, key=lambda s: s.npv_usd, reverse=True)
                by_risk = sorted(scenarios, key=lambda s: s.risk_score)

                # Determine recommendation (simple scoring)
                scores = {}
                for s in scenarios:
                    score = 0
                    score += (len(scenarios) - by_reduction.index(s)) * 3  # Weight reduction highest
                    score += (len(scenarios) - by_cost_eff.index(s)) * 2
                    score += (len(scenarios) - by_npv.index(s)) * 2
                    score += (len(scenarios) - by_risk.index(s)) * 1
                    scores[s.scenario_id] = score

                recommended_id = max(scores.keys(), key=lambda k: scores[k])
                recommended = next(s for s in scenarios if s.scenario_id == recommended_id)

                comparison = ScenarioComparison(
                    comparison_id=deterministic_id({"count": len(scenarios)}, "compare_"),
                    scenarios=scenarios,
                    rank_by_reduction=[s.scenario_id for s in by_reduction],
                    rank_by_cost_effectiveness=[s.scenario_id for s in by_cost_eff],
                    rank_by_npv=[s.scenario_id for s in by_npv],
                    rank_by_risk=[s.scenario_id for s in by_risk],
                    recommended_scenario=recommended_id,
                    recommendation_rationale=f"'{recommended.scenario_name}' achieves {recommended.reduction_percent:.1f}% reduction at ${recommended.average_cost_per_tco2e:.0f}/tCO2e with moderate risk"
                )
                comparison.provenance_hash = content_hash(comparison.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Compared {len(scenarios)} scenarios, recommended: {recommended_id}")

                self._capture_audit_entry(
                    operation="compare",
                    inputs=inputs,
                    outputs={"recommended": recommended_id},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "compare",
                    "success": True,
                    "comparison": comparison.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {sc_input.operation}")

        except Exception as e:
            self.logger.error(f"Comparison failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
