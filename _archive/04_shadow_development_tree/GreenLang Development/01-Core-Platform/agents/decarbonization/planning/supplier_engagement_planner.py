# -*- coding: utf-8 -*-
"""
GL-DECARB-X-016: Supplier Engagement Planner Agent
===================================================

Plans supplier engagement programs for Scope 3 emission reductions.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class EngagementTier(str, Enum):
    STRATEGIC = "strategic"  # Top suppliers, deep engagement
    COLLABORATIVE = "collaborative"  # Key suppliers, joint initiatives
    MONITORING = "monitoring"  # Track and report
    BASIC = "basic"  # Questionnaire only


class SupplierSegment(BaseModel):
    segment_name: str = Field(...)
    tier: EngagementTier = Field(...)
    supplier_count: int = Field(..., ge=0)
    spend_percent: float = Field(..., ge=0, le=100)
    emission_percent: float = Field(..., ge=0, le=100)
    engagement_activities: List[str] = Field(default_factory=list)
    target_reduction_percent: float = Field(default=0, ge=0, le=100)


class SupplierEngagementPlan(BaseModel):
    plan_id: str = Field(...)
    total_suppliers: int = Field(..., ge=0)
    scope_3_emissions_tco2e: float = Field(..., ge=0)
    segments: List[SupplierSegment] = Field(default_factory=list)

    # Coverage targets
    sbti_target_coverage_percent: float = Field(default=67, ge=0, le=100)
    achieved_coverage_percent: float = Field(default=0, ge=0, le=100)

    # Expected impact
    expected_reduction_tco2e: float = Field(default=0, ge=0)
    expected_reduction_percent: float = Field(default=0, ge=0, le=100)

    provenance_hash: str = Field(default="")


class SupplierEngagementInput(BaseModel):
    operation: str = Field(default="plan")
    total_suppliers: int = Field(default=500, ge=0)
    scope_3_emissions_tco2e: float = Field(default=500000, ge=0)
    top_supplier_spend_percent: float = Field(default=70, ge=0, le=100)
    sbti_coverage_target: float = Field(default=67, ge=0, le=100)


class SupplierEngagementOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    plan: Optional[SupplierEngagementPlan] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class SupplierEngagementPlanner(DeterministicAgent):
    """GL-DECARB-X-016: Supplier Engagement Planner Agent"""

    AGENT_ID = "GL-DECARB-X-016"
    AGENT_NAME = "Supplier Engagement Planner"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="SupplierEngagementPlanner",
        category=AgentCategory.CRITICAL,
        description="Plans supplier engagement for Scope 3 reductions"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Plans supplier engagement", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            se_input = SupplierEngagementInput(**inputs)
            calculation_trace.append(f"Operation: {se_input.operation}")

            if se_input.operation == "plan":
                segments = [
                    SupplierSegment(
                        segment_name="Strategic Partners",
                        tier=EngagementTier.STRATEGIC,
                        supplier_count=int(se_input.total_suppliers * 0.02),
                        spend_percent=40,
                        emission_percent=45,
                        engagement_activities=[
                            "Joint decarbonization roadmaps",
                            "Shared R&D initiatives",
                            "Quarterly reviews",
                            "SBTi alignment requirement"
                        ],
                        target_reduction_percent=30
                    ),
                    SupplierSegment(
                        segment_name="Key Suppliers",
                        tier=EngagementTier.COLLABORATIVE,
                        supplier_count=int(se_input.total_suppliers * 0.08),
                        spend_percent=30,
                        emission_percent=30,
                        engagement_activities=[
                            "Annual CDP disclosure",
                            "Capacity building programs",
                            "Data sharing agreements"
                        ],
                        target_reduction_percent=20
                    ),
                    SupplierSegment(
                        segment_name="Monitored Suppliers",
                        tier=EngagementTier.MONITORING,
                        supplier_count=int(se_input.total_suppliers * 0.20),
                        spend_percent=20,
                        emission_percent=18,
                        engagement_activities=[
                            "Annual questionnaire",
                            "Emission factor updates"
                        ],
                        target_reduction_percent=10
                    ),
                    SupplierSegment(
                        segment_name="Basic Suppliers",
                        tier=EngagementTier.BASIC,
                        supplier_count=int(se_input.total_suppliers * 0.70),
                        spend_percent=10,
                        emission_percent=7,
                        engagement_activities=["Self-assessment questionnaire"],
                        target_reduction_percent=5
                    ),
                ]

                # Calculate coverage and expected reduction
                covered_emission_pct = sum(
                    s.emission_percent for s in segments
                    if s.tier in [EngagementTier.STRATEGIC, EngagementTier.COLLABORATIVE]
                )

                expected_reduction = sum(
                    se_input.scope_3_emissions_tco2e * (s.emission_percent / 100) * (s.target_reduction_percent / 100)
                    for s in segments
                )

                plan = SupplierEngagementPlan(
                    plan_id=deterministic_id({"suppliers": se_input.total_suppliers}, "seplan_"),
                    total_suppliers=se_input.total_suppliers,
                    scope_3_emissions_tco2e=se_input.scope_3_emissions_tco2e,
                    segments=segments,
                    sbti_target_coverage_percent=se_input.sbti_coverage_target,
                    achieved_coverage_percent=covered_emission_pct,
                    expected_reduction_tco2e=expected_reduction,
                    expected_reduction_percent=(expected_reduction / se_input.scope_3_emissions_tco2e * 100) if se_input.scope_3_emissions_tco2e > 0 else 0
                )
                plan.provenance_hash = content_hash(plan.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Created plan with {len(segments)} supplier segments")

                self._capture_audit_entry(
                    operation="plan",
                    inputs=inputs,
                    outputs={"segments": len(segments)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "plan",
                    "success": True,
                    "plan": plan.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {se_input.operation}")

        except Exception as e:
            self.logger.error(f"Planning failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
