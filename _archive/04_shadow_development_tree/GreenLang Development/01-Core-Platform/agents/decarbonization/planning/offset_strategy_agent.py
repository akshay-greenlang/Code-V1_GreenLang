# -*- coding: utf-8 -*-
"""
GL-DECARB-X-015: Offset Strategy Agent
=======================================

Plans carbon offset and credit strategy including quality criteria,
portfolio composition, and procurement recommendations.

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


class OffsetType(str, Enum):
    NATURE_BASED = "nature_based"
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CARBON_REMOVAL = "carbon_removal"
    METHANE_AVOIDANCE = "methane_avoidance"


class OffsetStandard(str, Enum):
    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CORSIA = "corsia"


class QualityRating(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OffsetProject(BaseModel):
    project_id: str = Field(...)
    name: str = Field(...)
    offset_type: OffsetType = Field(...)
    standard: OffsetStandard = Field(...)
    country: str = Field(default="")
    vintage_year: int = Field(default=2024, ge=2015)

    # Volume and price
    available_credits_tco2e: float = Field(..., ge=0)
    price_usd_tco2e: float = Field(..., ge=0)

    # Quality
    quality_rating: QualityRating = Field(default=QualityRating.MEDIUM)
    additionality_score: float = Field(default=0.7, ge=0, le=1)
    permanence_score: float = Field(default=0.8, ge=0, le=1)
    co_benefits: List[str] = Field(default_factory=list)

    # Risk
    reversal_risk: str = Field(default="medium")
    political_risk: str = Field(default="medium")


class OffsetPortfolio(BaseModel):
    portfolio_id: str = Field(...)
    target_offset_tco2e: float = Field(..., ge=0)
    projects: List[OffsetProject] = Field(default_factory=list)

    # Summary
    total_credits_tco2e: float = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0, ge=0)
    average_price_usd_tco2e: float = Field(default=0, ge=0)

    # Quality metrics
    high_quality_percent: float = Field(default=0, ge=0, le=100)
    removal_percent: float = Field(default=0, ge=0, le=100)
    nature_based_percent: float = Field(default=0, ge=0, le=100)

    provenance_hash: str = Field(default="")


class OffsetStrategyInput(BaseModel):
    operation: str = Field(default="plan")
    target_offset_tco2e: float = Field(default=10000, ge=0)
    budget_usd: Optional[float] = Field(None, ge=0)
    min_quality: QualityRating = Field(default=QualityRating.MEDIUM)
    removal_target_percent: float = Field(default=0, ge=0, le=100)
    preferred_types: List[str] = Field(default_factory=list)


class OffsetStrategyOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    portfolio: Optional[OffsetPortfolio] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class OffsetStrategyAgent(DeterministicAgent):
    """GL-DECARB-X-015: Offset Strategy Agent"""

    AGENT_ID = "GL-DECARB-X-015"
    AGENT_NAME = "Offset Strategy Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="OffsetStrategyAgent",
        category=AgentCategory.CRITICAL,
        description="Plans carbon offset strategy"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Plans offset strategy", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            offset_input = OffsetStrategyInput(**inputs)
            calculation_trace.append(f"Operation: {offset_input.operation}")

            if offset_input.operation == "plan":
                # Create sample offset projects
                projects = [
                    OffsetProject(
                        project_id="offset_001",
                        name="Verified Forest Conservation",
                        offset_type=OffsetType.NATURE_BASED,
                        standard=OffsetStandard.VERRA_VCS,
                        country="Brazil",
                        available_credits_tco2e=offset_input.target_offset_tco2e * 0.5,
                        price_usd_tco2e=15,
                        quality_rating=QualityRating.HIGH,
                        co_benefits=["Biodiversity", "Community livelihoods"]
                    ),
                    OffsetProject(
                        project_id="offset_002",
                        name="Direct Air Capture Credits",
                        offset_type=OffsetType.CARBON_REMOVAL,
                        standard=OffsetStandard.VERRA_VCS,
                        country="USA",
                        available_credits_tco2e=offset_input.target_offset_tco2e * 0.2,
                        price_usd_tco2e=300,
                        quality_rating=QualityRating.HIGH,
                        additionality_score=1.0,
                        permanence_score=1.0,
                        co_benefits=["Technology development"]
                    ),
                    OffsetProject(
                        project_id="offset_003",
                        name="Renewable Energy Project",
                        offset_type=OffsetType.RENEWABLE_ENERGY,
                        standard=OffsetStandard.GOLD_STANDARD,
                        country="India",
                        available_credits_tco2e=offset_input.target_offset_tco2e * 0.3,
                        price_usd_tco2e=8,
                        quality_rating=QualityRating.MEDIUM,
                        co_benefits=["Energy access", "Local employment"]
                    ),
                ]

                # Calculate totals
                total_credits = sum(p.available_credits_tco2e for p in projects)
                total_cost = sum(p.available_credits_tco2e * p.price_usd_tco2e for p in projects)
                avg_price = total_cost / total_credits if total_credits > 0 else 0

                high_quality = sum(
                    p.available_credits_tco2e for p in projects
                    if p.quality_rating == QualityRating.HIGH
                )
                removal = sum(
                    p.available_credits_tco2e for p in projects
                    if p.offset_type == OffsetType.CARBON_REMOVAL
                )
                nature = sum(
                    p.available_credits_tco2e for p in projects
                    if p.offset_type == OffsetType.NATURE_BASED
                )

                portfolio = OffsetPortfolio(
                    portfolio_id=deterministic_id({"target": offset_input.target_offset_tco2e}, "offport_"),
                    target_offset_tco2e=offset_input.target_offset_tco2e,
                    projects=projects,
                    total_credits_tco2e=total_credits,
                    total_cost_usd=total_cost,
                    average_price_usd_tco2e=avg_price,
                    high_quality_percent=(high_quality / total_credits * 100) if total_credits > 0 else 0,
                    removal_percent=(removal / total_credits * 100) if total_credits > 0 else 0,
                    nature_based_percent=(nature / total_credits * 100) if total_credits > 0 else 0
                )
                portfolio.provenance_hash = content_hash(portfolio.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Created portfolio with {len(projects)} projects")

                self._capture_audit_entry(
                    operation="plan",
                    inputs=inputs,
                    outputs={"projects": len(projects), "total_credits": total_credits},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "plan",
                    "success": True,
                    "portfolio": portfolio.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {offset_input.operation}")

        except Exception as e:
            self.logger.error(f"Planning failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
