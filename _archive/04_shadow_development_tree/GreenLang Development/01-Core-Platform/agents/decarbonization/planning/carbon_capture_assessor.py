# -*- coding: utf-8 -*-
"""
GL-DECARB-X-014: Carbon Capture Assessor Agent
===============================================

Evaluates carbon capture, utilization, and storage (CCUS)
opportunities for industrial facilities.

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


class CaptureType(str, Enum):
    POST_COMBUSTION = "post_combustion"
    PRE_COMBUSTION = "pre_combustion"
    OXYFUEL = "oxyfuel"
    DIRECT_AIR_CAPTURE = "direct_air_capture"


class StorageType(str, Enum):
    GEOLOGICAL = "geological"
    ENHANCED_OIL_RECOVERY = "enhanced_oil_recovery"
    MINERALIZATION = "mineralization"
    UTILIZATION = "utilization"


class CCUSOpportunity(BaseModel):
    opportunity_id: str = Field(...)
    name: str = Field(...)
    capture_type: CaptureType = Field(...)
    storage_type: StorageType = Field(...)

    # Technical
    capture_rate: float = Field(default=0.9, ge=0, le=1, description="Capture efficiency")
    emissions_source_tco2e: float = Field(..., ge=0)
    capture_potential_tco2e: float = Field(..., ge=0)

    # Costs
    capital_cost_usd: float = Field(default=0, ge=0)
    operating_cost_usd_tco2e: float = Field(default=0, ge=0)
    total_cost_usd_tco2e: float = Field(default=0, ge=0)

    # Energy
    energy_penalty_percent: float = Field(default=20, ge=0, le=50)
    additional_energy_mwh: float = Field(default=0, ge=0)

    # Feasibility
    trl: int = Field(default=7, ge=1, le=9)
    is_feasible: bool = Field(default=True)
    barriers: List[str] = Field(default_factory=list)
    storage_available: bool = Field(default=False)
    pipeline_distance_km: Optional[float] = Field(None, ge=0)


class CCUSAssessment(BaseModel):
    assessment_id: str = Field(...)
    facility_name: str = Field(...)
    total_emissions_tco2e: float = Field(..., ge=0)
    opportunities: List[CCUSOpportunity] = Field(default_factory=list)
    total_capture_potential_tco2e: float = Field(default=0, ge=0)
    total_investment_usd: float = Field(default=0, ge=0)
    average_cost_usd_tco2e: float = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")


class CarbonCaptureInput(BaseModel):
    operation: str = Field(default="assess")
    facility_name: str = Field(default="Industrial Facility")
    total_emissions_tco2e: float = Field(default=100000, ge=0)
    emission_sources: List[Dict[str, Any]] = Field(default_factory=list)
    has_storage_access: bool = Field(default=False)
    pipeline_distance_km: Optional[float] = Field(None, ge=0)


class CarbonCaptureOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    assessment: Optional[CCUSAssessment] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class CarbonCaptureAssessor(DeterministicAgent):
    """GL-DECARB-X-014: Carbon Capture Assessor Agent"""

    AGENT_ID = "GL-DECARB-X-014"
    AGENT_NAME = "Carbon Capture Assessor"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="CarbonCaptureAssessor",
        category=AgentCategory.CRITICAL,
        description="Evaluates CCUS opportunities"
    )

    # Cost estimates per capture type ($/tCO2e)
    CAPTURE_COSTS = {
        CaptureType.POST_COMBUSTION: {"capital": 80, "operating": 40},
        CaptureType.PRE_COMBUSTION: {"capital": 70, "operating": 35},
        CaptureType.OXYFUEL: {"capital": 90, "operating": 50},
        CaptureType.DIRECT_AIR_CAPTURE: {"capital": 200, "operating": 150},
    }

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Evaluates CCUS opportunities", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            cc_input = CarbonCaptureInput(**inputs)
            calculation_trace.append(f"Operation: {cc_input.operation}")

            if cc_input.operation == "assess":
                opportunities = []

                # Post-combustion capture opportunity
                post_comb = self._assess_capture_option(
                    CaptureType.POST_COMBUSTION,
                    cc_input.total_emissions_tco2e,
                    cc_input.has_storage_access,
                    cc_input.pipeline_distance_km
                )
                opportunities.append(post_comb)

                # DAC opportunity (if suitable)
                if cc_input.total_emissions_tco2e < 50000:
                    dac = self._assess_capture_option(
                        CaptureType.DIRECT_AIR_CAPTURE,
                        cc_input.total_emissions_tco2e * 0.1,
                        cc_input.has_storage_access,
                        cc_input.pipeline_distance_km
                    )
                    opportunities.append(dac)

                # Sort by cost
                opportunities.sort(key=lambda o: o.total_cost_usd_tco2e)

                assessment = CCUSAssessment(
                    assessment_id=deterministic_id({"facility": cc_input.facility_name}, "ccus_"),
                    facility_name=cc_input.facility_name,
                    total_emissions_tco2e=cc_input.total_emissions_tco2e,
                    opportunities=opportunities,
                    total_capture_potential_tco2e=sum(o.capture_potential_tco2e for o in opportunities),
                    total_investment_usd=sum(o.capital_cost_usd for o in opportunities),
                    average_cost_usd_tco2e=sum(o.total_cost_usd_tco2e for o in opportunities) / len(opportunities) if opportunities else 0
                )
                assessment.provenance_hash = content_hash(assessment.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Assessed {len(opportunities)} CCUS opportunities")

                self._capture_audit_entry(
                    operation="assess",
                    inputs=inputs,
                    outputs={"opportunities": len(opportunities)},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "assess",
                    "success": True,
                    "assessment": assessment.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {cc_input.operation}")

        except Exception as e:
            self.logger.error(f"Assessment failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }

    def _assess_capture_option(
        self,
        capture_type: CaptureType,
        emissions: float,
        storage_available: bool,
        pipeline_km: Optional[float]
    ) -> CCUSOpportunity:
        """Assess a specific capture option."""
        costs = self.CAPTURE_COSTS[capture_type]
        capture_rate = 0.9 if capture_type != CaptureType.DIRECT_AIR_CAPTURE else 0.95
        capture_potential = emissions * capture_rate

        capital_cost = capture_potential * costs["capital"]
        operating_cost = costs["operating"]

        # Add transport costs
        if pipeline_km:
            operating_cost += pipeline_km * 0.05  # $0.05/tCO2e per km

        total_cost = costs["capital"] / 20 + operating_cost  # Amortized over 20 years

        barriers = []
        if not storage_available:
            barriers.append("No storage site access")
        if pipeline_km and pipeline_km > 100:
            barriers.append("Long transport distance")
        if capture_type == CaptureType.DIRECT_AIR_CAPTURE:
            barriers.append("High cost")

        return CCUSOpportunity(
            opportunity_id=deterministic_id({"type": capture_type.value}, "ccopt_"),
            name=f"{capture_type.value.replace('_', ' ').title()} Capture",
            capture_type=capture_type,
            storage_type=StorageType.GEOLOGICAL if storage_available else StorageType.UTILIZATION,
            capture_rate=capture_rate,
            emissions_source_tco2e=emissions,
            capture_potential_tco2e=capture_potential,
            capital_cost_usd=capital_cost,
            operating_cost_usd_tco2e=operating_cost,
            total_cost_usd_tco2e=total_cost,
            trl=8 if capture_type == CaptureType.POST_COMBUSTION else 6,
            is_feasible=storage_available or capture_type == CaptureType.DIRECT_AIR_CAPTURE,
            barriers=barriers,
            storage_available=storage_available,
            pipeline_distance_km=pipeline_km
        )
