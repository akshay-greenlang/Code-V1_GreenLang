# -*- coding: utf-8 -*-
"""
GL-ADAPT-WAT-005: Groundwater Management Agent
=============================================

Adaptation agent for aquifer and groundwater management.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AquiferType(str, Enum):
    UNCONFINED = "unconfined"
    CONFINED = "confined"
    SEMI_CONFINED = "semi_confined"


class AquiferCondition(str, Enum):
    SUSTAINABLE = "sustainable"
    STRESSED = "stressed"
    OVEREXPLOITED = "overexploited"
    CRITICAL = "critical"


class AquiferStatus(BaseModel):
    """Aquifer status assessment."""
    aquifer_id: str
    aquifer_name: str
    aquifer_type: AquiferType
    annual_recharge_m3: float
    annual_abstraction_m3: float
    storage_change_m3_year: float
    water_table_change_m_year: float
    condition: AquiferCondition
    years_to_depletion: Optional[float] = None


class Well(BaseModel):
    """Groundwater well."""
    well_id: str
    aquifer_id: str
    permitted_abstraction_m3_year: float
    actual_abstraction_m3_year: float
    water_level_m: float
    water_level_trend: str  # rising, stable, declining


class RechargeAssessment(BaseModel):
    """Natural and artificial recharge assessment."""
    aquifer_id: str
    natural_recharge_m3_year: float
    artificial_recharge_potential_m3_year: float
    recharge_zones_area_km2: float
    urbanization_impact_percent: float
    recommended_mar_capacity_m3_year: float


class GroundwaterInput(BaseModel):
    """Input for groundwater management analysis."""
    region_id: str
    aquifers: List[AquiferStatus]
    wells: List[Well] = Field(default_factory=list)
    precipitation_mm_year: float = Field(default=500)
    climate_change_precipitation_factor: float = Field(default=0.95)


class ManagementAction(BaseModel):
    """Groundwater management action."""
    action_id: str
    aquifer_id: str
    action_type: str
    description: str
    expected_impact_m3_year: float
    estimated_cost: float
    priority: str


class GroundwaterOutput(BaseModel):
    """Output from groundwater management analysis."""
    region_id: str
    total_recharge_m3_year: float
    total_abstraction_m3_year: float
    overall_balance_m3_year: float
    aquifers_overexploited: int
    aquifer_assessments: List[AquiferStatus]
    recharge_assessments: List[RechargeAssessment]
    management_actions: List[ManagementAction]
    sustainability_outlook: str
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class GroundwaterManagementAgent(BaseAgent):
    """
    GL-ADAPT-WAT-005: Groundwater Management Agent

    Manages groundwater resources and aquifer sustainability.
    """

    AGENT_ID = "GL-ADAPT-WAT-005"
    AGENT_NAME = "Groundwater Management Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Groundwater and aquifer management",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            gw_input = GroundwaterInput(**input_data)
            recharge_assessments = []
            management_actions = []
            action_id = 1
            overexploited_count = 0

            total_recharge = 0.0
            total_abstraction = 0.0

            for aquifer in gw_input.aquifers:
                total_recharge += aquifer.annual_recharge_m3
                total_abstraction += aquifer.annual_abstraction_m3

                if aquifer.condition in [AquiferCondition.OVEREXPLOITED, AquiferCondition.CRITICAL]:
                    overexploited_count += 1

                # Calculate recharge assessment
                # Estimate natural recharge from precipitation
                recharge_efficiency = 0.15 if aquifer.aquifer_type == AquiferType.UNCONFINED else 0.05
                natural_recharge = gw_input.precipitation_mm_year * 1000 * recharge_efficiency

                # MAR potential
                mar_potential = aquifer.annual_abstraction_m3 * 0.3 if aquifer.condition != AquiferCondition.SUSTAINABLE else 0

                # Urbanization impact
                urban_impact = 30 if aquifer.aquifer_type == AquiferType.UNCONFINED else 10

                recharge = RechargeAssessment(
                    aquifer_id=aquifer.aquifer_id,
                    natural_recharge_m3_year=round(natural_recharge, 0),
                    artificial_recharge_potential_m3_year=round(mar_potential, 0),
                    recharge_zones_area_km2=100,  # Placeholder
                    urbanization_impact_percent=urban_impact,
                    recommended_mar_capacity_m3_year=round(mar_potential * 0.5, 0),
                )
                recharge_assessments.append(recharge)

                # Generate management actions
                if aquifer.condition == AquiferCondition.CRITICAL:
                    management_actions.append(ManagementAction(
                        action_id=f"GW-{action_id}",
                        aquifer_id=aquifer.aquifer_id,
                        action_type="abstraction_reduction",
                        description=f"Mandatory abstraction reduction for {aquifer.aquifer_name}",
                        expected_impact_m3_year=aquifer.annual_abstraction_m3 * 0.3,
                        estimated_cost=1000000,
                        priority="immediate",
                    ))
                    action_id += 1

                if aquifer.condition in [AquiferCondition.OVEREXPLOITED, AquiferCondition.STRESSED]:
                    management_actions.append(ManagementAction(
                        action_id=f"GW-{action_id}",
                        aquifer_id=aquifer.aquifer_id,
                        action_type="managed_aquifer_recharge",
                        description=f"Implement MAR scheme for {aquifer.aquifer_name}",
                        expected_impact_m3_year=mar_potential,
                        estimated_cost=5000000,
                        priority="high",
                    ))
                    action_id += 1

            # Overall balance
            overall_balance = total_recharge - total_abstraction

            # Sustainability outlook
            if overall_balance > 0:
                outlook = "Sustainable - abstraction within recharge limits"
            elif overall_balance > -total_recharge * 0.1:
                outlook = "Marginally sustainable - minor deficit"
            elif overall_balance > -total_recharge * 0.3:
                outlook = "Unsustainable - significant deficit requiring action"
            else:
                outlook = "Critical - severe overexploitation requiring immediate intervention"

            # Add general recommendations
            if overexploited_count > 0:
                management_actions.append(ManagementAction(
                    action_id=f"GW-{action_id}",
                    aquifer_id="region_wide",
                    action_type="monitoring",
                    description="Enhance groundwater monitoring network",
                    expected_impact_m3_year=0,
                    estimated_cost=500000,
                    priority="high",
                ))

            provenance_hash = hashlib.sha256(
                json.dumps({"region": gw_input.region_id, "balance": overall_balance}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = GroundwaterOutput(
                region_id=gw_input.region_id,
                total_recharge_m3_year=round(total_recharge, 0),
                total_abstraction_m3_year=round(total_abstraction, 0),
                overall_balance_m3_year=round(overall_balance, 0),
                aquifers_overexploited=overexploited_count,
                aquifer_assessments=gw_input.aquifers,
                recharge_assessments=recharge_assessments,
                management_actions=management_actions,
                sustainability_outlook=outlook,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Groundwater management analysis failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
