# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-002: Reforestation Planner Agent
==============================================

Plans forest restoration projects on previously forested land.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class DegradationType(str, Enum):
    """Type of forest degradation."""
    CLEAR_CUT = "clear_cut"
    SELECTIVE_LOGGING = "selective_logging"
    FIRE_DAMAGE = "fire_damage"
    PEST_DAMAGE = "pest_damage"
    NATURAL_DISASTER = "natural_disaster"


class RestorationMethod(str, Enum):
    """Restoration method options."""
    NATURAL_REGENERATION = "natural_regeneration"
    ASSISTED_REGENERATION = "assisted_regeneration"
    ACTIVE_PLANTING = "active_planting"
    ENRICHMENT_PLANTING = "enrichment_planting"


class ReforestationSite(BaseModel):
    """Reforestation site assessment."""
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    degradation_type: DegradationType = Field(...)
    years_since_degradation: int = Field(..., ge=0)
    remaining_tree_cover_percent: float = Field(default=0, ge=0, le=100)
    seed_tree_presence: bool = Field(default=False)
    invasive_species_present: bool = Field(default=False)


class ReforestationPlan(BaseModel):
    """Reforestation plan for a site."""
    site_id: str = Field(...)
    recommended_method: RestorationMethod = Field(...)
    intervention_intensity: str = Field(...)  # low, medium, high

    # Carbon projections
    baseline_carbon_tonnes_c: float = Field(...)
    projected_carbon_30yr_tonnes_c: float = Field(...)
    net_sequestration_tonnes_co2e: float = Field(...)

    # Costs
    total_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)

    # Timeline
    recovery_timeline_years: int = Field(...)
    monitoring_frequency: str = Field(...)


class ReforestationInput(BaseModel):
    """Input for Reforestation Planner."""
    project_id: str = Field(...)
    sites: List[ReforestationSite] = Field(..., min_length=1)
    target_forest_type: str = Field(default="native_mixed")
    project_duration_years: int = Field(default=30)


class ReforestationOutput(BaseModel):
    """Output from Reforestation Planner."""
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_net_sequestration_co2e: float = Field(...)
    total_project_cost_usd: float = Field(...)
    average_cost_per_tonne: float = Field(...)
    site_plans: List[ReforestationPlan] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class ReforestationPlannerAgent(BaseAgent):
    """
    GL-DECARB-NBS-002: Reforestation Planner Agent

    Plans forest restoration projects with method selection and cost estimation.
    """

    AGENT_ID = "GL-DECARB-NBS-002"
    AGENT_NAME = "Reforestation Planner Agent"
    VERSION = "1.0.0"
    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Forest restoration planning",
                version=self.VERSION,
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute reforestation planning."""
        try:
            agent_input = ReforestationInput(**input_data)
            site_plans = []
            warnings = []

            for site in agent_input.sites:
                plan = self._create_site_plan(site, agent_input.project_duration_years)
                site_plans.append(plan)

            total_area = sum(s.area_ha for s in agent_input.sites)
            total_seq = sum(p.net_sequestration_tonnes_co2e for p in site_plans)
            total_cost = sum(p.total_cost_usd for p in site_plans)
            avg_cost = total_cost / total_seq if total_seq > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = ReforestationOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_net_sequestration_co2e=total_seq,
                total_project_cost_usd=total_cost,
                average_cost_per_tonne=avg_cost,
                site_plans=site_plans,
                provenance_hash=provenance_hash,
                warnings=warnings
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            logger.error(f"Reforestation planning failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _create_site_plan(self, site: ReforestationSite, duration: int) -> ReforestationPlan:
        """Create plan for a site."""

        # Determine method based on site conditions
        if site.remaining_tree_cover_percent > 30 and site.seed_tree_presence:
            method = RestorationMethod.NATURAL_REGENERATION
            intensity = "low"
            cost_per_ha = 200
            recovery_years = 15
        elif site.remaining_tree_cover_percent > 10:
            method = RestorationMethod.ASSISTED_REGENERATION
            intensity = "medium"
            cost_per_ha = 600
            recovery_years = 12
        else:
            method = RestorationMethod.ACTIVE_PLANTING
            intensity = "high"
            cost_per_ha = 1200
            recovery_years = 10

        if site.invasive_species_present:
            cost_per_ha += 400

        # Carbon calculations
        baseline_c = site.remaining_tree_cover_percent * 2 * site.area_ha  # Simplified
        mature_forest_c = 150 * site.area_ha  # t C/ha at maturity
        projected_c = baseline_c + (mature_forest_c - baseline_c) * min(1, duration / recovery_years)
        net_seq_c = projected_c - baseline_c
        net_seq_co2e = net_seq_c * self.CO2_TO_C_RATIO

        total_cost = cost_per_ha * site.area_ha
        cost_per_tonne = total_cost / net_seq_co2e if net_seq_co2e > 0 else 0

        return ReforestationPlan(
            site_id=site.site_id,
            recommended_method=method,
            intervention_intensity=intensity,
            baseline_carbon_tonnes_c=round(baseline_c, 2),
            projected_carbon_30yr_tonnes_c=round(projected_c, 2),
            net_sequestration_tonnes_co2e=round(net_seq_co2e, 2),
            total_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2),
            recovery_timeline_years=recovery_years,
            monitoring_frequency="Annual" if intensity == "high" else "Biennial"
        )
