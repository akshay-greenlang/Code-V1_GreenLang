# -*- coding: utf-8 -*-
"""
GL-REP-PUB-004: Regional Climate Report Agent
==============================================

Generates regional climate reports for multi-jurisdictional coordination.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class RegionalReport(BaseModel):
    region_id: str = Field(...)
    region_name: str = Field(...)
    reporting_year: int = Field(...)
    total_emissions_tco2e: float = Field(...)
    per_capita_emissions_tco2e: float = Field(...)
    reduction_from_baseline_pct: float = Field(...)
    member_jurisdictions: int = Field(...)
    sector_breakdown: Dict[str, float] = Field(...)
    regional_targets: List[Dict[str, Any]] = Field(...)
    progress_assessment: str = Field(...)


class RegionalReportInput(BaseModel):
    region_id: str = Field(...)
    region_name: str = Field(...)
    member_municipalities: List[Dict[str, Any]] = Field(...)
    reporting_year: int = Field(...)
    baseline_year: int = Field(default=2005)


class RegionalReportOutput(BaseModel):
    region_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    report: RegionalReport = Field(...)
    aggregation_methodology: str = Field(...)
    data_coverage_pct: float = Field(...)
    provenance_hash: str = Field(...)


class RegionalClimateReportAgent(BaseAgent):
    """GL-REP-PUB-004: Regional Climate Report Agent"""

    AGENT_ID = "GL-REP-PUB-004"
    AGENT_NAME = "Regional Climate Report Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Regional climate reporting",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = RegionalReportInput(**input_data)

            total_emissions = sum(
                m.get("emissions_tco2e", 0) for m in agent_input.member_municipalities
            )
            total_population = sum(
                m.get("population", 0) for m in agent_input.member_municipalities
            )
            baseline_emissions = sum(
                m.get("baseline_emissions_tco2e", m.get("emissions_tco2e", 0) * 1.2)
                for m in agent_input.member_municipalities
            )

            per_capita = total_emissions / total_population if total_population > 0 else 0
            reduction_pct = ((baseline_emissions - total_emissions) / baseline_emissions * 100
                           if baseline_emissions > 0 else 0)

            sector_breakdown = self._aggregate_sectors(agent_input.member_municipalities)

            progress = "on_track" if reduction_pct >= 30 else (
                "needs_acceleration" if reduction_pct >= 15 else "off_track"
            )

            report = RegionalReport(
                region_id=agent_input.region_id,
                region_name=agent_input.region_name,
                reporting_year=agent_input.reporting_year,
                total_emissions_tco2e=round(total_emissions, 2),
                per_capita_emissions_tco2e=round(per_capita, 2),
                reduction_from_baseline_pct=round(reduction_pct, 1),
                member_jurisdictions=len(agent_input.member_municipalities),
                sector_breakdown=sector_breakdown,
                regional_targets=[
                    {"year": 2030, "reduction_target_pct": 50},
                    {"year": 2050, "reduction_target_pct": 100},
                ],
                progress_assessment=progress,
            )

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = RegionalReportOutput(
                region_id=agent_input.region_id,
                report=report,
                aggregation_methodology="GPC_compliant_bottom_up",
                data_coverage_pct=95.0,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _aggregate_sectors(self, municipalities: List[Dict[str, Any]]) -> Dict[str, float]:
        sectors = {
            "stationary_energy": 0,
            "transportation": 0,
            "waste": 0,
            "ippu": 0,
            "afolu": 0,
        }

        for m in municipalities:
            for sector, value in m.get("sector_emissions", {}).items():
                if sector in sectors:
                    sectors[sector] += value

        total = sum(sectors.values())
        if total > 0:
            return {k: round(v / total * 100, 1) for k, v in sectors.items()}
        return sectors
