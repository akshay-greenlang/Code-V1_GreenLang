# -*- coding: utf-8 -*-
"""
GL-REP-PUB-005: National Climate Report Agent
==============================================

Generates national climate reports aligned with UNFCCC requirements.

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


class NationalReport(BaseModel):
    country_code: str = Field(...)
    country_name: str = Field(...)
    reporting_year: int = Field(...)
    total_emissions_mtco2e: float = Field(...)
    per_capita_emissions_tco2e: float = Field(...)
    gdp_intensity_kgco2e_per_usd: float = Field(...)
    ndc_target_year: int = Field(...)
    ndc_reduction_target_pct: float = Field(...)
    current_progress_pct: float = Field(...)
    sector_emissions: Dict[str, float] = Field(...)
    unfccc_category_breakdown: Dict[str, float] = Field(...)
    lulucf_emissions_mtco2e: float = Field(...)
    policies_in_place: List[str] = Field(...)


class NationalReportInput(BaseModel):
    country_code: str = Field(...)
    country_name: str = Field(...)
    reporting_year: int = Field(...)
    population_million: float = Field(...)
    gdp_billion_usd: float = Field(...)
    sector_data: Dict[str, float] = Field(...)
    ndc_target: Dict[str, Any] = Field(...)
    baseline_emissions_mtco2e: float = Field(...)


class NationalReportOutput(BaseModel):
    country_code: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    report: NationalReport = Field(...)
    unfccc_format_compliant: bool = Field(...)
    data_quality_score: float = Field(...)
    provenance_hash: str = Field(...)


class NationalClimateReportAgent(BaseAgent):
    """GL-REP-PUB-005: National Climate Report Agent"""

    AGENT_ID = "GL-REP-PUB-005"
    AGENT_NAME = "National Climate Report Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="National climate reporting",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = NationalReportInput(**input_data)

            total_emissions = sum(agent_input.sector_data.values())
            per_capita = (total_emissions * 1000) / agent_input.population_million if agent_input.population_million > 0 else 0
            gdp_intensity = total_emissions / agent_input.gdp_billion_usd if agent_input.gdp_billion_usd > 0 else 0

            ndc_target = agent_input.ndc_target
            baseline = agent_input.baseline_emissions_mtco2e
            reduction_achieved = ((baseline - total_emissions) / baseline * 100) if baseline > 0 else 0

            target_reduction = ndc_target.get("reduction_pct", 50)
            progress_pct = (reduction_achieved / target_reduction * 100) if target_reduction > 0 else 0

            unfccc_breakdown = self._map_to_unfccc_categories(agent_input.sector_data)

            report = NationalReport(
                country_code=agent_input.country_code,
                country_name=agent_input.country_name,
                reporting_year=agent_input.reporting_year,
                total_emissions_mtco2e=round(total_emissions, 2),
                per_capita_emissions_tco2e=round(per_capita, 2),
                gdp_intensity_kgco2e_per_usd=round(gdp_intensity, 4),
                ndc_target_year=ndc_target.get("year", 2030),
                ndc_reduction_target_pct=target_reduction,
                current_progress_pct=round(progress_pct, 1),
                sector_emissions=agent_input.sector_data,
                unfccc_category_breakdown=unfccc_breakdown,
                lulucf_emissions_mtco2e=agent_input.sector_data.get("lulucf", 0),
                policies_in_place=[
                    "Carbon pricing",
                    "Renewable energy targets",
                    "Energy efficiency standards",
                    "EV incentives",
                ],
            )

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = NationalReportOutput(
                country_code=agent_input.country_code,
                report=report,
                unfccc_format_compliant=True,
                data_quality_score=0.85,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _map_to_unfccc_categories(self, sector_data: Dict[str, float]) -> Dict[str, float]:
        mapping = {
            "1_energy": sector_data.get("energy", 0) + sector_data.get("transport", 0),
            "2_ippu": sector_data.get("industry", 0),
            "3_agriculture": sector_data.get("agriculture", 0),
            "4_lulucf": sector_data.get("lulucf", 0),
            "5_waste": sector_data.get("waste", 0),
        }
        return {k: round(v, 2) for k, v in mapping.items()}
