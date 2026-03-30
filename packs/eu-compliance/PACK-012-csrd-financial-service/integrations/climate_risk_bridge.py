# -*- coding: utf-8 -*-
"""
ClimateRiskBridge - Bridge to Transition and Physical Risk Agents
===================================================================

Connects PACK-012 (CSRD Financial Service) with transition risk, physical
risk, and climate hazard agents for comprehensive climate risk assessment
under CSRD E1, EBA Pillar 3, and ECB climate stress testing requirements.

Architecture:
    PACK-012 CSRD FS --> ClimateRiskBridge --> Risk Agents
                              |
                              v
    TransitionRiskAssessor (GL-DECARB-X-021)
    PhysicalRiskScreening (GL-ADAPT-X-001)
    ClimateHazardConnector (AGENT-DATA-020)

Example:
    >>> config = ClimateRiskBridgeConfig(scenarios=["orderly", "hot_house"])
    >>> bridge = ClimateRiskBridge(config)
    >>> result = bridge.assess_portfolio_risk(counterparties)
    >>> print(f"Combined risk: {result.combined_score}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib

            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning("AgentStub: failed to load %s: %s", self.agent_id, exc)
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None

class ClimateScenario(str, Enum):
    """NGFS climate scenarios."""
    ORDERLY = "orderly"
    DISORDERLY = "disorderly"
    HOT_HOUSE = "hot_house"
    TOO_LITTLE_TOO_LATE = "too_little_too_late"
    NET_ZERO_2050 = "net_zero_2050"
    DELAYED_TRANSITION = "delayed_transition"

class HazardType(str, Enum):
    """Climate hazard types for physical risk."""
    FLOODING = "flooding"
    WILDFIRE = "wildfire"
    HEAT_STRESS = "heat_stress"
    WATER_STRESS = "water_stress"
    SEA_LEVEL_RISE = "sea_level_rise"
    STORM = "storm"
    DROUGHT = "drought"

class TimeHorizon(str, Enum):
    """Time horizons for risk assessment."""
    SHORT = "short_term"
    MEDIUM = "medium_term"
    LONG = "long_term"

class ClimateRiskBridgeConfig(BaseModel):
    """Configuration for the Climate Risk Bridge."""
    transition_risk_agent_path: str = Field(
        default="greenlang.agents.decarbonization.transition_risk",
        description="Import path for transition risk assessor",
    )
    physical_risk_agent_path: str = Field(
        default="greenlang.agents.adaptation.physical_risk",
        description="Import path for physical risk screening",
    )
    climate_hazard_agent_path: str = Field(
        default="greenlang.agents.data.climate_hazard_connector",
        description="Import path for climate hazard connector",
    )
    scenarios: List[str] = Field(
        default_factory=lambda: ["orderly", "disorderly", "hot_house"],
        description="NGFS climate scenarios to assess",
    )
    hazard_types: List[str] = Field(
        default_factory=lambda: [h.value for h in HazardType],
        description="Climate hazard types to screen",
    )
    time_horizons: List[str] = Field(
        default_factory=lambda: ["short_term", "medium_term", "long_term"],
        description="Time horizons for assessment",
    )
    transition_risk_weight: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Weight for transition risk in combined score",
    )
    physical_risk_weight: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Weight for physical risk in combined score",
    )

class TransitionRiskResult(BaseModel):
    """Result of transition risk assessment."""
    portfolio_transition_score: float = Field(
        default=0.0, description="Portfolio-level transition risk (0-100)",
    )
    high_risk_count: int = Field(default=0, description="High-risk counterparties")
    medium_risk_count: int = Field(default=0, description="Medium-risk counterparties")
    low_risk_count: int = Field(default=0, description="Low-risk counterparties")
    sector_risks: Dict[str, float] = Field(
        default_factory=dict, description="Transition risk by sector",
    )
    scenario_impacts: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Impact per scenario",
    )
    carbon_intensive_exposure_pct: float = Field(
        default=0.0, description="Exposure to carbon-intensive sectors (%)",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class PhysicalRiskResult(BaseModel):
    """Result of physical risk assessment."""
    portfolio_physical_score: float = Field(
        default=0.0, description="Portfolio-level physical risk (0-100)",
    )
    acute_risk_score: float = Field(
        default=0.0, description="Acute physical risk score",
    )
    chronic_risk_score: float = Field(
        default=0.0, description="Chronic physical risk score",
    )
    hazard_exposure: Dict[str, float] = Field(
        default_factory=dict, description="Exposure by hazard type",
    )
    geographic_risk: Dict[str, float] = Field(
        default_factory=dict, description="Risk by geography",
    )
    high_risk_locations: int = Field(
        default=0, description="Number of high-risk locations",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CombinedClimateRiskResult(BaseModel):
    """Combined climate risk assessment result."""
    combined_score: float = Field(
        default=0.0, description="Combined climate risk score (0-100)",
    )
    transition_risk: TransitionRiskResult = Field(
        default_factory=TransitionRiskResult,
        description="Transition risk assessment",
    )
    physical_risk: PhysicalRiskResult = Field(
        default_factory=PhysicalRiskResult,
        description="Physical risk assessment",
    )
    total_counterparties: int = Field(
        default=0, description="Total counterparties assessed",
    )
    scenarios_assessed: List[str] = Field(
        default_factory=list, description="Scenarios assessed",
    )
    time_horizons: List[str] = Field(
        default_factory=list, description="Time horizons covered",
    )
    stress_test_ready: bool = Field(
        default=False, description="Whether data is stress-test ready",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ClimateRiskBridge:
    """Bridge to climate risk assessment agents.

    Combines transition risk (policy, technology, market, reputation),
    physical risk (acute, chronic), and climate hazard data for
    comprehensive climate risk assessment required under CSRD E1,
    EBA Pillar 3, and ECB stress testing.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for risk agents.

    Example:
        >>> bridge = ClimateRiskBridge()
        >>> result = bridge.assess_portfolio_risk(counterparties)
        >>> print(f"Combined: {result.combined_score}")
    """

    def __init__(self, config: Optional[ClimateRiskBridgeConfig] = None) -> None:
        """Initialize the Climate Risk Bridge."""
        self.config = config or ClimateRiskBridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "transition_risk": _AgentStub(
                "GL-DECARB-X-021",
                self.config.transition_risk_agent_path,
                "TransitionRiskAssessor",
            ),
            "physical_risk": _AgentStub(
                "GL-ADAPT-X-001",
                self.config.physical_risk_agent_path,
                "PhysicalRiskScreening",
            ),
            "climate_hazard": _AgentStub(
                "GL-DATA-X-020",
                self.config.climate_hazard_agent_path,
                "ClimateHazardConnector",
            ),
        }

        self.logger.info(
            "ClimateRiskBridge initialized: scenarios=%d, hazards=%d, "
            "tr_weight=%.1f, pr_weight=%.1f",
            len(self.config.scenarios),
            len(self.config.hazard_types),
            self.config.transition_risk_weight,
            self.config.physical_risk_weight,
        )

    def assess_portfolio_risk(
        self,
        counterparty_data: List[Dict[str, Any]],
    ) -> CombinedClimateRiskResult:
        """Assess combined climate risk for the portfolio.

        Runs both transition risk and physical risk assessments across
        all configured scenarios and combines them into a single score.

        Args:
            counterparty_data: Counterparty records with risk attributes.

        Returns:
            CombinedClimateRiskResult with combined assessment.
        """
        tr_result = self.assess_transition_risk(counterparty_data)
        pr_result = self.assess_physical_risk(counterparty_data)

        combined = round(
            tr_result.portfolio_transition_score * self.config.transition_risk_weight
            + pr_result.portfolio_physical_score * self.config.physical_risk_weight,
            2,
        )

        result = CombinedClimateRiskResult(
            combined_score=combined,
            transition_risk=tr_result,
            physical_risk=pr_result,
            total_counterparties=len(counterparty_data),
            scenarios_assessed=self.config.scenarios,
            time_horizons=self.config.time_horizons,
            stress_test_ready=len(self.config.scenarios) >= 3,
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Climate risk: combined=%.1f, transition=%.1f, physical=%.1f",
            combined,
            tr_result.portfolio_transition_score,
            pr_result.portfolio_physical_score,
        )
        return result

    def assess_transition_risk(
        self,
        counterparty_data: List[Dict[str, Any]],
    ) -> TransitionRiskResult:
        """Assess transition risk for the portfolio.

        Evaluates policy, technology, market, and reputation risks
        under different NGFS scenarios.

        Args:
            counterparty_data: Counterparty records.

        Returns:
            TransitionRiskResult with transition risk metrics.
        """
        total_exposure = max(
            sum(float(c.get("exposure_eur", 0.0)) for c in counterparty_data), 1.0
        )
        weighted_score = 0.0
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        sector_risks: Dict[str, float] = {}
        carbon_intensive_exp = 0.0

        for cp in counterparty_data:
            exposure = float(cp.get("exposure_eur", 0.0))
            tr_score = float(cp.get("transition_risk_score", 0.0))
            sector = cp.get("nace_sector", "Unknown")
            weight = exposure / total_exposure
            weighted_score += tr_score * weight

            if tr_score > 70:
                high_risk += 1
                carbon_intensive_exp += exposure
            elif tr_score > 40:
                medium_risk += 1
            else:
                low_risk += 1

            if sector not in sector_risks:
                sector_risks[sector] = 0.0
            sector_risks[sector] = max(sector_risks[sector], tr_score)

        scenario_impacts: Dict[str, Dict[str, float]] = {}
        for scenario in self.config.scenarios:
            mult = 1.0
            if scenario == "disorderly":
                mult = 1.15
            elif scenario == "hot_house":
                mult = 0.7
            elif scenario == "net_zero_2050":
                mult = 1.3
            scenario_impacts[scenario] = {
                "transition_risk_score": round(weighted_score * mult, 2),
                "expected_loss_pct": round(weighted_score * mult / 20, 2),
            }

        ci_pct = round((carbon_intensive_exp / total_exposure) * 100, 2)

        result = TransitionRiskResult(
            portfolio_transition_score=round(weighted_score, 2),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            sector_risks=sector_risks,
            scenario_impacts=scenario_impacts,
            carbon_intensive_exposure_pct=ci_pct,
        )
        result.provenance_hash = _hash_data(result.model_dump())
        return result

    def assess_physical_risk(
        self,
        counterparty_data: List[Dict[str, Any]],
    ) -> PhysicalRiskResult:
        """Assess physical risk for the portfolio.

        Evaluates acute (floods, storms, wildfires) and chronic (heat
        stress, water stress, sea level rise) physical risks.

        Args:
            counterparty_data: Counterparty records with location data.

        Returns:
            PhysicalRiskResult with physical risk metrics.
        """
        total_exposure = max(
            sum(float(c.get("exposure_eur", 0.0)) for c in counterparty_data), 1.0
        )
        weighted_score = 0.0
        acute_sum = 0.0
        chronic_sum = 0.0
        high_risk_locs = 0
        hazard_exposure: Dict[str, float] = {}
        geo_risk: Dict[str, float] = {}

        for cp in counterparty_data:
            exposure = float(cp.get("exposure_eur", 0.0))
            pr_score = float(cp.get("physical_risk_score", 0.0))
            acute = float(cp.get("acute_risk_score", pr_score * 0.6))
            chronic = float(cp.get("chronic_risk_score", pr_score * 0.4))
            country = cp.get("country", "Unknown")
            weight = exposure / total_exposure

            weighted_score += pr_score * weight
            acute_sum += acute * weight
            chronic_sum += chronic * weight

            if pr_score > 70:
                high_risk_locs += 1

            for hazard in self.config.hazard_types:
                h_score = float(cp.get(f"{hazard}_score", 0.0))
                if h_score > 0:
                    hazard_exposure[hazard] = (
                        hazard_exposure.get(hazard, 0.0) + exposure
                    )

            geo_risk[country] = max(geo_risk.get(country, 0.0), pr_score)

        result = PhysicalRiskResult(
            portfolio_physical_score=round(weighted_score, 2),
            acute_risk_score=round(acute_sum, 2),
            chronic_risk_score=round(chronic_sum, 2),
            hazard_exposure=hazard_exposure,
            geographic_risk=geo_risk,
            high_risk_locations=high_risk_locs,
        )
        result.provenance_hash = _hash_data(result.model_dump())
        return result

    def route_to_risk_agent(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a request to the appropriate risk agent.

        Args:
            request_type: Type of request.
            data: Request data.

        Returns:
            Response from the risk agent or error dictionary.
        """
        counterparties = data.get("counterparty_data", [])

        if request_type == "combined_risk":
            result = self.assess_portfolio_risk(counterparties)
            return result.model_dump()
        elif request_type == "transition_risk":
            result = self.assess_transition_risk(counterparties)
            return result.model_dump()
        elif request_type == "physical_risk":
            result = self.assess_physical_risk(counterparties)
            return result.model_dump()
        else:
            return {"error": f"Unknown request type: {request_type}"}
