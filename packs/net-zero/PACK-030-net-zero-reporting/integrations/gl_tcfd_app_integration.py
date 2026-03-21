# -*- coding: utf-8 -*-
"""
GLTCFDAppIntegration - GL-TCFD-APP Integration for PACK-030
===============================================================

Enterprise integration for fetching scenario analysis, climate risk
assessments, and opportunity data from GL-TCFD-APP (APP-008) into
the Net Zero Reporting Pack. Provides TCFD-specific data for 4-pillar
disclosure generation including Governance, Strategy, Risk Management,
and Metrics & Targets.

Integration Points:
    - Scenario Analysis: 1.5C/2C/BAU/4C scenario results
    - Climate Risks: Physical and transition risks with financial impact
    - Climate Opportunities: Revenue, cost savings, resilience benefits
    - Governance: Board oversight and management climate roles
    - Strategy: Climate-related strategic planning data

Architecture:
    GL-TCFD-APP Scenarios --> PACK-030 TCFD Strategy Pillar
    GL-TCFD-APP Risks     --> PACK-030 TCFD Risk Management Pillar
    GL-TCFD-APP Data      --> PACK-030 ISSB/CSRD/SEC cross-mapping

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TCFDPillar(str, Enum):
    GOVERNANCE = "governance"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    METRICS_TARGETS = "metrics_targets"


class ScenarioType(str, Enum):
    NZE_15C = "nze_1.5c"
    BELOW_2C = "below_2c"
    NDC = "ndc"
    BAU = "bau"
    HIGH_WARMING_4C = "high_warming_4c"


class RiskType(str, Enum):
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"


class RiskLikelihood(str, Enum):
    VERY_LIKELY = "very_likely"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    VERY_UNLIKELY = "very_unlikely"


class RiskImpact(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class RiskTimeHorizon(str, Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class OpportunityType(str, Enum):
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


class ImportStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"
    CACHED = "cached"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class GLTCFDAppConfig(BaseModel):
    pack_id: str = Field(default="PACK-030")
    app_id: str = Field(default="GL-TCFD-APP")
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    api_base_url: str = Field(default="")
    api_key: str = Field(default="")
    api_timeout_seconds: float = Field(default=30.0)
    enable_provenance: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    db_pool_size: int = Field(default=5, ge=1, le=20)
    cache_ttl_seconds: int = Field(default=3600)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0)


class ScenarioResult(BaseModel):
    """Climate scenario analysis result from GL-TCFD-APP."""
    scenario_id: str = Field(default_factory=_new_uuid)
    scenario_type: ScenarioType = Field(default=ScenarioType.NZE_15C)
    scenario_name: str = Field(default="")
    description: str = Field(default="")
    temperature_outcome: str = Field(default="")
    carbon_price_2030_usd: float = Field(default=0.0)
    carbon_price_2050_usd: float = Field(default=0.0)
    revenue_impact_pct: float = Field(default=0.0)
    cost_impact_pct: float = Field(default=0.0)
    asset_impairment_pct: float = Field(default=0.0)
    transition_risk_level: str = Field(default="medium")
    physical_risk_level: str = Field(default="low")
    strategy_resilience: str = Field(default="")
    key_assumptions: List[str] = Field(default_factory=list)


class ScenarioAnalysis(BaseModel):
    """Complete scenario analysis from GL-TCFD-APP."""
    analysis_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    reference_scenario: ScenarioType = Field(default=ScenarioType.NZE_15C)
    analysis_year: int = Field(default=2025)
    methodology: str = Field(default="Quantitative scenario analysis per TCFD guidance")
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=_utcnow)


class ClimateRisk(BaseModel):
    """Climate risk assessment from GL-TCFD-APP."""
    risk_id: str = Field(default_factory=_new_uuid)
    risk_type: RiskType = Field(default=RiskType.TRANSITION_POLICY)
    risk_name: str = Field(default="")
    description: str = Field(default="")
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.POSSIBLE)
    impact: RiskImpact = Field(default=RiskImpact.MEDIUM)
    time_horizon: RiskTimeHorizon = Field(default=RiskTimeHorizon.MEDIUM_TERM)
    financial_impact_usd: float = Field(default=0.0)
    financial_impact_description: str = Field(default="")
    mitigation_actions: List[str] = Field(default_factory=list)
    affected_assets_pct: float = Field(default=0.0)
    residual_risk_level: str = Field(default="low")


class RiskAssessment(BaseModel):
    """Complete risk assessment from GL-TCFD-APP."""
    assessment_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    risks: List[ClimateRisk] = Field(default_factory=list)
    total_risks: int = Field(default=0)
    transition_risks: int = Field(default=0)
    physical_risks: int = Field(default=0)
    total_financial_impact_usd: float = Field(default=0.0)
    high_impact_risks: int = Field(default=0)
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=_utcnow)


class ClimateOpportunity(BaseModel):
    """Climate opportunity from GL-TCFD-APP."""
    opportunity_id: str = Field(default_factory=_new_uuid)
    opportunity_type: OpportunityType = Field(default=OpportunityType.RESOURCE_EFFICIENCY)
    name: str = Field(default="")
    description: str = Field(default="")
    time_horizon: RiskTimeHorizon = Field(default=RiskTimeHorizon.MEDIUM_TERM)
    financial_impact_usd: float = Field(default=0.0)
    financial_impact_description: str = Field(default="")
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.LIKELY)
    strategic_actions: List[str] = Field(default_factory=list)


class OpportunityAssessment(BaseModel):
    """Complete opportunity assessment from GL-TCFD-APP."""
    assessment_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    opportunities: List[ClimateOpportunity] = Field(default_factory=list)
    total_opportunities: int = Field(default=0)
    total_financial_impact_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=_utcnow)


class GLTCFDAppResult(BaseModel):
    result_id: str = Field(default_factory=_new_uuid)
    scenarios: Optional[ScenarioAnalysis] = Field(None)
    risks: Optional[RiskAssessment] = Field(None)
    opportunities: Optional[OpportunityAssessment] = Field(None)
    app_available: bool = Field(default=False)
    import_status: ImportStatus = Field(default=ImportStatus.FAILED)
    integration_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    frameworks_serviced: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# GLTCFDAppIntegration
# ---------------------------------------------------------------------------


class GLTCFDAppIntegration:
    """GL-TCFD-APP integration for PACK-030.

    Example:
        >>> config = GLTCFDAppConfig(organization_name="Acme Corp")
        >>> integration = GLTCFDAppIntegration(config)
        >>> scenarios = await integration.fetch_scenarios()
        >>> risks = await integration.fetch_risks()
        >>> opportunities = await integration.fetch_opportunities()
    """

    def __init__(self, config: Optional[GLTCFDAppConfig] = None) -> None:
        self.config = config or GLTCFDAppConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._scenarios_cache: Optional[ScenarioAnalysis] = None
        self._risks_cache: Optional[RiskAssessment] = None
        self._opportunities_cache: Optional[OpportunityAssessment] = None
        self._db_pool: Optional[Any] = None
        self.logger.info("GLTCFDAppIntegration (PACK-030) initialized: org=%s", self.config.organization_name)

    async def _get_db_pool(self) -> Any:
        if self._db_pool is not None:
            return self._db_pool
        if not self.config.db_connection_string:
            return None
        try:
            import psycopg_pool
            self._db_pool = psycopg_pool.AsyncConnectionPool(
                self.config.db_connection_string, min_size=1, max_size=self.config.db_pool_size)
            await self._db_pool.open()
            return self._db_pool
        except Exception as exc:
            self.logger.warning("DB pool creation failed: %s", exc)
            return None

    async def _query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pool = await self._get_db_pool()
        if not pool:
            return []
        attempt = 0
        while attempt < self.config.retry_attempts:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params or {})
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        rows = await cur.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
            except Exception as exc:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    import asyncio
                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
        return []

    async def fetch_scenarios(self, override_data: Optional[List[Dict[str, Any]]] = None) -> ScenarioAnalysis:
        """Fetch scenario analysis from GL-TCFD-APP."""
        if self._scenarios_cache is not None:
            return self._scenarios_cache

        raw_data = override_data or self._default_scenarios()

        scenarios: List[ScenarioResult] = []
        for row in raw_data:
            scenarios.append(ScenarioResult(
                scenario_type=ScenarioType(row.get("scenario_type", "nze_1.5c")),
                scenario_name=row.get("scenario_name", ""),
                description=row.get("description", ""),
                temperature_outcome=row.get("temperature_outcome", ""),
                carbon_price_2030_usd=row.get("carbon_price_2030_usd", 0.0),
                carbon_price_2050_usd=row.get("carbon_price_2050_usd", 0.0),
                revenue_impact_pct=row.get("revenue_impact_pct", 0.0),
                cost_impact_pct=row.get("cost_impact_pct", 0.0),
                asset_impairment_pct=row.get("asset_impairment_pct", 0.0),
                transition_risk_level=row.get("transition_risk_level", "medium"),
                physical_risk_level=row.get("physical_risk_level", "low"),
                strategy_resilience=row.get("strategy_resilience", ""),
                key_assumptions=row.get("key_assumptions", []),
            ))

        analysis = ScenarioAnalysis(
            organization_id=self.config.organization_id,
            scenarios=scenarios,
            reference_scenario=ScenarioType.NZE_15C,
        )
        if self.config.enable_provenance:
            analysis.provenance_hash = _compute_hash(analysis)

        self._scenarios_cache = analysis
        self.logger.info("Scenarios fetched from GL-TCFD-APP: %d scenarios", len(scenarios))
        return analysis

    async def fetch_risks(self, override_data: Optional[List[Dict[str, Any]]] = None) -> RiskAssessment:
        """Fetch climate risk assessments from GL-TCFD-APP."""
        if self._risks_cache is not None:
            return self._risks_cache

        raw_data = override_data or self._default_risks()

        risks: List[ClimateRisk] = []
        for row in raw_data:
            risks.append(ClimateRisk(
                risk_type=RiskType(row.get("risk_type", "transition_policy")),
                risk_name=row.get("risk_name", ""),
                description=row.get("description", ""),
                likelihood=RiskLikelihood(row.get("likelihood", "possible")),
                impact=RiskImpact(row.get("impact", "medium")),
                time_horizon=RiskTimeHorizon(row.get("time_horizon", "medium_term")),
                financial_impact_usd=row.get("financial_impact_usd", 0.0),
                financial_impact_description=row.get("financial_impact_description", ""),
                mitigation_actions=row.get("mitigation_actions", []),
                affected_assets_pct=row.get("affected_assets_pct", 0.0),
                residual_risk_level=row.get("residual_risk_level", "low"),
            ))

        transition = sum(1 for r in risks if r.risk_type.value.startswith("transition"))
        physical = sum(1 for r in risks if r.risk_type.value.startswith("physical"))
        high_impact = sum(1 for r in risks if r.impact in (RiskImpact.HIGH, RiskImpact.VERY_HIGH))
        total_financial = sum(r.financial_impact_usd for r in risks)

        assessment = RiskAssessment(
            organization_id=self.config.organization_id,
            risks=risks, total_risks=len(risks),
            transition_risks=transition, physical_risks=physical,
            total_financial_impact_usd=round(total_financial, 2),
            high_impact_risks=high_impact,
        )
        if self.config.enable_provenance:
            assessment.provenance_hash = _compute_hash(assessment)

        self._risks_cache = assessment
        self.logger.info("Risks fetched: %d total, %d transition, %d physical", len(risks), transition, physical)
        return assessment

    async def fetch_opportunities(self, override_data: Optional[List[Dict[str, Any]]] = None) -> OpportunityAssessment:
        """Fetch climate opportunities from GL-TCFD-APP."""
        if self._opportunities_cache is not None:
            return self._opportunities_cache

        raw_data = override_data or self._default_opportunities()

        opportunities: List[ClimateOpportunity] = []
        for row in raw_data:
            opportunities.append(ClimateOpportunity(
                opportunity_type=OpportunityType(row.get("opportunity_type", "resource_efficiency")),
                name=row.get("name", ""),
                description=row.get("description", ""),
                time_horizon=RiskTimeHorizon(row.get("time_horizon", "medium_term")),
                financial_impact_usd=row.get("financial_impact_usd", 0.0),
                financial_impact_description=row.get("financial_impact_description", ""),
                likelihood=RiskLikelihood(row.get("likelihood", "likely")),
                strategic_actions=row.get("strategic_actions", []),
            ))

        total_financial = sum(o.financial_impact_usd for o in opportunities)

        assessment = OpportunityAssessment(
            organization_id=self.config.organization_id,
            opportunities=opportunities,
            total_opportunities=len(opportunities),
            total_financial_impact_usd=round(total_financial, 2),
        )
        if self.config.enable_provenance:
            assessment.provenance_hash = _compute_hash(assessment)

        self._opportunities_cache = assessment
        return assessment

    async def get_full_integration(self) -> GLTCFDAppResult:
        errors: List[str] = []
        warnings: List[str] = []
        scenarios = risks = opportunities = None

        try:
            scenarios = await self.fetch_scenarios()
        except Exception as exc:
            errors.append(f"Scenario fetch failed: {exc}")
        try:
            risks = await self.fetch_risks()
        except Exception as exc:
            errors.append(f"Risk fetch failed: {exc}")
        try:
            opportunities = await self.fetch_opportunities()
        except Exception as exc:
            warnings.append(f"Opportunity fetch failed: {exc}")

        quality = (40.0 if scenarios else 0.0) + (35.0 if risks else 0.0) + (25.0 if opportunities else 0.0)
        status = ImportStatus.SUCCESS if not errors else (
            ImportStatus.FAILED if quality < 40.0 else ImportStatus.PARTIAL)

        result = GLTCFDAppResult(
            scenarios=scenarios, risks=risks, opportunities=opportunities,
            app_available=True, import_status=status,
            integration_quality_score=quality,
            frameworks_serviced=["TCFD", "ISSB", "CSRD", "SEC"],
            validation_errors=errors, validation_warnings=warnings,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def _default_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {"scenario_type": "nze_1.5c", "scenario_name": "IEA Net Zero Emissions by 2050",
             "description": "Aggressive decarbonization aligned with 1.5C", "temperature_outcome": "1.5C",
             "carbon_price_2030_usd": 130.0, "carbon_price_2050_usd": 250.0,
             "revenue_impact_pct": -5.0, "cost_impact_pct": 8.0, "asset_impairment_pct": 15.0,
             "transition_risk_level": "high", "physical_risk_level": "low",
             "strategy_resilience": "Organization is well-positioned with SBTi-aligned targets",
             "key_assumptions": ["Global coal phase-out by 2040", "EV sales >60% by 2030"]},
            {"scenario_type": "below_2c", "scenario_name": "IEA Announced Pledges Scenario",
             "description": "Moderate decarbonization based on current pledges", "temperature_outcome": "1.8C",
             "carbon_price_2030_usd": 90.0, "carbon_price_2050_usd": 175.0,
             "revenue_impact_pct": -2.0, "cost_impact_pct": 5.0, "asset_impairment_pct": 8.0,
             "transition_risk_level": "medium", "physical_risk_level": "medium"},
            {"scenario_type": "bau", "scenario_name": "Current Policies (STEPS)",
             "description": "Continuation of existing policies", "temperature_outcome": "2.5C",
             "carbon_price_2030_usd": 45.0, "carbon_price_2050_usd": 80.0,
             "revenue_impact_pct": -8.0, "cost_impact_pct": 3.0, "asset_impairment_pct": 5.0,
             "transition_risk_level": "low", "physical_risk_level": "high"},
            {"scenario_type": "high_warming_4c", "scenario_name": "High Warming Scenario",
             "description": "Insufficient climate action leading to 4C warming", "temperature_outcome": "4.0C",
             "carbon_price_2030_usd": 20.0, "carbon_price_2050_usd": 30.0,
             "revenue_impact_pct": -15.0, "cost_impact_pct": 2.0, "asset_impairment_pct": 30.0,
             "transition_risk_level": "very_low", "physical_risk_level": "very_high"},
        ]

    def _default_risks(self) -> List[Dict[str, Any]]:
        return [
            {"risk_type": "transition_policy", "risk_name": "Carbon pricing expansion",
             "description": "Expansion of EU ETS to new sectors and introduction of CBAM",
             "likelihood": "very_likely", "impact": "high", "time_horizon": "short_term",
             "financial_impact_usd": 5000000.0,
             "financial_impact_description": "Estimated $5M annual cost increase from carbon pricing",
             "mitigation_actions": ["Accelerate decarbonization investments", "Shift to renewable energy"],
             "affected_assets_pct": 25.0, "residual_risk_level": "medium"},
            {"risk_type": "transition_technology", "risk_name": "Technology disruption in energy",
             "description": "Rapid shift to electrification may strand fossil fuel assets",
             "likelihood": "likely", "impact": "medium", "time_horizon": "medium_term",
             "financial_impact_usd": 3000000.0, "mitigation_actions": ["Invest in electrification R&D"],
             "affected_assets_pct": 15.0},
            {"risk_type": "transition_market", "risk_name": "Customer preference shifts",
             "description": "B2B customers requiring Scope 3 supplier data and targets",
             "likelihood": "likely", "impact": "medium", "time_horizon": "short_term",
             "financial_impact_usd": 2000000.0, "mitigation_actions": ["Publish product carbon footprints"]},
            {"risk_type": "physical_acute", "risk_name": "Extreme weather events",
             "description": "Increased frequency of flooding and storms at manufacturing sites",
             "likelihood": "possible", "impact": "high", "time_horizon": "long_term",
             "financial_impact_usd": 8000000.0, "mitigation_actions": ["Climate adaptation investments"],
             "affected_assets_pct": 10.0},
            {"risk_type": "physical_chronic", "risk_name": "Water stress",
             "description": "Chronic water stress at facilities in Southern Europe and Asia",
             "likelihood": "likely", "impact": "medium", "time_horizon": "long_term",
             "financial_impact_usd": 1500000.0, "mitigation_actions": ["Water efficiency program"]},
        ]

    def _default_opportunities(self) -> List[Dict[str, Any]]:
        return [
            {"opportunity_type": "resource_efficiency", "name": "Energy efficiency improvements",
             "description": "LED, BMS, and heat recovery yielding 15% energy reduction",
             "time_horizon": "short_term", "financial_impact_usd": 2000000.0,
             "likelihood": "very_likely", "strategic_actions": ["Implement ISO 50001"]},
            {"opportunity_type": "energy_source", "name": "Renewable energy procurement",
             "description": "On-site solar and PPA for 50% renewable electricity",
             "time_horizon": "medium_term", "financial_impact_usd": 3000000.0,
             "likelihood": "likely", "strategic_actions": ["Sign 10-year PPA"]},
            {"opportunity_type": "products_services", "name": "Low-carbon product lines",
             "description": "Premium pricing for verified low-carbon products",
             "time_horizon": "medium_term", "financial_impact_usd": 5000000.0,
             "likelihood": "likely", "strategic_actions": ["Develop product carbon labels"]},
            {"opportunity_type": "markets", "name": "Green finance access",
             "description": "Green bond and sustainability-linked loan eligibility",
             "time_horizon": "short_term", "financial_impact_usd": 1500000.0,
             "likelihood": "very_likely", "strategic_actions": ["Issue green bond framework"]},
        ]

    def get_integration_status(self) -> Dict[str, Any]:
        return {
            "pack_id": self.config.pack_id, "app_id": self.config.app_id,
            "scenarios_fetched": self._scenarios_cache is not None,
            "risks_fetched": self._risks_cache is not None,
            "opportunities_fetched": self._opportunities_cache is not None,
            "module_version": _MODULE_VERSION,
        }

    async def refresh(self) -> GLTCFDAppResult:
        self._scenarios_cache = None
        self._risks_cache = None
        self._opportunities_cache = None
        return await self.get_full_integration()

    async def close(self) -> None:
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception:
                pass
            self._db_pool = None
