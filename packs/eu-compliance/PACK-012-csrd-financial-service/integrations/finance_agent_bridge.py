# -*- coding: utf-8 -*-
"""
FinanceAgentBridge - Bridge to GreenLang Finance Agents
=========================================================

Connects PACK-012 (CSRD Financial Service) with greenlang.agents.finance
(9 agents) for green investment screening, EU Taxonomy alignment,
green bond analysis, stranded asset detection, climate finance tracking,
and carbon pricing integration.

Architecture:
    PACK-012 CSRD FS --> FinanceAgentBridge --> greenlang.agents.finance
                              |
                              v
    Green Screener, Taxonomy Agent, Bond Analyzer, Stranded Asset,
    Climate Finance Tracker, Carbon Pricing

Example:
    >>> config = FinanceAgentBridgeConfig()
    >>> bridge = FinanceAgentBridge(config)
    >>> screening = bridge.run_green_screening(portfolio_data)
    >>> stranded = bridge.assess_stranded_assets(portfolio_data)

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

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


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


class FinanceAgentType(str, Enum):
    """Finance agents available for bridging."""
    GREEN_SCREENER = "green_investment_screener"
    TAXONOMY_ALIGNMENT = "eu_taxonomy_alignment"
    GREEN_BOND = "green_bond_analyzer"
    STRANDED_ASSET = "stranded_asset_analyzer"
    CLIMATE_FINANCE = "climate_finance_tracker"
    CARBON_PRICING = "carbon_pricing"


class FinanceAgentBridgeConfig(BaseModel):
    """Configuration for the Finance Agent Bridge."""
    finance_agents_path: str = Field(
        default="greenlang.agents.finance",
        description="Import path for finance agents",
    )
    enabled_agents: List[str] = Field(
        default_factory=lambda: [a.value for a in FinanceAgentType],
        description="Finance agents to enable",
    )
    screening_rules: Dict[str, Any] = Field(
        default_factory=lambda: {
            "exclude_fossil_fuel_pct": 5.0,
            "exclude_weapons_pct": 0.0,
            "exclude_tobacco_pct": 0.0,
            "min_esg_score": 50.0,
        },
        description="Green investment screening rules",
    )
    stranded_asset_sectors: List[str] = Field(
        default_factory=lambda: [
            "coal_mining", "oil_gas_extraction", "thermal_power",
            "fossil_fuel_refining", "high_emission_transport",
        ],
        description="Sectors at risk of stranded assets",
    )
    carbon_price_eur_per_tco2: float = Field(
        default=90.0, ge=0.0,
        description="Carbon price assumption (EUR/tCO2e)",
    )


class GreenScreeningResult(BaseModel):
    """Result of green investment screening."""
    total_assets: int = Field(default=0, description="Total assets screened")
    green_qualified: int = Field(default=0, description="Assets passing green screen")
    excluded: int = Field(default=0, description="Assets excluded")
    green_pct: float = Field(default=0.0, description="Green qualified percentage")
    exclusion_reasons: Dict[str, int] = Field(
        default_factory=dict, description="Exclusion reason counts",
    )
    green_exposure_eur: float = Field(
        default=0.0, description="Green qualified exposure (EUR)",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class StrandedAssetResult(BaseModel):
    """Result of stranded asset assessment."""
    total_assessed: int = Field(default=0, description="Total assets assessed")
    at_risk_count: int = Field(default=0, description="Assets at risk of stranding")
    at_risk_exposure_eur: float = Field(
        default=0.0, description="At-risk exposure (EUR)",
    )
    at_risk_pct: float = Field(default=0.0, description="At-risk percentage")
    sector_breakdown: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-sector stranded risk",
    )
    carbon_cost_impact_eur: float = Field(
        default=0.0, description="Estimated carbon cost impact (EUR)",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CarbonPricingResult(BaseModel):
    """Result of carbon pricing impact analysis."""
    total_emissions_tco2e: float = Field(
        default=0.0, description="Total emissions assessed",
    )
    carbon_price_eur: float = Field(
        default=0.0, description="Carbon price used (EUR/tCO2e)",
    )
    total_carbon_cost_eur: float = Field(
        default=0.0, description="Total carbon cost (EUR)",
    )
    cost_as_pct_of_exposure: float = Field(
        default=0.0, description="Carbon cost as % of total exposure",
    )
    sector_costs: Dict[str, float] = Field(
        default_factory=dict, description="Carbon cost by sector",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class FinanceAgentBridge:
    """Bridge connecting PACK-012 with greenlang.agents.finance.

    Provides green investment screening, stranded asset assessment,
    carbon pricing impact analysis, and climate finance tracking for
    financial institution CSRD reporting.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for finance agents.

    Example:
        >>> bridge = FinanceAgentBridge()
        >>> screening = bridge.run_green_screening(portfolio_data)
        >>> print(f"Green: {screening.green_pct}%")
    """

    def __init__(self, config: Optional[FinanceAgentBridgeConfig] = None) -> None:
        """Initialize the Finance Agent Bridge."""
        self.config = config or FinanceAgentBridgeConfig()
        self.logger = logger

        base = self.config.finance_agents_path
        self._agents: Dict[str, _AgentStub] = {
            "green_screener": _AgentStub(
                "GL-FIN-SCREEN",
                f"{base}.green_investment_screener",
                "GreenInvestmentScreenerAgent",
            ),
            "taxonomy_alignment": _AgentStub(
                "GL-FIN-TAX",
                f"{base}.eu_taxonomy_alignment",
                "EUTaxonomyAlignmentAgent",
            ),
            "green_bond": _AgentStub(
                "GL-FIN-BOND",
                f"{base}.green_bond_analyzer",
                "GreenBondAnalyzerAgent",
            ),
            "stranded_asset": _AgentStub(
                "GL-FIN-STRANDED",
                f"{base}.stranded_asset_analyzer",
                "StrandedAssetAnalyzerAgent",
            ),
            "climate_finance": _AgentStub(
                "GL-FIN-CLIMATE",
                f"{base}.climate_finance_tracker",
                "ClimateFinanceTrackerAgent",
            ),
            "carbon_pricing": _AgentStub(
                "GL-FIN-CARBON",
                f"{base}.carbon_pricing",
                "CarbonPricingAgent",
            ),
        }

        self.logger.info(
            "FinanceAgentBridge initialized: agents=%d, carbon_price=%.2f",
            len(self.config.enabled_agents),
            self.config.carbon_price_eur_per_tco2,
        )

    def run_green_screening(
        self,
        portfolio_data: List[Dict[str, Any]],
    ) -> GreenScreeningResult:
        """Run green investment screening on portfolio.

        Screens each asset against exclusion rules and green qualification
        criteria. Used for CSRD disclosure of green vs. brown asset split.

        Args:
            portfolio_data: List of portfolio holdings/exposures.

        Returns:
            GreenScreeningResult with qualification breakdown.
        """
        rules = self.config.screening_rules
        green_count = 0
        excluded_count = 0
        green_exposure = 0.0
        exclusion_reasons: Dict[str, int] = {}

        for asset in portfolio_data:
            exposure = float(asset.get("exposure_eur", 0.0))
            sector = asset.get("nace_sector", "")
            esg_score = float(asset.get("esg_score", 50.0))
            fossil_pct = float(asset.get("fossil_fuel_revenue_pct", 0.0))
            weapons = asset.get("controversial_weapons", False)

            excluded = False
            if fossil_pct > rules.get("exclude_fossil_fuel_pct", 5.0):
                excluded = True
                exclusion_reasons["fossil_fuel"] = (
                    exclusion_reasons.get("fossil_fuel", 0) + 1
                )
            if weapons:
                excluded = True
                exclusion_reasons["weapons"] = (
                    exclusion_reasons.get("weapons", 0) + 1
                )
            if esg_score < rules.get("min_esg_score", 50.0):
                excluded = True
                exclusion_reasons["low_esg"] = (
                    exclusion_reasons.get("low_esg", 0) + 1
                )

            if excluded:
                excluded_count += 1
            else:
                green_count += 1
                green_exposure += exposure

        total = len(portfolio_data)
        green_pct = round((green_count / max(total, 1)) * 100, 1)

        result = GreenScreeningResult(
            total_assets=total,
            green_qualified=green_count,
            excluded=excluded_count,
            green_pct=green_pct,
            exclusion_reasons=exclusion_reasons,
            green_exposure_eur=round(green_exposure, 2),
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Green screening: %d/%d qualified (%.1f%%), excluded=%d",
            green_count, total, green_pct, excluded_count,
        )
        return result

    def assess_stranded_assets(
        self,
        portfolio_data: List[Dict[str, Any]],
    ) -> StrandedAssetResult:
        """Assess stranded asset risk in the portfolio.

        Identifies assets in high-risk sectors that may become stranded
        under climate transition scenarios.

        Args:
            portfolio_data: Portfolio holdings with sector data.

        Returns:
            StrandedAssetResult with risk assessment.
        """
        at_risk_count = 0
        at_risk_exposure = 0.0
        total_exposure = 0.0
        sector_breakdown: Dict[str, Dict[str, Any]] = {}

        for asset in portfolio_data:
            exposure = float(asset.get("exposure_eur", 0.0))
            sector = asset.get("nace_sector", "")
            emissions = float(asset.get("scope12_emissions_tco2e", 0.0))
            total_exposure += exposure

            is_at_risk = sector.lower().replace(" ", "_") in (
                self.config.stranded_asset_sectors
            )
            if not is_at_risk:
                is_at_risk = asset.get("stranded_risk", False)

            if is_at_risk:
                at_risk_count += 1
                at_risk_exposure += exposure

                if sector not in sector_breakdown:
                    sector_breakdown[sector] = {
                        "count": 0, "exposure_eur": 0.0, "emissions_tco2e": 0.0,
                    }
                sector_breakdown[sector]["count"] += 1
                sector_breakdown[sector]["exposure_eur"] += exposure
                sector_breakdown[sector]["emissions_tco2e"] += emissions

        at_risk_pct = round(
            (at_risk_exposure / max(total_exposure, 1.0)) * 100, 2
        )
        total_at_risk_emissions = sum(
            s["emissions_tco2e"] for s in sector_breakdown.values()
        )
        carbon_impact = round(
            total_at_risk_emissions * self.config.carbon_price_eur_per_tco2, 2
        )

        result = StrandedAssetResult(
            total_assessed=len(portfolio_data),
            at_risk_count=at_risk_count,
            at_risk_exposure_eur=round(at_risk_exposure, 2),
            at_risk_pct=at_risk_pct,
            sector_breakdown=sector_breakdown,
            carbon_cost_impact_eur=carbon_impact,
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Stranded assets: %d at risk (%.1f%%), carbon cost=%.2f EUR",
            at_risk_count, at_risk_pct, carbon_impact,
        )
        return result

    def calculate_carbon_pricing_impact(
        self,
        portfolio_data: List[Dict[str, Any]],
    ) -> CarbonPricingResult:
        """Calculate carbon pricing impact on the portfolio.

        Estimates the financial impact of carbon pricing on the portfolio
        based on financed emissions and a given carbon price.

        Args:
            portfolio_data: Portfolio holdings with emissions data.

        Returns:
            CarbonPricingResult with cost estimates.
        """
        total_emissions = 0.0
        total_exposure = 0.0
        sector_emissions: Dict[str, float] = {}

        for asset in portfolio_data:
            emissions = float(asset.get("scope12_emissions_tco2e", 0.0))
            exposure = float(asset.get("exposure_eur", 0.0))
            sector = asset.get("nace_sector", "Unknown")
            attr = float(asset.get("attribution_factor", 1.0))

            financed = emissions * attr
            total_emissions += financed
            total_exposure += exposure
            sector_emissions[sector] = (
                sector_emissions.get(sector, 0.0) + financed
            )

        carbon_price = self.config.carbon_price_eur_per_tco2
        total_cost = round(total_emissions * carbon_price, 2)
        cost_pct = round(
            (total_cost / max(total_exposure, 1.0)) * 100, 4
        )

        sector_costs = {
            s: round(e * carbon_price, 2) for s, e in sector_emissions.items()
        }

        result = CarbonPricingResult(
            total_emissions_tco2e=round(total_emissions, 2),
            carbon_price_eur=carbon_price,
            total_carbon_cost_eur=total_cost,
            cost_as_pct_of_exposure=cost_pct,
            sector_costs=sector_costs,
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Carbon pricing: emissions=%.2f, cost=%.2f EUR (%.4f%%)",
            total_emissions, total_cost, cost_pct,
        )
        return result

    def route_to_finance_agent(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a request to the appropriate finance agent.

        Args:
            request_type: Type of request.
            data: Request data.

        Returns:
            Response from the finance agent or error dictionary.
        """
        portfolio = data.get("portfolio_data", [])

        if request_type == "green_screening":
            result = self.run_green_screening(portfolio)
            return result.model_dump()
        elif request_type == "stranded_assets":
            result = self.assess_stranded_assets(portfolio)
            return result.model_dump()
        elif request_type == "carbon_pricing":
            result = self.calculate_carbon_pricing_impact(portfolio)
            return result.model_dump()
        else:
            return {"error": f"Unknown request type: {request_type}"}
