# -*- coding: utf-8 -*-
"""
MRVEmissionsBridge - MRV Agent Emissions Data for SFDR PAI Indicators
======================================================================

This module routes emissions data from the 30 MRV agents to SFDR PAI
indicators 1 through 6. It maps GHG emissions (Scope 1/2/3), carbon
footprint, GHG intensity, fossil fuel exposure, non-renewable energy
share, and energy intensity from MRV engine outputs to the PAI indicator
data structures required for SFDR disclosures.

Architecture:
    MRV Agents (001-030) --> MRVEmissionsBridge --> PAI Indicators 1-6
                                  |
                                  v
    Portfolio Emissions --> Weighted Aggregation --> Per-Investee Data

Example:
    >>> config = MRVEmissionsBridgeConfig()
    >>> bridge = MRVEmissionsBridge(config)
    >>> pai_data = bridge.get_emissions_for_pai(1, holdings)
    >>> print(f"PAI 1 GHG: {pai_data['total_tco2e']:.2f} tCO2e")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


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
            logger.warning(
                "AgentStub: failed to load %s: %s", self.agent_id, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class EmissionScope(str, Enum):
    """GHG Protocol emission scope."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    TOTAL = "total"


class PAIIndicator(str, Enum):
    """Climate-related PAI indicators (1-6)."""
    GHG_EMISSIONS = "pai_1"
    CARBON_FOOTPRINT = "pai_2"
    GHG_INTENSITY = "pai_3"
    FOSSIL_FUEL_EXPOSURE = "pai_4"
    NON_RENEWABLE_ENERGY = "pai_5"
    ENERGY_INTENSITY = "pai_6"


# =============================================================================
# Data Models
# =============================================================================


class MRVEmissionsBridgeConfig(BaseModel):
    """Configuration for the MRV Emissions Bridge."""
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
        description="Emission scopes to include",
    )
    emission_factor_source: str = Field(
        default="mrv_agents",
        description="Emission factor source (mrv_agents, manual, hybrid)",
    )
    include_estimated: bool = Field(
        default=True,
        description="Include estimated emissions where reported data unavailable",
    )
    scope_3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Scope 3 categories to include (1-15)",
    )
    aggregation_method: str = Field(
        default="enterprise_value",
        description="Attribution method (enterprise_value, equity, revenue)",
    )
    nace_sector_mapping: bool = Field(
        default=True,
        description="Enable NACE sector mapping for PAI 6",
    )


class InvesteeEmissions(BaseModel):
    """Emissions data for a single investee company."""
    investee_id: str = Field(default="", description="Investee identifier (ISIN)")
    investee_name: str = Field(default="", description="Investee name")
    scope_1_tco2e: float = Field(default=0.0, description="Scope 1 tCO2e")
    scope_2_tco2e: float = Field(default=0.0, description="Scope 2 tCO2e")
    scope_3_tco2e: float = Field(default=0.0, description="Scope 3 tCO2e")
    total_tco2e: float = Field(default=0.0, description="Total tCO2e")
    revenue_eur: float = Field(default=0.0, description="Revenue in EUR")
    enterprise_value_eur: float = Field(
        default=0.0, description="Enterprise value in EUR"
    )
    nace_sector: str = Field(default="", description="NACE sector code")
    energy_consumption_gwh: float = Field(
        default=0.0, description="Total energy consumption GWh"
    )
    non_renewable_energy_pct: float = Field(
        default=0.0, description="Non-renewable energy share %"
    )
    fossil_fuel_involved: bool = Field(
        default=False, description="Whether involved in fossil fuels"
    )
    data_source: str = Field(
        default="estimated", description="Data source (reported, estimated, proxy)"
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Data quality score"
    )


class PAIResult(BaseModel):
    """Result for a single PAI indicator."""
    indicator_id: str = Field(default="", description="PAI indicator ID")
    indicator_name: str = Field(default="", description="PAI indicator name")
    value: float = Field(default=0.0, description="Calculated value")
    unit: str = Field(default="", description="Unit of measurement")
    coverage_pct: float = Field(default=0.0, description="Data coverage %")
    investee_count: int = Field(default=0, description="Investees assessed")
    methodology: str = Field(default="", description="Calculation methodology")
    scope_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Breakdown by scope"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")


class PortfolioEmissions(BaseModel):
    """Aggregated portfolio-level emissions."""
    total_scope_1: float = Field(default=0.0, description="Total Scope 1 tCO2e")
    total_scope_2: float = Field(default=0.0, description="Total Scope 2 tCO2e")
    total_scope_3: float = Field(default=0.0, description="Total Scope 3 tCO2e")
    total_emissions: float = Field(default=0.0, description="Total emissions tCO2e")
    carbon_footprint: float = Field(
        default=0.0, description="Carbon footprint (tCO2e/M EUR)"
    )
    ghg_intensity: float = Field(
        default=0.0, description="GHG intensity (tCO2e/M EUR revenue)"
    )
    fossil_fuel_exposure_pct: float = Field(
        default=0.0, description="Fossil fuel exposure %"
    )
    non_renewable_energy_pct: float = Field(
        default=0.0, description="Non-renewable energy %"
    )
    energy_intensity_by_sector: Dict[str, float] = Field(
        default_factory=dict, description="Energy intensity by NACE sector"
    )
    investee_emissions: List[InvesteeEmissions] = Field(
        default_factory=list, description="Per-investee emissions"
    )
    holdings_covered: int = Field(default=0, description="Holdings with data")
    holdings_total: int = Field(default=0, description="Total holdings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# PAI to MRV Mapping
# =============================================================================


PAI_TO_MRV_MAP: Dict[str, Dict[str, Any]] = {
    "pai_1": {
        "name": "GHG emissions",
        "description": "Scope 1, 2, 3 and total GHG emissions",
        "unit": "tCO2e",
        "mrv_agents": [
            "GL-MRV-X-001", "GL-MRV-X-002", "GL-MRV-X-003", "GL-MRV-X-004",
            "GL-MRV-X-005", "GL-MRV-X-006", "GL-MRV-X-007", "GL-MRV-X-008",
            "GL-MRV-X-009", "GL-MRV-X-010", "GL-MRV-X-011", "GL-MRV-X-012",
            "GL-MRV-X-013", "GL-MRV-X-014", "GL-MRV-X-015", "GL-MRV-X-016",
            "GL-MRV-X-017", "GL-MRV-X-018", "GL-MRV-X-019", "GL-MRV-X-020",
            "GL-MRV-X-021", "GL-MRV-X-022", "GL-MRV-X-023", "GL-MRV-X-024",
            "GL-MRV-X-025", "GL-MRV-X-026", "GL-MRV-X-027", "GL-MRV-X-028",
            "GL-MRV-X-029", "GL-MRV-X-030",
        ],
        "scopes": ["scope_1", "scope_2", "scope_3"],
    },
    "pai_2": {
        "name": "Carbon footprint",
        "description": "Total GHG emissions / current value of all investments (M EUR)",
        "unit": "tCO2e/M EUR invested",
        "formula": "total_ghg / portfolio_value_m_eur",
        "mrv_agents": ["GL-MRV-X-029"],
        "scopes": ["scope_1", "scope_2", "scope_3"],
    },
    "pai_3": {
        "name": "GHG intensity of investee companies",
        "description": "Investee GHG emissions / investee revenue (M EUR)",
        "unit": "tCO2e/M EUR revenue",
        "formula": "investee_ghg / investee_revenue_m_eur",
        "mrv_agents": ["GL-MRV-X-029"],
        "scopes": ["scope_1", "scope_2"],
    },
    "pai_4": {
        "name": "Exposure to fossil fuel sector",
        "description": "Share of investments in fossil fuel companies",
        "unit": "%",
        "mrv_agents": ["GL-MRV-X-001", "GL-MRV-X-003", "GL-MRV-X-016"],
        "scopes": ["scope_1"],
    },
    "pai_5": {
        "name": "Non-renewable energy share",
        "description": "Share of non-renewable energy consumption and production",
        "unit": "%",
        "mrv_agents": ["GL-MRV-X-009", "GL-MRV-X-010", "GL-MRV-X-011"],
        "scopes": ["scope_2"],
    },
    "pai_6": {
        "name": "Energy consumption intensity per high impact climate sector",
        "description": "Energy consumption (GWh) per M EUR revenue, by NACE sector",
        "unit": "GWh/M EUR revenue",
        "mrv_agents": ["GL-MRV-X-009", "GL-MRV-X-010"],
        "scopes": ["scope_1", "scope_2"],
        "nace_sectors": [
            "A", "B", "C", "D", "E", "F", "G", "H", "L",
        ],
    },
}


# =============================================================================
# MRV Emissions Bridge
# =============================================================================


class MRVEmissionsBridge:
    """Bridge routing MRV agent emissions data to SFDR PAI indicators.

    Routes emissions data from 30 MRV agents to PAI indicators 1-6,
    performing portfolio-level aggregation using enterprise value or
    equity attribution methods.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for MRV agents.

    Example:
        >>> bridge = MRVEmissionsBridge(MRVEmissionsBridgeConfig())
        >>> result = bridge.get_emissions_for_pai("pai_1", holdings)
        >>> print(f"Total GHG: {result.value:.2f} {result.unit}")
    """

    def __init__(self, config: Optional[MRVEmissionsBridgeConfig] = None) -> None:
        """Initialize the MRV Emissions Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or MRVEmissionsBridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {}
        mrv_agent_defs = [
            ("GL-MRV-X-001", "stationary_combustion", "StationaryCombustionAgent"),
            ("GL-MRV-X-002", "refrigerants_fgas", "RefrigerantsFGasAgent"),
            ("GL-MRV-X-003", "mobile_combustion", "MobileCombustionAgent"),
            ("GL-MRV-X-004", "process_emissions", "ProcessEmissionsAgent"),
            ("GL-MRV-X-005", "fugitive_emissions", "FugitiveEmissionsAgent"),
            ("GL-MRV-X-006", "land_use_emissions", "LandUseEmissionsAgent"),
            ("GL-MRV-X-007", "waste_treatment", "WasteTreatmentAgent"),
            ("GL-MRV-X-008", "agricultural_emissions", "AgriculturalEmissionsAgent"),
            ("GL-MRV-X-009", "scope2_location_based", "Scope2LocationBasedAgent"),
            ("GL-MRV-X-010", "scope2_market_based", "Scope2MarketBasedAgent"),
            ("GL-MRV-X-011", "steam_heat_purchase", "SteamHeatPurchaseAgent"),
            ("GL-MRV-X-012", "cooling_purchase", "CoolingPurchaseAgent"),
            ("GL-MRV-X-013", "dual_reporting_recon", "DualReportingReconciliation"),
            ("GL-MRV-X-014", "purchased_goods_services", "PurchasedGoodsServicesAgent"),
            ("GL-MRV-X-029", "scope3_category_mapper", "Scope3CategoryMapper"),
        ]

        for agent_id, module_name, class_name in mrv_agent_defs:
            self._agents[agent_id] = _AgentStub(
                agent_id,
                f"greenlang.agents.mrv.{module_name}",
                class_name,
            )

        self.logger.info(
            "MRVEmissionsBridge initialized: scopes=%s, source=%s, agents=%d",
            self.config.scope_coverage,
            self.config.emission_factor_source,
            len(self._agents),
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def get_emissions_for_pai(
        self,
        pai_indicator: str,
        holdings: List[Dict[str, Any]],
        portfolio_value_eur: float = 0.0,
    ) -> PAIResult:
        """Calculate emissions data for a specific PAI indicator.

        Routes the calculation to the appropriate MRV agents and performs
        portfolio-level aggregation.

        Args:
            pai_indicator: PAI indicator ID (pai_1 through pai_6).
            holdings: List of portfolio holdings with investee data.
            portfolio_value_eur: Total portfolio value in EUR.

        Returns:
            PAIResult with calculated indicator value.
        """
        pai_def = PAI_TO_MRV_MAP.get(pai_indicator)
        if pai_def is None:
            self.logger.error("Unknown PAI indicator: %s", pai_indicator)
            return PAIResult(
                indicator_id=pai_indicator,
                indicator_name="Unknown",
                calculated_at=_utcnow().isoformat(),
            )

        self.logger.info(
            "Calculating %s (%s) for %d holdings",
            pai_indicator, pai_def["name"], len(holdings),
        )

        if pai_indicator == "pai_1":
            return self._calculate_pai_1(holdings, pai_def)
        elif pai_indicator == "pai_2":
            return self._calculate_pai_2(holdings, pai_def, portfolio_value_eur)
        elif pai_indicator == "pai_3":
            return self._calculate_pai_3(holdings, pai_def)
        elif pai_indicator == "pai_4":
            return self._calculate_pai_4(holdings, pai_def)
        elif pai_indicator == "pai_5":
            return self._calculate_pai_5(holdings, pai_def)
        elif pai_indicator == "pai_6":
            return self._calculate_pai_6(holdings, pai_def)
        else:
            return PAIResult(
                indicator_id=pai_indicator,
                indicator_name=pai_def["name"],
                calculated_at=_utcnow().isoformat(),
            )

    def route_to_mrv_agent(
        self,
        agent_id: str,
        request_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Route a request to a specific MRV agent.

        Args:
            agent_id: MRV agent ID (e.g. GL-MRV-X-001).
            request_data: Data to pass to the agent.

        Returns:
            Agent response, or None if agent unavailable.
        """
        stub = self._agents.get(agent_id)
        if stub is None:
            self.logger.warning("No stub registered for agent %s", agent_id)
            return None

        agent = stub.load()
        if agent is None:
            return None

        try:
            result = agent.process(request_data)
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        except Exception as exc:
            self.logger.warning("MRV agent %s failed: %s", agent_id, exc)
            return None

    def aggregate_portfolio_emissions(
        self,
        holdings: List[Dict[str, Any]],
        portfolio_value_eur: float = 0.0,
    ) -> PortfolioEmissions:
        """Aggregate emissions across all portfolio holdings.

        Calculates portfolio-level emissions using enterprise value
        attribution for all PAI indicators 1-6.

        Args:
            holdings: List of portfolio holdings with emissions data.
            portfolio_value_eur: Total portfolio value in EUR.

        Returns:
            PortfolioEmissions with aggregated data.
        """
        investee_list: List[InvesteeEmissions] = []
        total_s1 = 0.0
        total_s2 = 0.0
        total_s3 = 0.0
        total_revenue = 0.0
        total_ev = 0.0
        fossil_count = 0
        covered = 0
        weighted_nre = 0.0
        total_weight = 0.0

        energy_by_sector: Dict[str, Dict[str, float]] = {}

        for h in holdings:
            isin = h.get("isin", "")
            emissions = h.get("emissions", {})
            weight = float(h.get("weight", 0.0))
            total_weight += weight

            s1 = float(emissions.get("scope_1_tco2e", 0.0))
            s2 = float(emissions.get("scope_2_tco2e", 0.0))
            s3 = float(emissions.get("scope_3_tco2e", 0.0))
            revenue = float(h.get("revenue_eur", 0.0))
            ev = float(h.get("enterprise_value_eur", 0.0))
            energy = float(h.get("energy_consumption_gwh", 0.0))
            nre_pct = float(h.get("non_renewable_energy_pct", 0.0))
            fossil = h.get("fossil_fuel_involved", False)
            sector = h.get("nace_sector", "")

            # Attribution by weight
            attr_factor = weight / 100.0 if total_weight > 0 else 0.0
            total_s1 += s1 * attr_factor
            total_s2 += s2 * attr_factor
            total_s3 += s3 * attr_factor
            total_revenue += revenue * attr_factor
            total_ev += ev * attr_factor

            weighted_nre += nre_pct * attr_factor
            if fossil:
                fossil_count += 1

            # Energy by NACE sector
            if sector and energy > 0:
                if sector not in energy_by_sector:
                    energy_by_sector[sector] = {
                        "energy_gwh": 0.0, "revenue_m_eur": 0.0,
                    }
                energy_by_sector[sector]["energy_gwh"] += energy * attr_factor
                energy_by_sector[sector]["revenue_m_eur"] += (
                    revenue / 1_000_000.0 * attr_factor
                )

            has_data = s1 > 0 or s2 > 0 or s3 > 0
            if has_data:
                covered += 1

            investee_list.append(InvesteeEmissions(
                investee_id=isin,
                investee_name=h.get("name", ""),
                scope_1_tco2e=s1,
                scope_2_tco2e=s2,
                scope_3_tco2e=s3,
                total_tco2e=s1 + s2 + s3,
                revenue_eur=revenue,
                enterprise_value_eur=ev,
                nace_sector=sector,
                energy_consumption_gwh=energy,
                non_renewable_energy_pct=nre_pct,
                fossil_fuel_involved=fossil,
                data_source="reported" if has_data else "estimated",
                data_quality_score=0.8 if has_data else 0.3,
            ))

        total_emissions = total_s1 + total_s2 + total_s3
        portfolio_m_eur = portfolio_value_eur / 1_000_000.0 if portfolio_value_eur > 0 else 1.0
        total_revenue_m_eur = total_revenue / 1_000_000.0 if total_revenue > 0 else 1.0

        # Energy intensity per NACE sector
        energy_intensity: Dict[str, float] = {}
        for sector, data in energy_by_sector.items():
            rev = data["revenue_m_eur"]
            if rev > 0:
                energy_intensity[sector] = round(data["energy_gwh"] / rev, 4)

        fossil_pct = (
            round((fossil_count / len(holdings)) * 100, 2)
            if holdings else 0.0
        )

        result = PortfolioEmissions(
            total_scope_1=round(total_s1, 4),
            total_scope_2=round(total_s2, 4),
            total_scope_3=round(total_s3, 4),
            total_emissions=round(total_emissions, 4),
            carbon_footprint=round(total_emissions / portfolio_m_eur, 4),
            ghg_intensity=round(total_emissions / total_revenue_m_eur, 4),
            fossil_fuel_exposure_pct=fossil_pct,
            non_renewable_energy_pct=round(weighted_nre, 2),
            energy_intensity_by_sector=energy_intensity,
            investee_emissions=investee_list,
            holdings_covered=covered,
            holdings_total=len(holdings),
        )
        result.provenance_hash = _hash_data({
            "total_emissions": result.total_emissions,
            "holdings": result.holdings_total,
            "covered": result.holdings_covered,
        })

        self.logger.info(
            "Portfolio emissions aggregated: total=%.2f tCO2e, "
            "coverage=%d/%d, fossil=%.1f%%",
            total_emissions, covered, len(holdings), fossil_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # PAI Calculation Methods
    # -------------------------------------------------------------------------

    def _calculate_pai_1(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
    ) -> PAIResult:
        """PAI 1: GHG emissions (Scope 1, 2, 3 and total).

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.

        Returns:
            PAIResult with GHG emissions data.
        """
        portfolio = self.aggregate_portfolio_emissions(holdings)

        return PAIResult(
            indicator_id="pai_1",
            indicator_name=pai_def["name"],
            value=portfolio.total_emissions,
            unit=pai_def["unit"],
            coverage_pct=round(
                (portfolio.holdings_covered / max(portfolio.holdings_total, 1)) * 100, 1
            ),
            investee_count=portfolio.holdings_total,
            methodology="enterprise_value_attribution",
            scope_breakdown={
                "scope_1": portfolio.total_scope_1,
                "scope_2": portfolio.total_scope_2,
                "scope_3": portfolio.total_scope_3,
            },
            data_sources=["mrv_agents", "estimated"],
            provenance_hash=portfolio.provenance_hash,
            calculated_at=_utcnow().isoformat(),
        )

    def _calculate_pai_2(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
        portfolio_value_eur: float,
    ) -> PAIResult:
        """PAI 2: Carbon footprint (total GHG / portfolio value).

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.
            portfolio_value_eur: Total portfolio value in EUR.

        Returns:
            PAIResult with carbon footprint value.
        """
        portfolio = self.aggregate_portfolio_emissions(
            holdings, portfolio_value_eur
        )

        return PAIResult(
            indicator_id="pai_2",
            indicator_name=pai_def["name"],
            value=portfolio.carbon_footprint,
            unit=pai_def["unit"],
            coverage_pct=round(
                (portfolio.holdings_covered / max(portfolio.holdings_total, 1)) * 100, 1
            ),
            investee_count=portfolio.holdings_total,
            methodology="total_ghg_divided_by_portfolio_value",
            scope_breakdown={
                "scope_1": portfolio.total_scope_1,
                "scope_2": portfolio.total_scope_2,
                "scope_3": portfolio.total_scope_3,
                "portfolio_value_m_eur": portfolio_value_eur / 1_000_000.0,
            },
            data_sources=["mrv_agents", "estimated"],
            provenance_hash=_hash_data({
                "pai": "pai_2",
                "footprint": portfolio.carbon_footprint,
            }),
            calculated_at=_utcnow().isoformat(),
        )

    def _calculate_pai_3(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
    ) -> PAIResult:
        """PAI 3: GHG intensity (GHG / revenue).

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.

        Returns:
            PAIResult with GHG intensity value.
        """
        portfolio = self.aggregate_portfolio_emissions(holdings)

        return PAIResult(
            indicator_id="pai_3",
            indicator_name=pai_def["name"],
            value=portfolio.ghg_intensity,
            unit=pai_def["unit"],
            coverage_pct=round(
                (portfolio.holdings_covered / max(portfolio.holdings_total, 1)) * 100, 1
            ),
            investee_count=portfolio.holdings_total,
            methodology="investee_ghg_scope12_divided_by_revenue",
            data_sources=["mrv_agents", "estimated"],
            provenance_hash=_hash_data({
                "pai": "pai_3", "intensity": portfolio.ghg_intensity,
            }),
            calculated_at=_utcnow().isoformat(),
        )

    def _calculate_pai_4(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
    ) -> PAIResult:
        """PAI 4: Exposure to companies active in the fossil fuel sector.

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.

        Returns:
            PAIResult with fossil fuel exposure percentage.
        """
        fossil_weight = 0.0
        total_weight = 0.0
        fossil_names: List[str] = []

        for h in holdings:
            weight = float(h.get("weight", 0.0))
            total_weight += weight
            if h.get("fossil_fuel_involved", False):
                fossil_weight += weight
                fossil_names.append(h.get("name", h.get("isin", "")))

        exposure_pct = (
            round((fossil_weight / total_weight) * 100, 2) if total_weight > 0
            else 0.0
        )

        return PAIResult(
            indicator_id="pai_4",
            indicator_name=pai_def["name"],
            value=exposure_pct,
            unit=pai_def["unit"],
            coverage_pct=100.0,
            investee_count=len(holdings),
            methodology="weight_of_fossil_fuel_companies",
            scope_breakdown={"fossil_weight_pct": exposure_pct},
            data_sources=["mrv_agents", "sector_classification"],
            provenance_hash=_hash_data({
                "pai": "pai_4", "exposure": exposure_pct,
            }),
            calculated_at=_utcnow().isoformat(),
        )

    def _calculate_pai_5(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
    ) -> PAIResult:
        """PAI 5: Share of non-renewable energy consumption and production.

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.

        Returns:
            PAIResult with non-renewable energy share.
        """
        total_weight = 0.0
        weighted_nre = 0.0
        covered = 0

        for h in holdings:
            weight = float(h.get("weight", 0.0))
            total_weight += weight
            nre = float(h.get("non_renewable_energy_pct", 0.0))
            if nre > 0 or h.get("energy_consumption_gwh", 0) > 0:
                weighted_nre += nre * weight
                covered += 1

        nre_pct = (
            round(weighted_nre / total_weight, 2) if total_weight > 0 else 0.0
        )

        return PAIResult(
            indicator_id="pai_5",
            indicator_name=pai_def["name"],
            value=nre_pct,
            unit=pai_def["unit"],
            coverage_pct=round(
                (covered / max(len(holdings), 1)) * 100, 1
            ),
            investee_count=len(holdings),
            methodology="weighted_average_nre_share",
            data_sources=["mrv_agents", "energy_data"],
            provenance_hash=_hash_data({
                "pai": "pai_5", "nre_pct": nre_pct,
            }),
            calculated_at=_utcnow().isoformat(),
        )

    def _calculate_pai_6(
        self,
        holdings: List[Dict[str, Any]],
        pai_def: Dict[str, Any],
    ) -> PAIResult:
        """PAI 6: Energy consumption intensity per high impact climate sector.

        Args:
            holdings: Portfolio holdings.
            pai_def: PAI indicator definition.

        Returns:
            PAIResult with energy intensity by NACE sector.
        """
        nace_sectors = pai_def.get("nace_sectors", [])
        sector_data: Dict[str, Dict[str, float]] = {}

        for h in holdings:
            sector = h.get("nace_sector", "")
            if not sector or sector[0] not in nace_sectors:
                continue

            sector_key = sector[0] if len(sector) >= 1 else sector
            energy = float(h.get("energy_consumption_gwh", 0.0))
            revenue = float(h.get("revenue_eur", 0.0))

            if sector_key not in sector_data:
                sector_data[sector_key] = {"energy_gwh": 0.0, "revenue_m_eur": 0.0}
            sector_data[sector_key]["energy_gwh"] += energy
            sector_data[sector_key]["revenue_m_eur"] += revenue / 1_000_000.0

        intensity_by_sector: Dict[str, float] = {}
        for sector, data in sector_data.items():
            rev = data["revenue_m_eur"]
            if rev > 0:
                intensity_by_sector[sector] = round(data["energy_gwh"] / rev, 4)

        avg_intensity = (
            sum(intensity_by_sector.values()) / len(intensity_by_sector)
            if intensity_by_sector else 0.0
        )

        return PAIResult(
            indicator_id="pai_6",
            indicator_name=pai_def["name"],
            value=round(avg_intensity, 4),
            unit=pai_def["unit"],
            coverage_pct=round(
                (len(sector_data) / max(len(nace_sectors), 1)) * 100, 1
            ),
            investee_count=len(holdings),
            methodology="energy_gwh_per_m_eur_revenue_by_nace",
            scope_breakdown=intensity_by_sector,
            data_sources=["mrv_agents", "energy_data", "nace_classification"],
            provenance_hash=_hash_data({
                "pai": "pai_6", "sectors": intensity_by_sector,
            }),
            calculated_at=_utcnow().isoformat(),
        )
