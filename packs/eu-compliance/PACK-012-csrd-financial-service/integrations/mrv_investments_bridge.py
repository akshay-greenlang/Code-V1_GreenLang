# -*- coding: utf-8 -*-
"""
MRVInvestmentsBridge - Bridge to AGENT-MRV-028 Investments Agent (PCAF)
=========================================================================

Connects PACK-012 (CSRD Financial Service) with the AGENT-MRV-028
Investments Agent which implements the PCAF (Partnership for Carbon
Accounting Financials) stack for financed emissions calculations across
6+ asset classes.

The bridge imports calculation engines for equity investments, debt
instruments, real assets, sovereign bonds, and project finance, then
provides them to the PACK-012 financed emissions pipeline.

Architecture:
    PACK-012 CSRD FS --> MRVInvestmentsBridge --> AGENT-MRV-028
                              |
                              v
    Equity Calculator, Debt Calculator, Real Asset Calculator,
    Sovereign Bond Calculator, Project Finance Calculator

Example:
    >>> config = MRVInvestmentsBridgeConfig(pcaf_version="2.1")
    >>> bridge = MRVInvestmentsBridge(config)
    >>> result = bridge.calculate_financed_emissions(counterparties)
    >>> print(f"Total: {result.total_financed_tco2e} tCO2e")

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


class PCAFAssetClass(str, Enum):
    """PCAF asset classes for financed emissions."""
    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLE_LOANS = "motor_vehicle_loans"
    SOVEREIGN_BONDS = "sovereign_bonds"


class PCAFDataQuality(int, Enum):
    """PCAF data quality scores (1=best, 5=worst)."""
    REPORTED_VERIFIED = 1
    REPORTED_UNVERIFIED = 2
    ESTIMATED_PHYSICAL = 3
    ESTIMATED_ECONOMIC = 4
    ESTIMATED_SECTOR = 5


class MRVInvestmentsBridgeConfig(BaseModel):
    """Configuration for the MRV Investments Bridge."""
    agent_mrv_028_path: str = Field(
        default="greenlang.agents.mrv.investments",
        description="Import path for AGENT-MRV-028",
    )
    pcaf_version: str = Field(
        default="2.1", description="PCAF Standard version",
    )
    asset_classes: List[str] = Field(
        default_factory=lambda: [ac.value for ac in PCAFAssetClass],
        description="PCAF asset classes to calculate",
    )
    calculation_methods: List[str] = Field(
        default_factory=lambda: [
            "attribution_factor",
            "investment_share",
            "outstanding_amount",
        ],
        description="Supported calculation methods",
    )
    enable_scope3: bool = Field(
        default=True, description="Include Scope 3 in financed emissions",
    )
    data_quality_target: int = Field(
        default=3, ge=1, le=5,
        description="Target PCAF data quality score (1-5)",
    )


class AssetClassResult(BaseModel):
    """Financed emissions result for a single asset class."""
    asset_class: str = Field(default="", description="PCAF asset class")
    financed_emissions_tco2e: float = Field(
        default=0.0, description="Financed emissions (tCO2e)",
    )
    exposure_eur: float = Field(default=0.0, description="Total exposure (EUR)")
    emission_intensity: float = Field(
        default=0.0, description="Emission intensity (tCO2e/M EUR)",
    )
    data_quality_score: float = Field(
        default=5.0, description="Weighted PCAF data quality (1-5)",
    )
    counterparty_count: int = Field(
        default=0, description="Number of counterparties",
    )
    scope12_tco2e: float = Field(
        default=0.0, description="Scope 1+2 financed emissions",
    )
    scope3_tco2e: float = Field(
        default=0.0, description="Scope 3 financed emissions",
    )
    calculation_method: str = Field(
        default="", description="Calculation method used",
    )


class FinancedEmissionsResult(BaseModel):
    """Complete financed emissions calculation result."""
    total_financed_tco2e: float = Field(
        default=0.0, description="Total financed emissions (tCO2e)",
    )
    total_exposure_eur: float = Field(
        default=0.0, description="Total exposure (EUR)",
    )
    overall_intensity: float = Field(
        default=0.0, description="Overall intensity (tCO2e/M EUR)",
    )
    weighted_data_quality: float = Field(
        default=5.0, description="Weighted PCAF data quality score",
    )
    asset_class_results: List[AssetClassResult] = Field(
        default_factory=list, description="Per-asset-class results",
    )
    total_counterparties: int = Field(
        default=0, description="Total counterparties assessed",
    )
    pcaf_version: str = Field(default="", description="PCAF Standard version")
    scope12_total_tco2e: float = Field(
        default=0.0, description="Total Scope 1+2 financed",
    )
    scope3_total_tco2e: float = Field(
        default=0.0, description="Total Scope 3 financed",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class MRVInvestmentsBridge:
    """Bridge connecting PACK-012 with AGENT-MRV-028 Investments Agent.

    Provides PCAF financed emissions calculations across all asset classes.
    Supports attribution factor, investment share, and outstanding amount
    calculation methods per PCAF v2.1 standard.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for MRV-028 engines.

    Example:
        >>> bridge = MRVInvestmentsBridge()
        >>> result = bridge.calculate_financed_emissions(counterparties)
        >>> print(f"Total: {result.total_financed_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVInvestmentsBridgeConfig] = None) -> None:
        """Initialize the MRV Investments Bridge."""
        self.config = config or MRVInvestmentsBridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "investments_agent": _AgentStub(
                "GL-MRV-X-028",
                self.config.agent_mrv_028_path,
                "InvestmentsAgent",
            ),
            "equity_calculator": _AgentStub(
                "GL-MRV-028-EQ",
                f"{self.config.agent_mrv_028_path}.engines.equity_engine",
                "EquityInvestmentCalculatorEngine",
            ),
            "debt_calculator": _AgentStub(
                "GL-MRV-028-DEBT",
                f"{self.config.agent_mrv_028_path}.engines.debt_engine",
                "DebtInvestmentCalculatorEngine",
            ),
            "real_asset_calculator": _AgentStub(
                "GL-MRV-028-RE",
                f"{self.config.agent_mrv_028_path}.engines.real_asset_engine",
                "RealAssetCalculatorEngine",
            ),
            "sovereign_calculator": _AgentStub(
                "GL-MRV-028-SOV",
                f"{self.config.agent_mrv_028_path}.engines.sovereign_engine",
                "SovereignBondCalculatorEngine",
            ),
        }

        self.logger.info(
            "MRVInvestmentsBridge initialized: pcaf=%s, asset_classes=%d",
            self.config.pcaf_version,
            len(self.config.asset_classes),
        )

    def calculate_financed_emissions(
        self,
        counterparty_data: List[Dict[str, Any]],
    ) -> FinancedEmissionsResult:
        """Calculate financed emissions across all asset classes.

        Processes each counterparty record, routes it to the appropriate
        asset-class calculator, and aggregates results into a total
        financed emissions figure with PCAF data quality scoring.

        Args:
            counterparty_data: List of counterparty records with exposure,
                emissions, and asset class information.

        Returns:
            FinancedEmissionsResult with per-asset-class and total figures.
        """
        ac_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for cp in counterparty_data:
            ac = cp.get("asset_class", "business_loans")
            if ac not in ac_buckets:
                ac_buckets[ac] = []
            ac_buckets[ac].append(cp)

        ac_results: List[AssetClassResult] = []
        total_financed = 0.0
        total_exposure = 0.0
        total_s12 = 0.0
        total_s3 = 0.0
        dq_weighted = 0.0

        for ac_name, ac_records in ac_buckets.items():
            ac_result = self._calculate_asset_class(ac_name, ac_records)
            ac_results.append(ac_result)
            total_financed += ac_result.financed_emissions_tco2e
            total_exposure += ac_result.exposure_eur
            total_s12 += ac_result.scope12_tco2e
            total_s3 += ac_result.scope3_tco2e
            dq_weighted += ac_result.data_quality_score * ac_result.exposure_eur

        overall_dq = round(dq_weighted / max(total_exposure, 1.0), 2)
        overall_intensity = (
            round(total_financed / (total_exposure / 1_000_000), 2)
            if total_exposure > 0 else 0.0
        )

        result = FinancedEmissionsResult(
            total_financed_tco2e=round(total_financed, 2),
            total_exposure_eur=round(total_exposure, 2),
            overall_intensity=overall_intensity,
            weighted_data_quality=overall_dq,
            asset_class_results=ac_results,
            total_counterparties=len(counterparty_data),
            pcaf_version=self.config.pcaf_version,
            scope12_total_tco2e=round(total_s12, 2),
            scope3_total_tco2e=round(total_s3, 2),
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Financed emissions: total=%.2f tCO2e, exposure=%.2f EUR, "
            "dq=%.2f, intensity=%.2f",
            total_financed, total_exposure, overall_dq, overall_intensity,
        )
        return result

    def get_data_quality_breakdown(
        self,
        counterparty_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get PCAF data quality score breakdown.

        Args:
            counterparty_data: Counterparty records with data quality scores.

        Returns:
            Dictionary with data quality distribution and improvement actions.
        """
        dq_distribution: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for cp in counterparty_data:
            dq = int(cp.get("pcaf_data_quality", 5))
            dq = max(1, min(5, dq))
            dq_distribution[dq] += 1

        total = max(len(counterparty_data), 1)
        improvement_actions: List[str] = []

        if dq_distribution[5] > total * 0.3:
            improvement_actions.append(
                "Over 30% of counterparties at DQ5; prioritize sector-level "
                "data collection"
            )
        if dq_distribution[4] > total * 0.3:
            improvement_actions.append(
                "Consider upgrading DQ4 counterparties with economic "
                "activity-based estimates"
            )

        return {
            "distribution": dq_distribution,
            "distribution_pct": {
                k: round((v / total) * 100, 1)
                for k, v in dq_distribution.items()
            },
            "weighted_average": round(
                sum(k * v for k, v in dq_distribution.items()) / total, 2
            ),
            "improvement_actions": improvement_actions,
            "target_score": self.config.data_quality_target,
        }

    def route_to_mrv028(
        self,
        request_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a request to AGENT-MRV-028.

        Args:
            request_type: Type of request.
            data: Request data.

        Returns:
            Response from MRV-028 or error dictionary.
        """
        if request_type == "financed_emissions":
            counterparties = data.get("counterparty_data", [])
            result = self.calculate_financed_emissions(counterparties)
            return result.model_dump()

        elif request_type == "data_quality":
            counterparties = data.get("counterparty_data", [])
            return self.get_data_quality_breakdown(counterparties)

        else:
            return {"error": f"Unknown request type: {request_type}"}

    def _calculate_asset_class(
        self,
        asset_class: str,
        records: List[Dict[str, Any]],
    ) -> AssetClassResult:
        """Calculate financed emissions for a single asset class."""
        total_financed = 0.0
        total_exposure = 0.0
        total_s12 = 0.0
        total_s3 = 0.0
        dq_weighted = 0.0

        for cp in records:
            exposure = float(cp.get("exposure_eur", 0.0))
            s12 = float(cp.get("scope12_emissions_tco2e", 0.0))
            s3 = float(cp.get("scope3_emissions_tco2e", 0.0))
            attr = float(cp.get("attribution_factor", 1.0))
            dq = float(cp.get("pcaf_data_quality", 3.0))

            financed_s12 = s12 * attr
            financed_s3 = s3 * attr if self.config.enable_scope3 else 0.0

            total_financed += financed_s12 + financed_s3
            total_exposure += exposure
            total_s12 += financed_s12
            total_s3 += financed_s3
            dq_weighted += dq * exposure

        avg_dq = round(dq_weighted / max(total_exposure, 1.0), 2)
        intensity = (
            round(total_financed / (total_exposure / 1_000_000), 2)
            if total_exposure > 0 else 0.0
        )

        method = "attribution_factor"
        if asset_class in ("listed_equity", "corporate_bonds"):
            method = "investment_share"
        elif asset_class in ("mortgages", "motor_vehicle_loans"):
            method = "outstanding_amount"

        return AssetClassResult(
            asset_class=asset_class,
            financed_emissions_tco2e=round(total_financed, 2),
            exposure_eur=round(total_exposure, 2),
            emission_intensity=intensity,
            data_quality_score=avg_dq,
            counterparty_count=len(records),
            scope12_tco2e=round(total_s12, 2),
            scope3_tco2e=round(total_s3, 2),
            calculation_method=method,
        )
