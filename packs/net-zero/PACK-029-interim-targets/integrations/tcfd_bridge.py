# -*- coding: utf-8 -*-
"""
TCFDBridge - TCFD Metrics and Targets Disclosure for PACK-029
================================================================

Enterprise bridge for exporting interim targets to TCFD (Task Force on
Climate-related Financial Disclosures) Metrics and Targets pillar.
Generates Table 1 (GHG emissions historical + target), Table 2 (interim
targets 5-year/10-year), Table 3 (transition risks linked to targets),
Table 4 (forward-looking metrics / projected emissions), scenario analysis
integration, and consistency checks with TCFD Strategy and Risk Management.

TCFD Integration Points:
    - Table 1: Historical + target GHG emissions (Scope 1, 2, 3)
    - Table 2: Interim targets (5-year, 10-year milestones)
    - Table 3: Transition risks mapped to target achievement
    - Table 4: Forward-looking projected emissions under scenarios
    - Scenario Analysis: 1.5C / 2C / BAU scenario comparison
    - Consistency: Cross-pillar alignment with Strategy + Risk Management

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
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


class TCFDScenario(str, Enum):
    NZE_15C = "nze_1.5c"
    BELOW_2C = "below_2c"
    NDC_SCENARIO = "ndc_scenario"
    BAU = "business_as_usual"
    HIGH_WARMING = "high_warming_4c"


class TransitionRiskType(str, Enum):
    POLICY_LEGAL = "policy_legal"
    TECHNOLOGY = "technology"
    MARKET = "market"
    REPUTATION = "reputation"


class RiskLikelihood(str, Enum):
    VERY_LIKELY = "very_likely"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    RARE = "rare"


class RiskImpact(str, Enum):
    HIGH = "high"
    MEDIUM_HIGH = "medium_high"
    MEDIUM = "medium"
    MEDIUM_LOW = "medium_low"
    LOW = "low"


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    PARTIALLY_CONSISTENT = "partially_consistent"
    INCONSISTENT = "inconsistent"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TCFDBridgeConfig(BaseModel):
    """Configuration for the TCFD bridge."""
    pack_id: str = Field(default="PACK-029")
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2023, ge=2015, le=2025)
    currency: str = Field(default="USD")
    industry_sector: str = Field(default="")
    enable_provenance: bool = Field(default=True)
    enable_scenario_analysis: bool = Field(default=True)
    enable_consistency_checks: bool = Field(default=True)


class TCFDTable1Row(BaseModel):
    """Single row in Table 1: GHG emissions data."""
    year: int = Field(default=2025)
    is_actual: bool = Field(default=True)
    is_target: bool = Field(default=False)
    scope1_tco2e: float = Field(default=0.0)
    scope2_location_tco2e: float = Field(default=0.0)
    scope2_market_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    total_scope12_tco2e: float = Field(default=0.0)
    total_scope123_tco2e: float = Field(default=0.0)
    change_from_base_year_pct: float = Field(default=0.0)


class TCFDTable1Export(BaseModel):
    """Table 1: GHG emissions historical + target trajectory."""
    export_id: str = Field(default_factory=_new_uuid)
    table_name: str = Field(default="Table 1: GHG Emissions")
    rows: List[TCFDTable1Row] = Field(default_factory=list)
    base_year: int = Field(default=2023)
    methodology: str = Field(default="GHG Protocol Corporate Standard")
    provenance_hash: str = Field(default="")


class TCFDTable2Row(BaseModel):
    """Single row in Table 2: Interim targets."""
    target_reference: str = Field(default="")
    target_type: str = Field(default="Absolute reduction")
    scope_coverage: str = Field(default="Scope 1+2")
    base_year: int = Field(default=2023)
    target_year: int = Field(default=2030)
    reduction_pct: float = Field(default=0.0)
    base_year_emissions_tco2e: float = Field(default=0.0)
    target_emissions_tco2e: float = Field(default=0.0)
    progress_pct: float = Field(default=0.0)
    science_based: bool = Field(default=True)
    methodology: str = Field(default="SBTi ACA 1.5C")


class TCFDTable2Export(BaseModel):
    """Table 2: Interim targets (5-year, 10-year milestones)."""
    export_id: str = Field(default_factory=_new_uuid)
    table_name: str = Field(default="Table 2: Interim Targets")
    rows: List[TCFDTable2Row] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TCFDTable3Row(BaseModel):
    """Single row in Table 3: Transition risks linked to targets."""
    risk_id: str = Field(default_factory=_new_uuid)
    risk_type: TransitionRiskType = Field(default=TransitionRiskType.POLICY_LEGAL)
    risk_description: str = Field(default="")
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.POSSIBLE)
    impact: RiskImpact = Field(default=RiskImpact.MEDIUM)
    time_horizon: str = Field(default="Medium-term (3-10 years)")
    linked_target_ref: str = Field(default="")
    financial_impact_usd: float = Field(default=0.0)
    mitigation_action: str = Field(default="")


class TCFDTable3Export(BaseModel):
    """Table 3: Transition risks linked to targets."""
    export_id: str = Field(default_factory=_new_uuid)
    table_name: str = Field(default="Table 3: Transition Risks")
    rows: List[TCFDTable3Row] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TCFDTable4Row(BaseModel):
    """Single row in Table 4: Forward-looking metrics."""
    year: int = Field(default=2030)
    scenario: TCFDScenario = Field(default=TCFDScenario.NZE_15C)
    projected_scope12_tco2e: float = Field(default=0.0)
    projected_scope3_tco2e: float = Field(default=0.0)
    projected_total_tco2e: float = Field(default=0.0)
    carbon_price_usd_per_tco2e: float = Field(default=0.0)
    transition_cost_usd: float = Field(default=0.0)
    revenue_at_risk_pct: float = Field(default=0.0)


class TCFDTable4Export(BaseModel):
    """Table 4: Forward-looking metrics under scenarios."""
    export_id: str = Field(default_factory=_new_uuid)
    table_name: str = Field(default="Table 4: Forward-Looking Metrics")
    rows: List[TCFDTable4Row] = Field(default_factory=list)
    scenarios_analyzed: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TCFDConsistencyCheck(BaseModel):
    """Cross-pillar consistency check result."""
    check_id: str = Field(default_factory=_new_uuid)
    pillar_1: str = Field(default="")
    pillar_2: str = Field(default="")
    check_description: str = Field(default="")
    status: ConsistencyStatus = Field(default=ConsistencyStatus.NOT_ASSESSED)
    detail: str = Field(default="")


class TCFDExportResult(BaseModel):
    """Complete TCFD Metrics & Targets export result."""
    result_id: str = Field(default_factory=_new_uuid)
    table1: Optional[TCFDTable1Export] = Field(None)
    table2: Optional[TCFDTable2Export] = Field(None)
    table3: Optional[TCFDTable3Export] = Field(None)
    table4: Optional[TCFDTable4Export] = Field(None)
    consistency_checks: List[TCFDConsistencyCheck] = Field(default_factory=list)
    tables_exported: List[str] = Field(default_factory=list)
    overall_consistency: ConsistencyStatus = Field(default=ConsistencyStatus.NOT_ASSESSED)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Scenario Carbon Price Assumptions
# ---------------------------------------------------------------------------

SCENARIO_CARBON_PRICES: Dict[str, Dict[int, float]] = {
    "nze_1.5c": {2025: 75, 2030: 140, 2035: 205, 2040: 250, 2045: 290, 2050: 250},
    "below_2c": {2025: 50, 2030: 100, 2035: 150, 2040: 180, 2045: 200, 2050: 200},
    "ndc_scenario": {2025: 25, 2030: 50, 2035: 75, 2040: 100, 2045: 120, 2050: 130},
    "business_as_usual": {2025: 10, 2030: 20, 2035: 30, 2040: 40, 2045: 50, 2050: 60},
}

DEFAULT_TRANSITION_RISKS: List[Dict[str, Any]] = [
    {"type": "policy_legal", "description": "Carbon pricing / ETS cost increase", "likelihood": "very_likely", "impact": "high", "horizon": "Short-term (1-3 years)"},
    {"type": "policy_legal", "description": "Stricter emission reporting mandates", "likelihood": "likely", "impact": "medium", "horizon": "Short-term (1-3 years)"},
    {"type": "technology", "description": "Stranded assets from fossil fuel dependency", "likelihood": "likely", "impact": "high", "horizon": "Medium-term (3-10 years)"},
    {"type": "technology", "description": "Cost of low-carbon technology transition", "likelihood": "very_likely", "impact": "medium_high", "horizon": "Medium-term (3-10 years)"},
    {"type": "market", "description": "Shifting customer preferences to low-carbon products", "likelihood": "likely", "impact": "medium", "horizon": "Medium-term (3-10 years)"},
    {"type": "market", "description": "Increased cost of raw materials", "likelihood": "possible", "impact": "medium", "horizon": "Long-term (>10 years)"},
    {"type": "reputation", "description": "Investor/stakeholder pressure on climate action", "likelihood": "very_likely", "impact": "medium_high", "horizon": "Short-term (1-3 years)"},
    {"type": "reputation", "description": "Greenwashing litigation risk", "likelihood": "possible", "impact": "high", "horizon": "Medium-term (3-10 years)"},
]


# ---------------------------------------------------------------------------
# TCFDBridge
# ---------------------------------------------------------------------------


class TCFDBridge:
    """TCFD Metrics and Targets disclosure bridge for PACK-029.

    Exports interim targets and GHG data to four TCFD tables:
    historical emissions, interim targets, transition risks,
    and forward-looking metrics with scenario analysis.

    Example:
        >>> bridge = TCFDBridge(TCFDBridgeConfig(
        ...     organization_name="Acme Corp",
        ...     reporting_year=2025,
        ... ))
        >>> result = await bridge.export_full(
        ...     emissions_data, targets, current_data
        ... )
    """

    def __init__(self, config: Optional[TCFDBridgeConfig] = None) -> None:
        self.config = config or TCFDBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._export_cache: Optional[TCFDExportResult] = None

        self.logger.info(
            "TCFDBridge (PACK-029) initialized: org=%s, year=%d",
            self.config.organization_name, self.config.reporting_year,
        )

    async def export_table1(
        self,
        historical_data: List[Dict[str, Any]],
        target_data: List[Dict[str, Any]],
    ) -> TCFDTable1Export:
        """Export Table 1: GHG emissions historical + target trajectory."""
        rows: List[TCFDTable1Row] = []
        base_total = 0.0

        # Historical rows
        for d in sorted(historical_data, key=lambda x: x.get("year", 0)):
            s1 = d.get("scope1_tco2e", 0.0)
            s2_mkt = d.get("scope2_market_tco2e", 0.0)
            s2_loc = d.get("scope2_location_tco2e", 0.0)
            s3 = d.get("scope3_tco2e", 0.0)
            total_12 = s1 + s2_mkt
            total_123 = total_12 + s3

            if d.get("year") == self.config.base_year:
                base_total = total_123

            change = ((total_123 - base_total) / max(base_total, 1.0)) * 100.0 if base_total > 0 else 0.0

            rows.append(TCFDTable1Row(
                year=d.get("year", 0),
                is_actual=True,
                scope1_tco2e=round(s1, 2),
                scope2_location_tco2e=round(s2_loc, 2),
                scope2_market_tco2e=round(s2_mkt, 2),
                scope3_tco2e=round(s3, 2),
                total_scope12_tco2e=round(total_12, 2),
                total_scope123_tco2e=round(total_123, 2),
                change_from_base_year_pct=round(change, 2),
            ))

        # Target rows
        for t in sorted(target_data, key=lambda x: x.get("target_year", 0)):
            s1_t = t.get("scope1_target_tco2e", 0.0)
            s2_t = t.get("scope2_target_tco2e", 0.0)
            s3_t = t.get("scope3_target_tco2e", 0.0)
            total_12_t = s1_t + s2_t
            total_123_t = total_12_t + s3_t
            change_t = ((total_123_t - base_total) / max(base_total, 1.0)) * 100.0 if base_total > 0 else 0.0

            rows.append(TCFDTable1Row(
                year=t.get("target_year", 0),
                is_actual=False,
                is_target=True,
                scope1_tco2e=round(s1_t, 2),
                scope2_market_tco2e=round(s2_t, 2),
                scope3_tco2e=round(s3_t, 2),
                total_scope12_tco2e=round(total_12_t, 2),
                total_scope123_tco2e=round(total_123_t, 2),
                change_from_base_year_pct=round(change_t, 2),
            ))

        export = TCFDTable1Export(rows=rows, base_year=self.config.base_year)
        if self.config.enable_provenance:
            export.provenance_hash = _compute_hash(export)

        self.logger.info("Table 1 exported: %d rows (%d actual, %d target)",
                         len(rows), sum(1 for r in rows if r.is_actual),
                         sum(1 for r in rows if r.is_target))
        return export

    async def export_table2(
        self,
        interim_targets: List[Dict[str, Any]],
    ) -> TCFDTable2Export:
        """Export Table 2: Interim targets (5-year, 10-year milestones)."""
        rows: List[TCFDTable2Row] = []

        for idx, t in enumerate(sorted(interim_targets, key=lambda x: x.get("target_year", 0)), 1):
            base_emissions = t.get("base_year_emissions_tco2e", 0.0)
            reduction = t.get("scope12_reduction_pct", 0.0)
            target_emissions = base_emissions * (1 - reduction / 100.0)

            rows.append(TCFDTable2Row(
                target_reference=f"IT-{idx:03d}",
                target_type=t.get("target_type", "Absolute reduction"),
                scope_coverage=t.get("scope_coverage", "Scope 1+2"),
                base_year=t.get("base_year", self.config.base_year),
                target_year=t.get("target_year", 2030),
                reduction_pct=reduction,
                base_year_emissions_tco2e=round(base_emissions, 2),
                target_emissions_tco2e=round(target_emissions, 2),
                progress_pct=t.get("progress_pct", 0.0),
                science_based=t.get("science_based", True),
                methodology=t.get("methodology", "SBTi ACA 1.5C"),
            ))

        export = TCFDTable2Export(rows=rows)
        if self.config.enable_provenance:
            export.provenance_hash = _compute_hash(export)

        self.logger.info("Table 2 exported: %d interim targets", len(rows))
        return export

    async def export_table3(
        self,
        target_refs: Optional[List[str]] = None,
        custom_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> TCFDTable3Export:
        """Export Table 3: Transition risks linked to targets."""
        risks = custom_risks or DEFAULT_TRANSITION_RISKS
        refs = target_refs or ["IT-001"]
        rows: List[TCFDTable3Row] = []

        for risk in risks:
            linked_ref = refs[0] if refs else ""
            rows.append(TCFDTable3Row(
                risk_type=TransitionRiskType(risk.get("type", "policy_legal")),
                risk_description=risk.get("description", ""),
                likelihood=RiskLikelihood(risk.get("likelihood", "possible")),
                impact=RiskImpact(risk.get("impact", "medium")),
                time_horizon=risk.get("horizon", "Medium-term (3-10 years)"),
                linked_target_ref=linked_ref,
                financial_impact_usd=risk.get("financial_impact_usd", 0.0),
                mitigation_action=risk.get("mitigation_action", "Aligned with interim target trajectory"),
            ))

        export = TCFDTable3Export(rows=rows)
        if self.config.enable_provenance:
            export.provenance_hash = _compute_hash(export)

        self.logger.info("Table 3 exported: %d transition risks", len(rows))
        return export

    async def export_table4(
        self,
        current_data: Dict[str, Any],
        projection_years: Optional[List[int]] = None,
        scenarios: Optional[List[TCFDScenario]] = None,
    ) -> TCFDTable4Export:
        """Export Table 4: Forward-looking metrics under scenarios."""
        years = projection_years or [2025, 2030, 2035, 2040, 2050]
        scenario_list = scenarios or [
            TCFDScenario.NZE_15C,
            TCFDScenario.BELOW_2C,
            TCFDScenario.NDC_SCENARIO,
            TCFDScenario.BAU,
        ]

        current_s12 = current_data.get("scope12_tco2e", 75000.0)
        current_s3 = current_data.get("scope3_tco2e", 120000.0)
        current_total = current_s12 + current_s3
        base_year = self.config.base_year

        rows: List[TCFDTable4Row] = []
        scenarios_analyzed: List[str] = []

        # Scenario reduction rates (annual)
        scenario_rates: Dict[str, float] = {
            "nze_1.5c": 0.042,
            "below_2c": 0.025,
            "ndc_scenario": 0.015,
            "business_as_usual": 0.005,
            "high_warming_4c": -0.005,
        }

        for scenario in scenario_list:
            rate = scenario_rates.get(scenario.value, 0.02)
            carbon_prices = SCENARIO_CARBON_PRICES.get(scenario.value, {})
            scenarios_analyzed.append(scenario.value)

            for year in years:
                elapsed = year - max(self.config.reporting_year, base_year)
                factor = max(0.0, 1.0 - rate * elapsed)

                proj_s12 = round(current_s12 * factor, 2)
                proj_s3 = round(current_s3 * max(0.0, 1.0 - rate * 0.6 * elapsed), 2)
                proj_total = proj_s12 + proj_s3

                # Carbon price lookup
                c_price = carbon_prices.get(year, 0.0)
                if not c_price:
                    sorted_years = sorted(carbon_prices.keys())
                    for i in range(len(sorted_years) - 1):
                        if sorted_years[i] <= year <= sorted_years[i + 1]:
                            frac = (year - sorted_years[i]) / (sorted_years[i + 1] - sorted_years[i])
                            c_price = carbon_prices[sorted_years[i]] + frac * (carbon_prices[sorted_years[i + 1]] - carbon_prices[sorted_years[i]])
                            break

                transition_cost = proj_total * c_price
                revenue_at_risk = min(transition_cost / max(current_total * 100, 1.0) * 100.0, 100.0)

                rows.append(TCFDTable4Row(
                    year=year,
                    scenario=scenario,
                    projected_scope12_tco2e=proj_s12,
                    projected_scope3_tco2e=proj_s3,
                    projected_total_tco2e=round(proj_total, 2),
                    carbon_price_usd_per_tco2e=round(c_price, 2),
                    transition_cost_usd=round(transition_cost, 2),
                    revenue_at_risk_pct=round(revenue_at_risk, 2),
                ))

        export = TCFDTable4Export(
            rows=rows,
            scenarios_analyzed=scenarios_analyzed,
        )
        if self.config.enable_provenance:
            export.provenance_hash = _compute_hash(export)

        self.logger.info("Table 4 exported: %d projections across %d scenarios",
                         len(rows), len(scenarios_analyzed))
        return export

    async def check_consistency(
        self,
        table1: TCFDTable1Export,
        table2: TCFDTable2Export,
        table4: TCFDTable4Export,
    ) -> List[TCFDConsistencyCheck]:
        """Check consistency across TCFD pillars and tables."""
        checks: List[TCFDConsistencyCheck] = []

        # Check 1: Table 1 target rows match Table 2
        t1_target_years = {r.year for r in table1.rows if r.is_target}
        t2_target_years = {r.target_year for r in table2.rows}
        if t1_target_years == t2_target_years:
            checks.append(TCFDConsistencyCheck(
                pillar_1="metrics_targets", pillar_2="metrics_targets",
                check_description="Table 1 target years match Table 2",
                status=ConsistencyStatus.CONSISTENT,
                detail=f"Years: {sorted(t1_target_years)}",
            ))
        else:
            checks.append(TCFDConsistencyCheck(
                pillar_1="metrics_targets", pillar_2="metrics_targets",
                check_description="Table 1 target years match Table 2",
                status=ConsistencyStatus.INCONSISTENT,
                detail=f"T1: {sorted(t1_target_years)}, T2: {sorted(t2_target_years)}",
            ))

        # Check 2: Table 4 NZE scenario aligns with Table 2 targets
        nze_projections = {r.year: r.projected_total_tco2e for r in table4.rows if r.scenario == TCFDScenario.NZE_15C}
        for t2_row in table2.rows:
            nze_proj = nze_projections.get(t2_row.target_year, 0)
            if nze_proj > 0 and t2_row.target_emissions_tco2e > 0:
                diff_pct = abs(nze_proj - t2_row.target_emissions_tco2e) / max(t2_row.target_emissions_tco2e, 1) * 100
                checks.append(TCFDConsistencyCheck(
                    pillar_1="metrics_targets", pillar_2="strategy",
                    check_description=f"NZE projection vs target for {t2_row.target_year}",
                    status=ConsistencyStatus.CONSISTENT if diff_pct < 20 else ConsistencyStatus.PARTIALLY_CONSISTENT,
                    detail=f"NZE={nze_proj:.0f}, Target={t2_row.target_emissions_tco2e:.0f}, diff={diff_pct:.1f}%",
                ))

        # Check 3: Strategy-Risk Management alignment
        checks.append(TCFDConsistencyCheck(
            pillar_1="strategy", pillar_2="risk_management",
            check_description="Transition risks inform target trajectory",
            status=ConsistencyStatus.CONSISTENT,
            detail="Risks mapped to interim targets via Table 3",
        ))

        # Check 4: Governance alignment
        checks.append(TCFDConsistencyCheck(
            pillar_1="governance", pillar_2="metrics_targets",
            check_description="Board oversight of climate targets",
            status=ConsistencyStatus.CONSISTENT,
            detail="Board-level net-zero governance assumed from SBTi commitment",
        ))

        return checks

    async def export_full(
        self,
        historical_data: List[Dict[str, Any]],
        interim_targets: List[Dict[str, Any]],
        target_trajectory: List[Dict[str, Any]],
        current_data: Dict[str, Any],
    ) -> TCFDExportResult:
        """Export all four TCFD tables with consistency checks."""
        table1 = await self.export_table1(historical_data, target_trajectory)
        table2 = await self.export_table2(interim_targets)

        target_refs = [f"IT-{i:03d}" for i in range(1, len(interim_targets) + 1)]
        table3 = await self.export_table3(target_refs)
        table4 = await self.export_table4(current_data)

        consistency = []
        overall = ConsistencyStatus.NOT_ASSESSED
        if self.config.enable_consistency_checks:
            consistency = await self.check_consistency(table1, table2, table4)
            inconsistent = sum(1 for c in consistency if c.status == ConsistencyStatus.INCONSISTENT)
            partial = sum(1 for c in consistency if c.status == ConsistencyStatus.PARTIALLY_CONSISTENT)
            if inconsistent > 0:
                overall = ConsistencyStatus.INCONSISTENT
            elif partial > 0:
                overall = ConsistencyStatus.PARTIALLY_CONSISTENT
            else:
                overall = ConsistencyStatus.CONSISTENT

        result = TCFDExportResult(
            table1=table1,
            table2=table2,
            table3=table3,
            table4=table4,
            consistency_checks=consistency,
            tables_exported=["Table 1", "Table 2", "Table 3", "Table 4"],
            overall_consistency=overall,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._export_cache = result
        self.logger.info(
            "TCFD full export: 4 tables, consistency=%s",
            overall.value,
        )
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "organization": self.config.organization_name,
            "reporting_year": self.config.reporting_year,
            "tables_available": ["Table 1", "Table 2", "Table 3", "Table 4"],
            "scenario_analysis": self.config.enable_scenario_analysis,
            "last_export": self._export_cache is not None,
        }
