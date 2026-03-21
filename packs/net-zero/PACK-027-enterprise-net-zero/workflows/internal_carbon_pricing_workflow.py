# -*- coding: utf-8 -*-
"""
Internal Carbon Pricing Workflow
====================================

4-phase workflow for setting and applying internal carbon price within
PACK-027 Enterprise Net Zero Pack.

Phases:
    1. PriceDesign          -- Design carbon pricing approach & price level
    2. AllocationSetup      -- Configure allocation to BUs and product lines
    3. ImpactAnalysis       -- Analyze impact on investments, margins, BU performance
    4. Reporting            -- Generate carbon pricing report & ESRS E1-8 disclosure

Uses: carbon_pricing_engine, financial_integration_engine.

Zero-hallucination: deterministic financial calculations.
SHA-256 provenance hashes.

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
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

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CarbonPricingApproach(str, Enum):
    SHADOW_PRICE = "shadow_price"
    INTERNAL_FEE = "internal_fee"
    IMPLICIT_PRICE = "implicit_price"
    REGULATORY_PRICE = "regulatory_price"


# =============================================================================
# CARBON PRICE TRAJECTORY BENCHMARKS
# =============================================================================

PRICE_TRAJECTORY_BENCHMARKS: Dict[str, Dict[int, float]] = {
    "low": {2025: 30, 2030: 50, 2035: 75, 2040: 100, 2050: 150},
    "medium": {2025: 60, 2030: 100, 2035: 150, 2040: 200, 2050: 300},
    "high": {2025: 100, 2030: 200, 2035: 300, 2040: 400, 2050: 500},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class BusinessUnit(BaseModel):
    bu_id: str = Field(...)
    bu_name: str = Field(default="")
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_allocated_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    ebitda_usd: float = Field(default=0.0)
    employee_count: int = Field(default=0, ge=0)


class InvestmentProposal(BaseModel):
    project_id: str = Field(...)
    project_name: str = Field(default="")
    capex_usd: float = Field(default=0.0, ge=0.0)
    annual_opex_usd: float = Field(default=0.0, ge=0.0)
    annual_emission_change_tco2e: float = Field(default=0.0, description="+increase/-decrease")
    project_life_years: int = Field(default=10, ge=1, le=50)
    standard_npv_usd: float = Field(default=0.0)
    standard_irr_pct: float = Field(default=0.0)
    discount_rate_pct: float = Field(default=8.0)


class CarbonAdjustedInvestment(BaseModel):
    project_id: str = Field(...)
    project_name: str = Field(default="")
    standard_npv_usd: float = Field(default=0.0)
    carbon_adjusted_npv_usd: float = Field(default=0.0)
    carbon_cost_impact_usd: float = Field(default=0.0)
    standard_irr_pct: float = Field(default=0.0)
    carbon_adjusted_irr_pct: float = Field(default=0.0)
    payback_change_months: int = Field(default=0)
    ranking_change: int = Field(default=0, description="Positive=moved up, negative=moved down")


class BUCarbonAllocation(BaseModel):
    bu_id: str = Field(...)
    bu_name: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    carbon_charge_usd: float = Field(default=0.0)
    carbon_intensity_tco2e_per_musd: float = Field(default=0.0)
    ebitda_before_carbon: float = Field(default=0.0)
    ebitda_after_carbon: float = Field(default=0.0)
    ebitda_impact_pct: float = Field(default=0.0)


class CBAMExposure(BaseModel):
    product_category: str = Field(default="")
    import_origin: str = Field(default="")
    embedded_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    cbam_certificate_price_eur: float = Field(default=0.0, ge=0.0)
    annual_cbam_cost_eur: float = Field(default=0.0, ge=0.0)


class InternalCarbonPricingConfig(BaseModel):
    pricing_approach: str = Field(default="shadow_price")
    carbon_price_usd_per_tco2e: float = Field(default=100.0, ge=0.0, le=1000.0)
    escalation_pct_per_year: float = Field(default=5.0, ge=0.0, le=50.0)
    price_trajectory: str = Field(default="medium", description="low|medium|high|custom")
    scope_covered: str = Field(default="scope_1_2", description="scope_1_2|scope_1_2_3")
    cbam_enabled: bool = Field(default=True)
    ets_positions: Dict[str, float] = Field(default_factory=dict)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class InternalCarbonPricingInput(BaseModel):
    config: InternalCarbonPricingConfig = Field(default_factory=InternalCarbonPricingConfig)
    business_units: List[BusinessUnit] = Field(default_factory=list)
    investment_proposals: List[InvestmentProposal] = Field(default_factory=list)
    cbam_products: List[Dict[str, Any]] = Field(default_factory=list)
    total_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    total_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)


class InternalCarbonPricingResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="enterprise_internal_carbon_pricing")
    pack_id: str = Field(default="PACK-027")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    carbon_price_usd: float = Field(default=0.0)
    price_trajectory: Dict[int, float] = Field(default_factory=dict)
    bu_allocations: List[BUCarbonAllocation] = Field(default_factory=list)
    investment_appraisals: List[CarbonAdjustedInvestment] = Field(default_factory=list)
    cbam_exposures: List[CBAMExposure] = Field(default_factory=list)
    total_carbon_charge_usd: float = Field(default=0.0)
    total_cbam_cost_eur: float = Field(default=0.0)
    esrs_e1_8_disclosure: Dict[str, Any] = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class InternalCarbonPricingWorkflow:
    """
    4-phase internal carbon pricing workflow.

    Phase 1: Price Design -- Set carbon price level and escalation.
    Phase 2: Allocation Setup -- Allocate carbon costs to BUs and products.
    Phase 3: Impact Analysis -- Analyze investment, margin, and BU impact.
    Phase 4: Reporting -- Generate carbon pricing report and ESRS E1-8.

    Example:
        >>> wf = InternalCarbonPricingWorkflow()
        >>> inp = InternalCarbonPricingInput(
        ...     business_units=[BusinessUnit(bu_id="bu-01", scope1_tco2e=5000)],
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[InternalCarbonPricingConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or InternalCarbonPricingConfig()
        self._phase_results: List[PhaseResult] = []
        self._bu_allocations: List[BUCarbonAllocation] = []
        self._investments: List[CarbonAdjustedInvestment] = []
        self._cbam: List[CBAMExposure] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: InternalCarbonPricingInput) -> InternalCarbonPricingResult:
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_price_design(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_allocation_setup(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_impact_analysis(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_reporting(input_data)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Carbon pricing workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        trajectory = PRICE_TRAJECTORY_BENCHMARKS.get(
            self.config.price_trajectory, PRICE_TRAJECTORY_BENCHMARKS["medium"],
        )

        total_charge = sum(a.carbon_charge_usd for a in self._bu_allocations)
        total_cbam = sum(c.annual_cbam_cost_eur for c in self._cbam)

        esrs = self._build_esrs_e1_8(input_data, trajectory, total_charge)

        result = InternalCarbonPricingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            carbon_price_usd=self.config.carbon_price_usd_per_tco2e,
            price_trajectory=trajectory,
            bu_allocations=self._bu_allocations,
            investment_appraisals=self._investments,
            cbam_exposures=self._cbam,
            total_carbon_charge_usd=round(total_charge, 2),
            total_cbam_cost_eur=round(total_cbam, 2),
            esrs_e1_8_disclosure=esrs,
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    async def _phase_price_design(self, input_data: InternalCarbonPricingInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        price = self.config.carbon_price_usd_per_tco2e
        approach = self.config.pricing_approach

        if price < 50:
            warnings.append(
                f"Carbon price (${price}/tCO2e) is below IEA 2030 1.5C recommendation ($130/tCO2e)"
            )

        trajectory = PRICE_TRAJECTORY_BENCHMARKS.get(
            self.config.price_trajectory, PRICE_TRAJECTORY_BENCHMARKS["medium"],
        )

        outputs["pricing_approach"] = approach
        outputs["current_price_usd"] = price
        outputs["escalation_pct_yr"] = self.config.escalation_pct_per_year
        outputs["price_trajectory"] = trajectory
        outputs["scope_covered"] = self.config.scope_covered
        outputs["benchmarks"] = {
            "eu_ets_current": "~$60-80/tCO2e",
            "iea_nze_2030": "$130/tCO2e",
            "iea_nze_2050": "$250/tCO2e",
        }

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="price_design", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_price_design",
        )

    async def _phase_allocation_setup(self, input_data: InternalCarbonPricingInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        price = self.config.carbon_price_usd_per_tco2e
        self._bu_allocations = []

        for bu in input_data.business_units:
            if self.config.scope_covered == "scope_1_2_3":
                total_em = bu.scope1_tco2e + bu.scope2_tco2e + bu.scope3_allocated_tco2e
            else:
                total_em = bu.scope1_tco2e + bu.scope2_tco2e

            charge = total_em * price
            intensity = (total_em / (bu.revenue_usd / 1_000_000.0)) if bu.revenue_usd > 0 else 0.0
            ebitda_after = bu.ebitda_usd - charge
            ebitda_impact = (charge / bu.ebitda_usd * 100.0) if bu.ebitda_usd != 0 else 0.0

            alloc = BUCarbonAllocation(
                bu_id=bu.bu_id,
                bu_name=bu.bu_name,
                total_emissions_tco2e=round(total_em, 2),
                carbon_charge_usd=round(charge, 2),
                carbon_intensity_tco2e_per_musd=round(intensity, 2),
                ebitda_before_carbon=round(bu.ebitda_usd, 2),
                ebitda_after_carbon=round(ebitda_after, 2),
                ebitda_impact_pct=round(ebitda_impact, 2),
            )
            self._bu_allocations.append(alloc)

        outputs["business_units_allocated"] = len(self._bu_allocations)
        outputs["total_emissions_covered"] = round(
            sum(a.total_emissions_tco2e for a in self._bu_allocations), 2,
        )
        outputs["total_carbon_charge_usd"] = round(
            sum(a.carbon_charge_usd for a in self._bu_allocations), 2,
        )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="allocation_setup", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_allocation_setup",
        )

    async def _phase_impact_analysis(self, input_data: InternalCarbonPricingInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        price = self.config.carbon_price_usd_per_tco2e
        self._investments = []

        for prop in input_data.investment_proposals:
            # Carbon cost over project life
            annual_carbon_cost = prop.annual_emission_change_tco2e * price
            total_carbon_cost = annual_carbon_cost * prop.project_life_years

            # Simplified carbon-adjusted NPV
            discount_rate = prop.discount_rate_pct / 100.0
            pv_carbon = 0.0
            for yr in range(1, prop.project_life_years + 1):
                escalated_price = price * ((1 + self.config.escalation_pct_per_year / 100.0) ** yr)
                annual_cost = prop.annual_emission_change_tco2e * escalated_price
                pv_carbon += annual_cost / ((1 + discount_rate) ** yr)

            carbon_adjusted_npv = prop.standard_npv_usd - pv_carbon

            adj = CarbonAdjustedInvestment(
                project_id=prop.project_id,
                project_name=prop.project_name,
                standard_npv_usd=round(prop.standard_npv_usd, 2),
                carbon_adjusted_npv_usd=round(carbon_adjusted_npv, 2),
                carbon_cost_impact_usd=round(pv_carbon, 2),
                standard_irr_pct=round(prop.standard_irr_pct, 2),
                carbon_adjusted_irr_pct=round(
                    prop.standard_irr_pct - (pv_carbon / max(prop.capex_usd, 1)) * 100 * 0.1, 2,
                ),
                payback_change_months=int(abs(pv_carbon) / max(prop.capex_usd / 120, 1)),
                ranking_change=1 if prop.annual_emission_change_tco2e < 0 else -1,
            )
            self._investments.append(adj)

        # CBAM exposure
        self._cbam = []
        if self.config.cbam_enabled:
            for prod in input_data.cbam_products:
                cbam = CBAMExposure(
                    product_category=prod.get("category", ""),
                    import_origin=prod.get("origin", ""),
                    embedded_emissions_tco2e=float(prod.get("embedded_tco2e", 0)),
                    cbam_certificate_price_eur=float(prod.get("certificate_price_eur", 80)),
                    annual_cbam_cost_eur=round(
                        float(prod.get("embedded_tco2e", 0)) *
                        float(prod.get("certificate_price_eur", 80)), 2,
                    ),
                )
                self._cbam.append(cbam)

        outputs["investments_appraised"] = len(self._investments)
        outputs["ranking_changes"] = sum(1 for i in self._investments if i.ranking_change != 0)
        outputs["cbam_products_assessed"] = len(self._cbam)
        outputs["total_cbam_cost_eur"] = round(sum(c.annual_cbam_cost_eur for c in self._cbam), 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="impact_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_impact_analysis",
        )

    async def _phase_reporting(self, input_data: InternalCarbonPricingInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        outputs["report_sections"] = [
            "Internal Carbon Price Design",
            "Price Trajectory & Benchmarking",
            "Business Unit Allocation & Impact",
            "Investment Appraisal with Carbon Price",
            "CBAM Exposure Assessment",
            "Carbon-Adjusted Financial Metrics",
            "ESRS E1-8 Disclosure",
            "Recommendations",
        ]
        outputs["report_formats"] = ["MD", "HTML", "JSON", "PDF"]
        outputs["esrs_e1_8_included"] = True

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="reporting", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_reporting",
        )

    def _build_esrs_e1_8(
        self, input_data: InternalCarbonPricingInput,
        trajectory: Dict[int, float], total_charge: float,
    ) -> Dict[str, Any]:
        """Build ESRS E1-8 internal carbon pricing disclosure."""
        total_emissions = (
            input_data.total_scope1_tco2e + input_data.total_scope2_tco2e
        )
        if self.config.scope_covered == "scope_1_2_3":
            total_emissions += input_data.total_scope3_tco2e

        return {
            "datapoint": "ESRS E1-8",
            "description": "Internal carbon pricing",
            "carbon_price_type": self.config.pricing_approach,
            "carbon_price_level_usd_per_tco2e": self.config.carbon_price_usd_per_tco2e,
            "scope_of_application": self.config.scope_covered,
            "emissions_covered_tco2e": round(total_emissions, 2),
            "total_revenue_generated_usd": round(total_charge, 2),
            "price_trajectory": trajectory,
            "methodology": (
                "Shadow pricing applied to all capital allocation decisions "
                "and business unit performance reporting"
            ),
            "benchmark_reference": "IEA Net Zero Emissions by 2050 Scenario",
        }

    def _generate_next_steps(self) -> List[str]:
        return [
            "Present carbon pricing framework to CFO and board for approval.",
            "Integrate carbon price into capital allocation process.",
            "Communicate BU carbon charges to BU heads.",
            "Set up quarterly carbon P&L reporting.",
            "Review and adjust carbon price level annually.",
            "Monitor CBAM transitional phase developments.",
        ]
