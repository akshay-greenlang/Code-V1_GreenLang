"""
GL-TCFD-APP Financial Impact API

Models the financial impact of climate-related risks and opportunities on
the income statement, balance sheet, and cash flow statement.  Provides
NPV and IRR analysis, MACC data, carbon price sensitivity, Monte Carlo
financial simulation, stranded asset valuation, and adaptation cost estimation.

TCFD Financial Impact Categories:
    - Income Statement: Revenue changes, cost increases, impairment charges
    - Balance Sheet: Asset revaluation, provisions, intangible write-downs
    - Cash Flow: Capex requirements, operating cost changes, insurance costs

ISSB/IFRS S2 references: paragraphs 16-21 (Financial effects).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import math
import random

router = APIRouter(prefix="/api/v1/tcfd/financial", tags=["Financial Impact"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class NPVRequest(BaseModel):
    """Request for NPV analysis."""
    org_id: str = Field(..., description="Organization ID")
    initial_investment_usd: float = Field(..., ge=0, description="Initial investment")
    annual_cash_flows: List[float] = Field(..., min_length=1, description="Annual cash flows (positive or negative)")
    discount_rate_pct: float = Field(8.0, ge=0, le=50, description="Discount rate (%)")
    terminal_value_usd: float = Field(0.0, ge=0, description="Terminal/residual value")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "initial_investment_usd": 10000000,
                "annual_cash_flows": [1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4000000, 4000000, 4000000, 4000000],
                "discount_rate_pct": 8.0,
                "terminal_value_usd": 5000000,
            }
        }


class IRRRequest(BaseModel):
    """Request for IRR analysis."""
    org_id: str = Field(..., description="Organization ID")
    initial_investment_usd: float = Field(..., ge=0, description="Initial investment")
    annual_cash_flows: List[float] = Field(..., min_length=1, description="Annual cash flows")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "initial_investment_usd": 10000000,
                "annual_cash_flows": [1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4000000, 4000000, 4000000, 4000000],
            }
        }


class CarbonSensitivityRequest(BaseModel):
    """Request for carbon price sensitivity analysis."""
    org_id: str = Field(..., description="Organization ID")
    scope1_tco2e: float = Field(25000, ge=0, description="Annual Scope 1 emissions")
    scope2_tco2e: float = Field(15000, ge=0, description="Annual Scope 2 emissions")
    annual_revenue_usd: float = Field(500000000, ge=0, description="Annual revenue")
    ebitda_margin_pct: float = Field(15.0, ge=0, le=100, description="EBITDA margin (%)")
    carbon_prices: List[float] = Field(
        default=[25, 50, 75, 100, 150, 200, 250, 300],
        description="Carbon prices to evaluate (USD/tCO2e)"
    )


class MonteCarloFinancialRequest(BaseModel):
    """Request for Monte Carlo financial simulation."""
    org_id: str = Field(..., description="Organization ID")
    base_ebitda_usd: float = Field(75000000, ge=0, description="Base EBITDA")
    scope1_tco2e: float = Field(25000, ge=0, description="Annual Scope 1 emissions")
    scope2_tco2e: float = Field(15000, ge=0, description="Annual Scope 2 emissions")
    carbon_price_mean: float = Field(100, ge=0, description="Mean carbon price (USD/tCO2e)")
    carbon_price_std: float = Field(30, ge=0, description="Carbon price std deviation")
    revenue_growth_mean_pct: float = Field(3.0, description="Mean revenue growth (%)")
    revenue_growth_std_pct: float = Field(5.0, ge=0, description="Revenue growth std deviation (%)")
    iterations: int = Field(1000, ge=100, le=10000, description="Number of iterations")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class IncomeStatementImpactResponse(BaseModel):
    """Income statement impact from climate risks/opportunities."""
    org_id: str
    scenario_id: str
    revenue_impact_usd: float
    cogs_impact_usd: float
    operating_expense_impact_usd: float
    carbon_cost_usd: float
    depreciation_impact_usd: float
    impairment_charges_usd: float
    net_income_impact_usd: float
    ebitda_impact_usd: float
    ebitda_impact_pct: float
    time_horizon: str
    generated_at: datetime


class BalanceSheetImpactResponse(BaseModel):
    """Balance sheet impact from climate risks/opportunities."""
    org_id: str
    scenario_id: str
    asset_revaluation_usd: float
    stranded_asset_writedown_usd: float
    new_capex_usd: float
    provision_for_climate_liabilities_usd: float
    intangible_impact_usd: float
    net_asset_impact_usd: float
    equity_impact_usd: float
    generated_at: datetime


class CashFlowImpactResponse(BaseModel):
    """Cash flow impact from climate risks/opportunities."""
    org_id: str
    scenario_id: str
    operating_cash_flow_impact_usd: float
    capex_requirement_usd: float
    carbon_cost_cash_impact_usd: float
    insurance_cost_change_usd: float
    green_revenue_usd: float
    cost_savings_usd: float
    net_cash_flow_impact_usd: float
    free_cash_flow_impact_usd: float
    generated_at: datetime


class TotalFinancialImpactResponse(BaseModel):
    """Total financial impact summary."""
    org_id: str
    scenario_id: str
    scenario_name: str
    income_statement: Dict[str, float]
    balance_sheet: Dict[str, float]
    cash_flow: Dict[str, float]
    net_financial_impact_usd: float
    climate_var_usd: float
    key_drivers: List[str]
    generated_at: datetime


class NPVResponse(BaseModel):
    """NPV analysis result."""
    org_id: str
    npv_usd: float
    profitability_index: float
    payback_period_years: Optional[float]
    discounted_payback_years: Optional[float]
    cash_flow_timeline: Dict[str, float]
    recommendation: str
    generated_at: datetime


class IRRResponse(BaseModel):
    """IRR analysis result."""
    org_id: str
    irr_pct: float
    npv_at_wacc_usd: float
    exceeds_hurdle_rate: bool
    hurdle_rate_pct: float
    recommendation: str
    generated_at: datetime


class MACCResponse(BaseModel):
    """MACC data for the organization."""
    org_id: str
    measures: List[Dict[str, Any]]
    total_abatement_tco2e: float
    total_cost_usd: float
    average_cost_per_tco2e: float
    negative_cost_measures: int
    generated_at: datetime


class CarbonSensitivityResponse(BaseModel):
    """Carbon price sensitivity analysis result."""
    org_id: str
    data_points: List[Dict[str, float]]
    breakeven_carbon_price_usd: float
    ebitda_elimination_price_usd: float
    generated_at: datetime


class MonteCarloFinancialResponse(BaseModel):
    """Monte Carlo financial simulation result."""
    simulation_id: str
    org_id: str
    iterations: int
    ebitda_mean_usd: float
    ebitda_median_usd: float
    ebitda_p5_usd: float
    ebitda_p95_usd: float
    probability_of_loss_pct: float
    climate_var_95_usd: float
    generated_at: datetime


class StrandedAssetResponse(BaseModel):
    """Stranded asset valuation under a scenario."""
    org_id: str
    scenario_id: str
    total_assets_assessed_usd: float
    stranded_value_usd: float
    stranding_pct: float
    by_asset_type: Dict[str, float]
    impairment_timeline: Dict[str, float]
    generated_at: datetime


class AdaptationCostResponse(BaseModel):
    """Adaptation cost estimate."""
    org_id: str
    total_adaptation_cost_usd: float
    by_category: Dict[str, float]
    by_time_horizon: Dict[str, float]
    cost_as_pct_of_revenue: float
    benefit_cost_ratio: float
    priority_investments: List[Dict[str, Any]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _npv_calc(rate: float, cash_flows: List[float]) -> float:
    """Calculate NPV given a rate and series of cash flows."""
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows, 1))


def _irr_calc(initial: float, cash_flows: List[float]) -> float:
    """Approximate IRR using bisection method."""
    all_flows = [-initial] + cash_flows
    low, high = -0.5, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        npv = sum(cf / (1 + mid) ** i for i, cf in enumerate(all_flows))
        if abs(npv) < 1:
            return mid
        if npv > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/income/{org_id}/{scenario_id}",
    response_model=IncomeStatementImpactResponse,
    summary="Income statement impacts",
    description="Model the impact of climate risks/opportunities on the income statement under a scenario.",
)
async def get_income_impact(
    org_id: str,
    scenario_id: str,
    time_horizon: str = Query("2030", description="Time horizon: 2030, 2040, 2050"),
) -> IncomeStatementImpactResponse:
    """Model income statement impacts."""
    multiplier = {"2030": 1.0, "2040": 1.8, "2050": 2.5}.get(time_horizon, 1.0)
    revenue_impact = round(-5000000 * multiplier, 2)
    cogs_impact = round(-3000000 * multiplier, 2)
    opex_impact = round(-2000000 * multiplier, 2)
    carbon_cost = round(-4000000 * multiplier, 2)
    depreciation = round(-1500000 * multiplier, 2)
    impairment = round(-8000000 * multiplier, 2)
    net_income = round(revenue_impact + cogs_impact + opex_impact + carbon_cost + depreciation + impairment, 2)
    ebitda = round(revenue_impact + cogs_impact + opex_impact + carbon_cost, 2)

    return IncomeStatementImpactResponse(
        org_id=org_id,
        scenario_id=scenario_id,
        revenue_impact_usd=revenue_impact,
        cogs_impact_usd=cogs_impact,
        operating_expense_impact_usd=opex_impact,
        carbon_cost_usd=carbon_cost,
        depreciation_impact_usd=depreciation,
        impairment_charges_usd=impairment,
        net_income_impact_usd=net_income,
        ebitda_impact_usd=ebitda,
        ebitda_impact_pct=round(ebitda / 75000000 * 100, 2),
        time_horizon=time_horizon,
        generated_at=_now(),
    )


@router.get(
    "/balance-sheet/{org_id}/{scenario_id}",
    response_model=BalanceSheetImpactResponse,
    summary="Balance sheet impacts",
    description="Model the impact of climate risks/opportunities on the balance sheet.",
)
async def get_balance_sheet_impact(org_id: str, scenario_id: str) -> BalanceSheetImpactResponse:
    """Model balance sheet impacts."""
    return BalanceSheetImpactResponse(
        org_id=org_id,
        scenario_id=scenario_id,
        asset_revaluation_usd=-25000000,
        stranded_asset_writedown_usd=-45000000,
        new_capex_usd=35000000,
        provision_for_climate_liabilities_usd=-15000000,
        intangible_impact_usd=-5000000,
        net_asset_impact_usd=-55000000,
        equity_impact_usd=-55000000,
        generated_at=_now(),
    )


@router.get(
    "/cash-flow/{org_id}/{scenario_id}",
    response_model=CashFlowImpactResponse,
    summary="Cash flow impacts",
    description="Model the impact of climate risks/opportunities on cash flows.",
)
async def get_cash_flow_impact(org_id: str, scenario_id: str) -> CashFlowImpactResponse:
    """Model cash flow impacts."""
    opex_cf = -8000000
    capex = -35000000
    carbon = -4000000
    insurance = -2000000
    green_rev = 5000000
    savings = 3000000
    net_cf = round(opex_cf + carbon + insurance + green_rev + savings, 2)
    fcf = round(net_cf + capex, 2)

    return CashFlowImpactResponse(
        org_id=org_id,
        scenario_id=scenario_id,
        operating_cash_flow_impact_usd=opex_cf,
        capex_requirement_usd=capex,
        carbon_cost_cash_impact_usd=carbon,
        insurance_cost_change_usd=insurance,
        green_revenue_usd=green_rev,
        cost_savings_usd=savings,
        net_cash_flow_impact_usd=net_cf,
        free_cash_flow_impact_usd=fcf,
        generated_at=_now(),
    )


@router.get(
    "/total/{org_id}/{scenario_id}",
    response_model=TotalFinancialImpactResponse,
    summary="Total financial impact summary",
    description="Comprehensive financial impact summary across income statement, balance sheet, and cash flow.",
)
async def get_total_financial_impact(org_id: str, scenario_id: str) -> TotalFinancialImpactResponse:
    """Get total financial impact summary."""
    income = {
        "revenue_impact": -5000000,
        "cost_impact": -9000000,
        "net_income_impact": -23500000,
        "ebitda_impact": -14000000,
    }
    balance = {
        "net_asset_impact": -55000000,
        "stranded_assets": -45000000,
        "new_investment": 35000000,
    }
    cash = {
        "net_cash_flow": -6000000,
        "free_cash_flow": -41000000,
        "capex_required": -35000000,
    }
    net_impact = round(income["net_income_impact"] + balance["net_asset_impact"], 2)

    return TotalFinancialImpactResponse(
        org_id=org_id,
        scenario_id=scenario_id,
        scenario_name="IEA Net Zero 2050",
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash,
        net_financial_impact_usd=net_impact,
        climate_var_usd=round(abs(net_impact) * 1.5, 2),
        key_drivers=[
            "Carbon pricing is the largest income statement driver",
            "Asset stranding represents the largest balance sheet risk",
            "Transition capex dominates cash flow requirements",
            "Green revenue partially offsets operating cost increases",
        ],
        generated_at=_now(),
    )


@router.post(
    "/npv",
    response_model=NPVResponse,
    summary="NPV analysis",
    description="Calculate net present value for a climate investment with discounted cash flows.",
)
async def npv_analysis(request: NPVRequest) -> NPVResponse:
    """Calculate NPV."""
    rate = request.discount_rate_pct / 100
    pv_cfs = _npv_calc(rate, request.annual_cash_flows)
    terminal_pv = request.terminal_value_usd / (1 + rate) ** len(request.annual_cash_flows)
    npv = round(pv_cfs + terminal_pv - request.initial_investment_usd, 2)
    pi = round((pv_cfs + terminal_pv) / request.initial_investment_usd, 3) if request.initial_investment_usd > 0 else 0

    cumulative = 0.0
    payback = None
    for i, cf in enumerate(request.annual_cash_flows, 1):
        cumulative += cf
        if cumulative >= request.initial_investment_usd and payback is None:
            payback = round(i - (cumulative - request.initial_investment_usd) / cf, 2)

    cum_disc = 0.0
    disc_payback = None
    for i, cf in enumerate(request.annual_cash_flows, 1):
        cum_disc += cf / (1 + rate) ** i
        if cum_disc >= request.initial_investment_usd and disc_payback is None:
            disc_payback = round(i - (cum_disc - request.initial_investment_usd) / (cf / (1 + rate) ** i), 2)

    timeline = {}
    for i, cf in enumerate(request.annual_cash_flows):
        timeline[str(2025 + i + 1)] = round(cf, 2)

    rec = "Investment is value-creating (NPV > 0)" if npv > 0 else "Investment destroys value (NPV < 0); reconsider or restructure"

    return NPVResponse(
        org_id=request.org_id,
        npv_usd=npv,
        profitability_index=pi,
        payback_period_years=payback,
        discounted_payback_years=disc_payback,
        cash_flow_timeline=timeline,
        recommendation=rec,
        generated_at=_now(),
    )


@router.post(
    "/irr",
    response_model=IRRResponse,
    summary="IRR analysis",
    description="Calculate internal rate of return for a climate investment.",
)
async def irr_analysis(request: IRRRequest) -> IRRResponse:
    """Calculate IRR."""
    irr = _irr_calc(request.initial_investment_usd, request.annual_cash_flows)
    irr_pct = round(irr * 100, 2)
    hurdle = 8.0
    npv_at_wacc = round(
        _npv_calc(hurdle / 100, request.annual_cash_flows) - request.initial_investment_usd, 2
    )

    return IRRResponse(
        org_id=request.org_id,
        irr_pct=irr_pct,
        npv_at_wacc_usd=npv_at_wacc,
        exceeds_hurdle_rate=irr_pct > hurdle,
        hurdle_rate_pct=hurdle,
        recommendation="Accept: IRR exceeds hurdle rate" if irr_pct > hurdle else "Reject: IRR below hurdle rate",
        generated_at=_now(),
    )


@router.get(
    "/macc/{org_id}",
    response_model=MACCResponse,
    summary="MACC data",
    description="Generate marginal abatement cost curve data for the organization.",
)
async def get_macc(org_id: str) -> MACCResponse:
    """Generate MACC data."""
    measures = [
        {"measure": "LED lighting", "abatement_tco2e": 120, "cost_per_tco2e": -45, "total_cost": -5400},
        {"measure": "HVAC optimization", "abatement_tco2e": 350, "cost_per_tco2e": -20, "total_cost": -7000},
        {"measure": "Building insulation", "abatement_tco2e": 200, "cost_per_tco2e": -5, "total_cost": -1000},
        {"measure": "Solar PV", "abatement_tco2e": 800, "cost_per_tco2e": 15, "total_cost": 12000},
        {"measure": "Fleet electrification", "abatement_tco2e": 1500, "cost_per_tco2e": 35, "total_cost": 52500},
        {"measure": "Green hydrogen", "abatement_tco2e": 2000, "cost_per_tco2e": 80, "total_cost": 160000},
        {"measure": "CCS retrofit", "abatement_tco2e": 3000, "cost_per_tco2e": 120, "total_cost": 360000},
        {"measure": "Renewable PPA", "abatement_tco2e": 4000, "cost_per_tco2e": 10, "total_cost": 40000},
    ]
    measures.sort(key=lambda m: m["cost_per_tco2e"])
    total_abatement = sum(m["abatement_tco2e"] for m in measures)
    total_cost = sum(m["total_cost"] for m in measures)
    avg_cost = round(total_cost / total_abatement, 2) if total_abatement > 0 else 0
    neg_count = sum(1 for m in measures if m["cost_per_tco2e"] < 0)

    return MACCResponse(
        org_id=org_id,
        measures=measures,
        total_abatement_tco2e=total_abatement,
        total_cost_usd=total_cost,
        average_cost_per_tco2e=avg_cost,
        negative_cost_measures=neg_count,
        generated_at=_now(),
    )


@router.post(
    "/carbon-sensitivity",
    response_model=CarbonSensitivityResponse,
    summary="Carbon price sensitivity",
    description="Analyze the sensitivity of EBITDA to different carbon price levels.",
)
async def carbon_sensitivity(request: CarbonSensitivityRequest) -> CarbonSensitivityResponse:
    """Analyze carbon price sensitivity."""
    total_emissions = request.scope1_tco2e + request.scope2_tco2e
    ebitda = request.annual_revenue_usd * request.ebitda_margin_pct / 100
    data_points = []
    elimination_price = 0.0

    for price in request.carbon_prices:
        carbon_cost = total_emissions * price
        adjusted_ebitda = round(ebitda - carbon_cost, 2)
        margin_pct = round(adjusted_ebitda / request.annual_revenue_usd * 100, 2) if request.annual_revenue_usd > 0 else 0
        data_points.append({
            "carbon_price_usd": price,
            "carbon_cost_usd": round(carbon_cost, 2),
            "adjusted_ebitda_usd": adjusted_ebitda,
            "adjusted_margin_pct": margin_pct,
        })
        if adjusted_ebitda <= 0 and elimination_price == 0:
            elimination_price = price

    breakeven_price = round(ebitda / total_emissions, 2) if total_emissions > 0 else 0

    return CarbonSensitivityResponse(
        org_id=request.org_id,
        data_points=data_points,
        breakeven_carbon_price_usd=breakeven_price,
        ebitda_elimination_price_usd=elimination_price if elimination_price > 0 else breakeven_price,
        generated_at=_now(),
    )


@router.post(
    "/monte-carlo",
    response_model=MonteCarloFinancialResponse,
    summary="Monte Carlo financial simulation",
    description="Run Monte Carlo simulation modeling uncertainty in carbon prices and revenue growth on EBITDA.",
)
async def monte_carlo_financial(request: MonteCarloFinancialRequest) -> MonteCarloFinancialResponse:
    """Run Monte Carlo financial simulation."""
    random.seed(42)
    total_emissions = request.scope1_tco2e + request.scope2_tco2e
    results = []
    for _ in range(request.iterations):
        carbon_price = max(0, random.gauss(request.carbon_price_mean, request.carbon_price_std))
        growth = random.gauss(request.revenue_growth_mean_pct, request.revenue_growth_std_pct) / 100
        adjusted_ebitda = request.base_ebitda_usd * (1 + growth) - total_emissions * carbon_price
        results.append(adjusted_ebitda)

    results.sort()
    n = len(results)
    mean_val = sum(results) / n
    median_val = results[n // 2]
    p5 = results[int(n * 0.05)]
    p95 = results[int(n * 0.95)]
    loss_count = sum(1 for r in results if r < 0)
    loss_pct = round(loss_count / n * 100, 2)
    var_95 = round(request.base_ebitda_usd - p5, 2)

    return MonteCarloFinancialResponse(
        simulation_id=_generate_id("mcf"),
        org_id=request.org_id,
        iterations=request.iterations,
        ebitda_mean_usd=round(mean_val, 2),
        ebitda_median_usd=round(median_val, 2),
        ebitda_p5_usd=round(p5, 2),
        ebitda_p95_usd=round(p95, 2),
        probability_of_loss_pct=loss_pct,
        climate_var_95_usd=var_95,
        generated_at=_now(),
    )


@router.get(
    "/stranded-assets/{org_id}/{scenario_id}",
    response_model=StrandedAssetResponse,
    summary="Stranded asset value",
    description="Calculate the value of stranded assets under a specific climate scenario.",
)
async def get_stranded_assets(org_id: str, scenario_id: str) -> StrandedAssetResponse:
    """Calculate stranded asset value."""
    total = 500000000
    by_type = {
        "fossil_infrastructure": round(total * 0.25 * 0.6, 2),
        "carbon_equipment": round(total * 0.20 * 0.3, 2),
        "vehicle_fleet": round(total * 0.08 * 0.5, 2),
        "real_estate": round(total * 0.15 * 0.1, 2),
    }
    stranded = round(sum(by_type.values()), 2)
    pct = round(stranded / total * 100, 2)

    timeline = {
        "2030": round(stranded * 0.15, 2),
        "2035": round(stranded * 0.35, 2),
        "2040": round(stranded * 0.60, 2),
        "2050": round(stranded, 2),
    }

    return StrandedAssetResponse(
        org_id=org_id,
        scenario_id=scenario_id,
        total_assets_assessed_usd=total,
        stranded_value_usd=stranded,
        stranding_pct=pct,
        by_asset_type=by_type,
        impairment_timeline=timeline,
        generated_at=_now(),
    )


@router.get(
    "/adaptation-cost/{org_id}",
    response_model=AdaptationCostResponse,
    summary="Adaptation cost estimate",
    description="Estimate the total cost of climate adaptation measures.",
)
async def get_adaptation_cost(
    org_id: str,
    annual_revenue_usd: float = Query(500000000, ge=0, description="Annual revenue"),
) -> AdaptationCostResponse:
    """Estimate adaptation costs."""
    by_category = {
        "physical_resilience": 8000000,
        "supply_chain_diversification": 5000000,
        "technology_transition": 15000000,
        "operational_efficiency": 4000000,
        "regulatory_compliance": 3000000,
    }
    total = sum(by_category.values())
    by_horizon = {
        "2025-2030": round(total * 0.40, 2),
        "2030-2040": round(total * 0.35, 2),
        "2040-2050": round(total * 0.25, 2),
    }
    cost_pct = round(total / annual_revenue_usd * 100, 2) if annual_revenue_usd > 0 else 0
    avoided_loss = round(total * 2.5, 2)
    bcr = round(avoided_loss / total, 2) if total > 0 else 0

    priorities = [
        {"investment": "Flood defense for coastal facilities", "cost_usd": 3000000, "avoided_loss_usd": 12000000, "bcr": 4.0},
        {"investment": "Dual-source critical materials", "cost_usd": 2000000, "avoided_loss_usd": 8000000, "bcr": 4.0},
        {"investment": "On-site renewable generation", "cost_usd": 8000000, "avoided_loss_usd": 15000000, "bcr": 1.9},
        {"investment": "Heat stress mitigation for workforce", "cost_usd": 1000000, "avoided_loss_usd": 3000000, "bcr": 3.0},
        {"investment": "Climate risk monitoring platform", "cost_usd": 500000, "avoided_loss_usd": 2000000, "bcr": 4.0},
    ]

    return AdaptationCostResponse(
        org_id=org_id,
        total_adaptation_cost_usd=total,
        by_category=by_category,
        by_time_horizon=by_horizon,
        cost_as_pct_of_revenue=cost_pct,
        benefit_cost_ratio=bcr,
        priority_investments=priorities,
        generated_at=_now(),
    )
