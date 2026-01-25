"""
GL-082: ROI Calculator Agent (ROICALCULATOR)

This module implements the ROICalculatorAgent for comprehensive return on
investment analysis for energy efficiency and sustainability projects.

The agent provides:
- NPV (Net Present Value) calculation
- IRR (Internal Rate of Return)
- MIRR (Modified IRR)
- Simple and discounted payback period
- ROI percentage calculation
- Sensitivity analysis
- Complete SHA-256 provenance tracking

Example:
    >>> agent = ROICalculatorAgent()
    >>> result = agent.run(ROICalculatorInput(...))
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class InvestmentType(str, Enum):
    """Types of energy investments."""
    EFFICIENCY = "EFFICIENCY"
    RENEWABLE = "RENEWABLE"
    STORAGE = "STORAGE"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    CONTROLS = "CONTROLS"
    ELECTRIFICATION = "ELECTRIFICATION"


class CashFlowType(str, Enum):
    """Types of cash flows."""
    ENERGY_SAVINGS = "ENERGY_SAVINGS"
    MAINTENANCE_SAVINGS = "MAINTENANCE_SAVINGS"
    INCENTIVE = "INCENTIVE"
    TAX_BENEFIT = "TAX_BENEFIT"
    REVENUE = "REVENUE"
    OPERATING_COST = "OPERATING_COST"
    CAPITAL_COST = "CAPITAL_COST"


# =============================================================================
# INPUT MODELS
# =============================================================================

class InvestmentCost(BaseModel):
    """Investment cost details."""
    description: str = Field(...)
    amount_usd: float = Field(..., ge=0)
    year: int = Field(default=0)
    is_recurring: bool = Field(default=False)


class CashFlow(BaseModel):
    """Cash flow details."""
    year: int = Field(...)
    amount_usd: float = Field(...)
    flow_type: CashFlowType = Field(...)
    description: Optional[str] = None
    is_recurring: bool = Field(default=True)
    escalation_rate: float = Field(default=0)


class ROICalculatorInput(BaseModel):
    """Complete input model for ROI Calculator."""
    project_name: str = Field(...)
    investment_type: InvestmentType = Field(...)

    initial_investment_usd: float = Field(..., ge=0)
    additional_investments: List[InvestmentCost] = Field(default_factory=list)
    cash_flows: List[CashFlow] = Field(...)

    analysis_period_years: int = Field(default=10, ge=1, le=30)
    discount_rate_percent: float = Field(default=8.0, ge=0, le=30)
    reinvestment_rate_percent: float = Field(default=6.0, ge=0, le=20)
    tax_rate_percent: float = Field(default=25.0, ge=0, le=50)

    include_sensitivity: bool = Field(default=True)
    sensitivity_range_percent: float = Field(default=20, ge=5, le=50)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ROIMetrics(BaseModel):
    """Core ROI metrics."""
    npv_usd: float
    irr_percent: Optional[float]
    mirr_percent: Optional[float]
    simple_payback_years: Optional[float]
    discounted_payback_years: Optional[float]
    roi_percent: float
    profitability_index: float
    total_return_usd: float


class YearlyCashFlow(BaseModel):
    """Yearly cash flow summary."""
    year: int
    inflows_usd: float
    outflows_usd: float
    net_cash_flow_usd: float
    cumulative_cash_flow_usd: float
    discounted_cash_flow_usd: float


class SensitivityResult(BaseModel):
    """Sensitivity analysis result."""
    parameter: str
    base_npv: float
    low_npv: float
    high_npv: float
    npv_range: float
    sensitivity_index: float


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ROICalculatorOutput(BaseModel):
    """Complete output model for ROI Calculator."""
    analysis_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    project_name: str

    # Core metrics
    roi_metrics: ROIMetrics
    investment_decision: str  # ACCEPT, REJECT, MARGINAL

    # Cash flow analysis
    yearly_cash_flows: List[YearlyCashFlow]
    total_investment_usd: float
    total_returns_usd: float

    # Sensitivity
    sensitivity_analysis: List[SensitivityResult]

    # Recommendations
    recommendations: List[str]

    # Provenance
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str

    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# ROI CALCULATOR AGENT
# =============================================================================

class ROICalculatorAgent:
    """GL-082: ROI Calculator Agent."""

    AGENT_ID = "GL-082"
    AGENT_NAME = "ROICALCULATOR"
    VERSION = "1.0.0"
    DESCRIPTION = "Return on Investment Calculator Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ROICalculatorAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []

        logger.info(f"ROICalculatorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: ROICalculatorInput) -> ROICalculatorOutput:
        """Execute ROI analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting ROI analysis for {input_data.project_name}")

        try:
            # Build cash flow timeline
            yearly_flows, net_flows = self._build_cash_flows(input_data)

            self._track_provenance(
                "cash_flow_construction",
                {"years": input_data.analysis_period_years},
                {"total_flows": len(net_flows)},
                "Cash Flow Builder"
            )

            # Calculate ROI metrics
            metrics = self._calculate_metrics(
                input_data.initial_investment_usd,
                net_flows,
                input_data.discount_rate_percent,
                input_data.reinvestment_rate_percent,
            )

            self._track_provenance(
                "roi_calculation",
                {"discount_rate": input_data.discount_rate_percent},
                {"npv": metrics.npv_usd, "irr": metrics.irr_percent},
                "ROI Engine"
            )

            # Investment decision
            decision = self._make_decision(metrics, input_data.discount_rate_percent)

            # Sensitivity analysis
            sensitivity = []
            if input_data.include_sensitivity:
                sensitivity = self._run_sensitivity(
                    input_data, net_flows, input_data.sensitivity_range_percent
                )

            # Calculate totals
            total_investment = input_data.initial_investment_usd + sum(
                i.amount_usd for i in input_data.additional_investments
            )
            total_returns = sum(cf.net_cash_flow_usd for cf in yearly_flows if cf.net_cash_flow_usd > 0)

            # Recommendations
            recommendations = self._generate_recommendations(metrics, decision)

            # Calculate provenance
            provenance_hash = self._calculate_provenance_hash()
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            analysis_id = (
                f"ROI-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.project_name.encode()).hexdigest()[:8]}"
            )

            return ROICalculatorOutput(
                analysis_id=analysis_id,
                project_name=input_data.project_name,
                roi_metrics=metrics,
                investment_decision=decision,
                yearly_cash_flows=yearly_flows,
                total_investment_usd=round(total_investment, 2),
                total_returns_usd=round(total_returns, 2),
                sensitivity_analysis=sensitivity,
                recommendations=recommendations,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=s["operation"],
                        timestamp=s["timestamp"],
                        input_hash=s["input_hash"],
                        output_hash=s["output_hash"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {}),
                    )
                    for s in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS" if not self._validation_errors else "FAIL",
                validation_errors=self._validation_errors,
            )

        except Exception as e:
            logger.error(f"ROI analysis failed: {str(e)}", exc_info=True)
            raise

    def _build_cash_flows(self, input_data: ROICalculatorInput):
        """Build yearly cash flow timeline."""
        yearly_flows = []
        net_flows = [-input_data.initial_investment_usd]  # Year 0

        cumulative = -input_data.initial_investment_usd
        discount_rate = input_data.discount_rate_percent / 100

        # Year 0
        yearly_flows.append(YearlyCashFlow(
            year=0,
            inflows_usd=0,
            outflows_usd=input_data.initial_investment_usd,
            net_cash_flow_usd=-input_data.initial_investment_usd,
            cumulative_cash_flow_usd=cumulative,
            discounted_cash_flow_usd=-input_data.initial_investment_usd,
        ))

        # Years 1-N
        for year in range(1, input_data.analysis_period_years + 1):
            inflows = 0
            outflows = 0

            for cf in input_data.cash_flows:
                if cf.year == year or (cf.is_recurring and cf.year <= year):
                    escalation = (1 + cf.escalation_rate/100) ** (year - max(cf.year, 1))
                    amount = cf.amount_usd * escalation

                    if cf.amount_usd > 0:
                        inflows += amount
                    else:
                        outflows += abs(amount)

            # Additional investments
            for inv in input_data.additional_investments:
                if inv.year == year:
                    outflows += inv.amount_usd

            net = inflows - outflows
            cumulative += net
            discounted = net / ((1 + discount_rate) ** year)

            net_flows.append(net)

            yearly_flows.append(YearlyCashFlow(
                year=year,
                inflows_usd=round(inflows, 2),
                outflows_usd=round(outflows, 2),
                net_cash_flow_usd=round(net, 2),
                cumulative_cash_flow_usd=round(cumulative, 2),
                discounted_cash_flow_usd=round(discounted, 2),
            ))

        return yearly_flows, net_flows

    def _calculate_metrics(
        self, initial_investment: float, cash_flows: List[float],
        discount_rate: float, reinvestment_rate: float
    ) -> ROIMetrics:
        """Calculate ROI metrics."""
        # NPV
        npv = self._calculate_npv(cash_flows, discount_rate)

        # IRR
        irr = self._calculate_irr(cash_flows)

        # MIRR
        mirr = self._calculate_mirr(cash_flows, discount_rate, reinvestment_rate)

        # Simple Payback
        simple_payback = self._calculate_simple_payback(cash_flows)

        # Discounted Payback
        discounted_payback = self._calculate_discounted_payback(cash_flows, discount_rate)

        # ROI
        total_return = sum(cf for cf in cash_flows[1:] if cf > 0)
        roi = ((total_return - initial_investment) / initial_investment * 100) if initial_investment > 0 else 0

        # Profitability Index
        pv_inflows = sum(
            cf / ((1 + discount_rate/100) ** i)
            for i, cf in enumerate(cash_flows[1:], 1) if cf > 0
        )
        pi = pv_inflows / initial_investment if initial_investment > 0 else 0

        return ROIMetrics(
            npv_usd=round(npv, 2),
            irr_percent=round(irr, 2) if irr else None,
            mirr_percent=round(mirr, 2) if mirr else None,
            simple_payback_years=round(simple_payback, 2) if simple_payback else None,
            discounted_payback_years=round(discounted_payback, 2) if discounted_payback else None,
            roi_percent=round(roi, 2),
            profitability_index=round(pi, 2),
            total_return_usd=round(total_return, 2),
        )

    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = sum(cf / ((1 + discount_rate/100) ** i) for i, cf in enumerate(cash_flows))
        return npv

    def _calculate_irr(self, cash_flows: List[float], max_iter: int = 100, tol: float = 0.0001) -> Optional[float]:
        """Calculate Internal Rate of Return using Newton-Raphson."""
        rate = 0.10  # Initial guess

        for _ in range(max_iter):
            npv = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
            npv_derivative = sum(
                -i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows)
            )

            if abs(npv_derivative) < 1e-10:
                return None

            new_rate = rate - npv / npv_derivative

            if abs(new_rate - rate) < tol:
                return new_rate * 100

            rate = new_rate

        return None

    def _calculate_mirr(
        self, cash_flows: List[float], finance_rate: float, reinvest_rate: float
    ) -> Optional[float]:
        """Calculate Modified IRR."""
        n = len(cash_flows) - 1
        if n <= 0:
            return None

        # PV of negative cash flows
        pv_negatives = sum(
            cf / ((1 + finance_rate/100) ** i)
            for i, cf in enumerate(cash_flows) if cf < 0
        )

        # FV of positive cash flows
        fv_positives = sum(
            cf * ((1 + reinvest_rate/100) ** (n - i))
            for i, cf in enumerate(cash_flows) if cf > 0
        )

        if pv_negatives >= 0 or fv_positives <= 0:
            return None

        mirr = ((fv_positives / abs(pv_negatives)) ** (1/n) - 1) * 100
        return mirr

    def _calculate_simple_payback(self, cash_flows: List[float]) -> Optional[float]:
        """Calculate simple payback period."""
        cumulative = 0
        for i, cf in enumerate(cash_flows):
            cumulative += cf
            if cumulative >= 0 and i > 0:
                # Interpolate
                prev_cumulative = cumulative - cf
                if cf != 0:
                    fraction = abs(prev_cumulative) / cf
                    return i - 1 + fraction
                return float(i)
        return None

    def _calculate_discounted_payback(self, cash_flows: List[float], discount_rate: float) -> Optional[float]:
        """Calculate discounted payback period."""
        cumulative = 0
        for i, cf in enumerate(cash_flows):
            discounted = cf / ((1 + discount_rate/100) ** i)
            cumulative += discounted
            if cumulative >= 0 and i > 0:
                prev_cumulative = cumulative - discounted
                if discounted != 0:
                    fraction = abs(prev_cumulative) / discounted
                    return i - 1 + fraction
                return float(i)
        return None

    def _make_decision(self, metrics: ROIMetrics, hurdle_rate: float) -> str:
        """Make investment decision."""
        if metrics.npv_usd > 0 and (metrics.irr_percent is None or metrics.irr_percent > hurdle_rate):
            return "ACCEPT"
        elif metrics.npv_usd < -10000 or (metrics.irr_percent and metrics.irr_percent < hurdle_rate * 0.5):
            return "REJECT"
        else:
            return "MARGINAL"

    def _run_sensitivity(
        self, input_data: ROICalculatorInput, base_flows: List[float], range_pct: float
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis."""
        results = []
        base_npv = self._calculate_npv(base_flows, input_data.discount_rate_percent)

        # Sensitivity to discount rate
        low_rate = input_data.discount_rate_percent * (1 - range_pct/100)
        high_rate = input_data.discount_rate_percent * (1 + range_pct/100)
        low_npv = self._calculate_npv(base_flows, low_rate)
        high_npv = self._calculate_npv(base_flows, high_rate)

        results.append(SensitivityResult(
            parameter="Discount Rate",
            base_npv=round(base_npv, 2),
            low_npv=round(low_npv, 2),
            high_npv=round(high_npv, 2),
            npv_range=round(abs(high_npv - low_npv), 2),
            sensitivity_index=round(abs(high_npv - low_npv) / abs(base_npv) if base_npv != 0 else 0, 4),
        ))

        # Sensitivity to cash flows
        low_flows = [base_flows[0]] + [cf * (1 - range_pct/100) for cf in base_flows[1:]]
        high_flows = [base_flows[0]] + [cf * (1 + range_pct/100) for cf in base_flows[1:]]

        low_npv = self._calculate_npv(low_flows, input_data.discount_rate_percent)
        high_npv = self._calculate_npv(high_flows, input_data.discount_rate_percent)

        results.append(SensitivityResult(
            parameter="Cash Flows",
            base_npv=round(base_npv, 2),
            low_npv=round(low_npv, 2),
            high_npv=round(high_npv, 2),
            npv_range=round(abs(high_npv - low_npv), 2),
            sensitivity_index=round(abs(high_npv - low_npv) / abs(base_npv) if base_npv != 0 else 0, 4),
        ))

        return results

    def _generate_recommendations(self, metrics: ROIMetrics, decision: str) -> List[str]:
        """Generate recommendations."""
        recommendations = []

        if decision == "ACCEPT":
            recommendations.append("Project meets investment criteria - proceed with implementation")
            if metrics.simple_payback_years and metrics.simple_payback_years < 3:
                recommendations.append("Fast payback - consider prioritizing this project")
        elif decision == "REJECT":
            recommendations.append("Project does not meet hurdle rate - review assumptions")
        else:
            recommendations.append("Marginal project - conduct further analysis")
            if metrics.irr_percent and metrics.irr_percent > 0:
                recommendations.append("Consider negotiating lower costs or higher incentives")

        return recommendations

    def _track_provenance(
        self, operation: str, inputs: Dict, outputs: Dict, tool_name: str
    ) -> None:
        """Track provenance step."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate provenance chain hash."""
        data = {
            "agent_id": self.AGENT_ID,
            "steps": [
                {"operation": s["operation"], "input_hash": s["input_hash"]}
                for s in self._provenance_steps
            ],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-082",
    "name": "ROICALCULATOR - Return on Investment Calculator Agent",
    "version": "1.0.0",
    "summary": "Comprehensive ROI analysis with NPV, IRR, and sensitivity",
    "tags": ["roi", "npv", "irr", "financial-analysis", "payback"],
    "owners": ["finance-team"],
    "compute": {
        "entrypoint": "python://agents.gl_082_roi_calculator.agent:ROICalculatorAgent",
        "deterministic": True,
    },
    "provenance": {"calculation_verified": True, "enable_audit": True},
}
