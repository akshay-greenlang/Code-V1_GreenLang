# -*- coding: utf-8 -*-
"""
Certificate Trading Workflow
===============================

Six-phase weekly certificate trading cycle workflow for CBAM certificate
portfolio management. Orchestrates price monitoring, obligation forecasting,
purchase decision-making across five configurable strategies, order
execution via registry API, portfolio rebalancing with expiry management,
and comprehensive reporting.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 20: CBAM certificates priced at EU ETS weekly weighted
      average auction price. 1 certificate = 1 tonne CO2e.
    - Article 22(1): Surrender certificates equal to embedded emissions
      by May 31 annually.
    - Article 22(2): Quarterly holding >= 50% of estimated annual obligation.
    - Article 23: Certificates valid for 2 years; NCA may repurchase
      unsurrendered certificates at original purchase price.
    - Article 24: Re-selling up to 1/3 of purchased certificates within
      12 months of purchase to NCA at original price.

Buying Strategies:
    - BUDGET_PACED: Spread purchases evenly across remaining weeks in year.
    - PRICE_TRIGGERED: Buy when price drops below configured threshold.
    - BULK_QUARTERLY: Purchase entire quarterly need in one transaction.
    - DCA: Dollar-cost averaging with fixed weekly purchase amount.
    - OPPORTUNISTIC: Buy on dips (price < 20-day moving average * factor).

Phases:
    1. PriceMonitor - Fetch EU ETS clearing price, compare budget thresholds
    2. ObligationForecast - Calculate upcoming certificate needs
    3. PurchaseDecision - Apply buying strategy to determine quantity/timing
    4. OrderExecution - Submit purchase orders via Registry API
    5. PortfolioRebalance - Expiry alerts, surplus/deficit, re-sell analysis
    6. Reporting - Dashboard update, portfolio report, finance notification

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class BuyingStrategy(str, Enum):
    """Certificate purchase strategy."""
    BUDGET_PACED = "BUDGET_PACED"
    PRICE_TRIGGERED = "PRICE_TRIGGERED"
    BULK_QUARTERLY = "BULK_QUARTERLY"
    DCA = "DCA"
    OPPORTUNISTIC = "OPPORTUNISTIC"


class OrderType(str, Enum):
    """Certificate purchase order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Purchase order execution status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PriceTrend(str, Enum):
    """Weekly price trend direction."""
    RISING = "RISING"
    FALLING = "FALLING"
    STABLE = "STABLE"


# =============================================================================
# CONSTANTS
# =============================================================================

PENALTY_EUR_PER_TCO2E = Decimal("100.00")
QUARTERLY_HOLDING_PCT = Decimal("0.50")
CERTIFICATE_VALIDITY_YEARS = 2
MAX_RESELL_FRACTION = Decimal("0.3333")
RESELL_WINDOW_MONTHS = 12
WEEKS_PER_YEAR = 52
DEFAULT_PRICE_THRESHOLD_EUR = Decimal("85.00")
DEFAULT_DCA_WEEKLY_AMOUNT_EUR = Decimal("50000.00")
DEFAULT_OPPORTUNISTIC_FACTOR = Decimal("0.95")
MOVING_AVERAGE_WINDOW = 20


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """
    Shared state passed between workflow phases.

    Carries accumulated data, configuration, and intermediate results
    so each phase can build on the outputs of previous phases.
    """
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed (for resume)."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - CERTIFICATE TRADING
# =============================================================================


class PriceAlert(BaseModel):
    """Price alert configuration."""
    alert_type: str = Field(..., description="above_threshold or below_threshold")
    threshold_eur: float = Field(..., ge=0)
    notification_channels: List[str] = Field(default_factory=lambda: ["email"])


class CertificateHolding(BaseModel):
    """A batch of certificates held in portfolio."""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    purchase_date: str = Field(..., description="ISO date YYYY-MM-DD")
    quantity: float = Field(..., ge=0, description="Number of certificates")
    unit_price_eur: float = Field(..., ge=0, description="Purchase price per cert")
    total_cost_eur: float = Field(default=0.0, ge=0)
    expiry_date: str = Field(default="", description="2 years from purchase")
    resell_eligible: bool = Field(default=True)
    resell_deadline: str = Field(default="", description="12 months from purchase")


class ImportShipment(BaseModel):
    """Upcoming import shipment for obligation forecasting."""
    shipment_id: str = Field(...)
    cn_code: str = Field(..., description="Combined Nomenclature code")
    goods_category: str = Field(default="")
    quantity_tonnes: float = Field(..., ge=0)
    country_of_origin: str = Field(...)
    estimated_embedded_emissions_tco2e: float = Field(default=0.0, ge=0)
    expected_arrival_date: str = Field(default="")
    status: str = Field(default="confirmed", description="confirmed, planned, tentative")


class PurchaseOrder(BaseModel):
    """Certificate purchase order."""
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_type: OrderType = Field(default=OrderType.MARKET)
    quantity: float = Field(..., ge=0, description="Certificates to purchase")
    limit_price_eur: Optional[float] = Field(None, ge=0)
    estimated_cost_eur: float = Field(default=0.0, ge=0)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    submitted_at: Optional[datetime] = Field(None)
    filled_at: Optional[datetime] = Field(None)
    fill_price_eur: Optional[float] = Field(None, ge=0)
    receipt_id: Optional[str] = Field(None)


class CertificateTradingInput(BaseModel):
    """Input configuration for the certificate trading workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2026, le=2050)
    current_week: int = Field(..., ge=1, le=53, description="ISO week number")
    buying_strategy: BuyingStrategy = Field(default=BuyingStrategy.BUDGET_PACED)
    annual_budget_eur: float = Field(default=0.0, ge=0, description="Annual cert budget")
    budget_spent_ytd_eur: float = Field(default=0.0, ge=0, description="Budget spent YTD")
    current_ets_price_eur: float = Field(..., ge=0, description="Current EU ETS price")
    price_history: List[float] = Field(
        default_factory=list,
        description="Recent weekly ETS prices for trend analysis"
    )
    price_alerts: List[PriceAlert] = Field(default_factory=list)
    current_holdings: List[CertificateHolding] = Field(default_factory=list)
    confirmed_shipments: List[ImportShipment] = Field(default_factory=list)
    planned_shipments: List[ImportShipment] = Field(default_factory=list)
    annual_obligation_estimate_tco2e: float = Field(default=0.0, ge=0)
    certificates_surrendered_ytd: float = Field(default=0.0, ge=0)
    price_trigger_threshold_eur: Optional[float] = Field(None, ge=0)
    dca_weekly_amount_eur: Optional[float] = Field(None, ge=0)
    opportunistic_factor: Optional[float] = Field(None, ge=0)
    auto_execute_orders: bool = Field(default=False)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Validate reporting year is in CBAM definitive period."""
        if v < 2026:
            raise ValueError("CBAM definitive period starts 2026")
        return v


class CertificateTradingResult(WorkflowResult):
    """Complete result from the certificate trading workflow."""
    current_price_eur: float = Field(default=0.0)
    price_trend: str = Field(default="STABLE")
    total_obligation_tco2e: float = Field(default=0.0)
    certificates_held: float = Field(default=0.0)
    certificates_needed: float = Field(default=0.0)
    orders_created: int = Field(default=0)
    orders_executed: int = Field(default=0)
    portfolio_value_eur: float = Field(default=0.0)
    budget_remaining_eur: float = Field(default=0.0)
    resell_opportunities: int = Field(default=0)
    expiring_soon: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class PriceMonitorPhase:
    """
    Phase 1: Price Monitoring.

    Fetches current EU ETS clearing price, compares against configured
    budget thresholds and price alerts, and tracks weekly price trend
    direction using moving average analysis.
    """

    PHASE_NAME = "price_monitor"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute price monitoring phase.

        Args:
            context: Workflow context with configuration and prior outputs.

        Returns:
            PhaseResult with current price, trend, and alert triggers.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            current_price = Decimal(str(config.get("current_ets_price_eur", 0)))
            price_history = config.get("price_history", [])
            price_alerts = config.get("price_alerts", [])
            annual_budget = Decimal(str(config.get("annual_budget_eur", 0)))
            budget_spent = Decimal(str(config.get("budget_spent_ytd_eur", 0)))

            # Compute price trend
            trend = self._compute_price_trend(current_price, price_history)
            outputs["current_price_eur"] = float(current_price)
            outputs["price_trend"] = trend.value

            # Compute moving average
            ma_window = min(MOVING_AVERAGE_WINDOW, len(price_history))
            if ma_window > 0:
                recent_prices = price_history[-ma_window:]
                moving_avg = sum(Decimal(str(p)) for p in recent_prices) / ma_window
                outputs["moving_average_eur"] = float(moving_avg.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                ))
            else:
                outputs["moving_average_eur"] = float(current_price)

            # Check price alerts
            triggered_alerts = self._check_price_alerts(
                current_price, price_alerts
            )
            outputs["triggered_alerts"] = triggered_alerts
            if triggered_alerts:
                warnings.append(
                    f"{len(triggered_alerts)} price alert(s) triggered "
                    f"at EUR {current_price}"
                )

            # Budget analysis
            budget_remaining = annual_budget - budget_spent
            outputs["budget_remaining_eur"] = float(budget_remaining)
            outputs["budget_utilization_pct"] = float(
                (budget_spent / annual_budget * 100).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            ) if annual_budget > 0 else 0.0

            if budget_remaining <= 0:
                warnings.append("Annual certificate budget fully exhausted")
            elif budget_remaining < annual_budget * Decimal("0.1"):
                warnings.append(
                    f"Certificate budget nearly exhausted: "
                    f"EUR {budget_remaining:.2f} remaining"
                )

            # Weekly price change
            if price_history:
                prev_price = Decimal(str(price_history[-1]))
                if prev_price > 0:
                    change_pct = (
                        (current_price - prev_price) / prev_price * 100
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    outputs["weekly_change_pct"] = float(change_pct)
                else:
                    outputs["weekly_change_pct"] = 0.0
            else:
                outputs["weekly_change_pct"] = 0.0

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PriceMonitor failed: %s", exc, exc_info=True)
            errors.append(f"Price monitoring failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _compute_price_trend(
        self, current: Decimal, history: List[float]
    ) -> PriceTrend:
        """Determine price trend from recent history."""
        if len(history) < 3:
            return PriceTrend.STABLE
        recent = [Decimal(str(p)) for p in history[-3:]]
        avg_recent = sum(recent) / len(recent)
        diff = current - avg_recent
        threshold = avg_recent * Decimal("0.02")
        if diff > threshold:
            return PriceTrend.RISING
        elif diff < -threshold:
            return PriceTrend.FALLING
        return PriceTrend.STABLE

    def _check_price_alerts(
        self, current_price: Decimal, alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check which price alerts are triggered."""
        triggered = []
        for alert in alerts:
            threshold = Decimal(str(alert.get("threshold_eur", 0)))
            alert_type = alert.get("alert_type", "")
            if alert_type == "above_threshold" and current_price > threshold:
                triggered.append({
                    "alert_type": alert_type,
                    "threshold_eur": float(threshold),
                    "current_price_eur": float(current_price),
                    "channels": alert.get("notification_channels", []),
                })
            elif alert_type == "below_threshold" and current_price < threshold:
                triggered.append({
                    "alert_type": alert_type,
                    "threshold_eur": float(threshold),
                    "current_price_eur": float(current_price),
                    "channels": alert.get("notification_channels", []),
                })
        return triggered


class ObligationForecastPhase:
    """
    Phase 2: Obligation Forecasting.

    Calculates upcoming certificate needs based on import pipeline
    (confirmed and planned shipments) and projects quarterly obligations
    for purchase planning.
    """

    PHASE_NAME = "obligation_forecast"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute obligation forecasting phase.

        Args:
            context: Workflow context with shipment data and prior outputs.

        Returns:
            PhaseResult with projected obligations and certificate needs.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            confirmed = config.get("confirmed_shipments", [])
            planned = config.get("planned_shipments", [])
            annual_estimate = Decimal(str(
                config.get("annual_obligation_estimate_tco2e", 0)
            ))
            surrendered_ytd = Decimal(str(
                config.get("certificates_surrendered_ytd", 0)
            ))
            current_holdings = config.get("current_holdings", [])

            # Calculate confirmed emissions from shipments
            confirmed_emissions = self._sum_emissions(confirmed)
            planned_emissions = self._sum_emissions(planned)
            total_pipeline = confirmed_emissions + planned_emissions

            outputs["confirmed_emissions_tco2e"] = float(confirmed_emissions)
            outputs["planned_emissions_tco2e"] = float(planned_emissions)
            outputs["total_pipeline_tco2e"] = float(total_pipeline)
            outputs["confirmed_shipment_count"] = len(confirmed)
            outputs["planned_shipment_count"] = len(planned)

            # Use higher of annual estimate or pipeline
            effective_obligation = max(annual_estimate, total_pipeline)
            outputs["effective_annual_obligation_tco2e"] = float(
                effective_obligation
            )

            # Calculate certificates currently held
            total_held = Decimal("0")
            for holding in current_holdings:
                total_held += Decimal(str(holding.get("quantity", 0)))
            outputs["certificates_held"] = float(total_held)

            # Net certificate need
            net_need = effective_obligation - total_held - surrendered_ytd
            if net_need < 0:
                net_need = Decimal("0")
            outputs["net_certificates_needed"] = float(net_need)
            outputs["certificates_surrendered_ytd"] = float(surrendered_ytd)

            # Quarterly projection
            quarterly_breakdown = self._project_quarterly(
                confirmed, planned, effective_obligation
            )
            outputs["quarterly_projection"] = quarterly_breakdown

            # Quarterly holding compliance check
            quarterly_required = effective_obligation * QUARTERLY_HOLDING_PCT
            outputs["quarterly_holding_required"] = float(quarterly_required)
            outputs["quarterly_holding_compliant"] = total_held >= quarterly_required
            if total_held < quarterly_required:
                shortfall = quarterly_required - total_held
                warnings.append(
                    f"Quarterly holding shortfall: {shortfall:.2f} certificates "
                    f"below 50% requirement ({quarterly_required:.2f})"
                )
                outputs["quarterly_holding_shortfall"] = float(shortfall)

            status = PhaseStatus.COMPLETED
            records = len(confirmed) + len(planned)

        except Exception as exc:
            logger.error("ObligationForecast failed: %s", exc, exc_info=True)
            errors.append(f"Obligation forecasting failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _sum_emissions(self, shipments: List[Dict[str, Any]]) -> Decimal:
        """Sum embedded emissions from a list of shipments."""
        total = Decimal("0")
        for s in shipments:
            total += Decimal(str(
                s.get("estimated_embedded_emissions_tco2e", 0)
            ))
        return total

    def _project_quarterly(
        self,
        confirmed: List[Dict[str, Any]],
        planned: List[Dict[str, Any]],
        annual: Decimal,
    ) -> Dict[str, Any]:
        """Project quarterly certificate obligations."""
        quarters: Dict[str, Decimal] = {
            "Q1": Decimal("0"), "Q2": Decimal("0"),
            "Q3": Decimal("0"), "Q4": Decimal("0"),
        }

        all_shipments = confirmed + planned
        for s in all_shipments:
            arrival = s.get("expected_arrival_date", "")
            emissions = Decimal(str(
                s.get("estimated_embedded_emissions_tco2e", 0)
            ))
            quarter = self._date_to_quarter(arrival)
            if quarter in quarters:
                quarters[quarter] += emissions

        # If projected total is less than annual estimate, distribute remainder
        projected = sum(quarters.values())
        if projected < annual:
            remainder = annual - projected
            per_quarter = remainder / 4
            for q in quarters:
                quarters[q] += per_quarter

        return {k: float(v.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
                for k, v in quarters.items()}

    def _date_to_quarter(self, date_str: str) -> str:
        """Convert ISO date to quarter label."""
        if not date_str or len(date_str) < 7:
            return "Q1"
        try:
            month = int(date_str[5:7])
            if month <= 3:
                return "Q1"
            elif month <= 6:
                return "Q2"
            elif month <= 9:
                return "Q3"
            return "Q4"
        except (ValueError, IndexError):
            return "Q1"


class PurchaseDecisionPhase:
    """
    Phase 3: Purchase Decision.

    Applies the configured buying strategy to determine certificate
    purchase quantity and timing. Supports five strategies:
    BUDGET_PACED, PRICE_TRIGGERED, BULK_QUARTERLY, DCA, OPPORTUNISTIC.
    """

    PHASE_NAME = "purchase_decision"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute purchase decision phase.

        Args:
            context: Workflow context with price and obligation data.

        Returns:
            PhaseResult with purchase recommendation.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            strategy = BuyingStrategy(config.get(
                "buying_strategy", BuyingStrategy.BUDGET_PACED.value
            ))
            price_output = context.get_phase_output("price_monitor")
            obligation_output = context.get_phase_output("obligation_forecast")

            current_price = Decimal(str(
                price_output.get("current_price_eur", 0)
            ))
            budget_remaining = Decimal(str(
                price_output.get("budget_remaining_eur", 0)
            ))
            net_needed = Decimal(str(
                obligation_output.get("net_certificates_needed", 0)
            ))
            current_week = config.get("current_week", 1)
            moving_avg = Decimal(str(
                price_output.get("moving_average_eur", float(current_price))
            ))

            outputs["strategy"] = strategy.value
            outputs["current_price_eur"] = float(current_price)
            outputs["net_certificates_needed"] = float(net_needed)
            outputs["budget_remaining_eur"] = float(budget_remaining)

            # Determine purchase quantity based on strategy
            if net_needed <= 0 or budget_remaining <= 0:
                purchase_qty = Decimal("0")
                outputs["decision"] = "NO_PURCHASE"
                outputs["reason"] = (
                    "No certificates needed" if net_needed <= 0
                    else "Budget exhausted"
                )
            elif strategy == BuyingStrategy.BUDGET_PACED:
                purchase_qty = self._strategy_budget_paced(
                    net_needed, current_price, budget_remaining, current_week
                )
                outputs["decision"] = "BUY"
                outputs["reason"] = "Budget-paced weekly allocation"
            elif strategy == BuyingStrategy.PRICE_TRIGGERED:
                threshold = Decimal(str(
                    config.get("price_trigger_threshold_eur",
                               float(DEFAULT_PRICE_THRESHOLD_EUR))
                ))
                purchase_qty = self._strategy_price_triggered(
                    net_needed, current_price, threshold
                )
                if purchase_qty > 0:
                    outputs["decision"] = "BUY"
                    outputs["reason"] = (
                        f"Price EUR {current_price} below "
                        f"threshold EUR {threshold}"
                    )
                else:
                    outputs["decision"] = "HOLD"
                    outputs["reason"] = (
                        f"Price EUR {current_price} above "
                        f"threshold EUR {threshold}"
                    )
            elif strategy == BuyingStrategy.BULK_QUARTERLY:
                purchase_qty = self._strategy_bulk_quarterly(
                    net_needed, current_week
                )
                if purchase_qty > 0:
                    outputs["decision"] = "BUY"
                    outputs["reason"] = "Quarterly bulk purchase week"
                else:
                    outputs["decision"] = "HOLD"
                    outputs["reason"] = "Not a quarterly purchase week"
            elif strategy == BuyingStrategy.DCA:
                weekly_amount = Decimal(str(
                    config.get("dca_weekly_amount_eur",
                               float(DEFAULT_DCA_WEEKLY_AMOUNT_EUR))
                ))
                purchase_qty = self._strategy_dca(
                    current_price, weekly_amount, net_needed
                )
                outputs["decision"] = "BUY"
                outputs["reason"] = (
                    f"DCA: EUR {weekly_amount} weekly allocation"
                )
            elif strategy == BuyingStrategy.OPPORTUNISTIC:
                factor = Decimal(str(
                    config.get("opportunistic_factor",
                               float(DEFAULT_OPPORTUNISTIC_FACTOR))
                ))
                purchase_qty = self._strategy_opportunistic(
                    net_needed, current_price, moving_avg, factor
                )
                if purchase_qty > 0:
                    outputs["decision"] = "BUY"
                    outputs["reason"] = (
                        f"Price EUR {current_price} below MA*{factor} = "
                        f"EUR {moving_avg * factor:.2f}"
                    )
                else:
                    outputs["decision"] = "HOLD"
                    outputs["reason"] = (
                        f"Price EUR {current_price} above "
                        f"opportunistic threshold"
                    )
            else:
                purchase_qty = Decimal("0")
                outputs["decision"] = "NO_PURCHASE"
                outputs["reason"] = f"Unknown strategy: {strategy}"

            # Cap by budget
            if current_price > 0 and purchase_qty > 0:
                max_affordable = budget_remaining / current_price
                if purchase_qty > max_affordable:
                    purchase_qty = max_affordable
                    warnings.append(
                        f"Purchase capped to {max_affordable:.2f} certificates "
                        f"by budget constraint"
                    )

            purchase_qty = purchase_qty.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            estimated_cost = (purchase_qty * current_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            outputs["recommended_quantity"] = float(purchase_qty)
            outputs["estimated_cost_eur"] = float(estimated_cost)
            outputs["order_type"] = (
                OrderType.MARKET.value if purchase_qty > 0
                else OrderType.LIMIT.value
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PurchaseDecision failed: %s", exc, exc_info=True)
            errors.append(f"Purchase decision failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _strategy_budget_paced(
        self,
        needed: Decimal,
        price: Decimal,
        budget: Decimal,
        current_week: int,
    ) -> Decimal:
        """Spread purchases evenly across remaining weeks."""
        remaining_weeks = max(WEEKS_PER_YEAR - current_week, 1)
        weekly_budget = budget / remaining_weeks
        if price <= 0:
            return Decimal("0")
        weekly_certs = weekly_budget / price
        return min(weekly_certs, needed)

    def _strategy_price_triggered(
        self,
        needed: Decimal,
        price: Decimal,
        threshold: Decimal,
    ) -> Decimal:
        """Buy full needed amount when price is below threshold."""
        if price <= threshold:
            return needed
        return Decimal("0")

    def _strategy_bulk_quarterly(
        self,
        needed: Decimal,
        current_week: int,
    ) -> Decimal:
        """Buy quarterly tranche on first week of each quarter."""
        quarterly_weeks = [1, 14, 27, 40]
        if current_week in quarterly_weeks:
            return needed / 4
        return Decimal("0")

    def _strategy_dca(
        self,
        price: Decimal,
        weekly_amount: Decimal,
        needed: Decimal,
    ) -> Decimal:
        """Fixed weekly EUR spend regardless of price."""
        if price <= 0:
            return Decimal("0")
        qty = weekly_amount / price
        return min(qty, needed)

    def _strategy_opportunistic(
        self,
        needed: Decimal,
        price: Decimal,
        moving_avg: Decimal,
        factor: Decimal,
    ) -> Decimal:
        """Buy when price dips below moving average * factor."""
        threshold = moving_avg * factor
        if price <= threshold:
            return needed
        return Decimal("0")


class OrderExecutionPhase:
    """
    Phase 4: Order Execution.

    Creates and submits purchase orders via the CBAM Registry API engine.
    Handles both market and limit orders, tracks execution status, and
    logs all order details for the audit trail.
    """

    PHASE_NAME = "order_execution"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute order submission phase.

        Args:
            context: Workflow context with purchase decision outputs.

        Returns:
            PhaseResult with order creation and execution results.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            decision = context.get_phase_output("purchase_decision")
            config = context.config
            auto_execute = config.get("auto_execute_orders", False)
            quantity = Decimal(str(decision.get("recommended_quantity", 0)))
            order_type_str = decision.get("order_type", OrderType.MARKET.value)
            price = Decimal(str(decision.get("current_price_eur", 0)))
            decision_action = decision.get("decision", "NO_PURCHASE")

            orders_created: List[Dict[str, Any]] = []
            orders_executed: List[Dict[str, Any]] = []

            if decision_action in ("BUY",) and quantity > 0:
                # Create purchase order
                order = PurchaseOrder(
                    order_type=OrderType(order_type_str),
                    quantity=float(quantity),
                    limit_price_eur=(
                        float(price) if order_type_str == OrderType.LIMIT.value
                        else None
                    ),
                    estimated_cost_eur=float(quantity * price),
                    status=OrderStatus.PENDING,
                )
                orders_created.append(order.model_dump())

                if auto_execute:
                    # Submit to registry API
                    execution_result = await self._submit_order(order, config)
                    orders_executed.append(execution_result)
                    if execution_result.get("status") == OrderStatus.REJECTED.value:
                        warnings.append(
                            f"Order {order.order_id} rejected: "
                            f"{execution_result.get('rejection_reason', 'Unknown')}"
                        )
                else:
                    warnings.append(
                        f"Order created but not auto-executed. "
                        f"Quantity: {quantity}, "
                        f"Est. cost: EUR {quantity * price:.2f}"
                    )
            else:
                outputs["decision"] = decision_action
                outputs["reason"] = decision.get("reason", "No purchase needed")

            outputs["orders_created"] = orders_created
            outputs["orders_executed"] = orders_executed
            outputs["orders_created_count"] = len(orders_created)
            outputs["orders_executed_count"] = len(orders_executed)
            outputs["total_quantity_ordered"] = float(quantity)
            outputs["total_estimated_cost_eur"] = float(quantity * price)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("OrderExecution failed: %s", exc, exc_info=True)
            errors.append(f"Order execution failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=len(orders_created),
        )

    async def _submit_order(
        self, order: PurchaseOrder, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit purchase order to CBAM Registry API.

        In production, this calls the RegistryApiEngine. Here we
        provide a stub that logs the order and returns a receipt.
        """
        logger.info(
            "Submitting order %s: %s %.4f certs at EUR %.2f",
            order.order_id, order.order_type.value,
            order.quantity, order.estimated_cost_eur,
        )
        receipt_id = str(uuid.uuid4())
        return {
            "order_id": order.order_id,
            "receipt_id": receipt_id,
            "status": OrderStatus.FILLED.value,
            "fill_price_eur": order.estimated_cost_eur / max(order.quantity, 1),
            "filled_at": datetime.utcnow().isoformat(),
            "submitted_at": datetime.utcnow().isoformat(),
        }


class PortfolioRebalancePhase:
    """
    Phase 5: Portfolio Rebalancing.

    Checks certificate expiry dates, performs surplus/deficit analysis,
    evaluates re-sell opportunities (max 1/3 within 12 months per
    Article 24), and optimizes the surrender sequence to use oldest
    certificates first.
    """

    PHASE_NAME = "portfolio_rebalance"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute portfolio rebalancing phase.

        Args:
            context: Workflow context with holdings and obligation data.

        Returns:
            PhaseResult with rebalancing recommendations.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            current_holdings = config.get("current_holdings", [])
            obligation_output = context.get_phase_output("obligation_forecast")
            effective_obligation = Decimal(str(
                obligation_output.get("effective_annual_obligation_tco2e", 0)
            ))
            today = datetime.utcnow()

            # Expiry analysis
            expiring_soon: List[Dict[str, Any]] = []
            expired: List[Dict[str, Any]] = []
            valid_holdings: List[Dict[str, Any]] = []
            total_held = Decimal("0")
            total_portfolio_value = Decimal("0")

            for h in current_holdings:
                qty = Decimal(str(h.get("quantity", 0)))
                unit_price = Decimal(str(h.get("unit_price_eur", 0)))
                expiry_str = h.get("expiry_date", "")
                purchase_str = h.get("purchase_date", "")

                total_held += qty
                total_portfolio_value += qty * unit_price

                if expiry_str:
                    try:
                        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
                        days_to_expiry = (expiry_date - today).days
                        if days_to_expiry < 0:
                            expired.append({
                                **h, "days_expired": abs(days_to_expiry)
                            })
                        elif days_to_expiry <= 90:
                            expiring_soon.append({
                                **h, "days_to_expiry": days_to_expiry
                            })
                            valid_holdings.append(h)
                        else:
                            valid_holdings.append(h)
                    except (ValueError, TypeError):
                        valid_holdings.append(h)
                else:
                    valid_holdings.append(h)

            outputs["total_certificates_held"] = float(total_held)
            outputs["portfolio_value_eur"] = float(total_portfolio_value)
            outputs["expiring_within_90_days"] = len(expiring_soon)
            outputs["expired_certificates"] = len(expired)
            outputs["valid_holdings_count"] = len(valid_holdings)

            if expiring_soon:
                warnings.append(
                    f"{len(expiring_soon)} certificate batch(es) expiring "
                    f"within 90 days"
                )
            if expired:
                warnings.append(
                    f"{len(expired)} certificate batch(es) have expired"
                )

            # Surplus/deficit analysis
            surplus = total_held - effective_obligation
            outputs["surplus_deficit_tco2e"] = float(surplus)
            outputs["has_surplus"] = surplus > 0
            outputs["has_deficit"] = surplus < 0

            # Re-sell opportunity analysis (Article 24)
            resell_opportunities: List[Dict[str, Any]] = []
            if surplus > 0:
                for h in valid_holdings:
                    qty = Decimal(str(h.get("quantity", 0)))
                    resell_eligible = h.get("resell_eligible", True)
                    resell_deadline_str = h.get("resell_deadline", "")

                    if not resell_eligible:
                        continue

                    # Check 12-month resell window
                    if resell_deadline_str:
                        try:
                            deadline = datetime.strptime(
                                resell_deadline_str, "%Y-%m-%d"
                            )
                            if deadline < today:
                                continue
                        except (ValueError, TypeError):
                            pass

                    max_resell = (qty * MAX_RESELL_FRACTION).quantize(
                        Decimal("0.0001"), rounding=ROUND_HALF_UP
                    )
                    if max_resell > 0:
                        resell_opportunities.append({
                            "batch_id": h.get("batch_id", ""),
                            "max_resell_quantity": float(max_resell),
                            "unit_price_eur": float(
                                Decimal(str(h.get("unit_price_eur", 0)))
                            ),
                            "max_resell_value_eur": float(
                                max_resell * Decimal(str(
                                    h.get("unit_price_eur", 0)
                                ))
                            ),
                        })

            outputs["resell_opportunities"] = resell_opportunities
            outputs["resell_opportunity_count"] = len(resell_opportunities)

            # Optimized surrender sequence (FIFO - oldest first)
            surrender_sequence = self._compute_surrender_sequence(
                valid_holdings, effective_obligation
            )
            outputs["surrender_sequence"] = surrender_sequence

            status = PhaseStatus.COMPLETED
            records = len(current_holdings)

        except Exception as exc:
            logger.error("PortfolioRebalance failed: %s", exc, exc_info=True)
            errors.append(f"Portfolio rebalancing failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _compute_surrender_sequence(
        self,
        holdings: List[Dict[str, Any]],
        obligation: Decimal,
    ) -> List[Dict[str, Any]]:
        """
        Compute optimal surrender sequence (FIFO: oldest purchased first).

        Surrenders oldest certificates first to minimize expiry risk
        and maximize remaining portfolio flexibility.
        """
        sorted_holdings = sorted(
            holdings, key=lambda h: h.get("purchase_date", "9999-12-31")
        )
        sequence: List[Dict[str, Any]] = []
        remaining = obligation

        for h in sorted_holdings:
            if remaining <= 0:
                break
            qty = Decimal(str(h.get("quantity", 0)))
            surrender_qty = min(qty, remaining)
            sequence.append({
                "batch_id": h.get("batch_id", ""),
                "purchase_date": h.get("purchase_date", ""),
                "surrender_quantity": float(surrender_qty),
                "unit_price_eur": float(
                    Decimal(str(h.get("unit_price_eur", 0)))
                ),
            })
            remaining -= surrender_qty

        return sequence


class ReportingPhase:
    """
    Phase 6: Reporting.

    Updates the certificate dashboard, generates portfolio report,
    notifies finance team of purchase activity, and logs compliance
    status for the audit trail.
    """

    PHASE_NAME = "reporting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute reporting phase.

        Args:
            context: Workflow context with all prior phase outputs.

        Returns:
            PhaseResult with reporting outputs and notifications.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            org_id = config.get("organization_id", "")
            year = config.get("reporting_year", 0)
            current_week = config.get("current_week", 0)

            price_output = context.get_phase_output("price_monitor")
            obligation_output = context.get_phase_output("obligation_forecast")
            decision_output = context.get_phase_output("purchase_decision")
            execution_output = context.get_phase_output("order_execution")
            portfolio_output = context.get_phase_output("portfolio_rebalance")

            # Dashboard summary
            dashboard = {
                "organization_id": org_id,
                "reporting_year": year,
                "week": current_week,
                "generated_at": datetime.utcnow().isoformat(),
                "price": {
                    "current_eur": price_output.get("current_price_eur", 0),
                    "trend": price_output.get("price_trend", "STABLE"),
                    "weekly_change_pct": price_output.get(
                        "weekly_change_pct", 0
                    ),
                    "moving_average_eur": price_output.get(
                        "moving_average_eur", 0
                    ),
                },
                "obligation": {
                    "annual_tco2e": obligation_output.get(
                        "effective_annual_obligation_tco2e", 0
                    ),
                    "certificates_held": obligation_output.get(
                        "certificates_held", 0
                    ),
                    "net_needed": obligation_output.get(
                        "net_certificates_needed", 0
                    ),
                    "quarterly_compliant": obligation_output.get(
                        "quarterly_holding_compliant", False
                    ),
                },
                "trading": {
                    "strategy": decision_output.get("strategy", ""),
                    "decision": decision_output.get("decision", ""),
                    "quantity_ordered": execution_output.get(
                        "total_quantity_ordered", 0
                    ),
                    "cost_eur": execution_output.get(
                        "total_estimated_cost_eur", 0
                    ),
                },
                "portfolio": {
                    "total_held": portfolio_output.get(
                        "total_certificates_held", 0
                    ),
                    "value_eur": portfolio_output.get(
                        "portfolio_value_eur", 0
                    ),
                    "surplus_deficit": portfolio_output.get(
                        "surplus_deficit_tco2e", 0
                    ),
                    "expiring_soon": portfolio_output.get(
                        "expiring_within_90_days", 0
                    ),
                    "resell_opportunities": portfolio_output.get(
                        "resell_opportunity_count", 0
                    ),
                },
                "budget": {
                    "remaining_eur": price_output.get(
                        "budget_remaining_eur", 0
                    ),
                    "utilization_pct": price_output.get(
                        "budget_utilization_pct", 0
                    ),
                },
            }
            outputs["dashboard"] = dashboard

            # Portfolio report
            report_id = str(uuid.uuid4())
            outputs["portfolio_report_id"] = report_id
            outputs["report_format"] = "JSON"

            # Finance notification
            notification = {
                "notification_id": str(uuid.uuid4()),
                "type": "certificate_trading_summary",
                "recipients": ["finance_team"],
                "subject": (
                    f"CBAM Certificate Trading Summary - "
                    f"Week {current_week}/{year}"
                ),
                "summary": {
                    "orders_placed": execution_output.get(
                        "orders_created_count", 0
                    ),
                    "total_cost_eur": execution_output.get(
                        "total_estimated_cost_eur", 0
                    ),
                    "budget_remaining_eur": price_output.get(
                        "budget_remaining_eur", 0
                    ),
                    "action_items": [],
                },
                "sent_at": datetime.utcnow().isoformat(),
            }

            # Build action items
            action_items = []
            if not obligation_output.get("quarterly_holding_compliant", True):
                action_items.append(
                    "URGENT: Quarterly holding compliance shortfall detected"
                )
            if portfolio_output.get("expiring_within_90_days", 0) > 0:
                action_items.append(
                    f"{portfolio_output['expiring_within_90_days']} "
                    f"certificate batch(es) expiring within 90 days"
                )
            if portfolio_output.get("resell_opportunity_count", 0) > 0:
                action_items.append(
                    f"{portfolio_output['resell_opportunity_count']} "
                    f"re-sell opportunity(ies) available"
                )
            notification["summary"]["action_items"] = action_items
            outputs["finance_notification"] = notification

            # Compliance status log
            compliance_log = {
                "log_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "organization_id": org_id,
                "year": year,
                "week": current_week,
                "quarterly_holding_compliant": obligation_output.get(
                    "quarterly_holding_compliant", False
                ),
                "certificates_held": portfolio_output.get(
                    "total_certificates_held", 0
                ),
                "obligation_tco2e": obligation_output.get(
                    "effective_annual_obligation_tco2e", 0
                ),
            }
            outputs["compliance_log"] = compliance_log

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Reporting failed: %s", exc, exc_info=True)
            errors.append(f"Reporting failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class CertificateTradingWorkflow:
    """
    Six-phase weekly certificate trading cycle workflow.

    Orchestrates the complete CBAM certificate trading process from
    price monitoring through purchase execution, portfolio management,
    and compliance reporting. Supports five configurable buying strategies
    and checkpoint/resume for interrupted executions.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered list of phase executor instances.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = CertificateTradingWorkflow()
        >>> input_data = CertificateTradingInput(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     current_week=10,
        ...     current_ets_price_eur=82.50,
        ...     annual_budget_eur=500000.0,
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "certificate_trading"

    PHASE_ORDER = [
        "price_monitor",
        "obligation_forecast",
        "purchase_decision",
        "order_execution",
        "portfolio_rebalance",
        "reporting",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the certificate trading workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "price_monitor": PriceMonitorPhase(),
            "obligation_forecast": ObligationForecastPhase(),
            "purchase_decision": PurchaseDecisionPhase(),
            "order_execution": OrderExecutionPhase(),
            "portfolio_rebalance": PortfolioRebalancePhase(),
            "reporting": ReportingPhase(),
        }

    async def run(
        self, input_data: CertificateTradingInput
    ) -> CertificateTradingResult:
        """
        Execute the complete 6-phase certificate trading workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            CertificateTradingResult with per-phase details and summary.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting certificate trading workflow %s for org=%s year=%d week=%d",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year, input_data.current_week,
        )

        # Build workflow context from input
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                logger.info("Phase '%s' already completed, skipping", phase_name)
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name in ("price_monitor", "obligation_forecast"):
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting workflow",
                            phase_name,
                        )
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=datetime.utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        # Determine final status
        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Build summary from phase outputs
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Certificate trading workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return CertificateTradingResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            current_price_eur=summary.get("current_price_eur", 0.0),
            price_trend=summary.get("price_trend", "STABLE"),
            total_obligation_tco2e=summary.get("total_obligation_tco2e", 0.0),
            certificates_held=summary.get("certificates_held", 0.0),
            certificates_needed=summary.get("certificates_needed", 0.0),
            orders_created=summary.get("orders_created", 0),
            orders_executed=summary.get("orders_executed", 0),
            portfolio_value_eur=summary.get("portfolio_value_eur", 0.0),
            budget_remaining_eur=summary.get("budget_remaining_eur", 0.0),
            resell_opportunities=summary.get("resell_opportunities", 0),
            expiring_soon=summary.get("expiring_soon", 0),
        )

    def _build_config(self, input_data: CertificateTradingInput) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "current_week": input_data.current_week,
            "buying_strategy": input_data.buying_strategy.value,
            "annual_budget_eur": input_data.annual_budget_eur,
            "budget_spent_ytd_eur": input_data.budget_spent_ytd_eur,
            "current_ets_price_eur": input_data.current_ets_price_eur,
            "price_history": input_data.price_history,
            "price_alerts": [a.model_dump() for a in input_data.price_alerts],
            "current_holdings": [h.model_dump() for h in input_data.current_holdings],
            "confirmed_shipments": [s.model_dump() for s in input_data.confirmed_shipments],
            "planned_shipments": [s.model_dump() for s in input_data.planned_shipments],
            "annual_obligation_estimate_tco2e": input_data.annual_obligation_estimate_tco2e,
            "certificates_surrendered_ytd": input_data.certificates_surrendered_ytd,
            "price_trigger_threshold_eur": input_data.price_trigger_threshold_eur,
            "dca_weekly_amount_eur": input_data.dca_weekly_amount_eur,
            "opportunistic_factor": input_data.opportunistic_factor,
            "auto_execute_orders": input_data.auto_execute_orders,
        }

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        price_out = context.get_phase_output("price_monitor")
        obligation_out = context.get_phase_output("obligation_forecast")
        execution_out = context.get_phase_output("order_execution")
        portfolio_out = context.get_phase_output("portfolio_rebalance")

        return {
            "current_price_eur": price_out.get("current_price_eur", 0.0),
            "price_trend": price_out.get("price_trend", "STABLE"),
            "total_obligation_tco2e": obligation_out.get(
                "effective_annual_obligation_tco2e", 0.0
            ),
            "certificates_held": portfolio_out.get(
                "total_certificates_held",
                obligation_out.get("certificates_held", 0.0),
            ),
            "certificates_needed": obligation_out.get(
                "net_certificates_needed", 0.0
            ),
            "orders_created": execution_out.get("orders_created_count", 0),
            "orders_executed": execution_out.get("orders_executed_count", 0),
            "portfolio_value_eur": portfolio_out.get(
                "portfolio_value_eur", 0.0
            ),
            "budget_remaining_eur": price_out.get(
                "budget_remaining_eur", 0.0
            ),
            "resell_opportunities": portfolio_out.get(
                "resell_opportunity_count", 0
            ),
            "expiring_soon": portfolio_out.get(
                "expiring_within_90_days", 0
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
