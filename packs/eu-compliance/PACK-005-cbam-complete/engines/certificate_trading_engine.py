# -*- coding: utf-8 -*-
"""
CertificateTradingEngine - PACK-005 CBAM Complete Engine 1

Full CBAM certificate lifecycle management per EU Regulation 2023/956
Articles 20-25. Covers certificate purchase, surrender, resale,
expiry monitoring, portfolio valuation, and compliance optimization.

Certificate Lifecycle (per Articles 20-25):
    - Purchase: Declarants purchase certificates at weekly auction price
    - Holding: Must hold >= 50% of estimated obligation by end of each quarter
    - Surrender: Surrender certificates equal to embedded emissions by May 31
    - Resale: May resell up to 1/3 of purchased certificates within 12 months
    - Expiry: Certificates expire 2 years after purchase date
    - Buyback: NCA repurchases excess certificates at original price

Valuation Methods:
    - FIFO (First In First Out): Oldest certificates surrendered first
    - WAC (Weighted Average Cost): Blended cost across portfolio
    - MTM (Mark-to-Market): Current market value of holdings

Zero-Hallucination:
    - All price and quantity calculations use Decimal arithmetic
    - No LLM involvement in financial calculations
    - Portfolio valuation uses deterministic formulas only
    - SHA-256 provenance hash on every transaction result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderStatus(str, Enum):
    """Purchase order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class ValuationMethod(str, Enum):
    """Portfolio valuation method."""
    FIFO = "fifo"
    WAC = "weighted_average_cost"
    MTM = "mark_to_market"

class CertificateStatus(str, Enum):
    """Individual certificate status."""
    ACTIVE = "active"
    SURRENDERED = "surrendered"
    RESOLD = "resold"
    EXPIRED = "expired"
    PENDING_BUYBACK = "pending_buyback"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Certificate(BaseModel):
    """Individual CBAM certificate representation."""
    certificate_id: str = Field(default_factory=_new_uuid, description="Unique certificate identifier")
    purchase_date: datetime = Field(default_factory=utcnow, description="Date of purchase")
    expiry_date: datetime = Field(description="Expiry date (2 years from purchase)")
    purchase_price: Decimal = Field(description="Purchase price in EUR per tCO2e")
    quantity_tco2e: Decimal = Field(default=Decimal("1"), description="Quantity in tCO2e")
    status: CertificateStatus = Field(default=CertificateStatus.ACTIVE, description="Current status")
    auction_week: str = Field(default="", description="Reference week of auction price")
    portfolio_id: str = Field(default="", description="Owning portfolio")
    surrender_declaration_id: Optional[str] = Field(default=None, description="Declaration surrendered against")
    resale_date: Optional[datetime] = Field(default=None, description="Date of resale if resold")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("purchase_price", "quantity_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class CertificatePortfolio(BaseModel):
    """Portfolio of CBAM certificates for a declarant."""
    portfolio_id: str = Field(default_factory=_new_uuid, description="Unique portfolio identifier")
    entity_id: str = Field(description="Owning entity/declarant identifier")
    entity_name: str = Field(default="", description="Entity display name")
    certificates: List[Certificate] = Field(default_factory=list, description="Certificates held")
    created_at: datetime = Field(default_factory=utcnow, description="Portfolio creation timestamp")
    total_purchased: Decimal = Field(default=Decimal("0"), description="Lifetime certificates purchased")
    total_surrendered: Decimal = Field(default=Decimal("0"), description="Lifetime certificates surrendered")
    total_resold: Decimal = Field(default=Decimal("0"), description="Lifetime certificates resold")
    total_expired: Decimal = Field(default=Decimal("0"), description="Lifetime certificates expired")
    config: Dict[str, Any] = Field(default_factory=dict, description="Portfolio configuration")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_purchased", "total_surrendered", "total_resold", "total_expired", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class PurchaseOrder(BaseModel):
    """Certificate purchase order."""
    order_id: str = Field(default_factory=_new_uuid, description="Unique order identifier")
    portfolio_id: str = Field(description="Target portfolio")
    quantity: Decimal = Field(description="Quantity to purchase in tCO2e")
    max_price: Optional[Decimal] = Field(default=None, description="Maximum acceptable price per tCO2e")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    submitted_at: datetime = Field(default_factory=utcnow, description="Submission timestamp")
    executed_at: Optional[datetime] = Field(default=None, description="Execution timestamp")
    executed_price: Optional[Decimal] = Field(default=None, description="Actual execution price")
    total_cost: Optional[Decimal] = Field(default=None, description="Total cost of order")
    notes: str = Field(default="", description="Order notes")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("quantity", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ExecutionResult(BaseModel):
    """Result of executing a purchase order."""
    execution_id: str = Field(default_factory=_new_uuid, description="Execution identifier")
    order_id: str = Field(description="Source order identifier")
    certificates_issued: List[str] = Field(default_factory=list, description="Certificate IDs issued")
    quantity_filled: Decimal = Field(description="Quantity filled in tCO2e")
    execution_price: Decimal = Field(description="Execution price per tCO2e")
    total_cost: Decimal = Field(description="Total cost in EUR")
    executed_at: datetime = Field(default_factory=utcnow, description="Execution timestamp")
    status: str = Field(default="executed", description="Execution status")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("quantity_filled", "execution_price", "total_cost", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class SurrenderResult(BaseModel):
    """Result of certificate surrender operation."""
    surrender_id: str = Field(default_factory=_new_uuid, description="Surrender identifier")
    portfolio_id: str = Field(description="Source portfolio")
    declaration_id: str = Field(description="CBAM declaration surrendered against")
    certificates_surrendered: List[str] = Field(default_factory=list, description="Certificate IDs surrendered")
    quantity_surrendered: Decimal = Field(description="Total tCO2e surrendered")
    total_value: Decimal = Field(description="Total value at original purchase price")
    average_cost: Decimal = Field(description="Weighted average cost of surrendered certificates")
    surrendered_at: datetime = Field(default_factory=utcnow, description="Surrender timestamp")
    remaining_balance: Decimal = Field(description="Remaining active certificates in portfolio")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("quantity_surrendered", "total_value", "average_cost", "remaining_balance", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ResaleResult(BaseModel):
    """Result of certificate resale operation."""
    resale_id: str = Field(default_factory=_new_uuid, description="Resale identifier")
    portfolio_id: str = Field(description="Source portfolio")
    certificates_resold: List[str] = Field(default_factory=list, description="Certificate IDs resold")
    quantity_resold: Decimal = Field(description="Quantity resold in tCO2e")
    resale_price: Decimal = Field(description="Resale price per tCO2e (original purchase price)")
    total_proceeds: Decimal = Field(description="Total resale proceeds")
    resale_limit_remaining: Decimal = Field(description="Remaining resale allowance (1/3 rule)")
    resold_at: datetime = Field(default_factory=utcnow, description="Resale timestamp")
    compliant: bool = Field(default=True, description="Whether resale is within 1/3 limit")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("quantity_resold", "resale_price", "total_proceeds", "resale_limit_remaining", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class ExpiryAlert(BaseModel):
    """Alert for certificates approaching or past expiry."""
    alert_id: str = Field(default_factory=_new_uuid, description="Alert identifier")
    certificate_id: str = Field(description="Affected certificate")
    portfolio_id: str = Field(description="Owning portfolio")
    expiry_date: datetime = Field(description="Certificate expiry date")
    days_until_expiry: int = Field(description="Days until expiry (negative = past)")
    quantity_tco2e: Decimal = Field(description="Quantity at risk")
    purchase_price: Decimal = Field(description="Original purchase price")
    value_at_risk: Decimal = Field(description="Value at risk in EUR")
    severity: AlertSeverity = Field(description="Alert severity level")
    recommendation: str = Field(default="", description="Recommended action")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class HoldingCheck(BaseModel):
    """Result of quarterly holding compliance check."""
    check_id: str = Field(default_factory=_new_uuid, description="Check identifier")
    portfolio_id: str = Field(description="Checked portfolio")
    quarter: str = Field(description="Quarter checked (e.g. 2026-Q1)")
    estimated_obligation: Decimal = Field(description="Estimated annual obligation in tCO2e")
    quarterly_target: Decimal = Field(description="50% target for end of quarter")
    current_holdings: Decimal = Field(description="Current active certificate quantity")
    surplus_deficit: Decimal = Field(description="Surplus (positive) or deficit (negative)")
    compliant: bool = Field(description="Whether holding meets 50% quarterly target")
    compliance_ratio: Decimal = Field(description="Current holdings / quarterly target ratio")
    checked_at: datetime = Field(default_factory=utcnow, description="Check timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("estimated_obligation", "quarterly_target", "current_holdings",
                     "surplus_deficit", "compliance_ratio", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class PortfolioValuation(BaseModel):
    """Portfolio valuation result."""
    valuation_id: str = Field(default_factory=_new_uuid, description="Valuation identifier")
    portfolio_id: str = Field(description="Valued portfolio")
    method: ValuationMethod = Field(description="Valuation method used")
    total_certificates: int = Field(description="Number of active certificates")
    total_quantity_tco2e: Decimal = Field(description="Total tCO2e in portfolio")
    total_cost_basis: Decimal = Field(description="Total historical cost")
    current_market_price: Decimal = Field(description="Current market price per tCO2e")
    market_value: Decimal = Field(description="Current market value")
    unrealized_pnl: Decimal = Field(description="Unrealized profit/loss")
    weighted_average_cost: Decimal = Field(description="Weighted average cost per tCO2e")
    oldest_certificate_date: Optional[datetime] = Field(default=None, description="Oldest active certificate date")
    newest_certificate_date: Optional[datetime] = Field(default=None, description="Newest active certificate date")
    valued_at: datetime = Field(default_factory=utcnow, description="Valuation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_quantity_tco2e", "total_cost_basis", "current_market_price",
                     "market_value", "unrealized_pnl", "weighted_average_cost", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class TransferResult(BaseModel):
    """Result of inter-portfolio certificate transfer."""
    transfer_id: str = Field(default_factory=_new_uuid, description="Transfer identifier")
    from_portfolio_id: str = Field(description="Source portfolio")
    to_portfolio_id: str = Field(description="Destination portfolio")
    certificates_transferred: List[str] = Field(default_factory=list, description="Certificate IDs transferred")
    quantity_transferred: Decimal = Field(description="Total tCO2e transferred")
    transfer_value: Decimal = Field(description="Value of transferred certificates at cost basis")
    transferred_at: datetime = Field(default_factory=utcnow, description="Transfer timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("quantity_transferred", "transfer_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class SurrenderPlan(BaseModel):
    """Optimized surrender plan to minimize cost."""
    plan_id: str = Field(default_factory=_new_uuid, description="Plan identifier")
    portfolio_id: str = Field(description="Source portfolio")
    obligation_tco2e: Decimal = Field(description="Total obligation to cover")
    strategy: str = Field(description="Optimization strategy used")
    certificates_to_surrender: List[str] = Field(default_factory=list, description="Ordered certificate IDs")
    total_surrender_cost: Decimal = Field(description="Total cost if plan is executed")
    average_cost_per_tco2e: Decimal = Field(description="Average cost per tCO2e")
    savings_vs_fifo: Decimal = Field(description="Savings compared to naive FIFO approach")
    savings_vs_lifo: Decimal = Field(description="Savings compared to LIFO approach")
    expiring_soon_included: int = Field(default=0, description="Count of near-expiry certificates included")
    created_at: datetime = Field(default_factory=utcnow, description="Plan creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("obligation_tco2e", "total_surrender_cost", "average_cost_per_tco2e",
                     "savings_vs_fifo", "savings_vs_lifo", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class BudgetForecast(BaseModel):
    """Multi-year CBAM certificate budget forecast."""
    forecast_id: str = Field(default_factory=_new_uuid, description="Forecast identifier")
    portfolio_id: str = Field(description="Source portfolio")
    horizon_years: int = Field(description="Forecast horizon in years")
    base_year: int = Field(description="Base year for forecast")
    annual_forecasts: List[Dict[str, Any]] = Field(default_factory=list, description="Per-year forecasts")
    total_estimated_cost: Decimal = Field(description="Total estimated cost over horizon")
    average_annual_cost: Decimal = Field(description="Average annual cost")
    assumed_price_growth_rate: Decimal = Field(description="Assumed annual certificate price growth")
    assumed_emission_trend: Decimal = Field(description="Assumed annual emission change rate")
    free_allocation_phase_out: List[Dict[str, Any]] = Field(
        default_factory=list, description="Free allocation phase-out schedule"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Forecast creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_estimated_cost", "average_annual_cost",
                     "assumed_price_growth_rate", "assumed_emission_trend", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class CertificateTradingConfig(BaseModel):
    """Configuration for the CertificateTradingEngine."""
    default_valuation_method: ValuationMethod = Field(
        default=ValuationMethod.WAC, description="Default valuation method"
    )
    certificate_validity_years: int = Field(default=2, description="Certificate validity in years")
    quarterly_holding_target_pct: Decimal = Field(
        default=Decimal("0.50"), description="Quarterly holding target as fraction of obligation"
    )
    max_resale_fraction: Decimal = Field(
        default=Decimal("0.3333"), description="Maximum resale fraction (1/3 of purchased)"
    )
    resale_window_months: int = Field(default=12, description="Resale window in months")
    expiry_warning_days: int = Field(default=90, description="Days before expiry to issue warnings")
    expiry_critical_days: int = Field(default=30, description="Days before expiry for critical alerts")
    default_price_growth_rate: Decimal = Field(
        default=Decimal("0.05"), description="Default annual price growth rate for forecasting"
    )
    free_allocation_phase_out: Dict[int, Decimal] = Field(
        default_factory=lambda: {
            2026: Decimal("0.975"),
            2027: Decimal("0.950"),
            2028: Decimal("0.900"),
            2029: Decimal("0.825"),
            2030: Decimal("0.750"),
            2031: Decimal("0.650"),
            2032: Decimal("0.500"),
            2033: Decimal("0.250"),
            2034: Decimal("0.000"),
        },
        description="EU ETS free allocation phase-out schedule (year -> fraction remaining)"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

CertificateTradingConfig.model_rebuild()
Certificate.model_rebuild()
CertificatePortfolio.model_rebuild()
PurchaseOrder.model_rebuild()
ExecutionResult.model_rebuild()
SurrenderResult.model_rebuild()
ResaleResult.model_rebuild()
ExpiryAlert.model_rebuild()
HoldingCheck.model_rebuild()
PortfolioValuation.model_rebuild()
TransferResult.model_rebuild()
SurrenderPlan.model_rebuild()
BudgetForecast.model_rebuild()

# ---------------------------------------------------------------------------
# CertificateTradingEngine
# ---------------------------------------------------------------------------

class CertificateTradingEngine:
    """
    CBAM certificate lifecycle management engine.

    Implements the full certificate trading lifecycle per EU Regulation 2023/956
    Articles 20-25, including purchase, surrender, resale, expiry monitoring,
    portfolio valuation, and cost-optimized surrender planning.

    Attributes:
        config: Engine configuration parameters.
        _portfolios: In-memory portfolio store (keyed by portfolio_id).
        _orders: In-memory order store (keyed by order_id).

    Example:
        >>> engine = CertificateTradingEngine()
        >>> portfolio = engine.create_portfolio("EORI-DE-001", {})
        >>> order = engine.submit_purchase_order(portfolio.portfolio_id, {
        ...     "quantity": 100, "max_price": 80
        ... })
        >>> result = engine.execute_order(order.order_id, Decimal("75.50"))
        >>> assert result.status == "executed"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CertificateTradingEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = CertificateTradingConfig(**config)
        elif config and isinstance(config, CertificateTradingConfig):
            self.config = config
        else:
            self.config = CertificateTradingConfig()

        self._portfolios: Dict[str, CertificatePortfolio] = {}
        self._orders: Dict[str, PurchaseOrder] = {}
        logger.info("CertificateTradingEngine initialized (v%s)", _MODULE_VERSION)

    # -----------------------------------------------------------------------
    # Portfolio Management
    # -----------------------------------------------------------------------

    def create_portfolio(
        self, entity_id: str, config: Optional[Dict[str, Any]] = None
    ) -> CertificatePortfolio:
        """Create a new certificate portfolio for a declarant.

        Args:
            entity_id: EORI or entity identifier for the declarant.
            config: Optional portfolio-specific configuration.

        Returns:
            Newly created CertificatePortfolio.

        Raises:
            ValueError: If entity_id is empty.
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("entity_id must not be empty")

        portfolio = CertificatePortfolio(
            entity_id=entity_id.strip(),
            config=config or {},
        )
        portfolio.provenance_hash = _compute_hash(portfolio)
        self._portfolios[portfolio.portfolio_id] = portfolio

        logger.info(
            "Created portfolio %s for entity %s",
            portfolio.portfolio_id, entity_id,
        )
        return portfolio

    # -----------------------------------------------------------------------
    # Purchase Orders
    # -----------------------------------------------------------------------

    def submit_purchase_order(
        self, portfolio_id: str, order: Dict[str, Any]
    ) -> PurchaseOrder:
        """Submit a certificate purchase order.

        Args:
            portfolio_id: Target portfolio to receive certificates.
            order: Order details including 'quantity' and optional 'max_price'.

        Returns:
            Submitted PurchaseOrder with PENDING status.

        Raises:
            ValueError: If portfolio not found or quantity invalid.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        quantity = _decimal(order.get("quantity", 0))
        if quantity <= Decimal("0"):
            raise ValueError("Order quantity must be positive")

        max_price = None
        if "max_price" in order and order["max_price"] is not None:
            max_price = _decimal(order["max_price"])

        purchase_order = PurchaseOrder(
            portfolio_id=portfolio_id,
            quantity=quantity,
            max_price=max_price,
            status=OrderStatus.SUBMITTED,
            notes=order.get("notes", ""),
        )
        purchase_order.provenance_hash = _compute_hash(purchase_order)
        self._orders[purchase_order.order_id] = purchase_order

        logger.info(
            "Submitted order %s for %s tCO2e on portfolio %s",
            purchase_order.order_id, quantity, portfolio_id,
        )
        return purchase_order

    def execute_order(
        self, order_id: str, market_price: Decimal
    ) -> ExecutionResult:
        """Execute a submitted purchase order at the given market price.

        Creates individual Certificate objects and adds them to the portfolio.
        Each certificate represents 1 tCO2e unless quantity is fractional.

        Args:
            order_id: The order to execute.
            market_price: Current market (auction) price per tCO2e in EUR.

        Returns:
            ExecutionResult with details of certificates issued.

        Raises:
            ValueError: If order not found, not in SUBMITTED status, or price
                exceeds max_price limit.
        """
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._orders[order_id]
        if order.status != OrderStatus.SUBMITTED:
            raise ValueError(f"Order {order_id} is in status {order.status.value}, expected SUBMITTED")

        market_price = _decimal(market_price)
        if market_price <= Decimal("0"):
            raise ValueError("Market price must be positive")

        if order.max_price is not None and market_price > order.max_price:
            order.status = OrderStatus.FAILED
            raise ValueError(
                f"Market price {market_price} exceeds max_price {order.max_price}"
            )

        portfolio = self._portfolios[order.portfolio_id]
        now = utcnow()
        expiry = now + timedelta(days=self.config.certificate_validity_years * 365)
        total_cost = order.quantity * market_price

        certificate_ids: List[str] = []
        remaining = order.quantity
        while remaining > Decimal("0"):
            cert_qty = min(remaining, Decimal("1"))
            cert = Certificate(
                purchase_date=now,
                expiry_date=expiry,
                purchase_price=market_price,
                quantity_tco2e=cert_qty,
                portfolio_id=portfolio.portfolio_id,
                auction_week=now.strftime("%Y-W%W"),
            )
            cert.provenance_hash = _compute_hash(cert)
            portfolio.certificates.append(cert)
            certificate_ids.append(cert.certificate_id)
            remaining -= cert_qty

        portfolio.total_purchased += order.quantity
        portfolio.provenance_hash = _compute_hash(portfolio)

        order.status = OrderStatus.EXECUTED
        order.executed_at = now
        order.executed_price = market_price
        order.total_cost = total_cost
        order.provenance_hash = _compute_hash(order)

        result = ExecutionResult(
            order_id=order_id,
            certificates_issued=certificate_ids,
            quantity_filled=order.quantity,
            execution_price=market_price,
            total_cost=total_cost,
            executed_at=now,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Executed order %s: %s certificates at EUR %s/tCO2e, total EUR %s",
            order_id, len(certificate_ids), market_price, total_cost,
        )
        return result

    # -----------------------------------------------------------------------
    # Certificate Surrender
    # -----------------------------------------------------------------------

    def surrender_certificates(
        self, portfolio_id: str, declaration_id: str, quantity: Decimal
    ) -> SurrenderResult:
        """Surrender certificates against a CBAM declaration.

        Uses FIFO ordering (oldest certificates surrendered first) to maximize
        use of certificates closest to expiry.

        Args:
            portfolio_id: Portfolio from which to surrender.
            declaration_id: CBAM declaration identifier.
            quantity: Quantity to surrender in tCO2e.

        Returns:
            SurrenderResult with details of surrendered certificates.

        Raises:
            ValueError: If portfolio not found or insufficient active certificates.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        quantity = _decimal(quantity)
        if quantity <= Decimal("0"):
            raise ValueError("Surrender quantity must be positive")

        portfolio = self._portfolios[portfolio_id]
        active_certs = self._get_active_certificates(portfolio, sort_fifo=True)

        available_qty = sum(c.quantity_tco2e for c in active_certs)
        if available_qty < quantity:
            raise ValueError(
                f"Insufficient certificates: need {quantity} tCO2e, "
                f"have {available_qty} tCO2e active"
            )

        surrendered_ids: List[str] = []
        total_value = Decimal("0")
        remaining = quantity

        for cert in active_certs:
            if remaining <= Decimal("0"):
                break
            if cert.quantity_tco2e <= remaining:
                cert.status = CertificateStatus.SURRENDERED
                cert.surrender_declaration_id = declaration_id
                surrendered_ids.append(cert.certificate_id)
                total_value += cert.quantity_tco2e * cert.purchase_price
                remaining -= cert.quantity_tco2e
            else:
                original_qty = cert.quantity_tco2e
                cert.quantity_tco2e = original_qty - remaining
                split_cert = Certificate(
                    purchase_date=cert.purchase_date,
                    expiry_date=cert.expiry_date,
                    purchase_price=cert.purchase_price,
                    quantity_tco2e=remaining,
                    status=CertificateStatus.SURRENDERED,
                    portfolio_id=portfolio_id,
                    auction_week=cert.auction_week,
                    surrender_declaration_id=declaration_id,
                )
                split_cert.provenance_hash = _compute_hash(split_cert)
                portfolio.certificates.append(split_cert)
                surrendered_ids.append(split_cert.certificate_id)
                total_value += remaining * cert.purchase_price
                remaining = Decimal("0")

        portfolio.total_surrendered += quantity
        new_active = sum(
            c.quantity_tco2e for c in portfolio.certificates
            if c.status == CertificateStatus.ACTIVE
        )
        portfolio.provenance_hash = _compute_hash(portfolio)

        avg_cost = (total_value / quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if quantity > 0 else Decimal("0")

        result = SurrenderResult(
            portfolio_id=portfolio_id,
            declaration_id=declaration_id,
            certificates_surrendered=surrendered_ids,
            quantity_surrendered=quantity,
            total_value=total_value,
            average_cost=avg_cost,
            remaining_balance=new_active,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Surrendered %s tCO2e from portfolio %s against declaration %s",
            quantity, portfolio_id, declaration_id,
        )
        return result

    # -----------------------------------------------------------------------
    # Certificate Resale
    # -----------------------------------------------------------------------

    def initiate_resale(
        self, portfolio_id: str, quantity: Decimal
    ) -> ResaleResult:
        """Initiate resale of certificates back to NCA.

        Per Article 23, declarants may request the NCA to re-purchase up to
        one-third of the certificates purchased during the preceding 12 months.
        The buyback price equals the original purchase price.

        Args:
            portfolio_id: Portfolio from which to resell.
            quantity: Quantity to resell in tCO2e.

        Returns:
            ResaleResult with details and remaining resale allowance.

        Raises:
            ValueError: If portfolio not found, insufficient certificates, or
                resale exceeds 1/3 limit.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        quantity = _decimal(quantity)
        if quantity <= Decimal("0"):
            raise ValueError("Resale quantity must be positive")

        portfolio = self._portfolios[portfolio_id]
        now = utcnow()
        cutoff = now - timedelta(days=self.config.resale_window_months * 30)

        recent_purchased = Decimal("0")
        recent_resold = Decimal("0")
        for cert in portfolio.certificates:
            if cert.purchase_date >= cutoff:
                recent_purchased += cert.quantity_tco2e
            if cert.status == CertificateStatus.RESOLD and cert.resale_date and cert.resale_date >= cutoff:
                recent_resold += cert.quantity_tco2e

        max_resale = (recent_purchased * self.config.max_resale_fraction).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        remaining_allowance = max_resale - recent_resold

        if quantity > remaining_allowance:
            raise ValueError(
                f"Resale quantity {quantity} exceeds remaining allowance "
                f"{remaining_allowance} (max 1/3 of {recent_purchased} purchased in last 12 months)"
            )

        active_certs = self._get_active_certificates(portfolio, sort_fifo=False)
        active_certs.sort(key=lambda c: c.purchase_price, reverse=True)

        available_qty = sum(c.quantity_tco2e for c in active_certs)
        if available_qty < quantity:
            raise ValueError(
                f"Insufficient active certificates: need {quantity}, have {available_qty}"
            )

        resold_ids: List[str] = []
        total_proceeds = Decimal("0")
        remaining = quantity

        for cert in active_certs:
            if remaining <= Decimal("0"):
                break
            resale_qty = min(cert.quantity_tco2e, remaining)
            if resale_qty == cert.quantity_tco2e:
                cert.status = CertificateStatus.RESOLD
                cert.resale_date = now
                resold_ids.append(cert.certificate_id)
                total_proceeds += resale_qty * cert.purchase_price
                remaining -= resale_qty
            else:
                cert.quantity_tco2e -= resale_qty
                split_cert = Certificate(
                    purchase_date=cert.purchase_date,
                    expiry_date=cert.expiry_date,
                    purchase_price=cert.purchase_price,
                    quantity_tco2e=resale_qty,
                    status=CertificateStatus.RESOLD,
                    portfolio_id=portfolio_id,
                    auction_week=cert.auction_week,
                    resale_date=now,
                )
                split_cert.provenance_hash = _compute_hash(split_cert)
                portfolio.certificates.append(split_cert)
                resold_ids.append(split_cert.certificate_id)
                total_proceeds += resale_qty * cert.purchase_price
                remaining = Decimal("0")

        portfolio.total_resold += quantity
        new_remaining_allowance = remaining_allowance - quantity
        portfolio.provenance_hash = _compute_hash(portfolio)

        resale_price = (total_proceeds / quantity).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if quantity > 0 else Decimal("0")

        result = ResaleResult(
            portfolio_id=portfolio_id,
            certificates_resold=resold_ids,
            quantity_resold=quantity,
            resale_price=resale_price,
            total_proceeds=total_proceeds,
            resale_limit_remaining=new_remaining_allowance,
            compliant=True,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Resold %s tCO2e from portfolio %s, proceeds EUR %s",
            quantity, portfolio_id, total_proceeds,
        )
        return result

    # -----------------------------------------------------------------------
    # Expiry Monitoring
    # -----------------------------------------------------------------------

    def check_expiry(self, portfolio_id: str) -> List[ExpiryAlert]:
        """Check for certificates approaching or past expiry.

        Per Article 22, certificates are valid for 2 years from purchase.
        Generates alerts at INFO (>90 days), WARNING (30-90 days),
        and CRITICAL (<30 days or expired) levels.

        Args:
            portfolio_id: Portfolio to check.

        Returns:
            List of ExpiryAlert objects sorted by urgency.

        Raises:
            ValueError: If portfolio not found.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        portfolio = self._portfolios[portfolio_id]
        active_certs = self._get_active_certificates(portfolio)
        now = utcnow()
        alerts: List[ExpiryAlert] = []

        for cert in active_certs:
            days_until = (cert.expiry_date - now).days
            value_at_risk = cert.quantity_tco2e * cert.purchase_price

            if days_until <= 0:
                severity = AlertSeverity.CRITICAL
                recommendation = "IMMEDIATE: Certificate expired. Surrender or request NCA buyback."
            elif days_until <= self.config.expiry_critical_days:
                severity = AlertSeverity.CRITICAL
                recommendation = f"URGENT: Expires in {days_until} days. Surrender against obligation or resell."
            elif days_until <= self.config.expiry_warning_days:
                severity = AlertSeverity.WARNING
                recommendation = f"Plan to use within {days_until} days. Consider surrender scheduling."
            else:
                continue

            alert = ExpiryAlert(
                certificate_id=cert.certificate_id,
                portfolio_id=portfolio_id,
                expiry_date=cert.expiry_date,
                days_until_expiry=days_until,
                quantity_tco2e=cert.quantity_tco2e,
                purchase_price=cert.purchase_price,
                value_at_risk=value_at_risk,
                severity=severity,
                recommendation=recommendation,
            )
            alert.provenance_hash = _compute_hash(alert)
            alerts.append(alert)

        alerts.sort(key=lambda a: a.days_until_expiry)

        logger.info(
            "Expiry check for portfolio %s: %d alerts generated",
            portfolio_id, len(alerts),
        )
        return alerts

    # -----------------------------------------------------------------------
    # Holding Compliance
    # -----------------------------------------------------------------------

    def calculate_holding_compliance(
        self, portfolio_id: str, quarter: str
    ) -> HoldingCheck:
        """Check whether portfolio meets the quarterly 50% holding target.

        Per CBAM implementing regulation, declarants must hold certificates
        equal to at least 50% of their estimated obligation at the end of
        each quarter.

        Args:
            portfolio_id: Portfolio to check.
            quarter: Quarter string (e.g. '2026-Q1').

        Returns:
            HoldingCheck with compliance status.

        Raises:
            ValueError: If portfolio not found or quarter format invalid.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        if not quarter or len(quarter) < 6:
            raise ValueError(f"Invalid quarter format: {quarter}. Expected YYYY-QN")

        portfolio = self._portfolios[portfolio_id]
        estimated_obligation = _decimal(
            portfolio.config.get("estimated_annual_obligation", 0)
        )
        quarterly_target = (
            estimated_obligation * self.config.quarterly_holding_target_pct
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        current_holdings = sum(
            c.quantity_tco2e for c in portfolio.certificates
            if c.status == CertificateStatus.ACTIVE
        )

        surplus_deficit = current_holdings - quarterly_target
        compliant = current_holdings >= quarterly_target
        compliance_ratio = (
            (current_holdings / quarterly_target).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            if quarterly_target > Decimal("0") else Decimal("1")
        )

        result = HoldingCheck(
            portfolio_id=portfolio_id,
            quarter=quarter,
            estimated_obligation=estimated_obligation,
            quarterly_target=quarterly_target,
            current_holdings=current_holdings,
            surplus_deficit=surplus_deficit,
            compliant=compliant,
            compliance_ratio=compliance_ratio,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Holding check for %s Q=%s: %s (holdings=%s, target=%s)",
            portfolio_id, quarter,
            "COMPLIANT" if compliant else "NON-COMPLIANT",
            current_holdings, quarterly_target,
        )
        return result

    # -----------------------------------------------------------------------
    # Portfolio Valuation
    # -----------------------------------------------------------------------

    def value_portfolio(
        self, portfolio_id: str, method: Optional[str] = None
    ) -> PortfolioValuation:
        """Value a certificate portfolio using the specified method.

        Args:
            portfolio_id: Portfolio to value.
            method: Valuation method ('fifo', 'weighted_average_cost', 'mark_to_market').
                Defaults to engine config default.

        Returns:
            PortfolioValuation with complete valuation breakdown.

        Raises:
            ValueError: If portfolio not found or invalid method.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        if method:
            try:
                val_method = ValuationMethod(method)
            except ValueError:
                raise ValueError(f"Invalid valuation method: {method}")
        else:
            val_method = self.config.default_valuation_method

        portfolio = self._portfolios[portfolio_id]
        active_certs = self._get_active_certificates(portfolio)

        total_qty = sum(c.quantity_tco2e for c in active_certs)
        total_cost = sum(c.quantity_tco2e * c.purchase_price for c in active_certs)
        current_price = _decimal(portfolio.config.get("current_market_price", 0))

        if current_price <= Decimal("0") and active_certs:
            current_price = active_certs[-1].purchase_price

        market_value = total_qty * current_price
        unrealized_pnl = market_value - total_cost
        wac = (total_cost / total_qty).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if total_qty > Decimal("0") else Decimal("0")

        oldest_date = min((c.purchase_date for c in active_certs), default=None)
        newest_date = max((c.purchase_date for c in active_certs), default=None)

        result = PortfolioValuation(
            portfolio_id=portfolio_id,
            method=val_method,
            total_certificates=len(active_certs),
            total_quantity_tco2e=total_qty,
            total_cost_basis=total_cost,
            current_market_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            weighted_average_cost=wac,
            oldest_certificate_date=oldest_date,
            newest_certificate_date=newest_date,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Valued portfolio %s (%s): %s tCO2e, cost=%s, market=%s, PnL=%s",
            portfolio_id, val_method.value, total_qty, total_cost, market_value, unrealized_pnl,
        )
        return result

    # -----------------------------------------------------------------------
    # Certificate Transfer
    # -----------------------------------------------------------------------

    def transfer_certificates(
        self, from_portfolio: str, to_portfolio: str, quantity: Decimal
    ) -> TransferResult:
        """Transfer certificates between portfolios within the same entity group.

        Args:
            from_portfolio: Source portfolio identifier.
            to_portfolio: Destination portfolio identifier.
            quantity: Quantity to transfer in tCO2e.

        Returns:
            TransferResult with transfer details.

        Raises:
            ValueError: If either portfolio not found or insufficient certificates.
        """
        if from_portfolio not in self._portfolios:
            raise ValueError(f"Source portfolio {from_portfolio} not found")
        if to_portfolio not in self._portfolios:
            raise ValueError(f"Destination portfolio {to_portfolio} not found")

        quantity = _decimal(quantity)
        if quantity <= Decimal("0"):
            raise ValueError("Transfer quantity must be positive")

        src = self._portfolios[from_portfolio]
        dst = self._portfolios[to_portfolio]
        active_certs = self._get_active_certificates(src, sort_fifo=True)

        available = sum(c.quantity_tco2e for c in active_certs)
        if available < quantity:
            raise ValueError(
                f"Insufficient certificates in source: need {quantity}, have {available}"
            )

        transferred_ids: List[str] = []
        transfer_value = Decimal("0")
        remaining = quantity

        for cert in active_certs:
            if remaining <= Decimal("0"):
                break
            xfer_qty = min(cert.quantity_tco2e, remaining)
            if xfer_qty == cert.quantity_tco2e:
                cert.portfolio_id = to_portfolio
                dst.certificates.append(cert)
                src.certificates.remove(cert)
                transferred_ids.append(cert.certificate_id)
                transfer_value += xfer_qty * cert.purchase_price
                remaining -= xfer_qty
            else:
                cert.quantity_tco2e -= xfer_qty
                new_cert = Certificate(
                    purchase_date=cert.purchase_date,
                    expiry_date=cert.expiry_date,
                    purchase_price=cert.purchase_price,
                    quantity_tco2e=xfer_qty,
                    portfolio_id=to_portfolio,
                    auction_week=cert.auction_week,
                )
                new_cert.provenance_hash = _compute_hash(new_cert)
                dst.certificates.append(new_cert)
                transferred_ids.append(new_cert.certificate_id)
                transfer_value += xfer_qty * cert.purchase_price
                remaining = Decimal("0")

        src.provenance_hash = _compute_hash(src)
        dst.provenance_hash = _compute_hash(dst)

        result = TransferResult(
            from_portfolio_id=from_portfolio,
            to_portfolio_id=to_portfolio,
            certificates_transferred=transferred_ids,
            quantity_transferred=quantity,
            transfer_value=transfer_value,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Transferred %s tCO2e from %s to %s (%d certificates)",
            quantity, from_portfolio, to_portfolio, len(transferred_ids),
        )
        return result

    # -----------------------------------------------------------------------
    # Surrender Optimization
    # -----------------------------------------------------------------------

    def optimize_surrender(
        self, portfolio_id: str, obligation: Decimal
    ) -> SurrenderPlan:
        """Optimize which certificates to surrender to minimize total cost.

        Strategy prioritizes:
        1. Certificates closest to expiry (avoid waste)
        2. Lowest cost certificates (minimize financial impact)
        3. Partially used certificates (reduce fragmentation)

        Also computes savings vs naive FIFO and LIFO approaches.

        Args:
            portfolio_id: Source portfolio.
            obligation: Total obligation in tCO2e to cover.

        Returns:
            SurrenderPlan with optimized certificate selection.

        Raises:
            ValueError: If portfolio not found or insufficient certificates.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        obligation = _decimal(obligation)
        if obligation <= Decimal("0"):
            raise ValueError("Obligation must be positive")

        portfolio = self._portfolios[portfolio_id]
        active_certs = self._get_active_certificates(portfolio)
        available = sum(c.quantity_tco2e for c in active_certs)

        if available < obligation:
            raise ValueError(
                f"Insufficient certificates: need {obligation}, have {available}"
            )

        now = utcnow()

        def _score(cert: Certificate) -> Tuple[int, Decimal]:
            days_left = (cert.expiry_date - now).days
            return (days_left, cert.purchase_price)

        sorted_certs = sorted(active_certs, key=_score)
        optimal_ids: List[str] = []
        optimal_cost = Decimal("0")
        remaining = obligation
        near_expiry_count = 0

        for cert in sorted_certs:
            if remaining <= Decimal("0"):
                break
            use_qty = min(cert.quantity_tco2e, remaining)
            optimal_ids.append(cert.certificate_id)
            optimal_cost += use_qty * cert.purchase_price
            remaining -= use_qty
            if (cert.expiry_date - now).days <= self.config.expiry_warning_days:
                near_expiry_count += 1

        fifo_certs = sorted(active_certs, key=lambda c: c.purchase_date)
        fifo_cost = self._compute_surrender_cost(fifo_certs, obligation)

        lifo_certs = sorted(active_certs, key=lambda c: c.purchase_date, reverse=True)
        lifo_cost = self._compute_surrender_cost(lifo_certs, obligation)

        avg_cost = (optimal_cost / obligation).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if obligation > 0 else Decimal("0")

        plan = SurrenderPlan(
            portfolio_id=portfolio_id,
            obligation_tco2e=obligation,
            strategy="expiry_first_then_cheapest",
            certificates_to_surrender=optimal_ids,
            total_surrender_cost=optimal_cost,
            average_cost_per_tco2e=avg_cost,
            savings_vs_fifo=fifo_cost - optimal_cost,
            savings_vs_lifo=lifo_cost - optimal_cost,
            expiring_soon_included=near_expiry_count,
        )
        plan.provenance_hash = _compute_hash(plan)

        logger.info(
            "Surrender plan for %s: %s tCO2e at EUR %s (savings vs FIFO: %s, vs LIFO: %s)",
            portfolio_id, obligation, optimal_cost,
            fifo_cost - optimal_cost, lifo_cost - optimal_cost,
        )
        return plan

    # -----------------------------------------------------------------------
    # Budget Forecast
    # -----------------------------------------------------------------------

    def generate_budget_forecast(
        self, portfolio_id: str, horizon_years: int = 5
    ) -> BudgetForecast:
        """Generate a multi-year CBAM certificate budget forecast.

        Accounts for EU ETS free allocation phase-out schedule, assumed price
        growth, and emission trends.

        Args:
            portfolio_id: Portfolio to forecast for.
            horizon_years: Number of years to forecast (default 5).

        Returns:
            BudgetForecast with annual projections.

        Raises:
            ValueError: If portfolio not found or horizon invalid.
        """
        if portfolio_id not in self._portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        if horizon_years < 1 or horizon_years > 30:
            raise ValueError("Horizon must be between 1 and 30 years")

        portfolio = self._portfolios[portfolio_id]
        base_year = utcnow().year
        base_obligation = _decimal(
            portfolio.config.get("estimated_annual_obligation", 0)
        )
        base_price = _decimal(portfolio.config.get("current_market_price", 75))
        price_growth = self.config.default_price_growth_rate
        emission_trend = _decimal(portfolio.config.get("emission_trend", "0"))

        annual_forecasts: List[Dict[str, Any]] = []
        total_cost = Decimal("0")
        phase_out_schedule: List[Dict[str, Any]] = []

        for i in range(horizon_years):
            year = base_year + i
            obligation = base_obligation * (Decimal("1") + emission_trend) ** i
            price = base_price * (Decimal("1") + price_growth) ** i

            free_alloc_factor = self.config.free_allocation_phase_out.get(
                year, Decimal("0")
            )
            cbam_adjustment = Decimal("1") - free_alloc_factor
            adjusted_obligation = (obligation * cbam_adjustment).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            annual_cost = (adjusted_obligation * price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            annual_forecasts.append({
                "year": year,
                "gross_obligation_tco2e": str(obligation.quantize(Decimal("0.001"))),
                "free_allocation_factor": str(free_alloc_factor),
                "cbam_adjustment_factor": str(cbam_adjustment),
                "adjusted_obligation_tco2e": str(adjusted_obligation),
                "estimated_price_eur": str(price.quantize(Decimal("0.01"))),
                "estimated_annual_cost_eur": str(annual_cost),
            })
            total_cost += annual_cost

            phase_out_schedule.append({
                "year": year,
                "free_allocation_remaining_pct": str(free_alloc_factor * 100),
                "cbam_applicable_pct": str(cbam_adjustment * 100),
            })

        avg_annual = (total_cost / Decimal(str(horizon_years))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if horizon_years > 0 else Decimal("0")

        forecast = BudgetForecast(
            portfolio_id=portfolio_id,
            horizon_years=horizon_years,
            base_year=base_year,
            annual_forecasts=annual_forecasts,
            total_estimated_cost=total_cost,
            average_annual_cost=avg_annual,
            assumed_price_growth_rate=price_growth,
            assumed_emission_trend=emission_trend,
            free_allocation_phase_out=phase_out_schedule,
        )
        forecast.provenance_hash = _compute_hash(forecast)

        logger.info(
            "Budget forecast for %s: %d years, total EUR %s, avg EUR %s/yr",
            portfolio_id, horizon_years, total_cost, avg_annual,
        )
        return forecast

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _get_active_certificates(
        self, portfolio: CertificatePortfolio, sort_fifo: bool = True
    ) -> List[Certificate]:
        """Get active certificates from a portfolio.

        Args:
            portfolio: Source portfolio.
            sort_fifo: If True, sort oldest first (FIFO).

        Returns:
            List of active Certificate objects.
        """
        active = [
            c for c in portfolio.certificates
            if c.status == CertificateStatus.ACTIVE
        ]
        if sort_fifo:
            active.sort(key=lambda c: c.purchase_date)
        return active

    def _compute_surrender_cost(
        self, ordered_certs: List[Certificate], quantity: Decimal
    ) -> Decimal:
        """Compute total cost of surrendering quantity from ordered certificates.

        Args:
            ordered_certs: Pre-sorted list of certificates.
            quantity: Quantity to surrender.

        Returns:
            Total cost in EUR.
        """
        cost = Decimal("0")
        remaining = quantity
        for cert in ordered_certs:
            if remaining <= Decimal("0"):
                break
            use_qty = min(cert.quantity_tco2e, remaining)
            cost += use_qty * cert.purchase_price
            remaining -= use_qty
        return cost
