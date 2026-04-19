# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Carbon Trading Schemas

Pydantic models and enums for carbon trading, positions, and offsets.

Author: GreenLang GL-010 EmissionsGuardian
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib

from pydantic import BaseModel, Field


class OrderType(str, Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class CarbonMarket(str, Enum):
    """Carbon markets."""
    EU_ETS = "eu_ets"
    UK_ETS = "uk_ets"
    RGGI = "rggi"
    WCI = "wci"
    CARB = "carb"
    KR_ETS = "kr_ets"
    CN_ETS = "cn_ets"
    VOLUNTARY = "voluntary"


class OffsetStandard(str, Enum):
    """Offset verification standards."""
    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CORSIA = "corsia"
    CDM = "cdm"
    INTERNAL = "internal"


class OffsetProjectType(str, Enum):
    """Offset project types."""
    FORESTRY = "forestry"
    RENEWABLE_ENERGY = "renewable_energy"
    METHANE_CAPTURE = "methane_capture"
    COOKSTOVES = "cookstoves"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BLUE_CARBON = "blue_carbon"
    SOIL_CARBON = "soil_carbon"
    OTHER = "other"


class RetirementStatus(str, Enum):
    """Offset retirement status."""
    ACTIVE = "active"
    PENDING_RETIREMENT = "pending_retirement"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RecommendationAction(str, Enum):
    """Trading recommendation actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SURRENDER = "surrender"
    RETIRE = "retire"


class Urgency(str, Enum):
    """Urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Currency(str, Enum):
    """Currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    KRW = "KRW"
    CNY = "CNY"


class MarketPrice(BaseModel):
    """Market price data."""
    market: CarbonMarket = Field(...)
    instrument: str = Field(...)
    vintage: Optional[int] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bid: Decimal = Field(..., ge=0)
    ask: Decimal = Field(..., ge=0)
    last: Decimal = Field(..., ge=0)
    volume: Decimal = Field(default=Decimal("0"), ge=0)
    currency: Currency = Field(default=Currency.USD)
    source: str = Field(default="market_data")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


class CarbonPosition(BaseModel):
    """Carbon position."""
    position_id: str = Field(...)
    facility_id: str = Field(...)
    market: CarbonMarket = Field(...)
    instrument: str = Field(...)
    vintage: Optional[int] = Field(None)
    quantity: Decimal = Field(...)
    unit: str = Field(default="tCO2e")
    acquisition_date: date = Field(...)
    acquisition_price: Decimal = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    serial_numbers: List[str] = Field(default_factory=list)
    is_encumbered: bool = Field(default=False)
    expiry_date: Optional[date] = Field(None)

    def current_value(self, price: Decimal) -> Decimal:
        return self.quantity * price

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        return (price - self.acquisition_price) * self.quantity


class TradeOrder(BaseModel):
    """Trade order."""
    order_id: str = Field(...)
    facility_id: str = Field(...)
    market: CarbonMarket = Field(...)
    instrument: str = Field(...)
    vintage: Optional[int] = Field(None)
    order_type: OrderType = Field(...)
    action: RecommendationAction = Field(...)
    quantity: Decimal = Field(..., gt=0)
    limit_price: Optional[Decimal] = Field(None, ge=0)
    stop_price: Optional[Decimal] = Field(None, ge=0)
    currency: Currency = Field(default=Currency.USD)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None)
    requires_approval: bool = Field(default=True)
    approved_by: Optional[str] = Field(None)
    approved_at: Optional[datetime] = Field(None)


class TradeExecution(BaseModel):
    """Trade execution record."""
    execution_id: str = Field(...)
    order_id: str = Field(...)
    market: CarbonMarket = Field(...)
    instrument: str = Field(...)
    vintage: Optional[int] = Field(None)
    action: RecommendationAction = Field(...)
    quantity: Decimal = Field(..., gt=0)
    price: Decimal = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    fees: Decimal = Field(default=Decimal("0"), ge=0)
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    counterparty: Optional[str] = Field(None)
    settlement_date: Optional[date] = Field(None)
    provenance_hash: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.provenance_hash:
            content = f"{self.execution_id}|{self.quantity}|{self.price}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


class OffsetCertificate(BaseModel):
    """Offset certificate."""
    certificate_id: str = Field(...)
    serial_number: str = Field(...)
    facility_id: str = Field(...)
    standard: OffsetStandard = Field(...)
    project_type: OffsetProjectType = Field(...)
    project_id: str = Field(...)
    project_name: str = Field(...)
    vintage: int = Field(...)
    quantity: Decimal = Field(..., gt=0)
    unit: str = Field(default="tCO2e")
    issuance_date: date = Field(...)
    acquisition_date: date = Field(...)
    acquisition_price: Decimal = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    status: RetirementStatus = Field(default=RetirementStatus.ACTIVE)
    retirement_date: Optional[date] = Field(None)
    retirement_reason: Optional[str] = Field(None)
    beneficiary: Optional[str] = Field(None)
    registry_link: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


class TradingRecommendation(BaseModel):
    """Trading recommendation."""
    recommendation_id: str = Field(...)
    facility_id: str = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    action: RecommendationAction = Field(...)
    market: CarbonMarket = Field(...)
    instrument: str = Field(...)
    vintage: Optional[int] = Field(None)
    quantity: Decimal = Field(..., gt=0)
    target_price: Decimal = Field(..., ge=0)
    price_limit: Decimal = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    urgency: Urgency = Field(default=Urgency.MEDIUM)
    rationale: str = Field(...)
    expected_savings: Decimal = Field(default=Decimal("0"))
    risk_score: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    complies_with_policy: bool = Field(default=True)
    requires_approval: bool = Field(default=True)
    status: str = Field(default="pending")
    approved_by: Optional[str] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    expires_at: datetime = Field(...)
    provenance_hash: str = Field(default="")


class MTMResult(BaseModel):
    """Mark-to-market result."""
    position_id: str
    market_value: Decimal
    unrealized_pnl: Decimal
    pnl_pct: Decimal
    as_of: datetime


class LimitBreach(BaseModel):
    """Risk limit breach."""
    breach_id: str
    limit_type: str
    limit_value: Decimal
    actual_value: Decimal
    breach_pct: Decimal
    severity: Urgency
    detected_at: datetime


class RetirementResult(BaseModel):
    """Offset retirement result."""
    certificate_id: str
    serial_number: str
    quantity_retired: Decimal
    retirement_date: date
    beneficiary: str
    registry_confirmation: str
    provenance_hash: str


class VerificationResult(BaseModel):
    """Offset verification result."""
    certificate_id: str
    is_valid: bool
    verification_date: datetime
    registry_status: str
    issues: List[str]


class PositionAnalysis(BaseModel):
    """Position analysis."""
    facility_id: str
    total_positions: int
    total_quantity: Decimal
    total_value: Decimal
    total_unrealized_pnl: Decimal
    by_market: Dict[str, Decimal]


class RiskCheckResult(BaseModel):
    """Risk check result."""
    check_id: str
    passed: bool
    breaches: List[LimitBreach]
    checked_at: datetime


class VaRResult(BaseModel):
    """Value at Risk result."""
    facility_id: str
    var_1d_95: Decimal
    var_1d_99: Decimal
    var_10d_95: Decimal
    calculated_at: datetime


class ExposureResult(BaseModel):
    """Exposure analysis result."""
    facility_id: str
    gross_exposure: Decimal
    net_exposure: Decimal
    by_market: Dict[str, Decimal]


class StopLossAction(BaseModel):
    """Stop loss action."""
    position_id: str
    action: str
    trigger_price: Decimal
    current_price: Decimal
    loss_pct: Decimal


class DailyRiskReport(BaseModel):
    """Daily risk report."""
    report_date: date
    facility_id: str
    var_result: VaRResult
    exposure_result: ExposureResult
    breaches: List[LimitBreach]
    stop_loss_actions: List[StopLossAction]


__all__ = [
    "OrderType", "OrderStatus", "CarbonMarket", "OffsetStandard",
    "OffsetProjectType", "RetirementStatus", "RecommendationAction",
    "Urgency", "Currency", "MarketPrice", "CarbonPosition",
    "TradeOrder", "TradeExecution", "OffsetCertificate",
    "TradingRecommendation", "MTMResult", "LimitBreach",
    "RetirementResult", "VerificationResult", "PositionAnalysis",
    "RiskCheckResult", "VaRResult", "ExposureResult",
    "StopLossAction", "DailyRiskReport",
]
