# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Risk Manager

Carbon trading risk management and controls.

Author: GreenLang GL-010 EmissionsGuardian
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging
import math

from .schemas import (
    CarbonPosition, MarketPrice, Urgency,
    LimitBreach, RiskCheckResult, VaRResult,
    ExposureResult, StopLossAction, DailyRiskReport
)
from .position_manager import PositionManager
from .market_data import MarketDataAggregator

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Carbon Trading Risk Manager.

    Implements risk controls including:
    - Position limits
    - Value at Risk (VaR)
    - Stop-loss monitoring
    - Exposure analysis
    """

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataAggregator,
        max_position_value: Decimal = Decimal("10000000"),
        max_single_trade: Decimal = Decimal("1000000"),
        stop_loss_pct: Decimal = Decimal("10")
    ):
        self._position_manager = position_manager
        self._market_data = market_data
        self._max_position_value = max_position_value
        self._max_single_trade = max_single_trade
        self._stop_loss_pct = stop_loss_pct
        self._breach_counter = 0
        logger.info("RiskManager initialized")

    def check_limits(
        self,
        facility_id: str
    ) -> RiskCheckResult:
        """Check all risk limits."""
        breaches: List[LimitBreach] = []
        check_id = f"CHK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Get portfolio analysis
        portfolio = self._position_manager.analyze_portfolio(facility_id)

        # Check position value limit
        if portfolio.total_value > self._max_position_value:
            self._breach_counter += 1
            breaches.append(LimitBreach(
                breach_id=f"BRC-{self._breach_counter:06d}",
                limit_type="max_position_value",
                limit_value=self._max_position_value,
                actual_value=portfolio.total_value,
                breach_pct=((portfolio.total_value / self._max_position_value) - 1) * 100,
                severity=Urgency.HIGH,
                detected_at=datetime.utcnow()
            ))

        return RiskCheckResult(
            check_id=check_id,
            passed=len(breaches) == 0,
            breaches=breaches,
            checked_at=datetime.utcnow()
        )

    def calculate_var(
        self,
        facility_id: str,
        confidence_95: bool = True,
        confidence_99: bool = True,
        horizon_days: int = 1
    ) -> VaRResult:
        """
        Calculate Value at Risk.

        Uses parametric VaR with assumed volatility.
        """
        portfolio = self._position_manager.analyze_portfolio(facility_id)

        # Assumed daily volatility (would be calculated from historical data)
        daily_vol = Decimal("0.02")  # 2% daily volatility

        # Z-scores for confidence levels
        z_95 = Decimal("1.645")
        z_99 = Decimal("2.326")

        # Scale for horizon
        horizon_factor = Decimal(str(math.sqrt(horizon_days)))

        var_1d_95 = portfolio.total_value * daily_vol * z_95
        var_1d_99 = portfolio.total_value * daily_vol * z_99
        var_10d_95 = portfolio.total_value * daily_vol * z_95 * Decimal(str(math.sqrt(10)))

        return VaRResult(
            facility_id=facility_id,
            var_1d_95=var_1d_95.quantize(Decimal("0.01")),
            var_1d_99=var_1d_99.quantize(Decimal("0.01")),
            var_10d_95=var_10d_95.quantize(Decimal("0.01")),
            calculated_at=datetime.utcnow()
        )

    def analyze_exposure(
        self,
        facility_id: str
    ) -> ExposureResult:
        """Analyze market exposure."""
        portfolio = self._position_manager.analyze_portfolio(facility_id)

        return ExposureResult(
            facility_id=facility_id,
            gross_exposure=abs(portfolio.total_value),
            net_exposure=portfolio.total_value,
            by_market={k: abs(v) for k, v in portfolio.by_market.items()}
        )

    def check_stop_loss(
        self,
        facility_id: str
    ) -> List[StopLossAction]:
        """Check for stop-loss triggers."""
        actions: List[StopLossAction] = []
        positions = self._position_manager.get_positions(facility_id=facility_id)

        for position in positions:
            mtm = self._position_manager.mark_to_market(position.position_id)
            if not mtm:
                continue

            # Check if loss exceeds threshold
            if mtm.pnl_pct < -self._stop_loss_pct:
                price = self._market_data.get_price(
                    position.market,
                    position.instrument,
                    position.vintage
                )
                current_price = price.mid if price else Decimal("0")

                actions.append(StopLossAction(
                    position_id=position.position_id,
                    action="SELL",
                    trigger_price=position.acquisition_price * (1 - self._stop_loss_pct / 100),
                    current_price=current_price,
                    loss_pct=abs(mtm.pnl_pct)
                ))

        return actions

    def generate_daily_report(
        self,
        facility_id: str
    ) -> DailyRiskReport:
        """Generate daily risk report."""
        var_result = self.calculate_var(facility_id)
        exposure_result = self.analyze_exposure(facility_id)
        limit_check = self.check_limits(facility_id)
        stop_losses = self.check_stop_loss(facility_id)

        return DailyRiskReport(
            report_date=date.today(),
            facility_id=facility_id,
            var_result=var_result,
            exposure_result=exposure_result,
            breaches=limit_check.breaches,
            stop_loss_actions=stop_losses
        )


__all__ = [
    "RiskManager",
]
