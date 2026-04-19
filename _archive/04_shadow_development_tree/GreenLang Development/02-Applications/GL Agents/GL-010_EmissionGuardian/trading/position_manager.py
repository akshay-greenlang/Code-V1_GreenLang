# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Position Manager

Carbon position tracking and mark-to-market calculations.

Author: GreenLang GL-010 EmissionsGuardian
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging

from .schemas import (
    CarbonMarket, CarbonPosition, MarketPrice,
    MTMResult, PositionAnalysis
)
from .market_data import MarketDataAggregator

logger = logging.getLogger(__name__)


class PositionHistory:
    """Historical position tracking."""

    def __init__(self):
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    def record(self, position: CarbonPosition, event: str) -> None:
        """Record position change."""
        if position.position_id not in self._history:
            self._history[position.position_id] = []

        self._history[position.position_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "quantity": str(position.quantity),
            "market": position.market.value,
        })

    def get_history(self, position_id: str) -> List[Dict[str, Any]]:
        """Get position history."""
        return self._history.get(position_id, [])


class PositionManager:
    """
    Carbon Position Manager.

    Tracks carbon positions with mark-to-market valuations.
    """

    def __init__(
        self,
        market_data: Optional[MarketDataAggregator] = None
    ):
        self._positions: Dict[str, CarbonPosition] = {}
        self._market_data = market_data or MarketDataAggregator()
        self._history = PositionHistory()
        logger.info("PositionManager initialized")

    def add_position(self, position: CarbonPosition) -> None:
        """Add or update a position."""
        self._positions[position.position_id] = position
        self._history.record(position, "added")
        logger.info(f"Added position: {position.position_id}")

    def remove_position(self, position_id: str) -> bool:
        """Remove a position."""
        if position_id in self._positions:
            del self._positions[position_id]
            return True
        return False

    def get_position(self, position_id: str) -> Optional[CarbonPosition]:
        """Get a position by ID."""
        return self._positions.get(position_id)

    def get_positions(
        self,
        facility_id: Optional[str] = None,
        market: Optional[CarbonMarket] = None
    ) -> List[CarbonPosition]:
        """Get positions with optional filters."""
        positions = list(self._positions.values())

        if facility_id:
            positions = [p for p in positions if p.facility_id == facility_id]

        if market:
            positions = [p for p in positions if p.market == market]

        return positions

    def mark_to_market(
        self,
        position_id: str
    ) -> Optional[MTMResult]:
        """Calculate mark-to-market for a position."""
        position = self._positions.get(position_id)
        if not position:
            return None

        price = self._market_data.get_price(
            position.market,
            position.instrument,
            position.vintage
        )

        if not price:
            return None

        current_price = price.mid
        market_value = position.quantity * current_price
        cost_basis = position.quantity * position.acquisition_price
        unrealized_pnl = market_value - cost_basis
        pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else Decimal("0")

        return MTMResult(
            position_id=position_id,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            pnl_pct=pnl_pct,
            as_of=datetime.utcnow()
        )

    def analyze_portfolio(
        self,
        facility_id: str
    ) -> PositionAnalysis:
        """Analyze portfolio for a facility."""
        positions = self.get_positions(facility_id=facility_id)

        total_quantity = sum(p.quantity for p in positions)
        total_value = Decimal("0")
        total_pnl = Decimal("0")
        by_market: Dict[str, Decimal] = {}

        for position in positions:
            mtm = self.mark_to_market(position.position_id)
            if mtm:
                total_value += mtm.market_value
                total_pnl += mtm.unrealized_pnl

            market_key = position.market.value
            by_market[market_key] = by_market.get(market_key, Decimal("0")) + position.quantity

        return PositionAnalysis(
            facility_id=facility_id,
            total_positions=len(positions),
            total_quantity=total_quantity,
            total_value=total_value,
            total_unrealized_pnl=total_pnl,
            by_market=by_market
        )


__all__ = [
    "PositionManager",
    "PositionHistory",
]
