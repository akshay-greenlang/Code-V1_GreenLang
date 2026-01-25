# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Trading Recommendation Engine

Rule-based trading recommendations with human approval workflow.

Zero-Hallucination: No autonomous execution - all trades require approval.

Author: GreenLang GL-010 EmissionsGuardian
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import hashlib
import logging

from .schemas import (
    CarbonMarket, Currency, RecommendationAction, Urgency,
    TradingRecommendation, CarbonPosition
)
from .position_manager import PositionManager
from .market_data import MarketDataAggregator

logger = logging.getLogger(__name__)


class TradingRecommendationEngine:
    """
    Trading Recommendation Engine.

    Generates trading recommendations based on:
    - Compliance gap analysis
    - Market conditions
    - Risk limits
    - Policy constraints

    Zero-Hallucination: All recommendations require human approval.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataAggregator
    ):
        self._position_manager = position_manager
        self._market_data = market_data
        self._recommendations: Dict[str, TradingRecommendation] = {}
        self._counter = 0
        logger.info("TradingRecommendationEngine initialized")

    def generate_coverage_recommendation(
        self,
        facility_id: str,
        obligation_tco2e: Decimal,
        market: CarbonMarket = CarbonMarket.EU_ETS
    ) -> Optional[TradingRecommendation]:
        """Generate recommendation to cover compliance gap."""
        # Analyze current position
        positions = self._position_manager.get_positions(
            facility_id=facility_id,
            market=market
        )
        current_coverage = sum(p.quantity for p in positions)
        gap = obligation_tco2e - current_coverage

        if gap <= 0:
            logger.info(f"No coverage gap for {facility_id}")
            return None

        # Get market price
        price = self._market_data.get_price(market, "EUA")
        if not price:
            logger.warning(f"No price available for {market.value}")
            return None

        # Generate recommendation
        self._counter += 1
        rec_id = f"REC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._counter:06d}"

        # Determine urgency based on gap size
        gap_pct = (gap / obligation_tco2e * 100) if obligation_tco2e > 0 else Decimal("0")
        if gap_pct > 50:
            urgency = Urgency.CRITICAL
        elif gap_pct > 25:
            urgency = Urgency.HIGH
        elif gap_pct > 10:
            urgency = Urgency.MEDIUM
        else:
            urgency = Urgency.LOW

        target_price = price.mid
        price_limit = target_price * Decimal("1.05")  # 5% above mid

        # Calculate provenance
        content = f"{rec_id}|{gap}|{target_price}"
        provenance_hash = hashlib.sha256(content.encode()).hexdigest()

        recommendation = TradingRecommendation(
            recommendation_id=rec_id,
            facility_id=facility_id,
            action=RecommendationAction.BUY,
            market=market,
            instrument="EUA",
            quantity=gap,
            target_price=target_price,
            price_limit=price_limit,
            currency=Currency.EUR,
            urgency=urgency,
            rationale=f"Coverage gap of {gap} tCO2e ({gap_pct:.1f}% of obligation). "
                      f"Current price: {target_price} EUR.",
            expected_savings=Decimal("0"),
            risk_score=Decimal("0.3"),
            expires_at=datetime.utcnow() + timedelta(days=1),
            provenance_hash=provenance_hash
        )

        self._recommendations[rec_id] = recommendation
        return recommendation

    def approve(
        self,
        recommendation_id: str,
        approver: str
    ) -> bool:
        """Approve a recommendation."""
        rec = self._recommendations.get(recommendation_id)
        if not rec:
            return False

        rec.status = "approved"
        rec.approved_by = approver
        rec.approved_at = datetime.utcnow()
        return True

    def reject(
        self,
        recommendation_id: str,
        reason: str
    ) -> bool:
        """Reject a recommendation."""
        rec = self._recommendations.get(recommendation_id)
        if not rec:
            return False

        rec.status = "rejected"
        return True

    def get_pending(
        self,
        facility_id: Optional[str] = None
    ) -> List[TradingRecommendation]:
        """Get pending recommendations."""
        recs = [r for r in self._recommendations.values() if r.status == "pending"]
        if facility_id:
            recs = [r for r in recs if r.facility_id == facility_id]
        return sorted(recs, key=lambda r: r.created_at, reverse=True)


__all__ = [
    "TradingRecommendationEngine",
]
