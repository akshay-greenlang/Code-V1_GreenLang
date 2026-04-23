# -*- coding: utf-8 -*-
"""
Carbon Trading and Offsets Integration Module for GL-010 EmissionsGuardian

This module provides comprehensive carbon trading and offset certificate
management capabilities including:
    - Market data integration (ICE, CME, CBL)
    - Position tracking and mark-to-market calculations
    - Rule-based trading recommendations (zero-hallucination)
    - Offset certificate lifecycle management
    - Risk management and controls

Zero-Hallucination Principle:
    - All trading calculations use deterministic formulas
    - No LLM calls in trading/calculation logic
    - LLM usage limited to: classification, narrative generation
    - Complete provenance tracking with SHA-256 hashes
    - Human approval required for all trade executions
"""

from trading.schemas import (
    # Enums
    OrderType,
    OrderStatus,
    CarbonMarket,
    OffsetStandard,
    OffsetProjectType,
    RetirementStatus,
    RecommendationAction,
    Urgency,
    Currency,
    # Core Models
    CarbonPosition,
    TradeOrder,
    TradeExecution,
    OffsetCertificate,
    MarketPrice,
    TradingRecommendation,
    # Result Models
    MTMResult,
    LimitBreach,
    RetirementResult,
    VerificationResult,
    PositionAnalysis,
    RiskCheckResult,
    VaRResult,
    ExposureResult,
    StopLossAction,
    DailyRiskReport,
)

from trading.market_data import (
    MarketDataProvider,
    ICEMarketProvider,
    CMEMarketProvider,
    CBLMarketProvider,
    MarketDataAggregator,
)

from trading.position_manager import (
    PositionManager,
    PositionHistory,
)

from trading.recommendation_engine import (
    TradingRecommendationEngine,
)

from trading.offset_tracker import (
    OffsetTracker,
    RetirementWorkflow,
)

from trading.risk_manager import (
    RiskManager,
)

__all__ = [
    # Enums
    "OrderType",
    "OrderStatus",
    "CarbonMarket",
    "OffsetStandard",
    "OffsetProjectType",
    "RetirementStatus",
    "RecommendationAction",
    "Urgency",
    "Currency",
    # Core Models
    "CarbonPosition",
    "TradeOrder",
    "TradeExecution",
    "OffsetCertificate",
    "MarketPrice",
    "TradingRecommendation",
    # Result Models
    "MTMResult",
    "LimitBreach",
    "RetirementResult",
    "VerificationResult",
    "PositionAnalysis",
    "RiskCheckResult",
    "VaRResult",
    "ExposureResult",
    "StopLossAction",
    "DailyRiskReport",
    # Providers
    "MarketDataProvider",
    "ICEMarketProvider",
    "CMEMarketProvider",
    "CBLMarketProvider",
    "MarketDataAggregator",
    # Managers
    "PositionManager",
    "PositionHistory",
    "TradingRecommendationEngine",
    "OffsetTracker",
    "RetirementWorkflow",
    "RiskManager",
]

__version__ = "1.0.0"
