# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer Engines - AGENT-EUDR-018

Eight specialized engines for EUDR commodity risk analysis:

    1. CommodityProfiler       - Base commodity risk profiling
    2. DerivedProductAnalyzer   - Annex I derived product traceability
    3. PriceVolatilityEngine    - Price volatility monitoring
    4. ProductionForecastEngine - Yield and production forecasting
    5. SubstitutionRiskAnalyzer - Commodity substitution fraud detection
    6. RegulatoryComplianceEngine - EUDR article-specific compliance
    7. CommodityDueDiligenceEngine - Commodity DD workflows
    8. PortfolioRiskAggregator  - Cross-commodity portfolio analysis

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

__all__: list[str] = [
    "SubstitutionRiskAnalyzer",
    "RegulatoryComplianceEngine",
    "CommodityDueDiligenceEngine",
    "PortfolioRiskAggregator",
]
