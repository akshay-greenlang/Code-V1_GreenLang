# -*- coding: utf-8 -*-
"""
AGENT-EUDR-019: Corruption Index Monitor - Advanced Engines Package

Sub-engines for trend analysis, deforestation correlation, alerts, and
compliance impact assessment. These four engines extend the core CPI/WGI/
Bribery/Institutional engines with advanced analytical capabilities:

    Engine 5 - TrendAnalysisEngine:
        Temporal analysis of corruption indices to identify improving/
        deteriorating trends, predict future trajectories, detect structural
        breakpoints, and provide early warning of governance deterioration.

    Engine 6 - DeforestationCorrelationEngine:
        Statistical correlation between corruption levels and deforestation
        rates using Pearson/Spearman correlation, regression models, and
        causal pathway analysis for EUDR risk assessment.

    Engine 7 - AlertEngine:
        Real-time alerting for corruption index changes, threshold breaches,
        trend reversals, and country reclassification events with
        configurable severity levels and notification workflows.

    Engine 8 - ComplianceImpactEngine:
        Evaluates the impact of corruption levels on EUDR compliance
        obligations, determines due diligence requirements, and maps to
        EUDR Article 29 country benchmarking/classification.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.agents.eudr.corruption_index_monitor.engines.trend_analysis_engine import (
    TrendAnalysisEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.engines.deforestation_correlation_engine import (
    DeforestationCorrelationEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.engines.alert_engine import (
    AlertEngine,
)
from greenlang.agents.eudr.corruption_index_monitor.engines.compliance_impact_engine import (
    ComplianceImpactEngine,
)

__all__: list[str] = [
    "TrendAnalysisEngine",
    "DeforestationCorrelationEngine",
    "AlertEngine",
    "ComplianceImpactEngine",
]
