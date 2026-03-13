# -*- coding: utf-8 -*-
"""
AGENT-EUDR-020: Deforestation Alert System - Advanced Engines Package

Sub-engines for cutoff verification, historical baselines, alert workflow,
and compliance impact assessment. These four engines complement the core
four engines (SatelliteChangeDetector, AlertGenerator, SeverityClassifier,
SpatialBufferMonitor) to form the complete eight-engine deforestation
alert system for EUDR compliance.

Engines:
    5. CutoffDateVerifier       - EUDR cutoff date (2020-12-31) verification
    6. HistoricalBaselineEngine  - 2018-2020 forest cover baseline management
    7. AlertWorkflowEngine       - Full alert lifecycle workflow management
    8. ComplianceImpactAssessor  - Supply chain compliance impact assessment

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.agents.eudr.deforestation_alert_system.engines.cutoff_date_verifier import (
    CutoffDateVerifier,
)
from greenlang.agents.eudr.deforestation_alert_system.engines.historical_baseline_engine import (
    HistoricalBaselineEngine,
)
from greenlang.agents.eudr.deforestation_alert_system.engines.alert_workflow_engine import (
    AlertWorkflowEngine,
)
from greenlang.agents.eudr.deforestation_alert_system.engines.compliance_impact_assessor import (
    ComplianceImpactAssessor,
)

__all__ = [
    "CutoffDateVerifier",
    "HistoricalBaselineEngine",
    "AlertWorkflowEngine",
    "ComplianceImpactAssessor",
]
