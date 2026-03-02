# -*- coding: utf-8 -*-
"""
GL-CBAM-APP De Minimis Exemption Engine v1.1

Implements the 50-tonne annual threshold monitoring system per the CBAM Omnibus
Simplification Package (October 2025). Importers whose total CBAM-goods imports
remain below 50 metric tonnes per calendar year are exempt from full CBAM
reporting obligations.

Key regulatory references:
    - Regulation (EU) 2023/956 (CBAM Regulation), Article 2(3a)
    - Omnibus Simplification Package COM(2025) 508 final, Article 1(3)
    - Delegated Regulation (EU) 2025/XXX (De Minimis Implementing Rules)

Design principles:
    - Thread-safe singleton for threshold monitoring across concurrent requests
    - Electricity (CN 2716) and hydrogen (CN 2804) excluded from threshold
    - SHA-256 provenance on every state mutation for audit trail
    - Decimal arithmetic throughout to avoid floating-point drift
    - Alert escalation at 80%, 90%, 95%, and 100% of threshold

Modules:
    threshold_monitor: Real-time cumulative import tracking and forecasting
    exemption_manager: Exemption determination, certificate issuance, revocation

Example:
    >>> from deminimis_engine import ThresholdMonitorEngine, ExemptionManagerEngine
    >>> monitor = ThresholdMonitorEngine.get_instance()
    >>> status = monitor.add_import("IMP-001", 2026, "72011000", Decimal("12.5"))
    >>> print(status.percentage)  # 25.0
    >>> mgr = ExemptionManagerEngine(monitor)
    >>> result = mgr.determine_exemption("IMP-001", 2026)
    >>> print(result.status)  # "exempt"

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

__version__ = "1.1.0"
__author__ = "GreenLang CBAM Team"

from deminimis_engine.threshold_monitor import (
    ThresholdMonitorEngine,
    ThresholdStatus,
    ImportRecord,
    ThresholdAlert,
    SectorBreakdown,
)
from deminimis_engine.exemption_manager import (
    ExemptionManagerEngine,
    ExemptionResult,
    ExemptionCertificate,
    SMESimplifiedPath,
)

__all__ = [
    # Threshold monitoring
    "ThresholdMonitorEngine",
    "ThresholdStatus",
    "ImportRecord",
    "ThresholdAlert",
    "SectorBreakdown",
    # Exemption management
    "ExemptionManagerEngine",
    "ExemptionResult",
    "ExemptionCertificate",
    "SMESimplifiedPath",
]

# CBAM De Minimis constants (from Omnibus Simplification)
DE_MINIMIS_THRESHOLD_MT = 50
EXCLUDED_SECTORS = frozenset({"electricity", "hydrogen"})
ALERT_THRESHOLDS_PCT = (80, 90, 95, 100)
