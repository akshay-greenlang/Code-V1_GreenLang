"""
ValueChain Intake Agent - Main Package

Multi-format data ingestion agent for Scope 3 value chain data.

Key Components:
- Multi-format parsers (CSV, JSON, Excel, XML, PDF)
- Entity resolution with confidence scoring
- Human review queue management
- Data quality assessment (DQI integration)
- ERP connector stubs (SAP, Oracle, Workday)
- Gap analysis reporting

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from .agent import ValueChainIntakeAgent
from .models import (
    IngestionRecord,
    ResolvedEntity,
    ReviewQueueItem,
    DataQualityAssessment,
    IngestionResult,
)
from .config import get_config, IntakeAgentConfig
from .exceptions import IntakeAgentError

__version__ = "1.0.0"

__all__ = [
    "ValueChainIntakeAgent",
    "IngestionRecord",
    "ResolvedEntity",
    "ReviewQueueItem",
    "DataQualityAssessment",
    "IngestionResult",
    "get_config",
    "IntakeAgentConfig",
    "IntakeAgentError",
]
