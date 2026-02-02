# -*- coding: utf-8 -*-
"""
SupplierEngagementAgent v1 - Consent-aware supplier engagement and data collection.

Phase 3, Weeks 16-18 deliverable for GL-VCCI Scope 3 Platform.
"""
from .agent import SupplierEngagementAgent
from .models import (
    Campaign,
    CampaignStatus,
    ConsentRecord,
    ConsentStatus,
    EmailSequence,
    EmailTemplate,
    ValidationResult,
    SupplierProgress,
    Leaderboard
)


__version__ = "1.0.0"

__all__ = [
    "SupplierEngagementAgent",
    "Campaign",
    "CampaignStatus",
    "ConsentRecord",
    "ConsentStatus",
    "EmailSequence",
    "EmailTemplate",
    "ValidationResult",
    "SupplierProgress",
    "Leaderboard",
]
