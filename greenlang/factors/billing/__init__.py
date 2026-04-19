# -*- coding: utf-8 -*-
"""
Billing integration for the GreenLang Factors API.

Provides usage aggregation, quota checking, and Stripe billing integration
for tiered API access (Community, Pro, Enterprise).

Modules:
    usage_sink: Durable SQLite usage event recording
    aggregator: Usage aggregation and quota management
    stripe_provider: Stripe billing provider (abstract + implementation)
    webhook_handler: Stripe webhook FastAPI router
"""

from greenlang.factors.billing.usage_sink import record_path_hit

__all__ = [
    "record_path_hit",
]
