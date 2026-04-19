# -*- coding: utf-8 -*-
"""GreenLang commercial infrastructure — tier enforcement, billing, usage metering.

Module layout:
- tiers.py: Tier enum + TierSpec with rate limits and feature flags
- enforcement.py: Middleware for FastAPI tier gating
- metering.py: Usage event ingestion (API calls, compute minutes, storage GB)
- billing/stripe_webhook.py: Stripe subscription event handlers (skeleton)

FY27 commercial model:
- Community: free, 100 req/day, community-tier factors only
- Pro: $99/mo + usage, 10K req/day, preview factors
- Enterprise: custom, unlimited, connector factors + SLA
"""

from greenlang.commercial.tiers import Tier, TierSpec, TIER_SPECS, feature_allowed, get_spec  # noqa: F401

__all__ = ["Tier", "TierSpec", "TIER_SPECS", "feature_allowed", "get_spec"]
