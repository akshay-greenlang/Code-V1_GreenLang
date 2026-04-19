# -*- coding: utf-8 -*-
"""Design partner onboarding kit for GreenLang Factors API.

Provides automated partner environment setup, health checking, and
pre-built sample queries for common personas.
"""

from greenlang.factors.onboarding.partner_setup import (
    PartnerConfig,
    create_partner_environment,
)
from greenlang.factors.onboarding.health_check import (
    HealthReport,
    HealthCheckResult,
    run_partner_health_check,
)
from greenlang.factors.onboarding.sample_queries import (
    SampleQuery,
    get_sample_queries,
    get_queries_for_persona,
)

__all__ = [
    "PartnerConfig",
    "create_partner_environment",
    "HealthReport",
    "HealthCheckResult",
    "run_partner_health_check",
    "SampleQuery",
    "get_sample_queries",
    "get_queries_for_persona",
]
