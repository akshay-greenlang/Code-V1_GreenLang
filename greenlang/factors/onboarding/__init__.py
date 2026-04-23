# -*- coding: utf-8 -*-
"""Design partner onboarding kit for GreenLang Factors API.

Provides automated partner environment setup, health checking, and
pre-built sample queries for common personas.
"""

from greenlang.factors.onboarding.partner_setup import (
    PartnerConfig,
    create_partner_environment,
    # Track C-5: OEM lifecycle
    OemError,
    OemPartner,
    SubTenant,
    RedistributionGrant,
    OEM_GRANT_CLASSES,
    OEM_ELIGIBLE_PARENT_PLANS,
    create_oem_partner,
    provision_subtenant,
    update_branding,
    revoke_subtenant,
    list_subtenants,
    list_oem_partners,
    get_oem_partner,
    get_redistribution_grant,
)
from greenlang.factors.onboarding.branding_config import BrandingConfig
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
    # Track C-5: OEM lifecycle
    "BrandingConfig",
    "OemError",
    "OemPartner",
    "SubTenant",
    "RedistributionGrant",
    "OEM_GRANT_CLASSES",
    "OEM_ELIGIBLE_PARENT_PLANS",
    "create_oem_partner",
    "provision_subtenant",
    "update_branding",
    "revoke_subtenant",
    "list_subtenants",
    "list_oem_partners",
    "get_oem_partner",
    "get_redistribution_grant",
]
