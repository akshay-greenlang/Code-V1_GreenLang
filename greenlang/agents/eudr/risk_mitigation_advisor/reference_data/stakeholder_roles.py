# -*- coding: utf-8 -*-
"""
Stakeholder Role Definitions - AGENT-EUDR-025

6 stakeholder types with differentiated access levels for the
multi-party collaboration platform.

Roles:
    1. Internal Compliance Team - Full access
    2. Procurement Team - Full read, limited write
    3. Supplier - Own plan access only
    4. NGO Partner - Landscape-level aggregates
    5. Certification Body - Scheme-related access
    6. Competent Authority - Read-only compliance docs

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List

STAKEHOLDER_ROLES: List[Dict[str, Any]] = [
    {
        "role_id": "internal_compliance",
        "name": "Internal Compliance Team",
        "description": "Full access to all plans, risk scores, and analytics",
        "access_level": "full",
    },
    {
        "role_id": "procurement",
        "name": "Procurement Team",
        "description": "Full read access with limited plan editing",
        "access_level": "full_read",
    },
    {
        "role_id": "supplier",
        "name": "Supplier",
        "description": "Access to own plan, progress reporting, evidence upload",
        "access_level": "own_only",
    },
    {
        "role_id": "ngo_partner",
        "name": "NGO Partner",
        "description": "Landscape-level aggregates and joint plan participation",
        "access_level": "landscape",
    },
    {
        "role_id": "certification_body",
        "name": "Certification Body",
        "description": "Scheme-related plans and audit result submission",
        "access_level": "scheme",
    },
    {
        "role_id": "competent_authority",
        "name": "Competent Authority",
        "description": "Read-only access to compliance documentation",
        "access_level": "read_only",
    },
]


def get_role_count() -> int:
    """Return the number of stakeholder roles."""
    return len(STAKEHOLDER_ROLES)


def get_all_roles() -> List[Dict[str, Any]]:
    """Return all stakeholder role definitions."""
    return list(STAKEHOLDER_ROLES)
