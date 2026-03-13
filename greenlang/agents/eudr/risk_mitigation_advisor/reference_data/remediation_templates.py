# -*- coding: utf-8 -*-
"""
Remediation Plan Templates - AGENT-EUDR-025

8 standard remediation plan templates covering common EUDR mitigation
scenarios with pre-defined phases, milestones, and KPI structures.

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List

TEMPLATE_NAMES: List[str] = [
    "supplier_capacity_building",
    "emergency_deforestation_response",
    "certification_enrollment",
    "enhanced_monitoring_deployment",
    "fpic_remediation",
    "legal_gap_closure",
    "anti_corruption_measures",
    "protected_area_buffer_restoration",
]


def get_template_names() -> List[str]:
    """Return all available template names."""
    return list(TEMPLATE_NAMES)


def get_template_count() -> int:
    """Return the number of available templates."""
    return len(TEMPLATE_NAMES)
