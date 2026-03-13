# -*- coding: utf-8 -*-
"""
Workflow Templates - AGENT-EUDR-026

Pre-built workflow DAG templates for all 7 EUDR-regulated commodities
plus standard and simplified workflow templates. Templates provide
commodity-specific agent configurations, fallback strategies, and
timeout overrides.

Commodity Templates:
    - Cattle: Enhanced supply chain traceability, pasture monitoring
    - Cocoa: Multi-tier supply chain, West Africa focus
    - Coffee: Smallholder plot verification, altitude mapping
    - Palm Oil: Plantation boundary verification, RSPO alignment
    - Rubber: Smallholder aggregation, deforestation hotspot focus
    - Soya: Large-scale farming, Cerrado/Amazon monitoring
    - Wood: Forest management plan verification, FSC/PEFC alignment

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    EUDRCommodity,
    FallbackStrategy,
    WorkflowType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commodity-specific agent configuration overrides
# ---------------------------------------------------------------------------

#: Per-commodity timeout overrides (seconds) for agents with known
#: longer processing times for certain commodities.
COMMODITY_TIMEOUTS: Dict[str, Dict[str, int]] = {
    "cattle": {
        "EUDR-003": 180,  # Satellite: larger pasture areas
        "EUDR-008": 180,  # Multi-tier: complex livestock supply chains
    },
    "cocoa": {
        "EUDR-002": 150,  # Geolocation: many smallholder plots
        "EUDR-008": 240,  # Multi-tier: deep supply chains (4+ tiers)
        "EUDR-015": 120,  # Mobile: field verification needed
    },
    "coffee": {
        "EUDR-002": 180,  # Geolocation: mountain terrain coordinates
        "EUDR-006": 150,  # Plot boundary: irregular smallholder shapes
        "EUDR-015": 120,  # Mobile: field data collection
    },
    "palm_oil": {
        "EUDR-003": 240,  # Satellite: large plantation monitoring
        "EUDR-004": 180,  # Forest cover: extensive canopy analysis
        "EUDR-005": 180,  # Land use change: historical analysis
    },
    "rubber": {
        "EUDR-002": 150,  # Geolocation: scattered smallholder plots
        "EUDR-003": 180,  # Satellite: deforestation hotspot monitoring
        "EUDR-008": 180,  # Multi-tier: aggregator supply chains
    },
    "soya": {
        "EUDR-003": 300,  # Satellite: vast farm areas
        "EUDR-004": 240,  # Forest cover: Cerrado/Amazon analysis
        "EUDR-005": 240,  # Land use change: conversion tracking
    },
    "wood": {
        "EUDR-004": 300,  # Forest cover: forest management plan verification
        "EUDR-012": 180,  # Document auth: FSC/PEFC certificates
        "EUDR-023": 180,  # Legal compliance: forestry law verification
    },
}

#: Per-commodity fallback strategy overrides for non-critical agents.
COMMODITY_FALLBACKS: Dict[str, Dict[str, FallbackStrategy]] = {
    "cattle": {
        "EUDR-014": FallbackStrategy.DEGRADED_MODE,  # QR codes optional
    },
    "cocoa": {
        "EUDR-013": FallbackStrategy.CACHED_RESULT,  # Blockchain optional
        "EUDR-014": FallbackStrategy.DEGRADED_MODE,  # QR codes optional
    },
    "coffee": {
        "EUDR-013": FallbackStrategy.CACHED_RESULT,
        "EUDR-014": FallbackStrategy.DEGRADED_MODE,
    },
    "palm_oil": {},
    "rubber": {
        "EUDR-014": FallbackStrategy.DEGRADED_MODE,
    },
    "soya": {
        "EUDR-013": FallbackStrategy.CACHED_RESULT,
    },
    "wood": {
        "EUDR-013": FallbackStrategy.CACHED_RESULT,
    },
}

#: Per-commodity critical agent designations (agents that MUST succeed).
COMMODITY_CRITICAL_AGENTS: Dict[str, List[str]] = {
    "cattle": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-008",
        "EUDR-009", "EUDR-016", "EUDR-023",
    ],
    "cocoa": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-004",
        "EUDR-008", "EUDR-016", "EUDR-023",
    ],
    "coffee": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-007",
        "EUDR-016", "EUDR-023",
    ],
    "palm_oil": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-004",
        "EUDR-005", "EUDR-006", "EUDR-016", "EUDR-020", "EUDR-023",
    ],
    "rubber": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-004",
        "EUDR-016", "EUDR-020", "EUDR-023",
    ],
    "soya": [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-004",
        "EUDR-005", "EUDR-016", "EUDR-020", "EUDR-023",
    ],
    "wood": [
        "EUDR-001", "EUDR-002", "EUDR-004", "EUDR-012",
        "EUDR-016", "EUDR-023", "EUDR-024",
    ],
}


# ---------------------------------------------------------------------------
# Template retrieval functions
# ---------------------------------------------------------------------------


def get_commodity_timeouts(
    commodity: Optional[EUDRCommodity] = None,
) -> Dict[str, int]:
    """Get agent timeout overrides for a commodity.

    Args:
        commodity: EUDR commodity or None for defaults.

    Returns:
        Dictionary mapping agent_id to timeout in seconds.
    """
    if commodity is None:
        return {}
    return dict(COMMODITY_TIMEOUTS.get(commodity.value, {}))


def get_commodity_fallbacks(
    commodity: Optional[EUDRCommodity] = None,
) -> Dict[str, FallbackStrategy]:
    """Get agent fallback strategy overrides for a commodity.

    Args:
        commodity: EUDR commodity or None for defaults.

    Returns:
        Dictionary mapping agent_id to FallbackStrategy.
    """
    if commodity is None:
        return {}
    return dict(COMMODITY_FALLBACKS.get(commodity.value, {}))


def get_critical_agents(
    commodity: Optional[EUDRCommodity] = None,
) -> List[str]:
    """Get critical agent list for a commodity.

    Args:
        commodity: EUDR commodity or None for all agents critical.

    Returns:
        List of critical agent IDs.
    """
    if commodity is None:
        from greenlang.agents.eudr.due_diligence_orchestrator.models import (
            ALL_EUDR_AGENTS,
        )
        return list(ALL_EUDR_AGENTS)
    return list(COMMODITY_CRITICAL_AGENTS.get(commodity.value, []))


def get_simplified_agents() -> List[str]:
    """Get the reduced agent set for simplified due diligence (Article 13).

    Returns:
        List of agent IDs for simplified workflow.
    """
    return [
        "EUDR-001", "EUDR-002", "EUDR-003", "EUDR-007",
        "EUDR-016", "EUDR-018", "EUDR-023",
    ]


def get_workflow_template_metadata(
    workflow_type: WorkflowType,
    commodity: Optional[EUDRCommodity] = None,
) -> Dict[str, Any]:
    """Get metadata for a workflow template.

    Args:
        workflow_type: Standard, simplified, or custom.
        commodity: Optional EUDR commodity.

    Returns:
        Template metadata dictionary.
    """
    commodity_label = commodity.value if commodity else "all"

    return {
        "workflow_type": workflow_type.value,
        "commodity": commodity_label,
        "agent_count": (
            7 if workflow_type == WorkflowType.SIMPLIFIED
            else 25
        ),
        "quality_gates": (
            ["QG-1", "QG-2"]
            if workflow_type == WorkflowType.SIMPLIFIED
            else ["QG-1", "QG-2", "QG-3"]
        ),
        "timeout_overrides": get_commodity_timeouts(commodity),
        "fallback_overrides": {
            k: v.value
            for k, v in get_commodity_fallbacks(commodity).items()
        },
        "critical_agents": get_critical_agents(commodity),
    }
