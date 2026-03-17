# -*- coding: utf-8 -*-
"""
MRVClaimsBridge - AGENT-MRV Claim Verification Bridge for PACK-018
=====================================================================

This module routes carbon and climate-related marketing claims to the
appropriate MRV agents (001-030) for emission data verification. It maps
claim types like "carbon neutral", "net zero", or "reduced emissions"
to specific MRV agent calculations, ensuring that environmental marketing
claims are backed by verified GHG Protocol emission data.

MRV Agent Routing:
    Scope 1 (Direct Emissions):
        MRV-001: Stationary Combustion   --> "low-carbon energy" claims
        MRV-002: Refrigerants & F-Gas    --> "zero-ODP" claims
        MRV-003: Mobile Combustion       --> "fleet electrification" claims
        MRV-004: Process Emissions       --> "clean manufacturing" claims
        MRV-005: Fugitive Emissions      --> "leak-free" claims
        MRV-006: Land Use Emissions      --> "nature-positive" claims
        MRV-007: Waste Treatment         --> "zero-waste" claims
        MRV-008: Agricultural Emissions  --> "sustainable farming" claims

    Scope 2 (Indirect - Energy):
        MRV-009: Location-Based          --> "renewable energy" claims
        MRV-010: Market-Based            --> "100% green power" claims
        MRV-011: Steam/Heat Purchase     --> "green heat" claims
        MRV-012: Cooling Purchase        --> "green cooling" claims
        MRV-013: Dual Reporting          --> scope 2 reconciliation

    Scope 3 (Value Chain):
        MRV-014 to MRV-028: Cat 1-15    --> supply chain claims
        MRV-029: Category Mapper         --> cross-category routing
        MRV-030: Audit Trail             --> provenance verification

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GHGScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_CUTTING = "cross_cutting"


class ClaimVerificationStatus(str, Enum):
    """Claim verification outcome status."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    PENDING = "pending"


class RoutingStatus(str, Enum):
    """Status of a routing operation."""

    ROUTED = "routed"
    NO_AGENT = "no_agent_found"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Agent Routing Tables
# ---------------------------------------------------------------------------

SCOPE1_AGENTS: Dict[str, str] = {
    "MRV-001": "stationary_combustion",
    "MRV-002": "refrigerants_fgas",
    "MRV-003": "mobile_combustion",
    "MRV-004": "process_emissions",
    "MRV-005": "fugitive_emissions",
    "MRV-006": "land_use_emissions",
    "MRV-007": "waste_treatment",
    "MRV-008": "agricultural_emissions",
}

SCOPE2_AGENTS: Dict[str, str] = {
    "MRV-009": "scope2_location_based",
    "MRV-010": "scope2_market_based",
    "MRV-011": "steam_heat_purchase",
    "MRV-012": "cooling_purchase",
    "MRV-013": "dual_reporting_reconciliation",
}

SCOPE3_AGENTS: Dict[str, str] = {
    "MRV-014": "purchased_goods_services",
    "MRV-015": "capital_goods",
    "MRV-016": "fuel_energy_activities",
    "MRV-017": "upstream_transportation",
    "MRV-018": "waste_generated",
    "MRV-019": "business_travel",
    "MRV-020": "employee_commuting",
    "MRV-021": "upstream_leased_assets",
    "MRV-022": "downstream_transportation",
    "MRV-023": "processing_sold_products",
    "MRV-024": "use_sold_products",
    "MRV-025": "end_of_life_treatment",
    "MRV-026": "downstream_leased_assets",
    "MRV-027": "franchises",
    "MRV-028": "investments",
}

CROSS_CUTTING_AGENTS: Dict[str, str] = {
    "MRV-029": "scope3_category_mapper",
    "MRV-030": "audit_trail_lineage",
}

CLAIM_TO_AGENT_MAP: Dict[str, List[str]] = {
    "carbon_neutral": ["MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005",
                        "MRV-009", "MRV-010", "MRV-014", "MRV-029"],
    "net_zero": ["MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005",
                 "MRV-006", "MRV-007", "MRV-008", "MRV-009", "MRV-010",
                 "MRV-014", "MRV-017", "MRV-029"],
    "carbon_reduced": ["MRV-001", "MRV-003", "MRV-009", "MRV-029"],
    "low_carbon": ["MRV-001", "MRV-003", "MRV-009", "MRV-010"],
    "renewable_energy": ["MRV-009", "MRV-010", "MRV-011"],
    "green_power": ["MRV-010", "MRV-011", "MRV-012"],
    "fleet_electrification": ["MRV-003"],
    "clean_manufacturing": ["MRV-004", "MRV-005"],
    "zero_waste": ["MRV-007", "MRV-018"],
    "sustainable_supply_chain": ["MRV-014", "MRV-017", "MRV-022", "MRV-029"],
    "nature_positive": ["MRV-006"],
    "sustainable_farming": ["MRV-008"],
    "scope3_reduction": ["MRV-014", "MRV-015", "MRV-016", "MRV-017", "MRV-018",
                          "MRV-019", "MRV-020", "MRV-021", "MRV-022", "MRV-023",
                          "MRV-024", "MRV-025", "MRV-026", "MRV-027", "MRV-028"],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVRoutingConfig(BaseModel):
    """Configuration for MRV claim verification routing."""

    pack_id: str = Field(default="PACK-018")
    enable_scope1: bool = Field(default=True)
    enable_scope2: bool = Field(default=True)
    enable_scope3: bool = Field(default=True)
    enable_cross_cutting: bool = Field(default=True)
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Minimum confidence for claim verification",
    )


class AgentRoutingEntry(BaseModel):
    """A single agent routing entry for claim verification."""

    agent_id: str = Field(..., description="MRV agent ID (e.g., MRV-001)")
    agent_name: str = Field(default="")
    scope: GHGScope = Field(default=GHGScope.SCOPE_1)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    data_required: List[str] = Field(default_factory=list)


class MRVRoutingResult(BaseModel):
    """Result of an MRV claim verification routing operation."""

    routing_id: str = Field(default_factory=_new_uuid)
    claim_type: str = Field(default="")
    status: RoutingStatus = Field(default=RoutingStatus.ROUTED)
    agents_routed: List[AgentRoutingEntry] = Field(default_factory=list)
    scopes_involved: List[str] = Field(default_factory=list)
    verification_status: ClaimVerificationStatus = Field(
        default=ClaimVerificationStatus.PENDING
    )
    total_agents: int = Field(default=0)
    timestamp: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# MRVClaimsBridge
# ---------------------------------------------------------------------------


class MRVClaimsBridge:
    """Routes carbon and climate claims to MRV agents for verification.

    Maps environmental marketing claim types to the appropriate AGENT-MRV
    agents (001-030) to verify the underlying emission data that supports
    or contradicts the claim.

    Attributes:
        config: MRV routing configuration.

    Example:
        >>> config = MRVRoutingConfig()
        >>> bridge = MRVClaimsBridge(config)
        >>> result = bridge.route_claim_verification("net_zero", {"year": 2025})
        >>> assert result["status"] == "routed"
    """

    def __init__(self, config: Optional[MRVRoutingConfig] = None) -> None:
        """Initialize MRVClaimsBridge.

        Args:
            config: Routing configuration. Defaults used if None.
        """
        self.config = config or MRVRoutingConfig()
        logger.info(
            "MRVClaimsBridge initialized (scope1=%s, scope2=%s, scope3=%s)",
            self.config.enable_scope1,
            self.config.enable_scope2,
            self.config.enable_scope3,
        )

    def route_claim_verification(
        self,
        claim_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route a claim to the appropriate MRV agents for verification.

        Args:
            claim_type: Type of environmental claim (e.g., "net_zero").
            data: Optional context data for verification.

        Returns:
            Dict with routing result including agents, scopes, and hash.
        """
        start = _utcnow()
        context = data or {}
        result = MRVRoutingResult(claim_type=claim_type)

        agent_ids = CLAIM_TO_AGENT_MAP.get(claim_type, [])
        if not agent_ids:
            result.status = RoutingStatus.NO_AGENT
            logger.warning("No MRV agents found for claim type: %s", claim_type)
        else:
            entries = self._build_routing_entries(agent_ids)
            result.agents_routed = entries
            result.total_agents = len(entries)
            result.scopes_involved = list(set(e.scope.value for e in entries))
            result.status = RoutingStatus.ROUTED

        result.duration_ms = (_utcnow() - start).total_seconds() * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "MRVClaimsBridge routed '%s' to %d agents across %s",
            claim_type,
            result.total_agents,
            result.scopes_involved,
        )

        return result.model_dump(mode="json")

    def get_agents_for_scope(self, scope: GHGScope) -> Dict[str, str]:
        """Get all MRV agents for a given GHG scope.

        Args:
            scope: GHG Protocol scope.

        Returns:
            Dict mapping agent IDs to agent names.
        """
        scope_map = {
            GHGScope.SCOPE_1: SCOPE1_AGENTS,
            GHGScope.SCOPE_2: SCOPE2_AGENTS,
            GHGScope.SCOPE_3: SCOPE3_AGENTS,
            GHGScope.CROSS_CUTTING: CROSS_CUTTING_AGENTS,
        }
        return scope_map.get(scope, {})

    def get_supported_claim_types(self) -> List[str]:
        """Return list of all supported claim types for MRV verification."""
        return list(CLAIM_TO_AGENT_MAP.keys())

    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary of MRV routing configuration.

        Returns:
            Dict with agent counts per scope and total claim types.
        """
        return {
            "scope1_agents": len(SCOPE1_AGENTS),
            "scope2_agents": len(SCOPE2_AGENTS),
            "scope3_agents": len(SCOPE3_AGENTS),
            "cross_cutting_agents": len(CROSS_CUTTING_AGENTS),
            "total_agents": (
                len(SCOPE1_AGENTS) + len(SCOPE2_AGENTS)
                + len(SCOPE3_AGENTS) + len(CROSS_CUTTING_AGENTS)
            ),
            "supported_claim_types": len(CLAIM_TO_AGENT_MAP),
            "claim_types": list(CLAIM_TO_AGENT_MAP.keys()),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_routing_entries(self, agent_ids: List[str]) -> List[AgentRoutingEntry]:
        """Build routing entries for a list of agent IDs."""
        entries = []
        for agent_id in agent_ids:
            scope = self._determine_scope(agent_id)
            if not self._is_scope_enabled(scope):
                continue
            agent_name = self._resolve_agent_name(agent_id)
            entries.append(AgentRoutingEntry(
                agent_id=agent_id,
                agent_name=agent_name,
                scope=scope,
                relevance_score=self._compute_relevance(agent_id),
            ))
        return entries

    def _determine_scope(self, agent_id: str) -> GHGScope:
        """Determine the GHG scope for an agent ID."""
        if agent_id in SCOPE1_AGENTS:
            return GHGScope.SCOPE_1
        if agent_id in SCOPE2_AGENTS:
            return GHGScope.SCOPE_2
        if agent_id in SCOPE3_AGENTS:
            return GHGScope.SCOPE_3
        return GHGScope.CROSS_CUTTING

    def _is_scope_enabled(self, scope: GHGScope) -> bool:
        """Check if a scope is enabled in configuration."""
        scope_flags = {
            GHGScope.SCOPE_1: self.config.enable_scope1,
            GHGScope.SCOPE_2: self.config.enable_scope2,
            GHGScope.SCOPE_3: self.config.enable_scope3,
            GHGScope.CROSS_CUTTING: self.config.enable_cross_cutting,
        }
        return scope_flags.get(scope, True)

    def _resolve_agent_name(self, agent_id: str) -> str:
        """Resolve an agent ID to its human-readable name."""
        all_agents = {**SCOPE1_AGENTS, **SCOPE2_AGENTS, **SCOPE3_AGENTS, **CROSS_CUTTING_AGENTS}
        return all_agents.get(agent_id, "unknown")

    def _compute_relevance(self, agent_id: str) -> float:
        """Compute relevance score for an agent in claim verification."""
        primary_agents = {"MRV-001", "MRV-009", "MRV-010", "MRV-014", "MRV-029"}
        if agent_id in primary_agents:
            return 1.0
        return 0.8
