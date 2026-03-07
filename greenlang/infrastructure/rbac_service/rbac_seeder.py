# -*- coding: utf-8 -*-
"""
RBAC Seeder - RBAC Authorization Service (SEC-002)

Seeds default system roles, standard permissions, and role-permission
mappings into the ``security`` schema tables created by V010.

Used for:
    - Development and testing environments where the database is
      recreated frequently without running Flyway migrations.
    - Programmatic re-seeding when the V010 seed data needs to be
      refreshed (e.g. after a schema-only restore).
    - Integration tests that need a fully populated RBAC catalogue.

All operations are idempotent: running the seeder multiple times
produces the same result thanks to ``ON CONFLICT DO NOTHING``.

Example:
    >>> seeder = RBACSeeder(db_pool)
    >>> result = await seeder.seed_all()
    >>> print(result)
    {'roles': 10, 'permissions': 61, 'role_permissions': 215}

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed data definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemRoleDef:
    """Definition of a system role to be seeded."""

    name: str
    display_name: str
    description: str


@dataclass(frozen=True)
class PermissionDef:
    """Definition of a standard permission to be seeded."""

    resource: str
    action: str
    description: str


# ---------------------------------------------------------------------------
# System roles (matches V010 seed exactly)
# ---------------------------------------------------------------------------

SYSTEM_ROLES: List[SystemRoleDef] = [
    SystemRoleDef(
        name="super_admin",
        display_name="Super Administrator",
        description="Full system access across all tenants. Reserved for platform operators.",
    ),
    SystemRoleDef(
        name="admin",
        display_name="Administrator",
        description="Tenant-level administrator with full access to tenant resources.",
    ),
    SystemRoleDef(
        name="manager",
        display_name="Manager",
        description=(
            "Manages agents, emissions, jobs, compliance, and factory "
            "resources within a tenant."
        ),
    ),
    SystemRoleDef(
        name="developer",
        display_name="Developer",
        description=(
            "Develops and configures agents, manages emissions data, "
            "and factory resources."
        ),
    ),
    SystemRoleDef(
        name="operator",
        display_name="Operator",
        description=(
            "Executes agents, calculates emissions, manages jobs, and "
            "operates factory pipelines."
        ),
    ),
    SystemRoleDef(
        name="analyst",
        display_name="Analyst",
        description=(
            "Read-only access to agents and factory, full access to "
            "emissions and compliance data."
        ),
    ),
    SystemRoleDef(
        name="viewer",
        display_name="Viewer",
        description=(
            "Read-only and list access to all resources. Cannot modify "
            "or execute anything."
        ),
    ),
    SystemRoleDef(
        name="auditor",
        display_name="Auditor",
        description=(
            "Compliance auditor with read access to audit logs, sessions, "
            "compliance, and RBAC metadata."
        ),
    ),
    SystemRoleDef(
        name="service_account",
        display_name="Service Account",
        description=(
            "Machine-to-machine identity for automated agent execution "
            "and emissions calculations."
        ),
    ),
    SystemRoleDef(
        name="compliance_officer",
        display_name="Compliance Officer",
        description=(
            "EUDR compliance officer with read, write, map, analyze, and "
            "export access to supply chain and regulatory resources."
        ),
    ),
    SystemRoleDef(
        name="supply_chain_analyst",
        display_name="Supply Chain Analyst",
        description=(
            "Analyzes EUDR supply chain graphs with read, write, map, and "
            "analyze access. Cannot export or delete."
        ),
    ),
    SystemRoleDef(
        name="procurement_manager",
        display_name="Procurement Manager",
        description=(
            "Manages procurement and supplier data with read, write, and "
            "map access to EUDR supply chain resources."
        ),
    ),
    SystemRoleDef(
        name="guest",
        display_name="Guest",
        description="Unauthenticated or minimal-access identity. No default permissions.",
    ),
]

# ---------------------------------------------------------------------------
# Standard permissions (matches V010 seed exactly)
# ---------------------------------------------------------------------------

STANDARD_PERMISSIONS: List[PermissionDef] = [
    # agents
    PermissionDef("agents", "list", "List agents"),
    PermissionDef("agents", "read", "View agent details"),
    PermissionDef("agents", "execute", "Execute an agent"),
    PermissionDef("agents", "configure", "Configure agent settings"),
    PermissionDef("agents", "create", "Create a new agent"),
    PermissionDef("agents", "update", "Update an existing agent"),
    PermissionDef("agents", "delete", "Delete an agent"),
    # emissions
    PermissionDef("emissions", "list", "List emission records"),
    PermissionDef("emissions", "read", "View emission record details"),
    PermissionDef("emissions", "calculate", "Execute emissions calculation"),
    PermissionDef("emissions", "create", "Create emission records"),
    PermissionDef("emissions", "update", "Update emission records"),
    PermissionDef("emissions", "delete", "Delete emission records"),
    PermissionDef("emissions", "export", "Export emission data"),
    # jobs
    PermissionDef("jobs", "list", "List jobs"),
    PermissionDef("jobs", "read", "View job details"),
    PermissionDef("jobs", "create", "Create a new job"),
    PermissionDef("jobs", "cancel", "Cancel a running job"),
    PermissionDef("jobs", "delete", "Delete a job"),
    # compliance
    PermissionDef("compliance", "list", "List compliance reports"),
    PermissionDef("compliance", "read", "View compliance report details"),
    PermissionDef("compliance", "create", "Create compliance reports"),
    PermissionDef("compliance", "update", "Update compliance reports"),
    PermissionDef("compliance", "delete", "Delete compliance reports"),
    PermissionDef("compliance", "approve", "Approve compliance reports"),
    # factory
    PermissionDef("factory", "list", "List factory agents"),
    PermissionDef("factory", "read", "View factory agent details"),
    PermissionDef("factory", "create", "Create factory agent entries"),
    PermissionDef("factory", "update", "Update factory agent entries"),
    PermissionDef("factory", "delete", "Delete factory agent entries"),
    PermissionDef("factory", "execute", "Execute factory agent pipelines"),
    PermissionDef("factory", "metrics", "View factory agent metrics"),
    PermissionDef("factory", "deploy", "Deploy factory agents"),
    PermissionDef("factory", "rollback", "Rollback factory agent deployments"),
    # flags
    PermissionDef("flags", "list", "List feature flags"),
    PermissionDef("flags", "read", "View feature flag details"),
    PermissionDef("flags", "create", "Create feature flags"),
    PermissionDef("flags", "update", "Update feature flags"),
    PermissionDef("flags", "delete", "Delete feature flags"),
    PermissionDef("flags", "evaluate", "Evaluate feature flags"),
    PermissionDef("flags", "rollout", "Manage flag rollout percentages"),
    PermissionDef("flags", "kill", "Activate kill switch on a flag"),
    PermissionDef("flags", "restore", "Restore a killed flag"),
    # admin
    PermissionDef("admin:users", "list", "List users"),
    PermissionDef("admin:users", "read", "View user details"),
    PermissionDef("admin:users", "unlock", "Unlock locked accounts"),
    PermissionDef("admin:users", "revoke", "Revoke user tokens"),
    PermissionDef("admin:users", "reset", "Force password reset"),
    PermissionDef("admin:users", "mfa", "Manage user MFA settings"),
    PermissionDef("admin:sessions", "list", "List active sessions"),
    PermissionDef("admin:sessions", "terminate", "Terminate sessions"),
    PermissionDef("admin:audit", "read", "Read audit logs"),
    PermissionDef("admin:lockouts", "list", "List account lockouts"),
    # rbac
    PermissionDef("rbac:roles", "list", "List RBAC roles"),
    PermissionDef("rbac:roles", "read", "View RBAC role details"),
    PermissionDef("rbac:roles", "create", "Create RBAC roles"),
    PermissionDef("rbac:roles", "update", "Update RBAC roles"),
    PermissionDef("rbac:roles", "delete", "Delete RBAC roles"),
    PermissionDef("rbac:permissions", "list", "List RBAC permissions"),
    PermissionDef("rbac:permissions", "read", "View RBAC permission details"),
    PermissionDef("rbac:assignments", "list", "List role assignments"),
    PermissionDef("rbac:assignments", "read", "View role assignment details"),
    PermissionDef("rbac:assignments", "create", "Assign roles to users"),
    PermissionDef("rbac:assignments", "revoke", "Revoke role assignments"),
    # eudr-supply-chain (AGENT-EUDR-001)
    PermissionDef("eudr-supply-chain", "read", "View supply chain graphs and traceability data"),
    PermissionDef("eudr-supply-chain", "write", "Create and update supply chain graphs"),
    PermissionDef("eudr-supply-chain", "delete", "Archive/delete supply chain graphs"),
    PermissionDef("eudr-supply-chain", "map", "Trigger multi-tier supply chain discovery"),
    PermissionDef("eudr-supply-chain", "analyze", "Run risk propagation and gap analysis"),
    PermissionDef("eudr-supply-chain", "export", "Generate DDS regulatory exports"),
    PermissionDef("eudr-supply-chain", "admin", "Full EUDR supply chain mapper access"),
    # eudr-geo (AGENT-EUDR-002 Geolocation Verification)
    PermissionDef("eudr-geo", "coordinates:verify", "Validate GPS coordinates"),
    PermissionDef("eudr-geo", "polygon:verify", "Verify polygon topology"),
    PermissionDef("eudr-geo", "polygon:repair", "Attempt polygon auto-repair"),
    PermissionDef("eudr-geo", "protected-areas:check", "Screen plots against protected areas"),
    PermissionDef("eudr-geo", "deforestation:verify", "Verify deforestation cutoff status"),
    PermissionDef("eudr-geo", "plots:verify", "Perform full plot verification"),
    PermissionDef("eudr-geo", "plots:read", "View verification results"),
    PermissionDef("eudr-geo", "batch:submit", "Submit batch verification jobs"),
    PermissionDef("eudr-geo", "batch:read", "View batch job status"),
    PermissionDef("eudr-geo", "batch:cancel", "Cancel running batch jobs"),
    PermissionDef("eudr-geo", "scores:read", "View accuracy scores"),
    PermissionDef("eudr-geo", "scores:configure", "Configure score weights"),
    PermissionDef("eudr-geo", "compliance:generate", "Generate compliance reports"),
    PermissionDef("eudr-geo", "compliance:read", "View compliance reports"),
    PermissionDef("eudr-geo", "audit:read", "View verification audit trail"),
    PermissionDef("eudr-geo", "admin", "Full EUDR geolocation verification access"),
    # eudr-sat (AGENT-EUDR-003 Satellite Monitoring)
    PermissionDef("eudr-sat", "imagery:search", "Search satellite imagery scenes"),
    PermissionDef("eudr-sat", "imagery:download", "Download satellite scene bands"),
    PermissionDef("eudr-sat", "imagery:read", "View satellite scene metadata"),
    PermissionDef("eudr-sat", "analysis:create", "Run satellite analysis (NDVI, change detection, fusion)"),
    PermissionDef("eudr-sat", "analysis:read", "View satellite analysis results"),
    PermissionDef("eudr-sat", "baseline:create", "Establish Dec 2020 baselines"),
    PermissionDef("eudr-sat", "baseline:read", "View baseline data"),
    PermissionDef("eudr-sat", "monitoring:create", "Create monitoring schedules"),
    PermissionDef("eudr-sat", "monitoring:read", "View monitoring results"),
    PermissionDef("eudr-sat", "monitoring:update", "Update monitoring schedules"),
    PermissionDef("eudr-sat", "monitoring:delete", "Delete monitoring schedules"),
    PermissionDef("eudr-sat", "alerts:read", "View satellite alerts"),
    PermissionDef("eudr-sat", "alerts:acknowledge", "Acknowledge satellite alerts"),
    PermissionDef("eudr-sat", "evidence:create", "Generate evidence packages"),
    PermissionDef("eudr-sat", "evidence:read", "View evidence packages"),
    PermissionDef("eudr-sat", "evidence:download", "Download evidence packages"),
    PermissionDef("eudr-sat", "admin", "Full EUDR satellite monitoring access"),
    # eudr-fca (AGENT-EUDR-004 Forest Cover Analysis)
    PermissionDef("eudr-fca", "read", "View forest cover analysis results"),
    PermissionDef("eudr-fca", "write", "Create/update forest cover analyses"),
    PermissionDef("eudr-fca", "density:analyze", "Run canopy density analysis"),
    PermissionDef("eudr-fca", "density:batch", "Run batch canopy density analysis"),
    PermissionDef("eudr-fca", "classify:run", "Run forest type classification"),
    PermissionDef("eudr-fca", "classify:batch", "Run batch forest classification"),
    PermissionDef("eudr-fca", "historical:reconstruct", "Reconstruct historical forest cover"),
    PermissionDef("eudr-fca", "historical:compare", "Compare historical vs current cover"),
    PermissionDef("eudr-fca", "verify:single", "Verify deforestation-free status"),
    PermissionDef("eudr-fca", "verify:batch", "Batch deforestation-free verification"),
    PermissionDef("eudr-fca", "height:estimate", "Estimate canopy height"),
    PermissionDef("eudr-fca", "fragmentation:analyze", "Analyze forest fragmentation"),
    PermissionDef("eudr-fca", "biomass:estimate", "Estimate above-ground biomass"),
    PermissionDef("eudr-fca", "reports:generate", "Generate compliance reports"),
    PermissionDef("eudr-fca", "reports:download", "Download compliance reports"),
    PermissionDef("eudr-fca", "batch:submit", "Submit batch analysis jobs"),
    PermissionDef("eudr-fca", "batch:cancel", "Cancel batch analysis jobs"),
    PermissionDef("eudr-fca", "admin", "Full EUDR forest cover analysis access"),

    # eudr-luc (AGENT-EUDR-005 Land Use Change Detector)
    PermissionDef("eudr-luc", "read", "View land use change analysis results"),
    PermissionDef("eudr-luc", "write", "Create/update land use analyses"),
    PermissionDef("eudr-luc", "classify:run", "Run land use classification"),
    PermissionDef("eudr-luc", "classify:batch", "Run batch land use classification"),
    PermissionDef("eudr-luc", "transitions:detect", "Detect land use transitions"),
    PermissionDef("eudr-luc", "transitions:batch", "Run batch transition detection"),
    PermissionDef("eudr-luc", "trajectory:analyze", "Analyze temporal trajectories"),
    PermissionDef("eudr-luc", "trajectory:batch", "Run batch trajectory analysis"),
    PermissionDef("eudr-luc", "verify:cutoff", "Verify EUDR cutoff compliance"),
    PermissionDef("eudr-luc", "verify:batch", "Batch cutoff verification"),
    PermissionDef("eudr-luc", "risk:assess", "Assess conversion risk"),
    PermissionDef("eudr-luc", "risk:batch", "Batch risk assessment"),
    PermissionDef("eudr-luc", "urban:analyze", "Analyze urban encroachment"),
    PermissionDef("eudr-luc", "urban:batch", "Batch urban analysis"),
    PermissionDef("eudr-luc", "reports:generate", "Generate compliance reports"),
    PermissionDef("eudr-luc", "reports:download", "Download compliance reports"),
    PermissionDef("eudr-luc", "batch:submit", "Submit batch analysis jobs"),
    PermissionDef("eudr-luc", "batch:cancel", "Cancel batch analysis jobs"),
    PermissionDef("eudr-luc", "admin", "Full EUDR land use change access"),

    # eudr-pbm (AGENT-EUDR-006 Plot Boundary Manager)
    PermissionDef("eudr-pbm", "read", "View plot boundaries and related data"),
    PermissionDef("eudr-pbm", "write", "Create/update plot boundaries"),
    PermissionDef("eudr-pbm", "delete", "Delete plot boundaries"),
    PermissionDef("eudr-pbm", "validate:run", "Run topology validation"),
    PermissionDef("eudr-pbm", "validate:batch", "Run batch validation"),
    PermissionDef("eudr-pbm", "repair:run", "Run auto-repair on boundaries"),
    PermissionDef("eudr-pbm", "repair:batch", "Run batch auto-repair"),
    PermissionDef("eudr-pbm", "area:calculate", "Calculate geodetic area"),
    PermissionDef("eudr-pbm", "area:batch", "Batch area calculation"),
    PermissionDef("eudr-pbm", "overlaps:detect", "Detect plot overlaps"),
    PermissionDef("eudr-pbm", "overlaps:scan", "Full registry overlap scan"),
    PermissionDef("eudr-pbm", "overlaps:resolve", "Suggest overlap resolution"),
    PermissionDef("eudr-pbm", "export:run", "Export boundaries"),
    PermissionDef("eudr-pbm", "export:batch", "Batch export boundaries"),
    PermissionDef("eudr-pbm", "export:download", "Download exported files"),
    PermissionDef("eudr-pbm", "split:run", "Split plot boundaries"),
    PermissionDef("eudr-pbm", "merge:run", "Merge plot boundaries"),
    PermissionDef("eudr-pbm", "batch:submit", "Submit batch boundary jobs"),
    PermissionDef("eudr-pbm", "batch:cancel", "Cancel batch boundary jobs"),
    PermissionDef("eudr-pbm", "admin", "Full plot boundary manager access"),

    # eudr-gcv (AGENT-EUDR-007 GPS Coordinate Validator)
    PermissionDef("eudr-gcv", "read", "View coordinate validation data"),
    PermissionDef("eudr-gcv", "parse:single", "Parse single coordinate"),
    PermissionDef("eudr-gcv", "parse:batch", "Batch parse coordinates"),
    PermissionDef("eudr-gcv", "parse:detect", "Detect coordinate format"),
    PermissionDef("eudr-gcv", "parse:normalize", "Normalize coordinates to WGS84"),
    PermissionDef("eudr-gcv", "validate:single", "Validate single coordinate"),
    PermissionDef("eudr-gcv", "validate:batch", "Batch validate coordinates"),
    PermissionDef("eudr-gcv", "validate:swap", "Detect lat/lon swaps"),
    PermissionDef("eudr-gcv", "validate:duplicates", "Detect duplicate coordinates"),
    PermissionDef("eudr-gcv", "plausibility:check", "Run full plausibility check"),
    PermissionDef("eudr-gcv", "plausibility:land-ocean", "Check land/ocean classification"),
    PermissionDef("eudr-gcv", "plausibility:country", "Verify country match"),
    PermissionDef("eudr-gcv", "plausibility:commodity", "Check commodity plausibility"),
    PermissionDef("eudr-gcv", "plausibility:elevation", "Check elevation plausibility"),
    PermissionDef("eudr-gcv", "assess:single", "Assess single coordinate accuracy"),
    PermissionDef("eudr-gcv", "assess:batch", "Batch accuracy assessment"),
    PermissionDef("eudr-gcv", "assess:precision", "Precision-only analysis"),
    PermissionDef("eudr-gcv", "reports:generate", "Generate compliance reports"),
    PermissionDef("eudr-gcv", "reports:read", "View compliance reports"),
    PermissionDef("eudr-gcv", "reports:download", "Download compliance reports"),
    PermissionDef("eudr-gcv", "geocode:reverse", "Reverse geocode coordinate"),
    PermissionDef("eudr-gcv", "geocode:batch", "Batch reverse geocode"),
    PermissionDef("eudr-gcv", "geocode:country", "Country lookup from coordinate"),
    PermissionDef("eudr-gcv", "datum:transform", "Transform datum to WGS84"),
    PermissionDef("eudr-gcv", "datum:batch", "Batch datum transformation"),
    PermissionDef("eudr-gcv", "batch:submit", "Submit batch validation jobs"),
    PermissionDef("eudr-gcv", "batch:cancel", "Cancel batch validation jobs"),
    PermissionDef("eudr-gcv", "admin", "Full GPS coordinate validator access"),
]

# ---------------------------------------------------------------------------
# Default role-permission mappings
# ---------------------------------------------------------------------------
# Each key is a role name.  The value is either:
#   - "__all__"   : grant ALL permissions
#   - A list of (resource, action) tuples that should be granted
#   - "__all_except__": a tuple of (list_to_grant_as_all, exclusions)
# ---------------------------------------------------------------------------

_ALL_MARKER = "__all__"

# Mapping from role name to a filter function that selects which
# permissions to grant.  Each function receives (resource, action)
# and returns True if the permission should be granted.

DEFAULT_MAPPINGS: Dict[str, Any] = {
    "super_admin": _ALL_MARKER,
    "admin": "__all_except__",  # handled specially
    "manager": [
        ("agents", None),           # all agents actions
        ("emissions", None),        # all emissions actions
        ("jobs", None),             # all jobs actions
        ("compliance", None),       # all compliance actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "execute"),
    ],
    "developer": [
        ("agents", "list"),
        ("agents", "read"),
        ("agents", "execute"),
        ("agents", "configure"),
        ("emissions", None),        # all emissions actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "create"),
        ("factory", "update"),
        ("factory", "execute"),
    ],
    "operator": [
        ("agents", "list"),
        ("agents", "read"),
        ("agents", "execute"),
        ("emissions", "list"),
        ("emissions", "read"),
        ("emissions", "calculate"),
        ("jobs", None),             # all jobs actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "execute"),
    ],
    "analyst": [
        ("agents", "list"),
        ("agents", "read"),
        ("emissions", None),        # all emissions actions
        ("compliance", "list"),
        ("compliance", "read"),
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "metrics"),
    ],
    "viewer": "__viewer__",         # all list + read permissions
    "auditor": [
        ("admin:audit", "read"),
        ("admin:sessions", "list"),
        ("compliance", None),       # all compliance actions
        ("rbac:roles", "list"),
        ("rbac:permissions", "list"),
        ("rbac:assignments", "list"),
        # EUDR auditor_readonly: read + export (AGENT-EUDR-001)
        ("eudr-supply-chain", "read"),
        ("eudr-supply-chain", "export"),
        # EUDR geolocation verification auditor (AGENT-EUDR-002)
        ("eudr-geo", "plots:read"),
        ("eudr-geo", "scores:read"),
        ("eudr-geo", "compliance:read"),
        ("eudr-geo", "audit:read"),
        # EUDR satellite monitoring auditor (AGENT-EUDR-003)
        ("eudr-sat", "imagery:read"),
        ("eudr-sat", "analysis:read"),
        ("eudr-sat", "baseline:read"),
        ("eudr-sat", "monitoring:read"),
        ("eudr-sat", "alerts:read"),
        ("eudr-sat", "evidence:read"),
        # EUDR forest cover analysis auditor (AGENT-EUDR-004)
        ("eudr-fca", "read"),
        ("eudr-fca", "reports:download"),
        # EUDR land use change detector auditor (AGENT-EUDR-005)
        ("eudr-luc", "read"),
        ("eudr-luc", "reports:download"),
        # EUDR plot boundary manager auditor (AGENT-EUDR-006)
        ("eudr-pbm", "read"),
        ("eudr-pbm", "export:download"),
        # EUDR GPS coordinate validator auditor (AGENT-EUDR-007)
        ("eudr-gcv", "read"),
        ("eudr-gcv", "reports:read"),
        ("eudr-gcv", "reports:download"),
    ],
    "compliance_officer": [
        # EUDR compliance officer: read, write, map, analyze, export (AGENT-EUDR-001)
        ("eudr-supply-chain", "read"),
        ("eudr-supply-chain", "write"),
        ("eudr-supply-chain", "map"),
        ("eudr-supply-chain", "analyze"),
        ("eudr-supply-chain", "export"),
        # Inherit base compliance access
        ("compliance", None),       # all compliance actions
        # EUDR geolocation verification (AGENT-EUDR-002) -- full verify + report access
        ("eudr-geo", "coordinates:verify"),
        ("eudr-geo", "polygon:verify"),
        ("eudr-geo", "polygon:repair"),
        ("eudr-geo", "protected-areas:check"),
        ("eudr-geo", "deforestation:verify"),
        ("eudr-geo", "plots:verify"),
        ("eudr-geo", "plots:read"),
        ("eudr-geo", "batch:submit"),
        ("eudr-geo", "batch:read"),
        ("eudr-geo", "batch:cancel"),
        ("eudr-geo", "scores:read"),
        ("eudr-geo", "scores:configure"),
        ("eudr-geo", "compliance:generate"),
        ("eudr-geo", "compliance:read"),
        ("eudr-geo", "audit:read"),
        # EUDR satellite monitoring compliance officer (AGENT-EUDR-003) -- full access
        ("eudr-sat", "imagery:search"),
        ("eudr-sat", "imagery:download"),
        ("eudr-sat", "imagery:read"),
        ("eudr-sat", "analysis:create"),
        ("eudr-sat", "analysis:read"),
        ("eudr-sat", "baseline:create"),
        ("eudr-sat", "baseline:read"),
        ("eudr-sat", "monitoring:create"),
        ("eudr-sat", "monitoring:read"),
        ("eudr-sat", "monitoring:update"),
        ("eudr-sat", "monitoring:delete"),
        ("eudr-sat", "alerts:read"),
        ("eudr-sat", "alerts:acknowledge"),
        ("eudr-sat", "evidence:create"),
        ("eudr-sat", "evidence:read"),
        ("eudr-sat", "evidence:download"),
        # EUDR forest cover analysis compliance officer (AGENT-EUDR-004) -- full access
        ("eudr-fca", "read"),
        ("eudr-fca", "write"),
        ("eudr-fca", "density:analyze"),
        ("eudr-fca", "density:batch"),
        ("eudr-fca", "classify:run"),
        ("eudr-fca", "classify:batch"),
        ("eudr-fca", "historical:reconstruct"),
        ("eudr-fca", "historical:compare"),
        ("eudr-fca", "verify:single"),
        ("eudr-fca", "verify:batch"),
        ("eudr-fca", "height:estimate"),
        ("eudr-fca", "fragmentation:analyze"),
        ("eudr-fca", "biomass:estimate"),
        ("eudr-fca", "reports:generate"),
        ("eudr-fca", "reports:download"),
        ("eudr-fca", "batch:submit"),
        ("eudr-fca", "batch:cancel"),
        # EUDR land use change detector compliance officer (AGENT-EUDR-005) -- full access
        ("eudr-luc", "read"),
        ("eudr-luc", "write"),
        ("eudr-luc", "classify:run"),
        ("eudr-luc", "classify:batch"),
        ("eudr-luc", "transitions:detect"),
        ("eudr-luc", "transitions:batch"),
        ("eudr-luc", "trajectory:analyze"),
        ("eudr-luc", "trajectory:batch"),
        ("eudr-luc", "verify:cutoff"),
        ("eudr-luc", "verify:batch"),
        ("eudr-luc", "risk:assess"),
        ("eudr-luc", "risk:batch"),
        ("eudr-luc", "urban:analyze"),
        ("eudr-luc", "urban:batch"),
        ("eudr-luc", "reports:generate"),
        ("eudr-luc", "reports:download"),
        ("eudr-luc", "batch:submit"),
        ("eudr-luc", "batch:cancel"),
        # EUDR plot boundary manager compliance_officer (AGENT-EUDR-006)
        ("eudr-pbm", "read"),
        ("eudr-pbm", "write"),
        ("eudr-pbm", "delete"),
        ("eudr-pbm", "validate:run"),
        ("eudr-pbm", "validate:batch"),
        ("eudr-pbm", "repair:run"),
        ("eudr-pbm", "repair:batch"),
        ("eudr-pbm", "area:calculate"),
        ("eudr-pbm", "area:batch"),
        ("eudr-pbm", "overlaps:detect"),
        ("eudr-pbm", "overlaps:scan"),
        ("eudr-pbm", "overlaps:resolve"),
        ("eudr-pbm", "export:run"),
        ("eudr-pbm", "export:batch"),
        ("eudr-pbm", "export:download"),
        ("eudr-pbm", "split:run"),
        ("eudr-pbm", "merge:run"),
        ("eudr-pbm", "batch:submit"),
        ("eudr-pbm", "batch:cancel"),
        # EUDR GPS coordinate validator compliance_officer (AGENT-EUDR-007) -- full access
        ("eudr-gcv", "read"),
        ("eudr-gcv", "parse:single"),
        ("eudr-gcv", "parse:batch"),
        ("eudr-gcv", "parse:detect"),
        ("eudr-gcv", "parse:normalize"),
        ("eudr-gcv", "validate:single"),
        ("eudr-gcv", "validate:batch"),
        ("eudr-gcv", "validate:swap"),
        ("eudr-gcv", "validate:duplicates"),
        ("eudr-gcv", "plausibility:check"),
        ("eudr-gcv", "plausibility:land-ocean"),
        ("eudr-gcv", "plausibility:country"),
        ("eudr-gcv", "plausibility:commodity"),
        ("eudr-gcv", "plausibility:elevation"),
        ("eudr-gcv", "assess:single"),
        ("eudr-gcv", "assess:batch"),
        ("eudr-gcv", "assess:precision"),
        ("eudr-gcv", "reports:generate"),
        ("eudr-gcv", "reports:read"),
        ("eudr-gcv", "reports:download"),
        ("eudr-gcv", "geocode:reverse"),
        ("eudr-gcv", "geocode:batch"),
        ("eudr-gcv", "geocode:country"),
        ("eudr-gcv", "datum:transform"),
        ("eudr-gcv", "datum:batch"),
        ("eudr-gcv", "batch:submit"),
        ("eudr-gcv", "batch:cancel"),
    ],
    "supply_chain_analyst": [
        # EUDR supply chain analyst: read, write, map, analyze (AGENT-EUDR-001)
        ("eudr-supply-chain", "read"),
        ("eudr-supply-chain", "write"),
        ("eudr-supply-chain", "map"),
        ("eudr-supply-chain", "analyze"),
        # EUDR geolocation verification (AGENT-EUDR-002) -- verify + read access
        ("eudr-geo", "coordinates:verify"),
        ("eudr-geo", "polygon:verify"),
        ("eudr-geo", "protected-areas:check"),
        ("eudr-geo", "deforestation:verify"),
        ("eudr-geo", "plots:verify"),
        ("eudr-geo", "plots:read"),
        ("eudr-geo", "batch:submit"),
        ("eudr-geo", "batch:read"),
        ("eudr-geo", "scores:read"),
        ("eudr-geo", "compliance:read"),
        # EUDR satellite monitoring analyst (AGENT-EUDR-003) -- search + analysis + read
        ("eudr-sat", "imagery:search"),
        ("eudr-sat", "imagery:read"),
        ("eudr-sat", "analysis:create"),
        ("eudr-sat", "analysis:read"),
        ("eudr-sat", "baseline:read"),
        ("eudr-sat", "monitoring:read"),
        ("eudr-sat", "alerts:read"),
        ("eudr-sat", "evidence:read"),
        # EUDR forest cover analysis analyst (AGENT-EUDR-004) -- analyze + read
        ("eudr-fca", "read"),
        ("eudr-fca", "density:analyze"),
        ("eudr-fca", "classify:run"),
        ("eudr-fca", "historical:reconstruct"),
        ("eudr-fca", "historical:compare"),
        ("eudr-fca", "verify:single"),
        ("eudr-fca", "height:estimate"),
        ("eudr-fca", "fragmentation:analyze"),
        ("eudr-fca", "biomass:estimate"),
        ("eudr-fca", "reports:download"),
        # EUDR land use change detector analyst (AGENT-EUDR-005) -- classify + analyze + read
        ("eudr-luc", "read"),
        ("eudr-luc", "classify:run"),
        ("eudr-luc", "transitions:detect"),
        ("eudr-luc", "trajectory:analyze"),
        ("eudr-luc", "verify:cutoff"),
        ("eudr-luc", "risk:assess"),
        ("eudr-luc", "urban:analyze"),
        ("eudr-luc", "reports:download"),
        ("eudr-luc", "reports:generate"),
        ("eudr-luc", "batch:submit"),
        # EUDR plot boundary manager supply_chain_analyst (AGENT-EUDR-006)
        ("eudr-pbm", "read"),
        ("eudr-pbm", "write"),
        ("eudr-pbm", "validate:run"),
        ("eudr-pbm", "area:calculate"),
        ("eudr-pbm", "overlaps:detect"),
        ("eudr-pbm", "export:run"),
        ("eudr-pbm", "export:download"),
        ("eudr-pbm", "split:run"),
        ("eudr-pbm", "merge:run"),
        ("eudr-pbm", "batch:submit"),
        # EUDR GPS coordinate validator supply_chain_analyst (AGENT-EUDR-007)
        ("eudr-gcv", "read"),
        ("eudr-gcv", "parse:single"),
        ("eudr-gcv", "parse:batch"),
        ("eudr-gcv", "parse:detect"),
        ("eudr-gcv", "parse:normalize"),
        ("eudr-gcv", "validate:single"),
        ("eudr-gcv", "validate:batch"),
        ("eudr-gcv", "validate:swap"),
        ("eudr-gcv", "plausibility:check"),
        ("eudr-gcv", "plausibility:land-ocean"),
        ("eudr-gcv", "plausibility:country"),
        ("eudr-gcv", "assess:single"),
        ("eudr-gcv", "assess:precision"),
        ("eudr-gcv", "geocode:reverse"),
        ("eudr-gcv", "geocode:country"),
        ("eudr-gcv", "datum:transform"),
        ("eudr-gcv", "reports:read"),
        ("eudr-gcv", "reports:download"),
        ("eudr-gcv", "batch:submit"),
    ],
    "procurement_manager": [
        # EUDR procurement manager: read, write, map (AGENT-EUDR-001)
        ("eudr-supply-chain", "read"),
        ("eudr-supply-chain", "write"),
        ("eudr-supply-chain", "map"),
        # EUDR geolocation verification (AGENT-EUDR-002) -- read + basic verify
        ("eudr-geo", "coordinates:verify"),
        ("eudr-geo", "polygon:verify"),
        ("eudr-geo", "protected-areas:check"),
        ("eudr-geo", "plots:read"),
        ("eudr-geo", "scores:read"),
        ("eudr-geo", "compliance:read"),
        # EUDR satellite monitoring procurement (AGENT-EUDR-003) -- read + alerts + evidence
        ("eudr-sat", "imagery:read"),
        ("eudr-sat", "analysis:read"),
        ("eudr-sat", "baseline:read"),
        ("eudr-sat", "monitoring:read"),
        ("eudr-sat", "alerts:read"),
        ("eudr-sat", "evidence:read"),
        ("eudr-sat", "evidence:download"),
        # EUDR forest cover analysis procurement (AGENT-EUDR-004) -- read + reports
        ("eudr-fca", "read"),
        ("eudr-fca", "verify:single"),
        ("eudr-fca", "reports:generate"),
        ("eudr-fca", "reports:download"),
        # EUDR land use change detector procurement (AGENT-EUDR-005) -- read + verify + reports
        ("eudr-luc", "read"),
        ("eudr-luc", "verify:cutoff"),
        ("eudr-luc", "risk:assess"),
        ("eudr-luc", "reports:generate"),
        ("eudr-luc", "reports:download"),
        # EUDR plot boundary manager procurement_manager (AGENT-EUDR-006)
        ("eudr-pbm", "read"),
        ("eudr-pbm", "validate:run"),
        ("eudr-pbm", "area:calculate"),
        ("eudr-pbm", "export:run"),
        ("eudr-pbm", "export:download"),
        # EUDR GPS coordinate validator procurement_manager (AGENT-EUDR-007)
        ("eudr-gcv", "read"),
        ("eudr-gcv", "parse:single"),
        ("eudr-gcv", "parse:detect"),
        ("eudr-gcv", "validate:single"),
        ("eudr-gcv", "plausibility:check"),
        ("eudr-gcv", "assess:single"),
        ("eudr-gcv", "geocode:reverse"),
        ("eudr-gcv", "geocode:country"),
        ("eudr-gcv", "reports:read"),
        ("eudr-gcv", "reports:download"),
    ],
    "service_account": [
        ("agents", "execute"),
        ("emissions", "calculate"),
        ("factory", "execute"),
    ],
    "guest": [],                    # no permissions
}


# ---------------------------------------------------------------------------
# Seeder class
# ---------------------------------------------------------------------------


class RBACSeeder:
    """Seeds default roles and permissions into the RBAC tables.

    Used for development/testing and as a fallback when V010 migration
    seeding needs to be re-run programmatically.

    All operations are idempotent (use ``ON CONFLICT ... DO NOTHING``).

    Args:
        db_pool: An async ``psycopg_pool.AsyncConnectionPool`` or any
            object exposing an ``async with pool.connection() as conn``
            interface.

    Example:
        >>> seeder = RBACSeeder(db_pool)
        >>> result = await seeder.seed_all()
        >>> print(result)
        {'roles': 10, 'permissions': 61, 'role_permissions': 215}
    """

    def __init__(self, db_pool: Any) -> None:
        """Initialize the RBAC seeder.

        Args:
            db_pool: Async PostgreSQL connection pool.
        """
        self._pool = db_pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def seed_all(self) -> Dict[str, int]:
        """Seed roles, permissions, and role-permission mappings.

        Runs all three seed operations in sequence within a single
        transaction for atomicity.

        Returns:
            Dictionary with counts of inserted rows per category.
        """
        logger.info("Starting RBAC seed operation")

        async with self._pool.connection() as conn:
            async with conn.transaction():
                roles_count = await self._seed_roles(conn)
                perms_count = await self._seed_permissions(conn)
                mappings_count = await self._seed_role_permissions(conn)

        result = {
            "roles": roles_count,
            "permissions": perms_count,
            "role_permissions": mappings_count,
        }
        logger.info("RBAC seed complete: %s", result)
        return result

    async def seed_roles(self) -> int:
        """Seed system roles only.

        Returns:
            Number of roles inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_roles(conn)

    async def seed_permissions(self) -> int:
        """Seed standard permissions only.

        Returns:
            Number of permissions inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_permissions(conn)

    async def seed_role_permissions(self) -> int:
        """Seed default role-permission mappings only.

        Requires roles and permissions to already exist.

        Returns:
            Number of role-permission mappings inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_role_permissions(conn)

    # ------------------------------------------------------------------
    # Internal: seed operations
    # ------------------------------------------------------------------

    async def _seed_roles(self, conn: Any) -> int:
        """Insert system roles into security.roles."""
        count = 0
        for role_def in SYSTEM_ROLES:
            result = await conn.execute(
                """
                INSERT INTO security.roles
                    (name, display_name, description, is_system_role, created_by)
                VALUES (%s, %s, %s, true, 'system')
                ON CONFLICT (tenant_id, name) DO NOTHING
                """,
                (role_def.name, role_def.display_name, role_def.description),
            )
            if result.statusmessage and "INSERT 0 1" in result.statusmessage:
                count += 1

        logger.info("Seeded %d system roles (of %d)", count, len(SYSTEM_ROLES))
        return count

    async def _seed_permissions(self, conn: Any) -> int:
        """Insert standard permissions into security.permissions."""
        count = 0
        for perm_def in STANDARD_PERMISSIONS:
            result = await conn.execute(
                """
                INSERT INTO security.permissions
                    (resource, action, description, is_system_permission)
                VALUES (%s, %s, %s, true)
                ON CONFLICT (resource, action) DO NOTHING
                """,
                (perm_def.resource, perm_def.action, perm_def.description),
            )
            if result.statusmessage and "INSERT 0 1" in result.statusmessage:
                count += 1

        logger.info(
            "Seeded %d standard permissions (of %d)",
            count,
            len(STANDARD_PERMISSIONS),
        )
        return count

    async def _seed_role_permissions(self, conn: Any) -> int:
        """Insert default role-permission mappings."""
        total = 0

        for role_name, mapping in DEFAULT_MAPPINGS.items():
            perm_filters = self._resolve_permission_filters(role_name, mapping)
            if perm_filters is None:
                # All permissions
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = %s AND r.is_system_role = true
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    (role_name,),
                )
                total += self._extract_row_count(result)

            elif role_name == "admin":
                # All except rbac:roles:delete
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = 'admin' AND r.is_system_role = true
                      AND NOT (p.resource = 'rbac:roles' AND p.action = 'delete')
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                )
                total += self._extract_row_count(result)

            elif mapping == "__viewer__":
                # All list + read permissions
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = %s AND r.is_system_role = true
                      AND p.action IN ('list', 'read')
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    (role_name,),
                )
                total += self._extract_row_count(result)

            elif isinstance(perm_filters, list):
                for resource, action in perm_filters:
                    if action is None:
                        # All actions for this resource
                        result = await conn.execute(
                            """
                            INSERT INTO security.role_permissions
                                (role_id, permission_id, effect, granted_by)
                            SELECT r.id, p.id, 'allow', 'system'
                            FROM security.roles r
                            CROSS JOIN security.permissions p
                            WHERE r.name = %s AND r.is_system_role = true
                              AND p.resource = %s
                            ON CONFLICT (role_id, permission_id) DO NOTHING
                            """,
                            (role_name, resource),
                        )
                    else:
                        # Specific resource:action
                        result = await conn.execute(
                            """
                            INSERT INTO security.role_permissions
                                (role_id, permission_id, effect, granted_by)
                            SELECT r.id, p.id, 'allow', 'system'
                            FROM security.roles r
                            CROSS JOIN security.permissions p
                            WHERE r.name = %s AND r.is_system_role = true
                              AND p.resource = %s AND p.action = %s
                            ON CONFLICT (role_id, permission_id) DO NOTHING
                            """,
                            (role_name, resource, action),
                        )
                    total += self._extract_row_count(result)

        logger.info("Seeded %d role-permission mappings", total)
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_permission_filters(
        role_name: str,
        mapping: Any,
    ) -> Optional[List[Tuple[str, Optional[str]]]]:
        """Resolve mapping value to a list of permission filters.

        Returns:
            ``None`` for ALL permissions, or a list of
            ``(resource, action_or_none)`` tuples.
        """
        if mapping == _ALL_MARKER:
            return None
        if mapping == "__all_except__":
            # Handled specially in caller
            return None  # pragma: no cover -- sentinel
        if mapping == "__viewer__":
            return mapping  # type: ignore[return-value]
        if isinstance(mapping, list):
            return mapping
        return []

    @staticmethod
    def _extract_row_count(result: Any) -> int:
        """Extract the number of affected rows from a psycopg cursor result.

        Args:
            result: The result from ``conn.execute()``.

        Returns:
            Number of rows inserted (0 if not determinable).
        """
        if result is None:
            return 0
        msg = getattr(result, "statusmessage", "") or ""
        # Format: "INSERT 0 N" where N is the count
        parts = msg.split()
        if len(parts) == 3 and parts[0] == "INSERT":
            try:
                return int(parts[2])
            except ValueError:
                return 0
        return 0
