# -*- coding: utf-8 -*-
"""
AGENT-MRV-030: Audit Trail & Lineage Agent (Cross-Cutting)

Provides immutable, tamper-evident audit trails and end-to-end calculation lineage
for all MRV emissions calculations across Scope 1, 2, and 3.

Every MRV calculation -- from raw activity data ingestion through emission factor
lookup, formula evaluation, aggregation, and final disclosure -- is recorded as a
cryptographically chained audit event with full provenance metadata.  The lineage
graph links inputs, intermediate values, and outputs into a traversable DAG that
enables forward-impact analysis ("what downstream values change if this emission
factor is updated?") and backward-traceability ("which source records contributed
to this disclosed total?").

Agent ID: GL-MRV-X-042
Package: greenlang.audit_trail_lineage
API: /api/v1/audit-trail-lineage
DB Migration: V081
Metrics Prefix: gl_atl_
Table Prefix: gl_atl_

Seven Engines:
    1. AuditEventEngine - Immutable event recording with SHA-256 hash chains
    2. LineageGraphEngine - MRV calculation lineage DAG construction and traversal
    3. EvidencePackagerEngine - Audit evidence bundling for third-party verification
    4. ComplianceTracerEngine - Regulatory framework requirement traceability
    5. ChangeDetectorEngine - Recalculation change tracking and version comparison
    6. ComplianceCheckerEngine - Multi-framework audit trail compliance validation
    7. AuditTrailPipelineEngine - 10-stage orchestration pipeline

Compliance Frameworks:
    - GHG Protocol Corporate Standard / Scope 3 Standard
    - ISO 14064-1:2018 / ISO 14064-3:2019
    - CSRD ESRS E1 (Climate Change)
    - California SB 253
    - EU CBAM
    - CDP Climate Change Questionnaire
    - TCFD Recommendations
    - PCAF Global Standard
    - SBTi

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

__all__ = [
    # Engine classes
    "AuditEventEngine",
    "LineageGraphEngine",
    "EvidencePackagerEngine",
    "ComplianceTracerEngine",
    "ChangeDetectorEngine",
    "ComplianceCheckerEngine",
    "AuditTrailPipelineEngine",
    # Metadata constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Configuration helper
    "get_config",
    # Info helpers
    "get_version",
    "get_agent_info",
]

AGENT_ID: str = "GL-MRV-X-042"
AGENT_COMPONENT: str = "AGENT-MRV-030"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_atl_"

# ---------------------------------------------------------------------------
# Graceful imports -- each engine with try/except so the package can be
# imported even when optional engine dependencies are not yet installed.
# ---------------------------------------------------------------------------

try:
    from greenlang.audit_trail_lineage.audit_event_engine import AuditEventEngine
except Exception:
    AuditEventEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.lineage_graph_engine import LineageGraphEngine
except Exception:
    LineageGraphEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.evidence_packager_engine import EvidencePackagerEngine
except Exception:
    EvidencePackagerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.compliance_tracer_engine import ComplianceTracerEngine
except Exception:
    ComplianceTracerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.change_detector_engine import ChangeDetectorEngine
except Exception:
    ChangeDetectorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.compliance_checker import ComplianceCheckerEngine
except Exception:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.audit_trail_lineage.audit_trail_pipeline import AuditTrailPipelineEngine
except Exception:
    AuditTrailPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.audit_trail_lineage.config import get_config
except Exception:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None


def get_version() -> str:
    """Return the current version string for AGENT-MRV-030.

    Returns:
        Semantic version string (e.g., ``'1.0.0'``).

    Example:
        >>> get_version()
        '1.0.0'
    """
    return VERSION


def get_agent_info() -> dict:
    """Return metadata dictionary describing this agent.

    Returns:
        Dictionary with keys ``agent_id``, ``component``, ``version``,
        ``table_prefix``, ``package``, ``scope``, ``role``,
        ``engines``, ``compliance_frameworks``, and
        ``capabilities``.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-MRV-X-042'
    """
    return {
        "agent_id": AGENT_ID,
        "component": AGENT_COMPONENT,
        "version": VERSION,
        "table_prefix": TABLE_PREFIX,
        "package": "greenlang.audit_trail_lineage",
        "scope": "Scope 1, 2, and 3",
        "role": "Cross-Cutting -- Audit Trail & Calculation Lineage",
        "engines": [
            "AuditEventEngine",
            "LineageGraphEngine",
            "EvidencePackagerEngine",
            "ComplianceTracerEngine",
            "ChangeDetectorEngine",
            "ComplianceCheckerEngine",
            "AuditTrailPipelineEngine",
        ],
        "compliance_frameworks": [
            "GHG Protocol Corporate Standard",
            "GHG Protocol Scope 3 Standard",
            "ISO 14064-1:2018",
            "ISO 14064-3:2019",
            "CSRD ESRS E1",
            "California SB 253",
            "EU CBAM",
            "CDP Climate Change",
            "TCFD Recommendations",
            "PCAF Global Standard",
            "SBTi",
        ],
        "capabilities": [
            "Immutable SHA-256 hash-chained audit events",
            "Calculation lineage DAG (forward & backward traversal)",
            "Evidence packaging for third-party verification",
            "Regulatory requirement traceability mapping",
            "Recalculation change detection and version comparison",
            "Multi-framework audit trail compliance validation",
            "10-stage orchestration pipeline",
        ],
    }
