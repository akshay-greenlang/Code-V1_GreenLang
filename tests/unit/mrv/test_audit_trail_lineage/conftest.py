# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-030: Audit Trail & Lineage Agent.

Provides shared test fixtures for:
- Sample organization/reporting identifiers
- Sample audit event, lineage node, lineage edge dictionaries
- Sample evidence request, change event dictionaries
- Fresh engine instances (reset before each test) for all 7 engines
- Config fixtures and mock objects

Usage:
    def test_something(audit_event_engine, sample_audit_event):
        result = audit_event_engine.record_event(**sample_audit_event)
        assert result["success"] is True

Author: GL-TestEngineer
Date: March 2026
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, Mock

import pytest

# ---------------------------------------------------------------------------
# Graceful imports for engine classes
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.audit_event_engine import AuditEventEngine
    AUDIT_EVENT_ENGINE_AVAILABLE = True
except ImportError:
    AUDIT_EVENT_ENGINE_AVAILABLE = False
    AuditEventEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.lineage_graph_engine import LineageGraphEngine
    LINEAGE_GRAPH_ENGINE_AVAILABLE = True
except ImportError:
    LINEAGE_GRAPH_ENGINE_AVAILABLE = False
    LineageGraphEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.evidence_packager_engine import EvidencePackagerEngine
    EVIDENCE_PACKAGER_ENGINE_AVAILABLE = True
except ImportError:
    EVIDENCE_PACKAGER_ENGINE_AVAILABLE = False
    EvidencePackagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.compliance_tracer_engine import ComplianceTracerEngine
    COMPLIANCE_TRACER_ENGINE_AVAILABLE = True
except ImportError:
    COMPLIANCE_TRACER_ENGINE_AVAILABLE = False
    ComplianceTracerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.change_detector_engine import ChangeDetectorEngine
    CHANGE_DETECTOR_ENGINE_AVAILABLE = True
except ImportError:
    CHANGE_DETECTOR_ENGINE_AVAILABLE = False
    ChangeDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.compliance_checker import ComplianceCheckerEngine
    COMPLIANCE_CHECKER_ENGINE_AVAILABLE = True
except ImportError:
    COMPLIANCE_CHECKER_ENGINE_AVAILABLE = False
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.audit_trail_pipeline import AuditTrailPipelineEngine
    PIPELINE_ENGINE_AVAILABLE = True
except ImportError:
    PIPELINE_ENGINE_AVAILABLE = False
    AuditTrailPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.audit_trail_lineage.config import (
        AuditTrailLineageConfig,
        reset_config,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    AuditTrailLineageConfig = None  # type: ignore[assignment, misc]
    reset_config = None  # type: ignore[assignment, misc]


# ============================================================================
# IDENTIFIER FIXTURES
# ============================================================================

@pytest.fixture
def sample_organization_id() -> str:
    """Sample organization identifier for testing."""
    return "org-test-001"


@pytest.fixture
def sample_reporting_year() -> int:
    """Sample reporting year for testing."""
    return 2025


@pytest.fixture
def sample_calculation_id() -> str:
    """Sample calculation identifier for testing."""
    return "calc-test-abc123"


@pytest.fixture
def sample_event_id() -> str:
    """Sample event identifier for testing."""
    return f"atl-{uuid.uuid4().hex}"


# ============================================================================
# AUDIT EVENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_audit_event(
    sample_organization_id: str,
    sample_reporting_year: int,
    sample_calculation_id: str,
) -> Dict[str, Any]:
    """Sample audit event dictionary with all fields populated."""
    return {
        "event_type": "CALCULATION_COMPLETED",
        "agent_id": "GL-MRV-S1-001",
        "scope": "scope_1",
        "category": None,
        "organization_id": sample_organization_id,
        "reporting_year": sample_reporting_year,
        "calculation_id": sample_calculation_id,
        "payload": {
            "total_co2e_tonnes": "1234.56",
            "fuel_type": "diesel",
            "method": "tier_1",
        },
        "data_quality_score": Decimal("0.85"),
        "metadata": {
            "source": "unit_test",
            "engine": "stationary_combustion",
        },
    }


@pytest.fixture
def sample_audit_event_scope3(
    sample_organization_id: str,
    sample_reporting_year: int,
) -> Dict[str, Any]:
    """Sample Scope 3 audit event dictionary."""
    return {
        "event_type": "CALCULATION_COMPLETED",
        "agent_id": "GL-MRV-S3-006",
        "scope": "scope_3",
        "category": 6,
        "organization_id": sample_organization_id,
        "reporting_year": sample_reporting_year,
        "calculation_id": "calc-bt-001",
        "payload": {
            "total_co2e_tonnes": "45.89",
            "travel_type": "air",
        },
        "data_quality_score": Decimal("0.72"),
    }


# ============================================================================
# LINEAGE FIXTURES
# ============================================================================

@pytest.fixture
def sample_lineage_node() -> Dict[str, Any]:
    """Sample lineage graph node dictionary."""
    return {
        "node_id": "node-ef-diesel-001",
        "node_type": "emission_factor",
        "label": "Diesel Emission Factor (DEFRA 2024)",
        "scope": "scope_1",
        "agent_id": "GL-MRV-S1-001",
        "organization_id": "org-test-001",
        "reporting_year": 2025,
        "metadata": {
            "source": "DEFRA",
            "year": 2024,
            "value": "2.68",
            "unit": "kg_co2e_per_litre",
        },
    }


@pytest.fixture
def sample_lineage_edge() -> Dict[str, Any]:
    """Sample lineage graph edge dictionary."""
    return {
        "source_node_id": "node-activity-001",
        "target_node_id": "node-calc-001",
        "edge_type": "feeds_into",
        "label": "Activity data used in calculation",
        "metadata": {
            "weight": 1.0,
            "transformation": "multiply",
        },
    }


# ============================================================================
# EVIDENCE FIXTURES
# ============================================================================

@pytest.fixture
def sample_evidence_request(
    sample_organization_id: str,
    sample_reporting_year: int,
) -> Dict[str, Any]:
    """Sample evidence package request dictionary."""
    return {
        "organization_id": sample_organization_id,
        "reporting_year": sample_reporting_year,
        "scope": "scope_1",
        "assurance_level": "limited",
        "frameworks": ["GHG_PROTOCOL", "ISO_14064"],
        "include_audit_events": True,
        "include_lineage_graph": True,
        "include_change_log": True,
    }


# ============================================================================
# CHANGE EVENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_change_event(
    sample_organization_id: str,
    sample_reporting_year: int,
) -> Dict[str, Any]:
    """Sample change event dictionary."""
    return {
        "change_type": "EMISSION_FACTOR_UPDATE",
        "organization_id": sample_organization_id,
        "reporting_year": sample_reporting_year,
        "scope": "scope_1",
        "agent_id": "GL-MRV-S1-001",
        "previous_value": {"ef_value": "2.68"},
        "new_value": {"ef_value": "2.71"},
        "reason": "DEFRA 2025 update published",
        "metadata": {"source": "DEFRA", "effective_date": "2025-01-01"},
    }


# ============================================================================
# ENGINE FIXTURES (Fresh instance, reset before each test)
# ============================================================================

@pytest.fixture
def audit_event_engine():
    """Create a fresh AuditEventEngine (singleton reset before each test)."""
    if not AUDIT_EVENT_ENGINE_AVAILABLE:
        pytest.skip("AuditEventEngine not available")
    engine = AuditEventEngine()
    engine.reset()
    yield engine
    engine.reset()
    # Reset singleton so next test gets clean state
    AuditEventEngine._instance = None
    AuditEventEngine._AuditEventEngine__init = False  # type: ignore[attr-defined]


@pytest.fixture
def lineage_graph_engine():
    """Create a fresh LineageGraphEngine (reset before each test)."""
    if not LINEAGE_GRAPH_ENGINE_AVAILABLE:
        pytest.skip("LineageGraphEngine not available")
    engine = LineageGraphEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture
def evidence_packager_engine():
    """Create a fresh EvidencePackagerEngine (reset before each test)."""
    if not EVIDENCE_PACKAGER_ENGINE_AVAILABLE:
        pytest.skip("EvidencePackagerEngine not available")
    engine = EvidencePackagerEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture
def compliance_tracer_engine():
    """Create a fresh ComplianceTracerEngine (reset before each test)."""
    if not COMPLIANCE_TRACER_ENGINE_AVAILABLE:
        pytest.skip("ComplianceTracerEngine not available")
    engine = ComplianceTracerEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture
def change_detector_engine():
    """Create a fresh ChangeDetectorEngine (reset before each test)."""
    if not CHANGE_DETECTOR_ENGINE_AVAILABLE:
        pytest.skip("ChangeDetectorEngine not available")
    engine = ChangeDetectorEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture
def compliance_checker_engine():
    """Create a fresh ComplianceCheckerEngine (reset before each test)."""
    if not COMPLIANCE_CHECKER_ENGINE_AVAILABLE:
        pytest.skip("ComplianceCheckerEngine not available")
    engine = ComplianceCheckerEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture
def pipeline_engine():
    """Create a fresh AuditTrailPipelineEngine (reset before each test)."""
    if not PIPELINE_ENGINE_AVAILABLE:
        pytest.skip("AuditTrailPipelineEngine not available")
    engine = AuditTrailPipelineEngine()
    engine.reset()
    yield engine
    engine.reset()


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset config singleton before each test to ensure isolation."""
    if CONFIG_AVAILABLE:
        reset_config()
    yield
    if CONFIG_AVAILABLE:
        reset_config()
