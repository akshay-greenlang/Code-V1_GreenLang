# -*- coding: utf-8 -*-
"""
Tests for PACK-030 OrchestratorIntegration (DAG pipeline registration).

Covers: OrchestratorIntegration instantiation, PACK030_CAPABILITIES
constants, PACK030_DEPENDENCIES declarations, pipeline enums, Pydantic
models (OrchestratorConfig, PackRegistration, HealthReport,
OrchestrationRequest, OrchestrationResponse), config defaults,
DAG capability registry structure, and health status lifecycle.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    OrchestratorIntegration,
    OrchestratorConfig,
    PackStatus,
    WorkflowStatus,
    OrchestratorHealthStatus,
    PACK030_CAPABILITIES,
    PACK030_DEPENDENCIES,
    PackRegistration,
    HealthReport,
    OrchestrationRequest,
    OrchestrationResponse,
)


# ========================================================================
# PackStatus Enum
# ========================================================================


class TestPackStatusEnum:
    """Validate PackStatus enum values."""

    def test_pack_status_values(self):
        expected = {"registered", "active", "degraded", "maintenance", "offline"}
        actual = {s.value for s in PackStatus}
        assert actual == expected

    def test_registered_status(self):
        assert PackStatus.REGISTERED.value == "registered"

    def test_active_status(self):
        assert PackStatus.ACTIVE.value == "active"

    def test_degraded_status(self):
        assert PackStatus.DEGRADED.value == "degraded"

    def test_maintenance_status(self):
        assert PackStatus.MAINTENANCE.value == "maintenance"

    def test_offline_status(self):
        assert PackStatus.OFFLINE.value == "offline"


# ========================================================================
# WorkflowStatus Enum
# ========================================================================


class TestWorkflowStatusEnum:
    """Validate WorkflowStatus enum values."""

    def test_workflow_status_values(self):
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        actual = {s.value for s in WorkflowStatus}
        assert actual == expected

    def test_pending_status(self):
        assert WorkflowStatus.PENDING.value == "pending"

    def test_completed_status(self):
        assert WorkflowStatus.COMPLETED.value == "completed"

    def test_failed_status(self):
        assert WorkflowStatus.FAILED.value == "failed"

    def test_cancelled_status(self):
        assert WorkflowStatus.CANCELLED.value == "cancelled"


# ========================================================================
# HealthStatus Enum
# ========================================================================


class TestHealthStatusEnum:
    """Validate HealthStatus enum values."""

    def test_health_status_values(self):
        expected = {"healthy", "degraded", "unhealthy", "unknown"}
        actual = {s.value for s in OrchestratorHealthStatus}
        assert actual == expected

    def test_healthy_status(self):
        assert OrchestratorHealthStatus.HEALTHY.value == "healthy"

    def test_unknown_status(self):
        assert OrchestratorHealthStatus.UNKNOWN.value == "unknown"


# ========================================================================
# PACK030_CAPABILITIES Constants
# ========================================================================


class TestPACK030Capabilities:
    """Validate PACK030_CAPABILITIES registry structure."""

    def test_capabilities_is_dict(self):
        assert isinstance(PACK030_CAPABILITIES, dict)

    def test_capabilities_has_engines(self):
        assert "engines" in PACK030_CAPABILITIES

    def test_capabilities_has_workflows(self):
        assert "workflows" in PACK030_CAPABILITIES

    def test_capabilities_has_10_engines(self):
        engines = PACK030_CAPABILITIES["engines"]
        assert len(engines) == 10

    def test_capabilities_has_8_workflows(self):
        workflows = PACK030_CAPABILITIES["workflows"]
        assert len(workflows) == 8

    def test_capabilities_has_frameworks_supported(self):
        assert "frameworks_supported" in PACK030_CAPABILITIES

    def test_capabilities_has_7_frameworks(self):
        frameworks = PACK030_CAPABILITIES["frameworks_supported"]
        assert len(frameworks) == 7

    def test_capabilities_frameworks_include_sbti(self):
        assert "SBTi" in PACK030_CAPABILITIES["frameworks_supported"]

    def test_capabilities_frameworks_include_cdp(self):
        assert "CDP" in PACK030_CAPABILITIES["frameworks_supported"]

    def test_capabilities_frameworks_include_tcfd(self):
        assert "TCFD" in PACK030_CAPABILITIES["frameworks_supported"]

    def test_capabilities_frameworks_include_csrd(self):
        assert "CSRD" in PACK030_CAPABILITIES["frameworks_supported"]

    def test_capabilities_frameworks_include_sec(self):
        assert "SEC" in PACK030_CAPABILITIES["frameworks_supported"]

    def test_capabilities_has_output_formats(self):
        assert "output_formats" in PACK030_CAPABILITIES

    def test_capabilities_has_languages(self):
        assert "languages" in PACK030_CAPABILITIES

    def test_each_engine_has_name(self):
        for engine_id, engine_info in PACK030_CAPABILITIES["engines"].items():
            assert "name" in engine_info, f"Engine {engine_id} missing 'name'"

    def test_each_workflow_has_name(self):
        for wf_id, wf_info in PACK030_CAPABILITIES["workflows"].items():
            assert "name" in wf_info, f"Workflow {wf_id} missing 'name'"


# ========================================================================
# PACK030_DEPENDENCIES Constants
# ========================================================================


class TestPACK030Dependencies:
    """Validate PACK030_DEPENDENCIES declarations."""

    def test_dependencies_is_list(self):
        assert isinstance(PACK030_DEPENDENCIES, list)

    def test_dependencies_has_entries(self):
        assert len(PACK030_DEPENDENCIES) >= 4

    def test_pack021_dependency_present(self):
        pack_ids = [d.get("pack_id") for d in PACK030_DEPENDENCIES]
        assert "PACK-021" in pack_ids

    def test_pack022_dependency_present(self):
        pack_ids = [d.get("pack_id") for d in PACK030_DEPENDENCIES]
        assert "PACK-022" in pack_ids

    def test_pack028_dependency_present(self):
        pack_ids = [d.get("pack_id") for d in PACK030_DEPENDENCIES]
        assert "PACK-028" in pack_ids

    def test_pack029_dependency_present(self):
        pack_ids = [d.get("pack_id") for d in PACK030_DEPENDENCIES]
        assert "PACK-029" in pack_ids

    def test_each_dependency_has_name(self):
        for dep in PACK030_DEPENDENCIES:
            assert "name" in dep, f"Dependency missing 'name': {dep}"


# ========================================================================
# OrchestratorConfig Defaults
# ========================================================================


class TestOrchestratorConfig:
    """Validate OrchestratorConfig model and defaults."""

    def test_default_instantiation(self):
        config = OrchestratorConfig()
        assert config is not None

    def test_default_pack_id(self):
        config = OrchestratorConfig()
        assert config.pack_id == "PACK-030"

    def test_default_pack_name(self):
        config = OrchestratorConfig()
        assert config.pack_name == "Net Zero Reporting Pack"

    def test_default_pack_version(self):
        config = OrchestratorConfig()
        assert config.pack_version == "1.0.0"

    def test_default_heartbeat_interval(self):
        config = OrchestratorConfig()
        assert config.heartbeat_interval_seconds == 60

    def test_default_auto_registration(self):
        config = OrchestratorConfig()
        assert config.enable_auto_registration is True

    def test_custom_orchestrator_url(self):
        config = OrchestratorConfig(orchestrator_url="http://localhost:8080")
        assert config.orchestrator_url == "http://localhost:8080"


# ========================================================================
# Orchestrator Instantiation
# ========================================================================


class TestOrchestratorInstantiation:
    """Tests for OrchestratorIntegration creation."""

    def test_default_instantiation(self):
        orch = OrchestratorIntegration()
        assert orch is not None

    def test_with_config(self):
        config = OrchestratorConfig()
        orch = OrchestratorIntegration(config=config)
        assert orch is not None

    def test_orchestrator_class_name(self):
        assert OrchestratorIntegration.__name__ == "OrchestratorIntegration"

    def test_orchestrator_has_docstring(self):
        assert OrchestratorIntegration.__doc__ is not None

    def test_orchestrator_has_register_pack(self):
        orch = OrchestratorIntegration()
        assert callable(getattr(orch, "register_pack", None))

    def test_orchestrator_has_report_health(self):
        orch = OrchestratorIntegration()
        assert callable(getattr(orch, "report_health", None))

    def test_orchestrator_config_accessible(self):
        config = OrchestratorConfig(pack_id="PACK-030")
        orch = OrchestratorIntegration(config=config)
        assert orch.config.pack_id == "PACK-030"


# ========================================================================
# Pydantic Models
# ========================================================================


class TestOrchestratorModels:
    """Validate orchestrator Pydantic models."""

    def test_pack_registration_constructs(self):
        pr = PackRegistration()
        assert pr is not None
        assert pr.pack_id == "PACK-030"

    def test_pack_registration_default_status(self):
        pr = PackRegistration()
        assert pr.status == PackStatus.REGISTERED

    def test_pack_registration_has_registration_id(self):
        pr = PackRegistration()
        assert pr.registration_id is not None
        assert len(pr.registration_id) > 0

    def test_health_report_constructs(self):
        hr = HealthReport()
        assert hr is not None
        assert hr.pack_id == "PACK-030"

    def test_health_report_default_status(self):
        hr = HealthReport()
        assert hr.status == OrchestratorHealthStatus.HEALTHY

    def test_health_report_default_engine_counts(self):
        hr = HealthReport()
        assert hr.engines_total == 10
        assert hr.integrations_total == 12

    def test_orchestration_request_constructs(self):
        req = OrchestrationRequest(workflow_name="sbti_progress_report")
        assert req.workflow_name == "sbti_progress_report"

    def test_orchestration_request_default_priority(self):
        req = OrchestrationRequest()
        assert req.priority == 5

    def test_orchestration_response_constructs(self):
        resp = OrchestrationResponse()
        assert resp is not None
        assert resp.status == WorkflowStatus.PENDING

    def test_orchestration_response_has_response_id(self):
        resp = OrchestrationResponse()
        assert resp.response_id is not None
        assert len(resp.response_id) > 0
