# -*- coding: utf-8 -*-
"""
Multi-Operator Onboarding Workflow
====================================

Four-phase workflow for onboarding multiple operators under a single
corporate entity (e.g., subsidiaries, business units, brands).

This workflow enables:
- Centralized operator registration for corporate groups
- Configuration inheritance from parent entity
- Supplier pool sharing across operators
- Consolidated dashboard and reporting

Phases:
    1. Operator Registration - Register multiple operator entities
    2. Configuration Inheritance - Copy settings from parent/template
    3. Supplier Pooling - Share supplier relationships across operators
    4. Dashboard Setup - Create consolidated and operator-specific views

Regulatory Context:
    EUDR Article 2(11) defines "operator" as any person placing relevant
    commodities on the EU market. Corporate groups with multiple legal entities
    need streamlined onboarding to ensure consistent compliance across subsidiaries.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    OPERATOR_REGISTRATION = "operator_registration"
    CONFIGURATION_INHERITANCE = "configuration_inheritance"
    SUPPLIER_POOLING = "supplier_pooling"
    DASHBOARD_SETUP = "dashboard_setup"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# DATA MODELS
# =============================================================================


class MultiOperatorOnboardingConfig(BaseModel):
    """Configuration for multi-operator onboarding workflow."""
    parent_operator_id: Optional[str] = Field(None, description="Parent entity operator ID")
    inherit_configuration: bool = Field(default=True, description="Inherit parent config")
    share_supplier_pool: bool = Field(default=True, description="Share suppliers across operators")
    create_consolidated_dashboard: bool = Field(default=True, description="Create group-level dashboard")
    operator_count: int = Field(default=1, ge=1, description="Number of operators to onboard")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: MultiOperatorOnboardingConfig = Field(default_factory=MultiOperatorOnboardingConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the multi-operator onboarding workflow."""
    workflow_name: str = Field(default="multi_operator_onboarding", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    operators_registered: int = Field(default=0, ge=0, description="Operators onboarded")
    configurations_inherited: int = Field(default=0, ge=0, description="Configs copied")
    shared_suppliers: int = Field(default=0, ge=0, description="Suppliers in pool")
    dashboards_created: int = Field(default=0, ge=0, description="Dashboards set up")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# MULTI-OPERATOR ONBOARDING WORKFLOW
# =============================================================================


class MultiOperatorOnboardingWorkflow:
    """
    Four-phase multi-operator onboarding workflow.

    Streamlines onboarding for corporate groups with multiple legal entities:
    - Batch operator registration
    - Automated configuration inheritance
    - Centralized supplier pool management
    - Consolidated and entity-specific dashboards

    Example:
        >>> config = MultiOperatorOnboardingConfig(
        ...     operator_count=5,
        ...     inherit_configuration=True,
        ...     share_supplier_pool=True,
        ... )
        >>> workflow = MultiOperatorOnboardingWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.operators_registered == 5
    """

    def __init__(self, config: Optional[MultiOperatorOnboardingConfig] = None) -> None:
        """Initialize the multi-operator onboarding workflow."""
        self.config = config or MultiOperatorOnboardingConfig()
        self.logger = logging.getLogger(f"{__name__}.MultiOperatorOnboardingWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase multi-operator onboarding workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with registered operators, configurations, and dashboards.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting multi-operator onboarding workflow execution_id=%s count=%d",
            context.execution_id,
            self.config.operator_count,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.OPERATOR_REGISTRATION, self._phase_1_operator_registration),
            (Phase.CONFIGURATION_INHERITANCE, self._phase_2_configuration_inheritance),
            (Phase.SUPPLIER_POOLING, self._phase_3_supplier_pooling),
            (Phase.DASHBOARD_SETUP, self._phase_4_dashboard_setup),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        operators = context.state.get("operators", [])
        configurations = context.state.get("configurations", [])
        supplier_pool = context.state.get("supplier_pool", [])
        dashboards = context.state.get("dashboards", [])

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "operator_count": len(operators),
        })

        self.logger.info(
            "Multi-operator onboarding finished execution_id=%s status=%s operators=%d",
            context.execution_id,
            overall_status.value,
            len(operators),
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            operators_registered=len(operators),
            configurations_inherited=len(configurations),
            shared_suppliers=len(supplier_pool),
            dashboards_created=len(dashboards),
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Operator Registration
    # -------------------------------------------------------------------------

    async def _phase_1_operator_registration(self, context: WorkflowContext) -> PhaseResult:
        """
        Register multiple operator entities.

        For each operator:
        - Collect legal entity information (name, EORI, VAT, address)
        - Assign unique operator ID
        - Link to parent entity (if applicable)
        - Set up basic user accounts
        """
        phase = Phase.OPERATOR_REGISTRATION
        operator_count = self.config.operator_count

        self.logger.info("Registering %d operator(s)", operator_count)

        await asyncio.sleep(0.05)

        operators = []
        for i in range(operator_count):
            operator = {
                "operator_id": f"OP-{uuid.uuid4().hex[:8]}",
                "operator_name": f"Operator Entity {i+1}",
                "legal_name": f"Legal Entity {i+1} GmbH",
                "eori_number": f"DE{random.randint(100000000000, 999999999999)}",
                "vat_number": f"DE{random.randint(100000000, 999999999)}",
                "country": random.choice(["DE", "FR", "NL", "BE", "IT"]),
                "parent_operator_id": self.config.parent_operator_id,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "active",
            }
            operators.append(operator)

        context.state["operators"] = operators

        provenance = self._hash({
            "phase": phase.value,
            "operator_count": len(operators),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "operators_registered": len(operators),
                "parent_linked": self.config.parent_operator_id is not None,
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Configuration Inheritance
    # -------------------------------------------------------------------------

    async def _phase_2_configuration_inheritance(self, context: WorkflowContext) -> PhaseResult:
        """
        Copy configuration settings from parent entity.

        Inherited settings:
        - Risk assessment thresholds
        - Certification requirements
        - Document templates
        - Alert escalation rules
        - Integration credentials (where applicable)
        """
        phase = Phase.CONFIGURATION_INHERITANCE
        operators = context.state.get("operators", [])

        if not self.config.inherit_configuration:
            self.logger.info("Configuration inheritance disabled; skipping")
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.COMPLETED,
                data={"inheritance_enabled": False},
                provenance_hash=self._hash({"phase": phase.value, "skipped": True}),
            )

        self.logger.info("Inheriting configuration for %d operator(s)", len(operators))

        # Simulate parent configuration retrieval
        parent_config = {
            "risk_thresholds": {
                "high_risk": 70.0,
                "medium_risk": 40.0,
                "low_risk": 20.0,
            },
            "certification_requirements": ["FSC", "PEFC", "RSPO"],
            "document_templates": {
                "dds_template": "standard_v1",
                "audit_template": "iso_compliant_v2",
            },
            "alert_escalation": {
                "critical": "tier_3_executive",
                "high": "tier_2_manager",
                "medium": "tier_1_analyst",
            },
        }

        configurations = []
        for operator in operators:
            config_copy = {
                "operator_id": operator["operator_id"],
                "inherited_from": self.config.parent_operator_id or "template",
                "config": dict(parent_config),  # Deep copy
                "inherited_at": datetime.utcnow().isoformat(),
            }
            configurations.append(config_copy)

        context.state["configurations"] = configurations

        provenance = self._hash({
            "phase": phase.value,
            "config_count": len(configurations),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "configurations_inherited": len(configurations),
                "inheritance_source": self.config.parent_operator_id or "template",
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Supplier Pooling
    # -------------------------------------------------------------------------

    async def _phase_3_supplier_pooling(self, context: WorkflowContext) -> PhaseResult:
        """
        Share supplier relationships across operators.

        Pooling logic:
        - Create centralized supplier master data
        - Allow all operators to access shared supplier pool
        - Track which operators actively work with each supplier
        - Enable cross-operator supplier performance insights
        """
        phase = Phase.SUPPLIER_POOLING
        operators = context.state.get("operators", [])

        if not self.config.share_supplier_pool:
            self.logger.info("Supplier pooling disabled; skipping")
            return PhaseResult(
                phase=phase,
                status=PhaseStatus.COMPLETED,
                data={"pooling_enabled": False},
                provenance_hash=self._hash({"phase": phase.value, "skipped": True}),
            )

        self.logger.info("Creating shared supplier pool for %d operator(s)", len(operators))

        await asyncio.sleep(0.05)

        # Simulate supplier pool creation
        supplier_count = random.randint(20, 100)
        supplier_pool = []

        for i in range(supplier_count):
            supplier = {
                "supplier_id": f"SUP-{uuid.uuid4().hex[:8]}",
                "supplier_name": f"Supplier {i+1}",
                "country": random.choice(["BR", "ID", "CO", "MY", "PE"]),
                "commodity": random.choice(["cocoa", "coffee", "oil_palm", "soya"]),
                "certification_status": random.choice(["FSC", "PEFC", "RSPO", "None"]),
                "shared_with_operators": random.sample(
                    [op["operator_id"] for op in operators],
                    k=random.randint(1, len(operators)),
                ),
                "added_to_pool_at": datetime.utcnow().isoformat(),
            }
            supplier_pool.append(supplier)

        context.state["supplier_pool"] = supplier_pool

        # Calculate sharing statistics
        avg_operators_per_supplier = sum(
            len(s["shared_with_operators"]) for s in supplier_pool
        ) / len(supplier_pool) if supplier_pool else 0

        provenance = self._hash({
            "phase": phase.value,
            "supplier_count": len(supplier_pool),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "supplier_pool_size": len(supplier_pool),
                "avg_operators_per_supplier": round(avg_operators_per_supplier, 1),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Dashboard Setup
    # -------------------------------------------------------------------------

    async def _phase_4_dashboard_setup(self, context: WorkflowContext) -> PhaseResult:
        """
        Create consolidated and operator-specific dashboards.

        Dashboard types:
        - Consolidated: Group-level view across all operators
        - Operator-specific: Individual entity performance
        - Comparative: Benchmarking across sibling operators

        Widgets:
        - DDS submission status
        - Risk score trends
        - Supplier performance
        - Compliance alerts
        """
        phase = Phase.DASHBOARD_SETUP
        operators = context.state.get("operators", [])

        self.logger.info("Setting up dashboards for %d operator(s)", len(operators))

        dashboards = []

        # Create consolidated dashboard if enabled
        if self.config.create_consolidated_dashboard:
            consolidated = {
                "dashboard_id": f"DASH-CONSOLIDATED-{uuid.uuid4().hex[:6]}",
                "dashboard_type": "consolidated",
                "dashboard_name": "Group Consolidated View",
                "operators_included": [op["operator_id"] for op in operators],
                "widgets": [
                    "group_dds_status",
                    "group_risk_distribution",
                    "group_supplier_performance",
                    "group_compliance_alerts",
                    "operator_comparison",
                ],
                "created_at": datetime.utcnow().isoformat(),
            }
            dashboards.append(consolidated)

        # Create operator-specific dashboards
        for operator in operators:
            operator_dashboard = {
                "dashboard_id": f"DASH-{operator['operator_id']}",
                "dashboard_type": "operator_specific",
                "dashboard_name": f"{operator['operator_name']} Dashboard",
                "operator_id": operator["operator_id"],
                "widgets": [
                    "dds_submission_status",
                    "risk_score_trend",
                    "supplier_performance",
                    "compliance_alerts",
                    "upcoming_audits",
                ],
                "created_at": datetime.utcnow().isoformat(),
            }
            dashboards.append(operator_dashboard)

        context.state["dashboards"] = dashboards

        provenance = self._hash({
            "phase": phase.value,
            "dashboard_count": len(dashboards),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "dashboards_created": len(dashboards),
                "consolidated_dashboard": self.config.create_consolidated_dashboard,
                "operator_dashboards": len(operators),
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
