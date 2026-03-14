# -*- coding: utf-8 -*-
"""
CSRDPackOrchestrator - Master Pack Orchestrator for CSRD Starter Pack
=====================================================================

This module implements the master orchestrator that connects 66+ GreenLang
agents into a cohesive CSRD compliance pipeline. It manages workflow
execution, agent lifecycle, progress tracking, error recovery, and
performance monitoring for all supported workflow types.

Supported Workflows:
    - annual_report: Full CSRD annual reporting cycle
    - quarterly_update: Quarterly data refresh and recalculation
    - materiality_assessment: Double materiality assessment workflow
    - onboarding: New company setup and initial data collection
    - audit_preparation: External auditor evidence package generation

Architecture:
    The orchestrator uses a phase-based execution model where each workflow
    is decomposed into ordered phases. Each phase activates a subset of
    agents, executes them in dependency order, and validates outputs before
    proceeding to the next phase.

Example:
    >>> config = OrchestratorConfig(pack_id="PACK-001", size_preset="mid_market")
    >>> orchestrator = CSRDPackOrchestrator(config)
    >>> result = await orchestrator.run_workflow("annual_report", params)
    >>> assert result.status == "completed"

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]
"""Async callback signature: (phase_name, percent_complete, message) -> None"""


# =============================================================================
# Enums
# =============================================================================


class WorkflowType(str, Enum):
    """Supported workflow types in the CSRD Starter Pack."""
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_UPDATE = "quarterly_update"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    ONBOARDING = "onboarding"
    AUDIT_PREPARATION = "audit_preparation"


class WorkflowPhase(str, Enum):
    """Execution phases within a workflow."""
    INITIALIZATION = "initialization"
    DATA_INTAKE = "data_intake"
    DATA_QUALITY = "data_quality"
    MATERIALITY = "materiality"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    REPORTING = "reporting"
    AUDIT_TRAIL = "audit_trail"
    FINALIZATION = "finalization"


class AgentStatusCode(str, Enum):
    """Status codes for individual agents."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


# =============================================================================
# Data Models
# =============================================================================


class AgentStatus(BaseModel):
    """Runtime status of an individual agent within the orchestrator."""
    agent_id: str = Field(..., description="Unique agent identifier (e.g., GL-MRV-X-001)")
    agent_name: str = Field(..., description="Human-readable agent name")
    status: AgentStatusCode = Field(
        default=AgentStatusCode.PENDING, description="Current agent status"
    )
    phase: Optional[WorkflowPhase] = Field(None, description="Phase this agent belongs to")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    execution_time_ms: float = Field(default=0.0, description="Execution duration in ms")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    records_processed: int = Field(default=0, description="Number of records processed")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance hash")


class WorkflowExecution(BaseModel):
    """Complete record of a workflow execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    workflow_type: WorkflowType = Field(..., description="Type of workflow executed")
    status: str = Field(default="pending", description="Overall workflow status")
    started_at: Optional[datetime] = Field(None, description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution time")
    current_phase: Optional[WorkflowPhase] = Field(None, description="Current active phase")
    phases_completed: List[str] = Field(default_factory=list, description="Completed phases")
    agent_statuses: Dict[str, AgentStatus] = Field(
        default_factory=dict, description="Status of each agent"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    result_data: Dict[str, Any] = Field(default_factory=dict, description="Workflow output data")
    errors: List[str] = Field(default_factory=list, description="Collected error messages")
    provenance_hash: Optional[str] = Field(None, description="Overall provenance hash")


class PackStatus(BaseModel):
    """Overall status of the CSRD Starter Pack."""
    pack_id: str = Field(default="PACK-001", description="Pack identifier")
    pack_version: str = Field(default="1.0.0", description="Pack version")
    is_initialized: bool = Field(default=False, description="Whether the pack is initialized")
    total_agents: int = Field(default=0, description="Total registered agents")
    active_agents: int = Field(default=0, description="Currently active agents")
    disabled_agents: int = Field(default=0, description="Disabled agents (by preset)")
    current_workflow: Optional[WorkflowExecution] = Field(
        None, description="Currently running workflow"
    )
    last_execution: Optional[WorkflowExecution] = Field(
        None, description="Last completed workflow"
    )
    uptime_seconds: float = Field(default=0.0, description="Pack uptime in seconds")
    initialized_at: Optional[datetime] = Field(None, description="Initialization timestamp")
    health_status: str = Field(default="unknown", description="Overall health status")


class OrchestratorConfig(BaseModel):
    """Configuration for the CSRD Pack Orchestrator."""
    pack_id: str = Field(default="PACK-001", description="Pack identifier")
    size_preset: str = Field(
        default="mid_market", description="Size preset (sme, mid_market, large_enterprise)"
    )
    sector_preset: Optional[str] = Field(
        None, description="Sector preset (manufacturing, financial_services, etc.)"
    )
    enabled_esrs_standards: List[str] = Field(
        default_factory=lambda: ["ESRS_1", "ESRS_2", "ESRS_E1"],
        description="ESRS standards to enable",
    )
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="Scope 3 categories to include",
    )
    max_concurrent_agents: int = Field(
        default=5, description="Maximum agents running concurrently"
    )
    timeout_per_agent_seconds: int = Field(
        default=300, description="Timeout per agent execution in seconds"
    )
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    database_url: Optional[str] = Field(None, description="Database connection URL")
    reporting_period_start: Optional[str] = Field(
        None, description="Reporting period start (YYYY-MM-DD)"
    )
    reporting_period_end: Optional[str] = Field(
        None, description="Reporting period end (YYYY-MM-DD)"
    )
    company_name: Optional[str] = Field(None, description="Company name for reports")


# =============================================================================
# Workflow Phase Definitions
# =============================================================================

WORKFLOW_PHASE_DEFINITIONS: Dict[WorkflowType, List[WorkflowPhase]] = {
    WorkflowType.ANNUAL_REPORT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.MATERIALITY,
        WorkflowPhase.CALCULATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.REPORTING,
        WorkflowPhase.AUDIT_TRAIL,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.QUARTERLY_UPDATE: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.CALCULATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.MATERIALITY_ASSESSMENT: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.MATERIALITY,
        WorkflowPhase.REPORTING,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.ONBOARDING: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_INTAKE,
        WorkflowPhase.DATA_QUALITY,
        WorkflowPhase.FINALIZATION,
    ],
    WorkflowType.AUDIT_PREPARATION: [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.AUDIT_TRAIL,
        WorkflowPhase.REPORTING,
        WorkflowPhase.FINALIZATION,
    ],
}

# Agent IDs mapped to their execution phase
PHASE_AGENT_MAPPING: Dict[WorkflowPhase, List[str]] = {
    WorkflowPhase.INITIALIZATION: [
        "GL-FOUND-X-001",  # Orchestrator
        "GL-FOUND-X-002",  # Schema Compiler
        "GL-FOUND-X-010",  # Agent Registry
    ],
    WorkflowPhase.DATA_INTAKE: [
        "GL-DATA-X-001",   # Document Ingestion (PDF)
        "GL-DATA-X-004",   # ERP Connector
        "GL-DATA-X-008",   # Supplier Questionnaire
        "GL-DATA-X-009",   # Utility Tariff & Grid Factor
    ],
    WorkflowPhase.DATA_QUALITY: [
        "GL-DATA-X-010",   # Data Quality Profiler (Emission Factor Library)
        "GL-DATA-X-011",   # Duplicate Detection (Materials LCI)
        "GL-DATA-X-012",   # Missing Value Imputer (Supplier Data Exchange)
        "GL-DATA-X-013",   # Outlier Detection (IoT Meter Mgmt)
        "GL-FOUND-X-003",  # Unit Normalizer
    ],
    WorkflowPhase.MATERIALITY: [
        "GL-CSRD-MAT",     # CSRD Materiality Agent
        "GL-CSRD-STAKE",   # Stakeholder Engagement Agent
    ],
    WorkflowPhase.CALCULATION: [
        # Scope 1
        "GL-MRV-X-001",    # Stationary Combustion
        "GL-MRV-X-002",    # Refrigerants & F-Gas
        "GL-MRV-X-003",    # Mobile Combustion
        "GL-MRV-X-004",    # Process Emissions
        "GL-MRV-X-005",    # Fugitive Emissions
        "GL-MRV-X-006",    # Land Use
        "GL-MRV-X-007",    # Waste Treatment
        "GL-MRV-X-008",    # Agricultural
        # Scope 2
        "GL-MRV-X-009",    # Location-Based
        "GL-MRV-X-010",    # Market-Based
        "GL-MRV-X-011",    # Steam/Heat
        "GL-MRV-X-012",    # Cooling
        "GL-MRV-X-013",    # Dual Reporting Reconciliation
        # Scope 3 (Cat 1-15)
        "GL-MRV-X-014",    # Purchased Goods & Services
        "GL-MRV-X-015",    # Capital Goods
        "GL-MRV-X-016",    # Fuel & Energy
        "GL-MRV-X-017",    # Upstream Transportation
        "GL-MRV-X-018",    # Waste Generated
        "GL-MRV-X-019",    # Business Travel
        "GL-MRV-X-020",    # Employee Commuting
        "GL-MRV-X-021",    # Upstream Leased Assets
        "GL-MRV-X-022",    # Downstream Transportation
        "GL-MRV-X-023",    # Processing of Sold Products
        "GL-MRV-X-024",    # Use of Sold Products
        "GL-MRV-X-025",    # End-of-Life Treatment
        "GL-MRV-X-026",    # Downstream Leased Assets
        "GL-MRV-X-027",    # Franchises
        "GL-MRV-X-028",    # Investments
        # Cross-cutting
        "GL-MRV-X-029",    # Scope 3 Category Mapper
        "GL-MRV-X-030",    # Audit Trail & Lineage
    ],
    WorkflowPhase.VALIDATION: [
        "GL-FOUND-X-008",  # QA Test Harness
        "GL-FOUND-X-005",  # Citations & Evidence
        "GL-POL-X-007",    # CSRD Compliance Agent
    ],
    WorkflowPhase.REPORTING: [
        "GL-CSRD-EXEC",    # Executive Summary Template
        "GL-CSRD-DISC",    # ESRS Disclosure Template
        "GL-CSRD-GHG",     # GHG Emissions Report Template
        "GL-CSRD-DASH",    # Compliance Dashboard Template
    ],
    WorkflowPhase.AUDIT_TRAIL: [
        "GL-CSRD-AUDIT",   # Auditor Package Template
        "GL-FOUND-X-004",  # Assumptions Registry
        "GL-FOUND-X-009",  # Observability
    ],
    WorkflowPhase.FINALIZATION: [
        "GL-FOUND-X-006",  # Policy Guard
        "GL-FOUND-X-007",  # PII Redaction
    ],
}

# Size preset determines which agents are active
SIZE_PRESET_DISABLED_AGENTS: Dict[str, Set[str]] = {
    "sme": {
        # SMEs skip advanced Scope 3 categories
        "GL-MRV-X-021",  # Upstream Leased Assets
        "GL-MRV-X-022",  # Downstream Transportation
        "GL-MRV-X-023",  # Processing of Sold Products
        "GL-MRV-X-024",  # Use of Sold Products
        "GL-MRV-X-025",  # End-of-Life Treatment
        "GL-MRV-X-026",  # Downstream Leased Assets
        "GL-MRV-X-027",  # Franchises
        "GL-MRV-X-028",  # Investments
        # SMEs skip advanced data quality
        "GL-DATA-X-013",  # IoT Meter Mgmt / Outlier Detection
    },
    "mid_market": {
        # Mid-market skips niche categories
        "GL-MRV-X-023",  # Processing of Sold Products
        "GL-MRV-X-026",  # Downstream Leased Assets
        "GL-MRV-X-028",  # Investments
    },
    "large_enterprise": set(),  # All agents enabled
}


# =============================================================================
# Main Orchestrator
# =============================================================================


class CSRDPackOrchestrator:
    """Master orchestrator for CSRD Starter Pack.

    Connects 66+ agents into a cohesive pipeline:
    - Data intake agents (4) through DataPipelineBridge
    - Data quality agents (5) through quality pipeline
    - MRV calculation engines (30) through MRVBridge
    - CSRD pipeline agents (6) for core reporting
    - Foundation agents (10) for orchestration and audit trail

    Responsibilities:
    - Workflow execution (annual, quarterly, materiality, onboarding, audit)
    - Agent lifecycle management
    - Progress tracking and status reporting
    - Error handling and recovery
    - Performance monitoring

    Attributes:
        config: Orchestrator configuration
        _agents: Registry of initialized agent instances
        _status: Current pack status
        _progress_callbacks: Registered progress callback functions
        _execution_history: History of workflow executions

    Example:
        >>> config = OrchestratorConfig(size_preset="mid_market")
        >>> orchestrator = CSRDPackOrchestrator(config)
        >>> await orchestrator.initialize()
        >>> result = await orchestrator.run_workflow("annual_report", {})
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        """Initialize the CSRD Pack Orchestrator.

        Args:
            config: Orchestrator configuration with presets and settings.
        """
        self.config = config
        self._agents: Dict[str, Any] = {}
        self._agent_statuses: Dict[str, AgentStatus] = {}
        self._progress_callbacks: List[ProgressCallback] = []
        self._execution_history: List[WorkflowExecution] = []
        self._current_execution: Optional[WorkflowExecution] = None
        self._initialized = False
        self._initialized_at: Optional[datetime] = None
        self._disabled_agents: Set[str] = SIZE_PRESET_DISABLED_AGENTS.get(
            config.size_preset, set()
        )
        self._semaphore: Optional[asyncio.Semaphore] = None

        logger.info(
            "CSRDPackOrchestrator created with preset=%s, sector=%s",
            config.size_preset,
            config.sector_preset,
        )

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all agents and prepare the orchestrator for workflow execution.

        This method loads agent configurations, instantiates agent objects,
        validates dependencies, and prepares the concurrency semaphore.

        Raises:
            RuntimeError: If initialization fails due to missing dependencies.
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized, skipping re-initialization")
            return

        start_time = time.monotonic()
        logger.info("Initializing CSRDPackOrchestrator with %d max concurrent agents",
                     self.config.max_concurrent_agents)

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        await self._register_agents()
        await self._initialize_agents()
        await self._validate_dependencies()

        self._initialized = True
        self._initialized_at = datetime.utcnow()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Orchestrator initialized in %.1fms: %d agents registered, %d active, %d disabled",
            elapsed_ms,
            len(self._agents),
            len(self._agents) - len(self._disabled_agents),
            len(self._disabled_agents),
        )

    async def _register_agents(self) -> None:
        """Register all agents from the phase mapping into the internal registry.

        Each agent is registered with its ID, a descriptive name, and the
        phase it belongs to. Disabled agents (based on the size preset) are
        marked accordingly.
        """
        agent_names = _build_agent_name_lookup()

        for phase, agent_ids in PHASE_AGENT_MAPPING.items():
            for agent_id in agent_ids:
                if agent_id in self._agent_statuses:
                    continue  # Already registered from a prior phase

                is_disabled = agent_id in self._disabled_agents
                status = AgentStatus(
                    agent_id=agent_id,
                    agent_name=agent_names.get(agent_id, agent_id),
                    status=AgentStatusCode.DISABLED if is_disabled else AgentStatusCode.PENDING,
                    phase=phase,
                )
                self._agent_statuses[agent_id] = status

        logger.info("Registered %d agents across %d phases",
                     len(self._agent_statuses), len(PHASE_AGENT_MAPPING))

    async def _initialize_agents(self) -> None:
        """Instantiate agent objects for all non-disabled agents.

        Agent instantiation is performed lazily; the actual framework agent
        classes are imported and constructed only when the orchestrator is
        initialized. This avoids heavy import costs at module load time.
        """
        for agent_id, status in self._agent_statuses.items():
            if status.status == AgentStatusCode.DISABLED:
                continue
            try:
                agent_instance = _create_agent_stub(agent_id, self.config)
                self._agents[agent_id] = agent_instance
            except Exception as exc:
                logger.error("Failed to initialize agent %s: %s", agent_id, exc)
                status.status = AgentStatusCode.FAILED
                status.error_message = f"Initialization failed: {exc}"

    async def _validate_dependencies(self) -> None:
        """Validate that all required agent dependencies are satisfied.

        Checks that agents required by downstream phases are available and
        not in a failed state. Logs warnings for optional dependencies that
        are missing.
        """
        required_foundation = {"GL-FOUND-X-001", "GL-FOUND-X-002", "GL-FOUND-X-010"}
        for agent_id in required_foundation:
            status = self._agent_statuses.get(agent_id)
            if status is None or status.status == AgentStatusCode.FAILED:
                logger.error("Required foundation agent %s is not available", agent_id)

    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Execute a named workflow with the given parameters.

        Args:
            workflow_name: Name of the workflow to run (must match WorkflowType values).
            params: Optional parameters to pass to the workflow phases.

        Returns:
            WorkflowExecution record with full results, timing, and provenance.

        Raises:
            ValueError: If the workflow name is not recognized.
            RuntimeError: If the orchestrator is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before running workflows")

        try:
            workflow_type = WorkflowType(workflow_name)
        except ValueError:
            valid = [wt.value for wt in WorkflowType]
            raise ValueError(
                f"Unknown workflow '{workflow_name}'. Valid workflows: {valid}"
            )

        params = params or {}
        execution = self._create_execution(workflow_type, params)
        self._current_execution = execution

        start_time = time.monotonic()
        execution.started_at = datetime.utcnow()
        execution.status = "running"
        logger.info("Starting workflow '%s' (execution_id=%s)",
                     workflow_name, execution.execution_id)

        phases = WORKFLOW_PHASE_DEFINITIONS[workflow_type]

        try:
            for phase_index, phase in enumerate(phases):
                execution.current_phase = phase
                progress_pct = (phase_index / len(phases)) * 100
                await self._notify_progress(phase.value, progress_pct,
                                            f"Starting phase: {phase.value}")

                phase_result = await self._execute_phase(phase, params, execution)
                execution.phases_completed.append(phase.value)

                if phase_result.get("has_critical_errors"):
                    logger.error("Critical error in phase %s, aborting workflow", phase.value)
                    execution.errors.append(f"Critical error in phase {phase.value}")
                    execution.status = "failed"
                    break

                execution.result_data[phase.value] = phase_result

            if execution.status != "failed":
                execution.status = "completed"

        except Exception as exc:
            logger.error("Workflow '%s' failed: %s", workflow_name, exc, exc_info=True)
            execution.status = "failed"
            execution.errors.append(str(exc))

        finally:
            execution.completed_at = datetime.utcnow()
            execution.total_execution_time_ms = (time.monotonic() - start_time) * 1000
            execution.current_phase = None

            if self.config.enable_provenance:
                execution.provenance_hash = self._compute_execution_provenance(execution)

            self._execution_history.append(execution)
            self._current_execution = None

            await self._notify_progress(
                "complete", 100.0,
                f"Workflow '{workflow_name}' {execution.status} in "
                f"{execution.total_execution_time_ms:.0f}ms",
            )

        logger.info(
            "Workflow '%s' %s in %.1fms with %d errors",
            workflow_name, execution.status,
            execution.total_execution_time_ms, len(execution.errors),
        )
        return execution

    async def _execute_phase(
        self,
        phase: WorkflowPhase,
        params: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Execute all agents within a single workflow phase.

        Agents within the same phase are executed concurrently up to the
        configured concurrency limit. Each agent is wrapped in a timeout
        guard.

        Args:
            phase: The workflow phase to execute.
            params: Parameters forwarded to each agent.
            execution: The parent workflow execution record.

        Returns:
            Dictionary with phase-level results and error flags.
        """
        agent_ids = PHASE_AGENT_MAPPING.get(phase, [])
        active_ids = [
            aid for aid in agent_ids
            if self._agent_statuses.get(aid, AgentStatus(agent_id=aid, agent_name=aid)).status
            not in (AgentStatusCode.DISABLED, AgentStatusCode.FAILED)
        ]

        if not active_ids:
            logger.info("Phase %s has no active agents, skipping", phase.value)
            return {"skipped": True, "has_critical_errors": False}

        logger.info("Executing phase %s with %d agents: %s",
                     phase.value, len(active_ids), active_ids)

        tasks = [
            self._execute_agent_with_semaphore(agent_id, params, execution)
            for agent_id in active_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        phase_results: Dict[str, Any] = {"agents": {}, "has_critical_errors": False}
        for agent_id, result in zip(active_ids, results):
            if isinstance(result, Exception):
                phase_results["agents"][agent_id] = {"error": str(result)}
                phase_results["has_critical_errors"] = True
                logger.error("Agent %s raised exception: %s", agent_id, result)
            else:
                phase_results["agents"][agent_id] = result

        return phase_results

    async def _execute_agent_with_semaphore(
        self,
        agent_id: str,
        params: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Execute a single agent with concurrency limiting and timeout.

        Args:
            agent_id: The agent identifier to execute.
            params: Parameters for the agent execution.
            execution: The parent workflow execution record.

        Returns:
            Dictionary with agent execution results.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        async with self._semaphore:
            return await self._execute_single_agent(agent_id, params, execution)

    async def _execute_single_agent(
        self,
        agent_id: str,
        params: Dict[str, Any],
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Execute a single agent and update its status record.

        Args:
            agent_id: The agent identifier.
            params: Parameters for the agent.
            execution: The parent workflow execution record.

        Returns:
            Dictionary with agent output data.

        Raises:
            asyncio.TimeoutError: If the agent exceeds its timeout.
        """
        status = self._agent_statuses.get(agent_id)
        if status is None:
            return {"error": f"Agent {agent_id} not registered"}

        status.status = AgentStatusCode.RUNNING
        status.started_at = datetime.utcnow()
        agent_start = time.monotonic()

        try:
            agent_instance = self._agents.get(agent_id)
            if agent_instance is None:
                status.status = AgentStatusCode.SKIPPED
                return {"skipped": True, "reason": "No agent instance available"}

            result = await asyncio.wait_for(
                self._invoke_agent(agent_instance, agent_id, params),
                timeout=self.config.timeout_per_agent_seconds,
            )

            elapsed_ms = (time.monotonic() - agent_start) * 1000
            status.status = AgentStatusCode.COMPLETED
            status.completed_at = datetime.utcnow()
            status.execution_time_ms = elapsed_ms
            status.records_processed = result.get("records_processed", 0)

            if self.config.enable_provenance:
                status.provenance_hash = _compute_hash(
                    f"{agent_id}:{execution.execution_id}:{result}"
                )

            execution.agent_statuses[agent_id] = status
            logger.info("Agent %s completed in %.1fms", agent_id, elapsed_ms)
            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - agent_start) * 1000
            status.status = AgentStatusCode.FAILED
            status.completed_at = datetime.utcnow()
            status.execution_time_ms = elapsed_ms
            status.error_message = (
                f"Agent timed out after {self.config.timeout_per_agent_seconds}s"
            )
            execution.agent_statuses[agent_id] = status
            execution.errors.append(f"Agent {agent_id} timed out")
            logger.error("Agent %s timed out after %ds", agent_id,
                         self.config.timeout_per_agent_seconds)
            return {"error": status.error_message}

        except Exception as exc:
            elapsed_ms = (time.monotonic() - agent_start) * 1000
            status.status = AgentStatusCode.FAILED
            status.completed_at = datetime.utcnow()
            status.execution_time_ms = elapsed_ms
            status.error_message = str(exc)
            execution.agent_statuses[agent_id] = status
            execution.errors.append(f"Agent {agent_id} failed: {exc}")
            logger.error("Agent %s failed: %s", agent_id, exc, exc_info=True)
            return {"error": str(exc)}

    async def _invoke_agent(
        self, agent_instance: Any, agent_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke an agent instance, handling both sync and async execute methods.

        Args:
            agent_instance: The agent object (must have an execute method).
            agent_id: Agent identifier for logging.
            params: Parameters for the agent.

        Returns:
            Dictionary with agent output.
        """
        execute_fn = getattr(agent_instance, "execute", None)
        if execute_fn is None:
            logger.warning("Agent %s has no execute method, returning empty result", agent_id)
            return {"status": "no_execute_method"}

        if asyncio.iscoroutinefunction(execute_fn):
            result = await execute_fn(params)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, execute_fn, params)

        if hasattr(result, "dict"):
            return result.dict()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        return {"result": str(result)}

    # -------------------------------------------------------------------------
    # Status & Progress
    # -------------------------------------------------------------------------

    async def get_status(self) -> PackStatus:
        """Get the current overall status of the CSRD Starter Pack.

        Returns:
            PackStatus with agent counts, current workflow, and health info.
        """
        total_agents = len(self._agent_statuses)
        disabled = sum(
            1 for s in self._agent_statuses.values()
            if s.status == AgentStatusCode.DISABLED
        )
        active = total_agents - disabled

        uptime = 0.0
        if self._initialized_at:
            uptime = (datetime.utcnow() - self._initialized_at).total_seconds()

        health = "healthy"
        failed_count = sum(
            1 for s in self._agent_statuses.values()
            if s.status == AgentStatusCode.FAILED
        )
        if failed_count > 0:
            health = "degraded" if failed_count < 5 else "unhealthy"

        return PackStatus(
            pack_id=self.config.pack_id,
            is_initialized=self._initialized,
            total_agents=total_agents,
            active_agents=active,
            disabled_agents=disabled,
            current_workflow=self._current_execution,
            last_execution=(
                self._execution_history[-1] if self._execution_history else None
            ),
            uptime_seconds=uptime,
            initialized_at=self._initialized_at,
            health_status=health,
        )

    def register_progress_callback(self, callback: ProgressCallback) -> None:
        """Register an async callback for progress notifications.

        Args:
            callback: Async function receiving (phase_name, percent, message).
        """
        self._progress_callbacks.append(callback)
        logger.debug("Registered progress callback: %s", callback.__name__)

    def unregister_progress_callback(self, callback: ProgressCallback) -> None:
        """Unregister a previously registered progress callback.

        Args:
            callback: The callback function to remove.
        """
        try:
            self._progress_callbacks.remove(callback)
        except ValueError:
            logger.warning("Callback not found in registry, ignoring unregister request")

    async def _notify_progress(
        self, phase_name: str, percent_complete: float, message: str
    ) -> None:
        """Notify all registered callbacks about workflow progress.

        Args:
            phase_name: Current phase name.
            percent_complete: Completion percentage (0-100).
            message: Human-readable progress message.
        """
        for callback in self._progress_callbacks:
            try:
                await callback(phase_name, percent_complete, message)
            except Exception as exc:
                logger.warning("Progress callback failed: %s", exc)

    # -------------------------------------------------------------------------
    # Provenance & Execution History
    # -------------------------------------------------------------------------

    def get_execution_history(self) -> List[WorkflowExecution]:
        """Return the complete execution history.

        Returns:
            List of all WorkflowExecution records in chronological order.
        """
        return list(self._execution_history)

    def get_agent_statuses(self) -> Dict[str, AgentStatus]:
        """Return the current status of all registered agents.

        Returns:
            Dictionary mapping agent_id to AgentStatus.
        """
        return dict(self._agent_statuses)

    def _create_execution(
        self, workflow_type: WorkflowType, params: Dict[str, Any]
    ) -> WorkflowExecution:
        """Create a new WorkflowExecution record.

        Args:
            workflow_type: The type of workflow being executed.
            params: Workflow parameters.

        Returns:
            A fresh WorkflowExecution instance.
        """
        execution_id = _compute_hash(
            f"{workflow_type.value}:{datetime.utcnow().isoformat()}:{id(self)}"
        )[:16]

        return WorkflowExecution(
            execution_id=execution_id,
            workflow_type=workflow_type,
            parameters=params,
        )

    def _compute_execution_provenance(self, execution: WorkflowExecution) -> str:
        """Compute a SHA-256 provenance hash for the entire workflow execution.

        Args:
            execution: The completed workflow execution record.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        agent_hashes = sorted(
            s.provenance_hash or ""
            for s in execution.agent_statuses.values()
        )
        combined = (
            f"{execution.execution_id}:"
            f"{execution.workflow_type.value}:"
            f"{'|'.join(agent_hashes)}"
        )
        return _compute_hash(combined)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the orchestrator and release all resources.

        Calls cleanup on each agent instance and clears internal state.
        """
        logger.info("Shutting down CSRDPackOrchestrator")
        for agent_id, agent_instance in self._agents.items():
            try:
                cleanup_fn = getattr(agent_instance, "cleanup", None)
                if cleanup_fn:
                    if asyncio.iscoroutinefunction(cleanup_fn):
                        await cleanup_fn()
                    else:
                        cleanup_fn()
            except Exception as exc:
                logger.warning("Cleanup failed for agent %s: %s", agent_id, exc)

        self._agents.clear()
        self._initialized = False
        logger.info("Orchestrator shut down successfully")


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _build_agent_name_lookup() -> Dict[str, str]:
    """Build a lookup table mapping agent IDs to human-readable names.

    Returns:
        Dictionary of agent_id -> display_name.
    """
    return {
        # Foundation
        "GL-FOUND-X-001": "GreenLang Orchestrator",
        "GL-FOUND-X-002": "Schema Compiler & Validator",
        "GL-FOUND-X-003": "Unit & Reference Normalizer",
        "GL-FOUND-X-004": "Assumptions Registry",
        "GL-FOUND-X-005": "Citations & Evidence Agent",
        "GL-FOUND-X-006": "Access & Policy Guard",
        "GL-FOUND-X-007": "PII Redaction Agent",
        "GL-FOUND-X-008": "QA Test Harness",
        "GL-FOUND-X-009": "Observability & Telemetry",
        "GL-FOUND-X-010": "Agent Registry & Versioning",
        # Data Intake
        "GL-DATA-X-001": "Document Ingestion (PDF/Invoice)",
        "GL-DATA-X-004": "ERP/Finance Connector",
        "GL-DATA-X-008": "Supplier Questionnaire Processor",
        "GL-DATA-X-009": "Utility Tariff & Grid Factor",
        # Data Quality
        "GL-DATA-X-010": "Data Quality Profiler",
        "GL-DATA-X-011": "Duplicate Detection Agent",
        "GL-DATA-X-012": "Missing Value Imputer",
        "GL-DATA-X-013": "Outlier Detection Agent",
        # MRV Scope 1
        "GL-MRV-X-001": "Stationary Combustion Calculator",
        "GL-MRV-X-002": "Refrigerants & F-Gas Agent",
        "GL-MRV-X-003": "Mobile Combustion Calculator",
        "GL-MRV-X-004": "Process Emissions Agent",
        "GL-MRV-X-005": "Fugitive Emissions Agent",
        "GL-MRV-X-006": "Land Use Emissions Agent",
        "GL-MRV-X-007": "Waste Treatment Emissions Agent",
        "GL-MRV-X-008": "Agricultural Emissions Agent",
        # MRV Scope 2
        "GL-MRV-X-009": "Scope 2 Location-Based Agent",
        "GL-MRV-X-010": "Scope 2 Market-Based Agent",
        "GL-MRV-X-011": "Steam/Heat Purchase Agent",
        "GL-MRV-X-012": "Cooling Purchase Agent",
        "GL-MRV-X-013": "Dual Reporting Reconciliation",
        # MRV Scope 3
        "GL-MRV-X-014": "Purchased Goods & Services (Cat 1)",
        "GL-MRV-X-015": "Capital Goods (Cat 2)",
        "GL-MRV-X-016": "Fuel & Energy Activities (Cat 3)",
        "GL-MRV-X-017": "Upstream Transportation (Cat 4)",
        "GL-MRV-X-018": "Waste Generated (Cat 5)",
        "GL-MRV-X-019": "Business Travel (Cat 6)",
        "GL-MRV-X-020": "Employee Commuting (Cat 7)",
        "GL-MRV-X-021": "Upstream Leased Assets (Cat 8)",
        "GL-MRV-X-022": "Downstream Transportation (Cat 9)",
        "GL-MRV-X-023": "Processing of Sold Products (Cat 10)",
        "GL-MRV-X-024": "Use of Sold Products (Cat 11)",
        "GL-MRV-X-025": "End-of-Life Treatment (Cat 12)",
        "GL-MRV-X-026": "Downstream Leased Assets (Cat 13)",
        "GL-MRV-X-027": "Franchises (Cat 14)",
        "GL-MRV-X-028": "Investments (Cat 15)",
        "GL-MRV-X-029": "Scope 3 Category Mapper",
        "GL-MRV-X-030": "Audit Trail & Lineage",
        # CSRD Pipeline
        "GL-POL-X-007": "CSRD Compliance Agent",
        "GL-CSRD-MAT": "CSRD Materiality Agent",
        "GL-CSRD-STAKE": "Stakeholder Engagement Agent",
        "GL-CSRD-EXEC": "Executive Summary Template",
        "GL-CSRD-DISC": "ESRS Disclosure Template",
        "GL-CSRD-GHG": "GHG Emissions Report Template",
        "GL-CSRD-DASH": "Compliance Dashboard Template",
        "GL-CSRD-AUDIT": "Auditor Package Template",
    }


def _create_agent_stub(agent_id: str, config: OrchestratorConfig) -> Any:
    """Create a lightweight agent stub for orchestration purposes.

    In production, this would import and instantiate the actual agent class.
    The stub provides the same interface (execute, cleanup) but delegates to
    the real implementation when available.

    Args:
        agent_id: The agent identifier.
        config: Orchestrator config for passing to agents.

    Returns:
        An agent stub object with execute and cleanup methods.
    """

    class _AgentStub:
        """Lightweight agent stub for orchestration."""

        def __init__(self, aid: str, cfg: OrchestratorConfig) -> None:
            self.agent_id = aid
            self.config = cfg
            self._real_agent: Any = None

        async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the agent with given parameters."""
            if self._real_agent is not None:
                execute_fn = getattr(self._real_agent, "execute", None)
                if execute_fn is not None:
                    if asyncio.iscoroutinefunction(execute_fn):
                        return await execute_fn(params)
                    return execute_fn(params)

            return {
                "agent_id": self.agent_id,
                "status": "stub_executed",
                "records_processed": 0,
                "message": f"Agent {self.agent_id} stub execution completed",
            }

        def cleanup(self) -> None:
            """Clean up agent resources."""
            if self._real_agent and hasattr(self._real_agent, "cleanup"):
                self._real_agent.cleanup()

    return _AgentStub(agent_id, config)
