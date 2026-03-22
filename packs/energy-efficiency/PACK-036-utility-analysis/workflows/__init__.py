# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Workflow Orchestration
================================================================

Utility analysis workflow orchestrators for bill auditing, rate optimization,
demand management, cost allocation, budget planning, procurement analysis,
benchmarking, and full end-to-end utility assessment. Each workflow coordinates
GreenLang calculation engines, data pipelines, and validation systems
into structured multi-phase processes with SHA-256 provenance hashing.

Workflows:
    - BillAuditWorkflow: 4-phase bill audit with bill ingestion,
      line-item validation, error detection, and discrepancy reporting.

    - RateOptimizationWorkflow: 3-phase rate optimization with current
      rate analysis, alternative rate simulation, and savings-ranked
      recommendation generation.

    - DemandManagementWorkflow: 4-phase demand management with load
      profile analysis, peak detection, demand response strategy
      development, and demand charge reduction planning.

    - CostAllocationWorkflow: 3-phase cost allocation with meter
      mapping, consumption disaggregation by department/tenant/process,
      and chargeback report generation.

    - BudgetPlanningWorkflow: 4-phase budget planning with historical
      trend analysis, weather normalization, rate escalation modeling,
      and multi-scenario budget forecast generation.

    - ProcurementAnalysisWorkflow: 3-phase procurement analysis with
      contract review, market price benchmarking, and procurement
      strategy recommendation with RFP template generation.

    - BenchmarkAnalysisWorkflow: 3-phase benchmarking with peer group
      selection, EUI/cost-per-m2 comparison, and performance gap
      identification with improvement target setting.

    - FullUtilityAnalysisWorkflow: 8-phase end-to-end assessment
      orchestrating bill audit, rate optimization, demand management,
      cost allocation, budget planning, procurement analysis,
      benchmarking, and consolidated reporting into a single pipeline.

Author: GreenLang Team
Version: 36.0.0
"""

# ---------------------------------------------------------------------------
# Bill Audit Workflow
# ---------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-036"
__pack_name__: str = "Utility Analysis Pack"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Bill Audit Workflow
# ---------------------------------------------------------------------------
try:
    from .bill_audit_workflow import (
        BillAuditWorkflow,
        BillAuditInput,
        BillAuditResult,
        BillDiscrepancy,
    )
    _loaded_workflows.append("BillAuditWorkflow")
except ImportError as e:
    logger.debug("BillAuditWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Rate Optimization Workflow
# ---------------------------------------------------------------------------
try:
    from .rate_optimization_workflow import (
        RateOptimizationWorkflow,
        RateOptimizationInput,
        RateOptimizationResult,
        RateRecommendation,
    )
    _loaded_workflows.append("RateOptimizationWorkflow")
except ImportError as e:
    logger.debug("RateOptimizationWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Demand Management Workflow
# ---------------------------------------------------------------------------
try:
    from .demand_management_workflow import (
        DemandManagementWorkflow,
        DemandManagementInput,
        DemandManagementResult,
        DemandReductionStrategy,
    )
    _loaded_workflows.append("DemandManagementWorkflow")
except ImportError as e:
    logger.debug("DemandManagementWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Cost Allocation Workflow
# ---------------------------------------------------------------------------
try:
    from .cost_allocation_workflow import (
        CostAllocationWorkflow,
        CostAllocationInput,
        CostAllocationResult,
        AllocationEntry,
    )
    _loaded_workflows.append("CostAllocationWorkflow")
except ImportError as e:
    logger.debug("CostAllocationWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Budget Planning Workflow
# ---------------------------------------------------------------------------
try:
    from .budget_planning_workflow import (
        BudgetPlanningWorkflow,
        BudgetPlanningInput,
        BudgetPlanningResult,
        BudgetScenario,
    )
    _loaded_workflows.append("BudgetPlanningWorkflow")
except ImportError as e:
    logger.debug("BudgetPlanningWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Procurement Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from .procurement_analysis_workflow import (
        ProcurementAnalysisWorkflow,
        ProcurementAnalysisInput,
        ProcurementAnalysisResult,
        ProcurementRecommendation,
    )
    _loaded_workflows.append("ProcurementAnalysisWorkflow")
except ImportError as e:
    logger.debug("ProcurementAnalysisWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Benchmark Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from .benchmark_analysis_workflow import (
        BenchmarkAnalysisWorkflow,
        BenchmarkAnalysisInput,
        BenchmarkAnalysisResult,
        BenchmarkComparison,
    )
    _loaded_workflows.append("BenchmarkAnalysisWorkflow")
except ImportError as e:
    logger.debug("BenchmarkAnalysisWorkflow not available: %s", e)

# ---------------------------------------------------------------------------
# Full Utility Analysis Workflow
# ---------------------------------------------------------------------------
try:
    from .full_utility_analysis_workflow import (
        FullUtilityAnalysisWorkflow,
        FullUtilityAnalysisInput,
        FullUtilityAnalysisResult,
    )
    _loaded_workflows.append("FullUtilityAnalysisWorkflow")
except ImportError as e:
    logger.debug("FullUtilityAnalysisWorkflow not available: %s", e)

__all__ = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__workflows_count__",
    # --- Bill Audit Workflow ---
    "BillAuditWorkflow",
    "BillAuditInput",
    "BillAuditResult",
    "BillDiscrepancy",
    # --- Rate Optimization Workflow ---
    "RateOptimizationWorkflow",
    "RateOptimizationInput",
    "RateOptimizationResult",
    "RateRecommendation",
    # --- Demand Management Workflow ---
    "DemandManagementWorkflow",
    "DemandManagementInput",
    "DemandManagementResult",
    "DemandReductionStrategy",
    # --- Cost Allocation Workflow ---
    "CostAllocationWorkflow",
    "CostAllocationInput",
    "CostAllocationResult",
    "AllocationEntry",
    # --- Budget Planning Workflow ---
    "BudgetPlanningWorkflow",
    "BudgetPlanningInput",
    "BudgetPlanningResult",
    "BudgetScenario",
    # --- Procurement Analysis Workflow ---
    "ProcurementAnalysisWorkflow",
    "ProcurementAnalysisInput",
    "ProcurementAnalysisResult",
    "ProcurementRecommendation",
    # --- Benchmark Analysis Workflow ---
    "BenchmarkAnalysisWorkflow",
    "BenchmarkAnalysisInput",
    "BenchmarkAnalysisResult",
    "BenchmarkComparison",
    # --- Full Utility Analysis Workflow ---
    "FullUtilityAnalysisWorkflow",
    "FullUtilityAnalysisInput",
    "FullUtilityAnalysisResult",
]


def get_loaded_workflows() -> list[str]:
    """Return list of workflow class names that loaded successfully."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return count of workflows that loaded successfully."""
    return len(_loaded_workflows)


logger.info(
    "PACK-036 Utility Analysis workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
