# -*- coding: utf-8 -*-
"""
PACK-039 Energy Monitoring Pack - Workflows Module
====================================================

Orchestration workflows for energy monitoring management covering meter setup,
data collection, anomaly response, EnPI tracking, cost allocation, budget
review, reporting, and the full energy monitoring lifecycle.

Workflows:
    1. MeterSetupWorkflow        - 4-phase meter registration and commissioning
    2. DataCollectionWorkflow    - 4-phase protocol polling and data storage
    3. AnomalyResponseWorkflow   - 3-phase anomaly detection and resolution
    4. EnPITrackingWorkflow      - 4-phase performance indicator tracking
    5. CostAllocationWorkflow    - 3-phase internal cost billing
    6. BudgetReviewWorkflow      - 3-phase budget variance and forecasting
    7. ReportingWorkflow         - 3-phase report generation and distribution
    8. FullMonitoringWorkflow    - 8-phase complete monitoring lifecycle

Pack Tier: Professional (PACK-039)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-039"
__workflows_count__: int = 8

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow 1: Meter Setup
# ---------------------------------------------------------------------------
try:
    from .meter_setup_workflow import (
        MeterSetupWorkflow,
        MeterSetupInput,
        MeterSetupResult,
        MeterDefinition,
    )
    _loaded_workflows.append("MeterSetupWorkflow")
except ImportError as e:
    logger.debug("Workflow 1 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 2: Data Collection
# ---------------------------------------------------------------------------
try:
    from .data_collection_workflow import (
        DataCollectionWorkflow,
        DataCollectionInput,
        DataCollectionResult,
        MeterChannel,
    )
    _loaded_workflows.append("DataCollectionWorkflow")
except ImportError as e:
    logger.debug("Workflow 2 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 3: Anomaly Response
# ---------------------------------------------------------------------------
try:
    from .anomaly_response_workflow import (
        AnomalyResponseWorkflow,
        AnomalyResponseInput,
        AnomalyResponseResult,
        EnergyReading,
    )
    _loaded_workflows.append("AnomalyResponseWorkflow")
except ImportError as e:
    logger.debug("Workflow 3 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 4: EnPI Tracking
# ---------------------------------------------------------------------------
try:
    from .enpi_tracking_workflow import (
        EnPITrackingWorkflow,
        EnPITrackingInput,
        EnPITrackingResult,
        PeriodData,
    )
    _loaded_workflows.append("EnPITrackingWorkflow")
except ImportError as e:
    logger.debug("Workflow 4 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 5: Cost Allocation
# ---------------------------------------------------------------------------
try:
    from .cost_allocation_workflow import (
        CostAllocationWorkflow,
        CostAllocationInput,
        CostAllocationResult,
        CostCentre,
    )
    _loaded_workflows.append("CostAllocationWorkflow")
except ImportError as e:
    logger.debug("Workflow 5 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 6: Budget Review
# ---------------------------------------------------------------------------
try:
    from .budget_review_workflow import (
        BudgetReviewWorkflow,
        BudgetReviewInput,
        BudgetReviewResult,
        BudgetLineItem,
    )
    _loaded_workflows.append("BudgetReviewWorkflow")
except ImportError as e:
    logger.debug("Workflow 6 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 7: Reporting
# ---------------------------------------------------------------------------
try:
    from .reporting_workflow import (
        ReportingWorkflow,
        ReportingInput,
        ReportingResult,
        ReportRequest,
    )
    _loaded_workflows.append("ReportingWorkflow")
except ImportError as e:
    logger.debug("Workflow 7 not available: %s", e)

# ---------------------------------------------------------------------------
# Workflow 8: Full Energy Monitoring Lifecycle
# ---------------------------------------------------------------------------
try:
    from .full_monitoring_workflow import (
        FullMonitoringWorkflow,
        FullMonitoringInput,
        FullMonitoringResult,
    )
    _loaded_workflows.append("FullMonitoringWorkflow")
except ImportError as e:
    logger.debug("Workflow 8 not available: %s", e)


__all__: list[str] = [
    "__version__",
    "__pack__",
    "__workflows_count__",
    # --- Meter Setup Workflow ---
    "MeterSetupWorkflow",
    "MeterSetupInput",
    "MeterSetupResult",
    "MeterDefinition",
    # --- Data Collection Workflow ---
    "DataCollectionWorkflow",
    "DataCollectionInput",
    "DataCollectionResult",
    "MeterChannel",
    # --- Anomaly Response Workflow ---
    "AnomalyResponseWorkflow",
    "AnomalyResponseInput",
    "AnomalyResponseResult",
    "EnergyReading",
    # --- EnPI Tracking Workflow ---
    "EnPITrackingWorkflow",
    "EnPITrackingInput",
    "EnPITrackingResult",
    "PeriodData",
    # --- Cost Allocation Workflow ---
    "CostAllocationWorkflow",
    "CostAllocationInput",
    "CostAllocationResult",
    "CostCentre",
    # --- Budget Review Workflow ---
    "BudgetReviewWorkflow",
    "BudgetReviewInput",
    "BudgetReviewResult",
    "BudgetLineItem",
    # --- Reporting Workflow ---
    "ReportingWorkflow",
    "ReportingInput",
    "ReportingResult",
    "ReportRequest",
    # --- Full Energy Monitoring Lifecycle Workflow ---
    "FullMonitoringWorkflow",
    "FullMonitoringInput",
    "FullMonitoringResult",
]


def get_loaded_workflows() -> list[str]:
    """Return list of workflow class names that loaded successfully."""
    return list(_loaded_workflows)


logger.info(
    "PACK-039 Energy Monitoring workflows: %d/%d loaded",
    len(_loaded_workflows),
    __workflows_count__,
)
