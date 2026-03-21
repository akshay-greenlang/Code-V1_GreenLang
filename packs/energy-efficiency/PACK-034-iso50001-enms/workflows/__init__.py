# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 Energy Management System Pack - Workflow Orchestration
================================================================

ISO 50001 EnMS workflow orchestrators for energy review, baseline
establishment, action planning, operational control, monitoring,
performance analysis, M&V verification, and audit/certification readiness.
Each workflow coordinates GreenLang calculation engines, data pipelines,
and validation systems into structured multi-phase processes with
SHA-256 provenance hashing.

Workflows:
    - EnergyReviewWorkflow: 4-phase energy review per ISO 50001 Clause 6.3
      with data collection, SEU identification via Pareto analysis,
      EnB/EnPI establishment, and improvement opportunity identification.

    - BaselineEstablishmentWorkflow: 3-phase baseline (EnB) setup with
      data validation, regression modelling (simple/single/multi-variable),
      and approval package generation with confidence intervals.

    - ActionPlanWorkflow: 4-phase planning per ISO 50001 Clause 6.2 with
      SMART objective setting, action definition with savings estimates,
      resource allocation within budget, and implementation timeline.

    - OperationalControlWorkflow: 3-phase operational control per Clause 8
      with operating criteria definition, monitoring setup with 3-sigma
      control limits, and deviation response procedures with escalation.

    - MonitoringWorkflow: 4-phase M&V monitoring per Clause 9.1 with meter
      verification, data collection with quality assessment, EnPI/CUSUM
      analysis, and monitoring report generation.

    - PerformanceAnalysisWorkflow: 3-phase EnPI tracking with calculation,
      CUSUM shift detection, and trend analysis with projections and
      weighted performance scoring.

    - MVVerificationWorkflow: 3-phase IPMVP verification with baseline
      adjustment, gross/net savings quantification with non-routine
      adjustments, and uncertainty analysis with significance testing.

    - AuditCertificationWorkflow: 4-phase audit and certification readiness
      with ISO 50001 Clauses 4-10 gap analysis, internal audit simulation,
      corrective action planning, and Stage 1/Stage 2 readiness assessment.

Author: GreenLang Team
Version: 34.0.0
"""

# ---------------------------------------------------------------------------
# Energy Review Workflow
# ---------------------------------------------------------------------------
from .energy_review_workflow import (
    EnergyReviewWorkflow,
    EnergyReviewInput,
    EnergyReviewResult,
    ReviewPhase,
)

# ---------------------------------------------------------------------------
# Baseline Establishment Workflow
# ---------------------------------------------------------------------------
from .baseline_establishment_workflow import (
    BaselineEstablishmentWorkflow,
    BaselineEstablishmentInput,
    BaselineEstablishmentResult,
    BaselinePhase,
)

# ---------------------------------------------------------------------------
# Action Plan Workflow
# ---------------------------------------------------------------------------
from .action_plan_workflow import (
    ActionPlanWorkflow,
    ActionPlanInput,
    ActionPlanResult,
    PlanningPhase,
)

# ---------------------------------------------------------------------------
# Operational Control Workflow
# ---------------------------------------------------------------------------
from .operational_control_workflow import (
    OperationalControlWorkflow,
    OperationalControlInput,
    OperationalControlResult,
    ControlPhase,
)

# ---------------------------------------------------------------------------
# Monitoring Workflow
# ---------------------------------------------------------------------------
from .monitoring_workflow import (
    MonitoringWorkflow,
    MonitoringInput,
    MonitoringResult,
    MonitoringPhase,
)

# ---------------------------------------------------------------------------
# Performance Analysis Workflow
# ---------------------------------------------------------------------------
from .performance_analysis_workflow import (
    PerformanceAnalysisWorkflow,
    PerformanceAnalysisInput,
    PerformanceAnalysisResult,
    AnalysisPhase,
)

# ---------------------------------------------------------------------------
# M&V Verification Workflow
# ---------------------------------------------------------------------------
from .mv_verification_workflow import (
    MVVerificationWorkflow,
    MVVerificationInput,
    MVVerificationResult,
    VerificationPhase,
)

# ---------------------------------------------------------------------------
# Audit & Certification Workflow
# ---------------------------------------------------------------------------
from .audit_certification_workflow import (
    AuditCertificationWorkflow,
    AuditCertificationInput,
    AuditCertificationResult,
    AuditPhase,
)

__all__ = [
    # --- Energy Review Workflow ---
    "EnergyReviewWorkflow",
    "EnergyReviewInput",
    "EnergyReviewResult",
    "ReviewPhase",
    # --- Baseline Establishment Workflow ---
    "BaselineEstablishmentWorkflow",
    "BaselineEstablishmentInput",
    "BaselineEstablishmentResult",
    "BaselinePhase",
    # --- Action Plan Workflow ---
    "ActionPlanWorkflow",
    "ActionPlanInput",
    "ActionPlanResult",
    "PlanningPhase",
    # --- Operational Control Workflow ---
    "OperationalControlWorkflow",
    "OperationalControlInput",
    "OperationalControlResult",
    "ControlPhase",
    # --- Monitoring Workflow ---
    "MonitoringWorkflow",
    "MonitoringInput",
    "MonitoringResult",
    "MonitoringPhase",
    # --- Performance Analysis Workflow ---
    "PerformanceAnalysisWorkflow",
    "PerformanceAnalysisInput",
    "PerformanceAnalysisResult",
    "AnalysisPhase",
    # --- M&V Verification Workflow ---
    "MVVerificationWorkflow",
    "MVVerificationInput",
    "MVVerificationResult",
    "VerificationPhase",
    # --- Audit & Certification Workflow ---
    "AuditCertificationWorkflow",
    "AuditCertificationInput",
    "AuditCertificationResult",
    "AuditPhase",
]
