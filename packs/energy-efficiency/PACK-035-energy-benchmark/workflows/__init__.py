# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Workflow Orchestration
==========================================================

Energy benchmarking workflow orchestrators for initial benchmark setup,
continuous monitoring, peer comparison, portfolio benchmarking, performance
gap analysis, regulatory compliance, target setting, and full end-to-end
assessment. Each workflow coordinates GreenLang calculation engines, data
pipelines, and validation systems into structured multi-phase processes
with SHA-256 provenance hashing.

Workflows:
    - InitialBenchmarkWorkflow: 4-phase initial benchmark establishment
      with facility registration, data collection, EUI calculation, and
      peer comparison against ENERGY STAR / CIBSE TM46 / DIN V 18599.

    - ContinuousMonitoringWorkflow: 4-phase real-time monitoring with
      data ingestion, CUSUM deviation detection, SPC rule checks, and
      automated alerting with trend decomposition.

    - PeerComparisonWorkflow: 3-phase peer comparison with peer group
      definition, percentile ranking, and gap-to-best-practice analysis
      across multiple benchmark datasets.

    - PortfolioBenchmarkWorkflow: 4-phase portfolio benchmarking with
      facility inventory, cross-portfolio EUI ranking, outlier detection,
      and improvement prioritisation for 1-1000+ sites.

    - PerformanceGapWorkflow: 3-phase gap analysis with end-use
      disaggregation, gap identification against benchmarks, and
      savings opportunity quantification by end use.

    - RegulatoryComplianceWorkflow: 3-phase compliance with EPBD/EED
      obligation assessment, EPC/DEC rating generation, and MEPS
      compliance reporting with deadline tracking.

    - TargetSettingWorkflow: 3-phase target setting with baseline
      definition, target trajectory generation (peer-based, absolute,
      SBTi-aligned), and milestone scheduling.

    - FullAssessmentWorkflow: 6-phase end-to-end assessment orchestrating
      EUI calculation, weather normalisation, peer comparison, gap
      analysis, target setting, and report generation.

Author: GreenLang Team
Version: 35.0.0
"""

# ---------------------------------------------------------------------------
# Initial Benchmark Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.initial_benchmark_workflow import (
    InitialBenchmarkWorkflow,
    InitialBenchmarkInput,
    InitialBenchmarkResult,
)

# ---------------------------------------------------------------------------
# Continuous Monitoring Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.continuous_monitoring_workflow import (
    ContinuousMonitoringWorkflow,
    ContinuousMonitoringInput,
    ContinuousMonitoringResult,
)

# ---------------------------------------------------------------------------
# Peer Comparison Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.peer_comparison_workflow import (
    PeerComparisonWorkflow,
    PeerComparisonInput,
    PeerComparisonResult,
)

# ---------------------------------------------------------------------------
# Portfolio Benchmark Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.portfolio_benchmark_workflow import (
    PortfolioBenchmarkWorkflow,
    PortfolioBenchmarkInput,
    PortfolioBenchmarkResult,
)

# ---------------------------------------------------------------------------
# Performance Gap Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.performance_gap_workflow import (
    PerformanceGapWorkflow,
    PerformanceGapInput,
    PerformanceGapResult,
)

# ---------------------------------------------------------------------------
# Regulatory Compliance Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.regulatory_compliance_workflow import (
    RegulatoryComplianceWorkflow,
    RegulatoryComplianceInput,
    RegulatoryComplianceResult,
)

# ---------------------------------------------------------------------------
# Target Setting Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.target_setting_workflow import (
    TargetSettingWorkflow,
    TargetSettingInput,
    TargetSettingResult,
)

# ---------------------------------------------------------------------------
# Full Assessment Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.workflows.full_assessment_workflow import (
    FullAssessmentWorkflow,
    FullAssessmentInput,
    FullAssessmentResult,
)

__all__ = [
    # --- Initial Benchmark Workflow ---
    "InitialBenchmarkWorkflow",
    "InitialBenchmarkInput",
    "InitialBenchmarkResult",
    # --- Continuous Monitoring Workflow ---
    "ContinuousMonitoringWorkflow",
    "ContinuousMonitoringInput",
    "ContinuousMonitoringResult",
    # --- Peer Comparison Workflow ---
    "PeerComparisonWorkflow",
    "PeerComparisonInput",
    "PeerComparisonResult",
    # --- Portfolio Benchmark Workflow ---
    "PortfolioBenchmarkWorkflow",
    "PortfolioBenchmarkInput",
    "PortfolioBenchmarkResult",
    # --- Performance Gap Workflow ---
    "PerformanceGapWorkflow",
    "PerformanceGapInput",
    "PerformanceGapResult",
    # --- Regulatory Compliance Workflow ---
    "RegulatoryComplianceWorkflow",
    "RegulatoryComplianceInput",
    "RegulatoryComplianceResult",
    # --- Target Setting Workflow ---
    "TargetSettingWorkflow",
    "TargetSettingInput",
    "TargetSettingResult",
    # --- Full Assessment Workflow ---
    "FullAssessmentWorkflow",
    "FullAssessmentInput",
    "FullAssessmentResult",
]
