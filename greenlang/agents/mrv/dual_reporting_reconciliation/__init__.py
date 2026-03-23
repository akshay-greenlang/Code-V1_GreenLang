# -*- coding: utf-8 -*-
"""
AGENT-MRV-013: Dual Reporting Reconciliation Agent.

Reconciles GHG Protocol Scope 2 dual reporting between location-based
and market-based emission accounting methods.  Collects pre-calculated
results from upstream Scope 2 agents (MRV-009 through MRV-012),
identifies and explains discrepancies, scores data quality, generates
multi-framework reporting tables, tracks trends, and checks regulatory
compliance across 7 frameworks.

Package: greenlang.agents.mrv.dual_reporting_reconciliation
DB Migration: V064
API Prefix: /api/v1/dual-reporting
Metrics Prefix: gl_drr_
Table Prefix: gl_drr_

Engines:
    1. DualResultCollectorEngine   - Upstream result collection and alignment
    2. DiscrepancyAnalyzerEngine   - Discrepancy identification and waterfall
    3. QualityScorerEngine         - 4-dimension weighted quality scoring
    4. ReportingTableGeneratorEngine - Multi-framework table generation
    5. TrendAnalysisEngine         - YoY, CAGR, PIF, RE100 trend analysis
    6. ComplianceCheckerEngine     - 7-framework regulatory compliance
    7. DualReportingPipelineEngine - 10-stage orchestrated pipeline

Author: GreenLang Platform Team
Date: February 2026
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-X-024"


def __getattr__(name: str):
    """Lazy-load exports to avoid circular imports and improve startup time."""
    _exports = {
        # Engines
        "DualResultCollectorEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector",
        "DiscrepancyAnalyzerEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer",
        "QualityScorerEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.quality_scorer",
        "ReportingTableGeneratorEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.reporting_table_generator",
        "TrendAnalysisEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.trend_analyzer",
        "ComplianceCheckerEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.compliance_checker",
        "DualReportingPipelineEngine": "greenlang.agents.mrv.dual_reporting_reconciliation.dual_reporting_pipeline",
        # Core singletons
        "DualReportingReconciliationConfig": "greenlang.agents.mrv.dual_reporting_reconciliation.config",
        "DualReportingReconciliationMetrics": "greenlang.agents.mrv.dual_reporting_reconciliation.metrics",
        "DualReportingReconciliationProvenance": "greenlang.agents.mrv.dual_reporting_reconciliation.provenance",
        # Service facade
        "DualReportingService": "greenlang.agents.mrv.dual_reporting_reconciliation.setup",
        "get_service": "greenlang.agents.mrv.dual_reporting_reconciliation.setup",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Engines
    "DualResultCollectorEngine",
    "DiscrepancyAnalyzerEngine",
    "QualityScorerEngine",
    "ReportingTableGeneratorEngine",
    "TrendAnalysisEngine",
    "ComplianceCheckerEngine",
    "DualReportingPipelineEngine",
    # Core singletons
    "DualReportingReconciliationConfig",
    "DualReportingReconciliationMetrics",
    "DualReportingReconciliationProvenance",
    # Service facade
    "DualReportingService",
    "get_service",
]
