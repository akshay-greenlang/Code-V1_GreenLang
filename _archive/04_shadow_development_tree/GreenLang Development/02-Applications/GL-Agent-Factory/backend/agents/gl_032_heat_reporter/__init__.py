"""
GL-032: Process Heat Reporter Agent (HEATREPORTER)

This package provides the HeatReporterAgent for generating comprehensive
performance and compliance reports on process heat systems.

Key Features:
- Performance metrics calculation and trending
- Compliance reporting per ISO 50001, NFPA 86
- Energy efficiency benchmarking
- KPI tracking and visualization
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISO 50001: Energy Management Systems
- NFPA 86: Standard for Ovens and Furnaces
- API 560: Fired Heaters for General Refinery Service
- EN 16247: Energy Audits

Example Usage:
    >>> from backend.agents.gl_032_heat_reporter import (
    ...     HeatReporterAgent,
    ...     HeatReporterInput,
    ... )
    >>> agent = HeatReporterAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Overall Efficiency: {result.overall_efficiency_pct}%")
"""

from .agent import (
    HeatReporterAgent,
    HeatReporterInput,
    HeatReporterOutput,
    PerformanceMetric,
    ComplianceItem,
    EfficiencyTrend,
    EnergyConsumption,
    PACK_SPEC,
)

__all__ = [
    "HeatReporterAgent",
    "HeatReporterInput",
    "HeatReporterOutput",
    "PerformanceMetric",
    "ComplianceItem",
    "EfficiencyTrend",
    "EnergyConsumption",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-032"
__agent_name__ = "HEATREPORTER"
