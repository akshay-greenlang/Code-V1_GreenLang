# -*- coding: utf-8 -*-
"""
GL-OPS-X-006: Operational Benchmarking Agent
=============================================

Benchmarks operational performance against industry standards, peers, and
best practices to identify relative positioning and improvement potential.

Capabilities:
    - Industry benchmark comparison
    - Peer group performance analysis
    - Best practice identification
    - Performance quartile ranking
    - Gap analysis against benchmarks
    - Trend analysis vs benchmarks

Zero-Hallucination Guarantees:
    - All benchmark calculations use deterministic formulas
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path
    - All comparisons traceable to source benchmarks

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class BenchmarkSource(str, Enum):
    """Sources of benchmark data."""
    INDUSTRY_AVERAGE = "industry_average"
    INDUSTRY_BEST = "industry_best"
    PEER_GROUP = "peer_group"
    REGULATORY = "regulatory"
    INTERNAL_BEST = "internal_best"
    SCIENCE_BASED = "science_based"
    CUSTOM = "custom"


class PerformanceQuartile(str, Enum):
    """Performance quartile rankings."""
    TOP_QUARTILE = "top_quartile"  # Top 25%
    SECOND_QUARTILE = "second_quartile"  # 25-50%
    THIRD_QUARTILE = "third_quartile"  # 50-75%
    BOTTOM_QUARTILE = "bottom_quartile"  # Bottom 25%


class BenchmarkCategory(str, Enum):
    """Categories of benchmarks."""
    EMISSIONS_INTENSITY = "emissions_intensity"
    ENERGY_INTENSITY = "energy_intensity"
    WATER_INTENSITY = "water_intensity"
    WASTE_INTENSITY = "waste_intensity"
    RENEWABLE_SHARE = "renewable_share"
    EFFICIENCY_RATIO = "efficiency_ratio"
    COST_EFFICIENCY = "cost_efficiency"


class IndustrySector(str, Enum):
    """Industry sectors for benchmarking."""
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    TRANSPORTATION = "transportation"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    FINANCE = "finance"


# =============================================================================
# Pydantic Models
# =============================================================================

class BenchmarkMetric(BaseModel):
    """A benchmark metric definition."""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Metric name")
    category: BenchmarkCategory = Field(..., description="Benchmark category")
    source: BenchmarkSource = Field(..., description="Benchmark source")
    sector: IndustrySector = Field(..., description="Industry sector")

    # Benchmark values
    benchmark_value: float = Field(..., description="Benchmark value")
    top_quartile_threshold: float = Field(..., description="Top 25% threshold")
    median_value: float = Field(..., description="Median value")
    bottom_quartile_threshold: float = Field(..., description="Bottom 25% threshold")

    # Metadata
    unit: str = Field(..., description="Unit of measurement")
    year: int = Field(..., description="Benchmark year")
    sample_size: Optional[int] = Field(None, description="Number of companies in benchmark")
    geographic_scope: str = Field(default="global", description="Geographic scope")

    # Direction (lower_is_better for intensity metrics)
    lower_is_better: bool = Field(default=True, description="Whether lower values are better")

    # Source details
    source_reference: Optional[str] = Field(None, description="Source reference/citation")
    last_updated: datetime = Field(default_factory=DeterministicClock.now)


class FacilityPerformance(BaseModel):
    """Performance data for a facility."""
    facility_id: str = Field(..., description="Facility identifier")
    metric_id: str = Field(..., description="Metric identifier")
    value: float = Field(..., description="Performance value")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    unit: str = Field(..., description="Unit of measurement")

    # Normalizing factors
    production_volume: Optional[float] = Field(None, description="Production volume")
    floor_area_sqm: Optional[float] = Field(None, description="Floor area")
    revenue: Optional[float] = Field(None, description="Revenue")
    headcount: Optional[int] = Field(None, description="Employee headcount")


class BenchmarkComparison(BaseModel):
    """Result of benchmark comparison."""
    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    facility_id: str = Field(..., description="Facility identifier")
    metric_id: str = Field(..., description="Metric identifier")
    benchmark_id: str = Field(..., description="Benchmark metric ID")

    # Values
    facility_value: float = Field(..., description="Facility performance value")
    benchmark_value: float = Field(..., description="Benchmark value")
    unit: str = Field(..., description="Unit of measurement")

    # Analysis
    gap: float = Field(..., description="Gap to benchmark")
    gap_percent: float = Field(..., description="Gap as percentage")
    quartile: PerformanceQuartile = Field(..., description="Performance quartile")

    # Context
    better_than_benchmark: bool = Field(..., description="Whether performance is better")
    improvement_potential: float = Field(default=0.0, description="Potential improvement")

    # Timestamp
    comparison_date: datetime = Field(default_factory=DeterministicClock.now)


class GapAnalysis(BaseModel):
    """Detailed gap analysis result."""
    facility_id: str = Field(..., description="Facility identifier")
    analysis_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    metrics_analyzed: int = Field(default=0, description="Number of metrics analyzed")
    top_quartile_count: int = Field(default=0)
    second_quartile_count: int = Field(default=0)
    third_quartile_count: int = Field(default=0)
    bottom_quartile_count: int = Field(default=0)

    # Aggregate scores
    overall_percentile: float = Field(default=50.0, ge=0, le=100)
    weighted_score: float = Field(default=0.0, ge=0, le=100)

    # Gaps
    comparisons: List[BenchmarkComparison] = Field(default_factory=list)
    largest_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    strengths: List[Dict[str, Any]] = Field(default_factory=list)

    # Recommendations
    improvement_priorities: List[str] = Field(default_factory=list)


class BenchmarkingInput(BaseModel):
    """Input for the Operational Benchmarking Agent."""
    operation: str = Field(..., description="Operation to perform")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    performance_data: List[FacilityPerformance] = Field(
        default_factory=list, description="Facility performance data"
    )
    benchmarks: List[BenchmarkMetric] = Field(
        default_factory=list, description="Benchmark metrics"
    )
    sector: Optional[IndustrySector] = Field(None, description="Industry sector filter")
    category: Optional[BenchmarkCategory] = Field(None, description="Benchmark category filter")
    benchmark_id: Optional[str] = Field(None, description="Specific benchmark ID")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'compare_to_benchmark', 'rank_facilities', 'analyze_gaps',
            'get_industry_benchmarks', 'add_benchmark', 'get_peer_comparison',
            'get_improvement_potential', 'get_trends', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class BenchmarkingOutput(BaseModel):
    """Output from the Operational Benchmarking Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Default Industry Benchmarks
# =============================================================================

DEFAULT_BENCHMARKS = [
    {
        "name": "Emissions Intensity - Manufacturing",
        "category": BenchmarkCategory.EMISSIONS_INTENSITY,
        "source": BenchmarkSource.INDUSTRY_AVERAGE,
        "sector": IndustrySector.MANUFACTURING,
        "benchmark_value": 0.35,
        "top_quartile_threshold": 0.20,
        "median_value": 0.35,
        "bottom_quartile_threshold": 0.55,
        "unit": "kg CO2e / $ revenue",
        "year": 2024,
        "lower_is_better": True,
    },
    {
        "name": "Energy Intensity - Manufacturing",
        "category": BenchmarkCategory.ENERGY_INTENSITY,
        "source": BenchmarkSource.INDUSTRY_AVERAGE,
        "sector": IndustrySector.MANUFACTURING,
        "benchmark_value": 0.25,
        "top_quartile_threshold": 0.15,
        "median_value": 0.25,
        "bottom_quartile_threshold": 0.40,
        "unit": "kWh / $ revenue",
        "year": 2024,
        "lower_is_better": True,
    },
    {
        "name": "Renewable Share - All Sectors",
        "category": BenchmarkCategory.RENEWABLE_SHARE,
        "source": BenchmarkSource.INDUSTRY_AVERAGE,
        "sector": IndustrySector.MANUFACTURING,
        "benchmark_value": 35.0,
        "top_quartile_threshold": 60.0,
        "median_value": 35.0,
        "bottom_quartile_threshold": 15.0,
        "unit": "%",
        "year": 2024,
        "lower_is_better": False,
    },
]


# =============================================================================
# Operational Benchmarking Agent Implementation
# =============================================================================

class OperationalBenchmarkingAgent(BaseAgent):
    """
    GL-OPS-X-006: Operational Benchmarking Agent

    Benchmarks operational performance against industry standards, peers, and
    best practices to identify relative positioning and improvement potential.

    Zero-Hallucination Guarantees:
        - All benchmark calculations use deterministic formulas
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the calculation path
        - All comparisons traceable to source benchmarks

    Usage:
        agent = OperationalBenchmarkingAgent()

        # Compare to benchmark
        result = agent.run({
            "operation": "compare_to_benchmark",
            "facility_id": "FAC-001",
            "performance_data": [...],
            "sector": "manufacturing"
        })

        # Analyze gaps
        result = agent.run({
            "operation": "analyze_gaps",
            "facility_id": "FAC-001"
        })
    """

    AGENT_ID = "GL-OPS-X-006"
    AGENT_NAME = "Operational Benchmarking Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Operational Benchmarking Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Operational benchmarking against industry standards",
                version=self.VERSION,
                parameters={
                    "top_quartile_percentile": 75,
                    "bottom_quartile_percentile": 25,
                }
            )
        super().__init__(config)

        # Benchmark storage
        self._benchmarks: Dict[str, BenchmarkMetric] = {}

        # Load default benchmarks
        self._load_default_benchmarks()

        # Facility performance data
        self._performance_data: Dict[str, List[FacilityPerformance]] = defaultdict(list)

        # Comparison history
        self._comparisons: Dict[str, List[BenchmarkComparison]] = defaultdict(list)

        # Gap analyses
        self._gap_analyses: Dict[str, GapAnalysis] = {}

        # Statistics
        self._total_comparisons = 0
        self._total_analyses = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def _load_default_benchmarks(self):
        """Load default industry benchmarks."""
        for benchmark_data in DEFAULT_BENCHMARKS:
            benchmark = BenchmarkMetric(**benchmark_data)
            self._benchmarks[benchmark.metric_id] = benchmark

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute benchmarking operations."""
        start_time = time.time()

        try:
            bench_input = BenchmarkingInput(**input_data)
            operation = bench_input.operation

            result_data = self._route_operation(bench_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = BenchmarkingOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Benchmarking operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, bench_input: BenchmarkingInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = bench_input.operation

        if operation == "compare_to_benchmark":
            return self._handle_compare_to_benchmark(
                bench_input.facility_id,
                bench_input.performance_data,
                bench_input.sector,
                bench_input.category,
            )
        elif operation == "rank_facilities":
            return self._handle_rank_facilities(
                bench_input.performance_data,
                bench_input.sector,
            )
        elif operation == "analyze_gaps":
            return self._handle_analyze_gaps(bench_input.facility_id)
        elif operation == "get_industry_benchmarks":
            return self._handle_get_industry_benchmarks(
                bench_input.sector,
                bench_input.category,
            )
        elif operation == "add_benchmark":
            return self._handle_add_benchmark(bench_input.benchmarks)
        elif operation == "get_peer_comparison":
            return self._handle_get_peer_comparison(
                bench_input.facility_id,
                bench_input.performance_data,
            )
        elif operation == "get_improvement_potential":
            return self._handle_get_improvement_potential(bench_input.facility_id)
        elif operation == "get_trends":
            return self._handle_get_trends(bench_input.facility_id)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Benchmark Comparison
    # =========================================================================

    def _handle_compare_to_benchmark(
        self,
        facility_id: Optional[str],
        performance_data: List[FacilityPerformance],
        sector: Optional[IndustrySector],
        category: Optional[BenchmarkCategory],
    ) -> Dict[str, Any]:
        """Compare facility performance to benchmarks."""
        if not facility_id:
            return {"error": "facility_id is required"}

        # Store performance data
        for perf in performance_data:
            self._performance_data[facility_id].append(perf)

        # Get applicable benchmarks
        benchmarks = self._get_applicable_benchmarks(sector, category)

        if not benchmarks:
            return {
                "facility_id": facility_id,
                "message": "No applicable benchmarks found",
            }

        comparisons = []

        for perf in performance_data:
            # Find matching benchmark
            matching_benchmark = self._find_matching_benchmark(perf, benchmarks)

            if matching_benchmark:
                comparison = self._create_comparison(perf, matching_benchmark)
                comparisons.append(comparison)
                self._comparisons[facility_id].append(comparison)
                self._total_comparisons += 1

        # Summary statistics
        if comparisons:
            quartile_counts = defaultdict(int)
            for comp in comparisons:
                quartile_counts[comp.quartile.value] += 1

            avg_gap_percent = sum(c.gap_percent for c in comparisons) / len(comparisons)
            better_count = sum(1 for c in comparisons if c.better_than_benchmark)

            return {
                "facility_id": facility_id,
                "comparisons": [c.model_dump() for c in comparisons],
                "comparison_count": len(comparisons),
                "quartile_distribution": dict(quartile_counts),
                "average_gap_percent": round(avg_gap_percent, 2),
                "better_than_benchmark_count": better_count,
                "performance_rate": round(better_count / len(comparisons) * 100, 2),
            }

        return {
            "facility_id": facility_id,
            "comparisons": [],
            "message": "No matching benchmarks for provided metrics",
        }

    def _get_applicable_benchmarks(
        self,
        sector: Optional[IndustrySector],
        category: Optional[BenchmarkCategory],
    ) -> List[BenchmarkMetric]:
        """Get benchmarks applicable to filters."""
        benchmarks = list(self._benchmarks.values())

        if sector:
            benchmarks = [b for b in benchmarks if b.sector == sector]

        if category:
            benchmarks = [b for b in benchmarks if b.category == category]

        return benchmarks

    def _find_matching_benchmark(
        self,
        perf: FacilityPerformance,
        benchmarks: List[BenchmarkMetric],
    ) -> Optional[BenchmarkMetric]:
        """Find benchmark matching performance metric."""
        # Simple matching by category (in real system, would be more sophisticated)
        for benchmark in benchmarks:
            # Match by unit similarity
            if perf.unit.lower() == benchmark.unit.lower():
                return benchmark

            # Match by metric name pattern
            metric_lower = perf.metric_id.lower()
            if benchmark.category == BenchmarkCategory.EMISSIONS_INTENSITY and "emission" in metric_lower:
                return benchmark
            if benchmark.category == BenchmarkCategory.ENERGY_INTENSITY and "energy" in metric_lower:
                return benchmark

        return benchmarks[0] if benchmarks else None

    def _create_comparison(
        self,
        perf: FacilityPerformance,
        benchmark: BenchmarkMetric,
    ) -> BenchmarkComparison:
        """Create benchmark comparison result."""
        gap = perf.value - benchmark.benchmark_value
        gap_percent = (gap / benchmark.benchmark_value * 100) if benchmark.benchmark_value != 0 else 0

        # Determine if better
        if benchmark.lower_is_better:
            better_than_benchmark = perf.value < benchmark.benchmark_value
        else:
            better_than_benchmark = perf.value > benchmark.benchmark_value

        # Determine quartile
        quartile = self._determine_quartile(perf.value, benchmark)

        # Calculate improvement potential
        if benchmark.lower_is_better:
            improvement_potential = max(0, perf.value - benchmark.top_quartile_threshold)
        else:
            improvement_potential = max(0, benchmark.top_quartile_threshold - perf.value)

        return BenchmarkComparison(
            facility_id=perf.facility_id,
            metric_id=perf.metric_id,
            benchmark_id=benchmark.metric_id,
            facility_value=perf.value,
            benchmark_value=benchmark.benchmark_value,
            unit=perf.unit,
            gap=round(gap, 4),
            gap_percent=round(gap_percent, 2),
            quartile=quartile,
            better_than_benchmark=better_than_benchmark,
            improvement_potential=round(improvement_potential, 4),
        )

    def _determine_quartile(
        self, value: float, benchmark: BenchmarkMetric
    ) -> PerformanceQuartile:
        """Determine performance quartile."""
        if benchmark.lower_is_better:
            if value <= benchmark.top_quartile_threshold:
                return PerformanceQuartile.TOP_QUARTILE
            elif value <= benchmark.median_value:
                return PerformanceQuartile.SECOND_QUARTILE
            elif value <= benchmark.bottom_quartile_threshold:
                return PerformanceQuartile.THIRD_QUARTILE
            else:
                return PerformanceQuartile.BOTTOM_QUARTILE
        else:
            if value >= benchmark.top_quartile_threshold:
                return PerformanceQuartile.TOP_QUARTILE
            elif value >= benchmark.median_value:
                return PerformanceQuartile.SECOND_QUARTILE
            elif value >= benchmark.bottom_quartile_threshold:
                return PerformanceQuartile.THIRD_QUARTILE
            else:
                return PerformanceQuartile.BOTTOM_QUARTILE

    # =========================================================================
    # Facility Ranking
    # =========================================================================

    def _handle_rank_facilities(
        self,
        performance_data: List[FacilityPerformance],
        sector: Optional[IndustrySector],
    ) -> Dict[str, Any]:
        """Rank facilities against each other."""
        # Group by metric
        by_metric: Dict[str, List[FacilityPerformance]] = defaultdict(list)
        for perf in performance_data:
            by_metric[perf.metric_id].append(perf)

        rankings = {}

        for metric_id, perfs in by_metric.items():
            # Sort by value (assuming lower is better - adjust in production)
            sorted_perfs = sorted(perfs, key=lambda p: p.value)

            rankings[metric_id] = [
                {
                    "rank": i + 1,
                    "facility_id": p.facility_id,
                    "value": p.value,
                    "unit": p.unit,
                    "percentile": round((1 - i / len(sorted_perfs)) * 100, 2),
                }
                for i, p in enumerate(sorted_perfs)
            ]

        return {
            "metrics_ranked": len(rankings),
            "facilities_compared": len(set(p.facility_id for p in performance_data)),
            "rankings": rankings,
        }

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    def _handle_analyze_gaps(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Perform comprehensive gap analysis."""
        if not facility_id:
            return {"error": "facility_id is required"}

        comparisons = self._comparisons.get(facility_id, [])

        if not comparisons:
            return {
                "facility_id": facility_id,
                "message": "No comparison data available. Run compare_to_benchmark first.",
            }

        # Count by quartile
        quartile_counts = defaultdict(int)
        for comp in comparisons:
            quartile_counts[comp.quartile.value] += 1

        # Calculate overall percentile
        quartile_scores = {
            PerformanceQuartile.TOP_QUARTILE: 87.5,
            PerformanceQuartile.SECOND_QUARTILE: 62.5,
            PerformanceQuartile.THIRD_QUARTILE: 37.5,
            PerformanceQuartile.BOTTOM_QUARTILE: 12.5,
        }
        overall_percentile = sum(
            quartile_scores[comp.quartile] for comp in comparisons
        ) / len(comparisons)

        # Find largest gaps (weaknesses)
        sorted_by_gap = sorted(comparisons, key=lambda c: abs(c.gap_percent), reverse=True)
        largest_gaps = [
            {
                "metric_id": c.metric_id,
                "gap_percent": c.gap_percent,
                "quartile": c.quartile.value,
                "improvement_potential": c.improvement_potential,
            }
            for c in sorted_by_gap[:5] if not c.better_than_benchmark
        ]

        # Find strengths
        strengths = [
            {
                "metric_id": c.metric_id,
                "gap_percent": c.gap_percent,
                "quartile": c.quartile.value,
            }
            for c in comparisons if c.better_than_benchmark and c.quartile == PerformanceQuartile.TOP_QUARTILE
        ]

        # Generate improvement priorities
        priorities = []
        for gap in largest_gaps:
            if abs(gap["gap_percent"]) > 20:
                priorities.append(f"Critical: Improve {gap['metric_id']} (gap: {gap['gap_percent']:.1f}%)")
            elif abs(gap["gap_percent"]) > 10:
                priorities.append(f"High: Improve {gap['metric_id']} (gap: {gap['gap_percent']:.1f}%)")

        gap_analysis = GapAnalysis(
            facility_id=facility_id,
            metrics_analyzed=len(comparisons),
            top_quartile_count=quartile_counts.get(PerformanceQuartile.TOP_QUARTILE.value, 0),
            second_quartile_count=quartile_counts.get(PerformanceQuartile.SECOND_QUARTILE.value, 0),
            third_quartile_count=quartile_counts.get(PerformanceQuartile.THIRD_QUARTILE.value, 0),
            bottom_quartile_count=quartile_counts.get(PerformanceQuartile.BOTTOM_QUARTILE.value, 0),
            overall_percentile=round(overall_percentile, 2),
            comparisons=comparisons,
            largest_gaps=largest_gaps,
            strengths=strengths,
            improvement_priorities=priorities,
        )

        self._gap_analyses[facility_id] = gap_analysis
        self._total_analyses += 1

        return gap_analysis.model_dump()

    # =========================================================================
    # Benchmark Management
    # =========================================================================

    def _handle_get_industry_benchmarks(
        self,
        sector: Optional[IndustrySector],
        category: Optional[BenchmarkCategory],
    ) -> Dict[str, Any]:
        """Get available industry benchmarks."""
        benchmarks = self._get_applicable_benchmarks(sector, category)

        return {
            "benchmarks": [b.model_dump() for b in benchmarks],
            "count": len(benchmarks),
            "sectors": list(set(b.sector.value for b in benchmarks)),
            "categories": list(set(b.category.value for b in benchmarks)),
        }

    def _handle_add_benchmark(
        self, benchmarks: List[BenchmarkMetric]
    ) -> Dict[str, Any]:
        """Add custom benchmarks."""
        added = 0

        for benchmark in benchmarks:
            self._benchmarks[benchmark.metric_id] = benchmark
            added += 1

        return {
            "added": added,
            "total_benchmarks": len(self._benchmarks),
        }

    # =========================================================================
    # Peer Comparison
    # =========================================================================

    def _handle_get_peer_comparison(
        self,
        facility_id: Optional[str],
        performance_data: List[FacilityPerformance],
    ) -> Dict[str, Any]:
        """Compare facility to peers (other facilities in data)."""
        if not facility_id:
            return {"error": "facility_id is required"}

        # Get facility's data
        facility_perfs = [p for p in performance_data if p.facility_id == facility_id]
        peer_perfs = [p for p in performance_data if p.facility_id != facility_id]

        if not facility_perfs:
            return {"error": "No performance data for facility"}

        if not peer_perfs:
            return {"error": "No peer data for comparison"}

        # Compare by metric
        comparisons = []
        for fac_perf in facility_perfs:
            peer_values = [
                p.value for p in peer_perfs
                if p.metric_id == fac_perf.metric_id
            ]

            if peer_values:
                peer_avg = sum(peer_values) / len(peer_values)
                peer_min = min(peer_values)
                peer_max = max(peer_values)

                # Rank facility among peers
                all_values = sorted(peer_values + [fac_perf.value])
                rank = all_values.index(fac_perf.value) + 1
                percentile = (1 - rank / len(all_values)) * 100

                comparisons.append({
                    "metric_id": fac_perf.metric_id,
                    "facility_value": fac_perf.value,
                    "peer_average": round(peer_avg, 4),
                    "peer_min": peer_min,
                    "peer_max": peer_max,
                    "peer_count": len(peer_values),
                    "rank": rank,
                    "percentile": round(percentile, 2),
                    "vs_average_percent": round((fac_perf.value - peer_avg) / peer_avg * 100, 2) if peer_avg else 0,
                })

        return {
            "facility_id": facility_id,
            "peer_comparisons": comparisons,
            "total_peers": len(set(p.facility_id for p in peer_perfs)),
        }

    # =========================================================================
    # Improvement Potential
    # =========================================================================

    def _handle_get_improvement_potential(
        self, facility_id: Optional[str]
    ) -> Dict[str, Any]:
        """Calculate improvement potential to reach benchmarks."""
        if not facility_id:
            return {"error": "facility_id is required"}

        comparisons = self._comparisons.get(facility_id, [])

        if not comparisons:
            return {"facility_id": facility_id, "message": "No comparison data available"}

        improvement_opportunities = []
        total_improvement_potential = 0.0

        for comp in comparisons:
            if comp.improvement_potential > 0:
                improvement_opportunities.append({
                    "metric_id": comp.metric_id,
                    "current_value": comp.facility_value,
                    "target_value": comp.benchmark_value,
                    "improvement_potential": comp.improvement_potential,
                    "current_quartile": comp.quartile.value,
                })
                total_improvement_potential += comp.improvement_potential

        # Sort by potential
        improvement_opportunities.sort(
            key=lambda x: x["improvement_potential"],
            reverse=True
        )

        return {
            "facility_id": facility_id,
            "improvement_opportunities": improvement_opportunities,
            "total_improvement_potential": round(total_improvement_potential, 4),
            "opportunities_count": len(improvement_opportunities),
        }

    # =========================================================================
    # Trends
    # =========================================================================

    def _handle_get_trends(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get performance trends over time."""
        if not facility_id:
            return {"error": "facility_id is required"}

        comparisons = self._comparisons.get(facility_id, [])

        if len(comparisons) < 2:
            return {
                "facility_id": facility_id,
                "message": "Insufficient data for trend analysis",
            }

        # Group by metric and analyze trend
        by_metric: Dict[str, List[BenchmarkComparison]] = defaultdict(list)
        for comp in comparisons:
            by_metric[comp.metric_id].append(comp)

        trends = {}
        for metric_id, comps in by_metric.items():
            sorted_comps = sorted(comps, key=lambda c: c.comparison_date)

            if len(sorted_comps) >= 2:
                first = sorted_comps[0]
                last = sorted_comps[-1]

                gap_change = last.gap_percent - first.gap_percent
                improving = gap_change < 0  # Gap is shrinking

                trends[metric_id] = {
                    "first_gap_percent": first.gap_percent,
                    "latest_gap_percent": last.gap_percent,
                    "gap_change": round(gap_change, 2),
                    "improving": improving,
                    "data_points": len(sorted_comps),
                }

        return {
            "facility_id": facility_id,
            "trends": trends,
            "metrics_with_trends": len(trends),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_benchmarks": len(self._benchmarks),
            "total_comparisons": self._total_comparisons,
            "total_analyses": self._total_analyses,
            "facilities_tracked": len(self._comparisons),
            "benchmarks_by_category": {
                cat.value: sum(1 for b in self._benchmarks.values() if b.category == cat)
                for cat in BenchmarkCategory
            },
            "benchmarks_by_sector": {
                sec.value: sum(1 for b in self._benchmarks.values() if b.sector == sec)
                for sec in IndustrySector
            },
        }

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
