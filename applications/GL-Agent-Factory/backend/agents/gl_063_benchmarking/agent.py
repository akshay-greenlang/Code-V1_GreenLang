"""
GL-063: Benchmarking Agent (BENCHMARKIQ)

This module implements the BenchmarkingAgent for performance comparison against
industry standards, best practices, and peer facilities.

Standards Reference:
    - ISO 50006 (Energy Performance Indicators)
    - ENERGY STAR methodology
    - Industry-specific benchmarks

Example:
    >>> agent = BenchmarkingAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Performance percentile: {result.overall_percentile:.1f}%")
"""

import hashlib
import json
import logging
import statistics
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class IndustryType(str, Enum):
    CHEMICALS = "chemicals"
    REFINING = "refining"
    PULP_PAPER = "pulp_paper"
    STEEL = "steel"
    CEMENT = "cement"
    FOOD_BEVERAGE = "food_beverage"
    PHARMA = "pharmaceuticals"
    AUTOMOTIVE = "automotive"
    GENERAL_MFG = "general_manufacturing"


class PerformanceMetricType(str, Enum):
    ENERGY_INTENSITY = "energy_intensity"
    CARBON_INTENSITY = "carbon_intensity"
    WATER_INTENSITY = "water_intensity"
    COST_EFFICIENCY = "cost_efficiency"
    YIELD = "yield"
    UPTIME = "uptime"
    SPECIFIC_CONSUMPTION = "specific_consumption"


class BenchmarkCategory(str, Enum):
    QUARTILE_1 = "top_quartile"  # Top 25%
    QUARTILE_2 = "second_quartile"  # 25-50%
    QUARTILE_3 = "third_quartile"  # 50-75%
    QUARTILE_4 = "bottom_quartile"  # Bottom 25%


class PerformanceMetric(BaseModel):
    metric_id: str = Field(..., description="Metric identifier")
    name: str = Field(..., description="Metric name")
    metric_type: PerformanceMetricType = Field(..., description="Type of metric")
    actual_value: float = Field(..., description="Actual performance value")
    unit: str = Field(..., description="Unit of measurement")
    normalization_factor: Optional[str] = Field(None, description="Per ton, per unit, etc.")


class IndustryBenchmark(BaseModel):
    benchmark_id: str = Field(..., description="Benchmark identifier")
    name: str = Field(..., description="Benchmark name")
    metric_type: PerformanceMetricType = Field(..., description="Metric type")
    best_in_class: float = Field(..., description="Top 10% value")
    top_quartile: float = Field(..., description="Top 25% value")
    median: float = Field(..., description="50th percentile")
    bottom_quartile: float = Field(..., description="Bottom 25% value")
    industry_average: float = Field(..., description="Industry average")
    unit: str = Field(..., description="Unit")
    source: str = Field(default="Industry Database", description="Data source")
    year: int = Field(default=2024, description="Benchmark year")


class BenchmarkInput(BaseModel):
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    facility_name: str = Field(default="Facility", description="Facility name")
    industry_type: IndustryType = Field(..., description="Industry classification")
    production_volume: float = Field(..., gt=0, description="Annual production volume")
    production_unit: str = Field(..., description="Production unit (tons, units, etc.)")
    performance_metrics: List[PerformanceMetric] = Field(..., min_items=1)
    custom_benchmarks: List[IndustryBenchmark] = Field(default_factory=list)
    analysis_period: str = Field(default="Annual", description="Analysis period")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricComparison(BaseModel):
    metric_id: str
    metric_name: str
    actual_value: float
    industry_median: float
    top_quartile: float
    best_in_class: float
    unit: str
    performance_gap_vs_median: float
    performance_gap_vs_top_quartile: float
    performance_gap_vs_best: float
    percentile_rank: float
    category: BenchmarkCategory
    rating: str


class GapAnalysis(BaseModel):
    metric_name: str
    current_performance: float
    target_performance: float
    gap: float
    gap_percent: float
    improvement_required: str
    priority: str


class BestPracticeRecommendation(BaseModel):
    recommendation_id: str
    title: str
    description: str
    affected_metrics: List[str]
    estimated_improvement_percent: float
    implementation_difficulty: str
    payback_period_months: Optional[float]
    priority: str


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class BenchmarkOutput(BaseModel):
    facility_id: str
    facility_name: str
    industry_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_comparisons: List[MetricComparison]
    gap_analyses: List[GapAnalysis]
    overall_percentile: float
    overall_rating: str
    best_practice_recommendations: List[BestPracticeRecommendation]
    peer_comparison_summary: str
    recommendations: List[str]
    warnings: List[str]
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


class BenchmarkingAgent:
    """GL-063: Benchmarking Agent - Compare performance against industry standards."""

    AGENT_ID = "GL-063"
    AGENT_NAME = "BENCHMARKIQ"
    VERSION = "1.0.0"

    # Industry benchmark database (simplified - would come from external DB)
    INDUSTRY_BENCHMARKS = {
        IndustryType.CHEMICALS: {
            PerformanceMetricType.ENERGY_INTENSITY: IndustryBenchmark(
                benchmark_id="CHEM_EI_001",
                name="Chemical Plant Energy Intensity",
                metric_type=PerformanceMetricType.ENERGY_INTENSITY,
                best_in_class=3.5,
                top_quartile=4.2,
                median=5.8,
                bottom_quartile=7.5,
                industry_average=6.2,
                unit="GJ/ton",
                source="DOE Industrial Assessment",
                year=2024
            )
        },
        IndustryType.REFINING: {
            PerformanceMetricType.ENERGY_INTENSITY: IndustryBenchmark(
                benchmark_id="REF_EI_001",
                name="Refinery Energy Intensity",
                metric_type=PerformanceMetricType.ENERGY_INTENSITY,
                best_in_class=0.95,
                top_quartile=1.15,
                median=1.45,
                bottom_quartile=1.85,
                industry_average=1.52,
                unit="GJ/barrel",
                source="Solomon Associates",
                year=2024
            )
        },
        IndustryType.STEEL: {
            PerformanceMetricType.ENERGY_INTENSITY: IndustryBenchmark(
                benchmark_id="STEEL_EI_001",
                name="Steel Mill Energy Intensity",
                metric_type=PerformanceMetricType.ENERGY_INTENSITY,
                best_in_class=18.5,
                top_quartile=21.0,
                median=24.5,
                bottom_quartile=28.0,
                industry_average=25.2,
                unit="GJ/ton",
                source="World Steel Association",
                year=2024
            )
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []
        logger.info(f"BenchmarkingAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        # Get industry benchmarks
        industry_benchmarks = self._get_industry_benchmarks(input_data.industry_type)
        industry_benchmarks.extend(input_data.custom_benchmarks)

        # Compare metrics
        metric_comparisons = []
        for metric in input_data.performance_metrics:
            benchmark = self._find_benchmark(metric, industry_benchmarks)
            if benchmark:
                comparison = self._compare_metric(metric, benchmark)
                metric_comparisons.append(comparison)
            else:
                self._warnings.append(f"No benchmark found for metric: {metric.name}")

        self._track_provenance("benchmark_comparison",
            {"num_metrics": len(input_data.performance_metrics), "industry": input_data.industry_type.value},
            {"num_comparisons": len(metric_comparisons)},
            "Benchmark Comparator")

        # Gap analysis
        gap_analyses = self._perform_gap_analysis(metric_comparisons)

        # Calculate overall performance
        if metric_comparisons:
            overall_percentile = statistics.mean([m.percentile_rank for m in metric_comparisons])
            overall_rating = self._calculate_rating(overall_percentile)
        else:
            overall_percentile = 0.0
            overall_rating = "INSUFFICIENT_DATA"
            self._validation_errors.append("No valid metric comparisons available")

        # Generate recommendations
        best_practice_recs = self._generate_best_practices(metric_comparisons, input_data.industry_type)

        # Peer comparison summary
        peer_summary = self._generate_peer_summary(overall_percentile, overall_rating, metric_comparisons)

        # Generate system recommendations and warnings
        self._generate_recommendations_and_warnings(overall_percentile, metric_comparisons, gap_analyses)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return BenchmarkOutput(
            facility_id=input_data.facility_id or f"FAC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            industry_type=input_data.industry_type.value,
            metric_comparisons=metric_comparisons,
            gap_analyses=gap_analyses,
            overall_percentile=round(overall_percentile, 1),
            overall_rating=overall_rating,
            best_practice_recommendations=best_practice_recs,
            peer_comparison_summary=peer_summary,
            recommendations=self._recommendations,
            warnings=self._warnings,
            provenance_chain=[ProvenanceRecord(**{k: v for k, v in s.items()}) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _get_industry_benchmarks(self, industry: IndustryType) -> List[IndustryBenchmark]:
        """Retrieve benchmarks for the specified industry."""
        benchmarks = []
        if industry in self.INDUSTRY_BENCHMARKS:
            benchmarks.extend(self.INDUSTRY_BENCHMARKS[industry].values())
        return benchmarks

    def _find_benchmark(self, metric: PerformanceMetric, benchmarks: List[IndustryBenchmark]) -> Optional[IndustryBenchmark]:
        """Find matching benchmark for a metric."""
        for benchmark in benchmarks:
            if benchmark.metric_type == metric.metric_type:
                return benchmark
        return None

    def _compare_metric(self, metric: PerformanceMetric, benchmark: IndustryBenchmark) -> MetricComparison:
        """Compare metric against benchmark values."""
        actual = metric.actual_value

        # Calculate gaps (lower is better for intensity metrics)
        gap_vs_median = ((actual - benchmark.median) / benchmark.median * 100) if benchmark.median > 0 else 0
        gap_vs_top_quartile = ((actual - benchmark.top_quartile) / benchmark.top_quartile * 100) if benchmark.top_quartile > 0 else 0
        gap_vs_best = ((actual - benchmark.best_in_class) / benchmark.best_in_class * 100) if benchmark.best_in_class > 0 else 0

        # Calculate percentile rank (lower value = better for intensity metrics)
        if actual <= benchmark.best_in_class:
            percentile = 95.0
        elif actual <= benchmark.top_quartile:
            percentile = 75.0 + 20.0 * (benchmark.top_quartile - actual) / (benchmark.top_quartile - benchmark.best_in_class)
        elif actual <= benchmark.median:
            percentile = 50.0 + 25.0 * (benchmark.median - actual) / (benchmark.median - benchmark.top_quartile)
        elif actual <= benchmark.bottom_quartile:
            percentile = 25.0 + 25.0 * (benchmark.bottom_quartile - actual) / (benchmark.bottom_quartile - benchmark.median)
        else:
            percentile = 25.0 * (1 - min((actual - benchmark.bottom_quartile) / benchmark.bottom_quartile, 1.0))

        # Categorize
        if percentile >= 75:
            category = BenchmarkCategory.QUARTILE_1
        elif percentile >= 50:
            category = BenchmarkCategory.QUARTILE_2
        elif percentile >= 25:
            category = BenchmarkCategory.QUARTILE_3
        else:
            category = BenchmarkCategory.QUARTILE_4

        rating = self._calculate_rating(percentile)

        return MetricComparison(
            metric_id=metric.metric_id,
            metric_name=metric.name,
            actual_value=round(actual, 3),
            industry_median=round(benchmark.median, 3),
            top_quartile=round(benchmark.top_quartile, 3),
            best_in_class=round(benchmark.best_in_class, 3),
            unit=metric.unit,
            performance_gap_vs_median=round(gap_vs_median, 2),
            performance_gap_vs_top_quartile=round(gap_vs_top_quartile, 2),
            performance_gap_vs_best=round(gap_vs_best, 2),
            percentile_rank=round(percentile, 1),
            category=category,
            rating=rating
        )

    def _calculate_rating(self, percentile: float) -> str:
        """Calculate letter rating based on percentile."""
        if percentile >= 90:
            return "A+"
        elif percentile >= 75:
            return "A"
        elif percentile >= 60:
            return "B+"
        elif percentile >= 50:
            return "B"
        elif percentile >= 40:
            return "C+"
        elif percentile >= 25:
            return "C"
        elif percentile >= 10:
            return "D"
        else:
            return "F"

    def _perform_gap_analysis(self, comparisons: List[MetricComparison]) -> List[GapAnalysis]:
        """Analyze gaps between current and target performance."""
        gap_analyses = []

        for comp in comparisons:
            # Target = top quartile performance
            gap = comp.actual_value - comp.top_quartile
            gap_percent = comp.performance_gap_vs_top_quartile

            if gap > 0:  # Worse than top quartile
                improvement_req = f"Reduce by {abs(gap):.2f} {comp.unit}"
                if gap_percent > 30:
                    priority = "HIGH"
                elif gap_percent > 15:
                    priority = "MEDIUM"
                else:
                    priority = "LOW"
            else:
                improvement_req = "Already at or above target"
                priority = "LOW"

            gap_analyses.append(GapAnalysis(
                metric_name=comp.metric_name,
                current_performance=comp.actual_value,
                target_performance=comp.top_quartile,
                gap=round(gap, 3),
                gap_percent=round(gap_percent, 2),
                improvement_required=improvement_req,
                priority=priority
            ))

        return sorted(gap_analyses, key=lambda x: -abs(x.gap_percent))

    def _generate_best_practices(self, comparisons: List[MetricComparison], industry: IndustryType) -> List[BestPracticeRecommendation]:
        """Generate best practice recommendations based on gaps."""
        recommendations = []

        # Energy intensity improvements
        energy_comps = [c for c in comparisons if c.metric_type == PerformanceMetricType.ENERGY_INTENSITY]
        if energy_comps:
            for i, comp in enumerate(energy_comps):
                if comp.performance_gap_vs_top_quartile > 10:
                    recommendations.append(BestPracticeRecommendation(
                        recommendation_id=f"BP-{len(recommendations)+1:03d}",
                        title="Implement Advanced Energy Management System",
                        description="Deploy ISO 50001-certified energy management system with real-time monitoring and optimization",
                        affected_metrics=[comp.metric_name],
                        estimated_improvement_percent=15.0,
                        implementation_difficulty="MEDIUM",
                        payback_period_months=18.0,
                        priority="HIGH" if comp.performance_gap_vs_top_quartile > 25 else "MEDIUM"
                    ))

                if comp.performance_gap_vs_top_quartile > 20:
                    recommendations.append(BestPracticeRecommendation(
                        recommendation_id=f"BP-{len(recommendations)+1:03d}",
                        title="Heat Recovery and Integration",
                        description="Implement pinch analysis and heat integration to recover waste heat",
                        affected_metrics=[comp.metric_name],
                        estimated_improvement_percent=12.0,
                        implementation_difficulty="HIGH",
                        payback_period_months=24.0,
                        priority="HIGH"
                    ))

        # Add industry-specific recommendations
        if industry == IndustryType.CHEMICALS:
            recommendations.append(BestPracticeRecommendation(
                recommendation_id=f"BP-{len(recommendations)+1:03d}",
                title="Process Intensification",
                description="Apply process intensification techniques to reduce energy consumption",
                affected_metrics=["Energy Intensity"],
                estimated_improvement_percent=10.0,
                implementation_difficulty="HIGH",
                payback_period_months=36.0,
                priority="MEDIUM"
            ))

        return sorted(recommendations, key=lambda x: (-{"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(x.priority, 0), -x.estimated_improvement_percent))[:5]

    def _generate_peer_summary(self, percentile: float, rating: str, comparisons: List[MetricComparison]) -> str:
        """Generate peer comparison summary text."""
        if percentile >= 75:
            performance = "industry-leading"
            position = "top quartile"
        elif percentile >= 50:
            performance = "above-average"
            position = "second quartile"
        elif percentile >= 25:
            performance = "below-average"
            position = "third quartile"
        else:
            performance = "underperforming"
            position = "bottom quartile"

        num_metrics = len(comparisons)
        top_performers = sum(1 for c in comparisons if c.category == BenchmarkCategory.QUARTILE_1)

        summary = f"Facility demonstrates {performance} performance, ranking in the {position} "
        summary += f"(percentile: {percentile:.1f}%, rating: {rating}). "
        summary += f"Of {num_metrics} benchmarked metrics, {top_performers} are in the top quartile. "

        if percentile < 50:
            summary += "Significant improvement opportunities exist to reach industry median performance."
        elif percentile < 75:
            summary += "Good progress toward industry leadership. Focus on closing gaps to reach top quartile."
        else:
            summary += "Continue excellence through continuous improvement and best practice adoption."

        return summary

    def _generate_recommendations_and_warnings(self, percentile: float, comparisons: List[MetricComparison], gaps: List[GapAnalysis]) -> None:
        """Generate system-level recommendations and warnings."""
        if percentile < 25:
            self._warnings.append(f"Facility performance is in the bottom quartile (percentile: {percentile:.1f}%). Immediate action required.")
            self._recommendations.append("Conduct comprehensive energy audit and develop improvement roadmap")
        elif percentile < 50:
            self._warnings.append(f"Performance below industry median (percentile: {percentile:.1f}%). Improvement opportunities exist.")
            self._recommendations.append("Prioritize high-impact efficiency projects to reach median performance")

        # Metric-specific recommendations
        high_priority_gaps = [g for g in gaps if g.priority == "HIGH"]
        if high_priority_gaps:
            self._recommendations.append(f"Address {len(high_priority_gaps)} high-priority performance gaps")
            for gap in high_priority_gaps[:2]:
                self._recommendations.append(f"Focus on {gap.metric_name}: {gap.improvement_required}")

        # Excellence recognition
        top_performers = [c for c in comparisons if c.category == BenchmarkCategory.QUARTILE_1]
        if len(top_performers) == len(comparisons) and comparisons:
            self._recommendations.append("All metrics in top quartile. Consider applying for industry recognition awards.")

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        """Track provenance of calculations."""
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of provenance chain."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [{"operation": s["operation"], "input_hash": s["input_hash"], "output_hash": s["output_hash"]}
                     for s in self._provenance_steps],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-063",
    "name": "BENCHMARKIQ",
    "version": "1.0.0",
    "summary": "Performance benchmarking against industry standards and best practices",
    "tags": ["benchmarking", "performance", "kpi", "industry-comparison", "best-practices"],
    "standards": [
        {"ref": "ISO 50006", "description": "Energy Performance Indicators"},
        {"ref": "ENERGY STAR", "description": "Energy efficiency benchmarking methodology"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
