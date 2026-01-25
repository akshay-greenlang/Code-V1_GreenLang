"""GL-085: Benchmark Comparator Agent (BENCHMARK-COMPARATOR).

Compares facility performance against industry benchmarks.

Standards: ENERGY STAR, EPA, ISO 50001
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IndustryType(str, Enum):
    MANUFACTURING = "MANUFACTURING"
    COMMERCIAL = "COMMERCIAL"
    HEALTHCARE = "HEALTHCARE"
    DATA_CENTER = "DATA_CENTER"
    WAREHOUSE = "WAREHOUSE"


class BenchmarkMetric(BaseModel):
    metric_id: str
    metric_name: str
    facility_value: float
    benchmark_25th: float  # Bottom quartile
    benchmark_50th: float  # Median
    benchmark_75th: float  # Top quartile
    benchmark_best: float  # Best in class
    unit: str
    lower_is_better: bool = Field(default=True)


class BenchmarkComparatorInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    industry: IndustryType = Field(default=IndustryType.MANUFACTURING)
    floor_area_m2: float = Field(default=10000, gt=0)
    annual_production_units: float = Field(default=100000, ge=0)
    metrics: List[BenchmarkMetric] = Field(default_factory=list)
    energy_star_score: Optional[int] = Field(default=None, ge=1, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricComparison(BaseModel):
    metric_name: str
    facility_value: float
    percentile: int
    gap_to_median_pct: float
    gap_to_best_pct: float
    improvement_potential: str
    unit: str


class BenchmarkComparatorOutput(BaseModel):
    facility_id: str
    industry: str
    overall_percentile: int
    energy_star_score: Optional[int]
    comparisons: List[MetricComparison]
    metrics_above_median: int
    metrics_below_median: int
    top_improvement_areas: List[str]
    estimated_savings_if_median_pct: float
    estimated_savings_if_best_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class BenchmarkComparatorAgent:
    AGENT_ID = "GL-085"
    AGENT_NAME = "BENCHMARK-COMPARATOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"BenchmarkComparatorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = BenchmarkComparatorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_percentile(self, value: float, p25: float, p50: float, p75: float, best: float, lower_is_better: bool) -> int:
        """Estimate percentile from quartile values."""
        if lower_is_better:
            if value <= best:
                return 99
            elif value <= p75:
                return 75 + int(25 * (p75 - value) / (p75 - best)) if p75 != best else 75
            elif value <= p50:
                return 50 + int(25 * (p50 - value) / (p50 - p75)) if p50 != p75 else 50
            elif value <= p25:
                return 25 + int(25 * (p25 - value) / (p25 - p50)) if p25 != p50 else 25
            else:
                return max(1, 25 - int(25 * (value - p25) / p25)) if p25 > 0 else 1
        else:
            if value >= best:
                return 99
            elif value >= p75:
                return 75 + int(25 * (value - p75) / (best - p75)) if best != p75 else 75
            elif value >= p50:
                return 50 + int(25 * (value - p50) / (p75 - p50)) if p75 != p50 else 50
            elif value >= p25:
                return 25 + int(25 * (value - p25) / (p50 - p25)) if p50 != p25 else 25
            else:
                return max(1, int(25 * value / p25)) if p25 > 0 else 1

    def _process(self, inp: BenchmarkComparatorInput) -> BenchmarkComparatorOutput:
        recommendations = []
        comparisons = []
        percentiles = []
        above_median = 0
        below_median = 0
        improvement_areas = []
        savings_to_median = []
        savings_to_best = []

        for metric in inp.metrics:
            percentile = self._calculate_percentile(
                metric.facility_value,
                metric.benchmark_25th,
                metric.benchmark_50th,
                metric.benchmark_75th,
                metric.benchmark_best,
                metric.lower_is_better
            )
            percentiles.append(percentile)

            # Gap calculations
            if metric.lower_is_better:
                gap_median = ((metric.facility_value - metric.benchmark_50th) / metric.benchmark_50th * 100) if metric.benchmark_50th > 0 else 0
                gap_best = ((metric.facility_value - metric.benchmark_best) / metric.benchmark_best * 100) if metric.benchmark_best > 0 else 0
            else:
                gap_median = ((metric.benchmark_50th - metric.facility_value) / metric.facility_value * 100) if metric.facility_value > 0 else 0
                gap_best = ((metric.benchmark_best - metric.facility_value) / metric.facility_value * 100) if metric.facility_value > 0 else 0

            # Improvement potential
            if percentile >= 75:
                potential = "LOW - Top quartile"
            elif percentile >= 50:
                potential = "MEDIUM - Above median"
            elif percentile >= 25:
                potential = "HIGH - Below median"
            else:
                potential = "VERY HIGH - Bottom quartile"

            comparisons.append(MetricComparison(
                metric_name=metric.metric_name,
                facility_value=round(metric.facility_value, 2),
                percentile=percentile,
                gap_to_median_pct=round(gap_median, 1),
                gap_to_best_pct=round(gap_best, 1),
                improvement_potential=potential,
                unit=metric.unit
            ))

            if percentile >= 50:
                above_median += 1
            else:
                below_median += 1
                improvement_areas.append((metric.metric_name, gap_median))
                savings_to_median.append(abs(gap_median))

            savings_to_best.append(abs(gap_best))

        # Sort improvement areas by gap
        improvement_areas.sort(key=lambda x: -x[1])
        top_areas = [f"{name}: {gap:.1f}% below median" for name, gap in improvement_areas[:3]]

        # Overall percentile
        overall_percentile = int(sum(percentiles) / len(percentiles)) if percentiles else 50

        # Savings estimates
        avg_savings_median = sum(savings_to_median) / len(savings_to_median) if savings_to_median else 0
        avg_savings_best = sum(savings_to_best) / len(savings_to_best) if savings_to_best else 0

        # Recommendations
        if overall_percentile < 25:
            recommendations.append("Performance in bottom quartile - implement comprehensive improvement program")
        elif overall_percentile < 50:
            recommendations.append("Below median performance - prioritize top improvement areas")
        elif overall_percentile < 75:
            recommendations.append("Above median - focus on reaching top quartile")
        else:
            recommendations.append("Top quartile performance - maintain best practices")

        if below_median > above_median:
            recommendations.append(f"{below_median} metrics below median - systematic improvement needed")

        if top_areas:
            recommendations.append(f"Top priority: {top_areas[0]}")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "percentile": overall_percentile,
            "metrics": len(inp.metrics)
        }).encode()).hexdigest()

        return BenchmarkComparatorOutput(
            facility_id=inp.facility_id,
            industry=inp.industry.value,
            overall_percentile=overall_percentile,
            energy_star_score=inp.energy_star_score,
            comparisons=comparisons,
            metrics_above_median=above_median,
            metrics_below_median=below_median,
            top_improvement_areas=top_areas,
            estimated_savings_if_median_pct=round(avg_savings_median, 1),
            estimated_savings_if_best_pct=round(avg_savings_best, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-085", "name": "BENCHMARK-COMPARATOR", "version": "1.0.0",
    "summary": "Performance benchmarking against industry standards",
    "standards": [{"ref": "ENERGY STAR"}, {"ref": "EPA"}, {"ref": "ISO 50001"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
