"""GL-084: Sustainability Reporter Agent (SUSTAINABILITY-REPORTER).

Generates sustainability reports for stakeholders.

Standards: GRI Standards, SASB, TCFD
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReportFramework(str, Enum):
    GRI = "GRI"
    SASB = "SASB"
    TCFD = "TCFD"
    CDP = "CDP"
    INTEGRATED = "INTEGRATED"


class MetricCategory(str, Enum):
    EMISSIONS = "EMISSIONS"
    ENERGY = "ENERGY"
    WATER = "WATER"
    WASTE = "WASTE"
    SOCIAL = "SOCIAL"


class SustainabilityMetric(BaseModel):
    metric_id: str
    category: MetricCategory
    name: str
    value: float
    unit: str
    year_over_year_change_pct: float = Field(default=0)
    target_value: Optional[float] = None
    gri_indicator: Optional[str] = None


class SustainabilityReporterInput(BaseModel):
    organization_id: str
    organization_name: str = Field(default="Organization")
    reporting_year: int = Field(default=2024)
    framework: ReportFramework = Field(default=ReportFramework.GRI)
    metrics: List[SustainabilityMetric] = Field(default_factory=list)
    revenue_usd: float = Field(default=100000000, gt=0)
    employees: int = Field(default=500, ge=1)
    facilities_count: int = Field(default=5, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CategorySummary(BaseModel):
    category: str
    metrics_count: int
    targets_met: int
    targets_missed: int
    improvement_pct: float


class SustainabilityReporterOutput(BaseModel):
    organization_id: str
    reporting_year: int
    framework: str
    total_metrics: int
    category_summaries: List[CategorySummary]
    emissions_intensity: float
    energy_intensity: float
    water_intensity: float
    overall_improvement_pct: float
    targets_achievement_pct: float
    disclosure_completeness_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class SustainabilityReporterAgent:
    AGENT_ID = "GL-084B"
    AGENT_NAME = "SUSTAINABILITY-REPORTER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"SustainabilityReporterAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = SustainabilityReporterInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: SustainabilityReporterInput) -> SustainabilityReporterOutput:
        recommendations = []

        # Group by category
        categories = {}
        for metric in inp.metrics:
            cat = metric.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(metric)

        # Calculate category summaries
        summaries = []
        total_improvement = 0
        targets_met = 0
        targets_total = 0

        for cat, metrics in categories.items():
            met = sum(1 for m in metrics if m.target_value and m.value >= m.target_value)
            missed = sum(1 for m in metrics if m.target_value and m.value < m.target_value)
            improvement = sum(m.year_over_year_change_pct for m in metrics) / len(metrics) if metrics else 0

            summaries.append(CategorySummary(
                category=cat,
                metrics_count=len(metrics),
                targets_met=met,
                targets_missed=missed,
                improvement_pct=round(improvement, 1)
            ))

            total_improvement += improvement
            targets_met += met
            targets_total += met + missed

        # Calculate intensities
        revenue_m = inp.revenue_usd / 1000000

        emissions_metrics = [m for m in inp.metrics if m.category == MetricCategory.EMISSIONS]
        emissions_total = sum(m.value for m in emissions_metrics if "CO2" in m.unit.upper() or "TONNE" in m.unit.upper())
        emissions_intensity = emissions_total / revenue_m if revenue_m > 0 else 0

        energy_metrics = [m for m in inp.metrics if m.category == MetricCategory.ENERGY]
        energy_total = sum(m.value for m in energy_metrics if "MWH" in m.unit.upper() or "KWH" in m.unit.upper())
        energy_intensity = energy_total / revenue_m if revenue_m > 0 else 0

        water_metrics = [m for m in inp.metrics if m.category == MetricCategory.WATER]
        water_total = sum(m.value for m in water_metrics if "M3" in m.unit.upper() or "GAL" in m.unit.upper())
        water_intensity = water_total / revenue_m if revenue_m > 0 else 0

        # Overall metrics
        overall_improvement = total_improvement / len(categories) if categories else 0
        targets_achievement = (targets_met / targets_total * 100) if targets_total > 0 else 100

        # Disclosure completeness (based on expected GRI indicators)
        expected_indicators = 20  # Simplified
        actual_indicators = len([m for m in inp.metrics if m.gri_indicator])
        disclosure_completeness = min(100, actual_indicators / expected_indicators * 100)

        # Recommendations
        if disclosure_completeness < 70:
            recommendations.append(f"Disclosure completeness {disclosure_completeness:.0f}% - add more GRI indicators")
        if targets_achievement < 80:
            recommendations.append(f"Target achievement {targets_achievement:.0f}% - review target-setting process")
        if overall_improvement < 0:
            recommendations.append("Overall performance declining - immediate action required")

        categories_missing = set(["EMISSIONS", "ENERGY", "WATER", "WASTE"]) - set(categories.keys())
        if categories_missing:
            recommendations.append(f"Missing categories: {', '.join(categories_missing)}")

        if inp.framework == ReportFramework.TCFD and not any(m.category == MetricCategory.EMISSIONS for m in inp.metrics):
            recommendations.append("TCFD reporting requires climate metrics - add emissions data")

        calc_hash = hashlib.sha256(json.dumps({
            "organization": inp.organization_id,
            "year": inp.reporting_year,
            "metrics": len(inp.metrics)
        }).encode()).hexdigest()

        return SustainabilityReporterOutput(
            organization_id=inp.organization_id,
            reporting_year=inp.reporting_year,
            framework=inp.framework.value,
            total_metrics=len(inp.metrics),
            category_summaries=summaries,
            emissions_intensity=round(emissions_intensity, 2),
            energy_intensity=round(energy_intensity, 2),
            water_intensity=round(water_intensity, 2),
            overall_improvement_pct=round(overall_improvement, 1),
            targets_achievement_pct=round(targets_achievement, 1),
            disclosure_completeness_pct=round(disclosure_completeness, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-084B", "name": "SUSTAINABILITY-REPORTER", "version": "1.0.0",
    "summary": "Sustainability reporting generation",
    "standards": [{"ref": "GRI Standards"}, {"ref": "SASB"}, {"ref": "TCFD"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
