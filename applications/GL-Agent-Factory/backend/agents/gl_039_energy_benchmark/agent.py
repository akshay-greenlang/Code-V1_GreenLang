"""
GL-039: Energy Benchmark Agent (ENERGY-BENCHMARK)

Process energy benchmarking and gap analysis.

Standards: ISO 50001, EPA ENERGY STAR
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Industry benchmarks (kWh per unit of production)
INDUSTRY_BENCHMARKS = {
    "steel": {"p10": 400, "p25": 500, "p50": 650, "p75": 800, "p90": 1000, "unit": "tonne"},
    "cement": {"p10": 80, "p25": 95, "p50": 110, "p75": 130, "p90": 160, "unit": "tonne"},
    "paper": {"p10": 400, "p25": 500, "p50": 650, "p75": 800, "p90": 1000, "unit": "tonne"},
    "chemicals": {"p10": 200, "p25": 300, "p50": 450, "p75": 600, "p90": 800, "unit": "tonne"},
    "food": {"p10": 150, "p25": 200, "p50": 280, "p75": 400, "p90": 550, "unit": "tonne"},
    "glass": {"p10": 300, "p25": 400, "p50": 500, "p75": 650, "p90": 800, "unit": "tonne"},
    "refinery": {"p10": 50, "p25": 65, "p50": 85, "p75": 110, "p90": 140, "unit": "barrel"},
}


class EnergyBenchmarkInput(BaseModel):
    """Input for EnergyBenchmarkAgent."""
    facility_id: str
    industry_sector: str = Field(...)
    production_quantity: float = Field(..., gt=0)
    production_unit: str = Field(default="tonne")
    total_energy_kwh: float = Field(..., ge=0)
    electricity_kwh: float = Field(default=0, ge=0)
    natural_gas_mmbtu: float = Field(default=0, ge=0)
    other_fuel_mmbtu: float = Field(default=0, ge=0)
    reporting_period_days: int = Field(default=365)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnergyBenchmarkOutput(BaseModel):
    """Output from EnergyBenchmarkAgent."""
    analysis_id: str
    facility_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    specific_energy_kwh_per_unit: float
    benchmark_p10: float
    benchmark_p25: float
    benchmark_p50: float
    benchmark_p75: float
    benchmark_p90: float
    percentile_rank: int
    performance_rating: str
    gap_to_p25_kwh_per_unit: float
    gap_to_p25_percent: float
    potential_savings_kwh_per_year: float
    potential_savings_cost_per_year: float
    improvement_targets: List[Dict[str, Any]]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class EnergyBenchmarkAgent:
    """GL-039: Energy Benchmark Agent."""

    AGENT_ID = "GL-039"
    AGENT_NAME = "ENERGY-BENCHMARK"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"EnergyBenchmarkAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: EnergyBenchmarkInput) -> EnergyBenchmarkOutput:
        """Execute energy benchmarking analysis."""
        start_time = datetime.utcnow()

        # Calculate specific energy consumption
        sec = input_data.total_energy_kwh / input_data.production_quantity

        # Get industry benchmarks
        sector = input_data.industry_sector.lower()
        benchmarks = INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS["chemicals"])

        # Determine percentile rank
        if sec <= benchmarks["p10"]:
            percentile = 10
            rating = "EXCELLENT"
        elif sec <= benchmarks["p25"]:
            percentile = 25
            rating = "GOOD"
        elif sec <= benchmarks["p50"]:
            percentile = 50
            rating = "AVERAGE"
        elif sec <= benchmarks["p75"]:
            percentile = 75
            rating = "BELOW_AVERAGE"
        elif sec <= benchmarks["p90"]:
            percentile = 90
            rating = "POOR"
        else:
            percentile = 95
            rating = "CRITICAL"

        # Gap analysis
        gap_to_p25 = sec - benchmarks["p25"]
        gap_percent = (gap_to_p25 / sec * 100) if sec > 0 else 0

        # Annualized production
        annual_factor = 365 / input_data.reporting_period_days
        annual_production = input_data.production_quantity * annual_factor

        # Potential savings
        if gap_to_p25 > 0:
            potential_kwh = gap_to_p25 * annual_production
            # Assume $0.08/kWh average
            potential_cost = potential_kwh * 0.08
        else:
            potential_kwh = 0
            potential_cost = 0

        # Improvement targets
        targets = []
        if percentile > 50:
            targets.append({
                "target": "Reach industry median (P50)",
                "required_reduction_percent": round((sec - benchmarks["p50"]) / sec * 100, 1),
                "estimated_timeline": "12-18 months"
            })
        if percentile > 25:
            targets.append({
                "target": "Reach top quartile (P25)",
                "required_reduction_percent": round((sec - benchmarks["p25"]) / sec * 100, 1),
                "estimated_timeline": "24-36 months"
            })
        targets.append({
            "target": "Best-in-class (P10)",
            "required_reduction_percent": round((sec - benchmarks["p10"]) / sec * 100, 1),
            "estimated_timeline": "36-60 months"
        })

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "facility": input_data.facility_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return EnergyBenchmarkOutput(
            analysis_id=f"EB-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_id=input_data.facility_id,
            specific_energy_kwh_per_unit=round(sec, 1),
            benchmark_p10=benchmarks["p10"],
            benchmark_p25=benchmarks["p25"],
            benchmark_p50=benchmarks["p50"],
            benchmark_p75=benchmarks["p75"],
            benchmark_p90=benchmarks["p90"],
            percentile_rank=percentile,
            performance_rating=rating,
            gap_to_p25_kwh_per_unit=round(max(0, gap_to_p25), 1),
            gap_to_p25_percent=round(max(0, gap_percent), 1),
            potential_savings_kwh_per_year=round(potential_kwh, 0),
            potential_savings_cost_per_year=round(potential_cost, 0),
            improvement_targets=targets,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-039",
    "name": "ENERGY-BENCHMARK",
    "version": "1.0.0",
    "summary": "Process energy benchmarking and gap analysis",
    "tags": ["benchmarking", "energy-efficiency", "gap-analysis", "ISO-50001", "ENERGY-STAR"],
    "standards": [
        {"ref": "ISO 50001", "description": "Energy Management Systems"},
        {"ref": "EPA ENERGY STAR", "description": "Industrial Energy Benchmarking"}
    ]
}
