"""GL-069: Demand Forecast Agent (DEMAND-FORECAST).

Forecasts energy and thermal demand for optimization.

Standards: ISO 50006, IPMVP
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ForecastHorizon(str, Enum):
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class DemandType(str, Enum):
    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    STEAM = "STEAM"
    CHILLED_WATER = "CHILLED_WATER"


class HistoricalData(BaseModel):
    timestamp: datetime
    demand_value: float
    temperature_c: Optional[float] = None
    production_units: Optional[float] = None


class DemandForecastInput(BaseModel):
    facility_id: str
    demand_type: DemandType = Field(default=DemandType.ELECTRICITY)
    forecast_horizon: ForecastHorizon = Field(default=ForecastHorizon.DAILY)
    historical_data: List[HistoricalData] = Field(default_factory=list)
    base_load_kw: float = Field(default=100, ge=0)
    production_schedule: float = Field(default=100, ge=0, le=150)  # % of normal
    forecast_temp_c: float = Field(default=20)
    heating_degree_base_c: float = Field(default=18)
    cooling_degree_base_c: float = Field(default=22)
    temp_sensitivity_kw_c: float = Field(default=5, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastPeriod(BaseModel):
    period_index: int
    forecast_demand: float
    confidence_lower: float
    confidence_upper: float
    weather_impact: float
    production_impact: float


class DemandForecastOutput(BaseModel):
    facility_id: str
    demand_type: str
    forecast_horizon: str
    forecast_periods: List[ForecastPeriod]
    peak_demand_forecast: float
    average_demand_forecast: float
    total_demand_forecast: float
    forecast_confidence_pct: float
    weather_sensitivity_index: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class DemandForecastAgent:
    AGENT_ID = "GL-069"
    AGENT_NAME = "DEMAND-FORECAST"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"DemandForecastAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = DemandForecastInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_degree_days(self, temp: float, hdd_base: float, cdd_base: float) -> tuple:
        """Calculate heating and cooling degree days."""
        hdd = max(0, hdd_base - temp)
        cdd = max(0, temp - cdd_base)
        return hdd, cdd

    def _process(self, inp: DemandForecastInput) -> DemandForecastOutput:
        recommendations = []

        # Calculate baseline from historical data if available
        if inp.historical_data:
            avg_historical = sum(d.demand_value for d in inp.historical_data) / len(inp.historical_data)
            baseline = avg_historical
        else:
            baseline = inp.base_load_kw

        # Degree days impact
        hdd, cdd = self._calculate_degree_days(inp.forecast_temp_c, inp.heating_degree_base_c, inp.cooling_degree_base_c)

        # Weather impact on demand
        if inp.demand_type == DemandType.NATURAL_GAS:
            weather_impact = hdd * inp.temp_sensitivity_kw_c
        elif inp.demand_type == DemandType.CHILLED_WATER:
            weather_impact = cdd * inp.temp_sensitivity_kw_c
        else:
            weather_impact = (hdd + cdd) * inp.temp_sensitivity_kw_c * 0.5

        # Production impact
        production_factor = inp.production_schedule / 100
        production_impact = baseline * (production_factor - 1)

        # Generate forecast periods
        num_periods = {
            ForecastHorizon.HOURLY: 24,
            ForecastHorizon.DAILY: 7,
            ForecastHorizon.WEEKLY: 4,
            ForecastHorizon.MONTHLY: 12
        }.get(inp.forecast_horizon, 7)

        forecasts = []
        demands = []

        for i in range(num_periods):
            # Add time-of-day/week variation
            if inp.forecast_horizon == ForecastHorizon.HOURLY:
                # Peak at 2pm (index 14)
                variation = 0.8 + 0.4 * math.sin((i - 6) * math.pi / 12)
            else:
                variation = 0.95 + 0.1 * math.sin(i * math.pi / num_periods)

            forecast = (baseline + weather_impact + production_impact) * variation
            forecast = max(0, forecast)
            demands.append(forecast)

            # Confidence interval (±10% base)
            confidence = 0.10 + 0.02 * (i / num_periods)  # Increases with time
            lower = forecast * (1 - confidence)
            upper = forecast * (1 + confidence)

            forecasts.append(ForecastPeriod(
                period_index=i,
                forecast_demand=round(forecast, 1),
                confidence_lower=round(lower, 1),
                confidence_upper=round(upper, 1),
                weather_impact=round(weather_impact * variation, 1),
                production_impact=round(production_impact * variation, 1)
            ))

        # Summary statistics
        peak = max(demands)
        average = sum(demands) / len(demands)
        total = sum(demands)

        # Confidence based on data availability
        if len(inp.historical_data) > 30:
            confidence = 90
        elif len(inp.historical_data) > 7:
            confidence = 80
        else:
            confidence = 70

        # Weather sensitivity
        weather_sensitivity = abs(weather_impact / baseline * 100) if baseline > 0 else 0

        # Recommendations
        if weather_sensitivity > 20:
            recommendations.append(f"High weather sensitivity ({weather_sensitivity:.1f}%) - implement demand response")
        if peak > average * 1.5:
            recommendations.append(f"Peak demand {peak:.0f} kW significantly exceeds average - load shifting opportunity")
        if confidence < 80:
            recommendations.append("Low forecast confidence - collect more historical data")
        if production_factor > 1.2:
            recommendations.append("High production schedule increases demand risk - verify capacity")
        if hdd > 10:
            recommendations.append(f"High heating load expected ({hdd:.1f} HDD) - optimize heating system")
        if cdd > 10:
            recommendations.append(f"High cooling load expected ({cdd:.1f} CDD) - pre-cool if possible")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "peak": round(peak, 1),
            "average": round(average, 1)
        }).encode()).hexdigest()

        return DemandForecastOutput(
            facility_id=inp.facility_id,
            demand_type=inp.demand_type.value,
            forecast_horizon=inp.forecast_horizon.value,
            forecast_periods=forecasts,
            peak_demand_forecast=round(peak, 1),
            average_demand_forecast=round(average, 1),
            total_demand_forecast=round(total, 1),
            forecast_confidence_pct=confidence,
            weather_sensitivity_index=round(weather_sensitivity, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-069", "name": "DEMAND-FORECAST", "version": "1.0.0",
    "summary": "Energy and thermal demand forecasting",
    "standards": [{"ref": "ISO 50006"}, {"ref": "IPMVP"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
