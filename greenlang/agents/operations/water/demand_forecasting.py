# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-005: Demand Forecasting Agent
========================================

Operations agent for water demand forecasting.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ForecastHorizon(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DemandForecast(BaseModel):
    """Demand forecast entry."""
    timestamp: datetime
    forecasted_demand_m3: float
    lower_bound_m3: float
    upper_bound_m3: float
    confidence_percent: float


class HistoricalDemand(BaseModel):
    """Historical demand record."""
    timestamp: datetime
    actual_demand_m3: float
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    is_holiday: bool = False


class ForecastAccuracy(BaseModel):
    """Forecast accuracy metrics."""
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    r_squared: float


class DemandForecastInput(BaseModel):
    """Input for demand forecasting."""
    zone_id: str
    historical_data: List[HistoricalDemand]
    forecast_horizon: ForecastHorizon = ForecastHorizon.DAILY
    forecast_periods: int = Field(default=7, ge=1)
    weather_forecast: Optional[List[Dict[str, float]]] = None
    include_uncertainty: bool = True


class DemandForecastOutput(BaseModel):
    """Output from demand forecasting."""
    zone_id: str
    forecast_horizon: str
    forecasts: List[DemandForecast]
    total_forecasted_demand_m3: float
    peak_demand_m3: float
    peak_demand_time: datetime
    accuracy_metrics: Optional[ForecastAccuracy] = None
    seasonal_factors: Dict[str, float]
    recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class DemandForecastingAgent(BaseAgent):
    """
    GL-OPS-WAT-005: Demand Forecasting Agent

    Forecasts water demand using statistical methods.
    """

    AGENT_ID = "GL-OPS-WAT-005"
    AGENT_NAME = "Demand Forecasting Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water demand forecasting",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            df_input = DemandForecastInput(**input_data)

            # Calculate historical statistics
            demands = [h.actual_demand_m3 for h in df_input.historical_data]
            if not demands:
                return AgentResult(success=False, error="No historical data provided")

            mean_demand = sum(demands) / len(demands)
            std_demand = (sum((d - mean_demand) ** 2 for d in demands) / len(demands)) ** 0.5

            # Calculate day-of-week patterns (simplified)
            dow_factors = {i: 1.0 for i in range(7)}
            for h in df_input.historical_data:
                dow = h.timestamp.weekday()
                if mean_demand > 0:
                    dow_factors[dow] = h.actual_demand_m3 / mean_demand

            # Generate forecasts
            forecasts = []
            last_timestamp = df_input.historical_data[-1].timestamp if df_input.historical_data else DeterministicClock.now()

            for i in range(df_input.forecast_periods):
                if df_input.forecast_horizon == ForecastHorizon.HOURLY:
                    forecast_time = last_timestamp + timedelta(hours=i + 1)
                elif df_input.forecast_horizon == ForecastHorizon.DAILY:
                    forecast_time = last_timestamp + timedelta(days=i + 1)
                elif df_input.forecast_horizon == ForecastHorizon.WEEKLY:
                    forecast_time = last_timestamp + timedelta(weeks=i + 1)
                else:
                    forecast_time = last_timestamp + timedelta(days=30 * (i + 1))

                # Apply day-of-week factor
                dow = forecast_time.weekday()
                forecast_value = mean_demand * dow_factors.get(dow, 1.0)

                # Add weather adjustment if available
                if df_input.weather_forecast and i < len(df_input.weather_forecast):
                    temp = df_input.weather_forecast[i].get("temperature_c", 20)
                    # Higher temp = higher demand
                    temp_factor = 1 + (temp - 20) * 0.02
                    forecast_value *= temp_factor

                # Confidence bounds
                confidence = 90 - i * 2  # Decreasing confidence over time
                margin = std_demand * (1 + i * 0.1)

                forecast = DemandForecast(
                    timestamp=forecast_time,
                    forecasted_demand_m3=round(forecast_value, 2),
                    lower_bound_m3=round(max(0, forecast_value - margin), 2),
                    upper_bound_m3=round(forecast_value + margin, 2),
                    confidence_percent=max(50, confidence),
                )
                forecasts.append(forecast)

            # Calculate totals
            total_forecast = sum(f.forecasted_demand_m3 for f in forecasts)
            peak_forecast = max(forecasts, key=lambda f: f.forecasted_demand_m3)

            # Seasonal factors
            seasonal_factors = {
                "monday": round(dow_factors.get(0, 1.0), 3),
                "tuesday": round(dow_factors.get(1, 1.0), 3),
                "wednesday": round(dow_factors.get(2, 1.0), 3),
                "thursday": round(dow_factors.get(3, 1.0), 3),
                "friday": round(dow_factors.get(4, 1.0), 3),
                "saturday": round(dow_factors.get(5, 1.0), 3),
                "sunday": round(dow_factors.get(6, 1.0), 3),
            }

            # Recommendations
            recommendations = []
            if peak_forecast.forecasted_demand_m3 > mean_demand * 1.2:
                recommendations.append(f"Prepare for elevated demand on {peak_forecast.timestamp.date()}")
            recommendations.append("Review pump schedules to align with forecasted demand patterns")

            provenance_hash = hashlib.sha256(
                json.dumps({"zone": df_input.zone_id, "periods": df_input.forecast_periods}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = DemandForecastOutput(
                zone_id=df_input.zone_id,
                forecast_horizon=df_input.forecast_horizon.value,
                forecasts=forecasts,
                total_forecasted_demand_m3=round(total_forecast, 2),
                peak_demand_m3=peak_forecast.forecasted_demand_m3,
                peak_demand_time=peak_forecast.timestamp,
                seasonal_factors=seasonal_factors,
                recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Demand forecasting failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
