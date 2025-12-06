"""
GL-010 EmissionsGuardian Agent - Real-time Emissions Monitor

Provides real-time emissions monitoring with predictive exceedance alerts.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
)

logger = logging.getLogger(__name__)


class EmissionsInput(BaseModel):
    """Input for emissions monitoring."""

    source_id: str = Field(..., description="Emission source identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Fuel data
    fuel_type: str = Field(default="natural_gas")
    fuel_flow_rate: float = Field(..., gt=0, description="Fuel flow rate")
    fuel_flow_unit: str = Field(default="MMBTU/hr")

    # Stack measurements
    stack_o2_pct: float = Field(..., ge=0, le=21)
    stack_co_ppm: float = Field(default=0.0, ge=0)
    stack_nox_ppm: Optional[float] = Field(default=None, ge=0)
    stack_so2_ppm: Optional[float] = Field(default=None, ge=0)
    stack_pm_mg_m3: Optional[float] = Field(default=None, ge=0)
    stack_temperature_f: float = Field(...)
    stack_flow_rate_acfm: Optional[float] = Field(default=None)

    # Operating conditions
    load_pct: float = Field(default=100.0, ge=0, le=120)
    operating_mode: str = Field(default="normal")


class EmissionsOutput(BaseModel):
    """Output from emissions monitoring."""

    source_id: str
    timestamp: datetime
    status: str = Field(default="compliant")

    # Calculated emissions
    co2_lb_hr: float = Field(...)
    co2_kg_hr: float = Field(...)
    co2_ton_yr: float = Field(...)
    nox_lb_hr: Optional[float] = Field(default=None)
    so2_lb_hr: Optional[float] = Field(default=None)
    pm_lb_hr: Optional[float] = Field(default=None)

    # Emission rates
    co2_lb_mmbtu: float = Field(...)
    nox_lb_mmbtu: Optional[float] = Field(default=None)

    # Compliance status
    permit_limits: Dict[str, float] = Field(default_factory=dict)
    exceedances: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Predictions
    predicted_exceedance_risk: float = Field(default=0.0, ge=0, le=1)
    predicted_exceedance_time_hr: Optional[float] = Field(default=None)


class EmissionsMonitor:
    """
    Real-time emissions monitoring system.

    Calculates emissions using EPA Method 19 and monitors
    compliance with permit limits.
    """

    def __init__(
        self,
        source_id: str,
        permit_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize emissions monitor.

        Args:
            source_id: Emission source identifier
            permit_limits: Dictionary of permit limits (pollutant: limit_lb_hr)
        """
        self.source_id = source_id
        self.permit_limits = permit_limits or {}

        self.calc_library = ThermalIQCalculationLibrary()

        # History for trend analysis
        self._emissions_history: List[Dict[str, Any]] = []
        self._exceedance_history: List[Dict[str, Any]] = []

        logger.info(f"EmissionsMonitor initialized for {source_id}")

    def monitor(self, input_data: EmissionsInput) -> EmissionsOutput:
        """
        Monitor emissions and check compliance.

        Args:
            input_data: Current emissions data

        Returns:
            EmissionsOutput with calculations and compliance status
        """
        warnings = []
        exceedances = []

        # Calculate CO2 emissions
        co2_result = self.calc_library.calculate_co2_emissions(
            fuel_type=input_data.fuel_type,
            fuel_consumption=input_data.fuel_flow_rate,
            fuel_unit="MMBTU",
        )
        co2_kg_hr = co2_result.value
        co2_lb_hr = co2_kg_hr * 2.205

        # Annualize
        co2_ton_yr = co2_lb_hr * 8760 / 2000

        # CO2 emission rate (lb/MMBTU)
        co2_lb_mmbtu = co2_lb_hr / input_data.fuel_flow_rate if input_data.fuel_flow_rate > 0 else 0

        # Calculate NOx if measured
        nox_lb_hr = None
        nox_lb_mmbtu = None
        if input_data.stack_nox_ppm is not None and input_data.stack_flow_rate_acfm:
            nox_result = self.calc_library.calculate_nox_emissions(
                fuel_type=input_data.fuel_type,
                fuel_consumption_mmbtu=input_data.fuel_flow_rate,
                combustion_type="low_nox_burner",
            )
            nox_lb_hr = nox_result.value
            nox_lb_mmbtu = nox_lb_hr / input_data.fuel_flow_rate if input_data.fuel_flow_rate > 0 else 0

        # SO2 calculation (simplified)
        so2_lb_hr = None
        if input_data.stack_so2_ppm is not None:
            # Simplified calculation
            so2_lb_hr = input_data.stack_so2_ppm * 0.001 * input_data.fuel_flow_rate

        # PM calculation
        pm_lb_hr = None
        if input_data.stack_pm_mg_m3 is not None and input_data.stack_flow_rate_acfm:
            # Convert mg/m3 to lb/hr
            flow_m3_hr = input_data.stack_flow_rate_acfm * 60 * 0.0283168
            pm_lb_hr = input_data.stack_pm_mg_m3 * flow_m3_hr / 453592

        # Check compliance
        status = "compliant"

        if "co2_lb_hr" in self.permit_limits:
            if co2_lb_hr > self.permit_limits["co2_lb_hr"]:
                exceedances.append({
                    "pollutant": "CO2",
                    "measured": co2_lb_hr,
                    "limit": self.permit_limits["co2_lb_hr"],
                    "exceedance_pct": (co2_lb_hr / self.permit_limits["co2_lb_hr"] - 1) * 100,
                })
                status = "exceedance"

        if nox_lb_hr and "nox_lb_hr" in self.permit_limits:
            if nox_lb_hr > self.permit_limits["nox_lb_hr"]:
                exceedances.append({
                    "pollutant": "NOx",
                    "measured": nox_lb_hr,
                    "limit": self.permit_limits["nox_lb_hr"],
                    "exceedance_pct": (nox_lb_hr / self.permit_limits["nox_lb_hr"] - 1) * 100,
                })
                status = "exceedance"

        # High CO warning
        if input_data.stack_co_ppm > 200:
            warnings.append(f"High CO detected: {input_data.stack_co_ppm:.0f} ppm")

        # Predict exceedance risk
        exceedance_risk = 0.0
        exceedance_time = None

        # Add to history
        self._emissions_history.append({
            "timestamp": input_data.timestamp,
            "co2_lb_hr": co2_lb_hr,
            "nox_lb_hr": nox_lb_hr,
            "load_pct": input_data.load_pct,
        })

        # Keep last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._emissions_history = [
            h for h in self._emissions_history
            if h["timestamp"] > cutoff
        ]

        # Simple trend-based prediction
        if len(self._emissions_history) >= 10:
            recent = [h["co2_lb_hr"] for h in self._emissions_history[-10:]]
            trend = (recent[-1] - recent[0]) / 10  # Per reading

            if "co2_lb_hr" in self.permit_limits:
                limit = self.permit_limits["co2_lb_hr"]
                if trend > 0 and co2_lb_hr < limit:
                    readings_to_limit = (limit - co2_lb_hr) / trend
                    if readings_to_limit < 60:  # Within next 60 readings
                        exceedance_risk = min(1.0, 60 / max(readings_to_limit, 1))
                        exceedance_time = readings_to_limit * 0.5  # Assume 30 min/reading

        if exceedance_risk > 0.5:
            warnings.append(
                f"Exceedance predicted in ~{exceedance_time:.1f} hours based on trend"
            )

        if exceedances:
            self._exceedance_history.extend(exceedances)

        return EmissionsOutput(
            source_id=input_data.source_id,
            timestamp=input_data.timestamp,
            status=status,
            co2_lb_hr=round(co2_lb_hr, 2),
            co2_kg_hr=round(co2_kg_hr, 2),
            co2_ton_yr=round(co2_ton_yr, 1),
            nox_lb_hr=round(nox_lb_hr, 4) if nox_lb_hr else None,
            so2_lb_hr=round(so2_lb_hr, 4) if so2_lb_hr else None,
            pm_lb_hr=round(pm_lb_hr, 6) if pm_lb_hr else None,
            co2_lb_mmbtu=round(co2_lb_mmbtu, 2),
            nox_lb_mmbtu=round(nox_lb_mmbtu, 4) if nox_lb_mmbtu else None,
            permit_limits=self.permit_limits,
            exceedances=exceedances,
            warnings=warnings,
            predicted_exceedance_risk=round(exceedance_risk, 2),
            predicted_exceedance_time_hr=round(exceedance_time, 1) if exceedance_time else None,
        )

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get daily emissions summary."""
        if not self._emissions_history:
            return {"message": "No data available"}

        co2_values = [h["co2_lb_hr"] for h in self._emissions_history]

        return {
            "period_hours": len(self._emissions_history) * 0.5,
            "co2_avg_lb_hr": sum(co2_values) / len(co2_values),
            "co2_max_lb_hr": max(co2_values),
            "co2_min_lb_hr": min(co2_values),
            "exceedance_count": len(self._exceedance_history),
        }
