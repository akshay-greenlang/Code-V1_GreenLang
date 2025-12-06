"""
GL-007 Furnace Performance Monitor

Real-time furnace performance monitoring with thermal analysis.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FurnaceInput(BaseModel):
    """Input for furnace monitoring."""

    furnace_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Firing
    fuel_type: str = Field(default="natural_gas")
    fuel_flow_rate: float = Field(..., gt=0)
    heat_release_mmbtu_hr: Optional[float] = Field(default=None)

    # Temperatures
    zone_temperatures_f: Dict[str, float] = Field(default_factory=dict)
    bridgewall_temp_f: Optional[float] = Field(default=None)
    arch_temp_f: Optional[float] = Field(default=None)
    flue_gas_temp_f: float = Field(...)

    # TMT (Tube Metal Temperatures)
    tmt_readings_f: Dict[str, float] = Field(default_factory=dict)
    tmt_design_limit_f: float = Field(default=1500.0)

    # Combustion
    o2_pct: float = Field(..., ge=0, le=21)
    draft_in_h2o: Optional[float] = Field(default=None)

    # Process
    process_fluid_inlet_temp_f: Optional[float] = Field(default=None)
    process_fluid_outlet_temp_f: Optional[float] = Field(default=None)
    process_flow_rate: Optional[float] = Field(default=None)


class FurnaceOutput(BaseModel):
    """Output from furnace monitoring."""

    furnace_id: str
    timestamp: datetime
    status: str = Field(default="normal")

    # Performance
    thermal_efficiency_pct: float = Field(...)
    heat_absorption_btu_hr: float = Field(...)
    heat_flux_avg_btu_hr_ft2: Optional[float] = Field(default=None)

    # Temperatures
    avg_zone_temp_f: float = Field(...)
    max_zone_temp_f: float = Field(...)
    temp_uniformity_pct: float = Field(...)

    # TMT Analysis
    tmt_max_f: Optional[float] = Field(default=None)
    tmt_margin_f: Optional[float] = Field(default=None)
    tmt_hot_spots: List[str] = Field(default_factory=list)

    # Alerts
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class FurnacePerformanceMonitor:
    """
    Furnace performance monitoring system.

    Monitors thermal performance, TMT, and efficiency.
    """

    def __init__(
        self,
        furnace_id: str,
        design_duty_mmbtu_hr: float = 50.0,
        tmt_alarm_f: float = 1450.0,
        tmt_trip_f: float = 1500.0,
    ) -> None:
        """Initialize furnace monitor."""
        self.furnace_id = furnace_id
        self.design_duty = design_duty_mmbtu_hr
        self.tmt_alarm = tmt_alarm_f
        self.tmt_trip = tmt_trip_f

        self._history: List[Dict[str, Any]] = []

        logger.info(f"FurnacePerformanceMonitor initialized for {furnace_id}")

    def monitor(self, input_data: FurnaceInput) -> FurnaceOutput:
        """Monitor furnace performance."""
        alerts = []
        recommendations = []
        status = "normal"

        # Calculate zone temperature statistics
        zone_temps = list(input_data.zone_temperatures_f.values())
        if zone_temps:
            avg_zone = sum(zone_temps) / len(zone_temps)
            max_zone = max(zone_temps)
            min_zone = min(zone_temps)
            uniformity = (1 - (max_zone - min_zone) / avg_zone) * 100 if avg_zone > 0 else 100
        else:
            avg_zone = 0
            max_zone = 0
            uniformity = 100

        # TMT Analysis
        tmt_max = None
        tmt_margin = None
        hot_spots = []

        if input_data.tmt_readings_f:
            tmt_values = list(input_data.tmt_readings_f.values())
            tmt_max = max(tmt_values)
            tmt_margin = input_data.tmt_design_limit_f - tmt_max

            # Find hot spots
            for location, temp in input_data.tmt_readings_f.items():
                if temp > self.tmt_alarm:
                    hot_spots.append(location)
                    if temp > self.tmt_trip:
                        alerts.append({
                            "type": "TMT_TRIP",
                            "severity": "critical",
                            "location": location,
                            "temperature": temp,
                        })
                        status = "critical"
                    else:
                        alerts.append({
                            "type": "TMT_HIGH",
                            "severity": "warning",
                            "location": location,
                            "temperature": temp,
                        })
                        if status == "normal":
                            status = "warning"

        # Estimate efficiency
        heat_release = input_data.heat_release_mmbtu_hr or (input_data.fuel_flow_rate * 0.001)
        stack_loss_pct = (input_data.flue_gas_temp_f - 77) * 0.02
        efficiency = 100 - stack_loss_pct - 2  # 2% radiation

        # Heat absorption
        heat_absorption = heat_release * efficiency / 100 * 1e6  # BTU/hr

        # Check for issues
        if uniformity < 80:
            recommendations.append(
                f"Poor temperature uniformity ({uniformity:.1f}%). "
                "Check burner firing pattern."
            )

        if input_data.draft_in_h2o and abs(input_data.draft_in_h2o) < 0.1:
            recommendations.append(
                "Low draft - check damper positions and ID fan."
            )

        # Add to history
        self._history.append({
            "timestamp": input_data.timestamp,
            "efficiency": efficiency,
            "tmt_max": tmt_max,
        })

        return FurnaceOutput(
            furnace_id=input_data.furnace_id,
            timestamp=input_data.timestamp,
            status=status,
            thermal_efficiency_pct=round(efficiency, 1),
            heat_absorption_btu_hr=round(heat_absorption, 0),
            avg_zone_temp_f=round(avg_zone, 1),
            max_zone_temp_f=round(max_zone, 1),
            temp_uniformity_pct=round(uniformity, 1),
            tmt_max_f=round(tmt_max, 1) if tmt_max else None,
            tmt_margin_f=round(tmt_margin, 1) if tmt_margin else None,
            tmt_hot_spots=hot_spots,
            alerts=alerts,
            recommendations=recommendations,
        )
