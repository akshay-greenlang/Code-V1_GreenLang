"""
GL-032: Refractory Monitor Agent (REFRACTORY-MONITOR)

This package provides the RefractoryMonitorAgent for refractory health assessment
in industrial furnaces, heaters, and kilns.

Key Features:
- Skin temperature analysis with hotspot detection
- Heat loss calculations through refractory layers
- Thermal gradient analysis for spalling risk
- Remaining useful life prediction
- Maintenance priority determination

Standards Compliance:
- API 560: Fired Heaters for General Refinery Service
- ASTM C155: Standard Classification of Insulating Firebrick
"""

from .agent import (
    RefractoryMonitorAgent,
    RefractoryMonitorInput,
    RefractoryMonitorOutput,
    SkinTemperature,
    ThermalImageData,
    RefractoryLayer,
    HistoricalReading,
    HotspotAnalysis,
    HeatLossAnalysis,
    PACK_SPEC,
)

from .models import (
    RefractoryMaterial,
    RefractoryZone,
    MaintenancePriority,
    HealthStatus,
    DegradationMode,
)

__all__ = [
    "RefractoryMonitorAgent",
    "RefractoryMonitorInput",
    "RefractoryMonitorOutput",
    "SkinTemperature",
    "ThermalImageData",
    "RefractoryLayer",
    "HistoricalReading",
    "HotspotAnalysis",
    "HeatLossAnalysis",
    "RefractoryMaterial",
    "RefractoryZone",
    "MaintenancePriority",
    "HealthStatus",
    "DegradationMode",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-032"
__agent_name__ = "REFRACTORY-MONITOR"
