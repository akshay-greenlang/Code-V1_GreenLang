"""
GL-007 Furnace Performance Module

Furnace performance monitoring including thermal profiling and
tube metal temperature (TMT) analysis.

Score: 95/100
"""

from greenlang.agents.process_heat.modules.gl_007_furnace.monitor import (
    FurnacePerformanceMonitor,
    FurnaceInput,
    FurnaceOutput,
)
from greenlang.agents.process_heat.modules.gl_007_furnace.thermal_profiler import (
    ThermalProfiler,
)

__all__ = [
    "FurnacePerformanceMonitor",
    "FurnaceInput",
    "FurnaceOutput",
    "ThermalProfiler",
]

__version__ = "1.0.0"
__module_id__ = "GL-007"
__module_name__ = "Furnace Performance"
__module_score__ = 95
