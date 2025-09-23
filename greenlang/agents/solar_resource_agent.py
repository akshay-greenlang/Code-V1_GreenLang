# greenlang/agents/solar_resource_agent.py

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for the SolarResourceAgent. "
        "Install it with: pip install greenlang[analytics]"
    )

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult


class SolarResourceAgent(BaseAgent):
    """
    Fetches or loads TMY solar data for the site.
    """

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Args:
            input_data: Dictionary containing 'lat' and 'lon'.

        Returns:
            AgentResult with DNI and temperature time-series.
        """
        try:
            lat = input_data.get("lat")
            lon = input_data.get("lon")

            if lat is None or lon is None:
                return AgentResult(success=False, error="lat and lon must be provided")

            # For v1, we'll use a placeholder for loading a local TMY file.
            # In a real implementation, you would have a library to find the
            # closest TMY file based on lat/lon.

            print(f"Fetching solar resource for Lat: {lat}, Lon: {lon}")

            # Create a dummy time-series for a full year (8760 hours)
            # This simulates reading a TMY file.
            timestamps = pd.to_datetime(
                pd.date_range(start="2023-01-01", end="2023-12-31 23:00", freq="h")
            )

            # Dummy DNI data (in W/m^2) - sinusoidal pattern for daytime
            dni = [max(0, 600 * (1 - abs(h - 12) / 6)) for h in range(24)] * 365

            # Dummy temperature data (in C)
            temp = [15 + 10 * (1 - abs(h - 14) / 10) for h in range(24)] * 365

            df = pd.DataFrame(
                {"dni_w_per_m2": dni[:8760], "temp_c": temp[:8760]}, index=timestamps
            )

            return AgentResult(
                success=True, data={"solar_resource_df": df.to_json(orient="split")}
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
