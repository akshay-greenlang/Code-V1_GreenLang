# greenlang/agents/load_profile_agent.py

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for the LoadProfileAgent. "
        "Install it with: pip install greenlang[analytics]"
    )

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult

# Constants
SPECIFIC_HEAT_WATER_KJ_KG_C = 4.186  # Specific heat capacity of water in kJ/kg°C
SECONDS_PER_HOUR = 3600


class LoadProfileAgent(BaseAgent):
    """
    Generates an hourly thermal load profile from process demand data.
    """

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Args:
            input_data: Dictionary containing 'process_demand'.

        Returns:
            AgentResult with hourly load profile as a JSON string.
        """
        try:
            process_demand = input_data.get("process_demand")
            if not process_demand:
                return AgentResult(success=False, error="process_demand not provided")

            flow_profile_path = process_demand["flow_profile"]
            temp_in = process_demand["temp_in_C"]
            temp_out = process_demand["temp_out_C"]

            # Load the flow profile data
            df = pd.read_csv(flow_profile_path, index_col="timestamp", parse_dates=True)

            # Calculate the temperature difference (delta T)
            delta_t = temp_out - temp_in

            # Calculate the energy demand in kWh for each timestamp
            # Formula: Q (kJ) = m (kg/s) * c (kJ/kg°C) * ΔT (°C) * 3600 (s/h)
            # Then convert kJ to kWh by dividing by 3600
            # The 3600 terms cancel out, so Q (kWh) = m * c * ΔT
            df["demand_kWh"] = (
                df["flow_kg_s"]
                * SPECIFIC_HEAT_WATER_KJ_KG_C
                * delta_t
                / SECONDS_PER_HOUR
                * SECONDS_PER_HOUR
            )

            # For v1, we assume the schedule is reflected in the CSV.
            # A future version would apply the 'schedule' dictionary to modify the profile.

            total_annual_demand_gwh = df["demand_kWh"].sum() / 1e6

            return AgentResult(
                success=True,
                data={
                    "load_profile_df_json": df[["demand_kWh"]].to_json(orient="split"),
                    "total_annual_demand_gwh": total_annual_demand_gwh,
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
