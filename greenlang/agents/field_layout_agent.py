# greenlang/agents/field_layout_agent.py

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult

# Constants - these are typical values for a good solar location
# In a real model, this would be calculated from TMY data.
ANNUAL_DNI_KWH_PER_M2 = 2000  # Annual Direct Normal Irradiance in kWh/m^2/year
SYSTEM_EFFICIENCY_HEURISTIC = 0.50  # Overall system efficiency (optical, thermal, etc.)


class FieldLayoutAgent(BaseAgent):
    """
    Performs high-level sizing of the solar collector field.
    """

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Args:
            input_data: Dictionary containing 'total_annual_demand_gwh' and 'solar_config'.

        Returns:
            AgentResult with required aperture area and collector count.
        """
        try:
            total_annual_demand_gwh = input_data.get("total_annual_demand_gwh")
            solar_config = input_data.get("solar_config")

            if total_annual_demand_gwh is None or not solar_config:
                return AgentResult(
                    success=False,
                    error="total_annual_demand_gwh and solar_config must be provided",
                )

            target_solar_fraction = 0.50  # For v1, we hardcode a 50% target

            # Calculate the total solar energy we need to generate in a year
            required_solar_energy_kwh = (
                total_annual_demand_gwh * 1e6 * target_solar_fraction
            )

            # Calculate the energy generated per square meter of collector
            annual_yield_per_m2 = ANNUAL_DNI_KWH_PER_M2 * SYSTEM_EFFICIENCY_HEURISTIC

            # Calculate the required aperture area
            required_aperture_area_m2 = required_solar_energy_kwh / annual_yield_per_m2

            # Get collector-specific data (this would come from a config file)
            collector_aperture_area = 50  # m^2 per collector for a typical model

            num_collectors = round(required_aperture_area_m2 / collector_aperture_area)

            # Recalculate the actual area based on the integer number of collectors
            actual_aperture_area_m2 = num_collectors * collector_aperture_area

            # Estimate land area
            land_area_per_aperture_area = solar_config["row_spacing_factor"]
            required_land_area_m2 = (
                actual_aperture_area_m2 * land_area_per_aperture_area
            )

            return AgentResult(
                success=True,
                data={
                    "required_aperture_area_m2": actual_aperture_area_m2,
                    "num_collectors": num_collectors,
                    "required_land_area_m2": required_land_area_m2,
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
