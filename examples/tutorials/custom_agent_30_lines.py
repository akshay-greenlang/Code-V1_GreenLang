"""Tutorial: Write a new agent in 30 lines.

This example shows how to create a custom GreenLang agent
that calculates water usage emissions.
"""

from greenlang.core.agent import Agent

class WaterUsageAgent(Agent):
    """Agent that calculates emissions from water usage."""
    
    def __init__(self):
        super().__init__(
            agent_id="water_usage",
            name="Water Usage Emissions Calculator",
            version="1.0.0"
        )
        # Water treatment emission factor: ~0.0003 kg CO2/liter
        self.emission_factor = 0.0003
    
    def run(self, payload):
        """Calculate emissions from water consumption."""
        try:
            # Extract water usage in liters
            water_liters = payload.get("water_consumption", {}).get("value", 0)
            
            # Calculate emissions
            emissions_kg = water_liters * self.emission_factor
            
            return {
                "success": True,
                "data": {
                    "water_consumption_liters": water_liters,
                    "co2e_emissions_kg": emissions_kg,
                    "emission_factor": self.emission_factor,
                    "unit": "kg CO2e"
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}