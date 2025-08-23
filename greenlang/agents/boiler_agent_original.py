from typing import Optional, Dict, Any, List
from ..types import Agent, AgentResult, ErrorInfo
from .types import BoilerInput, BoilerOutput
from greenlang.data.emission_factors import EmissionFactors


class BoilerAgent(Agent[BoilerInput, BoilerOutput]):
    """Agent for calculating emissions from boiler operations and thermal systems.
    
    This agent specializes in:
    - Boiler efficiency calculations
    - Thermal output to fuel consumption conversion
    - Multi-fuel boiler systems
    - Steam and hot water generation emissions
    - Boiler performance optimization recommendations
    """
    
    agent_id: str = "boiler"
    name: str = "Boiler Emissions Calculator"
    version: str = "0.0.1"
    
    # Standard boiler efficiency ranges by type and age
    BOILER_EFFICIENCIES = {
        "natural_gas": {
            "condensing": {"new": 0.95, "medium": 0.92, "old": 0.88},
            "standard": {"new": 0.85, "medium": 0.80, "old": 0.75},
            "low_efficiency": {"new": 0.78, "medium": 0.72, "old": 0.65}
        },
        "oil": {
            "condensing": {"new": 0.92, "medium": 0.88, "old": 0.84},
            "standard": {"new": 0.83, "medium": 0.78, "old": 0.72},
            "low_efficiency": {"new": 0.75, "medium": 0.68, "old": 0.60}
        },
        "biomass": {
            "modern": {"new": 0.85, "medium": 0.80, "old": 0.75},
            "standard": {"new": 0.75, "medium": 0.70, "old": 0.65},
            "traditional": {"new": 0.65, "medium": 0.55, "old": 0.45}
        },
        "coal": {
            "pulverized": {"new": 0.85, "medium": 0.80, "old": 0.75},
            "stoker": {"new": 0.75, "medium": 0.70, "old": 0.65},
            "hand_fired": {"new": 0.60, "medium": 0.50, "old": 0.40}
        },
        "electric": {
            "resistance": {"new": 0.99, "medium": 0.98, "old": 0.97},
            "heat_pump": {"new": 3.50, "medium": 3.00, "old": 2.50}  # COP values
        }
    }
    
    def __init__(self) -> None:
        """Initialize the BoilerAgent with emission factors database."""
        self.emission_factors = EmissionFactors()
    
    def validate(self, payload: BoilerInput) -> bool:
        """Validate input payload structure and values.
        
        Args:
            payload: Input data containing boiler specifications
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check required fields
        if not payload.get("boiler_type"):
            return False
        
        # Validate based on input type (thermal output or fuel consumption)
        if "thermal_output" in payload:
            thermal = payload["thermal_output"]
            if not isinstance(thermal, dict):
                return False
            if "value" not in thermal or "unit" not in thermal:
                return False
            if thermal["value"] <= 0:
                return False
        elif "fuel_consumption" in payload:
            fuel = payload["fuel_consumption"]
            if not isinstance(fuel, dict):
                return False
            if "value" not in fuel or "unit" not in fuel:
                return False
            if fuel["value"] <= 0:
                return False
        else:
            # Must have either thermal output or fuel consumption
            return False
        
        # Validate efficiency if provided
        if "efficiency" in payload:
            eff = payload["efficiency"]
            if not isinstance(eff, (int, float)):
                return False
            # Efficiency should be between 0 and 1 (or 0-100 for percentage)
            if eff <= 0:
                return False
            # Convert percentage to decimal if needed
            if eff > 1 and eff <= 100:
                payload["efficiency"] = eff / 100
            elif eff > 100:
                return False
        
        return True
    
    def run(self, payload: BoilerInput) -> AgentResult[BoilerOutput]:
        """Calculate emissions from boiler operations.
        
        Args:
            payload: Input data with boiler specifications
            
        Returns:
            AgentResult containing calculated emissions and boiler metrics
        """
        if not self.validate(payload):
            error_info: ErrorInfo = {
                "type": "ValidationError",
                "message": "Invalid input payload for boiler calculations",
                "agent_id": self.agent_id,
                "context": {"payload": payload}
            }
            return {"success": False, "error": error_info}
        
        boiler_type = payload["boiler_type"]
        fuel_type = payload.get("fuel_type", "natural_gas")
        country = payload.get("country", "US")
        year = payload.get("year", 2025)
        
        # Determine boiler efficiency
        efficiency = self._get_efficiency(payload, boiler_type, fuel_type)
        
        try:
            # Calculate fuel consumption based on input type
            if "thermal_output" in payload:
                fuel_consumption = self._calculate_fuel_from_thermal(
                    payload["thermal_output"],
                    efficiency,
                    fuel_type
                )
            else:
                fuel_consumption = payload["fuel_consumption"]
            
            # Get emission factor
            factor_info = self.emission_factors.get_factor_with_metadata(
                fuel_type=fuel_type,
                unit=fuel_consumption["unit"],
                region=country,
                year=year
            )
            
            if factor_info is None:
                error_info: ErrorInfo = {
                    "type": "DataError",
                    "message": f"No emission factor found for {fuel_type} in {country}",
                    "agent_id": self.agent_id,
                    "context": {"fuel_type": fuel_type, "country": country}
                }
                return {"success": False, "error": error_info}
            
            emission_factor = factor_info["emission_factor"]
            co2e_emissions_kg = fuel_consumption["value"] * emission_factor
            
            # Calculate thermal output if not provided
            if "thermal_output" in payload:
                thermal_output = payload["thermal_output"]
            else:
                thermal_output = self._calculate_thermal_from_fuel(
                    fuel_consumption,
                    efficiency,
                    fuel_type
                )
            
            # Calculate performance metrics
            thermal_efficiency_percent = efficiency * 100
            fuel_intensity = fuel_consumption["value"] / thermal_output["value"] if thermal_output["value"] > 0 else 0
            emission_intensity = co2e_emissions_kg / thermal_output["value"] if thermal_output["value"] > 0 else 0
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(
                boiler_type, fuel_type, efficiency, payload.get("age", "medium")
            )
            
            output: BoilerOutput = {
                "co2e_emissions_kg": co2e_emissions_kg,
                "boiler_type": boiler_type,
                "fuel_type": fuel_type,
                "fuel_consumption_value": fuel_consumption["value"],
                "fuel_consumption_unit": fuel_consumption["unit"],
                "thermal_output_value": thermal_output["value"],
                "thermal_output_unit": thermal_output["unit"],
                "efficiency": efficiency,
                "thermal_efficiency_percent": thermal_efficiency_percent,
                "emission_factor": emission_factor,
                "emission_factor_unit": f"kgCO2e/{fuel_consumption['unit']}",
                "fuel_intensity": fuel_intensity,
                "emission_intensity": emission_intensity,
                "recommendations": recommendations,
                "source": factor_info.get("source", "GreenLang Global Dataset"),
                "version": factor_info.get("version", "1.0.0"),
                "last_updated": factor_info.get("last_updated", "2025-08-14"),
            }
            
            if "confidence" in factor_info:
                output["confidence"] = factor_info["confidence"]
            
            # Add performance rating
            output["performance_rating"] = self._get_performance_rating(efficiency, boiler_type, fuel_type)
            
            return {
                "success": True,
                "data": output,
                "metadata": {
                    "agent_id": self.agent_id,
                    "calculation": f"Fuel: {fuel_consumption['value']} {fuel_consumption['unit']} × {emission_factor} kgCO2e/{fuel_consumption['unit']}",
                    "efficiency_used": f"{thermal_efficiency_percent:.1f}%",
                    "thermal_output": f"{thermal_output['value']} {thermal_output['unit']}"
                }
            }
            
        except Exception as e:
            error_info: ErrorInfo = {
                "type": "CalculationError",
                "message": f"Failed to calculate boiler emissions: {str(e)}",
                "agent_id": self.agent_id,
                "traceback": str(e)
            }
            return {"success": False, "error": error_info}
    
    def _get_efficiency(self, payload: Dict[str, Any], boiler_type: str, fuel_type: str) -> float:
        """Determine boiler efficiency from input or defaults.
        
        Args:
            payload: Input payload
            boiler_type: Type of boiler (e.g., condensing, standard)
            fuel_type: Type of fuel used
            
        Returns:
            float: Efficiency value (0-1 scale)
        """
        if "efficiency" in payload:
            return payload["efficiency"]
        
        # Use defaults based on boiler type and age
        age = payload.get("age", "medium")
        
        if fuel_type in self.BOILER_EFFICIENCIES:
            fuel_efficiencies = self.BOILER_EFFICIENCIES[fuel_type]
            if boiler_type in fuel_efficiencies:
                return fuel_efficiencies[boiler_type].get(age, 0.75)
        
        # Default efficiency if not found in lookup
        return 0.75
    
    def _calculate_fuel_from_thermal(self, thermal_output: Dict[str, Any], 
                                    efficiency: float, fuel_type: str) -> Dict[str, Any]:
        """Calculate fuel consumption from thermal output.
        
        Args:
            thermal_output: Thermal output with value and unit
            efficiency: Boiler efficiency (0-1 scale)
            fuel_type: Type of fuel
            
        Returns:
            Dict containing fuel consumption value and unit
        """
        thermal_value = thermal_output["value"]
        thermal_unit = thermal_output["unit"]
        
        # Convert thermal output to standard unit (MMBtu)
        if thermal_unit == "MMBtu":
            thermal_mmbtu = thermal_value
        elif thermal_unit == "kWh":
            thermal_mmbtu = thermal_value * 0.003412  # kWh to MMBtu
        elif thermal_unit == "MJ":
            thermal_mmbtu = thermal_value * 0.000948  # MJ to MMBtu
        elif thermal_unit == "therms":
            thermal_mmbtu = thermal_value * 0.1  # therms to MMBtu
        else:
            thermal_mmbtu = thermal_value  # Assume MMBtu if unknown
        
        # Calculate fuel input needed
        fuel_mmbtu = thermal_mmbtu / efficiency if efficiency > 0 else thermal_mmbtu
        
        # Convert to appropriate fuel unit
        if fuel_type == "natural_gas":
            return {"value": fuel_mmbtu * 10, "unit": "therms"}  # MMBtu to therms
        elif fuel_type == "oil":
            return {"value": fuel_mmbtu * 7.15, "unit": "gallons"}  # MMBtu to gallons (heating oil)
        elif fuel_type == "propane":
            return {"value": fuel_mmbtu * 10.92, "unit": "gallons"}  # MMBtu to gallons (propane)
        elif fuel_type == "electricity":
            return {"value": fuel_mmbtu / 0.003412, "unit": "kWh"}  # MMBtu to kWh
        else:
            return {"value": fuel_mmbtu, "unit": "MMBtu"}
    
    def _calculate_thermal_from_fuel(self, fuel_consumption: Dict[str, Any],
                                    efficiency: float, fuel_type: str) -> Dict[str, Any]:
        """Calculate thermal output from fuel consumption.
        
        Args:
            fuel_consumption: Fuel consumption with value and unit
            efficiency: Boiler efficiency (0-1 scale)
            fuel_type: Type of fuel
            
        Returns:
            Dict containing thermal output value and unit
        """
        fuel_value = fuel_consumption["value"]
        fuel_unit = fuel_consumption["unit"]
        
        # Convert fuel to MMBtu
        if fuel_unit == "MMBtu":
            fuel_mmbtu = fuel_value
        elif fuel_unit == "therms":
            fuel_mmbtu = fuel_value * 0.1
        elif fuel_unit == "kWh":
            fuel_mmbtu = fuel_value * 0.003412
        elif fuel_unit == "gallons":
            if fuel_type == "oil":
                fuel_mmbtu = fuel_value / 7.15
            elif fuel_type == "propane":
                fuel_mmbtu = fuel_value / 10.92
            else:
                fuel_mmbtu = fuel_value / 7.15  # Default to oil
        else:
            fuel_mmbtu = fuel_value
        
        # Calculate thermal output
        thermal_mmbtu = fuel_mmbtu * efficiency
        
        # Return in MMBtu
        return {"value": thermal_mmbtu, "unit": "MMBtu"}
    
    def _get_performance_rating(self, efficiency: float, boiler_type: str, fuel_type: str) -> str:
        """Determine performance rating based on efficiency.
        
        Args:
            efficiency: Boiler efficiency (0-1 scale)
            boiler_type: Type of boiler
            fuel_type: Type of fuel
            
        Returns:
            str: Performance rating (Excellent/Good/Average/Poor)
        """
        if fuel_type == "natural_gas":
            if efficiency >= 0.90:
                return "Excellent"
            elif efficiency >= 0.80:
                return "Good"
            elif efficiency >= 0.70:
                return "Average"
            else:
                return "Poor"
        elif fuel_type == "oil":
            if efficiency >= 0.85:
                return "Excellent"
            elif efficiency >= 0.75:
                return "Good"
            elif efficiency >= 0.65:
                return "Average"
            else:
                return "Poor"
        elif fuel_type == "electric":
            if boiler_type == "heat_pump" and efficiency >= 3.0:
                return "Excellent"
            elif boiler_type == "heat_pump" and efficiency >= 2.5:
                return "Good"
            elif efficiency >= 0.95:
                return "Average"
            else:
                return "Poor"
        else:
            # Generic rating
            if efficiency >= 0.85:
                return "Excellent"
            elif efficiency >= 0.75:
                return "Good"
            elif efficiency >= 0.65:
                return "Average"
            else:
                return "Poor"
    
    def _generate_recommendations(self, boiler_type: str, fuel_type: str, 
                                 efficiency: float, age: str) -> List[Dict[str, str]]:
        """Generate optimization recommendations for the boiler.
        
        Args:
            boiler_type: Type of boiler
            fuel_type: Type of fuel
            efficiency: Current efficiency
            age: Age category of boiler
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Efficiency-based recommendations
        if efficiency < 0.70:
            recommendations.append({
                "priority": "high",
                "action": "Replace boiler with high-efficiency condensing model",
                "impact": "30-40% emissions reduction",
                "payback": "3-5 years"
            })
        elif efficiency < 0.80:
            recommendations.append({
                "priority": "medium",
                "action": "Upgrade to modern efficient boiler",
                "impact": "15-25% emissions reduction",
                "payback": "4-6 years"
            })
        
        # Maintenance recommendations
        if age in ["old", "medium"]:
            recommendations.append({
                "priority": "high",
                "action": "Perform comprehensive boiler tune-up and cleaning",
                "impact": "5-10% efficiency improvement",
                "payback": "< 1 year"
            })
        
        # Fuel switching recommendations
        if fuel_type == "oil":
            recommendations.append({
                "priority": "medium",
                "action": "Consider switching to natural gas if available",
                "impact": "20-30% emissions reduction",
                "payback": "2-4 years"
            })
        elif fuel_type == "coal":
            recommendations.append({
                "priority": "high",
                "action": "Switch to cleaner fuel source (gas/biomass)",
                "impact": "40-50% emissions reduction",
                "payback": "3-5 years"
            })
        
        # Control system recommendations
        recommendations.append({
            "priority": "medium",
            "action": "Install smart boiler controls and weather compensation",
            "impact": "10-15% fuel savings",
            "payback": "2-3 years"
        })
        
        # Heat recovery recommendations
        if boiler_type != "condensing" and fuel_type in ["natural_gas", "oil"]:
            recommendations.append({
                "priority": "medium",
                "action": "Install flue gas heat recovery system",
                "impact": "5-8% efficiency improvement",
                "payback": "3-4 years"
            })
        
        # Insulation recommendations
        recommendations.append({
            "priority": "low",
            "action": "Improve boiler and pipe insulation",
            "impact": "2-5% heat loss reduction",
            "payback": "1-2 years"
        })
        
        return recommendations[:5]  # Return top 5 recommendations