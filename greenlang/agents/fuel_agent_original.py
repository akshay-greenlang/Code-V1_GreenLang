from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
from ..types import Agent, AgentResult, ErrorInfo
from .types import FuelInput, FuelOutput
from greenlang.data.emission_factors import EmissionFactors


class FuelAgent(Agent[FuelInput, FuelOutput]):
    """Agent for calculating emissions based on fuel consumption.
    
    This agent specializes in:
    - Direct fuel consumption to emissions conversion
    - Multi-fuel type support (electricity, gas, oil, diesel, renewable)
    - Country-specific emission factors
    - Renewable energy offset calculations
    - Fuel switching recommendations
    - Batch processing for multiple fuel sources
    """
    
    agent_id: str = "fuel"
    name: str = "Fuel Emissions Calculator"
    version: str = "0.0.1"
    
    # Standard fuel types and their properties
    FUEL_CONSTANTS = {
        "STANDARD_FUELS": [
            "electricity", "natural_gas", "diesel", "gasoline", "propane",
            "fuel_oil", "coal", "biomass", "biogas", "hydrogen"
        ],
        "RENEWABLE_FUELS": [
            "solar_pv", "wind", "hydro", "geothermal", "biomass"
        ],
        "FOSSIL_FUELS": [
            "natural_gas", "diesel", "gasoline", "propane", "fuel_oil", "coal"
        ],
        "SCOPE_MAPPING": {
            "natural_gas": "1",
            "diesel": "1",
            "gasoline": "1",
            "propane": "1",
            "fuel_oil": "1",
            "coal": "1",
            "biomass": "1",
            "biogas": "1",
            "electricity": "2",
            "district_heating": "2",
            "district_cooling": "2",
            "hydrogen": "2"  # If produced elsewhere
        },
        "DEFAULT_UNITS": {
            "electricity": "kWh",
            "natural_gas": "therms",
            "diesel": "gallons",
            "gasoline": "gallons",
            "propane": "gallons",
            "fuel_oil": "gallons",
            "coal": "tons",
            "biomass": "tons",
            "biogas": "m3",
            "hydrogen": "kg"
        },
        "UNIT_CONVERSIONS": {
            "therms_to_mmbtu": 0.1,
            "kwh_to_mmbtu": 0.003412,
            "gallons_diesel_to_mmbtu": 0.138,
            "gallons_gasoline_to_mmbtu": 0.125,
            "gallons_propane_to_mmbtu": 0.0915,
            "tons_coal_to_mmbtu": 20.0,
            "m3_to_therms": 0.36
        }
    }
    
    # Cache configuration
    CACHE_TTL_SECONDS = 3600  # 1 hour cache
    
    def __init__(self) -> None:
        """Initialize the FuelAgent with emission factors database.
        
        Sets up the emission factors database connection and initializes
        the cache for factor lookups.
        """
        self.emission_factors = EmissionFactors()
        self._cache = {}
        self._cache_timestamps = {}
    
    def validate(self, payload: FuelInput) -> bool:
        """Validate input payload structure and values.
        
        Args:
            payload: Input data containing fuel consumption information
            
        Returns:
            bool: True if validation passes, False otherwise
            
        Validates:
        - Required fields presence
        - Consumption data structure
        - Positive values for standard fuels
        - Negative values allowed for renewable generation
        """
        if not payload.get("fuel_type") or not payload.get("consumption"):
            return False
        
        consumption = payload["consumption"]
        if not isinstance(consumption, dict):
            return False
        if "value" not in consumption or "unit" not in consumption:
            return False
        
        fuel_type = payload["fuel_type"]
        
        # Allow negative values for renewable generation (offsets)
        if fuel_type in self.FUEL_CONSTANTS["RENEWABLE_FUELS"]:
            if consumption["value"] == 0:
                return False
        else:
            # Standard fuels must have positive consumption
            if consumption["value"] <= 0:
                return False
        
        return True
    
    @lru_cache(maxsize=128)
    def _get_cached_factor(self, fuel_type: str, unit: str, region: str, year: int) -> Optional[Dict]:
        """Get emission factor with caching for performance.
        
        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement
            region: Country or region code
            year: Year for historical factors
            
        Returns:
            Optional[Dict]: Emission factor info or None if not found
        """
        cache_key = f"{fuel_type}_{unit}_{region}_{year}"
        
        # Check if cache is still valid
        if cache_key in self._cache:
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp).seconds < self.CACHE_TTL_SECONDS:
                return self._cache[cache_key]
        
        # Fetch and cache the factor
        factor_info = self.emission_factors.get_factor_with_metadata(
            fuel_type=fuel_type,
            unit=unit,
            region=region,
            year=year
        )
        
        if factor_info:
            self._cache[cache_key] = factor_info
            self._cache_timestamps[cache_key] = datetime.now()
        
        return factor_info
    
    def run(self, payload: FuelInput) -> AgentResult[FuelOutput]:
        """Calculate emissions from fuel consumption.
        
        Args:
            payload: Input data with fuel consumption details
            
        Returns:
            AgentResult containing calculated emissions and metadata
            
        Calculates CO2e emissions based on fuel type, consumption amount,
        and region-specific emission factors. Includes recommendations
        for fuel optimization when applicable.
        """
        if not self.validate(payload):
            error_info: ErrorInfo = {
                "type": "ValidationError",
                "message": "Invalid input payload for fuel emissions calculation",
                "agent_id": self.agent_id,
                "context": {"payload": payload}
            }
            return {"success": False, "error": error_info}
        
        fuel_type = payload["fuel_type"]
        consumption = payload["consumption"]
        country = payload.get("country", "US")
        year = payload.get("year", 2025)
        
        try:
            # Use cached factor lookup
            factor_info = self._get_cached_factor(
                fuel_type=fuel_type,
                unit=consumption["unit"],
                region=country,
                year=year
            )
            
            if factor_info is None:
                error_info: ErrorInfo = {
                    "type": "DataError",
                    "message": f"No emission factor found for {fuel_type} in {country}",
                    "agent_id": self.agent_id,
                    "context": {"fuel_type": fuel_type, "country": country, "unit": consumption["unit"]}
                }
                return {"success": False, "error": error_info}
            
            emission_factor = factor_info["emission_factor"]
            co2e_emissions_kg = consumption["value"] * emission_factor
            
            # Determine scope
            scope = self.FUEL_CONSTANTS["SCOPE_MAPPING"].get(fuel_type, "3")
            
            # Calculate intensity metrics
            intensity_metrics = self._calculate_intensity_metrics(
                fuel_type, consumption, co2e_emissions_kg
            )
            
            # Generate recommendations
            recommendations = self._generate_fuel_recommendations(
                fuel_type, consumption, country, co2e_emissions_kg
            )
            
            output: FuelOutput = {
                "co2e_emissions_kg": co2e_emissions_kg,
                "fuel_type": fuel_type,
                "consumption_value": consumption["value"],
                "consumption_unit": consumption["unit"],
                "emission_factor": emission_factor,
                "emission_factor_unit": f"kgCO2e/{consumption['unit']}",
                "source": factor_info.get("source", "GreenLang Global Dataset"),
                "version": factor_info.get("version", "1.0.0"),
                "last_updated": factor_info.get("last_updated", "2025-08-14"),
                "scope": scope,
                "intensity_metrics": intensity_metrics,
                "recommendations": recommendations
            }
            
            if "confidence" in factor_info:
                output["confidence"] = factor_info["confidence"]
            
            # Add renewable offset information
            if fuel_type in self.FUEL_CONSTANTS["RENEWABLE_FUELS"] and consumption["value"] < 0:
                output["offset_type"] = "renewable_generation"
                output["avoided_emissions_kg"] = abs(co2e_emissions_kg)
            
            return {
                "success": True,
                "data": output,
                "metadata": {
                    "agent_id": self.agent_id,
                    "calculation": f"{consumption['value']} {consumption['unit']} × {emission_factor} kgCO2e/{consumption['unit']}",
                    "scope": scope,
                    "fuel_category": self._categorize_fuel(fuel_type)
                }
            }
            
        except Exception as e:
            error_info: ErrorInfo = {
                "type": "CalculationError",
                "message": f"Failed to calculate fuel emissions: {str(e)}",
                "agent_id": self.agent_id,
                "traceback": str(e)
            }
            return {"success": False, "error": error_info}
    
    def batch_process(self, fuels: List[FuelInput]) -> List[AgentResult[FuelOutput]]:
        """Process multiple fuel sources in batch for performance.
        
        Args:
            fuels: List of fuel input data
            
        Returns:
            List of AgentResult for each fuel source
            
        Efficiently processes multiple fuel sources, useful for
        facilities with mixed energy sources.
        """
        results = []
        for fuel_input in fuels:
            result = self.run(fuel_input)
            results.append(result)
        return results
    
    def _calculate_intensity_metrics(self, fuel_type: str, consumption: Dict, 
                                    emissions_kg: float) -> Dict[str, float]:
        """Calculate fuel intensity metrics.
        
        Args:
            fuel_type: Type of fuel
            consumption: Consumption data with value and unit
            emissions_kg: Calculated emissions in kg CO2e
            
        Returns:
            Dict containing intensity metrics
        """
        metrics = {}
        
        # Convert to common energy unit (MMBtu) for comparison
        if consumption["unit"] in ["kWh", "MWh"]:
            mmbtu = consumption["value"] * 0.003412
            if consumption["unit"] == "MWh":
                mmbtu *= 1000
        elif consumption["unit"] == "therms":
            mmbtu = consumption["value"] * 0.1
        elif consumption["unit"] == "gallons":
            if fuel_type == "diesel":
                mmbtu = consumption["value"] * 0.138
            elif fuel_type == "gasoline":
                mmbtu = consumption["value"] * 0.125
            elif fuel_type == "propane":
                mmbtu = consumption["value"] * 0.0915
            else:
                mmbtu = consumption["value"] * 0.138  # Default to diesel
        else:
            mmbtu = consumption["value"]  # Assume MMBtu if unknown
        
        if mmbtu > 0:
            metrics["emissions_per_mmbtu"] = emissions_kg / mmbtu
            metrics["energy_content_mmbtu"] = mmbtu
        
        return metrics
    
    def _generate_fuel_recommendations(self, fuel_type: str, consumption: Dict,
                                      country: str, emissions_kg: float) -> List[Dict[str, str]]:
        """Generate fuel optimization recommendations.
        
        Args:
            fuel_type: Type of fuel being used
            consumption: Consumption data
            country: Country/region
            emissions_kg: Calculated emissions
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Fossil fuel switching recommendations
        if fuel_type in self.FUEL_CONSTANTS["FOSSIL_FUELS"]:
            if fuel_type == "coal":
                recommendations.append({
                    "priority": "high",
                    "action": "Switch from coal to natural gas or biomass",
                    "impact": "50-60% emissions reduction",
                    "category": "fuel_switching"
                })
            elif fuel_type == "fuel_oil":
                recommendations.append({
                    "priority": "high",
                    "action": "Convert oil heating to natural gas or heat pump",
                    "impact": "25-30% emissions reduction",
                    "category": "fuel_switching"
                })
            elif fuel_type == "propane":
                recommendations.append({
                    "priority": "medium",
                    "action": "Consider switching to natural gas if available",
                    "impact": "10-15% emissions reduction",
                    "category": "fuel_switching"
                })
        
        # Electricity recommendations based on grid mix
        if fuel_type == "electricity":
            if country in ["IN", "CN", "PL"]:  # High coal grid countries
                recommendations.append({
                    "priority": "high",
                    "action": "Install on-site solar PV system",
                    "impact": "30-50% grid emissions offset",
                    "category": "renewable_energy"
                })
                recommendations.append({
                    "priority": "medium",
                    "action": "Purchase renewable energy certificates (RECs)",
                    "impact": "100% renewable attribution",
                    "category": "procurement"
                })
            
            recommendations.append({
                "priority": "medium",
                "action": "Implement demand response programs",
                "impact": "10-15% consumption reduction",
                "category": "efficiency"
            })
        
        # Natural gas recommendations
        if fuel_type == "natural_gas":
            recommendations.append({
                "priority": "medium",
                "action": "Upgrade to high-efficiency condensing equipment",
                "impact": "10-15% gas consumption reduction",
                "category": "efficiency"
            })
            recommendations.append({
                "priority": "low",
                "action": "Consider renewable natural gas (RNG) procurement",
                "impact": "Carbon neutral fuel source",
                "category": "procurement"
            })
        
        # General efficiency recommendations
        if consumption["value"] > 10000:  # Large consumer
            recommendations.append({
                "priority": "high",
                "action": "Conduct comprehensive energy audit",
                "impact": "Identify 15-30% savings opportunities",
                "category": "assessment"
            })
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _categorize_fuel(self, fuel_type: str) -> str:
        """Categorize fuel type for reporting.
        
        Args:
            fuel_type: Type of fuel
            
        Returns:
            str: Fuel category
        """
        if fuel_type in self.FUEL_CONSTANTS["RENEWABLE_FUELS"]:
            return "renewable"
        elif fuel_type in self.FUEL_CONSTANTS["FOSSIL_FUELS"]:
            return "fossil"
        elif fuel_type == "electricity":
            return "grid"
        else:
            return "other"
    
    def clear_cache(self) -> None:
        """Clear the emission factor cache.
        
        Useful when factors are updated or memory needs to be freed.
        """
        self._cache.clear()
        self._cache_timestamps.clear()
        self._get_cached_factor.cache_clear()
    
    def get_supported_fuels(self) -> Dict[str, List[str]]:
        """Get list of supported fuel types and their categories.
        
        Returns:
            Dict containing fuel categories and types
        """
        return {
            "standard": self.FUEL_CONSTANTS["STANDARD_FUELS"],
            "renewable": self.FUEL_CONSTANTS["RENEWABLE_FUELS"],
            "fossil": self.FUEL_CONSTANTS["FOSSIL_FUELS"]
        }
    
    def get_default_unit(self, fuel_type: str) -> Optional[str]:
        """Get the default unit for a fuel type.
        
        Args:
            fuel_type: Type of fuel
            
        Returns:
            Optional[str]: Default unit or None if not found
        """
        return self.FUEL_CONSTANTS["DEFAULT_UNITS"].get(fuel_type)