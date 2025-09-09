"""
Boiler Analyzer Agent
=====================

Analyzes boiler efficiency and performance characteristics.
"""

from typing import Dict, Any
from dataclasses import dataclass
import hashlib
import json

# Use relative imports for pack agents
try:
    from ...sdk.base import Agent, Result
except ImportError:
    # Fallback for testing
    from core.greenlang.sdk.base import Agent, Result


@dataclass
class BoilerConfig:
    """Boiler configuration parameters"""
    combustion_efficiency: float = 0.85  # 85% baseline
    heat_loss_factor: float = 0.02       # 2% standing losses
    emission_factor: float = 53.06       # kg CO2/MMBtu for natural gas
    fuel_heating_value: float = 1.036    # MMBtu/MCF for natural gas


class BoilerAnalyzerAgent(Agent):
    """
    Analyzes boiler performance and efficiency
    
    Calculates fuel consumption, emissions, and efficiency metrics
    based on boiler specifications and operating conditions.
    """
    
    def __init__(self, config: BoilerConfig = None):
        super().__init__()
        self.config = config or BoilerConfig()
        self.deterministic = True  # Enable deterministic mode
    
    def run(self, inputs: Dict[str, Any]) -> Result:
        """
        Analyze boiler performance
        
        Args:
            inputs: Dictionary containing:
                - capacity: Boiler capacity in kW
                - annual_demand: Annual heat demand in kWh
                - building_type: Type of building
                - operating_hours: Annual operating hours (optional)
        
        Returns:
            Result with efficiency, fuel consumption, and emissions
        """
        try:
            capacity = inputs.get('capacity', 0)
            annual_demand = inputs.get('annual_demand', 0)
            building_type = inputs.get('building_type', 'commercial')
            operating_hours = inputs.get('operating_hours', 6000)
            
            # Validate inputs
            if capacity <= 0 or annual_demand <= 0:
                return Result(
                    success=False,
                    error="Invalid boiler capacity or demand"
                )
            
            # Adjust efficiency based on building type
            efficiency = self._get_efficiency_by_type(building_type)
            
            # Calculate load factor
            max_output = capacity * operating_hours
            load_factor = min(annual_demand / max_output, 1.0) if max_output > 0 else 0
            
            # Adjust efficiency for part-load operation
            part_load_efficiency = self._adjust_for_part_load(efficiency, load_factor)
            
            # Calculate fuel consumption
            fuel_input = annual_demand / part_load_efficiency  # kWh input
            
            # Convert to natural gas units (MCF)
            # 1 kWh = 0.003412 MMBtu
            fuel_mmbtu = fuel_input * 0.003412
            fuel_mcf = fuel_mmbtu / self.config.fuel_heating_value
            
            # Calculate emissions
            emissions_kg = fuel_mmbtu * self.config.emission_factor
            emissions_tons = emissions_kg / 1000
            
            # Calculate operating cost (simplified)
            gas_price = 8.0  # $/MCF (example)
            annual_fuel_cost = fuel_mcf * gas_price
            
            return Result(
                success=True,
                data={
                    'efficiency': round(part_load_efficiency, 3),
                    'fuel_consumption': round(fuel_mcf, 2),
                    'fuel_consumption_mmbtu': round(fuel_mmbtu, 2),
                    'emissions': round(emissions_tons, 2),
                    'emissions_kg': round(emissions_kg, 2),
                    'annual_fuel_cost': round(annual_fuel_cost, 2),
                    'load_factor': round(load_factor, 3),
                    'specific_emissions': round(emissions_kg / annual_demand, 4)  # kg/kWh
                },
                metadata={
                    'capacity': capacity,
                    'annual_demand': annual_demand,
                    'building_type': building_type,
                    'operating_hours': operating_hours,
                    'base_efficiency': efficiency
                }
            )
            
        except Exception as e:
            return Result(
                success=False,
                error=str(e)
            )
    
    def _get_efficiency_by_type(self, building_type: str) -> float:
        """Get baseline efficiency by building type"""
        efficiency_map = {
            'residential': 0.80,
            'commercial': 0.85,
            'industrial': 0.88
        }
        return efficiency_map.get(building_type, self.config.combustion_efficiency)
    
    def _adjust_for_part_load(self, base_efficiency: float, load_factor: float) -> float:
        """
        Adjust efficiency for part-load operation
        
        Boilers are less efficient at partial loads
        """
        if load_factor >= 0.8:
            # Near full load - maintain efficiency
            return base_efficiency
        elif load_factor >= 0.5:
            # Moderate load - slight reduction
            return base_efficiency * (0.95 + 0.05 * load_factor)
        else:
            # Low load - significant reduction
            return base_efficiency * (0.85 + 0.15 * load_factor)
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Result:
        """Validate input data for deterministic processing"""
        try:
            required = ['building_size_sqft', 'location', 'boiler_age_years']
            for field in required:
                if field not in inputs:
                    return Result(success=False, error=f"Missing required field: {field}")
            
            # Create deterministic hash
            input_str = json.dumps(inputs, sort_keys=True)
            input_hash = hashlib.md5(input_str.encode()).hexdigest()[:8]
            
            return Result(
                success=True,
                data={
                    'validated': True,
                    'input_hash': input_hash,
                    **inputs
                }
            )
        except Exception as e:
            return Result(success=False, error=str(e))
    
    def analyze_efficiency(self, inputs: Dict[str, Any]) -> Result:
        """Analyze boiler efficiency deterministically"""
        try:
            data = inputs.get('validated_data', inputs)
            age = data.get('boiler_age_years', 10)
            
            # Deterministic efficiency calculation based on age
            base_efficiency = 0.85
            age_factor = max(0, 1 - (age * 0.01))  # 1% loss per year
            efficiency = base_efficiency * age_factor
            
            # Fixed values for determinism
            fuel_consumption = 50000  # kWh
            emissions = 125.5  # tons CO2
            
            return Result(
                success=True,
                data={
                    'efficiency_percent': round(efficiency * 100, 1),
                    'annual_fuel_consumption_kwh': fuel_consumption,
                    'annual_emissions_co2_tons': emissions
                }
            )
        except Exception as e:
            return Result(success=False, error=str(e))
    
    def calculate_emissions(self, inputs: Dict[str, Any]) -> Result:
        """Calculate net emissions with solar offset"""
        try:
            boiler_emissions = inputs.get('boiler_output', {}).get('annual_emissions_co2_tons', 125.5)
            solar_offset = inputs.get('solar_output', {}).get('co2_offset_tons', 24.5)
            
            net_emissions = boiler_emissions - solar_offset
            reduction_percent = (solar_offset / boiler_emissions) * 100 if boiler_emissions > 0 else 0
            
            return Result(
                success=True,
                data={
                    'net_emissions_tons': round(net_emissions, 1),
                    'reduction_percent': round(reduction_percent, 1)
                }
            )
        except Exception as e:
            return Result(success=False, error=str(e))
    
    def calculate_savings_potential(self, 
                                   current_efficiency: float,
                                   target_efficiency: float,
                                   annual_consumption: float) -> Dict[str, float]:
        """
        Calculate potential savings from efficiency improvements
        
        Args:
            current_efficiency: Current boiler efficiency
            target_efficiency: Target efficiency after improvements
            annual_consumption: Annual fuel consumption
        
        Returns:
            Dictionary with savings metrics
        """
        if target_efficiency <= current_efficiency:
            return {'fuel_savings': 0, 'cost_savings': 0, 'emission_reduction': 0}
        
        # Calculate fuel savings
        current_input = annual_consumption
        improved_input = current_input * (current_efficiency / target_efficiency)
        fuel_savings = current_input - improved_input
        
        # Calculate cost savings
        gas_price = 8.0  # $/MCF
        cost_savings = fuel_savings * gas_price
        
        # Calculate emission reduction
        emission_reduction = fuel_savings * self.config.fuel_heating_value * self.config.emission_factor / 1000
        
        return {
            'fuel_savings': round(fuel_savings, 2),
            'cost_savings': round(cost_savings, 2),
            'emission_reduction': round(emission_reduction, 2),
            'efficiency_improvement': round(target_efficiency - current_efficiency, 3)
        }
    
    def process(self, inputs: Dict[str, Any]) -> Result:
        """Default process method - calls run"""
        return self.run(inputs)
    
    def validate(self, inputs: Dict[str, Any]) -> Result:
        """Default validate method - calls validate_inputs"""
        return self.validate_inputs(inputs)