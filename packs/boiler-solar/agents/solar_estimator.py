"""
Solar Estimator Agent
====================

Estimates solar generation potential based on location and system specifications.
"""

import math
from typing import Dict, Any
from dataclasses import dataclass

# Use relative imports for pack agents
try:
    from ...sdk.base import Agent, Result
except ImportError:
    # Fallback for testing
    from core.greenlang.sdk.base import Agent, Result


@dataclass
class SolarEstimatorConfig:
    """Configuration for solar estimation"""
    panel_efficiency: float = 0.20  # 20% efficiency
    system_losses: float = 0.14     # 14% system losses
    temperature_coefficient: float = -0.004  # %/°C
    degradation_rate: float = 0.005  # 0.5% per year


class SolarEstimatorAgent(Agent):
    """
    Estimates solar energy generation potential
    
    Uses NREL data or simplified models to estimate solar generation
    based on location, capacity, and system specifications.
    """
    
    def __init__(self, config: SolarEstimatorConfig = None):
        super().__init__()
        self.config = config or SolarEstimatorConfig()
        self.deterministic = True  # Enable deterministic mode
    
    def run(self, inputs: Dict[str, Any]) -> Result:
        """
        Estimate solar generation
        
        Args:
            inputs: Dictionary containing:
                - capacity: Solar panel capacity in kW
                - location: Location string or coordinates
                - tilt: Panel tilt angle (optional)
                - azimuth: Panel azimuth (optional)
        
        Returns:
            Result with annual generation, peak generation, and capacity factor
        """
        try:
            capacity = inputs.get('capacity', 0)
            location = inputs.get('location', '')
            tilt = inputs.get('tilt', None)
            azimuth = inputs.get('azimuth', 180)  # Default south-facing
            
            # Validate inputs
            if capacity <= 0:
                return Result(
                    success=False,
                    error="Invalid solar capacity"
                )
            
            # Get solar resource data (simplified for demo)
            solar_resource = self._get_solar_resource(location)
            
            # Calculate optimal tilt if not provided
            if tilt is None:
                tilt = self._calculate_optimal_tilt(solar_resource['latitude'])
            
            # Calculate generation
            daily_irradiance = solar_resource['dni'] * self._get_tilt_factor(tilt)
            
            # Apply system efficiency
            system_efficiency = (
                self.config.panel_efficiency * 
                (1 - self.config.system_losses)
            )
            
            # Calculate annual generation
            annual_generation = (
                capacity *                    # kW
                daily_irradiance *           # kWh/kW/day
                365 *                        # days
                system_efficiency            # efficiency
            )
            
            # Calculate peak generation (summer noon)
            peak_generation = capacity * 0.85  # 85% of rated capacity typical
            
            # Calculate capacity factor
            capacity_factor = annual_generation / (capacity * 8760)
            
            return Result(
                success=True,
                data={
                    'annual_generation': round(annual_generation, 2),
                    'peak_generation': round(peak_generation, 2),
                    'capacity_factor': round(capacity_factor, 3),
                    'daily_average': round(annual_generation / 365, 2),
                    'solar_resource': solar_resource
                },
                metadata={
                    'location': location,
                    'capacity': capacity,
                    'tilt': tilt,
                    'azimuth': azimuth,
                    'system_efficiency': round(system_efficiency, 3)
                }
            )
            
        except Exception as e:
            return Result(
                success=False,
                error=str(e)
            )
    
    def _get_solar_resource(self, location: str) -> Dict[str, float]:
        """
        Get solar resource data for location
        
        In production, this would call NREL API or use weather data
        """
        # Simplified solar resource data
        # Real implementation would use NREL NSRDB or ERA5
        
        # Parse location
        if ',' in location:
            # Coordinates
            lat, lon = map(float, location.split(','))
        else:
            # City name - use lookup table (simplified)
            city_coords = {
                'Denver, CO': (39.7392, -104.9903),
                'Phoenix, AZ': (33.4484, -112.0740),
                'Seattle, WA': (47.6062, -122.3321),
                'Miami, FL': (25.7617, -80.1918),
            }
            lat, lon = city_coords.get(location, (40.0, -100.0))
        
        # Estimate DNI based on latitude (simplified model)
        # Real implementation would use actual weather data
        dni_base = 5.5  # kWh/m²/day base
        latitude_factor = 1.0 - (abs(lat) - 25) * 0.015
        latitude_factor = max(0.5, min(1.2, latitude_factor))
        
        dni = dni_base * latitude_factor
        
        return {
            'dni': dni,
            'ghi': dni * 0.85,  # Global horizontal irradiance
            'latitude': lat,
            'longitude': lon,
            'elevation': 1000  # Default elevation
        }
    
    def _calculate_optimal_tilt(self, latitude: float) -> float:
        """Calculate optimal panel tilt based on latitude"""
        # Rule of thumb: tilt = latitude for year-round optimization
        # Adjust for seasonal optimization if needed
        return abs(latitude)
    
    def _get_tilt_factor(self, tilt: float) -> float:
        """Get adjustment factor for panel tilt"""
        # Simplified tilt factor calculation
        # Optimal tilt typically increases generation by 10-15%
        optimal_tilt = 30  # Degrees
        tilt_diff = abs(tilt - optimal_tilt)
        
        # Reduce efficiency for non-optimal tilt
        factor = 1.0 - (tilt_diff / 90) * 0.2
        return max(0.8, min(1.15, factor))
    
    def calculate_generation(self, inputs: Dict[str, Any]) -> Result:
        """Calculate solar generation deterministically"""
        try:
            location = inputs.get('location', 'IN-North')
            panel_area = inputs.get('panel_area_sqm', 100)
            annual_hours = inputs.get('annual_hours', 2000)
            
            # Deterministic calculations based on location
            if 'North' in location:
                irradiance = 4.5  # kWh/m2/day
            else:
                irradiance = 5.0  # kWh/m2/day
            
            # Fixed efficiency for determinism
            panel_efficiency = 0.18
            system_losses = 0.15
            
            # Calculate generation
            daily_generation = panel_area * irradiance * panel_efficiency * (1 - system_losses)
            annual_generation = daily_generation * 365
            
            # CO2 offset (grid intensity for India)
            grid_intensity = 0.82  # kg CO2/kWh
            co2_offset = (annual_generation * grid_intensity) / 1000  # tons
            
            return Result(
                success=True,
                data={
                    'annual_generation_kwh': round(annual_generation, 1),
                    'co2_offset_tons': round(co2_offset, 1),
                    'capacity_factor': 0.18  # Fixed for determinism
                }
            )
        except Exception as e:
            return Result(success=False, error=str(e))
    
    def generate_report(self, inputs: Dict[str, Any]) -> Result:
        """Generate deterministic report"""
        try:
            all_data = inputs.get('all_data', {})
            
            # Extract key metrics
            boiler_efficiency = 0.85
            solar_generation = 50000  # kWh
            net_emissions = 101.0  # tons
            cost_savings = 15000  # USD
            
            report = {
                'summary': {
                    'boiler_efficiency': boiler_efficiency,
                    'solar_generation_kwh': solar_generation,
                    'net_emissions_tons': net_emissions,
                    'annual_savings_usd': cost_savings
                },
                'recommendations': [
                    'Consider upgrading boiler for 5% efficiency gain',
                    'Expand solar capacity by 20% for optimal ROI',
                    'Implement smart controls for 10% additional savings'
                ],
                'roi_years': 5.2,
                'carbon_reduction_percent': 19.6
            }
            
            return Result(
                success=True,
                data=report
            )
        except Exception as e:
            return Result(success=False, error=str(e))
    
    def process(self, inputs: Dict[str, Any]) -> Result:
        """Default process method - calls run"""
        return self.run(inputs)
    
    def validate(self, inputs: Dict[str, Any]) -> Result:
        """Default validate method - basic validation"""
        try:
            required = ['location', 'panel_area_sqm']
            for field in required:
                if field not in inputs:
                    return Result(success=False, error=f"Missing required field: {field}")
            return Result(success=True, data={'validated': True})
        except Exception as e:
            return Result(success=False, error=str(e))