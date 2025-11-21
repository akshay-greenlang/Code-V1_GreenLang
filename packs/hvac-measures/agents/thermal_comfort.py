# -*- coding: utf-8 -*-
"""Thermal Comfort Analysis Agent for HVAC Optimization."""

import json
import math
from typing import Dict, Any, Optional
from datetime import datetime
from greenlang.determinism import DeterministicClock

class ThermalComfortAgent:
    """Agent for analyzing and optimizing thermal comfort in buildings."""
    
    def __init__(self):
        self.name = "thermal_comfort"
        self.version = "1.0.0"
        
    def load_building_data(self, 
                          building_type: str = "office",
                          floor_area: float = 10000,
                          zone_count: int = 5) -> Dict[str, Any]:
        """Load and structure building data for analysis."""
        return {
            "building_id": f"bldg_{DeterministicClock.now().strftime('%Y%m%d%H%M%S')}",
            "type": building_type,
            "floor_area_sqft": floor_area,
            "zones": [
                {
                    "id": f"zone_{i+1}",
                    "area": floor_area / zone_count,
                    "occupancy": self._get_typical_occupancy(building_type),
                    "orientation": ["north", "south", "east", "west"][i % 4]
                }
                for i in range(zone_count)
            ],
            "envelope": {
                "wall_r_value": 19,
                "roof_r_value": 30,
                "window_u_value": 0.35,
                "infiltration_ach": 0.5
            },
            "hvac_system": {
                "type": "VAV",
                "efficiency_cooling": 3.5,  # COP
                "efficiency_heating": 0.95   # AFUE
            }
        }
    
    def analyze_comfort(self,
                       building_model: Dict[str, Any],
                       target_pmv: float = 0.0,
                       target_ppd: float = 10.0) -> Dict[str, Any]:
        """Analyze thermal comfort and generate recommendations."""
        
        comfort_zones = []
        recommendations = []
        
        for zone in building_model["zones"]:
            # Calculate PMV/PPD for each zone
            pmv, ppd = self._calculate_pmv_ppd(
                air_temp=72,  # Default setpoint
                mean_radiant_temp=71,
                air_velocity=0.15,
                relative_humidity=50,
                metabolic_rate=1.2,  # Office work
                clothing_insulation=1.0  # Business clothing
            )
            
            comfort_zones.append({
                "zone_id": zone["id"],
                "current_pmv": pmv,
                "current_ppd": ppd,
                "target_pmv": target_pmv,
                "target_ppd": target_ppd,
                "in_comfort_range": abs(pmv) <= 0.5 and ppd <= 10
            })
            
            # Generate recommendations if outside comfort range
            if abs(pmv) > 0.5:
                if pmv > 0.5:
                    recommendations.append({
                        "zone_id": zone["id"],
                        "action": "decrease_temperature",
                        "adjustment": -2.0
                    })
                else:
                    recommendations.append({
                        "zone_id": zone["id"],
                        "action": "increase_temperature",
                        "adjustment": 2.0
                    })
        
        return {
            "comfort_analysis": {
                "timestamp": DeterministicClock.now().isoformat(),
                "zones": comfort_zones,
                "overall_comfort": sum(1 for z in comfort_zones if z["in_comfort_range"]) / len(comfort_zones)
            },
            "setpoint_recommendations": {
                "cooling_setpoint": 74,
                "heating_setpoint": 70,
                "humidity_setpoint": 45,
                "zone_adjustments": recommendations
            }
        }
    
    def _calculate_pmv_ppd(self, air_temp, mean_radiant_temp, air_velocity,
                          relative_humidity, metabolic_rate, clothing_insulation):
        """Calculate Predicted Mean Vote and Predicted Percentage Dissatisfied."""
        # Simplified PMV calculation (Fanger's equation)
        # In production, use full ISO 7730 implementation
        
        # Convert to SI units
        ta = air_temp * 5/9 - 32  # F to C
        tr = mean_radiant_temp * 5/9 - 32
        
        # Simplified PMV calculation
        pmv = (0.303 * math.exp(-0.036 * metabolic_rate * 58) + 0.028) * \
              ((metabolic_rate * 58 - 0.0014 * metabolic_rate * 58 * (34 - ta)) - \
               3.05 * (5.73 - 0.007 * metabolic_rate * 58) - \
               0.42 * (metabolic_rate * 58 - 58.15))
        
        # Clamp PMV to valid range
        pmv = max(-3, min(3, pmv))
        
        # Calculate PPD from PMV
        ppd = 100 - 95 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        
        return round(pmv, 2), round(ppd, 1)
    
    def _get_typical_occupancy(self, building_type: str) -> float:
        """Get typical occupancy density by building type."""
        occupancy_map = {
            "office": 0.005,  # people/sqft
            "retail": 0.015,
            "hospital": 0.010,
            "school": 0.025,
            "residential": 0.003
        }
        return occupancy_map.get(building_type, 0.005)

# Export agent instance
agent = ThermalComfortAgent()