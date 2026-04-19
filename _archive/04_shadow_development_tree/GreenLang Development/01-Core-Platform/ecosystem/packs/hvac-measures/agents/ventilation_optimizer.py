# -*- coding: utf-8 -*-
"""Ventilation Optimizer Agent for indoor air quality and efficiency."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class VentilationOptimizerAgent:
    """Agent for optimizing ventilation strategies."""
    
    def __init__(self):
        self.name = "ventilation_optimizer"
        self.version = "1.0.0"
        
        # Ventilation standards
        self.standards = {
            "ASHRAE62.1": {
                "office": {"cfm_per_person": 5, "cfm_per_sqft": 0.06},
                "retail": {"cfm_per_person": 7.5, "cfm_per_sqft": 0.06},
                "school": {"cfm_per_person": 10, "cfm_per_sqft": 0.12},
                "hospital": {"cfm_per_person": 15, "cfm_per_sqft": 0.18}
            }
        }
    
    def optimize(self,
                building_model: Dict[str, Any],
                energy_data: Dict[str, Any],
                iaq_requirements: str = "ASHRAE62.1",
                covid_measures: str = "standard") -> Dict[str, Any]:
        """Optimize ventilation strategy for IAQ and energy efficiency."""
        
        building_type = building_model["type"]
        floor_area = building_model["floor_area_sqft"]
        zones = building_model["zones"]
        
        # Calculate minimum ventilation requirements
        min_ventilation = self._calculate_minimum_ventilation(
            building_type, floor_area, zones, iaq_requirements
        )
        
        # Apply COVID-19 enhancements if needed
        if covid_measures in ["enhanced", "maximum"]:
            min_ventilation = self._apply_covid_enhancements(
                min_ventilation, covid_measures
            )
        
        # Generate ventilation schedule
        ventilation_schedule = self._generate_ventilation_schedule(
            min_ventilation, energy_data
        )
        
        # Calculate air changes per hour
        room_volume = floor_area * 10  # Assume 10ft ceiling
        total_cfm = sum(v["cfm"] for v in ventilation_schedule)
        ach = (total_cfm * 60) / room_volume
        
        # Generate filter recommendations
        filter_recommendations = self._recommend_filters(
            building_type, covid_measures
        )
        
        return {
            "ventilation_schedule": ventilation_schedule,
            "air_changes_per_hour": round(ach, 1),
            "filter_recommendations": filter_recommendations
        }
    
    def _calculate_minimum_ventilation(self, building_type, floor_area, zones, standard):
        """Calculate minimum ventilation requirements."""
        if standard not in self.standards:
            standard = "ASHRAE62.1"
        
        req = self.standards[standard].get(building_type, self.standards[standard]["office"])
        
        ventilation_zones = []
        for zone in zones:
            zone_area = zone["area"]
            zone_occupancy = zone["occupancy"] * zone_area
            
            # Calculate required CFM
            people_cfm = zone_occupancy * req["cfm_per_person"]
            area_cfm = zone_area * req["cfm_per_sqft"]
            total_cfm = people_cfm + area_cfm
            
            ventilation_zones.append({
                "zone_id": zone["id"],
                "cfm": round(total_cfm, 0),
                "type": "minimum_required"
            })
        
        return ventilation_zones
    
    def _apply_covid_enhancements(self, base_ventilation, level):
        """Apply COVID-19 ventilation enhancements."""
        multiplier = 1.5 if level == "enhanced" else 2.0
        
        enhanced_ventilation = []
        for zone in base_ventilation:
            enhanced_zone = zone.copy()
            enhanced_zone["cfm"] = round(zone["cfm"] * multiplier, 0)
            enhanced_zone["type"] = f"covid_{level}"
            enhanced_ventilation.append(enhanced_zone)
        
        return enhanced_ventilation
    
    def _generate_ventilation_schedule(self, min_ventilation, energy_data):
        """Generate hourly ventilation schedule."""
        schedule = []
        
        # Create 24-hour schedule
        for hour in range(24):
            # Determine ventilation mode based on time and energy data
            if 6 <= hour <= 20:  # Occupied hours
                mode = "normal"
                cfm_multiplier = 1.0
            elif 20 < hour <= 22:  # Cleaning hours
                mode = "purge"
                cfm_multiplier = 1.5
            else:  # Unoccupied
                mode = "minimum"
                cfm_multiplier = 0.3
            
            for zone in min_ventilation:
                schedule.append({
                    "hour": hour,
                    "zone_id": zone["zone_id"],
                    "cfm": round(zone["cfm"] * cfm_multiplier, 0),
                    "mode": mode
                })
        
        return schedule
    
    def _recommend_filters(self, building_type, covid_measures):
        """Recommend appropriate filters."""
        base_recommendations = {
            "office": "MERV 13",
            "hospital": "HEPA",
            "school": "MERV 13",
            "retail": "MERV 11"
        }
        
        filter_type = base_recommendations.get(building_type, "MERV 11")
        
        # Upgrade for COVID measures
        if covid_measures == "enhanced" and filter_type == "MERV 11":
            filter_type = "MERV 13"
        elif covid_measures == "maximum" and filter_type != "HEPA":
            filter_type = "MERV 14"
        
        return {
            "primary_filter": filter_type,
            "pre_filter": "MERV 8" if filter_type in ["HEPA", "MERV 14"] else None,
            "replacement_schedule": "3 months" if covid_measures != "standard" else "6 months",
            "uv_c_recommended": covid_measures in ["enhanced", "maximum"],
            "notes": [
                f"Selected {filter_type} based on {building_type} requirements",
                f"COVID-19 measures: {covid_measures}",
                "Monitor pressure drop across filters",
                "Consider portable HEPA units for high-risk areas"
            ]
        }

# Export agent instance
agent = VentilationOptimizerAgent()