"""Energy Calculator Agent for HVAC system analysis."""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random

class EnergyCalculatorAgent:
    """Agent for calculating HVAC energy consumption and costs."""
    
    def __init__(self):
        self.name = "energy_calculator"
        self.version = "1.0.0"
        self.electricity_rate = 0.12  # $/kWh
        self.gas_rate = 0.80  # $/therm
    
    def calculate_usage(self,
                       building_model: Dict[str, Any],
                       setpoints: Dict[str, Any],
                       weather_file: str = "standard",
                       occupancy_schedule: str = "standard") -> Dict[str, Any]:
        """Calculate energy usage based on building model and setpoints."""
        
        # Simulate hourly energy calculation for a typical day
        hourly_data = []
        total_cooling = 0
        total_heating = 0
        peak_demand = 0
        
        for hour in range(24):
            # Get outdoor temperature (simulated)
            outdoor_temp = self._get_outdoor_temp(hour, weather_file)
            
            # Get occupancy factor
            occupancy = self._get_occupancy_factor(hour, occupancy_schedule)
            
            # Calculate loads
            cooling_load = self._calculate_cooling_load(
                building_model, setpoints, outdoor_temp, occupancy
            )
            heating_load = self._calculate_heating_load(
                building_model, setpoints, outdoor_temp, occupancy
            )
            
            # Convert to energy consumption
            cooling_energy = cooling_load / building_model["hvac_system"]["efficiency_cooling"]
            heating_energy = heating_load / building_model["hvac_system"]["efficiency_heating"]
            
            total_cooling += cooling_energy
            total_heating += heating_energy
            peak_demand = max(peak_demand, cooling_energy + heating_energy)
            
            hourly_data.append({
                "hour": hour,
                "outdoor_temp": outdoor_temp,
                "occupancy": occupancy,
                "cooling_kw": cooling_energy,
                "heating_kw": heating_energy
            })
        
        # Calculate costs
        cooling_cost = total_cooling * self.electricity_rate
        heating_cost = total_heating * self.gas_rate / 29.3  # Convert kWh to therms
        
        return {
            "energy_consumption": {
                "daily_cooling_kwh": round(total_cooling, 2),
                "daily_heating_kwh": round(total_heating, 2),
                "annual_cooling_kwh": round(total_cooling * 365, 2),
                "annual_heating_kwh": round(total_heating * 365, 2),
                "hourly_profile": hourly_data
            },
            "peak_demand": round(peak_demand, 2),
            "cost_estimate": {
                "daily_cost": round(cooling_cost + heating_cost, 2),
                "monthly_cost": round((cooling_cost + heating_cost) * 30, 2),
                "annual_cost": round((cooling_cost + heating_cost) * 365, 2),
                "breakdown": {
                    "cooling": round(cooling_cost * 365, 2),
                    "heating": round(heating_cost * 365, 2)
                }
            }
        }
    
    def create_report(self,
                     comfort_analysis: Dict[str, Any],
                     energy_consumption: Dict[str, Any],
                     ventilation_plan: Dict[str, Any],
                     report_format: str = "pdf") -> Dict[str, Any]:
        """Generate optimization report."""
        
        # Create report content
        report = {
            "title": "HVAC Optimization Report",
            "generated": datetime.now().isoformat(),
            "executive_summary": self._generate_executive_summary(
                comfort_analysis, energy_consumption, ventilation_plan
            ),
            "sections": [
                {
                    "title": "Thermal Comfort Analysis",
                    "content": self._format_comfort_section(comfort_analysis)
                },
                {
                    "title": "Energy Consumption",
                    "content": self._format_energy_section(energy_consumption)
                },
                {
                    "title": "Ventilation Strategy",
                    "content": self._format_ventilation_section(ventilation_plan)
                },
                {
                    "title": "Recommendations",
                    "content": self._generate_recommendations(
                        comfort_analysis, energy_consumption, ventilation_plan
                    )
                }
            ]
        }
        
        # Save report (simulated)
        report_path = f"out/hvac_optimization_report.{report_format}"
        
        return {
            "optimization_report": report_path,
            "executive_summary": report["executive_summary"],
            "implementation_plan": self._generate_implementation_plan()
        }
    
    def _calculate_cooling_load(self, building_model, setpoints, outdoor_temp, occupancy):
        """Calculate cooling load in kW."""
        floor_area = building_model["floor_area_sqft"]
        
        # Simplified cooling load calculation
        if outdoor_temp > setpoints["cooling_setpoint"]:
            # Heat gain from envelope
            envelope_load = (outdoor_temp - setpoints["cooling_setpoint"]) * floor_area * 0.5
            # Internal gains
            internal_load = floor_area * occupancy * 3.0
            # Ventilation load
            ventilation_load = floor_area * 0.2 * (outdoor_temp - setpoints["cooling_setpoint"])
            
            total_load = (envelope_load + internal_load + ventilation_load) / 1000  # Convert to kW
            return max(0, total_load)
        return 0
    
    def _calculate_heating_load(self, building_model, setpoints, outdoor_temp, occupancy):
        """Calculate heating load in kW."""
        floor_area = building_model["floor_area_sqft"]
        
        # Simplified heating load calculation
        if outdoor_temp < setpoints["heating_setpoint"]:
            # Heat loss through envelope
            envelope_load = (setpoints["heating_setpoint"] - outdoor_temp) * floor_area * 0.4
            # Credit for internal gains
            internal_credit = floor_area * occupancy * 2.0
            # Ventilation load
            ventilation_load = floor_area * 0.2 * (setpoints["heating_setpoint"] - outdoor_temp)
            
            total_load = (envelope_load + ventilation_load - internal_credit) / 1000  # Convert to kW
            return max(0, total_load)
        return 0
    
    def _get_outdoor_temp(self, hour, weather_file):
        """Get outdoor temperature for given hour."""
        # Simulated diurnal temperature variation
        base_temp = 70
        amplitude = 15
        return base_temp + amplitude * math.sin((hour - 6) * math.pi / 12)
    
    def _get_occupancy_factor(self, hour, schedule):
        """Get occupancy factor for given hour."""
        if schedule == "standard":
            if 8 <= hour <= 18:
                return 0.9
            elif 6 <= hour < 8 or 18 < hour <= 20:
                return 0.3
            else:
                return 0.1
        return 0.5
    
    def _generate_executive_summary(self, comfort, energy, ventilation):
        """Generate executive summary."""
        return {
            "key_findings": [
                f"Overall thermal comfort: {comfort['overall_comfort']*100:.1f}%",
                f"Annual energy cost: ${energy['cost_estimate']['annual_cost']:,.0f}",
                f"Potential savings: 20-25% with recommended optimizations"
            ],
            "priority_actions": [
                "Implement zone-based temperature setpoints",
                "Upgrade to demand-controlled ventilation",
                "Schedule retro-commissioning"
            ]
        }
    
    def _format_comfort_section(self, comfort_analysis):
        """Format comfort analysis section."""
        return {
            "summary": f"Comfort compliance: {comfort_analysis['overall_comfort']*100:.1f}%",
            "zones": comfort_analysis["zones"],
            "recommendations": "Adjust setpoints per zone requirements"
        }
    
    def _format_energy_section(self, energy_consumption):
        """Format energy consumption section."""
        return {
            "annual_consumption": f"{energy_consumption['annual_cooling_kwh'] + energy_consumption['annual_heating_kwh']:,.0f} kWh",
            "peak_demand": f"{energy_consumption['peak_demand']:.1f} kW",
            "cost_breakdown": energy_consumption["cost_estimate"]["breakdown"]
        }
    
    def _format_ventilation_section(self, ventilation_plan):
        """Format ventilation section."""
        return {
            "strategy": ventilation_plan.get("strategy", "Demand-controlled ventilation"),
            "air_changes": ventilation_plan.get("air_changes_per_hour", 4),
            "filtration": ventilation_plan.get("filter_type", "MERV 13")
        }
    
    def _generate_recommendations(self, comfort, energy, ventilation):
        """Generate optimization recommendations."""
        return [
            {
                "category": "Immediate",
                "items": [
                    "Adjust zone setpoints based on occupancy",
                    "Clean or replace air filters",
                    "Calibrate sensors"
                ]
            },
            {
                "category": "Short-term",
                "items": [
                    "Implement occupancy-based scheduling",
                    "Optimize economizer operation",
                    "Install CO2 sensors for DCV"
                ]
            },
            {
                "category": "Long-term",
                "items": [
                    "Upgrade to high-efficiency equipment",
                    "Add thermal energy storage",
                    "Implement predictive controls"
                ]
            }
        ]
    
    def _generate_implementation_plan(self):
        """Generate implementation plan."""
        return {
            "phase_1": {
                "duration": "1-2 weeks",
                "tasks": ["Sensor calibration", "Setpoint optimization", "Schedule updates"]
            },
            "phase_2": {
                "duration": "1-2 months",
                "tasks": ["DCV implementation", "Economizer tuning", "Staff training"]
            },
            "phase_3": {
                "duration": "6-12 months",
                "tasks": ["Equipment upgrades", "Control system modernization", "Continuous commissioning"]
            }
        }

import math  # Add missing import

# Export agent instance
agent = EnergyCalculatorAgent()