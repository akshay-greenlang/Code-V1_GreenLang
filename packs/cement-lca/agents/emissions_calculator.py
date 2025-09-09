"""Emissions Calculator Agent for cement and concrete LCA."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class EmissionsCalculatorAgent:
    """Agent for calculating life cycle emissions."""
    
    def __init__(self):
        self.name = "emissions_calculator"
        self.version = "1.0.0"
        
        # Emission factors (kg CO2-eq per unit)
        self.emission_factors = {
            "clinker": 0.866,  # kg CO2/kg clinker
            "gypsum": 0.01,
            "limestone": 0.008,
            "fly_ash": 0.027,
            "ggbs": 0.143,
            "silica_fume": 0.014,
            "calcined_clay": 0.3,
            "fine_aggregate": 0.0139,
            "coarse_aggregate": 0.0075,
            "water": 0.000344,
            "admixtures": 0.25,
            "electricity": 0.527,  # kg CO2/kWh (US grid average)
            "coal": 2.86,  # kg CO2/kg coal
            "natural_gas": 2.75,  # kg CO2/m³
            "diesel": 2.68,  # kg CO2/liter
            "truck_transport": 0.0001,  # kg CO2/kg/km
            "rail_transport": 0.00003,  # kg CO2/kg/km
            "ship_transport": 0.00001   # kg CO2/kg/km
        }
    
    def calculate_production_emissions(self,
                                      material_inventory: Dict[str, Any],
                                      production_method: str = "dry_process",
                                      fuel_type: str = "coal",
                                      electricity_grid: str = "US_average",
                                      plant_efficiency: float = 0.85) -> Dict[str, Any]:
        """Calculate production stage emissions."""
        
        materials = material_inventory["materials"]
        total_emissions = 0
        process_emissions = 0
        energy_emissions = 0
        
        # Calculate material emissions
        material_emissions = {}
        for material, data in materials.items():
            if material in self.emission_factors:
                emissions = data["amount_kg"] * self.emission_factors[material]
                material_emissions[material] = round(emissions, 2)
                total_emissions += emissions
                
                # Clinker has additional process emissions
                if material == "clinker":
                    process_co2 = data["amount_kg"] * 0.525  # CaCO3 decomposition
                    process_emissions += process_co2
                    total_emissions += process_co2
        
        # Calculate energy emissions
        energy_consumption = self._calculate_energy_use(
            material_inventory, production_method, plant_efficiency
        )
        
        for energy_type, amount in energy_consumption.items():
            if energy_type == "electricity":
                energy_emissions += amount * self.emission_factors["electricity"]
            elif energy_type == "thermal":
                fuel_emissions = amount * self.emission_factors.get(fuel_type, 2.5)
                energy_emissions += fuel_emissions
        
        total_emissions += energy_emissions
        
        return {
            "production_emissions": {
                "total_kg_co2": round(total_emissions, 2),
                "material_emissions": material_emissions,
                "process_emissions": round(process_emissions, 2),
                "energy_emissions": round(energy_emissions, 2),
                "emissions_per_m3": round(total_emissions / material_inventory["volume_m3"], 2)
            },
            "energy_consumption": energy_consumption,
            "process_emissions": {
                "calcination_co2": round(process_emissions, 2),
                "fuel_combustion_co2": round(energy_emissions, 2)
            }
        }
    
    def calculate_transport_emissions(self,
                                     material_inventory: Dict[str, Any],
                                     transport_mode: str = "truck",
                                     distance_km: float = 100,
                                     return_empty: bool = True) -> Dict[str, Any]:
        """Calculate transportation emissions."""
        
        total_mass = material_inventory["total_mass_kg"]
        
        # Get emission factor for transport mode
        transport_factor = self.emission_factors.get(f"{transport_mode}_transport", 0.0001)
        
        # Calculate emissions
        one_way_emissions = total_mass * distance_km * transport_factor
        
        if return_empty:
            total_emissions = one_way_emissions * 1.5  # Account for empty return
        else:
            total_emissions = one_way_emissions
        
        # Calculate fuel consumption (estimated)
        fuel_consumption = self._estimate_fuel_consumption(
            total_mass, distance_km, transport_mode
        )
        
        return {
            "transport_emissions": {
                "total_kg_co2": round(total_emissions, 2),
                "emissions_per_m3": round(total_emissions / material_inventory["volume_m3"], 2),
                "transport_mode": transport_mode,
                "distance_km": distance_km,
                "mass_transported_kg": total_mass
            },
            "fuel_consumption": fuel_consumption
        }
    
    def calculate_use_phase(self,
                           material_inventory: Dict[str, Any],
                           service_life_years: int = 50,
                           carbonation_rate: str = "standard",
                           maintenance_schedule: str = "minimal") -> Dict[str, Any]:
        """Calculate use phase emissions and carbonation."""
        
        # Calculate carbonation sequestration
        cement_content = material_inventory["materials"].get("cement", {}).get("amount_kg", 0)
        clinker_content = material_inventory["materials"].get("clinker", {}).get("amount_kg", 0)
        
        # Carbonation can sequester up to 20% of process emissions
        carbonation_rates = {
            "slow": 0.1,
            "standard": 0.15,
            "fast": 0.2
        }
        rate = carbonation_rates.get(carbonation_rate, 0.15)
        
        # Calculate CO2 sequestration
        max_sequestration = clinker_content * 0.525 * rate  # Based on CaO content
        annual_sequestration = max_sequestration / service_life_years
        
        # Maintenance emissions
        maintenance_emissions = self._calculate_maintenance_emissions(
            material_inventory, service_life_years, maintenance_schedule
        )
        
        return {
            "use_phase_emissions": {
                "total_kg_co2": round(maintenance_emissions - max_sequestration, 2),
                "maintenance_emissions": round(maintenance_emissions, 2),
                "carbonation_credit": round(-max_sequestration, 2),
                "net_annual_emissions": round(
                    (maintenance_emissions / service_life_years) - annual_sequestration, 2
                )
            },
            "carbonation_sequestration": {
                "total_sequestered_kg_co2": round(max_sequestration, 2),
                "annual_rate_kg_co2": round(annual_sequestration, 2),
                "sequestration_rate": rate
            }
        }
    
    def calculate_eol_emissions(self,
                               material_inventory: Dict[str, Any],
                               disposal_method: str = "landfill",
                               recycling_rate: float = 0.3,
                               crushing_energy: str = "standard") -> Dict[str, Any]:
        """Calculate end-of-life emissions."""
        
        total_mass = material_inventory["total_mass_kg"]
        recycled_mass = total_mass * recycling_rate
        landfilled_mass = total_mass * (1 - recycling_rate)
        
        # Crushing energy for recycling
        crushing_energy_kwh = recycled_mass * 0.01  # 10 kWh/tonne
        crushing_emissions = crushing_energy_kwh * self.emission_factors["electricity"]
        
        # Transport to disposal
        disposal_transport = total_mass * 30 * self.emission_factors["truck_transport"]
        
        # Recycling credits (avoided production)
        recycling_credits = recycled_mass * 0.05  # 50 kg CO2/tonne credit
        
        total_eol_emissions = crushing_emissions + disposal_transport - recycling_credits
        
        return {
            "eol_emissions": {
                "total_kg_co2": round(total_eol_emissions, 2),
                "crushing_emissions": round(crushing_emissions, 2),
                "transport_emissions": round(disposal_transport, 2),
                "disposal_method": disposal_method,
                "recycled_mass_kg": round(recycled_mass, 2),
                "landfilled_mass_kg": round(landfilled_mass, 2)
            },
            "recycling_credits": {
                "total_credits_kg_co2": round(-recycling_credits, 2),
                "recycling_rate": recycling_rate,
                "avoided_production": round(recycling_credits, 2)
            }
        }
    
    def _calculate_energy_use(self, inventory, method, efficiency):
        """Calculate energy consumption for production."""
        clinker = inventory["materials"].get("clinker", {}).get("amount_kg", 0)
        
        # Energy requirements (per kg clinker)
        if method == "dry_process":
            thermal_energy = 3.2 / efficiency  # MJ/kg
            electrical_energy = 0.095 / efficiency  # kWh/kg
        else:  # wet_process
            thermal_energy = 5.0 / efficiency
            electrical_energy = 0.110 / efficiency
        
        return {
            "thermal": round(clinker * thermal_energy, 2),
            "electricity": round(clinker * electrical_energy, 2)
        }
    
    def _estimate_fuel_consumption(self, mass_kg, distance_km, mode):
        """Estimate fuel consumption for transport."""
        # Fuel consumption factors (liters per tonne-km)
        factors = {
            "truck": 0.035,
            "rail": 0.01,
            "ship": 0.004
        }
        
        fuel_per_tkm = factors.get(mode, 0.035)
        total_fuel = (mass_kg / 1000) * distance_km * fuel_per_tkm
        
        return {
            "diesel_liters": round(total_fuel, 2),
            "energy_mj": round(total_fuel * 35.8, 2)  # Energy content of diesel
        }
    
    def _calculate_maintenance_emissions(self, inventory, service_life, schedule):
        """Calculate emissions from maintenance activities."""
        # Maintenance intensity factors
        maintenance_factors = {
            "minimal": 0.001,  # 0.1% of initial emissions per year
            "standard": 0.005,
            "intensive": 0.01
        }
        
        factor = maintenance_factors.get(schedule, 0.005)
        initial_emissions = inventory["total_mass_kg"] * 0.3  # Rough estimate
        
        return initial_emissions * factor * service_life

# Export agent instance
agent = EmissionsCalculatorAgent()