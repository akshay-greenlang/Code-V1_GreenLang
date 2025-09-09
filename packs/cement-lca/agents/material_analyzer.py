"""Material Analyzer Agent for cement and concrete composition analysis."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class MaterialAnalyzerAgent:
    """Agent for analyzing cement and concrete material compositions."""
    
    def __init__(self):
        self.name = "material_analyzer"
        self.version = "1.0.0"
        
        # Standard cement compositions (% by mass)
        self.cement_types = {
            "OPC": {  # Ordinary Portland Cement (CEM I)
                "clinker": 95,
                "gypsum": 5,
                "scm": 0
            },
            "PPC": {  # Portland Pozzolana Cement (CEM II)
                "clinker": 70,
                "gypsum": 5,
                "fly_ash": 25
            },
            "PSC": {  # Portland Slag Cement
                "clinker": 45,
                "gypsum": 5,
                "ggbs": 50
            },
            "LC3": {  # Limestone Calcined Clay Cement
                "clinker": 50,
                "gypsum": 5,
                "limestone": 15,
                "calcined_clay": 30
            }
        }
        
        # Standard mix designs (kg/m³)
        self.mix_designs = {
            "standard": {
                "cement": 350,
                "water": 175,
                "fine_aggregate": 750,
                "coarse_aggregate": 1100,
                "admixtures": 3.5
            },
            "high_strength": {
                "cement": 450,
                "water": 160,
                "fine_aggregate": 700,
                "coarse_aggregate": 1050,
                "admixtures": 9.0,
                "silica_fume": 45
            },
            "eco_friendly": {
                "cement": 250,
                "water": 165,
                "fine_aggregate": 800,
                "coarse_aggregate": 1150,
                "fly_ash": 100,
                "admixtures": 2.5
            }
        }
    
    def analyze_composition(self,
                           cement_type: str = "OPC",
                           strength_class: str = "42.5",
                           volume_m3: float = 1000,
                           mix_design: str = "standard") -> Dict[str, Any]:
        """Analyze cement and concrete composition for LCA."""
        
        # Get cement composition
        cement_comp = self.cement_types.get(cement_type, self.cement_types["OPC"])
        
        # Get mix design
        mix = self.mix_designs.get(mix_design, self.mix_designs["standard"])
        
        # Calculate total materials
        material_inventory = self._calculate_inventory(
            cement_comp, mix, volume_m3
        )
        
        # Generate composition breakdown
        composition_breakdown = self._generate_breakdown(
            cement_comp, mix, material_inventory
        )
        
        return {
            "material_inventory": material_inventory,
            "composition_breakdown": composition_breakdown
        }
    
    def _calculate_inventory(self, cement_comp, mix, volume_m3):
        """Calculate total material inventory."""
        inventory = {
            "id": f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "volume_m3": volume_m3,
            "materials": {}
        }
        
        # Calculate cement components
        cement_total_kg = mix["cement"] * volume_m3
        for component, percentage in cement_comp.items():
            amount = cement_total_kg * percentage / 100
            if amount > 0:
                inventory["materials"][component] = {
                    "amount_kg": round(amount, 2),
                    "unit": "kg",
                    "source": "cement_production"
                }
        
        # Add other concrete components
        for material, amount_per_m3 in mix.items():
            if material != "cement" and amount_per_m3 > 0:
                inventory["materials"][material] = {
                    "amount_kg": round(amount_per_m3 * volume_m3, 2),
                    "unit": "kg",
                    "source": "local_supplier"
                }
        
        # Calculate totals
        inventory["total_mass_kg"] = sum(
            m["amount_kg"] for m in inventory["materials"].values()
        )
        inventory["density_kg_m3"] = inventory["total_mass_kg"] / volume_m3
        
        return inventory
    
    def _generate_breakdown(self, cement_comp, mix, inventory):
        """Generate detailed composition breakdown."""
        breakdown = {
            "cement_type": cement_comp,
            "mix_proportions": mix,
            "performance_class": self._estimate_strength_class(mix),
            "sustainability_metrics": self._calculate_sustainability_metrics(
                cement_comp, mix, inventory
            ),
            "material_sources": self._identify_sources(inventory)
        }
        
        return breakdown
    
    def _estimate_strength_class(self, mix):
        """Estimate concrete strength class from mix design."""
        w_c_ratio = mix["water"] / mix["cement"]
        
        if w_c_ratio < 0.4:
            return "C50/60"
        elif w_c_ratio < 0.45:
            return "C40/50"
        elif w_c_ratio < 0.5:
            return "C30/37"
        elif w_c_ratio < 0.55:
            return "C25/30"
        else:
            return "C20/25"
    
    def _calculate_sustainability_metrics(self, cement_comp, mix, inventory):
        """Calculate sustainability metrics."""
        total_cement = mix["cement"]
        clinker_content = cement_comp.get("clinker", 95) / 100
        
        return {
            "clinker_factor": round(clinker_content, 3),
            "cement_content_kg_m3": total_cement,
            "scm_content_percent": round(100 - cement_comp.get("clinker", 95), 1),
            "recycled_content_percent": self._estimate_recycled_content(mix),
            "water_cement_ratio": round(mix["water"] / mix["cement"], 3)
        }
    
    def _estimate_recycled_content(self, mix):
        """Estimate recycled content percentage."""
        recycled_materials = ["fly_ash", "ggbs", "silica_fume"]
        total_mass = sum(mix.values())
        recycled_mass = sum(mix.get(m, 0) for m in recycled_materials)
        return round(100 * recycled_mass / total_mass, 1)
    
    def _identify_sources(self, inventory):
        """Identify material sources and transport requirements."""
        sources = []
        
        for material, data in inventory["materials"].items():
            source = {
                "material": material,
                "supplier": data["source"],
                "transport_distance_km": self._estimate_transport_distance(material),
                "transport_mode": self._determine_transport_mode(material, data["amount_kg"])
            }
            sources.append(source)
        
        return sources
    
    def _estimate_transport_distance(self, material):
        """Estimate typical transport distances."""
        distances = {
            "clinker": 200,
            "gypsum": 150,
            "limestone": 50,
            "fly_ash": 100,
            "ggbs": 150,
            "water": 5,
            "fine_aggregate": 30,
            "coarse_aggregate": 40,
            "admixtures": 200,
            "silica_fume": 300,
            "calcined_clay": 100
        }
        return distances.get(material, 50)
    
    def _determine_transport_mode(self, material, amount_kg):
        """Determine appropriate transport mode."""
        if amount_kg > 100000:
            return "rail"
        elif amount_kg > 10000:
            return "truck"
        else:
            return "light_truck"

# Export agent instance
agent = MaterialAnalyzerAgent()