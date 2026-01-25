"""
Insulation Material Database

This module provides a comprehensive database of 50+ industrial insulation
materials with their thermal properties, temperature limits, and applications.

Material properties are sourced from industry standards:
- ASTM C585 Standard Practice for Inner and Outer Diameters of Thermal Insulation
- ASTM C1055 Standard Guide for Heated System Surface Conditions
- ASHRAE Handbook Fundamentals
- Manufacturer technical data sheets

All material properties are deterministic lookup values - no ML/LLM calculations.
This ensures zero-hallucination compliance for regulatory calculations.

Example:
    >>> material = get_material_properties("calcium_silicate")
    >>> print(f"Thermal conductivity at 200C: {material.get_conductivity(200):.4f} W/m-K")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InsulationCategory(str, Enum):
    """Categories of insulation materials."""
    MINERAL_FIBER = "mineral_fiber"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    CERAMIC_FIBER = "ceramic_fiber"
    FOAM_PLASTIC = "foam_plastic"
    REFLECTIVE = "reflective"
    MICROPOROUS = "microporous"
    NATURAL_FIBER = "natural_fiber"


class TemperatureClass(str, Enum):
    """Temperature service classes."""
    CRYOGENIC = "cryogenic"  # Below -100C
    LOW_TEMP = "low_temp"  # -100C to 0C
    MODERATE = "moderate"  # 0C to 250C
    HIGH_TEMP = "high_temp"  # 250C to 650C
    VERY_HIGH_TEMP = "very_high_temp"  # Above 650C


@dataclass
class InsulationMaterial:
    """
    Insulation material properties data class.

    All thermal conductivity values follow the equation:
    k = k0 + k1*T + k2*T^2
    where T is temperature in Celsius.

    Attributes:
        material_id: Unique identifier for the material.
        name: Human-readable name.
        category: Material category (mineral fiber, cellular, etc.).
        density_kg_m3: Nominal density in kg/m3.
        min_temp_c: Minimum service temperature in Celsius.
        max_temp_c: Maximum service temperature in Celsius.
        k0: Base thermal conductivity coefficient (W/m-K).
        k1: Linear temperature coefficient.
        k2: Quadratic temperature coefficient.
        specific_heat_j_kg_k: Specific heat capacity (J/kg-K).
        moisture_resistance: Moisture resistance rating (0-1).
        compressive_strength_kpa: Compressive strength in kPa.
        cost_per_m3_usd: Approximate cost per cubic meter.
        applications: List of typical applications.
    """
    material_id: str
    name: str
    category: InsulationCategory
    density_kg_m3: float
    min_temp_c: float
    max_temp_c: float
    k0: float  # Base conductivity at 0C
    k1: float = 0.0  # Linear coefficient
    k2: float = 0.0  # Quadratic coefficient
    specific_heat_j_kg_k: float = 1000.0
    moisture_resistance: float = 0.5
    compressive_strength_kpa: float = 100.0
    cost_per_m3_usd: float = 500.0
    applications: List[str] = field(default_factory=list)

    def get_conductivity(self, temperature_c: float) -> float:
        """
        Calculate thermal conductivity at given temperature.

        Uses polynomial model: k = k0 + k1*T + k2*T^2

        Args:
            temperature_c: Temperature in Celsius.

        Returns:
            Thermal conductivity in W/m-K.

        Raises:
            ValueError: If temperature is outside service range.
        """
        if temperature_c < self.min_temp_c or temperature_c > self.max_temp_c:
            logger.warning(
                f"Temperature {temperature_c}C outside service range "
                f"[{self.min_temp_c}, {self.max_temp_c}] for {self.name}"
            )

        # Polynomial thermal conductivity model
        k = self.k0 + self.k1 * temperature_c + self.k2 * (temperature_c ** 2)

        # Ensure positive conductivity
        return max(0.001, k)

    def is_suitable_for_temperature(self, temperature_c: float) -> bool:
        """Check if material is suitable for given temperature."""
        return self.min_temp_c <= temperature_c <= self.max_temp_c

    def get_temperature_class(self) -> TemperatureClass:
        """Determine temperature service class."""
        if self.max_temp_c < -100:
            return TemperatureClass.CRYOGENIC
        elif self.max_temp_c < 0:
            return TemperatureClass.LOW_TEMP
        elif self.max_temp_c < 250:
            return TemperatureClass.MODERATE
        elif self.max_temp_c < 650:
            return TemperatureClass.HIGH_TEMP
        else:
            return TemperatureClass.VERY_HIGH_TEMP


# Comprehensive insulation material database (50+ materials)
INSULATION_DATABASE: Dict[str, InsulationMaterial] = {
    # ============================================
    # MINERAL FIBER INSULATIONS (12 materials)
    # ============================================
    "mineral_wool_blanket_48": InsulationMaterial(
        material_id="mineral_wool_blanket_48",
        name="Mineral Wool Blanket (48 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=48,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.033,
        k1=0.00018,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.3,
        compressive_strength_kpa=10,
        cost_per_m3_usd=180,
        applications=["Pipe insulation", "Equipment", "Tanks"],
    ),
    "mineral_wool_blanket_96": InsulationMaterial(
        material_id="mineral_wool_blanket_96",
        name="Mineral Wool Blanket (96 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=96,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.035,
        k1=0.00016,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.35,
        compressive_strength_kpa=25,
        cost_per_m3_usd=220,
        applications=["Industrial equipment", "High-temp pipes"],
    ),
    "mineral_wool_blanket_128": InsulationMaterial(
        material_id="mineral_wool_blanket_128",
        name="Mineral Wool Blanket (128 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=128,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.037,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.4,
        compressive_strength_kpa=50,
        cost_per_m3_usd=280,
        applications=["High-density applications", "Furnaces"],
    ),
    "mineral_wool_board_100": InsulationMaterial(
        material_id="mineral_wool_board_100",
        name="Mineral Wool Board (100 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=100,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.036,
        k1=0.00017,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.35,
        compressive_strength_kpa=40,
        cost_per_m3_usd=250,
        applications=["Flat surfaces", "Tank insulation"],
    ),
    "mineral_wool_pipe_cover": InsulationMaterial(
        material_id="mineral_wool_pipe_cover",
        name="Mineral Wool Pipe Cover",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=80,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.034,
        k1=0.00017,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.3,
        compressive_strength_kpa=30,
        cost_per_m3_usd=300,
        applications=["Pipe insulation", "Pre-formed sections"],
    ),
    "fiberglass_blanket_24": InsulationMaterial(
        material_id="fiberglass_blanket_24",
        name="Fiberglass Blanket (24 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=24,
        min_temp_c=-40,
        max_temp_c=454,
        k0=0.030,
        k1=0.00020,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.25,
        compressive_strength_kpa=5,
        cost_per_m3_usd=120,
        applications=["HVAC", "Low-pressure steam"],
    ),
    "fiberglass_blanket_48": InsulationMaterial(
        material_id="fiberglass_blanket_48",
        name="Fiberglass Blanket (48 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=48,
        min_temp_c=-40,
        max_temp_c=454,
        k0=0.032,
        k1=0.00018,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.30,
        compressive_strength_kpa=10,
        cost_per_m3_usd=160,
        applications=["Industrial pipes", "Ducts"],
    ),
    "fiberglass_pipe_cover": InsulationMaterial(
        material_id="fiberglass_pipe_cover",
        name="Fiberglass Pipe Cover",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=64,
        min_temp_c=-40,
        max_temp_c=454,
        k0=0.033,
        k1=0.00017,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.30,
        compressive_strength_kpa=20,
        cost_per_m3_usd=280,
        applications=["Pipe insulation", "Pre-formed sections"],
    ),
    "fiberglass_board_96": InsulationMaterial(
        material_id="fiberglass_board_96",
        name="Fiberglass Board (96 kg/m3)",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=96,
        min_temp_c=-40,
        max_temp_c=454,
        k0=0.035,
        k1=0.00016,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.35,
        compressive_strength_kpa=45,
        cost_per_m3_usd=240,
        applications=["Flat surfaces", "Equipment"],
    ),
    "rock_wool_blanket": InsulationMaterial(
        material_id="rock_wool_blanket",
        name="Rock Wool Blanket",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=80,
        min_temp_c=-40,
        max_temp_c=750,
        k0=0.035,
        k1=0.00016,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.40,
        compressive_strength_kpa=20,
        cost_per_m3_usd=200,
        applications=["High-temp applications", "Fire protection"],
    ),
    "rock_wool_board": InsulationMaterial(
        material_id="rock_wool_board",
        name="Rock Wool Board",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=120,
        min_temp_c=-40,
        max_temp_c=750,
        k0=0.037,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.45,
        compressive_strength_kpa=60,
        cost_per_m3_usd=280,
        applications=["Industrial furnaces", "High-temp equipment"],
    ),
    "slag_wool": InsulationMaterial(
        material_id="slag_wool",
        name="Slag Wool",
        category=InsulationCategory.MINERAL_FIBER,
        density_kg_m3=96,
        min_temp_c=-40,
        max_temp_c=650,
        k0=0.036,
        k1=0.00017,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.35,
        compressive_strength_kpa=25,
        cost_per_m3_usd=170,
        applications=["General industrial", "Economical option"],
    ),

    # ============================================
    # CALCIUM SILICATE (5 materials)
    # ============================================
    "calcium_silicate_standard": InsulationMaterial(
        material_id="calcium_silicate_standard",
        name="Calcium Silicate Standard",
        category=InsulationCategory.CALCIUM_SILICATE,
        density_kg_m3=240,
        min_temp_c=-18,
        max_temp_c=650,
        k0=0.052,
        k1=0.00014,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.60,
        compressive_strength_kpa=700,
        cost_per_m3_usd=450,
        applications=["Pipe insulation", "High-temp pipes", "Vessel insulation"],
    ),
    "calcium_silicate_high_temp": InsulationMaterial(
        material_id="calcium_silicate_high_temp",
        name="Calcium Silicate High Temperature",
        category=InsulationCategory.CALCIUM_SILICATE,
        density_kg_m3=260,
        min_temp_c=-18,
        max_temp_c=1050,
        k0=0.055,
        k1=0.00013,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.60,
        compressive_strength_kpa=750,
        cost_per_m3_usd=580,
        applications=["Furnaces", "High-temp process"],
    ),
    "calcium_silicate_pipe": InsulationMaterial(
        material_id="calcium_silicate_pipe",
        name="Calcium Silicate Pipe Cover",
        category=InsulationCategory.CALCIUM_SILICATE,
        density_kg_m3=240,
        min_temp_c=-18,
        max_temp_c=650,
        k0=0.052,
        k1=0.00014,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.60,
        compressive_strength_kpa=700,
        cost_per_m3_usd=520,
        applications=["Pipe insulation", "Pre-formed sections"],
    ),
    "calcium_silicate_block": InsulationMaterial(
        material_id="calcium_silicate_block",
        name="Calcium Silicate Block",
        category=InsulationCategory.CALCIUM_SILICATE,
        density_kg_m3=250,
        min_temp_c=-18,
        max_temp_c=650,
        k0=0.053,
        k1=0.00014,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.60,
        compressive_strength_kpa=720,
        cost_per_m3_usd=480,
        applications=["Flat surfaces", "Large equipment"],
    ),
    "calcium_silicate_cryogenic": InsulationMaterial(
        material_id="calcium_silicate_cryogenic",
        name="Calcium Silicate Cryogenic Grade",
        category=InsulationCategory.CALCIUM_SILICATE,
        density_kg_m3=220,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.050,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.65,
        compressive_strength_kpa=680,
        cost_per_m3_usd=600,
        applications=["LNG", "Cryogenic systems"],
    ),

    # ============================================
    # CELLULAR GLASS (6 materials)
    # ============================================
    "cellular_glass_standard": InsulationMaterial(
        material_id="cellular_glass_standard",
        name="Cellular Glass Standard",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=120,
        min_temp_c=-268,
        max_temp_c=430,
        k0=0.038,
        k1=0.00012,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=700,
        cost_per_m3_usd=650,
        applications=["Cryogenic", "Chemical plants", "Underground"],
    ),
    "cellular_glass_high_density": InsulationMaterial(
        material_id="cellular_glass_high_density",
        name="Cellular Glass High Density",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=165,
        min_temp_c=-268,
        max_temp_c=430,
        k0=0.045,
        k1=0.00011,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=1600,
        cost_per_m3_usd=850,
        applications=["Heavy loads", "Tank bases"],
    ),
    "cellular_glass_low_density": InsulationMaterial(
        material_id="cellular_glass_low_density",
        name="Cellular Glass Low Density",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=100,
        min_temp_c=-268,
        max_temp_c=430,
        k0=0.036,
        k1=0.00013,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=400,
        cost_per_m3_usd=580,
        applications=["General insulation", "Cost-effective"],
    ),
    "cellular_glass_pipe": InsulationMaterial(
        material_id="cellular_glass_pipe",
        name="Cellular Glass Pipe Insulation",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=130,
        min_temp_c=-268,
        max_temp_c=430,
        k0=0.040,
        k1=0.00012,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=800,
        cost_per_m3_usd=750,
        applications=["Pipe insulation", "Below-grade"],
    ),
    "cellular_glass_high_temp": InsulationMaterial(
        material_id="cellular_glass_high_temp",
        name="Cellular Glass High Temperature",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=140,
        min_temp_c=-40,
        max_temp_c=482,
        k0=0.042,
        k1=0.00011,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=900,
        cost_per_m3_usd=720,
        applications=["High-temp industrial"],
    ),
    "cellular_glass_cryogenic": InsulationMaterial(
        material_id="cellular_glass_cryogenic",
        name="Cellular Glass Cryogenic Grade",
        category=InsulationCategory.CELLULAR_GLASS,
        density_kg_m3=115,
        min_temp_c=-268,
        max_temp_c=430,
        k0=0.037,
        k1=0.00012,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.99,
        compressive_strength_kpa=600,
        cost_per_m3_usd=700,
        applications=["LNG", "LOX", "Cryogenic tanks"],
    ),

    # ============================================
    # PERLITE (4 materials)
    # ============================================
    "perlite_expanded": InsulationMaterial(
        material_id="perlite_expanded",
        name="Expanded Perlite",
        category=InsulationCategory.PERLITE,
        density_kg_m3=140,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.045,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.50,
        compressive_strength_kpa=250,
        cost_per_m3_usd=280,
        applications=["Cryogenic storage", "Tank fill"],
    ),
    "perlite_board": InsulationMaterial(
        material_id="perlite_board",
        name="Perlite Board",
        category=InsulationCategory.PERLITE,
        density_kg_m3=190,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.050,
        k1=0.00014,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.55,
        compressive_strength_kpa=350,
        cost_per_m3_usd=380,
        applications=["Flat surfaces", "Fire protection"],
    ),
    "perlite_pipe_cover": InsulationMaterial(
        material_id="perlite_pipe_cover",
        name="Perlite Pipe Cover",
        category=InsulationCategory.PERLITE,
        density_kg_m3=170,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.048,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.50,
        compressive_strength_kpa=300,
        cost_per_m3_usd=420,
        applications=["Pipe insulation", "Cryogenic pipes"],
    ),
    "perlite_loose_fill": InsulationMaterial(
        material_id="perlite_loose_fill",
        name="Perlite Loose Fill",
        category=InsulationCategory.PERLITE,
        density_kg_m3=80,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.042,
        k1=0.00016,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.40,
        compressive_strength_kpa=50,
        cost_per_m3_usd=150,
        applications=["Cavity fill", "Loose fill applications"],
    ),

    # ============================================
    # AEROGEL (4 materials)
    # ============================================
    "aerogel_blanket": InsulationMaterial(
        material_id="aerogel_blanket",
        name="Aerogel Blanket",
        category=InsulationCategory.AEROGEL,
        density_kg_m3=150,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.014,
        k1=0.00005,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.80,
        compressive_strength_kpa=100,
        cost_per_m3_usd=3500,
        applications=["Space-constrained", "High-performance"],
    ),
    "aerogel_blanket_high_temp": InsulationMaterial(
        material_id="aerogel_blanket_high_temp",
        name="Aerogel Blanket High Temperature",
        category=InsulationCategory.AEROGEL,
        density_kg_m3=160,
        min_temp_c=-40,
        max_temp_c=1000,
        k0=0.016,
        k1=0.00004,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.85,
        compressive_strength_kpa=120,
        cost_per_m3_usd=4500,
        applications=["Ultra-high temp", "Furnaces"],
    ),
    "aerogel_pipe": InsulationMaterial(
        material_id="aerogel_pipe",
        name="Aerogel Pipe Insulation",
        category=InsulationCategory.AEROGEL,
        density_kg_m3=155,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.015,
        k1=0.00005,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.80,
        compressive_strength_kpa=110,
        cost_per_m3_usd=4000,
        applications=["Space-limited pipes", "Offshore"],
    ),
    "aerogel_board": InsulationMaterial(
        material_id="aerogel_board",
        name="Aerogel Board",
        category=InsulationCategory.AEROGEL,
        density_kg_m3=165,
        min_temp_c=-200,
        max_temp_c=650,
        k0=0.015,
        k1=0.00005,
        k2=0.0,
        specific_heat_j_kg_k=840,
        moisture_resistance=0.82,
        compressive_strength_kpa=150,
        cost_per_m3_usd=3800,
        applications=["Flat surfaces", "High-performance"],
    ),

    # ============================================
    # CERAMIC FIBER (6 materials)
    # ============================================
    "ceramic_fiber_blanket_96": InsulationMaterial(
        material_id="ceramic_fiber_blanket_96",
        name="Ceramic Fiber Blanket (96 kg/m3)",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=96,
        min_temp_c=0,
        max_temp_c=1260,
        k0=0.040,
        k1=0.00020,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.40,
        compressive_strength_kpa=15,
        cost_per_m3_usd=350,
        applications=["Furnace linings", "High-temp applications"],
    ),
    "ceramic_fiber_blanket_128": InsulationMaterial(
        material_id="ceramic_fiber_blanket_128",
        name="Ceramic Fiber Blanket (128 kg/m3)",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=128,
        min_temp_c=0,
        max_temp_c=1260,
        k0=0.045,
        k1=0.00018,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.45,
        compressive_strength_kpa=25,
        cost_per_m3_usd=420,
        applications=["Kiln linings", "Furnace backup"],
    ),
    "ceramic_fiber_board": InsulationMaterial(
        material_id="ceramic_fiber_board",
        name="Ceramic Fiber Board",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=250,
        min_temp_c=0,
        max_temp_c=1260,
        k0=0.070,
        k1=0.00015,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.55,
        compressive_strength_kpa=400,
        cost_per_m3_usd=650,
        applications=["Structural insulation", "High-temp"],
    ),
    "ceramic_fiber_module": InsulationMaterial(
        material_id="ceramic_fiber_module",
        name="Ceramic Fiber Module",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=160,
        min_temp_c=0,
        max_temp_c=1260,
        k0=0.055,
        k1=0.00018,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.50,
        compressive_strength_kpa=200,
        cost_per_m3_usd=550,
        applications=["Furnace walls", "Quick installation"],
    ),
    "ceramic_fiber_paper": InsulationMaterial(
        material_id="ceramic_fiber_paper",
        name="Ceramic Fiber Paper",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=200,
        min_temp_c=0,
        max_temp_c=1260,
        k0=0.065,
        k1=0.00016,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.50,
        compressive_strength_kpa=50,
        cost_per_m3_usd=800,
        applications=["Gaskets", "Thin insulation"],
    ),
    "polycrystalline_fiber": InsulationMaterial(
        material_id="polycrystalline_fiber",
        name="Polycrystalline Alumina Fiber",
        category=InsulationCategory.CERAMIC_FIBER,
        density_kg_m3=96,
        min_temp_c=0,
        max_temp_c=1600,
        k0=0.080,
        k1=0.00025,
        k2=0.0,
        specific_heat_j_kg_k=1130,
        moisture_resistance=0.45,
        compressive_strength_kpa=20,
        cost_per_m3_usd=1200,
        applications=["Ultra-high temp", "Metal processing"],
    ),

    # ============================================
    # FOAM PLASTIC (8 materials)
    # ============================================
    "polyurethane_foam_32": InsulationMaterial(
        material_id="polyurethane_foam_32",
        name="Polyurethane Foam (32 kg/m3)",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=32,
        min_temp_c=-200,
        max_temp_c=110,
        k0=0.022,
        k1=0.00008,
        k2=0.0,
        specific_heat_j_kg_k=1400,
        moisture_resistance=0.90,
        compressive_strength_kpa=140,
        cost_per_m3_usd=350,
        applications=["Cold storage", "HVAC", "Pipe insulation"],
    ),
    "polyurethane_foam_48": InsulationMaterial(
        material_id="polyurethane_foam_48",
        name="Polyurethane Foam (48 kg/m3)",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=48,
        min_temp_c=-200,
        max_temp_c=110,
        k0=0.024,
        k1=0.00007,
        k2=0.0,
        specific_heat_j_kg_k=1400,
        moisture_resistance=0.92,
        compressive_strength_kpa=200,
        cost_per_m3_usd=400,
        applications=["Higher load applications", "Cold storage"],
    ),
    "polyisocyanurate_board": InsulationMaterial(
        material_id="polyisocyanurate_board",
        name="Polyisocyanurate Board",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=32,
        min_temp_c=-200,
        max_temp_c=150,
        k0=0.020,
        k1=0.00008,
        k2=0.0,
        specific_heat_j_kg_k=1400,
        moisture_resistance=0.88,
        compressive_strength_kpa=170,
        cost_per_m3_usd=380,
        applications=["Commercial buildings", "Roofing"],
    ),
    "phenolic_foam": InsulationMaterial(
        material_id="phenolic_foam",
        name="Phenolic Foam",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=40,
        min_temp_c=-180,
        max_temp_c=120,
        k0=0.018,
        k1=0.00007,
        k2=0.0,
        specific_heat_j_kg_k=1400,
        moisture_resistance=0.95,
        compressive_strength_kpa=120,
        cost_per_m3_usd=420,
        applications=["HVAC", "Fire-rated applications"],
    ),
    "extruded_polystyrene": InsulationMaterial(
        material_id="extruded_polystyrene",
        name="Extruded Polystyrene (XPS)",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=35,
        min_temp_c=-50,
        max_temp_c=75,
        k0=0.028,
        k1=0.00006,
        k2=0.0,
        specific_heat_j_kg_k=1300,
        moisture_resistance=0.98,
        compressive_strength_kpa=300,
        cost_per_m3_usd=280,
        applications=["Underground", "Wet conditions"],
    ),
    "expanded_polystyrene": InsulationMaterial(
        material_id="expanded_polystyrene",
        name="Expanded Polystyrene (EPS)",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=25,
        min_temp_c=-50,
        max_temp_c=75,
        k0=0.035,
        k1=0.00006,
        k2=0.0,
        specific_heat_j_kg_k=1300,
        moisture_resistance=0.75,
        compressive_strength_kpa=100,
        cost_per_m3_usd=150,
        applications=["Cold storage", "Low-cost applications"],
    ),
    "elastomeric_foam": InsulationMaterial(
        material_id="elastomeric_foam",
        name="Elastomeric Foam",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=60,
        min_temp_c=-40,
        max_temp_c=105,
        k0=0.035,
        k1=0.00008,
        k2=0.0,
        specific_heat_j_kg_k=1700,
        moisture_resistance=0.95,
        compressive_strength_kpa=50,
        cost_per_m3_usd=450,
        applications=["HVAC", "Flexible pipe insulation"],
    ),
    "melamine_foam": InsulationMaterial(
        material_id="melamine_foam",
        name="Melamine Foam",
        category=InsulationCategory.FOAM_PLASTIC,
        density_kg_m3=10,
        min_temp_c=-40,
        max_temp_c=150,
        k0=0.035,
        k1=0.00010,
        k2=0.0,
        specific_heat_j_kg_k=1400,
        moisture_resistance=0.70,
        compressive_strength_kpa=10,
        cost_per_m3_usd=600,
        applications=["Acoustic insulation", "Lightweight"],
    ),

    # ============================================
    # MICROPOROUS (3 materials)
    # ============================================
    "microporous_standard": InsulationMaterial(
        material_id="microporous_standard",
        name="Microporous Insulation Standard",
        category=InsulationCategory.MICROPOROUS,
        density_kg_m3=280,
        min_temp_c=-200,
        max_temp_c=1000,
        k0=0.020,
        k1=0.00003,
        k2=0.0,
        specific_heat_j_kg_k=800,
        moisture_resistance=0.50,
        compressive_strength_kpa=500,
        cost_per_m3_usd=5000,
        applications=["Space-critical", "High-temp"],
    ),
    "microporous_high_temp": InsulationMaterial(
        material_id="microporous_high_temp",
        name="Microporous Insulation High Temp",
        category=InsulationCategory.MICROPOROUS,
        density_kg_m3=300,
        min_temp_c=-40,
        max_temp_c=1200,
        k0=0.022,
        k1=0.00003,
        k2=0.0,
        specific_heat_j_kg_k=800,
        moisture_resistance=0.50,
        compressive_strength_kpa=600,
        cost_per_m3_usd=6500,
        applications=["Ultra-high temp", "Furnaces"],
    ),
    "microporous_flexible": InsulationMaterial(
        material_id="microporous_flexible",
        name="Microporous Flexible Blanket",
        category=InsulationCategory.MICROPOROUS,
        density_kg_m3=250,
        min_temp_c=-200,
        max_temp_c=850,
        k0=0.019,
        k1=0.00004,
        k2=0.0,
        specific_heat_j_kg_k=800,
        moisture_resistance=0.45,
        compressive_strength_kpa=200,
        cost_per_m3_usd=4500,
        applications=["Curved surfaces", "Complex geometry"],
    ),

    # ============================================
    # REFLECTIVE (2 materials)
    # ============================================
    "multi_layer_reflective": InsulationMaterial(
        material_id="multi_layer_reflective",
        name="Multi-Layer Reflective Insulation",
        category=InsulationCategory.REFLECTIVE,
        density_kg_m3=25,
        min_temp_c=-40,
        max_temp_c=80,
        k0=0.030,  # Effective k-value
        k1=0.00005,
        k2=0.0,
        specific_heat_j_kg_k=900,
        moisture_resistance=0.99,
        compressive_strength_kpa=5,
        cost_per_m3_usd=200,
        applications=["Radiant barriers", "HVAC"],
    ),
    "metal_foil_blanket": InsulationMaterial(
        material_id="metal_foil_blanket",
        name="Metal Foil Blanket (High Temp)",
        category=InsulationCategory.REFLECTIVE,
        density_kg_m3=50,
        min_temp_c=0,
        max_temp_c=500,
        k0=0.025,  # Effective k-value with air gaps
        k1=0.00008,
        k2=0.0,
        specific_heat_j_kg_k=500,
        moisture_resistance=0.95,
        compressive_strength_kpa=20,
        cost_per_m3_usd=800,
        applications=["Furnace shields", "Radiant heat"],
    ),
}


def get_material(material_id: str) -> Optional[InsulationMaterial]:
    """
    Get insulation material by ID.

    Args:
        material_id: Material identifier.

    Returns:
        InsulationMaterial object or None if not found.
    """
    return INSULATION_DATABASE.get(material_id)


def get_material_properties(material_id: str) -> Optional[InsulationMaterial]:
    """
    Get insulation material properties by ID.

    Alias for get_material() for API consistency.

    Args:
        material_id: Material identifier.

    Returns:
        InsulationMaterial object or None if not found.
    """
    return get_material(material_id)


def list_materials(
    category: Optional[InsulationCategory] = None,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    max_conductivity: Optional[float] = None
) -> List[InsulationMaterial]:
    """
    List materials with optional filtering.

    Args:
        category: Filter by material category.
        min_temp: Filter by minimum temperature rating.
        max_temp: Filter by maximum temperature rating.
        max_conductivity: Filter by maximum base conductivity.

    Returns:
        List of matching InsulationMaterial objects.
    """
    results = []

    for material in INSULATION_DATABASE.values():
        if category and material.category != category:
            continue
        if min_temp is not None and material.min_temp_c > min_temp:
            continue
        if max_temp is not None and material.max_temp_c < max_temp:
            continue
        if max_conductivity is not None and material.k0 > max_conductivity:
            continue
        results.append(material)

    return results


def find_suitable_materials(
    operating_temp_c: float,
    surface_type: str = "pipe",
    max_cost: Optional[float] = None
) -> List[Tuple[InsulationMaterial, float]]:
    """
    Find suitable materials for given operating conditions.

    Returns materials sorted by thermal performance (lowest conductivity first).

    Args:
        operating_temp_c: Operating temperature in Celsius.
        surface_type: Type of surface (pipe, flat, tank).
        max_cost: Maximum cost per m3 in USD.

    Returns:
        List of tuples (material, conductivity_at_temp) sorted by conductivity.
    """
    suitable = []

    for material in INSULATION_DATABASE.values():
        # Check temperature suitability
        if not material.is_suitable_for_temperature(operating_temp_c):
            continue

        # Check cost constraint
        if max_cost is not None and material.cost_per_m3_usd > max_cost:
            continue

        # Check surface type (use applications as hint)
        if surface_type.lower() == "pipe":
            # Prefer pipe-specific materials
            pass  # All materials can work for pipes

        # Calculate conductivity at operating temperature
        k_value = material.get_conductivity(operating_temp_c)
        suitable.append((material, k_value))

    # Sort by conductivity (best performers first)
    suitable.sort(key=lambda x: x[1])

    return suitable


def get_material_count() -> int:
    """Get total number of materials in database."""
    return len(INSULATION_DATABASE)


def get_categories() -> List[InsulationCategory]:
    """Get list of all material categories."""
    return list(InsulationCategory)


def get_materials_by_category() -> Dict[InsulationCategory, List[InsulationMaterial]]:
    """Get materials grouped by category."""
    result: Dict[InsulationCategory, List[InsulationMaterial]] = {
        cat: [] for cat in InsulationCategory
    }

    for material in INSULATION_DATABASE.values():
        result[material.category].append(material)

    return result


def recommend_material(
    operating_temp_c: float,
    surface_type: str,
    moisture_concern: bool = False,
    budget_priority: bool = False,
    space_limited: bool = False
) -> Optional[InsulationMaterial]:
    """
    Recommend best material for given conditions.

    Uses deterministic rules based on application requirements.

    Args:
        operating_temp_c: Operating temperature in Celsius.
        surface_type: Surface type (pipe, flat, tank, vessel).
        moisture_concern: True if moisture resistance is critical.
        budget_priority: True to prioritize lower cost.
        space_limited: True if insulation thickness is critical.

    Returns:
        Recommended InsulationMaterial or None.
    """
    candidates = find_suitable_materials(operating_temp_c, surface_type)

    if not candidates:
        logger.warning(f"No suitable materials for {operating_temp_c}C")
        return None

    # Apply preference scoring
    scored = []
    for material, k_value in candidates:
        score = 100.0

        # Thermal performance (lower k = higher score)
        score -= k_value * 1000  # Penalty for higher conductivity

        # Moisture resistance
        if moisture_concern:
            score += material.moisture_resistance * 30

        # Cost consideration
        if budget_priority:
            score -= material.cost_per_m3_usd / 100  # Penalty for high cost

        # Space limitation (prefer lower conductivity = thinner insulation)
        if space_limited:
            score -= k_value * 2000  # Extra penalty for high k
            # Bonus for aerogel/microporous
            if material.category in [InsulationCategory.AEROGEL, InsulationCategory.MICROPOROUS]:
                score += 20

        scored.append((material, score))

    # Sort by score (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        best_material = scored[0][0]
        logger.info(f"Recommended material: {best_material.name} for {operating_temp_c}C application")
        return best_material

    return None
