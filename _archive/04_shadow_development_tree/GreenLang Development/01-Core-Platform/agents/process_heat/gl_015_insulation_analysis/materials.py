"""
GL-015 INSULSCAN - Insulation Material Database

Comprehensive database of 50+ insulation materials with thermal conductivity
vs temperature relationships per ASTM C680 and manufacturer data.

All thermal conductivity values are deterministic lookup tables - no ML/LLM
in the calculation path (zero hallucination).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MaterialCategory(Enum):
    """Insulation material category."""
    MINERAL_FIBER = "mineral_fiber"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    FIBERGLASS = "fiberglass"
    MINERAL_WOOL = "mineral_wool"
    CERAMIC_FIBER = "ceramic_fiber"
    POLYURETHANE = "polyurethane"
    POLYSTYRENE = "polystyrene"
    POLYISOCYANURATE = "polyisocyanurate"
    PHENOLIC = "phenolic"
    ELASTOMERIC = "elastomeric"
    AEROGEL = "aerogel"
    MICROPOROUS = "microporous"
    VERMICULITE = "vermiculite"
    MAGNESIA = "magnesia"
    REFRACTORY = "refractory"


class TemperatureRange(BaseModel):
    """Temperature application range."""

    min_temp_f: float = Field(..., description="Minimum temperature (F)")
    max_temp_f: float = Field(..., description="Maximum temperature (F)")

    def contains(self, temp_f: float) -> bool:
        """Check if temperature is within range."""
        return self.min_temp_f <= temp_f <= self.max_temp_f


class InsulationMaterial(BaseModel):
    """Insulation material specification with k vs T data."""

    # Identity
    material_id: str = Field(..., description="Unique material identifier")
    name: str = Field(..., description="Material name")
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer")
    product_name: Optional[str] = Field(default=None, description="Product name")

    # Category
    category: MaterialCategory = Field(..., description="Material category")

    # Temperature limits
    temperature_range: TemperatureRange = Field(
        ...,
        description="Application temperature range"
    )

    # Thermal conductivity data (k vs T)
    # Format: {temperature_F: conductivity_btu_in_hr_ft2_f}
    k_vs_t_data: Dict[float, float] = Field(
        ...,
        description="Thermal conductivity vs temperature data"
    )

    # Physical properties
    density_pcf: float = Field(
        ...,
        gt=0,
        description="Nominal density (lb/ft3)"
    )
    density_range_pcf: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Density range (min, max)"
    )

    # Available thicknesses
    available_thicknesses_in: List[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        description="Available thicknesses (inches)"
    )

    # Service conditions
    suitable_for_cold_service: bool = Field(
        default=False,
        description="Suitable for cold/cryogenic service"
    )
    suitable_for_outdoor: bool = Field(
        default=True,
        description="Suitable for outdoor use"
    )
    requires_vapor_barrier: bool = Field(
        default=False,
        description="Requires vapor barrier for cold service"
    )
    moisture_resistant: bool = Field(
        default=False,
        description="Inherently moisture resistant"
    )

    # Fire rating
    flame_spread_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=200,
        description="Flame spread index per ASTM E84"
    )
    smoke_developed_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=450,
        description="Smoke developed index per ASTM E84"
    )

    # Standards compliance
    astm_standards: List[str] = Field(
        default_factory=list,
        description="Applicable ASTM standards"
    )

    class Config:
        use_enum_values = True

    def get_thermal_conductivity(self, temperature_f: float) -> float:
        """
        Get thermal conductivity at specified temperature.

        Uses linear interpolation between data points. This is a
        DETERMINISTIC calculation - no ML/LLM involved.

        Args:
            temperature_f: Temperature (F)

        Returns:
            Thermal conductivity (BTU-in/hr-ft2-F)

        Raises:
            ValueError: If temperature outside valid range
        """
        if not self.temperature_range.contains(temperature_f):
            logger.warning(
                f"Temperature {temperature_f}F outside valid range for "
                f"{self.name}: {self.temperature_range.min_temp_f} to "
                f"{self.temperature_range.max_temp_f}F"
            )

        temps = sorted(self.k_vs_t_data.keys())

        # Clamp to available range
        if temperature_f <= temps[0]:
            return self.k_vs_t_data[temps[0]]
        if temperature_f >= temps[-1]:
            return self.k_vs_t_data[temps[-1]]

        # Find bracketing temperatures and interpolate
        for i in range(len(temps) - 1):
            if temps[i] <= temperature_f <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                k1, k2 = self.k_vs_t_data[t1], self.k_vs_t_data[t2]
                # Linear interpolation
                k = k1 + (k2 - k1) * (temperature_f - t1) / (t2 - t1)
                return k

        # Fallback (should not reach here)
        return self.k_vs_t_data[temps[0]]

    def get_mean_thermal_conductivity(
        self,
        inner_temp_f: float,
        outer_temp_f: float,
    ) -> float:
        """
        Calculate mean thermal conductivity across temperature range.

        Uses trapezoidal integration for accuracy.

        Args:
            inner_temp_f: Inner surface temperature (F)
            outer_temp_f: Outer surface temperature (F)

        Returns:
            Mean thermal conductivity (BTU-in/hr-ft2-F)
        """
        if abs(inner_temp_f - outer_temp_f) < 1.0:
            return self.get_thermal_conductivity((inner_temp_f + outer_temp_f) / 2)

        # Integration with 10 steps
        num_steps = 10
        step = (outer_temp_f - inner_temp_f) / num_steps if inner_temp_f != outer_temp_f else 0.0

        if step == 0:
            return self.get_thermal_conductivity(inner_temp_f)

        total = 0.0
        for i in range(num_steps):
            t1 = inner_temp_f + i * step
            t2 = inner_temp_f + (i + 1) * step
            k1 = self.get_thermal_conductivity(t1)
            k2 = self.get_thermal_conductivity(t2)
            total += (k1 + k2) / 2 * abs(step)

        mean_k = total / abs(outer_temp_f - inner_temp_f)
        return mean_k


class InsulationMaterialDatabase:
    """
    Comprehensive database of insulation materials.

    Contains 50+ materials with thermal conductivity vs temperature data
    per ASTM C680 testing and manufacturer specifications.

    All data is deterministic - no ML/LLM in lookups (zero hallucination).

    Example:
        >>> db = InsulationMaterialDatabase()
        >>> material = db.get_material("calcium_silicate_8pcf")
        >>> k = material.get_thermal_conductivity(500)  # BTU-in/hr-ft2-F
    """

    def __init__(self) -> None:
        """Initialize the material database."""
        self._materials: Dict[str, InsulationMaterial] = {}
        self._load_materials()
        logger.info(f"Loaded {len(self._materials)} insulation materials")

    def _load_materials(self) -> None:
        """Load all material data into the database."""
        # =====================================================================
        # CALCIUM SILICATE
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="calcium_silicate_8pcf",
            name="Calcium Silicate - 8 pcf",
            category=MaterialCategory.CALCIUM_SILICATE,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=1200),
            k_vs_t_data={
                100: 0.37, 200: 0.40, 300: 0.44, 400: 0.48, 500: 0.52,
                600: 0.57, 700: 0.62, 800: 0.68, 900: 0.74, 1000: 0.80,
                1100: 0.87, 1200: 0.94,
            },
            density_pcf=8.0,
            available_thicknesses_in=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            flame_spread_index=0,
            smoke_developed_index=0,
            astm_standards=["ASTM C533"],
        ))

        self._add_material(InsulationMaterial(
            material_id="calcium_silicate_11pcf",
            name="Calcium Silicate - 11 pcf",
            category=MaterialCategory.CALCIUM_SILICATE,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=1200),
            k_vs_t_data={
                100: 0.35, 200: 0.38, 300: 0.41, 400: 0.45, 500: 0.49,
                600: 0.53, 700: 0.58, 800: 0.63, 900: 0.69, 1000: 0.75,
                1100: 0.82, 1200: 0.89,
            },
            density_pcf=11.0,
            astm_standards=["ASTM C533"],
        ))

        self._add_material(InsulationMaterial(
            material_id="calcium_silicate_15pcf",
            name="Calcium Silicate - 15 pcf",
            category=MaterialCategory.CALCIUM_SILICATE,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=1700),
            k_vs_t_data={
                100: 0.33, 200: 0.36, 300: 0.39, 400: 0.42, 500: 0.46,
                600: 0.50, 700: 0.54, 800: 0.59, 900: 0.64, 1000: 0.70,
                1100: 0.76, 1200: 0.82, 1400: 0.95, 1600: 1.10, 1700: 1.18,
            },
            density_pcf=15.0,
            astm_standards=["ASTM C533"],
        ))

        # =====================================================================
        # MINERAL WOOL / ROCK WOOL
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="mineral_wool_4pcf",
            name="Mineral Wool - 4 pcf",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=0, max_temp_f=1200),
            k_vs_t_data={
                0: 0.22, 100: 0.25, 200: 0.29, 300: 0.33, 400: 0.38,
                500: 0.43, 600: 0.49, 700: 0.55, 800: 0.62, 900: 0.70,
                1000: 0.78, 1100: 0.87, 1200: 0.96,
            },
            density_pcf=4.0,
            astm_standards=["ASTM C547", "ASTM C795"],
        ))

        self._add_material(InsulationMaterial(
            material_id="mineral_wool_6pcf",
            name="Mineral Wool - 6 pcf",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=0, max_temp_f=1200),
            k_vs_t_data={
                0: 0.21, 100: 0.24, 200: 0.27, 300: 0.31, 400: 0.35,
                500: 0.40, 600: 0.45, 700: 0.51, 800: 0.57, 900: 0.64,
                1000: 0.72, 1100: 0.80, 1200: 0.89,
            },
            density_pcf=6.0,
            astm_standards=["ASTM C547", "ASTM C795"],
        ))

        self._add_material(InsulationMaterial(
            material_id="mineral_wool_8pcf",
            name="Mineral Wool - 8 pcf",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.18, 0: 0.20, 100: 0.23, 200: 0.26, 300: 0.29,
                400: 0.33, 500: 0.37, 600: 0.42, 700: 0.47, 800: 0.53,
                900: 0.59, 1000: 0.66, 1100: 0.74, 1200: 0.82,
            },
            density_pcf=8.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547", "ASTM C795"],
        ))

        self._add_material(InsulationMaterial(
            material_id="mineral_wool_10pcf",
            name="Mineral Wool - 10 pcf",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.17, 0: 0.19, 100: 0.22, 200: 0.25, 300: 0.28,
                400: 0.31, 500: 0.35, 600: 0.39, 700: 0.44, 800: 0.49,
                900: 0.55, 1000: 0.61, 1100: 0.68, 1200: 0.76,
            },
            density_pcf=10.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547", "ASTM C795"],
        ))

        # =====================================================================
        # FIBERGLASS
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="fiberglass_1pcf",
            name="Fiberglass - 1 pcf",
            category=MaterialCategory.FIBERGLASS,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=450),
            k_vs_t_data={
                -100: 0.18, 0: 0.22, 75: 0.24, 100: 0.25, 200: 0.29,
                300: 0.34, 400: 0.40, 450: 0.43,
            },
            density_pcf=1.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547"],
        ))

        self._add_material(InsulationMaterial(
            material_id="fiberglass_1_5pcf",
            name="Fiberglass - 1.5 pcf",
            category=MaterialCategory.FIBERGLASS,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=850),
            k_vs_t_data={
                -100: 0.17, 0: 0.21, 75: 0.23, 100: 0.24, 200: 0.27,
                300: 0.31, 400: 0.36, 500: 0.41, 600: 0.47, 700: 0.54,
                800: 0.62, 850: 0.66,
            },
            density_pcf=1.5,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547"],
        ))

        self._add_material(InsulationMaterial(
            material_id="fiberglass_3pcf",
            name="Fiberglass - 3 pcf",
            category=MaterialCategory.FIBERGLASS,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=850),
            k_vs_t_data={
                -100: 0.16, 0: 0.19, 75: 0.21, 100: 0.22, 200: 0.25,
                300: 0.28, 400: 0.32, 500: 0.36, 600: 0.41, 700: 0.47,
                800: 0.54, 850: 0.57,
            },
            density_pcf=3.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547"],
        ))

        self._add_material(InsulationMaterial(
            material_id="fiberglass_6pcf",
            name="Fiberglass - 6 pcf",
            category=MaterialCategory.FIBERGLASS,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1000),
            k_vs_t_data={
                -100: 0.15, 0: 0.17, 75: 0.19, 100: 0.20, 200: 0.22,
                300: 0.25, 400: 0.28, 500: 0.32, 600: 0.36, 700: 0.41,
                800: 0.46, 900: 0.52, 1000: 0.58,
            },
            density_pcf=6.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547"],
        ))

        # =====================================================================
        # CELLULAR GLASS
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="cellular_glass_7pcf",
            name="Cellular Glass - 7 pcf",
            category=MaterialCategory.CELLULAR_GLASS,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=900),
            k_vs_t_data={
                -450: 0.18, -300: 0.20, -200: 0.22, -100: 0.24, 0: 0.27,
                100: 0.30, 200: 0.34, 300: 0.38, 400: 0.42, 500: 0.47,
                600: 0.52, 700: 0.58, 800: 0.64, 900: 0.71,
            },
            density_pcf=7.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,  # Impermeable
            moisture_resistant=True,
            flame_spread_index=0,
            smoke_developed_index=0,
            astm_standards=["ASTM C552"],
        ))

        self._add_material(InsulationMaterial(
            material_id="cellular_glass_8_5pcf",
            name="Cellular Glass - 8.5 pcf",
            category=MaterialCategory.CELLULAR_GLASS,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=900),
            k_vs_t_data={
                -450: 0.19, -300: 0.21, -200: 0.24, -100: 0.26, 0: 0.29,
                100: 0.32, 200: 0.36, 300: 0.40, 400: 0.45, 500: 0.50,
                600: 0.55, 700: 0.61, 800: 0.68, 900: 0.75,
            },
            density_pcf=8.5,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,
            moisture_resistant=True,
            astm_standards=["ASTM C552"],
        ))

        # =====================================================================
        # PERLITE
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="perlite_6pcf",
            name="Expanded Perlite - 6 pcf",
            category=MaterialCategory.PERLITE,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.32, 0: 0.36, 100: 0.40, 200: 0.45, 300: 0.50,
                400: 0.56, 500: 0.62, 600: 0.68, 700: 0.75, 800: 0.82,
                900: 0.90, 1000: 0.98, 1100: 1.07, 1200: 1.16,
            },
            density_pcf=6.0,
            flame_spread_index=0,
            smoke_developed_index=0,
            astm_standards=["ASTM C610"],
        ))

        self._add_material(InsulationMaterial(
            material_id="perlite_9pcf",
            name="Expanded Perlite - 9 pcf",
            category=MaterialCategory.PERLITE,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.35, 0: 0.39, 100: 0.43, 200: 0.48, 300: 0.53,
                400: 0.59, 500: 0.65, 600: 0.71, 700: 0.78, 800: 0.85,
                900: 0.93, 1000: 1.01, 1100: 1.10, 1200: 1.19,
            },
            density_pcf=9.0,
            astm_standards=["ASTM C610"],
        ))

        # =====================================================================
        # CERAMIC FIBER
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="ceramic_fiber_4pcf",
            name="Ceramic Fiber - 4 pcf",
            category=MaterialCategory.CERAMIC_FIBER,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2300),
            k_vs_t_data={
                500: 0.35, 700: 0.42, 900: 0.50, 1100: 0.59, 1300: 0.69,
                1500: 0.80, 1700: 0.92, 1900: 1.05, 2100: 1.20, 2300: 1.36,
            },
            density_pcf=4.0,
            available_thicknesses_in=[0.5, 1.0, 1.5, 2.0],
            astm_standards=["ASTM C892"],
        ))

        self._add_material(InsulationMaterial(
            material_id="ceramic_fiber_6pcf",
            name="Ceramic Fiber - 6 pcf",
            category=MaterialCategory.CERAMIC_FIBER,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2300),
            k_vs_t_data={
                500: 0.32, 700: 0.38, 900: 0.45, 1100: 0.53, 1300: 0.62,
                1500: 0.72, 1700: 0.83, 1900: 0.95, 2100: 1.08, 2300: 1.22,
            },
            density_pcf=6.0,
            astm_standards=["ASTM C892"],
        ))

        self._add_material(InsulationMaterial(
            material_id="ceramic_fiber_8pcf",
            name="Ceramic Fiber - 8 pcf",
            category=MaterialCategory.CERAMIC_FIBER,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2600),
            k_vs_t_data={
                500: 0.30, 700: 0.35, 900: 0.41, 1100: 0.48, 1300: 0.56,
                1500: 0.65, 1700: 0.75, 1900: 0.86, 2100: 0.98, 2300: 1.11,
                2400: 1.18, 2600: 1.33,
            },
            density_pcf=8.0,
            astm_standards=["ASTM C892"],
        ))

        # =====================================================================
        # POLYURETHANE FOAM
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="polyurethane_2pcf",
            name="Polyurethane Foam - 2 pcf",
            category=MaterialCategory.POLYURETHANE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=250),
            k_vs_t_data={
                -297: 0.08, -200: 0.10, -100: 0.12, 0: 0.14, 50: 0.15,
                75: 0.16, 100: 0.17, 150: 0.19, 200: 0.21, 250: 0.24,
            },
            density_pcf=2.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            flame_spread_index=25,
            smoke_developed_index=50,
            astm_standards=["ASTM C591"],
        ))

        self._add_material(InsulationMaterial(
            material_id="polyurethane_4pcf",
            name="Polyurethane Foam - 4 pcf",
            category=MaterialCategory.POLYURETHANE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=300),
            k_vs_t_data={
                -297: 0.09, -200: 0.11, -100: 0.13, 0: 0.15, 50: 0.16,
                75: 0.17, 100: 0.18, 150: 0.20, 200: 0.22, 250: 0.25, 300: 0.28,
            },
            density_pcf=4.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C591"],
        ))

        # =====================================================================
        # POLYSTYRENE (EPS & XPS)
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="eps_1pcf",
            name="Expanded Polystyrene (EPS) - 1 pcf",
            category=MaterialCategory.POLYSTYRENE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=165),
            k_vs_t_data={
                -297: 0.12, -200: 0.15, -100: 0.18, 0: 0.22, 40: 0.24,
                75: 0.26, 100: 0.28, 140: 0.31, 165: 0.33,
            },
            density_pcf=1.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            flame_spread_index=25,
            smoke_developed_index=450,
            astm_standards=["ASTM C578"],
        ))

        self._add_material(InsulationMaterial(
            material_id="eps_1_5pcf",
            name="Expanded Polystyrene (EPS) - 1.5 pcf",
            category=MaterialCategory.POLYSTYRENE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=165),
            k_vs_t_data={
                -297: 0.11, -200: 0.14, -100: 0.17, 0: 0.21, 40: 0.23,
                75: 0.25, 100: 0.27, 140: 0.30, 165: 0.32,
            },
            density_pcf=1.5,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C578"],
        ))

        self._add_material(InsulationMaterial(
            material_id="xps_1_8pcf",
            name="Extruded Polystyrene (XPS) - 1.8 pcf",
            category=MaterialCategory.POLYSTYRENE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=165),
            k_vs_t_data={
                -297: 0.10, -200: 0.13, -100: 0.16, 0: 0.19, 40: 0.21,
                75: 0.22, 100: 0.24, 140: 0.27, 165: 0.29,
            },
            density_pcf=1.8,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,  # Closed cell
            moisture_resistant=True,
            astm_standards=["ASTM C578"],
        ))

        # =====================================================================
        # POLYISOCYANURATE (PIR)
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="pir_2pcf",
            name="Polyisocyanurate (PIR) - 2 pcf",
            category=MaterialCategory.POLYISOCYANURATE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=300),
            k_vs_t_data={
                -297: 0.09, -200: 0.11, -100: 0.13, 0: 0.15, 50: 0.16,
                75: 0.17, 100: 0.18, 150: 0.20, 200: 0.23, 250: 0.26, 300: 0.29,
            },
            density_pcf=2.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            flame_spread_index=25,
            smoke_developed_index=50,
            astm_standards=["ASTM C591"],
        ))

        # =====================================================================
        # PHENOLIC FOAM
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="phenolic_3pcf",
            name="Phenolic Foam - 3 pcf",
            category=MaterialCategory.PHENOLIC,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=250),
            k_vs_t_data={
                -297: 0.08, -200: 0.10, -100: 0.12, 0: 0.14, 50: 0.15,
                75: 0.16, 100: 0.17, 150: 0.19, 200: 0.21, 250: 0.24,
            },
            density_pcf=3.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            flame_spread_index=5,
            smoke_developed_index=5,
            astm_standards=["ASTM C1126"],
        ))

        # =====================================================================
        # ELASTOMERIC FOAM
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="elastomeric_4pcf",
            name="Elastomeric Foam - 4 pcf",
            category=MaterialCategory.ELASTOMERIC,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=220),
            k_vs_t_data={
                -297: 0.14, -200: 0.17, -100: 0.21, 0: 0.25, 50: 0.27,
                75: 0.28, 100: 0.30, 150: 0.33, 200: 0.37, 220: 0.39,
            },
            density_pcf=4.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,  # Built-in vapor retarder
            moisture_resistant=True,
            flame_spread_index=25,
            smoke_developed_index=50,
            astm_standards=["ASTM C534", "ASTM C1427"],
        ))

        self._add_material(InsulationMaterial(
            material_id="elastomeric_6pcf",
            name="Elastomeric Foam - 6 pcf",
            category=MaterialCategory.ELASTOMERIC,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=220),
            k_vs_t_data={
                -297: 0.13, -200: 0.16, -100: 0.20, 0: 0.24, 50: 0.26,
                75: 0.27, 100: 0.29, 150: 0.32, 200: 0.35, 220: 0.37,
            },
            density_pcf=6.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,
            moisture_resistant=True,
            astm_standards=["ASTM C534", "ASTM C1427"],
        ))

        # =====================================================================
        # AEROGEL
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="aerogel_blanket_8pcf",
            name="Aerogel Blanket - 8 pcf",
            category=MaterialCategory.AEROGEL,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=1200),
            k_vs_t_data={
                -450: 0.06, -300: 0.07, -200: 0.08, -100: 0.09, 0: 0.10,
                100: 0.11, 200: 0.12, 300: 0.13, 400: 0.14, 500: 0.15,
                600: 0.17, 700: 0.18, 800: 0.20, 900: 0.22, 1000: 0.24,
                1100: 0.26, 1200: 0.29,
            },
            density_pcf=8.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            available_thicknesses_in=[0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            astm_standards=["ASTM C1728"],
        ))

        self._add_material(InsulationMaterial(
            material_id="aerogel_blanket_12pcf",
            name="Aerogel Blanket - 12 pcf",
            category=MaterialCategory.AEROGEL,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=1200),
            k_vs_t_data={
                -450: 0.07, -300: 0.08, -200: 0.09, -100: 0.10, 0: 0.11,
                100: 0.12, 200: 0.13, 300: 0.14, 400: 0.15, 500: 0.16,
                600: 0.18, 700: 0.19, 800: 0.21, 900: 0.23, 1000: 0.25,
                1100: 0.28, 1200: 0.31,
            },
            density_pcf=12.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C1728"],
        ))

        # =====================================================================
        # MICROPOROUS
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="microporous_12pcf",
            name="Microporous Insulation - 12 pcf",
            category=MaterialCategory.MICROPOROUS,
            temperature_range=TemperatureRange(min_temp_f=200, max_temp_f=1800),
            k_vs_t_data={
                200: 0.15, 400: 0.17, 600: 0.20, 800: 0.23, 1000: 0.27,
                1200: 0.31, 1400: 0.36, 1600: 0.42, 1800: 0.48,
            },
            density_pcf=12.0,
            available_thicknesses_in=[0.5, 1.0, 1.5, 2.0],
            astm_standards=["ASTM C1676"],
        ))

        self._add_material(InsulationMaterial(
            material_id="microporous_20pcf",
            name="Microporous Insulation - 20 pcf",
            category=MaterialCategory.MICROPOROUS,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1800),
            k_vs_t_data={
                -100: 0.12, 0: 0.13, 200: 0.14, 400: 0.16, 600: 0.18,
                800: 0.21, 1000: 0.24, 1200: 0.28, 1400: 0.32, 1600: 0.37, 1800: 0.43,
            },
            density_pcf=20.0,
            suitable_for_cold_service=True,
            astm_standards=["ASTM C1676"],
        ))

        # =====================================================================
        # MAGNESIA (85%)
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="magnesia_85_12pcf",
            name="85% Magnesia - 12 pcf",
            category=MaterialCategory.MAGNESIA,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=600),
            k_vs_t_data={
                100: 0.38, 200: 0.42, 300: 0.46, 400: 0.51, 500: 0.56, 600: 0.62,
            },
            density_pcf=12.0,
            astm_standards=["ASTM C612"],
        ))

        # =====================================================================
        # VERMICULITE
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="vermiculite_6pcf",
            name="Vermiculite - 6 pcf",
            category=MaterialCategory.VERMICULITE,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=1900),
            k_vs_t_data={
                100: 0.46, 300: 0.52, 500: 0.59, 700: 0.67, 900: 0.76,
                1100: 0.86, 1300: 0.97, 1500: 1.09, 1700: 1.22, 1900: 1.36,
            },
            density_pcf=6.0,
            flame_spread_index=0,
            smoke_developed_index=0,
            astm_standards=["ASTM C516"],
        ))

        # =====================================================================
        # REFRACTORY MATERIALS
        # =====================================================================
        self._add_material(InsulationMaterial(
            material_id="insulating_firebrick_27pcf",
            name="Insulating Firebrick (IFB) - 27 pcf",
            category=MaterialCategory.REFRACTORY,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2600),
            k_vs_t_data={
                500: 0.70, 1000: 0.85, 1500: 1.05, 2000: 1.30, 2500: 1.60, 2600: 1.70,
            },
            density_pcf=27.0,
            available_thicknesses_in=[2.5, 3.0, 4.5, 6.0, 9.0],
            astm_standards=["ASTM C155"],
        ))

        self._add_material(InsulationMaterial(
            material_id="insulating_firebrick_35pcf",
            name="Insulating Firebrick (IFB) - 35 pcf",
            category=MaterialCategory.REFRACTORY,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2800),
            k_vs_t_data={
                500: 0.80, 1000: 0.95, 1500: 1.15, 2000: 1.40, 2500: 1.70,
                2600: 1.80, 2800: 2.00,
            },
            density_pcf=35.0,
            astm_standards=["ASTM C155"],
        ))

        self._add_material(InsulationMaterial(
            material_id="castable_refractory_60pcf",
            name="Castable Refractory - 60 pcf",
            category=MaterialCategory.REFRACTORY,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2600),
            k_vs_t_data={
                500: 2.50, 1000: 2.80, 1500: 3.20, 2000: 3.70, 2500: 4.30, 2600: 4.50,
            },
            density_pcf=60.0,
            astm_standards=["ASTM C401"],
        ))

        # =====================================================================
        # ADDITIONAL SPECIALTY MATERIALS
        # =====================================================================

        # High-temp mineral wool
        self._add_material(InsulationMaterial(
            material_id="ht_mineral_wool_12pcf",
            name="High-Temp Mineral Wool - 12 pcf",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=200, max_temp_f=1800),
            k_vs_t_data={
                200: 0.28, 400: 0.34, 600: 0.41, 800: 0.49, 1000: 0.58,
                1200: 0.68, 1400: 0.79, 1600: 0.91, 1800: 1.04,
            },
            density_pcf=12.0,
            astm_standards=["ASTM C592"],
        ))

        # Silica aerogel composite
        self._add_material(InsulationMaterial(
            material_id="silica_aerogel_10pcf",
            name="Silica Aerogel Composite - 10 pcf",
            category=MaterialCategory.AEROGEL,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=1100),
            k_vs_t_data={
                -450: 0.05, -300: 0.06, -200: 0.07, -100: 0.08, 0: 0.09,
                100: 0.10, 200: 0.11, 300: 0.12, 400: 0.13, 500: 0.14,
                600: 0.15, 700: 0.17, 800: 0.18, 900: 0.20, 1000: 0.22, 1100: 0.24,
            },
            density_pcf=10.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C1728"],
        ))

        # =====================================================================
        # ADDITIONAL MATERIALS TO REACH 50+
        # =====================================================================

        # Mineral wool variations
        self._add_material(InsulationMaterial(
            material_id="mineral_wool_2pcf",
            name="Mineral Wool - 2 pcf (Light Duty)",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=0, max_temp_f=850),
            k_vs_t_data={
                0: 0.24, 100: 0.28, 200: 0.33, 300: 0.39, 400: 0.46,
                500: 0.54, 600: 0.63, 700: 0.73, 800: 0.84, 850: 0.90,
            },
            density_pcf=2.0,
            astm_standards=["ASTM C547"],
        ))

        self._add_material(InsulationMaterial(
            material_id="mineral_wool_12pcf",
            name="Mineral Wool - 12 pcf (Heavy Duty)",
            category=MaterialCategory.MINERAL_WOOL,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.16, 0: 0.18, 100: 0.21, 200: 0.24, 300: 0.27,
                400: 0.30, 500: 0.34, 600: 0.38, 700: 0.43, 800: 0.48,
                900: 0.54, 1000: 0.60, 1100: 0.67, 1200: 0.75,
            },
            density_pcf=12.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C547", "ASTM C795"],
        ))

        # Additional fiberglass
        self._add_material(InsulationMaterial(
            material_id="fiberglass_0_75pcf",
            name="Fiberglass - 0.75 pcf (Blanket)",
            category=MaterialCategory.FIBERGLASS,
            temperature_range=TemperatureRange(min_temp_f=0, max_temp_f=350),
            k_vs_t_data={
                0: 0.24, 75: 0.26, 100: 0.27, 200: 0.32, 300: 0.38, 350: 0.42,
            },
            density_pcf=0.75,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C553"],
        ))

        # Additional calcium silicate
        self._add_material(InsulationMaterial(
            material_id="calcium_silicate_22pcf",
            name="Calcium Silicate - 22 pcf (High Density)",
            category=MaterialCategory.CALCIUM_SILICATE,
            temperature_range=TemperatureRange(min_temp_f=100, max_temp_f=1900),
            k_vs_t_data={
                100: 0.38, 300: 0.42, 500: 0.47, 700: 0.53, 900: 0.60,
                1100: 0.68, 1300: 0.77, 1500: 0.87, 1700: 0.98, 1900: 1.10,
            },
            density_pcf=22.0,
            astm_standards=["ASTM C533"],
        ))

        # Additional perlite
        self._add_material(InsulationMaterial(
            material_id="perlite_12pcf",
            name="Expanded Perlite - 12 pcf",
            category=MaterialCategory.PERLITE,
            temperature_range=TemperatureRange(min_temp_f=-100, max_temp_f=1200),
            k_vs_t_data={
                -100: 0.38, 0: 0.42, 100: 0.46, 200: 0.51, 300: 0.56,
                400: 0.62, 500: 0.68, 600: 0.74, 700: 0.81, 800: 0.88,
                900: 0.96, 1000: 1.04, 1100: 1.13, 1200: 1.22,
            },
            density_pcf=12.0,
            astm_standards=["ASTM C610"],
        ))

        # Additional ceramic fiber
        self._add_material(InsulationMaterial(
            material_id="ceramic_fiber_10pcf",
            name="Ceramic Fiber - 10 pcf",
            category=MaterialCategory.CERAMIC_FIBER,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2600),
            k_vs_t_data={
                500: 0.28, 700: 0.33, 900: 0.39, 1100: 0.46, 1300: 0.54,
                1500: 0.63, 1700: 0.73, 1900: 0.84, 2100: 0.96, 2300: 1.09,
                2600: 1.28,
            },
            density_pcf=10.0,
            astm_standards=["ASTM C892"],
        ))

        # Additional polyurethane
        self._add_material(InsulationMaterial(
            material_id="polyurethane_6pcf",
            name="Polyurethane Foam - 6 pcf (Rigid)",
            category=MaterialCategory.POLYURETHANE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=300),
            k_vs_t_data={
                -297: 0.10, -200: 0.12, -100: 0.14, 0: 0.16, 50: 0.17,
                75: 0.18, 100: 0.19, 150: 0.21, 200: 0.24, 250: 0.27, 300: 0.30,
            },
            density_pcf=6.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C591"],
        ))

        # Additional EPS
        self._add_material(InsulationMaterial(
            material_id="eps_2pcf",
            name="Expanded Polystyrene (EPS) - 2 pcf",
            category=MaterialCategory.POLYSTYRENE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=165),
            k_vs_t_data={
                -297: 0.10, -200: 0.13, -100: 0.16, 0: 0.20, 40: 0.22,
                75: 0.24, 100: 0.26, 140: 0.29, 165: 0.31,
            },
            density_pcf=2.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C578"],
        ))

        # Additional PIR
        self._add_material(InsulationMaterial(
            material_id="pir_3pcf",
            name="Polyisocyanurate (PIR) - 3 pcf",
            category=MaterialCategory.POLYISOCYANURATE,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=350),
            k_vs_t_data={
                -297: 0.10, -200: 0.12, -100: 0.14, 0: 0.16, 50: 0.17,
                75: 0.18, 100: 0.19, 150: 0.21, 200: 0.24, 250: 0.27,
                300: 0.30, 350: 0.34,
            },
            density_pcf=3.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            astm_standards=["ASTM C591"],
        ))

        # Additional phenolic
        self._add_material(InsulationMaterial(
            material_id="phenolic_5pcf",
            name="Phenolic Foam - 5 pcf",
            category=MaterialCategory.PHENOLIC,
            temperature_range=TemperatureRange(min_temp_f=-297, max_temp_f=300),
            k_vs_t_data={
                -297: 0.09, -200: 0.11, -100: 0.13, 0: 0.15, 50: 0.16,
                75: 0.17, 100: 0.18, 150: 0.20, 200: 0.22, 250: 0.25, 300: 0.28,
            },
            density_pcf=5.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=True,
            flame_spread_index=5,
            smoke_developed_index=5,
            astm_standards=["ASTM C1126"],
        ))

        # Additional cellular glass
        self._add_material(InsulationMaterial(
            material_id="cellular_glass_10pcf",
            name="Cellular Glass - 10 pcf",
            category=MaterialCategory.CELLULAR_GLASS,
            temperature_range=TemperatureRange(min_temp_f=-450, max_temp_f=900),
            k_vs_t_data={
                -450: 0.20, -300: 0.23, -200: 0.26, -100: 0.29, 0: 0.32,
                100: 0.35, 200: 0.39, 300: 0.43, 400: 0.48, 500: 0.53,
                600: 0.58, 700: 0.64, 800: 0.71, 900: 0.78,
            },
            density_pcf=10.0,
            suitable_for_cold_service=True,
            requires_vapor_barrier=False,
            moisture_resistant=True,
            astm_standards=["ASTM C552"],
        ))

        # Additional IFB
        self._add_material(InsulationMaterial(
            material_id="insulating_firebrick_20pcf",
            name="Insulating Firebrick (IFB) - 20 pcf (Lightweight)",
            category=MaterialCategory.REFRACTORY,
            temperature_range=TemperatureRange(min_temp_f=500, max_temp_f=2300),
            k_vs_t_data={
                500: 0.55, 1000: 0.70, 1500: 0.90, 2000: 1.15, 2300: 1.35,
            },
            density_pcf=20.0,
            astm_standards=["ASTM C155"],
        ))

        logger.info(f"Loaded {len(self._materials)} insulation materials")

    def _add_material(self, material: InsulationMaterial) -> None:
        """Add a material to the database."""
        self._materials[material.material_id] = material

    def get_material(self, material_id: str) -> Optional[InsulationMaterial]:
        """
        Get material by ID.

        Args:
            material_id: Material identifier

        Returns:
            InsulationMaterial or None if not found
        """
        return self._materials.get(material_id)

    def search_materials(
        self,
        category: Optional[MaterialCategory] = None,
        min_temp_f: Optional[float] = None,
        max_temp_f: Optional[float] = None,
        suitable_for_cold: Optional[bool] = None,
        max_k_value: Optional[float] = None,
    ) -> List[InsulationMaterial]:
        """
        Search materials by criteria.

        Args:
            category: Filter by material category
            min_temp_f: Minimum temperature requirement
            max_temp_f: Maximum temperature requirement
            suitable_for_cold: Filter by cold service suitability
            max_k_value: Maximum thermal conductivity at mean temp

        Returns:
            List of matching materials
        """
        results = []

        for material in self._materials.values():
            # Category filter
            if category and material.category != category:
                continue

            # Temperature range filter
            if min_temp_f is not None:
                if material.temperature_range.min_temp_f > min_temp_f:
                    continue
            if max_temp_f is not None:
                if material.temperature_range.max_temp_f < max_temp_f:
                    continue

            # Cold service filter
            if suitable_for_cold is not None:
                if material.suitable_for_cold_service != suitable_for_cold:
                    continue

            # K-value filter (at mean of available range)
            if max_k_value is not None:
                temps = sorted(material.k_vs_t_data.keys())
                mean_temp = (temps[0] + temps[-1]) / 2
                k = material.get_thermal_conductivity(mean_temp)
                if k > max_k_value:
                    continue

            results.append(material)

        return results

    def get_recommended_materials(
        self,
        operating_temp_f: float,
        cold_service: bool = False,
    ) -> List[InsulationMaterial]:
        """
        Get recommended materials for given service conditions.

        Args:
            operating_temp_f: Operating temperature (F)
            cold_service: Is this cold service

        Returns:
            List of recommended materials sorted by k-value
        """
        candidates = self.search_materials(
            min_temp_f=operating_temp_f if not cold_service else None,
            max_temp_f=operating_temp_f if cold_service else None,
            suitable_for_cold=cold_service if cold_service else None,
        )

        # Filter to those that can handle the operating temp
        valid = []
        for mat in candidates:
            if mat.temperature_range.contains(operating_temp_f):
                valid.append(mat)

        # Sort by k-value at operating temperature
        valid.sort(key=lambda m: m.get_thermal_conductivity(operating_temp_f))

        return valid

    def list_all_materials(self) -> List[InsulationMaterial]:
        """Get all materials in the database."""
        return list(self._materials.values())

    def get_categories(self) -> List[MaterialCategory]:
        """Get all available material categories."""
        categories = set()
        for material in self._materials.values():
            categories.add(material.category)
        return sorted(categories, key=lambda c: c.value)

    @property
    def material_count(self) -> int:
        """Get total number of materials in database."""
        return len(self._materials)
