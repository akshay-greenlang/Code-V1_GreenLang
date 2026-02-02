"""
Thermal Fluid Library
=====================

Zero-hallucination thermal fluid property database supporting 25+ fluids.

Provides thermophysical properties (density, specific heat, viscosity,
thermal conductivity, enthalpy, entropy) for heat transfer applications.

Supported Fluid Categories:
--------------------------
1. Water/Steam (IAPWS-IF97)
2. Therminol heat transfer fluids (55, 59, 62, 66, VP-1)
3. Dowtherm heat transfer fluids (A, G, J, MX, Q, RP)
4. Syltherm silicone fluids (800, XLT)
5. Glycol solutions (Ethylene/Propylene at 20-60%)
6. Molten salts (Solar Salt, Hitec, Hitec XL)
7. Mineral oils and supercritical CO2

Data Sources:
-------------
- IAPWS-IF97 (Water/Steam)
- Solutia/Eastman Technical Data Sheets (Therminol)
- Dow Chemical Technical Data Sheets (Dowtherm, Syltherm)
- Coastal Chemical Technical Data (Hitec)
- NIST Chemistry WebBook
- Published correlations from peer-reviewed journals

Author: GL-009_ThermalIQ
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import hashlib
import json
import math
from datetime import datetime, timezone

# Import property correlations
from .property_correlations import PropertyCorrelations, CorrelationResult


class FluidCategory(Enum):
    """Classification of thermal fluids."""
    WATER_STEAM = "water_steam"
    THERMINOL = "therminol"
    DOWTHERM = "dowtherm"
    SYLTHERM = "syltherm"
    ETHYLENE_GLYCOL = "ethylene_glycol"
    PROPYLENE_GLYCOL = "propylene_glycol"
    MOLTEN_SALT = "molten_salt"
    MINERAL_OIL = "mineral_oil"
    SUPERCRITICAL = "supercritical"


@dataclass
class FluidProperties:
    """
    Complete thermophysical properties of a fluid at given conditions.

    All properties include provenance tracking for zero-hallucination guarantee.

    Attributes:
        fluid_name: Name of the fluid
        temperature_K: Temperature (K)
        pressure_kPa: Pressure (kPa)
        density_kg_m3: Density (kg/m3)
        specific_heat_kJ_kg_K: Specific heat at constant pressure (kJ/(kg*K))
        viscosity_mPa_s: Dynamic viscosity (mPa*s = cP)
        conductivity_W_m_K: Thermal conductivity (W/(m*K))
        enthalpy_kJ_kg: Specific enthalpy (kJ/kg)
        entropy_kJ_kg_K: Specific entropy (kJ/(kg*K))
        prandtl: Prandtl number (dimensionless)
        uncertainty_percent: Property uncertainty (%)
        provenance_hash: SHA-256 hash of calculation
        data_source: Reference for property data
        valid_range: Temperature validity range
    """
    fluid_name: str
    temperature_K: float
    pressure_kPa: float
    density_kg_m3: float
    specific_heat_kJ_kg_K: float
    viscosity_mPa_s: float
    conductivity_W_m_K: float
    enthalpy_kJ_kg: Optional[float] = None
    entropy_kJ_kg_K: Optional[float] = None
    prandtl: Optional[float] = None
    uncertainty_percent: float = 2.0
    provenance_hash: str = ""
    data_source: str = ""
    valid_range: Tuple[float, float] = (0.0, 0.0)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate Prandtl number if not provided."""
        if self.prandtl is None and self.viscosity_mPa_s and self.conductivity_W_m_K:
            # Pr = Cp * mu / k
            # Units: (kJ/(kg*K)) * (mPa*s) / (W/(m*K))
            # Convert: kJ = 1000 J, mPa*s = 0.001 Pa*s
            mu_Pa_s = self.viscosity_mPa_s * 0.001
            Cp_J_kg_K = self.specific_heat_kJ_kg_K * 1000
            self.prandtl = (Cp_J_kg_K * mu_Pa_s) / self.conductivity_W_m_K

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fluid_name": self.fluid_name,
            "temperature_K": self.temperature_K,
            "pressure_kPa": self.pressure_kPa,
            "density_kg_m3": self.density_kg_m3,
            "specific_heat_kJ_kg_K": self.specific_heat_kJ_kg_K,
            "viscosity_mPa_s": self.viscosity_mPa_s,
            "conductivity_W_m_K": self.conductivity_W_m_K,
            "enthalpy_kJ_kg": self.enthalpy_kJ_kg,
            "entropy_kJ_kg_K": self.entropy_kJ_kg_K,
            "prandtl": self.prandtl,
            "uncertainty_percent": self.uncertainty_percent,
            "provenance_hash": self.provenance_hash,
            "data_source": self.data_source,
            "valid_range": self.valid_range,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class FluidRecommendation:
    """
    Fluid recommendation for specific application.

    Attributes:
        fluid_name: Recommended fluid
        category: Fluid category
        score: Suitability score (0-100)
        reasons: Reasons for recommendation
        limitations: Known limitations
        alternatives: Alternative fluids
    """
    fluid_name: str
    category: FluidCategory
    score: float
    reasons: List[str]
    limitations: List[str]
    alternatives: List[str]
    temperature_range_K: Tuple[float, float]
    max_pressure_kPa: float
    typical_applications: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fluid_name": self.fluid_name,
            "category": self.category.value,
            "score": self.score,
            "reasons": self.reasons,
            "limitations": self.limitations,
            "alternatives": self.alternatives,
            "temperature_range_K": self.temperature_range_K,
            "max_pressure_kPa": self.max_pressure_kPa,
            "typical_applications": self.typical_applications
        }


class ThermalFluidLibrary:
    """
    Zero-hallucination thermal fluid property library.

    Provides validated thermophysical properties for 25+ thermal fluids
    with complete provenance tracking.

    Guarantees:
    - DETERMINISTIC: Same input produces identical output
    - REPRODUCIBLE: Full provenance tracking with SHA-256 hashes
    - VALIDATED: All correlations from published sources
    - NO LLM: Zero hallucination risk

    Supported Fluids:
    - Water/Steam (IAPWS-IF97)
    - Therminol 55, 59, 62, 66, VP-1
    - Dowtherm A, G, J, MX, Q, RP
    - Syltherm 800, XLT
    - Ethylene Glycol (20-60%)
    - Propylene Glycol (20-60%)
    - Molten Salts (Solar Salt, Hitec, Hitec XL)
    - Mineral Oil, Supercritical CO2

    Example:
    --------
    >>> library = ThermalFluidLibrary()
    >>> props = library.get_properties("therminol_66", T=573.15, P=101.325)
    >>> print(f"Density: {props.density_kg_m3} kg/m3")
    >>> print(f"Cp: {props.specific_heat_kJ_kg_K} kJ/(kg*K)")
    """

    # Fluid metadata: {fluid_name: (category, T_min, T_max, P_max, data_source)}
    FLUID_METADATA = {
        # Water/Steam
        "water": (FluidCategory.WATER_STEAM, 273.15, 647.096, 100000, "IAPWS-IF97"),
        "steam": (FluidCategory.WATER_STEAM, 373.15, 863.15, 100000, "IAPWS-IF97"),

        # Therminol Series
        "therminol_55": (FluidCategory.THERMINOL, 233.15, 573.15, 1000, "Eastman Technical Data"),
        "therminol_59": (FluidCategory.THERMINOL, 248.15, 588.15, 1000, "Eastman Technical Data"),
        "therminol_62": (FluidCategory.THERMINOL, 264.15, 598.15, 1000, "Eastman Technical Data"),
        "therminol_66": (FluidCategory.THERMINOL, 264.15, 618.15, 1000, "Eastman Technical Data"),
        "therminol_vp1": (FluidCategory.THERMINOL, 285.15, 673.15, 1000, "Eastman Technical Data"),

        # Dowtherm Series
        "dowtherm_a": (FluidCategory.DOWTHERM, 288.15, 673.15, 1000, "Dow Chemical Technical Data"),
        "dowtherm_g": (FluidCategory.DOWTHERM, 269.15, 633.15, 1000, "Dow Chemical Technical Data"),
        "dowtherm_j": (FluidCategory.DOWTHERM, 193.15, 588.15, 1000, "Dow Chemical Technical Data"),
        "dowtherm_mx": (FluidCategory.DOWTHERM, 248.15, 603.15, 1000, "Dow Chemical Technical Data"),
        "dowtherm_q": (FluidCategory.DOWTHERM, 238.15, 603.15, 1000, "Dow Chemical Technical Data"),
        "dowtherm_rp": (FluidCategory.DOWTHERM, 253.15, 623.15, 1000, "Dow Chemical Technical Data"),

        # Syltherm Series
        "syltherm_800": (FluidCategory.SYLTHERM, 233.15, 673.15, 1000, "Dow Chemical Technical Data"),
        "syltherm_xlt": (FluidCategory.SYLTHERM, 173.15, 533.15, 1000, "Dow Chemical Technical Data"),

        # Ethylene Glycol Solutions (% concentration)
        "ethylene_glycol_20": (FluidCategory.ETHYLENE_GLYCOL, 263.15, 383.15, 1000, "ASHRAE Handbook"),
        "ethylene_glycol_30": (FluidCategory.ETHYLENE_GLYCOL, 255.15, 383.15, 1000, "ASHRAE Handbook"),
        "ethylene_glycol_40": (FluidCategory.ETHYLENE_GLYCOL, 248.15, 383.15, 1000, "ASHRAE Handbook"),
        "ethylene_glycol_50": (FluidCategory.ETHYLENE_GLYCOL, 237.15, 383.15, 1000, "ASHRAE Handbook"),
        "ethylene_glycol_60": (FluidCategory.ETHYLENE_GLYCOL, 225.15, 383.15, 1000, "ASHRAE Handbook"),

        # Propylene Glycol Solutions (% concentration)
        "propylene_glycol_20": (FluidCategory.PROPYLENE_GLYCOL, 264.15, 383.15, 1000, "ASHRAE Handbook"),
        "propylene_glycol_30": (FluidCategory.PROPYLENE_GLYCOL, 258.15, 383.15, 1000, "ASHRAE Handbook"),
        "propylene_glycol_40": (FluidCategory.PROPYLENE_GLYCOL, 251.15, 383.15, 1000, "ASHRAE Handbook"),
        "propylene_glycol_50": (FluidCategory.PROPYLENE_GLYCOL, 241.15, 383.15, 1000, "ASHRAE Handbook"),
        "propylene_glycol_60": (FluidCategory.PROPYLENE_GLYCOL, 230.15, 383.15, 1000, "ASHRAE Handbook"),

        # Molten Salts
        "solar_salt": (FluidCategory.MOLTEN_SALT, 533.15, 873.15, 500, "Zavoico (2001)"),
        "hitec": (FluidCategory.MOLTEN_SALT, 415.15, 808.15, 500, "Coastal Chemical"),
        "hitec_xl": (FluidCategory.MOLTEN_SALT, 393.15, 773.15, 500, "Coastal Chemical"),

        # Other
        "mineral_oil": (FluidCategory.MINERAL_OIL, 273.15, 573.15, 1000, "Generic mineral oil"),
        "co2_supercritical": (FluidCategory.SUPERCRITICAL, 305.15, 573.15, 30000, "NIST Webbook"),
    }

    def __init__(self):
        """Initialize thermal fluid library."""
        self.correlations = PropertyCorrelations()
        self._property_cache: Dict[str, FluidProperties] = {}

    def get_properties(
        self,
        fluid_name: str,
        T: float,
        P: float = 101.325,
    ) -> FluidProperties:
        """
        Get complete thermophysical properties for a fluid.

        Args:
            fluid_name: Name of the fluid (e.g., 'therminol_66')
            T: Temperature (K)
            P: Pressure (kPa), default atmospheric

        Returns:
            FluidProperties with complete property set

        Raises:
            ValueError: If fluid not found or T/P out of range

        Example:
            >>> library = ThermalFluidLibrary()
            >>> props = library.get_properties("therminol_66", T=573.15)
            >>> print(f"Density: {props.density_kg_m3} kg/m3")
        """
        # Validate fluid name
        fluid_name_lower = fluid_name.lower()
        if fluid_name_lower not in self.FLUID_METADATA:
            raise ValueError(
                f"Unknown fluid: {fluid_name}. "
                f"Available fluids: {list(self.FLUID_METADATA.keys())}"
            )

        # Get metadata
        category, T_min, T_max, P_max, data_source = self.FLUID_METADATA[fluid_name_lower]

        # Validate temperature range
        if T < T_min or T > T_max:
            raise ValueError(
                f"Temperature {T} K outside valid range [{T_min}, {T_max}] K "
                f"for {fluid_name}"
            )

        # Validate pressure
        if P > P_max:
            raise ValueError(
                f"Pressure {P} kPa exceeds maximum {P_max} kPa for {fluid_name}"
            )

        # Get properties from correlations
        density = self.get_density(fluid_name_lower, T, P)
        Cp = self.get_Cp(fluid_name_lower, T, P)
        viscosity = self.get_viscosity(fluid_name_lower, T, P)
        conductivity = self.get_conductivity(fluid_name_lower, T, P)
        enthalpy = self.get_enthalpy(fluid_name_lower, T, P)
        entropy = self.get_entropy(fluid_name_lower, T, P)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            fluid_name_lower, T, P,
            {"density": density, "Cp": Cp, "viscosity": viscosity,
             "conductivity": conductivity, "enthalpy": enthalpy, "entropy": entropy}
        )

        return FluidProperties(
            fluid_name=fluid_name_lower,
            temperature_K=T,
            pressure_kPa=P,
            density_kg_m3=density,
            specific_heat_kJ_kg_K=Cp,
            viscosity_mPa_s=viscosity,
            conductivity_W_m_K=conductivity,
            enthalpy_kJ_kg=enthalpy,
            entropy_kJ_kg_K=entropy,
            uncertainty_percent=self._get_uncertainty(fluid_name_lower),
            provenance_hash=provenance_hash,
            data_source=data_source,
            valid_range=(T_min, T_max),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"category": category.value}
        )

    def get_density(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get density for a fluid at given conditions.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Density (kg/m3)
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_density(fluid_lower, T, P)
        return round(result.value, 4)

    def get_Cp(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get specific heat capacity for a fluid.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Specific heat (kJ/(kg*K))
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_specific_heat(fluid_lower, T, P)
        return round(result.value, 4)

    def get_enthalpy(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get specific enthalpy for a fluid.

        Calculated by integrating Cp from reference temperature.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Specific enthalpy (kJ/kg)
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_enthalpy(fluid_lower, T, P)
        return round(result.value, 3)

    def get_entropy(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get specific entropy for a fluid.

        Calculated by integrating Cp/T from reference temperature.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Specific entropy (kJ/(kg*K))
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_entropy(fluid_lower, T, P)
        return round(result.value, 4)

    def get_viscosity(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get dynamic viscosity for a fluid.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Dynamic viscosity (mPa*s = cP)
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_viscosity(fluid_lower, T, P)
        return round(result.value, 4)

    def get_conductivity(
        self,
        fluid: str,
        T: float,
        P: float = 101.325
    ) -> float:
        """
        Get thermal conductivity for a fluid.

        Args:
            fluid: Fluid name
            T: Temperature (K)
            P: Pressure (kPa)

        Returns:
            Thermal conductivity (W/(m*K))
        """
        fluid_lower = fluid.lower()
        result = self.correlations.get_conductivity(fluid_lower, T, P)
        return round(result.value, 4)

    def recommend_fluid(
        self,
        T_range: Tuple[float, float],
        application: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[FluidRecommendation]:
        """
        Recommend fluids for a specific application.

        Analyzes temperature requirements, application type, and constraints
        to recommend suitable thermal fluids.

        Args:
            T_range: Operating temperature range (T_min, T_max) in K
            application: Application type:
                - 'solar_thermal': Concentrated solar power
                - 'process_heating': Industrial process heating
                - 'hvac': HVAC systems
                - 'food_grade': Food-safe applications
                - 'general': General heat transfer
            constraints: Additional constraints:
                - 'max_viscosity': Maximum viscosity (mPa*s)
                - 'min_conductivity': Minimum conductivity (W/(m*K))
                - 'non_toxic': Require non-toxic fluid
                - 'max_cost': Cost constraint

        Returns:
            List of FluidRecommendation sorted by suitability score

        Example:
            >>> library = ThermalFluidLibrary()
            >>> recommendations = library.recommend_fluid(
            ...     T_range=(473.15, 623.15),  # 200-350 C
            ...     application='solar_thermal'
            ... )
            >>> print(recommendations[0].fluid_name)
        """
        T_min, T_max = T_range
        constraints = constraints or {}
        recommendations = []

        # Application-specific preferences
        app_preferences = {
            "solar_thermal": {
                "preferred_categories": [FluidCategory.THERMINOL, FluidCategory.DOWTHERM, FluidCategory.MOLTEN_SALT],
                "min_temp": 473.15,
                "max_temp": 873.15,
            },
            "process_heating": {
                "preferred_categories": [FluidCategory.THERMINOL, FluidCategory.DOWTHERM, FluidCategory.MINERAL_OIL],
                "min_temp": 373.15,
                "max_temp": 673.15,
            },
            "hvac": {
                "preferred_categories": [FluidCategory.WATER_STEAM, FluidCategory.ETHYLENE_GLYCOL, FluidCategory.PROPYLENE_GLYCOL],
                "min_temp": 253.15,
                "max_temp": 383.15,
            },
            "food_grade": {
                "preferred_categories": [FluidCategory.PROPYLENE_GLYCOL, FluidCategory.WATER_STEAM],
                "min_temp": 253.15,
                "max_temp": 373.15,
            },
            "general": {
                "preferred_categories": list(FluidCategory),
                "min_temp": 233.15,
                "max_temp": 873.15,
            },
        }

        prefs = app_preferences.get(application.lower(), app_preferences["general"])

        # Evaluate each fluid
        for fluid_name, (category, fluid_T_min, fluid_T_max, P_max, source) in self.FLUID_METADATA.items():
            # Check temperature range compatibility
            if fluid_T_min > T_min or fluid_T_max < T_max:
                continue  # Fluid cannot cover required range

            # Calculate suitability score
            score = 0.0
            reasons = []
            limitations = []

            # Temperature range margin (higher is better)
            temp_margin = min(T_min - fluid_T_min, fluid_T_max - T_max)
            score += min(temp_margin / 50, 20)  # Up to 20 points for margin
            if temp_margin > 50:
                reasons.append(f"Good temperature margin ({temp_margin:.0f} K)")

            # Category preference
            if category in prefs["preferred_categories"]:
                score += 30
                reasons.append(f"Preferred category for {application}")

            # Property evaluation at mid-range temperature
            T_mid = (T_min + T_max) / 2
            try:
                props = self.get_properties(fluid_name, T_mid)

                # Viscosity (lower is better for pumping)
                if props.viscosity_mPa_s < 5:
                    score += 15
                    reasons.append("Low viscosity - easy pumping")
                elif props.viscosity_mPa_s > 50:
                    limitations.append("High viscosity may require larger pumps")

                # Thermal conductivity (higher is better)
                if props.conductivity_W_m_K > 0.12:
                    score += 10
                    reasons.append("Good thermal conductivity")

                # Specific heat (higher is better for storage)
                if props.specific_heat_kJ_kg_K > 2.0:
                    score += 10
                    reasons.append("High heat capacity")

                # Check constraints
                if constraints.get("max_viscosity") and props.viscosity_mPa_s > constraints["max_viscosity"]:
                    score -= 20
                    limitations.append(f"Viscosity exceeds constraint")

                if constraints.get("min_conductivity") and props.conductivity_W_m_K < constraints["min_conductivity"]:
                    score -= 10
                    limitations.append(f"Conductivity below requirement")

            except Exception:
                score -= 10
                limitations.append("Property calculation failed at mid-range")

            # Special considerations
            if category == FluidCategory.MOLTEN_SALT:
                limitations.append("Requires freeze protection")
                if application == "solar_thermal" and T_max > 773.15:
                    score += 20
                    reasons.append("Excellent for high-temperature solar thermal")

            if category == FluidCategory.PROPYLENE_GLYCOL:
                score += 10 if constraints.get("non_toxic") else 0
                reasons.append("Non-toxic, food-grade compatible")

            # Determine alternatives
            alternatives = self._find_alternatives(fluid_name, category)

            # Get typical applications
            typical_apps = self._get_typical_applications(category)

            # Create recommendation
            if score > 0:
                rec = FluidRecommendation(
                    fluid_name=fluid_name,
                    category=category,
                    score=min(score, 100),
                    reasons=reasons,
                    limitations=limitations,
                    alternatives=alternatives,
                    temperature_range_K=(fluid_T_min, fluid_T_max),
                    max_pressure_kPa=P_max,
                    typical_applications=typical_apps
                )
                recommendations.append(rec)

        # Sort by score (descending)
        recommendations.sort(key=lambda x: x.score, reverse=True)

        return recommendations[:10]  # Return top 10

    def list_fluids(self, category: Optional[FluidCategory] = None) -> List[str]:
        """
        List available fluids, optionally filtered by category.

        Args:
            category: Filter by category (optional)

        Returns:
            List of fluid names
        """
        fluids = []
        for name, (cat, *_) in self.FLUID_METADATA.items():
            if category is None or cat == category:
                fluids.append(name)
        return sorted(fluids)

    def get_fluid_info(self, fluid_name: str) -> Dict[str, Any]:
        """
        Get metadata and valid ranges for a fluid.

        Args:
            fluid_name: Name of the fluid

        Returns:
            Dictionary with fluid information
        """
        fluid_lower = fluid_name.lower()
        if fluid_lower not in self.FLUID_METADATA:
            raise ValueError(f"Unknown fluid: {fluid_name}")

        category, T_min, T_max, P_max, source = self.FLUID_METADATA[fluid_lower]

        return {
            "name": fluid_lower,
            "category": category.value,
            "temperature_range_K": (T_min, T_max),
            "temperature_range_C": (T_min - 273.15, T_max - 273.15),
            "max_pressure_kPa": P_max,
            "data_source": source,
            "typical_applications": self._get_typical_applications(category)
        }

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _get_uncertainty(self, fluid_name: str) -> float:
        """Get typical uncertainty for fluid property correlations."""
        category, *_ = self.FLUID_METADATA.get(fluid_name, (None,))

        uncertainties = {
            FluidCategory.WATER_STEAM: 0.1,       # IAPWS-IF97 very accurate
            FluidCategory.THERMINOL: 2.0,          # Manufacturer data
            FluidCategory.DOWTHERM: 2.0,           # Manufacturer data
            FluidCategory.SYLTHERM: 2.0,           # Manufacturer data
            FluidCategory.ETHYLENE_GLYCOL: 1.5,    # ASHRAE data
            FluidCategory.PROPYLENE_GLYCOL: 1.5,   # ASHRAE data
            FluidCategory.MOLTEN_SALT: 3.0,        # Less well-characterized
            FluidCategory.MINERAL_OIL: 5.0,        # Generic
            FluidCategory.SUPERCRITICAL: 1.0,      # NIST data
        }

        return uncertainties.get(category, 2.0)

    def _find_alternatives(self, fluid_name: str, category: FluidCategory) -> List[str]:
        """Find alternative fluids in the same category."""
        alternatives = []
        for name, (cat, *_) in self.FLUID_METADATA.items():
            if cat == category and name != fluid_name:
                alternatives.append(name)
        return alternatives[:3]

    def _get_typical_applications(self, category: FluidCategory) -> List[str]:
        """Get typical applications for a fluid category."""
        applications = {
            FluidCategory.WATER_STEAM: [
                "Power generation", "Process heating", "HVAC",
                "Steam turbines", "Boilers"
            ],
            FluidCategory.THERMINOL: [
                "Chemical processing", "Oil refining", "Solar thermal",
                "Pharmaceuticals", "Food processing"
            ],
            FluidCategory.DOWTHERM: [
                "Chemical processing", "Polymer production",
                "Heat transfer systems", "Solar thermal"
            ],
            FluidCategory.SYLTHERM: [
                "Semiconductor manufacturing", "Low-temperature applications",
                "Clean room processes", "High-temperature systems"
            ],
            FluidCategory.ETHYLENE_GLYCOL: [
                "HVAC systems", "Industrial cooling",
                "Chilled water systems", "Freeze protection"
            ],
            FluidCategory.PROPYLENE_GLYCOL: [
                "Food processing", "Beverage industry",
                "Pharmaceuticals", "HVAC (food-safe)"
            ],
            FluidCategory.MOLTEN_SALT: [
                "Concentrated solar power", "Thermal energy storage",
                "High-temperature processes"
            ],
            FluidCategory.MINERAL_OIL: [
                "Transformers", "Industrial heating",
                "Hydraulic systems"
            ],
            FluidCategory.SUPERCRITICAL: [
                "Next-gen power cycles", "sCO2 Brayton cycles",
                "Research applications"
            ],
        }
        return applications.get(category, ["General heat transfer"])

    def _calculate_provenance_hash(
        self,
        fluid_name: str,
        T: float,
        P: float,
        properties: Dict[str, float]
    ) -> str:
        """Calculate SHA-256 provenance hash for property calculation."""
        provenance_data = {
            "fluid": fluid_name,
            "T": T,
            "P": P,
            "properties": properties
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()
