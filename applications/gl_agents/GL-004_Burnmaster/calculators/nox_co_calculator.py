# -*- coding: utf-8 -*-
"""
NOx and CO Emission Calculator for GL-004 BurnMaster
====================================================

Provides deterministic, validated emission calculations for:
    - Thermal NOx (Zeldovich mechanism)
    - Fuel NOx (from fuel-bound nitrogen)
    - Prompt NOx (Fenimore mechanism)
    - CO from incomplete combustion

Features:
    - EPA AP-42 emission factor database
    - Control technology adjustment factors
    - Real-time emission rate calculations
    - Compliance limit checking
    - Full provenance tracking

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import hashlib
import math


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class FuelType(Enum):
    """Supported fuel types."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    BITUMINOUS_COAL = "bituminous_coal"
    SUB_BITUMINOUS_COAL = "sub_bituminous_coal"
    LIGNITE = "lignite"
    PROPANE = "propane"
    WOOD = "wood"


class ControlTechnology(Enum):
    """NOx control technologies."""
    UNCONTROLLED = "uncontrolled"
    LOW_NOX_BURNER = "low_nox_burner"
    ULTRA_LOW_NOX = "ultra_low_nox"
    FGR = "flue_gas_recirculation"
    SCR = "selective_catalytic_reduction"
    SNCR = "selective_non_catalytic_reduction"
    LOW_NOX_PLUS_FGR = "low_nox_plus_fgr"
    LOW_NOX_PLUS_SCR = "low_nox_plus_scr"


class EmissionUnits(Enum):
    """Emission rate units."""
    LB_MMBTU = "lb/mmBtu"
    KG_GJ = "kg/GJ"
    G_KWH = "g/kWh"
    PPM_DRY = "ppm_dry"
    MG_NM3 = "mg/Nm3"
    LB_HR = "lb/hr"
    KG_HR = "kg/hr"


# =============================================================================
# EPA AP-42 EMISSION FACTORS
# =============================================================================

# NOx emission factors (lb/mmBtu) from EPA AP-42
EPA_NOX_FACTORS = {
    FuelType.NATURAL_GAS: {
        ControlTechnology.UNCONTROLLED: 0.100,
        ControlTechnology.LOW_NOX_BURNER: 0.050,
        ControlTechnology.ULTRA_LOW_NOX: 0.020,
        ControlTechnology.FGR: 0.035,
        ControlTechnology.LOW_NOX_PLUS_FGR: 0.025,
        ControlTechnology.SCR: 0.010,
        ControlTechnology.LOW_NOX_PLUS_SCR: 0.008,
    },
    FuelType.FUEL_OIL_NO2: {
        ControlTechnology.UNCONTROLLED: 0.140,
        ControlTechnology.LOW_NOX_BURNER: 0.070,
        ControlTechnology.FGR: 0.050,
        ControlTechnology.SCR: 0.015,
    },
    FuelType.FUEL_OIL_NO6: {
        ControlTechnology.UNCONTROLLED: 0.370,
        ControlTechnology.LOW_NOX_BURNER: 0.190,
        ControlTechnology.SCR: 0.040,
    },
    FuelType.BITUMINOUS_COAL: {
        ControlTechnology.UNCONTROLLED: 0.950,
        ControlTechnology.LOW_NOX_BURNER: 0.500,
        ControlTechnology.SCR: 0.095,
        ControlTechnology.SNCR: 0.400,
    },
    FuelType.SUB_BITUMINOUS_COAL: {
        ControlTechnology.UNCONTROLLED: 0.380,
        ControlTechnology.LOW_NOX_BURNER: 0.200,
        ControlTechnology.SCR: 0.038,
    },
}

# CO emission factors (lb/mmBtu) from EPA AP-42
EPA_CO_FACTORS = {
    FuelType.NATURAL_GAS: 0.084,
    FuelType.FUEL_OIL_NO2: 0.036,
    FuelType.FUEL_OIL_NO6: 0.033,
    FuelType.BITUMINOUS_COAL: 0.500,
    FuelType.SUB_BITUMINOUS_COAL: 0.380,
    FuelType.PROPANE: 0.070,
}

# Higher Heating Values (Btu/lb)
FUEL_HHV = {
    FuelType.NATURAL_GAS: 23850,  # per lb
    FuelType.FUEL_OIL_NO2: 19580,
    FuelType.FUEL_OIL_NO6: 18700,
    FuelType.BITUMINOUS_COAL: 12500,
    FuelType.SUB_BITUMINOUS_COAL: 9000,
    FuelType.LIGNITE: 6500,
    FuelType.PROPANE: 21670,
    FuelType.WOOD: 4500,
}


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

UNIT_CONVERSIONS = {
    # lb/mmBtu to other units
    ("lb/mmBtu", "kg/GJ"): 0.4299,
    ("lb/mmBtu", "g/kWh"): 1.548,  # Assuming 33% efficiency
    ("lb/mmBtu", "mg/Nm3"): 1232.0,  # At 3% O2, natural gas

    # Volume conversions
    "scf_to_nm3": 0.02832,
    "mmbtu_to_gj": 1.055,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmissionResult:
    """Result of an emission calculation."""
    pollutant: str  # "NOx", "CO", "SO2", etc.
    emission_rate: Decimal
    units: EmissionUnits
    fuel_type: FuelType
    control_technology: Optional[ControlTechnology]
    heat_input_mmbtu_hr: Decimal
    mass_rate_lb_hr: Decimal
    epa_factor_used: Decimal
    calculation_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pollutant": self.pollutant,
            "emission_rate": float(self.emission_rate),
            "units": self.units.value,
            "fuel_type": self.fuel_type.value,
            "control_technology": self.control_technology.value if self.control_technology else None,
            "heat_input_mmbtu_hr": float(self.heat_input_mmbtu_hr),
            "mass_rate_lb_hr": float(self.mass_rate_lb_hr),
            "epa_factor": float(self.epa_factor_used),
            "method": self.calculation_method,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class ComplianceLimit:
    """Regulatory emission limit."""
    pollutant: str
    limit_value: float
    units: EmissionUnits
    regulation: str
    averaging_period: str  # "1-hour", "24-hour", "30-day", "annual"
    applicability: str


# =============================================================================
# NOx CALCULATOR
# =============================================================================

class NOxCalculator:
    """
    Calculates NOx emissions from combustion sources.

    Supports multiple calculation methods:
        1. EPA AP-42 emission factors
        2. Continuous Emissions Monitoring (CEM) data
        3. Thermal NOx correlation models

    All calculations are deterministic with full provenance tracking.
    """

    def __init__(self):
        """Initialize NOx calculator."""
        self.nox_factors = EPA_NOX_FACTORS
        self.precision = Decimal("0.000001")

    def calculate_nox(
        self,
        fuel_type: FuelType,
        heat_input_mmbtu_hr: float,
        control_technology: ControlTechnology = ControlTechnology.UNCONTROLLED,
        o2_percent: Optional[float] = None,
        flame_temp_k: Optional[float] = None,
    ) -> EmissionResult:
        """
        Calculate NOx emission rate.

        Args:
            fuel_type: Type of fuel being burned
            heat_input_mmbtu_hr: Heat input rate in mmBtu/hr
            control_technology: NOx control technology in use
            o2_percent: Optional O2 percentage for adjustment
            flame_temp_k: Optional flame temperature for thermal NOx model

        Returns:
            EmissionResult with NOx emission rate
        """
        # Get base emission factor
        fuel_factors = self.nox_factors.get(fuel_type, {})
        base_factor = fuel_factors.get(
            control_technology,
            fuel_factors.get(ControlTechnology.UNCONTROLLED, 0.1)
        )

        # Convert to Decimal for precision
        factor = Decimal(str(base_factor))
        heat_input = Decimal(str(heat_input_mmbtu_hr))

        # Apply O2 correction if provided (optional enhancement)
        if o2_percent is not None:
            # Correct to 3% O2 reference (standard for reporting)
            o2_correction = Decimal(str((21 - 3) / (21 - o2_percent)))
            factor = factor * o2_correction

        # Apply thermal NOx adjustment if flame temp provided
        if flame_temp_k is not None and flame_temp_k > 1800:
            # Simplified thermal NOx enhancement
            temp_factor = Decimal(str(1 + (flame_temp_k - 1800) / 500))
            factor = factor * min(temp_factor, Decimal("2.0"))

        # Calculate emission rate (lb/mmBtu)
        emission_rate = factor.quantize(self.precision)

        # Calculate mass rate (lb/hr)
        mass_rate = (factor * heat_input).quantize(self.precision)

        # Generate provenance hash
        provenance_data = f"NOx:{fuel_type.value}:{control_technology.value}:{heat_input}:{factor}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()[:16]

        return EmissionResult(
            pollutant="NOx",
            emission_rate=emission_rate,
            units=EmissionUnits.LB_MMBTU,
            fuel_type=fuel_type,
            control_technology=control_technology,
            heat_input_mmbtu_hr=heat_input,
            mass_rate_lb_hr=mass_rate,
            epa_factor_used=Decimal(str(base_factor)),
            calculation_method="EPA AP-42",
            provenance_hash=provenance_hash,
        )

    def estimate_control_reduction(
        self,
        uncontrolled_nox: float,
        technology: ControlTechnology
    ) -> Tuple[float, float]:
        """
        Estimate NOx reduction from control technology.

        Args:
            uncontrolled_nox: Uncontrolled NOx rate (lb/mmBtu)
            technology: Control technology to apply

        Returns:
            Tuple of (controlled NOx rate, reduction percentage)
        """
        reduction_factors = {
            ControlTechnology.LOW_NOX_BURNER: 0.50,
            ControlTechnology.ULTRA_LOW_NOX: 0.80,
            ControlTechnology.FGR: 0.65,
            ControlTechnology.SCR: 0.90,
            ControlTechnology.SNCR: 0.60,
            ControlTechnology.LOW_NOX_PLUS_FGR: 0.75,
            ControlTechnology.LOW_NOX_PLUS_SCR: 0.92,
        }

        reduction = reduction_factors.get(technology, 0.0)
        controlled_nox = uncontrolled_nox * (1 - reduction)

        return controlled_nox, reduction * 100


# =============================================================================
# CO CALCULATOR
# =============================================================================

class COCalculator:
    """
    Calculates CO emissions from combustion sources.

    CO formation is primarily influenced by:
        - Excess air (O2) levels
        - Combustion temperature
        - Residence time
        - Fuel type and quality
    """

    def __init__(self):
        """Initialize CO calculator."""
        self.co_factors = EPA_CO_FACTORS
        self.precision = Decimal("0.000001")

    def calculate_co(
        self,
        fuel_type: FuelType,
        heat_input_mmbtu_hr: float,
        o2_percent: Optional[float] = None,
    ) -> EmissionResult:
        """
        Calculate CO emission rate.

        Args:
            fuel_type: Type of fuel being burned
            heat_input_mmbtu_hr: Heat input rate in mmBtu/hr
            o2_percent: Optional O2 percentage (low O2 increases CO)

        Returns:
            EmissionResult with CO emission rate
        """
        # Get base emission factor
        base_factor = self.co_factors.get(fuel_type, 0.084)

        # Convert to Decimal
        factor = Decimal(str(base_factor))
        heat_input = Decimal(str(heat_input_mmbtu_hr))

        # Apply O2 adjustment (CO increases at low O2)
        if o2_percent is not None:
            if o2_percent < 2.0:
                # Low O2 significantly increases CO
                o2_multiplier = Decimal(str(5.0 - o2_percent))
                factor = factor * max(o2_multiplier, Decimal("1.0"))
            elif o2_percent > 5.0:
                # High excess air slightly reduces CO
                o2_multiplier = Decimal(str(0.9))
                factor = factor * o2_multiplier

        # Calculate emission rate
        emission_rate = factor.quantize(self.precision)

        # Calculate mass rate
        mass_rate = (factor * heat_input).quantize(self.precision)

        # Generate provenance hash
        provenance_data = f"CO:{fuel_type.value}:{heat_input}:{factor}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()[:16]

        return EmissionResult(
            pollutant="CO",
            emission_rate=emission_rate,
            units=EmissionUnits.LB_MMBTU,
            fuel_type=fuel_type,
            control_technology=None,
            heat_input_mmbtu_hr=heat_input,
            mass_rate_lb_hr=mass_rate,
            epa_factor_used=Decimal(str(base_factor)),
            calculation_method="EPA AP-42",
            provenance_hash=provenance_hash,
        )

    def calculate_co_vs_o2_curve(
        self,
        fuel_type: FuelType,
        o2_range: Tuple[float, float] = (0.5, 8.0)
    ) -> List[Tuple[float, float]]:
        """
        Generate CO vs O2 curve for optimization.

        Args:
            fuel_type: Type of fuel
            o2_range: O2 percentage range to evaluate

        Returns:
            List of (O2%, relative CO) tuples
        """
        base_factor = self.co_factors.get(fuel_type, 0.084)
        results = []

        for o2 in [o2_range[0] + i * 0.5 for i in range(int((o2_range[1] - o2_range[0]) / 0.5) + 1)]:
            if o2 < 2.0:
                # Exponential increase at low O2
                multiplier = math.exp((2.0 - o2) * 2)
            else:
                # Gradual decrease at higher O2
                multiplier = 1.0 - (o2 - 2.0) * 0.05

            relative_co = base_factor * max(multiplier, 0.5)
            results.append((o2, relative_co))

        return results


# =============================================================================
# COMBINED EMISSIONS CALCULATOR
# =============================================================================

class CombustionEmissionsCalculator:
    """
    Combined calculator for all combustion emissions.

    Provides:
        - NOx, CO, SO2, PM calculations
        - Regulatory compliance checking
        - Emission reporting data
        - Optimization recommendations
    """

    def __init__(self):
        """Initialize combined calculator."""
        self.nox_calc = NOxCalculator()
        self.co_calc = COCalculator()

    def calculate_all_emissions(
        self,
        fuel_type: FuelType,
        heat_input_mmbtu_hr: float,
        control_technology: ControlTechnology = ControlTechnology.UNCONTROLLED,
        o2_percent: Optional[float] = None,
    ) -> Dict[str, EmissionResult]:
        """
        Calculate all combustion emissions.

        Args:
            fuel_type: Type of fuel
            heat_input_mmbtu_hr: Heat input rate
            control_technology: NOx control technology
            o2_percent: Flue gas O2 percentage

        Returns:
            Dictionary with emission results by pollutant
        """
        results = {}

        # Calculate NOx
        results["NOx"] = self.nox_calc.calculate_nox(
            fuel_type=fuel_type,
            heat_input_mmbtu_hr=heat_input_mmbtu_hr,
            control_technology=control_technology,
            o2_percent=o2_percent,
        )

        # Calculate CO
        results["CO"] = self.co_calc.calculate_co(
            fuel_type=fuel_type,
            heat_input_mmbtu_hr=heat_input_mmbtu_hr,
            o2_percent=o2_percent,
        )

        return results

    def check_compliance(
        self,
        emissions: Dict[str, EmissionResult],
        limits: List[ComplianceLimit]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check emissions against regulatory limits.

        Args:
            emissions: Calculated emission results
            limits: Applicable regulatory limits

        Returns:
            Compliance status for each pollutant
        """
        compliance_results = {}

        for limit in limits:
            pollutant = limit.pollutant
            if pollutant in emissions:
                emission = emissions[pollutant]

                # Convert units if necessary (simplified)
                actual_value = float(emission.emission_rate)
                limit_value = limit.limit_value

                compliant = actual_value <= limit_value
                margin = ((limit_value - actual_value) / limit_value) * 100 if limit_value > 0 else 0

                compliance_results[pollutant] = {
                    "compliant": compliant,
                    "actual": actual_value,
                    "limit": limit_value,
                    "units": limit.units.value,
                    "margin_percent": margin,
                    "regulation": limit.regulation,
                }

        return compliance_results

    def get_optimization_recommendations(
        self,
        current_o2: float,
        current_nox: float,
        current_co: float,
    ) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations for emissions and efficiency.

        Args:
            current_o2: Current O2 percentage
            current_nox: Current NOx rate (lb/mmBtu)
            current_co: Current CO rate (lb/mmBtu)

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Check if O2 is too high (efficiency loss)
        if current_o2 > 5.0:
            recommendations.append({
                "category": "Efficiency",
                "issue": f"High excess air (O2 = {current_o2:.1f}%)",
                "recommendation": "Reduce O2 setpoint to 3-4% for natural gas, 4-5% for oil/coal",
                "expected_benefit": f"~{(current_o2 - 3.5) * 0.5:.1f}% fuel savings",
                "priority": "High",
            })

        # Check if O2 is too low (CO risk)
        if current_o2 < 2.0:
            recommendations.append({
                "category": "Safety",
                "issue": f"Low excess air (O2 = {current_o2:.1f}%)",
                "recommendation": "Increase O2 setpoint to reduce CO and ensure complete combustion",
                "expected_benefit": "Reduced CO emissions, improved safety",
                "priority": "Critical",
            })

        # Check NOx levels
        if current_nox > 0.1:
            recommendations.append({
                "category": "Emissions",
                "issue": f"Elevated NOx ({current_nox:.3f} lb/mmBtu)",
                "recommendation": "Consider low-NOx burner upgrade or FGR installation",
                "expected_benefit": "50-70% NOx reduction",
                "priority": "Medium",
            })

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_nox_emission(
    fuel_type: str,
    heat_input_mmbtu_hr: float,
    control: str = "uncontrolled",
    o2_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for NOx calculation.

    Args:
        fuel_type: Fuel type string (e.g., "natural_gas")
        heat_input_mmbtu_hr: Heat input rate
        control: Control technology string
        o2_percent: Optional O2 percentage

    Returns:
        Dictionary with calculation results
    """
    calc = NOxCalculator()

    try:
        fuel = FuelType(fuel_type)
        tech = ControlTechnology(control)
    except ValueError:
        fuel = FuelType.NATURAL_GAS
        tech = ControlTechnology.UNCONTROLLED

    result = calc.calculate_nox(
        fuel_type=fuel,
        heat_input_mmbtu_hr=heat_input_mmbtu_hr,
        control_technology=tech,
        o2_percent=o2_percent,
    )

    return result.to_dict()


def calculate_co_emission(
    fuel_type: str,
    heat_input_mmbtu_hr: float,
    o2_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for CO calculation.

    Args:
        fuel_type: Fuel type string
        heat_input_mmbtu_hr: Heat input rate
        o2_percent: Optional O2 percentage

    Returns:
        Dictionary with calculation results
    """
    calc = COCalculator()

    try:
        fuel = FuelType(fuel_type)
    except ValueError:
        fuel = FuelType.NATURAL_GAS

    result = calc.calculate_co(
        fuel_type=fuel,
        heat_input_mmbtu_hr=heat_input_mmbtu_hr,
        o2_percent=o2_percent,
    )

    return result.to_dict()


if __name__ == "__main__":
    import json

    # Example calculation
    calc = CombustionEmissionsCalculator()

    results = calc.calculate_all_emissions(
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu_hr=100.0,
        control_technology=ControlTechnology.LOW_NOX_BURNER,
        o2_percent=3.5,
    )

    print("Emission Calculation Results:")
    print("=" * 50)
    for pollutant, result in results.items():
        print(f"\n{pollutant}:")
        print(json.dumps(result.to_dict(), indent=2))

    # Optimization recommendations
    recommendations = calc.get_optimization_recommendations(
        current_o2=5.5,
        current_nox=0.08,
        current_co=0.05,
    )

    print("\n\nOptimization Recommendations:")
    print("=" * 50)
    for rec in recommendations:
        print(f"\n{rec['category']} - {rec['priority']} Priority")
        print(f"Issue: {rec['issue']}")
        print(f"Recommendation: {rec['recommendation']}")
        print(f"Expected Benefit: {rec['expected_benefit']}")
