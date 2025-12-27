"""
GreenLang Framework - Testing Utilities

Comprehensive testing utilities for GreenLang agents including:
- Golden value test fixtures from NIST/IAPWS/EPA reference data
- Property-based testing helpers with Hypothesis
- Test data generators for thermodynamic calculations
- Coverage reporting utilities

Target Coverage: 85%+ for all agents GL-001 through GL-016

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import json
import math
import random
import statistics
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, settings, strategies as st, assume, Phase
    from hypothesis.strategies import SearchStrategy
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    st = None
    given = None
    settings = None
    SearchStrategy = None

# Try to import pytest-cov for coverage
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


# =============================================================================
# Enums and Constants
# =============================================================================

class ReferenceDataSource(Enum):
    """Source of reference data for golden value testing."""
    NIST = "NIST"              # National Institute of Standards and Technology
    IAPWS = "IAPWS"            # International Association for Properties of Water and Steam
    EPA = "EPA"                # US Environmental Protection Agency
    IPCC = "IPCC"              # Intergovernmental Panel on Climate Change
    DEFRA = "DEFRA"            # UK Dept for Environment, Food & Rural Affairs
    ASHRAE = "ASHRAE"          # ASHRAE Handbook
    ISO = "ISO"                # ISO Standards
    CALCULATED = "CALCULATED"  # Analytically calculated reference


class ToleranceType(Enum):
    """Type of tolerance for golden value comparison."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    RELATIVE_PERCENT = "relative_percent"
    ULP = "ulp"  # Units in Last Place


@dataclass
class GoldenValue:
    """
    A golden reference value for testing.

    Attributes:
        name: Descriptive name for the test case
        inputs: Input parameters for the calculation
        expected_output: Expected result value(s)
        source: Reference data source
        tolerance: Acceptable deviation
        tolerance_type: How to interpret the tolerance
        metadata: Additional information about the test case
    """
    name: str
    inputs: Dict[str, Any]
    expected_output: Any
    source: ReferenceDataSource
    tolerance: float = 1e-6
    tolerance_type: ToleranceType = ToleranceType.RELATIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify(self, actual: Any) -> Tuple[bool, str]:
        """
        Verify actual value against expected.

        Returns:
            Tuple of (passed, message)
        """
        if isinstance(self.expected_output, dict):
            return self._verify_dict(actual)
        elif isinstance(self.expected_output, (list, tuple)):
            return self._verify_sequence(actual)
        else:
            return self._verify_scalar(actual)

    def _verify_scalar(self, actual: float) -> Tuple[bool, str]:
        """Verify a scalar value."""
        expected = float(self.expected_output)
        actual = float(actual)

        if self.tolerance_type == ToleranceType.ABSOLUTE:
            diff = abs(actual - expected)
            passed = diff <= self.tolerance
            msg = f"Expected {expected}, got {actual}, diff={diff}, tolerance={self.tolerance}"
        elif self.tolerance_type == ToleranceType.RELATIVE:
            if expected == 0:
                diff = abs(actual)
            else:
                diff = abs((actual - expected) / expected)
            passed = diff <= self.tolerance
            msg = f"Expected {expected}, got {actual}, rel_diff={diff:.6e}, tolerance={self.tolerance}"
        elif self.tolerance_type == ToleranceType.RELATIVE_PERCENT:
            if expected == 0:
                diff = abs(actual) * 100
            else:
                diff = abs((actual - expected) / expected) * 100
            passed = diff <= self.tolerance
            msg = f"Expected {expected}, got {actual}, diff={diff:.4f}%, tolerance={self.tolerance}%"
        elif self.tolerance_type == ToleranceType.ULP:
            # Units in Last Place comparison
            ulp_diff = self._ulp_diff(expected, actual)
            passed = ulp_diff <= self.tolerance
            msg = f"Expected {expected}, got {actual}, ULP diff={ulp_diff}, tolerance={self.tolerance}"
        else:
            passed = actual == expected
            msg = f"Expected {expected}, got {actual}"

        return passed, msg

    def _verify_dict(self, actual: Dict) -> Tuple[bool, str]:
        """Verify a dictionary of values."""
        if not isinstance(actual, dict):
            return False, f"Expected dict, got {type(actual)}"

        all_passed = True
        messages = []

        for key, expected_val in self.expected_output.items():
            if key not in actual:
                all_passed = False
                messages.append(f"Missing key: {key}")
                continue

            actual_val = actual[key]
            if isinstance(expected_val, (int, float)):
                passed, msg = self._verify_scalar(actual_val)
                if not passed:
                    all_passed = False
                    messages.append(f"{key}: {msg}")
            elif actual_val != expected_val:
                all_passed = False
                messages.append(f"{key}: expected {expected_val}, got {actual_val}")

        return all_passed, "; ".join(messages) if messages else "All values match"

    def _verify_sequence(self, actual: List) -> Tuple[bool, str]:
        """Verify a sequence of values."""
        if len(actual) != len(self.expected_output):
            return False, f"Length mismatch: expected {len(self.expected_output)}, got {len(actual)}"

        all_passed = True
        messages = []

        for i, (exp, act) in enumerate(zip(self.expected_output, actual)):
            if isinstance(exp, (int, float)):
                # Create temporary golden value for scalar comparison
                temp = GoldenValue(
                    name=f"{self.name}[{i}]",
                    inputs={},
                    expected_output=exp,
                    source=self.source,
                    tolerance=self.tolerance,
                    tolerance_type=self.tolerance_type
                )
                passed, msg = temp._verify_scalar(act)
                if not passed:
                    all_passed = False
                    messages.append(f"[{i}]: {msg}")

        return all_passed, "; ".join(messages) if messages else "All values match"

    @staticmethod
    def _ulp_diff(a: float, b: float) -> int:
        """Calculate difference in Units in Last Place."""
        if a == b:
            return 0
        if math.isnan(a) or math.isnan(b):
            return float('inf')
        if math.isinf(a) or math.isinf(b):
            return float('inf')

        # Convert to integer representation
        import struct
        a_int = struct.unpack('q', struct.pack('d', a))[0]
        b_int = struct.unpack('q', struct.pack('d', b))[0]

        # Handle sign
        if a_int < 0:
            a_int = 0x8000000000000000 - a_int
        if b_int < 0:
            b_int = 0x8000000000000000 - b_int

        return abs(a_int - b_int)


# =============================================================================
# NIST Reference Data
# =============================================================================

class NISTReferenceData:
    """
    NIST Standard Reference Database values for testing.

    Includes:
    - Thermophysical properties of fluids
    - Chemical thermodynamic properties
    - Physical constants

    Reference: NIST Chemistry WebBook, SRD 69
    """

    # Physical constants (CODATA 2018)
    PHYSICAL_CONSTANTS = {
        "gas_constant": GoldenValue(
            name="Universal Gas Constant (R)",
            inputs={},
            expected_output=8.314462618,
            source=ReferenceDataSource.NIST,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"unit": "J/(mol*K)", "year": 2019, "exact": True}
        ),
        "boltzmann": GoldenValue(
            name="Boltzmann Constant (k)",
            inputs={},
            expected_output=1.380649e-23,
            source=ReferenceDataSource.NIST,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"unit": "J/K", "year": 2019, "exact": True}
        ),
        "avogadro": GoldenValue(
            name="Avogadro Constant (NA)",
            inputs={},
            expected_output=6.02214076e23,
            source=ReferenceDataSource.NIST,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"unit": "1/mol", "year": 2019, "exact": True}
        ),
        "stefan_boltzmann": GoldenValue(
            name="Stefan-Boltzmann Constant",
            inputs={},
            expected_output=5.670374419e-8,
            source=ReferenceDataSource.NIST,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"unit": "W/(m^2*K^4)", "year": 2019, "exact": True}
        ),
        "planck": GoldenValue(
            name="Planck Constant (h)",
            inputs={},
            expected_output=6.62607015e-34,
            source=ReferenceDataSource.NIST,
            tolerance=0.0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"unit": "J*s", "year": 2019, "exact": True}
        ),
    }

    # Ideal gas properties
    IDEAL_GAS_TESTS = [
        GoldenValue(
            name="Ideal Gas Volume at STP",
            inputs={"n": 1.0, "T": 273.15, "P": 101325.0},
            expected_output=0.022414,  # m^3
            source=ReferenceDataSource.NIST,
            tolerance=1e-4,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"formula": "V = nRT/P", "unit": "m^3"}
        ),
        GoldenValue(
            name="Ideal Gas at 25C, 1 atm",
            inputs={"n": 1.0, "T": 298.15, "P": 101325.0},
            expected_output=0.024789,  # m^3
            source=ReferenceDataSource.NIST,
            tolerance=1e-4,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"formula": "V = nRT/P", "unit": "m^3"}
        ),
    ]

    # Heat capacity data for common gases (J/(mol*K) at 25C)
    HEAT_CAPACITIES = {
        "N2": GoldenValue(
            name="Nitrogen Cp at 25C",
            inputs={"gas": "N2", "T": 298.15},
            expected_output=29.124,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "J/(mol*K)"}
        ),
        "O2": GoldenValue(
            name="Oxygen Cp at 25C",
            inputs={"gas": "O2", "T": 298.15},
            expected_output=29.378,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "J/(mol*K)"}
        ),
        "CO2": GoldenValue(
            name="Carbon Dioxide Cp at 25C",
            inputs={"gas": "CO2", "T": 298.15},
            expected_output=37.135,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "J/(mol*K)"}
        ),
        "H2O_vapor": GoldenValue(
            name="Water Vapor Cp at 25C",
            inputs={"gas": "H2O", "T": 298.15},
            expected_output=33.577,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "J/(mol*K)"}
        ),
        "CH4": GoldenValue(
            name="Methane Cp at 25C",
            inputs={"gas": "CH4", "T": 298.15},
            expected_output=35.69,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "J/(mol*K)"}
        ),
    }

    # Standard enthalpies of formation (kJ/mol at 25C)
    FORMATION_ENTHALPIES = {
        "H2O_l": GoldenValue(
            name="Water (liquid) formation enthalpy",
            inputs={"compound": "H2O", "phase": "liquid"},
            expected_output=-285.83,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol", "T": 298.15}
        ),
        "H2O_g": GoldenValue(
            name="Water (gas) formation enthalpy",
            inputs={"compound": "H2O", "phase": "gas"},
            expected_output=-241.826,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol", "T": 298.15}
        ),
        "CO2": GoldenValue(
            name="Carbon Dioxide formation enthalpy",
            inputs={"compound": "CO2", "phase": "gas"},
            expected_output=-393.509,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol", "T": 298.15}
        ),
        "CH4": GoldenValue(
            name="Methane formation enthalpy",
            inputs={"compound": "CH4", "phase": "gas"},
            expected_output=-74.87,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol", "T": 298.15}
        ),
    }

    # Combustion enthalpies (kJ/mol)
    COMBUSTION_ENTHALPIES = {
        "CH4": GoldenValue(
            name="Methane combustion (HHV)",
            inputs={"fuel": "CH4", "products": "H2O(l)"},
            expected_output=-890.36,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol", "reaction": "CH4 + 2O2 -> CO2 + 2H2O(l)"}
        ),
        "C2H6": GoldenValue(
            name="Ethane combustion (HHV)",
            inputs={"fuel": "C2H6", "products": "H2O(l)"},
            expected_output=-1560.69,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol"}
        ),
        "C3H8": GoldenValue(
            name="Propane combustion (HHV)",
            inputs={"fuel": "C3H8", "products": "H2O(l)"},
            expected_output=-2219.17,
            source=ReferenceDataSource.NIST,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"unit": "kJ/mol"}
        ),
    }

    @classmethod
    def get_all_golden_values(cls) -> List[GoldenValue]:
        """Get all NIST golden values for testing."""
        values = []
        values.extend(cls.PHYSICAL_CONSTANTS.values())
        values.extend(cls.IDEAL_GAS_TESTS)
        values.extend(cls.HEAT_CAPACITIES.values())
        values.extend(cls.FORMATION_ENTHALPIES.values())
        values.extend(cls.COMBUSTION_ENTHALPIES.values())
        return values


# =============================================================================
# IAPWS Reference Data
# =============================================================================

class IAPWSReferenceData:
    """
    IAPWS (International Association for Properties of Water and Steam)
    reference data for steam table testing.

    Based on:
    - IAPWS-IF97: Industrial Formulation 1997
    - IAPWS-95: Formulation 1995 for scientific use

    Reference: http://www.iapws.org/
    """

    # Critical point properties
    CRITICAL_POINT = GoldenValue(
        name="Water Critical Point",
        inputs={},
        expected_output={
            "T_c": 647.096,      # K
            "P_c": 22.064e6,     # Pa
            "rho_c": 322.0,      # kg/m^3
        },
        source=ReferenceDataSource.IAPWS,
        tolerance=1e-6,
        tolerance_type=ToleranceType.RELATIVE,
        metadata={"standard": "IAPWS-95"}
    )

    # Triple point properties
    TRIPLE_POINT = GoldenValue(
        name="Water Triple Point",
        inputs={},
        expected_output={
            "T_t": 273.16,       # K
            "P_t": 611.657,      # Pa
        },
        source=ReferenceDataSource.IAPWS,
        tolerance=1e-6,
        tolerance_type=ToleranceType.RELATIVE,
        metadata={"standard": "IAPWS-95"}
    )

    # IF97 Region 1 verification (compressed liquid)
    IF97_REGION1_TESTS = [
        GoldenValue(
            name="IF97 Region 1 Test 1",
            inputs={"T": 300.0, "P": 3.0e6},  # K, Pa
            expected_output={
                "v": 0.00100215168e-2,  # m^3/kg (specific volume * 100)
                "h": 0.115331273e3,     # kJ/kg
                "s": 0.392294792,       # kJ/(kg*K)
                "cp": 0.417301218e1,    # kJ/(kg*K)
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 1, "standard": "IF97"}
        ),
        GoldenValue(
            name="IF97 Region 1 Test 2",
            inputs={"T": 300.0, "P": 80.0e6},
            expected_output={
                "v": 0.000971180894e-2,
                "h": 0.184142828e3,
                "s": 0.368563852,
                "cp": 0.401008987e1,
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 1, "standard": "IF97"}
        ),
        GoldenValue(
            name="IF97 Region 1 Test 3",
            inputs={"T": 500.0, "P": 3.0e6},
            expected_output={
                "v": 0.00120241800e-2,
                "h": 0.975542239e3,
                "s": 0.258041912e1,
                "cp": 0.465580682e1,
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 1, "standard": "IF97"}
        ),
    ]

    # IF97 Region 2 verification (superheated vapor)
    IF97_REGION2_TESTS = [
        GoldenValue(
            name="IF97 Region 2 Test 1",
            inputs={"T": 300.0, "P": 0.0035e6},
            expected_output={
                "v": 0.394913866e2,
                "h": 0.254991145e4,
                "s": 0.852238967e1,
                "cp": 0.191300162e1,
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 2, "standard": "IF97"}
        ),
        GoldenValue(
            name="IF97 Region 2 Test 2",
            inputs={"T": 700.0, "P": 0.0035e6},
            expected_output={
                "v": 0.923015898e2,
                "h": 0.333568375e4,
                "s": 0.101749996e2,
                "cp": 0.208141274e1,
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 2, "standard": "IF97"}
        ),
        GoldenValue(
            name="IF97 Region 2 Test 3",
            inputs={"T": 700.0, "P": 30.0e6},
            expected_output={
                "v": 0.00542946619e-2,
                "h": 0.263149474e4,
                "s": 0.517540298e1,
                "cp": 0.103505092e2,
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=1e-8,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"region": 2, "standard": "IF97"}
        ),
    ]

    # Saturation properties verification
    SATURATION_TESTS = [
        GoldenValue(
            name="Saturation at 100C",
            inputs={"T": 373.15},
            expected_output={
                "P_sat": 101325.0,      # Pa (approximately)
                "h_f": 419.04,          # kJ/kg
                "h_g": 2676.1,          # kJ/kg
                "h_fg": 2257.0,         # kJ/kg
                "s_f": 1.3069,          # kJ/(kg*K)
                "s_g": 7.3549,          # kJ/(kg*K)
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"condition": "normal boiling point"}
        ),
        GoldenValue(
            name="Saturation at 1 MPa",
            inputs={"P": 1.0e6},
            expected_output={
                "T_sat": 453.03,        # K
                "h_f": 762.79,          # kJ/kg
                "h_g": 2778.1,          # kJ/kg
                "s_f": 2.1386,          # kJ/(kg*K)
                "s_g": 6.5863,          # kJ/(kg*K)
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"condition": "1 MPa saturation"}
        ),
        GoldenValue(
            name="Saturation at 10 MPa",
            inputs={"P": 10.0e6},
            expected_output={
                "T_sat": 584.15,        # K
                "h_f": 1407.54,         # kJ/kg
                "h_g": 2724.7,          # kJ/kg
                "s_f": 3.3596,          # kJ/(kg*K)
                "s_g": 5.6141,          # kJ/(kg*K)
            },
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"condition": "10 MPa saturation"}
        ),
    ]

    # Water density at various conditions
    DENSITY_TESTS = [
        GoldenValue(
            name="Water density at 4C (max density)",
            inputs={"T": 277.15, "P": 101325.0},
            expected_output=999.972,  # kg/m^3
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"note": "Maximum density of water"}
        ),
        GoldenValue(
            name="Water density at 20C",
            inputs={"T": 293.15, "P": 101325.0},
            expected_output=998.204,  # kg/m^3
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
        ),
        GoldenValue(
            name="Water density at 25C",
            inputs={"T": 298.15, "P": 101325.0},
            expected_output=997.047,  # kg/m^3
            source=ReferenceDataSource.IAPWS,
            tolerance=0.001,
            tolerance_type=ToleranceType.RELATIVE,
        ),
    ]

    @classmethod
    def get_all_golden_values(cls) -> List[GoldenValue]:
        """Get all IAPWS golden values for testing."""
        values = [cls.CRITICAL_POINT, cls.TRIPLE_POINT]
        values.extend(cls.IF97_REGION1_TESTS)
        values.extend(cls.IF97_REGION2_TESTS)
        values.extend(cls.SATURATION_TESTS)
        values.extend(cls.DENSITY_TESTS)
        return values


# =============================================================================
# EPA Reference Data
# =============================================================================

class EPAReferenceData:
    """
    EPA reference data for emissions calculations.

    Sources:
    - EPA eGRID (Emissions & Generation Resource Integrated Database)
    - EPA GHG Emission Factors Hub
    - EPA AP-42 (Compilation of Air Pollutant Emission Factors)

    Reference: https://www.epa.gov/ghgemissions/ghg-emission-factors-hub
    """

    # Stationary combustion emission factors (kg CO2/unit)
    STATIONARY_COMBUSTION = {
        "natural_gas_per_scf": GoldenValue(
            name="Natural Gas CO2 per SCF",
            inputs={"fuel": "natural_gas", "unit": "scf"},
            expected_output=0.05444,  # kg CO2/scf
            source=ReferenceDataSource.EPA,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024, "table": "Table 1"}
        ),
        "natural_gas_per_mmbtu": GoldenValue(
            name="Natural Gas CO2 per MMBtu",
            inputs={"fuel": "natural_gas", "unit": "MMBtu"},
            expected_output=53.06,  # kg CO2/MMBtu
            source=ReferenceDataSource.EPA,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "diesel_per_gallon": GoldenValue(
            name="Diesel CO2 per gallon",
            inputs={"fuel": "diesel", "unit": "gallon"},
            expected_output=10.21,  # kg CO2/gallon
            source=ReferenceDataSource.EPA,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "gasoline_per_gallon": GoldenValue(
            name="Gasoline CO2 per gallon",
            inputs={"fuel": "gasoline", "unit": "gallon"},
            expected_output=8.78,  # kg CO2/gallon
            source=ReferenceDataSource.EPA,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "coal_bituminous_per_short_ton": GoldenValue(
            name="Bituminous Coal CO2 per short ton",
            inputs={"fuel": "coal_bituminous", "unit": "short_ton"},
            expected_output=2328.0,  # kg CO2/short ton
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "propane_per_gallon": GoldenValue(
            name="Propane CO2 per gallon",
            inputs={"fuel": "propane", "unit": "gallon"},
            expected_output=5.72,  # kg CO2/gallon
            source=ReferenceDataSource.EPA,
            tolerance=0.01,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
    }

    # Electricity grid emission factors by region (kg CO2e/MWh)
    EGRID_FACTORS_2022 = {
        "US_average": GoldenValue(
            name="US Average Grid Factor",
            inputs={"region": "US", "year": 2022},
            expected_output=386.6,  # kg CO2e/MWh
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"source": "eGRID 2022"}
        ),
        "CAMX": GoldenValue(
            name="WECC California Grid Factor",
            inputs={"region": "CAMX", "year": 2022},
            expected_output=225.2,  # kg CO2e/MWh
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"source": "eGRID 2022"}
        ),
        "RFCW": GoldenValue(
            name="RFC West Grid Factor",
            inputs={"region": "RFCW", "year": 2022},
            expected_output=452.8,  # kg CO2e/MWh
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"source": "eGRID 2022"}
        ),
        "SRSO": GoldenValue(
            name="SERC South Grid Factor",
            inputs={"region": "SRSO", "year": 2022},
            expected_output=381.9,  # kg CO2e/MWh
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"source": "eGRID 2022"}
        ),
        "ERCT": GoldenValue(
            name="ERCOT (Texas) Grid Factor",
            inputs={"region": "ERCT", "year": 2022},
            expected_output=382.8,  # kg CO2e/MWh
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"source": "eGRID 2022"}
        ),
    }

    # Global Warming Potentials (100-year, AR5)
    GWP_AR5 = {
        "CO2": GoldenValue(
            name="CO2 GWP",
            inputs={"gas": "CO2", "timeframe": 100},
            expected_output=1,
            source=ReferenceDataSource.EPA,
            tolerance=0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"IPCC": "AR5"}
        ),
        "CH4": GoldenValue(
            name="CH4 GWP (fossil)",
            inputs={"gas": "CH4", "timeframe": 100, "source": "fossil"},
            expected_output=30,
            source=ReferenceDataSource.EPA,
            tolerance=0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"IPCC": "AR5", "includes_feedback": True}
        ),
        "N2O": GoldenValue(
            name="N2O GWP",
            inputs={"gas": "N2O", "timeframe": 100},
            expected_output=265,
            source=ReferenceDataSource.EPA,
            tolerance=0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"IPCC": "AR5"}
        ),
        "SF6": GoldenValue(
            name="SF6 GWP",
            inputs={"gas": "SF6", "timeframe": 100},
            expected_output=23500,
            source=ReferenceDataSource.EPA,
            tolerance=0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"IPCC": "AR5"}
        ),
        "HFC134a": GoldenValue(
            name="HFC-134a GWP",
            inputs={"gas": "HFC-134a", "timeframe": 100},
            expected_output=1300,
            source=ReferenceDataSource.EPA,
            tolerance=0,
            tolerance_type=ToleranceType.ABSOLUTE,
            metadata={"IPCC": "AR5"}
        ),
    }

    # Mobile combustion factors (kg CO2/mile or kg CO2/gallon)
    MOBILE_COMBUSTION = {
        "passenger_car_gasoline": GoldenValue(
            name="Passenger Car Gasoline",
            inputs={"vehicle": "passenger_car", "fuel": "gasoline"},
            expected_output=0.404,  # kg CO2/mile
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "light_truck_gasoline": GoldenValue(
            name="Light-Duty Truck Gasoline",
            inputs={"vehicle": "light_truck", "fuel": "gasoline"},
            expected_output=0.548,  # kg CO2/mile
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
        "heavy_truck_diesel": GoldenValue(
            name="Heavy-Duty Truck Diesel",
            inputs={"vehicle": "heavy_truck", "fuel": "diesel"},
            expected_output=1.689,  # kg CO2/mile
            source=ReferenceDataSource.EPA,
            tolerance=0.02,
            tolerance_type=ToleranceType.RELATIVE,
            metadata={"year": 2024}
        ),
    }

    @classmethod
    def get_all_golden_values(cls) -> List[GoldenValue]:
        """Get all EPA golden values for testing."""
        values = []
        values.extend(cls.STATIONARY_COMBUSTION.values())
        values.extend(cls.EGRID_FACTORS_2022.values())
        values.extend(cls.GWP_AR5.values())
        values.extend(cls.MOBILE_COMBUSTION.values())
        return values


# =============================================================================
# Golden Value Fixtures
# =============================================================================

class GoldenValueFixtures:
    """
    Central repository for golden value test fixtures.

    Provides easy access to reference data from multiple authoritative sources
    for validating GreenLang agent calculations.
    """

    def __init__(self):
        """Initialize fixtures with all reference data."""
        self._fixtures: Dict[str, List[GoldenValue]] = {
            "nist": NISTReferenceData.get_all_golden_values(),
            "iapws": IAPWSReferenceData.get_all_golden_values(),
            "epa": EPAReferenceData.get_all_golden_values(),
        }

    def get_by_source(self, source: str) -> List[GoldenValue]:
        """Get all fixtures from a specific source."""
        return self._fixtures.get(source.lower(), [])

    def get_all(self) -> List[GoldenValue]:
        """Get all golden value fixtures."""
        all_values = []
        for values in self._fixtures.values():
            all_values.extend(values)
        return all_values

    def get_by_category(self, category: str) -> List[GoldenValue]:
        """
        Get fixtures by category.

        Categories: thermodynamic, emissions, combustion, steam, grid
        """
        category_map = {
            "thermodynamic": (
                NISTReferenceData.PHYSICAL_CONSTANTS.values(),
                NISTReferenceData.HEAT_CAPACITIES.values(),
            ),
            "steam": (
                IAPWSReferenceData.IF97_REGION1_TESTS,
                IAPWSReferenceData.IF97_REGION2_TESTS,
                IAPWSReferenceData.SATURATION_TESTS,
            ),
            "emissions": (
                EPAReferenceData.STATIONARY_COMBUSTION.values(),
                EPAReferenceData.MOBILE_COMBUSTION.values(),
            ),
            "combustion": (
                NISTReferenceData.COMBUSTION_ENTHALPIES.values(),
                NISTReferenceData.FORMATION_ENTHALPIES.values(),
            ),
            "grid": (
                EPAReferenceData.EGRID_FACTORS_2022.values(),
            ),
        }

        result = []
        for group in category_map.get(category.lower(), []):
            result.extend(group)
        return result

    def verify_calculation(
        self,
        calculation_fn: Callable,
        golden_value: GoldenValue,
    ) -> Tuple[bool, str]:
        """
        Verify a calculation function against a golden value.

        Args:
            calculation_fn: Function that takes inputs and returns result
            golden_value: Golden value to test against

        Returns:
            Tuple of (passed, message)
        """
        try:
            result = calculation_fn(**golden_value.inputs)
            return golden_value.verify(result)
        except Exception as e:
            return False, f"Exception during calculation: {str(e)}"


# =============================================================================
# Thermodynamic Test Data Generator
# =============================================================================

class ThermodynamicTestData:
    """
    Generator for thermodynamic test data.

    Creates realistic test cases for:
    - Heat transfer calculations
    - Steam systems
    - Combustion analysis
    - Energy balances
    """

    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @dataclass
    class HeatTransferCase:
        """Heat transfer test case."""
        name: str
        hot_inlet_temp: float  # K
        hot_outlet_temp: float  # K
        cold_inlet_temp: float  # K
        cold_outlet_temp: float  # K
        hot_flow_rate: float  # kg/s
        cold_flow_rate: float  # kg/s
        hot_cp: float  # J/(kg*K)
        cold_cp: float  # J/(kg*K)
        expected_duty: float  # W
        heat_exchanger_type: str

    @dataclass
    class CombustionCase:
        """Combustion test case."""
        name: str
        fuel_type: str
        fuel_flow_rate: float  # kg/s
        air_flow_rate: float  # kg/s
        excess_air_percent: float
        fuel_hhv: float  # J/kg
        expected_flue_gas_temp: float  # K
        expected_efficiency: float
        expected_co2_rate: float  # kg/s

    @dataclass
    class SteamSystemCase:
        """Steam system test case."""
        name: str
        boiler_pressure: float  # Pa
        boiler_temp: float  # K
        feedwater_temp: float  # K
        steam_flow_rate: float  # kg/s
        blowdown_rate: float  # fraction
        expected_duty: float  # W
        expected_efficiency: float

    def generate_heat_exchanger_cases(self, count: int = 10) -> List[HeatTransferCase]:
        """Generate heat exchanger test cases."""
        cases = []
        hx_types = ["shell_tube", "plate", "finned_tube", "double_pipe"]

        for i in range(count):
            # Generate consistent temperatures (hot > cold, outlets between inlets)
            t_hot_in = self.rng.uniform(400, 600)  # K
            t_cold_in = self.rng.uniform(280, 350)  # K
            approach = self.rng.uniform(10, 50)

            t_hot_out = self.rng.uniform(t_cold_in + approach, t_hot_in - 10)
            t_cold_out = self.rng.uniform(t_cold_in + 10, t_hot_out - approach)

            m_hot = self.rng.uniform(0.5, 10)  # kg/s
            cp_hot = self.rng.uniform(1800, 4200)  # J/(kg*K)

            # Calculate cold side flow for energy balance
            q_hot = m_hot * cp_hot * (t_hot_in - t_hot_out)
            cp_cold = self.rng.uniform(2000, 4200)
            m_cold = q_hot / (cp_cold * (t_cold_out - t_cold_in))

            cases.append(self.HeatTransferCase(
                name=f"HX_Case_{i+1}",
                hot_inlet_temp=t_hot_in,
                hot_outlet_temp=t_hot_out,
                cold_inlet_temp=t_cold_in,
                cold_outlet_temp=t_cold_out,
                hot_flow_rate=m_hot,
                cold_flow_rate=m_cold,
                hot_cp=cp_hot,
                cold_cp=cp_cold,
                expected_duty=q_hot,
                heat_exchanger_type=self.rng.choice(hx_types),
            ))

        return cases

    def generate_combustion_cases(self, count: int = 10) -> List[CombustionCase]:
        """Generate combustion calculation test cases."""
        cases = []
        fuels = {
            "natural_gas": {"hhv": 55.5e6, "carbon_frac": 0.75, "stoich_afr": 17.2},
            "diesel": {"hhv": 45.6e6, "carbon_frac": 0.87, "stoich_afr": 14.5},
            "coal": {"hhv": 27.0e6, "carbon_frac": 0.70, "stoich_afr": 11.5},
            "propane": {"hhv": 50.4e6, "carbon_frac": 0.82, "stoich_afr": 15.6},
        }

        for i in range(count):
            fuel_type = self.rng.choice(list(fuels.keys()))
            fuel_props = fuels[fuel_type]

            fuel_flow = self.rng.uniform(0.1, 5.0)  # kg/s
            excess_air = self.rng.uniform(5, 30)  # percent

            stoich_air = fuel_flow * fuel_props["stoich_afr"]
            actual_air = stoich_air * (1 + excess_air / 100)

            # Estimate efficiency and flue gas temp
            efficiency = self.rng.uniform(0.80, 0.95)
            flue_temp = 400 + (1 - efficiency) * 600  # K, rough estimate

            # CO2 production
            co2_rate = fuel_flow * fuel_props["carbon_frac"] * (44 / 12)  # kg/s

            cases.append(self.CombustionCase(
                name=f"Combustion_Case_{i+1}",
                fuel_type=fuel_type,
                fuel_flow_rate=fuel_flow,
                air_flow_rate=actual_air,
                excess_air_percent=excess_air,
                fuel_hhv=fuel_props["hhv"],
                expected_flue_gas_temp=flue_temp,
                expected_efficiency=efficiency,
                expected_co2_rate=co2_rate,
            ))

        return cases

    def generate_steam_system_cases(self, count: int = 10) -> List[SteamSystemCase]:
        """Generate steam system test cases."""
        cases = []

        for i in range(count):
            # Pressure and corresponding saturation temperature
            pressure = self.rng.uniform(0.5e6, 15e6)  # Pa
            # Approximate Tsat from pressure (simplified)
            t_sat = 373.15 + 40 * np.log10(pressure / 101325)
            boiler_temp = t_sat + self.rng.uniform(0, 100)  # Superheat

            feedwater_temp = self.rng.uniform(350, 420)  # K
            steam_flow = self.rng.uniform(1, 50)  # kg/s
            blowdown = self.rng.uniform(0.01, 0.05)  # fraction

            # Rough duty estimate (enthalpy rise)
            h_fw = 4.186 * (feedwater_temp - 273.15)  # kJ/kg approx
            h_steam = 2800 + 2 * (boiler_temp - t_sat)  # kJ/kg approx
            duty = steam_flow * (h_steam - h_fw) * 1000  # W

            efficiency = self.rng.uniform(0.82, 0.94)

            cases.append(self.SteamSystemCase(
                name=f"Steam_Case_{i+1}",
                boiler_pressure=pressure,
                boiler_temp=boiler_temp,
                feedwater_temp=feedwater_temp,
                steam_flow_rate=steam_flow,
                blowdown_rate=blowdown,
                expected_duty=duty,
                expected_efficiency=efficiency,
            ))

        return cases

    def generate_energy_balance_inputs(
        self,
        num_streams: int = 5,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Generate inputs for energy balance calculations."""
        hot_streams = []
        cold_streams = []

        for i in range(num_streams):
            # Hot streams (need cooling)
            t_in = self.rng.uniform(400, 600)
            t_out = self.rng.uniform(300, t_in - 20)
            mcp = self.rng.uniform(10, 100)  # kW/K

            hot_streams.append({
                "name": f"Hot_{i+1}",
                "T_in": t_in,
                "T_out": t_out,
                "mCp": mcp,
                "duty": mcp * (t_in - t_out),
            })

            # Cold streams (need heating)
            t_in = self.rng.uniform(280, 350)
            t_out = self.rng.uniform(t_in + 20, 500)
            mcp = self.rng.uniform(10, 100)

            cold_streams.append({
                "name": f"Cold_{i+1}",
                "T_in": t_in,
                "T_out": t_out,
                "mCp": mcp,
                "duty": mcp * (t_out - t_in),
            })

        return {
            "hot_streams": hot_streams,
            "cold_streams": cold_streams,
        }


# =============================================================================
# Property-Based Testing Helpers
# =============================================================================

class PropertyTestHelpers:
    """
    Helpers for property-based testing with Hypothesis.

    Provides strategies and invariant checkers for thermodynamic calculations.
    """

    @staticmethod
    def is_hypothesis_available() -> bool:
        """Check if Hypothesis is available."""
        return HYPOTHESIS_AVAILABLE

    # Hypothesis strategies for physical quantities
    @staticmethod
    def temperature_kelvin(
        min_val: float = 200.0,
        max_val: float = 2000.0,
    ) -> "SearchStrategy[float]":
        """Strategy for generating temperature values in Kelvin."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def temperature_celsius(
        min_val: float = -50.0,
        max_val: float = 1500.0,
    ) -> "SearchStrategy[float]":
        """Strategy for generating temperature values in Celsius."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def pressure_pa(
        min_val: float = 100.0,
        max_val: float = 100e6,
    ) -> "SearchStrategy[float]":
        """Strategy for generating pressure values in Pascal."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def mass_flow_rate(
        min_val: float = 0.001,
        max_val: float = 1000.0,
    ) -> "SearchStrategy[float]":
        """Strategy for generating mass flow rates in kg/s."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def efficiency(
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> "SearchStrategy[float]":
        """Strategy for generating efficiency values (0-1)."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def positive_float(
        min_val: float = 1e-10,
        max_val: float = 1e10,
    ) -> "SearchStrategy[float]":
        """Strategy for generating positive floats."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")
        return st.floats(min_value=min_val, max_value=max_val, allow_nan=False)

    @staticmethod
    def heat_exchanger_inputs() -> "SearchStrategy[Dict[str, float]]":
        """Strategy for generating heat exchanger input data."""
        if not HYPOTHESIS_AVAILABLE:
            raise ImportError("Hypothesis is required for property-based testing")

        return st.fixed_dictionaries({
            "hot_inlet_temp": st.floats(min_value=350, max_value=600),
            "cold_inlet_temp": st.floats(min_value=280, max_value=340),
            "hot_flow_rate": st.floats(min_value=0.1, max_value=100),
            "cold_flow_rate": st.floats(min_value=0.1, max_value=100),
            "hot_cp": st.floats(min_value=1000, max_value=5000),
            "cold_cp": st.floats(min_value=1000, max_value=5000),
            "effectiveness": st.floats(min_value=0.3, max_value=0.95),
        })

    # Invariant checkers
    @staticmethod
    def check_energy_conservation(
        q_in: float,
        q_out: float,
        q_loss: float = 0.0,
        tolerance: float = 1e-6,
    ) -> bool:
        """Check that energy is conserved (first law of thermodynamics)."""
        balance = abs(q_in - q_out - q_loss)
        return balance <= tolerance * max(abs(q_in), abs(q_out), 1.0)

    @staticmethod
    def check_second_law(
        entropy_generation: float,
        tolerance: float = -1e-10,
    ) -> bool:
        """Check that entropy generation is non-negative (second law)."""
        return entropy_generation >= tolerance

    @staticmethod
    def check_temperature_ordering(
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
    ) -> bool:
        """Check physically valid temperature ordering in heat exchanger."""
        return (
            t_hot_in >= t_hot_out and  # Hot stream cools down
            t_cold_out >= t_cold_in and  # Cold stream heats up
            t_hot_in >= t_cold_out and  # Hot inlet >= Cold outlet
            t_hot_out >= t_cold_in  # Hot outlet >= Cold inlet
        )

    @staticmethod
    def check_mass_conservation(
        mass_in: float,
        mass_out: float,
        mass_accumulated: float = 0.0,
        tolerance: float = 1e-6,
    ) -> bool:
        """Check mass conservation."""
        balance = abs(mass_in - mass_out - mass_accumulated)
        return balance <= tolerance * max(abs(mass_in), abs(mass_out), 1.0)

    @staticmethod
    def check_positive_quantities(*values: float) -> bool:
        """Check that all physical quantities are positive."""
        return all(v > 0 for v in values)

    @staticmethod
    def check_bounded_efficiency(efficiency: float) -> bool:
        """Check that efficiency is between 0 and 1."""
        return 0.0 <= efficiency <= 1.0


# =============================================================================
# Coverage Reporting Utilities
# =============================================================================

class CoverageReporter:
    """
    Coverage reporting utilities for GreenLang agents.

    Provides tools for:
    - Measuring test coverage
    - Generating coverage reports
    - Checking coverage thresholds
    """

    COVERAGE_TARGET = 85.0  # Minimum required coverage percentage

    def __init__(self, source_dirs: List[str] = None):
        """
        Initialize coverage reporter.

        Args:
            source_dirs: List of source directories to measure coverage for
        """
        self.source_dirs = source_dirs or []
        self._cov = None
        self._results: Dict[str, Any] = {}

    def start_coverage(self) -> None:
        """Start coverage measurement."""
        if not COVERAGE_AVAILABLE:
            raise ImportError("coverage package is required")

        self._cov = coverage.Coverage(
            source=self.source_dirs if self.source_dirs else None,
            branch=True,
        )
        self._cov.start()

    def stop_coverage(self) -> None:
        """Stop coverage measurement."""
        if self._cov:
            self._cov.stop()
            self._cov.save()

    def get_coverage_percentage(self) -> float:
        """Get overall coverage percentage."""
        if not self._cov:
            return 0.0

        report_data = self._cov.get_data()
        total_statements = 0
        covered_statements = 0

        for filename in report_data.measured_files():
            analysis = self._cov.analysis2(filename)
            total_statements += len(analysis[1]) + len(analysis[2])
            covered_statements += len(analysis[1])

        if total_statements == 0:
            return 100.0

        return (covered_statements / total_statements) * 100

    def check_coverage_threshold(
        self,
        threshold: float = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if coverage meets threshold.

        Returns:
            Tuple of (passed, coverage_percent, message)
        """
        threshold = threshold or self.COVERAGE_TARGET
        coverage_pct = self.get_coverage_percentage()
        passed = coverage_pct >= threshold

        if passed:
            msg = f"Coverage {coverage_pct:.1f}% meets target {threshold}%"
        else:
            msg = f"Coverage {coverage_pct:.1f}% below target {threshold}%"

        return passed, coverage_pct, msg

    def generate_report(
        self,
        output_format: str = "text",
        output_file: str = None,
    ) -> str:
        """
        Generate coverage report.

        Args:
            output_format: 'text', 'html', 'xml', or 'json'
            output_file: Optional output file path

        Returns:
            Report as string (for text format)
        """
        if not self._cov:
            return "No coverage data available"

        import io

        if output_format == "text":
            output = io.StringIO()
            self._cov.report(file=output)
            result = output.getvalue()
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result)
            return result

        elif output_format == "html":
            directory = output_file or "htmlcov"
            self._cov.html_report(directory=directory)
            return f"HTML report generated in {directory}/"

        elif output_format == "xml":
            outfile = output_file or "coverage.xml"
            self._cov.xml_report(outfile=outfile)
            return f"XML report generated: {outfile}"

        elif output_format == "json":
            outfile = output_file or "coverage.json"
            self._cov.json_report(outfile=outfile)
            return f"JSON report generated: {outfile}"

        else:
            raise ValueError(f"Unknown format: {output_format}")

    @staticmethod
    def create_coverage_badge(
        coverage_pct: float,
        output_file: str = "coverage-badge.svg",
    ) -> str:
        """
        Create a coverage badge SVG.

        Args:
            coverage_pct: Coverage percentage
            output_file: Output file path

        Returns:
            Path to generated badge
        """
        # Determine color based on coverage
        if coverage_pct >= 90:
            color = "#4c1"  # Green
        elif coverage_pct >= 85:
            color = "#97CA00"  # Yellow-green
        elif coverage_pct >= 70:
            color = "#dfb317"  # Yellow
        elif coverage_pct >= 50:
            color = "#fe7d37"  # Orange
        else:
            color = "#e05d44"  # Red

        svg_template = f'''<svg xmlns="http://www.w3.org/2000/svg" width="106" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="106" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h61v20H0z"/>
    <path fill="{color}" d="M61 0h45v20H61z"/>
    <path fill="url(#b)" d="M0 0h106v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="30.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="30.5" y="14">coverage</text>
    <text x="82.5" y="15" fill="#010101" fill-opacity=".3">{coverage_pct:.0f}%</text>
    <text x="82.5" y="14">{coverage_pct:.0f}%</text>
  </g>
</svg>'''

        with open(output_file, 'w') as f:
            f.write(svg_template)

        return output_file


# =============================================================================
# Test Data Factory
# =============================================================================

class TestDataFactory:
    """
    Factory for generating test data for GreenLang agents.

    Supports all agent types from GL-001 to GL-016.
    """

    def __init__(self, seed: int = 42):
        """Initialize factory with seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_provenance_hash(self, data: Any) -> str:
        """Generate SHA-256 provenance hash for data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def generate_timestamp(
        self,
        start: datetime = None,
        end: datetime = None,
    ) -> datetime:
        """Generate random timestamp within range."""
        start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = end or datetime.now(timezone.utc)

        delta = end - start
        random_seconds = self.rng.uniform(0, delta.total_seconds())
        return start + timedelta(seconds=random_seconds)

    def generate_time_series(
        self,
        start: datetime,
        interval_seconds: int,
        count: int,
        base_value: float,
        noise_std: float = 0.0,
        trend: float = 0.0,
    ) -> List[Tuple[datetime, float]]:
        """Generate time series data."""
        series = []
        for i in range(count):
            timestamp = start + timedelta(seconds=i * interval_seconds)
            value = base_value + trend * i + self.rng.normal(0, noise_std)
            series.append((timestamp, value))
        return series

    def generate_sensor_reading(
        self,
        sensor_id: str,
        value_range: Tuple[float, float],
        timestamp: datetime = None,
    ) -> Dict[str, Any]:
        """Generate a sensor reading."""
        return {
            "sensor_id": sensor_id,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "value": self.rng.uniform(*value_range),
            "quality": self.rng.choice(["Good", "Good", "Good", "Uncertain", "Bad"]),
            "unit": "unit",
        }

    def generate_emission_report(
        self,
        facility_id: str,
        reporting_period: str,
    ) -> Dict[str, Any]:
        """Generate emission report data."""
        return {
            "facility_id": facility_id,
            "reporting_period": reporting_period,
            "scope1_emissions": {
                "stationary_combustion": self.rng.uniform(1000, 50000),
                "mobile_sources": self.rng.uniform(100, 5000),
                "fugitive_emissions": self.rng.uniform(50, 500),
            },
            "scope2_emissions": {
                "purchased_electricity": self.rng.uniform(5000, 100000),
                "purchased_steam": self.rng.uniform(100, 10000),
            },
            "total_emissions": 0,  # Will be calculated
            "unit": "kg CO2e",
            "provenance_hash": "",  # Will be calculated
        }

    def generate_cbam_shipment(
        self,
        product_category: str = None,
    ) -> Dict[str, Any]:
        """Generate CBAM shipment data."""
        categories = ["cement", "steel", "aluminum", "fertilizers", "hydrogen", "electricity"]
        category = product_category or self.rng.choice(categories)

        return {
            "shipment_id": f"CBAM-{self.rng.integers(100000, 999999)}",
            "product_category": category,
            "hs_code": f"{self.rng.integers(2500, 8500)}.{self.rng.integers(10, 99)}",
            "weight_tonnes": float(self.rng.uniform(1, 1000)),
            "origin_country": self.rng.choice(["CN", "IN", "TR", "RU", "UA", "EG"]),
            "import_date": self.generate_timestamp().date().isoformat(),
            "supplier_name": f"Supplier_{self.rng.integers(1, 100)}",
            "embedded_emissions": float(self.rng.uniform(0.5, 5.0)),  # tCO2/tonne
            "price_paid": float(self.rng.uniform(100, 10000)),
        }


# =============================================================================
# Assertion Helpers
# =============================================================================

class AssertionHelpers:
    """Collection of custom assertion helpers for GreenLang testing."""

    @staticmethod
    def assert_almost_equal(
        actual: float,
        expected: float,
        rel_tol: float = 1e-6,
        abs_tol: float = 1e-12,
        msg: str = "",
    ) -> None:
        """Assert two floats are almost equal."""
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(
                f"Values not equal: {actual} != {expected} "
                f"(rel_tol={rel_tol}, abs_tol={abs_tol}). {msg}"
            )

    @staticmethod
    def assert_valid_provenance_hash(hash_value: str) -> None:
        """Assert that a provenance hash is valid SHA-256."""
        if not isinstance(hash_value, str):
            raise AssertionError(f"Hash must be string, got {type(hash_value)}")
        if len(hash_value) != 64:
            raise AssertionError(f"SHA-256 hash must be 64 chars, got {len(hash_value)}")
        try:
            int(hash_value, 16)
        except ValueError:
            raise AssertionError(f"Invalid hex string: {hash_value}")

    @staticmethod
    def assert_deterministic(
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 10,
    ) -> None:
        """Assert that a function is deterministic."""
        kwargs = kwargs or {}
        results = [func(*args, **kwargs) for _ in range(iterations)]

        first_result = results[0]
        for i, result in enumerate(results[1:], 2):
            if result != first_result:
                raise AssertionError(
                    f"Function not deterministic: result {i} differs from result 1"
                )

    @staticmethod
    def assert_execution_time(
        func: Callable,
        max_seconds: float,
        args: tuple = (),
        kwargs: dict = None,
    ) -> float:
        """Assert function executes within time limit."""
        kwargs = kwargs or {}
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed > max_seconds:
            raise AssertionError(
                f"Execution time {elapsed:.3f}s exceeds limit {max_seconds}s"
            )
        return elapsed

    @staticmethod
    def assert_within_tolerance(
        actual: float,
        expected: float,
        tolerance: float,
        tolerance_type: ToleranceType = ToleranceType.RELATIVE,
    ) -> None:
        """Assert value is within tolerance of expected."""
        golden = GoldenValue(
            name="test",
            inputs={},
            expected_output=expected,
            source=ReferenceDataSource.CALCULATED,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
        )
        passed, msg = golden.verify(actual)
        if not passed:
            raise AssertionError(msg)

    @staticmethod
    def assert_energy_conserved(
        energy_in: float,
        energy_out: float,
        losses: float = 0.0,
        rel_tol: float = 1e-6,
    ) -> None:
        """Assert energy conservation (first law of thermodynamics)."""
        balance_error = abs(energy_in - energy_out - losses)
        max_energy = max(abs(energy_in), abs(energy_out), 1.0)

        if balance_error > rel_tol * max_energy:
            raise AssertionError(
                f"Energy not conserved: in={energy_in}, out={energy_out}, "
                f"losses={losses}, error={balance_error}"
            )

    @staticmethod
    def assert_valid_efficiency(efficiency: float, allow_zero: bool = False) -> None:
        """Assert efficiency is physically valid."""
        if allow_zero:
            if not (0.0 <= efficiency <= 1.0):
                raise AssertionError(f"Efficiency {efficiency} not in [0, 1]")
        else:
            if not (0.0 < efficiency <= 1.0):
                raise AssertionError(f"Efficiency {efficiency} not in (0, 1]")


# =============================================================================
# Export all public classes and functions
# =============================================================================

__all__ = [
    # Enums
    "ReferenceDataSource",
    "ToleranceType",
    # Data classes
    "GoldenValue",
    # Reference data
    "NISTReferenceData",
    "IAPWSReferenceData",
    "EPAReferenceData",
    "GoldenValueFixtures",
    # Test data generators
    "ThermodynamicTestData",
    "TestDataFactory",
    # Property-based testing
    "PropertyTestHelpers",
    # Coverage
    "CoverageReporter",
    # Assertions
    "AssertionHelpers",
]
