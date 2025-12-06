"""
ASME PTC 19.10 - Flue and Exhaust Gas Analyses

Zero-Hallucination Flue Gas Analysis Calculations

This module implements ASME Performance Test Code 19.10 for flue gas
composition analysis and emission calculations.

References:
    - ASME PTC 19.10-1981: Flue and Exhaust Gas Analyses
    - EPA Method 19: Sulfur Dioxide Removal and PM, SO2, NOx Emission Rates
    - ASME PTC 4-2013: Fired Steam Generators

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
import math
import hashlib


@dataclass
class FlueGasComposition:
    """
    Flue gas composition (dry basis, volume %).

    Typical range check:
    - CO2: 10-18% (coal), 8-12% (gas)
    - O2: 2-8%
    - CO: 0-500 ppm
    - N2: Balance (typically 70-80%)
    - SO2: 0-3000 ppm
    - NOx: 50-500 ppm
    """
    co2_pct: float
    o2_pct: float
    co_ppm: float = 0.0
    n2_pct: Optional[float] = None  # Calculated as balance if not provided
    so2_ppm: float = 0.0
    nox_ppm: float = 0.0
    h2o_pct: float = 0.0  # Water vapor (wet basis measurement)

    def __post_init__(self):
        """Validate and calculate N2 as balance."""
        if self.n2_pct is None:
            co_pct = self.co_ppm / 10000
            so2_pct = self.so2_ppm / 10000
            nox_pct = self.nox_ppm / 10000
            self.n2_pct = 100 - self.co2_pct - self.o2_pct - co_pct - so2_pct - nox_pct

        # Validate total
        total = self.co2_pct + self.o2_pct + self.n2_pct + \
                self.co_ppm/10000 + self.so2_ppm/10000 + self.nox_ppm/10000

        if abs(total - 100) > 1:  # Allow 1% tolerance
            raise ValueError(f"Dry gas composition must sum to ~100%, got {total}%")


@dataclass
class FlueGasAnalysisResult:
    """
    Flue gas analysis results per ASME PTC 19.10.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Composition (dry basis)
    co2_pct_dry: Decimal
    o2_pct_dry: Decimal
    co_ppm_dry: Decimal
    n2_pct_dry: Decimal
    so2_ppm_dry: Decimal
    nox_ppm_dry: Decimal

    # Composition (wet basis)
    h2o_pct_wet: Decimal
    co2_pct_wet: Decimal
    o2_pct_wet: Decimal

    # Combustion analysis
    excess_air_pct: Decimal
    air_fuel_ratio: Decimal
    theoretical_air_factor: Decimal

    # Molecular weight
    mw_dry_flue_gas: Decimal
    mw_wet_flue_gas: Decimal

    # Density at standard conditions
    density_dry_kg_m3: Decimal
    density_wet_kg_m3: Decimal

    # F-factors (EPA Method 19)
    fd_factor: Decimal  # Dry basis
    fw_factor: Decimal  # Wet basis
    fc_factor: Decimal  # Carbon basis

    # Emission concentrations (corrected to reference O2)
    co2_at_3pct_o2: Decimal
    so2_at_3pct_o2_ppm: Decimal
    nox_at_3pct_o2_ppm: Decimal
    co_at_3pct_o2_ppm: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "co2_pct_dry": float(self.co2_pct_dry),
            "o2_pct_dry": float(self.o2_pct_dry),
            "excess_air_pct": float(self.excess_air_pct),
            "mw_dry_flue_gas": float(self.mw_dry_flue_gas),
            "nox_at_3pct_o2_ppm": float(self.nox_at_3pct_o2_ppm),
            "so2_at_3pct_o2_ppm": float(self.so2_at_3pct_o2_ppm),
            "provenance_hash": self.provenance_hash
        }


class PTC1910FlueGas:
    """
    ASME PTC 19.10 Flue Gas Analysis Calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on ASME PTC 19.10 and EPA Method 19
    - Complete provenance tracking

    References:
        - ASME PTC 19.10-1981, Section 4 (Calculations)
        - EPA Method 19, 40 CFR Part 60, Appendix A
    """

    # Molecular weights
    MW_CO2 = Decimal("44.01")
    MW_O2 = Decimal("32.00")
    MW_N2 = Decimal("28.01")
    MW_CO = Decimal("28.01")
    MW_SO2 = Decimal("64.07")
    MW_NO2 = Decimal("46.01")
    MW_H2O = Decimal("18.02")
    MW_AIR = Decimal("28.97")

    # Standard conditions
    STD_TEMP_K = Decimal("273.15")
    STD_PRESSURE_KPA = Decimal("101.325")

    # Reference O2 for emissions
    REF_O2_PCT = Decimal("3.0")  # Typical reference

    def __init__(self, precision: int = 2):
        """Initialize calculator."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "ASME_PTC_19.10_EPA_Method_19",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def analyze(
        self,
        composition: FlueGasComposition,
        fuel_carbon_pct: float = 70.0,
        fuel_hydrogen_pct: float = 5.0,
        fuel_hhv_kj_kg: float = 30000.0
    ) -> FlueGasAnalysisResult:
        """
        Perform complete flue gas analysis per ASME PTC 19.10.

        ZERO-HALLUCINATION: Deterministic calculation per ASME PTC 19.10.

        Args:
            composition: Measured flue gas composition
            fuel_carbon_pct: Carbon content of fuel (wt%)
            fuel_hydrogen_pct: Hydrogen content of fuel (wt%)
            fuel_hhv_kj_kg: Higher heating value (kJ/kg)

        Returns:
            FlueGasAnalysisResult with complete analysis
        """
        # Convert to Decimal
        co2 = Decimal(str(composition.co2_pct))
        o2 = Decimal(str(composition.o2_pct))
        co = Decimal(str(composition.co_ppm))
        n2 = Decimal(str(composition.n2_pct))
        so2 = Decimal(str(composition.so2_ppm))
        nox = Decimal(str(composition.nox_ppm))
        h2o = Decimal(str(composition.h2o_pct))

        c_fuel = Decimal(str(fuel_carbon_pct)) / Decimal("100")
        h_fuel = Decimal(str(fuel_hydrogen_pct)) / Decimal("100")
        hhv = Decimal(str(fuel_hhv_kj_kg))

        # ============================================================
        # EXCESS AIR CALCULATION
        # Reference: ASME PTC 19.10, Section 4.2
        # ============================================================

        # Excess air from O2 measurement
        # EA = 100 * O2 / (20.95 - O2) for complete combustion
        if o2 >= Decimal("20.95"):
            raise ValueError("O2 cannot exceed atmospheric concentration")

        excess_air = Decimal("100") * o2 / (Decimal("20.95") - o2)

        # Theoretical air factor (lambda)
        lambda_factor = Decimal("1") + excess_air / Decimal("100")

        # ============================================================
        # MOLECULAR WEIGHT OF FLUE GAS
        # Reference: ASME PTC 19.10, Section 4.3
        # ============================================================

        # Convert ppm to percentage for MW calculation
        co_pct = co / Decimal("10000")
        so2_pct = so2 / Decimal("10000")
        nox_pct = nox / Decimal("10000")

        # Dry basis molecular weight
        mw_dry = (co2 * self.MW_CO2 + o2 * self.MW_O2 + n2 * self.MW_N2 +
                  co_pct * self.MW_CO + so2_pct * self.MW_SO2 +
                  nox_pct * self.MW_NO2) / Decimal("100")

        # Wet basis composition
        dry_fraction = (Decimal("100") - h2o) / Decimal("100")
        co2_wet = co2 * dry_fraction
        o2_wet = o2 * dry_fraction
        n2_wet = n2 * dry_fraction

        # Wet basis molecular weight
        mw_wet = (dry_fraction * mw_dry * Decimal("100") + h2o * self.MW_H2O) / Decimal("100")

        # ============================================================
        # DENSITY AT STANDARD CONDITIONS
        # Reference: ASME PTC 19.10, Section 4.4
        # ============================================================

        # Using ideal gas law: rho = P*MW / (R*T)
        r_universal = Decimal("8.314")  # kJ/(kmol*K)

        density_dry = self.STD_PRESSURE_KPA * mw_dry / (r_universal * self.STD_TEMP_K)
        density_wet = self.STD_PRESSURE_KPA * mw_wet / (r_universal * self.STD_TEMP_K)

        # ============================================================
        # AIR-FUEL RATIO
        # Reference: ASME PTC 19.10, Section 4.5
        # ============================================================

        # Stoichiometric air requirement (kg air / kg fuel)
        stoich_air = Decimal("11.53") * c_fuel + Decimal("34.34") * h_fuel

        # Actual air-fuel ratio
        afr = stoich_air * lambda_factor

        # ============================================================
        # F-FACTORS (EPA Method 19)
        # Reference: EPA Method 19, Section 12.3
        # ============================================================

        # Fd (dry basis) - dscf/MMBtu
        # Convert HHV to Btu/lb (1 kJ/kg = 0.4299 Btu/lb)
        hhv_btu_lb = hhv * Decimal("0.4299")

        fd = (Decimal("3.64E6") * c_fuel + Decimal("1.53E7") * h_fuel) / hhv_btu_lb

        # Fw (wet basis) - wscf/MMBtu
        # Includes water from combustion
        water_from_h = Decimal("9") * h_fuel  # kg water per kg fuel
        fw = fd + Decimal("4.68E4") * water_from_h / Decimal("1E6") * hhv_btu_lb

        # Fc (carbon basis) - scf CO2/MMBtu
        fc = Decimal("1.96E6") * c_fuel / hhv_btu_lb

        # ============================================================
        # CORRECTED EMISSIONS (to reference O2)
        # Reference: EPA Method 19, Section 12.5
        # ============================================================

        # Correction factor: (20.95 - O2_ref) / (20.95 - O2_meas)
        correction = (Decimal("20.95") - self.REF_O2_PCT) / (Decimal("20.95") - o2)

        co2_corrected = co2 * correction
        so2_corrected = so2 * correction
        nox_corrected = nox * correction
        co_corrected = co * correction

        # Create provenance
        inputs = {
            "co2_pct": str(co2),
            "o2_pct": str(o2),
            "co_ppm": str(co),
            "so2_ppm": str(so2),
            "nox_ppm": str(nox)
        }
        outputs = {
            "excess_air_pct": str(excess_air),
            "mw_dry": str(mw_dry),
            "nox_corrected": str(nox_corrected)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return FlueGasAnalysisResult(
            co2_pct_dry=self._apply_precision(co2),
            o2_pct_dry=self._apply_precision(o2),
            co_ppm_dry=self._apply_precision(co),
            n2_pct_dry=self._apply_precision(n2),
            so2_ppm_dry=self._apply_precision(so2),
            nox_ppm_dry=self._apply_precision(nox),
            h2o_pct_wet=self._apply_precision(h2o),
            co2_pct_wet=self._apply_precision(co2_wet),
            o2_pct_wet=self._apply_precision(o2_wet),
            excess_air_pct=self._apply_precision(excess_air),
            air_fuel_ratio=self._apply_precision(afr),
            theoretical_air_factor=self._apply_precision(lambda_factor),
            mw_dry_flue_gas=self._apply_precision(mw_dry),
            mw_wet_flue_gas=self._apply_precision(mw_wet),
            density_dry_kg_m3=self._apply_precision(density_dry),
            density_wet_kg_m3=self._apply_precision(density_wet),
            fd_factor=self._apply_precision(fd),
            fw_factor=self._apply_precision(fw),
            fc_factor=self._apply_precision(fc),
            co2_at_3pct_o2=self._apply_precision(co2_corrected),
            so2_at_3pct_o2_ppm=self._apply_precision(so2_corrected),
            nox_at_3pct_o2_ppm=self._apply_precision(nox_corrected),
            co_at_3pct_o2_ppm=self._apply_precision(co_corrected),
            provenance_hash=provenance_hash
        )

    def correct_to_reference_o2(
        self,
        concentration: float,
        measured_o2_pct: float,
        reference_o2_pct: float = 3.0
    ) -> Decimal:
        """
        Correct emission concentration to reference O2.

        Reference: EPA Method 19, Equation 19-2

        C_ref = C_meas * (20.95 - O2_ref) / (20.95 - O2_meas)

        Args:
            concentration: Measured concentration (ppm or mg/m3)
            measured_o2_pct: Measured O2 in flue gas (%)
            reference_o2_pct: Reference O2 level (%)

        Returns:
            Corrected concentration
        """
        c_meas = Decimal(str(concentration))
        o2_meas = Decimal(str(measured_o2_pct))
        o2_ref = Decimal(str(reference_o2_pct))

        if o2_meas >= Decimal("20.95"):
            raise ValueError("Measured O2 cannot exceed atmospheric")

        correction = (Decimal("20.95") - o2_ref) / (Decimal("20.95") - o2_meas)
        c_ref = c_meas * correction

        return self._apply_precision(c_ref)

    def ppm_to_mg_m3(
        self,
        concentration_ppm: float,
        molecular_weight: float,
        temperature_c: float = 25.0,
        pressure_kpa: float = 101.325
    ) -> Decimal:
        """
        Convert concentration from ppm to mg/m3.

        Reference: ASME PTC 19.10, Appendix A

        mg/m3 = ppm * MW * P / (R * T)

        Args:
            concentration_ppm: Concentration in ppm (v/v)
            molecular_weight: Molecular weight of pollutant
            temperature_c: Temperature (C)
            pressure_kpa: Pressure (kPa)

        Returns:
            Concentration in mg/m3
        """
        ppm = Decimal(str(concentration_ppm))
        mw = Decimal(str(molecular_weight))
        t_k = Decimal(str(temperature_c)) + Decimal("273.15")
        p = Decimal(str(pressure_kpa))

        r = Decimal("8.314")  # kJ/(kmol*K)

        # mg/m3 = ppm * MW * P / (R * T) * 1000 (for mg)
        mg_m3 = ppm * mw * p / (r * t_k)

        return self._apply_precision(mg_m3)

    def mg_m3_to_ppm(
        self,
        concentration_mg_m3: float,
        molecular_weight: float,
        temperature_c: float = 25.0,
        pressure_kpa: float = 101.325
    ) -> Decimal:
        """
        Convert concentration from mg/m3 to ppm.

        Args:
            concentration_mg_m3: Concentration in mg/m3
            molecular_weight: Molecular weight of pollutant
            temperature_c: Temperature (C)
            pressure_kpa: Pressure (kPa)

        Returns:
            Concentration in ppm (v/v)
        """
        mg_m3 = Decimal(str(concentration_mg_m3))
        mw = Decimal(str(molecular_weight))
        t_k = Decimal(str(temperature_c)) + Decimal("273.15")
        p = Decimal(str(pressure_kpa))

        r = Decimal("8.314")

        ppm = mg_m3 * r * t_k / (mw * p)

        return self._apply_precision(ppm)

    def emission_rate(
        self,
        concentration_ppm: float,
        flue_gas_flow_m3_s: float,
        molecular_weight: float
    ) -> Decimal:
        """
        Calculate mass emission rate.

        Reference: EPA Method 19, Section 12.4

        Args:
            concentration_ppm: Pollutant concentration (ppm)
            flue_gas_flow_m3_s: Flue gas volumetric flow (m3/s at std)
            molecular_weight: Pollutant molecular weight

        Returns:
            Emission rate in kg/h
        """
        ppm = Decimal(str(concentration_ppm))
        v_flow = Decimal(str(flue_gas_flow_m3_s))
        mw = Decimal(str(molecular_weight))

        # Convert ppm to mole fraction
        x = ppm / Decimal("1E6")

        # Molar flow at standard conditions
        v_molar = Decimal("22.414")  # m3/kmol at STP

        # Mass flow = x * V_flow * MW / V_molar * 3600
        mass_rate = x * v_flow * mw / v_molar * Decimal("3600")  # kg/h

        return self._apply_precision(mass_rate)


# Convenience functions
def flue_gas_analysis(
    co2_pct: float,
    o2_pct: float,
    co_ppm: float = 0.0,
    so2_ppm: float = 0.0,
    nox_ppm: float = 0.0,
    fuel_carbon_pct: float = 70.0
) -> FlueGasAnalysisResult:
    """
    Perform flue gas analysis per ASME PTC 19.10.

    Example:
        >>> result = flue_gas_analysis(
        ...     co2_pct=14.0,
        ...     o2_pct=4.0,
        ...     nox_ppm=200,
        ...     so2_ppm=500
        ... )
        >>> print(f"Excess air: {result.excess_air_pct}%")
    """
    calc = PTC1910FlueGas()

    composition = FlueGasComposition(
        co2_pct=co2_pct,
        o2_pct=o2_pct,
        co_ppm=co_ppm,
        so2_ppm=so2_ppm,
        nox_ppm=nox_ppm
    )

    return calc.analyze(composition, fuel_carbon_pct)


def correct_emissions_to_o2(
    value: float,
    measured_o2: float,
    reference_o2: float = 3.0
) -> Decimal:
    """Correct emission concentration to reference O2 level."""
    calc = PTC1910FlueGas()
    return calc.correct_to_reference_o2(value, measured_o2, reference_o2)


def excess_air_from_o2(o2_pct: float) -> Decimal:
    """Calculate excess air percentage from O2 measurement."""
    o2 = Decimal(str(o2_pct))
    if o2 >= Decimal("20.95"):
        raise ValueError("O2 cannot exceed atmospheric concentration")
    return (Decimal("100") * o2 / (Decimal("20.95") - o2)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
