"""
Psychrometric Calculations

Zero-Hallucination Air-Water Vapor Property Calculations

This module implements deterministic psychrometric calculations for:
- Humidity calculations (relative, absolute, specific)
- Wet-bulb and dew-point temperatures
- Enthalpy calculations
- Air density calculations
- Psychrometric chart relationships

References:
    - ASHRAE Fundamentals Handbook, Chapter 1 (Psychrometrics)
    - ASME PTC 4.1: Steam Generating Units (Combustion Air)
    - ISO 5167-1: Measurement of Fluid Flow

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
import math
import hashlib


@dataclass
class PsychrometricProperties:
    """
    Complete psychrometric state with provenance.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Input conditions
    dry_bulb_temperature_c: Decimal
    pressure_kpa: Decimal

    # Humidity
    relative_humidity_percent: Decimal
    humidity_ratio_kg_kg: Decimal  # kg water / kg dry air
    specific_humidity_kg_kg: Decimal  # kg water / kg moist air
    absolute_humidity_kg_m3: Decimal  # kg water / m3 moist air

    # Temperatures
    wet_bulb_temperature_c: Decimal
    dew_point_temperature_c: Decimal

    # Thermodynamic properties
    enthalpy_kj_kg_da: Decimal  # kJ per kg dry air
    specific_volume_m3_kg_da: Decimal  # m3 per kg dry air
    moist_air_density_kg_m3: Decimal

    # Water vapor properties
    saturation_pressure_kpa: Decimal
    partial_pressure_water_kpa: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "dry_bulb_temperature_c": float(self.dry_bulb_temperature_c),
            "pressure_kpa": float(self.pressure_kpa),
            "relative_humidity_percent": float(self.relative_humidity_percent),
            "humidity_ratio_kg_kg": float(self.humidity_ratio_kg_kg),
            "specific_humidity_kg_kg": float(self.specific_humidity_kg_kg),
            "absolute_humidity_kg_m3": float(self.absolute_humidity_kg_m3),
            "wet_bulb_temperature_c": float(self.wet_bulb_temperature_c),
            "dew_point_temperature_c": float(self.dew_point_temperature_c),
            "enthalpy_kj_kg_da": float(self.enthalpy_kj_kg_da),
            "specific_volume_m3_kg_da": float(self.specific_volume_m3_kg_da),
            "moist_air_density_kg_m3": float(self.moist_air_density_kg_m3),
            "saturation_pressure_kpa": float(self.saturation_pressure_kpa),
            "partial_pressure_water_kpa": float(self.partial_pressure_water_kpa),
            "provenance_hash": self.provenance_hash
        }


class PsychrometricCalculator:
    """
    Psychrometric property calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on ASHRAE Fundamentals equations
    - Complete provenance tracking

    References:
        - ASHRAE Fundamentals 2021, Chapter 1
        - Buck, A.L. (1981). "New Equations for Computing Vapor Pressure"
        - WMO Technical Note No. 8 (1966)
    """

    # Universal gas constant
    R_UNIVERSAL = Decimal("8.31446261815324")  # J/(mol*K)

    # Gas constants for dry air and water vapor
    R_DA = Decimal("287.055")  # J/(kg*K) - dry air
    R_W = Decimal("461.520")  # J/(kg*K) - water vapor

    # Molecular weights
    MW_DA = Decimal("28.9647")  # kg/kmol - dry air
    MW_W = Decimal("18.01528")  # kg/kmol - water

    # Ratio of molecular weights
    EPSILON = MW_W / MW_DA  # 0.621945

    # Standard pressure
    P_STD = Decimal("101.325")  # kPa

    def __init__(self, precision: int = 4):
        """Initialize psychrometric calculator."""
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
            "method": "ASHRAE_Psychrometrics",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def saturation_pressure(self, temperature_c: float) -> Decimal:
        """
        Calculate saturation vapor pressure of water.

        Reference: ASHRAE Fundamentals 2021, Equations 5 and 6

        For temperatures from -100C to 200C.

        Args:
            temperature_c: Temperature in Celsius

        Returns:
            Saturation pressure in kPa
        """
        t = Decimal(str(temperature_c))
        t_k = t + Decimal("273.15")

        if t < Decimal("-100") or t > Decimal("200"):
            raise ValueError(f"Temperature {temperature_c}C outside valid range [-100, 200]C")

        if t >= Decimal("0"):
            # Over liquid water (0C to 200C)
            # ASHRAE Equation 6
            c1 = Decimal("-5.8002206E+03")
            c2 = Decimal("1.3914993E+00")
            c3 = Decimal("-4.8640239E-02")
            c4 = Decimal("4.1764768E-05")
            c5 = Decimal("-1.4452093E-08")
            c6 = Decimal("6.5459673E+00")

            ln_pws = (c1 / t_k + c2 + c3 * t_k + c4 * t_k ** 2 +
                      c5 * t_k ** 3 + c6 * Decimal(str(math.log(float(t_k)))))
        else:
            # Over ice (-100C to 0C)
            # ASHRAE Equation 5
            c1 = Decimal("-5.6745359E+03")
            c2 = Decimal("6.3925247E+00")
            c3 = Decimal("-9.6778430E-03")
            c4 = Decimal("6.2215701E-07")
            c5 = Decimal("2.0747825E-09")
            c6 = Decimal("-9.4840240E-13")
            c7 = Decimal("4.1635019E+00")

            ln_pws = (c1 / t_k + c2 + c3 * t_k + c4 * t_k ** 2 +
                      c5 * t_k ** 3 + c6 * t_k ** 4 +
                      c7 * Decimal(str(math.log(float(t_k)))))

        pws = Decimal(str(math.exp(float(ln_pws)))) / Decimal("1000")  # Convert Pa to kPa

        return self._apply_precision(pws)

    def dew_point_from_partial_pressure(self, pw_kpa: float) -> Decimal:
        """
        Calculate dew point temperature from water vapor partial pressure.

        Reference: ASHRAE Fundamentals 2021, Equation 39

        Args:
            pw_kpa: Water vapor partial pressure in kPa

        Returns:
            Dew point temperature in Celsius
        """
        pw = Decimal(str(pw_kpa))

        if pw <= 0:
            raise ValueError("Partial pressure must be positive")

        # Convert to Pa for ASHRAE equations
        pw_pa = pw * Decimal("1000")
        alpha = Decimal(str(math.log(float(pw_pa))))

        if pw_pa >= Decimal("611.2"):  # Above 0C
            # ASHRAE Equation 39
            c1 = Decimal("6.54")
            c2 = Decimal("14.526")
            c3 = Decimal("0.7389")
            c4 = Decimal("0.09486")
            c5 = Decimal("0.4569")

            td = c1 + c2 * alpha + c3 * alpha ** 2 + c4 * alpha ** 3 + c5 * pw_pa ** Decimal("0.1984")
        else:  # Below 0C (frost point)
            # ASHRAE Equation 40
            c1 = Decimal("6.09")
            c2 = Decimal("12.608")
            c3 = Decimal("0.4959")

            td = c1 + c2 * alpha + c3 * alpha ** 2

        return self._apply_precision(td)

    def humidity_ratio_from_partial_pressure(
        self,
        pw_kpa: float,
        p_kpa: float
    ) -> Decimal:
        """
        Calculate humidity ratio from water vapor partial pressure.

        Reference: ASHRAE Fundamentals 2021, Equation 22

        W = 0.621945 * pw / (p - pw)

        Args:
            pw_kpa: Water vapor partial pressure in kPa
            p_kpa: Total pressure in kPa

        Returns:
            Humidity ratio in kg water / kg dry air
        """
        pw = Decimal(str(pw_kpa))
        p = Decimal(str(p_kpa))

        if pw >= p:
            raise ValueError("Water vapor pressure cannot exceed total pressure")

        w = self.EPSILON * pw / (p - pw)

        return self._apply_precision(w)

    def partial_pressure_from_humidity_ratio(
        self,
        w: float,
        p_kpa: float
    ) -> Decimal:
        """
        Calculate water vapor partial pressure from humidity ratio.

        Reference: ASHRAE Fundamentals 2021, Equation 22 (rearranged)

        pw = p * W / (0.621945 + W)

        Args:
            w: Humidity ratio in kg water / kg dry air
            p_kpa: Total pressure in kPa

        Returns:
            Water vapor partial pressure in kPa
        """
        w_dec = Decimal(str(w))
        p = Decimal(str(p_kpa))

        pw = p * w_dec / (self.EPSILON + w_dec)

        return self._apply_precision(pw)

    def enthalpy(self, t_c: float, w: float) -> Decimal:
        """
        Calculate specific enthalpy of moist air.

        Reference: ASHRAE Fundamentals 2021, Equation 32

        h = 1.006 * t + W * (2501 + 1.86 * t)

        Args:
            t_c: Dry-bulb temperature in Celsius
            w: Humidity ratio in kg water / kg dry air

        Returns:
            Specific enthalpy in kJ/kg dry air
        """
        t = Decimal(str(t_c))
        w_dec = Decimal(str(w))

        # Specific heat of dry air (kJ/kg-K)
        cp_da = Decimal("1.006")

        # Specific heat of water vapor (kJ/kg-K)
        cp_w = Decimal("1.86")

        # Latent heat of vaporization at 0C (kJ/kg)
        hfg0 = Decimal("2501")

        h = cp_da * t + w_dec * (hfg0 + cp_w * t)

        return self._apply_precision(h)

    def specific_volume(
        self,
        t_c: float,
        w: float,
        p_kpa: float
    ) -> Decimal:
        """
        Calculate specific volume of moist air.

        Reference: ASHRAE Fundamentals 2021, Equation 28

        v = Ra * T * (1 + 1.6078 * W) / p

        Args:
            t_c: Dry-bulb temperature in Celsius
            w: Humidity ratio in kg water / kg dry air
            p_kpa: Total pressure in kPa

        Returns:
            Specific volume in m3/kg dry air
        """
        t = Decimal(str(t_c))
        w_dec = Decimal(str(w))
        p = Decimal(str(p_kpa))

        t_k = t + Decimal("273.15")

        # Ratio 1/epsilon - 1 = 0.6078
        v = self.R_DA / Decimal("1000") * t_k * (Decimal("1") + w_dec / self.EPSILON) / p

        return self._apply_precision(v)

    def wet_bulb_temperature(
        self,
        t_c: float,
        rh_percent: float,
        p_kpa: float
    ) -> Decimal:
        """
        Calculate wet-bulb temperature.

        Reference: ASHRAE Fundamentals 2021, Equations 35-37

        Uses iterative solution of the psychrometric equation.

        Args:
            t_c: Dry-bulb temperature in Celsius
            rh_percent: Relative humidity in percent
            p_kpa: Total pressure in kPa

        Returns:
            Wet-bulb temperature in Celsius
        """
        t = Decimal(str(t_c))
        rh = Decimal(str(rh_percent)) / Decimal("100")
        p = Decimal(str(p_kpa))

        # Calculate actual humidity ratio
        pws = self.saturation_pressure(float(t))
        pw = rh * pws
        w_actual = self.humidity_ratio_from_partial_pressure(float(pw), float(p))

        # Iterative solution for wet-bulb temperature
        # Initial guess
        t_wb = t - Decimal("5")
        tolerance = Decimal("0.0001")
        max_iterations = 50

        for _ in range(max_iterations):
            # Saturation pressure at wet-bulb temperature
            pws_wb = self.saturation_pressure(float(t_wb))

            # Saturation humidity ratio at wet-bulb
            ws_wb = self.humidity_ratio_from_partial_pressure(float(pws_wb), float(p))

            # Calculate humidity ratio from psychrometric equation
            # ASHRAE Equation 35 (for above 0C)
            if t_wb >= Decimal("0"):
                w_calc = ((Decimal("2501") - Decimal("2.326") * t_wb) * ws_wb -
                          Decimal("1.006") * (t - t_wb)) / \
                         (Decimal("2501") + Decimal("1.86") * t - Decimal("4.186") * t_wb)
            else:
                # Below 0C, use ice
                w_calc = ((Decimal("2830") - Decimal("0.24") * t_wb) * ws_wb -
                          Decimal("1.006") * (t - t_wb)) / \
                         (Decimal("2830") + Decimal("1.86") * t - Decimal("2.1") * t_wb)

            # Check convergence
            error = w_calc - w_actual
            if abs(error) < tolerance:
                break

            # Adjust wet-bulb temperature
            # Simple bisection-like adjustment
            if error > 0:
                t_wb = t_wb - Decimal("0.1") * abs(error) * Decimal("100")
            else:
                t_wb = t_wb + Decimal("0.1") * abs(error) * Decimal("100")

            # Ensure wet-bulb doesn't exceed dry-bulb
            if t_wb > t:
                t_wb = t

        return self._apply_precision(t_wb)

    def properties_from_t_rh(
        self,
        temperature_c: float,
        relative_humidity_percent: float,
        pressure_kpa: float = 101.325
    ) -> PsychrometricProperties:
        """
        Calculate all psychrometric properties from temperature and RH.

        ZERO-HALLUCINATION: Deterministic ASHRAE-based calculation.

        Args:
            temperature_c: Dry-bulb temperature in Celsius
            relative_humidity_percent: Relative humidity in percent
            pressure_kpa: Atmospheric pressure in kPa (default: 101.325)

        Returns:
            PsychrometricProperties with complete provenance
        """
        t = Decimal(str(temperature_c))
        rh = Decimal(str(relative_humidity_percent))
        p = Decimal(str(pressure_kpa))

        if rh < 0 or rh > 100:
            raise ValueError(f"Relative humidity must be 0-100%, got {relative_humidity_percent}%")

        # Step 1: Calculate saturation pressure
        pws = self.saturation_pressure(temperature_c)

        # Step 2: Calculate partial pressure of water vapor
        pw = rh / Decimal("100") * pws

        # Step 3: Calculate humidity ratio
        w = self.humidity_ratio_from_partial_pressure(float(pw), pressure_kpa)

        # Step 4: Calculate specific humidity
        # q = W / (1 + W)
        specific_humidity = w / (Decimal("1") + w)

        # Step 5: Calculate specific volume
        v = self.specific_volume(temperature_c, float(w), pressure_kpa)

        # Step 6: Calculate absolute humidity
        # rho_w = W / v
        absolute_humidity = w / v

        # Step 7: Calculate moist air density
        # rho = (1 + W) / v
        density = (Decimal("1") + w) / v

        # Step 8: Calculate enthalpy
        h = self.enthalpy(temperature_c, float(w))

        # Step 9: Calculate wet-bulb temperature
        t_wb = self.wet_bulb_temperature(temperature_c, relative_humidity_percent, pressure_kpa)

        # Step 10: Calculate dew point temperature
        t_dp = self.dew_point_from_partial_pressure(float(pw))

        # Create provenance
        inputs = {
            "temperature_c": str(t),
            "relative_humidity_percent": str(rh),
            "pressure_kpa": str(p)
        }
        outputs = {
            "humidity_ratio": str(w),
            "wet_bulb": str(t_wb),
            "dew_point": str(t_dp),
            "enthalpy": str(h)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PsychrometricProperties(
            dry_bulb_temperature_c=self._apply_precision(t),
            pressure_kpa=self._apply_precision(p),
            relative_humidity_percent=self._apply_precision(rh),
            humidity_ratio_kg_kg=self._apply_precision(w),
            specific_humidity_kg_kg=self._apply_precision(specific_humidity),
            absolute_humidity_kg_m3=self._apply_precision(absolute_humidity),
            wet_bulb_temperature_c=self._apply_precision(t_wb),
            dew_point_temperature_c=self._apply_precision(t_dp),
            enthalpy_kj_kg_da=self._apply_precision(h),
            specific_volume_m3_kg_da=self._apply_precision(v),
            moist_air_density_kg_m3=self._apply_precision(density),
            saturation_pressure_kpa=self._apply_precision(pws),
            partial_pressure_water_kpa=self._apply_precision(pw),
            provenance_hash=provenance_hash
        )

    def properties_from_t_wb(
        self,
        temperature_c: float,
        wet_bulb_c: float,
        pressure_kpa: float = 101.325
    ) -> PsychrometricProperties:
        """
        Calculate psychrometric properties from dry-bulb and wet-bulb temperatures.

        Args:
            temperature_c: Dry-bulb temperature in Celsius
            wet_bulb_c: Wet-bulb temperature in Celsius
            pressure_kpa: Atmospheric pressure in kPa

        Returns:
            PsychrometricProperties with complete provenance
        """
        t = Decimal(str(temperature_c))
        t_wb = Decimal(str(wet_bulb_c))
        p = Decimal(str(pressure_kpa))

        if t_wb > t:
            raise ValueError("Wet-bulb temperature cannot exceed dry-bulb temperature")

        # Calculate saturation pressure at wet-bulb
        pws_wb = self.saturation_pressure(wet_bulb_c)

        # Calculate saturation humidity ratio at wet-bulb
        ws_wb = self.humidity_ratio_from_partial_pressure(float(pws_wb), pressure_kpa)

        # Calculate actual humidity ratio from psychrometric equation
        if t_wb >= Decimal("0"):
            w = ((Decimal("2501") - Decimal("2.326") * t_wb) * ws_wb -
                 Decimal("1.006") * (t - t_wb)) / \
                (Decimal("2501") + Decimal("1.86") * t - Decimal("4.186") * t_wb)
        else:
            w = ((Decimal("2830") - Decimal("0.24") * t_wb) * ws_wb -
                 Decimal("1.006") * (t - t_wb)) / \
                (Decimal("2830") + Decimal("1.86") * t - Decimal("2.1") * t_wb)

        # Calculate partial pressure of water vapor
        pw = self.partial_pressure_from_humidity_ratio(float(w), pressure_kpa)

        # Calculate saturation pressure at dry-bulb
        pws = self.saturation_pressure(temperature_c)

        # Calculate relative humidity
        rh = pw / pws * Decimal("100")

        # Calculate remaining properties
        specific_humidity = w / (Decimal("1") + w)
        v = self.specific_volume(temperature_c, float(w), pressure_kpa)
        absolute_humidity = w / v
        density = (Decimal("1") + w) / v
        h = self.enthalpy(temperature_c, float(w))
        t_dp = self.dew_point_from_partial_pressure(float(pw))

        inputs = {
            "temperature_c": str(t),
            "wet_bulb_c": str(t_wb),
            "pressure_kpa": str(p)
        }
        outputs = {
            "humidity_ratio": str(w),
            "relative_humidity": str(rh),
            "dew_point": str(t_dp)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PsychrometricProperties(
            dry_bulb_temperature_c=self._apply_precision(t),
            pressure_kpa=self._apply_precision(p),
            relative_humidity_percent=self._apply_precision(rh),
            humidity_ratio_kg_kg=self._apply_precision(w),
            specific_humidity_kg_kg=self._apply_precision(specific_humidity),
            absolute_humidity_kg_m3=self._apply_precision(absolute_humidity),
            wet_bulb_temperature_c=self._apply_precision(t_wb),
            dew_point_temperature_c=self._apply_precision(t_dp),
            enthalpy_kj_kg_da=self._apply_precision(h),
            specific_volume_m3_kg_da=self._apply_precision(v),
            moist_air_density_kg_m3=self._apply_precision(density),
            saturation_pressure_kpa=self._apply_precision(pws),
            partial_pressure_water_kpa=self._apply_precision(pw),
            provenance_hash=provenance_hash
        )

    def properties_from_t_dp(
        self,
        temperature_c: float,
        dew_point_c: float,
        pressure_kpa: float = 101.325
    ) -> PsychrometricProperties:
        """
        Calculate psychrometric properties from dry-bulb and dew point temperatures.

        Args:
            temperature_c: Dry-bulb temperature in Celsius
            dew_point_c: Dew point temperature in Celsius
            pressure_kpa: Atmospheric pressure in kPa

        Returns:
            PsychrometricProperties with complete provenance
        """
        t = Decimal(str(temperature_c))
        t_dp = Decimal(str(dew_point_c))
        p = Decimal(str(pressure_kpa))

        if t_dp > t:
            raise ValueError("Dew point cannot exceed dry-bulb temperature")

        # Partial pressure of water vapor equals saturation pressure at dew point
        pw = self.saturation_pressure(dew_point_c)

        # Calculate saturation pressure at dry-bulb
        pws = self.saturation_pressure(temperature_c)

        # Calculate relative humidity
        rh = pw / pws * Decimal("100")

        # Calculate humidity ratio
        w = self.humidity_ratio_from_partial_pressure(float(pw), pressure_kpa)

        # Calculate remaining properties
        specific_humidity = w / (Decimal("1") + w)
        v = self.specific_volume(temperature_c, float(w), pressure_kpa)
        absolute_humidity = w / v
        density = (Decimal("1") + w) / v
        h = self.enthalpy(temperature_c, float(w))
        t_wb = self.wet_bulb_temperature(temperature_c, float(rh), pressure_kpa)

        inputs = {
            "temperature_c": str(t),
            "dew_point_c": str(t_dp),
            "pressure_kpa": str(p)
        }
        outputs = {
            "humidity_ratio": str(w),
            "relative_humidity": str(rh),
            "wet_bulb": str(t_wb)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PsychrometricProperties(
            dry_bulb_temperature_c=self._apply_precision(t),
            pressure_kpa=self._apply_precision(p),
            relative_humidity_percent=self._apply_precision(rh),
            humidity_ratio_kg_kg=self._apply_precision(w),
            specific_humidity_kg_kg=self._apply_precision(specific_humidity),
            absolute_humidity_kg_m3=self._apply_precision(absolute_humidity),
            wet_bulb_temperature_c=self._apply_precision(t_wb),
            dew_point_temperature_c=self._apply_precision(t_dp),
            enthalpy_kj_kg_da=self._apply_precision(h),
            specific_volume_m3_kg_da=self._apply_precision(v),
            moist_air_density_kg_m3=self._apply_precision(density),
            saturation_pressure_kpa=self._apply_precision(pws),
            partial_pressure_water_kpa=self._apply_precision(pw),
            provenance_hash=provenance_hash
        )

    def mixing_air_streams(
        self,
        props1: PsychrometricProperties,
        mass_flow1_kg_s: float,
        props2: PsychrometricProperties,
        mass_flow2_kg_s: float
    ) -> PsychrometricProperties:
        """
        Calculate properties of mixed air streams.

        Reference: ASHRAE Fundamentals 2021, Equation 44

        Args:
            props1: Properties of first air stream
            mass_flow1_kg_s: Dry air mass flow of first stream (kg/s)
            props2: Properties of second air stream
            mass_flow2_kg_s: Dry air mass flow of second stream (kg/s)

        Returns:
            Properties of mixed stream
        """
        m1 = Decimal(str(mass_flow1_kg_s))
        m2 = Decimal(str(mass_flow2_kg_s))
        m_total = m1 + m2

        # Mixed humidity ratio (mass balance)
        w_mix = (m1 * props1.humidity_ratio_kg_kg + m2 * props2.humidity_ratio_kg_kg) / m_total

        # Mixed enthalpy (energy balance)
        h_mix = (m1 * props1.enthalpy_kj_kg_da + m2 * props2.enthalpy_kj_kg_da) / m_total

        # Calculate mixed temperature from enthalpy and humidity ratio
        # h = 1.006*t + W*(2501 + 1.86*t)
        # h = t*(1.006 + 1.86*W) + 2501*W
        # t = (h - 2501*W) / (1.006 + 1.86*W)
        t_mix = (h_mix - Decimal("2501") * w_mix) / (Decimal("1.006") + Decimal("1.86") * w_mix)

        # Calculate remaining properties at mixed conditions
        # First get partial pressure from humidity ratio
        p = props1.pressure_kpa  # Assume same pressure
        pw_mix = self.partial_pressure_from_humidity_ratio(float(w_mix), float(p))

        # Get saturation pressure at mixed temperature
        pws_mix = self.saturation_pressure(float(t_mix))

        # Relative humidity
        rh_mix = pw_mix / pws_mix * Decimal("100")

        if rh_mix > Decimal("100"):
            rh_mix = Decimal("100")  # Condensation would occur

        return self.properties_from_t_rh(float(t_mix), float(rh_mix), float(p))


# Convenience functions
def psychrometric_properties(
    temperature_c: float,
    relative_humidity: float,
    pressure_kpa: float = 101.325
) -> PsychrometricProperties:
    """
    Calculate psychrometric properties.

    Example:
        >>> props = psychrometric_properties(25, 50)
        >>> print(f"Wet bulb: {props.wet_bulb_temperature_c}C")
    """
    calc = PsychrometricCalculator()
    return calc.properties_from_t_rh(temperature_c, relative_humidity, pressure_kpa)


def saturation_vapor_pressure(temperature_c: float) -> Decimal:
    """Get saturation vapor pressure in kPa."""
    calc = PsychrometricCalculator()
    return calc.saturation_pressure(temperature_c)


def dew_point(temperature_c: float, relative_humidity: float) -> Decimal:
    """Calculate dew point temperature."""
    calc = PsychrometricCalculator()
    props = calc.properties_from_t_rh(temperature_c, relative_humidity)
    return props.dew_point_temperature_c


def wet_bulb(temperature_c: float, relative_humidity: float, pressure_kpa: float = 101.325) -> Decimal:
    """Calculate wet bulb temperature."""
    calc = PsychrometricCalculator()
    return calc.wet_bulb_temperature(temperature_c, relative_humidity, pressure_kpa)
