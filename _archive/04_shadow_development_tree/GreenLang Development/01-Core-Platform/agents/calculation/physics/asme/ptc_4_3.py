"""
ASME PTC 4.3 - Air Heater Performance

Zero-Hallucination Air Preheater Calculations

This module implements ASME Performance Test Code 4.3 for determining
the performance of air preheaters used in steam generating units.

References:
    - ASME PTC 4.3-1968: Air Heaters
    - ASME PTC 4-2013: Fired Steam Generators
    - ASME PTC 19.10: Flue and Exhaust Gas Analyses

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
import math
import hashlib


@dataclass
class AirHeaterInputData:
    """Input data for air heater performance calculation."""
    # Air side
    air_inlet_temp_c: float
    air_outlet_temp_c: float
    air_mass_flow_kg_s: float

    # Gas side
    gas_inlet_temp_c: float
    gas_outlet_temp_c: float
    gas_mass_flow_kg_s: float

    # Optional: Leakage
    air_leakage_pct: float = 0.0

    # Optional: Specific heats (if not using defaults)
    cp_air_kj_kgk: float = 1.006
    cp_gas_kj_kgk: float = 1.05


@dataclass
class AirHeaterResult:
    """
    Air heater performance results per ASME PTC 4.3.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Heat transfer
    air_side_heat_kw: Decimal
    gas_side_heat_kw: Decimal
    heat_balance_error_pct: Decimal

    # Temperatures
    air_temp_rise_c: Decimal
    gas_temp_drop_c: Decimal
    lmtd_c: Decimal

    # Performance metrics
    effectiveness: Decimal
    x_ratio: Decimal  # Gas temp drop / Air temp rise

    # Corrected values (no leakage)
    corrected_air_outlet_temp_c: Decimal
    corrected_gas_outlet_temp_c: Decimal

    # Leakage analysis
    air_leakage_fraction: Decimal
    gas_side_air_ingress_kg_s: Decimal

    # UA value
    ua_kw_k: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "air_side_heat_kw": float(self.air_side_heat_kw),
            "gas_side_heat_kw": float(self.gas_side_heat_kw),
            "effectiveness": float(self.effectiveness),
            "x_ratio": float(self.x_ratio),
            "lmtd_c": float(self.lmtd_c),
            "ua_kw_k": float(self.ua_kw_k),
            "provenance_hash": self.provenance_hash
        }


class PTC43AirHeater:
    """
    ASME PTC 4.3 Air Heater Performance Calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on ASME PTC 4.3 standard formulas
    - Complete provenance tracking

    Types of air heaters covered:
    - Recuperative (tubular, plate)
    - Regenerative (Ljungstrom, Rothemuhle)

    References:
        - ASME PTC 4.3-1968, Section 4 (Performance Calculations)
        - ASME PTC 4-2013, Section 5.8 (Air Heater)
    """

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
            "method": "ASME_PTC_4.3",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_performance(self, data: AirHeaterInputData) -> AirHeaterResult:
        """
        Calculate air heater performance per ASME PTC 4.3.

        ZERO-HALLUCINATION: Deterministic calculation per ASME PTC 4.3.

        Args:
            data: Air heater operating data

        Returns:
            AirHeaterResult with complete analysis
        """
        # Convert to Decimal
        t_air_in = Decimal(str(data.air_inlet_temp_c))
        t_air_out = Decimal(str(data.air_outlet_temp_c))
        m_air = Decimal(str(data.air_mass_flow_kg_s))

        t_gas_in = Decimal(str(data.gas_inlet_temp_c))
        t_gas_out = Decimal(str(data.gas_outlet_temp_c))
        m_gas = Decimal(str(data.gas_mass_flow_kg_s))

        cp_air = Decimal(str(data.cp_air_kj_kgk))
        cp_gas = Decimal(str(data.cp_gas_kj_kgk))

        leakage = Decimal(str(data.air_leakage_pct)) / Decimal("100")

        # ============================================================
        # BASIC HEAT TRANSFER CALCULATIONS
        # Reference: ASME PTC 4.3, Section 4.2
        # ============================================================

        # Temperature changes
        air_temp_rise = t_air_out - t_air_in
        gas_temp_drop = t_gas_in - t_gas_out

        # Heat transferred (kW)
        q_air = m_air * cp_air * air_temp_rise
        q_gas = m_gas * cp_gas * gas_temp_drop

        # Heat balance check
        if q_air > 0:
            heat_balance_error = abs(q_air - q_gas) / q_air * Decimal("100")
        else:
            heat_balance_error = Decimal("0")

        # ============================================================
        # LEAKAGE CORRECTION
        # Reference: ASME PTC 4.3, Section 4.3
        # ============================================================

        # Air ingress to gas side
        air_ingress = m_air * leakage

        # Corrected gas outlet temperature (without leakage cooling)
        # The gas is diluted by cold air leakage
        if m_gas > 0 and leakage > 0:
            # Energy balance: m_gas*cp*(T_gas_out_corr - T_ref) =
            #                 (m_gas - m_leak)*cp*(T_gas_out - T_ref) + m_leak*cp*(T_air_in - T_ref)
            t_gas_out_corr = t_gas_out + leakage * m_air / m_gas * (t_gas_out - t_air_in)
        else:
            t_gas_out_corr = t_gas_out

        # Corrected air outlet (accounts for bypassed air)
        t_air_out_corr = t_air_out  # No significant correction for air side

        # ============================================================
        # LOG MEAN TEMPERATURE DIFFERENCE
        # Reference: ASME PTC 4.3, Section 4.4
        # ============================================================

        # Counter-flow arrangement (typical for air heaters)
        delta_t1 = t_gas_in - t_air_out_corr  # Hot end
        delta_t2 = t_gas_out_corr - t_air_in  # Cold end

        if delta_t1 <= 0 or delta_t2 <= 0:
            raise ValueError("Temperature cross detected - check operating data")

        if abs(delta_t1 - delta_t2) < Decimal("0.1"):
            lmtd = (delta_t1 + delta_t2) / Decimal("2")
        else:
            lmtd = (delta_t1 - delta_t2) / Decimal(str(math.log(float(delta_t1 / delta_t2))))

        # ============================================================
        # EFFECTIVENESS CALCULATION
        # Reference: ASME PTC 4.3, Section 4.5
        # ============================================================

        # Heat capacity rates
        c_air = m_air * cp_air
        c_gas = m_gas * cp_gas

        c_min = min(c_air, c_gas)
        c_max = max(c_air, c_gas)

        # Maximum possible heat transfer
        q_max = c_min * (t_gas_in - t_air_in)

        # Effectiveness
        if q_max > 0:
            effectiveness = q_air / q_max
        else:
            effectiveness = Decimal("0")

        # X-ratio (characteristic of air heater)
        if air_temp_rise > 0:
            x_ratio = gas_temp_drop / air_temp_rise
        else:
            x_ratio = Decimal("0")

        # ============================================================
        # UA CALCULATION
        # Reference: ASME PTC 4.3, Section 4.6
        # ============================================================

        # UA = Q / LMTD
        if lmtd > 0:
            ua = q_air / lmtd
        else:
            ua = Decimal("0")

        # Create provenance
        inputs = {
            "t_air_in": str(t_air_in),
            "t_air_out": str(t_air_out),
            "t_gas_in": str(t_gas_in),
            "t_gas_out": str(t_gas_out),
            "m_air": str(m_air),
            "m_gas": str(m_gas)
        }
        outputs = {
            "q_air": str(q_air),
            "effectiveness": str(effectiveness),
            "lmtd": str(lmtd)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return AirHeaterResult(
            air_side_heat_kw=self._apply_precision(q_air),
            gas_side_heat_kw=self._apply_precision(q_gas),
            heat_balance_error_pct=self._apply_precision(heat_balance_error),
            air_temp_rise_c=self._apply_precision(air_temp_rise),
            gas_temp_drop_c=self._apply_precision(gas_temp_drop),
            lmtd_c=self._apply_precision(lmtd),
            effectiveness=self._apply_precision(effectiveness),
            x_ratio=self._apply_precision(x_ratio),
            corrected_air_outlet_temp_c=self._apply_precision(t_air_out_corr),
            corrected_gas_outlet_temp_c=self._apply_precision(t_gas_out_corr),
            air_leakage_fraction=self._apply_precision(leakage),
            gas_side_air_ingress_kg_s=self._apply_precision(air_ingress),
            ua_kw_k=self._apply_precision(ua),
            provenance_hash=provenance_hash
        )

    def correct_to_design(
        self,
        test_result: AirHeaterResult,
        design_air_inlet_c: float,
        design_gas_inlet_c: float,
        design_air_flow_kg_s: float,
        design_gas_flow_kg_s: float
    ) -> Dict[str, Decimal]:
        """
        Correct test results to design conditions.

        Reference: ASME PTC 4.3, Section 5 (Corrections)

        Args:
            test_result: Results from test
            design_air_inlet_c: Design air inlet temperature
            design_gas_inlet_c: Design gas inlet temperature
            design_air_flow_kg_s: Design air flow rate
            design_gas_flow_kg_s: Design gas flow rate

        Returns:
            Dictionary with corrected performance
        """
        # Use test UA value to predict design performance
        ua = test_result.ua_kw_k

        t_air_in_d = Decimal(str(design_air_inlet_c))
        t_gas_in_d = Decimal(str(design_gas_inlet_c))
        m_air_d = Decimal(str(design_air_flow_kg_s))
        m_gas_d = Decimal(str(design_gas_flow_kg_s))

        cp_air = Decimal("1.006")
        cp_gas = Decimal("1.05")

        # Heat capacity rates at design
        c_air_d = m_air_d * cp_air
        c_gas_d = m_gas_d * cp_gas

        c_min_d = min(c_air_d, c_gas_d)
        c_max_d = max(c_air_d, c_gas_d)
        cr_d = c_min_d / c_max_d

        # NTU at design
        ntu_d = ua / c_min_d

        # Effectiveness at design (counter-flow)
        if abs(cr_d - Decimal("1")) < Decimal("0.0001"):
            eff_d = ntu_d / (Decimal("1") + ntu_d)
        else:
            exp_term = Decimal(str(math.exp(-float(ntu_d * (Decimal("1") - cr_d)))))
            eff_d = (Decimal("1") - exp_term) / (Decimal("1") - cr_d * exp_term)

        # Heat transfer at design
        q_d = eff_d * c_min_d * (t_gas_in_d - t_air_in_d)

        # Outlet temperatures at design
        t_air_out_d = t_air_in_d + q_d / c_air_d
        t_gas_out_d = t_gas_in_d - q_d / c_gas_d

        return {
            "design_air_outlet_c": self._apply_precision(t_air_out_d),
            "design_gas_outlet_c": self._apply_precision(t_gas_out_d),
            "design_effectiveness": self._apply_precision(eff_d),
            "design_heat_kw": self._apply_precision(q_d),
            "design_ntu": self._apply_precision(ntu_d)
        }

    def calculate_leakage(
        self,
        o2_air_in_pct: float,
        o2_gas_in_pct: float,
        o2_gas_out_pct: float
    ) -> Decimal:
        """
        Calculate air heater leakage from O2 measurements.

        Reference: ASME PTC 4.3, Section 4.3.2

        Air-side to gas-side leakage causes O2 increase across gas pass.

        Args:
            o2_air_in_pct: O2 in inlet air (typically 20.95%)
            o2_gas_in_pct: O2 in flue gas entering air heater
            o2_gas_out_pct: O2 in flue gas leaving air heater

        Returns:
            Leakage as fraction of inlet air
        """
        o2_air = Decimal(str(o2_air_in_pct))
        o2_in = Decimal(str(o2_gas_in_pct))
        o2_out = Decimal(str(o2_gas_out_pct))

        if o2_out <= o2_in:
            return Decimal("0")

        # Mass balance on O2
        # (1 + L) * O2_out = O2_in + L * O2_air
        # L = (O2_out - O2_in) / (O2_air - O2_out)

        if o2_air - o2_out <= 0:
            raise ValueError("Invalid O2 readings for leakage calculation")

        leakage = (o2_out - o2_in) / (o2_air - o2_out)

        return self._apply_precision(leakage)


# Convenience functions
def air_heater_performance(
    air_inlet_temp_c: float,
    air_outlet_temp_c: float,
    gas_inlet_temp_c: float,
    gas_outlet_temp_c: float,
    air_mass_flow_kg_s: float,
    gas_mass_flow_kg_s: float,
    air_leakage_pct: float = 0.0
) -> AirHeaterResult:
    """
    Calculate air heater performance per ASME PTC 4.3.

    Example:
        >>> result = air_heater_performance(
        ...     air_inlet_temp_c=30,
        ...     air_outlet_temp_c=280,
        ...     gas_inlet_temp_c=350,
        ...     gas_outlet_temp_c=150,
        ...     air_mass_flow_kg_s=50,
        ...     gas_mass_flow_kg_s=55
        ... )
        >>> print(f"Effectiveness: {result.effectiveness}")
    """
    calc = PTC43AirHeater()

    data = AirHeaterInputData(
        air_inlet_temp_c=air_inlet_temp_c,
        air_outlet_temp_c=air_outlet_temp_c,
        air_mass_flow_kg_s=air_mass_flow_kg_s,
        gas_inlet_temp_c=gas_inlet_temp_c,
        gas_outlet_temp_c=gas_outlet_temp_c,
        gas_mass_flow_kg_s=gas_mass_flow_kg_s,
        air_leakage_pct=air_leakage_pct
    )

    return calc.calculate_performance(data)


def air_heater_leakage(
    o2_gas_in_pct: float,
    o2_gas_out_pct: float
) -> Decimal:
    """Calculate air heater leakage from O2 measurements."""
    calc = PTC43AirHeater()
    return calc.calculate_leakage(20.95, o2_gas_in_pct, o2_gas_out_pct)
