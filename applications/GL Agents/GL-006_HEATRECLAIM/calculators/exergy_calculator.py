"""
GL-006 HEATRECLAIM - Exergy Calculator

Implements second-law (exergy) analysis for heat recovery systems.
Quantifies thermodynamic irreversibility and improvement potential.

Reference: Bejan, Tsatsaronis, Moran, "Thermal Design and Optimization",
Wiley, 1996.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    ExergyAnalysisResult,
)
from ..core.config import REFERENCE_TEMPERATURE_K, Phase

logger = logging.getLogger(__name__)


@dataclass
class StreamExergy:
    """Exergy analysis for a single stream."""

    stream_id: str
    exergy_rate_kW: float
    exergy_change_kW: float
    specific_exergy_kJ_kg: float
    entropy_generation_kW_K: float


@dataclass
class ExchangerExergy:
    """Exergy analysis for a heat exchanger."""

    exchanger_id: str
    hot_exergy_in_kW: float
    hot_exergy_out_kW: float
    cold_exergy_in_kW: float
    cold_exergy_out_kW: float
    exergy_destruction_kW: float
    exergy_efficiency: float
    entropy_generation_kW_K: float


class ExergyCalculator:
    """
    Exergy (second-law) analysis calculator.

    Provides deterministic calculations for:
    - Stream exergy rates
    - Heat exchanger exergy destruction
    - System exergy efficiency
    - Entropy generation
    - Improvement potential

    Example:
        >>> calc = ExergyCalculator(T0_K=298.15)
        >>> result = calc.analyze_network(streams, exchangers)
        >>> print(f"Exergy efficiency: {result.exergy_efficiency:.1%}")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "EXERGY_v1.0"

    def __init__(
        self,
        T0_K: float = REFERENCE_TEMPERATURE_K,
        p0_kPa: float = 101.325,
    ) -> None:
        """
        Initialize exergy calculator.

        Args:
            T0_K: Reference (dead state) temperature in Kelvin
            p0_kPa: Reference pressure in kPa
        """
        self.T0_K = T0_K
        self.p0_kPa = p0_kPa
        self.T0_C = T0_K - 273.15

    def calculate_stream_exergy_rate(
        self,
        stream: HeatStream,
    ) -> StreamExergy:
        """
        Calculate exergy rate for a heat stream.

        For streams with negligible pressure effects (incompressible
        liquids at constant pressure), exergy rate is:

        E_dot = m_dot * Cp * [(T - T0) - T0 * ln(T/T0)]

        Args:
            stream: Heat stream specification

        Returns:
            StreamExergy with exergy rates and changes
        """
        T_supply_K = stream.T_supply_C + 273.15
        T_target_K = stream.T_target_C + 273.15
        m_dot = stream.m_dot_kg_s
        Cp = stream.Cp_kJ_kgK

        # Exergy at supply temperature
        ex_supply = self._specific_exergy(T_supply_K, Cp)
        E_dot_supply = m_dot * ex_supply

        # Exergy at target temperature
        ex_target = self._specific_exergy(T_target_K, Cp)
        E_dot_target = m_dot * ex_target

        # Exergy change
        delta_E = E_dot_target - E_dot_supply

        # Entropy change (for ideal process)
        if T_supply_K > 0 and T_target_K > 0:
            delta_s = Cp * math.log(T_target_K / T_supply_K)
        else:
            delta_s = 0.0

        S_gen = 0.0  # Zero for reversible heating/cooling

        return StreamExergy(
            stream_id=stream.stream_id,
            exergy_rate_kW=round(E_dot_supply, 3),
            exergy_change_kW=round(delta_E, 3),
            specific_exergy_kJ_kg=round(ex_supply, 3),
            entropy_generation_kW_K=round(S_gen, 6),
        )

    def calculate_exchanger_exergy(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream] = None,
        cold_stream: Optional[HeatStream] = None,
    ) -> ExchangerExergy:
        """
        Calculate exergy destruction in a heat exchanger.

        Exergy destruction = T0 * S_gen

        where S_gen = m_hot*Cp_hot*ln(T_hot_out/T_hot_in) +
                      m_cold*Cp_cold*ln(T_cold_out/T_cold_in)

        Args:
            exchanger: Heat exchanger specification
            hot_stream: Hot side stream (for m_dot, Cp)
            cold_stream: Cold side stream (for m_dot, Cp)

        Returns:
            ExchangerExergy with destruction and efficiency
        """
        # Temperature in Kelvin
        T_hot_in_K = exchanger.hot_inlet_T_C + 273.15
        T_hot_out_K = exchanger.hot_outlet_T_C + 273.15
        T_cold_in_K = exchanger.cold_inlet_T_C + 273.15
        T_cold_out_K = exchanger.cold_outlet_T_C + 273.15

        # Get stream properties
        if hot_stream:
            m_hot = hot_stream.m_dot_kg_s
            Cp_hot = hot_stream.Cp_kJ_kgK
        else:
            # Estimate from duty
            m_hot = exchanger.duty_kW / (
                abs(T_hot_in_K - T_hot_out_K) * 4.186 + 0.001
            )
            Cp_hot = 4.186  # Assume water

        if cold_stream:
            m_cold = cold_stream.m_dot_kg_s
            Cp_cold = cold_stream.Cp_kJ_kgK
        else:
            m_cold = exchanger.duty_kW / (
                abs(T_cold_out_K - T_cold_in_K) * 4.186 + 0.001
            )
            Cp_cold = 4.186

        # Calculate exergy at each state
        ex_hot_in = self._specific_exergy(T_hot_in_K, Cp_hot)
        ex_hot_out = self._specific_exergy(T_hot_out_K, Cp_hot)
        ex_cold_in = self._specific_exergy(T_cold_in_K, Cp_cold)
        ex_cold_out = self._specific_exergy(T_cold_out_K, Cp_cold)

        E_hot_in = m_hot * ex_hot_in
        E_hot_out = m_hot * ex_hot_out
        E_cold_in = m_cold * ex_cold_in
        E_cold_out = m_cold * ex_cold_out

        # Exergy destruction by balance
        E_destruction = (E_hot_in + E_cold_in) - (E_hot_out + E_cold_out)
        E_destruction = max(0.0, E_destruction)  # Must be non-negative

        # Alternative: from entropy generation
        if T_hot_in_K > 0 and T_hot_out_K > 0 and T_cold_in_K > 0 and T_cold_out_K > 0:
            S_gen_hot = m_hot * Cp_hot * math.log(T_hot_out_K / T_hot_in_K)
            S_gen_cold = m_cold * Cp_cold * math.log(T_cold_out_K / T_cold_in_K)
            S_gen = S_gen_hot + S_gen_cold
        else:
            S_gen = E_destruction / self.T0_K if self.T0_K > 0 else 0.0

        # Exergy efficiency
        # For HX: efficiency = exergy gained by cold / exergy lost by hot
        E_gained = E_cold_out - E_cold_in
        E_lost = E_hot_in - E_hot_out

        if E_lost > 0:
            exergy_efficiency = E_gained / E_lost
        else:
            exergy_efficiency = 0.0

        exergy_efficiency = max(0.0, min(1.0, exergy_efficiency))

        return ExchangerExergy(
            exchanger_id=exchanger.exchanger_id,
            hot_exergy_in_kW=round(E_hot_in, 3),
            hot_exergy_out_kW=round(E_hot_out, 3),
            cold_exergy_in_kW=round(E_cold_in, 3),
            cold_exergy_out_kW=round(E_cold_out, 3),
            exergy_destruction_kW=round(E_destruction, 3),
            exergy_efficiency=round(exergy_efficiency, 4),
            entropy_generation_kW_K=round(S_gen, 6),
        )

    def analyze_network(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        exchangers: List[HeatExchanger],
        hot_utility_duty_kW: float = 0.0,
        cold_utility_duty_kW: float = 0.0,
        hot_utility_T_C: float = 200.0,
        cold_utility_T_C: float = 20.0,
    ) -> ExergyAnalysisResult:
        """
        Perform complete exergy analysis of heat exchanger network.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            exchangers: Heat exchangers in network
            hot_utility_duty_kW: Hot utility consumption
            cold_utility_duty_kW: Cold utility consumption
            hot_utility_T_C: Hot utility temperature
            cold_utility_T_C: Cold utility temperature

        Returns:
            ExergyAnalysisResult with complete analysis
        """
        # Build stream lookup
        stream_map = {s.stream_id: s for s in hot_streams + cold_streams}

        # Calculate exergy for each exchanger
        exergy_by_exchanger = {}
        total_destruction = 0.0
        total_entropy_gen = 0.0

        for hx in exchangers:
            hot_stream = stream_map.get(hx.hot_stream_id)
            cold_stream = stream_map.get(hx.cold_stream_id)

            hx_exergy = self.calculate_exchanger_exergy(
                hx, hot_stream, cold_stream
            )

            exergy_by_exchanger[hx.exchanger_id] = {
                "exergy_destruction_kW": hx_exergy.exergy_destruction_kW,
                "exergy_efficiency": hx_exergy.exergy_efficiency,
                "entropy_generation_kW_K": hx_exergy.entropy_generation_kW_K,
            }

            total_destruction += hx_exergy.exergy_destruction_kW
            total_entropy_gen += hx_exergy.entropy_generation_kW_K

        # Utility exergy destruction
        exergy_by_utility = {}

        if hot_utility_duty_kW > 0:
            # Exergy destruction in hot utility heat transfer
            T_utility_K = hot_utility_T_C + 273.15
            # Average cold stream target temperature
            avg_cold_target_K = sum(
                s.T_target_C for s in cold_streams
            ) / max(1, len(cold_streams)) + 273.15

            E_dest_hot_util = self._utility_exergy_destruction(
                hot_utility_duty_kW, T_utility_K, avg_cold_target_K
            )
            exergy_by_utility["hot_utility"] = round(E_dest_hot_util, 3)
            total_destruction += E_dest_hot_util

        if cold_utility_duty_kW > 0:
            # Exergy destruction in cold utility heat transfer
            T_utility_K = cold_utility_T_C + 273.15
            avg_hot_target_K = sum(
                s.T_target_C for s in hot_streams
            ) / max(1, len(hot_streams)) + 273.15

            E_dest_cold_util = self._utility_exergy_destruction(
                cold_utility_duty_kW, avg_hot_target_K, T_utility_K
            )
            exergy_by_utility["cold_utility"] = round(E_dest_cold_util, 3)
            total_destruction += E_dest_cold_util

        # Total exergy input (from hot streams)
        total_exergy_input = sum(
            self.calculate_stream_exergy_rate(s).exergy_rate_kW
            for s in hot_streams
        )

        # Total exergy output (to cold streams at target)
        total_exergy_output = total_exergy_input - total_destruction

        # System exergy efficiency
        if total_exergy_input > 0:
            system_efficiency = total_exergy_output / total_exergy_input
        else:
            system_efficiency = 0.0

        system_efficiency = max(0.0, min(1.0, system_efficiency))

        # Improvement potential
        # Maximum efficiency would be with reversible heat transfer
        improvement_potential = total_destruction  # All destruction is theoretically avoidable
        improvement_percent = (
            100 * total_destruction / total_exergy_input
            if total_exergy_input > 0 else 0.0
        )

        # Build result
        input_hash = self._compute_hash({
            "hot_streams": [s.stream_id for s in hot_streams],
            "cold_streams": [s.stream_id for s in cold_streams],
            "exchangers": [hx.exchanger_id for hx in exchangers],
            "T0_K": self.T0_K,
        })

        result = ExergyAnalysisResult(
            reference_temperature_K=self.T0_K,
            reference_pressure_kPa=self.p0_kPa,
            total_exergy_input_kW=round(total_exergy_input, 2),
            total_exergy_output_kW=round(max(0, total_exergy_output), 2),
            total_exergy_destruction_kW=round(total_destruction, 2),
            exergy_efficiency=round(system_efficiency, 4),
            exergy_by_exchanger=exergy_by_exchanger,
            exergy_by_utility=exergy_by_utility,
            improvement_potential_kW=round(improvement_potential, 2),
            improvement_potential_percent=round(improvement_percent, 2),
            total_entropy_generation_kW_K=round(total_entropy_gen, 6),
            input_hash=input_hash,
            formula_version=self.FORMULA_VERSION,
        )

        # Compute output hash
        result.output_hash = self._compute_hash({
            "total_exergy_destruction_kW": result.total_exergy_destruction_kW,
            "exergy_efficiency": result.exergy_efficiency,
        })

        return result

    def _specific_exergy(self, T_K: float, Cp_kJ_kgK: float) -> float:
        """
        Calculate specific physical exergy for sensible heat.

        ex = Cp * [(T - T0) - T0 * ln(T/T0)]

        Args:
            T_K: Temperature in Kelvin
            Cp_kJ_kgK: Specific heat capacity

        Returns:
            Specific exergy in kJ/kg
        """
        if T_K <= 0 or self.T0_K <= 0:
            return 0.0

        delta_T = T_K - self.T0_K
        ln_term = self.T0_K * math.log(T_K / self.T0_K)

        return Cp_kJ_kgK * (delta_T - ln_term)

    def _utility_exergy_destruction(
        self,
        duty_kW: float,
        T_hot_K: float,
        T_cold_K: float,
    ) -> float:
        """
        Calculate exergy destruction in utility heat transfer.

        For heat Q transferred from T_hot to T_cold:
        E_dest = Q * T0 * (1/T_cold - 1/T_hot)
        """
        if T_hot_K <= 0 or T_cold_K <= 0:
            return 0.0

        if T_hot_K <= T_cold_K:
            return 0.0

        # Carnot factor difference
        carnot_diff = (1 / T_cold_K) - (1 / T_hot_K)
        E_dest = duty_kW * self.T0_K * carnot_diff

        return max(0.0, E_dest)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def calculate_carnot_efficiency(T_hot_K: float, T_cold_K: float) -> float:
    """
    Calculate Carnot efficiency for heat engine between two temperatures.

    η_carnot = 1 - T_cold / T_hot

    Args:
        T_hot_K: Hot reservoir temperature (K)
        T_cold_K: Cold reservoir temperature (K)

    Returns:
        Carnot efficiency (0 to 1)
    """
    if T_hot_K <= 0 or T_cold_K <= 0 or T_cold_K >= T_hot_K:
        return 0.0

    return 1.0 - T_cold_K / T_hot_K


def calculate_exergy_factor(T_K: float, T0_K: float = REFERENCE_TEMPERATURE_K) -> float:
    """
    Calculate Carnot (exergy) factor for heat at temperature T.

    τ = 1 - T0/T (for T > T0)
    τ = T0/T - 1 (for T < T0, cooling exergy)

    Args:
        T_K: Temperature in Kelvin
        T0_K: Reference temperature in Kelvin

    Returns:
        Exergy factor (dimensionless)
    """
    if T_K <= 0 or T0_K <= 0:
        return 0.0

    if T_K >= T0_K:
        return 1.0 - T0_K / T_K
    else:
        return T0_K / T_K - 1.0
