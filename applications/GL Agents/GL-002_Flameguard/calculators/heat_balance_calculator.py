"""
GL-002 FLAMEGUARD - Heat Balance Calculator

Complete heat balance for boiler systems per ASME PTC 4.1.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HeatBalanceInput:
    """Heat balance input data."""
    fuel_input_mmbtu_hr: float
    steam_output_mmbtu_hr: float
    blowdown_mmbtu_hr: float = 0.0
    sootblowing_mmbtu_hr: float = 0.0
    aux_steam_mmbtu_hr: float = 0.0
    air_preheat_recovery_mmbtu_hr: float = 0.0
    economizer_recovery_mmbtu_hr: float = 0.0
    stack_loss_mmbtu_hr: float = 0.0
    radiation_loss_mmbtu_hr: float = 0.0


@dataclass
class HeatBalanceResult:
    """Heat balance calculation result."""
    timestamp: datetime

    # Heat input
    total_heat_input_mmbtu_hr: float
    fuel_heat_mmbtu_hr: float
    sensible_heat_credits_mmbtu_hr: float

    # Heat output
    total_heat_output_mmbtu_hr: float
    steam_heat_mmbtu_hr: float
    blowdown_heat_mmbtu_hr: float

    # Losses
    total_losses_mmbtu_hr: float
    stack_loss_mmbtu_hr: float
    radiation_loss_mmbtu_hr: float
    unaccounted_loss_mmbtu_hr: float

    # Recovery
    total_recovery_mmbtu_hr: float
    air_preheat_mmbtu_hr: float
    economizer_mmbtu_hr: float

    # Balance check
    balance_error_percent: float
    balanced: bool


class HeatBalanceCalculator:
    """
    Heat balance calculator for boiler systems.

    Verifies energy balance:
    Input = Output + Losses - Recovery
    """

    def __init__(self, tolerance_percent: float = 2.0) -> None:
        self.tolerance = tolerance_percent

    def calculate(self, inp: HeatBalanceInput) -> HeatBalanceResult:
        """Calculate heat balance."""

        # Heat input
        fuel_heat = inp.fuel_input_mmbtu_hr
        sensible_credits = 0.0  # Could add air/fuel sensible heat
        total_input = fuel_heat + sensible_credits

        # Heat output
        steam_heat = inp.steam_output_mmbtu_hr
        blowdown_heat = inp.blowdown_mmbtu_hr
        total_output = steam_heat + blowdown_heat

        # Losses
        stack_loss = inp.stack_loss_mmbtu_hr
        radiation_loss = inp.radiation_loss_mmbtu_hr
        total_losses = stack_loss + radiation_loss

        # Recovery
        air_preheat = inp.air_preheat_recovery_mmbtu_hr
        economizer = inp.economizer_recovery_mmbtu_hr
        total_recovery = air_preheat + economizer

        # Balance check
        # Input = Output + Losses - Recovery + Unaccounted
        expected_output = total_output + total_losses - total_recovery
        unaccounted = total_input - expected_output

        balance_error = abs(unaccounted) / total_input * 100 if total_input > 0 else 0
        balanced = balance_error <= self.tolerance

        return HeatBalanceResult(
            timestamp=datetime.now(timezone.utc),
            total_heat_input_mmbtu_hr=round(total_input, 3),
            fuel_heat_mmbtu_hr=round(fuel_heat, 3),
            sensible_heat_credits_mmbtu_hr=round(sensible_credits, 3),
            total_heat_output_mmbtu_hr=round(total_output, 3),
            steam_heat_mmbtu_hr=round(steam_heat, 3),
            blowdown_heat_mmbtu_hr=round(blowdown_heat, 3),
            total_losses_mmbtu_hr=round(total_losses, 3),
            stack_loss_mmbtu_hr=round(stack_loss, 3),
            radiation_loss_mmbtu_hr=round(radiation_loss, 3),
            unaccounted_loss_mmbtu_hr=round(unaccounted, 3),
            total_recovery_mmbtu_hr=round(total_recovery, 3),
            air_preheat_mmbtu_hr=round(air_preheat, 3),
            economizer_mmbtu_hr=round(economizer, 3),
            balance_error_percent=round(balance_error, 2),
            balanced=balanced,
        )
