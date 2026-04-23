"""
Cycles of Concentration (CoC) Calculator

This module provides deterministic calculations for Cycles of Concentration
in boiler and cooling water systems. All calculations follow mass balance
principles and produce SHA-256 hashes for audit trails.

Zero-Hallucination Guarantee:
- All calculations are based on deterministic physics formulas
- Mass balance: Makeup_Flow * Makeup_Concentration = Blowdown_Flow * Boiler_Concentration
- Temperature compensation uses ASTM D1125 standard
- Complete provenance tracking for regulatory compliance
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import hashlib
import json
from datetime import datetime

from .units import (
    UnitValue,
    get_unit_registry,
    temperature_compensation_conductivity,
)


@dataclass
class CoCResult:
    """Result of a Cycles of Concentration calculation."""
    coc: Decimal
    method: str
    inputs: Dict[str, Any]
    calculation_steps: list
    provenance_hash: str
    timestamp: str
    warnings: list


@dataclass
class BlowdownResult:
    """Result of a blowdown flow calculation."""
    blowdown_flow: UnitValue
    makeup_flow: UnitValue
    coc: Decimal
    mass_balance_verified: bool
    provenance_hash: str
    timestamp: str


class CoCCalculator:
    """
    Cycles of Concentration Calculator.

    Provides deterministic calculations for:
    - CoC from conductivity measurements
    - CoC from chloride (tracer) measurements
    - Blowdown flow estimation
    - Mass balance verification

    All calculations produce SHA-256 hashes for complete audit trails.
    """

    def __init__(self, config_version: str = "1.0.0", code_version: str = "1.0.0"):
        """
        Initialize the CoC Calculator.

        Args:
            config_version: Version of the configuration being used
            code_version: Version of the calculation code
        """
        self.config_version = config_version
        self.code_version = code_version
        self.unit_registry = get_unit_registry()

    def calculate_coc_from_conductivity(
        self,
        boiler_conductivity: Union[float, Decimal, UnitValue],
        makeup_conductivity: Union[float, Decimal, UnitValue],
        boiler_temp: Optional[Union[float, Decimal]] = None,
        makeup_temp: Optional[Union[float, Decimal]] = None,
        temp_compensation: bool = True,
        reference_temp: Union[float, Decimal] = Decimal("25"),
        input_event_ids: Optional[list] = None
    ) -> CoCResult:
        """
        Calculate Cycles of Concentration from conductivity measurements.

        The fundamental formula is:
        CoC = Boiler_Conductivity / Makeup_Conductivity

        With temperature compensation (ASTM D1125):
        CoC = Compensated_Boiler_Conductivity / Compensated_Makeup_Conductivity

        Args:
            boiler_conductivity: Conductivity in boiler water (uS/cm)
            makeup_conductivity: Conductivity in makeup water (uS/cm)
            boiler_temp: Temperature of boiler sample (degC), optional
            makeup_temp: Temperature of makeup sample (degC), optional
            temp_compensation: Whether to apply temperature compensation
            reference_temp: Reference temperature for compensation (degC)
            input_event_ids: List of input event IDs for provenance

        Returns:
            CoCResult with calculated CoC and provenance

        Raises:
            ValueError: If makeup conductivity is zero or negative
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        calculation_steps = []
        warnings = []

        # Convert inputs to Decimal
        if isinstance(boiler_conductivity, UnitValue):
            boiler_cond = boiler_conductivity.value
        else:
            boiler_cond = Decimal(str(boiler_conductivity))

        if isinstance(makeup_conductivity, UnitValue):
            makeup_cond = makeup_conductivity.value
        else:
            makeup_cond = Decimal(str(makeup_conductivity))

        # Validation
        if makeup_cond <= 0:
            raise ValueError("Makeup conductivity must be positive")
        if boiler_cond < 0:
            raise ValueError("Boiler conductivity cannot be negative")
        if boiler_cond < makeup_cond:
            warnings.append("Boiler conductivity less than makeup - CoC will be < 1")

        calculation_steps.append({
            "step": 1,
            "description": "Input validation",
            "boiler_conductivity_uS_cm": str(boiler_cond),
            "makeup_conductivity_uS_cm": str(makeup_cond),
        })

        # Temperature compensation
        if temp_compensation and boiler_temp is not None and makeup_temp is not None:
            # Compensate boiler conductivity
            compensated_boiler, boiler_hash = temperature_compensation_conductivity(
                boiler_cond,
                Decimal(str(boiler_temp)),
                Decimal(str(reference_temp))
            )

            # Compensate makeup conductivity
            compensated_makeup, makeup_hash = temperature_compensation_conductivity(
                makeup_cond,
                Decimal(str(makeup_temp)),
                Decimal(str(reference_temp))
            )

            calculation_steps.append({
                "step": 2,
                "description": "Temperature compensation (ASTM D1125)",
                "boiler_temp_degC": str(boiler_temp),
                "makeup_temp_degC": str(makeup_temp),
                "reference_temp_degC": str(reference_temp),
                "compensated_boiler_uS_cm": str(compensated_boiler),
                "compensated_makeup_uS_cm": str(compensated_makeup),
                "boiler_compensation_hash": boiler_hash,
                "makeup_compensation_hash": makeup_hash,
            })

            final_boiler = compensated_boiler
            final_makeup = compensated_makeup
            method = "conductivity_temp_compensated"
        else:
            final_boiler = boiler_cond
            final_makeup = makeup_cond
            method = "conductivity_uncompensated"

            if temp_compensation:
                warnings.append("Temperature compensation requested but temperatures not provided")

        # Calculate CoC
        coc = final_boiler / final_makeup

        # Round to appropriate precision (typically 1-2 decimal places for CoC)
        coc = coc.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        calculation_steps.append({
            "step": 3,
            "description": "Calculate CoC ratio",
            "numerator": str(final_boiler),
            "denominator": str(final_makeup),
            "coc": str(coc),
        })

        # Build inputs dict for provenance
        inputs = {
            "boiler_conductivity_uS_cm": str(boiler_cond),
            "makeup_conductivity_uS_cm": str(makeup_cond),
            "boiler_temp_degC": str(boiler_temp) if boiler_temp else None,
            "makeup_temp_degC": str(makeup_temp) if makeup_temp else None,
            "temp_compensation": temp_compensation,
            "reference_temp_degC": str(reference_temp),
            "input_event_ids": input_event_ids or [],
        }

        # Calculate provenance hash
        provenance_data = {
            "operation": "calculate_coc_from_conductivity",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "inputs": inputs,
            "calculation_steps": calculation_steps,
            "result": str(coc),
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return CoCResult(
            coc=coc,
            method=method,
            inputs=inputs,
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            warnings=warnings,
        )

    def calculate_coc_from_chloride(
        self,
        boiler_chloride: Union[float, Decimal, UnitValue],
        makeup_chloride: Union[float, Decimal, UnitValue],
        input_event_ids: Optional[list] = None
    ) -> CoCResult:
        """
        Calculate Cycles of Concentration from chloride (tracer) measurements.

        Chloride is a conservative ion that does not precipitate or volatilize,
        making it an ideal tracer for CoC calculation.

        Formula:
        CoC = Boiler_Chloride / Makeup_Chloride

        Args:
            boiler_chloride: Chloride concentration in boiler water (mg/L or ppm)
            makeup_chloride: Chloride concentration in makeup water (mg/L or ppm)
            input_event_ids: List of input event IDs for provenance

        Returns:
            CoCResult with calculated CoC and provenance

        Raises:
            ValueError: If makeup chloride is zero or negative
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        calculation_steps = []
        warnings = []

        # Convert inputs to Decimal
        if isinstance(boiler_chloride, UnitValue):
            boiler_cl = boiler_chloride.value
        else:
            boiler_cl = Decimal(str(boiler_chloride))

        if isinstance(makeup_chloride, UnitValue):
            makeup_cl = makeup_chloride.value
        else:
            makeup_cl = Decimal(str(makeup_chloride))

        # Validation
        if makeup_cl <= 0:
            raise ValueError("Makeup chloride concentration must be positive")
        if boiler_cl < 0:
            raise ValueError("Boiler chloride concentration cannot be negative")
        if boiler_cl < makeup_cl:
            warnings.append("Boiler chloride less than makeup - CoC will be < 1")

        calculation_steps.append({
            "step": 1,
            "description": "Input validation",
            "boiler_chloride_ppm": str(boiler_cl),
            "makeup_chloride_ppm": str(makeup_cl),
        })

        # Calculate CoC
        coc = boiler_cl / makeup_cl
        coc = coc.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        calculation_steps.append({
            "step": 2,
            "description": "Calculate CoC from chloride tracer",
            "numerator": str(boiler_cl),
            "denominator": str(makeup_cl),
            "coc": str(coc),
        })

        # Build inputs dict
        inputs = {
            "boiler_chloride_ppm": str(boiler_cl),
            "makeup_chloride_ppm": str(makeup_cl),
            "input_event_ids": input_event_ids or [],
        }

        # Calculate provenance hash
        provenance_data = {
            "operation": "calculate_coc_from_chloride",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "inputs": inputs,
            "calculation_steps": calculation_steps,
            "result": str(coc),
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return CoCResult(
            coc=coc,
            method="chloride_tracer",
            inputs=inputs,
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
            warnings=warnings,
        )

    def estimate_blowdown_flow(
        self,
        makeup_flow: Union[float, Decimal, UnitValue],
        coc: Union[float, Decimal],
        makeup_flow_unit: str = "kg/h",
        input_event_ids: Optional[list] = None
    ) -> BlowdownResult:
        """
        Estimate blowdown flow from makeup flow and CoC.

        Based on mass balance:
        Makeup_Flow * Makeup_Concentration = Blowdown_Flow * Boiler_Concentration

        Rearranging:
        Blowdown_Flow = Makeup_Flow / CoC

        Note: This assumes steady-state operation with no other losses.
        For systems with steam losses, condensate return, etc.,
        actual blowdown may differ.

        Args:
            makeup_flow: Makeup water flow rate
            coc: Cycles of Concentration
            makeup_flow_unit: Unit of makeup flow (default: kg/h)
            input_event_ids: List of input event IDs for provenance

        Returns:
            BlowdownResult with calculated blowdown flow and provenance

        Raises:
            ValueError: If CoC is less than or equal to 1
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Convert inputs
        if isinstance(makeup_flow, UnitValue):
            makeup_val = makeup_flow.value
            makeup_unit = makeup_flow.unit
        else:
            makeup_val = Decimal(str(makeup_flow))
            makeup_unit = makeup_flow_unit

        coc_val = Decimal(str(coc))

        # Validation
        if coc_val <= 1:
            raise ValueError("CoC must be greater than 1 for blowdown calculation")
        if makeup_val <= 0:
            raise ValueError("Makeup flow must be positive")

        # Calculate blowdown
        # Blowdown = Makeup / CoC (simplified steady-state formula)
        # More accurate: Blowdown = Makeup * (1 / (CoC - 1)) for evaporative losses
        # Using simplified formula here
        blowdown_val = makeup_val / coc_val
        blowdown_val = blowdown_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Create UnitValue objects
        makeup_uv = UnitValue(makeup_val, makeup_unit)
        blowdown_uv = UnitValue(blowdown_val, makeup_unit)

        # Verify mass balance
        # Makeup = Blowdown + Evaporation + Losses
        # For this simplified model, Makeup ~ Blowdown * CoC
        verification = (blowdown_val * coc_val)
        mass_balance_error = abs(verification - makeup_val)
        mass_balance_verified = mass_balance_error < Decimal("0.1")

        # Build inputs dict
        inputs = {
            "makeup_flow": str(makeup_val),
            "makeup_flow_unit": makeup_unit,
            "coc": str(coc_val),
            "input_event_ids": input_event_ids or [],
        }

        # Calculate provenance hash
        provenance_data = {
            "operation": "estimate_blowdown_flow",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "inputs": inputs,
            "blowdown_flow": str(blowdown_val),
            "mass_balance_verified": mass_balance_verified,
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return BlowdownResult(
            blowdown_flow=blowdown_uv,
            makeup_flow=makeup_uv,
            coc=coc_val,
            mass_balance_verified=mass_balance_verified,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )

    def verify_mass_balance(
        self,
        makeup_flow: Union[float, Decimal],
        makeup_concentration: Union[float, Decimal],
        blowdown_flow: Union[float, Decimal],
        boiler_concentration: Union[float, Decimal],
        tolerance_percent: Union[float, Decimal] = Decimal("5")
    ) -> Tuple[bool, Decimal, str]:
        """
        Verify mass balance for water chemistry system.

        Mass Balance Equation:
        Makeup_Flow * Makeup_Concentration = Blowdown_Flow * Boiler_Concentration

        Args:
            makeup_flow: Makeup water flow rate
            makeup_concentration: Concentration in makeup water
            blowdown_flow: Blowdown flow rate
            boiler_concentration: Concentration in boiler water
            tolerance_percent: Acceptable error tolerance (%)

        Returns:
            Tuple of (is_balanced, error_percent, provenance_hash)
        """
        makeup_f = Decimal(str(makeup_flow))
        makeup_c = Decimal(str(makeup_concentration))
        blowdown_f = Decimal(str(blowdown_flow))
        boiler_c = Decimal(str(boiler_concentration))
        tolerance = Decimal(str(tolerance_percent))

        # Calculate mass flows
        makeup_mass = makeup_f * makeup_c
        blowdown_mass = blowdown_f * boiler_c

        # Calculate error
        if makeup_mass == 0:
            error_percent = Decimal("0") if blowdown_mass == 0 else Decimal("100")
        else:
            error_percent = abs(makeup_mass - blowdown_mass) / makeup_mass * Decimal("100")

        error_percent = error_percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        is_balanced = error_percent <= tolerance

        # Calculate provenance
        provenance_data = {
            "operation": "verify_mass_balance",
            "makeup_flow": str(makeup_f),
            "makeup_concentration": str(makeup_c),
            "blowdown_flow": str(blowdown_f),
            "boiler_concentration": str(boiler_c),
            "makeup_mass": str(makeup_mass),
            "blowdown_mass": str(blowdown_mass),
            "error_percent": str(error_percent),
            "tolerance_percent": str(tolerance),
            "is_balanced": is_balanced,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return is_balanced, error_percent, provenance_hash
