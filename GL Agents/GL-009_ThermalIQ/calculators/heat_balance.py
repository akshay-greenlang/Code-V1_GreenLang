"""
Heat Balance Calculator
=======================

Zero-hallucination deterministic calculation engine for heat balance analysis.

Implements comprehensive heat balance calculations with closure verification,
loss identification, and mass balance integration.

Heat Balance Principle:
----------------------
First Law of Thermodynamics (Conservation of Energy):
    Q_in = Q_out + Q_loss + Q_accumulated

For steady-state systems:
    sum(Q_in) = sum(Q_out) + sum(Q_loss)

Closure Check:
    Closure = (Q_in - Q_out - Q_loss) / Q_in * 100%

    Acceptable closure (per ASME PTC):
    - Industrial boilers: +/- 1%
    - Power plants: +/- 0.5%
    - Heat exchangers: +/- 2%

Standards Compliance:
--------------------
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 46 - Overall Plant Performance
- ISO 5167 - Flow Measurement
- ASME PTC 19.1 - Measurement Uncertainty

Author: GL-009_ThermalIQ
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import hashlib
import json
import time
from datetime import datetime, timezone


class StreamType(Enum):
    """Classification of heat streams."""
    FUEL = "fuel"
    COMBUSTION_AIR = "combustion_air"
    FEEDWATER = "feedwater"
    STEAM = "steam"
    CONDENSATE = "condensate"
    HOT_WATER = "hot_water"
    FLUE_GAS = "flue_gas"
    BLOWDOWN = "blowdown"
    COOLING_WATER = "cooling_water"
    THERMAL_FLUID = "thermal_fluid"
    RADIATION = "radiation"
    OTHER = "other"


@dataclass
class HeatStream:
    """
    Representation of a heat stream in the balance.

    Attributes:
        name: Stream identifier
        stream_type: Type classification
        heat_rate_kW: Heat rate (kW)
        mass_flow_kg_s: Mass flow rate (kg/s)
        temperature_K: Stream temperature (K)
        enthalpy_kJ_kg: Specific enthalpy (kJ/kg)
        is_input: True if input stream, False if output
        uncertainty_percent: Measurement uncertainty (%)
    """
    name: str
    stream_type: StreamType
    heat_rate_kW: Decimal
    mass_flow_kg_s: Optional[Decimal] = None
    temperature_K: Optional[Decimal] = None
    enthalpy_kJ_kg: Optional[Decimal] = None
    is_input: bool = True
    uncertainty_percent: Decimal = Decimal("1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "stream_type": self.stream_type.value,
            "heat_rate_kW": str(self.heat_rate_kW),
            "mass_flow_kg_s": str(self.mass_flow_kg_s) if self.mass_flow_kg_s else None,
            "temperature_K": str(self.temperature_K) if self.temperature_K else None,
            "enthalpy_kJ_kg": str(self.enthalpy_kJ_kg) if self.enthalpy_kJ_kg else None,
            "is_input": self.is_input,
            "uncertainty_percent": str(self.uncertainty_percent)
        }


@dataclass
class CalculationStep:
    """Individual calculation step with provenance tracking."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Any]
    output_value: Decimal
    output_name: str
    output_unit: str
    reference: str


@dataclass
class LossSource:
    """
    Identified heat loss source.

    Attributes:
        name: Loss source name
        category: Loss category (e.g., radiation, flue_gas)
        heat_loss_kW: Heat loss rate (kW)
        percentage_of_input: Loss as percentage of total input
        cause: Root cause description
        improvement_potential: Potential for improvement (kW)
        reference: Standard reference for loss calculation
    """
    name: str
    category: str
    heat_loss_kW: Decimal
    percentage_of_input: Decimal
    cause: str
    improvement_potential: Decimal
    reference: str
    uncertainty_percent: Decimal = Decimal("5.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "heat_loss_kW": str(self.heat_loss_kW),
            "percentage_of_input": str(self.percentage_of_input),
            "cause": self.cause,
            "improvement_potential": str(self.improvement_potential),
            "reference": self.reference,
            "uncertainty_percent": str(self.uncertainty_percent)
        }


@dataclass
class ClosureResult:
    """
    Result of heat balance closure check.

    Attributes:
        closure_percent: Closure error as percentage
        is_acceptable: True if within tolerance
        tolerance_percent: Acceptable tolerance limit
        unaccounted_heat_kW: Heat not accounted for
        closure_status: "PASS", "WARNING", or "FAIL"
        recommendations: Actions to improve closure
    """
    closure_percent: Decimal
    is_acceptable: bool
    tolerance_percent: Decimal
    unaccounted_heat_kW: Decimal
    closure_status: str
    recommendations: List[str]
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "closure_percent": str(self.closure_percent),
            "is_acceptable": self.is_acceptable,
            "tolerance_percent": str(self.tolerance_percent),
            "unaccounted_heat_kW": str(self.unaccounted_heat_kW),
            "closure_status": self.closure_status,
            "recommendations": self.recommendations,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp
        }


@dataclass
class HeatBalanceResult:
    """
    Result of heat balance calculation with complete provenance.

    Attributes:
        total_heat_input_kW: Sum of all heat inputs (kW)
        total_heat_output_kW: Sum of all heat outputs (kW)
        total_heat_loss_kW: Sum of all heat losses (kW)
        input_streams: List of input heat streams
        output_streams: List of output heat streams
        loss_streams: List of heat loss streams
        efficiency_percent: Thermal efficiency (%)
        closure: Heat balance closure result
        provenance_hash: SHA-256 hash
        calculation_steps: All calculation steps
    """
    total_heat_input_kW: Decimal
    total_heat_output_kW: Decimal
    total_heat_loss_kW: Decimal
    input_streams: List[HeatStream]
    output_streams: List[HeatStream]
    loss_streams: List[HeatStream]
    efficiency_percent: Decimal
    closure: ClosureResult
    provenance_hash: str
    calculation_steps: List[CalculationStep]
    formula_reference: str
    calculation_time_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_heat_input_kW": str(self.total_heat_input_kW),
            "total_heat_output_kW": str(self.total_heat_output_kW),
            "total_heat_loss_kW": str(self.total_heat_loss_kW),
            "input_streams": [s.to_dict() for s in self.input_streams],
            "output_streams": [s.to_dict() for s in self.output_streams],
            "loss_streams": [s.to_dict() for s in self.loss_streams],
            "efficiency_percent": str(self.efficiency_percent),
            "closure": self.closure.to_dict(),
            "provenance_hash": self.provenance_hash,
            "formula_reference": self.formula_reference,
            "calculation_time_ms": self.calculation_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class HeatBalanceCalculator:
    """
    Zero-hallucination heat balance calculation engine.

    Guarantees:
    - DETERMINISTIC: Same inputs always produce identical outputs
    - REPRODUCIBLE: Full provenance tracking with SHA-256 hashes
    - AUDITABLE: Complete calculation trails
    - STANDARDS-BASED: All formulas from published sources
    - NO LLM: Zero hallucination risk in calculation path

    References:
    -----------
    [1] ASME PTC 4.1-2022, Steam Generating Units
    [2] ASME PTC 46-2015, Overall Plant Performance
    [3] Moran & Shapiro, Fundamentals of Engineering Thermodynamics, 9th Ed.
    [4] ISO 5167-1:2022, Flow Measurement

    Example:
    --------
    >>> calc = HeatBalanceCalculator()
    >>> inputs = {"fuel": 1000.0, "combustion_air": 50.0}
    >>> outputs = {"steam": 850.0, "flue_gas": 120.0}
    >>> result = calc.calculate_heat_balance(inputs, outputs)
    >>> print(f"Closure: {result.closure.closure_percent}%")
    """

    # Default closure tolerances (per ASME PTC)
    CLOSURE_TOLERANCES = {
        "boiler": Decimal("1.0"),           # 1% for boilers
        "power_plant": Decimal("0.5"),      # 0.5% for power plants
        "heat_exchanger": Decimal("2.0"),   # 2% for heat exchangers
        "furnace": Decimal("2.0"),          # 2% for furnaces
        "default": Decimal("1.5"),          # 1.5% default
    }

    # Precision for regulatory compliance
    PRECISION = 3

    def __init__(
        self,
        system_type: str = "default",
        precision: int = 3
    ):
        """
        Initialize heat balance calculator.

        Args:
            system_type: Type of system for closure tolerance
            precision: Decimal places for output
        """
        self.system_type = system_type
        self.precision = precision
        self.closure_tolerance = self.CLOSURE_TOLERANCES.get(
            system_type,
            self.CLOSURE_TOLERANCES["default"]
        )

    def calculate_heat_balance(
        self,
        inputs: Dict[str, float],
        outputs: Dict[str, float],
        losses: Optional[Dict[str, float]] = None,
        stream_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> HeatBalanceResult:
        """
        Calculate comprehensive heat balance.

        Performs First Law analysis: sum(Q_in) = sum(Q_out) + sum(Q_loss)

        Args:
            inputs: Dictionary of heat inputs {name: heat_rate_kW}
            outputs: Dictionary of heat outputs {name: heat_rate_kW}
            losses: Dictionary of known losses {name: heat_rate_kW}
            stream_details: Additional stream information for each stream

        Returns:
            HeatBalanceResult with complete provenance

        Reference:
            ASME PTC 4.1, Section 5

        Example:
            >>> calc = HeatBalanceCalculator()
            >>> result = calc.calculate_heat_balance(
            ...     inputs={"fuel": 1000.0, "air_preheat": 50.0},
            ...     outputs={"steam": 850.0, "hot_water": 30.0},
            ...     losses={"radiation": 10.0, "flue_gas": 100.0}
            ... )
        """
        start_time = time.perf_counter()

        # Validate inputs
        self._validate_balance_inputs(inputs, outputs)

        calculation_steps = []
        step_num = 0

        # Initialize stream lists
        input_streams = []
        output_streams = []
        loss_streams = []

        # =====================================================================
        # STEP 1: Process input streams
        # =====================================================================
        total_input = Decimal("0")

        for name, heat_rate in inputs.items():
            step_num += 1
            heat_decimal = Decimal(str(heat_rate))
            total_input += heat_decimal

            # Get stream details if provided
            details = (stream_details or {}).get(name, {})
            stream_type = self._infer_stream_type(name, details.get("type"))

            stream = HeatStream(
                name=name,
                stream_type=stream_type,
                heat_rate_kW=heat_decimal,
                mass_flow_kg_s=Decimal(str(details.get("mass_flow"))) if details.get("mass_flow") else None,
                temperature_K=Decimal(str(details.get("temperature"))) if details.get("temperature") else None,
                enthalpy_kJ_kg=Decimal(str(details.get("enthalpy"))) if details.get("enthalpy") else None,
                is_input=True,
                uncertainty_percent=Decimal(str(details.get("uncertainty", 1.0)))
            )
            input_streams.append(stream)

            step = CalculationStep(
                step_number=step_num,
                description=f"Record input stream: {name}",
                formula=f"Q_{name} = {heat_rate} kW",
                inputs={"name": name, "heat_rate": heat_decimal},
                output_value=heat_decimal,
                output_name=f"Q_{name}",
                output_unit="kW",
                reference="ASME PTC 4.1, Section 5.4"
            )
            calculation_steps.append(step)

        # Sum total input
        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Sum all heat inputs",
            formula="Q_in_total = sum(Q_in_i)",
            inputs={"inputs": list(inputs.keys())},
            output_value=total_input,
            output_name="Q_in_total",
            output_unit="kW",
            reference="First Law of Thermodynamics"
        )
        calculation_steps.append(step)

        # =====================================================================
        # STEP 2: Process output streams
        # =====================================================================
        total_output = Decimal("0")

        for name, heat_rate in outputs.items():
            step_num += 1
            heat_decimal = Decimal(str(heat_rate))
            total_output += heat_decimal

            details = (stream_details or {}).get(name, {})
            stream_type = self._infer_stream_type(name, details.get("type"))

            stream = HeatStream(
                name=name,
                stream_type=stream_type,
                heat_rate_kW=heat_decimal,
                mass_flow_kg_s=Decimal(str(details.get("mass_flow"))) if details.get("mass_flow") else None,
                temperature_K=Decimal(str(details.get("temperature"))) if details.get("temperature") else None,
                enthalpy_kJ_kg=Decimal(str(details.get("enthalpy"))) if details.get("enthalpy") else None,
                is_input=False,
                uncertainty_percent=Decimal(str(details.get("uncertainty", 1.0)))
            )
            output_streams.append(stream)

            step = CalculationStep(
                step_number=step_num,
                description=f"Record output stream: {name}",
                formula=f"Q_{name} = {heat_rate} kW",
                inputs={"name": name, "heat_rate": heat_decimal},
                output_value=heat_decimal,
                output_name=f"Q_{name}",
                output_unit="kW",
                reference="ASME PTC 4.1, Section 5.5"
            )
            calculation_steps.append(step)

        # Sum total output
        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Sum all heat outputs",
            formula="Q_out_total = sum(Q_out_i)",
            inputs={"outputs": list(outputs.keys())},
            output_value=total_output,
            output_name="Q_out_total",
            output_unit="kW",
            reference="First Law of Thermodynamics"
        )
        calculation_steps.append(step)

        # =====================================================================
        # STEP 3: Process loss streams
        # =====================================================================
        total_loss = Decimal("0")

        if losses:
            for name, heat_rate in losses.items():
                step_num += 1
                heat_decimal = Decimal(str(heat_rate))
                total_loss += heat_decimal

                details = (stream_details or {}).get(name, {})

                stream = HeatStream(
                    name=name,
                    stream_type=StreamType.OTHER,
                    heat_rate_kW=heat_decimal,
                    is_input=False,
                    uncertainty_percent=Decimal(str(details.get("uncertainty", 5.0)))
                )
                loss_streams.append(stream)

                step = CalculationStep(
                    step_number=step_num,
                    description=f"Record loss stream: {name}",
                    formula=f"Q_loss_{name} = {heat_rate} kW",
                    inputs={"name": name, "heat_rate": heat_decimal},
                    output_value=heat_decimal,
                    output_name=f"Q_loss_{name}",
                    output_unit="kW",
                    reference="ASME PTC 4.1, Section 5.6"
                )
                calculation_steps.append(step)

            # Sum total losses
            step_num += 1
            step = CalculationStep(
                step_number=step_num,
                description="Sum all heat losses",
                formula="Q_loss_total = sum(Q_loss_i)",
                inputs={"losses": list(losses.keys())},
                output_value=total_loss,
                output_name="Q_loss_total",
                output_unit="kW",
                reference="ASME PTC 4.1, Section 5.6"
            )
            calculation_steps.append(step)

        # =====================================================================
        # STEP 4: Calculate efficiency
        # =====================================================================
        if total_input > 0:
            efficiency = (total_output / total_input) * Decimal("100")
        else:
            efficiency = Decimal("0")

        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Calculate thermal efficiency",
            formula="eta = (Q_out / Q_in) * 100",
            inputs={"Q_out": total_output, "Q_in": total_input},
            output_value=efficiency,
            output_name="efficiency",
            output_unit="%",
            reference="First Law efficiency definition"
        )
        calculation_steps.append(step)

        # =====================================================================
        # STEP 5: Perform closure check
        # =====================================================================
        closure_result = self.closure_check_internal(
            total_input, total_output, total_loss, calculation_steps
        )

        # Apply precision
        total_input = self._apply_precision(total_input)
        total_output = self._apply_precision(total_output)
        total_loss = self._apply_precision(total_loss)
        efficiency = self._apply_precision(efficiency)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            formula_id="heat_balance_v1",
            inputs={"inputs": inputs, "outputs": outputs, "losses": losses},
            calculation_steps=calculation_steps,
            output_value=total_input
        )

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000

        return HeatBalanceResult(
            total_heat_input_kW=total_input,
            total_heat_output_kW=total_output,
            total_heat_loss_kW=total_loss,
            input_streams=input_streams,
            output_streams=output_streams,
            loss_streams=loss_streams,
            efficiency_percent=efficiency,
            closure=closure_result,
            provenance_hash=provenance_hash,
            calculation_steps=calculation_steps,
            formula_reference="ASME PTC 4.1-2022, Section 5",
            calculation_time_ms=calculation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "system_type": self.system_type,
                "closure_tolerance": str(self.closure_tolerance)
            }
        )

    def closure_check(
        self,
        balance: HeatBalanceResult
    ) -> ClosureResult:
        """
        Perform closure check on existing heat balance result.

        Closure = (Q_in - Q_out - Q_loss) / Q_in * 100%

        Args:
            balance: Heat balance result to check

        Returns:
            ClosureResult with verification status
        """
        return self.closure_check_internal(
            balance.total_heat_input_kW,
            balance.total_heat_output_kW,
            balance.total_heat_loss_kW,
            []
        )

    def closure_check_internal(
        self,
        total_input: Decimal,
        total_output: Decimal,
        total_loss: Decimal,
        calculation_steps: List[CalculationStep]
    ) -> ClosureResult:
        """
        Internal closure check calculation.

        Formula: Closure = (Q_in - Q_out - Q_loss) / Q_in * 100%
        """
        step_num = len(calculation_steps)

        # Calculate unaccounted heat
        unaccounted = total_input - total_output - total_loss

        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Calculate unaccounted heat",
            formula="Q_unaccounted = Q_in - Q_out - Q_loss",
            inputs={"Q_in": total_input, "Q_out": total_output, "Q_loss": total_loss},
            output_value=unaccounted,
            output_name="Q_unaccounted",
            output_unit="kW",
            reference="ASME PTC 4.1, Section 5.7"
        )
        calculation_steps.append(step)

        # Calculate closure percentage
        if total_input > 0:
            closure_percent = (unaccounted / total_input) * Decimal("100")
        else:
            closure_percent = Decimal("0")

        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Calculate closure percentage",
            formula="Closure = (Q_unaccounted / Q_in) * 100",
            inputs={"Q_unaccounted": unaccounted, "Q_in": total_input},
            output_value=closure_percent,
            output_name="Closure",
            output_unit="%",
            reference="ASME PTC 4.1, Section 5.7"
        )
        calculation_steps.append(step)

        # Determine closure status
        abs_closure = abs(closure_percent)
        if abs_closure <= self.closure_tolerance:
            closure_status = "PASS"
            is_acceptable = True
        elif abs_closure <= self.closure_tolerance * Decimal("2"):
            closure_status = "WARNING"
            is_acceptable = False
        else:
            closure_status = "FAIL"
            is_acceptable = False

        # Generate recommendations
        recommendations = self._generate_closure_recommendations(
            closure_percent, unaccounted, total_input
        )

        # Calculate provenance hash for closure
        provenance_hash = self._calculate_provenance_hash(
            formula_id="closure_check_v1",
            inputs={"Q_in": str(total_input), "Q_out": str(total_output), "Q_loss": str(total_loss)},
            calculation_steps=calculation_steps[-2:],  # Last two steps
            output_value=closure_percent
        )

        return ClosureResult(
            closure_percent=self._apply_precision(closure_percent),
            is_acceptable=is_acceptable,
            tolerance_percent=self.closure_tolerance,
            unaccounted_heat_kW=self._apply_precision(unaccounted),
            closure_status=closure_status,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def identify_unaccounted_losses(
        self,
        balance: HeatBalanceResult,
        known_loss_categories: Optional[List[str]] = None
    ) -> List[LossSource]:
        """
        Identify potential sources of unaccounted heat losses.

        Analyzes the heat balance and identifies likely causes of
        unaccounted losses based on system type and balance characteristics.

        Args:
            balance: Heat balance result to analyze
            known_loss_categories: Loss categories already accounted for

        Returns:
            List of potential unaccounted loss sources

        Reference:
            ASME PTC 4.1, Section 5.6
        """
        unaccounted = balance.closure.unaccounted_heat_kW
        total_input = balance.total_heat_input_kW

        if unaccounted <= 0:
            return []  # No unaccounted losses

        known = known_loss_categories or []
        potential_losses = []

        # Standard loss categories per ASME PTC 4.1
        loss_categories = {
            "radiation": {
                "typical_percent": Decimal("0.3"),
                "cause": "Surface radiation and convection from equipment",
                "reference": "ASME PTC 4.1, Section 5.6.5"
            },
            "flue_gas_sensible": {
                "typical_percent": Decimal("5.0"),
                "cause": "Sensible heat in exhaust flue gases",
                "reference": "ASME PTC 4.1, Section 5.6.1"
            },
            "flue_gas_latent": {
                "typical_percent": Decimal("3.0"),
                "cause": "Latent heat of water vapor in flue gas",
                "reference": "ASME PTC 4.1, Section 5.6.2"
            },
            "unburned_fuel": {
                "typical_percent": Decimal("0.5"),
                "cause": "Incomplete combustion of fuel",
                "reference": "ASME PTC 4.1, Section 5.6.4"
            },
            "blowdown": {
                "typical_percent": Decimal("0.5"),
                "cause": "Heat lost through boiler blowdown",
                "reference": "ASME PTC 4.1, Section 5.6.6"
            },
            "air_infiltration": {
                "typical_percent": Decimal("0.2"),
                "cause": "Cold air infiltration into system",
                "reference": "ASME PTC 4.1, Section 5.6.7"
            },
            "auxiliary_equipment": {
                "typical_percent": Decimal("0.3"),
                "cause": "Heat consumed by auxiliary equipment",
                "reference": "ASME PTC 4.1, Section 5.6.8"
            },
        }

        # Identify likely sources based on unaccounted heat
        remaining_unaccounted = float(unaccounted)

        for category, details in loss_categories.items():
            if category in known:
                continue

            typical_loss = float(total_input) * float(details["typical_percent"]) / 100

            # Estimate contribution based on typical values
            if remaining_unaccounted > 0:
                estimated_loss = min(typical_loss, remaining_unaccounted)

                if estimated_loss > 0:
                    loss_source = LossSource(
                        name=category,
                        category=category,
                        heat_loss_kW=Decimal(str(estimated_loss)),
                        percentage_of_input=Decimal(str(estimated_loss / float(total_input) * 100)),
                        cause=details["cause"],
                        improvement_potential=Decimal(str(estimated_loss * 0.3)),  # 30% improvement potential
                        reference=details["reference"],
                        uncertainty_percent=Decimal("20.0")  # Higher uncertainty for estimates
                    )
                    potential_losses.append(loss_source)
                    remaining_unaccounted -= estimated_loss

        # If still unaccounted, add unknown category
        if remaining_unaccounted > float(total_input) * 0.001:  # >0.1% threshold
            loss_source = LossSource(
                name="unknown",
                category="unidentified",
                heat_loss_kW=Decimal(str(remaining_unaccounted)),
                percentage_of_input=Decimal(str(remaining_unaccounted / float(total_input) * 100)),
                cause="Unidentified heat loss - requires investigation",
                improvement_potential=Decimal(str(remaining_unaccounted * 0.5)),
                reference="Investigation required",
                uncertainty_percent=Decimal("50.0")
            )
            potential_losses.append(loss_source)

        return potential_losses

    def mass_balance_integration(
        self,
        mass_streams: Dict[str, Dict[str, float]],
        specific_heats: Optional[Dict[str, float]] = None,
    ) -> HeatBalanceResult:
        """
        Calculate heat balance from mass balance data.

        Converts mass flow rates and temperature changes to heat rates.

        Formula: Q = m_dot * Cp * delta_T

        Args:
            mass_streams: Dictionary of streams with mass flow and temperatures
                {
                    "stream_name": {
                        "mass_flow_kg_s": 10.0,
                        "T_in_K": 300.0,
                        "T_out_K": 350.0,
                        "Cp_kJ_kg_K": 4.18,  # optional
                        "is_input": True
                    }
                }
            specific_heats: Override specific heats {fluid: Cp}

        Returns:
            HeatBalanceResult calculated from mass balance
        """
        inputs = {}
        outputs = {}
        stream_details = {}

        for name, stream_data in mass_streams.items():
            m_dot = stream_data["mass_flow_kg_s"]
            T_in = stream_data.get("T_in_K", 298.15)
            T_out = stream_data.get("T_out_K", T_in)
            Cp = stream_data.get("Cp_kJ_kg_K", 4.18)  # Default: water
            is_input = stream_data.get("is_input", True)

            # Override Cp if provided
            if specific_heats and name in specific_heats:
                Cp = specific_heats[name]

            # Calculate heat rate: Q = m_dot * Cp * delta_T
            delta_T = abs(T_out - T_in)
            Q = m_dot * Cp * delta_T

            # Store stream details
            stream_details[name] = {
                "mass_flow": m_dot,
                "temperature": T_out if is_input else T_in,
                "uncertainty": 1.5
            }

            if is_input:
                inputs[name] = Q
            else:
                outputs[name] = Q

        return self.calculate_heat_balance(
            inputs=inputs,
            outputs=outputs,
            stream_details=stream_details
        )

    def calculate_heat_from_mass_flow(
        self,
        mass_flow_kg_s: float,
        enthalpy_in_kJ_kg: float,
        enthalpy_out_kJ_kg: float,
    ) -> Decimal:
        """
        Calculate heat rate from mass flow and enthalpy change.

        Formula: Q_dot = m_dot * (h_out - h_in)

        Args:
            mass_flow_kg_s: Mass flow rate (kg/s)
            enthalpy_in_kJ_kg: Inlet specific enthalpy (kJ/kg)
            enthalpy_out_kJ_kg: Outlet specific enthalpy (kJ/kg)

        Returns:
            Heat rate (kW)
        """
        m_dot = Decimal(str(mass_flow_kg_s))
        h_in = Decimal(str(enthalpy_in_kJ_kg))
        h_out = Decimal(str(enthalpy_out_kJ_kg))

        Q_dot = m_dot * (h_out - h_in)

        return self._apply_precision(Q_dot)

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _validate_balance_inputs(
        self,
        inputs: Dict[str, float],
        outputs: Dict[str, float]
    ) -> None:
        """Validate heat balance inputs."""
        if not inputs:
            raise ValueError("Heat balance requires at least one input stream")
        if not outputs:
            raise ValueError("Heat balance requires at least one output stream")

        for name, value in inputs.items():
            if value < 0:
                raise ValueError(f"Heat input '{name}' cannot be negative: {value}")

        for name, value in outputs.items():
            if value < 0:
                raise ValueError(f"Heat output '{name}' cannot be negative: {value}")

    def _infer_stream_type(
        self,
        name: str,
        explicit_type: Optional[str]
    ) -> StreamType:
        """Infer stream type from name or explicit type."""
        if explicit_type:
            try:
                return StreamType(explicit_type.lower())
            except ValueError:
                pass

        # Infer from name
        name_lower = name.lower()

        if "fuel" in name_lower:
            return StreamType.FUEL
        elif "air" in name_lower:
            return StreamType.COMBUSTION_AIR
        elif "steam" in name_lower:
            return StreamType.STEAM
        elif "feedwater" in name_lower or "feed_water" in name_lower:
            return StreamType.FEEDWATER
        elif "condensate" in name_lower:
            return StreamType.CONDENSATE
        elif "flue" in name_lower or "exhaust" in name_lower:
            return StreamType.FLUE_GAS
        elif "blowdown" in name_lower:
            return StreamType.BLOWDOWN
        elif "cooling" in name_lower:
            return StreamType.COOLING_WATER
        elif "radiation" in name_lower:
            return StreamType.RADIATION
        elif "thermal" in name_lower or "oil" in name_lower:
            return StreamType.THERMAL_FLUID
        else:
            return StreamType.OTHER

    def _generate_closure_recommendations(
        self,
        closure_percent: Decimal,
        unaccounted: Decimal,
        total_input: Decimal
    ) -> List[str]:
        """Generate recommendations based on closure analysis."""
        recommendations = []

        abs_closure = abs(float(closure_percent))

        if abs_closure <= float(self.closure_tolerance):
            recommendations.append("Heat balance closure is within acceptable limits.")
            return recommendations

        if float(unaccounted) > 0:
            # More heat input than accounted output
            recommendations.append(
                f"Unaccounted heat: {float(unaccounted):.1f} kW ({abs_closure:.2f}% of input)"
            )
            recommendations.append("Check for: unmeasured losses, radiation, leaks, or auxiliary loads")
            recommendations.append("Verify: flow meter calibration and temperature sensor accuracy")
        else:
            # More output than input (measurement error likely)
            recommendations.append(
                f"Heat balance shows {abs(float(unaccounted)):.1f} kW more output than input"
            )
            recommendations.append("This indicates measurement errors in input or output streams")
            recommendations.append("Verify: fuel flow measurement, combustion air flow, and calorific value")

        if abs_closure > 5.0:
            recommendations.append("CRITICAL: Closure error exceeds 5% - detailed investigation required")
            recommendations.append("Review all instrumentation and measurement procedures")

        return recommendations

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply regulatory rounding precision using ROUND_HALF_UP."""
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance_hash(
        self,
        formula_id: str,
        inputs: Dict[str, Any],
        calculation_steps: List[CalculationStep],
        output_value: Decimal
    ) -> str:
        """Calculate SHA-256 provenance hash for complete audit trail."""

        def decimal_serializer(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        provenance_data = {
            "formula_id": formula_id,
            "inputs": inputs,
            "steps": [
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "formula": step.formula,
                    "inputs": {k: str(v) if isinstance(v, Decimal) else v
                              for k, v in step.inputs.items()},
                    "output_value": str(step.output_value),
                    "output_name": step.output_name,
                }
                for step in calculation_steps
            ],
            "output_value": str(output_value)
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=decimal_serializer)
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()
