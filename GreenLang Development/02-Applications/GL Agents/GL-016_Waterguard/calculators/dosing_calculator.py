"""
Chemical Dosing Calculator

This module provides deterministic chemical dosing calculations for
water treatment applications. Supports multiple chemical types with
feedforward and feedback control strategies.

Zero-Hallucination Guarantee:
- All dosing calculations are based on deterministic formulas
- Pump constraints are enforced deterministically
- Complete audit trail with SHA-256 hashes
- No ML/AI in the calculation path
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime

from .units import UnitValue, get_unit_registry


class ChemicalType(Enum):
    """Types of treatment chemicals."""
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    ALKALINITY_BUILDER = "alkalinity_builder"
    DISPERSANT = "dispersant"
    PHOSPHATE = "phosphate"
    BIOCIDE = "biocide"
    SCALE_INHIBITOR = "scale_inhibitor"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    AMINE = "amine"
    GENERIC = "generic"


@dataclass
class ChemicalConfig:
    """Configuration for a treatment chemical."""
    chemical_type: ChemicalType
    chemical_name: str
    active_concentration: Decimal  # % active ingredient
    density: Decimal  # kg/L
    target_dose_ppm: Decimal  # Target dose in ppm (mg/L)
    min_dose_ppm: Optional[Decimal] = None
    max_dose_ppm: Optional[Decimal] = None
    stoichiometric_ratio: Optional[Decimal] = None  # For reaction-based dosing


@dataclass
class PumpConstraints:
    """Constraints for a dosing pump."""
    pump_id: str
    min_stroke_percent: Decimal = Decimal("0")
    max_stroke_percent: Decimal = Decimal("100")
    min_flow_rate: Decimal = Decimal("0")  # L/h
    max_flow_rate: Decimal = Decimal("100")  # L/h
    ramp_rate_limit: Optional[Decimal] = None  # % per second
    current_stroke_percent: Decimal = Decimal("50")


@dataclass
class DosingResult:
    """Result of a dosing calculation."""
    chemical_type: ChemicalType
    chemical_name: str
    calculated_dose_rate: Decimal  # L/h of chemical
    calculated_dose_ppm: Decimal  # ppm in treated water
    stroke_percent: Decimal  # Pump stroke setting
    is_within_constraints: bool
    constraint_messages: List[str]
    calculation_method: str
    calculation_steps: List[Dict[str, Any]]
    provenance_hash: str
    timestamp: str


@dataclass
class ReconciliationResult:
    """Result of comparing commanded vs actual dosing."""
    commanded_dose_rate: Decimal
    actual_dose_rate: Decimal
    deviation_absolute: Decimal
    deviation_percent: Decimal
    is_acceptable: bool
    deviation_threshold: Decimal
    provenance_hash: str
    timestamp: str


class DosingCalculator:
    """
    Chemical dosing calculator for water treatment.

    Provides:
    - Feedforward dosing based on water flow
    - Feedback correction based on measured residual
    - Pump constraint enforcement
    - Dose reconciliation (commanded vs actual)
    - Complete audit trails
    """

    def __init__(
        self,
        config_version: str = "1.0.0",
        code_version: str = "1.0.0"
    ):
        """
        Initialize the dosing calculator.

        Args:
            config_version: Version of the configuration
            code_version: Version of the calculation code
        """
        self.config_version = config_version
        self.code_version = code_version
        self.unit_registry = get_unit_registry()

        # Default chemical configurations
        self.default_chemicals: Dict[ChemicalType, ChemicalConfig] = {
            ChemicalType.OXYGEN_SCAVENGER: ChemicalConfig(
                chemical_type=ChemicalType.OXYGEN_SCAVENGER,
                chemical_name="Sodium Sulfite (Na2SO3)",
                active_concentration=Decimal("25"),  # 25% solution
                density=Decimal("1.2"),
                target_dose_ppm=Decimal("20"),  # 20 ppm residual target
                min_dose_ppm=Decimal("10"),
                max_dose_ppm=Decimal("60"),
                stoichiometric_ratio=Decimal("8"),  # 8 ppm sulfite per 1 ppm O2
            ),
            ChemicalType.ALKALINITY_BUILDER: ChemicalConfig(
                chemical_type=ChemicalType.ALKALINITY_BUILDER,
                chemical_name="Sodium Hydroxide (NaOH)",
                active_concentration=Decimal("50"),  # 50% solution
                density=Decimal("1.53"),
                target_dose_ppm=Decimal("100"),  # Target alkalinity
                min_dose_ppm=Decimal("50"),
                max_dose_ppm=Decimal("200"),
            ),
            ChemicalType.DISPERSANT: ChemicalConfig(
                chemical_type=ChemicalType.DISPERSANT,
                chemical_name="Polymer Dispersant",
                active_concentration=Decimal("30"),
                density=Decimal("1.1"),
                target_dose_ppm=Decimal("10"),
                min_dose_ppm=Decimal("5"),
                max_dose_ppm=Decimal("20"),
            ),
            ChemicalType.PHOSPHATE: ChemicalConfig(
                chemical_type=ChemicalType.PHOSPHATE,
                chemical_name="Trisodium Phosphate (Na3PO4)",
                active_concentration=Decimal("100"),  # Solid
                density=Decimal("2.54"),
                target_dose_ppm=Decimal("20"),  # 20 ppm PO4
                min_dose_ppm=Decimal("10"),
                max_dose_ppm=Decimal("50"),
            ),
        }

    def calculate_feedforward_dose(
        self,
        target_dose_ppm: Union[float, Decimal],
        water_flow: Union[float, Decimal, UnitValue],
        chemical_config: Optional[ChemicalConfig] = None,
        chemical_type: Optional[ChemicalType] = None,
        water_flow_unit: str = "kg/h",
        pump_constraints: Optional[PumpConstraints] = None,
        input_event_ids: Optional[list] = None
    ) -> DosingResult:
        """
        Calculate feedforward chemical dose based on water flow.

        The basic formula is:
        Chemical_Flow (L/h) = Water_Flow (kg/h) * Target_Dose (mg/kg) /
                              (Active_Concentration (%) * Density (kg/L) * 10000)

        Args:
            target_dose_ppm: Target dose in ppm (mg/L or mg/kg)
            water_flow: Water flow rate
            chemical_config: Configuration for the chemical
            chemical_type: Type of chemical (uses default config if provided)
            water_flow_unit: Unit of water flow
            pump_constraints: Optional pump constraints to enforce
            input_event_ids: List of input event IDs for provenance

        Returns:
            DosingResult with calculated dose and provenance
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        calculation_steps = []
        constraint_messages = []

        # Get chemical configuration
        if chemical_config is None and chemical_type is not None:
            chemical_config = self.default_chemicals.get(chemical_type)
        if chemical_config is None:
            raise ValueError("Either chemical_config or valid chemical_type must be provided")

        # Convert inputs to Decimal
        target = Decimal(str(target_dose_ppm))

        if isinstance(water_flow, UnitValue):
            flow_val = water_flow.value
            flow_unit = water_flow.unit
        else:
            flow_val = Decimal(str(water_flow))
            flow_unit = water_flow_unit

        calculation_steps.append({
            "step": 1,
            "description": "Input parameters",
            "target_dose_ppm": str(target),
            "water_flow": str(flow_val),
            "water_flow_unit": flow_unit,
            "chemical_name": chemical_config.chemical_name,
            "active_concentration_percent": str(chemical_config.active_concentration),
            "density_kg_L": str(chemical_config.density),
        })

        # Validate dose against limits
        if chemical_config.min_dose_ppm and target < chemical_config.min_dose_ppm:
            constraint_messages.append(
                f"Target dose {target} ppm is below minimum {chemical_config.min_dose_ppm} ppm"
            )
        if chemical_config.max_dose_ppm and target > chemical_config.max_dose_ppm:
            constraint_messages.append(
                f"Target dose {target} ppm exceeds maximum {chemical_config.max_dose_ppm} ppm"
            )

        # Calculate chemical flow rate
        # Formula: Chemical_Flow (L/h) = Water_Flow (kg/h) * Dose (mg/kg) /
        #          (Active% * Density (kg/L) * 10000)
        active_fraction = chemical_config.active_concentration / Decimal("100")
        denominator = active_fraction * chemical_config.density * Decimal("10000")

        chemical_flow_rate = (flow_val * target) / denominator
        chemical_flow_rate = chemical_flow_rate.quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        calculation_steps.append({
            "step": 2,
            "description": "Calculate chemical flow rate",
            "formula": "Chemical_Flow = Water_Flow * Dose / (Active% * Density * 10000)",
            "active_fraction": str(active_fraction),
            "denominator": str(denominator),
            "chemical_flow_rate_L_h": str(chemical_flow_rate),
        })

        # Calculate pump stroke percentage if constraints provided
        stroke_percent = Decimal("50")  # Default
        is_within_constraints = True

        if pump_constraints:
            # Calculate stroke based on pump capacity
            flow_range = pump_constraints.max_flow_rate - pump_constraints.min_flow_rate
            if flow_range > 0:
                stroke_percent = (
                    (chemical_flow_rate - pump_constraints.min_flow_rate) /
                    flow_range * Decimal("100")
                )
                stroke_percent = stroke_percent.quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )

            # Enforce constraints
            if stroke_percent < pump_constraints.min_stroke_percent:
                constraint_messages.append(
                    f"Calculated stroke {stroke_percent}% below minimum "
                    f"{pump_constraints.min_stroke_percent}%"
                )
                stroke_percent = pump_constraints.min_stroke_percent
                is_within_constraints = False

            if stroke_percent > pump_constraints.max_stroke_percent:
                constraint_messages.append(
                    f"Calculated stroke {stroke_percent}% exceeds maximum "
                    f"{pump_constraints.max_stroke_percent}%"
                )
                stroke_percent = pump_constraints.max_stroke_percent
                is_within_constraints = False

            # Check ramp rate if applicable
            if pump_constraints.ramp_rate_limit:
                stroke_change = abs(stroke_percent - pump_constraints.current_stroke_percent)
                if stroke_change > pump_constraints.ramp_rate_limit:
                    constraint_messages.append(
                        f"Stroke change {stroke_change}% exceeds ramp rate limit "
                        f"{pump_constraints.ramp_rate_limit}%"
                    )

            calculation_steps.append({
                "step": 3,
                "description": "Calculate pump stroke percentage",
                "min_flow_L_h": str(pump_constraints.min_flow_rate),
                "max_flow_L_h": str(pump_constraints.max_flow_rate),
                "calculated_stroke_percent": str(stroke_percent),
                "is_within_constraints": is_within_constraints,
            })

        # Calculate provenance hash
        provenance_data = {
            "operation": "calculate_feedforward_dose",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "chemical_type": chemical_config.chemical_type.value,
            "target_dose_ppm": str(target),
            "water_flow": str(flow_val),
            "chemical_flow_rate": str(chemical_flow_rate),
            "stroke_percent": str(stroke_percent),
            "input_event_ids": input_event_ids or [],
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return DosingResult(
            chemical_type=chemical_config.chemical_type,
            chemical_name=chemical_config.chemical_name,
            calculated_dose_rate=chemical_flow_rate,
            calculated_dose_ppm=target,
            stroke_percent=stroke_percent,
            is_within_constraints=is_within_constraints,
            constraint_messages=constraint_messages,
            calculation_method="feedforward",
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )

    def calculate_feedback_correction(
        self,
        measured_residual: Union[float, Decimal],
        target_residual: Union[float, Decimal],
        current_dose_rate: Union[float, Decimal],
        gain: Union[float, Decimal] = Decimal("0.5"),
        deadband: Union[float, Decimal] = Decimal("1.0"),
        pump_constraints: Optional[PumpConstraints] = None,
        input_event_ids: Optional[list] = None
    ) -> Tuple[Decimal, Dict[str, Any], str]:
        """
        Calculate feedback correction to dosing rate.

        Uses proportional control:
        Correction = Gain * (Target - Measured)
        New_Dose = Current_Dose + Correction

        Args:
            measured_residual: Measured chemical residual (ppm)
            target_residual: Target chemical residual (ppm)
            current_dose_rate: Current dosing rate (L/h)
            gain: Proportional gain factor (0-1)
            deadband: Error deadband - no correction within this range
            pump_constraints: Optional pump constraints
            input_event_ids: List of input event IDs for provenance

        Returns:
            Tuple of (corrected_dose_rate, calculation_details, provenance_hash)
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        measured = Decimal(str(measured_residual))
        target = Decimal(str(target_residual))
        current = Decimal(str(current_dose_rate))
        k_gain = Decimal(str(gain))
        db = Decimal(str(deadband))

        # Calculate error
        error = target - measured

        calculation_details = {
            "measured_residual_ppm": str(measured),
            "target_residual_ppm": str(target),
            "error_ppm": str(error),
            "current_dose_rate_L_h": str(current),
            "gain": str(k_gain),
            "deadband_ppm": str(db),
        }

        # Apply deadband
        if abs(error) <= db:
            corrected = current
            calculation_details["action"] = "Within deadband - no correction"
            calculation_details["correction"] = "0"
        else:
            # Calculate proportional correction
            # Correction as fraction of current dose
            error_fraction = error / target if target != 0 else Decimal("0")
            correction = current * k_gain * error_fraction
            corrected = current + correction

            # Ensure non-negative
            if corrected < 0:
                corrected = Decimal("0")

            corrected = corrected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            calculation_details["action"] = "Proportional correction applied"
            calculation_details["error_fraction"] = str(error_fraction)
            calculation_details["correction_L_h"] = str(correction)

        calculation_details["corrected_dose_rate_L_h"] = str(corrected)

        # Apply pump constraints if provided
        if pump_constraints:
            if corrected < pump_constraints.min_flow_rate:
                calculation_details["constraint_applied"] = "Min flow limit"
                corrected = pump_constraints.min_flow_rate
            elif corrected > pump_constraints.max_flow_rate:
                calculation_details["constraint_applied"] = "Max flow limit"
                corrected = pump_constraints.max_flow_rate

            calculation_details["final_dose_rate_L_h"] = str(corrected)

        # Calculate provenance hash
        provenance_data = {
            "operation": "calculate_feedback_correction",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "calculation_details": calculation_details,
            "input_event_ids": input_event_ids or [],
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return corrected, calculation_details, provenance_hash

    def calculate_oxygen_scavenger_dose(
        self,
        dissolved_oxygen_ppb: Union[float, Decimal],
        water_flow: Union[float, Decimal],
        excess_ratio: Union[float, Decimal] = Decimal("1.5"),
        scavenger_config: Optional[ChemicalConfig] = None,
        input_event_ids: Optional[list] = None
    ) -> DosingResult:
        """
        Calculate oxygen scavenger dose based on dissolved oxygen.

        For sodium sulfite:
        Stoichiometric: 8 ppm Na2SO3 per 1 ppm O2
        With excess: Dose = 8 * O2 * Excess_Ratio

        Args:
            dissolved_oxygen_ppb: Dissolved oxygen in ppb
            water_flow: Water flow rate (kg/h)
            excess_ratio: Excess scavenger ratio (typically 1.5-2.0)
            scavenger_config: Optional custom scavenger configuration
            input_event_ids: List of input event IDs for provenance

        Returns:
            DosingResult with calculated dose
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        calculation_steps = []

        # Get configuration
        if scavenger_config is None:
            scavenger_config = self.default_chemicals[ChemicalType.OXYGEN_SCAVENGER]

        # Convert oxygen from ppb to ppm
        do_ppb = Decimal(str(dissolved_oxygen_ppb))
        do_ppm = do_ppb / Decimal("1000")

        calculation_steps.append({
            "step": 1,
            "description": "Convert dissolved oxygen",
            "dissolved_oxygen_ppb": str(do_ppb),
            "dissolved_oxygen_ppm": str(do_ppm),
        })

        # Calculate stoichiometric dose
        stoich_ratio = scavenger_config.stoichiometric_ratio or Decimal("8")
        excess = Decimal(str(excess_ratio))

        stoich_dose = stoich_ratio * do_ppm
        target_dose = stoich_dose * excess

        calculation_steps.append({
            "step": 2,
            "description": "Calculate stoichiometric and target dose",
            "stoichiometric_ratio": str(stoich_ratio),
            "stoichiometric_dose_ppm": str(stoich_dose),
            "excess_ratio": str(excess),
            "target_dose_ppm": str(target_dose),
        })

        # Now calculate chemical flow using feedforward
        result = self.calculate_feedforward_dose(
            target_dose_ppm=target_dose,
            water_flow=Decimal(str(water_flow)),
            chemical_config=scavenger_config,
            input_event_ids=input_event_ids,
        )

        # Add our steps to the result
        result.calculation_steps = calculation_steps + result.calculation_steps
        result.calculation_method = "oxygen_scavenger_stoichiometric"

        return result

    def reconcile_dosing(
        self,
        commanded_dose_rate: Union[float, Decimal],
        actual_dose_rate: Union[float, Decimal],
        acceptable_deviation_percent: Union[float, Decimal] = Decimal("10"),
        input_event_ids: Optional[list] = None
    ) -> ReconciliationResult:
        """
        Reconcile commanded vs actual pump feedback.

        Compares the commanded dosing rate against actual pump feedback
        to detect dosing discrepancies.

        Args:
            commanded_dose_rate: Commanded dose rate (L/h)
            actual_dose_rate: Actual measured dose rate (L/h)
            acceptable_deviation_percent: Acceptable deviation threshold (%)
            input_event_ids: List of input event IDs for provenance

        Returns:
            ReconciliationResult with deviation analysis
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        commanded = Decimal(str(commanded_dose_rate))
        actual = Decimal(str(actual_dose_rate))
        threshold = Decimal(str(acceptable_deviation_percent))

        # Calculate deviation
        deviation_absolute = abs(commanded - actual)

        if commanded > 0:
            deviation_percent = (deviation_absolute / commanded * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            deviation_percent = Decimal("0") if actual == 0 else Decimal("100")

        is_acceptable = deviation_percent <= threshold

        # Calculate provenance hash
        provenance_data = {
            "operation": "reconcile_dosing",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "commanded_dose_rate": str(commanded),
            "actual_dose_rate": str(actual),
            "deviation_absolute": str(deviation_absolute),
            "deviation_percent": str(deviation_percent),
            "threshold_percent": str(threshold),
            "is_acceptable": is_acceptable,
            "input_event_ids": input_event_ids or [],
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return ReconciliationResult(
            commanded_dose_rate=commanded,
            actual_dose_rate=actual,
            deviation_absolute=deviation_absolute,
            deviation_percent=deviation_percent,
            is_acceptable=is_acceptable,
            deviation_threshold=threshold,
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )
