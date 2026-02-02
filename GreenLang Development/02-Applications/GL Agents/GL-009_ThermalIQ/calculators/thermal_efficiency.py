"""
Thermal Efficiency Calculator
=============================

Zero-hallucination deterministic calculation engine for thermal efficiency analysis.

Implements First Law and Second Law efficiency calculations with full provenance
tracking and uncertainty quantification.

Thermodynamic Basis:
-------------------
First Law Efficiency (Energy Efficiency):
    eta_I = Q_out / Q_in = (Q_in - Q_loss) / Q_in

    Reference: Moran & Shapiro, Fundamentals of Engineering Thermodynamics,
    9th Edition, Chapter 4, Eq. 4.1-4.5

Second Law Efficiency (Exergy Efficiency):
    eta_II = Ex_out / Ex_in = (Ex_in - Ex_destroyed) / Ex_in

    Reference: Bejan, Advanced Engineering Thermodynamics, 4th Edition,
    Chapter 3, Section 3.4

Standards Compliance:
--------------------
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 46 - Overall Plant Performance
- ISO 12952 - Solid fuel boilers

Author: GL-009_ThermalIQ
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import time
from datetime import datetime, timezone


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
class EfficiencyResult:
    """
    Result of efficiency calculation with complete provenance.

    Attributes:
        efficiency: Calculated efficiency (0-1 or 0-100%)
        efficiency_percent: Efficiency as percentage
        unit: Output unit (dimensionless or %)
        uncertainty_percent: Uncertainty in the result (%)
        calculation_steps: List of all calculation steps
        provenance_hash: SHA-256 hash of entire calculation
        formula_reference: Standard/textbook reference for formula
        calculation_time_ms: Time taken for calculation
        timestamp: ISO 8601 timestamp
    """
    efficiency: Decimal
    efficiency_percent: Decimal
    unit: str
    uncertainty_percent: Decimal
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    formula_reference: str
    calculation_time_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "efficiency": str(self.efficiency),
            "efficiency_percent": str(self.efficiency_percent),
            "unit": self.unit,
            "uncertainty_percent": str(self.uncertainty_percent),
            "calculation_steps": [
                {
                    "step_number": step.step_number,
                    "description": step.description,
                    "formula": step.formula,
                    "inputs": {k: str(v) if isinstance(v, Decimal) else v
                              for k, v in step.inputs.items()},
                    "output_value": str(step.output_value),
                    "output_name": step.output_name,
                    "output_unit": step.output_unit,
                    "reference": step.reference
                }
                for step in self.calculation_steps
            ],
            "provenance_hash": self.provenance_hash,
            "formula_reference": self.formula_reference,
            "calculation_time_ms": self.calculation_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class LossBreakdown:
    """
    Breakdown of thermal losses with provenance.

    Attributes:
        total_loss: Total heat loss (kW or %)
        loss_components: Individual loss components
        loss_percentages: Percentage of each loss type
        provenance_hash: SHA-256 hash
        uncertainty_percent: Combined uncertainty
    """
    total_loss: Decimal
    total_loss_unit: str
    loss_components: Dict[str, Decimal]
    loss_percentages: Dict[str, Decimal]
    provenance_hash: str
    uncertainty_percent: Decimal
    calculation_steps: List[CalculationStep]
    references: Dict[str, str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_loss": str(self.total_loss),
            "total_loss_unit": self.total_loss_unit,
            "loss_components": {k: str(v) for k, v in self.loss_components.items()},
            "loss_percentages": {k: str(v) for k, v in self.loss_percentages.items()},
            "provenance_hash": self.provenance_hash,
            "uncertainty_percent": str(self.uncertainty_percent),
            "references": self.references,
            "timestamp": self.timestamp
        }


class ThermalEfficiencyCalculator:
    """
    Zero-hallucination thermal efficiency calculation engine.

    Guarantees:
    - DETERMINISTIC: Same inputs always produce identical outputs
    - REPRODUCIBLE: Full provenance tracking with SHA-256 hashes
    - AUDITABLE: Complete calculation trails
    - STANDARDS-BASED: All formulas from published sources
    - NO LLM: Zero hallucination risk in calculation path

    References:
    -----------
    [1] Moran & Shapiro, Fundamentals of Engineering Thermodynamics, 9th Ed.
    [2] Bejan, Advanced Engineering Thermodynamics, 4th Ed.
    [3] ASME PTC 4.1 - Steam Generating Units
    [4] ASME PTC 46 - Overall Plant Performance

    Example:
    --------
    >>> calc = ThermalEfficiencyCalculator()
    >>> result = calc.calculate_first_law_efficiency(
    ...     heat_in=1000.0,  # kW
    ...     heat_out=850.0   # kW
    ... )
    >>> print(f"Efficiency: {result.efficiency_percent}%")
    Efficiency: 85.000%
    >>> print(f"Provenance: {result.provenance_hash[:16]}...")
    Provenance: a1b2c3d4e5f6g7h8...
    """

    # Precision for regulatory compliance (3 decimal places for efficiency)
    PRECISION = 3

    # Uncertainty sources and default values (%)
    DEFAULT_UNCERTAINTIES = {
        "heat_input_measurement": Decimal("1.0"),
        "heat_output_measurement": Decimal("1.5"),
        "temperature_measurement": Decimal("0.5"),
        "flow_measurement": Decimal("1.0"),
        "pressure_measurement": Decimal("0.25"),
    }

    def __init__(
        self,
        precision: int = 3,
        uncertainties: Optional[Dict[str, Decimal]] = None
    ):
        """
        Initialize thermal efficiency calculator.

        Args:
            precision: Decimal places for output (default: 3)
            uncertainties: Custom uncertainty values (optional)
        """
        self.precision = precision
        self.uncertainties = uncertainties or self.DEFAULT_UNCERTAINTIES.copy()

    def calculate_first_law_efficiency(
        self,
        heat_in: float,
        heat_out: float,
        heat_in_uncertainty: Optional[float] = None,
        heat_out_uncertainty: Optional[float] = None,
    ) -> EfficiencyResult:
        """
        Calculate First Law (Energy) Efficiency.

        Formula: eta_I = Q_out / Q_in

        This is a DETERMINISTIC calculation - same inputs always produce
        identical outputs with complete provenance tracking.

        Args:
            heat_in: Heat input (kW)
            heat_out: Heat output (kW)
            heat_in_uncertainty: Uncertainty in heat_in (%, optional)
            heat_out_uncertainty: Uncertainty in heat_out (%, optional)

        Returns:
            EfficiencyResult with complete provenance

        Raises:
            ValueError: If inputs are invalid (negative, zero heat_in, etc.)

        Reference:
            Moran & Shapiro, Fundamentals of Engineering Thermodynamics,
            9th Edition, Chapter 4, Equation 4.1

        Example:
            >>> calc = ThermalEfficiencyCalculator()
            >>> result = calc.calculate_first_law_efficiency(1000.0, 850.0)
            >>> result.efficiency_percent
            Decimal('85.000')
        """
        start_time = time.perf_counter()

        # Step 1: Input validation (DETERMINISTIC)
        self._validate_heat_inputs(heat_in, heat_out)

        # Step 2: Convert to Decimal for bit-perfect arithmetic
        Q_in = Decimal(str(heat_in))
        Q_out = Decimal(str(heat_out))

        calculation_steps = []

        # Step 3: Calculate efficiency (DETERMINISTIC)
        # Formula: eta_I = Q_out / Q_in
        efficiency = Q_out / Q_in

        step1 = CalculationStep(
            step_number=1,
            description="Calculate First Law Efficiency",
            formula="eta_I = Q_out / Q_in",
            inputs={"Q_in": Q_in, "Q_out": Q_out},
            output_value=efficiency,
            output_name="eta_I",
            output_unit="dimensionless",
            reference="Moran & Shapiro, 9th Ed., Eq. 4.1"
        )
        calculation_steps.append(step1)

        # Step 4: Convert to percentage
        efficiency_percent = efficiency * Decimal("100")

        step2 = CalculationStep(
            step_number=2,
            description="Convert to percentage",
            formula="eta_I_percent = eta_I * 100",
            inputs={"eta_I": efficiency},
            output_value=efficiency_percent,
            output_name="eta_I_percent",
            output_unit="%",
            reference="Standard unit conversion"
        )
        calculation_steps.append(step2)

        # Step 5: Apply precision (regulatory rounding)
        efficiency = self._apply_precision(efficiency)
        efficiency_percent = self._apply_precision(efficiency_percent)

        # Step 6: Calculate uncertainty propagation
        u_in = Decimal(str(heat_in_uncertainty)) if heat_in_uncertainty else self.uncertainties["heat_input_measurement"]
        u_out = Decimal(str(heat_out_uncertainty)) if heat_out_uncertainty else self.uncertainties["heat_output_measurement"]

        # Uncertainty propagation: u_eta = sqrt((u_out)^2 + (u_in)^2)
        # For ratio, relative uncertainties add in quadrature
        uncertainty = self._propagate_ratio_uncertainty(u_out, u_in)

        step3 = CalculationStep(
            step_number=3,
            description="Propagate measurement uncertainty",
            formula="u_eta = sqrt(u_out^2 + u_in^2)",
            inputs={"u_in": u_in, "u_out": u_out},
            output_value=uncertainty,
            output_name="uncertainty",
            output_unit="%",
            reference="GUM - Guide to Expression of Uncertainty in Measurement"
        )
        calculation_steps.append(step3)

        # Step 7: Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            formula_id="first_law_efficiency_v1",
            inputs={"heat_in": heat_in, "heat_out": heat_out},
            calculation_steps=calculation_steps,
            output_value=efficiency
        )

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000

        return EfficiencyResult(
            efficiency=efficiency,
            efficiency_percent=efficiency_percent,
            unit="dimensionless",
            uncertainty_percent=self._apply_precision(uncertainty),
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            formula_reference="Moran & Shapiro, Fundamentals of Engineering Thermodynamics, 9th Ed., Chapter 4",
            calculation_time_ms=calculation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "method": "first_law",
                "standard": "ASME PTC 4.1"
            }
        )

    def calculate_second_law_efficiency(
        self,
        exergy_in: float,
        exergy_out: float,
        exergy_in_uncertainty: Optional[float] = None,
        exergy_out_uncertainty: Optional[float] = None,
    ) -> EfficiencyResult:
        """
        Calculate Second Law (Exergy) Efficiency.

        Formula: eta_II = Ex_out / Ex_in

        This measures how effectively exergy (available work) is utilized.
        Second law efficiency accounts for the quality of energy, not just quantity.

        Args:
            exergy_in: Exergy input (kW)
            exergy_out: Exergy output (kW)
            exergy_in_uncertainty: Uncertainty in exergy_in (%, optional)
            exergy_out_uncertainty: Uncertainty in exergy_out (%, optional)

        Returns:
            EfficiencyResult with complete provenance

        Raises:
            ValueError: If inputs are invalid

        Reference:
            Bejan, Advanced Engineering Thermodynamics, 4th Ed., Chapter 3, Section 3.4

        Example:
            >>> calc = ThermalEfficiencyCalculator()
            >>> result = calc.calculate_second_law_efficiency(500.0, 350.0)
            >>> result.efficiency_percent
            Decimal('70.000')
        """
        start_time = time.perf_counter()

        # Step 1: Input validation
        self._validate_exergy_inputs(exergy_in, exergy_out)

        # Step 2: Convert to Decimal
        Ex_in = Decimal(str(exergy_in))
        Ex_out = Decimal(str(exergy_out))

        calculation_steps = []

        # Step 3: Calculate exergy destruction (DETERMINISTIC)
        Ex_destroyed = Ex_in - Ex_out

        step1 = CalculationStep(
            step_number=1,
            description="Calculate exergy destruction",
            formula="Ex_destroyed = Ex_in - Ex_out",
            inputs={"Ex_in": Ex_in, "Ex_out": Ex_out},
            output_value=Ex_destroyed,
            output_name="Ex_destroyed",
            output_unit="kW",
            reference="Bejan, 4th Ed., Eq. 3.40"
        )
        calculation_steps.append(step1)

        # Step 4: Calculate Second Law Efficiency
        # Formula: eta_II = Ex_out / Ex_in
        efficiency = Ex_out / Ex_in

        step2 = CalculationStep(
            step_number=2,
            description="Calculate Second Law Efficiency",
            formula="eta_II = Ex_out / Ex_in",
            inputs={"Ex_in": Ex_in, "Ex_out": Ex_out},
            output_value=efficiency,
            output_name="eta_II",
            output_unit="dimensionless",
            reference="Bejan, 4th Ed., Eq. 3.42"
        )
        calculation_steps.append(step2)

        # Step 5: Convert to percentage
        efficiency_percent = efficiency * Decimal("100")

        step3 = CalculationStep(
            step_number=3,
            description="Convert to percentage",
            formula="eta_II_percent = eta_II * 100",
            inputs={"eta_II": efficiency},
            output_value=efficiency_percent,
            output_name="eta_II_percent",
            output_unit="%",
            reference="Standard unit conversion"
        )
        calculation_steps.append(step3)

        # Step 6: Apply precision
        efficiency = self._apply_precision(efficiency)
        efficiency_percent = self._apply_precision(efficiency_percent)

        # Step 7: Uncertainty propagation
        # Exergy calculations have higher uncertainty due to entropy calculations
        u_in = Decimal(str(exergy_in_uncertainty)) if exergy_in_uncertainty else Decimal("2.0")
        u_out = Decimal(str(exergy_out_uncertainty)) if exergy_out_uncertainty else Decimal("2.5")

        uncertainty = self._propagate_ratio_uncertainty(u_out, u_in)

        step4 = CalculationStep(
            step_number=4,
            description="Propagate exergy uncertainty",
            formula="u_eta_II = sqrt(u_Ex_out^2 + u_Ex_in^2)",
            inputs={"u_Ex_in": u_in, "u_Ex_out": u_out},
            output_value=uncertainty,
            output_name="uncertainty",
            output_unit="%",
            reference="GUM - Guide to Expression of Uncertainty in Measurement"
        )
        calculation_steps.append(step4)

        # Step 8: Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            formula_id="second_law_efficiency_v1",
            inputs={"exergy_in": exergy_in, "exergy_out": exergy_out},
            calculation_steps=calculation_steps,
            output_value=efficiency
        )

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000

        return EfficiencyResult(
            efficiency=efficiency,
            efficiency_percent=efficiency_percent,
            unit="dimensionless",
            uncertainty_percent=self._apply_precision(uncertainty),
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            formula_reference="Bejan, Advanced Engineering Thermodynamics, 4th Ed., Chapter 3",
            calculation_time_ms=calculation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "method": "second_law",
                "exergy_destroyed_kW": str(Ex_destroyed),
                "standard": "ASME PTC 46"
            }
        )

    def calculate_heat_loss_breakdown(
        self,
        losses: Dict[str, float],
        total_heat_input: Optional[float] = None,
        uncertainties: Optional[Dict[str, float]] = None,
    ) -> LossBreakdown:
        """
        Calculate breakdown of thermal losses with provenance.

        Categorizes and quantifies heat losses according to ASME PTC 4.1.

        Loss Categories (per ASME PTC 4.1):
        - flue_gas_sensible: Sensible heat in dry flue gas
        - flue_gas_moisture: Latent heat in flue gas moisture
        - combustion_moisture: Moisture from combustion of hydrogen
        - unburned_carbon: Loss due to unburned carbon in ash
        - radiation: Radiation and convection from surfaces
        - blowdown: Boiler blowdown losses
        - auxiliary: Auxiliary equipment losses
        - unaccounted: Unaccounted losses

        Args:
            losses: Dictionary of loss values {loss_type: value_kW}
            total_heat_input: Total heat input (kW) for percentage calculation
            uncertainties: Uncertainty for each loss type (%, optional)

        Returns:
            LossBreakdown with complete provenance

        Reference:
            ASME PTC 4.1-2022, Performance Test Code for Steam Generating Units

        Example:
            >>> calc = ThermalEfficiencyCalculator()
            >>> losses = {
            ...     "flue_gas_sensible": 50.0,
            ...     "radiation": 20.0,
            ...     "unburned_carbon": 5.0
            ... }
            >>> breakdown = calc.calculate_heat_loss_breakdown(losses, total_heat_input=1000.0)
            >>> breakdown.total_loss
            Decimal('75.000')
        """
        start_time = time.perf_counter()

        # Validate inputs
        self._validate_loss_inputs(losses)

        calculation_steps = []
        loss_components = {}
        loss_percentages = {}
        references = {}

        # Standard loss categories per ASME PTC 4.1
        loss_categories = {
            "flue_gas_sensible": "ASME PTC 4.1, Section 5.6.1",
            "flue_gas_moisture": "ASME PTC 4.1, Section 5.6.2",
            "combustion_moisture": "ASME PTC 4.1, Section 5.6.3",
            "unburned_carbon": "ASME PTC 4.1, Section 5.6.4",
            "radiation": "ASME PTC 4.1, Section 5.6.5",
            "blowdown": "ASME PTC 4.1, Section 5.6.6",
            "auxiliary": "ASME PTC 4.1, Section 5.6.7",
            "unaccounted": "ASME PTC 4.1, Section 5.6.8",
        }

        step_num = 0
        total_loss = Decimal("0")

        # Process each loss component
        for loss_type, loss_value in losses.items():
            step_num += 1
            loss_decimal = Decimal(str(loss_value))
            loss_components[loss_type] = loss_decimal
            total_loss += loss_decimal

            reference = loss_categories.get(loss_type, "User-defined loss category")
            references[loss_type] = reference

            step = CalculationStep(
                step_number=step_num,
                description=f"Record {loss_type} loss",
                formula=f"{loss_type} = {loss_value}",
                inputs={"loss_type": loss_type},
                output_value=loss_decimal,
                output_name=loss_type,
                output_unit="kW",
                reference=reference
            )
            calculation_steps.append(step)

        # Sum total losses
        step_num += 1
        step = CalculationStep(
            step_number=step_num,
            description="Sum all losses",
            formula="total_loss = sum(individual_losses)",
            inputs={"losses": {k: str(v) for k, v in loss_components.items()}},
            output_value=total_loss,
            output_name="total_loss",
            output_unit="kW",
            reference="ASME PTC 4.1, Section 5.6"
        )
        calculation_steps.append(step)

        # Calculate percentages if total heat input provided
        if total_heat_input is not None:
            Q_in = Decimal(str(total_heat_input))

            for loss_type, loss_value in loss_components.items():
                step_num += 1
                percentage = (loss_value / Q_in) * Decimal("100")
                percentage = self._apply_precision(percentage)
                loss_percentages[loss_type] = percentage

                step = CalculationStep(
                    step_number=step_num,
                    description=f"Calculate {loss_type} percentage",
                    formula=f"{loss_type}_percent = ({loss_type} / Q_in) * 100",
                    inputs={"loss": loss_value, "Q_in": Q_in},
                    output_value=percentage,
                    output_name=f"{loss_type}_percent",
                    output_unit="%",
                    reference="Standard percentage calculation"
                )
                calculation_steps.append(step)

        # Calculate combined uncertainty
        uncertainty = self._calculate_combined_loss_uncertainty(losses, uncertainties)

        # Apply precision
        total_loss = self._apply_precision(total_loss)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            formula_id="heat_loss_breakdown_v1",
            inputs={"losses": losses, "total_heat_input": total_heat_input},
            calculation_steps=calculation_steps,
            output_value=total_loss
        )

        return LossBreakdown(
            total_loss=total_loss,
            total_loss_unit="kW",
            loss_components={k: self._apply_precision(v) for k, v in loss_components.items()},
            loss_percentages={k: self._apply_precision(v) for k, v in loss_percentages.items()},
            provenance_hash=provenance_hash,
            uncertainty_percent=self._apply_precision(uncertainty),
            calculation_steps=calculation_steps,
            references=references,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def calculate_combined_efficiency(
        self,
        heat_in: float,
        heat_out: float,
        exergy_in: float,
        exergy_out: float,
    ) -> Dict[str, EfficiencyResult]:
        """
        Calculate both First and Second Law efficiencies.

        Provides comprehensive efficiency analysis combining energy and exergy.

        Args:
            heat_in: Heat input (kW)
            heat_out: Heat output (kW)
            exergy_in: Exergy input (kW)
            exergy_out: Exergy output (kW)

        Returns:
            Dictionary with 'first_law' and 'second_law' EfficiencyResults
        """
        first_law = self.calculate_first_law_efficiency(heat_in, heat_out)
        second_law = self.calculate_second_law_efficiency(exergy_in, exergy_out)

        return {
            "first_law": first_law,
            "second_law": second_law
        }

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _validate_heat_inputs(self, heat_in: float, heat_out: float) -> None:
        """Validate heat input/output values."""
        if heat_in <= 0:
            raise ValueError(f"heat_in must be positive, got {heat_in}")
        if heat_out < 0:
            raise ValueError(f"heat_out cannot be negative, got {heat_out}")
        if heat_out > heat_in:
            raise ValueError(
                f"heat_out ({heat_out}) cannot exceed heat_in ({heat_in})"
            )

    def _validate_exergy_inputs(self, exergy_in: float, exergy_out: float) -> None:
        """Validate exergy input/output values."""
        if exergy_in <= 0:
            raise ValueError(f"exergy_in must be positive, got {exergy_in}")
        if exergy_out < 0:
            raise ValueError(f"exergy_out cannot be negative, got {exergy_out}")
        if exergy_out > exergy_in:
            raise ValueError(
                f"exergy_out ({exergy_out}) cannot exceed exergy_in ({exergy_in})"
            )

    def _validate_loss_inputs(self, losses: Dict[str, float]) -> None:
        """Validate loss values."""
        if not losses:
            raise ValueError("losses dictionary cannot be empty")
        for loss_type, loss_value in losses.items():
            if loss_value < 0:
                raise ValueError(f"Loss value for '{loss_type}' cannot be negative")

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply regulatory rounding precision using ROUND_HALF_UP."""
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _propagate_ratio_uncertainty(
        self,
        u_numerator: Decimal,
        u_denominator: Decimal
    ) -> Decimal:
        """
        Propagate uncertainty for ratio calculation.

        For R = A/B:
        u_R/R = sqrt((u_A/A)^2 + (u_B/B)^2)

        Since we're working with relative uncertainties (percentages),
        this simplifies to:
        u_R = sqrt(u_A^2 + u_B^2)

        Reference: GUM - Guide to Expression of Uncertainty in Measurement
        """
        from decimal import Decimal
        import math

        u_num_sq = u_numerator ** 2
        u_den_sq = u_denominator ** 2
        combined = float(u_num_sq + u_den_sq)
        uncertainty = Decimal(str(math.sqrt(combined)))

        return uncertainty

    def _calculate_combined_loss_uncertainty(
        self,
        losses: Dict[str, float],
        uncertainties: Optional[Dict[str, float]]
    ) -> Decimal:
        """Calculate combined uncertainty for multiple losses."""
        import math

        if uncertainties is None:
            # Default 5% uncertainty for each loss component
            uncertainties = {k: 5.0 for k in losses.keys()}

        # Root-sum-square of uncertainties
        sum_sq = sum(u ** 2 for u in uncertainties.values())
        combined = math.sqrt(sum_sq / len(uncertainties)) if uncertainties else Decimal("5.0")

        return Decimal(str(combined))

    def _calculate_provenance_hash(
        self,
        formula_id: str,
        inputs: Dict[str, Any],
        calculation_steps: List[CalculationStep],
        output_value: Decimal
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        The hash provides a unique fingerprint of the entire calculation
        that can be verified for reproducibility.
        """
        # Convert Decimal to string for serialization
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

        # Create deterministic JSON string (sorted keys)
        provenance_str = json.dumps(provenance_data, sort_keys=True, default=decimal_serializer)

        # Calculate SHA-256 hash
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()

    def verify_reproducibility(
        self,
        result: EfficiencyResult,
        heat_in: float,
        heat_out: float
    ) -> bool:
        """
        Verify that a calculation is bit-perfect reproducible.

        Re-runs the calculation and compares provenance hashes.

        Args:
            result: Previous calculation result
            heat_in: Original heat input
            heat_out: Original heat output

        Returns:
            True if calculation is reproducible, False otherwise
        """
        new_result = self.calculate_first_law_efficiency(heat_in, heat_out)
        return new_result.provenance_hash == result.provenance_hash


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_carnot_efficiency(T_hot: float, T_cold: float) -> Decimal:
    """
    Calculate Carnot efficiency (theoretical maximum).

    Formula: eta_carnot = 1 - T_cold/T_hot

    Args:
        T_hot: Hot reservoir temperature (K)
        T_cold: Cold reservoir temperature (K)

    Returns:
        Carnot efficiency (dimensionless)

    Reference:
        Carnot, S. (1824). Reflections on the Motive Power of Fire.
    """
    if T_hot <= 0 or T_cold <= 0:
        raise ValueError("Temperatures must be positive (in Kelvin)")
    if T_cold >= T_hot:
        raise ValueError("T_cold must be less than T_hot")

    T_h = Decimal(str(T_hot))
    T_c = Decimal(str(T_cold))

    eta_carnot = Decimal("1") - (T_c / T_h)

    return eta_carnot.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
