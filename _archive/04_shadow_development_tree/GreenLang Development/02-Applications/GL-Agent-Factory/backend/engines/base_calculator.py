"""
Base Calculator Module

This module provides the abstract base class for all GreenLang calculation engines,
implementing zero-hallucination guarantees through:

- Deterministic calculations (no LLM in calculation path)
- Complete provenance tracking with SHA-256 hashes
- Regulatory-compliant rounding rules
- Unit conversion utilities
- Calculation step documentation

CRITICAL: All numeric calculations MUST use this base class to ensure
reproducibility and audit compliance.

Example:
    >>> class MyCalculator(BaseCalculator):
    ...     def calculate(self, inputs):
    ...         # Implementation with provenance tracking
    ...         pass
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP, ROUND_CEILING
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RoundingRule(str, Enum):
    """
    Regulatory rounding rules for emissions calculations.

    Different regulatory frameworks require specific rounding approaches:
    - HALF_UP: Standard rounding (0.5 rounds up) - GHG Protocol default
    - DOWN: Always round towards zero - Conservative for offsets
    - UP: Always round away from zero - Conservative for emissions
    - CEILING: Round up to regulatory threshold
    """
    HALF_UP = "ROUND_HALF_UP"
    DOWN = "ROUND_DOWN"
    UP = "ROUND_UP"
    CEILING = "ROUND_CEILING"

    def get_decimal_mode(self):
        """Get the decimal module rounding mode."""
        modes = {
            "ROUND_HALF_UP": ROUND_HALF_UP,
            "ROUND_DOWN": ROUND_DOWN,
            "ROUND_UP": ROUND_UP,
            "ROUND_CEILING": ROUND_CEILING,
        }
        return modes[self.value]


class CalculationStep(BaseModel):
    """
    Individual calculation step with complete provenance.

    Each step documents:
    - What operation was performed
    - What inputs were used
    - What output was produced
    - The formula applied

    This enables bit-perfect reproducibility and audit compliance.
    """
    step_number: int = Field(..., description="Step sequence number (1-indexed)")
    description: str = Field(..., description="Human-readable step description")
    operation: str = Field(..., description="Operation type (multiply, divide, lookup, etc.)")
    formula: str = Field(..., description="Formula applied (e.g., 'emissions = quantity * ef')")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values with names")
    output_name: str = Field(..., description="Name of the output variable")
    output_value: Union[Decimal, float, int] = Field(..., description="Calculated output value")
    output_unit: Optional[str] = Field(None, description="Unit of the output value")
    source_reference: Optional[str] = Field(None, description="Reference (e.g., 'EPA 2024 Table 2')")
    step_hash: str = Field(..., description="SHA-256 hash of this step")

    def model_post_init(self, __context):
        """Calculate step hash after initialization."""
        if not self.step_hash:
            self.step_hash = self._compute_step_hash()

    def _compute_step_hash(self) -> str:
        """Compute SHA-256 hash of this calculation step."""
        data = {
            "step_number": self.step_number,
            "operation": self.operation,
            "formula": self.formula,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "output_name": self.output_name,
            "output_value": str(self.output_value),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class CalculationResult(BaseModel):
    """
    Complete calculation result with provenance chain.

    This is the standard output format for all GreenLang calculations,
    containing:
    - The final calculated value
    - Unit of measurement
    - Complete calculation steps for audit
    - SHA-256 provenance hash for verification
    - Performance metrics

    The provenance_hash allows verification that the calculation
    has not been tampered with and can be reproduced exactly.
    """
    formula_id: str = Field(..., description="Unique formula identifier")
    formula_version: str = Field(..., description="Formula version for reproducibility")
    value: Decimal = Field(..., description="Final calculated value")
    unit: str = Field(..., description="Result unit (e.g., 'tCO2e', 'kgCO2e')")
    precision: int = Field(3, description="Decimal places in result")

    # Provenance chain
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list,
        description="Complete calculation steps for audit"
    )
    inputs_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of all inputs"
    )
    emission_factors_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Emission factors used with sources"
    )

    # Provenance verification
    provenance_hash: str = Field(..., description="SHA-256 hash of complete calculation")
    input_hash: str = Field(..., description="SHA-256 hash of inputs only")

    # Uncertainty
    uncertainty_percent: Optional[float] = Field(
        None,
        description="Uncertainty as percentage (e.g., 5.0 = +/- 5%)"
    )
    confidence_interval_lower: Optional[Decimal] = Field(None)
    confidence_interval_upper: Optional[Decimal] = Field(None)

    # Metadata
    calculation_time_ms: float = Field(..., description="Calculation time in milliseconds")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    calculator_version: str = Field("1.0.0", description="Calculator version")
    regulatory_standard: Optional[str] = Field(
        None,
        description="Regulatory standard (e.g., 'GHG Protocol', 'ESRS E1')"
    )

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat(),
        }

    def verify_provenance(self) -> bool:
        """
        Verify the provenance hash matches the calculation steps.

        Returns:
            True if provenance is valid, False if tampered
        """
        step_data = [
            {
                "step_number": s.step_number,
                "operation": s.operation,
                "formula": s.formula,
                "inputs": {k: str(v) for k, v in s.inputs.items()},
                "output_value": str(s.output_value),
            }
            for s in self.calculation_steps
        ]
        recalculated = hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()
        return recalculated == self.provenance_hash

    def to_audit_dict(self) -> Dict[str, Any]:
        """Export as audit-ready dictionary."""
        return {
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "result": {
                "value": str(self.value),
                "unit": self.unit,
                "precision": self.precision,
            },
            "provenance": {
                "provenance_hash": self.provenance_hash,
                "input_hash": self.input_hash,
                "step_count": len(self.calculation_steps),
                "steps": [s.dict() for s in self.calculation_steps],
            },
            "uncertainty": {
                "percent": self.uncertainty_percent,
                "lower": str(self.confidence_interval_lower) if self.confidence_interval_lower else None,
                "upper": str(self.confidence_interval_upper) if self.confidence_interval_upper else None,
            },
            "metadata": {
                "calculated_at": self.calculated_at.isoformat(),
                "calculation_time_ms": self.calculation_time_ms,
                "calculator_version": self.calculator_version,
                "regulatory_standard": self.regulatory_standard,
            },
            "emission_factors": self.emission_factors_used,
        }


class ProvenanceMixin:
    """
    Mixin class providing provenance tracking capabilities.

    This mixin adds SHA-256 hashing and audit trail functionality
    to any calculator class. All calculations should use these
    methods to ensure complete provenance.
    """

    def __init__(self):
        self._steps: List[CalculationStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None

    def _start_calculation(self):
        """Start tracking a new calculation."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()

    def _record_step(
        self,
        description: str,
        operation: str,
        formula: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Union[Decimal, float, int],
        output_unit: Optional[str] = None,
        source_reference: Optional[str] = None,
    ) -> CalculationStep:
        """
        Record a calculation step with full provenance.

        Args:
            description: Human-readable step description
            operation: Operation type (multiply, divide, lookup, etc.)
            formula: Formula string (e.g., 'emissions = quantity * ef')
            inputs: Dictionary of input values
            output_name: Name of output variable
            output_value: Calculated output
            output_unit: Optional unit of output
            source_reference: Optional reference (e.g., 'EPA 2024')

        Returns:
            Recorded calculation step
        """
        self._step_counter += 1

        # Compute step hash
        hash_data = {
            "step_number": self._step_counter,
            "operation": operation,
            "formula": formula,
            "inputs": {k: str(v) for k, v in inputs.items()},
            "output_name": output_name,
            "output_value": str(output_value),
        }
        step_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            formula=formula,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            output_unit=output_unit,
            source_reference=source_reference,
            step_hash=step_hash,
        )

        self._steps.append(step)
        return step

    def _compute_provenance_hash(self) -> str:
        """
        Compute SHA-256 hash of the complete calculation chain.

        Returns:
            64-character hex string
        """
        step_data = [
            {
                "step_number": s.step_number,
                "operation": s.operation,
                "formula": s.formula,
                "inputs": {k: str(v) for k, v in s.inputs.items()},
                "output_value": str(s.output_value),
            }
            for s in self._steps
        ]
        return hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()

    def _compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of inputs only.

        Args:
            inputs: Input parameters dictionary

        Returns:
            64-character hex string
        """
        serializable = {k: str(v) for k, v in inputs.items()}
        return hashlib.sha256(
            json.dumps(serializable, sort_keys=True).encode()
        ).hexdigest()

    def _get_calculation_time_ms(self) -> float:
        """Get calculation time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def _build_result(
        self,
        formula_id: str,
        formula_version: str,
        value: Decimal,
        unit: str,
        inputs_summary: Dict[str, Any],
        emission_factors_used: List[Dict[str, Any]],
        precision: int = 3,
        uncertainty_percent: Optional[float] = None,
        regulatory_standard: Optional[str] = None,
    ) -> CalculationResult:
        """
        Build a complete calculation result with provenance.

        Args:
            formula_id: Unique formula identifier
            formula_version: Formula version
            value: Final calculated value
            unit: Result unit
            inputs_summary: Summary of inputs
            emission_factors_used: List of emission factors with sources
            precision: Decimal places
            uncertainty_percent: Optional uncertainty percentage
            regulatory_standard: Optional regulatory reference

        Returns:
            Complete CalculationResult with provenance
        """
        # Calculate confidence interval if uncertainty provided
        confidence_lower = None
        confidence_upper = None
        if uncertainty_percent is not None:
            factor = Decimal(str(uncertainty_percent)) / Decimal("100")
            confidence_lower = value * (Decimal("1") - factor)
            confidence_upper = value * (Decimal("1") + factor)

        return CalculationResult(
            formula_id=formula_id,
            formula_version=formula_version,
            value=value,
            unit=unit,
            precision=precision,
            calculation_steps=self._steps.copy(),
            inputs_summary=inputs_summary,
            emission_factors_used=emission_factors_used,
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            uncertainty_percent=uncertainty_percent,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            calculation_time_ms=self._get_calculation_time_ms(),
            regulatory_standard=regulatory_standard,
        )


class BaseCalculator(ABC, ProvenanceMixin):
    """
    Abstract base class for all GreenLang calculation engines.

    This class enforces zero-hallucination guarantees:
    - All calculations are deterministic (no LLM)
    - Complete provenance tracking with SHA-256
    - Regulatory-compliant rounding
    - Performance tracking (<5ms target)

    Subclasses must implement:
    - calculate(): Main calculation method
    - validate_inputs(): Input validation

    Example:
        >>> class Scope1Calculator(BaseCalculator):
        ...     def calculate(self, inputs):
        ...         self._start_calculation()
        ...         # ... calculation steps ...
        ...         return self._build_result(...)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        emission_factor_db: Optional[Any] = None,
        rounding_rule: RoundingRule = RoundingRule.HALF_UP,
        default_precision: int = 3,
    ):
        """
        Initialize the base calculator.

        Args:
            emission_factor_db: Emission factor database instance
            rounding_rule: Default rounding rule for calculations
            default_precision: Default decimal precision
        """
        ProvenanceMixin.__init__(self)
        self.emission_factor_db = emission_factor_db
        self.rounding_rule = rounding_rule
        self.default_precision = default_precision

    @abstractmethod
    def calculate(self, inputs: Dict[str, Any]) -> CalculationResult:
        """
        Perform the main calculation.

        This method MUST:
        1. Call self._start_calculation() at the beginning
        2. Use self._record_step() for each calculation step
        3. Return self._build_result() with complete provenance

        Args:
            inputs: Dictionary of input parameters

        Returns:
            CalculationResult with provenance

        Raises:
            ValueError: If inputs are invalid
            CalculationError: If calculation fails
        """
        pass

    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters.

        Args:
            inputs: Dictionary of input parameters

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails with details
        """
        pass

    def round_value(
        self,
        value: Union[Decimal, float],
        precision: Optional[int] = None,
        rule: Optional[RoundingRule] = None,
    ) -> Decimal:
        """
        Apply regulatory rounding to a value.

        Args:
            value: Value to round
            precision: Decimal places (default: self.default_precision)
            rule: Rounding rule (default: self.rounding_rule)

        Returns:
            Rounded Decimal value
        """
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        precision = precision if precision is not None else self.default_precision
        rule = rule if rule is not None else self.rounding_rule

        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=rule.get_decimal_mode())

    def multiply(
        self,
        a: Union[Decimal, float],
        b: Union[Decimal, float],
    ) -> Decimal:
        """
        Multiply two values using Decimal for precision.

        Args:
            a: First operand
            b: Second operand

        Returns:
            Product as Decimal
        """
        if not isinstance(a, Decimal):
            a = Decimal(str(a))
        if not isinstance(b, Decimal):
            b = Decimal(str(b))
        return a * b

    def divide(
        self,
        numerator: Union[Decimal, float],
        denominator: Union[Decimal, float],
    ) -> Decimal:
        """
        Divide two values using Decimal for precision.

        Args:
            numerator: Numerator
            denominator: Denominator

        Returns:
            Quotient as Decimal

        Raises:
            ValueError: If denominator is zero
        """
        if not isinstance(numerator, Decimal):
            numerator = Decimal(str(numerator))
        if not isinstance(denominator, Decimal):
            denominator = Decimal(str(denominator))

        if denominator == Decimal("0"):
            raise ValueError("Division by zero")

        return numerator / denominator

    def add(self, *values: Union[Decimal, float]) -> Decimal:
        """
        Add multiple values using Decimal for precision.

        Args:
            *values: Values to add

        Returns:
            Sum as Decimal
        """
        result = Decimal("0")
        for v in values:
            if not isinstance(v, Decimal):
                v = Decimal(str(v))
            result += v
        return result

    def subtract(
        self,
        a: Union[Decimal, float],
        b: Union[Decimal, float],
    ) -> Decimal:
        """
        Subtract two values using Decimal for precision.

        Args:
            a: Value to subtract from
            b: Value to subtract

        Returns:
            Difference as Decimal
        """
        if not isinstance(a, Decimal):
            a = Decimal(str(a))
        if not isinstance(b, Decimal):
            b = Decimal(str(b))
        return a - b

    def convert_to_tonnes(self, kg_value: Union[Decimal, float]) -> Decimal:
        """
        Convert kg to tonnes (t).

        Args:
            kg_value: Value in kilograms

        Returns:
            Value in tonnes
        """
        return self.divide(kg_value, Decimal("1000"))

    def convert_to_kg(self, tonne_value: Union[Decimal, float]) -> Decimal:
        """
        Convert tonnes to kg.

        Args:
            tonne_value: Value in tonnes

        Returns:
            Value in kilograms
        """
        return self.multiply(tonne_value, Decimal("1000"))


class CalculationError(Exception):
    """
    Exception raised when a calculation fails.

    Attributes:
        message: Error message
        formula_id: Formula that failed
        step_number: Step where failure occurred
        inputs: Input parameters that caused failure
    """

    def __init__(
        self,
        message: str,
        formula_id: Optional[str] = None,
        step_number: Optional[int] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.formula_id = formula_id
        self.step_number = step_number
        self.inputs = inputs
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API response."""
        return {
            "error": self.message,
            "formula_id": self.formula_id,
            "step_number": self.step_number,
            "inputs": self.inputs,
        }
