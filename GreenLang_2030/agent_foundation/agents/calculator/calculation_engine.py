"""
Calculation Engine - Zero-Hallucination Deterministic Calculations

This module implements the core calculation engine that orchestrates:
- Formula evaluation
- Emission factor lookups
- Multi-step calculations
- Complete provenance tracking
- Bit-perfect reproducibility

All calculations are 100% deterministic with SHA-256 audit trails.
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .formula_engine import FormulaEngine, Formula, FormulaLibrary, FormulaStep
from .emission_factors import EmissionFactorDatabase, EmissionFactor


class CalculationStep(BaseModel):
    """Individual calculation step with complete provenance."""

    step_number: int = Field(..., description="Step sequence number")
    description: str = Field(..., description="Step description")
    operation: str = Field(..., description="Operation type")
    inputs: Dict[str, Any] = Field(..., description="Step input values")
    output_value: Union[Decimal, float, int] = Field(..., description="Step output value")
    output_name: str = Field(..., description="Output variable name")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CalculationResult(BaseModel):
    """Complete calculation result with audit trail."""

    formula_id: str = Field(..., description="Formula identifier")
    formula_version: str = Field(..., description="Formula version")
    output_value: Decimal = Field(..., description="Final calculated value")
    output_unit: str = Field(..., description="Output unit")
    calculation_steps: List[CalculationStep] = Field(..., description="All calculation steps")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_time_ms: float = Field(..., description="Total calculation time")
    input_parameters: Dict[str, Any] = Field(..., description="Input parameters used")
    emission_factors_used: List[Dict[str, Any]] = Field(default_factory=list, description="Emission factors applied")
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")
    uncertainty_percentage: Optional[float] = Field(None, description="Total uncertainty if applicable")


class CalculationEngine:
    """
    Zero-hallucination calculation engine.

    Guarantees:
    - Deterministic: Same input → Same output (bit-perfect)
    - Reproducible: Full provenance tracking
    - Auditable: SHA-256 hash of all calculation steps
    - NO LLM: Zero hallucination risk
    """

    def __init__(
        self,
        formula_library: FormulaLibrary,
        emission_factor_db: EmissionFactorDatabase
    ):
        """
        Initialize calculation engine.

        Args:
            formula_library: Formula library instance
            emission_factor_db: Emission factor database instance
        """
        self.formula_library = formula_library
        self.emission_factor_db = emission_factor_db
        self.formula_engine = FormulaEngine()
        self.calculation_history: List[CalculationResult] = []

    def calculate(
        self,
        formula_id: str,
        parameters: Dict[str, Any],
        formula_version: str = "latest",
        reference_date: Optional[date] = None
    ) -> CalculationResult:
        """
        Execute calculation with zero hallucination guarantee.

        Args:
            formula_id: Formula identifier
            parameters: Input parameters
            formula_version: Formula version (default: latest)
            reference_date: Reference date for emission factors (default: today)

        Returns:
            CalculationResult with complete provenance

        Raises:
            ValueError: If formula not found or parameters invalid
        """
        start_time = datetime.now(timezone.utc)

        # Step 1: Load formula definition
        formula = self.formula_library.get_formula(formula_id, formula_version)
        if not formula:
            raise ValueError(f"Formula not found: {formula_id} version {formula_version}")

        # Step 2: Validate parameters
        validation_errors = self.formula_engine.validate_parameters(formula, parameters)
        if validation_errors:
            raise ValueError(f"Parameter validation failed: {validation_errors}")

        # Step 3: Execute calculation steps
        calculation_steps = []
        context = parameters.copy()
        emission_factors_used = []
        warnings = []
        uncertainties = []

        for step_def_data in formula.calculation['steps']:
            step_def = FormulaStep(**step_def_data)
            step_result = self._execute_step(
                step_def,
                context,
                emission_factors_used,
                warnings,
                uncertainties,
                reference_date
            )
            calculation_steps.append(step_result)
            context[step_result.output_name] = step_result.output_value

        # Step 4: Get final output with precision
        output_def = formula.output
        final_value = context[output_def['value']]

        # Apply precision (regulatory rounding)
        if 'precision' in output_def:
            final_value = self._apply_precision(final_value, output_def['precision'])
        else:
            final_value = Decimal(str(final_value))

        # Step 5: Calculate total uncertainty
        total_uncertainty = None
        if uncertainties:
            # Propagate uncertainties (simple root sum of squares)
            import math
            total_uncertainty = math.sqrt(sum(u**2 for u in uncertainties))

        # Step 6: Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Step 7: Generate provenance hash
        provenance_hash = self._calculate_provenance(
            formula_id,
            formula_version,
            parameters,
            calculation_steps,
            final_value
        )

        # Step 8: Create result
        result = CalculationResult(
            formula_id=formula_id,
            formula_version=formula.version,
            output_value=final_value,
            output_unit=output_def['unit'],
            calculation_steps=calculation_steps,
            provenance_hash=provenance_hash,
            calculation_time_ms=processing_time,
            input_parameters=parameters,
            emission_factors_used=emission_factors_used,
            warnings=warnings,
            uncertainty_percentage=total_uncertainty
        )

        # Store in history
        self.calculation_history.append(result)
        if len(self.calculation_history) > 1000:
            self.calculation_history.pop(0)

        return result

    def _execute_step(
        self,
        step_def: FormulaStep,
        context: Dict[str, Any],
        emission_factors_used: List[Dict[str, Any]],
        warnings: List[str],
        uncertainties: List[float],
        reference_date: Optional[date] = None
    ) -> CalculationStep:
        """
        Execute single calculation step - ZERO HALLUCINATION.

        All operations are deterministic database lookups or arithmetic operations.
        NO LLM calls in calculation path.

        Args:
            step_def: Step definition
            context: Calculation context (variables)
            emission_factors_used: List to track emission factors
            warnings: List to collect warnings
            uncertainties: List to propagate uncertainties
            reference_date: Reference date for factor lookups

        Returns:
            CalculationStep with result
        """
        operation = step_def.operation

        # Extract inputs for this step
        step_inputs = {}
        if step_def.operands:
            for operand in step_def.operands:
                if operand in context:
                    step_inputs[operand] = context[operand]

        # Execute operation (all deterministic)
        if operation == 'lookup':
            result = self._lookup_emission_factor(
                step_def, context, emission_factors_used,
                warnings, uncertainties, reference_date
            )

        elif operation == 'multiply':
            result = self._multiply(step_def.operands, context)

        elif operation == 'divide':
            result = self._divide(step_def.operands, context)

        elif operation == 'add':
            result = self._add(step_def.operands, context)

        elif operation == 'subtract':
            result = self._subtract(step_def.operands, context)

        elif operation == 'expression':
            # Evaluate mathematical expression using safe AST parser
            if not step_def.expression:
                raise ValueError(f"Step {step_def.step} requires 'expression' field")
            result = self.formula_engine.evaluate_expression(
                step_def.expression, context
            )

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return CalculationStep(
            step_number=step_def.step,
            description=step_def.description,
            operation=operation,
            inputs=step_inputs,
            output_value=result,
            output_name=step_def.output,
            metadata={'step_definition': step_def.dict()}
        )

    def _lookup_emission_factor(
        self,
        step_def: FormulaStep,
        context: Dict[str, Any],
        emission_factors_used: List[Dict[str, Any]],
        warnings: List[str],
        uncertainties: List[float],
        reference_date: Optional[date] = None
    ) -> Decimal:
        """
        Lookup emission factor from database - DETERMINISTIC.

        This is a database lookup, NOT an LLM call.
        Same lookup keys → Same emission factor (guaranteed).

        Args:
            step_def: Step definition with lookup parameters
            context: Calculation context
            emission_factors_used: List to track factors
            warnings: List to collect warnings
            uncertainties: List to propagate uncertainties
            reference_date: Reference date for lookup

        Returns:
            Emission factor value as Decimal

        Raises:
            ValueError: If emission factor not found
        """
        if not step_def.lookup_keys:
            raise ValueError(f"Step {step_def.step} requires 'lookup_keys' for lookup operation")

        # Resolve lookup keys from context
        lookup_params = {}
        for key, value_template in step_def.lookup_keys.items():
            if isinstance(value_template, str) and value_template.startswith('{') and value_template.endswith('}'):
                variable = value_template[1:-1]
                if variable not in context:
                    raise ValueError(f"Variable '{variable}' not found in context")
                lookup_params[key] = context[variable]
            else:
                lookup_params[key] = value_template

        # Database lookup (deterministic)
        factor = self.emission_factor_db.get_factor(
            category=lookup_params.get('category', 'scope1'),
            activity_type=lookup_params['activity_type'],
            material_or_fuel=lookup_params['material_or_fuel'],
            region=lookup_params.get('region', 'GLOBAL'),
            reference_date=reference_date,
            unit=lookup_params.get('unit')
        )

        if factor is None:
            raise ValueError(
                f"Emission factor not found for: {lookup_params}"
            )

        # Track emission factor used
        emission_factors_used.append({
            'factor_id': factor.factor_id,
            'material_or_fuel': factor.material_or_fuel,
            'factor_co2e': float(factor.factor_co2e),
            'unit': factor.unit,
            'region': factor.region,
            'source': factor.source,
            'source_year': factor.source_year,
            'data_quality': factor.data_quality
        })

        # Track uncertainty if available
        if factor.uncertainty_percentage:
            uncertainties.append(factor.uncertainty_percentage)

        # Add data quality warning if low
        if factor.data_quality == 'low':
            warnings.append(
                f"Low data quality for emission factor: {factor.material_or_fuel} ({factor.region})"
            )

        return factor.factor_co2e

    def _multiply(self, operands: List[str], context: Dict[str, Any]) -> Decimal:
        """Multiply operands - DETERMINISTIC."""
        if not operands or len(operands) < 2:
            raise ValueError("Multiply requires at least 2 operands")

        result = Decimal('1')
        for operand in operands:
            if operand not in context:
                raise ValueError(f"Operand '{operand}' not found in context")
            value = context[operand]
            result *= Decimal(str(value))

        return result

    def _divide(self, operands: List[str], context: Dict[str, Any]) -> Decimal:
        """Divide operands - DETERMINISTIC."""
        if not operands or len(operands) != 2:
            raise ValueError("Divide requires exactly 2 operands")

        numerator_name = operands[0]
        denominator_name = operands[1]

        if numerator_name not in context:
            raise ValueError(f"Numerator '{numerator_name}' not found in context")
        if denominator_name not in context:
            raise ValueError(f"Denominator '{denominator_name}' not found in context")

        numerator = Decimal(str(context[numerator_name]))
        denominator = Decimal(str(context[denominator_name]))

        if denominator == 0:
            raise ValueError("Division by zero")

        return numerator / denominator

    def _add(self, operands: List[str], context: Dict[str, Any]) -> Decimal:
        """Add operands - DETERMINISTIC."""
        if not operands or len(operands) < 2:
            raise ValueError("Add requires at least 2 operands")

        result = Decimal('0')
        for operand in operands:
            if operand not in context:
                raise ValueError(f"Operand '{operand}' not found in context")
            value = context[operand]
            result += Decimal(str(value))

        return result

    def _subtract(self, operands: List[str], context: Dict[str, Any]) -> Decimal:
        """Subtract operands - DETERMINISTIC."""
        if not operands or len(operands) != 2:
            raise ValueError("Subtract requires exactly 2 operands")

        minuend_name = operands[0]
        subtrahend_name = operands[1]

        if minuend_name not in context:
            raise ValueError(f"Minuend '{minuend_name}' not found in context")
        if subtrahend_name not in context:
            raise ValueError(f"Subtrahend '{subtrahend_name}' not found in context")

        minuend = Decimal(str(context[minuend_name]))
        subtrahend = Decimal(str(context[subtrahend_name]))

        return minuend - subtrahend

    def _apply_precision(self, value: Union[Decimal, float, int], precision: int) -> Decimal:
        """
        Apply regulatory rounding precision.

        Uses ROUND_HALF_UP (banker's rounding) for consistency.

        Args:
            value: Value to round
            precision: Number of decimal places

        Returns:
            Rounded Decimal value
        """
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * precision
        return decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)

    def _calculate_provenance(
        self,
        formula_id: str,
        formula_version: str,
        parameters: Dict[str, Any],
        steps: List[CalculationStep],
        final_value: Decimal
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        Args:
            formula_id: Formula identifier
            formula_version: Formula version
            parameters: Input parameters
            steps: Calculation steps
            final_value: Final calculated value

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            'formula_id': formula_id,
            'formula_version': formula_version,
            'parameters': {k: str(v) for k, v in parameters.items()},
            'steps': [
                {
                    'step': s.step_number,
                    'operation': s.operation,
                    'output_name': s.output_name,
                    'output_value': str(s.output_value)
                }
                for s in steps
            ],
            'final_value': str(final_value)
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def verify_calculation(self, result: CalculationResult) -> bool:
        """
        Verify calculation reproducibility.

        Re-runs the calculation with the same inputs and verifies
        the output is identical (bit-perfect).

        Args:
            result: Calculation result to verify

        Returns:
            True if reproducible, False otherwise
        """
        try:
            # Re-run calculation
            verification_result = self.calculate(
                formula_id=result.formula_id,
                parameters=result.input_parameters,
                formula_version=result.formula_version
            )

            # Compare results (must be identical)
            return (
                verification_result.output_value == result.output_value and
                verification_result.provenance_hash == result.provenance_hash
            )

        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    from datetime import date

    # Initialize components
    formula_library = FormulaLibrary()
    emission_factor_db = EmissionFactorDatabase()
    engine = CalculationEngine(formula_library, emission_factor_db)

    # Load formulas
    formula_count = formula_library.load_formulas()
    print(f"Loaded {formula_count} formulas")

    # Example calculation (if formulas are available)
    if formula_count > 0:
        try:
            result = engine.calculate(
                formula_id="scope1_stationary_combustion",
                parameters={
                    "fuel_quantity": 1000,
                    "fuel_type": "diesel",
                    "region": "GB"
                }
            )

            print(f"\nCalculation Result:")
            print(f"  Value: {result.output_value} {result.output_unit}")
            print(f"  Provenance Hash: {result.provenance_hash}")
            print(f"  Processing Time: {result.calculation_time_ms:.2f}ms")
            print(f"  Steps: {len(result.calculation_steps)}")

            # Verify reproducibility
            is_reproducible = engine.verify_calculation(result)
            print(f"  Reproducible: {is_reproducible}")

        except Exception as e:
            print(f"Calculation example failed: {e}")
