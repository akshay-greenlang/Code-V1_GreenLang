# -*- coding: utf-8 -*-
"""
CalculatorAgent - Zero-hallucination calculator agent for deterministic computations.

This module implements the CalculatorAgent for GreenLang applications requiring
guaranteed accurate calculations with complete audit trails. All calculations
are deterministic and provenance-tracked.

Example:
    >>> agent = CalculatorAgent(config)
    >>> result = await agent.execute(CalculationInput(
    ...     operation="carbon_emissions",
    ...     inputs={"activity": 1000, "factor": 2.5}
    ... ))
"""

import hashlib
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class CalculationType(str, Enum):
    """Supported calculation types."""

    ARITHMETIC = "arithmetic"
    CARBON_EMISSIONS = "carbon_emissions"
    FINANCIAL = "financial"
    STATISTICAL = "statistical"
    AGGREGATION = "aggregation"
    CONVERSION = "conversion"
    FORMULA = "formula"


class CalculationInput(BaseModel):
    """Input data model for CalculatorAgent."""

    operation: str = Field(..., description="Operation to perform")
    calculation_type: CalculationType = Field(
        CalculationType.ARITHMETIC,
        description="Type of calculation"
    )
    inputs: Dict[str, Union[float, int, Decimal, List[float]]] = Field(
        ...,
        description="Input values for calculation"
    )
    formula: Optional[str] = Field(None, description="Formula for custom calculations")
    precision: int = Field(4, ge=0, le=10, description="Decimal precision")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('inputs')
    def validate_inputs(cls, v):
        """Validate inputs are numeric."""
        if not v:
            raise ValueError("Inputs cannot be empty")
        return v

    @validator('formula')
    def validate_formula(cls, v):
        """Validate formula is safe (no exec/eval of arbitrary code)."""
        if v:
            # Only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/()., abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
            if not all(c in allowed_chars for c in v):
                raise ValueError("Formula contains unsafe characters")
        return v


class CalculationOutput(BaseModel):
    """Output data model for CalculatorAgent."""

    result: Union[float, Dict[str, float], List[float]] = Field(
        ...,
        description="Calculation result"
    )
    operation: str = Field(..., description="Operation performed")
    calculation_type: CalculationType = Field(..., description="Type of calculation")
    formula_used: Optional[str] = Field(None, description="Formula applied")
    unit: Optional[str] = Field(None, description="Result unit")
    precision: int = Field(..., description="Decimal precision used")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed calculation steps"
    )
    processing_time_ms: float = Field(..., description="Processing duration")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Result confidence (1.0 for deterministic)")
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")


class CalculatorAgent(BaseAgent):
    """
    CalculatorAgent implementation for zero-hallucination calculations.

    This agent performs deterministic mathematical calculations with complete
    provenance tracking. It follows GreenLang's zero-hallucination principle
    by using only mathematical operations, never LLM-based computations for
    numeric results.

    Attributes:
        config: Agent configuration
        calculation_registry: Registry of available calculations
        formula_engine: Safe formula evaluation engine

    Example:
        >>> config = AgentConfig(name="carbon_calculator", version="1.0.0")
        >>> agent = CalculatorAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(calculation_input)
        >>> assert result.result.confidence == 1.0  # Always deterministic
    """

    def __init__(self, config: AgentConfig):
        """Initialize CalculatorAgent."""
        super().__init__(config)
        self.calculation_registry: Dict[str, callable] = {}
        self.formula_engine = None
        self.calculation_history: List[CalculationOutput] = []

    async def _initialize_core(self) -> None:
        """Initialize calculator resources."""
        self._logger.info("Initializing CalculatorAgent resources")

        # Register built-in calculations
        self._register_calculations()

        # Initialize formula engine
        self.formula_engine = FormulaEngine()

        self._logger.info(f"Registered {len(self.calculation_registry)} calculations")

    def _register_calculations(self) -> None:
        """Register available calculations."""
        # Arithmetic operations
        self.calculation_registry["add"] = self._add
        self.calculation_registry["subtract"] = self._subtract
        self.calculation_registry["multiply"] = self._multiply
        self.calculation_registry["divide"] = self._divide
        self.calculation_registry["power"] = self._power

        # Carbon calculations
        self.calculation_registry["carbon_emissions"] = self._calculate_carbon_emissions
        self.calculation_registry["scope3_emissions"] = self._calculate_scope3_emissions

        # Statistical operations
        self.calculation_registry["mean"] = self._calculate_mean
        self.calculation_registry["median"] = self._calculate_median
        self.calculation_registry["std_dev"] = self._calculate_std_dev

        # Aggregations
        self.calculation_registry["sum"] = self._calculate_sum
        self.calculation_registry["min"] = self._calculate_min
        self.calculation_registry["max"] = self._calculate_max

        # Financial
        self.calculation_registry["compound_interest"] = self._calculate_compound_interest
        self.calculation_registry["npv"] = self._calculate_npv

    async def _execute_core(self, input_data: CalculationInput, context: ExecutionContext) -> CalculationOutput:
        """
        Core execution logic for calculations.

        This method implements deterministic processing only.
        No LLM calls allowed for numeric calculations.
        """
        start_time = datetime.now(timezone.utc)
        calculation_steps = []
        warnings = []

        try:
            # Step 1: Validate calculation type and operation
            if input_data.operation not in self.calculation_registry and not input_data.formula:
                raise ValueError(f"Unknown operation: {input_data.operation}")

            # Step 2: Log calculation start
            calculation_steps.append({
                "step": "initialization",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": input_data.operation,
                "inputs": input_data.inputs
            })

            # Step 3: Perform calculation
            if input_data.formula:
                # Use formula engine for custom formulas
                result = self.formula_engine.evaluate(
                    input_data.formula,
                    input_data.inputs,
                    precision=input_data.precision
                )
                formula_used = input_data.formula
            else:
                # Use registered calculation
                calc_function = self.calculation_registry[input_data.operation]
                result = calc_function(input_data.inputs, input_data.precision)
                formula_used = None

            # Step 4: Apply precision
            result = self._apply_precision(result, input_data.precision)

            # Step 5: Record calculation completion
            calculation_steps.append({
                "step": "calculation_complete",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result
            })

            # Step 6: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                result,
                context.execution_id
            )

            # Step 7: Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 8: Create output
            output = CalculationOutput(
                result=result,
                operation=input_data.operation,
                calculation_type=input_data.calculation_type,
                formula_used=formula_used,
                unit=input_data.unit,
                precision=input_data.precision,
                provenance_hash=provenance_hash,
                calculation_steps=calculation_steps,
                processing_time_ms=processing_time,
                confidence=1.0,  # Always 1.0 for deterministic calculations
                warnings=warnings
            )

            # Store in history
            self.calculation_history.append(output)
            if len(self.calculation_history) > 1000:
                self.calculation_history.pop(0)

            return output

        except ZeroDivisionError as e:
            warnings.append(f"Division by zero detected: {str(e)}")
            raise ValueError(f"Invalid calculation: division by zero")
        except Exception as e:
            self._logger.error(f"Calculation failed: {str(e)}", exc_info=True)
            raise

    def _apply_precision(self, value: Union[float, Dict, List], precision: int) -> Union[float, Dict, List]:
        """Apply decimal precision to results."""
        if isinstance(value, (int, float)):
            return round(value, precision)
        elif isinstance(value, dict):
            return {k: self._apply_precision(v, precision) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._apply_precision(v, precision) for v in value]
        return value

    def _calculate_provenance_hash(self, inputs: Dict, result: Any, execution_id: str) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": str(inputs),
            "result": str(result)
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    # Calculation implementations (deterministic only)

    def _add(self, inputs: Dict, precision: int) -> float:
        """Addition operation."""
        values = [v for v in inputs.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            raise ValueError("Addition requires at least 2 values")
        return sum(values)

    def _subtract(self, inputs: Dict, precision: int) -> float:
        """Subtraction operation."""
        if "minuend" not in inputs or "subtrahend" not in inputs:
            raise ValueError("Subtraction requires 'minuend' and 'subtrahend'")
        return inputs["minuend"] - inputs["subtrahend"]

    def _multiply(self, inputs: Dict, precision: int) -> float:
        """Multiplication operation."""
        values = [v for v in inputs.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            raise ValueError("Multiplication requires at least 2 values")
        result = 1
        for v in values:
            result *= v
        return result

    def _divide(self, inputs: Dict, precision: int) -> float:
        """Division operation."""
        if "dividend" not in inputs or "divisor" not in inputs:
            raise ValueError("Division requires 'dividend' and 'divisor'")
        if inputs["divisor"] == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return inputs["dividend"] / inputs["divisor"]

    def _power(self, inputs: Dict, precision: int) -> float:
        """Power operation."""
        if "base" not in inputs or "exponent" not in inputs:
            raise ValueError("Power requires 'base' and 'exponent'")
        return math.pow(inputs["base"], inputs["exponent"])

    def _calculate_carbon_emissions(self, inputs: Dict, precision: int) -> float:
        """Calculate carbon emissions (activity * emission_factor)."""
        if "activity" not in inputs or "emission_factor" not in inputs:
            raise ValueError("Carbon calculation requires 'activity' and 'emission_factor'")
        return inputs["activity"] * inputs["emission_factor"]

    def _calculate_scope3_emissions(self, inputs: Dict, precision: int) -> Dict[str, float]:
        """Calculate Scope 3 emissions by category."""
        categories = [
            "purchased_goods", "capital_goods", "fuel_energy", "upstream_transport",
            "waste", "business_travel", "employee_commuting", "upstream_leased",
            "downstream_transport", "processing", "use_of_products", "end_of_life",
            "downstream_leased", "franchises", "investments"
        ]

        results = {}
        total = 0.0

        for category in categories:
            if category in inputs:
                value = inputs[category] * inputs.get(f"{category}_factor", 1.0)
                results[category] = value
                total += value

        results["total_scope3"] = total
        return results

    def _calculate_mean(self, inputs: Dict, precision: int) -> float:
        """Calculate arithmetic mean."""
        if "values" not in inputs or not isinstance(inputs["values"], list):
            raise ValueError("Mean requires 'values' as a list")
        values = inputs["values"]
        if not values:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(values) / len(values)

    def _calculate_median(self, inputs: Dict, precision: int) -> float:
        """Calculate median."""
        if "values" not in inputs or not isinstance(inputs["values"], list):
            raise ValueError("Median requires 'values' as a list")
        values = sorted(inputs["values"])
        n = len(values)
        if n == 0:
            raise ValueError("Cannot calculate median of empty list")
        if n % 2 == 0:
            return (values[n//2-1] + values[n//2]) / 2
        return values[n//2]

    def _calculate_std_dev(self, inputs: Dict, precision: int) -> float:
        """Calculate standard deviation."""
        if "values" not in inputs or not isinstance(inputs["values"], list):
            raise ValueError("Standard deviation requires 'values' as a list")
        values = inputs["values"]
        if len(values) < 2:
            raise ValueError("Need at least 2 values for standard deviation")
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _calculate_sum(self, inputs: Dict, precision: int) -> float:
        """Calculate sum of values."""
        if "values" in inputs and isinstance(inputs["values"], list):
            return sum(inputs["values"])
        return sum(v for v in inputs.values() if isinstance(v, (int, float)))

    def _calculate_min(self, inputs: Dict, precision: int) -> float:
        """Find minimum value."""
        if "values" in inputs and isinstance(inputs["values"], list):
            return min(inputs["values"])
        values = [v for v in inputs.values() if isinstance(v, (int, float))]
        return min(values) if values else 0

    def _calculate_max(self, inputs: Dict, precision: int) -> float:
        """Find maximum value."""
        if "values" in inputs and isinstance(inputs["values"], list):
            return max(inputs["values"])
        values = [v for v in inputs.values() if isinstance(v, (int, float))]
        return max(values) if values else 0

    def _calculate_compound_interest(self, inputs: Dict, precision: int) -> float:
        """Calculate compound interest."""
        required = ["principal", "rate", "time", "compound_frequency"]
        for field in required:
            if field not in inputs:
                raise ValueError(f"Compound interest requires: {required}")

        P = inputs["principal"]
        r = inputs["rate"]
        t = inputs["time"]
        n = inputs["compound_frequency"]

        # A = P(1 + r/n)^(nt)
        return P * math.pow(1 + r/n, n*t)

    def _calculate_npv(self, inputs: Dict, precision: int) -> float:
        """Calculate Net Present Value."""
        if "cash_flows" not in inputs or "discount_rate" not in inputs:
            raise ValueError("NPV requires 'cash_flows' list and 'discount_rate'")

        cash_flows = inputs["cash_flows"]
        rate = inputs["discount_rate"]

        npv = 0
        for i, cf in enumerate(cash_flows):
            npv += cf / math.pow(1 + rate, i)

        return npv

    async def _terminate_core(self) -> None:
        """Cleanup calculator resources."""
        self._logger.info("Cleaning up CalculatorAgent resources")
        self.calculation_registry.clear()
        self.calculation_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect calculator-specific metrics."""
        return {
            "total_calculations": len(self.calculation_history),
            "registered_operations": len(self.calculation_registry),
            "average_precision": sum(c.precision for c in self.calculation_history[-100:]) / min(100, len(self.calculation_history)) if self.calculation_history else 0,
            "calculation_types_used": list(set(c.calculation_type for c in self.calculation_history[-100:]))
        }


class FormulaEngine:
    """
    Safe formula evaluation engine.

    Only allows mathematical operations, no arbitrary code execution.
    """

    def __init__(self):
        """Initialize formula engine."""
        self.allowed_functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan
        }

    def evaluate(self, formula: str, variables: Dict[str, float], precision: int = 4) -> float:
        """
        Safely evaluate a mathematical formula.

        Args:
            formula: Mathematical formula string
            variables: Variable values
            precision: Decimal precision

        Returns:
            Calculated result
        """
        # Replace variables with values
        safe_formula = formula
        for var, value in variables.items():
            safe_formula = safe_formula.replace(var, str(value))

        # Parse and evaluate safely (without eval)
        # This is a simplified implementation - production would use a proper parser
        try:
            # For now, we'll use a restricted eval with only math operations
            # In production, use a proper expression parser like simpleeval
            import ast
            import operator

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def eval_expr(expr):
                return eval_node(ast.parse(expr, mode='eval').body)

            def eval_node(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_node(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {node}")

            result = eval_expr(safe_formula)
            return round(result, precision)

        except Exception as e:
            raise ValueError(f"Formula evaluation failed: {str(e)}")