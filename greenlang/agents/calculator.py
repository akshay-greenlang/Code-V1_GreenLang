"""
GreenLang Calculator Agent
Specialized base class for computational and mathematical operations.
"""

from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from pydantic import BaseModel, Field, validator
from abc import abstractmethod
import logging
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from functools import lru_cache

from .base import BaseAgent, AgentConfig, AgentResult, AgentMetrics

logger = logging.getLogger(__name__)


class CalculatorConfig(AgentConfig):
    """Configuration for calculator agents."""
    precision: int = Field(default=6, description="Decimal precision for calculations")
    enable_caching: bool = Field(default=True, description="Enable caching of calculation results")
    cache_size: int = Field(default=128, description="Maximum number of cached results")
    validate_inputs: bool = Field(default=True, description="Validate inputs before calculation")
    deterministic: bool = Field(default=True, description="Ensure deterministic calculations")
    allow_division_by_zero: bool = Field(default=False, description="Allow division by zero (returns None)")
    rounding_mode: str = Field(default="ROUND_HALF_UP", description="Decimal rounding mode")

    @validator('precision')
    def validate_precision(cls, v):
        if v < 0 or v > 28:
            raise ValueError("precision must be between 0 and 28")
        return v

    @validator('cache_size')
    def validate_cache_size(cls, v):
        if v < 0:
            raise ValueError("cache_size must be non-negative")
        return v


class CalculationStep(BaseModel):
    """Record of a single calculation step."""
    step_name: str = Field(..., description="Name of the calculation step")
    formula: str = Field(..., description="Formula or expression")
    inputs: Dict[str, Any] = Field(..., description="Input values")
    result: Any = Field(..., description="Calculated result")
    units: Optional[str] = Field(default=None, description="Units of result")
    timestamp: datetime = Field(default_factory=datetime.now, description="When calculated")


class CalculatorResult(AgentResult):
    """Enhanced result from calculator with calculation trace."""
    result_value: Any = Field(default=None, description="Main calculation result")
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list, description="Step-by-step calculation trace"
    )
    units: Optional[str] = Field(default=None, description="Units of final result")
    precision: int = Field(default=6, description="Precision used")
    cached: bool = Field(default=False, description="Whether result was cached")


class UnitConverter:
    """Simple unit conversion utility."""

    # Conversion factors to base units
    CONVERSIONS = {
        # Energy
        "kWh": {"base": "J", "factor": 3600000},
        "MWh": {"base": "J", "factor": 3600000000},
        "GJ": {"base": "J", "factor": 1000000000},
        "J": {"base": "J", "factor": 1},

        # Mass
        "kg": {"base": "kg", "factor": 1},
        "g": {"base": "kg", "factor": 0.001},
        "t": {"base": "kg", "factor": 1000},
        "ton": {"base": "kg", "factor": 1000},

        # Volume
        "m3": {"base": "m3", "factor": 1},
        "L": {"base": "m3", "factor": 0.001},
        "gal": {"base": "m3", "factor": 0.00378541},

        # Temperature (requires offset)
        "C": {"base": "K", "factor": 1, "offset": 273.15},
        "F": {"base": "K", "factor": 5/9, "offset": 255.372},
        "K": {"base": "K", "factor": 1},
    }

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value from one unit to another.

        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units are incompatible or unknown
        """
        if from_unit == to_unit:
            return value

        if from_unit not in cls.CONVERSIONS or to_unit not in cls.CONVERSIONS:
            raise ValueError(f"Unknown units: {from_unit} or {to_unit}")

        from_conv = cls.CONVERSIONS[from_unit]
        to_conv = cls.CONVERSIONS[to_unit]

        if from_conv["base"] != to_conv["base"]:
            raise ValueError(f"Incompatible units: {from_unit} ({from_conv['base']}) vs {to_unit} ({to_conv['base']})")

        # Convert to base unit
        base_value = value * from_conv["factor"]
        if "offset" in from_conv:
            base_value += from_conv["offset"]

        # Convert from base unit to target
        result = base_value / to_conv["factor"]
        if "offset" in to_conv:
            result -= to_conv["offset"]

        return result


class BaseCalculator(BaseAgent):
    """
    Base class for calculator agents.

    Provides:
    - High-precision decimal arithmetic
    - Deterministic calculations
    - Calculation caching for performance
    - Step-by-step calculation trace
    - Unit conversion support
    - Input validation
    - Error handling for edge cases (division by zero, etc.)

    Example:
        class CarbonCalculator(BaseCalculator):
            def calculate(self, inputs: Dict[str, Any]) -> Any:
                energy_kwh = inputs['energy_kwh']
                emission_factor = inputs['emission_factor']
                return energy_kwh * emission_factor

            def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
                return 'energy_kwh' in inputs and 'emission_factor' in inputs
    """

    def __init__(self, config: Optional[CalculatorConfig] = None):
        """Initialize calculator with configuration."""
        if config is None:
            config = CalculatorConfig(
                name=self.__class__.__name__,
                description=self.__class__.__doc__ or "Calculator agent"
            )
        super().__init__(config)
        self.config: CalculatorConfig = config

        # Cache for calculation results
        self._calc_cache: Dict[str, Any] = {}
        self._calculation_steps: List[CalculationStep] = []

    @abstractmethod
    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """
        Perform the calculation.
        Must be implemented by subclasses.

        Args:
            inputs: Input values for calculation

        Returns:
            Calculation result
        """
        pass

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate calculation inputs.
        Override to add custom validation logic.

        Args:
            inputs: Inputs to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def round_decimal(self, value: Union[float, Decimal], precision: Optional[int] = None) -> Decimal:
        """
        Round a value to specified precision.

        Args:
            value: Value to round
            precision: Decimal places (uses config.precision if None)

        Returns:
            Rounded Decimal value
        """
        if precision is None:
            precision = self.config.precision

        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def safe_divide(self, numerator: float, denominator: float) -> Optional[float]:
        """
        Safely divide two numbers.

        Args:
            numerator: Numerator
            denominator: Denominator

        Returns:
            Division result, or None if division by zero and allowed

        Raises:
            ZeroDivisionError: If division by zero and not allowed
        """
        if denominator == 0:
            if self.config.allow_division_by_zero:
                self.logger.warning("Division by zero, returning None")
                return None
            raise ZeroDivisionError("Division by zero")

        return numerator / denominator

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        return UnitConverter.convert(value, from_unit, to_unit)

    def add_calculation_step(
        self,
        step_name: str,
        formula: str,
        inputs: Dict[str, Any],
        result: Any,
        units: Optional[str] = None
    ):
        """
        Record a calculation step for traceability.

        Args:
            step_name: Name of the step
            formula: Formula or expression
            inputs: Input values
            result: Calculated result
            units: Units of result
        """
        step = CalculationStep(
            step_name=step_name,
            formula=formula,
            inputs=inputs,
            result=result,
            units=units
        )
        self._calculation_steps.append(step)

    def get_cache_key(self, inputs: Dict[str, Any]) -> str:
        """
        Generate cache key for inputs.

        Args:
            inputs: Input dictionary

        Returns:
            Cache key as hex string
        """
        # Sort keys for deterministic hashing
        sorted_inputs = json.dumps(inputs, sort_keys=True)
        return hashlib.sha256(sorted_inputs.encode()).hexdigest()

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached calculation result.

        Args:
            cache_key: Cache key

        Returns:
            Cached result if exists, None otherwise
        """
        if not self.config.enable_caching:
            return None

        return self._calc_cache.get(cache_key)

    def cache_result(self, cache_key: str, result: Any):
        """
        Cache a calculation result.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.config.enable_caching:
            return

        # Implement simple LRU by removing oldest entry if cache is full
        if len(self._calc_cache) >= self.config.cache_size:
            # Remove first (oldest) entry
            first_key = next(iter(self._calc_cache))
            del self._calc_cache[first_key]

        self._calc_cache[cache_key] = result

    def execute(self, input_data: Dict[str, Any]) -> CalculatorResult:
        """
        Execute calculation with caching and tracing.

        Args:
            input_data: Must contain 'inputs' key with calculation parameters

        Returns:
            CalculatorResult with calculation results and trace
        """
        # Extract inputs
        inputs = input_data.get("inputs", {})
        if not inputs:
            return CalculatorResult(
                success=False,
                error="No inputs provided in input_data['inputs']"
            )

        # Validate inputs if enabled
        if self.config.validate_inputs and not self.validate_calculation_inputs(inputs):
            return CalculatorResult(
                success=False,
                error="Calculation input validation failed"
            )

        # Clear previous steps
        self._calculation_steps = []

        # Check cache
        cache_key = self.get_cache_key(inputs)
        cached_result = self.get_cached_result(cache_key)

        if cached_result is not None:
            self.logger.debug(f"Using cached result for {cache_key[:8]}...")
            self.stats.increment("cache_hits")
            return CalculatorResult(
                success=True,
                result_value=cached_result,
                cached=True,
                data={"result": cached_result}
            )

        self.stats.increment("cache_misses")

        # Perform calculation
        try:
            result = self.calculate(inputs)

            # Round if numeric
            if isinstance(result, (int, float, Decimal)):
                result = float(self.round_decimal(result))

            # Cache result
            self.cache_result(cache_key, result)

            return CalculatorResult(
                success=True,
                result_value=result,
                calculation_steps=self._calculation_steps,
                precision=self.config.precision,
                cached=False,
                data={"result": result, "inputs": inputs}
            )

        except Exception as e:
            self.logger.error(f"Calculation failed: {str(e)}", exc_info=True)
            return CalculatorResult(
                success=False,
                error=str(e),
                calculation_steps=self._calculation_steps
            )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input contains calculation inputs."""
        if "inputs" not in input_data:
            self.logger.error("Input data must contain 'inputs' key")
            return False

        inputs = input_data["inputs"]
        if not isinstance(inputs, dict):
            self.logger.error("'inputs' must be a dictionary")
            return False

        return True

    def clear_cache(self):
        """Clear the calculation cache."""
        self._calc_cache.clear()
        self.logger.info("Calculation cache cleared")
