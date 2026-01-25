# -*- coding: utf-8 -*-
"""
Calculator Agent Template
Zero-Hallucination Calculations with Provenance

Base agent template for calculations in sustainability applications.
Ensures deterministic, auditable calculations with full provenance tracking.

Version: 1.0.0
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class CalculationStatus(str, Enum):
    """Calculation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class CalculationProvenance:
    """Provenance information for calculation."""
    calculation_id: str
    formula: str
    inputs: Dict[str, Any]
    timestamp: datetime
    agent_version: str
    dependencies: List[str]
    hash: str


@dataclass
class CalculationResult:
    """Result of calculation operation."""
    success: bool
    value: Optional[float] = None
    unit: Optional[str] = None
    uncertainty: Optional[float] = None
    provenance: Optional[CalculationProvenance] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class CalculatorAgent:
    """
    Base Calculator Agent Template.

    Provides zero-hallucination calculation patterns:
    - Deterministic formula execution
    - Full provenance tracking
    - Uncertainty quantification
    - Caching integration
    - Batch processing
    - Parallel processing (thread and process pools)
    - Async execution

    Never generates data - only calculates from provided inputs.
    """

    def __init__(
        self,
        formulas: Optional[Dict[str, Callable]] = None,
        factor_broker: Optional[Any] = None,
        methodologies: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Calculator Agent.

        Args:
            formulas: Dictionary of calculation formulas
            factor_broker: FactorBroker instance for emission factors
            methodologies: Methodologies service for uncertainty
            config: Agent configuration
        """
        self.formulas = formulas or {}
        self.factor_broker = factor_broker
        self.methodologies = methodologies
        self.config = config or {}
        self.version = "1.0.0"

        self._stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "cache_hits": 0,
            "parallel_calculations": 0,
        }

        # Simple in-memory cache (should use Redis in production)
        self._cache = {}

        # Thread and process pools for parallel execution
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get("thread_workers", min(32, (multiprocessing.cpu_count() or 1) + 4))
        )
        self._process_pool = ProcessPoolExecutor(
            max_workers=self.config.get("process_workers", multiprocessing.cpu_count() or 1)
        )

        logger.info("Initialized CalculatorAgent")

    async def calculate(
        self,
        formula_name: str,
        inputs: Dict[str, Any],
        with_uncertainty: bool = False,
        use_cache: bool = True,
    ) -> CalculationResult:
        """
        Execute calculation.

        Args:
            formula_name: Name of formula to execute
            inputs: Input parameters for calculation
            with_uncertainty: Whether to quantify uncertainty
            use_cache: Whether to use cached results

        Returns:
            CalculationResult with value and provenance
        """
        self._stats["total_calculations"] += 1

        try:
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(formula_name, inputs)
                if cache_key in self._cache:
                    self._stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for {formula_name}")
                    return self._cache[cache_key]

            # Validate formula exists
            if formula_name not in self.formulas:
                return CalculationResult(
                    success=False,
                    errors=[f"Unknown formula: {formula_name}"]
                )

            # Validate inputs
            validation_errors = self._validate_inputs(formula_name, inputs)
            if validation_errors:
                return CalculationResult(
                    success=False,
                    errors=validation_errors
                )

            # Execute formula
            formula = self.formulas[formula_name]
            result_value = formula(**inputs)

            # Quantify uncertainty if requested
            uncertainty = None
            if with_uncertainty and self.methodologies:
                uncertainty = await self._quantify_uncertainty(
                    formula_name,
                    inputs,
                    result_value
                )

            # Create provenance
            provenance = self._create_provenance(
                formula_name,
                inputs,
                result_value
            )

            # Create result
            result = CalculationResult(
                success=True,
                value=result_value,
                unit=inputs.get("unit", "unknown"),
                uncertainty=uncertainty,
                provenance=provenance,
                metadata={
                    "formula": formula_name,
                    "with_uncertainty": with_uncertainty,
                }
            )

            # Cache result
            if use_cache:
                self._cache[cache_key] = result

            self._stats["successful_calculations"] += 1
            return result

        except Exception as e:
            logger.error(f"Calculation failed: {e}", exc_info=True)
            self._stats["failed_calculations"] += 1

            return CalculationResult(
                success=False,
                errors=[f"Calculation failed: {str(e)}"]
            )

    async def batch_calculate(
        self,
        formula_name: str,
        inputs_list: List[Dict[str, Any]],
        with_uncertainty: bool = False,
        parallel: bool = False,
        use_processes: bool = False,
    ) -> List[CalculationResult]:
        """
        Execute batch calculations.

        Args:
            formula_name: Name of formula to execute
            inputs_list: List of input parameter sets
            with_uncertainty: Whether to quantify uncertainty
            parallel: Whether to execute in parallel
            use_processes: Use process pool instead of thread pool

        Returns:
            List of CalculationResults
        """
        if parallel:
            return await self.batch_calculate_parallel(
                formula_name=formula_name,
                inputs_list=inputs_list,
                with_uncertainty=with_uncertainty,
                use_processes=use_processes,
            )

        # Sequential execution
        results = []

        for inputs in inputs_list:
            result = await self.calculate(
                formula_name=formula_name,
                inputs=inputs,
                with_uncertainty=with_uncertainty,
            )
            results.append(result)

        return results

    async def batch_calculate_parallel(
        self,
        formula_name: str,
        inputs_list: List[Dict[str, Any]],
        with_uncertainty: bool = False,
        use_processes: bool = False,
    ) -> List[CalculationResult]:
        """
        Execute batch calculations in parallel.

        Args:
            formula_name: Name of formula to execute
            inputs_list: List of input parameter sets
            with_uncertainty: Whether to quantify uncertainty
            use_processes: Use process pool instead of thread pool

        Returns:
            List of CalculationResults
        """
        self._stats["parallel_calculations"] += 1

        if use_processes:
            # Process pool execution (for CPU-intensive calculations)
            loop = asyncio.get_event_loop()
            tasks = []

            for inputs in inputs_list:
                task = loop.run_in_executor(
                    self._process_pool,
                    self._calculate_sync,
                    formula_name,
                    inputs,
                    with_uncertainty,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        else:
            # Thread pool execution (for I/O-bound calculations)
            tasks = []

            for inputs in inputs_list:
                task = self.calculate(
                    formula_name=formula_name,
                    inputs=inputs,
                    with_uncertainty=with_uncertainty,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel calculation {i} failed: {result}")
                final_results.append(
                    CalculationResult(
                        success=False,
                        errors=[f"Parallel calculation failed: {str(result)}"]
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def _calculate_sync(
        self,
        formula_name: str,
        inputs: Dict[str, Any],
        with_uncertainty: bool = False,
    ) -> CalculationResult:
        """
        Synchronous calculation for process pool execution.

        Args:
            formula_name: Name of formula to execute
            inputs: Input parameters
            with_uncertainty: Whether to quantify uncertainty

        Returns:
            CalculationResult
        """
        try:
            # Validate formula exists
            if formula_name not in self.formulas:
                return CalculationResult(
                    success=False,
                    errors=[f"Unknown formula: {formula_name}"]
                )

            # Validate inputs
            validation_errors = self._validate_inputs(formula_name, inputs)
            if validation_errors:
                return CalculationResult(
                    success=False,
                    errors=validation_errors
                )

            # Execute formula
            formula = self.formulas[formula_name]
            result_value = formula(**inputs)

            # Create provenance
            provenance = self._create_provenance(
                formula_name,
                inputs,
                result_value
            )

            # Create result
            result = CalculationResult(
                success=True,
                value=result_value,
                unit=inputs.get("unit", "unknown"),
                uncertainty=None,  # Uncertainty not available in sync mode
                provenance=provenance,
                metadata={
                    "formula": formula_name,
                    "with_uncertainty": with_uncertainty,
                    "execution_mode": "parallel_process",
                }
            )

            return result

        except Exception as e:
            logger.error(f"Sync calculation failed: {e}", exc_info=True)
            return CalculationResult(
                success=False,
                errors=[f"Calculation failed: {str(e)}"]
            )

    def register_formula(
        self,
        name: str,
        formula: Callable,
        required_inputs: Optional[List[str]] = None,
    ):
        """
        Register a new calculation formula.

        Args:
            name: Formula name
            formula: Callable that performs calculation
            required_inputs: List of required input parameter names
        """
        self.formulas[name] = formula

        # Store metadata about formula
        if not hasattr(self, '_formula_metadata'):
            self._formula_metadata = {}

        self._formula_metadata[name] = {
            "required_inputs": required_inputs or [],
            "registered_at": DeterministicClock.utcnow(),
        }

        logger.info(f"Registered formula: {name}")

    def _validate_inputs(
        self,
        formula_name: str,
        inputs: Dict[str, Any]
    ) -> List[str]:
        """Validate calculation inputs."""
        errors = []

        # Check required inputs
        if hasattr(self, '_formula_metadata'):
            metadata = self._formula_metadata.get(formula_name, {})
            required_inputs = metadata.get("required_inputs", [])

            for required in required_inputs:
                if required not in inputs:
                    errors.append(f"Missing required input: {required}")

        # Check for negative values in quantities
        for key, value in inputs.items():
            if "quantity" in key.lower() or "amount" in key.lower():
                if isinstance(value, (int, float)) and value < 0:
                    errors.append(f"Negative value not allowed for {key}: {value}")

        return errors

    def _generate_cache_key(
        self,
        formula_name: str,
        inputs: Dict[str, Any]
    ) -> str:
        """Generate cache key from formula and inputs."""
        # Create deterministic representation
        key_data = {
            "formula": formula_name,
            "inputs": inputs,
        }

        # Sort keys for deterministic hash
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _create_provenance(
        self,
        formula_name: str,
        inputs: Dict[str, Any],
        result_value: float,
    ) -> CalculationProvenance:
        """Create provenance record for calculation."""
        # Generate calculation ID
        calc_id = hashlib.sha256(
            f"{formula_name}{json.dumps(inputs, sort_keys=True)}{result_value}".encode()
        ).hexdigest()[:16]

        # Generate provenance hash
        provenance_data = {
            "formula": formula_name,
            "inputs": inputs,
            "result": result_value,
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "version": self.version,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return CalculationProvenance(
            calculation_id=calc_id,
            formula=formula_name,
            inputs=inputs,
            timestamp=DeterministicClock.utcnow(),
            agent_version=self.version,
            dependencies=[],
            hash=provenance_hash
        )

    async def _quantify_uncertainty(
        self,
        formula_name: str,
        inputs: Dict[str, Any],
        result_value: float,
    ) -> Optional[float]:
        """Quantify calculation uncertainty."""
        if not self.methodologies:
            return None

        try:
            # This would use the Methodologies service
            # For now, return simple estimate
            # In production, would use Monte Carlo simulation
            uncertainty = 0.10  # 10% default uncertainty

            logger.debug(f"Uncertainty for {formula_name}: {uncertainty}")
            return uncertainty

        except Exception as e:
            logger.error(f"Uncertainty quantification failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self._stats.copy()
        stats["cache_size"] = len(self._cache)
        stats["registered_formulas"] = len(self.formulas)
        return stats

    def clear_cache(self):
        """Clear calculation cache."""
        self._cache.clear()
        logger.info("Calculation cache cleared")

    def shutdown(self):
        """Shutdown thread and process pools."""
        logger.info("Shutting down CalculatorAgent executor pools")
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
