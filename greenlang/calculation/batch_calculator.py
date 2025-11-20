"""
Batch Calculator

High-performance batch processing for 1000+ calculations.

Features:
- Parallel processing
- Progress tracking
- Error isolation (one failure doesn't stop batch)
- Performance metrics
- Result aggregation

Target: <5 seconds for 1000 calculations
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable, Dict, Any
from greenlang.calculation.core_calculator import (
    EmissionCalculator,
    CalculationRequest,
    CalculationResult,
    CalculationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """
    Result of batch calculation.

    Attributes:
        calculations: List of individual calculation results
        total_emissions_kg_co2e: Sum of all emissions
        successful_count: Number of successful calculations
        failed_count: Number of failed calculations
        warning_count: Number of calculations with warnings
        batch_duration_seconds: Total batch processing time
        average_duration_ms: Average time per calculation
    """
    calculations: List[CalculationResult]
    total_emissions_kg_co2e: float = 0
    successful_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    batch_duration_seconds: float = 0
    average_duration_ms: float = 0
    batch_start_time: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Calculate summary statistics"""
        self.total_emissions_kg_co2e = sum(
            float(calc.emissions_kg_co2e) for calc in self.calculations
        )

        self.successful_count = len([
            c for c in self.calculations if c.status == CalculationStatus.SUCCESS
        ])

        self.failed_count = len([
            c for c in self.calculations if c.status == CalculationStatus.FAILED
        ])

        self.warning_count = len([
            c for c in self.calculations if c.status == CalculationStatus.WARNING
        ])

        durations = [
            calc.calculation_duration_ms
            for calc in self.calculations
            if calc.calculation_duration_ms is not None
        ]

        if durations:
            self.average_duration_ms = sum(durations) / len(durations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_calculations': len(self.calculations),
            'total_emissions_kg_co2e': self.total_emissions_kg_co2e,
            'total_emissions_tonnes_co2e': self.total_emissions_kg_co2e / 1000,
            'successful_count': self.successful_count,
            'failed_count': self.failed_count,
            'warning_count': self.warning_count,
            'batch_duration_seconds': self.batch_duration_seconds,
            'average_duration_ms': self.average_duration_ms,
            'calculations_per_second': len(self.calculations) / self.batch_duration_seconds if self.batch_duration_seconds > 0 else 0,
            'batch_start_time': self.batch_start_time.isoformat(),
        }

    def get_failed_calculations(self) -> List[CalculationResult]:
        """Get all failed calculations"""
        return [c for c in self.calculations if c.status == CalculationStatus.FAILED]

    def get_warnings(self) -> List[str]:
        """Get all warnings from calculations"""
        warnings = []
        for calc in self.calculations:
            warnings.extend(calc.warnings)
        return warnings

    def get_errors(self) -> List[str]:
        """Get all errors from calculations"""
        errors = []
        for calc in self.calculations:
            errors.extend(calc.errors)
        return errors


class BatchCalculator:
    """
    High-performance batch calculator.

    Efficiently processes large batches of calculations with:
    - Parallel execution (thread pool)
    - Progress tracking
    - Error isolation
    - Performance monitoring

    Performance Target: <5 seconds for 1000 calculations
    """

    def __init__(
        self,
        emission_calculator: Optional[EmissionCalculator] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize batch calculator.

        Args:
            emission_calculator: Core calculator (auto-creates if None)
            max_workers: Max parallel workers (defaults to CPU count)
        """
        self.calculator = emission_calculator or EmissionCalculator()
        self.max_workers = max_workers

    def calculate_batch(
        self,
        requests: List[CalculationRequest],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        continue_on_error: bool = True,
    ) -> BatchResult:
        """
        Calculate emissions for batch of requests.

        Args:
            requests: List of CalculationRequests
            progress_callback: Optional callback function(completed, total)
            continue_on_error: Continue processing if individual calculation fails

        Returns:
            BatchResult with all calculations and summary

        Example:
            >>> from greenlang.calculation import BatchCalculator, CalculationRequest
            >>> requests = [
            ...     CalculationRequest(factor_id='diesel', activity_amount=100, activity_unit='gallons'),
            ...     CalculationRequest(factor_id='natural_gas', activity_amount=500, activity_unit='therms'),
            ...     CalculationRequest(factor_id='electricity', activity_amount=10000, activity_unit='kwh', region='US_NATIONAL'),
            ... ]
            >>> batch_calc = BatchCalculator()
            >>> result = batch_calc.calculate_batch(requests)
            >>> print(f"Total emissions: {result.total_emissions_kg_co2e:,.3f} kg CO2e")
            >>> print(f"Processed {len(requests)} calculations in {result.batch_duration_seconds:.2f} seconds")
        """
        start_time = datetime.utcnow()
        results = []
        completed = 0

        logger.info(f"Starting batch calculation: {len(requests)} requests")

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all calculations
            future_to_request = {
                executor.submit(self._safe_calculate, req, continue_on_error): req
                for req in requests
            }

            # Collect results as they complete
            for future in as_completed(future_to_request):
                result = future.result()
                results.append(result)

                completed += 1

                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(requests))

        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()

        logger.info(
            f"Batch calculation completed: {len(results)} calculations in {duration_seconds:.2f}s "
            f"({len(results)/duration_seconds:.1f} calc/sec)"
        )

        # Create batch result
        batch_result = BatchResult(
            calculations=results,
            batch_duration_seconds=duration_seconds,
            batch_start_time=start_time,
        )

        return batch_result

    def _safe_calculate(
        self,
        request: CalculationRequest,
        continue_on_error: bool
    ) -> CalculationResult:
        """
        Safely execute calculation with error handling.

        Args:
            request: CalculationRequest
            continue_on_error: Whether to return error result or raise

        Returns:
            CalculationResult (may have failed status)
        """
        try:
            return self.calculator.calculate(request)
        except Exception as e:
            logger.error(f"Calculation failed for {request.factor_id}: {str(e)}")

            if not continue_on_error:
                raise

            # Return failed result
            from decimal import Decimal
            return CalculationResult(
                request=request,
                emissions_kg_co2e=Decimal('0'),
                status=CalculationStatus.FAILED,
                errors=[f"Calculation failed: {str(e)}"],
                calculation_steps=[],
            )

    def calculate_batch_by_groups(
        self,
        requests: List[CalculationRequest],
        group_by: str = 'factor_id',
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, BatchResult]:
        """
        Calculate batch grouped by attribute.

        Useful for analyzing emissions by category, region, etc.

        Args:
            requests: List of CalculationRequests
            group_by: Attribute to group by ('factor_id', 'region', 'tenant_id')
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping group key to BatchResult

        Example:
            >>> results_by_fuel = batch_calc.calculate_batch_by_groups(
            ...     requests,
            ...     group_by='factor_id'
            ... )
            >>> for fuel_type, result in results_by_fuel.items():
            ...     print(f"{fuel_type}: {result.total_emissions_kg_co2e:,.0f} kg CO2e")
        """
        # Group requests
        groups: Dict[str, List[CalculationRequest]] = {}

        for req in requests:
            key = getattr(req, group_by, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(req)

        logger.info(f"Grouped {len(requests)} requests into {len(groups)} groups by {group_by}")

        # Calculate each group
        results = {}
        total_processed = 0

        for group_key, group_requests in groups.items():
            group_result = self.calculate_batch(
                requests=group_requests,
                progress_callback=None,  # Handle progress at top level
            )

            results[group_key] = group_result
            total_processed += len(group_requests)

            if progress_callback:
                progress_callback(total_processed, len(requests))

        return results

    def estimate_batch_duration(self, batch_size: int) -> float:
        """
        Estimate batch processing duration.

        Args:
            batch_size: Number of calculations

        Returns:
            Estimated duration in seconds

        Note: Based on target of 200 calculations/second
        """
        target_calc_per_sec = 200  # Conservative estimate
        return batch_size / target_calc_per_sec
