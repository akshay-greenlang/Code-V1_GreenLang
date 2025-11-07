"""
Scope3CalculatorAgent - Main Agent
GL-VCCI Scope 3 Platform

Production-ready Scope 3 emissions calculator for Categories 1, 4, and 6.

Features:
- 3-tier calculation waterfall (Category 1)
- ISO 14083 conformance (Category 4)
- Business travel calculations (Category 6)
- Monte Carlo uncertainty propagation
- Complete provenance chains
- Batch processing
- Performance optimization

Version: 1.0.0
Date: 2025-10-30
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import time

from .models import (
    Category1Input,
    Category4Input,
    Category6Input,
    CalculationResult,
    BatchResult,
)
from .config import get_config, CalculatorConfig
from .categories import Category1Calculator, Category4Calculator, Category6Calculator
from .calculations import UncertaintyEngine
from .provenance import ProvenanceChainBuilder
from .exceptions import CalculatorError, BatchProcessingError

logger = logging.getLogger(__name__)


class Scope3CalculatorAgent:
    """
    Main Scope 3 emissions calculator agent.

    Supports:
    - Category 1: Purchased Goods & Services (3-tier waterfall)
    - Category 4: Upstream Transportation & Distribution (ISO 14083)
    - Category 6: Business Travel (flights, hotels, ground transport)

    Features:
    - Automatic tier selection
    - Uncertainty propagation
    - Provenance tracking
    - Batch processing
    - Performance monitoring
    """

    def __init__(
        self,
        factor_broker: Any,
        industry_mapper: Optional[Any] = None,
        config: Optional[CalculatorConfig] = None
    ):
        """
        Initialize Scope3CalculatorAgent.

        Args:
            factor_broker: FactorBroker instance for emission factors
            industry_mapper: IndustryMapper instance for product categorization
            config: Calculator configuration
        """
        self.config = config or get_config()
        self.factor_broker = factor_broker

        # Initialize supporting services
        self.uncertainty_engine = UncertaintyEngine() if self.config.enable_monte_carlo else None
        self.provenance_builder = ProvenanceChainBuilder() if self.config.enable_provenance else None

        # Initialize category calculators
        self.category_1 = Category1Calculator(
            factor_broker=factor_broker,
            industry_mapper=industry_mapper,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_4 = Category4Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_6 = Category6Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        # Performance statistics
        self.stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "category_breakdown": {1: 0, 4: 0, 6: 0},
            "total_processing_time_ms": 0.0,
        }

        logger.info(
            "Initialized Scope3CalculatorAgent with categories: [1, 4, 6], "
            f"monte_carlo={self.config.enable_monte_carlo}, "
            f"provenance={self.config.enable_provenance}"
        )

    async def calculate_category_1(
        self, data: Union[Category1Input, Dict[str, Any]]
    ) -> CalculationResult:
        """
        Calculate Category 1 emissions (Purchased Goods & Services).

        Uses 3-tier waterfall:
        1. Supplier-specific PCF (Tier 1)
        2. Product emission factors (Tier 2)
        3. Spend-based (Tier 3)

        Args:
            data: Category 1 input data or dictionary

        Returns:
            CalculationResult

        Raises:
            CalculatorError: If calculation fails
        """
        start_time = time.time()

        try:
            # Convert dict to model if needed
            if isinstance(data, dict):
                data = Category1Input(**data)

            # Calculate
            result = await self.category_1.calculate(data)

            # Update stats
            self._update_stats(
                category=1,
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000
            )

            logger.info(
                f"Category 1 calculation completed: "
                f"{result.emissions_kgco2e:.2f} kgCO2e, tier={result.tier}"
            )

            return result

        except Exception as e:
            self._update_stats(category=1, success=False)
            logger.error(f"Category 1 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_4(
        self, data: Union[Category4Input, Dict[str, Any]]
    ) -> CalculationResult:
        """
        Calculate Category 4 emissions (Upstream Transportation & Distribution).

        ISO 14083 compliant:
        emissions = distance × weight × emission_factor

        Args:
            data: Category 4 input data or dictionary

        Returns:
            CalculationResult

        Raises:
            CalculatorError: If calculation fails
            ISO14083ComplianceError: If compliance check fails
        """
        start_time = time.time()

        try:
            # Convert dict to model if needed
            if isinstance(data, dict):
                data = Category4Input(**data)

            # Calculate
            result = await self.category_4.calculate(data)

            # Update stats
            self._update_stats(
                category=4,
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000
            )

            logger.info(
                f"Category 4 calculation completed: "
                f"{result.emissions_kgco2e:.2f} kgCO2e, "
                f"mode={data.transport_mode.value}"
            )

            return result

        except Exception as e:
            self._update_stats(category=4, success=False)
            logger.error(f"Category 4 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_6(
        self, data: Union[Category6Input, Dict[str, Any]]
    ) -> CalculationResult:
        """
        Calculate Category 6 emissions (Business Travel).

        Components:
        - Flights (with radiative forcing)
        - Hotels
        - Ground transport

        Args:
            data: Category 6 input data or dictionary

        Returns:
            CalculationResult

        Raises:
            CalculatorError: If calculation fails
        """
        start_time = time.time()

        try:
            # Convert dict to model if needed
            if isinstance(data, dict):
                data = Category6Input(**data)

            # Calculate
            result = await self.category_6.calculate(data)

            # Update stats
            self._update_stats(
                category=6,
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000
            )

            logger.info(
                f"Category 6 calculation completed: "
                f"{result.emissions_kgco2e:.2f} kgCO2e, "
                f"{len(data.flights)} flights, {len(data.hotels)} hotels"
            )

            return result

        except Exception as e:
            self._update_stats(category=6, success=False)
            logger.error(f"Category 6 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_batch(
        self,
        records: List[Union[Dict[str, Any], Category1Input, Category4Input, Category6Input]],
        category: int
    ) -> BatchResult:
        """
        Calculate emissions for batch of records.

        Supports parallel processing if enabled in config.

        Args:
            records: List of input records
            category: Scope 3 category (1, 4, or 6)

        Returns:
            BatchResult with aggregated results

        Raises:
            BatchProcessingError: If batch processing encounters errors
        """
        start_time = time.time()

        logger.info(
            f"Starting batch calculation: {len(records)} records, category {category}"
        )

        # Select calculation method
        if category == 1:
            calc_func = self.calculate_category_1
        elif category == 4:
            calc_func = self.calculate_category_4
        elif category == 6:
            calc_func = self.calculate_category_6
        else:
            raise ValueError(f"Unsupported category: {category}")

        # Process records
        results = []
        errors = []
        total_emissions = 0.0

        if self.config.enable_parallel_processing and len(records) > self.config.batch_size:
            # Parallel processing for large batches
            tasks = [calc_func(record) for record in records]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    errors.append({
                        "record_index": i,
                        "error": str(result),
                        "record": records[i] if isinstance(records[i], dict) else records[i].dict()
                    })
                else:
                    results.append(result)
                    total_emissions += result.emissions_kgco2e
        else:
            # Sequential processing
            for i, record in enumerate(records):
                try:
                    result = await calc_func(record)
                    results.append(result)
                    total_emissions += result.emissions_kgco2e
                except Exception as e:
                    errors.append({
                        "record_index": i,
                        "error": str(e),
                        "record": record if isinstance(record, dict) else record.dict()
                    })

        # Calculate average DQI
        avg_dqi = (
            sum(r.data_quality.dqi_score for r in results) / len(results)
            if results else 0.0
        )

        processing_time = time.time() - start_time

        batch_result = BatchResult(
            total_records=len(records),
            successful_records=len(results),
            failed_records=len(errors),
            total_emissions_kgco2e=total_emissions,
            total_emissions_tco2e=total_emissions / 1000,
            results=results,
            errors=errors,
            average_dqi_score=avg_dqi,
            processing_time_seconds=processing_time,
            category=category
        )

        logger.info(
            f"Batch calculation completed: {batch_result.successful_records}/"
            f"{batch_result.total_records} successful, "
            f"total emissions: {batch_result.total_emissions_tco2e:.3f} tCO2e, "
            f"time: {processing_time:.2f}s"
        )

        # Raise error if too many failures
        if batch_result.failed_records > batch_result.total_records * 0.5:
            raise BatchProcessingError(
                total_records=batch_result.total_records,
                failed_records=batch_result.failed_records,
                failure_details=errors,
                category=category
            )

        return batch_result

    def _update_stats(
        self, category: int, success: bool, processing_time_ms: float = 0.0
    ):
        """Update performance statistics."""
        self.stats["total_calculations"] += 1

        if success:
            self.stats["successful_calculations"] += 1
        else:
            self.stats["failed_calculations"] += 1

        self.stats["category_breakdown"][category] = \
            self.stats["category_breakdown"].get(category, 0) + 1

        self.stats["total_processing_time_ms"] += processing_time_ms

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        total = self.stats["total_calculations"]

        stats = self.stats.copy()
        stats["success_rate"] = (
            self.stats["successful_calculations"] / total if total > 0 else 0.0
        )
        stats["average_processing_time_ms"] = (
            self.stats["total_processing_time_ms"] / total if total > 0 else 0.0
        )

        # Throughput (calculations per second)
        total_time_seconds = self.stats["total_processing_time_ms"] / 1000
        stats["throughput_per_second"] = (
            total / total_time_seconds if total_time_seconds > 0 else 0.0
        )

        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "category_breakdown": {1: 0, 4: 0, 6: 0},
            "total_processing_time_ms": 0.0,
        }
        logger.info("Performance statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Scope3CalculatorAgent(categories=[1,4,6], "
            f"calculations={self.stats['total_calculations']}, "
            f"success_rate={self.get_performance_stats()['success_rate']:.2%})"
        )


__all__ = ["Scope3CalculatorAgent"]
