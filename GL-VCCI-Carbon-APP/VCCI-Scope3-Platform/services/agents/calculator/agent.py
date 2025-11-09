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

Version: 2.0.0 - Enhanced with GreenLang SDK
Phase: 5 (Agent Architecture Compliance)
Date: 2025-11-09
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import time

# GreenLang SDK Integration
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.cache import CacheManager, get_cache_manager
from greenlang.telemetry import (
    MetricsCollector,
    get_logger,
    track_execution,
    create_span,
)

from .models import (
    Category1Input,
    Category2Input,
    Category3Input,
    Category4Input,
    Category5Input,
    Category6Input,
    Category7Input,
    Category8Input,
    Category9Input,
    Category10Input,
    Category11Input,
    Category12Input,
    Category13Input,
    Category14Input,
    Category15Input,
    CalculationResult,
    BatchResult,
)
from .config import get_config, CalculatorConfig
from .categories import (
    Category1Calculator,
    Category2Calculator,
    Category3Calculator,
    Category4Calculator,
    Category5Calculator,
    Category6Calculator,
    Category7Calculator,
    Category8Calculator,
    Category9Calculator,
    Category10Calculator,
    Category11Calculator,
    Category12Calculator,
    Category13Calculator,
    Category14Calculator,
    Category15Calculator,
)
from .calculations import UncertaintyEngine
from .provenance import ProvenanceChainBuilder
from .exceptions import CalculatorError, BatchProcessingError

logger = get_logger(__name__)


class Scope3CalculatorAgent(Agent[Dict[str, Any], CalculationResult]):
    """
    Main Scope 3 emissions calculator agent.

    Supports ALL 15 Scope 3 Categories:
    - Category 1: Purchased Goods & Services (3-tier waterfall)
    - Category 2: Capital Goods
    - Category 3: Fuel & Energy-Related Activities
    - Category 4: Upstream Transportation & Distribution (ISO 14083)
    - Category 5: Waste Generated in Operations
    - Category 6: Business Travel (flights, hotels, ground transport)
    - Category 7: Employee Commuting
    - Category 8: Upstream Leased Assets
    - Category 9: Downstream Transportation & Distribution
    - Category 10: Processing of Sold Products
    - Category 11: Use of Sold Products
    - Category 12: End-of-Life Treatment of Sold Products
    - Category 13: Downstream Leased Assets
    - Category 14: Franchises
    - Category 15: Investments (PCAF Standard)

    Features:
    - Automatic tier selection
    - LLM-powered intelligent classification
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
        # Initialize base Agent with metadata
        metadata = Metadata(
            id="scope3_calculator_agent",
            name="Scope3CalculatorAgent",
            version="2.0.0",
            description="Production-ready Scope 3 emissions calculator for all 15 categories",
            tags=["scope3", "emissions", "calculator", "ghg-protocol"],
        )
        super().__init__(metadata)

        self.config = config or get_config()
        self.factor_broker = factor_broker

        # Initialize GreenLang infrastructure
        self.cache_manager = get_cache_manager() if self.config.enable_caching else None
        self.metrics = MetricsCollector(namespace="vcci.calculator")

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

        self.category_2 = Category2Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_3 = Category3Calculator(
            factor_broker=factor_broker,
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

        self.category_5 = Category5Calculator(
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

        self.category_7 = Category7Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_8 = Category8Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_9 = Category9Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_10 = Category10Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_11 = Category11Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_12 = Category12Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_13 = Category13Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_14 = Category14Calculator(
            factor_broker=factor_broker,
            uncertainty_engine=self.uncertainty_engine,
            provenance_builder=self.provenance_builder,
            config=self.config
        )

        self.category_15 = Category15Calculator(
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
            "category_breakdown": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0},
            "total_processing_time_ms": 0.0,
        }

        logger.info(
            "Initialized Scope3CalculatorAgent with ALL 15 categories, "
            f"monte_carlo={self.config.enable_monte_carlo}, "
            f"provenance={self.config.enable_provenance}"
        )

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input data containing category and calculation data

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            logger.error("Input data must be a dictionary")
            return False

        if "category" not in input_data:
            logger.error("Input data must contain 'category' field")
            return False

        category = input_data.get("category")
        if not isinstance(category, int) or category < 1 or category > 15:
            logger.error(f"Invalid category: {category}. Must be 1-15")
            return False

        if "data" not in input_data:
            logger.error("Input data must contain 'data' field")
            return False

        return True

    @track_execution(metric_name="calculator_process")
    async def process(self, input_data: Dict[str, Any]) -> CalculationResult:
        """
        Process calculation request.

        Args:
            input_data: Dictionary with 'category' and 'data' fields

        Returns:
            CalculationResult with emissions and metadata
        """
        category = input_data["category"]
        data = input_data["data"]

        with create_span(name="calculate_emissions", attributes={"category": category}):
            result = await self.calculate_by_category(category, data)

        # Record metrics
        if self.metrics:
            self.metrics.record_metric(
                f"emissions.category_{category}",
                result.emissions_kgco2e,
                unit="kgCO2e"
            )

        return result

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

    async def calculate_category_2(
        self, data: Union[Category2Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 2 emissions (Capital Goods)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category2Input(**data)
            result = await self.category_2.calculate(data)
            self._update_stats(category=2, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 2 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=2, success=False)
            logger.error(f"Category 2 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_3(
        self, data: Union[Category3Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 3 emissions (Fuel & Energy-Related Activities)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category3Input(**data)
            result = await self.category_3.calculate(data)
            self._update_stats(category=3, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 3 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=3, success=False)
            logger.error(f"Category 3 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_5(
        self, data: Union[Category5Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 5 emissions (Waste Generated in Operations)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category5Input(**data)
            result = await self.category_5.calculate(data)
            self._update_stats(category=5, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 5 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=5, success=False)
            logger.error(f"Category 5 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_7(
        self, data: Union[Category7Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 7 emissions (Employee Commuting)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category7Input(**data)
            result = await self.category_7.calculate(data)
            self._update_stats(category=7, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 7 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=7, success=False)
            logger.error(f"Category 7 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_8(
        self, data: Union[Category8Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 8 emissions (Upstream Leased Assets)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category8Input(**data)
            result = await self.category_8.calculate(data)
            self._update_stats(category=8, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 8 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=8, success=False)
            logger.error(f"Category 8 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_9(
        self, data: Union[Category9Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 9 emissions (Downstream Transportation & Distribution)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category9Input(**data)
            result = await self.category_9.calculate(data)
            self._update_stats(category=9, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 9 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=9, success=False)
            logger.error(f"Category 9 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_10(
        self, data: Union[Category10Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 10 emissions (Processing of Sold Products)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category10Input(**data)
            result = await self.category_10.calculate(data)
            self._update_stats(category=10, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 10 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=10, success=False)
            logger.error(f"Category 10 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_11(
        self, data: Union[Category11Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 11 emissions (Use of Sold Products)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category11Input(**data)
            result = await self.category_11.calculate(data)
            self._update_stats(category=11, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 11 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=11, success=False)
            logger.error(f"Category 11 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_12(
        self, data: Union[Category12Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 12 emissions (End-of-Life Treatment)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category12Input(**data)
            result = await self.category_12.calculate(data)
            self._update_stats(category=12, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 12 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=12, success=False)
            logger.error(f"Category 12 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_13(
        self, data: Union[Category13Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 13 emissions (Downstream Leased Assets)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category13Input(**data)
            result = await self.category_13.calculate(data)
            self._update_stats(category=13, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 13 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=13, success=False)
            logger.error(f"Category 13 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_14(
        self, data: Union[Category14Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 14 emissions (Franchises)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category14Input(**data)
            result = await self.category_14.calculate(data)
            self._update_stats(category=14, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 14 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=14, success=False)
            logger.error(f"Category 14 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_category_15(
        self, data: Union[Category15Input, Dict[str, Any]]
    ) -> CalculationResult:
        """Calculate Category 15 emissions (Investments - PCAF)."""
        start_time = time.time()
        try:
            if isinstance(data, dict):
                data = Category15Input(**data)
            result = await self.category_15.calculate(data)
            self._update_stats(category=15, success=True, processing_time_ms=(time.time()-start_time)*1000)
            logger.info(f"Category 15 calculation completed: {result.emissions_kgco2e:.2f} kgCO2e")
            return result
        except Exception as e:
            self._update_stats(category=15, success=False)
            logger.error(f"Category 15 calculation failed: {e}", exc_info=True)
            raise

    async def calculate_by_category(
        self, category: int, data: Dict[str, Any]
    ) -> CalculationResult:
        """
        Route calculation to appropriate category calculator.

        Args:
            category: Scope 3 category number (1-15)
            data: Input data dictionary

        Returns:
            CalculationResult

        Raises:
            ValueError: If category is not supported
        """
        if category == 1:
            return await self.calculate_category_1(data)
        elif category == 2:
            return await self.calculate_category_2(data)
        elif category == 3:
            return await self.calculate_category_3(data)
        elif category == 4:
            return await self.calculate_category_4(data)
        elif category == 5:
            return await self.calculate_category_5(data)
        elif category == 6:
            return await self.calculate_category_6(data)
        elif category == 7:
            return await self.calculate_category_7(data)
        elif category == 8:
            return await self.calculate_category_8(data)
        elif category == 9:
            return await self.calculate_category_9(data)
        elif category == 10:
            return await self.calculate_category_10(data)
        elif category == 11:
            return await self.calculate_category_11(data)
        elif category == 12:
            return await self.calculate_category_12(data)
        elif category == 13:
            return await self.calculate_category_13(data)
        elif category == 14:
            return await self.calculate_category_14(data)
        elif category == 15:
            return await self.calculate_category_15(data)
        else:
            raise ValueError(
                f"Unsupported category: {category}. "
                f"Supported categories: 1-15"
            )

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

        # Select calculation method using routing
        calc_func = lambda data: self.calculate_by_category(category, data)

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
            "category_breakdown": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0},
            "total_processing_time_ms": 0.0,
        }
        logger.info("Performance statistics reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Scope3CalculatorAgent(categories=[1-15], "
            f"calculations={self.stats['total_calculations']}, "
            f"success_rate={self.get_performance_stats()['success_rate']:.2%})"
        )


__all__ = ["Scope3CalculatorAgent"]
