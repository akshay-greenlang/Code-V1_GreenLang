# -*- coding: utf-8 -*-
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
from greenlang.determinism import DeterministicClock
from greenlang.telemetry import (
    MetricsCollector,
    get_logger,
    track_execution,
    create_span,
)
from greenlang.validation import ValidationFramework, ValidationResult as VResult, Rule, RuleOperator, ValidationSeverity

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

        # Initialize ValidationFramework with security rules
        self.validator = ValidationFramework()
        self._setup_validation_rules()

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

    def _setup_validation_rules(self):
        """
        Setup validation rules for input data security.

        This adds validators for:
        - Positive numeric values (quantity, emission_factor, distance, weight, etc.)
        - Valid category ranges
        - Required fields presence
        """
        def validate_positive_numbers(data: Dict[str, Any]) -> VResult:
            """Validate that numeric fields are positive."""
            result = VResult(valid=True)

            # Common numeric fields that should be positive
            positive_fields = [
                "quantity", "emission_factor", "distance", "weight",
                "mass_kg", "distance_km", "spend_amount", "price",
                "emissions_kgco2e", "value", "amount"
            ]

            for field in positive_fields:
                value = data.get(field)
                if value is not None:
                    try:
                        num_value = float(value)
                        if num_value < 0:
                            from greenlang.validation import ValidationError as VError
                            error = VError(
                                field=field,
                                message=f"{field} must be positive, got {num_value}",
                                severity=ValidationSeverity.ERROR,
                                validator="positive_numbers",
                                value=num_value,
                                expected="positive number"
                            )
                            result.add_error(error)
                    except (ValueError, TypeError):
                        pass  # Type validation is handled elsewhere

            return result

        # Register validators
        self.validator.add_validator("positive_numbers", validate_positive_numbers)
        logger.debug("Validation rules configured for Scope3CalculatorAgent")

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

        Raises:
            CalculatorError: If validation fails or calculation error occurs
        """
        category = input_data["category"]
        data = input_data["data"]

        # Validate input data with ValidationFramework
        validation_result = self.validator.validate(data)
        if not validation_result.valid:
            error_msg = f"Input validation failed: {validation_result.get_summary()}"
            logger.error(error_msg)
            for error in validation_result.errors:
                logger.error(f"  - {error}")
            raise CalculatorError(error_msg)

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

    async def process_suppliers_optimized(
        self,
        suppliers: List[Union[Dict[str, Any], Category1Input]],
        category: int = 1,
        chunk_size: int = 1000,
        db_connection: Optional[Any] = None
    ):
        """
        Process suppliers in optimized chunks for 100K/hour throughput.

        PERFORMANCE TARGET: 100,000 suppliers per hour
        - Chunk size: 1000 suppliers per batch
        - Parallel processing within chunks
        - Bulk database operations
        - Memory-efficient streaming

        Args:
            suppliers: List of supplier records to process
            category: Scope 3 category (default: 1 for purchased goods)
            chunk_size: Number of suppliers per chunk (default: 1000)
            db_connection: Optional database connection for bulk inserts

        Yields:
            Tuple of (chunk_index, chunk_results, chunk_metrics)

        Example:
            >>> async for chunk_idx, results, metrics in agent.process_suppliers_optimized(suppliers):
            ...     print(f"Chunk {chunk_idx}: {len(results)} processed in {metrics['time_ms']:.2f}ms")
            ...     print(f"Throughput: {metrics['throughput_per_hour']:.0f} suppliers/hour")
        """
        start_time = time.time()
        total_suppliers = len(suppliers)

        logger.info(
            f"Starting optimized batch processing: {total_suppliers} suppliers, "
            f"chunk_size={chunk_size}, category={category}"
        )

        # Track overall metrics
        total_processed = 0
        total_successful = 0
        total_failed = 0
        total_emissions = 0.0

        # Process suppliers in chunks
        for chunk_idx in range(0, total_suppliers, chunk_size):
            chunk_start = time.time()

            # Extract chunk
            chunk = suppliers[chunk_idx:chunk_idx + chunk_size]
            chunk_num = chunk_idx // chunk_size + 1

            logger.info(
                f"Processing chunk {chunk_num}/{(total_suppliers + chunk_size - 1) // chunk_size}: "
                f"{len(chunk)} suppliers"
            )

            # Process chunk in parallel
            calc_func = lambda data: self.calculate_by_category(category, data)
            tasks = [calc_func(supplier) for supplier in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate successful results from errors
            successful_results = []
            chunk_errors = []
            chunk_emissions = 0.0

            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    chunk_errors.append({
                        "supplier_index": chunk_idx + i,
                        "error": str(result),
                        "supplier_data": chunk[i] if isinstance(chunk[i], dict) else chunk[i].dict()
                    })
                    total_failed += 1
                else:
                    successful_results.append(result)
                    chunk_emissions += result.emissions_kgco2e
                    total_successful += 1

            total_processed += len(chunk)
            total_emissions += chunk_emissions

            # Bulk insert results if database connection provided
            if db_connection and successful_results:
                try:
                    await self._bulk_insert_results(db_connection, successful_results)
                    logger.debug(f"Bulk inserted {len(successful_results)} results to database")
                except Exception as e:
                    logger.error(f"Bulk insert failed for chunk {chunk_num}: {e}")

            # Calculate chunk metrics
            chunk_time = time.time() - chunk_start
            chunk_throughput_per_second = len(chunk) / chunk_time if chunk_time > 0 else 0
            chunk_throughput_per_hour = chunk_throughput_per_second * 3600

            chunk_metrics = {
                "chunk_index": chunk_num,
                "chunk_size": len(chunk),
                "successful": len(successful_results),
                "failed": len(chunk_errors),
                "emissions_kgco2e": chunk_emissions,
                "emissions_tco2e": chunk_emissions / 1000,
                "time_seconds": chunk_time,
                "time_ms": chunk_time * 1000,
                "throughput_per_second": chunk_throughput_per_second,
                "throughput_per_hour": chunk_throughput_per_hour,
                "errors": chunk_errors
            }

            # Log chunk performance
            logger.info(
                f"Chunk {chunk_num} completed: {len(successful_results)}/{len(chunk)} successful, "
                f"{chunk_emissions / 1000:.3f} tCO2e, "
                f"{chunk_time:.2f}s, "
                f"throughput: {chunk_throughput_per_hour:.0f}/hour"
            )

            # Yield chunk results for streaming processing
            yield chunk_num, successful_results, chunk_metrics

            # Memory management: force garbage collection for large batches
            if chunk_num % 10 == 0:
                import gc
                gc.collect()

        # Calculate final metrics
        total_time = time.time() - start_time
        overall_throughput_per_second = total_processed / total_time if total_time > 0 else 0
        overall_throughput_per_hour = overall_throughput_per_second * 3600

        logger.info(
            f"Optimized batch processing completed: "
            f"{total_successful}/{total_processed} successful "
            f"({total_failed} failed), "
            f"total emissions: {total_emissions / 1000:.3f} tCO2e, "
            f"total time: {total_time:.2f}s, "
            f"overall throughput: {overall_throughput_per_hour:.0f} suppliers/hour"
        )

        # Record final metrics
        if self.metrics:
            self.metrics.record_metric(
                "batch_processing.throughput_per_hour",
                overall_throughput_per_hour,
                unit="suppliers/hour"
            )
            self.metrics.record_metric(
                "batch_processing.total_emissions",
                total_emissions,
                unit="kgCO2e"
            )
            self.metrics.record_metric(
                "batch_processing.success_rate",
                total_successful / total_processed if total_processed > 0 else 0,
                unit="percentage"
            )

    async def process_single(self, supplier_data: Union[Dict[str, Any], Category1Input]) -> CalculationResult:
        """
        Process a single supplier calculation.

        Helper method for batch processing optimization.

        Args:
            supplier_data: Supplier input data

        Returns:
            CalculationResult
        """
        return await self.calculate_category_1(supplier_data)

    async def _bulk_insert_results(
        self,
        db_connection: Any,
        results: List[CalculationResult]
    ):
        """
        Bulk insert calculation results to database.

        Optimized for high-throughput batch processing.

        Args:
            db_connection: Database connection object
            results: List of calculation results to insert
        """
        if not results:
            return

        # Prepare bulk insert data
        insert_data = [
            {
                "emissions_kgco2e": r.emissions_kgco2e,
                "emissions_tco2e": r.emissions_tco2e,
                "tier": r.tier,
                "dqi_score": r.data_quality.dqi_score if r.data_quality else None,
                "calculation_method": r.calculation_method,
                "timestamp": DeterministicClock.utcnow(),
                "provenance_chain": r.provenance_chain if hasattr(r, 'provenance_chain') else None,
            }
            for r in results
        ]

        # Execute bulk insert (implementation depends on database type)
        # This is a placeholder - actual implementation would depend on db_connection type
        try:
            if hasattr(db_connection, 'bulk_insert'):
                await db_connection.bulk_insert('calculation_results', insert_data)
            elif hasattr(db_connection, 'executemany'):
                # For SQL-based databases
                placeholders = ', '.join(['?' for _ in insert_data[0].keys()])
                columns = ', '.join(insert_data[0].keys())
                query = f"INSERT INTO calculation_results ({columns}) VALUES ({placeholders})"
                await db_connection.executemany(query, [list(d.values()) for d in insert_data])
            else:
                logger.warning("Database connection does not support bulk insert - using individual inserts")
                for data in insert_data:
                    await db_connection.insert('calculation_results', data)
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}", exc_info=True)
            raise

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
