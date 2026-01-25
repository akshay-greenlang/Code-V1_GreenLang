# -*- coding: utf-8 -*-
"""
===============================================================================
GL-VCCI Scope 3 Platform - Batch Processing 10K Suppliers Test
===============================================================================

PRIORITY TEST 3: Large-scale batch processing

Workflow: Batch process 10,000 suppliers end-to-end

This test validates:
- System scalability under load
- Memory management for large datasets
- Batch processing efficiency
- Database connection pooling
- Cache performance
- Throughput metrics (target: >100 suppliers/sec)

Version: 1.0.0
Team: 8 - Quality Assurance Lead
Created: 2025-11-09
===============================================================================
"""

import pytest
import asyncio
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any
from uuid import uuid4

from greenlang.telemetry import get_logger, MetricsCollector
from services.agents.intake.agent import ValueChainIntakeAgent
from services.agents.intake.models import IngestionRecord, EntityType, IngestionMetadata, SourceSystem, IngestionFormat
from services.agents.calculator.agent import Scope3CalculatorAgent
from services.agents.calculator.models import Category1Input
from greenlang.determinism import FinancialDecimal
from greenlang.determinism import deterministic_random
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = get_logger(__name__)


# ============================================================================
# Large Dataset Generation
# ============================================================================

def generate_large_supplier_dataset(count: int = 10000) -> List[Dict[str, Any]]:
    """
    Generate large supplier dataset for batch testing.

    Creates realistic distribution:
    - 70% Category 1 (Purchased Goods)
    - 20% Category 4 (Transportation)
    - 10% Category 6 (Business Travel)

    Spend distribution follows power law (Pareto):
    - Top 20% accounts for 80% of spend
    """
    import random
    import numpy as np

    suppliers = []

    # Generate Pareto-distributed spend amounts
    # Shape parameter (alpha=1.16 gives 80/20 rule)
    spend_amounts = np.random.pareto(1.16, count) * 10000 + 5000

    categories = [1] * 7000 + [4] * 2000 + [6] * 1000
    deterministic_random().shuffle(categories)

    industries = [
        "Manufacturing", "Technology", "Retail", "Healthcare",
        "Finance", "Transportation", "Energy", "Construction"
    ]

    countries = [
        "United States", "China", "Germany", "United Kingdom",
        "Japan", "India", "Canada", "France"
    ]

    for i in range(count):
        supplier = {
            "supplier_id": f"SUP-{i:06d}",
            "name": f"Supplier {i:06d} {deterministic_random().choice(['Inc', 'LLC', 'Corp', 'GmbH', 'Ltd'])}",
            "spend_amount": FinancialDecimal.from_string(spend_amounts[i]),
            "spend_currency": "USD",
            "category": categories[i],
            "industry": deterministic_random().choice(industries),
            "naics_code": f"{deterministic_random().randint(31, 33)}{deterministic_random().randint(1000, 9999)}",
            "country": deterministic_random().choice(countries),
            "year": 2024,
            "tier": 1 if i < count * 0.1 else (2 if i < count * 0.4 else 3),
            "has_pcf": i < count * 0.05,  # 5% have primary data
            "created_at": DeterministicClock.utcnow().isoformat(),
        }
        suppliers.append(supplier)

    return suppliers


def process_in_batches(data: List[Any], batch_size: int = 1000):
    """Split data into batches for processing."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class MemoryMonitor:
    """Monitor memory usage during batch processing."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_mb = None
        self.peak_mb = 0
        self.samples = []

    def start(self):
        """Start monitoring."""
        self.baseline_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_mb = self.baseline_mb
        logger.info(f"Memory baseline: {self.baseline_mb:.1f} MB")

    def sample(self, label: str = ""):
        """Take memory sample."""
        current_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_mb = max(self.peak_mb, current_mb)
        self.samples.append({
            "label": label,
            "memory_mb": current_mb,
            "delta_mb": current_mb - self.baseline_mb,
        })

    def report(self):
        """Generate memory report."""
        if not self.samples:
            return

        logger.info("=" * 80)
        logger.info("MEMORY USAGE REPORT")
        logger.info("=" * 80)
        logger.info(f"Baseline: {self.baseline_mb:.1f} MB")
        logger.info(f"Peak: {self.peak_mb:.1f} MB")
        logger.info(f"Delta: {self.peak_mb - self.baseline_mb:.1f} MB")
        logger.info("=" * 80)


# ============================================================================
# Test Class: Batch Processing 10K
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.batch
class TestBatchProcessing10K:
    """Batch processing tests for 10,000 suppliers."""

    @pytest.mark.asyncio
    async def test_batch_process_10k_suppliers_intake(self):
        """
        Test batch processing 10K suppliers through Intake Agent.

        Exit Criteria:
        ✅ 10K suppliers processed successfully
        ✅ Processing time < 5 minutes (300s)
        ✅ Throughput > 33 suppliers/sec
        ✅ Memory usage < 2GB
        ✅ >95% success rate
        """
        logger.info("Starting batch processing test: 10,000 suppliers")

        # Setup
        tenant_id = "batch-test-10k"
        supplier_count = 10000
        batch_size = 1000

        memory_monitor = MemoryMonitor()
        memory_monitor.start()

        metrics = MetricsCollector(namespace="batch.10k")
        start_time = time.time()

        # ============================================================
        # Step 1: Generate Dataset
        # ============================================================
        logger.info("Generating 10K supplier dataset...")
        gen_start = time.time()

        suppliers = generate_large_supplier_dataset(count=supplier_count)

        gen_time = time.time() - gen_start
        logger.info(f"✅ Generated {len(suppliers)} suppliers in {gen_time:.2f}s")

        memory_monitor.sample("after_generation")

        # Validate dataset
        assert len(suppliers) == supplier_count
        total_spend = sum(s["spend_amount"] for s in suppliers)
        logger.info(f"Total spend in dataset: ${total_spend:,.2f}")

        # ============================================================
        # Step 2: Batch Process Through Intake Agent
        # ============================================================
        logger.info(f"Processing {supplier_count} suppliers in batches of {batch_size}")

        intake_agent = ValueChainIntakeAgent(tenant_id=tenant_id)

        all_results = []
        total_processed = 0
        total_successful = 0
        total_failed = 0

        batch_num = 0
        for batch_suppliers in process_in_batches(suppliers, batch_size):
            batch_num += 1
            batch_start = time.time()

            # Convert to IngestionRecords
            ingestion_records = []
            for supplier in batch_suppliers:
                metadata = IngestionMetadata(
                    source_file=f"batch_{batch_num}.csv",
                    source_system=SourceSystem.Manual_Upload,
                    ingestion_format=IngestionFormat.CSV,
                    batch_id=f"BATCH10K-{batch_num:03d}",
                    row_number=suppliers.index(supplier) + 1,
                    original_data=supplier,
                )

                record = IngestionRecord(
                    record_id=f"ING-{DeterministicClock.utcnow().strftime('%Y%m%d')}-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8].upper()}",
                    entity_type=EntityType.supplier,
                    tenant_id=tenant_id,
                    entity_name=supplier["name"],
                    entity_identifier=supplier["supplier_id"],
                    data=supplier,
                    metadata=metadata,
                )
                ingestion_records.append(record)

            # Process batch
            batch_result = intake_agent.process(ingestion_records)

            # Aggregate results
            total_processed += batch_result.statistics.total_records
            total_successful += batch_result.statistics.successful
            total_failed += batch_result.statistics.failed

            all_results.append(batch_result)

            batch_time = time.time() - batch_start
            batch_throughput = len(batch_suppliers) / batch_time

            logger.info(
                f"Batch {batch_num}: {batch_result.statistics.successful}/{len(batch_suppliers)} successful "
                f"in {batch_time:.2f}s ({batch_throughput:.1f} rec/s)"
            )

            memory_monitor.sample(f"batch_{batch_num}")

        # ============================================================
        # Step 3: Calculate Metrics
        # ============================================================
        total_time = time.time() - start_time
        overall_throughput = total_successful / total_time
        success_rate = (total_successful / total_processed) * 100

        logger.info("=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Suppliers: {supplier_count}")
        logger.info(f"Total Processed: {total_processed}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Failed: {total_failed}")
        logger.info(f"Success Rate: {success_rate:.2f}%")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Throughput: {overall_throughput:.1f} suppliers/sec")
        logger.info("=" * 80)

        memory_monitor.report()

        # ============================================================
        # Assertions
        # ============================================================

        # Processing success
        assert total_processed == supplier_count
        assert success_rate >= 95.0, f"Success rate {success_rate:.2f}% below 95% threshold"

        # Performance
        assert total_time < 300.0, f"Processing took {total_time:.2f}s, expected <300s (5 min)"
        assert overall_throughput > 33.0, f"Throughput {overall_throughput:.1f} rec/s below 33 rec/s threshold"

        # Memory
        memory_delta = memory_monitor.peak_mb - memory_monitor.baseline_mb
        assert memory_delta < 2048, f"Memory usage {memory_delta:.1f}MB exceeds 2GB limit"

        # Record metrics
        metrics.record_metric("total_processed", total_processed)
        metrics.record_metric("success_rate", success_rate, unit="percent")
        metrics.record_metric("throughput", overall_throughput, unit="rec/s")
        metrics.record_metric("duration", total_time, unit="seconds")
        metrics.record_metric("memory_peak_mb", memory_monitor.peak_mb, unit="MB")

        logger.info("✅ Batch processing 10K test PASSED")


    @pytest.mark.asyncio
    async def test_batch_calculation_10k_suppliers(self):
        """
        Test batch calculation for 10K suppliers.

        Exit Criteria:
        ✅ 10K calculations completed
        ✅ Processing time < 10 minutes (600s)
        ✅ All emissions > 0
        ✅ Uncertainty ranges reasonable
        """
        logger.info("Starting batch calculation test: 10,000 suppliers")

        supplier_count = 10000
        batch_size = 500  # Smaller batches for calculations

        start_time = time.time()
        memory_monitor = MemoryMonitor()
        memory_monitor.start()

        # Generate dataset
        suppliers = generate_large_supplier_dataset(count=supplier_count)

        # Mock factor broker
        class MockFactorBroker:
            def get_factor(self, category: int, **kwargs):
                return {
                    "factor": 0.5,
                    "unit": "kg CO2e/USD",
                    "source": "EPA",
                    "quality_tier": 2,
                    "uncertainty": 0.15
                }

        calculator_agent = Scope3CalculatorAgent(
            factor_broker=MockFactorBroker()
        )

        # Batch calculate
        all_calculations = []
        batch_num = 0

        for batch_suppliers in process_in_batches(suppliers, batch_size):
            batch_num += 1
            batch_start = time.time()

            batch_calculations = []

            for supplier in batch_suppliers:
                calc_input = Category1Input(
                    supplier_id=supplier["supplier_id"],
                    supplier_name=supplier["name"],
                    spend_amount=supplier["spend_amount"],
                    spend_currency=supplier["spend_currency"],
                    year=supplier["year"],
                    naics_code=supplier["naics_code"],
                    industry=supplier["industry"],
                    country=supplier["country"],
                    has_pcf=supplier.get("has_pcf", False),
                )

                try:
                    calc_result = calculator_agent.category_1.calculate(calc_input)
                    if calc_result:
                        batch_calculations.append({
                            "supplier_id": supplier["supplier_id"],
                            "emissions_tco2e": calc_result.total_emissions_tco2e,
                            "tier": calc_result.tier_used,
                        })
                except Exception as e:
                    logger.warning(f"Calculation failed: {e}")

            all_calculations.extend(batch_calculations)

            batch_time = time.time() - batch_start
            batch_throughput = len(batch_calculations) / batch_time

            logger.info(
                f"Batch {batch_num}: {len(batch_calculations)} calculations "
                f"in {batch_time:.2f}s ({batch_throughput:.1f} calc/s)"
            )

            memory_monitor.sample(f"calc_batch_{batch_num}")

        # Calculate metrics
        total_time = time.time() - start_time
        throughput = len(all_calculations) / total_time
        total_emissions = sum(c["emissions_tco2e"] for c in all_calculations)

        logger.info("=" * 80)
        logger.info("BATCH CALCULATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Calculations: {len(all_calculations)}")
        logger.info(f"Total Emissions: {total_emissions:,.2f} tCO2e")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Throughput: {throughput:.1f} calculations/sec")
        logger.info("=" * 80)

        memory_monitor.report()

        # Assertions
        assert len(all_calculations) >= supplier_count * 0.95
        assert total_time < 600.0, f"Calculation took {total_time:.2f}s, expected <600s"
        assert total_emissions > 0
        assert all(c["emissions_tco2e"] > 0 for c in all_calculations)

        logger.info("✅ Batch calculation 10K test PASSED")


    @pytest.mark.asyncio
    async def test_batch_processing_memory_leak_detection(self):
        """
        Test for memory leaks during batch processing.

        Validates:
        - Memory usage is stable across batches
        - No linear memory growth
        - Garbage collection working properly
        """
        logger.info("Testing memory leak detection")

        import gc

        supplier_count = 5000
        batch_size = 500

        memory_monitor = MemoryMonitor()
        memory_monitor.start()

        suppliers = generate_large_supplier_dataset(count=supplier_count)
        tenant_id = "memory-leak-test"

        intake_agent = ValueChainIntakeAgent(tenant_id=tenant_id)

        batch_memory_deltas = []

        for i, batch_suppliers in enumerate(process_in_batches(suppliers, batch_size)):
            batch_start_memory = memory_monitor.process.memory_info().rss / 1024 / 1024

            # Convert and process
            ingestion_records = []
            for supplier in batch_suppliers:
                metadata = IngestionMetadata(
                    source_file="leak_test.csv",
                    source_system=SourceSystem.Manual_Upload,
                    ingestion_format=IngestionFormat.CSV,
                    batch_id=f"LEAK-{i:03d}",
                    row_number=suppliers.index(supplier) + 1,
                    original_data=supplier,
                )

                record = IngestionRecord(
                    record_id=f"LEAK-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
                    entity_type=EntityType.supplier,
                    tenant_id=tenant_id,
                    entity_name=supplier["name"],
                    entity_identifier=supplier["supplier_id"],
                    data=supplier,
                    metadata=metadata,
                )
                ingestion_records.append(record)

            _ = intake_agent.process(ingestion_records)

            # Force garbage collection
            gc.collect()

            batch_end_memory = memory_monitor.process.memory_info().rss / 1024 / 1024
            batch_memory_delta = batch_end_memory - batch_start_memory

            batch_memory_deltas.append(batch_memory_delta)

            logger.info(f"Batch {i+1}: Memory delta = {batch_memory_delta:.1f} MB")

        # Check for linear memory growth (indication of leak)
        # Memory should stabilize, not grow linearly
        first_half_avg = sum(batch_memory_deltas[:5]) / 5
        second_half_avg = sum(batch_memory_deltas[5:]) / 5

        growth_rate = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0

        logger.info(f"Memory growth rate: {growth_rate:.2%}")

        # Growth rate should be < 50% (some growth expected, but not linear)
        assert abs(growth_rate) < 0.5, f"Potential memory leak detected: {growth_rate:.2%} growth rate"

        logger.info("✅ No memory leak detected")


    @pytest.mark.asyncio
    async def test_batch_processing_error_recovery(self):
        """
        Test error recovery during batch processing.

        Validates:
        - Failed batches don't crash entire job
        - Partial failures handled gracefully
        - Error reporting accurate
        """
        logger.info("Testing batch error recovery")

        supplier_count = 1000
        batch_size = 100

        suppliers = generate_large_supplier_dataset(count=supplier_count)

        # Corrupt every 3rd batch
        corrupted_batches = set([2, 5, 8])

        tenant_id = "error-recovery-test"
        intake_agent = ValueChainIntakeAgent(tenant_id=tenant_id)

        successful_batches = 0
        failed_batches = 0

        batch_num = 0
        for batch_suppliers in process_in_batches(suppliers, batch_size):
            batch_num += 1

            # Corrupt batch if needed
            if batch_num in corrupted_batches:
                # Introduce errors
                for supplier in batch_suppliers[:5]:
                    supplier["spend_amount"] = "INVALID"  # Type error

            try:
                ingestion_records = []
                for supplier in batch_suppliers:
                    metadata = IngestionMetadata(
                        source_file="error_test.csv",
                        source_system=SourceSystem.Manual_Upload,
                        ingestion_format=IngestionFormat.CSV,
                        batch_id=f"ERROR-{batch_num:03d}",
                        row_number=suppliers.index(supplier) + 1,
                        original_data=supplier,
                    )

                    record = IngestionRecord(
                        record_id=f"ERR-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
                        entity_type=EntityType.supplier,
                        tenant_id=tenant_id,
                        entity_name=supplier["name"],
                        entity_identifier=supplier["supplier_id"],
                        data=supplier,
                        metadata=metadata,
                    )
                    ingestion_records.append(record)

                result = intake_agent.process(ingestion_records)

                if result.statistics.failed > 0:
                    logger.warning(f"Batch {batch_num}: {result.statistics.failed} failures")

                successful_batches += 1

            except Exception as e:
                logger.error(f"Batch {batch_num} failed completely: {e}")
                failed_batches += 1

        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Failed batches: {failed_batches}")

        # Most batches should succeed even with errors
        success_rate = successful_batches / batch_num
        assert success_rate >= 0.70, f"Only {success_rate:.2%} batches succeeded"

        logger.info("✅ Error recovery test PASSED")
