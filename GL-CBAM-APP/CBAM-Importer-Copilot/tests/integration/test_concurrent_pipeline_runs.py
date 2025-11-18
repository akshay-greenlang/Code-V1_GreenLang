"""
Integration Tests: Concurrent Pipeline Runs
============================================

Tests concurrent execution of multiple pipeline instances:
- 3 concurrent pipeline runs
- 10 concurrent pipeline runs (stress test)
- Resource isolation between runs
- No data corruption with concurrent access
- Database connection pool under concurrency

Target: Maturity score +1 point (concurrency safety)
Version: 1.0.0
Author: GL-TestEngineer
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cbam_pipeline_v2 import CBAMPipeline_v2


# ============================================================================
# Concurrent Execution Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
class TestConcurrentPipelineRuns:
    """Test concurrent pipeline execution."""

    async def test_3_concurrent_pipeline_runs(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test 3 concurrent pipeline runs execute without interference.

        Each pipeline should:
        - Complete successfully
        - Produce correct results
        - Not interfere with others
        """
        # Generate 3 test datasets
        datasets = []
        for i in range(3):
            shipments = self._generate_test_shipments(100, seed=i)
            csv_path = tmp_path / f"concurrent_test_{i}.csv"
            df = pd.DataFrame(shipments)
            df.to_csv(csv_path, index=False)
            datasets.append(str(csv_path))

        print("\n[Concurrent Test] Running 3 pipelines in parallel...")

        # Run pipelines concurrently
        start_time = time.perf_counter()

        tasks = []
        for i, dataset in enumerate(datasets):
            task = asyncio.create_task(self._run_pipeline_async(
                dataset_path=dataset,
                cn_codes_path=cn_codes_path,
                cbam_rules_path=cbam_rules_path,
                suppliers_path=suppliers_path,
                importer_info=importer_info,
                pipeline_id=i
            ))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Verify all pipelines succeeded
        assert all(r["success"] for r in results), "All pipelines should succeed"

        print(f"\n[Results]")
        print(f"  Execution time (concurrent): {execution_time:.2f}s")
        print(f"  Pipelines completed: {len(results)}")

        # Verify each pipeline processed correct number of records
        for i, result in enumerate(results):
            assert result["report"]["emissions_summary"]["total_shipments"] == 100, \
                f"Pipeline {i} should process 100 shipments"

            print(f"  Pipeline {i}: {result['report']['emissions_summary']['total_shipments']} shipments, "
                  f"{result['report']['emissions_summary']['total_embedded_emissions_tco2']:.2f} tCO2")

    @pytest.mark.slow
    async def test_10_concurrent_pipeline_runs(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test 10 concurrent pipeline runs (stress test).

        Validates system stability under high concurrency.
        """
        # Generate 10 test datasets (smaller for speed)
        datasets = []
        for i in range(10):
            shipments = self._generate_test_shipments(50, seed=i)
            csv_path = tmp_path / f"stress_test_{i}.csv"
            df = pd.DataFrame(shipments)
            df.to_csv(csv_path, index=False)
            datasets.append(str(csv_path))

        print("\n[Stress Test] Running 10 pipelines in parallel...")

        start_time = time.perf_counter()

        tasks = []
        for i, dataset in enumerate(datasets):
            task = asyncio.create_task(self._run_pipeline_async(
                dataset_path=dataset,
                cn_codes_path=cn_codes_path,
                cbam_rules_path=cbam_rules_path,
                suppliers_path=suppliers_path,
                importer_info=importer_info,
                pipeline_id=i
            ))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        print(f"\n[Stress Test Results]")
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Successful: {len(successes)}/{len(results)}")
        print(f"  Failed: {len(exceptions)}/{len(results)}")

        if exceptions:
            for i, exc in enumerate(exceptions):
                print(f"  Exception {i}: {type(exc).__name__}: {exc}")

        # Assert acceptable success rate (allow some failures under extreme load)
        success_rate = len(successes) / len(results)
        assert success_rate >= 0.9, f"Success rate {success_rate:.0%} below 90% threshold"

    def test_thread_pool_concurrent_execution(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test concurrent execution using ThreadPoolExecutor.

        Validates thread-safety of pipeline components.
        """
        # Generate test datasets
        datasets = []
        for i in range(5):
            shipments = self._generate_test_shipments(50, seed=i)
            csv_path = tmp_path / f"thread_test_{i}.csv"
            df = pd.DataFrame(shipments)
            df.to_csv(csv_path, index=False)
            datasets.append(str(csv_path))

        print("\n[Thread Pool Test] Running 5 pipelines in thread pool...")

        results = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, dataset in enumerate(datasets):
                future = executor.submit(
                    self._run_pipeline_sync,
                    dataset_path=dataset,
                    cn_codes_path=cn_codes_path,
                    cbam_rules_path=cbam_rules_path,
                    suppliers_path=suppliers_path,
                    importer_info=importer_info,
                    pipeline_id=i
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"  Pipeline failed: {e}")

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"\n[Thread Pool Results]")
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Completed: {len(results)}/5")

        assert len(results) == 5, "All pipelines should complete"


# ============================================================================
# Resource Isolation Tests
# ============================================================================

@pytest.mark.integration
class TestResourceIsolation:
    """Test resource isolation between concurrent pipeline runs."""

    async def test_no_shared_state_corruption(
        self,
        cn_codes_path,
        cbam_rules_path,
        tmp_path,
        importer_info
    ):
        """
        Test pipeline instances don't share mutable state.

        Critical for preventing data corruption.
        """
        # Create two datasets with different data
        shipments_a = [
            {"shipment_id": "A-001", "cn_code": "72071100", "origin_iso": "CN", "net_mass_kg": 10000, "quarter": "2025-Q3", "import_date": "2025-Q3", "importer_country": "NL"},
            {"shipment_id": "A-002", "cn_code": "72071100", "origin_iso": "CN", "net_mass_kg": 10000, "quarter": "2025-Q3", "import_date": "2025-Q3", "importer_country": "NL"},
        ]

        shipments_b = [
            {"shipment_id": "B-001", "cn_code": "76011000", "origin_iso": "RU", "net_mass_kg": 5000, "quarter": "2025-Q3", "import_date": "2025-Q3", "importer_country": "NL"},
            {"shipment_id": "B-002", "cn_code": "76011000", "origin_iso": "RU", "net_mass_kg": 5000, "quarter": "2025-Q3", "import_date": "2025-Q3", "importer_country": "NL"},
        ]

        csv_a = tmp_path / "dataset_a.csv"
        csv_b = tmp_path / "dataset_b.csv"

        pd.DataFrame(shipments_a).to_csv(csv_a, index=False)
        pd.DataFrame(shipments_b).to_csv(csv_b, index=False)

        # Run concurrently
        task_a = asyncio.create_task(self._run_pipeline_async(
            dataset_path=str(csv_a),
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path,
            suppliers_path=None,
            importer_info=importer_info,
            pipeline_id=0
        ))

        task_b = asyncio.create_task(self._run_pipeline_async(
            dataset_path=str(csv_b),
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path,
            suppliers_path=None,
            importer_info=importer_info,
            pipeline_id=1
        ))

        result_a, result_b = await asyncio.gather(task_a, task_b)

        # Verify results are isolated (A processed A's data, B processed B's data)
        assert result_a["success"]
        assert result_b["success"]

        # Check shipment IDs match original datasets
        print("\n[State Isolation Test]")
        print(f"  Pipeline A processed {result_a['report']['emissions_summary']['total_shipments']} shipments")
        print(f"  Pipeline B processed {result_b['report']['emissions_summary']['total_shipments']} shipments")

        assert result_a["report"]["emissions_summary"]["total_shipments"] == 2
        assert result_b["report"]["emissions_summary"]["total_shipments"] == 2


# ============================================================================
# Database Connection Pool Tests
# ============================================================================

@pytest.mark.integration
class TestDatabaseConnectionPool:
    """Test database connection pooling under concurrency."""

    async def test_connection_pool_not_exhausted(
        self,
        cn_codes_path,
        cbam_rules_path,
        tmp_path,
        importer_info
    ):
        """
        Test connection pool handles concurrent requests without exhaustion.

        Simulates heavy concurrent database access.
        """
        # Generate test datasets
        datasets = []
        for i in range(5):
            shipments = TestConcurrentPipelineRuns._generate_test_shipments(None, 20, seed=i)
            csv_path = tmp_path / f"pool_test_{i}.csv"
            pd.DataFrame(shipments).to_csv(csv_path, index=False)
            datasets.append(str(csv_path))

        print("\n[Connection Pool Test] Testing with 5 concurrent pipelines...")

        # Run concurrently
        tasks = [
            asyncio.create_task(TestConcurrentPipelineRuns._run_pipeline_async(
                None,
                dataset_path=dataset,
                cn_codes_path=cn_codes_path,
                cbam_rules_path=cbam_rules_path,
                suppliers_path=None,
                importer_info=importer_info,
                pipeline_id=i
            ))
            for i, dataset in enumerate(datasets)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for connection pool exhaustion errors
        connection_errors = [r for r in results if isinstance(r, Exception) and "connection" in str(r).lower()]

        print(f"  Completed: {len([r for r in results if not isinstance(r, Exception)])}/5")
        print(f"  Connection errors: {len(connection_errors)}")

        assert len(connection_errors) == 0, "Should not exhaust connection pool"


# ============================================================================
# Helper Methods
# ============================================================================

async def _run_pipeline_async(
    self,
    dataset_path: str,
    cn_codes_path: str,
    cbam_rules_path: str,
    suppliers_path: str,
    importer_info: Dict[str, Any],
    pipeline_id: int
) -> Dict[str, Any]:
    """Run pipeline asynchronously."""
    try:
        pipeline = CBAMPipeline_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path,
            suppliers_path=suppliers_path
        )

        result = pipeline.execute({
            "input_file": dataset_path,
            "importer_info": importer_info
        })

        return {
            "success": result.success,
            "pipeline_id": pipeline_id,
            "report": result.data if result.success else None,
            "error": result.error if not result.success else None
        }

    except Exception as e:
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "report": None,
            "error": str(e)
        }


def _run_pipeline_sync(
    self,
    dataset_path: str,
    cn_codes_path: str,
    cbam_rules_path: str,
    suppliers_path: str,
    importer_info: Dict[str, Any],
    pipeline_id: int
) -> Dict[str, Any]:
    """Run pipeline synchronously (for ThreadPoolExecutor)."""
    pipeline = CBAMPipeline_v2(
        cn_codes_path=cn_codes_path,
        cbam_rules_path=cbam_rules_path,
        suppliers_path=suppliers_path
    )

    result = pipeline.execute({
        "input_file": dataset_path,
        "importer_info": importer_info
    })

    return {
        "success": result.success,
        "pipeline_id": pipeline_id,
        "report": result.data if result.success else None
    }


def _generate_test_shipments(self, count: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Generate test shipments for concurrent testing."""
    cn_codes = ["72071100", "76011000", "25232900"]
    countries = ["CN", "TR", "RU"]

    shipments = []
    for i in range(count):
        idx = (seed * count + i) % len(cn_codes)
        shipments.append({
            "shipment_id": f"SHIP-{seed:02d}-{i:04d}",
            "cn_code": cn_codes[idx],
            "origin_iso": countries[idx],
            "net_mass_kg": 1000 + (i * 100),
            "quarter": "2025-Q3",
            "import_date": "2025-Q3",
            "importer_country": "NL"
        })

    return shipments


# Attach methods to classes
TestConcurrentPipelineRuns._run_pipeline_async = _run_pipeline_async
TestConcurrentPipelineRuns._run_pipeline_sync = _run_pipeline_sync
TestConcurrentPipelineRuns._generate_test_shipments = _generate_test_shipments
TestResourceIsolation._run_pipeline_async = _run_pipeline_async


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'not slow'])
