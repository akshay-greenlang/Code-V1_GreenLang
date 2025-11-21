# -*- coding: utf-8 -*-
"""
Unit Tests for Batch Calculator

Tests high-performance batch processing of emission calculations.
"""

import pytest
from decimal import Decimal
import time


class TestBatchCalculator:
    """Test BatchCalculator functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup batch calculator"""
        try:
            from greenlang.calculation.batch_calculator import BatchCalculator
            self.calculator = BatchCalculator()
        except ImportError:
            pytest.skip("BatchCalculator not available")

    def test_batch_calculate_100_requests(self):
        """Test processing batch of 100 calculations"""
        requests = []

        for i in range(100):
            requests.append({
                "factor_id": "diesel-us-stationary",
                "activity_amount": 100 + i,
                "activity_unit": "liters",
                "request_id": f"REQ-{i:04d}"
            })

        results = self.calculator.calculate_batch(requests)

        assert len(results) == 100
        assert all(r.get("status") == "success" for r in results)

    @pytest.mark.performance
    def test_batch_throughput_target(self):
        """Test batch processing meets throughput target (>1000 req/s)"""
        requests = []

        for i in range(1000):
            requests.append({
                "factor_id": "diesel-us-stationary",
                "activity_amount": 100,
                "activity_unit": "liters",
                "request_id": f"REQ-{i:06d}"
            })

        start = time.perf_counter()
        results = self.calculator.calculate_batch(requests)
        duration = time.perf_counter() - start

        throughput = len(requests) / duration

        # Target: >1000 calculations/second
        assert throughput >= 1000, f"Throughput {throughput:.0f} req/s below target"

    def test_batch_parallel_processing(self):
        """Test batch calculator uses parallel processing"""
        requests = []

        for i in range(100):
            requests.append({
                "factor_id": "diesel-us-stationary",
                "activity_amount": 100,
                "activity_unit": "liters"
            })

        # Process with parallelism
        start = time.perf_counter()
        results_parallel = self.calculator.calculate_batch(
            requests,
            parallel=True,
            workers=4
        )
        duration_parallel = time.perf_counter() - start

        # Process sequentially
        start = time.perf_counter()
        results_sequential = self.calculator.calculate_batch(
            requests,
            parallel=False
        )
        duration_sequential = time.perf_counter() - start

        # Parallel should be faster
        assert duration_parallel < duration_sequential


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
