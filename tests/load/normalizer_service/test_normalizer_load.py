# -*- coding: utf-8 -*-
"""
Load Tests for Normalizer Service (AGENT-FOUND-003)

Tests throughput and concurrency for conversions, batch operations,
entity resolutions, and caching behavior under load.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Import inline test implementations
# ---------------------------------------------------------------------------

from tests.unit.normalizer_service.test_converter import UnitConverter
from tests.unit.normalizer_service.test_entity_resolver import EntityResolver


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestConcurrentConversions:
    """Test concurrent conversion throughput."""

    @pytest.mark.slow
    def test_50_concurrent_conversions(self):
        """50 conversions running concurrently should all succeed."""
        converter = UnitConverter()

        def do_convert(i: int):
            r = converter.convert(i * 100, "kg", "t")
            assert r.ok
            return r

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_convert, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50
        assert all(r.ok for r in results)

    @pytest.mark.slow
    def test_50_concurrent_mixed_dimensions(self):
        """50 concurrent conversions across different dimensions."""
        converter = UnitConverter()
        pairs = [
            ("kg", "t"), ("kWh", "MWh"), ("L", "m3"), ("m", "km"),
            ("m2", "hectare"),
        ]

        def do_convert(i: int):
            from_u, to_u = pairs[i % len(pairs)]
            r = converter.convert(100, from_u, to_u)
            assert r.ok
            return r

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_convert, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50


class TestBatchConversionLoad:
    """Test batch conversion under load."""

    @pytest.mark.slow
    def test_100_item_batch(self):
        """Process 100 items in a single batch."""
        converter = UnitConverter()
        items = [
            {"value": i * 10, "from_unit": "kg", "to_unit": "t"}
            for i in range(100)
        ]
        results = converter.batch_convert(items)
        assert len(results) == 100
        assert all(r.ok for r in results)

    @pytest.mark.slow
    def test_500_item_batch(self):
        """Process 500 items in a single batch."""
        converter = UnitConverter()
        items = [
            {"value": i, "from_unit": "kWh", "to_unit": "MWh"}
            for i in range(500)
        ]
        results = converter.batch_convert(items)
        assert len(results) == 500
        assert all(r.ok for r in results)


class TestRepeatedConversions:
    """Test repeated conversions for caching behavior."""

    @pytest.mark.slow
    def test_1000_repeated_conversions(self):
        """1000 identical conversions should all produce the same result."""
        converter = UnitConverter()
        first = converter.convert(100, "kg", "t")
        assert first.ok

        for _ in range(999):
            r = converter.convert(100, "kg", "t")
            assert r.ok
            assert r.value == first.value
            assert r.provenance_hash == first.provenance_hash

    @pytest.mark.slow
    def test_1000_varied_conversions_throughput(self):
        """1000 conversions with varying values should complete quickly."""
        converter = UnitConverter()
        start = time.time()

        for i in range(1000):
            r = converter.convert(i, "kWh", "MWh")
            assert r.ok

        elapsed = time.time() - start
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"1000 conversions took {elapsed:.2f}s (target: <5s)"


class TestConcurrentEntityResolution:
    """Test concurrent entity resolution."""

    @pytest.mark.slow
    def test_50_concurrent_fuel_resolutions(self):
        """50 fuel resolutions running concurrently."""
        resolver = EntityResolver()
        fuels = [
            "Natural Gas", "Diesel", "Coal", "LPG", "Biogas",
            "Biomass", "Gasoline", "Fuel Oil", "Kerosene", "Propane",
        ]

        def do_resolve(i: int):
            name = fuels[i % len(fuels)]
            match = resolver.resolve_fuel(name)
            assert match.confidence > 0
            return match

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_resolve, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50

    @pytest.mark.slow
    def test_50_concurrent_material_resolutions(self):
        """50 material resolutions running concurrently."""
        resolver = EntityResolver()
        materials = ["Steel", "Aluminum", "Cement", "Glass", "Copper"]

        def do_resolve(i: int):
            name = materials[i % len(materials)]
            match = resolver.resolve_material(name)
            assert match.confidence > 0
            return match

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_resolve, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50


class TestConversionLatency:
    """Test that individual conversion latency is under target."""

    @pytest.mark.slow
    def test_single_conversion_under_1ms(self):
        """A single conversion should complete in under 1ms on average."""
        converter = UnitConverter()

        # Warm up
        converter.convert(1, "kg", "t")

        total = 0.0
        n = 100
        for _ in range(n):
            start = time.time()
            converter.convert(1000, "kg", "t")
            total += (time.time() - start) * 1000  # ms

        avg_ms = total / n
        assert avg_ms < 1.0, f"Average conversion time: {avg_ms:.3f}ms (target: <1ms)"
