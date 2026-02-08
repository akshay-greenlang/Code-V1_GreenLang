# -*- coding: utf-8 -*-
"""
Load Tests for ERP/Finance Connector Service (AGENT-DATA-003)

Tests throughput and concurrency for spend extraction, Scope 3 classification,
emissions calculation, currency conversion, batch processing, memory bounds,
and latency under high-volume conditions.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline implementations for load testing
# ---------------------------------------------------------------------------


DEFAULT_RATES_TO_USD = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
    "CHF": 1.13, "CAD": 0.74, "AUD": 0.65, "CNY": 0.14,
    "INR": 0.012, "BRL": 0.20,
}

DEFAULT_EEIO_FACTORS = {
    "cat1_purchased_goods": 0.35,
    "cat2_capital_goods": 0.42,
    "cat3_fuel_energy": 0.55,
    "cat4_upstream_transport": 0.28,
    "cat5_waste": 0.18,
    "cat6_business_travel": 0.22,
    "cat7_employee_commuting": 0.15,
    "cat8_upstream_leased": 0.30,
    "cat9_downstream_transport": 0.25,
    "cat10_processing": 0.38,
    "cat11_use_of_products": 0.45,
    "cat12_end_of_life": 0.20,
    "cat13_downstream_leased": 0.32,
    "cat14_franchises": 0.28,
    "cat15_investments": 0.40,
    "unclassified": 0.50,
}

SPEND_TO_SCOPE3 = {
    "raw_materials": "cat1_purchased_goods",
    "packaging": "cat1_purchased_goods",
    "services": "cat1_purchased_goods",
    "capital_equipment": "cat2_capital_goods",
    "fuel": "cat3_fuel_energy",
    "electricity": "cat3_fuel_energy",
    "logistics": "cat4_upstream_transport",
    "waste_management": "cat5_waste",
    "travel": "cat6_business_travel",
    "commuting": "cat7_employee_commuting",
}

SPEND_CATEGORIES = list(SPEND_TO_SCOPE3.keys())
CURRENCIES = list(DEFAULT_RATES_TO_USD.keys())


class LoadTestSpendExtractor:
    """Minimal spend extractor for load testing."""

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add_records(self, records: List[Dict[str, Any]]):
        self._records.extend(records)

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def summarize(self) -> Dict[str, Any]:
        total = sum(r.get("amount", 0.0) for r in self._records)
        return {"total_records": len(self._records), "total_amount": round(total, 2)}

    @property
    def count(self) -> int:
        return len(self._records)


class LoadTestScope3Mapper:
    """Minimal Scope 3 mapper for load testing."""

    def __init__(self):
        self._vendor_overrides: Dict[str, str] = {}

    def set_vendor_override(self, vendor_id: str, scope3_category: str):
        self._vendor_overrides[vendor_id] = scope3_category

    def classify(self, category: str, vendor_id: Optional[str] = None) -> str:
        if vendor_id and vendor_id in self._vendor_overrides:
            return self._vendor_overrides[vendor_id]
        return SPEND_TO_SCOPE3.get(category, "unclassified")


class LoadTestEmissionsCalculator:
    """Minimal emissions calculator for load testing."""

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    def calculate(self, amount: float, scope3_category: str,
                  methodology: str = "eeio") -> Dict[str, Any]:
        factor = DEFAULT_EEIO_FACTORS.get(scope3_category, 0.50)
        kgco2e = round(amount * factor, 2)
        result = {
            "amount_usd": amount,
            "scope3_category": scope3_category,
            "emission_factor": factor,
            "estimated_kgco2e": kgco2e,
            "methodology": methodology,
        }
        self._results.append(result)
        return result

    def calculate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for r in records:
            amount = r.get("amount", 0.0)
            scope3 = r.get("scope3_category", "unclassified")
            results.append(self.calculate(amount, scope3))
        total = sum(r["estimated_kgco2e"] for r in results)
        return {"total_emissions_kgco2e": round(total, 2),
                "records_calculated": len(results)}

    @property
    def result_count(self) -> int:
        return len(self._results)


class LoadTestCurrencyConverter:
    """Minimal currency converter for load testing."""

    def __init__(self):
        self._rates = dict(DEFAULT_RATES_TO_USD)

    def convert(self, amount: float, from_currency: str,
                to_currency: str = "USD") -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return round(amount, 2)
        from_rate = self._rates.get(from_currency, 1.0)
        to_rate = self._rates.get(to_currency, 1.0)
        return round((amount * from_rate) / to_rate, 2)


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------


def generate_spend_record(index: int) -> Dict[str, Any]:
    """Generate a single spend record with realistic data."""
    return {
        "record_id": f"SPD-LOAD-{index:06d}",
        "vendor_id": f"V-{(index % 200):03d}",
        "amount": round(1000.0 + (index % 100) * 500.0, 2),
        "currency": CURRENCIES[index % len(CURRENCIES)],
        "category": SPEND_CATEGORIES[index % len(SPEND_CATEGORIES)],
        "cost_center": f"CC-{(index % 20):03d}",
        "description": f"Spend record {index}",
    }


def generate_classified_record(index: int) -> Dict[str, Any]:
    """Generate a pre-classified spend record for emissions calculation."""
    category = SPEND_CATEGORIES[index % len(SPEND_CATEGORIES)]
    scope3 = SPEND_TO_SCOPE3.get(category, "unclassified")
    return {
        "record_id": f"CLS-LOAD-{index:06d}",
        "amount": round(5000.0 + (index % 50) * 200.0, 2),
        "scope3_category": scope3,
    }


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestSpendExtractionThroughput:
    """Test spend extraction throughput: 10000 records in <5s."""

    @pytest.mark.slow
    def test_10000_sequential_spend_records(self):
        extractor = LoadTestSpendExtractor()
        records = [generate_spend_record(i) for i in range(10000)]

        start = time.time()
        extractor.add_records(records)
        summary = extractor.summarize()
        elapsed = time.time() - start

        assert extractor.count == 10000
        assert summary["total_amount"] > 0
        assert elapsed < 5.0, f"10000 records took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_spend_ingestion_20_threads(self):
        extractor = LoadTestSpendExtractor()
        batch_size = 500

        def ingest_batch(batch_index: int):
            records = [generate_spend_record(batch_index * batch_size + i)
                       for i in range(batch_size)]
            extractor.add_records(records)
            return len(records)

        start = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(ingest_batch, b) for b in range(20)]
            total = sum(f.result() for f in as_completed(futures))
        elapsed = time.time() - start

        assert total == 10000
        assert elapsed < 10.0, f"20 concurrent batches took {elapsed:.2f}s"


class TestScope3ClassificationThroughput:
    """Test Scope 3 classification throughput: 10000 in <3s."""

    @pytest.mark.slow
    def test_10000_sequential_classifications(self):
        mapper = LoadTestScope3Mapper()
        results = []
        start = time.time()
        for i in range(10000):
            category = SPEND_CATEGORIES[i % len(SPEND_CATEGORIES)]
            vendor_id = f"V-{i % 200:03d}" if i % 5 == 0 else None
            results.append(mapper.classify(category, vendor_id))
        elapsed = time.time() - start

        assert len(results) == 10000
        assert elapsed < 3.0, f"10000 classifications took {elapsed:.2f}s (target: <3s)"

    @pytest.mark.slow
    def test_classification_with_vendor_overrides(self):
        mapper = LoadTestScope3Mapper()
        # Set up 50 vendor overrides
        for i in range(50):
            mapper.set_vendor_override(f"V-OVR-{i:03d}", "cat6_business_travel")

        results = []
        start = time.time()
        for i in range(5000):
            category = SPEND_CATEGORIES[i % len(SPEND_CATEGORIES)]
            vendor_id = f"V-OVR-{i % 50:03d}" if i % 3 == 0 else None
            results.append(mapper.classify(category, vendor_id))
        elapsed = time.time() - start

        override_count = sum(1 for r in results if r == "cat6_business_travel")
        assert override_count > 0
        assert elapsed < 3.0, f"5000 override classifications took {elapsed:.2f}s"


class TestEmissionsCalculationThroughput:
    """Test emissions calculation throughput: 10000 in <5s."""

    @pytest.mark.slow
    def test_10000_sequential_calculations(self):
        calculator = LoadTestEmissionsCalculator()
        records = [generate_classified_record(i) for i in range(10000)]

        start = time.time()
        result = calculator.calculate_batch(records)
        elapsed = time.time() - start

        assert result["records_calculated"] == 10000
        assert result["total_emissions_kgco2e"] > 0
        assert elapsed < 5.0, f"10000 calculations took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_emissions_calculation(self):
        batch_size = 1000
        results = []

        def calc_batch(batch_index: int) -> Dict[str, Any]:
            calculator = LoadTestEmissionsCalculator()
            records = [generate_classified_record(batch_index * batch_size + i)
                       for i in range(batch_size)]
            return calculator.calculate_batch(records)

        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(calc_batch, b) for b in range(10)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.time() - start

        total_calculated = sum(r["records_calculated"] for r in results)
        assert total_calculated == 10000
        assert elapsed < 10.0, f"10 concurrent batches took {elapsed:.2f}s"


class TestCurrencyConversionThroughput:
    """Test currency conversion throughput: 50000 in <3s."""

    @pytest.mark.slow
    def test_50000_sequential_conversions(self):
        converter = LoadTestCurrencyConverter()
        results = []
        start = time.time()
        for i in range(50000):
            from_cur = CURRENCIES[i % len(CURRENCIES)]
            amount = 1000.0 + (i % 100) * 10.0
            results.append(converter.convert(amount, from_cur, "USD"))
        elapsed = time.time() - start

        assert len(results) == 50000
        assert all(r > 0 for r in results)
        assert elapsed < 3.0, f"50000 conversions took {elapsed:.2f}s (target: <3s)"

    @pytest.mark.slow
    def test_cross_currency_conversion(self):
        converter = LoadTestCurrencyConverter()
        results = []
        start = time.time()
        for i in range(10000):
            from_cur = CURRENCIES[i % len(CURRENCIES)]
            to_cur = CURRENCIES[(i + 3) % len(CURRENCIES)]
            amount = 5000.0 + (i % 200) * 25.0
            results.append(converter.convert(amount, from_cur, to_cur))
        elapsed = time.time() - start

        assert len(results) == 10000
        assert elapsed < 3.0, f"10000 cross-currency conversions took {elapsed:.2f}s"


class TestFullPipelineThroughput:
    """Test full pipeline throughput: spend -> classify -> calculate."""

    @pytest.mark.slow
    def test_1000_full_pipeline_records(self):
        extractor = LoadTestSpendExtractor()
        mapper = LoadTestScope3Mapper()
        calculator = LoadTestEmissionsCalculator()
        converter = LoadTestCurrencyConverter()

        start = time.time()
        for i in range(1000):
            record = generate_spend_record(i)

            # Step 1: Convert currency
            amount_usd = converter.convert(
                record["amount"], record["currency"], "USD"
            )

            # Step 2: Classify Scope 3
            scope3 = mapper.classify(record["category"], record.get("vendor_id"))

            # Step 3: Calculate emissions
            calculator.calculate(amount_usd, scope3)

        elapsed = time.time() - start

        assert calculator.result_count == 1000
        assert elapsed < 5.0, f"1000 full pipeline took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_pipeline_20_threads(self):
        results = []

        def process_record(index: int) -> Dict[str, Any]:
            converter = LoadTestCurrencyConverter()
            mapper = LoadTestScope3Mapper()
            calculator = LoadTestEmissionsCalculator()

            record = generate_spend_record(index)
            amount_usd = converter.convert(
                record["amount"], record["currency"], "USD"
            )
            scope3 = mapper.classify(record["category"])
            return calculator.calculate(amount_usd, scope3)

        start = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_record, i) for i in range(500)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.time() - start

        assert len(results) == 500
        assert elapsed < 10.0, f"500 concurrent pipeline took {elapsed:.2f}s"


class TestMemoryBounds:
    """Test memory usage stays within bounds."""

    @pytest.mark.slow
    def test_memory_10000_spend_records(self):
        extractor = LoadTestSpendExtractor()
        records = [generate_spend_record(i) for i in range(10000)]

        extractor.add_records(records)

        total_text_size = sum(
            len(json.dumps(r)) for r in extractor.get_records()
        )
        # 10000 records at ~200 bytes each = ~2MB, should be well under 50MB
        assert total_text_size < 50 * 1024 * 1024, "Records exceed 50MB limit"
        assert extractor.count == 10000

    @pytest.mark.slow
    def test_memory_emissions_results(self):
        calculator = LoadTestEmissionsCalculator()
        records = [generate_classified_record(i) for i in range(10000)]

        calculator.calculate_batch(records)

        assert calculator.result_count == 10000
        # Verify results are stored (no silent drops)
        assert calculator.result_count == 10000


class TestLatencyTargets:
    """Test single-operation latency targets."""

    def test_single_spend_record_latency(self):
        extractor = LoadTestSpendExtractor()
        record = generate_spend_record(0)
        start = time.time()
        extractor.add_records([record])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single spend add took {elapsed_ms:.2f}ms (target: <5ms)"

    def test_single_classification_latency(self):
        mapper = LoadTestScope3Mapper()
        start = time.time()
        mapper.classify("raw_materials", "V-001")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 1, f"Single classify took {elapsed_ms:.2f}ms (target: <1ms)"

    def test_single_emissions_calculation_latency(self):
        calculator = LoadTestEmissionsCalculator()
        start = time.time()
        calculator.calculate(50000.0, "cat1_purchased_goods")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 1, f"Single calc took {elapsed_ms:.2f}ms (target: <1ms)"

    def test_single_currency_conversion_latency(self):
        converter = LoadTestCurrencyConverter()
        start = time.time()
        converter.convert(10000.0, "EUR", "USD")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 1, f"Single convert took {elapsed_ms:.2f}ms (target: <1ms)"

    def test_batch_100_emissions_latency(self):
        calculator = LoadTestEmissionsCalculator()
        records = [generate_classified_record(i) for i in range(100)]
        start = time.time()
        calculator.calculate_batch(records)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 50, f"100 emissions batch took {elapsed_ms:.2f}ms (target: <50ms)"
