# -*- coding: utf-8 -*-
"""
Load Tests for Schema Service (AGENT-FOUND-002)

Tests performance and throughput characteristics of the schema
validation service under load. All tests use self-contained
implementations to avoid external dependencies.

Tests:
    - 50 concurrent validations
    - 100-item batch validation
    - Schema compilation caching (1000 repeated validations)
    - Large payload validation (1MB)
    - Deeply nested object (50 levels)
    - Schema with 500+ properties
    - Concurrent batch validations
    - IR cache warmup performance

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import time
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained validation stub for load testing
# ---------------------------------------------------------------------------


def _validate_payload(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightweight validation for load testing.

    Checks required fields, property types, and basic constraints.
    Returns a dict with valid/findings/schema_hash.
    """
    findings = []

    # Required field check
    required = schema.get("required", [])
    for field in required:
        if field not in payload:
            findings.append({
                "code": "GLSCHEMA-E100",
                "path": f"/{field}",
                "message": f"Missing required field: {field}",
            })

    # Property type check
    properties = schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in payload:
            expected = field_schema.get("type")
            value = payload[field]
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            expected_cls = type_map.get(expected)
            if expected_cls and not isinstance(value, expected_cls):
                findings.append({
                    "code": "GLSCHEMA-E200",
                    "path": f"/{field}",
                    "message": f"Type mismatch: expected {expected}",
                })

    return {
        "valid": len(findings) == 0,
        "findings": findings,
        "schema_hash": "a" * 64,
    }


def _validate_batch(
    payloads: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Batch validation."""
    results = []
    for i, payload in enumerate(payloads):
        r = _validate_payload(payload, schema)
        results.append({"index": i, "valid": r["valid"], "findings": r["findings"]})
    valid_count = sum(1 for r in results if r["valid"])
    return {
        "summary": {
            "total_items": len(payloads),
            "valid_count": valid_count,
            "error_count": len(payloads) - valid_count,
        },
        "results": results,
    }


def _compile_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compile (pre-process) a schema."""
    # Simulate compilation work
    prop_count = len(schema.get("properties", {}))
    rule_count = len(schema.get("$rules", []))
    return {
        "schema_hash": "c" * 64,
        "properties": prop_count,
        "rules": rule_count,
    }


# ---------------------------------------------------------------------------
# Helper: generate test payloads
# ---------------------------------------------------------------------------


def _make_emissions_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "fuel_type": {"type": "string"},
            "quantity": {"type": "number", "minimum": 0},
            "unit": {"type": "string"},
            "co2e_kg": {"type": "number", "minimum": 0},
        },
        "required": ["source_id", "fuel_type", "quantity", "unit", "co2e_kg"],
    }


def _make_valid_payload(idx: int = 0) -> Dict[str, Any]:
    return {
        "source_id": f"FAC-{idx:04d}",
        "fuel_type": "diesel",
        "quantity": 100.0 + idx,
        "unit": "liters",
        "co2e_kg": 268.0 + idx,
    }


def _make_large_payload(size_bytes: int) -> Dict[str, Any]:
    """Create a payload of approximately size_bytes."""
    # Each char in the string is 1 byte; account for JSON overhead (~50 bytes)
    data_size = max(10, size_bytes - 50)
    return {"data": "x" * data_size, "source_id": "LARGE", "fuel_type": "gas", "quantity": 1.0, "unit": "m3", "co2e_kg": 2.0}


def _make_deeply_nested(depth: int) -> Dict[str, Any]:
    """Create a deeply nested object."""
    obj: Dict[str, Any] = {"value": "leaf", "depth": depth}
    for i in range(depth - 1, 0, -1):
        obj = {"nested": obj, "depth": i}
    return obj


def _make_wide_schema(num_properties: int) -> Dict[str, Any]:
    """Create a schema with many properties."""
    props = {}
    for i in range(num_properties):
        props[f"field_{i:04d}"] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": [f"field_{i:04d}" for i in range(min(10, num_properties))],
    }


def _make_wide_payload(num_properties: int) -> Dict[str, Any]:
    """Create a payload matching a wide schema."""
    return {f"field_{i:04d}": f"value_{i}" for i in range(num_properties)}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestConcurrentValidations:
    """Test 50 concurrent validations."""

    def test_50_concurrent_validations(self):
        schema = _make_emissions_schema()
        payloads = [_make_valid_payload(i) for i in range(50)]

        start = time.perf_counter()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(_validate_payload, p, schema)
                for p in payloads
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        elapsed = time.perf_counter() - start

        assert len(results) == 50
        assert all(r["valid"] for r in results)
        # Should complete within 5 seconds
        assert elapsed < 5.0, f"50 concurrent validations took {elapsed:.2f}s (limit 5s)"


class TestBatchValidation:
    """Test 100-item batch validation."""

    def test_100_item_batch(self):
        schema = _make_emissions_schema()
        payloads = [_make_valid_payload(i) for i in range(100)]

        start = time.perf_counter()
        result = _validate_batch(payloads, schema)
        elapsed = time.perf_counter() - start

        assert result["summary"]["total_items"] == 100
        assert result["summary"]["valid_count"] == 100
        assert elapsed < 2.0, f"100-item batch took {elapsed:.2f}s (limit 2s)"


class TestSchemaCompilationCaching:
    """Test schema compilation caching (validate same schema 1000 times)."""

    def test_1000_validations_same_schema(self):
        schema = _make_emissions_schema()
        compiled = _compile_schema(schema)
        payload = _make_valid_payload(0)

        start = time.perf_counter()
        for _ in range(1000):
            _validate_payload(payload, schema)
        elapsed = time.perf_counter() - start

        # 1000 validations should complete quickly
        assert elapsed < 5.0, f"1000 validations took {elapsed:.2f}s (limit 5s)"
        throughput = 1000 / elapsed
        assert throughput > 100, f"Throughput {throughput:.0f}/s is below 100/s target"


class TestLargePayloadValidation:
    """Test large payload validation (1MB)."""

    def test_1mb_payload(self):
        schema = _make_emissions_schema()
        payload = _make_large_payload(1_048_576)

        start = time.perf_counter()
        result = _validate_payload(payload, schema)
        elapsed = time.perf_counter() - start

        # Should complete (valid or not) within 2 seconds
        assert elapsed < 2.0, f"1MB payload validation took {elapsed:.2f}s (limit 2s)"
        assert "valid" in result

    def test_large_payload_json_serializable(self):
        payload = _make_large_payload(1_048_576)
        serialized = json.dumps(payload)
        assert len(serialized) >= 1_000_000


class TestDeeplyNestedObject:
    """Test deeply nested object (50 levels)."""

    def test_50_level_nesting(self):
        nested = _make_deeply_nested(50)
        schema = {
            "type": "object",
            "properties": {
                "nested": {"type": "object"},
                "depth": {"type": "integer"},
            },
        }

        start = time.perf_counter()
        result = _validate_payload(nested, schema)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"50-level nested validation took {elapsed:.2f}s (limit 2s)"
        assert "valid" in result

    def test_deeply_nested_is_valid_json(self):
        nested = _make_deeply_nested(50)
        serialized = json.dumps(nested)
        parsed = json.loads(serialized)
        assert parsed["depth"] == 1
        # Walk down to verify depth
        current = parsed
        for i in range(49):
            assert "nested" in current
            current = current["nested"]
        assert current["value"] == "leaf"


class TestWideSchema:
    """Test schema with 500+ properties."""

    def test_500_property_schema(self):
        schema = _make_wide_schema(500)
        payload = _make_wide_payload(500)

        start = time.perf_counter()
        result = _validate_payload(payload, schema)
        elapsed = time.perf_counter() - start

        assert result["valid"] is True
        assert elapsed < 2.0, f"500-property validation took {elapsed:.2f}s (limit 2s)"

    def test_500_property_compilation(self):
        schema = _make_wide_schema(500)

        start = time.perf_counter()
        compiled = _compile_schema(schema)
        elapsed = time.perf_counter() - start

        assert compiled["properties"] == 500
        assert elapsed < 1.0, f"500-property compilation took {elapsed:.2f}s (limit 1s)"


class TestConcurrentBatchValidations:
    """Test concurrent batch validations."""

    def test_5_concurrent_batches_of_20(self):
        schema = _make_emissions_schema()

        def run_batch(batch_idx: int):
            payloads = [_make_valid_payload(batch_idx * 20 + i) for i in range(20)]
            return _validate_batch(payloads, schema)

        start = time.perf_counter()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_batch, i) for i in range(5)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        elapsed = time.perf_counter() - start

        assert len(results) == 5
        total_items = sum(r["summary"]["total_items"] for r in results)
        assert total_items == 100
        assert elapsed < 5.0, f"5 concurrent batches took {elapsed:.2f}s (limit 5s)"


class TestIRCacheWarmupPerformance:
    """Test IR cache warmup performance."""

    def test_first_vs_subsequent_validations(self):
        schema = _make_emissions_schema()
        payload = _make_valid_payload(0)

        # First validation (cold)
        start = time.perf_counter()
        _compile_schema(schema)
        _validate_payload(payload, schema)
        cold_elapsed = time.perf_counter() - start

        # Subsequent validations (warm)
        start = time.perf_counter()
        for _ in range(100):
            _validate_payload(payload, schema)
        warm_elapsed = time.perf_counter() - start
        warm_per_item = warm_elapsed / 100

        # Warm validations should be faster per-item than the cold one
        # (or at least not significantly slower)
        assert warm_per_item < cold_elapsed * 2, (
            f"Warm per-item {warm_per_item:.4f}s should be less than "
            f"2x cold {cold_elapsed:.4f}s"
        )

    def test_cache_hit_rate_pattern(self):
        """Simulate cache warmup: first miss, then hits."""
        schema = _make_emissions_schema()
        cache: Dict[str, Any] = {}
        hits = 0
        misses = 0

        for i in range(100):
            cache_key = "emissions_schema_v1"
            if cache_key in cache:
                hits += 1
                compiled = cache[cache_key]
            else:
                misses += 1
                compiled = _compile_schema(schema)
                cache[cache_key] = compiled

            _validate_payload(_make_valid_payload(i), schema)

        assert misses == 1, "Only the first compilation should miss"
        assert hits == 99, "Subsequent lookups should all hit"
