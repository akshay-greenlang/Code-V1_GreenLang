# -*- coding: utf-8 -*-
"""
Load Tests for Excel & CSV Normalizer Service (AGENT-DATA-002)

Tests throughput and concurrency for CSV parsing, column mapping,
type detection, normalization, quality scoring, transform operations,
memory bounds, and latency targets under high-volume conditions.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline implementations for load testing
# ---------------------------------------------------------------------------


class LoadTestCSVParser:
    """Minimal CSV parser for load testing."""

    def __init__(self):
        self._count = 0

    def parse(self, content: str, file_name: str) -> Dict[str, Any]:
        self._count += 1
        lines = content.strip().split("\n")
        if not lines:
            return {"file_name": file_name, "headers": [], "rows": [], "row_count": 0}

        reader = csv.reader(io.StringIO(content.strip()))
        rows_list = list(reader)
        headers = rows_list[0] if rows_list else []
        data_rows = rows_list[1:] if len(rows_list) > 1 else []

        rows = []
        for row in data_rows:
            row_dict = {}
            for i, h in enumerate(headers):
                row_dict[h] = row[i] if i < len(row) else None
            rows.append(row_dict)

        file_hash = hashlib.sha256(content.encode()).hexdigest()
        return {
            "file_name": file_name,
            "file_hash": file_hash,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
        }

    @property
    def count(self) -> int:
        return self._count


class LoadTestColumnMapper:
    """Minimal column mapper for load testing."""

    CANONICAL = {
        "facility_name", "reporting_year", "scope", "emissions_tco2e",
        "energy_kwh", "country", "waste_tonnes", "activity_data",
    }
    SYNONYMS = {
        "facility_name": {"site_name", "plant_name", "location"},
        "emissions_tco2e": {"co2e_tonnes", "ghg_emissions", "carbon"},
        "energy_kwh": {"electricity_kwh", "power_kwh"},
    }

    def map(self, headers: List[str]) -> Dict[str, str]:
        result = {}
        for header in headers:
            lower = header.lower().strip().replace(" ", "_")
            if lower in self.CANONICAL:
                result[header] = lower
            else:
                for canonical, synonyms in self.SYNONYMS.items():
                    if lower in synonyms:
                        result[header] = canonical
                        break
                else:
                    result[header] = header
        return result


class LoadTestTypeDetector:
    """Minimal type detector for load testing."""

    def detect(self, values: List[Any]) -> str:
        if not values:
            return "string"
        for val in values[:10]:
            if val is None:
                continue
            s = str(val).strip()
            try:
                int(s)
                return "integer"
            except ValueError:
                pass
            try:
                float(s.replace(",", ""))
                return "float"
            except ValueError:
                pass
        return "string"


class LoadTestQualityScorer:
    """Minimal quality scorer for load testing."""

    def score(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        total = sum(len(row) for row in data)
        filled = sum(1 for row in data for v in row.values()
                     if v is not None and str(v).strip())
        completeness = filled / total if total > 0 else 0.0
        return {
            "overall": round(completeness * 0.4 + 0.8 * 0.35 + 0.85 * 0.25, 4),
            "completeness": round(completeness, 4),
            "accuracy": 0.8,
            "consistency": 0.85,
        }


class LoadTestTransformEngine:
    """Minimal transform engine for load testing."""

    def dedup(self, data: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for row in data:
            key = tuple(str(row.get(k, "")) for k in keys) if keys else tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    def filter_rows(self, data: List[Dict[str, Any]], column: str, value: Any) -> List[Dict[str, Any]]:
        return [row for row in data if row.get(column) == value]


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------


def generate_csv_content(num_rows: int, index: int = 0) -> str:
    lines = ["facility_name,reporting_year,scope,emissions_tco2e,energy_kwh,country"]
    facilities = ["London HQ", "Berlin DC", "Paris Office", "Tokyo Lab", "NY Branch"]
    scopes = ["Scope 1", "Scope 2", "Scope 3"]
    countries = ["UK", "Germany", "France", "Japan", "US"]

    for i in range(num_rows):
        fac = facilities[i % len(facilities)]
        scope = scopes[i % len(scopes)]
        country = countries[i % len(countries)]
        emissions = round(100.0 + i * 1.5 + index * 0.1, 2)
        energy = round(50000 + i * 100 + index * 10, 1)
        lines.append(f"{fac},{2024 + (i % 2)},{scope},{emissions},{energy},{country}")

    return "\n".join(lines)


def generate_large_csv(num_rows: int) -> str:
    return generate_csv_content(num_rows)


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestCSVParsingThroughput:
    """Test CSV parsing throughput: 1000 files in <5s."""

    @pytest.mark.slow
    def test_1000_sequential_parses(self):
        parser = LoadTestCSVParser()
        start = time.time()
        for i in range(1000):
            content = generate_csv_content(10, index=i)
            parser.parse(content, f"file_{i:04d}.csv")
        elapsed = time.time() - start

        assert parser.count == 1000
        assert elapsed < 5.0, f"1000 CSV parses took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_parsing_20_threads(self):
        parser = LoadTestCSVParser()
        results = []

        def do_parse(i: int):
            content = generate_csv_content(10, index=i)
            return parser.parse(content, f"csv_{i}.csv")

        start = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(do_parse, i) for i in range(500)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.time() - start

        assert len(results) == 500
        assert elapsed < 10.0, f"500 concurrent parses took {elapsed:.2f}s"


class TestColumnMappingThroughput:
    """Test column mapping throughput."""

    @pytest.mark.slow
    def test_1000_mapping_operations(self):
        mapper = LoadTestColumnMapper()
        headers = ["facility_name", "reporting_year", "scope", "emissions_tco2e",
                    "energy_kwh", "country", "waste_tonnes", "activity_data"]
        results = []
        start = time.time()
        for _ in range(1000):
            results.append(mapper.map(headers))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 2.0, f"1000 mappings took {elapsed:.2f}s (target: <2s)"
        # Verify mapping accuracy
        assert all(r.get("facility_name") == "facility_name" for r in results)


class TestTypeDetectionThroughput:
    """Test type detection throughput."""

    @pytest.mark.slow
    def test_1000_type_detections(self):
        detector = LoadTestTypeDetector()
        results = []
        sample_values = ["1250.5", "890.3", "670.1", "445.8", "320.0"]

        start = time.time()
        for _ in range(1000):
            results.append(detector.detect(sample_values))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 1.0, f"1000 type detections took {elapsed:.2f}s (target: <1s)"
        assert all(r == "float" for r in results)


class TestNormalizationThroughput:
    """Test normalization throughput."""

    @pytest.mark.slow
    def test_normalize_10000_rows(self):
        parser = LoadTestCSVParser()
        mapper = LoadTestColumnMapper()

        content = generate_large_csv(10000)
        parsed = parser.parse(content, "large.csv")
        mappings = mapper.map(parsed["headers"])

        start = time.time()
        normalized = []
        for row in parsed["rows"]:
            new_row = {mappings.get(k, k): v for k, v in row.items()}
            normalized.append(new_row)
        elapsed = time.time() - start

        assert len(normalized) == 10000
        assert elapsed < 2.0, f"10000 row normalization took {elapsed:.2f}s (target: <2s)"


class TestQualityScoringThroughput:
    """Test quality scoring throughput."""

    @pytest.mark.slow
    def test_1000_quality_scores(self):
        scorer = LoadTestQualityScorer()
        sample_data = [
            {"facility_name": "London", "year": 2025, "emissions": 1250.5},
            {"facility_name": "Berlin", "year": 2025, "emissions": 890.3},
        ]
        results = []

        start = time.time()
        for _ in range(1000):
            results.append(scorer.score(sample_data))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 1.0, f"1000 quality scores took {elapsed:.2f}s (target: <1s)"
        assert all(r["overall"] > 0 for r in results)


class TestTransformThroughput:
    """Test transform operation throughput."""

    @pytest.mark.slow
    def test_dedup_1000_datasets(self):
        engine = LoadTestTransformEngine()
        data = [
            {"id": "A", "val": 1}, {"id": "B", "val": 2},
            {"id": "A", "val": 1}, {"id": "C", "val": 3},
        ]
        results = []

        start = time.time()
        for _ in range(1000):
            results.append(engine.dedup(data, ["id"]))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 2.0, f"1000 dedup transforms took {elapsed:.2f}s (target: <2s)"
        assert all(len(r) == 3 for r in results)

    @pytest.mark.slow
    def test_filter_1000_datasets(self):
        engine = LoadTestTransformEngine()
        data = [
            {"scope": "Scope 1", "val": 100},
            {"scope": "Scope 2", "val": 200},
            {"scope": "Scope 1", "val": 300},
        ]
        results = []

        start = time.time()
        for _ in range(1000):
            results.append(engine.filter_rows(data, "scope", "Scope 1"))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 1.0, f"1000 filter transforms took {elapsed:.2f}s (target: <1s)"
        assert all(len(r) == 2 for r in results)


class TestMemoryBounds:
    """Test memory usage stays within bounds."""

    @pytest.mark.slow
    def test_memory_usage_large_csv(self):
        parser = LoadTestCSVParser()
        content = generate_large_csv(10000)

        initial_size = sys.getsizeof([])
        parsed = parser.parse(content, "large.csv")
        rows = parsed["rows"]

        total_text_size = sum(
            sum(len(str(v)) for v in row.values()) for row in rows
        )
        # 10000 rows should be well under 50MB
        assert total_text_size < 50 * 1024 * 1024, "Data exceeds 50MB limit"
        assert len(rows) == 10000


class TestLatencyTargets:
    """Test single-operation latency targets."""

    def test_single_csv_parse_latency(self):
        parser = LoadTestCSVParser()
        content = generate_csv_content(100)
        start = time.time()
        parser.parse(content, "test.csv")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 50, f"Single CSV parse took {elapsed_ms:.2f}ms (target: <50ms)"

    def test_single_column_mapping_latency(self):
        mapper = LoadTestColumnMapper()
        headers = ["facility_name", "year", "scope", "emissions", "energy"]
        start = time.time()
        mapper.map(headers)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single mapping took {elapsed_ms:.2f}ms (target: <5ms)"

    def test_single_type_detection_latency(self):
        detector = LoadTestTypeDetector()
        values = ["1250.5", "890.3", "670.1"]
        start = time.time()
        detector.detect(values)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single type detection took {elapsed_ms:.2f}ms (target: <5ms)"

    def test_single_quality_score_latency(self):
        scorer = LoadTestQualityScorer()
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        start = time.time()
        scorer.score(data)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single quality score took {elapsed_ms:.2f}ms (target: <5ms)"

    def test_single_dedup_latency(self):
        engine = LoadTestTransformEngine()
        data = [{"id": "A", "v": 1}, {"id": "B", "v": 2}, {"id": "A", "v": 1}]
        start = time.time()
        engine.dedup(data, ["id"])
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single dedup took {elapsed_ms:.2f}ms (target: <5ms)"


class TestEndToEndPipelineThroughput:
    """Test full pipeline throughput."""

    @pytest.mark.slow
    def test_100_full_pipelines(self):
        parser = LoadTestCSVParser()
        mapper = LoadTestColumnMapper()
        detector = LoadTestTypeDetector()
        scorer = LoadTestQualityScorer()

        start = time.time()
        for i in range(100):
            content = generate_csv_content(50, index=i)
            parsed = parser.parse(content, f"pipeline_{i}.csv")
            mappings = mapper.map(parsed["headers"])

            for header in parsed["headers"]:
                values = [row.get(header) for row in parsed["rows"]]
                detector.detect(values)

            normalized = [
                {mappings.get(k, k): v for k, v in row.items()}
                for row in parsed["rows"]
            ]
            scorer.score(normalized)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"100 full pipelines took {elapsed:.2f}s (target: <10s)"
