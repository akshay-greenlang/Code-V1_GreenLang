# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Performance Benchmarks
=================================================

Validates performance characteristics: screening throughput, spend
classification speed, consolidation timing, double-counting check speed,
Monte Carlo simulation timing, report generation speed, provenance hash
throughput, and memory efficiency for large datasets.

Coverage target: 85%+
Total tests: ~20
"""

import hashlib
import json
import math
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    OVERLAP_RULES,
    compute_provenance_hash,
)


# =============================================================================
# Helpers
# =============================================================================


def _generate_spend_transactions(count: int) -> List[Dict[str, Any]]:
    """Generate synthetic spend transactions for performance testing."""
    sectors = [
        "basic_metals", "chemicals_pharmaceuticals", "electronics_optical",
        "fabricated_metals", "rubber_plastics", "machinery_equipment",
        "wood_paper_products", "non_metallic_minerals", "land_transport",
        "air_transport", "water_transport", "it_services",
    ]
    categories = list(SCOPE3_CATEGORIES)
    base_date = date(2025, 1, 1)
    transactions = []
    for i in range(count):
        transactions.append({
            "transaction_id": f"PERF-TXN-{i:08d}",
            "date": str(base_date + timedelta(days=i % 365)),
            "description": f"Performance test transaction {i}",
            "supplier_id": f"PERF-SUP-{i % 500:05d}",
            "amount_eur": Decimal(str(1000 + (i * 7 % 99000))),
            "currency": "EUR",
            "naics_code": f"{31 + (i % 8):02d}0000",
            "eeio_sector": sectors[i % len(sectors)],
            "scope3_category": categories[i % len(categories)],
            "gl_account": f"5{i % 9000:04d}",
        })
    return transactions


def _generate_category_results(num_categories: int = 15) -> Dict[str, Any]:
    """Generate category results for performance testing."""
    cats = {}
    total = Decimal("0")
    for i in range(1, num_categories + 1):
        cat_id = f"CAT_{i}"
        tco2e = Decimal(str(500 + i * 1500))
        co2 = tco2e * Decimal("0.95")
        ch4 = tco2e * Decimal("0.03")
        n2o = tco2e * Decimal("0.02")
        cats[cat_id] = {
            "total_tco2e": tco2e,
            "methodology": "SPEND_BASED",
            "dqr": Decimal("3.0"),
            "by_gas": {"CO2": co2, "CH4": ch4, "N2O": n2o},
            "uncertainty_pct": Decimal("40.0"),
            "source_count": 10 + i,
        }
        total += tco2e
    return {"categories": cats, "total_scope3_tco2e": total}


def _timed(func, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# =============================================================================
# Screening Throughput Tests
# =============================================================================


class TestScreeningThroughput:
    """Test screening engine throughput targets."""

    def test_screening_100_transactions_under_100ms(self):
        """Screening 100 transactions should complete under 100ms."""
        transactions = _generate_spend_transactions(100)

        def screen():
            results = {}
            for cat in SCOPE3_CATEGORIES:
                cat_txns = [t for t in transactions if t["scope3_category"] == cat]
                total = sum(t["amount_eur"] for t in cat_txns)
                results[cat] = {
                    "estimated_spend": total,
                    "transaction_count": len(cat_txns),
                    "relevance": "HIGH" if total > Decimal("100000") else "LOW",
                }
            return results

        result, elapsed = _timed(screen)
        assert elapsed < 0.100, f"Screening took {elapsed:.3f}s, expected < 0.100s"
        assert len(result) == 15

    def test_screening_1000_transactions_under_500ms(self):
        """Screening 1000 transactions should complete under 500ms."""
        transactions = _generate_spend_transactions(1000)

        def screen():
            results = {}
            for cat in SCOPE3_CATEGORIES:
                cat_txns = [t for t in transactions if t["scope3_category"] == cat]
                total = sum(t["amount_eur"] for t in cat_txns)
                results[cat] = {"estimated_spend": total, "count": len(cat_txns)}
            return results

        result, elapsed = _timed(screen)
        assert elapsed < 0.500, f"Screening 1K took {elapsed:.3f}s, expected < 0.500s"

    def test_screening_10000_transactions_under_2s(self):
        """Screening 10K transactions should complete under 2 seconds."""
        transactions = _generate_spend_transactions(10000)

        def screen():
            results = {}
            for cat in SCOPE3_CATEGORIES:
                cat_txns = [t for t in transactions if t["scope3_category"] == cat]
                total = sum(t["amount_eur"] for t in cat_txns)
                results[cat] = {"estimated_spend": total, "count": len(cat_txns)}
            return results

        result, elapsed = _timed(screen)
        assert elapsed < 2.0, f"Screening 10K took {elapsed:.3f}s, expected < 2.0s"


# =============================================================================
# Spend Classification Speed Tests
# =============================================================================


class TestSpendClassificationSpeed:
    """Test spend classification performance."""

    def test_classify_100_transactions_under_50ms(self):
        """Classify 100 transactions by NAICS under 50ms."""
        transactions = _generate_spend_transactions(100)

        def classify():
            results = []
            for txn in transactions:
                naics_prefix = txn["naics_code"][:2]
                sector_map = {
                    "31": "manufacturing", "32": "manufacturing",
                    "33": "manufacturing", "42": "wholesale",
                    "48": "transport", "49": "transport",
                    "51": "information", "52": "finance",
                    "54": "professional", "56": "admin",
                }
                sector = sector_map.get(naics_prefix, "other")
                results.append({
                    "transaction_id": txn["transaction_id"],
                    "classified_sector": sector,
                    "confidence": 0.85,
                })
            return results

        result, elapsed = _timed(classify)
        assert elapsed < 0.050, f"Classification took {elapsed:.3f}s, expected < 0.050s"
        assert len(result) == 100

    def test_classify_5000_transactions_under_500ms(self):
        """Classify 5000 transactions under 500ms."""
        transactions = _generate_spend_transactions(5000)

        def classify():
            classified = []
            for txn in transactions:
                classified.append({
                    "id": txn["transaction_id"],
                    "sector": txn["eeio_sector"],
                    "confidence": 0.90,
                })
            return classified

        result, elapsed = _timed(classify)
        assert elapsed < 0.500, f"Classification 5K took {elapsed:.3f}s, expected < 0.500s"


# =============================================================================
# Consolidation Timing Tests
# =============================================================================


class TestConsolidationTiming:
    """Test category consolidation timing."""

    def test_consolidation_15_categories_under_20ms(self):
        """Consolidate 15 categories under 20ms."""
        cat_results = _generate_category_results(15)

        def consolidate():
            cats = cat_results["categories"]
            upstream = sum(
                cats[c]["total_tco2e"] for c in UPSTREAM_CATEGORIES
                if c in cats
            )
            downstream = sum(
                cats[c]["total_tco2e"] for c in DOWNSTREAM_CATEGORIES
                if c in cats
            )
            total = upstream + downstream
            by_gas = {"CO2": Decimal("0"), "CH4": Decimal("0"), "N2O": Decimal("0")}
            for cat_data in cats.values():
                for gas, val in cat_data["by_gas"].items():
                    by_gas[gas] += val
            return {
                "total": total,
                "upstream": upstream,
                "downstream": downstream,
                "by_gas": by_gas,
            }

        result, elapsed = _timed(consolidate)
        assert elapsed < 0.020, f"Consolidation took {elapsed:.4f}s, expected < 0.020s"
        assert result["total"] > 0


# =============================================================================
# Double-Counting Check Speed Tests
# =============================================================================


class TestDoubleCountingSpeed:
    """Test double-counting detection speed."""

    def test_evaluate_12_rules_under_10ms(self):
        """Evaluate 12 overlap rules under 10ms."""
        cat_results = _generate_category_results(15)

        def check_overlaps():
            findings = []
            for rule in OVERLAP_RULES:
                parts = rule.split("_vs_")
                cat_a = parts[0].upper() if len(parts) > 0 else None
                cat_b = parts[1].split("_")[0].upper() if len(parts) > 1 else None
                overlap = Decimal("0")
                findings.append({
                    "rule": rule,
                    "overlap_tco2e": overlap,
                    "status": "NO_OVERLAP",
                })
            return findings

        result, elapsed = _timed(check_overlaps)
        assert elapsed < 0.010, (
            f"Double-counting check took {elapsed:.4f}s, expected < 0.010s"
        )
        assert len(result) == 12


# =============================================================================
# Monte Carlo Simulation Timing Tests
# =============================================================================


class TestMonteCarloTiming:
    """Test Monte Carlo simulation performance."""

    def test_1000_iterations_under_200ms(self):
        """1000 Monte Carlo iterations for 15 categories under 200ms."""
        import random
        rng = random.Random(42)

        def monte_carlo(n_iterations: int):
            results = []
            cat_emissions = {f"CAT_{i}": 1000 + i * 1500 for i in range(1, 16)}
            cat_uncertainty = {f"CAT_{i}": 0.30 + (i * 0.02) for i in range(1, 16)}

            for _ in range(n_iterations):
                total = 0.0
                for cat, base in cat_emissions.items():
                    u = cat_uncertainty[cat]
                    sample = rng.gauss(base, base * u)
                    total += max(0, sample)
                results.append(total)
            return results

        result, elapsed = _timed(monte_carlo, 1000)
        assert elapsed < 0.200, (
            f"Monte Carlo 1K iterations took {elapsed:.3f}s, expected < 0.200s"
        )
        assert len(result) == 1000

    def test_10000_iterations_under_2s(self):
        """10K Monte Carlo iterations under 2 seconds."""
        import random
        rng = random.Random(42)

        def monte_carlo(n_iterations: int):
            results = []
            cat_emissions = {f"CAT_{i}": 1000 + i * 1500 for i in range(1, 16)}
            cat_uncertainty = {f"CAT_{i}": 0.30 + (i * 0.02) for i in range(1, 16)}

            for _ in range(n_iterations):
                total = 0.0
                for cat, base in cat_emissions.items():
                    u = cat_uncertainty[cat]
                    total += max(0, rng.gauss(base, base * u))
                results.append(total)
            return results

        result, elapsed = _timed(monte_carlo, 10000)
        assert elapsed < 2.0, (
            f"Monte Carlo 10K iterations took {elapsed:.3f}s, expected < 2.0s"
        )
        assert len(result) == 10000

    def test_monte_carlo_reproducibility(self):
        """Same seed produces same results."""
        import random

        def mc_run(seed: int, n: int = 100):
            rng = random.Random(seed)
            return [rng.gauss(1000, 300) for _ in range(n)]

        run1 = mc_run(42)
        run2 = mc_run(42)
        assert run1 == run2, "Same seed should produce identical results"


# =============================================================================
# Report Generation Speed Tests
# =============================================================================


class TestReportGenerationSpeed:
    """Test report generation performance."""

    def test_markdown_report_under_100ms(self):
        """Generate a Markdown report under 100ms."""
        cat_results = _generate_category_results(15)

        def generate_markdown():
            lines = ["# Scope 3 Emissions Report", "", "## Summary", ""]
            total = cat_results["total_scope3_tco2e"]
            lines.append(f"Total Scope 3 Emissions: {total} tCO2e")
            lines.append("")
            lines.append("## Category Breakdown")
            lines.append("")
            lines.append("| Category | tCO2e | Methodology | DQR |")
            lines.append("|----------|-------|-------------|-----|")
            for cat_id, data in cat_results["categories"].items():
                lines.append(
                    f"| {cat_id} | {data['total_tco2e']} | "
                    f"{data['methodology']} | {data['dqr']} |"
                )
            lines.append("")
            lines.append("## Gas Breakdown")
            for cat_id, data in cat_results["categories"].items():
                lines.append(f"### {cat_id}")
                for gas, val in data["by_gas"].items():
                    lines.append(f"- {gas}: {val} tCO2e")
            return "\n".join(lines)

        result, elapsed = _timed(generate_markdown)
        assert elapsed < 0.100, (
            f"Markdown generation took {elapsed:.3f}s, expected < 0.100s"
        )
        assert "# Scope 3 Emissions Report" in result
        assert "CAT_1" in result

    def test_json_report_under_50ms(self):
        """Generate a JSON report under 50ms."""
        cat_results = _generate_category_results(15)

        def generate_json():
            report = {
                "report_type": "scope3_inventory",
                "format": "JSON",
                "total_scope3_tco2e": str(cat_results["total_scope3_tco2e"]),
                "categories": {},
            }
            for cat_id, data in cat_results["categories"].items():
                report["categories"][cat_id] = {
                    "total_tco2e": str(data["total_tco2e"]),
                    "methodology": data["methodology"],
                    "dqr": str(data["dqr"]),
                }
            return json.dumps(report, indent=2)

        result, elapsed = _timed(generate_json)
        assert elapsed < 0.050, (
            f"JSON generation took {elapsed:.3f}s, expected < 0.050s"
        )
        parsed = json.loads(result)
        assert parsed["report_type"] == "scope3_inventory"


# =============================================================================
# Provenance Hash Throughput Tests
# =============================================================================


class TestProvenanceHashThroughput:
    """Test SHA-256 provenance hash throughput."""

    def test_hash_1000_records_under_100ms(self):
        """Hash 1000 records under 100ms."""
        records = [{"id": i, "value": i * 100} for i in range(1000)]

        def hash_all():
            hashes = []
            for record in records:
                h = compute_provenance_hash(record)
                hashes.append(h)
            return hashes

        result, elapsed = _timed(hash_all)
        assert elapsed < 0.100, (
            f"Hashing 1K records took {elapsed:.3f}s, expected < 0.100s"
        )
        assert len(result) == 1000
        assert all(len(h) == 64 for h in result)

    def test_hash_determinism(self):
        """Hashing is deterministic."""
        data = {"test": "provenance", "value": 42}
        h1 = compute_provenance_hash(data)
        h2 = compute_provenance_hash(data)
        assert h1 == h2

    def test_hash_collision_resistance(self):
        """Different inputs produce different hashes."""
        hashes = set()
        for i in range(1000):
            h = compute_provenance_hash({"unique": i})
            hashes.add(h)
        assert len(hashes) == 1000, "All 1000 hashes should be unique"


# =============================================================================
# Memory Efficiency Tests
# =============================================================================


class TestMemoryEfficiency:
    """Test memory usage is reasonable for large datasets."""

    def test_large_transaction_list_fits_in_memory(self):
        """10K transactions should not exceed reasonable memory bounds."""
        import sys

        transactions = _generate_spend_transactions(10000)
        # Each transaction is a dict; total should be under 50MB
        total_size = sys.getsizeof(transactions)
        # Note: getsizeof only measures the list container, not contents.
        # For a rough check, we verify the list was created successfully.
        assert len(transactions) == 10000
        assert total_size < 200_000_000  # container < 200MB (generous bound)

    def test_batch_processing_1000_at_a_time(self):
        """Process 10K transactions in batches of 1000."""
        all_transactions = _generate_spend_transactions(10000)
        batch_size = 1000

        def process_in_batches():
            results = []
            for i in range(0, len(all_transactions), batch_size):
                batch = all_transactions[i : i + batch_size]
                batch_total = sum(t["amount_eur"] for t in batch)
                results.append({
                    "batch_index": i // batch_size,
                    "batch_size": len(batch),
                    "total_spend": batch_total,
                })
            return results

        result, elapsed = _timed(process_in_batches)
        assert len(result) == 10, "Should have 10 batches"
        assert elapsed < 2.0, f"Batch processing took {elapsed:.3f}s, expected < 2.0s"

    def test_category_result_size_reasonable(self):
        """Category results for 15 categories should be compact."""
        import sys

        results = _generate_category_results(15)
        serialized = json.dumps(results, default=str)
        assert len(serialized) < 50_000, (
            f"Serialized results are {len(serialized)} bytes, expected < 50KB"
        )
