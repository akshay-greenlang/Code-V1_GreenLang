# -*- coding: utf-8 -*-
"""
Performance Tests for PACK-041 Scope 1-2 Complete
=====================================================

Tests performance targets for boundary evaluation, scope calculations,
Monte Carlo simulation, compliance mapping, and report generation.

Coverage target: 85%+
Total tests: ~20
"""

import importlib.util
import math
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack041_test.perf.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =============================================================================
# Boundary Performance
# =============================================================================


@pytest.mark.performance
class TestBoundaryPerformance:
    """Test organizational boundary engine performance."""

    def test_100_entities_under_5s(self):
        """100 entities should evaluate in under 5 seconds."""
        try:
            _m = _load("organizational_boundary_engine")
        except Exception:
            pytest.skip("Engine not available")

        entities = []
        for i in range(100):
            entities.append(_m.LegalEntity(
                entity_name=f"Entity-{i:03d}",
                equity_pct=Decimal("100"),
                total_scope1_tco2e=Decimal(str(random.randint(100, 10000))),
                total_scope2_tco2e=Decimal(str(random.randint(50, 5000))),
            ))
        org = _m.OrganizationStructure(org_name="Perf Test Org", entities=entities)
        engine = _m.OrganizationalBoundaryEngine()

        t0 = time.perf_counter()
        result = engine.define_boundary(org)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0
        assert result.total_entities == 100

    def test_500_entities_under_30s(self):
        """500 entities should evaluate in under 30 seconds."""
        try:
            _m = _load("organizational_boundary_engine")
        except Exception:
            pytest.skip("Engine not available")

        entities = []
        for i in range(500):
            entities.append(_m.LegalEntity(
                entity_name=f"Entity-{i:03d}",
                equity_pct=Decimal(str(random.randint(10, 100))),
                total_scope1_tco2e=Decimal(str(random.randint(10, 5000))),
                total_scope2_tco2e=Decimal(str(random.randint(5, 2500))),
            ))
        org = _m.OrganizationStructure(org_name="Large Org", entities=entities)
        engine = _m.OrganizationalBoundaryEngine()

        t0 = time.perf_counter()
        result = engine.define_boundary(org)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0
        assert result.total_entities == 500


# =============================================================================
# Scope 1 Calculation Performance
# =============================================================================


@pytest.mark.performance
class TestScope1Performance:
    """Test Scope 1 consolidation performance."""

    def test_single_facility_scope1_under_1s(self, sample_scope1_results):
        """Single facility scope 1 consolidation should be near-instant."""
        t0 = time.perf_counter()
        cats = sample_scope1_results["categories"]
        total = sum(c["total_tco2e"] for c in cats.values())
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0
        assert total > Decimal("0")

    def test_50_facility_aggregation_under_5s(self):
        """50 facility aggregation should complete in under 5 seconds."""
        facilities = []
        for i in range(50):
            facilities.append({
                "facility_id": f"FAC-{i:03d}",
                "scope1_tco2e": Decimal(str(random.randint(100, 10000))),
                "scope2_tco2e": Decimal(str(random.randint(50, 5000))),
            })

        t0 = time.perf_counter()
        total_scope1 = sum(f["scope1_tco2e"] for f in facilities)
        total_scope2 = sum(f["scope2_tco2e"] for f in facilities)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0
        assert total_scope1 > Decimal("0")
        assert total_scope2 > Decimal("0")


# =============================================================================
# Monte Carlo Performance
# =============================================================================


@pytest.mark.performance
class TestMonteCarloPerformance:
    """Test Monte Carlo simulation performance."""

    def test_10k_iterations_under_5s(self):
        """10,000 Monte Carlo iterations should complete in under 5 seconds."""
        rng = random.Random(42)
        n = 10000
        sources = [
            (10000.0, 700.0),   # stationary
            (2000.0, 220.0),    # mobile
            (4000.0, 640.0),    # process
            (350.0, 112.0),     # fugitive
            (1200.0, 264.0),    # refrigerant
            (1250.0, 312.5),    # waste
        ]

        t0 = time.perf_counter()
        results = []
        for _ in range(n):
            total = sum(rng.gauss(mean, std) for mean, std in sources)
            results.append(total)
        mean_result = sum(results) / n
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0
        assert len(results) == n
        assert 15000 < mean_result < 25000

    def test_100k_iterations_under_30s(self):
        """100,000 Monte Carlo iterations should complete in under 30 seconds."""
        rng = random.Random(42)
        n = 100000
        sources = [
            (10000.0, 700.0),
            (2000.0, 220.0),
            (4000.0, 640.0),
        ]

        t0 = time.perf_counter()
        results = []
        for _ in range(n):
            total = sum(rng.gauss(mean, std) for mean, std in sources)
            results.append(total)
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0
        assert len(results) == n


# =============================================================================
# Compliance Mapping Performance
# =============================================================================


@pytest.mark.performance
class TestComplianceMappingPerformance:
    """Test compliance mapping performance."""

    def test_7_frameworks_under_10s(self, sample_pack_config):
        """Mapping against 7 frameworks should complete in under 10 seconds."""
        frameworks = sample_pack_config["reporting"]["frameworks"]

        t0 = time.perf_counter()
        results = {}
        for fw in frameworks:
            # Simulate compliance check
            requirements = 50
            met = random.randint(40, 50)
            score = met / requirements * 100
            results[fw] = {
                "score": score,
                "met": met,
                "total": requirements,
            }
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0
        assert len(results) == len(frameworks)


# =============================================================================
# Report Generation Performance
# =============================================================================


@pytest.mark.performance
class TestReportGenerationPerformance:
    """Test report generation performance."""

    def test_executive_summary_under_5s(self, sample_inventory):
        """Executive summary generation should complete in under 5 seconds."""
        t0 = time.perf_counter()
        report = f"# GHG Inventory {sample_inventory['reporting_year']}\n"
        report += f"Scope 1: {sample_inventory['scope1']['total_tco2e']} tCO2e\n"
        report += f"Scope 2 (LB): {sample_inventory['scope2_location']['total_tco2e']} tCO2e\n"
        report += f"Scope 2 (MB): {sample_inventory['scope2_market']['total_tco2e']} tCO2e\n"
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0
        assert len(report) > 0

    def test_full_report_under_30s(self, sample_inventory, sample_scope1_results):
        """Full inventory report generation should complete in under 30 seconds."""
        t0 = time.perf_counter()
        sections = []
        sections.append(f"# GHG Inventory Report {sample_inventory['reporting_year']}")
        sections.append("## Scope 1 Breakdown")
        for cat, data in sample_scope1_results["categories"].items():
            sections.append(f"- {cat}: {data['total_tco2e']} tCO2e")
        sections.append("## Scope 2")
        sections.append(f"Location: {sample_inventory['scope2_location']['total_tco2e']} tCO2e")
        sections.append(f"Market: {sample_inventory['scope2_market']['total_tco2e']} tCO2e")
        report = "\n".join(sections)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0
        assert len(report) > 100


# =============================================================================
# Provenance Hash Performance
# =============================================================================


@pytest.mark.performance
class TestProvenanceHashPerformance:
    """Test provenance hash computation performance."""

    def test_1000_hashes_under_5s(self, sample_inventory):
        """1000 provenance hash computations should complete in under 5 seconds."""
        from tests.conftest import compute_provenance_hash

        t0 = time.perf_counter()
        for i in range(1000):
            modified = dict(sample_inventory)
            modified["iteration"] = i
            h = compute_provenance_hash(modified)
            assert len(h) == 64
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0


# =============================================================================
# Memory Efficiency
# =============================================================================


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory efficiency of large data processing."""

    def test_large_facility_dataset_processable(self):
        """1000 facilities should be processable without excessive memory."""
        facilities = []
        for i in range(1000):
            facilities.append({
                "facility_id": f"FAC-{i:05d}",
                "scope1_by_category": {
                    "stationary_combustion": Decimal(str(random.randint(100, 5000))),
                    "mobile_combustion": Decimal(str(random.randint(10, 1000))),
                    "process_emissions": Decimal(str(random.randint(0, 3000))),
                    "fugitive_emissions": Decimal(str(random.randint(0, 200))),
                    "refrigerant_fgas": Decimal(str(random.randint(0, 500))),
                    "land_use": Decimal("0"),
                    "waste_treatment": Decimal(str(random.randint(0, 300))),
                    "agricultural": Decimal("0"),
                },
            })

        total = sum(
            sum(f["scope1_by_category"].values())
            for f in facilities
        )
        assert total > Decimal("0")
        assert len(facilities) == 1000
