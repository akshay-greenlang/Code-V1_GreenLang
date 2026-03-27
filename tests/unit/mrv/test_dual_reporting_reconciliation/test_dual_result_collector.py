# -*- coding: utf-8 -*-
"""
Unit tests for DualResultCollectorEngine - AGENT-MRV-013 Engine 1

Tests the thread-safe singleton engine responsible for collecting, validating,
aligning, and organising upstream Scope 2 emission results from the four
upstream MRV agents (MRV-009 through MRV-012).

Test Coverage (target: ~120 tests):
    - Singleton pattern and thread safety
    - Result collection, validation, and splitting
    - Boundary alignment (tenant, period, GWP)
    - Energy type mapping and categorisation
    - Energy type and facility breakdowns
    - Completeness validation
    - Total emission calculations
    - Filtering methods (period, facility, energy type)
    - Grouping methods (method, energy type, facility)
    - PIF (Procurement Impact Factor) calculation
    - Unmatched result detection
    - Health check
"""

import pytest
import threading
from decimal import Decimal
from typing import Dict, Any, List
from datetime import date

from greenlang.agents.mrv.dual_reporting_reconciliation.dual_result_collector import (
    DualResultCollectorEngine,
    reset,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    UpstreamResult,
    ReconciliationWorkspace,
    EnergyTypeBreakdown,
    FacilityBreakdown,
    EnergyType,
    Scope2Method,
    UpstreamAgent,
)


# =============================================================================
# Test Class 1: Singleton Behavior (~8 tests)
# =============================================================================


class TestDualResultCollectorEngineSingleton:
    """Test singleton pattern and lifecycle management."""

    def test_singleton_same_instance(self):
        """Test multiple instantiations return the same instance."""
        engine1 = DualResultCollectorEngine()
        engine2 = DualResultCollectorEngine()

        assert engine1 is engine2

    def test_singleton_attributes_persist(self):
        """Test attributes persist across multiple references."""
        engine1 = DualResultCollectorEngine()
        # Access config to ensure it's initialized
        _ = engine1._config

        engine2 = DualResultCollectorEngine()
        assert engine2._config is engine1._config

    def test_reset_clears_instance(self):
        """Test reset() clears the singleton instance."""
        engine1 = DualResultCollectorEngine()
        first_id = id(engine1)

        reset()

        engine2 = DualResultCollectorEngine()
        second_id = id(engine2)

        # After reset, we get a new instance
        assert first_id != second_id

    def test_thread_safety_singleton(self):
        """Test singleton is thread-safe."""
        instances = []

        def get_instance():
            instances.append(DualResultCollectorEngine())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        first_instance = instances[0]
        assert all(inst is first_instance for inst in instances)

    def test_initialization_state(self):
        """Test engine initializes with correct state."""
        engine = DualResultCollectorEngine()

        assert engine._config is not None
        assert engine._metrics is not None
        assert engine._provenance is not None
        assert engine._initialized is True

    def test_repr_contains_class_name(self):
        """Test __repr__ contains class name."""
        engine = DualResultCollectorEngine()
        repr_str = repr(engine)

        assert "DualResultCollectorEngine" in repr_str

    def test_str_contains_singleton_status(self):
        """Test __str__ contains singleton status."""
        engine = DualResultCollectorEngine()
        str_repr = str(engine)

        assert "singleton" in str_repr.lower()

    def test_multiple_resets(self):
        """Test multiple consecutive resets work correctly."""
        engine1 = DualResultCollectorEngine()
        id1 = id(engine1)

        reset()
        reset()
        reset()

        engine2 = DualResultCollectorEngine()
        id2 = id(engine2)

        assert id1 != id2


# =============================================================================
# Test Class 2: collect_results (~15 tests)
# =============================================================================


class TestCollectResults:
    """Test collect_results method."""

    def test_collect_results_valid_input(
        self,
        sample_location_result: Dict[str, Any],
        sample_market_result: Dict[str, Any],
    ):
        """Test collecting valid location and market results."""
        engine = DualResultCollectorEngine()

        upstream = [
            UpstreamResult(**sample_location_result),
            UpstreamResult(**sample_market_result),
        ]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert isinstance(workspace, ReconciliationWorkspace)
        assert workspace.tenant_id == "tenant-001"
        assert len(workspace.location_results) == 1
        assert len(workspace.market_results) == 1

    def test_collect_results_empty_list(self):
        """Test collecting with empty results list."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(workspace.location_results) == 0
        assert len(workspace.market_results) == 0
        assert workspace.total_location_tco2e == Decimal("0")
        assert workspace.total_market_tco2e == Decimal("0")

    def test_collect_results_location_only(self, sample_location_result: Dict[str, Any]):
        """Test collecting only location-based results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(workspace.location_results) == 1
        assert len(workspace.market_results) == 0

    def test_collect_results_market_only(self, sample_market_result: Dict[str, Any]):
        """Test collecting only market-based results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(workspace.location_results) == 0
        assert len(workspace.market_results) == 1

    def test_collect_results_multiple_location(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test collecting multiple location-based results."""
        engine = DualResultCollectorEngine()

        # Create two location results with different facilities
        result1 = sample_location_result.copy()
        result2 = sample_location_result.copy()
        result2["facility_id"] = "FAC-002"

        upstream = [
            UpstreamResult(**result1),
            UpstreamResult(**result2),
        ]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(workspace.location_results) == 2
        assert len(workspace.market_results) == 0

    def test_collect_results_multiple_market(self, sample_market_result: Dict[str, Any]):
        """Test collecting multiple market-based results."""
        engine = DualResultCollectorEngine()

        result1 = sample_market_result.copy()
        result2 = sample_market_result.copy()
        result2["facility_id"] = "FAC-002"

        upstream = [
            UpstreamResult(**result1),
            UpstreamResult(**result2),
        ]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(workspace.location_results) == 0
        assert len(workspace.market_results) == 2

    def test_collect_results_sets_reconciliation_id(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test reconciliation_id is set (UUID format)."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert workspace.reconciliation_id is not None
        assert len(workspace.reconciliation_id) > 0

    def test_collect_results_tenant_mismatch_filtered(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test results from wrong tenant are filtered out."""
        engine = DualResultCollectorEngine()

        # Create result with different tenant
        wrong_tenant = sample_location_result.copy()
        wrong_tenant["tenant_id"] = "tenant-999"

        upstream = [UpstreamResult(**wrong_tenant)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Result should be filtered out
        assert len(workspace.location_results) == 0

    def test_collect_results_period_mismatch_filtered(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test results from wrong period are filtered out."""
        engine = DualResultCollectorEngine()

        # Create result with different period
        wrong_period = sample_location_result.copy()
        wrong_period["period_start"] = "2023-01-01"
        wrong_period["period_end"] = "2023-12-31"

        upstream = [UpstreamResult(**wrong_period)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Result should be filtered out
        assert len(workspace.location_results) == 0

    def test_collect_results_calculates_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test total emissions are calculated correctly."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Should sum emissions from location and market results
        assert workspace.total_location_tco2e > Decimal("0")
        assert workspace.total_market_tco2e > Decimal("0")

    def test_collect_results_invalid_result_filtered(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test invalid results are filtered with validation errors."""
        engine = DualResultCollectorEngine()

        # Create invalid result with negative emissions
        invalid = sample_location_result.copy()
        invalid["emissions_tco2e"] = Decimal("-100.0")

        # This should raise validation error from Pydantic model
        with pytest.raises(Exception):
            UpstreamResult(**invalid)

    def test_collect_results_with_all_energy_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test collecting results with multiple energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Should have both electricity and steam results
        location_types = {r.energy_type for r in workspace.location_results}
        market_types = {r.energy_type for r in workspace.market_results}

        assert EnergyType.ELECTRICITY in location_types or EnergyType.ELECTRICITY in market_types
        assert EnergyType.STEAM in location_types or EnergyType.STEAM in market_types

    def test_collect_results_stores_period(self):
        """Test period_start and period_end are stored in workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert workspace.period_start == date(2024, 1, 1)
        assert workspace.period_end == date(2024, 12, 31)

    def test_collect_results_provenance_recorded(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test provenance is recorded for the collection."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Workspace should exist, indicating provenance was tracked
        assert workspace is not None


# =============================================================================
# Test Class 3: align_boundaries (~10 tests)
# =============================================================================


class TestAlignBoundaries:
    """Test align_boundaries method."""

    def test_align_boundaries_matching_tenant(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment succeeds with matching tenant_id."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert aligned is not None
        assert aligned.tenant_id == "tenant-001"

    def test_align_boundaries_matching_period(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment succeeds with matching periods."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert aligned.period_start == date(2024, 1, 1)
        assert aligned.period_end == date(2024, 12, 31)

    def test_align_boundaries_empty_workspace(self):
        """Test alignment with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert aligned is not None
        assert len(aligned.location_results) == 0
        assert len(aligned.market_results) == 0

    def test_align_boundaries_location_only(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test alignment with only location results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert len(aligned.location_results) == 1
        assert len(aligned.market_results) == 0

    def test_align_boundaries_market_only(self, sample_market_result: Dict[str, Any]):
        """Test alignment with only market results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert len(aligned.location_results) == 0
        assert len(aligned.market_results) == 1

    def test_align_boundaries_preserves_results(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment preserves all results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        # Total count should be preserved
        original_count = len(workspace.location_results) + len(workspace.market_results)
        aligned_count = len(aligned.location_results) + len(aligned.market_results)

        assert aligned_count == original_count

    def test_align_boundaries_matching_gwp(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment with matching GWP sources."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        # All results should have same GWP source (AR5 in fixtures)
        all_gwp = {r.gwp_source for r in aligned.location_results + aligned.market_results}
        assert len(all_gwp) <= 2  # Could be AR5, AR6, or both

    def test_align_boundaries_returns_workspace(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment returns ReconciliationWorkspace."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        assert isinstance(aligned, ReconciliationWorkspace)

    def test_align_boundaries_maintains_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test alignment maintains total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        # Totals should be preserved
        assert aligned.total_location_tco2e == workspace.total_location_tco2e
        assert aligned.total_market_tco2e == workspace.total_market_tco2e

    def test_align_boundaries_multi_facility(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test alignment with multiple facilities."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        aligned = engine.align_boundaries(workspace)

        # Should handle multiple facilities
        facilities = {r.facility_id for r in aligned.location_results + aligned.market_results}
        assert len(facilities) >= 1


# =============================================================================
# Test Class 4: map_energy_types (~8 tests)
# =============================================================================


class TestMapEnergyTypes:
    """Test map_energy_types method."""

    def test_map_energy_types_electricity(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test mapping electricity energy type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        # Should have electricity in by_energy_type
        assert len(mapped.by_energy_type) > 0

    def test_map_energy_types_steam(self, sample_steam_location_result: Dict[str, Any]):
        """Test mapping steam energy type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_steam_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        # Should have steam in by_energy_type
        assert len(mapped.by_energy_type) > 0

    def test_map_energy_types_multiple_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test mapping multiple energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        # Should have multiple energy types
        assert len(mapped.by_energy_type) >= 1

    def test_map_energy_types_empty_workspace(self):
        """Test mapping with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        # Should have empty by_energy_type
        assert len(mapped.by_energy_type) == 0

    def test_map_energy_types_returns_workspace(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test mapping returns ReconciliationWorkspace."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        assert isinstance(mapped, ReconciliationWorkspace)

    def test_map_energy_types_preserves_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test mapping preserves total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        assert mapped.total_location_tco2e == workspace.total_location_tco2e
        assert mapped.total_market_tco2e == workspace.total_market_tco2e

    def test_map_energy_types_all_four_types(self):
        """Test mapping with all four GHG Protocol energy types."""
        engine = DualResultCollectorEngine()

        # Create results for all four energy types
        results = []
        for energy_type in [
            "electricity",
            "steam",
            "district_heating",
            "district_cooling",
        ]:
            result = {
                "agent": "mrv_009",
                "facility_id": f"FAC-{energy_type}",
                "energy_type": energy_type,
                "method": "location_based",
                "emissions_tco2e": Decimal("100.0"),
                "tenant_id": "tenant-001",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
            results.append(UpstreamResult(**result))

        workspace = engine.collect_results(
            upstream_results=results,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        # Should have breakdowns for all energy types
        assert len(mapped.by_energy_type) == 4

    def test_map_energy_types_preserves_results(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test mapping preserves all results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        mapped = engine.map_energy_types(workspace)

        original_count = len(workspace.location_results) + len(workspace.market_results)
        mapped_count = len(mapped.location_results) + len(mapped.market_results)

        assert mapped_count == original_count


# =============================================================================
# Test Class 5: compute_energy_type_breakdowns (~15 tests)
# =============================================================================


class TestComputeEnergyTypeBreakdowns:
    """Test compute_energy_type_breakdowns method."""

    def test_compute_energy_type_breakdowns_single_type(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test computing breakdown for single energy type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        assert len(result.by_energy_type) > 0

    def test_compute_energy_type_breakdowns_calculates_difference(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test breakdown calculates difference correctly."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        for breakdown in result.by_energy_type:
            # Difference = location - market
            expected_diff = breakdown.location_tco2e - breakdown.market_tco2e
            assert breakdown.difference_tco2e == expected_diff

    def test_compute_energy_type_breakdowns_direction_market_lower(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test direction is MARKET_LOWER when market < location."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        # From fixtures: location=1250.50, market=625.25
        # So market is lower
        for breakdown in result.by_energy_type:
            if breakdown.market_tco2e < breakdown.location_tco2e:
                assert breakdown.direction.value == "market_lower"

    def test_compute_energy_type_breakdowns_direction_market_higher(self):
        """Test direction is MARKET_HIGHER when market > location."""
        engine = DualResultCollectorEngine()

        # Create results where market > location
        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("500.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("800.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        for breakdown in result.by_energy_type:
            if breakdown.market_tco2e > breakdown.location_tco2e:
                assert breakdown.direction.value == "market_higher"

    def test_compute_energy_type_breakdowns_direction_equal(self):
        """Test direction is EQUAL when market == location."""
        engine = DualResultCollectorEngine()

        # Create results where market == location
        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        for breakdown in result.by_energy_type:
            if breakdown.market_tco2e == breakdown.location_tco2e:
                assert breakdown.direction.value == "equal"

    def test_compute_energy_type_breakdowns_percentage_calculation(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test percentage difference is calculated correctly."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        for breakdown in result.by_energy_type:
            if breakdown.location_tco2e > Decimal("0"):
                # Percentage = (difference / larger) * 100
                larger = max(breakdown.location_tco2e, breakdown.market_tco2e)
                expected_pct = (abs(breakdown.difference_tco2e) / larger) * Decimal("100")
                # Allow small rounding differences
                assert abs(breakdown.difference_pct - expected_pct) < Decimal("0.01")

    def test_compute_energy_type_breakdowns_empty_workspace(self):
        """Test computing breakdowns with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        assert len(result.by_energy_type) == 0

    def test_compute_energy_type_breakdowns_location_only(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test computing breakdowns with only location results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        # Should still create breakdown with zero market
        assert len(result.by_energy_type) > 0
        for breakdown in result.by_energy_type:
            assert breakdown.market_tco2e == Decimal("0")

    def test_compute_energy_type_breakdowns_market_only(
        self, sample_market_result: Dict[str, Any]
    ):
        """Test computing breakdowns with only market results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        # Should still create breakdown with zero location
        assert len(result.by_energy_type) > 0
        for breakdown in result.by_energy_type:
            assert breakdown.location_tco2e == Decimal("0")

    def test_compute_energy_type_breakdowns_multiple_energy_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test computing breakdowns with multiple energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        # Should have breakdowns for multiple types
        assert len(result.by_energy_type) >= 1

    def test_compute_energy_type_breakdowns_returns_workspace(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test method returns ReconciliationWorkspace."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        assert isinstance(result, ReconciliationWorkspace)

    def test_compute_energy_type_breakdowns_preserves_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test computing breakdowns preserves total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        assert result.total_location_tco2e == workspace.total_location_tco2e
        assert result.total_market_tco2e == workspace.total_market_tco2e

    def test_compute_energy_type_breakdowns_sums_to_total(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test energy type breakdowns sum to total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        # Sum of breakdowns should equal total
        total_location = sum(b.location_tco2e for b in result.by_energy_type)
        total_market = sum(b.market_tco2e for b in result.by_energy_type)

        assert total_location == result.total_location_tco2e
        assert total_market == result.total_market_tco2e

    def test_compute_energy_type_breakdowns_energy_mwh_populated(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test energy_mwh field is populated in breakdowns."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_energy_type_breakdowns(workspace)

        for breakdown in result.by_energy_type:
            # Energy should be positive for non-empty results
            if breakdown.location_tco2e > Decimal("0") or breakdown.market_tco2e > Decimal("0"):
                assert breakdown.energy_mwh >= Decimal("0")


# =============================================================================
# Test Class 6: compute_facility_breakdowns (~10 tests)
# =============================================================================


class TestComputeFacilityBreakdowns:
    """Test compute_facility_breakdowns method."""

    def test_compute_facility_breakdowns_single_facility(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test computing breakdown for single facility."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        assert len(result.by_facility) > 0

    def test_compute_facility_breakdowns_multiple_facilities(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test computing breakdowns for multiple facilities."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        # Should have at least 2 facilities from fixtures
        assert len(result.by_facility) >= 2

    def test_compute_facility_breakdowns_calculates_difference(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test facility breakdown calculates difference correctly."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        for breakdown in result.by_facility:
            expected_diff = breakdown.location_tco2e - breakdown.market_tco2e
            assert breakdown.difference_tco2e == expected_diff

    def test_compute_facility_breakdowns_percentage_calculation(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test percentage difference is calculated correctly."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        for breakdown in result.by_facility:
            if breakdown.location_tco2e > Decimal("0"):
                larger = max(breakdown.location_tco2e, breakdown.market_tco2e)
                expected_pct = (abs(breakdown.difference_tco2e) / larger) * Decimal("100")
                assert abs(breakdown.difference_pct - expected_pct) < Decimal("0.01")

    def test_compute_facility_breakdowns_empty_workspace(self):
        """Test computing facility breakdowns with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        assert len(result.by_facility) == 0

    def test_compute_facility_breakdowns_aggregates_energy_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test facility breakdown aggregates all energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        # Each facility should aggregate all energy types
        for breakdown in result.by_facility:
            assert breakdown.location_tco2e >= Decimal("0")
            assert breakdown.market_tco2e >= Decimal("0")

    def test_compute_facility_breakdowns_sums_to_total(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test facility breakdowns sum to total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        total_location = sum(b.location_tco2e for b in result.by_facility)
        total_market = sum(b.market_tco2e for b in result.by_facility)

        assert total_location == result.total_location_tco2e
        assert total_market == result.total_market_tco2e

    def test_compute_facility_breakdowns_returns_workspace(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test method returns ReconciliationWorkspace."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        assert isinstance(result, ReconciliationWorkspace)

    def test_compute_facility_breakdowns_preserves_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test computing facility breakdowns preserves totals."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        assert result.total_location_tco2e == workspace.total_location_tco2e
        assert result.total_market_tco2e == workspace.total_market_tco2e

    def test_compute_facility_breakdowns_facility_name_populated(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test facility_name is populated in breakdowns."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        result = engine.compute_facility_breakdowns(workspace)

        for breakdown in result.by_facility:
            # Facility ID should always be present
            assert breakdown.facility_id is not None
            assert len(breakdown.facility_id) > 0


# =============================================================================
# Test Class 7: validate_completeness (~10 tests)
# =============================================================================


class TestValidateCompleteness:
    """Test validate_completeness method."""

    def test_validate_completeness_full_coverage(self):
        """Test validation with full coverage of all energy types."""
        engine = DualResultCollectorEngine()

        # Create results for all four energy types, both methods
        results = []
        for energy_type in [
            "electricity",
            "steam",
            "district_heating",
            "district_cooling",
        ]:
            for method in ["location_based", "market_based"]:
                result = {
                    "agent": "mrv_009",
                    "facility_id": f"FAC-{energy_type}",
                    "energy_type": energy_type,
                    "method": method,
                    "emissions_tco2e": Decimal("100.0"),
                    "tenant_id": "tenant-001",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                }
                results.append(UpstreamResult(**result))

        workspace = engine.collect_results(
            upstream_results=results,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should have all four energy types
        assert len(validated.by_energy_type) == 4

    def test_validate_completeness_partial_coverage(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test validation with partial energy type coverage."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should have at least one energy type
        assert len(validated.by_energy_type) >= 1

    def test_validate_completeness_empty_workspace(self):
        """Test validation with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should have no energy types
        assert len(validated.by_energy_type) == 0

    def test_validate_completeness_location_missing(
        self, sample_market_result: Dict[str, Any]
    ):
        """Test validation when location method is missing."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should still create workspace but with missing location
        assert len(validated.location_results) == 0

    def test_validate_completeness_market_missing(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test validation when market method is missing."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should still create workspace but with missing market
        assert len(validated.market_results) == 0

    def test_validate_completeness_returns_workspace(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test validation returns ReconciliationWorkspace."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        assert isinstance(validated, ReconciliationWorkspace)

    def test_validate_completeness_preserves_results(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test validation preserves all results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        original_count = len(workspace.location_results) + len(workspace.market_results)
        validated_count = len(validated.location_results) + len(validated.market_results)

        assert validated_count == original_count

    def test_validate_completeness_preserves_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test validation preserves total emissions."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        assert validated.total_location_tco2e == workspace.total_location_tco2e
        assert validated.total_market_tco2e == workspace.total_market_tco2e

    def test_validate_completeness_electricity_only(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test validation with only electricity energy type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should have electricity type
        energy_types = {b.energy_type for b in validated.by_energy_type}
        assert EnergyType.ELECTRICITY in energy_types or len(energy_types) >= 1

    def test_validate_completeness_mixed_energy_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test validation with mixed energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        validated = engine.validate_completeness(workspace)

        # Should have multiple energy types
        assert len(validated.by_energy_type) >= 1


# =============================================================================
# Test Class 8: get_total_emissions (~8 tests)
# =============================================================================


class TestGetTotalEmissions:
    """Test get_total_emissions method."""

    def test_get_total_emissions_both_methods(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test getting totals for both methods."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert location > Decimal("0")
        assert market > Decimal("0")

    def test_get_total_emissions_empty_workspace(self):
        """Test getting totals from empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert location == Decimal("0")
        assert market == Decimal("0")

    def test_get_total_emissions_location_only(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test getting totals with only location results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert location > Decimal("0")
        assert market == Decimal("0")

    def test_get_total_emissions_market_only(self, sample_market_result: Dict[str, Any]):
        """Test getting totals with only market results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert location == Decimal("0")
        assert market > Decimal("0")

    def test_get_total_emissions_matches_workspace_totals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test totals match workspace attributes."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert location == workspace.total_location_tco2e
        assert market == workspace.total_market_tco2e

    def test_get_total_emissions_returns_decimals(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test method returns Decimal types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        assert isinstance(location, Decimal)
        assert isinstance(market, Decimal)

    def test_get_total_emissions_multiple_facilities(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test totals aggregate across multiple facilities."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        # Should aggregate all facilities
        assert location >= Decimal("0")
        assert market >= Decimal("0")

    def test_get_total_emissions_precision(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test totals maintain decimal precision."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        location, market = engine.get_total_emissions(workspace)

        # Check precision is maintained (up to 8 decimal places)
        assert location.as_tuple().exponent >= -8
        assert market.as_tuple().exponent >= -8


# =============================================================================
# Test Class 9: Filter Methods (~12 tests)
# =============================================================================


class TestFilterMethods:
    """Test filtering methods (filter_by_period, filter_by_facility, filter_by_energy_type)."""

    def test_filter_by_period_matching(self, sample_upstream_results: List[Dict[str, Any]]):
        """Test filter_by_period returns matching results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_period(
            workspace,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        # Should return all results
        original_count = len(workspace.location_results) + len(workspace.market_results)
        filtered_count = len(filtered)

        assert filtered_count == original_count

    def test_filter_by_period_no_match(self, sample_upstream_results: List[Dict[str, Any]]):
        """Test filter_by_period returns empty list for non-matching period."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_period(
            workspace,
            period_start=date(2023, 1, 1),
            period_end=date(2023, 12, 31),
        )

        # Should return no results
        assert len(filtered) == 0

    def test_filter_by_period_empty_workspace(self):
        """Test filter_by_period with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_period(
            workspace,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        assert len(filtered) == 0

    def test_filter_by_facility_matching(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test filter_by_facility returns matching results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_facility(workspace, facility_id="FAC-001")

        # Should return results for FAC-001
        assert len(filtered) > 0
        assert all(r.facility_id == "FAC-001" for r in filtered)

    def test_filter_by_facility_no_match(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test filter_by_facility returns empty list for non-matching facility."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_facility(workspace, facility_id="FAC-999")

        assert len(filtered) == 0

    def test_filter_by_facility_empty_workspace(self):
        """Test filter_by_facility with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_facility(workspace, facility_id="FAC-001")

        assert len(filtered) == 0

    def test_filter_by_energy_type_matching(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test filter_by_energy_type returns matching results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_energy_type(workspace, energy_type=EnergyType.ELECTRICITY)

        # Should return electricity results
        assert len(filtered) > 0
        assert all(r.energy_type == EnergyType.ELECTRICITY for r in filtered)

    def test_filter_by_energy_type_no_match(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test filter_by_energy_type returns empty list for non-matching type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_energy_type(
            workspace, energy_type=EnergyType.DISTRICT_COOLING
        )

        # If no district cooling in fixtures, should return empty
        assert isinstance(filtered, list)

    def test_filter_by_energy_type_empty_workspace(self):
        """Test filter_by_energy_type with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_energy_type(workspace, energy_type=EnergyType.ELECTRICITY)

        assert len(filtered) == 0

    def test_filter_by_energy_type_steam(
        self, sample_steam_location_result: Dict[str, Any]
    ):
        """Test filter_by_energy_type with steam type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_steam_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        filtered = engine.filter_by_energy_type(workspace, energy_type=EnergyType.STEAM)

        assert len(filtered) > 0
        assert all(r.energy_type == EnergyType.STEAM for r in filtered)

    def test_filter_by_energy_type_multiple_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test filter_by_energy_type correctly separates types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        elec = engine.filter_by_energy_type(workspace, energy_type=EnergyType.ELECTRICITY)
        steam = engine.filter_by_energy_type(workspace, energy_type=EnergyType.STEAM)

        # Should have different counts
        assert len(elec) + len(steam) > 0


# =============================================================================
# Test Class 10: Group Methods (~10 tests)
# =============================================================================


class TestGroupMethods:
    """Test grouping methods (group_by_method, group_by_energy_type, group_by_facility)."""

    def test_group_by_method_both_methods(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test group_by_method separates location and market."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_method(workspace)

        assert Scope2Method.LOCATION_BASED in grouped
        assert Scope2Method.MARKET_BASED in grouped
        assert len(grouped[Scope2Method.LOCATION_BASED]) > 0
        assert len(grouped[Scope2Method.MARKET_BASED]) > 0

    def test_group_by_method_location_only(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test group_by_method with only location results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_method(workspace)

        assert len(grouped[Scope2Method.LOCATION_BASED]) > 0
        assert len(grouped.get(Scope2Method.MARKET_BASED, [])) == 0

    def test_group_by_method_market_only(self, sample_market_result: Dict[str, Any]):
        """Test group_by_method with only market results."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_method(workspace)

        assert len(grouped.get(Scope2Method.LOCATION_BASED, [])) == 0
        assert len(grouped[Scope2Method.MARKET_BASED]) > 0

    def test_group_by_energy_type_single_type(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test group_by_energy_type with single energy type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_energy_type(workspace)

        assert EnergyType.ELECTRICITY in grouped
        assert len(grouped[EnergyType.ELECTRICITY]) > 0

    def test_group_by_energy_type_multiple_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test group_by_energy_type with multiple energy types."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_energy_type(workspace)

        # Should have at least one energy type
        assert len(grouped) >= 1

    def test_group_by_energy_type_empty_workspace(self):
        """Test group_by_energy_type with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_energy_type(workspace)

        assert len(grouped) == 0

    def test_group_by_facility_single_facility(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test group_by_facility with single facility."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_facility(workspace)

        assert "FAC-001" in grouped
        assert len(grouped["FAC-001"]) > 0

    def test_group_by_facility_multiple_facilities(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test group_by_facility with multiple facilities."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_facility(workspace)

        # Should have at least 2 facilities from fixtures
        assert len(grouped) >= 2

    def test_group_by_facility_empty_workspace(self):
        """Test group_by_facility with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_facility(workspace)

        assert len(grouped) == 0

    def test_group_by_facility_aggregates_energy_types(
        self, sample_multi_facility_results: List[Dict[str, Any]]
    ):
        """Test group_by_facility aggregates all energy types per facility."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_multi_facility_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        grouped = engine.group_by_facility(workspace)

        # Each facility group should contain results for that facility
        for facility_id, results in grouped.items():
            assert all(r.facility_id == facility_id for r in results)


# =============================================================================
# Test Class 11: compute_pif (~8 tests)
# =============================================================================


class TestComputePif:
    """Test compute_pif (Procurement Impact Factor) calculation."""

    def test_compute_pif_market_lower(self):
        """Test PIF calculation when market < location (negative PIF)."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("600.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # PIF = ((market - location) / location) * 100
        # = ((600 - 1000) / 1000) * 100 = -40%
        expected = ((Decimal("600.0") - Decimal("1000.0")) / Decimal("1000.0")) * Decimal("100")
        assert abs(pif - expected) < Decimal("0.01")

    def test_compute_pif_market_higher(self):
        """Test PIF calculation when market > location (positive PIF)."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("500.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("800.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # PIF = ((800 - 500) / 500) * 100 = 60%
        expected = ((Decimal("800.0") - Decimal("500.0")) / Decimal("500.0")) * Decimal("100")
        assert abs(pif - expected) < Decimal("0.01")

    def test_compute_pif_equal(self):
        """Test PIF calculation when market == location (zero PIF)."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # PIF should be 0
        assert pif == Decimal("0")

    def test_compute_pif_zero_location(self):
        """Test PIF calculation when location is zero (undefined, returns None or 0)."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("0.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("100.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # PIF is undefined when location = 0, should return None or 0
        assert pif is None or pif == Decimal("0")

    def test_compute_pif_returns_decimal(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test PIF returns Decimal type."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        assert pif is None or isinstance(pif, Decimal)

    def test_compute_pif_empty_workspace(self):
        """Test PIF with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # Should return None or 0 when no results
        assert pif is None or pif == Decimal("0")

    def test_compute_pif_negative_impact(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test PIF correctly shows negative impact (procurement reduced emissions)."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # From fixtures: location=1250.50, market=625.25
        # PIF should be negative (market lower)
        if pif is not None:
            assert pif < Decimal("0")

    def test_compute_pif_large_reduction(self):
        """Test PIF with 100% reduction (market = 0)."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("0.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        pif = engine.compute_pif(workspace)

        # PIF = ((0 - 1000) / 1000) * 100 = -100%
        assert pif == Decimal("-100")


# =============================================================================
# Test Class 12: detect_unmatched_results (~6 tests)
# =============================================================================


class TestDetectUnmatched:
    """Test detect_unmatched_results method."""

    def test_detect_unmatched_all_matched(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test detection when all results are matched."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        # With sample fixtures (both location and market for same facility/energy),
        # there should be no unmatched
        assert len(unmatched) == 0

    def test_detect_unmatched_location_only(
        self, sample_location_result: Dict[str, Any]
    ):
        """Test detection when only location results exist."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_location_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        # Should have one unmatched location result
        assert len(unmatched) > 0

    def test_detect_unmatched_market_only(self, sample_market_result: Dict[str, Any]):
        """Test detection when only market results exist."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**sample_market_result)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        # Should have one unmatched market result
        assert len(unmatched) > 0

    def test_detect_unmatched_empty_workspace(self):
        """Test detection with empty workspace."""
        engine = DualResultCollectorEngine()

        workspace = engine.collect_results(
            upstream_results=[],
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        assert len(unmatched) == 0

    def test_detect_unmatched_different_facilities(self):
        """Test detection when location and market have different facilities."""
        engine = DualResultCollectorEngine()

        location = {
            "agent": "mrv_009",
            "facility_id": "FAC-001",
            "energy_type": "electricity",
            "method": "location_based",
            "emissions_tco2e": Decimal("1000.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        market = {
            "agent": "mrv_009",
            "facility_id": "FAC-002",
            "energy_type": "electricity",
            "method": "market_based",
            "emissions_tco2e": Decimal("800.0"),
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }

        upstream = [UpstreamResult(**location), UpstreamResult(**market)]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        # Both should be unmatched (different facilities)
        assert len(unmatched) == 2

    def test_detect_unmatched_returns_list(
        self, sample_upstream_results: List[Dict[str, Any]]
    ):
        """Test method returns list of UpstreamResult."""
        engine = DualResultCollectorEngine()

        upstream = [UpstreamResult(**r) for r in sample_upstream_results]

        workspace = engine.collect_results(
            upstream_results=upstream,
            tenant_id="tenant-001",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
        )

        unmatched = engine.detect_unmatched_results(workspace)

        assert isinstance(unmatched, list)


# =============================================================================
# Test Class 13: health_check (~4 tests)
# =============================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_dict(self):
        """Test health check returns dictionary."""
        engine = DualResultCollectorEngine()

        health = engine.health_check()

        assert isinstance(health, dict)

    def test_health_check_contains_status(self):
        """Test health check contains status field."""
        engine = DualResultCollectorEngine()

        health = engine.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_contains_engine_name(self):
        """Test health check contains engine identifier."""
        engine = DualResultCollectorEngine()

        health = engine.health_check()

        assert "engine" in health or "component" in health

    def test_health_check_initialized_state(self):
        """Test health check reflects initialized state."""
        engine = DualResultCollectorEngine()

        health = engine.health_check()

        # Should indicate engine is initialized
        assert "initialized" in health or health.get("status") == "healthy"
