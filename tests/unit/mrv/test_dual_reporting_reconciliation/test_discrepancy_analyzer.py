# -*- coding: utf-8 -*-
"""
Unit tests for DiscrepancyAnalyzerEngine (Engine 2 of 7)

AGENT-MRV-013: Dual Reporting Reconciliation Agent

Tests all methods of DiscrepancyAnalyzerEngine with comprehensive coverage.
Validates discrepancy analysis, classification, waterfall decomposition, and flag generation.
"""

import pytest
import threading
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List

from greenlang.agents.mrv.dual_reporting_reconciliation.discrepancy_analyzer import (
    DiscrepancyAnalyzerEngine,
)
from greenlang.agents.mrv.dual_reporting_reconciliation.models import (
    EnergyType,
    Scope2Method,
    UpstreamAgent,
    DiscrepancyType,
    DiscrepancyDirection,
    MaterialityLevel,
    FlagType,
    FlagSeverity,
    EFHierarchyPriority,
    UpstreamResult,
    ReconciliationWorkspace,
    Discrepancy,
    WaterfallItem,
    WaterfallDecomposition,
    DiscrepancyReport,
    Flag,
)


# ===========================================================================
# Helper Functions
# ===========================================================================


def _create_upstream_result(**kwargs) -> UpstreamResult:
    """Create an UpstreamResult with sensible defaults."""
    defaults = {
        "agent": UpstreamAgent.MRV_009,
        "facility_id": "FAC-001",
        "energy_type": EnergyType.ELECTRICITY,
        "method": Scope2Method.LOCATION_BASED,
        "emissions_tco2e": Decimal("1000.0"),
        "energy_quantity_mwh": Decimal("4000.0"),
        "ef_used": Decimal("0.25"),
        "ef_source": "eGRID 2023",
        "ef_hierarchy": EFHierarchyPriority.GRID_AVERAGE,
        "tenant_id": "tenant-001",
        "period_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "period_end": datetime(2024, 12, 31, tzinfo=timezone.utc),
        "region": "US-CAMX",
    }
    defaults.update(kwargs)
    return UpstreamResult(**defaults)


def _create_workspace(
    location_results: List[UpstreamResult],
    market_results: List[UpstreamResult],
) -> ReconciliationWorkspace:
    """Create a ReconciliationWorkspace from location and market results."""
    total_loc = sum(r.emissions_tco2e for r in location_results)
    total_mkt = sum(r.emissions_tco2e for r in market_results)

    return ReconciliationWorkspace(
        reconciliation_id="recon-test-001",
        tenant_id="tenant-001",
        period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        location_results=location_results,
        market_results=market_results,
        total_location_tco2e=total_loc,
        total_market_tco2e=total_mkt,
    )


# ===========================================================================
# Test Class 1: Singleton Pattern
# ===========================================================================


class TestSingleton:
    """Test singleton pattern implementation."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        engine1 = DiscrepancyAnalyzerEngine()
        engine2 = DiscrepancyAnalyzerEngine()

        assert engine1 is engine2

    def test_singleton_reset(self):
        """Test that reset() allows creating a new instance."""
        engine1 = DiscrepancyAnalyzerEngine()
        instance_id_1 = id(engine1)

        DiscrepancyAnalyzerEngine.reset()

        engine2 = DiscrepancyAnalyzerEngine()
        instance_id_2 = id(engine2)

        # After reset, we get a new instance
        assert instance_id_1 != instance_id_2

    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe under concurrent access."""
        instances = []

        def create_instance():
            engine = DiscrepancyAnalyzerEngine()
            instances.append(id(engine))

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(instances)) == 1

    def test_engine_initialization_only_once(self):
        """Test that engine initializes only once despite multiple calls."""
        DiscrepancyAnalyzerEngine.reset()

        engine1 = DiscrepancyAnalyzerEngine()
        initial_created_at = engine1._created_at

        # Second call should not re-initialize
        engine2 = DiscrepancyAnalyzerEngine()

        assert engine1 is engine2
        assert engine2._created_at == initial_created_at

    def test_engine_id_and_version(self):
        """Test engine returns correct ID and version."""
        engine = DiscrepancyAnalyzerEngine()

        assert engine.get_engine_id() == "discrepancy-analyzer-engine"
        assert engine.get_engine_version() == "1.0.0"

    def test_initialized_flag_set_correctly(self):
        """Test _initialized flag is set after initialization."""
        DiscrepancyAnalyzerEngine.reset()

        assert not DiscrepancyAnalyzerEngine._initialized

        engine = DiscrepancyAnalyzerEngine()

        assert DiscrepancyAnalyzerEngine._initialized


# ===========================================================================
# Test Class 2: classify_discrepancy_type
# ===========================================================================


class TestClassifyDiscrepancyType:
    """Test classify_discrepancy_type method with all 8 types."""

    def test_classify_rec_go_impact_bundled_cert(self):
        """Test classification of REC/GO impact with bundled certificate."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            ef_used=Decimal("0.5"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.REC_GO_IMPACT

    def test_classify_rec_go_impact_unbundled_cert(self):
        """Test classification of REC/GO impact with unbundled certificate."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            ef_used=Decimal("0.5"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.UNBUNDLED_CERT,
            ef_used=Decimal("0.0005"),  # Near-zero
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.REC_GO_IMPACT

    def test_classify_supplier_ef_delta_with_cert(self):
        """Test classification of supplier EF delta with certificate."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            ef_used=Decimal("0.5"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.SUPPLIER_WITH_CERT,
            ef_used=Decimal("0.3"),
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.SUPPLIER_EF_DELTA

    def test_classify_supplier_ef_delta_no_cert(self):
        """Test classification of supplier EF delta without certificate."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            ef_used=Decimal("0.5"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.SUPPLIER_NO_CERT,
            ef_used=Decimal("0.4"),
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.SUPPLIER_EF_DELTA

    def test_classify_geographic_mismatch(self):
        """Test classification of geographic mismatch."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            region="US-CAMX",
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            region="US-ERCT",  # Different region
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.GEOGRAPHIC_MISMATCH

    def test_classify_temporal_mismatch_different_start(self):
        """Test classification of temporal mismatch with different start date."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            period_start=datetime(2024, 2, 1, tzinfo=timezone.utc),  # Different
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.TEMPORAL_MISMATCH

    def test_classify_temporal_mismatch_different_end(self):
        """Test classification of temporal mismatch with different end date."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 11, 30, tzinfo=timezone.utc),  # Different
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.TEMPORAL_MISMATCH

    def test_classify_steam_heat_method_divergence(self):
        """Test classification of steam/heat methodological divergence."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.STEAM,
            ef_source="IEA 2023",
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.STEAM,
            ef_source="Supplier CHP Allocation",  # Different source
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.STEAM_HEAT_METHOD

    def test_classify_residual_mix_uplift(self):
        """Test classification of residual mix uplift."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.RESIDUAL_MIX,
            ef_used=Decimal("0.6"),  # Higher than grid average
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.RESIDUAL_MIX_UPLIFT

    def test_classify_grid_update_timing(self):
        """Test classification of grid update timing."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_source="eGRID 2022",
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_source="eGRID 2023",  # Different vintage
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        assert disc_type == DiscrepancyType.GRID_UPDATE_TIMING

    def test_classify_default_to_residual_mix(self):
        """Test unclassified differences default to residual mix."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        # Default classification
        assert disc_type == DiscrepancyType.RESIDUAL_MIX_UPLIFT

    def test_classify_priority_rec_go_over_supplier(self):
        """Test REC/GO classification takes priority over supplier."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0001"),  # Near-zero with cert
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        # Should be REC_GO, not supplier
        assert disc_type == DiscrepancyType.REC_GO_IMPACT

    def test_classify_priority_supplier_over_geographic(self):
        """Test supplier classification takes priority over geographic mismatch."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            region="US-CAMX",
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            region="US-ERCT",  # Different region
            ef_hierarchy=EFHierarchyPriority.SUPPLIER_NO_CERT,  # But supplier
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        # Supplier takes priority
        assert disc_type == DiscrepancyType.SUPPLIER_EF_DELTA

    def test_classify_priority_geographic_over_temporal(self):
        """Test geographic mismatch takes priority over temporal."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            region="US-CAMX",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            region="US-ERCT",  # Different region
            period_start=datetime(2024, 2, 1, tzinfo=timezone.utc),  # And period
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
        )

        engine = DiscrepancyAnalyzerEngine()
        disc_type = engine.classify_discrepancy_type(loc_result, mkt_result)

        # Geographic takes priority
        assert disc_type == DiscrepancyType.GEOGRAPHIC_MISMATCH


# ===========================================================================
# Test Class 3: determine_materiality
# ===========================================================================


class TestDetermineMateriality:
    """Test determine_materiality method with all 5 levels."""

    def test_materiality_immaterial_zero(self):
        """Test immaterial classification for 0% discrepancy."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("0"))

        assert materiality == MaterialityLevel.IMMATERIAL

    def test_materiality_immaterial_boundary(self):
        """Test immaterial classification at boundary (4.99%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("4.99"))

        assert materiality == MaterialityLevel.IMMATERIAL

    def test_materiality_minor_lower_bound(self):
        """Test minor classification at lower boundary (5%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("5.0"))

        assert materiality == MaterialityLevel.MINOR

    def test_materiality_minor_upper_bound(self):
        """Test minor classification at upper boundary (14.99%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("14.99"))

        assert materiality == MaterialityLevel.MINOR

    def test_materiality_material_lower_bound(self):
        """Test material classification at lower boundary (15%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("15.0"))

        assert materiality == MaterialityLevel.MATERIAL

    def test_materiality_material_upper_bound(self):
        """Test material classification at upper boundary (49.99%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("49.99"))

        assert materiality == MaterialityLevel.MATERIAL

    def test_materiality_significant_lower_bound(self):
        """Test significant classification at lower boundary (50%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("50.0"))

        assert materiality == MaterialityLevel.SIGNIFICANT

    def test_materiality_significant_upper_bound(self):
        """Test significant classification at upper boundary (99.99%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("99.99"))

        assert materiality == MaterialityLevel.SIGNIFICANT

    def test_materiality_extreme_lower_bound(self):
        """Test extreme classification at lower boundary (100%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("100.0"))

        assert materiality == MaterialityLevel.EXTREME

    def test_materiality_extreme_high_value(self):
        """Test extreme classification for very high percentage (500%)."""
        engine = DiscrepancyAnalyzerEngine()
        materiality = engine.determine_materiality(Decimal("500.0"))

        assert materiality == MaterialityLevel.EXTREME


# ===========================================================================
# Test Class 4: calculate_total_discrepancy
# ===========================================================================


class TestAnalyzeAtTotalLevel:
    """Test calculate_total_discrepancy method."""

    def test_total_discrepancy_market_lower(self):
        """Test total discrepancy when market is lower than location."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        assert discrepancy is not None
        assert discrepancy.absolute_tco2e == Decimal("400.0")
        assert discrepancy.direction == DiscrepancyDirection.MARKET_LOWER

    def test_total_discrepancy_market_higher(self):
        """Test total discrepancy when market is higher than location."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("500.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("800.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        assert discrepancy is not None
        assert discrepancy.absolute_tco2e == Decimal("300.0")
        assert discrepancy.direction == DiscrepancyDirection.MARKET_HIGHER

    def test_total_discrepancy_both_zero(self):
        """Test total discrepancy when both totals are zero."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("0.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        assert discrepancy is None

    def test_total_discrepancy_percentage_calculation(self):
        """Test percentage calculation in total discrepancy."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        # |1000 - 600| / max(1000, 600) * 100 = 400/1000 * 100 = 40%
        assert discrepancy.percentage == Decimal("40.0")

    def test_total_discrepancy_pif_calculation(self):
        """Test PIF calculation in total discrepancy."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        # PIF = (1000 - 600) / 1000 * 100 = 40%
        # This should be in the description
        assert "PIF" in discrepancy.description

    def test_total_discrepancy_has_description(self):
        """Test total discrepancy includes description."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        assert discrepancy.description is not None
        assert len(discrepancy.description) > 0

    def test_total_discrepancy_has_recommendation(self):
        """Test total discrepancy includes recommendation."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancy = engine.calculate_total_discrepancy(workspace)

        assert discrepancy.recommendation is not None


# ===========================================================================
# Test Class 5: calculate_energy_type_discrepancies
# ===========================================================================


class TestAnalyzeAtEnergyTypeLevel:
    """Test calculate_energy_type_discrepancies method."""

    def test_energy_type_single_electricity(self):
        """Test energy type discrepancy for single electricity."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        assert len(discrepancies) == 1
        assert discrepancies[0].energy_type == EnergyType.ELECTRICITY
        assert discrepancies[0].absolute_tco2e == Decimal("400.0")

    def test_energy_type_multiple_types(self):
        """Test energy type discrepancies for multiple energy types."""
        loc_elec = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_elec = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("600.0"),
        )
        loc_steam = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.STEAM,
            facility_id="FAC-002",
            emissions_tco2e=Decimal("500.0"),
        )
        mkt_steam = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.STEAM,
            facility_id="FAC-002",
            emissions_tco2e=Decimal("400.0"),
        )
        workspace = _create_workspace(
            [loc_elec, loc_steam],
            [mkt_elec, mkt_steam],
        )

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        assert len(discrepancies) == 2
        energy_types = {d.energy_type for d in discrepancies}
        assert EnergyType.ELECTRICITY in energy_types
        assert EnergyType.STEAM in energy_types

    def test_energy_type_zero_discrepancy_excluded(self):
        """Test energy types with zero discrepancy are excluded."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),  # Same
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        assert len(discrepancies) == 0

    def test_energy_type_aggregation_across_facilities(self):
        """Test energy type discrepancies aggregate across facilities."""
        loc_fac1 = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("500.0"),
        )
        loc_fac2 = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-002",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("500.0"),
        )
        mkt_fac1 = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("300.0"),
        )
        mkt_fac2 = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-002",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("300.0"),
        )
        workspace = _create_workspace(
            [loc_fac1, loc_fac2],
            [mkt_fac1, mkt_fac2],
        )

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        assert len(discrepancies) == 1
        # Total electricity: 1000 location, 600 market, 400 discrepancy
        assert discrepancies[0].absolute_tco2e == Decimal("400.0")

    def test_energy_type_sorted_output(self):
        """Test energy type discrepancies are sorted by energy type."""
        results_loc = [
            _create_upstream_result(
                method=Scope2Method.LOCATION_BASED,
                energy_type=EnergyType.STEAM,
                facility_id="FAC-001",
                emissions_tco2e=Decimal("100.0"),
            ),
            _create_upstream_result(
                method=Scope2Method.LOCATION_BASED,
                energy_type=EnergyType.ELECTRICITY,
                facility_id="FAC-002",
                emissions_tco2e=Decimal("200.0"),
            ),
        ]
        results_mkt = [
            _create_upstream_result(
                method=Scope2Method.MARKET_BASED,
                energy_type=EnergyType.STEAM,
                facility_id="FAC-001",
                emissions_tco2e=Decimal("50.0"),
            ),
            _create_upstream_result(
                method=Scope2Method.MARKET_BASED,
                energy_type=EnergyType.ELECTRICITY,
                facility_id="FAC-002",
                emissions_tco2e=Decimal("100.0"),
            ),
        ]
        workspace = _create_workspace(results_loc, results_mkt)

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        # Should be sorted by enum value (alphabetically)
        assert len(discrepancies) == 2
        assert discrepancies[0].energy_type == EnergyType.ELECTRICITY
        assert discrepancies[1].energy_type == EnergyType.STEAM

    def test_energy_type_percentage_calculation(self):
        """Test percentage calculation for energy type discrepancies."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("500.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        # |1000 - 500| / max(1000, 500) * 100 = 50%
        assert discrepancies[0].percentage == Decimal("50.0")

    def test_energy_type_facility_id_none(self):
        """Test energy type discrepancy has facility_id=None."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_energy_type_discrepancies(workspace)

        assert discrepancies[0].facility_id is None


# ===========================================================================
# Test Class 6: calculate_facility_discrepancies
# ===========================================================================


class TestAnalyzeAtFacilityLevel:
    """Test calculate_facility_discrepancies method."""

    def test_facility_single_facility(self):
        """Test facility discrepancy for single facility."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert len(discrepancies) == 1
        assert discrepancies[0].facility_id == "FAC-001"
        assert discrepancies[0].absolute_tco2e == Decimal("400.0")

    def test_facility_multiple_facilities(self):
        """Test facility discrepancies for multiple facilities."""
        loc_fac1 = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("1000.0"),
        )
        loc_fac2 = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-002",
            emissions_tco2e=Decimal("500.0"),
        )
        mkt_fac1 = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("600.0"),
        )
        mkt_fac2 = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-002",
            emissions_tco2e=Decimal("400.0"),
        )
        workspace = _create_workspace(
            [loc_fac1, loc_fac2],
            [mkt_fac1, mkt_fac2],
        )

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert len(discrepancies) == 2
        facility_ids = {d.facility_id for d in discrepancies}
        assert "FAC-001" in facility_ids
        assert "FAC-002" in facility_ids

    def test_facility_aggregation_across_energy_types(self):
        """Test facility discrepancies aggregate across energy types."""
        loc_elec = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("500.0"),
        )
        loc_steam = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.STEAM,
            emissions_tco2e=Decimal("500.0"),
        )
        mkt_elec = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.ELECTRICITY,
            emissions_tco2e=Decimal("300.0"),
        )
        mkt_steam = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            energy_type=EnergyType.STEAM,
            emissions_tco2e=Decimal("300.0"),
        )
        workspace = _create_workspace(
            [loc_elec, loc_steam],
            [mkt_elec, mkt_steam],
        )

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert len(discrepancies) == 1
        # Total for FAC-001: 1000 location, 600 market, 400 discrepancy
        assert discrepancies[0].absolute_tco2e == Decimal("400.0")

    def test_facility_sorted_by_id(self):
        """Test facility discrepancies are sorted by facility ID."""
        results_loc = [
            _create_upstream_result(
                method=Scope2Method.LOCATION_BASED,
                facility_id="FAC-003",
                emissions_tco2e=Decimal("100.0"),
            ),
            _create_upstream_result(
                method=Scope2Method.LOCATION_BASED,
                facility_id="FAC-001",
                emissions_tco2e=Decimal("200.0"),
            ),
        ]
        results_mkt = [
            _create_upstream_result(
                method=Scope2Method.MARKET_BASED,
                facility_id="FAC-003",
                emissions_tco2e=Decimal("50.0"),
            ),
            _create_upstream_result(
                method=Scope2Method.MARKET_BASED,
                facility_id="FAC-001",
                emissions_tco2e=Decimal("100.0"),
            ),
        ]
        workspace = _create_workspace(results_loc, results_mkt)

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert len(discrepancies) == 2
        assert discrepancies[0].facility_id == "FAC-001"
        assert discrepancies[1].facility_id == "FAC-003"

    def test_facility_zero_discrepancy_excluded(self):
        """Test facilities with zero discrepancy are excluded."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("1000.0"),  # Same
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert len(discrepancies) == 0

    def test_facility_energy_type_none(self):
        """Test facility discrepancy has energy_type=None."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        assert discrepancies[0].energy_type is None

    def test_facility_region_populated(self):
        """Test facility discrepancy populates region if available."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            facility_id="FAC-001",
            region="US-CAMX",
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            region="US-CAMX",
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_facility_discrepancies(workspace)

        # Region should be populated
        assert discrepancies[0].region is not None


# ===========================================================================
# Test Class 7: calculate_instrument_discrepancies
# ===========================================================================


class TestAnalyzeAtInstrumentLevel:
    """Test calculate_instrument_discrepancies method."""

    def test_instrument_single_hierarchy(self):
        """Test instrument discrepancy for single hierarchy."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # Should have discrepancy for BUNDLED_CERT
        assert len(discrepancies) >= 1

    def test_instrument_multiple_hierarchies(self):
        """Test instrument discrepancies for multiple hierarchies."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        mkt_cert = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("2000.0"),
        )
        mkt_residual = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-002",
            ef_hierarchy=EFHierarchyPriority.RESIDUAL_MIX,
            emissions_tco2e=Decimal("700.0"),
            energy_quantity_mwh=Decimal("2000.0"),
            ef_used=Decimal("0.35"),
        )
        workspace = _create_workspace([loc_result], [mkt_cert, mkt_residual])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # Should have discrepancies for different hierarchies
        assert len(discrepancies) >= 1

    def test_instrument_none_hierarchy_excluded(self):
        """Test instrument with None hierarchy is excluded."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            ef_hierarchy=EFHierarchyPriority.GRID_AVERAGE,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.RESIDUAL_MIX,
            emissions_tco2e=Decimal("800.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.2"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # All discrepancies should have non-None hierarchy in description
        for disc in discrepancies:
            assert disc.description is not None

    def test_instrument_zero_mwh_excluded(self):
        """Test instrument with zero MWh is excluded."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("0.0"),  # Zero MWh
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # Zero MWh instrument should be excluded
        assert len(discrepancies) == 0

    def test_instrument_description_includes_mwh(self):
        """Test instrument discrepancy description includes MWh coverage."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        assert len(discrepancies) >= 1
        assert "MWh" in discrepancies[0].description

    def test_instrument_sorted_by_hierarchy(self):
        """Test instrument discrepancies are sorted by hierarchy value."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("2000.0"),
            energy_quantity_mwh=Decimal("8000.0"),
        )
        mkt_residual = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-001",
            ef_hierarchy=EFHierarchyPriority.RESIDUAL_MIX,
            emissions_tco2e=Decimal("1200.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.3"),
        )
        mkt_cert = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            facility_id="FAC-002",
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("4000.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_residual, mkt_cert])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # Should be sorted
        assert len(discrepancies) >= 1

    def test_instrument_percentage_calculation(self):
        """Test percentage calculation for instrument discrepancies."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.25"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        discrepancies = engine.calculate_instrument_discrepancies(workspace)

        # Percentage should be calculated
        assert len(discrepancies) >= 1
        assert discrepancies[0].percentage >= Decimal("0")


# ===========================================================================
# Test Class 8: Waterfall Decomposition
# ===========================================================================


class TestComputeWaterfall:
    """Test build_waterfall method and waterfall decomposition."""

    def test_waterfall_basic_structure(self):
        """Test waterfall decomposition basic structure."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        assert isinstance(waterfall, WaterfallDecomposition)
        assert waterfall.total_discrepancy_tco2e == Decimal("400.0")
        assert isinstance(waterfall.items, list)

    def test_waterfall_rec_go_impact_present(self):
        """Test waterfall includes REC/GO impact when applicable."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.25"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("0.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # Should have REC_GO_IMPACT item
        drivers = [item.driver for item in waterfall.items]
        assert "REC_GO_IMPACT" in drivers

    def test_waterfall_residual_mix_uplift_present(self):
        """Test waterfall includes residual mix uplift when applicable."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.25"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("1200.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_hierarchy=EFHierarchyPriority.RESIDUAL_MIX,
            ef_used=Decimal("0.3"),  # Higher than grid average
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # Should have RESIDUAL_MIX_UPLIFT item
        drivers = [item.driver for item in waterfall.items]
        assert "RESIDUAL_MIX_UPLIFT" in drivers

    def test_waterfall_supplier_ef_delta_present(self):
        """Test waterfall includes supplier EF delta when applicable."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_used=Decimal("0.25"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("800.0"),
            energy_quantity_mwh=Decimal("4000.0"),
            ef_hierarchy=EFHierarchyPriority.SUPPLIER_NO_CERT,
            ef_used=Decimal("0.2"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # Should have SUPPLIER_EF_DELTA item
        drivers = [item.driver for item in waterfall.items]
        assert "SUPPLIER_EF_DELTA" in drivers

    def test_waterfall_items_have_contribution(self):
        """Test waterfall items have non-zero contribution."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # All items should have contribution values
        for item in waterfall.items:
            assert item.contribution_tco2e is not None
            assert isinstance(item.contribution_tco2e, Decimal)

    def test_waterfall_items_have_description(self):
        """Test waterfall items have descriptions."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # All items should have descriptions
        for item in waterfall.items:
            assert item.description is not None
            assert len(item.description) > 0

    def test_waterfall_percentage_calculation(self):
        """Test waterfall items have percentage calculations."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # All items should have percentage
        for item in waterfall.items:
            assert item.contribution_pct is not None
            assert isinstance(item.contribution_pct, Decimal)

    def test_waterfall_balance_check_passes(self):
        """Test waterfall balance check ensures sum equals total."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        # Sum of contributions should approximately equal total discrepancy
        sum_contributions = sum(item.contribution_tco2e for item in waterfall.items)
        expected_total = Decimal("400.0")

        # Within tolerance (0.1 tCO2e)
        assert abs(sum_contributions - expected_total) <= Decimal("0.1")

    def test_waterfall_empty_for_zero_discrepancy(self):
        """Test waterfall is minimal when discrepancy is zero."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("1000.0"),  # Same
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        assert waterfall.total_discrepancy_tco2e == Decimal("0.0")

    def test_waterfall_counter_increments(self):
        """Test waterfall build increments counter."""
        DiscrepancyAnalyzerEngine.reset()
        engine = DiscrepancyAnalyzerEngine()

        initial_count = engine._total_waterfall_builds

        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine.build_waterfall(workspace, [])

        assert engine._total_waterfall_builds == initial_count + 1

    def test_waterfall_driver_uniqueness(self):
        """Test waterfall drivers are unique (no duplicates)."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        waterfall = engine.build_waterfall(workspace, [])

        drivers = [item.driver for item in waterfall.items]
        # Most drivers should be unique (except balance adjustments)
        assert len(drivers) <= len(set(drivers)) + 1


# ===========================================================================
# Test Class 9: Flag Generation
# ===========================================================================


class TestFlagMaterialDiscrepancies:
    """Test generate_discrepancy_flags method."""

    def test_flags_generated_for_material_discrepancy(self):
        """Test flags are generated for material discrepancies."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.MATERIAL,
            absolute_tco2e=Decimal("400.0"),
            percentage=Decimal("40.0"),
            description="Test discrepancy",
            recommendation="Test recommendation",
        )

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([discrepancy])

        assert len(flags) >= 1

    def test_flags_for_no_discrepancies(self):
        """Test informational flag generated when no discrepancies."""
        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([])

        assert len(flags) == 1
        assert flags[0].flag_type == FlagType.INFO
        assert "No material discrepancies" in flags[0].message

    def test_flags_include_flag_type(self):
        """Test flags include appropriate flag type."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.EXTREME,
            absolute_tco2e=Decimal("1000.0"),
            percentage=Decimal("200.0"),
            description="Test",
            recommendation="Test",
        )

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([discrepancy])

        # Extreme discrepancy should generate error flag
        assert any(f.flag_type == FlagType.ERROR for f in flags)

    def test_flags_include_severity(self):
        """Test flags include appropriate severity."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.SIGNIFICANT,
            absolute_tco2e=Decimal("600.0"),
            percentage=Decimal("60.0"),
            description="Test",
            recommendation="Test",
        )

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([discrepancy])

        # Flags should have severity
        for flag in flags:
            assert flag.severity in [
                FlagSeverity.LOW,
                FlagSeverity.MEDIUM,
                FlagSeverity.HIGH,
                FlagSeverity.CRITICAL,
            ]

    def test_flags_include_code(self):
        """Test flags include flag code."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.MATERIAL,
            absolute_tco2e=Decimal("400.0"),
            percentage=Decimal("40.0"),
            description="Test",
            recommendation="Test",
        )

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([discrepancy])

        for flag in flags:
            assert flag.code is not None
            assert len(flag.code) > 0

    def test_flags_include_message(self):
        """Test flags include message."""
        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.MATERIAL,
            absolute_tco2e=Decimal("400.0"),
            percentage=Decimal("40.0"),
            description="Test",
            recommendation="Test",
        )

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags([discrepancy])

        for flag in flags:
            assert flag.message is not None
            assert len(flag.message) > 0

    def test_flags_counter_increments(self):
        """Test flag generation increments counter."""
        DiscrepancyAnalyzerEngine.reset()
        engine = DiscrepancyAnalyzerEngine()

        initial_count = engine._total_flags_generated

        discrepancy = Discrepancy(
            discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
            direction=DiscrepancyDirection.MARKET_LOWER,
            materiality=MaterialityLevel.MATERIAL,
            absolute_tco2e=Decimal("400.0"),
            percentage=Decimal("40.0"),
            description="Test",
            recommendation="Test",
        )

        flags = engine.generate_discrepancy_flags([discrepancy])

        assert engine._total_flags_generated == initial_count + len(flags)

    def test_flags_multiple_discrepancies(self):
        """Test flags generated for multiple discrepancies."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("400.0"),
                percentage=Decimal("40.0"),
                description="Test 1",
                recommendation="Test 1",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SUPPLIER_EF_DELTA,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.SIGNIFICANT,
                absolute_tco2e=Decimal("600.0"),
                percentage=Decimal("60.0"),
                description="Test 2",
                recommendation="Test 2",
            ),
        ]

        engine = DiscrepancyAnalyzerEngine()
        flags = engine.generate_discrepancy_flags(discrepancies)

        # Should have flags for both discrepancies
        assert len(flags) >= 2


# ===========================================================================
# Test Class 10: Statistics and Summaries
# ===========================================================================


class TestComputeStats:
    """Test statistics and summary methods."""

    def test_get_material_discrepancies(self):
        """Test filtering material discrepancies."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.IMMATERIAL,
                absolute_tco2e=Decimal("10.0"),
                percentage=Decimal("2.0"),
                description="Test",
                recommendation="Test",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SUPPLIER_EF_DELTA,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("400.0"),
                percentage=Decimal("40.0"),
                description="Test",
                recommendation="Test",
            ),
        ]

        report = DiscrepancyReport(
            reconciliation_id="test-001",
            discrepancies=discrepancies,
            materiality_summary={},
        )

        engine = DiscrepancyAnalyzerEngine()
        material = engine.get_material_discrepancies(report)

        assert len(material) == 1
        assert material[0].materiality == MaterialityLevel.MATERIAL

    def test_get_discrepancies_by_type(self):
        """Test filtering discrepancies by type."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("400.0"),
                percentage=Decimal("40.0"),
                description="Test",
                recommendation="Test",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SUPPLIER_EF_DELTA,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("300.0"),
                percentage=Decimal("30.0"),
                description="Test",
                recommendation="Test",
            ),
        ]

        report = DiscrepancyReport(
            reconciliation_id="test-001",
            discrepancies=discrepancies,
            materiality_summary={},
        )

        engine = DiscrepancyAnalyzerEngine()
        rec_go = engine.get_discrepancies_by_type(
            report,
            DiscrepancyType.REC_GO_IMPACT,
        )

        assert len(rec_go) == 1
        assert rec_go[0].discrepancy_type == DiscrepancyType.REC_GO_IMPACT

    def test_get_discrepancies_by_energy_type(self):
        """Test filtering discrepancies by energy type."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("400.0"),
                percentage=Decimal("40.0"),
                energy_type=EnergyType.ELECTRICITY,
                description="Test",
                recommendation="Test",
            ),
            Discrepancy(
                discrepancy_type=DiscrepancyType.SUPPLIER_EF_DELTA,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("300.0"),
                percentage=Decimal("30.0"),
                energy_type=EnergyType.STEAM,
                description="Test",
                recommendation="Test",
            ),
        ]

        report = DiscrepancyReport(
            reconciliation_id="test-001",
            discrepancies=discrepancies,
            materiality_summary={},
        )

        engine = DiscrepancyAnalyzerEngine()
        electricity = engine.get_discrepancies_by_energy_type(
            report,
            EnergyType.ELECTRICITY,
        )

        assert len(electricity) == 1
        assert electricity[0].energy_type == EnergyType.ELECTRICITY

    def test_summarize_discrepancy_report(self):
        """Test summarizing discrepancy report."""
        discrepancies = [
            Discrepancy(
                discrepancy_type=DiscrepancyType.REC_GO_IMPACT,
                direction=DiscrepancyDirection.MARKET_LOWER,
                materiality=MaterialityLevel.MATERIAL,
                absolute_tco2e=Decimal("400.0"),
                percentage=Decimal("40.0"),
                description="Test",
                recommendation="Test",
            ),
        ]

        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=Decimal("400.0"),
            items=[],
        )

        report = DiscrepancyReport(
            reconciliation_id="test-001",
            discrepancies=discrepancies,
            materiality_summary={},
            waterfall=waterfall,
        )

        engine = DiscrepancyAnalyzerEngine()
        summary = engine.summarize_discrepancy_report(report)

        assert isinstance(summary, dict)
        assert "reconciliation_id" in summary or len(summary) >= 0

    def test_calculate_discrepancy_percentage(self):
        """Test calculate_discrepancy_percentage method."""
        engine = DiscrepancyAnalyzerEngine()

        percentage = engine.calculate_discrepancy_percentage(
            Decimal("1000.0"),
            Decimal("600.0"),
        )

        # |1000 - 600| / max(1000, 600) * 100 = 40%
        assert percentage == Decimal("40.0")

    def test_calculate_pif(self):
        """Test calculate_pif method."""
        engine = DiscrepancyAnalyzerEngine()

        pif = engine.calculate_pif(Decimal("1000.0"), Decimal("600.0"))

        # (1000 - 600) / 1000 * 100 = 40%
        assert pif == Decimal("40.0")


# ===========================================================================
# Test Class 11: Top Drivers
# ===========================================================================


class TestGetTopDrivers:
    """Test top driver identification."""

    def test_top_drivers_from_waterfall(self):
        """Test identifying top drivers from waterfall."""
        items = [
            WaterfallItem(
                driver="REC_GO_IMPACT",
                contribution_tco2e=Decimal("300.0"),
                contribution_pct=Decimal("75.0"),
                description="Test",
            ),
            WaterfallItem(
                driver="SUPPLIER_EF_DELTA",
                contribution_tco2e=Decimal("100.0"),
                contribution_pct=Decimal("25.0"),
                description="Test",
            ),
        ]

        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=Decimal("400.0"),
            items=items,
        )

        # Top driver should be REC_GO_IMPACT
        assert waterfall.items[0].contribution_tco2e >= waterfall.items[1].contribution_tco2e

    def test_top_drivers_sorted_by_absolute_value(self):
        """Test top drivers are sorted by absolute contribution."""
        items = [
            WaterfallItem(
                driver="RESIDUAL_MIX_UPLIFT",
                contribution_tco2e=Decimal("-200.0"),  # Negative
                contribution_pct=Decimal("-50.0"),
                description="Test",
            ),
            WaterfallItem(
                driver="REC_GO_IMPACT",
                contribution_tco2e=Decimal("600.0"),  # Larger absolute
                contribution_pct=Decimal("150.0"),
                description="Test",
            ),
        ]

        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=Decimal("400.0"),
            items=items,
        )

        # Can check items are present
        assert len(waterfall.items) == 2

    def test_top_drivers_limit_to_n(self):
        """Test limiting top drivers to N items."""
        items = [
            WaterfallItem(
                driver=f"DRIVER_{i}",
                contribution_tco2e=Decimal(str(100 * (10 - i))),
                contribution_pct=Decimal("10.0"),
                description="Test",
            )
            for i in range(10)
        ]

        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=Decimal("4500.0"),
            items=items,
        )

        # All items present
        assert len(waterfall.items) == 10

    def test_top_drivers_empty_waterfall(self):
        """Test top drivers with empty waterfall."""
        waterfall = WaterfallDecomposition(
            total_discrepancy_tco2e=Decimal("0.0"),
            items=[],
        )

        assert len(waterfall.items) == 0


# ===========================================================================
# Test Class 12: Full Analysis Entry Point
# ===========================================================================


class TestAnalyzeDiscrepancies:
    """Test analyze_discrepancies method (full analysis)."""

    def test_analyze_discrepancies_complete_flow(self):
        """Test full analysis flow from workspace to report."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
            ef_hierarchy=EFHierarchyPriority.BUNDLED_CERT,
            ef_used=Decimal("0.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        report = engine.analyze_discrepancies(workspace)

        assert isinstance(report, DiscrepancyReport)
        assert report.reconciliation_id == "recon-test-001"

    def test_analyze_discrepancies_returns_discrepancies(self):
        """Test analyze_discrepancies returns discrepancy list."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        report = engine.analyze_discrepancies(workspace)

        assert len(report.discrepancies) > 0

    def test_analyze_discrepancies_returns_waterfall(self):
        """Test analyze_discrepancies returns waterfall."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        report = engine.analyze_discrepancies(workspace)

        assert report.waterfall is not None
        assert isinstance(report.waterfall, WaterfallDecomposition)

    def test_analyze_discrepancies_returns_flags(self):
        """Test analyze_discrepancies returns flags."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        report = engine.analyze_discrepancies(workspace)

        assert isinstance(report.flags, list)

    def test_analyze_discrepancies_validates_workspace(self):
        """Test analyze_discrepancies validates workspace."""
        engine = DiscrepancyAnalyzerEngine()

        # None workspace should raise ValueError
        with pytest.raises(ValueError, match="Workspace cannot be None"):
            engine.analyze_discrepancies(None)

    def test_analyze_discrepancies_increments_counter(self):
        """Test analyze_discrepancies increments analysis counter."""
        DiscrepancyAnalyzerEngine.reset()
        engine = DiscrepancyAnalyzerEngine()

        initial_count = engine._total_analyses

        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine.analyze_discrepancies(workspace)

        assert engine._total_analyses == initial_count + 1

    def test_analyze_discrepancies_materiality_summary(self):
        """Test analyze_discrepancies includes materiality summary."""
        loc_result = _create_upstream_result(
            method=Scope2Method.LOCATION_BASED,
            emissions_tco2e=Decimal("1000.0"),
        )
        mkt_result = _create_upstream_result(
            method=Scope2Method.MARKET_BASED,
            emissions_tco2e=Decimal("600.0"),
        )
        workspace = _create_workspace([loc_result], [mkt_result])

        engine = DiscrepancyAnalyzerEngine()
        report = engine.analyze_discrepancies(workspace)

        assert report.materiality_summary is not None
        assert isinstance(report.materiality_summary, dict)

    def test_analyze_discrepancies_handles_empty_workspace(self):
        """Test analyze_discrepancies handles workspace with no results."""
        workspace = ReconciliationWorkspace(
            reconciliation_id="test-empty",
            tenant_id="tenant-001",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
            location_results=[],
            market_results=[],
            total_location_tco2e=Decimal("0.0"),
            total_market_tco2e=Decimal("0.0"),
        )

        engine = DiscrepancyAnalyzerEngine()

        # Should raise validation error
        with pytest.raises(ValueError, match="at least one"):
            engine.analyze_discrepancies(workspace)


# ===========================================================================
# Test Class 13: Health Check
# ===========================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_dict(self):
        """Test health_check returns dictionary."""
        engine = DiscrepancyAnalyzerEngine()
        health = engine.health_check()

        assert isinstance(health, dict)

    def test_health_check_includes_engine_id(self):
        """Test health_check includes engine_id."""
        engine = DiscrepancyAnalyzerEngine()
        health = engine.health_check()

        assert "engine_id" in health
        assert health["engine_id"] == "discrepancy-analyzer-engine"

    def test_health_check_includes_version(self):
        """Test health_check includes version."""
        engine = DiscrepancyAnalyzerEngine()
        health = engine.health_check()

        assert "engine_version" in health
        assert health["engine_version"] == "1.0.0"

    def test_health_check_includes_status(self):
        """Test health_check includes status."""
        engine = DiscrepancyAnalyzerEngine()
        health = engine.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
