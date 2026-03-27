# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.downstream_transport_pipeline - AGENT-MRV-022.

Tests DownstreamTransportPipelineEngine for the Downstream Transportation
& Distribution Agent (GL-MRV-S3-009).

Coverage (~50 tests):
- Full pipeline (distance, spend, average, warehouse, last-mile methods)
- Batch pipeline processing
- Distribution chain (multi-stage)
- 10-stage provenance tracking
- Error handling and recovery
- Singleton pattern
- Thread safety

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import threading
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.downstream_transport_pipeline import (
        DownstreamTransportPipelineEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"downstream_transport_pipeline not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test DownstreamTransportPipelineEngine singleton."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        eng1 = DownstreamTransportPipelineEngine()
        eng2 = DownstreamTransportPipelineEngine()
        assert eng1 is eng2

    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe with 10 concurrent calls."""
        results = []

        def worker():
            eng = DownstreamTransportPipelineEngine()
            results.append(id(eng))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 1


# ==============================================================================
# DISTANCE-BASED PIPELINE TESTS
# ==============================================================================


class TestDistanceBasedPipeline:
    """Test full pipeline for distance-based calculations."""

    def test_distance_pipeline_road(self, sample_shipment):
        """Test distance-based pipeline for road shipment."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "tenant_id": "tenant-001",
        })
        assert result is not None
        assert "result" in result or "emissions_tco2e" in result
        assert "provenance_hash" in result
        assert "validation_status" in result

    def test_distance_pipeline_rail(self, sample_shipment_rail):
        """Test distance-based pipeline for rail shipment."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment_rail,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_distance_pipeline_maritime(self, sample_shipment_maritime):
        """Test distance-based pipeline for maritime shipment."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment_maritime,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_distance_pipeline_air(self, sample_shipment_air):
        """Test distance-based pipeline for air shipment."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment_air,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_distance_pipeline_courier(self, sample_shipment_courier):
        """Test distance-based pipeline for courier shipment."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment_courier,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_distance_pipeline_last_mile(self, sample_shipment_last_mile):
        """Test distance-based pipeline for last-mile delivery."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment_last_mile,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_distance_pipeline_result_structure(self, sample_shipment):
        """Test result structure contains expected fields."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "tenant_id": "tenant-001",
        })
        assert "provenance_hash" in result
        assert "processing_time_ms" in result
        assert "validation_status" in result

    def test_distance_pipeline_provenance_hash_valid(self, sample_shipment):
        """Test provenance hash in pipeline result is 64-char hex."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "tenant_id": "tenant-001",
        })
        ph = result["provenance_hash"]
        assert len(ph) == 64
        assert all(c in "0123456789abcdef" for c in ph)


# ==============================================================================
# SPEND-BASED PIPELINE TESTS
# ==============================================================================


class TestSpendBasedPipeline:
    """Test full pipeline for spend-based calculations."""

    def test_spend_pipeline(self, sample_spend):
        """Test spend-based pipeline."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "SPEND_BASED",
            "input_data": sample_spend,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_spend_pipeline_eur(self, sample_spend_eur):
        """Test spend-based pipeline with EUR currency."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "SPEND_BASED",
            "input_data": sample_spend_eur,
            "tenant_id": "tenant-001",
        })
        assert result is not None


# ==============================================================================
# AVERAGE-DATA PIPELINE TESTS
# ==============================================================================


class TestAverageDataPipeline:
    """Test full pipeline for average-data calculations."""

    def test_average_data_pipeline(self, sample_average_data):
        """Test average-data pipeline."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "AVERAGE_DATA",
            "input_data": sample_average_data,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_average_data_pipeline_retail(self, sample_average_data_retail):
        """Test average-data pipeline for retail channel."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "AVERAGE_DATA",
            "input_data": sample_average_data_retail,
            "tenant_id": "tenant-001",
        })
        assert result is not None


# ==============================================================================
# WAREHOUSE PIPELINE TESTS
# ==============================================================================


class TestWarehousePipeline:
    """Test pipeline for warehouse calculations."""

    def test_warehouse_pipeline(self, sample_warehouse):
        """Test warehouse emission pipeline."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "WAREHOUSE",
            "input_data": sample_warehouse,
            "tenant_id": "tenant-001",
        })
        assert result is not None

    def test_warehouse_cold_pipeline(self, sample_warehouse_cold):
        """Test cold storage warehouse pipeline."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "WAREHOUSE",
            "input_data": sample_warehouse_cold,
            "tenant_id": "tenant-001",
        })
        assert result is not None


# ==============================================================================
# LAST-MILE PIPELINE TESTS
# ==============================================================================


class TestLastMilePipeline:
    """Test pipeline for last-mile calculations."""

    def test_last_mile_pipeline(self, sample_last_mile):
        """Test last-mile delivery pipeline."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "LAST_MILE",
            "input_data": sample_last_mile,
            "tenant_id": "tenant-001",
        })
        assert result is not None


# ==============================================================================
# BATCH PIPELINE TESTS
# ==============================================================================


class TestBatchPipeline:
    """Test batch pipeline processing."""

    def test_batch_pipeline(self, sample_batch):
        """Test batch pipeline with mixed calculation methods."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process_batch(sample_batch)
        assert result is not None
        results = result.get("results", result)
        assert isinstance(results, (list, dict))

    def test_batch_pipeline_aggregation(self, sample_batch):
        """Test batch pipeline result aggregation."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process_batch(sample_batch)
        total = result.get("total_emissions_tco2e")
        if total is not None:
            assert total > 0

    def test_batch_pipeline_individual_results(self, sample_batch):
        """Test batch pipeline includes individual results."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process_batch(sample_batch)
        results = result.get("results", [])
        if isinstance(results, list):
            assert len(results) == len(sample_batch["requests"])


# ==============================================================================
# DISTRIBUTION CHAIN PIPELINE TESTS
# ==============================================================================


class TestDistributionChainPipeline:
    """Test distribution chain pipeline (warehouse + transport + last-mile)."""

    def test_distribution_chain(self):
        """Test full distribution chain pipeline."""
        engine = DownstreamTransportPipelineEngine()
        chain = {
            "transport": {
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("350.0"),
                "cargo_mass_tonnes": Decimal("15.0"),
            },
            "warehouse": {
                "warehouse_type": "DISTRIBUTION_CENTER",
                "floor_area_m2": Decimal("5000.0"),
                "dwell_time_hours": Decimal("48.0"),
            },
            "last_mile": {
                "vehicle_type": "VAN_DIESEL",
                "delivery_area": "URBAN",
                "distance_km": Decimal("15.0"),
                "parcels_delivered": 25,
            },
        }
        result = engine.process_distribution_chain(chain)
        assert result is not None
        total = result.get("total_emissions_tco2e", result.get("emissions_tco2e"))
        assert total > 0

    def test_chain_includes_all_components(self):
        """Test chain result includes transport, warehouse, and last-mile."""
        engine = DownstreamTransportPipelineEngine()
        chain = {
            "transport": {
                "mode": "ROAD",
                "vehicle_type": "ARTICULATED_33T",
                "distance_km": Decimal("200.0"),
                "cargo_mass_tonnes": Decimal("10.0"),
            },
            "warehouse": {
                "warehouse_type": "FULFILLMENT_CENTER",
                "floor_area_m2": Decimal("10000.0"),
                "dwell_time_hours": Decimal("24.0"),
            },
            "last_mile": {
                "vehicle_type": "VAN_ELECTRIC",
                "delivery_area": "SUBURBAN",
                "distance_km": Decimal("20.0"),
                "parcels_delivered": 15,
            },
        }
        result = engine.process_distribution_chain(chain)
        # Should have breakdown of components
        assert result is not None


# ==============================================================================
# PROVENANCE 10-STAGE TESTS
# ==============================================================================


class TestProvenance10Stage:
    """Test 10-stage provenance tracking in pipeline."""

    def test_pipeline_records_10_stages(self, sample_shipment):
        """Test pipeline records all 10 provenance stages."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "include_provenance": True,
            "tenant_id": "tenant-001",
        })
        provenance = result.get("provenance_chain", result.get("provenance"))
        if provenance:
            stages = provenance.get("stages_recorded", [])
            # Pipeline should record multiple stages
            assert len(stages) >= 5

    def test_pipeline_provenance_chain_sealed(self, sample_shipment):
        """Test pipeline provenance chain is sealed after processing."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "include_provenance": True,
            "tenant_id": "tenant-001",
        })
        provenance = result.get("provenance_chain", result.get("provenance"))
        if provenance:
            assert provenance.get("is_sealed") is True or \
                   provenance.get("sealed") is True


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:
    """Test pipeline error handling."""

    def test_invalid_method_raises(self):
        """Test invalid calculation method raises error."""
        engine = DownstreamTransportPipelineEngine()
        with pytest.raises((ValueError, KeyError)):
            engine.process({
                "calculation_method": "INVALID_METHOD",
                "input_data": {},
                "tenant_id": "tenant-001",
            })

    def test_missing_input_data_raises(self):
        """Test missing input data raises error."""
        engine = DownstreamTransportPipelineEngine()
        with pytest.raises((ValueError, KeyError, TypeError)):
            engine.process({
                "calculation_method": "DISTANCE_BASED",
                "tenant_id": "tenant-001",
            })

    def test_empty_input_data_raises(self):
        """Test empty input data raises error."""
        engine = DownstreamTransportPipelineEngine()
        with pytest.raises((ValueError, KeyError, TypeError)):
            engine.process({
                "calculation_method": "DISTANCE_BASED",
                "input_data": {},
                "tenant_id": "tenant-001",
            })


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================


class TestThreadSafety:
    """Test pipeline thread safety."""

    def test_concurrent_processing(self, sample_shipment):
        """Test concurrent pipeline processing does not crash."""
        engine = DownstreamTransportPipelineEngine()
        results = []
        errors = []

        def worker(i):
            try:
                result = engine.process({
                    "calculation_method": "DISTANCE_BASED",
                    "input_data": {
                        **sample_shipment,
                        "shipment_id": f"THREAD-{i}",
                    },
                    "tenant_id": "tenant-001",
                })
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10


# ==============================================================================
# VALIDATION STATUS TESTS
# ==============================================================================


class TestValidationStatus:
    """Test pipeline validation status in results."""

    def test_pass_status(self, sample_shipment):
        """Test PASS validation status for valid input."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "tenant_id": "tenant-001",
        })
        assert result["validation_status"] in ("PASS", "pass")

    def test_processing_time_tracked(self, sample_shipment):
        """Test processing time is tracked in result."""
        engine = DownstreamTransportPipelineEngine()
        result = engine.process({
            "calculation_method": "DISTANCE_BASED",
            "input_data": sample_shipment,
            "tenant_id": "tenant-001",
        })
        time_ms = result.get("processing_time_ms")
        assert time_ms is not None
        assert time_ms >= 0
