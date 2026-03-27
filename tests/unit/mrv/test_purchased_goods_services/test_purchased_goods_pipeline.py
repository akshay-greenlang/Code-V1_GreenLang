"""
Unit tests for PurchasedGoodsPipelineEngine (AGENT-MRV-014).

Tests cover:
- Singleton pattern
- Individual pipeline stages
- Full pipeline execution
- Batch processing
- Export functionality
- Input validation
- Error handling
- Health checks
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import csv

try:
    from greenlang.agents.mrv.purchased_goods_services.purchased_goods_pipeline import (
        PurchasedGoodsPipelineEngine,
        PipelineInput,
        PipelineOutput,
        PipelineStage,
        StageResult,
        ExportFormat,
    )
except ImportError:
    pytest.skip("PurchasedGoodsPipelineEngine not available", allow_module_level=True)


class TestPurchasedGoodsPipelineSingleton:
    """Test singleton pattern for PurchasedGoodsPipelineEngine."""

    def test_singleton_same_instance(self):
        """Test that get_instance returns same instance."""
        engine1 = PurchasedGoodsPipelineEngine.get_instance()
        engine2 = PurchasedGoodsPipelineEngine.get_instance()
        assert engine1 is engine2

    def test_singleton_thread_safe(self):
        """Test thread-safe singleton creation."""
        import threading
        instances = []

        def get_instance():
            instances.append(PurchasedGoodsPipelineEngine.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_singleton_reset(self):
        """Test singleton reset functionality."""
        engine1 = PurchasedGoodsPipelineEngine.get_instance()
        PurchasedGoodsPipelineEngine.reset_instance()
        engine2 = PurchasedGoodsPipelineEngine.get_instance()
        assert engine1 is not engine2


class TestPipelineStages:
    """Test individual pipeline stages."""

    def test_stage1_data_validation(self):
        """Test Stage 1: Data validation and cleaning."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            {"id": "P002", "spend": Decimal("-500"), "category": "packaging"},  # Invalid
        ]

        result = engine.execute_stage_1_validation(items)

        assert result["valid_items"] == 1
        assert result["invalid_items"] == 1
        assert len(result["cleaned_data"]) == 1

    def test_stage2_categorization(self):
        """Test Stage 2: Spend categorization."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "description": "Steel rebar", "spend": Decimal("10000")},
            {"id": "P002", "description": "Packaging materials", "spend": Decimal("5000")},
        ]

        result = engine.execute_stage_2_categorization(items)

        assert all("category" in item for item in result["categorized_items"])

    def test_stage3_boundary_enforcement(self):
        """Test Stage 3: Boundary enforcement."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "category": "raw_materials", "spend": Decimal("10000")},
            {"id": "P002", "category": "capital_goods", "spend": Decimal("50000")},
            {"id": "P003", "category": "fuel_energy", "spend": Decimal("20000")},
        ]

        result = engine.execute_stage_3_boundary(items)

        assert result["included_items"] == 1  # Only P001
        assert result["excluded_items"] == 2  # P002, P003

    def test_stage4_method_selection(self):
        """Test Stage 4: Method selection for each item."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "has_supplier_data": True, "has_avgdata": True},
            {"id": "P002", "has_supplier_data": False, "has_avgdata": True},
            {"id": "P003", "has_supplier_data": False, "has_avgdata": False},
        ]

        result = engine.execute_stage_4_method_selection(items)

        methods = {item["id"]: item["selected_method"] for item in result["items_with_methods"]}
        assert methods["P001"] == "supplier"
        assert methods["P002"] == "avgdata"
        assert methods["P003"] == "spend"

    def test_stage5_calculation(self):
        """Test Stage 5: Emission calculations."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "method": "spend", "spend": Decimal("10000"), "ef": Decimal("0.5")},
            {"id": "P002", "method": "supplier", "quantity": Decimal("1000"), "ef": Decimal("2.0")},
        ]

        result = engine.execute_stage_5_calculation(items)

        assert all("emissions" in item for item in result["calculated_items"])
        assert result["total_emissions"] > 0

    def test_stage6_aggregation(self):
        """Test Stage 6: Result aggregation."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "supplier_id": "S001", "emissions": Decimal("100")},
            {"id": "P002", "supplier_id": "S001", "emissions": Decimal("150")},
            {"id": "P003", "supplier_id": "S002", "emissions": Decimal("200")},
        ]

        result = engine.execute_stage_6_aggregation(items)

        assert "S001" in result["by_supplier"]
        assert result["by_supplier"]["S001"] == Decimal("250")
        assert result["by_supplier"]["S002"] == Decimal("200")

    def test_stage7_dqi_calculation(self):
        """Test Stage 7: DQI calculation."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "method": "supplier", "emissions": Decimal("1000"), "dqi": Decimal("1.5")},
            {"id": "P002", "method": "spend", "emissions": Decimal("500"), "dqi": Decimal("3.0")},
        ]

        result = engine.execute_stage_7_dqi(items)

        # Weighted DQI = (1000*1.5 + 500*3.0) / 1500 = 2.0
        assert result["weighted_dqi"] == Decimal("2.0")

    def test_stage8_coverage_analysis(self):
        """Test Stage 8: Coverage analysis."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        data = {
            "total_spend": Decimal("1000000"),
            "covered_spend": Decimal("750000"),
            "total_items": 100,
            "covered_items": 80,
        }

        result = engine.execute_stage_8_coverage(data)

        assert result["spend_coverage"] == Decimal("75.0")
        assert result["item_coverage"] == Decimal("80.0")

    def test_stage9_hotspot_analysis(self):
        """Test Stage 9: Hot-spot analysis."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "emissions": Decimal("5000")},
            {"id": "P002", "emissions": Decimal("3000")},
            {"id": "P003", "emissions": Decimal("1500")},
            {"id": "P004", "emissions": Decimal("500")},
        ]

        result = engine.execute_stage_9_hotspots(items)

        assert len(result["top_emitters"]) > 0
        assert result["top_emitters"][0]["id"] == "P001"

    def test_stage10_compliance_check(self):
        """Test Stage 10: Compliance checking."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = engine.execute_stage_10_compliance(data)

        assert "ghg_protocol" in result["framework_checks"]
        assert "csrd_esrs" in result["framework_checks"]


class TestFullPipeline:
    """Test full pipeline execution."""

    def test_full_pipeline_hybrid_method(self):
        """Test full pipeline with hybrid method."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials", "has_supplier_data": True},
                {"id": "P002", "spend": Decimal("8000"), "category": "packaging", "has_avgdata": True},
                {"id": "P003", "spend": Decimal("5000"), "category": "components"},
            ],
            "reporting_year": 2024,
        }

        result = engine.run_pipeline(input_data)

        assert "total_emissions" in result
        assert "coverage" in result
        assert "dqi" in result
        assert len(result["stages"]) == 10

    def test_full_pipeline_spend_only(self):
        """Test full pipeline with spend-based method only."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
                {"id": "P002", "spend": Decimal("8000"), "category": "packaging"},
            ],
            "reporting_year": 2024,
            "force_method": "spend",
        }

        result = engine.run_pipeline(input_data)

        assert all(item["method"] == "spend" for item in result["item_results"])

    def test_full_pipeline_supplier_only(self):
        """Test full pipeline with supplier-specific method only."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "quantity": Decimal("1000"), "ef": Decimal("2.0"), "has_supplier_data": True},
                {"id": "P002", "quantity": Decimal("500"), "ef": Decimal("1.5"), "has_supplier_data": True},
            ],
            "reporting_year": 2024,
            "force_method": "supplier",
        }

        result = engine.run_pipeline(input_data)

        assert all(item["method"] == "supplier" for item in result["item_results"])

    def test_pipeline_with_exclusions(self):
        """Test pipeline correctly excludes boundary items."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
                {"id": "P002", "spend": Decimal("50000"), "category": "capital_goods"},
            ],
            "reporting_year": 2024,
        }

        result = engine.run_pipeline(input_data)

        # Only P001 should be included
        assert len(result["item_results"]) == 1
        assert result["item_results"][0]["id"] == "P001"

    def test_pipeline_stage_execution_order(self):
        """Test pipeline stages execute in correct order."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            ],
            "reporting_year": 2024,
        }

        result = engine.run_pipeline(input_data)

        stage_order = [stage["stage_number"] for stage in result["stages"]]
        assert stage_order == list(range(1, 11))


class TestBatchProcessing:
    """Test batch processing capabilities."""

    def test_multi_period_batch(self):
        """Test batch processing across multiple periods."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        periods = [
            {
                "period": "2024-Q1",
                "items": [{"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"}],
            },
            {
                "period": "2024-Q2",
                "items": [{"id": "P001", "spend": Decimal("12000"), "category": "raw_materials"}],
            },
        ]

        results = engine.run_batch(periods)

        assert len(results) == 2
        assert results[0]["period"] == "2024-Q1"
        assert results[1]["period"] == "2024-Q2"

    def test_error_handling_in_batch(self):
        """Test error handling in batch processing."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        periods = [
            {
                "period": "2024-Q1",
                "items": [{"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"}],
            },
            {
                "period": "2024-Q2",
                "items": [],  # Empty items - should error
            },
            {
                "period": "2024-Q3",
                "items": [{"id": "P002", "spend": Decimal("15000"), "category": "packaging"}],
            },
        ]

        results = engine.run_batch(periods, continue_on_error=True)

        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "success"

    def test_batch_parallel_execution(self):
        """Test parallel execution of batch items."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        periods = [
            {
                "period": f"2024-Q{i}",
                "items": [{"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"}],
            }
            for i in range(1, 5)
        ]

        import time
        start = time.time()
        results = engine.run_batch(periods, parallel=True, max_workers=4)
        parallel_time = time.time() - start

        assert len(results) == 4
        # Parallel execution should be faster than 4x single execution
        # (but we won't assert timing in tests due to variability)

    def test_batch_aggregation(self):
        """Test aggregation across batch results."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        batch_results = [
            {"period": "2024-Q1", "total_emissions": Decimal("5000")},
            {"period": "2024-Q2", "total_emissions": Decimal("6000")},
            {"period": "2024-Q3", "total_emissions": Decimal("5500")},
        ]

        aggregated = engine.aggregate_batch_results(batch_results)

        assert aggregated["total_emissions"] == Decimal("16500")
        assert aggregated["average_emissions"] == Decimal("5500")


class TestExportFunctionality:
    """Test export functionality."""

    def test_export_to_json(self):
        """Test export to JSON format."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        result = {
            "total_emissions": Decimal("10000"),
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        engine.export_to_json(result, output_path)

        with open(output_path, 'r') as f:
            loaded = json.load(f)

        assert float(loaded["total_emissions"]) == 10000.0
        assert float(loaded["coverage"]) == 95.0

    def test_export_to_csv(self):
        """Test export to CSV format."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        items = [
            {"id": "P001", "emissions": Decimal("100"), "method": "supplier"},
            {"id": "P002", "emissions": Decimal("200"), "method": "spend"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name

        engine.export_to_csv(items, output_path)

        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["id"] == "P001"

    def test_export_to_excel(self):
        """Test export to Excel format."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        result = {
            "total_emissions": Decimal("10000"),
            "items": [
                {"id": "P001", "emissions": Decimal("100")},
                {"id": "P002", "emissions": Decimal("200")},
            ],
        }

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            output_path = f.name

        engine.export_to_excel(result, output_path)

        # Verify file exists and is non-empty
        import os
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


class TestInputValidation:
    """Test input validation."""

    def test_valid_input_accepted(self):
        """Test valid input is accepted."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            ],
            "reporting_year": 2024,
        }

        validation = engine.validate_input(input_data)

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_empty_items_list_rejected(self):
        """Test empty items list is rejected."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [],
            "reporting_year": 2024,
        }

        validation = engine.validate_input(input_data)

        assert validation["valid"] is False
        assert any("empty" in error.lower() for error in validation["errors"])

    def test_too_many_items_rejected(self):
        """Test excessive items count is rejected."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": f"P{i:05d}", "spend": Decimal("100"), "category": "raw_materials"}
                for i in range(100001)  # Over limit
            ],
            "reporting_year": 2024,
        }

        validation = engine.validate_input(input_data)

        assert validation["valid"] is False
        assert any("too many" in error.lower() or "limit" in error.lower() for error in validation["errors"])

    def test_missing_required_fields(self):
        """Test missing required fields are detected."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001"},  # Missing spend/quantity
            ],
            "reporting_year": 2024,
        }

        validation = engine.validate_input(input_data)

        assert validation["valid"] is False


class TestErrorHandling:
    """Test error handling."""

    def test_stage_failure_isolation(self):
        """Test failure in one stage doesn't crash pipeline."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        with patch.object(engine, 'execute_stage_5_calculation', side_effect=Exception("Calculation error")):
            input_data = {
                "items": [
                    {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
                ],
                "reporting_year": 2024,
                "continue_on_error": True,
            }

            result = engine.run_pipeline(input_data)

            # Pipeline should complete but flag error
            assert "errors" in result
            assert any("stage_5" in error.lower() for error in result["errors"])

    def test_invalid_method_fallback(self):
        """Test fallback when invalid method specified."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            ],
            "reporting_year": 2024,
            "force_method": "invalid_method",
        }

        result = engine.run_pipeline(input_data)

        # Should fall back to available method
        assert result["item_results"][0]["method"] in ["supplier", "avgdata", "spend"]

    def test_data_type_error_handling(self):
        """Test handling of incorrect data types."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        input_data = {
            "items": [
                {"id": "P001", "spend": "not_a_number", "category": "raw_materials"},
            ],
            "reporting_year": 2024,
        }

        with pytest.raises((ValueError, TypeError)):
            engine.run_pipeline(input_data)


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(self):
        """Test health check returns healthy status."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert "engine" in health
        assert health["engine"] == "PurchasedGoodsPipelineEngine"

    def test_health_check_includes_stats(self):
        """Test health check includes statistics."""
        engine = PurchasedGoodsPipelineEngine.get_instance()

        # Run a pipeline
        input_data = {
            "items": [
                {"id": "P001", "spend": Decimal("10000"), "category": "raw_materials"},
            ],
            "reporting_year": 2024,
        }
        engine.run_pipeline(input_data)

        health = engine.health_check()

        assert "pipelines_executed" in health
        assert health["pipelines_executed"] > 0
