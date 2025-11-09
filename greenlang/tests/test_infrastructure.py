"""
GreenLang Core Infrastructure - Test Suite
===========================================

Tests for core infrastructure modules:
1. Validation Framework - Schema validation
2. Cache Manager - Multi-tier caching
3. Telemetry - Metrics collection
4. Provenance Tracker - Data lineage
5. Agent Templates - Batch processing

Version: 1.0.0
Author: Testing & QA Team
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Validation Framework Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.critical
class TestValidationFrameworkSchema:
    """Test validation framework schema validation."""

    def test_validation_framework_schema(self):
        """Test validation framework validates data against JSON schemas."""
        from greenlang.infrastructure.validation import ValidationFramework

        validator = ValidationFramework()

        schema = {
            "type": "object",
            "properties": {
                "emissions": {"type": "number", "minimum": 0},
                "company": {"type": "string", "minLength": 1}
            },
            "required": ["emissions", "company"]
        }

        # Valid data
        valid_data = {"emissions": 1000, "company": "Test Corp"}
        result = validator.validate(valid_data, schema)
        assert result.is_valid
        assert len(result.errors) == 0

        # Invalid data
        invalid_data = {"emissions": -100}  # Negative, missing company
        result = validator.validate(invalid_data, schema)
        assert not result.is_valid
        assert len(result.errors) > 0


# ============================================================================
# Cache Manager Tests
# ============================================================================

@pytest.mark.infrastructure
class TestCacheManagerGetOrCompute:
    """Test cache manager get_or_compute pattern."""

    def test_cache_manager_get_or_compute(self):
        """Test cache manager get_or_compute pattern for lazy evaluation."""
        from greenlang.infrastructure.cache import CacheManager

        cache = CacheManager()

        call_count = 0

        def expensive_computation():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return "computed_value"

        # First call - should compute
        result1 = cache.get_or_compute("test_key", expensive_computation)
        assert result1 == "computed_value"
        assert call_count == 1

        # Second call - should use cache
        result2 = cache.get_or_compute("test_key", expensive_computation)
        assert result2 == "computed_value"
        assert call_count == 1  # Not called again


# ============================================================================
# Telemetry Tests
# ============================================================================

@pytest.mark.infrastructure
class TestTelemetryMetricsCollection:
    """Test telemetry metrics collection."""

    def test_telemetry_metrics_collection(self):
        """Test telemetry collects and aggregates metrics."""
        from greenlang.infrastructure.telemetry import TelemetryCollector

        telemetry = TelemetryCollector()

        # Record metrics
        telemetry.record("agent_execution_time", 1.5, tags={"agent": "intake"})
        telemetry.record("llm_tokens", 1000, tags={"model": "gpt-4"})

        # Get metrics
        metrics = telemetry.get_metrics()
        assert len(metrics) >= 2


# ============================================================================
# Provenance Tracker Tests
# ============================================================================

@pytest.mark.infrastructure
@pytest.mark.critical
class TestProvenanceTrackerLineage:
    """Test provenance tracker data lineage."""

    def test_provenance_tracker_lineage(self):
        """Test provenance tracker maintains data lineage."""
        from greenlang.infrastructure.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()

        # Track data transformation
        tracker.record_transformation(
            input_data={"raw_value": 100},
            output_data={"processed_value": 105},
            transformation="apply_correction_factor",
            metadata={"factor": 1.05}
        )

        # Get lineage
        lineage = tracker.get_lineage(data_id="processed_value")
        assert lineage is not None


# ============================================================================
# Agent Templates Tests
# ============================================================================

@pytest.mark.infrastructure
class TestAgentTemplatesBatchProcessing:
    """Test agent templates batch processing."""

    def test_agent_templates_batch_processing(self):
        """Test agent templates support efficient batch processing."""
        from greenlang.infrastructure.agent_templates import BatchAgent

        agent = BatchAgent()

        # Process batch
        batch_data = [{"value": i} for i in range(100)]
        results = agent.process_batch(batch_data, batch_size=10)

        assert len(results) == 100


# ============================================================================
# Mock Infrastructure Classes
# ============================================================================

try:
    from greenlang.infrastructure.validation import ValidationFramework
except ImportError:
    class ValidationFramework:
        def validate(self, data, schema):
            from collections import namedtuple
            Result = namedtuple('Result', ['is_valid', 'errors'])

            # Simple mock validation
            errors = []
            if 'required' in schema:
                for field in schema['required']:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")

            for field, value in data.items():
                if field in schema.get('properties', {}):
                    field_schema = schema['properties'][field]
                    if field_schema.get('type') == 'number':
                        if not isinstance(value, (int, float)):
                            errors.append(f"Field {field} must be number")
                        if 'minimum' in field_schema and value < field_schema['minimum']:
                            errors.append(f"Field {field} below minimum")

            return Result(is_valid=len(errors) == 0, errors=errors)


try:
    from greenlang.infrastructure.cache import CacheManager
except ImportError:
    class CacheManager:
        def __init__(self):
            self.cache = {}

        def get_or_compute(self, key, compute_fn):
            if key in self.cache:
                return self.cache[key]
            value = compute_fn()
            self.cache[key] = value
            return value


try:
    from greenlang.infrastructure.telemetry import TelemetryCollector
except ImportError:
    class TelemetryCollector:
        def __init__(self):
            self.metrics = []

        def record(self, name, value, tags=None):
            self.metrics.append({'name': name, 'value': value, 'tags': tags or {}})

        def get_metrics(self):
            return self.metrics


try:
    from greenlang.infrastructure.provenance import ProvenanceTracker
except ImportError:
    class ProvenanceTracker:
        def __init__(self):
            self.lineage = {}

        def record_transformation(self, input_data, output_data, transformation, metadata=None):
            self.lineage[str(output_data)] = {
                'input': input_data,
                'transformation': transformation,
                'metadata': metadata
            }

        def get_lineage(self, data_id):
            return self.lineage.get(data_id, {})


try:
    from greenlang.infrastructure.agent_templates import BatchAgent
except ImportError:
    class BatchAgent:
        def process_batch(self, data, batch_size=10):
            return [{"processed": item} for item in data]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'infrastructure'])
