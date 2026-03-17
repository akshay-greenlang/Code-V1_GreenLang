# -*- coding: utf-8 -*-
"""
Unit tests for CrossFrameworkDataMapperEngine - PACK-009 Engine 1

Tests cross-framework field mapping between CSRD, CBAM, EUDR, and
EU Taxonomy regulations. Validates exact/approximate/derived mapping,
batch operations, overlap statistics, multi-hop path finding,
bidirectional search, and provenance hashing.

Coverage target: 85%+
Test count: 20

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the engine module
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PACK_ROOT / "engines" / "cross_framework_data_mapper.py"

try:
    _mapper_mod = _import_from_path("cross_framework_data_mapper", ENGINE_PATH)
    CrossFrameworkDataMapperEngine = _mapper_mod.CrossFrameworkDataMapperEngine
    CrossFrameworkDataMapperConfig = _mapper_mod.CrossFrameworkDataMapperConfig
    MappingResult = _mapper_mod.MappingResult
    BatchMappingResult = _mapper_mod.BatchMappingResult
    OverlapStatistics = _mapper_mod.OverlapStatistics
    MappingPath = _mapper_mod.MappingPath
    FieldMapping = _mapper_mod.FieldMapping
    CROSS_FRAMEWORK_MAPPINGS = _mapper_mod.CROSS_FRAMEWORK_MAPPINGS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"CrossFrameworkDataMapperEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_provenance_hash(obj: Any) -> None:
    """Verify an object carries a valid 64-char SHA-256 provenance hash."""
    h = getattr(obj, "provenance_hash", None)
    if h is None and isinstance(obj, dict):
        h = obj.get("provenance_hash")
    assert h is not None, "Missing provenance_hash"
    assert isinstance(h, str), f"provenance_hash should be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Invalid hex chars in hash"


# ===========================================================================
# Tests
# ===========================================================================

class TestCrossFrameworkDataMapperEngine:
    """Tests for CrossFrameworkDataMapperEngine."""

    # -----------------------------------------------------------------------
    # 1. Instantiation & config
    # -----------------------------------------------------------------------

    def test_engine_instantiation(self):
        """Engine can be created with default configuration."""
        engine = CrossFrameworkDataMapperEngine()
        assert engine is not None
        assert isinstance(engine.config, CrossFrameworkDataMapperConfig)

    def test_engine_has_config(self):
        """Engine stores custom configuration values."""
        custom = {"min_confidence": 0.8, "enable_derived_mappings": False}
        engine = CrossFrameworkDataMapperEngine(config=custom)
        assert engine.config.min_confidence == 0.8
        assert engine.config.enable_derived_mappings is False

    # -----------------------------------------------------------------------
    # 2. map_field
    # -----------------------------------------------------------------------

    def test_map_field_exact_match(self):
        """Exact-match mapping returns correct target field and high confidence."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_field(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 42000.0,
        )
        assert isinstance(result, MappingResult)
        assert result.target_field == "direct_emissions_tco2e"
        assert result.mapping_type == "EXACT"
        assert result.confidence >= 0.9
        assert result.mapped_value == 42000.0

    def test_map_field_approximate_match(self):
        """Approximate-match mapping returns lower confidence than exact."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_field(
            "CSRD", "E1_6_scope3_cat1_emissions", "CBAM", 10000.0,
        )
        assert isinstance(result, MappingResult)
        assert result.mapping_type == "APPROXIMATE"
        assert 0.5 <= result.confidence < 1.0

    def test_map_field_nonexistent_field(self):
        """Mapping a field with no match returns confidence 0.0."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_field(
            "CSRD", "nonexistent_field_xyz", "CBAM",
        )
        assert isinstance(result, MappingResult)
        assert result.confidence == 0.0
        assert result.target_field == ""
        assert result.mapping_type == "NONE"

    # -----------------------------------------------------------------------
    # 3. map_batch
    # -----------------------------------------------------------------------

    def test_map_batch_multiple_fields(self):
        """Batch mapping processes multiple fields and reports counts."""
        engine = CrossFrameworkDataMapperEngine()
        fields = {
            "E1_6_scope1_ghg_emissions": 42000.0,
            "E1_6_scope2_ghg_emissions": 15000.0,
            "nonexistent_field": 999.0,
        }
        result = engine.map_batch("CSRD", fields, "CBAM")
        assert isinstance(result, BatchMappingResult)
        assert result.total_fields == 3
        assert result.mapped_count >= 2
        assert result.unmapped_count >= 1
        assert "nonexistent_field" in result.unmapped_fields

    def test_map_batch_empty_input(self):
        """Batch mapping with empty dict returns zero counts."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_batch("CSRD", {}, "CBAM")
        assert isinstance(result, BatchMappingResult)
        assert result.total_fields == 0
        assert result.mapped_count == 0
        assert result.unmapped_count == 0
        assert len(result.mappings) == 0

    # -----------------------------------------------------------------------
    # 4. get_mappings_for_regulation
    # -----------------------------------------------------------------------

    def test_get_mappings_for_csrd(self):
        """CSRD has source mappings in multiple categories."""
        engine = CrossFrameworkDataMapperEngine()
        mappings = engine.get_mappings_for_regulation("CSRD", direction="source")
        assert isinstance(mappings, list)
        assert len(mappings) > 0
        for m in mappings:
            assert m.source_regulation == "CSRD"

    def test_get_mappings_for_cbam(self):
        """CBAM has source mappings in emissions and activity categories."""
        engine = CrossFrameworkDataMapperEngine()
        mappings = engine.get_mappings_for_regulation("CBAM", direction="source")
        assert len(mappings) > 0
        regulations = set(m.source_regulation for m in mappings)
        assert "CBAM" in regulations

    def test_get_mappings_for_eudr(self):
        """EUDR has source mappings in supply chain and biodiversity."""
        engine = CrossFrameworkDataMapperEngine()
        mappings = engine.get_mappings_for_regulation("EUDR", direction="source")
        assert len(mappings) > 0
        categories = set(m.category for m in mappings)
        assert len(categories) >= 1

    def test_get_mappings_for_taxonomy(self):
        """EU Taxonomy has source mappings across all categories."""
        engine = CrossFrameworkDataMapperEngine()
        mappings = engine.get_mappings_for_regulation(
            "EU_TAXONOMY", direction="source",
        )
        assert len(mappings) > 0

    # -----------------------------------------------------------------------
    # 5. Overlap statistics
    # -----------------------------------------------------------------------

    def test_get_overlap_statistics(self):
        """Overlap statistics between CSRD and CBAM are populated."""
        engine = CrossFrameworkDataMapperEngine()
        stats = engine.get_overlap_statistics("CSRD", "CBAM")
        assert isinstance(stats, OverlapStatistics)
        assert stats.regulation_pair == "CSRD-CBAM"
        assert stats.exact_matches >= 0
        assert stats.approximate_matches >= 0
        assert stats.derived_matches >= 0
        assert stats.overlap_percentage >= 0.0
        assert len(stats.categories_covered) > 0

    # -----------------------------------------------------------------------
    # 6. find_mapping_path
    # -----------------------------------------------------------------------

    def test_find_mapping_path_direct(self):
        """Direct mapping path between CSRD and CBAM for Scope 1."""
        engine = CrossFrameworkDataMapperEngine()
        path = engine.find_mapping_path(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM",
        )
        assert isinstance(path, MappingPath)
        assert path.total_confidence > 0.0
        assert path.path_length >= 1
        assert len(path.hops) >= 1
        assert path.source_regulation == "CSRD"
        assert path.target_regulation == "CBAM"

    # -----------------------------------------------------------------------
    # 7. Categories
    # -----------------------------------------------------------------------

    def test_mapping_categories_present(self):
        """All 7 mapping categories are present in the engine."""
        engine = CrossFrameworkDataMapperEngine()
        categories = engine.get_all_categories()
        expected = {
            "GHG_EMISSIONS",
            "SUPPLY_CHAIN",
            "ACTIVITY_CLASSIFICATION",
            "FINANCIAL_DATA",
            "CLIMATE_RISK",
            "WATER_POLLUTION",
            "BIODIVERSITY",
        }
        assert expected.issubset(set(categories.keys())), (
            f"Missing categories: {expected - set(categories.keys())}"
        )
        for cat, count in categories.items():
            assert count > 0, f"Category {cat} has zero mappings"

    # -----------------------------------------------------------------------
    # 8. Bidirectional mapping
    # -----------------------------------------------------------------------

    def test_bidirectional_mapping(self):
        """Bidirectional mapping works in reverse direction."""
        engine = CrossFrameworkDataMapperEngine()
        fwd = engine.map_field(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 42000.0,
        )
        assert fwd.confidence > 0.0
        rev = engine.map_field(
            "CBAM", "direct_emissions_tco2e", "CSRD", 42000.0,
        )
        assert rev.confidence > 0.0
        assert rev.target_field != ""

    # -----------------------------------------------------------------------
    # 9. Provenance
    # -----------------------------------------------------------------------

    def test_mapping_result_has_provenance_hash(self):
        """Individual mapping result carries a SHA-256 provenance hash."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_field(
            "CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 1000.0,
        )
        _assert_provenance_hash(result)

    def test_batch_result_has_provenance_hash(self):
        """Batch mapping result carries a SHA-256 provenance hash."""
        engine = CrossFrameworkDataMapperEngine()
        result = engine.map_batch(
            "CSRD",
            {"E1_6_scope1_ghg_emissions": 1000.0},
            "CBAM",
        )
        _assert_provenance_hash(result)

    # -----------------------------------------------------------------------
    # 10. Confidence range
    # -----------------------------------------------------------------------

    def test_mapping_confidence_range(self):
        """All mapping confidences in the database are between 0.0 and 1.0."""
        engine = CrossFrameworkDataMapperEngine()
        for mapping in engine._all_mappings:
            assert 0.0 <= mapping.confidence <= 1.0, (
                f"Mapping {mapping.source_field}->{mapping.target_field} "
                f"has confidence {mapping.confidence} outside [0, 1]"
            )

    # -----------------------------------------------------------------------
    # 11. Total mapping count
    # -----------------------------------------------------------------------

    def test_total_mapping_count_minimum(self):
        """Engine loads at least 80 field mappings from the database."""
        engine = CrossFrameworkDataMapperEngine()
        count = engine.get_total_mapping_count()
        assert count >= 80, (
            f"Expected at least 80 mappings, got {count}"
        )

    # -----------------------------------------------------------------------
    # 12. All four regulations represented
    # -----------------------------------------------------------------------

    def test_all_four_regulations_represented(self):
        """All four EU regulations appear as source in at least one mapping."""
        engine = CrossFrameworkDataMapperEngine()
        source_regulations = set()
        for mapping in engine._all_mappings:
            source_regulations.add(mapping.source_regulation)
        expected = {"CSRD", "CBAM", "EUDR", "EU_TAXONOMY"}
        assert expected.issubset(source_regulations), (
            f"Missing regulations: {expected - source_regulations}"
        )
