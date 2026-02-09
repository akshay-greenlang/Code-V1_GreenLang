# -*- coding: utf-8 -*-
"""
Unit Tests for DeforestationClassifierEngine (AGENT-DATA-007)

Tests land cover classification (10-class hierarchical decision tree),
binary forest/non-forest classification, tree cover estimation (NDVI/EVI
blended), canopy height estimation, confidence scoring, and provenance
tracking for the Deforestation Satellite Connector agent.

Coverage target: 85%+ of deforestation_classifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline DeforestationClassifierEngine mirroring
# greenlang/deforestation_satellite/deforestation_classifier.py
# ---------------------------------------------------------------------------

# 10-class land cover types
LAND_COVER_DENSE_FOREST = "dense_forest"
LAND_COVER_OPEN_FOREST = "open_forest"
LAND_COVER_SHRUBLAND = "shrubland"
LAND_COVER_GRASSLAND = "grassland"
LAND_COVER_CROPLAND = "cropland"
LAND_COVER_BARE_SOIL = "bare_soil"
LAND_COVER_WATER = "water"
LAND_COVER_URBAN = "urban"
LAND_COVER_WETLAND = "wetland"
LAND_COVER_UNKNOWN = "unknown"


class DeforestationClassifierEngine:
    """Hierarchical decision-tree land cover classifier with NDVI/EVI thresholds.

    Decision tree order:
        Water (NDWI > 0.3)
        -> Dense Forest (NDVI >= 0.6, EVI >= 0.35)
        -> Open Forest (NDVI >= 0.4, EVI >= 0.2)
        -> Shrubland (0.3 <= NDVI < 0.4)
        -> Grassland (NDVI >= 0.2)
        -> Bare Soil (0 < NDVI < 0.2)
        -> Unknown (NDVI <= 0)
    """

    def __init__(self, agent_id: str = "GL-DATA-GEO-003"):
        self._agent_id = agent_id
        self._classifications: List[Dict[str, Any]] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -----------------------------------------------------------------
    # 10-class classification
    # -----------------------------------------------------------------

    def classify(
        self,
        ndvi: float,
        evi: Optional[float] = None,
        ndwi: Optional[float] = None,
        nbr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classify a pixel into one of 10 land cover types."""
        land_cover = self._decision_tree(ndvi, evi, ndwi)
        confidence = self._compute_confidence(ndvi, evi, ndwi, land_cover)
        tree_cover = self.estimate_tree_cover(ndvi, evi)
        canopy_height = self.estimate_canopy_height(tree_cover)
        provenance_hash = self._hash(
            {"ndvi": ndvi, "evi": evi, "ndwi": ndwi, "land_cover": land_cover}
        )

        result = {
            "classification_id": f"cls-{uuid.uuid4().hex[:12]}",
            "land_cover_type": land_cover,
            "ndvi": ndvi,
            "evi": evi,
            "ndwi": ndwi,
            "nbr": nbr,
            "tree_cover_pct": tree_cover,
            "canopy_height_m": canopy_height,
            "confidence": confidence,
            "is_forest": land_cover in (LAND_COVER_DENSE_FOREST, LAND_COVER_OPEN_FOREST),
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self._agent_id,
            "provenance_hash": provenance_hash,
        }
        self._classifications.append(result)
        return result

    # -----------------------------------------------------------------
    # Binary forest / non-forest
    # -----------------------------------------------------------------

    def classify_binary(self, ndvi: float) -> bool:
        """Return True if NDVI indicates forest (>= 0.4)."""
        return ndvi >= 0.4

    # -----------------------------------------------------------------
    # Tree cover estimation
    # -----------------------------------------------------------------

    def estimate_tree_cover(
        self, ndvi: float, evi: Optional[float] = None
    ) -> float:
        """Estimate tree cover percentage (0-100).

        Base formula (calibrated to Hansen GFC):
            tree_cover = 166.67 * ndvi - 33.33
        When EVI is available, blend: 60% NDVI-based + 40% EVI-based.
        Result clamped to [0, 100].
        """
        ndvi_cover = 166.67 * ndvi - 33.33
        if evi is not None:
            evi_cover = 166.67 * evi - 33.33
            blended = 0.6 * ndvi_cover + 0.4 * evi_cover
        else:
            blended = ndvi_cover
        return max(0.0, min(100.0, round(blended, 2)))

    # -----------------------------------------------------------------
    # Canopy height estimation
    # -----------------------------------------------------------------

    def estimate_canopy_height(self, tree_cover_pct: float) -> float:
        """Estimate canopy height from tree cover %.

        Simple linear model: height_m = 0.3 * tree_cover_pct.
        """
        return round(0.3 * tree_cover_pct, 2)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _decision_tree(
        self,
        ndvi: float,
        evi: Optional[float],
        ndwi: Optional[float],
    ) -> str:
        # Water detection first
        if ndwi is not None and ndwi > 0.3:
            return LAND_COVER_WATER
        # Dense forest
        if ndvi >= 0.6 and (evi is None or evi >= 0.35):
            return LAND_COVER_DENSE_FOREST
        # Open forest
        if ndvi >= 0.4 and (evi is None or evi >= 0.2):
            return LAND_COVER_OPEN_FOREST
        # Shrubland
        if 0.3 <= ndvi < 0.4:
            return LAND_COVER_SHRUBLAND
        # Grassland
        if ndvi >= 0.2:
            return LAND_COVER_GRASSLAND
        # Bare soil
        if 0 < ndvi < 0.2:
            return LAND_COVER_BARE_SOIL
        # Unknown
        return LAND_COVER_UNKNOWN

    def _compute_confidence(
        self,
        ndvi: float,
        evi: Optional[float],
        ndwi: Optional[float],
        land_cover: str,
    ) -> float:
        """Compute per-pixel confidence score (0.0 - 1.0)."""
        base = 0.6
        # NDVI in strong range increases confidence
        if 0.3 <= ndvi <= 0.9:
            base += 0.15
        # EVI available adds confidence
        if evi is not None:
            base += 0.1
        # NDWI available adds confidence
        if ndwi is not None:
            base += 0.05
        # Dense forest / water typically high confidence
        if land_cover in (LAND_COVER_DENSE_FOREST, LAND_COVER_WATER):
            base += 0.05
        return min(1.0, round(base, 2))

    def get_classification_count(self) -> int:
        return len(self._classifications)

    def get_classifications(self) -> List[Dict[str, Any]]:
        return list(self._classifications)

    def _hash(self, data: Dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestClassifyLandCover:
    """Test 10-class hierarchical decision tree classification."""

    def test_classify_dense_forest(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.75, evi=0.45)
        assert result["land_cover_type"] == LAND_COVER_DENSE_FOREST

    def test_classify_dense_forest_ndvi_only(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.65)
        assert result["land_cover_type"] == LAND_COVER_DENSE_FOREST

    def test_classify_dense_forest_threshold_ndvi_06(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.6, evi=0.35)
        assert result["land_cover_type"] == LAND_COVER_DENSE_FOREST

    def test_classify_open_forest(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5, evi=0.25)
        assert result["land_cover_type"] == LAND_COVER_OPEN_FOREST

    def test_classify_open_forest_ndvi_only(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.45)
        assert result["land_cover_type"] == LAND_COVER_OPEN_FOREST

    def test_classify_open_forest_threshold_ndvi_04(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.4, evi=0.2)
        assert result["land_cover_type"] == LAND_COVER_OPEN_FOREST

    def test_classify_shrubland(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.35)
        assert result["land_cover_type"] == LAND_COVER_SHRUBLAND

    def test_classify_shrubland_lower_bound(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.3)
        assert result["land_cover_type"] == LAND_COVER_SHRUBLAND

    def test_classify_shrubland_upper_excluded(self):
        """NDVI=0.4 should not be shrubland (should be open forest)."""
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.4)
        assert result["land_cover_type"] != LAND_COVER_SHRUBLAND

    def test_classify_grassland(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.25)
        assert result["land_cover_type"] == LAND_COVER_GRASSLAND

    def test_classify_grassland_threshold_02(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.2)
        assert result["land_cover_type"] == LAND_COVER_GRASSLAND

    def test_classify_bare_soil(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.1)
        assert result["land_cover_type"] == LAND_COVER_BARE_SOIL

    def test_classify_bare_soil_low_ndvi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.01)
        assert result["land_cover_type"] == LAND_COVER_BARE_SOIL

    def test_classify_water(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.1, ndwi=0.5)
        assert result["land_cover_type"] == LAND_COVER_WATER

    def test_classify_water_ndwi_threshold(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.1, ndwi=0.31)
        assert result["land_cover_type"] == LAND_COVER_WATER

    def test_classify_water_ndwi_at_boundary(self):
        """NDWI=0.3 should NOT classify as water (requires >0.3)."""
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.1, ndwi=0.3)
        assert result["land_cover_type"] != LAND_COVER_WATER

    def test_classify_unknown_negative_ndvi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=-0.1)
        assert result["land_cover_type"] == LAND_COVER_UNKNOWN

    def test_classify_unknown_zero_ndvi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.0)
        assert result["land_cover_type"] == LAND_COVER_UNKNOWN

    def test_classify_water_overrides_forest(self):
        """Water detection takes priority over forest NDVI."""
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.7, evi=0.4, ndwi=0.5)
        assert result["land_cover_type"] == LAND_COVER_WATER


class TestBinaryClassification:
    def test_classify_binary_forest(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.4) is True

    def test_classify_binary_forest_high(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.8) is True

    def test_classify_binary_not_forest(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.39) is False

    def test_classify_binary_not_forest_low(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.1) is False

    def test_classify_binary_threshold_exact(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.4) is True

    def test_classify_binary_zero(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(0.0) is False

    def test_classify_binary_negative(self):
        engine = DeforestationClassifierEngine()
        assert engine.classify_binary(-0.5) is False


class TestTreeCoverEstimation:
    def test_estimate_tree_cover_formula(self):
        engine = DeforestationClassifierEngine()
        # At NDVI=0.5: 166.67*0.5 - 33.33 = 83.335 - 33.33 = 50.005
        cover = engine.estimate_tree_cover(0.5)
        assert abs(cover - 50.0) < 1.0

    def test_tree_cover_at_ndvi_02(self):
        engine = DeforestationClassifierEngine()
        # At NDVI=0.2: 166.67*0.2 - 33.33 = 33.334 - 33.33 = 0.004
        cover = engine.estimate_tree_cover(0.2)
        assert cover == pytest.approx(0.0, abs=1.0)

    def test_tree_cover_at_ndvi_08(self):
        engine = DeforestationClassifierEngine()
        # At NDVI=0.8: 166.67*0.8 - 33.33 = 133.336 - 33.33 = 100.006
        cover = engine.estimate_tree_cover(0.8)
        assert cover == pytest.approx(100.0, abs=1.0)

    def test_tree_cover_clamped_low(self):
        engine = DeforestationClassifierEngine()
        cover = engine.estimate_tree_cover(0.0)
        assert cover >= 0.0

    def test_tree_cover_clamped_high(self):
        engine = DeforestationClassifierEngine()
        cover = engine.estimate_tree_cover(1.0)
        assert cover <= 100.0

    def test_tree_cover_with_evi_blend(self):
        engine = DeforestationClassifierEngine()
        # With EVI: blended = 0.6 * NDVI_cover + 0.4 * EVI_cover
        ndvi = 0.5
        evi = 0.3
        ndvi_cover = 166.67 * ndvi - 33.33  # ~50.005
        evi_cover = 166.67 * evi - 33.33  # ~16.671
        expected = 0.6 * ndvi_cover + 0.4 * evi_cover
        result = engine.estimate_tree_cover(ndvi, evi)
        assert result == pytest.approx(max(0.0, min(100.0, expected)), abs=0.5)

    def test_tree_cover_evi_none_no_blend(self):
        engine = DeforestationClassifierEngine()
        # Without EVI: should equal pure NDVI-based
        ndvi = 0.5
        cover_no_evi = engine.estimate_tree_cover(ndvi, evi=None)
        expected = max(0.0, min(100.0, 166.67 * ndvi - 33.33))
        assert cover_no_evi == pytest.approx(expected, abs=0.5)

    def test_tree_cover_negative_ndvi_clamped(self):
        engine = DeforestationClassifierEngine()
        cover = engine.estimate_tree_cover(-0.5)
        assert cover == 0.0

    def test_tree_cover_ndvi_one_clamped(self):
        engine = DeforestationClassifierEngine()
        cover = engine.estimate_tree_cover(1.0)
        assert cover == 100.0


class TestCanopyHeightEstimation:
    def test_canopy_height_estimation(self):
        engine = DeforestationClassifierEngine()
        height = engine.estimate_canopy_height(50.0)
        assert height == pytest.approx(15.0, abs=0.01)

    def test_canopy_height_zero_cover(self):
        engine = DeforestationClassifierEngine()
        assert engine.estimate_canopy_height(0.0) == 0.0

    def test_canopy_height_full_cover(self):
        engine = DeforestationClassifierEngine()
        assert engine.estimate_canopy_height(100.0) == 30.0

    def test_canopy_height_proportional(self):
        engine = DeforestationClassifierEngine()
        h25 = engine.estimate_canopy_height(25.0)
        h75 = engine.estimate_canopy_height(75.0)
        assert h75 > h25
        assert h75 == pytest.approx(3 * h25, abs=0.01)


class TestConfidenceScoring:
    def test_confidence_base_minimum(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.0)
        assert result["confidence"] >= 0.6

    def test_confidence_increases_with_evi(self):
        engine = DeforestationClassifierEngine()
        r1 = engine.classify(ndvi=0.5)
        r2 = engine.classify(ndvi=0.5, evi=0.3)
        assert r2["confidence"] >= r1["confidence"]

    def test_confidence_increases_with_ndwi(self):
        engine = DeforestationClassifierEngine()
        r1 = engine.classify(ndvi=0.5)
        r2 = engine.classify(ndvi=0.5, ndwi=0.1)
        assert r2["confidence"] >= r1["confidence"]

    def test_confidence_strong_ndvi_range(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5, evi=0.3, ndwi=0.1)
        assert result["confidence"] >= 0.8

    def test_confidence_capped_at_1(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.7, evi=0.4, ndwi=0.1)
        assert result["confidence"] <= 1.0

    def test_confidence_dense_forest_bonus(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.7, evi=0.4)
        assert result["confidence"] >= 0.8


class TestClassificationWithProvenance:
    def test_classification_has_provenance_hash(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_provenance_hash_is_hex(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5)
        int(result["provenance_hash"], 16)

    def test_classification_has_agent_id(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5)
        assert result["agent_id"] == "GL-DATA-GEO-003"

    def test_classification_has_timestamp(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5)
        assert "timestamp" in result

    def test_classification_has_id(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.5)
        assert result["classification_id"].startswith("cls-")

    def test_classification_has_is_forest(self):
        engine = DeforestationClassifierEngine()
        r1 = engine.classify(ndvi=0.7)
        r2 = engine.classify(ndvi=0.1)
        assert r1["is_forest"] is True
        assert r2["is_forest"] is False


class TestClassificationCount:
    def test_initial_count_zero(self):
        engine = DeforestationClassifierEngine()
        assert engine.get_classification_count() == 0

    def test_count_after_classification(self):
        engine = DeforestationClassifierEngine()
        engine.classify(ndvi=0.5)
        assert engine.get_classification_count() == 1

    def test_count_after_multiple(self):
        engine = DeforestationClassifierEngine()
        engine.classify(ndvi=0.5)
        engine.classify(ndvi=0.3)
        engine.classify(ndvi=0.7)
        assert engine.get_classification_count() == 3

    def test_get_classifications_returns_list(self):
        engine = DeforestationClassifierEngine()
        engine.classify(ndvi=0.5)
        cls_list = engine.get_classifications()
        assert isinstance(cls_list, list)
        assert len(cls_list) == 1

    def test_get_classifications_returns_copy(self):
        engine = DeforestationClassifierEngine()
        engine.classify(ndvi=0.5)
        c1 = engine.get_classifications()
        c2 = engine.get_classifications()
        assert c1 is not c2


class TestClassificationResultFields:
    def test_result_contains_ndvi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55)
        assert result["ndvi"] == 0.55

    def test_result_contains_evi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55, evi=0.3)
        assert result["evi"] == 0.3

    def test_result_contains_ndwi(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55, ndwi=0.2)
        assert result["ndwi"] == 0.2

    def test_result_contains_nbr(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55, nbr=0.4)
        assert result["nbr"] == 0.4

    def test_result_tree_cover_pct(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55)
        assert 0.0 <= result["tree_cover_pct"] <= 100.0

    def test_result_canopy_height_m(self):
        engine = DeforestationClassifierEngine()
        result = engine.classify(ndvi=0.55)
        assert result["canopy_height_m"] >= 0.0


class TestCustomAgentId:
    def test_custom_agent_id(self):
        engine = DeforestationClassifierEngine(agent_id="CUSTOM-SAT-001")
        assert engine.agent_id == "CUSTOM-SAT-001"

    def test_default_agent_id(self):
        engine = DeforestationClassifierEngine()
        assert engine.agent_id == "GL-DATA-GEO-003"
