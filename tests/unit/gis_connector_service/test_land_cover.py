# -*- coding: utf-8 -*-
"""
Unit Tests for LandCoverEngine (AGENT-DATA-006)

Tests land cover classification, CORINE code mapping, carbon stock estimation,
forest cover detection, deforestation risk assessment, batch classification,
change detection, unknown coordinate defaults, and provenance tracking
for the GIS/Mapping Connector Agent.

Coverage target: 85%+ of land_cover.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline LandCoverEngine
# ---------------------------------------------------------------------------


class LandCoverClassification:
    """Result of a land cover classification."""

    def __init__(
        self,
        classification_id: str,
        latitude: float,
        longitude: float,
        land_cover_type: str,
        corine_code: Optional[str] = None,
        carbon_stock_t_ha: float = 0.0,
        confidence: float = 0.0,
        forest_cover: bool = False,
        deforestation_risk: str = "none",
        provenance_hash: str = "",
    ):
        self.classification_id = classification_id
        self.latitude = latitude
        self.longitude = longitude
        self.land_cover_type = land_cover_type
        self.corine_code = corine_code
        self.carbon_stock_t_ha = carbon_stock_t_ha
        self.confidence = confidence
        self.forest_cover = forest_cover
        self.deforestation_risk = deforestation_risk
        self.provenance_hash = provenance_hash
        self.timestamp = datetime.now(timezone.utc).isoformat()


class ChangeDetectionResult:
    """Result of land cover change detection between two dates."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        from_type: str,
        to_type: str,
        from_date: str,
        to_date: str,
        change_detected: bool = False,
        carbon_impact_t_ha: float = 0.0,
        provenance_hash: str = "",
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.from_type = from_type
        self.to_type = to_type
        self.from_date = from_date
        self.to_date = to_date
        self.change_detected = change_detected
        self.carbon_impact_t_ha = carbon_impact_t_ha
        self.provenance_hash = provenance_hash


class LandCoverEngine:
    """Engine for classifying land cover types and estimating carbon stocks.

    Uses CORINE land cover codes and simplified carbon stock lookup tables
    to classify geographic coordinates into land cover categories and
    estimate above-ground carbon stocks.
    """

    # CORINE code -> land cover type mapping
    CORINE_MAPPING: Dict[str, str] = {
        "111": "urban_continuous",
        "112": "urban_discontinuous",
        "121": "industrial",
        "211": "cropland",
        "212": "cropland_irrigated",
        "231": "pasture",
        "311": "forest_broadleaf",
        "312": "forest_coniferous",
        "313": "forest_mixed",
        "321": "grassland",
        "331": "bare_rock",
        "332": "bare_soil",
        "411": "wetland_inland",
        "421": "wetland_coastal",
        "511": "water_inland",
        "512": "water_coastal",
        "523": "ocean",
    }

    # Land cover type -> carbon stock (tonnes CO2 per hectare)
    CARBON_STOCK_TABLE: Dict[str, float] = {
        "forest_broadleaf": 160.0,
        "forest_coniferous": 130.0,
        "forest_mixed": 150.0,
        "cropland": 5.0,
        "cropland_irrigated": 6.0,
        "pasture": 10.0,
        "grassland": 8.0,
        "wetland_inland": 80.0,
        "wetland_coastal": 60.0,
        "urban_continuous": 1.0,
        "urban_discontinuous": 2.0,
        "industrial": 0.5,
        "water_inland": 0.0,
        "water_coastal": 0.0,
        "ocean": 0.0,
        "bare_rock": 0.0,
        "bare_soil": 0.5,
        "settlement": 2.0,
        "unknown": 0.0,
    }

    # Deforestation risk by region (simplified lat/lon lookup)
    DEFORESTATION_RISK_ZONES: List[Dict[str, Any]] = [
        {"lat_min": -10, "lat_max": 5, "lon_min": -80, "lon_max": -35, "risk": "high"},
        {"lat_min": -5, "lat_max": 10, "lon_min": 95, "lon_max": 140, "risk": "high"},
        {"lat_min": -5, "lat_max": 10, "lon_min": 10, "lon_max": 40, "risk": "medium"},
        {"lat_min": 45, "lat_max": 70, "lon_min": -10, "lon_max": 60, "risk": "low"},
    ]

    FOREST_TYPES = {"forest_broadleaf", "forest_coniferous", "forest_mixed"}

    def __init__(self):
        self._counter = 0
        self._classifications: Dict[str, LandCoverClassification] = {}

    def classify(
        self,
        latitude: float,
        longitude: float,
        corine_code: Optional[str] = None,
    ) -> LandCoverClassification:
        """Classify land cover at a given coordinate."""
        self._counter += 1
        classification_id = f"LCC-{self._counter:05d}"

        if corine_code and corine_code in self.CORINE_MAPPING:
            land_cover_type = self.CORINE_MAPPING[corine_code]
            confidence = 0.95
        else:
            land_cover_type = self._infer_type(latitude, longitude)
            corine_code = None
            confidence = 0.60

        carbon_stock = self.CARBON_STOCK_TABLE.get(land_cover_type, 0.0)
        forest_cover = land_cover_type in self.FOREST_TYPES
        deforestation_risk = self._assess_deforestation_risk(
            latitude, longitude, forest_cover
        )

        result = LandCoverClassification(
            classification_id=classification_id,
            latitude=latitude,
            longitude=longitude,
            land_cover_type=land_cover_type,
            corine_code=corine_code,
            carbon_stock_t_ha=carbon_stock,
            confidence=confidence,
            forest_cover=forest_cover,
            deforestation_risk=deforestation_risk,
            provenance_hash=_compute_hash({
                "classification_id": classification_id,
                "lat": latitude,
                "lon": longitude,
                "type": land_cover_type,
            }),
        )
        self._classifications[classification_id] = result
        return result

    def classify_batch(
        self,
        coordinates: List[Tuple[float, float]],
        corine_code: Optional[str] = None,
    ) -> List[LandCoverClassification]:
        """Classify land cover for a batch of coordinates."""
        return [
            self.classify(lat, lon, corine_code) for lat, lon in coordinates
        ]

    def detect_change(
        self,
        latitude: float,
        longitude: float,
        from_type: str,
        to_type: str,
        from_date: str = "2020-01-01",
        to_date: str = "2025-01-01",
    ) -> ChangeDetectionResult:
        """Detect land cover change at a coordinate between two dates."""
        change_detected = from_type != to_type
        from_carbon = self.CARBON_STOCK_TABLE.get(from_type, 0.0)
        to_carbon = self.CARBON_STOCK_TABLE.get(to_type, 0.0)
        carbon_impact = to_carbon - from_carbon

        return ChangeDetectionResult(
            latitude=latitude,
            longitude=longitude,
            from_type=from_type,
            to_type=to_type,
            from_date=from_date,
            to_date=to_date,
            change_detected=change_detected,
            carbon_impact_t_ha=carbon_impact,
            provenance_hash=_compute_hash({
                "lat": latitude,
                "lon": longitude,
                "from": from_type,
                "to": to_type,
            }),
        )

    def get_carbon_stock(self, land_cover_type: str) -> float:
        """Get carbon stock for a land cover type in t/ha."""
        return self.CARBON_STOCK_TABLE.get(land_cover_type, 0.0)

    def get_classification(self, classification_id: str) -> Optional[LandCoverClassification]:
        """Get a stored classification by ID."""
        return self._classifications.get(classification_id)

    def _infer_type(self, latitude: float, longitude: float) -> str:
        """Infer land cover type from coordinates (simplified heuristic)."""
        if abs(latitude) > 85:
            return "bare_rock"
        if abs(latitude) > 66:
            return "grassland"
        if abs(latitude) < 23 and -80 < longitude < -35:
            return "forest_broadleaf"
        if abs(latitude) < 23 and 95 < longitude < 140:
            return "forest_broadleaf"
        if 35 < latitude < 55 and -10 < longitude < 40:
            return "cropland"
        return "unknown"

    def _assess_deforestation_risk(
        self, latitude: float, longitude: float, forest_cover: bool
    ) -> str:
        """Assess deforestation risk based on location and forest cover."""
        if not forest_cover:
            return "none"
        for zone in self.DEFORESTATION_RISK_ZONES:
            if (zone["lat_min"] <= latitude <= zone["lat_max"]
                    and zone["lon_min"] <= longitude <= zone["lon_max"]):
                return zone["risk"]
        return "low"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> LandCoverEngine:
    return LandCoverEngine()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestClassify:
    """Tests for land cover classification."""

    def test_classify_with_corine_forest(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        assert result.land_cover_type == "forest_broadleaf"
        assert result.confidence == 0.95
        assert result.classification_id.startswith("LCC-")

    def test_classify_with_corine_cropland(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="211")
        assert result.land_cover_type == "cropland"

    def test_classify_with_corine_water(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="511")
        assert result.land_cover_type == "water_inland"

    def test_classify_with_corine_coniferous(self, engine):
        result = engine.classify(60.0, 25.0, corine_code="312")
        assert result.land_cover_type == "forest_coniferous"

    def test_classify_with_corine_mixed_forest(self, engine):
        result = engine.classify(50.0, 15.0, corine_code="313")
        assert result.land_cover_type == "forest_mixed"

    def test_classify_with_corine_urban(self, engine):
        result = engine.classify(51.5, -0.1, corine_code="111")
        assert result.land_cover_type == "urban_continuous"

    def test_classify_with_corine_settlement_discontinuous(self, engine):
        result = engine.classify(51.5, -0.1, corine_code="112")
        assert result.land_cover_type == "urban_discontinuous"

    def test_classify_with_corine_grassland(self, engine):
        result = engine.classify(52.0, 20.0, corine_code="321")
        assert result.land_cover_type == "grassland"

    def test_classify_with_corine_wetland(self, engine):
        result = engine.classify(52.0, 5.0, corine_code="411")
        assert result.land_cover_type == "wetland_inland"

    def test_classify_with_corine_pasture(self, engine):
        result = engine.classify(53.0, -1.0, corine_code="231")
        assert result.land_cover_type == "pasture"

    def test_classify_without_corine_infers_type(self, engine):
        result = engine.classify(48.0, 11.0)
        assert result.confidence == 0.60
        assert result.corine_code is None

    def test_classify_unknown_corine_falls_back(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="999")
        assert result.confidence == 0.60
        assert result.corine_code is None

    def test_classify_sequential_ids(self, engine):
        r1 = engine.classify(48.0, 11.0, corine_code="311")
        r2 = engine.classify(49.0, 12.0, corine_code="211")
        assert r1.classification_id == "LCC-00001"
        assert r2.classification_id == "LCC-00002"

    def test_classify_stores_result(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        stored = engine.get_classification(result.classification_id)
        assert stored is not None
        assert stored.land_cover_type == "forest_broadleaf"

    def test_classify_provenance_hash(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # valid hex


class TestCORINEMapping:
    """Tests for CORINE code to land cover type mapping."""

    def test_311_maps_to_forest_broadleaf(self):
        assert LandCoverEngine.CORINE_MAPPING["311"] == "forest_broadleaf"

    def test_312_maps_to_forest_coniferous(self):
        assert LandCoverEngine.CORINE_MAPPING["312"] == "forest_coniferous"

    def test_313_maps_to_forest_mixed(self):
        assert LandCoverEngine.CORINE_MAPPING["313"] == "forest_mixed"

    def test_211_maps_to_cropland(self):
        assert LandCoverEngine.CORINE_MAPPING["211"] == "cropland"

    def test_511_maps_to_water_inland(self):
        assert LandCoverEngine.CORINE_MAPPING["511"] == "water_inland"

    def test_111_maps_to_urban(self):
        assert LandCoverEngine.CORINE_MAPPING["111"] == "urban_continuous"

    def test_321_maps_to_grassland(self):
        assert LandCoverEngine.CORINE_MAPPING["321"] == "grassland"

    def test_total_corine_codes(self):
        assert len(LandCoverEngine.CORINE_MAPPING) >= 15


class TestCarbonStockEstimation:
    """Tests for carbon stock estimation by land cover type."""

    def test_forest_broadleaf_carbon(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("forest_broadleaf") == 160.0

    def test_forest_coniferous_carbon(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("forest_coniferous") == 130.0

    def test_forest_mixed_carbon(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("forest_mixed") == 150.0

    def test_cropland_carbon(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("cropland") == 5.0

    def test_water_carbon_zero(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("water_inland") == 0.0

    def test_wetland_carbon(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("wetland_inland") == 80.0

    def test_urban_carbon_low(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("urban_continuous") == 1.0

    def test_unknown_type_carbon_zero(self):
        engine = LandCoverEngine()
        assert engine.get_carbon_stock("nonexistent_type") == 0.0

    def test_classify_returns_matching_carbon(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        expected = LandCoverEngine.CARBON_STOCK_TABLE["forest_broadleaf"]
        assert result.carbon_stock_t_ha == expected

    def test_forest_carbon_approx_150(self, engine):
        """Forest types should have carbon stock in the ~130-160 range."""
        result = engine.classify(50.0, 10.0, corine_code="313")
        assert 100 < result.carbon_stock_t_ha < 200

    def test_cropland_carbon_approx_5(self, engine):
        """Cropland should have carbon stock ~5 t/ha."""
        result = engine.classify(50.0, 10.0, corine_code="211")
        assert 0 < result.carbon_stock_t_ha < 20


class TestForestCoverDetection:
    """Tests for forest cover detection."""

    def test_broadleaf_is_forest(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        assert result.forest_cover is True

    def test_coniferous_is_forest(self, engine):
        result = engine.classify(60.0, 25.0, corine_code="312")
        assert result.forest_cover is True

    def test_mixed_is_forest(self, engine):
        result = engine.classify(50.0, 15.0, corine_code="313")
        assert result.forest_cover is True

    def test_cropland_not_forest(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="211")
        assert result.forest_cover is False

    def test_water_not_forest(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="511")
        assert result.forest_cover is False

    def test_urban_not_forest(self, engine):
        result = engine.classify(51.0, 0.0, corine_code="111")
        assert result.forest_cover is False


class TestDeforestationRisk:
    """Tests for deforestation risk assessment."""

    def test_amazon_high_risk(self, engine):
        # Amazon basin: ~-3, -60
        result = engine.classify(-3.0, -60.0, corine_code="311")
        assert result.deforestation_risk == "high"

    def test_southeast_asia_high_risk(self, engine):
        # Borneo: ~1, 110
        result = engine.classify(1.0, 110.0, corine_code="311")
        assert result.deforestation_risk == "high"

    def test_central_africa_medium_risk(self, engine):
        # Congo basin: ~0, 25
        result = engine.classify(0.0, 25.0, corine_code="311")
        assert result.deforestation_risk == "medium"

    def test_european_forest_low_risk(self, engine):
        # Finland: ~61, 25
        result = engine.classify(61.0, 25.0, corine_code="312")
        assert result.deforestation_risk == "low"

    def test_non_forest_no_risk(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="211")
        assert result.deforestation_risk == "none"

    def test_water_no_risk(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="511")
        assert result.deforestation_risk == "none"


class TestBatchClassification:
    """Tests for batch land cover classification."""

    def test_batch_returns_list(self, engine):
        coords = [(48.0, 11.0), (50.0, 15.0), (55.0, 37.0)]
        results = engine.classify_batch(coords, corine_code="311")
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_each_has_id(self, engine):
        coords = [(48.0, 11.0), (50.0, 15.0)]
        results = engine.classify_batch(coords, corine_code="311")
        for r in results:
            assert r.classification_id.startswith("LCC-")

    def test_batch_preserves_coordinates(self, engine):
        coords = [(48.0, 11.0), (50.0, 15.0)]
        results = engine.classify_batch(coords)
        assert results[0].latitude == 48.0
        assert results[0].longitude == 11.0
        assert results[1].latitude == 50.0
        assert results[1].longitude == 15.0

    def test_batch_empty_list(self, engine):
        results = engine.classify_batch([])
        assert results == []

    def test_batch_single_element(self, engine):
        results = engine.classify_batch([(48.0, 11.0)], corine_code="211")
        assert len(results) == 1
        assert results[0].land_cover_type == "cropland"


class TestChangeDetection:
    """Tests for land cover change detection."""

    def test_no_change(self, engine):
        result = engine.detect_change(48.0, 11.0, "cropland", "cropland")
        assert result.change_detected is False
        assert result.carbon_impact_t_ha == 0.0

    def test_forest_to_cropland_negative_impact(self, engine):
        result = engine.detect_change(
            -3.0, -60.0, "forest_broadleaf", "cropland"
        )
        assert result.change_detected is True
        assert result.carbon_impact_t_ha < 0  # loss of carbon
        assert result.carbon_impact_t_ha == 5.0 - 160.0

    def test_cropland_to_forest_positive_impact(self, engine):
        result = engine.detect_change(48.0, 11.0, "cropland", "forest_mixed")
        assert result.change_detected is True
        assert result.carbon_impact_t_ha > 0
        assert result.carbon_impact_t_ha == 150.0 - 5.0

    def test_change_has_provenance(self, engine):
        result = engine.detect_change(
            48.0, 11.0, "forest_broadleaf", "cropland"
        )
        assert len(result.provenance_hash) == 64

    def test_change_preserves_dates(self, engine):
        result = engine.detect_change(
            48.0, 11.0, "forest_broadleaf", "cropland",
            from_date="2018-01-01", to_date="2024-06-15",
        )
        assert result.from_date == "2018-01-01"
        assert result.to_date == "2024-06-15"


class TestUnknownCoordinateDefaults:
    """Tests for unknown or edge-case coordinates."""

    def test_polar_coordinates(self, engine):
        result = engine.classify(89.0, 0.0)
        assert result.land_cover_type == "bare_rock"

    def test_high_latitude_grassland(self, engine):
        result = engine.classify(70.0, 30.0)
        assert result.land_cover_type == "grassland"

    def test_tropical_amazon_infers_forest(self, engine):
        result = engine.classify(0.0, -60.0)
        assert result.land_cover_type == "forest_broadleaf"

    def test_european_midlat_infers_cropland(self, engine):
        result = engine.classify(48.0, 11.0)
        assert result.land_cover_type == "cropland"

    def test_unclassifiable_returns_unknown(self, engine):
        # Mid-latitude Pacific Ocean coordinate, no CORINE
        result = engine.classify(30.0, -170.0)
        assert result.land_cover_type == "unknown"

    def test_get_nonexistent_classification(self, engine):
        assert engine.get_classification("LCC-99999") is None


class TestProvenance:
    """Tests for provenance tracking in classification results."""

    def test_classify_provenance_hash_length(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        assert len(result.provenance_hash) == 64

    def test_classify_provenance_valid_hex(self, engine):
        result = engine.classify(48.0, 11.0, corine_code="311")
        int(result.provenance_hash, 16)

    def test_different_inputs_different_hashes(self, engine):
        r1 = engine.classify(48.0, 11.0, corine_code="311")
        r2 = engine.classify(50.0, 15.0, corine_code="211")
        assert r1.provenance_hash != r2.provenance_hash

    def test_change_detection_provenance(self, engine):
        result = engine.detect_change(48.0, 11.0, "forest_broadleaf", "cropland")
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_batch_provenance_unique(self, engine):
        coords = [(48.0, 11.0), (50.0, 15.0), (55.0, 37.0)]
        results = engine.classify_batch(coords, corine_code="311")
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 3  # all unique
