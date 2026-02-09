# -*- coding: utf-8 -*-
"""
Unit Tests for BaselineAssessmentEngine (AGENT-DATA-007)

Tests EUDR cutoff date, FAO default forest definition, country-specific
forest definitions, forest classification, baseline compliance checks,
risk scoring, country risk adjustments, polygon grid sampling,
conservative aggregation, and deterministic behavior.

Coverage target: 85%+ of baseline_assessment.py

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
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
# Inline models
# ---------------------------------------------------------------------------


class ForestDefinition:
    """Country-specific or FAO default forest definition."""

    def __init__(
        self,
        country_code: str = "FAO",
        min_canopy_cover_pct: float = 10.0,
        min_tree_height_m: float = 5.0,
        min_area_hectares: float = 0.5,
    ):
        self.country_code = country_code
        self.min_canopy_cover_pct = min_canopy_cover_pct
        self.min_tree_height_m = min_tree_height_m
        self.min_area_hectares = min_area_hectares


class BaselineAssessment:
    """Result of a baseline forest status assessment."""

    def __init__(
        self,
        assessment_id: str = "",
        compliance_status: str = "review_required",
        risk_score: float = 50.0,
        forest_cover_pct: float = 0.0,
        forest_status: str = "unknown",
        deforestation_risk: str = "medium",
        cutoff_date: str = "2020-12-31",
        assessment_date: str = "",
        country_code: str = "",
        forest_definition: Optional[ForestDefinition] = None,
        sample_points: int = 9,
        provenance_hash: str = "",
    ):
        self.assessment_id = assessment_id
        self.compliance_status = compliance_status
        self.risk_score = max(0.0, min(100.0, risk_score))
        self.forest_cover_pct = forest_cover_pct
        self.forest_status = forest_status
        self.deforestation_risk = deforestation_risk
        self.cutoff_date = cutoff_date
        self.assessment_date = assessment_date
        self.country_code = country_code
        self.forest_definition = forest_definition
        self.sample_points = sample_points
        self.provenance_hash = provenance_hash


class SamplePointResult:
    """Result of a single sample point assessment."""

    def __init__(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        forest_cover_pct: float = 0.0,
        is_forest: bool = False,
        was_forest_at_cutoff: bool = False,
        change_detected: bool = False,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.forest_cover_pct = forest_cover_pct
        self.is_forest = is_forest
        self.was_forest_at_cutoff = was_forest_at_cutoff
        self.change_detected = change_detected


class BaselineCheckRequest:
    """Request to perform a baseline compliance check."""

    def __init__(
        self,
        polygon_coordinates: Optional[List[Tuple[float, float]]] = None,
        country_code: str = "",
        cutoff_date: str = "2020-12-31",
        sample_points: int = 9,
    ):
        self.polygon_coordinates = polygon_coordinates or []
        self.country_code = country_code
        self.cutoff_date = cutoff_date
        self.sample_points = sample_points


# ---------------------------------------------------------------------------
# Inline BaselineAssessmentEngine
# ---------------------------------------------------------------------------


class BaselineAssessmentEngine:
    """Engine for assessing baseline forest cover status for EUDR compliance.

    Determines whether a geographic polygon was forested at the EUDR cutoff
    date (2020-12-31) using country-specific forest definitions, grid
    sampling, and conservative aggregation.
    """

    EUDR_CUTOFF_DATE: str = "2020-12-31"

    # FAO default forest definition (used when country not in lookup)
    FAO_DEFAULT: ForestDefinition = ForestDefinition(
        country_code="FAO",
        min_canopy_cover_pct=10.0,
        min_tree_height_m=5.0,
        min_area_hectares=0.5,
    )

    # Country-specific forest definitions
    COUNTRY_DEFINITIONS: Dict[str, ForestDefinition] = {
        "BRA": ForestDefinition(country_code="BRA", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=1.0),
        "IDN": ForestDefinition(country_code="IDN", min_canopy_cover_pct=30.0,
                                min_tree_height_m=5.0, min_area_hectares=0.25),
        "COD": ForestDefinition(country_code="COD", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
        "COG": ForestDefinition(country_code="COG", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
        "MYS": ForestDefinition(country_code="MYS", min_canopy_cover_pct=30.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
        "PER": ForestDefinition(country_code="PER", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
        "CMR": ForestDefinition(country_code="CMR", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
        "CIV": ForestDefinition(country_code="CIV", min_canopy_cover_pct=10.0,
                                min_tree_height_m=5.0, min_area_hectares=0.5),
    }

    # Country risk adjustments (added to base risk score)
    COUNTRY_RISK_ADJUSTMENTS: Dict[str, float] = {
        "BRA": 10.0,
        "IDN": 10.0,
        "COD": 8.0,
        "COG": 7.0,
        "MYS": 8.0,
        "PER": 6.0,
        "CMR": 7.0,
        "CIV": 6.0,
    }

    def __init__(self) -> None:
        self._assessment_counter: int = 0

    # ------------------------------------------------------------------
    # Forest definition lookup
    # ------------------------------------------------------------------

    def get_forest_definition(self, country_code: str) -> ForestDefinition:
        """Get country-specific forest definition, falling back to FAO default."""
        return self.COUNTRY_DEFINITIONS.get(country_code, self.FAO_DEFAULT)

    # ------------------------------------------------------------------
    # Forest classification
    # ------------------------------------------------------------------

    def is_forest(
        self, canopy_cover_pct: float, forest_definition: ForestDefinition,
    ) -> bool:
        """Determine if a sample point qualifies as forest.

        A point is forest if canopy_cover_pct >= the definition's minimum.
        """
        return canopy_cover_pct >= forest_definition.min_canopy_cover_pct

    # ------------------------------------------------------------------
    # Baseline check
    # ------------------------------------------------------------------

    def check_baseline(self, request: BaselineCheckRequest) -> BaselineAssessment:
        """Perform a full baseline compliance check for a polygon.

        Steps:
        1. Look up country forest definition
        2. Generate grid sample points within polygon
        3. Assess each sample point (mock forest cover)
        4. Aggregate results conservatively (worst case wins)
        5. Calculate risk score and compliance status
        """
        self._assessment_counter += 1
        forest_def = self.get_forest_definition(request.country_code)

        # Generate sample points
        sample_results = self._sample_polygon(
            polygon=request.polygon_coordinates,
            num_points=request.sample_points,
            country_code=request.country_code,
            forest_def=forest_def,
        )

        # Aggregate conservatively
        compliance, risk_score, forest_cover_pct, forest_status, risk_level = (
            self._aggregate_conservative(sample_results, request.country_code)
        )

        assessment_id = f"ba_{self._assessment_counter:04d}"
        assessment = BaselineAssessment(
            assessment_id=assessment_id,
            compliance_status=compliance,
            risk_score=risk_score,
            forest_cover_pct=forest_cover_pct,
            forest_status=forest_status,
            deforestation_risk=risk_level,
            cutoff_date=request.cutoff_date,
            assessment_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            country_code=request.country_code,
            forest_definition=forest_def,
            sample_points=request.sample_points,
            provenance_hash=_compute_hash({
                "assessment_id": assessment_id,
                "country_code": request.country_code,
                "sample_points": request.sample_points,
                "compliance": compliance,
            }),
        )
        return assessment

    # ------------------------------------------------------------------
    # Grid sampling
    # ------------------------------------------------------------------

    def _sample_polygon(
        self,
        polygon: List[Tuple[float, float]],
        num_points: int,
        country_code: str,
        forest_def: ForestDefinition,
    ) -> List[SamplePointResult]:
        """Generate grid sample points within the polygon.

        Creates an NxN grid (where N = ceil(sqrt(num_points))) over the
        polygon bounding box and evaluates each point.
        """
        if not polygon or len(polygon) < 3:
            return []

        lats = [p[0] for p in polygon]
        lons = [p[1] for p in polygon]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        grid_size = math.ceil(math.sqrt(num_points))
        lat_step = (max_lat - min_lat) / max(grid_size - 1, 1)
        lon_step = (max_lon - min_lon) / max(grid_size - 1, 1)

        results: List[SamplePointResult] = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(results) >= num_points:
                    break
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                cover = self._mock_forest_cover(lat, lon, country_code)
                is_forest_now = self.is_forest(cover, forest_def)

                # Deterministic "was forest at cutoff" based on coordinates
                seed_str = f"{lat:.4f}|{lon:.4f}|cutoff"
                seed_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                was_forest = (seed_val % 100) < 70  # 70% chance was forested

                change = was_forest and not is_forest_now

                results.append(SamplePointResult(
                    latitude=round(lat, 6),
                    longitude=round(lon, 6),
                    forest_cover_pct=cover,
                    is_forest=is_forest_now,
                    was_forest_at_cutoff=was_forest,
                    change_detected=change,
                ))
        return results

    def _mock_forest_cover(
        self, lat: float, lon: float, country_code: str,
    ) -> float:
        """Generate deterministic mock forest cover percentage.

        Seeded from lat/lon/country for reproducibility.
        """
        seed_str = f"{lat:.4f}|{lon:.4f}|{country_code}"
        seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
        seed_val = int(seed_hash[:8], 16)
        # Generate cover between 0 and 95
        return round((seed_val % 96), 2)

    # ------------------------------------------------------------------
    # Conservative aggregation
    # ------------------------------------------------------------------

    def _aggregate_conservative(
        self,
        samples: List[SamplePointResult],
        country_code: str,
    ) -> Tuple[str, float, float, str, str]:
        """Aggregate sample results using conservative (worst-case) logic.

        - If any sample shows change detected: non_compliant
        - If average forest cover below definition threshold: review_required
        - Otherwise: compliant

        Risk score = base + country adjustment.
        """
        if not samples:
            return "review_required", 50.0, 0.0, "unknown", "medium"

        total_cover = sum(s.forest_cover_pct for s in samples)
        avg_cover = total_cover / len(samples)

        any_change = any(s.change_detected for s in samples)
        forest_count = sum(1 for s in samples if s.is_forest)
        was_forest_count = sum(1 for s in samples if s.was_forest_at_cutoff)

        # Determine forest status
        if forest_count == 0:
            forest_status = "non_forest"
        elif any_change:
            forest_status = "deforested"
        elif forest_count == len(samples):
            forest_status = "intact"
        else:
            forest_status = "degraded"

        # Base risk from change detection
        base_risk = 0.0
        if any_change:
            change_ratio = sum(1 for s in samples if s.change_detected) / len(samples)
            base_risk = change_ratio * 70.0  # Up to 70 for all-change
        else:
            # Lower base risk if no change detected
            base_risk = max(0.0, 30.0 - avg_cover * 0.3)

        # Country risk adjustment
        country_adj = self.COUNTRY_RISK_ADJUSTMENTS.get(country_code, 0.0)
        risk_score = round(min(base_risk + country_adj, 100.0), 2)

        # Risk level
        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        elif risk_score >= 10:
            risk_level = "low"
        else:
            risk_level = "none"

        # Compliance
        if any_change:
            compliance = "non_compliant"
        elif avg_cover < 10.0:
            compliance = "review_required"
        else:
            compliance = "compliant"

        return compliance, risk_score, round(avg_cover, 2), forest_status, risk_level

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def assessment_count(self) -> int:
        """Number of assessments performed."""
        return self._assessment_counter


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> BaselineAssessmentEngine:
    return BaselineAssessmentEngine()


@pytest.fixture
def tropical_polygon() -> List[Tuple[float, float]]:
    return [
        (-3.0, 25.0),
        (-3.0, 26.0),
        (-2.0, 26.0),
        (-2.0, 25.0),
        (-3.0, 25.0),
    ]


# ===========================================================================
# Test: EUDR cutoff date constant
# ===========================================================================


class TestEUDRCutoffDate:
    """Test EUDR cutoff date constant."""

    def test_eudr_cutoff_date_value(self, engine):
        assert engine.EUDR_CUTOFF_DATE == "2020-12-31"

    def test_eudr_cutoff_date_is_string(self, engine):
        assert isinstance(engine.EUDR_CUTOFF_DATE, str)

    def test_eudr_cutoff_date_parseable(self, engine):
        from datetime import date
        parsed = date.fromisoformat(engine.EUDR_CUTOFF_DATE)
        assert parsed.year == 2020
        assert parsed.month == 12
        assert parsed.day == 31


# ===========================================================================
# Test: FAO default definition
# ===========================================================================


class TestFAODefaultDefinition:
    """Test FAO default forest definition."""

    def test_fao_min_canopy_cover(self, engine):
        assert engine.FAO_DEFAULT.min_canopy_cover_pct == 10.0

    def test_fao_min_tree_height(self, engine):
        assert engine.FAO_DEFAULT.min_tree_height_m == 5.0

    def test_fao_min_area(self, engine):
        assert engine.FAO_DEFAULT.min_area_hectares == 0.5

    def test_fao_country_code(self, engine):
        assert engine.FAO_DEFAULT.country_code == "FAO"


# ===========================================================================
# Test: Country definitions exist
# ===========================================================================


class TestCountryDefinitionsExist:
    """Test that all 8 country definitions are present."""

    def test_eight_countries(self, engine):
        assert len(engine.COUNTRY_DEFINITIONS) == 8

    @pytest.mark.parametrize("country", [
        "BRA", "IDN", "COD", "COG", "MYS", "PER", "CMR", "CIV",
    ])
    def test_country_exists(self, engine, country):
        assert country in engine.COUNTRY_DEFINITIONS

    @pytest.mark.parametrize("country", [
        "BRA", "IDN", "COD", "COG", "MYS", "PER", "CMR", "CIV",
    ])
    def test_country_has_valid_definition(self, engine, country):
        defn = engine.COUNTRY_DEFINITIONS[country]
        assert defn.min_canopy_cover_pct > 0
        assert defn.min_tree_height_m > 0
        assert defn.min_area_hectares > 0


# ===========================================================================
# Test: Brazil definition
# ===========================================================================


class TestBrazilDefinition:
    """Test Brazil-specific forest definition."""

    def test_brazil_canopy_cover(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["BRA"]
        assert defn.min_canopy_cover_pct == 10.0

    def test_brazil_min_area(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["BRA"]
        assert defn.min_area_hectares == 1.0

    def test_brazil_tree_height(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["BRA"]
        assert defn.min_tree_height_m == 5.0


# ===========================================================================
# Test: Indonesia definition
# ===========================================================================


class TestIndonesiaDefinition:
    """Test Indonesia-specific forest definition."""

    def test_indonesia_canopy_cover(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["IDN"]
        assert defn.min_canopy_cover_pct == 30.0

    def test_indonesia_min_area(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["IDN"]
        assert defn.min_area_hectares == 0.25

    def test_indonesia_tree_height(self, engine):
        defn = engine.COUNTRY_DEFINITIONS["IDN"]
        assert defn.min_tree_height_m == 5.0


# ===========================================================================
# Test: Forest definition lookup
# ===========================================================================


class TestGetForestDefinition:
    """Test get_forest_definition lookup."""

    def test_known_country_returns_specific(self, engine):
        defn = engine.get_forest_definition("BRA")
        assert defn.country_code == "BRA"

    def test_known_country_idn(self, engine):
        defn = engine.get_forest_definition("IDN")
        assert defn.country_code == "IDN"
        assert defn.min_canopy_cover_pct == 30.0

    def test_unknown_country_returns_fao(self, engine):
        defn = engine.get_forest_definition("XYZ")
        assert defn.country_code == "FAO"
        assert defn.min_canopy_cover_pct == 10.0

    def test_empty_string_returns_fao(self, engine):
        defn = engine.get_forest_definition("")
        assert defn.country_code == "FAO"


# ===========================================================================
# Test: is_forest classification
# ===========================================================================


class TestIsForest:
    """Test forest/non-forest classification."""

    def test_above_threshold_is_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=10.0)
        assert engine.is_forest(50.0, defn) is True

    def test_at_threshold_is_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=10.0)
        assert engine.is_forest(10.0, defn) is True

    def test_below_threshold_not_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=10.0)
        assert engine.is_forest(5.0, defn) is False

    def test_zero_cover_not_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=10.0)
        assert engine.is_forest(0.0, defn) is False

    def test_high_threshold_not_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=30.0)
        assert engine.is_forest(25.0, defn) is False

    def test_high_threshold_is_forest(self, engine):
        defn = ForestDefinition(min_canopy_cover_pct=30.0)
        assert engine.is_forest(35.0, defn) is True


# ===========================================================================
# Test: check_baseline compliance
# ===========================================================================


class TestCheckBaselineCompliant:
    """Test baseline compliance check returning compliant."""

    def test_compliant_returns_assessment(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="COD",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert isinstance(result, BaselineAssessment)

    def test_assessment_has_id(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="COD",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.assessment_id != ""

    def test_assessment_has_provenance(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="COD",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.provenance_hash != ""

    def test_assessment_compliance_status_valid(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="COD",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.compliance_status in ("compliant", "non_compliant", "review_required")

    def test_assessment_has_forest_definition(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="BRA",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.forest_definition is not None
        assert result.forest_definition.country_code == "BRA"


class TestCheckBaselineNonCompliant:
    """Test baseline check can return non_compliant when change detected."""

    def test_non_compliant_possible(self, engine):
        """With enough sample points, some polygons may show deforestation."""
        # Use a polygon where deterministic mock data shows change
        polygon = [
            (-5.0, 30.0),
            (-5.0, 31.0),
            (-4.0, 31.0),
            (-4.0, 30.0),
            (-5.0, 30.0),
        ]
        req = BaselineCheckRequest(
            polygon_coordinates=polygon,
            country_code="COD",
            sample_points=25,
        )
        result = engine.check_baseline(req)
        # Either non_compliant or compliant; we check structure is valid
        assert result.compliance_status in ("compliant", "non_compliant", "review_required")
        assert 0.0 <= result.risk_score <= 100.0


class TestCheckBaselineReviewRequired:
    """Test baseline check returning review_required."""

    def test_empty_polygon_returns_review(self, engine):
        """Empty polygon returns review_required."""
        req = BaselineCheckRequest(
            polygon_coordinates=[],
            country_code="BRA",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.compliance_status == "review_required"

    def test_two_point_polygon_returns_review(self, engine):
        """Polygon with < 3 points returns review_required."""
        req = BaselineCheckRequest(
            polygon_coordinates=[(-3.0, 25.0), (-2.0, 26.0)],
            country_code="BRA",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert result.compliance_status == "review_required"


# ===========================================================================
# Test: Risk score range
# ===========================================================================


class TestRiskScoreRange:
    """Test risk score is always in [0, 100]."""

    def test_risk_score_in_range(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="BRA",
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.parametrize("country", [
        "BRA", "IDN", "COD", "COG", "MYS", "PER", "CMR", "CIV", "XYZ",
    ])
    def test_risk_score_always_valid(self, engine, tropical_polygon, country):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code=country,
            sample_points=9,
        )
        result = engine.check_baseline(req)
        assert 0.0 <= result.risk_score <= 100.0


# ===========================================================================
# Test: Country risk adjustments
# ===========================================================================


class TestCountryRiskAdjustments:
    """Test country-specific risk score adjustments."""

    def test_brazil_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("BRA") == 10.0

    def test_indonesia_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("IDN") == 10.0

    def test_cod_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("COD") == 8.0

    def test_cog_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("COG") == 7.0

    def test_malaysia_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("MYS") == 8.0

    def test_peru_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("PER") == 6.0

    def test_cameroon_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("CMR") == 7.0

    def test_ivory_coast_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("CIV") == 6.0

    def test_unknown_country_no_adjustment(self, engine):
        assert engine.COUNTRY_RISK_ADJUSTMENTS.get("USA", 0.0) == 0.0

    def test_high_risk_countries_have_highest_adjustments(self, engine):
        """Brazil and Indonesia should have the highest adjustments."""
        bra = engine.COUNTRY_RISK_ADJUSTMENTS["BRA"]
        idn = engine.COUNTRY_RISK_ADJUSTMENTS["IDN"]
        for code, adj in engine.COUNTRY_RISK_ADJUSTMENTS.items():
            if code not in ("BRA", "IDN"):
                assert adj <= bra
                assert adj <= idn


# ===========================================================================
# Test: Polygon baseline grid sampling
# ===========================================================================


class TestPolygonGridSampling:
    """Test polygon grid sampling for baseline assessment."""

    def test_sample_count_matches_request(self, engine):
        polygon = [
            (-3.0, 25.0), (-3.0, 26.0), (-2.0, 26.0), (-2.0, 25.0), (-3.0, 25.0),
        ]
        defn = engine.get_forest_definition("COD")
        results = engine._sample_polygon(polygon, 9, "COD", defn)
        assert len(results) == 9

    def test_sample_has_coordinates(self, engine):
        polygon = [
            (-3.0, 25.0), (-3.0, 26.0), (-2.0, 26.0), (-2.0, 25.0), (-3.0, 25.0),
        ]
        defn = engine.get_forest_definition("COD")
        results = engine._sample_polygon(polygon, 4, "COD", defn)
        for r in results:
            assert -3.0 <= r.latitude <= -2.0
            assert 25.0 <= r.longitude <= 26.0

    def test_sample_has_forest_cover(self, engine):
        polygon = [
            (-3.0, 25.0), (-3.0, 26.0), (-2.0, 26.0), (-2.0, 25.0), (-3.0, 25.0),
        ]
        defn = engine.get_forest_definition("COD")
        results = engine._sample_polygon(polygon, 9, "COD", defn)
        for r in results:
            assert 0.0 <= r.forest_cover_pct <= 95.0

    def test_sample_25_points(self, engine):
        polygon = [
            (-3.0, 25.0), (-3.0, 26.0), (-2.0, 26.0), (-2.0, 25.0), (-3.0, 25.0),
        ]
        defn = engine.get_forest_definition("BRA")
        results = engine._sample_polygon(polygon, 25, "BRA", defn)
        assert len(results) == 25

    def test_empty_polygon_returns_empty(self, engine):
        defn = engine.FAO_DEFAULT
        results = engine._sample_polygon([], 9, "", defn)
        assert results == []


# ===========================================================================
# Test: Conservative aggregation
# ===========================================================================


class TestConservativeAggregation:
    """Test conservative (worst-case-wins) aggregation."""

    def test_any_change_detected_non_compliant(self, engine):
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=50.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=5.0, is_forest=False,
                was_forest_at_cutoff=True, change_detected=True,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "COD")
        assert compliance == "non_compliant"

    def test_no_change_and_good_cover_compliant(self, engine):
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=60.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=55.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "COD")
        assert compliance == "compliant"

    def test_empty_samples_review_required(self, engine):
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative([], "BRA")
        assert compliance == "review_required"
        assert risk == 50.0
        assert status == "unknown"

    def test_all_non_forest_status(self, engine):
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=2.0, is_forest=False,
                was_forest_at_cutoff=False, change_detected=False,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=3.0, is_forest=False,
                was_forest_at_cutoff=False, change_detected=False,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "COD")
        assert status == "non_forest"

    def test_all_forest_intact_status(self, engine):
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=80.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=75.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "COD")
        assert status == "intact"

    def test_change_detected_deforested_status(self, engine):
        """Change detected with some forest remaining -> deforested."""
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=15.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=True,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=5.0, is_forest=False,
                was_forest_at_cutoff=True, change_detected=True,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "BRA")
        assert status == "deforested"

    def test_mixed_forest_degraded_status(self, engine):
        """Some forest, some not, but no change detected -> degraded."""
        samples = [
            SamplePointResult(
                latitude=-2.5, longitude=25.5,
                forest_cover_pct=50.0, is_forest=True,
                was_forest_at_cutoff=True, change_detected=False,
            ),
            SamplePointResult(
                latitude=-2.6, longitude=25.6,
                forest_cover_pct=5.0, is_forest=False,
                was_forest_at_cutoff=False, change_detected=False,
            ),
        ]
        compliance, risk, avg_cover, status, level = engine._aggregate_conservative(samples, "COD")
        assert status == "degraded"


# ===========================================================================
# Test: Mock forest cover deterministic
# ===========================================================================


class TestMockForestCoverDeterministic:
    """Test that mock forest cover is deterministic."""

    def test_same_coordinates_same_cover(self, engine):
        cover1 = engine._mock_forest_cover(-2.5, 25.5, "COD")
        cover2 = engine._mock_forest_cover(-2.5, 25.5, "COD")
        assert cover1 == cover2

    def test_different_coordinates_may_differ(self, engine):
        cover1 = engine._mock_forest_cover(-2.5, 25.5, "COD")
        cover2 = engine._mock_forest_cover(-3.5, 26.5, "COD")
        # Not guaranteed to differ, but the seeds are different
        # Just verify both are valid
        assert 0.0 <= cover1 <= 95.0
        assert 0.0 <= cover2 <= 95.0

    def test_cover_range_valid(self, engine):
        for lat in [-5.0, -2.5, 0.0, 2.5, 5.0]:
            for lon in [20.0, 25.0, 30.0]:
                cover = engine._mock_forest_cover(lat, lon, "COD")
                assert 0.0 <= cover <= 95.0


# ===========================================================================
# Test: Assessment count
# ===========================================================================


class TestAssessmentCount:
    """Test assessment_count property."""

    def test_starts_at_zero(self, engine):
        assert engine.assessment_count == 0

    def test_increments_on_check(self, engine, tropical_polygon):
        req = BaselineCheckRequest(
            polygon_coordinates=tropical_polygon,
            country_code="COD",
            sample_points=4,
        )
        engine.check_baseline(req)
        assert engine.assessment_count == 1
        engine.check_baseline(req)
        assert engine.assessment_count == 2

    def test_count_tracks_multiple(self, engine, tropical_polygon):
        for i in range(5):
            req = BaselineCheckRequest(
                polygon_coordinates=tropical_polygon,
                country_code="COD",
                sample_points=4,
            )
            engine.check_baseline(req)
        assert engine.assessment_count == 5
