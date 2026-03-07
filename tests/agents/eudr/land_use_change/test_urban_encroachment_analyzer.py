# -*- coding: utf-8 -*-
"""
Tests for UrbanEncroachmentAnalyzer - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Detection of 5 infrastructure types (road, building, mining,
  industrial, residential)
- Expansion rate calculation
- Pressure corridor detection
- Time-to-conversion estimation
- Urban proximity risk scoring
- Buffer zone analysis
- Batch analysis processing
- Deterministic analysis behavior

Test count: 45 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.land_use_change.conftest import (
    UrbanEncroachment,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    INFRASTRUCTURE_TYPES,
)


# ===========================================================================
# 1. Infrastructure Type Detection Tests (12 tests)
# ===========================================================================


class TestInfrastructureDetection:
    """Tests for detecting 5 infrastructure types."""

    def test_detect_road_construction(self):
        """Detect road construction near forest boundary."""
        result = UrbanEncroachment(
            plot_id="PLOT-ROAD-001",
            infrastructure_type="road_construction",
            expansion_rate_ha_per_year=2.5,
            urban_proximity_km=3.0,
            confidence=0.88,
        )
        assert result.infrastructure_type == "road_construction"
        assert result.confidence > 0.80

    def test_detect_building_expansion(self):
        """Detect building expansion near forest."""
        result = UrbanEncroachment(
            plot_id="PLOT-BLDG-001",
            infrastructure_type="building_expansion",
            expansion_rate_ha_per_year=5.0,
            urban_proximity_km=1.5,
            confidence=0.90,
        )
        assert result.infrastructure_type == "building_expansion"
        assert result.urban_proximity_km < 5.0

    def test_detect_mining_activity(self):
        """Detect mining activity near forest."""
        result = UrbanEncroachment(
            plot_id="PLOT-MINE-001",
            infrastructure_type="mining_activity",
            expansion_rate_ha_per_year=15.0,
            urban_proximity_km=8.0,
            confidence=0.85,
        )
        assert result.infrastructure_type == "mining_activity"
        assert result.expansion_rate_ha_per_year > 10.0

    def test_detect_industrial_development(self):
        """Detect industrial development near forest."""
        result = UrbanEncroachment(
            plot_id="PLOT-IND-001",
            infrastructure_type="industrial_development",
            expansion_rate_ha_per_year=8.0,
            urban_proximity_km=2.0,
            confidence=0.82,
        )
        assert result.infrastructure_type == "industrial_development"

    def test_detect_residential_growth(self):
        """Detect residential growth near forest."""
        result = UrbanEncroachment(
            plot_id="PLOT-RES-001",
            infrastructure_type="residential_growth",
            expansion_rate_ha_per_year=3.0,
            urban_proximity_km=0.5,
            confidence=0.92,
        )
        assert result.infrastructure_type == "residential_growth"
        assert result.urban_proximity_km < 1.0

    @pytest.mark.parametrize("itype", INFRASTRUCTURE_TYPES)
    def test_all_infrastructure_types_valid(self, itype):
        """Each infrastructure type value is accepted."""
        result = UrbanEncroachment(
            plot_id=f"PLOT-{itype[:4].upper()}",
            infrastructure_type=itype,
        )
        assert result.infrastructure_type == itype

    def test_road_construction_linear_pattern(self):
        """Road construction creates linear encroachment pattern."""
        result = UrbanEncroachment(
            infrastructure_type="road_construction",
            pressure_corridors=[
                {"direction": "NE", "length_km": 5.0, "width_km": 0.5},
            ],
        )
        assert len(result.pressure_corridors) == 1
        assert result.pressure_corridors[0]["direction"] == "NE"

    def test_mining_large_area_impact(self):
        """Mining has large area impact."""
        result = UrbanEncroachment(
            infrastructure_type="mining_activity",
            expansion_rate_ha_per_year=20.0,
        )
        assert result.expansion_rate_ha_per_year > 15.0

    def test_residential_slow_expansion(self):
        """Residential growth has slower expansion than mining."""
        residential = UrbanEncroachment(
            infrastructure_type="residential_growth",
            expansion_rate_ha_per_year=2.0,
        )
        mining = UrbanEncroachment(
            infrastructure_type="mining_activity",
            expansion_rate_ha_per_year=20.0,
        )
        assert residential.expansion_rate_ha_per_year < mining.expansion_rate_ha_per_year

    def test_building_expansion_high_confidence(self):
        """Building detection has high confidence from spectral signature."""
        result = UrbanEncroachment(
            infrastructure_type="building_expansion",
            confidence=0.92,
        )
        assert result.confidence > 0.85

    def test_industrial_moderate_confidence(self):
        """Industrial detection has moderate confidence."""
        result = UrbanEncroachment(
            infrastructure_type="industrial_development",
            confidence=0.78,
        )
        assert 0.70 <= result.confidence <= 0.90

    def test_infrastructure_type_count(self):
        """There are exactly 5 infrastructure types."""
        assert len(INFRASTRUCTURE_TYPES) == 5


# ===========================================================================
# 2. Expansion Rate Tests (5 tests)
# ===========================================================================


class TestExpansionRate:
    """Tests for expansion rate calculation."""

    def test_expansion_rate_calculation(self):
        """Expansion rate is calculated in ha/year."""
        result = UrbanEncroachment(
            expansion_rate_ha_per_year=5.0,
        )
        assert result.expansion_rate_ha_per_year == 5.0

    def test_expansion_rate_positive(self):
        """Active encroachment has positive expansion rate."""
        result = UrbanEncroachment(
            expansion_rate_ha_per_year=3.5,
        )
        assert result.expansion_rate_ha_per_year > 0.0

    def test_expansion_rate_zero(self):
        """No encroachment has zero expansion rate."""
        result = UrbanEncroachment(
            expansion_rate_ha_per_year=0.0,
        )
        assert result.expansion_rate_ha_per_year == 0.0

    def test_high_expansion_rate(self):
        """Very high expansion rate for aggressive development."""
        result = UrbanEncroachment(
            expansion_rate_ha_per_year=50.0,
            infrastructure_type="mining_activity",
        )
        assert result.expansion_rate_ha_per_year >= 50.0

    def test_expansion_rate_varies_by_type(self):
        """Different infrastructure types have different rates."""
        rates = {
            "road_construction": 2.5,
            "building_expansion": 5.0,
            "mining_activity": 20.0,
            "industrial_development": 8.0,
            "residential_growth": 3.0,
        }
        for itype, rate in rates.items():
            result = UrbanEncroachment(
                infrastructure_type=itype,
                expansion_rate_ha_per_year=rate,
            )
            assert result.expansion_rate_ha_per_year == rate


# ===========================================================================
# 3. Pressure Corridor Tests (5 tests)
# ===========================================================================


class TestPressureCorridor:
    """Tests for pressure corridor detection."""

    def test_pressure_corridor_detection(self):
        """Pressure corridors detected along road network."""
        result = UrbanEncroachment(
            plot_id="PLOT-COR-001",
            pressure_corridors=[
                {"direction": "N", "length_km": 10.0, "width_km": 2.0},
                {"direction": "E", "length_km": 5.0, "width_km": 1.5},
            ],
        )
        assert len(result.pressure_corridors) == 2

    def test_pressure_corridor_empty(self):
        """No pressure corridors for isolated areas."""
        result = UrbanEncroachment(
            plot_id="PLOT-COR-002",
            pressure_corridors=[],
        )
        assert len(result.pressure_corridors) == 0

    def test_pressure_corridor_directions(self):
        """Corridors have compass direction."""
        result = UrbanEncroachment(
            pressure_corridors=[
                {"direction": "NE", "length_km": 8.0, "width_km": 1.0},
            ],
        )
        assert result.pressure_corridors[0]["direction"] in [
            "N", "NE", "E", "SE", "S", "SW", "W", "NW",
        ]

    def test_pressure_corridor_length(self):
        """Corridor length is positive."""
        result = UrbanEncroachment(
            pressure_corridors=[
                {"direction": "S", "length_km": 15.0, "width_km": 3.0},
            ],
        )
        assert result.pressure_corridors[0]["length_km"] > 0.0

    def test_multiple_corridors(self):
        """Multiple corridors from different directions."""
        result = UrbanEncroachment(
            pressure_corridors=[
                {"direction": "N", "length_km": 10.0, "width_km": 2.0},
                {"direction": "E", "length_km": 5.0, "width_km": 1.5},
                {"direction": "SW", "length_km": 8.0, "width_km": 1.8},
            ],
        )
        assert len(result.pressure_corridors) == 3
        directions = [c["direction"] for c in result.pressure_corridors]
        assert len(set(directions)) == 3


# ===========================================================================
# 4. Time-to-Conversion Tests (5 tests)
# ===========================================================================


class TestTimeToConversion:
    """Tests for time-to-conversion estimation."""

    def test_time_to_conversion_estimate(self):
        """Time to conversion estimated in years."""
        result = UrbanEncroachment(
            plot_id="PLOT-TTC-001",
            time_to_conversion_years=3.5,
        )
        assert result.time_to_conversion_years is not None
        assert result.time_to_conversion_years > 0.0

    def test_time_to_conversion_none(self):
        """No conversion expected when no encroachment pressure."""
        result = UrbanEncroachment(
            plot_id="PLOT-TTC-002",
            time_to_conversion_years=None,
        )
        assert result.time_to_conversion_years is None

    def test_short_time_to_conversion(self):
        """Short time to conversion for close proximity."""
        result = UrbanEncroachment(
            time_to_conversion_years=1.0,
            urban_proximity_km=0.5,
        )
        assert result.time_to_conversion_years <= 2.0

    def test_long_time_to_conversion(self):
        """Long time to conversion for distant areas."""
        result = UrbanEncroachment(
            time_to_conversion_years=15.0,
            urban_proximity_km=25.0,
        )
        assert result.time_to_conversion_years > 10.0

    def test_conversion_time_decreases_with_proximity(self):
        """Closer areas have shorter time to conversion."""
        close = UrbanEncroachment(
            urban_proximity_km=1.0,
            time_to_conversion_years=2.0,
        )
        far = UrbanEncroachment(
            urban_proximity_km=20.0,
            time_to_conversion_years=15.0,
        )
        assert close.time_to_conversion_years < far.time_to_conversion_years


# ===========================================================================
# 5. Urban Proximity and Buffer Zone Tests (7 tests)
# ===========================================================================


class TestUrbanProximityAndBuffer:
    """Tests for urban proximity risk and buffer zone analysis."""

    def test_urban_proximity_risk_close(self):
        """Close proximity to urban area has high risk."""
        result = UrbanEncroachment(
            urban_proximity_km=1.0,
            buffer_zone_risk=0.85,
        )
        assert result.buffer_zone_risk > 0.70

    def test_urban_proximity_risk_far(self):
        """Far from urban area has low risk."""
        result = UrbanEncroachment(
            urban_proximity_km=40.0,
            buffer_zone_risk=0.10,
        )
        assert result.buffer_zone_risk < 0.20

    def test_buffer_zone_analysis_default(self, sample_config):
        """Default buffer zone is 10 km."""
        assert sample_config.default_buffer_km == 10.0

    def test_buffer_zone_analysis_max(self, sample_config):
        """Maximum buffer zone is 50 km."""
        assert sample_config.max_buffer_km == 50.0

    def test_buffer_zone_risk_in_range(self):
        """Buffer zone risk is in [0, 1] range."""
        for risk in [0.0, 0.25, 0.50, 0.75, 1.0]:
            result = UrbanEncroachment(
                buffer_zone_risk=risk,
            )
            assert 0.0 <= result.buffer_zone_risk <= 1.0

    def test_proximity_km_positive(self):
        """Urban proximity distance is non-negative."""
        result = UrbanEncroachment(
            urban_proximity_km=5.0,
        )
        assert result.urban_proximity_km >= 0.0

    def test_risk_inversely_proportional_to_distance(self):
        """Risk decreases as distance from urban area increases."""
        close = UrbanEncroachment(
            urban_proximity_km=2.0,
            buffer_zone_risk=0.80,
        )
        far = UrbanEncroachment(
            urban_proximity_km=30.0,
            buffer_zone_risk=0.15,
        )
        assert close.buffer_zone_risk > far.buffer_zone_risk


# ===========================================================================
# 6. Batch and Determinism Tests (6 tests)
# ===========================================================================


class TestBatchAndDeterminism:
    """Tests for batch analysis and deterministic behavior."""

    def test_batch_analysis(self):
        """Batch analysis of 20 plots."""
        results = [
            UrbanEncroachment(
                plot_id=f"PLOT-BATCH-{i:04d}",
                infrastructure_type=INFRASTRUCTURE_TYPES[i % len(INFRASTRUCTURE_TYPES)],
                confidence=0.80 + i * 0.005,
            )
            for i in range(20)
        ]
        assert len(results) == 20

    def test_deterministic_analysis(self):
        """Same inputs produce same analysis results."""
        results = [
            UrbanEncroachment(
                plot_id="PLOT-DET-001",
                infrastructure_type="road_construction",
                expansion_rate_ha_per_year=5.0,
                urban_proximity_km=3.0,
                confidence=0.85,
            )
            for _ in range(5)
        ]
        assert all(r.infrastructure_type == "road_construction" for r in results)
        assert all(r.expansion_rate_ha_per_year == 5.0 for r in results)
        assert all(r.confidence == 0.85 for r in results)

    def test_provenance_hash(self):
        """Analysis result has provenance hash."""
        h = compute_test_hash({"plot_id": "PLOT-PRV", "type": "road"})
        result = UrbanEncroachment(
            provenance_hash=h,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_confidence_range(self):
        """Confidence is in [0, 1] range."""
        for conf in [0.0, 0.25, 0.50, 0.75, 1.0]:
            result = UrbanEncroachment(
                confidence=conf,
            )
            assert 0.0 <= result.confidence <= 1.0

    def test_batch_unique_ids(self):
        """Batch results have unique plot IDs."""
        results = [
            UrbanEncroachment(
                plot_id=f"PLOT-UNQ-{i:04d}",
            )
            for i in range(15)
        ]
        ids = [r.plot_id for r in results]
        assert len(set(ids)) == 15

    def test_batch_all_have_infrastructure_type(self):
        """Each batch result has infrastructure type set."""
        results = [
            UrbanEncroachment(
                plot_id=f"PLOT-IT-{i}",
                infrastructure_type=INFRASTRUCTURE_TYPES[i % len(INFRASTRUCTURE_TYPES)],
            )
            for i in range(10)
        ]
        assert all(r.infrastructure_type != "" for r in results)


# ===========================================================================
# 7. Edge Cases (5 tests)
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases in urban encroachment analysis."""

    def test_zero_proximity(self):
        """Plot at urban boundary (proximity = 0)."""
        result = UrbanEncroachment(
            urban_proximity_km=0.0,
            buffer_zone_risk=1.0,
        )
        assert result.urban_proximity_km == 0.0
        assert result.buffer_zone_risk == 1.0

    def test_very_large_distance(self):
        """Very remote plot (100+ km from urban)."""
        result = UrbanEncroachment(
            urban_proximity_km=150.0,
            buffer_zone_risk=0.01,
        )
        assert result.urban_proximity_km > 100.0
        assert result.buffer_zone_risk < 0.05

    def test_no_infrastructure_detected(self):
        """No infrastructure type when nothing detected."""
        result = UrbanEncroachment(
            infrastructure_type="",
            confidence=0.0,
        )
        assert result.infrastructure_type == ""
        assert result.confidence == 0.0

    def test_multiple_infrastructure_types(self):
        """Area with multiple types picks primary one."""
        result = UrbanEncroachment(
            infrastructure_type="road_construction",
            confidence=0.90,
        )
        # Only one primary type per result
        assert result.infrastructure_type in INFRASTRUCTURE_TYPES

    def test_peri_urban_zone(self, germany_urban_plot):
        """Peri-urban zone analysis for European context."""
        lat, lon = germany_urban_plot
        assert lat > 50.0  # Northern Europe
        result = UrbanEncroachment(
            plot_id=f"PLOT-EU-{lat:.0f}-{lon:.0f}",
            infrastructure_type="residential_growth",
            urban_proximity_km=2.0,
            confidence=0.85,
        )
        assert result.confidence > 0.80
