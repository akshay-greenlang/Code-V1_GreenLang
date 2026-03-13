# -*- coding: utf-8 -*-
"""
Unit tests for GeographicSourcingAnalyzer - AGENT-EUDR-017 Engine 5

Tests geographic sourcing pattern analysis with country risk integration,
concentration analysis, proximity scoring, seasonal patterns, and supply
chain depth tracking.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest

from greenlang.agents.eudr.supplier_risk_scorer.geographic_sourcing_analyzer import (
    GeographicSourcingAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    CommodityType,
    RiskLevel,
)


class TestGeographicSourcingAnalyzerInit:
    """Tests for GeographicSourcingAnalyzer initialization."""

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        analyzer = GeographicSourcingAnalyzer()
        assert analyzer._sourcing_profiles == {}


class TestAnalyzeSourcing:
    """Tests for analyze_sourcing method."""

    @pytest.mark.unit
    def test_analyze_sourcing_returns_profile(
        self, geographic_sourcing_analyzer, sample_sourcing
    ):
        result = geographic_sourcing_analyzer.analyze_sourcing(
            supplier_id=sample_sourcing["supplier_id"],
            sourcing_locations=sample_sourcing["sourcing_locations"],
            commodity=sample_sourcing["commodity"],
        )
        assert result is not None
        assert "profile_id" in result
        assert result["supplier_id"] == sample_sourcing["supplier_id"]


class TestRiskZones:
    """Tests for risk zone detection."""

    @pytest.mark.unit
    def test_detect_high_risk_zones(self, geographic_sourcing_analyzer):
        # Brazil Amazon region (high risk)
        locations = [
            {"country": "BR", "region": "Amazonas", "latitude": -3.0, "longitude": -60.0}
        ]
        risk_zones = geographic_sourcing_analyzer.detect_risk_zones(locations)
        assert len(risk_zones) > 0


class TestConcentrationIndex:
    """Tests for concentration index calculation (HHI)."""

    @pytest.mark.unit
    def test_calculate_hhi_high_concentration(
        self, geographic_sourcing_analyzer
    ):
        # Single source = max concentration
        locations = [
            {"country": "BR", "volume_percentage": 100.0}
        ]
        hhi = geographic_sourcing_analyzer.calculate_concentration_index(locations)
        assert hhi == Decimal("10000.0")  # Max HHI

    @pytest.mark.unit
    def test_calculate_hhi_diversified(self, geographic_sourcing_analyzer):
        # Multiple balanced sources = low concentration
        locations = [
            {"country": "BR", "volume_percentage": 25.0},
            {"country": "AR", "volume_percentage": 25.0},
            {"country": "UY", "volume_percentage": 25.0},
            {"country": "PY", "volume_percentage": 25.0},
        ]
        hhi = geographic_sourcing_analyzer.calculate_concentration_index(locations)
        assert hhi < Decimal("2500.0")  # Below high concentration threshold


class TestPatternChanges:
    """Tests for sourcing pattern change detection."""

    @pytest.mark.unit
    def test_detect_pattern_changes(self, geographic_sourcing_analyzer):
        # Historical vs current sourcing
        historical = [{"country": "BR", "volume_percentage": 100.0}]
        current = [
            {"country": "BR", "volume_percentage": 50.0},
            {"country": "AR", "volume_percentage": 50.0},
        ]
        changes = geographic_sourcing_analyzer.detect_pattern_changes(
            historical, current
        )
        assert len(changes) > 0


class TestProtectedAreaProximity:
    """Tests for protected area proximity scoring."""

    @pytest.mark.unit
    def test_check_protected_area_proximity(
        self, geographic_sourcing_analyzer, mock_config
    ):
        # Location near protected area
        location = {"latitude": -3.5, "longitude": -60.0}
        proximity = geographic_sourcing_analyzer.check_protected_area_proximity(
            location,
            buffer_km=mock_config.proximity_buffer_km,
        )
        assert "distance_km" in proximity


class TestIndigenousProximity:
    """Tests for indigenous territory proximity."""

    @pytest.mark.unit
    def test_check_indigenous_proximity(
        self, geographic_sourcing_analyzer, mock_config
    ):
        location = {"latitude": -3.5, "longitude": -60.0}
        proximity = geographic_sourcing_analyzer.check_indigenous_proximity(
            location,
            buffer_km=mock_config.proximity_buffer_km,
        )
        assert "distance_km" in proximity


class TestSeasonalPatterns:
    """Tests for seasonal sourcing pattern analysis."""

    @pytest.mark.unit
    def test_analyze_seasonal_patterns(self, geographic_sourcing_analyzer):
        # Sourcing data with timestamps
        sourcing_history = [
            {"month": 1, "volume": 1000.0, "country": "BR"},
            {"month": 4, "volume": 2000.0, "country": "BR"},
            {"month": 7, "volume": 1500.0, "country": "BR"},
            {"month": 10, "volume": 1800.0, "country": "BR"},
        ]
        patterns = geographic_sourcing_analyzer.analyze_seasonal_patterns(
            sourcing_history
        )
        assert "peak_months" in patterns


class TestNewRegion:
    """Tests for new sourcing region risk assessment."""

    @pytest.mark.unit
    def test_assess_new_region_risk(self, geographic_sourcing_analyzer):
        new_region = {"country": "ID", "region": "Kalimantan"}
        risk = geographic_sourcing_analyzer.assess_new_region_risk(new_region)
        assert "risk_score" in risk
        assert Decimal("0.0") <= risk["risk_score"] <= Decimal("100.0")


class TestSupplyDepth:
    """Tests for supply chain depth analysis."""

    @pytest.mark.unit
    def test_analyze_supply_depth(self, geographic_sourcing_analyzer):
        sourcing = {
            "tier_1": [{"country": "BR", "volume": 5000.0}],
            "tier_2": [{"country": "BR", "volume": 3000.0}],
            "tier_3": [{"country": "BR", "volume": 1000.0}],
        }
        depth_analysis = geographic_sourcing_analyzer.analyze_supply_depth(sourcing)
        assert "max_tier" in depth_analysis
        assert "visibility_score" in depth_analysis


class TestSatelliteCrossReference:
    """Tests for satellite data cross-reference."""

    @pytest.mark.unit
    def test_cross_reference_satellite_data(self, geographic_sourcing_analyzer):
        location = {"latitude": -3.5, "longitude": -60.0}
        result = geographic_sourcing_analyzer.cross_reference_satellite(
            location,
            date_range=("2024-01-01", "2024-12-31"),
        )
        assert "deforestation_alerts" in result


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_analysis_includes_provenance_hash(
        self, geographic_sourcing_analyzer, sample_sourcing
    ):
        result = geographic_sourcing_analyzer.analyze_sourcing(
            supplier_id=sample_sourcing["supplier_id"],
            sourcing_locations=sample_sourcing["sourcing_locations"],
            commodity=sample_sourcing["commodity"],
        )
        assert "provenance_hash" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_coordinates_raises_error(self, geographic_sourcing_analyzer):
        with pytest.raises(ValueError):
            geographic_sourcing_analyzer.analyze_sourcing(
                supplier_id="SUPP-001",
                sourcing_locations=[
                    {"latitude": 200.0, "longitude": 300.0}  # Invalid
                ],
                commodity=CommodityType.SOYA,
            )
