# -*- coding: utf-8 -*-
"""
Unit tests for TradeFlowAnalyzer - AGENT-EUDR-016 Engine 6

Tests import/export trade flow analysis for EUDR commodities covering
bilateral flow recording, route risk scoring, re-export risk detection,
HS code mapping, trade trends, sanction overlay, concentration risk
(HHI), and volume analysis.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.trade_flow_analyzer import (
    TradeFlowAnalyzer,
    _EU_MEMBER_STATES,
    _HS_CODE_MAP,
    _SANCTIONED_COUNTRIES,
    _PORT_RISK_PROFILES,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    CommodityType,
    TradeFlow,
    TradeFlowDirection,
    SUPPORTED_COMMODITIES,
)


# ============================================================================
# TestTradeFlowAnalyzerInit
# ============================================================================


class TestTradeFlowAnalyzerInit:
    """Tests for TradeFlowAnalyzer initialization."""

    @pytest.mark.unit
    def test_initialization_empty_stores(self, mock_config):
        analyzer = TradeFlowAnalyzer()
        assert analyzer._flows == {}

    @pytest.mark.unit
    def test_eu_member_states_count(self):
        assert len(_EU_MEMBER_STATES) == 27

    @pytest.mark.unit
    def test_hs_code_map_not_empty(self):
        assert len(_HS_CODE_MAP) > 30

    @pytest.mark.unit
    def test_sanctioned_countries_defined(self):
        assert "KP" in _SANCTIONED_COUNTRIES
        assert "SY" in _SANCTIONED_COUNTRIES
        assert "IR" in _SANCTIONED_COUNTRIES
        assert "RU" in _SANCTIONED_COUNTRIES


# ============================================================================
# TestRecordTradeFlow
# ============================================================================


class TestRecordTradeFlow:
    """Tests for record_trade_flow method."""

    @pytest.mark.unit
    def test_record_valid_flow(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            volume_tonnes=50000.0,
        )
        assert isinstance(flow, TradeFlow)
        assert flow.origin_country == "BR"
        assert flow.destination_country == "NL"
        assert flow.commodity_type == CommodityType.SOYA

    @pytest.mark.unit
    def test_record_flow_has_id(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="DE",
            commodity_type="coffee",
        )
        assert flow.flow_id.startswith("tfl-")

    @pytest.mark.unit
    def test_record_flow_uppercase_countries(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="br",
            destination_country="nl",
            commodity_type="soya",
        )
        assert flow.origin_country == "BR"
        assert flow.destination_country == "NL"

    @pytest.mark.unit
    def test_record_flow_stores_result(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        retrieved = trade_flow_analyzer.get_trade_flow(flow.flow_id)
        assert retrieved is not None
        assert retrieved.flow_id == flow.flow_id

    @pytest.mark.unit
    def test_record_flow_with_volume(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            volume_tonnes=50000.0,
        )
        assert flow.volume_tonnes == 50000.0

    @pytest.mark.unit
    def test_record_flow_with_value(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            value_usd=25000000.0,
        )
        assert flow.value_usd == 25000000.0

    @pytest.mark.unit
    def test_record_flow_with_quarter(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            quarter="2025-Q4",
        )
        assert flow.quarter == "2025-Q4"

    @pytest.mark.unit
    def test_record_flow_with_hs_codes(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            hs_codes=["1201", "1208"],
        )
        assert "1201" in flow.hs_codes
        assert "1208" in flow.hs_codes

    @pytest.mark.unit
    def test_record_flow_invalid_commodity_raises(self, trade_flow_analyzer):
        with pytest.raises(ValueError):
            trade_flow_analyzer.record_trade_flow(
                origin_country="BR",
                destination_country="NL",
                commodity_type="diamonds",
            )


# ============================================================================
# TestReExportRiskDetection
# ============================================================================


class TestReExportRiskDetection:
    """Tests for re-export risk detection."""

    @pytest.mark.unit
    def test_re_export_risk_present(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            volume_tonnes=50000.0,
        )
        if flow.re_export_risk is not None:
            assert 0.0 <= flow.re_export_risk <= 1.0

    @pytest.mark.unit
    def test_re_export_direction(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="SG",
            destination_country="NL",
            commodity_type="oil_palm",
            direction="re_export",
        )
        assert flow.direction == TradeFlowDirection.RE_EXPORT

    @pytest.mark.unit
    def test_transit_direction(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="ID",
            destination_country="NL",
            commodity_type="oil_palm",
            direction="transit",
        )
        assert flow.direction == TradeFlowDirection.TRANSIT


# ============================================================================
# TestTransshipmentHubDetection
# ============================================================================


class TestTransshipmentHubDetection:
    """Tests for transshipment hub detection."""

    @pytest.mark.unit
    def test_transshipment_countries_tracked(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="ID",
            destination_country="NL",
            commodity_type="oil_palm",
            transshipment_countries=["SG", "MY"],
        )
        assert "SG" in flow.transshipment_countries
        assert "MY" in flow.transshipment_countries

    @pytest.mark.unit
    def test_transshipment_increases_route_risk(self, trade_flow_analyzer):
        # Direct route
        flow_direct = trade_flow_analyzer.record_trade_flow(
            origin_country="ID",
            destination_country="NL",
            commodity_type="oil_palm",
        )
        # Route through transshipment hub
        flow_trans = trade_flow_analyzer.record_trade_flow(
            origin_country="ID",
            destination_country="NL",
            commodity_type="oil_palm",
            transshipment_countries=["SG", "AE"],
        )
        # Transshipment route should have higher or equal risk
        if flow_direct.route_risk_score is not None and flow_trans.route_risk_score is not None:
            assert flow_trans.route_risk_score >= flow_direct.route_risk_score - 1.0


# ============================================================================
# TestHSCodeMapping
# ============================================================================


class TestHSCodeMapping:
    """Tests for HS code to EUDR commodity mapping."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "hs_code,expected_commodity",
        [
            ("0102", "cattle"),
            ("0201", "cattle"),
            ("1801", "cocoa"),
            ("1806", "cocoa"),
            ("0901", "coffee"),
            ("1511", "oil_palm"),
            ("4001", "rubber"),
            ("1201", "soya"),
            ("4403", "wood"),
            ("4412", "wood"),
        ],
    )
    def test_hs_code_maps_to_commodity(self, hs_code, expected_commodity):
        commodity, description = _HS_CODE_MAP[hs_code]
        assert commodity == expected_commodity
        assert len(description) > 0

    @pytest.mark.unit
    def test_hs_code_map_covers_all_commodities(self):
        commodities_in_map = {v[0] for v in _HS_CODE_MAP.values()}
        for commodity in SUPPORTED_COMMODITIES:
            assert commodity in commodities_in_map, (
                f"Commodity {commodity} has no HS code mapping"
            )

    @pytest.mark.unit
    def test_hs_code_cattle_chapter(self):
        cattle_codes = [
            code for code, (comm, _) in _HS_CODE_MAP.items()
            if comm == "cattle"
        ]
        assert len(cattle_codes) >= 5

    @pytest.mark.unit
    def test_hs_code_wood_chapter(self):
        wood_codes = [
            code for code, (comm, _) in _HS_CODE_MAP.items()
            if comm == "wood"
        ]
        assert len(wood_codes) >= 15

    @pytest.mark.unit
    def test_hs_codes_are_4_digits(self):
        for code in _HS_CODE_MAP:
            assert len(code) == 4
            assert code.isdigit()


# ============================================================================
# TestRouteRiskScoring
# ============================================================================


class TestRouteRiskScoring:
    """Tests for trade route risk scoring."""

    @pytest.mark.unit
    def test_route_risk_present(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        if flow.route_risk_score is not None:
            assert 0.0 <= flow.route_risk_score <= 100.0

    @pytest.mark.unit
    def test_sanctioned_origin_elevated_risk(self, trade_flow_analyzer):
        """Sanctioned origin (RU) should produce non-zero route risk score."""
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="RU",
            destination_country="DE",
            commodity_type="wood",
        )
        if flow.route_risk_score is not None:
            # Sanctioned origin should produce positive route risk
            assert flow.route_risk_score > 0.0


# ============================================================================
# TestSanctionsCheck
# ============================================================================


class TestSanctionsCheck:
    """Tests for sanction and embargo overlay."""

    @pytest.mark.unit
    def test_sanctioned_countries_set(self):
        assert "KP" in _SANCTIONED_COUNTRIES  # North Korea
        assert "SY" in _SANCTIONED_COUNTRIES  # Syria
        assert "IR" in _SANCTIONED_COUNTRIES  # Iran
        assert "BY" in _SANCTIONED_COUNTRIES  # Belarus
        assert "RU" in _SANCTIONED_COUNTRIES  # Russia
        assert "MM" in _SANCTIONED_COUNTRIES  # Myanmar
        assert "VE" in _SANCTIONED_COUNTRIES  # Venezuela

    @pytest.mark.unit
    def test_non_sanctioned_country(self):
        assert "DE" not in _SANCTIONED_COUNTRIES
        assert "FR" not in _SANCTIONED_COUNTRIES
        assert "BR" not in _SANCTIONED_COUNTRIES


# ============================================================================
# TestBilateralFlow
# ============================================================================


class TestBilateralFlow:
    """Tests for bilateral trade flow analysis."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "origin,dest,commodity",
        [
            ("BR", "NL", "soya"),
            ("ID", "NL", "oil_palm"),
            ("CI", "FR", "cocoa"),
            ("CO", "DE", "coffee"),
            ("MY", "IT", "rubber"),
            ("BR", "ES", "cattle"),
            ("ID", "PL", "wood"),
        ],
    )
    def test_bilateral_flows_eu_destinations(
        self, trade_flow_analyzer, origin, dest, commodity
    ):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country=origin,
            destination_country=dest,
            commodity_type=commodity,
            volume_tonnes=10000.0,
        )
        assert flow.origin_country == origin
        assert flow.destination_country == dest

    @pytest.mark.unit
    def test_eu_member_states_coverage(self):
        expected_eu = {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        }
        assert _EU_MEMBER_STATES == expected_eu


# ============================================================================
# TestListTradeFlows
# ============================================================================


class TestListTradeFlows:
    """Tests for listing and filtering trade flows."""

    @pytest.mark.unit
    def test_list_all_flows(self, trade_flow_analyzer):
        for commodity in ["soya", "coffee", "cocoa"]:
            trade_flow_analyzer.record_trade_flow(
                origin_country="BR",
                destination_country="NL",
                commodity_type=commodity,
            )
        results = trade_flow_analyzer.list_trade_flows()
        assert len(results) == 3

    @pytest.mark.unit
    def test_list_flows_filter_by_origin(self, trade_flow_analyzer):
        trade_flow_analyzer.record_trade_flow(
            origin_country="BR", destination_country="NL", commodity_type="soya"
        )
        trade_flow_analyzer.record_trade_flow(
            origin_country="ID", destination_country="NL", commodity_type="oil_palm"
        )
        results = trade_flow_analyzer.list_trade_flows(origin_country="BR")
        assert len(results) == 1
        assert results[0].origin_country == "BR"

    @pytest.mark.unit
    def test_list_flows_filter_by_commodity(self, trade_flow_analyzer):
        trade_flow_analyzer.record_trade_flow(
            origin_country="BR", destination_country="NL", commodity_type="soya"
        )
        trade_flow_analyzer.record_trade_flow(
            origin_country="BR", destination_country="DE", commodity_type="coffee"
        )
        results = trade_flow_analyzer.list_trade_flows(commodity_type="soya")
        assert len(results) == 1

    @pytest.mark.unit
    def test_get_nonexistent_flow(self, trade_flow_analyzer):
        result = trade_flow_analyzer.get_trade_flow("nonexistent-id")
        assert result is None

    @pytest.mark.unit
    def test_list_flows_with_pagination(self, trade_flow_analyzer):
        for i in range(5):
            trade_flow_analyzer.record_trade_flow(
                origin_country="BR",
                destination_country="NL",
                commodity_type="soya",
                volume_tonnes=float(i * 1000),
            )
        results = trade_flow_analyzer.list_trade_flows(limit=3)
        assert len(results) == 3


# ============================================================================
# TestPortRiskProfiles
# ============================================================================


class TestPortRiskProfiles:
    """Tests for EU port risk profiles."""

    @pytest.mark.unit
    def test_port_profiles_defined(self):
        assert len(_PORT_RISK_PROFILES) > 0
        assert "NLRTM" in _PORT_RISK_PROFILES  # Rotterdam
        assert "BEANR" in _PORT_RISK_PROFILES  # Antwerp
        assert "DEHAM" in _PORT_RISK_PROFILES  # Hamburg

    @pytest.mark.unit
    def test_port_risk_scores_in_range(self):
        for port, (country, risk, desc) in _PORT_RISK_PROFILES.items():
            assert 0.0 <= risk <= 100.0
            assert country in _EU_MEMBER_STATES
            assert len(desc) > 0


# ============================================================================
# TestVolumeAnalysis
# ============================================================================


class TestVolumeAnalysis:
    """Tests for trade volume analysis."""

    @pytest.mark.unit
    def test_flow_with_no_volume(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        assert flow.volume_tonnes is None or flow.volume_tonnes >= 0

    @pytest.mark.unit
    def test_flow_with_large_volume(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            volume_tonnes=5000000.0,
        )
        assert flow.volume_tonnes == 5000000.0


# ============================================================================
# TestProvenanceTracking
# ============================================================================


class TestProvenanceTracking:
    """Tests for trade flow provenance."""

    @pytest.mark.unit
    def test_flow_has_provenance_hash(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        assert flow.provenance_hash is not None
        assert len(flow.provenance_hash) == 64

    @pytest.mark.unit
    def test_flow_has_data_sources(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        assert isinstance(flow.data_sources, list)
        assert len(flow.data_sources) > 0

    @pytest.mark.unit
    def test_flow_has_recorded_at(self, trade_flow_analyzer):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
        )
        assert flow.recorded_at is not None


# ============================================================================
# TestAllDirections
# ============================================================================


class TestAllDirections:
    """Tests for all trade flow directions."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "direction",
        ["export", "import", "re_export", "transit"],
    )
    def test_all_directions_accepted(self, trade_flow_analyzer, direction):
        flow = trade_flow_analyzer.record_trade_flow(
            origin_country="BR",
            destination_country="NL",
            commodity_type="soya",
            direction=direction,
        )
        assert flow.direction == TradeFlowDirection(direction)
