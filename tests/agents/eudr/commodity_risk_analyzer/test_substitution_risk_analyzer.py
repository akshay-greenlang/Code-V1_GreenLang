# -*- coding: utf-8 -*-
"""
Unit tests for SubstitutionRiskAnalyzer (AGENT-EUDR-018 Engine 5).

Tests commodity switching detection, greenwashing identification,
declaration verification, cross-commodity risk matrix, seasonal
switching analysis, and batch screening for all 7 EUDR commodities.

Coverage target: 85%+
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.engines.substitution_risk_analyzer import (
    SubstitutionRiskAnalyzer,
    EUDR_COMMODITIES,
    SUBSTITUTION_RISK_MATRIX,
    RECOGNIZED_CERTIFICATIONS,
    SEVERITY_THRESHOLDS,
    SubstitutionAlert,
    SwitchingPattern,
    GreenwashingResult,
)

SEVEN_COMMODITIES = sorted(EUDR_COMMODITIES)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for SubstitutionRiskAnalyzer initialization."""

    @pytest.mark.unit
    def test_init_creates_empty_stores(self):
        """Analyzer initializes with empty supplier history and alerts."""
        analyzer = SubstitutionRiskAnalyzer()
        assert analyzer._supplier_history == {}
        assert analyzer._alerts == {}

    @pytest.mark.unit
    def test_init_creates_lock(self):
        """Analyzer creates a reentrant lock on init."""
        analyzer = SubstitutionRiskAnalyzer()
        assert analyzer._lock is not None


# ---------------------------------------------------------------------------
# TestDetectSubstitution
# ---------------------------------------------------------------------------

class TestDetectSubstitution:
    """Tests for detect_substitution method."""

    @pytest.mark.unit
    def test_no_switch_no_detection(self, substitution_risk_analyzer):
        """Same commodity in history and declaration -> not detected."""
        history = [
            {"commodity": "soya", "date": "2025-01-15", "volume": 100},
            {"commodity": "soya", "date": "2025-06-20", "volume": 110},
        ]
        declaration = {"commodity": "soya", "volume": 105}
        result = substitution_risk_analyzer.detect_substitution(
            "SUP-001", history, declaration,
        )
        assert result["substitution_detected"] is False

    @pytest.mark.unit
    def test_switch_detected(self, substitution_risk_analyzer):
        """Different commodity triggers substitution detection."""
        history = [
            {"commodity": "soya", "date": "2025-01-15", "volume": 100},
            {"commodity": "soya", "date": "2025-06-20", "volume": 110},
        ]
        declaration = {"commodity": "cattle", "volume": 90}
        result = substitution_risk_analyzer.detect_substitution(
            "SUP-002", history, declaration,
        )
        assert result["substitution_detected"] is True
        assert result["previous_commodity"] == "soya"
        assert result["current_commodity"] == "cattle"
        assert result["alert"] is not None

    @pytest.mark.unit
    def test_empty_history_no_switch(self, substitution_risk_analyzer):
        """Empty history results in no substitution detected."""
        result = substitution_risk_analyzer.detect_substitution(
            "SUP-003", [], {"commodity": "cocoa", "volume": 50},
        )
        assert result["substitution_detected"] is False

    @pytest.mark.unit
    def test_invalid_supplier_id_raises(self, substitution_risk_analyzer):
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            substitution_risk_analyzer.detect_substitution(
                "", [], {"commodity": "cocoa", "volume": 50},
            )

    @pytest.mark.unit
    def test_invalid_commodity_in_declaration_raises(self, substitution_risk_analyzer):
        """Invalid commodity in declaration raises ValueError."""
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            substitution_risk_analyzer.detect_substitution(
                "SUP-004", [], {"commodity": "banana", "volume": 50},
            )


# ---------------------------------------------------------------------------
# TestSwitchingPattern
# ---------------------------------------------------------------------------

class TestSwitchingPattern:
    """Tests for analyze_switching_pattern method."""

    @pytest.mark.unit
    def test_pattern_with_transitions(self, substitution_risk_analyzer):
        """Supplier with transitions shows transition_count > 0."""
        # Seed history via detect_substitution
        history = [
            {"commodity": "soya", "date": "2025-01-15", "volume": 100},
            {"commodity": "cattle", "date": "2025-06-20", "volume": 80},
            {"commodity": "soya", "date": "2025-09-10", "volume": 110},
        ]
        substitution_risk_analyzer.detect_substitution(
            "SUP-PAT", history, {"commodity": "soya", "volume": 105},
        )
        result = substitution_risk_analyzer.analyze_switching_pattern("SUP-PAT")
        assert result["transition_count"] >= 2

    @pytest.mark.unit
    def test_pattern_no_history(self, substitution_risk_analyzer):
        """Supplier with no history returns 0 transitions."""
        result = substitution_risk_analyzer.analyze_switching_pattern("SUP-NONE")
        assert result["transition_count"] == 0

    @pytest.mark.unit
    def test_invalid_time_window(self, substitution_risk_analyzer):
        """time_window_days < 1 raises ValueError."""
        with pytest.raises(ValueError, match="time_window_days"):
            substitution_risk_analyzer.analyze_switching_pattern("SUP-X", 0)


# ---------------------------------------------------------------------------
# TestSubstitutionRisk
# ---------------------------------------------------------------------------

class TestSubstitutionRisk:
    """Tests for calculate_substitution_risk method."""

    @pytest.mark.unit
    def test_no_history_base_risk(self, substitution_risk_analyzer):
        """No history returns base risk of 10."""
        score = substitution_risk_analyzer.calculate_substitution_risk("SUP-NEW", "cocoa")
        assert score == Decimal("10")

    @pytest.mark.unit
    def test_with_switch_history(self, substitution_risk_analyzer):
        """Supplier that switched commodities has risk > base."""
        history = [
            {"commodity": "soya", "date": "2025-01-15", "volume": 100},
            {"commodity": "cattle", "date": "2025-06-20", "volume": 80},
        ]
        substitution_risk_analyzer.detect_substitution(
            "SUP-RISK", history, {"commodity": "cattle", "volume": 90},
        )
        score = substitution_risk_analyzer.calculate_substitution_risk("SUP-RISK", "cattle")
        assert score > Decimal("10")
        assert score <= Decimal("100")

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, substitution_risk_analyzer):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError):
            substitution_risk_analyzer.calculate_substitution_risk("SUP-X", "wheat")


# ---------------------------------------------------------------------------
# TestGreenwashingDetection
# ---------------------------------------------------------------------------

class TestGreenwashingDetection:
    """Tests for detect_greenwashing method."""

    @pytest.mark.unit
    def test_valid_certification_passes(self, substitution_risk_analyzer):
        """RSPO cert on oil_palm is verified, no greenwashing detected."""
        result = substitution_risk_analyzer.detect_greenwashing(
            "oil_palm",
            {"origin": "ID"},
            ["RSPO"],
        )
        assert "RSPO" in result["certifications_verified"]
        # Greenwashing may still be detected if origin is high risk
        # but RSPO itself should pass

    @pytest.mark.unit
    def test_wrong_commodity_cert_fails(self, substitution_risk_analyzer):
        """RSPO cert on coffee is not applicable and fails verification."""
        result = substitution_risk_analyzer.detect_greenwashing(
            "coffee",
            {"origin": "BR"},
            ["RSPO"],
        )
        assert "RSPO" in result["certifications_failed"]
        assert result["greenwashing_detected"] is True

    @pytest.mark.unit
    def test_unrecognized_cert_fails(self, substitution_risk_analyzer):
        """Unrecognized certification is flagged as failed."""
        result = substitution_risk_analyzer.detect_greenwashing(
            "cocoa",
            {"origin": "GH"},
            ["FAKE_CERT"],
        )
        assert "FAKE_CERT" in result["certifications_failed"]

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, substitution_risk_analyzer):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            substitution_risk_analyzer.detect_greenwashing("banana", {}, [])


# ---------------------------------------------------------------------------
# TestVerifyDeclaration
# ---------------------------------------------------------------------------

class TestVerifyDeclaration:
    """Tests for verify_commodity_declaration method."""

    @pytest.mark.unit
    def test_consistent_evidence_passes(self, substitution_risk_analyzer):
        """Consistent trade record evidence passes verification."""
        declaration = {
            "commodity": "soya",
            "volume": 100,
            "origin_country": "BR",
            "supplier_id": "SUP-V1",
        }
        evidence = [
            {"type": "trade_record", "source": "customs", "data": {"commodity": "soya", "volume": 100}},
        ]
        result = substitution_risk_analyzer.verify_commodity_declaration(declaration, evidence)
        assert result["declaration_verified"] is True

    @pytest.mark.unit
    def test_inconsistent_trade_record_fails(self, substitution_risk_analyzer):
        """Trade record with different commodity flags inconsistency."""
        declaration = {
            "commodity": "soya",
            "volume": 100,
            "origin_country": "BR",
            "supplier_id": "SUP-V2",
        }
        evidence = [
            {"type": "trade_record", "source": "customs", "data": {"commodity": "cattle", "volume": 100}},
        ]
        result = substitution_risk_analyzer.verify_commodity_declaration(declaration, evidence)
        assert result["declaration_verified"] is False
        assert result["alert"] is not None

    @pytest.mark.unit
    def test_missing_declaration_fields_raises(self, substitution_risk_analyzer):
        """Declaration missing required fields raises ValueError."""
        with pytest.raises(ValueError, match="missing required fields"):
            substitution_risk_analyzer.verify_commodity_declaration(
                {"commodity": "soya"},
                [],
            )


# ---------------------------------------------------------------------------
# TestAlerts
# ---------------------------------------------------------------------------

class TestAlerts:
    """Tests for get_substitution_alerts method."""

    @pytest.mark.unit
    def test_get_all_alerts(self, substitution_risk_analyzer):
        """Get alerts returns list."""
        alerts = substitution_risk_analyzer.get_substitution_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.unit
    def test_filter_by_severity(self, substitution_risk_analyzer):
        """Filter by severity returns only matching or higher severity."""
        # First create an alert by detecting a substitution
        history = [{"commodity": "wood", "date": "2025-01-15", "volume": 100}]
        substitution_risk_analyzer.detect_substitution(
            "SUP-ALT", history, {"commodity": "cattle", "volume": 90},
        )
        all_alerts = substitution_risk_analyzer.get_substitution_alerts()
        filtered = substitution_risk_analyzer.get_substitution_alerts("HIGH")
        assert len(filtered) <= len(all_alerts)

    @pytest.mark.unit
    def test_invalid_severity_raises(self, substitution_risk_analyzer):
        """Invalid severity threshold raises ValueError."""
        with pytest.raises(ValueError, match="Invalid severity_threshold"):
            substitution_risk_analyzer.get_substitution_alerts("ULTRA")


# ---------------------------------------------------------------------------
# TestCrossCommodityRisk
# ---------------------------------------------------------------------------

class TestCrossCommodityRisk:
    """Tests for calculate_cross_commodity_risk method."""

    @pytest.mark.unit
    def test_known_pair_risk(self, substitution_risk_analyzer):
        """Known pair (wood -> cattle) returns CRITICAL concern."""
        result = substitution_risk_analyzer.calculate_cross_commodity_risk("wood", "cattle")
        assert result["risk_weight"] == str(Decimal("92"))
        assert result["regulatory_concern"] == "CRITICAL"

    @pytest.mark.unit
    def test_unknown_pair_default(self, substitution_risk_analyzer):
        """Unknown pair gets default risk_weight 40."""
        result = substitution_risk_analyzer.calculate_cross_commodity_risk("cattle", "cocoa")
        # This pair is not in the matrix
        if ("cattle", "cocoa") not in SUBSTITUTION_RISK_MATRIX:
            assert result["risk_weight"] == str(Decimal("40"))

    @pytest.mark.unit
    def test_same_commodity_raises(self, substitution_risk_analyzer):
        """Same from/to commodity raises ValueError."""
        with pytest.raises(ValueError, match="must be different"):
            substitution_risk_analyzer.calculate_cross_commodity_risk("soya", "soya")

    @pytest.mark.unit
    def test_asymmetric_flag(self, substitution_risk_analyzer):
        """soya->cattle vs cattle->soya shows asymmetric risk."""
        result = substitution_risk_analyzer.calculate_cross_commodity_risk("soya", "cattle")
        assert "asymmetric" in result


# ---------------------------------------------------------------------------
# TestSeasonalSwitching
# ---------------------------------------------------------------------------

class TestSeasonalSwitching:
    """Tests for analyze_seasonal_switching method."""

    @pytest.mark.unit
    def test_seasonal_no_history(self, substitution_risk_analyzer):
        """Supplier with no history returns no seasonal pattern."""
        result = substitution_risk_analyzer.analyze_seasonal_switching("SUP-SEAS-NEW")
        assert result["seasonal_pattern_detected"] is False

    @pytest.mark.unit
    def test_seasonal_with_switching(self, substitution_risk_analyzer):
        """Supplier that switches by quarter triggers detection."""
        history = [
            {"commodity": "soya", "date": "2025-01-15", "volume": 100},
            {"commodity": "soya", "date": "2025-03-15", "volume": 110},
            {"commodity": "cattle", "date": "2025-07-15", "volume": 80},
            {"commodity": "cattle", "date": "2025-09-15", "volume": 85},
        ]
        substitution_risk_analyzer.detect_substitution(
            "SUP-SEAS", history, {"commodity": "cattle", "volume": 85},
        )
        result = substitution_risk_analyzer.analyze_seasonal_switching("SUP-SEAS")
        assert result["seasonal_pattern_detected"] is True

    @pytest.mark.unit
    def test_empty_supplier_raises(self, substitution_risk_analyzer):
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            substitution_risk_analyzer.analyze_seasonal_switching("")


# ---------------------------------------------------------------------------
# TestBatchScreen
# ---------------------------------------------------------------------------

class TestBatchScreen:
    """Tests for batch_screen method."""

    @pytest.mark.unit
    def test_batch_returns_results(self, substitution_risk_analyzer):
        """Batch screening returns one result per supplier."""
        suppliers = [
            {"supplier_id": "S1", "declared_commodity": "soya"},
            {"supplier_id": "S2", "declared_commodity": "cocoa"},
        ]
        results = substitution_risk_analyzer.batch_screen(suppliers)
        assert len(results) == 2

    @pytest.mark.unit
    def test_batch_missing_fields_handled(self, substitution_risk_analyzer):
        """Supplier with missing fields gets error entry."""
        suppliers = [{"supplier_id": "", "declared_commodity": ""}]
        results = substitution_risk_analyzer.batch_screen(suppliers)
        assert "error" in results[0]

    @pytest.mark.unit
    def test_batch_exceeds_limit_raises(self, substitution_risk_analyzer):
        """Batch exceeding 10000 raises ValueError."""
        big_batch = [{"supplier_id": f"S{i}", "declared_commodity": "soya"} for i in range(10001)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            substitution_risk_analyzer.batch_screen(big_batch)

    @pytest.mark.unit
    def test_batch_not_list_raises(self, substitution_risk_analyzer):
        """Non-list input raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            substitution_risk_analyzer.batch_screen("not_a_list")


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance hash integrity."""

    @pytest.mark.unit
    def test_detection_provenance_hash(self, substitution_risk_analyzer):
        """detect_substitution result has a 64-char provenance hash."""
        result = substitution_risk_analyzer.detect_substitution(
            "SUP-PROV", [], {"commodity": "wood", "volume": 50},
        )
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_cross_commodity_provenance(self, substitution_risk_analyzer):
        """Cross-commodity risk result has provenance hash."""
        result = substitution_risk_analyzer.calculate_cross_commodity_risk("oil_palm", "rubber")
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for boundary conditions and error handling."""

    @pytest.mark.unit
    def test_detect_non_dict_declaration_raises(self, substitution_risk_analyzer):
        """Non-dict current_declaration raises ValueError."""
        with pytest.raises(ValueError, match="current_declaration must be a dict"):
            substitution_risk_analyzer.detect_substitution(
                "SUP-ERR", [], "not_a_dict",
            )

    @pytest.mark.unit
    def test_detect_missing_commodity_in_declaration(self, substitution_risk_analyzer):
        """Declaration missing 'commodity' raises ValueError."""
        with pytest.raises(ValueError, match="must include 'commodity'"):
            substitution_risk_analyzer.detect_substitution(
                "SUP-ERR2", [], {"volume": 50},
            )

    @pytest.mark.unit
    def test_verify_invalid_commodity_raises(self, substitution_risk_analyzer):
        """Declaration with invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="not valid"):
            substitution_risk_analyzer.verify_commodity_declaration(
                {"commodity": "banana", "volume": 10, "origin_country": "EC", "supplier_id": "S1"},
                [],
            )
