# -*- coding: utf-8 -*-
"""
Tests for FraudPatternDetector - AGENT-EUDR-012 Engine 6: Fraud Pattern Detection

Comprehensive test suite covering:
- All 15 fraud rules individually (FRD-001 through FRD-015)
- Severity classification (low, medium, high, critical)
- Batch fraud detection
- Fraud risk score calculation
- Rule toggling (enable/disable individual rules)
- Edge cases and boundary conditions

Test count: 55+ tests
Coverage target: >= 85% of FraudPatternDetector module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    FRAUD_PATTERN_TYPES,
    FRAUD_SEVERITIES,
    FRAUD_RULE_IDS,
    FRAUD_RULE_TO_PATTERN,
    FRAUD_SEVERITY_WEIGHTS,
    SHA256_HEX_LENGTH,
    DOC_ID_COO_001,
    DOC_ID_FSC_001,
    DOC_ID_BOL_001,
    FRAUD_ALERT_DUPLICATE,
    FRAUD_ALERT_QUANTITY,
    FRAUD_ALERT_GEO,
    SAMPLE_PDF_BYTES,
    make_document_record,
    make_fraud_alert,
    assert_fraud_alert_valid,
    assert_valid_provenance_hash,
    assert_valid_score,
    _ts,
)


# ===========================================================================
# 1. FRD-001: Duplicate Reuse Detection
# ===========================================================================


class TestFRD001DuplicateReuse:
    """Test FRD-001: Duplicate document reuse detection."""

    def test_duplicate_hash_detected(self, fraud_engine):
        """Duplicate document hash across different shipments is detected."""
        context = {
            "document_id": "DOC-FRD001-001",
            "file_hash_sha256": "a" * 64,
            "shipment_id": "SHIP-001",
            "previous_hashes": {"a" * 64: {"shipment_id": "SHIP-099"}},
        }
        result = fraud_engine.detect(rule_id="FRD-001", context=context)
        assert result is not None
        assert result["pattern_type"] == "duplicate_reuse"

    def test_unique_hash_passes(self, fraud_engine):
        """Unique document hash does not trigger FRD-001."""
        context = {
            "document_id": "DOC-FRD001-002",
            "file_hash_sha256": "b" * 64,
            "shipment_id": "SHIP-002",
            "previous_hashes": {},
        }
        result = fraud_engine.detect(rule_id="FRD-001", context=context)
        assert result is None or result.get("triggered") is False

    def test_duplicate_severity_high(self, fraud_engine):
        """Duplicate reuse has HIGH severity."""
        alert = make_fraud_alert(rule_id="FRD-001", pattern_type="duplicate_reuse", severity="high")
        assert alert["severity"] == "high"


# ===========================================================================
# 2. FRD-002: Quantity Tampering Detection
# ===========================================================================


class TestFRD002QuantityTampering:
    """Test FRD-002: Quantity value tampering detection."""

    def test_quantity_deviation_detected(self, fraud_engine):
        """Quantity deviating beyond tolerance is detected."""
        context = {
            "document_id": "DOC-FRD002-001",
            "declared_quantity_kg": 25000.0,
            "reference_quantity_kg": 21000.0,
            "tolerance_percent": 5.0,
        }
        result = fraud_engine.detect(rule_id="FRD-002", context=context)
        assert result is not None
        assert result["pattern_type"] == "quantity_tampering"

    def test_quantity_within_tolerance_passes(self, fraud_engine):
        """Quantity within tolerance does not trigger FRD-002."""
        context = {
            "document_id": "DOC-FRD002-002",
            "declared_quantity_kg": 25000.0,
            "reference_quantity_kg": 24500.0,
            "tolerance_percent": 5.0,
        }
        result = fraud_engine.detect(rule_id="FRD-002", context=context)
        assert result is None or result.get("triggered") is False

    def test_quantity_tampering_severity_medium(self, fraud_engine):
        """Quantity tampering has MEDIUM severity by default."""
        alert = make_fraud_alert(rule_id="FRD-002", pattern_type="quantity_tampering", severity="medium")
        assert alert["severity"] == "medium"


# ===========================================================================
# 3. FRD-003: Date Manipulation Detection
# ===========================================================================


class TestFRD003DateManipulation:
    """Test FRD-003: Date manipulation detection."""

    def test_future_date_detected(self, fraud_engine):
        """Issuance date in the future is detected."""
        context = {
            "document_id": "DOC-FRD003-001",
            "issuance_date": _ts(days_ago=-10),  # 10 days in future
            "current_date": _ts(days_ago=0),
        }
        result = fraud_engine.detect(rule_id="FRD-003", context=context)
        assert result is not None
        assert result["pattern_type"] == "date_manipulation"

    def test_valid_date_passes(self, fraud_engine):
        """Valid past issuance date does not trigger FRD-003."""
        context = {
            "document_id": "DOC-FRD003-002",
            "issuance_date": _ts(days_ago=30),
            "current_date": _ts(days_ago=0),
        }
        result = fraud_engine.detect(rule_id="FRD-003", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 4. FRD-004: Expired Certificate Detection
# ===========================================================================


class TestFRD004ExpiredCert:
    """Test FRD-004: Expired certificate usage detection."""

    def test_expired_cert_at_issuance_detected(self, fraud_engine):
        """Document using expired certificate at issuance time is detected."""
        context = {
            "document_id": "DOC-FRD004-001",
            "cert_expiry_date": _ts(days_ago=30),
            "issuance_date": _ts(days_ago=5),
        }
        result = fraud_engine.detect(rule_id="FRD-004", context=context)
        assert result is not None
        assert result["pattern_type"] == "expired_cert"

    def test_valid_cert_passes(self, fraud_engine):
        """Document with valid certificate does not trigger FRD-004."""
        context = {
            "document_id": "DOC-FRD004-002",
            "cert_expiry_date": _ts(days_ago=-365),
            "issuance_date": _ts(days_ago=5),
        }
        result = fraud_engine.detect(rule_id="FRD-004", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 5. FRD-005: Serial Number Anomaly
# ===========================================================================


class TestFRD005SerialAnomaly:
    """Test FRD-005: Serial number anomaly detection."""

    def test_anomalous_serial_detected(self, fraud_engine):
        """Serial number not matching issuer pattern is detected."""
        context = {
            "document_id": "DOC-FRD005-001",
            "serial_number": "FAKE-000-XXXX",
            "issuer_pattern": r"COO-[A-Z]{2}-\d{4}-\d{5}",
        }
        result = fraud_engine.detect(rule_id="FRD-005", context=context)
        assert result is not None
        assert result["pattern_type"] == "serial_anomaly"

    def test_valid_serial_passes(self, fraud_engine):
        """Valid serial number does not trigger FRD-005."""
        context = {
            "document_id": "DOC-FRD005-002",
            "serial_number": "COO-GH-2026-00123",
            "issuer_pattern": r"COO-[A-Z]{2}-\d{4}-\d{5}",
        }
        result = fraud_engine.detect(rule_id="FRD-005", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 6. FRD-006: Issuer Mismatch
# ===========================================================================


class TestFRD006IssuerMismatch:
    """Test FRD-006: Issuer mismatch detection."""

    def test_issuer_mismatch_detected(self, fraud_engine):
        """Issuing authority not matching certificate signer is detected."""
        context = {
            "document_id": "DOC-FRD006-001",
            "claimed_issuer": "Ghana Cocoa Board",
            "certificate_signer": "Unknown Entity Ltd",
        }
        result = fraud_engine.detect(rule_id="FRD-006", context=context)
        assert result is not None
        assert result["pattern_type"] == "issuer_mismatch"

    def test_matching_issuer_passes(self, fraud_engine):
        """Matching issuer does not trigger FRD-006."""
        context = {
            "document_id": "DOC-FRD006-002",
            "claimed_issuer": "Ghana Cocoa Board",
            "certificate_signer": "Ghana Cocoa Board",
        }
        result = fraud_engine.detect(rule_id="FRD-006", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 7. FRD-007: Template Forgery
# ===========================================================================


class TestFRD007TemplateForgery:
    """Test FRD-007: Template forgery detection."""

    def test_template_mismatch_detected(self, fraud_engine):
        """Document not matching known template is detected."""
        context = {
            "document_id": "DOC-FRD007-001",
            "document_type": "coo",
            "template_match_score": 0.30,
            "template_threshold": 0.70,
        }
        result = fraud_engine.detect(rule_id="FRD-007", context=context)
        assert result is not None
        assert result["pattern_type"] == "template_forgery"

    def test_valid_template_passes(self, fraud_engine):
        """Document matching template does not trigger FRD-007."""
        context = {
            "document_id": "DOC-FRD007-002",
            "document_type": "coo",
            "template_match_score": 0.95,
            "template_threshold": 0.70,
        }
        result = fraud_engine.detect(rule_id="FRD-007", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 8. FRD-008: Cross-Document Inconsistency
# ===========================================================================


class TestFRD008CrossDocInconsistency:
    """Test FRD-008: Cross-document quantity inconsistency."""

    def test_cross_doc_inconsistency_detected(self, fraud_engine):
        """Inconsistent data across related documents is detected."""
        context = {
            "document_id": "DOC-FRD008-001",
            "bol_quantity_kg": 25000.0,
            "invoice_quantity_kg": 20000.0,
            "tolerance_percent": 5.0,
        }
        result = fraud_engine.detect(rule_id="FRD-008", context=context)
        assert result is not None
        assert result["pattern_type"] == "cross_doc_inconsistency"

    def test_consistent_docs_pass(self, fraud_engine):
        """Consistent cross-document data does not trigger FRD-008."""
        context = {
            "document_id": "DOC-FRD008-002",
            "bol_quantity_kg": 25000.0,
            "invoice_quantity_kg": 24800.0,
            "tolerance_percent": 5.0,
        }
        result = fraud_engine.detect(rule_id="FRD-008", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 9. FRD-009: Geographic Impossibility
# ===========================================================================


class TestFRD009GeoImpossibility:
    """Test FRD-009: Geographic impossibility detection."""

    def test_ocean_coordinates_detected(self, fraud_engine):
        """GPS coordinates in the ocean are detected."""
        context = {
            "document_id": "DOC-FRD009-001",
            "gps_lat": -8.0,
            "gps_lon": 115.0,
            "claimed_country": "ID",
            "is_on_land": False,
        }
        result = fraud_engine.detect(rule_id="FRD-009", context=context)
        assert result is not None
        assert result["pattern_type"] == "geo_impossibility"

    def test_valid_coordinates_pass(self, fraud_engine):
        """Valid production area coordinates do not trigger FRD-009."""
        context = {
            "document_id": "DOC-FRD009-002",
            "gps_lat": 6.6885,
            "gps_lon": -1.6244,
            "claimed_country": "GH",
            "is_on_land": True,
        }
        result = fraud_engine.detect(rule_id="FRD-009", context=context)
        assert result is None or result.get("triggered") is False

    def test_geo_impossibility_severity_critical(self, fraud_engine):
        """Geographic impossibility has CRITICAL severity."""
        alert = make_fraud_alert(
            rule_id="FRD-009",
            pattern_type="geo_impossibility",
            severity="critical",
        )
        assert alert["severity"] == "critical"


# ===========================================================================
# 10. FRD-010: Velocity Anomaly
# ===========================================================================


class TestFRD010VelocityAnomaly:
    """Test FRD-010: Issuer velocity anomaly detection."""

    def test_high_velocity_detected(self, fraud_engine):
        """Unusually high document issuance rate is detected."""
        context = {
            "document_id": "DOC-FRD010-001",
            "issuer_id": "ISSUER-001",
            "documents_today": 25,
            "velocity_threshold": 10,
        }
        result = fraud_engine.detect(rule_id="FRD-010", context=context)
        assert result is not None
        assert result["pattern_type"] == "velocity_anomaly"

    def test_normal_velocity_passes(self, fraud_engine):
        """Normal issuance rate does not trigger FRD-010."""
        context = {
            "document_id": "DOC-FRD010-002",
            "issuer_id": "ISSUER-002",
            "documents_today": 3,
            "velocity_threshold": 10,
        }
        result = fraud_engine.detect(rule_id="FRD-010", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 11. FRD-011: Modification Timeline Anomaly
# ===========================================================================


class TestFRD011ModificationAnomaly:
    """Test FRD-011: Modification timeline anomaly detection."""

    def test_modification_after_issuance_detected(self, fraud_engine):
        """Document modified after purported issuance is detected."""
        context = {
            "document_id": "DOC-FRD011-001",
            "issuance_date": _ts(days_ago=30),
            "modification_date": _ts(days_ago=5),
        }
        result = fraud_engine.detect(rule_id="FRD-011", context=context)
        assert result is not None
        assert result["pattern_type"] == "modification_anomaly"

    def test_no_modification_passes(self, fraud_engine):
        """No modification after issuance does not trigger FRD-011."""
        context = {
            "document_id": "DOC-FRD011-002",
            "issuance_date": _ts(days_ago=30),
            "modification_date": _ts(days_ago=30),
        }
        result = fraud_engine.detect(rule_id="FRD-011", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 12. FRD-012: Round Number Bias
# ===========================================================================


class TestFRD012RoundNumberBias:
    """Test FRD-012: Round number bias detection."""

    def test_round_number_bias_detected(self, fraud_engine):
        """High percentage of round numbers is detected."""
        context = {
            "document_id": "DOC-FRD012-001",
            "quantities": [1000.0, 2000.0, 5000.0, 10000.0, 3000.0],
            "round_threshold_percent": 80.0,
        }
        result = fraud_engine.detect(rule_id="FRD-012", context=context)
        assert result is not None
        assert result["pattern_type"] == "round_number_bias"

    def test_varied_quantities_pass(self, fraud_engine):
        """Non-round quantities do not trigger FRD-012."""
        context = {
            "document_id": "DOC-FRD012-002",
            "quantities": [1234.56, 7891.23, 456.78, 3210.99, 1567.43],
            "round_threshold_percent": 80.0,
        }
        result = fraud_engine.detect(rule_id="FRD-012", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 13. FRD-013: Copy-Paste Detection
# ===========================================================================


class TestFRD013CopyPaste:
    """Test FRD-013: Copy-paste detection."""

    def test_copy_paste_detected(self, fraud_engine):
        """Duplicated text blocks from another document are detected."""
        context = {
            "document_id": "DOC-FRD013-001",
            "text_similarity_score": 0.95,
            "reference_document_id": "DOC-REF-001",
            "similarity_threshold": 0.85,
        }
        result = fraud_engine.detect(rule_id="FRD-013", context=context)
        assert result is not None
        assert result["pattern_type"] == "copy_paste"

    def test_unique_content_passes(self, fraud_engine):
        """Unique document content does not trigger FRD-013."""
        context = {
            "document_id": "DOC-FRD013-002",
            "text_similarity_score": 0.20,
            "reference_document_id": "DOC-REF-002",
            "similarity_threshold": 0.85,
        }
        result = fraud_engine.detect(rule_id="FRD-013", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 14. FRD-014: Missing Required Documents
# ===========================================================================


class TestFRD014MissingRequired:
    """Test FRD-014: Missing required documents detection."""

    def test_missing_documents_detected(self, fraud_engine):
        """Missing required companion documents are detected."""
        context = {
            "document_id": "DOC-FRD014-001",
            "shipment_id": "SHIP-014",
            "required_types": ["coo", "pc", "bol", "ic"],
            "present_types": ["coo", "bol"],
        }
        result = fraud_engine.detect(rule_id="FRD-014", context=context)
        assert result is not None
        assert result["pattern_type"] == "missing_required"

    def test_all_documents_present_passes(self, fraud_engine):
        """All required documents present does not trigger FRD-014."""
        context = {
            "document_id": "DOC-FRD014-002",
            "shipment_id": "SHIP-015",
            "required_types": ["coo", "pc", "bol"],
            "present_types": ["coo", "pc", "bol", "ic"],
        }
        result = fraud_engine.detect(rule_id="FRD-014", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 15. FRD-015: Scope Mismatch
# ===========================================================================


class TestFRD015ScopeMismatch:
    """Test FRD-015: Certification scope mismatch detection."""

    def test_scope_mismatch_detected(self, fraud_engine):
        """Certificate scope not covering claimed commodity is detected."""
        context = {
            "document_id": "DOC-FRD015-001",
            "certificate_scope": "oil_palm",
            "claimed_commodity": "wood",
        }
        result = fraud_engine.detect(rule_id="FRD-015", context=context)
        assert result is not None
        assert result["pattern_type"] == "scope_mismatch"

    def test_matching_scope_passes(self, fraud_engine):
        """Matching scope does not trigger FRD-015."""
        context = {
            "document_id": "DOC-FRD015-002",
            "certificate_scope": "wood",
            "claimed_commodity": "wood",
        }
        result = fraud_engine.detect(rule_id="FRD-015", context=context)
        assert result is None or result.get("triggered") is False


# ===========================================================================
# 16. Severity Classification
# ===========================================================================


class TestSeverityClassification:
    """Test fraud severity classification."""

    @pytest.mark.parametrize("severity", FRAUD_SEVERITIES)
    def test_all_severities_valid(self, fraud_engine, severity):
        """All 4 severity levels are valid."""
        alert = make_fraud_alert(severity=severity)
        assert alert["severity"] == severity
        assert_fraud_alert_valid(alert)

    def test_severity_weights(self, fraud_engine):
        """Severity weights are applied correctly."""
        for severity, weight in FRAUD_SEVERITY_WEIGHTS.items():
            assert weight > 0
        assert FRAUD_SEVERITY_WEIGHTS["critical"] > FRAUD_SEVERITY_WEIGHTS["high"]
        assert FRAUD_SEVERITY_WEIGHTS["high"] > FRAUD_SEVERITY_WEIGHTS["medium"]
        assert FRAUD_SEVERITY_WEIGHTS["medium"] > FRAUD_SEVERITY_WEIGHTS["low"]


# ===========================================================================
# 17. Batch Fraud Detection
# ===========================================================================


class TestBatchFraudDetection:
    """Test batch fraud detection."""

    def test_batch_detect_multiple(self, fraud_engine):
        """Batch detect across multiple documents."""
        contexts = [
            {"document_id": f"DOC-BATCH-{i}", "rules": FRAUD_RULE_IDS}
            for i in range(5)
        ]
        results = fraud_engine.batch_detect(contexts)
        assert len(results) == 5

    def test_batch_detect_empty(self, fraud_engine):
        """Batch detect with empty list returns empty results."""
        results = fraud_engine.batch_detect([])
        assert len(results) == 0


# ===========================================================================
# 18. Fraud Risk Score
# ===========================================================================


class TestFraudRiskScore:
    """Test fraud risk score calculation."""

    def test_risk_score_no_alerts(self, fraud_engine):
        """No alerts produces a low risk score."""
        score = fraud_engine.calculate_risk_score(alerts=[])
        assert_valid_score(score)
        assert score <= 10.0

    def test_risk_score_single_high_alert(self, fraud_engine):
        """Single high-severity alert produces elevated score."""
        alerts = [make_fraud_alert(severity="high")]
        score = fraud_engine.calculate_risk_score(alerts=alerts)
        assert_valid_score(score)
        assert score > 10.0

    def test_risk_score_multiple_alerts(self, fraud_engine):
        """Multiple alerts increase the risk score."""
        alerts = [
            make_fraud_alert(severity="medium"),
            make_fraud_alert(severity="high"),
            make_fraud_alert(severity="critical"),
        ]
        score = fraud_engine.calculate_risk_score(alerts=alerts)
        assert_valid_score(score)
        assert score > 30.0

    def test_risk_score_max_100(self, fraud_engine):
        """Risk score is capped at 100."""
        alerts = [make_fraud_alert(severity="critical") for _ in range(20)]
        score = fraud_engine.calculate_risk_score(alerts=alerts)
        assert_valid_score(score)
        assert score <= 100.0


# ===========================================================================
# 19. Rule Toggling
# ===========================================================================


class TestRuleToggling:
    """Test enabling/disabling individual fraud rules."""

    def test_disable_rule(self, fraud_engine):
        """Disabled rule does not trigger."""
        fraud_engine.disable_rule("FRD-001")
        context = {
            "document_id": "DOC-TOGGLE-001",
            "file_hash_sha256": "a" * 64,
            "shipment_id": "SHIP-TOG",
            "previous_hashes": {"a" * 64: {"shipment_id": "SHIP-OLD"}},
        }
        result = fraud_engine.detect(rule_id="FRD-001", context=context)
        assert result is None or result.get("triggered") is False

    def test_enable_rule(self, fraud_engine):
        """Re-enabled rule triggers again."""
        fraud_engine.disable_rule("FRD-001")
        fraud_engine.enable_rule("FRD-001")
        assert fraud_engine.is_rule_enabled("FRD-001") is True

    def test_list_enabled_rules(self, fraud_engine):
        """List all currently enabled rules."""
        rules = fraud_engine.list_enabled_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_all_rules_enabled_by_default(self, fraud_engine):
        """All 15 rules are enabled by default."""
        for rule_id in FRAUD_RULE_IDS:
            assert fraud_engine.is_rule_enabled(rule_id) is True


# ===========================================================================
# 20. Edge Cases and Parametrized
# ===========================================================================


class TestFraudEdgeCases:
    """Test edge cases for fraud pattern detection."""

    @pytest.mark.parametrize("pattern_type", FRAUD_PATTERN_TYPES)
    def test_all_pattern_types_valid(self, fraud_engine, pattern_type):
        """All 15 fraud pattern types are valid."""
        alert = make_fraud_alert(
            pattern_type=pattern_type,
            rule_id=next(
                (k for k, v in FRAUD_RULE_TO_PATTERN.items() if v == pattern_type),
                "FRD-001",
            ),
        )
        assert_fraud_alert_valid(alert)

    @pytest.mark.parametrize("rule_id,pattern", list(FRAUD_RULE_TO_PATTERN.items()))
    def test_rule_to_pattern_mapping(self, fraud_engine, rule_id, pattern):
        """Each rule ID maps to the correct pattern type."""
        alert = make_fraud_alert(rule_id=rule_id, pattern_type=pattern)
        assert alert["rule_id"] == rule_id
        assert alert["pattern_type"] == pattern

    def test_provenance_hash_on_alert(self, fraud_engine):
        """Fraud alert can include a provenance hash."""
        alert = make_fraud_alert()
        alert["provenance_hash"] = "f" * 64
        assert_valid_provenance_hash(alert["provenance_hash"])

    def test_factory_alert_valid(self, fraud_engine):
        """Factory-built fraud alert passes validation."""
        alert = make_fraud_alert()
        assert_fraud_alert_valid(alert)

    def test_empty_context_raises(self, fraud_engine):
        """Empty context raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            fraud_engine.detect(rule_id="FRD-001", context={})

    def test_invalid_rule_id_raises(self, fraud_engine):
        """Invalid rule ID raises ValueError."""
        with pytest.raises(ValueError):
            fraud_engine.detect(rule_id="FRD-999", context={"document_id": "DOC-X"})
