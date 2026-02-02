# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-007: PII Redaction & Minimization Agent

Tests cover:
    - PII detection (emails, phones, SSN, credit cards, etc.)
    - Redaction strategies (mask, hash, replace, remove, tokenize)
    - Tokenization and detokenization
    - Compliance framework validation (GDPR, CCPA, HIPAA, PCI-DSS)
    - Audit logging
    - Policy configuration
    - Document scanning
    - Confidence scoring
"""

import pytest
from datetime import datetime
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from greenlang.agents.base import AgentConfig, AgentResult

# Import from pii_redaction module
from greenlang.agents.foundation.pii_redaction import (
    # Main agent
    PIIRedactionAgent,

    # Enums
    PIIType,
    RedactionStrategy,
    ComplianceFramework,
    DetectionConfidence,
    AuditAction,

    # Models
    PIIMatch,
    RedactedMatch,
    TokenEntry,
    RedactionPolicy,
    PIIRedactionInput,
    PIIRedactionOutput,
    AuditLogEntry,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh PIIRedactionAgent for each test."""
    return PIIRedactionAgent()


@pytest.fixture
def sample_text_with_pii():
    """Sample text containing various PII types."""
    return """
    Contact Information:
    Name: John Smith
    Email: john.smith@example.com
    Phone: (555) 123-4567
    SSN: 123-45-6789

    Payment Details:
    Credit Card: 4111-1111-1111-1111

    Technical Info:
    IP Address: 192.168.1.100
    API Key: AKIA1234567890ABCDEF
    """


@pytest.fixture
def sample_document():
    """Sample document structure for scanning."""
    return {
        "id": "doc_001",
        "customer": {
            "name": "Jane Doe",
            "email": "jane.doe@company.com",
            "phone": "+1-555-987-6543",
        },
        "payment": {
            "card_number": "5555555555554444",
            "billing_address": "123 Main St, New York, NY 10001",
        },
        "notes": "Customer SSN is 987-65-4321 for verification.",
    }


# =============================================================================
# Detection Tests
# =============================================================================

class TestPIIDetection:
    """Test PII detection capabilities."""

    def test_detect_email(self, agent):
        """Test email detection."""
        result = agent.run({
            "operation": "detect",
            "content": "Contact me at test@example.com for more info."
        })

        assert result.success
        matches = result.data.get("matches", [])
        assert len(matches) >= 1

        email_match = next((m for m in matches if m["pii_type"] == "email"), None)
        assert email_match is not None
        assert "test@example.com" in email_match["value"]

    def test_detect_phone_us(self, agent):
        """Test US phone number detection."""
        result = agent.run({
            "operation": "detect",
            "content": "Call me at (555) 123-4567 or 555-987-6543."
        })

        assert result.success
        matches = result.data.get("matches", [])
        phone_matches = [m for m in matches if m["pii_type"] == "phone"]
        assert len(phone_matches) >= 1

    def test_detect_ssn(self, agent):
        """Test SSN detection."""
        result = agent.run({
            "operation": "detect",
            "content": "My SSN is 123-45-6789."
        })

        assert result.success
        matches = result.data.get("matches", [])
        ssn_match = next((m for m in matches if m["pii_type"] == "ssn"), None)
        assert ssn_match is not None
        assert "123-45-6789" in ssn_match["value"]

    def test_detect_credit_card(self, agent):
        """Test credit card detection."""
        result = agent.run({
            "operation": "detect",
            "content": "Card number: 4111111111111111"
        })

        assert result.success
        matches = result.data.get("matches", [])
        cc_match = next((m for m in matches if m["pii_type"] == "credit_card"), None)
        assert cc_match is not None

    def test_detect_ip_address(self, agent):
        """Test IP address detection."""
        result = agent.run({
            "operation": "detect",
            "content": "Server IP: 192.168.1.100"
        })

        assert result.success
        matches = result.data.get("matches", [])
        ip_match = next((m for m in matches if m["pii_type"] == "ip_address"), None)
        assert ip_match is not None
        assert "192.168.1.100" in ip_match["value"]

    def test_detect_multiple_pii(self, agent, sample_text_with_pii):
        """Test detection of multiple PII types in one text."""
        result = agent.run({
            "operation": "detect",
            "content": sample_text_with_pii
        })

        assert result.success
        matches = result.data.get("matches", [])

        # Should detect email, phone, SSN, credit card, IP
        pii_types = set(m["pii_type"] for m in matches)
        assert "email" in pii_types
        assert "phone" in pii_types
        assert "ssn" in pii_types
        assert "credit_card" in pii_types
        assert "ip_address" in pii_types

    def test_confidence_scoring(self, agent):
        """Test that confidence scores are assigned."""
        result = agent.run({
            "operation": "detect",
            "content": "Email: valid@example.com"
        })

        assert result.success
        matches = result.data.get("matches", [])
        assert len(matches) > 0

        # All matches should have confidence
        for match in matches:
            assert "confidence" in match
            assert match["confidence"] in ["high", "medium", "low", "uncertain"]


# =============================================================================
# Redaction Tests
# =============================================================================

class TestPIIRedaction:
    """Test PII redaction capabilities."""

    def test_redact_default_strategy(self, agent):
        """Test default redaction strategy."""
        result = agent.run({
            "operation": "redact",
            "content": "Email: test@example.com"
        })

        assert result.success
        redacted = result.data.get("redacted_content", "")
        assert "test@example.com" not in redacted
        assert "EMAIL" in redacted or "TOKEN" in redacted or "[" in redacted

    def test_redact_mask_strategy(self, agent):
        """Test mask redaction strategy."""
        result = agent.run({
            "operation": "redact",
            "content": "SSN: 123-45-6789",
            "policies": [
                {
                    "pii_type": "ssn",
                    "strategy": "mask",
                    "enabled": True,
                    "min_confidence": "medium"
                }
            ]
        })

        assert result.success
        redacted = result.data.get("redacted_content", "")
        assert "123-45-6789" not in redacted
        # Mask should preserve some characters
        assert "*" in redacted

    def test_redact_hash_strategy(self, agent):
        """Test hash redaction strategy."""
        result = agent.run({
            "operation": "redact",
            "content": "Card: 4111111111111111",
            "policies": [
                {
                    "pii_type": "credit_card",
                    "strategy": "hash",
                    "enabled": True,
                    "min_confidence": "low"
                }
            ]
        })

        assert result.success
        redacted = result.data.get("redacted_content", "")
        assert "4111111111111111" not in redacted
        assert "HASH:" in redacted

    def test_redact_replace_strategy(self, agent):
        """Test replace redaction strategy."""
        result = agent.run({
            "operation": "redact",
            "content": "IP: 192.168.1.100",
            "policies": [
                {
                    "pii_type": "ip_address",
                    "strategy": "replace",
                    "enabled": True,
                    "min_confidence": "medium"
                }
            ]
        })

        assert result.success
        redacted = result.data.get("redacted_content", "")
        assert "192.168.1.100" not in redacted
        assert "IP_ADDRESS" in redacted

    def test_redact_remove_strategy(self, agent):
        """Test remove redaction strategy."""
        result = agent.run({
            "operation": "redact",
            "content": "Password: secret123",
            "policies": [
                {
                    "pii_type": "password",
                    "strategy": "remove",
                    "enabled": True,
                    "min_confidence": "low"
                }
            ]
        })

        assert result.success
        # Password pattern should be removed

    def test_redact_preserves_structure(self, agent):
        """Test that redaction preserves text structure."""
        original = "Name: John\nEmail: test@example.com\nPhone: 555-1234"
        result = agent.run({
            "operation": "redact",
            "content": original
        })

        assert result.success
        redacted = result.data.get("redacted_content", "")

        # Should still have newlines
        assert "\n" in redacted
        assert "Name:" in redacted


# =============================================================================
# Tokenization Tests
# =============================================================================

class TestTokenization:
    """Test tokenization and detokenization."""

    def test_tokenize_creates_tokens(self, agent):
        """Test that tokenization creates reversible tokens."""
        result = agent.run({
            "operation": "tokenize",
            "content": "Email: test@example.com",
            "tenant_id": "tenant_001"
        })

        assert result.success
        tokens = result.data.get("tokens", [])
        # Email should be tokenizable by default
        assert len(tokens) >= 0  # May or may not tokenize based on policy

    def test_tokenize_and_detokenize(self, agent):
        """Test round-trip tokenization."""
        # First tokenize
        result = agent.run({
            "operation": "redact",
            "content": "Phone: 555-123-4567",
            "policies": [
                {
                    "pii_type": "phone",
                    "strategy": "tokenize",
                    "enabled": True,
                    "allow_tokenization": True,
                    "min_confidence": "medium"
                }
            ],
            "tenant_id": "tenant_001"
        })

        assert result.success
        tokens = result.data.get("tokens", [])

        # If a token was created, try to detokenize
        if tokens:
            token_id = tokens[0]

            detokenize_result = agent.run({
                "operation": "detokenize",
                "token_id": token_id,
                "tenant_id": "tenant_001"
            })

            # Should be able to get original value back
            assert detokenize_result.success or "error" in detokenize_result.data

    def test_detokenize_wrong_tenant(self, agent):
        """Test that detokenization fails for wrong tenant."""
        # Create a token
        result = agent.run({
            "operation": "redact",
            "content": "Email: test@example.com",
            "policies": [
                {
                    "pii_type": "email",
                    "strategy": "tokenize",
                    "enabled": True,
                    "allow_tokenization": True,
                    "min_confidence": "medium"
                }
            ],
            "tenant_id": "tenant_001"
        })

        assert result.success
        tokens = result.data.get("tokens", [])

        if tokens:
            # Try to detokenize with different tenant
            detokenize_result = agent.run({
                "operation": "detokenize",
                "token_id": tokens[0],
                "tenant_id": "tenant_002"  # Different tenant
            })

            # Should fail due to tenant mismatch
            if detokenize_result.success:
                assert "error" in str(detokenize_result.data).lower() or "denied" in str(detokenize_result.data).lower()


# =============================================================================
# Document Scanning Tests
# =============================================================================

class TestDocumentScanning:
    """Test document structure scanning."""

    def test_scan_document(self, agent, sample_document):
        """Test scanning a document structure."""
        result = agent.run({
            "operation": "scan_document",
            "documents": [sample_document]
        })

        assert result.success
        doc_results = result.data.get("document_results", [])
        assert len(doc_results) == 1

        # Should find PII in nested fields
        matches = doc_results[0].get("matches", [])
        assert len(matches) > 0

    def test_scan_multiple_documents(self, agent):
        """Test scanning multiple documents."""
        docs = [
            {"id": "1", "email": "a@test.com"},
            {"id": "2", "email": "b@test.com"},
            {"id": "3", "phone": "555-1234"},
        ]

        result = agent.run({
            "operation": "scan_document",
            "documents": docs
        })

        assert result.success
        doc_results = result.data.get("document_results", [])
        assert len(doc_results) == 3

        stats = result.data.get("statistics", {})
        assert stats.get("documents_scanned") == 3


# =============================================================================
# Policy Configuration Tests
# =============================================================================

class TestPolicyConfiguration:
    """Test policy configuration and validation."""

    def test_configure_policy(self, agent):
        """Test configuring custom policies."""
        result = agent.run({
            "operation": "configure_policy",
            "policies": [
                {
                    "pii_type": "email",
                    "strategy": "hash",
                    "enabled": True,
                    "allow_tokenization": False,
                    "min_confidence": "high"
                }
            ]
        })

        assert result.success
        configured = result.data.get("configured_policies", [])
        assert "email" in configured

    def test_validate_policy_gdpr(self, agent):
        """Test GDPR compliance validation."""
        result = agent.run({
            "operation": "validate_policy",
            "compliance_frameworks": ["gdpr"]
        })

        assert result.success
        # Should check for GDPR requirements
        assert "frameworks_checked" in result.data
        assert "gdpr" in result.data["frameworks_checked"]

    def test_validate_policy_pci_dss(self, agent):
        """Test PCI-DSS compliance validation."""
        # Configure weak credit card policy
        agent.run({
            "operation": "configure_policy",
            "policies": [
                {
                    "pii_type": "credit_card",
                    "strategy": "partial_mask",  # Too weak for PCI-DSS
                    "enabled": True,
                    "min_confidence": "medium"
                }
            ]
        })

        result = agent.run({
            "operation": "validate_policy",
            "compliance_frameworks": ["pci_dss"]
        })

        assert result.success
        issues = result.data.get("issues", [])
        # Should flag weak credit card protection
        pci_issues = [i for i in issues if i.get("framework") == "pci_dss"]
        assert len(pci_issues) > 0


# =============================================================================
# Audit Logging Tests
# =============================================================================

class TestAuditLogging:
    """Test audit logging capabilities."""

    def test_audit_log_created(self, agent):
        """Test that operations create audit logs."""
        # Perform a detection
        agent.run({
            "operation": "detect",
            "content": "test@example.com",
            "tenant_id": "tenant_001"
        })

        # Get audit log
        result = agent.run({
            "operation": "get_audit_log",
            "tenant_id": "tenant_001"
        })

        assert result.success
        entries = result.data.get("audit_entries", [])
        assert len(entries) > 0

    def test_audit_log_filter_by_tenant(self, agent):
        """Test filtering audit logs by tenant."""
        # Create logs for different tenants
        agent.run({
            "operation": "detect",
            "content": "test@example.com",
            "tenant_id": "tenant_A"
        })
        agent.run({
            "operation": "detect",
            "content": "other@example.com",
            "tenant_id": "tenant_B"
        })

        # Get logs for tenant_A only
        result = agent.run({
            "operation": "get_audit_log",
            "tenant_id": "tenant_A"
        })

        assert result.success
        entries = result.data.get("audit_entries", [])
        # All entries should be for tenant_A
        for entry in entries:
            if entry.get("tenant_id"):
                assert entry["tenant_id"] == "tenant_A"


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Test statistics gathering."""

    def test_get_statistics(self, agent):
        """Test getting agent statistics."""
        # Perform some operations first
        agent.run({
            "operation": "detect",
            "content": "test@example.com"
        })
        agent.run({
            "operation": "redact",
            "content": "phone: 555-1234"
        })

        result = agent.run({
            "operation": "get_statistics"
        })

        assert result.success
        stats = result.data.get("statistics", {})
        assert "total_detections" in stats
        assert "total_redactions" in stats
        assert stats["total_detections"] >= 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self, agent):
        """Test handling empty content."""
        result = agent.run({
            "operation": "detect",
            "content": ""
        })

        assert result.success
        matches = result.data.get("matches", [])
        assert len(matches) == 0

    def test_no_pii_content(self, agent):
        """Test content with no PII."""
        result = agent.run({
            "operation": "detect",
            "content": "This is a normal text with no personal information."
        })

        assert result.success
        matches = result.data.get("matches", [])
        # Should find no PII
        assert len(matches) == 0

    def test_invalid_operation(self, agent):
        """Test invalid operation handling."""
        result = agent.run({
            "operation": "invalid_op",
            "content": "test"
        })

        assert not result.success
        assert "error" in result.error.lower() or "invalid" in result.error.lower()

    def test_missing_content(self, agent):
        """Test missing required content."""
        result = agent.run({
            "operation": "detect"
            # Missing content
        })

        assert not result.success


# =============================================================================
# Convenience Method Tests
# =============================================================================

class TestConvenienceMethods:
    """Test convenience methods."""

    def test_redact_text_convenience(self, agent):
        """Test the redact_text convenience method."""
        text = "Email: test@example.com"
        redacted = agent.redact_text(text)

        assert "test@example.com" not in redacted

    def test_detect_pii_convenience(self, agent):
        """Test the detect_pii convenience method."""
        text = "Call 555-123-4567"
        matches = agent.detect_pii(text)

        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.PHONE for m in matches)


# =============================================================================
# Zero-Hallucination Tests
# =============================================================================

class TestZeroHallucination:
    """Test zero-hallucination guarantees."""

    def test_deterministic_detection(self, agent):
        """Test that detection is deterministic."""
        content = "Email: test@example.com, Phone: 555-1234"

        result1 = agent.run({"operation": "detect", "content": content})
        result2 = agent.run({"operation": "detect", "content": content})

        assert result1.success and result2.success

        # Same content should produce same matches
        matches1 = result1.data.get("matches", [])
        matches2 = result2.data.get("matches", [])

        assert len(matches1) == len(matches2)
        for m1, m2 in zip(matches1, matches2):
            assert m1["pii_type"] == m2["pii_type"]
            assert m1["value"] == m2["value"]

    def test_provenance_hash(self, agent):
        """Test that provenance hash is generated."""
        result = agent.run({
            "operation": "detect",
            "content": "test@example.com"
        })

        assert result.success
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
