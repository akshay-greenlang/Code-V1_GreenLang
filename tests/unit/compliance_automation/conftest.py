"""
Test fixtures for compliance_automation module.

Provides mock compliance data, frameworks, and configuration for testing.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


@pytest.fixture
def iso27001_control() -> Dict[str, Any]:
    """Create a sample ISO 27001 control."""
    return {
        "control_id": "A.5.1.1",
        "name": "Information Security Policy",
        "description": "A set of policies shall be defined, approved by management, published and communicated.",
        "category": "Information Security Policies",
        "status": "implemented",
        "evidence": ["policy_document.pdf", "approval_record.pdf"],
        "last_review": datetime.utcnow() - timedelta(days=30),
        "next_review": datetime.utcnow() + timedelta(days=335),
    }


@pytest.fixture
def gdpr_dsar_request() -> Dict[str, Any]:
    """Create a sample GDPR DSAR request."""
    return {
        "request_id": str(uuid4()),
        "request_type": "access",
        "data_subject_email": "user@example.com",
        "submitted_at": datetime.utcnow().isoformat(),
        "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "status": "in_progress",
        "data_categories": ["profile", "transactions", "communications"],
    }


@pytest.fixture
def pci_dss_requirement() -> Dict[str, Any]:
    """Create a sample PCI-DSS requirement."""
    return {
        "requirement_id": "3.4",
        "name": "Render PAN unreadable anywhere it is stored",
        "description": "Primary account numbers must be rendered unreadable using encryption, hashing, etc.",
        "status": "compliant",
        "evidence": ["encryption_config.yaml", "key_management_policy.pdf"],
        "validation_method": "automated_scan",
        "last_validated": datetime.utcnow() - timedelta(days=7),
    }


@pytest.fixture
def ccpa_consumer_request() -> Dict[str, Any]:
    """Create a sample CCPA consumer request."""
    return {
        "request_id": str(uuid4()),
        "request_type": "delete",
        "consumer_email": "consumer@example.com",
        "submitted_at": datetime.utcnow().isoformat(),
        "deadline": (datetime.utcnow() + timedelta(days=45)).isoformat(),
        "status": "pending",
        "verification_status": "verified",
    }


@pytest.fixture
def compliance_config() -> Dict[str, Any]:
    """Create compliance automation configuration."""
    return {
        "frameworks": ["iso27001", "gdpr", "pci_dss", "ccpa"],
        "auto_evidence_collection": True,
        "notification_channels": ["slack", "email"],
        "review_reminder_days": 30,
        "dsar_deadline_days": 30,
        "ccpa_deadline_days": 45,
    }


@pytest.fixture
def compliance_admin_headers() -> Dict[str, str]:
    """Create headers for compliance admin role."""
    return {
        "Authorization": "Bearer test-compliance-admin-token",
        "X-User-Id": "compliance-admin",
        "X-User-Roles": "compliance-admin,security-analyst",
    }
