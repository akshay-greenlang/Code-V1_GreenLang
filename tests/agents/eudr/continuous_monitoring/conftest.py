# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-033 Continuous Monitoring Agent tests.

Provides reusable test fixtures for config, provenance, models, sample
supply chain records, deforestation alerts, compliance audit data, change
detection events, risk score histories, data freshness records, regulatory
updates, and API request/response objects across all 13 test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.continuous_monitoring.config import (
    ContinuousMonitoringConfig,
    reset_config,
)
from greenlang.agents.eudr.continuous_monitoring.models import (
    AGENT_ID,
    AGENT_VERSION,
    AlertSeverity,
    AlertStatus,
    AuditAction,
    AuditEntry,
    ChangeType,
    ChangeImpact,
    ChangeDetectionRecord,
    ComplianceAuditRecord,
    ComplianceCheckItem,
    ComplianceStatus,
    DataFreshnessRecord,
    DeforestationCorrelation,
    DeforestationMonitorRecord,
    FreshnessStatus,
    HealthStatus,
    InvestigationRecord,
    InvestigationStatus,
    MonitoringAlert,
    MonitoringScope,
    MonitoringSummary,
    RegulatoryImpact,
    RegulatoryTrackingRecord,
    RegulatoryUpdate,
    RiskLevel,
    RiskScoreMonitorRecord,
    RiskScoreSnapshot,
    ScanStatus,
    StaleEntity,
    SupplyChainScanRecord,
    TrendDirection,
)
from greenlang.agents.eudr.continuous_monitoring.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> ContinuousMonitoringConfig:
    """Create a default ContinuousMonitoringConfig instance."""
    return ContinuousMonitoringConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Supply Chain Monitoring fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_suppliers() -> List[Dict[str, Any]]:
    """Create sample supplier data for supply chain monitoring tests."""
    now = datetime.now(timezone.utc)
    return [
        {
            "supplier_id": "SUP-001",
            "name": "PT Sawit Hijau",
            "status": "suspended",
            "previous_status": "active",
            "owner": "Green Palm Corp",
            "previous_owner": "PT Sawit Hijau",
            "certifications": [
                {
                    "certification_id": "CERT-RSPO-001",
                    "type": "RSPO",
                    "expiry_date": (now + timedelta(days=15)).isoformat(),
                },
            ],
            "plots": [
                {
                    "plot_id": "PLOT-001",
                    "original_lat": -2.5,
                    "original_lon": 112.9,
                    "current_lat": -2.5,
                    "current_lon": 112.9,
                },
            ],
        },
        {
            "supplier_id": "SUP-002",
            "name": "Malaysia Palm Co",
            "status": "active",
            "previous_status": "active",
            "certifications": [
                {
                    "certification_id": "CERT-RSPO-002",
                    "type": "RSPO",
                    "expiry_date": (now - timedelta(days=5)).isoformat(),
                },
            ],
            "plots": [],
        },
        {
            "supplier_id": "SUP-003",
            "name": "Colombia Coffee LLC",
            "status": "active",
            "previous_status": "active",
            "certifications": [],
            "plots": [
                {
                    "plot_id": "PLOT-002",
                    "original_lat": 4.6,
                    "original_lon": -74.1,
                    "current_lat": 4.61,
                    "current_lon": -74.1,
                },
            ],
        },
    ]


@pytest.fixture
def sample_supply_chain_events() -> List[Dict[str, Any]]:
    """Create sample supply chain events for backward compat with old tests."""
    return [
        {
            "supplier_id": "SUP-001",
            "name": "PT Sawit Hijau",
            "status": "suspended",
            "previous_status": "active",
            "certifications": [],
            "plots": [],
        },
        {
            "supplier_id": "SUP-002",
            "name": "Malaysia Palm Co",
            "status": "active",
            "previous_status": "active",
            "certifications": [],
            "plots": [],
        },
    ]


# ---------------------------------------------------------------------------
# Deforestation Alert fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_deforestation_alerts() -> List[Dict[str, Any]]:
    """Create sample deforestation alert data for testing."""
    return [
        {
            "alert_id": "DEF-001",
            "lat": -2.5,
            "lon": 112.9,
            "area_ha": 15.3,
        },
        {
            "alert_id": "DEF-002",
            "lat": 3.1,
            "lon": 101.7,
            "area_ha": 8.7,
        },
        {
            "alert_id": "DEF-003",
            "lat": -12.0,
            "lon": -55.0,
            "area_ha": 3.2,
        },
    ]


@pytest.fixture
def sample_supply_chain_entities() -> List[Dict[str, Any]]:
    """Create sample supply chain entities for deforestation correlation."""
    return [
        {"entity_id": "PLOT-001", "entity_type": "plot", "lat": -2.5, "lon": 112.9},
        {"entity_id": "PLOT-002", "entity_type": "plot", "lat": 3.1, "lon": 101.7},
        {"entity_id": "SUP-005", "entity_type": "supplier", "lat": -12.0, "lon": -55.0},
    ]


# ---------------------------------------------------------------------------
# Compliance Audit fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_compliance_checks() -> Dict[str, Any]:
    """Create sample compliance check operator data."""
    now = datetime.now(timezone.utc)
    return {
        "dds_date": (now - timedelta(days=30)).isoformat(),
        "supply_chain_last_updated": (now - timedelta(days=10)).isoformat(),
        "risk_assessments": [
            {
                "assessment_id": "RA-001",
                "assessment_date": (now - timedelta(days=60)).isoformat(),
                "scope": "supplier_risk",
            },
            {
                "assessment_id": "RA-002",
                "assessment_date": (now - timedelta(days=45)).isoformat(),
                "scope": "commodity_risk",
            },
        ],
        "due_diligence_statements": [
            {
                "statement_id": "DDS-001",
                "statement_date": (now - timedelta(days=20)).isoformat(),
                "commodity": "palm_oil",
                "origin_country": "ID",
                "supplier_info": "PT Sawit Hijau",
            },
        ],
        "retention_years": 5,
        "competent_authority_registered": True,
    }


@pytest.fixture
def sample_compliance_audit() -> ComplianceAuditRecord:
    """Create a sample compliance audit record."""
    return ComplianceAuditRecord(
        audit_id="CA-001",
        operator_id="OP-001",
        compliance_status=ComplianceStatus.PARTIALLY_COMPLIANT,
        overall_score=Decimal("78.00"),
        checks_passed=3,
        checks_failed=1,
        checks_total=5,
        check_items=[],
    )


# ---------------------------------------------------------------------------
# Change Detection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_change_events() -> List[Dict[str, Any]]:
    """Create sample entity snapshots for change detection."""
    return [
        {
            "entity_id": "SUP-001",
            "entity_type": "supplier",
            "old_state": {"owner": "PT Sawit Hijau", "status": "active"},
            "new_state": {"owner": "Green Palm Corp", "status": "active"},
        },
        {
            "entity_id": "PLOT-A1",
            "entity_type": "plot",
            "old_state": {"lat": "-2.500", "lon": "112.900"},
            "new_state": {"lat": "-2.510", "lon": "112.910"},
        },
        {
            "entity_id": "CERT-RSPO-001",
            "entity_type": "certification",
            "old_state": {"certification_status": "valid"},
            "new_state": {"certification_status": "suspended"},
        },
        {
            "entity_id": "RP-001",
            "entity_type": "risk_profile",
            "old_state": {"risk_score": "35"},
            "new_state": {"risk_score": "72"},
        },
        {
            "entity_id": "REG-EUDR-2023",
            "entity_type": "regulation",
            "old_state": {"regulatory_version": "v1.0"},
            "new_state": {"regulatory_version": "v1.1"},
        },
    ]


# ---------------------------------------------------------------------------
# Risk Score fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_risk_score_history() -> List[Dict[str, Any]]:
    """Create sample risk score history data."""
    now = datetime.now(timezone.utc)
    return [
        {
            "timestamp": (now - timedelta(days=30 - i)).isoformat(),
            "score": 40 + i * 3,
        }
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Data Freshness fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data_freshness_records() -> List[Dict[str, Any]]:
    """Create sample entity data for freshness validation."""
    now = datetime.now(timezone.utc)
    return [
        {
            "entity_id": "SUP-001",
            "entity_type": "supplier",
            "last_updated": (now - timedelta(hours=6)).isoformat(),
        },
        {
            "entity_id": "SUP-002",
            "entity_type": "supplier",
            "last_updated": (now - timedelta(days=5)).isoformat(),
        },
        {
            "entity_id": "CERT-001",
            "entity_type": "certification",
            "last_updated": (now - timedelta(hours=12)).isoformat(),
        },
        {
            "entity_id": "PLOT-001",
            "entity_type": "plot",
            "last_updated": (now - timedelta(days=15)).isoformat(),
        },
        {
            "entity_id": "PLOT-002",
            "entity_type": "plot",
            "last_updated": (now - timedelta(hours=2)).isoformat(),
        },
    ]


# ---------------------------------------------------------------------------
# Regulatory Tracker fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_regulatory_changes() -> List[Dict[str, Any]]:
    """Create sample regulatory update data."""
    now = datetime.now(timezone.utc)
    return [
        {
            "update_id": "REG-001",
            "source": "eur-lex",
            "title": "EUDR Implementation Guidance Update v2.1",
            "summary": "Updated guidance on geolocation precision requirements for Article 9",
            "published_date": (now - timedelta(days=5)).isoformat(),
            "impact_level": "high",
            "affected_articles": ["Article 8", "Article 10"],
        },
        {
            "update_id": "REG-002",
            "source": "eur-lex",
            "title": "EUDR Country Benchmarking List - First Publication",
            "summary": "EC published initial country risk benchmarking per Article 29",
            "published_date": (now - timedelta(days=2)).isoformat(),
            "impact_level": "breaking",
            "affected_articles": ["Article 29"],
        },
        {
            "update_id": "REG-003",
            "source": "national",
            "title": "Penalty framework clarification for EUDR enforcement",
            "summary": "Clarification of penalty calculation methodology",
            "published_date": (now - timedelta(days=15)).isoformat(),
            "impact_level": "moderate",
            "affected_articles": [],
        },
    ]


# ---------------------------------------------------------------------------
# Audit Entry fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit_entries() -> List[AuditEntry]:
    """Create sample audit log entries."""
    now = datetime.now(timezone.utc)
    return [
        AuditEntry(
            entry_id="AUD-001",
            entity_type="supply_chain_scan",
            entity_id="SCE-001",
            actor="AGENT-EUDR-033",
            action=AuditAction.SCAN,
            timestamp=now - timedelta(hours=2),
        ),
        AuditEntry(
            entry_id="AUD-002",
            entity_type="deforestation_alert",
            entity_id="DEF-001",
            actor="AGENT-EUDR-033",
            action=AuditAction.ALERT,
            timestamp=now - timedelta(hours=1),
        ),
        AuditEntry(
            entry_id="AUD-003",
            entity_type="compliance_audit",
            entity_id="CA-001",
            actor="user:admin@greenlang.io",
            action=AuditAction.AUDIT,
            timestamp=now,
        ),
    ]
