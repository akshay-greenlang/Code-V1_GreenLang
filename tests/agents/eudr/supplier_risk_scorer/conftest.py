# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-017 Supplier Risk Scorer test suite.

Provides reusable fixtures for configuration, engine instances, sample data,
and shared test utilities used across all test modules.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.supplier_risk_scorer.config import (
    SupplierRiskScorerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.supplier_risk_scorer.provenance import (
    reset_tracker as reset_provenance_tracker,
)
from greenlang.agents.eudr.supplier_risk_scorer.supplier_risk_scorer import (
    SupplierRiskScorer,
)
from greenlang.agents.eudr.supplier_risk_scorer.due_diligence_tracker import (
    DueDiligenceTracker,
)
from greenlang.agents.eudr.supplier_risk_scorer.documentation_analyzer import (
    DocumentationAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.certification_validator import (
    CertificationValidator,
)
from greenlang.agents.eudr.supplier_risk_scorer.geographic_sourcing_analyzer import (
    GeographicSourcingAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.network_analyzer import (
    NetworkAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.monitoring_alert_engine import (
    MonitoringAlertEngine,
)
from greenlang.agents.eudr.supplier_risk_scorer.risk_reporting_engine import (
    RiskReportingEngine,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    RiskLevel,
    SupplierType,
    CommodityType,
    CertificationScheme,
    CertificationStatus,
    DocumentType,
    DocumentStatus,
    DDLevel,
    DDStatus,
    NonConformanceType,
    AlertSeverity,
    AlertType,
    ReportType,
    ReportFormat,
    MonitoringFrequency,
    SUPPORTED_COMMODITIES,
    SUPPORTED_SCHEMES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
    DEFAULT_FACTOR_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a SupplierRiskScorerConfig with test defaults."""
    cfg = SupplierRiskScorerConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        pool_size=2,
        # Risk scoring weights (8 factors, sum to 100)
        geographic_sourcing_weight=20,
        compliance_history_weight=15,
        documentation_quality_weight=15,
        certification_status_weight=15,
        traceability_completeness_weight=10,
        financial_stability_weight=10,
        environmental_performance_weight=10,
        social_compliance_weight=5,
        # Risk thresholds
        low_risk_threshold=25,
        medium_risk_threshold=50,
        high_risk_threshold=75,
        critical_risk_threshold=90,
        confidence_threshold=0.6,
        score_normalization="minmax",
        trend_window_months=12,
        aggregation_method="weighted_average",
        # Due diligence
        dd_tracking_period_months=12,
        minor_nc_threshold=5,
        major_nc_threshold=2,
        critical_nc_threshold=1,
        audit_interval_months=12,
        corrective_action_deadline_days=90,
        dd_overdue_limit_days=30,
        # Documentation
        required_documents=[
            "geolocation", "dds_reference", "product_description",
            "quantity_declaration", "harvest_date", "compliance_declaration"
        ],
        completeness_threshold=0.8,
        expiry_warning_days=90,
        quality_scoring_enabled=True,
        gap_detection_enabled=True,
        # Certification
        supported_cert_schemes=[
            "FSC", "PEFC", "RSPO", "RAINFOREST_ALLIANCE",
            "UTZ", "ORGANIC", "FAIR_TRADE", "ISCC"
        ],
        cert_expiry_buffer_days=90,
        chain_of_custody_required=True,
        multi_site_enabled=True,
        # Geographic sourcing
        concentration_threshold=0.25,
        proximity_buffer_km=10.0,
        high_risk_zone_enabled=True,
        deforestation_overlay_enabled=True,
        protected_area_detection=True,
        indigenous_territory_overlap=True,
        # Network analysis
        network_max_depth=3,
        risk_propagation_decay=0.80,
        sub_supplier_evaluation=True,
        intermediary_tracking=True,
        circular_dependency_detection=True,
        # Monitoring
        monitoring_default_frequency="monthly",
        alert_info_threshold=25,
        alert_warning_threshold=50,
        alert_high_threshold=75,
        alert_critical_threshold=90,
        watchlist_max_size=1000,
        behavior_change_detection=True,
        # Reporting
        default_language="en",
        report_retention_days=1825,
        template_dir="/tmp/templates",
        max_report_size_mb=50,
        dds_package_generation=True,
        audit_package_assembly=True,
        # Redis
        redis_ttl_s=3600,
        redis_key_prefix="srs_test:",
        # Batch
        batch_max_size=500,
        batch_concurrency=4,
        batch_timeout_s=300,
        retention_years=5,
        # Provenance
        enable_provenance=True,
        genesis_hash="GL-EUDR-SRS-017-TEST-GENESIS",
        chain_algorithm="sha256",
        # Metrics
        enable_metrics=False,
        metrics_prefix="gl_eudr_srs_",
    )
    set_config(cfg)
    yield cfg
    reset_config()


@pytest.fixture(autouse=True)
def reset_state():
    """Reset config and provenance between tests."""
    yield
    reset_config()
    reset_provenance_tracker()


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def supplier_risk_scorer(mock_config):
    """Create a SupplierRiskScorer instance for testing."""
    return SupplierRiskScorer()


@pytest.fixture
def due_diligence_tracker(mock_config):
    """Create a DueDiligenceTracker instance for testing."""
    return DueDiligenceTracker()


@pytest.fixture
def documentation_analyzer(mock_config):
    """Create a DocumentationAnalyzer instance for testing."""
    return DocumentationAnalyzer()


@pytest.fixture
def certification_validator(mock_config):
    """Create a CertificationValidator instance for testing."""
    return CertificationValidator()


@pytest.fixture
def geographic_sourcing_analyzer(mock_config):
    """Create a GeographicSourcingAnalyzer instance for testing."""
    return GeographicSourcingAnalyzer()


@pytest.fixture
def network_analyzer(mock_config):
    """Create a NetworkAnalyzer instance for testing."""
    return NetworkAnalyzer()


@pytest.fixture
def monitoring_alert_engine(mock_config):
    """Create a MonitoringAlertEngine instance for testing."""
    return MonitoringAlertEngine()


@pytest.fixture
def risk_reporting_engine(mock_config):
    """Create a RiskReportingEngine instance for testing."""
    return RiskReportingEngine()


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_supplier():
    """Sample supplier profile data."""
    return {
        "supplier_id": "SUPP-001",
        "name": "Green Forest Suppliers Ltd",
        "type": SupplierType.PRODUCER,
        "country": "BR",
        "commodities": [CommodityType.SOYA, CommodityType.CATTLE],
        "registration_date": utcnow() - timedelta(days=365),
        "tax_id": "BR123456789",
        "address": "Rua Verde 123, Mato Grosso, Brazil",
        "contact_email": "contact@greenforest.br",
        "active": True,
    }


@pytest.fixture
def sample_factor_scores():
    """Sample 8-factor risk scores for composite assessment."""
    return {
        "geographic_sourcing": Decimal("65.0"),
        "compliance_history": Decimal("40.0"),
        "documentation_quality": Decimal("75.0"),
        "certification_status": Decimal("50.0"),
        "traceability_completeness": Decimal("60.0"),
        "financial_stability": Decimal("70.0"),
        "environmental_performance": Decimal("55.0"),
        "social_compliance": Decimal("80.0"),
    }


@pytest.fixture
def sample_dd_record():
    """Sample due diligence record."""
    return {
        "dd_id": "dd-test-001",
        "supplier_id": "SUPP-001",
        "dd_level": DDLevel.STANDARD,
        "status": DDStatus.IN_PROGRESS,
        "activities": ["document_review", "screening"],
        "non_conformances": [],
        "corrective_actions": [],
        "start_date": utcnow() - timedelta(days=30),
        "auditor": "auditor@example.com",
    }


@pytest.fixture
def sample_documentation():
    """Sample documentation profile."""
    return {
        "profile_id": "doc-test-001",
        "supplier_id": "SUPP-001",
        "documents": [
            {
                "type": DocumentType.GEOLOCATION,
                "status": DocumentStatus.VERIFIED,
                "submitted_date": (utcnow() - timedelta(days=10)).isoformat(),
            },
            {
                "type": DocumentType.DDS_REFERENCE,
                "status": DocumentStatus.VERIFIED,
                "submitted_date": (utcnow() - timedelta(days=5)).isoformat(),
            },
        ],
        "completeness_score": Decimal("0.75"),
        "quality_score": Decimal("80.0"),
        "gaps": ["harvest_date", "compliance_declaration"],
        "expiring_soon": [],
    }


@pytest.fixture
def sample_certification():
    """Sample certification record."""
    return {
        "cert_id": "cert-test-001",
        "supplier_id": "SUPP-001",
        "scheme": CertificationScheme.FSC,
        "certificate_number": "FSC-C123456",
        "status": CertificationStatus.VALID,
        "issue_date": utcnow() - timedelta(days=365),
        "expiry_date": utcnow() + timedelta(days=365),
        "certification_body": "FSC International",
        "scope": ["wood"],
        "commodities": [CommodityType.WOOD],
    }


@pytest.fixture
def sample_sourcing():
    """Sample geographic sourcing data."""
    return {
        "supplier_id": "SUPP-001",
        "sourcing_locations": [
            {
                "country": "BR",
                "region": "Mato Grosso",
                "latitude": -15.5,
                "longitude": -56.0,
                "volume_percentage": 60.0,
            },
            {
                "country": "BR",
                "region": "Para",
                "latitude": -3.5,
                "longitude": -52.0,
                "volume_percentage": 40.0,
            },
        ],
        "commodity": CommodityType.SOYA,
        "total_volume": 10000.0,
    }


@pytest.fixture
def sample_network():
    """Sample supplier network data."""
    return {
        "supplier_id": "SUPP-001",
        "sub_suppliers": [
            {
                "supplier_id": "SUPP-SUB-001",
                "name": "Farm Cooperative A",
                "type": SupplierType.COOPERATIVE,
                "tier": 2,
                "risk_score": 45.0,
            },
            {
                "supplier_id": "SUPP-SUB-002",
                "name": "Farm Cooperative B",
                "type": SupplierType.COOPERATIVE,
                "tier": 2,
                "risk_score": 55.0,
            },
        ],
        "intermediaries": [],
        "max_depth": 2,
    }


@pytest.fixture
def sample_low_risk_factors():
    """Factor scores resulting in LOW risk classification."""
    return {
        "geographic_sourcing": Decimal("15.0"),
        "compliance_history": Decimal("10.0"),
        "documentation_quality": Decimal("20.0"),
        "certification_status": Decimal("12.0"),
        "traceability_completeness": Decimal("18.0"),
        "financial_stability": Decimal("15.0"),
        "environmental_performance": Decimal("10.0"),
        "social_compliance": Decimal("8.0"),
    }


@pytest.fixture
def sample_high_risk_factors():
    """Factor scores resulting in HIGH risk classification."""
    return {
        "geographic_sourcing": Decimal("85.0"),
        "compliance_history": Decimal("75.0"),
        "documentation_quality": Decimal("80.0"),
        "certification_status": Decimal("70.0"),
        "traceability_completeness": Decimal("65.0"),
        "financial_stability": Decimal("75.0"),
        "environmental_performance": Decimal("80.0"),
        "social_compliance": Decimal("70.0"),
    }


@pytest.fixture
def sample_critical_risk_factors():
    """Factor scores resulting in CRITICAL risk classification."""
    return {
        "geographic_sourcing": Decimal("95.0"),
        "compliance_history": Decimal("90.0"),
        "documentation_quality": Decimal("85.0"),
        "certification_status": Decimal("88.0"),
        "traceability_completeness": Decimal("80.0"),
        "financial_stability": Decimal("85.0"),
        "environmental_performance": Decimal("92.0"),
        "social_compliance": Decimal("87.0"),
    }


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_uuid():
    """Provide a deterministic UUID generator for tests."""
    return DeterministicUUID()
