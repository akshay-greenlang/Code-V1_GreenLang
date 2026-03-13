# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-027 Information Gathering Agent test suite.

Provides reusable fixtures for configuration, provenance tracker, sample
query results, certificate verification results, supplier profiles,
Article 9 element statuses, and gathering operations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 Information Gathering Agent (GL-EUDR-IGA-027)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict

import pytest

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    reset_config,
)
from greenlang.agents.eudr.information_gathering.models import (
    Article9ElementName,
    Article9ElementStatus,
    CertificateVerificationResult,
    CertificationBody,
    CertVerificationStatus,
    CompletenessClassification,
    ElementStatus,
    EUDRCommodity,
    ExternalDatabaseSource,
    GatheringOperation,
    GatheringOperationStatus,
    QueryResult,
    QueryStatus,
    SupplierProfile,
)
from greenlang.agents.eudr.information_gathering.provenance import ProvenanceTracker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset config singleton before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config() -> InformationGatheringConfig:
    """Create InformationGatheringConfig with default test values."""
    return InformationGatheringConfig()


@pytest.fixture
def tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


@pytest.fixture
def sample_query_result() -> QueryResult:
    """Create a sample QueryResult fixture."""
    return QueryResult(
        query_id=f"qry_eu_traces_{uuid.uuid4().hex[:12]}",
        source=ExternalDatabaseSource.EU_TRACES,
        query_parameters={"certificate_number": "TRACES-2026-001"},
        status=QueryStatus.SUCCESS,
        records=[{"certificate_number": "TRACES-2026-001", "status": "validated"}],
        record_count=1,
        query_timestamp=_utcnow(),
        response_time_ms=42,
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sample_cert_result() -> CertificateVerificationResult:
    """Create a sample CertificateVerificationResult fixture."""
    return CertificateVerificationResult(
        certificate_id="FSC-C012345",
        certification_body=CertificationBody.FSC,
        holder_name="Sample Forest Products Ltd",
        verification_status=CertVerificationStatus.VALID,
        valid_from=_utcnow() - timedelta(days=365),
        valid_until=_utcnow() + timedelta(days=730),
        scope=["FM/CoC", "Controlled Wood"],
        commodity_scope=[EUDRCommodity.WOOD],
        chain_of_custody_model="Transfer",
        days_until_expiry=730,
        provenance_hash="b" * 64,
    )


@pytest.fixture
def sample_supplier_profile() -> SupplierProfile:
    """Create a sample SupplierProfile fixture."""
    return SupplierProfile(
        supplier_id="SUP-001",
        name="Green Coffee Exporters Ltd",
        alternative_names=["GCE Ltd", "Green Coffee"],
        postal_address="123 Export Road, Bogota, Colombia",
        country_code="CO",
        email="contact@greencoffee.co",
        registration_number="REG-CO-12345",
        commodities=[EUDRCommodity.COFFEE],
        plot_ids=["PLOT-CO-001", "PLOT-CO-002"],
        tier_depth=1,
        data_sources=["government_registry", "supplier_self_declared"],
        completeness_score=Decimal("85.00"),
        confidence_score=Decimal("72.00"),
        provenance_hash="c" * 64,
    )


@pytest.fixture
def sample_article9_elements() -> Dict[str, Article9ElementStatus]:
    """Create a dict of all 10 Article 9 element statuses (all complete)."""
    elements: Dict[str, Article9ElementStatus] = {}
    for elem in Article9ElementName:
        elements[elem.value] = Article9ElementStatus(
            element_name=elem.value,
            status=ElementStatus.COMPLETE,
            source="government_registry",
            value_summary=f"Complete data for {elem.value}",
            confidence=Decimal("0.95"),
            last_updated=_utcnow(),
        )
    return elements


@pytest.fixture
def sample_gathering_operation() -> GatheringOperation:
    """Create a sample GatheringOperation fixture."""
    return GatheringOperation(
        operation_id=f"op_{uuid.uuid4().hex[:12]}",
        operator_id="OP-DE-001",
        commodity=EUDRCommodity.COFFEE,
        status=GatheringOperationStatus.IN_PROGRESS,
        sources_queried=["eu_traces", "fao_stat"],
        sources_completed=["eu_traces"],
        completeness_score=Decimal("45.00"),
        total_records_collected=12,
        provenance_hash="d" * 64,
    )
