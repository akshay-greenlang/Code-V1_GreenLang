# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-037 Due Diligence Statement Creator tests.

Provides reusable test fixtures for config, models, provenance, engines,
geolocation data, risk references, supply chain data, compliance checks,
document packages, statement versions, and digital signatures.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from greenlang.agents.eudr.due_diligence_statement_creator.config import (
    DDSCreatorConfig,
    reset_config,
)
from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    AGENT_ID,
    AGENT_VERSION,
    ARTICLE_4_MANDATORY_FIELDS,
    EU_OFFICIAL_LANGUAGES,
    EUDR_REGULATED_COMMODITIES,
    AmendmentReason,
    AmendmentRecord,
    CommodityType,
    ComplianceCheck,
    ComplianceStatus,
    DDSStatement,
    DDSStatus,
    DDSValidationReport,
    DigitalSignature,
    DocumentPackage,
    DocumentType,
    GeolocationData,
    GeolocationMethod,
    HealthStatus,
    LanguageCode,
    LanguageTranslation,
    RiskLevel,
    RiskReference,
    SignatureType,
    StatementSummary,
    StatementType,
    StatementVersion,
    SubmissionPackage,
    SubmissionStatus,
    TemplateConfig,
    SupplyChainData,
    ValidationResult,
)
from greenlang.agents.eudr.due_diligence_statement_creator.provenance import (
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
def sample_config() -> DDSCreatorConfig:
    """Create a default DDSCreatorConfig instance."""
    return DDSCreatorConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Geolocation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_geolocation() -> GeolocationData:
    """Create a sample geolocation record."""
    return GeolocationData(
        plot_id="PLT-001",
        latitude=Decimal("5.123456"),
        longitude=Decimal("-3.456789"),
        area_hectares=Decimal("2.50"),
        country_code="CI",
        region="Ivory Coast - San Pedro",
        collection_method=GeolocationMethod.GPS_FIELD_SURVEY,
        accuracy_meters=Decimal("5"),
        verified=True,
        verification_source="EUDR-002",
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sample_polygon_geolocation() -> GeolocationData:
    """Create geolocation with polygon for plot >= 4ha."""
    return GeolocationData(
        plot_id="PLT-002",
        latitude=Decimal("5.200000"),
        longitude=Decimal("-3.500000"),
        area_hectares=Decimal("12.50"),
        polygon_coordinates=[
            [Decimal("5.19"), Decimal("-3.49")],
            [Decimal("5.21"), Decimal("-3.49")],
            [Decimal("5.21"), Decimal("-3.51")],
            [Decimal("5.19"), Decimal("-3.51")],
            [Decimal("5.19"), Decimal("-3.49")],
        ],
        country_code="CI",
        collection_method=GeolocationMethod.SATELLITE_DERIVED,
        verified=True,
        provenance_hash="b" * 64,
    )


# ---------------------------------------------------------------------------
# Risk fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_risk_reference() -> RiskReference:
    """Create a sample risk reference."""
    return RiskReference(
        risk_id="RISK-001",
        source_agent="EUDR-016",
        risk_category="country",
        risk_level=RiskLevel.STANDARD,
        risk_score=Decimal("45.00"),
        factors=["tropical_deforestation", "governance_gaps"],
        mitigation_measures=["enhanced_monitoring"],
        data_sources=["global_forest_watch", "transparency_international"],
        provenance_hash="c" * 64,
    )


@pytest.fixture
def multiple_risk_references() -> List[RiskReference]:
    """Create multiple risk references from different agents."""
    return [
        RiskReference(
            risk_id="RISK-001",
            source_agent="EUDR-016",
            risk_category="country",
            risk_level=RiskLevel.HIGH,
            risk_score=Decimal("72.00"),
            provenance_hash="d" * 64,
        ),
        RiskReference(
            risk_id="RISK-002",
            source_agent="EUDR-017",
            risk_category="supplier",
            risk_level=RiskLevel.STANDARD,
            risk_score=Decimal("35.00"),
            provenance_hash="e" * 64,
        ),
        RiskReference(
            risk_id="RISK-003",
            source_agent="EUDR-018",
            risk_category="commodity",
            risk_level=RiskLevel.LOW,
            risk_score=Decimal("15.00"),
            provenance_hash="f" * 64,
        ),
    ]


# ---------------------------------------------------------------------------
# Supply chain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_supply_chain() -> SupplyChainData:
    """Create a sample supply chain data record."""
    return SupplyChainData(
        supply_chain_id="SC-001",
        operator_id="OP-001",
        commodity=CommodityType.COCOA,
        tier_count=3,
        supplier_count=12,
        suppliers=[
            {"name": "Farm A", "tier": 1, "country_code": "CI", "plot_count": 5},
            {"name": "Cooperative B", "tier": 2, "country_code": "CI", "plot_count": 0},
        ],
        chain_of_custody_model="segregation",
        plot_count=45,
        countries_of_production=["CI", "GH"],
        traceability_score=Decimal("78.50"),
        provenance_hash="g" * 64,
    )


# ---------------------------------------------------------------------------
# DDS Statement fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_statement() -> DDSStatement:
    """Create a sample DDS statement in DRAFT status."""
    now = datetime.now(tz=timezone.utc)
    return DDSStatement(
        statement_id="DDS-TEST001",
        reference_number="GL-DDS-20260313-ABCDEF",
        operator_id="OP-001",
        operator_name="Acme Trading Ltd",
        operator_address="123 Trade Street, Brussels, Belgium",
        operator_eori_number="BE1234567890",
        statement_type=StatementType.PLACING,
        status=DDSStatus.DRAFT,
        version_number=1,
        commodities=[CommodityType.COCOA, CommodityType.COFFEE],
        countries_of_production=["CI", "GH"],
        total_quantity=Decimal("500.00"),
        language="en",
        date_of_statement=now,
        created_at=now,
        updated_at=now,
        provenance_hash="h" * 64,
    )


@pytest.fixture
def complete_statement(
    sample_statement, sample_geolocation, sample_risk_reference, sample_supply_chain
) -> DDSStatement:
    """Create a DDS statement with geolocation, risk, and supply chain data."""
    sample_statement.geolocations = [sample_geolocation]
    sample_statement.risk_references = [sample_risk_reference]
    sample_statement.supply_chain_data = sample_supply_chain
    sample_statement.compliance_declaration = "I hereby declare compliance with EUDR."
    sample_statement.deforestation_free = True
    sample_statement.legally_produced = True
    sample_statement.risk_mitigation_measures = ["enhanced_monitoring"]
    sample_statement.hs_codes = ["1801.00"]
    sample_statement.product_descriptions = [{"name": "Raw cocoa beans"}]
    return sample_statement


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> DocumentPackage:
    """Create a sample document package."""
    return DocumentPackage(
        document_id="DOC-TEST001",
        document_type=DocumentType.CERTIFICATE_OF_ORIGIN,
        filename="certificate_cocoa_CI.pdf",
        mime_type="application/pdf",
        size_bytes=1024000,
        hash_sha256="abcd1234" * 8,
        description="Certificate of origin for cocoa from Ivory Coast",
        provenance_hash="i" * 64,
    )


# ---------------------------------------------------------------------------
# Version fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_version() -> StatementVersion:
    """Create a sample statement version."""
    return StatementVersion(
        version_id="VER-TEST001",
        statement_id="DDS-TEST001",
        version_number=1,
        status=DDSStatus.DRAFT,
        created_by="OP-001",
        provenance_hash="j" * 64,
    )


# ---------------------------------------------------------------------------
# Signature fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signature() -> DigitalSignature:
    """Create a sample digital signature."""
    now = datetime.now(tz=timezone.utc)
    return DigitalSignature(
        signature_id="SIG-TEST001",
        statement_id="DDS-TEST001",
        signer_name="John Smith",
        signer_role="Compliance Officer",
        signer_organization="Acme Trading Ltd",
        signature_type=SignatureType.QUALIFIED_ELECTRONIC,
        algorithm="RSA-SHA256",
        signed_hash="e" * 64,
        timestamp=now,
        valid_from=now,
        valid_until=now + timedelta(days=365),
        is_valid=True,
        provenance_hash="k" * 64,
    )
