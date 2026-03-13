# -*- coding: utf-8 -*-
"""
Shared test fixtures for AGENT-EUDR-023: Legal Compliance Verifier.

Provides mock configurations, sample data, test factories, and engine
instances for all test modules in the Legal Compliance Verifier test suite.
Fixtures cover legal framework management, document verification,
certification scheme validation, red flag detection, country compliance
checking, third-party audit integration, and compliance reporting.

Fixture Categories (70+ fixtures):
    - Configuration fixtures: mock_config, strict_config, reset_singleton_config
    - Legal framework fixtures: sample_legal_frameworks, sample_legislation_db
    - Document fixtures: sample_documents, expired_documents, valid_documents
    - Certification fixtures: fsc_certificates, pefc_certificates, rspo_certificates
    - Red flag fixtures: sample_red_flags, red_flag_indicators_all
    - Country compliance fixtures: sample_country_rules, sample_gap_analysis
    - Audit fixtures: sample_audit_reports, sample_findings
    - Report fixtures: report_config, sample_report_data
    - Engine mock fixtures: framework_engine, doc_engine, cert_engine, etc.
    - External API mocks: mock_faolex, mock_ecolex, mock_fsc_api, mock_rspo_api
    - Provenance fixtures: mock_provenance
    - Metrics fixtures: mock_metrics
    - Constants: legislation categories, commodities, document types

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.legal_compliance_verifier.config import (
    LegalComplianceVerifierConfig,
    get_config,
    set_config,
    reset_config,
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
# Shared Constants
# ---------------------------------------------------------------------------

#: SHA-256 hash length in hexadecimal characters.
SHA256_HEX_LENGTH = 64

#: EUDR Article 2(40) legislation categories (8 categories).
LEGISLATION_CATEGORIES = [
    "land_use_rights",
    "environmental_protection",
    "forest_related_rules",
    "third_party_rights",
    "labour_rights",
    "tax_and_royalty",
    "trade_and_customs",
    "anti_corruption",
]

#: EUDR Article 1 commodities (7 commodities).
EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

#: Document types supported by the verification engine (12 types).
DOCUMENT_TYPES = [
    "land_title", "concession_permit", "environmental_impact_assessment",
    "forest_management_plan", "harvest_permit", "export_license",
    "phytosanitary_certificate", "certificate_of_origin",
    "tax_clearance_certificate", "labour_compliance_certificate",
    "indigenous_consent_document", "anti_corruption_declaration",
]

#: Certification schemes supported (6 schemes).
CERTIFICATION_SCHEMES = [
    "fsc", "pefc", "rspo", "rainforest_alliance", "iscc", "other",
]

#: FSC sub-schemes (5 sub-schemes).
FSC_SUB_SCHEMES = [
    "fsc_fm",       # Forest Management
    "fsc_coc",      # Chain of Custody
    "fsc_cw",       # Controlled Wood
    "fsc_project",  # Project Certification
    "fsc_ecosystem",  # Ecosystem Services
]

#: Red flag severity levels.
RED_FLAG_SEVERITIES = ["critical", "high", "moderate", "low"]

#: Red flag categories (6 categories).
RED_FLAG_CATEGORIES = [
    "documentation", "certification", "geographic",
    "supplier", "regulatory", "operational",
]

#: Compliance determination values.
COMPLIANCE_DETERMINATIONS = ["COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT"]

#: Audit report formats (6 formats).
AUDIT_REPORT_FORMATS = [
    "iso_19011", "fsc_audit", "pefc_audit", "rspo_audit",
    "custom_pdf", "structured_json",
]

#: Report types (8 types).
REPORT_TYPES = [
    "full_assessment", "category_specific", "supplier_scorecard",
    "red_flag_summary", "document_status", "certification_validity",
    "country_framework", "dds_annex",
]

#: Report output formats (5 formats).
REPORT_FORMATS = ["pdf", "json", "html", "xbrl", "xml"]

#: Supported languages (5 languages).
SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "pt"]

#: 27 EUDR-relevant countries for legal framework coverage.
EUDR_COUNTRIES_27 = [
    "BR", "ID", "MY", "CO", "CM", "GH", "CI", "CD", "CG", "PE",
    "BO", "PY", "EC", "VN", "TH", "MM", "PH", "NG", "LR", "SL",
    "GA", "GN", "TZ", "MZ", "PG", "LA", "KH",
]

#: High-risk EUDR countries (for testing).
HIGH_RISK_COUNTRIES = ["BR", "ID", "CD", "CM", "CI", "CG", "MM", "NG"]

#: Low-risk EUDR countries (for testing).
LOW_RISK_COUNTRIES = ["DK", "FI", "SE", "DE", "NL"]

#: Compliance score boundary values.
COMPLIANCE_BOUNDARIES = [0, 25, 49, 50, 79, 80, 100]

#: Red flag score boundary values.
RED_FLAG_BOUNDARIES = [0, 10, 24, 25, 49, 50, 74, 75, 100]

#: Country risk multipliers for red flag scoring.
COUNTRY_MULTIPLIERS = {
    "CD": Decimal("1.8"),
    "CM": Decimal("1.6"),
    "MM": Decimal("1.7"),
    "NG": Decimal("1.5"),
    "BR": Decimal("1.3"),
    "ID": Decimal("1.2"),
    "MY": Decimal("1.1"),
    "GH": Decimal("1.2"),
    "DK": Decimal("0.5"),
    "FI": Decimal("0.5"),
}

#: Commodity risk multipliers for red flag scoring.
COMMODITY_MULTIPLIERS = {
    "oil_palm": Decimal("1.5"),
    "cattle": Decimal("1.4"),
    "soya": Decimal("1.3"),
    "cocoa": Decimal("1.3"),
    "coffee": Decimal("1.2"),
    "rubber": Decimal("1.2"),
    "wood": Decimal("1.1"),
}


# ---------------------------------------------------------------------------
# 40 Red Flag Indicators (for test_red_flag_detection_engine)
# ---------------------------------------------------------------------------

RED_FLAG_INDICATORS = [
    # Documentation (8 indicators)
    {"id": "RF-DOC-001", "name": "missing_land_title", "category": "documentation",
     "base_score": 85, "description": "No land title or concession permit provided"},
    {"id": "RF-DOC-002", "name": "expired_permit", "category": "documentation",
     "base_score": 70, "description": "Operating permit has expired"},
    {"id": "RF-DOC-003", "name": "mismatched_coordinates", "category": "documentation",
     "base_score": 75, "description": "GPS coordinates do not match permit boundaries"},
    {"id": "RF-DOC-004", "name": "unsigned_declaration", "category": "documentation",
     "base_score": 60, "description": "Due diligence declaration not signed"},
    {"id": "RF-DOC-005", "name": "forged_signature_suspect", "category": "documentation",
     "base_score": 90, "description": "Signature authenticity questionable"},
    {"id": "RF-DOC-006", "name": "incomplete_eia", "category": "documentation",
     "base_score": 65, "description": "Environmental impact assessment incomplete"},
    {"id": "RF-DOC-007", "name": "outdated_forest_plan", "category": "documentation",
     "base_score": 55, "description": "Forest management plan older than 5 years"},
    {"id": "RF-DOC-008", "name": "missing_tax_clearance", "category": "documentation",
     "base_score": 50, "description": "Tax clearance certificate not provided"},
    # Certification (7 indicators)
    {"id": "RF-CRT-001", "name": "revoked_certification", "category": "certification",
     "base_score": 95, "description": "Certification has been revoked"},
    {"id": "RF-CRT-002", "name": "suspended_certificate", "category": "certification",
     "base_score": 85, "description": "Certification currently suspended"},
    {"id": "RF-CRT-003", "name": "expired_certificate", "category": "certification",
     "base_score": 70, "description": "Certification has expired"},
    {"id": "RF-CRT-004", "name": "unrecognized_scheme", "category": "certification",
     "base_score": 60, "description": "Certification scheme not EUDR-recognized"},
    {"id": "RF-CRT-005", "name": "scope_mismatch", "category": "certification",
     "base_score": 65, "description": "Certificate scope does not cover commodity"},
    {"id": "RF-CRT-006", "name": "coc_chain_break", "category": "certification",
     "base_score": 80, "description": "Chain of custody break detected"},
    {"id": "RF-CRT-007", "name": "audit_nonconformity", "category": "certification",
     "base_score": 55, "description": "Recent audit found major nonconformities"},
    # Geographic (6 indicators)
    {"id": "RF-GEO-001", "name": "protected_area_overlap", "category": "geographic",
     "base_score": 95, "description": "Production area overlaps protected zone"},
    {"id": "RF-GEO-002", "name": "deforestation_hotspot", "category": "geographic",
     "base_score": 90, "description": "Located in active deforestation hotspot"},
    {"id": "RF-GEO-003", "name": "indigenous_territory", "category": "geographic",
     "base_score": 85, "description": "Overlaps indigenous territory without FPIC"},
    {"id": "RF-GEO-004", "name": "border_conflict_zone", "category": "geographic",
     "base_score": 70, "description": "Near active conflict zone"},
    {"id": "RF-GEO-005", "name": "high_biodiversity_area", "category": "geographic",
     "base_score": 65, "description": "Located in high-biodiversity area"},
    {"id": "RF-GEO-006", "name": "unverifiable_coordinates", "category": "geographic",
     "base_score": 75, "description": "GPS coordinates cannot be verified"},
    # Supplier (7 indicators)
    {"id": "RF-SUP-001", "name": "sanctions_list_match", "category": "supplier",
     "base_score": 98, "description": "Supplier matches sanctions/debarment list"},
    {"id": "RF-SUP-002", "name": "shell_company_suspect", "category": "supplier",
     "base_score": 85, "description": "Supplier characteristics match shell company"},
    {"id": "RF-SUP-003", "name": "ownership_opacity", "category": "supplier",
     "base_score": 70, "description": "Beneficial ownership not transparent"},
    {"id": "RF-SUP-004", "name": "historical_violations", "category": "supplier",
     "base_score": 75, "description": "Supplier has history of environmental violations"},
    {"id": "RF-SUP-005", "name": "rapid_ownership_change", "category": "supplier",
     "base_score": 65, "description": "Multiple ownership changes in last 2 years"},
    {"id": "RF-SUP-006", "name": "no_physical_presence", "category": "supplier",
     "base_score": 60, "description": "No verifiable physical presence at origin"},
    {"id": "RF-SUP-007", "name": "blacklisted_network", "category": "supplier",
     "base_score": 90, "description": "Part of known non-compliant supplier network"},
    # Regulatory (6 indicators)
    {"id": "RF-REG-001", "name": "pending_legal_action", "category": "regulatory",
     "base_score": 80, "description": "Active legal proceedings against supplier"},
    {"id": "RF-REG-002", "name": "regulatory_warning", "category": "regulatory",
     "base_score": 55, "description": "Regulatory authority issued warning"},
    {"id": "RF-REG-003", "name": "non_compliant_history", "category": "regulatory",
     "base_score": 70, "description": "History of regulatory non-compliance"},
    {"id": "RF-REG-004", "name": "tax_evasion_suspect", "category": "regulatory",
     "base_score": 75, "description": "Evidence of tax irregularities"},
    {"id": "RF-REG-005", "name": "customs_fraud_suspect", "category": "regulatory",
     "base_score": 80, "description": "Customs declaration inconsistencies"},
    {"id": "RF-REG-006", "name": "embargo_violation", "category": "regulatory",
     "base_score": 95, "description": "Potential embargo or trade restriction violation"},
    # Operational (6 indicators)
    {"id": "RF-OPS-001", "name": "forced_labour_risk", "category": "operational",
     "base_score": 95, "description": "Indicators of forced or child labour"},
    {"id": "RF-OPS-002", "name": "unsafe_conditions", "category": "operational",
     "base_score": 70, "description": "Reported unsafe working conditions"},
    {"id": "RF-OPS-003", "name": "community_conflict", "category": "operational",
     "base_score": 65, "description": "Active community land conflict"},
    {"id": "RF-OPS-004", "name": "night_harvesting", "category": "operational",
     "base_score": 60, "description": "Evidence of unauthorized night operations"},
    {"id": "RF-OPS-005", "name": "waste_dumping", "category": "operational",
     "base_score": 75, "description": "Evidence of illegal waste disposal"},
    {"id": "RF-OPS-006", "name": "water_pollution", "category": "operational",
     "base_score": 70, "description": "Evidence of water source contamination"},
]

assert len(RED_FLAG_INDICATORS) == 40, f"Expected 40 indicators, got {len(RED_FLAG_INDICATORS)}"


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a LegalComplianceVerifierConfig with test defaults.

    Provides a configuration suitable for testing with sensible defaults
    that mirror production but with test database URLs and reduced limits.
    """
    return LegalComplianceVerifierConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        compliant_threshold=80,
        partial_threshold=50,
        red_flag_critical_threshold=75,
        red_flag_high_threshold=50,
        red_flag_moderate_threshold=25,
        document_expiry_warning_days=[90, 60, 30],
        doc_verification_weights={
            "documents_present": 0.40,
            "document_validity": 0.30,
            "scope_alignment": 0.20,
            "authenticity": 0.10,
        },
        faolex_api_url="https://faolex.fao.org/api/v1",
        ecolex_api_url="https://www.ecolex.org/api/v1",
        fsc_api_url="https://info.fsc.org/api/v1",
        rspo_api_url="https://rspo.org/api/v1",
        pefc_api_url="https://pefc.org/api/v1",
        iscc_api_url="https://www.iscc-system.org/api/v1",
        external_api_timeout_s=30,
        batch_max_size=1000,
        batch_concurrency=10,
        batch_timeout_s=120,
        default_report_format="pdf",
        default_language="en",
        retention_years=5,
        enable_provenance=True,
        genesis_hash="GL-EUDR-LCV-023-TEST-GENESIS",
        chain_algorithm="sha256",
        enable_metrics=False,
        pool_size=5,
        rate_limit_default=100,
        rate_limit_batch=10,
        rate_limit_admin=20,
    )


@pytest.fixture
def strict_config():
    """Create a LegalComplianceVerifierConfig with strict thresholds.

    Uses higher compliant threshold (90) and lower partial threshold (40)
    to test boundary behavior under strict configuration.
    """
    return LegalComplianceVerifierConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        compliant_threshold=90,
        partial_threshold=40,
        red_flag_critical_threshold=70,
        red_flag_high_threshold=45,
        red_flag_moderate_threshold=20,
        enable_provenance=True,
        genesis_hash="GL-EUDR-LCV-023-TEST-STRICT-GENESIS",
        enable_metrics=False,
        pool_size=5,
    )


@pytest.fixture(autouse=True)
def reset_singleton_config():
    """Reset the singleton config after each test to avoid cross-test leaks."""
    yield
    reset_config()


@pytest.fixture
def uuid_gen():
    """Create a deterministic UUID generator."""
    return DeterministicUUID()


# ---------------------------------------------------------------------------
# Legal Framework Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_legal_frameworks():
    """Sample legal framework entries for multiple countries.

    Covers all 8 EUDR Article 2(40) legislation categories for Brazil,
    Indonesia, and DR Congo with realistic legislation references.
    """
    return {
        "BR": {
            "country_code": "BR",
            "country_name": "Brazil",
            "frameworks": {
                "land_use_rights": {
                    "legislation": "Lei 4.504/1964 - Estatuto da Terra",
                    "authority": "INCRA",
                    "status": "active",
                    "last_updated": "2023-06-15",
                    "reliability_score": Decimal("85"),
                },
                "environmental_protection": {
                    "legislation": "Lei 12.651/2012 - Codigo Florestal",
                    "authority": "IBAMA",
                    "status": "active",
                    "last_updated": "2023-09-01",
                    "reliability_score": Decimal("90"),
                },
                "forest_related_rules": {
                    "legislation": "Decreto 6.514/2008 - Forest Infractions",
                    "authority": "IBAMA / ICMBio",
                    "status": "active",
                    "last_updated": "2022-12-01",
                    "reliability_score": Decimal("88"),
                },
                "third_party_rights": {
                    "legislation": "FPIC - Convencao OIT 169",
                    "authority": "FUNAI",
                    "status": "active",
                    "last_updated": "2023-03-15",
                    "reliability_score": Decimal("75"),
                },
                "labour_rights": {
                    "legislation": "CLT - Consolidacao das Leis do Trabalho",
                    "authority": "Ministerio do Trabalho",
                    "status": "active",
                    "last_updated": "2023-07-01",
                    "reliability_score": Decimal("82"),
                },
                "tax_and_royalty": {
                    "legislation": "Codigo Tributario Nacional",
                    "authority": "Receita Federal",
                    "status": "active",
                    "last_updated": "2024-01-01",
                    "reliability_score": Decimal("92"),
                },
                "trade_and_customs": {
                    "legislation": "Regulamento Aduaneiro - Decreto 6.759/2009",
                    "authority": "Receita Federal",
                    "status": "active",
                    "last_updated": "2023-11-15",
                    "reliability_score": Decimal("88"),
                },
                "anti_corruption": {
                    "legislation": "Lei 12.846/2013 - Lei Anticorrupcao",
                    "authority": "CGU",
                    "status": "active",
                    "last_updated": "2023-08-01",
                    "reliability_score": Decimal("80"),
                },
            },
        },
        "ID": {
            "country_code": "ID",
            "country_name": "Indonesia",
            "frameworks": {
                "land_use_rights": {
                    "legislation": "UU 5/1960 Agrarian Law",
                    "authority": "BPN",
                    "status": "active",
                    "last_updated": "2022-09-01",
                    "reliability_score": Decimal("70"),
                },
                "environmental_protection": {
                    "legislation": "UU 32/2009 Environmental Protection",
                    "authority": "KLHK",
                    "status": "active",
                    "last_updated": "2023-04-01",
                    "reliability_score": Decimal("75"),
                },
                "forest_related_rules": {
                    "legislation": "PP 23/2021 Forestry Management",
                    "authority": "KLHK",
                    "status": "active",
                    "last_updated": "2023-06-01",
                    "reliability_score": Decimal("72"),
                },
                "third_party_rights": {
                    "legislation": "UU 39/1999 Human Rights",
                    "authority": "Komnas HAM",
                    "status": "active",
                    "last_updated": "2022-06-01",
                    "reliability_score": Decimal("60"),
                },
                "labour_rights": {
                    "legislation": "UU 13/2003 Manpower",
                    "authority": "Kemnaker",
                    "status": "active",
                    "last_updated": "2023-01-01",
                    "reliability_score": Decimal("68"),
                },
                "tax_and_royalty": {
                    "legislation": "UU 7/2021 Harmonization of Tax Regulations",
                    "authority": "DJP",
                    "status": "active",
                    "last_updated": "2023-10-01",
                    "reliability_score": Decimal("78"),
                },
                "trade_and_customs": {
                    "legislation": "UU 17/2006 Customs",
                    "authority": "DJBC",
                    "status": "active",
                    "last_updated": "2023-08-01",
                    "reliability_score": Decimal("74"),
                },
                "anti_corruption": {
                    "legislation": "UU 31/1999 Anti-Corruption",
                    "authority": "KPK",
                    "status": "active",
                    "last_updated": "2023-05-01",
                    "reliability_score": Decimal("65"),
                },
            },
        },
        "CD": {
            "country_code": "CD",
            "country_name": "Democratic Republic of Congo",
            "frameworks": {
                "land_use_rights": {
                    "legislation": "Loi 73-021 du 20 juillet 1973 - Regime foncier",
                    "authority": "Ministere des Affaires Foncieres",
                    "status": "active",
                    "last_updated": "2020-01-01",
                    "reliability_score": Decimal("35"),
                },
                "environmental_protection": {
                    "legislation": "Loi 11/009 du 9 juillet 2011 - Protection de l'Environnement",
                    "authority": "Ministere de l'Environnement",
                    "status": "active",
                    "last_updated": "2021-06-01",
                    "reliability_score": Decimal("30"),
                },
                "forest_related_rules": {
                    "legislation": "Loi 011/2002 du 29 aout 2002 - Code Forestier",
                    "authority": "Ministere de l'Environnement",
                    "status": "active",
                    "last_updated": "2022-03-01",
                    "reliability_score": Decimal("40"),
                },
                "third_party_rights": {
                    "legislation": "Loi portant protection des Peuples Autochtones Pygmees",
                    "authority": "Ministere des Droits Humains",
                    "status": "active",
                    "last_updated": "2022-01-01",
                    "reliability_score": Decimal("25"),
                },
                "labour_rights": {
                    "legislation": "Code du Travail - Loi 015/2002",
                    "authority": "Ministere du Travail",
                    "status": "active",
                    "last_updated": "2021-09-01",
                    "reliability_score": Decimal("30"),
                },
                "tax_and_royalty": {
                    "legislation": "Loi 004/2003 - Regime Fiscal",
                    "authority": "DGI",
                    "status": "active",
                    "last_updated": "2022-07-01",
                    "reliability_score": Decimal("35"),
                },
                "trade_and_customs": {
                    "legislation": "Code des Douanes - Ordonnance-Loi 10/002",
                    "authority": "DGDA",
                    "status": "active",
                    "last_updated": "2021-12-01",
                    "reliability_score": Decimal("32"),
                },
                "anti_corruption": {
                    "legislation": "Loi Organique 13/011-B",
                    "authority": "CNLC",
                    "status": "active",
                    "last_updated": "2022-05-01",
                    "reliability_score": Decimal("20"),
                },
            },
        },
    }


@pytest.fixture
def sample_legislation_db():
    """Flat list of legislation entries for search/filter testing."""
    entries = []
    for i, (country, category) in enumerate([
        ("BR", "land_use_rights"), ("BR", "environmental_protection"),
        ("BR", "forest_related_rules"), ("BR", "labour_rights"),
        ("ID", "land_use_rights"), ("ID", "environmental_protection"),
        ("ID", "forest_related_rules"), ("ID", "labour_rights"),
        ("CD", "land_use_rights"), ("CD", "environmental_protection"),
        ("GH", "land_use_rights"), ("GH", "environmental_protection"),
        ("CM", "land_use_rights"), ("CM", "forest_related_rules"),
        ("CO", "environmental_protection"), ("CO", "anti_corruption"),
    ]):
        entries.append({
            "id": f"LEG-{i+1:04d}",
            "country_code": country,
            "category": category,
            "title": f"Legislation {country}-{category}",
            "status": "active" if i % 5 != 0 else "repealed",
            "year_enacted": 2000 + (i % 25),
            "last_amended": f"202{i % 5}-01-01",
            "reliability_score": Decimal(str(50 + (i * 3) % 50)),
        })
    return entries


# ---------------------------------------------------------------------------
# Document Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents():
    """Sample documents of all 12 types with varied validity states."""
    docs = []
    today = date.today()
    for i, doc_type in enumerate(DOCUMENT_TYPES):
        # Alternate between valid, expiring_soon, and expired
        if i % 3 == 0:
            expiry = today + timedelta(days=365)
            status = "valid"
        elif i % 3 == 1:
            expiry = today + timedelta(days=45)
            status = "expiring_soon"
        else:
            expiry = today - timedelta(days=30)
            status = "expired"

        docs.append({
            "document_id": f"DOC-{i+1:04d}",
            "document_type": doc_type,
            "issuing_authority": f"Authority-{doc_type[:4].upper()}",
            "issue_date": (today - timedelta(days=365)).isoformat(),
            "expiry_date": expiry.isoformat(),
            "country_code": EUDR_COUNTRIES_27[i % len(EUDR_COUNTRIES_27)],
            "supplier_id": f"SUP-{(i % 5)+1:04d}",
            "status": status,
            "file_hash": compute_test_hash({"doc_id": f"DOC-{i+1:04d}", "type": doc_type}),
            "verified": status == "valid",
            "verification_notes": f"Verification notes for {doc_type}",
        })
    return docs


@pytest.fixture
def valid_documents():
    """Set of 12 documents (one per type), all currently valid."""
    docs = []
    today = date.today()
    for i, doc_type in enumerate(DOCUMENT_TYPES):
        docs.append({
            "document_id": f"VDOC-{i+1:04d}",
            "document_type": doc_type,
            "issuing_authority": f"Official-Authority-{i+1}",
            "issue_date": (today - timedelta(days=180)).isoformat(),
            "expiry_date": (today + timedelta(days=365)).isoformat(),
            "country_code": "BR",
            "supplier_id": "SUP-0001",
            "status": "valid",
            "file_hash": compute_test_hash({"doc_id": f"VDOC-{i+1:04d}"}),
            "verified": True,
        })
    return docs


@pytest.fixture
def expired_documents():
    """Set of documents that are all expired."""
    docs = []
    today = date.today()
    for i in range(6):
        docs.append({
            "document_id": f"EDOC-{i+1:04d}",
            "document_type": DOCUMENT_TYPES[i],
            "issuing_authority": f"Authority-{i+1}",
            "issue_date": (today - timedelta(days=730)).isoformat(),
            "expiry_date": (today - timedelta(days=i * 30 + 1)).isoformat(),
            "country_code": "BR",
            "supplier_id": "SUP-0002",
            "status": "expired",
            "file_hash": compute_test_hash({"doc_id": f"EDOC-{i+1:04d}"}),
            "verified": False,
        })
    return docs


@pytest.fixture
def expiring_soon_documents():
    """Set of documents expiring within 90 days."""
    docs = []
    today = date.today()
    for i, days_remaining in enumerate([5, 15, 29, 30, 59, 60, 89, 90]):
        docs.append({
            "document_id": f"XDOC-{i+1:04d}",
            "document_type": DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)],
            "issuing_authority": f"Authority-{i+1}",
            "issue_date": (today - timedelta(days=365)).isoformat(),
            "expiry_date": (today + timedelta(days=days_remaining)).isoformat(),
            "country_code": "ID",
            "supplier_id": "SUP-0003",
            "status": "expiring_soon",
            "file_hash": compute_test_hash({"doc_id": f"XDOC-{i+1:04d}"}),
            "verified": True,
        })
    return docs


# ---------------------------------------------------------------------------
# Certification Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fsc_certificates():
    """Sample FSC certificates covering all 5 sub-schemes."""
    certs = []
    today = date.today()
    for i, sub_scheme in enumerate(FSC_SUB_SCHEMES):
        certs.append({
            "certificate_id": f"FSC-{sub_scheme.upper()}-{i+1:04d}",
            "scheme": "fsc",
            "sub_scheme": sub_scheme,
            "holder_name": f"FSC Holder {i+1}",
            "certificate_code": f"FSC-C{100000+i:06d}",
            "issue_date": (today - timedelta(days=365)).isoformat(),
            "expiry_date": (today + timedelta(days=365*(4-i))).isoformat(),
            "status": "valid" if i < 4 else "expired",
            "scope": ["wood", "rubber"] if "fm" in sub_scheme else EUDR_COMMODITIES[:3],
            "country_code": "BR",
            "audit_date": (today - timedelta(days=90)).isoformat(),
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": f"FSC-{sub_scheme}"}),
        })
    return certs


@pytest.fixture
def pefc_certificates():
    """Sample PEFC certificates."""
    today = date.today()
    return [
        {
            "certificate_id": "PEFC-001",
            "scheme": "pefc",
            "sub_scheme": "pefc_sfm",
            "holder_name": "PEFC Holder 1",
            "certificate_code": "PEFC/XX-XX-XXXXXXX",
            "issue_date": (today - timedelta(days=200)).isoformat(),
            "expiry_date": (today + timedelta(days=900)).isoformat(),
            "status": "valid",
            "scope": ["wood"],
            "country_code": "ID",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "PEFC-001"}),
        },
        {
            "certificate_id": "PEFC-002",
            "scheme": "pefc",
            "sub_scheme": "pefc_coc",
            "holder_name": "PEFC Holder 2",
            "certificate_code": "PEFC/XX-XX-YYYYYYY",
            "issue_date": (today - timedelta(days=800)).isoformat(),
            "expiry_date": (today - timedelta(days=10)).isoformat(),
            "status": "expired",
            "scope": ["wood"],
            "country_code": "GH",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "PEFC-002"}),
        },
    ]


@pytest.fixture
def rspo_certificates():
    """Sample RSPO certificates for oil palm."""
    today = date.today()
    return [
        {
            "certificate_id": "RSPO-001",
            "scheme": "rspo",
            "sub_scheme": "rspo_ip",
            "holder_name": "Palm Oil Producer 1",
            "certificate_code": "RSPO-1234567",
            "issue_date": (today - timedelta(days=300)).isoformat(),
            "expiry_date": (today + timedelta(days=700)).isoformat(),
            "status": "valid",
            "scope": ["oil_palm"],
            "country_code": "MY",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "RSPO-001"}),
        },
        {
            "certificate_id": "RSPO-002",
            "scheme": "rspo",
            "sub_scheme": "rspo_mb",
            "holder_name": "Palm Oil Producer 2",
            "certificate_code": "RSPO-7654321",
            "issue_date": (today - timedelta(days=400)).isoformat(),
            "expiry_date": (today + timedelta(days=200)).isoformat(),
            "status": "valid",
            "scope": ["oil_palm"],
            "country_code": "ID",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "RSPO-002"}),
        },
        {
            "certificate_id": "RSPO-003",
            "scheme": "rspo",
            "sub_scheme": "rspo_sg",
            "holder_name": "Palm Oil Trader 1",
            "certificate_code": "RSPO-SUSPEND-001",
            "issue_date": (today - timedelta(days=500)).isoformat(),
            "expiry_date": (today + timedelta(days=100)).isoformat(),
            "status": "suspended",
            "scope": ["oil_palm"],
            "country_code": "ID",
            "eudr_equivalent": False,
            "provenance_hash": compute_test_hash({"cert": "RSPO-003"}),
        },
    ]


@pytest.fixture
def rainforest_alliance_certificates():
    """Sample Rainforest Alliance certificates."""
    today = date.today()
    return [
        {
            "certificate_id": "RA-001",
            "scheme": "rainforest_alliance",
            "sub_scheme": "ra_2020",
            "holder_name": "Coffee Farm 1",
            "certificate_code": "RA-CF-2024-001",
            "issue_date": (today - timedelta(days=180)).isoformat(),
            "expiry_date": (today + timedelta(days=550)).isoformat(),
            "status": "valid",
            "scope": ["coffee", "cocoa"],
            "country_code": "CO",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "RA-001"}),
        },
    ]


@pytest.fixture
def iscc_certificates():
    """Sample ISCC certificates."""
    today = date.today()
    return [
        {
            "certificate_id": "ISCC-001",
            "scheme": "iscc",
            "sub_scheme": "iscc_eu",
            "holder_name": "Soya Producer 1",
            "certificate_code": "ISCC-EU-2024-001",
            "issue_date": (today - timedelta(days=100)).isoformat(),
            "expiry_date": (today + timedelta(days=265)).isoformat(),
            "status": "valid",
            "scope": ["soya", "oil_palm"],
            "country_code": "BR",
            "eudr_equivalent": True,
            "provenance_hash": compute_test_hash({"cert": "ISCC-001"}),
        },
    ]


@pytest.fixture
def all_certificates(fsc_certificates, pefc_certificates, rspo_certificates,
                     rainforest_alliance_certificates, iscc_certificates):
    """Combined list of all certification scheme certificates."""
    return (fsc_certificates + pefc_certificates + rspo_certificates
            + rainforest_alliance_certificates + iscc_certificates)


# ---------------------------------------------------------------------------
# Red Flag Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_red_flags():
    """Sample detected red flags for a supplier."""
    return [
        {
            "flag_id": "FLAG-001",
            "indicator_id": "RF-DOC-001",
            "indicator_name": "missing_land_title",
            "category": "documentation",
            "severity": "critical",
            "base_score": 85,
            "adjusted_score": Decimal("110.5"),  # 85 * 1.3 (BR multiplier)
            "country_code": "BR",
            "commodity": "soya",
            "supplier_id": "SUP-0001",
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "description": "No land title or concession permit provided",
            "evidence": ["Missing document in DDS submission"],
            "false_positive": False,
        },
        {
            "flag_id": "FLAG-002",
            "indicator_id": "RF-CRT-003",
            "indicator_name": "expired_certificate",
            "category": "certification",
            "severity": "high",
            "base_score": 70,
            "adjusted_score": Decimal("91.0"),  # 70 * 1.3 (BR) * 1.0 (commodity)
            "country_code": "BR",
            "commodity": "soya",
            "supplier_id": "SUP-0001",
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "description": "Certification has expired",
            "evidence": ["FSC certificate expired 2024-06-01"],
            "false_positive": False,
        },
        {
            "flag_id": "FLAG-003",
            "indicator_id": "RF-GEO-002",
            "indicator_name": "deforestation_hotspot",
            "category": "geographic",
            "severity": "critical",
            "base_score": 90,
            "adjusted_score": Decimal("117.0"),  # 90 * 1.3 (BR)
            "country_code": "BR",
            "commodity": "cattle",
            "supplier_id": "SUP-0002",
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "description": "Located in active deforestation hotspot",
            "evidence": ["GLAD alert within 5km radius"],
            "false_positive": False,
        },
    ]


@pytest.fixture
def red_flag_indicators_all():
    """All 40 red flag indicators as defined in the specification."""
    return list(RED_FLAG_INDICATORS)


# ---------------------------------------------------------------------------
# Country Compliance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_country_rules():
    """Country-specific compliance rules for 27 countries.

    Returns a dictionary mapping country codes to their required documents,
    specific rules, and compliance requirements per legislation category.
    """
    rules = {}
    for country in EUDR_COUNTRIES_27:
        country_rules = {}
        for category in LEGISLATION_CATEGORIES:
            country_rules[category] = {
                "required_documents": _get_required_docs(country, category),
                "minimum_score": 50,
                "special_requirements": _get_special_requirements(country, category),
                "evidence_level": "primary" if country in HIGH_RISK_COUNTRIES else "secondary",
            }
        rules[country] = {
            "country_code": country,
            "rules": country_rules,
            "overall_risk_level": "high" if country in HIGH_RISK_COUNTRIES else "standard",
        }
    return rules


@pytest.fixture
def sample_gap_analysis():
    """Sample gap analysis result for a partially compliant supplier."""
    return {
        "supplier_id": "SUP-0001",
        "country_code": "BR",
        "commodity": "soya",
        "assessment_date": datetime.now(timezone.utc).isoformat(),
        "overall_score": Decimal("62"),
        "determination": "PARTIALLY_COMPLIANT",
        "category_scores": {
            "land_use_rights": {"score": Decimal("80"), "status": "COMPLIANT"},
            "environmental_protection": {"score": Decimal("75"), "status": "PARTIALLY_COMPLIANT"},
            "forest_related_rules": {"score": Decimal("70"), "status": "PARTIALLY_COMPLIANT"},
            "third_party_rights": {"score": Decimal("45"), "status": "NON_COMPLIANT"},
            "labour_rights": {"score": Decimal("55"), "status": "PARTIALLY_COMPLIANT"},
            "tax_and_royalty": {"score": Decimal("85"), "status": "COMPLIANT"},
            "trade_and_customs": {"score": Decimal("60"), "status": "PARTIALLY_COMPLIANT"},
            "anti_corruption": {"score": Decimal("30"), "status": "NON_COMPLIANT"},
        },
        "gaps": [
            {
                "category": "third_party_rights",
                "gap": "Missing FPIC documentation for indigenous communities",
                "severity": "high",
                "remediation": "Obtain and submit FPIC consent documentation",
            },
            {
                "category": "anti_corruption",
                "gap": "No anti-corruption compliance declaration",
                "severity": "high",
                "remediation": "Submit signed anti-corruption declaration",
            },
            {
                "category": "environmental_protection",
                "gap": "EIA report incomplete - missing water impact section",
                "severity": "moderate",
                "remediation": "Complete water impact assessment section of EIA",
            },
        ],
        "evidence_sufficiency": Decimal("65"),
        "provenance_hash": compute_test_hash({"supplier": "SUP-0001", "score": "62"}),
    }


# ---------------------------------------------------------------------------
# Audit Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit_reports():
    """Sample third-party audit reports in various formats."""
    today = date.today()
    return [
        {
            "audit_id": "AUD-001",
            "format": "iso_19011",
            "auditor": "Bureau Veritas",
            "audit_date": (today - timedelta(days=60)).isoformat(),
            "scope": "Full EUDR compliance assessment",
            "supplier_id": "SUP-0001",
            "country_code": "BR",
            "overall_result": "conditional_pass",
            "findings_count": 3,
            "major_findings": 1,
            "minor_findings": 2,
            "observations": 4,
            "corrective_actions_required": 2,
            "corrective_actions_closed": 1,
            "next_audit_date": (today + timedelta(days=180)).isoformat(),
        },
        {
            "audit_id": "AUD-002",
            "format": "fsc_audit",
            "auditor": "SGS",
            "audit_date": (today - timedelta(days=120)).isoformat(),
            "scope": "FSC Forest Management",
            "supplier_id": "SUP-0002",
            "country_code": "ID",
            "overall_result": "pass",
            "findings_count": 1,
            "major_findings": 0,
            "minor_findings": 1,
            "observations": 2,
            "corrective_actions_required": 1,
            "corrective_actions_closed": 1,
            "next_audit_date": (today + timedelta(days=365)).isoformat(),
        },
        {
            "audit_id": "AUD-003",
            "format": "rspo_audit",
            "auditor": "TUV Rheinland",
            "audit_date": (today - timedelta(days=30)).isoformat(),
            "scope": "RSPO Principles & Criteria",
            "supplier_id": "SUP-0003",
            "country_code": "MY",
            "overall_result": "fail",
            "findings_count": 8,
            "major_findings": 4,
            "minor_findings": 4,
            "observations": 6,
            "corrective_actions_required": 5,
            "corrective_actions_closed": 0,
            "next_audit_date": (today + timedelta(days=90)).isoformat(),
        },
        {
            "audit_id": "AUD-004",
            "format": "pefc_audit",
            "auditor": "DNV",
            "audit_date": (today - timedelta(days=90)).isoformat(),
            "scope": "PEFC Sustainable Forest Management",
            "supplier_id": "SUP-0004",
            "country_code": "GH",
            "overall_result": "pass",
            "findings_count": 0,
            "major_findings": 0,
            "minor_findings": 0,
            "observations": 1,
            "corrective_actions_required": 0,
            "corrective_actions_closed": 0,
            "next_audit_date": (today + timedelta(days=365)).isoformat(),
        },
        {
            "audit_id": "AUD-005",
            "format": "custom_pdf",
            "auditor": "Local Auditor Ltd",
            "audit_date": (today - timedelta(days=200)).isoformat(),
            "scope": "General compliance review",
            "supplier_id": "SUP-0005",
            "country_code": "CD",
            "overall_result": "fail",
            "findings_count": 12,
            "major_findings": 7,
            "minor_findings": 5,
            "observations": 8,
            "corrective_actions_required": 8,
            "corrective_actions_closed": 2,
            "next_audit_date": (today + timedelta(days=60)).isoformat(),
        },
        {
            "audit_id": "AUD-006",
            "format": "structured_json",
            "auditor": "Control Union",
            "audit_date": (today - timedelta(days=45)).isoformat(),
            "scope": "EUDR Due Diligence verification",
            "supplier_id": "SUP-0006",
            "country_code": "CO",
            "overall_result": "conditional_pass",
            "findings_count": 2,
            "major_findings": 0,
            "minor_findings": 2,
            "observations": 3,
            "corrective_actions_required": 2,
            "corrective_actions_closed": 1,
            "next_audit_date": (today + timedelta(days=270)).isoformat(),
        },
    ]


@pytest.fixture
def sample_findings():
    """Sample audit findings for testing extraction and tracking."""
    return [
        {
            "finding_id": "FND-001",
            "audit_id": "AUD-001",
            "severity": "major",
            "category": "environmental_protection",
            "description": "EIA does not cover water impact assessment",
            "evidence": "Section 4.3 of EIA report missing",
            "corrective_action": "Complete water impact assessment",
            "due_date": (date.today() + timedelta(days=60)).isoformat(),
            "status": "open",
        },
        {
            "finding_id": "FND-002",
            "audit_id": "AUD-001",
            "severity": "minor",
            "category": "labour_rights",
            "description": "Worker safety signage not in local language",
            "evidence": "Photo evidence from site visit",
            "corrective_action": "Install bilingual safety signage",
            "due_date": (date.today() + timedelta(days=30)).isoformat(),
            "status": "closed",
        },
        {
            "finding_id": "FND-003",
            "audit_id": "AUD-001",
            "severity": "minor",
            "category": "tax_and_royalty",
            "description": "Tax payment receipt for Q3 2024 missing",
            "evidence": "Records review",
            "corrective_action": "Submit Q3 2024 tax receipt",
            "due_date": (date.today() + timedelta(days=15)).isoformat(),
            "status": "open",
        },
    ]


# ---------------------------------------------------------------------------
# Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def report_config():
    """Configuration for compliance report generation."""
    return {
        "report_type": "full_assessment",
        "format": "pdf",
        "language": "en",
        "include_evidence": True,
        "include_recommendations": True,
        "include_provenance": True,
        "template_version": "2.0",
    }


@pytest.fixture
def sample_report_data():
    """Sample data for populating a compliance report."""
    return {
        "report_id": "RPT-2025-001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "supplier_id": "SUP-0001",
        "supplier_name": "Agro Brasil Ltda",
        "country_code": "BR",
        "commodity": "soya",
        "assessment_period": "2024-01-01 to 2024-12-31",
        "overall_score": Decimal("72"),
        "determination": "PARTIALLY_COMPLIANT",
        "category_count": 8,
        "compliant_categories": 3,
        "partial_categories": 3,
        "non_compliant_categories": 2,
        "red_flags_detected": 3,
        "critical_red_flags": 1,
        "documents_verified": 10,
        "documents_valid": 7,
        "certifications_checked": 2,
        "certifications_valid": 1,
        "audit_findings": 3,
        "open_corrective_actions": 2,
    }


# ---------------------------------------------------------------------------
# External API Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_faolex_api():
    """Mock FAO FAOLEX API responses."""
    mock = MagicMock()
    mock.search_legislation.return_value = {
        "results": [
            {
                "id": "FAOLEX-BR-001",
                "title": "Brazilian Forest Code",
                "country": "BR",
                "type": "legislation",
                "year": 2012,
                "status": "in_force",
                "url": "https://faolex.fao.org/docs/pdf/bra001.pdf",
            },
        ],
        "total": 1,
    }
    mock.get_legislation.return_value = {
        "id": "FAOLEX-BR-001",
        "title": "Brazilian Forest Code",
        "full_text_available": True,
        "amendments": [],
    }
    mock.check_health.return_value = True
    return mock


@pytest.fixture
def mock_ecolex_api():
    """Mock ECOLEX API responses."""
    mock = MagicMock()
    mock.search.return_value = {
        "results": [
            {
                "id": "ECX-001",
                "title": "Convention on Biological Diversity",
                "type": "treaty",
                "parties": ["BR", "ID", "CD"],
                "status": "in_force",
            },
        ],
        "total": 1,
    }
    mock.check_health.return_value = True
    return mock


@pytest.fixture
def mock_fsc_api():
    """Mock FSC Certificate Checker API responses."""
    mock = MagicMock()
    mock.validate_certificate.return_value = {
        "certificate_code": "FSC-C100001",
        "valid": True,
        "holder": "Test Holder",
        "type": "FM/COC",
        "status": "valid",
        "issue_date": "2023-01-01",
        "expiry_date": "2028-01-01",
        "scope": ["wood", "rubber"],
    }
    mock.check_certificate_status.return_value = "valid"
    mock.check_health.return_value = True
    return mock


@pytest.fixture
def mock_rspo_api():
    """Mock RSPO API responses."""
    mock = MagicMock()
    mock.validate_certificate.return_value = {
        "certificate_code": "RSPO-1234567",
        "valid": True,
        "holder": "Palm Oil Producer",
        "supply_chain_model": "IP",
        "status": "valid",
        "issue_date": "2023-06-01",
        "expiry_date": "2028-06-01",
        "scope": ["oil_palm"],
    }
    mock.check_health.return_value = True
    return mock


@pytest.fixture
def mock_pefc_api():
    """Mock PEFC API responses."""
    mock = MagicMock()
    mock.validate_certificate.return_value = {
        "certificate_code": "PEFC/XX-XX-XXXXXXX",
        "valid": True,
        "status": "valid",
    }
    mock.check_health.return_value = True
    return mock


@pytest.fixture
def mock_iscc_api():
    """Mock ISCC API responses."""
    mock = MagicMock()
    mock.validate_certificate.return_value = {
        "certificate_code": "ISCC-EU-2024-001",
        "valid": True,
        "status": "valid",
    }
    mock.check_health.return_value = True
    return mock


# ---------------------------------------------------------------------------
# Engine Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def framework_engine(mock_config):
    """Create a mock LegalFrameworkDatabaseEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.get_country_framework = MagicMock(return_value={})
    engine.search_legislation = MagicMock(return_value=[])
    engine.get_category_coverage = MagicMock(return_value={})
    engine.sync_external_sources = AsyncMock(return_value={"synced": 0})
    engine.calculate_reliability_score = MagicMock(return_value=Decimal("75"))
    return engine


@pytest.fixture
def doc_engine(mock_config):
    """Create a mock DocumentVerificationEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.verify_document = MagicMock(return_value={"verified": True})
    engine.verify_batch = MagicMock(return_value=[])
    engine.check_validity = MagicMock(return_value=True)
    engine.check_expiry = MagicMock(return_value={"expired": False})
    engine.validate_authority = MagicMock(return_value=True)
    engine.calculate_verification_score = MagicMock(return_value=Decimal("85"))
    return engine


@pytest.fixture
def cert_engine(mock_config):
    """Create a mock CertificationSchemeValidatorEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.validate_certificate = MagicMock(return_value={"valid": True})
    engine.check_eudr_equivalence = MagicMock(return_value=True)
    engine.validate_scope = MagicMock(return_value=True)
    engine.check_expiry = MagicMock(return_value={"expired": False})
    return engine


@pytest.fixture
def red_flag_engine(mock_config):
    """Create a mock RedFlagDetectionEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.detect_flags = MagicMock(return_value=[])
    engine.score_severity = MagicMock(return_value=Decimal("0"))
    engine.classify_severity = MagicMock(return_value="low")
    engine.apply_multipliers = MagicMock(return_value=Decimal("0"))
    engine.check_false_positive = MagicMock(return_value=False)
    return engine


@pytest.fixture
def country_engine(mock_config):
    """Create a mock CountryComplianceCheckerEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.check_compliance = MagicMock(return_value={})
    engine.get_country_rules = MagicMock(return_value={})
    engine.generate_gap_analysis = MagicMock(return_value={})
    engine.calculate_evidence_sufficiency = MagicMock(return_value=Decimal("70"))
    return engine


@pytest.fixture
def audit_engine(mock_config):
    """Create a mock ThirdPartyAuditEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.parse_report = MagicMock(return_value={})
    engine.extract_findings = MagicMock(return_value=[])
    engine.track_corrective_actions = MagicMock(return_value=[])
    engine.validate_auditor = MagicMock(return_value=True)
    return engine


@pytest.fixture
def report_engine(mock_config):
    """Create a mock ComplianceReportingEngine."""
    set_config(mock_config)
    engine = MagicMock()
    engine.config = mock_config
    engine.generate_report = MagicMock(return_value={})
    engine.export_format = MagicMock(return_value=b"")
    engine.translate = MagicMock(return_value={})
    engine.compute_provenance = MagicMock(
        return_value=compute_test_hash({"report": "test"})
    )
    return engine


# ---------------------------------------------------------------------------
# Mock Provenance Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provenance():
    """Create a mock ProvenanceTracker for testing.

    Provides a MagicMock that simulates provenance chain operations
    including record(), get_hash(), build_hash(), and verify_chain().
    """
    tracker = MagicMock()
    tracker.genesis_hash = compute_test_hash({"genesis": "GL-EUDR-LCV-023"})
    tracker._entries = []
    tracker.entry_count = 0

    def record_side_effect(entity_type, action, entity_id, data=None, metadata=None):
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "hash_value": compute_test_hash({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
            }),
            "parent_hash": (
                tracker.genesis_hash
                if not tracker._entries
                else tracker._entries[-1]["hash_value"]
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        tracker._entries.append(entry)
        tracker.entry_count = len(tracker._entries)
        return entry

    tracker.record = MagicMock(side_effect=record_side_effect)
    tracker.verify_chain = MagicMock(return_value=True)
    tracker.get_entries = MagicMock(return_value=[])
    tracker.build_hash = MagicMock(side_effect=lambda d: compute_test_hash(d))
    tracker.clear = MagicMock()
    return tracker


# ---------------------------------------------------------------------------
# Mock Metrics Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metrics():
    """Create a mock MetricsCollector for testing."""
    metrics = MagicMock()
    metrics.increment = MagicMock()
    metrics.observe = MagicMock()
    metrics.set_gauge = MagicMock()
    metrics.start_timer = MagicMock(return_value=MagicMock())
    metrics.labels = MagicMock(return_value=metrics)
    return metrics


# ---------------------------------------------------------------------------
# Mock Database / Connection Pool Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_pool():
    """Create a mock async database connection pool."""
    pool = AsyncMock()
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.rowcount = 0
    conn.cursor = MagicMock(return_value=cursor)
    conn.execute = AsyncMock()
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    pool.acquire = AsyncMock(return_value=conn)
    pool.release = AsyncMock()
    pool.getconn = MagicMock(return_value=conn)
    pool.putconn = MagicMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for caching tests."""
    redis = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.set = MagicMock(return_value=True)
    redis.delete = MagicMock(return_value=1)
    redis.exists = MagicMock(return_value=False)
    redis.expire = MagicMock(return_value=True)
    redis.keys = MagicMock(return_value=[])
    redis.pipeline = MagicMock(return_value=redis)
    redis.execute = MagicMock(return_value=[])
    return redis


# ---------------------------------------------------------------------------
# Authentication Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_auth_token():
    """Create a mock JWT token for authenticated requests."""
    return "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test-token"


@pytest.fixture
def mock_auth_headers():
    """Create mock authentication headers for API testing."""
    return {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test-token",
        "X-Tenant-ID": "tenant-001",
        "X-Request-ID": "req-test-001",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_admin_headers():
    """Create mock admin authentication headers."""
    return {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.admin-token",
        "X-Tenant-ID": "tenant-001",
        "X-Request-ID": "req-admin-001",
        "X-Role": "admin",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_unauthorized_headers():
    """Create headers missing authorization for negative testing."""
    return {
        "X-Tenant-ID": "tenant-001",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Supplier Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_supplier():
    """Sample supplier data for compliance assessment."""
    return {
        "supplier_id": "SUP-0001",
        "name": "Agro Brasil Ltda",
        "country_code": "BR",
        "commodity": "soya",
        "registration_number": "BR-12345678",
        "coordinates": {"latitude": Decimal("-12.5"), "longitude": Decimal("-55.3")},
        "plot_ids": ["PLOT-BR-001", "PLOT-BR-002"],
        "certifications": ["FSC-C100001"],
        "contact_email": "compliance@agrobrasil.com",
    }


@pytest.fixture
def sample_suppliers():
    """Multiple suppliers across different countries and commodities."""
    return [
        {
            "supplier_id": f"SUP-{i+1:04d}",
            "name": f"Supplier {i+1}",
            "country_code": EUDR_COUNTRIES_27[i % len(EUDR_COUNTRIES_27)],
            "commodity": EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)],
        }
        for i in range(20)
    ]


# ---------------------------------------------------------------------------
# Computation Helpers for Tests
# ---------------------------------------------------------------------------


def compute_compliance_score(category_scores: Dict[str, Decimal]) -> Decimal:
    """Compute overall compliance score as weighted average of 8 categories."""
    if not category_scores:
        return Decimal("0")
    total = sum(category_scores.values())
    return (total / Decimal(str(len(category_scores)))).quantize(Decimal("0.01"))


def determine_compliance(score: Decimal, compliant: int = 80, partial: int = 50) -> str:
    """Determine compliance status from overall score."""
    if score >= compliant:
        return "COMPLIANT"
    elif score >= partial:
        return "PARTIALLY_COMPLIANT"
    else:
        return "NON_COMPLIANT"


def classify_red_flag_severity(score: Decimal, critical: int = 75,
                                high: int = 50, moderate: int = 25) -> str:
    """Classify red flag severity from adjusted score."""
    if score >= critical:
        return "critical"
    elif score >= high:
        return "high"
    elif score >= moderate:
        return "moderate"
    else:
        return "low"


def apply_country_multiplier(base_score: int, country_code: str) -> Decimal:
    """Apply country-specific risk multiplier to base red flag score."""
    multiplier = COUNTRY_MULTIPLIERS.get(country_code, Decimal("1.0"))
    return Decimal(str(base_score)) * multiplier


def apply_commodity_multiplier(score: Decimal, commodity: str) -> Decimal:
    """Apply commodity-specific risk multiplier."""
    multiplier = COMMODITY_MULTIPLIERS.get(commodity, Decimal("1.0"))
    return score * multiplier


def is_document_expired(expiry_date_str: str) -> bool:
    """Check if a document has expired based on its expiry date string."""
    expiry = date.fromisoformat(expiry_date_str)
    return expiry < date.today()


def days_until_expiry(expiry_date_str: str) -> int:
    """Calculate days until document expiry (negative if expired)."""
    expiry = date.fromisoformat(expiry_date_str)
    return (expiry - date.today()).days


def _get_required_docs(country: str, category: str) -> List[str]:
    """Get required document types for a country-category combination."""
    base_docs = {
        "land_use_rights": ["land_title", "concession_permit"],
        "environmental_protection": ["environmental_impact_assessment"],
        "forest_related_rules": ["forest_management_plan", "harvest_permit"],
        "third_party_rights": ["indigenous_consent_document"],
        "labour_rights": ["labour_compliance_certificate"],
        "tax_and_royalty": ["tax_clearance_certificate"],
        "trade_and_customs": ["export_license", "certificate_of_origin"],
        "anti_corruption": ["anti_corruption_declaration"],
    }
    return base_docs.get(category, [])


def _get_special_requirements(country: str, category: str) -> List[str]:
    """Get special requirements for high-risk country-category combinations."""
    if country in HIGH_RISK_COUNTRIES:
        return [f"Enhanced due diligence for {category} in {country}"]
    return []
