# -*- coding: utf-8 -*-
"""
Unit tests for Information Gathering Agent Models - AGENT-EUDR-027

Tests all 10 enums, 16 Pydantic models, and model constants including
ARTICLE_9_ELEMENTS, ARTICLE_9_DEFAULT_WEIGHTS, SUPPORTED_COMMODITIES,
CERTIFICATION_COMMODITY_MAP, FLEGT_VPA_COUNTRIES, and LOW_RISK_COUNTRIES.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.information_gathering.models import (
    ARTICLE_9_DEFAULT_WEIGHTS,
    ARTICLE_9_ELEMENTS,
    CERTIFICATION_COMMODITY_MAP,
    FLEGT_VPA_COUNTRIES,
    LOW_RISK_COUNTRIES,
    SUPPORTED_COMMODITIES,
    Article9ElementName,
    Article9ElementStatus,
    CertificateVerificationResult,
    CertificationBody,
    CertVerificationStatus,
    CompletenessClassification,
    CompletenessReport,
    DataDiscrepancy,
    DataFreshnessRecord,
    DataSourcePriority,
    ElementStatus,
    EUDRCommodity,
    EvidenceArtifact,
    ExternalDatabaseSource,
    FreshnessStatus,
    GapReport,
    GapReportItem,
    GatheringOperation,
    GatheringOperationStatus,
    HarvestResult,
    InformationPackage,
    NormalizationRecord,
    NormalizationType,
    PackageDiff,
    ProvenanceEntry,
    QueryResult,
    QueryStatus,
    SupplierProfile,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestExternalDatabaseSourceEnum:
    def test_values_count(self):
        assert len(ExternalDatabaseSource) == 11

    def test_contains_eu_traces(self):
        assert ExternalDatabaseSource.EU_TRACES.value == "eu_traces"

    def test_contains_national_land_registry(self):
        assert ExternalDatabaseSource.NATIONAL_LAND_REGISTRY.value == "national_land_registry"


class TestCertificationBodyEnum:
    def test_values_count(self):
        assert len(CertificationBody) == 6

    def test_contains_fsc(self):
        assert CertificationBody.FSC.value == "fsc"

    def test_contains_eu_organic(self):
        assert CertificationBody.EU_ORGANIC.value == "eu_organic"


class TestEUDRCommodityEnum:
    def test_values_count(self):
        assert len(EUDRCommodity) == 7

    def test_contains_all_commodities(self):
        expected = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        actual = {c.value for c in EUDRCommodity}
        assert actual == expected


class TestQueryStatusEnum:
    def test_values_count(self):
        assert len(QueryStatus) == 5

    def test_values(self):
        expected = {"success", "partial", "failed", "cached", "timeout"}
        actual = {s.value for s in QueryStatus}
        assert actual == expected


class TestCertVerificationStatusEnum:
    def test_values_count(self):
        assert len(CertVerificationStatus) == 6

    def test_values(self):
        expected = {"valid", "expired", "suspended", "withdrawn", "not_found", "error"}
        actual = {s.value for s in CertVerificationStatus}
        assert actual == expected


class TestCompletenessClassificationEnum:
    def test_values_count(self):
        assert len(CompletenessClassification) == 3

    def test_values(self):
        expected = {"insufficient", "partial", "complete"}
        actual = {c.value for c in CompletenessClassification}
        assert actual == expected


class TestNormalizationTypeEnum:
    def test_values_count(self):
        assert len(NormalizationType) == 8

    def test_values(self):
        expected = {
            "unit", "date", "coordinate", "currency",
            "country_code", "product_code", "address", "certificate_id",
        }
        actual = {t.value for t in NormalizationType}
        assert actual == expected


class TestGatheringOperationStatusEnum:
    def test_values_count(self):
        assert len(GatheringOperationStatus) == 5

    def test_values(self):
        expected = {"initiated", "in_progress", "completed", "failed", "cancelled"}
        actual = {s.value for s in GatheringOperationStatus}
        assert actual == expected


class TestFreshnessStatusEnum:
    def test_values_count(self):
        assert len(FreshnessStatus) == 4

    def test_values(self):
        expected = {"fresh", "stale", "expired", "unknown"}
        actual = {s.value for s in FreshnessStatus}
        assert actual == expected


class TestDataSourcePriorityEnum:
    def test_values_count(self):
        assert len(DataSourcePriority) == 6

    def test_values(self):
        expected = {
            "government_registry", "certification_body", "customs_record",
            "trade_database", "supplier_self_declared", "public_database",
        }
        actual = {p.value for p in DataSourcePriority}
        assert actual == expected


class TestArticle9ElementNameEnum:
    def test_values_count(self):
        assert len(Article9ElementName) == 10

    def test_values(self):
        expected = {
            "product_description", "quantity", "country_of_production",
            "geolocation", "production_date_range", "supplier_identification",
            "buyer_identification", "deforestation_free_evidence",
            "legal_production_evidence", "supply_chain_information",
        }
        actual = {e.value for e in Article9ElementName}
        assert actual == expected


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


class TestQueryResultModel:
    def test_defaults(self):
        qr = QueryResult(
            query_id="qry_test_001",
            source=ExternalDatabaseSource.EU_TRACES,
        )
        assert qr.status == QueryStatus.SUCCESS
        assert qr.records == []
        assert qr.record_count == 0
        assert qr.cached is False
        assert qr.cache_age_seconds is None
        assert qr.provenance_hash == ""

    def test_populated_fields(self, sample_query_result):
        assert sample_query_result.source == ExternalDatabaseSource.EU_TRACES
        assert sample_query_result.status == QueryStatus.SUCCESS
        assert sample_query_result.record_count == 1


class TestCertificateVerificationResultModel:
    def test_defaults(self):
        cvr = CertificateVerificationResult(
            certificate_id="TEST-001",
            certification_body=CertificationBody.FSC,
        )
        assert cvr.holder_name == ""
        assert cvr.verification_status == CertVerificationStatus.NOT_FOUND
        assert cvr.valid_from is None
        assert cvr.valid_until is None
        assert cvr.scope == []
        assert cvr.commodity_scope == []
        assert cvr.chain_of_custody_model is None
        assert cvr.days_until_expiry is None

    def test_populated_fields(self, sample_cert_result):
        assert sample_cert_result.verification_status == CertVerificationStatus.VALID
        assert sample_cert_result.days_until_expiry == 730


class TestSupplierProfileModel:
    def test_defaults(self):
        sp = SupplierProfile(supplier_id="SUP-TEST", name="Test Supplier")
        assert sp.alternative_names == []
        assert sp.postal_address == ""
        assert sp.country_code == ""
        assert sp.email is None
        assert sp.registration_number is None
        assert sp.commodities == []
        assert sp.certifications == []
        assert sp.plot_ids == []
        assert sp.tier_depth == 0
        assert sp.data_sources == []
        assert sp.completeness_score == Decimal("0")
        assert sp.confidence_score == Decimal("0")
        assert sp.discrepancies == []

    def test_populated_fields(self, sample_supplier_profile):
        assert sample_supplier_profile.completeness_score == Decimal("85.00")
        assert len(sample_supplier_profile.commodities) == 1


class TestCompletenessReportModel:
    def test_defaults(self):
        cr = CompletenessReport(
            operation_id="op_test",
            commodity=EUDRCommodity.COFFEE,
        )
        assert cr.completeness_score == Decimal("0")
        assert cr.completeness_classification == CompletenessClassification.INSUFFICIENT
        assert cr.elements == []
        assert cr.is_simplified_dd is False

    def test_with_score(self):
        cr = CompletenessReport(
            operation_id="op_test",
            commodity=EUDRCommodity.COCOA,
            completeness_score=Decimal("95.50"),
            completeness_classification=CompletenessClassification.COMPLETE,
        )
        assert cr.completeness_score == Decimal("95.50")


class TestInformationPackageModel:
    def test_defaults(self):
        ip = InformationPackage(
            package_id="pkg_001",
            operator_id="OP-DE-001",
            commodity=EUDRCommodity.WOOD,
        )
        assert ip.version == 1
        assert ip.article_9_elements == {}
        assert ip.completeness_score == Decimal("0")
        assert ip.completeness_classification == "insufficient"
        assert ip.supplier_profiles == []
        assert ip.external_data == {}
        assert ip.certification_results == []
        assert ip.package_hash == ""
        assert ip.valid_until is None


class TestGatheringOperationModel:
    def test_defaults(self):
        go = GatheringOperation(
            operation_id="op_001",
            operator_id="OP-DE-001",
            commodity=EUDRCommodity.COFFEE,
        )
        assert go.status == GatheringOperationStatus.INITIATED
        assert go.sources_queried == []
        assert go.sources_completed == []
        assert go.sources_failed == []
        assert go.completeness_score == Decimal("0")
        assert go.total_records_collected == 0
        assert go.package_id is None
        assert go.completed_at is None
        assert go.duration_ms is None


class TestGapReportItemModel:
    def test_creation(self):
        item = GapReportItem(
            element_name="geolocation",
            gap_type="missing",
            severity="critical",
            remediation_action="Collect GPS coordinates",
            estimated_effort="3-7 days",
        )
        assert item.element_name == "geolocation"
        assert item.severity == "critical"


class TestNormalizationRecordModel:
    def test_creation(self):
        nr = NormalizationRecord(
            field_name="country",
            source_value="Brazil",
            normalized_value="BRA",
            normalization_type=NormalizationType.COUNTRY_CODE,
            confidence=Decimal("1.0"),
        )
        assert nr.normalization_type == NormalizationType.COUNTRY_CODE
        assert nr.confidence == Decimal("1.0")


class TestEvidenceArtifactModel:
    def test_creation(self):
        ea = EvidenceArtifact(
            artifact_id="art_001",
            article_9_element="geolocation",
            source="satellite_imagery",
            format="json",
            content_hash="abcd1234" * 8,
        )
        assert ea.format == "json"
        assert ea.s3_path is None


class TestProvenanceEntryModel:
    def test_creation(self):
        pe = ProvenanceEntry(
            step="collect",
            source="eu_traces",
            actor="AGENT-EUDR-027",
            input_hash="a" * 64,
            output_hash="b" * 64,
        )
        assert pe.actor == "AGENT-EUDR-027"


class TestPackageDiffModel:
    def test_creation(self):
        pd = PackageDiff(
            package_a_id="pkg_001",
            package_b_id="pkg_002",
            added_elements=["geolocation"],
            changed_elements=["quantity"],
            score_delta=Decimal("15.5"),
        )
        assert pd.score_delta == Decimal("15.5")
        assert pd.removed_elements == []


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestModelConstants:
    def test_article_9_elements_count(self):
        assert len(ARTICLE_9_ELEMENTS) == 10

    def test_article_9_default_weights_sum(self):
        total = sum(ARTICLE_9_DEFAULT_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_article_9_default_weights_count(self):
        assert len(ARTICLE_9_DEFAULT_WEIGHTS) == 10

    def test_supported_commodities_count(self):
        assert len(SUPPORTED_COMMODITIES) == 7

    def test_certification_commodity_map_keys(self):
        expected_keys = {"fsc", "rspo", "pefc", "rainforest_alliance", "utz", "eu_organic"}
        actual_keys = set(CERTIFICATION_COMMODITY_MAP.keys())
        assert actual_keys == expected_keys

    def test_certification_commodity_map_fsc_has_wood(self):
        assert "wood" in CERTIFICATION_COMMODITY_MAP["fsc"]

    def test_certification_commodity_map_eu_organic_covers_all(self):
        assert len(CERTIFICATION_COMMODITY_MAP["eu_organic"]) == 7

    def test_flegt_vpa_countries_not_empty(self):
        assert len(FLEGT_VPA_COUNTRIES) > 0
        assert "ID" in FLEGT_VPA_COUNTRIES  # Indonesia

    def test_low_risk_countries_initially_empty(self):
        assert LOW_RISK_COUNTRIES == []


class TestDecimalPrecisionInModels:
    def test_supplier_profile_decimal_scores(self):
        sp = SupplierProfile(
            supplier_id="SUP-X",
            name="Test",
            completeness_score=Decimal("99.99"),
            confidence_score=Decimal("87.50"),
        )
        assert isinstance(sp.completeness_score, Decimal)
        assert isinstance(sp.confidence_score, Decimal)

    def test_article9_element_status_decimal_confidence(self):
        status = Article9ElementStatus(
            element_name="quantity",
            status=ElementStatus.COMPLETE,
            confidence=Decimal("0.9876"),
        )
        assert isinstance(status.confidence, Decimal)

    def test_completeness_report_decimal_score(self):
        cr = CompletenessReport(
            operation_id="test",
            commodity=EUDRCommodity.SOYA,
            completeness_score=Decimal("75.25"),
        )
        assert isinstance(cr.completeness_score, Decimal)

    def test_package_diff_decimal_delta(self):
        pd = PackageDiff(
            package_a_id="a",
            package_b_id="b",
            score_delta=Decimal("-5.50"),
        )
        assert isinstance(pd.score_delta, Decimal)
