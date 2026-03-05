# -*- coding: utf-8 -*-
"""
Unit tests for ISO 14064-1:2018 Platform Domain Models.

Tests all Pydantic v2 domain models including helpers, validation,
computed properties, and provenance hashing with 25+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal

import pytest

from services.config import (
    ConsolidationApproach,
    DataQualityTier,
    FindingSeverity,
    FindingStatus,
    GWPSource,
    ISOCategory,
    PermanenceLevel,
    QuantificationMethod,
    RemovalType,
    ReportFormat,
    ReportingPeriod,
    SignificanceLevel,
    VerificationLevel,
    VerificationStage,
)
from services.models import (
    ApiError,
    ApiResponse,
    BaseYear,
    BiogenicEmissions,
    CategoryBreakdown,
    CategoryResult,
    CrossWalkMapping,
    CrossWalkResult,
    DashboardAlert,
    DashboardMetrics,
    EmissionSource,
    Entity,
    EntityEmissions,
    FacilityEmissions,
    Finding,
    FindingsSummary,
    GHGGasBreakdown,
    ISOInventory,
    InventoryBoundary,
    ManagementPlan,
    ImprovementAction,
    MandatoryElement,
    NetEmissionsResult,
    Organization,
    PaginatedResponse,
    Recalculation,
    RemovalSource,
    Report,
    ReportSection,
    SetTargetRequest,
    SignificanceAssessment,
    UncertaintyResult,
    VerificationRecord,
    _new_id,
    _now,
    _sha256,
)


class TestHelpers:
    """Test utility functions."""

    def test_new_id_returns_string(self):
        result = _new_id()
        assert isinstance(result, str)
        assert len(result) == 36  # UUID4 format

    def test_new_id_unique(self):
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_now_returns_datetime(self):
        result = _now()
        assert isinstance(result, datetime)

    def test_now_no_microseconds(self):
        result = _now()
        assert result.microsecond == 0

    def test_sha256_deterministic(self):
        h1 = _sha256("test_payload")
        h2 = _sha256("test_payload")
        assert h1 == h2

    def test_sha256_length_64(self):
        result = _sha256("some data")
        assert len(result) == 64

    def test_sha256_different_inputs(self):
        h1 = _sha256("input_a")
        h2 = _sha256("input_b")
        assert h1 != h2


class TestOrganization:
    """Test Organization model."""

    def test_create_organization(self):
        org = Organization(name="Test Corp", industry="tech", country="US")
        assert org.name == "Test Corp"
        assert org.industry == "tech"
        assert org.country == "US"
        assert len(org.id) == 36

    def test_organization_default_entities_empty(self):
        org = Organization(name="Test", industry="tech", country="US")
        assert org.entities == []

    def test_organization_timestamps(self):
        org = Organization(name="Test", industry="tech", country="US")
        assert isinstance(org.created_at, datetime)
        assert isinstance(org.updated_at, datetime)


class TestEntity:
    """Test Entity model."""

    def test_create_entity(self):
        entity = Entity(name="Factory A", country="DE")
        assert entity.name == "Factory A"
        assert entity.entity_type == "facility"
        assert entity.ownership_pct == Decimal("100.0")

    def test_entity_with_partial_ownership(self):
        entity = Entity(name="JV Plant", country="CN", ownership_pct=Decimal("51.0"))
        assert entity.ownership_pct == Decimal("51.0")


class TestGHGGasBreakdown:
    """Test GHGGasBreakdown model."""

    def test_auto_compute_total(self):
        breakdown = GHGGasBreakdown(
            co2=Decimal("100"), ch4=Decimal("50"), n2o=Decimal("25"),
        )
        assert breakdown.total_co2e == Decimal("175")

    def test_explicit_total_preserved(self):
        breakdown = GHGGasBreakdown(
            co2=Decimal("100"), total_co2e=Decimal("999"),
        )
        assert breakdown.total_co2e == Decimal("999")

    def test_all_zeros(self):
        breakdown = GHGGasBreakdown()
        assert breakdown.total_co2e == Decimal("0")


class TestEmissionSource:
    """Test EmissionSource model."""

    def test_provenance_hash_auto_generated(self):
        source = EmissionSource(
            name="Boiler NG",
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            total_tco2e=Decimal("100"),
        )
        assert len(source.provenance_hash) == 64

    def test_same_input_same_hash(self):
        kwargs = dict(
            name="Boiler",
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            activity_data=Decimal("1000"),
            emission_factor=Decimal("2.5"),
            total_tco2e=Decimal("2500"),
        )
        s1 = EmissionSource(**kwargs)
        s2 = EmissionSource(**kwargs)
        assert s1.provenance_hash == s2.provenance_hash


class TestCategoryResult:
    """Test CategoryResult model."""

    def test_create_category_result(self):
        cr = CategoryResult(
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            total_tco2e=Decimal("5000"),
        )
        assert cr.total_tco2e == Decimal("5000")
        assert cr.data_quality_tier == DataQualityTier.TIER_1

    def test_provenance_hash_generated(self):
        cr = CategoryResult(
            iso_category=ISOCategory.CATEGORY_2_ENERGY,
            total_tco2e=Decimal("3000"),
        )
        assert len(cr.provenance_hash) == 64


class TestRemovalSource:
    """Test RemovalSource model."""

    def test_create_removal_source(self):
        rs = RemovalSource(
            name="Reforestation Project",
            removal_type=RemovalType.FORESTRY,
            quantity_tco2e=Decimal("500"),
        )
        assert rs.removal_type == RemovalType.FORESTRY
        assert rs.permanence == PermanenceLevel.LONG_TERM

    def test_provenance_hash_generated(self):
        rs = RemovalSource(
            name="DAC Project",
            removal_type=RemovalType.DIRECT_AIR_CAPTURE,
            quantity_tco2e=Decimal("1000"),
            permanence=PermanenceLevel.PERMANENT,
        )
        assert len(rs.provenance_hash) == 64


class TestNetEmissionsResult:
    """Test NetEmissionsResult model."""

    def test_net_computed_automatically(self):
        result = NetEmissionsResult(
            gross_emissions=Decimal("10000"),
            total_removals=Decimal("2000"),
        )
        assert result.net_emissions == Decimal("8000")

    def test_net_negative_possible(self):
        result = NetEmissionsResult(
            gross_emissions=Decimal("1000"),
            total_removals=Decimal("3000"),
        )
        assert result.net_emissions == Decimal("-2000")


class TestBaseYear:
    """Test BaseYear model."""

    def test_current_total_no_recalculations(self):
        by = BaseYear(
            org_id="org-1",
            year=2020,
            total_emissions=Decimal("10000"),
            justification="Representative year for emissions",
        )
        assert by.current_total == Decimal("10000")

    def test_current_total_with_recalculation(self):
        recalc = Recalculation(
            trigger="structural_change",
            original_total=Decimal("10000"),
            new_total=Decimal("12000"),
            reason="Acquisition of subsidiary increased scope",
        )
        by = BaseYear(
            org_id="org-1",
            year=2020,
            total_emissions=Decimal("10000"),
            justification="Representative year for emissions",
            recalculations=[recalc],
        )
        assert by.current_total == Decimal("12000")


class TestISOInventory:
    """Test ISOInventory model."""

    def test_recalculate_totals(self):
        inv = ISOInventory(
            org_id="org-1",
            year=2025,
            category_1=CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
            ),
            category_2=CategoryResult(
                iso_category=ISOCategory.CATEGORY_2_ENERGY,
                total_tco2e=Decimal("3000"),
            ),
        )
        assert inv.gross_emissions_tco2e == Decimal("8000")

    def test_provenance_hash_auto(self):
        inv = ISOInventory(org_id="org-1", year=2025)
        assert len(inv.provenance_hash) == 64

    def test_removals_subtracted_from_net(self):
        inv = ISOInventory(
            org_id="org-1",
            year=2025,
            category_1=CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("10000"),
            ),
            removal_sources=[
                RemovalSource(
                    name="Forest",
                    removal_type=RemovalType.FORESTRY,
                    quantity_tco2e=Decimal("2000"),
                ),
            ],
        )
        assert inv.net_emissions_tco2e == Decimal("8000")


class TestVerificationRecord:
    """Test VerificationRecord model."""

    def test_open_findings_count(self):
        record = VerificationRecord(
            inventory_id="inv-1",
            findings=[
                Finding(description="Issue found in Category 1 calculation", status=FindingStatus.OPEN),
                Finding(description="Documentation gap in methodology section", status=FindingStatus.RESOLVED),
                Finding(description="Data quality concern in Category 3", status=FindingStatus.IN_PROGRESS),
            ],
        )
        assert record.open_findings_count == 2

    def test_has_critical_findings(self):
        record = VerificationRecord(
            inventory_id="inv-1",
            findings=[
                Finding(
                    description="Critical error in Scope 1 emissions calculation",
                    severity=FindingSeverity.CRITICAL,
                    status=FindingStatus.OPEN,
                ),
            ],
        )
        assert record.has_critical_findings is True


class TestPaginatedResponse:
    """Test PaginatedResponse model."""

    def test_pagination_computed(self):
        resp = PaginatedResponse(
            items=list(range(10)),
            total=100,
            page=2,
            page_size=10,
        )
        assert resp.total_pages == 10
        assert resp.has_next is True
        assert resp.has_previous is True


class TestSetTargetRequest:
    """Test SetTargetRequest validation."""

    def test_target_year_after_base_year(self):
        with pytest.raises(Exception):
            SetTargetRequest(
                base_year=2025,
                target_year=2020,
                base_year_emissions=Decimal("10000"),
                reduction_pct=Decimal("50"),
            )
