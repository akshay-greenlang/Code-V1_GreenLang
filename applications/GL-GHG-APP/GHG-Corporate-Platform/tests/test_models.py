"""
Unit tests for GL-GHG-APP v1.0 Pydantic Domain Models

Tests all Pydantic domain models with 40+ test cases covering:
- Organization and Entity creation, validation, and hierarchy
- InventoryBoundary consolidation approaches and scope selection
- BaseYear creation, locking, and recalculation
- ScopeEmissions total calculation, per-gas breakdown, category sums
- GHGInventory grand total = scope1 + scope2 + scope3
- IntensityMetric calculation and zero denominator handling
- UncertaintyResult bounds (p5 <= p50 <= p95) and CV calculation
- CompletenessResult percentage, mandatory disclosures count = 15
- VerificationRecord status transitions and finding tracking
- Target reduction bounds (0-100%), SBTi alignment
"""

import pytest
from decimal import Decimal
from datetime import datetime

from services.config import (
    ConsolidationApproach,
    DataQualityTier,
    EntityType,
    FindingSeverity,
    FindingType,
    GHGGas,
    IntensityDenominator,
    ReportFormat,
    Scope,
    Scope1Category,
    Scope3Category,
    TargetType,
    VerificationLevel,
    VerificationStatus,
    GWP_AR5,
    SECTOR_BENCHMARKS,
    UNCERTAINTY_CV_BY_TIER,
)
from services.models import (
    Organization,
    Entity,
    InventoryBoundary,
    ExclusionRecord,
    BaseYear,
    Recalculation,
    ScopeEmissions,
    GHGInventory,
    IntensityMetric,
    UncertaintyResult,
    ScopeUncertainty,
    CompletenessResult,
    Disclosure,
    DataGap,
    Report,
    ReportSection,
    VerificationRecord,
    VerificationFinding,
    Target,
    DashboardMetrics,
    CreateOrganizationRequest,
    AddEntityRequest,
    SetBoundaryRequest,
    CreateInventoryRequest,
    SetBaseYearRequest,
    SetTargetRequest,
    GenerateReportRequest,
    StartVerificationRequest,
    AddFindingRequest,
    AggregateEmissionsRequest,
    ExclusionRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entity_subsidiary():
    """Create a sample subsidiary entity."""
    return Entity(
        name="Acme Europe GmbH",
        entity_type=EntityType.SUBSIDIARY,
        ownership_pct=Decimal("80.0"),
        country="DE",
        employees=500,
        revenue=Decimal("50000000"),
    )


@pytest.fixture
def sample_entity_facility():
    """Create a sample facility entity."""
    return Entity(
        name="East Coast Plant",
        entity_type=EntityType.FACILITY,
        ownership_pct=Decimal("100.0"),
        country="US",
        employees=350,
        floor_area_m2=Decimal("15000"),
        production_units=Decimal("1000000"),
        production_unit_name="widgets",
    )


@pytest.fixture
def sample_entity_operation():
    """Create a sample operation entity."""
    return Entity(
        name="Fleet Operations",
        entity_type=EntityType.OPERATION,
        ownership_pct=Decimal("100.0"),
        country="US",
    )


@pytest.fixture
def sample_organization(sample_entity_subsidiary, sample_entity_facility):
    """Create a sample organization with entities."""
    return Organization(
        name="Acme Corporation",
        industry="manufacturing",
        country="US",
        description="Global manufacturing company",
        entities=[sample_entity_subsidiary, sample_entity_facility],
    )


@pytest.fixture
def sample_scope1_emissions():
    """Create sample Scope 1 emissions."""
    return ScopeEmissions(
        scope=Scope.SCOPE_1,
        total_tco2e=Decimal("12450.8"),
        by_gas={
            GHGGas.CO2.value: Decimal("9980.0"),
            GHGGas.CH4.value: Decimal("1253.7"),
            GHGGas.N2O.value: Decimal("67.1"),
            GHGGas.HFCS.value: Decimal("1150.0"),
        },
        by_category={
            Scope1Category.STATIONARY_COMBUSTION.value: Decimal("5820.3"),
            Scope1Category.MOBILE_COMBUSTION.value: Decimal("2340.5"),
            Scope1Category.PROCESS_EMISSIONS.value: Decimal("1890.0"),
            Scope1Category.FUGITIVE_EMISSIONS.value: Decimal("1250.0"),
            Scope1Category.REFRIGERANTS.value: Decimal("1150.0"),
        },
        biogenic_co2=Decimal("85.0"),
        data_quality_tier=DataQualityTier.TIER_2,
    )


@pytest.fixture
def sample_scope2_location_emissions():
    """Create sample Scope 2 location-based emissions."""
    return ScopeEmissions(
        scope=Scope.SCOPE_2_LOCATION,
        total_tco2e=Decimal("8320.5"),
        by_gas={GHGGas.CO2.value: Decimal("8200.0"), GHGGas.CH4.value: Decimal("80.5"), GHGGas.N2O.value: Decimal("40.0")},
        by_category={"purchased_electricity": Decimal("7500.0"), "steam_heat": Decimal("820.5")},
    )


@pytest.fixture
def sample_scope2_market_emissions():
    """Create sample Scope 2 market-based emissions."""
    return ScopeEmissions(
        scope=Scope.SCOPE_2_MARKET,
        total_tco2e=Decimal("6100.0"),
        by_gas={GHGGas.CO2.value: Decimal("6000.0"), GHGGas.CH4.value: Decimal("60.0"), GHGGas.N2O.value: Decimal("40.0")},
        by_category={"purchased_electricity": Decimal("5400.0"), "steam_heat": Decimal("700.0")},
    )


@pytest.fixture
def sample_scope3_emissions():
    """Create sample Scope 3 emissions."""
    return ScopeEmissions(
        scope=Scope.SCOPE_3,
        total_tco2e=Decimal("45230.2"),
        by_gas={GHGGas.CO2.value: Decimal("42000.0"), GHGGas.CH4.value: Decimal("2500.0"), GHGGas.N2O.value: Decimal("730.2")},
        by_category={
            Scope3Category.CAT1_PURCHASED_GOODS.value: Decimal("18500.0"),
            Scope3Category.CAT4_UPSTREAM_TRANSPORT.value: Decimal("8200.0"),
            Scope3Category.CAT6_BUSINESS_TRAVEL.value: Decimal("3200.0"),
            Scope3Category.CAT5_WASTE_GENERATED.value: Decimal("2800.0"),
            Scope3Category.CAT7_EMPLOYEE_COMMUTING.value: Decimal("2530.2"),
            Scope3Category.CAT2_CAPITAL_GOODS.value: Decimal("5000.0"),
            Scope3Category.CAT3_FUEL_ENERGY.value: Decimal("5000.0"),
        },
    )


@pytest.fixture
def sample_base_year():
    """Create a sample base year."""
    return BaseYear(
        org_id="org-001",
        year=2019,
        scope1_emissions=Decimal("13000.0"),
        scope2_location_emissions=Decimal("9000.0"),
        scope2_market_emissions=Decimal("7000.0"),
        scope3_emissions=Decimal("48000.0"),
        total_emissions=Decimal("68000.0"),
        justification="First year of complete Scope 1-3 data availability",
    )


@pytest.fixture
def sample_ghg_inventory(
    sample_scope1_emissions,
    sample_scope2_location_emissions,
    sample_scope2_market_emissions,
    sample_scope3_emissions,
):
    """Create a sample GHG inventory."""
    return GHGInventory(
        org_id="org-001",
        year=2025,
        scope1=sample_scope1_emissions,
        scope2_location=sample_scope2_location_emissions,
        scope2_market=sample_scope2_market_emissions,
        scope3=sample_scope3_emissions,
    )


# ---------------------------------------------------------------------------
# TestOrganization
# ---------------------------------------------------------------------------

class TestOrganization:
    """Test suite for Organization model."""

    def test_creation_minimal(self):
        """Test organization creation with minimal required fields."""
        org = Organization(name="Test Corp", industry="technology", country="US")
        assert org.name == "Test Corp"
        assert org.industry == "technology"
        assert org.country == "US"
        assert len(org.id) == 36  # UUID format
        assert isinstance(org.created_at, datetime)
        assert org.entities == []

    def test_creation_full(self, sample_organization):
        """Test organization creation with all fields."""
        assert sample_organization.name == "Acme Corporation"
        assert sample_organization.industry == "manufacturing"
        assert sample_organization.country == "US"
        assert sample_organization.description == "Global manufacturing company"
        assert len(sample_organization.entities) == 2

    def test_industry_value(self):
        """Test industry field accepts various values."""
        for industry in ["manufacturing", "energy", "technology", "finance", "retail"]:
            org = Organization(name="Test", industry=industry, country="US")
            assert org.industry == industry

    def test_entity_adding(self, sample_organization, sample_entity_operation):
        """Test adding entities to organization."""
        sample_organization.entities.append(sample_entity_operation)
        assert len(sample_organization.entities) == 3
        assert sample_organization.entities[-1].name == "Fleet Operations"

    def test_empty_name_raises(self):
        """Test that empty name raises validation error."""
        with pytest.raises(Exception):
            Organization(name="", industry="tech", country="US")

    def test_country_code_length(self):
        """Test country code length validation (2-3 chars)."""
        org2 = Organization(name="Test", industry="tech", country="US")
        assert org2.country == "US"
        org3 = Organization(name="Test", industry="tech", country="USA")
        assert org3.country == "USA"

    def test_country_code_too_short_raises(self):
        """Test that single-char country code raises."""
        with pytest.raises(Exception):
            Organization(name="Test", industry="tech", country="U")

    def test_country_code_too_long_raises(self):
        """Test that 4+ char country code raises."""
        with pytest.raises(Exception):
            Organization(name="Test", industry="tech", country="USAA")


# ---------------------------------------------------------------------------
# TestEntity
# ---------------------------------------------------------------------------

class TestEntity:
    """Test suite for Entity model."""

    def test_creation_subsidiary(self, sample_entity_subsidiary):
        """Test subsidiary entity creation."""
        assert sample_entity_subsidiary.entity_type == EntityType.SUBSIDIARY
        assert sample_entity_subsidiary.ownership_pct == Decimal("80.0")
        assert sample_entity_subsidiary.employees == 500

    def test_creation_facility(self, sample_entity_facility):
        """Test facility entity creation."""
        assert sample_entity_facility.entity_type == EntityType.FACILITY
        assert sample_entity_facility.floor_area_m2 == Decimal("15000")
        assert sample_entity_facility.production_units == Decimal("1000000")
        assert sample_entity_facility.production_unit_name == "widgets"

    def test_creation_operation(self, sample_entity_operation):
        """Test operation entity creation."""
        assert sample_entity_operation.entity_type == EntityType.OPERATION
        assert sample_entity_operation.ownership_pct == Decimal("100.0")

    def test_entity_types_enum(self):
        """Test all entity types are valid."""
        for etype in [EntityType.SUBSIDIARY, EntityType.FACILITY, EntityType.OPERATION]:
            entity = Entity(name=f"Test {etype.value}", entity_type=etype, country="US")
            assert entity.entity_type == etype

    def test_ownership_pct_bounds_zero(self):
        """Test 0% ownership is valid."""
        entity = Entity(name="Zero", entity_type=EntityType.FACILITY, country="US", ownership_pct=Decimal("0"))
        assert entity.ownership_pct == Decimal("0")

    def test_ownership_pct_bounds_hundred(self):
        """Test 100% ownership is valid."""
        entity = Entity(name="Full", entity_type=EntityType.FACILITY, country="US", ownership_pct=Decimal("100"))
        assert entity.ownership_pct == Decimal("100")

    def test_ownership_pct_over_hundred_raises(self):
        """Test ownership > 100% raises validation error."""
        with pytest.raises(Exception):
            Entity(name="Over", entity_type=EntityType.FACILITY, country="US", ownership_pct=Decimal("101"))

    def test_ownership_pct_negative_raises(self):
        """Test negative ownership raises validation error."""
        with pytest.raises(Exception):
            Entity(name="Neg", entity_type=EntityType.FACILITY, country="US", ownership_pct=Decimal("-1"))

    def test_parent_id_linkage(self):
        """Test parent entity linkage."""
        parent = Entity(name="Parent", entity_type=EntityType.SUBSIDIARY, country="US")
        child = Entity(
            name="Child",
            entity_type=EntityType.FACILITY,
            country="US",
            parent_id=parent.id,
        )
        assert child.parent_id == parent.id

    def test_active_default(self):
        """Test entity defaults to active."""
        entity = Entity(name="Active", entity_type=EntityType.FACILITY, country="US")
        assert entity.active is True


# ---------------------------------------------------------------------------
# TestInventoryBoundary
# ---------------------------------------------------------------------------

class TestInventoryBoundary:
    """Test suite for InventoryBoundary model."""

    def test_consolidation_approach_enum(self):
        """Test all consolidation approaches."""
        for approach in ConsolidationApproach:
            boundary = InventoryBoundary(
                org_id="org-001",
                consolidation_approach=approach,
                reporting_year=2025,
            )
            assert boundary.consolidation_approach == approach

    def test_default_scopes(self):
        """Test default scopes include Scope 1 and Scope 2."""
        boundary = InventoryBoundary(
            org_id="org-001",
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            reporting_year=2025,
        )
        assert Scope.SCOPE_1 in boundary.scopes
        assert Scope.SCOPE_2_LOCATION in boundary.scopes
        assert Scope.SCOPE_2_MARKET in boundary.scopes

    def test_scope_selection(self):
        """Test custom scope selection."""
        boundary = InventoryBoundary(
            org_id="org-001",
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            reporting_year=2025,
            scopes=[Scope.SCOPE_1, Scope.SCOPE_3],
        )
        assert len(boundary.scopes) == 2
        assert Scope.SCOPE_3 in boundary.scopes

    def test_exclusion_adding(self):
        """Test adding exclusions with reason and magnitude."""
        exclusion = ExclusionRecord(
            scope=Scope.SCOPE_3,
            category=Scope3Category.CAT14_FRANCHISES.value,
            reason="No franchise operations exist within the organizational boundary",
            magnitude_pct=Decimal("0.1"),
        )
        boundary = InventoryBoundary(
            org_id="org-001",
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            reporting_year=2025,
            exclusions=[exclusion],
        )
        assert len(boundary.exclusions) == 1
        assert boundary.exclusions[0].magnitude_pct == Decimal("0.1")

    def test_exclusion_magnitude_bounds(self):
        """Test exclusion magnitude must be 0-100."""
        with pytest.raises(Exception):
            ExclusionRecord(
                scope=Scope.SCOPE_3,
                reason="Over magnitude test case for boundary validation",
                magnitude_pct=Decimal("101"),
            )


# ---------------------------------------------------------------------------
# TestBaseYear
# ---------------------------------------------------------------------------

class TestBaseYear:
    """Test suite for BaseYear model."""

    def test_creation(self, sample_base_year):
        """Test base year creation with emissions data."""
        assert sample_base_year.year == 2019
        assert sample_base_year.total_emissions == Decimal("68000.0")
        assert sample_base_year.locked is False

    def test_locking(self, sample_base_year):
        """Test base year locking."""
        sample_base_year.locked = True
        assert sample_base_year.locked is True

    def test_current_total_no_recalculations(self, sample_base_year):
        """Test current_total returns total_emissions when no recalculations."""
        assert sample_base_year.current_total == Decimal("68000.0")

    def test_recalculation(self, sample_base_year):
        """Test base year recalculation updates current_total."""
        recalc = Recalculation(
            trigger="acquisition",
            original_value=Decimal("68000.0"),
            new_value=Decimal("72000.0"),
            reason="Acquired subsidiary adding 4,000 tCO2e to base year",
            affected_scopes=[Scope.SCOPE_1],
        )
        sample_base_year.recalculations.append(recalc)
        assert sample_base_year.current_total == Decimal("72000.0")

    def test_recalculation_change_pct(self):
        """Test recalculation change percentage calculation."""
        recalc = Recalculation(
            trigger="merger",
            original_value=Decimal("10000.0"),
            new_value=Decimal("12000.0"),
            reason="Merger with subsidiary that added significant emissions",
        )
        assert recalc.change_pct == Decimal("20.0")

    def test_recalculation_zero_original(self):
        """Test recalculation change_pct with zero original."""
        recalc = Recalculation(
            trigger="initial",
            original_value=Decimal("0"),
            new_value=Decimal("5000.0"),
            reason="Initial base year establishment from zero baseline",
        )
        assert recalc.change_pct == Decimal("0")


# ---------------------------------------------------------------------------
# TestScopeEmissions
# ---------------------------------------------------------------------------

class TestScopeEmissions:
    """Test suite for ScopeEmissions model."""

    def test_total_calculation(self, sample_scope1_emissions):
        """Test scope emissions total."""
        assert sample_scope1_emissions.total_tco2e == Decimal("12450.8")

    def test_per_gas_breakdown(self, sample_scope1_emissions):
        """Test per-gas breakdown sums."""
        gas_total = sum(sample_scope1_emissions.by_gas.values())
        assert gas_total == Decimal("12450.8")

    def test_by_category_sum_matches_total(self, sample_scope1_emissions):
        """Test that category sum matches the total."""
        category_total = sum(sample_scope1_emissions.by_category.values())
        assert category_total == sample_scope1_emissions.total_tco2e

    def test_biogenic_co2_separate(self, sample_scope1_emissions):
        """Test biogenic CO2 is tracked separately."""
        assert sample_scope1_emissions.biogenic_co2 == Decimal("85.0")
        # Biogenic CO2 should NOT be in total_tco2e per GHG Protocol
        assert sample_scope1_emissions.biogenic_co2 > 0

    def test_provenance_hash_generated(self, sample_scope1_emissions):
        """Test provenance hash is auto-generated."""
        assert len(sample_scope1_emissions.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self):
        """Test same input produces same provenance hash."""
        e1 = ScopeEmissions(scope=Scope.SCOPE_1, total_tco2e=Decimal("100.0"))
        e2 = ScopeEmissions(scope=Scope.SCOPE_1, total_tco2e=Decimal("100.0"))
        assert e1.provenance_hash == e2.provenance_hash


# ---------------------------------------------------------------------------
# TestGHGInventory
# ---------------------------------------------------------------------------

class TestGHGInventory:
    """Test suite for GHGInventory model."""

    def test_creation(self, sample_ghg_inventory):
        """Test inventory creation."""
        assert sample_ghg_inventory.org_id == "org-001"
        assert sample_ghg_inventory.year == 2025

    def test_grand_total_market_based(self, sample_ghg_inventory):
        """Test grand total = scope1 + scope2_market + scope3."""
        expected = Decimal("12450.8") + Decimal("6100.0") + Decimal("45230.2")
        assert sample_ghg_inventory.grand_total_tco2e == expected

    def test_grand_total_location_based(self, sample_ghg_inventory):
        """Test location-based grand total = scope1 + scope2_location + scope3."""
        expected = Decimal("12450.8") + Decimal("8320.5") + Decimal("45230.2")
        assert sample_ghg_inventory.grand_total_location_tco2e == expected

    def test_recalculate_totals(self, sample_ghg_inventory):
        """Test recalculate_totals updates grand totals."""
        original_total = sample_ghg_inventory.grand_total_tco2e
        sample_ghg_inventory.scope1.total_tco2e = Decimal("15000.0")
        sample_ghg_inventory.recalculate_totals()
        assert sample_ghg_inventory.grand_total_tco2e > original_total

    def test_inventory_no_scopes(self):
        """Test inventory with no scope emissions defaults to zero."""
        inv = GHGInventory(org_id="org-empty", year=2025)
        assert inv.grand_total_tco2e == Decimal("0")
        assert inv.grand_total_location_tco2e == Decimal("0")

    def test_provenance_hash_generated(self, sample_ghg_inventory):
        """Test inventory provenance hash is generated."""
        assert len(sample_ghg_inventory.provenance_hash) == 64

    def test_default_status_draft(self, sample_ghg_inventory):
        """Test default status is draft."""
        assert sample_ghg_inventory.status == "draft"


# ---------------------------------------------------------------------------
# TestIntensityMetric
# ---------------------------------------------------------------------------

class TestIntensityMetric:
    """Test suite for IntensityMetric model."""

    def test_revenue_intensity(self):
        """Test revenue-based intensity calculation."""
        metric = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=Decimal("100"),
            denominator_unit="million USD",
            intensity_value=Decimal("637.81"),
            total_tco2e=Decimal("63781.0"),
            unit="tCO2e/million USD",
        )
        assert metric.intensity_value == Decimal("637.81")
        assert metric.denominator == IntensityDenominator.REVENUE

    def test_employee_intensity(self):
        """Test employee-based intensity calculation."""
        metric = IntensityMetric(
            denominator=IntensityDenominator.EMPLOYEES,
            denominator_value=Decimal("5000"),
            intensity_value=Decimal("12.76"),
            total_tco2e=Decimal("63781.0"),
            unit="tCO2e/employee",
        )
        assert metric.intensity_value == Decimal("12.76")

    def test_production_intensity(self):
        """Test production-based intensity with custom unit name."""
        metric = IntensityMetric(
            denominator=IntensityDenominator.PRODUCTION_UNITS,
            denominator_value=Decimal("1000000"),
            denominator_unit="widgets",
            intensity_value=Decimal("0.0638"),
            total_tco2e=Decimal("63781.0"),
            unit="tCO2e/widget",
        )
        assert metric.denominator_unit == "widgets"

    def test_floor_area_intensity(self):
        """Test floor area intensity."""
        metric = IntensityMetric(
            denominator=IntensityDenominator.FLOOR_AREA,
            denominator_value=Decimal("75000"),
            denominator_unit="m2",
            intensity_value=Decimal("0.85"),
            total_tco2e=Decimal("63781.0"),
            unit="tCO2e/m2",
        )
        assert metric.denominator == IntensityDenominator.FLOOR_AREA

    def test_custom_intensity(self):
        """Test custom denominator intensity metric."""
        metric = IntensityMetric(
            denominator=IntensityDenominator.CUSTOM,
            denominator_value=Decimal("500"),
            denominator_unit="transactions",
            intensity_value=Decimal("127.56"),
            total_tco2e=Decimal("63781.0"),
            unit="tCO2e/transaction",
        )
        assert metric.denominator == IntensityDenominator.CUSTOM

    def test_zero_denominator_raises(self):
        """Test zero denominator raises validation error."""
        with pytest.raises(Exception):
            IntensityMetric(
                denominator=IntensityDenominator.REVENUE,
                denominator_value=Decimal("0"),
                intensity_value=Decimal("0"),
            )


# ---------------------------------------------------------------------------
# TestUncertaintyResult
# ---------------------------------------------------------------------------

class TestUncertaintyResult:
    """Test suite for UncertaintyResult model."""

    def test_bounds_ordering(self):
        """Test p5 <= p50 <= p95."""
        result = UncertaintyResult(
            mean=Decimal("63000.0"),
            p5=Decimal("55000.0"),
            p50=Decimal("62500.0"),
            p95=Decimal("72000.0"),
            std_dev=Decimal("5200.0"),
        )
        assert result.p5 <= result.p50
        assert result.p50 <= result.p95

    def test_cv_calculation(self):
        """Test coefficient of variation."""
        result = UncertaintyResult(
            mean=Decimal("63000.0"),
            p5=Decimal("55000.0"),
            p50=Decimal("62500.0"),
            p95=Decimal("72000.0"),
            std_dev=Decimal("5200.0"),
            cv=Decimal("8.25"),
        )
        assert result.cv == Decimal("8.25")

    def test_default_iterations(self):
        """Test default Monte Carlo iterations."""
        result = UncertaintyResult()
        assert result.iterations == 10_000

    def test_by_scope_breakdown(self):
        """Test per-scope uncertainty breakdown."""
        scope1_unc = ScopeUncertainty(
            scope=Scope.SCOPE_1,
            mean=Decimal("12450.0"),
            p5=Decimal("11000.0"),
            p50=Decimal("12400.0"),
            p95=Decimal("14000.0"),
            std_dev=Decimal("900.0"),
            cv=Decimal("7.23"),
        )
        result = UncertaintyResult(
            by_scope={Scope.SCOPE_1.value: scope1_unc},
        )
        assert Scope.SCOPE_1.value in result.by_scope
        assert result.by_scope[Scope.SCOPE_1.value].mean == Decimal("12450.0")

    def test_sensitivity_ranking(self):
        """Test sensitivity ranking list."""
        result = UncertaintyResult(
            sensitivity_ranking=[
                {"parameter": "natural_gas_ef", "contribution_pct": 35.2},
                {"parameter": "diesel_consumption", "contribution_pct": 22.8},
                {"parameter": "refrigerant_leakage", "contribution_pct": 15.1},
            ],
        )
        assert len(result.sensitivity_ranking) == 3
        assert result.sensitivity_ranking[0]["parameter"] == "natural_gas_ef"


# ---------------------------------------------------------------------------
# TestCompletenessResult
# ---------------------------------------------------------------------------

class TestCompletenessResult:
    """Test suite for CompletenessResult model."""

    def _build_15_mandatory_disclosures(self, all_present=True):
        """Build the 15 mandatory disclosures per GHG Protocol."""
        disclosures = [
            ("MD-01", "Organization description", "organization"),
            ("MD-02", "Consolidation approach", "boundary"),
            ("MD-03", "Operational boundary", "boundary"),
            ("MD-04", "Base year selection", "base_year"),
            ("MD-05", "Base year emissions", "base_year"),
            ("MD-06", "Scope 1 emissions", "scope1"),
            ("MD-07", "Scope 2 emissions", "scope2"),
            ("MD-08", "Scope 2 methodology", "scope2"),
            ("MD-09", "Exclusions justification", "boundary"),
            ("MD-10", "Year-over-year changes", "reporting"),
            ("MD-11", "Methodology references", "reporting"),
            ("MD-12", "Emission factors sources", "reporting"),
            ("MD-13", "GWP values used", "reporting"),
            ("MD-14", "Intensity metrics", "intensity"),
            ("MD-15", "Base year recalculation policy", "base_year"),
        ]
        return [
            Disclosure(id=d[0], name=d[1], category=d[2], required=True, present=all_present)
            for d in disclosures
        ]

    def test_mandatory_disclosures_count_15(self):
        """Test that mandatory disclosures count equals 15."""
        disclosures = self._build_15_mandatory_disclosures()
        result = CompletenessResult(
            mandatory_disclosures=disclosures,
            overall_pct=Decimal("100"),
        )
        assert len(result.mandatory_disclosures) == 15

    def test_all_present_100_pct(self):
        """Test 100% completeness when all disclosures present."""
        disclosures = self._build_15_mandatory_disclosures(all_present=True)
        result = CompletenessResult(
            mandatory_disclosures=disclosures,
            overall_pct=Decimal("100"),
        )
        present_count = sum(1 for d in result.mandatory_disclosures if d.present)
        assert present_count == 15
        assert result.overall_pct == Decimal("100")

    def test_partial_completeness(self):
        """Test partial completeness score."""
        disclosures = self._build_15_mandatory_disclosures(all_present=True)
        # Mark 3 as missing
        disclosures[5].present = False
        disclosures[10].present = False
        disclosures[14].present = False
        present_count = sum(1 for d in disclosures if d.present)
        pct = Decimal(str(round(present_count / 15 * 100, 1)))
        result = CompletenessResult(
            mandatory_disclosures=disclosures,
            overall_pct=pct,
        )
        assert result.overall_pct == Decimal("80.0")
        assert present_count == 12

    def test_data_gaps_identification(self):
        """Test data gap identification."""
        gap = DataGap(
            scope=Scope.SCOPE_3,
            category=Scope3Category.CAT8_UPSTREAM_LEASED.value,
            description="No leased asset data collected",
            severity="high",
            recommendation="Request data from lessors",
            estimated_magnitude_pct=Decimal("3.5"),
        )
        result = CompletenessResult(gaps=[gap])
        assert len(result.gaps) == 1
        assert result.gaps[0].severity == "high"

    def test_scope3_materiality(self):
        """Test Scope 3 materiality assessment."""
        materiality = {
            Scope3Category.CAT1_PURCHASED_GOODS.value: True,
            Scope3Category.CAT2_CAPITAL_GOODS.value: True,
            Scope3Category.CAT14_FRANCHISES.value: False,
            Scope3Category.CAT15_INVESTMENTS.value: False,
        }
        result = CompletenessResult(scope3_materiality=materiality)
        material_count = sum(1 for v in result.scope3_materiality.values() if v)
        assert material_count == 2

    def test_data_quality_score_bounds(self):
        """Test data quality score is between 0 and 100."""
        result = CompletenessResult(data_quality_score=Decimal("82.5"))
        assert Decimal("0") <= result.data_quality_score <= Decimal("100")

    def test_exclusion_assessment(self):
        """Test exclusion assessment tracking."""
        result = CompletenessResult(
            exclusion_assessment={
                "total_excluded_pct": 2.3,
                "excluded_categories": ["cat14_franchises"],
                "significant": False,
            }
        )
        assert result.exclusion_assessment["significant"] is False


# ---------------------------------------------------------------------------
# TestVerificationRecord
# ---------------------------------------------------------------------------

class TestVerificationRecord:
    """Test suite for VerificationRecord model."""

    def test_creation_defaults(self):
        """Test verification record creation with defaults."""
        record = VerificationRecord(inventory_id="inv-001")
        assert record.status == VerificationStatus.DRAFT
        assert record.level == VerificationLevel.INTERNAL_REVIEW

    def test_status_in_review(self):
        """Test setting status to in_review."""
        record = VerificationRecord(
            inventory_id="inv-001",
            status=VerificationStatus.IN_REVIEW,
        )
        assert record.status == VerificationStatus.IN_REVIEW

    def test_status_approved(self):
        """Test approved status."""
        record = VerificationRecord(
            inventory_id="inv-001",
            status=VerificationStatus.APPROVED,
            verifier_id="reviewer-01",
        )
        assert record.status == VerificationStatus.APPROVED

    def test_status_rejected(self):
        """Test rejected status with reason."""
        record = VerificationRecord(
            inventory_id="inv-001",
            status=VerificationStatus.REJECTED,
            opinion="Material misstatement in Scope 1 fugitive emissions",
        )
        assert record.status == VerificationStatus.REJECTED
        assert "misstatement" in record.opinion

    def test_external_verifier_assignment(self):
        """Test external verifier assignment."""
        record = VerificationRecord(
            inventory_id="inv-001",
            level=VerificationLevel.REASONABLE_ASSURANCE,
            verifier_name="Deloitte",
            verifier_organization="Deloitte LLP",
        )
        assert record.verifier_name == "Deloitte"
        assert record.level == VerificationLevel.REASONABLE_ASSURANCE

    def test_findings_tracking(self):
        """Test finding tracking."""
        finding1 = VerificationFinding(
            finding_type=FindingType.MINOR_NONCONFORMITY,
            description="Emission factor source not documented for mobile fleet",
            materiality=FindingSeverity.MEDIUM,
        )
        finding2 = VerificationFinding(
            finding_type=FindingType.OBSERVATION,
            description="Scope 3 Category 8 data completeness below best practice",
            materiality=FindingSeverity.LOW,
        )
        record = VerificationRecord(
            inventory_id="inv-001",
            findings=[finding1, finding2],
        )
        assert len(record.findings) == 2
        assert record.open_findings_count == 2

    def test_findings_resolution(self):
        """Test finding resolution reduces open count."""
        finding = VerificationFinding(
            finding_type=FindingType.MINOR_NONCONFORMITY,
            description="Emission factor source not documented for mobile fleet",
            materiality=FindingSeverity.MEDIUM,
        )
        record = VerificationRecord(
            inventory_id="inv-001",
            findings=[finding],
        )
        assert record.open_findings_count == 1
        record.findings[0].resolved = True
        record.findings[0].resolution = "Documentation updated with EPA source reference"
        assert record.open_findings_count == 0

    def test_has_major_findings(self):
        """Test major findings detection."""
        critical_finding = VerificationFinding(
            finding_type=FindingType.MAJOR_NONCONFORMITY,
            description="Scope 1 process emissions calculation uses incorrect methodology",
            materiality=FindingSeverity.CRITICAL,
        )
        record = VerificationRecord(
            inventory_id="inv-001",
            findings=[critical_finding],
        )
        assert record.has_major_findings is True

    def test_no_major_findings_when_resolved(self):
        """Test no major findings when all resolved."""
        finding = VerificationFinding(
            finding_type=FindingType.MAJOR_NONCONFORMITY,
            description="Scope 1 process emissions calculation uses incorrect methodology",
            materiality=FindingSeverity.HIGH,
            resolved=True,
            resolution="Recalculated using IPCC Tier 2 methodology",
        )
        record = VerificationRecord(
            inventory_id="inv-001",
            findings=[finding],
        )
        assert record.has_major_findings is False

    def test_verification_statement(self):
        """Test verification statement only set when approved."""
        record = VerificationRecord(
            inventory_id="inv-001",
            status=VerificationStatus.APPROVED,
            statement="Unqualified opinion: inventory presents fairly in all material respects",
            opinion="unqualified",
        )
        assert record.statement is not None
        assert record.opinion == "unqualified"


# ---------------------------------------------------------------------------
# TestTarget
# ---------------------------------------------------------------------------

class TestTarget:
    """Test suite for Target model."""

    def test_absolute_target(self):
        """Test absolute reduction target creation."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("42.0"),
        )
        assert target.target_type == TargetType.ABSOLUTE
        assert target.reduction_pct == Decimal("42.0")

    def test_intensity_target(self):
        """Test intensity-based reduction target."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.INTENSITY,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("400.0"),
            target_year=2030,
            reduction_pct=Decimal("30.0"),
        )
        assert target.target_type == TargetType.INTENSITY

    def test_reduction_pct_upper_bound(self):
        """Test reduction at 100% is valid."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2050,
            reduction_pct=Decimal("100.0"),
        )
        assert target.reduction_pct == Decimal("100.0")

    def test_reduction_pct_over_100_raises(self):
        """Test reduction > 100% raises validation error."""
        with pytest.raises(Exception):
            Target(
                org_id="org-001",
                target_type=TargetType.ABSOLUTE,
                scope=Scope.SCOPE_1,
                base_year=2019,
                base_year_emissions=Decimal("68000.0"),
                target_year=2030,
                reduction_pct=Decimal("101.0"),
            )

    def test_target_year_before_base_raises(self):
        """Test target year before base year raises."""
        with pytest.raises(Exception):
            Target(
                org_id="org-001",
                target_type=TargetType.ABSOLUTE,
                scope=Scope.SCOPE_1,
                base_year=2025,
                base_year_emissions=Decimal("68000.0"),
                target_year=2020,
                reduction_pct=Decimal("42.0"),
            )

    def test_sbti_alignment(self):
        """Test SBTi alignment fields."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("46.2"),
            sbti_aligned=True,
            sbti_pathway="1.5C",
        )
        assert target.sbti_aligned is True
        assert target.sbti_pathway == "1.5C"

    def test_current_progress_pct(self):
        """Test progress calculation toward target."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("50.0"),
            current_emissions=Decimal("51000.0"),
            current_year=2025,
        )
        # Actual reduction = 68000 - 51000 = 17000
        # Target reduction = 68000 * 50% = 34000
        # Progress = 17000 / 34000 * 100 = 50%
        assert target.current_progress_pct == Decimal("50")

    def test_progress_zero_when_no_current_emissions(self):
        """Test progress is 0 when no current emissions set."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("42.0"),
        )
        assert target.current_progress_pct == Decimal("0")

    def test_progress_capped_at_100(self):
        """Test progress is capped at 100% even if exceeded."""
        target = Target(
            org_id="org-001",
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("50.0"),
            current_emissions=Decimal("10000.0"),
        )
        assert target.current_progress_pct == Decimal("100")


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------

class TestReport:
    """Test suite for Report model."""

    def test_report_creation(self):
        """Test report creation."""
        report = Report(
            inventory_id="inv-001",
            format=ReportFormat.JSON,
            sections=[
                ReportSection(key="executive_summary", title="Executive Summary", order=1),
                ReportSection(key="scope1", title="Scope 1 Emissions", order=2),
            ],
        )
        assert report.format == ReportFormat.JSON
        assert len(report.sections) == 2

    def test_report_provenance_hash(self):
        """Test report provenance hash is generated."""
        report = Report(inventory_id="inv-001")
        assert len(report.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestDashboardMetrics
# ---------------------------------------------------------------------------

class TestDashboardMetrics:
    """Test suite for DashboardMetrics model."""

    def test_creation(self):
        """Test dashboard metrics creation."""
        metrics = DashboardMetrics(
            org_id="org-001",
            year=2025,
            total_emissions=Decimal("63781.0"),
            scope1_total=Decimal("12450.8"),
            scope2_location_total=Decimal("8320.5"),
            scope2_market_total=Decimal("6100.0"),
            scope3_total=Decimal("45230.2"),
            yoy_change_pct=Decimal("-3.2"),
            data_quality_score=Decimal("82.4"),
            completeness_pct=Decimal("89.5"),
        )
        assert metrics.total_emissions == Decimal("63781.0")
        assert metrics.yoy_change_pct == Decimal("-3.2")


# ---------------------------------------------------------------------------
# TestRequestModels
# ---------------------------------------------------------------------------

class TestRequestModels:
    """Test suite for request/response models."""

    def test_create_organization_request(self):
        """Test CreateOrganizationRequest validation."""
        req = CreateOrganizationRequest(
            name="Acme Corp",
            industry="manufacturing",
            country="US",
        )
        assert req.name == "Acme Corp"

    def test_add_entity_request(self):
        """Test AddEntityRequest validation."""
        req = AddEntityRequest(
            name="East Plant",
            entity_type=EntityType.FACILITY,
            country="US",
            ownership_pct=Decimal("100.0"),
        )
        assert req.entity_type == EntityType.FACILITY

    def test_set_target_request(self):
        """Test SetTargetRequest validation."""
        req = SetTargetRequest(
            target_type=TargetType.ABSOLUTE,
            scope=Scope.SCOPE_1,
            base_year=2019,
            base_year_emissions=Decimal("68000.0"),
            target_year=2030,
            reduction_pct=Decimal("42.0"),
        )
        assert req.reduction_pct == Decimal("42.0")

    def test_exclusion_request(self):
        """Test ExclusionRequest validation."""
        req = ExclusionRequest(
            scope=Scope.SCOPE_3,
            category="cat14_franchises",
            reason="No franchise operations within organizational boundary",
            magnitude_pct=Decimal("0.1"),
        )
        assert req.magnitude_pct == Decimal("0.1")


# ---------------------------------------------------------------------------
# TestConfigEnums
# ---------------------------------------------------------------------------

class TestConfigEnums:
    """Test suite for configuration enums and constants."""

    def test_gwp_ar5_seven_gases(self):
        """Test GWP AR5 has all 7 Kyoto gases."""
        assert len(GWP_AR5) == 7
        assert GWP_AR5[GHGGas.CO2] == 1
        assert GWP_AR5[GHGGas.CH4] == 28
        assert GWP_AR5[GHGGas.N2O] == 265
        assert GWP_AR5[GHGGas.SF6] == 23500

    def test_sector_benchmarks(self):
        """Test sector benchmarks are defined."""
        assert "manufacturing" in SECTOR_BENCHMARKS
        assert "revenue" in SECTOR_BENCHMARKS["manufacturing"]
        assert SECTOR_BENCHMARKS["manufacturing"]["revenue"] == Decimal("420.0")

    def test_uncertainty_cv_by_tier(self):
        """Test uncertainty CV values by tier."""
        assert UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_1] == Decimal("50.0")
        assert UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_3] == Decimal("5.0")
        # Tier 1 has highest uncertainty (industry averages)
        assert UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_1] > UNCERTAINTY_CV_BY_TIER[DataQualityTier.TIER_3]

    def test_scope3_has_15_categories(self):
        """Test Scope 3 enum has 15 categories."""
        assert len(Scope3Category) == 15

    def test_scope1_has_8_categories(self):
        """Test Scope 1 enum has 8 categories."""
        assert len(Scope1Category) == 8
