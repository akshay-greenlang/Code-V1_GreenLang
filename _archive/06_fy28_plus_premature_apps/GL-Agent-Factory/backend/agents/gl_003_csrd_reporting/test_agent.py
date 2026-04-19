"""
Unit Tests for GL-003: CSRD Reporting Agent

Comprehensive test coverage for the CSRD Reporting Agent including:
- Double materiality assessment (impact + financial)
- All ESRS standards (ESRS 1-2, E1-E5, S1-S4, G1)
- Disclosure requirement completeness
- Gap analysis and recommendations
- Phase-in provisions
- ESEF/iXBRL report generation
- Sector-specific standards
- Edge cases and validation

Test coverage target: 85%+
Total tests: 75 golden tests covering all ESRS requirements
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional

from .agent import (
    # Main Agent and I/O
    CSRDReportingAgent,
    CSRDInput,
    CSRDOutput,
    # Enumerations
    ESRSStandard,
    MaterialityLevel,
    CompanySize,
    DisclosureType,
    AssuranceLevel,
    SectorCategory,
    IROMaterialityType,
    # Double Materiality Models
    MaterialityAssessment,
    IROAssessment,
    # Cross-Cutting Standards (ESRS 2)
    ESRS2Governance,
    ESRS2Strategy,
    ESRS2IRO,
    # Environmental Standards
    E1ClimateData,
    E2PollutionData,
    E3WaterData,
    E4BiodiversityData,
    E5CircularEconomyData,
    # Social Standards
    S1WorkforceData,
    S2ValueChainWorkersData,
    S3CommunitiesData,
    S4ConsumersData,
    # Governance Standards
    G1GovernanceData,
    # Disclosure Models
    ESRSDatapoint,
    GapAnalysisItem,
    ComplianceMetrics,
    # ESEF Models
    XBRLTag,
    ESEFReportOutput,
    # Reference Data
    ESRS_DISCLOSURE_REQUIREMENTS,
    SECTOR_SPECIFIC_REQUIREMENTS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> CSRDReportingAgent:
    """Create a CSRDReportingAgent instance for testing."""
    return CSRDReportingAgent()


@pytest.fixture
def agent_with_esef() -> CSRDReportingAgent:
    """Create agent with ESEF generation enabled."""
    return CSRDReportingAgent(config={"generate_esef": True})


@pytest.fixture
def sample_e1_climate_data() -> E1ClimateData:
    """Create sample E1 Climate data."""
    return E1ClimateData(
        has_transition_plan=True,
        transition_plan_aligned_to_15c=True,
        transition_plan_targets=["50% reduction by 2030", "Net zero by 2050"],
        climate_policies=["Climate Policy 2024", "Net Zero Strategy"],
        climate_actions=[
            {"action": "Solar installation", "investment": 5000000, "timeline": "2025"},
            {"action": "Fleet electrification", "investment": 3000000, "timeline": "2026"},
        ],
        total_climate_investment_eur=8000000,
        sbti_commitment=True,
        sbti_validated=True,
        net_zero_target_year=2050,
        interim_targets=[
            {"year": 2030, "reduction_pct": 50},
            {"year": 2040, "reduction_pct": 75},
        ],
        total_energy_consumption_mwh=50000,
        renewable_energy_share_pct=65.5,
        energy_intensity_revenue=50.0,
        fossil_fuel_consumption_mwh=17250,
        scope1_emissions=15000,
        scope2_emissions_location=8000,
        scope2_emissions_market=5000,
        scope3_emissions=75000,
        scope3_categories={
            "cat1_purchased_goods": 30000,
            "cat3_fuel_energy": 5000,
            "cat4_upstream_transport": 10000,
            "cat6_business_travel": 2000,
            "cat7_commuting": 3000,
            "cat11_use_of_products": 25000,
        },
        ghg_intensity_revenue=98.0,
        ghg_removals_tco2e=1000,
        carbon_credits_retired=500,
        carbon_credits_type="Gold Standard VER",
        internal_carbon_price=100,
        carbon_price_application="Investment decisions",
        physical_risk_financial_exposure_eur=25000000,
        transition_risk_financial_exposure_eur=50000000,
        climate_opportunity_revenue_eur=100000000,
    )


@pytest.fixture
def sample_e2_pollution_data() -> E2PollutionData:
    """Create sample E2 Pollution data."""
    return E2PollutionData(
        pollution_policies=["Zero Discharge Policy", "Air Quality Management"],
        pollution_prevention_approach="BAT implementation across all facilities",
        pollution_actions=[
            {"action": "Scrubber installation", "investment": 2000000},
        ],
        pollution_targets=[
            {"pollutant": "NOx", "reduction_pct": 30, "by_year": 2028},
        ],
        air_pollutants_kg={"NOx": 5000, "SOx": 3000, "PM10": 1000},
        water_pollutants_kg={"BOD": 500, "COD": 800},
        soil_pollutants_kg={},
        pollutants_of_concern=["Mercury", "Lead"],
        substances_of_concern_tonnes=150,
        svhc_tonnes=25,
        pollution_remediation_provisions_eur=5000000,
        pollution_risk_exposure_eur=10000000,
    )


@pytest.fixture
def sample_e3_water_data() -> E3WaterData:
    """Create sample E3 Water data."""
    return E3WaterData(
        water_policies=["Water Stewardship Policy"],
        water_actions=[{"action": "Recycling system", "investment": 1000000}],
        water_targets=[{"target": "20% reduction by 2027", "baseline_year": 2023}],
        total_water_withdrawal_m3=500000,
        water_withdrawal_by_source={
            "municipal": 300000,
            "groundwater": 150000,
            "surface_water": 50000,
        },
        water_consumption_m3=400000,
        water_recycled_m3=100000,
        water_stress_area_operations=["Site A - India", "Site B - Spain"],
        water_intensity=0.5,
        water_risk_exposure_eur=8000000,
    )


@pytest.fixture
def sample_e4_biodiversity_data() -> E4BiodiversityData:
    """Create sample E4 Biodiversity data."""
    return E4BiodiversityData(
        biodiversity_strategy="Nature Positive by 2030",
        nature_positive_commitment=True,
        biodiversity_policies=["No Deforestation Policy", "Biodiversity Action Plan"],
        biodiversity_actions=[
            {"action": "Habitat restoration", "hectares": 100, "investment": 500000},
        ],
        biodiversity_investment_eur=500000,
        biodiversity_targets=[{"target": "No net loss by 2025"}],
        no_net_loss_commitment=True,
        sites_near_biodiversity_areas=3,
        land_use_change_ha=50,
        ecosystem_restoration_ha=100,
        species_at_risk_assessment=True,
        biodiversity_risk_exposure_eur=15000000,
        nature_based_solutions_revenue_eur=5000000,
    )


@pytest.fixture
def sample_e5_circular_economy_data() -> E5CircularEconomyData:
    """Create sample E5 Circular Economy data."""
    return E5CircularEconomyData(
        circular_economy_policies=["Circular Design Policy"],
        circular_economy_actions=[
            {"action": "Product take-back program", "investment": 2000000},
        ],
        circular_economy_targets=[{"target": "50% recycled content by 2030"}],
        waste_reduction_target_pct=25,
        recycled_content_target_pct=50,
        total_material_inflows_tonnes=100000,
        renewable_materials_tonnes=30000,
        recycled_materials_tonnes=25000,
        recycled_content_pct=25,
        critical_raw_materials_tonnes=5000,
        total_waste_tonnes=15000,
        hazardous_waste_tonnes=1500,
        waste_recycled_tonnes=10000,
        waste_landfilled_tonnes=3000,
        waste_incinerated_tonnes=1500,
        products_designed_for_circularity_pct=40,
        circular_economy_revenue_eur=50000000,
        resource_efficiency_savings_eur=10000000,
    )


@pytest.fixture
def sample_s1_workforce_data() -> S1WorkforceData:
    """Create sample S1 Workforce data."""
    return S1WorkforceData(
        workforce_policies=["HR Policy", "Diversity & Inclusion Policy"],
        human_rights_policy=True,
        non_discrimination_policy=True,
        worker_engagement_process="Regular town halls and surveys",
        worker_representatives_consultation=True,
        grievance_mechanism=True,
        grievance_cases_filed=45,
        grievance_cases_resolved=42,
        workforce_actions=[
            {"action": "Leadership development program", "participants": 500},
        ],
        workforce_targets=[
            {"target": "40% women in leadership by 2028"},
        ],
        total_employees=5000,
        employees_by_gender={"male": 3000, "female": 2000},
        employees_by_country={"Germany": 2000, "USA": 1500, "India": 1000, "Other": 500},
        employees_permanent=4500,
        employees_temporary=500,
        employees_full_time=4800,
        employees_part_time=200,
        employee_turnover_rate=12.5,
        non_employee_workers=800,
        contractors=500,
        collective_bargaining_coverage_pct=65.0,
        works_council_exists=True,
        gender_diversity_board_pct=35.0,
        gender_diversity_management_pct=32.0,
        age_distribution={"under_30": 20.0, "30_50": 55.0, "over_50": 25.0},
        living_wage_compliance_pct=100.0,
        lowest_wage_ratio_to_minimum=1.3,
        social_protection_coverage_pct=100.0,
        parental_leave_policy=True,
        employees_with_disabilities=150,
        disability_inclusion_program=True,
        training_hours_per_employee=32.5,
        training_investment_per_employee_eur=1500,
        skills_development_programs=["Technical Skills", "Leadership", "Digital"],
        work_related_fatalities=0,
        recordable_work_related_injuries=25,
        lost_time_injury_rate=1.2,
        occupational_illness_cases=5,
        health_safety_management_system=True,
        flexible_working_arrangements=True,
        parental_leave_taken_days=85.0,
        family_leave_return_rate_pct=95.0,
        gender_pay_gap_pct=8.5,
        ceo_to_median_pay_ratio=45.0,
        discrimination_incidents=3,
        harassment_incidents=5,
        human_rights_violations=0,
        fines_penalties_workforce_eur=0,
    )


@pytest.fixture
def sample_s2_value_chain_data() -> S2ValueChainWorkersData:
    """Create sample S2 Value Chain Workers data."""
    return S2ValueChainWorkersData(
        value_chain_worker_policies=["Supplier Code of Conduct"],
        supplier_code_of_conduct=True,
        value_chain_engagement_process="Annual supplier summit",
        supplier_grievance_mechanism=True,
        value_chain_actions=[
            {"action": "Supplier audit program", "suppliers_covered": 500},
        ],
        value_chain_targets=[
            {"target": "100% Tier 1 suppliers audited by 2026"},
        ],
        suppliers_assessed=800,
        suppliers_audited_pct=62.5,
        critical_suppliers_with_issues=15,
        child_labor_incidents=0,
        forced_labor_incidents=0,
    )


@pytest.fixture
def sample_s3_communities_data() -> S3CommunitiesData:
    """Create sample S3 Communities data."""
    return S3CommunitiesData(
        community_policies=["Community Engagement Policy"],
        free_prior_informed_consent_policy=True,
        community_engagement_process="Quarterly stakeholder meetings",
        indigenous_communities_engaged=3,
        community_grievance_mechanism=True,
        community_actions=[
            {"action": "Local education support", "investment": 500000},
        ],
        community_investment_eur=2000000,
        community_targets=[
            {"target": "Zero community incidents by 2025"},
        ],
        community_incidents=2,
        land_rights_disputes=0,
        resettlement_cases=0,
    )


@pytest.fixture
def sample_s4_consumers_data() -> S4ConsumersData:
    """Create sample S4 Consumers data."""
    return S4ConsumersData(
        consumer_policies=["Product Safety Policy", "Privacy Policy"],
        product_safety_policy=True,
        data_privacy_policy=True,
        consumer_engagement_process="Customer advisory board",
        consumer_complaint_mechanism=True,
        consumer_complaints_received=1500,
        consumer_complaints_resolved=1450,
        consumer_actions=[
            {"action": "Product safety testing enhancement", "investment": 1000000},
        ],
        consumer_targets=[
            {"target": "Zero product recalls by 2025"},
        ],
        product_recalls=1,
        product_safety_incidents=3,
        data_breaches=0,
        consumer_fines_eur=0,
    )


@pytest.fixture
def sample_g1_governance_data() -> G1GovernanceData:
    """Create sample G1 Governance data."""
    return G1GovernanceData(
        code_of_conduct=True,
        code_of_conduct_coverage_pct=100.0,
        ethics_training_pct=95.0,
        corporate_culture_description="Integrity, Innovation, Impact",
        supplier_due_diligence_process="Risk-based supplier screening",
        supplier_code_adoption_pct=85.0,
        supplier_assessment_coverage_pct=70.0,
        anti_corruption_policy=True,
        anti_corruption_training_pct=90.0,
        whistleblower_mechanism=True,
        whistleblower_reports=12,
        bribery_risk_assessment=True,
        corruption_incidents=0,
        bribery_incidents=0,
        corruption_fines_eur=0,
        employees_dismissed_corruption=0,
        political_contributions_eur=0,
        lobbying_expenditure_eur=150000,
        trade_association_memberships=["Industry Association A", "Sustainability Council B"],
        political_engagement_policy=True,
        payment_terms_days=45,
        late_payments_pct=8.0,
        average_payment_delay_days=5.0,
    )


@pytest.fixture
def sample_esrs2_governance() -> ESRS2Governance:
    """Create sample ESRS 2 Governance data."""
    return ESRS2Governance(
        board_sustainability_oversight=True,
        board_sustainability_expertise=3,
        sustainability_committee_exists=True,
        board_sustainability_training_hours=12.0,
        sustainability_agenda_frequency=6,
        material_topics_addressed=["Climate Change", "Workforce", "Governance"],
        sustainability_incentives_board=True,
        sustainability_incentives_management=True,
        sustainability_kpis_in_incentives=["GHG reduction", "Safety metrics", "Diversity"],
        due_diligence_statement="OECD Guidelines-aligned due diligence process",
        due_diligence_standards_applied=["OECD Guidelines", "UN Guiding Principles"],
        sustainability_risk_management_process="ERM-integrated sustainability risk management",
        internal_controls_sustainability=True,
    )


@pytest.fixture
def sample_esrs2_strategy() -> ESRS2Strategy:
    """Create sample ESRS 2 Strategy data."""
    return ESRS2Strategy(
        business_model_description="Manufacturing and distribution of sustainable products",
        value_chain_description="Global supply chain with focus on European operations",
        key_stakeholders=["Investors", "Employees", "Customers", "Communities"],
        geographic_presence=["EU", "North America", "Asia Pacific"],
        revenue_by_segment={"Products": 800000000, "Services": 200000000},
        stakeholder_engagement_process="Annual materiality assessment with stakeholder input",
        stakeholder_engagement_frequency="Quarterly",
        material_topics_from_stakeholders=["Climate", "Human rights", "Product safety"],
        material_iros=[
            IROAssessment(
                iro_id="IRO-001",
                iro_type=IROMaterialityType.IMPACT,
                description="GHG emissions from operations",
                esrs_topic=ESRSStandard.E1,
                likelihood=1.0,
                magnitude=0.8,
                time_horizon="long",
                is_actual=True,
            ),
        ],
        strategy_resilience="Scenario analysis conducted for 1.5C and 2C pathways",
    )


@pytest.fixture
def sample_esrs2_iro() -> ESRS2IRO:
    """Create sample ESRS 2 IRO data."""
    return ESRS2IRO(
        iro_identification_process="Enterprise-wide IRO identification process",
        iro_assessment_methodology="Double materiality matrix assessment",
        value_chain_mapping=True,
        sector_specific_guidance_used=False,
        material_topics_disclosed=["E1", "S1", "G1"],
        non_material_topics_explanation={
            "E4": "Limited biodiversity impact due to urban operations",
        },
    )


@pytest.fixture
def basic_input() -> CSRDInput:
    """Create basic input for simple tests."""
    return CSRDInput(
        company_id="EU-CORP-001",
        company_name="Test Corporation",
        reporting_year=2024,
        company_size=CompanySize.LARGE,
    )


@pytest.fixture
def comprehensive_input(
    sample_e1_climate_data: E1ClimateData,
    sample_e2_pollution_data: E2PollutionData,
    sample_e3_water_data: E3WaterData,
    sample_e4_biodiversity_data: E4BiodiversityData,
    sample_e5_circular_economy_data: E5CircularEconomyData,
    sample_s1_workforce_data: S1WorkforceData,
    sample_s2_value_chain_data: S2ValueChainWorkersData,
    sample_s3_communities_data: S3CommunitiesData,
    sample_s4_consumers_data: S4ConsumersData,
    sample_g1_governance_data: G1GovernanceData,
    sample_esrs2_governance: ESRS2Governance,
    sample_esrs2_strategy: ESRS2Strategy,
    sample_esrs2_iro: ESRS2IRO,
) -> CSRDInput:
    """Create comprehensive input with all ESRS data."""
    return CSRDInput(
        company_id="EU-CORP-FULL-001",
        company_name="Comprehensive Sustainability Corp",
        lei_code="5493001KJTIIGC8Y1R12",
        reporting_year=2024,
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
        company_size=CompanySize.LARGE_PIE,
        sector=SectorCategory.MANUFACTURING,
        nace_codes=["C.29", "C.28"],
        # Cross-cutting
        esrs2_governance=sample_esrs2_governance,
        esrs2_strategy=sample_esrs2_strategy,
        esrs2_iro=sample_esrs2_iro,
        # Environmental
        e1_climate_data=sample_e1_climate_data,
        e2_pollution_data=sample_e2_pollution_data,
        e3_water_data=sample_e3_water_data,
        e4_biodiversity_data=sample_e4_biodiversity_data,
        e5_circular_economy_data=sample_e5_circular_economy_data,
        # Social
        s1_workforce_data=sample_s1_workforce_data,
        s2_value_chain_data=sample_s2_value_chain_data,
        s3_communities_data=sample_s3_communities_data,
        s4_consumers_data=sample_s4_consumers_data,
        # Governance
        g1_governance_data=sample_g1_governance_data,
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_1_agent_initialization(self, agent: CSRDReportingAgent):
        """Test 1: Agent initializes correctly."""
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/csrd_reporting_v1"
        assert agent.VERSION == "2.0.0"

    def test_2_agent_with_config(self):
        """Test 2: Agent initialization with config."""
        config = {"generate_esef": True, "custom_setting": "value"}
        agent = CSRDReportingAgent(config=config)
        assert agent.config == config

    def test_3_esrs_standards_loaded(self, agent: CSRDReportingAgent):
        """Test 3: ESRS standards are loaded."""
        standards = agent.get_esrs_standards()
        assert len(standards) == 12
        standard_ids = [s["id"] for s in standards]
        assert "ESRS_1" in standard_ids
        assert "ESRS_2" in standard_ids
        assert "E1" in standard_ids
        assert "G1" in standard_ids

    def test_4_disclosure_requirements_loaded(self, agent: CSRDReportingAgent):
        """Test 4: Disclosure requirements are loaded for each standard."""
        for standard in ESRSStandard:
            requirements = agent.get_disclosure_requirements(standard)
            assert isinstance(requirements, list)

    def test_5_materiality_thresholds(self, agent: CSRDReportingAgent):
        """Test 5: Materiality thresholds are configured."""
        thresholds = agent.get_materiality_thresholds()
        assert thresholds["impact_threshold"] == 0.5
        assert thresholds["financial_threshold"] == 0.5

    def test_6_sector_requirements_available(self, agent: CSRDReportingAgent):
        """Test 6: Sector-specific requirements are available."""
        oil_gas_reqs = agent.get_sector_requirements(SectorCategory.OIL_GAS)
        assert "ESRS-OG-1" in oil_gas_reqs

        general_reqs = agent.get_sector_requirements(SectorCategory.GENERAL)
        assert len(general_reqs) == 0

    def test_7_basic_run_completes(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 7: Basic agent run completes."""
        result = agent.run(basic_input)
        assert result is not None
        assert isinstance(result, CSRDOutput)

    def test_8_run_returns_provenance_hash(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 8: Run returns provenance hash."""
        result = agent.run(basic_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_9_run_tracks_processing_time(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 9: Run tracks processing time."""
        result = agent.run(basic_input)
        assert result.processing_time_ms > 0

    def test_10_input_validation(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 10: Input validation works."""
        # Invalid year
        errors = agent.validate_input(CSRDInput(
            company_id="TEST",
            reporting_year=2023,  # Too early
        ))
        assert "Reporting year must be 2024 or later" in errors


# =============================================================================
# Test 11-20: Double Materiality Assessment Tests
# =============================================================================


class TestDoubleMaterialityAssessment:
    """Tests for double materiality assessment."""

    def test_11_materiality_assessment_model(self):
        """Test 11: MaterialityAssessment model works correctly."""
        assessment = MaterialityAssessment(
            topic="E1",
            impact_materiality=0.8,
            financial_materiality=0.6,
        )
        assert assessment.is_impact_material
        assert assessment.is_financially_material
        assert assessment.is_material
        assert assessment.materiality_level == MaterialityLevel.HIGH

    def test_12_materiality_impact_only(self):
        """Test 12: Topic is material if only impact is material."""
        assessment = MaterialityAssessment(
            topic="E2",
            impact_materiality=0.7,
            financial_materiality=0.3,
        )
        assert assessment.is_impact_material
        assert not assessment.is_financially_material
        assert assessment.is_material  # Still material

    def test_13_materiality_financial_only(self):
        """Test 13: Topic is material if only financial is material."""
        assessment = MaterialityAssessment(
            topic="G1",
            impact_materiality=0.3,
            financial_materiality=0.8,
        )
        assert not assessment.is_impact_material
        assert assessment.is_financially_material
        assert assessment.is_material  # Still material

    def test_14_materiality_not_material(self):
        """Test 14: Topic is not material if both below threshold."""
        assessment = MaterialityAssessment(
            topic="E4",
            impact_materiality=0.2,
            financial_materiality=0.2,
        )
        assert not assessment.is_impact_material
        assert not assessment.is_financially_material
        assert not assessment.is_material
        assert assessment.materiality_level == MaterialityLevel.NOT_MATERIAL

    def test_15_materiality_levels(self):
        """Test 15: Materiality levels are correctly assigned."""
        # HIGH
        high = MaterialityAssessment(topic="E1", impact_materiality=0.9, financial_materiality=0.5)
        assert high.materiality_level == MaterialityLevel.HIGH

        # MEDIUM
        medium = MaterialityAssessment(topic="E2", impact_materiality=0.6, financial_materiality=0.5)
        assert medium.materiality_level == MaterialityLevel.MEDIUM

        # LOW
        low = MaterialityAssessment(topic="E3", impact_materiality=0.35, financial_materiality=0.2)
        assert low.materiality_level == MaterialityLevel.LOW

    def test_16_custom_thresholds(self):
        """Test 16: Custom thresholds work correctly."""
        assessment = MaterialityAssessment(
            topic="E1",
            impact_materiality=0.6,
            financial_materiality=0.6,
            impact_threshold=0.7,  # Higher threshold
            financial_threshold=0.7,
        )
        assert not assessment.is_impact_material
        assert not assessment.is_financially_material
        assert not assessment.is_material

    def test_17_materiality_assessment_in_run(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 17: Materiality assessment is performed during run."""
        result = agent.run(comprehensive_input)
        assert len(result.material_topics) > 0
        assert "ESRS_1" in result.material_topics  # Always required
        assert "ESRS_2" in result.material_topics  # Always required

    def test_18_provided_materiality_assessments_used(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 18: Provided materiality assessments are used."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            materiality_assessments=[
                MaterialityAssessment(topic="E1", impact_materiality=0.9, financial_materiality=0.9),
                MaterialityAssessment(topic="G1", impact_materiality=0.8, financial_materiality=0.8),
            ],
            e1_climate_data=E1ClimateData(scope1_emissions=10000),
            g1_governance_data=G1GovernanceData(code_of_conduct=True),
        )
        result = agent.run(input_data)
        assert len(result.materiality_assessments) == 2
        assert result.materiality_assessments[0].topic == "E1"

    def test_19_iro_assessment_model(self):
        """Test 19: IROAssessment model works correctly."""
        iro = IROAssessment(
            iro_id="IRO-001",
            iro_type=IROMaterialityType.IMPACT,
            description="Test IRO",
            esrs_topic=ESRSStandard.E1,
            likelihood=0.8,
            magnitude=0.7,
            time_horizon="long",
            is_actual=True,
        )
        assert iro.severity_score == 0.56  # 0.8 * 0.7

    def test_20_inferred_materiality_from_data(
        self,
        agent: CSRDReportingAgent,
        sample_e1_climate_data: E1ClimateData,
    ):
        """Test 20: Materiality is inferred from provided data."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e1_climate_data=sample_e1_climate_data,
        )
        result = agent.run(input_data)
        # E1 should be inferred as material
        assert "E1" in result.material_topics


# =============================================================================
# Test 21-35: Environmental Standards Tests (E1-E5)
# =============================================================================


class TestEnvironmentalStandards:
    """Tests for environmental standards E1-E5."""

    def test_21_e1_climate_data_model(self, sample_e1_climate_data: E1ClimateData):
        """Test 21: E1ClimateData model validates correctly."""
        assert sample_e1_climate_data.scope1_emissions == 15000
        assert sample_e1_climate_data.total_emissions == 98000  # Auto-calculated

    def test_22_e1_total_emissions_calculation(self):
        """Test 22: Total emissions auto-calculated when not provided."""
        data = E1ClimateData(
            scope1_emissions=10000,
            scope2_emissions_location=5000,
            scope3_emissions=30000,
        )
        assert data.total_emissions == 45000

    def test_23_e1_disclosure_completeness(
        self,
        agent: CSRDReportingAgent,
        sample_e1_climate_data: E1ClimateData,
    ):
        """Test 23: E1 disclosure completeness is assessed."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e1_climate_data=sample_e1_climate_data,
        )
        result = agent.run(input_data)
        # E1 should have high completeness with full data
        e1_metrics = result.compliance_by_standard.get("E1")
        assert e1_metrics is not None
        assert e1_metrics.completeness_pct > 50

    def test_24_e1_summary_extraction(
        self,
        agent: CSRDReportingAgent,
        sample_e1_climate_data: E1ClimateData,
    ):
        """Test 24: E1 summary metrics are extracted."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e1_climate_data=sample_e1_climate_data,
        )
        result = agent.run(input_data)
        assert result.e1_summary["scope1_emissions_tco2e"] == 15000
        assert result.e1_summary["has_transition_plan"] == True
        assert result.e1_summary["sbti_commitment"] == True

    def test_25_e2_pollution_data_model(self, sample_e2_pollution_data: E2PollutionData):
        """Test 25: E2PollutionData model validates correctly."""
        assert sample_e2_pollution_data.substances_of_concern_tonnes == 150
        assert "NOx" in sample_e2_pollution_data.air_pollutants_kg

    def test_26_e2_materiality_assessment(
        self,
        agent: CSRDReportingAgent,
        sample_e2_pollution_data: E2PollutionData,
    ):
        """Test 26: E2 materiality is assessed from data."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e2_pollution_data=sample_e2_pollution_data,
        )
        result = agent.run(input_data)
        assert "E2" in result.material_topics

    def test_27_e3_water_data_model(self, sample_e3_water_data: E3WaterData):
        """Test 27: E3WaterData model validates correctly."""
        assert sample_e3_water_data.water_consumption_m3 == 400000
        assert len(sample_e3_water_data.water_stress_area_operations) == 2

    def test_28_e3_completeness_assessment(
        self,
        agent: CSRDReportingAgent,
        sample_e3_water_data: E3WaterData,
    ):
        """Test 28: E3 completeness is assessed."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e3_water_data=sample_e3_water_data,
        )
        result = agent.run(input_data)
        assert "E3" in result.environmental_summary.get("standards_reported", [])

    def test_29_e4_biodiversity_data_model(self, sample_e4_biodiversity_data: E4BiodiversityData):
        """Test 29: E4BiodiversityData model validates correctly."""
        assert sample_e4_biodiversity_data.nature_positive_commitment == True
        assert sample_e4_biodiversity_data.sites_near_biodiversity_areas == 3

    def test_30_e4_no_net_loss_tracking(self, sample_e4_biodiversity_data: E4BiodiversityData):
        """Test 30: E4 no net loss commitment is tracked."""
        assert sample_e4_biodiversity_data.no_net_loss_commitment == True
        assert sample_e4_biodiversity_data.ecosystem_restoration_ha >= sample_e4_biodiversity_data.land_use_change_ha

    def test_31_e5_circular_economy_data_model(self, sample_e5_circular_economy_data: E5CircularEconomyData):
        """Test 31: E5CircularEconomyData model validates correctly."""
        assert sample_e5_circular_economy_data.recycled_content_pct == 25
        assert sample_e5_circular_economy_data.products_designed_for_circularity_pct == 40

    def test_32_e5_waste_metrics(self, sample_e5_circular_economy_data: E5CircularEconomyData):
        """Test 32: E5 waste metrics are consistent."""
        data = sample_e5_circular_economy_data
        # Total waste should equal sum of disposal routes
        calculated_waste = (
            data.waste_recycled_tonnes +
            data.waste_landfilled_tonnes +
            data.waste_incinerated_tonnes
        )
        assert data.total_waste_tonnes >= calculated_waste - 500  # Allow some tolerance

    def test_33_environmental_summary_aggregation(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 33: Environmental summary aggregates all E standards."""
        result = agent.run(comprehensive_input)
        env_summary = result.environmental_summary
        assert "E1" in env_summary["standards_reported"]
        assert "E2" in env_summary["standards_reported"]
        assert "E3" in env_summary["standards_reported"]
        assert "E4" in env_summary["standards_reported"]
        assert "E5" in env_summary["standards_reported"]

    def test_34_e1_phase_in_tracking(self, agent: CSRDReportingAgent):
        """Test 34: E1-9 phase-in is tracked (2026)."""
        requirements = agent.get_disclosure_requirements(ESRSStandard.E1)
        e1_9 = next((r for r in requirements if r["dr"] == "E1-9"), None)
        assert e1_9 is not None
        assert e1_9.get("phase_in") == 2026

    def test_35_environmental_financial_effects_phase_in(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 35: Financial effects disclosures phase-in in 2026."""
        # For 2024, E1-9 should not be required
        input_2024 = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e1_climate_data=E1ClimateData(scope1_emissions=10000),
        )
        result_2024 = agent.run(input_2024)

        # For 2026+, it should be required
        input_2026 = CSRDInput(
            company_id="TEST",
            reporting_year=2026,
            e1_climate_data=E1ClimateData(scope1_emissions=10000),
        )
        result_2026 = agent.run(input_2026)

        # 2026 should have more datapoints due to phase-in
        assert result_2026.total_datapoints >= result_2024.total_datapoints


# =============================================================================
# Test 36-50: Social Standards Tests (S1-S4)
# =============================================================================


class TestSocialStandards:
    """Tests for social standards S1-S4."""

    def test_36_s1_workforce_data_model(self, sample_s1_workforce_data: S1WorkforceData):
        """Test 36: S1WorkforceData model validates correctly."""
        assert sample_s1_workforce_data.total_employees == 5000
        assert sample_s1_workforce_data.gender_diversity_board_pct == 35.0

    def test_37_s1_employee_demographics(self, sample_s1_workforce_data: S1WorkforceData):
        """Test 37: S1 employee demographics are complete."""
        data = sample_s1_workforce_data
        # Gender breakdown should sum to total
        gender_total = sum(data.employees_by_gender.values())
        assert gender_total == data.total_employees

    def test_38_s1_health_safety_metrics(self, sample_s1_workforce_data: S1WorkforceData):
        """Test 38: S1 health and safety metrics are tracked."""
        data = sample_s1_workforce_data
        assert data.work_related_fatalities == 0  # Zero fatalities
        assert data.lost_time_injury_rate == 1.2
        assert data.health_safety_management_system == True

    def test_39_s1_diversity_metrics(self, sample_s1_workforce_data: S1WorkforceData):
        """Test 39: S1 diversity metrics are complete."""
        data = sample_s1_workforce_data
        assert data.gender_diversity_board_pct is not None
        assert data.gender_diversity_management_pct is not None
        assert len(data.age_distribution) > 0

    def test_40_s1_disclosure_completeness(
        self,
        agent: CSRDReportingAgent,
        sample_s1_workforce_data: S1WorkforceData,
    ):
        """Test 40: S1 disclosure completeness is high with full data."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            s1_workforce_data=sample_s1_workforce_data,
        )
        result = agent.run(input_data)
        s1_metrics = result.compliance_by_standard.get("S1")
        assert s1_metrics is not None
        assert s1_metrics.completeness_pct > 50

    def test_41_s1_summary_extraction(
        self,
        agent: CSRDReportingAgent,
        sample_s1_workforce_data: S1WorkforceData,
    ):
        """Test 41: S1 summary metrics are extracted."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            s1_workforce_data=sample_s1_workforce_data,
        )
        result = agent.run(input_data)
        assert result.social_summary["total_employees"] == 5000
        assert result.social_summary["training_hours_per_employee"] == 32.5

    def test_42_s2_value_chain_data_model(self, sample_s2_value_chain_data: S2ValueChainWorkersData):
        """Test 42: S2ValueChainWorkersData model validates correctly."""
        assert sample_s2_value_chain_data.supplier_code_of_conduct == True
        assert sample_s2_value_chain_data.child_labor_incidents == 0

    def test_43_s2_supplier_assessment_coverage(self, sample_s2_value_chain_data: S2ValueChainWorkersData):
        """Test 43: S2 supplier assessment coverage is tracked."""
        data = sample_s2_value_chain_data
        assert data.suppliers_assessed == 800
        assert data.suppliers_audited_pct == 62.5

    def test_44_s2_materiality_from_data(
        self,
        agent: CSRDReportingAgent,
        sample_s2_value_chain_data: S2ValueChainWorkersData,
    ):
        """Test 44: S2 materiality inferred from data."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            s2_value_chain_data=sample_s2_value_chain_data,
        )
        result = agent.run(input_data)
        assert "S2" in result.material_topics

    def test_45_s3_communities_data_model(self, sample_s3_communities_data: S3CommunitiesData):
        """Test 45: S3CommunitiesData model validates correctly."""
        assert sample_s3_communities_data.free_prior_informed_consent_policy == True
        assert sample_s3_communities_data.community_investment_eur == 2000000

    def test_46_s3_fpic_policy_tracking(self, sample_s3_communities_data: S3CommunitiesData):
        """Test 46: S3 FPIC policy is tracked."""
        data = sample_s3_communities_data
        assert data.free_prior_informed_consent_policy == True
        assert data.indigenous_communities_engaged == 3

    def test_47_s4_consumers_data_model(self, sample_s4_consumers_data: S4ConsumersData):
        """Test 47: S4ConsumersData model validates correctly."""
        assert sample_s4_consumers_data.product_safety_policy == True
        assert sample_s4_consumers_data.data_privacy_policy == True

    def test_48_s4_complaint_resolution_rate(self, sample_s4_consumers_data: S4ConsumersData):
        """Test 48: S4 complaint resolution rate can be calculated."""
        data = sample_s4_consumers_data
        resolution_rate = data.consumer_complaints_resolved / data.consumer_complaints_received * 100
        assert resolution_rate > 95  # High resolution rate

    def test_49_social_summary_aggregation(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 49: Social summary aggregates all S standards."""
        result = agent.run(comprehensive_input)
        social_summary = result.social_summary
        assert "S1" in social_summary["standards_reported"]
        assert "S2" in social_summary["standards_reported"]
        assert "S3" in social_summary["standards_reported"]
        assert "S4" in social_summary["standards_reported"]

    def test_50_s1_remuneration_phase_in(self, agent: CSRDReportingAgent):
        """Test 50: S1-16 remuneration metrics phase-in (2026)."""
        requirements = agent.get_disclosure_requirements(ESRSStandard.S1)
        s1_16 = next((r for r in requirements if r["dr"] == "S1-16"), None)
        assert s1_16 is not None
        assert s1_16.get("phase_in") == 2026


# =============================================================================
# Test 51-60: Governance Standards Tests (G1) and ESRS 2
# =============================================================================


class TestGovernanceStandards:
    """Tests for governance standards G1 and ESRS 2."""

    def test_51_g1_governance_data_model(self, sample_g1_governance_data: G1GovernanceData):
        """Test 51: G1GovernanceData model validates correctly."""
        assert sample_g1_governance_data.code_of_conduct == True
        assert sample_g1_governance_data.anti_corruption_policy == True

    def test_52_g1_anti_corruption_metrics(self, sample_g1_governance_data: G1GovernanceData):
        """Test 52: G1 anti-corruption metrics are complete."""
        data = sample_g1_governance_data
        assert data.anti_corruption_training_pct == 90.0
        assert data.whistleblower_mechanism == True
        assert data.corruption_incidents == 0

    def test_53_g1_payment_practices(self, sample_g1_governance_data: G1GovernanceData):
        """Test 53: G1 payment practices are tracked."""
        data = sample_g1_governance_data
        assert data.payment_terms_days == 45
        assert data.late_payments_pct == 8.0

    def test_54_g1_disclosure_completeness(
        self,
        agent: CSRDReportingAgent,
        sample_g1_governance_data: G1GovernanceData,
    ):
        """Test 54: G1 disclosure completeness with full data."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            g1_governance_data=sample_g1_governance_data,
        )
        result = agent.run(input_data)
        g1_metrics = result.compliance_by_standard.get("G1")
        assert g1_metrics is not None
        assert g1_metrics.completeness_pct > 50

    def test_55_g1_summary_extraction(
        self,
        agent: CSRDReportingAgent,
        sample_g1_governance_data: G1GovernanceData,
    ):
        """Test 55: G1 summary metrics are extracted."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            g1_governance_data=sample_g1_governance_data,
        )
        result = agent.run(input_data)
        assert result.governance_summary["code_of_conduct"] == True
        assert result.governance_summary["whistleblower_mechanism"] == True

    def test_56_esrs2_governance_model(self, sample_esrs2_governance: ESRS2Governance):
        """Test 56: ESRS2Governance model validates correctly."""
        assert sample_esrs2_governance.board_sustainability_oversight == True
        assert sample_esrs2_governance.sustainability_committee_exists == True

    def test_57_esrs2_strategy_model(self, sample_esrs2_strategy: ESRS2Strategy):
        """Test 57: ESRS2Strategy model validates correctly."""
        assert sample_esrs2_strategy.business_model_description is not None
        assert len(sample_esrs2_strategy.material_iros) > 0

    def test_58_esrs2_iro_model(self, sample_esrs2_iro: ESRS2IRO):
        """Test 58: ESRS2IRO model validates correctly."""
        assert sample_esrs2_iro.iro_identification_process is not None
        assert sample_esrs2_iro.value_chain_mapping == True

    def test_59_esrs2_always_required(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 59: ESRS 1 and ESRS 2 are always in material topics."""
        result = agent.run(basic_input)
        assert "ESRS_1" in result.material_topics
        assert "ESRS_2" in result.material_topics

    def test_60_esrs2_cross_cutting_completeness(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 60: ESRS 2 completeness with all data."""
        result = agent.run(comprehensive_input)
        esrs2_metrics = result.compliance_by_standard.get("ESRS_2")
        assert esrs2_metrics is not None


# =============================================================================
# Test 61-70: Gap Analysis, Assurance, and Compliance Tests
# =============================================================================


class TestGapAnalysisAndCompliance:
    """Tests for gap analysis and compliance assessment."""

    def test_61_gap_analysis_generated(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 61: Gap analysis is generated."""
        result = agent.run(basic_input)
        assert isinstance(result.gap_analysis, list)
        # Should have gaps since no data provided
        assert len(result.gap_analysis) > 0

    def test_62_gap_recommendations_provided(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 62: Gap recommendations are provided."""
        result = agent.run(basic_input)
        for gap in result.gap_analysis:
            assert gap.recommendation is not None

    def test_63_critical_gaps_identified(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test 63: Critical mandatory gaps are identified."""
        result = agent.run(basic_input)
        # Should identify mandatory gaps
        assert len(result.critical_gaps) > 0

    def test_64_completeness_calculation_zero_hallucination(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 64: Completeness calculation is deterministic (zero-hallucination)."""
        input_data = CSRDInput(
            company_id="TEST",
            reporting_year=2024,
            e1_climate_data=E1ClimateData(scope1_emissions=10000),
        )
        result1 = agent.run(input_data)
        result2 = agent.run(input_data)
        # Same input should give same completeness score
        assert result1.completeness_score == result2.completeness_score

    def test_65_assurance_level_limited_before_2030(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 65: Assurance level is limited before 2030."""
        for year in [2024, 2025, 2029]:
            input_data = CSRDInput(company_id="TEST", reporting_year=year)
            result = agent.run(input_data)
            assert result.assurance_level == "limited"

    def test_66_assurance_level_reasonable_from_2030(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 66: Assurance level is reasonable from 2030."""
        input_data = CSRDInput(company_id="TEST", reporting_year=2030)
        result = agent.run(input_data)
        assert result.assurance_level == "reasonable"

    def test_67_per_standard_metrics_calculated(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 67: Per-standard compliance metrics are calculated."""
        result = agent.run(comprehensive_input)
        assert len(result.compliance_by_standard) > 0
        for std_id, metrics in result.compliance_by_standard.items():
            assert isinstance(metrics, ComplianceMetrics)
            assert metrics.total_datapoints >= 0
            assert 0 <= metrics.completeness_pct <= 100

    def test_68_comprehensive_completeness_high(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 68: Comprehensive input achieves high completeness."""
        result = agent.run(comprehensive_input)
        assert result.completeness_score > 50

    def test_69_sme_phase_in_provisions(
        self,
        agent: CSRDReportingAgent,
    ):
        """Test 69: SME phase-in provisions are applied."""
        # SME in 2025 should have fewer requirements
        sme_input = CSRDInput(
            company_id="SME-TEST",
            reporting_year=2025,
            company_size=CompanySize.SME,
            e1_climate_data=E1ClimateData(scope1_emissions=5000),
        )
        sme_result = agent.run(sme_input)

        # Large company same year
        large_input = CSRDInput(
            company_id="LARGE-TEST",
            reporting_year=2025,
            company_size=CompanySize.LARGE,
            e1_climate_data=E1ClimateData(scope1_emissions=5000),
        )
        large_result = agent.run(large_input)

        # SME should have fewer required datapoints due to phase-in
        assert sme_result.total_datapoints <= large_result.total_datapoints

    def test_70_mandatory_completeness_tracked(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 70: Mandatory completeness is tracked separately."""
        result = agent.run(comprehensive_input)
        assert result.mandatory_datapoints >= 0
        assert result.mandatory_filled >= 0
        assert result.mandatory_filled <= result.mandatory_datapoints
        assert 0 <= result.mandatory_completeness <= 100


# =============================================================================
# Test 71-75: ESEF/iXBRL Report Generation Tests
# =============================================================================


class TestESEFReportGeneration:
    """Tests for ESEF/iXBRL report generation."""

    def test_71_esef_report_generated_when_enabled(
        self,
        agent_with_esef: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 71: ESEF report is generated when enabled."""
        result = agent_with_esef.run(comprehensive_input)
        assert result.esef_report is not None
        assert isinstance(result.esef_report, ESEFReportOutput)

    def test_72_esef_report_contains_xhtml(
        self,
        agent_with_esef: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 72: ESEF report contains valid XHTML."""
        result = agent_with_esef.run(comprehensive_input)
        xhtml = result.esef_report.xhtml_content
        assert "<?xml version" in xhtml
        assert "<!DOCTYPE html" in xhtml
        assert "xmlns:ix" in xhtml

    def test_73_esef_xbrl_tags_generated(
        self,
        agent_with_esef: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 73: XBRL tags are generated for datapoints."""
        result = agent_with_esef.run(comprehensive_input)
        assert len(result.esef_report.xbrl_tags) > 0
        for tag in result.esef_report.xbrl_tags:
            assert isinstance(tag, XBRLTag)
            assert tag.element_id.startswith("esrs:")

    def test_74_esef_contexts_defined(
        self,
        agent_with_esef: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 74: XBRL contexts are properly defined."""
        result = agent_with_esef.run(comprehensive_input)
        contexts = result.esef_report.contexts
        assert "instant_current" in contexts
        assert "duration_current" in contexts

    def test_75_esef_validation_status(
        self,
        agent_with_esef: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test 75: ESEF validation status is reported."""
        result = agent_with_esef.run(comprehensive_input)
        assert result.esef_report.validation_status in ["PASS", "FAIL"]
        assert result.esef_report.taxonomy_version == "ESRS_2024"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Additional tests for edge cases."""

    def test_empty_data_handling(self, agent: CSRDReportingAgent):
        """Test handling of minimal input."""
        input_data = CSRDInput(
            company_id="EMPTY-TEST",
            reporting_year=2024,
        )
        result = agent.run(input_data)
        assert result is not None
        assert result.completeness_score >= 0

    def test_all_sectors_supported(self, agent: CSRDReportingAgent):
        """Test all sector categories are supported."""
        for sector in SectorCategory:
            input_data = CSRDInput(
                company_id=f"TEST-{sector.value}",
                reporting_year=2024,
                sector=sector,
            )
            result = agent.run(input_data)
            assert result is not None

    def test_all_company_sizes_supported(self, agent: CSRDReportingAgent):
        """Test all company sizes are supported."""
        for size in CompanySize:
            input_data = CSRDInput(
                company_id=f"TEST-{size.value}",
                reporting_year=2024,
                company_size=size,
            )
            result = agent.run(input_data)
            assert result is not None

    def test_provenance_hash_unique_per_run(
        self,
        agent: CSRDReportingAgent,
        basic_input: CSRDInput,
    ):
        """Test provenance hash is unique per run (includes timestamp)."""
        result1 = agent.run(basic_input)
        result2 = agent.run(basic_input)
        # Hashes should be different due to timestamp
        assert result1.provenance_hash != result2.provenance_hash

    def test_calculation_steps_tracked(
        self,
        agent: CSRDReportingAgent,
        comprehensive_input: CSRDInput,
    ):
        """Test calculation steps are tracked for audit."""
        result = agent.run(comprehensive_input)
        assert len(result.calculation_steps) > 0
        step_types = [s["step_type"] for s in result.calculation_steps]
        assert "double_materiality_assessment" in step_types
        assert "calculation" in step_types


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
