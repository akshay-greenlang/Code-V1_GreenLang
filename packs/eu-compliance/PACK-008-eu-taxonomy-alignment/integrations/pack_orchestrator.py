"""
PACK-008 EU Taxonomy Alignment Pack Orchestrator

This module implements the 10-phase pipeline orchestration for the EU Taxonomy
Alignment Pack. It coordinates GL-Taxonomy-APP engines, MRV agents, data intake
agents, and foundation agents to perform end-to-end taxonomy alignment assessment.

The orchestrator manages:
- Health verification (20 categories)
- Configuration setup (organization type, objectives, DA versions)
- Activity inventory (NACE code mapping to ~240 taxonomy activities)
- Eligibility screening (activity-level eligibility per objective)
- Substantial Contribution assessment (TSC evaluation)
- DNSH assessment (6-objective matrix)
- Minimum Safeguards verification (human rights, anti-corruption, taxation, competition)
- KPI/GAR calculation (Turnover/CapEx/OpEx ratios, Green Asset Ratio)
- Disclosure generation (Article 8, EBA Pillar 3 templates)
- Audit trail (SHA-256 provenance tracking)

Example:
    >>> config = TaxonomyOrchestratorConfig(
    ...     organization_type="non_financial_undertaking",
    ...     environmental_objectives=["CCM", "CCA"],
    ...     reporting_period_year=2025
    ... )
    >>> orchestrator = TaxonomyPackOrchestrator(config)
    >>> result = await orchestrator.execute(data)
    >>> assert result.overall_status == "PASS"
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import hashlib
import logging
import asyncio

logger = logging.getLogger(__name__)


class TaxonomyPipelinePhase(str, Enum):
    """Pipeline execution phases for taxonomy alignment."""
    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    ACTIVITY_INVENTORY = "activity_inventory"
    ELIGIBILITY_SCREENING = "eligibility_screening"
    SC_ASSESSMENT = "sc_assessment"
    DNSH_ASSESSMENT = "dnsh_assessment"
    MS_VERIFICATION = "ms_verification"
    KPI_GAR_CALCULATION = "kpi_gar_calculation"
    DISCLOSURE_GENERATION = "disclosure_generation"
    AUDIT_TRAIL = "audit_trail"


class TaxonomyOrchestratorConfig(BaseModel):
    """Configuration for PACK-008 Taxonomy Alignment orchestrator."""

    organization_type: Literal["non_financial_undertaking", "financial_institution", "asset_manager"] = Field(
        default="non_financial_undertaking",
        description="Type of reporting entity"
    )
    environmental_objectives: List[str] = Field(
        default=["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
        description="Environmental objectives to assess"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    delegated_act_version: str = Field(
        default="2023",
        description="Active Delegated Act version"
    )
    enable_gar_calculation: bool = Field(
        default=False,
        description="Enable GAR/BTAR for financial institutions"
    )
    enable_cross_framework: bool = Field(
        default=True,
        description="Enable CSRD/SFDR/TCFD cross-framework mapping"
    )
    enable_capex_plan: bool = Field(
        default=True,
        description="Enable CapEx plan recognition (5-year)"
    )
    batch_size: int = Field(
        default=500,
        ge=1,
        description="Batch size for bulk activity processing"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel agent execution"
    )
    double_counting_prevention: bool = Field(
        default=True,
        description="Prevent double counting across objectives"
    )
    include_nuclear_gas: bool = Field(
        default=False,
        description="Include complementary DA for nuclear/gas activities"
    )
    disclosure_format: Literal["article_8", "eba_pillar_3", "both"] = Field(
        default="article_8",
        description="Disclosure output format"
    )


class TaxonomyPhaseResult(BaseModel):
    """Result from a single pipeline phase."""

    phase: TaxonomyPipelinePhase
    status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = ""


class OrchestratorResult(BaseModel):
    """Complete pipeline execution result."""

    overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    phases: List[TaxonomyPhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""


class _AgentStub:
    """
    Stub for agent injection pattern.

    Real agent instances are injected at runtime by agent_loader.
    If no real agent is available, uses deterministic fallback.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._real_agent: Any = None
        logger.debug(f"Created agent stub for {agent_name}")

    def inject(self, real_agent: Any) -> None:
        """Inject real agent instance."""
        self._real_agent = real_agent
        logger.info(f"Injected real agent for {self.agent_name}")

    async def execute(self, method_name: str, **kwargs) -> Any:
        """Execute agent method (real or fallback)."""
        if self._real_agent and hasattr(self._real_agent, method_name):
            method = getattr(self._real_agent, method_name)
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            else:
                return method(**kwargs)
        else:
            logger.warning(
                f"Agent {self.agent_name} not available, using fallback for {method_name}"
            )
            return await self._fallback_execute(method_name, **kwargs)

    async def _fallback_execute(self, method_name: str, **kwargs) -> Any:
        """Deterministic fallback when real agent not available."""
        return {
            "status": "fallback",
            "agent": self.agent_name,
            "method": method_name,
            "message": f"Executed fallback for {self.agent_name}.{method_name}",
            "data": kwargs
        }


class TaxonomyPackOrchestrator:
    """
    PACK-008 EU Taxonomy Alignment Pack orchestrator.

    Coordinates 10-phase pipeline with GL-Taxonomy-APP engines, 30 MRV agents,
    10 data agents, and 10 foundation agents for comprehensive taxonomy alignment.
    Supports agent injection for flexible deployment.

    Example:
        >>> config = TaxonomyOrchestratorConfig(organization_type="non_financial_undertaking")
        >>> orchestrator = TaxonomyPackOrchestrator(config)
        >>> orchestrator.inject_agent("mrv_001_stationary_combustion", real_agent)
        >>> result = await orchestrator.execute({"activities": [...]})
    """

    def __init__(self, config: TaxonomyOrchestratorConfig):
        """Initialize orchestrator with agent stubs."""
        self.config = config
        self._agents: Dict[str, _AgentStub] = {}
        self._services: Dict[str, Any] = {}
        self._initialize_agent_stubs()
        logger.info("TaxonomyPackOrchestrator initialized with agent injection pattern")

    def _initialize_agent_stubs(self) -> None:
        """Create agent stubs for all required agents."""
        # MRV Scope 1 agents (001-008)
        scope1_agents = [
            "mrv_001_stationary_combustion",
            "mrv_002_refrigerants",
            "mrv_003_mobile_combustion",
            "mrv_004_process_emissions",
            "mrv_005_fugitive_emissions",
            "mrv_006_land_use",
            "mrv_007_waste_treatment",
            "mrv_008_agricultural"
        ]

        # MRV Scope 2 agents (009-013)
        scope2_agents = [
            "mrv_009_scope2_location",
            "mrv_010_scope2_market",
            "mrv_011_steam_heat",
            "mrv_012_cooling",
            "mrv_013_dual_reporting"
        ]

        # MRV Scope 3 agents (014-030)
        scope3_agents = [
            "mrv_014_purchased_goods",
            "mrv_015_capital_goods",
            "mrv_016_fuel_energy",
            "mrv_017_upstream_transport",
            "mrv_018_waste_generated",
            "mrv_019_business_travel",
            "mrv_020_employee_commuting",
            "mrv_021_upstream_leased",
            "mrv_022_downstream_transport",
            "mrv_023_processing_sold",
            "mrv_024_use_sold",
            "mrv_025_end_of_life",
            "mrv_026_downstream_leased",
            "mrv_027_franchises",
            "mrv_028_investments",
            "mrv_029_scope3_mapper",
            "mrv_030_audit_trail"
        ]

        # Data intake agents
        data_agents = [
            "data_001_pdf_extractor",
            "data_002_excel_normalizer",
            "data_003_erp_connector",
            "data_004_api_gateway",
            "data_008_questionnaire",
            "data_009_spend_categorizer",
            "data_010_quality_profiler",
            "data_018_lineage_tracker",
            "data_019_validation_rules"
        ]

        # Foundation agents
        foundation_agents = [
            "found_001_orchestrator",
            "found_002_schema_compiler",
            "found_003_unit_normalizer",
            "found_005_citations",
            "found_008_reproducibility",
            "found_009_qa_harness",
            "found_010_telemetry"
        ]

        # Taxonomy APP engines (mapped as stubs)
        taxonomy_engines = [
            "taxonomy_eligibility_engine",
            "taxonomy_sc_engine",
            "taxonomy_dnsh_engine",
            "taxonomy_ms_engine",
            "taxonomy_kpi_engine",
            "taxonomy_gar_engine",
            "taxonomy_tsc_engine",
            "taxonomy_transition_engine",
            "taxonomy_enabling_engine",
            "taxonomy_reporting_engine"
        ]

        all_agents = (
            scope1_agents + scope2_agents + scope3_agents +
            data_agents + foundation_agents + taxonomy_engines
        )

        for agent_name in all_agents:
            self._agents[agent_name] = _AgentStub(agent_name)

        logger.info(f"Initialized {len(self._agents)} agent stubs")

    def inject_agent(self, agent_name: str, real_agent: Any) -> None:
        """Inject real agent instance into stub."""
        if agent_name in self._agents:
            self._agents[agent_name].inject(real_agent)
        else:
            logger.warning(f"Unknown agent name: {agent_name}")

    def inject_service(self, service_name: str, service: Any) -> None:
        """Inject external service (database, cache, etc.)."""
        self._services[service_name] = service
        logger.info(f"Injected service: {service_name}")

    async def execute(self, data: Dict[str, Any]) -> OrchestratorResult:
        """
        Execute complete 10-phase taxonomy alignment pipeline.

        Args:
            data: Input data containing activities, financial data, and context

        Returns:
            Complete pipeline result with all phase results
        """
        start_time = datetime.utcnow()
        phases: List[TaxonomyPhaseResult] = []
        overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"

        logger.info("Starting PACK-008 Taxonomy Alignment pipeline")

        try:
            # Phase 1: Health Check
            phase_result = await self._phase_health_check()
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"
                logger.error("Health check failed, aborting pipeline")
                return self._create_result(phases, overall_status, start_time)

            # Phase 2: Configuration
            phase_result = await self._phase_configuration()
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"

            # Phase 3: Activity Inventory
            phase_result = await self._phase_activity_inventory(data)
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"

            # Phase 4: Eligibility Screening
            phase_result = await self._phase_eligibility_screening(data)
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"

            # Phase 5: Substantial Contribution Assessment
            phase_result = await self._phase_sc_assessment(data)
            phases.append(phase_result)
            if phase_result.status == "WARN" and overall_status == "PASS":
                overall_status = "WARN"

            # Phase 6: DNSH Assessment
            phase_result = await self._phase_dnsh_assessment(data)
            phases.append(phase_result)
            if phase_result.status == "WARN" and overall_status == "PASS":
                overall_status = "WARN"

            # Phase 7: Minimum Safeguards Verification
            phase_result = await self._phase_ms_verification(data)
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"

            # Phase 8: KPI/GAR Calculation
            phase_result = await self._phase_kpi_gar_calculation(data)
            phases.append(phase_result)
            if phase_result.status == "FAIL":
                overall_status = "FAIL"

            # Phase 9: Disclosure Generation
            phase_result = await self._phase_disclosure_generation(data)
            phases.append(phase_result)
            if phase_result.status == "WARN" and overall_status == "PASS":
                overall_status = "WARN"

            # Phase 10: Audit Trail
            phase_result = await self._phase_audit_trail(data, phases)
            phases.append(phase_result)

            logger.info(f"Pipeline completed with status: {overall_status}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            overall_status = "FAIL"

        return self._create_result(phases, overall_status, start_time)

    async def _phase_health_check(self) -> TaxonomyPhaseResult:
        """Phase 1: Health verification of all taxonomy components."""
        start = datetime.utcnow()
        logger.info("Phase 1: Health Check")

        try:
            health_data = await self._agents["found_010_telemetry"].execute(
                "check_system_health"
            )

            # Verify taxonomy-specific engines
            engine_checks = {}
            for engine_name in [
                "taxonomy_eligibility_engine", "taxonomy_sc_engine",
                "taxonomy_dnsh_engine", "taxonomy_ms_engine",
                "taxonomy_kpi_engine", "taxonomy_reporting_engine"
            ]:
                engine_result = await self._agents[engine_name].execute("health_check")
                engine_checks[engine_name] = engine_result.get("status", "fallback")

            health_data["engine_checks"] = engine_checks

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.HEALTH_CHECK,
                status="PASS",
                message="Health check passed",
                data=health_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(health_data)
            )

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.HEALTH_CHECK,
                status="FAIL",
                message=f"Health check failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_configuration(self) -> TaxonomyPhaseResult:
        """Phase 2: Configuration validation and setup."""
        start = datetime.utcnow()
        logger.info("Phase 2: Configuration")

        try:
            config_data = {
                "organization_type": self.config.organization_type,
                "environmental_objectives": self.config.environmental_objectives,
                "reporting_period_year": self.config.reporting_period_year,
                "delegated_act_version": self.config.delegated_act_version,
                "features": {
                    "gar_calculation": self.config.enable_gar_calculation,
                    "cross_framework": self.config.enable_cross_framework,
                    "capex_plan": self.config.enable_capex_plan,
                    "double_counting_prevention": self.config.double_counting_prevention,
                    "nuclear_gas": self.config.include_nuclear_gas
                },
                "disclosure_format": self.config.disclosure_format
            }

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.CONFIGURATION,
                status="PASS",
                message="Configuration validated",
                data=config_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(config_data)
            )

        except Exception as e:
            logger.error(f"Configuration failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.CONFIGURATION,
                status="FAIL",
                message=f"Configuration failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_activity_inventory(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 3: Build activity inventory with NACE code mapping."""
        start = datetime.utcnow()
        logger.info("Phase 3: Activity Inventory")

        try:
            activities = data.get("activities", [])

            # Map NACE codes to taxonomy activities
            inventory_result = await self._agents["taxonomy_eligibility_engine"].execute(
                "map_nace_to_taxonomy",
                activities=activities
            )

            # Validate schema
            schema_result = await self._agents["found_002_schema_compiler"].execute(
                "validate_activity_schema",
                activities=activities
            )

            inventory_data = {
                "total_activities": len(activities),
                "mapped_activities": inventory_result.get("mapped_count", 0),
                "unmapped_activities": inventory_result.get("unmapped_count", 0),
                "nace_sectors": inventory_result.get("sectors", []),
                "schema_valid": schema_result.get("valid", True)
            }

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.ACTIVITY_INVENTORY,
                status="PASS",
                message=f"Inventoried {len(activities)} activities",
                data=inventory_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(inventory_data)
            )

        except Exception as e:
            logger.error(f"Activity inventory failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.ACTIVITY_INVENTORY,
                status="FAIL",
                message=f"Activity inventory failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_eligibility_screening(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 4: Screen activities for taxonomy eligibility."""
        start = datetime.utcnow()
        logger.info("Phase 4: Eligibility Screening")

        try:
            activities = data.get("activities", [])

            screening_result = await self._agents["taxonomy_eligibility_engine"].execute(
                "screen_activities",
                activities=activities,
                objectives=self.config.environmental_objectives
            )

            eligible_count = screening_result.get("eligible_count", 0)
            total = len(activities) if activities else 1

            screening_data = {
                "total_screened": len(activities),
                "eligible": eligible_count,
                "not_eligible": len(activities) - eligible_count,
                "eligibility_ratio": eligible_count / total if total > 0 else 0.0,
                "by_objective": screening_result.get("by_objective", {}),
                "by_sector": screening_result.get("by_sector", {})
            }

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.ELIGIBILITY_SCREENING,
                status="PASS",
                message=f"{eligible_count}/{len(activities)} activities eligible",
                data=screening_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(screening_data)
            )

        except Exception as e:
            logger.error(f"Eligibility screening failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.ELIGIBILITY_SCREENING,
                status="FAIL",
                message=f"Eligibility screening failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_sc_assessment(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 5: Substantial Contribution assessment with TSC evaluation."""
        start = datetime.utcnow()
        logger.info("Phase 5: Substantial Contribution Assessment")

        try:
            # Evaluate TSC for each eligible activity
            sc_result = await self._agents["taxonomy_sc_engine"].execute(
                "assess_substantial_contribution",
                activities=data.get("activities", []),
                objectives=self.config.environmental_objectives
            )

            # Fetch emissions data from MRV agents for CCM/CCA TSC
            mrv_tasks = []
            if "CCM" in self.config.environmental_objectives:
                mrv_tasks.append(
                    self._agents["mrv_001_stationary_combustion"].execute("get_emissions_summary")
                )
                mrv_tasks.append(
                    self._agents["mrv_003_mobile_combustion"].execute("get_emissions_summary")
                )

            if self.config.parallel_processing and mrv_tasks:
                mrv_results = await asyncio.gather(*mrv_tasks, return_exceptions=True)
            elif mrv_tasks:
                mrv_results = [await task for task in mrv_tasks]
            else:
                mrv_results = []

            sc_data = {
                "activities_assessed": sc_result.get("assessed_count", 0),
                "sc_pass": sc_result.get("pass_count", 0),
                "sc_fail": sc_result.get("fail_count", 0),
                "enabling_activities": sc_result.get("enabling_count", 0),
                "transitional_activities": sc_result.get("transitional_count", 0),
                "mrv_data_available": len(mrv_results) > 0,
                "by_objective": sc_result.get("by_objective", {})
            }

            status = "PASS" if sc_data["sc_fail"] == 0 else "WARN"

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.SC_ASSESSMENT,
                status=status,
                message=f"{sc_data['sc_pass']} activities meet SC criteria",
                data=sc_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(sc_data)
            )

        except Exception as e:
            logger.error(f"SC assessment failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.SC_ASSESSMENT,
                status="FAIL",
                message=f"SC assessment failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_dnsh_assessment(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 6: Do No Significant Harm assessment across 6 objectives."""
        start = datetime.utcnow()
        logger.info("Phase 6: DNSH Assessment")

        try:
            dnsh_result = await self._agents["taxonomy_dnsh_engine"].execute(
                "assess_dnsh",
                activities=data.get("activities", []),
                objectives=self.config.environmental_objectives
            )

            # Fetch additional environmental data from MRV agents
            env_tasks = []
            if "WTR" in self.config.environmental_objectives:
                env_tasks.append(
                    self._agents["mrv_007_waste_treatment"].execute("get_water_metrics")
                )
            if "BIO" in self.config.environmental_objectives:
                env_tasks.append(
                    self._agents["mrv_006_land_use"].execute("get_biodiversity_metrics")
                )

            if self.config.parallel_processing and env_tasks:
                await asyncio.gather(*env_tasks, return_exceptions=True)

            dnsh_data = {
                "activities_assessed": dnsh_result.get("assessed_count", 0),
                "dnsh_pass": dnsh_result.get("pass_count", 0),
                "dnsh_fail": dnsh_result.get("fail_count", 0),
                "objective_results": {
                    "CCM": dnsh_result.get("ccm_pass", 0),
                    "CCA": dnsh_result.get("cca_pass", 0),
                    "WTR": dnsh_result.get("wtr_pass", 0),
                    "CE": dnsh_result.get("ce_pass", 0),
                    "PPC": dnsh_result.get("ppc_pass", 0),
                    "BIO": dnsh_result.get("bio_pass", 0)
                }
            }

            status = "PASS" if dnsh_data["dnsh_fail"] == 0 else "WARN"

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.DNSH_ASSESSMENT,
                status=status,
                message=f"{dnsh_data['dnsh_pass']} activities pass DNSH",
                data=dnsh_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(dnsh_data)
            )

        except Exception as e:
            logger.error(f"DNSH assessment failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.DNSH_ASSESSMENT,
                status="FAIL",
                message=f"DNSH assessment failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_ms_verification(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 7: Minimum Safeguards verification (4 topics)."""
        start = datetime.utcnow()
        logger.info("Phase 7: Minimum Safeguards Verification")

        try:
            ms_result = await self._agents["taxonomy_ms_engine"].execute(
                "verify_minimum_safeguards",
                activities=data.get("activities", []),
                organization_data=data.get("organization", {})
            )

            ms_data = {
                "human_rights": ms_result.get("human_rights", "not_assessed"),
                "anti_corruption": ms_result.get("anti_corruption", "not_assessed"),
                "taxation": ms_result.get("taxation", "not_assessed"),
                "fair_competition": ms_result.get("fair_competition", "not_assessed"),
                "overall_pass": ms_result.get("overall_pass", False),
                "evidence_completeness": ms_result.get("evidence_completeness", 0.0)
            }

            all_topics_pass = ms_data["overall_pass"]
            status = "PASS" if all_topics_pass else "FAIL"

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.MS_VERIFICATION,
                status=status,
                message=f"Minimum Safeguards: {'PASS' if all_topics_pass else 'FAIL'}",
                data=ms_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(ms_data)
            )

        except Exception as e:
            logger.error(f"MS verification failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.MS_VERIFICATION,
                status="FAIL",
                message=f"MS verification failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_kpi_gar_calculation(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 8: KPI calculation (Turnover/CapEx/OpEx) and GAR if applicable."""
        start = datetime.utcnow()
        logger.info("Phase 8: KPI/GAR Calculation")

        try:
            # KPI calculation for all organization types
            kpi_result = await self._agents["taxonomy_kpi_engine"].execute(
                "calculate_kpis",
                activities=data.get("activities", []),
                financial_data=data.get("financial_data", {}),
                prevent_double_counting=self.config.double_counting_prevention
            )

            kpi_data = {
                "turnover_ratio": kpi_result.get("turnover_ratio", 0.0),
                "capex_ratio": kpi_result.get("capex_ratio", 0.0),
                "opex_ratio": kpi_result.get("opex_ratio", 0.0),
                "eligible_turnover_ratio": kpi_result.get("eligible_turnover_ratio", 0.0),
                "eligible_capex_ratio": kpi_result.get("eligible_capex_ratio", 0.0),
                "eligible_opex_ratio": kpi_result.get("eligible_opex_ratio", 0.0)
            }

            # GAR calculation for financial institutions
            if self.config.enable_gar_calculation:
                gar_result = await self._agents["taxonomy_gar_engine"].execute(
                    "calculate_gar",
                    exposures=data.get("exposures", {}),
                    counterparty_data=data.get("counterparty_data", {})
                )

                kpi_data["gar_stock"] = gar_result.get("gar_stock", 0.0)
                kpi_data["gar_flow"] = gar_result.get("gar_flow", 0.0)
                kpi_data["btar"] = gar_result.get("btar", 0.0)

            # ERP financial data integration
            erp_data = await self._agents["data_003_erp_connector"].execute(
                "fetch_financial_summary",
                fiscal_year=self.config.reporting_period_year
            )
            kpi_data["erp_data_available"] = erp_data.get("status") != "fallback"

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.KPI_GAR_CALCULATION,
                status="PASS",
                message="KPI/GAR calculation completed",
                data=kpi_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(kpi_data)
            )

        except Exception as e:
            logger.error(f"KPI/GAR calculation failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.KPI_GAR_CALCULATION,
                status="FAIL",
                message=f"KPI/GAR calculation failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_disclosure_generation(self, data: Dict[str, Any]) -> TaxonomyPhaseResult:
        """Phase 9: Generate Article 8 / EBA Pillar 3 disclosures."""
        start = datetime.utcnow()
        logger.info("Phase 9: Disclosure Generation")

        try:
            disclosure_result = await self._agents["taxonomy_reporting_engine"].execute(
                "generate_disclosures",
                format=self.config.disclosure_format,
                reporting_year=self.config.reporting_period_year,
                include_nuclear_gas=self.config.include_nuclear_gas
            )

            disclosure_data = {
                "format": self.config.disclosure_format,
                "templates_generated": disclosure_result.get("templates_count", 0),
                "article_8_tables": disclosure_result.get("article_8_tables", []),
                "eba_templates": disclosure_result.get("eba_templates", []),
                "year_over_year": disclosure_result.get("yoy_available", False)
            }

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.DISCLOSURE_GENERATION,
                status="PASS",
                message=f"Generated {disclosure_data['templates_generated']} disclosure templates",
                data=disclosure_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(disclosure_data)
            )

        except Exception as e:
            logger.error(f"Disclosure generation failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.DISCLOSURE_GENERATION,
                status="WARN",
                message=f"Disclosure generation failed: {str(e)}",
                duration_seconds=duration
            )

    async def _phase_audit_trail(
        self, data: Dict[str, Any], phases: List[TaxonomyPhaseResult]
    ) -> TaxonomyPhaseResult:
        """Phase 10: Complete audit trail with provenance hashing."""
        start = datetime.utcnow()
        logger.info("Phase 10: Audit Trail")

        try:
            # Collect all phase hashes
            phase_hashes = {p.phase.value: p.provenance_hash for p in phases}

            audit_result = await self._agents["found_008_reproducibility"].execute(
                "record_audit_trail",
                pipeline_execution="pack_008_taxonomy_alignment",
                phase_hashes=phase_hashes
            )

            # Citations for regulatory references
            citation_result = await self._agents["found_005_citations"].execute(
                "record_citations",
                regulation="EU_Taxonomy_2020_852",
                delegated_acts=[
                    "Climate_DA_2021_2139",
                    "Environmental_DA_2023_2486",
                    "Disclosures_DA_2021_2178"
                ]
            )

            audit_data = {
                "phase_count": len(phases),
                "phase_hashes": phase_hashes,
                "audit_id": audit_result.get("audit_id", ""),
                "citations_recorded": citation_result.get("count", 0),
                "pipeline_hash": self._calculate_hash(phase_hashes)
            }

            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.AUDIT_TRAIL,
                status="PASS",
                message="Audit trail recorded",
                data=audit_data,
                duration_seconds=duration,
                provenance_hash=self._calculate_hash(audit_data)
            )

        except Exception as e:
            logger.error(f"Audit trail failed: {str(e)}")
            duration = (datetime.utcnow() - start).total_seconds()
            return TaxonomyPhaseResult(
                phase=TaxonomyPipelinePhase.AUDIT_TRAIL,
                status="WARN",
                message=f"Audit trail failed: {str(e)}",
                duration_seconds=duration
            )

    def _create_result(
        self,
        phases: List[TaxonomyPhaseResult],
        overall_status: Literal["PASS", "WARN", "FAIL"],
        start_time: datetime
    ) -> OrchestratorResult:
        """Create final pipeline result."""
        total_duration = (datetime.utcnow() - start_time).total_seconds()

        summary = {
            "total_phases": len(phases),
            "passed_phases": sum(1 for p in phases if p.status == "PASS"),
            "warned_phases": sum(1 for p in phases if p.status == "WARN"),
            "failed_phases": sum(1 for p in phases if p.status == "FAIL"),
            "organization_type": self.config.organization_type,
            "environmental_objectives": self.config.environmental_objectives,
            "reporting_year": self.config.reporting_period_year
        }

        result = OrchestratorResult(
            overall_status=overall_status,
            phases=phases,
            total_duration_seconds=total_duration,
            summary=summary
        )

        result.provenance_hash = self._calculate_hash(result.model_dump())

        return result

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
