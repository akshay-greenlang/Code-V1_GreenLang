"""
PACK-013 CSRD Manufacturing Pack - Setup Wizard.

8-step guided setup for manufacturing-specific CSRD compliance.
Walks the user through company profiling, sub-sector selection,
facility registration, regulation mapping, data source connection,
baseline calculation, target setting, and workflow activation.
"""

import hashlib
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SetupStep(str, Enum):
    """The 8 setup wizard steps."""
    COMPANY_PROFILE = "company_profile"
    SUB_SECTOR_SELECTION = "sub_sector_selection"
    FACILITY_REGISTRATION = "facility_registration"
    REGULATION_MAPPING = "regulation_mapping"
    DATA_SOURCE_CONNECTION = "data_source_connection"
    BASELINE_CALCULATION = "baseline_calculation"
    TARGET_SETTING = "target_setting"
    WORKFLOW_ACTIVATION = "workflow_activation"


class SubSector(str, Enum):
    """Manufacturing sub-sectors supported by the pack."""
    CEMENT = "cement"
    STEEL = "steel"
    CHEMICALS = "chemicals"
    AUTOMOTIVE = "automotive"
    FOOD_BEVERAGE = "food_beverage"
    ELECTRONICS = "electronics"
    PAPER_PULP = "paper_pulp"
    TEXTILES = "textiles"
    GLASS = "glass"
    CERAMICS = "ceramics"
    PHARMACEUTICALS = "pharmaceuticals"
    AEROSPACE = "aerospace"
    GENERAL = "general"


class ERPSystem(str, Enum):
    """Supported ERP systems."""
    SAP = "sap"
    ORACLE = "oracle"
    WORKDAY = "workday"
    DYNAMICS = "dynamics"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """Company profile collected in step 1."""
    company_name: str = Field(default="")
    legal_entity_id: str = Field(default="")
    country_code: str = Field(default="DE")
    industry_nace: str = Field(default="C")
    employee_count: int = Field(default=0, ge=0)
    annual_revenue_eur: float = Field(default=0.0, ge=0.0)
    csrd_in_scope: bool = Field(default=True)
    reporting_year: int = Field(default=2025)
    contact_email: str = Field(default="")


class FacilityData(BaseModel):
    """Facility data collected in step 3."""
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    address: str = Field(default="")
    country_code: str = Field(default="DE")
    sub_sector: SubSector = Field(default=SubSector.GENERAL)
    production_capacity_tonnes: float = Field(default=0.0, ge=0.0)
    eu_ets_covered: bool = Field(default=False)
    ets_installation_id: Optional[str] = Field(default=None)
    ied_permit: bool = Field(default=False)
    seveso_site: bool = Field(default=False)
    water_stress_area: bool = Field(default=False)


class DataSourceConfig(BaseModel):
    """Data source configuration from step 5."""
    erp_system: ERPSystem = Field(default=ERPSystem.NONE)
    erp_connection_string: str = Field(default="")
    mes_enabled: bool = Field(default=False)
    scada_enabled: bool = Field(default=False)
    manual_upload: bool = Field(default=True)
    api_endpoint: str = Field(default="")
    file_formats: List[str] = Field(
        default_factory=lambda: ["xlsx", "csv"]
    )


class BaselineData(BaseModel):
    """Baseline calculation results from step 6."""
    baseline_year: int = Field(default=2019)
    scope1_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_baseline_tco2e: float = Field(default=0.0, ge=0.0)
    energy_baseline_mwh: float = Field(default=0.0, ge=0.0)
    water_baseline_m3: float = Field(default=0.0, ge=0.0)
    waste_baseline_kg: float = Field(default=0.0, ge=0.0)
    production_baseline_tonnes: float = Field(default=0.0, ge=0.0)


class TargetData(BaseModel):
    """Target setting data from step 7."""
    target_year: int = Field(default=2030)
    scope1_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    scope2_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    scope3_reduction_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    energy_efficiency_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    renewable_energy_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    water_reduction_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    waste_reduction_pct: float = Field(default=30.0, ge=0.0, le=100.0)
    sbti_aligned: bool = Field(default=False)
    net_zero_year: Optional[int] = Field(default=2050)


class WorkflowConfig(BaseModel):
    """Workflow activation configuration from step 8."""
    annual_reporting: bool = Field(default=True)
    quarterly_review: bool = Field(default=True)
    monthly_data_collection: bool = Field(default=True)
    bat_assessment_schedule: str = Field(default="annual")
    cbam_quarterly_reporting: bool = Field(default=False)
    ets_compliance_tracking: bool = Field(default=False)
    taxonomy_reporting: bool = Field(default=True)
    notifications_enabled: bool = Field(default=True)
    notification_emails: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Wizard state and result
# ---------------------------------------------------------------------------

class WizardState(BaseModel):
    """Current state of the setup wizard."""
    current_step: SetupStep = Field(default=SetupStep.COMPANY_PROFILE)
    completed_steps: List[SetupStep] = Field(default_factory=list)
    company_data: CompanyProfile = Field(
        default_factory=CompanyProfile
    )
    facility_data: List[FacilityData] = Field(default_factory=list)
    sub_sector: SubSector = Field(default=SubSector.GENERAL)
    regulations: List[str] = Field(default_factory=list)
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig
    )
    baseline: BaselineData = Field(default_factory=BaselineData)
    targets: TargetData = Field(default_factory=TargetData)
    workflows: WorkflowConfig = Field(default_factory=WorkflowConfig)
    config: Dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.time)


class WizardResult(BaseModel):
    """Final result after completing all setup steps."""
    setup_complete: bool = Field(default=False)
    generated_config: Dict[str, Any] = Field(default_factory=dict)
    recommended_preset: str = Field(default="general")
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig
    )
    baseline_summary: Dict[str, Any] = Field(default_factory=dict)
    target_summary: Dict[str, Any] = Field(default_factory=dict)
    applicable_regulations: List[str] = Field(default_factory=list)
    validation_issues: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    duration_seconds: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Step order and dependencies
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupStep] = [
    SetupStep.COMPANY_PROFILE,
    SetupStep.SUB_SECTOR_SELECTION,
    SetupStep.FACILITY_REGISTRATION,
    SetupStep.REGULATION_MAPPING,
    SetupStep.DATA_SOURCE_CONNECTION,
    SetupStep.BASELINE_CALCULATION,
    SetupStep.TARGET_SETTING,
    SetupStep.WORKFLOW_ACTIVATION,
]

STEP_DEPENDENCIES: Dict[SetupStep, List[SetupStep]] = {
    SetupStep.COMPANY_PROFILE: [],
    SetupStep.SUB_SECTOR_SELECTION: [SetupStep.COMPANY_PROFILE],
    SetupStep.FACILITY_REGISTRATION: [SetupStep.SUB_SECTOR_SELECTION],
    SetupStep.REGULATION_MAPPING: [SetupStep.FACILITY_REGISTRATION],
    SetupStep.DATA_SOURCE_CONNECTION: [SetupStep.REGULATION_MAPPING],
    SetupStep.BASELINE_CALCULATION: [SetupStep.DATA_SOURCE_CONNECTION],
    SetupStep.TARGET_SETTING: [SetupStep.BASELINE_CALCULATION],
    SetupStep.WORKFLOW_ACTIVATION: [SetupStep.TARGET_SETTING],
}

# Regulation auto-detection rules
REGULATION_RULES: Dict[str, Dict[str, Any]] = {
    "CSRD": {
        "condition": "employee_count >= 250 or annual_revenue_eur >= 40_000_000",
        "description": "Corporate Sustainability Reporting Directive",
    },
    "EU_ETS": {
        "condition": "eu_ets_covered",
        "description": "EU Emissions Trading System",
    },
    "CBAM": {
        "condition": "sub_sector in ['cement', 'steel', 'aluminium', 'chemicals']",
        "description": "Carbon Border Adjustment Mechanism",
    },
    "IED": {
        "condition": "ied_permit",
        "description": "Industrial Emissions Directive",
    },
    "EU_TAXONOMY": {
        "condition": "csrd_in_scope",
        "description": "EU Taxonomy for Sustainable Activities",
    },
    "SEVESO_III": {
        "condition": "seveso_site",
        "description": "Control of major-accident hazards (Seveso III)",
    },
    "WFD": {
        "condition": "water_stress_area",
        "description": "Water Framework Directive",
    },
}

# Sub-sector preset recommendations
SUB_SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    SubSector.CEMENT.value: {
        "preset": "cement_intensive",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-005"],
        "esrs_focus": ["E1", "E2", "E3"],
        "bat_reference": "cement_lime_magnesia",
    },
    SubSector.STEEL.value: {
        "preset": "steel_intensive",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-005", "MRV-011"],
        "esrs_focus": ["E1", "E2", "E5"],
        "bat_reference": "iron_steel",
    },
    SubSector.CHEMICALS.value: {
        "preset": "chemicals_intensive",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-002", "MRV-005"],
        "esrs_focus": ["E1", "E2", "E3", "E5"],
        "bat_reference": "large_volume_organic",
    },
    SubSector.AUTOMOTIVE.value: {
        "preset": "automotive_assembly",
        "primary_agents": ["MRV-001", "MRV-003", "MRV-009", "MRV-014"],
        "esrs_focus": ["E1", "E5"],
        "bat_reference": "surface_treatment",
    },
    SubSector.FOOD_BEVERAGE.value: {
        "preset": "food_processing",
        "primary_agents": ["MRV-001", "MRV-008", "MRV-007", "MRV-009"],
        "esrs_focus": ["E1", "E3", "E5"],
        "bat_reference": "food_drink_milk",
    },
    SubSector.ELECTRONICS.value: {
        "preset": "electronics_assembly",
        "primary_agents": ["MRV-001", "MRV-002", "MRV-009", "MRV-014"],
        "esrs_focus": ["E1", "E2", "E5"],
        "bat_reference": "surface_treatment",
    },
    SubSector.PAPER_PULP.value: {
        "preset": "paper_pulp_intensive",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-009", "MRV-007"],
        "esrs_focus": ["E1", "E3", "E5"],
        "bat_reference": "pulp_paper",
    },
    SubSector.TEXTILES.value: {
        "preset": "textiles_processing",
        "primary_agents": ["MRV-001", "MRV-009", "MRV-014", "MRV-007"],
        "esrs_focus": ["E1", "E2", "E3"],
        "bat_reference": "textiles",
    },
    SubSector.GLASS.value: {
        "preset": "glass_manufacturing",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-009"],
        "esrs_focus": ["E1", "E2"],
        "bat_reference": "glass",
    },
    SubSector.CERAMICS.value: {
        "preset": "ceramics_manufacturing",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-009"],
        "esrs_focus": ["E1", "E2"],
        "bat_reference": "ceramics",
    },
    SubSector.PHARMACEUTICALS.value: {
        "preset": "pharma_manufacturing",
        "primary_agents": ["MRV-001", "MRV-002", "MRV-009", "MRV-007"],
        "esrs_focus": ["E1", "E2", "E3"],
        "bat_reference": "large_volume_organic",
    },
    SubSector.AEROSPACE.value: {
        "preset": "aerospace_assembly",
        "primary_agents": ["MRV-001", "MRV-009", "MRV-014"],
        "esrs_focus": ["E1", "E5"],
        "bat_reference": "surface_treatment",
    },
    SubSector.GENERAL.value: {
        "preset": "general_manufacturing",
        "primary_agents": ["MRV-001", "MRV-004", "MRV-009"],
        "esrs_focus": ["E1", "E2", "E3", "E5"],
        "bat_reference": "general",
    },
}


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class ManufacturingSetupWizard:
    """
    8-step guided setup for CSRD manufacturing compliance.

    Walks through company profiling, sub-sector selection, facility
    registration, regulation mapping, data-source connection, baseline
    calculation, target setting, and workflow activation.
    """

    def __init__(self) -> None:
        self._state: Optional[WizardState] = None
        self._step_handlers: Dict[SetupStep, Any] = {
            SetupStep.COMPANY_PROFILE: self._handle_company_profile,
            SetupStep.SUB_SECTOR_SELECTION: self._handle_sub_sector,
            SetupStep.FACILITY_REGISTRATION: self._handle_facility,
            SetupStep.REGULATION_MAPPING: self._handle_regulation,
            SetupStep.DATA_SOURCE_CONNECTION: self._handle_data_source,
            SetupStep.BASELINE_CALCULATION: self._handle_baseline,
            SetupStep.TARGET_SETTING: self._handle_targets,
            SetupStep.WORKFLOW_ACTIVATION: self._handle_workflows,
        }

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- public API ----------------------------------------------------------

    def start_setup(self) -> WizardState:
        """Begin a new setup wizard session."""
        self._state = WizardState()
        logger.info("Setup wizard started")
        return self._state

    def complete_step(
        self, step: SetupStep, data: Dict[str, Any]
    ) -> WizardState:
        """
        Complete a setup step with the provided data.

        Args:
            step: The setup step to complete.
            data: Data for this step.

        Returns:
            Updated WizardState.

        Raises:
            ValueError: If dependencies are not met.
        """
        if self._state is None:
            self._state = WizardState()

        # Check dependencies
        deps = STEP_DEPENDENCIES.get(step, [])
        for dep in deps:
            if dep not in self._state.completed_steps:
                raise ValueError(
                    f"Step {step.value} requires {dep.value} to be "
                    f"completed first"
                )

        # Already completed?  Allow re-doing.
        handler = self._step_handlers.get(step)
        if handler is None:
            raise ValueError(f"Unknown step: {step.value}")

        handler(data)

        if step not in self._state.completed_steps:
            self._state.completed_steps.append(step)

        # Advance current_step to next incomplete step
        for s in STEP_ORDER:
            if s not in self._state.completed_steps:
                self._state.current_step = s
                break
        else:
            self._state.current_step = step  # All done

        return self._state

    def generate_config(
        self, state: Optional[WizardState] = None
    ) -> Dict[str, Any]:
        """Generate the full pack configuration from wizard state."""
        st = state or self._state
        if st is None:
            return {}

        preset_info = SUB_SECTOR_PRESETS.get(
            st.sub_sector.value,
            SUB_SECTOR_PRESETS[SubSector.GENERAL.value],
        )

        config: Dict[str, Any] = {
            "pack_id": "PACK-013",
            "pack_name": "CSRD Manufacturing Pack",
            "company": st.company_data.model_dump(),
            "sub_sector": st.sub_sector.value,
            "preset": preset_info["preset"],
            "facilities": [f.model_dump() for f in st.facility_data],
            "regulations": st.regulations,
            "data_sources": st.data_sources.model_dump(),
            "baseline": st.baseline.model_dump(),
            "targets": st.targets.model_dump(),
            "workflows": st.workflows.model_dump(),
            "agents": {
                "primary": preset_info["primary_agents"],
                "esrs_focus": preset_info["esrs_focus"],
                "bat_reference": preset_info["bat_reference"],
            },
            "orchestrator": {
                "phases_enabled": {
                    p.value: True for p in
                    __import__("packs.eu_compliance.PACK_013_csrd_manufacturing.integrations.pack_orchestrator", fromlist=["PipelinePhase"]).PipelinePhase
                } if False else {
                    "initialization": True,
                    "data_intake": True,
                    "quality_assurance": True,
                    "process_emissions": True,
                    "energy_analysis": True,
                    "product_pcf": True,
                    "circular_economy": True,
                    "water_pollution": True,
                    "bat_compliance": True,
                    "supply_chain": True,
                    "reporting": True,
                },
                "parallel_execution": False,
                "timeout_per_phase": 300_000,
            },
            "cbam_enabled": "CBAM" in st.regulations,
            "ets_enabled": "EU_ETS" in st.regulations,
            "taxonomy_enabled": "EU_TAXONOMY" in st.regulations,
        }
        return config

    def recommend_preset(
        self, company_data: Dict[str, Any]
    ) -> str:
        """
        Recommend a sub-sector preset based on company data.

        Uses NACE code, product types, and employee count to determine
        the best-fit manufacturing preset.
        """
        nace = company_data.get("industry_nace", "C")

        nace_to_preset: Dict[str, str] = {
            "C23.5": SubSector.CEMENT.value,
            "C24.1": SubSector.STEEL.value,
            "C24.4": SubSector.STEEL.value,
            "C20": SubSector.CHEMICALS.value,
            "C29": SubSector.AUTOMOTIVE.value,
            "C10": SubSector.FOOD_BEVERAGE.value,
            "C11": SubSector.FOOD_BEVERAGE.value,
            "C26": SubSector.ELECTRONICS.value,
            "C27": SubSector.ELECTRONICS.value,
            "C17": SubSector.PAPER_PULP.value,
            "C13": SubSector.TEXTILES.value,
            "C14": SubSector.TEXTILES.value,
            "C23.1": SubSector.GLASS.value,
            "C23.3": SubSector.CERAMICS.value,
            "C23.4": SubSector.CERAMICS.value,
            "C21": SubSector.PHARMACEUTICALS.value,
            "C30.3": SubSector.AEROSPACE.value,
        }

        # Try exact match first, then match keys that are a prefix of
        # the input NACE code (longest prefix wins).
        if nace in nace_to_preset:
            return nace_to_preset[nace]

        # Find all matching prefixes, pick the longest
        matches = [
            (key, val) for key, val in nace_to_preset.items()
            if nace.startswith(key)
        ]
        if matches:
            best_key, best_val = max(matches, key=lambda kv: len(kv[0]))
            logger.info(
                "Recommended preset for NACE %s: %s (matched %s)",
                nace, best_val, best_key,
            )
            return best_val

        # Fall back: try progressively shorter input prefixes
        for prefix_len in [5, 4, 3, 2]:
            prefix = nace[:prefix_len]
            if prefix in nace_to_preset:
                preset = nace_to_preset[prefix]
                logger.info(
                    "Recommended preset for NACE %s: %s",
                    nace, preset,
                )
                return preset

        return SubSector.GENERAL.value

    def validate_setup(
        self, state: Optional[WizardState] = None
    ) -> List[str]:
        """
        Validate the current setup state and return a list of issues.

        Returns an empty list if setup is valid.
        """
        st = state or self._state
        if st is None:
            return ["Wizard has not been started"]

        issues: List[str] = []

        # Check all steps completed
        for step in STEP_ORDER:
            if step not in st.completed_steps:
                issues.append(f"Step {step.value} not completed")

        # Company profile validation
        if not st.company_data.company_name:
            issues.append("Company name is required")
        if st.company_data.employee_count == 0:
            issues.append(
                "Employee count should be set for CSRD scope check"
            )

        # Facility validation
        if not st.facility_data:
            issues.append("At least one facility must be registered")
        for i, fac in enumerate(st.facility_data):
            if not fac.facility_name:
                issues.append(f"Facility {i}: name is required")
            if fac.eu_ets_covered and not fac.ets_installation_id:
                issues.append(
                    f"Facility {i}: ETS installation ID required "
                    f"when EU ETS covered"
                )

        # Baseline validation
        if st.baseline.baseline_year >= st.company_data.reporting_year:
            issues.append(
                "Baseline year must be before the reporting year"
            )

        # Target validation
        if st.targets.target_year <= st.company_data.reporting_year:
            issues.append(
                "Target year must be after the reporting year"
            )

        # Data source validation
        if (
            st.data_sources.erp_system != ERPSystem.NONE
            and not st.data_sources.erp_connection_string
        ):
            issues.append(
                "ERP connection string required when ERP system selected"
            )

        return issues

    def finalize(
        self, state: Optional[WizardState] = None
    ) -> WizardResult:
        """
        Finalize the setup and generate the complete result.

        Validates the setup and generates the pack configuration.
        """
        st = state or self._state
        if st is None:
            return WizardResult(
                setup_complete=False,
                validation_issues=["Wizard not started"],
            )

        issues = self.validate_setup(st)
        config = self.generate_config(st)

        preset_info = SUB_SECTOR_PRESETS.get(
            st.sub_sector.value,
            SUB_SECTOR_PRESETS[SubSector.GENERAL.value],
        )

        elapsed = time.time() - st.started_at

        data = {
            "config": config,
            "issues": issues,
            "preset": preset_info["preset"],
        }

        return WizardResult(
            setup_complete=len(issues) == 0,
            generated_config=config,
            recommended_preset=preset_info["preset"],
            data_sources=st.data_sources,
            baseline_summary={
                "baseline_year": st.baseline.baseline_year,
                "scope1_tco2e": st.baseline.scope1_baseline_tco2e,
                "scope2_tco2e": st.baseline.scope2_baseline_tco2e,
                "scope3_tco2e": st.baseline.scope3_baseline_tco2e,
                "total_tco2e": (
                    st.baseline.scope1_baseline_tco2e
                    + st.baseline.scope2_baseline_tco2e
                    + st.baseline.scope3_baseline_tco2e
                ),
            },
            target_summary={
                "target_year": st.targets.target_year,
                "scope1_reduction_pct": st.targets.scope1_reduction_pct,
                "scope2_reduction_pct": st.targets.scope2_reduction_pct,
                "scope3_reduction_pct": st.targets.scope3_reduction_pct,
                "sbti_aligned": st.targets.sbti_aligned,
            },
            applicable_regulations=st.regulations,
            validation_issues=issues,
            provenance_hash=self._compute_hash(data),
            duration_seconds=round(elapsed, 2),
        )

    # -----------------------------------------------------------------------
    # Step handlers
    # -----------------------------------------------------------------------

    def _handle_company_profile(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 1: Company Profile."""
        assert self._state is not None
        self._state.company_data = CompanyProfile(**data)
        logger.info(
            "Company profile set: %s",
            self._state.company_data.company_name,
        )

    def _handle_sub_sector(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 2: Sub-Sector Selection."""
        assert self._state is not None
        sub_sector_val = data.get("sub_sector", "general")
        try:
            self._state.sub_sector = SubSector(sub_sector_val)
        except ValueError:
            self._state.sub_sector = SubSector.GENERAL
            logger.warning(
                "Unknown sub-sector %s; defaulting to general",
                sub_sector_val,
            )
        logger.info("Sub-sector selected: %s", self._state.sub_sector.value)

    def _handle_facility(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 3: Facility Registration."""
        assert self._state is not None
        facilities = data.get("facilities", [data])
        self._state.facility_data = []
        for fac_data in facilities:
            # Inject sub-sector if not provided
            if "sub_sector" not in fac_data:
                fac_data["sub_sector"] = self._state.sub_sector.value
            self._state.facility_data.append(FacilityData(**fac_data))
        logger.info(
            "Registered %d facilities", len(self._state.facility_data)
        )

    def _handle_regulation(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 4: Regulation Mapping."""
        assert self._state is not None
        # Auto-detect applicable regulations
        regs: Set[str] = set(data.get("regulations", []))

        # Apply auto-detection rules
        cp = self._state.company_data
        for reg_name, rule in REGULATION_RULES.items():
            if reg_name == "CSRD":
                if cp.employee_count >= 250 or cp.annual_revenue_eur >= 40_000_000:
                    regs.add("CSRD")
            elif reg_name == "EU_TAXONOMY":
                if cp.csrd_in_scope:
                    regs.add("EU_TAXONOMY")
            elif reg_name == "CBAM":
                cbam_sectors = {"cement", "steel", "chemicals"}
                if self._state.sub_sector.value in cbam_sectors:
                    regs.add("CBAM")
            elif reg_name == "EU_ETS":
                if any(f.eu_ets_covered for f in self._state.facility_data):
                    regs.add("EU_ETS")
            elif reg_name == "IED":
                if any(f.ied_permit for f in self._state.facility_data):
                    regs.add("IED")
            elif reg_name == "SEVESO_III":
                if any(f.seveso_site for f in self._state.facility_data):
                    regs.add("SEVESO_III")
            elif reg_name == "WFD":
                if any(f.water_stress_area for f in self._state.facility_data):
                    regs.add("WFD")

        self._state.regulations = sorted(regs)
        logger.info(
            "Applicable regulations: %s", self._state.regulations
        )

    def _handle_data_source(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 5: Data Source Connection."""
        assert self._state is not None
        self._state.data_sources = DataSourceConfig(**data)
        logger.info(
            "Data sources configured: ERP=%s, MES=%s, SCADA=%s",
            self._state.data_sources.erp_system.value,
            self._state.data_sources.mes_enabled,
            self._state.data_sources.scada_enabled,
        )

    def _handle_baseline(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 6: Baseline Calculation."""
        assert self._state is not None
        self._state.baseline = BaselineData(**data)
        logger.info(
            "Baseline set: year=%d, Scope1=%.2f, Scope2=%.2f tCO2e",
            self._state.baseline.baseline_year,
            self._state.baseline.scope1_baseline_tco2e,
            self._state.baseline.scope2_baseline_tco2e,
        )

    def _handle_targets(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 7: Target Setting."""
        assert self._state is not None
        self._state.targets = TargetData(**data)
        logger.info(
            "Targets set: year=%d, S1 reduction=%.1f%%",
            self._state.targets.target_year,
            self._state.targets.scope1_reduction_pct,
        )

    def _handle_workflows(
        self, data: Dict[str, Any]
    ) -> None:
        """Handle step 8: Workflow Activation."""
        assert self._state is not None
        self._state.workflows = WorkflowConfig(**data)
        logger.info(
            "Workflows activated: annual=%s, quarterly=%s",
            self._state.workflows.annual_reporting,
            self._state.workflows.quarterly_review,
        )
