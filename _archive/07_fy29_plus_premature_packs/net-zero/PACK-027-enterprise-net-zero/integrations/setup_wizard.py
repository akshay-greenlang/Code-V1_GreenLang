# -*- coding: utf-8 -*-
"""
EnterpriseSetupWizard - 8-Step Enterprise Onboarding for PACK-027
=====================================================================

This module implements an 8-step enterprise onboarding wizard for
organizations with 250+ employees, $50M+ revenue, and complex
multi-entity structures. More comprehensive than PACK-026's 5-step
SME wizard, covering ERP integration, multi-entity setup, SBTi
pathway selection, and sector-specific preset configuration.

Wizard Steps (8):
    1. organization_profile     -- Corporate details, sector, size
    2. entity_hierarchy         -- Multi-entity structure (100+ entities)
    3. erp_integration          -- SAP/Oracle/Workday connection
    4. consolidation_approach   -- Financial/operational/equity share
    5. scope_configuration      -- Scope 1+2+3 coverage selection
    6. sbti_pathway             -- ACA/SDA/FLAG/Mixed pathway selection
    7. sector_preset            -- 8 sector-specific presets
    8. assurance_readiness      -- Assurance level and provider selection

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnterpriseWizardStep(str, Enum):
    ORGANIZATION_PROFILE = "organization_profile"
    ENTITY_HIERARCHY = "entity_hierarchy"
    ERP_INTEGRATION = "erp_integration"
    CONSOLIDATION_APPROACH = "consolidation_approach"
    SCOPE_CONFIGURATION = "scope_configuration"
    SBTI_PATHWAY = "sbti_pathway"
    SECTOR_PRESET = "sector_preset"
    ASSURANCE_READINESS = "assurance_readiness"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ERPSystem(str, Enum):
    SAP_S4HANA = "sap_s4hana"
    ORACLE_ERP_CLOUD = "oracle_erp_cloud"
    WORKDAY = "workday"
    MICROSOFT_D365 = "microsoft_d365"
    OTHER = "other"
    NONE = "none"

class SectorPreset(str, Enum):
    MANUFACTURING = "manufacturing_enterprise"
    ENERGY_UTILITIES = "energy_utilities"
    FINANCIAL_SERVICES = "financial_services"
    TECHNOLOGY = "technology"
    CONSUMER_GOODS = "consumer_goods"
    TRANSPORT_LOGISTICS = "transport_logistics"
    REAL_ESTATE = "real_estate"
    HEALTHCARE_PHARMA = "healthcare_pharma"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EnterpriseOrgProfile(BaseModel):
    organization_name: str = Field(..., min_length=1, max_length=500)
    sector: str = Field(default="manufacturing")
    sub_sector: str = Field(default="")
    headquarters_country: str = Field(default="US")
    operating_countries: List[str] = Field(default_factory=lambda: ["US"])
    employee_count: int = Field(default=5000, ge=250)
    annual_revenue_usd: float = Field(default=500_000_000.0, ge=50_000_000)
    fiscal_year_end: str = Field(default="12-31")
    publicly_listed: bool = Field(default=True)
    stock_exchange: str = Field(default="")
    parent_company: str = Field(default="")

class EntityHierarchySetup(BaseModel):
    total_entities: int = Field(default=50, ge=1, le=500)
    countries: int = Field(default=10)
    wholly_owned: int = Field(default=30)
    joint_ventures: int = Field(default=5)
    associates: int = Field(default=3)
    franchises: int = Field(default=0)

class ERPSetup(BaseModel):
    primary_erp: ERPSystem = Field(default=ERPSystem.SAP_S4HANA)
    secondary_erps: List[ERPSystem] = Field(default_factory=list)
    hcm_system: ERPSystem = Field(default=ERPSystem.WORKDAY)
    connected: bool = Field(default=False)
    company_codes: List[str] = Field(default_factory=list)

class WizardStepState(BaseModel):
    name: EnterpriseWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)

class WizardState(BaseModel):
    wizard_id: str = Field(default="")
    current_step: EnterpriseWizardStep = Field(
        default=EnterpriseWizardStep.ORGANIZATION_PROFILE
    )
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class EnterpriseSetupResult(BaseModel):
    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    preset: str = Field(default="")
    entity_count: int = Field(default=0)
    erp_system: str = Field(default="")
    consolidation_approach: str = Field(default="")
    sbti_pathway: str = Field(default="")
    scope3_categories: List[int] = Field(default_factory=list)
    assurance_level: str = Field(default="")
    assurance_provider: str = Field(default="")
    carbon_price_per_tco2e: float = Field(default=100.0)
    base_year: int = Field(default=2023)
    target_year: int = Field(default=2030)
    engines_enabled: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[EnterpriseWizardStep] = [
    EnterpriseWizardStep.ORGANIZATION_PROFILE,
    EnterpriseWizardStep.ENTITY_HIERARCHY,
    EnterpriseWizardStep.ERP_INTEGRATION,
    EnterpriseWizardStep.CONSOLIDATION_APPROACH,
    EnterpriseWizardStep.SCOPE_CONFIGURATION,
    EnterpriseWizardStep.SBTI_PATHWAY,
    EnterpriseWizardStep.SECTOR_PRESET,
    EnterpriseWizardStep.ASSURANCE_READINESS,
]

STEP_DISPLAY_NAMES: Dict[EnterpriseWizardStep, str] = {
    EnterpriseWizardStep.ORGANIZATION_PROFILE: "Corporate Profile",
    EnterpriseWizardStep.ENTITY_HIERARCHY: "Entity Structure",
    EnterpriseWizardStep.ERP_INTEGRATION: "ERP System Integration",
    EnterpriseWizardStep.CONSOLIDATION_APPROACH: "GHG Consolidation Approach",
    EnterpriseWizardStep.SCOPE_CONFIGURATION: "Scope & Category Configuration",
    EnterpriseWizardStep.SBTI_PATHWAY: "SBTi Target Pathway",
    EnterpriseWizardStep.SECTOR_PRESET: "Sector-Specific Configuration",
    EnterpriseWizardStep.ASSURANCE_READINESS: "Assurance & Verification",
}

# ---------------------------------------------------------------------------
# EnterpriseSetupWizard
# ---------------------------------------------------------------------------

class EnterpriseSetupWizard:
    """8-step enterprise onboarding wizard for PACK-027.

    Example:
        >>> wizard = EnterpriseSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {...})
        >>> result = wizard.generate_config()
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state: Optional[WizardState] = None
        self.logger.info("EnterpriseSetupWizard initialized: 8 steps")

    def start(self) -> WizardState:
        wizard_id = _compute_hash(f"ent-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step in STEP_ORDER:
            steps[step.value] = WizardStepState(
                name=step,
                display_name=STEP_DISPLAY_NAMES.get(step, step.value),
            )
        self._state = WizardState(
            wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps,
        )
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = EnterpriseWizardStep(step_name)
        except ValueError:
            raise ValueError(f"Unknown step '{step_name}'")

        step = self._state.steps.get(step_name)
        if not step:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = StepStatus.IN_PROGRESS
        start_time = time.monotonic()

        try:
            errors = self._validate_step(step_enum, data)
            step.data = data
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.validation_errors = []
                self._advance_step(step_enum)

        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]

        return self._state

    def generate_config(self) -> EnterpriseSetupResult:
        if self._state is None:
            return EnterpriseSetupResult()

        completed = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        org_data = self._state.steps.get("organization_profile", WizardStepState(name=EnterpriseWizardStep.ORGANIZATION_PROFILE)).data
        entity_data = self._state.steps.get("entity_hierarchy", WizardStepState(name=EnterpriseWizardStep.ENTITY_HIERARCHY)).data
        erp_data = self._state.steps.get("erp_integration", WizardStepState(name=EnterpriseWizardStep.ERP_INTEGRATION)).data
        consol_data = self._state.steps.get("consolidation_approach", WizardStepState(name=EnterpriseWizardStep.CONSOLIDATION_APPROACH)).data
        sbti_data = self._state.steps.get("sbti_pathway", WizardStepState(name=EnterpriseWizardStep.SBTI_PATHWAY)).data
        preset_data = self._state.steps.get("sector_preset", WizardStepState(name=EnterpriseWizardStep.SECTOR_PRESET)).data
        assurance_data = self._state.steps.get("assurance_readiness", WizardStepState(name=EnterpriseWizardStep.ASSURANCE_READINESS)).data

        result = EnterpriseSetupResult(
            organization_name=org_data.get("organization_name", ""),
            sector=org_data.get("sector", "manufacturing"),
            preset=preset_data.get("preset", "manufacturing_enterprise"),
            entity_count=entity_data.get("total_entities", 50),
            erp_system=erp_data.get("primary_erp", "sap_s4hana"),
            consolidation_approach=consol_data.get("approach", "operational_control"),
            sbti_pathway=sbti_data.get("pathway", "aca_15c"),
            scope3_categories=list(range(1, 16)),
            assurance_level=assurance_data.get("level", "limited"),
            assurance_provider=assurance_data.get("provider", ""),
            carbon_price_per_tco2e=100.0,
            base_year=2023,
            target_year=2030,
            engines_enabled=[
                "enterprise_baseline_engine", "sbti_target_engine",
                "scenario_modeling_engine", "carbon_pricing_engine",
                "scope4_avoided_engine", "supply_chain_mapping_engine",
                "financial_integration_engine", "assurance_engine",
            ],
            total_steps_completed=completed,
            configuration_hash=_compute_hash(org_data),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def run_demo(self) -> EnterpriseSetupResult:
        self.start()
        demo_steps = {
            "organization_profile": {
                "organization_name": "Global Manufacturing Corp",
                "sector": "manufacturing", "headquarters_country": "US",
                "employee_count": 15000, "annual_revenue_usd": 2_000_000_000,
            },
            "entity_hierarchy": {
                "total_entities": 120, "countries": 25,
                "wholly_owned": 80, "joint_ventures": 15,
            },
            "erp_integration": {
                "primary_erp": "sap_s4hana", "hcm_system": "workday",
                "connected": True, "company_codes": ["1000", "2000", "3000"],
            },
            "consolidation_approach": {"approach": "operational_control"},
            "scope_configuration": {"scope3_categories": list(range(1, 16))},
            "sbti_pathway": {"pathway": "aca_15c", "flag_enabled": False},
            "sector_preset": {"preset": "manufacturing_enterprise"},
            "assurance_readiness": {"level": "limited", "provider": "kpmg"},
        }
        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)
        return self.generate_config()

    def get_state(self) -> Optional[WizardState]:
        return self._state

    def get_step_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": s.value,
                "display_name": STEP_DISPLAY_NAMES.get(s, ""),
                "status": (
                    self._state.steps[s.value].status.value
                    if self._state and s.value in self._state.steps
                    else "pending"
                ),
            }
            for s in STEP_ORDER
        ]

    def _validate_step(self, step: EnterpriseWizardStep, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if step == EnterpriseWizardStep.ORGANIZATION_PROFILE:
            if not data.get("organization_name"):
                errors.append("Organization name is required")
            emp = data.get("employee_count", 0)
            if emp < 250:
                errors.append("PACK-027 is designed for enterprises with 250+ employees")
        elif step == EnterpriseWizardStep.ENTITY_HIERARCHY:
            if data.get("total_entities", 0) < 1:
                errors.append("At least 1 entity is required")
        return errors

    def _advance_step(self, current: EnterpriseWizardStep) -> None:
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = utcnow()
        except ValueError:
            pass
