# -*- coding: utf-8 -*-
"""
Regulatory Compliance Workflow
====================================

3-phase workflow for multi-regulation compliance tracking within
PACK-014 CSRD Retail and Consumer Goods Pack.

Phases:
    1. RegulationMapping       -- Map applicable regulations by retail sub-sector
    2. ComplianceAssessment    -- Check compliance status per regulation
    3. ActionPlanning          -- Generate remediation actions with deadlines

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ComplianceLevel(str, Enum):
    """Compliance status level."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    NOT_APPLICABLE = "not_applicable"


class RegulationCategory(str, Enum):
    """Regulation category types."""
    CLIMATE = "climate"
    PACKAGING = "packaging"
    SUPPLY_CHAIN = "supply_chain"
    PRODUCT = "product"
    FOOD_SAFETY = "food_safety"
    CONSUMER = "consumer"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class ActionPriority(str, Enum):
    """Action item priority."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RetailSubSector(str, Enum):
    """Retail sub-sector classification."""
    FOOD_GROCERY = "food_grocery"
    FASHION_APPAREL = "fashion_apparel"
    ELECTRONICS = "electronics"
    HOME_GARDEN = "home_garden"
    HEALTH_BEAUTY = "health_beauty"
    GENERAL_MERCHANDISE = "general_merchandise"
    ECOMMERCE = "ecommerce"
    CONVENIENCE = "convenience"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class CompanyData(BaseModel):
    """Company information for regulation mapping."""
    entity_id: str = Field(default="")
    name: str = Field(default="")
    sub_sectors: List[RetailSubSector] = Field(default_factory=list)
    countries_of_operation: List[str] = Field(default_factory=list)
    employee_count: int = Field(default=0, ge=0)
    annual_revenue_eur: float = Field(default=0.0, ge=0.0)
    total_assets_eur: float = Field(default=0.0, ge=0.0)
    is_listed: bool = Field(default=False)
    is_large_undertaking: bool = Field(default=False)
    has_eu_operations: bool = Field(default=True)
    imports_from_non_eu: bool = Field(default=False)


class RegulationRecord(BaseModel):
    """Individual regulation record."""
    regulation_id: str = Field(default_factory=lambda: f"reg-{uuid.uuid4().hex[:6]}")
    name: str = Field(..., description="Regulation name")
    short_name: str = Field(default="", description="Abbreviation")
    category: RegulationCategory = Field(default=RegulationCategory.CLIMATE)
    effective_date: str = Field(default="", description="YYYY-MM-DD")
    compliance_deadline: str = Field(default="", description="YYYY-MM-DD")
    applicable_sub_sectors: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    key_requirements: List[str] = Field(default_factory=list)
    penalties: str = Field(default="")


class ComplianceAssessmentItem(BaseModel):
    """Compliance assessment for a single regulation."""
    regulation_id: str = Field(default="")
    regulation_name: str = Field(default="")
    compliance_level: ComplianceLevel = Field(default=ComplianceLevel.NOT_ASSESSED)
    score_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    requirements_met: int = Field(default=0, ge=0)
    requirements_total: int = Field(default=0, ge=0)
    gaps: List[str] = Field(default_factory=list)
    risk_level: str = Field(default="medium", description="critical|high|medium|low")
    notes: str = Field(default="")


class ActionItem(BaseModel):
    """Remediation action item."""
    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:6]}")
    regulation_id: str = Field(default="")
    regulation_name: str = Field(default="")
    priority: ActionPriority = Field(default=ActionPriority.MEDIUM)
    description: str = Field(default="")
    responsible_team: str = Field(default="")
    deadline: str = Field(default="", description="YYYY-MM-DD")
    estimated_effort_days: int = Field(default=0, ge=0)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="open", description="open|in_progress|completed")


class RegulatoryComplianceInput(BaseModel):
    """Input data model for RegulatoryComplianceWorkflow."""
    company_data: CompanyData = Field(default_factory=CompanyData)
    regulation_list: List[RegulationRecord] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RegulatoryComplianceResult(BaseModel):
    """Complete result from regulatory compliance workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="regulatory_compliance")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    compliance_matrix: List[ComplianceAssessmentItem] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    deadlines: List[Dict[str, str]] = Field(default_factory=list)
    overall_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    regulations_applicable: int = Field(default=0)
    regulations_compliant: int = Field(default=0)
    critical_actions: int = Field(default=0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# EU RETAIL REGULATION DATABASE
# =============================================================================

EU_RETAIL_REGULATIONS: List[Dict[str, Any]] = [
    {
        "name": "Corporate Sustainability Reporting Directive",
        "short_name": "CSRD",
        "category": "climate",
        "effective_date": "2024-01-01",
        "sub_sectors": ["all"],
        "key_requirements": [
            "Double materiality assessment",
            "ESRS-compliant disclosures",
            "Third-party limited assurance",
            "Digital tagging (ESEF/iXBRL)",
        ],
    },
    {
        "name": "Packaging and Packaging Waste Regulation",
        "short_name": "PPWR",
        "category": "packaging",
        "effective_date": "2025-01-01",
        "sub_sectors": ["food_grocery", "fashion_apparel", "electronics", "health_beauty", "general_merchandise"],
        "key_requirements": [
            "Recycled content targets (PET: 25% by 2025, 30% by 2030)",
            "Recyclability requirements",
            "Labeling and sorting instructions",
            "EPR fee payments with eco-modulation",
            "Reuse targets for transport packaging",
        ],
    },
    {
        "name": "Corporate Sustainability Due Diligence Directive",
        "short_name": "CSDDD",
        "category": "supply_chain",
        "effective_date": "2026-01-01",
        "sub_sectors": ["all"],
        "key_requirements": [
            "Human rights due diligence in supply chain",
            "Environmental due diligence",
            "Climate transition plan",
            "Stakeholder engagement",
            "Grievance mechanism",
        ],
    },
    {
        "name": "EU Deforestation-Free Products Regulation",
        "short_name": "EUDR",
        "category": "supply_chain",
        "effective_date": "2024-12-30",
        "sub_sectors": ["food_grocery", "fashion_apparel"],
        "key_requirements": [
            "Deforestation-free due diligence",
            "Geolocation traceability",
            "Due diligence statements",
            "Risk assessment for sourcing countries",
        ],
    },
    {
        "name": "Ecodesign for Sustainable Products Regulation",
        "short_name": "ESPR",
        "category": "product",
        "effective_date": "2025-07-01",
        "sub_sectors": ["fashion_apparel", "electronics", "home_garden"],
        "key_requirements": [
            "Digital Product Passport (DPP)",
            "Product durability requirements",
            "Repairability requirements",
            "Recycled content targets",
            "Ban on destruction of unsold goods",
        ],
    },
    {
        "name": "EU Green Claims Directive",
        "short_name": "ECGT",
        "category": "consumer",
        "effective_date": "2026-06-01",
        "sub_sectors": ["all"],
        "key_requirements": [
            "Substantiation of environmental claims",
            "Third-party verification",
            "Ban on generic green claims",
            "Prohibition of carbon-neutral claims via offsets",
        ],
    },
    {
        "name": "EU Strategy for Sustainable and Circular Textiles",
        "short_name": "EU Textiles Strategy",
        "category": "product",
        "effective_date": "2025-01-01",
        "sub_sectors": ["fashion_apparel"],
        "key_requirements": [
            "Textile waste EPR",
            "Microplastic release standards",
            "Greenwashing prevention",
            "Mandatory recycled fibre content",
        ],
    },
    {
        "name": "EU Taxonomy Regulation",
        "short_name": "EU Taxonomy",
        "category": "climate",
        "effective_date": "2022-01-01",
        "sub_sectors": ["all"],
        "key_requirements": [
            "Taxonomy-aligned revenue disclosure",
            "Taxonomy-aligned CapEx disclosure",
            "Taxonomy-aligned OpEx disclosure",
            "DNSH criteria assessment",
        ],
    },
]


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatoryComplianceWorkflow:
    """
    3-phase multi-regulation compliance workflow.

    Maps applicable regulations by retail sub-sector, assesses
    compliance status, and generates action plans with deadlines.

    Example:
        >>> wf = RegulatoryComplianceWorkflow()
        >>> inp = RegulatoryComplianceInput(company_data=CompanyData(...))
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RegulatoryComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._applicable_regulations: List[RegulationRecord] = []
        self._assessments: List[ComplianceAssessmentItem] = []
        self._actions: List[ActionItem] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[RegulatoryComplianceInput] = None,
        company_data: Optional[CompanyData] = None,
        regulation_list: Optional[List[RegulationRecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RegulatoryComplianceResult:
        """Execute the 3-phase regulatory compliance workflow."""
        if input_data is None:
            input_data = RegulatoryComplianceInput(
                company_data=company_data or CompanyData(),
                regulation_list=regulation_list or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting regulatory compliance workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_regulation_mapping(input_data))
            phase_results.append(await self._phase_compliance_assessment(input_data))
            phase_results.append(await self._phase_action_planning(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Regulatory compliance workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        compliant_count = sum(1 for a in self._assessments if a.compliance_level == ComplianceLevel.COMPLIANT)
        total_applicable = len(self._assessments)
        overall_pct = (compliant_count / max(total_applicable, 1)) * 100
        critical_count = sum(1 for a in self._actions if a.priority == ActionPriority.CRITICAL)

        deadlines = []
        for reg in self._applicable_regulations:
            if reg.compliance_deadline:
                deadlines.append({
                    "regulation": reg.short_name or reg.name,
                    "deadline": reg.compliance_deadline,
                })
        deadlines.sort(key=lambda d: d["deadline"])

        result = RegulatoryComplianceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            compliance_matrix=self._assessments,
            action_items=self._actions,
            deadlines=deadlines,
            overall_compliance_pct=round(overall_pct, 2),
            regulations_applicable=total_applicable,
            regulations_compliant=compliant_count,
            critical_actions=critical_count,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Regulation Mapping
    # -------------------------------------------------------------------------

    async def _phase_regulation_mapping(self, input_data: RegulatoryComplianceInput) -> PhaseResult:
        """Map applicable regulations by retail sub-sector."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        company_sectors = [s.value for s in input_data.company_data.sub_sectors]
        if not company_sectors:
            company_sectors = ["general_merchandise"]
            warnings.append("No sub-sectors specified; defaulting to general_merchandise")

        # Use provided regulations or default EU regulations
        if input_data.regulation_list:
            self._applicable_regulations = input_data.regulation_list
        else:
            self._applicable_regulations = []
            for reg_data in EU_RETAIL_REGULATIONS:
                applicable_sectors = reg_data.get("sub_sectors", [])
                if "all" in applicable_sectors or any(s in applicable_sectors for s in company_sectors):
                    self._applicable_regulations.append(RegulationRecord(
                        name=reg_data["name"],
                        short_name=reg_data.get("short_name", ""),
                        category=RegulationCategory(reg_data["category"]),
                        effective_date=reg_data.get("effective_date", ""),
                        applicable_sub_sectors=applicable_sectors,
                        key_requirements=reg_data.get("key_requirements", []),
                    ))

        by_category: Dict[str, int] = {}
        for reg in self._applicable_regulations:
            by_category[reg.category.value] = by_category.get(reg.category.value, 0) + 1

        outputs["total_applicable"] = len(self._applicable_regulations)
        outputs["by_category"] = by_category
        outputs["company_sectors"] = company_sectors

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 RegulationMapping: %d regulations applicable", len(self._applicable_regulations))
        return PhaseResult(
            phase_name="regulation_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Compliance Assessment
    # -------------------------------------------------------------------------

    async def _phase_compliance_assessment(self, input_data: RegulatoryComplianceInput) -> PhaseResult:
        """Assess compliance status per regulation."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._assessments = []

        for reg in self._applicable_regulations:
            total_reqs = len(reg.key_requirements)
            # Simulate assessment: in production, this would query actual compliance data
            met_count = 0
            gaps: List[str] = []

            for req in reg.key_requirements:
                # Default: assume partial compliance for demonstration
                met_count += 0  # Start with zero; real implementation checks actual state
                gaps.append(req)

            score = (met_count / max(total_reqs, 1)) * 100
            if score >= 90:
                level = ComplianceLevel.COMPLIANT
                risk = "low"
            elif score >= 60:
                level = ComplianceLevel.PARTIALLY_COMPLIANT
                risk = "medium"
            else:
                level = ComplianceLevel.NOT_ASSESSED
                risk = "high"

            self._assessments.append(ComplianceAssessmentItem(
                regulation_id=reg.regulation_id,
                regulation_name=reg.short_name or reg.name,
                compliance_level=level,
                score_pct=round(score, 2),
                requirements_met=met_count,
                requirements_total=total_reqs,
                gaps=gaps,
                risk_level=risk,
            ))

        outputs["total_assessed"] = len(self._assessments)
        outputs["compliant"] = sum(1 for a in self._assessments if a.compliance_level == ComplianceLevel.COMPLIANT)
        outputs["partially_compliant"] = sum(1 for a in self._assessments if a.compliance_level == ComplianceLevel.PARTIALLY_COMPLIANT)
        outputs["non_compliant"] = sum(1 for a in self._assessments if a.compliance_level == ComplianceLevel.NON_COMPLIANT)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 ComplianceAssessment: %d assessed", len(self._assessments))
        return PhaseResult(
            phase_name="compliance_assessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_action_planning(self, input_data: RegulatoryComplianceInput) -> PhaseResult:
        """Generate remediation actions with deadlines."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._actions = []

        for assessment in self._assessments:
            if assessment.compliance_level == ComplianceLevel.COMPLIANT:
                continue

            reg = next(
                (r for r in self._applicable_regulations if r.regulation_id == assessment.regulation_id),
                None,
            )
            if not reg:
                continue

            for gap in assessment.gaps:
                if assessment.risk_level == "high" or assessment.risk_level == "critical":
                    priority = ActionPriority.CRITICAL
                elif assessment.risk_level == "medium":
                    priority = ActionPriority.HIGH
                else:
                    priority = ActionPriority.MEDIUM

                deadline = reg.compliance_deadline or (datetime.utcnow() + timedelta(days=180)).strftime("%Y-%m-%d")

                self._actions.append(ActionItem(
                    regulation_id=assessment.regulation_id,
                    regulation_name=assessment.regulation_name,
                    priority=priority,
                    description=f"Address gap: {gap}",
                    responsible_team=self._map_responsible_team(reg.category),
                    deadline=deadline,
                    estimated_effort_days=self._estimate_effort(gap),
                    estimated_cost_eur=self._estimate_cost(gap),
                ))

        outputs["total_actions"] = len(self._actions)
        outputs["critical_actions"] = sum(1 for a in self._actions if a.priority == ActionPriority.CRITICAL)
        outputs["high_actions"] = sum(1 for a in self._actions if a.priority == ActionPriority.HIGH)
        outputs["total_estimated_cost_eur"] = round(sum(a.estimated_cost_eur for a in self._actions), 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ActionPlanning: %d actions generated", len(self._actions))
        return PhaseResult(
            phase_name="action_planning", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _map_responsible_team(self, category: RegulationCategory) -> str:
        """Map regulation category to responsible team."""
        mapping: Dict[str, str] = {
            "climate": "Sustainability",
            "packaging": "Packaging & Supply Chain",
            "supply_chain": "Procurement & Compliance",
            "product": "Product Development",
            "food_safety": "Quality Assurance",
            "consumer": "Marketing & Legal",
            "social": "Human Resources",
            "governance": "Legal & Compliance",
        }
        return mapping.get(category.value, "Compliance")

    def _estimate_effort(self, gap: str) -> int:
        """Estimate remediation effort in days."""
        gap_lower = gap.lower()
        if "due diligence" in gap_lower or "transition plan" in gap_lower:
            return 60
        elif "assessment" in gap_lower or "audit" in gap_lower:
            return 30
        elif "disclosure" in gap_lower or "reporting" in gap_lower:
            return 20
        elif "labeling" in gap_lower or "tagging" in gap_lower:
            return 15
        return 20

    def _estimate_cost(self, gap: str) -> float:
        """Estimate remediation cost in EUR."""
        gap_lower = gap.lower()
        if "due diligence" in gap_lower:
            return 50000.0
        elif "assurance" in gap_lower or "verification" in gap_lower:
            return 30000.0
        elif "system" in gap_lower or "digital" in gap_lower:
            return 40000.0
        elif "training" in gap_lower or "engagement" in gap_lower:
            return 10000.0
        return 15000.0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RegulatoryComplianceResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
